/* Copyright 2023 Architect Financial Technologies LLC. This is free
 * software released under the MIT license */

use anyhow::{bail, Result};
use bytes::Buf;
use chrono::{DateTime, NaiveDate, Utc};
use compact_str::CompactString;
use futures::future;
use fxhash::FxHashMap;
use immutable_chunkmap::set::SetM as Set;
use log::{error, warn};
use netidx_core::pack::Pack;
use netidx_derive::Pack;
use serde::{Deserialize, Serialize};
use sled::IVec;
use smallvec::SmallVec;
use std::{
    any::Any,
    cmp::{self, Ordering},
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    io::SeekFrom,
    iter,
    marker::PhantomData,
    path::{Path, PathBuf},
    result,
    time::UNIX_EPOCH,
};
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncBufReadExt, AsyncSeekExt, AsyncWriteExt, BufStream},
    task,
};

#[cfg(test)]
mod test;

#[derive(Debug)]
pub enum Query {
    Eq(Box<dyn IndexableField>),
    Gt(Box<dyn IndexableField>),
    Gte(Box<dyn IndexableField>),
    Lt(Box<dyn IndexableField>),
    Lte(Box<dyn IndexableField>),
    And(Vec<Query>),
    Or(Vec<Query>),
    Not(Box<Query>),
}

/// An indexable field. Indexable fields must be byte equal, meaning
/// the decoded representation will be equal if the encoded bytes are
/// equal. They may or may not be byte comparable.
pub trait IndexableField: Debug + Send + Any {
    /// return the database key that should be used for this field
    fn key(&self) -> &'static str;

    /// return true if these objects can be compared bytewise
    fn byte_compareable(&self) -> bool;

    /// Place the database representation of the value into the
    /// specified buffer
    fn encode(&self, buf: &mut SmallVec<[u8; 128]>) -> Result<()>;

    /// Compare the value with the specified encoded value. This will
    /// only be called if byte_comparable is false. The default
    /// implementation just does byte comparison.
    fn decode_cmp(&self, b: &[u8]) -> Result<Ordering> {
        let mut buf: SmallVec<[u8; 128]> = SmallVec::new();
        self.encode(&mut buf)?;
        Ok(b.cmp(&buf[..]))
    }

    /// return a downcastable object to recover the original type of
    /// the indexable value in a query. This is a work around until
    /// trait upcasting is stabilized.
    fn as_any(&self) -> &dyn Any;
}

/// An indexable type
pub trait Indexable {
    type Iter: IntoIterator<Item = Box<dyn IndexableField>>;

    /// Return an iterator of indexable values in this record.
    fn index(&self) -> Self::Iter;

    /// return the timestamp of this record
    fn timestamp(&self) -> DateTime<Utc>;
}

fn min_key_with_prefix(index: &sled::Tree, k: &[u8]) -> Result<Option<IVec>> {
    let mut iter = index.scan_prefix(k);
    Ok(iter.next().transpose()?.map(|(k, _)| k))
}

fn max_key_with_prefix(index: &sled::Tree, k: &[u8]) -> Result<Option<IVec>> {
    let mut iter = index.scan_prefix(k);
    Ok(iter.next_back().transpose()?.map(|(k, _)| k))
}

#[derive(Debug, Clone, Copy, Pack, PartialEq, Eq, PartialOrd, Ord)]
#[pack(unwrapped)]
pub struct IndexEntry {
    pub file: NaiveDate,
    pub offset: u64,
}

struct JsonFile<T: Indexable + Serialize + for<'a> Deserialize<'a>> {
    phantom: PhantomData<T>,
    last_used: DateTime<Utc>,
    file: Option<BufStream<File>>,
    path: PathBuf,
    name: NaiveDate,
    len: u64,
    pos: u64,
    rbuf: String,
    wbuf: Vec<u8>,
}

macro_rules! open_file {
    ($self:expr) => {
        if $self.file.is_some() {
            $self.file.as_mut().unwrap()
        } else {
            let f = OpenOptions::new()
                .read(true)
                .append(true)
                .write(true)
                .create(true)
                .open(&$self.path)
                .await?;
            let mut f = BufStream::new(f);
            let len = f.seek(SeekFrom::End(0)).await?;
            $self.len = len;
            $self.pos = len;
            $self.file = Some(f);
            $self.file.as_mut().unwrap()
        }
    };
}

impl<T: Indexable + Serialize + for<'a> Deserialize<'a>> JsonFile<T> {
    fn new(base: impl AsRef<Path>, name: NaiveDate) -> Self {
        Self {
            phantom: PhantomData,
            last_used: DateTime::<Utc>::MIN_UTC,
            file: None,
            path: base.as_ref().join(&format!("{name}")),
            name,
            pos: 0,
            len: 0,
            rbuf: String::new(),
            wbuf: Vec::new(),
        }
    }

    async fn mtime(&self) -> Result<u128> {
        Ok(fs::metadata(&self.path)
            .await?
            .modified()?
            .duration_since(UNIX_EPOCH)?
            .as_nanos())
    }

    fn close_if_idle(&mut self, now: DateTime<Utc>) {
        if now - self.last_used > chrono::Duration::minutes(5) {
            self.file = None
        }
    }

    async fn get(&mut self, pos: u64) -> Result<Option<(u64, T)>> {
        self.last_used = Utc::now();
        let file = open_file!(self);
        if pos != self.pos {
            let new_pos = file.seek(SeekFrom::Start(pos)).await?;
            if new_pos != pos {
                bail!("{pos} doesn't exist in {:?}", &self.name)
            }
            self.pos = new_pos;
        }
        self.rbuf.clear();
        let read = file.read_line(&mut self.rbuf).await?;
        if self.rbuf.len() == 0 {
            Ok(None)
        } else {
            self.pos = self.pos + read as u64;
            Ok(Some((self.pos, serde_json::from_str(&self.rbuf.trim())?)))
        }
    }

    async fn flush(&mut self) -> Result<()> {
        if let Some(file) = &mut self.file {
            file.flush().await?
        }
        Ok(())
    }

    fn db_name(&self) -> CompactString {
        use std::fmt::Write;
        let mut buf = CompactString::new("");
        write!(buf, "{}", self.name).unwrap();
        buf
    }

    async fn update_mtime(&mut self, db: &sled::Tree) -> Result<()> {
        let mtime = self.mtime().await?;
        db.insert(self.db_name().as_bytes(), &u128::to_be_bytes(mtime))?;
        Ok(())
    }

    /// write record to the file, return the position of the beginning
    /// of the newly written record.
    async fn append(&mut self, record: &T) -> Result<u64> {
        self.last_used = Utc::now();
        let file = open_file!(self);
        let pos = self.pos;
        if self.pos != self.len {
            file.flush().await?;
            self.pos = file.seek(SeekFrom::End(0)).await?;
            self.len = self.pos;
        }
        self.wbuf.clear();
        serde_json::to_writer(&mut self.wbuf, record)?;
        self.wbuf.push(b'\n');
        file.write_all(&self.wbuf).await?;
        self.pos += self.wbuf.len() as u64;
        self.len += self.wbuf.len() as u64;
        Ok(pos)
    }
}

/// Simple line structured json archives with database like indexing
///
/// Maintain a set of newline delimited json formatted text files and
/// a database indexing various fields in them for efficient query
/// execution.
///
/// records are written to date named files corresponding to when they
/// begin. e.g. `2023-09-12`. New records are indexed automatically
/// after they are written.
pub struct IndexedJson<T: Indexable + Serialize + for<'a> Deserialize<'a>> {
    phantom: PhantomData<T>,
    base: PathBuf,
    db: sled::Db,
    index_status: sled::Tree,
    trees: FxHashMap<CompactString, sled::Tree>,
    files: BTreeMap<NaiveDate, JsonFile<T>>,
    gc: DateTime<Utc>,
}

impl<T: Indexable + Serialize + for<'a> Deserialize<'a>> IndexedJson<T> {
    /// given the path to a directory where date named json files are
    /// stored, create a new IndexedJson archive. If the index is
    /// missing or outdated then it will be rebuilt.
    pub async fn new(base: impl AsRef<Path>) -> Result<Self> {
        if !fs::metadata(&base).await?.is_dir() {
            bail!("{:?} is not a directory", base.as_ref())
        }
        let db = task::spawn_blocking({
            let path = base.as_ref().join("db");
            move || {
                sled::Config::new()
                    .mode(sled::Mode::LowSpace)
                    .flush_every_ms(None)
                    .path(path)
                    .open()
            }
        })
        .await??;
        let index_status = task::spawn_blocking({
            let db = db.clone();
            move || db.open_tree("status")
        })
        .await??;
        let files = BTreeMap::new();
        let mut t = Self {
            phantom: PhantomData,
            base: PathBuf::from(base.as_ref()),
            db,
            index_status,
            trees: HashMap::default(),
            files,
            gc: Utc::now(),
        };
        t.maybe_reindex().await?;
        Ok(t)
    }

    // note this will close all open files as well as add any new
    // files that have appeared on disk to the database.
    async fn rescan_files(&mut self) -> Result<()> {
        self.trees.clear();
        for name in self.db.tree_names() {
            if name.starts_with(b"index_") {
                let tree = self.db.open_tree(&name)?;
                self.trees
                    .insert(CompactString::from_utf8_lossy(&*name), tree);
            }
        }
        self.files.clear();
        let mut dir = fs::read_dir(&self.base).await?;
        loop {
            match dir.next_entry().await? {
                None => break,
                Some(e) => {
                    if e.file_type().await?.is_file() {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        if let Ok(d) = name.parse::<NaiveDate>() {
                            self.files.insert(d, JsonFile::new(&self.base, d));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn needs_reindex(&mut self) -> Result<bool> {
        self.rescan_files().await?;
        for file in self.files.values() {
            match self.index_status.get(file.db_name().as_bytes())? {
                None => return Ok(true),
                Some(dbmtime) => {
                    let dbmtime = {
                        if dbmtime.len() != 16 {
                            return Ok(true);
                        }
                        (&mut &*dbmtime).get_u128()
                    };
                    let fsmtime = file.mtime().await?;
                    if fsmtime != dbmtime {
                        return Ok(true);
                    }
                }
            }
        }
        for r in self.index_status.iter() {
            let (file, _) = r?;
            let file = match String::from_utf8_lossy(&*file).parse::<NaiveDate>() {
                Err(_) => return Ok(true),
                Ok(file) => file,
            };
            if !self.files.contains_key(&file) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Rebuild the index only if necessary. This is called
    /// automatically by `new`. However if you know, or suspect, the
    /// files have been modified since then, you can call again and it
    /// will check and rebuild the index if they have. Before checking
    /// it will also rescan the filesystem. As a result, all open
    /// files will be closed. Any new files will be added to the
    /// index, and missing files will be removed from it. If any new
    /// files appear or old files disappear the index will be rebuilt.
    ///
    /// If the index is damaged you can call `reindex`, which will
    /// force it to rebuild.
    pub async fn maybe_reindex(&mut self) -> Result<()> {
        if self.needs_reindex().await? {
            self.reindex().await?
        }
        Ok(())
    }

    /// force rebuild the index
    pub async fn reindex(&mut self) -> Result<()> {
        for (_, tree) in self.trees.drain() {
            tree.clear()?
        }
        self.index_status.clear()?;
        self.rescan_files().await?;
        for (file, f) in self.files.iter_mut() {
            let mut pos = 0;
            loop {
                match f.get(pos).await {
                    Ok(Some((next_pos, t))) => {
                        let entry = IndexEntry {
                            file: *file,
                            offset: pos,
                        };
                        Self::index_record(&self.db, &mut self.trees, entry, &t)?;
                        pos = next_pos;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        error!("error reindexing file {file} pos {pos} error {:?}", e);
                        break;
                    }
                }
            }
            f.update_mtime(&self.index_status).await?
        }
        Ok(())
    }

    // index format in btree {value}/{i} => {IndexEntry}
    // where i is the number of times the same value
    // appears in the index. Each key is stored in a separate
    // sled tree according to it's name
    fn index_record<'a>(
        db: &sled::Db,
        trees: &mut FxHashMap<CompactString, sled::Tree>,
        pos: IndexEntry,
        record: &'a T,
    ) -> Result<()> {
        let mut kbuf: SmallVec<[u8; 128]> = SmallVec::new();
        let mut vbuf: SmallVec<[u8; 16]> = SmallVec::new();
        vbuf.resize(pos.encoded_len(), 0u8);
        pos.encode(&mut &mut *vbuf)?;
        for field in record.index() {
            let tree = match trees.get(field.key()) {
                Some(t) => t,
                None => {
                    let tree = db.open_tree(format!("index_{}", field.key()))?;
                    trees
                        .entry(CompactString::from(field.key()))
                        .or_insert(tree)
                }
            };
            kbuf.clear();
            field.encode(&mut kbuf)?;
            kbuf.push(b'/');
            let count = match max_key_with_prefix(tree, &kbuf)? {
                None => 0,
                Some(k) if k.len() < 8 => 0,
                Some(k) => (&mut &k[k.len() - 8..]).get_u64() + 1,
            };
            kbuf.extend_from_slice(&u64::to_be_bytes(count));
            tree.insert(&kbuf[..], &vbuf[..])?;
        }
        Ok::<_, anyhow::Error>(())
    }

    fn maybe_gc(&mut self) {
        let now = Utc::now();
        if now - self.gc > chrono::Duration::minutes(5) {
            self.gc = now;
            for f in self.files.values_mut() {
                f.close_if_idle(now)
            }
        }
    }

    /// flush the files and the index database
    pub async fn flush(&mut self) -> Result<()> {
        future::join_all(self.files.values_mut().map(|f| f.flush()))
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        future::join_all(
            (iter::once(&*self.db)
                .chain(iter::once(&self.index_status).chain(self.trees.values())))
            .map(|t| t.flush_async()),
        )
        .await
        .into_iter()
        .collect::<result::Result<Vec<_>, _>>()?;
        Ok(())
    }

    /// append the record to the end of the file corresponding to it's
    /// timestamp and then index it.
    pub async fn append(&mut self, record: &T) -> Result<()> {
        let name = record.timestamp().date_naive();
        let file = self
            .files
            .entry(name)
            .or_insert_with(|| JsonFile::new(&self.base, name));
        let pos = file.append(record).await?;
        Self::index_record(
            &self.db,
            &mut self.trees,
            IndexEntry {
                file: name,
                offset: pos,
            },
            record,
        )?;
        file.update_mtime(&self.index_status).await?;
        Ok(self.maybe_gc())
    }

    /// return the index of the first record in the first file, or
    /// none if there are no records
    pub fn first(&self) -> Option<IndexEntry> {
        self.files.first_key_value().map(|(k, _)| IndexEntry {
            file: *k,
            offset: 0,
        })
    }

    /// retreive the specified record from the json files. Returns a
    /// pair of the record and the next entry index if there is
    /// one. If None is returned then there were no more entries in
    /// the archive.
    pub async fn get(&mut self, mut entry: IndexEntry) -> Result<Option<(IndexEntry, T)>> {
        loop {
            let file = self
                .files
                .entry(entry.file)
                .or_insert_with(|| JsonFile::new(&self.base, entry.file));
            match file.get(entry.offset).await? {
                Some((offset, t)) => break Ok(Some((IndexEntry { offset, ..entry }, t))),
                None => {
                    let mut r = self.files.range(entry.file..);
                    r.next(); // will be the current entry
                    match r.next() {
                        Some((e, _)) => {
                            entry = IndexEntry {
                                file: *e,
                                offset: 0,
                            };
                        }
                        None => break Ok(None),
                    }
                }
            }
        }
    }

    /// execute the specified query against the index and return the
    /// set of matching entries.
    pub fn query(&self, query: &Query) -> Result<Set<IndexEntry>> {
        fn field_k(field: &Box<dyn IndexableField>) -> Result<SmallVec<[u8; 128]>> {
            let mut buf = SmallVec::new();
            field.encode(&mut buf)?;
            buf.push(b'/');
            Ok(buf)
        }
        fn insert(set: &mut Set<IndexEntry>, k: &[u8], mut v: &[u8]) {
            match IndexEntry::decode(&mut v) {
                Ok(ent) => {
                    set.insert_cow(ent);
                }
                Err(e) => {
                    warn!("could not decode entry with key {:?}, {:?}", k, e)
                }
            }
        }
        fn gt_gte(
            trees: &FxHashMap<CompactString, sled::Tree>,
            field: &Box<dyn IndexableField>,
            gte: bool,
        ) -> Result<Set<IndexEntry>> {
            match trees.get(field.key()) {
                None => Ok(Set::new()),
                Some(tree) => {
                    let mut set = Set::new();
                    let key = field_k(field)?;
                    if field.byte_compareable() {
                        let min = match min_key_with_prefix(tree, &key[..])? {
                            Some(k) => k,
                            None => match tree.first()? {
                                Some((k, _)) => k,
                                None => return Ok(set),
                            },
                        };
                        for r in tree.range(&min[..]..) {
                            let (k, v) = r?;
                            let k = &k[0..cmp::min(key.len(), k.len() - 8)];
                            if (gte && k >= &key[..]) || k > &key[..] {
                                insert(&mut set, &*k, &*v)
                            }
                        }
                    } else {
                        for r in tree.iter() {
                            let (k, v) = r?;
                            match field.decode_cmp(&k[..cmp::min(key.len(), k.len() - 8) - 1])? {
                                Ordering::Greater => insert(&mut set, &*k, &*v),
                                Ordering::Equal if gte => insert(&mut set, &*k, &*v),
                                Ordering::Equal | Ordering::Less => (),
                            }
                        }
                    }
                    Ok(set)
                }
            }
        }
        fn lt_lte(
            trees: &FxHashMap<CompactString, sled::Tree>,
            field: &Box<dyn IndexableField>,
            lte: bool,
        ) -> Result<Set<IndexEntry>> {
            match trees.get(field.key()) {
                None => Ok(Set::new()),
                Some(tree) => {
                    let key = field_k(field)?;
                    let mut set = Set::new();
                    if field.byte_compareable() {
                        let max = match max_key_with_prefix(tree, &key)? {
                            Some(k) => k,
                            None => match tree.last()? {
                                Some((k, _)) => k,
                                None => return Ok(set),
                            },
                        };
                        for r in tree.range(..=&max[..]) {
                            let (k, v) = r?;
                            let k = &k[0..cmp::min(key.len(), k.len() - 8)];
                            if (lte && k <= &key[..]) || k < &key[..] {
                                insert(&mut set, &*k, &*v)
                            }
                        }
                    } else {
                        for r in tree.iter() {
                            let (k, v) = r?;
                            match field.decode_cmp(&k[..cmp::min(key.len(), k.len() - 8) - 1])? {
                                Ordering::Less => insert(&mut set, &*k, &*v),
                                Ordering::Equal if lte => insert(&mut set, &*k, &*v),
                                Ordering::Equal | Ordering::Greater => (),
                            }
                        }
                    }
                    Ok(set)
                }
            }
        }
        fn eq(
            trees: &FxHashMap<CompactString, sled::Tree>,
            field: &Box<dyn IndexableField>,
        ) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            match trees.get(field.key()) {
                None => Ok(Set::new()),
                Some(tree) => {
                    for r in tree.scan_prefix(&field_k(field)?) {
                        let (k, v) = r?;
                        insert(&mut set, &*k, &*v)
                    }
                    Ok(set)
                }
            }
        }
        fn all_for_key(
            trees: &FxHashMap<CompactString, sled::Tree>,
            field: &Box<dyn IndexableField>,
        ) -> Result<Set<IndexEntry>> {
            match trees.get(field.key()) {
                None => Ok(Set::new()),
                Some(tree) => {
                    let mut set = Set::new();
                    for r in tree.iter() {
                        let (k, v) = r?;
                        insert(&mut set, &*k, &*v)
                    }
                    Ok(set)
                }
            }
        }
        fn all(trees: &FxHashMap<CompactString, sled::Tree>) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            for tree in trees.values() {
                for r in tree.iter() {
                    let (k, v) = r?;
                    insert(&mut set, &*k, &*v)
                }
            }
            Ok(set)
        }
        match query {
            Query::Eq(field) => eq(&self.trees, field),
            Query::Gt(field) => gt_gte(&self.trees, field, false),
            Query::Gte(field) => gt_gte(&self.trees, field, true),
            Query::Lt(field) => lt_lte(&self.trees, field, false),
            Query::Lte(field) => lt_lte(&self.trees, field, true),
            Query::And(qs) => Ok(qs
                .iter()
                .map(|q| self.query(q))
                .collect::<Result<Vec<Set<_>>>>()?
                .into_iter()
                .fold(None, |acc: Option<Set<_>>, s| match acc {
                    Some(acc) => Some(acc.intersect(&s)),
                    None => Some(s),
                })
                .unwrap_or_else(Set::new)),
            Query::Or(qs) => Ok(qs
                .iter()
                .map(|q| self.query(q))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .fold(Set::new(), |acc, s| acc.union(&s))),
            Query::Not(q) => match &**q {
                Query::Eq(field) => {
                    let matches = eq(&self.trees, field)?;
                    let all = all_for_key(&self.trees, field)?;
                    Ok(all.diff(&matches))
                }
                Query::Gt(field) => lt_lte(&self.trees, field, true),
                Query::Gte(field) => lt_lte(&self.trees, field, false),
                Query::Lt(field) => gt_gte(&self.trees, field, true),
                Query::Lte(field) => gt_gte(&self.trees, field, false),
                q @ Query::And(_) | q @ Query::Or(_) | q @ Query::Not(_) => {
                    let matches = self.query(q)?;
                    Ok(all(&self.trees)?.diff(&matches))
                }
            },
        }
    }
}
