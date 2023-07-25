/* Copyright 2023 Architect Financial Technologies LLC. This is free
 * software released under the MIT license */

use anyhow::{bail, Result};
use bytes::Buf;
use chrono::{DateTime, NaiveDate, Utc};
use compact_str::CompactString;
use immutable_chunkmap::set::SetM as Set;
use log::{error, warn};
use netidx_core::pack::Pack;
use netidx_derive::Pack;
use serde::{Deserialize, Serialize};
use sled::IVec;
use smallvec::SmallVec;
use std::{
    cmp::{max, Ordering},
    collections::BTreeMap,
    fmt::Debug,
    io::SeekFrom,
    marker::PhantomData,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
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
pub trait IndexableField: Debug + Send + 'static {
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
}

/// An indexable type
pub trait Indexable {
    type Iter: IntoIterator<Item = Box<dyn IndexableField>>;

    /// Return an iterator of indexable values in this record.
    fn index(&self) -> Self::Iter;

    /// return the timestamp of this record
    fn timestamp(&self) -> DateTime<Utc>;
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
    file: Option<File>,
    path: PathBuf,
    name: NaiveDate,
    buf: Vec<u8>,
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
            buf: vec![],
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

    // buf will contain the line on success
    async fn read_line(buf: &mut Vec<u8>, file: &mut File) -> Result<()> {
        let mut pos = 0;
        loop {
            if pos >= buf.len() {
                buf.resize(max(1024, pos << 1), 0u8);
            }
            let count = file.read(&mut buf[pos..]).await?;
            if count == 0 {
                buf.resize(0, 0);
                return Ok(());
            }
            for i in pos..pos + count {
                if buf[i] == b'\n' {
                    buf.resize(pos + i + 1, 0);
                    return Ok(());
                }
            }
            pos += count;
        }
    }

    async fn get(&mut self, pos: u64) -> Result<Option<(u64, T)>> {
        self.last_used = Utc::now();
        let file = open_file!(self);
        let new_pos = file.seek(SeekFrom::Start(pos)).await?;
        if new_pos != pos {
            bail!("{pos} doesn't exist in {:?}", &self.name)
        }
        Self::read_line(&mut self.buf, file).await?;
        if self.buf.len() == 0 {
            Ok(None)
        } else {
            let new_pos = pos + self.buf.len() as u64;
            Ok(Some((new_pos, serde_json::from_slice(&self.buf[..])?)))
        }
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
        let pos = file.seek(SeekFrom::End(0)).await?;
        self.buf.clear();
        serde_json::to_writer(&mut self.buf, record)?;
        self.buf.push(b'\n');
        file.write_all(&self.buf).await?;
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
    index: sled::Db,
    index_status: sled::Tree,
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
        let index = task::spawn_blocking({
            let path = base.as_ref().join("db");
            move || sled::open(path)
        })
        .await??;
        let index_status = task::spawn_blocking({
            let index = index.clone();
            move || index.open_tree("index_status")
        })
        .await??;
        let files = BTreeMap::new();
        let mut t = Self {
            phantom: PhantomData,
            base: PathBuf::from(base.as_ref()),
            index,
            index_status,
            files,
            gc: Utc::now(),
        };
        t.maybe_reindex().await?;
        Ok(t)
    }

    // note this will close all open files as well as add any new
    // files that have appeared on disk to the database.
    async fn rescan_files(&mut self) -> Result<()> {
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
        self.index.clear()?;
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
                        Self::index_record(&*self.index, entry, &t)?;
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

    // index format in btree     {key}/{value}/{i}     => {IndexEntry}
    // every index has count key {key}/{value}/"count" => {count}
    fn index_record<'a>(index: &sled::Tree, pos: IndexEntry, record: &'a T) -> Result<()> {
        let mut kbuf: SmallVec<[u8; 128]> = SmallVec::new();
        let mut vbuf: SmallVec<[u8; 16]> = SmallVec::new();
        vbuf.resize(pos.encoded_len(), 0u8);
        pos.encode(&mut &mut *vbuf)?;
        for field in record.index() {
            kbuf.clear();
            kbuf.extend_from_slice(field.key().as_bytes());
            kbuf.push(b'/');
            field.encode(&mut kbuf)?;
            kbuf.push(b'/');
            kbuf.extend_from_slice(b"count");
            let i = match index.get(&*kbuf)? {
                Some(ibuf) if ibuf.len() == 4 => {
                    let i = (&mut &*ibuf).get_u32();
                    index.insert(&*kbuf, &u32::to_be_bytes(i + 1))?;
                    i
                }
                Some(_) | None => {
                    index.insert(&*kbuf, &u32::to_be_bytes(1))?;
                    0
                }
            };
            kbuf.truncate(kbuf.len() - 5);
            kbuf.extend_from_slice(&u32::to_be_bytes(i));
            index.insert(&kbuf[..], &vbuf[..])?;
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
            &self.index,
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
        fn field_kv(field: &Box<dyn IndexableField>) -> Result<SmallVec<[u8; 128]>> {
            let mut buf = SmallVec::new();
            buf.extend_from_slice(field.key().as_bytes());
            buf.push(b'/');
            field.encode(&mut buf)?;
            buf.push(b'/');
            Ok(buf)
        }
        fn field_k(field: &Box<dyn IndexableField>) -> SmallVec<[u8; 128]> {
            let mut buf = SmallVec::new();
            buf.extend_from_slice(field.key().as_bytes());
            buf.push(b'/');
            buf
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
        fn min_key_with_prefix(index: &sled::Tree, k: &[u8]) -> Result<Option<IVec>> {
            let mut iter = index.scan_prefix(k);
            Ok(iter.next().transpose()?.map(|(k, _)| k))
        }
        fn max_key_with_prefix(index: &sled::Tree, k: &[u8]) -> Result<Option<IVec>> {
            let mut iter = index.scan_prefix(k);
            Ok(iter.next_back().transpose()?.map(|(k, _)| k))
        }
        fn gt_gte(
            index: &sled::Tree,
            field: &Box<dyn IndexableField>,
            gte: bool,
        ) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            if field.byte_compareable() {
                let kv = field_kv(field)?;
                let key = field_k(field);
                let min = match min_key_with_prefix(index, &kv)? {
                    Some(k) => k,
                    None => match min_key_with_prefix(index, &key)? {
                        Some(k) => k,
                        None => return Ok(set),
                    },
                };
                let max = match max_key_with_prefix(index, &key)? {
                    Some(k) => k,
                    None => return Ok(set),
                };
                for r in index.range(&min[..]..=&max[..]) {
                    let (k, v) = r?;
                    let k = &k[0..kv.len()];
                    if !k.ends_with(b"count") && ((gte && k >= &kv[..]) || k > &kv[..]) {
                        insert(&mut set, &*k, &*v)
                    }
                }
            } else {
                let key = field_k(field);
                for r in index.scan_prefix(&key[..]) {
                    let (k, v) = r?;
                    if !k.ends_with(b"count") {
                        match field.decode_cmp(&k[key.len()..k.len() - 4])? {
                            Ordering::Greater => insert(&mut set, &*k, &*v),
                            Ordering::Equal if gte => insert(&mut set, &*k, &*v),
                            Ordering::Equal | Ordering::Less => (),
                        }
                    }
                }
            }
            Ok(set)
        }
        fn lt_lte(
            index: &sled::Tree,
            field: &Box<dyn IndexableField>,
            lte: bool,
        ) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            if field.byte_compareable() {
                let kv = field_kv(field)?;
                let key = field_k(field);
                let min = match min_key_with_prefix(index, &key)? {
                    Some(k) => k,
                    None => return Ok(set),
                };
                let max = match max_key_with_prefix(index, &kv)? {
                    Some(k) => k,
                    None => match max_key_with_prefix(index, &key)? {
                        Some(k) => k,
                        None => return Ok(set),
                    },
                };
                for r in index.range(&min[..]..=&max[..]) {
                    let (k, v) = r?;
                    let k = &k[0..kv.len()];
                    if !k.ends_with(b"count") && ((lte && k <= &kv[..]) || k < &kv[..]) {
                        insert(&mut set, &*k, &*v)
                    }
                }
            } else {
                let key = field_k(field);
                for r in index.scan_prefix(&key[..]) {
                    let (k, v) = r?;
                    if !k.ends_with(b"count") {
                        match field.decode_cmp(&k[key.len()..k.len() - 4])? {
                            Ordering::Less => insert(&mut set, &*k, &*v),
                            Ordering::Equal if lte => insert(&mut set, &*k, &*v),
                            Ordering::Equal | Ordering::Greater => (),
                        }
                    }
                }
            }
            Ok(set)
        }
        fn eq(index: &sled::Tree, field: &Box<dyn IndexableField>) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            for r in index.scan_prefix(&field_kv(field)?) {
                let (k, v) = r?;
                if !k.ends_with(b"count") {
                    insert(&mut set, &*k, &*v)
                }
            }
            Ok(set)
        }
        fn all_for_key(
            index: &sled::Tree,
            field: &Box<dyn IndexableField>,
        ) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            let key = field_k(field);
            for r in index.scan_prefix(&key[..]) {
                let (k, v) = r?;
                if !k.ends_with(b"count") {
                    insert(&mut set, &*k, &*v)
                }
            }
            Ok(set)
        }
        fn all(index: &sled::Tree) -> Result<Set<IndexEntry>> {
            let mut set = Set::new();
            for r in index.iter() {
                let (k, v) = r?;
                if !k.ends_with(b"count") {
                    insert(&mut set, &*k, &*v)
                }
            }
            Ok(set)
        }
        match query {
            Query::Eq(field) => eq(&self.index, field),
            Query::Gt(field) => gt_gte(&self.index, field, false),
            Query::Gte(field) => gt_gte(&self.index, field, true),
            Query::Lt(field) => lt_lte(&self.index, field, false),
            Query::Lte(field) => lt_lte(&self.index, field, true),
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
                    let matches = eq(&self.index, field)?;
                    let all = all_for_key(&self.index, field)?;
                    Ok(all.diff(&matches))
                }
                Query::Gt(field) => lt_lte(&self.index, field, true),
                Query::Gte(field) => lt_lte(&self.index, field, false),
                Query::Lt(field) => gt_gte(&self.index, field, true),
                Query::Lte(field) => gt_gte(&self.index, field, false),
                q @ Query::And(_) | q @ Query::Or(_) | q @ Query::Not(_) => {
                    let matches = self.query(q)?;
                    Ok(all(&self.index)?.diff(&matches))
                }
            },
        }
    }
}
