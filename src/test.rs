use super::*;
use rand::{thread_rng, Rng};
use serde_derive::{Deserialize, Serialize};
use std::{
    ops::Deref,
    sync::atomic::{AtomicUsize, Ordering as MemOrdering},
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
struct Count(u64);

impl Deref for Count {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IndexableField for Count {
    fn byte_compareable(&self) -> bool {
        true
    }

    fn key(&self) -> &'static str {
        "count"
    }

    fn encode(&self, buf: &mut SmallVec<[u8; 128]>) -> Result<()> {
        Ok(buf.extend_from_slice(&u64::to_be_bytes(self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
struct Index(u64);

impl Deref for Index {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IndexableField for Index {
    fn byte_compareable(&self) -> bool {
        true
    }

    fn key(&self) -> &'static str {
        "index"
    }

    fn encode(&self, buf: &mut SmallVec<[u8; 128]>) -> Result<()> {
        Ok(buf.extend_from_slice(&u64::to_be_bytes(self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
struct Time(DateTime<Utc>);

impl Deref for Time {
    type Target = DateTime<Utc>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IndexableField for Time {
    fn byte_compareable(&self) -> bool {
        false
    }

    fn key(&self) -> &'static str {
        "time"
    }

    fn encode(&self, buf: &mut SmallVec<[u8; 128]>) -> Result<()> {
        let l = Pack::encoded_len(&self.0);
        buf.resize(l, 0);
        Ok(Pack::encode(&self.0, &mut &mut buf[..])?)
    }

    fn decode_cmp(&self, mut b: &[u8]) -> Result<Ordering> {
        let a: DateTime<Utc> = Pack::decode(&mut b)?;
        Ok(self.0.cmp(&a))
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
struct Id(CompactString);

impl Deref for Id {
    type Target = CompactString;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IndexableField for Id {
    fn byte_compareable(&self) -> bool {
        true
    }

    fn key(&self) -> &'static str {
        "id"
    }

    fn encode(&self, buf: &mut SmallVec<[u8; 128]>) -> Result<()> {
        Ok(buf.extend_from_slice(self.0.as_bytes()))
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

fn gen_string() -> CompactString {
    let mut rng = thread_rng();
    let len = rng.gen_range(8..24);
    let mut s = CompactString::new("");
    for _ in 0..len {
        s.push(rng.gen_range('a'..'z'))
    }
    s
}

fn gen_time() -> DateTime<Utc> {
    let base = Utc::now();
    let off = thread_rng().gen::<i32>() >> 1;
    base + chrono::Duration::milliseconds(off as i64)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct Foo {
    time: Time,
    count: Count,
    index: Index,
    desc: CompactString,
}

impl Foo {
    fn gen() -> Self {
        Self {
            time: Time(gen_time()),
            count: Count(thread_rng().gen_range(0..100)),
            index: Index(thread_rng().gen_range(0..100)),
            desc: gen_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct Bar {
    time: Time,
    id: Id,
    blob: CompactString,
}

impl Bar {
    fn gen() -> Self {
        Self {
            time: Time(gen_time()),
            id: Id(gen_string()),
            blob: gen_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "type")]
enum TestRec {
    Foo(Foo),
    Bar(Bar),
}

impl TestRec {
    fn gen() -> Self {
        if thread_rng().gen_bool(0.5) {
            Self::Foo(Foo::gen())
        } else {
            Self::Bar(Bar::gen())
        }
    }
}

impl Indexable for TestRec {
    type Iter = SmallVec<[Box<dyn IndexableField>; 3]>;

    fn index(&self) -> Self::Iter {
        let mut res = SmallVec::new();
        match self {
            TestRec::Foo(f) => {
                res.push(Box::new(f.time) as Box<dyn IndexableField>);
                res.push(Box::new(f.count) as Box<dyn IndexableField>);
                res.push(Box::new(f.index) as Box<dyn IndexableField>);
            }
            TestRec::Bar(b) => {
                res.push(Box::new(b.time) as Box<dyn IndexableField>);
                res.push(Box::new(b.id.clone()) as Box<dyn IndexableField>);
            }
        }
        res
    }

    fn timestamp(&self) -> DateTime<Utc> {
        match self {
            TestRec::Foo(f) => *f.time,
            TestRec::Bar(b) => *b.time,
        }
    }
}

struct Model {
    records: Vec<TestRec>,
    by_time: BTreeMap<Time, Set<usize>>,
    by_count: BTreeMap<Count, Set<usize>>,
    by_index: BTreeMap<Index, Set<usize>>,
    by_id: BTreeMap<Id, Set<usize>>,
}

impl Model {
    fn eq(&self, field: &Box<dyn IndexableField>) -> Set<usize> {
        let mut res = Set::new();
        if let Some(time) = field.as_any().downcast_ref::<Time>() {
            res = self.by_time.get(time).cloned().unwrap_or(Set::new());
        }
        if let Some(count) = field.as_any().downcast_ref::<Count>() {
            res = self.by_count.get(count).cloned().unwrap_or(Set::new());
        }
        if let Some(index) = field.as_any().downcast_ref::<Index>() {
            res = self.by_index.get(index).cloned().unwrap_or(Set::new());
        }
        if let Some(id) = field.as_any().downcast_ref::<Id>() {
            res = self.by_id.get(id).cloned().unwrap_or(Set::new());
        }
        res
    }

    fn gt_gte(&self, gte: bool, field: &Box<dyn IndexableField>) -> Set<usize> {
        fn gt_gte<T: Ord + Eq>(tbl: &BTreeMap<T, Set<usize>>, gte: bool, t: &T) -> Set<usize> {
            let mut res = Set::new();
            for (k, v) in tbl.range(t..) {
                if (gte && t == k) || k > t {
                    res = res.union(v);
                }
            }
            res
        }
        let mut res = Set::new();
        if let Some(time) = field.as_any().downcast_ref::<Time>() {
            res = gt_gte(&self.by_time, gte, time);
        }
        if let Some(count) = field.as_any().downcast_ref::<Count>() {
            res = gt_gte(&self.by_count, gte, count);
        }
        if let Some(index) = field.as_any().downcast_ref::<Index>() {
            res = gt_gte(&self.by_index, gte, index);
        }
        if let Some(id) = field.as_any().downcast_ref::<Id>() {
            res = gt_gte(&self.by_id, gte, id);
        }
        res
    }

    fn lt_lte(&self, lte: bool, field: &Box<dyn IndexableField>) -> Set<usize> {
        fn lt_lte<T: Ord + Eq>(tbl: &BTreeMap<T, Set<usize>>, lte: bool, t: &T) -> Set<usize> {
            let mut res = Set::new();
            for (k, v) in tbl.range(..=t) {
                if (lte && t == k) || k < t {
                    res = res.union(v);
                }
            }
            res
        }
        let mut res = Set::new();
        if let Some(time) = field.as_any().downcast_ref::<Time>() {
            res = lt_lte(&self.by_time, lte, time);
        }
        if let Some(count) = field.as_any().downcast_ref::<Count>() {
            res = lt_lte(&self.by_count, lte, count);
        }
        if let Some(index) = field.as_any().downcast_ref::<Index>() {
            res = lt_lte(&self.by_index, lte, index);
        }
        if let Some(id) = field.as_any().downcast_ref::<Id>() {
            res = lt_lte(&self.by_id, lte, id);
        }
        res
    }

    fn all_for_key(&self, key: &str) -> Set<usize> {
        match key {
            "count" => self
                .by_count
                .values()
                .fold(Set::new(), |acc, s| acc.union(s)),
            "index" => self
                .by_index
                .values()
                .fold(Set::new(), |acc, s| acc.union(s)),
            "time" => self
                .by_time
                .values()
                .fold(Set::new(), |acc, s| acc.union(s)),
            "id" => self.by_id.values().fold(Set::new(), |acc, s| acc.union(s)),
            _ => Set::new(),
        }
    }

    fn all(&self) -> Set<usize> {
        self.by_count
            .values()
            .chain(self.by_index.values())
            .chain(self.by_time.values())
            .chain(self.by_id.values())
            .fold(Set::new(), |acc, s| acc.union(s))
    }

    fn query(&self, q: &Query) -> Set<usize> {
        match q {
            Query::Eq(f) => self.eq(f),
            Query::Gt(f) => self.gt_gte(false, f),
            Query::Gte(f) => self.gt_gte(true, f),
            Query::Lt(f) => self.lt_lte(false, f),
            Query::Lte(f) => self.lt_lte(true, f),
            Query::And(qs) => qs
                .iter()
                .map(|q| self.query(q))
                .fold(None, |acc: Option<Set<_>>, s| match acc {
                    Some(acc) => Some(acc.intersect(&s)),
                    None => Some(s),
                })
                .unwrap_or_else(Set::new),
            Query::Or(qs) => qs
                .iter()
                .map(|q| self.query(q))
                .fold(Set::new(), |acc, s| acc.union(&s)),
            Query::Not(q) => match &**q {
                Query::Eq(f) => self.all_for_key(f.key()).diff(&self.eq(f)),
                Query::Gt(f) => self.lt_lte(true, f),
                Query::Gte(f) => self.lt_lte(false, f),
                Query::Lt(f) => self.gt_gte(true, f),
                Query::Lte(f) => self.gt_gte(false, f),
                q @ Query::And(_) | q @ Query::Or(_) | q @ Query::Not(_) => {
                    self.all().diff(&self.query(q))
                }
            },
        }
    }

    fn new(iter: impl IntoIterator<Item = TestRec>) -> Model {
        let mut model = Self {
            records: vec![],
            by_time: BTreeMap::new(),
            by_count: BTreeMap::new(),
            by_index: BTreeMap::new(),
            by_id: BTreeMap::new(),
        };
	model.records.extend(iter);
        model.records.sort_by_key(|r| r.timestamp());
        for (i, r) in model.records.iter().enumerate() {
            match r {
                TestRec::Foo(f) => {
                    model
                        .by_time
                        .entry(f.time)
                        .or_insert(Set::new())
                        .insert_cow(i);
                    model
                        .by_count
                        .entry(f.count)
                        .or_insert(Set::new())
                        .insert_cow(i);
                    model
                        .by_index
                        .entry(f.index)
                        .or_insert(Set::new())
                        .insert_cow(i);
                }
                TestRec::Bar(b) => {
                    model
                        .by_time
                        .entry(b.time)
                        .or_insert(Set::new())
                        .insert_cow(i);
                    model
                        .by_id
                        .entry(b.id.clone())
                        .or_insert(Set::new())
                        .insert_cow(i);
                }
            }
        }
	model
    }

    fn generate(n: usize) -> Model {
	Self::new((0..n).into_iter().map(|_| TestRec::gen()))
    }
}

static N: AtomicUsize = AtomicUsize::new(0);

// creates the indexed json and the model but doesn't add anything to it
async fn init(n: usize) -> Result<(IndexedJson<TestRec>, Model)> {
    let name = format!("test-data{}", N.fetch_add(1, MemOrdering::Relaxed));
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(&name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;
    let db = IndexedJson::open(&dir).await?;
    let model = Model::generate(n);
    Ok((db, model))
}

async fn init_and_append(n: usize) -> Result<(IndexedJson<TestRec>, Model)> {
    let (mut db, model) = init(n).await?;
    dbg!("starting write");
    for r in model.records.iter() {
        db.append(r).await?
    }
    dbg!("starting flush");
    db.flush().await?;
    dbg!("flushed");
    Ok((db, model))
}

async fn test_append_get_n(n: usize) {
    let (mut db, model) = init(n).await.unwrap();
    for r in model.records.iter() {
        db.append(r).await.unwrap()
    }
    db.flush().await.unwrap();
    let mut i = db.first().unwrap();
    for j in 0..n {
        let (new_i, t) = db.get(i).await.unwrap().unwrap();
        assert_eq!(&t, &model.records[j]);
        i = new_i;
    }
    assert_eq!(None, db.get(i).await.unwrap())
}

async fn exec_query<'a>(
    model: &'a Model,
    db: &mut IndexedJson<TestRec>,
    q: &Query,
) -> (Vec<&'a TestRec>, Vec<TestRec>) {
    let mut model_res = model
        .query(&q)
        .into_iter()
        .map(|i| &model.records[*i])
        .collect::<Vec<_>>();
    let mut db_res = vec![];
    for i in &db.query(q).unwrap() {
        let (_next, t) = db.get(*i).await.unwrap().unwrap();
        db_res.push(t);
    }
    model_res.sort();
    db_res.sort();
    (model_res, db_res)
}

#[tokio::test]
async fn test_append_get_small() {
    test_append_get_n(10).await
}

#[tokio::test]
async fn test_append_get_large() {
    test_append_get_n(100_000).await
}

async fn test_query_gen(q: impl Fn(&TestRec) -> Query) {
    let (mut db, model) = init_and_append(100_000).await.unwrap();
    let r = &model.records[0];
    let q = dbg!(q(r));
    let (model_res, db_res) = exec_query(&model, &mut db, &q).await;
    if !model_res
        .iter()
        .zip(db_res.iter())
        .all(|(mr, dr)| *mr == dr)
    {
        panic!("{:?} differs from {:?}", model_res, db_res)
    }
    let (model_res, db_res) = exec_query(&model, &mut db, &Query::Not(Box::new(q))).await;
    if !model_res
        .iter()
        .zip(db_res.iter())
        .all(|(mr, dr)| *mr == dr)
    {
        panic!("{:?} differs from {:?}", model_res, db_res)
    }
}

async fn test_single_query_gen(f: impl Fn(Box<dyn IndexableField>) -> Query) {
    test_query_gen(|r| match r {
        TestRec::Bar(b) => {
	    if thread_rng().gen_bool(0.5) {
		f(Box::new(b.id.clone()))
	    } else {
		f(Box::new(b.time))
	    }
	},
        TestRec::Foo(b) => {
	    if thread_rng().gen_bool(0.33) {
		f(Box::new(b.count))
	    } else if thread_rng().gen_bool(0.33) {
		f(Box::new(b.index))
	    } else {
		f(Box::new(b.time))
	    }
	},
    })
    .await
}

#[tokio::test]
async fn test_query_eq() {
    test_single_query_gen(Query::Eq).await
}

#[tokio::test]
async fn test_query_lt() {
    test_single_query_gen(Query::Lt).await
}

#[tokio::test]
async fn test_query_lte() {
    test_single_query_gen(Query::Lte).await
}

#[tokio::test]
async fn test_query_gt() {
    test_single_query_gen(Query::Gt).await
}

#[tokio::test]
async fn test_query_gte() {
    test_single_query_gen(Query::Gte).await
}

#[tokio::test]
async fn test_query_and() {
    test_query_gen(|r| match r {
        TestRec::Bar(b) => Query::And(vec![
            Query::Eq(Box::new(b.time)),
            Query::Gte(Box::new(b.id.clone())),
        ]),
        TestRec::Foo(f) => Query::And(vec![
            Query::Lte(Box::new(f.count)),
            Query::Eq(Box::new(f.time)),
            Query::Eq(Box::new(f.index)),
        ]),
    })
    .await
}

#[tokio::test]
async fn test_query_or() {
    test_query_gen(|r| match r {
        TestRec::Bar(b) => dbg!(Query::Or(vec![
            Query::Lt(Box::new(b.time)),
            Query::Eq(Box::new(b.id.clone())),
        ])),
        TestRec::Foo(f) => dbg!(Query::Or(vec![
	    Query::Lt(Box::new(f.time)),
            Query::Eq(Box::new(f.count)),
            Query::Eq(Box::new(f.index)),
        ])),
    })
    .await
}

#[tokio::test]
async fn test_reindex() {
    use std::{fs::OpenOptions, io::Write};
    let (db, mut model) = init_and_append(100_000).await.unwrap();
    let path = PathBuf::from(db.path());
    let file = path.read_dir().unwrap().filter_map(|e| {
	let e = e.unwrap();
	if e.file_type().unwrap().is_file() {
	    Some(e.path())
	} else {
	    None
	}
    }).next().unwrap();
    let new_t = TestRec::gen();
    let mut fd = OpenOptions::new().append(true).open(&file).unwrap();
    serde_json::to_writer(&mut fd, &new_t).unwrap();
    fd.write_all(&[b'\n']).unwrap();
    drop(fd);
    drop(db);
    model.records.push(new_t.clone());
    model.records.sort_by_key(|t| t.timestamp());
    let model = Model::new(model.records);
    // should detect that a file is modified and reindex the archive
    let mut db = IndexedJson::open(&path).await.unwrap();
    let q = match &new_t {
	TestRec::Bar(b) => dbg!(Query::Eq(Box::new(b.id.clone()))),
	TestRec::Foo(f) => dbg!(Query::Eq(Box::new(f.time)))
    };
    let (model_res, db_res) = exec_query(&model, &mut db, &q).await;
    if !model_res
        .iter()
        .zip(db_res.iter())
        .all(|(mr, dr)| *mr == dr)
    {
        panic!("{:?} differs from {:?}", model_res, db_res)
    }
}
