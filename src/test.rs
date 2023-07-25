use super::*;
use rand::{thread_rng, Rng};
use serde_derive::{Deserialize, Serialize};
use std::{ops::Deref, sync::atomic::{AtomicUsize, Ordering as MemOrdering}};

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
        Ok(buf.copy_from_slice(self.0.as_bytes()))
    }
}

fn gen_string() -> CompactString {
    let mut rng = thread_rng();
    let len = rng.gen_range(16..128);
    let mut s = CompactString::new("");
    for _ in 0..len {
        s.push(rng.gen())
    }
    s
}

fn gen_time() -> DateTime<Utc> {
    let base = Utc::now();
    let off = thread_rng().gen::<i32>() >> 1;
    base + chrono::Duration::milliseconds(off as i64)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
            count: Count(thread_rng().gen()),
            index: Index(thread_rng().gen()),
            desc: gen_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
    by_time: BTreeMap<Time, usize>,
    by_count: BTreeMap<Count, usize>,
    by_index: BTreeMap<Index, usize>,
    by_id: BTreeMap<Id, usize>,
}

impl Model {
    fn generate(n: usize) -> Model {
	let mut model = Self {
	    records: vec![],
	    by_time: BTreeMap::new(),
	    by_count: BTreeMap::new(),
	    by_index: BTreeMap::new(),
	    by_id: BTreeMap::new(),
	};
	for i in 0..n {
	    model.records.push(TestRec::gen());
	    match &model.records[i] {
		TestRec::Foo(f) => {
		    model.by_time.insert(f.time, i);
		    model.by_count.insert(f.count, i);
		    model.by_index.insert(f.index, i);
		}
		TestRec::Bar(b) => {
		    model.by_time.insert(b.time, i);
		    model.by_id.insert(b.id.clone(), i);
		}
	    }
	}
	model
    }
}

static N: AtomicUsize = AtomicUsize::new(0);

// creates the indexed json and the model but doesn't add anything to it
async fn init(n: usize) -> Result<(IndexedJson<TestRec>, Model)> {
    let name = format!("test-data{}", N.fetch_add(1, MemOrdering::Relaxed));
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(&name);
    std::fs::remove_dir_all(&dir)?;
    std::fs::create_dir_all(&dir)?;
    let db = IndexedJson::new(&dir).await?;
    let model = Model::generate(n);
    Ok((db, model))
}

#[tokio::test]
async fn test_append_get() {
    let (mut db, model) = init(5).await.unwrap();
    for r in model.records.iter() {
	db.append(r).await.unwrap()
    }
    let mut i = db.first().unwrap();
    for j in 0..5 {
	let (new_i, t) = db.get(i).await.unwrap().unwrap();
	assert_eq!(&t, &model.records[j]);
	i = new_i;
    }
    assert_eq!(None, db.get(i).await.unwrap())
}
