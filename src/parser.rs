use crate::{IndexableField, Query};
use anyhow::Result;
use combine::{
    attempt, between, choice,
    error::Format,
    many1, one_of, optional, parser,
    parser::{
        char::{alpha_num, spaces, string},
        combinator::recognize,
        range::{take_while, take_while1},
        repeat::escaped,
    },
    sep_by1,
    stream::{position, Range},
    token, unexpected_any, value, EasyParser, ParseError, Parser, RangeStream,
};
use fxhash::FxHashMap;
use netidx_core::utils;
use std::{borrow::Cow, sync::Arc};

pub type LeafFn = Box<dyn Fn(&str) -> Result<Arc<dyn IndexableField + Send + Sync + 'static>> + Send + Sync + 'static>;
pub type LeafTbl = FxHashMap<&'static str, LeafFn>;

fn key<I>() -> impl Parser<I, Output = String>
where
    I: RangeStream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    many1(choice((alpha_num(), token('_'), token('-'))))
}

fn op<I>() -> impl Parser<I, Output = String>
where
    I: RangeStream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    choice((
        string("==").map(String::from),
        attempt(string(">=")).map(String::from),
        attempt(string("<=")).map(String::from),
        token('>').map(String::from),
        token('<').map(String::from),
        string("!=").map(String::from),
    ))
}

fn escaped_string<I>() -> impl Parser<I, Output = String>
where
    I: RangeStream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    recognize(escaped(
        take_while1(move |c| c != '"' && c != '\\'),
        '\\',
        one_of(r#""\"#.chars()),
    ))
    .map(|s| match utils::unescape(&s, '\\') {
        Cow::Borrowed(_) => s, // it didn't need unescaping, so just return it
        Cow::Owned(s) => s,
    })
}

fn quoted<I>() -> impl Parser<I, Output = String>
where
    I: RangeStream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    spaces().with(between(token('"'), token('"'), escaped_string()))
}

fn int<I>() -> impl Parser<I, Output = String>
where
    I: RangeStream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    attempt(recognize((
        optional(token('-')),
        take_while1(|c: char| c.is_digit(10)),
    )))
}

fn flt<I>() -> impl Parser<I, Output = String>
where
    I: RangeStream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    choice((
        attempt(recognize((
            optional(token('-')),
            take_while1(|c: char| c.is_digit(10)),
            optional(token('.')),
            take_while(|c: char| c.is_digit(10)),
            token('e'),
            int(),
        ))),
        attempt(recognize((
            optional(token('-')),
            take_while1(|c: char| c.is_digit(10)),
            token('.'),
            take_while(|c: char| c.is_digit(10)),
        ))),
    ))
}

fn query_<'a, I>(leaf: &'a LeafTbl) -> impl Parser<I, Output = Query> + 'a
where
    I: RangeStream<Token = char> + 'a,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Range: Range,
{
    spaces().with(choice((
        token('!')
            .with(query(leaf))
            .map(|q| Query::Not(Box::new(q))),
        attempt(between(
            token('('),
            token(')'),
            sep_by1(query(leaf), spaces().with(string("&&"))),
        ))
        .map(Query::And),
        attempt(between(
            token('('),
            token(')'),
            sep_by1(query(leaf), spaces().with(string("||"))),
        ))
        .map(Query::Or),
        (
            key(),
            spaces().with(op()),
            spaces().with(choice((quoted(), flt(), int()))),
        )
        .then(|(k, op, v)| match leaf.get(&k.as_str()) {
            None => unexpected_any(Format(format!("invalid key {}", k))).right(),
            Some(f) => match f(v.as_str()) {
                Err(e) => {
                    unexpected_any(Format(format!("invalid value {v} error {:?}", e))).right()
                }
                Ok(ql) => value(match op.as_str() {
                    "==" => Query::Eq(ql),
                    "!=" => Query::Not(Box::new(Query::Eq(ql))),
                    ">=" => Query::Gte(ql),
                    "<=" => Query::Lte(ql),
                    ">" => Query::Gt(ql),
                    "<" => Query::Lt(ql),
                    _ => unreachable!(),
                })
                .left(),
            },
        }),
    )))
}

parser! {
    fn query['a, I](leaf: &'a LeafTbl)(I) -> Query
    where [I: RangeStream<Token = char>, I::Range: Range]
    {
        query_(leaf)
    }
}

pub(crate) fn parse_query(leaf: &LeafTbl, s: &str) -> anyhow::Result<Query> {
    query(leaf)
        .easy_parse(position::Stream::new(s))
        .map(|(r, _)| r)
        .map_err(|e| anyhow::anyhow!(format!("{}", e)))
}
