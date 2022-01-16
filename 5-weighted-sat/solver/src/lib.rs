// ~\~ language=Rust filename=solver/src/lib.rs
// ~\~ begin <<lit/main.md|solver/src/lib.rs>>[0]
// ~\~ begin <<lit/main.md|imports>>[0]
use std::{cmp, cmp::max,
    ops::Range,
    str::FromStr,
    io::{BufRead, BufReader},
    collections::{BTreeMap, HashMap},
    fs::{read_dir, File, DirEntry},
    num::NonZeroU16,
};
use anyhow::{Context, Result, anyhow};
use bitvec::prelude::BitArr;
use arrayvec::ArrayVec;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
// ~\~ end

// ~\~ begin <<lit/main.md|algorithm-map>>[0]
// ~\~ end

use std::result::Result as IOResult;
pub fn list_input_files(set: &str, r: Range<u32>) -> Result<Vec<IOResult<DirEntry, std::io::Error>>> {
    let f = |res: &IOResult<DirEntry, std::io::Error> | res.as_ref().ok().filter(|f| {
        let file_name = f.file_name();
        let file_name = file_name.to_str().unwrap();
        // keep only regular files
        f.file_type().unwrap().is_file() &&
        // ... whose names start with the set name,
        file_name.starts_with(set) &&
        // ... continue with an integer between 0 and 15,
        file_name[set.len()..]
        .split('_').next().unwrap().parse::<u32>().ok()
        .filter(|n| r.contains(n)).is_some() &&
        // ... and end with `_inst.dat` (for "instance").
        file_name.ends_with("_inst.dat")
    }).is_some();
    Ok(read_dir("./ds/")?.filter(f).collect())
}

// ~\~ begin <<lit/main.md|problem-instance-definition>>[0]
type Id = NonZeroU16;
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Literal(bool, Id);
pub type Clause = ArrayVec<Literal, 3>;

const MAX_CLAUSES: usize = 512;
const MAX_VARIABLES: usize = 256;
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Instance {
    pub id: i32,
    pub weights: ArrayVec<Id, MAX_VARIABLES>,
    pub clauses: ArrayVec<Clause, MAX_CLAUSES>,
}

// ~\~ end

// ~\~ begin <<lit/main.md|solution-definition>>[0]
pub type Config = BitArr!(for MAX_VARIABLES);

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Solution<'a> { weight: u32, cfg: Config, pub inst: &'a Instance }

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OptimalSolution {
    pub id: String,
    pub weight: u32,
    pub cfg: Config,
    pub params: InstanceParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceParams {
    variables: u8,
    clauses: u16,
}

impl From<Instance> for InstanceParams {
    fn from(inst: Instance) -> Self {
        InstanceParams {
            variables: inst.weights.len() as u8,
            clauses:   inst.clauses.len() as u16,
        }
    }
}

impl From<OptimalSolution> for InstanceParams {
    fn from(opt: OptimalSolution) -> Self {
        opt.params
    }
}

// ~\~ begin <<lit/main.md|solution-helpers>>[0]
impl <'a> PartialOrd for Solution<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.weight.cmp(&other.weight))
    }
}

impl <'a> Ord for Solution<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl <'a> Solution<'a> {
    fn with(mut self, i: usize) -> Solution<'a> {
        self.set(i, true)
    }

    fn without(mut self, i: usize) -> Solution<'a> {
        self.set(i, false)
    }

    fn invert(mut self) -> Solution<'a> {
        for i in 0..self.inst.weights.len() {
            self.set(i, !self.cfg[i]);
        }
        self
    }

    fn set(&mut self, i: usize, set: bool) -> Solution<'a> {
        let w = self.inst.weights[i];
        let k = if set { 1 } else { -1 };
        if self.cfg[i] != set {
            self.cfg.set(i, set);
            self.weight = (self.weight as i32 + k * w.get() as i32) as u32;
        }
        *self
    }

    fn default(inst: &'a Instance) -> Solution<'a> {
        Solution { weight: 0, cfg: Config::default(), inst }
    }
}
// ~\~ end
// ~\~ end

// ~\~ begin <<lit/main.md|parser>>[0]
// ~\~ begin <<lit/main.md|boilerplate>>[0]
trait Boilerplate {
    fn parse_next<T: FromStr>(&mut self) -> Result<T>
      where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static;
}

impl Boilerplate for std::str::SplitWhitespace<'_> {
    fn parse_next<T: FromStr>(&mut self) -> Result<T>
      where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static {
        let str = self.next().ok_or_else(|| anyhow!("unexpected end of input"))?;
        str.parse::<T>()
           .with_context(|| format!("cannot parse {}", str))
    }
}
// ~\~ end

pub fn parse_clauses<T: BufRead>(stream: &mut T) -> Result<ArrayVec<Clause, MAX_CLAUSES>> {
    let to_literal: fn(i16) -> Result<Literal> = |n| Ok(Literal(
        n.is_positive(), NonZeroU16::new(n.abs() as u16).ok_or_else(|| anyhow!("variables start from 1"))?
    ));
    let mut clauses = ArrayVec::new();
    let mut input = String::new();

    while stream.read_line(&mut input)? != 0 {
        let mut numbers = input.split_whitespace();
        clauses.push(ArrayVec::from([
            to_literal(numbers.parse_next()?)?,
            to_literal(numbers.parse_next()?)?,
            to_literal(numbers.parse_next()?)?,
        ]));
    }

    Ok(clauses)
}

fn parse_solution_line<T: BufRead>(mut stream: T, params: InstanceParams) -> Result<Option<OptimalSolution>> {
    let mut input = String::new();
    if stream.read_line(&mut input)? == 0 {
        return Ok(None)
    }

    let mut numbers = input.split_whitespace();
    let id = numbers.parse_next()?;
    let weight = numbers.parse_next()?;

    let mut cfg = Config::default();
    let mut i = 0;
    loop {
        let a: i16 = numbers.parse_next()?;
        if a == 0 { break }
        cfg.set(i, a.is_positive());
        i += 1;
    }

    Ok(Some(OptimalSolution {id, weight, cfg, params}))
}

pub fn load_solutions(set: char) -> Result<HashMap<InstanceParams, OptimalSolution>> {
    let mut solutions = HashMap::new();

    let files = read_dir("../data/")?
        .filter(|res| res.as_ref().ok().filter(|f| {
            let name = f.file_name().into_string().unwrap();
            f.file_type().unwrap().is_file() &&
            name.ends_with(&(set.to_string() + "-opt.dat"))
        }).is_some());

    for file in files {
        let file = file?;
        let filename = file.file_name().into_string().expect("FS error");

        let mut params = filename[3..].split('-').take(2).map(|n| n.parse::<u16>());
        let variables = params.next().unwrap()? as u8;
        let clauses = params.next().unwrap()?;
        let params = InstanceParams { variables, clauses };

        let mut stream = BufReader::new(File::open(file.path())?);
        while let Some(opt) = parse_solution_line(&mut stream, params)? {
            solutions.insert(params, opt);
        }
    }

    Ok(solutions)
}
// ~\~ end

trait IteratorRandomWeighted: Iterator + Sized + Clone {
    fn choose_weighted<Rng: ?Sized, W>(&mut self, rng: &mut Rng, f: fn(Self::Item) -> W) -> Option<Self::Item>
    where
        Rng: rand::Rng,
        W: for<'a> core::ops::AddAssign<&'a W>
         + rand::distributions::uniform::SampleUniform
         + std::cmp::PartialOrd
         + Default
         + Clone {
        use rand::prelude::*;
        let dist = rand::distributions::WeightedIndex::new(self.clone().map(f)).ok()?;
        self.nth(dist.sample(rng))
    }
}

impl<I> IteratorRandomWeighted for I where I: Iterator + Sized + Clone {}

impl Instance {
    // ~\~ begin <<lit/main.md|solver-dpw>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-dpc>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-fptas>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-greedy>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-greedy-redux>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bb>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bf>>[0]
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-sa>>[0]
    // ~\~ end
}

// ~\~ begin <<lit/main.md|tests>>[0]
#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};
    use std::{fs::File, io::BufReader};

    #[derive(Clone, Debug)]
    #[repr(transparent)]
    struct ArrayVecProxy<T, const CAP: usize>(ArrayVec<T, CAP>);

    type LiteralProxy = ArrayVecProxy<Literal, 3>;
    type ClauseProxy = ArrayVecProxy<LiteralProxy, MAX_CLAUSES>;

    impl<T, const CAP: usize> From<ArrayVec<T, CAP>> for ArrayVecProxy<T, CAP> {
        fn from(av: ArrayVec<T, CAP>) -> Self {
            ArrayVecProxy(av)
        }
    }

    impl<T, const CAP: usize> From<ArrayVecProxy<T, CAP>> for ArrayVec<T, CAP> {
        fn from(ArrayVecProxy(av): ArrayVecProxy<T, CAP>) -> Self {
            av
        }
    }

    impl<T: Arbitrary + core::fmt::Debug, const CAP: usize> Arbitrary for ArrayVecProxy<T, CAP> {
        fn arbitrary(g: &mut Gen) -> Self {
            let arr: [T; CAP] = Vec::arbitrary(g)
                .into_iter()
                .take(CAP)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            ArrayVecProxy(arr.into())
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            Box::new(self.0.clone()
                .into_iter()
                .collect::<Vec<T>>()
                .shrink()
                .map(|vec| {
                    let arr: [T; CAP] = vec.try_into().unwrap();
                    ArrayVecProxy(arr.into())
                })
                .into_iter()
            )
        }
    }

    impl Arbitrary for Literal {
        fn arbitrary(g: &mut Gen) -> Self {
            Literal(bool::arbitrary(g), Id::arbitrary(g))
        }
    }

    impl Arbitrary for Instance {
        fn arbitrary(g: &mut Gen) -> Instance {
            let proxy: ArrayVec<LiteralProxy, MAX_CLAUSES> = (ArrayVecProxy::arbitrary(g) as ClauseProxy).into();
            Instance {
                id:      i32::arbitrary(g),
                weights: ArrayVecProxy::arbitrary(g).into(),
                clauses: proxy.into_iter().map(Into::into).collect(),
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let data = self.clone();
            #[allow(clippy::needless_collect)]
            let chain: Vec<Instance> = quickcheck::empty_shrinker()
                .chain(self.id.shrink().map(|id| Instance {id, ..(&data).clone()}))
                .chain(ArrayVecProxy(self.weights.clone())
                    .shrink()
                    .map(|weights| Instance {
                        weights: weights.into(),
                        ..(&data).clone()
                    })
                )
                .chain(ArrayVecProxy(
                        self.clauses.clone()
                            .into_iter()
                            .map(|c| ArrayVecProxy(c))
                            .collect()
                    )
                    .shrink()
                    .map(|clauses| {
                        let av: ArrayVec<LiteralProxy, MAX_CLAUSES> = clauses.into();
                        Instance {
                            clauses: av.into_iter().map(Into::into).collect(),
                            ..(&data).clone()
                        }
                    })
                    .filter(|i| !i.clauses.is_empty())
                )
                .collect();
            Box::new(chain.into_iter())
        }
    }

    impl <'a> Solution<'a> {
        fn assert_valid(&self) {
            let Solution { weight, cfg, inst } = self;
            let Instance { weights, clauses, .. } = inst;

            let computed_weight = weights
                .iter()
                .zip(cfg)
                .map(|(w, b)| {
                    if *b { w.get() as u32 } else { 0 }
                })
                .sum::<u32>();

            let satisfied = clauses.into_iter()
                .all(|clause| clause
                    .into_iter()
                    .any(|&Literal(pos, id)| pos == cfg[id.get() as usize])
                );

            assert_eq!(computed_weight, *weight);
            assert!(satisfied);
        }
    }
}

// ~\~ end
// ~\~ end
