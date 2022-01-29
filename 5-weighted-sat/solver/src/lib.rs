// ~\~ language=Rust filename=solver/src/lib.rs
// ~\~ begin <<lit/main.md|solver/src/lib.rs>>[0]
// ~\~ begin <<lit/main.md|imports>>[0]
#![feature(iter_intersperse)]

use serde::{Deserialize, Serialize};
use std::{cmp,
    ops::Range,
    str::FromStr,
    io::{BufRead, BufReader},
    collections::{HashMap},
    fs::{read_dir, File, DirEntry},
    num::NonZeroU16,
};
use anyhow::{Context, Result, anyhow};
use bitvec::prelude::BitArr;
use arrayvec::ArrayVec;
use rand::prelude::SliceRandom;

#[cfg(test)]
extern crate quickcheck_macros;
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
pub type Id = NonZeroU16;
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Literal(pub bool, pub Id);
pub type Clause = [Literal; 3];

const MAX_CLAUSES: usize = 512;
const MAX_VARIABLES: usize = 256;
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Instance {
    pub id: i32,
    pub weights: ArrayVec<NonZeroU16, MAX_VARIABLES>,
    pub total_weight: u32,
    pub clauses: ArrayVec<Clause, MAX_CLAUSES>,
}

// ~\~ end

// ~\~ begin <<lit/main.md|solution-definition>>[0]
pub type Config = BitArr!(for MAX_VARIABLES);

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Solution<'a> {
    pub weight: u32,
    pub fitness: u64,
    pub cfg: Config,
    pub inst: &'a Instance,
    pub satisfied: bool,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OptimalSolution {
    pub full_id: String,
    pub id: i32,
    pub weight: u32,
    pub cfg: Config,
    pub params: InstanceParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstanceParams {
    variables: u8,
    clauses: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvolutionaryConfig {
    pub set: char,
    pub mutation_chance: f64,
    pub n_instances: u16,
    pub generations: u32,
    pub population_size: usize,
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

#[allow(unused)]
impl <'a> Solution<'a> {
    fn with(mut self, i: usize) -> Solution<'a> {
        self.set(i, true)
    }

    fn without(mut self, i: usize) -> Solution<'a> {
        self.set(i, false)
    }

    fn invert(mut self) -> Solution<'a> {
        for i in 0..self.inst.weights.len() {
            self.set_unsafe(i, !self.cfg[i]);
        }
        self.satisfied = satisfied(&self.inst.clauses, &self.cfg);
        self
    }

    fn set(&mut self, i: usize, set: bool) -> Solution<'a> {
        self.set_unsafe(i, set);
        self.satisfied = satisfied(&self.inst.clauses, &self.cfg);
        *self
    }

    fn set_unsafe(&mut self, i: usize, set: bool) -> Solution<'a> {
        let w = self.inst.weights[i];
        let k = if set { 1 } else { -1 };
        if self.cfg[i] != set {
            self.cfg.set(i, set);
            self.weight = (self.weight as i32 + k * w.get() as i32) as u32;
        }
        *self
    }

    fn default(inst: &'a Instance, evo_config: &EvolutionaryConfig) -> Solution<'a> {
        Solution::new(0, Config::default(), inst, evo_config)
    }

    pub fn new(weight: u32, cfg: Config, inst: &'a Instance, evo_config: &EvolutionaryConfig) -> Solution<'a> {
        let sln = Solution {
            weight, cfg, inst, satisfied: satisfied(&inst.clauses, &cfg), fitness: 0
        };
        Solution { fitness: compute_fitness(&sln, evo_config), ..sln }
    }

    pub fn valid(&self, evo_config: &EvolutionaryConfig) -> bool {
        let Solution { weight, cfg, inst, satisfied, fitness } = *self;
        let Instance { weights, clauses, .. } = inst;

        let computed_weight = weights
            .iter()
            .zip(cfg)
            .map(|(w, b)| {
                if b { w.get() as u32 } else { 0 }
            })
            .sum::<u32>();

        let computed_fitness = compute_fitness(self, evo_config);
        computed_fitness == fitness && computed_weight == weight && satisfied
    }

    pub fn dump(&self) -> String {
        dump_solution(self.inst.id, self.weight, &self.cfg, &self.inst.clone().into())
    }
}

impl OptimalSolution {
    pub fn dump(&self) -> String {
        dump_solution(self.id, self.weight, &self.cfg, &self.clone().into())
    }
}

fn dump_solution(id: i32, weight: u32, cfg: &Config, params: &InstanceParams) -> String {
    use core::iter::once;
    once(format!("uf{}-0{}", params.variables, id))
    .chain(once(weight.to_string()))
    .chain((1..=params.variables as usize)
        .map(|i| if cfg[i - 1] { 1 } else { -1 } * i as i16)
        .chain(once(0))
        .map(|id| id.to_string())
    )
    .intersperse(" ".into())
    .collect()
}

pub fn compute_fitness(sln: &Solution, _evo_config: &EvolutionaryConfig) -> u64 {
    let sat_clauses: u32 = sln.inst.clauses.iter()
        .map(|clause|
            clause.iter().all(|&Literal(pos, id)|
                pos == sln.cfg[id.get() as usize - 1]
            )
        )
        .map(|sat| sat as u32)
        .sum();

    let sat_component = (1u32 << 12) as f64;
    let weight_component = (1u32 << 8) as f64;
    let score = sat_component * sat_clauses as f64 / sln.inst.clauses.len() as f64
        + weight_component * sln.weight as f64 / sln.inst.total_weight as f64;
    score as u64
}

pub fn satisfied(clauses: &ArrayVec<Clause, MAX_CLAUSES>, cfg: &Config) -> bool {
    clauses.iter().all(|clause| clause
        .iter()
        .any(|&Literal(pos, id)| pos == cfg[id.get() as usize - 1])
    )
}

// ~\~ end
// ~\~ end

// ~\~ begin <<lit/main.md|parser>>[0]
// ~\~ begin <<lit/main.md|boilerplate>>[0]
trait Boilerplate {
    fn parse_next<T: FromStr>(&mut self) -> Result<T>
      where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static;
}

impl<'a, Iter> Boilerplate for Iter where Iter: Iterator<Item = &'a str> {
    fn parse_next<T: FromStr>(&mut self) -> Result<T>
      where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static {
        let str = self.next().ok_or_else(|| anyhow!("unexpected end of input"))?;
        str.parse::<T>()
           .with_context(|| anyhow!("cannot parse {}", str))
    }
}
// ~\~ end

pub fn parse_clauses<T: Iterator<Item = String>>(lines: &mut T) -> Result<ArrayVec<Clause, MAX_CLAUSES>> {
    let to_literal: fn(i16) -> Result<Literal> = |n| Ok(Literal(
        n.is_positive(), NonZeroU16::new(n.abs() as u16).ok_or_else(|| anyhow!("variables start from 1"))?
    ));
    let mut clauses = ArrayVec::new();

    for line in lines {
        let mut numbers = line.split_whitespace();
        clauses.push([
            to_literal(numbers.parse_next()?)?,
            to_literal(numbers.parse_next()?)?,
            to_literal(numbers.parse_next()?)?,
        ]);
    }

    Ok(clauses)
}

fn parse_solution_line<T: BufRead>(mut stream: T, params: InstanceParams) -> Result<Option<OptimalSolution>> {
    let mut input = String::new();
    if stream.read_line(&mut input)? == 0 {
        return Ok(None)
    }

    let mut line = input.split_whitespace();
    let full_id: String = line.parse_next()?;
    let id = full_id.split('-').skip(1).parse_next()?;
    let weight = line.parse_next()?;

    let mut cfg = Config::default();
    let mut i = 0;
    loop {
        let a: i16 = line.parse_next()?;
        if a == 0 { break }
        cfg.set(i, a.is_positive());
        i += 1;
    }

    Ok(Some(OptimalSolution {full_id, id, weight, cfg, params}))
}

pub fn load_instances(set: char) -> Result<Vec<Instance>> {
    read_dir("../data/")?.filter_map(|entry| entry.ok()
        .filter(|entry|
            entry.file_name().into_string().unwrap().ends_with(&(set.to_string() + "1"))
        )
        .and_then(|entry|
            entry.file_type().ok().filter(|&typ| typ.is_dir()).and(Some(entry))
        )
    )
    .flat_map(|dir| {
        let params = params_from_filename(&dir.file_name().into_string().unwrap()).unwrap();
        read_dir(dir.path()).into_iter()
            .flatten().flatten()
            .map(move |file| (params, file))
    })
    .map(|(_params, file)| {
        let id = file.file_name().into_string().unwrap().split('-')
            .nth(1).unwrap().split('.').next().unwrap().parse().unwrap();

        let mut lines = BufReader::new(File::open(file.path()).unwrap())
            .lines()
            .map(|l| l.unwrap());

        let weights_row = lines.find(|s| s.starts_with('w'))
            .ok_or_else(|| anyhow!("could not find the weights row"))?;

        let weights: ArrayVec<NonZeroU16, MAX_VARIABLES> = weights_row
            .split_whitespace()
            .skip(1)
            .flat_map(|w| w /* will fail for w == 0 */.parse().into_iter()).collect();

        let mut lines = lines.filter(|l| !l.starts_with('c'));
        let clauses = parse_clauses(&mut lines)?;
        let total_weight = weights.iter().map(|x| x.get() as u32).sum();
        Ok(Instance { id, weights, clauses, total_weight })
    }).collect()
}

fn params_from_filename(filename: &str) -> Result<InstanceParams> {
    let mut params = filename[3..].split('-').take(2).map(|n| n.parse::<u16>());
    let variables = params.next().unwrap()? as u8;
    let clauses = params.next().unwrap()?;
    Ok(InstanceParams { variables, clauses })
}

pub fn load_solutions(set: char) -> Result<HashMap<(InstanceParams, i32), OptimalSolution>> {
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
        let params = params_from_filename(&filename)?;

        let mut stream = BufReader::new(File::open(file.path())?);
        while let Some(opt) = parse_solution_line(&mut stream, params)? {
            let prev = solutions.insert((params, opt.id), opt.clone());
            if prev.is_some() {
                eprintln!(
                    "WARN: solution to ({:?}, {}), full ID: {}, is not unique",
                    params,
                    opt.id,
                    opt.full_id,
                );
            }
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
    pub fn evolutionary<Rng: rand::Rng + Send + Sync + Clone>(
        &self,
        rng: &mut Rng,
        mut ecfg: EvolutionaryConfig,
        opt: Option<&OptimalSolution>
    ) -> Solution {
        use rayon::prelude::*;
        use std::iter::repeat_with;

        impl<'a> Solution<'a> {
            fn crossover<Rng: rand::Rng>(
                self, other: Self, evo_config: &EvolutionaryConfig, rng: &mut Rng
            ) -> [Solution<'a>; 2] {
                let mut cfgs = [Config::zeroed(), Config::zeroed()];
                let mut weights = [0, 0];
                for (i, (l, r)) in self
                    .cfg.iter()
                    .zip(other.cfg.iter())
                    .take(self.inst.weights.len())
                    .enumerate() {
                    let bits = if rng.gen_bool(0.5) { (*l, *r) } else { (*r, *l) };
                    cfgs[0].set(i, bits.0);
                    cfgs[1].set(i, bits.1);
                    let w = self.inst.weights[i].get() as u32;
                    weights[0] += w * bits.0 as u32;
                    weights[1] += w * bits.1 as u32;
                }

                cfgs.into_iter()
                    .zip(weights.into_iter())
                    .map(|(cfg, weight)|
                        Solution { weight, cfg, inst: self.inst, satisfied: false, fitness: 0 }
                    )
                    .map(|sln| sln.mutate_unsafe(evo_config, rng))
                    .map(|sln| Solution { satisfied: satisfied(&sln.inst.clauses, &sln.cfg), ..sln })
                    .map(|sln| Solution { fitness: compute_fitness(&sln, evo_config), ..sln })
                    .collect::<ArrayVec<_, 2>>()
                    .into_inner()
                    .unwrap()
            }

            pub fn mutate_unsafe<Rng: rand::Rng>(
                &self, evo_config: &EvolutionaryConfig, rng: &mut Rng
            ) -> Solution<'a> {
                let mut new = *self;
                for i in 0..self.inst.weights.len() {
                    let flip = rng.gen_bool(evo_config.mutation_chance);
                    new.set_unsafe(i, if flip { !self.cfg[i] } else { self.cfg[i] });
                }

                new
            }
        }

        let random = {
            let ecfg = ecfg;
            move |rng: &mut Rng| {
                let mut cfg = Config::zeroed();
                let mut weight = 0;
                for i in 0..self.weights.len() {
                    let b = rng.gen_bool(0.5);
                    cfg.set(i, b);
                    weight += self.weights[i].get() as u32 * b as u32;
                }

                Solution::new(weight, cfg, self, &ecfg)
            }
        };

        fn stats(pop: &[Solution], _evo_config: EvolutionaryConfig, opt: Option<&OptimalSolution>) -> String {
            let identity = core::iter::repeat(0u16).take(MAX_VARIABLES)
                .collect::<ArrayVec<_, MAX_VARIABLES>>().into_inner().unwrap();
            let counts = pop.par_iter()
                .map(|sln| sln.cfg.iter()
                    .map(|b| *b as u16)
                    .collect::<ArrayVec<_, MAX_VARIABLES>>().into_inner().unwrap()
                )
                .reduce(|| identity, |l, r|
                    l.iter().zip(r.iter())
                    .map(|(x, y)| x + y)
                    .collect::<ArrayVec<_, MAX_VARIABLES>>().into_inner().unwrap()
                );

            let vars = pop[0].inst.weights.len();
            opt.and_then(|opt| {
                    pop.iter()
                        .filter(|sln| sln.satisfied)
                        .map(|sln| (1, sln.weight as f64 / opt.weight as f64))
                        .reduce(|acc, (n, w)| (acc.0 + n, acc.1 + w))
                        .map(|(count, sum)| 1.0 - sum / count as f64)
                }).or(Some(2.0)) // fill in the error if there's no known optimum
                .into_iter()
                .chain(counts.into_iter().take(vars).map(|x| x as f64 / pop.len() as f64))
                .map(|x| x.to_string())
                .intersperse(" ".into())
                .collect::<String>()
        }

        const DISASTER_INTERVAL: u32 = 100;
        const MUTATION_ADJUSTMENT_INTERVAL: u32 = 10;
        const MUTATION_ADJUSTMENT: f64 = 1.0001;

        let mut population = (0..ecfg.population_size).map(|_| random(rng)).collect::<Vec<_>>();
        let mut buffer = Vec::with_capacity(population.len() / 2);
        let mut shuffler: Vec<Solution> = buffer.clone();
        let mut best = population[0];
        println!("0 {}", stats(&population[..], ecfg, opt));

        (0..ecfg.generations).for_each(|i| {
            if i % DISASTER_INTERVAL == 0 {
                population.shuffle(rng);
                let n = (population.len() as f64 * 0.99) as usize;
                population.drain(.. n);
                population.extend(repeat_with(|| random(rng)).take(n));
            }

            population.par_sort_by_key(|sln| -(sln.fitness as i64));
            if (population[0].satisfied && !best.satisfied) || population[0].fitness > best.fitness {
                best = population[0];
            }

            shuffler.par_extend(population.par_iter());
            shuffler.shuffle(rng);
            // move unsatisfying solutions to the end
            shuffler.par_sort_by_key(|sln| sln.fitness == 0);

            // how many individuals to cross over
            let n = population.len() / 5;
            buffer.extend(shuffler.drain(..)
                .zip(population.drain(.. n * 2).take(n))
                .flat_map(|(a, b)| {
                    a.crossover(b, &ecfg, rng).into_iter()
                })
            );

            population.append(&mut buffer);
            #[allow(clippy::modulo_one)]
            if (i + 1) % (ecfg.generations / 100) == 0 {
                println!("{} {}", i + 1, stats(&population[..], ecfg, opt))
            }
            assert_eq!(population.len(), ecfg.population_size);

            // adjust the mutation modifier adaptively
            if i % MUTATION_ADJUSTMENT_INTERVAL == 0 {
                let n = population.iter()
                    .map(|sln| sln.fitness)
                    .filter(|f| *f == 0)
                    .count();
                let n = n as f64 / population.len() as f64;
                ecfg.mutation_chance = match () {
                    _ if n > 0.5 => (ecfg.mutation_chance * MUTATION_ADJUSTMENT).min(0.5),
                    _ if n < 0.2 => (ecfg.mutation_chance / MUTATION_ADJUSTMENT).max(0.0001),
                    _            => ecfg.mutation_chance
                };
            }
        });

        best
    }

    pub fn dump(&self) -> String {
        use core::iter::once;

        once("w".into())
        .chain(self.weights.iter()
            .map(|id| id.get())
            .chain(once(0))
            .map(|w| w.to_string())
        )
        .intersperse(" ".into())
        .chain(once("\n".into()))
        .chain(self.clauses.iter().flat_map(|clause|
            clause.iter().map(|&Literal(pos, id)|
                    id.get() as i16 * if pos { 1 } else { -1 }
                )
                .chain(once(0))
                .map(|l| l.to_string())
                .intersperse(" ".into())
                .chain(once("\n".into()))
        ))
        .collect()
    }
}

// ~\~ begin <<lit/main.md|tests>>[0]
#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};

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
            let weights: ArrayVec<NonZeroU16, MAX_VARIABLES> = ArrayVecProxy::arbitrary(g).into();
            let total_weight = weights.iter().map(|x| x.get() as u32).sum();
            Instance {
                id: i32::arbitrary(g),
                weights,
                total_weight,
                clauses: proxy.into_iter().map(|clause| clause.0.into_inner().unwrap()).collect(),
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
                            .map(|c| ArrayVecProxy(c.into()))
                            .collect()
                    )
                    .shrink()
                    .map(|clauses| {
                        let av: ArrayVec<LiteralProxy, MAX_CLAUSES> = clauses.into();
                        Instance {
                            clauses: av.into_iter().map(|clause| clause.0.into_inner().unwrap()).collect(),
                            ..(&data).clone()
                        }
                    })
                    .filter(|i| !i.clauses.is_empty())
                )
                .collect();
            Box::new(chain.into_iter())
        }
    }
}

// ~\~ end
// ~\~ end
