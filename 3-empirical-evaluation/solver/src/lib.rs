// ~\~ language=Rust filename=solver/src/lib.rs
// ~\~ begin <<lit/main.md|solver/src/lib.rs>>[0]
// ~\~ begin <<lit/main.md|imports>>[0]
use std::{cmp, cmp::max,
    ops::Range,
    str::FromStr,
    io::{BufRead, BufReader},
    collections::{BTreeMap, HashMap},
    fs::{read_dir, File, DirEntry},
};
use anyhow::{Context, Result, anyhow};
use bitvec::prelude::BitArr;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
// ~\~ end

// ~\~ begin <<lit/main.md|algorithm-map>>[0]
pub fn get_algorithms() -> BTreeMap<&'static str, fn(&Instance) -> Solution> {
    let cast = |x: fn(&Instance) -> Solution| x;
    // the BTreeMap works as a trie, maintaining alphabetic order
    BTreeMap::from([
        ("bf",     cast(Instance::brute_force)),
        ("bb",     cast(Instance::branch_and_bound)),
        ("dpc",    cast(Instance::dynamic_programming_c)),
        ("dpw",    cast(Instance::dynamic_programming_w)),
        ("fptas1", cast(|inst| inst.fptas(10f64.powi(-1)))),
        ("fptas2", cast(|inst| inst.fptas(10f64.powi(-2)))),
        ("greedy", cast(Instance::greedy)),
        ("redux",  cast(Instance::greedy_redux)),
    ])
}
// ~\~ end

pub fn solve_stream<T>(
    alg: for <'b> fn(&'b Instance) -> Solution<'b>,
    solutions: HashMap<(u32, i32), OptimalSolution>,
    stream: &mut T
) -> Result<Vec<(u32, Option<f64>)>> where T: BufRead {
    let mut results = vec![];
    loop {
        match parse_line(stream)?.as_ref().map(|inst| (inst, alg(inst))) {
            Some((inst, sln)) => {
                let optimal = &solutions.get(&(inst.items.len() as u32, inst.id));
                let error = optimal.map(|opt| 1.0 - sln.cost as f64 / opt.cost as f64);
                results.push((sln.cost, error))
            },
            None => return Ok(results)
        }
    }
}

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
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Instance {
    pub id: i32, m: u32, pub items: Vec<(u32, u32)>
}
// ~\~ end

// ~\~ begin <<lit/main.md|solution-definition>>[0]
pub type Config = BitArr!(for 64);

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Solution<'a> { weight: u32, pub cost: u32, cfg: Config, pub inst: &'a Instance }

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OptimalSolution { id: i32, pub cost: u32, cfg: Config }

// ~\~ begin <<lit/main.md|solution-helpers>>[0]
impl <'a> PartialOrd for Solution<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        let Solution {weight, cost, ..} = self;
        Some(match cost.cmp(&other.cost) {
            Ordering::Equal => weight.cmp(&other.weight).reverse(),
            other => other,
        })
    }
}

impl <'a> Ord for Solution<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl <'a> Solution<'a> {
    fn with(mut self, i: usize) -> Solution<'a> {
        let (w, c) = self.inst.items[i];
        if !self.cfg[i] {
            self.cfg.set(i, true);
            self.weight += w;
            self.cost += c;
        }
        self
    }

    fn default(inst: &'a Instance) -> Solution<'a> {
        Solution { weight: 0, cost: 0, cfg: Config::default(), inst }
    }

    fn overweight(inst: &'a Instance) -> Solution<'a> {
        Solution { weight: u32::MAX, cost: 0, cfg: Config::default(), inst }
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

pub fn parse_line<T>(stream: &mut T) -> Result<Option<Instance>> where T: BufRead {
    let mut input = String::new();
    if stream.read_line(&mut input)? == 0 {
        return Ok(None)
    }

    let mut  numbers = input.split_whitespace();
    let id = numbers.parse_next()?;
    let  n = numbers.parse_next()?;
    let  m = numbers.parse_next()?;

    let mut items: Vec<(u32, u32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let w = numbers.parse_next()?;
        let c = numbers.parse_next()?;
        items.push((w, c));
    }

    Ok(Some(Instance {id, m, items}))
}

fn parse_solution_line<T>(mut stream: T) -> Result<Option<OptimalSolution>> where T: BufRead {
    let mut input = String::new();
    if stream.read_line(&mut input)? == 0 {
        return Ok(None)
    }

    let mut    numbers = input.split_whitespace();
    let   id = numbers.parse_next()?;
    let    n = numbers.parse_next()?;
    let cost = numbers.parse_next()?;

    let mut items = Config::default();
    for i in 0..n {
        let a: u8 = numbers.parse_next()?;
        items.set(i, a == 1);
    }

    Ok(Some(OptimalSolution {id, cost, cfg: items}))
}

pub fn load_solutions(set: &str) -> Result<HashMap<(u32, i32), OptimalSolution>> {
    let mut solutions = HashMap::new();

    let files = read_dir("../data/constructive/")?
        .filter(|res| res.as_ref().ok().filter(|f| {
            let name = f.file_name().into_string().unwrap();
            f.file_type().unwrap().is_file() &&
            name.starts_with(set) &&
            name.ends_with("_sol.dat")
        }).is_some());

    for file in files {
        let file = file?;
        let n = file.file_name().into_string().unwrap()[set.len()..].split('_').next().unwrap().parse()?;
        let mut stream = BufReader::new(File::open(file.path())?);
        while let Some(opt) = parse_solution_line(&mut stream)? {
            solutions.insert((n, opt.id), opt);
        }
    }

    Ok(solutions)
}
// ~\~ end

impl Instance {
    // ~\~ begin <<lit/main.md|solver-dpw>>[0]
    fn dynamic_programming_w(&self) -> Solution {
        let Instance {m, items, ..} = self;
        let mut next = vec![Solution::default(self); *m as usize + 1];
        let mut last = vec![];

        for (i, &(weight, _cost)) in items.iter().enumerate() {
            last.clone_from(&next);

            for cap in 0 ..= *m as usize {
                let s = if (cap as u32) < weight {
                        last[cap]
                    } else {
                        let rem_weight = max(0, cap as isize - weight as isize) as usize;
                        max(last[cap], last[rem_weight].with(i))
                    };
                next[cap] = s;
            }
        }

        *next.last().unwrap()
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-dpc>>[0]
    fn dynamic_programming_c(&self) -> Solution {
        let Instance {items, ..} = self;
        let max_profit = items.iter().map(|(_, c)| *c).max().unwrap() as usize;
        let mut next = vec![Solution::overweight(self); max_profit * items.len() + 1];
        let mut last = vec![];
        next[0] = Solution::default(self);

        for (i, &(_weight, cost)) in items.iter().enumerate() {
            last.clone_from(&next);

            for cap in 1 ..= max_profit * items.len() {
                let s = if (cap as u32) < cost {
                        last[cap]
                    } else {
                        let rem_cost = (cap as isize - cost as isize) as usize;
                        let lightest_for_cost = if last[rem_cost].weight == u32::MAX {
                            last[0] // replace the overweight solution with the empty one
                        } else { last[rem_cost] };

                        max(last[cap], lightest_for_cost.with(i))
                    };
                next[cap] = s;
            }
        }

        *next.iter().filter(|sln| sln.weight <= self.m).last().unwrap()
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-fptas>>[0]
    // TODO: are items heavier than the knapsack capacity a problem? if so, we
    // can just zero them out
    fn fptas(&self, eps: f64) -> Solution {
        let Instance {m: _, items, ..} = self;
        let max_profit = items.iter().map(|(_, c)| *c).max().unwrap();
        let scaling_factor = eps * max_profit as f64 / items.len() as f64;
        let items: Vec<(u32, u32)> = items.iter().map(|(w, c)|
            (*w, (*c as f64 / scaling_factor).floor() as u32
        )).collect();

        let iso = Instance { items, ..*self };
        let sln = iso.dynamic_programming_c();
        let cost = (0usize..).zip(self.items.iter()).fold(0, |acc, (i, (_w, c))|
            acc + sln.cfg[i] as u32 * c
        );
        Solution { inst: self, cost, ..sln }
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-greedy>>[0]
    fn greedy(&self) -> Solution {
        use ::permutation::*;
        let Instance {m, items, ..} = self;
        fn ratio((w, c): (u32, u32)) -> f64 {
            let r = c as f64 / w as f64;
            if r.is_nan() { f64::NEG_INFINITY } else { r }
        }
        let permutation = sort_by(
            &(items)[..],
            |a, b|
                ratio(*a)
                .partial_cmp(&ratio(*b))
                .unwrap()
                .reverse() // max item first
        );
        let ord = { #[inline] |i| permutation.apply_idx(i) };

        let mut sol = Solution::default(self);
        for i in (0..items.len()).map(ord) {
            let (w, _c) = items[i];
            if sol.weight + w <= *m {
                sol = sol.with(i);
            } else { break }
        }

        sol
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-greedy-redux>>[0]
    fn greedy_redux(&self) -> Solution {
        let greedy = self.greedy();
        (0_usize..)
            .zip(self.items.iter())
            .filter(|(_, (w, _))| *w <= self.m)
            .max_by_key(|(_, (_, c))| c)
            .map(|(highest_price_index, _)|
                max(greedy, Solution::default(self).with(highest_price_index))
            ).unwrap_or(greedy)
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bb>>[0]
    fn branch_and_bound(&self) -> Solution {
        struct State<'a>(&'a Vec<(u32, u32)>, Vec<u32>);
        let prices: Vec<u32> = {
            self.items.iter().rev()
            .scan(0, |sum, (_w, c)| {
                *sum += c;
                Some(*sum)
            })
            .collect::<Vec<_>>().into_iter().rev().collect()
        };

        fn go<'a>(state: &'a State, current: Solution<'a>, best: Solution<'a>, i: usize, m: u32) -> Solution<'a> {
            let State(items, prices) = state;
            if i >= items.len() || current.cost + prices[i] <= best.cost {
                return current
            }

            let (w, _c) = items[i];
            let next = |current, best, m| go(state, current, best, i + 1, m);
            let include = || {
                let current = current.with(i);
                next(current, max(current, best), m - w)
            };
            let exclude = |best: Solution<'a>| next(current, best, m);

            if w <= m {
                let x = include();
                max(x, exclude(x))
            }
            else { exclude(best) }
        }

        // FIXME borrowck issues
        let state = State(&self.items, prices);
        let empty = Solution::default(self);
        Solution { inst: self, ..go(&state, empty, empty, 0, self.m) }
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bf>>[0]
    fn brute_force(&self) -> Solution {
        fn go<'a>(items: &'a [(u32, u32)], current: Solution<'a>, i: usize, m: u32) -> Solution<'a> {
            if i >= items.len() { return current }

            let (w, _c) = items[i];
            let next = |current, m| go(items, current, i + 1, m);
            let include = || {
                let current = current.with(i);
                next(current, m - w)
            };
            let exclude = || next(current, m);

            if w <= m {
                max(include(), exclude())
            }
            else { exclude() }
        }

        go(&self.items, Solution::default(self), 0, self.m)
    }
    // ~\~ end
}

// ~\~ begin <<lit/main.md|tests>>[0]
#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};
    use std::{fs::File, io::BufReader};

    impl Arbitrary for Instance {
        fn arbitrary(g: &mut Gen) -> Instance {
            Instance {
                id:    i32::arbitrary(g),
                m:     u32::arbitrary(g).min(10_000),
                items: vec![<(u32, u32)>::arbitrary(g)]
                           .into_iter()
                           .chain(Vec::arbitrary(g).into_iter())
                           .take(10)
                           .map(|(w, c): (u32, u32)| (w.min(10_000), c % 10_000))
                           .collect(),
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let data = self.clone();
            let chain: Vec<Instance> = quickcheck::empty_shrinker()
                .chain(self.id   .shrink().map(|id   | Instance {id,    ..(&data).clone()}))
                .chain(self.m    .shrink().map(|m    | Instance {m,     ..(&data).clone()}))
                .chain(self.items.shrink().map(|items| Instance { items, ..(&data).clone() })
                        .filter(|i| !i.items.is_empty()))
                .collect();
            Box::new(chain.into_iter())
        }
    }

    impl <'a> Solution<'a> {
        fn assert_valid(&self) {
            let Solution { weight, cost, cfg, inst } = self;
            let Instance { m, items, .. } = inst;

            let (computed_weight, computed_cost) = items
                .into_iter()
                .zip(cfg)
                .map(|((w, c), b)| {
                    if *b { (*w, *c) } else { (0, 0) }
                })
                .reduce(|(a0, b0), (a1, b1)| (a0 + a1, b0 + b1))
                .unwrap_or_default();

            assert!(computed_weight <= *m);
            assert_eq!(computed_cost, *cost);
            assert_eq!(computed_weight, *weight);
        }
    }

    #[test]
    fn stupid() {
        // let i = Instance { id: 0, m: 1, b: 0, items: vec![(1, 0), (1, 0)] };
        // i.branch_and_bound2().assert_valid(&i);
        let i = Instance { id: 0, m: 1, items: vec![(1, 1), (1, 2), (0, 1)] };
        let bb = i.branch_and_bound();
        assert_eq!(bb.cost, i.dynamic_programming_w().cost);
        assert_eq!(bb.cost, i.dynamic_programming_c().cost);
        assert_eq!(bb.cost, i.greedy_redux().cost);
        assert_eq!(bb.cost, i.brute_force().cost);
        assert_eq!(bb.cost, i.greedy().cost);
    }

    #[ignore]
    #[test]
    fn proper() -> Result<()> {
        type Solver = (&'static str, for<'a> fn(&'a Instance) -> Solution<'a>);
        let algs = get_algorithms();
        let algs: Vec<Solver> = algs.iter().map(|(s, f)| (*s, *f)).collect();
        let opts = load_solutions("NK")?;
        println!("loaded {} optimal solutions", opts.len());

        let solve: for<'a> fn(&Vec<_>, &'a _) -> Vec<(&'static str, Solution<'a>)> =
            |algs, inst|
            algs.iter().map(|(name, alg): &Solver| (*name, alg(inst))).collect();

        let mut files = list_input_files("NK", 0..5)?.into_iter();
        // make sure `files` is not empty
        let first = files.next().ok_or(anyhow!("no instance files loaded"))?;
        for file in vec![first].into_iter().chain(files) {
            let file = file?;
            println!("Testing {}", file.file_name().to_str().unwrap());
            // open the file
            let mut r = BufReader::new(File::open(file.path())?);
            // solve each instance with all algorithms
            while let Some(slns) = parse_line(&mut r)?.as_ref().map(|x| solve(&algs, x)) {
                // verify correctness
                slns.iter().for_each(|(alg, s)| {
                    eprint!("\rid: {} alg: {}\t", s.inst.id, alg);
                    s.assert_valid();
                    let key = (s.inst.items.len() as u32, s.inst.id);
                    assert!(s.cost <= opts[&key].cost);
                });
            }
        }
        Ok(())
    }

    #[test]
    fn dpc_simple() {
        let i = Instance { id: 0, m: 0, items: vec![(0, 1), (0, 1)] };
        let s = i.dynamic_programming_c();
        assert_eq!(s.cost, 2);
        assert_eq!(s.weight, 0);
        s.assert_valid();
    }

    #[test]
    fn fptas_is_within_bounds() -> Result<()> {
        let opts = load_solutions("NK")?;
        for eps in [0.1, 0.01] {
            for file in list_input_files("NK", 0..5)? {
                let file = file?;
                let mut r = BufReader::new(File::open(file.path())?);
                while let Some(sln) = parse_line(&mut r)?.as_ref().map(|x| x.fptas(eps)) {
                    // make sure the solution from fptas is at least (1 - eps) * optimal cost
                    let key = (sln.inst.items.len() as u32, sln.inst.id);
                    println!("{} {} {}", sln.cost, opts[&key].cost, (1.0 - eps) * opts[&key].cost as f64);
                    assert!(sln.cost as f64 >= opts[&key].cost as f64 * (1.0 - eps));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn small_bb_is_correct() {
        let a = Instance {
            id: -10,
            m: 165,
            items: vec![ (86,  744)
                       , (214, 1373)
                       , (236, 1571)
                       , (239, 2388)
                       ],
        };
        a.branch_and_bound().assert_valid();
    }

    #[test]
    fn bb_is_correct() -> Result<()> {
        use std::fs::File;
        use std::io::BufReader;
        let inst = parse_line(
            &mut BufReader::new(File::open("ds/NK15_inst.dat")?)
        )?.unwrap();
        println!("testing {:?}", inst);
        inst.branch_and_bound().assert_valid();
        Ok(())
    }

    #[quickcheck]
    fn qc_bb_is_really_correct(inst: Instance) {
        assert_eq!(inst.branch_and_bound().cost, inst.brute_force().cost);
    }

    #[quickcheck]
    fn qc_dp_matches_bb(inst: Instance) {
        assert!(inst.branch_and_bound().cost <= inst.dynamic_programming_w().cost);
    }

    #[quickcheck]
    fn qc_dps_match(inst: Instance) {
        assert_eq!(inst.dynamic_programming_w().cost, inst.dynamic_programming_c().cost);
    }

    #[quickcheck]
    fn qc_greedy_is_valid(inst: Instance) {
        inst.greedy().assert_valid();
        inst.greedy_redux().assert_valid();
    }
}

// ~\~ end
// ~\~ end
