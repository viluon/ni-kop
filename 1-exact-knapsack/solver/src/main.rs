// ~\~ language=Rust filename=solver/src/main.rs
// ~\~ begin <<lit/main.md|solver/src/main.rs>>[0]
use std::{io::stdin, str::FromStr, cmp, cmp::max};
use anyhow::{Context, Result, anyhow};
use bitvec::prelude::BitArr;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

// ~\~ begin <<lit/main.md|problem-instance-definition>>[0]
#[derive(Debug, PartialEq, Eq, Clone)]
struct Instance {
    id: i32, m: u32, b: u32, items: Vec<(u32, u32)>
}
// ~\~ end

fn main() -> Result<()> {
    let alg = {
        // ~\~ begin <<lit/main.md|select-algorithm>>[0]
        let args: Vec<String> = std::env::args().collect();
        if args.len() == 2 {
            let ok = |x: fn(&Instance) -> u32| Ok(x);
            match &args[1][..] {
                "bf"    => ok(Instance::brute_force),
                "bb"    => ok(Instance::branch_and_bound),
                "dp"    => ok(Instance::dynamic_programming),
                invalid => Err(anyhow!("\"{}\" is not a known algorithm", invalid)),
            }
        } else {
            println!(
                "Usage: {} <algorithm>, where <algorithm> is one of bf, bb, dp",
                args[0]
            );
            Err(anyhow!("Expected 1 argument, got {}", args.len() - 1))
        }
        // ~\~ end
    }?;

    loop {
        match parse_line(stdin().lock())? {
            Some(inst) => println!("{}", alg(&inst)),
            None => return Ok(())
        }
    }
}

// ~\~ begin <<lit/main.md|parser>>[0]
// ~\~ begin <<lit/main.md|boilerplate>>[0]
trait Boilerplate {
    fn parse_next<T: FromStr>(&mut self) -> Result<T>
      where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static;
}

impl Boilerplate for std::str::SplitWhitespace<'_> {
    fn parse_next<T: FromStr>(&mut self) -> Result<T>
      where <T as FromStr>::Err: std::error::Error + Send + Sync + 'static {
        let str = self.next().ok_or(anyhow!("unexpected end of input"))?;
        str.parse::<T>()
           .with_context(|| format!("cannot parse {}", str))
    }
}
// ~\~ end

fn parse_line<T>(mut stream: T) -> Result<Option<Instance>> where T: std::io::BufRead {
    let mut input = String::new();
    match stream.read_line(&mut input)? {
        0 => return Ok(None),
        _ => ()
    };

    let mut  numbers = input.split_whitespace();
    let id = numbers.parse_next()?;
    let  n = numbers.parse_next()?;
    let  m = numbers.parse_next()?;
    let  b = numbers.parse_next()?;

    let mut items: Vec<(u32, u32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let w = numbers.parse_next()?;
        let c = numbers.parse_next()?;
        items.push((w, c));
    }

    Ok(Some(Instance {id, m, b, items}))
}
// ~\~ end

type Config = BitArr!(for 64);
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct Solution<'a> { weight: u32, cost: u32, cfg: Config, inst: &'a Instance }

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
        self.partial_cmp(&other).unwrap()
    }
}

impl Instance {
    // ~\~ begin <<lit/main.md|solver-dp>>[0]
    fn dynamic_programming(&self) -> u32 {
        let (m, b, items) = (self.m, self.b, &self.items);
        let mut next = Vec::with_capacity(m as usize + 1);
        next.resize(m as usize + 1, 0);
        let mut last = Vec::new();

        for i in 1..=items.len() {
            let (weight, cost) = items[i - 1];
            last.clone_from(&next);

            for cap in 0..=m as usize {
                next[cap] = if (cap as u32) < weight {
                    last[cap]
                } else {
                    let rem_weight = max(0, cap as isize - weight as isize) as usize;
                    max(last[cap], last[rem_weight] + cost)
                };
            }
        }

        *next.last().unwrap() //>= b
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bb>>[0]
    fn branch_and_bound(&self) -> u32 {
        self.ill_fuckin_do_it_again().cost
    }

    fn branch_and_bound2(&self) -> Solution {
        // TODO: shouldn't we update the best solution before we enter the recursive case? I mean, we've just added sth!
        let Instance { m, b, items, .. } = self;
        let prices: Vec<u32> = items.iter().rev()
            .scan(0, |sum, (_w, c)| {
                *sum = *sum + c;
                Some(*sum)
            })
            .collect::<Vec<_>>().into_iter().rev().collect();

        struct State<'a>(&'a Vec<(u32, u32)>, Vec<u32>, &'a Instance);
        // invariant: the recursion depth corresponds precisely to the index of
        // the item being considered for inclusion in the solution.
        fn go<'a>(state: &'a State, best: Solution<'a>, cap: u32, i: usize) -> Solution<'a> {
            let State(items, prices, inst) = state;
            let indent = " ".repeat(i * 2);
            let bitvec: Vec<bool> = best.cfg.to_bitvec().into_iter().take(3).collect();
            let default: Solution = Solution { weight: 0, cost: 0, cfg: Default::default(), inst };
            println!("{}at i = {} with best = Sol({}, {}, {:?}) cap = {}", indent, i, best.weight, best.cost, bitvec, cap);
            if i >= items.len() || best.cost > prices[i] { return default }

            let (w, c) = items[i];
            let next = |best, cap| {
                let x = go(state, best, cap, i + 1);
                println!("{}out of i = {}", indent, i + 1);
                x
            };
            let include = || {
                println!("{}include?", indent);
                let s = next(best, cap - w);
                let Solution { weight: sub_w, cost: sub_c, mut cfg, inst } = s;
                if cfg[i] || sub_w > cap - w { s }
                else {
                    cfg.set(i, true);
                    Solution { weight: sub_w + w, cost: c + sub_c, cfg, inst }
                }
            };
            let exclude = |best| {
                println!("{}exclude?", indent);
                let s = next(best, cap);
                let Solution { weight: sub_w, cost: sub_c, mut cfg, inst } = s;
                if cfg[i] {
                    cfg.set(i, false);
                    Solution { weight: sub_w - w, cost: sub_c - c, cfg, inst }
                } else { s }
            };
            if w <= cap {
                println!("{}fits", indent);
                let new_best = max(include(), best);
                max(exclude(new_best), new_best)
            } else {
                println!("{}doesn't fit", indent);
                exclude(best)
            }
        }

        let state = State(items, prices, self);
        let solution = go( &state
          , Solution { weight: 0, cost: 0, cfg: Default::default(), inst: self }
          , *m
          , 0
        );
        Solution { inst: self, ..solution }
    }

    fn ill_fuckin_do_it_again(&self) -> Solution {
        struct State<'a>(&'a Vec<(u32, u32)>, Vec<u32>);
        let Instance { m, items, .. } = self;
        let prices: Vec<u32> = {
            items.iter().rev()
            .scan(0, |sum, (_w, c)| {
                *sum = *sum + c;
                Some(*sum)
            })
            .collect::<Vec<_>>().into_iter().rev().collect()
        };

        fn default<'a>(inst: &'a Instance) -> Solution<'a> {
            Solution { weight: 0, cost: 0, cfg: Default::default(), inst }
        }

        impl <'a> Solution<'a> {
            // TODO what about capacity checking?
            fn with(mut self, i: usize) -> Solution<'a> {
                let (w, c) = self.inst.items[i];
                if !self.cfg[i] {
                    self.cfg.set(i, true);
                    self.weight += w;
                    self.cost += c;
                }
                self
            }
        }

        #[inline]
        fn smart_max<A, F, G>(f: F, g: G) -> A
        where F: Fn()  -> A
            , G: Fn(A) -> A
            , A: cmp::Ord + Copy {
            let x = f();
            max(x, g(x))
        }

        fn go<'a>(state: &'a State, current: Solution<'a>, best: Solution<'a>, i: usize, m: u32) -> Solution<'a> {
            let State(items, prices) = state;
            if i >= items.len() || current.cost + prices[i] <= best.cost { return current }

            let (w, _c) = items[i];
            let next = |current, best, m| go(state, current, best, i + 1, m);
            let include = || {
                let current = current.clone().with(i);
                next(current, max(current, best), m - w)
            };
            let exclude = |best| {
                next(current, best, m)
            };

            if w <= m { smart_max(include, exclude) }
            else { exclude(best) }
        }

        // FIXME what the hell is this? ಠ_ಠ
        let state = State(items, prices);
        let x = go(&state, default(self), default(self), 0, *m);
        Solution {inst: self, ..x}
    }
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bf>>[0]
    fn brute_force(&self) -> u32 {
        let (m, b, items) = (self.m, self.b, &self.items);
        fn go(items: &Vec<(u32, u32)>, cap: u32, i: usize) -> u32 {
            if i >= items.len() { return 0; }

            let (w, c) = items[i];
            let next = |cap| go(items, cap, i + 1);
            let include = || next(cap - w);
            let exclude = || next(cap);
            if w <= cap {
                max(c + include(), exclude())
            } else {
                exclude()
            }
        }

        go(items, m, 0)
    }
    // ~\~ end
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for Instance {
        fn arbitrary(g: &mut Gen) -> Instance {
            Instance {
                id:    i32::arbitrary(g),
                m:     u32::arbitrary(g),
                b:     u32::arbitrary(g),
                items: Vec::arbitrary(g)
                           .into_iter()
                           .take(10)
                           .map(|(w, c): (u32, u32)| (w.min(10_000), c.min(10_000)))
                           .collect(),
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let data = self.clone();
            let chain: Vec<Instance> = quickcheck::empty_shrinker()
                .chain(self.id   .shrink().map(|id   | Instance {id,    ..(&data).clone()}))
                .chain(self.m    .shrink().map(|m    | Instance {m,     ..(&data).clone()}))
                .chain(self.b    .shrink().map(|b    | Instance {b,     ..(&data).clone()}))
                .chain(self.items.shrink().map(|items| Instance {items, ..data}))
                .collect();
            Box::new(chain.into_iter())
        }
    }

    impl <'a> Solution<'a> {
        fn assert_valid(&self, i: &Instance) {
            let Instance { m, b, items, .. } = i;
            let Solution { weight: w, cost: c, cfg, inst: _ } = self;

            println!("{} >= {}", c, b);
            // assert!(c >= b);

            let (weight, cost) = items
                .into_iter()
                .zip(cfg)
                .map(|((w, c), b)| {
                    if *b { (*w, *c) } else { (0, 0) }
                })
                .reduce(|(a0, b0), (a1, b1)| (a0 + a1, b0 + b1))
                .unwrap_or_default();

            println!("{} <= {}", weight, *m);
            assert!(weight <= *m);

            println!("{} == {}", cost, *c);
            assert_eq!(cost, *c);

            println!("{} == {}", weight, *w);
            assert_eq!(weight, *w);
        }
    }

    #[test]
    fn stupid() {
        // let i = Instance { id: 0, m: 1, b: 0, items: vec![(1, 0), (1, 0)] };
        // i.branch_and_bound2().assert_valid(&i);
        let i = Instance { id: 0, m: 1, b: 0, items: vec![(1, 1), (1, 2), (0, 1)] };
        assert_eq!(i.branch_and_bound(), i.brute_force())
    }

    #[test]
    fn small_bb_is_correct() {
        let a = Instance {
            id: -10,
            m: 165,
            b: 384,
            items: vec![ (86,  744)
                       , (214, 1373)
                       , (236, 1571)
                       , (239, 2388)
                       ],
        };
        a.branch_and_bound2().assert_valid(&a);
    }

    #[test]
    fn bb_is_correct() -> Result<()> {
        use std::fs::File;
        use std::io::BufReader;
        let inst = parse_line(
            BufReader::new(File::open("ds/NR15_inst.dat")?)
        )?.unwrap();
        println!("testing {:?}", inst);
        inst.branch_and_bound2().assert_valid(&inst);
        Ok(())
    }

    #[quickcheck]
    fn qc_bb_is_really_correct(inst: Instance) {
        assert_eq!(inst.ill_fuckin_do_it_again().cost, inst.brute_force());
    }
}

// ~\~ end
