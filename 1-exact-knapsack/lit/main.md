---
title: 'NI-KOP -- úkol 1'
---

# Kombinatorická optimalizace: problém batohu
Hello world! This is a test of [Entangled](https://entangled.github.io/).

## Build instructions

``` {.zsh .eval .bootstrap-fold #build-instructions}
cd solver
cargo build --release
```

## Benchmarking

``` {.zsh .eval #benchmark}
cd solver
hyperfine --export-csv bench.csv --ignore-failure 'cargo run --release < ../data/decision/NR4_inst.dat'
```

Let's have a look at the logged data:
``` {.zsh .eval #analysis}
cat solver/bench.csv
```

## Code

``` {.rust #boilerplate .bootstrap-fold}
trait Boilerplate {
    fn parse_next<T: FromStr>(&mut self) -> T
    where <T as FromStr>::Err: Debug;
}

impl Boilerplate for std::str::SplitWhitespace<'_> {
    fn parse_next<T: FromStr>(&mut self) -> T
    where <T as FromStr>::Err: Debug {
        self.next().unwrap().parse().unwrap()
    }
}
```
foo?

``` {.rust #instance-definition}
struct Instance {
    id: i32, m: u32, b: u32, items: Vec<(u32, u32)>
}
```

``` {.rust #parser .bootstrap-fold}
fn parse_line() -> Result<Instance, std::io::Error> {
    let mut input = String::new();
    stdin().read_line(&mut input)?;

    let mut numbers = input.split_whitespace();
    let id: i32   = numbers.parse_next();
    let  n: usize = numbers.parse_next();
    let  m: u32   = numbers.parse_next();
    let  b: u32   = numbers.parse_next();

    let mut items: Vec<(u32, u32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let w = numbers.parse_next();
        let c = numbers.parse_next();
        items.push((w, c));
    }

    Ok(Instance {id, m, b, items})
}
```

``` {.rust file=solver/src/main.rs}

use std::{fmt::Debug, io::stdin, str::FromStr};

fn main() -> Result<(), std::io::Error> {
    loop {
        println!("{}", parse_line()?.solve_stupider());
    }
}

<<boilerplate>>
<<instance-definition>>
<<parser>>

impl Instance {
    <<solver-dp>>

    <<solver-bb>>

    <<solver-bf>>
}
```

## Solvers

### Dynamic programming

``` {.rust #solver-dp}
fn solve(&self) -> u32 {
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
                use std::cmp::max;
                let rem_weight = max(0, cap as isize - weight as isize) as usize;
                max(last[cap], last[rem_weight] + cost)
            };
        }
    }

    *next.last().unwrap() //>= b
}
```

### Branch & bound
``` {.rust #solver-bb}
// branch & bound
fn solve_stupid(&self) -> u32 {
    let (m, b, items) = (self.m, self.b, &self.items);
    let prices: Vec<u32> = items.iter().rev()
        .scan(0, |sum, (_w, c)| {
            *sum = *sum + c;
            Some(*sum)
        })
        .collect();
    fn go(items: &Vec<(u32, u32)>, best: u32, cap: u32, i: usize) -> u32 {
        use std::cmp::max;
        if i >= items.len() { return 0; }

        let (w, c) = items[i];
        let next = |best, cap| go(items, best, cap, i + 1);
        let include = || next(best, cap - w);
        let exclude = || next(best, cap);
        let current = if w <= cap {
            max(c + include(), exclude())
        } else {
            exclude()
        };
        max(current, best)
    }

    go(items, 0, m, 0)
}
```

### Brute force
``` {.rust #solver-bf}
fn solve_stupider(&self) -> u32 {
    let (m, b, items) = (self.m, self.b, &self.items);
    fn go(items: &Vec<(u32, u32)>, cap: u32, i: usize) -> u32 {
        use std::cmp::max;
        if i >= items.len() { return 0; }

        let (w, c) = items[i];
        let next = |cap| go(items, cap, i + 1);
        let include = || next(cap - w);
        let exclude = || next(cap);
        let current = if w <= cap {
            max(c + include(), exclude())
        } else {
            exclude()
        };
        current
    }

    go(items, m, 0)
}
```
