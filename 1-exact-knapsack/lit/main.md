---
title: 'NI-KOP -- úkol 1'
---

# Kombinatorická optimalizace: problém batohu

První úkol předmětu NI-KOP jsem se rozhodl implementovat v jazyce Rust za pomoci
nástrojů na *literate programming* -- přístup k psaní zdrojového kódu, který
upřednostňuje lidsky čitelný popis před seznamem příkazů pro počítač. Tento
soubor obsahuje veškerý zdrojový kód nutný k reprodukci mojí práce.

## Instrukce k sestavení programu
Program využívá standardních nástrojů jazyka Rust. O sestavení stačí požádat
`cargo`.

``` {.zsh .eval .bootstrap-fold #build-instructions}
cd solver
cargo build --release --color always
```

## Benchmarking

Pro provedení měření výkonu programu jsem využil nástroje Hyperfine.

``` {.zsh .eval #benchmark .bootstrap-fold}
uname -a
cd solver
hyperfine --export-json ../docs/bench.json \
          --parameter-list n 4,10,15,20 \
          --parameter-list alg bf,bb,dp \
          --runs 2 \
          --style color \
          'cargo run --release -- {alg} < ds/NR{n}_inst.dat' 2>&1 \
    | fold -w 120 -s
```

Měření ze spuštění Hyperfine jsou uložena v souboru `docs/bench.json`, který
následně zpracujeme do tabulky níže.

``` {.zsh .eval #json-to-csv .bootstrap-fold}
echo "algoritmus,\$n\$,průměr,\$\pm \sigma\$,minimum,medián,maximum" > docs/bench.csv
jq -r \
   '.[] | .[] | [.parameters.alg, .parameters.n
                , ([.mean, .stddev, .min, .median, .max]
                   | map("**" + (100000 * . + 0.5
                         | floor
                         | . / 100
                         | tostring
                         | if test("\\.") then sub("\\."; "**.") else . + "**" end
                         ) + " ms"
                        )
                  )
                ] | flatten | @csv' \
   docs/bench.json \
>> docs/bench.csv
echo ""
```

![Měření výkonu pro různé kombinace velikosti instancí problému ($n$) a
zvoleného algoritmu. y](docs/bench.csv)

### Srovnání algoritmů

```{.python .eval file=analysis/chart.py}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.tools.numeric import to_numeric

bench = pd.read_csv("docs/bench.csv", dtype = "string")
bench.rename({
        "algoritmus": "alg",
        "$n$": "n",
        "průměr": "avg",
        "$\pm \sigma$": "sigma",
        "medián": "median",
        "minimum": "min",
        "maximum": "max",
    },
    inplace = True,
    errors  = "raise",
    axis    = 1,
)

numeric_columns = ["n", "avg", "sigma", "min", "median", "max"]
bench[numeric_columns] = bench[numeric_columns].apply(lambda c: c.apply(lambda x: to_numeric(x.replace("**", "").replace(" ms", ""))))
# plt.figure()
# bench.groupby("alg").plot("n", "avg", kind = "line", ax = plt.gca())

df = bench

# Create a figure and a set of subplots.
fig, ax = plt.subplots()

# Group the dataframe by alg and create a line for each group.
for name, group in df.groupby("alg"):
    ax.errorbar(group["n"], group["avg"], yerr = group["sigma"], label = name)

# Set the axis labels.

ax.set_xlabel("n")
ax.set_ylabel("time (ms)")
ax.set_yscale("log")

# Add a legend.
ax.legend(loc="upper left")


plt.savefig("docs/graph.svg")
```

![box plot](graph.svg)

## Implementace

Program začíná definicí datové struktury reprezentující instanci problému
batohu.

``` {.rust #problem-instance-definition}
struct Instance {
    id: i32, m: u32, b: u32, items: Vec<(u32, u32)>
}
```

``` {.rust file=solver/src/main.rs}
use std::{io::stdin, str::FromStr};
use anyhow::{Context, Result, anyhow};

<<problem-instance-definition>>

fn main() -> Result<()> {
    let alg = {
        <<select-algorithm>>
    };

    loop {
        match parse_line()? {
            Some(inst) => println!("{}", alg(&inst)),
            None => return Ok(())
        }
    }
}

<<parser>>

impl Instance {
    <<solver-dp>>

    <<solver-bb>>

    <<solver-bf>>
}
```

## Solvers

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

## Appendix

Zpracování vstupu neošetřuje chyby ve vstupním formátu,

``` {.rust #parser .bootstrap-fold}
<<boilerplate>>

fn parse_line() -> Result<Option<Instance>> {
    let mut input = String::new();
    match stdin().read_line(&mut input)? {
        0 => return Ok(None),
        _ => ()
    };

    let mut numbers = input.split_whitespace();
    let id: i32   = numbers.parse_next()?;
    let  n: usize = numbers.parse_next()?;
    let  m: u32   = numbers.parse_next()?;
    let  b: u32   = numbers.parse_next()?;

    let mut items: Vec<(u32, u32)> = Vec::with_capacity(n);
    for _ in 0..n {
        let w = numbers.parse_next()?;
        let c = numbers.parse_next()?;
        items.push((w, c));
    }

    Ok(Some(Instance {id, m, b, items}))
}
```

Výběr algoritmu je řízen argumentem předaným na příkazové řádce. Příslušnou
funkci vrátíme jako hodnotu tohoto bloku:

``` {.rust #select-algorithm .bootstrap-fold}
use std::env;
let args: Vec<String> = env::args().collect();
if args.len() != 2 {
    println!(
        "Usage: {} <algorithm>, where <algorithm> is one of bf, bb, dp",
        args[0]
    );
    return Err(anyhow!("Expected 1 argument, got {}", args.len() - 1));
}
match &args[1][..] {
    "bf"    => Instance::solve_stupider,
    "bb"    => Instance::solve_stupid,
    "dp"    => Instance::solve,
    invalid => panic!("\"{}\" is not a known algorithm", invalid),
}
```

Trait `Boilerplate` definuje funkci `parse_next` pro zkrácení zápisu zpracování
vstupu.

``` {.rust #boilerplate .bootstrap-fold}
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
```
