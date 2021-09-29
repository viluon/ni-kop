// ~\~ language=Rust filename=solver/src/main.rs
// ~\~ begin <<lit/main.md|solver/src/main.rs>>[0]

use std::{fmt::Debug, io::stdin, str::FromStr};

fn main() -> Result<(), std::io::Error> {
    loop {
        println!("{}", parse_line()?.solve_stupider());
    }
}

// ~\~ begin <<lit/main.md|boilerplate>>[0]
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
// ~\~ end
// ~\~ begin <<lit/main.md|instance-definition>>[0]
struct Instance {
    id: i32, m: u32, b: u32, items: Vec<(u32, u32)>
}
// ~\~ end
// ~\~ begin <<lit/main.md|parser>>[0]
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
// ~\~ end

impl Instance {
    // ~\~ begin <<lit/main.md|solver-dp>>[0]
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
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bb>>[0]
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
    // ~\~ end

    // ~\~ begin <<lit/main.md|solver-bf>>[0]
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
    // ~\~ end
}
// ~\~ end
