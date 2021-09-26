
use std::{fmt::Debug, io::stdin, str::FromStr};

fn main() -> Result<(), std::io::Error> {
    loop {
        println!("{}", parse_line()?.solve());
    }
}

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

struct Instance {
    id: i32, m: u32, b: u32, items: Vec<(u32, u32)>
}

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

impl Instance {
    fn solve(&self) -> bool {
        fn alloc(y: usize, x: usize) -> Vec<Vec<u32>> {
            let mut dp = Vec::with_capacity(y);
            dp.resize_with(y, || Vec::with_capacity(x));
            for v in &mut dp {
                v.resize(x, 0);
            }
            dp
        }

        let (m, b, items) = (self.m, self.b, &self.items);
        let mut dp = alloc(items.len() + 1, m as usize + 1);

        for i in 1..=items.len() {
            let (weight, cost) = items[i - 1];
            for j in 0..=m as usize {
                dp[i][j] = if (j as u32) < items[i as usize - 1].0 {
                    dp[i - 1][j]
                } else {
                    use std::cmp::max;
                    let rem_weight = max(0, j as isize - weight as isize) as usize;
                    max(dp[i - 1][j], dp[i - 1][rem_weight] + cost)
                };
            }
        }

        *dp.last().unwrap().last().unwrap() >= b
    }
}
