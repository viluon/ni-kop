// ~\~ language=Rust filename=solver/benches/bench.rs
// ~\~ begin <<lit/main.md|solver/benches/bench.rs>>[0]
extern crate solver;

use solver::*;
use anyhow::{Result, anyhow};
use std::{collections::HashMap, fs::File, io::{BufReader, Write}, ops::Range, time::Duration};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn full(c: &mut Criterion) -> Result<()> {
    let algs = get_algorithms();
    let solutions = load_solutions()?;
    let ranges = HashMap::from([
        ("bb", 0..=20),
        ("dpw", 0..=20),
        ("dpc", 0..=20),
        ("fptas1", 0..=20),
        ("fptas2", 0..=20),
        ("greedy", 0..=20),
        ("redux", 0..=20),
    ]);

    let mut input: HashMap<u32, Vec<Instance>> = HashMap::new();
    let ns = [4, 10, 15, 20];
    for n in ns { input.insert(n, load_input(n .. n + 1)?); }

    for (name, alg) in algs.iter() {
        let mut group = c.benchmark_group(*name);
        group.sample_size(10).warm_up_time(Duration::from_millis(200));

        for n in ns {
            if !ranges.get(*name).filter(|r| r.contains(&n)).is_some() {
                continue;
            }

            let (max, avg) = measure(&mut group, *alg, &solutions, n, &input[&n]);
            let avg = avg / n as f64;

            let mut file = File::create(format!("../docs/measurements/{}_{}.txt", name, n))?;
            file.write_all(format!("max,avg\n{},{}", max, avg).as_bytes())?;
        }
        group.finish();
    }
    Ok(())
}

fn measure(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    alg: for<'a> fn(&'a Instance) -> Solution<'a>,
    solutions: &HashMap<(u32, i32), OptimalSolution>,
    n: u32,
    instances: &Vec<Instance>
) -> (f64, f64) {
    let mut stats = (0.0, 0.0);
    group.bench_with_input(
        BenchmarkId::from_parameter(n),
        instances,
        |b, ins| b.iter(
            || ins.iter().for_each(|inst| {
                let sln = alg(inst);
                let optimal = &solutions[&(n, inst.id)];
                let error = 1.0 - sln.cost as f64 / optimal.cost as f64;
                let (max, avg) = stats;
                stats = (if error > max { error } else { max }, avg + error);
            })
        )
    );

    stats
}

fn load_input(r: Range<u32>) -> Result<Vec<Instance>> {
    let mut instances = Vec::new();

    for file in list_input_files(r)? {
        let file = file?;
        let mut r = BufReader::new(File::open(file.path())?);
        while let Some(inst) = parse_line(&mut r)? {
            instances.push(inst);
        }
    }

    Ok(instances)
}

fn proxy(c: &mut Criterion) {
    full(c).unwrap()
}

criterion_group!(benches, proxy);
criterion_main!(benches);
// ~\~ end
