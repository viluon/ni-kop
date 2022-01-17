// ~\~ language=Rust filename=solver/src/bin/main.rs
// ~\~ begin <<lit/main.md|solver/src/bin/main.rs>>[0]
extern crate solver;

use std::mem::size_of;

use solver::*;
use anyhow::{Result, anyhow};

fn main() -> Result<()> {
    let set = 'M';
    let solutions = load_solutions(set)?;
    let rng: rand_chacha::ChaCha8Rng = rand::SeedableRng::seed_from_u64(42);

    println!(
        "info:\n\
        |   Id size: {}\n\
        |   Literal size: {}\n\
        |   Clause size: {}\n\
        |   Config size: {}\n\
        |   Solution size: {}\n\
        |   Instance size: {}\n\
        ",
        size_of::<Id>(),
        size_of::<Literal>(),
        size_of::<Clause>(),
        size_of::<Config>(),
        size_of::<Solution>(),
        size_of::<Instance>(),
    );

    for inst in load_instances(set)? {
        use std::time::Instant;

        let mut rng = rng.clone();
        let now = Instant::now();
        let sln = inst.evolutionary(&mut rng);
        println!("took {} ms", now.elapsed().as_millis());

        let optimal = &solutions.get(&inst.clone().into());
        let error = optimal.map(|opt| 1.0 - sln.weight as f64 / opt.weight as f64);
        println!("{} {} {}", sln.satisfied, sln.weight, error.map(|e| e.to_string()).unwrap_or_default());
    }
    Ok(())
}
// ~\~ end
