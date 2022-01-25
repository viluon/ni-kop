// ~\~ language=Rust filename=solver/src/bin/main.rs
// ~\~ begin <<lit/main.md|solver/src/bin/main.rs>>[0]
extern crate solver;

use std::mem::size_of;

use rayon::prelude::*;
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

    let mut instances = load_instances(set)?
        .into_par_iter()
        .map(|inst| (inst.clone().into(), inst))
        .collect::<Vec<(InstanceParams, _)>>();
    instances.par_sort_unstable_by(|(p1, i1), (p2, i2)|
        p1.cmp(p2).then(i1.id.cmp(&i2.id))
    );

    instances.into_iter().take(3).for_each(|(params, inst)| {
        use std::time::Instant;

        println!("solving {} ({:?} from set {})", inst.id, params, set);

        let mut rng = rng.clone();
        let now = Instant::now();
        let sln = inst.evolutionary(&mut rng, EvolutionaryConfig {
            mutation_chance: 0.02,
        });
        println!("took {} ms", now.elapsed().as_millis());

        let optimal = solutions.get(&(inst.clone().into(), inst.id));
        let error = optimal.map(|opt| 1.0 - sln.weight as f64 / opt.weight as f64);
        println!("{} {} {}", sln.satisfied, sln.weight, error.map(|e| e.to_string()).unwrap_or_default());
        println!("valid? {}", sln.valid());

        println!("ours:    {}", sln.dump());
        optimal.into_iter().for_each(|opt| println!("optimal: {}\n", opt.dump()));
    });
    Ok(())
}
// ~\~ end
