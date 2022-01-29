// ~\~ language=Rust filename=solver/src/bin/main.rs
// ~\~ begin <<lit/main.md|solver/src/bin/main.rs>>[0]
extern crate solver;

use std::mem::size_of;

use rayon::prelude::*;
use solver::*;
use anyhow::{Result, anyhow};

fn main() -> Result<()> {
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

    let evo_config: EvolutionaryConfig = serde_json::from_str(std::env::args()
        .collect::<Vec<_>>()
        .get(1)
        .ok_or_else(|| anyhow!("Expected the evolutionary configuration in JSON format as the first argument"))?)?;

    let solutions = load_solutions(evo_config)?;
    let rng: rand_chacha::ChaCha8Rng = rand::SeedableRng::seed_from_u64(42);

    let mut instances = load_instances(evo_config.set)?
        .into_par_iter()
        .map(|inst| (inst.clone().into(), inst))
        .filter(|(params, _)| params == &evo_config.instance_params)
        .collect::<Vec<(InstanceParams, _)>>();
    instances.par_sort_unstable_by(|(p1, i1), (p2, i2)|
        p1.cmp(p2).then(i1.id.cmp(&i2.id))
    );

    solutions.iter().for_each(|((params, _), opt)| {
        // some instances (e.g. 33 in M1) have been removed,
        // but their optimal solutions are still here
        if let Ok(inst_index) = instances.binary_search_by(
            |(p, inst)| p.cmp(params).then(inst.id.cmp(&opt.id))
        ) {
            let inst = &instances[inst_index].1;
            let sln = Solution::new(opt.weight, opt.cfg, inst, &evo_config);
            assert!(sln.valid(&evo_config),
                "optimal solution to instance {} is invalid (satisfied: {})\n{}",
                inst.id,
                sln.satisfied,
                opt.dump(),
            );
        }
    });

    if instances.is_empty() { eprintln!("WARN: No instances match the given parameters"); }
    if solutions.is_empty() { eprintln!("WARN: No solutions match the given parameters"); }

    instances.into_iter().take(evo_config.n_instances as usize).for_each(|(_params, inst)| {
        use std::time::Instant;

        // println!("solving {} ({:?} from set {})", inst.id, params, evo_config.set);

        let mut rng = rng.clone();
        let optimal = solutions.get(&(inst.clone().into(), inst.id));
        let now = Instant::now();
        let sln = inst.evolutionary(&mut rng, evo_config, optimal);
        let time = now.elapsed().as_millis();

        let error = optimal.map(|opt| 1.0 - sln.weight as f64 / opt.weight as f64);
        println!("done: {time} {id} {satisfied} {valid} {weight} {err}",
            time = time,
            id = inst.id,
            satisfied = sln.satisfied,
            valid = sln.valid(&evo_config),
            weight = sln.weight,
            err = error.unwrap_or(2.0)
        );
        assert!(!sln.satisfied || sln.valid(&evo_config),
            "the following satisfied solution isn't valid! Instance {}:\n{}",
            inst.id,
            sln.dump()
        );
        assert!(!sln.satisfied || error.is_none() || error.unwrap() >= 0.0,
            "the following satisfied solution has a negative error of {:?}!\n{}\nInstance {}:\n{}",
            error,
            sln.dump(),
            inst.id,
            inst.dump(),
        );

        // println!("ours:    {}", sln.dump());
        // println!("optimal: {}\n", optimal.map(|opt| opt.dump()).unwrap_or_else(|| "None".into()));
    });
    Ok(())
}
// ~\~ end
