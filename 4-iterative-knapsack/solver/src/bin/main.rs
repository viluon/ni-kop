// ~\~ language=Rust filename=solver/src/bin/main.rs
// ~\~ begin <<lit/main.md|solver/src/bin/main.rs>>[0]
extern crate solver;

use std::io::stdin;
use solver::*;
use anyhow::{Result, anyhow};

fn main() -> Result<()> {
    let algorithms = get_algorithms();
    let solutions = load_solutions("NK")?;

    enum Either<A, B> { Left(A), Right(B) }
    use  Either::*;

    let alg = {
        // ~\~ begin <<lit/main.md|select-algorithm>>[0]
        let args: Vec<String> = std::env::args().collect();
        if args.len() >= 2 {
            let alg = &args[1][..];
            if let Some(&f) = algorithms.get(alg) {
                Ok(Right(f))
            } else if alg == "sa" { #[allow(clippy::or_fun_call)] { // simulated annealing
                let mut iter = args[2..].iter().map(|str| &str[..]);
                let max_iterations = iter.next().ok_or(anyhow!("not enough params"))?.parse()?;
                let scaling_factor = iter.next().ok_or(anyhow!("not enough params"))?.parse()?;
                let temp_modifier = iter.next().ok_or(anyhow!("not enough params"))?.parse()?;
                let equilibrium_width = iter.next().ok_or(anyhow!("not enough params"))?.parse()?;
                Ok(Left((max_iterations, scaling_factor, temp_modifier, equilibrium_width)))
            } } else {
                Err(anyhow!("\"{}\" is not a known algorithm", alg))
            }
        } else {
            println!(
                "Usage: {} <algorithm>\n\twhere <algorithm> is one of {}\n\tor 'sa' for simulated annealing.",
                args[0],
                algorithms.keys().map(ToString::to_string).collect::<Vec<_>>().join(", ")
            );
            Err(anyhow!("Expected 1 argument, got {}", args.len() - 1))
        }
        // ~\~ end
    }?;

    for inst in load_instances(&mut stdin().lock())? {
        use std::time::Instant;
        let (now, sln) = match alg {
            Right(f) => (Instant::now(), f(&inst)),
            Left(cfg) => {
                let mut rng: rand_chacha::ChaCha8Rng = rand::SeedableRng::seed_from_u64(42);
                (Instant::now(), inst.simulated_annealing(&mut rng, cfg))
            },
        };
        println!("took {} ms", now.elapsed().as_millis());
        let optimal = &solutions.get(&(inst.items.len() as u32, inst.id));
        let error = optimal.map(|opt| 1.0 - sln.cost as f64 / opt.cost as f64);
        println!("{} {}", sln.cost, error.map(|e| e.to_string()).unwrap_or_default());
    }
    Ok(())
}
// ~\~ end
