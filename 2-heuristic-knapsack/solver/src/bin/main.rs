// ~\~ language=Rust filename=solver/src/bin/main.rs
// ~\~ begin <<lit/main.md|solver/src/bin/main.rs>>[0]
extern crate solver;

use std::io::stdin;
use solver::*;
use anyhow::{Result, anyhow};

fn main() -> Result<()> {
    let algorithms = get_algorithms();
    let solutions = load_solutions()?;

    let alg = *{
        // ~\~ begin <<lit/main.md|select-algorithm>>[0]
        let args: Vec<String> = std::env::args().collect();
        if args.len() == 2 {
            let alg = &args[1][..];
            if let Some(f) = algorithms.get(alg) {
                Ok(f)
            } else {
                Err(anyhow!("\"{}\" is not a known algorithm", alg))
            }
        } else {
            println!(
                "Usage: {} <algorithm>\n\twhere <algorithm> is one of {}",
                args[0],
                algorithms.keys().map(ToString::to_string).collect::<Vec<_>>().join(", ")
            );
            Err(anyhow!("Expected 1 argument, got {}", args.len() - 1))
        }
        // ~\~ end
    }?;

    for (cost, error) in solve_stream(alg, solutions, &mut stdin().lock())? {
        println!("{} {}", cost, error);
    }
    Ok(())
}
// ~\~ end
