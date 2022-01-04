#!/bin/zsh

set -euxo pipefail
# Taken from https://doc.rust-lang.org/rustc/profile-guided-optimization.html

# STEP 0: Make sure there is no left-over profiling data from previous runs
rm -rf /tmp/pgo-data

# STEP 1: Build the instrumented binaries
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
    cargo build --release --target=x86_64-unknown-linux-gnu

# STEP 2: Run the instrumented binaries with some typical data
./target/x86_64-unknown-linux-gnu/release/solver bb < ds/NR15_inst.dat
./target/x86_64-unknown-linux-gnu/release/solver bb < ds/NR20_inst.dat

# STEP 3: Merge the `.profraw` files into a `.profdata` file
~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata \
    merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# STEP 4: Use the `.profdata` file for guiding optimizations
RUSTFLAGS="\
-Cprofile-use=/tmp/pgo-data/merged.profdata \
-Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --target=x86_64-unknown-linux-gnu
