# Contributing to Stockcrab

Stockcrab is a line-for-line port of Stockfish. Its goals, in order: performance parity with
C++ Stockfish; the same code structure as upstream (one module per upstream file, same function
names and order), so that upstream commits stay easy to port; then Rust idioms where they
conflict with neither. A restructuring that makes the next upstream port harder is usually not
wanted.

## Build, test, lint

```bash
scripts/fetch-net.sh                            # NNUE network, needed by most tests
make build-native
STOCKCRAB_REQUIRE_NET=1 cargo test --release
RUST_MIN_STACK=50000000 cargo test --lib        # debug-mode tests need a larger stack
cargo fmt --check
cargo clippy --all-targets -- -D warnings
scripts/check-all.sh                            # type-check every supported target
```

`scripts/check-all.sh` needs `rustup target add aarch64-unknown-linux-gnu aarch64-apple-darwin
x86_64-pc-windows-gnu`.

## Rules

* **SIMD paths are selected by `cfg(target_feature)` at compile time**, so code that builds on
  one machine can be broken on another target. Run `scripts/check-all.sh` after touching
  `src/nnue/`, `bitboard.rs`, `movegen.rs`, `movepick.rs`, `misc.rs` or `shm*.rs`.
* **Delete dead code only if every target agrees it is unused.** Unused-code warnings,
  `cargo fix` and `clippy --fix` only see the configuration being compiled.
* **Every `unsafe` block needs a `// SAFETY:` comment** stating the invariant that makes it
  sound (`clippy::undocumented_unsafe_blocks` is enabled).
* **Compiling is not evidence of correctness.** After a functional change, `bench` must still
  print the node count in `config/upstream.env` and `scripts/parity.sh` must report EXACT MATCH
  against the C++ engine (see below).
* **Golden test values** (NNUE evaluations, fixed-depth best moves, Syzygy search lines) change
  with the network and the search. Take them from the C++ engine, never from Stockcrab's output.
* Measure performance changes with repeated, interleaved runs of both binaries on an otherwise
  idle machine; a single run is noise.

## Parity testing

Build the C++ reference at the commit pinned in `config/upstream.env`, with the same network in
its `src/` directory:

```bash
git clone https://github.com/official-stockfish/Stockfish
git -C Stockfish checkout <STOCKFISH_REVISION>
cp nn-*.nnue Stockfish/src/
make -C Stockfish/src -j build ARCH=x86-64-avx2        # or native, apple-silicon, ...
STOCKFISH_DIR=$PWD/Stockfish scripts/parity.sh
```

`scripts/parity.sh` compares perft counts, FEN/key state, static evaluations and every `info`
line of the bench searches; it exits non-zero on any difference. From `Stockfish/src`,
`python3 <stockcrab>/scripts/fuzz_diff.py` compares searches on random positions (add
`--chess960` for Chess960), and `scripts/syzygy_diff.sh` compares tablebase probes.

## Porting upstream changes

1. Review the upstream commits since `STOCKFISH_REVISION` and port each applicable change to the
   like-named module, keeping upstream's structure, names and comments (the step headings in
   `search()` follow upstream's numbering).
2. Update `config/upstream.env` (revision, network and its SHA-256, bench node count).
3. Check that `bench` prints upstream's node count, run `scripts/parity.sh`, and re-derive any
   golden test values from the C++ engine.

## License

By contributing you agree that your contribution is licensed under GPL-3.0-or-later, like the
rest of the project.
