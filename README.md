# Stockcrab

Stockcrab is a Rust port of [Stockfish](https://github.com/official-stockfish/Stockfish), the
UCI chess engine. It follows a pinned Stockfish commit (`config/upstream.env`) and, with the
same network, searches the same tree: a single-threaded `bench` matches the C++ engine node for
node. The code keeps Stockfish's structure, with one Rust module per C++ source file and the same
function names, so that upstream changes can be ported directly.

Stockcrab is a derivative work of Stockfish and is distributed under the same license, the GNU
General Public License version 3 or later. All credit for the engine and its strength belongs to
the [Stockfish developers](AUTHORS).

The current reference is Stockfish `88337401b7a4d5724611cd1ac32bdacf9e778ebd` with
`nn-684d0e60e71d.nnue`. At Hash 16 and Threads 1, the depth-13 bench matches C++ at
1,128,273 nodes and all 639 search records. [Upstream sync notes](docs/upstream-sync.md)

## Quickstart

From a source checkout with Rust 1.98 or newer:

```bash
scripts/fetch-net.sh
make build-native
./target/release/stockcrab
```

Enter `uci`, `isready`, `position startpos`, and `go depth 8`, one command per line.
The engine responds with `uciok`, `readyok`, and then `bestmove`. Enter `quit` after
the `bestmove` line. A UCI chess GUI can use the built executable directly.
[CI binary downloads](docs/binaries.md) are also available.

## Building

Stockcrab needs Rust 1.98 or newer. The NNUE kernels are selected from the target CPU at compile
time, so build for the machine that will run the engine:

```bash
make build-native        # this CPU (RUSTFLAGS="-C target-cpu=native" cargo build --release)
make build-x86-64-v3     # any x86-64 CPU with AVX2 and BMI2
```

The binary is `target/release/stockcrab`. A plain `cargo build --release` targets baseline
x86-64 and uses the scalar kernels, which are about ten times slower; the engine prints a
warning when it detects this. On x86-64 the accumulator updates also use the widest instruction
set the CPU supports at run time.

On AArch64 the NEON kernels are always used. Apple Silicon also gets the dot-product (SDOT)
kernels by default; on other ARMv8.2+ CPUs enable them with `--features neon-dotprod`.

The UCI `compiler` command reports which kernels a binary uses.

## Network

The engine needs the Stockfish NNUE network named in `config/upstream.env` (about 132 MiB), which
is not stored in this repository. Download and verify it with:

```bash
scripts/fetch-net.sh
```

The file is looked up next to the executable and in the working directory; the `EvalFile` UCI
option selects another file.

## Usage

Stockcrab is a UCI engine for use with any UCI-compatible GUI. Like Stockfish it also accepts
commands from a terminal, including `bench`, `speedtest`, `eval`, `d`, `go perft`, `flip`,
`compiler` and `export_net`:

```bash
./target/release/stockcrab bench
```

The UCI options are those of Stockfish, including `Threads`, `Hash`, `MultiPV`, `Skill Level`,
`UCI_LimitStrength`, `UCI_Elo`, `SyzygyPath` and `NumaPolicy`.

## Strength and speed

Recorded at Stockfish `3e8fde75`: Stockcrab (`make build-native`) against C++ at the same commit
built with `make profile-build ARCH=x86-64-avx512icl` (GCC, PGO), on an AMD Ryzen 9 9950X3D. Games use the
UHO_Lichess_4852_v1 openings, each played with both colours, and fishtest's adjudication rules;
error bars are 95% intervals.

| | Result |
|---|---|
| 1 thread, 10+0.1, 16 MB hash, 878 games | +1.2 ± 10.4 Elo |
| 4 threads, 5+0.05, 64 MB hash, 480 games | +1.5 ± 14.5 Elo |
| Speed, 1 thread (`bench 16 1 18`) | 98.5% of the PGO build's nodes per second, 103.6% of a non-PGO build's |
| Speed, 16 threads (`bench 256 16 20`) | about 98% and 101%, within the run-to-run spread |

## Testing

```bash
scripts/fetch-net.sh
cargo test --release
```

Tests that need the network are skipped when it is missing (`STOCKCRAB_REQUIRE_NET=1` makes that
a failure). The Syzygy tests need the 3-5 man tablebases: `scripts/fetch-syzygy.sh` downloads them
(about 1 GB) and `SYZYGY_PATH` points the tests at them. `scripts/check-all.sh` type-checks every
supported target.

`scripts/parity.sh` compares perft, evaluation and search output line by line with a C++ build of
the pinned Stockfish commit; `scripts/fuzz_diff.py` and `scripts/syzygy_diff.sh` extend the
comparison to random positions and tablebase probing. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Platforms

* Linux x86-64: fully tested, including the parity checks against C++.
* AArch64: tested on Apple Silicon (macOS) and Linux, with and without the dot-product kernels.
* Windows x86-64: tested in CI; NUMA binding, large pages and the shared network memory are not
  implemented there.
* Not ported: the RISC-V vector, LoongArch and PowerPC kernels. Other targets use the scalar code.

The library API is not stable; the crate is meant to be used as an engine binary. As in
Stockfish, tablebase files are memory-mapped and must not be modified while the engine runs,
and engines of the same user share one copy of the network weights in memory.

## License

Stockcrab is free software under the [GNU General Public License version 3](LICENSE) or later.
If you distribute it, or a modified version, you must make the complete corresponding source code
available under the same license.

Copyright (C) 2004-2026 The Stockfish developers (see [AUTHORS](AUTHORS)). Stockfish is derived
from Glaurung 2.1.

## Acknowledgements

* The [Stockfish developers](AUTHORS), whose code, comments and tests this port translates, and
  the [Fishtest](https://tests.stockfishchess.org/) contributors who test and tune it (see
  [Top CPU Contributors.txt](Top%20CPU%20Contributors.txt)).
* Stockfish's networks are trained on
  [data provided by the Leela Chess Zero project](https://storage.lczero.org/files/training_data),
  made available under the [Open Database License](https://opendatacommons.org/licenses/odbl/odbl-10.txt).
* The pseudo-random number generator is Sebastiano Vigna's xorshift64*, and the Syzygy tablebase
  code follows Ronald de Man's format as implemented in Stockfish.
