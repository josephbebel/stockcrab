// Stockcrab, a Rust port of Stockfish, a UCI chess playing engine derived from Glaurung 2.1
// Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
//
// Stockcrab is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Stockcrab is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! Stockcrab: a Rust port of the Stockfish UCI chess engine.
//!
//! The crate is built to produce the `stockcrab` binary; the library exists so the
//! integration tests, benchmarks and tools can reach the engine's internals. **It is not
//! a stable API**: modules are public for those users and may change with any upstream
//! sync. Each module mirrors the like-named Stockfish source file (`search` <->
//! `search.cpp`, `nnue::network` <-> `nnue/network.cpp`, ...), which is what keeps new
//! upstream commits easy to port.
//!
//! The usual entry points are [`uci::UciEngine`] (the UCI command loop, as in `main`)
//! and [`engine::Engine`] (position, search and options without the text protocol).
//! Low-level [`position::Position`] move/null-move and undo operations are unsafe:
//! callers must supply legal moves (and not in check for a null move) and undo in reverse
//! order; [`position::Position::do_move_owned`] is the checked alternative. The states live
//! in a ring buffer owned by the position, so the search's per-ply path is allocation-free.
//! With panic unwinding enabled, worker-job panics propagate from completion waits
//! (or the next submission); workers are restored and peer searches stopped first.
//! Builds using `panic = "abort"`, including the default release binary, still abort.
//!
//! SIMD kernels mainly follow compile-time target features. Selected x86 stages
//! additionally dispatch to supported instructions at runtime; `compiler` reports
//! both policies. See the README for build flags and controlled ISA comparisons.

// Every AArch64 kernel is NEON and is selected by `target_arch`, so a target
// description without NEON cannot be built.
#[cfg(all(target_arch = "aarch64", not(target_feature = "neon")))]
compile_error!(
    "stockcrab requires NEON on AArch64; check -C target-cpu / target-feature \
     (an x86 CPU name was probably passed on an ARM host)"
);

pub mod benchmark;
#[path = "bitboard_api.rs"]
pub mod bitboard;
#[path = "bitboard.rs"]
pub(crate) mod bitboard_fast;
pub mod engine;
pub mod evaluate;
pub mod history;
pub mod misc;
pub mod movegen;
pub mod movepick;
pub mod nnue;
pub mod perft;
pub mod position;
pub mod score;
pub mod search;
pub mod shm;
#[cfg(unix)]
pub mod shm_unix;
pub mod syzygy;
pub mod thread;
pub mod timeman;
pub mod tt;
pub mod tune;
pub mod types;
pub mod uci;
pub mod ucioption;
