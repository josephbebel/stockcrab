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

//! Regression guards for parity-sensitive behaviour of the port.
//!
//! Each test pins behaviour that a green general-purpose test run does not
//! notice breaking. Expected values are taken from upstream C++ Stockfish:
//! FENs from its `d` command, NNUE evaluations from its `eval` command, and
//! constants from `src/types.h` / `src/search.cpp` / `src/benchmark.cpp`.
//! The tests do not need the C++ engine at run time, with one exception:
//! `bench_defaults_match_cpp_benchmark_cpp` reads the C++ `src/benchmark.cpp`
//! (looked up under `$STOCKFISH_DIR`, then `../Stockfish`, then `..`) and
//! skips if none exists, unless `STOCKCRAB_REQUIRE_CPP_SRC` is set (CI sets it
//! next to the pinned reference checkout), in which case it fails.
//!
//! For the full C++/Rust differential harness (perft, position state, eval,
//! search node-for-node), see `scripts/parity.sh`; this file is the cheap
//! always-on guard.
//!
//! Areas covered:
//!   1. A long move replay keeps castling rights, the ep square and rule50
//!      (the `StateInfo` ring buffer).
//!   2. En-passant legality tests the capturing side's king.
//!   3. NNUE evaluations (layer stack and output scaling) match C++ golden
//!      values.
//!   4. `Position::do_move` fills `DirtyThreats`, so incremental NNUE updates
//!      match a full refresh.
//!   5. UCI `setoption` has side effects (Hash / Threads / Clear Hash), and
//!      `NumaPolicy` replaces the NUMA config.
//!   6. `correctionHistory` / `pawnHistory` and the other histories are
//!      cleared to the C++ values.
//!   7. `make_key`'s LCG constants match C++ `types.h`.
//!   8. `VALUE_TB` and the classification thresholds are consistent.
//!   9. The bench position list matches `src/benchmark.cpp`.
//!   * An aborted MultiPV line never overtakes a proven loss.
//!
//! Run with:
//!   RUST_MIN_STACK=50000000 cargo test --release --test regression_parity

use std::sync::{Arc, RwLock};

use stockcrab::engine::{Engine, LazyNumaReplicated};
use stockcrab::history::SharedHistories;
use stockcrab::movegen::{self, GenType};
use stockcrab::nnue::network::Network;
use stockcrab::nnue::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};
use stockcrab::position::Position;
use stockcrab::search::Worker;
use stockcrab::thread::{
    NumaReplicatedAccessToken, RootMove, SearchControl, SearchManager, SharedState, ThreadPool,
    UpdateContext,
};
use stockcrab::tt::{Cluster, ThreadPool as TtThreadPool, TranspositionTable};
use stockcrab::types::{Dirties, Move, Square, VALUE_INFINITE, Value, mated_in};
use stockcrab::uci::UciEngine;
use stockcrab::ucioption::OptionsMap;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

mod common;

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE_FEN: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

/// The single network file that must be present in the crate root for the NNUE
/// tests to be meaningful.
const NET_FILE: &str = stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME;

fn init_tables() {
    stockcrab::bitboard::init();
    stockcrab::movegen::init();
}

/// Load the default network from the crate root. Returns `None` when the file is
/// absent so NNUE tests can skip instead of silently comparing zeros.
fn load_network() -> Option<Network> {
    if !std::path::Path::new(NET_FILE).exists() {
        return None;
    }
    let mut net = Network::new();
    // The file exists, so a load failure means a corrupt network: fail loudly
    // rather than skipping the NNUE tests.
    net.load(".", String::new())
        .unwrap_or_else(|e| panic!("failed to load {NET_FILE}: {e}"));
    Some(net)
}

/// A position replayed from a FEN and UCI moves.
struct Replay {
    pos: Position,
}

impl Replay {
    fn new(fen: &str) -> Replay {
        init_tables();
        let mut pos = Position::default();
        pos.set_fen(fen, false)
            .unwrap_or_else(|e| panic!("bad FEN {fen}: {e}"));
        Replay { pos }
    }

    fn push_uci(&mut self, uci: &str) {
        let mv = UciEngine::to_move(&self.pos, uci);
        assert_ne!(
            mv,
            Move::none(),
            "illegal/unparsable move {uci} in position {}",
            self.pos.fen()
        );
        // SAFETY: `to_move` only returns legal moves.
        unsafe { self.pos.do_move(mv) };
    }
}

/// Zobrist key computed from scratch for `fen` (the `set_fen` path), used as an
/// independent oracle for the incrementally maintained key.
fn key_from_scratch(fen: &str) -> u64 {
    let r = Replay::new(fen);
    r.pos.key()
}

fn castling_field(fen: &str) -> &str {
    fen.split_whitespace().nth(2).unwrap_or("?")
}

fn ep_field(fen: &str) -> &str {
    fen.split_whitespace().nth(3).unwrap_or("?")
}

// ---------------------------------------------------------------------------
// StateInfo history over a long replay
//
// A lost or stale `StateInfo` corrupts castling rights, the ep square and rule50
// from the 4th move of a game onward.
// ---------------------------------------------------------------------------

/// (uci move, FEN after that move) for a full 60-ply game, generated by upstream
/// C++ Stockfish (self-play at depth 6, FENs read back with `d`).
/// Both sides castle (plies 19/20), so a dropped castling right shows up as a
/// FEN mismatch long before the game ends.
const GAME_STARTPOS: &[(&str, &str)] = &[
    (
        "e2e4",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    ),
    (
        "c7c5",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ),
    (
        "g1f3",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    ),
    (
        "b8c6",
        "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ),
    (
        "b1c3",
        "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 3",
    ),
    (
        "g7g6",
        "r1bqkbnr/pp1ppp1p/2n3p1/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",
    ),
    (
        "h2h3",
        "r1bqkbnr/pp1ppp1p/2n3p1/2p5/4P3/2N2N1P/PPPP1PP1/R1BQKB1R b KQkq - 0 4",
    ),
    (
        "f8g7",
        "r1bqk1nr/pp1pppbp/2n3p1/2p5/4P3/2N2N1P/PPPP1PP1/R1BQKB1R w KQkq - 1 5",
    ),
    (
        "f1c4",
        "r1bqk1nr/pp1pppbp/2n3p1/2p5/2B1P3/2N2N1P/PPPP1PP1/R1BQK2R b KQkq - 2 5",
    ),
    (
        "e7e6",
        "r1bqk1nr/pp1p1pbp/2n1p1p1/2p5/2B1P3/2N2N1P/PPPP1PP1/R1BQK2R w KQkq - 0 6",
    ),
    (
        "d2d4",
        "r1bqk1nr/pp1p1pbp/2n1p1p1/2p5/2BPP3/2N2N1P/PPP2PP1/R1BQK2R b KQkq - 0 6",
    ),
    (
        "c5d4",
        "r1bqk1nr/pp1p1pbp/2n1p1p1/8/2BpP3/2N2N1P/PPP2PP1/R1BQK2R w KQkq - 0 7",
    ),
    (
        "c3b5",
        "r1bqk1nr/pp1p1pbp/2n1p1p1/1N6/2BpP3/5N1P/PPP2PP1/R1BQK2R b KQkq - 1 7",
    ),
    (
        "d7d5",
        "r1bqk1nr/pp3pbp/2n1p1p1/1N1p4/2BpP3/5N1P/PPP2PP1/R1BQK2R w KQkq - 0 8",
    ),
    (
        "e4d5",
        "r1bqk1nr/pp3pbp/2n1p1p1/1N1P4/2Bp4/5N1P/PPP2PP1/R1BQK2R b KQkq - 0 8",
    ),
    (
        "e6d5",
        "r1bqk1nr/pp3pbp/2n3p1/1N1p4/2Bp4/5N1P/PPP2PP1/R1BQK2R w KQkq - 0 9",
    ),
    (
        "c4e2",
        "r1bqk1nr/pp3pbp/2n3p1/1N1p4/3p4/5N1P/PPP1BPP1/R1BQK2R b KQkq - 1 9",
    ),
    (
        "g8f6",
        "r1bqk2r/pp3pbp/2n2np1/1N1p4/3p4/5N1P/PPP1BPP1/R1BQK2R w KQkq - 2 10",
    ),
    (
        "e1g1",
        "r1bqk2r/pp3pbp/2n2np1/1N1p4/3p4/5N1P/PPP1BPP1/R1BQ1RK1 b kq - 3 10",
    ),
    (
        "e8g8",
        "r1bq1rk1/pp3pbp/2n2np1/1N1p4/3p4/5N1P/PPP1BPP1/R1BQ1RK1 w - - 4 11",
    ),
    (
        "b5d4",
        "r1bq1rk1/pp3pbp/2n2np1/3p4/3N4/5N1P/PPP1BPP1/R1BQ1RK1 b - - 0 11",
    ),
    (
        "f8e8",
        "r1bqr1k1/pp3pbp/2n2np1/3p4/3N4/5N1P/PPP1BPP1/R1BQ1RK1 w - - 1 12",
    ),
    (
        "c2c3",
        "r1bqr1k1/pp3pbp/2n2np1/3p4/3N4/2P2N1P/PP2BPP1/R1BQ1RK1 b - - 0 12",
    ),
    (
        "a7a6",
        "r1bqr1k1/1p3pbp/p1n2np1/3p4/3N4/2P2N1P/PP2BPP1/R1BQ1RK1 w - - 0 13",
    ),
    (
        "c1e3",
        "r1bqr1k1/1p3pbp/p1n2np1/3p4/3N4/2P1BN1P/PP2BPP1/R2Q1RK1 b - - 1 13",
    ),
    (
        "c6a5",
        "r1bqr1k1/1p3pbp/p4np1/n2p4/3N4/2P1BN1P/PP2BPP1/R2Q1RK1 w - - 2 14",
    ),
    (
        "f1e1",
        "r1bqr1k1/1p3pbp/p4np1/n2p4/3N4/2P1BN1P/PP2BPP1/R2QR1K1 b - - 3 14",
    ),
    (
        "c8d7",
        "r2qr1k1/1p1b1pbp/p4np1/n2p4/3N4/2P1BN1P/PP2BPP1/R2QR1K1 w - - 4 15",
    ),
    (
        "f3d2",
        "r2qr1k1/1p1b1pbp/p4np1/n2p4/3N4/2P1B2P/PP1NBPP1/R2QR1K1 b - - 5 15",
    ),
    (
        "b7b5",
        "r2qr1k1/3b1pbp/p4np1/np1p4/3N4/2P1B2P/PP1NBPP1/R2QR1K1 w - - 0 16",
    ),
    (
        "d4c2",
        "r2qr1k1/3b1pbp/p4np1/np1p4/8/2P1B2P/PPNNBPP1/R2QR1K1 b - - 1 16",
    ),
    (
        "d8c7",
        "r3r1k1/2qb1pbp/p4np1/np1p4/8/2P1B2P/PPNNBPP1/R2QR1K1 w - - 2 17",
    ),
    (
        "c2b4",
        "r3r1k1/2qb1pbp/p4np1/np1p4/1N6/2P1B2P/PP1NBPP1/R2QR1K1 b - - 3 17",
    ),
    (
        "a5c6",
        "r3r1k1/2qb1pbp/p1n2np1/1p1p4/1N6/2P1B2P/PP1NBPP1/R2QR1K1 w - - 4 18",
    ),
    (
        "b4c6",
        "r3r1k1/2qb1pbp/p1N2np1/1p1p4/8/2P1B2P/PP1NBPP1/R2QR1K1 b - - 0 18",
    ),
    (
        "d7c6",
        "r3r1k1/2q2pbp/p1b2np1/1p1p4/8/2P1B2P/PP1NBPP1/R2QR1K1 w - - 0 19",
    ),
    (
        "d2b3",
        "r3r1k1/2q2pbp/p1b2np1/1p1p4/8/1NP1B2P/PP2BPP1/R2QR1K1 b - - 1 19",
    ),
    (
        "h7h5",
        "r3r1k1/2q2pb1/p1b2np1/1p1p3p/8/1NP1B2P/PP2BPP1/R2QR1K1 w - - 0 20",
    ),
    (
        "e3d4",
        "r3r1k1/2q2pb1/p1b2np1/1p1p3p/3B4/1NP4P/PP2BPP1/R2QR1K1 b - - 1 20",
    ),
    (
        "a8c8",
        "2r1r1k1/2q2pb1/p1b2np1/1p1p3p/3B4/1NP4P/PP2BPP1/R2QR1K1 w - - 2 21",
    ),
    (
        "e2f3",
        "2r1r1k1/2q2pb1/p1b2np1/1p1p3p/3B4/1NP2B1P/PP3PP1/R2QR1K1 b - - 3 21",
    ),
    (
        "a6a5",
        "2r1r1k1/2q2pb1/2b2np1/pp1p3p/3B4/1NP2B1P/PP3PP1/R2QR1K1 w - - 0 22",
    ),
    (
        "e1e8",
        "2r1R1k1/2q2pb1/2b2np1/pp1p3p/3B4/1NP2B1P/PP3PP1/R2Q2K1 b - - 0 22",
    ),
    (
        "c8e8",
        "4r1k1/2q2pb1/2b2np1/pp1p3p/3B4/1NP2B1P/PP3PP1/R2Q2K1 w - - 0 23",
    ),
    (
        "a2a3",
        "4r1k1/2q2pb1/2b2np1/pp1p3p/3B4/PNP2B1P/1P3PP1/R2Q2K1 b - - 0 23",
    ),
    (
        "h5h4",
        "4r1k1/2q2pb1/2b2np1/pp1p4/3B3p/PNP2B1P/1P3PP1/R2Q2K1 w - - 0 24",
    ),
    (
        "d4e3",
        "4r1k1/2q2pb1/2b2np1/pp1p4/7p/PNP1BB1P/1P3PP1/R2Q2K1 b - - 1 24",
    ),
    (
        "f6e4",
        "4r1k1/2q2pb1/2b3p1/pp1p4/4n2p/PNP1BB1P/1P3PP1/R2Q2K1 w - - 2 25",
    ),
    (
        "b3d4",
        "4r1k1/2q2pb1/2b3p1/pp1p4/3Nn2p/P1P1BB1P/1P3PP1/R2Q2K1 b - - 3 25",
    ),
    (
        "g7e5",
        "4r1k1/2q2p2/2b3p1/pp1pb3/3Nn2p/P1P1BB1P/1P3PP1/R2Q2K1 w - - 4 26",
    ),
    (
        "d4c6",
        "4r1k1/2q2p2/2N3p1/pp1pb3/4n2p/P1P1BB1P/1P3PP1/R2Q2K1 b - - 0 26",
    ),
    (
        "c7c6",
        "4r1k1/5p2/2q3p1/pp1pb3/4n2p/P1P1BB1P/1P3PP1/R2Q2K1 w - - 0 27",
    ),
    (
        "d1d3",
        "4r1k1/5p2/2q3p1/pp1pb3/4n2p/P1PQBB1P/1P3PP1/R5K1 b - - 1 27",
    ),
    (
        "e5c7",
        "4r1k1/2b2p2/2q3p1/pp1p4/4n2p/P1PQBB1P/1P3PP1/R5K1 w - - 2 28",
    ),
    (
        "a1d1",
        "4r1k1/2b2p2/2q3p1/pp1p4/4n2p/P1PQBB1P/1P3PP1/3R2K1 b - - 3 28",
    ),
    (
        "e8d8",
        "3r2k1/2b2p2/2q3p1/pp1p4/4n2p/P1PQBB1P/1P3PP1/3R2K1 w - - 4 29",
    ),
    (
        "g1f1",
        "3r2k1/2b2p2/2q3p1/pp1p4/4n2p/P1PQBB1P/1P3PP1/3R1K2 b - - 5 29",
    ),
    (
        "g8g7",
        "3r4/2b2pk1/2q3p1/pp1p4/4n2p/P1PQBB1P/1P3PP1/3R1K2 w - - 6 30",
    ),
    (
        "f3e2",
        "3r4/2b2pk1/2q3p1/pp1p4/4n2p/P1PQB2P/1P2BPP1/3R1K2 b - - 7 30",
    ),
    (
        "c6d6",
        "3r4/2b2pk1/3q2p1/pp1p4/4n2p/P1PQB2P/1P2BPP1/3R1K2 w - - 8 31",
    ),
];

/// Same idea from kiwipete: castling rights are lost one at a time
/// (KQkq -> Qkq -> Qq -> Q -> -) and a real castling move is played (e1c1),
/// so this sequence is a fine-grained probe for silently dropped rights.
const GAME_KIWIPETE: &[(&str, &str)] = &[
    (
        "e2a6",
        "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R3K2R b KQkq - 0 1",
    ),
    (
        "h3g2",
        "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 2",
    ),
    (
        "h1g1",
        "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K1R1 b Qkq - 1 2",
    ),
    (
        "b4c3",
        "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/4P3/2p2Q2/PPPB1PpP/R3K1R1 w Qkq - 0 3",
    ),
    (
        "d2c3",
        "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/4P3/2B2Q2/PPP2PpP/R3K1R1 b Qkq - 0 3",
    ),
    (
        "e6d5",
        "r3k2r/p1ppqpb1/Bn3np1/3pN3/4P3/2B2Q2/PPP2PpP/R3K1R1 w Qkq - 0 4",
    ),
    (
        "e5g4",
        "r3k2r/p1ppqpb1/Bn3np1/3p4/4P1N1/2B2Q2/PPP2PpP/R3K1R1 b Qkq - 1 4",
    ),
    (
        "f6g4",
        "r3k2r/p1ppqpb1/Bn4p1/3p4/4P1n1/2B2Q2/PPP2PpP/R3K1R1 w Qkq - 0 5",
    ),
    (
        "c3g7",
        "r3k2r/p1ppqpB1/Bn4p1/3p4/4P1n1/5Q2/PPP2PpP/R3K1R1 b Qkq - 0 5",
    ),
    (
        "h8h7",
        "r3k3/p1ppqpBr/Bn4p1/3p4/4P1n1/5Q2/PPP2PpP/R3K1R1 w Qq - 1 6",
    ),
    (
        "f3g2",
        "r3k3/p1ppqpBr/Bn4p1/3p4/4P1n1/8/PPP2PQP/R3K1R1 b Qq - 0 6",
    ),
    (
        "f7f5",
        "r3k3/p1ppq1Br/Bn4p1/3p1p2/4P1n1/8/PPP2PQP/R3K1R1 w Qq - 0 7",
    ),
    (
        "g7d4",
        "r3k3/p1ppq2r/Bn4p1/3p1p2/3BP1n1/8/PPP2PQP/R3K1R1 b Qq - 1 7",
    ),
    (
        "h7h2",
        "r3k3/p1ppq3/Bn4p1/3p1p2/3BP1n1/8/PPP2PQr/R3K1R1 w Qq - 0 8",
    ),
    (
        "g2f3",
        "r3k3/p1ppq3/Bn4p1/3p1p2/3BP1n1/5Q2/PPP2P1r/R3K1R1 b Qq - 1 8",
    ),
    (
        "e7e4",
        "r3k3/p1pp4/Bn4p1/3p1p2/3Bq1n1/5Q2/PPP2P1r/R3K1R1 w Qq - 0 9",
    ),
    (
        "f3e4",
        "r3k3/p1pp4/Bn4p1/3p1p2/3BQ1n1/8/PPP2P1r/R3K1R1 b Qq - 0 9",
    ),
    (
        "d5e4",
        "r3k3/p1pp4/Bn4p1/5p2/3Bp1n1/8/PPP2P1r/R3K1R1 w Qq - 0 10",
    ),
    (
        "a6e2",
        "r3k3/p1pp4/1n4p1/5p2/3Bp1n1/8/PPP1BP1r/R3K1R1 b Qq - 1 10",
    ),
    (
        "e8e7",
        "r7/p1ppk3/1n4p1/5p2/3Bp1n1/8/PPP1BP1r/R3K1R1 w Q - 2 11",
    ),
    (
        "e2g4",
        "r7/p1ppk3/1n4p1/5p2/3Bp1B1/8/PPP2P1r/R3K1R1 b Q - 0 11",
    ),
    (
        "f5g4",
        "r7/p1ppk3/1n4p1/8/3Bp1p1/8/PPP2P1r/R3K1R1 w Q - 0 12",
    ),
    (
        "d4e5",
        "r7/p1ppk3/1n4p1/4B3/4p1p1/8/PPP2P1r/R3K1R1 b Q - 1 12",
    ),
    (
        "h2h4",
        "r7/p1ppk3/1n4p1/4B3/4p1pr/8/PPP2P2/R3K1R1 w Q - 2 13",
    ),
    ("e5c7", "r7/p1Bpk3/1n4p1/8/4p1pr/8/PPP2P2/R3K1R1 b Q - 0 13"),
    (
        "d7d5",
        "r7/p1B1k3/1n4p1/3p4/4p1pr/8/PPP2P2/R3K1R1 w Q - 0 14",
    ),
    (
        "e1c1",
        "r7/p1B1k3/1n4p1/3p4/4p1pr/8/PPP2P2/2KR2R1 b - - 1 14",
    ),
    (
        "a8f8",
        "5r2/p1B1k3/1n4p1/3p4/4p1pr/8/PPP2P2/2KR2R1 w - - 2 15",
    ),
    (
        "d1d2",
        "5r2/p1B1k3/1n4p1/3p4/4p1pr/8/PPPR1P2/2K3R1 b - - 3 15",
    ),
    (
        "e7e6",
        "5r2/p1B5/1n2k1p1/3p4/4p1pr/8/PPPR1P2/2K3R1 w - - 4 16",
    ),
    (
        "b2b3",
        "5r2/p1B5/1n2k1p1/3p4/4p1pr/1P6/P1PR1P2/2K3R1 b - - 0 16",
    ),
    (
        "g6g5",
        "5r2/p1B5/1n2k3/3p2p1/4p1pr/1P6/P1PR1P2/2K3R1 w - - 0 17",
    ),
    (
        "c7g3",
        "5r2/p7/1n2k3/3p2p1/4p1pr/1P4B1/P1PR1P2/2K3R1 b - - 1 17",
    ),
    (
        "h4h8",
        "5r1r/p7/1n2k3/3p2p1/4p1p1/1P4B1/P1PR1P2/2K3R1 w - - 2 18",
    ),
    (
        "g1e1",
        "5r1r/p7/1n2k3/3p2p1/4p1p1/1P4B1/P1PR1P2/2K1R3 b - - 3 18",
    ),
    (
        "f8f3",
        "7r/p7/1n2k3/3p2p1/4p1p1/1P3rB1/P1PR1P2/2K1R3 w - - 4 19",
    ),
    (
        "c2c4",
        "7r/p7/1n2k3/3p2p1/2P1p1p1/1P3rB1/P2R1P2/2K1R3 b - - 0 19",
    ),
    (
        "h8c8",
        "2r5/p7/1n2k3/3p2p1/2P1p1p1/1P3rB1/P2R1P2/2K1R3 w - - 1 20",
    ),
    (
        "c1d1",
        "2r5/p7/1n2k3/3p2p1/2P1p1p1/1P3rB1/P2R1P2/3KR3 b - - 2 20",
    ),
    (
        "e6f5",
        "2r5/p7/1n6/3p1kp1/2P1p1p1/1P3rB1/P2R1P2/3KR3 w - - 3 21",
    ),
    (
        "c4d5",
        "2r5/p7/1n6/3P1kp1/4p1p1/1P3rB1/P2R1P2/3KR3 b - - 0 21",
    ),
    (
        "c8c3",
        "8/p7/1n6/3P1kp1/4p1p1/1Pr2rB1/P2R1P2/3KR3 w - - 1 22",
    ),
    (
        "d5d6",
        "8/p7/1n1P4/5kp1/4p1p1/1Pr2rB1/P2R1P2/3KR3 b - - 0 22",
    ),
    (
        "c3d3",
        "8/p7/1n1P4/5kp1/4p1p1/1P1r1rB1/P2R1P2/3KR3 w - - 1 23",
    ),
    (
        "d2d3",
        "8/p7/1n1P4/5kp1/4p1p1/1P1R1rB1/P4P2/3KR3 b - - 0 23",
    ),
    ("f3d3", "8/p7/1n1P4/5kp1/4p1p1/1P1r2B1/P4P2/3KR3 w - - 0 24"),
];

/// Replay a game through a `Position` + explicit boxed state list and check the
/// FEN after every ply, plus the incrementally maintained zobrist key against a
/// from-scratch recomputation.
fn check_game(start_fen: &str, game: &[(&str, &str)]) {
    let mut r = Replay::new(start_fen);
    assert_eq!(
        r.pos.key(),
        key_from_scratch(start_fen),
        "root key mismatch for {start_fen}"
    );

    for (ply, (uci, expected_fen)) in game.iter().enumerate() {
        let before = r.pos.fen();
        r.push_uci(uci);
        let got = r.pos.fen();
        assert_eq!(
            &got,
            expected_fen,
            "ply {} ({uci}) diverged.\n  from: {before}\n   got: {got}\n  want: {expected_fen}",
            ply + 1
        );

        // Castling rights must never be silently dropped: compare the castling
        // field explicitly so the failure message names the corrupted field.
        assert_eq!(
            castling_field(&got),
            castling_field(expected_fen),
            "ply {} ({uci}): castling rights corrupted (got {}, want {})",
            ply + 1,
            castling_field(&got),
            castling_field(expected_fen)
        );

        // The incrementally updated key must equal a from-scratch recomputation
        // of the same position; a wrong ep decision breaks this.
        assert_eq!(
            r.pos.key(),
            key_from_scratch(expected_fen),
            "ply {} ({uci}): incremental zobrist key != key recomputed from {expected_fen}",
            ply + 1
        );
    }
}

#[test]
fn state_chain_survives_long_startpos_game() {
    check_game(START_FEN, GAME_STARTPOS);
}

#[test]
fn state_chain_survives_long_kiwipete_game() {
    check_game(KIWIPETE_FEN, GAME_KIWIPETE);
}

/// Castling rights survive the 4th move of an Italian opening.
#[test]
fn castling_rights_preserved_after_move_four() {
    let mut r = Replay::new(START_FEN);
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"] {
        r.push_uci(uci);
    }
    let fen = r.pos.fen();
    assert_eq!(
        fen, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "startpos + e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"
    );
    assert_eq!(
        castling_field(&fen),
        "KQkq",
        "castling rights were silently dropped"
    );
}

/// `Engine::set_position` owns the state list. Every prefix of both games is
/// replayed through it.
#[test]
fn engine_set_position_state_list_is_stable() {
    let mut engine = Engine::new(None);

    for (start, game) in [(START_FEN, GAME_STARTPOS), (KIWIPETE_FEN, GAME_KIWIPETE)] {
        // Checkpoints spread across the game: ply 4 (the first move at which a
        // moved state list would corrupt the chain), then every 8th ply, then
        // the final position.
        let mut checkpoints: Vec<usize> = vec![4];
        checkpoints.extend((8..=game.len()).step_by(8));
        if *checkpoints.last().unwrap() != game.len() {
            checkpoints.push(game.len());
        }

        for n in checkpoints {
            let moves: Vec<String> = game[..n].iter().map(|(m, _)| m.to_string()).collect();
            assert!(
                engine.set_position(start, &moves).is_none(),
                "set_position rejected a legal prefix of length {n}"
            );
            assert_eq!(
                engine.fen(),
                game[n - 1].1,
                "Engine FEN wrong after {n} plies from {start}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// En-passant square legality
//
// After a double push the ep square is recorded only if the capture is legal for
// the *capturing* side: its own king must not be exposed once both pawns leave
// the rank. Vectors 3 and 5 are the discriminators: in 3 the capturing side's
// king is exposed (ep must be cleared), in 5 only the moving side's king is
// exposed (ep must be kept).
//
// All expected FENs come from upstream C++ Stockfish.
// ---------------------------------------------------------------------------

struct EpVector {
    name: &'static str,
    fen: &'static str,
    moves: &'static [&'static str],
    expect_fen: &'static str,
    expect_ep: &'static str,
}

const EP_VECTORS: &[EpVector] = &[
    EpVector {
        name: "kiwipete line: ep is set and only KQ castling remains",
        fen: KIWIPETE_FEN,
        moves: &["e5d7", "e8d7", "f3d3", "c7c5"],
        expect_fen: "r6r/p2kqpb1/bn2pnp1/2pP4/1p2P3/2NQ3p/PPPBBPPP/R3K2R w KQ c6 0 3",
        expect_ep: "c6",
    },
    EpVector {
        // Stockfish only records an ep square when the capture is actually
        // available: after 1.e4 no black pawn attacks e3.
        name: "1.e4: no black pawn can capture on e3",
        fen: START_FEN,
        moves: &["e2e4"],
        expect_fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        expect_ep: "-",
    },
    EpVector {
        name: "1.d4 d5 2.c4 c5: no white pawn on the 5th rank, so no ep",
        fen: START_FEN,
        moves: &["d2d4", "d7d5", "c2c4", "c7c5"],
        expect_fen: "rnbqkbnr/pp2pppp/8/2pp4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        expect_ep: "-",
    },
    EpVector {
        // Capturing side (White) has Ka5; exd6 e.p. would clear both d5 and e5
        // from rank 5 and expose the king to Qh5, so the ep square must NOT be
        // recorded.
        name: "capturing side's king is exposed along rank 5 -> ep cleared",
        fen: "3k4/3p4/8/K3P2q/8/8/8/8 b - - 0 1",
        moves: &["d7d5"],
        expect_fen: "3k4/8/8/K2pP2q/8/8/8/8 w - - 0 2",
        expect_ep: "-",
    },
    EpVector {
        // Same position with a knight on g5 blocking the queen: ep is legal.
        name: "same rank-5 pin blocked by a knight -> ep set",
        fen: "3k4/3p4/8/K3P1nq/8/8/8/8 b - - 0 1",
        moves: &["d7d5"],
        expect_fen: "3k4/8/8/K2pP1nq/8/8/8/8 w - d6 0 2",
        expect_ep: "d6",
    },
    EpVector {
        // Mirror image: the *moving* side's king (Black, a5) would be exposed
        // after the hypothetical exd6, but that is irrelevant -- White's king is
        // on a1 and the capture is legal, so ep IS recorded.
        name: "moving side's king exposed is irrelevant -> ep set",
        fen: "8/3p4/8/k3P2R/8/8/8/K7 b - - 0 1",
        moves: &["d7d5"],
        expect_fen: "8/8/8/k2pP2R/8/8/8/K7 w - d6 0 2",
        expect_ep: "d6",
    },
    EpVector {
        name: "plain legal ep in a king-and-pawn position",
        fen: "4k3/4p3/8/3P4/8/8/8/4K3 b - - 0 1",
        moves: &["e7e5"],
        expect_fen: "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2",
        expect_ep: "e6",
    },
    EpVector {
        name: "ep capture itself is playable and produces the right position",
        fen: START_FEN,
        moves: &["e2e4", "a7a6", "e4e5", "d7d5", "e5d6", "c7d6"],
        expect_fen: "rnbqkbnr/1p2pppp/p2p4/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 4",
        expect_ep: "-",
    },
];

#[test]
fn en_passant_square_legality() {
    for v in EP_VECTORS {
        let mut r = Replay::new(v.fen);
        for m in v.moves {
            r.push_uci(m);
        }
        let got = r.pos.fen();
        assert_eq!(&got, v.expect_fen, "[{}] full FEN mismatch", v.name);
        assert_eq!(
            ep_field(&got),
            v.expect_ep,
            "[{}] ep square wrong (got {}, want {})",
            v.name,
            ep_field(&got),
            v.expect_ep
        );
        // The ep square is hashed into the key, so a wrong ep decision also
        // desynchronises the incremental key from a from-scratch computation.
        assert_eq!(
            r.pos.key(),
            key_from_scratch(v.expect_fen),
            "[{}] zobrist key disagrees with from-scratch key",
            v.name
        );
    }
}

// ---------------------------------------------------------------------------
// NNUE
//
// `Position::do_move` must fill `DirtyThreats`, otherwise the incremental
// accumulator update uses stale threat features while the full-refresh path
// (which is what the `eval` UCI command exercises) stays correct.
//
// `nnue_incremental_matches_full_refresh` is the oracle for that: for every legal
// move it compares
//     push(dirties) + do_move + evaluate           (incremental, what search does)
// against
//     set_fen(resulting FEN) + evaluate            (full refresh, from biases)
// with a freshly cleared refresh cache for the oracle so the second number really
// is computed from scratch.
//
// The layer stack and output scaling below the accumulator (`propagate`) affect
// the incremental and refresh paths identically, so the differential test cannot
// see them. `nnue_eval_matches_cpp_golden_values` covers them instead, with
// evaluations taken from C++ Stockfish's `eval` command ("internal units" line).
//
// The two tests are complementary: a wrong threat update fails both incremental
// tests and neither golden test; a wrong layer-stack output fails only the
// golden test.
// ---------------------------------------------------------------------------

fn full_refresh_eval(net: &Network, fen: &str) -> Value {
    let r = Replay::new(fen);
    let mut accs = AccumulatorStack::new();
    accs.reset();
    let mut cache = Box::new(AccumulatorCaches::new());
    cache.clear(net);
    net.evaluate(&r.pos, &mut accs, &mut cache)
}

/// Positions chosen to exercise threats (kiwipete, the dedicated threat
/// position), pawn pairs (pawn endgames) and a sparse rook endgame.
const NNUE_POSITIONS: &[(&str, &str)] = &[
    ("startpos", START_FEN),
    ("kiwipete", KIWIPETE_FEN),
    ("rook endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11"),
    ("pawn endgame", "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1"),
    (
        "many threats",
        "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
    ),
    (
        "middlegame with pawn chains",
        "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    ),
];

#[test]
fn nnue_incremental_matches_full_refresh() {
    let Some(net) = load_network() else {
        common::network_missing("nnue_incremental_matches_full_refresh");
        return;
    };

    for (name, fen) in NNUE_POSITIONS {
        let mut r = Replay::new(fen);
        let mut accs = AccumulatorStack::new();
        accs.reset();
        let mut cache = Box::new(AccumulatorCaches::new());
        cache.clear(&net);

        // Compute the root accumulators first: this is what makes the following
        // per-move evaluations take the *incremental* path.
        let root = net.evaluate(&r.pos, &mut accs, &mut cache);
        assert_eq!(
            root,
            full_refresh_eval(&net, fen),
            "[{name}] root evaluation itself differs between the stack and a fresh refresh"
        );

        let moves = movegen::generate(&r.pos, GenType::All);
        assert!(!moves.as_slice().is_empty(), "[{name}] no legal moves");

        for &mv in moves.as_slice() {
            let gives_check = r.pos.gives_check(mv);
            // SAFETY: the pushed `Dirties` is filled by `do_move_internal` right below, before
            // the next evaluation.
            let dirties: *mut Dirties = unsafe { accs.push(&r.pos) };
            // SAFETY: `dirties` points into the accumulator stack entry just
            // pushed; it does not alias `pos`. This is the same dance
            // `Search::Worker::do_move` performs.
            unsafe { r.pos.do_move_internal(mv, gives_check, &mut *dirties) };

            let incremental = net.evaluate(&r.pos, &mut accs, &mut cache);
            let fen_after = r.pos.fen();

            // SAFETY: the move is legal.
            unsafe { r.pos.undo_move(mv) };
            accs.pop();

            let refreshed = full_refresh_eval(&net, &fen_after);
            assert_eq!(
                incremental, refreshed,
                "[{name}] move {mv}: incremental {incremental:?} != \
                 full refresh {refreshed:?} for {fen_after}"
            );
        }
    }
}

#[test]
fn nnue_incremental_matches_full_refresh_at_depth() {
    let Some(net) = load_network() else {
        common::network_missing("nnue_incremental_matches_full_refresh_at_depth");
        return;
    };

    // Lines chosen to include captures, a check, castling and a double push, so
    // the threat set changes a lot between plies, and errors compound with depth.
    let lines: &[(&str, &[&str])] = &[
        (START_FEN, &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),
        (KIWIPETE_FEN, &["e5g6", "f7g6", "e2a6", "b4c3", "e1g1"]),
        (
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
            &["b4f4", "h4g3", "f4c4", "h5c5", "a5b4", "c5c4"],
        ),
    ];

    for (fen, moves) in lines {
        let mut r = Replay::new(fen);
        let mut accs = AccumulatorStack::new();
        accs.reset();
        let mut cache = Box::new(AccumulatorCaches::new());
        cache.clear(&net);
        let _ = net.evaluate(&r.pos, &mut accs, &mut cache);

        for (i, uci) in moves.iter().enumerate() {
            let mv = UciEngine::to_move(&r.pos, uci);
            assert_ne!(mv, Move::none(), "illegal move {uci} in {}", r.pos.fen());
            let gives_check = r.pos.gives_check(mv);
            // SAFETY: as above; `do_move_internal` fills the pushed `Dirties` before the next
            // evaluation.
            let dirties: *mut Dirties = unsafe { accs.push(&r.pos) };
            // SAFETY: `to_move` only returns legal moves.
            unsafe { r.pos.do_move_internal(mv, gives_check, &mut *dirties) };

            let incremental = net.evaluate(&r.pos, &mut accs, &mut cache);
            let fen_after = r.pos.fen();
            let refreshed = full_refresh_eval(&net, &fen_after);
            assert_eq!(
                incremental,
                refreshed,
                "after {} plies of {fen} (last move {uci}): incremental {incremental:?} != \
                 refresh {refreshed:?} for {fen_after}",
                i + 1
            );
        }
    }
}

/// Golden NNUE evaluations from upstream C++ Stockfish's `eval` command
/// ("NNUE evaluation <v> (side to move, internal units)"), i.e. the network
/// output after the output scaling. These check layer stack and output scaling
/// parity, which the incremental-vs-refresh test above cannot see; every target
/// runs them against whichever kernel it compiles.
/// Re-derived whenever the default network changes; these are for
/// `nn-bddd9d57eb87.nnue` (SFNNv20).
const NNUE_GOLDEN: &[(&str, Value)] = &[
    (START_FEN, 45),
    (KIWIPETE_FEN, -625),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11", 20),
    ("8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1", -1031),
    (
        "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
        6577,
    ),
    (
        "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
        -155,
    ),
];

#[test]
fn nnue_eval_matches_cpp_golden_values() {
    let Some(net) = load_network() else {
        common::network_missing("nnue_eval_matches_cpp_golden_values");
        return;
    };

    for (fen, expected) in NNUE_GOLDEN {
        let got = full_refresh_eval(&net, fen);
        assert_eq!(
            got, *expected,
            "NNUE evaluation (internal units) for {fen}: got {got}, C++ Stockfish says {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// UCI setoption side effects
//
// The Hash / Threads / Clear Hash on-change callbacks must take effect:
// `setoption name Hash value N` resizes the transposition table. This test
// drives the real `UciEngine::setoption` (the same entry point the UCI loop
// uses) and reads the TT's cluster count back.
// ---------------------------------------------------------------------------

#[test]
fn setoption_hash_resizes_the_transposition_table() {
    let mut uci = UciEngine::new(&[]);

    let clusters_for = |mb: usize| mb * 1024 * 1024 / core::mem::size_of::<Cluster>();

    let start = uci.tt_cluster_count();
    assert!(start > 0, "TT should be allocated at startup");

    for mb in [1usize, 64, 8, 256] {
        uci.setoption(&format!("setoption name Hash value {mb}"));
        assert_eq!(
            uci.tt_cluster_count(),
            clusters_for(mb),
            "setoption name Hash value {mb} did not resize the transposition table"
        );
        assert_eq!(
            uci.engine_options().get_int("Hash", 0),
            mb as i32,
            "Hash option value not stored"
        );
    }

    // Two *distinct* sizes must give distinct cluster counts (guards against an
    // accessor that just reports the requested value).
    uci.setoption("setoption name Hash value 32");
    let c32 = uci.tt_cluster_count();
    uci.setoption("setoption name Hash value 128");
    let c128 = uci.tt_cluster_count();
    assert_ne!(c32, c128);
    assert_eq!(c128, 4 * c32);

    // Clear Hash must not resize, but must be dispatched without panicking.
    uci.setoption("setoption name Clear Hash");
    assert_eq!(uci.tt_cluster_count(), c128);
}

/// A rejected value (out of range) must not run the option's side effects.
#[test]
fn setoption_rejected_value_has_no_side_effects() {
    let mut uci = UciEngine::new(&[]);
    uci.setoption("setoption name Hash value 8");
    let before = uci.tt_cluster_count();

    uci.setoption("setoption name Hash value 0");
    assert_eq!(
        uci.tt_cluster_count(),
        before,
        "rejected Hash value resized the TT"
    );
    assert_eq!(uci.engine_options().get_int("Hash", 0), 8);
}

// ---------------------------------------------------------------------------
// NumaPolicy setoption
// ---------------------------------------------------------------------------

/// `NumaPolicy` setoption replaces the config, ignores unparsable values, and
/// `none` collapses to one node.
#[test]
fn setoption_numapolicy_changes_the_numa_config() {
    let mut uci = UciEngine::new(&[]);

    // A custom config replaces the system topology...
    uci.setoption("setoption name NumaPolicy value 0-1:2-3");
    assert_eq!(uci.numa_config_string(), "0-1:2-3");

    // ...an unparsable value keeps the previous config...
    uci.setoption("setoption name NumaPolicy value garbage");
    assert_eq!(uci.numa_config_string(), "0-1:2-3");

    // ...and `none` collapses everything into one node.
    uci.setoption("setoption name NumaPolicy value none");
    assert!(
        !uci.numa_config_string().contains(':'),
        "NumaPolicy=none must yield a single node, got {}",
        uci.numa_config_string()
    );
}

// ---------------------------------------------------------------------------
// Shared history initialisation values
//
// `correctionHistory` is filled with -5 and `pawnHistory` with -1338 (C++
// `Search::Worker::clear()`, src/search.cpp). A wrong fill is invisible to any
// test that only checks that search runs.
// ---------------------------------------------------------------------------

/// Minimal single-threaded pool so the test can size a `TranspositionTable`
/// without standing up the real thread pool.
struct OneThreadPool;

// SAFETY: jobs execute inline and no callbacks remain pending.
unsafe impl TtThreadPool for OneThreadPool {
    fn num_threads(&self) -> usize {
        1
    }
    fn run_on_thread<F: FnOnce() + Send + 'static>(&self, _i: usize, f: F) {
        f()
    }
    fn wait_on_thread(&self, _i: usize) {}
}

/// Build a standalone main-thread `Worker` (thread index 0) on top of a private
/// `SharedState`, so the test owns the shared histories it inspects. `tt_mb > 0`
/// allocates a transposition table, which a real search needs.
fn make_worker(shared: &Arc<SharedHistories>, tt_mb: usize) -> Box<Worker> {
    make_worker_with_network(shared, tt_mb, Network::new())
}

/// Like [`make_worker`], with a loaded `network` for tests that evaluate positions (a
/// `Network::new()` owns no weights, as in C++).
fn make_worker_with_network(
    shared: &Arc<SharedHistories>,
    tt_mb: usize,
    network: Network,
) -> Box<Worker> {
    init_tables();
    let tt = Arc::new(RwLock::new(TranspositionTable::new()));
    if tt_mb > 0 {
        tt.write().unwrap().resize(tt_mb, &OneThreadPool);
    }
    let shared_state = SharedState::new(
        Arc::new(RwLock::new(OptionsMap::new())),
        Arc::new(RwLock::new(ThreadPool::new())),
        tt,
        Arc::new(LazyNumaReplicated::new(network)),
        Arc::new(SearchControl::new()),
        Arc::clone(shared),
    );
    // Thread index 0 is the main thread, so it owns a `SearchManager` (upstream
    // 0e24f3f2: only the helper threads get `None`). The callbacks discard
    // everything; the tests below inspect worker state, not UCI output.
    let updates = Arc::new(RwLock::new(UpdateContext {
        on_update_no_moves: Box::new(|_| {}),
        on_update_full: Box::new(|_| {}),
        on_iter: Box::new(|_| {}),
        on_bestmove: Box::new(|_, _| {}),
        on_start: Box::new(|| {}),
    }));
    Box::new(Worker::new(
        shared_state,
        Some(Box::new(SearchManager::new(updates))),
        0,
        0,
        1,
        NumaReplicatedAccessToken::new(0),
    ))
}

#[test]
fn worker_clear_uses_cpp_history_init_values() {
    let shared = Arc::new(SharedHistories::new(1));
    let mut worker = make_worker(&shared, 0);

    // Dirty a few entries so `clear()` has something to overwrite: the test
    // fails if `clear()` skips a table.
    worker.main_history[0][0].set(1234);
    worker.capture_history[1][2][3].set(1234);
    shared.pawn_history[0][1][2].set(1234);
    shared.correction_history[0].pawn.set(1000);
    for (p1, s1, p2, s2) in [(0usize, 0usize, 0usize, 0usize), (5, 17, 9, 42)] {
        // Within the correction-history limit (1024), which `set` asserts in debug.
        worker.continuation_correction_history[p1][s1][p2][s2].set(1000);
    }
    for in_check in 0..2 {
        for capture in 0..2 {
            for (p1, s1, p2, s2) in [(0usize, 0usize, 0usize, 0usize), (7, 33, 2, 60)] {
                worker.continuation_history()[in_check][capture][p1][s1][p2][s2].set(1234);
            }
        }
    }

    worker.clear();

    // mainHistory.fill(-5)
    for c in 0..2 {
        for i in [0usize, 1, 12345, 65535] {
            assert_eq!(
                worker.main_history[c][i].get(),
                -5,
                "mainHistory[{c}][{i}] should be filled with -5"
            );
        }
    }

    // captureHistory.fill(-742)
    for (pc, sq, pt) in [(0usize, 0usize, 0usize), (1, 2, 3), (15, 63, 7)] {
        assert_eq!(
            worker.capture_history[pc][sq][pt].get(),
            -742,
            "captureHistory[{pc}][{sq}][{pt}] should be filled with -742"
        );
    }

    // continuationCorrectionHistory.fill(5)
    for (p1, s1, p2, s2) in [
        (0usize, 0usize, 0usize, 0usize),
        (5, 17, 9, 42),
        (15, 63, 15, 63),
    ] {
        assert_eq!(
            worker.continuation_correction_history[p1][s1][p2][s2].get(),
            5,
            "continuationCorrectionHistory[{p1}][{s1}][{p2}][{s2}] should be filled with 5"
        );
    }

    // continuationHistory[inCheck][capture].fill(-586)
    for in_check in 0..2 {
        for capture in 0..2 {
            for (p1, s1, p2, s2) in [(0usize, 0usize, 0usize, 0usize), (7, 33, 2, 60)] {
                assert_eq!(
                    worker.continuation_history()[in_check][capture][p1][s1][p2][s2].get(),
                    -586,
                    "continuationHistory[{in_check}][{capture}][{p1}][{s1}][{p2}][{s2}] \
                     should be filled with -586"
                );
            }
        }
    }

    // sharedHistory.correctionHistory.clear_range(-5, ...)
    let corr_size = shared.correction_history.get_size();
    for i in [0usize, 1, corr_size / 2, corr_size - 1] {
        let bundle = &shared.correction_history[i];
        assert_eq!(bundle.pawn.get(), -5, "correctionHistory[{i}].pawn");
        assert_eq!(bundle.minor.get(), -5, "correctionHistory[{i}].minor");
        assert_eq!(
            bundle.non_pawn_white.get(),
            -5,
            "correctionHistory[{i}].nonPawnWhite"
        );
        assert_eq!(
            bundle.non_pawn_black.get(),
            -5,
            "correctionHistory[{i}].nonPawnBlack"
        );
    }

    // sharedHistory.pawnHistory.clear_range(-1338, ...)
    let pawn_size = shared.pawn_history.get_size();
    for i in [0usize, 1, pawn_size / 2, pawn_size - 1] {
        for (pc, sq) in [(0usize, 0usize), (6, 31), (15, 63)] {
            assert_eq!(
                shared.pawn_history[i][pc][sq].get(),
                -1338,
                "pawnHistory[{i}][{pc}][{sq}] should be filled with -1338 (was -1238)"
            );
        }
    }
}

/// `lowPlyHistory` is filled with 102 at the start of `iterative_deepening`
/// (C++ `search.cpp`), not in `clear()`, so it needs an actual search. A depth-1
/// search on a standalone main-thread worker is enough and takes ~1s.
#[test]
fn low_ply_history_is_filled_with_102_by_iterative_deepening() {
    // The search evaluates positions, so it needs the real network.
    let Some(network) = load_network() else {
        common::network_missing("low_ply_history_is_filled_with_102_by_iterative_deepening");
        return;
    };
    let shared = Arc::new(SharedHistories::new(1));
    let mut worker = make_worker_with_network(&shared, 1, network);

    // Poison the table first, so a missing (or differently valued) fill fails.
    // The value must stay inside `StatsEntry`'s |value| <= D range (D = 7183 for
    // LowPlyHistory), otherwise the `debug_assert!` in `StatsEntry::set` fires in
    // debug builds.
    const POISON: i16 = 7000;
    for ply in 0..worker.low_ply_history.len() {
        for i in [0usize, 1234, 65535] {
            worker.low_ply_history[ply][i].set(POISON);
        }
    }
    worker.clear();
    assert_eq!(
        worker.low_ply_history[0][0].get(),
        POISON,
        "clear() is not expected to touch lowPlyHistory (C++ fills it in iterative_deepening)"
    );

    worker.root_pos.set_fen(START_FEN, false).unwrap();
    worker.root_moves = movegen::generate(&worker.root_pos, GenType::All)
        .as_slice()
        .iter()
        .map(|&m| RootMove::new(m))
        .collect();
    worker.limits.depth = 1;
    worker.iterative_deepening();

    // This public entry point bypasses start_searching/new_search. Its writes
    // still invalidate the pristine-table shortcut without per-node stores.
    let key = worker.root_pos.key();
    let mut tt = worker.shared_state.tt.write().unwrap();
    assert!(
        tt.probe(key).0,
        "direct iterative deepening must write its root result"
    );
    tt.clear(&OneThreadPool);
    assert!(!tt.probe(key).0);
    drop(tt);

    for ply in 0..worker.low_ply_history.len() {
        for i in [0usize, 1234, 65535] {
            assert_eq!(
                worker.low_ply_history[ply][i].get(),
                102,
                "lowPlyHistory[{ply}][{i}] should be filled with 102"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// make_key LCG constants
//
// `make_key` must match C++ `types.h` `make_key()`:
//     seed * 6364136223846793005 + 1442695040888963407
// A wrong additive constant perturbs the TT probe key through `adjust_key50`
// once rule50 >= 14: identical search trees, different TT slots. The expected
// value below is computed independently in the test.
// ---------------------------------------------------------------------------

#[test]
fn make_key_matches_cpp_lcg() {
    const MUL: u64 = 6_364_136_223_846_793_005;
    const ADD: u64 = 1_442_695_040_888_963_407;

    // Guard the literal constants themselves (C++ types.h `make_key()`).
    assert_eq!(MUL, 0x5851_F42D_4C95_7F2D);
    assert_eq!(ADD, 0x1405_7B7E_F767_814F);

    for seed in [
        0u64,
        1,
        2,
        3,
        7,
        8,
        13,
        14,
        u64::from(u32::MAX),
        0x0123_4567_89AB_CDEF,
        u64::MAX,
    ] {
        let expected = seed.wrapping_mul(MUL).wrapping_add(ADD);
        assert_eq!(
            stockcrab::types::make_key(seed),
            expected,
            "make_key({seed}) must equal seed * {MUL} + {ADD} (mod 2^64)"
        );
    }

    // A couple of fully independent, hand-evaluated values (u128 arithmetic, so
    // the reference does not share the wrapping-multiply code path above).
    for seed in [1u64, 5, 1234567] {
        let expected =
            (((seed as u128) * (MUL as u128) + (ADD as u128)) & u128::from(u64::MAX)) as u64;
        assert_eq!(stockcrab::types::make_key(seed), expected);
    }
}

// ---------------------------------------------------------------------------
// VALUE_TB consistency
//
// `VALUE_TB` is `VALUE_MATE_IN_MAX_PLY - 1 = 31753`, so `is_win` / `is_loss` /
// `is_decisive` use thresholds +/-31507.
//
// `search.rs`'s copies of these constants are private, so this test covers
// `types.rs` while `search.rs` carries a compile-time assert that its constants
// equal the ones in `types.rs` (see the `const _: () = assert!(...)` block
// there). A mismatch fails the build, which fails this test binary too.
// ---------------------------------------------------------------------------

#[test]
fn value_tb_and_classification_thresholds() {
    use stockcrab::types::{
        MAX_PLY, VALUE_INFINITE, VALUE_MATE, VALUE_MATE_IN_MAX_PLY, VALUE_NONE, VALUE_TB,
        VALUE_TB_LOSS_IN_MAX_PLY, VALUE_TB_WIN_IN_MAX_PLY, is_decisive, is_loss, is_valid, is_win,
    };

    // C++ types.h VALUE_MATE / VALUE_TB constants.
    assert_eq!(VALUE_MATE, 32000);
    assert_eq!(VALUE_INFINITE, 32001);
    assert_eq!(VALUE_NONE, 32002);
    assert_eq!(MAX_PLY, 246);
    assert_eq!(VALUE_MATE_IN_MAX_PLY, VALUE_MATE - MAX_PLY as Value);
    assert_eq!(VALUE_MATE_IN_MAX_PLY, 31754);
    assert_eq!(VALUE_TB, VALUE_MATE_IN_MAX_PLY - 1);
    assert_eq!(VALUE_TB, 31753);
    assert_eq!(VALUE_TB_WIN_IN_MAX_PLY, VALUE_TB - MAX_PLY as Value);
    assert_eq!(VALUE_TB_WIN_IN_MAX_PLY, 31507);
    assert_eq!(VALUE_TB_LOSS_IN_MAX_PLY, -VALUE_TB_WIN_IN_MAX_PLY);

    // Boundary behaviour of the classification helpers (C++ types.h `is_win` / `is_loss` / `is_decisive`).
    assert!(is_win(VALUE_TB_WIN_IN_MAX_PLY));
    assert!(!is_win(VALUE_TB_WIN_IN_MAX_PLY - 1));
    assert!(is_loss(VALUE_TB_LOSS_IN_MAX_PLY));
    assert!(!is_loss(VALUE_TB_LOSS_IN_MAX_PLY + 1));
    assert!(is_decisive(VALUE_TB_WIN_IN_MAX_PLY));
    assert!(is_decisive(VALUE_TB_LOSS_IN_MAX_PLY));
    assert!(!is_decisive(VALUE_TB_WIN_IN_MAX_PLY - 1));
    assert!(!is_decisive(VALUE_TB_LOSS_IN_MAX_PLY + 1));
    assert!(!is_decisive(0));

    // A score well below the TB-win threshold is not decisive.
    let below_threshold: Value = 29707;
    assert!(
        !is_decisive(below_threshold),
        "29707 must not be a decisive score"
    );

    assert!(is_valid(0));
    assert!(is_valid(VALUE_MATE));
    assert!(!is_valid(VALUE_NONE));
}

// ---------------------------------------------------------------------------
// Bench position list must match C++
//
// The Rust DEFAULTS list must equal the one in src/benchmark.cpp, or any
// aggregate bench comparison between the engines is meaningless.
// ---------------------------------------------------------------------------

/// Extract the `Defaults` string list from the C++ benchmark source.
fn parse_cpp_defaults(src: &str) -> Vec<String> {
    let start = src
        .find("const std::vector<std::string> Defaults = {")
        .expect("Defaults list not found in benchmark.cpp");
    let body = &src[start..];
    let end = body.find("\n};").expect("unterminated Defaults list");
    let body = &body[..end];

    let mut out = Vec::new();
    for line in body.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        // Take the first double-quoted literal on the line (no escaped quotes
        // occur in this list).
        if let Some(open) = line.find('"')
            && let Some(close) = line[open + 1..].find('"')
        {
            out.push(line[open + 1..open + 1 + close].to_string());
        }
    }
    out
}

/// Recover the Rust DEFAULTS list from `setup_bench`, which is the only public
/// consumer of it.
fn rust_defaults() -> Vec<String> {
    let cmds = stockcrab::benchmark::setup_bench(START_FEN, "");
    // setup_bench emits: setoption Threads, setoption Hash, ucinewgame, then for
    // each entry either the entry itself (setoption lines) or
    // "position fen <entry>" followed by the go command.
    let mut out = Vec::new();
    for cmd in cmds.into_iter().skip(3) {
        if let Some(rest) = cmd.strip_prefix("position fen ") {
            out.push(rest.to_string());
        } else if cmd.starts_with("setoption") {
            out.push(cmd);
        }
    }
    out
}

#[test]
fn bench_defaults_match_cpp_benchmark_cpp() {
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os("STOCKFISH_DIR") {
        candidates.push(std::path::Path::new(&dir).join("src/benchmark.cpp"));
    }
    candidates.push("../Stockfish/src/benchmark.cpp".into());
    candidates.push("../src/benchmark.cpp".into());
    let Some(path) = candidates.iter().find(|p| p.is_file()) else {
        common::cpp_source_missing(
            "bench_defaults_match_cpp_benchmark_cpp",
            "src/benchmark.cpp",
        );
        return;
    };
    let src = std::fs::read_to_string(path).expect("read benchmark.cpp");
    let cpp = parse_cpp_defaults(&src);
    let rust = rust_defaults();

    assert!(cpp.len() > 40, "sanity: parsed {} C++ entries", cpp.len());
    assert_eq!(
        rust.len(),
        cpp.len(),
        "bench position count differs: Rust has {}, C++ has {}",
        rust.len(),
        cpp.len()
    );
    for (i, (r, c)) in rust.iter().zip(cpp.iter()).enumerate() {
        assert_eq!(r, c, "bench entry {i} differs:\n  rust: {r}\n  cpp:  {c}");
    }
}

// ---------------------------------------------------------------------------
// MultiPV: an aborted search of a later PV line must not overtake a proven loss
// ---------------------------------------------------------------------------

fn root_move(from: Square, to: Square, score: Value) -> RootMove {
    let mut rm = RootMove::new(Move::from_to(from, to));
    rm.score = score;
    rm.uci_score = score;
    rm
}

/// Three root moves where line 0 is a proven loss and line 1 (aborted) claims a
/// better score than it, which is exactly what upstream's `search.cpp` guards against.
fn aborted_multipv_worker() -> Box<Worker> {
    let shared = Arc::new(SharedHistories::new(1));
    let mut worker = make_worker(&shared, 0);
    worker.root_moves = vec![
        root_move(Square::SQ_E2, Square::SQ_E4, mated_in(10)),
        root_move(Square::SQ_D2, Square::SQ_D4, 500),
        root_move(Square::SQ_G1, Square::SQ_F3, mated_in(8)),
    ];
    worker
}

#[test]
fn multipv_aborted_line_falls_back_to_its_exact_previous_score() {
    // mated_in(5) is a worse score than the proven loss above it (mated_in(10)), so
    // restoring it cannot overtake that line.
    let mut worker = aborted_multipv_worker();
    let previous_pv = vec![
        Move::from_to(Square::SQ_D2, Square::SQ_D4),
        Move::from_to(Square::SQ_D7, Square::SQ_D5),
    ];
    {
        let rm = &mut worker.root_moves[1];
        rm.previous_score = mated_in(5);
        rm.previous_score_exact = true;
        rm.previous_pv = previous_pv.clone();
    }

    worker.guard_aborted_multi_pv_line(1, 3);

    let rm = &worker.root_moves[1];
    assert_eq!(
        rm.score,
        mated_in(5),
        "score must be restored from the previous iteration"
    );
    assert_eq!(rm.uci_score, mated_in(5));
    assert_eq!(rm.previous_score, -VALUE_INFINITE);
    assert_eq!(rm.pv, previous_pv);
    assert!(!rm.is_inexact());
    // Line 2 is a loss that was only partially searched, so it becomes a lower bound.
    assert!(worker.root_moves[2].inexact_lower);
}

#[test]
fn multipv_aborted_line_is_capped_to_the_loss_above_it() {
    let mut worker = aborted_multipv_worker();
    worker.root_moves[1].previous_score = mated_in(20);
    worker.root_moves[1].previous_score_exact = false; // not trustworthy
    worker.root_moves[1].pv = vec![
        Move::from_to(Square::SQ_D2, Square::SQ_D4),
        Move::from_to(Square::SQ_D7, Square::SQ_D5),
    ];

    worker.guard_aborted_multi_pv_line(1, 3);

    let rm = &worker.root_moves[1];
    assert_eq!(
        rm.score,
        mated_in(10),
        "capped to the score of the line above"
    );
    assert_eq!(rm.previous_score, -VALUE_INFINITE);
    assert_eq!(rm.pv.len(), 1, "the unreliable PV tail is dropped");
    assert!(
        rm.inexact_upper && !rm.inexact_lower,
        "an upper bound, not an exact score"
    );
}

#[test]
fn multipv_guard_leaves_ordinary_lines_alone() {
    let mut worker = aborted_multipv_worker();
    worker.root_moves[0].score = 100;
    worker.root_moves[0].uci_score = 100;
    worker.root_moves[1].score = 50;
    worker.root_moves[1].uci_score = 50;

    worker.guard_aborted_multi_pv_line(1, 3);

    assert_eq!(worker.root_moves[1].score, 50);
    assert!(!worker.root_moves[1].is_inexact());
    // The loss in a later, partially searched line is still marked inexact.
    assert!(worker.root_moves[2].inexact_lower);
}
