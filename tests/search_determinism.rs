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

//! Search Determinism Test Suite
//!
//! Verifies that the search produces deterministic and correct results:
//! - Fixed-depth search returns the expected best move
//! - Tactical positions find correct moves
//! - Endgame positions find winning continuations
//!
//! Test positions sourced from:
//! - TCEC (Top Chess Engine Championship) games
//! - Strategic Test Suite (STS)
//! - Eigenmann Rapid Engine Test
//! - Famous tactical positions
//! - Endgame databases
//!
//! Note: this suite focuses on VERIFICATION, not on exercising every search
//! feature. Nothing here is #[ignore]d.
//!
//! `expected_move` is the move the *C++ reference engine* plays at that depth,
//! not the move a human would pick, and the trailing comments naming a different
//! move are historical annotations -- do not "fix" an expectation to match them.
//! Any Elo-affecting upstream commit can legitimately shift these. When one does,
//! re-derive the value from the C++ engine rather than from stockcrab alone:
//!
//! ```text
//! { printf 'setoption name Threads value 1\nposition fen <FEN>\ngo depth <D>\n'; sleep 300; } \
//!   | ../src/stockfish | grep -m1 '^bestmove'
//! ```
//!
//! Sending `quit` on the same line aborts the search and yields a bogus
//! `depth 0 ... nodes 0` result, so keep stdin open until `bestmove` appears.
//!
//! Expectations last re-derived from C++ at upstream `88337401` (network `nn-684d0e60e71d`).
//! Each position was searched independently with stdin kept open until `bestmove`.

mod common;

use std::sync::{Arc, Mutex};

use stockcrab::engine::Engine;
use stockcrab::misc::now;
use stockcrab::search::LimitsType;

/// Test case for search determinism
struct SearchTest {
    name: &'static str,
    fen: &'static str,
    depth: i32,
    /// Expected best move in UCI notation (e.g., "e2e4")
    expected_move: &'static str,
    /// Optional: expected evaluation (centipawns).
    ///
    /// Documentation only, not asserted: the `Engine` harness surfaces just the
    /// best move via `on_bestmove`.
    #[allow(dead_code)]
    expected_eval: Option<i32>,
}

/// Run a single search test.
///
/// This uses the `Engine` API and captures the best move via the `on_bestmove`
/// callback.
fn run_search_test(test: &SearchTest) {
    let mut engine = Engine::new(None);

    if let Some(err) = engine.set_position(test.fen, &[]) {
        panic!("Failed to set position for '{}': {:?}", test.name, err);
    }

    // Capture the best move reported by the search.
    let best_move = Arc::new(Mutex::new(String::new()));
    let best_move_cb = best_move.clone();
    engine.set_on_bestmove(move |best, _ponder| {
        *best_move_cb.lock().unwrap() = best.to_string();
    });

    // Setup search limits (fixed depth).
    //
    // `start_time` is stamped exactly as `UciEngine::parse_limits` does. The
    // `Default` value of 0 makes the search compute an elapsed time of
    // `now() - 0`, i.e. decades; fixed-depth searches do not consult the clock
    // so the results are the same either way, but leaving it unset would be a
    // trap for anyone adding a time-limited case here.
    let mut limits = LimitsType::default();
    limits.start_time = now();
    limits.depth = test.depth;

    engine.go(limits);
    engine.wait_for_search_finished();

    let actual_move_uci = best_move.lock().unwrap().clone();

    // Check expected move
    assert_eq!(
        actual_move_uci, test.expected_move,
        "Move mismatch for '{}' at depth {}\nFEN: {}\nExpected: {}, Got: {}",
        test.name, test.depth, test.fen, test.expected_move, actual_move_uci
    );

    println!(
        "OK {}: {} (depth {})",
        test.name, actual_move_uci, test.depth
    );
}

// ===========================================================================
// TACTICAL POSITIONS - Forced Wins
// ===========================================================================

#[test]
fn search_mate_in_one_1() {
    if common::skip_without_network("search_mate_in_one_1") {
        return;
    }
    let test = SearchTest {
        name: "Mate in 1 - Back Rank",
        fen: "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        depth: 10,
        expected_move: "a1a8",
        expected_eval: Some(30000), // Mate score
    };
    run_search_test(&test);
}

#[test]
fn search_bishop_retreat_best_move() {
    if common::skip_without_network("search_bishop_retreat_best_move") {
        return;
    }
    let test = SearchTest {
        name: "Bishop Retreat - Best Move",
        fen: "r1b1kb1r/pppp1ppp/2n2q2/4n3/2B1P3/2N5/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        depth: 10,
        expected_move: "c3d5",
        expected_eval: Some(30000),
    };
    run_search_test(&test);
}

#[test]
fn search_back_rank_queen_mate() {
    if common::skip_without_network("search_back_rank_queen_mate") {
        return;
    }
    let test = SearchTest {
        name: "Back Rank - Qd8#",
        fen: "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",
        depth: 15,
        expected_move: "d1d8",      // Qd8+
        expected_eval: Some(29998), // Mate score
    };
    run_search_test(&test);
}

#[test]
fn search_avoid_stalemate() {
    if common::skip_without_network("search_avoid_stalemate") {
        return;
    }
    let test = SearchTest {
        name: "Avoid Stalemate Trap",
        fen: "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        depth: 20,
        expected_move: "f7g7", // Qg7# -- Kg6 guards g7, so this is mate, not stalemate
        expected_eval: Some(20000), // Winning
    };
    run_search_test(&test);
}

// ===========================================================================
// OPENING THEORY - Known Best Moves
// ===========================================================================

#[test]
fn search_opening_italian_game() {
    if common::skip_without_network("search_opening_italian_game") {
        return;
    }
    let test = SearchTest {
        name: "Italian Game - Common Line",
        fen: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        depth: 18,
        expected_move: "f8c5",
        expected_eval: Some(0), // Roughly equal
    };
    run_search_test(&test);
}

#[test]
fn search_opening_sicilian_najdorf() {
    if common::skip_without_network("search_opening_sicilian_najdorf") {
        return;
    }
    let test = SearchTest {
        name: "Sicilian Najdorf",
        fen: "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
        depth: 18,
        expected_move: "f1e2",   // f4 or Be3 are main lines
        expected_eval: Some(30), // Slight edge for White
    };
    run_search_test(&test);
}

#[test]
fn search_opening_queens_gambit() {
    if common::skip_without_network("search_opening_queens_gambit") {
        return;
    }
    let test = SearchTest {
        name: "Queen's Gambit Declined",
        fen: "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 1",
        depth: 18,
        expected_move: "e7e6",
        expected_eval: Some(0),
    };
    run_search_test(&test);
}

// ===========================================================================
// MIDDLEGAME TACTICS
// ===========================================================================

#[test]
fn search_fork_knight() {
    if common::skip_without_network("search_fork_knight") {
        return;
    }
    let test = SearchTest {
        name: "Knight Fork Tactic",
        fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        depth: 20,
        expected_move: "d2d3",
        expected_eval: Some(100),
    };
    run_search_test(&test);
}

#[test]
fn search_pin_absolute() {
    if common::skip_without_network("search_pin_absolute") {
        return;
    }
    let test = SearchTest {
        name: "Absolute Pin Exploitation",
        fen: "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        depth: 20,
        expected_move: "c4d5",
        expected_eval: Some(300),
    };
    run_search_test(&test);
}

#[test]
fn search_discovered_attack() {
    if common::skip_without_network("search_discovered_attack") {
        return;
    }
    let test = SearchTest {
        name: "Discovered Attack",
        fen: "r1bqkb1r/pppp1ppp/2n5/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        depth: 20,
        expected_move: "d7d5",
        expected_eval: Some(-100),
    };
    run_search_test(&test);
}

#[test]
fn search_skewer() {
    if common::skip_without_network("search_skewer") {
        return;
    }
    let test = SearchTest {
        name: "Skewer Tactic",
        fen: "r1bqk2r/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 1",
        depth: 20,
        expected_move: "d7d6",
        expected_eval: Some(50),
    };
    run_search_test(&test);
}

// ===========================================================================
// ENDGAME POSITIONS
// ===========================================================================

#[test]
fn search_endgame_kpk_winning() {
    if common::skip_without_network("search_endgame_kpk_winning") {
        return;
    }
    let test = SearchTest {
        name: "KPK - Winning",
        fen: "8/8/8/3k4/8/8/4P3/4K3 w - - 0 1",
        depth: 30,
        expected_move: "e1f2",
        expected_eval: Some(200), // Winning
    };
    run_search_test(&test);
}

#[test]
fn search_endgame_kqk_mate() {
    if common::skip_without_network("search_endgame_kqk_mate") {
        return;
    }
    let test = SearchTest {
        name: "KQK - Force Mate",
        // Queen on b1: its b1-h7 diagonal is blocked by White's king on g6, so the
        // FEN is legal with White to move.
        fen: "7k/8/6K1/8/8/8/8/1Q6 w - - 0 1",
        depth: 25,
        expected_move: "b1b8", // Qb8#
        expected_eval: Some(30000),
    };
    run_search_test(&test);
}

#[test]
fn search_endgame_krk_mate() {
    if common::skip_without_network("search_endgame_krk_mate") {
        return;
    }
    let test = SearchTest {
        name: "KRK - Mate Pattern",
        // Rook on a1 does not attack the black king on h8, so the FEN is legal.
        fen: "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
        depth: 25,
        expected_move: "a1a8", // Ra8#
        expected_eval: Some(30000),
    };
    run_search_test(&test);
}

#[test]
fn search_endgame_rook_endgame() {
    if common::skip_without_network("search_endgame_rook_endgame") {
        return;
    }
    let test = SearchTest {
        name: "Rook Endgame - Lucena Position",
        fen: "1K6/1P6/8/8/8/8/r7/k7 w - - 0 1",
        depth: 25,
        expected_move: "b8c7",
        expected_eval: Some(1000),
    };
    run_search_test(&test);
}

// ===========================================================================
// POSITIONAL PLAY
// ===========================================================================

#[test]
fn search_positional_space_advantage() {
    if common::skip_without_network("search_positional_space_advantage") {
        return;
    }
    let test = SearchTest {
        name: "Positional - Space Advantage",
        fen: "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
        depth: 20,
        expected_move: "c2c4", // c4 gains space
        expected_eval: Some(20),
    };
    run_search_test(&test);
}

#[test]
fn search_positional_weak_squares() {
    if common::skip_without_network("search_positional_weak_squares") {
        return;
    }
    let test = SearchTest {
        name: "Exploit Weak Squares",
        fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        depth: 20,
        expected_move: "d2d3", // Solid development
        expected_eval: Some(30),
    };
    run_search_test(&test);
}

// ===========================================================================
// MIDDLEGAME POSITIONS
// ===========================================================================

#[test]
fn search_middlegame_knight_retreat() {
    if common::skip_without_network("search_middlegame_knight_retreat") {
        return;
    }
    let test = SearchTest {
        name: "Middlegame - Knight Retreat",
        fen: "r1k4r/p2nb1p1/2b4p/1p1n1p2/2PP4/3Q1NB1/1P3PPP/R5K1 b - - 0 1",
        depth: 22,
        expected_move: "b5c4",
        expected_eval: Some(-50),
    };
    run_search_test(&test);
}

#[test]
fn search_middlegame_queen_raid() {
    if common::skip_without_network("search_middlegame_queen_raid") {
        return;
    }
    let test = SearchTest {
        name: "Middlegame - Queen Raid",
        fen: "1r4k1/2p2ppp/p7/4bq2/Q7/1P4P1/P4P1P/3R2K1 w - - 0 1",
        depth: 22,
        expected_move: "a4a6",
        expected_eval: Some(400),
    };
    run_search_test(&test);
}

// ===========================================================================
// SPECIAL MOVES VERIFICATION
// ===========================================================================

#[test]
fn search_en_passant_capture() {
    if common::skip_without_network("search_en_passant_capture") {
        return;
    }
    let test = SearchTest {
        name: "En Passant Available",
        fen: "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1",
        depth: 18,
        expected_move: "c2c3",
        expected_eval: Some(50),
    };
    run_search_test(&test);
}

#[test]
fn search_italian_quiet_development() {
    if common::skip_without_network("search_italian_quiet_development") {
        return;
    }
    let test = SearchTest {
        name: "Italian Game - Quiet Development",
        fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        depth: 18,
        expected_move: "d2d3",
        expected_eval: Some(20),
    };
    run_search_test(&test);
}

#[test]
fn search_promotion_underpromotion() {
    if common::skip_without_network("search_promotion_underpromotion") {
        return;
    }
    let test = SearchTest {
        name: "Promotion to Queen",
        fen: "8/4P3/8/2k5/8/8/8/4K3 w - - 0 1",
        depth: 25,
        expected_move: "e7e8q",
        expected_eval: Some(800),
    };
    run_search_test(&test);
}

// ===========================================================================
// REPETITION DETECTION
// ===========================================================================

#[test]
fn search_avoid_threefold_repetition() {
    if common::skip_without_network("search_avoid_threefold_repetition") {
        return;
    }
    let test = SearchTest {
        name: "Avoid Repetition when Winning",
        fen: "8/8/8/4k3/8/8/3K4/R7 w - - 0 1",
        depth: 20,
        expected_move: "d2e3",
        expected_eval: Some(500),
    };
    run_search_test(&test);
}

// ===========================================================================
// ZUGZWANG POSITIONS
// ===========================================================================

#[test]
fn search_zugzwang_mutual() {
    if common::skip_without_network("search_zugzwang_mutual") {
        return;
    }
    let test = SearchTest {
        name: "Mutual Zugzwang",
        fen: "8/8/p7/P7/8/1k6/8/1K6 w - - 0 1",
        depth: 30,
        expected_move: "b1c1",
        expected_eval: Some(0),
    };
    run_search_test(&test);
}

// ===========================================================================
// COMPLEX EVALUATION TESTS
// ===========================================================================

#[test]
fn search_pawn_structure_evaluation() {
    if common::skip_without_network("search_pawn_structure_evaluation") {
        return;
    }
    let test = SearchTest {
        name: "Pawn Structure - Isolated Pawn",
        fen: "r1bqkb1r/pp1p1ppp/2n2n2/2p1p3/2P1P3/2N2N2/PP1P1PPP/R1BQKB1R w KQkq - 0 1",
        depth: 20,
        expected_move: "g2g3",
        expected_eval: Some(15),
    };
    run_search_test(&test);
}
