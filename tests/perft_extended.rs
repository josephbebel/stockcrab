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

//! Extended Perft Test Suite
//!
//! Comprehensive perft tests covering a wide range of positions to verify move
//! generation correctness across various scenarios:
//! - Standard opening positions
//! - Tactical positions (pins, discoveries, checks)
//! - Endgames
//! - Special moves (en passant, castling, promotions)
//! - Chess960 positions
//! - Edge cases and complex positions
//!
//! Expected node counts verified against Stockfish C++.

use stockcrab::bitboard;
use stockcrab::perft::perft_from_fen;

/// Perft test case structure
struct PerftTest {
    name: &'static str,
    fen: &'static str,
    is_chess960: bool,
    /// Expected node counts at depths 1, 2, 3, 4, 5...
    expected: Vec<u64>,
}

/// Initialize required tables before tests
fn init() {
    bitboard::init();
}

/// Helper function to run a single perft test
fn run_perft_test(test: &PerftTest, max_depth: usize) {
    init();
    assert_eq!(
        test.fen.split_whitespace().count(),
        6,
        "malformed FEN for '{}': {}",
        test.name,
        test.fen
    );

    for depth in 1..=max_depth.min(test.expected.len()) {
        let result = perft_from_fen(test.fen, depth as i32, test.is_chess960)
            .expect("perft_from_fen returned a PositionSetError for a valid test FEN");
        let expected = test.expected[depth - 1];

        assert_eq!(
            result, expected,
            "Perft mismatch for '{}' at depth {}\nFEN: {}\nExpected: {}, Got: {}",
            test.name, depth, test.fen, expected, result
        );
    }
}

// ===========================================================================
// STANDARD POSITIONS (Already covered in perft.rs, included for completeness)
// ===========================================================================

#[test]
fn perft_starting_position() {
    let test = PerftTest {
        name: "Starting Position",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        is_chess960: false,
        expected: vec![20, 400, 8902, 197281, 4865609],
    };
    run_perft_test(&test, 5);
}

#[test]
fn perft_kiwipete() {
    let test = PerftTest {
        name: "Kiwipete",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![48, 2039, 97862, 4085603],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_position_3() {
    let test = PerftTest {
        name: "Position 3 - Endgame",
        fen: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        is_chess960: false,
        expected: vec![14, 191, 2812, 43238, 674624],
    };
    run_perft_test(&test, 5);
}

#[test]
fn perft_position_4_promotions() {
    let test = PerftTest {
        name: "Position 4 - Promotions",
        fen: "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        is_chess960: false,
        expected: vec![6, 264, 9467, 422333],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_position_5() {
    let test = PerftTest {
        name: "Position 5 - Complex Middlegame",
        fen: "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        is_chess960: false,
        expected: vec![44, 1486, 62379, 2103487],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_position_6() {
    let test = PerftTest {
        name: "Position 6 - Symmetrical",
        fen: "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        is_chess960: false,
        expected: vec![46, 2079, 89890, 3894594],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// EN PASSANT POSITIONS
// ===========================================================================

#[test]
fn perft_en_passant_1() {
    let test = PerftTest {
        name: "En Passant Available",
        fen: "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
        is_chess960: false,
        expected: vec![21, 607, 13119, 382294],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_en_passant_2() {
    let test = PerftTest {
        name: "Double En Passant Capture",
        fen: "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
        is_chess960: false,
        expected: vec![9, 50, 379, 2369],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_en_passant_pinned() {
    let test = PerftTest {
        name: "En Passant with Pin",
        fen: "8/5k2/8/2Pp4/2B5/1K6/8/8 w - d6 0 1",
        is_chess960: false,
        expected: vec![15, 126, 1928, 13931],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_en_passant_discovered_check() {
    let test = PerftTest {
        name: "En Passant Causes Discovered Check",
        fen: "5k2/8/8/8/3pP3/8/8/3K4 b - e3 0 1",
        is_chess960: false,
        expected: vec![7, 38, 276, 1786],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// CASTLING POSITIONS
// ===========================================================================

#[test]
fn perft_castling_all_rights() {
    let test = PerftTest {
        name: "All Castling Rights",
        fen: "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![26, 568, 13744, 314346, 7594526],
    };
    run_perft_test(&test, 5);
}

#[test]
fn perft_castling_blocked() {
    let test = PerftTest {
        name: "Castling Blocked by Pieces",
        fen: "r3k2r/8/8/8/8/8/8/RN2K1NR w KQkq - 0 1",
        is_chess960: false,
        expected: vec![25, 552, 14605, 340597],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_castling_through_check() {
    let test = PerftTest {
        name: "Cannot Castle Through Check",
        fen: "r3k2r/8/8/8/3q4/8/8/R3K2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![21, 943, 17226, 735942],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_castling_in_check() {
    let test = PerftTest {
        name: "Cannot Castle Out of Check",
        fen: "r3k2r/4q3/8/8/8/8/8/R3K2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![4, 188, 3766, 156849],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_castling_rook_attacked() {
    let test = PerftTest {
        name: "Castling with Rook Attacked",
        fen: "r3k2r/8/8/8/8/8/8/R3K2q w Qkq - 0 1",
        is_chess960: false,
        expected: vec![3, 132, 1777, 70771],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// PROMOTION POSITIONS
// ===========================================================================

#[test]
fn perft_promotion_simple() {
    let test = PerftTest {
        name: "Simple Pawn Promotion",
        fen: "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        is_chess960: false,
        expected: vec![24, 496, 9483, 182838, 3605103],
    };
    run_perft_test(&test, 5);
}

#[test]
fn perft_promotion_captures() {
    let test = PerftTest {
        name: "Promotion with Captures",
        fen: "r3k2r/p1P4p/8/8/8/8/p6P/R3K2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![19, 254, 5476, 93006],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_underpromotion() {
    let test = PerftTest {
        name: "Underpromotion Positions",
        fen: "8/P7/8/8/8/2k5/p7/2K5 w - - 0 1",
        is_chess960: false,
        expected: vec![5, 46, 446, 3512],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// CHECK AND CHECKMATE POSITIONS
// ===========================================================================

#[test]
fn perft_escape_check() {
    let test = PerftTest {
        name: "King in Check - Must Escape",
        fen: "8/8/8/8/8/4k3/3r4/4K3 w - - 0 1",
        is_chess960: false,
        expected: vec![1, 19, 42, 792],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_double_check() {
    let test = PerftTest {
        name: "Double Check",
        fen: "8/8/8/4k3/8/3nB3/3K4/4R3 b - - 0 1",
        is_chess960: false,
        expected: vec![13, 265, 2785, 64290],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_discovered_check() {
    let test = PerftTest {
        name: "Discovered Check",
        fen: "8/8/8/8/3k4/4N3/4R3/4K3 w - - 0 1",
        is_chess960: false,
        expected: vec![19, 95, 2075, 10123],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_checkmate_in_one() {
    let test = PerftTest {
        name: "Checkmate in One Available",
        fen: "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        is_chess960: false,
        expected: vec![20, 152, 3029, 23005],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// ENDGAME POSITIONS
// ===========================================================================

#[test]
fn perft_kpk_endgame() {
    let test = PerftTest {
        name: "KPK Endgame",
        fen: "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
        is_chess960: false,
        expected: vec![6, 44, 294, 2100, 14991],
    };
    run_perft_test(&test, 5);
}

#[test]
fn perft_kqk_endgame() {
    let test = PerftTest {
        name: "KQK Endgame",
        fen: "8/8/8/4k3/8/8/8/4K2Q w - - 0 1",
        is_chess960: false,
        expected: vec![21, 110, 2801, 14273],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_krk_endgame() {
    let test = PerftTest {
        name: "KRK Endgame",
        fen: "8/8/8/4k3/8/8/8/4K2R w - - 0 1",
        is_chess960: false,
        expected: vec![14, 101, 1822, 12136],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_kbnk_endgame() {
    let test = PerftTest {
        name: "KBNK Endgame",
        fen: "8/8/8/4k3/8/8/8/2BNK3 w - - 0 1",
        is_chess960: false,
        expected: vec![15, 99, 1585, 9959],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// TACTICAL POSITIONS
// ===========================================================================

#[test]
fn perft_pin_horizontal() {
    let test = PerftTest {
        name: "Horizontal Pin",
        fen: "4k3/8/8/8/3Rn2r/8/8/4K3 b - - 0 1",
        is_chess960: false,
        expected: vec![20, 296, 6015, 90025],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_pin_diagonal() {
    let test = PerftTest {
        name: "Diagonal Pin",
        fen: "4k3/8/8/2b5/3N4/8/8/B3K3 w - - 0 1",
        is_chess960: false,
        expected: vec![15, 214, 3310, 43006],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_skewer() {
    let test = PerftTest {
        name: "Skewer Tactic",
        // White to move: the black king sits on d8 because a king on e8 would be
        // attacked by Re2, which is illegal. Expected values generated from C++
        // Stockfish.
        fen: "3k4/8/8/8/8/8/4R3/R3K3 w - - 0 1",
        is_chess960: false,
        expected: vec![27, 85, 2389, 12365],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_fork() {
    let test = PerftTest {
        name: "Knight Fork",
        fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![33, 930, 30542, 914790],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// COMPLEX MIDDLEGAME POSITIONS
// ===========================================================================

#[test]
fn perft_open_position() {
    let test = PerftTest {
        name: "Open Position",
        fen: "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
        is_chess960: false,
        expected: vec![37, 1323, 48159, 1741282],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_closed_position() {
    let test = PerftTest {
        name: "Closed Position",
        fen: "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![29, 837, 24742, 726359],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_ruy_lopez() {
    let test = PerftTest {
        name: "Ruy Lopez Opening",
        fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        is_chess960: false,
        expected: vec![30, 959, 28579, 908001],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_sicilian_dragon() {
    let test = PerftTest {
        name: "Sicilian Dragon",
        fen: "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![43, 1335, 57279, 1795344],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// EDGE CASES
// ===========================================================================

#[test]
fn perft_stalemate_position() {
    let test = PerftTest {
        name: "Stalemate Position",
        fen: "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        is_chess960: false,
        expected: vec![0], // Stalemate - no legal moves
    };
    run_perft_test(&test, 1);
}

#[test]
fn perft_only_king_moves() {
    let test = PerftTest {
        name: "Only King Can Move",
        fen: "r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        is_chess960: false,
        expected: vec![20, 400, 8885, 198572],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_many_pieces() {
    let test = PerftTest {
        name: "Many Pieces on Board",
        fen: "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![32, 893, 28360, 832412],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_few_pieces() {
    let test = PerftTest {
        name: "Few Pieces - Open Board",
        fen: "8/5k2/8/8/8/8/3K4/8 w - - 0 1",
        is_chess960: false,
        expected: vec![8, 64, 440, 2995, 21700],
    };
    run_perft_test(&test, 5);
}

// ===========================================================================
// BUGGY POSITIONS (historical bugs in move generators)
// ===========================================================================

#[test]
fn perft_avoid_illegal_ep() {
    let test = PerftTest {
        name: "Illegal EP Exposes King to Check",
        // dxe3 e.p. would clear the fourth rank and expose the black king to the rook.
        fen: "8/8/8/8/k2pP2R/8/8/4K3 b - e3 0 1",
        is_chess960: false,
        expected: vec![6, 89, 635, 10672],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_promotion_gives_check() {
    let test = PerftTest {
        name: "Promotion Gives Check",
        fen: "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        is_chess960: false,
        expected: vec![9, 40, 497, 2821],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// CHESS960 POSITIONS
// ===========================================================================

#[test]
fn perft_chess960_position_1() {
    let test = PerftTest {
        name: "Chess960 Starting Position (SP 518)",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1",
        is_chess960: true,
        expected: vec![20, 400, 8902, 197281],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_chess960_position_2() {
    let test = PerftTest {
        name: "Chess960 Position with Castling",
        fen: "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 0 1",
        is_chess960: true,
        expected: vec![21, 528, 12189, 326672],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// SPECIAL MOVE COMBINATIONS
// ===========================================================================

#[test]
fn perft_ep_and_castling() {
    let test = PerftTest {
        name: "En Passant and Castling Available",
        fen: "r3k2r/ppp2ppp/2n5/3pP3/1b1P4/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
        is_chess960: false,
        expected: vec![22, 746, 17302, 570685],
    };
    run_perft_test(&test, 4);
}

#[test]
fn perft_promotion_and_check() {
    let test = PerftTest {
        name: "Promotion Blocks Check",
        fen: "4k3/8/8/8/8/8/4p3/4K2R b K - 0 1",
        is_chess960: false,
        expected: vec![5, 60, 423, 6677],
    };
    run_perft_test(&test, 4);
}

// ===========================================================================
// STRESS TEST POSITIONS (for performance benchmarking)
// ===========================================================================

// Both stress tests together take about a second in release builds, so they are
// not #[ignore]d.
#[test]
fn perft_stress_test_1() {
    let test = PerftTest {
        name: "Stress Test - Many Captures",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        is_chess960: false,
        expected: vec![48, 2039, 97862, 4085603, 193690690],
    };
    run_perft_test(&test, 5);
}

#[test]
fn perft_stress_test_2() {
    let test = PerftTest {
        name: "Stress Test - Starting Position Depth 6",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        is_chess960: false,
        expected: vec![20, 400, 8902, 197281, 4865609, 119060324],
    };
    run_perft_test(&test, 6);
}
