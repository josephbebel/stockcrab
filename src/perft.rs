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

//! Perft (performance test) module for move generation verification
//!
//! This module provides utilities to verify move generation by counting
//! all leaf nodes up to a given depth. This is used for testing and
//! benchmarking the move generator.

use crate::movegen::{GenType, generate};
use crate::position::{Position, PositionSetError, ScopedMove};
use crate::types::{Depth, MAX_HIST};
use crate::uci::UciEngine;
use std::io::{self, Write};

// ---------------------------------------------------------------------------
// PERFT IMPLEMENTATION
// ---------------------------------------------------------------------------

/// Generic perft function that counts leaf nodes.
/// ROOT selects the root-level variant, which prints each root move and its node count.
fn perft_internal<const ROOT: bool>(pos: &mut Position, depth: Depth) -> u64 {
    let mut nodes = 0u64;

    // Optimization: at depth 2, just count legal moves instead of recursing
    let leaf = depth == 2;

    let moves = generate(pos, GenType::All);

    for &m in &moves {
        let cnt;

        if ROOT && depth <= 1 {
            cnt = 1;
            nodes += 1;
        } else {
            // SAFETY: generated moves are legal; ScopedMove undoes the move even if a
            // recursive call unwinds.
            let mut moved = unsafe { ScopedMove::new(pos, m) };

            if leaf {
                // At depth 2, just count the number of legal moves
                cnt = generate(&moved, GenType::All).len() as u64;
            } else {
                // Recurse deeper
                cnt = perft_internal::<false>(&mut moved, depth - 1);
            }

            nodes += cnt;
            drop(moved);
        }

        if ROOT {
            // Print move and count for root moves
            UciEngine::write_stdout_bytes(
                format!("{}: {}\n", UciEngine::move_str(m, pos.is_chess960()), cnt).as_bytes(),
            );
            io::stdout().flush().unwrap();
        }
    }

    nodes
}

/// Public perft function that counts all leaf nodes up to the given depth.
/// This is the main entry point for perft testing. Like C++, a depth of 1 or less
/// (`go perft -1` reaches here) counts the root moves.
///
/// Panics unless `depth < MAX_HIST`: deeper lines would outrun the position's state history.
pub fn perft(pos: &mut Position, depth: Depth) -> u64 {
    assert!(
        depth < MAX_HIST as Depth,
        "perft depth must be below {MAX_HIST}"
    );
    perft_internal::<true>(pos, depth)
}

/// Convenience function that takes a FEN string and runs perft.
/// Returns the total number of leaf nodes, or a `PositionSetError` if the
/// FEN string could not be parsed (mirrors the C++
/// `std::variant<u64, PositionSetError>` return type).
pub fn perft_from_fen(fen: &str, depth: Depth, is_chess960: bool) -> Result<u64, PositionSetError> {
    // Ensure all tables are initialized before running perft
    crate::bitboard::init();

    let mut pos = Position::default();
    pos.set_fen(fen, is_chess960)?;

    Ok(perft(&mut pos, depth))
}

// ---------------------------------------------------------------------------
// BENCHMARKING SUPPORT
// ---------------------------------------------------------------------------

/// Structure to hold perft results for a specific position
#[derive(Debug, Clone)]
pub struct PerftResult {
    pub fen: String,
    pub depth: Depth,
    pub nodes: u64,
    pub time_ms: u128,
}

/// Run perft on a position and return detailed results
pub fn perft_timed(fen: &str, depth: Depth, is_chess960: bool) -> PerftResult {
    use std::time::Instant;

    let start = Instant::now();
    let nodes = perft_from_fen(fen, depth, is_chess960).expect("perft_timed: invalid FEN string");
    let time_ms = start.elapsed().as_millis();

    PerftResult {
        fen: fen.to_string(),
        depth,
        nodes,
        time_ms,
    }
}

/// Standard perft test positions with known node counts
pub fn standard_positions() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        // Starting position
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            vec![20, 400, 8902, 197281, 4865609, 119060324],
        ),
        // Kiwipete position - good for testing special moves
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            vec![48, 2039, 97862, 4085603, 193690690],
        ),
        // Position 3
        (
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            vec![14, 191, 2812, 43238, 674624, 11030083],
        ),
        // Position 4 - testing promotions
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            vec![6, 264, 9467, 422333, 15833292],
        ),
        // Position 5 - complex middle game
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            vec![44, 1486, 62379, 2103487, 89941194],
        ),
        // Position 6 - endgame
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            vec![46, 2079, 89890, 3894594, 164075551],
        ),
    ]
}

/// Run perft tests on standard positions and verify correctness
pub fn run_standard_tests(max_depth: Depth) -> bool {
    let positions = standard_positions();
    let mut all_passed = true;

    for (fen, expected_nodes) in positions {
        println!("\nTesting position: {}", fen);
        println!("=====================================");

        for depth in 1..=max_depth.min(expected_nodes.len() as i32) {
            let result = perft_timed(fen, depth, false);
            let expected = expected_nodes[(depth - 1) as usize];
            let passed = result.nodes == expected;

            println!(
                "Depth {}: {} nodes in {}ms (expected: {}) {}",
                depth,
                result.nodes,
                result.time_ms,
                expected,
                if passed { "✓" } else { "✗ FAILED" }
            );

            if !passed {
                all_passed = false;
                eprintln!(
                    "ERROR: Perft mismatch at depth {} for position: {}",
                    depth, fen
                );
                eprintln!("  Expected: {}, Got: {}", expected, result.nodes);
            }
        }
    }

    all_passed
}

// ---------------------------------------------------------------------------
// DIVIDE COMMAND SUPPORT
// ---------------------------------------------------------------------------

/// Divide command - shows perft results for each root move. Panics unless `depth < MAX_HIST`.
pub fn divide(pos: &mut Position, depth: Depth) -> u64 {
    assert!(
        depth < MAX_HIST as Depth,
        "perft depth must be below {MAX_HIST}"
    );
    if depth <= 0 {
        return 1;
    }

    let mut total = 0u64;

    let moves = generate(pos, GenType::All);

    println!("\n Move     Nodes");
    println!("----------------");

    for &m in &moves {
        // SAFETY: generated moves are legal; the guard restores the position
        // before its stack state is released, including if output unwinds.
        let mut moved = unsafe { ScopedMove::new(pos, m) };

        // perft_internal::<false> has no depth-1 base case, so depth 2 counts the
        // replies directly.
        let nodes = if depth == 1 {
            1
        } else if depth == 2 {
            generate(&moved, GenType::All).len() as u64
        } else {
            perft_internal::<false>(&mut moved, depth - 1)
        };

        total += nodes;
        println!(
            "{:6} {:8}",
            UciEngine::move_str(m, moved.is_chess960()),
            nodes
        );

        drop(moved);
    }

    println!("----------------");
    println!("Total: {:9}", total);

    total
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard_fast as bitboard;

    #[test]
    fn test_perft_starting_position() {
        bitboard::init();

        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        // Test depth 1-3 for starting position
        assert_eq!(perft_from_fen(fen, 1, false).unwrap(), 20);
        assert_eq!(perft_from_fen(fen, 2, false).unwrap(), 400);
        assert_eq!(perft_from_fen(fen, 3, false).unwrap(), 8902);
    }

    #[test]
    fn test_perft_simple_position() {
        bitboard::init();

        // Simple endgame position
        let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

        assert_eq!(perft_from_fen(fen, 1, false).unwrap(), 14);
        assert_eq!(perft_from_fen(fen, 2, false).unwrap(), 191);
    }

    #[test]
    fn test_divide() {
        bitboard::init();

        let mut pos = Position::default();
        pos.set_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        )
        .unwrap();

        // Divide at depth 1 should give 20 (number of legal moves)
        let total = divide(&mut pos, 1);
        assert_eq!(total, 20);
        assert_eq!(divide(&mut pos, 2), 400);
        assert_eq!(divide(&mut pos, 3), 8902);
    }

    #[test]
    fn test_perft_result() {
        bitboard::init();

        let result = perft_timed(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            2,
            false,
        );

        assert_eq!(result.depth, 2);
        assert_eq!(result.nodes, 400);
    }
}
