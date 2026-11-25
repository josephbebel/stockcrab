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

//! Search Determinism Test Database
//!
//! A curated set of test positions with known best moves. The tests here check
//! that every FEN loads, that every `best_moves` entry is a legal move, and that
//! the catalogue has enough coverage; the
//! `best_moves` and `depth` fields are informational data. Real search checks
//! live in `search_determinism.rs`.
//!
//! Position Categories:
//! - Mate in 1: Forced mates
//! - Tactical: Forced wins, combinations
//! - Positional: Strategic moves
//! - Endgame: Winning/drawing endgames
//! - Opening: Theory positions
//! - Special: En passant, castling, promotion
//!
//! Test data: see `TestPosition` below (name, FEN, best moves, category, depth).

use stockcrab::bitboard;
use stockcrab::position::Position;

/// Test position structure
#[derive(Debug, Clone)]
pub struct TestPosition {
    pub name: &'static str,
    pub fen: &'static str,
    /// Best move(s) in UCI notation (e.g., "e2e4" or ["e2e4", "d2d4"] for alternatives)
    pub best_moves: Vec<&'static str>,
    pub category: &'static str,
    pub depth: i32, // Recommended search depth
}

/// Get all test positions
pub fn get_test_positions() -> Vec<TestPosition> {
    vec![
        // ========== MATE IN 1 ==========
        TestPosition {
            name: "Mate in 1 - Back Rank",
            fen: "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
            best_moves: vec!["a1a8"],
            category: "Mate in 1",
            depth: 5,
        },
        TestPosition {
            name: "Mate in 1 - Queen",
            fen: "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            best_moves: vec!["f7f8"],
            category: "Mate in 1",
            depth: 5,
        },
        TestPosition {
            name: "Mate in 1 - Rook",
            // The rook starts on a1, not h1: from h1 it would already be
            // attacking the black king down the open h-file, which is not a
            // legal position with White to move (and so is not a mate *in 1*).
            fen: "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
            best_moves: vec!["a1a8"],
            category: "Mate in 1",
            depth: 5,
        },
        TestPosition {
            name: "Mate in 1 - Qxf7",
            fen: "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            best_moves: vec!["h5f7"],
            category: "Mate in 1",
            depth: 10,
        },
        TestPosition {
            name: "Mate in 1 - Back Rank Queen",
            fen: "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",
            best_moves: vec!["d1d8"],
            category: "Mate in 1",
            depth: 10,
        },
        // ========== TACTICAL WINS ==========
        TestPosition {
            name: "Win Knight - Fork",
            fen: "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            best_moves: vec!["f3e5"],
            category: "Tactical",
            depth: 12,
        },
        TestPosition {
            name: "Win Piece - Pin",
            fen: "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            best_moves: vec!["f3e5"],
            category: "Tactical",
            depth: 12,
        },
        TestPosition {
            name: "Win Queen - Skewer",
            fen: "r2qkb1r/ppp2ppp/2n5/3Q4/2B5/8/PPP2PPP/RNB1K2R b KQkq - 0 1",
            best_moves: vec!["c6e5"],
            category: "Tactical",
            depth: 12,
        },
        TestPosition {
            name: "Bishop Pin - Ruy Lopez Setup",
            fen: "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            best_moves: vec!["f1b5"],
            category: "Tactical",
            depth: 12,
        },
        TestPosition {
            name: "Deflection Tactic",
            fen: "r2qkb1r/ppp2ppp/2n2n2/3pp1B1/1b1P4/2N2N2/PPP2PPP/R2QKB1R w KQkq - 0 1",
            best_moves: vec!["g5f6"],
            category: "Tactical",
            depth: 12,
        },
        // ========== OPENING THEORY ==========
        TestPosition {
            name: "Italian Game",
            fen: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            best_moves: vec!["g8f6", "f8c5"],
            category: "Opening",
            depth: 18,
        },
        TestPosition {
            name: "Ruy Lopez",
            fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            best_moves: vec!["a7a6", "g8f6"],
            category: "Opening",
            depth: 18,
        },
        TestPosition {
            name: "Sicilian Dragon",
            fen: "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
            best_moves: vec!["f2f3", "c1e3"],
            category: "Opening",
            depth: 18,
        },
        TestPosition {
            name: "Queen's Gambit Declined",
            fen: "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 1",
            best_moves: vec!["e7e6"],
            category: "Opening",
            depth: 18,
        },
        TestPosition {
            name: "King's Indian Defense",
            fen: "rnbqkb1r/pppppp1p/5np1/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 1",
            best_moves: vec!["f8g7"],
            category: "Opening",
            depth: 18,
        },
        // ========== ENDGAMES ==========
        TestPosition {
            name: "KPK - Winning",
            fen: "8/8/8/3k4/8/8/4P3/4K3 w - - 0 1",
            best_moves: vec!["e1f2"],
            category: "Endgame",
            depth: 30,
        },
        TestPosition {
            name: "KPK - Drawing",
            fen: "8/8/8/8/8/8/3k1p2/5K2 w - - 0 1",
            best_moves: vec!["f1f2"],
            category: "Endgame",
            depth: 30,
        },
        TestPosition {
            name: "Lucena Position",
            fen: "1K6/1P6/8/8/8/8/r7/k7 w - - 0 1",
            best_moves: vec!["b8c8"],
            category: "Endgame",
            depth: 25,
        },
        TestPosition {
            name: "Philidor Position",
            fen: "3k4/R7/3K4/8/8/8/r7/8 b - - 0 1",
            best_moves: vec!["a2a6"],
            category: "Endgame",
            depth: 25,
        },
        TestPosition {
            name: "Opposition Draw",
            fen: "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
            best_moves: vec!["e3d3", "e3f3"],
            category: "Endgame",
            depth: 30,
        },
        // ========== POSITIONAL ==========
        TestPosition {
            name: "Central Control",
            fen: "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
            best_moves: vec!["c2c4", "e2e3"],
            category: "Positional",
            depth: 18,
        },
        TestPosition {
            name: "Space Advantage",
            fen: "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
            best_moves: vec!["c2c4"],
            category: "Positional",
            depth: 18,
        },
        TestPosition {
            name: "Weak Squares",
            fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            best_moves: vec!["d2d3", "e1g1"],
            category: "Positional",
            depth: 18,
        },
        // ========== SPECIAL MOVES ==========
        TestPosition {
            name: "En Passant Available",
            fen: "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1",
            best_moves: vec!["d2d4"],
            category: "Special",
            depth: 18,
        },
        TestPosition {
            name: "Castling Available - Quiet Development",
            fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            best_moves: vec!["d2d3"],
            category: "Special",
            depth: 18,
        },
        TestPosition {
            name: "Promotion to Queen",
            fen: "8/4P3/8/2k5/8/8/8/4K3 w - - 0 1",
            best_moves: vec!["e7e8q"],
            category: "Special",
            depth: 20,
        },
        // ========== MORE TACTICAL POSITIONS ==========
        TestPosition {
            name: "Back Rank Weakness",
            fen: "r5k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1",
            best_moves: vec!["e1e8"],
            category: "Tactical",
            depth: 12,
        },
        TestPosition {
            name: "Queen Takes f7 - Mate",
            fen: "r2qkb1r/ppp2ppp/2n5/3Qb3/2B5/8/PPP2PPP/RNB1K2R w KQkq - 0 1",
            best_moves: vec!["d5f7"],
            category: "Tactical",
            depth: 12,
        },
        // ========== MORE ENDGAMES ==========
        TestPosition {
            name: "Rook vs Pawn - Winning",
            fen: "8/8/8/8/8/4k3/4p3/4K2R w - - 0 1",
            best_moves: vec!["h1h5"],
            category: "Endgame",
            depth: 25,
        },
        TestPosition {
            name: "Bishop and Wrong Rook Pawn - Draw",
            // Light-squared bishop with an h-pawn: it cannot control the h8
            // promotion square, so the black king holds the corner.
            fen: "7k/8/8/8/8/7P/8/5B1K w - - 0 1",
            best_moves: vec!["f1a6"],
            category: "Endgame",
            depth: 30,
        },
    ]
}

/// Try to load `fen`, returning the validation error message on failure.
///
/// Reporting the error instead of unwrapping is what makes the caller's
/// assertion useful: it names the offending position *and* says why `set_fen`
/// rejected it, rather than panicking inside the helper with no context.
fn verify_position_loads(fen: &str) -> Result<(), String> {
    let mut pos = Position::default();
    pos.set_fen(fen, false)
        .map(|_| ())
        .map_err(|e| e.message().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_positions_load() {
        bitboard::init();

        let positions = get_test_positions();
        println!("\nVerifying {} test positions can load...", positions.len());

        for pos in &positions {
            if let Err(why) = verify_position_loads(pos.fen) {
                panic!(
                    "Failed to load position {} ({}): {}",
                    pos.name, pos.fen, why
                );
            }
        }

        println!("✓ All {} positions verified", positions.len());
    }

    /// Every `best_moves` entry must at least be a legal move in its position.
    #[test]
    fn test_best_moves_are_legal() {
        bitboard::init();
        stockcrab::movegen::init();

        let mut problems = Vec::new();
        for tp in &get_test_positions() {
            let mut pos = Position::default();
            pos.set_fen(tp.fen, false)
                .unwrap_or_else(|e| panic!("{} ({}): {}", tp.name, tp.fen, e.message()));
            for uci in &tp.best_moves {
                if stockcrab::uci::UciEngine::to_move(&pos, uci) == stockcrab::types::Move::none() {
                    problems.push(format!(
                        "{}: best move {uci} is not legal in {}",
                        tp.name, tp.fen
                    ));
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn test_position_categories() {
        let positions = get_test_positions();

        let mut categories = std::collections::HashMap::new();
        for pos in &positions {
            *categories.entry(pos.category).or_insert(0) += 1;
        }

        println!("\nTest Suite Statistics:");
        for (category, count) in &categories {
            println!("  {}: {} positions", category, count);
        }
        println!("  Total: {} positions", positions.len());

        // Verify we have good coverage
        assert!(
            positions.len() >= 30,
            "Should have at least 30 test positions"
        );
        assert!(categories.len() >= 5, "Should cover at least 5 categories");
    }

    #[test]
    #[ignore] // Run manually to see position list
    fn test_list_all_positions() {
        let positions = get_test_positions();

        println!("\n{:<40} {:<15} Depth", "Position Name", "Category");
        println!("{}", "=".repeat(70));

        for pos in &positions {
            println!("{:<40} {:<15} {}", pos.name, pos.category, pos.depth);
        }
    }
}
