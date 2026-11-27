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

//! Evaluate module - Rust port of Stockfish's evaluate.h/evaluate.cpp
//!
//! This module provides position evaluation functions using NNUE (efficiently
//! updatable neural network) for chess position assessment.
//!
//! Evaluation uses a single NNUE `Network` and `AccumulatorCaches`, matching C++ Eval::evaluate.

use crate::position::Position;
use crate::types::{Color, PieceType, VALUE_TB_LOSS_IN_MAX_PLY, VALUE_TB_WIN_IN_MAX_PLY, Value};
use crate::uci::UciEngine;
// Re-export types needed by other modules (for use in search, engine, and thread modules).
// `Networks` is a type alias for `Network`, kept so call sites that use the upstream name compile unchanged.
pub use crate::nnue::network::{Network, Networks};
pub use crate::nnue::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

// The default net name MUST follow the format nn-[SHA256 first 12 digits].nnue
// for checksum validation. build.rs supplies it from config/upstream.env, where
// the network, upstream commit and bench expectation are pinned together.
// Matches `EvalFileDefaultName` in src/evaluate.h.
pub const EVAL_FILE_DEFAULT_NAME: &str = env!("STOCKCRAB_DEFAULT_NETWORK");

// ---------------------------------------------------------------------------
// EVALUATION FUNCTIONS
// ---------------------------------------------------------------------------

/// Main evaluation function - returns a static evaluation of the position
/// from the point of view of the side to move.
/// Matches C++ signature from src/evaluate.cpp `Eval::evaluate`.
pub fn evaluate(
    network: &Network,
    pos: &Position,
    accumulators: &mut AccumulatorStack,
    caches: &mut AccumulatorCaches,
) -> Value {
    debug_assert!(
        pos.checkers() == 0,
        "Cannot evaluate position with checkers"
    );

    let nnue = network.evaluate(pos, accumulators, caches);
    scale_evaluation(nnue, pos)
}

/// Applies material and rule50 scaling to the raw NNUE eval.
/// Matches C++ `scale_evaluation` in evaluate.cpp.
pub fn scale_evaluation(nnue: Value, pos: &Position) -> Value {
    let mut v = nnue;

    // Scale the evaluation by total material
    let material = 521 * pos.count(PieceType::Pawn) + pos.non_pawn_material();
    v += (material as i64 * v as i64 / 90649) as i32;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 189;

    // Guarantee that the evaluation does not hit the tablebase range
    v.clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

// ---------------------------------------------------------------------------
// TRACE SUPPORT
// ---------------------------------------------------------------------------

/// Generate a trace string for debugging evaluation.
/// Matches C++ implementation from src/evaluate.cpp `Eval::trace`.
///
/// Like evaluate(), but instead of returning a value, it returns
/// a string (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.
/// Trace scores are from white's point of view
pub fn trace(pos: &mut Position, network: &Network) -> String {
    if pos.checkers() != 0 {
        return "Final evaluation: none (in check)".to_string();
    }

    let mut accumulators = AccumulatorStack::new();
    let mut caches = AccumulatorCaches::new();
    caches.clear(network); // Initialize caches with network biases

    let mut result = String::new();

    // Add NNUE trace info with proper formatting
    result.push_str(&format!(
        "\n{}\n",
        crate::nnue::nnue_misc::trace(pos, network, &mut caches)
    ));

    let mut nnue = network.evaluate(pos, &mut accumulators, &mut caches);
    // Requires the side to move's perspective
    let mut s_v = scale_evaluation(nnue, pos);

    // Raw internal units (from the side to move's point of view)
    result.push_str(&format!(
        "NNUE evaluation          {:+} (side to move, internal units)\n",
        nnue
    ));

    // Convert to white's perspective
    if pos.side_to_move() == Color::Black {
        nnue = -nnue;
        s_v = -s_v;
    }

    // Format with proper precision matching C++ std::setprecision(2)
    let cp = UciEngine::to_cp(nnue, pos) as f64 * 0.01;
    result.push_str(&format!("NNUE evaluation        {:+.2} (white side)\n", cp));

    let final_cp = UciEngine::to_cp(s_v, pos) as f64 * 0.01;
    result.push_str(&format!(
        "Final evaluation      {:+.2} (white side) [with scaled NNUE, ...]\n",
        final_cp
    ));

    result
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_eval_file_default_name() {
        // Matches EvalFileDefaultName in src/evaluate.h
        assert_eq!(EVAL_FILE_DEFAULT_NAME, "nn-684d0e60e71d.nnue");
    }

    #[test]
    fn test_rule50_damping_formula() {
        crate::bitboard_fast::init();

        let mut pos = Position::default();
        pos.set_fen("8/8/8/8/8/8/8/K6k w - - 100 1", false).unwrap(); // High rule50 count

        // With kings only, material is zero. Check the actual
        // production entry point against upstream's /189 shuffling scale.
        assert_eq!(scale_evaluation(1000, &pos), 471);
        assert_eq!(scale_evaluation(-1000, &pos), -471);

        // Without a rule50 count the evaluation is left undamped.
        pos.set_fen("8/8/8/8/8/8/8/K6k w - - 0 1", false).unwrap();
        assert_eq!(scale_evaluation(1000, &pos), 1000);
        assert_eq!(scale_evaluation(-1000, &pos), -1000);
    }
}
