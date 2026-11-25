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

//! NNUE miscellaneous functionality for tracing and debugging
//!
//! This module provides the NNUE trace table (network contributions per layer-stack
//! bucket) printed by the `eval` command.

use core::fmt::Write;

use crate::position::Position;
use crate::types::*;
use crate::uci::UciEngine;

use super::network::Network;
use super::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};
use super::nnue_architecture::LAYER_STACKS;

/// Represents an evaluation file configuration
///
/// Owned strings for the default net name, the selected net name and the net description; the
/// struct is cloned when passed to the loader.
#[derive(Clone, Debug, Default)]
pub struct EvalFile {
    /// Default net name; see `crate::evaluate::EVAL_FILE_DEFAULT_NAME`
    pub default_name: String,
    /// Selected net name, either via uci option or default
    pub current: String,
    /// Net description extracted from the net file
    pub net_description: String,
}

/// Trace information for NNUE evaluation
#[derive(Clone, Debug)]
pub struct NnueEvalTrace {
    pub positional: [Value; LAYER_STACKS],
    pub correct_bucket: usize,
    /// Number of layer stacks stored in the network (the rows of `positional` that are set).
    pub layer_stacks: usize,
}

impl Default for NnueEvalTrace {
    fn default() -> Self {
        Self {
            positional: [VALUE_NONE; LAYER_STACKS],
            correct_bucket: 0,
            layer_stacks: 0,
        }
    }
}

/// Converts a Value into pawns, always keeping two decimals
fn format_cp_aligned_dot(v: Value, pos: &Position) -> String {
    let pawns = (0.01 * UciEngine::to_cp(v, pos) as f64).abs();

    let sign = if v < 0 {
        '-'
    } else if v > 0 {
        '+'
    } else {
        ' '
    };
    format!("{}{:6.2}", sign, pawns)
}

/// Returns a string with the table for evaluation values bucket by bucket.
pub fn trace(pos: &mut Position, network: &Network, caches: &mut AccumulatorCaches) -> String {
    let mut result = String::new();

    let mut accumulators = AccumulatorStack::default();
    accumulators.reset();

    let t = network.trace_evaluate(pos, &mut accumulators, caches);

    writeln!(
        &mut result,
        "NNUE network contributions (Normalized, {}",
        if pos.side_to_move() == Color::White {
            "White to move)"
        } else {
            "Black to move)"
        }
    )
    .unwrap();
    writeln!(&mut result, "+------------+------------+").unwrap();
    writeln!(&mut result, "|   Bucket   | Evaluation |").unwrap();
    writeln!(&mut result, "+------------+------------+").unwrap();

    for bucket in 0..t.layer_stacks {
        write!(&mut result, "|  {:<10}|  ", bucket).unwrap();
        write!(
            &mut result,
            "{}",
            format_cp_aligned_dot(t.positional[bucket], pos)
        )
        .unwrap();
        write!(&mut result, "   |").unwrap();

        if bucket == t.correct_bucket {
            write!(&mut result, " <-- this bucket is used").unwrap();
        }
        writeln!(&mut result).unwrap();
    }

    writeln!(&mut result, "+------------+------------+").unwrap();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_cp_aligned_dot() {
        crate::bitboard_fast::init();

        let mut pos = Position::default();
        pos.set_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        )
        .unwrap();

        // Positive value should be formatted with a leading '+'
        let formatted = format_cp_aligned_dot(378, &pos);
        assert!(
            formatted.starts_with('+'),
            "Expected leading '+', got: {}",
            formatted
        );

        // Negative value should be formatted with a leading '-'
        let formatted_negative = format_cp_aligned_dot(-567, &pos);
        assert!(
            formatted_negative.starts_with('-'),
            "Expected leading '-', got: {}",
            formatted_negative
        );
    }

    #[test]
    fn test_eval_file_default() {
        let file = EvalFile::default();
        assert!(file.default_name.is_empty());
        assert!(file.current.is_empty());
        assert!(file.net_description.is_empty());
    }

    #[test]
    fn test_nnue_eval_trace_default() {
        let trace = NnueEvalTrace::default();
        assert_eq!(trace.positional[0], VALUE_NONE);
        assert_eq!(trace.correct_bucket, 0);
    }
}
