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

//! Score module - Rust port of Stockfish's score.h/score.cpp
//!
//! This module provides a type-safe representation of chess engine scores,
//! which can be either regular scores, mate scores, or tablebase scores.

use crate::position::Position;
use crate::types::{VALUE_INFINITE, VALUE_MATE, VALUE_TB, Value, is_decisive};
use crate::uci::UciEngine;

// ---------------------------------------------------------------------------
// Score variants
// ---------------------------------------------------------------------------

/// Represents a checkmate score with the number of plies to mate
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Mate {
    /// Plies to mate; positive if the side to move mates, negative if it is being mated.
    pub plies: i32,
}

/// Represents a tablebase score with plies to the position and win/loss
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Tablebase {
    /// Distance to the TB result in plies, signed like `win` (negative for a loss).
    pub plies: i32,
    /// True for a TB win, false for a TB loss.
    pub win: bool,
}

/// Represents a regular internal score in centipawns
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InternalUnits {
    /// Centipawns, already converted from internal `Value` via `UciEngine::to_cp`.
    pub value: i32,
}

// ---------------------------------------------------------------------------
// Score type - Rust enum instead of C++ std::variant
// ---------------------------------------------------------------------------

/// A chess engine score that can be one of three types:
/// - Mate: checkmate in N plies
/// - Tablebase: tablebase win/loss in N plies
/// - InternalUnits: regular score in centipawns
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Score {
    /// Forced mate; `plies` is signed.
    Mate(Mate),
    /// Tablebase win or loss.
    Tablebase(Tablebase),
    /// Regular evaluation in centipawns.
    InternalUnits(InternalUnits),
}

impl Score {
    /// Create a Score from a Value and Position
    pub fn from_value(v: Value, pos: &Position) -> Self {
        debug_assert!(v > -VALUE_INFINITE && v < VALUE_INFINITE);

        if !is_decisive(v) {
            // Regular score
            Score::InternalUnits(InternalUnits {
                value: UciEngine::to_cp(v, pos),
            })
        } else if v.abs() <= VALUE_TB {
            // Tablebase score
            let distance = VALUE_TB - v.abs();
            if v > 0 {
                Score::Tablebase(Tablebase {
                    plies: distance,
                    win: true,
                })
            } else {
                Score::Tablebase(Tablebase {
                    plies: -distance,
                    win: false,
                })
            }
        } else {
            // Mate score
            let distance = VALUE_MATE - v.abs();
            Score::Mate(Mate {
                plies: if v > 0 { distance } else { -distance },
            })
        }
    }

    /// Check if this Score is of type T
    #[inline(always)]
    pub fn is<T: ScoreVariant>(&self) -> bool {
        T::is_variant(self)
    }

    /// Get the inner value if this Score is of type T
    #[inline(always)]
    pub fn get<T: ScoreVariant>(&self) -> Option<T> {
        T::get_variant(self)
    }

    /// Get the raw value of this score (for display/debugging)
    pub fn raw_value(&self) -> i32 {
        match self {
            Score::Mate(m) => m.plies,
            Score::Tablebase(tb) => tb.plies,
            Score::InternalUnits(iu) => iu.value,
        }
    }

    /// Check if this is a mate score
    #[inline(always)]
    pub fn is_mate(&self) -> bool {
        matches!(self, Score::Mate(_))
    }

    /// Check if this is a tablebase score
    #[inline(always)]
    pub fn is_tablebase(&self) -> bool {
        matches!(self, Score::Tablebase(_))
    }

    /// Check if this is an internal units score
    #[inline(always)]
    pub fn is_internal_units(&self) -> bool {
        matches!(self, Score::InternalUnits(_))
    }
}

impl Default for Score {
    fn default() -> Self {
        Score::InternalUnits(InternalUnits { value: 0 })
    }
}

// ---------------------------------------------------------------------------
// Trait for type-safe variant access
// ---------------------------------------------------------------------------

/// Trait for types that can be extracted from a Score
pub trait ScoreVariant: Sized {
    fn is_variant(score: &Score) -> bool;
    fn get_variant(score: &Score) -> Option<Self>;
}

impl ScoreVariant for Mate {
    #[inline(always)]
    fn is_variant(score: &Score) -> bool {
        matches!(score, Score::Mate(_))
    }

    #[inline(always)]
    fn get_variant(score: &Score) -> Option<Self> {
        match score {
            Score::Mate(m) => Some(*m),
            _ => None,
        }
    }
}

impl ScoreVariant for Tablebase {
    #[inline(always)]
    fn is_variant(score: &Score) -> bool {
        matches!(score, Score::Tablebase(_))
    }

    #[inline(always)]
    fn get_variant(score: &Score) -> Option<Self> {
        match score {
            Score::Tablebase(tb) => Some(*tb),
            _ => None,
        }
    }
}

impl ScoreVariant for InternalUnits {
    #[inline(always)]
    fn is_variant(score: &Score) -> bool {
        matches!(score, Score::InternalUnits(_))
    }

    #[inline(always)]
    fn get_variant(score: &Score) -> Option<Self> {
        match score {
            Score::InternalUnits(iu) => Some(*iu),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl core::fmt::Display for Mate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.plies > 0 {
            write!(f, "mate {}", (self.plies + 1) / 2)
        } else {
            write!(f, "mate {}", -((-self.plies) / 2))
        }
    }
}

impl core::fmt::Display for Tablebase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let prefix = if self.win { "tb win" } else { "tb loss" };
        write!(f, "{} {}", prefix, self.plies.abs())
    }
}

impl core::fmt::Display for InternalUnits {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cp {}", self.value)
    }
}

impl core::fmt::Display for Score {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Score::Mate(m) => write!(f, "{}", m),
            Score::Tablebase(tb) => write!(f, "{}", tb),
            Score::InternalUnits(iu) => write!(f, "{}", iu),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_position() -> Position {
        crate::bitboard::init();
        let mut pos = Position::default();
        pos.set_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        )
        .unwrap();
        pos
    }

    #[test]
    fn test_score_from_regular_value() {
        let pos = setup_position();
        let score = Score::from_value(100, &pos);

        assert!(score.is::<InternalUnits>());
        assert!(!score.is::<Mate>());
        assert!(!score.is::<Tablebase>());

        if let Some(iu) = score.get::<InternalUnits>() {
            // Value should be converted to centipawns
            assert_eq!(iu.value, UciEngine::to_cp(100, &pos));
        } else {
            panic!("Expected InternalUnits");
        }
    }

    #[test]
    fn test_score_from_mate_value() {
        let pos = setup_position();

        // Mate in 5 plies
        let mate_value = VALUE_MATE - 5;
        let score = Score::from_value(mate_value, &pos);

        assert!(score.is::<Mate>());
        if let Some(m) = score.get::<Mate>() {
            assert_eq!(m.plies, 5);
        } else {
            panic!("Expected Mate");
        }

        // Mated in 3 plies
        let mated_value = -VALUE_MATE + 3;
        let score = Score::from_value(mated_value, &pos);

        assert!(score.is::<Mate>());
        if let Some(m) = score.get::<Mate>() {
            assert_eq!(m.plies, -3);
        } else {
            panic!("Expected Mate");
        }
    }

    #[test]
    fn test_score_from_tablebase_value() {
        let pos = setup_position();

        // TB win in 10 plies
        let tb_win = VALUE_TB - 10;
        let score = Score::from_value(tb_win, &pos);

        assert!(score.is::<Tablebase>());
        if let Some(tb) = score.get::<Tablebase>() {
            assert_eq!(tb.plies, 10);
            assert!(tb.win);
        } else {
            panic!("Expected Tablebase");
        }

        // TB loss in 15 plies
        let tb_loss = -VALUE_TB + 15;
        let score = Score::from_value(tb_loss, &pos);

        assert!(score.is::<Tablebase>());
        if let Some(tb) = score.get::<Tablebase>() {
            assert_eq!(tb.plies, -15);
            assert!(!tb.win);
        } else {
            panic!("Expected Tablebase");
        }
    }

    #[test]
    fn test_score_display() {
        let mate_score = Score::Mate(Mate { plies: 5 });
        assert_eq!(format!("{}", mate_score), "mate 3");

        let mated_score = Score::Mate(Mate { plies: -4 });
        assert_eq!(format!("{}", mated_score), "mate -2");

        let tb_win = Score::Tablebase(Tablebase {
            plies: 10,
            win: true,
        });
        assert_eq!(format!("{}", tb_win), "tb win 10");

        let tb_loss = Score::Tablebase(Tablebase {
            plies: -15,
            win: false,
        });
        assert_eq!(format!("{}", tb_loss), "tb loss 15");

        let regular = Score::InternalUnits(InternalUnits { value: 25 });
        assert_eq!(format!("{}", regular), "cp 25");
    }

    #[test]
    fn test_score_default() {
        let score = Score::default();
        assert!(score.is::<InternalUnits>());
        if let Some(iu) = score.get::<InternalUnits>() {
            assert_eq!(iu.value, 0);
        }
    }

    #[test]
    fn test_score_type_checks() {
        let mate_score = Score::Mate(Mate { plies: 5 });
        assert!(mate_score.is_mate());
        assert!(!mate_score.is_tablebase());
        assert!(!mate_score.is_internal_units());

        let tb_score = Score::Tablebase(Tablebase {
            plies: 10,
            win: true,
        });
        assert!(!tb_score.is_mate());
        assert!(tb_score.is_tablebase());
        assert!(!tb_score.is_internal_units());

        let regular_score = Score::InternalUnits(InternalUnits { value: 100 });
        assert!(!regular_score.is_mate());
        assert!(!regular_score.is_tablebase());
        assert!(regular_score.is_internal_units());
    }
}
