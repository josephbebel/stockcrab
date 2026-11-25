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

//! Safe public bitboard operations. Attack-table entry points establish initialization;
//! the engine uses private initialized lookups so search does not repeat those checks.

use crate::bitboard_fast as fast;
use crate::types::*;

pub use fast::pawn_attacks;
pub use fast::shift;
pub use fast::{
    FILE_A_BB, FILE_B_BB, FILE_C_BB, FILE_D_BB, FILE_E_BB, FILE_F_BB, FILE_G_BB, FILE_H_BB,
    PAWN_PAIR_BB, RANK_1_BB, RANK_2_BB, RANK_3_BB, RANK_4_BB, RANK_5_BB, RANK_6_BB, RANK_7_BB,
    RANK_8_BB,
};

#[inline]
pub fn pawn_attacks_from_square(sq: Square, color: Color) -> Bitboard {
    fast::init();
    fast::pawn_attacks_from_square(sq, color)
}
pub use fast::file_bb;
pub use fast::file_bb_sq;
pub use fast::init;
pub use fast::pawn_pair_bb;
pub use fast::pawn_single_push_bb;
pub use fast::pretty;
pub use fast::rank_bb;
pub use fast::rank_bb_sq;
pub use fast::square_bb_pub;

#[inline]
pub fn line_bb_pub(s1: Square, s2: Square) -> Bitboard {
    fast::init();
    fast::line_bb_pub(s1, s2)
}

#[inline]
pub fn between_bb_pub(s1: Square, s2: Square) -> Bitboard {
    fast::init();
    fast::between_bb_pub(s1, s2)
}

#[inline]
pub fn ray_pass_bb(s1: Square, s2: Square) -> Bitboard {
    fast::init();
    fast::ray_pass_bb(s1, s2)
}

#[inline]
pub fn diag_bb(s: Square) -> Bitboard {
    fast::init();
    fast::diag_bb(s)
}

#[inline]
pub fn antidiag_bb(s: Square) -> Bitboard {
    fast::init();
    fast::antidiag_bb(s)
}
pub use fast::constexpr_popcount;

#[inline]
pub fn aligned(s1: Square, s2: Square, s3: Square) -> bool {
    fast::init();
    fast::aligned(s1, s2, s3)
}

#[inline]
pub fn attacks_empty(pt: PieceType, sq: Square) -> Bitboard {
    fast::init();
    fast::attacks_empty(pt, sq)
}

#[inline]
pub fn attacks_bb_with_color(pt: PieceType, sq: Square, color: Color) -> Bitboard {
    fast::init();
    fast::attacks_bb_with_color(pt, sq, color)
}

#[inline]
pub fn attacks_bb(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    fast::init();
    fast::attacks_bb(pt, sq, occ)
}

#[inline]
pub fn attacks_bb_bishop(sq: Square, occ: Bitboard) -> Bitboard {
    fast::init();
    fast::attacks_bb_bishop(sq, occ)
}

#[inline]
pub fn attacks_bb_rook(sq: Square, occ: Bitboard) -> Bitboard {
    fast::init();
    fast::attacks_bb_rook(sq, occ)
}

#[inline]
pub fn attacks_bb_queen(sq: Square, occ: Bitboard) -> Bitboard {
    fast::init();
    fast::attacks_bb_queen(sq, occ)
}

#[inline]
pub fn both_attacks_bb(sq: Square, occ: Bitboard) -> (Bitboard, Bitboard) {
    fast::init();
    fast::both_attacks_bb(sq, occ)
}

#[inline]
pub fn attacks_bb_knight(sq: Square) -> Bitboard {
    fast::init();
    fast::attacks_bb_knight(sq)
}

#[inline]
pub fn attacks_bb_king(sq: Square) -> Bitboard {
    fast::init();
    fast::attacks_bb_king(sq)
}

#[inline]
pub fn attacks_bb_scan(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    fast::init();
    fast::attacks_bb_scan(pt, sq, occ)
}
pub use fast::distance_file;
pub use fast::distance_rank;
pub use fast::least_significant_square_bb;
pub use fast::more_than_one;
pub use fast::popcount;

#[inline]
pub fn distance_square(x: Square, y: Square) -> u8 {
    fast::init();
    fast::distance_square(x, y)
}
pub use fast::edge_distance;

/// Least significant square. Panics for an empty bitboard, including in release builds.
#[inline]
pub fn lsb(bb: Bitboard) -> Square {
    lsb_nonzero(core::num::NonZeroU64::new(bb).expect("lsb requires a nonempty bitboard"))
}

/// Least significant square of a bitboard whose nonempty invariant is carried by its type.
#[inline]
pub fn lsb_nonzero(bb: core::num::NonZeroU64) -> Square {
    fast::lsb(bb.get())
}

/// Most significant square. Panics for an empty bitboard.
#[inline]
pub fn msb(bb: Bitboard) -> Square {
    let bb = core::num::NonZeroU64::new(bb).expect("msb requires a nonempty bitboard");
    fast::msb(bb.get())
}

/// Remove and return the least significant square. Panics without changing an empty bitboard.
#[inline]
pub fn pop_lsb(bb: &mut Bitboard) -> Square {
    let square = lsb(*bb);
    *bb &= bb.wrapping_sub(1);
    square
}
