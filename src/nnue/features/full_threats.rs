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

// NNUE Full Threats feature implementation
// Ported from Stockfish (SFNNv16 full-threats feature set); the tables are const-initialised.

use super::super::nnue_common::{
    FeatureIndexList, FeatureIndexType, IndexType, MAX_FEATURE_INDEXES, ORIENT_TBL,
};
use crate::bitboard_fast::{attacks_bb, pop_lsb, shift};
use crate::position::Position;
use crate::types::{
    Bitboard, Color, Direction, DirtyThreats, NORTH_EAST, NORTH_WEST, PIECE_NB, PIECE_TYPE_NB,
    Piece, PieceType, SOUTH_EAST, SOUTH_WEST, SQUARE_NB, Square, make_piece,
};

/// Table entry / feature-index arithmetic type (u32 like the C++ `IndexType`; half the
/// table footprint of `usize`).
type TableIdx = u32;

/// Number of valid targets for each piece type
/// Layout: [NoPiece, WPawn, WKnight, WBishop, WRook, WQueen, WKing, ?, ?, BPawn, ...]
///
/// SFNNv16: pawn diagonal threats only target knights and rooks (pawn-pawn
/// relationships are handled by the PP_3Wide feature set), so pawns have
/// 4 valid targets (knight/rook x own/enemy).
pub const NUM_VALID_TARGETS: [i32; PIECE_NB] = [0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0];

// ============================================================================
// Compile-time attack calculation helpers (per 73b3b185)
// ============================================================================

/// Const fn to check if destination is safe (within board)
const fn const_safe_destination(sq: usize, step: i32) -> u64 {
    let to = sq as i32 + step;
    if to >= 0 && to < 64 {
        let file_dist = ((to % 8) - (sq as i32 % 8)).abs();
        let rank_dist = ((to / 8) - (sq as i32 / 8)).abs();
        // Knight offsets only: file and rank distance must each be at most 2.
        if file_dist <= 2 && rank_dist <= 2 {
            1u64 << to
        } else {
            0
        }
    } else {
        0
    }
}

/// Const fn for knight pseudo-attacks
const fn const_knight_attacks(sq: usize) -> u64 {
    const_safe_destination(sq, 17)
        | const_safe_destination(sq, 15)
        | const_safe_destination(sq, 10)
        | const_safe_destination(sq, 6)
        | const_safe_destination(sq, -17)
        | const_safe_destination(sq, -15)
        | const_safe_destination(sq, -10)
        | const_safe_destination(sq, -6)
}

/// Const fn for king pseudo-attacks
const fn const_king_attacks(sq: usize) -> u64 {
    let file = sq % 8;
    let rank = sq / 8;
    let mut attacks = 0u64;

    // All 8 directions
    if rank < 7 {
        attacks |= 1u64 << (sq + 8);
    }
    if rank > 0 {
        attacks |= 1u64 << (sq - 8);
    }
    if file < 7 {
        attacks |= 1u64 << (sq + 1);
    }
    if file > 0 {
        attacks |= 1u64 << (sq - 1);
    }
    if rank < 7 && file < 7 {
        attacks |= 1u64 << (sq + 9);
    }
    if rank < 7 && file > 0 {
        attacks |= 1u64 << (sq + 7);
    }
    if rank > 0 && file < 7 {
        attacks |= 1u64 << (sq - 7);
    }
    if rank > 0 && file > 0 {
        attacks |= 1u64 << (sq - 9);
    }

    attacks
}

/// Const fn for sliding attack (empty board)
const fn const_sliding_attack(deltas: &[i32], sq: usize) -> u64 {
    let mut attack = 0u64;
    let mut i = 0;
    while i < deltas.len() {
        let d = deltas[i];
        let mut s = sq as i32 + d;
        let mut prev = sq as i32;
        while s >= 0 && s < 64 {
            // Check for wraparound
            let file_diff = (s % 8 - prev % 8).abs();
            if file_diff > 1 {
                break;
            }
            attack |= 1u64 << s;
            prev = s;
            s += d;
        }
        i += 1;
    }
    attack
}

/// Const fn for rook pseudo-attacks (empty board)
const fn const_rook_attacks(sq: usize) -> u64 {
    const_sliding_attack(&[8, -8, 1, -1], sq)
}

/// Const fn for bishop pseudo-attacks (empty board)
const fn const_bishop_attacks(sq: usize) -> u64 {
    const_sliding_attack(&[9, 7, -9, -7], sq)
}

/// Const fn for queen pseudo-attacks (empty board)
const fn const_queen_attacks(sq: usize) -> u64 {
    const_rook_attacks(sq) | const_bishop_attacks(sq)
}

/// Const fn for white pawn attacks
const fn const_wpawn_attacks(sq: usize) -> u64 {
    let file = sq % 8;
    let rank = sq / 8;
    let mut attacks = 0u64;
    if rank < 7 {
        if file < 7 {
            attacks |= 1u64 << (sq + 9);
        }
        if file > 0 {
            attacks |= 1u64 << (sq + 7);
        }
    }
    attacks
}

/// Const fn for black pawn attacks
const fn const_bpawn_attacks(sq: usize) -> u64 {
    let file = sq % 8;
    let rank = sq / 8;
    let mut attacks = 0u64;
    if rank > 0 {
        if file < 7 {
            attacks |= 1u64 << (sq - 7);
        }
        if file > 0 {
            attacks |= 1u64 << (sq - 9);
        }
    }
    attacks
}

/// Const fn to get pseudo-attacks for a piece on a square (empty board)
/// Note: SFNNv16 reverted the pawn tables to plain diagonal attacks; the C++
/// offset/index_lut2 construction uses `Attacks::PseudoAttacks[color]` for pawns
/// (pawn pushes are no longer threat features).
const fn const_pseudo_attacks(piece: u8, sq: usize) -> u64 {
    // piece values: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
    // White pieces: 1-6, Black pieces: 9-14
    let pt = piece & 7; // Mask to get piece type
    let is_black = piece >= 8;

    match pt {
        1 => {
            // Pawn (diagonal attacks only)
            if is_black {
                const_bpawn_attacks(sq)
            } else {
                const_wpawn_attacks(sq)
            }
        }
        2 => const_knight_attacks(sq),
        3 => const_bishop_attacks(sq),
        4 => const_rook_attacks(sq),
        5 => const_queen_attacks(sq),
        6 => const_king_attacks(sq),
        _ => 0,
    }
}

/// Const fn popcount
const fn const_popcount(mut bb: u64) -> u32 {
    let mut count = 0;
    while bb != 0 {
        count += 1;
        bb &= bb - 1;
    }
    count
}

// ============================================================================
// Compile-time generated lookup tables (per 73b3b185)
// ============================================================================

/// All pieces for iteration (using u8 values)
const ALL_PIECES: [u8; 12] = [
    1, 2, 3, 4, 5, 6, // W_Pawn=1, W_Knight=2, W_Bishop=3, W_Rook=4, W_Queen=5, W_King=6
    9, 10, 11, 12, 13,
    14, // B_Pawn=9, B_Knight=10, B_Bishop=11, B_Rook=12, B_Queen=13, B_King=14
];

/// The final index is calculated from summing data found in these two LUTs
#[repr(C, align(64))]
pub struct ThreatLuts {
    /// \[attacker\]\[attacked\]\[from < to\]
    pub lut1: [[[TableIdx; 2]; PIECE_NB]; PIECE_NB],
    /// \[pt\]\[from\]\[to\]
    pub lut2: [[[u16; SQUARE_NB]; SQUARE_NB]; PIECE_TYPE_NB],
}

/// Compute both lookup tables at compile time (C++ `init_threat_luts`).
///
/// `lut2[pt][from][to]` is the offset of `from` within the attacker's block plus the rank of
/// `to` among its attacks; the block size of each piece type then spaces the `lut1` features.
/// Excluded / semi-excluded pairs store `Dimensions` in `lut1` so that the summed index in
/// make_index exceeds `Dimensions` and is filtered out by the caller (push_back_if_lt), avoiding
/// a branch in make_index.
const fn init_threat_luts() -> ThreatLuts {
    let mut tables = ThreatLuts {
        lut1: [[[0; 2]; PIECE_NB]; PIECE_NB],
        lut2: [[[0; SQUARE_NB]; SQUARE_NB]; PIECE_TYPE_NB],
    };
    let mut offsets = [0 as TableIdx; PIECE_TYPE_NB];

    let mut pt = PieceType::Pawn as usize;
    while pt <= PieceType::King as usize {
        let mut offset: u16 = 0;
        let mut from_idx = 0;
        while from_idx < 64 {
            // White pieces stand for every colour; a pawn attacks forwards, so a
            // target below `from` can only be a black pawn's.
            let (attacks_up, attacks_down) = if pt != PieceType::Pawn as usize {
                let attacks = const_pseudo_attacks(pt as u8, from_idx);
                (attacks, attacks)
            } else {
                (
                    const_pseudo_attacks(Piece::W_Pawn as u8, from_idx),
                    const_pseudo_attacks(Piece::B_Pawn as u8, from_idx),
                )
            };
            let mut to_idx = 0;
            while to_idx < 64 {
                let attacks = if to_idx < from_idx {
                    attacks_down
                } else {
                    attacks_up
                };
                let to_bb = 1u64 << to_idx;
                tables.lut2[pt][from_idx][to_idx] =
                    offset + const_popcount(to_bb.wrapping_sub(1) & attacks) as u16;
                to_idx += 1;
            }

            if pt != PieceType::Pawn as usize || (from_idx >= 8 && from_idx <= 55) {
                offset += const_popcount(attacks_up) as u16;
            }
            from_idx += 1;
        }

        offsets[pt] = offset as TableIdx;
        pt += 1;
    }

    let mut cumulative_offset: TableIdx = 0;
    let mut attacker_i = 0;
    while attacker_i < 12 {
        let attacker_u8 = ALL_PIECES[attacker_i];
        let attacker_type = (attacker_u8 & 7) as usize;
        let cumulative_piece_offset = offsets[attacker_type];
        let valid_targets = NUM_VALID_TARGETS[attacker_u8 as usize] as TableIdx;

        let mut attacked_i = 0;
        while attacked_i < 12 {
            let attacked_u8 = ALL_PIECES[attacked_i];
            let attacked_type = (attacked_u8 & 7) as usize;
            let attacked_color = if attacked_u8 >= 8 { 1 } else { 0 };
            let enemy = (attacker_u8 ^ attacked_u8) == 8;

            let map_val = FullThreats::MAP[attacker_type - 1][attacked_type - 1];
            let semi_excluded = attacker_type == attacked_type && (enemy || attacker_type != 1);

            let feature = if map_val >= 0 {
                cumulative_offset
                    + (attacked_color * (valid_targets / 2) + map_val as TableIdx)
                        * cumulative_piece_offset
            } else {
                0
            };

            let excluded = map_val < 0;
            tables.lut1[attacker_u8 as usize][attacked_u8 as usize][0] = if excluded {
                FullThreats::DIMENSIONS as TableIdx
            } else {
                feature
            };
            tables.lut1[attacker_u8 as usize][attacked_u8 as usize][1] =
                if excluded || semi_excluded {
                    FullThreats::DIMENSIONS as TableIdx
                } else {
                    feature
                };

            attacked_i += 1;
        }

        cumulative_offset += valid_targets * cumulative_piece_offset;
        attacker_i += 1;
    }

    tables
}

/// The threat-feature lookup tables, built at compile time.
pub static THREAT_LUTS: ThreatLuts = init_threat_luts();

/// The threat feature set (C++ `Features::FullThreats`): one feature per attacker, attacked
/// piece and squares.
pub struct FullThreats;

impl FullThreats {
    /// Hash value embedded in the evaluation file
    pub const HASH_VALUE: u32 = 0x2c41eaea;

    /// Number of feature dimensions
    /// (SFNNv16: pawn pushes and pawn-pawn threats moved to PP_3Wide)
    pub const DIMENSIONS: IndexType = 59808;
    /// `DIMENSIONS` in the index arithmetic type.
    const DIMENSIONS_T: TableIdx = Self::DIMENSIONS as TableIdx;

    /// Maximum number of simultaneously active features
    /// (C++ now sizes the IndexList as `ValueList<u16, 256>`)
    pub const MAX_ACTIVE_DIMENSIONS: usize = 256;

    /// Piece-type to piece-type mapping for valid threats
    /// map[attacker_type - 1][attacked_type - 1], -1 means excluded
    pub const MAP: [[i8; PIECE_TYPE_NB - 2]; PIECE_TYPE_NB - 2] = [
        [-1, 0, -1, 1, -1, -1],   // Pawn attacks (knights and rooks only)
        [0, 1, 2, 3, 4, -1],      // Knight attacks
        [0, 1, 2, 3, -1, -1],     // Bishop attacks
        [0, 1, 2, 3, -1, -1],     // Rook attacks
        [0, 1, 2, 3, 4, -1],      // Queen attacks
        [-1, -1, -1, -1, -1, -1], // King attacks (king is no longer a threat attacker)
    ];

    /// Make index for a threat feature (per f4244e13: perspective as parameter)
    ///
    /// Index of a feature for a given king position and another piece on some square
    #[inline(always)]
    pub fn make_index(
        perspective: Color,
        attacker: Piece,
        from: Square,
        to: Square,
        attacked: Piece,
        ksq: Square,
    ) -> TableIdx {
        let orientation = ORIENT_TBL[ksq as usize] ^ (56 * perspective as u8);
        Self::make_index_with_orientation(perspective, attacker, from, to, attacked, orientation)
    }

    /// Make index with precomputed orientation (per 3ae76847: avoid recomputing orientation)
    /// Tables are now compile-time initialized (per 73b3b185)
    #[inline(always)]
    pub fn make_index_with_orientation(
        perspective: Color,
        attacker: Piece,
        from: Square,
        to: Square,
        attacked: Piece,
        orientation: u8,
    ) -> TableIdx {
        let from_oriented = (from as u8) ^ orientation;
        let to_oriented = (to as u8) ^ orientation;

        // Flip pieces for perspective
        let swap = 8 * perspective as u8;
        let attacker_oriented = (attacker as u8) ^ swap;
        let attacked_oriented = (attacked as u8) ^ swap;

        // Excluded / semi-excluded pairs store `Dimensions` in INDEX_LUT1, so the
        // resulting index exceeds `Dimensions` and is filtered by the caller.
        let less_than = (from_oriented < to_oriented) as usize;

        THREAT_LUTS.lut1[attacker_oriented as usize][attacked_oriented as usize][less_than]
            + THREAT_LUTS.lut2[attacker as usize & 7][from_oriented as usize][to_oriented as usize]
                as TableIdx
    }

    /// Append active threat indices for a position (in ascending order)
    /// Per 3ae76847: precompute orientation once for all calls
    ///
    /// Get a list of indices for active features in ascending order
    #[inline(never)]
    pub fn append_active_indices(
        perspective: Color,
        pos: &Position,
        active: &mut FeatureIndexList,
    ) {
        let ksq = pos.square_of(perspective, PieceType::King);
        // Per 3ae76847: precompute orientation once
        let orientation = ORIENT_TBL[ksq as usize] ^ (56 * perspective as u8);
        let occupied = pos.pieces();

        // Sets of valid target pieces for each attacker category.
        // SFNNv16: pawns no longer threaten pawns (see PP_3Wide).
        let pawn_targets = pos.pieces_of_types(&[PieceType::Knight, PieceType::Rook]);
        let minor_slider_targets = pos.pieces_of_types(&[
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
        ]);
        let queen_targets = pos.pieces_of_types(&[
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ]);

        // Helper to process a bitboard of attacked squares reached via `attk_dir`.
        let process_pawn_attacks =
            |mut attacks: Bitboard,
             attk_dir: Direction,
             attacker: Piece,
             active: &mut FeatureIndexList| {
                while attacks != 0 {
                    let to = pop_lsb(&mut attacks);
                    let from = Square::from(((to as i8) - attk_dir) as u8);
                    let attacked = pos.piece_on(to);
                    let index = Self::make_index_with_orientation(
                        perspective,
                        attacker,
                        from,
                        to,
                        attacked,
                        orientation,
                    );
                    active.push_if(index as FeatureIndexType, index < Self::DIMENSIONS_T);
                }
            };

        for color_idx in 0u8..2 {
            let c = if ((perspective as u8) ^ color_idx) == 0 {
                Color::White
            } else {
                Color::Black
            };

            // Pawn threats
            {
                let attacker = make_piece(c, PieceType::Pawn);
                let c_pawns = pos.pieces_color_types(c, &[PieceType::Pawn]);

                if c == Color::White {
                    process_pawn_attacks(
                        shift(c_pawns, NORTH_EAST) & pawn_targets,
                        NORTH_EAST,
                        attacker,
                        active,
                    );
                    process_pawn_attacks(
                        shift(c_pawns, NORTH_WEST) & pawn_targets,
                        NORTH_WEST,
                        attacker,
                        active,
                    );
                } else {
                    process_pawn_attacks(
                        shift(c_pawns, SOUTH_WEST) & pawn_targets,
                        SOUTH_WEST,
                        attacker,
                        active,
                    );
                    process_pawn_attacks(
                        shift(c_pawns, SOUTH_EAST) & pawn_targets,
                        SOUTH_EAST,
                        attacker,
                        active,
                    );
                }
            }

            // Knight, Bishop, Rook, Queen threats (king is no longer a threat attacker)
            for pt_idx in (PieceType::Knight as u8)..(PieceType::King as u8) {
                let pt = PieceType::from(pt_idx);
                let attacker = make_piece(c, pt);
                let mut bb = pos.pieces_color_types(c, &[pt]);
                let targets = if pt == PieceType::Knight || pt == PieceType::Queen {
                    queen_targets
                } else {
                    minor_slider_targets
                };

                while bb != 0 {
                    let from = pop_lsb(&mut bb);
                    let mut attacks = attacks_bb(pt, from, occupied) & targets;

                    while attacks != 0 {
                        let to = pop_lsb(&mut attacks);
                        let attacked = pos.piece_on(to);
                        let index = Self::make_index_with_orientation(
                            perspective,
                            attacker,
                            from,
                            to,
                            attacked,
                            orientation,
                        );
                        active.push_if(index as FeatureIndexType, index < Self::DIMENSIONS_T);
                    }
                }
            }
        }
    }

    /// Append the threat feature indices that changed between two positions.
    ///
    /// Scalar port of the C++ `FullThreats::append_changed_indices`. Iterates the
    /// packed [`DirtyThreats`] list produced by make/undo move, computes each
    /// feature index, and routes it to `added` or `removed` (filtering indices
    /// that are `>= Dimensions`, i.e. excluded/semi-excluded pairs).
    #[inline(never)]
    pub(crate) fn append_changed_indices(
        perspective: Color,
        ksq: Square,
        diff: &DirtyThreats,
        removed: &mut FeatureIndexList,
        added: &mut FeatureIndexList,
    ) {
        // Each dirty entry consumes at most one slot of one list. The unchecked pushes below rely on
        // the in-crate callers passing freshly cleared lists (checked in debug builds).
        debug_assert!(
            removed.len() + diff.list.len <= MAX_FEATURE_INDEXES
                && added.len() + diff.list.len <= MAX_FEATURE_INDEXES
        );
        let orient = (ORIENT_TBL[ksq as usize] ^ (56 * perspective as u8)) as usize;
        let pair_swap = 0x88 * perspective as u8;
        let lut1_flat = THREAT_LUTS.lut1.as_flattened();

        for dirty in diff.list.iter() {
            let raw = dirty.raw();
            let from = raw as u8 as usize ^ orient;
            let to = (raw >> 8) as u8 as usize ^ orient;
            let pair = (raw >> 16) as u8;
            // SAFETY: a dirty threat packs two squares (< 64) and two pieces (< 16), so the
            // oriented squares stay below 64 and `(pair >> 4) & 7` is a piece type.
            let index = unsafe {
                let plut2 = THREAT_LUTS.lut2.get_unchecked(((pair >> 4) & 7) as usize);
                lut1_flat[(pair ^ pair_swap) as usize][(from < to) as usize]
                    + *plut2.get_unchecked(from).get_unchecked(to) as TableIdx
            };
            let insert = if raw >> 31 != 0 {
                &mut *added
            } else {
                &mut *removed
            };
            // SAFETY: the callers' contract (checked by the debug_assert above in debug builds)
            // is that the lists have `len + diff.list.len <= MAX_FEATURE_INDEXES`, and at most
            // one slot per dirty entry is consumed, so `len < MAX_FEATURE_INDEXES` holds on every
            // push (a rejected index is stored in the spare slot but not counted).
            unsafe {
                insert.push_if_unchecked(index as FeatureIndexType, index < Self::DIMENSIONS_T)
            };
        }
    }

    /// Append the changed threat feature indices for *both* perspectives in a
    /// single pass over the dirty list.
    ///
    /// Port of the C++ `FullThreats::append_changed_indices_both`. Walking the
    /// packed list once and deriving both king-relative indices per entry halves
    /// the traversal cost versus calling
    /// [`append_changed_indices`](Self::append_changed_indices) per colour; the
    /// resulting index lists are identical.
    #[inline(never)]
    pub(crate) fn append_changed_indices_both(
        white_ksq: Square,
        black_ksq: Square,
        diff: &DirtyThreats,
        white_removed: &mut FeatureIndexList,
        white_added: &mut FeatureIndexList,
        black_removed: &mut FeatureIndexList,
        black_added: &mut FeatureIndexList,
    ) {
        // Each dirty entry consumes at most one slot per list. The unchecked pushes below rely on
        // the in-crate callers passing freshly cleared lists (checked in debug builds).
        let n = diff.list.len;
        debug_assert!(
            white_removed.len() + n <= MAX_FEATURE_INDEXES
                && white_added.len() + n <= MAX_FEATURE_INDEXES
                && black_removed.len() + n <= MAX_FEATURE_INDEXES
                && black_added.len() + n <= MAX_FEATURE_INDEXES
        );
        let w_orient = ORIENT_TBL[white_ksq as usize] as usize;
        let b_orient = (ORIENT_TBL[black_ksq as usize] ^ 56) as usize;
        let lut1_flat = THREAT_LUTS.lut1.as_flattened();

        for dirty in diff.list.iter() {
            let raw = dirty.raw();
            let from = raw as u8 as usize;
            let to = (raw >> 8) as u8 as usize;
            let pair = (raw >> 16) as u8;

            let (w_from, w_to) = (from ^ w_orient, to ^ w_orient);
            let (b_from, b_to) = (from ^ b_orient, to ^ b_orient);
            // SAFETY: a dirty threat packs two squares (< 64) and two pieces (< 16), so the
            // oriented squares stay below 64 and `(pair >> 4) & 7` is a piece type.
            let (white_index, black_index) = unsafe {
                let plut2 = THREAT_LUTS.lut2.get_unchecked(((pair >> 4) & 7) as usize);
                (
                    lut1_flat[pair as usize][(w_from < w_to) as usize]
                        + *plut2.get_unchecked(w_from).get_unchecked(w_to) as TableIdx,
                    lut1_flat[(pair ^ 0x88) as usize][(b_from < b_to) as usize]
                        + *plut2.get_unchecked(b_from).get_unchecked(b_to) as TableIdx,
                )
            };

            let (white_insert, black_insert) = if raw >> 31 != 0 {
                (&mut *white_added, &mut *black_added)
            } else {
                (&mut *white_removed, &mut *black_removed)
            };

            // SAFETY: the callers' contract (checked by the debug_assert above in debug builds)
            // is that all four lists have `len + diff.list.len <= MAX_FEATURE_INDEXES`, and
            // every dirty entry appends at most one index per list, so
            // `len < MAX_FEATURE_INDEXES` holds on every push.
            unsafe {
                white_insert.push_if_unchecked(
                    white_index as FeatureIndexType,
                    white_index < Self::DIMENSIONS_T,
                );
                black_insert.push_if_unchecked(
                    black_index as FeatureIndexType,
                    black_index < Self::DIMENSIONS_T,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `lut1` equals the pre-a1e94f95 construction, whose per-piece offsets counted each
    /// piece's own attacks (a black pawn's for black pawns) instead of reusing `lut2`'s.
    #[test]
    fn lut1_matches_per_piece_offsets() {
        let mut cumulative_offset = 0u32;
        for &attacker in &ALL_PIECES {
            let pt = attacker & 7;
            let piece_offset: u32 = (0..64)
                .filter(|&from| pt != 1 || (8..=55).contains(&from))
                .map(|from| const_popcount(const_pseudo_attacks(attacker, from)))
                .sum();
            if attacker == Piece::W_Knight as u8 {
                // Sum of knight pseudo-attacks over all 64 squares (edge squares have fewer than 8)
                assert_eq!(piece_offset, 336);
            }
            let valid_targets = NUM_VALID_TARGETS[attacker as usize] as u32;
            for &attacked in &ALL_PIECES {
                let map = FullThreats::MAP[pt as usize - 1][(attacked & 7) as usize - 1];
                let semi_excluded = pt == attacked & 7 && ((attacker ^ attacked) == 8 || pt != 1);
                let feature = (map >= 0).then(|| {
                    cumulative_offset
                        + ((attacked >> 3) as u32 * (valid_targets / 2) + map as u32) * piece_offset
                });
                let expect = |excluded: bool| match feature {
                    Some(f) if !excluded => f,
                    _ => FullThreats::DIMENSIONS as u32,
                };
                let lut = THREAT_LUTS.lut1[attacker as usize][attacked as usize];
                assert_eq!(
                    lut,
                    [expect(false), expect(semi_excluded)],
                    "{attacker} -> {attacked}"
                );
            }
            cumulative_offset += valid_targets * piece_offset;
        }
    }

    /// `lut2[pt][from][to]` equals the pre-b877fbde per-piece `offsets[pc][from]` plus
    /// `index_lut2[pc][from][to]` for every (pseudo-)attack the threat features can see.
    #[test]
    fn folded_lut2_matches_per_piece_tables() {
        for &pc in &ALL_PIECES {
            let pt = (pc & 7) as usize;
            let mut offset = 0u32;
            for from in 0..64 {
                let attacks = const_pseudo_attacks(pc, from);
                let mut targets = attacks;
                while targets != 0 {
                    let to = targets.trailing_zeros() as usize;
                    targets &= targets - 1;
                    let rank = (attacks & ((1u64 << to) - 1)).count_ones();
                    assert_eq!(
                        THREAT_LUTS.lut2[pt][from][to] as u32,
                        offset + rank,
                        "piece {pc} from {from} to {to}"
                    );
                }
                if pt != 1 || (8..=55).contains(&from) {
                    offset += attacks.count_ones();
                }
            }
        }
    }

    /// The raw-word decoding in `append_changed_indices{,_both}` matches `make_index`.
    #[test]
    fn changed_indices_match_make_index() {
        use crate::types::{DirtyThreat, DirtyThreats};
        let mut diff = DirtyThreats::default();
        let threats = [
            (Piece::W_Knight, Piece::B_Rook, 18u8, 35u8, true),
            (Piece::B_Queen, Piece::W_Bishop, 59, 31, false),
            (Piece::W_Pawn, Piece::B_Knight, 12, 21, true),
            (Piece::B_Pawn, Piece::W_Rook, 52, 43, false),
            (Piece::W_Rook, Piece::W_Rook, 0, 7, true),
            (Piece::B_Bishop, Piece::B_Pawn, 61, 52, true),
        ];
        for &(pc, tpc, from, to, add) in &threats {
            diff.list.push(DirtyThreat::new(
                pc,
                tpc,
                Square::from(from),
                Square::from(to),
                add,
            ));
        }
        for (wk, bk) in [(4u8, 60u8), (1, 62), (30, 33)] {
            let (wk, bk) = (Square::from(wk), Square::from(bk));
            let mut lists: [FeatureIndexList; 4] = Default::default();
            let [wr, wa, br, ba] = &mut lists;
            FullThreats::append_changed_indices_both(wk, bk, &diff, wr, wa, br, ba);
            for (perspective, ksq, removed, added) in
                [(Color::White, wk, 0, 1), (Color::Black, bk, 2, 3)]
            {
                let mut single: [FeatureIndexList; 2] = Default::default();
                let [r, a] = &mut single;
                FullThreats::append_changed_indices(perspective, ksq, &diff, r, a);
                assert_eq!(&*single[0], &*lists[removed]);
                assert_eq!(&*single[1], &*lists[added]);

                let mut want: [Vec<FeatureIndexType>; 2] = Default::default();
                for &(pc, tpc, from, to, add) in &threats {
                    let index = FullThreats::make_index(
                        perspective,
                        pc,
                        Square::from(from),
                        Square::from(to),
                        tpc,
                        ksq,
                    );
                    if index < FullThreats::DIMENSIONS_T {
                        want[add as usize].push(index as FeatureIndexType);
                    }
                }
                assert_eq!(&*single[0], &want[0][..]);
                assert_eq!(&*single[1], &want[1][..]);
            }
        }
    }
}
