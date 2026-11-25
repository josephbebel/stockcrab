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

//! Move generation. Port of upstream `movegen.cpp`.

use crate::bitboard_fast::{
    RANK_2_BB, RANK_3_BB, RANK_6_BB, RANK_7_BB, antidiag_bb, attacks_bb, attacks_bb_bishop,
    attacks_bb_knight, attacks_bb_queen, attacks_bb_rook, between_bb_pub, diag_bb, file_bb,
    line_bb_pub, lsb, more_than_one, pop_lsb, shift, square_bb_pub,
};
use crate::position::Position;
use crate::types::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi2"
))]
use core::arch::x86_64::*;
use core::mem::MaybeUninit;

pub const MAX_MOVES: usize = 256;

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// Type of moves to generate
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GenType {
    /// Generate capture moves and queen promotions
    Captures = 0,
    /// Generate quiet moves and underpromotions
    Quiets = 1,
    /// Generate evasions when in check
    Evasions = 2,
    /// Generate all legal moves, captures before quiets when not in check
    All = 3,
}

/// Extended move with value for move ordering.
/// Uses composition (mv: Move, value: i32) - idiomatic Rust equivalent to C++ inheritance.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ExtMove {
    pub mv: Move,
    pub value: i32,
}

impl ExtMove {
    #[inline(always)]
    pub const fn new(mv: Move) -> Self {
        ExtMove { mv, value: 0 }
    }

    #[inline(always)]
    pub const fn with_value(mv: Move, value: i32) -> Self {
        ExtMove { mv, value }
    }
}

impl From<Move> for ExtMove {
    #[inline(always)]
    fn from(mv: Move) -> Self {
        ExtMove::new(mv)
    }
}

impl PartialEq for ExtMove {
    fn eq(&self, other: &Self) -> bool {
        self.mv == other.mv
    }
}

// `partial_cmp` compares by value ascending (what `<` means for callers), while `Ord`
// is the descending order C++ `std::sort` produces; they intentionally differ.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for ExtMove {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Ord for ExtMove {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Note: C++ uses < operator for sorting, so higher values come first
        other.value.cmp(&self.value)
    }
}

impl Eq for ExtMove {}

/// Container for generated moves
pub struct MoveList {
    // Matches C++ MoveList storage: a plain `Move` buffer (no `value` field).
    // Move ordering/scoring lives in MovePicker's separate `ExtMove` arrays.
    moves: [MaybeUninit<Move>; MAX_MOVES],
    count: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList {
            // SAFETY: MaybeUninit doesn't require initialization - this is a no-op
            moves: unsafe { MaybeUninit::<[MaybeUninit<Move>; MAX_MOVES]>::uninit().assume_init() },
            count: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline(always)]
    pub fn contains(&self, mv: Move) -> bool {
        self.as_slice().contains(&mv)
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: indices 0..count are initialized via move generation.
        unsafe { core::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.count) }
    }

    /// C++-compatible begin() returning Move pointer
    #[inline(always)]
    pub fn begin(&self) -> *const Move {
        self.moves.as_ptr() as *const Move
    }

    #[inline(always)]
    pub fn end(&self) -> *const Move {
        // SAFETY: count <= MAX_MOVES (push debug-asserts count < MAX_MOVES before incrementing;
        // in release this relies on MAX_MOVES exceeding the maximum legal move count of
        // any position), so `add(count)` stays within or one past the end of the `moves` array.
        unsafe { (self.moves.as_ptr() as *const Move).add(self.count) }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.count = 0;
    }

    #[inline(always)]
    fn push(&mut self, mv: Move) {
        debug_assert!(self.count < MAX_MOVES);
        // SAFETY: count is always < MAX_MOVES.
        unsafe {
            (self.moves.as_mut_ptr().add(self.count) as *mut Move).write(mv);
        }
        self.count += 1;
    }

    /// Get a raw pointer to the next slot for C++-style move generation.
    /// (Used by the AVX-512 splat path only.)
    /// (Avoids creating a `&mut Move` pointing at uninitialized memory.)
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    #[inline(always)]
    fn next_slot_ptr(&mut self) -> *mut Move {
        debug_assert!(self.count < MAX_MOVES);
        // SAFETY: count is always < MAX_MOVES and the slot is within the array.
        unsafe { self.moves.as_mut_ptr().add(self.count) as *mut Move }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    #[inline(always)]
    fn advance_by(&mut self, n: usize) {
        debug_assert!(self.count + n <= MAX_MOVES);
        self.count += n;
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for MoveList {
    type Item = Move;
    type IntoIter = MoveListIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        MoveListIntoIter { list: self, idx: 0 }
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = core::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Owning iterator over a [`MoveList`].
pub struct MoveListIntoIter {
    list: MoveList,
    idx: usize,
}

impl Iterator for MoveListIntoIter {
    type Item = Move;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.list.count {
            return None;
        }
        // SAFETY: indices 0..count are initialized via move generation.
        let item = unsafe { self.list.moves.get_unchecked(self.idx).assume_init_read() };
        self.idx += 1;
        Some(item)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.count - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MoveListIntoIter {}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

// AVX-512 SIMD move generation. Matches C++ USE_AVX512ICL optimizations.
// Table-free scheme: compress the target bitboard into square indices via
// _mm512_maskz_compress_epi8(to_bb, AllSquares), then build packed Move values arithmetically.
// `vcompressb` is an AVX-512 VBMI2 instruction, so AVX-512F/BW alone is not
// sufficient. Keep this on exactly the same ICL-class feature set as Stockfish.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi2"
))]
mod simd {
    use super::*;

    // Byte vector holding square indices 0..63, matching C++ bitboard.h `AllSquares`.
    const fn all_squares_bytes() -> [i8; 64] {
        let mut a = [0i8; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = i as i8;
            i += 1;
        }
        a
    }
    static ALL_SQUARES: [i8; 64] = all_squares_bytes();

    #[inline(always)]
    fn all_squares() -> __m512i {
        // SAFETY: ALL_SQUARES is a 64-byte array, exactly one __m512i lane.
        unsafe { _mm512_loadu_si512(ALL_SQUARES.as_ptr() as *const __m512i) }
    }

    // The SIMD path is table-free, so there is nothing to build; kept for parity with
    // `movegen::init()`.
    #[inline(always)]
    pub fn init_tables() {}

    #[inline(always)]
    pub fn splat_pawn_moves<const OFFSET: i8>(list: &mut MoveList, to_bb: Bitboard) {
        debug_assert!(to_bb.count_ones() <= 8); // <= 8 pawns per side
        let count = to_bb.count_ones() as usize;
        debug_assert!(list.count + 8 <= MAX_MOVES);
        // SAFETY: the full-width store writes 8 Move (u16) slots (16 bytes) starting at
        // `list.count`; this is in bounds because count + 8 <= MAX_MOVES holds for any
        // position (at most ~218 moves), as in upstream.
        unsafe {
            // Compress the low 8 bytes (up to 8 squares) into a __m128i, widen to i16.
            let to_squares = _mm_cvtepi8_epi16(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(
                to_bb,
                all_squares(),
            )));
            let from_squares = _mm_subs_epi16(to_squares, _mm_set1_epi16(OFFSET as i16));
            let moves = _mm_or_si128(
                _mm_slli_epi16::<{ Move::FROM_SQ_SHIFT as i32 }>(from_squares),
                _mm_slli_epi16::<{ Move::TO_SQ_SHIFT as i32 }>(to_squares),
            );
            _mm_storeu_si128(list.next_slot_ptr() as *mut __m128i, moves);
            list.advance_by(count);
        }
    }

    #[inline(always)]
    pub fn splat_moves(list: &mut MoveList, from: Square, to_bb: Bitboard) {
        debug_assert!(to_bb.count_ones() <= 32); // Q can attack up to 27 squares
        let count = to_bb.count_ones() as usize;
        debug_assert!(list.count + 32 <= MAX_MOVES);
        // SAFETY: the full-width store writes 32 Move (u16) slots (64 bytes) starting at
        // `list.count`; this is in bounds because count + 32 <= MAX_MOVES holds for any
        // position (at most ~218 moves), as in upstream.
        unsafe {
            let from_vec = _mm512_set1_epi16(Move::from_to(from, Square::SQ_A1).raw() as i16);
            // Compress the low 32 bytes (up to 32 squares) into a __m256i, widen to i16.
            let to_squares = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(
                _mm512_maskz_compress_epi8(to_bb, all_squares()),
            ));
            let moves = _mm512_or_si512(
                from_vec,
                _mm512_slli_epi16::<{ Move::TO_SQ_SHIFT as u32 }>(to_squares),
            );
            _mm512_storeu_si512(list.next_slot_ptr() as *mut __m512i, moves);
            list.advance_by(count);
        }
    }
}

/// Write moves from a square to all target squares in a bitboard
#[inline(always)]
fn splat_moves(list: &mut MoveList, from: Square, to_bb: Bitboard) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    simd::splat_moves(list, from, to_bb);

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    )))]
    {
        let mut to_bb = to_bb;
        while to_bb != 0 {
            let to = pop_lsb(&mut to_bb);
            list.push(Move::from_to(from, to));
        }
    }
}

/// Write pawn moves with the given direction offset
#[inline(always)]
fn splat_pawn_moves<const OFFSET: i8>(list: &mut MoveList, to_bb: Bitboard) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    simd::splat_pawn_moves::<OFFSET>(list, to_bb);

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    )))]
    {
        let mut to_bb = to_bb;
        while to_bb != 0 {
            let to = pop_lsb(&mut to_bb);
            list.push(Move::pawn_move(to, OFFSET));
        }
    }
}

/// Generate promotion moves
#[inline(always)]
fn make_promotions<const TYPE: u8, const DIR: i8, const ENEMY: bool>(
    list: &mut MoveList,
    to: Square,
) {
    const CAPTURES: u8 = GenType::Captures as u8;
    const QUIETS: u8 = GenType::Quiets as u8;
    const EVASIONS: u8 = GenType::Evasions as u8;
    const ALL: u8 = GenType::All as u8;

    let from = Square::from((to as i8 - DIR) as u8);
    let all = TYPE == EVASIONS || TYPE == ALL;

    // Queen promotions for captures or all moves
    if TYPE == CAPTURES || all {
        let promotion_move = Move::make_promotion(from, to, PieceType::Queen);

        // Debug validation: ensure promotion moves are valid
        debug_assert!(
            (from as u8) < 64 && (to as u8) < 64,
            "Invalid promotion move squares: from={:?} ({}), to={:?} ({})",
            from,
            from as u8,
            to,
            to as u8
        );
        debug_assert!(
            promotion_move.promotion_type() == PieceType::Queen,
            "Promotion move should promote to Queen: {:?}",
            promotion_move
        );

        list.push(promotion_move);
    }

    // Underpromotions
    if (TYPE == CAPTURES && ENEMY) || (TYPE == QUIETS && !ENEMY) || all {
        list.push(Move::make_promotion(from, to, PieceType::Rook));
        list.push(Move::make_promotion(from, to, PieceType::Bishop));
        list.push(Move::make_promotion(from, to, PieceType::Knight));
    }
}

// ---------------------------------------------------------------------------
// Pawn Move Generation
// ---------------------------------------------------------------------------

#[inline(always)]
fn generate_pawn_moves<const US: bool, const TYPE: u8>(
    pos: &Position,
    list: &mut MoveList,
    target: Bitboard,
) {
    const WHITE: bool = true;
    const CAPTURES: u8 = GenType::Captures as u8;
    const EVASIONS: u8 = GenType::Evasions as u8;
    const ALL: u8 = GenType::All as u8;

    let them = if US == WHITE {
        Color::Black
    } else {
        Color::White
    };
    let us = if US == WHITE {
        Color::White
    } else {
        Color::Black
    };

    const fn trank7bb<const C: bool>() -> Bitboard {
        if C == WHITE { RANK_7_BB } else { RANK_2_BB }
    }

    const fn trank3bb<const C: bool>() -> Bitboard {
        if C == WHITE { RANK_3_BB } else { RANK_6_BB }
    }

    let up = if US == WHITE { NORTH } else { SOUTH };
    let up_right = if US == WHITE { NORTH_EAST } else { SOUTH_WEST };
    let up_left = if US == WHITE { NORTH_WEST } else { SOUTH_EAST };

    let empty_squares = !pos.pieces();
    let enemies = if TYPE == EVASIONS {
        pos.checkers()
    } else {
        pos.pieces_of_color(them)
    };

    // Use const-generic lookup instead of slice-based for performance
    let pawns = pos.pieces_c_pt::<{ PieceType::Pawn as u8 }>(us);
    let pawns_on7 = pawns & trank7bb::<US>();
    let pawns_not_on7 = pawns & !trank7bb::<US>();
    debug_assert!(TYPE != ALL);
    let pinned = pos.blockers_for_king(us);
    let ksq = pos.square_of(us, PieceType::King);
    let pushable = !pinned | file_bb(file_of(ksq));
    let can_capture_left = !pinned | antidiag_bb(ksq);
    let can_capture_right = !pinned | diag_bb(ksq);

    // Single and double pawn pushes (no promotions)
    if TYPE != CAPTURES {
        let b1 = shift(pawns_not_on7 & pushable, up) & empty_squares;
        let b2 = shift(b1 & trank3bb::<US>(), up) & empty_squares;

        let b1_filtered = if TYPE == EVASIONS { b1 & target } else { b1 };
        let b2_filtered = if TYPE == EVASIONS { b2 & target } else { b2 };

        if US == WHITE {
            splat_pawn_moves::<{ NORTH }>(list, b1_filtered);
            splat_pawn_moves::<{ NORTH + NORTH }>(list, b2_filtered);
        } else {
            splat_pawn_moves::<{ SOUTH }>(list, b1_filtered);
            splat_pawn_moves::<{ SOUTH + SOUTH }>(list, b2_filtered);
        }
    }

    // Promotions
    if pawns_on7 != 0 {
        let b1 = shift(pawns_on7 & can_capture_right, up_right) & enemies;
        let b2 = shift(pawns_on7 & can_capture_left, up_left) & enemies;
        let mut b3 = shift(pawns_on7 & pushable, up) & empty_squares;

        if TYPE == EVASIONS {
            b3 &= target;
        }

        // Promotion captures to the right
        let mut bb = b1;
        while bb != 0 {
            let to = pop_lsb(&mut bb);
            if US == WHITE {
                make_promotions::<TYPE, { NORTH_EAST }, true>(list, to);
            } else {
                make_promotions::<TYPE, { SOUTH_WEST }, true>(list, to);
            }
        }

        // Promotion captures to the left
        let mut bb = b2;
        while bb != 0 {
            let to = pop_lsb(&mut bb);
            if US == WHITE {
                make_promotions::<TYPE, { NORTH_WEST }, true>(list, to);
            } else {
                make_promotions::<TYPE, { SOUTH_EAST }, true>(list, to);
            }
        }

        // Promotion non-captures
        while b3 != 0 {
            let to = pop_lsb(&mut b3);
            if US == WHITE {
                make_promotions::<TYPE, { NORTH }, false>(list, to);
            } else {
                make_promotions::<TYPE, { SOUTH }, false>(list, to);
            }
        }
    }

    // Standard and en passant captures
    if TYPE == CAPTURES || TYPE == EVASIONS {
        let b1 = shift(pawns_not_on7 & can_capture_right, up_right) & enemies;
        let b2 = shift(pawns_not_on7 & can_capture_left, up_left) & enemies;

        if US == WHITE {
            splat_pawn_moves::<{ NORTH_EAST }>(list, b1);
            splat_pawn_moves::<{ NORTH_WEST }>(list, b2);
        } else {
            splat_pawn_moves::<{ SOUTH_WEST }>(list, b1);
            splat_pawn_moves::<{ SOUTH_EAST }>(list, b2);
        }

        // En passant captures
        if let Some(ep_sq) = pos.ep_square() {
            debug_assert!(
                rank_of(ep_sq) == relative_rank(us, Rank::R6),
                "Invalid en passant square: {:?} should be on rank {:?} for {:?}, but is on rank {:?}",
                ep_sq,
                relative_rank(us, Rank::R6),
                us,
                rank_of(ep_sq)
            );

            // An en passant capture cannot resolve a discovered check: if the double-pushed
            // pawn (ep_sq + up) is in the evasion target, no en passant move helps, so return.
            if TYPE == EVASIONS
                && (target & square_bb_pub(Square::from((ep_sq as i8 + up) as u8))) != 0
            {
                return;
            }

            let attackers = pawns_not_on7
                & ((can_capture_right & shift(square_bb_pub(ep_sq), -up_right))
                    | (can_capture_left & shift(square_bb_pub(ep_sq), -up_left)));

            // C++ asserts that at least one pawn can capture en passant
            debug_assert!(
                attackers != 0,
                "En passant square set but no pawns can capture"
            );

            let mut bb = attackers;
            while bb != 0 {
                let from = pop_lsb(&mut bb);
                list.push(Move::make_en_passant(from, ep_sq));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Piece Move Generation (Knight, Bishop, Rook, Queen)
// ---------------------------------------------------------------------------

#[inline(always)]
fn generate_moves<const PT: u8>(pos: &Position, list: &mut MoveList, target: Bitboard, us: Color) {
    debug_assert!(PT != PieceType::King as u8 && PT != PieceType::Pawn as u8);
    let ksq = pos.square_of(us, PieceType::King);
    let pinned = pos.blockers_for_king(us);
    let mut bb = pos.pieces_c_pt::<PT>(us) & !pinned;
    let occ = pos.pieces();
    while bb != 0 {
        let from = pop_lsb(&mut bb);
        let attacks = piece_attacks::<PT>(from, occ) & target;
        splat_moves(list, from, attacks);
    }
    // Pinned knights cannot move. Other pinned pieces must stay on the king's line.
    if PT != PieceType::Knight as u8 {
        bb = pos.pieces_c_pt::<PT>(us) & pinned;
        while bb != 0 {
            let from = pop_lsb(&mut bb);
            let attacks = piece_attacks::<PT>(from, occ) & target & line_bb_pub(ksq, from);
            splat_moves(list, from, attacks);
        }
    }
}

#[inline(always)]
fn piece_attacks<const PT: u8>(from: Square, occupied: Bitboard) -> Bitboard {
    if PT == PieceType::Knight as u8 {
        attacks_bb_knight(from)
    } else if PT == PieceType::Bishop as u8 {
        attacks_bb_bishop(from, occupied)
    } else if PT == PieceType::Rook as u8 {
        attacks_bb_rook(from, occupied)
    } else {
        attacks_bb_queen(from, occupied)
    }
}

// ---------------------------------------------------------------------------
// Main Generation Dispatcher
// ---------------------------------------------------------------------------

#[inline(always)]
fn generate_typed<const TYPE: u8>(pos: &Position, list: &mut MoveList) {
    const EVASIONS: u8 = GenType::Evasions as u8;
    const CAPTURES: u8 = GenType::Captures as u8;
    const QUIETS: u8 = GenType::Quiets as u8;
    debug_assert!(TYPE != GenType::All as u8);
    debug_assert!((TYPE == EVASIONS) == (pos.checkers() != 0));

    let us = pos.side_to_move();
    let ksq = pos.square_of(us, PieceType::King);
    let mut target = if TYPE == EVASIONS {
        !pos.pieces_of_color(us)
    } else if TYPE == CAPTURES {
        pos.pieces_of_color(!us)
    } else {
        !pos.pieces()
    };

    // Skip generating non-king moves when in double check.
    if TYPE != EVASIONS || !more_than_one(pos.checkers()) {
        if TYPE == EVASIONS {
            target = between_bb_pub(ksq, lsb(pos.checkers()));
        }
        if us == Color::White {
            generate_pawn_moves::<true, TYPE>(pos, list, target);
        } else {
            generate_pawn_moves::<false, TYPE>(pos, list, target);
        }
        generate_moves::<{ PieceType::Knight as u8 }>(pos, list, target, us);
        generate_moves::<{ PieceType::Bishop as u8 }>(pos, list, target, us);
        generate_moves::<{ PieceType::Rook as u8 }>(pos, list, target, us);
        generate_moves::<{ PieceType::Queen as u8 }>(pos, list, target, us);
    }

    if TYPE == EVASIONS {
        target = !pos.pieces_of_color(us);
    }
    let b = attacks_bb(PieceType::King, ksq, 0) & target;
    if b != 0 {
        splat_moves(list, ksq, b & !pos.threats_by(ALL_PIECES));
    }

    if TYPE == QUIETS && pos.can_castle(us & CastlingRights::ANY_CASTLING) {
        for cr in [
            us & CastlingRights::KING_SIDE,
            us & CastlingRights::QUEEN_SIDE,
        ] {
            if !pos.can_castle(cr) || pos.castling_impeded(cr) {
                continue;
            }

            let rook_sq = pos.castling_rook_square(cr);
            let to = relative_square(
                us,
                if rook_sq as u8 > ksq as u8 {
                    Square::SQ_G1
                } else {
                    Square::SQ_C1
                },
            );
            if between_bb_pub(ksq, to) & pos.threats_by(ALL_PIECES) == 0
                && pos.blockers_for_king(us) & square_bb_pub(rook_sq) == 0
            {
                list.push(Move::make_castling(ksq, rook_sq));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Initialize movegen module. Currently a no-op (the SIMD path is table-free); kept for parity
/// with upstream and call-site stability.
pub fn init() {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    simd::init_tables();
}

// =============================================================================
// Public API - matches C++ template<GenType> generate() pattern
// =============================================================================

/// Generate legal captures and queen promotions.
#[inline(never)]
pub fn generate_captures(pos: &Position, list: &mut MoveList) {
    list.clear();
    generate_typed::<{ GenType::Captures as u8 }>(pos, list);
}

/// Generate legal quiet moves and underpromotions.
#[inline(never)]
pub fn generate_quiets(pos: &Position, list: &mut MoveList) {
    list.clear();
    generate_typed::<{ GenType::Quiets as u8 }>(pos, list);
}

/// Generate legal check evasions.
#[inline(never)]
pub fn generate_evasions(pos: &Position, list: &mut MoveList) {
    list.clear();
    generate_typed::<{ GenType::Evasions as u8 }>(pos, list);
}

/// Generate all legal moves. Outside check, captures precede quiet moves.
#[inline(never)]
pub fn generate_all(pos: &Position, list: &mut MoveList) {
    list.clear();
    if pos.checkers() != 0 {
        generate_typed::<{ GenType::Evasions as u8 }>(pos, list);
    } else {
        generate_typed::<{ GenType::Captures as u8 }>(pos, list);
        generate_typed::<{ GenType::Quiets as u8 }>(pos, list);
    }
}

/// Compatibility entry point for callers requesting the complete legal move list.
#[inline(always)]
pub fn generate_legal(pos: &Position, list: &mut MoveList) {
    generate_all(pos, list);
}

/// Generate legal moves of the requested type.
#[inline(always)]
pub fn generate_into(pos: &Position, gen_type: GenType, list: &mut MoveList) {
    match gen_type {
        GenType::All => generate_all(pos, list),
        GenType::Captures => generate_captures(pos, list),
        GenType::Quiets => generate_quiets(pos, list),
        GenType::Evasions => generate_evasions(pos, list),
    }
}

#[inline(always)]
pub fn generate(pos: &Position, gen_type: GenType) -> MoveList {
    let mut list = MoveList::new();
    generate_into(pos, gen_type, &mut list);
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_position(fen: &str) -> Position {
        crate::bitboard::init();
        let mut pos = Position::default();
        pos.set_fen(fen, false).unwrap();
        pos
    }

    #[test]
    fn test_move_list_basic() {
        let mut list = MoveList::new();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());

        let mv = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        list.push(mv);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert!(list.contains(mv));
    }

    #[test]
    fn test_generate_starting_position() {
        let pos = setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Should have 20 moves from starting position (16 pawn + 4 knight)
        let moves = generate(&pos, GenType::All);
        assert_eq!(moves.len(), 20);

        // All moves should be quiet (no captures available)
        let captures = generate(&pos, GenType::Captures);
        assert_eq!(captures.len(), 0);

        let quiets = generate(&pos, GenType::Quiets);
        assert_eq!(quiets.len(), 20);
    }

    #[test]
    fn test_generate_captures() {
        let pos = setup_position("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");

        // White can capture d5 with e4
        let captures = generate(&pos, GenType::Captures);
        assert_eq!(captures.len(), 1);
        assert!(captures.contains(Move::from_to(Square::SQ_E4, Square::SQ_D5)));
    }

    #[test]
    fn test_generate_promotions() {
        let pos = setup_position("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");

        // Pawn on a7 can promote
        let moves = generate(&pos, GenType::All);

        // Should have 4 promotions (Q, R, B, N) + king moves
        let pawn_moves: Vec<_> = moves
            .as_slice()
            .iter()
            .filter(|m| m.from_sq() == Square::SQ_A7)
            .collect();

        assert_eq!(pawn_moves.len(), 4);

        // Check all promotion types are generated
        let has_queen = pawn_moves
            .iter()
            .any(|m| m.promotion_type() == PieceType::Queen);
        let has_rook = pawn_moves
            .iter()
            .any(|m| m.promotion_type() == PieceType::Rook);
        let has_bishop = pawn_moves
            .iter()
            .any(|m| m.promotion_type() == PieceType::Bishop);
        let has_knight = pawn_moves
            .iter()
            .any(|m| m.promotion_type() == PieceType::Knight);

        assert!(has_queen && has_rook && has_bishop && has_knight);
    }

    #[test]
    fn test_generate_en_passant() {
        let pos = setup_position("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");

        // White can capture en passant on f6
        let moves = generate(&pos, GenType::Captures);
        let ep_move = Move::make_en_passant(Square::SQ_E5, Square::SQ_F6);
        assert!(moves.contains(ep_move));
    }

    #[test]
    fn test_generate_castling() {
        let pos = setup_position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");

        let moves = generate(&pos, GenType::All);

        // Should contain both castling moves
        let white_oo = Move::make_castling(Square::SQ_E1, Square::SQ_H1);
        let white_ooo = Move::make_castling(Square::SQ_E1, Square::SQ_A1);

        assert!(moves.contains(white_oo));
        assert!(moves.contains(white_ooo));
    }

    #[test]
    fn test_generate_evasions() {
        // Position with king in check from queen
        let pos = setup_position("4k3/8/8/8/8/2q5/8/4K3 w - - 0 1");

        let moves = generate(&pos, GenType::Evasions);

        // King must move out of check (cannot block or capture)
        for &m in moves.as_slice() {
            assert_eq!(m.from_sq(), Square::SQ_E1);
        }

        // Should have multiple king moves available
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_generate_legal_moves() {
        let pos = setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let legal_moves = generate(&pos, GenType::All);

        // All 20 moves from starting position are legal
        assert_eq!(legal_moves.len(), 20);
    }

    #[test]
    fn test_pinned_piece_moves() {
        // White bishop on e2 is pinned to the king by the black rook on e8
        let pos = setup_position("4r1k1/8/8/8/8/8/4B3/4K3 w - - 0 1");

        let legal_moves = generate(&pos, GenType::All);
        assert!(!legal_moves.is_empty());
        for &m in legal_moves.as_slice() {
            assert_ne!(m.from_sq(), Square::SQ_E2, "pinned bishop must not move");
        }

        // All moves should be legal (filtering works)
        for &m in legal_moves.as_slice() {
            assert!(pos.legal(m));
        }
    }

    #[test]
    fn test_en_passant_assertion() {
        // Test that en passant generation works correctly with assertion
        let pos = setup_position("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");

        let moves = generate(&pos, GenType::Captures);
        let ep_move = Move::make_en_passant(Square::SQ_E5, Square::SQ_F6);
        assert!(
            moves.contains(ep_move),
            "En passant move should be generated"
        );

        // Verify the move is legal
        assert!(pos.legal(ep_move), "En passant move should be legal");
    }

    #[test]
    fn test_move_generation_consistency() {
        // Test that Captures + Quiets = NonEvasions
        let positions = vec![
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        for fen in positions {
            let pos = setup_position(fen);

            if pos.checkers() == 0 {
                let captures = generate(&pos, GenType::Captures);
                let quiets = generate(&pos, GenType::Quiets);
                let non_evasions = generate(&pos, GenType::All);

                // Combine captures and quiets
                let mut combined_moves = std::collections::HashSet::new();
                for &m in captures.as_slice() {
                    combined_moves.insert(m.raw());
                }
                for &m in quiets.as_slice() {
                    combined_moves.insert(m.raw());
                }

                let mut non_evasions_moves = std::collections::HashSet::new();
                for &m in non_evasions.as_slice() {
                    non_evasions_moves.insert(m.raw());
                }

                assert_eq!(
                    combined_moves.len(),
                    non_evasions_moves.len(),
                    "Captures + Quiets should equal NonEvasions for position {}",
                    fen
                );

                // Check that all moves match
                for mv in &combined_moves {
                    assert!(
                        non_evasions_moves.contains(mv),
                        "Move missing from NonEvasions"
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Validation tests guarding against the king-disappearance bugs
    // ------------------------------------------------------------------

    mod comprehensive_validation_tests {
        use super::*;

        /// Captures and quiets require a side not in check, evasions require check,
        /// and All is always valid. Every generator returns only legal moves.
        fn valid_gen_types(pos: &Position) -> Vec<GenType> {
            if pos.checkers() != 0 {
                vec![GenType::Evasions, GenType::All]
            } else {
                vec![GenType::Captures, GenType::Quiets, GenType::All]
            }
        }

        #[test]
        fn test_move_application_validation() {
            // Test that applying every generated move results in valid positions
            let test_positions = vec![
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            ];

            for fen in test_positions {
                let mut pos = setup_position(fen);
                let moves = generate(&pos, GenType::All);

                for &mv in moves.as_slice() {
                    // Apply the move
                    // SAFETY: the test move is legal.
                    unsafe { pos.do_move(mv) };

                    // Critical validations after applying move:

                    // 1. Both kings must exist
                    assert!(
                        pos.pieces_color_types(Color::White, &[PieceType::King])
                            .count_ones()
                            == 1,
                        "White King missing after applying move {} to position {}. Resulting position may be invalid.",
                        mv,
                        fen
                    );
                    assert!(
                        pos.pieces_color_types(Color::Black, &[PieceType::King])
                            .count_ones()
                            == 1,
                        "Black King missing after applying move {} to position {}. Resulting position may be invalid.",
                        mv,
                        fen
                    );

                    // 2. Position should have reasonable piece counts (2-32 total pieces)
                    let total_pieces = pos.pieces().count_ones();
                    assert!(
                        (2..=32).contains(&total_pieces),
                        "Invalid piece count {} after applying move {} to position {}",
                        total_pieces,
                        mv,
                        fen
                    );

                    // 3. Each side should have exactly one king with valid square
                    for color in [Color::White, Color::Black] {
                        let king_count = pos
                            .pieces_color_types(color, &[PieceType::King])
                            .count_ones();
                        assert_eq!(
                            king_count, 1,
                            "{:?} has {} kings (should be 1) after applying move {} to position {}",
                            color, king_count, mv, fen
                        );
                    }

                    // Undo the move to restore original position
                    // SAFETY: the test move is legal.
                    unsafe { pos.undo_move(mv) };
                }
            }
        }

        #[test]
        fn test_no_king_captures() {
            // Ensure that no generated moves can capture the opponent's king
            let test_positions = vec![
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                // Position where king might seem capturable but shouldn't be in generated moves
                "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            ];

            for fen in test_positions {
                let pos = setup_position(fen);

                // Test all generation types
                for gen_type in valid_gen_types(&pos) {
                    let moves = generate(&pos, gen_type);
                    let opponent_color = !pos.side_to_move();
                    let enemy_king_square = pos.square_of(opponent_color, PieceType::King);

                    for &mv in moves.as_slice() {
                        // No move should capture the enemy king
                        assert_ne!(
                            mv.to_sq(),
                            enemy_king_square,
                            "Generated move {} captures the {:?} king on {} in position {} (GenType: {:?})",
                            mv,
                            opponent_color,
                            enemy_king_square,
                            fen,
                            gen_type
                        );

                        // No move should target the enemy king square
                        let target_piece = pos.piece_on(mv.to_sq());
                        if target_piece != crate::types::Piece::NoPiece {
                            assert_ne!(
                                type_of(target_piece),
                                PieceType::King,
                                "Generated move {} targets a king at {} in position {} (GenType: {:?})",
                                mv,
                                mv.to_sq(),
                                fen,
                                gen_type
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn test_bounds_checking_and_square_validation() {
            // Test that all square calculations stay within valid bounds
            let test_positions = vec![
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                // Edge positions that might trigger bounds issues
                "8/8/8/8/8/k7/8/K7 w - - 0 1",   // Kings on edge
                "r6k/8/8/8/8/8/8/R6K w - - 0 1", // Pieces on edges
                "7k/8/8/8/8/8/8/K7 w - - 0 1",   // Kings on opposite corners
            ];

            for fen in test_positions {
                let pos = setup_position(fen);

                // Test all generation types
                for gen_type in valid_gen_types(&pos) {
                    let moves = generate(&pos, gen_type);

                    for &mv in moves.as_slice() {
                        let from_sq = mv.from_sq();
                        let to_sq = mv.to_sq();

                        // Bounds checking: squares must be in 0-63 range
                        assert!(
                            (from_sq as usize) < 64,
                            "Move {} from_sq {} out of bounds (0-63) in position {} (GenType: {:?})",
                            mv,
                            from_sq as usize,
                            fen,
                            gen_type
                        );
                        assert!(
                            (to_sq as usize) < 64,
                            "Move {} to_sq {} out of bounds (0-63) in position {} (GenType: {:?})",
                            mv,
                            to_sq as usize,
                            fen,
                            gen_type
                        );

                        // For promotion moves, validate promotion piece type
                        if mv.move_type() == MoveType::Promotion {
                            let promo_type = mv.promotion_type();
                            assert!(
                                matches!(
                                    promo_type,
                                    PieceType::Queen
                                        | PieceType::Rook
                                        | PieceType::Bishop
                                        | PieceType::Knight
                                ),
                                "Move {} has invalid promotion type {:?} in position {} (GenType: {:?})",
                                mv,
                                promo_type,
                                fen,
                                gen_type
                            );
                        }

                        // For en passant moves, validate en passant square
                        if mv.move_type() == MoveType::EnPassant {
                            let ep_sq = pos.ep_square();
                            assert_eq!(
                                Some(to_sq),
                                ep_sq,
                                "En passant move {} to_sq {} doesn't match ep_square {:?} in position {} (GenType: {:?})",
                                mv,
                                to_sq,
                                ep_sq,
                                fen,
                                gen_type
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn test_edge_square_moves() {
            // Specifically test moves involving edge squares to catch boundary issues
            let edge_positions = vec![
                // Rook on corner squares
                "r5k1/8/8/8/8/8/8/4K2R w - - 0 1",
                "R3k3/8/8/8/8/5K2/8/7r b - - 0 1",
                // Knights on edges (notorious for boundary issues)
                "n3k3/8/8/8/8/8/8/4K2N w - - 0 1",
                // Pawns near promotion
                "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
                // Kings on edges
                "k7/8/8/8/8/8/8/7K w - - 0 1",
            ];

            for fen in edge_positions {
                let pos = setup_position(fen);
                let moves = generate(&pos, GenType::All);

                for &mv in moves.as_slice() {
                    let from_sq = mv.from_sq();
                    let to_sq = mv.to_sq();

                    // Specific validation for edge squares

                    // Test that edge squares are handled correctly
                    let from_is_edge = crate::types::file_of(from_sq) == crate::types::File::A
                        || crate::types::file_of(from_sq) == crate::types::File::H
                        || crate::types::rank_of(from_sq) == crate::types::Rank::R1
                        || crate::types::rank_of(from_sq) == crate::types::Rank::R8;

                    let to_is_edge = crate::types::file_of(to_sq) == crate::types::File::A
                        || crate::types::file_of(to_sq) == crate::types::File::H
                        || crate::types::rank_of(to_sq) == crate::types::Rank::R1
                        || crate::types::rank_of(to_sq) == crate::types::Rank::R8;

                    if from_is_edge || to_is_edge {
                        assert!(
                            pos.legal(mv),
                            "Edge move {} should be legal in position {}",
                            mv,
                            fen
                        );
                    }
                }
            }
        }

        #[test]
        fn test_position_integrity_after_moves() {
            // Test that position state remains consistent after move sequences
            let test_positions = vec![
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                // Position with castling rights
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                // Position with en passant
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            ];

            for fen in test_positions {
                let mut pos = setup_position(fen);
                let original_castling = pos.can_castle(crate::types::CastlingRights::ANY_CASTLING);
                let original_ep = pos.ep_square();
                let original_key = pos.key();

                let moves = generate(&pos, GenType::All);

                for &mv in moves.as_slice() {
                    // Apply move
                    // SAFETY: the test move is legal.
                    unsafe { pos.do_move(mv) };

                    // Both kings must still exist and be on valid squares.
                    for color in [Color::White, Color::Black] {
                        let king_sq = pos.square_of(color, PieceType::King);

                        // King should be on a square that actually contains a king
                        let piece_on_king_sq = pos.piece_on(king_sq);
                        assert_eq!(
                            type_of(piece_on_king_sq),
                            PieceType::King,
                            "{:?} King square {} doesn't contain king after move {} in position {}",
                            color,
                            king_sq,
                            mv,
                            fen
                        );
                        assert_eq!(
                            color_of(piece_on_king_sq),
                            color,
                            "{:?} King square {} contains wrong color king after move {} in position {}",
                            color,
                            king_sq,
                            mv,
                            fen
                        );
                    }

                    // Position integrity checks
                    let total_pieces = pos.pieces().count_ones();
                    assert!(
                        (2..=32).contains(&total_pieces),
                        "Invalid total piece count {} after move {} in position {}",
                        total_pieces,
                        mv,
                        fen
                    );

                    // Each color should have reasonable piece counts
                    for color in [Color::White, Color::Black] {
                        let color_pieces = pos.pieces_of_color(color).count_ones();
                        assert!(
                            (1..=16).contains(&color_pieces),
                            "{:?} has invalid piece count {} after move {} in position {}",
                            color,
                            color_pieces,
                            mv,
                            fen
                        );
                    }

                    // A move must never gain castling rights
                    let new_castling = pos.can_castle(crate::types::CastlingRights::ANY_CASTLING);
                    assert!(
                        !new_castling || original_castling,
                        "Castling rights appeared from nowhere after move {} in position {}",
                        mv,
                        fen
                    );

                    // En passant square should be valid if set
                    if let Some(new_ep) = pos.ep_square() {
                        // En passant square should be on correct rank
                        let ep_rank = crate::types::rank_of(new_ep);
                        assert!(
                            ep_rank == crate::types::Rank::R3 || ep_rank == crate::types::Rank::R6,
                            "En passant square {} on invalid rank {:?} after move {} in position {}",
                            new_ep,
                            ep_rank,
                            mv,
                            fen
                        );
                    }

                    // Undo move and verify position is restored
                    // SAFETY: the test move is legal.
                    unsafe { pos.undo_move(mv) };

                    // Verify original state restored
                    assert_eq!(
                        pos.can_castle(crate::types::CastlingRights::ANY_CASTLING),
                        original_castling,
                        "Castling rights not restored after undoing move {} in position {}",
                        mv,
                        fen
                    );
                    assert_eq!(
                        pos.ep_square(),
                        original_ep,
                        "En passant square not restored after undoing move {} in position {}",
                        mv,
                        fen
                    );
                    assert_eq!(
                        pos.key(),
                        original_key,
                        "Zobrist key not restored after undoing move {} in position {}",
                        mv,
                        fen
                    );
                }
            }
        }

        #[test]
        fn test_king_safety_comprehensive() {
            // Comprehensive king safety tests including edge cases
            let dangerous_positions = vec![
                // Kings facing each other (but not adjacent)
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                // Kings with pieces between them
                "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1",
                // Position where kings might seem to approach each other
                "8/8/3k4/8/3K4/8/8/8 w - - 0 1",
                // Kings near corners
                "k7/8/8/8/8/8/8/7K w - - 0 1",
            ];

            for fen in dangerous_positions {
                let pos = setup_position(fen);

                for gen_type in valid_gen_types(&pos) {
                    let moves = generate(&pos, gen_type);
                    let our_king = pos.square_of(pos.side_to_move(), PieceType::King);
                    let enemy_king = pos.square_of(!pos.side_to_move(), PieceType::King);

                    for &mv in moves.as_slice() {
                        // Every generator must keep the kings apart.
                        if mv.from_sq() == our_king {
                            let to_sq = mv.to_sq();
                            let distance = core::cmp::max(
                                ((crate::types::file_of(to_sq) as i8)
                                    - (crate::types::file_of(enemy_king) as i8))
                                    .abs(),
                                ((crate::types::rank_of(to_sq) as i8)
                                    - (crate::types::rank_of(enemy_king) as i8))
                                    .abs(),
                            );
                            assert!(
                                distance > 1,
                                "King move {} places kings adjacent in position {} (GenType: {:?})",
                                mv,
                                fen,
                                gen_type
                            );
                        }

                        // No move should capture the enemy king
                        assert_ne!(
                            mv.to_sq(),
                            enemy_king,
                            "Move {} captures enemy king in position {} (GenType: {:?})",
                            mv,
                            fen,
                            gen_type
                        );

                        // No move should target a square containing a king
                        let target_piece = pos.piece_on(mv.to_sq());
                        if target_piece != crate::types::Piece::NoPiece {
                            assert_ne!(
                                type_of(target_piece),
                                PieceType::King,
                                "Move {} targets king at {} in position {} (GenType: {:?})",
                                mv,
                                mv.to_sq(),
                                fen,
                                gen_type
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn test_minimal_positions_movegen_invariants() {
            // Illegal positions are rejected by set_fen (see test_critical_position_regression).
            // This test checks move-generation invariants on small legal positions.
            let potentially_problematic_positions = vec![
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",   // Just kings
                "4k3/4r3/8/8/8/8/8/4K3 w - - 0 1", // Kings + rooks
                "4k3/7R/8/8/8/8/8/4K3 w - - 0 1",  // White rook on h7
                // Positions with limited pieces that might cause boundary issues
                "8/8/8/8/8/4k3/8/4K3 w - - 0 1",
                "4k3/8/8/8/8/8/4K3/8 w - - 0 1",
            ];

            for fen in potentially_problematic_positions {
                let pos = setup_position(fen);

                // Critical validation: both kings must exist
                assert!(
                    pos.pieces_color_types(Color::White, &[PieceType::King])
                        .count_ones()
                        == 1,
                    "White King missing in position {}",
                    fen
                );
                assert!(
                    pos.pieces_color_types(Color::Black, &[PieceType::King])
                        .count_ones()
                        == 1,
                    "Black King missing in position {}",
                    fen
                );

                // Generate moves and ensure all are valid
                for gen_type in valid_gen_types(&pos) {
                    let moves = generate(&pos, gen_type);

                    for &mv in moves.as_slice() {
                        // All generated moves should have valid squares

                        // Moves should be on pieces that actually exist
                        let moving_piece = pos.piece_on(mv.from_sq());
                        assert_ne!(
                            moving_piece,
                            crate::types::Piece::NoPiece,
                            "Move {} from empty square in position {} (GenType: {:?})",
                            mv,
                            fen,
                            gen_type
                        );
                        assert_eq!(
                            color_of(moving_piece),
                            pos.side_to_move(),
                            "Move {} tries to move opponent piece in position {} (GenType: {:?})",
                            mv,
                            fen,
                            gen_type
                        );
                    }
                }
            }
        }

        #[test]
        fn test_extreme_edge_cases() {
            // Additional edge cases that might trigger unusual behavior
            let extreme_positions = vec![
                // Kings next to the corner with few king moves
                "8/8/8/8/8/1k6/8/K7 w - - 0 1",
                "8/8/8/8/8/6k1/8/7K w - - 0 1",
                // Positions with only pawns and kings
                "4k3/p7/8/8/8/8/7P/4K3 w - - 0 1",
                "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
                // Positions with queens near kings
                "4k3/4q3/8/8/8/8/4Q3/4K3 w - - 0 1",
                // Underpromotion scenarios
                "4k3/P7/8/8/8/8/p7/4K3 w - - 0 1",
                // Knights in corners (boundary edge cases)
                "n3k3/8/8/8/8/8/8/4K2N w - - 0 1",
                "4k2n/8/8/8/8/8/8/N3K3 w - - 0 1",
            ];

            for fen in extreme_positions {
                let pos = setup_position(fen);

                // Validate both kings exist before testing
                assert!(
                    pos.pieces_color_types(Color::White, &[PieceType::King])
                        .count_ones()
                        == 1,
                    "White King missing in extreme position {}",
                    fen
                );
                assert!(
                    pos.pieces_color_types(Color::Black, &[PieceType::King])
                        .count_ones()
                        == 1,
                    "Black King missing in extreme position {}",
                    fen
                );

                let moves = generate(&pos, GenType::All);

                // Even in extreme positions, all moves should be valid
                for &mv in moves.as_slice() {
                    assert!(
                        pos.legal(mv),
                        "Illegal move {} generated in extreme position {}",
                        mv,
                        fen
                    );

                    // Test that piece exists on from square
                    let piece = pos.piece_on(mv.from_sq());
                    assert_ne!(
                        piece,
                        crate::types::Piece::NoPiece,
                        "Move {} from empty square in extreme position {}",
                        mv,
                        fen
                    );
                }
            }
        }

        #[test]
        fn test_enhanced_move_legality_validation() {
            // Comprehensive legality validation for various scenarios
            let validation_positions = vec![
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                // Complex middle game position
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
                // Endgame positions
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            ];

            for fen in validation_positions {
                let pos = setup_position(fen);

                // Test each generation type thoroughly
                for gen_type in valid_gen_types(&pos) {
                    let moves = generate(&pos, gen_type);

                    for &mv in moves.as_slice() {
                        // Comprehensive validation of each move
                        assert!(
                            mv != Move::none(),
                            "None move generated in position {} (GenType: {:?})",
                            fen,
                            gen_type
                        );

                        // Validate from square
                        let from_sq = mv.from_sq();

                        // Validate to square
                        let to_sq = mv.to_sq();

                        // From and to should be different (except for null moves, which shouldn't be generated)
                        assert_ne!(
                            from_sq, to_sq,
                            "Move {} has same from/to squares in position {} (GenType: {:?})",
                            mv, fen, gen_type
                        );

                        // Piece on from square should exist and be the right color
                        let moving_piece = pos.piece_on(from_sq);
                        assert_ne!(
                            moving_piece,
                            crate::types::Piece::NoPiece,
                            "No piece on from_sq {} for move {} in position {} (GenType: {:?})",
                            from_sq,
                            mv,
                            fen,
                            gen_type
                        );
                        assert_eq!(
                            color_of(moving_piece),
                            pos.side_to_move(),
                            "Wrong color piece {} on from_sq {} for move {} in position {} (GenType: {:?})",
                            moving_piece,
                            from_sq,
                            mv,
                            fen,
                            gen_type
                        );

                        // Validate move type consistency
                        match mv.move_type() {
                            MoveType::Normal => {
                                // `promotion_type()` is `KNIGHT + ((data >> 12) & 3)`
                                // in both engines, so it never returns NoPieceType
                                // and is only *meaningful* for promotions. What can
                                // be asserted is that a normal move leaves the two
                                // promotion bits clear.
                                assert_eq!(
                                    (mv.raw() >> 12) & 3,
                                    0,
                                    "Normal move {} has promotion bits set in position {} (GenType: {:?})",
                                    mv,
                                    fen,
                                    gen_type
                                );
                            }
                            MoveType::Promotion => {
                                // Promotion moves should be by pawns to back rank
                                assert_eq!(
                                    type_of(moving_piece),
                                    PieceType::Pawn,
                                    "Non-pawn {:?} making promotion move {} in position {} (GenType: {:?})",
                                    type_of(moving_piece),
                                    mv,
                                    fen,
                                    gen_type
                                );

                                let to_rank = crate::types::rank_of(to_sq);
                                let expected_rank = if pos.side_to_move() == Color::White {
                                    crate::types::Rank::R8
                                } else {
                                    crate::types::Rank::R1
                                };
                                assert_eq!(
                                    to_rank, expected_rank,
                                    "Promotion move {} to wrong rank {:?} in position {} (GenType: {:?})",
                                    mv, to_rank, fen, gen_type
                                );

                                let promo_type = mv.promotion_type();
                                assert!(
                                    matches!(
                                        promo_type,
                                        PieceType::Queen
                                            | PieceType::Rook
                                            | PieceType::Bishop
                                            | PieceType::Knight
                                    ),
                                    "Invalid promotion type {:?} in move {} in position {} (GenType: {:?})",
                                    promo_type,
                                    mv,
                                    fen,
                                    gen_type
                                );
                            }
                            MoveType::EnPassant => {
                                // En passant should only be by pawns
                                assert_eq!(
                                    type_of(moving_piece),
                                    PieceType::Pawn,
                                    "Non-pawn {:?} making en passant move {} in position {} (GenType: {:?})",
                                    type_of(moving_piece),
                                    mv,
                                    fen,
                                    gen_type
                                );

                                // Should target the en passant square
                                assert_eq!(
                                    Some(to_sq),
                                    pos.ep_square(),
                                    "En passant move {} doesn't target ep square {:?} in position {} (GenType: {:?})",
                                    mv,
                                    pos.ep_square(),
                                    fen,
                                    gen_type
                                );
                            }
                            MoveType::Castling => {
                                // Castling should only be by kings
                                assert_eq!(
                                    type_of(moving_piece),
                                    PieceType::King,
                                    "Non-king {:?} making castling move {} in position {} (GenType: {:?})",
                                    type_of(moving_piece),
                                    mv,
                                    fen,
                                    gen_type
                                );
                            }
                        }

                        // Legal moves should actually be legal
                        {
                            assert!(
                                pos.legal(mv),
                                "Legal generation produced illegal move {} in position {}",
                                mv,
                                fen
                            );
                        }
                    }
                }
            }
        }
    } // End comprehensive_validation_tests module

    #[test]
    fn test_critical_position_regression() {
        // Regression test for an illegal position with no white king. `set_fen`
        // reports this as a recoverable error rather than panicking, matching
        // C++ `Position::set` returning std::optional<PositionSetError>.
        let fen = "5k2/7R/4Pr1p/8/p4P1p/8/8/8 w - - 0 4";
        crate::bitboard::init();
        let mut pos = Position::default();
        // `Position` has no Debug impl, so match on the Result rather than using
        // `expect_err`.
        match pos.set_fen(fen, false) {
            Ok(_) => panic!("a position with no white king must be rejected"),
            Err(err) => assert_eq!(
                err.message(),
                "Unsupported position. Incorrect number of kings."
            ),
        }
    }

    // =========================================================================
    // Perft Tests - Verify move generation matches C++ Stockfish exactly
    // =========================================================================
    //
    // These tests use standard perft positions to verify that Rust move generation
    // is behaviorally equivalent to C++ Stockfish. Any difference indicates a bug.

    use crate::perft::perft_from_fen;

    #[test]
    fn test_perft_startpos_depth4() {
        // Starting position perft - fundamental correctness test
        let nodes = perft_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            4,
            false,
        )
        .unwrap();
        assert_eq!(nodes, 197281, "Perft 4 from startpos");
    }

    #[test]
    fn test_perft_startpos_depth5() {
        // Deeper startpos test
        let nodes = perft_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            5,
            false,
        )
        .unwrap();
        assert_eq!(nodes, 4865609, "Perft 5 from startpos");
    }

    #[test]
    fn test_perft_kiwipete_depth4() {
        // Kiwipete position - tests many tactical elements
        // Castling, en passant, pins, discovered attacks
        let nodes = perft_from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4,
            false,
        )
        .unwrap();
        assert_eq!(nodes, 4085603, "Perft 4 from Kiwipete");
    }

    #[test]
    fn test_perft_position3_depth5() {
        // Position 3 - en passant edge cases, pins on the rook
        let nodes = perft_from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, false).unwrap();
        assert_eq!(nodes, 674624, "Perft 5 from position 3");
    }

    #[test]
    fn test_perft_position4_depth4() {
        // Position 4 - promotions galore
        let nodes = perft_from_fen(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            4,
            false,
        )
        .unwrap();
        assert_eq!(nodes, 422333, "Perft 4 from position 4");
    }

    #[test]
    fn test_perft_position5_depth4() {
        // Position 5 - promotion on first move
        let nodes = perft_from_fen(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            4,
            false,
        )
        .unwrap();
        assert_eq!(nodes, 2103487, "Perft 4 from position 5");
    }

    #[test]
    fn test_perft_position6_depth4() {
        // Position 6 - alternative starting position with different castling
        let nodes = perft_from_fen(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            4,
            false,
        )
        .unwrap();
        assert_eq!(nodes, 3894594, "Perft 4 from position 6");
    }

    #[test]
    fn test_perft_en_passant_evasion() {
        // Regression test: en passant that captures the checking pawn
        // Position after b4b1 f4f3 a5b4 c7c5 from position 3
        // White king on b4 is in check from pawn on c5
        // En passant b5xc6 removes the checker - must be legal!
        let pos = setup_position("8/8/3p4/1Pp4r/1K5k/5p2/4P1P1/1R6 w - c6 0 3");

        // Verify we have 7 legal moves (including b5c6 en passant)
        let moves = generate(&pos, GenType::All);
        assert_eq!(
            moves.len(),
            7,
            "Should have 7 legal moves including en passant"
        );

        // Verify en passant move is present
        let ep_move = Move::make_en_passant(Square::SQ_B5, Square::SQ_C6);
        let has_ep = moves.as_slice().contains(&ep_move);
        assert!(
            has_ep,
            "En passant b5c6 should be legal - it captures the checking pawn"
        );
    }
}
