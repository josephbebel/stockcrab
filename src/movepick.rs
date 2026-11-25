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

//! Move picker module - Rust port of Stockfish's movepick.h/movepick.cpp
//!
//! The MovePicker class is used to pick one legal move at a time from the
//! current position. The most important method is next_move(), which emits one
//! new legal move on every call, until there are no moves left.
//! MovePicker attempts to return the moves which are most likely to get a cut-off first.

use core::ptr::NonNull;

use crate::bitboard_fast::square_bb_pub;
use crate::history::*;
use crate::movegen::{
    ExtMove, MAX_MOVES, MoveList, generate_captures, generate_evasions, generate_quiets,
};
use crate::position::Position;
use crate::types::*;

// This is the Rust equivalent of Stockfish's USE_AVX512 MoveSorter.  The
// compile-time feature check keeps the portable build free of AVX-512 code;
// native builds on supported x86-64 CPUs use the same 16-entry register
// sorter as the C++ implementation.  `_kadd_mask16` requires AVX512DQ, so the
// sorter is gated on both avx512f and avx512dq.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
// The helpers below are `unsafe fn`s made entirely of AVX-512 intrinsics.
#[allow(unsafe_op_in_unsafe_fn)]
mod avx512_move_sorter {
    use super::ExtMove;
    use core::arch::x86_64::*;

    // Lane packing requires an 8-byte ExtMove with `mv` in the low dword and `value` in the high
    // dword.
    const _: () = assert!(core::mem::size_of::<ExtMove>() == 8);
    const _: () = assert!(core::mem::offset_of!(ExtMove, mv) == 0);
    const _: () = assert!(core::mem::offset_of!(ExtMove, value) == 4);

    /// Load the Move, and the ExtMove value, into all lanes of 512-bit registers
    #[inline(always)]
    unsafe fn splat_extmove(m: ExtMove) -> (__m512i, __m512i) {
        (
            _mm512_set1_epi32(m.mv.raw() as i32),
            _mm512_set1_epi32(m.value),
        )
    }

    pub struct MoveSorter {
        sorted_values: __m512i,
        sorted_moves: __m512i,
    }

    impl MoveSorter {
        pub const MAX_ELEMENTS: usize = 16;

        #[inline(always)]
        pub unsafe fn new(first: ExtMove) -> Self {
            let (sorted_moves, mut sorted_values) = splat_extmove(first);
            // Set the uninitialized move values to INT_MIN, so that they sort less than any other move
            sorted_values = _mm512_mask_set1_epi32(sorted_values, !1u16, i32::MIN);
            Self {
                sorted_values,
                sorted_moves,
            }
        }

        #[inline(always)]
        pub unsafe fn insert(&mut self, m: ExtMove) {
            debug_assert_ne!(m.value, i32::MIN);
            let (move_value, value) = splat_extmove(m);
            let less_than = _mm512_cmplt_epi32_mask(self.sorted_values, value);
            // This is _kadd_mask16(less_than, -1) from the C++ source.
            let expand = _kadd_mask16(less_than, !0u16);
            self.sorted_values = _mm512_mask_expand_epi32(value, expand, self.sorted_values);
            self.sorted_moves = _mm512_mask_expand_epi32(move_value, expand, self.sorted_moves);
        }

        #[inline(always)]
        unsafe fn write_one(
            &self,
            moves: *mut ExtMove,
            offset: usize,
            count: usize,
            indices: __m512i,
        ) {
            let store_count = count.saturating_sub(offset).min(8);
            if store_count != 0 {
                let ext_moves =
                    _mm512_permutex2var_epi32(self.sorted_moves, indices, self.sorted_values);
                let mask = ((1u16 << store_count) - 1) as __mmask8;
                _mm512_mask_storeu_epi64(moves.add(offset) as *mut i64, mask, ext_moves);
            }
        }

        #[inline(always)]
        pub unsafe fn write_sorted(&self, moves: *mut ExtMove, count: usize) {
            debug_assert!(count <= Self::MAX_ELEMENTS);
            self.write_one(
                moves,
                0,
                count,
                _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23),
            );
            self.write_one(
                moves,
                8,
                count,
                _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Move generation stages
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(i8)]
// The variants mirror C++ `Stages`, whose INIT_FIRST_STAGE ends in "Stage".
#[allow(clippy::enum_variant_names)]
enum Stage {
    TT,
    InitFirstStage,

    GoodCapture,
    GoodQuiet,
    BadMoves,

    QSearch,

    Evasion,

    Probcut,
}

// ---------------------------------------------------------------------------
// MovePicker - Incrementally generates and orders moves
// ---------------------------------------------------------------------------

/// MovePicker incrementally generates and orders moves.
///
/// Uses NonNull pointers for niche optimization (`Option<NonNull<T>>` is 8 bytes vs 16 for Option<*const T>).
/// SAFETY: All pointers must remain valid for the lifetime of MovePicker.
/// This is guaranteed because the search functions own these resources.
pub struct MovePicker<'a> {
    pos: NonNull<Position>,
    main_history: Option<NonNull<ButterflyHistory>>,
    low_ply_history: Option<NonNull<LowPlyHistory>>,
    // capture_history is always set by both constructors (new_main and new_probcut)
    capture_history: NonNull<CapturePieceToHistory>,
    continuation_history: Option<&'a [&'a PieceToHistory]>,
    pawn_history: Option<NonNull<AtomicPawnHistoryEntry>>,
    tt_move: Move,
    threshold: i32,
    depth: Depth,
    ply: i32,
    stage: Stage,
    // The kind of picker: GoodCapture (main search), QSearch, Evasion or Probcut.
    first_stage: Stage,
    skip_quiets: bool,

    // Move buffers and pointers
    // PERF: Using MaybeUninit avoids 2KB zero-init per node (matches C++ behavior)
    moves: [core::mem::MaybeUninit<ExtMove>; MAX_MOVES],
    // Each generate_and_score_* call generates into a stack-local MoveList (as in C++).
    cur: usize,     // Current position in moves array
    end_cur: usize, // End of current stage moves
    end_bad_captures: usize,
    end_bad_moves: usize,
}

impl<'a> MovePicker<'a> {
    /// Constructor for main search and quiescence search.
    ///
    /// Not inlined, to match the C++ constructor being a separate function.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the position pointer and all history table pointers
    /// remain valid for the lifetime of the MovePicker instance. `pos`, `main_history`,
    /// `low_ply_history`, `capture_history` and `pawn_history` must be valid non-null pointers,
    /// `0 <= ply < MAX_PLY`, and `continuation_history` must hold at least 6 entries for the main
    /// search (indices 0..=3 and 5 are read unchecked) and at least 1 for quiescence and evasions.
    #[inline(never)]
    pub unsafe fn new_main(
        pos: *const Position,
        tt_move: Move,
        depth: Depth,
        main_history: *const ButterflyHistory,
        low_ply_history: *const LowPlyHistory,
        capture_history: *const CapturePieceToHistory,
        continuation_history: &'a [&'a PieceToHistory],
        pawn_history: *const AtomicPawnHistoryEntry,
        ply: i32,
    ) -> Self {
        // Debug validations for position state
        debug_assert!(!pos.is_null(), "Position pointer cannot be null");

        // SAFETY: pos is guaranteed to be valid by the caller
        let pos_ref = unsafe { &*pos };

        debug_assert!(
            (0..MAX_PLY).contains(&ply),
            "Ply must be reasonable (0-{}), got {}",
            MAX_PLY - 1,
            ply
        );
        debug_assert!(
            (-10..MAX_PLY).contains(&depth),
            "Depth must be below MAX_PLY, got {}",
            depth
        );

        let first_stage = if pos_ref.checkers() != 0 {
            Stage::Evasion
        } else if depth > 0 {
            Stage::GoodCapture
        } else {
            Stage::QSearch
        };

        let stage = if tt_move != Move::none() && pos_ref.legal(tt_move) {
            Stage::TT
        } else {
            Stage::InitFirstStage
        };

        // Convert raw pointers to NonNull (niche optimization: Option<NonNull<T>> is 8 bytes)
        // SAFETY: Caller guarantees pointers are non-null
        MovePicker {
            // SAFETY: `pos` came from the caller's `*const Position`, which the `new_main` contract requires to be non-null
            // (already dereferenced above) and to outlive the picker.
            pos: unsafe { NonNull::new_unchecked(pos as *mut _) },
            main_history: NonNull::new(main_history as *mut _),
            low_ply_history: NonNull::new(low_ply_history as *mut _),
            // SAFETY: capture_history is always provided and non-null
            capture_history: unsafe { NonNull::new_unchecked(capture_history as *mut _) },
            continuation_history: Some(continuation_history),
            pawn_history: NonNull::new(pawn_history as *mut _),
            tt_move,
            threshold: 0,
            depth,
            ply,
            stage,
            first_stage,
            skip_quiets: false,
            // PERF: MaybeUninit avoids 2KB zero-init (matches C++ behavior)
            // SAFETY: an array of `MaybeUninit<ExtMove>` needs no initialization, so `uninit().assume_init()` is valid;
            // slots are only read after `write` (see `get_move`/`get_value` callers).
            moves: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_bad_moves: 0,
        }
    }

    /// Constructor for ProbCut: generates captures with SEE >= threshold
    /// NOT inlined to match C++ (MovePicker constructor is a separate function)
    ///
    /// # Safety
    ///
    /// The caller must ensure that the position pointer and capture_history pointer
    /// remain valid for the lifetime of the MovePicker instance.
    ///
    /// The side to move must not be in check (`pos.checkers() == 0`); this is only
    /// checked by a debug assertion.
    #[inline(never)]
    pub unsafe fn new_probcut(
        pos: *const Position,
        tt_move: Move,
        threshold: i32,
        capture_history: *const CapturePieceToHistory,
    ) -> Self {
        debug_assert!(!pos.is_null(), "Position pointer cannot be null");

        // SAFETY: pos is guaranteed to be valid by the caller
        let pos_ref = unsafe { &*pos };

        debug_assert!(
            pos_ref.checkers() == 0,
            "ProbCut MovePicker cannot be created when king is in check (checkers = {:#018x})",
            pos_ref.checkers()
        );

        // Per af181d9f: removed SEE check from TT move condition
        let first_stage = Stage::Probcut;
        let stage = if tt_move != Move::none()
            && pos_ref.capture_stage(tt_move)
            && pos_ref.legal(tt_move)
        {
            Stage::TT
        } else {
            Stage::InitFirstStage
        };

        // Convert raw pointers to NonNull (niche optimization: Option<NonNull<T>> is 8 bytes)
        MovePicker {
            // SAFETY: `pos` came from the caller's `*const Position`, which the `new_probcut` contract requires to be non-null
            // (already dereferenced above) and to outlive the picker.
            pos: unsafe { NonNull::new_unchecked(pos as *mut _) },
            main_history: None,
            low_ply_history: None,
            // SAFETY: capture_history is always provided and non-null
            capture_history: unsafe { NonNull::new_unchecked(capture_history as *mut _) },
            continuation_history: None,
            pawn_history: None,
            tt_move,
            threshold,
            depth: 0,
            ply: 0,
            stage,
            first_stage,
            skip_quiets: false,
            // PERF: MaybeUninit avoids 2KB zero-init (matches C++ behavior)
            // SAFETY: an array of `MaybeUninit<ExtMove>` needs no initialization, so `uninit().assume_init()` is valid;
            // slots are only read after `write` (see `get_move`/`get_value` callers).
            moves: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_bad_moves: 0,
        }
    }

    /// Skip generating quiet moves (used in some search conditions)
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
        if self.stage == Stage::BadMoves || self.stage == Stage::GoodQuiet {
            // discards current-stage and bad quiet moves
            self.end_cur = self.end_bad_captures;
            self.end_bad_moves = self.end_bad_captures;
        }
    }

    // Helper methods for MaybeUninit array access

    /// Write an ExtMove to the moves array at the given index
    ///
    /// # Safety
    ///
    /// `idx < MAX_MOVES`.
    #[inline(always)]
    unsafe fn set_move(&mut self, idx: usize, em: ExtMove) {
        debug_assert!(idx < MAX_MOVES);
        // SAFETY: `idx < MAX_MOVES` by the caller's contract
        unsafe { self.moves.get_unchecked_mut(idx).write(em) };
    }

    /// Read an ExtMove from the moves array at the given index
    /// SAFETY: Caller must ensure idx has been previously written to
    #[inline(always)]
    unsafe fn get_move(&self, idx: usize) -> ExtMove {
        debug_assert!(idx < MAX_MOVES);
        // SAFETY: idx is bounded by MAX_MOVES in all callers
        unsafe { self.moves.get_unchecked(idx).assume_init() }
    }

    /// Get the value field of the ExtMove at the given index
    /// SAFETY: Caller must ensure idx has been previously written to
    #[inline(always)]
    unsafe fn get_value(&self, idx: usize) -> i32 {
        debug_assert!(idx < MAX_MOVES);
        // SAFETY: idx is bounded by MAX_MOVES in all callers
        unsafe { self.moves.get_unchecked(idx).assume_init_ref().value }
    }

    /// Generate the capture moves and score them for ordering (C++
    /// `generate_and_score<CAPTURES>`); writes from `cur` and returns the end.
    ///
    /// Assigns a numerical value to each move in a list, used for sorting.
    /// Captures are ordered by Most Valuable Victim (MVV), preferring captures
    /// with a good history. Quiet moves are ordered using the history tables.
    #[inline(always)]
    fn generate_and_score_captures(&mut self) -> usize {
        // SAFETY: pos is guaranteed to be valid by the caller
        let pos = unsafe { self.pos.as_ref() };

        // PERF: Create local MoveList (matches C++ pattern - goes out of scope after scoring)
        let mut ml = MoveList::new(); // move generation
        generate_captures(pos, &mut ml);

        // Debug validation: verify all generated moves are actually captures
        if cfg!(debug_assertions) {
            for &m in ml.as_slice() {
                let is_capture = pos.capture_stage(m);
                let is_en_passant = m.move_type() == MoveType::EnPassant;
                let has_piece_on_to = pos.piece_on(m.to_sq()) != Piece::NoPiece;

                if !is_capture && !is_en_passant {
                    panic!(
                        "Move generator produced non-capture move when GenType::Captures was requested"
                    );
                }

                if !is_en_passant && !has_piece_on_to && m.promotion_type() != PieceType::Queen {
                    panic!("Move generator produced capture move to empty square");
                }
            }
        }

        let mut count = self.cur;

        // SAFETY: capture_history is always set by both constructors (new_main and new_probcut)
        let capture_history = unsafe { self.capture_history.as_ref() };

        for &m in ml.as_slice() {
            let to = m.to_sq();
            let pc = pos.moved_piece(m);
            let captured = pos.piece_on(to);

            // For en passant moves, the captured piece is not on the destination square
            // For queen promotions, they're treated as "capture stage" even if non-capturing
            debug_assert!(
                captured != Piece::NoPiece
                    || m.move_type() == MoveType::EnPassant
                    || m.promotion_type() == PieceType::Queen,
                "Capture moves must capture something (or be en passant or queen promotion): move {} to {:?} has no piece",
                m,
                to
            );

            // Score based on MVV-LVA and history - exact match to C++ movepick.cpp
            // C++: m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
            //             + 7 * int(PieceValue[capturedPiece]);
            // Note: For en passant and non-capturing queen promos, captured = NoPiece,
            // so type_of(NoPiece) = NoPieceType and PIECE_VALUE[NoPiece] = 0
            // SAFETY: All indices are bounded: pc (0-15), to (0-63), captured type (0-7)
            let value = unsafe {
                (**capture_history
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get_unchecked(type_of(captured) as usize)) as i32
                    + 7 * *PIECE_VALUE.get_unchecked(captured as usize)
            };

            // Per 8551f86e: removed check bonus for captures

            // SAFETY: indices are bounded by the legal move count, which is <= MAX_MOVES
            unsafe {
                self.moves
                    .get_unchecked_mut(count)
                    .write(ExtMove::with_value(m, value))
            };
            count += 1;
        }

        debug_assert!(count <= MAX_MOVES, "Generated too many moves");
        count
    }

    /// Generate the quiet moves and score them for ordering (C++ `generate_and_score<QUIETS>`).
    #[inline(always)]
    fn generate_and_score_quiets(&mut self) -> usize {
        // SAFETY: pos is guaranteed to be valid by the caller
        let pos = unsafe { self.pos.as_ref() };

        // PERF: Create local MoveList (matches C++ pattern - goes out of scope after scoring)
        let mut ml = MoveList::new(); // move generation
        generate_quiets(pos, &mut ml);

        let mut count = self.cur;
        let us = pos.side_to_move();

        // SAFETY: All these histories are always set by new_main constructor.
        // Quiets are only generated in the GoodCapture stage, which only a new_main picker reaches.
        // Using unwrap_unchecked to match C++ behavior (no null check).
        debug_assert!(
            self.main_history.is_some(),
            "main_history must be set for scoring quiets"
        );
        debug_assert!(
            self.pawn_history.is_some(),
            "pawn_history must be set for scoring quiets"
        );
        debug_assert!(
            self.continuation_history.is_some(),
            "continuation_history must be set for scoring quiets"
        );
        // SAFETY: `new_main` sets main_history to a non-null pointer; Stage::GoodCapture only exists on the `new_main` path
        // (`new_probcut` sets None and never reaches it), and the table outlives the picker per the constructor contract.
        let main_history = unsafe { self.main_history.unwrap_unchecked().as_ref() };
        // SAFETY: `us` is 0 or 1.
        let main_hist = unsafe { main_history.get_unchecked(us as usize) };
        // SAFETY: same as above for pawn_history: always Some on the `new_main` path that can reach Stage::GoodCapture.
        let pawn_entry = unsafe { self.pawn_history.unwrap_unchecked().as_ref() };
        // SAFETY: same as above for continuation_history: always Some on the `new_main` path; the caller passes 6 entries
        // in the main search.
        let cont_history = unsafe { self.continuation_history.unwrap_unchecked() };

        // Compute threat bitboards for move ordering
        // Per 20bc1955: extended to include Pawn (=0) and King
        // Upstream (SF18): threatByLesser[KING] is 0 (kings have no lesser attacker relevant here)
        let threat_by_lesser_pawn: Bitboard = 0;
        let threat_by_lesser_knight_bishop = pos.threats_by(PieceType::Pawn);
        let threat_by_lesser_rook = threat_by_lesser_knight_bishop
            | pos.threats_by(PieceType::Knight)
            | pos.threats_by(PieceType::Bishop);
        let threat_by_lesser_queen = threat_by_lesser_rook | pos.threats_by(PieceType::Rook);
        let threat_by_lesser_king: Bitboard = 0;
        // Indexed by PieceType (NoPieceType, Pawn, Knight, Bishop, Rook, Queen, King, 7)
        let threats: [Bitboard; 8] = [
            0,
            threat_by_lesser_pawn,
            threat_by_lesser_knight_bishop,
            threat_by_lesser_knight_bishop,
            threat_by_lesser_rook,
            threat_by_lesser_queen,
            threat_by_lesser_king,
            0,
        ];

        // Hoist the continuation-history table pointers out of the loop: the stores to `self.moves` could alias the
        // slice contents as far as the compiler knows, which forces them to be reloaded for every move.
        debug_assert!(
            cont_history.len() >= 6,
            "generate_and_score_quiets requires 6 continuation history entries"
        );
        // SAFETY: main-search callers pass a 6-element array (debug_assert above), so indices 0..=3 and 5 are in bounds.
        let (ch0, ch1, ch2, ch3, ch5) = unsafe {
            (
                *cont_history.get_unchecked(0),
                *cont_history.get_unchecked(1),
                *cont_history.get_unchecked(2),
                *cont_history.get_unchecked(3),
                *cont_history.get_unchecked(5),
            )
        };
        // Low ply history row for this ply, if the table is in use (loop invariant).
        let lph_row = if self.ply < LOW_PLY_HISTORY_SIZE as i32 {
            self.low_ply_history.map(|lph| {
                // SAFETY: lph is a valid table pointer per the constructor contract and `self.ply` was just checked
                // to be < LOW_PLY_HISTORY_SIZE (and is non-negative: it is a search ply).
                unsafe { lph.as_ref().get_unchecked(self.ply as usize) }
            })
        } else {
            None
        };

        for &m in ml.as_slice() {
            let from = m.from_sq();
            let to = m.to_sq();
            let pc = pos.moved_piece(m);
            let pt = type_of(pc);

            // Note: Some special moves (en passant, castling) may have special handling
            // so we don't assert that the destination square is empty

            // History scores - exact match to C++ lines 160-166
            // SAFETY: m.raw() is a u16, within the UINT_16_HISTORY_SIZE row.
            // Per 7b7a9485: use m.raw() instead of from_to(m)
            let mut value = unsafe { 2 * (**main_hist.get_unchecked(m.raw() as usize)) as i32 };
            // pawn_entry is already the entry for this position
            // SAFETY: pc < 16 (Piece) and to < 64 (Square) are in bounds of the [PIECE_NB][SQUARE_NB] pawn entry.
            value += unsafe {
                2 * (pawn_entry
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get()) as i32
            };

            // Continuation history - C++ accesses fixed indices without bounds checking
            // We access the exact same indices: 0, 1, 2, 3, and 5 (skipping 4)
            // SAFETY: main-search callers pass a 6-element array (debug_assert before the loop), so indices 0..=3 and 5
            // are in bounds; pc < 16 and to < 64.
            unsafe {
                value += ch0
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get() as i32;
                value += ch1
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get() as i32;
                value += ch2
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get() as i32;
                value += ch3
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get() as i32;
                value += ch5
                    .get_unchecked(pc as usize)
                    .get_unchecked(to as usize)
                    .get() as i32;
            }

            // Penalty for moving to a square threatened by a lesser piece,
            // or bonus for escaping an attack by a lesser piece.
            // Upstream (SF18): v = 20 * (bool(threat & from) - bool(threat & to))
            // SAFETY: pt is masked to 0..=7 (pc & 7), and `threats` has 8 entries.
            let threat_bb = unsafe { *threats.get_unchecked(pt as usize) };
            let from_threatened = ((threat_bb & square_bb_pub(from)) != 0) as i32;
            let to_threatened = ((threat_bb & square_bb_pub(to)) != 0) as i32;
            let v = 20 * (from_threatened - to_threatened);
            // SAFETY: make_piece returns 0-15 (PIECE_NB = 16)
            value +=
                unsafe { *PIECE_VALUE.get_unchecked(make_piece(Color::White, pt) as usize) } * v;

            // Low ply history bonus - exact match to C++ lines 180-181
            // Per 7b7a9485: use m.raw() instead of from_to(m)
            if let Some(row) = lph_row {
                // SAFETY: m.raw() is a u16, within UINT_16_HISTORY_SIZE.
                value +=
                    unsafe { 8 * (**row.get_unchecked(m.raw() as usize)) as i32 / (1 + self.ply) };
            }

            // bonus for checks
            if (pos.check_squares(pt) & square_bb_pub(to)) != 0 && pos.see_ge(m, -75) {
                value += 16384;
            }

            // SAFETY: indices are bounded by the legal move count, which is <= MAX_MOVES
            unsafe {
                self.moves
                    .get_unchecked_mut(count)
                    .write(ExtMove::with_value(m, value))
            };
            count += 1;
        }

        debug_assert!(count <= MAX_MOVES, "Generated too many moves");
        count
    }

    /// Generate the evasions and score them for ordering (C++ `generate_and_score<EVASIONS>`).
    #[inline(always)]
    fn generate_and_score_evasions(&mut self) -> usize {
        // SAFETY: pos is guaranteed to be valid by the caller
        let pos = unsafe { self.pos.as_ref() };

        // PERF: Create local MoveList (matches C++ pattern - goes out of scope after scoring)
        let mut ml = MoveList::new(); // move generation
        generate_evasions(pos, &mut ml);

        let mut count = self.cur;
        let us = pos.side_to_move();

        // SAFETY: main_history and continuation_history are always set by new_main constructor.
        // Evasions are only generated for first_stage Evasion, which only new_main sets.
        // Using unwrap_unchecked to match C++ behavior (no null check).
        debug_assert!(
            self.main_history.is_some(),
            "main_history must be set for scoring evasions"
        );
        debug_assert!(
            self.continuation_history.is_some(),
            "continuation_history must be set for scoring evasions"
        );
        debug_assert!(
            pos.checkers() != 0,
            "Evasions should only be generated when in check"
        );
        // SAFETY: `new_main` sets main_history; first_stage Evasion only exists on that path (`new_probcut` sets None and never
        // reaches it), and the table outlives the picker per the constructor contract.
        let main_history = unsafe { self.main_history.unwrap_unchecked().as_ref() };
        // SAFETY: same as above for continuation_history: always Some on the `new_main` path.
        let cont_history = unsafe { self.continuation_history.unwrap_unchecked() };
        // Hoist the main-history row and the first continuation-history table out of the loop.
        // SAFETY: `us` is 0 or 1.
        let main_hist = unsafe { main_history.get_unchecked(us as usize) };
        debug_assert!(
            !cont_history.is_empty(),
            "Evasion scoring requires at least one continuation history entry"
        );
        // SAFETY: cont_history has at least 1 element (qsearch passes exactly 1, the main search 6).
        let ch0 = unsafe { *cont_history.get_unchecked(0) };

        for &m in ml.as_slice() {
            let to = m.to_sq();
            let pc = pos.moved_piece(m);

            // Exact match to C++ lines 186-193
            let value = if pos.capture_stage(m) {
                let captured = pos.piece_on(to);

                // For en passant moves, the captured piece is not on the destination square
                // For queen promotions, they're treated as "capture stage" even if non-capturing
                debug_assert!(
                    captured != Piece::NoPiece
                        || m.move_type() == MoveType::EnPassant
                        || m.promotion_type() == PieceType::Queen,
                    "Capture evasion must capture something (or be en passant or queen promotion): move {} to {:?} has no piece",
                    m,
                    to
                );

                // C++ movepick.cpp score<EVASIONS>:
                //     m.value = PieceValue[capturedPiece] + (1 << 28);
                // with `capturedPiece = pos.piece_on(to)`. There is deliberately NO
                // special casing: an en-passant capture's destination square is empty
                // and a non-capturing queen promotion's destination square is empty, so
                // both score PieceValue[NO_PIECE] == 0 + (1 << 28) and therefore sort
                // last among capture-stage evasions.
                // SAFETY: All indices are bounded by their type constraints
                unsafe { *PIECE_VALUE.get_unchecked(captured as usize) + (1 << 28) }
            } else {
                // Per fb41f295: removed low ply history for evasions
                // Per 7b7a9485: use m.raw() instead of from_to(m)
                // SAFETY: m.raw() is a u16, within the UINT_16_HISTORY_SIZE row; pc < 16, to < 64.
                unsafe {
                    (**main_hist.get_unchecked(m.raw() as usize)) as i32
                        + ch0
                            .get_unchecked(pc as usize)
                            .get_unchecked(to as usize)
                            .get() as i32
                }
            };

            // SAFETY: indices are bounded by the legal move count, which is <= MAX_MOVES
            unsafe {
                self.moves
                    .get_unchecked_mut(count)
                    .write(ExtMove::with_value(m, value))
            };
            count += 1;
        }

        debug_assert!(count <= MAX_MOVES, "Generated too many moves");
        count
    }

    /// Partial insertion sort - sorts moves above threshold to the front
    /// Exact match to C++ implementation at movepick.cpp
    ///
    /// Sort moves in descending order up to and including a given limit.
    /// The order of moves smaller than the limit is left unspecified.
    fn partial_insertion_sort(&mut self, begin: usize, end: usize, limit: i32) {
        debug_assert!(begin <= end, "Invalid range for partial_insertion_sort");
        debug_assert!(end <= MAX_MOVES, "End index out of bounds");

        if end - begin <= 1 {
            return;
        }

        let mut sorted_end = begin;
        let mut p = begin + 1;

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq"
        ))]
        {
            // SAFETY: avx512f and avx512dq are guaranteed by the cfg gate on this block (compile time).
            let mut sorter = unsafe { avx512_move_sorter::MoveSorter::new(self.get_move(begin)) };

            while p < end {
                // SAFETY: indices begin..end have been initialized by scoring functions and are below MAX_MOVES.
                unsafe {
                    if self.get_value(p) >= limit {
                        if sorted_end - begin + 1 >= avx512_move_sorter::MoveSorter::MAX_ELEMENTS {
                            break;
                        }
                        sorter.insert(self.get_move(p));
                        sorted_end += 1;
                        self.set_move(p, self.get_move(sorted_end));
                    }
                }
                p += 1;
            }

            // SAFETY: the first sorted_end - begin + 1 entries are initialized ExtMoves.
            unsafe {
                sorter.write_sorted(
                    self.moves.as_mut_ptr().add(begin).cast(),
                    sorted_end - begin + 1,
                );
            }
        }

        while p < end {
            // SAFETY: indices begin..end have been initialized by scoring functions and are below MAX_MOVES.
            unsafe {
                if self.get_value(p) >= limit {
                    let tmp = self.get_move(p);
                    sorted_end += 1;
                    self.set_move(p, self.get_move(sorted_end));

                    let mut q = sorted_end;
                    while q > begin && self.get_value(q - 1) < tmp.value {
                        self.set_move(q, self.get_move(q - 1));
                        q -= 1;
                    }
                    self.set_move(q, tmp);
                }
            }
            p += 1;
        }
    }

    /// Select next move matching a predicate.
    ///
    /// Pass the already-loaded current ExtMove to the predicate.  The C++
    /// pointer implementation naturally reuses `*cur`; doing the same here
    /// avoids reloading through `MaybeUninit` for the common SEE filters.
    /// This matches the C++ template implementation at movepick.cpp
    ///
    /// Returns the next move satisfying a predicate function.
    /// This never returns the TT move, as it was emitted before.
    #[inline]
    fn select<F>(&mut self, mut filter: F) -> Move
    where
        F: FnMut(&mut Self, ExtMove) -> bool,
    {
        while self.cur < self.end_cur {
            // SAFETY: indices 0..end_cur have been initialized by scoring functions
            let em = unsafe { self.get_move(self.cur) };
            if em.mv != self.tt_move {
                if filter(self, em) {
                    self.cur += 1;
                    return em.mv;
                }
            }
            self.cur += 1;
        }
        Move::none()
    }

    /// Get the next move from the move picker
    /// Exact match to C++ implementation at movepick.cpp
    /// NOT inlined to match C++ (next_move is a separate function in C++ binary)
    ///
    /// This is the most important method of the MovePicker class. We emit one
    /// new legal move on every call until there are no more moves left,
    /// picking the move with the highest score from a list of generated moves.
    #[inline(never)]
    pub fn next_move(&mut self) -> Move {
        const GOOD_QUIET_THRESHOLD: i32 = -14000; // C++ `goodQuietThreshold`

        loop {
            match self.stage {
                Stage::TT => {
                    self.stage = Stage::InitFirstStage;
                    return self.tt_move;
                }

                Stage::InitFirstStage => {
                    self.cur = 0;
                    self.end_bad_moves = 0;

                    self.end_cur = if self.first_stage == Stage::Evasion {
                        self.generate_and_score_evasions()
                    } else {
                        self.generate_and_score_captures()
                    };

                    self.partial_insertion_sort(self.cur, self.end_cur, i32::MIN);

                    self.stage = self.first_stage;
                }

                Stage::GoodCapture => {
                    let m = self.select(|mp, cur_move| {
                        // SAFETY: pos is guaranteed to be valid by the caller
                        let pos = unsafe { mp.pos.as_ref() };

                        if pos.see_ge(cur_move.mv, -cur_move.value / 18) {
                            true
                        } else {
                            // SAFETY: `end_bad_moves` trails the select cursor, so it is below the number of
                            // generated moves, which is <= MAX_MOVES.
                            unsafe { mp.set_move(mp.end_bad_moves, cur_move) };
                            mp.end_bad_moves += 1;
                            false
                        }
                    });

                    if m != Move::none() {
                        return m;
                    }

                    self.end_bad_captures = self.end_bad_moves;

                    if !self.skip_quiets {
                        // initialize quiet moves
                        self.cur = self.end_bad_moves;
                        self.end_cur = self.generate_and_score_quiets();

                        self.partial_insertion_sort(self.cur, self.end_cur, -3560 * self.depth);

                        self.stage = Stage::GoodQuiet;
                        continue;
                    }

                    // Prepare the pointers to loop over the bad captures
                    self.cur = 0;
                    self.end_cur = self.end_bad_moves;

                    self.stage = Stage::BadMoves;
                }

                Stage::GoodQuiet => {
                    let m = self.select(|mp, cur_move| {
                        if cur_move.value > GOOD_QUIET_THRESHOLD {
                            true
                        } else {
                            // SAFETY: `end_bad_moves` trails the select cursor, so it is below the number of
                            // generated moves, which is <= MAX_MOVES.
                            unsafe { mp.set_move(mp.end_bad_moves, cur_move) };
                            mp.end_bad_moves += 1;
                            false
                        }
                    });

                    if m != Move::none() {
                        return m;
                    }

                    // Prepare the pointers to loop over the bad captures
                    self.cur = 0;
                    self.end_cur = self.end_bad_moves;

                    self.stage = Stage::BadMoves;
                }

                Stage::BadMoves | Stage::QSearch | Stage::Evasion => {
                    return self.select(|_, _| true);
                }

                Stage::Probcut => {
                    return self.select(|mp, cur_move| {
                        // SAFETY: pos is guaranteed to be valid by the caller
                        let pos = unsafe { mp.pos.as_ref() };
                        pos.see_ge(cur_move.mv, mp.threshold)
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq"
    ))]
    fn test_avx512_move_sorter_ordering_and_packing() {
        let values = [7, -3, 42, 0, 19, -100, 8, 5, 31, -1, 12, 4, 27, -7, 16, 2];
        let mut expected: Vec<(i32, u16)> = values
            .iter()
            .enumerate()
            .map(|(i, &value)| (value, 100 + i as u16))
            .collect();
        expected.sort_by_key(|e| core::cmp::Reverse(e.0));

        // SAFETY: avx512f and avx512dq are guaranteed by the cfg gate on this test (compile time).
        let mut sorter = unsafe {
            avx512_move_sorter::MoveSorter::new(ExtMove::with_value(
                Move::new(expected[0].1),
                expected[0].0,
            ))
        };
        for &(value, raw) in expected.iter().skip(1).rev() {
            // Insert in a deliberately different order from the expected sort.
            // SAFETY: avx512f and avx512dq are guaranteed by the cfg gate on this test; `sorter` was created above.
            unsafe {
                sorter.insert(ExtMove::with_value(Move::new(raw), value));
            }
        }

        let mut output = [ExtMove::new(Move::none()); 16];
        // SAFETY: this test is cfg-gated on `target_feature = "avx512f"` and `"avx512dq"` (compile time), so the
        // AVX-512 intrinsics are available; `output` holds 16 ExtMoves and `write_sorted` writes at most `count` = 16.
        unsafe {
            sorter.write_sorted(output.as_mut_ptr(), output.len());
        }

        for (actual, &(value, raw)) in output.iter().zip(expected.iter()) {
            assert_eq!(actual.value, value);
            assert_eq!(actual.mv.raw(), raw);
        }
    }

    fn setup_position(fen: &str) -> Position {
        crate::bitboard::init();
        let mut pos = Position::default();
        pos.set_fen(fen, false).unwrap();
        pos
    }

    fn create_empty_histories() -> (
        Box<ButterflyHistory>,
        Box<LowPlyHistory>,
        Box<CapturePieceToHistory>,
        Box<AtomicPawnHistoryEntry>,
        Vec<Box<PieceToHistory>>,
    ) {
        let main_history = new_butterfly_history();
        let low_ply_history = new_low_ply_history();
        let capture_history = new_capture_piece_to_history();
        // Create atomic pawn entry for tests
        let pawn_entry = new_atomic_pawn_entry();
        let mut cont_history = Vec::new();
        for _ in 0..6 {
            cont_history.push(new_piece_to_history());
        }
        (
            main_history,
            low_ply_history,
            capture_history,
            pawn_entry,
            cont_history,
        )
    }

    #[test]
    fn test_movepicker_basic() {
        let pos = setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        // Should generate moves
        let mut count = 0;
        let mut moves_seen = HashSet::new();
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert!(pos.legal(m), "Move must be legal");
            assert!(
                !moves_seen.contains(&m),
                "No duplicate moves should be generated"
            );
            moves_seen.insert(m);
            count += 1;
        }
        assert_eq!(count, 20, "Starting position has 20 legal moves");
    }

    #[test]
    fn test_movepicker_captures_only() {
        // Position with captures available
        let pos =
            setup_position("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");

        let capture_history = new_capture_piece_to_history();

        // SAFETY: `pos` and `capture_history` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_probcut(
                &pos as *const _,
                Move::none(),
                0,
                capture_history.as_ref() as *const _,
            )
        };

        // Should only generate captures that pass SEE threshold
        let mut count = 0;
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert!(
                pos.capture_stage(m),
                "ProbCut should only generate captures"
            );
            assert!(
                pos.see_ge(m, 0),
                "ProbCut captures should pass SEE threshold"
            );
            count += 1;
        }
        assert!(count > 0, "Should generate at least one capture");
    }

    #[test]
    fn test_movepicker_evasions() {
        // Position with king in check
        let pos = setup_position("4k3/8/8/8/8/2q5/8/4K3 w - - 0 1");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        // Should only generate evasions
        let mut count = 0;
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            // All moves should be king moves (only way to evade in this position)
            assert_eq!(
                m.from_sq(),
                Square::SQ_E1,
                "All evasions should be king moves"
            );
            count += 1;
        }
        assert!(count > 0, "Should generate at least one evasion");
    }

    #[test]
    fn test_bad_capture_handling() {
        // Position where bad captures will be separated
        // Covers the bad-capture handling fix.
        // Using a position with clear capture opportunities
        let pos =
            setup_position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                10,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        let mut captures = Vec::new();
        let mut quiets = Vec::new();
        let mut bad_captures_started = false;
        let mut bad_quiets_started = false;

        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }

            if pos.capture_stage(m) {
                // Track when bad captures start appearing
                if !bad_captures_started && !quiets.is_empty() {
                    bad_captures_started = true;
                }
                captures.push(m);
            } else {
                // Track when bad quiets start appearing
                if !bad_quiets_started && bad_captures_started {
                    bad_quiets_started = true;
                }
                quiets.push(m);
            }
        }

        // Verify move ordering: good captures -> good quiets -> bad captures -> bad quiets
        assert!(!captures.is_empty(), "Should have some captures");
        assert!(!quiets.is_empty(), "Should have some quiet moves");

        // Bad captures should come after good quiets
        assert!(bad_captures_started, "Bad captures should be generated");
    }

    #[test]
    fn test_move_ordering_consistency() {
        // Test that move ordering is consistent across multiple calls
        let pos =
            setup_position("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // Generate moves twice with same parameters
        let mut moves1 = Vec::new();
        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp1 = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        loop {
            let m = mp1.next_move();
            if m == Move::none() {
                break;
            }
            moves1.push(m);
        }

        let mut moves2 = Vec::new();
        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp2 = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        loop {
            let m = mp2.next_move();
            if m == Move::none() {
                break;
            }
            moves2.push(m);
        }

        // Move ordering should be deterministic
        assert_eq!(
            moves1.len(),
            moves2.len(),
            "Should generate same number of moves"
        );
        for i in 0..moves1.len() {
            assert_eq!(
                moves1[i], moves2[i],
                "Move ordering should be consistent at index {}",
                i
            );
        }
    }

    #[test]
    fn test_skip_quiets() {
        let pos =
            setup_position("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        mp.skip_quiet_moves();

        // Should only generate captures when quiets are skipped
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert!(
                pos.capture_stage(m),
                "When skip_quiets is true, only captures should be generated"
            );
        }
    }

    #[test]
    fn test_tt_move_filtering() {
        let pos = setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // Use e2e4 as TT move
        let tt_move = Move::from_to(Square::SQ_E2, Square::SQ_E4);

        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                tt_move,
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                0,
            )
        };

        // First move should be TT move
        let first = mp.next_move();
        assert_eq!(first, tt_move, "TT move should be returned first");

        // TT move should not appear again
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert_ne!(m, tt_move, "TT move should not be returned twice");
        }
    }

    #[test]
    fn test_probcut_threshold() {
        // Position with various captures of different values
        let pos =
            setup_position("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1");

        let capture_history = new_capture_piece_to_history();

        // Test with high threshold - should only get very good captures
        // SAFETY: `pos` and `capture_history` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_probcut(
                &pos as *const _,
                Move::none(),
                300, // High threshold
                capture_history.as_ref() as *const _,
            )
        };

        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert!(
                pos.capture_stage(m),
                "ProbCut should only generate captures"
            );
            assert!(
                pos.see_ge(m, 300),
                "All moves should pass the high SEE threshold"
            );
        }
    }

    #[test]
    fn test_movepicker_with_fresh_continuation_history() {
        // Fresh (zeroed) continuation history
        let pos = setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let main_history = new_butterfly_history();
        let low_ply_history = new_low_ply_history();
        let capture_history = new_capture_piece_to_history();
        let pawn_entry = new_atomic_pawn_entry();

        // Six zeroed tables, as the main search requires
        let mut cont_history_storage = vec![];
        for _ in 0..6 {
            cont_history_storage.push(new_piece_to_history());
        }
        let cont_history: Vec<&PieceToHistory> =
            cont_history_storage.iter().map(|h| h.as_ref()).collect();

        // SAFETY: `pos`, the history tables and `cont_history` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_history,
                pawn_entry.as_ref() as *const _,
                0,
            )
        };

        // Should still work with empty continuation history
        let mut count = 0;
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert!(pos.legal(m));
            count += 1;
        }
        assert_eq!(
            count, 20,
            "Should generate all moves even with empty continuation history"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "ProbCut MovePicker cannot be created when king is in check")]
    fn test_probcut_with_check_should_panic() {
        // Position with king in check - ProbCut should panic
        let pos = setup_position("4k3/8/8/8/8/2q5/8/4K3 w - - 0 1");

        assert!(pos.checkers() != 0, "Position should have king in check");

        let capture_history = new_capture_piece_to_history();

        // Use a king move instead of Move::none() to avoid Move validation issues
        let king_move = Move::from_to(Square::SQ_E1, Square::SQ_D1);

        // This should panic because king is in check
        // SAFETY: `pos` and `capture_history` are locals that outlive the call (which panics before building a picker).
        let _mp = unsafe {
            MovePicker::new_probcut(
                &pos as *const _,
                king_move,
                0,
                capture_history.as_ref() as *const _,
            )
        };
    }

    #[test]
    fn test_probcut_valid_position() {
        // Position without check - should work fine
        let pos =
            setup_position("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1");

        assert_eq!(pos.checkers(), 0, "Position should not have king in check");

        let capture_history = new_capture_piece_to_history();

        // This should work fine
        // SAFETY: `pos` and `capture_history` are locals that outlive the picker.
        let mut mp = unsafe {
            MovePicker::new_probcut(
                &pos as *const _,
                Move::none(),
                0,
                capture_history.as_ref() as *const _,
            )
        };

        // Verify it generates captures
        let mut count = 0;
        loop {
            let m = mp.next_move();
            if m == Move::none() {
                break;
            }
            assert!(
                pos.capture_stage(m),
                "ProbCut should only generate captures"
            );
            count += 1;
        }
        assert!(count > 0, "Should generate at least one capture");
    }

    #[test]
    fn test_debug_assertions_validation() {
        // Test various debug assertion conditions
        let pos = setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let (main_history, low_ply_history, capture_history, pawn_history, cont_history) =
            create_empty_histories();
        let cont_refs: Vec<&PieceToHistory> = cont_history.iter().map(|h| h.as_ref()).collect();

        // Test with valid parameters
        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let _mp = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                Move::none(),
                5,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                10, // Valid ply
            )
        };

        // Test with valid TT move
        let tt_move = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        // SAFETY: `pos`, the history tables and `cont_refs` are locals that outlive the picker.
        let _mp2 = unsafe {
            MovePicker::new_main(
                &pos as *const _,
                tt_move,
                10,
                main_history.as_ref() as *const _,
                low_ply_history.as_ref() as *const _,
                capture_history.as_ref() as *const _,
                &cont_refs,
                pawn_history.as_ref() as *const _,
                5,
            )
        };
    }
}
