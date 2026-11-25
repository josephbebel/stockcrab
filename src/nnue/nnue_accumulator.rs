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

//! NNUE accumulator implementation for incremental updates
//!
//! This module implements the accumulator structure and logic for efficiently
//! updating neural network hidden layer values incrementally as pieces move.
//!
//! A single network feeds one combined accumulation of width `L1` (HalfKA plus
//! threats and pawn pairs). `AccumulatorState` carries a whole `Dirties` (dirty
//! piece, dirty threats and dirty pawn pairs), so all three are updated
//! incrementally. King moves that stay on the same board half use a "hybrid"
//! update that reuses the previous accumulator for the threat/pawn-pair part.

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use core::ops::{Deref, DerefMut};

use crate::bitboard_fast as bitboard;
use crate::position::Position;
use crate::types::{
    Bitboard, COLOR_NB, Color, Dirties, Piece, PieceType, SQUARE_NB, Square, make_piece,
};

use super::features::IndexList;
use super::features::k32h3::BlockState;
use super::network::Network;
use super::nnue_architecture::{L1 as ARCH_L1, PSQFeatureSet, PairFeatureSet, ThreatFeatureSet};
use super::nnue_common::{BiasType, FeatureIndexList, FeatureIndexType, WeightType};
use super::nnue_feature_transformer::FeatureTransformer;

/// Maximum ply depth for accumulator stack
const MAX_PLY: usize = 246;

/// Combined transformed feature width (HalfKA + Threats share the same
/// accumulation now). Mirrors C++ `Stockfish::Eval::NNUE::L1`.
pub const L1: usize = ARCH_L1;

/// Accumulator that holds the result of affine transformation of input
/// features, combined HalfKA + Threats.
#[repr(C, align(8))] // The cache-line alignment lives on `AccumulatorState`
#[derive(Clone)]
pub struct Accumulator {
    /// Combined (HalfKA + Threats) feature accumulation per color.
    pub accumulation: [[i16; L1]; COLOR_NB],
    /// Whether the accumulation is computed for each color.
    pub computed: [bool; COLOR_NB],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            accumulation: [[0; L1]; COLOR_NB],
            computed: [false; COLOR_NB],
        }
    }
}

/// Cache entry for accumulator refresh optimization ("Finny Tables").
///
/// Holds only the PSQ part of the accumulation (threats are re-derived from
/// the active threat features during a refresh). Uses a `pieces` array +
/// `piece_bb` instead of separate color/type bitboards.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Entry {
    pub accumulation: [BiasType; L1],
    pub pieces: [Piece; SQUARE_NB],
    pub piece_bb: Bitboard,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            accumulation: [0; L1],
            pieces: [Piece::NoPiece; SQUARE_NB],
            piece_bb: 0,
        }
    }
}

impl Entry {
    /// To initialize a refresh entry, we set all its bitboards empty, so we put
    /// the biases in the accumulation, without any weights on top.
    pub fn clear(&mut self, biases: &[BiasType]) {
        self.accumulation.copy_from_slice(&biases[..L1]);
        self.pieces = [Piece::NoPiece; SQUARE_NB];
        self.piece_bb = 0;
    }
}

/// Per-thread accumulator caches: one entry for each (PSQ block, king square, perspective)
/// triple. When the accumulator needs to be refreshed, the cached entry is used
/// to more efficiently update the accumulator instead of rebuilding it from
/// scratch. First described by Luecx (author of Koivisto), commonly referred to
/// as "Finny Tables".
///
/// C++ holds the ~0.8 MiB of entries inline; they are boxed here so that building a cache (or a
/// `Worker` holding one) never needs a stack temporary of that size.
pub struct AccumulatorCaches {
    pub entries: Box<[[[Entry; COLOR_NB]; SQUARE_NB]; PSQFeatureSet::NUM_BLOCKS]>,
}

impl AccumulatorCaches {
    /// Create new (zeroed) accumulator caches.
    ///
    /// IMPORTANT: Caches MUST be initialized with biases via `clear` before
    /// use, otherwise the accumulation will be wrong (zeros instead of biases
    /// as the base).
    pub fn new() -> Self {
        const { assert!(Piece::NoPiece as u8 == 0) };
        Self {
            // SAFETY: an all-zero `Entry` is `Entry::default()`: zero accumulation and bitboard,
            // and `Piece::NoPiece` is discriminant 0 (asserted above).
            entries: unsafe { Box::new_zeroed().assume_init() },
        }
    }

    /// Clear all cache entries, initializing them with the feature transformer
    /// biases. Mirrors C++ `AccumulatorCaches::clear(const Network& network)`
    /// which reads `network.featureTransformer.biases`.
    pub fn clear(&mut self, network: &Network) {
        if let Some(feature_transformer) = network.feature_transformer() {
            self.clear_with_transformer(feature_transformer);
        }
    }

    /// Clear all cache entries directly from a feature transformer's biases.
    pub fn clear_with_transformer(&mut self, feature_transformer: &FeatureTransformer) {
        let biases = feature_transformer.biases.as_slice();
        // `iter_mut()`: Rust 1.98 (the declared minimum) cannot iterate `&mut Box<[T; N]>`.
        for entries_2d in self.entries.iter_mut() {
            for entries_1d in entries_2d {
                for entry in entries_1d {
                    entry.clear(biases);
                }
            }
        }
    }

    /// Get cache entries for a PSQ block (C++ `operator[](int psqBlock)`), indexed
    /// `[king square][perspective]`.
    #[inline]
    pub fn get(&mut self, psq_block: usize) -> &mut [[Entry; COLOR_NB]; SQUARE_NB] {
        &mut self.entries[psq_block]
    }

    /// Borrow the entries of two *distinct* king squares for one perspective at
    /// once, the first shared and the second mutable. C++ simply takes
    /// `const auto& oldEntry = cache[psqBlock][oldKsq][perspective]` and
    /// `auto& newEntry = cache[psqBlock][newKsq][perspective]`; Rust needs a disjoint split.
    #[inline]
    fn get_pair(
        &mut self,
        psq_block: usize,
        shared_sq: Square,
        mut_sq: Square,
        perspective: Color,
    ) -> (&Entry, &mut Entry) {
        let p = perspective as usize;
        let s = shared_sq as usize;
        let m = mut_sq as usize;
        debug_assert_ne!(s, m);
        let entries = self.get(psq_block);

        if s < m {
            let (lo, hi) = entries.split_at_mut(m);
            (&lo[s][p], &mut hi[0][p])
        } else {
            let (lo, hi) = entries.split_at_mut(s);
            (&hi[0][p], &mut lo[m][p])
        }
    }
}

impl Default for AccumulatorCaches {
    fn default() -> Self {
        Self::new()
    }
}

/// State of the accumulator at a given position.
///
/// Mirrors C++ `struct AccumulatorState: public Accumulator, Dirties { BlockState blockBefore; }`:
/// it carries the [`Dirties`] aggregate (dirty piece, dirty threats and dirty pawn
/// pairs) describing the change from the previous state, and the PSQ blocks of the position
/// before that change. The embedded
/// [`Accumulator`] is exposed transparently through `Deref` / `DerefMut`, so
/// `state.accumulation` and `state.computed` work as
/// if inherited; the `Dirties` members are reached through `state.dirties`.
///
/// `Accumulator` is 4104 bytes (align 8), so `dirties` starts at offset 4104. This reproduces C++
/// `sizeof(AccumulatorState) == 4544` (71 cache lines) instead of the 4608 (72 lines) a 64-byte
/// aligned `Accumulator` would give; `block_before` fits in the tail padding, as in C++.
#[repr(C, align(64))]
#[derive(Clone, Default)]
pub struct AccumulatorState {
    pub acc: Accumulator,
    pub dirties: Dirties,
    /// PSQ blocks of both perspectives before the move (C++ `blockBefore`)
    pub block_before: BlockState,
}

impl Deref for AccumulatorState {
    type Target = Accumulator;
    #[inline]
    fn deref(&self) -> &Accumulator {
        &self.acc
    }
}

impl DerefMut for AccumulatorState {
    #[inline]
    fn deref_mut(&mut self) -> &mut Accumulator {
        &mut self.acc
    }
}

/// Stack of accumulators for incremental updates.
pub struct AccumulatorStack {
    accumulators: Vec<AccumulatorState>,
    size: usize,
}

impl AccumulatorStack {
    /// Maximum number of entries the stack can hold.
    pub const MAX_SIZE: usize = MAX_PLY + 1;

    /// Create a new accumulator stack.
    pub fn new() -> Self {
        Self {
            accumulators: vec![AccumulatorState::default(); Self::MAX_SIZE],
            size: 1,
        }
    }

    /// Get the latest accumulator state.
    pub fn latest(&self) -> &AccumulatorState {
        &self.accumulators[self.size - 1]
    }

    /// Get mutable reference to latest accumulator state.
    fn mut_latest(&mut self) -> &mut AccumulatorState {
        &mut self.accumulators[self.size - 1]
    }

    /// Reset the accumulator stack to a single (uncomputed) root state.
    pub fn reset(&mut self) {
        self.accumulators[0].dirties = Dirties::default();
        self.accumulators[0].computed.fill(false);
        self.size = 1;
    }

    /// Push a new accumulator state, returning a mutable reference to its
    /// [`Dirties`] for the caller to fill in (mirrors C++ `Dirties& push()`).
    ///
    /// # Safety
    ///
    /// The new state starts with empty dirty lists, and the incremental update reads the first
    /// dirty PSQ feature of every state it applies without a runtime check (like C++, it only
    /// asserts). The returned `Dirties` must therefore be filled by `Position::do_move_internal`
    /// (which records at least one PSQ feature change) before the next [`evaluate`] on this
    /// stack, unless the state is popped first. Evaluating a pushed-but-unfilled state reads an
    /// uninitialised feature index.
    ///
    /// [`evaluate`]: AccumulatorStack::evaluate
    pub unsafe fn push(&mut self, pos: &Position) -> &mut Dirties {
        debug_assert!(self.size < self.accumulators.len());
        let idx = self.size;
        self.size += 1;

        let st = &mut self.accumulators[idx];
        st.computed.fill(false);
        st.block_before = PSQFeatureSet::block_state(pos);
        // C++ re-constructs the threat list and pawn-pair bitboards in place.
        st.dirties.reset();
        &mut st.dirties
    }

    /// Pop the latest accumulator state.
    pub fn pop(&mut self) {
        debug_assert!(self.size > 1);
        self.size -= 1;
    }

    /// Evaluate accumulators for both colors.
    pub fn evaluate(
        &mut self,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
    ) {
        let last_white = self.find_last_usable_accumulator(Color::WHITE, pos);
        let last_black = self.find_last_usable_accumulator(Color::BLACK, pos);

        // When both perspectives can be brought up to date incrementally, the
        // common suffix of the stack is walked once for both of them.
        if self.accumulators[last_white].computed[Color::WHITE as usize]
            && self.accumulators[last_black].computed[Color::BLACK as usize]
        {
            self.forward_update_incremental_both(pos, feature_transformer, last_white, last_black);
        } else {
            self.evaluate_side(Color::WHITE, pos, feature_transformer, cache, last_white);
            self.evaluate_side(Color::BLACK, pos, feature_transformer, cache, last_black);
        }
    }

    /// Evaluate the accumulator for one color.
    fn evaluate_side(
        &mut self,
        perspective: Color,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        cache: &mut AccumulatorCaches,
        last_usable: usize,
    ) {
        // Below this piece count a full refresh is cheap enough that the hybrid
        // update is not worth it. C++ `MIN_PC_COUNT_HYBRID`.
        const MIN_PC_COUNT_HYBRID: u32 = 15;

        if self.accumulators[last_usable].computed[perspective as usize] {
            self.forward_update_incremental(perspective, pos, feature_transformer, last_usable);
        } else {
            let dirty_piece = self.latest().dirties.dirty_piece;

            // For a king move that stays on the same side of the board we can
            // reuse the previous accumulator for the threat/pawn-pair part and
            // only fix up the PSQ part via the two Finny cache entries.
            if self.size >= 2
                && dirty_piece.pc == make_piece(perspective, PieceType::King)
                && self.latest().block_before[perspective as usize] as usize
                    == PSQFeatureSet::block(pos, perspective)
                && self.accumulators[self.size - 2].computed[perspective as usize]
                && bitboard::popcount(pos.pieces()) >= MIN_PC_COUNT_HYBRID
                && matches!(
                    (dirty_piece.from, dirty_piece.to),
                    (Some(from), Some(to)) if ((from as i32) & 0b100) == ((to as i32) & 0b100)
                )
                && dirty_piece.add_sq.is_none()
            // excludes castling
            {
                let idx = self.size - 1;
                let (before, after) = self.accumulators.split_at_mut(idx);
                update_accumulator_hybrid(
                    perspective,
                    pos,
                    feature_transformer,
                    &mut after[0],
                    &before[idx - 1],
                    cache,
                );
                return;
            }

            update_accumulator_refresh_cache(
                perspective,
                feature_transformer,
                pos,
                self.mut_latest(),
                cache,
            );
            self.backward_update_incremental(perspective, pos, feature_transformer, last_usable);
        }
    }

    /// Find the earliest usable accumulator, this can either be a computed accumulator or the
    /// accumulator state just after a change that requires full refresh.
    fn find_last_usable_accumulator(&self, perspective: Color, pos: &Position) -> usize {
        // All traversed states must use the current block.
        let psq_block = PSQFeatureSet::block(pos, perspective);

        for curr_idx in (1..self.size).rev() {
            let state = &self.accumulators[curr_idx];
            if state.computed[perspective as usize] {
                return curr_idx;
            }

            let before = state.block_before[perspective as usize] as usize;

            // Threat feature set refreshes require a king move across the
            // center, i.e., a subset of PSQ refreshes, so testing the PSQ
            // feature set is sufficient.
            if PSQFeatureSet::requires_refresh(
                &state.dirties.dirty_piece,
                perspective,
                before,
                psq_block,
            ) {
                return curr_idx;
            }
        }

        0
    }

    /// Forward incremental update.
    fn forward_update_incremental(
        &mut self,
        perspective: Color,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        begin: usize,
    ) {
        debug_assert!(begin < self.accumulators.len());
        debug_assert!(self.accumulators[begin].computed[perspective as usize]);

        let ksq = pos.king_square(perspective);
        let psq_block = PSQFeatureSet::block(pos, perspective);

        for next in (begin + 1)..self.size {
            let (before, after) = self.accumulators.split_at_mut(next);
            let computed = &before[next - 1];
            let target_state = &mut after[0];

            update_accumulator_incremental::<true>(
                perspective,
                feature_transformer,
                ksq,
                psq_block,
                target_state,
                computed,
            );
        }

        debug_assert!(self.latest().computed[perspective as usize]);
    }

    /// Backward incremental update.
    fn backward_update_incremental(
        &mut self,
        perspective: Color,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        end: usize,
    ) {
        debug_assert!(end < self.accumulators.len());
        debug_assert!(end < self.size);
        debug_assert!(self.latest().computed[perspective as usize]);

        let ksq = pos.king_square(perspective);
        let psq_block = PSQFeatureSet::block(pos, perspective);

        for next in (end..(self.size - 1)).rev() {
            let (before, after) = self.accumulators.split_at_mut(next + 1);
            let target_state = &mut before[next];
            let computed = &after[0];

            update_accumulator_incremental::<false>(
                perspective,
                feature_transformer,
                ksq,
                psq_block,
                target_state,
                computed,
            );
        }

        debug_assert!(self.accumulators[end].computed[perspective as usize]);
    }

    /// Forward incremental update of both perspectives at once.
    ///
    /// The two perspectives may have different starting points, so the lagging
    /// one is first caught up on its own; from `max(white_begin, black_begin)`
    /// onwards the two colours share the same suffix and are updated together,
    /// which walks each state's dirty lists once instead of twice.
    fn forward_update_incremental_both(
        &mut self,
        pos: &Position,
        feature_transformer: &FeatureTransformer,
        white_begin: usize,
        black_begin: usize,
    ) {
        debug_assert!(white_begin < self.size);
        debug_assert!(black_begin < self.size);
        debug_assert!(self.accumulators[white_begin].computed[Color::WHITE as usize]);
        debug_assert!(self.accumulators[black_begin].computed[Color::BLACK as usize]);

        let white_ksq = pos.king_square(Color::WHITE);
        let black_ksq = pos.king_square(Color::BLACK);
        let white_block = PSQFeatureSet::block(pos, Color::WHITE);
        let black_block = PSQFeatureSet::block(pos, Color::BLACK);
        let shared_begin = white_begin.max(black_begin);

        // Catch up the lagging perspective, then traverse the common suffix once.
        for next in (white_begin + 1)..=shared_begin {
            let (before, after) = self.accumulators.split_at_mut(next);
            update_accumulator_incremental::<true>(
                Color::WHITE,
                feature_transformer,
                white_ksq,
                white_block,
                &mut after[0],
                &before[next - 1],
            );
        }
        for next in (black_begin + 1)..=shared_begin {
            let (before, after) = self.accumulators.split_at_mut(next);
            update_accumulator_incremental::<true>(
                Color::BLACK,
                feature_transformer,
                black_ksq,
                black_block,
                &mut after[0],
                &before[next - 1],
            );
        }

        for next in (shared_begin + 1)..self.size {
            let (before, after) = self.accumulators.split_at_mut(next);
            update_accumulator_incremental_both(
                feature_transformer,
                white_ksq,
                black_ksq,
                white_block,
                black_block,
                &mut after[0],
                &before[next - 1],
            );
        }

        debug_assert!(self.latest().computed[Color::WHITE as usize]);
        debug_assert!(self.latest().computed[Color::BLACK as usize]);
    }
}

impl Default for AccumulatorStack {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free functions (mirror the anonymous namespace in nnue_accumulator.cpp)
// ---------------------------------------------------------------------------

/// ARM NEON port of the C++ `#ifdef VECTOR` accumulator update path.
///
/// The tile geometry mirrors `SIMDTiling<Dimensions, Dimensions>`
/// from `simd.h`. With `vec_t == int16x8_t` and `NumRegistersSIMD == 16` the
/// ideal register count `L1 * 2 / 16 == 128` exceeds 16, so `NumRegs` collapses
/// to its largest divisor `<= 16`, which is 16; each tile therefore covers
/// `TileHeight == 128` accumulator entries and `L1 / 128 == 8` tiles are walked.
///
/// The point of the tiling is that a tile of the accumulator stays in registers
/// for the whole added/removed index list, so the 2 KiB accumulation is loaded
/// and stored once per tile instead of once per feature index (which is what the
/// scalar `#else` path below does).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
// Every function here is `unsafe fn` whose whole body is intrinsics and pointer
// arithmetic, as in `layers/affine_transform_sparse_input.rs`; the edition-2024
// lint would fire on essentially every line for no added information.
mod neon {
    use core::arch::aarch64::*;

    use super::{
        AccumulatorState, BiasType, Color, Entry, FeatureIndexType, FeatureTransformer, L1,
    };

    /// C++ `Tiling::NumRegs`.
    const NUM_REGS: usize = 16;
    /// C++ `Tiling::TileHeight`: accumulator entries covered by one tile.
    const TILE_HEIGHT: usize = NUM_REGS * 8;
    const NUM_TILES: usize = L1 / TILE_HEIGHT;

    const _: () = assert!(L1.is_multiple_of(TILE_HEIGHT), "TileHeight must divide L1");

    /// One tile of the i16 accumulation, held in registers.
    type AccTile = [int16x8_t; NUM_REGS];

    /// SAFETY (shared by every function in this module): all pointer arithmetic
    /// stays inside a tile of a fixed-size accumulation array (`NUM_TILES *
    /// TILE_HEIGHT == L1`)
    /// or inside a weight column `index * L1 .. (index + 1) * L1`, whose index
    /// the feature sets bound by the corresponding input dimension.
    #[inline(always)]
    unsafe fn load_acc_tile(src: *const BiasType) -> AccTile {
        let mut acc: AccTile = [vdupq_n_s16(0); NUM_REGS];
        for (k, a) in acc.iter_mut().enumerate() {
            *a = vld1q_s16(src.add(k * 8));
        }
        acc
    }

    #[inline(always)]
    unsafe fn store_acc_tile(dst: *mut BiasType, acc: &AccTile) {
        for (k, a) in acc.iter().enumerate() {
            vst1q_s16(dst.add(k * 8), *a);
        }
    }

    /// C++ `apply_psq_features<sign>`. SFNNv18 PSQ weights are `i8`, so each column
    /// goes through [`apply_feature_single`] like the threat columns.
    #[inline(always)]
    unsafe fn apply_psq_features<const SIGN: i32, const INCREMENTAL: bool>(
        tile_off: usize,
        acc: &mut AccTile,
        list: &[u16],
        psq_weights: *const i8,
    ) {
        debug_assert!(SIGN == 1 || SIGN == -1);

        let base = psq_weights;
        if INCREMENTAL {
            debug_assert!(list.len() == 1 || list.len() == 2);
            apply_feature_single::<SIGN>(
                acc,
                base.add(*list.get_unchecked(0) as usize * L1 + tile_off),
            );
            if list.len() > 1 {
                apply_feature_single::<SIGN>(
                    acc,
                    base.add(*list.get_unchecked(1) as usize * L1 + tile_off),
                );
            }
            return;
        }

        for &index in list {
            apply_feature_single::<SIGN>(acc, base.add(index as usize * L1 + tile_off));
        }
    }

    /// C++ `apply_feature_single<sign>` (the `USE_NEON` branch). The weights are
    /// `i8`, so two accumulator registers are fed from one `int8x16_t` load via the
    /// widening add/subtract pair `vaddw_s8` / `vaddw_high_s8`, which sign-extend
    /// the bytes for free.
    #[inline(always)]
    unsafe fn apply_feature_single<const SIGN: i32>(acc: &mut AccTile, column: *const i8) {
        debug_assert!(SIGN == 1 || SIGN == -1);
        let mut k = 0;
        while k < NUM_REGS {
            // Registers k and k+1 cover accumulator entries
            // [k * 8, k * 8 + 16), i.e. 16 consecutive i8 weights.
            let w = vld1q_s8(column.add(k * 8));
            if SIGN == 1 {
                acc[k] = vaddw_s8(acc[k], vget_low_s8(w));
                acc[k + 1] = vaddw_high_s8(acc[k + 1], w);
            } else {
                acc[k] = vsubw_s8(acc[k], vget_low_s8(w));
                acc[k + 1] = vsubw_high_s8(acc[k + 1], w);
            }
            k += 2;
        }
    }

    /// C++ `apply_threat_features<sign>`.
    #[inline(always)]
    unsafe fn apply_threat_features<const SIGN: i32>(
        tile_off: usize,
        acc: &mut AccTile,
        list: &[FeatureIndexType],
        ft: &FeatureTransformer,
    ) {
        let base = ft.threat_and_pp_weights.as_ptr();
        for &index in list {
            apply_feature_single::<SIGN>(acc, base.add(index as usize * L1 + tile_off));
        }
    }

    /// C++ `apply_combined`, `#ifdef VECTOR` branch.
    pub unsafe fn apply_combined(
        perspective: Color,
        ft: &FeatureTransformer,
        psq_weights: *const i8,
        from: &AccumulatorState,
        to: &mut AccumulatorState,
        psq_added: &[u16],
        psq_removed: &[u16],
        thr_added: &[FeatureIndexType],
        thr_removed: &[FeatureIndexType],
    ) {
        let p = perspective as usize;

        let from_acc = from.accumulation[p].as_ptr();
        let to_acc = to.acc.accumulation[p].as_mut_ptr();

        for j in 0..NUM_TILES {
            let off = j * TILE_HEIGHT;
            let mut acc = load_acc_tile(from_acc.add(off));

            apply_psq_features::<-1, true>(off, &mut acc, psq_removed, psq_weights);
            apply_psq_features::<1, true>(off, &mut acc, psq_added, psq_weights);

            apply_threat_features::<-1>(off, &mut acc, thr_removed, ft);
            apply_threat_features::<1>(off, &mut acc, thr_added, ft);

            store_acc_tile(to_acc.add(off), &acc);
        }
    }

    /// C++ `update_accumulator_hybrid`, `#ifdef VECTOR` branch (the part that
    /// touches the accumulations).
    pub unsafe fn hybrid_combine(
        ft: &FeatureTransformer,
        psq_weights: *const i8,
        new_entry: &mut Entry,
        old_entry: &Entry,
        from_acc: &[BiasType; L1],
        to_acc: &mut [BiasType; L1],
        new_remove: &[u16],
        new_add: &[u16],
        old_remove: &[u16],
        old_add: &[u16],
        thr_removed: &[FeatureIndexType],
        thr_added: &[FeatureIndexType],
    ) {
        let from_acc = from_acc.as_ptr();
        let to_acc = to_acc.as_mut_ptr();
        let old_acc = old_entry.accumulation.as_ptr();
        let new_acc = new_entry.accumulation.as_mut_ptr();

        for j in 0..NUM_TILES {
            let off = j * TILE_HEIGHT;
            let mut acc = load_acc_tile(new_acc.add(off));

            apply_psq_features::<-1, false>(off, &mut acc, new_remove, psq_weights);
            apply_psq_features::<1, false>(off, &mut acc, new_add, psq_weights);

            store_acc_tile(new_acc.add(off), &acc);

            // Adding the old accumulator adds (most of) the threat and pawn-pair
            // weights we need, but also a whole bunch of PSQ weights for the
            // wrong king bucket, which the old cache entry removes again.
            for (k, a) in acc.iter_mut().enumerate() {
                *a = vaddq_s16(*a, vld1q_s16(from_acc.add(off + k * 8)));
                *a = vsubq_s16(*a, vld1q_s16(old_acc.add(off + k * 8)));
            }

            // ... then we adjust. The signs are inverted: the old cache entry's
            // contribution is being undone.
            apply_psq_features::<1, false>(off, &mut acc, old_remove, psq_weights);
            apply_psq_features::<-1, false>(off, &mut acc, old_add, psq_weights);

            apply_threat_features::<-1>(off, &mut acc, thr_removed, ft);
            apply_threat_features::<1>(off, &mut acc, thr_added, ft);

            store_acc_tile(to_acc.add(off), &acc);
        }
    }

    /// C++ `update_accumulator_refresh_cache`, `#ifdef VECTOR` branch (the part
    /// that touches the accumulations).
    pub unsafe fn refresh_combine(
        ft: &FeatureTransformer,
        psq_weights: *const i8,
        entry: &mut Entry,
        to_acc: &mut [BiasType; L1],
        removed: &[u16],
        added: &[u16],
        active: &[FeatureIndexType],
    ) {
        let to_acc = to_acc.as_mut_ptr();
        let entry_acc = entry.accumulation.as_mut_ptr();

        for j in 0..NUM_TILES {
            let off = j * TILE_HEIGHT;
            let mut acc = load_acc_tile(entry_acc.add(off));

            apply_psq_features::<-1, false>(off, &mut acc, removed, psq_weights);
            apply_psq_features::<1, false>(off, &mut acc, added, psq_weights);

            // The cache entry holds the PSQ part only.
            store_acc_tile(entry_acc.add(off), &acc);

            apply_threat_features::<1>(off, &mut acc, active, ft);

            store_acc_tile(to_acc.add(off), &acc);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{AccumulatorState, BiasType, Entry, FeatureIndexType, FeatureTransformer, L1};
    use core::arch::x86_64::*;

    // Tiled accumulator updates used by the incremental, hybrid, and Finny-cache
    // refresh paths. Each accumulator stripe stays in registers while every
    // changed weight row is applied (Stockfish's `#ifdef VECTOR` tiling).
    //
    // The weight rows are permuted at load time (`FeatureTransformer::
    // widening_pair_orders`, C++ `WideningPairOrder`) for the vector width of the
    // kernel selected here, so every row is widened with unpack{lo,hi}_epi8: two
    // rows at once by maddubs against a vector of ones (`apply_feature_pair`), a
    // leftover removed and added row together by one maddubs against (+1, -1) pairs
    // (`apply_feature_diff`), and a lone leftover row by unpacking against zero and
    // shifting (`apply_feature_single`).
    // The PSQ and the threat/pawn-pair rows share that order, so a PSQ row can be
    // paired with a threat row.
    // The tile loops below walk all changed weight rows together, one tile at a time,
    // and the hardware prefetcher doesn't keep up with that many rows. Prefetch the
    // next tile of a row while its current tile is being added. On a row's last tile
    // this hits the next row, which is harmless since prefetches never fault. Rows of
    // only two tiles (AVX-512) get no lead time from this and are faster without it,
    // and so are AMD Zen 1/2, which only get builds without pext.
    /// C++ `prefetch_next_tile`; `TILE` is the tile height, which is also its size in bytes in
    /// an `i8` weight row. Only builds with BMI2 (C++ `USE_PEXT`) prefetch.
    #[inline(always)]
    fn prefetch_next_tile<const TILE: usize>(row: *const i8) {
        use crate::nnue::nnue_common::CACHE_LINE_SIZE;
        if cfg!(target_feature = "bmi2") && L1 / TILE > 2 {
            for k in (0..TILE).step_by(CACHE_LINE_SIZE) {
                // `wrapping_add`: on a row's last tile the address is past the row (and, for
                // the last row, possibly past the table), which a prefetch may touch but `add`
                // may not compute.
                // SAFETY: SSE is part of the x86-64 baseline, and a prefetch never faults.
                unsafe { _mm_prefetch::<_MM_HINT_T0>(row.wrapping_add(TILE + k)) };
            }
        }
    }

    /// C++ `apply_features_but_odd` for one fused kernel (see `define_fused_x86_path!`): applies
    /// one sign's psq and threat/pp rows for the tile at `off` and evaluates to the held-back
    /// odd row (null if none), so it can be combined with the other sign's odd row.
    /// `$psq_count` PSQ rows are given by `$psq_row(i)`, a pointer to the tile's first weight
    /// of row `i` (C++ `psqRow`: from an index list, or from row pointers hoisted out of the
    /// tile loop).
    ///
    /// The helpers are macros rather than functions so that they are always expanded in place:
    /// a helper call that is not inlined spills the whole register tile to memory around it.
    macro_rules! fused_but_odd {
        (
            $apply_pair:ident,
            $apply_features:ident,
            $sign:expr,
            $off:expr,
            $regs:expr,
            $psq_count:expr,
            $psq_row:expr,
            $thr:expr,
            $ft:expr
        ) => {{
            let regs = $regs;
            let psq_count: usize = $psq_count;
            let psq_row = $psq_row;
            let mut thr: &[FeatureIndexType] = $thr;
            let thr_weights = $ft.threat_and_pp_weights.as_ptr().add($off);
            let mut odd: *const i8 = core::ptr::null();
            let mut i = 0;
            while i + 1 < psq_count {
                $apply_pair::<{ $sign }>(&mut *regs, psq_row(i), psq_row(i + 1));
                i += 2;
            }

            // Pair the odd psq row with the first threat/pp row of the same sign, if any.
            if i < psq_count {
                if let Some((&first, rest)) = thr.split_first() {
                    $apply_pair::<{ $sign }>(
                        &mut *regs,
                        psq_row(i),
                        thr_weights.add(first as usize * L1),
                    );
                    thr = rest;
                } else {
                    odd = psq_row(i);
                }
            }

            if thr.len() % 2 != 0 {
                let (&last, rest) = thr.split_last().unwrap_unchecked();
                odd = thr_weights.add(last as usize * L1);
                thr = rest;
            }
            $apply_features::<{ $sign }>(&mut *regs, thr, thr_weights);
            odd
        }};
    }

    /// `(count, row accessor)` of the PSQ rows of an index list, for `fused_but_odd!`.
    macro_rules! psq_index_rows {
        ($list:expr, $psq_base:expr, $off:expr) => {{
            let list: &[u16] = $list;
            let base = $psq_base.add($off);
            (list.len(), move |i: usize| {
                base.add(*list.get_unchecked(i) as usize * L1)
            })
        }};
    }

    /// C++ `apply_combined_features` (and, with no threat rows, `apply_psq_features`): removes
    /// the PSQ rows `$psq_removed` and the threat/pawn-pair rows of `thr_removed`, and adds
    /// `$psq_added` and those of `thr_added`, for the tile at `off`. The PSQ rows are
    /// `(count, row accessor)` pairs (see `fused_but_odd!`).
    macro_rules! fused_combined_features {
        (
            $apply_pair:ident,
            $apply_features:ident,
            $apply_diff:ident,
            $apply_single:ident,
            $off:expr,
            $regs:expr,
            $psq_removed:expr,
            $thr_removed:expr,
            $psq_added:expr,
            $thr_added:expr,
            $ft:expr $(,)?
        ) => {{
            let regs = $regs;
            let (removed_count, removed_row) = $psq_removed;
            let (added_count, added_row) = $psq_added;
            let odd_removed = fused_but_odd!(
                $apply_pair,
                $apply_features,
                -1,
                $off,
                &mut *regs,
                removed_count,
                removed_row,
                $thr_removed,
                $ft
            );
            let odd_added = fused_but_odd!(
                $apply_pair,
                $apply_features,
                1,
                $off,
                &mut *regs,
                added_count,
                added_row,
                $thr_added,
                $ft
            );
            // C++ `apply_odd_features`: the held-back rows of both signs, as one diff when
            // there is one of each.
            if !odd_removed.is_null() && !odd_added.is_null() {
                $apply_diff(&mut *regs, odd_added, odd_removed);
            } else {
                if !odd_removed.is_null() {
                    $apply_single::<-1>(&mut *regs, odd_removed);
                }
                if !odd_added.is_null() {
                    $apply_single::<1>(&mut *regs, odd_added);
                }
            }
        }};
    }

    macro_rules! define_fused_x86_path {
        (
            $vec:ty,
            $num_regs:expr,
            $lanes:expr,
            $tile_height:expr,
            $features:literal,
            $load:ident,
            $store:ident,
            $load_acc:ident,
            $store_acc:ident,
            $zero:ident,
            $set1_8:ident,
            $set1_16:ident,
            $xor:ident,
            $add16:ident,
            $sub16:ident,
            $srai16:ident,
            $unpacklo8:ident,
            $unpackhi8:ident,
            $maddubs:ident,
            $apply_single:ident,
            $apply_pair:ident,
            $apply_diff:ident,
            $apply_features:ident,
            $apply_features_to_acc:ident,
            $apply_combined:ident,
            $hybrid_combine:ident,
            $refresh_combine:ident
        ) => {
            const _: () = assert!(L1.is_multiple_of($tile_height), "TileHeight must divide L1");
            const _: () = assert!($num_regs % 2 == 0);

            /// C++ `apply_feature_single` (`USE_PAIRED_WIDENING_ADD` branch); `row` points at
            /// the tile's first weight.
            #[target_feature(enable = $features)]
            #[inline]
            unsafe fn $apply_single<const SIGN: i32>(regs: &mut [$vec; $num_regs], row: *const i8) {
                prefetch_next_tile::<$tile_height>(row);
                let zero = $zero();
                for k in (0..$num_regs).step_by(2) {
                    let weight = $load(row.add(k / 2 * $lanes * 2) as *const $vec);
                    let lo = $srai16::<8>($unpacklo8(zero, weight));
                    let hi = $srai16::<8>($unpackhi8(zero, weight));
                    if SIGN == 1 {
                        regs[k] = $add16(regs[k], lo);
                        regs[k + 1] = $add16(regs[k + 1], hi);
                    } else {
                        regs[k] = $sub16(regs[k], lo);
                        regs[k + 1] = $sub16(regs[k + 1], hi);
                    }
                }
            }

            /// C++ `apply_feature_pair`.
            #[target_feature(enable = $features)]
            #[inline]
            unsafe fn $apply_pair<const SIGN: i32>(
                regs: &mut [$vec; $num_regs],
                row0: *const i8,
                row1: *const i8,
            ) {
                prefetch_next_tile::<$tile_height>(row0);
                prefetch_next_tile::<$tile_height>(row1);
                let ones = $set1_8(1);
                for k in (0..$num_regs).step_by(2) {
                    let w0 = $load(row0.add(k / 2 * $lanes * 2) as *const $vec);
                    let w1 = $load(row1.add(k / 2 * $lanes * 2) as *const $vec);
                    let lo = $maddubs(ones, $unpacklo8(w0, w1));
                    let hi = $maddubs(ones, $unpackhi8(w0, w1));
                    if SIGN == 1 {
                        regs[k] = $add16(regs[k], lo);
                        regs[k + 1] = $add16(regs[k + 1], hi);
                    } else {
                        regs[k] = $sub16(regs[k], lo);
                        regs[k + 1] = $sub16(regs[k + 1], hi);
                    }
                }
            }

            /// C++ `apply_feature_diff`: acc += add - sub in one maddubs per half. Bias both
            /// rows to u8 so maddubs' unsigned operand is valid; the bias cancels and
            /// |add - sub| <= 255 can't saturate.
            #[target_feature(enable = $features)]
            #[inline]
            unsafe fn $apply_diff(regs: &mut [$vec; $num_regs], add: *const i8, sub: *const i8) {
                prefetch_next_tile::<$tile_height>(add);
                prefetch_next_tile::<$tile_height>(sub);
                let bias = $set1_8(-128);
                let signs = $set1_16(-255); // i8 pairs (+1, -1)
                for k in (0..$num_regs).step_by(2) {
                    let ua = $xor($load(add.add(k / 2 * $lanes * 2) as *const $vec), bias);
                    let us = $xor($load(sub.add(k / 2 * $lanes * 2) as *const $vec), bias);
                    regs[k] = $add16(regs[k], $maddubs($unpacklo8(ua, us), signs));
                    regs[k + 1] = $add16(regs[k + 1], $maddubs($unpackhi8(ua, us), signs));
                }
            }

            /// C++ `apply_features`: the rows of `indices`, in pairs. The weight-table base
            /// already includes the tile offset.
            #[target_feature(enable = $features)]
            #[inline]
            unsafe fn $apply_features<const SIGN: i32>(
                regs: &mut [$vec; $num_regs],
                indices: &[u16],
                weights: *const i8,
            ) {
                let mut i = 0;
                while i + 1 < indices.len() {
                    $apply_pair::<SIGN>(
                        regs,
                        weights.add(*indices.get_unchecked(i) as usize * L1),
                        weights.add(*indices.get_unchecked(i + 1) as usize * L1),
                    );
                    i += 2;
                }
                if i < indices.len() {
                    $apply_single::<SIGN>(
                        regs,
                        weights.add(*indices.get_unchecked(i) as usize * L1),
                    );
                }
            }

            /// Add (`SIGN == 1`) or subtract the row of every index in `indices` of the
            /// weight table `weights` to/from `acc`.
            #[target_feature(enable = $features)]
            pub unsafe fn $apply_features_to_acc<const SIGN: i32>(
                acc: &mut [BiasType; L1],
                indices: &[u16],
                weights: *const i8,
            ) {
                let acc_ptr = acc.as_mut_ptr();
                for off in (0..L1).step_by($tile_height) {
                    let mut regs = [$zero(); $num_regs];
                    for k in 0..$num_regs {
                        regs[k] = $load(acc_ptr.add(off + k * $lanes) as *const $vec);
                    }
                    $apply_features::<SIGN>(&mut regs, indices, weights.add(off));
                    for k in 0..$num_regs {
                        $store(acc_ptr.add(off + k * $lanes) as *mut $vec, regs[k]);
                    }
                }
            }

            /// C++ `apply_combined`: an incremental update, with one or two PSQ rows of each sign.
            /// Dispatches once on the PSQ row counts; the counts and row addresses then stay fixed
            /// across tiles.
            #[target_feature(enable = $features)]
            #[inline(never)]
            pub unsafe fn $apply_combined(
                perspective: usize,
                ft: &FeatureTransformer,
                psq_base: *const i8,
                from: &AccumulatorState,
                to: &mut AccumulatorState,
                psq_added: &[u16],
                psq_removed: &[u16],
                thr_added: &[FeatureIndexType],
                thr_removed: &[FeatureIndexType],
            ) {
                /// C++ `apply_combined<RemovedCount, AddedCount>`.
                #[target_feature(enable = $features)]
                #[inline]
                unsafe fn combined<const REMOVED: usize, const ADDED: usize>(
                    perspective: usize,
                    ft: &FeatureTransformer,
                    psq_base: *const i8,
                    from: &AccumulatorState,
                    to: &mut AccumulatorState,
                    psq_added: &[u16],
                    psq_removed: &[u16],
                    thr_added: &[FeatureIndexType],
                    thr_removed: &[FeatureIndexType],
                ) {
                    let from_acc = from.acc.accumulation[perspective].as_ptr();
                    let to_acc = to.acc.accumulation[perspective].as_mut_ptr();

                    let mut removed_weights = [core::ptr::null::<i8>(); REMOVED];
                    let mut added_weights = [core::ptr::null::<i8>(); ADDED];
                    for i in 0..REMOVED {
                        removed_weights[i] =
                            psq_base.add(*psq_removed.get_unchecked(i) as usize * L1);
                    }
                    for i in 0..ADDED {
                        added_weights[i] = psq_base.add(*psq_added.get_unchecked(i) as usize * L1);
                    }

                    for off in (0..L1).step_by($tile_height) {
                        let mut regs = [$zero(); $num_regs];
                        for k in 0..$num_regs {
                            regs[k] = $load_acc(from_acc.add(off + k * $lanes) as *const $vec);
                        }

                        fused_combined_features!(
                            $apply_pair,
                            $apply_features,
                            $apply_diff,
                            $apply_single,
                            off,
                            &mut regs,
                            (REMOVED, |i: usize| removed_weights
                                .get_unchecked(i)
                                .add(off)),
                            thr_removed,
                            (ADDED, |i: usize| added_weights.get_unchecked(i).add(off)),
                            thr_added,
                            ft,
                        );

                        for k in 0..$num_regs {
                            $store_acc(to_acc.add(off + k * $lanes) as *mut $vec, regs[k]);
                        }
                    }
                }

                debug_assert!(psq_removed.len() == 1 || psq_removed.len() == 2);
                debug_assert!(psq_added.len() == 1 || psq_added.len() == 2);
                let (p, base) = (perspective, psq_base);
                match (psq_removed.len() == 1, psq_added.len() == 1) {
                    (true, true) => combined::<1, 1>(
                        p,
                        ft,
                        base,
                        from,
                        to,
                        psq_added,
                        psq_removed,
                        thr_added,
                        thr_removed,
                    ),
                    (true, false) => combined::<1, 2>(
                        p,
                        ft,
                        base,
                        from,
                        to,
                        psq_added,
                        psq_removed,
                        thr_added,
                        thr_removed,
                    ),
                    (false, true) => combined::<2, 1>(
                        p,
                        ft,
                        base,
                        from,
                        to,
                        psq_added,
                        psq_removed,
                        thr_added,
                        thr_removed,
                    ),
                    (false, false) => combined::<2, 2>(
                        p,
                        ft,
                        base,
                        from,
                        to,
                        psq_added,
                        psq_removed,
                        thr_added,
                        thr_removed,
                    ),
                }
            }

            #[target_feature(enable = $features)]
            pub unsafe fn $hybrid_combine(
                ft: &FeatureTransformer,
                psq_base: *const i8,
                new_entry: &mut Entry,
                old_entry: &Entry,
                from_acc: &[BiasType; L1],
                to_acc: &mut [BiasType; L1],
                new_remove: &[u16],
                new_add: &[u16],
                old_remove: &[u16],
                old_add: &[u16],
                thr_removed: &[FeatureIndexType],
                thr_added: &[FeatureIndexType],
            ) {
                let from_acc = from_acc.as_ptr();
                let to_acc = to_acc.as_mut_ptr();
                let old_acc = old_entry.accumulation.as_ptr();
                let new_acc = new_entry.accumulation.as_mut_ptr();

                for off in (0..L1).step_by($tile_height) {
                    let mut regs = [$zero(); $num_regs];
                    for k in 0..$num_regs {
                        regs[k] = $load(new_acc.add(off + k * $lanes) as *const $vec);
                    }
                    fused_combined_features!(
                        $apply_pair,
                        $apply_features,
                        $apply_diff,
                        $apply_single,
                        off,
                        &mut regs,
                        psq_index_rows!(new_remove, psq_base, off),
                        &[],
                        psq_index_rows!(new_add, psq_base, off),
                        &[],
                        ft,
                    );
                    for k in 0..$num_regs {
                        $store(new_acc.add(off + k * $lanes) as *mut $vec, regs[k]);
                        // Adding the old accumulator adds (most of) the threat and pawn-pair
                        // weights we need, but also a whole bunch of PSQ weights for the wrong
                        // king bucket, which the old cache entry removes again.
                        regs[k] = $add16(
                            regs[k],
                            $load(from_acc.add(off + k * $lanes) as *const $vec),
                        );
                        regs[k] =
                            $sub16(regs[k], $load(old_acc.add(off + k * $lanes) as *const $vec));
                    }
                    // ... then we adjust
                    // The old entry's removed psq features have to be added back, its added ones removed
                    fused_combined_features!(
                        $apply_pair,
                        $apply_features,
                        $apply_diff,
                        $apply_single,
                        off,
                        &mut regs,
                        psq_index_rows!(old_add, psq_base, off),
                        thr_removed,
                        psq_index_rows!(old_remove, psq_base, off),
                        thr_added,
                        ft,
                    );
                    for k in 0..$num_regs {
                        $store(to_acc.add(off + k * $lanes) as *mut $vec, regs[k]);
                    }
                }
            }

            #[target_feature(enable = $features)]
            pub unsafe fn $refresh_combine(
                ft: &FeatureTransformer,
                psq_base: *const i8,
                entry: &mut Entry,
                to_acc: &mut [BiasType; L1],
                removed: &[u16],
                added: &[u16],
                active: &[FeatureIndexType],
            ) {
                let to_acc = to_acc.as_mut_ptr();
                let entry_acc = entry.accumulation.as_mut_ptr();

                for off in (0..L1).step_by($tile_height) {
                    let mut regs = [$zero(); $num_regs];
                    for k in 0..$num_regs {
                        regs[k] = $load(entry_acc.add(off + k * $lanes) as *const $vec);
                    }
                    fused_combined_features!(
                        $apply_pair,
                        $apply_features,
                        $apply_diff,
                        $apply_single,
                        off,
                        &mut regs,
                        psq_index_rows!(removed, psq_base, off),
                        &[],
                        psq_index_rows!(added, psq_base, off),
                        &[],
                        ft,
                    );
                    for k in 0..$num_regs {
                        $store(entry_acc.add(off + k * $lanes) as *mut $vec, regs[k]);
                    }
                    $apply_features::<1>(
                        &mut regs,
                        active,
                        ft.threat_and_pp_weights.as_ptr().add(off),
                    );
                    for k in 0..$num_regs {
                        $store(to_acc.add(off + k * $lanes) as *mut $vec, regs[k]);
                    }
                }
            }
        };
    }

    define_fused_x86_path!(
        __m256i,
        8,
        16,
        128,
        "avx2",
        _mm256_loadu_si256,
        _mm256_storeu_si256,
        _mm256_loadu_si256,
        _mm256_storeu_si256,
        _mm256_setzero_si256,
        _mm256_set1_epi8,
        _mm256_set1_epi16,
        _mm256_xor_si256,
        _mm256_add_epi16,
        _mm256_sub_epi16,
        _mm256_srai_epi16,
        _mm256_unpacklo_epi8,
        _mm256_unpackhi_epi8,
        _mm256_maddubs_epi16,
        apply_single_avx2,
        apply_pair_avx2,
        apply_diff_avx2,
        apply_features_avx2,
        apply_features_to_acc_avx2,
        apply_combined_avx2,
        hybrid_combine_avx2,
        refresh_combine_avx2
    );

    define_fused_x86_path!(
        __m512i,
        16,
        32,
        512,
        "avx512f,avx512bw",
        _mm512_loadu_si512,
        _mm512_storeu_si512,
        _mm512_load_si512,
        _mm512_store_si512,
        _mm512_setzero_si512,
        _mm512_set1_epi8,
        _mm512_set1_epi16,
        _mm512_xor_si512,
        _mm512_add_epi16,
        _mm512_sub_epi16,
        _mm512_srai_epi16,
        _mm512_unpacklo_epi8,
        _mm512_unpackhi_epi8,
        _mm512_maddubs_epi16,
        apply_single_avx512,
        apply_pair_avx512,
        apply_diff_avx512,
        apply_features_avx512,
        apply_features_to_acc_avx512,
        apply_combined_avx512,
        hybrid_combine_avx512,
        refresh_combine_avx512
    );
}

/// Add (`SIGN == 1`) or subtract (`SIGN == -1`) the PSQ weight columns
/// of every index in `list`.
///
/// Port of the C++ `apply_psq_features` helper. The x86 paths keep one L1
/// stripe live while applying the complete feature list.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline(always)]
fn apply_psq_features<const SIGN: i32>(
    acc: &mut [BiasType; L1],
    list: &[u16],
    psq_weights: &[WeightType],
) {
    debug_assert!(SIGN == 1 || SIGN == -1);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        // SAFETY: AVX-512F/BW were just verified by `is_x86_feature_detected!`. `acc` is a fixed
        // `[i16; L1]` walked in 512-entry tiles (`L1 % 512 == 0`), and every index in `list` is a
        // PSQ feature below `BLOCK_DIMENSIONS`, so each column `index * L1 + base ..` lies
        // inside `psq_weights` (`L1 * BLOCK_DIMENSIONS` entries); all accesses are unaligned.
        unsafe { x86::apply_features_to_acc_avx512::<SIGN>(acc, list, psq_weights.as_ptr()) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. `acc` is a fixed `[i16;
        // L1]` walked in 128-entry tiles (`L1 % 128 == 0`), and every index in `list` is a PSQ
        // feature below `BLOCK_DIMENSIONS`, so each column lies inside `psq_weights`; all
        // accesses are unaligned.
        unsafe { x86::apply_features_to_acc_avx2::<SIGN>(acc, list, psq_weights.as_ptr()) };
        return;
    }

    for &index in list {
        let index = index as usize;
        let offset = L1 * index;
        for j in 0..L1 {
            if SIGN == 1 {
                acc[j] += psq_weights[offset + j] as BiasType;
            } else {
                acc[j] -= psq_weights[offset + j] as BiasType;
            }
        }
    }
}

/// Add (`SIGN == 1`) or subtract (`SIGN == -1`) the threat / pawn-pair weight
/// columns of every index in `list`.
///
/// Port of the C++ `apply_threat_features`. Threat weights are stored as
/// `i8` and are sign-extended to `i16` before accumulation.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline(always)]
fn apply_threat_features<const SIGN: i32>(
    acc: &mut [BiasType; L1],
    list: &[FeatureIndexType],
    ft: &FeatureTransformer,
) {
    debug_assert!(SIGN == 1 || SIGN == -1);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        // SAFETY: AVX-512F/BW were just verified by `is_x86_feature_detected!`. `acc` is a fixed
        // `[i16; L1]` walked in 512-entry tiles, and every index in `list` is a threat/pawn-pair
        // feature below `THREAT_AND_PAIR_INPUT_DIMENSIONS`, so each i8 column lies inside
        // `ft.threat_and_pp_weights`; all accesses are unaligned.
        unsafe {
            x86::apply_features_to_acc_avx512::<SIGN>(acc, list, ft.threat_and_pp_weights.as_ptr())
        };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. `acc` is a fixed `[i16;
        // L1]` walked in 128-entry tiles, and every index in `list` is a threat/pawn-pair feature
        // below `THREAT_AND_PAIR_INPUT_DIMENSIONS`, so each i8 column lies inside
        // `ft.threat_and_pp_weights`; all accesses are unaligned.
        unsafe {
            x86::apply_features_to_acc_avx2::<SIGN>(acc, list, ft.threat_and_pp_weights.as_ptr())
        };
        return;
    }

    for &index in list {
        let index = index as usize;
        let offset = L1 * index;
        for j in 0..L1 {
            if SIGN == 1 {
                acc[j] += ft.threat_and_pp_weights[offset + j] as BiasType;
            } else {
                acc[j] -= ft.threat_and_pp_weights[offset + j] as BiasType;
            }
        }
    }
}

/// Apply a combined PSQ + Threats update from `from` to `to`.
///
/// Port of the C++ `apply_combined`: tiled NEON on aarch64; AVX-512 then AVX2 kernels on
/// x86_64 when detected; otherwise the scalar reference path (C++ `#else` branch). The PSQ
/// and threat weights are `i8` (SFNNv18) and are sign-extended to `i16` before accumulation.
#[inline(never)]
fn apply_combined(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    psq_weights: &[WeightType],
    from: &AccumulatorState,
    to: &mut AccumulatorState,
    psq_added: &[u16],
    psq_removed: &[u16],
    thr_added: &[FeatureIndexType],
    thr_removed: &[FeatureIndexType],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    // SAFETY: see the module-level safety note on `neon`.
    unsafe {
        neon::apply_combined(
            perspective,
            feature_transformer,
            psq_weights.as_ptr(),
            from,
            to,
            psq_added,
            psq_removed,
            thr_added,
            thr_removed,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let p = perspective as usize;

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F/BW were just verified by `is_x86_feature_detected!`. Tiles stay
            // within the `[i16; L1]` accumulations of `from`/`to`, feature indices are bounded
            // by the feature-set dimensions so weight columns are in bounds, and all accesses are
            // unaligned (`loadu`/`storeu`).
            unsafe {
                x86::apply_combined_avx512(
                    p,
                    feature_transformer,
                    psq_weights.as_ptr(),
                    from,
                    to,
                    psq_added,
                    psq_removed,
                    thr_added,
                    thr_removed,
                );
            }
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. Tiles stay within the
            // `[i16; L1]` accumulations of `from`/`to` (unaligned accesses), and feature indices
            // are bounded by the feature-set dimensions so weight columns are in bounds.
            unsafe {
                x86::apply_combined_avx2(
                    p,
                    feature_transformer,
                    psq_weights.as_ptr(),
                    from,
                    to,
                    psq_added,
                    psq_removed,
                    thr_added,
                    thr_removed,
                );
            }
            return;
        }

        to.accumulation[p] = from.accumulation[p];

        // Reach through `.acc` explicitly: `AccumulatorState`'s `DerefMut` would make
        // the two field borrows below look like two borrows of the whole state.
        let to_acc = &mut to.acc;
        let acc = &mut to_acc.accumulation[p];

        // PSQ
        apply_psq_features::<-1>(acc, psq_removed, psq_weights);
        apply_psq_features::<1>(acc, psq_added, psq_weights);

        // Threats and pawn pairs (one concatenated index space)
        apply_threat_features::<-1>(acc, thr_removed, feature_transformer);
        apply_threat_features::<1>(acc, thr_added, feature_transformer);
    }
}

/// Rebucket the PSQ part of a king-move accumulator update, and bring the
/// Finny entry of the new king square up to date along the way.
///
/// Port of the accumulation half of C++ `update_accumulator_hybrid`.
#[inline(never)]
fn hybrid_combine(
    feature_transformer: &FeatureTransformer,
    psq_weights: &[WeightType],
    new_entry: &mut Entry,
    old_entry: &Entry,
    from_acc: &[BiasType; L1],
    to_acc: &mut [BiasType; L1],
    new_remove: &[u16],
    new_add: &[u16],
    old_remove: &[u16],
    old_add: &[u16],
    thr_removed: &[FeatureIndexType],
    thr_added: &[FeatureIndexType],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    // SAFETY: see the module-level safety note on `neon`.
    unsafe {
        neon::hybrid_combine(
            feature_transformer,
            psq_weights.as_ptr(),
            new_entry,
            old_entry,
            from_acc,
            to_acc,
            new_remove,
            new_add,
            old_remove,
            old_add,
            thr_removed,
            thr_added,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F/BW were just verified by `is_x86_feature_detected!`. All accumulator
            // accesses are unaligned tiles within the `[i16; L1]` arrays behind `new_entry`,
            // `old_entry`, `from_acc` and `to_acc`, and feature indices are bounded by the
            // feature-set dimensions so weight columns are in bounds.
            unsafe {
                x86::hybrid_combine_avx512(
                    feature_transformer,
                    psq_weights.as_ptr(),
                    new_entry,
                    old_entry,
                    from_acc,
                    to_acc,
                    new_remove,
                    new_add,
                    old_remove,
                    old_add,
                    thr_removed,
                    thr_added,
                );
            }
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. All accumulator
            // accesses are unaligned tiles within the `[i16; L1]` arrays behind `new_entry`,
            // `old_entry`, `from_acc` and `to_acc`, and feature indices are bounded by the
            // feature-set dimensions so weight columns are in bounds.
            unsafe {
                x86::hybrid_combine_avx2(
                    feature_transformer,
                    psq_weights.as_ptr(),
                    new_entry,
                    old_entry,
                    from_acc,
                    to_acc,
                    new_remove,
                    new_add,
                    old_remove,
                    old_add,
                    thr_removed,
                    thr_added,
                );
            }
            return;
        }

        // Bring the cache entry of the new king square up to date with the position.
        apply_psq_features::<-1>(&mut new_entry.accumulation, new_remove, psq_weights);
        apply_psq_features::<1>(&mut new_entry.accumulation, new_add, psq_weights);

        *to_acc = new_entry.accumulation;

        // Adding the old accumulator adds (most of) the threat and pawn-pair weights
        // that we need, but also a whole bunch of PSQ weights for the wrong king
        // bucket, which we remove via the old cache entry (and then adjust).
        for j in 0..L1 {
            to_acc[j] += from_acc[j];
            to_acc[j] -= old_entry.accumulation[j];
        }

        // The signs are inverted here: the old cache entry's contribution is being
        // undone, so what it "removed" has to be added back and vice versa.
        apply_psq_features::<1>(to_acc, old_remove, psq_weights);
        apply_psq_features::<-1>(to_acc, old_add, psq_weights);

        apply_threat_features::<-1>(to_acc, thr_removed, feature_transformer);
        apply_threat_features::<1>(to_acc, thr_added, feature_transformer);
    }
}

/// Update the Finny entry from the position, then build the accumulator as the
/// entry plus the active threat / pawn-pair contribution.
///
/// Port of the accumulation half of C++ `update_accumulator_refresh_cache`.
#[inline(never)]
fn refresh_combine(
    feature_transformer: &FeatureTransformer,
    psq_weights: &[WeightType],
    entry: &mut Entry,
    to_acc: &mut [BiasType; L1],
    removed: &[u16],
    added: &[u16],
    active: &[FeatureIndexType],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    // SAFETY: see the module-level safety note on `neon`.
    unsafe {
        neon::refresh_combine(
            feature_transformer,
            psq_weights.as_ptr(),
            entry,
            to_acc,
            removed,
            added,
            active,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F/BW were just verified by `is_x86_feature_detected!`. All accesses
            // are unaligned tiles within the `[i16; L1]` arrays behind `entry` and `to_acc`, and
            // feature indices are bounded by the feature-set dimensions so weight columns are in
            // bounds.
            unsafe {
                x86::refresh_combine_avx512(
                    feature_transformer,
                    psq_weights.as_ptr(),
                    entry,
                    to_acc,
                    removed,
                    added,
                    active,
                );
            }
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. All accesses are
            // unaligned tiles within the `[i16; L1]` arrays behind `entry` and `to_acc`, and
            // feature indices are bounded by the feature-set dimensions so weight columns are in
            // bounds.
            unsafe {
                x86::refresh_combine_avx2(
                    feature_transformer,
                    psq_weights.as_ptr(),
                    entry,
                    to_acc,
                    removed,
                    added,
                    active,
                );
            }
            return;
        }

        // Update the cache entry (PSQ only).
        apply_psq_features::<-1>(&mut entry.accumulation, removed, psq_weights);
        apply_psq_features::<1>(&mut entry.accumulation, added, psq_weights);

        // The cache entry has been updated; copy its content to the actual
        // accumulator we were refreshing.
        to_acc.copy_from_slice(&entry.accumulation);

        // Add the threat / pawn-pair contribution (those weights are i8, sign-extended).
        apply_threat_features::<1>(to_acc, active, feature_transformer);
    }
}

/// Single incremental update (forward or backward).
fn update_accumulator_incremental<const FORWARD: bool>(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    ksq: Square,
    psq_block: usize,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    debug_assert!(computed.computed[perspective as usize]);
    debug_assert!(!target_state.computed[perspective as usize]);

    // The size must be enough to contain the largest possible update. That
    // depends on the feature set and generally relies on the feature set's
    // update-cost calculation to be correct and never allow updates with more
    // added/removed features than MaxActiveDimensions.
    let mut psq_removed = IndexList::new();
    let mut psq_added = IndexList::new();
    // Threat and pawn-pair indices share one list: the pair indices are simply
    // concatenated to the threat ones (offset by `PairFeatureSet::INDEX_BASE`).
    let mut thr_removed = FeatureIndexList::new();
    let mut thr_added = FeatureIndexList::new();

    let dirties = if FORWARD {
        &target_state.dirties
    } else {
        &computed.dirties
    };

    if FORWARD {
        ThreatFeatureSet::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_threats,
            &mut thr_removed,
            &mut thr_added,
        );
        PairFeatureSet::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_pawn_pairs,
            &mut thr_removed,
            &mut thr_added,
        );
        PSQFeatureSet::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_piece,
            &mut psq_removed,
            &mut psq_added,
        );
    } else {
        ThreatFeatureSet::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_threats,
            &mut thr_added,
            &mut thr_removed,
        );
        PairFeatureSet::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_pawn_pairs,
            &mut thr_added,
            &mut thr_removed,
        );
        PSQFeatureSet::append_changed_indices(
            perspective,
            ksq,
            &dirties.dirty_piece,
            &mut psq_added,
            &mut psq_removed,
        );
    }

    apply_combined(
        perspective,
        feature_transformer,
        feature_transformer.psq_weights(psq_block),
        computed,
        target_state,
        &psq_added,
        &psq_removed,
        &thr_added,
        &thr_removed,
    );

    target_state.computed[perspective as usize] = true;
}

/// Forward incremental update of both perspectives from one dirty state.
///
/// Port of the C++ `update_accumulator_incremental_both`. Equivalent to calling
/// [`update_accumulator_incremental::<true>`] once per colour, except that the
/// threat and pawn-pair dirty lists are traversed a single time and both
/// king-relative indices are derived per entry.
fn update_accumulator_incremental_both(
    feature_transformer: &FeatureTransformer,
    white_ksq: Square,
    black_ksq: Square,
    white_block: usize,
    black_block: usize,
    target_state: &mut AccumulatorState,
    computed: &AccumulatorState,
) {
    debug_assert!(computed.computed[Color::WHITE as usize]);
    debug_assert!(computed.computed[Color::BLACK as usize]);
    debug_assert!(!target_state.computed[Color::WHITE as usize]);
    debug_assert!(!target_state.computed[Color::BLACK as usize]);

    let mut psq_removed = [IndexList::new(), IndexList::new()];
    let mut psq_added = [IndexList::new(), IndexList::new()];
    // Threat and pawn-pair indices share one list per colour: the pair indices
    // are concatenated to the threat ones (offset by `PairFeatureSet::INDEX_BASE`).
    let mut thr_removed = [FeatureIndexList::new(), FeatureIndexList::new()];
    let mut thr_added = [FeatureIndexList::new(), FeatureIndexList::new()];

    const W: usize = Color::WHITE as usize;
    const B: usize = Color::BLACK as usize;

    {
        let dirties = &target_state.dirties;
        let (thr_removed_w, thr_removed_b) = thr_removed.split_at_mut(B);
        let (thr_added_w, thr_added_b) = thr_added.split_at_mut(B);

        ThreatFeatureSet::append_changed_indices_both(
            white_ksq,
            black_ksq,
            &dirties.dirty_threats,
            &mut thr_removed_w[0],
            &mut thr_added_w[0],
            &mut thr_removed_b[0],
            &mut thr_added_b[0],
        );
        PairFeatureSet::append_changed_indices_both(
            white_ksq,
            black_ksq,
            &dirties.dirty_pawn_pairs,
            &mut thr_removed_w[0],
            &mut thr_added_w[0],
            &mut thr_removed_b[0],
            &mut thr_added_b[0],
        );
        PSQFeatureSet::append_changed_indices(
            Color::WHITE,
            white_ksq,
            &dirties.dirty_piece,
            &mut psq_removed[W],
            &mut psq_added[W],
        );
        PSQFeatureSet::append_changed_indices(
            Color::BLACK,
            black_ksq,
            &dirties.dirty_piece,
            &mut psq_removed[B],
            &mut psq_added[B],
        );
    }

    apply_combined(
        Color::WHITE,
        feature_transformer,
        feature_transformer.psq_weights(white_block),
        computed,
        target_state,
        &psq_added[W],
        &psq_removed[W],
        &thr_added[W],
        &thr_removed[W],
    );
    apply_combined(
        Color::BLACK,
        feature_transformer,
        feature_transformer.psq_weights(black_block),
        computed,
        target_state,
        &psq_added[B],
        &psq_removed[B],
        &thr_added[B],
        &thr_removed[B],
    );

    target_state.computed[Color::WHITE as usize] = true;
    target_state.computed[Color::BLACK as usize] = true;
}

/// Compare two piece arrays and return a bitboard of changed squares.
/// SIMD-optimized comparison uses the same AVX2/SSE2/NEON choices as C++.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn get_changed_pieces_avx2(old: &[Piece; SQUARE_NB], new: &[Piece; SQUARE_NB]) -> Bitboard {
    use core::arch::x86_64::*;

    debug_assert_eq!(core::mem::size_of::<Piece>(), 1);
    let old_ptr = old.as_ptr() as *const __m256i;
    let new_ptr = new.as_ptr() as *const __m256i;
    let mut same = 0u64;
    for i in 0..2 {
        let old_v = _mm256_loadu_si256(old_ptr.add(i));
        let new_v = _mm256_loadu_si256(new_ptr.add(i));
        let equal = _mm256_cmpeq_epi8(old_v, new_v);
        same |= (_mm256_movemask_epi8(equal) as u32 as u64) << (i * 32);
    }
    !same
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn get_changed_pieces_sse2(old: &[Piece; SQUARE_NB], new: &[Piece; SQUARE_NB]) -> Bitboard {
    use core::arch::x86_64::*;

    debug_assert_eq!(core::mem::size_of::<Piece>(), 1);
    let old_ptr = old.as_ptr() as *const __m128i;
    let new_ptr = new.as_ptr() as *const __m128i;
    let mut same = 0u64;
    for i in 0..4 {
        let old_v = _mm_loadu_si128(old_ptr.add(i));
        let new_v = _mm_loadu_si128(new_ptr.add(i));
        let equal = _mm_cmpeq_epi8(old_v, new_v);
        same |= (_mm_movemask_epi8(equal) as u32 as u64) << (i * 16);
    }
    !same
}

#[inline]
fn get_changed_pieces(old: &[Piece; SQUARE_NB], new: &[Piece; SQUARE_NB]) -> Bitboard {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. The kernel does two
            // unaligned 32-byte loads from each 64-byte `[Piece; SQUARE_NB]` array (`Piece` is 1
            // byte).
            return unsafe { get_changed_pieces_avx2(old, new) };
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 was just verified by `is_x86_feature_detected!`. The kernel does four
            // unaligned 16-byte loads from each 64-byte `[Piece; SQUARE_NB]` array (`Piece` is 1
            // byte).
            return unsafe { get_changed_pieces_sse2(old, new) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64. `vld4q_u8` reads 64 bytes from each `[Piece;
        // SQUARE_NB]` array (`Piece` is 1 byte, 64 squares); the remaining intrinsics are
        // register-only.
        unsafe {
            // vld4q_u8 deinterleaves 64 bytes: lane i of vector k is square 4*i + k.
            let old_ptr = old.as_ptr() as *const u8;
            let new_ptr = new.as_ptr() as *const u8;

            let old_v = vld4q_u8(old_ptr);
            let new_v = vld4q_u8(new_ptr);

            // Compare each lane
            let cmp0 = vceqq_u8(old_v.0, new_v.0);
            let cmp1 = vceqq_u8(old_v.1, new_v.1);
            let cmp2 = vceqq_u8(old_v.2, new_v.2);
            let cmp3 = vceqq_u8(old_v.3, new_v.3);

            // Merge comparisons using shift-right-insert
            let cmp0_1 = vsriq_n_u8(cmp1, cmp0, 1);
            let cmp2_3 = vsriq_n_u8(cmp3, cmp2, 1);
            let merged = vsriq_n_u8(cmp2_3, cmp0_1, 2);
            let merged = vsriq_n_u8(merged, merged, 4);

            // Narrow to 8 bytes
            let same_bb = vshrn_n_u16(vreinterpretq_u16_u8(merged), 4);

            // Extract as u64 and invert
            !vget_lane_u64(vreinterpret_u64_u8(same_bb), 0)
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        // Scalar fallback
        let mut changed: Bitboard = 0;
        for sq in 0..SQUARE_NB {
            if old[sq] != new[sq] {
                changed |= 1u64 << sq;
            }
        }
        changed
    }
}

/// Append the PSQ feature indices that differ between two piece arrays, given
/// the bitboards of the pieces present before and after the change.
///
/// `write_indices` takes the AVX-512 VBMI2 path when the build has it (C++ `USE_AVX512ICL`) and
/// the `make_index` loops otherwise.
#[inline(always)]
fn append_changed_psq_indices(
    perspective: Color,
    ksq: Square,
    old_pieces: &[Piece; SQUARE_NB],
    old_bb: Bitboard,
    new_pieces: &[Piece; SQUARE_NB],
    new_bb: Bitboard,
    removed: &mut IndexList,
    added: &mut IndexList,
) {
    let changed_bb = get_changed_pieces(old_pieces, new_pieces);
    let removed_bb = changed_bb & old_bb;
    let added_bb = changed_bb & new_bb;

    PSQFeatureSet::write_indices(
        old_pieces,
        new_pieces,
        removed_bb,
        added_bb,
        perspective,
        ksq,
        removed,
        added,
    );
}

/// Update the accumulator for a king move that stays on the same board half,
/// and update the accumulator cache for the new king position along the way.
///
/// Port of the C++ `update_accumulator_hybrid`; the accumulation kernels are dispatched in
/// `hybrid_combine`.
///
/// The threat and pawn-pair contributions do not depend on the king bucket, so
/// they can be carried over from the previous accumulator; only the PSQ part
/// has to be rebucketed. That is done by taking the (updated) Finny entry for the
/// new king square, adding the previous accumulator and subtracting the Finny
/// entry for the old king square adjusted to the pre-move board.
fn update_accumulator_hybrid(
    perspective: Color,
    pos: &Position,
    feature_transformer: &FeatureTransformer,
    target: &mut AccumulatorState,
    computed: &AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective as usize;
    let dirty_piece = target.dirties.dirty_piece;

    debug_assert_eq!(dirty_piece.pc, make_piece(perspective, PieceType::King));
    let (Some(old_ksq), Some(new_ksq)) = (dirty_piece.from, dirty_piece.to) else {
        unreachable!("a king move has both from and to");
    };
    debug_assert_eq!((old_ksq as i32) & 0b100, (new_ksq as i32) & 0b100);
    debug_assert!(computed.computed[p]);
    debug_assert!(!target.computed[p]);

    debug_assert_ne!(old_ksq, new_ksq);

    let current_pieces = pos.piece_array();
    let mut previous_pieces = *current_pieces; // Copy of the 64-byte piece array; modified below.

    let mut previous_piece_bb = pos.pieces();

    debug_assert_eq!(previous_pieces[new_ksq as usize], dirty_piece.pc);

    if dirty_piece.remove_sq.is_some() {
        debug_assert_eq!(dirty_piece.remove_sq, Some(new_ksq));
        previous_pieces[new_ksq as usize] = dirty_piece.remove_pc;
    } else {
        previous_pieces[new_ksq as usize] = Piece::NoPiece;
        previous_piece_bb &= !bitboard::square_bb_pub(new_ksq);
    }

    debug_assert_eq!(previous_pieces[old_ksq as usize], Piece::NoPiece);
    previous_pieces[old_ksq as usize] = make_piece(perspective, PieceType::King);
    previous_piece_bb |= bitboard::square_bb_pub(old_ksq);

    let psq_block = PSQFeatureSet::block(pos, perspective);
    let (old_entry, new_entry) = cache.get_pair(psq_block, old_ksq, new_ksq, perspective);
    let psq_weights = feature_transformer.psq_weights(psq_block);

    // "Remove" means we need to remove them from the cache entry,
    // "Add" means add them to the entry to get the accumulator we want.
    let mut old_remove = IndexList::new();
    let mut old_add = IndexList::new();
    let mut new_remove = IndexList::new();
    let mut new_add = IndexList::new();

    append_changed_psq_indices(
        perspective,
        old_ksq,
        &old_entry.pieces,
        old_entry.piece_bb,
        &previous_pieces,
        previous_piece_bb,
        &mut old_remove,
        &mut old_add,
    );
    append_changed_psq_indices(
        perspective,
        new_ksq,
        &new_entry.pieces,
        new_entry.piece_bb,
        current_pieces,
        pos.pieces(),
        &mut new_remove,
        &mut new_add,
    );

    // Threat and pawn-pair indices (one concatenated index space).
    let mut thr_removed = FeatureIndexList::new();
    let mut thr_added = FeatureIndexList::new();
    ThreatFeatureSet::append_changed_indices(
        perspective,
        new_ksq,
        &target.dirties.dirty_threats,
        &mut thr_removed,
        &mut thr_added,
    );
    PairFeatureSet::append_changed_indices(
        perspective,
        new_ksq,
        &target.dirties.dirty_pawn_pairs,
        &mut thr_removed,
        &mut thr_added,
    );

    hybrid_combine(
        feature_transformer,
        psq_weights,
        new_entry,
        old_entry,
        &computed.accumulation[p],
        &mut target.acc.accumulation[p],
        &new_remove,
        &new_add,
        &old_remove,
        &old_add,
        &thr_removed,
        &thr_added,
    );

    new_entry.pieces.copy_from_slice(current_pieces);
    new_entry.piece_bb = pos.pieces();

    target.computed[p] = true;
}

/// Refresh the PSQ part of the accumulator from the Finny cache entry, then
/// rebuild the threat contribution from the active threat features.
///
/// Port of the C++ `update_accumulator_refresh_cache`; the accumulation kernels are dispatched
/// in `refresh_combine`.
///
/// PSQ data comes from the Finny table entry, while the threats are built
/// from the active threat features.
fn update_accumulator_refresh_cache(
    perspective: Color,
    feature_transformer: &FeatureTransformer,
    pos: &Position,
    accumulator: &mut AccumulatorState,
    cache: &mut AccumulatorCaches,
) {
    let p = perspective as usize;
    let psq_block = PSQFeatureSet::block(pos, perspective);
    let ksq = pos.king_square(perspective);
    let entry = &mut cache.get(psq_block)[ksq as usize][p];
    let psq_weights = feature_transformer.psq_weights(psq_block);

    let mut removed = IndexList::new();
    let mut added = IndexList::new();

    append_changed_psq_indices(
        perspective,
        ksq,
        &entry.pieces,
        entry.piece_bb,
        pos.piece_array(),
        pos.pieces(),
        &mut removed,
        &mut added,
    );

    entry.piece_bb = pos.pieces();
    entry.pieces.copy_from_slice(pos.piece_array());

    // Active threat and pawn-pair features are added on top of the PSQ cache
    // result (pawn-pair indices are concatenated to the threat ones).
    let mut active = FeatureIndexList::new();
    ThreatFeatureSet::append_active_indices(perspective, pos, &mut active);
    PairFeatureSet::append_active_indices(perspective, pos, &mut active);

    accumulator.computed[p] = true;

    refresh_combine(
        feature_transformer,
        psq_weights,
        entry,
        &mut accumulator.acc.accumulation[p],
        &removed,
        &added,
        &active,
    );
}

// ---------------------------------------------------------------------------
// Position helpers
// ---------------------------------------------------------------------------

/// Extension trait for Position - wrapping existing Position methods.
pub trait PositionExt {
    fn king_square(&self, color: Color) -> Square;
}

impl PositionExt for Position {
    /// Get king square for a color
    #[inline(always)]
    fn king_square(&self, color: Color) -> Square {
        self.square_of(color, PieceType::King)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_default() {
        let acc = Accumulator::default();
        assert!(!acc.computed[0]);
        assert!(!acc.computed[1]);
        assert_eq!(acc.accumulation[0][0], 0);
    }

    #[test]
    fn test_accumulator_stack() {
        crate::bitboard::init();
        let mut pos = Position::default();
        if pos
            .set_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", false)
            .is_err()
        {
            panic!("bad fen");
        }
        let mut stack = AccumulatorStack::new();
        assert_eq!(stack.size, 1);

        // SAFETY: popped again below without an evaluation, so the empty `Dirties` is never read.
        unsafe { stack.push(&pos) };
        assert_eq!(stack.size, 2);
        // The PSQ blocks of the position before the move are recorded: rook, no queen.
        assert_eq!(stack.latest().block_before, [1, 1]);

        stack.pop();
        assert_eq!(stack.size, 1);
    }

    #[test]
    fn test_cache_entry() {
        let mut entry = Entry::default();
        let biases = vec![42i16; L1];
        entry.clear(&biases);
        assert_eq!(entry.accumulation[0], 42);
    }

    #[test]
    fn accumulator_state_layout_matches_cpp() {
        // C++ places `Dirties` in the accumulator's tail padding and `blockBefore` after it
        // (offset 4536, sizeof 4544).
        assert_eq!(core::mem::size_of::<AccumulatorState>(), 4544);
        assert_eq!(core::mem::offset_of!(AccumulatorState, block_before), 4536);
        assert_eq!(core::mem::align_of::<AccumulatorState>(), 64);
    }

    #[test]
    fn test_accumulator_state() {
        let state = AccumulatorState::default();
        assert!(!state.computed[0]);
        assert!(!state.computed[1]);
    }

    #[test]
    fn test_find_last_usable() {
        crate::bitboard::init();
        let pos = Position::default();
        let stack = AccumulatorStack::new();
        // At initialization, the last usable should be 0 (the root)
        let last = stack.find_last_usable_accumulator(Color::WHITE, &pos);
        assert_eq!(last, 0);
    }

    #[test]
    fn changed_piece_mask_matches_reference() {
        let old = [Piece::NoPiece; SQUARE_NB];
        let mut new = old;
        new[0] = Piece::W_King;
        new[17] = Piece::B_Queen;
        new[63] = Piece::W_Pawn;
        assert_eq!(
            get_changed_pieces(&old, &new),
            (1u64 << 0) | (1u64 << 17) | (1u64 << 63)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_column_updates_match_scalar_reference() {
        let mut ft = FeatureTransformer::new();
        let indices = [0usize, 1, 2];
        let psq_indices = [0u16, 1, 2];
        let threat_indices = [0u16, 1, 2];
        for &index in &indices {
            for j in 0..L1 {
                ft.weights[index * L1 + j] = ((j as i32 * 13 + index as i32 * 7) % 127 - 63) as i8;
                ft.threat_and_pp_weights[index * L1 + j] =
                    ((j as i32 * 5 + index as i32 * 11) % 101 - 50) as i8;
            }
        }
        let mut got_acc = [17i16; L1];
        let mut want_acc = got_acc;
        for &index in &indices {
            for j in 0..L1 {
                want_acc[j] += ft.weights[index * L1 + j] as i16;
            }
        }
        let mut want_after_threats = want_acc;
        for &index in &indices {
            for j in 0..L1 {
                want_after_threats[j] -= ft.threat_and_pp_weights[index * L1 + j] as i16;
            }
        }
        // The kernels read the rows in the load-time (widening pair) order.
        FeatureTransformer::permute_widening_pairs(
            &mut ft.weights,
            &mut ft.threat_and_pp_weights,
            false,
        );
        apply_psq_features::<1>(&mut got_acc, &psq_indices, &ft.weights);
        assert_eq!(got_acc, want_acc);
        apply_threat_features::<-1>(&mut got_acc, &threat_indices, &ft);
        assert_eq!(got_acc, want_after_threats);
    }

    /// The fused x86 kernels (incremental, hybrid and refresh, AVX2 and AVX-512) match a scalar
    /// reference, including the leftover add/sub pair applied as one `maddubs` diff and extreme
    /// weights. Each kernel reads the rows in its own widening-pair order.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_fused_kernels_match_scalar_reference() {
        const PSQ_BLOCK: usize = 2;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let psq_rows: Vec<u16> = (0..12u16).map(|i| i * 1877 + 3).collect();
        let thr_rows: Vec<u16> = (0..12u16).map(|i| i * 5333 + 11).collect();
        assert!(
            psq_rows
                .iter()
                .all(|&r| (r as usize) < PSQFeatureSet::BLOCK_DIMENSIONS)
        );
        let random_row = |next: &mut dyn FnMut() -> u64| -> Vec<i8> {
            (0..L1)
                .map(|_| match next() % 8 {
                    0 => -128,
                    1 => 127,
                    _ => next() as i8,
                })
                .collect()
        };
        let psq_w: Vec<Vec<i8>> = psq_rows.iter().map(|_| random_row(&mut next)).collect();
        let thr_w: Vec<Vec<i8>> = thr_rows.iter().map(|_| random_row(&mut next)).collect();
        let apply = |acc: &mut [i16; L1], rows: &[u16], ids: &[u16], w: &[Vec<i8>], sign: i16| {
            for &r in rows {
                let row = &w[ids.iter().position(|&x| x == r).unwrap()];
                for (a, &x) in acc.iter_mut().zip(row) {
                    *a = a.wrapping_add(sign.wrapping_mul(x as i16));
                }
            }
        };
        let pick = |next: &mut dyn FnMut() -> u64, rows: &[u16], lo: u64, hi: u64| -> Vec<u16> {
            let n = lo + next() % (hi - lo + 1);
            let mut v = rows.to_vec();
            for i in (1..v.len()).rev() {
                v.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            v.truncate(n as usize);
            v
        };
        let random_acc = |next: &mut dyn FnMut() -> u64| -> [i16; L1] {
            core::array::from_fn(|_| next() as i16)
        };

        let mut ft = FeatureTransformer::new();
        let block_base = PSQFeatureSet::block_offset(PSQ_BLOCK) * L1;
        // `order[j]` is the source 8-byte block of destination block `j` (C++ WideningPairOrder).
        let isas: [(&str, bool, &[usize]); 2] = [
            (
                "avx2",
                std::arch::is_x86_feature_detected!("avx2"),
                &[0, 2, 1, 3],
            ),
            (
                "avx512",
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw"),
                &[0, 4, 1, 5, 2, 6, 3, 7],
            ),
        ];
        for (name, available, order) in isas {
            if !available {
                continue;
            }
            let permuted = |row: &[i8]| -> Vec<i8> {
                let chunk = order.len() * 8;
                let mut out = row.to_vec();
                for c in (0..L1).step_by(chunk) {
                    for (j, &src) in order.iter().enumerate() {
                        out[c + j * 8..c + j * 8 + 8]
                            .copy_from_slice(&row[c + src * 8..c + src * 8 + 8]);
                    }
                }
                out
            };
            for (&r, w) in psq_rows.iter().zip(&psq_w) {
                let at = block_base + r as usize * L1;
                ft.weights[at..at + L1].copy_from_slice(&permuted(w));
            }
            for (&r, w) in thr_rows.iter().zip(&thr_w) {
                let at = r as usize * L1;
                ft.threat_and_pp_weights[at..at + L1].copy_from_slice(&permuted(w));
            }
            let psq_base = ft.psq_weights(PSQ_BLOCK).as_ptr();

            for _ in 0..300 {
                // Incremental update: one or two PSQ rows of each sign.
                let (pr, pa) = (
                    pick(&mut next, &psq_rows, 1, 2),
                    pick(&mut next, &psq_rows, 1, 2),
                );
                let (tr, ta) = (
                    pick(&mut next, &thr_rows, 0, 6),
                    pick(&mut next, &thr_rows, 0, 6),
                );
                let mut from = AccumulatorState::default();
                from.acc.accumulation[1] = random_acc(&mut next);
                let mut want = from.acc.accumulation[1];
                apply(&mut want, &pr, &psq_rows, &psq_w, -1);
                apply(&mut want, &tr, &thr_rows, &thr_w, -1);
                apply(&mut want, &pa, &psq_rows, &psq_w, 1);
                apply(&mut want, &ta, &thr_rows, &thr_w, 1);
                let mut to = AccumulatorState::default();
                // SAFETY: the ISA was detected above; all indices are in range for the weights.
                unsafe {
                    if name == "avx2" {
                        x86::apply_combined_avx2(
                            1, &ft, psq_base, &from, &mut to, &pa, &pr, &ta, &tr,
                        );
                    } else {
                        x86::apply_combined_avx512(
                            1, &ft, psq_base, &from, &mut to, &pa, &pr, &ta, &tr,
                        );
                    }
                }
                assert_eq!(to.acc.accumulation[1], want, "{name} incremental");

                // Hybrid update and refresh: any number of rows.
                let lists: [Vec<u16>; 4] =
                    core::array::from_fn(|_| pick(&mut next, &psq_rows, 0, 5));
                let [nr, na, or, oa] = &lists;
                let mut new_entry = Entry::default();
                new_entry.accumulation = random_acc(&mut next);
                let mut old_entry = Entry::default();
                old_entry.accumulation = random_acc(&mut next);
                let from_acc = random_acc(&mut next);
                let mut want_entry = new_entry.accumulation;
                apply(&mut want_entry, nr, &psq_rows, &psq_w, -1);
                apply(&mut want_entry, na, &psq_rows, &psq_w, 1);
                let mut want_to = want_entry;
                for j in 0..L1 {
                    want_to[j] = want_to[j]
                        .wrapping_add(from_acc[j])
                        .wrapping_sub(old_entry.accumulation[j]);
                }
                apply(&mut want_to, oa, &psq_rows, &psq_w, -1);
                apply(&mut want_to, &tr, &thr_rows, &thr_w, -1);
                apply(&mut want_to, or, &psq_rows, &psq_w, 1);
                apply(&mut want_to, &ta, &thr_rows, &thr_w, 1);
                let mut to_acc = [0i16; L1];
                // SAFETY: as above.
                unsafe {
                    if name == "avx2" {
                        x86::hybrid_combine_avx2(
                            &ft,
                            psq_base,
                            &mut new_entry,
                            &old_entry,
                            &from_acc,
                            &mut to_acc,
                            nr,
                            na,
                            or,
                            oa,
                            &tr,
                            &ta,
                        );
                    } else {
                        x86::hybrid_combine_avx512(
                            &ft,
                            psq_base,
                            &mut new_entry,
                            &old_entry,
                            &from_acc,
                            &mut to_acc,
                            nr,
                            na,
                            or,
                            oa,
                            &tr,
                            &ta,
                        );
                    }
                }
                assert_eq!(new_entry.accumulation, want_entry, "{name} hybrid entry");
                assert_eq!(to_acc, want_to, "{name} hybrid");

                let mut entry = Entry::default();
                entry.accumulation = random_acc(&mut next);
                let mut want_entry = entry.accumulation;
                apply(&mut want_entry, nr, &psq_rows, &psq_w, -1);
                apply(&mut want_entry, na, &psq_rows, &psq_w, 1);
                let mut want_to = want_entry;
                apply(&mut want_to, &ta, &thr_rows, &thr_w, 1);
                // SAFETY: as above.
                unsafe {
                    if name == "avx2" {
                        x86::refresh_combine_avx2(
                            &ft,
                            psq_base,
                            &mut entry,
                            &mut to_acc,
                            nr,
                            na,
                            &ta,
                        );
                    } else {
                        x86::refresh_combine_avx512(
                            &ft,
                            psq_base,
                            &mut entry,
                            &mut to_acc,
                            nr,
                            na,
                            &ta,
                        );
                    }
                }
                assert_eq!(entry.accumulation, want_entry, "{name} refresh entry");
                assert_eq!(to_acc, want_to, "{name} refresh");
            }
        }
    }
}
