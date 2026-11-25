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

//! NNUE PP_3Wide (pawn pair) feature set.
//!
//! Ported from `src/nnue/features/pp_3wide.{h,cpp}` (SFNNv16). This is the third
//! feature block, appended right after the threat features in the feature
//! transformer, hence [`PP_3Wide::INDEX_BASE`] must equal
//! `FullThreats::DIMENSIONS`.
//!
//! A feature is an unordered pair of pawns that sit within a "3 wide" band of
//! each other (same or adjacent file, ranks 2-7), encoded by their oriented
//! (perspective relative) pawn ids.

// A scalar implementation and an x86-64 AVX-512 (VBMI2) implementation of `append_changed_indices`
// coexist in this file and the target picks one at compile time (`scripts/check-all.sh` compiles
// every target to keep both building).

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::bitboard_fast::{pawn_pair_bb, pop_lsb, square_bb_pub};
use crate::position::Position;
use crate::types::{Bitboard, COLOR_NB, Color, DirtyPawnPairs, PieceType, Square};

use super::super::nnue_common::{FeatureIndexList, IndexType, ORIENT_TBL};

/// C++ `make_pawn_id(Color color, Square square)`.
///
/// Pawn ids are dense over the 48 squares on ranks 2-7, offset by 48 for black.
#[inline(always)]
const fn make_pawn_id(color: Color, square: u8) -> IndexType {
    debug_assert!(square >= Square::SQ_A2 as u8 && square <= Square::SQ_H7 as u8);
    48 * (color as IndexType) + square as IndexType - Square::SQ_A2 as IndexType
}

/// Flip a color when the perspective is black. C++ `Color(c ^ perspective)`.
#[inline(always)]
const fn flip_color(c: Color, perspective: Color) -> Color {
    if ((c as u8) ^ (perspective as u8)) == 0 {
        Color::White
    } else {
        Color::Black
    }
}

/// Pawn pair feature set (C++ `Features::PP_3Wide`).
#[allow(non_camel_case_types)] // named after the C++ type
pub struct PP_3Wide;

impl PP_3Wide {
    /// Hash value embedded in the evaluation file
    pub const HASH_VALUE: u32 = 0x5c80fa69;

    /// Number of distinct pawn ids: 48 squares (ranks 2-7) per color.
    pub const PAWN_IDS: IndexType = COLOR_NB * 48;

    /// Number of feature dimensions: all unordered pairs of pawn ids.
    pub const DIMENSIONS: IndexType = Self::PAWN_IDS * (Self::PAWN_IDS - 1) / 2;

    /// Pawn pair feature indices are concatenated to threats, so this must equal
    /// `FullThreats::DIMENSIONS`; see `nnue_feature_transformer.rs`.
    pub const INDEX_BASE: IndexType = 59808;

    /// Index of the pawn pair feature (`from`, `to`) seen from `perspective`.
    ///
    /// `color` is the color of the pawn on `from`, `paired_color` the color of
    /// the pawn on `to`; the pair is unordered, so the two ids are sorted.
    #[inline(always)]
    pub fn make_index(
        perspective: Color,
        color: Color,
        from: Square,
        to: Square,
        paired_color: Color,
        ksq: Square,
    ) -> IndexType {
        let orientation = ORIENT_TBL[ksq as usize] ^ (56 * perspective as u8);
        Self::make_index_with_orientation(perspective, color, from, to, paired_color, orientation)
    }

    /// [`Self::make_index`] with a precomputed orientation, so callers that emit
    /// many indices for one king square only compute it once.
    #[inline(always)]
    pub fn make_index_with_orientation(
        perspective: Color,
        color: Color,
        from: Square,
        to: Square,
        paired_color: Color,
        orientation: u8,
    ) -> IndexType {
        let from_oriented = (from as u8) ^ orientation;
        let to_oriented = (to as u8) ^ orientation;

        let color_oriented = flip_color(color, perspective);
        let paired_color_oriented = flip_color(paired_color, perspective);

        debug_assert!(from_oriented >= Square::SQ_A2 as u8 && from_oriented <= Square::SQ_H7 as u8);
        debug_assert!(to_oriented >= Square::SQ_A2 as u8 && to_oriented <= Square::SQ_H7 as u8);

        let id_a = make_pawn_id(color_oriented, from_oriented);
        let id_b = make_pawn_id(paired_color_oriented, to_oriented);
        pp_idx(id_a, id_b)
    }

    /// Append all active pawn pair features for `pos`.
    ///
    /// Every pawn is "updated", so each unordered pair is emitted exactly once by
    /// `append_indices_impl`.
    #[inline(never)]
    pub(crate) fn append_active_indices(
        perspective: Color,
        pos: &Position,
        active: &mut FeatureIndexList,
    ) {
        let white = pos.pieces_color_types(Color::White, &[PieceType::Pawn]);
        let black = pos.pieces_color_types(Color::Black, &[PieceType::Pawn]);
        append_indices_impl::<1>(
            perspective,
            [pos.square_of(perspective, PieceType::King)],
            white | black,
            white,
            black,
            [active],
        );
    }

    /// Append the pawn pair features that changed between two positions.
    ///
    /// For each side of the diff, the pairs that involve at least one *updated* pawn square are
    /// regenerated: partners are taken from the unchanged pawns plus the still-unvisited updated
    /// pawns, so a pair with two updated pawns is emitted exactly once.
    #[inline(always)]
    pub(crate) fn append_changed_indices(
        perspective: Color,
        ksq: Square,
        diff: &DirtyPawnPairs,
        removed: &mut FeatureIndexList,
        added: &mut FeatureIndexList,
    ) {
        // The "no pawn changed" exit (most moves) is tested inline, like C++, so that the
        // out-of-line body's prologue/epilogue is only paid when there is work to do.
        if diff.before == diff.after {
            return;
        }
        append_changed_indices_impl::<1>(perspective, [ksq], diff, [removed], [added]);
    }

    /// Append the changed pawn-pair features for *both* perspectives in a single
    /// pass over the diff (C++ `PP_3Wide::append_changed_indices_both`).
    #[inline(never)]
    pub(crate) fn append_changed_indices_both(
        white_ksq: Square,
        black_ksq: Square,
        diff: &DirtyPawnPairs,
        white_removed: &mut FeatureIndexList,
        white_added: &mut FeatureIndexList,
        black_removed: &mut FeatureIndexList,
        black_added: &mut FeatureIndexList,
    ) {
        if diff.before == diff.after {
            return;
        }
        append_changed_indices_impl::<2>(
            Color::White,
            [white_ksq, black_ksq],
            diff,
            [white_removed, black_removed],
            [white_added, black_added],
        );
    }
}

/// C++ `pp_idx`: the index of the unordered pair of pawn ids `a` and `b`.
#[inline(always)]
const fn pp_idx(a: IndexType, b: IndexType) -> IndexType {
    let hi = if a > b { a } else { b };
    let lo = if a > b { b } else { a };
    hi * (hi - 1) / 2 + lo + PP_3Wide::INDEX_BASE
}

/// Perspective `i` of an `N`-perspective walk: `perspective` for 0, its opposite for 1 (C++
/// `perspective ^ i`).
#[inline(always)]
const fn nth_perspective(perspective: Color, i: usize) -> Color {
    flip_color(
        perspective,
        if i == 0 { Color::White } else { Color::Black },
    )
}

/// Append the pairs of every pawn in `updated` (with the unchanged pawns and the updated pawns
/// not yet visited) to `out[i]`, seen from `nth_perspective(perspective, i)` with king square
/// `ksq[i]`. The pawns are enumerated once for all `N` perspectives.
#[inline(always)]
fn append_indices_impl<const N: usize>(
    perspective: Color,
    ksq: [Square; N],
    updated: Bitboard,
    pawns_w: Bitboard,
    pawns_b: Bitboard,
    out: [&mut FeatureIndexList; N],
) {
    const { assert!(N == 1 || N == 2) };

    if updated == 0 {
        return;
    }

    let mut orientation = [0u8; N];
    for i in 0..N {
        orientation[i] = ORIENT_TBL[ksq[i] as usize] ^ (56 * nth_perspective(perspective, i) as u8);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    // SAFETY: the native build has all instructions used by the ICL path enabled at compile time.
    unsafe {
        append_indices_avx512::<N>(perspective, orientation, updated, pawns_w, pawns_b, out);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    )))]
    {
        use super::super::nnue_common::FeatureIndexType;

        let mut out = out;
        // Enumerate changed pawns and their partners once for all requested perspectives.
        let unchanged = (pawns_w | pawns_b) & !updated;
        let mut u = updated;
        while u != 0 {
            let a = pop_lsb(&mut u);
            let partners = pawn_pair_bb(a) & (unchanged | u);
            if partners == 0 {
                continue;
            }

            let a_col = if pawns_b & square_bb_pub(a) != 0 {
                Color::Black
            } else {
                Color::White
            };
            let mut a_id = [0; N];
            for i in 0..N {
                a_id[i] = make_pawn_id(
                    flip_color(a_col, nth_perspective(perspective, i)),
                    a as u8 ^ orientation[i],
                );
            }

            let mut push = |to: Square, paired_color: Color| {
                for i in 0..N {
                    let b_id = make_pawn_id(
                        flip_color(paired_color, nth_perspective(perspective, i)),
                        to as u8 ^ orientation[i],
                    );
                    out[i].push(pp_idx(a_id[i], b_id) as FeatureIndexType);
                }
            };

            let mut pb = pawns_b & partners;
            while pb != 0 {
                push(pop_lsb(&mut pb), Color::Black);
            }
            let mut pw = pawns_w & partners;
            while pw != 0 {
                push(pop_lsb(&mut pw), Color::White);
            }
        }
    }
}

/// `USE_AVX512ICL` body of [`append_indices_impl`]: the partner ids of each updated pawn are
/// gathered with one byte compress per perspective, and their pair indices are computed and
/// stored 16 at a time.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi2"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
unsafe fn append_indices_avx512<const N: usize>(
    perspective: Color,
    orientation: [u8; N],
    updated: Bitboard,
    pawns_w: Bitboard,
    pawns_b: Bitboard,
    out: [&mut FeatureIndexList; N],
) {
    use core::arch::x86_64::*;

    const fn all_squares() -> [i8; 64] {
        let mut squares = [0i8; 64];
        let mut i = 0;
        while i < 64 {
            squares[i] = i as i8;
            i += 1;
        }
        squares
    }

    static ALL_SQUARES: [i8; 64] = all_squares();

    let enemy_of = |i: usize| {
        if nth_perspective(perspective, i) == Color::White {
            pawns_b
        } else {
            pawns_w
        }
    };

    let iota = _mm512_loadu_si512(ALL_SQUARES.as_ptr() as *const __m512i);
    let mut ids = [_mm512_setzero_si512(); N];
    for i in 0..N {
        let adjusted = _mm512_sub_epi8(
            _mm512_xor_si512(iota, _mm512_set1_epi8(orientation[i] as i8)),
            _mm512_set1_epi8(8),
        );
        ids[i] = _mm512_mask_add_epi8(adjusted, enemy_of(i), adjusted, _mm512_set1_epi8(48));
    }

    // Enumerate changed pawns and their partners once for all requested perspectives.
    let unchanged = (pawns_w | pawns_b) & !updated;
    let mut u = updated;
    while u != 0 {
        let a = pop_lsb(&mut u);
        let partners = pawn_pair_bb(a) & (unchanged | u);
        if partners == 0 {
            continue;
        }

        let n = partners.count_ones() as usize;
        debug_assert!(n <= 16);

        for i in 0..N {
            let color_offset = if enemy_of(i) & square_bb_pub(a) != 0 {
                48u16
            } else {
                0
            };
            let a_id = ((a as u8 ^ orientation[i]).wrapping_sub(8) as u16) + color_offset;
            let compressed = _mm512_maskz_compress_epi8(partners, ids[i]);
            let pids = _mm256_cvtepu8_epi16(_mm512_castsi512_si128(compressed));
            let a_vec = _mm256_set1_epi16(a_id as i16);
            let hi = _mm256_max_epu16(pids, a_vec);
            let lo = _mm256_min_epu16(pids, a_vec);
            let product = _mm256_mullo_epi16(hi, _mm256_sub_epi16(hi, _mm256_set1_epi16(1)));
            // INDEX_BASE (59808) does not fit i16; `as i16` wraps, and the lane add is modular
            // mod 2^16, so the lanes read back as u16 equal the true index (max
            // 59808 + 4559 < 65536).
            let feats = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_srli_epi16::<1>(product), lo),
                _mm256_set1_epi16(PP_3Wide::INDEX_BASE as i16),
            );

            // SAFETY: `n <= 16` (at most 15 partners in a legal position). Position::set material
            // limits bound refresh lists to 232 entries: 80 + 8 * (16 - P) + 2 * P + P * (P - 1) / 2,
            // for P <= 16 pawns. Updates have at most 96 threats + 120 pairs. Both fit in 256
            // entries including the 15 extra lanes written by an unmasked store, so
            // `len + 16 <= MAX_FEATURE_INDEXES`. The first `n` of the 16 stored lanes are valid.
            let dst = out[i].make_space(n.min(16), 16);
            _mm256_storeu_si256(dst as *mut __m256i, feats);
        }
    }
}

/// Shared body of [`PP_3Wide::append_changed_indices`] and
/// [`PP_3Wide::append_changed_indices_both`].
#[inline(never)]
fn append_changed_indices_impl<const N: usize>(
    perspective: Color,
    ksq: [Square; N],
    diff: &DirtyPawnPairs,
    removed: [&mut FeatureIndexList; N],
    added: [&mut FeatureIndexList; N],
) {
    let white_before = diff.before[Color::White as usize];
    let black_before = diff.before[Color::Black as usize];
    let white_after = diff.after[Color::White as usize];
    let black_after = diff.after[Color::Black as usize];

    if white_before == white_after && black_before == black_after {
        return;
    }

    append_indices_impl::<N>(
        perspective,
        ksq,
        (white_after & !white_before) | (black_after & !black_before),
        white_after,
        black_after,
        added,
    );
    append_indices_impl::<N>(
        perspective,
        ksq,
        (white_before & !white_after) | (black_before & !black_after),
        white_before,
        black_before,
        removed,
    );
}

#[cfg(test)]
mod tests {
    use super::super::full_threats::FullThreats;
    use super::*;

    #[test]
    fn test_dimensions() {
        assert_eq!(PP_3Wide::PAWN_IDS, 96);
        assert_eq!(PP_3Wide::DIMENSIONS, 4560);
        // Pawn pair features are concatenated after the threat features.
        assert_eq!(PP_3Wide::INDEX_BASE, FullThreats::DIMENSIONS);
    }

    /// Slow reference: every pair (with at least one pawn in `updated`) of the pawn sets,
    /// through `make_index`, sorted.
    fn reference_pairs(
        perspective: Color,
        ksq: Square,
        updated: Bitboard,
        white: Bitboard,
        black: Bitboard,
    ) -> Vec<u16> {
        let color_of = |s: Square| {
            if black & square_bb_pub(s) != 0 {
                Color::Black
            } else {
                Color::White
            }
        };
        let mut out = Vec::new();
        let all = white | black;
        for a in 0..64u8 {
            let a = Square::from(a);
            if all & square_bb_pub(a) == 0 {
                continue;
            }
            for b in (a as u8 + 1)..64 {
                let b = Square::from(b);
                if all & square_bb_pub(b) == 0
                    || pawn_pair_bb(a) & square_bb_pub(b) == 0
                    || (updated & (square_bb_pub(a) | square_bb_pub(b))) == 0
                {
                    continue;
                }
                out.push(
                    PP_3Wide::make_index(perspective, color_of(a), a, b, color_of(b), ksq) as u16,
                );
            }
        }
        out.sort_unstable();
        out
    }

    fn sorted(list: &FeatureIndexList) -> Vec<u16> {
        let mut v = list.as_slice().to_vec();
        v.sort_unstable();
        v
    }

    /// The shared enumeration (scalar or AVX-512 ICL, whichever the build uses) emits exactly
    /// the reference pairs, for one perspective and for both at once.
    #[test]
    fn test_changed_indices_match_reference() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Ranks 2-7.
        const PAWN_SQUARES: Bitboard = 0x00FF_FFFF_FFFF_FF00;
        let random_pawns = |next: &mut dyn FnMut() -> u64, taken: Bitboard| {
            let mut bb = 0;
            for _ in 0..(next() % 9) {
                bb |= 1u64 << (8 + next() % 48);
            }
            bb & PAWN_SQUARES & !taken
        };

        for _ in 0..3000 {
            let w0 = random_pawns(&mut next, 0);
            let b0 = random_pawns(&mut next, w0);
            // Move, remove or add up to two pawns of each color.
            let (mut w1, mut b1) = (w0, b0);
            for _ in 0..(next() % 3) {
                w1 ^= 1u64 << (8 + next() % 48);
                b1 ^= 1u64 << (8 + next() % 48);
            }
            w1 &= PAWN_SQUARES;
            b1 &= PAWN_SQUARES & !w1;
            let diff = DirtyPawnPairs {
                before: [w0, b0],
                after: [w1, b1],
            };
            let ksq = [
                Square::from((next() % 64) as u8),
                Square::from((next() % 64) as u8),
            ];
            let (up_added, up_removed) = ((w1 & !w0) | (b1 & !b0), (w0 & !w1) | (b0 & !b1));

            let mut removed = FeatureIndexList::new();
            let mut added = FeatureIndexList::new();
            PP_3Wide::append_changed_indices(Color::Black, ksq[1], &diff, &mut removed, &mut added);
            assert_eq!(
                sorted(&added),
                reference_pairs(Color::Black, ksq[1], up_added, w1, b1)
            );
            assert_eq!(
                sorted(&removed),
                reference_pairs(Color::Black, ksq[1], up_removed, w0, b0)
            );

            let mut lists: [FeatureIndexList; 4] =
                core::array::from_fn(|_| FeatureIndexList::new());
            let [wr, wa, br, ba] = &mut lists;
            PP_3Wide::append_changed_indices_both(ksq[0], ksq[1], &diff, wr, wa, br, ba);
            for (c, k, r, a) in [
                (Color::White, ksq[0], &*wr, &*wa),
                (Color::Black, ksq[1], &*br, &*ba),
            ] {
                assert_eq!(sorted(a), reference_pairs(c, k, up_added, w1, b1));
                assert_eq!(sorted(r), reference_pairs(c, k, up_removed, w0, b0));
            }
        }
    }

    #[test]
    fn test_active_indices_match_reference() {
        crate::bitboard::init();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1BP2/PP1Q2PP/R3KB1R b KQ - 0 9",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "2k5/8/1pP1p3/1P2P3/p1pP1p2/P1P2P2/6K1/8 w - - 0 1",
        ] {
            let mut pos = Position::default();
            if pos.set_fen(fen, false).is_err() {
                panic!("bad fen {fen}");
            }
            let white = pos.pieces_color_types(Color::White, &[PieceType::Pawn]);
            let black = pos.pieces_color_types(Color::Black, &[PieceType::Pawn]);
            for perspective in [Color::White, Color::Black] {
                let ksq = pos.square_of(perspective, PieceType::King);
                let mut active = FeatureIndexList::new();
                PP_3Wide::append_active_indices(perspective, &pos, &mut active);
                assert_eq!(
                    sorted(&active),
                    reference_pairs(perspective, ksq, white | black, white, black),
                    "{fen}"
                );
            }
        }
    }

    #[test]
    fn test_make_index_is_symmetric() {
        // The pair is unordered: swapping the two pawns yields the same index.
        let ksq = Square::SQ_E1;
        let a = Square::SQ_D2;
        let b = Square::SQ_E3;
        let i1 = PP_3Wide::make_index(Color::White, Color::White, a, b, Color::Black, ksq);
        let i2 = PP_3Wide::make_index(Color::White, Color::Black, b, a, Color::White, ksq);
        assert_eq!(i1, i2);
        assert!(i1 >= PP_3Wide::INDEX_BASE);
        assert!(i1 < PP_3Wide::INDEX_BASE + PP_3Wide::DIMENSIONS);
    }
}
