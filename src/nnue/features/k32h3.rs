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

//! Definition of input features K32H3 of NNUE evaluation function
//!
//! Feature K32H3: Combination of the position of own king (32 buckets), a piece-square block
//! (3 states) and piece-square positions. Position mirrored such that king is always on e..h
//! files.

// A scalar implementation and an x86-64 AVX-512 implementation (`write_indices_avx512`) coexist
// in this file and the target picks one at compile time (`scripts/check-all.sh` compiles every
// target to keep both building).

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::nnue::nnue_common::{IndexType, ORIENT_TBL};
use crate::position::Position;
use crate::types::DirtyPiece;
use crate::types::{Bitboard, COLOR_NB, Color, Piece, PieceType, SQUARE_NB, Square, make_piece};

/// Maximum number of simultaneously active features
pub const MAX_ACTIVE_DIMENSIONS: usize = 32;

/// Stack-allocated list for feature indices - avoids heap allocation in hot path
#[derive(Clone)]
pub struct IndexList {
    // Invariant: `data[..len]` is initialised; the tail is left uninitialised so
    // creating a list costs no memset.
    data: [core::mem::MaybeUninit<u16>; MAX_ACTIVE_DIMENSIONS],
    len: usize,
}

impl IndexList {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            data: [core::mem::MaybeUninit::uninit(); MAX_ACTIVE_DIMENSIONS],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, val: IndexType) {
        debug_assert!(self.len < MAX_ACTIVE_DIMENSIONS);
        debug_assert!(val <= u16::MAX as IndexType);
        self.data[self.len] = core::mem::MaybeUninit::new(val as u16);
        self.len += 1;
    }

    /// Append without a capacity check (C++ `ValueList::push_back`).
    ///
    /// # Safety
    /// `len() < MAX_ACTIVE_DIMENSIONS` must hold.
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, val: IndexType) {
        debug_assert!(self.len < MAX_ACTIVE_DIMENSIONS);
        debug_assert!(val <= u16::MAX as IndexType);
        // SAFETY: the caller guarantees `len < MAX_ACTIVE_DIMENSIONS`, so the slot is inside `data`.
        unsafe {
            *self.data.get_unchecked_mut(self.len) = core::mem::MaybeUninit::new(val as u16);
        }
        self.len += 1;
    }

    /// Append all of `src` (panics if it does not fit).
    #[inline(always)]
    pub fn extend_from_slice(&mut self, src: &[u16]) {
        let start = self.len;
        let end = start + src.len();
        debug_assert!(end <= MAX_ACTIVE_DIMENSIONS);
        for (slot, &v) in self.data[start..end].iter_mut().zip(src) {
            *slot = core::mem::MaybeUninit::new(v);
        }
        self.len = end;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u16] {
        // SAFETY: the type invariant is that `data[..len]` is initialised (`push` and
        // `extend_from_slice` initialise every slot before bumping `len`, and their slice
        // indexing keeps `len <= MAX_ACTIVE_DIMENSIONS`); `MaybeUninit<u16>` has the layout of `u16`.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const u16, self.len) }
    }
}

impl Default for IndexList {
    fn default() -> Self {
        Self::new()
    }
}

impl core::ops::Deref for IndexList {
    type Target = [u16];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Feature K32H3 implementation
pub struct K32H3;

/// Block of each perspective
pub type BlockState = [u8; COLOR_NB];

impl K32H3 {
    /// Hash value embedded in the evaluation file
    pub const HASH_VALUE: u32 = 0xab005cefu32;

    /// Piece-square block of a perspective, in order of increasing heavy material:
    ///   0: no rook or queen on the board,
    ///   1: some rook or queen on the board, but the opponent has no queen,
    ///   2: the opponent has a queen.
    pub const NUM_BLOCKS: usize = 3;

    // Piece-square indices
    const PS_NONE: u16 = 0;
    const PS_W_PAWN: u16 = 0;
    const PS_B_PAWN: u16 = 1 * 64;
    const PS_W_KNIGHT: u16 = 2 * 64;
    const PS_B_KNIGHT: u16 = 3 * 64;
    const PS_W_BISHOP: u16 = 4 * 64;
    const PS_B_BISHOP: u16 = 5 * 64;
    const PS_W_ROOK: u16 = 6 * 64;
    const PS_B_ROOK: u16 = 7 * 64;
    const PS_W_QUEEN: u16 = 8 * 64;
    const PS_B_QUEEN: u16 = 9 * 64;
    const PS_KING: u16 = 10 * 64;
    const PS_NB: u16 = 11 * 64;

    /// Number of features per block (22,528 piece-squares)
    pub const BLOCK_DIMENSIONS: IndexType = (SQUARE_NB / 2) * Self::PS_NB as IndexType;

    /// First feature of `block` (blocks are contiguous in the network)
    #[inline(always)]
    pub const fn block_offset(block: usize) -> IndexType {
        block * Self::BLOCK_DIMENSIONS
    }

    /// Number of feature dimensions (67,584 piece-squares)
    pub const DIMENSIONS: IndexType = Self::NUM_BLOCKS * Self::BLOCK_DIMENSIONS;

    /// Piece-square index lookup table
    const PIECE_SQUARE_INDEX: [[u16; 16]; 2] = [
        // White perspective - Convention: W - us, B - them
        // Viewed from other side, W and B are reversed
        [
            Self::PS_NONE,
            Self::PS_W_PAWN,
            Self::PS_W_KNIGHT,
            Self::PS_W_BISHOP,
            Self::PS_W_ROOK,
            Self::PS_W_QUEEN,
            Self::PS_KING,
            Self::PS_NONE,
            Self::PS_NONE,
            Self::PS_B_PAWN,
            Self::PS_B_KNIGHT,
            Self::PS_B_BISHOP,
            Self::PS_B_ROOK,
            Self::PS_B_QUEEN,
            Self::PS_KING,
            Self::PS_NONE,
        ],
        // Black perspective - reversed
        [
            Self::PS_NONE,
            Self::PS_B_PAWN,
            Self::PS_B_KNIGHT,
            Self::PS_B_BISHOP,
            Self::PS_B_ROOK,
            Self::PS_B_QUEEN,
            Self::PS_KING,
            Self::PS_NONE,
            Self::PS_NONE,
            Self::PS_W_PAWN,
            Self::PS_W_KNIGHT,
            Self::PS_W_BISHOP,
            Self::PS_W_ROOK,
            Self::PS_W_QUEEN,
            Self::PS_KING,
            Self::PS_NONE,
        ],
    ];

    /// King buckets for feature indexing (C++ `B(v)` = `v * PS_NB`); black uses `ksq ^ 56`
    #[rustfmt::skip]
    const KING_BUCKETS: [u16; 64] = [
        28*Self::PS_NB, 29*Self::PS_NB, 30*Self::PS_NB, 31*Self::PS_NB, 31*Self::PS_NB, 30*Self::PS_NB, 29*Self::PS_NB, 28*Self::PS_NB,
        24*Self::PS_NB, 25*Self::PS_NB, 26*Self::PS_NB, 27*Self::PS_NB, 27*Self::PS_NB, 26*Self::PS_NB, 25*Self::PS_NB, 24*Self::PS_NB,
        20*Self::PS_NB, 21*Self::PS_NB, 22*Self::PS_NB, 23*Self::PS_NB, 23*Self::PS_NB, 22*Self::PS_NB, 21*Self::PS_NB, 20*Self::PS_NB,
        16*Self::PS_NB, 17*Self::PS_NB, 18*Self::PS_NB, 19*Self::PS_NB, 19*Self::PS_NB, 18*Self::PS_NB, 17*Self::PS_NB, 16*Self::PS_NB,
        12*Self::PS_NB, 13*Self::PS_NB, 14*Self::PS_NB, 15*Self::PS_NB, 15*Self::PS_NB, 14*Self::PS_NB, 13*Self::PS_NB, 12*Self::PS_NB,
         8*Self::PS_NB,  9*Self::PS_NB, 10*Self::PS_NB, 11*Self::PS_NB, 11*Self::PS_NB, 10*Self::PS_NB,  9*Self::PS_NB,  8*Self::PS_NB,
         4*Self::PS_NB,  5*Self::PS_NB,  6*Self::PS_NB,  7*Self::PS_NB,  7*Self::PS_NB,  6*Self::PS_NB,  5*Self::PS_NB,  4*Self::PS_NB,
         0*Self::PS_NB,  1*Self::PS_NB,  2*Self::PS_NB,  3*Self::PS_NB,  3*Self::PS_NB,  2*Self::PS_NB,  1*Self::PS_NB,  0*Self::PS_NB,
    ];

    /// Index of a feature within its block for a given king position and another piece on
    /// some square
    #[inline]
    pub fn make_index(perspective: Color, s: Square, pc: Piece, ksq: Square) -> IndexType {
        // Everything but the square is folded into one table, indexed by
        // [perspective * 64 + ksq][pc]. The folded offset is a multiple of 64 plus the
        // orientation (< 64), so XOR-ing the square into it equals the original
        // `(s ^ orient ^ flip) + piece_square + king_bucket`.
        static OFFSETS: [[u16; 16]; 2 * 64] = K32H3::build_offsets();
        s as IndexType ^ OFFSETS[perspective as usize * 64 + ksq as usize][pc as usize] as IndexType
    }

    /// Builds the `make_index` offset table (upstream 'Optimize HalfKAv2_hm::make_index()').
    const fn build_offsets() -> [[u16; 16]; 2 * 64] {
        let mut table = [[0u16; 16]; 2 * 64];
        let mut c = 0;
        while c < 2 {
            let mut sq = 0;
            while sq < 64 {
                let flip = (56 * c) as u16;
                let orient = ORIENT_TBL[sq] as u16 ^ flip;
                let king_bucket = Self::KING_BUCKETS[sq ^ flip as usize];
                let mut pi = 0;
                while pi < 16 {
                    table[c * 64 + sq][pi] = Self::PIECE_SQUARE_INDEX[c][pi] + king_bucket + orient;
                    pi += 1;
                }
                sq += 1;
            }
            c += 1;
        }
        table
    }

    /// Compute all changed K32H3 feature indices for a king-move update.
    ///
    /// Native AVX-512 builds use the same compressed-byte lookup as Stockfish's
    /// `USE_AVX512ICL` path.  The scalar path is retained for portable builds.
    ///
    /// Compute all changed feature indices and write them to the given lists
    #[inline(never)]
    pub fn write_indices(
        old_pieces: &[Piece; SQUARE_NB],
        new_pieces: &[Piece; SQUARE_NB],
        removed_bb: Bitboard,
        added_bb: Bitboard,
        perspective: Color,
        ksq: Square,
        removed: &mut IndexList,
        added: &mut IndexList,
    ) {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi2"
        ))]
        {
            // SAFETY: the function is compiled only when the required native
            // instructions are enabled.
            unsafe {
                Self::write_indices_avx512(
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
        }

        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi2"
        )))]
        {
            use crate::bitboard_fast::pop_lsb;
            let (mut removed_bb, mut added_bb) = (removed_bb, added_bb);
            while removed_bb != 0 {
                let sq = pop_lsb(&mut removed_bb);
                removed.push(Self::make_index(
                    perspective,
                    sq,
                    old_pieces[sq as usize],
                    ksq,
                ));
            }
            while added_bb != 0 {
                let sq = pop_lsb(&mut added_bb);
                added.push(Self::make_index(
                    perspective,
                    sq,
                    new_pieces[sq as usize],
                    ksq,
                ));
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    ))]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
    unsafe fn write_indices_avx512(
        old_pieces: &[Piece; SQUARE_NB],
        new_pieces: &[Piece; SQUARE_NB],
        removed_bb: Bitboard,
        added_bb: Bitboard,
        perspective: Color,
        ksq: Square,
        removed: &mut IndexList,
        added: &mut IndexList,
    ) {
        use core::arch::x86_64::*;

        let removed_count = removed_bb.count_ones() as usize;
        let added_count = added_bb.count_ones() as usize;

        // PieceSquareIndex and KingBuckets are multiples of 64, while s and orient use only the
        // low six bits. Therefore no carry crosses bit 6, and
        // (s ^ orient) + psi[pc] + kingBucket == s ^ (psi[pc] + kingBucket + orient),
        // allowing the orientation to be folded into the per-piece lookup offset.
        let flip = 56u16 * perspective as u16;
        let orient = ORIENT_TBL[ksq as usize] as u16 ^ flip;
        let king_bucket = Self::KING_BUCKETS[(ksq as u16 ^ flip) as usize];
        let psi = _mm256_loadu_si256(
            Self::PIECE_SQUARE_INDEX[perspective as usize].as_ptr() as *const __m256i
        );
        let psi_plus_offset = _mm512_add_epi16(
            _mm512_castsi256_si512(psi),
            _mm512_set1_epi16((king_bucket + orient) as i16),
        );

        let all_squares = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );
        let old_pieces_vec = _mm512_loadu_si512(old_pieces.as_ptr() as *const __m512i);
        let new_pieces_vec = _mm512_loadu_si512(new_pieces.as_ptr() as *const __m512i);

        let removed_squares = _mm512_maskz_compress_epi8(removed_bb, all_squares);
        let added_squares = _mm512_maskz_compress_epi8(added_bb, all_squares);
        let removed_pieces = _mm512_maskz_compress_epi8(removed_bb, old_pieces_vec);
        let added_pieces = _mm512_maskz_compress_epi8(added_bb, new_pieces_vec);

        let removed_squares = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(removed_squares));
        let added_squares = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(added_squares));
        let removed_pieces = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(removed_pieces));
        let added_pieces = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(added_pieces));

        let removed_indices = _mm512_xor_si512(
            removed_squares,
            _mm512_permutexvar_epi16(removed_pieces, psi_plus_offset),
        );
        let added_indices = _mm512_xor_si512(
            added_squares,
            _mm512_permutexvar_epi16(added_pieces, psi_plus_offset),
        );

        let mut removed_values = [0u16; MAX_ACTIVE_DIMENSIONS];
        let mut added_values = [0u16; MAX_ACTIVE_DIMENSIONS];
        _mm512_storeu_si512(removed_values.as_mut_ptr() as *mut __m512i, removed_indices);
        _mm512_storeu_si512(added_values.as_mut_ptr() as *mut __m512i, added_indices);

        removed.extend_from_slice(&removed_values[..removed_count]);
        added.extend_from_slice(&added_values[..added_count]);
    }

    /// Get a list of indices for recently changed features
    pub(crate) fn append_changed_indices(
        perspective: Color,
        ksq: Square,
        dp: &DirtyPiece,
        removed: &mut IndexList,
        added: &mut IndexList,
    ) {
        debug_assert!(
            removed.len() + 2 <= MAX_ACTIVE_DIMENSIONS && added.len() + 2 <= MAX_ACTIVE_DIMENSIONS
        );
        // SAFETY: in-crate callers pass freshly cleared lists (the debug assert above checks it),
        // so there is room for at most two pushes per list (one remove plus one add each).
        unsafe {
            if let Some(from) = dp.from {
                removed.push_unchecked(Self::make_index(perspective, from, dp.pc, ksq));
            }

            if let Some(to) = dp.to {
                added.push_unchecked(Self::make_index(perspective, to, dp.pc, ksq));
            }

            if let Some(remove_sq) = dp.remove_sq {
                removed.push_unchecked(Self::make_index(perspective, remove_sq, dp.remove_pc, ksq));
            }

            if let Some(add_sq) = dp.add_sq {
                added.push_unchecked(Self::make_index(perspective, add_sq, dp.add_pc, ksq));
            }
        }
    }

    /// Piece-square block of `perspective` in `pos` (see [`Self::NUM_BLOCKS`])
    #[inline(always)]
    pub fn block(pos: &Position, perspective: Color) -> usize {
        if pos.count_of(!perspective, PieceType::Queen) != 0 {
            2
        } else {
            (pos.pieces_pt2::<{ PieceType::Rook as u8 }, { PieceType::Queen as u8 }>() != 0)
                as usize
        }
    }

    /// Blocks of both perspectives in `pos`
    #[inline(always)]
    pub fn block_state(pos: &Position) -> BlockState {
        [
            Self::block(pos, Color::WHITE) as u8,
            Self::block(pos, Color::BLACK) as u8,
        ]
    }

    /// Returns whether the change stored in this DirtyPiece means that a full
    /// accumulator refresh is required
    #[inline(always)]
    pub fn requires_refresh(
        dirty_piece: &DirtyPiece,
        perspective: Color,
        block_before: usize,
        block_after: usize,
    ) -> bool {
        dirty_piece.pc == make_piece(perspective, PieceType::King) || block_before != block_after
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions() {
        // Verify dimensions calculation
        assert_eq!(K32H3::BLOCK_DIMENSIONS, 32 * 11 * 64);
        assert_eq!(K32H3::BLOCK_DIMENSIONS, 22528);
        assert_eq!(K32H3::DIMENSIONS, 67584);
        assert_eq!(K32H3::block_offset(2), 2 * 22528);
    }

    #[test]
    fn test_make_index() {
        // Every index lies within one block, and an e..h-file king square gives the same index
        // as its a..d mirror (with the board mirrored too).
        // Valid piece indices: 1-6 (W_PAWN..W_KING), 9-14 (B_PAWN..B_KING)
        // Invalid: 0 (NO_PIECE), 7-8 (gap), 15 (unused)
        for perspective in [Color::WHITE, Color::BLACK] {
            for ksq in 0..64u8 {
                for sq in 0..64u8 {
                    for pc in 0..16u8 {
                        if pc == 0 || pc == 7 || pc == 8 || pc == 15 {
                            continue;
                        }
                        let idx = K32H3::make_index(
                            perspective,
                            Square::from(sq),
                            Piece::from(pc),
                            Square::from(ksq),
                        );
                        assert!(
                            idx < K32H3::BLOCK_DIMENSIONS,
                            "Index {idx} out of bounds for {perspective:?}, ksq {ksq}, sq {sq}, pc {pc}"
                        );
                        let mirrored = K32H3::make_index(
                            perspective,
                            Square::from(sq ^ 7),
                            Piece::from(pc),
                            Square::from(ksq ^ 7),
                        );
                        assert_eq!(idx, mirrored);
                    }
                }
            }
        }
    }

    #[test]
    fn test_king_buckets() {
        // Verify all king buckets are in valid range [0, 32) * PS_NB
        for sq in 0..64 {
            let bucket = K32H3::KING_BUCKETS[sq];
            assert_eq!(bucket % K32H3::PS_NB, 0);
            let bucket_idx = bucket / K32H3::PS_NB;
            assert!(
                bucket_idx < 32,
                "King bucket {bucket_idx} out of range for square {sq}"
            );
        }

        // Verify horizontal mirror symmetry: sq and its file-flipped square share a bucket
        for rank in 0..8 {
            for file in 0..4 {
                let sq1 = rank * 8 + file;
                let sq2 = rank * 8 + (7 - file);
                assert_eq!(
                    K32H3::KING_BUCKETS[sq1],
                    K32H3::KING_BUCKETS[sq2],
                    "King bucket horizontal mirroring failed for rank {rank}"
                );
            }
        }
    }

    #[test]
    fn test_requires_refresh() {
        let mut dp = DirtyPiece::new();

        // Own king moves always require a refresh
        dp.pc = make_piece(Color::WHITE, PieceType::King);
        assert!(K32H3::requires_refresh(&dp, Color::WHITE, 1, 1));
        assert!(!K32H3::requires_refresh(&dp, Color::BLACK, 1, 1));

        dp.pc = make_piece(Color::BLACK, PieceType::King);
        assert!(!K32H3::requires_refresh(&dp, Color::WHITE, 2, 2));
        assert!(K32H3::requires_refresh(&dp, Color::BLACK, 2, 2));

        // Any other move requires one exactly when the block changes
        dp.pc = make_piece(Color::WHITE, PieceType::Knight);
        for before in 0..K32H3::NUM_BLOCKS {
            for after in 0..K32H3::NUM_BLOCKS {
                for perspective in [Color::WHITE, Color::BLACK] {
                    assert_eq!(
                        K32H3::requires_refresh(&dp, perspective, before, after),
                        before != after
                    );
                }
            }
        }
    }

    #[test]
    fn test_block() {
        crate::bitboard::init();
        let cases = [
            // fen, white block, black block
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                2,
                2,
            ),
            (
                "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                1,
                2,
            ),
            ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 1, 1),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 1, 2),
            ("4k3/8/8/2n5/8/8/8/2B1K3 w - - 0 1", 0, 0),
            ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", 2, 2),
        ];
        for (fen, white, black) in cases {
            let mut pos = Position::default();
            if pos.set_fen(fen, false).is_err() {
                panic!("bad fen {fen}");
            }
            assert_eq!(K32H3::block(&pos, Color::WHITE), white, "{fen}");
            assert_eq!(K32H3::block(&pos, Color::BLACK), black, "{fen}");
            assert_eq!(
                K32H3::block_state(&pos),
                [white as u8, black as u8],
                "{fen}"
            );
        }
    }
}
