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

//! Sparse input affine transform layer for NNUE
//!
//! This layer performs affine transformation with block sparse input,
//! optimizing for cases where many input values are zero.
//!
//! The fc_0 inner loop here is hand-shaped to work around ARM64 codegen rather
//! than written for readability: check the generated assembly before
//! "simplifying" anything in the hot loop.
//!
//! On AArch64 `Network::evaluate` does not scan for non-zero chunks here: the feature
//! transformer records a bitset while it writes its output and `propagate_with_bitset` walks
//! it, as in C++ (`NNZInfo::bitset`). `propagate` with its `find_nnz` pass is the
//! self-contained entry point used by `trace_evaluate` and the tests.

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::nnue::nnue_common::{AlignedVec, IndexType, MAX_SIMD_WIDTH};
use core::mem::MaybeUninit;
use std::io::{Read, Write};

/// Non-zero four-byte input chunks collected while the feature transformer
/// produces its AVX-512 output.  Stockfish passes this alongside the
/// transformed features so `fc_0` does not have to scan the buffer again.
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )),
    allow(dead_code) // only the AVX-512 transform records and reads NNZ indices
)]
pub(crate) struct NnzInfo<const CAPACITY: usize> {
    count: u32,
    indices: MaybeUninit<[u16; CAPACITY]>,
}

#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )),
    allow(dead_code) // only the AVX-512 transform records and reads NNZ indices
)]
impl<const CAPACITY: usize> NnzInfo<CAPACITY> {
    #[inline(always)]
    pub(crate) const fn new() -> Self {
        Self {
            count: 0,
            indices: MaybeUninit::uninit(),
        }
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.count = 0;
    }

    #[inline(always)]
    pub(crate) fn count(&self) -> usize {
        self.count as usize
    }

    /// # Safety
    ///
    /// `count <= CAPACITY` and `indices[0..count]` must have been initialized (by
    /// `record_avx512`); `indices()` exposes exactly that prefix.
    #[inline(always)]
    pub(crate) unsafe fn set_count(&mut self, count: u32) {
        debug_assert!(count as usize <= CAPACITY);
        self.count = count;
    }

    #[inline(always)]
    pub(crate) fn indices(&self) -> &[u16] {
        // SAFETY: `record_avx512` initialized `indices[0..count]` (it is the only writer) and
        // `count <= CAPACITY` (`record_avx512` debug-asserts this and `set_count`'s safety
        // contract requires it), so the slice is in bounds and initialized.
        unsafe {
            core::slice::from_raw_parts(self.indices.as_ptr() as *const u16, self.count as usize)
        }
    }

    /// Writes the compressed indices of the non-zero dword lanes of one AVX-512
    /// transformed-feature vector at `count` and returns the new count. `indices` contains the
    /// dword indices for its lanes.
    ///
    /// # Safety
    ///
    /// `count + 16 <= CAPACITY` (the store always writes 16 lanes), and the caller must pass the
    /// returned count to `set_count`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[inline(always)]
    pub(crate) unsafe fn record_avx512(
        &mut self,
        count: u32,
        values: core::arch::x86_64::__m512i,
        indices: core::arch::x86_64::__m512i,
    ) -> u32 {
        use core::arch::x86_64::*;

        let mask = _mm512_test_epi32_mask(values, values);
        let added = mask.count_ones() as usize;
        // No `added == 0` early-out (as in C++ `NNZCursor::record2`): it is a data-dependent
        // branch, and a zero-length compress/store is harmless.
        // The 16-lane store below bounds the write, not just the `added` valid lanes.
        debug_assert!(count as usize + 16 <= CAPACITY);
        let compressed = _mm512_maskz_compress_epi32(mask, indices);
        _mm512_mask_cvtepi32_storeu_epi16(
            self.indices.as_mut_ptr().cast::<i16>().add(count as usize),
            0xffff,
            compressed,
        );
        count + added as u32
    }
}

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// Lookup table for finding indices of nonzero bits. Used by the NEON `find_nnz` and by the SSSE3
// `find_nnz` (the x86 fallback taken when neither the AVX2 nor the AVX-512 variant applies).
#[cfg(any(
    target_arch = "aarch64",
    all(
        target_feature = "ssse3",
        not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )),
        not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))
    )
))]
mod offset_lookup {
    #[repr(align(64))]
    pub(super) struct OffsetIndices {
        pub(super) offset_indices: [[u16; 8]; 256],
    }

    impl OffsetIndices {
        const fn new() -> Self {
            let mut result = Self {
                offset_indices: [[0; 8]; 256],
            };

            let mut i = 0;
            while i < 256 {
                let mut j = i as u64;
                let mut k = 0;
                while j != 0 {
                    result.offset_indices[i][k] = j.trailing_zeros() as u16;
                    j &= j - 1;
                    k += 1;
                }
                while k < 8 {
                    result.offset_indices[i][k] = 0;
                    k += 1;
                }
                i += 1;
            }
            result
        }
    }

    pub(super) static LOOKUP: OffsetIndices = OffsetIndices::new();
}

/// Views an initialised `u16` buffer as a possibly-uninitialised one (for `find_nnz`, which
/// only ever stores initialised values).
#[inline(always)]
fn as_uninit_u16(buf: &mut [u16]) -> &mut [MaybeUninit<u16>] {
    // SAFETY: `MaybeUninit<u16>` has the same layout as `u16`, and `find_nnz` only writes
    // initialised values, so the original `u16`s stay initialised.
    unsafe { &mut *(buf as *mut [u16] as *mut [MaybeUninit<u16>]) }
}

/// The initialised prefix `buf[..count]` of a buffer filled by `find_nnz`.
///
/// # Safety
///
/// `count <= buf.len()` and `buf[..count]` must have been written.
#[inline(always)]
#[cfg(any(target_feature = "ssse3", target_arch = "aarch64"))]
unsafe fn init_prefix(buf: &[MaybeUninit<u16>], count: usize) -> &[u16] {
    debug_assert!(count <= buf.len());
    // SAFETY: by the contract above, the first `count` elements are in bounds and initialised.
    unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<u16>(), count) }
}

/// Find indices of nonzero elements in the input
/// NEON version processes 8 i32s per iteration as two 4-lane vtstq_u32 tests
/// Uses SIMD for output storage matching C++ implementation
///
/// Contract (all `find_nnz` variants; the function is module-private and every caller
/// satisfies it, so it is only debug-asserted): `input.len() >= input_dimensions` and
/// `out.len() >= input_dimensions`. Returns the number of non-zero `input[..input_dimensions]`
/// and initialises `out[..count]` with their ascending indices; `out[count..input_dimensions]`
/// may be clobbered. `input_dimensions` need not be a multiple of the vector width (a scalar
/// tail handles the remainder, which is empty for the production sizes).
#[cfg(target_arch = "aarch64")]
fn find_nnz(input: &[i32], out: &mut [MaybeUninit<u16>], input_dimensions: usize) -> IndexType {
    use core::arch::aarch64::*;
    use offset_lookup::LOOKUP;

    debug_assert!(input.len() >= input_dimensions && out.len() >= input_dimensions);

    // Mask for converting lane results to bit positions: [1, 2, 4, 8]
    static MASK: [u32; 4] = [1, 2, 4, 8];

    let mut count = 0usize;
    const CHUNK_SIZE: usize = 8; // Process 8 i32s per iteration
    const SIMD_WIDTH: usize = 4; // NEON processes 4 i32s at a time
    let num_chunks = input_dimensions / CHUNK_SIZE;

    let input_ptr = input.as_ptr() as *const u32;
    let out_ptr = out.as_mut_ptr().cast::<u16>();

    // SAFETY: NEON is baseline on aarch64. By the contract above `input` holds at least
    // `input_dimensions` i32 and `out` at least `input_dimensions` u16; the loads read `chunk * 8
    // + 8 <= num_chunks * 8 <= input_dimensions` and the 8-lane store at `count <= chunk * 8`
    // ends at `count + 8 <= input_dimensions`, so it stays within `out` (unaligned NEON
    // accesses; `LOOKUP` rows are `[u16; 8]`).
    unsafe {
        // Load mask once outside loop
        let mask_vec = vld1q_u32(MASK.as_ptr());
        // Increment for base: adds 8 to each lane
        let increment = vdupq_n_u16(CHUNK_SIZE as u16);
        // Base vector: [0, 0, 0, 0, 0, 0, 0, 0]
        let mut base_vec = vdupq_n_u16(0);

        for chunk in 0..num_chunks {
            // Process 8 values using 2 SIMD operations
            // First 4 values
            let v0 = vld1q_u32(input_ptr.add(chunk * 8));
            let test0 = vtstq_u32(v0, v0);
            let bits0 = vaddvq_u32(vandq_u32(test0, mask_vec));

            // Next 4 values (shifted by 4 bit positions)
            let v1 = vld1q_u32(input_ptr.add(chunk * 8 + 4));
            let test1 = vtstq_u32(v1, v1);
            let bits1 = vaddvq_u32(vandq_u32(test1, mask_vec));

            let nnz = bits0 | (bits1 << SIMD_WIDTH);

            // Always store - when nnz==0, count doesn't increment so values get overwritten
            // Removing branch improves performance on hot loop
            let offsets_ptr = LOOKUP.offset_indices[nnz as usize].as_ptr();
            let offsets = vld1q_u16(offsets_ptr);
            let indices = vaddq_u16(base_vec, offsets);
            vst1q_u16(out_ptr.add(count), indices);
            count += (nnz.count_ones()) as usize;

            base_vec = vaddq_u16(base_vec, increment);
        }
    }

    // Scalar tail for `input_dimensions % 8` (no iterations for the production sizes). Here
    // `count <= i < input_dimensions`, so the write is in bounds.
    for i in num_chunks * CHUNK_SIZE..input_dimensions {
        if input[i] != 0 {
            out[count].write(i as u16);
            count += 1;
        }
    }

    count as IndexType
}

/// Find nonzero chunks with AVX-512's mask/compress instructions.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
fn find_nnz(input: &[i32], out: &mut [MaybeUninit<u16>], input_dimensions: usize) -> IndexType {
    use core::arch::x86_64::*;

    debug_assert!(input.len() >= input_dimensions && out.len() >= input_dimensions);
    let mut count = 0usize;
    let mut base = 0usize;
    // SAFETY: AVX-512F is enabled by this fn's `cfg(target_feature = "avx512f")` (register-only
    // intrinsic).
    let zero = unsafe { _mm512_setzero_si512() };

    while base + 16 <= input_dimensions {
        // SAFETY: AVX-512F is enabled by this fn's cfg; `base + 16 <= input_dimensions <=
        // input.len()` (contract on `find_nnz`), so the unaligned 64-byte load is in bounds.
        let values = unsafe { _mm512_loadu_si512(input.as_ptr().add(base) as *const __m512i) };
        // SAFETY: AVX-512F is enabled by this fn's cfg (register-only intrinsic).
        let mask = unsafe { _mm512_cmpneq_epi32_mask(values, zero) };
        if mask != 0 {
            let offsets =
                // SAFETY: AVX-512F is enabled by this fn's cfg (register-only intrinsic).
                unsafe { _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) };
            // SAFETY: AVX-512F is enabled by this fn's cfg (register-only intrinsic).
            let offsets = unsafe { _mm512_add_epi32(offsets, _mm512_set1_epi32(base as i32)) };
            // SAFETY: AVX-512F is enabled by this fn's cfg (register-only intrinsic).
            let indices = unsafe { _mm512_maskz_compress_epi32(mask, offsets) };
            // SAFETY: AVX-512BW/F are enabled by this fn's cfg. The 16-lane store writes
            // `out[count..count + 16]` with `count <= base` and `base + 16 <= input_dimensions <=
            // out.len()` (callers size `out` to `input_dimensions`).
            unsafe {
                _mm512_mask_cvtepi32_storeu_epi16(
                    out.as_mut_ptr().add(count) as *mut i16,
                    0xFFFF,
                    indices,
                );
            }
            count += mask.count_ones() as usize;
        }
        base += 16;
    }

    while base < input_dimensions {
        if input[base] != 0 {
            out[count].write(base as u16);
            count += 1;
        }
        base += 1;
    }

    count as IndexType
}

/// Find nonzero chunks with AVX2's eight-lane comparison and bit mask.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
fn find_nnz(input: &[i32], out: &mut [MaybeUninit<u16>], input_dimensions: usize) -> IndexType {
    use core::arch::x86_64::*;

    debug_assert!(input.len() >= input_dimensions && out.len() >= input_dimensions);
    let mut count = 0usize;
    let mut base = 0usize;
    // SAFETY: AVX2 is enabled by this fn's `cfg(target_feature = "avx2")` (register-only
    // intrinsic).
    let zero = unsafe { _mm256_setzero_si256() };

    while base + 8 <= input_dimensions {
        // SAFETY: AVX2 is enabled by this fn's cfg; `base + 8 <= input_dimensions <=
        // input.len()` (contract on `find_nnz`), so the unaligned 32-byte load is in bounds.
        let values = unsafe { _mm256_loadu_si256(input.as_ptr().add(base) as *const __m256i) };
        // SAFETY: AVX2 is enabled by this fn's cfg (register-only intrinsic).
        let equal = unsafe { _mm256_cmpeq_epi32(values, zero) };
        // SAFETY: AVX2 is enabled by this fn's cfg (register-only intrinsic).
        let equal_mask = unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(equal)) as u32 };
        let mut nonzero_mask = (!equal_mask) & 0xFF;
        while nonzero_mask != 0 {
            let bit = nonzero_mask.trailing_zeros() as usize;
            out[count].write((base + bit) as u16);
            count += 1;
            nonzero_mask &= nonzero_mask - 1;
        }
        base += 8;
    }

    while base < input_dimensions {
        if input[base] != 0 {
            out[count].write(base as u16);
            count += 1;
        }
        base += 1;
    }

    count as IndexType
}

/// Find indices of nonzero elements - scalar/SSSE3 fallback for non-NEON x86. Same contract as
/// the NEON version above.
#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    )
)))]
fn find_nnz(input: &[i32], out: &mut [MaybeUninit<u16>], input_dimensions: usize) -> IndexType {
    debug_assert!(input.len() >= input_dimensions && out.len() >= input_dimensions);
    let mut count = 0;

    #[cfg(target_feature = "ssse3")]
    {
        use crate::bitboard_fast::popcount;
        use offset_lookup::LOOKUP;
        const CHUNK_SIZE: usize = 8;
        let num_chunks = input_dimensions / CHUNK_SIZE;

        let mut base = 0u16;
        for chunk in 0..num_chunks {
            // Process 8 values at a time
            let mut nnz = 0u8;

            for j in 0..CHUNK_SIZE {
                let idx = chunk * CHUNK_SIZE + j;
                if idx < input.len() && input[idx] != 0 {
                    nnz |= 1 << j;
                }
            }

            if nnz != 0 {
                // Use lookup table to get indices
                let indices = &LOOKUP.offset_indices[nnz as usize];
                for k in 0..popcount(nnz as u64) {
                    out[count].write(base + indices[k as usize]);
                    count += 1;
                }
            }

            base += CHUNK_SIZE as u16;
        }

        // Scalar tail for `input_dimensions % 8` (no iterations for the production sizes). Here
        // `count <= i < input_dimensions <= out.len()`.
        for i in num_chunks * CHUNK_SIZE..input_dimensions {
            if input[i] != 0 {
                out[count].write(i as u16);
                count += 1;
            }
        }
    }

    #[cfg(not(target_feature = "ssse3"))]
    {
        for (i, &val) in input.iter().enumerate().take(input_dimensions) {
            if val != 0 {
                out[count].write(i as u16);
                count += 1;
            }
        }
    }

    count as IndexType
}

/// Sparse input affine transform layer
#[derive(Clone)]
pub struct AffineTransformSparseInput<const IN_DIMS: usize, const OUT_DIMS: usize> {
    // Layer-specific type overrides to match C++ implementation
    // C++ uses: BiasType = OutputType (i32), WeightType = i8
    // 64-byte aligned for SIMD performance
    weights: AlignedVec<i8>,
    biases: AlignedVec<i32>,
}

impl<const IN_DIMS: usize, const OUT_DIMS: usize> AffineTransformSparseInput<IN_DIMS, OUT_DIMS> {
    // Dimensions
    pub const INPUT_DIMENSIONS: IndexType = IN_DIMS as IndexType;
    pub const OUTPUT_DIMENSIONS: IndexType = OUT_DIMS as IndexType;

    const PADDED_INPUT_DIMENSIONS: usize = IN_DIMS.div_ceil(MAX_SIMD_WIDTH) * MAX_SIMD_WIDTH;

    #[cfg(any(target_feature = "ssse3", target_feature = "neon"))]
    const CHUNK_SIZE: usize = 4;
    #[cfg(not(any(target_feature = "ssse3", target_feature = "neon")))]
    const CHUNK_SIZE: usize = 1;

    /// Number of 32-bit chunks for sparse propagation
    const NUM_CHUNKS: usize = IN_DIMS.div_ceil(8) * 2;

    /// Create a new sparse input affine transform layer
    pub fn new() -> Self {
        Self {
            weights: AlignedVec::new_zeroed(OUT_DIMS * Self::PADDED_INPUT_DIMENSIONS),
            biases: AlignedVec::new_zeroed(OUT_DIMS),
        }
    }

    /// Get hash value for this layer
    ///
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value(prev_hash: u32) -> u32 {
        let mut hash_value = 0xCC03DAE4u32;
        hash_value = hash_value.wrapping_add(OUT_DIMS as u32);
        hash_value ^= prev_hash >> 1;
        hash_value ^= prev_hash << 31;
        hash_value
    }

    /// Get weight index with scrambling for better memory access
    // `CHUNK_SIZE` is 1 in scalar builds, which makes `i % CHUNK_SIZE` trivially zero there.
    #[allow(clippy::modulo_one)]
    const fn get_weight_index_scrambled(i: usize) -> usize {
        (i / Self::CHUNK_SIZE) % (Self::PADDED_INPUT_DIMENSIONS / Self::CHUNK_SIZE)
            * OUT_DIMS
            * Self::CHUNK_SIZE
            + i / Self::PADDED_INPUT_DIMENSIONS * Self::CHUNK_SIZE
            + i % Self::CHUNK_SIZE
    }

    /// Get weight index
    const fn get_weight_index(i: usize) -> usize {
        if cfg!(any(target_feature = "ssse3", target_feature = "neon")) {
            Self::get_weight_index_scrambled(i)
        } else {
            i
        }
    }

    /// Read parameters from stream
    pub fn read_parameters<R: Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // Read biases (BiasType = i32, 4 bytes each - matches C++ OutputType)
        for bias in self.biases.iter_mut() {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *bias = i32::from_le_bytes(buf);
        }

        // Read weights (WeightType = i8, 1 byte each - matches C++ implementation)
        for i in 0..(OUT_DIMS * Self::PADDED_INPUT_DIMENSIONS) {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            let weight_val = i8::from_le_bytes(buf);
            self.weights[Self::get_weight_index(i)] = weight_val;
        }

        Ok(())
    }

    /// Write parameters to stream
    pub fn write_parameters<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write biases
        for &bias in &self.biases {
            writer.write_all(&bias.to_le_bytes())?;
        }

        // Write weights with scrambling
        for i in 0..(OUT_DIMS * Self::PADDED_INPUT_DIMENSIONS) {
            let weight = self.weights[Self::get_weight_index(i)];
            writer.write_all(&weight.to_le_bytes())?;
        }

        Ok(())
    }

    /// Forward propagation with sparse input optimization
    /// Takes TransformedFeatureType (u8) as input and outputs i32
    /// The input is the clipped ReLU output from the feature transformer (values 0-127)
    pub fn propagate(&self, input: &[u8], output: &mut [i32]) {
        // The kernels view `input` as `NUM_CHUNKS` i32 (so it must be 4-byte aligned) and write
        // `OUT_DIMS` outputs, all unchecked.
        assert!(input.len() >= Self::NUM_CHUNKS * 4 && output.len() >= OUT_DIMS);
        assert!(
            (input.as_ptr() as usize).is_multiple_of(4),
            "input must be 4-byte aligned"
        );

        // Initialize output with biases (biases are already i32)
        for i in 0..OUT_DIMS {
            output[i] = self.biases[i];
        }

        #[cfg(any(target_feature = "ssse3", target_feature = "neon"))]
        {
            // The NEON kernels store whole 4-lane registers, so they cover only `OUT_DIMS % 4 == 0`
            // (C++ requires `% 16 == 0`, which the production layers satisfy). Other shapes take the
            // layout-agnostic dense reference; `OUT_DIMS` is a const, so for production shapes the
            // branch is resolved at compile time and the kernel call is unchanged.
            #[cfg(target_arch = "aarch64")]
            if !OUT_DIMS.is_multiple_of(4) {
                self.propagate_dense_i32(input, output);
                return;
            }
            // SAFETY: input is 4-byte aligned with >= NUM_CHUNKS*4 bytes and output holds >=
            // OUT_DIMS i32 (both asserted above). The AArch64 SDOT kernel exists only under
            // cfg(stockcrab_dotprod): either the target has `dotprod` or the user asserted
            // dot-product hardware with the `neon-dotprod` Cargo feature.
            unsafe { self.propagate_sparse_i32(input, output) };
        }
        #[cfg(not(any(target_feature = "ssse3", target_feature = "neon")))]
        {
            self.propagate_dense_i32(input, output);
        }
    }

    /// AVX-512 entry point used when the feature transformer already collected
    /// the non-zero four-byte input chunks.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    pub(crate) fn propagate_with_nnz(&self, input: &[u8], output: &mut [i32], nnz: &NnzInfo<256>) {
        assert!(input.len() >= Self::NUM_CHUNKS * 4 && output.len() >= OUT_DIMS);
        assert!((input.as_ptr() as usize).is_multiple_of(4));
        // Every recorded index must address one of the input's i32 chunks.
        debug_assert!(
            nnz.indices()
                .iter()
                .all(|&i| (i as usize) < Self::NUM_CHUNKS)
        );
        for i in 0..OUT_DIMS {
            output[i] = self.biases[i];
        }

        // SAFETY: AVX-512F/BW are enabled by the enclosing `cfg(target_feature)`. `input` is a
        // 4-byte aligned buffer of at least `NUM_CHUNKS * 4` bytes and `output` holds `OUT_DIMS`
        // i32 (both asserted above). `nnz` must come from the feature transformer's compression
        // of that same buffer (indices below `NUM_CHUNKS`; only debug-asserted, the sole producer
        // is `record_avx512` over that buffer).
        unsafe { self.propagate_sparse_i32_with_nnz(input, output, Some(nnz)) };
    }

    /// Sparse propagation for SIMD architectures (i32 output) - NON-DOTPROD version
    /// Uses NEON SIMD on aarch64 for vectorized dot products using vmull/vpadal sequence
    #[cfg(all(target_arch = "aarch64", not(stockcrab_dotprod)))]
    #[inline(always)]
    unsafe fn propagate_sparse_i32(&self, input: &[u8], output: &mut [i32]) {
        // Max chunks for Big network (IN_DIMS=3072): ((3072+7)/8)*2 = 768
        // Stack allocation instead of heap Vec (matches C++)
        const MAX_CHUNKS: usize = 768;
        const OUTPUT_SIMD_WIDTH: usize = 4;
        let num_chunks = Self::NUM_CHUNKS;
        let num_regs = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        debug_assert!(num_chunks <= MAX_CHUNKS);

        // `find_nnz` writes nnz[0..count]; only that prefix is ever viewed as initialised.
        let mut nnz_storage = [MaybeUninit::<u16>::uninit(); MAX_CHUNKS];
        let nnz_buf = &mut nnz_storage[..num_chunks];

        debug_assert_eq!(
            (input.as_ptr() as usize) & 3,
            0,
            "NNUE sparse affine input must be 4-byte aligned"
        );
        debug_assert!(input.len() >= num_chunks * 4);
        let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
        let count = find_nnz(input32, nnz_buf, num_chunks);
        // SAFETY: `find_nnz` initialised `nnz_buf[..count]` and `count <= num_chunks`.
        let nnz = init_prefix(nnz_buf, count as usize);

        // The accumulator file below holds 16 registers.
        const {
            assert!(
                OUT_DIMS / OUTPUT_SIMD_WIDTH <= 16,
                "OUT_DIMS too large for the NEON accumulator file"
            )
        };

        // Registers beyond `num_regs` are never read, so the compiler drops their initialisation.
        let mut acc = [vdupq_n_s32(0); 16];

        let bias_ptr = self.biases.as_ptr();
        for k in 0..num_regs {
            *acc.get_unchecked_mut(k) = vld1q_s32(bias_ptr.add(k * OUTPUT_SIMD_WIDTH));
        }

        // Hoist weight pointer extraction outside the loop (C++ does this)
        let weights_ptr = self.weights.as_ptr();

        for j in 0..count as usize {
            let i = *nnz.get_unchecked(j) as usize;
            let in_val = vreinterpretq_s8_u32(vdupq_n_u32(*input32.get_unchecked(i) as u32));
            let col = weights_ptr.add(i * OUT_DIMS * Self::CHUNK_SIZE) as *const int8x16_t;

            for k in 0..num_regs {
                let weights = vld1q_s8(col.add(k) as *const i8);
                // vmull + vpadal sequence (4 instructions)
                let product0 = vmull_s8(vget_low_s8(in_val), vget_low_s8(weights));
                let product1 = vmull_high_s8(in_val, weights);
                let sum = vpaddq_s16(product0, product1);
                *acc.get_unchecked_mut(k) = vpadalq_s16(*acc.get_unchecked(k), sum);
            }
        }

        let out_ptr = output.as_mut_ptr();
        for k in 0..num_regs {
            vst1q_s32(out_ptr.add(k * OUTPUT_SIMD_WIDTH), *acc.get_unchecked(k));
        }
    }

    /// Sparse propagation for SIMD architectures (i32 output) - DOTPROD version
    /// Uses single SDOT instruction for ~2x faster dot products (requires ARMv8.2+)
    ///
    /// Mirrors the C++ `USE_NEON_DOTPROD` branch of
    /// `AffineTransformSparseInput::propagate` (affine_transform_sparse_input.h):
    ///   OutputSimdWidth = sizeof(int32x4_t) / sizeof(i32) = 4
    ///   NumAccums       = OutputDimensions / OutputSimdWidth
    ///   NumRegs         = 3 * NumAccums  (three independent dependency chains that are
    ///                                     folded together with vaddq_s32 before the store)
    ///
    /// NOTE: the accumulator count must be derived from `OUT_DIMS`; a fixed count would leave
    /// outputs beyond it unwritten.
    #[cfg(all(target_arch = "aarch64", stockcrab_dotprod))]
    #[target_feature(enable = "dotprod")]
    unsafe fn propagate_sparse_i32(&self, input: &[u8], output: &mut [i32]) {
        // Max chunks for the largest supported input (IN_DIMS=3072): ((3072+7)/8)*2 = 768
        const MAX_CHUNKS: usize = 768;
        const OUTPUT_SIMD_WIDTH: usize = 4;
        // Upper bound on int32x4_t accumulators (supports OUT_DIMS up to 64).
        const MAX_ACCUMS: usize = 16;

        // Both derived from const generics, so these are compile-time constants after
        // monomorphization and every loop below is fully unrolled.
        let num_chunks = Self::NUM_CHUNKS;
        let num_accums = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        debug_assert!(num_chunks <= MAX_CHUNKS);
        // `acc` below holds `3 * MAX_ACCUMS` registers, `3 * num_accums` of them in use.
        const {
            assert!(
                OUT_DIMS / OUTPUT_SIMD_WIDTH <= MAX_ACCUMS,
                "OUT_DIMS too large for the NEON dotprod accumulator file"
            )
        };

        // `find_nnz` writes nnz[0..count]; only that prefix is ever viewed as initialised.
        let mut nnz_storage = [MaybeUninit::<u16>::uninit(); MAX_CHUNKS];
        let nnz_buf = &mut nnz_storage[..num_chunks];

        debug_assert_eq!(
            (input.as_ptr() as usize) & 3,
            0,
            "NNUE sparse affine input must be 4-byte aligned"
        );
        debug_assert!(input.len() >= num_chunks * 4);
        let input32 = input.as_ptr() as *const i32;
        let count = find_nnz(
            core::slice::from_raw_parts(input32, num_chunks),
            nnz_buf,
            num_chunks,
        ) as usize;
        // SAFETY: `find_nnz` initialised `nnz_buf[..count]` and `count <= num_chunks`.
        let nnz = init_prefix(nnz_buf, count);

        let bias_ptr = self.biases.as_ptr();
        let weights_base = self.weights.as_ptr();

        // acc[k], acc[k + num_accums] and acc[k + 2 * num_accums] are three independent
        // dependency chains for output group k (C++ NumRegs = 3 * NumAccums).
        let mut acc: [int32x4_t; 3 * MAX_ACCUMS] = [vdupq_n_s32(0); 3 * MAX_ACCUMS];
        for k in 0..num_accums {
            *acc.get_unchecked_mut(k) = vld1q_s32(bias_ptr.add(k * OUTPUT_SIMD_WIDTH));
        }

        // Main loop: 3 non-zero chunks per iteration for instruction-level parallelism.
        // Indices (not pointers) are used so that `count < 3` cannot underflow.
        let mut j = 0usize;
        while j + 3 <= count {
            let i0 = *nnz.get_unchecked(j) as usize;
            let i1 = *nnz.get_unchecked(j + 1) as usize;
            let i2 = *nnz.get_unchecked(j + 2) as usize;
            j += 3;

            // Broadcast the three 4-byte input chunks
            let in_val0 = vreinterpretq_s8_u32(vdupq_n_u32(*input32.add(i0) as u32));
            let in_val1 = vreinterpretq_s8_u32(vdupq_n_u32(*input32.add(i1) as u32));
            let in_val2 = vreinterpretq_s8_u32(vdupq_n_u32(*input32.add(i2) as u32));

            let col0 = weights_base.add(i0 * OUT_DIMS * Self::CHUNK_SIZE) as *const int8x16_t;
            let col1 = weights_base.add(i1 * OUT_DIMS * Self::CHUNK_SIZE) as *const int8x16_t;
            let col2 = weights_base.add(i2 * OUT_DIMS * Self::CHUNK_SIZE) as *const int8x16_t;

            for k in 0..num_accums {
                let w0 = vld1q_s8(col0.add(k) as *const i8);
                let w1 = vld1q_s8(col1.add(k) as *const i8);
                let w2 = vld1q_s8(col2.add(k) as *const i8);
                *acc.get_unchecked_mut(k) = vdotq_s32(*acc.get_unchecked(k), in_val0, w0);
                *acc.get_unchecked_mut(k + num_accums) =
                    vdotq_s32(*acc.get_unchecked(k + num_accums), in_val1, w1);
                *acc.get_unchecked_mut(k + 2 * num_accums) =
                    vdotq_s32(*acc.get_unchecked(k + 2 * num_accums), in_val2, w2);
            }
        }

        // Remainder loop: 0-2 leftover chunks, one at a time
        while j < count {
            let i = *nnz.get_unchecked(j) as usize;
            j += 1;

            let in_val = vreinterpretq_s8_u32(vdupq_n_u32(*input32.add(i) as u32));
            let col = weights_base.add(i * OUT_DIMS * Self::CHUNK_SIZE) as *const int8x16_t;

            for k in 0..num_accums {
                let w = vld1q_s8(col.add(k) as *const i8);
                *acc.get_unchecked_mut(k) = vdotq_s32(*acc.get_unchecked(k), in_val, w);
            }
        }

        // Fold the three dependency chains (C++ affine_transform_sparse_input.h)
        for k in 0..num_accums {
            *acc.get_unchecked_mut(k) = vaddq_s32(
                vaddq_s32(*acc.get_unchecked(k), *acc.get_unchecked(k + num_accums)),
                *acc.get_unchecked(k + 2 * num_accums),
            );
        }

        // Store results
        let out_ptr = output.as_mut_ptr();
        for k in 0..num_accums {
            vst1q_s32(out_ptr.add(k * OUTPUT_SIMD_WIDTH), *acc.get_unchecked(k));
        }
    }

    /// AArch64 entry point used when the feature transformer already recorded the
    /// non-zero-chunk bitset (C++ `NNZInfo::bitset`, nnz_helper.h): bit `i` of the
    /// little-endian bitset is set iff the four input bytes `4 * i .. 4 * i + 4` are not all
    /// zero. A set bit over an all-zero chunk is harmless (it adds zero); a clear bit over a
    /// non-zero chunk would drop that chunk's contribution.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) fn propagate_with_bitset(&self, input: &[u8], output: &mut [i32], bitset: &[u8]) {
        // C++: static_assert(InputDimensions % 256 == 0); the kernels store whole 4-lane
        // registers into a 16-register accumulator file.
        const {
            assert!(IN_DIMS.is_multiple_of(256));
            assert!(OUT_DIMS.is_multiple_of(4) && OUT_DIMS / 4 <= 16);
        };
        assert!(input.len() >= Self::NUM_CHUNKS * 4 && output.len() >= OUT_DIMS);
        assert!(
            (input.as_ptr() as usize).is_multiple_of(4),
            "input must be 4-byte aligned"
        );
        assert!(bitset.len() >= IN_DIMS / 32);

        // SAFETY: input is 4-byte aligned with >= NUM_CHUNKS*4 == IN_DIMS bytes, output holds >=
        // OUT_DIMS i32 and the bitset >= IN_DIMS / 32 bytes (all asserted above), so every bit
        // index is below NUM_CHUNKS. The SDOT kernel exists only under cfg(stockcrab_dotprod):
        // either the target has `dotprod` or the user asserted dot-product hardware with the
        // `neon-dotprod` Cargo feature.
        unsafe { self.propagate_sparse_bitset(input, output, bitset) };
    }

    /// Bitset-driven sparse propagation - NON-DOTPROD version (the generic `#else` branch of
    /// the C++ bitset walk, affine_transform_sparse_input.h: one chunk per iteration, one
    /// dependency chain).
    #[cfg(all(target_arch = "aarch64", not(stockcrab_dotprod)))]
    #[inline(always)]
    unsafe fn propagate_sparse_bitset(&self, input: &[u8], output: &mut [i32], bitset: &[u8]) {
        const OUTPUT_SIMD_WIDTH: usize = 4;
        let num_regs = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        let input32 = input.as_ptr() as *const i32;
        let bias_ptr = self.biases.as_ptr();
        let weights_cp = self.weights.as_ptr();

        // Registers beyond `num_regs` are never read, so the compiler drops their initialisation.
        let mut acc = [vdupq_n_s32(0); 16];
        for k in 0..num_regs {
            *acc.get_unchecked_mut(k) = vld1q_s32(bias_ptr.add(k * OUTPUT_SIMD_WIDTH));
        }

        for k in 0..IN_DIMS / 256 {
            let mut bits = (bitset.as_ptr().add(k * 8) as *const u64).read_unaligned();
            let base = k * 64;

            let base_addr = input32.add(base);
            let weights_base = weights_cp.add(base * OUT_DIMS * Self::CHUNK_SIZE);

            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;

                let in_val = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i) as u32));
                let col = weights_base.add(i * OUT_DIMS * Self::CHUNK_SIZE);

                for l in 0..num_regs {
                    let weights = vld1q_s8(col.add(l * 16));
                    // vmull + vpadal sequence (4 instructions)
                    let product0 = vmull_s8(vget_low_s8(in_val), vget_low_s8(weights));
                    let product1 = vmull_high_s8(in_val, weights);
                    let sum = vpaddq_s16(product0, product1);
                    *acc.get_unchecked_mut(l) = vpadalq_s16(*acc.get_unchecked(l), sum);
                }
            }
        }

        let out_ptr = output.as_mut_ptr();
        for k in 0..num_regs {
            vst1q_s32(out_ptr.add(k * OUTPUT_SIMD_WIDTH), *acc.get_unchecked(k));
        }
    }

    /// Bitset-driven sparse propagation - DOTPROD version.
    ///
    /// Mirrors the C++ `USE_NEON_DOTPROD` bitset walk of
    /// `AffineTransformSparseInput::propagate` (affine_transform_sparse_input.h): for each
    /// 64-chunk word of the bitset, pop up to three set bits per iteration and feed three
    /// independent SDOT dependency chains (`NumRegs = 3 * NumAccums`), with one- and
    /// two-column bodies for the tail of a word; the chains are folded before the store.
    #[cfg(all(target_arch = "aarch64", stockcrab_dotprod))]
    #[target_feature(enable = "dotprod")]
    unsafe fn propagate_sparse_bitset(&self, input: &[u8], output: &mut [i32], bitset: &[u8]) {
        const OUTPUT_SIMD_WIDTH: usize = 4;
        // Upper bound on int32x4_t accumulators (supports OUT_DIMS up to 64).
        const MAX_ACCUMS: usize = 16;
        let num_accums = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        let input32 = input.as_ptr() as *const i32;
        let bias_ptr = self.biases.as_ptr();
        // convince the compiler to not do weird pointer arithmetic in the following loops
        let weights_cp = self.weights.as_ptr();

        // acc[l], acc[l + num_accums] and acc[l + 2 * num_accums] are three independent
        // dependency chains for output group l.
        let mut acc: [int32x4_t; 3 * MAX_ACCUMS] = [vdupq_n_s32(0); 3 * MAX_ACCUMS];
        for k in 0..num_accums {
            *acc.get_unchecked_mut(k) = vld1q_s32(bias_ptr.add(k * OUTPUT_SIMD_WIDTH));
        }

        for k in 0..IN_DIMS / 256 {
            let mut bits = (bitset.as_ptr().add(k * 8) as *const u64).read_unaligned();
            let base = k * 64;

            let base_addr = input32.add(base);
            let weights_base = weights_cp.add(base * OUT_DIMS * Self::CHUNK_SIZE);

            while bits != 0 {
                let i0 = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if bits == 0 {
                    let in0 = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i0) as u32));
                    let col0 = weights_base.add(i0 * OUT_DIMS * Self::CHUNK_SIZE);
                    for l in 0..num_accums {
                        *acc.get_unchecked_mut(l) =
                            vdotq_s32(*acc.get_unchecked(l), in0, vld1q_s8(col0.add(l * 16)));
                    }
                    break;
                }

                let i1 = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if bits == 0 {
                    let in0 = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i0) as u32));
                    let in1 = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i1) as u32));
                    let col0 = weights_base.add(i0 * OUT_DIMS * Self::CHUNK_SIZE);
                    let col1 = weights_base.add(i1 * OUT_DIMS * Self::CHUNK_SIZE);
                    for l in 0..num_accums {
                        *acc.get_unchecked_mut(l) =
                            vdotq_s32(*acc.get_unchecked(l), in0, vld1q_s8(col0.add(l * 16)));
                        *acc.get_unchecked_mut(l + num_accums) = vdotq_s32(
                            *acc.get_unchecked(l + num_accums),
                            in1,
                            vld1q_s8(col1.add(l * 16)),
                        );
                    }
                    break;
                }

                let i2 = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let in0 = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i0) as u32));
                let in1 = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i1) as u32));
                let in2 = vreinterpretq_s8_u32(vdupq_n_u32(*base_addr.add(i2) as u32));
                let col0 = weights_base.add(i0 * OUT_DIMS * Self::CHUNK_SIZE);
                let col1 = weights_base.add(i1 * OUT_DIMS * Self::CHUNK_SIZE);
                let col2 = weights_base.add(i2 * OUT_DIMS * Self::CHUNK_SIZE);
                for l in 0..num_accums {
                    *acc.get_unchecked_mut(l) =
                        vdotq_s32(*acc.get_unchecked(l), in0, vld1q_s8(col0.add(l * 16)));
                    *acc.get_unchecked_mut(l + num_accums) = vdotq_s32(
                        *acc.get_unchecked(l + num_accums),
                        in1,
                        vld1q_s8(col1.add(l * 16)),
                    );
                    *acc.get_unchecked_mut(l + 2 * num_accums) = vdotq_s32(
                        *acc.get_unchecked(l + 2 * num_accums),
                        in2,
                        vld1q_s8(col2.add(l * 16)),
                    );
                }
            }
        }

        // Fold the three dependency chains and store.
        let out_ptr = output.as_mut_ptr();
        for l in 0..num_accums {
            let sum = vaddq_s32(
                vaddq_s32(*acc.get_unchecked(l), *acc.get_unchecked(l + num_accums)),
                *acc.get_unchecked(l + 2 * num_accums),
            );
            vst1q_s32(out_ptr.add(l * OUTPUT_SIMD_WIDTH), sum);
        }
    }

    /// AVX-512 sparse propagation.  VNNI uses three independent accumulator
    /// chains, matching Stockfish's `USE_VNNI` path; the fallback keeps one.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    unsafe fn propagate_sparse_i32(&self, input: &[u8], output: &mut [i32]) {
        self.propagate_sparse_i32_with_nnz(input, output, None);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    unsafe fn propagate_sparse_i32_with_nnz(
        &self,
        input: &[u8],
        output: &mut [i32],
        nnz_info: Option<&NnzInfo<256>>,
    ) {
        use core::arch::x86_64::*;

        if !OUT_DIMS.is_multiple_of(16) {
            self.propagate_sparse_i32_scalar(input, output);
            return;
        }

        const MAX_CHUNKS: usize = 768;
        const MAX_ACCUMS: usize = 16;
        #[cfg(target_feature = "avx512vnni")]
        const NUM_REGS: usize = 3 * MAX_ACCUMS;
        #[cfg(not(target_feature = "avx512vnni"))]
        const NUM_REGS: usize = MAX_ACCUMS;
        let num_chunks = Self::NUM_CHUNKS;
        let num_accums = OUT_DIMS / 16;
        debug_assert!(num_chunks <= MAX_CHUNKS);
        debug_assert!(num_accums <= MAX_ACCUMS);

        let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
        let mut nnz_storage = [MaybeUninit::<u16>::uninit(); MAX_CHUNKS];
        let (nnz_ptr, count) = if let Some(info) = nnz_info {
            (info.indices().as_ptr(), info.count())
        } else {
            let nnz = &mut nnz_storage[..num_chunks];
            let count = find_nnz(input32, nnz, num_chunks) as usize;
            // SAFETY: `find_nnz` initialised `nnz[..count]` and `count <= num_chunks`.
            (unsafe { init_prefix(nnz, count) }.as_ptr(), count)
        };

        let mut acc = [_mm512_setzero_si512(); NUM_REGS];
        for k in 0..num_accums {
            acc[k] = _mm512_loadu_si512(self.biases.as_ptr().add(k * 16) as *const __m512i);
        }
        #[cfg(target_feature = "avx512vnni")]
        for k in num_accums..NUM_REGS {
            acc[k] = _mm512_setzero_si512();
        }

        let weights = self.weights.as_ptr();
        let mut pos = 0usize;
        #[cfg(target_feature = "avx512vnni")]
        {
            while pos + 2 < count {
                let i0 = *nnz_ptr.add(pos) as usize;
                let i1 = *nnz_ptr.add(pos + 1) as usize;
                let i2 = *nnz_ptr.add(pos + 2) as usize;
                let in0 = _mm512_set1_epi32(*input32.get_unchecked(i0));
                let in1 = _mm512_set1_epi32(*input32.get_unchecked(i1));
                let in2 = _mm512_set1_epi32(*input32.get_unchecked(i2));
                let col0 = weights.add(i0 * OUT_DIMS * Self::CHUNK_SIZE) as *const __m512i;
                let col1 = weights.add(i1 * OUT_DIMS * Self::CHUNK_SIZE) as *const __m512i;
                let col2 = weights.add(i2 * OUT_DIMS * Self::CHUNK_SIZE) as *const __m512i;
                for k in 0..num_accums {
                    let w0 = _mm512_loadu_si512(col0.add(k));
                    let w1 = _mm512_loadu_si512(col1.add(k));
                    let w2 = _mm512_loadu_si512(col2.add(k));
                    crate::nnue::simd::avx512::m512_add_dpbusd_epi32(&mut acc[k], in0, w0);
                    crate::nnue::simd::avx512::m512_add_dpbusd_epi32(
                        &mut acc[k + num_accums],
                        in1,
                        w1,
                    );
                    crate::nnue::simd::avx512::m512_add_dpbusd_epi32(
                        &mut acc[k + 2 * num_accums],
                        in2,
                        w2,
                    );
                }
                pos += 3;
            }

            for k in 0..num_accums {
                acc[k] = _mm512_add_epi32(
                    _mm512_add_epi32(acc[k], acc[k + num_accums]),
                    acc[k + 2 * num_accums],
                );
            }
        }

        while pos < count {
            let i = *nnz_ptr.add(pos) as usize;
            let in_vec = _mm512_set1_epi32(*input32.get_unchecked(i));
            let col = weights.add(i * OUT_DIMS * Self::CHUNK_SIZE) as *const __m512i;
            for k in 0..num_accums {
                let weights = _mm512_loadu_si512(col.add(k));
                crate::nnue::simd::avx512::m512_add_dpbusd_epi32(&mut acc[k], in_vec, weights);
            }
            pos += 1;
        }
        for k in 0..num_accums {
            _mm512_storeu_si512(output.as_mut_ptr().add(k * 16) as *mut __m512i, acc[k]);
        }
    }

    /// AVX2 sparse propagation, ported from Stockfish's `USE_AVX2` path.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_sparse_i32(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;

        // fc_0 is 1024 -> 32. Keep unusual test-only output widths on the
        // scalar path rather than dropping a partial AVX2 register.
        if !OUT_DIMS.is_multiple_of(8) {
            self.propagate_sparse_i32_scalar(input, output);
            return;
        }

        const MAX_CHUNKS: usize = 768;
        const MAX_ACCUMS: usize = 16;
        let num_chunks = Self::NUM_CHUNKS;
        let num_accums = OUT_DIMS / 8;
        debug_assert!(num_chunks <= MAX_CHUNKS);
        debug_assert!(num_accums <= MAX_ACCUMS);
        debug_assert_eq!((input.as_ptr() as usize) & 3, 0);
        debug_assert!(input.len() >= num_chunks * 4);

        let mut nnz_storage = [MaybeUninit::<u16>::uninit(); MAX_CHUNKS];
        let nnz_buf = &mut nnz_storage[..num_chunks];
        let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
        let count = find_nnz(input32, nnz_buf, num_chunks) as usize;
        // SAFETY: `find_nnz` initialised `nnz_buf[..count]` and `count <= num_chunks`.
        let nnz = init_prefix(nnz_buf, count);

        #[cfg(target_feature = "avxvnni")]
        const NUM_REGS: usize = 2 * MAX_ACCUMS;
        #[cfg(not(target_feature = "avxvnni"))]
        const NUM_REGS: usize = MAX_ACCUMS;
        let mut acc = [_mm256_setzero_si256(); NUM_REGS];
        for k in 0..num_accums {
            acc[k] = _mm256_loadu_si256(self.biases.as_ptr().add(k * 8) as *const __m256i);
        }
        #[cfg(target_feature = "avxvnni")]
        for k in num_accums..NUM_REGS {
            acc[k] = _mm256_setzero_si256();
        }

        let weights = self.weights.as_ptr();
        let mut pos = 0usize;
        #[cfg(target_feature = "avxvnni")]
        {
            while pos + 1 < count {
                let i0 = *nnz.get_unchecked(pos) as usize;
                let i1 = *nnz.get_unchecked(pos + 1) as usize;
                let in0 = _mm256_set1_epi32(*input32.get_unchecked(i0));
                let in1 = _mm256_set1_epi32(*input32.get_unchecked(i1));
                let col0 = weights.add(i0 * OUT_DIMS * Self::CHUNK_SIZE) as *const __m256i;
                let col1 = weights.add(i1 * OUT_DIMS * Self::CHUNK_SIZE) as *const __m256i;
                for k in 0..num_accums {
                    let w0 = _mm256_loadu_si256(col0.add(k));
                    let w1 = _mm256_loadu_si256(col1.add(k));
                    crate::nnue::simd::avx2::m256_add_dpbusd_epi32(&mut acc[k], in0, w0);
                    crate::nnue::simd::avx2::m256_add_dpbusd_epi32(
                        &mut acc[k + num_accums],
                        in1,
                        w1,
                    );
                }
                pos += 2;
            }

            for k in 0..num_accums {
                acc[k] = _mm256_add_epi32(acc[k], acc[k + num_accums]);
            }
        }

        while pos < count {
            let i = *nnz.get_unchecked(pos) as usize;
            let in_vec = _mm256_set1_epi32(*input32.get_unchecked(i));
            let col = weights.add(i * OUT_DIMS * Self::CHUNK_SIZE) as *const __m256i;
            for k in 0..num_accums {
                let w = _mm256_loadu_si256(col.add(k));
                crate::nnue::simd::avx2::m256_add_dpbusd_epi32(&mut acc[k], in_vec, w);
            }
            pos += 1;
        }

        for k in 0..num_accums {
            _mm256_storeu_si256(output.as_mut_ptr().add(k * 8) as *mut __m256i, acc[k]);
        }
    }

    /// SSSE3 fallback for pre-AVX x86-64 CPUs.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "ssse3",
        not(target_feature = "avx2")
    ))]
    #[target_feature(enable = "ssse3")]
    unsafe fn propagate_sparse_i32(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;
        if !OUT_DIMS.is_multiple_of(4) {
            self.propagate_sparse_i32_scalar(input, output);
            return;
        }

        const MAX_CHUNKS: usize = 768;
        const MAX_ACCUMS: usize = 16;
        let num_chunks = Self::NUM_CHUNKS;
        let num_accums = OUT_DIMS / 4;
        debug_assert!(num_chunks <= MAX_CHUNKS);
        debug_assert!(num_accums <= MAX_ACCUMS);

        let mut nnz_storage = [MaybeUninit::<u16>::uninit(); MAX_CHUNKS];
        let nnz_buf = &mut nnz_storage[..num_chunks];
        let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
        let count = find_nnz(input32, nnz_buf, num_chunks) as usize;
        // SAFETY: `find_nnz` initialised `nnz_buf[..count]` and `count <= num_chunks`.
        let nnz = init_prefix(nnz_buf, count);

        let mut acc = [_mm_setzero_si128(); MAX_ACCUMS];
        for k in 0..num_accums {
            acc[k] = _mm_loadu_si128(self.biases.as_ptr().add(k * 4) as *const __m128i);
        }
        let weights = self.weights.as_ptr();
        for &chunk in nnz {
            let i = chunk as usize;
            let in_vec = _mm_set1_epi32(*input32.get_unchecked(i));
            let col = weights.add(i * OUT_DIMS * Self::CHUNK_SIZE) as *const __m128i;
            for k in 0..num_accums {
                let product = _mm_maddubs_epi16(in_vec, _mm_loadu_si128(col.add(k)));
                acc[k] = _mm_add_epi32(acc[k], _mm_madd_epi16(product, _mm_set1_epi16(1)));
            }
        }
        for k in 0..num_accums {
            _mm_storeu_si128(output.as_mut_ptr().add(k * 4) as *mut __m128i, acc[k]);
        }
    }

    /// Scalar fallback for propagate_sparse_i32
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
    fn propagate_sparse_i32_scalar(&self, input: &[u8], output: &mut [i32]) {
        let num_chunks = Self::NUM_CHUNKS;

        // Use uninitialized stack buffer like C++ (no zero-init overhead)
        const MAX_CHUNKS: usize = 768;
        debug_assert!(num_chunks <= MAX_CHUNKS);

        // `find_nnz` writes nnz[0..count]; only that prefix is ever viewed as initialised.
        let mut nnz_storage = [MaybeUninit::<u16>::uninit(); MAX_CHUNKS];
        let nnz_buf = &mut nnz_storage[..num_chunks];

        let input32 =
            // SAFETY: `input` is a 4-byte aligned padded layer-input buffer with at least
            // `num_chunks * 4` bytes (asserted in `propagate`), so viewing it as
            // `num_chunks` i32 is valid.
            unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks) };
        let count = find_nnz(input32, nnz_buf, num_chunks);
        // SAFETY: count <= num_chunks, and find_nnz initialized nnz_buf[0..count]
        let nnz = unsafe { init_prefix(nnz_buf, count as usize) };
        for j in 0..count {
            let idx = nnz[j] as usize;
            if idx < input32.len() {
                let in_val = input32[idx];
                let weight_offset = idx * OUT_DIMS * Self::CHUNK_SIZE;

                for k in 0..OUT_DIMS {
                    let mut sum = 0i32;
                    for b in 0..Self::CHUNK_SIZE {
                        let input_byte = ((in_val >> (b * 8)) & 0xFF) as u8;
                        let weight_idx = weight_offset + k * Self::CHUNK_SIZE + b;
                        if weight_idx < self.weights.len() {
                            sum += (input_byte as i32) * (self.weights[weight_idx] as i32);
                        }
                    }
                    output[k] += sum;
                }
            }
        }
    }

    /// Dense propagation fallback (i32 output): scalar builds, and AArch64 shapes the NEON
    /// kernels do not cover. Adds to the biases already in `output`.
    ///
    /// Weights are addressed through `get_weight_index`, which is the identity in scalar builds
    /// and the chunk-scrambled layout under NEON.
    #[cfg(any(
        not(any(target_feature = "ssse3", target_feature = "neon")),
        target_arch = "aarch64"
    ))]
    fn propagate_dense_i32(&self, input: &[u8], output: &mut [i32]) {
        // Dense matrix multiplication
        for i in 0..OUT_DIMS {
            for j in 0..IN_DIMS.min(input.len()) {
                if j < Self::PADDED_INPUT_DIMENSIONS {
                    output[i] += (input[j] as i32)
                        * (self.weights
                            [Self::get_weight_index(i * Self::PADDED_INPUT_DIMENSIONS + j)]
                            as i32);
                }
            }
        }
    }

    /// Test-only reference implementation, in terms of the *logical* weight
    /// matrix: `output[k] = biases[k] + sum_j input[j] * W[k][j]`.
    ///
    /// The physical storage of `W` depends on `CHUNK_SIZE`: 4 on SSSE3/NEON,
    /// where `read_parameters` scrambles the weights into the column-major
    /// chunk order the sparse kernels index, and 1 otherwise, where they are
    /// stored plainly as `k * PADDED_INPUT_DIMENSIONS + j`. Addressing through
    /// `get_weight_index` — a bijection onto the storage array under both
    /// layouts — keeps this reference valid for either build.
    #[cfg(test)]
    pub fn propagate_dense_i32_test(&self, input: &[u8], output: &mut [i32]) {
        let n = IN_DIMS.min(input.len()).min(Self::PADDED_INPUT_DIMENSIONS);
        for k in 0..OUT_DIMS {
            let mut sum = self.biases[k];
            for j in 0..n {
                let w = self.weights[Self::get_weight_index(k * Self::PADDED_INPUT_DIMENSIONS + j)];
                sum += (input[j] as i32) * (w as i32);
            }
            output[k] = sum;
        }
    }
}

/// Runs `find_nnz` on `input` after checking its contracts, writing the indices of the
/// non-zero 4-byte chunks into `nnz_out` and returning how many were written. A safe entry
/// point to the per-ISA kernel for the differential tests.
pub fn find_nnz_into(input: &[u8], num_chunks: usize, nnz_out: &mut [u16]) -> usize {
    assert!(
        num_chunks <= u16::MAX as usize,
        "NNZ indices must fit in u16"
    );
    assert!(input.len() >= num_chunks * 4, "NNZ input is too short");
    assert!(nnz_out.len() >= num_chunks, "NNZ output is too short");
    assert!(
        (input.as_ptr() as usize).is_multiple_of(4),
        "NNZ input must be 4-byte aligned"
    );
    // SAFETY: the release assertions establish alignment, input length and
    // worst-case output capacity, including the kernels' full vector stores.
    let input32 = unsafe { core::slice::from_raw_parts(input.as_ptr().cast::<i32>(), num_chunks) };
    find_nnz(input32, as_uninit_u16(nnz_out), num_chunks) as usize
}

impl<const IN_DIMS: usize, const OUT_DIMS: usize> Default
    for AffineTransformSparseInput<IN_DIMS, OUT_DIMS>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sets the *logical* weight connecting input `j` to output `k`.
    ///
    /// These tests must hold for both storage layouts (`CHUNK_SIZE` 4 on
    /// SSSE3/NEON builds, 1 on the portable baseline), so they never poke raw
    /// storage indices; `get_weight_index` maps the logical position to
    /// whichever slot the active layout uses.
    fn set_weight<const IN: usize, const OUT: usize>(
        layer: &mut AffineTransformSparseInput<IN, OUT>,
        k: usize,
        j: usize,
        v: i8,
    ) {
        // Map the logical (k, j) position to whichever storage slot the active layout uses.
        let padded = AffineTransformSparseInput::<IN, OUT>::PADDED_INPUT_DIMENSIONS;
        let idx = AffineTransformSparseInput::<IN, OUT>::get_weight_index(k * padded + j);
        layer.weights[idx] = v;
    }

    /// Guards against accumulator counts that do not cover all OUT_DIMS outputs; all 32 outputs
    /// of the real fc_0 geometry (`AffineTransformSparseInput<1024, 32>`) must match the dense
    /// reference.
    #[test]
    fn test_propagate_matches_dense_for_network_dimensions() {
        type NetLayer = AffineTransformSparseInput<1024, 32>;

        #[repr(align(64))]
        struct AlignedInput([u8; 1024]);

        let mut layer = NetLayer::new();

        // Deterministic xorshift PRNG so the test is reproducible.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for i in 0..layer.biases.len() {
            layer.biases[i] = (next() % 4001) as i32 - 2000;
        }
        for i in 0..layer.weights.len() {
            layer.weights[i] = (next() % 256) as u8 as i8;
        }

        let mut input = AlignedInput([0u8; 1024]);
        for i in 0..1024 {
            // Sparse, in the clipped-ReLU output range [0, 127]
            input.0[i] = if next() % 5 == 0 {
                (next() % 128) as u8
            } else {
                0
            };
        }

        let mut simd_out = [0i32; 32];
        let mut dense_out = [0i32; 32];
        layer.propagate(&input.0, &mut simd_out);
        layer.propagate_dense_i32_test(&input.0, &mut dense_out);

        assert_eq!(
            simd_out, dense_out,
            "SIMD propagate must match the reference dot product for all 32 outputs"
        );
    }

    /// The bitset walk (fed by the feature transformer in `Network::evaluate`) must give the
    /// dense reference for every bit pattern class: empty words, words with 1, 2 and 3 + n
    /// chunks (the three tail bodies of the SDOT kernel), full words, and set bits over
    /// all-zero chunks.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_propagate_with_bitset_matches_dense() {
        type NetLayer = AffineTransformSparseInput<1024, 32>;

        #[repr(align(64))]
        struct AlignedInput([u8; 1024]);

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut layer = NetLayer::new();
        for i in 0..layer.biases.len() {
            layer.biases[i] = (next() % 4001) as i32 - 2000;
        }
        for i in 0..layer.weights.len() {
            layer.weights[i] = (next() % 256) as u8 as i8;
        }

        // Non-zero chunks per 64-chunk word, for the four words of the input.
        let patterns: [[usize; 4]; 8] = [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 64],
            [64, 0, 1, 63],
            [0, 0, 0, 1],
            [2, 0, 0, 0],
            [9, 10, 11, 12],
            [64, 64, 64, 64],
        ];
        for (round, pattern) in patterns.iter().enumerate() {
            let mut input = AlignedInput([0u8; 1024]);
            let mut bitset = [0u8; 1024 / 32];
            for (word, &chunks) in pattern.iter().enumerate() {
                let mut placed = 0;
                while placed < chunks {
                    let chunk = word * 64 + (next() % 64) as usize;
                    if bitset[chunk / 8] & (1 << (chunk % 8)) != 0 {
                        continue;
                    }
                    bitset[chunk / 8] |= 1 << (chunk % 8);
                    // One to four non-zero bytes in the clipped range [1, 127].
                    let always = (next() % 4) as usize;
                    for b in 0..4 {
                        if b == always || next() % 2 == 0 {
                            input.0[chunk * 4 + b] = 1 + (next() % 127) as u8;
                        }
                    }
                    placed += 1;
                }
            }

            let mut dense_out = [0i32; 32];
            layer.propagate_dense_i32_test(&input.0, &mut dense_out);

            let mut bitset_out = [0i32; 32];
            layer.propagate_with_bitset(&input.0, &mut bitset_out, &bitset);
            assert_eq!(bitset_out, dense_out, "pattern {round}");

            let mut find_nnz_out = [0i32; 32];
            layer.propagate(&input.0, &mut find_nnz_out);
            assert_eq!(find_nnz_out, dense_out, "pattern {round} (find_nnz path)");

            // Extra set bits over all-zero chunks must not change the result.
            let mut bitset_out = [0i32; 32];
            layer.propagate_with_bitset(&input.0, &mut bitset_out, &[0xff; 1024 / 32]);
            assert_eq!(bitset_out, dense_out, "pattern {round} with a full bitset");
        }
    }

    #[test]
    fn test_weight_index_scrambling() {
        type TestLayer = AffineTransformSparseInput<768, 15>;

        // Test that scrambling produces valid indices
        for i in 0..100 {
            let idx = TestLayer::get_weight_index(i);
            assert!(idx < 15 * TestLayer::PADDED_INPUT_DIMENSIONS);
        }
    }

    #[test]
    fn test_hash_value() {
        let hash1 = AffineTransformSparseInput::<768, 15>::get_hash_value(0x12345678);
        let hash2 = AffineTransformSparseInput::<768, 16>::get_hash_value(0x12345678);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_propagate() {
        let layer = AffineTransformSparseInput::<4, 2>::new();
        // Padded to 8 bytes; sparse path reads in 4-byte chunks after rounding IN_DIMS up to 8.
        let input = [0u8; 8]; // TransformedFeatureType = u8
        let mut output = [0i32; 2];

        layer.propagate(&input, &mut output);

        // With zero weights and biases, output should be zero
        assert_eq!(output, [0, 0]);
    }

    /// Test that num_chunks calculation is correct for sparse input
    /// C++ formula: ceil_to_multiple(InputDimensions, 8) / ChunkSize
    /// = ((IN_DIMS + 7) / 8 * 8) / 4 = ((IN_DIMS + 7) / 8) * 2
    #[test]
    fn test_num_chunks_calculation() {
        // For IN_DIMS = 768: ((768 + 7) / 8) * 2 = 96 * 2 = 192 chunks
        // For IN_DIMS = 32:  ((32 + 7) / 8) * 2 = 4 * 2 = 8 chunks
        // For IN_DIMS = 16:  ((16 + 7) / 8) * 2 = 2 * 2 = 4 chunks

        assert_eq!(AffineTransformSparseInput::<768, 15>::NUM_CHUNKS, 192);
        assert_eq!(AffineTransformSparseInput::<32, 15>::NUM_CHUNKS, 8);
        assert_eq!(AffineTransformSparseInput::<16, 15>::NUM_CHUNKS, 4);
    }

    /// Every byte of each 4-byte chunk must contribute: the weight index is
    /// `weight_offset + k * CHUNK_SIZE + b`.
    #[test]
    fn test_dot_product_processes_all_chunk_bytes() {
        type TestLayer = AffineTransformSparseInput<8, 2>;
        let mut layer = TestLayer::new();

        layer.biases[0] = 0;
        layer.biases[1] = 0;

        // W[0] = [1, 2, 3, 4, 1, 1, 1, 1], W[1] = [1, 1, 1, 1, 2, 2, 2, 2]
        for (j, w) in [1i8, 2, 3, 4, 1, 1, 1, 1].iter().enumerate() {
            set_weight(&mut layer, 0, j, *w);
        }
        for (j, w) in [1i8, 1, 1, 1, 2, 2, 2, 2].iter().enumerate() {
            set_weight(&mut layer, 1, j, *w);
        }

        let input_bytes: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        let mut output = [0i32; 2];
        layer.propagate_dense_i32_test(&input_bytes, &mut output);

        // output[0] = (1*1 + 2*2 + 3*3 + 4*4) + (5 + 6 + 7 + 8)     = 30 + 26 = 56
        // output[1] = (1 + 2 + 3 + 4) + (5*2 + 6*2 + 7*2 + 8*2)     = 10 + 52 = 62
        assert_eq!(
            output[0], 56,
            "Output 0 should process all 4 bytes per chunk"
        );
        assert_eq!(
            output[1], 62,
            "Output 1 should process all 4 bytes per chunk"
        );
    }

    /// Test public propagate API produces consistent results
    /// Uses realistic dimensions that exercise the chunked processing
    #[test]
    fn test_propagate_public_api() {
        // Use 32 u8 inputs (TransformedFeatureType = u8 after clipped ReLU)
        type TestLayer = AffineTransformSparseInput<32, 2>;

        #[repr(align(64))]
        struct AlignedInput([u8; 32]);

        let mut layer = TestLayer::new();

        // Set biases
        layer.biases[0] = 100;
        layer.biases[1] = 200;

        // Distinct non-zero logical weights for every (output, input) pair.
        for k in 0..2 {
            for j in 0..32 {
                set_weight(&mut layer, k, j, (3 * j + 7 * k) as i8 - 40);
            }
        }

        // Create u8 input that has some nonzero values
        // TransformedFeatureType = u8 (clipped ReLU output in range 0-127)
        let mut input = AlignedInput([0u8; 32]); // 32 bytes for 32 input elements
        input.0[0] = 1; // First byte nonzero
        input.0[8] = 2; // Ninth byte nonzero
        input.0[31] = 5; // Last byte of the last chunk

        let mut output = [0i32; 2];
        layer.propagate(&input.0, &mut output);

        // Exact values: biases plus the logical dot products.
        let mut expected = [100i32, 200];
        for k in 0..2 {
            for j in 0..32 {
                expected[k] += i32::from(input.0[j]) * ((3 * j + 7 * k) as i32 - 40);
            }
        }
        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic(expected = "input must be 4-byte aligned")]
    fn test_propagate_panics_on_unaligned_input() {
        #[repr(align(64))]
        struct Aligned([u8; 16]);
        let buf = Aligned([0u8; 16]);
        let input = &buf.0[1..9]; // len=8, unaligned pointer
        let layer = AffineTransformSparseInput::<4, 2>::new();
        let mut output = [0i32; 2];
        layer.propagate(input, &mut output);
    }

    /// The reference implementation must apply biases and weights as
    /// `output[k] = biases[k] + sum_j input[j] * W[k][j]`.
    #[test]
    fn test_dense_test_function_correctness() {
        type TestLayer = AffineTransformSparseInput<8, 2>;
        let mut layer = TestLayer::new();

        layer.biases[0] = 50;
        layer.biases[1] = 100;

        // W[0][0..4] = 1, W[1][0..4] = 2; everything else stays zero.
        for j in 0..4 {
            set_weight(&mut layer, 0, j, 1);
            set_weight(&mut layer, 1, j, 2);
        }

        let input_bytes: [u8; 8] = [1, 1, 1, 1, 0, 0, 0, 0];

        let mut output = [0i32; 2];
        layer.propagate_dense_i32_test(&input_bytes, &mut output);

        // output[0] = 50  + (1 + 1 + 1 + 1) = 54
        // output[1] = 100 + (2 + 2 + 2 + 2) = 108
        assert_eq!(output[0], 54, "Output 0 computation incorrect");
        assert_eq!(output[1], 108, "Output 1 computation incorrect");
    }

    /// Every chunk must be visited, not just the leading few.
    #[test]
    fn test_all_chunks_processed() {
        type TestLayer = AffineTransformSparseInput<32, 1>;
        let mut layer = TestLayer::new();

        layer.biases[0] = 0;
        for j in 0..32 {
            set_weight(&mut layer, 0, j, 1);
        }

        let input_bytes = [1u8; 32];

        let mut output = [0i32; 1];
        layer.propagate_dense_i32_test(&input_bytes, &mut output);

        assert_eq!(
            output[0], 32,
            "All 32 bytes across 8 chunks should be processed"
        );
    }

    /// Test with the actual dimensions used in the network (768 inputs)
    #[test]
    fn test_realistic_dimensions() {
        type TestLayer = AffineTransformSparseInput<768, 15>;

        #[repr(align(64))]
        struct AlignedInput([u8; 768]);

        // num_chunks should be 192
        assert_eq!(TestLayer::NUM_CHUNKS, 192);

        // PADDED_INPUT_DIMENSIONS should accommodate all chunks
        assert!(TestLayer::PADDED_INPUT_DIMENSIONS >= 768);

        // Non-zero biases and weights, sparse non-zero input (deterministic xorshift).
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut layer = TestLayer::new();
        for k in 0..15 {
            layer.biases[k] = (next() % 20001) as i32 - 10000;
            for j in 0..768 {
                set_weight(&mut layer, k, j, (next() % 256) as u8 as i8);
            }
        }
        let mut input = AlignedInput([0u8; 768]); // u8 = TransformedFeatureType
        for j in 0..768 {
            if next() % 4 == 0 {
                input.0[j] = (next() % 128) as u8;
            }
        }

        // The reference is itself checked in `test_dense_test_function_correctness`.
        let mut expected = [0i32; 15];
        layer.propagate_dense_i32_test(&input.0, &mut expected);
        assert!(expected.iter().any(|&v| v != 0));

        let mut output = [0i32; 15];
        layer.propagate(&input.0, &mut output);
        assert_eq!(output, expected);
    }

    /// Test that nonzero detection works correctly
    #[test]
    fn test_find_nnz() {
        // Create input with some nonzero chunks
        let input: [i32; 8] = [0, 1, 0, 0, 2, 0, 3, 0];
        let mut nnz = [0u16; 8];

        let count = find_nnz(&input, as_uninit_u16(&mut nnz), 8);

        // Should find 3 nonzero elements at indices 1, 4, 6
        assert_eq!(count, 3);
        assert_eq!(nnz[0], 1);
        assert_eq!(nnz[1], 4);
        assert_eq!(nnz[2], 6);
    }

    /// `get_weight_index` must be a bijection onto the weight storage, so no logical
    /// position aliases another, and a weight set at one logical position must not leak
    /// into any other.
    #[test]
    fn test_weight_indexing_correctness() {
        type TestLayer = AffineTransformSparseInput<8, 2>;
        const PADDED: usize = TestLayer::PADDED_INPUT_DIMENSIONS;
        const TOTAL: usize = 2 * PADDED;

        // Bijection: every logical index maps to a distinct in-range slot.
        let mut seen = vec![false; TOTAL];
        for i in 0..TOTAL {
            let idx = TestLayer::get_weight_index(i);
            assert!(
                idx < TOTAL,
                "logical index {} maps out of range to {}",
                i,
                idx
            );
            assert!(
                !seen[idx],
                "logical index {} aliases storage slot {}",
                i, idx
            );
            seen[idx] = true;
        }

        let mut layer = TestLayer::new();
        layer.biases[0] = 0;
        layer.biases[1] = 0;

        // W[0] = [100, 100, 1, 1, 100, 100, 1, 1], W[1] = all 1.
        for (j, w) in [100i8, 100, 1, 1, 100, 100, 1, 1].iter().enumerate() {
            set_weight(&mut layer, 0, j, *w);
        }
        for j in 0..8 {
            set_weight(&mut layer, 1, j, 1);
        }

        let input_bytes = [1u8; 8];

        let mut output = [0i32; 2];
        layer.propagate_dense_i32_test(&input_bytes, &mut output);

        // output[0] = 100 + 100 + 1 + 1 + 100 + 100 + 1 + 1 = 404
        // output[1] = 1 * 8                                  = 8
        assert_eq!(
            output[0], 404,
            "Output 0 should use all 4 weights per chunk"
        );
        assert_eq!(output[1], 8, "Output 1 should use correct weight indices");
    }
}
