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

//! Clipped ReLU activation layer for NNUE
//!
//! This layer applies ReLU activation followed by clipping to [0, 127]
//! and quantization from int32 to uint8.
//!
//! The formula is: output = clamp(input >> WeightScaleBitsLocal, 0, 127)
//!
//! `WEIGHT_SCALE_BITS_LOCAL` is a per-layer shift amount (C++ template parameter
//! `WeightScaleBitsLocal`, default `WEIGHT_SCALE_BITS`). SFNNv15 instantiates the first
//! activation with `WEIGHT_SCALE_BITS + 1`. It is a `usize` const generic to match how the
//! network architecture module instantiates it.
//!
//! NOTE ON SIMD SHIFTS: because the shift amount is a const generic and Rust's
//! legacy-const-generic intrinsic arguments (e.g. `_mm_srli_epi16`, `vqshrn_n_s32`) require a
//! *standalone i32* const argument (a `usize` param cannot be cast there without
//! `generic_const_exprs`), the SIMD paths use the runtime/variable-shift intrinsics
//! (`_mm_srl_epi16`, `_mm_sra_epi16`, `vshlq_s32`) which accept a computed shift.

// Scalar, SSE, AVX2, AVX-512 and NEON implementations of each kernel coexist in
// this file and the target picks one at compile time (`scripts/check-all.sh` compiles every
// target to keep all of them building).

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::nnue::nnue_common::{IndexType, WEIGHT_SCALE_BITS};
use std::io::{Read, Write};

/// Clipped ReLU activation: output = clamp(input >> WeightScaleBitsLocal, 0, 127)
/// Input type: i32, Output type: u8
#[derive(Clone)]
pub struct ClippedReLU<
    const IN_DIMS: usize,
    const WEIGHT_SCALE_BITS_LOCAL: usize = { WEIGHT_SCALE_BITS as usize },
> {
    _phantom: core::marker::PhantomData<()>,
}

impl<const IN_DIMS: usize, const WEIGHT_SCALE_BITS_LOCAL: usize>
    ClippedReLU<IN_DIMS, WEIGHT_SCALE_BITS_LOCAL>
{
    // Input/output dimensions
    pub const INPUT_DIMENSIONS: IndexType = IN_DIMS as IndexType;
    pub const OUTPUT_DIMENSIONS: IndexType = IN_DIMS as IndexType;
    pub const PADDED_OUTPUT_DIMENSIONS: usize = IN_DIMS.div_ceil(32) * 32;

    /// Create a new clipped ReLU layer
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }

    /// Get hash value for this layer
    ///
    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value(prev_hash: u32) -> u32 {
        let hash_value = 0x538D24C7u32;
        // NOTE: WeightScaleBitsLocal intentionally omitted from the hash value.
        // Matches upstream C++: not written by trainer (yet).
        hash_value.wrapping_add(prev_hash)
    }

    /// Read parameters from stream (no parameters for ReLU)
    pub fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> std::io::Result<()> {
        Ok(())
    }

    /// Write parameters to stream (no parameters for ReLU)
    pub fn write_parameters<W: Write>(&self, _writer: &mut W) -> std::io::Result<()> {
        Ok(())
    }

    /// Forward propagation from i32 input to u8 output
    pub fn propagate(&self, input: &[i32], output: &mut [u8]) {
        // The SIMD kernels below read/write exactly the first IN_DIMS elements, unchecked.
        assert!(input.len() >= IN_DIMS && output.len() >= IN_DIMS);

        // Runtime feature detection and dispatch
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 was just verified by `is_x86_feature_detected!`. The kernel's
                // unaligned loads/stores touch only `IN_DIMS` elements of `input`/`output`; the
                // length is asserted at the top of `propagate`.
                unsafe {
                    self.propagate_avx2(input, output);
                }
                return;
            }

            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: SSE4.1 was just verified by `is_x86_feature_detected!`; the kernel's
                // accesses stay within the first `IN_DIMS` elements of both slices (unaligned
                // load/store; lengths asserted at the top of `propagate`).
                unsafe {
                    self.propagate_sse41(input, output);
                }
                return;
            }

            if is_x86_feature_detected!("sse2") {
                // SAFETY: SSE2 was just verified by `is_x86_feature_detected!`; same slice-length
                // contract as above.
                unsafe {
                    self.propagate_sse2(input, output);
                }
            } else {
                // 32-bit x86 without SSE2.
                self.propagate_scalar(input, output);
            }
        }

        // NEON is always available on aarch64 - no runtime check needed
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is part of the aarch64 baseline; the kernel accesses only the first
            // `IN_DIMS` elements of both slices (lengths asserted at the top of `propagate`).
            unsafe {
                self.propagate_neon(input, output);
            }
            return;
        }

        // Fallback to scalar implementation
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        self.propagate_scalar(input, output);
    }

    /// AVX2 optimized implementation
    ///
    /// NOTE: upstream C++ dropped its dedicated 256-bit AVX2 path because paired activations
    /// (`SqrClippedReLU::propagate_pair`) keep the interleaved pack order. The Rust port builds
    /// with `USE_PAIR_ACTIVATIONS` only when avx2 is a compile-time target feature, and then does
    /// not call this layer. This kernel serves runtime-detected AVX2 on builds where pairing is
    /// off, and un-scrambles via VPERMD to natural byte order.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2(&self, input: &[i32], output: &mut [u8]) {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        const SIMD_WIDTH_AVX2: usize = 32;
        let shift = _mm_cvtsi32_si128(WEIGHT_SCALE_BITS_LOCAL as i32);

        if IN_DIMS.is_multiple_of(SIMD_WIDTH_AVX2) {
            // Aligned case: process 32 values at a time
            let num_chunks = IN_DIMS / SIMD_WIDTH_AVX2;

            // Permutation indices for proper byte ordering
            let offsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

            let in_ptr = input.as_ptr() as *const __m256i;
            let out_ptr = output.as_mut_ptr() as *mut __m256i;

            for i in 0..num_chunks {
                // Load 4 vectors of 8 int32 values each (32 values total)
                let in0 = _mm256_loadu_si256(in_ptr.add(i * 4 + 0));
                let in1 = _mm256_loadu_si256(in_ptr.add(i * 4 + 1));
                let in2 = _mm256_loadu_si256(in_ptr.add(i * 4 + 2));
                let in3 = _mm256_loadu_si256(in_ptr.add(i * 4 + 3));

                // Pack and shift: int32 -> uint16 with saturation, then shift right
                let words0 = _mm256_srl_epi16(_mm256_packus_epi32(in0, in1), shift);
                let words1 = _mm256_srl_epi16(_mm256_packus_epi32(in2, in3), shift);

                // Pack uint16 -> int8 with signed saturation
                let packed = _mm256_packs_epi16(words0, words1);

                // Permute to get correct byte ordering
                let result = _mm256_permutevar8x32_epi32(packed, offsets);

                // Store result
                _mm256_storeu_si256(out_ptr.add(i), result);
            }

            // No remainder to process
        } else {
            // Unaligned case: process 16 values at a time using 128-bit ops
            let num_chunks = IN_DIMS / 16;

            let in_ptr = input.as_ptr() as *const __m128i;
            let out_ptr = output.as_mut_ptr() as *mut __m128i;

            for i in 0..num_chunks {
                // Load 4 vectors of 4 int32 values each (16 values total)
                let in0 = _mm_loadu_si128(in_ptr.add(i * 4 + 0));
                let in1 = _mm_loadu_si128(in_ptr.add(i * 4 + 1));
                let in2 = _mm_loadu_si128(in_ptr.add(i * 4 + 2));
                let in3 = _mm_loadu_si128(in_ptr.add(i * 4 + 3));

                // Pack and shift
                let words0 = _mm_srl_epi16(_mm_packus_epi32(in0, in1), shift);
                let words1 = _mm_srl_epi16(_mm_packus_epi32(in2, in3), shift);

                // Pack to bytes
                let result = _mm_packs_epi16(words0, words1);

                // Store result
                _mm_storeu_si128(out_ptr.add(i), result);
            }

            // Process remainder with scalar code
            let start = num_chunks * 16;
            for i in start..IN_DIMS {
                output[i] = ((input[i] >> WEIGHT_SCALE_BITS_LOCAL).clamp(0, 127)) as u8;
            }
        }
    }

    /// SSE4.1 optimized implementation
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn propagate_sse41(&self, input: &[i32], output: &mut [u8]) {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        const SIMD_WIDTH_SSE: usize = 16;
        let num_chunks = IN_DIMS / SIMD_WIDTH_SSE;
        let shift = _mm_cvtsi32_si128(WEIGHT_SCALE_BITS_LOCAL as i32);

        let in_ptr = input.as_ptr() as *const __m128i;
        let out_ptr = output.as_mut_ptr() as *mut __m128i;

        for i in 0..num_chunks {
            // Load 4 vectors of 4 int32 values each
            let in0 = _mm_loadu_si128(in_ptr.add(i * 4 + 0));
            let in1 = _mm_loadu_si128(in_ptr.add(i * 4 + 1));
            let in2 = _mm_loadu_si128(in_ptr.add(i * 4 + 2));
            let in3 = _mm_loadu_si128(in_ptr.add(i * 4 + 3));

            // Pack int32 -> uint16 with unsigned saturation (SSE4.1)
            let words0 = _mm_srl_epi16(_mm_packus_epi32(in0, in1), shift);
            let words1 = _mm_srl_epi16(_mm_packus_epi32(in2, in3), shift);

            // Pack uint16 -> int8 with signed saturation
            let result = _mm_packs_epi16(words0, words1);

            // Store result
            _mm_storeu_si128(out_ptr.add(i), result);
        }

        // Process remainder with scalar code
        let start = num_chunks * SIMD_WIDTH_SSE;
        for i in start..IN_DIMS {
            output[i] = ((input[i] >> WEIGHT_SCALE_BITS_LOCAL).clamp(0, 127)) as u8;
        }
    }

    /// SSE2 optimized implementation (without SSE4.1)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn propagate_sse2(&self, input: &[i32], output: &mut [u8]) {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        const SIMD_WIDTH_SSE: usize = 16;
        let num_chunks = IN_DIMS / SIMD_WIDTH_SSE;
        let shift = _mm_cvtsi32_si128(WEIGHT_SCALE_BITS_LOCAL as i32);

        // Constant for saturation adjustment
        let k0x80s = _mm_set1_epi8(-128i8);

        let in_ptr = input.as_ptr() as *const __m128i;
        let out_ptr = output.as_mut_ptr() as *mut __m128i;

        for i in 0..num_chunks {
            // Load 4 vectors of 4 int32 values each
            let in0 = _mm_loadu_si128(in_ptr.add(i * 4 + 0));
            let in1 = _mm_loadu_si128(in_ptr.add(i * 4 + 1));
            let in2 = _mm_loadu_si128(in_ptr.add(i * 4 + 2));
            let in3 = _mm_loadu_si128(in_ptr.add(i * 4 + 3));

            // Pack int32 -> int16 with signed saturation (SSE2)
            // Then arithmetic shift right (preserves sign)
            let words0 = _mm_sra_epi16(_mm_packs_epi32(in0, in1), shift);
            let words1 = _mm_sra_epi16(_mm_packs_epi32(in2, in3), shift);

            // Pack int16 -> int8 with signed saturation
            let packed_bytes = _mm_packs_epi16(words0, words1);

            // Adjust for unsigned saturation using saturated add/sub
            // This converts signed saturation to unsigned [0, 127] range
            let adjusted = _mm_subs_epi8(_mm_adds_epi8(packed_bytes, k0x80s), k0x80s);

            // Store result
            _mm_storeu_si128(out_ptr.add(i), adjusted);
        }

        // Process remainder with scalar code
        let start = num_chunks * SIMD_WIDTH_SSE;
        for i in start..IN_DIMS {
            output[i] = ((input[i] >> WEIGHT_SCALE_BITS_LOCAL).clamp(0, 127)) as u8;
        }
    }

    /// NEON optimized implementation for ARM
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn propagate_neon(&self, input: &[i32], output: &mut [u8]) {
        use core::arch::aarch64::*;

        const SIMD_WIDTH_NEON: usize = 8; // Process 8 values at a time
        let num_chunks = IN_DIMS / SIMD_WIDTH_NEON;

        let zero = vdup_n_s8(0);
        // Variable right-shift (see module note): shift right by WEIGHT_SCALE_BITS_LOCAL.
        let neg_shift = vdupq_n_s32(-(WEIGHT_SCALE_BITS_LOCAL as i32));
        let in_ptr = input.as_ptr() as *const int32x4_t;
        let out_ptr = output.as_mut_ptr() as *mut int8x8_t;

        for i in 0..num_chunks {
            // Load 2 vectors of 4 int32 values each
            let in0 = vld1q_s32(in_ptr.add(i * 2) as *const i32);
            let in1 = vld1q_s32(in_ptr.add(i * 2 + 1) as *const i32);

            // Shift right (arithmetic) then narrow to int16 with saturation.
            // Equivalent to vqshrn_n_s32(in, WEIGHT_SCALE_BITS_LOCAL).
            let pack0 = vqmovn_s32(vshlq_s32(in0, neg_shift));
            let pack1 = vqmovn_s32(vshlq_s32(in1, neg_shift));

            // Combine two int16x4 into int16x8
            let shifted = vcombine_s16(pack0, pack1);

            // Narrow to int8 with saturation and clamp to [0, 127]
            let narrowed = vqmovn_s16(shifted);
            let result = vmax_s8(narrowed, zero);

            // Store result
            vst1_s8(out_ptr.add(i) as *mut i8, result);
        }

        // Process remainder with scalar code
        let start = num_chunks * SIMD_WIDTH_NEON;
        for i in start..IN_DIMS {
            output[i] = ((input[i] >> WEIGHT_SCALE_BITS_LOCAL).clamp(0, 127)) as u8;
        }
    }

    /// Scalar fallback for targets without a usable SIMD kernel (including x86 without SSE2);
    /// x86 tests use it as the reference.
    #[cfg(not(target_arch = "aarch64"))]
    fn propagate_scalar(&self, input: &[i32], output: &mut [u8]) {
        for i in 0..IN_DIMS {
            output[i] = ((input[i] >> WEIGHT_SCALE_BITS_LOCAL).clamp(0, 127)) as u8;
        }

        // Zero padding if needed
        for i in IN_DIMS..Self::PADDED_OUTPUT_DIMENSIONS {
            if i < output.len() {
                output[i] = 0;
            }
        }
    }
}

impl<const IN_DIMS: usize, const WEIGHT_SCALE_BITS_LOCAL: usize> Default
    for ClippedReLU<IN_DIMS, WEIGHT_SCALE_BITS_LOCAL>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_value() {
        let hash1 = ClippedReLU::<32>::get_hash_value(0x12345678);
        let hash2 = ClippedReLU::<32>::get_hash_value(0x87654321);
        assert_ne!(hash1, hash2);
        assert_eq!(hash1, 0x538D24C7u32.wrapping_add(0x12345678));
    }

    #[test]
    fn test_propagate_small_dims() {
        let layer = ClippedReLU::<4>::new();

        // Test with positive values
        let input = [64, 128, 256, 8192]; // Will be divided by 64 (>> 6)
        let mut output = [0u8; 4];
        layer.propagate(&input, &mut output);
        assert_eq!(output, [1, 2, 4, 127]); // 8192 >> 6 = 128, clamped to 127

        // Test with negative and zero values
        let input = [-100, 0, 100, -1];
        let mut output = [0u8; 4];
        layer.propagate(&input, &mut output);
        assert_eq!(output, [0, 0, 1, 0]);
    }

    /// Verify a non-default WeightScaleBitsLocal (WEIGHT_SCALE_BITS + 1 == 7) shifts correctly.
    #[test]
    fn test_propagate_scale_bits_plus_one() {
        let layer = ClippedReLU::<4, 7>::new();
        // 8192 >> 7 = 64, 256 >> 7 = 2, 128 >> 7 = 1, 64 >> 7 = 0
        let input = [64, 128, 256, 8192];
        let mut output = [0u8; 4];
        layer.propagate(&input, &mut output);
        assert_eq!(output, [0, 1, 2, 64]);
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(ClippedReLU::<15>::INPUT_DIMENSIONS, 15);
        assert_eq!(ClippedReLU::<15>::OUTPUT_DIMENSIONS, 15);
        assert_eq!(ClippedReLU::<15>::PADDED_OUTPUT_DIMENSIONS, 32);
        assert_eq!(ClippedReLU::<32>::PADDED_OUTPUT_DIMENSIONS, 32);
        assert_eq!(ClippedReLU::<33>::PADDED_OUTPUT_DIMENSIONS, 64);
    }

    #[test]
    fn test_edge_cases() {
        let layer = ClippedReLU::<5>::new();

        // Test edge values around clipping boundaries
        let input = [
            i32::MIN, // Large negative
            -1,       // Small negative
            0,        // Zero
            127 << 6, // Exactly 127 after shift
            i32::MAX, // Large positive
        ];
        let mut output = [0u8; 5];
        layer.propagate(&input, &mut output);
        assert_eq!(output, [0, 0, 0, 127, 127]);
    }

    /// Inputs that straddle every clamp boundary, cycled to fill `n` elements.
    fn boundary_input(n: usize) -> Vec<i32> {
        const VALUES: [i32; 12] = [
            i32::MIN,
            -70000,
            -1,
            0,
            63,
            64,
            127 << 6,
            (127 << 6) + 63,
            128 << 6,
            70000,
            1 << 20,
            i32::MAX,
        ];
        (0..n).map(|i| VALUES[i % VALUES.len()]).collect()
    }

    fn boundary_expected(input: &[i32]) -> Vec<u8> {
        input
            .iter()
            .map(|&x| (x >> 6).clamp(0, 127) as u8)
            .collect()
    }

    /// SIMD saturation on boundary values, including sizes with a scalar tail.
    #[test]
    fn test_boundary_values_all_sizes() {
        fn check<const N: usize>() {
            let layer = ClippedReLU::<N>::new();
            let input = boundary_input(N);
            let mut output = vec![0u8; N];
            layer.propagate(&input, &mut output);
            assert_eq!(output, boundary_expected(&input), "size {N}");
        }
        check::<16>();
        check::<32>();
        check::<37>();
        check::<48>();
        check::<64>();
    }

    /// The SSE4.1 and SSE2 kernels are never selected by `propagate` on an AVX2 host, so run
    /// them directly.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_sse_kernels_boundary_values() {
        fn check<const N: usize>() {
            let layer = ClippedReLU::<N>::new();
            let input = boundary_input(N);
            let expected = boundary_expected(&input);
            if is_x86_feature_detected!("sse4.1") {
                let mut output = vec![0u8; N];
                // SAFETY: SSE4.1 was confirmed at runtime; both slices hold N elements.
                unsafe { layer.propagate_sse41(&input, &mut output) };
                assert_eq!(output, expected, "sse4.1, size {N}");
            }
            if is_x86_feature_detected!("sse2") {
                let mut output = vec![0u8; N];
                // SAFETY: SSE2 was confirmed at runtime; both slices hold N elements.
                unsafe { layer.propagate_sse2(&input, &mut output) };
                assert_eq!(output, expected, "sse2, size {N}");
            }
        }
        check::<16>();
        check::<32>();
        check::<37>();
    }

    /// `propagate` (best SIMD kernel of the build: AVX2, SSE, NEON) against the plain scalar
    /// definition of the activation.
    #[test]
    fn test_simd_matches_scalar() {
        // Test sizes that exercise different code paths
        for size in [16, 32, 48, 64].iter() {
            match *size {
                16 => test_size::<16>(),
                32 => test_size::<32>(),
                48 => test_size::<48>(),
                64 => test_size::<64>(),
                _ => unreachable!(),
            }
        }

        fn test_size<const SIZE: usize>() {
            let layer = ClippedReLU::<SIZE>::new();

            // Generate test input with various values
            // Values spanning the unsaturated range (outputs 0..=127 after the shift), followed
            // by saturated ones.
            let mut input = vec![0i32; SIZE];
            for i in 0..SIZE {
                input[i] = ((i as i32) - SIZE as i32 / 2) * 97;
            }
            input[0] = i32::MIN;
            input[SIZE - 1] = i32::MAX;

            // Scalar definition (`propagate_scalar` does not exist on aarch64)
            let output_scalar = boundary_expected(&input);

            // Test SIMD version (will use best available)
            let mut output_simd = vec![0u8; SIZE];
            layer.propagate(&input, &mut output_simd);

            // Compare results
            assert_eq!(
                output_scalar, output_simd,
                "SIMD and scalar outputs differ for size {}",
                SIZE
            );
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_specific() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip if AVX2 not available
        }

        // Test both aligned (32) and unaligned (16) cases
        let layer32 = ClippedReLU::<32>::new();
        let layer16 = ClippedReLU::<16>::new();

        let mut input32 = [0i32; 32];
        let mut input16 = [0i32; 16];

        for i in 0..32 {
            input32[i] = (i as i32 - 16) * 64;
        }
        for i in 0..16 {
            input16[i] = (i as i32 - 8) * 64;
        }

        let mut output32 = [0u8; 32];
        let mut output16 = [0u8; 16];

        // SAFETY: AVX2 was confirmed at runtime by the `is_x86_feature_detected!` check above; the buffers match the layer sizes.
        unsafe {
            layer32.propagate_avx2(&input32, &mut output32);
            layer16.propagate_avx2(&input16, &mut output16);
        }

        // Verify results
        for i in 0..32 {
            let expected = ((input32[i] >> 6).clamp(0, 127)) as u8;
            assert_eq!(output32[i], expected, "Mismatch at index {} for size 32", i);
        }
        for i in 0..16 {
            let expected = ((input16[i] >> 6).clamp(0, 127)) as u8;
            assert_eq!(output16[i], expected, "Mismatch at index {} for size 16", i);
        }
    }
}
