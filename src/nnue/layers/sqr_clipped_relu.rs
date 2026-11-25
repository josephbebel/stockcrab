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

//! Squared Clipped ReLU activation layer for NNUE
//!
//! This layer applies squared ReLU activation followed by clipping and quantization.
//! The formula is: output = min(127, (input^2) >> (2*WeightScaleBitsLocal + 7))
//!
//! The squaring emphasizes larger activations while the right-shift provides
//! normalization. This needs to be accounted for in the trainer.
//!
//! `WEIGHT_SCALE_BITS_LOCAL` is a per-layer shift amount (C++ template parameter
//! `WeightScaleBitsLocal`, default `WEIGHT_SCALE_BITS`). SFNNv15 instantiates the first
//! squared activation with `WEIGHT_SCALE_BITS + 1`. Valid range is 5..=8. It is a `usize`
//! const generic to match how the network architecture module instantiates it.
//!
//! NOTE ON SIMD SHIFTS: the SIMD post-square shift is `SimdShiftAmount = 2*N + 7 - 16`,
//! an expression over the const generic `N`. Rust's legacy-const-generic intrinsic
//! arguments (e.g. `_mm_srli_epi16`, `vshrq_n_s16`) only accept a *standalone* const
//! parameter, not an arithmetic expression, and `generic_const_exprs` is not enabled.
//! We therefore use the runtime/variable shift intrinsics (`_mm_srl_epi16`, `vshlq_s16`)
//! which accept a computed shift, keeping correctness for any valid `N`.
//!
//! NOTE ON `propagate_pair`: upstream C++ has a fused `propagate_pair(input, squared, clipped)`
//! that produces the squared and the linear clipped activations together, sharing the input
//! loads and the initial signed 32-to-16-bit saturating narrowing. It is gated on
//! `USE_PAIR_ACTIVATIONS` (see `nnue::simd`), which is plain avx2 at compile time; every variant
//! is x86 vector-width specific. The port mirrors the AVX-512 and AVX2 variants and lets the
//! network architecture select it with `USE_PAIR_ACTIVATIONS`.

// Scalar, SSE, AVX2, AVX-512 and NEON implementations of each kernel coexist in
// this file and the target picks one at compile time (`scripts/check-all.sh` compiles every
// target to keep all of them building).

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::nnue::nnue_common::{IndexType, WEIGHT_SCALE_BITS};
use std::io::{Read, Write};

/// Squared Clipped ReLU activation
/// Input type: i32, Output type: u8
#[derive(Clone)]
pub struct SqrClippedReLU<
    const IN_DIMS: usize,
    const WEIGHT_SCALE_BITS_LOCAL: usize = { WEIGHT_SCALE_BITS as usize },
> {
    _phantom: core::marker::PhantomData<()>,
}

impl<const IN_DIMS: usize, const WEIGHT_SCALE_BITS_LOCAL: usize>
    SqrClippedReLU<IN_DIMS, WEIGHT_SCALE_BITS_LOCAL>
{
    // Input/output dimensions
    pub const INPUT_DIMENSIONS: IndexType = IN_DIMS as IndexType;
    pub const OUTPUT_DIMENSIONS: IndexType = IN_DIMS as IndexType;
    pub const PADDED_OUTPUT_DIMENSIONS: usize = IN_DIMS.div_ceil(32) * 32;

    /// Total scalar right-shift after squaring: 2 * WeightScaleBitsLocal + 7
    const TOTAL_SHIFT: usize = 2 * WEIGHT_SCALE_BITS_LOCAL + 7;
    /// SIMD post-square shift after the fixed >>16 from the high multiply:
    /// SimdShiftAmount = 2 * WeightScaleBitsLocal + 7 - 16
    const SIMD_SHIFT_AMOUNT: i32 = 2 * WEIGHT_SCALE_BITS_LOCAL as i32 + 7 - 16;

    /// Create a new squared clipped ReLU layer
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

    /// Read parameters from stream (no parameters for activation)
    pub fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> std::io::Result<()> {
        Ok(())
    }

    /// Write parameters to stream (no parameters for activation)
    pub fn write_parameters<W: Write>(&self, _writer: &mut W) -> std::io::Result<()> {
        Ok(())
    }

    /// Forward propagation from i32 input to u8 output
    pub fn propagate(&self, input: &[i32], output: &mut [u8]) {
        // Valid range for WeightScaleBitsLocal (matches C++ static_assert 5..=8).
        const {
            assert!(
                WEIGHT_SCALE_BITS_LOCAL >= 5 && WEIGHT_SCALE_BITS_LOCAL <= 8,
                "SqrClippedReLU only supports WeightScaleBitsLocal between 5 and 8"
            )
        };
        // The SIMD kernels below read/write exactly the first IN_DIMS elements, unchecked.
        assert!(input.len() >= IN_DIMS && output.len() >= IN_DIMS);

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is part of the aarch64 baseline; the kernel accesses only the first
            // `IN_DIMS` elements of both slices (unaligned NEON loads/stores; lengths asserted at the
            // top of `propagate`).
            unsafe {
                self.propagate_neon(input, output);
            }
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                // SAFETY: SSE2 was just verified at runtime (and is baseline on x86_64); the kernel
                // accesses only the first `IN_DIMS` elements of both slices with unaligned
                // loads/stores (lengths asserted at the top of `propagate`).
                unsafe {
                    self.propagate_sse2(input, output);
                }
                return;
            }
        }

        #[cfg(target_arch = "x86")]
        {
            if is_x86_feature_detected!("sse2") {
                // SAFETY: SSE2 was just verified at runtime; the kernel accesses only the first
                // `IN_DIMS` elements of both slices with unaligned loads/stores (lengths asserted
                // at the top of `propagate`).
                unsafe {
                    self.propagate_sse2(input, output);
                }
                return;
            }
        }

        // Fallback to scalar implementation (unreachable on aarch64, where NEON returns above)
        #[allow(unreachable_code)]
        self.propagate_scalar(input, output);
    }

    /// Produce the squared and linear clipped activations together, sharing the input loads and
    /// the initial signed 32-to-16-bit saturating narrowing. The x86 kernels deliberately keep the
    /// lane order created by the final pack, and `AffineTransform::get_weight_index_scrambled`
    /// pre-permutes the next layer's weights at load time to match, so no shuffle is needed
    /// (Stockfish's `USE_PAIR_ACTIVATIONS` path).
    pub fn propagate_pair(&self, input: &[i32], squared: &mut [u8], clipped: &mut [u8]) {
        assert!(
            IN_DIMS.is_multiple_of(32),
            "propagate_pair requires dimensions divisible by 32"
        );
        assert!(input.len() >= IN_DIMS && squared.len() >= IN_DIMS && clipped.len() >= IN_DIMS);

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F/BW are enabled at compile time by the enclosing
            // `cfg(target_feature)` and re-checked at runtime; the asserts above give `IN_DIMS % 32
            // == 0` and all three slices at least `IN_DIMS` long.
            unsafe { self.propagate_pair_avx512(input, squared, clipped) };
            return;
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: AVX2 is enabled at compile time by the enclosing `cfg(target_feature =
            // "avx2")`; the asserts above give `IN_DIMS % 32 == 0` and all three slices at least
            // `IN_DIMS` long, covering every access.
            unsafe { self.propagate_pair_avx2(input, squared, clipped) };
            return;
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            // This is only reached on builds without paired x86 SIMD support.
            // It keeps the API well-defined for tests and non-x86 callers.
            self.propagate_scalar(input, squared);
            for i in 0..IN_DIMS {
                clipped[i] = ((input[i] >> WEIGHT_SCALE_BITS_LOCAL).clamp(0, 127)) as u8;
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn propagate_pair_avx512(&self, input: &[i32], squared: &mut [u8], clipped: &mut [u8]) {
        use core::arch::x86_64::*;

        let zero = _mm512_setzero_si512();
        let square_shift = _mm_cvtsi32_si128(Self::SIMD_SHIFT_AMOUNT);
        let clipped_shift = _mm_cvtsi32_si128(WEIGHT_SCALE_BITS_LOCAL as i32);
        for i in 0..IN_DIMS / 32 {
            let input_ptr = input.as_ptr().add(i * 32);
            let words = _mm512_packs_epi32(
                _mm512_loadu_si512(input_ptr as *const __m512i),
                _mm512_loadu_si512(input_ptr.add(16) as *const __m512i),
            );
            let square_words = _mm512_srl_epi16(_mm512_mulhi_epi16(words, words), square_shift);
            _mm256_storeu_si256(
                squared.as_mut_ptr().add(i * 32) as *mut __m256i,
                _mm512_cvtsepi16_epi8(square_words),
            );
            let clipped_words = _mm512_srl_epi16(_mm512_max_epi16(words, zero), clipped_shift);
            _mm256_storeu_si256(
                clipped.as_mut_ptr().add(i * 32) as *mut __m256i,
                _mm512_cvtsepi16_epi8(clipped_words),
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_pair_avx2(&self, input: &[i32], squared: &mut [u8], clipped: &mut [u8]) {
        use core::arch::x86_64::*;

        let zero = _mm256_setzero_si256();
        let square_shift = _mm_cvtsi32_si128(Self::SIMD_SHIFT_AMOUNT);
        let clipped_shift = _mm_cvtsi32_si128(WEIGHT_SCALE_BITS_LOCAL as i32);
        for i in 0..IN_DIMS / 32 {
            let input_ptr = input.as_ptr().add(i * 32) as *const __m256i;
            let words0 = _mm256_packs_epi32(
                _mm256_loadu_si256(input_ptr),
                _mm256_loadu_si256(input_ptr.add(1)),
            );
            let words1 = _mm256_packs_epi32(
                _mm256_loadu_si256(input_ptr.add(2)),
                _mm256_loadu_si256(input_ptr.add(3)),
            );
            let square0 = _mm256_srl_epi16(_mm256_mulhi_epi16(words0, words0), square_shift);
            let square1 = _mm256_srl_epi16(_mm256_mulhi_epi16(words1, words1), square_shift);
            _mm256_storeu_si256(
                squared.as_mut_ptr().add(i * 32) as *mut __m256i,
                _mm256_packs_epi16(square0, square1),
            );
            let clipped0 = _mm256_srl_epi16(_mm256_max_epi16(words0, zero), clipped_shift);
            let clipped1 = _mm256_srl_epi16(_mm256_max_epi16(words1, zero), clipped_shift);
            _mm256_storeu_si256(
                clipped.as_mut_ptr().add(i * 32) as *mut __m256i,
                _mm256_packs_epi16(clipped0, clipped1),
            );
        }
    }

    /// NEON optimized implementation for ARM64
    #[cfg(target_arch = "aarch64")]
    unsafe fn propagate_neon(&self, input: &[i32], output: &mut [u8]) {
        use core::arch::aarch64::*;

        // Process 16 elements at a time (matching SSE2 which also does 16)
        let num_chunks = IN_DIMS / 16;

        // SQDMULH keeps the doubled product's high half (>>15), so the
        // remaining shift is one larger than the x86 high-multiply shift.
        let neg_shift = vdupq_n_s16(-((Self::SIMD_SHIFT_AMOUNT + 1) as i16));

        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..num_chunks {
            let base = i * 16;

            // Load 4 groups of 4 int32 values (16 values total)
            let in0 = vld1q_s32(in_ptr.add(base + 0));
            let in1 = vld1q_s32(in_ptr.add(base + 4));
            let in2 = vld1q_s32(in_ptr.add(base + 8));
            let in3 = vld1q_s32(in_ptr.add(base + 12));

            // Narrow int32 to int16 with saturation (vqmovn)
            // Each vqmovn_s32 takes 4 int32 and produces 4 int16
            let words0_lo = vqmovn_s32(in0); // 4 x i16
            let words0_hi = vqmovn_s32(in1); // 4 x i16
            let words1_lo = vqmovn_s32(in2); // 4 x i16
            let words1_hi = vqmovn_s32(in3); // 4 x i16

            // Combine into 8-element vectors
            let words0 = vcombine_s16(words0_lo, words0_hi); // 8 x i16
            let words1 = vcombine_s16(words1_lo, words1_hi); // 8 x i16

            let shifted0 = vshlq_s16(vqdmulhq_s16(words0, words0), neg_shift);
            let shifted1 = vshlq_s16(vqdmulhq_s16(words1, words1), neg_shift);

            // Narrow to i8 with signed saturation - values are positive, so [0, 127]
            // Using signed saturation to match SSE2's _mm_packs_epi16 behavior
            let result_lo = vqmovn_s16(shifted0); // saturate i16 -> i8 (clamped to 127)
            let result_hi = vqmovn_s16(shifted1);

            // Store 16 bytes (reinterpret i8 as u8)
            vst1_s8(out_ptr.add(base) as *mut i8, result_lo);
            vst1_s8(out_ptr.add(base + 8) as *mut i8, result_hi);
        }

        // Process remaining elements with scalar code
        let start = num_chunks * 16;
        for j in start..IN_DIMS {
            let val = input[j] as i64;
            let squared = val * val;
            let result = (squared >> Self::TOTAL_SHIFT).min(127);
            output[j] = result as u8;
        }
    }

    /// SSE2 optimized implementation
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn propagate_sse2(&self, input: &[i32], output: &mut [u8]) {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let num_chunks = IN_DIMS / 16;

        // Variable right-shift count for the post-square SimdShiftAmount (see module note).
        let shift_count = _mm_cvtsi32_si128(Self::SIMD_SHIFT_AMOUNT);

        let in_ptr = input.as_ptr() as *const __m128i;
        let out_ptr = output.as_mut_ptr() as *mut __m128i;

        for i in 0..num_chunks {
            // Load 4 groups of 4 int32 values (16 values total)
            let in0 = _mm_loadu_si128(in_ptr.add(i * 4 + 0));
            let in1 = _mm_loadu_si128(in_ptr.add(i * 4 + 1));
            let in2 = _mm_loadu_si128(in_ptr.add(i * 4 + 2));
            let in3 = _mm_loadu_si128(in_ptr.add(i * 4 + 3));

            // Pack int32 to int16 (with saturation)
            let words0 = _mm_packs_epi32(in0, in1);
            let words1 = _mm_packs_epi32(in2, in3);

            // Square the values using _mm_mulhi_epi16
            // This multiplies and keeps the high 16 bits of the 32-bit result (i.e. >>16)
            let squared0 = _mm_mulhi_epi16(words0, words0);
            let squared1 = _mm_mulhi_epi16(words1, words1);

            // Remaining SimdShiftAmount right-shift (2*N + 7 - 16), variable count.
            let shifted0 = _mm_srl_epi16(squared0, shift_count);
            let shifted1 = _mm_srl_epi16(squared1, shift_count);

            // Pack int16 to int8 (with saturation to [0, 127])
            let result = _mm_packs_epi16(shifted0, shifted1);

            // Store the result
            _mm_storeu_si128(out_ptr.add(i), result);
        }

        // Process remaining elements with scalar code
        let start = num_chunks * 16;
        for i in start..IN_DIMS {
            let val = input[i] as i64;
            let squared = val * val;
            let result = (squared >> Self::TOTAL_SHIFT).min(127);
            output[i] = result as u8;
        }
    }

    /// Scalar fallback implementation
    fn propagate_scalar(&self, input: &[i32], output: &mut [u8]) {
        for i in 0..IN_DIMS {
            let val = input[i] as i64;
            let squared = val * val;
            // Really should be /127 but we need to make it fast so we right-shift
            // by an extra 7 bits instead. Needs to be accounted for in the trainer.
            let result = (squared >> Self::TOTAL_SHIFT).min(127);
            output[i] = result as u8;
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
    for SqrClippedReLU<IN_DIMS, WEIGHT_SCALE_BITS_LOCAL>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqdmulh_matches_full_square_for_every_i16_and_scale() {
        for scale in 5..=8 {
            for value in i16::MIN as i64..=i16::MAX as i64 {
                let square = value * value;
                // SQDMULH saturates the high half of the doubled signed product.
                let high = ((square * 2) >> 16).min(i16::MAX as i64);
                let actual = (high >> (2 * scale + 7 - 16 + 1)).min(127);
                let expected = (square >> (2 * scale + 7)).min(127);
                assert_eq!(actual, expected, "value={value}, scale={scale}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_for_every_i16_and_scale() {
        fn check<const SCALE: usize>() {
            let layer = SqrClippedReLU::<16, SCALE>::new();
            for base in (i16::MIN as i32..=i16::MAX as i32).step_by(16) {
                let input: [i32; 16] = core::array::from_fn(|i| base + i as i32);
                let mut scalar = [0; 16];
                let mut neon = [0; 16];
                layer.propagate_scalar(&input, &mut scalar);
                // SAFETY: NEON is baseline; both arrays contain all 16 accessed elements.
                unsafe { layer.propagate_neon(&input, &mut neon) };
                assert_eq!(scalar, neon, "scale={SCALE}, base={base}");
            }
        }
        check::<5>();
        check::<6>();
        check::<7>();
        check::<8>();
    }

    #[test]
    fn test_hash_value() {
        let hash1 = SqrClippedReLU::<32>::get_hash_value(0x12345678);
        let hash2 = SqrClippedReLU::<32>::get_hash_value(0x87654321);
        assert_ne!(hash1, hash2);
        assert_eq!(hash1, 0x538D24C7u32.wrapping_add(0x12345678));
    }

    #[test]
    fn test_propagate_scalar() {
        let layer = SqrClippedReLU::<4>::new();

        // Test with various input values
        let input = [
            64 * 64,  // 4096 -> squared = 16777216, >> 19 = 32
            -64 * 64, // -4096 -> squared = 16777216, >> 19 = 32
            100000,   // Large value -> should clamp to 127
            0,        // Zero -> 0
        ];
        let mut output = [0u8; 4];
        layer.propagate_scalar(&input, &mut output);

        // Verify results
        assert_eq!(output[0], 32); // (4096^2) >> 19 = 32
        assert_eq!(output[1], 32); // (-4096^2) >> 19 = 32
        assert_eq!(output[2], 127); // Clamped to 127
        assert_eq!(output[3], 0); // 0^2 = 0
    }

    /// Verify a non-default WeightScaleBitsLocal (WEIGHT_SCALE_BITS + 1 == 7).
    /// Total shift becomes 2*7 + 7 = 21.
    #[test]
    fn test_propagate_scale_bits_plus_one() {
        let layer = SqrClippedReLU::<4, 7>::new();
        let input = [
            4096, // 4096^2 = 2^24, >> 21 = 2^3 = 8
            8192, // 8192^2 = 2^26, >> 21 = 2^5 = 32
            0,
            1 << 20, // huge -> clamp 127
        ];
        let mut output = [0u8; 4];
        layer.propagate(&input, &mut output);
        assert_eq!(output[0], 8);
        assert_eq!(output[1], 32);
        assert_eq!(output[2], 0);
        assert_eq!(output[3], 127);
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(SqrClippedReLU::<15>::INPUT_DIMENSIONS, 15);
        assert_eq!(SqrClippedReLU::<15>::OUTPUT_DIMENSIONS, 15);
        assert_eq!(SqrClippedReLU::<15>::PADDED_OUTPUT_DIMENSIONS, 32);
        assert_eq!(SqrClippedReLU::<32>::PADDED_OUTPUT_DIMENSIONS, 32);
        assert_eq!(SqrClippedReLU::<33>::PADDED_OUTPUT_DIMENSIONS, 64);
    }

    #[test]
    fn test_weight_scale_bits() {
        // Ensure the formula matches C++ version
        let layer = SqrClippedReLU::<1>::new();
        let input = [8192]; // 2^13
        let mut output = [0u8; 1];
        layer.propagate(&input, &mut output);

        // 8192^2 = 67108864 = 2^26
        // >> 19 = 2^7 = 128, clamped to 127
        assert_eq!(output[0], 127);

        let input = [4096]; // 2^12
        layer.propagate(&input, &mut output);

        // 4096^2 = 16777216 = 2^24
        // >> 19 = 2^5 = 32
        assert_eq!(output[0], 32);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_sse2_matches_scalar() {
        if !is_x86_feature_detected!("sse2") {
            return; // Skip test if SSE2 not available
        }

        const SIZE: usize = 32;
        let layer = SqrClippedReLU::<SIZE>::new();

        // Generate test input
        let mut input = [0i32; SIZE];
        for i in 0..SIZE {
            input[i] = ((i as i32) - 16) * 1000; // Range around zero
        }

        // Test scalar version
        let mut output_scalar = [0u8; SIZE];
        layer.propagate_scalar(&input, &mut output_scalar);

        // Test SSE2 version
        let mut output_sse2 = [0u8; SIZE];
        // SAFETY: SSE2 was confirmed at runtime by the `is_x86_feature_detected!` check above; the buffers match the layer size.
        unsafe {
            layer.propagate_sse2(&input, &mut output_sse2);
        }

        // Compare results
        assert_eq!(output_scalar, output_sse2);
    }

    /// The SSE2 kernel must also match scalar for a non-default WeightScaleBitsLocal, at a size
    /// that runs the vector loop (not just the scalar tail).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_sse2_matches_scalar_scale7() {
        if !is_x86_feature_detected!("sse2") {
            return; // Skip test if SSE2 not available
        }

        const SIZE: usize = 32;
        let layer = SqrClippedReLU::<SIZE, 7>::new();

        // Total shift is 2*7 + 7 = 21; these inputs span the unsaturated outputs (0..=127).
        let mut input = [0i32; SIZE];
        for i in 0..SIZE {
            input[i] = ((i as i32) - 16) * 1000;
        }

        let mut output_scalar = [0u8; SIZE];
        layer.propagate_scalar(&input, &mut output_scalar);

        let mut output_sse2 = [0u8; SIZE];
        // SAFETY: SSE2 was confirmed at runtime by the `is_x86_feature_detected!` check above; the buffers match the layer size.
        unsafe {
            layer.propagate_sse2(&input, &mut output_sse2);
        }

        assert_eq!(
            output_scalar, output_sse2,
            "SSE2(scale=7) differs from scalar"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar() {
        const SIZE: usize = 32;
        let layer = SqrClippedReLU::<SIZE>::new();

        // Generate test input
        let mut input = [0i32; SIZE];
        for i in 0..SIZE {
            input[i] = ((i as i32) - 16) * 1000; // Range around zero
        }

        // Test scalar version
        let mut output_scalar = [0u8; SIZE];
        layer.propagate_scalar(&input, &mut output_scalar);

        // Test NEON version
        let mut output_neon = [0u8; SIZE];
        // SAFETY: NEON is part of the aarch64 baseline; `input` and `output` both have SIZE (32) elements,
        // a multiple of 16, so every 16-wide load/store stays in bounds.
        unsafe {
            layer.propagate_neon(&input, &mut output_neon);
        }

        // Compare results
        assert_eq!(
            output_scalar, output_neon,
            "NEON output differs from scalar"
        );
    }

    /// NEON path must match scalar for a non-default WeightScaleBitsLocal too.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_scale7() {
        const SIZE: usize = 32;
        let layer = SqrClippedReLU::<SIZE, 7>::new();

        let mut input = [0i32; SIZE];
        for i in 0..SIZE {
            input[i] = ((i as i32) - 16) * 1000;
        }

        let mut output_scalar = [0u8; SIZE];
        layer.propagate_scalar(&input, &mut output_scalar);

        let mut output_neon = [0u8; SIZE];
        // SAFETY: NEON is part of the aarch64 baseline; `input` and `output` both have SIZE (32) elements,
        // a multiple of 16, so every 16-wide load/store stays in bounds.
        unsafe {
            layer.propagate_neon(&input, &mut output_neon);
        }

        assert_eq!(
            output_scalar, output_neon,
            "NEON(scale=7) differs from scalar"
        );
    }
}
