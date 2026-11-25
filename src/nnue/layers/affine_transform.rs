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

//! Affine transform layer for NNUE
//!
//! This layer implements a fully connected layer (affine transformation).
//! Per-ISA kernels cover a single output and output widths that are multiples of the SIMD lane
//! count; other widths use the scalar reference path.

// Scalar, SSE, AVX2, AVX-512 and NEON implementations of each kernel coexist in
// this file and the target picks one at compile time (`scripts/check-all.sh` compiles every
// target to keep all of them building).

// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::nnue::nnue_common::{AlignedVec, IndexType, MAX_SIMD_WIDTH};
use std::io::{Read, Write};

// Layer-specific type overrides to match C++ implementation
// C++ AffineTransform uses: BiasType = OutputType (i32), WeightType = i8
type BiasType = i32;
type WeightType = i8;

/// Check if sequential optimization should be enabled
#[cfg(any(target_feature = "ssse3", target_feature = "neon"))]
const ENABLE_SEQ_OPT: bool = true;
#[cfg(not(any(target_feature = "ssse3", target_feature = "neon")))]
const ENABLE_SEQ_OPT: bool = false;

/// Fallback implementation for architectures without SSSE3/NEON
fn affine_transform_non_ssse3<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>(
    output: &mut [i32],
    weights: &[WeightType],
    biases: &[BiasType],
    input: &[u8],
    padded_input_dimensions: usize,
) {
    // Copy biases to output (biases are already i32)
    output[..OUTPUT_DIMENSIONS].copy_from_slice(&biases[..OUTPUT_DIMENSIONS]);

    // Traverse weights in transpose order to take advantage of input sparsity
    for i in 0..INPUT_DIMENSIONS {
        if input[i] != 0 {
            let w = &weights[i..];
            let in_val = input[i] as i32;
            for j in 0..OUTPUT_DIMENSIONS {
                output[j] += (w[j * padded_input_dimensions] as i32) * in_val;
            }
        }
    }
}

/// Affine transform layer: output = weights * input + bias
/// Input type: u8, Output type: i32
#[derive(Clone)]
pub struct AffineTransform<const IN_DIMS: usize, const OUT_DIMS: usize> {
    weights: AlignedVec<WeightType>,
    biases: AlignedVec<BiasType>,
}

impl<const IN_DIMS: usize, const OUT_DIMS: usize> AffineTransform<IN_DIMS, OUT_DIMS> {
    // Dimensions
    pub const INPUT_DIMENSIONS: IndexType = IN_DIMS as IndexType;
    pub const OUTPUT_DIMENSIONS: IndexType = OUT_DIMS as IndexType;

    const PADDED_INPUT_DIMENSIONS: usize = IN_DIMS.div_ceil(MAX_SIMD_WIDTH) * MAX_SIMD_WIDTH;

    /// Create a new affine transform layer
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

    /// Maps a serialized weight index to its position in the layout that matches the previous
    /// layer's (pair-)activation output order, so no shuffle is needed at runtime.
    const fn get_weight_index_scrambled(i: usize) -> usize {
        // PaddedInputDimensions is a multiple of 4, so i % PADDED keeps each 4-byte block intact.
        let input_index = i % Self::PADDED_INPUT_DIMENSIONS;

        // AVX2+ builds use `SqrClippedReLU::propagate_pair`, whose packs interleave
        // 128-bit lanes.  Reorder the serialized input columns at load time so the
        // dense dot product sees the logical `[squared | clipped]` order without a
        // shuffle. This is Stockfish's current `USE_PAIR_ACTIVATIONS` mapping.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let input_index = {
            let block = input_index / 32;
            let chunk = (input_index % 32) / 4;
            #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
            let permuted = block * 32 + ((chunk % 4) * 2 + chunk / 4) * 4 + input_index % 4;
            #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
            let permuted = block * 32 + ((chunk % 2) * 4 + chunk / 2) * 4 + input_index % 4;
            permuted
        };
        input_index / 4 * OUT_DIMS * 4 + i / Self::PADDED_INPUT_DIMENSIONS * 4 + input_index % 4
    }

    /// Get weight index
    const fn get_weight_index(i: usize) -> usize {
        if ENABLE_SEQ_OPT {
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
        // Write biases (i32, 4 bytes each)
        for &bias in &self.biases {
            writer.write_all(&bias.to_le_bytes())?;
        }

        // Write weights with scrambling (i8, 1 byte each)
        for i in 0..(OUT_DIMS * Self::PADDED_INPUT_DIMENSIONS) {
            let weight = self.weights[Self::get_weight_index(i)];
            writer.write_all(&weight.to_le_bytes())?;
        }

        Ok(())
    }

    /// Forward propagation (u8 input -> i32 output)
    // AArch64: `NetworkArchitecture` has two propagate bodies there (the plain one and the
    // bitset one), and with two callers LLVM stops inlining this dense layer into them.
    #[cfg_attr(target_arch = "aarch64", inline(always))]
    pub fn propagate(&self, input: &[u8], output: &mut [i32]) {
        // The SIMD kernels read the padded input (as i32 chunks, hence 4-byte alignment) with
        // unchecked loads and write `OUT_DIMS` outputs.
        assert!(input.len() >= Self::PADDED_INPUT_DIMENSIONS && output.len() >= OUT_DIMS);
        assert!(
            (input.as_ptr() as usize).is_multiple_of(4),
            "input must be 4-byte aligned"
        );

        if ENABLE_SEQ_OPT && OUT_DIMS > 1 {
            // Optimized implementation for multiple outputs
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ))]
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && OUT_DIMS.is_multiple_of(16)
            {
                // SAFETY: AVX-512F/BW are enabled by the enclosing `cfg(target_feature)` and
                // re-checked at runtime; `OUT_DIMS % 16 == 0` is checked here. Other preconditions
                // (input 4-byte aligned and >= `num_chunks * 4 <= PADDED_INPUT_DIMENSIONS` bytes,
                // output >= `OUT_DIMS`) are asserted at the top of `propagate`.
                unsafe { self.propagate_simd_multi_avx512(input, output) };
                return;
            }

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            if OUT_DIMS.is_multiple_of(8) {
                self.propagate_simd_multi(input, output);
                return;
            }

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "ssse3",
                not(target_feature = "avx2")
            ))]
            if OUT_DIMS.is_multiple_of(4) {
                self.propagate_simd_multi(input, output);
                return;
            }

            #[cfg(target_arch = "aarch64")]
            if OUT_DIMS.is_multiple_of(4) {
                // SAFETY: NEON is baseline. The kernel enables `dotprod` only
                // under cfg(stockcrab_dotprod): either the target has it or the
                // user asserted dot-product hardware with the `neon-dotprod`
                // Cargo feature, like the other NNUE dot-product kernels.
                // Slice preconditions are checked above.
                unsafe { self.propagate_simd_multi(input, output) };
                return;
            }

            // SIMD output registers must cover the complete output layer.
            // Small test layers (and any future odd-width layer) use the
            // reference path instead of silently dropping the bias/output
            // lanes in a partial register.
            // Unreachable where a `cfg`'d branch above always returns (NEON).
            #[allow(unreachable_code)]
            output[..OUT_DIMS].copy_from_slice(&self.biases[..OUT_DIMS]);
            for column in 0..Self::PADDED_INPUT_DIMENSIONS {
                for row in 0..OUT_DIMS {
                    // Both input and weights use the physical activation order;
                    // only the column-major chunk layout needs decoding here.
                    let index = column / 4 * OUT_DIMS * 4 + row * 4 + column % 4;
                    output[row] += i32::from(input[column]) * i32::from(self.weights[index]);
                }
            }
        } else if ENABLE_SEQ_OPT && OUT_DIMS == 1 {
            // Optimized implementation for single output
            #[cfg(target_arch = "aarch64")]
            // SAFETY: slices are checked above; cfg(stockcrab_dotprod) carries
            // the same hardware requirement as for the multi-output kernel.
            unsafe {
                self.propagate_simd_single(input, output)
            };
            #[cfg(not(target_arch = "aarch64"))]
            self.propagate_simd_single(input, output);
        } else {
            // Fallback implementation
            affine_transform_non_ssse3::<IN_DIMS, OUT_DIMS>(
                output,
                &self.weights,
                &self.biases,
                input,
                Self::PADDED_INPUT_DIMENSIONS,
            );
        }
    }

    /// 512-bit counterpart of Stockfish's `USE_AVX512` multi-output path.
    /// The dense hidden layer has 32 outputs, so two vector accumulators cover
    /// the whole layer on the current SF18 network.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn propagate_simd_multi_avx512(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;

        const OUTPUT_SIMD_WIDTH: usize = 16;
        let num_chunks = IN_DIMS.div_ceil(8) * 2;
        let num_regs = OUT_DIMS / OUTPUT_SIMD_WIDTH;
        debug_assert_eq!(OUT_DIMS % OUTPUT_SIMD_WIDTH, 0);
        debug_assert_eq!((input.as_ptr() as usize) & 3, 0);
        debug_assert!(input.len() >= num_chunks * 4);

        let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
        const MAX_ACCUMS: usize = 16;
        #[cfg(target_feature = "avx512vnni")]
        const NUM_REGS: usize = 2 * MAX_ACCUMS;
        #[cfg(not(target_feature = "avx512vnni"))]
        const NUM_REGS: usize = MAX_ACCUMS;

        let mut acc = [_mm512_setzero_si512(); NUM_REGS];
        for k in 0..num_regs {
            acc[k] = _mm512_loadu_si512(
                self.biases.as_ptr().add(k * OUTPUT_SIMD_WIDTH) as *const __m512i
            );
        }

        #[cfg(target_feature = "avx512vnni")]
        let mut i = 0;
        #[cfg(target_feature = "avx512vnni")]
        {
            // Keep two independent dependency chains, matching Stockfish's
            // USE_VNNI path.  The second chain is folded after the input loop.
            while i + 1 < num_chunks {
                let in0 = _mm512_set1_epi32(*input32.get_unchecked(i));
                let in1 = _mm512_set1_epi32(*input32.get_unchecked(i + 1));
                let col0 = self.weights.as_ptr().add(i * OUT_DIMS * 4) as *const __m512i;
                let col1 = self.weights.as_ptr().add((i + 1) * OUT_DIMS * 4) as *const __m512i;
                for k in 0..num_regs {
                    let w0 = _mm512_loadu_si512(col0.add(k));
                    let w1 = _mm512_loadu_si512(col1.add(k));
                    crate::nnue::simd::avx512::m512_add_dpbusd_epi32(&mut acc[k], in0, w0);
                    crate::nnue::simd::avx512::m512_add_dpbusd_epi32(
                        &mut acc[k + num_regs],
                        in1,
                        w1,
                    );
                }
                i += 2;
            }

            for k in 0..num_regs {
                acc[k] = _mm512_add_epi32(acc[k], acc[k + num_regs]);
            }
        }

        #[cfg(not(target_feature = "avx512vnni"))]
        {
            for i in 0..num_chunks {
                let in_vec = _mm512_set1_epi32(*input32.get_unchecked(i));
                let column = self.weights.as_ptr().add(i * OUT_DIMS * 4) as *const __m512i;
                for k in 0..num_regs {
                    let weights = _mm512_loadu_si512(column.add(k));
                    crate::nnue::simd::avx512::m512_add_dpbusd_epi32(&mut acc[k], in_vec, weights);
                }
            }
        }

        #[cfg(target_feature = "avx512vnni")]
        if i < num_chunks {
            let in_vec = _mm512_set1_epi32(*input32.get_unchecked(i));
            let column = self.weights.as_ptr().add(i * OUT_DIMS * 4) as *const __m512i;
            for k in 0..num_regs {
                let weights = _mm512_loadu_si512(column.add(k));
                crate::nnue::simd::avx512::m512_add_dpbusd_epi32(&mut acc[k], in_vec, weights);
            }
        }

        for k in 0..(OUT_DIMS / OUTPUT_SIMD_WIDTH) {
            _mm512_storeu_si512(
                output.as_mut_ptr().add(k * OUTPUT_SIMD_WIDTH) as *mut __m512i,
                acc[k],
            );
        }
    }

    /// SIMD optimized propagation for multiple outputs using AVX2
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn propagate_simd_multi(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;

        const OUTPUT_SIMD_WIDTH: usize = 8; // AVX2: 256 bits / 32 bits = 8 i32s
        let num_chunks = IN_DIMS.div_ceil(8) * 2; // ceil_to_multiple(IN_DIMS, 8) / 4
        let num_regs = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        debug_assert_eq!(
            (input.as_ptr() as usize) & 3,
            0,
            "NNUE affine input must be 4-byte aligned"
        );
        debug_assert!(input.len() >= num_chunks * 4);

        // SAFETY: AVX2 is enabled by the enclosing `cfg(target_feature)`. The weights hold
        // `OUT_DIMS * PADDED_INPUT_DIMENSIONS` bytes and `num_chunks * 4 <=
        // PADDED_INPUT_DIMENSIONS`, so every `i * OUT_DIMS * 4 + k * 32` access is in bounds;
        // biases hold `OUT_DIMS` i32 (`OUT_DIMS % 8 == 0` per dispatch). The input slice must be
        // 4-byte aligned with >= `num_chunks * 4` bytes and `output` >= `OUT_DIMS` (asserted in `propagate`).
        unsafe {
            let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
            let biasvec = self.biases.as_ptr() as *const __m256i;

            const MAX_ACCUMS: usize = 16;
            #[cfg(target_feature = "avxvnni")]
            const NUM_REGS: usize = 2 * MAX_ACCUMS;
            #[cfg(not(target_feature = "avxvnni"))]
            const NUM_REGS: usize = MAX_ACCUMS;

            // Initialize accumulators with biases.  AVX-VNNI uses a second
            // independent chain to hide the longer dot-product dependency.
            let mut acc = [_mm256_setzero_si256(); NUM_REGS];
            for k in 0..num_regs {
                acc[k] = _mm256_loadu_si256(biasvec.add(k));
            }

            #[cfg(target_feature = "avxvnni")]
            let mut i = 0;
            #[cfg(target_feature = "avxvnni")]
            {
                while i + 1 < num_chunks {
                    let in0 = _mm256_set1_epi32(input32[i]);
                    let in1 = _mm256_set1_epi32(input32[i + 1]);
                    let col0 = (self.weights.as_ptr() as *const __m256i).add(i * OUT_DIMS * 4 / 32);
                    let col1 =
                        (self.weights.as_ptr() as *const __m256i).add((i + 1) * OUT_DIMS * 4 / 32);

                    for k in 0..num_regs {
                        let w0 = _mm256_loadu_si256(col0.add(k));
                        let w1 = _mm256_loadu_si256(col1.add(k));
                        crate::nnue::simd::avx2::m256_add_dpbusd_epi32(&mut acc[k], in0, w0);
                        crate::nnue::simd::avx2::m256_add_dpbusd_epi32(
                            &mut acc[k + num_regs],
                            in1,
                            w1,
                        );
                    }
                    i += 2;
                }

                for k in 0..num_regs {
                    acc[k] = _mm256_add_epi32(acc[k], acc[k + num_regs]);
                }
            }

            #[cfg(not(target_feature = "avxvnni"))]
            {
                for i in 0..num_chunks {
                    let in0 = _mm256_set1_epi32(input32[i]);
                    let col0 = (self.weights.as_ptr() as *const __m256i).add(i * OUT_DIMS * 4 / 32);
                    for k in 0..num_regs {
                        let weights = _mm256_loadu_si256(col0.add(k));
                        crate::nnue::simd::avx2::m256_add_dpbusd_epi32(&mut acc[k], in0, weights);
                    }
                }
            }

            #[cfg(target_feature = "avxvnni")]
            if i < num_chunks {
                let in0 = _mm256_set1_epi32(input32[i]);
                let col0 = (self.weights.as_ptr() as *const __m256i).add(i * OUT_DIMS * 4 / 32);
                for k in 0..num_regs {
                    let weights = _mm256_loadu_si256(col0.add(k));
                    crate::nnue::simd::avx2::m256_add_dpbusd_epi32(&mut acc[k], in0, weights);
                }
            }

            // Store results
            let outptr = output.as_mut_ptr() as *mut __m256i;
            for k in 0..num_regs {
                _mm256_storeu_si256(outptr.add(k), acc[k]);
            }
        }
    }

    /// SIMD optimized propagation for multiple outputs using SSSE3
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "ssse3",
        not(target_feature = "avx2")
    ))]
    fn propagate_simd_multi(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;

        const OUTPUT_SIMD_WIDTH: usize = 4; // SSE: 128 bits / 32 bits = 4 i32s
        let num_chunks = IN_DIMS.div_ceil(8) * 2;
        let num_regs = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        debug_assert_eq!(
            (input.as_ptr() as usize) & 3,
            0,
            "NNUE affine input must be 4-byte aligned"
        );
        debug_assert!(input.len() >= num_chunks * 4);

        // SAFETY: SSSE3 is enabled by the enclosing `cfg(target_feature)`. Weight and bias accesses
        // are in bounds exactly as in the AVX2 variant (16-byte steps, `OUT_DIMS % 4 == 0` per
        // dispatch); input must be 4-byte aligned with >= `num_chunks * 4` bytes and `output` >=
        // `OUT_DIMS` (asserted in `propagate`).
        unsafe {
            let input32 = core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks);
            let biasvec = self.biases.as_ptr() as *const __m128i;

            // Initialize accumulators with biases
            let mut acc = [_mm_setzero_si128(); 16];
            for k in 0..num_regs {
                acc[k] = _mm_loadu_si128(biasvec.add(k));
            }

            // Process input in chunks of 4 bytes
            for i in 0..num_chunks {
                let in0 = _mm_set1_epi32(input32[i]);
                let col0 = (self.weights.as_ptr() as *const __m128i).add(i * OUT_DIMS * 4 / 16);

                for k in 0..num_regs {
                    let weights = _mm_loadu_si128(col0.add(k));
                    crate::nnue::simd::sse2::m128_add_dpbusd_epi32(&mut acc[k], in0, weights);
                }
            }

            // Store results
            let outptr = output.as_mut_ptr() as *mut __m128i;
            for k in 0..num_regs {
                _mm_storeu_si128(outptr.add(k), acc[k]);
            }
        }
    }

    /// SIMD optimized propagation for multiple outputs using NEON (ARM)
    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(stockcrab_dotprod, target_feature(enable = "dotprod"))]
    unsafe fn propagate_simd_multi(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::aarch64::*;

        const OUTPUT_SIMD_WIDTH: usize = 4; // NEON: 128 bits / 32 bits = 4 i32s
        const CHUNK_SIZE: usize = 4;
        let num_chunks = IN_DIMS.div_ceil(8) * 2;
        let num_regs = OUT_DIMS / OUTPUT_SIMD_WIDTH;

        debug_assert_eq!(
            (input.as_ptr() as usize) & 3,
            0,
            "NNUE affine input must be 4-byte aligned"
        );
        debug_assert!(input.len() >= num_chunks * 4);
        let input32 =
            // SAFETY: `input` is 4-byte aligned and holds >= `num_chunks * 4` bytes (asserted in `propagate`;
            // the only caller), so reinterpreting it as
            // `num_chunks` i32 is valid.
            unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i32, num_chunks) };

        // Handle small OUT_DIMS (< 4) with scalar fallback
        if num_regs == 0 {
            // Initialize output with biases
            for j in 0..OUT_DIMS {
                output[j] = self.biases[j];
            }

            // Process input in chunks of 4 bytes (32 bits)
            for i in 0..num_chunks {
                let in_val = input32[i];
                if in_val != 0 {
                    let weight_offset = i * OUT_DIMS * CHUNK_SIZE;

                    // Process 4 input bytes at once (scalar version of DPBUSD)
                    for k in 0..OUT_DIMS {
                        let mut sum = 0i32;
                        for b in 0..4 {
                            let input_byte = ((in_val >> (b * 8)) & 0xFF) as u8;
                            let weight_idx = weight_offset + k * 4 + b;
                            if weight_idx < self.weights.len() {
                                sum += (input_byte as i32) * (self.weights[weight_idx] as i32);
                            }
                        }
                        output[k] += sum;
                    }
                }
            }
            return;
        }

        // SAFETY: NEON is baseline on aarch64. Biases hold `OUT_DIMS` i32 and `num_regs = OUT_DIMS
        // / 4`; weight reads stay below `num_chunks * OUT_DIMS * 4 <= OUT_DIMS *
        // PADDED_INPUT_DIMENSIONS` bytes; `output` holds >= `OUT_DIMS` i32 (asserted in
        // `propagate`).
        unsafe {
            // Initialize accumulators with biases
            let mut acc: [int32x4_t; 16] = [vdupq_n_s32(0); 16];
            for k in 0..num_regs {
                acc[k] = vld1q_s32(self.biases.as_ptr().add(k * 4));
            }

            let mut i = 0;
            #[cfg(stockcrab_dotprod)]
            {
                let mut acc2 = [vdupq_n_s32(0); 16];
                while i + 1 < num_chunks {
                    let in0 = vreinterpretq_s8_s32(vdupq_n_s32(input32[i]));
                    let in1 = vreinterpretq_s8_s32(vdupq_n_s32(input32[i + 1]));
                    let col0 = self.weights.as_ptr().add(i * OUT_DIMS * 4);
                    let col1 = self.weights.as_ptr().add((i + 1) * OUT_DIMS * 4);
                    for k in 0..num_regs {
                        crate::nnue::simd::neon::neon_m128_add_dpbusd_epi32(
                            &mut acc[k],
                            in0,
                            vld1q_s8(col0.add(k * 16)),
                        );
                        crate::nnue::simd::neon::neon_m128_add_dpbusd_epi32(
                            &mut acc2[k],
                            in1,
                            vld1q_s8(col1.add(k * 16)),
                        );
                    }
                    i += 2;
                }
                for k in 0..num_regs {
                    acc[k] = vaddq_s32(acc[k], acc2[k]);
                }
            }
            while i < num_chunks {
                let input_vector = vreinterpretq_s8_s32(vdupq_n_s32(input32[i]));
                let column = self.weights.as_ptr().add(i * OUT_DIMS * 4);
                for k in 0..num_regs {
                    crate::nnue::simd::neon::neon_m128_add_dpbusd_epi32(
                        &mut acc[k],
                        input_vector,
                        vld1q_s8(column.add(k * 16)),
                    );
                }
                i += 1;
            }

            // Store results
            for k in 0..num_regs {
                vst1q_s32(output.as_mut_ptr().add(k * 4), acc[k]);
            }
        }
    }

    /// SIMD optimized propagation for single output using AVX2
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn propagate_simd_single(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;

        const INPUT_SIMD_WIDTH: usize = 32; // AVX2: 256 bits / 8 bits = 32 bytes
        let num_chunks = Self::PADDED_INPUT_DIMENSIONS / INPUT_SIMD_WIDTH;

        // SAFETY: AVX2 is enabled by the enclosing `cfg(target_feature)`. `OUT_DIMS == 1`, so the
        // weights hold exactly `PADDED_INPUT_DIMENSIONS` bytes, i.e. `num_chunks * 32`; `input`
        // must hold at least that many bytes (asserted in `propagate`) and
        // unaligned loads are used.
        unsafe {
            let input_vec = input.as_ptr() as *const __m256i;
            let weights_vec = self.weights.as_ptr() as *const __m256i;

            let mut sum0 = _mm256_setzero_si256();

            for j in 0..num_chunks {
                let in_val = _mm256_loadu_si256(input_vec.add(j));
                let weights = _mm256_loadu_si256(weights_vec.add(j));
                crate::nnue::simd::avx2::m256_add_dpbusd_epi32(&mut sum0, in_val, weights);
            }

            output[0] = crate::nnue::simd::avx2::m256_hadd(sum0, self.biases[0]);
        }
    }

    /// SIMD optimized propagation for single output using SSSE3
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "ssse3",
        not(target_feature = "avx2")
    ))]
    fn propagate_simd_single(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::x86_64::*;

        const INPUT_SIMD_WIDTH: usize = 16; // SSE: 128 bits / 8 bits = 16 bytes
        let num_chunks = Self::PADDED_INPUT_DIMENSIONS / INPUT_SIMD_WIDTH;

        // SAFETY: SSSE3 is enabled by the enclosing `cfg(target_feature)`. `OUT_DIMS == 1`, so the
        // weights hold exactly `PADDED_INPUT_DIMENSIONS` bytes, i.e. `num_chunks * 16`; `input`
        // must hold at least that many bytes (asserted in `propagate`) and
        // unaligned loads are used.
        unsafe {
            let input_vec = input.as_ptr() as *const __m128i;
            let weights_vec = self.weights.as_ptr() as *const __m128i;

            let mut sum0 = _mm_setzero_si128();

            for j in 0..num_chunks {
                let in_val = _mm_loadu_si128(input_vec.add(j));
                let weights = _mm_loadu_si128(weights_vec.add(j));
                crate::nnue::simd::sse2::m128_add_dpbusd_epi32(&mut sum0, in_val, weights);
            }

            output[0] = crate::nnue::simd::sse2::m128_hadd(sum0, self.biases[0]);
        }
    }

    /// SIMD optimized propagation for single output using NEON (ARM)
    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(stockcrab_dotprod, target_feature(enable = "dotprod"))]
    unsafe fn propagate_simd_single(&self, input: &[u8], output: &mut [i32]) {
        use core::arch::aarch64::*;

        const INPUT_SIMD_WIDTH: usize = 16; // NEON: 128 bits / 8 bits = 16 bytes
        let num_chunks = Self::PADDED_INPUT_DIMENSIONS / INPUT_SIMD_WIDTH;

        // SAFETY: NEON is baseline on aarch64. `OUT_DIMS == 1`, so the weights hold exactly
        // `PADDED_INPUT_DIMENSIONS` bytes, i.e. `num_chunks * 16`; `input` must hold at least that
        // many bytes (asserted in `propagate`).
        unsafe {
            let mut sum0 = vdupq_n_s32(0);

            for j in 0..num_chunks {
                let in_val = vld1q_s8(input.as_ptr().add(j * 16) as *const i8);
                let weights = vld1q_s8(self.weights.as_ptr().add(j * 16));
                crate::nnue::simd::neon::neon_m128_add_dpbusd_epi32(&mut sum0, in_val, weights);
            }

            output[0] = crate::nnue::simd::neon::neon_m128_hadd(sum0, self.biases[0]);
        }
    }

    /// Fallback for architectures without x86 SIMD or NEON - single output
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64"
    )))]
    fn propagate_simd_single(&self, input: &[u8], output: &mut [i32]) {
        // Scalar dot product for single output
        let mut sum = self.biases[0];

        // Process all input bytes
        for i in 0..Self::PADDED_INPUT_DIMENSIONS.min(input.len()) {
            if input[i] != 0 {
                sum += (input[i] as i32) * (self.weights[i] as i32);
            }
        }

        output[0] = sum;
    }
}

impl<const IN_DIMS: usize, const OUT_DIMS: usize> Default for AffineTransform<IN_DIMS, OUT_DIMS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_output_fallback_decodes_the_chunked_weight_layout() {
        type Layer = AffineTransform<32, 3>;
        let mut layer = Layer::new();
        let mut serialized = vec![0u8; 3 * 4];
        for row in 1..=3 {
            serialized.extend(std::iter::repeat_n(
                row as u8,
                Layer::PADDED_INPUT_DIMENSIONS,
            ));
        }
        layer
            .read_parameters(&mut std::io::Cursor::new(serialized))
            .unwrap();
        let mut input = AlignedVec::<u8>::new_zeroed(Layer::PADDED_INPUT_DIMENSIONS);
        input[..32].fill(1);
        let mut output = [0; 3];
        layer.propagate(&input, &mut output);
        assert_eq!(output, [32, 64, 96]);
    }
    #[allow(unused_imports)] // only the AVX-512 tests use it
    use crate::nnue::layers::SqrClippedReLU;

    /// Zero-padded, 64-byte aligned copy of a test input: `propagate` requires an aligned buffer
    /// of at least the padded input width.
    #[repr(align(64))]
    struct AlignedBuf([u8; 128]);

    impl AlignedBuf {
        fn new(src: &[u8]) -> Self {
            let mut buf = Self([0; 128]);
            buf.0[..src.len()].copy_from_slice(src);
            buf
        }
    }

    #[test]
    fn test_weight_index_scrambling() {
        type TestLayer = AffineTransform<30, 32>;

        // The mapping must be a bijection onto the weight storage.
        let total = 32 * TestLayer::PADDED_INPUT_DIMENSIONS;
        let mut seen = vec![false; total];
        for i in 0..total {
            let idx = TestLayer::get_weight_index(i);
            assert!(idx < total, "index {idx} out of range for input {i}");
            assert!(!seen[idx], "index {idx} produced twice (input {i})");
            seen[idx] = true;
        }
    }

    /// Checks `propagate` (whichever kernel this build selects: NEON, NEON dot-product, SSSE3,
    /// AVX2, AVX-512, or the single-output kernels) against a plain scalar reference on random
    /// non-zero weights, biases and activations.
    ///
    /// The weights are placed through `get_weight_index`, so every storage slot is written, and
    /// the reference reads the storage back in its physical layout: column-major chunks for
    /// SIMD builds, row-major for the scalar baseline. Activations stay in physical order, so
    /// the test is independent of the AVX2+ pair-activation weight permutation, which only
    /// changes which logical column lands in which physical slot.
    fn check_matches_reference<const IN_DIMS: usize, const OUT_DIMS: usize>(seed: u64) {
        let padded = AffineTransform::<IN_DIMS, OUT_DIMS>::PADDED_INPUT_DIMENSIONS;
        assert!(padded <= 128, "AlignedBuf holds at most 128 bytes");

        // Deterministic xorshift64.
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for case in 0..300 {
            let mut layer = AffineTransform::<IN_DIMS, OUT_DIMS>::new();
            for bias in layer.biases.iter_mut() {
                // Large enough to matter, small enough that nothing wraps.
                *bias = (next() % 2_000_001) as i32 - 1_000_000;
            }
            for serial in 0..OUT_DIMS * padded {
                let weight = match case % 3 {
                    0 => next() as i8,
                    1 => [i8::MIN, i8::MAX][(next() % 2) as usize],
                    _ => (next() % 9) as i8 - 4,
                };
                layer.weights[AffineTransform::<IN_DIMS, OUT_DIMS>::get_weight_index(serial)] =
                    weight;
            }

            // Activations are clipped-ReLU outputs, 0..=127; the padding stays zero.
            let mut input = AlignedBuf([0; 128]);
            for byte in input.0[..IN_DIMS].iter_mut() {
                *byte = (next() % 128) as u8;
            }

            let mut expected = [0i32; OUT_DIMS];
            for (row, out) in expected.iter_mut().enumerate() {
                *out = layer.biases[row];
                for column in 0..padded {
                    let index = if ENABLE_SEQ_OPT {
                        column / 4 * OUT_DIMS * 4 + row * 4 + column % 4
                    } else {
                        row * padded + column
                    };
                    *out += i32::from(input.0[column]) * i32::from(layer.weights[index]);
                }
            }

            let mut got = [0i32; OUT_DIMS];
            layer.propagate(&input.0, &mut got);
            assert_eq!(
                got, expected,
                "AffineTransform<{IN_DIMS},{OUT_DIMS}> case {case}"
            );
        }
    }

    /// The multi-output kernels (including the two-accumulator NEON dot-product schedule) and
    /// the single-output kernel against the scalar reference, for the production fc_1 / fc_2
    /// shapes and a few register-boundary shapes.
    #[test]
    fn multi_output_matches_reference() {
        check_matches_reference::<64, 32>(0x2545_F491_4F6C_DD1D);
        check_matches_reference::<128, 1>(0x9E37_79B9_7F4A_7C15);
        check_matches_reference::<32, 4>(0xD1B5_4A32_D192_ED03);
        check_matches_reference::<96, 8>(0x8CB9_2BA7_2F3D_8DD7);
        check_matches_reference::<30, 12>(0x1234_5678_9ABC_DEF1);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn avx512_multi_output_matches_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512bw")
        {
            return;
        }

        type TestLayer = AffineTransform<64, 32>;
        let mut layer = TestLayer::new();
        for output in 0..32 {
            layer.biases[output] = output as i32 * 17 - 101;
        }
        for serial in 0..32 * TestLayer::PADDED_INPUT_DIMENSIONS {
            layer.weights[TestLayer::get_weight_index(serial)] =
                ((serial as i32 * 19 % 127) - 63) as i8;
        }
        let mut input = [0u8; 64];
        for (i, byte) in input.iter_mut().enumerate() {
            // Keep all 4-byte groups in a 32-byte block equal. This isolates
            // the dot-product kernel from the paired-activation permutation.
            *byte = (i as u8 % 4)
                .wrapping_mul(37)
                .wrapping_add((i as u8 / 32) * 13);
        }

        let mut expected = [0i32; 32];
        for output in 0..32 {
            expected[output] = layer.biases[output];
            for i in 0..64 {
                let weight =
                    ((output * TestLayer::PADDED_INPUT_DIMENSIONS + i) as i32 * 19 % 127) - 63;
                expected[output] += input[i] as i32 * weight;
            }
        }
        let mut got = [0i32; 32];
        let input = AlignedBuf::new(&input);
        // SAFETY: the test is cfg-gated on avx512f/avx512bw (compile time) and returns early unless
        // `is_x86_feature_detected!` confirms both at runtime; `input` is 4-byte aligned (64-byte
        // `AlignedBuf`) and holds the 64 input bytes, and `got` holds the layer's 32 outputs.
        unsafe { layer.propagate_simd_multi_avx512(&input.0, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn paired_activations_and_permuted_weights_match_logical_network() {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512bw")
        {
            return;
        }

        type TestLayer = AffineTransform<64, 32>;
        let activation = SqrClippedReLU::<32, 7>::new();
        let mut layer = TestLayer::new();
        for output in 0..32 {
            layer.biases[output] = output as i32 * 11 - 73;
        }
        for serial in 0..32 * TestLayer::PADDED_INPUT_DIMENSIONS {
            layer.weights[TestLayer::get_weight_index(serial)] =
                ((serial as i32 * 23 % 127) - 63) as i8;
        }

        let mut source = [0i32; 32];
        for (i, value) in source.iter_mut().enumerate() {
            *value = (i as i32 * 431 % 70_000) - 20_000;
        }
        let mut squared = [0u8; 32];
        let mut clipped = [0u8; 32];
        activation.propagate_pair(&source, &mut squared, &mut clipped);
        let mut packed_input = [0u8; 64];
        packed_input[..32].copy_from_slice(&squared);
        packed_input[32..].copy_from_slice(&clipped);

        let mut expected = [0i32; 32];
        for output in 0..32 {
            expected[output] = layer.biases[output];
            for i in 0..32 {
                let value = source[i] as i64;
                let square = (((value * value) >> 21).min(127)) as i32;
                let linear = (source[i] >> 7).clamp(0, 127);
                let square_weight = ((output * 64 + i) as i32 * 23 % 127) - 63;
                let linear_weight = ((output * 64 + 32 + i) as i32 * 23 % 127) - 63;
                expected[output] += square * square_weight + linear * linear_weight;
            }
        }
        let mut got = [0i32; 32];
        layer.propagate(&AlignedBuf::new(&packed_input).0, &mut got);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_hash_value() {
        let hash1 = AffineTransform::<30, 32>::get_hash_value(0x12345678);
        let hash2 = AffineTransform::<30, 33>::get_hash_value(0x12345678);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_propagate() {
        let layer = AffineTransform::<4, 2>::new();
        // `AlignedBuf` zero-pads the input to the PADDED_INPUT_DIMENSIONS (32 bytes) that
        // `propagate()` requires.
        let input = [0u8; 8];
        let mut output = [0i32; 2];

        layer.propagate(&AlignedBuf::new(&input).0, &mut output);

        // With zero weights and biases, output should be zero
        assert_eq!(output, [0, 0]);
    }

    #[test]
    fn test_single_output() {
        let mut layer = AffineTransform::<4, 1>::new();
        layer.biases[0] = 100;
        // Only the first 4 bytes are meaningful for IN_DIMS=4; `AlignedBuf` zero-pads the rest to
        // the PADDED_INPUT_DIMENSIONS (32 bytes) that `propagate()` requires.
        let input = [1u8, 2, 3, 4, 0, 0, 0, 0];
        let mut output = [0i32; 1];

        layer.propagate(&AlignedBuf::new(&input).0, &mut output);

        // Output should be bias value when weights are zero
        assert_eq!(output[0], 100);
    }

    /// Test dot product computation with known weights and inputs
    #[test]
    fn test_dot_product_multi_output() {
        // Use 8 inputs, 2 outputs to test the CHUNK_SIZE=4 dot product
        type TestLayer = AffineTransform<8, 2>;
        let mut layer = TestLayer::new();

        // Set up known biases
        layer.biases[0] = 100;
        layer.biases[1] = 200;

        // Set weights manually (without scrambling for this test)
        // For output 0: weights at positions 0,1,2,3 (first chunk), 4,5,6,7 (second chunk)
        // For output 1: weights at next set of positions
        // Since PADDED_INPUT_DIMENSIONS rounds up to 32, layout is:
        // output[k] = sum over i of (input[i] * weights[i + k * PADDED_INPUT_DIMENSIONS])

        // Set weights for output 0
        for i in 0..8 {
            layer.weights[i] = (i + 1) as i8; // weights: 1,2,3,4,5,6,7,8
        }
        // Set weights for output 1
        for i in 0..8 {
            layer.weights[TestLayer::PADDED_INPUT_DIMENSIONS + i] = 1; // all weights = 1
        }

        // Input: [1,2,3,4,5,6,7,8]
        let input: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut output = [0i32; 2];

        // Use non-SIMD fallback to test basic logic
        affine_transform_non_ssse3::<8, 2>(
            &mut output,
            &layer.weights,
            &layer.biases,
            &input,
            TestLayer::PADDED_INPUT_DIMENSIONS,
        );

        // Expected output[0] = 100 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8
        //                    = 100 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64
        //                    = 100 + 204 = 304
        // Expected output[1] = 200 + 1+2+3+4+5+6+7+8 = 200 + 36 = 236
        assert_eq!(output[0], 304, "Output 0 dot product incorrect");
        assert_eq!(output[1], 236, "Output 1 dot product incorrect");
    }

    /// Test that propagate handles all chunks correctly for larger inputs
    #[test]
    fn test_propagate_all_chunks_processed() {
        // Use 32 inputs to ensure multiple chunks are processed
        type TestLayer = AffineTransform<32, 1>;
        let mut layer = TestLayer::new();

        layer.biases[0] = 0;

        // Set all weights to 1
        for i in 0..32 {
            layer.weights[i] = 1;
        }

        // Input: all 1s
        let input = [1u8; 32];
        let mut output = [0i32; 1];

        layer.propagate(&AlignedBuf::new(&input).0, &mut output);

        // Expected: sum of 32 ones = 32
        assert_eq!(output[0], 32, "All 32 input elements should be processed");
    }

    /// Test that the last chunk is correctly processed (edge case)
    #[test]
    fn test_propagate_partial_last_chunk() {
        // 30 inputs means 8 chunks of 4 bytes, but only first 30 bytes have valid data
        type TestLayer = AffineTransform<30, 1>;
        let mut layer = TestLayer::new();

        layer.biases[0] = 0;

        // Set weights: 1 for first 30, 0 for padding
        for i in 0..30 {
            layer.weights[i] = 1;
        }

        // Input: all 1s for first 30 positions
        let mut input = [0u8; 32]; // padded
        for i in 0..30 {
            input[i] = 1;
        }

        let mut output = [0i32; 1];

        affine_transform_non_ssse3::<30, 1>(
            &mut output,
            &layer.weights,
            &layer.biases,
            &input,
            TestLayer::PADDED_INPUT_DIMENSIONS,
        );

        // Expected: sum of 30 ones = 30 (not 32!)
        assert_eq!(output[0], 30, "Only 30 input elements should be processed");
    }

    /// Test that propagate produces correct results for known input/weights
    /// This verifies the public API works correctly regardless of internal implementation
    #[test]
    fn test_propagate_known_values() {
        type TestLayer = AffineTransform<8, 2>;
        let mut layer = TestLayer::new();

        // Set biases
        layer.biases[0] = 50;
        layer.biases[1] = 100;

        // Set weights using direct indexing (for non-scrambled case)
        // For non-SIMD path: weights[i * padded_input + j] where i=output, j=input
        // We test with zero weights to avoid scrambling issues
        // Then verify bias passthrough works

        let input = [0u8; 8];
        let mut output = [0i32; 2];

        layer.propagate(&AlignedBuf::new(&input).0, &mut output);

        // With zero input, output should equal biases
        assert_eq!(output[0], 50, "Output 0 should be bias when input is zero");
        assert_eq!(output[1], 100, "Output 1 should be bias when input is zero");
    }

    /// Test propagate with single non-zero output neuron
    #[test]
    fn test_propagate_single_output_with_weights() {
        type TestLayer = AffineTransform<4, 1>;
        let mut layer = TestLayer::new();

        layer.biases[0] = 10;

        // Set first 4 weights for output 0 (direct indexing)
        layer.weights[0] = 1;
        layer.weights[1] = 2;
        layer.weights[2] = 3;
        layer.weights[3] = 4;

        // `AlignedBuf` zero-pads the input to the PADDED_INPUT_DIMENSIONS (32 bytes) that
        // `propagate()` requires: the SIMD kernels read a full vector of input.
        let mut input = [0u8; 32];
        input[..4].copy_from_slice(&[1, 1, 1, 1]);
        let mut output = [0i32; 1];

        layer.propagate(&AlignedBuf::new(&input).0, &mut output);

        // Expected: 10 + 1*1 + 1*2 + 1*3 + 1*4 = 10 + 10 = 20
        assert_eq!(
            output[0], 20,
            "Single output with unit input should sum weights + bias"
        );
    }

    #[test]
    #[should_panic(expected = "aligned")]
    fn test_propagate_panics_on_unaligned_input() {
        // Force a misaligned pointer by slicing at an odd offset; the slice is long enough to
        // pass the length assert so the alignment assert is the one that fires.
        let buf = AlignedBuf([0u8; 128]);
        let input = &buf.0[1..65];
        let layer = AffineTransform::<4, 2>::new();
        let mut output = [0i32; 2];
        layer.propagate(input, &mut output);
    }
}
