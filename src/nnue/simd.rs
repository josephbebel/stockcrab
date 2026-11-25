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

//! SIMD operations for NNUE evaluation
//!
//! This module provides SIMD-accelerated operations for the NNUE neural network,
//! supporting AVX512, AVX2, SSE2, and NEON instruction sets.

// `vec_t`, `vec_i8_t`, `NumRegistersSIMD` and `MaxChunkSize` mirror the C++ names.
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
// The `vec_*` helpers are one-line wrappers over a single intrinsic; each requires only
// valid, suitably aligned pointers and the matching target feature.
#![allow(clippy::missing_safety_doc)]
// SIMD kernel file: the `unsafe fn` bodies are intrinsics and raw-pointer arithmetic, so
// wrapping every statement in its own `unsafe {}` (the edition-2024 default) would add noise
// without information. Each kernel documents its invariants instead.
#![allow(unsafe_op_in_unsafe_fn)]

// Import architecture-specific intrinsics based on target features
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::nnue_common::WeightType;

// ---------------------------------------------------------------------------
// VECTOR TYPE DEFINITIONS AND OPERATIONS
// ---------------------------------------------------------------------------

// C++ derives one flag here:
//   #if defined(USE_AVX2)
//       #define USE_PAIR_ACTIVATIONS
//   #endif
// It gates `SqrClippedReLU::propagate_pair`, which produces the squared and the linear
// clipped activations together, sharing the input loads and the initial signed 32-to-16-bit
// saturating narrowing.
//
// Upstream `b52f0147` collapsed three flags into this one: `USE_AVX2_PAIR_ACTIVATIONS`
// (plain AVX2, no VNNI, no AVX512) and `USE_SCRAMBLED_ACTIVATIONS` are both gone, AVX512
// now permutes at load time instead of at run time, and AVXVNNI joined the paired path. The
// `ScrambledInput` template parameter that rode along with them was dropped from
// `AffineTransform` as well.
//
// The paired activation kernels live in `layers::sqr_clipped_relu`; their
// target-derived selection is kept next to the network-architecture consumer.

// ---------------------------------------------------------------------------
// AVX512 SUPPORT
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
/// AVX-512 (F + BW) vector type and wrappers.
pub mod avx512 {
    use super::*;

    pub type vec_t = __m512i;
    pub type vec_i8_t = __m256i;

    pub const VECTOR: bool = true;
    pub const NumRegistersSIMD: usize = 16;
    pub const MaxChunkSize: usize = 64;
    const _: () = assert!(
        MaxChunkSize == core::mem::size_of::<vec_t>(),
        "MaxChunkSize must equal one vector."
    );

    #[inline(always)]
    pub unsafe fn vec_load(a: *const __m512i) -> __m512i {
        _mm512_load_si512(a)
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut __m512i, b: __m512i) {
        _mm512_store_si512(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_add_16(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_sub_16(a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_mulhi_16(a: __m512i, b: __m512i) -> __m512i {
        _mm512_mulhi_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_zero() -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    pub unsafe fn vec_set_16(a: i16) -> __m512i {
        _mm512_set1_epi16(a)
    }

    #[inline(always)]
    pub unsafe fn vec_max_16(a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_min_16(a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_slli_16<const B: u32>(a: __m512i) -> __m512i {
        _mm512_slli_epi16::<B>(a)
    }

    #[inline(always)]
    pub unsafe fn vec_packus_16(a: __m512i, b: __m512i) -> __m512i {
        _mm512_packus_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_nnz(a: __m512i) -> u16 {
        _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512())
    }

    #[inline(always)]
    pub unsafe fn m512_hadd(sum: __m512i, bias: i32) -> i32 {
        _mm512_reduce_add_epi32(sum) + bias
    }

    #[inline(always)]
    pub unsafe fn m512_add_dpbusd_epi32(acc: &mut __m512i, a: __m512i, b: __m512i) {
        #[cfg(target_feature = "avx512vnni")]
        {
            *acc = _mm512_dpbusd_epi32(*acc, a, b);
        }
        #[cfg(not(target_feature = "avx512vnni"))]
        {
            let product0 = _mm512_maddubs_epi16(a, b);
            let product0 = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
            *acc = _mm512_add_epi32(*acc, product0);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 SUPPORT
// ---------------------------------------------------------------------------

/// AVX2 vector type and wrappers.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2 {
    use super::*;

    pub type vec_t = __m256i;
    pub type vec_i8_t = __m128i;

    pub const VECTOR: bool = true;
    // Stockfish reserves twelve YMM registers on x86-64, so the tiler must not pick
    // accumulator tiles that cannot stay register-resident.
    pub const NumRegistersSIMD: usize = 12;
    pub const MaxChunkSize: usize = 32;
    const _: () = assert!(
        MaxChunkSize == core::mem::size_of::<vec_t>(),
        "MaxChunkSize must equal one vector."
    );

    #[inline(always)]
    pub unsafe fn vec_load(a: *const __m256i) -> __m256i {
        _mm256_load_si256(a)
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut __m256i, b: __m256i) {
        _mm256_store_si256(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_add_16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_sub_16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_mulhi_16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_mulhi_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_zero() -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    pub unsafe fn vec_set_16(a: i16) -> __m256i {
        _mm256_set1_epi16(a)
    }

    #[inline(always)]
    pub unsafe fn vec_max_16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_min_16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_slli_16<const B: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi16::<B>(a)
    }

    #[inline(always)]
    pub unsafe fn vec_packus_16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_packus_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_nnz(a: __m256i) -> i32 {
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(
            a,
            _mm256_setzero_si256(),
        )))
    }

    #[inline(always)]
    pub unsafe fn m256_hadd(sum: __m256i, bias: i32) -> i32 {
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256(sum, 1),
        );
        let sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E)); // _MM_PERM_BADC
        let sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0xB1)); // _MM_PERM_CDAB
        _mm_cvtsi128_si32(sum128) + bias
    }

    #[inline(always)]
    pub unsafe fn m256_add_dpbusd_epi32(acc: &mut __m256i, a: __m256i, b: __m256i) {
        #[cfg(target_feature = "avxvnni")]
        {
            *acc = _mm256_dpbusd_epi32(*acc, a, b);
        }
        #[cfg(not(target_feature = "avxvnni"))]
        {
            let product0 = _mm256_maddubs_epi16(a, b);
            let product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
            *acc = _mm256_add_epi32(*acc, product0);
        }
    }
}

// ---------------------------------------------------------------------------
// SSE2 SUPPORT
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
pub mod sse2 {
    use super::*;

    pub type vec_t = __m128i;
    // for the correct size -- will be loaded into an xmm reg
    pub type vec_i8_t = u64;

    pub const VECTOR: bool = true;
    #[cfg(target_pointer_width = "64")]
    pub const NumRegistersSIMD: usize = 12;
    #[cfg(target_pointer_width = "32")]
    pub const NumRegistersSIMD: usize = 8;
    pub const MaxChunkSize: usize = 16;
    const _: () = assert!(
        MaxChunkSize == core::mem::size_of::<vec_t>(),
        "MaxChunkSize must equal one vector."
    );

    #[inline(always)]
    pub unsafe fn vec_load(a: *const __m128i) -> __m128i {
        *a
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut __m128i, b: __m128i) {
        *a = b;
    }

    #[inline(always)]
    pub unsafe fn vec_add_16(a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_sub_16(a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_mulhi_16(a: __m128i, b: __m128i) -> __m128i {
        _mm_mulhi_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_zero() -> __m128i {
        _mm_setzero_si128()
    }

    #[inline(always)]
    pub unsafe fn vec_set_16(a: i16) -> __m128i {
        _mm_set1_epi16(a)
    }

    #[inline(always)]
    pub unsafe fn vec_max_16(a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_min_16(a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn vec_slli_16(a: __m128i, b: i32) -> __m128i {
        // `_mm_slli_epi16` requires a compile-time immediate, while this
        // backend exposes the shift count as an argument.  Use the variable
        // count form so generic x86-64/SSE2 builds compile as well as native
        // builds; both produce the same lane-wise logical shift.
        _mm_sll_epi16(a, _mm_cvtsi32_si128(b))
    }

    #[inline(always)]
    pub unsafe fn vec_packus_16(a: __m128i, b: __m128i) -> __m128i {
        _mm_packus_epi16(a, b)
    }

    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    pub unsafe fn vec_nnz(a: __m128i) -> i32 {
        _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128())))
    }

    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    pub unsafe fn m128_hadd(sum: __m128i, bias: i32) -> i32 {
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E)); // _MM_PERM_BADC
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1)); // _MM_PERM_CDAB
        _mm_cvtsi128_si32(sum) + bias
    }

    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    pub unsafe fn m128_add_dpbusd_epi32(acc: &mut __m128i, a: __m128i, b: __m128i) {
        let product0 = _mm_maddubs_epi16(a, b);
        let product0 = _mm_madd_epi16(product0, _mm_set1_epi16(1));
        *acc = _mm_add_epi32(*acc, product0);
    }
}

// ---------------------------------------------------------------------------
// NEON SUPPORT (ARM)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;

    pub type vec_t = int16x8_t;
    pub type vec_i8_t = int8x16_t;

    pub const VECTOR: bool = true;
    pub const NumRegistersSIMD: usize = 16;
    pub const MaxChunkSize: usize = 16;
    const _: () = assert!(
        MaxChunkSize == core::mem::size_of::<vec_t>(),
        "MaxChunkSize must equal one vector."
    );

    #[inline(always)]
    pub unsafe fn vec_load(a: *const int16x8_t) -> int16x8_t {
        // SAFETY: the caller guarantees `a` is a valid, 16-byte-aligned pointer to a vector
        // (contract of this wrapper).
        unsafe { *a }
    }

    #[inline(always)]
    pub unsafe fn vec_store(a: *mut int16x8_t, b: int16x8_t) {
        // SAFETY: the caller guarantees `a` is a valid, 16-byte-aligned, writable pointer to a
        // vector.
        unsafe {
            *a = b;
        }
    }

    #[inline(always)]
    pub unsafe fn vec_add_16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline, so the intrinsic is available; it is a pure
        // register operation.
        unsafe { vaddq_s16(a, b) }
    }

    #[inline(always)]
    pub unsafe fn vec_sub_16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vsubq_s16(a, b) }
    }

    #[inline(always)]
    pub unsafe fn vec_mulhi_16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vqdmulhq_s16(a, b) }
    }

    #[inline(always)]
    pub unsafe fn vec_zero() -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vdupq_n_s16(0) }
    }

    #[inline(always)]
    pub unsafe fn vec_set_16(a: i16) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vdupq_n_s16(a) }
    }

    #[inline(always)]
    pub unsafe fn vec_max_16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vmaxq_s16(a, b) }
    }

    #[inline(always)]
    pub unsafe fn vec_min_16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vminq_s16(a, b) }
    }

    #[inline(always)]
    pub unsafe fn vec_slli_16(a: int16x8_t, b: i32) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vshlq_s16(a, vdupq_n_s16(b as i16)) }
    }

    #[inline(always)]
    pub unsafe fn vec_packus_16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        // SAFETY: NEON intrinsics as above; the transmute is between two 16-byte vector types with
        // no invalid bit patterns.
        unsafe {
            let packed = vcombine_u8(vqmovun_s16(a), vqmovun_s16(b));
            core::mem::transmute(packed)
        }
    }

    #[inline(always)]
    pub unsafe fn neon_m128_reduce_add_epi32(s: int32x4_t) -> i32 {
        // SAFETY: NEON is part of the aarch64 baseline; pure register operation.
        unsafe { vaddvq_s32(s) }
    }

    #[inline(always)]
    pub unsafe fn neon_m128_hadd(sum: int32x4_t, bias: i32) -> i32 {
        // SAFETY: calls another NEON wrapper; same baseline-NEON justification, no memory access.
        unsafe { neon_m128_reduce_add_epi32(sum) + bias }
    }

    /// NEON dot product accumulation - uses SDOT under `cfg(stockcrab_dotprod)`
    /// Falls back to vmull/vpaddq/vpadal sequence otherwise
    ///
    /// `build.rs` sets `stockcrab_dotprod` when the target has the `dotprod`
    /// feature (the Apple Silicon default) or the `neon-dotprod` Cargo feature
    /// asserts dot-product hardware. The production callers enable `dotprod` on
    /// their kernel, so the intrinsic can inline even without a global
    /// target-feature flag.
    #[cfg(all(target_arch = "aarch64", stockcrab_dotprod))]
    #[inline(always)]
    pub unsafe fn neon_m128_add_dpbusd_epi32(acc: &mut int32x4_t, a: int8x16_t, b: int8x16_t) {
        *acc = vdotq_s32(*acc, a, b);
    }

    /// NEON dot product accumulation - fallback without dotprod
    #[cfg(all(target_arch = "aarch64", not(stockcrab_dotprod)))]
    #[inline(always)]
    pub unsafe fn neon_m128_add_dpbusd_epi32(acc: &mut int32x4_t, a: int8x16_t, b: int8x16_t) {
        let product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let product1 = vmull_high_s8(a, b);
        let sum = vpaddq_s16(product0, product1);
        *acc = vpadalq_s16(*acc, sum);
    }
}

// ---------------------------------------------------------------------------
// SCALAR FALLBACK (NO SIMD)
// ---------------------------------------------------------------------------

#[cfg(not(any(
    all(
        target_arch = "x86_64",
        any(
            target_feature = "sse2",
            target_feature = "avx2",
            target_feature = "avx512f"
        )
    ),
    target_arch = "aarch64"
)))]
pub mod scalar {
    use crate::nnue::nnue_common::BiasType;

    pub type vec_t = BiasType;

    pub const VECTOR: bool = false;
    pub const NumRegistersSIMD: usize = 1;
    pub const MaxChunkSize: usize = 1;
}

// ---------------------------------------------------------------------------
// SELECTED SIMD IMPLEMENTATION
// ---------------------------------------------------------------------------

// Platform-specific implementations
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
use self::avx512 as simd_impl;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
use self::avx2 as simd_impl;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
use self::sse2 as simd_impl;

#[cfg(target_arch = "aarch64")]
use self::neon as simd_impl;

#[cfg(not(any(
    all(
        target_arch = "x86_64",
        any(
            target_feature = "sse2",
            target_feature = "avx2",
            target_feature = "avx512f"
        )
    ),
    target_arch = "aarch64"
)))]
use self::scalar as simd_impl;

// Re-export the selected implementation
pub use simd_impl::*;

// ---------------------------------------------------------------------------
// SIMD TILING
// ---------------------------------------------------------------------------

/// Compute optimal SIMD register count for feature transformer accumulation
pub struct SIMDTiling<const TRANSFORMED_FEATURE_WIDTH: usize, const HALF_DIMENSIONS: usize>;

impl<const TRANSFORMED_FEATURE_WIDTH: usize, const HALF_DIMENSIONS: usize>
    SIMDTiling<TRANSFORMED_FEATURE_WIDTH, HALF_DIMENSIONS>
{
    #[cfg(any(
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
        target_arch = "aarch64"
    ))]
    const fn best_register_count<
        const REGISTER_SIZE: usize,
        const LANE_SIZE: usize,
        const NUM_LANES: usize,
        const MAX_REGISTERS: usize,
    >() -> usize {
        assert!(REGISTER_SIZE >= LANE_SIZE);
        assert!(MAX_REGISTERS <= NumRegistersSIMD);
        assert!(MAX_REGISTERS > 0);
        assert!(NumRegistersSIMD > 0);
        assert!(REGISTER_SIZE.is_multiple_of(LANE_SIZE));
        assert!((NUM_LANES * LANE_SIZE).is_multiple_of(REGISTER_SIZE));

        let ideal = (NUM_LANES * LANE_SIZE) / REGISTER_SIZE;
        if ideal <= MAX_REGISTERS {
            return ideal;
        }

        // Look for the largest divisor of the ideal register count that is at most MAX_REGISTERS
        let mut divisor = MAX_REGISTERS;
        while divisor > 1 {
            if ideal.is_multiple_of(divisor) {
                return divisor;
            }
            divisor -= 1;
        }

        1
    }

    #[cfg(any(
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
        target_arch = "aarch64"
    ))]
    pub const NUM_REGS: usize = Self::best_register_count::<
        { core::mem::size_of::<vec_t>() },
        { core::mem::size_of::<WeightType>() },
        TRANSFORMED_FEATURE_WIDTH,
        NumRegistersSIMD,
    >();

    #[cfg(any(
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
        target_arch = "aarch64"
    ))]
    pub const TILE_HEIGHT: usize = Self::NUM_REGS * core::mem::size_of::<vec_t>() / 2;

    // Scalar fallback values
    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
        target_arch = "aarch64"
    )))]
    pub const NUM_REGS: usize = 1;

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
        target_arch = "aarch64"
    )))]
    pub const TILE_HEIGHT: usize = HALF_DIMENSIONS;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_constants() {
        // Verify that constants are set appropriately
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        {
            assert_eq!(NumRegistersSIMD, 16);
            assert_eq!(MaxChunkSize, 64);
            assert!(VECTOR);
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        ))]
        {
            assert_eq!(NumRegistersSIMD, 12);
            assert_eq!(MaxChunkSize, 32);
            assert!(VECTOR);
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            not(target_feature = "avx2")
        ))]
        {
            #[cfg(target_pointer_width = "64")]
            assert_eq!(NumRegistersSIMD, 12);
            #[cfg(target_pointer_width = "32")]
            assert_eq!(NumRegistersSIMD, 8);
            assert_eq!(MaxChunkSize, 16);
            assert!(VECTOR);
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(NumRegistersSIMD, 16);
            assert_eq!(MaxChunkSize, 16);
            assert!(VECTOR);
        }

        #[cfg(not(any(
            all(
                target_arch = "x86_64",
                any(
                    target_feature = "sse2",
                    target_feature = "avx2",
                    target_feature = "avx512f"
                )
            ),
            target_arch = "aarch64"
        )))]
        {
            assert_eq!(NumRegistersSIMD, 1);
            assert_eq!(MaxChunkSize, 1);
            assert!(!VECTOR);
        }
    }

    #[test]
    fn test_simd_tiling() {
        // Test tiling calculations
        type TestTiling = SIMDTiling<256, 1024>;

        // Just verify that the constants are defined
        let _ = TestTiling::NUM_REGS;
        let _ = TestTiling::TILE_HEIGHT;

        // On platforms with SIMD, verify divisibility
        #[cfg(any(
            all(
                target_arch = "x86_64",
                any(
                    target_feature = "sse2",
                    target_feature = "avx2",
                    target_feature = "avx512f"
                )
            ),
            target_arch = "aarch64"
        ))]
        {
            assert!(
                1024 % TestTiling::TILE_HEIGHT == 0,
                "TileHeight must divide HalfDimensions"
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_hadd() {
        // SAFETY: this test is cfg-gated on `target_feature = "avx2"` (compile time), so the AVX2
        // intrinsics and `m256_hadd` are available.
        unsafe {
            use core::arch::x86_64::*;

            // Create a vector with known values
            let v = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
            let bias = 100;

            let result = avx2::m256_hadd(v, bias);

            // Sum should be 1+2+3+4+5+6+7+8 = 36, plus bias = 136
            assert_eq!(result, 136);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "ssse3",
        not(target_feature = "avx2")
    ))]
    #[test]
    fn test_sse_hadd() {
        // SAFETY: this test is cfg-gated on `target_feature = "ssse3"` (compile time), so the SSE
        // intrinsics and `m128_hadd` are available.
        unsafe {
            use core::arch::x86_64::*;

            // Create a vector with known values
            let v = _mm_set_epi32(4, 3, 2, 1);
            let bias = 50;

            let result = sse2::m128_hadd(v, bias);

            // Sum should be 1+2+3+4 = 10, plus bias = 60
            assert_eq!(result, 60);
        }
    }
}
