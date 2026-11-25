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

//! Feature transformer for NNUE evaluation

// Scalar, SSE, AVX2, AVX-512 and NEON implementations of each kernel coexist in
// this file and the target picks one at compile time (`scripts/check-all.sh` compiles every
// target to keep all of them building).

// SIMD kernel file: unsafe fn bodies are intrinsics; per-statement unsafe{} would only add noise.
#![allow(unsafe_op_in_unsafe_fn)]

use super::layers::NnzInfo;
use super::network::{
    read_little_endian_i8, read_little_endian_i16, write_little_endian_i8, write_little_endian_i16,
};
use super::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};
use super::nnue_architecture::{L1, PSQFeatureSet, PairFeatureSet, ThreatFeatureSet};
use super::nnue_common::{
    AlignedVec, BiasType, CACHE_LINE_SIZE, ThreatWeightType, TransformedFeatureType, WeightType,
};
use crate::position::Position;
use bytemuck::Pod;
use core::mem::MaybeUninit;
use std::io::{Read, Write};

/// Maximum clipped feature transformer output value.
/// C++ nnue_common.h: `constexpr int FtMaxVal = 255;`
/// The SF18 network ships pre-scaled weights, so the bound is a plain 255.
pub const FT_MAX_VAL: i32 = 255;

/// Feature transformer that converts input features to hidden layer inputs.
///
/// SF18: single non-generic transformer. The combined PSQ + Threat
/// accumulation has width `OutputDimensions == L1 == 1024`.
#[repr(C, align(64))] // CACHE_LINE_SIZE alignment
#[derive(Clone)]
pub struct FeatureTransformer {
    /// Biases for the hidden layer - 64-byte aligned for SIMD
    pub(crate) biases: AlignedVec<BiasType>,
    /// Weights for PSQ feature transformation (SFNNv18: i8) - 64-byte aligned for SIMD
    pub(crate) weights: AlignedVec<WeightType>,
    /// SFNNv10/SFNNv16: threat + pawn-pair feature weights (i8).
    ///
    /// Threats and pawn-pair features are concatenated into one array so that a
    /// single index addresses either. The first pawn-pair feature is at index
    /// `ThreatFeatureSet::DIMENSIONS` (== `PairFeatureSet::INDEX_BASE`).
    pub(crate) threat_and_pp_weights: AlignedVec<ThreatWeightType>,
}

impl FeatureTransformer {
    // Constants (OutputDimensions == L1 == 1024)
    // Number of output dimensions for one side
    pub const HALF_DIMENSIONS: usize = L1;
    /// K32H3 (piece-square) input dimensions, three contiguous blocks.
    pub const PSQ_INPUT_DIMENSIONS: usize = PSQFeatureSet::DIMENSIONS; // 67584
    pub const THREAT_INPUT_DIMENSIONS: usize = ThreatFeatureSet::DIMENSIONS; // 59808
    /// SFNNv16: pawn-pair input dimensions.
    pub const PAIR_INPUT_DIMENSIONS: usize = PairFeatureSet::DIMENSIONS; // 4560
    /// Concatenated threat + pawn-pair input dimensions (single index space).
    pub const THREAT_AND_PAIR_INPUT_DIMENSIONS: usize =
        Self::THREAT_INPUT_DIMENSIONS + Self::PAIR_INPUT_DIMENSIONS;
    /// C++ `InputDimensions` (all three feature blocks).
    pub const INPUT_DIMENSIONS: usize =
        Self::PSQ_INPUT_DIMENSIONS + Self::THREAT_AND_PAIR_INPUT_DIMENSIONS;
    pub const OUTPUT_DIMENSIONS: usize = L1;
    pub const BUFFER_SIZE: usize = L1;

    // -----------------------------------------------------------------------
    // Shared-memory image
    // -----------------------------------------------------------------------
    //
    // The weights are by far the largest thing the engine allocates (roughly 107 MiB,
    // essentially all of what C++ shares), so they are what gets deduplicated across
    // processes. C++ can map its whole `Network` struct directly because its
    // members are inline `std::array`s; the Rust port keeps the arrays behind
    // `AlignedVec`, so instead of sharing the struct we define an explicit
    // *image*: the three weight arrays laid out back to back, each padded to a
    // cache line so every one of them is legally aligned for SIMD loads.
    //
    // The layer weights (`NetworkArchitecture`) are deliberately left private:
    // all 32 layer stacks together are ~1.1 MiB, i.e. about 1% of the network, and
    // sharing them would mean threading shared backing through four more layer
    // types for no measurable gain.

    /// Alignment of every array inside the shared image.
    const IMAGE_ALIGN: usize = CACHE_LINE_SIZE;

    const fn align_up(x: usize, a: usize) -> usize {
        x.div_ceil(a) * a
    }

    /// Byte length of each array in the image, in layout order.
    const IMAGE_SECTION_BYTES: [usize; 3] = [
        L1 * core::mem::size_of::<BiasType>(),
        L1 * Self::PSQ_INPUT_DIMENSIONS * core::mem::size_of::<WeightType>(),
        L1 * Self::THREAT_AND_PAIR_INPUT_DIMENSIONS * core::mem::size_of::<ThreatWeightType>(),
    ];

    /// Byte offset of section `i` within the image.
    const fn image_offset(i: usize) -> usize {
        let mut off = 0;
        let mut k = 0;
        while k < i {
            off = Self::align_up(off + Self::IMAGE_SECTION_BYTES[k], Self::IMAGE_ALIGN);
            k += 1;
        }
        off
    }

    /// Total byte length of the shared image.
    pub const fn shared_image_len() -> usize {
        Self::align_up(
            Self::image_offset(2) + Self::IMAGE_SECTION_BYTES[2],
            Self::IMAGE_ALIGN,
        )
    }

    /// Version of the shared-image format; bump it when the image layout or the weight
    /// permutation changes in a way the fingerprint below does not capture.
    #[cfg(unix)]
    const IMAGE_FORMAT_VERSION: u64 = 2;

    /// Fingerprint of everything besides the network file that shapes the image bytes: the format
    /// version, the SIMD weight permutations, the image length and the SIMD level of this build.
    /// Mixed into the shared region's key so a build with a different weight layout never
    /// attaches to a region written by another build.
    #[cfg(unix)]
    pub(crate) fn image_layout_key() -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let order = Self::packus_order();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        h = (h ^ Self::IMAGE_FORMAT_VERSION).wrapping_mul(PRIME);
        let mut i = 0;
        while i < order.len() {
            h = (h ^ order[i] as u64).wrapping_mul(PRIME);
            i += 1;
        }
        h = (h ^ Self::shared_image_len() as u64).wrapping_mul(PRIME);
        h = (h ^ cfg!(target_feature = "avx2") as u64).wrapping_mul(PRIME);
        h = (h ^ cfg!(target_feature = "avx512bw") as u64).wrapping_mul(PRIME);
        // The widening pair order depends on the CPU, not only on the build.
        for &o in Self::widening_pair_orders().map_or(&[][..], |(order, _)| order) {
            h = (h ^ ((o as u64 + 1) << 8)).wrapping_mul(PRIME);
        }
        h
    }

    /// Serialize this transformer's weights into `buf` in image layout.
    ///
    /// Only called by the process that creates the shared region.
    pub fn write_shared_image(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), Self::shared_image_len());

        fn put<T: Pod>(buf: &mut [u8], off: usize, src: &[T]) {
            let src_bytes: &[u8] = bytemuck::cast_slice(src);
            buf[off..off + src_bytes.len()].copy_from_slice(src_bytes);
        }

        put(buf, Self::image_offset(0), &self.biases);
        put(buf, Self::image_offset(1), &self.weights);
        put(buf, Self::image_offset(2), &self.threat_and_pp_weights);
    }

    /// Build a transformer whose arrays are windows into `region`.
    ///
    /// No weights are copied: every array points straight into the mapping, so
    /// the hot path reads exactly the pages other processes are reading.
    pub fn from_shared_region(
        region: &std::sync::Arc<crate::shm::SystemWideSharedConstant>,
    ) -> Self {
        Self {
            biases: AlignedVec::from_shared(region, Self::image_offset(0), L1),
            weights: AlignedVec::from_shared(
                region,
                Self::image_offset(1),
                L1 * Self::PSQ_INPUT_DIMENSIONS,
            ),
            threat_and_pp_weights: AlignedVec::from_shared(
                region,
                Self::image_offset(2),
                L1 * Self::THREAT_AND_PAIR_INPUT_DIMENSIONS,
            ),
        }
    }

    /// Hash of the weights, used to address the shared region.
    ///
    /// Content addressing makes implicit sharing safe in practice: two processes only attach
    /// to the same region when their weight hashes match, so a differing net is picked up only
    /// on a 64-bit hash collision. Mirrors C++
    /// `Network::get_content_hash`, which likewise hashes the raw bytes.
    pub fn content_hash(&self) -> u64 {
        fn feed<T: Pod>(h: &mut u64, src: &[T]) {
            let x = crate::misc::hash_bytes(bytemuck::cast_slice(src));
            // Same mixing constant and shifts as C++ `hash_combine`.
            *h ^= x
                .wrapping_add(0x9e37_79b9)
                .wrapping_add(*h << 6)
                .wrapping_add(*h >> 2);
        }

        let mut h: u64 = 0;
        feed(&mut h, &self.biases);
        feed(&mut h, &self.weights);
        feed(&mut h, &self.threat_and_pp_weights);
        h
    }

    /// Create a new feature transformer with zero-initialized weights
    pub fn new() -> Self {
        Self {
            biases: AlignedVec::new_zeroed(L1),
            weights: AlignedVec::new_zeroed(L1 * Self::PSQ_INPUT_DIMENSIONS),
            threat_and_pp_weights: AlignedVec::new_zeroed(
                L1 * Self::THREAT_AND_PAIR_INPUT_DIMENSIONS,
            ),
        }
    }

    // Accessors carving the concatenated threat/pawn-pair arrays into their two halves.
    // Mirror C++ `threatWeights()` / `ppWeights()`; currently unused since (de)serialization
    // splits the array directly.
    const THREAT_WEIGHTS_LEN: usize = Self::THREAT_INPUT_DIMENSIONS * L1;

    #[inline]
    pub fn threat_weights(&self) -> &[ThreatWeightType] {
        &self.threat_and_pp_weights[..Self::THREAT_WEIGHTS_LEN]
    }

    #[inline]
    pub fn pp_weights(&self) -> &[ThreatWeightType] {
        &self.threat_and_pp_weights[Self::THREAT_WEIGHTS_LEN..]
    }

    /// Combine component hashes by rotating the accumulator left by one bit and XOR-ing in
    /// each component. Matches C++ `combine_hash(std::initializer_list<u32>)`.
    const fn combine_hash(hashes: &[u32]) -> u32 {
        let mut hash: u32 = 0;
        let mut i = 0;
        while i < hashes.len() {
            hash = hash.rotate_left(1);
            hash ^= hashes[i];
            i += 1;
        }
        hash
    }

    /// Get hash value for the feature transformer
    /// SFNNv16: combine_hash({ThreatFeatureSet, PairFeatureSet, PSQFeatureSet})
    /// ^ (OutputDimensions * 2).
    ///
    /// Hash value embedded in the evaluation file
    pub fn get_hash_value() -> u32 {
        Self::combine_hash(&[
            ThreatFeatureSet::HASH_VALUE,
            PairFeatureSet::HASH_VALUE,
            PSQFeatureSet::HASH_VALUE,
        ]) ^ (L1 as u32 * 2)
    }

    /// PSQ weights of `block`: the `BLOCK_DIMENSIONS` rows starting at C++
    /// `&weights[block_offset(block) * HalfDimensions]`.
    #[inline(always)]
    pub fn psq_weights(&self, block: usize) -> &[WeightType] {
        const BLOCK_LEN: usize =
            PSQFeatureSet::BLOCK_DIMENSIONS * FeatureTransformer::HALF_DIMENSIONS;
        let start = PSQFeatureSet::block_offset(block) * Self::HALF_DIMENSIONS;
        &self.weights[start..start + BLOCK_LEN]
    }

    /// Read network parameters.
    /// SFNNv18: uncompressed little-endian biases (i16), then i8 threat weights, i8
    /// pawn-pair weights and i8 PSQ weights as independent arrays.
    /// Weights are stored pre-scaled in the network file.
    pub fn read_parameters<R: Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.threat_and_pp_weights
            .resize(Self::THREAT_AND_PAIR_INPUT_DIMENSIONS * L1);
        Self::read_parameters_into(
            reader,
            &mut self.biases,
            &mut self.threat_and_pp_weights,
            &mut self.weights,
        )
    }

    /// Parse the transformer section of a network stream straight into `image`,
    /// a buffer laid out as the shared image (see [`Self::write_shared_image`]).
    ///
    /// This lets the loader populate a shared region directly instead of
    /// parsing into a private copy first and then copying it across.
    pub fn read_parameters_into_image<R: Read>(
        reader: &mut R,
        image: &mut [u8],
    ) -> std::io::Result<()> {
        assert!(image.len() >= Self::shared_image_len());
        // The three sections, in layout order (`image_offset(0)` is 0). Each cast panics if
        // its section is misaligned for the element type.
        let [bias_bytes, weight_bytes, threat_bytes] = Self::IMAGE_SECTION_BYTES;
        let (head, rest) = image.split_at_mut(Self::image_offset(1));
        let (mid, tail) = rest.split_at_mut(Self::image_offset(2) - Self::image_offset(1));
        let biases: &mut [BiasType] = bytemuck::cast_slice_mut(&mut head[..bias_bytes]);
        let weights: &mut [WeightType] = bytemuck::cast_slice_mut(&mut mid[..weight_bytes]);
        let threat_and_pp: &mut [ThreatWeightType] =
            bytemuck::cast_slice_mut(&mut tail[..threat_bytes]);
        Self::read_parameters_into(reader, biases, threat_and_pp, weights)
    }

    fn read_parameters_into<R: Read>(
        reader: &mut R,
        biases: &mut [BiasType],
        threat_and_pp_weights: &mut [ThreatWeightType],
        weights: &mut [WeightType],
    ) -> std::io::Result<()> {
        // Read biases (little-endian i16)
        read_little_endian_i16(reader, biases)?;

        // SFNNv18 format:
        // 1. biases:        little-endian i16 (read above)
        // 2. threatWeights: raw little-endian i8
        // 3. ppWeights:     raw little-endian i8
        // 4. weights:       raw little-endian i8 (PSQ weights only)
        // Threats and pawn pairs live in one concatenated array each, but are
        // serialized as two consecutive blocks.
        let (threat, pp) = threat_and_pp_weights.split_at_mut(Self::THREAT_WEIGHTS_LEN);
        read_little_endian_i8(reader, threat)?;
        read_little_endian_i8(reader, pp)?;

        // Read PSQ weights (i8)
        read_little_endian_i8(reader, weights)?;

        // Permute weights for SIMD (C++ `permute_weights()`). The i8 PSQ and threat
        // weights use block size 8, not 16, and the threats are permuted as one
        // concatenated array (C++ `permute<8>(threatAndPpWeights, ...)`).
        let order = Self::packus_order();
        Self::permute_data(biases, order);
        Self::permute_data_block8(weights, order);
        Self::permute_data_block8(threat_and_pp_weights, order);

        Self::permute_widening_pairs(weights, threat_and_pp_weights, false);

        Ok(())
    }

    /// Write network parameters.
    /// SFNNv18: little-endian i16 biases, i8 threat weights, i8 pawn-pair weights and
    /// i8 PSQ weights.
    pub fn write_parameters<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // SFNNv18 format (matching C++ write order):
        // 1. biases        (little-endian i16)
        // 2. threatWeights (raw little-endian i8)
        // 3. ppWeights     (raw little-endian i8)
        // 4. weights       (raw little-endian i8, PSQ only)

        let mut temp_biases = self.biases.clone();
        let mut temp_weights = self.weights.clone();
        let mut temp_threat_and_pp_weights = self.threat_and_pp_weights.clone();

        Self::unpermute_weights_static_with_threats(
            &mut temp_biases,
            &mut temp_weights,
            &mut temp_threat_and_pp_weights,
        );

        let (temp_threat_weights, temp_pp_weights) =
            temp_threat_and_pp_weights.split_at(Self::THREAT_WEIGHTS_LEN);

        write_little_endian_i16(writer, &temp_biases)?;
        write_little_endian_i8(writer, temp_threat_weights)?;
        write_little_endian_i8(writer, temp_pp_weights)?;
        write_little_endian_i8(writer, &temp_weights)?;

        Ok(())
    }

    /// Transform features for the given position
    ///
    /// The accumulator holds the combined PSQ and threat sums in a single `accumulation`;
    /// transform clips and pairwise-multiplies it.
    pub fn transform(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: Option<&mut AccumulatorCaches>,
        transformed_features: &mut [TransformedFeatureType],
    ) {
        self.transform_uninit(
            pos,
            accumulator_stack,
            cache,
            as_uninit_bytes(transformed_features),
        )
    }

    /// Like [`FeatureTransformer::transform`] but the output buffer may be uninitialised.
    /// Every one of the first `L1` bytes of `transformed_features` is written (and only ever
    /// with initialised values), so after the call they may be treated as initialised.
    #[inline(always)]
    pub(crate) fn transform_uninit(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: Option<&mut AccumulatorCaches>,
        transformed_features: &mut [MaybeUninit<TransformedFeatureType>],
    ) {
        self.transform_impl::<false>(pos, accumulator_stack, cache, transformed_features, None)
    }

    /// AVX-512 entry point that records the non-zero transformed-feature
    /// chunks for reuse by the first sparse affine layer.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[inline(always)]
    pub(crate) fn transform_with_nnz(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: Option<&mut AccumulatorCaches>,
        transformed_features: &mut [MaybeUninit<TransformedFeatureType>],
        nnz: &mut NnzInfo<256>,
    ) {
        self.transform_impl::<true>(
            pos,
            accumulator_stack,
            cache,
            transformed_features,
            Some(nnz),
        )
    }

    #[inline(always)]
    fn transform_impl<const ALIGNED: bool>(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: Option<&mut AccumulatorCaches>,
        transformed_features: &mut [MaybeUninit<TransformedFeatureType>],
        mut nnz: Option<&mut NnzInfo<256>>,
    ) {
        assert!(
            transformed_features.len() >= L1,
            "feature output needs L1 elements"
        );
        if let Some(info) = nnz.as_mut() {
            info.clear();
        }
        // With a cache, bring the accumulator stack up to date for `pos`; without one the
        // accumulator is used as-is (the caller guarantees it is current).
        if let Some(cache) = cache {
            accumulator_stack.evaluate(pos, self, cache);
        }
        let accumulator_state = accumulator_stack.latest();

        // Get perspectives for both players
        let perspectives = [pos.side_to_move(), !pos.side_to_move()];

        // The combined accumulation (HalfKA + Threats) is exposed via `Deref` on
        // `AccumulatorState`, so we read `accumulation` directly.
        let accumulation = &accumulator_state.accumulation;

        // Process features for both perspectives
        for p in 0..2 {
            let element_offset = (L1 / 2) * p;
            let persp = perspectives[p] as usize;

            #[cfg(target_arch = "aarch64")]
            {
                self.transform_neon_uninit(
                    &accumulation[persp],
                    &mut transformed_features[element_offset..],
                    L1 / 2,
                );
            }
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ))]
            {
                let output = &mut transformed_features[element_offset..];
                // SAFETY: AVX-512F/BW are enabled by the enclosing `cfg(target_feature)`. Each
                // perspective's `[i16; L1]` accumulation and the `L1 / 2` output bytes at
                // `element_offset` cover every access (`half_dims = L1 / 2`; the length assert at
                // the top of `transform_impl` bounds the output). `ALIGNED == true` (only via
                // `transform_with_nnz`) needs 64-byte alignment: accumulations are
                // `repr(align(64))`, `element_offset` is a multiple of 64 and the output's base
                // alignment is debug-asserted at the top of `transform_impl` (the callers'
                // `AlignedTransformedFeatures` is `repr(align(64))`); the `ALIGNED == false` path
                // uses unaligned accesses.
                unsafe {
                    if ALIGNED {
                        self.transform_avx512::<true>(
                            &accumulation[persp],
                            output,
                            L1 / 2,
                            element_offset,
                            nnz.as_deref_mut(),
                        );
                    } else {
                        self.transform_avx512::<false>(
                            &accumulation[persp],
                            output,
                            L1 / 2,
                            element_offset,
                            nnz.as_deref_mut(),
                        );
                    }
                }
            }
            #[cfg(any(
                target_arch = "x86",
                all(target_arch = "x86_64", not(target_feature = "avx512f")),
                all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    not(target_feature = "avx512bw")
                )
            ))]
            {
                self.transform_simd(
                    &accumulation[persp],
                    &mut transformed_features[element_offset..],
                    L1 / 2,
                );
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
            {
                self.transform_scalar(
                    &accumulation[persp],
                    &mut transformed_features[element_offset..],
                    L1 / 2,
                );
            }
        }
    }

    #[cfg(any(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        all(test, target_arch = "x86_64", target_feature = "avx2")
    ))]
    /// Scalar implementation of feature transformation (targets without an x86_64/aarch64
    /// kernel, and the AVX-512 reference test). Applies clipped ReLU and outputs packed u8
    /// values matching C++ SIMD behavior.
    fn transform_scalar(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
    ) {
        // TransformedFeatureType is u8, so we write directly to output
        for j in 0..half_dims {
            // Get values from two halves of accumulation
            let sum0 = accumulation[j] as i32;
            let sum1 = accumulation[j + half_dims] as i32;

            // Clamp to [0, FtMaxVal] (SF18: FtMaxVal == 255)
            let sum0 = sum0.clamp(0, FT_MAX_VAL);
            let sum1 = sum1.clamp(0, FT_MAX_VAL);

            // Multiply and divide by 512.
            // Result is in range [0, 127] which fits in u8
            output[j].write(((sum0 * sum1) / 512) as u8);
        }
    }

    /// SIMD implementation of feature transformation (x86_64)
    ///
    /// Per the NNUE architecture, the kernels below multiply pairs of clipped
    /// elements and divide the product by 512. Naively that means clipping four
    /// int16 vectors with min/max, `mullo`-ing the pairs together and packing
    /// the results into one int8 vector — but there is a faster way.
    ///
    /// `mulhi` cuts off the low 16 bits of the product, i.e. it right-shifts by
    /// 16. Dividing by 512 is a 9-bit right shift, so shifting one clipped
    /// operand left by 7 and then taking `mulhi` nets exactly the right shift by
    /// 9. Only the lower clamp of the second operand is left over, and the final
    /// signed-to-unsigned `packus` supplies that by flushing negatives to zero.
    #[cfg(any(
        target_arch = "x86",
        all(
            target_arch = "x86_64",
            not(all(target_feature = "avx512f", target_feature = "avx512bw"))
        )
    ))]
    fn transform_simd(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
    ) {
        assert!(accumulation.len() >= 2 * half_dims && output.len() >= half_dims);

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        // SAFETY: AVX2 is enabled by the enclosing `cfg(target_feature)`. `accumulation` has
        // `2 * half_dims` entries and `output` at least `half_dims` bytes (asserted above).
        unsafe {
            self.transform_avx2(accumulation, output, half_dims);
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            not(target_feature = "avx2")
        ))]
        // SAFETY: SSE2 is enabled by the enclosing `cfg(target_feature)`; slice lengths as above.
        unsafe {
            self.transform_sse2(accumulation, output, half_dims);
        }

        // 32-bit x86 (x86_64 always has SSE2).
        #[cfg(not(target_arch = "x86_64"))]
        self.transform_scalar(accumulation, output, half_dims);
    }

    /// AVX-512 implementation. The 512-bit pack has a different lane order
    /// from AVX2; `read_parameters` pre-permutes the accumulator columns to
    /// compensate, exactly as Stockfish's `USE_AVX512` transformer does.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[inline(always)]
    unsafe fn transform_avx512<const ALIGNED: bool>(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
        output_offset: usize,
        mut nnz: Option<&mut NnzInfo<256>>,
    ) {
        use core::arch::x86_64::*;

        const CHUNK_SIZE: usize = 64;
        let chunks = half_dims / CHUNK_SIZE;
        let zero = _mm512_setzero_si512();
        let ft_max = _mm512_set1_epi16(FT_MAX_VAL as i16);
        let mut nnz_indices = _mm512_setr_epi32(
            (output_offset / 4) as i32,
            (output_offset / 4 + 1) as i32,
            (output_offset / 4 + 2) as i32,
            (output_offset / 4 + 3) as i32,
            (output_offset / 4 + 4) as i32,
            (output_offset / 4 + 5) as i32,
            (output_offset / 4 + 6) as i32,
            (output_offset / 4 + 7) as i32,
            (output_offset / 4 + 8) as i32,
            (output_offset / 4 + 9) as i32,
            (output_offset / 4 + 10) as i32,
            (output_offset / 4 + 11) as i32,
            (output_offset / 4 + 12) as i32,
            (output_offset / 4 + 13) as i32,
            (output_offset / 4 + 14) as i32,
            (output_offset / 4 + 15) as i32,
        );
        let nnz_increment = _mm512_set1_epi32(16);
        let mut nnz_count = nnz.as_ref().map_or(0, |info| info.count() as u32);

        for chunk in 0..chunks {
            let offset = chunk * CHUNK_SIZE;
            let acc0a = if ALIGNED {
                _mm512_load_si512(accumulation.as_ptr().add(offset) as *const __m512i)
            } else {
                _mm512_loadu_si512(accumulation.as_ptr().add(offset) as *const __m512i)
            };
            let acc0b = if ALIGNED {
                _mm512_load_si512(accumulation.as_ptr().add(offset + 32) as *const __m512i)
            } else {
                _mm512_loadu_si512(accumulation.as_ptr().add(offset + 32) as *const __m512i)
            };
            let acc1a = if ALIGNED {
                _mm512_load_si512(accumulation.as_ptr().add(offset + half_dims) as *const __m512i)
            } else {
                _mm512_loadu_si512(accumulation.as_ptr().add(offset + half_dims) as *const __m512i)
            };
            let acc1b = if ALIGNED {
                _mm512_load_si512(
                    accumulation.as_ptr().add(offset + half_dims + 32) as *const __m512i
                )
            } else {
                _mm512_loadu_si512(
                    accumulation.as_ptr().add(offset + half_dims + 32) as *const __m512i
                )
            };

            let sum0a =
                _mm512_slli_epi16::<7>(_mm512_max_epi16(_mm512_min_epi16(acc0a, ft_max), zero));
            let sum0b =
                _mm512_slli_epi16::<7>(_mm512_max_epi16(_mm512_min_epi16(acc0b, ft_max), zero));
            // `packus` supplies the lower clamp for this operand: a negative
            // product becomes zero during the final signed-to-unsigned pack.
            let sum1a = _mm512_min_epi16(acc1a, ft_max);
            let sum1b = _mm512_min_epi16(acc1b, ft_max);
            let product_a = _mm512_mulhi_epi16(sum0a, sum1a);
            let product_b = _mm512_mulhi_epi16(sum0b, sum1b);
            let packed = _mm512_packus_epi16(product_a, product_b);
            if ALIGNED {
                _mm512_store_si512(output.as_mut_ptr().add(offset) as *mut __m512i, packed);
            } else {
                _mm512_storeu_si512(output.as_mut_ptr().add(offset) as *mut __m512i, packed);
            }
            if let Some(info) = nnz.as_mut() {
                // SAFETY: AVX-512F/BW are enabled by this fn's cfg. `record_avx512` does a 16-lane
                // store at `nnz_count`; this stays inside the 256-entry `NnzInfo` because at most
                // 16 vectors of 16 lanes are recorded (8 chunks per perspective, 2 perspectives),
                // so `nnz_count <= 240` before each call.
                nnz_count = unsafe { (**info).record_avx512(nnz_count, packed, nnz_indices) };
                nnz_indices = _mm512_add_epi32(nnz_indices, nnz_increment);
            }
        }

        if let Some(info) = nnz.as_mut() {
            // SAFETY: `nnz_count` is the value returned by the `record_avx512` calls above, so
            // `indices[0..nnz_count]` is initialized and `nnz_count <= 256 == CAPACITY` (the last
            // `record_avx512` starts at `<= 240` and writes 16 lanes).
            unsafe { info.set_count(nnz_count) };
        }

        for j in chunks * CHUNK_SIZE..half_dims {
            let sum0 = (accumulation[j] as i32).clamp(0, FT_MAX_VAL);
            let sum1 = (accumulation[j + half_dims] as i32).clamp(0, FT_MAX_VAL);
            output[j].write(((sum0 * sum1) / 512) as u8);
        }
    }

    /// AVX2 implementation - outputs packed bytes like C++ SIMD.
    ///
    /// A YMM register holds sixteen i16 lanes, while `packus` produces a
    /// thirty-two-byte output chunk from two adjacent registers.  Keeping the
    /// two loads together is what lets one `packus` produce a full 32-byte chunk.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    #[inline(always)]
    unsafe fn transform_avx2(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
    ) {
        use core::arch::x86_64::*;

        // Get byte pointer for output (we write packed bytes, not int16)
        let output_bytes = output.as_mut_ptr().cast::<TransformedFeatureType>();

        const CHUNK_SIZE: usize = 32;
        let chunks = half_dims / CHUNK_SIZE;

        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi16(FT_MAX_VAL as i16); // SF18: FtMaxVal == 255
        const SHIFT: i32 = 7;

        for j in 0..chunks {
            let offset = j * CHUNK_SIZE;

            let acc0a = _mm256_loadu_si256(accumulation.as_ptr().add(offset) as *const __m256i);
            let acc0b =
                _mm256_loadu_si256(accumulation.as_ptr().add(offset + 16) as *const __m256i);
            let acc1a =
                _mm256_loadu_si256(accumulation.as_ptr().add(offset + half_dims) as *const __m256i);
            let acc1b = _mm256_loadu_si256(
                accumulation.as_ptr().add(offset + half_dims + 16) as *const __m256i
            );

            let sum0a =
                _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(acc0a, one), zero), SHIFT);
            let sum0b =
                _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(acc0b, one), zero), SHIFT);
            let sum1a = _mm256_min_epi16(acc1a, one);
            let sum1b = _mm256_min_epi16(acc1b, one);
            let product_a = _mm256_mulhi_epi16(sum0a, sum1a);
            let product_b = _mm256_mulhi_epi16(sum0b, sum1b);

            let packed = _mm256_packus_epi16(product_a, product_b);
            _mm256_storeu_si256(output_bytes.add(offset) as *mut __m256i, packed);
        }

        // Handle remaining elements - write as bytes
        let remaining = half_dims % CHUNK_SIZE;
        if remaining > 0 {
            let offset = chunks * CHUNK_SIZE;
            for j in 0..remaining {
                let sum0 = accumulation[offset + j] as i32;
                let sum1 = accumulation[offset + j + half_dims] as i32;
                let sum0 = sum0.clamp(0, FT_MAX_VAL);
                let sum1 = sum1.clamp(0, FT_MAX_VAL);
                let result = ((sum0 * sum1) / 512) as u8;
                *output_bytes.add(offset + j) = result;
            }
        }
    }

    /// SSE2 implementation - outputs packed bytes like C++ SIMD.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx2")
    ))]
    #[inline(always)]
    unsafe fn transform_sse2(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
    ) {
        use core::arch::x86_64::*;

        // Get byte pointer for output (we write packed bytes, not int16)
        let output_bytes = output.as_mut_ptr().cast::<TransformedFeatureType>();

        const CHUNK_SIZE: usize = 16;
        let chunks = half_dims / CHUNK_SIZE;

        let zero = _mm_setzero_si128();
        let one = _mm_set1_epi16(FT_MAX_VAL as i16); // SF18: FtMaxVal == 255
        const SHIFT: i32 = 7;

        for j in 0..chunks {
            let offset = j * CHUNK_SIZE;

            let acc0a = _mm_loadu_si128(accumulation.as_ptr().add(offset) as *const __m128i);
            let acc0b = _mm_loadu_si128(accumulation.as_ptr().add(offset + 8) as *const __m128i);
            let acc1a =
                _mm_loadu_si128(accumulation.as_ptr().add(offset + half_dims) as *const __m128i);
            let acc1b = _mm_loadu_si128(
                accumulation.as_ptr().add(offset + half_dims + 8) as *const __m128i
            );

            let sum0a = _mm_slli_epi16(_mm_max_epi16(_mm_min_epi16(acc0a, one), zero), SHIFT);
            let sum0b = _mm_slli_epi16(_mm_max_epi16(_mm_min_epi16(acc0b, one), zero), SHIFT);
            let sum1a = _mm_min_epi16(acc1a, one);
            let sum1b = _mm_min_epi16(acc1b, one);
            let product_a = _mm_mulhi_epi16(sum0a, sum1a);
            let product_b = _mm_mulhi_epi16(sum0b, sum1b);

            let packed = _mm_packus_epi16(product_a, product_b);
            _mm_storeu_si128(output_bytes.add(offset) as *mut __m128i, packed);
        }

        // Handle remaining elements - write as bytes
        let remaining = half_dims % CHUNK_SIZE;
        if remaining > 0 {
            let offset = chunks * CHUNK_SIZE;
            for j in 0..remaining {
                let sum0 = accumulation[offset + j] as i32;
                let sum1 = accumulation[offset + j + half_dims] as i32;
                let sum0 = sum0.clamp(0, FT_MAX_VAL);
                let sum1 = sum1.clamp(0, FT_MAX_VAL);
                let result = ((sum0 * sum1) / 512) as u8;
                *output_bytes.add(offset + j) = result;
            }
        }
    }

    /// NEON implementation for ARM - outputs packed bytes like C++ SIMD.
    ///
    /// SF18 algorithm (matches the `USE_NEON` path in nnue_feature_transformer.h):
    /// saturate each i16 accumulation lane to u8 in [0, 255] via `vqmovun_s16` (which is
    /// exactly the FtMaxVal=255 clip plus the lower clip to 0), widen-multiply the two
    /// halves with `vmull_u8`, then take the high byte of each u16 product (`vuzp2q_u8`)
    /// and shift right by one to realize the divide by 512 (`>>8` from the high byte and
    /// `>>1` from the shift).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn transform_neon(
        &self,
        accumulation: &[BiasType],
        output: &mut [TransformedFeatureType],
        half_dims: usize,
    ) {
        self.transform_neon_uninit(accumulation, as_uninit_bytes(output), half_dims)
    }

    /// [`FeatureTransformer::transform_neon`] writing into a possibly uninitialised buffer.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) fn transform_neon_uninit(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
    ) {
        use core::arch::aarch64::*;

        // Get byte pointer for output (we write packed bytes, not int16)
        let output_bytes = output.as_mut_ptr().cast::<TransformedFeatureType>();

        // Process 16 elements at a time (2x8), matching C++ NumOutputChunks
        const OUTPUT_CHUNK_SIZE: usize = 16;
        let num_chunks = half_dims / OUTPUT_CHUNK_SIZE;

        assert!(accumulation.len() >= 2 * half_dims && output.len() >= half_dims);

        // SAFETY: NEON is baseline on aarch64. The loads read `accumulation[..2 * half_dims]` and
        // the 16-byte stores write `output[..num_chunks * 16]` with `num_chunks * 16 <= half_dims`;
        // both lengths were asserted above.
        unsafe {
            let in0_ptr = accumulation.as_ptr();
            let in1_ptr = accumulation.as_ptr().add(half_dims);

            for j in 0..num_chunks {
                // Load two vectors of 8 elements each from first half
                let acc0a = vld1q_s16(in0_ptr.add(j * 16));
                let acc0b = vld1q_s16(in0_ptr.add(j * 16 + 8));

                // Load two vectors of 8 elements each from second half
                let acc1a = vld1q_s16(in1_ptr.add(j * 16));
                let acc1b = vld1q_s16(in1_ptr.add(j * 16 + 8));

                // Saturate to u8 [0, 255] then widening-multiply the pairs.
                let mul0 = vmull_u8(vqmovun_s16(acc0a), vqmovun_s16(acc1a));
                let mul1 = vmull_u8(vqmovun_s16(acc0b), vqmovun_s16(acc1b));

                // Take the high byte of each little-endian u16 product (== product >> 8)
                // and shift right by one more (net product / 512).
                let hi = vuzp2q_u8(vreinterpretq_u8_u16(mul0), vreinterpretq_u8_u16(mul1));
                let result = vshrq_n_u8(hi, 1);

                // Store 16 bytes
                vst1q_u8(output_bytes.add(j * 16), result);
            }

            // Handle remaining elements (scalar fallback)
            let remaining_start = num_chunks * OUTPUT_CHUNK_SIZE;
            for k in remaining_start..half_dims {
                let s0 = (accumulation[k] as i32).clamp(0, FT_MAX_VAL);
                let s1 = (accumulation[k + half_dims] as i32).clamp(0, FT_MAX_VAL);
                let result = ((s0 * s1) / 512) as u8;
                *output_bytes.add(k) = result;
            }
        }
    }

    /// AArch64 entry point that also records the non-zero-chunk bitset consumed by the first
    /// sparse affine layer, as the C++ `transform` does through `NNZInfo::make_cursor`
    /// (nnz_helper.h): perspective `p` owns bitset bytes `p * L1 / 64 ..`, one byte per 32
    /// transformed features (eight four-byte chunks).
    ///
    /// Every one of the first `L1` bytes of `transformed_features` and every byte of `bitset`
    /// is written, so after the call both may be treated as initialised.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) fn transform_with_bitset(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
        transformed_features: &mut [MaybeUninit<TransformedFeatureType>],
        bitset: &mut [MaybeUninit<u8>; L1 / 32],
    ) {
        assert!(
            transformed_features.len() >= L1,
            "feature output needs L1 elements"
        );
        accumulator_stack.evaluate(pos, self, cache);
        let accumulator_state = accumulator_stack.latest();

        let perspectives = [pos.side_to_move(), !pos.side_to_move()];
        let accumulation = &accumulator_state.accumulation;

        for p in 0..2 {
            self.transform_neon_bitset(
                &accumulation[perspectives[p] as usize],
                &mut transformed_features[(L1 / 2) * p..],
                L1 / 2,
                &mut bitset[(L1 / 64) * p..],
            );
        }
    }

    /// [`FeatureTransformer::transform_neon_uninit`] for one perspective, additionally recording
    /// the non-zero-chunk bitset: the C++ `USE_NEON` transform loop with `NNZCursor::record2`
    /// (nnz_helper.h, `USE_NEON >= 8`). Two 16-byte output vectors are produced per iteration
    /// and bit `i` of `bitset[j]` is set iff output bytes `32 * j + 4 * i .. + 4` are not all
    /// zero. `half_dims` must be a multiple of 32 and `bitset` holds `half_dims / 32` bytes.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(crate) fn transform_neon_bitset(
        &self,
        accumulation: &[BiasType],
        output: &mut [MaybeUninit<TransformedFeatureType>],
        half_dims: usize,
        bitset: &mut [MaybeUninit<u8>],
    ) {
        use core::arch::aarch64::*;

        // Lane 2k of the interleaved test vector is chunk k of the first output vector and lane
        // 2k + 1 chunk k of the second, so the mask maps them to bits k and k + 4.
        static MASK8: [u16; 8] = [1, 16, 2, 32, 4, 64, 8, 128];

        assert!(
            half_dims.is_multiple_of(32)
                && accumulation.len() >= 2 * half_dims
                && output.len() >= half_dims
                && bitset.len() >= half_dims / 32
        );

        let output_bytes = output.as_mut_ptr().cast::<TransformedFeatureType>();
        let bits_out = bitset.as_mut_ptr().cast::<u8>();

        // SAFETY: NEON is baseline on aarch64. The loads read `accumulation[..2 * half_dims]`,
        // the 16-byte stores write `output[..half_dims]` and the byte stores write
        // `bitset[..half_dims / 32]`; all three lengths were asserted above.
        unsafe {
            let in0_ptr = accumulation.as_ptr();
            let in1_ptr = accumulation.as_ptr().add(half_dims);
            let mask8 = vld1q_u16(MASK8.as_ptr());

            for j in 0..half_dims / 32 {
                let mut neurons = [vdupq_n_u8(0); 2];
                for k in 0..2 {
                    let i = (j * 2 + k) * 16;
                    let acc0a = vld1q_s16(in0_ptr.add(i));
                    let acc0b = vld1q_s16(in0_ptr.add(i + 8));
                    let acc1a = vld1q_s16(in1_ptr.add(i));
                    let acc1b = vld1q_s16(in1_ptr.add(i + 8));

                    // Same arithmetic as `transform_neon_uninit`.
                    let mul0 = vmull_u8(vqmovun_s16(acc0a), vqmovun_s16(acc1a));
                    let mul1 = vmull_u8(vqmovun_s16(acc0b), vqmovun_s16(acc1b));
                    let hi = vuzp2q_u8(vreinterpretq_u8_u16(mul0), vreinterpretq_u8_u16(mul1));
                    let result = vshrq_n_u8(hi, 1);

                    vst1q_u8(output_bytes.add(i), result);
                    neurons[k] = result;
                }

                // NNZCursor::record2
                let n1 = vreinterpretq_u32_u8(neurons[0]);
                let n2 = vreinterpretq_u32_u8(neurons[1]);
                let t1 = vtstq_u32(n1, n1);
                let t2 = vtstq_u32(n2, n2);
                let packed = vtrn1q_u16(vreinterpretq_u16_u32(t1), vreinterpretq_u16_u32(t2));
                let bits = vandq_u16(packed, mask8);
                *bits_out.add(j) = vaddvq_u16(bits) as u8;
            }
        }
    }

    /// Get the permutation order for packus_epi16 optimization based on SIMD width
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    const fn packus_order() -> [usize; 8] {
        // AVX-512: after _mm512_packus_epi16 permutation
        [0, 2, 4, 6, 1, 3, 5, 7]
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    const fn packus_order() -> [usize; 8] {
        // AVX2: after _mm256_packus_epi16 permutation
        [0, 2, 1, 3, 4, 6, 5, 7]
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    const fn packus_order() -> [usize; 8] {
        // No permutation needed for scalar/SSE/NEON
        [0, 1, 2, 3, 4, 5, 6, 7]
    }

    /// Permute data by reordering blocks within chunks
    /// block_size: 16 for biases (i16), 8 for the i8 PSQ and threat weights
    fn permute_data_with_block_size<T: Pod>(data: &mut [T], order: &[usize], block_size: usize) {
        let chunk_size = block_size * order.len();

        let total_bytes = core::mem::size_of_val(data);

        // Check if permutation is identity (no-op)
        let is_identity = order.iter().enumerate().all(|(i, &o)| i == o);
        if is_identity || total_bytes < chunk_size {
            return;
        }

        // Process each chunk
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(data);

        let mut buffer = vec![0u8; chunk_size];

        for chunk_start in (0..total_bytes).step_by(chunk_size) {
            if chunk_start + chunk_size > total_bytes {
                break; // Don't process partial chunks
            }

            let chunk = &mut bytes[chunk_start..chunk_start + chunk_size];

            // Reorder blocks into buffer according to permutation order
            for (j, &src_idx) in order.iter().enumerate() {
                let src_offset = src_idx * block_size;
                let dst_offset = j * block_size;
                buffer[dst_offset..dst_offset + block_size]
                    .copy_from_slice(&chunk[src_offset..src_offset + block_size]);
            }

            // Copy buffer back to chunk
            chunk.copy_from_slice(&buffer);
        }
    }

    /// Permute data by reordering 16-byte blocks within each 128-byte chunk
    fn permute_data<T: Pod>(data: &mut [T], order: [usize; 8]) {
        Self::permute_data_with_block_size(data, &order, 16);
    }

    /// Permute data by reordering 8-byte blocks within each 64-byte chunk
    /// Used for the PSQ and threat weights (i8 elements)
    fn permute_data_block8<T: Pod>(data: &mut [T], order: [usize; 8]) {
        Self::permute_data_with_block_size(data, &order, 8);
    }

    /// C++ `WideningPairOrder` for `NUM_BLOCKS` 8-byte blocks per vector register.
    #[cfg(target_arch = "x86_64")]
    const fn widening_pair_order<const NUM_BLOCKS: usize>() -> [usize; NUM_BLOCKS] {
        let mut order = [0; NUM_BLOCKS];
        let mut i = 0;
        while i < NUM_BLOCKS {
            order[i] = if i % 2 == 0 {
                i / 2
            } else {
                NUM_BLOCKS / 2 + i / 2
            };
            i += 1;
        }
        order
    }

    /// C++ `invert_permutation`.
    const fn invert_permutation<const N: usize>(order: [usize; N]) -> [usize; N] {
        let mut inverse = [0; N];
        let mut i = 0;
        while i < N {
            inverse[order[i]] = i;
            i += 1;
        }
        inverse
    }

    /// The int8 weight rows are consumed one full vector register (`size_of::<vec_t>()`
    /// bytes) at a time: unpack{lo,hi}_epi8 interleave the bytes of two rows and
    /// maddubs against a vector of ones sums each pair into an int16 lane. Both
    /// unpacks work within 128-bit lanes, so the int16 results of a chunk come out
    /// with its 8-byte blocks in the order {0, 2, 4, 6, ...} (lo) and {1, 3, 5, 7,
    /// ...} (hi). Storing the 8-byte blocks of every chunk in this order at load
    /// time makes the two output registers hold consecutive accumulator lanes, so
    /// the accumulator layout is unchanged.
    ///
    /// Returns `(WideningPairOrder, InverseWideningPairOrder)`, or `None` when rows are read in
    /// file order. C++ fixes the vector width at compile time (`USE_PAIRED_WIDENING_ADD`); the
    /// Rust accumulator kernels are chosen at run time (AVX-512, then AVX2, then scalar, see
    /// `nnue_accumulator.rs`), so the order follows the same detection. Every x86 kernel that
    /// reads the rows widens them with the unpacks, so it matches this order.
    pub(crate) fn widening_pair_orders() -> Option<(&'static [usize], &'static [usize])> {
        #[cfg(target_arch = "x86_64")]
        {
            const ORDER_512: [usize; 8] = FeatureTransformer::widening_pair_order::<8>();
            const INVERSE_512: [usize; 8] = FeatureTransformer::invert_permutation(ORDER_512);
            const ORDER_256: [usize; 4] = FeatureTransformer::widening_pair_order::<4>();
            const INVERSE_256: [usize; 4] = FeatureTransformer::invert_permutation(ORDER_256);
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                return Some((&ORDER_512, &INVERSE_512));
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Some((&ORDER_256, &INVERSE_256));
            }
        }
        None
    }

    /// Apply the [`Self::widening_pair_orders`] permutation (`inverse` undoes it) to the i8
    /// PSQ and threat weight rows.
    pub(crate) fn permute_widening_pairs(
        weights: &mut [WeightType],
        threat_and_pp_weights: &mut [ThreatWeightType],
        inverse: bool,
    ) {
        if let Some((order, inverse_order)) = Self::widening_pair_orders() {
            let order = if inverse { inverse_order } else { order };
            Self::permute_data_with_block_size(weights, order, 8);
            Self::permute_data_with_block_size(threat_and_pp_weights, order, 8);
        }
    }

    /// Unpermute weights including the concatenated threat + pawn-pair weights
    fn unpermute_weights_static_with_threats(
        biases: &mut [BiasType],
        weights: &mut [WeightType],
        threat_and_pp_weights: &mut [ThreatWeightType],
    ) {
        Self::permute_widening_pairs(weights, threat_and_pp_weights, true);

        let order = Self::packus_order();
        let inverse = Self::invert_permutation(order);
        Self::permute_data(biases, inverse);
        Self::permute_data_block8(weights, inverse);
        Self::permute_data_block8(threat_and_pp_weights, inverse);
    }
}

/// Views initialised bytes as possibly-uninitialised ones so they can be passed to the
/// `MaybeUninit` kernels.
#[inline(always)]
pub(crate) fn as_uninit_bytes(bytes: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`. The kernels only ever store
    // initialised bytes through the result, so the original `u8`s stay initialised.
    unsafe { &mut *(bytes as *mut [u8] as *mut [MaybeUninit<u8>]) }
}

impl Default for FeatureTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)] // only the per-ISA SIMD tests below use it
    use super::*;

    /// `transform_neon_bitset` must write the same bytes as the scalar definition and as
    /// `transform_neon`, and exactly the bits of the non-zero four-byte chunks.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_bitset_transform_matches_scalar_reference() {
        let ft = FeatureTransformer::new();
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..8 {
            // Rounds differ in how many products vanish: later rounds leave long zero runs,
            // so empty chunks, single non-zero chunks and full bytes all occur.
            let mut raw = [0i16; L1];
            for value in raw.iter_mut() {
                *value = if next() % 8 < round {
                    -((next() % 600) as i16)
                } else {
                    (next() % 1201) as i16 - 300
                };
            }
            let mut expected = [0u8; L1 / 2];
            for k in 0..L1 / 2 {
                let s0 = (raw[k] as i32).clamp(0, FT_MAX_VAL);
                let s1 = (raw[k + L1 / 2] as i32).clamp(0, FT_MAX_VAL);
                expected[k] = ((s0 * s1) / 512) as u8;
            }
            let mut expected_bits = [0u8; L1 / 64];
            for chunk in 0..L1 / 8 {
                if expected[chunk * 4..chunk * 4 + 4].iter().any(|&b| b != 0) {
                    expected_bits[chunk / 8] |= 1 << (chunk % 8);
                }
            }

            let mut plain = [0u8; L1 / 2];
            ft.transform_neon(&raw, &mut plain, L1 / 2);
            assert_eq!(plain, expected, "round {round}");

            let mut got = [0xa5u8; L1 / 2];
            let mut bits = [MaybeUninit::new(0xa5u8); L1 / 64];
            ft.transform_neon_bitset(&raw, as_uninit_bytes(&mut got), L1 / 2, &mut bits);
            assert_eq!(got, expected, "round {round}");
            // SAFETY: the array was created initialised.
            let bits = bits.map(|b| unsafe { b.assume_init() });
            assert_eq!(bits, expected_bits, "round {round}");
            if round > 0 {
                assert!(bits.iter().any(|&b| b != 0xff) && bits.iter().any(|&b| b != 0));
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    #[test]
    fn avx2_transform_matches_unpermuted_scalar_reference() {
        let ft = FeatureTransformer::new();
        let mut raw = [0i16; L1];
        for (i, value) in raw.iter_mut().enumerate() {
            *value = ((i as i32 * 113 % 1201) - 600) as i16;
        }
        let mut permuted = raw;
        FeatureTransformer::permute_data(&mut permuted, FeatureTransformer::packus_order());
        let mut expected = [0u8; L1 / 2];
        let mut got = [0u8; L1 / 2];
        ft.transform_scalar(&raw, as_uninit_bytes(&mut expected), L1 / 2);
        // SAFETY: AVX2 is enabled at compile time; the input and output contain
        // all L1 and L1/2 elements accessed by the unaligned kernel.
        unsafe { ft.transform_avx2(&permuted, as_uninit_bytes(&mut got), L1 / 2) };
        assert_eq!(got, expected);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn avx512_transform_matches_unpermuted_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512bw")
        {
            return;
        }

        let ft = FeatureTransformer::new();
        let mut raw = [0i16; L1];
        for (i, value) in raw.iter_mut().enumerate() {
            *value = ((i as i32 * 113 % 1201) - 600) as i16;
        }
        let mut permuted = raw;
        FeatureTransformer::permute_data(&mut permuted, FeatureTransformer::packus_order());

        let mut expected = [0u8; L1 / 2];
        let mut got = [0u8; L1 / 2];
        ft.transform_scalar(&raw, as_uninit_bytes(&mut expected), L1 / 2);
        // SAFETY: the test is cfg-gated on avx512f/avx512bw (compile time) and returns early unless
        // `is_x86_feature_detected!` confirms both at runtime; `permuted` and `got` have the L1 and L1/2 lengths the kernel expects.
        unsafe {
            ft.transform_avx512::<false>(&permuted, as_uninit_bytes(&mut got), L1 / 2, 0, None)
        };
        assert_eq!(got, expected);
    }
    /// The AVX-512 non-zero chunk collector must identify the same four-byte
    /// chunks as a scalar scan, and those indices must drive fc_0 to the same
    /// output as a dense matrix multiplication. The selected counts exercise
    /// empty/full masks, both perspectives, and all three VNNI tail lengths.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn avx512_nnz_and_sparse_layer_match_scalar() {
        use crate::nnue::layers::AffineTransformSparseInput;

        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512bw")
        {
            return;
        }

        #[repr(align(64))]
        struct AlignedAccumulator([i16; L1]);
        #[repr(align(64))]
        struct AlignedOutput([u8; L1]);

        let ft = FeatureTransformer::new();
        let mut layer = AffineTransformSparseInput::<1024, 32>::new();
        // Network-file order: 32 i32 biases followed by 32 * 1024 i8 weights.
        let mut params = vec![0u8; 32 * 4];
        params.extend((0..32 * 1024).map(|i| ((i * 13 % 7) as i8 - 3) as u8));
        layer.read_parameters(&mut &params[..]).unwrap();

        for count in [0, 1, 2, 3, 31, 32, 33, 127, 128, 129, 254, 255, 256] {
            let mut got = AlignedOutput([0; L1]);
            let mut scalar = [0u8; L1];
            let mut nnz = NnzInfo::<256>::new();

            for perspective in 0..2 {
                let mut raw = AlignedAccumulator([0; L1]);
                for chunk in 0..L1 / 8 {
                    // Multiplication by 73 permutes all 256 chunks, so every
                    // count gives a spread of zero and non-zero vectors.
                    let global_chunk = perspective * (L1 / 8) + chunk;
                    if (global_chunk * 73) % (L1 / 4) < count {
                        let byte = chunk * 4 + chunk % 4;
                        raw.0[byte] = 255;
                        raw.0[byte + L1 / 2] = 255;
                    }
                }

                let offset = perspective * (L1 / 2);
                ft.transform_scalar(
                    &raw.0,
                    as_uninit_bytes(&mut scalar[offset..offset + L1 / 2]),
                    L1 / 2,
                );
                let mut permuted = AlignedAccumulator(raw.0);
                FeatureTransformer::permute_data(
                    &mut permuted.0,
                    FeatureTransformer::packus_order(),
                );
                // SAFETY: compile-time AVX-512F/BW and the runtime check above
                // cover the intrinsics. The accumulation and output buffers are 64-byte
                // aligned, and each perspective supplies 512 output bytes.
                unsafe {
                    ft.transform_avx512::<true>(
                        &permuted.0,
                        as_uninit_bytes(&mut got.0[offset..offset + L1 / 2]),
                        L1 / 2,
                        offset,
                        Some(&mut nnz),
                    )
                };
            }

            assert_eq!(got.0, scalar, "transformed bytes, count {count}");
            let expected: Vec<u16> = scalar
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
                .filter_map(|(i, bytes)| bytes.iter().any(|&b| b != 0).then_some(i as u16))
                .collect();
            let mut recorded = nnz.indices().to_vec();
            recorded.sort_unstable(); // The sparse layer permits any ordering of the non-zero chunks.
            assert_eq!(recorded, expected, "NNZ indices, count {count}");
            assert_eq!(nnz.count(), count, "NNZ count");

            let mut sparse_output = [0i32; 32];
            let mut dense_output = [0i32; 32];
            layer.propagate_with_nnz(&got.0, &mut sparse_output, &nnz);
            layer.propagate_dense_i32_test(&scalar, &mut dense_output);
            assert_eq!(sparse_output, dense_output, "fc_0 output, count {count}");
        }
    }
}
