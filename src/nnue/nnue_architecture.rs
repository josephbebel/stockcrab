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

//! NNUE architecture definitions and constants
//!
//! This module defines the neural network architecture used in NNUE evaluation,
//! including the network structure, layer configuration, and feature set.
//!
//! A single `NetworkArchitecture` (L1 = 1024, L2 = 32, L3 = 32). `fc_0` feeds `ac_sqr_0`
//! and `ac_0`; `fc_1` feeds `ac_sqr_1` and `ac_1`; `fc_2` reads all four activation
//! outputs. A skip connection from the last two `fc_0` outputs is added to the `fc_2`
//! output, and the final output is rescaled so that 1.0 maps to `600 * OUTPUT_SCALE` (see
//! `propagate_impl`).

use std::io::{Read, Write};

use aligned::{A64, Aligned};

use super::features::{FullThreats, K32H3, PP_3Wide};
use super::layers::{
    AffineTransform, AffineTransformSparseInput, ClippedReLU, NnzInfo, SqrClippedReLU,
};
use super::nnue_common::{
    HIDDEN_ONE_VAL, IndexType, OUTPUT_SCALE, TransformedFeatureType, WEIGHT_SCALE_BITS,
};

// Input features used in evaluation function
// PSQ features (piece-square): K32H3 (SFNNv19)
// Threat features: FullThreats (SFNNv10)
// Pawn pair features: PP_3Wide (SFNNv16)
pub type PSQFeatureSet = K32H3;
pub type ThreatFeatureSet = FullThreats;
pub type PairFeatureSet = PP_3Wide;

// Transformed feature dimensions (feature-transformer output, fc_0 input); unified architecture.
// C++: constexpr IndexType L1 = 1024; constexpr int L2 = 32; constexpr int L3 = 32;
pub const L1: IndexType = 1024;
pub const L2: usize = 32;
pub const L3: usize = 32;

// Feature dimensions
pub const PSQ_INPUT_DIMENSIONS: IndexType = PSQFeatureSet::DIMENSIONS; // 67584
pub const THREAT_INPUT_DIMENSIONS: IndexType = ThreatFeatureSet::DIMENSIONS; // 59808
pub const PAIR_INPUT_DIMENSIONS: IndexType = PairFeatureSet::DIMENSIONS; // 4560

// Pawn pair feature indices are concatenated right after the threat features.
// C++: static_assert(PairFeatureSet::IndexBase == ThreatFeatureSet::Dimensions)
const _: () = assert!(PairFeatureSet::INDEX_BASE == ThreatFeatureSet::DIMENSIONS);

// Bucket counts: every raw layer-stack configuration (SFNNv20). The network file stores only
// the reachable stacks; `Network::layer_stack` maps a raw configuration to a stored one.
pub const RAW_LAYER_STACKS: usize = 192;
pub const LAYER_STACKS: usize = RAW_LAYER_STACKS;

/// Helper function to ceil to multiple
pub const fn ceil_to_multiple(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

// C++ enables the fused activation path whenever AVX2 is selected. Its dense
// layer loader pre-permutes input columns to consume the packed lane order.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub const USE_PAIR_ACTIVATIONS: bool = true;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub const USE_PAIR_ACTIVATIONS: bool = false;

/// Unified network architecture (SF18).
///
/// C++ equivalent:
/// ```cpp
/// struct NetworkArchitecture {
///     AffineTransformSparseInput<L1, FC_0_OUTPUTS>        fc_0;
///     SqrClippedReLU<FC_0_OUTPUTS, WEIGHT_SCALE_BITS + 1>   ac_sqr_0;
///     ClippedReLU<FC_0_OUTPUTS, WEIGHT_SCALE_BITS + 1>      ac_0;
///     AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>     fc_1;
///     SqrClippedReLU<FC_1_OUTPUTS, WEIGHT_SCALE_BITS>       ac_sqr_1;
///     ClippedReLU<FC_1_OUTPUTS, WEIGHT_SCALE_BITS>          ac_1;
///     AffineTransform<FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 1> fc_2;
/// };
/// ```
#[derive(Clone)]
pub struct NetworkArchitecture {
    pub fc_0: AffineTransformSparseInput<{ L1 }, L2>,
    pub ac_sqr_0: SqrClippedReLU<L2, { (WEIGHT_SCALE_BITS + 1) as usize }>,
    pub ac_0: ClippedReLU<L2, { (WEIGHT_SCALE_BITS + 1) as usize }>,
    pub fc_1: AffineTransform<{ L2 * 2 }, L3>,
    pub ac_sqr_1: SqrClippedReLU<L3, { WEIGHT_SCALE_BITS as usize }>,
    pub ac_1: ClippedReLU<L3, { WEIGHT_SCALE_BITS as usize }>,
    pub fc_2: AffineTransform<{ L2 * 2 + L3 * 2 }, 1>,
}

impl NetworkArchitecture {
    pub const TRANSFORMED_FEATURE_DIMENSIONS: IndexType = L1;
    pub const FC_0_OUTPUTS: usize = L2;
    pub const FC_1_OUTPUTS: usize = L3;

    /// Create a new network architecture
    pub fn new() -> Self {
        Self {
            fc_0: AffineTransformSparseInput::new(),
            ac_sqr_0: SqrClippedReLU::new(),
            ac_0: ClippedReLU::new(),
            fc_1: AffineTransform::new(),
            ac_sqr_1: SqrClippedReLU::new(),
            ac_1: ClippedReLU::new(),
            fc_2: AffineTransform::new(),
        }
    }

    /// Hash value embedded in the evaluation file
    pub const fn get_hash_value() -> u32 {
        // Input slice hash
        let mut hash_value = 0xEC42E90Du32;
        hash_value ^= (L1 * 2) as u32;

        hash_value = AffineTransformSparseInput::<{ L1 }, L2>::get_hash_value(hash_value);
        // C++ deliberately omits ac_sqr_0 (and ac_sqr_1) from the hash because the
        // trainer does not (yet) write those hash values.
        hash_value =
            ClippedReLU::<L2, { (WEIGHT_SCALE_BITS + 1) as usize }>::get_hash_value(hash_value);
        hash_value = AffineTransform::<{ L2 * 2 }, L3>::get_hash_value(hash_value);
        hash_value = ClippedReLU::<L3, { WEIGHT_SCALE_BITS as usize }>::get_hash_value(hash_value);
        hash_value = AffineTransform::<{ L2 * 2 + L3 * 2 }, 1>::get_hash_value(hash_value);

        hash_value
    }

    /// Read network parameters from stream
    pub fn read_parameters<R: Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.fc_0.read_parameters(reader)?;
        self.ac_0.read_parameters(reader)?;
        self.fc_1.read_parameters(reader)?;
        self.ac_1.read_parameters(reader)?;
        self.fc_2.read_parameters(reader)?;
        Ok(())
    }

    /// Write network parameters to stream
    pub fn write_parameters<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.fc_0.write_parameters(writer)?;
        self.ac_0.write_parameters(writer)?;
        self.fc_1.write_parameters(writer)?;
        self.ac_1.write_parameters(writer)?;
        self.fc_2.write_parameters(writer)?;
        Ok(())
    }

    /// Propagate through the network.
    ///
    /// SF18 topology:
    ///   fc_0 -> {ac_sqr_0, ac_0} concatenated (64 = L2*2)
    ///        -> fc_1 -> {ac_sqr_1, ac_1} appended to the concatenation (128 = L2*2 + L3*2)
    ///        -> fc_2 (single output)
    /// plus a skip connection: fc_0_out[L2-2] - fc_0_out[L2-1].
    pub fn propagate(&self, transformed_features: &[TransformedFeatureType]) -> i32 {
        self.propagate_impl(transformed_features, None)
    }

    /// AVX-512 path using the non-zero input chunks collected by the feature
    /// transformer. This mirrors C++'s `NNZInfo` handoff and avoids rescanning
    /// the transformed feature buffer in `fc_0`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    pub(crate) fn propagate_with_nnz(
        &self,
        transformed_features: &[TransformedFeatureType],
        nnz: &NnzInfo<256>,
    ) -> i32 {
        self.propagate_impl(transformed_features, Some(nnz))
    }

    /// AArch64 path using the non-zero-chunk bitset recorded by the feature transformer
    /// (`FeatureTransformer::transform_with_bitset`). This mirrors C++'s `NNZInfo::bitset`
    /// handoff and avoids rescanning the transformed feature buffer in `fc_0`.
    // Kept out of line so that `Network::evaluate` stays the size it has in C++; the layers
    // themselves are inlined into this body.
    #[cfg(target_arch = "aarch64")]
    #[inline(never)]
    pub(crate) fn propagate_with_bitset(
        &self,
        transformed_features: &[TransformedFeatureType],
        bitset: &[u8],
    ) -> i32 {
        self.propagate_layers(transformed_features, None, Some(bitset))
    }

    fn propagate_impl(
        &self,
        transformed_features: &[TransformedFeatureType],
        nnz: Option<&NnzInfo<256>>,
    ) -> i32 {
        self.propagate_layers(transformed_features, nnz, None)
    }

    #[inline(always)]
    fn propagate_layers(
        &self,
        transformed_features: &[TransformedFeatureType],
        nnz: Option<&NnzInfo<256>>,
        bitset: Option<&[u8]>,
    ) -> i32 {
        let _ = nnz; // only the AVX-512 path consumes the non-zero chunk list
        let _ = bitset; // only the AArch64 path consumes the non-zero chunk bitset
        const FC_0_OUTPUTS: usize = L2;
        const FC_1_OUTPUTS: usize = L3;
        // concat_buffer holds [ac_sqr_0 | ac_0 | ac_sqr_1 | ac_1]
        const CONCAT_LEN: usize = ceil_to_multiple(FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 32);

        // Buffer structure for intermediate computations, cache-line aligned per field.
        // Plain stack buffer; no per-call heap allocation.
        #[repr(C)]
        struct Buffer {
            fc_0_out: Aligned<A64, [i32; ceil_to_multiple(FC_0_OUTPUTS, 32)]>,
            concat_buffer: Aligned<A64, [u8; CONCAT_LEN]>,
            fc_1_out: Aligned<A64, [i32; ceil_to_multiple(FC_1_OUTPUTS, 32)]>,
            fc_2_out: Aligned<A64, [i32; ceil_to_multiple(1, 32)]>,
        }

        let mut buffer = Buffer {
            fc_0_out: Aligned([0; ceil_to_multiple(FC_0_OUTPUTS, 32)]),
            concat_buffer: Aligned([0; CONCAT_LEN]),
            fc_1_out: Aligned([0; ceil_to_multiple(FC_1_OUTPUTS, 32)]),
            fc_2_out: Aligned([0; ceil_to_multiple(1, 32)]),
        };

        // fc_0
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        if let Some(nnz) = nnz {
            self.fc_0.propagate_with_nnz(
                transformed_features,
                &mut buffer.fc_0_out[..FC_0_OUTPUTS],
                nnz,
            );
        } else {
            self.fc_0
                .propagate(transformed_features, &mut buffer.fc_0_out[..FC_0_OUTPUTS]);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(bitset) = bitset {
            self.fc_0.propagate_with_bitset(
                transformed_features,
                &mut buffer.fc_0_out[..FC_0_OUTPUTS],
                bitset,
            );
        } else {
            self.fc_0
                .propagate(transformed_features, &mut buffer.fc_0_out[..FC_0_OUTPUTS]);
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )
        )))]
        self.fc_0
            .propagate(transformed_features, &mut buffer.fc_0_out[..FC_0_OUTPUTS]);

        // ac_sqr_0 -> concat[0..L2], ac_0 -> concat[L2..2*L2]
        if USE_PAIR_ACTIVATIONS {
            let (squared, rest) =
                buffer.concat_buffer[..FC_0_OUTPUTS * 2].split_at_mut(FC_0_OUTPUTS);
            self.ac_sqr_0
                .propagate_pair(&buffer.fc_0_out[..FC_0_OUTPUTS], squared, rest);
        } else {
            self.ac_sqr_0.propagate(
                &buffer.fc_0_out[..FC_0_OUTPUTS],
                &mut buffer.concat_buffer[..FC_0_OUTPUTS],
            );
            self.ac_0.propagate(
                &buffer.fc_0_out[..FC_0_OUTPUTS],
                &mut buffer.concat_buffer[FC_0_OUTPUTS..FC_0_OUTPUTS * 2],
            );
        }

        // fc_1 over the first 2*L2 concat entries
        self.fc_1.propagate(
            &buffer.concat_buffer[..FC_0_OUTPUTS * 2],
            &mut buffer.fc_1_out[..FC_1_OUTPUTS],
        );

        // ac_sqr_1 -> concat[2*L2..2*L2+L3], ac_1 -> concat[2*L2+L3..2*L2+2*L3]
        if USE_PAIR_ACTIVATIONS {
            let (_, activation_out) = buffer.concat_buffer.split_at_mut(FC_0_OUTPUTS * 2);
            let (squared, clipped) = activation_out[..FC_1_OUTPUTS * 2].split_at_mut(FC_1_OUTPUTS);
            self.ac_sqr_1
                .propagate_pair(&buffer.fc_1_out[..FC_1_OUTPUTS], squared, clipped);
        } else {
            self.ac_sqr_1.propagate(
                &buffer.fc_1_out[..FC_1_OUTPUTS],
                &mut buffer.concat_buffer[FC_0_OUTPUTS * 2..FC_0_OUTPUTS * 2 + FC_1_OUTPUTS],
            );
            self.ac_1.propagate(
                &buffer.fc_1_out[..FC_1_OUTPUTS],
                &mut buffer.concat_buffer
                    [FC_0_OUTPUTS * 2 + FC_1_OUTPUTS..FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2],
            );
        }

        // fc_2 over the full concatenated buffer (2*L2 + 2*L3)
        self.fc_2.propagate(
            &buffer.concat_buffer[..FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2],
            &mut buffer.fc_2_out[..1],
        );

        // Forward (skip) connection plus fc_2 output.
        const _: () = assert!(FC_0_OUTPUTS >= 2);
        let mut fwd_out: i32 = buffer.fc_2_out[0];
        let skip_0: i32 = buffer.fc_0_out[FC_0_OUTPUTS - 2] - buffer.fc_0_out[FC_0_OUTPUTS - 1];
        fwd_out += skip_0;

        // fwd_out is such that 1.0 is equal to HIDDEN_ONE_VAL*(1<<WEIGHT_SCALE_BITS)*2 in
        // quantized form, but we want 1.0 to be equal to 600*OUTPUT_SCALE.
        // Cast to i64 to make overflow impossible.
        const MULTIPLIER: i64 = 600 * OUTPUT_SCALE as i64;
        const DENOMINATOR: i64 = HIDDEN_ONE_VAL as i64 * (1i64 << WEIGHT_SCALE_BITS) * 2;

        (fwd_out as i64 * MULTIPLIER / DENOMINATOR) as i32
    }
}

impl Default for NetworkArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_hash() {
        let hash = NetworkArchitecture::get_hash_value();
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_ceil_to_multiple() {
        assert_eq!(ceil_to_multiple(10, 8), 16);
        assert_eq!(ceil_to_multiple(16, 8), 16);
        assert_eq!(ceil_to_multiple(17, 8), 24);
    }
}
