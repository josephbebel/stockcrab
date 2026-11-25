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

//! Safe SIMD entry points must reject undersized slices before raw stores.

use stockcrab::nnue::layers::find_nnz_into;
use stockcrab::nnue::nnue_accumulator::AccumulatorStack;
use stockcrab::nnue::nnue_feature_transformer::FeatureTransformer;
use stockcrab::position::Position;

#[repr(align(64))]
struct Input([u8; 64]);

#[test]
#[should_panic(expected = "NNZ output is too short")]
fn nnz_rejects_a_one_element_output() {
    find_nnz_into(&Input([1; 64]).0, 16, &mut [0; 1]);
}

#[test]
#[should_panic(expected = "NNZ input is too short")]
fn nnz_rejects_a_short_input() {
    find_nnz_into(&Input([1; 64]).0[..4], 16, &mut [0; 16]);
}

#[test]
#[should_panic(expected = "NNZ input must be 4-byte aligned")]
fn nnz_rejects_misaligned_input() {
    find_nnz_into(&Input([1; 64]).0[1..], 8, &mut [0; 8]);
}

#[test]
#[should_panic(expected = "feature output needs L1 elements")]
fn transformer_rejects_a_half_size_output() {
    FeatureTransformer::new().transform(
        &Position::default(),
        &mut AccumulatorStack::new(),
        None,
        &mut [0; 512],
    );
}

#[test]
fn nnz_full_output_does_not_touch_the_following_elements() {
    let mut output = [u16::MAX; 32];
    let count = find_nnz_into(&Input([1; 64]).0, 16, &mut output[..16]);
    assert_eq!(count, 16);
    assert_eq!(&output[..16], &(0..16).collect::<Vec<_>>());
    assert_eq!(&output[16..], &[u16::MAX; 16]);
}
