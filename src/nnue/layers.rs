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

//! Neural network layers for NNUE evaluation

mod affine_transform;
mod affine_transform_sparse_input;
mod clipped_relu;
mod sqr_clipped_relu;

pub use affine_transform::AffineTransform;
pub(crate) use affine_transform_sparse_input::NnzInfo;
pub use affine_transform_sparse_input::{AffineTransformSparseInput, find_nnz_into};
pub use clipped_relu::ClippedReLU;
pub use sqr_clipped_relu::SqrClippedReLU;
