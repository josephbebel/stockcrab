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

//! Common types and constants for NNUE evaluation

// Type definitions for NNUE components
// Note: These types are for the FeatureTransformer layer (i16 biases, SFNNv18 i8 weights).
// The affine transform layers (AffineTransform, AffineTransformSparseInput)
// use different types and define them locally: BiasType = i32, WeightType = i8.
pub type BiasType = i16;
pub type WeightType = i8;
pub type ThreatWeightType = i8; // SFNNv10: threat weights use i8
pub type IndexType = usize;
pub type FeatureIndexType = u16;

/// Fixed-capacity list used for the threat and pawn-pair feature indices.
///
/// Stockfish's `ValueList<u16, 256>` keeps these short-lived lists on the
/// stack.  The feature index space fits in `u16`, and a position can never
/// produce more than 256 simultaneously active entries for these feature
/// sets.  Keeping the capacity in the type makes accidental heap growth in
/// the accumulator hot paths impossible.
#[derive(Clone)]
pub struct FixedList<T: Copy + Default, const CAPACITY: usize> {
    // Invariant: `data[..len]` is initialised; the tail is deliberately left
    // uninitialised so that creating a list costs no 512-byte memset.
    data: [core::mem::MaybeUninit<T>; CAPACITY],
    len: usize,
}

impl<T: Copy + Default, const CAPACITY: usize> FixedList<T, CAPACITY> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            data: [core::mem::MaybeUninit::uninit(); CAPACITY],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        debug_assert!(self.len < CAPACITY);
        self.data[self.len] = core::mem::MaybeUninit::new(value);
        self.len += 1;
    }

    /// Append without a capacity check (C++ `ValueList::push_back`).
    ///
    /// # Safety
    /// `len() < CAPACITY` must hold.
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.len < CAPACITY);
        // SAFETY: the caller guarantees `len < CAPACITY`, so the slot is inside `data`.
        unsafe {
            *self.data.get_unchecked_mut(self.len) = core::mem::MaybeUninit::new(value);
        }
        self.len += 1;
    }

    /// Branchless conditional append (C++ `ValueList::push_back_if_lt`): the value is
    /// always stored at `len`, but `len` only advances when `keep` is true, so a rejected
    /// value is overwritten by the next push. Replaces `if keep { push(value) }`, whose
    /// data-dependent branch mispredicts.
    ///
    /// Requires one spare slot even when `keep` is false: panics if `len() == CAPACITY`.
    #[inline(always)]
    pub fn push_if(&mut self, value: T, keep: bool) {
        self.data[self.len] = core::mem::MaybeUninit::new(value);
        self.len += keep as usize;
    }

    /// [`push_if`](Self::push_if) without the capacity check.
    ///
    /// # Safety
    /// `len() < CAPACITY` must hold (also when `keep` is false: the slot at `len` is
    /// written unconditionally).
    #[inline(always)]
    pub unsafe fn push_if_unchecked(&mut self, value: T, keep: bool) {
        debug_assert!(self.len < CAPACITY);
        // SAFETY: the caller guarantees `len < CAPACITY`, so the slot is inside `data`.
        unsafe {
            *self.data.get_unchecked_mut(self.len) = core::mem::MaybeUninit::new(value);
        }
        self.len += keep as usize;
    }

    /// Reserve `count` entries and return a pointer to the first one (C++
    /// `ValueList::make_space`). The caller must initialise the first `count`
    /// entries and may write up to `max_store` elements through the pointer, so
    /// a whole vector can be stored without masking.
    ///
    /// # Safety
    /// `count <= max_store` and `len() + max_store <= CAPACITY` must hold. The caller must
    /// initialise the first `count` entries through the returned pointer before `data[..len]` is
    /// read (e.g. via `as_slice` or `Deref`).
    #[inline(always)]
    pub unsafe fn make_space(&mut self, count: usize, max_store: usize) -> *mut T {
        debug_assert!(count <= max_store && self.len + max_store <= CAPACITY);
        // SAFETY: the caller guarantees `len + max_store <= CAPACITY`, so `len` is in bounds
        // (or one past the end when `max_store` is zero).
        let p = unsafe { self.data.as_mut_ptr().add(self.len) } as *mut T;
        self.len += count;
        p
    }

    /// Append all of `src` (panics if it does not fit).
    #[inline(always)]
    pub fn extend_from_slice(&mut self, src: &[T]) {
        let start = self.len;
        let end = start + src.len();
        debug_assert!(end <= CAPACITY);
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
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the type invariant is that `data[..len]` is initialised and `len <= CAPACITY`;
        // the invariant holds after every safe method, and after every unsafe method whose
        // caller contract (`len < CAPACITY`, entries initialised) was met. `MaybeUninit<T>` has
        // the same layout as `T`.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }
}

impl<T: Copy + Default, const CAPACITY: usize> Default for FixedList<T, CAPACITY> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const CAPACITY: usize> core::ops::Deref for FixedList<T, CAPACITY> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

pub const MAX_FEATURE_INDEXES: usize = 256;
pub type FeatureIndexList = FixedList<FeatureIndexType, MAX_FEATURE_INDEXES>;
// Matches C++ uint8_t. The feature transformer clips to [0, FT_MAX_VAL], so outputs lie in [0, 127].
pub type TransformedFeatureType = u8;

// Constants for quantization and scaling
pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: i32 = 6;
// Quantization reference constants. The feature-transformer clip ceiling is
// `nnue_feature_transformer::FT_MAX_VAL`; `HIDDEN_ONE_VAL` represents the quantized
// "1.0" of the hidden activations, used by the final output scaling in
// NetworkArchitecture::propagate.
pub const HIDDEN_ONE_VAL: i32 = 128;
// Size of cache line (in bytes)
pub const CACHE_LINE_SIZE: usize = 64;

/// Orient a square according to perspective
/// Reflect kings on files a..d, so that it lands on e..h
///
/// The entry is the file-mirror mask of the king square (`SQ_H1` = 7 on files a..d, `SQ_A1` = 0
/// on files e..h); a perspective adds `56 * perspective` for the rank flip. Shared by K32H3,
/// FullThreats and PP_3Wide since SFNNv19.
#[rustfmt::skip]
pub const ORIENT_TBL: [u8; 64] = [
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
];

pub const MAX_SIMD_WIDTH: IndexType = 32;

// Re-export AlignedVec from its own module
pub use super::aligned_vec::AlignedVec;

#[cfg(test)]
mod fixed_list_push_if_tests {
    use super::*;

    /// `push_if(_unchecked)` must build exactly the list the old conditional push built.
    #[test]
    fn push_if_matches_conditional_push() {
        const MAX: u32 = 59808;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200 {
            let mut old = FixedList::<u16, 256>::new();
            let mut safe = FixedList::<u16, 256>::new();
            let mut fast = FixedList::<u16, 256>::new();
            for _ in 0..96 {
                // Roughly one value in four is rejected (>= MAX), as FullThreats does.
                let r = next();
                let v = if r & 3 == 0 {
                    MAX + (r >> 8) as u32 % 1000
                } else {
                    (r >> 8) as u32 % MAX
                };
                if v < MAX {
                    old.push(v as u16);
                }
                safe.push_if(v as u16, v < MAX);
                // SAFETY: at most 96 slots are consumed in a 256-entry list.
                unsafe { fast.push_if_unchecked(v as u16, v < MAX) };
            }
            assert_eq!(old.as_slice(), safe.as_slice());
            assert_eq!(old.as_slice(), fast.as_slice());
        }
    }
}
