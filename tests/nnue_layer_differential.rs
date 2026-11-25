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

//! Randomised differential tests of the NNUE layer kernels (whatever ISA the build selects:
//! NEON, NEON+dotprod, SSSE3, AVX2, ...) against independent scalar references written in
//! terms of the *logical* (serialised) network layout.
//!
//! Only the public API is used: parameters are loaded through `read_parameters`, so the
//! weight scrambling (`get_weight_index`) is covered as well.
//!
//! Input activations are restricted to 0..=127, the range the network produces; the NEON
//! kernels (like the C++ `vdotq_s32` / `vmull_s8` ones) treat input bytes as signed.

use std::io::Cursor;
use stockcrab::nnue::layers::{
    AffineTransform, AffineTransformSparseInput, ClippedReLU, SqrClippedReLU, find_nnz_into,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Byte buffer whose data starts at a 64-byte boundary plus `offset` (a multiple of 4), so the
/// kernels see 4-byte aligned but not necessarily 16-byte aligned inputs.
struct Bytes {
    store: Vec<u64>,
    offset: usize,
    len: usize,
}
impl Bytes {
    fn new(len: usize, offset: usize) -> Self {
        assert!(offset.is_multiple_of(4) && offset < 64);
        let store = vec![0u64; (len + offset).div_ceil(8) + 16];
        Self { store, offset, len }
    }
    fn base(&self) -> usize {
        let p = self.store.as_ptr() as usize;
        (64 - p % 64) % 64 + self.offset
    }
    fn get(&self) -> &[u8] {
        let b = self.base();
        // SAFETY: the store holds `len + offset + 128` bytes; u8 has no alignment/validity needs.
        unsafe { core::slice::from_raw_parts((self.store.as_ptr() as *const u8).add(b), self.len) }
    }
    fn get_mut(&mut self) -> &mut [u8] {
        let b = self.base();
        // SAFETY: as in `get`, and `self` is borrowed mutably.
        unsafe {
            core::slice::from_raw_parts_mut((self.store.as_mut_ptr() as *mut u8).add(b), self.len)
        }
    }
}

const fn ceil_to(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// Logical parameters of an affine layer: `w[k * padded + j]`.
struct Params {
    biases: Vec<i32>,
    weights: Vec<i8>,
    padded: usize,
}

fn gen_params(rng: &mut Rng, class: u64, inputs: usize, outputs: usize) -> Params {
    let padded = ceil_to(inputs, 32);
    // Biases near the i32 limits make the sums wrap. The vector kernels wrap like the C++ ones,
    // but the scalar reference paths and the single-output horizontal add use checked `+` in
    // debug builds, so the wrapping classes are exercised in release builds only.
    let bias_classes = if cfg!(debug_assertions) { 2 } else { 4 };
    let biases = (0..outputs)
        .map(|_| match class % bias_classes {
            0 => 0,
            1 => rng.below(200_001) as i32 - 100_000,
            2 => [i32::MIN, i32::MAX, -1, 1][rng.below(4) as usize],
            _ => rng.next() as i32,
        })
        .collect();
    let weights = (0..outputs * padded)
        .map(|_| match class % 6 {
            0 => rng.next() as i8,
            1 => -128,
            2 => 127,
            3 => [-128i8, 127][rng.below(2) as usize],
            4 => [-128i8, -127, -1, 0, 1, 126, 127][rng.below(7) as usize],
            _ => (rng.below(9) as i8) - 4,
        })
        .collect();
    Params {
        biases,
        weights,
        padded,
    }
}

fn serialise(p: &Params) -> Vec<u8> {
    let mut v = Vec::with_capacity(p.biases.len() * 4 + p.weights.len());
    for b in &p.biases {
        v.extend_from_slice(&b.to_le_bytes());
    }
    v.extend(p.weights.iter().map(|&w| w as u8));
    v
}

fn reference(p: &Params, input: &[u8], inputs: usize) -> Vec<i32> {
    (0..p.biases.len())
        .map(|k| {
            let mut sum = p.biases[k];
            for j in 0..inputs {
                sum =
                    sum.wrapping_add(i32::from(input[j]) * i32::from(p.weights[k * p.padded + j]));
            }
            sum
        })
        .collect()
}

/// Input-buffer slot that the dense `AffineTransform` reads serialised input column `j` from.
///
/// On AVX2 and AVX-512 builds the dense layer permutes its serialised weight columns at load
/// time (Stockfish's `USE_PAIR_ACTIVATIONS` mapping) so that it consumes the interleaved output
/// of `SqrClippedReLU::propagate_pair` without a shuffle. Elsewhere the layout is the identity.
/// The sparse `AffineTransformSparseInput` layer has no such permutation.
fn affine_input_slot(j: usize) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let (block, chunk) = (j / 32, (j % 32) / 4);
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        let permuted = (chunk % 4) * 2 + chunk / 4;
        #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
        let permuted = (chunk % 2) * 4 + chunk / 2;
        block * 32 + permuted * 4 + j % 4
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        j
    }
}

/// `reference` for the dense layer: sums over all `padded` serialised columns (the permutation
/// can map a column past `inputs` onto a live slot), reading input slot `affine_input_slot(j)`.
fn reference_affine(p: &Params, input: &[u8]) -> Vec<i32> {
    (0..p.biases.len())
        .map(|k| {
            let mut sum = p.biases[k];
            for j in 0..p.padded {
                sum = sum.wrapping_add(
                    i32::from(input[affine_input_slot(j)]) * i32::from(p.weights[k * p.padded + j]),
                );
            }
            sum
        })
        .collect()
}

/// Fills `input[..inputs]` (activation bytes, 0..=127) by pattern class; the padding stays zero.
fn gen_input(rng: &mut Rng, class: u64, input: &mut [u8], inputs: usize) {
    input.fill(0);
    let chunks = inputs / 4;
    match class % 12 {
        0 => {}
        1 => input[..inputs].fill(127),
        2 => input[..inputs]
            .iter_mut()
            .for_each(|b| *b = rng.below(128) as u8),
        3 => input[..inputs].fill(1),
        // exactly n non-zero 4-byte chunks, n = 1..=7: every remainder class of the 2- and
        // 3-chain loops
        4 if chunks > 0 => {
            let n = 1 + rng.below(7.min(chunks as u64));
            for _ in 0..n {
                let c = rng.below(chunks as u64) as usize;
                input[c * 4 + rng.below(4) as usize] = 1 + rng.below(127) as u8;
            }
        }
        // one non-zero byte in a fixed byte lane of every chunk
        5 => {
            let lane = rng.below(4) as usize;
            for j in (lane..inputs).step_by(4) {
                input[j] = 127;
            }
        }
        // random density
        6 | 7 => {
            let per_mille = [5, 20, 100, 300, 600, 950][rng.below(6) as usize];
            for b in input[..inputs].iter_mut() {
                if rng.below(1000) < per_mille {
                    *b = 1 + rng.below(127) as u8;
                }
            }
        }
        // chunk-granular random density (whole chunks zero / non-zero)
        8 | 9 if chunks > 0 => {
            let per_mille = [10, 60, 250, 500, 900][rng.below(5) as usize];
            for c in 0..chunks {
                if rng.below(1000) < per_mille {
                    for b in 0..4 {
                        input[c * 4 + b] = rng.below(128) as u8;
                    }
                }
            }
        }
        // only the very first / very last input
        10 => input[0] = 127,
        _ => input[inputs - 1] = 127,
    }
}

fn check_affine<const IN: usize, const OUT: usize>(cases: usize, seed: u64) {
    let mut rng = Rng(seed ^ ((IN as u64) << 20) ^ OUT as u64);
    let padded = ceil_to(IN, 32);
    let mut layer = AffineTransform::<IN, OUT>::new();
    let mut params = gen_params(&mut rng, 0, IN, OUT);
    for case in 0..cases {
        if case % 8 == 0 {
            params = gen_params(&mut rng, (case / 8) as u64, IN, OUT);
            let bytes = serialise(&params);
            layer.read_parameters(&mut Cursor::new(&bytes)).unwrap();
            let mut written = Vec::new();
            layer.write_parameters(&mut written).unwrap();
            assert_eq!(
                written, bytes,
                "AffineTransform<{IN},{OUT}> parameter round trip"
            );
        }
        let mut buf = Bytes::new(padded, [0, 4, 8, 12, 32][case % 5]);
        gen_input(&mut rng, case as u64, buf.get_mut(), IN);
        let expected = reference_affine(&params, buf.get());
        const GUARD: i32 = 0x5A5A_5A5A;
        let mut out = vec![GUARD; OUT + 8];
        layer.propagate(buf.get(), &mut out[..OUT]);
        assert_eq!(
            &out[..OUT],
            &expected[..],
            "AffineTransform<{IN},{OUT}> case {case}"
        );
        assert!(
            out[OUT..].iter().all(|&g| g == GUARD),
            "write past OUT_DIMS"
        );
    }
}

fn check_sparse<const IN: usize, const OUT: usize>(cases: usize, seed: u64) {
    let mut rng = Rng(seed ^ ((IN as u64) << 24) ^ ((OUT as u64) << 4) ^ 7);
    let padded = ceil_to(IN, 32);
    let mut layer = AffineTransformSparseInput::<IN, OUT>::new();
    let mut params = gen_params(&mut rng, 0, IN, OUT);
    for case in 0..cases {
        if case % 8 == 0 {
            params = gen_params(&mut rng, (case / 8) as u64, IN, OUT);
            let bytes = serialise(&params);
            layer.read_parameters(&mut Cursor::new(&bytes)).unwrap();
            let mut written = Vec::new();
            layer.write_parameters(&mut written).unwrap();
            assert_eq!(
                written, bytes,
                "AffineTransformSparseInput<{IN},{OUT}> parameter round trip"
            );
        }
        let mut buf = Bytes::new(padded, [0, 4, 8, 12, 32][case % 5]);
        gen_input(&mut rng, case as u64, buf.get_mut(), IN);
        let expected = reference(&params, buf.get(), IN);
        const GUARD: i32 = 0x5A5A_5A5A;
        let mut out = vec![GUARD; OUT + 8];
        layer.propagate(buf.get(), &mut out[..OUT]);
        assert_eq!(
            &out[..OUT],
            &expected[..],
            "AffineTransformSparseInput<{IN},{OUT}> case {case}"
        );
        assert!(
            out[OUT..].iter().all(|&g| g == GUARD),
            "write past OUT_DIMS"
        );
    }
}

/// Every 8-chunk non-zero mask (all 256 lookup-table rows), at every group position, through
/// the full sparse layer: checks `find_nnz` + column selection together.
fn check_sparse_every_mask<const IN: usize, const OUT: usize>(seed: u64) {
    let mut rng = Rng(seed);
    let padded = ceil_to(IN, 32);
    let mut layer = AffineTransformSparseInput::<IN, OUT>::new();
    let params = gen_params(&mut rng, 0, IN, OUT);
    layer
        .read_parameters(&mut Cursor::new(serialise(&params)))
        .unwrap();
    let groups = IN / 32;
    for mask in 0..256usize {
        for group in 0..groups {
            let mut buf = Bytes::new(padded, 0);
            let input = buf.get_mut();
            for bit in 0..8 {
                if mask >> bit & 1 != 0 {
                    let c = group * 8 + bit;
                    input[c * 4 + rng.below(4) as usize] = 1 + rng.below(127) as u8;
                }
            }
            let expected = reference(&params, buf.get(), IN);
            let mut out = vec![0i32; OUT];
            layer.propagate(buf.get(), &mut out);
            assert_eq!(
                out, expected,
                "sparse<{IN},{OUT}> mask {mask:#x} group {group}"
            );
        }
    }
}

fn nnz_reference(input: &[u8], num_chunks: usize) -> Vec<u16> {
    (0..num_chunks)
        .filter(|&c| input[c * 4..c * 4 + 4].iter().any(|&b| b != 0))
        .map(|c| c as u16)
        .collect()
}

fn check_find_nnz(num_chunks: usize, cases: usize, seed: u64) {
    let mut rng = Rng(seed ^ num_chunks as u64);
    const VALUES: [u32; 9] = [
        1,
        0x80,
        0x100,
        0x8000,
        0x0001_0000,
        0x0100_0000,
        0x8000_0000,
        0xFFFF_FFFF,
        0x7F7F_7F7F,
    ];
    for case in 0..cases {
        let mut buf = Bytes::new(num_chunks * 4, [0, 4, 8, 12][case % 4]);
        let input = buf.get_mut();
        let per_mille = [0, 3, 30, 125, 500, 875, 990, 1000][case % 8];
        for c in 0..num_chunks {
            if rng.below(1000) < per_mille {
                let v = if rng.below(3) == 0 {
                    (rng.next() as u32) | 1
                } else {
                    VALUES[rng.below(VALUES.len() as u64) as usize]
                };
                input[c * 4..c * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let expected = nnz_reference(buf.get(), num_chunks);
        const GUARD: u16 = 0xAAAA;
        let mut out = vec![GUARD; num_chunks + 16];
        let count = find_nnz_into(buf.get(), num_chunks, &mut out[..num_chunks]);
        assert_eq!(
            count,
            expected.len(),
            "find_nnz({num_chunks}) count, case {case}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "find_nnz({num_chunks}) case {case}"
        );
        assert!(
            out[num_chunks..].iter().all(|&g| g == GUARD),
            "find_nnz wrote past the output slice"
        );
    }
}

/// All 256 masks for one 8-chunk group, at each group position, with each interesting value.
fn check_find_nnz_every_mask(num_chunks: usize) {
    for value in [1u32, 0x8000_0000, 0xFFFF_FFFF, 0x0000_FF00] {
        for group in 0..num_chunks / 8 {
            for mask in 0..256usize {
                let mut buf = Bytes::new(num_chunks * 4, 0);
                let input = buf.get_mut();
                for bit in 0..8 {
                    if mask >> bit & 1 != 0 {
                        let c = group * 8 + bit;
                        input[c * 4..c * 4 + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
                let expected = nnz_reference(buf.get(), num_chunks);
                let mut out = vec![0u16; num_chunks];
                let count = find_nnz_into(buf.get(), num_chunks, &mut out);
                assert_eq!(&out[..count], &expected[..], "mask {mask:#x} group {group}");
            }
        }
    }
}

fn activation_input(rng: &mut Rng, class: u64, n: usize, scale: u32) -> Vec<i32> {
    (0..n)
        .map(|i| match class % 10 {
            0 => 0,
            1 => i32::MIN,
            2 => i32::MAX,
            3 => rng.next() as i32,
            // around every clamp / narrowing boundary
            4 => {
                let k = [0i64, 1, 126, 127, 128, 255, 256, 32767, 32768, 65535, 65536]
                    [rng.below(11) as usize];
                let sign = if rng.below(2) == 0 { 1 } else { -1 };
                ((sign * (k << scale) + rng.below(5) as i64 - 2)
                    .clamp(i32::MIN as i64, i32::MAX as i64)) as i32
            }
            5 => rng.below(1 << 16) as i32 - (1 << 15),
            6 => rng.below(1 << 20) as i32 - (1 << 19),
            7 => [
                i32::MIN,
                i32::MIN + 1,
                -32769,
                -32768,
                -32767,
                -1,
                0,
                1,
                32766,
                32767,
                32768,
                i32::MAX - 1,
                i32::MAX,
            ][(i + rng.below(13) as usize) % 13],
            8 => rng.below(1 << 14) as i32,
            _ => (rng.next() as i32) >> rng.below(31),
        })
        .collect()
}

fn check_crelu<const N: usize, const S: usize>(cases: usize, seed: u64) {
    let mut rng = Rng(seed ^ ((N as u64) << 8) ^ S as u64);
    let layer = ClippedReLU::<N, S>::new();
    for case in 0..cases {
        let input = activation_input(&mut rng, case as u64, N, S as u32);
        let expected: Vec<u8> = input
            .iter()
            .map(|&x| (x >> S).clamp(0, 127) as u8)
            .collect();
        const GUARD: u8 = 0xA5;
        let mut out = vec![GUARD; N + 32];
        // odd output offset: the stores must be unaligned-safe
        let off = case % 3;
        layer.propagate(&input, &mut out[off..off + N]);
        assert_eq!(
            &out[off..off + N],
            &expected[..],
            "ClippedReLU<{N},{S}> case {case}"
        );
        assert!(
            out[..off]
                .iter()
                .chain(&out[off + N..])
                .all(|&g| g == GUARD)
        );
    }
}

fn check_sqr<const N: usize, const S: usize>(cases: usize, seed: u64) {
    let mut rng = Rng(seed ^ ((N as u64) << 8) ^ S as u64 ^ 0x55);
    let layer = SqrClippedReLU::<N, S>::new();
    for case in 0..cases {
        let input = activation_input(&mut rng, case as u64, N, S as u32);
        let expected: Vec<u8> = input
            .iter()
            .map(|&x| ((i64::from(x) * i64::from(x)) >> (2 * S + 7)).min(127) as u8)
            .collect();
        const GUARD: u8 = 0xA5;
        let mut out = vec![GUARD; N + 32];
        let off = case % 3;
        layer.propagate(&input, &mut out[off..off + N]);
        assert_eq!(
            &out[off..off + N],
            &expected[..],
            "SqrClippedReLU<{N},{S}> case {case}"
        );
        assert!(
            out[..off]
                .iter()
                .chain(&out[off + N..])
                .all(|&g| g == GUARD)
        );
    }
}

// --- production geometry (SFNNv16: 1024 -> 32 -> 32 -> 1) --------------------------------------

#[test]
fn fc_0_sparse_1024x32_matches_reference() {
    check_sparse::<1024, 32>(6000, 1);
    check_sparse_every_mask::<1024, 32>(2);
}

#[test]
fn fc_1_affine_64x32_matches_reference() {
    check_affine::<64, 32>(20000, 3);
}

#[test]
fn fc_2_affine_128x1_matches_reference() {
    check_affine::<128, 1>(20000, 4);
}

#[test]
fn activations_32_match_reference() {
    check_crelu::<32, 7>(20000, 5);
    check_crelu::<32, 6>(20000, 6);
    check_sqr::<32, 7>(20000, 7);
    check_sqr::<32, 6>(20000, 8);
}

// --- other geometries the kernels claim to support ---------------------------------------------

#[test]
fn affine_other_dims_match_reference() {
    check_affine::<32, 32>(3000, 10);
    check_affine::<30, 32>(3000, 11);
    check_affine::<60, 64>(2000, 12);
    check_affine::<8, 4>(3000, 13);
    check_affine::<16, 8>(3000, 14);
    check_affine::<512, 16>(1000, 15);
    check_affine::<32, 1>(3000, 16);
    check_affine::<30, 1>(3000, 17);
    check_affine::<100, 1>(3000, 18);
    check_affine::<4, 1>(3000, 19);
    // widths that are not a multiple of the 4-lane register: reference path on NEON
    check_affine::<32, 3>(2000, 20);
    check_affine::<8, 2>(2000, 21);
    check_affine::<64, 6>(2000, 22);
}

#[test]
fn sparse_other_dims_match_reference() {
    // IN a multiple of 32 (NUM_CHUNKS a multiple of 8) and OUT a multiple of 16, as upstream
    // static_asserts (`OutputDimensions % 16 == 0`, `InputDimensions % 256 == 0`), plus
    // multiples of 4 outputs.
    check_sparse::<256, 16>(3000, 30);
    check_sparse::<512, 64>(1500, 31);
    check_sparse::<3072, 16>(600, 32);
    check_sparse::<1024, 16>(1500, 33);
    check_sparse::<32, 4>(3000, 34);
    check_sparse::<64, 8>(3000, 35);
    check_sparse::<768, 12>(1000, 36);
    check_sparse_every_mask::<256, 16>(37);
}

#[test]
fn activations_other_dims_match_reference() {
    // sizes with a scalar tail (not a multiple of 8 / 16) and every valid scale
    check_crelu::<1, 6>(2000, 40);
    check_crelu::<7, 6>(2000, 41);
    check_crelu::<8, 5>(2000, 42);
    check_crelu::<16, 8>(2000, 43);
    check_crelu::<37, 6>(2000, 44);
    check_crelu::<64, 7>(2000, 45);
    check_crelu::<15, 7>(2000, 46);
    check_sqr::<1, 6>(2000, 50);
    check_sqr::<5, 5>(2000, 51);
    check_sqr::<16, 5>(2000, 52);
    check_sqr::<16, 8>(2000, 53);
    check_sqr::<37, 6>(2000, 54);
    check_sqr::<48, 7>(2000, 55);
    check_sqr::<15, 7>(2000, 56);
    check_sqr::<64, 8>(2000, 57);
}

#[test]
fn find_nnz_matches_reference() {
    for &n in &[8usize, 16, 64, 192, 256, 768] {
        check_find_nnz(n, 4000, 60);
    }
    check_find_nnz_every_mask(8);
    check_find_nnz_every_mask(32);
    check_find_nnz_every_mask(256);
}

// --- geometries outside what upstream static_asserts ------------------------------------------
// Upstream refuses to compile these (`OutputDimensions % 16 == 0`, `InputDimensions % 256 ==
// 0`). The Rust layer accepts them, so they must either work or be rejected at compile time.

/// NUM_CHUNKS not a multiple of 8: the 8-wide `find_nnz` has no tail loop.
macro_rules! nonprod_nnz {
    ($($name:ident: $n:expr),* $(,)?) => {$(
        #[test]
        fn $name() {
            check_find_nnz($n, 500, 70);
        }
    )*};
}
nonprod_nnz! {
    nonprod_find_nnz_1: 1, nonprod_find_nnz_2: 2, nonprod_find_nnz_4: 4, nonprod_find_nnz_10: 10,
    nonprod_find_nnz_12: 12, nonprod_find_nnz_20: 20, nonprod_find_nnz_257: 257,
}

macro_rules! nonprod_sparse {
    ($($name:ident: $i:literal, $o:literal),* $(,)?) => {$(
        #[test]
        fn $name() {
            check_sparse::<$i, $o>(500, 71);
        }
    )*};
}
nonprod_sparse! {
    // NUM_CHUNKS = ceil(IN / 8) * 2 not a multiple of 8
    nonprod_sparse_8x4: 8, 4, nonprod_sparse_40x16: 40, 16, nonprod_sparse_16x16: 16, 16,
    nonprod_sparse_48x32: 48, 32,
    // OUT_DIMS not a multiple of the 4-lane accumulator register
    nonprod_sparse_32x2: 32, 2, nonprod_sparse_64x6: 64, 6, nonprod_sparse_1024x15: 1024, 15,
    nonprod_sparse_1024x1: 1024, 1,
}
