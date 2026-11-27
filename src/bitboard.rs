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

//! Bitboard constants, lookup tables and attack generation.
//!
//! Covers C++ `bitboard.h` (constants, shifts, popcount/lsb/msb) and `attacks.h`
//! / `attacks.cpp` (sliding attacks, pseudo-attacks, between/line tables). The
//! port keeps them in one module because Rust has no header/implementation
//! split to mirror.
//!
//! Attack tables are initialized on first use or by an explicit [`init`] call.
//! C++ builds the equivalents as `constexpr` data or in
//! `Attacks::init()`.

use crate::types::*;
use core::cmp;
use core::mem::MaybeUninit;
use std::sync::Once;
use std::sync::OnceLock;

// Compile-time bitboard constants (files & ranks)

pub const FILE_A_BB: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_B_BB: Bitboard = FILE_A_BB << 1;
pub const FILE_C_BB: Bitboard = FILE_A_BB << 2;
pub const FILE_D_BB: Bitboard = FILE_A_BB << 3;
pub const FILE_E_BB: Bitboard = FILE_A_BB << 4;
pub const FILE_F_BB: Bitboard = FILE_A_BB << 5;
pub const FILE_G_BB: Bitboard = FILE_A_BB << 6;
pub const FILE_H_BB: Bitboard = FILE_A_BB << 7;

pub const RANK_1_BB: Bitboard = 0xFF;
pub const RANK_2_BB: Bitboard = RANK_1_BB << (8 * 1);
pub const RANK_3_BB: Bitboard = RANK_1_BB << (8 * 2);
pub const RANK_4_BB: Bitboard = RANK_1_BB << (8 * 3);
pub const RANK_5_BB: Bitboard = RANK_1_BB << (8 * 4);
pub const RANK_6_BB: Bitboard = RANK_1_BB << (8 * 5);
pub const RANK_7_BB: Bitboard = RANK_1_BB << (8 * 6);
pub const RANK_8_BB: Bitboard = RANK_1_BB << (8 * 7);

// Global tables - filled by `init()` once, then read-only

static POPCNT16: OnceLock<Box<[u8; 1 << 16]>> = OnceLock::new();
static SQUARE_DISTANCE: OnceLock<Box<[[u8; SQUARE_NB]; SQUARE_NB]>> = OnceLock::new();
static mut BETWEEN_BB_STORAGE: MaybeUninit<[[Bitboard; SQUARE_NB]; SQUARE_NB]> =
    MaybeUninit::uninit();
static mut LINE_BB_STORAGE: MaybeUninit<[[Bitboard; SQUARE_NB]; SQUARE_NB]> = MaybeUninit::uninit();
static mut INVERSE_RAY_PASS_BB_STORAGE: MaybeUninit<[[Bitboard; SQUARE_NB]; SQUARE_NB]> =
    MaybeUninit::uninit();
static mut DIAG_BB_STORAGE: MaybeUninit<[Bitboard; SQUARE_NB]> = MaybeUninit::uninit();
static mut ANTIDIAG_BB_STORAGE: MaybeUninit<[Bitboard; SQUARE_NB]> = MaybeUninit::uninit();
// Built once at startup and addressed directly in the hot attack paths, as in
// C++'s static `PseudoAttacks` table.
static mut PSEUDO_ATTACKS_STORAGE: MaybeUninit<[[Bitboard; SQUARE_NB]; PIECE_TYPE_NB]> =
    MaybeUninit::uninit();
// Pawn attacks per colour and square: the counterpart of C++'s use of the first two rows of
// `PseudoAttacks` (indexed by colour) for pawns, as a separate table because the port indexes
// that one by `PieceType`.
static mut PAWN_ATTACKS_STORAGE: MaybeUninit<[[Bitboard; SQUARE_NB]; COLOR_NB]> =
    MaybeUninit::uninit();

// Sliding attacks by hyperbola quintessence,
// see https://www.chessprogramming.org/Hyperbola_Quintessence
//
// Rook and bishop attacks are computed together from one 64-byte, 32-byte-aligned DualMagic
// entry per square, on every target (C++ `DualMagic`, attacks.h): with AVX2 the three line masks
// are three lanes of one vector, otherwise the scalar kernel handles each line in turn.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
struct DualMagic {
    mask_file: Bitboard,
    mask_diag: Bitboard,
    mask_antidiag: Bitboard,
    mask_none: Bitboard,
    r: Bitboard,
    rr: Bitboard,
    rank_attacks: &'static [u8; 64],
    shift: u32,
}

#[repr(align(32))]
#[derive(Debug)]
struct AlignedDualMagics([DualMagic; SQUARE_NB]);

// Initialize once in `init()` and then keep the table at a fixed address so
// the kernels can match C++'s direct `DualMagics[sq]` access.
static mut DUAL_MAGICS_STORAGE: MaybeUninit<AlignedDualMagics> = MaybeUninit::uninit();

const fn rank_attacks(file: usize, occupancy: usize) -> u8 {
    let mut attacks = 0u8;

    let mut f = file;
    while f > 0 {
        f -= 1;
        attacks |= 1u8 << f;
        if (occupancy & (1usize << f)) != 0 {
            break;
        }
    }

    let mut f = file;
    while f < 7 {
        f += 1;
        attacks |= 1u8 << f;
        if (occupancy & (1usize << f)) != 0 {
            break;
        }
    }

    attacks
}

const fn build_rank_attacks() -> [[u8; 64]; 8] {
    let mut table = [[0u8; 64]; 8];
    let mut file = 0;
    while file < 8 {
        let mut occupancy = 0;
        while occupancy < 64 {
            table[file][occupancy] = rank_attacks(file, occupancy << 1);
            occupancy += 1;
        }
        file += 1;
    }
    table
}

// Sliding attacks within a rank, indexed by the slider's file and the
// 6 inner bits of the rank occupancy (edge squares never affect the
// attack set), yielding the 8-bit attack set on that rank
static RANK_ATTACKS: [[u8; 64]; 8] = build_rank_attacks();

static INIT_ONCE: Once = Once::new();

#[inline(always)]
fn ensure_initialized() {
    if !INIT_ONCE.is_completed() {
        init();
    }
}

#[inline(always)]
fn get_pseudo_attacks() -> &'static [[Bitboard; SQUARE_NB]; PIECE_TYPE_NB] {
    ensure_initialized();
    // SAFETY: `init()` writes the table before any game logic starts, and it
    // remains immutable for the lifetime of the process.
    unsafe {
        &*core::ptr::addr_of!(PSEUDO_ATTACKS_STORAGE)
            .cast::<[[Bitboard; SQUARE_NB]; PIECE_TYPE_NB]>()
    }
}

// Internal helpers

/// Single-bit bitboard for a square
#[inline(always)]
fn square_bb(sq: Square) -> Bitboard {
    1u64 << (sq as u8)
}

/// Bitboard for a rank index
#[inline(always)]
fn rank_bb_r(r: Rank) -> Bitboard {
    RANK_1_BB << (8 * r as u8)
}

/// Bitboard for a file index
#[inline(always)]
fn file_bb_f(f: File) -> Bitboard {
    FILE_A_BB << (f as u8)
}

/// Moves a bitboard one or two steps as specified by the direction `d`
#[inline(always)]
pub fn shift(b: Bitboard, d: Direction) -> Bitboard {
    match d {
        NORTH => b << 8,
        SOUTH => b >> 8,
        x if x == NORTH + NORTH => b << 16,
        x if x == SOUTH + SOUTH => b >> 16,
        EAST => (b & !FILE_H_BB) << 1,
        WEST => (b & !FILE_A_BB) >> 1,
        NORTH_EAST => (b & !FILE_H_BB) << 9,
        NORTH_WEST => (b & !FILE_A_BB) << 7,
        SOUTH_EAST => (b & !FILE_H_BB) >> 7,
        SOUTH_WEST => (b & !FILE_A_BB) >> 9,
        _ => 0,
    }
}

/// Returns the squares attacked by pawns of the given color
/// from the squares in the given bitboard.
#[inline(always)]
pub fn pawn_attacks(color: Color, b: Bitboard) -> Bitboard {
    match color {
        Color::White => shift(b, NORTH_WEST) | shift(b, NORTH_EAST),
        Color::Black => shift(b, SOUTH_WEST) | shift(b, SOUTH_EAST),
    }
}

/// Returns the squares that a pawn of the given color on the given square would attack.
/// This matches the C++ `attacks_bb<PAWN>`(square, color) function.
#[inline(always)]
pub fn pawn_attacks_from_square(sq: Square, color: Color) -> Bitboard {
    ensure_initialized();
    // SAFETY: `init()` fills the table before any reader runs; a Color is 0 or 1 and a Square is below 64, so both
    // indices are in bounds.
    unsafe {
        *(*core::ptr::addr_of!(PAWN_ATTACKS_STORAGE).cast::<[[Bitboard; SQUARE_NB]; COLOR_NB]>())
            .get_unchecked(color as usize)
            .get_unchecked(sq as usize)
    }
}

/// Returns the single (non-double) pawn push of the given color for the squares
/// in `b`. Matches C++ `pawn_single_push_bb(Color c, Bitboard b)`.
#[inline(always)]
pub fn pawn_single_push_bb(c: Color, b: Bitboard) -> Bitboard {
    if c == Color::White {
        shift(b, NORTH)
    } else {
        shift(b, SOUTH)
    }
}

/// Builds the "pawn pair" table at compile time.
/// C++: `inline constexpr auto PawnPairBB = [](){ ... }()` in bitboard.h.
/// For each square: own file plus the two adjacent files, restricted to ranks
/// 2-7, excluding the square itself. The geometry is color-independent.
const fn build_pawn_pair_bb() -> [Bitboard; SQUARE_NB] {
    let mut result = [0u64; SQUARE_NB];
    let mut s = 0usize;
    while s < SQUARE_NB {
        // file_bb(s); `s % 8` is file_of(s)
        let file = FILE_A_BB << (s % 8);
        // file | shift<EAST>(file) | shift<WEST>(file)
        let files = file | ((file & !FILE_H_BB) << 1) | ((file & !FILE_A_BB) >> 1);
        result[s] = files & !(RANK_1_BB | RANK_8_BB) & !(1u64 << s);
        s += 1;
    }
    result
}

/// Compile-time table backing [`pawn_pair_bb`]. Matches C++ `PawnPairBB`.
pub static PAWN_PAIR_BB: [Bitboard; SQUARE_NB] = build_pawn_pair_bb();

/// Returns the squares that can host a pawn forming a "pawn pair" with a pawn
/// on `s`: own file plus adjacent files, restricted to ranks 2-7, excluding `s`.
/// The geometry is color-independent. Matches C++ `pawn_pair_bb(Square)`.
#[inline(always)]
pub fn pawn_pair_bb(s: Square) -> Bitboard {
    PAWN_PAIR_BB[s as usize]
}

/// Computes all sliding attacks for a piece (rook/bishop) given an occupancy.
/// Stops at the first blocker or when off-board, with no wrap-around.
fn sliding_attack(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    let mut attacks = 0;
    // Four cardinal or diagonal directions
    let dirs = if pt == PieceType::Rook {
        &[NORTH, SOUTH, EAST, WEST]
    } else {
        &[NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST]
    };
    for &d in dirs {
        // start from the square
        let mut s = sq;
        // step one square at a time, only while safe_destination says "in-bounds"
        while safe_destination(s, d) != 0 {
            s += d;
            attacks |= square_bb(s);
            // stop if we've hit an occupied square
            if occ & square_bb(s) != 0 {
                break;
            }
        }
    }
    attacks
}

/// Returns the bitboard of the target square for the given step from the given
/// square. If the step is off the board, returns an empty bitboard.
///
/// The `distance <= 2` test is what stops a step wrapping around a board edge
/// into the opposite file. Mirrors C++ `safe_destination` (attacks.h).
#[inline]
fn safe_destination(s: Square, step: i8) -> Bitboard {
    let to = s as i8 + step;
    if (0..64).contains(&to) {
        let t = Square::from(to as u8);
        if SQUARE_DISTANCE.get().unwrap()[s as usize][t as usize] <= 2 {
            return square_bb(t);
        }
    }
    0
}

// Matches C++ `line_mask` (attacks.cpp), used by both hyperbola-quintessence variants.
#[inline(always)]
fn dual_line_mask(sq: Square, first: Direction, second: Direction) -> Bitboard {
    let mut mask = 0u64;
    for direction in [first, second] {
        let mut current = sq;
        loop {
            let destination = safe_destination(current, direction);
            if destination == 0 {
                break;
            }
            mask |= destination;
            current = Square::from(destination.trailing_zeros() as u8);
        }
    }
    mask
}

fn build_dual_magics() -> AlignedDualMagics {
    AlignedDualMagics(core::array::from_fn(|square| {
        let sq = Square::from(square as u8);
        DualMagic {
            mask_file: dual_line_mask(sq, NORTH, SOUTH),
            mask_diag: dual_line_mask(sq, NORTH_EAST, SOUTH_WEST),
            mask_antidiag: dual_line_mask(sq, NORTH_WEST, SOUTH_EAST),
            mask_none: 0,
            r: (1u64 << square).wrapping_mul(2),
            rr: (1u64 << (63 - square)).wrapping_mul(2),
            rank_attacks: &RANK_ATTACKS[file_of(sq) as usize],
            shift: (8 * rank_of(sq) as usize) as u32,
        }
    }))
}

#[inline(always)]
fn get_dual_magics() -> &'static AlignedDualMagics {
    ensure_initialized();
    // SAFETY: `init()` writes the table before any game logic starts, and the
    // storage remains immutable for the lifetime of the process.
    unsafe { &*core::ptr::addr_of!(DUAL_MAGICS_STORAGE).cast::<AlignedDualMagics>() }
}

/// Builds empty-board pseudo-attacks for the non-pawn piece types, plus the between, line and
/// ray-pass tables. Mirrors the C++ initialization loops.
fn build_pseudo_and_between(
    pseudo: &mut [[Bitboard; SQUARE_NB]; PIECE_TYPE_NB],
    between: &mut [[Bitboard; SQUARE_NB]; SQUARE_NB],
    line: &mut [[Bitboard; SQUARE_NB]; SQUARE_NB],
    ray_pass: &mut [[Bitboard; SQUARE_NB]; SQUARE_NB],
) {
    for s1_u in 0u8..64 {
        let s1 = Square::from(s1_u);
        // The pawn slot is deliberately left empty; see `attacks_empty`.
        //
        // C++ reuses the first index of `PseudoAttacks` as a *colour* for pawns
        // -- `PseudoAttacks[WHITE][s]` and `PseudoAttacks[BLACK][s]`
        // (attacks.h) -- which works only because `WHITE`/`BLACK` are 0/1
        // and no piece type occupies slot 0. That pun does not survive a port to
        // typed indices, so pawn attacks are computed by `pawn_attacks()`
        // instead of being tabulated here.
        pseudo[PieceType::Pawn as usize][s1_u as usize] = 0;
        // King moves
        for &step in &[-9, -8, -7, -1, 1, 7, 8, 9] {
            pseudo[PieceType::King as usize][s1_u as usize] |= safe_destination(s1, step);
        }
        // Knight moves
        for &step in &[-17, -15, -10, -6, 6, 10, 15, 17] {
            pseudo[PieceType::Knight as usize][s1_u as usize] |= safe_destination(s1, step);
        }
        // Bishop, Rook, Queen empty-board
        pseudo[PieceType::Bishop as usize][s1_u as usize] =
            sliding_attack(PieceType::Bishop, s1, 0);
        pseudo[PieceType::Rook as usize][s1_u as usize] = sliding_attack(PieceType::Rook, s1, 0);
        pseudo[PieceType::Queen as usize][s1_u as usize] = pseudo[PieceType::Bishop as usize]
            [s1_u as usize]
            | pseudo[PieceType::Rook as usize][s1_u as usize];
        // Between/Line/RayPass tables
        for s2_u in 0u8..64 {
            let s2 = Square::from(s2_u);
            let mut ln = 0;
            let mut btw = 0;
            let mut rp = 0;
            for &pt in &[PieceType::Bishop, PieceType::Rook] {
                if pseudo[pt as usize][s1_u as usize] & square_bb(s2) != 0 {
                    ln = sliding_attack(pt, s1, 0) & sliding_attack(pt, s2, 0)
                        | square_bb(s1)
                        | square_bb(s2);
                    btw = sliding_attack(pt, s1, square_bb(s2))
                        & sliding_attack(pt, s2, square_bb(s1));
                    // C++: RayPassBB[s1][s2] = attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2);
                    rp = sliding_attack(pt, s1, 0)
                        & (sliding_attack(pt, s2, square_bb(s1)) | square_bb(s2));
                }
            }
            // C++ does: BetweenBB[s1][s2] = attacks_bb(pt, s1, square_bb(s2)) & attacks_bb(pt, s2, square_bb(s1));
            //           BetweenBB[s1][s2] |= s2;
            between[s1_u as usize][s2_u as usize] = btw | square_bb(s2);
            line[s1_u as usize][s2_u as usize] = ln;
            ray_pass[s2_u as usize][s1_u as usize] = rp;
        }
    }
}

/// Initializes various bitboard tables. Called at program startup.
pub fn init() {
    INIT_ONCE.call_once(|| {
        // Small tables
        POPCNT16.get_or_init(|| {
            let mut pc16 = Box::new([0u8; 1 << 16]);
            for (i, v) in pc16.iter_mut().enumerate() {
                *v = (i as u16).count_ones() as u8;
            }
            pc16
        });
        SQUARE_DISTANCE.get_or_init(|| {
            let mut dist = Box::new([[0u8; SQUARE_NB]; SQUARE_NB]);
            for s1 in 0..SQUARE_NB {
                for s2 in 0..SQUARE_NB {
                    let df = (file_of(Square::from(s1 as u8)) as i8
                        - file_of(Square::from(s2 as u8)) as i8)
                        .abs();
                    let dr = (rank_of(Square::from(s1 as u8)) as i8
                        - rank_of(Square::from(s2 as u8)) as i8)
                        .abs();
                    dist[s1][s2] = cmp::max(df, dr) as u8;
                }
            }
            dist
        });

        let dual_magics = build_dual_magics();

        let diag = core::array::from_fn(|s| {
            let sq = Square::from(s as u8);
            dual_line_mask(sq, NORTH_EAST, SOUTH_WEST) | square_bb(sq)
        });
        let antidiag = core::array::from_fn(|s| {
            let sq = Square::from(s as u8);
            dual_line_mask(sq, NORTH_WEST, SOUTH_EAST) | square_bb(sq)
        });
        // SAFETY: one-time initialization before readers; each cast matches the storage type.
        unsafe {
            core::ptr::addr_of_mut!(DIAG_BB_STORAGE)
                .cast::<[Bitboard; SQUARE_NB]>()
                .write(diag);
            core::ptr::addr_of_mut!(ANTIDIAG_BB_STORAGE)
                .cast::<[Bitboard; SQUARE_NB]>()
                .write(antidiag);
        }

        // Heavy tables
        let mut between = [[0u64; SQUARE_NB]; SQUARE_NB];
        let mut line = [[0u64; SQUARE_NB]; SQUARE_NB];
        let mut ray_pass = [[0u64; SQUARE_NB]; SQUARE_NB];
        let mut pseudo = [[0u64; SQUARE_NB]; PIECE_TYPE_NB];
        build_pseudo_and_between(&mut pseudo, &mut between, &mut line, &mut ray_pass);
        // SAFETY: runs once inside `INIT_ONCE.call_once` before any reader exists (readers require `init()` to have
        // run); each cast matches the static's declared MaybeUninit array type, so the writes are in bounds and aligned.
        unsafe {
            core::ptr::addr_of_mut!(BETWEEN_BB_STORAGE)
                .cast::<[[Bitboard; SQUARE_NB]; SQUARE_NB]>()
                .write(between);
            core::ptr::addr_of_mut!(LINE_BB_STORAGE)
                .cast::<[[Bitboard; SQUARE_NB]; SQUARE_NB]>()
                .write(line);
            core::ptr::addr_of_mut!(INVERSE_RAY_PASS_BB_STORAGE)
                .cast::<[[Bitboard; SQUARE_NB]; SQUARE_NB]>()
                .write(ray_pass);
        }
        // SAFETY: same as above: single-threaded one-time init through `call_once`, cast type equals the storage's type.
        unsafe {
            core::ptr::addr_of_mut!(PSEUDO_ATTACKS_STORAGE)
                .cast::<[[Bitboard; SQUARE_NB]; PIECE_TYPE_NB]>()
                .write(pseudo);
        }
        let mut pawn_attacks_table = [[0u64; SQUARE_NB]; COLOR_NB];
        for s in 0..SQUARE_NB {
            let b = square_bb(Square::from(s as u8));
            pawn_attacks_table[Color::White as usize][s] = pawn_attacks(Color::White, b);
            pawn_attacks_table[Color::Black as usize][s] = pawn_attacks(Color::Black, b);
        }
        // SAFETY: same as above: single-threaded one-time init through `call_once`, cast type equals the storage's type.
        unsafe {
            core::ptr::addr_of_mut!(PAWN_ATTACKS_STORAGE)
                .cast::<[[Bitboard; SQUARE_NB]; COLOR_NB]>()
                .write(pawn_attacks_table);
        }
        // SAFETY: same as above: one-time init under `call_once`; the cast matches the storage's
        // declared type, so the write is in bounds and aligned.
        unsafe {
            core::ptr::addr_of_mut!(DUAL_MAGICS_STORAGE)
                .cast::<AlignedDualMagics>()
                .write(dual_magics);
        }
    });
}

/// Renders a bitboard as an 8x8 ASCII board for debugging output.
pub fn pretty(bb: Bitboard) -> String {
    let mut s = String::from("+---+---+---+---+---+---+---+---+\n");
    for r in (0..8).rev() {
        for f in 0..8 {
            // Compute square index directly
            let idx = ((r as u8) << 3) + (f as u8);
            let sq = Square::from(idx);
            s.push_str(if bb & square_bb(sq) != 0 {
                "| X "
            } else {
                "|   "
            });
        }
        s.push_str(&format!("| {}\n+---+---+---+---+---+---+---+---+\n", r + 1));
    }
    s.push_str("  a   b   c   d   e   f   g   h\n");
    s
}

/// Returns a bitboard representing the given rank.
///
/// rank_bb() and file_bb() return a bitboard representing all the squares on
/// the given file or rank.
#[inline(always)]
pub fn rank_bb(r: Rank) -> Bitboard {
    rank_bb_r(r)
}

/// Returns a bitboard representing the given file.
#[inline(always)]
pub fn file_bb(f: File) -> Bitboard {
    file_bb_f(f)
}

/// Returns a bitboard representing the rank of a square.
#[inline(always)]
pub fn rank_bb_sq(s: Square) -> Bitboard {
    rank_bb_r(rank_of(s))
}

/// Returns a bitboard representing the file of a square.
#[inline(always)]
pub fn file_bb_sq(s: Square) -> Bitboard {
    file_bb_f(file_of(s))
}

/// Returns bitboard for a single square.
#[inline(always)]
pub fn square_bb_pub(s: Square) -> Bitboard {
    square_bb(s)
}

/// Full-board line (rank, file or diagonal) containing s1 and s2, or 0 if not
/// aligned. Use `between_bb_pub` for the segment.
#[inline(always)]
pub fn line_bb_pub(s1: Square, s2: Square) -> Bitboard {
    ensure_initialized();
    // SAFETY: `init()` writes the table before any game logic starts, and it
    // remains immutable for the lifetime of the process.
    unsafe {
        (*core::ptr::addr_of!(LINE_BB_STORAGE).cast::<[[Bitboard; SQUARE_NB]; SQUARE_NB]>())
            [s1 as usize][s2 as usize]
    }
}

/// Between bitboard from s1 (exclusive) to s2 (inclusive).
#[inline(always)]
pub fn between_bb_pub(s1: Square, s2: Square) -> Bitboard {
    ensure_initialized();
    // SAFETY: `init()` writes the table before any game logic starts, and it
    // remains immutable for the lifetime of the process.
    unsafe {
        (*core::ptr::addr_of!(BETWEEN_BB_STORAGE).cast::<[[Bitboard; SQUARE_NB]; SQUARE_NB]>())
            [s1 as usize][s2 as usize]
    }
}

/// Ray pass bitboard: rays from s1 that pass through s2 (or could reach s2).
/// C++: RayPassBB\[s1\]\[s2\] = attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2);
#[inline(always)]
pub fn ray_pass_bb(s1: Square, s2: Square) -> Bitboard {
    ensure_initialized();
    // SAFETY: `init()` writes the table before any game logic starts, and it
    // remains immutable for the lifetime of the process.
    unsafe {
        (*core::ptr::addr_of!(INVERSE_RAY_PASS_BB_STORAGE)
            .cast::<[[Bitboard; SQUARE_NB]; SQUARE_NB]>())[s2 as usize][s1 as usize]
    }
}

/// Full diagonal through the square, including the square itself.
#[inline(always)]
pub fn diag_bb(s: Square) -> Bitboard {
    ensure_initialized();
    // SAFETY: init() initializes this immutable table before game logic starts.
    unsafe { (*core::ptr::addr_of!(DIAG_BB_STORAGE).cast::<[Bitboard; SQUARE_NB]>())[s as usize] }
}

/// Full antidiagonal through the square, including the square itself.
#[inline(always)]
pub fn antidiag_bb(s: Square) -> Bitboard {
    ensure_initialized();
    // SAFETY: init() initializes this immutable table before game logic starts.
    unsafe {
        (*core::ptr::addr_of!(ANTIDIAG_BB_STORAGE).cast::<[Bitboard; SQUARE_NB]>())[s as usize]
    }
}

/// Compile-time-friendly population count, matching C++ `constexpr_popcount`.
/// Kept for API parity with the C++ code; on stable/nightly Rust `count_ones`
/// is already a const fn, so this simply forwards to it.
#[inline(always)]
pub const fn constexpr_popcount(b: Bitboard) -> i32 {
    b.count_ones() as i32
}

/// Returns true if s1, s2, s3 are aligned on rank/file/diagonal.
#[inline(always)]
pub fn aligned(s1: Square, s2: Square, s3: Square) -> bool {
    line_bb_pub(s1, s2) & square_bb(s3) != 0
}

/// Returns the pseudo attacks of the given piece type assuming an empty board.
///
/// Returns 0 for `Pawn`, which has no colour-independent attack set: use
/// [`pawn_attacks`] or [`attacks_bb_with_color`]. C++ answers pawn queries from
/// the same table by indexing it with a colour instead (attacks.h); see the
/// table's initialisation for why that does not carry over.
#[inline(always)]
pub fn attacks_empty(pt: PieceType, sq: Square) -> Bitboard {
    let sq = sq as usize;
    get_pseudo_attacks()[pt as usize][sq]
}

/// Pseudo attacks on an empty board, for a piece whose colour matters.
///
/// The counterpart of C++ `attacks_bb<Pt>(Square s, Color c)` (attacks.h).
/// Pawn attacks are computed rather than looked up, because the port does not
/// reproduce C++'s colour-as-piece-type indexing of `PseudoAttacks`.
pub fn attacks_bb_with_color(pt: PieceType, sq: Square, color: Color) -> Bitboard {
    let sq = sq as usize;
    match pt {
        PieceType::Pawn => pawn_attacks(color, square_bb(Square::from(sq as u8))),
        _ => get_pseudo_attacks()[pt as usize][sq],
    }
}

/// Returns the attacks by the given piece, assuming the board is occupied
/// according to `occ`. Sliding attacks do not continue past an occupied square.
#[inline(always)]
pub fn attacks_bb(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    let (bishop, rook) = dual_attacks_for_square(sq as usize, occ);
    match pt {
        PieceType::Bishop => bishop,
        PieceType::Rook => rook,
        PieceType::Queen => bishop | rook,
        _ => get_pseudo_attacks()[pt as usize][sq as usize],
    }
}

/// Bishop attacks - specialized version without runtime match (matches C++ `attacks_bb<BISHOP>`)
#[inline(always)]
pub fn attacks_bb_bishop(sq: Square, occ: Bitboard) -> Bitboard {
    dual_attacks_for_square(sq as usize, occ).0
}

/// Rook attacks - specialized version without runtime match (matches C++ `attacks_bb<ROOK>`)
#[inline(always)]
pub fn attacks_bb_rook(sq: Square, occ: Bitboard) -> Bitboard {
    dual_attacks_for_square(sq as usize, occ).1
}

/// Queen attacks - specialized version without runtime match (matches C++ `attacks_bb<QUEEN>`)
#[inline(always)]
pub fn attacks_bb_queen(sq: Square, occ: Bitboard) -> Bitboard {
    let (bishop, rook) = dual_attacks_for_square(sq as usize, occ);
    bishop | rook
}

/// Reads the rank attacks of a `DualMagic` entry: C++
/// `rankAttacksLookup[(occupied >> (shift + 1)) & 0x3f] << shift`.
#[inline(always)]
fn rank_attacks_of(magic: &DualMagic, occupied: Bitboard) -> Bitboard {
    let row = magic.rank_attacks[((occupied >> (magic.shift + 1)) & 0x3f) as usize];
    Bitboard::from(row) << magic.shift
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
// Entirely AVX2 intrinsics on a 32-byte aligned `DualMagic` entry.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dual_attacks(magic: &DualMagic, occupied: Bitboard) -> (Bitboard, Bitboard) {
    use core::arch::x86_64::*;

    let rank_attacks = rank_attacks_of(magic, occupied);

    // Byteswap within 128-bit elements (C++ `bswap` in `DualMagic::both_attacks_bb`).
    let bswap_mask = _mm256_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15,
    );
    let bswap = |value: __m256i| _mm256_shuffle_epi8(value, bswap_mask);

    // Each lane contains a mask and we follow the scalar algorithm below. The first four
    // fields are exactly the four 64-bit lanes loaded by the C++
    // `_mm256_load_si256(reinterpret_cast<const __m256i*>(this))`.
    let mask = _mm256_load_si256((&magic.mask_file as *const Bitboard).cast::<__m256i>());
    let occupied_vec = _mm256_set1_epi64x(occupied as i64);
    let r = _mm256_set1_epi64x(magic.r as i64);
    let rr = _mm256_set1_epi64x(magic.rr as i64);

    let occupied_vec = _mm256_and_si256(mask, occupied_vec);
    let forward = _mm256_sub_epi64(occupied_vec, r);
    let reverse = bswap(_mm256_sub_epi64(bswap(occupied_vec), rr));
    let result = _mm256_and_si256(_mm256_xor_si256(forward, reverse), mask);

    // Lane 0: rook attacks (file only); lanes 1 and 2: bishop attacks. Permuting
    // lanes (0, 2, 1, 3) puts the antidiagonal next to the diagonal, so lane 1
    // of the OR is the combined bishop attack (upstream 205f0052).
    let permuted = _mm256_permute4x64_epi64::<0b11_01_10_00>(result);
    let rook_bishop = _mm256_or_si256(result, permuted);
    let bishop = _mm_extract_epi64::<1>(_mm256_castsi256_si128(rook_bishop)) as u64;
    // The rook's file attack is lane 0 of `result`, read straight from it so the
    // OR above is not on this path's critical chain (upstream 82d73d0a).
    let rook =
        (_mm_cvtsi128_si64(_mm256_castsi256_si128(result)) as u64).wrapping_add(rank_attacks);

    // [bishop, rook]
    (bishop, rook)
}

/// The scalar `DualMagic::both_attacks_bb` (C++ `#else` branch). Compiled into AVX2 test builds
/// too, so the host's tests cover it.
#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn dual_attacks_scalar(magic: &DualMagic, occupied: Bitboard) -> (Bitboard, Bitboard) {
    let rank_attacks = rank_attacks_of(magic, occupied);

    let hyperbola = |mask: Bitboard| {
        let o = occupied & mask;
        (o.wrapping_sub(magic.r) ^ o.swap_bytes().wrapping_sub(magic.rr).swap_bytes()) & mask
    };

    // [bishop, rook]
    (
        hyperbola(magic.mask_diag) | hyperbola(magic.mask_antidiag),
        hyperbola(magic.mask_file).wrapping_add(rank_attacks),
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn dual_attacks_for_square(sq: usize, occupied: Bitboard) -> (Bitboard, Bitboard) {
    // SAFETY: this function is cfg-gated on `target_feature = "avx2"` (compile time), so the AVX2
    // intrinsics are available; `init()` publishes one 32-byte aligned DualMagic entry per square
    // and the array index is bounds-checked.
    unsafe { dual_attacks(&get_dual_magics().0[sq], occupied) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn dual_attacks_for_square(sq: usize, occupied: Bitboard) -> (Bitboard, Bitboard) {
    dual_attacks_scalar(&get_dual_magics().0[sq], occupied)
}

/// Returns `(bishop_attacks, rook_attacks)` for a square in a single call.
/// Matches C++ `both_attacks_bb(Square s, Bitboard occupied)` (attacks.h).
///
/// We always compute [bishop, rook] attacks at once, then rely on
/// compiler's DCE and CSE to eliminate unneeded re-computations or extractions.
///
/// When using hyperbola quintessence, file, diagonal and antidiagonal attacks
/// can use a byte reversal rather than a full bit reversal (because all squares
/// reside in different bytes). Rank attacks cannot. Thus, for rank attacks
/// only, we use a compact lookup table indexed by the 6 inner bits of the rank's
/// occupancy (the edge squares never affect the attack set).
#[inline(always)]
pub fn both_attacks_bb(sq: Square, occ: Bitboard) -> (Bitboard, Bitboard) {
    dual_attacks_for_square(sq as usize, occ)
}

/// Knight attacks - specialized version without runtime match (matches C++ `attacks_bb<KNIGHT>`)
#[inline(always)]
pub fn attacks_bb_knight(sq: Square) -> Bitboard {
    let sq = sq as usize;
    get_pseudo_attacks()[PieceType::Knight as usize][sq]
}

/// King attacks - specialized version without runtime match (matches C++ `attacks_bb<KING>`)
#[inline(always)]
pub fn attacks_bb_king(sq: Square) -> Bitboard {
    let sq = sq as usize;
    get_pseudo_attacks()[PieceType::King as usize][sq]
}

/// Reference sliding attacks by ray stepping (no hyperbola quintessence); equivalent to
/// `attacks_bb`. Not used on hot paths.
pub fn attacks_bb_scan(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    match pt {
        PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
            let mut attacks = 0u64;
            let dirs: &[i8] = match pt {
                PieceType::Bishop => &[NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST],
                PieceType::Rook => &[NORTH, SOUTH, EAST, WEST],
                PieceType::Queen => &[
                    NORTH, SOUTH, EAST, WEST, NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST,
                ],
                _ => unreachable!(),
            };
            for &d in dirs {
                let mut s = sq;
                loop {
                    let tobb = safe_destination(s, d);
                    if tobb == 0 {
                        break;
                    }
                    let t = s + d;
                    attacks |= tobb;
                    if occ & tobb != 0 {
                        break;
                    }
                    s = t;
                }
            }
            attacks
        }
        _ => attacks_bb(pt, sq, occ),
    }
}

/// Returns the number of set bits in a bitboard.
#[inline(always)]
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

/// Least significant bit index.
#[inline(always)]
pub fn lsb(bb: Bitboard) -> Square {
    lsb_nonzero(core::num::NonZeroU64::new(bb).expect("lsb requires a nonzero bitboard"))
}

/// Least significant bit index of a nonzero bitboard.
#[inline(always)]
pub fn lsb_nonzero(bb: core::num::NonZeroU64) -> Square {
    Square::from(bb.trailing_zeros() as u8)
}

/// Most significant bit index.
///
/// Returns the index of the most significant set bit of a non-zero bitboard.
#[inline(always)]
pub fn msb(bb: Bitboard) -> Square {
    assert!(bb != 0, "msb requires a nonzero bitboard");
    Square::from(63 - bb.leading_zeros() as u8)
}

/// Returns the bitboard of the least significant square of a non-zero bitboard.
/// Equivalent to `square_bb(lsb(bb))`.
#[inline(always)]
pub fn least_significant_square_bb(bb: Bitboard) -> Bitboard {
    debug_assert!(bb != 0);
    bb.isolate_lowest_one()
}

/// Pop and return least significant square.
///
/// Finds and clears the least significant bit in a non-zero bitboard.
#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> Square {
    let s = lsb(*bb);
    *bb &= *bb - 1;
    s
}

/// Returns true if the bitboard has more than one bit set.
#[inline(always)]
pub fn more_than_one(bb: Bitboard) -> bool {
    bb & (bb.wrapping_sub(1)) != 0
}

/// Returns |file(x) - file(y)|.
#[inline(always)]
pub fn distance_file(x: Square, y: Square) -> i32 {
    (file_of(x) as i8 - file_of(y) as i8).abs() as i32
}

/// Returns |rank(x) - rank(y)|.
#[inline(always)]
pub fn distance_rank(x: Square, y: Square) -> i32 {
    (rank_of(x) as i8 - rank_of(y) as i8).abs() as i32
}

/// Returns the chess distance (king move distance) between two squares
#[inline(always)]
pub fn distance_square(x: Square, y: Square) -> u8 {
    SQUARE_DISTANCE.get().unwrap()[x as usize][y as usize]
}

/// Returns the minimum distance from file to the board edges
#[inline(always)]
pub fn edge_distance(f: File) -> i32 {
    cmp::min(f as i32, File::H as i32 - f as i32)
}

// Not ported from C++ `bitboard.h`, deliberately:
//
// * The `Bitboard`/`Square` operator overloads (bitboard.h). `Bitboard` is
//   a plain `u64` here, so Rust's orphan rule forbids inherent operators on it.
//   `square_bb`, `&` and `|=` cover the same ground at the call sites.
// * The `template<Direction>` / `template<Color>` / `template<PieceType>` forms
//   of `shift`, `pawn_attacks_bb`, `distance` and `attacks_bb`. Each exists here
//   as a runtime-argument function (`shift`, `pawn_attacks`, `distance_file` /
//   `distance_rank` / `distance_square`, `attacks_bb`), which the callers all
//   invoke with constants, so LLVM specialises them the same way.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::PRNG;
    use proptest::prelude::*;

    #[test]
    fn test_popcnt16() {
        init();
        let tbl = POPCNT16.get().unwrap();
        assert_eq!(tbl[0], 0);
        assert_eq!(tbl[0xFFFF], 16);
        assert_eq!(tbl[0b1010_1010_1010_1010], 8);
    }

    #[test]
    fn test_square_distance() {
        init();
        let d = SQUARE_DISTANCE.get().unwrap();
        let a1 = Square::SQ_A1 as usize;
        let h8 = Square::SQ_H8 as usize;
        assert_eq!(d[a1][h8], 7);
        assert_eq!(d[a1][a1], 0);
    }

    #[test]
    fn test_pretty_single_square() {
        init();
        let bb = 1u64 << (Square::SQ_E4 as u8);
        let s = pretty(bb);
        assert_eq!(s.matches('X').count(), 1);
    }

    #[test]
    fn test_popcount_wrapper() {
        assert_eq!(popcount(0b1011), 3);
        assert_eq!(popcount(0), 0);
    }

    #[test]
    fn test_lsb_msb() {
        let bb = (1u64 << 5) | (1u64 << 60);
        assert_eq!(lsb(bb), Square::from(5));
        assert_eq!(msb(bb), Square::from(60));
    }

    #[test]
    fn test_least_significant_square_bb() {
        let bb = (1u64 << 10) | (1u64 << 20);
        assert_eq!(least_significant_square_bb(bb), 1u64 << 10);
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = (1u64 << 2) | (1u64 << 4);
        let sq = pop_lsb(&mut bb);
        assert_eq!(sq, Square::from(2));
        assert_eq!(bb, 1u64 << 4);
    }

    #[test]
    fn test_pretty_empty_and_rank_file() {
        init();
        let empty = pretty(0);
        assert_eq!(empty.matches('X').count(), 0);

        let rank1_str = pretty(RANK_1_BB);
        assert_eq!(rank1_str.matches('X').count(), 8);

        let file_a_str = pretty(FILE_A_BB);
        assert_eq!(file_a_str.matches('X').count(), 8);
    }

    #[test]
    fn test_attacks_on_empty_board() {
        init();
        let e4 = Square::SQ_E4;
        let rook = attacks_bb(PieceType::Rook, e4, 0);
        let bishop = attacks_bb(PieceType::Bishop, e4, 0);
        assert_eq!(popcount(rook), 14);
        assert_eq!(popcount(bishop), 13);
        // Queen equals union
        let queen = attacks_bb(PieceType::Queen, e4, 0);
        assert_eq!(queen, rook | bishop);
    }

    #[test]
    fn test_pawn_attacks_single() {
        init();
        let e2 = Square::SQ_E2;
        let white = pawn_attacks(Color::White, square_bb_pub(e2));
        assert_eq!(
            white,
            square_bb_pub(Square::SQ_D3) | square_bb_pub(Square::SQ_F3)
        );
        let e7 = Square::SQ_E7;
        let black = pawn_attacks(Color::Black, square_bb_pub(e7));
        assert_eq!(
            black,
            square_bb_pub(Square::SQ_D6) | square_bb_pub(Square::SQ_F6)
        );
    }

    #[test]
    fn test_line_between_and_aligned() {
        init();
        let c4 = Square::SQ_C4;
        let f7 = Square::SQ_F7;
        let ln = line_bb_pub(c4, f7);
        assert!(ln & square_bb_pub(Square::SQ_E6) != 0);
        assert!(aligned(c4, f7, Square::SQ_E6));
        let btw = between_bb_pub(c4, f7);
        assert_eq!(
            btw & square_bb_pub(Square::SQ_D5),
            square_bb_pub(Square::SQ_D5)
        );
        assert_eq!(
            btw & square_bb_pub(Square::SQ_E6),
            square_bb_pub(Square::SQ_E6)
        );
        assert_eq!(
            btw & square_bb_pub(Square::SQ_F7),
            square_bb_pub(Square::SQ_F7)
        );
    }

    #[test]
    fn test_more_than_one() {
        assert!(!more_than_one(0));
        assert!(!more_than_one(1));
        assert!(more_than_one(3));
    }

    proptest! {
        #[test]
        fn prop_lsb_msb_and_least_square(bb in any::<u64>().prop_filter("non-zero", |b| *b != 0)) {
            let l = lsb(bb) as u8;
            let m = msb(bb) as u8;
            prop_assert!(l <= m);
            prop_assert_eq!(least_significant_square_bb(bb), 1u64 << l);
            prop_assert!((bb >> l) & 1 == 1);
            prop_assert!((bb >> m) & 1 == 1);
        }

        #[test]
        fn prop_pop_lsb_counts_bits(mut bb in any::<u64>()) {
            let orig = bb;
            let mut c = 0u32;
            while bb != 0 { let _ = pop_lsb(&mut bb); c += 1; }
            prop_assert_eq!(c, popcount(orig));
        }

        #[test]
        fn prop_more_than_one_equiv(bb in any::<u64>()) {
            prop_assert_eq!(more_than_one(bb), popcount(bb) > 1);
        }

        #[test]
        fn prop_queen_union_rook_bishop(sq in 0u8..64, occ in any::<u64>()) {
            init();
            let s = Square::from(sq);
            let r = attacks_bb(PieceType::Rook, s, occ);
            let b = attacks_bb(PieceType::Bishop, s, occ);
            let q = attacks_bb(PieceType::Queen, s, occ);
            prop_assert_eq!(q, r | b);
        }

        #[test]
        fn prop_pawn_attacks_union(color in prop_oneof![Just(Color::White), Just(Color::Black)], bb in any::<u64>()) {
            init();
            // Shifted by hand rather than via `attacks_empty`, which returns 0
            // for pawns by design -- see that function.
            let ref_union = match color {
                Color::White => (bb & !FILE_A_BB) << 7 | (bb & !FILE_H_BB) << 9,
                Color::Black => (bb & !FILE_H_BB) >> 7 | (bb & !FILE_A_BB) >> 9,
            };
            prop_assert_eq!(pawn_attacks(color, bb), ref_union);
        }

        #[test]
        fn prop_square_distance(s1 in 0u8..64, s2 in 0u8..64) {
            init();
            let a = Square::from(s1);
            let b = Square::from(s2);
            let df = (file_of(a) as i8 - file_of(b) as i8).unsigned_abs();
            let dr = (rank_of(a) as i8 - rank_of(b) as i8).unsigned_abs();
            let d = SQUARE_DISTANCE.get().unwrap()[s1 as usize][s2 as usize];
            prop_assert_eq!(d, df.max(dr));
        }
    }

    // Naive reference sliding attack for tests
    fn naive_slide(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
        let mut attacks = 0u64;
        let dirs: &[i8] = if pt == PieceType::Rook {
            &[NORTH, SOUTH, EAST, WEST]
        } else {
            &[NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST]
        };
        for &d in dirs {
            let mut s = sq;
            loop {
                let tobb = safe_destination(s, d);
                if tobb == 0 {
                    break;
                }
                // we know it's in-bounds, so compute Square and advance
                let t = s + d;
                attacks |= square_bb_pub(t);
                if occ & square_bb_pub(t) != 0 {
                    break;
                }
                s = t;
            }
        }
        attacks
    }

    #[test]
    fn e2e_sliding_attacks_vs_naive_random_occ() {
        init();
        let mut rng = PRNG::new(0xDEADBEEFCAFEBABE);
        for s_u in 0u8..64 {
            let s = Square::from(s_u);
            // Empty and full boards first: the borrow of the hyperbola-quintessence
            // subtraction propagates furthest on these.
            for i in 0..202 {
                let occ = match i {
                    0 => 0,
                    1 => !0,
                    _ => rng.rand::<u64>(),
                };
                let bro = attacks_bb(PieceType::Rook, s, occ);
                let brexp = naive_slide(PieceType::Rook, s, occ);
                assert_eq!(bro, brexp, "rook at {:?}", s);
                let bbi = attacks_bb(PieceType::Bishop, s, occ);
                let bbexp = naive_slide(PieceType::Bishop, s, occ);
                assert_eq!(bbi, bbexp, "bishop at {:?}", s);
                let q = attacks_bb(PieceType::Queen, s, occ);
                assert_eq!(q, bro | bbi, "queen union at {:?}", s);
            }
        }
    }

    // The engine uses the AVX2 kernel on this target; check it against the scalar one, which
    // the other targets use.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn dual_sliding_attacks_match_scalar_kernel() {
        init();
        let dual_magics = get_dual_magics();
        let mut rng = PRNG::new(0xA11CE5EED1234567);

        for square in 0..SQUARE_NB {
            for i in 0..258 {
                let occupied = match i {
                    0 => 0,
                    1 => !0,
                    _ => rng.rand::<u64>(),
                };
                let expected = dual_attacks_scalar(&dual_magics.0[square], occupied);
                // SAFETY: this test is cfg-gated on `target_feature = "avx2"` (compile time), so the AVX2
                // intrinsics in `dual_attacks` are available; the entry is a valid, 32-byte aligned DualMagic.
                let actual = unsafe { dual_attacks(&dual_magics.0[square], occupied) };
                assert_eq!(actual, expected, "square {square}, occupancy {occupied:#x}");
            }
        }
    }

    #[test]
    fn between_and_line_tables() {
        init();
        for s1_u in 0u8..64 {
            let s1 = Square::from(s1_u);
            for s2_u in 0u8..64 {
                let s2 = Square::from(s2_u);
                let ln = line_bb_pub(s1, s2);
                if ln == 0 {
                    // not aligned
                    assert_eq!(between_bb_pub(s1, s2), square_bb_pub(s2));
                } else {
                    // For aligned squares, ensure both ends are set in line
                    assert!(ln & square_bb_pub(s1) != 0);
                    assert!(ln & square_bb_pub(s2) != 0);
                    // Between equals squares on the segment excluding s1
                    let btw = between_bb_pub(s1, s2);
                    assert!(btw & square_bb_pub(s2) != 0);
                    assert!(btw & square_bb_pub(s1) == 0);
                }
            }
        }
        let c4 = Square::SQ_C4;
        let f7 = Square::SQ_F7;
        assert_eq!(
            between_bb_pub(c4, f7),
            square_bb_pub(Square::SQ_D5) | square_bb_pub(Square::SQ_E6) | square_bb_pub(f7)
        );
    }
}
