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

//! Core chess types and constants. Port of upstream `types.h`.

use core::hash::Hasher;
use core::{
    cmp::Ordering,
    hash::Hash,
    mem,
    ops::{Add, AddAssign, BitAnd, Not, Sub, SubAssign},
};

// External crate solely for lightweight bit-flag support.
use bitflags::bitflags;

// ---------------------------------------------------------------------------
// Primitive aliases
// ---------------------------------------------------------------------------

pub type Key = u64;
pub type Bitboard = u64;

// ---------------------------------------------------------------------------
// Search-wide constants
// ---------------------------------------------------------------------------

pub const MAX_MOVES: usize = 256;
pub const MAX_PLY: i32 = 246;
/// Length of the `StateInfo` ring buffer in `Position`; must be a power of two.
pub const MAX_HIST: usize = 512;
// search depth + 50 move rule hist (C++: MAX_HIST >= MAX_PLY + 10 + 50 * 2 + 1)
const _: () = assert!(MAX_HIST.is_power_of_two() && MAX_HIST > MAX_PLY as usize + 10 + 50 * 2);

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

/// Side to move or piece colour (C++ `Color`).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const WHITE: Color = Color::White;
    pub const BLACK: Color = Color::Black;
}

pub const COLOR_NB: usize = 2;

impl Not for Color {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // toggles colour
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

// ---------------------------------------------------------------------------
// Castling rights (bit-flags)
// ---------------------------------------------------------------------------

bitflags! {
    /// Castling rights as bit flags (C++ `CastlingRights`).
    #[derive(Default, PartialEq, Eq, Debug, Copy, Clone)]
    pub struct CastlingRights: u8 {
        const NO_CASTLING  = 0;
        const WHITE_OO     = 1 << 0;
        const WHITE_OOO    = 1 << 1;
        const BLACK_OO     = 1 << 2;
        const BLACK_OOO    = 1 << 3;

        const KING_SIDE      = Self::WHITE_OO.bits() | Self::BLACK_OO.bits();
        const QUEEN_SIDE     = Self::WHITE_OOO.bits() | Self::BLACK_OOO.bits();
        const WHITE_CASTLING = Self::WHITE_OO.bits() | Self::WHITE_OOO.bits();
        const BLACK_CASTLING = Self::BLACK_OO.bits()  | Self::BLACK_OOO.bits();
        const ANY_CASTLING   = Self::WHITE_CASTLING.bits() | Self::BLACK_CASTLING.bits();
    }
}
pub const CASTLING_RIGHT_NB: usize = 16;

// Mask castling rights by side:  `color & CastlingRights` in C++
impl BitAnd<CastlingRights> for Color {
    type Output = CastlingRights;
    #[inline(always)]
    fn bitand(self, rhs: CastlingRights) -> CastlingRights {
        match self {
            Color::White => CastlingRights::WHITE_CASTLING & rhs,
            Color::Black => CastlingRights::BLACK_CASTLING & rhs,
        }
    }
}

// ---------------------------------------------------------------------------
// Bound flags (exact / upper / lower)
// ---------------------------------------------------------------------------

bitflags! {
    /// Transposition-table bound of a stored value (C++ `Bound`).
    #[derive(Default, PartialEq, Eq, Debug, Copy, Clone)]
    pub struct Bound: u8 {
        const NONE  = 0;
        const UPPER = 1 << 0;
        const LOWER = 1 << 1;
        const EXACT = Self::UPPER.bits() | Self::LOWER.bits();
    }
}

// ---------------------------------------------------------------------------
// Evaluation values
// ---------------------------------------------------------------------------

/// Value is used as an alias for int, this is done to differentiate between a search
/// value and any other integer value. The values used in search are always supposed
/// to be in the range (-VALUE_NONE, VALUE_NONE] and should not exceed this range.
pub type Value = i32;

pub const VALUE_ZERO: Value = 0;
pub const VALUE_DRAW: Value = 0;
pub const VALUE_NONE: Value = 32_002;
pub const VALUE_INFINITE: Value = 32_001;

pub const VALUE_MATE: Value = 32_000;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;

pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY;
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;

/// Returns true if the given value is not VALUE_NONE.
#[inline(always)]
pub const fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

/// Returns true if the given value is a win (>= VALUE_TB_WIN_IN_MAX_PLY).
#[inline(always)]
pub const fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// Returns true if the given value is a loss (<= VALUE_TB_LOSS_IN_MAX_PLY).
#[inline(always)]
pub const fn is_loss(v: Value) -> bool {
    v <= VALUE_TB_LOSS_IN_MAX_PLY
}

/// Returns true if the given value is decisive (win or loss).
#[inline(always)]
pub const fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

/// Returns true if the given value is a mate score (>= VALUE_MATE_IN_MAX_PLY).
#[inline(always)]
pub const fn is_mate(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v >= VALUE_MATE_IN_MAX_PLY
}

/// Returns true if the given value is a mated score (<= VALUE_MATED_IN_MAX_PLY).
#[inline(always)]
pub const fn is_mated(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v <= VALUE_MATED_IN_MAX_PLY
}

/// Returns true if the given value is either a mate or mated score.
#[inline(always)]
pub const fn is_mate_or_mated(v: Value) -> bool {
    is_mate(v) || is_mated(v)
}

// Material constants - tuned so `non_pawn_material()` is injective w.r.t. piece set.
/// In the code, we make the assumption that these values
/// are such that non_pawn_material() can be used to uniquely
/// identify the material on the board.
pub const PAWN_VALUE: Value = 208;
pub const KNIGHT_VALUE: Value = 781;
pub const BISHOP_VALUE: Value = 825;
pub const ROOK_VALUE: Value = 1_276;
pub const QUEEN_VALUE: Value = 2_538;

// ---------------------------------------------------------------------------
// Piece, piece type & helpers
// ---------------------------------------------------------------------------

/// Piece kind without colour (C++ `PieceType`); `NoPieceType` (0) also indexes "all pieces".
#[derive(Copy, Clone, Eq, PartialEq, Debug, PartialOrd)]
#[repr(u8)]
pub enum PieceType {
    NoPieceType = 0,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    /// Never a real piece type: exists only so that `type_of(Piece::Reserved7 | Reserved15)`
    /// (`pc & 7`) and `PieceType::from(v & 7)` are total, with no branch.
    Reserved7,
}

pub const PIECE_TYPE_NB: usize = 8;
pub const ALL_PIECES: PieceType = PieceType::NoPieceType;

impl PieceType {
    /// Returns the next piece type in the enumeration, wrapping to NoPieceType.
    #[inline(always)]
    pub fn succ(self) -> Self {
        match self {
            PieceType::NoPieceType => PieceType::Pawn,
            PieceType::Pawn => PieceType::Knight,
            PieceType::Knight => PieceType::Bishop,
            PieceType::Bishop => PieceType::Rook,
            PieceType::Rook => PieceType::Queen,
            PieceType::Queen => PieceType::King,
            PieceType::King | PieceType::Reserved7 => PieceType::NoPieceType,
        }
    }

    /// Returns the previous piece type, wrapping from NoPieceType to King.
    #[inline(always)]
    pub fn pred(self) -> Self {
        match self {
            PieceType::NoPieceType => PieceType::King,
            PieceType::Pawn => PieceType::NoPieceType,
            PieceType::Knight => PieceType::Pawn,
            PieceType::Bishop => PieceType::Knight,
            PieceType::Rook => PieceType::Bishop,
            PieceType::Queen => PieceType::Rook,
            PieceType::King => PieceType::Queen,
            PieceType::Reserved7 => PieceType::King,
        }
    }
}
/// Coloured piece (C++ `Piece`): white pieces are 1-6, black pieces 9-14.
#[allow(non_camel_case_types)] // W_Pawn.. mirror the C++ Piece enumerators
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Piece {
    NoPiece = 0,
    // White
    W_Pawn = PieceType::Pawn as u8,
    W_Knight,
    W_Bishop,
    W_Rook,
    W_Queen,
    W_King,
    // Never real pieces. They make every 4-bit value a valid `Piece`, so `Piece::from`
    // is sound for e.g. `make_piece(Black, NoPieceType)` and LLVM sees the range 0..=15.
    Reserved7,
    Reserved8,
    // Black (offset by 8)
    B_Pawn = PieceType::Pawn as u8 + 8,
    B_Knight,
    B_Bishop,
    B_Rook,
    B_Queen,
    B_King,
    Reserved15,
}
pub const PIECE_NB: usize = 16;

// Swap color of piece B_KNIGHT <-> W_KNIGHT
impl Not for Piece {
    type Output = Piece;
    #[inline(always)]
    fn not(self) -> Piece {
        // Toggle white <-> black by flipping bit 3.
        Piece::from(self as u8 ^ 8)
    }
}

impl Piece {
    /// Crate-internal conversion for values known to be in `0..16` (engine arithmetic such as
    /// `make_piece`). Not public: external code must use [`Piece::new`] (checked) or the total
    /// `From<u8>` (keeps the low 4 bits).
    ///
    /// The range is only checked by a debug assertion, so callers must pass `v < 16`.
    #[inline(always)]
    pub(crate) const fn from(v: u8) -> Self {
        debug_assert!(v < PIECE_NB as u8);
        // SAFETY: every value below 16 is a variant (see the reserved variants above); the
        // caller's contract is `v < 16`.
        unsafe { Self::from_unchecked(v) }
    }

    /// Checked conversion for external input: `None` unless `v < 16`.
    #[inline]
    pub const fn new(v: u8) -> Option<Self> {
        if v < PIECE_NB as u8 {
            // SAFETY: just checked `v < 16`.
            Some(unsafe { Self::from_unchecked(v) })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// `v` must be below 16.
    #[inline(always)]
    pub(crate) const unsafe fn from_unchecked(v: u8) -> Self {
        debug_assert!(v < PIECE_NB as u8);
        // SAFETY: the caller guarantees `v < 16`; every such value is a variant.
        unsafe { mem::transmute::<u8, Piece>(v) }
    }
}

pub const PIECE_VALUE: [Value; PIECE_NB] = [
    VALUE_ZERO,
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    VALUE_ZERO,
    VALUE_ZERO,
    VALUE_ZERO,
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    VALUE_ZERO,
    VALUE_ZERO,
];

/// Unchecked access to PIECE_VALUE.
///
/// # Safety
///
/// `pc` must be a valid piece (0-15).
#[inline(always)]
pub unsafe fn piece_value_unchecked(pc: Piece) -> Value {
    debug_assert!((pc as usize) < PIECE_NB);
    // SAFETY: the caller guarantees pc < PIECE_NB.
    unsafe { *PIECE_VALUE.get_unchecked(pc as usize) }
}

#[inline(always)]
pub const fn type_of(pc: Piece) -> PieceType {
    // SAFETY: `pc as u8 & 7` is in 0..=7 and PieceType has a variant for every value in 0..=7
    // (7 is the placeholder `Reserved7`, reached only from Piece::Reserved7/Reserved15).
    unsafe { mem::transmute::<u8, PieceType>(pc as u8 & 7) }
}

#[inline(always)]
pub fn color_of(pc: Piece) -> Color {
    debug_assert!(pc != Piece::NoPiece);
    if (pc as u8) >> 3 == 0 {
        Color::White
    } else {
        Color::Black
    }
}

// ---------------------------------------------------------------------------
// Search depth & constants
// ---------------------------------------------------------------------------

pub type Depth = i32;

/// The following DEPTH_ constants are used for transposition table entries
/// and quiescence search move generation stages. In regular search, the
/// depth stored in the transposition table is literal: the search depth
/// (effort) used to make the corresponding transposition table value. In
/// quiescence search, however, the transposition table entries only store
/// the current quiescence move generation stage (which should thus compare
/// lower than any regular search depth).
pub const DEPTH_QS: Depth = 0;
// For transposition table entries where no searching at all was done
// (whether regular or qsearch) we use DEPTH_UNSEARCHED, which should thus
// compare lower than any quiescence or regular depth. DEPTH_NONE is used
// for the transposition table entry occupancy check (see tt.rs), and
// should thus be lower than DEPTH_UNSEARCHED.
pub const DEPTH_UNSEARCHED: Depth = -2;
pub const DEPTH_NONE: Depth = -3;

// ---------------------------------------------------------------------------
// Board squares
// ---------------------------------------------------------------------------

/// Board square (C++ `Square`), numbered rank by rank from A1 (0) to H8 (63).
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
#[repr(u8)]
#[allow(non_camel_case_types)] // SQ_A1.. mirror the C++ Square enumerators
pub enum Square {
    SQ_A1,
    SQ_B1,
    SQ_C1,
    SQ_D1,
    SQ_E1,
    SQ_F1,
    SQ_G1,
    SQ_H1,
    SQ_A2,
    SQ_B2,
    SQ_C2,
    SQ_D2,
    SQ_E2,
    SQ_F2,
    SQ_G2,
    SQ_H2,
    SQ_A3,
    SQ_B3,
    SQ_C3,
    SQ_D3,
    SQ_E3,
    SQ_F3,
    SQ_G3,
    SQ_H3,
    SQ_A4,
    SQ_B4,
    SQ_C4,
    SQ_D4,
    SQ_E4,
    SQ_F4,
    SQ_G4,
    SQ_H4,
    SQ_A5,
    SQ_B5,
    SQ_C5,
    SQ_D5,
    SQ_E5,
    SQ_F5,
    SQ_G5,
    SQ_H5,
    SQ_A6,
    SQ_B6,
    SQ_C6,
    SQ_D6,
    SQ_E6,
    SQ_F6,
    SQ_G6,
    SQ_H6,
    SQ_A7,
    SQ_B7,
    SQ_C7,
    SQ_D7,
    SQ_E7,
    SQ_F7,
    SQ_G7,
    SQ_H7,
    SQ_A8,
    SQ_B8,
    SQ_C8,
    SQ_D8,
    SQ_E8,
    SQ_F8,
    SQ_G8,
    SQ_H8,
}

pub const SQUARE_ZERO: Square = Square::SQ_A1;
pub const SQUARE_NB: usize = 64;

pub type Direction = i8;

pub const NORTH: Direction = 8;
pub const EAST: Direction = 1;
pub const SOUTH: Direction = -NORTH;
pub const WEST: Direction = -EAST;

pub const NORTH_EAST: Direction = NORTH + EAST;
pub const SOUTH_EAST: Direction = SOUTH + EAST;
pub const SOUTH_WEST: Direction = SOUTH + WEST;
pub const NORTH_WEST: Direction = NORTH + WEST;

/// Board file A-H (C++ `File`).
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Debug)]
#[repr(u8)]
pub enum File {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}
pub const FILE_NB: usize = 8;

/// Board rank 1-8 (C++ `Rank`).
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Debug)]
#[repr(u8)]
pub enum Rank {
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
}
pub const RANK_NB: usize = 8;

// Utility helpers on squares / files / ranks
impl Square {
    /// Crate-internal conversion for values known to be in `0..64` (engine arithmetic).
    /// Not public: external code must use [`Square::new`].
    ///
    /// The range is only checked by a debug assertion, so callers must pass `v < 64`.
    #[inline(always)]
    pub(crate) const fn from(v: u8) -> Self {
        debug_assert!(v < SQUARE_NB as u8);
        // SAFETY: the caller's contract is `v < 64`, and Square has exactly the 64 variants 0..=63.
        unsafe { Self::from_unchecked(v) }
    }

    /// Checked conversion for external input: `None` unless `v < 64`.
    #[inline]
    pub const fn new(v: u8) -> Option<Self> {
        if v < SQUARE_NB as u8 {
            // SAFETY: just checked `v < 64`.
            Some(unsafe { Self::from_unchecked(v) })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// `v` must be below 64.
    #[inline(always)]
    pub(crate) const unsafe fn from_unchecked(v: u8) -> Self {
        debug_assert!(v < SQUARE_NB as u8);
        // SAFETY: the caller guarantees `v < 64`; Square has exactly the 64 variants 0..=63.
        unsafe { mem::transmute::<u8, Square>(v) }
    }

    #[inline(always)]
    pub fn offset(self, d: Direction) -> Option<Self> {
        let result = self as i8 + d;
        if (0..SQUARE_NB as i8).contains(&result) {
            Some(Square::from(result as u8))
        } else {
            None // out of the board
        }
    }
}

impl Add<Direction> for Square {
    type Output = Square;
    #[inline(always)]
    fn add(self, rhs: Direction) -> Square {
        debug_assert!(
            self.offset(rhs).is_some(),
            "Square + Direction left the board"
        );
        Square::from(((self as i8 + rhs) & 63) as u8)
    }
}

impl Sub<Direction> for Square {
    type Output = Square;
    #[inline(always)]
    fn sub(self, rhs: Direction) -> Square {
        debug_assert!(
            self.offset(-rhs).is_some(),
            "Square - Direction left the board"
        );
        Square::from(((self as i8 - rhs) & 63) as u8)
    }
}

impl AddAssign<Direction> for Square {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Direction) {
        *self = *self + rhs;
    }
}

impl SubAssign<Direction> for Square {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Direction) {
        *self = *self - rhs;
    }
}

// Lightly-checked conversions to ease migration of parsing/FFI code while
// remaining zero-cost in release builds.
impl From<u8> for Piece {
    /// Total: keeps the low 4 bits (an out-of-range value is a bug, caught in debug builds).
    /// Use [`Piece::new`] to reject it instead.
    #[inline(always)]
    fn from(v: u8) -> Self {
        debug_assert!(v < PIECE_NB as u8);
        // SAFETY: `v & 15 < 16`.
        unsafe { Piece::from_unchecked(v & 15) }
    }
}

impl PieceType {
    /// Checked conversion for external input: `None` unless `v < 8` (7 is the placeholder
    /// `Reserved7`).
    #[inline]
    pub const fn new(v: u8) -> Option<Self> {
        if v < 8 {
            // SAFETY: just checked `v < 8`; every value in 0..=7 is a PieceType variant.
            Some(unsafe { mem::transmute::<u8, PieceType>(v) })
        } else {
            None
        }
    }
}

impl File {
    /// Checked conversion for external input: `None` unless `v < 8`.
    #[inline]
    pub const fn new(v: u8) -> Option<Self> {
        if v < FILE_NB as u8 {
            // SAFETY: just checked `v < 8`; File has exactly the 8 variants 0..=7.
            Some(unsafe { mem::transmute::<u8, File>(v) })
        } else {
            None
        }
    }
}

impl Rank {
    /// Checked conversion for external input: `None` unless `v < 8`.
    #[inline]
    pub const fn new(v: u8) -> Option<Self> {
        if v < RANK_NB as u8 {
            // SAFETY: just checked `v < 8`; Rank has exactly the 8 variants 0..=7.
            Some(unsafe { mem::transmute::<u8, Rank>(v) })
        } else {
            None
        }
    }
}

// Conversions that are total: out-of-range input is a bug (debug assertion) and keeps only the
// low bits in release builds, never undefined behaviour. Use `new` for external input.
impl From<u8> for PieceType {
    #[inline(always)]
    fn from(v: u8) -> Self {
        debug_assert!(v <= PieceType::King as u8);
        // SAFETY: `v & 7` is in 0..=7 and every value in that range is a PieceType variant (7 is
        // `Reserved7`), so the transmute is total even for out-of-contract input.
        unsafe { mem::transmute::<u8, PieceType>(v & 7) }
    }
}

impl From<u8> for File {
    #[inline(always)]
    fn from(v: u8) -> Self {
        debug_assert!(v < FILE_NB as u8);
        // SAFETY: `v & 7` is in 0..=7 and File has exactly the 8 variants 0..=7.
        unsafe { mem::transmute::<u8, File>(v & 7) }
    }
}

impl From<u8> for Rank {
    #[inline(always)]
    fn from(v: u8) -> Self {
        debug_assert!(v < RANK_NB as u8);
        // SAFETY: `v & 7` is in 0..=7 and Rank has exactly the 8 variants 0..=7.
        unsafe { mem::transmute::<u8, Rank>(v & 7) }
    }
}

// ---------------------------------------------------------------------------
// Square helpers
// ---------------------------------------------------------------------------

#[inline(always)]
pub const fn is_ok(s: Square) -> bool {
    (s as u8) < 64
}

#[inline(always)]
pub const fn file_of(sq: Square) -> File {
    // SAFETY: `sq as u8` is < 64 for any Square, so `& 7` is in 0..=7, all File variants.
    unsafe { mem::transmute::<u8, File>((sq as u8) & 7) }
}

#[inline(always)]
pub const fn rank_of(sq: Square) -> Rank {
    // SAFETY: `sq as u8` is < 64 for any Square, so `>> 3` is in 0..=7, all Rank variants.
    unsafe { mem::transmute::<u8, Rank>((sq as u8) >> 3) }
}

/// Returns the square mirrored vertically (A1 <-> A8).
#[inline(always)]
pub const fn flip_rank(s: Square) -> Square {
    Square::from((s as u8) ^ (Square::SQ_A8 as u8))
}

/// Returns the square mirrored horizontally (A1 <-> H1).
#[inline(always)]
pub const fn flip_file(s: Square) -> Square {
    Square::from((s as u8) ^ (Square::SQ_H1 as u8))
}

/// Returns the square relative to the given color (flip ranks for black).
#[inline(always)]
pub fn relative_square(c: Color, s: Square) -> Square {
    Square::from((s as i8 ^ ((c as i8) * 56)) as u8)
}

/// Returns the rank relative to the given color (R1 <-> R8 for black).
#[inline(always)]
pub const fn relative_rank(c: Color, r: Rank) -> Rank {
    // SAFETY: r is 0..=7 and c*7 is 0 or 7, so the XOR stays within 0..=7 (flips 0<->7 etc.), all Rank variants.
    unsafe { mem::transmute::<u8, Rank>((r as u8) ^ ((c as u8) * 7)) }
}

/// Returns the rank of the square relative to the given color.
#[inline(always)]
pub fn relative_rank_sq(c: Color, s: Square) -> Rank {
    relative_rank(c, rank_of(s))
}

// C++ overloads `relative_rank(Color, Square)`; the Rust counterpart is `relative_rank_sq`.

/// Pawn push direction for a given colour.
#[inline(always)]
pub fn pawn_push(c: Color) -> Direction {
    if c == Color::White { NORTH } else { SOUTH }
}

// ---------------------------------------------------------------------------
// DirtyPiece helper (for NNUE)
// ---------------------------------------------------------------------------

/// Keep track of what a move changes on the board (used by NNUE)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct DirtyPiece {
    // pc is never NO_PIECE for real moves; default is NoPiece for reset.
    pub pc: Piece,
    // from/to of the moved piece; `to` is None for promotions (matches C++).
    pub from: Option<Square>,
    pub to: Option<Square>,
    // For captures/castling/promotions: extra remove/add slots.
    pub remove_sq: Option<Square>,
    pub add_sq: Option<Square>,
    pub remove_pc: Piece,
    pub add_pc: Piece,
}

impl Default for DirtyPiece {
    fn default() -> Self {
        Self::new()
    }
}

impl DirtyPiece {
    pub const fn new() -> Self {
        Self {
            pc: Piece::NoPiece,
            from: None,
            to: None,
            remove_sq: None,
            // if {add,remove}_sq is None, {add,remove}_pc is allowed to be
            // uninitialized
            // castling uses add_sq and remove_sq to remove and add the rook
            add_sq: None,
            remove_pc: Piece::NoPiece,
            add_pc: Piece::NoPiece,
        }
    }
}

// ---------------------------------------------------------------------------
// Dirty Threats for SFNNv10
// ---------------------------------------------------------------------------

/// A single threat: attacker at pc_sq attacks threatened_pc at threatened_sq
/// Data is packed into a u32 for efficiency
#[derive(Copy, Clone, Debug, Default)]
pub struct DirtyThreat {
    data: u32,
}

/// Keep track of what threats change on the board (used by NNUE)
impl DirtyThreat {
    // Bit offsets for packed data
    pub const PC_SQ_OFFSET: u32 = 0;
    pub const THREATENED_SQ_OFFSET: u32 = 8;
    pub const THREATENED_PC_OFFSET: u32 = 16;
    pub const PC_OFFSET: u32 = 20;
    const ADD_OFFSET: u32 = 31;

    pub fn new(
        pc: Piece,
        threatened_pc: Piece,
        pc_sq: Square,
        threatened_sq: Square,
        add: bool,
    ) -> Self {
        let data = ((add as u32) << Self::ADD_OFFSET)
            | ((pc as u32) << Self::PC_OFFSET)
            | ((threatened_pc as u32) << Self::THREATENED_PC_OFFSET)
            | ((threatened_sq as u32) << Self::THREATENED_SQ_OFFSET)
            | ((pc_sq as u32) << Self::PC_SQ_OFFSET);
        Self { data }
    }

    pub fn pc(&self) -> Piece {
        Piece::from(((self.data >> Self::PC_OFFSET) & 0xf) as u8)
    }

    pub fn threatened_pc(&self) -> Piece {
        Piece::from(((self.data >> Self::THREATENED_PC_OFFSET) & 0xf) as u8)
    }

    pub fn threatened_sq(&self) -> Square {
        Square::from(((self.data >> Self::THREATENED_SQ_OFFSET) & 0xff) as u8)
    }

    pub fn pc_sq(&self) -> Square {
        Square::from(((self.data >> Self::PC_SQ_OFFSET) & 0xff) as u8)
    }

    pub fn add(&self) -> bool {
        (self.data >> Self::ADD_OFFSET) != 0
    }

    /// Raw data accessor for vectorized operations
    pub fn raw(&self) -> u32 {
        self.data
    }
}

/// Maximum number of dirty threats per move
/// A piece can be involved in at most 8 outgoing attacks and 16 incoming attacks.
/// Moving a piece also can reveal at most 8 discovered attacks.
/// This implies that a non-castling move can change at most (8 + 16) * 3 + 8 = 80 features.
/// By similar logic, a castling move can change at most (5 + 1 + 3 + 9) * 2 = 36 features.
/// 16 entries added to accommodate unmasked vector stores near end of list.
pub const MAX_DIRTY_THREATS: usize = 96;

/// List of dirty threats
#[derive(Clone)]
pub struct DirtyThreatList {
    pub(crate) data: [DirtyThreat; MAX_DIRTY_THREATS],
    pub(crate) len: usize,
}

impl Default for DirtyThreatList {
    fn default() -> Self {
        Self {
            data: [DirtyThreat::default(); MAX_DIRTY_THREATS],
            len: 0,
        }
    }
}

impl DirtyThreatList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a threat without a release-mode bounds check.
    ///
    /// Crate-internal contract: callers (the threat generator) stay below
    /// `MAX_DIRTY_THREATS` entries by construction.
    #[inline(always)]
    pub(crate) fn push(&mut self, threat: DirtyThreat) {
        // The threat generator has a fixed, proven upper bound below
        // MAX_DIRTY_THREATS. Keep the invariant in debug builds, but match
        // Stockfish's fixed-capacity C++ list in release builds by avoiding a
        // bounds branch on every hot-path append.
        debug_assert!(self.len < MAX_DIRTY_THREATS);
        // SAFETY: len < MAX_DIRTY_THREATS: the threat generator's fixed upper bound (debug_assert-checked, not
        // checked in release) keeps the index inside `data`.
        unsafe {
            *self.data.get_unchecked_mut(self.len) = threat;
        }
        self.len += 1;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Reserves `count` entries and returns a pointer to the first one; the caller
    /// may store up to `MAX_DIRTY_THREATS - len` entries through it (C++
    /// `ValueList::make_space`). The extra 16 entries of capacity allow unmasked
    /// 64-byte stores near the end of the list.
    ///
    /// # Safety
    /// `len + count <= MAX_DIRTY_THREATS` must hold.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2"
    ))]
    #[inline(always)]
    pub unsafe fn make_space(&mut self, count: usize) -> *mut DirtyThreat {
        debug_assert!(self.len + count <= MAX_DIRTY_THREATS);
        // SAFETY: caller guarantees len + count <= MAX_DIRTY_THREATS >= len, so len is in bounds
        // (or one past the end).
        let p = unsafe { self.data.as_mut_ptr().add(self.len) };
        self.len += count;
        p
    }

    pub fn iter(&self) -> impl Iterator<Item = &DirtyThreat> {
        self.data[..self.len].iter()
    }

    /// Number of threats in the list.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the list holds no threats.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The threats in the list as a slice.
    pub fn as_slice(&self) -> &[DirtyThreat] {
        &self.data[..self.len]
    }
}

/// Complete dirty threats data for a position change
#[derive(Clone, Default)]
pub struct DirtyThreats {
    pub list: DirtyThreatList,
}

// ---------------------------------------------------------------------------
// Dirty pawn pairs for SFNNv16
// The pawn-pair feature block needs the pawn bitboards of both
// colors before and after the move, so that the accumulator can incrementally
// update the pawn-pair features.
// ---------------------------------------------------------------------------

/// C++ `struct DirtyPawnPairs { Bitboard before[COLOR_NB]; Bitboard after[COLOR_NB]; }`
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct DirtyPawnPairs {
    /// Pawn bitboards (indexed by color) before the move.
    pub before: [Bitboard; COLOR_NB],
    /// Pawn bitboards (indexed by color) after the move.
    pub after: [Bitboard; COLOR_NB],
}

impl DirtyPawnPairs {
    pub const fn new() -> Self {
        Self {
            before: [0; COLOR_NB],
            after: [0; COLOR_NB],
        }
    }
}

/// Aggregate of everything a move dirties, handed to `Position::do_move`.
/// C++ `struct Dirties { DirtyPiece dirtyPiece; DirtyThreats dirtyThreats; DirtyPawnPairs dirtyPawnPairs; }`
#[derive(Clone)]
pub struct Dirties {
    pub dirty_piece: DirtyPiece,
    pub dirty_threats: DirtyThreats,
    pub dirty_pawn_pairs: DirtyPawnPairs,
}

impl Default for Dirties {
    fn default() -> Self {
        Self {
            dirty_piece: DirtyPiece::new(),
            dirty_threats: DirtyThreats::default(),
            dirty_pawn_pairs: DirtyPawnPairs::new(),
        }
    }
}

impl Dirties {
    pub fn new() -> Self {
        Self::default()
    }

    /// Equivalent of the C++ placement-new resets performed before reusing a
    /// scratch `Dirties` (`new (&d.dirtyThreats) DirtyThreats;` and
    /// `new (&d.dirtyPawnPairs) DirtyPawnPairs;`): the threat list is emptied,
    /// the pawn-pair bitboards are fully overwritten by `do_move` anyway.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.dirty_threats.list.clear();
    }
}

// ---------------------------------------------------------------------------
// Random key generator
// ---------------------------------------------------------------------------

/// Based on a congruential pseudo-random number generator
#[inline(always)]
pub const fn make_key(seed: u64) -> Key {
    // Must match C++ types.h `make_key` exactly:
    //     seed * 6364136223846793005ULL + 1442695040888963407ULL
    // 6364136223846793005 == 0x5851F42D4C957F2D
    // 1442695040888963407 == 0x14057B7EF767814F
    seed.wrapping_mul(0x5851F42D4C957F2D)
        .wrapping_add(0x14057B7EF767814F)
}

// ---------------------------------------------------------------------------
// Move representation - 16-bit packed value
// ---------------------------------------------------------------------------

/// A move needs 16 bits to be stored
///
/// bit  0- 5: destination square (from 0 to 63)
/// bit  6-11: origin square (from 0 to 63)
/// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
/// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
/// NOTE: en passant bit is set only when a pawn can be captured
///
/// Special cases are Move::none() and Move::null(). We can sneak these in because
/// in any normal move the destination square and origin square are always different,
/// but Move::none() and Move::null() have the same origin and destination square.
#[derive(Copy, Clone, Eq, PartialEq, Default, Debug)]
#[repr(transparent)]
pub struct Move(pub u16);

/// Special move kind stored in bits 14-15 of a [`Move`] (C++ `MoveType`).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u16)]
pub enum MoveType {
    Normal = 0,
    Promotion = 1 << 14,
    EnPassant = 2 << 14,
    Castling = 3 << 14,
}

impl Move {
    // Bit positions of the from/to squares within the packed 16-bit value.
    pub const FROM_SQ_SHIFT: u16 = 6;
    pub const TO_SQ_SHIFT: u16 = 0;

    #[inline(always)]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    #[inline(always)]
    pub const fn from_to(from: Square, to: Square) -> Self {
        Self(((from as u16) << 6) | (to as u16))
    }

    /// Create a pawn move from 'to' square and direction offset.
    /// Equivalent to C++: Move(to - offset, to)
    #[inline(always)]
    pub const fn pawn_move(to: Square, offset: i8) -> Self {
        let from = (to as i8 - offset) as u16;
        Self((from << 6) | (to as u16))
    }

    #[inline(always)]
    pub const fn make<const T: u16>(from: Square, to: Square, pt: PieceType) -> Self {
        Self(
            T | (((pt as u16 - PieceType::Knight as u16) & 3) << 12)
                | ((from as u16) << 6)
                | (to as u16),
        )
    }

    // Ergonomic typed constructors mirroring common C++ call sites
    #[inline(always)]
    pub const fn make_normal(from: Square, to: Square) -> Self {
        Self::make::<{ MoveType::Normal as u16 }>(from, to, PieceType::Knight)
    }

    #[inline(always)]
    pub const fn make_promotion(from: Square, to: Square, pt: PieceType) -> Self {
        debug_assert!(matches!(
            pt,
            PieceType::Knight | PieceType::Bishop | PieceType::Rook | PieceType::Queen
        ));
        Self::make::<{ MoveType::Promotion as u16 }>(from, to, pt)
    }

    #[inline(always)]
    pub const fn make_en_passant(from: Square, to: Square) -> Self {
        Self::make::<{ MoveType::EnPassant as u16 }>(from, to, PieceType::Knight)
    }

    #[inline(always)]
    pub const fn make_castling(from: Square, to: Square) -> Self {
        Self::make::<{ MoveType::Castling as u16 }>(from, to, PieceType::Knight)
    }

    #[inline(always)]
    pub fn from_sq(self) -> Square {
        debug_assert!(self.is_ok());
        Square::from(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline(always)]
    pub fn to_sq(self) -> Square {
        debug_assert!(self.is_ok());
        Square::from((self.0 & 0x3F) as u8)
    }

    // C++ spells this `from_to()` and returns `int`.
    #[inline(always)]
    pub const fn from_to_bits(self) -> u16 {
        self.0 & 0x0FFF
    }

    // C++ spells this `type_of()`; renamed to avoid colliding with the free
    // `type_of(Piece)`.
    #[inline(always)]
    pub const fn move_type(self) -> MoveType {
        // SAFETY: `& (3 << 14)` leaves only bits 14/15, giving exactly 0, 1<<14, 2<<14, 3<<14, the four MoveType discriminants.
        unsafe { mem::transmute::<u16, MoveType>(self.0 & (3 << 14)) }
    }

    #[inline(always)]
    pub fn promotion_type(self) -> PieceType {
        // SAFETY: `((raw >> 12) & 3)` is 0..=3, so the value is Knight(2)+0..=3 = 2..=5 (Knight..Queen), all valid PieceType variants.
        unsafe {
            mem::transmute::<u8, PieceType>(((self.0 >> 12) & 3) as u8 + PieceType::Knight as u8)
        }
    }

    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub const fn none() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub const fn null() -> Self {
        Self(65)
    }

    #[inline(always)]
    pub fn is_ok(self) -> bool {
        self.0 != 0 && self.0 != 65
    }
}

// Cheap conversions for interop and IO code paths
impl From<u16> for Move {
    #[inline(always)]
    fn from(value: u16) -> Self {
        Move(value)
    }
}
impl From<Move> for u16 {
    #[inline(always)]
    fn from(m: Move) -> u16 {
        m.0
    }
}

impl Hash for Move {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(make_key(self.0 as u64));
    }
}

impl PartialOrd for Move {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Move {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

#[inline(always)]
pub const fn make_piece(c: Color, pt: PieceType) -> Piece {
    Piece::from(((c as u8) << 3) + pt as u8)
}

#[inline(always)]
pub const fn mate_in(ply: i32) -> Value {
    VALUE_MATE - ply
}

#[inline(always)]
pub const fn mated_in(ply: i32) -> Value {
    -VALUE_MATE + ply
}

/// Construct a square from a file and rank.
pub const fn make_square(f: File, r: Rank) -> Square {
    // (r << 3) + f  computes rank*8 + file
    debug_assert!((f as u8) < 8);
    debug_assert!((r as u8) < 8);
    Square::from(((r as u8) << 3) + (f as u8))
}

// ---------------------------------------------------------------------------
// Compile-time size/layout checks
// ---------------------------------------------------------------------------

const _: () = assert!(size_of::<Piece>() == size_of::<u8>());
const _: () = assert!(size_of::<PieceType>() == size_of::<u8>());
const _: () = assert!(size_of::<Square>() == size_of::<u8>());
const _: () = assert!(size_of::<Color>() == size_of::<u8>());
const _: () = assert!(size_of::<Move>() == size_of::<u16>());
const _: () = assert!(size_of::<DirtyPiece>() == size_of::<[u8; 7]>());

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_color_toggle() {
        assert_eq!(!Color::White, Color::Black);
        assert_eq!(!Color::Black, Color::White);
    }

    #[test]
    fn test_castling_rights_mask() {
        assert_eq!(
            Color::White & CastlingRights::WHITE_OO,
            CastlingRights::WHITE_OO
        );
        assert!((Color::Black & CastlingRights::WHITE_OO).is_empty());
    }

    #[test]
    fn test_bound_flags() {
        assert!(Bound::UPPER.contains(Bound::UPPER));
        assert!(Bound::EXACT.contains(Bound::UPPER));
        assert!(Bound::EXACT.contains(Bound::LOWER));
        assert_eq!(Bound::EXACT, Bound::UPPER | Bound::LOWER);
    }

    #[test]
    fn test_value_functions() {
        assert!(is_valid(VALUE_ZERO));
        assert!(!is_valid(VALUE_NONE));
        assert!(is_win(VALUE_TB_WIN_IN_MAX_PLY));
        assert!(is_loss(VALUE_TB_LOSS_IN_MAX_PLY));
        assert!(is_decisive(VALUE_TB_WIN_IN_MAX_PLY));
        assert!(!is_decisive(VALUE_DRAW));
    }

    #[test]
    fn test_piece_values() {
        assert_eq!(PIECE_VALUE[Piece::W_Pawn as usize], PAWN_VALUE);
        assert_eq!(PIECE_VALUE[Piece::B_Queen as usize], QUEEN_VALUE);
        assert_eq!(PIECE_VALUE[Piece::W_King as usize], VALUE_ZERO);
        assert_eq!(PIECE_VALUE[Piece::B_King as usize], VALUE_ZERO);
    }

    #[test]
    fn test_enum_succ_pred() {
        assert_eq!(PieceType::Pawn.succ(), PieceType::Knight);
        assert_eq!(PieceType::NoPieceType.pred(), PieceType::King);
    }

    #[test]
    fn test_square_ops() {
        let e4 = Square::from(28);
        assert!(is_ok(e4));
        assert_eq!(file_of(e4), File::E);
        assert_eq!(rank_of(e4), Rank::R4);
        assert_eq!(e4 + NORTH, Square::SQ_E5);
        let mut s = e4;
        s += EAST;
        assert_eq!(s, Square::SQ_F4);
    }

    #[test]
    fn test_flip_and_relative() {
        assert_eq!(flip_rank(Square::SQ_A1), Square::SQ_A8);
        assert_eq!(flip_file(Square::SQ_A1), Square::SQ_H1);
        assert_eq!(relative_square(Color::Black, Square::SQ_A2), Square::SQ_A7);
        assert_eq!(relative_rank(Color::Black, Rank::R2), Rank::R7);
        assert_eq!(relative_rank_sq(Color::Black, Square::SQ_B2), Rank::R7);
    }

    #[test]
    fn test_move_pack_unpack() {
        let mv = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        assert_eq!(mv.from_sq(), Square::SQ_E2);
        assert_eq!(mv.to_sq(), Square::SQ_E4);
        assert!(mv.is_ok());
        assert_eq!(Move::none().raw(), 0);
        assert_eq!(Move::null().raw(), 65);
    }

    #[test]
    fn test_move_display_none_and_null() {
        assert_eq!(Move::none().to_string(), "none");
        assert_eq!(Move::null().to_string(), "0000");
        assert_eq!(
            Move::from_to(Square::SQ_E2, Square::SQ_E4).to_string(),
            "e2e4"
        );
    }

    #[test]
    fn test_dirty_threat_pack_unpack() {
        let sq_pairs = [(0u8, 63u8), (17, 42), (63, 0)];
        for pc_v in 0..16u8 {
            for tpc_v in 0..16u8 {
                for &(a, b) in &sq_pairs {
                    for add in [true, false] {
                        let pc = Piece::new(pc_v).unwrap();
                        let tpc = Piece::new(tpc_v).unwrap();
                        let (pc_sq, t_sq) = (Square::from(a), Square::from(b));
                        let dt = DirtyThreat::new(pc, tpc, pc_sq, t_sq, add);
                        assert_eq!(dt.pc(), pc);
                        assert_eq!(dt.threatened_pc(), tpc);
                        assert_eq!(dt.pc_sq(), pc_sq);
                        assert_eq!(dt.threatened_sq(), t_sq);
                        assert_eq!(dt.add(), add);
                    }
                }
            }
        }
    }

    #[test]
    fn test_move_types_and_promotions() {
        // Normal
        let m1 = Move::make_normal(Square::SQ_E2, Square::SQ_E4);
        assert_eq!(m1.move_type(), MoveType::Normal);

        // Castling
        let mc = Move::make_castling(Square::SQ_E1, Square::SQ_G1);
        assert_eq!(mc.move_type(), MoveType::Castling);

        // En passant
        let mep = Move::make_en_passant(Square::SQ_E5, Square::SQ_D6);
        assert_eq!(mep.move_type(), MoveType::EnPassant);

        // Promotion variants
        for &pt in &[
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let mp = Move::make_promotion(Square::SQ_A7, Square::SQ_A8, pt);
            assert_eq!(mp.move_type(), MoveType::Promotion);
            assert_eq!(mp.promotion_type(), pt);
        }
    }

    #[test]
    fn test_make_piece_and_color_type() {
        let wp = make_piece(Color::White, PieceType::Pawn);
        let bq = make_piece(Color::Black, PieceType::Queen);
        assert_eq!(color_of(wp), Color::White);
        assert_eq!(type_of(wp), PieceType::Pawn);
        assert_eq!(color_of(bq), Color::Black);
        assert_eq!(type_of(bq), PieceType::Queen);
    }

    #[test]
    fn test_pawn_push_direction() {
        assert_eq!(pawn_push(Color::White), NORTH);
        assert_eq!(pawn_push(Color::Black), SOUTH);
    }

    #[test]
    fn test_sizes_and_layout() {
        // ABI/layout sanity for future FFI/ASM
        assert_eq!(size_of::<Piece>(), 1);
        assert_eq!(size_of::<PieceType>(), 1);
        assert_eq!(size_of::<Square>(), 1);
        assert_eq!(size_of::<Color>(), 1);
        assert_eq!(size_of::<Move>(), 2);
        assert_eq!(size_of::<DirtyPiece>(), 7);
    }

    #[test]
    fn test_piece_toggle_and_from_u8() {
        assert_eq!(!Piece::W_Knight, Piece::B_Knight);
        assert_eq!(!Piece::B_Bishop, Piece::W_Bishop);

        // From<u8> guards (debug only) and correctness
        let p: Piece = Piece::from(Piece::W_Queen as u8);
        assert_eq!(p, Piece::W_Queen);
        let pt: PieceType = PieceType::from(PieceType::Rook as u8);
        assert_eq!(pt, PieceType::Rook);
        let f: File = File::from(File::E as u8);
        assert_eq!(f, File::E);
        let r: Rank = Rank::from(Rank::R6 as u8);
        assert_eq!(r, Rank::R6);
    }

    #[test]
    fn test_constants_counts() {
        assert_eq!(COLOR_NB, 2);
        assert_eq!(FILE_NB, 8);
        assert_eq!(RANK_NB, 8);
        assert_eq!(SQUARE_NB, 64);
        assert_eq!(PIECE_NB, 16);
        assert_eq!(PIECE_TYPE_NB, 8);
    }

    #[test]
    fn test_square_roundtrip_and_relations() {
        for s_val in 0u8..64u8 {
            let s = Square::from(s_val);
            assert!(is_ok(s));
            let f = file_of(s) as u8;
            let r = rank_of(s) as u8;
            assert!(f < 8 && r < 8);
            assert_eq!(make_square(File::from(f), Rank::from(r)) as u8, s_val);
            // Relative square identity for white and XOR 56 for black
            assert_eq!(relative_square(Color::White, s) as u8, s_val);
            assert_eq!(relative_square(Color::Black, s) as u8, s_val ^ 56);
        }
    }

    #[test]
    fn test_castling_rights_masking() {
        assert_eq!(
            Color::Black & CastlingRights::ANY_CASTLING,
            CastlingRights::BLACK_CASTLING
        );
        assert_eq!(
            Color::White & CastlingRights::ANY_CASTLING,
            CastlingRights::WHITE_CASTLING
        );
        assert!((Color::Black & CastlingRights::WHITE_OO).is_empty());
        assert!(!(Color::White & CastlingRights::WHITE_OO).is_empty());
    }

    #[test]
    fn test_move_bit_encoding() {
        // from_to bits occupy low 12 bits irrespective of type
        let a = Move::from_to(Square::SQ_A2, Square::SQ_A4);
        let b = Move::make_castling(Square::SQ_A2, Square::SQ_A4);
        assert_eq!(a.from_to_bits(), b.from_to_bits());
        // Promotion encodes piece in bits 12..13
        let mp = Move::make_promotion(Square::SQ_A7, Square::SQ_A8, PieceType::Queen);
        assert_eq!(mp.promotion_type(), PieceType::Queen);
    }

    #[test]
    fn test_dirty_piece_defaults_and_patterns() {
        // Default should mean "no change", safe for reset paths.
        let dp = DirtyPiece::default();
        assert_eq!(dp.pc, Piece::NoPiece);
        assert_eq!(dp.from, None);
        assert_eq!(dp.to, None);
        assert_eq!(dp.remove_sq, None);
        assert_eq!(dp.add_sq, None);
        assert_eq!(dp.remove_pc, Piece::NoPiece);
        assert_eq!(dp.add_pc, Piece::NoPiece);

        // Capture pattern: moving piece from->to, plus remove captured piece at capsq.
        let mut cap = DirtyPiece {
            pc: Piece::W_Knight,
            from: Some(Square::SQ_C4),
            to: Some(Square::SQ_E5),
            ..Default::default()
        };
        cap.remove_sq = Some(Square::SQ_E5);
        cap.remove_pc = Piece::B_Pawn;
        assert_eq!(cap.to, Some(Square::SQ_E5));
        assert_eq!(cap.remove_sq, Some(Square::SQ_E5));
        assert_eq!(cap.remove_pc, Piece::B_Pawn);

        // Promotion pattern: to is None, add the promoted piece at destination.
        let mut promo = DirtyPiece {
            pc: Piece::W_Pawn,
            from: Some(Square::SQ_A7),
            to: None,
            ..Default::default()
        };
        promo.add_sq = Some(Square::SQ_A8);
        promo.add_pc = Piece::W_Queen;
        assert_eq!(promo.to, None);
        assert_eq!(promo.add_sq, Some(Square::SQ_A8));
        assert_eq!(promo.add_pc, Piece::W_Queen);

        // Castling pattern: king moves from->to, rook is removed and added.
        let mut castle = DirtyPiece {
            pc: Piece::W_King,
            from: Some(Square::SQ_E1),
            to: Some(Square::SQ_G1),
            ..Default::default()
        };
        castle.remove_sq = Some(Square::SQ_H1);
        castle.add_sq = Some(Square::SQ_F1);
        castle.remove_pc = Piece::W_Rook;
        castle.add_pc = Piece::W_Rook;
        assert_eq!(castle.pc, Piece::W_King);
        assert_eq!(castle.to, Some(Square::SQ_G1));
        assert_eq!(castle.remove_sq, Some(Square::SQ_H1));
        assert_eq!(castle.add_sq, Some(Square::SQ_F1));
        assert_eq!(castle.remove_pc, Piece::W_Rook);
        assert_eq!(castle.add_pc, Piece::W_Rook);
    }
}

// ---------------------------------------------------------------------------
// Display trait implementations for debugging and tests
// ---------------------------------------------------------------------------

impl core::fmt::Display for Square {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let file = *self as u8 % 8;
        let rank = *self as u8 / 8;
        let file_char = (b'a' + file) as char;
        let rank_char = (b'1' + rank) as char;
        write!(f, "{}{}", file_char, rank_char)
    }
}

impl core::fmt::Display for Piece {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if *self == Piece::NoPiece {
            write!(f, "NoPiece")
        } else {
            let piece_type = type_of(*self);
            let color = color_of(*self);
            let type_char = match piece_type {
                PieceType::Pawn => 'P',
                PieceType::Knight => 'N',
                PieceType::Bishop => 'B',
                PieceType::Rook => 'R',
                PieceType::Queen => 'Q',
                PieceType::King => 'K',
                _ => '?',
            };
            let color_prefix = match color {
                Color::White => 'W',
                Color::Black => 'B',
            };
            write!(f, "{}{}", color_prefix, type_char)
        }
    }
}

impl core::fmt::Display for Move {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if *self == Move::none() {
            write!(f, "none")
        } else if *self == Move::null() {
            write!(f, "0000")
        } else {
            let from = self.from_sq();
            let to = self.to_sq();
            match self.move_type() {
                MoveType::Promotion => {
                    let promo_char = match self.promotion_type() {
                        PieceType::Queen => 'q',
                        PieceType::Rook => 'r',
                        PieceType::Bishop => 'b',
                        PieceType::Knight => 'n',
                        _ => '?',
                    };
                    write!(f, "{}{}{}", from, to, promo_char)
                }
                _ => write!(f, "{}{}", from, to),
            }
        }
    }
}

#[cfg(test)]
mod checked_constructor_tests {
    use super::*;

    #[test]
    fn square_new_is_checked() {
        for v in 0..=255u8 {
            match Square::new(v) {
                Some(sq) => {
                    assert!(v < 64);
                    assert_eq!(sq as u8, v);
                }
                None => assert!(v >= 64),
            }
        }
        assert_eq!(Square::new(0), Some(Square::SQ_A1));
        assert_eq!(Square::new(63), Some(Square::SQ_H8));
        assert_eq!(Square::new(64), None);
    }

    #[test]
    fn piece_file_rank_piecetype_new_are_checked() {
        for v in 0..=255u8 {
            assert_eq!(Piece::new(v).is_some(), v < 16);
            assert_eq!(File::new(v).is_some(), v < 8);
            assert_eq!(Rank::new(v).is_some(), v < 8);
            assert_eq!(PieceType::new(v).is_some(), v < 8);
        }
        assert_eq!(Piece::new(9), Some(Piece::B_Pawn));
        assert_eq!(File::new(7), Some(File::H));
        assert_eq!(Rank::new(8), None);
    }

    /// In release builds the infallible conversions are total (they keep the low bits); in
    /// debug builds an out-of-range value trips a debug assertion instead.
    #[cfg(not(debug_assertions))]
    #[test]
    fn infallible_conversions_are_total() {
        assert_eq!(<Piece as From<u8>>::from(200) as u8, 200 & 15);
        assert_eq!(File::from(200) as u8, 200 & 7);
        assert_eq!(Rank::from(200) as u8, 200 & 7);
        assert_eq!(PieceType::from(200) as u8, 200 & 7);
    }
}
