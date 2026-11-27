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

//! Rust port of Stockfish's `position.h` header.
//!
//! Implements FEN parsing, move legality, make/undo, repetition bookkeeping,
//! static exchange evaluation, and the inline helpers used by search.

use crate::bitboard_fast as bitboard;
use crate::movegen::{GenType, generate};
use crate::syzygy::{self, ProbeState};
use crate::types::*;
use core::cell::Cell;
use core::fmt;

// ---------------------------------------------------------------------------
// Zobrist hashing tables (generated at compile time)
// ---------------------------------------------------------------------------

/// The twelve real pieces, in the order C++ `Pieces` lists them (the order in which the
/// Zobrist keys are drawn).
const PIECES: [Piece; 12] = [
    Piece::W_Pawn,
    Piece::W_Knight,
    Piece::W_Bishop,
    Piece::W_Rook,
    Piece::W_Queen,
    Piece::W_King,
    Piece::B_Pawn,
    Piece::B_Knight,
    Piece::B_Bishop,
    Piece::B_Rook,
    Piece::B_Queen,
    Piece::B_King,
];

struct ZobristTables {
    psq: [[Key; SQUARE_NB]; PIECE_NB],
    enpassant: [Key; FILE_NB],
    castling: [Key; CASTLING_RIGHT_NB],
}

impl ZobristTables {
    /// Unchecked access to psq.
    ///
    /// # Safety
    ///
    /// `piece < PIECE_NB` and `sq < SQUARE_NB` (the callers index with a `Piece`, a `Square`
    /// or the `8 + piece_count` material-key slot, all bounded by construction).
    #[inline(always)]
    unsafe fn psq_unchecked(&self, piece: usize, sq: usize) -> Key {
        debug_assert!(piece < PIECE_NB);
        debug_assert!(sq < SQUARE_NB);
        // SAFETY: bounds are the caller's contract (see above).
        unsafe { *self.psq.get_unchecked(piece).get_unchecked(sq) }
    }

    /// Unchecked access to enpassant.
    ///
    /// # Safety
    ///
    /// `file < FILE_NB`.
    #[inline(always)]
    unsafe fn enpassant_unchecked(&self, file: usize) -> Key {
        debug_assert!(file < FILE_NB);
        // SAFETY: bound is the caller's contract (see above).
        unsafe { *self.enpassant.get_unchecked(file) }
    }

    /// Unchecked access to castling.
    ///
    /// # Safety
    ///
    /// `rights < CASTLING_RIGHT_NB`.
    #[inline(always)]
    unsafe fn castling_unchecked(&self, rights: usize) -> Key {
        debug_assert!(rights < CASTLING_RIGHT_NB);
        // SAFETY: bound is the caller's contract (see above).
        unsafe { *self.castling.get_unchecked(rights) }
    }
}

/// constexpr-generated Zobrist constants (C++ `Zobrist::generate_keys`): the `PRNG(1070372)`
/// sequence, shifted left by one to leave the least significant bit to the side to move.
const fn generate_keys() -> ZobristTables {
    let mut k = ZobristTables {
        psq: [[0; SQUARE_NB]; PIECE_NB],
        enpassant: [0; FILE_NB],
        castling: [0; CASTLING_RIGHT_NB],
    };
    let mut s: u64 = 1_070_372;
    // `PRNG::rand64`
    const fn next(s: &mut u64) -> Key {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(2_685_821_657_736_338_717)
    }
    let mut p = 0;
    while p < PIECES.len() {
        let mut sq = 0;
        while sq < SQUARE_NB {
            k.psq[PIECES[p] as usize][sq] = next(&mut s) << 1;
            sq += 1;
        }
        p += 1;
    }
    // Pawns on these squares will promote: keep the impossible placements zero
    let mut f = 0;
    while f < FILE_NB {
        k.psq[Piece::W_Pawn as usize][Square::SQ_A8 as usize + f] = 0;
        k.psq[Piece::B_Pawn as usize][Square::SQ_A1 as usize + f] = 0;
        f += 1;
    }
    let mut f = 0;
    while f < FILE_NB {
        k.enpassant[f] = next(&mut s) << 1;
        f += 1;
    }
    let mut cr = 0;
    while cr < CASTLING_RIGHT_NB {
        k.castling[cr] = next(&mut s) << 1;
        cr += 1;
    }
    k
}

static ZOBRIST: ZobristTables = generate_keys();

/// The least significant bit of keys is used to store the side to move.
/// The psq, enpassant and castling keys should have that bit set to zero.
const ZOBRIST_SIDE: Key = 1;

// No combination of ep and castling key components may cancel, so that a
// matching piece placement and Zobrist key guarantee a unique position.
const _: () = {
    let z = generate_keys();
    let mut a = 0;
    while a < CASTLING_RIGHT_NB {
        let mut b = a;
        while b < CASTLING_RIGHT_NB {
            let mut f1 = 0;
            while f1 <= FILE_NB {
                // 0 means no ep component
                let mut f2 = f1;
                while f2 <= FILE_NB {
                    let cr = z.castling[a] ^ z.castling[b];
                    let ep = (if f1 > 0 { z.enpassant[f1 - 1] } else { 0 })
                        ^ (if f2 > 0 { z.enpassant[f2 - 1] } else { 0 });
                    // identical components (a == b && f1 == f2) are skipped
                    assert!(
                        (a == b && f1 == f2) || (cr ^ ep) != 0,
                        "Zobrist ep/castling keys must not cancel: matching piece placement and key guarantee a unique position"
                    );
                    f2 += 1;
                }
                f1 += 1;
            }
            b += 1;
        }
        a += 1;
    }
};

// the least significant bit is reserved for the side to move
const _: () = {
    let z = generate_keys();
    let mut any = 0;
    let mut pc = 0;
    while pc < PIECE_NB {
        let mut sq = 0;
        while sq < SQUARE_NB {
            any |= z.psq[pc][sq];
            sq += 1;
        }
        pc += 1;
    }
    let mut i = 0;
    while i < CASTLING_RIGHT_NB {
        any |= z.castling[i];
        i += 1;
    }
    let mut i = 0;
    while i < FILE_NB {
        any |= z.enpassant[i];
        i += 1;
    }
    assert!(
        any & 1 == 0,
        "Least significant bit of psq, castling and ep keys must be zero"
    );
};

#[inline(always)]
fn zobrist() -> &'static ZobristTables {
    &ZOBRIST
}

// Pack pieces into nibbles: lower half of the board to even nibbles, upper half to odd nibbles
#[inline(always)]
fn pack_pieces(board: &[Piece; SQUARE_NB]) -> [u8; 32] {
    // `Piece` is a `repr(u8)` enum below 16, so each square is one byte holding one nibble.
    let bytes: &[u8; SQUARE_NB] =
        // SAFETY: `Piece` is `repr(u8)`, so `[Piece; 64]` has the layout of `[u8; 64]`.
        unsafe { &*core::ptr::from_ref(board).cast::<[u8; SQUARE_NB]>() };
    let mut p = [0u8; 32];
    let mut k = 0;
    while k < 32 {
        let a = u64::from_ne_bytes(bytes[k..k + 8].try_into().expect("8 bytes"));
        let b = u64::from_ne_bytes(bytes[k + 32..k + 40].try_into().expect("8 bytes"));
        p[k..k + 8].copy_from_slice(&(a | (b << 4)).to_ne_bytes());
        k += 8;
    }
    p
}

#[inline(always)]
fn swap_nibble(p: &mut [u8; 32], s1: Square, s2: Square) {
    let (i1, sh1) = (s1 as usize & 31, (s1 as usize & 0x20) >> 3);
    let (i2, sh2) = (s2 as usize & 31, (s2 as usize & 0x20) >> 3);
    let swap = ((p[i1] >> sh1) ^ (p[i2] >> sh2)) & 0xF;
    p[i1] ^= swap << sh1;
    p[i2] ^= swap << sh2;
}

// Compile-time Robin Hood table of Zobrist hashes on top of Marcel van Kervinck's
// upcoming repetition algorithm for reversible moves.

// Robin Hood hashing is a variant of open addressing that reduces the variance of
// probe sequence lengths by moving keys with shorter probe sequences to make room
// for keys with longer probe sequences.
// It's O(1) on average, and O(n) in the worst case (extremely unlikely)

// References:
//   Upcoming repetition: http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf
//   Robin hood hashing:  https://cs.uwaterloo.ca/research/tr/1986/CS-86-14.pdf
const REPETITION_TABLE_SIZE: usize = 8192;
const REPETITION_TABLE_MASK: usize = REPETITION_TABLE_SIZE - 1;
#[inline(always)]
const fn repetition_hash(key: Key) -> usize {
    key as usize & REPETITION_TABLE_MASK
}

struct RepetitionTable {
    keys: [Key; REPETITION_TABLE_SIZE],
    moves: [Move; REPETITION_TABLE_SIZE],
    // Similar to probe sequence length. Zero marks an empty slot; occupied distances are one-based.
    distances: [u8; REPETITION_TABLE_SIZE],
    count: i32,
    max_distance: i32,
}

/// Empty-board attacks of a knight, bishop, rook, queen or king from `s` (C++ `PseudoAttacks`),
/// computed in a const context for the repetition table.
const fn pseudo_attacks(pt: PieceType, s: usize) -> Bitboard {
    const KNIGHT: [(i32, i32); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    const KING: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];
    const ROOK: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    const BISHOP: [(i32, i32); 4] = [(1, 1), (-1, 1), (-1, -1), (1, -1)];
    let (file, rank) = ((s % 8) as i32, (s / 8) as i32);
    let (steps, deltas): (bool, &[(i32, i32)]) = match pt {
        PieceType::Knight => (true, &KNIGHT),
        PieceType::King => (true, &KING),
        PieceType::Bishop => (false, &BISHOP),
        PieceType::Rook => (false, &ROOK),
        _ => (false, &KING),
    };
    let mut b: Bitboard = 0;
    let mut d = 0;
    while d < deltas.len() {
        let (mut f, mut r) = (file + deltas[d].0, rank + deltas[d].1);
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            b |= 1 << (r * 8 + f);
            if steps {
                break;
            }
            f += deltas[d].0;
            r += deltas[d].1;
        }
        d += 1;
    }
    if matches!(pt, PieceType::Queen) {
        b = pseudo_attacks(PieceType::Bishop, s) | pseudo_attacks(PieceType::Rook, s);
    }
    b
}

const fn generate_repetition_table() -> RepetitionTable {
    let mut t = RepetitionTable {
        keys: [0; REPETITION_TABLE_SIZE],
        moves: [Move::none(); REPETITION_TABLE_SIZE],
        distances: [0; REPETITION_TABLE_SIZE],
        count: 0,
        max_distance: 0,
    };
    let z = generate_keys();
    let mut p = 0;
    while p < PIECES.len() {
        let pc = PIECES[p];
        let pt = pc as u8 & 7;
        p += 1;
        // only non-pawn moves are reversible relocations
        if pt == PieceType::Pawn as u8 {
            continue;
        }
        let pt = match pt {
            2 => PieceType::Knight,
            3 => PieceType::Bishop,
            4 => PieceType::Rook,
            5 => PieceType::Queen,
            _ => PieceType::King,
        };
        let mut s1 = 0;
        while s1 < SQUARE_NB {
            let mut s2 = s1 + 1;
            while s2 < SQUARE_NB {
                if pseudo_attacks(pt, s1) & (1 << s2) != 0 {
                    let mut mv = Move::new(((s1 as u16) << 6) | s2 as u16);
                    let mut key = z.psq[pc as usize][s1] ^ z.psq[pc as usize][s2] ^ ZOBRIST_SIDE;
                    let mut i = repetition_hash(key);
                    let mut distance: i32 = 1;
                    loop {
                        // If the distance is too large, return an invalid table
                        if distance > 255 {
                            t.count = -1;
                            t.max_distance = distance;
                            return t;
                        }

                        // If the slot is empty, insert the key and move
                        if t.distances[i] == 0 {
                            t.keys[i] = key;
                            t.moves[i] = mv;
                            t.distances[i] = distance as u8;
                            if distance > t.max_distance {
                                t.max_distance = distance;
                            }
                            break;
                        }

                        // If the current slot has a smaller distance than the new key, swap the
                        // smaller ("poorer") key and continue probing with the displaced
                        // ("richer") key.
                        if (t.distances[i] as i32) < distance {
                            let (k, m) = (t.keys[i], t.moves[i]);
                            t.keys[i] = key;
                            t.moves[i] = mv;
                            key = k;
                            mv = m;
                            let displaced_distance = t.distances[i] as i32;
                            t.distances[i] = distance as u8;
                            distance = displaced_distance;
                            if (t.distances[i] as i32) > t.max_distance {
                                t.max_distance = t.distances[i] as i32;
                            }
                        }

                        // Advance to the next slot in the probe sequence
                        distance += 1;
                        i = (i + 1) & REPETITION_TABLE_MASK;
                    }
                    t.count += 1;
                }
                s2 += 1;
            }
            s1 += 1;
        }
    }
    t
}

static REPETITION_TABLE: RepetitionTable = generate_repetition_table();

const fn repetition_table_is_valid(t: &RepetitionTable) -> bool {
    let mut count = 0;
    let mut max_distance = 0;
    let mut i = 0;
    while i < REPETITION_TABLE_SIZE {
        let distance = t.distances[i] as i32;
        if distance != 0 {
            count += 1;
            if distance > max_distance {
                max_distance = distance;
            }
            let mut j = repetition_hash(t.keys[i]);
            let mut probe_distance = 1;
            while j != i {
                if (t.distances[j] as i32) < probe_distance || t.keys[j] == t.keys[i] {
                    return false;
                }
                probe_distance += 1;
                j = (j + 1) & REPETITION_TABLE_MASK;
            }
            if distance != probe_distance {
                return false;
            }
        }
        i += 1;
    }
    count == t.count && max_distance == t.max_distance
}

// Check if the repetition table is valid.
const _: () = {
    let t = generate_repetition_table();
    assert!(
        t.count == 3668,
        "repetition table insertion count must be stable"
    );
    assert!(
        t.max_distance <= 255,
        "repetition table displacement must fit in one byte"
    );
    assert!(
        repetition_table_is_valid(&t),
        "repetition table lookup invariants must hold"
    );
};

// ---------------------------------------------------------------------------
// StateInfo – mirrors C++ layout/semantics
// ---------------------------------------------------------------------------

/// Byte offset of `key`: the number of leading StateInfo bytes copied by do_move
/// (matches C++ offsetof(StateInfo, key)).
pub const STATEINFO_COPY_SIZE: usize = core::mem::offset_of!(StateInfo, key);

/// Error returned whenever a position is suspected to be unsupported by the
/// engine (in particular positions that may cause hard errors like a
/// segmentation fault). Mirrors C++ `struct PositionSetError : std::runtime_error`
/// (src/position.h), which simply carries an explanatory message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionSetError {
    message: String,
}

impl PositionSetError {
    /// Creates an error with the given message.
    #[inline]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The human-readable description of the problem.
    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The message as the bytes C++ would print: a non-ASCII byte echoed from the FEN (see
    /// `echo_byte`) is emitted raw, not as UTF-8.
    pub fn message_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.message.len());
        for c in self.message.chars() {
            match u32::from(c) {
                v @ 0xF780..=0xF7FF => out.push((v - 0xF700) as u8),
                _ => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
            }
        }
        out
    }
}

/// A byte of the FEN as it appears in an error message. C++ echoes the raw byte; a Rust `String`
/// cannot hold one, so bytes >= 0x80 are carried as U+F780..U+F7FF and turned back by
/// [`PositionSetError::message_bytes`].
fn echo_byte(b: u8) -> char {
    if b < 0x80 {
        char::from(b)
    } else {
        char::from_u32(0xF700 + u32::from(b)).unwrap_or('?')
    }
}

impl fmt::Display for PositionSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PositionSetError {}

/// StateInfo struct stores information needed to restore a Position object to
/// its previous state when we retract a move. The states live in a ring buffer
/// owned by the Position (`ring`); `do_move` advances into the next slot.
#[derive(Clone, Debug)]
#[repr(C)] // Ensure predictable layout for partial copy optimization
pub struct StateInfo {
    // Copied when making a move
    pub material_key: Key,
    pub pawn_key: Key,
    pub minor_piece_key: Key,
    pub non_pawn_key: [Key; COLOR_NB],
    pub non_pawn_material: [Value; COLOR_NB],
    pub castling_rights: CastlingRights,
    pub rule50: i32,
    pub rep_window: i32,
    pub ep_square: Option<Square>,

    // Not copied when making a move (recomputed)
    pub key: Key,
    pub checkers_bb: Bitboard,
    pub blockers_for_king: [Bitboard; COLOR_NB],
    pub pinners: [Bitboard; COLOR_NB],
    pub check_squares: [Bitboard; PIECE_TYPE_NB],
    /// Squares attacked by the side not to move, per piece type; filled lazily by
    /// `Position::threats_by` through a shared reference, hence the `Cell`.
    pub threats: Cell<[Bitboard; PIECE_TYPE_NB]>,
    pub captured_piece: Piece,
    pub repetition: i32,
    /// The board packed one nibble per square (see `pack_pieces`): with the key, it identifies
    /// the position exactly for repetition detection.
    pub piece_placement: [u8; 32],
}

impl Default for StateInfo {
    fn default() -> Self {
        StateInfo {
            material_key: 0,
            pawn_key: 0,
            minor_piece_key: 0,
            non_pawn_key: [0; COLOR_NB],
            non_pawn_material: [VALUE_ZERO; COLOR_NB],
            castling_rights: CastlingRights::NO_CASTLING,
            rule50: 0,
            rep_window: 0,
            ep_square: None,
            key: 0,
            checkers_bb: 0,
            blockers_for_king: [0; COLOR_NB],
            pinners: [0; COLOR_NB],
            check_squares: [0; PIECE_TYPE_NB],
            threats: Cell::new([0; PIECE_TYPE_NB]),
            captured_piece: Piece::NoPiece,
            repetition: 0,
            piece_placement: [0; 32],
        }
    }
}

// ---------------------------------------------------------------------------
// Position – board state and fast accessors
// ---------------------------------------------------------------------------

/// A chess position (C++ `Position`): the board (piece list and bitboards), the side to move,
/// castling data, and a ring buffer of [`StateInfo`]s holding the incremental keys, check
/// information and repetition history of the current and previous plies. Set it up with
/// [`Position::set_fen`]; make and unmake moves with [`Position::do_move`] /
/// [`Position::undo_move`] (legality unchecked) or [`Position::do_move_owned`] (checked).
///
/// A `Position` is `Send` but not `Sync`: [`Position::threats_by`] fills a cache through `&self`.
///
/// ```compile_fail
/// fn shared<T: Sync>() {}
/// shared::<stockcrab::position::Position>();
/// ```
pub struct Position {
    // Board representation
    board: [Piece; SQUARE_NB],
    by_type_bb: [Bitboard; PIECE_TYPE_NB], // index 0 (NoPieceType) stores ALL pieces
    by_color_bb: [Bitboard; COLOR_NB],
    piece_count: [i32; PIECE_NB], // includes slots 0 and 8 for per-color ALL
    castling_rights_mask: [i32; SQUARE_NB],
    castling_rook_square: [Option<Square>; CASTLING_RIGHT_NB],
    castling_path: [Bitboard; CASTLING_RIGHT_NB],
    game_ply: i32,
    side_to_move: Color,
    chess960: bool,
    // Scratch buffer used by the convenience `do_move()` overload that does not
    // care about NNUE bookkeeping. Mirrors C++ `Dirties scratchDirties` (a
    // single member). Boxed so that
    // `Position` itself stays small and cheap to move around.
    scratch_dirties: Box<Dirties>,
    state_idx: u16,
    /// Current state: always `ring + state_idx`.
    st: *mut StateInfo,
    /// The ring buffer of states (C++ `StateInfo stateInfos[MAX_HIST]`): slot 0 of a heap array
    /// of `MAX_HIST` states owned by this position (from `Box::into_raw`, freed in `Drop`). It is
    /// held as a raw pointer rather than a `Box`, because moving a `Box` would assert unique
    /// access and invalidate `st`; on the heap, `st` stays valid when the position is moved.
    ring: *mut StateInfo,
}

impl Drop for Position {
    fn drop(&mut self) {
        // SAFETY: `ring` came from `Box::into_raw` of a `Box<[StateInfo; MAX_HIST]>` in
        // `alloc_ring` and is freed only here.
        drop(unsafe { Box::from_raw(self.ring.cast::<[StateInfo; MAX_HIST]>()) });
    }
}

// SAFETY: the raw pointers, `st` and `ring`, point into the position's own heap ring, which
// moves with it and is never shared, so sending a Position sends everything it points to.
// It is not `Sync`: `threats_by` fills a `Cell` cache through `&self`.
unsafe impl Send for Position {}
// A panic cannot leave the threat cache half-written: it is replaced by one `Cell::set` of a
// fully computed array.
impl core::panic::RefUnwindSafe for Position {}

/// A new state ring holding `states` (C++ `set` memsets the whole Position, ring included), as
/// the raw slot-0 pointer that `Position::ring` owns.
fn alloc_ring(states: Vec<StateInfo>) -> *mut StateInfo {
    let ring: Box<[StateInfo; MAX_HIST]> = states
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("a state ring has MAX_HIST elements"));
    Box::into_raw(ring).cast::<StateInfo>()
}

#[inline(always)]
const fn hist_index(i: usize) -> usize {
    i & (MAX_HIST - 1)
}

/// Balanced moves for safe library operations that can unwind: the guard undoes the move
/// before the position is released. Search uses its own per-ply discipline, so this adds no
/// work to search nodes.
pub(crate) struct ScopedMove<'a> {
    position: &'a mut Position,
    mv: Move,
}

impl<'a> ScopedMove<'a> {
    /// # Safety
    /// The position must be valid and `mv` legal. It must not be reset or have
    /// unmatched moves left on it while borrowed through this guard.
    #[inline(always)]
    pub(crate) unsafe fn new(position: &'a mut Position, mv: Move) -> Self {
        // SAFETY: the caller guarantees legality; Drop undoes this move, including during
        // unwinding.
        unsafe { position.do_move(mv) };
        Self { position, mv }
    }
}

impl core::ops::Deref for ScopedMove<'_> {
    type Target = Position;
    #[inline(always)]
    fn deref(&self) -> &Position {
        self.position
    }
}

impl core::ops::DerefMut for ScopedMove<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Position {
        self.position
    }
}

impl Drop for ScopedMove<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: this guard made the most recent unmatched move and still
        // exclusively borrows the position.
        unsafe { self.position.undo_move(self.mv) };
    }
}

impl Clone for Position {
    /// Deep copy, history ring included (C++ `set` + `copy_state_infos`), so the clone keeps
    /// the same repetition history and is independent of the original.
    fn clone(&self) -> Self {
        // SAFETY: `ring` points to the `MAX_HIST` initialised states of this position's ring.
        let states = unsafe { core::slice::from_raw_parts(self.ring, MAX_HIST) };
        let ring = alloc_ring(states.to_vec());
        // SAFETY: `state_idx < MAX_HIST`; the pointer has the provenance of the whole ring.
        let st = unsafe { ring.add(usize::from(self.state_idx)) };
        Position {
            board: self.board,
            by_type_bb: self.by_type_bb,
            by_color_bb: self.by_color_bb,
            piece_count: self.piece_count,
            castling_rights_mask: self.castling_rights_mask,
            castling_rook_square: self.castling_rook_square,
            castling_path: self.castling_path,
            game_ply: self.game_ply,
            side_to_move: self.side_to_move,
            chess960: self.chess960,
            scratch_dirties: self.scratch_dirties.clone(),
            state_idx: self.state_idx,
            st,
            ring,
        }
    }
}

impl Default for Position {
    /// A valid starting position, with initialized attack and key tables.
    fn default() -> Self {
        crate::bitboard::init();
        let mut position = Self::empty_builder();
        position
            .set_fen_in_place(crate::uci::START_FEN, false)
            .expect("the built-in starting position is valid");
        position
    }
}

impl Position {
    // Private builder: never published by a safe constructor before FEN validation finishes.
    fn empty_builder() -> Self {
        // The pointer has the provenance of the whole ring.
        let ring = alloc_ring(vec![StateInfo::default(); MAX_HIST]);
        Position {
            board: [Piece::NoPiece; SQUARE_NB],
            by_type_bb: [0; PIECE_TYPE_NB],
            by_color_bb: [0; COLOR_NB],
            piece_count: [0; PIECE_NB],
            castling_rights_mask: [0; SQUARE_NB],
            castling_rook_square: [None; CASTLING_RIGHT_NB],
            castling_path: [0; CASTLING_RIGHT_NB],
            game_ply: 0,
            side_to_move: Color::White,
            chess960: false,
            scratch_dirties: Box::new(Dirties::default()),
            state_idx: 0,
            st: ring,
            ring,
        }
    }

    /// The current [`StateInfo`].
    ///
    /// Every method of `Position` reads the current state through this accessor, so the
    /// raw-pointer invariant is stated once: `self.ring` points to slot 0 of the position's own
    /// heap ring (with the provenance of all of it; it never moves) and `self.st` is always
    /// `ring + state_idx`. No reference to the whole ring is created while a state is borrowed
    /// mutably: every slot is reached through `ring`.
    #[inline(always)]
    pub(crate) fn st(&self) -> &StateInfo {
        // SAFETY: see the invariant above.
        unsafe { &*self.st }
    }

    /// The state `back` plies before the current one in the ring
    /// (C++ `stateInfos[(stateIdx - back) & (MAX_HIST - 1)]`).
    #[inline(always)]
    fn st_back(&self, back: usize) -> &StateInfo {
        let idx = hist_index(usize::from(self.state_idx).wrapping_sub(back));
        // SAFETY: `idx < MAX_HIST`, in bounds of the ring; for `back % MAX_HIST != 0` it is not
        // the current slot, which callers may hold mutably.
        unsafe { &*self.ring.add(idx) }
    }

    /// Advances `st` to the next ring slot (C++ `stateIdx = (stateIdx + 1) & (MAX_HIST - 1)`).
    /// Steps the pointer instead of recomputing `ring + idx * size_of::<StateInfo>()`.
    #[inline(always)]
    fn next_state(&mut self) {
        let idx = hist_index(usize::from(self.state_idx) + 1);
        self.state_idx = idx as u16;
        self.st = if idx == 0 {
            self.ring
        } else {
            // SAFETY: for idx > 0, `st` was slot idx - 1, so `st + 1` is slot idx; both in bounds.
            unsafe { self.st.add(1) }
        };
    }

    /// Steps `st` back to the previous ring slot.
    #[inline(always)]
    fn previous_state(&mut self) {
        let idx = usize::from(self.state_idx);
        self.state_idx = hist_index(idx.wrapping_sub(1)) as u16;
        // SAFETY: for idx > 0, `st - 1` is slot idx - 1; slot MAX_HIST - 1 is in bounds.
        self.st = unsafe {
            if idx == 0 {
                self.ring.add(MAX_HIST - 1)
            } else {
                self.st.sub(1)
            }
        };
    }

    /// Mutable access to the current [`StateInfo`]; same invariant as [`Position::st`].
    #[inline(always)]
    fn st_mut(&mut self) -> &mut StateInfo {
        // SAFETY: see the invariant on `st`; `&mut self` guarantees exclusive access.
        unsafe { &mut *self.st }
    }

    // -------------------------------------------------------------------
    // FEN I/O
    // -------------------------------------------------------------------

    /// Initializes the position object with the given FEN string.
    ///
    /// The FEN string is strictly validated; if it is invalid or inconsistent, a
    /// [`PositionSetError`] describing the problem is returned and the position
    /// and the caller's state are unchanged on error. Port of C++
    /// `Position::set` (position.cpp), including its error messages, so the two
    /// engines reject exactly the same inputs.
    ///
    /// A FEN string contains six fields separated by a space:
    ///
    /// 1. Piece placement (from white's perspective), rank 8 down to rank 1,
    ///    file A through H within each rank. Pieces are `PNBRQK` for white and
    ///    `pnbrqk` for black, runs of empty squares are digits 1-8, and `/`
    ///    separates ranks.
    /// 2. Active color: `w` or `b`.
    /// 3. Castling availability: `-`, or some of `KQkq`. Shredder-FEN and X-FEN
    ///    (rook file letters) are also accepted.
    /// 4. En passant target square, or `-`. Following the X-FEN standard this is
    ///    recorded only if a pawn is actually in position to capture.
    /// 5. Halfmove clock (plies since the last pawn move or capture).
    /// 6. Fullmove number, starting at 1.
    pub fn set_fen(&mut self, fen: &str, is_chess960: bool) -> Result<&mut Self, PositionSetError> {
        // Like C++ (which memsets the Position, ring included), the history is reset; the
        // position is only replaced once the FEN has been fully validated.
        let mut candidate = Self::empty_builder();
        candidate.set_fen_in_place(fen, is_chess960)?;
        *self = candidate;
        Ok(self)
    }

    fn set_fen_in_place(
        &mut self,
        fen: &str,
        is_chess960: bool,
    ) -> Result<&mut Self, PositionSetError> {
        // The private builder starts empty with `st` at ring slot 0. Only the public
        // transactional wrapper publishes this position.

        let bytes = fen.as_bytes();
        // Cursor into `bytes`. C++ reads through an istringstream with
        // `noskipws`, so every whitespace character is significant and is
        // consumed explicitly; the cursor mirrors that exactly.
        let mut i = 0usize;

        let mut num_pieces = 0i32;
        let mut file = File::A as i32;
        let mut rank = Rank::R8 as i32;

        // 1. Piece placement
        loop {
            if i >= bytes.len() {
                return Err(PositionSetError::new(
                    "Invalid FEN. Unexpected end of stream.",
                ));
            }
            let token = bytes[i];
            i += 1;

            if token.is_ascii_whitespace() {
                break;
            }

            if token.is_ascii_digit() {
                let diff = (token - b'0') as i32;
                if !(1..=8).contains(&diff) {
                    return Err(PositionSetError::new(
                        "Invalid FEN. Invalid number of squares to skip.",
                    ));
                }

                file += diff;
                if file > FILE_NB as i32 {
                    return Err(PositionSetError::new("Invalid FEN. Invalid file reached."));
                }
            } else if token == b'/' {
                if file != FILE_NB as i32 {
                    return Err(PositionSetError::new(
                        "Invalid FEN. Trying to end rank when not at the end of it.",
                    ));
                }

                rank -= 1;
                file = File::A as i32;

                if rank < Rank::R1 as i32 {
                    return Err(PositionSetError::new("Invalid FEN. Invalid rank reached."));
                }
            } else {
                if file >= FILE_NB as i32 {
                    return Err(PositionSetError::new("Invalid FEN. Invalid file reached."));
                }

                let pc = match char_to_piece(token as char) {
                    Some(pc) => pc,
                    None => {
                        return Err(PositionSetError::new(format!(
                            "Invalid FEN. Invalid piece: {}",
                            echo_byte(token)
                        )));
                    }
                };

                num_pieces += 1;
                if num_pieces > 32 {
                    return Err(PositionSetError::new(
                        "Invalid FEN. More than 32 pieces on the board.",
                    ));
                }

                self.put_piece(
                    pc,
                    make_square(File::from(file as u8), Rank::from(rank as u8)),
                );
                file += 1;
            }
        }

        if rank != Rank::R1 as i32 || file != FILE_NB as i32 {
            return Err(PositionSetError::new(
                "Invalid FEN. Board state encoding ended but cursor not at end.",
            ));
        }

        if self.pieces_of_types(&[PieceType::Pawn]) & (bitboard::RANK_1_BB | bitboard::RANK_8_BB)
            != 0
        {
            return Err(PositionSetError::new(
                "Unsupported position. Pawns on the first or eighth rank.",
            ));
        }

        if self.count_of(Color::White, PieceType::King) != 1
            || self.count_of(Color::Black, PieceType::King) != 1
        {
            return Err(PositionSetError::new(
                "Unsupported position. Incorrect number of kings.",
            ));
        }

        for c in [Color::White, Color::Black] {
            let name = if c == Color::White { "WHITE" } else { "BLACK" };

            if self.count_of(c, PieceType::Pawn) > 8 {
                return Err(PositionSetError::new(format!(
                    "Unsupported position. {} has more than 8 pawns.",
                    name
                )));
            }

            // Every piece beyond the initial complement must have come from a
            // promotion, which costs a pawn.
            let additional = (self.count_of(c, PieceType::Knight) - 2).max(0)
                + (self.count_of(c, PieceType::Bishop) - 2).max(0)
                + (self.count_of(c, PieceType::Rook) - 2).max(0)
                + (self.count_of(c, PieceType::Queen) - 1).max(0);
            if additional > 8 - self.count_of(c, PieceType::Pawn) {
                return Err(PositionSetError::new(format!(
                    "Unsupported position. Too many pieces for {}.",
                    name
                )));
            }
        }

        // 2. Active color
        if i >= bytes.len() {
            return Err(PositionSetError::new(
                "Invalid FEN. Unexpected end of stream.",
            ));
        }
        let token = bytes[i];
        i += 1;
        if token != b'w' && token != b'b' {
            return Err(PositionSetError::new(format!(
                "Invalid FEN. Invalid side to move: {}",
                echo_byte(token)
            )));
        }
        self.side_to_move = if token == b'w' {
            Color::White
        } else {
            Color::Black
        };

        // C++ is `if (!(ss >> token) || !isspace(token) || ss.eof())`. The
        // `ss.eof()` arm never fires: successfully extracting the final
        // character does not set eofbit, so the check reduces to "a whitespace
        // character must follow". Note this means a FEN ending immediately after
        // the side to move is rejected -- `UciEngine::position` appends a
        // trailing space to every token, exactly as C++ does, so a bare
        // `position fen <board> w` still reaches here with that space.
        if i >= bytes.len() || !bytes[i].is_ascii_whitespace() {
            return Err(PositionSetError::new(
                "Invalid FEN. Expected whitespace after side to move.",
            ));
        }
        i += 1;

        // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
        // Shredder-FEN that uses the letters of the columns on which the rooks began
        // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
        // if an inner rook is associated with the castling right, the castling tag is
        // replaced by the file letter of the involved rook, as for the Shredder-FEN.
        //
        // NOTE: Due to the prevalence of incorrect (or missing) castling rights the
        // validation is less strict. However, incorrect castling rights are still sanitized.
        let mut num_castling_rights = 0;
        loop {
            if i >= bytes.len() {
                break;
            }
            let token = bytes[i];
            i += 1;

            if token.is_ascii_whitespace() {
                break;
            }

            if num_castling_rights == 0 && token == b'-' {
                // C++ `ss >> std::ws` before breaking.
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                break;
            }

            num_castling_rights += 1;
            if num_castling_rights > 4 {
                return Err(PositionSetError::new(
                    "Invalid FEN. Maximum of 4 castling rights can be specified.",
                ));
            }

            let upper = token.to_ascii_uppercase();
            if upper != b'K' && upper != b'Q' && !(b'A'..=b'H').contains(&upper) {
                return Err(PositionSetError::new(format!(
                    "Invalid FEN. Expected castling rights. Got: {}",
                    echo_byte(token)
                )));
            }

            let c = if token.is_ascii_lowercase() {
                Color::Black
            } else {
                Color::White
            };
            let ksq = self.square_of(c, PieceType::King);
            if relative_rank_sq(c, ksq) != Rank::R1
                || file_of(ksq) == File::A
                || file_of(ksq) == File::H
            {
                continue;
            }

            let mut rooks = self.pieces_color_types(c, &[PieceType::Rook]);
            if upper == b'K' {
                rooks &= bitboard::between_bb_pub(ksq, relative_square(c, Square::SQ_H1));
            } else if upper == b'Q' {
                rooks &= bitboard::between_bb_pub(ksq, relative_square(c, Square::SQ_A1));
            } else {
                let rsq = make_square(File::from(upper - b'A'), relative_rank(c, Rank::R1));
                rooks &= bitboard::square_bb_pub(rsq);
            }

            if rooks != 0 {
                let rsq = if upper == b'K' {
                    bitboard::msb(rooks)
                } else {
                    bitboard::lsb(rooks)
                };
                self.set_castling_right(c, rsq);
            }
        }

        // 4. En passant square. Ignored unless the square is on the side to
        // move's relative rank 6 and the capture is actually available.
        let mut legal_ep = false;
        // C++ leaves `col` at '-' when the extraction fails at end of stream.
        let col = if i < bytes.len() {
            let c = bytes[i];
            i += 1;
            c
        } else {
            b'-'
        };

        if col != b'-' {
            if i >= bytes.len() {
                return Err(PositionSetError::new(
                    "Invalid FEN. Unexpected end of stream.",
                ));
            }
            let row = bytes[i];
            i += 1;

            let expected_row = if self.side_to_move == Color::White {
                b'6'
            } else {
                b'3'
            };
            if (b'a'..=b'h').contains(&col) && row == expected_row {
                let ep_sq = make_square(File::from(col - b'a'), Rank::from(row - b'1'));
                {
                    self.st_mut().ep_square = Some(ep_sq);
                }

                let us = self.side_to_move;
                let them = !us;
                let ep_bb = bitboard::square_bb_pub(ep_sq);

                let mut pawns = bitboard::pawn_attacks(them, ep_bb)
                    & self.pieces_color_types(us, &[PieceType::Pawn]);
                let target = self.pieces_color_types(them, &[PieceType::Pawn])
                    & bitboard::square_bb_pub(ep_sq + pawn_push(them));
                // Occupancy as it would be *after* the en passant capture: the
                // captured pawn is gone and the ep square is occupied.
                let occ = self.pieces() ^ target ^ ep_bb;

                // The en passant square is recorded only if
                //   a) the side to move has a pawn attacking it,
                //   b) there is an enemy pawn in front of it, and
                //   c) neither it nor the square behind it is occupied.
                let enpassant = pawns != 0
                    && target != 0
                    && (self.pieces() & (ep_bb | bitboard::square_bb_pub(ep_sq + pawn_push(us))))
                        == 0;

                // ... and only if at least one of those pawns can actually make
                // the capture without leaving its own king in check.
                let ksq = self.square_of(us, PieceType::King);
                while enpassant && pawns != 0 && !legal_ep {
                    let from = bitboard::pop_lsb(&mut pawns);
                    legal_ep = (self.attackers_to_occ(ksq, occ ^ bitboard::square_bb_pub(from))
                        & self.pieces_of_color(them)
                        & !target)
                        == 0;
                }
            } else {
                return Err(PositionSetError::new(
                    "Invalid FEN. Invalid en-passant square.",
                ));
            }
        }

        if !legal_ep {
            {
                self.st_mut().ep_square = None;
            }
        }

        // 5-6. Halfmove clock and fullmove number. C++ switches the stream back
        // to `skipws` and extracts two ints, leaving them at 0 if extraction
        // fails, so missing fields are not an error.
        // Stream semantics (`ss >> rule50 >> gamePly`): a value that overflows `int` saturates and
        // fails the stream, which then leaves `gamePly` at 0; a non-number reads as 0 and does the same.
        let mut cursor = i.min(bytes.len());
        let r50 = crate::misc::stream_extract_int(
            bytes,
            &mut cursor,
            i32::MIN as i128,
            i32::MAX as i128,
            false,
        );
        let rule50 = r50.unwrap_or_else(|v| v) as i32;
        let mut game_ply = if r50.is_ok() {
            crate::misc::stream_extract_int(
                bytes,
                &mut cursor,
                i32::MIN as i128,
                i32::MAX as i128,
                false,
            )
            .unwrap_or_else(|v| v) as i32
        } else {
            0
        };

        {
            self.st_mut().rule50 = rule50;
        }

        // Values larger than 99 are normally pointless, but ignoring the 50 move
        // rule is supported for tablebase purposes. The limit is 2^15 because
        // rule50 is used multiplicatively with the evaluation during search.
        if !(0..=32767).contains(&rule50) {
            return Err(PositionSetError::new(
                "Unsupported position. Rule50 counter out of range.",
            ));
        }

        if !(0..=100000).contains(&game_ply) {
            return Err(PositionSetError::new(
                "Unsupported position. Game ply out of range.",
            ));
        }

        // Convert from fullmove starting from 1 to gamePly starting from 0,
        // handle also common incorrect FEN with fullmove = 0.
        game_ply = (2 * (game_ply - 1)).max(0)
            + if self.side_to_move == Color::Black {
                1
            } else {
                0
            };
        self.game_ply = game_ply;

        self.chess960 = is_chess960;
        self.set_state();

        // A position where the side to move could capture the enemy king is not
        // a legal chess position and would break the search.
        if self.attackers_to_exist(
            self.square_of(!self.side_to_move, PieceType::King),
            self.pieces(),
            self.side_to_move,
        ) {
            return Err(PositionSetError::new(
                "Unsupported position. King can be captured.",
            ));
        }

        debug_assert!(self.pos_is_ok(), "Invalid position from FEN: {}", fen);
        Ok(self)
    }

    /// Sets up a position from a material code such as "KQvK" (strong side first,
    /// ASCII only), as used for tablebase/endgame probing. Panics if the code is
    /// malformed. Mirrors C++ `set(code, c)`.
    pub fn set_code(&mut self, code: &str, c: Color) -> Result<&mut Self, PositionSetError> {
        assert!(code.starts_with('K'));
        let find_next_k = code[1..].find('K').map(|i| i + 1).unwrap_or(code.len());
        let vpos = code.find('v').unwrap_or(code.len());
        let weak = &code[find_next_k..];
        let strong = &code[..core::cmp::min(vpos, find_next_k)];
        assert!(!weak.is_empty() && weak.len() < 8);
        assert!(!strong.is_empty() && strong.len() < 8);
        // C++ lowercases only sides[c]: for c == White that is the weak side (index 0), for
        // c == Black the strong side. Lowercasing both would give two black kings and a zero
        // material key, breaking Syzygy lookups.
        let sides = [weak.to_string(), strong.to_string()];
        let (weak_s, strong_s) = if c == Color::White {
            (sides[0].to_ascii_lowercase(), sides[1].clone())
        } else {
            (sides[0].clone(), sides[1].to_ascii_lowercase())
        };

        let mut fen = String::new();
        fen.push_str("8/");
        fen.push_str(&weak_s);
        fen.push(char::from_u32('0' as u32 + (8 - weak_s.len()) as u32).unwrap());
        fen.push_str("/8/8/8/8/");
        fen.push_str(&strong_s);
        fen.push(char::from_u32('0' as u32 + (8 - strong_s.len()) as u32).unwrap());
        fen.push_str("/8 w - - 0 10");
        self.set_fen(&fen, false)
    }

    /// Returns a FEN representation of the position. In case of
    /// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.
    pub fn fen(&self) -> String {
        use crate::uci::UciEngine;
        let mut ss = String::new();
        for r in (0..RANK_NB).rev() {
            let mut f = 0;
            while f < FILE_NB {
                let mut empty_cnt = 0;
                while f < FILE_NB
                    && self.empty(make_square(File::from(f as u8), Rank::from(r as u8)))
                {
                    empty_cnt += 1;
                    f += 1;
                }
                if empty_cnt > 0 {
                    ss.push(char::from_u32('0' as u32 + empty_cnt as u32).unwrap());
                }
                if f < FILE_NB {
                    let sq = make_square(File::from(f as u8), Rank::from(r as u8));
                    ss.push(piece_to_char(self.piece_on(sq)));
                    f += 1;
                }
            }
            if r > 0 {
                ss.push('/');
            }
        }
        ss.push_str(if self.side_to_move == Color::White {
            " w "
        } else {
            " b "
        });

        let mut any_castle = false;
        let w_oo = CastlingRights::WHITE_OO;
        let w_ooo = CastlingRights::WHITE_OOO;
        let b_oo = CastlingRights::BLACK_OO;
        let b_ooo = CastlingRights::BLACK_OOO;
        if self.can_castle(w_oo) {
            any_castle = true;
            if self.chess960 {
                ss.push((b'A' + file_of(self.castling_rook_square(w_oo)) as u8) as char);
            } else {
                ss.push('K');
            }
        }
        if self.can_castle(w_ooo) {
            any_castle = true;
            if self.chess960 {
                ss.push((b'A' + file_of(self.castling_rook_square(w_ooo)) as u8) as char);
            } else {
                ss.push('Q');
            }
        }
        if self.can_castle(b_oo) {
            any_castle = true;
            if self.chess960 {
                ss.push((b'a' + file_of(self.castling_rook_square(b_oo)) as u8) as char);
            } else {
                ss.push('k');
            }
        }
        if self.can_castle(b_ooo) {
            any_castle = true;
            if self.chess960 {
                ss.push((b'a' + file_of(self.castling_rook_square(b_ooo)) as u8) as char);
            } else {
                ss.push('q');
            }
        }
        if !any_castle {
            ss.push('-');
        }

        if let Some(ep) = self.ep_square() {
            ss.push(' ');
            ss.push_str(&UciEngine::square(ep));
            ss.push(' ');
        } else {
            ss.push_str(" - ");
        }
        ss.push_str(&format!(
            "{} {}",
            self.rule50_count(),
            1 + (self.game_ply - (self.side_to_move == Color::Black) as i32) / 2
        ));
        ss
    }

    // -------------------------------------------------------------------
    // Board access – mirrors inline helpers in C++
    // -------------------------------------------------------------------
    /// The side to move.
    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// The piece on `s`, or `Piece::NoPiece`.
    #[inline(always)]
    pub fn piece_on(&self, s: Square) -> Piece {
        debug_assert!(is_ok(s));
        // Square is always in range 0-63 (SQUARE_NB)
        self.board[s as usize]
    }

    /// Returns a reference to the internal piece array (for NNUE cache comparison)
    #[inline(always)]
    pub fn piece_array(&self) -> &[Piece; SQUARE_NB] {
        &self.board
    }

    /// Whether `s` is empty.
    #[inline(always)]
    pub fn empty(&self, s: Square) -> bool {
        self.piece_on(s) == Piece::NoPiece
    }

    /// All occupied squares.
    #[inline(always)]
    pub fn pieces(&self) -> Bitboard {
        // ALL_PIECES is always a valid index into by_type_bb
        self.by_type_bb[ALL_PIECES as usize]
    }

    /// Total number of pieces, using the maintained per-color counters like
    /// Stockfish's `Position::count<ALL_PIECES>()`.
    #[inline(always)]
    pub fn count_all_pieces(&self) -> usize {
        (self.count_of(Color::White, PieceType::NoPieceType)
            + self.count_of(Color::Black, PieceType::NoPieceType)) as usize
    }

    /// Unchecked access to piece_count.
    ///
    /// # Safety
    ///
    /// `pc` must be a valid piece index (0-15).
    #[inline(always)]
    pub unsafe fn piece_count_unchecked(&self, pc: Piece) -> i32 {
        debug_assert!((pc as usize) < PIECE_NB);
        self.piece_count[pc as usize]
    }

    /// Unchecked access to castling_rights_mask.
    ///
    /// # Safety
    ///
    /// `sq` must be a valid square (0-63).
    #[inline(always)]
    pub unsafe fn castling_rights_mask_unchecked(&self, sq: Square) -> i32 {
        debug_assert!((sq as usize) < SQUARE_NB);
        self.castling_rights_mask[sq as usize]
    }

    /// The squares occupied by `c`'s pieces.
    #[inline(always)]
    pub fn pieces_of_color(&self, c: Color) -> Bitboard {
        // Color is always 0 or 1 (COLOR_NB = 2)
        self.by_color_bb[c as usize]
    }

    /// The squares occupied by pieces of any of `types`, either colour.
    #[inline(always)]
    pub fn pieces_of_types(&self, types: &[PieceType]) -> Bitboard {
        let mut bb = 0u64;
        for &pt in types {
            // PieceType is always in range 0-7 (PIECE_TYPE_NB = 8)
            bb |= self.by_type_bb[pt as usize];
        }
        bb
    }

    /// Const generic version for single piece type - compile-time optimized
    #[inline(always)]
    pub fn pieces_pt<const PT: u8>(&self) -> Bitboard {
        // PT is a const generic; the index is bounds-checked, and the check folds away for
        // in-range constants.
        self.by_type_bb[PT as usize]
    }

    /// Const generic version for two piece types - compile-time optimized
    #[inline(always)]
    pub fn pieces_pt2<const PT1: u8, const PT2: u8>(&self) -> Bitboard {
        const { assert!((PT1 as usize) < PIECE_TYPE_NB && (PT2 as usize) < PIECE_TYPE_NB) };
        // SAFETY: the const assertion above guarantees PT1 and PT2 are below PIECE_TYPE_NB.
        unsafe {
            *self.by_type_bb.get_unchecked(PT1 as usize)
                | *self.by_type_bb.get_unchecked(PT2 as usize)
        }
    }

    /// Const generic version for piece type with color - compile-time optimized
    #[inline(always)]
    pub fn pieces_c_pt<const PT: u8>(&self, c: Color) -> Bitboard {
        // PT is a const generic; the index is bounds-checked, and the check folds away for
        // in-range constants.
        self.pieces_of_color(c) & self.by_type_bb[PT as usize]
    }

    /// Const generic version for two piece types with color - compile-time optimized
    #[inline(always)]
    pub fn pieces_c_pt2<const PT1: u8, const PT2: u8>(&self, c: Color) -> Bitboard {
        const { assert!((PT1 as usize) < PIECE_TYPE_NB && (PT2 as usize) < PIECE_TYPE_NB) };
        // SAFETY: the const assertion above guarantees PT1 and PT2 are below PIECE_TYPE_NB.
        unsafe {
            self.pieces_of_color(c)
                & (*self.by_type_bb.get_unchecked(PT1 as usize)
                    | *self.by_type_bb.get_unchecked(PT2 as usize))
        }
    }

    /// The squares occupied by `c`'s pieces of any of `types`.
    #[inline(always)]
    pub fn pieces_color_types(&self, c: Color, types: &[PieceType]) -> Bitboard {
        self.pieces_of_color(c) & self.pieces_of_types(types)
    }

    /// The en passant target square, recorded only when a pawn can actually capture there.
    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        self.st().ep_square
    }

    /// The number of `c`'s pieces of type `pt`.
    #[inline(always)]
    pub fn count_of(&self, c: Color, pt: PieceType) -> i32 {
        // make_piece returns a value in range 0-15 (PIECE_NB = 16)
        self.piece_count[make_piece(c, pt) as usize]
    }

    /// Const generic version for count - compile-time optimized
    #[inline(always)]
    pub fn count_pt<const PT: u8>(&self, c: Color) -> i32 {
        // SAFETY: make_piece returns a value in range 0-15 (PIECE_NB = 16)
        unsafe {
            *self
                .piece_count
                .get_unchecked(make_piece(c, PieceType::from(PT)) as usize)
        }
    }

    /// Const generic version for total count - compile-time optimized
    #[inline(always)]
    pub fn count_all_pt<const PT: u8>(&self) -> i32 {
        self.count_pt::<PT>(Color::White) + self.count_pt::<PT>(Color::Black)
    }

    /// The number of pieces of type `pt`, both colours.
    #[inline(always)]
    pub fn count(&self, pt: PieceType) -> i32 {
        self.count_of(Color::White, pt) + self.count_of(Color::Black, pt)
    }

    /// The square of `c`'s only piece of type `pt` (normally the king).
    #[inline(always)]
    pub fn square_of(&self, c: Color, pt: PieceType) -> Square {
        if pt != PieceType::King {
            assert_eq!(
                self.count_of(c, pt),
                1,
                "square_of requires exactly one piece"
            );
        }
        debug_assert_eq!(self.count_of(c, pt), 1);
        bitboard::lsb(self.pieces_color_types(c, &[pt]))
    }

    /// Const generic version for square - compile-time optimized
    #[inline(always)]
    pub fn square_pt<const PT: u8>(&self, c: Color) -> Square {
        if PT != PieceType::King as u8 {
            assert_eq!(
                self.count_pt::<PT>(c),
                1,
                "square_pt requires exactly one piece"
            );
        }
        debug_assert_eq!(self.count_pt::<PT>(c), 1);
        bitboard::lsb(self.pieces_c_pt::<PT>(c))
    }

    // -------------------------------------------------------------------
    // Castling
    // -------------------------------------------------------------------
    /// `c`'s remaining castling rights.
    #[inline(always)]
    pub fn castling_rights(&self, c: Color) -> CastlingRights {
        c & self.st().castling_rights
    }

    /// Whether any of the rights in `cr` remain.
    #[inline(always)]
    pub fn can_castle(&self, cr: CastlingRights) -> bool {
        !(self.st().castling_rights & cr).is_empty()
    }

    /// Whether a piece stands between the king and rook of castling right `cr`.
    #[inline(always)]
    pub fn castling_impeded(&self, cr: CastlingRights) -> bool {
        (self.pieces() & self.castling_path[cr.bits() as usize]) != 0
    }

    /// The initial square of the rook of castling right `cr`.
    #[inline(always)]
    pub fn castling_rook_square(&self, cr: CastlingRights) -> Square {
        self.castling_rook_square[cr.bits() as usize].expect("castling right without a rook square")
    }

    // -------------------------------------------------------------------
    // Checking
    // -------------------------------------------------------------------
    /// The pieces giving check to the side to move.
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.st().checkers_bb
    }

    /// Pieces (of either colour) that alone shield `c`'s king from a slider.
    #[inline(always)]
    pub fn blockers_for_king(&self, c: Color) -> Bitboard {
        // st is always valid, Color is always 0-1 (COLOR_NB = 2)
        self.st().blockers_for_king[c as usize]
    }

    /// Sliders of the other colour that pin a piece against `c`'s king.
    #[inline(always)]
    pub fn pinners(&self, c: Color) -> Bitboard {
        // st is always valid, Color is always 0-1 (COLOR_NB = 2)
        self.st().pinners[c as usize]
    }

    /// The squares from which a piece of type `pt` of the side to move would give check.
    #[inline(always)]
    pub fn check_squares(&self, pt: PieceType) -> Bitboard {
        // st is always valid, PieceType is always 0-7 (PIECE_TYPE_NB = 8)
        self.st().check_squares[pt as usize]
    }

    // -------------------------------------------------------------------
    // Attacks
    // -------------------------------------------------------------------
    #[inline(always)]
    // Attacks to/from a given square
    /// The pieces of both colours attacking `s`, with sliders blocked by `occupied`.
    pub fn attackers_to_occ(&self, s: Square, occupied: Bitboard) -> Bitboard {
        // C++: return (attacks_bb<ROOK>(s, occupied) & pieces(ROOK, QUEEN))
        //           | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP, QUEEN))
        //           | (attacks_bb<PAWN>(s, BLACK) & pieces(WHITE, PAWN))
        //           | (attacks_bb<PAWN>(s, WHITE) & pieces(BLACK, PAWN))
        //           | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT))
        //           | (attacks_bb<KING>(s) & pieces(KING));
        let white_pawns = self.pieces_c_pt::<{ PieceType::Pawn as u8 }>(Color::White);
        let black_pawns = self.pieces_c_pt::<{ PieceType::Pawn as u8 }>(Color::Black);
        let rooks_queens =
            self.pieces_pt2::<{ PieceType::Rook as u8 }, { PieceType::Queen as u8 }>();
        let bishops_queens =
            self.pieces_pt2::<{ PieceType::Bishop as u8 }, { PieceType::Queen as u8 }>();
        let knights = self.pieces_pt::<{ PieceType::Knight as u8 }>();
        let kings = self.pieces_pt::<{ PieceType::King as u8 }>();

        // Fetch both sliding attack sets from the same dual entry.
        let (b_attacks, r_attacks) = bitboard::both_attacks_bb(s, occupied);
        (r_attacks & rooks_queens)
            | (b_attacks & bishops_queens)
            | (bitboard::pawn_attacks_from_square(s, Color::Black) & white_pawns)
            | (bitboard::pawn_attacks_from_square(s, Color::White) & black_pawns)
            | (bitboard::attacks_bb_knight(s) & knights)
            | (bitboard::attacks_bb_king(s) & kings)
    }

    /// Computes a bitboard of all pieces which attack a given square.
    /// Slider attacks use the occupied bitboard to indicate occupancy.
    #[inline(always)]
    pub fn attackers_to(&self, s: Square) -> Bitboard {
        #[cfg(debug_assertions)]
        {
            // Debug-only: validate aggregated bitboards against the board array.
            let mut occ_from_board = 0u64;
            for idx in 0..SQUARE_NB {
                if self.board[idx] != Piece::NoPiece {
                    occ_from_board |= 1u64 << (idx as u8);
                }
            }
            debug_assert_eq!(
                occ_from_board,
                self.pieces(),
                "Position occupancy mismatch in attackers_to()"
            );
            return self.attackers_to_occ(s, occ_from_board);
        }
        #[cfg(not(debug_assertions))]
        {
            self.attackers_to_occ(s, self.pieces())
        }
    }

    /// Whether any of `c`'s pieces attacks `s`, with sliders blocked by `occupied`.
    #[inline(always)]
    pub fn attackers_to_exist(&self, s: Square, occupied: Bitboard, c: Color) -> bool {
        (self.attackers_to_occ(s, occupied) & self.pieces_of_color(c)) != 0
    }

    /// Calculates `st->blockersForKing[c]` and `st->pinners[~c]`,
    /// which store respectively the pieces preventing king of color c from being in check
    /// and the slider pieces of color ~c pinning pieces of color c to the king.
    #[inline(always)]
    fn update_slider_blockers_st(&self, st: &mut StateInfo, c: Color) {
        // `set_fen` guarantees exactly one king per colour (C++ only asserts this).
        let ksq = self.square_of(c, PieceType::King);
        // SAFETY: c is Color (0 or 1), arrays have 2 elements
        unsafe {
            *st.blockers_for_king.get_unchecked_mut(c as usize) = 0;
            *st.pinners.get_unchecked_mut((!c) as usize) = 0;
        }
        // Snipers are opposing sliders aligned with our king on empty board
        let rook_like = bitboard::attacks_empty(PieceType::Rook, ksq)
            & self.pieces_of_types(&[PieceType::Queen, PieceType::Rook]);
        let bishop_like = bitboard::attacks_empty(PieceType::Bishop, ksq)
            & self.pieces_of_types(&[PieceType::Queen, PieceType::Bishop]);
        // Snipers are sliders that attack 's' when a piece and other snipers are removed
        let snipers = (rook_like | bishop_like) & self.pieces_of_color(!c);
        let occupancy = self.pieces() ^ snipers;
        let mut s = snipers;
        while s != 0 {
            let sniper_sq = bitboard::pop_lsb(&mut s);
            let b = bitboard::between_bb_pub(ksq, sniper_sq) & occupancy;
            if b != 0 && !bitboard::more_than_one(b) {
                // SAFETY: c is Color (0 or 1), arrays have 2 elements
                unsafe {
                    *st.blockers_for_king.get_unchecked_mut(c as usize) |= b;
                }
                if (b & self.pieces_of_color(c)) != 0 {
                    // SAFETY: c is Color (0 or 1) and `pinners` has COLOR_NB = 2 elements, so `!c` is also in bounds.
                    unsafe {
                        *st.pinners.get_unchecked_mut((!c) as usize) |= 1u64 << (sniper_sq as u8);
                    }
                }
            }
        }
    }

    /// The squares attacked by `c`'s pieces of type `pt`, with sliders blocked by `occ`.
    #[inline(always)]
    pub fn xattacks_by(&self, pt: PieceType, c: Color, occ: Bitboard) -> Bitboard {
        if pt == PieceType::Pawn {
            bitboard::pawn_attacks(c, self.pieces_color_types(c, &[PieceType::Pawn]))
        } else {
            let mut threats = 0u64;
            let mut attackers = self.pieces_color_types(c, &[pt]);
            while attackers != 0 {
                let s = bitboard::pop_lsb(&mut attackers);
                threats |= bitboard::attacks_bb(pt, s, occ);
            }
            threats
        }
    }

    /// The squares attacked by `c`'s pieces of type `pt` on the current board.
    #[inline(always)]
    pub fn attacks_by(&self, c: Color, pt: PieceType) -> Bitboard {
        self.xattacks_by(pt, c, self.pieces())
    }

    /// The squares the opponent of the side to move attacks with pieces of type `pt`
    /// (`ALL_PIECES`: with any piece); computed on first use for each state and cached.
    #[inline(always)]
    pub fn threats_by(&self, pt: PieceType) -> Bitboard {
        if self.st().threats.get()[ALL_PIECES as usize] == 0 {
            self.compute_threats();
        }
        self.st().threats.get()[pt as usize]
    }

    // -------------------------------------------------------------------
    // Castling helper - template equivalent using const generic
    // Handles Chess960 square overlap correctly: removes both pieces first, then places them.
    // -------------------------------------------------------------------
    /// Helper used to do/undo a castling move. This is a bit tricky in Chess960
    /// where from/to squares can overlap.
    #[inline(always)]
    fn do_castling<const DO: bool>(
        &mut self,
        us: Color,
        from: Square,
        to: &mut Square,
        dts: Option<&mut DirtyThreats>,
        dp: Option<&mut DirtyPiece>,
    ) -> (Square, Square) {
        let king_side = *to as u8 > from as u8;
        let rfrom = *to; // Castling encoded as "king captures friendly rook"
        let rto = relative_square(
            us,
            if king_side {
                Square::SQ_F1
            } else {
                Square::SQ_D1
            },
        );
        *to = relative_square(
            us,
            if king_side {
                Square::SQ_G1
            } else {
                Square::SQ_C1
            },
        );

        if DO {
            if let Some(dp) = dp {
                dp.to = Some(*to);
                dp.remove_pc = make_piece(us, PieceType::Rook);
                dp.add_pc = make_piece(us, PieceType::Rook);
                dp.remove_sq = Some(rfrom);
                dp.add_sq = Some(rto);
            }
        }

        // Remove both pieces first since squares could overlap in Chess960
        let king_from = if DO { from } else { *to };
        let rook_from = if DO { rfrom } else { rto };

        // Add pieces to their final squares
        let king_to = if DO { *to } else { from };
        let rook_to = if DO { rto } else { rfrom };

        // remove_piece sets board[sq] to NoPiece, so no explicit clear is needed.
        match dts {
            Some(dts) => {
                self.remove_piece_dts(king_from, dts);
                self.remove_piece_dts(rook_from, dts);
                self.put_piece_dts(make_piece(us, PieceType::King), king_to, dts);
                self.put_piece_dts(make_piece(us, PieceType::Rook), rook_to, dts);
            }
            None => {
                self.remove_piece(king_from);
                self.remove_piece(rook_from);
                self.put_piece(make_piece(us, PieceType::King), king_to);
                self.put_piece(make_piece(us, PieceType::Rook), rook_to);
            }
        }

        (rfrom, rto)
    }

    // -------------------------------------------------------------------
    // Move properties
    // -------------------------------------------------------------------
    // Separate function call, matching C++.
    /// Takes a random move and tests whether the move is legal. It is used to validate moves from TT that can be corrupted
    /// due to SMP concurrent access or hash position key aliasing.
    #[inline(never)]
    pub fn legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.moved_piece(m);

        // Uncommon moves are checked against the complete legal move list.
        if m.move_type() != MoveType::Normal {
            return generate(self, GenType::All).contains(m);
        }

        // If the 'from' square is not occupied by a piece belonging to the side to
        // move, the move is obviously not legal.
        if pc == Piece::NoPiece || color_of(pc) != us {
            return false;
        }
        // Destination cannot be friendly occupied
        if (self.pieces_of_color(us) & (1u64 << (to as u8))) != 0 {
            return false;
        }
        if type_of(pc) == PieceType::Pawn {
            let rank1_8 = bitboard::RANK_1_BB | bitboard::RANK_8_BB;
            let is_promotion_rank = (rank1_8 & (1u64 << (to as u8))) != 0;

            // A Normal pawn move to rank 1 or 8 is never legal.
            if is_promotion_rank {
                return false;
            }

            // Check if it's a valid capture, single push, or double push
            let is_capture = (bitboard::pawn_attacks(us, 1u64 << (from as u8))
                & self.pieces_of_color(!us)
                & (1u64 << (to as u8)))
                != 0;
            let is_single = (from + pawn_push(us)) == to && self.empty(to);
            let is_double = (from + (2 * pawn_push(us))) == to
                && relative_rank_sq(us, from) == Rank::R2
                && self.empty(to)
                && self.empty(to - pawn_push(us));
            if !(is_capture || is_single || is_double) {
                return false;
            }
        } else {
            // attacks_bb takes occupancy for all piece types; it only affects sliders.
            if (bitboard::attacks_bb(type_of(pc), from, self.pieces()) & (1u64 << (to as u8))) == 0
            {
                return false;
            }
        }

        if type_of(pc) != PieceType::King {
            if self.checkers() != 0 {
                if bitboard::more_than_one(self.checkers()) {
                    return false;
                }
                if bitboard::between_bb_pub(
                    self.square_of(us, PieceType::King),
                    bitboard::lsb(self.checkers()),
                ) & bitboard::square_bb_pub(to)
                    == 0
                {
                    return false;
                }
            }
            self.blockers_for_king(us) & bitboard::square_bb_pub(from) == 0
                || bitboard::line_bb_pub(from, to)
                    & self.pieces_c_pt::<{ PieceType::King as u8 }>(us)
                    != 0
        } else {
            bitboard::square_bb_pub(to) & self.threats_by(ALL_PIECES) == 0
        }
    }

    /// Whether `m` captures (en passant included; castling never does).
    #[inline(always)]
    pub fn capture(&self, m: Move) -> bool {
        debug_assert!(m.is_ok());

        match m.move_type() {
            MoveType::Normal | MoveType::Promotion => !self.empty(m.to_sq()),
            MoveType::EnPassant => true,
            MoveType::Castling => false,
        }
    }

    /// Returns true if a move is generated from the capture stage, having also
    /// queen promotions covered, i.e. consistency with the capture stage move
    /// generation is needed to avoid the generation of duplicate moves.
    #[inline(always)]
    pub fn capture_stage(&self, m: Move) -> bool {
        debug_assert!(m.is_ok());

        match m.move_type() {
            MoveType::Normal => !self.empty(m.to_sq()),
            MoveType::Promotion => !self.empty(m.to_sq()) || m.promotion_type() == PieceType::Queen,
            MoveType::EnPassant => true,
            MoveType::Castling => false,
        }
    }

    // NOT inlined to match C++ (gives_check is a separate function call in C++)
    /// Tests whether a legal move gives a check
    #[inline(never)]
    pub fn gives_check(&self, m: Move) -> bool {
        debug_assert!(m.is_ok());
        debug_assert!(color_of(self.moved_piece(m)) == self.side_to_move);
        let from = m.from_sq();
        let to = m.to_sq();
        let them = !self.side_to_move;

        // Const generics for piece type lookups (compile-time optimized)
        const QUEEN: u8 = PieceType::Queen as u8;
        const ROOK: u8 = PieceType::Rook as u8;
        const BISHOP: u8 = PieceType::Bishop as u8;
        const KING: u8 = PieceType::King as u8;

        // Direct check
        if (self.check_squares(type_of(self.piece_on(from))) & (1u64 << (to as u8))) != 0 {
            return true;
        }
        // Discovered check
        if (self.blockers_for_king(them) & (1u64 << (from as u8))) != 0 {
            return (bitboard::line_bb_pub(from, to) & self.pieces_c_pt::<KING>(them)) == 0
                || m.move_type() == MoveType::Castling;
        }
        match m.move_type() {
            MoveType::Normal => false,
            MoveType::Promotion => {
                let occ = self.pieces() ^ (1u64 << (from as u8));
                (bitboard::attacks_bb(m.promotion_type(), to, occ) & self.pieces_c_pt::<KING>(them))
                    != 0
            }
            MoveType::EnPassant => {
                let capsq = make_square(file_of(to), rank_of(from));
                let b = (self.pieces() ^ (1u64 << (from as u8)) ^ (1u64 << (capsq as u8)))
                    | (1u64 << (to as u8));
                let ksq = bitboard::lsb(self.pieces_c_pt::<KING>(them));
                let (b_attacks, r_attacks) = bitboard::both_attacks_bb(ksq, b);
                (r_attacks & self.pieces_c_pt2::<QUEEN, ROOK>(self.side_to_move)) != 0
                    || (b_attacks & self.pieces_c_pt2::<QUEEN, BISHOP>(self.side_to_move)) != 0
            }
            MoveType::Castling => {
                let rto = relative_square(
                    self.side_to_move,
                    if to as u8 > from as u8 {
                        Square::SQ_F1
                    } else {
                        Square::SQ_D1
                    },
                );
                (self.check_squares(PieceType::Rook) & (1u64 << (rto as u8))) != 0
            }
        }
    }

    /// The piece `m` moves.
    #[inline(always)]
    pub fn moved_piece(&self, m: Move) -> Piece {
        self.piece_on(m.from_sq())
    }

    /// The piece captured by the last move, or `Piece::NoPiece`.
    #[inline(always)]
    pub fn captured_piece(&self) -> Piece {
        self.st().captured_piece
    }

    // -------------------------------------------------------------------
    // Make/undo
    // -------------------------------------------------------------------
    /// Makes a move, and saves all information necessary to the next StateInfo of the
    /// ring buffer. The move is assumed to be legal. Pseudo-legal moves should be filtered
    /// out before this function is called.
    ///
    /// # Safety
    /// `m` must be legal in this position (the move is not validated; use
    /// [`Position::do_move_owned`] for a checked move).
    ///
    /// ```compile_fail,E0133
    /// use stockcrab::position::Position;
    /// use stockcrab::types::{Move, Square};
    /// let mut pos = Position::default();
    /// pos.do_move(Move::from_to(Square::SQ_E2, Square::SQ_E4));
    /// ```
    #[inline(always)]
    pub unsafe fn do_move(&mut self, m: Move) {
        debug_assert!(self.pos_is_ok(), "Position invalid before do_move");

        let gives = self.gives_check(m);
        // C++ (position.h): re-initializes the scratch dirty threats/pawn pairs and
        // then forwards to the full do_move with `scratchDirties`.
        self.scratch_dirties.reset();
        // SAFETY: `scratch_dirties` is a pure output buffer; `do_move_internal`
        // never reads or writes the `scratch_dirties` field of `self`, so the
        // `&mut Dirties` handed out here cannot alias anything the call touches.
        let dirties: *mut Dirties = &mut *self.scratch_dirties;
        // SAFETY: `dirties` points to the boxed `scratch_dirties`, which `do_move_internal` never touches through `self`,
        // so the `&mut Dirties` does not alias any access made inside the call (see above); the Box outlives the call.
        unsafe {
            self.do_move_internal(m, gives, &mut *dirties);
        }

        debug_assert!(self.pos_is_ok(), "Position invalid after do_move");
    }

    /// Safe `do_move` for callers outside the search. Returns `false`, leaving the position
    /// unchanged, if `m` is not legal here.
    pub fn do_move_owned(&mut self, m: Move) -> bool {
        // `legal` decodes the squares, which requires `is_ok` (not `Move::none()`/`null()`).
        if !m.is_ok() || !self.legal(m) {
            return false;
        }
        // SAFETY: `m` was just checked to be legal.
        unsafe { self.do_move(m) };
        true
    }

    /// Internal implementation of do_move. Fills `dirties` with everything the
    /// NNUE side needs to update incrementally (moved piece, dirty threats and,
    /// since SFNNv16, the pawn bitboards before/after the move).
    /// Mirrors C++ `Position::do_move(Move, bool, Dirties&, const SharedHistories*)`
    /// (position.cpp) without the history argument: `Worker::do_move` prefetches the
    /// shared-history entries itself.
    ///
    /// # Safety
    /// The move requirement of [`Position::do_move`] applies.
    /// `gives_check` must equal `self.gives_check(m)`; `dirties` must not alias
    /// the position.
    #[inline(never)]
    pub unsafe fn do_move_internal(&mut self, m: Move, gives_check: bool, dirties: &mut Dirties) {
        debug_assert!(m.is_ok(), "Invalid move in do_move_internal");
        debug_assert!(
            color_of(self.moved_piece(m)) == self.side_to_move,
            "Wrong color piece moving"
        );
        debug_assert!(
            self.piece_on(self.square_of(self.side_to_move, PieceType::King))
                == make_piece(self.side_to_move, PieceType::King),
            "King not on expected square"
        );

        // Cache zobrist reference once for the entire function
        let z = zobrist();

        let prev_st = self.st;
        self.next_state();

        // Copy some fields of the old state to our new StateInfo object except the
        // ones which are going to be recalculated from scratch anyway
        // SAFETY: `prev_st` and the new `self.st` are distinct slots of the ring (MAX_HIST > 1),
        // and STATEINFO_COPY_SIZE = offset_of!(key) is within size_of::<StateInfo>(), so the
        // byte copy is in bounds and non-overlapping.
        unsafe {
            core::ptr::copy_nonoverlapping(
                prev_st as *const u8,
                self.st as *mut u8,
                STATEINFO_COPY_SIZE,
            );
        }

        // SAFETY: `prev_st` is a live slot of the ring (see `st()`).
        let mut k: Key = unsafe { (*prev_st).key } ^ ZOBRIST_SIDE;

        // Like C++'s `st`: a local, so the pointer is not re-read from `self` after every store
        // (the compiler cannot prove they do not alias).
        // SAFETY: `self.st` is the current slot of the ring; nothing else accesses it during this
        // call (the debug-only checks read it through `self.st()` but never while a write through
        // `st` is pending, and `st_back` reads only other slots).
        let st: &mut StateInfo = unsafe { &mut *self.st };

        // Increment ply counters. In particular, rule50 will be reset to zero later on in case of a
        // capture or a pawn move.
        self.game_ply += 1;
        st.rule50 += 1;
        st.rep_window = (st.rep_window + 1).min(MAX_HIST as i32 - 1);

        let dpps = &mut dirties.dirty_pawn_pairs;
        let dts = &mut dirties.dirty_threats;
        let dp = &mut dirties.dirty_piece;

        // SFNNv16: snapshot the pawn bitboards before the move; the matching
        // `after` snapshot is taken at the very end of this function.
        dpps.before[Color::White as usize] =
            self.pieces_c_pt::<{ PieceType::Pawn as u8 }>(Color::White);
        dpps.before[Color::Black as usize] =
            self.pieces_c_pt::<{ PieceType::Pawn as u8 }>(Color::Black);

        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        // For castling, do_castling() rewrites `to` from the rook square (the move
        // encoding) to the king's real destination, as C++'s by-reference `to`; every
        // later use (hash key, castling-rights mask) sees the updated value.
        let mut to = m.to_sq();
        let pc = self.piece_on(from);
        let mut captured = if m.move_type() == MoveType::EnPassant {
            make_piece(them, PieceType::Pawn)
        } else {
            self.piece_on(to)
        };

        // As in C++, `remove_pc`/`add_pc` are only written (and only read) together with
        // their square.
        dp.pc = pc;
        dp.from = Some(from);
        dp.to = Some(to);
        dp.add_sq = None;

        // Castling - matches C++ position.cpp
        if m.move_type() == MoveType::Castling {
            debug_assert!(pc == make_piece(us, PieceType::King));
            debug_assert!(captured == make_piece(us, PieceType::Rook));

            let (rfrom, rto) =
                self.do_castling::<true>(us, from, &mut to, Some(&mut *dts), Some(&mut *dp));

            // Update Zobrist and non-pawn key for rook move
            // SAFETY: `rfrom`/`rto` are board squares and `captured` is our rook (< PIECE_NB) for the zobrist lookups,
            // and us is Color (0 or 1) in bounds of the 2-element `non_pawn_key`.
            unsafe {
                let rook_keys = z.psq_unchecked(captured as usize, rfrom as usize)
                    ^ z.psq_unchecked(captured as usize, rto as usize);
                k ^= rook_keys;
                *st.non_pawn_key.get_unchecked_mut(us as usize) ^= rook_keys;
            }
            captured = Piece::NoPiece;
        } else if captured != Piece::NoPiece {
            // Capture (handle EP later)
            let mut capsq = to;

            // SAFETY: `capsq` is a board square and `captured` a real piece (< PIECE_NB) for the zobrist lookups and
            // `piece_value_unchecked`; them is Color (0 or 1) for the 2-element arrays.
            unsafe {
                // If the captured piece is a pawn, update pawn hash key, otherwise
                // update non-pawn material.
                if type_of(captured) == PieceType::Pawn {
                    if m.move_type() == MoveType::EnPassant {
                        capsq = to - pawn_push(us);
                        debug_assert!(pc == make_piece(us, PieceType::Pawn));
                        debug_assert!(Some(to) == st.ep_square);
                        debug_assert!(self.piece_on(to) == Piece::NoPiece);
                        debug_assert!(self.piece_on(capsq) == make_piece(them, PieceType::Pawn));
                        // Only the en passant victim is removed here; ordinary captures
                        // are folded into the `swap_piece` below, exactly as C++ does
                        // (position.cpp vs position.cpp).
                        self.remove_piece_dts(capsq, dts);
                    }
                    st.pawn_key ^= z.psq_unchecked(captured as usize, capsq as usize);
                } else {
                    *st.non_pawn_material.get_unchecked_mut(them as usize) -=
                        piece_value_unchecked(captured);
                    *st.non_pawn_key.get_unchecked_mut(them as usize) ^=
                        z.psq_unchecked(captured as usize, capsq as usize);
                    if type_of(captured) as u8 <= PieceType::Bishop as u8 {
                        st.minor_piece_key ^= z.psq_unchecked(captured as usize, capsq as usize);
                    }
                }

                dp.remove_pc = captured;
                dp.remove_sq = Some(capsq);

                k ^= z.psq_unchecked(captured as usize, capsq as usize);
                // Material key update for removal. C++ (position.cpp) uses
                // `8 + pieceCount[captured] - (type != EN_PASSANT)`: for an
                // en passant capture the victim has already been removed above,
                // for any other capture it is still on the board.
                st.material_key ^= z.psq_unchecked(
                    captured as usize,
                    8 + (self.piece_count_unchecked(captured) as usize)
                        - (m.move_type() != MoveType::EnPassant) as usize,
                );
            }

            // Reset rule 50 counter
            st.rule50 = 0;
            st.rep_window = 0;
        } else {
            dp.remove_sq = None;
        }

        // Update hash key
        // SAFETY: `from`/`to` are squares and `pc` the moved piece (< PIECE_NB) for the zobrist lookups.
        let move_key = unsafe {
            z.psq_unchecked(pc as usize, from as usize) ^ z.psq_unchecked(pc as usize, to as usize)
        };
        k ^= move_key;

        // Reset en passant square
        if let Some(ep) = st.ep_square {
            // SAFETY: `file_of` is a File (< FILE_NB).
            k ^= unsafe { z.enpassant_unchecked(file_of(ep) as usize) };
            st.ep_square = None;
        }

        // Update castling rights.
        // SAFETY: `from`/`to` are Squares (< 64), the length of `castling_rights_mask`; the rights are a 4-bit
        // set, in bounds of the 16-entry `castling` table.
        unsafe {
            k ^= z.castling_unchecked(st.castling_rights.bits() as usize);
            st.castling_rights &= CastlingRights::from_bits_retain(
                !((self.castling_rights_mask_unchecked(from)
                    | self.castling_rights_mask_unchecked(to)) as u8),
            );
            k ^= z.castling_unchecked(st.castling_rights.bits() as usize);
        }

        // If the moving piece is a pawn do some special extra work
        if type_of(pc) == PieceType::Pawn {
            // Check if the en passant square needs to be set. Accurate e.p. info is needed
            // for correct zobrist key generation and 3-fold checking.
            if (to as u8 ^ from as u8) == 16 {
                let ep_square = to - pawn_push(us);
                // If there are no pawns attacking the ep square, ep is not possible.
                let pawns = bitboard::pawn_attacks(us, 1u64 << (ep_square as u8))
                    & self.pieces_color_types(them, &[PieceType::Pawn]);

                if pawns != 0 {
                    let ksq = self.square_of(them, PieceType::King);
                    // SAFETY: `prev_st` is the previous slot of the ring, and them is
                    // Color (0 or 1) for the 2-element array.
                    let not_blockers =
                        unsafe { !*(*prev_st).blockers_for_king.get_unchecked(them as usize) };
                    let no_discovery = (bitboard::square_bb_pub(from) & not_blockers) != 0
                        || file_of(from) == file_of(ksq);

                    // If the pawn gives discovered check, ep is never legal. Else, if at least one
                    // pawn was not a blocker for the enemy king or lies on the same line as the
                    // enemy king and en passant square, a legal capture exists.
                    if no_discovery
                        && (pawns & (not_blockers | bitboard::line_bb_pub(ep_square, ksq))) != 0
                    {
                        st.ep_square = Some(ep_square);
                        // SAFETY: `file_of` is a File (< FILE_NB).
                        k ^= unsafe { z.enpassant_unchecked(file_of(ep_square) as usize) };
                    }
                }
            } else if m.move_type() == MoveType::Promotion {
                let pt = m.promotion_type();
                let promotion = make_piece(us, pt);

                dp.add_pc = promotion;
                dp.add_sq = Some(to);
                dp.to = None;

                // Update hash keys
                // Zobrist::psq[pc][to] is zero, so we don't need to clear it
                // SAFETY: `to` is a square, `pc` and `promotion` are real pieces (< PIECE_NB) for the zobrist lookups, piece_count and
                // `piece_value_unchecked`; us is Color (0 or 1) for the 2-element arrays.
                unsafe {
                    k ^= z.psq_unchecked(promotion as usize, to as usize);
                    st.material_key ^= z.psq_unchecked(
                        promotion as usize,
                        8 + (self.piece_count_unchecked(promotion) as usize),
                    ) ^ z.psq_unchecked(
                        pc as usize,
                        8 + (self.piece_count_unchecked(pc) as usize) - 1,
                    );
                    *st.non_pawn_key.get_unchecked_mut(us as usize) ^=
                        z.psq_unchecked(promotion as usize, to as usize);

                    if pt as u8 <= PieceType::Bishop as u8 {
                        st.minor_piece_key ^= z.psq_unchecked(promotion as usize, to as usize);
                    }

                    // Update material
                    *st.non_pawn_material.get_unchecked_mut(us as usize) +=
                        piece_value_unchecked(promotion);
                }
            }

            // Update pawn hash key
            st.pawn_key ^= move_key;

            // Reset rule 50 draw counter
            st.rule50 = 0;
            st.rep_window = 0;
        } else {
            // SAFETY: us is Color (0 or 1), non_pawn_key has 2 elements
            unsafe {
                *st.non_pawn_key.get_unchecked_mut(us as usize) ^= move_key;
            }

            if type_of(pc) as u8 <= PieceType::Bishop as u8 {
                st.minor_piece_key ^= move_key;
            }
        }

        // Update the key with the final value
        st.key = k;

        // Move the piece. The tricky Chess960 castling is handled earlier.
        // Mirrors C++ position.cpp (including the ordering, which the
        // dirty-threat generation depends on: for a capture the mover leaves
        // `from` first and then replaces the victim on `to` via swap_piece).
        if m.move_type() != MoveType::Castling {
            let to_pc = if m.move_type() == MoveType::Promotion {
                make_piece(us, m.promotion_type())
            } else {
                pc
            };

            if captured != Piece::NoPiece && m.move_type() != MoveType::EnPassant {
                self.remove_piece_dts(from, dts);
                self.swap_piece_dts(to, to_pc, dts);
            } else if pc == to_pc {
                self.move_piece_dts(from, to, dts);
            } else {
                self.remove_piece_dts(from, dts);
                self.put_piece_dts(to_pc, to, dts);
            }
        }

        // Set capture piece
        st.captured_piece = captured;

        // Calculate checkers bitboard (if move gives check)
        // Match upstream's hot path: ordinary moves use zero, and only moves
        // already known to give check pay for the attacker lookup.
        st.checkers_bb = if gives_check {
            self.attackers_to(self.square_of(them, PieceType::King)) & self.pieces_of_color(us)
        } else {
            0
        };

        self.side_to_move = !self.side_to_move;

        // Update king attacks used for fast check detection
        self.set_extra_bitboards_st(st);

        st.piece_placement = pack_pieces(&self.board);

        // Calculate the repetition info. It is the ply distance from the previous
        // occurrence of the same position, negative in the 3-fold case, or zero
        // if the position was not repeated.
        st.repetition = 0;
        let end = st.rep_window;
        if end >= 4 {
            let mut i = 4;
            while i <= end {
                let stp = self.st_back(i as usize);
                if stp.key == st.key && stp.piece_placement == st.piece_placement {
                    st.repetition = if stp.repetition != 0 { -i } else { i };
                    break;
                }
                i += 2;
            }
        }

        debug_assert!(
            self.pos_is_ok(),
            "Position corrupted after do_move_internal for move {:?}",
            m
        );

        // SFNNv16: pawn bitboards after the move (companion to the `before`
        // snapshot taken at the top of the function).
        dpps.after[Color::White as usize] =
            self.pieces_c_pt::<{ PieceType::Pawn as u8 }>(Color::White);
        dpps.after[Color::Black as usize] =
            self.pieces_c_pt::<{ PieceType::Pawn as u8 }>(Color::Black);
    }

    /// Unmakes a move. When it returns, the position should be restored to
    /// exactly the same state as before the move was made.
    ///
    /// # Safety
    /// `m` must be the most recent move made on this position, with no intervening
    /// unmatched move or null move, and fewer than `MAX_HIST` moves and null moves may have been
    /// made since: deeper lines overwrite the ring slot that holds the state to restore.
    // NOT inlined to match C++ (undo_move is a separate function call in C++)
    #[inline(never)]
    pub unsafe fn undo_move(&mut self, m: Move) {
        debug_assert!(m.is_ok(), "Invalid move in undo_move");

        self.side_to_move = !self.side_to_move;
        self.game_ply -= 1;

        let us = self.side_to_move; // side to move before making m (we flipped already)
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();

        debug_assert!(
            self.empty(from) || m.move_type() == MoveType::Castling,
            "From square not empty in undo_move"
        );
        debug_assert!(
            type_of(self.st().captured_piece) != PieceType::King,
            "Captured king in undo_move"
        );

        // Undo castling rook move first
        if m.move_type() == MoveType::Castling {
            let mut castle_to = to;
            self.do_castling::<false>(us, from, &mut castle_to, None, None);
        } else {
            // If promotion, replace promoted piece with pawn at from
            if m.move_type() == MoveType::Promotion {
                // Remove promoted piece at to, add pawn at from
                self.remove_piece(to);
                self.put_piece(make_piece(us, PieceType::Pawn), from);
            } else {
                self.move_piece(to, from);
            }

            // Restore captured piece
            let captured = self.st().captured_piece;
            if captured != Piece::NoPiece {
                if m.move_type() == MoveType::EnPassant {
                    let capsq = to - pawn_push(us);
                    debug_assert!(self.st_back(1).ep_square == Some(to));
                    self.put_piece(make_piece(them, PieceType::Pawn), capsq);
                } else {
                    self.put_piece(captured, to);
                }
            }
        }

        // Finally point our state pointer back to the previous state
        self.previous_state();

        debug_assert!(
            self.pos_is_ok(),
            "Position corrupted after undo_move for move {:?}",
            m
        );
    }

    /// Used to do a "null move": it flips
    /// the side to move without executing any move on the board.
    ///
    /// # Safety
    /// The side to move must not be in check.
    /// ```compile_fail,E0133
    /// use stockcrab::position::Position;
    /// let mut pos = Position::default();
    /// pos.do_null_move();
    /// ```
    pub unsafe fn do_null_move(&mut self) {
        debug_assert!(self.checkers() == 0, "do_null_move called when in check");

        // Cache zobrist reference
        let z = zobrist();

        let previous_st = self.st;
        self.next_state();

        // SAFETY: `previous_st` and the new `self.st` are distinct slots of the ring
        // (MAX_HIST > 1), and `offset_of!(blockers_for_king)` is within `size_of::<StateInfo>()`
        // (`repr(C)`), so the byte copy is in bounds and non-overlapping, and it only overwrites
        // plain-data fields of an initialised slot.
        unsafe {
            // Remaining fields are recomputed, invalidated or overwritten below.
            core::ptr::copy_nonoverlapping(
                previous_st as *const u8,
                self.st as *mut u8,
                core::mem::offset_of!(StateInfo, blockers_for_king),
            );
            (*self.st).piece_placement = (*previous_st).piece_placement;

            // Remove EP from key if present and clear it
            if let Some(ep) = self.st().ep_square {
                self.st_mut().key ^= z.enpassant_unchecked(file_of(ep) as usize);
                self.st_mut().ep_square = None;
            }
            // Toggle side in key
            self.st_mut().key ^= ZOBRIST_SIDE;
            self.st_mut().rep_window = 0;
            self.st_mut().captured_piece = Piece::NoPiece;
            self.st_mut().repetition = 0;
        }
        self.side_to_move = !self.side_to_move;
        self.game_ply += 1;
        // Recompute derived info dependent on stm
        self.set_extra_bitboards();

        debug_assert!(self.pos_is_ok(), "Position corrupted after do_null_move");
    }

    /// Must be used to undo a "null move"
    ///
    /// # Safety
    /// The most recent unmatched operation must be `do_null_move`, with fewer than `MAX_HIST` moves
    /// and null moves made since (see [`Position::undo_move`]).
    pub unsafe fn undo_null_move(&mut self) {
        debug_assert!(self.checkers() == 0, "undo_null_move called when in check");

        self.previous_state();
        self.side_to_move = !self.side_to_move;
        self.game_ply -= 1;

        debug_assert!(self.pos_is_ok(), "Position corrupted after undo_null_move");
    }

    // -------------------------------------------------------------------
    // SEE
    // -------------------------------------------------------------------
    /// Tests if the SEE (Static Exchange Evaluation)
    /// value of the move is greater or equal to the given threshold. We'll use an
    /// algorithm similar to alpha-beta pruning with a null window.
    #[inline(never)]
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        debug_assert!(m.is_ok());
        if m.move_type() != MoveType::Normal {
            return VALUE_ZERO >= threshold;
        }
        let from = m.from_sq();
        let to = m.to_sq();

        // Ensure we don't try to move a nonexistent piece
        // See: https://github.com/official-stockfish/Stockfish/pull/6342
        debug_assert!(self.piece_on(from) != Piece::NoPiece);

        // SAFETY: `piece_on(to)` is a Piece, so < PIECE_NB as `piece_value_unchecked` requires.
        let mut swap = unsafe { piece_value_unchecked(self.piece_on(to)) } - threshold;
        if swap < 0 {
            return false;
        }
        // SAFETY: `piece_on(from)` is a Piece, so < PIECE_NB as `piece_value_unchecked` requires.
        swap = unsafe { piece_value_unchecked(self.piece_on(from)) } - swap;
        if swap <= 0 {
            return true;
        }
        debug_assert!(color_of(self.piece_on(from)) == self.side_to_move);
        let mut occupied = self.pieces() ^ (1u64 << (from as u8)) ^ (1u64 << (to as u8));
        let mut stm = self.side_to_move;
        let mut attackers = self.attackers_to_occ(to, occupied);
        let mut res = 1;
        loop {
            stm = !stm;
            attackers &= occupied;
            // If stm has no more attackers then give up: stm loses
            let mut stm_attackers = attackers & self.pieces_of_color(stm);
            if stm_attackers == 0 {
                break;
            }
            // Don't allow pinned pieces to attack as long as there are
            // pinners on their original square.
            if (self.pinners(!stm) & occupied) != 0 {
                stm_attackers &= !self.blockers_for_king(stm);
                if stm_attackers == 0 {
                    break;
                }
            }
            res ^= 1;

            // Use const-generic piece lookups for performance (matches C++ template pieces<T>())
            let mut bb;
            if {
                // Locate and remove the next least valuable attacker, and add to
                // the bitboard 'attackers' any X-ray attackers behind it.
                bb = stm_attackers & self.pieces_pt::<{ PieceType::Pawn as u8 }>();
                bb != 0
            } {
                swap = PAWN_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bitboard::least_significant_square_bb(bb);
                attackers |= bitboard::attacks_bb_bishop(to, occupied)
                    & self.pieces_pt2::<{ PieceType::Bishop as u8 }, { PieceType::Queen as u8 }>();
            } else if {
                bb = stm_attackers & self.pieces_pt::<{ PieceType::Knight as u8 }>();
                bb != 0
            } {
                swap = KNIGHT_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bitboard::least_significant_square_bb(bb);
            } else if {
                bb = stm_attackers & self.pieces_pt::<{ PieceType::Bishop as u8 }>();
                bb != 0
            } {
                swap = BISHOP_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bitboard::least_significant_square_bb(bb);
                attackers |= bitboard::attacks_bb_bishop(to, occupied)
                    & self.pieces_pt2::<{ PieceType::Bishop as u8 }, { PieceType::Queen as u8 }>();
            } else if {
                bb = stm_attackers & self.pieces_pt::<{ PieceType::Rook as u8 }>();
                bb != 0
            } {
                swap = ROOK_VALUE - swap;
                if swap < res {
                    break;
                }
                occupied ^= bitboard::least_significant_square_bb(bb);
                attackers |= bitboard::attacks_bb_rook(to, occupied)
                    & self.pieces_pt2::<{ PieceType::Rook as u8 }, { PieceType::Queen as u8 }>();
            } else if {
                bb = stm_attackers & self.pieces_pt::<{ PieceType::Queen as u8 }>();
                bb != 0
            } {
                swap = QUEEN_VALUE - swap;
                // implies that the previous recapture was done by a higher rated piece than a Queen (King is excluded)
                debug_assert!(swap >= res);
                occupied ^= bitboard::least_significant_square_bb(bb);
                let (b_attacks, r_attacks) = bitboard::both_attacks_bb(to, occupied);
                attackers |= (b_attacks
                    & self.pieces_pt2::<{ PieceType::Bishop as u8 }, { PieceType::Queen as u8 }>())
                    | (r_attacks
                        & self
                            .pieces_pt2::<{ PieceType::Rook as u8 }, { PieceType::Queen as u8 }>())
            } else {
                // If we "capture" with the king but the opponent still has attackers,
                // reverse the result.
                return if (attackers & !self.pieces_of_color(stm)) != 0 {
                    (res ^ 1) != 0
                } else {
                    res != 0
                };
            }
        }
        res != 0
    }

    // -------------------------------------------------------------------
    // Hash keys
    // -------------------------------------------------------------------
    /// The position's Zobrist key, adjusted for the rule-50 count (C++ `key()`); bit 0 is the
    /// side to move. Equal positions at different rule-50 counts can have different keys.
    #[inline(always)]
    pub fn key(&self) -> Key {
        self.adjust_key50::<false>(self.st().key)
    }

    /// Computes the (rule50-adjusted) transposition table key that the position
    /// would have after making move `m`, without actually making it. Used to
    /// prefetch the TT entry early. Mirrors C++ `Position::prefetch_key`.
    #[inline(always)]
    pub fn prefetch_key(&self, m: Move) -> Key {
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.piece_on(from);
        let captured = self.piece_on(to);
        let z = zobrist();
        let mut k = { self.st().key ^ ZOBRIST_SIDE };

        // SAFETY: `captured` and `pc` are pieces (< PIECE_NB) and `to`/`from` squares (< SQUARE_NB).
        unsafe {
            k ^= z.psq_unchecked(captured as usize, to as usize)
                ^ z.psq_unchecked(pc as usize, to as usize)
                ^ z.psq_unchecked(pc as usize, from as usize);
        }

        if captured != Piece::NoPiece || type_of(pc) == PieceType::Pawn {
            return k;
        }

        self.adjust_key50::<true>(k)
    }

    /// The key of the material configuration only (used by the tablebases).
    #[inline(always)]
    pub fn material_key(&self) -> Key {
        self.st().material_key
    }

    /// The key of the pawns, with the side to move in bit 0.
    #[inline(always)]
    pub fn pawn_key(&self) -> Key {
        self.st().pawn_key ^ self.side_to_move as Key
    }

    /// The key of the knights and bishops, with the side to move in bit 0.
    #[inline(always)]
    pub fn minor_piece_key(&self) -> Key {
        self.st().minor_piece_key ^ self.side_to_move as Key
    }

    /// The key of `c`'s non-pawn pieces (king included), with the side to move in bit 0.
    #[inline(always)]
    pub fn non_pawn_key(&self, c: Color) -> Key {
        // c is Color (0 or 1), non_pawn_key has 2 elements
        self.st().non_pawn_key[c as usize] ^ self.side_to_move as Key
    }

    // -------------------------------------------------------------------
    // Other properties
    // -------------------------------------------------------------------
    /// The number of plies since the start of the game (from the FEN's move number).
    #[inline(always)]
    pub fn game_ply(&self) -> i32 {
        self.game_ply
    }

    /// The retained repetition window since the last irreversible move, null move, or FEN.
    #[inline(always)]
    pub fn rep_window(&self) -> i32 {
        self.st().rep_window
    }

    /// Whether the position uses Chess960 castling.
    #[inline(always)]
    pub fn is_chess960(&self) -> bool {
        self.chess960
    }

    /// Pawnless && (3-men || 4-men-minors-only). Mirrors C++ `Position::dtz_is_dtm`.
    #[inline(always)]
    pub fn dtz_is_dtm(&self) -> bool {
        let all_pieces = bitboard::popcount(self.pieces()) as i32;
        self.count(PieceType::Pawn) == 0
            && (all_pieces == 3
                || (all_pieces == 4
                    && self.pieces_of_types(&[PieceType::Queen, PieceType::Rook]) == 0))
    }

    /// Tests whether the position is drawn by 50-move rule
    /// or by repetition. It does not detect stalemates.
    pub fn is_draw(&self, ply: i32) -> bool {
        // C++ checks: rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size())
        // This means: draw if rule50 > 99 AND (not in check OR has at least one legal move)
        // If in check with no legal moves, it's checkmate, not a draw
        if self.rule50_count() > 99 {
            if self.checkers() == 0 {
                return true;
            }
            // In check - verify there's at least one legal move
            // (if no legal moves, it's checkmate, not a draw)
            use crate::movegen::{GenType, generate};
            if !generate(self, GenType::All).is_empty() {
                return true;
            }
        }
        self.is_repetition(ply)
    }

    /// Returns true if the position repeats once earlier but strictly after the root,
    /// or twice before or at the root.
    pub fn is_repetition(&self, ply: i32) -> bool {
        let rep = self.st().repetition;
        rep != 0 && rep < ply
    }

    /// Tests if the position has a move which draws by repetition.
    /// This function accurately matches the outcome of is_draw() over all legal moves.
    pub fn upcoming_repetition(&self, ply: i32) -> bool {
        let end = self.rep_window();
        if end < 3 {
            return false;
        }
        let original_key = self.st().key;
        let mut other = original_key ^ self.st_back(1).key ^ ZOBRIST_SIDE;
        for i in (3..=end).step_by(2) {
            let stp0 = self.st_back(i as usize - 1);
            let stp = self.st_back(i as usize);
            other ^= stp0.key ^ stp.key ^ ZOBRIST_SIDE;
            if other != 0 {
                continue;
            }
            let move_key = original_key ^ stp.key;

            let table = &REPETITION_TABLE;
            let mut j = repetition_hash(move_key);
            let mut distance = 1;

            // Probe the repetition table for a move with the same key and distance from the current position.
            // SAFETY: `j` is masked with REPETITION_TABLE_MASK, so it is below REPETITION_TABLE_SIZE, the
            // length of the three arrays (this covers the `get_unchecked`s in the loop body too).
            while unsafe { *table.distances.get_unchecked(j) } >= distance {
                // If the keys match, check if the move is a repetition of the current position.
                // SAFETY: see above.
                if unsafe { *table.keys.get_unchecked(j) } == move_key {
                    // SAFETY: see above.
                    let mv = unsafe { *table.moves.get_unchecked(j) };
                    let s1 = mv.from_sq();
                    let s2 = mv.to_sq();

                    // The path must be clear of pieces for a move to be legal.
                    if (bitboard::between_bb_pub(s1, s2) ^ (1u64 << (s2 as u8))) & self.pieces()
                        == 0
                    {
                        let mut b = self.st().piece_placement;
                        swap_nibble(&mut b, s1, s2);

                        // For nodes before or at the root, check that the move is a
                        // repetition rather than a move to the current position.
                        if b == stp.piece_placement && (ply > i || stp.repetition != 0) {
                            return true;
                        }
                    }

                    break;
                }

                // Advance to the next slot in the probe sequence
                distance += 1;
                j = (j + 1) & REPETITION_TABLE_MASK;
            }
        }
        false
    }

    /// Tests whether there has been at least one repetition
    /// of positions since the last capture or pawn move.
    pub fn has_repeated(&self) -> bool {
        let end = self.rep_window();
        (0..=end - 4).any(|i| self.st_back(i as usize).repetition != 0)
    }

    /// The halfmove clock: plies since the last capture or pawn move.
    #[inline(always)]
    pub fn rule50_count(&self) -> i32 {
        self.st().rule50
    }

    /// The total value of `c`'s non-pawn pieces.
    #[inline(always)]
    pub fn non_pawn_material_of(&self, c: Color) -> Value {
        // c is Color (0 or 1), non_pawn_material has 2 elements
        self.st().non_pawn_material[c as usize]
    }

    /// The total value of both sides' non-pawn pieces.
    #[inline(always)]
    pub fn non_pawn_material(&self) -> Value {
        self.non_pawn_material_of(Color::White) + self.non_pawn_material_of(Color::Black)
    }

    // -------------------------------------------------------------------
    // Consistency & utilities
    // -------------------------------------------------------------------
    /// Performs some consistency checks and returns false if a check fails.
    /// This is meant to be helpful when debugging.
    pub fn pos_is_ok(&self) -> bool {
        const FAST: bool = true; // Quick (default) or full check?

        // Basic validity checks (always performed)
        // Check that exactly one king exists for each side
        if self.count_of(Color::White, PieceType::King) != 1 {
            return false;
        }
        if self.count_of(Color::Black, PieceType::King) != 1 {
            return false;
        }

        // Kings exist and count is correct, so we can safely access their squares
        // Additional validation: check king pieces are on their expected squares using safe bitboard operations
        let white_king_bb = self.pieces_color_types(Color::White, &[PieceType::King]);
        let black_king_bb = self.pieces_color_types(Color::Black, &[PieceType::King]);

        // Double-check bitboard matches piece count (detect bitboard corruption)
        if white_king_bb.count_ones() != 1 || black_king_bb.count_ones() != 1 {
            return false;
        }

        // Safely extract king squares using bitboard operations
        let white_king_sq = Square::from(bitboard::lsb(white_king_bb) as u8);
        let black_king_sq = Square::from(bitboard::lsb(black_king_bb) as u8);

        // Verify pieces on king squares are actually kings
        if self.piece_on(white_king_sq) != Piece::W_King
            || self.piece_on(black_king_sq) != Piece::B_King
        {
            return false;
        }

        // EP square validation
        if let Some(ep) = self.ep_square()
            && relative_rank_sq(self.side_to_move, ep) != Rank::R6
        {
            return false;
        }

        if FAST {
            return true;
        }

        // Full validation (skipped while FAST is true)

        // Check king counts
        if self.piece_count[Piece::W_King as usize] != 1
            || self.piece_count[Piece::B_King as usize] != 1
        {
            return false;
        }

        // Check if opponent's king is under attack (illegal position)
        let opp_ksq = self.square_of(!self.side_to_move, PieceType::King);
        if self.attackers_to_exist(opp_ksq, self.pieces(), self.side_to_move) {
            return false;
        }

        // Check pawn constraints
        let pawns = self.pieces_of_types(&[PieceType::Pawn]);
        if (pawns & (bitboard::RANK_1_BB | bitboard::RANK_8_BB)) != 0 {
            return false;
        }

        if self.piece_count[Piece::W_Pawn as usize] > 8
            || self.piece_count[Piece::B_Pawn as usize] > 8
        {
            return false;
        }

        // Check bitboard consistency
        let white = self.pieces_of_color(Color::White);
        let black = self.pieces_of_color(Color::Black);
        let all = self.pieces();

        if (white & black) != 0 {
            return false;
        }

        if (white | black) != all {
            return false;
        }

        if bitboard::popcount(white) as i32 > 16 || bitboard::popcount(black) as i32 > 16 {
            return false;
        }

        // Check for overlapping piece type bitboards
        for pt1 in 1..PIECE_TYPE_NB {
            for pt2 in (pt1 + 1)..PIECE_TYPE_NB {
                let pt1_type = PieceType::from(pt1 as u8);
                let pt2_type = PieceType::from(pt2 as u8);
                if (self.pieces_of_types(&[pt1_type]) & self.pieces_of_types(&[pt2_type])) != 0 {
                    return false;
                }
            }
        }

        // Verify piece counts match bitboards and board array
        // Use valid pieces array (like C++ Pieces[]) to avoid invalid enum values 7, 8
        const PIECES: [Piece; 12] = [
            Piece::W_Pawn,
            Piece::W_Knight,
            Piece::W_Bishop,
            Piece::W_Rook,
            Piece::W_Queen,
            Piece::W_King,
            Piece::B_Pawn,
            Piece::B_Knight,
            Piece::B_Bishop,
            Piece::B_Rook,
            Piece::B_Queen,
            Piece::B_King,
        ];
        for pc in PIECES {
            let color = color_of(pc);
            let ptype = type_of(pc);
            let bb_count = bitboard::popcount(self.pieces_color_types(color, &[ptype])) as i32;
            let array_count = self.board.iter().filter(|&&p| p == pc).count() as i32;

            if self.piece_count[pc as usize] != bb_count
                || self.piece_count[pc as usize] != array_count
            {
                return false;
            }
        }

        // Verify castling rights consistency
        for &c in &[Color::White, Color::Black] {
            for &side in &[CastlingRights::KING_SIDE, CastlingRights::QUEEN_SIDE] {
                let cr = c & side;
                if !self.can_castle(cr) {
                    continue;
                }

                let rook_sq = self.castling_rook_square(cr);
                let expected_rook = make_piece(c, PieceType::Rook);

                if self.piece_on(rook_sq) != expected_rook {
                    return false;
                }

                if self.castling_rights_mask[rook_sq as usize] != cr.bits() as i32 {
                    return false;
                }

                let king_sq = self.square_of(c, PieceType::King);
                if (self.castling_rights_mask[king_sq as usize] & cr.bits() as i32)
                    != cr.bits() as i32
                {
                    return false;
                }
            }
        }

        true
    }

    /// Flips the position: mirrors the board vertically and swaps colours, with the white and
    /// black sides reversed. This is only useful for debugging, e.g. for finding evaluation
    /// symmetry bugs.
    ///
    /// Port of C++ `Position::flip`, which returns `std::optional<PositionSetError>` because it
    /// re-enters `set()` with the rewritten FEN. Flipping can produce a position the validator
    /// rejects (mirroring makes the other side to move, so a position where only the flipped
    /// side's king is capturable becomes illegal). On Some(e) the position has been reset and
    /// must not be used.
    pub fn flip(&mut self) -> Option<PositionSetError> {
        // Exactly mirror the C++ implementation from position.cpp
        let fen_str = self.fen();
        let mut parts = fen_str.split_whitespace();
        let mut f = String::new();

        // Step 1: Reverse the ranks (like C++ for loop)
        if let Some(piece_placement) = parts.next() {
            let ranks: Vec<&str> = piece_placement.split('/').collect();
            for (i, rank) in ranks.iter().rev().enumerate() {
                if i > 0 {
                    f.push('/');
                }
                f.push_str(rank);
            }
        }
        f.push(' ');

        // Step 2: Add flipped color (uppercase to be transformed later)
        if let Some(active_color) = parts.next() {
            if active_color == "w" {
                f.push_str("B ");
            } else {
                f.push_str("W ");
            }
        }

        // Step 3: Add castling rights as-is
        if let Some(castling) = parts.next() {
            f.push_str(castling);
            f.push(' ');
        }

        // Step 4: Apply case transformation (like C++ std::transform)
        let mut f_chars: Vec<char> = f.chars().collect();
        for c in &mut f_chars {
            if c.is_ascii_lowercase() {
                *c = c.to_ascii_uppercase();
            } else if c.is_ascii_uppercase() {
                *c = c.to_ascii_lowercase();
            }
        }
        f = f_chars.into_iter().collect();

        // En passant square - flip rank 3 <-> 6 (added AFTER transformation)
        if let Some(ep) = parts.next() {
            if ep == "-" {
                f.push_str(ep);
            } else if ep.len() == 2 {
                let chars: Vec<char> = ep.chars().collect();
                f.push(chars[0]);
                f.push(if chars[1] == '3' { '6' } else { '3' });
            }
        }

        // Half and full moves - append remaining (added AFTER transformation)
        if let Some(half_move) = parts.next() {
            f.push(' ');
            f.push_str(half_move);
        }
        if let Some(full_move) = parts.next() {
            f.push(' ');
            f.push_str(full_move);
        }

        // Set the position with the flipped FEN
        let chess960 = self.is_chess960();
        if let Err(e) = self.set_fen(&f, chess960) {
            return Some(e);
        }

        debug_assert!(self.pos_is_ok());
        None
    }

    // -------------------------------------------------------------------
    // Piece mutations (fast inline helpers)
    // -------------------------------------------------------------------

    /// A non-queen slider is never recorded as threatening a queen.
    /// Mirrors C++ `can_slider_threat` (position.cpp).
    #[inline(always)]
    fn can_slider_threat(pc: Piece, slider: Piece) -> bool {
        type_of(pc) != PieceType::Queen || type_of(slider) == PieceType::Queen
    }

    /// Slider part of `update_piece_threats`; mirrors the C++ `process_sliders`
    /// lambda (position.cpp). Emits the discovered threat that the
    /// slider gains/loses through `s`, and optionally the direct slider->pc threat.
    #[inline(always)]
    fn process_slider_threats(
        &self,
        mut sliders: Bitboard,
        pc: Piece,
        s: Square,
        dts: &mut DirtyThreats,
        put_piece: bool,
        add_direct_attacks: bool,
        no_rays_containing: Bitboard,
        r_attacks: Bitboard,
        b_attacks: Bitboard,
        occupied_no_k: Bitboard,
    ) {
        while sliders != 0 {
            let slider_sq = bitboard::pop_lsb(&mut sliders);
            let slider = self.piece_on(slider_sq);

            let ray = bitboard::ray_pass_bb(slider_sq, s);
            let discovered = ray & (r_attacks | b_attacks) & occupied_no_k;

            debug_assert!(bitboard::popcount(discovered) <= 1);

            // no_rays_containing lets a move skip the discovered threats along
            // the ray it stays on (from|to for a sliding move).
            if discovered != 0 && (ray & no_rays_containing) != no_rays_containing {
                let threatened_sq = bitboard::lsb(discovered);
                let threatened_pc = self.piece_on(threatened_sq);
                if Self::can_slider_threat(threatened_pc, slider) {
                    Self::add_dirty_threat(
                        dts,
                        slider,
                        threatened_pc,
                        slider_sq,
                        threatened_sq,
                        !put_piece,
                    );
                }
            }

            if add_direct_attacks && Self::can_slider_threat(pc, slider) {
                Self::add_dirty_threat(dts, slider, pc, slider_sq, s, put_piece);
            }
        }
    }

    /// Update threat tracking for piece placement/removal.
    /// put_piece: true for placement, false for removal
    /// `no_rays_containing` skips the discovered threats along rays that contain it.
    /// Mirrors C++ `Position::update_piece_threats` (position.cpp).
    pub(crate) fn update_piece_threats<const COMPUTE_RAY: bool>(
        &self,
        pc: Piece,
        s: Square,
        dts: &mut DirtyThreats,
        put_piece: bool,
        no_rays_containing: Bitboard,
    ) {
        let occupied = self.pieces();
        let (b_attacks, r_attacks) = bitboard::both_attacks_bb(s, occupied);

        let occupied_no_k = occupied ^ self.pieces_pt::<{ PieceType::King as u8 }>();

        // Sliders that (would) attack `s`.
        let sliders = (self.pieces_of_types(&[PieceType::Rook, PieceType::Queen]) & r_attacks)
            | (self.pieces_of_types(&[PieceType::Bishop, PieceType::Queen]) & b_attacks);
        let pt = type_of(pc);

        // Kings neither give nor receive threat features; only the discovered
        // threats they unblock/block matter. (C++ position.cpp)
        if pt == PieceType::King {
            if COMPUTE_RAY {
                self.process_slider_threats(
                    sliders,
                    pc,
                    s,
                    dts,
                    put_piece,
                    false,
                    no_rays_containing,
                    r_attacks,
                    b_attacks,
                    occupied_no_k,
                );
            }
            return;
        }

        // Pawns carry no push threats, and a pawn's outgoing threats are restricted
        // to knights and rooks (pawn-pair features cover pawn-vs-pawn relations).
        let threat_targets = if pt == PieceType::Pawn {
            self.pieces_pt2::<{ PieceType::Knight as u8 }, { PieceType::Rook as u8 }>()
        } else if pt == PieceType::Bishop || pt == PieceType::Rook {
            self.pieces_of_types(&[
                PieceType::Pawn,
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Rook,
            ])
        } else {
            occupied_no_k
        };
        // Threatened squares, using the pre-computed slider attacks.
        #[allow(unused_mut)] // only the scalar (non-AVX-512) path pops bits
        let mut threatened = (if pt == PieceType::Bishop {
            b_attacks
        } else if pt == PieceType::Rook {
            r_attacks
        } else if pt == PieceType::Queen {
            b_attacks | r_attacks
        } else if pt == PieceType::Pawn {
            bitboard::pawn_attacks_from_square(s, color_of(pc))
        } else {
            bitboard::attacks_empty(pt, s)
        }) & threat_targets;

        // Find non-slider incoming threats (knights, plus pawns for knights/rooks).
        let mut incoming_threats = bitboard::attacks_empty(PieceType::Knight, s)
            & self.pieces_pt::<{ PieceType::Knight as u8 }>();

        // Incoming pawn threats are only tracked for knights and rooks.
        if pt == PieceType::Knight || pt == PieceType::Rook {
            incoming_threats |= (bitboard::pawn_attacks_from_square(s, Color::White)
                & self.pieces_color_types(Color::Black, &[PieceType::Pawn]))
                | (bitboard::pawn_attacks_from_square(s, Color::Black)
                    & self.pieces_color_types(Color::White, &[PieceType::Pawn]));
        }

        // AVX512ICL path (C++ `USE_AVX512ICL`): write all direct threats with vector
        // stores; only the discovered threats are left to `process_sliders(false)`.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2"
        ))]
        {
            self.write_multiple_dirties::<{ DirtyThreat::THREATENED_SQ_OFFSET }, { DirtyThreat::THREATENED_PC_OFFSET }>(
                threatened,
                DirtyThreat::new(pc, Piece::NoPiece, s, Square::SQ_A1, put_piece).raw(),
                dts,
            );
            // A non-queen slider never threatens a queen, so filter those out up front.
            let direct_sliders = if pt == PieceType::Queen {
                sliders & self.pieces_pt::<{ PieceType::Queen as u8 }>()
            } else {
                sliders
            };
            self.write_multiple_dirties::<{ DirtyThreat::PC_SQ_OFFSET }, { DirtyThreat::PC_OFFSET }>(
                direct_sliders | incoming_threats,
                DirtyThreat::new(Piece::NoPiece, pc, Square::SQ_A1, s, put_piece).raw(),
                dts,
            );

            // For ICL, direct threats were written above
            if COMPUTE_RAY {
                self.process_slider_threats(
                    sliders,
                    pc,
                    s,
                    dts,
                    put_piece,
                    false,
                    no_rays_containing,
                    r_attacks,
                    b_attacks,
                    occupied_no_k,
                );
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2"
        )))]
        {
            // Add outgoing threats (pieces this piece attacks)
            while threatened != 0 {
                let threatened_sq = bitboard::pop_lsb(&mut threatened);
                let threatened_pc = self.piece_on(threatened_sq);

                debug_assert!(threatened_sq != s);
                debug_assert!(threatened_pc != Piece::NoPiece);

                Self::add_dirty_threat(dts, pc, threatened_pc, s, threatened_sq, put_piece);
            }

            if COMPUTE_RAY {
                self.process_slider_threats(
                    sliders,
                    pc,
                    s,
                    dts,
                    put_piece,
                    true,
                    no_rays_containing,
                    r_attacks,
                    b_attacks,
                    occupied_no_k,
                );
            } else {
                // A non-queen slider never threatens a queen, so filter those out up front.
                incoming_threats |= if pt == PieceType::Queen {
                    sliders & self.pieces_pt::<{ PieceType::Queen as u8 }>()
                } else {
                    sliders
                };
            }

            // Handle non-slider incoming threats
            while incoming_threats != 0 {
                let src_sq = bitboard::pop_lsb(&mut incoming_threats);
                let src_pc = self.piece_on(src_sq);

                debug_assert!(src_sq != s);
                debug_assert!(src_pc != Piece::NoPiece);

                Self::add_dirty_threat(dts, src_pc, pc, src_sq, s, put_piece);
            }
        }
    }

    /// Writes one `DirtyThreat` per square in `mask` (at most 16) with vector stores: `template`
    /// holds the fixed fields, and each square index / piece on that square is shifted into
    /// place by `SQ_SHIFT` / `PC_SHIFT`. Mirrors C++ `write_multiple_dirties`
    /// (position.cpp, `USE_AVX512ICL`).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2"
    ))]
    #[inline(always)]
    fn write_multiple_dirties<const SQ_SHIFT: u32, const PC_SHIFT: u32>(
        &self,
        mask: Bitboard,
        template: u32,
        dts: &mut DirtyThreats,
    ) {
        use core::arch::x86_64::*;
        // Byte vector holding square indices 0..63 (C++ bitboard.h `AllSquares`).
        static ALL_SQUARES: [u8; 64] = {
            let mut a = [0u8; 64];
            let mut i = 0;
            while i < 64 {
                a[i] = i as u8;
                i += 1;
            }
            a
        };
        const { assert!(size_of::<DirtyThreat>() == 4) };
        let count = bitboard::popcount(mask) as usize;
        debug_assert!(count <= 16);
        // The 16-entry store below needs len + 16 <= MAX_DIRTY_THREATS; the C++ bound is
        // 80 changed features per move, leaving exactly that slack.
        debug_assert!(dts.list.len + 16 <= MAX_DIRTY_THREATS);

        // SAFETY: the cfg guarantees AVX-512 F/BW/VBMI/VBMI2 at compile time. `ALL_SQUARES` and `board` is a
        // [Piece; 64] (repr(u8)), exactly one 64-byte vector. At most 16 entries are ever pending
        // (a piece has at most 8 outgoing + 16 incoming threats, and the move generator's
        // bound of 80 total entries leaves len + 16 <= MAX_DIRTY_THREATS), so the 64-byte unaligned
        // store at `write` stays inside the list's backing array; entries beyond `count` are
        // garbage that later pushes overwrite and `len` excludes.
        unsafe {
            let board = _mm512_loadu_si512(self.board.as_ptr() as *const __m512i);
            let template_v = _mm512_set1_epi32(template as i32);
            let write = dts.list.make_space(count);

            // Extract the list of squares and upconvert to 32 bits. There are never more than
            // 16 incoming threats so this is sufficient.
            let mut threat_squares = _mm512_maskz_compress_epi8(
                mask,
                _mm512_loadu_si512(ALL_SQUARES.as_ptr() as *const __m512i),
            );
            threat_squares = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(threat_squares));

            let mut threat_pieces =
                _mm512_maskz_permutexvar_epi8(0x1111_1111_1111_1111u64, threat_squares, board);

            // Shift the piece and square into place
            threat_squares = _mm512_slli_epi32::<SQ_SHIFT>(threat_squares);
            threat_pieces = _mm512_slli_epi32::<PC_SHIFT>(threat_pieces);

            let dirties =
                _mm512_ternarylogic_epi32::<254>(template_v, threat_squares, threat_pieces);
            _mm512_storeu_si512(write as *mut _, dirties);
        }
    }

    /// Helper to add a dirty threat entry
    #[inline(always)]
    fn add_dirty_threat(
        dts: &mut DirtyThreats,
        pc: Piece,
        threatened: Piece,
        s: Square,
        threatened_sq: Square,
        add: bool,
    ) {
        // Only appends to the list; the NNUE consumer reads only the list.
        dts.list
            .push(DirtyThreat::new(pc, threatened, s, threatened_sq, add));
    }

    /// Place a piece on the board. Updates board array, bitboards, and piece counts.
    /// Matches C++ position.h. Includes debug validation for consistency.
    #[inline(always)]
    pub(crate) fn put_piece(&mut self, pc: Piece, s: Square) {
        // Debug validation before placing piece
        debug_assert!(pc != Piece::NoPiece, "Cannot put NoPiece on board");
        debug_assert!(
            self.board[s as usize] == Piece::NoPiece,
            "Square {:?} already occupied by {:?}, cannot place {:?}",
            s,
            self.board[s as usize],
            pc
        );

        let s_bb = 1u64 << (s as u8);

        // Record original counts for validation in debug mode
        #[cfg(debug_assertions)]
        let original_piece_count = self.piece_count[pc as usize];
        #[cfg(debug_assertions)]
        let original_type_bb_count = self.by_type_bb[type_of(pc) as usize].count_ones();
        #[cfg(debug_assertions)]
        let original_color_bb_count = self.by_color_bb[color_of(pc) as usize].count_ones();

        // Perform actual piece placement
        // SAFETY: s is Square (0-63), board has 64 elements
        // ALL_PIECES is 0 (NoPieceType), type_of returns 0-7, by_type_bb has 8 elements
        // color_of returns 0 or 1, by_color_bb has 2 elements
        // pc is Piece (0-15), piece_count has 16 elements
        unsafe {
            *self.board.get_unchecked_mut(s as usize) = pc;
            // ALL_PIECES (index 0) stores union of all piece squares
            *self.by_type_bb.get_unchecked_mut(ALL_PIECES as usize) |= s_bb;
            *self.by_type_bb.get_unchecked_mut(type_of(pc) as usize) |= s_bb;
            *self.by_color_bb.get_unchecked_mut(color_of(pc) as usize) |= s_bb;
            *self.piece_count.get_unchecked_mut(pc as usize) += 1;
        }
        // Per-color ALL slot is piece index 0 (white) or 8 (black)
        // Use raw index calculation to avoid invalid Piece enum value for index 8
        // SAFETY: color_of returns 0 or 1, so (color << 3) is 0 or 8, piece_count has 16 elements
        unsafe {
            *self
                .piece_count
                .get_unchecked_mut((color_of(pc) as usize) << 3) += 1;
        }

        // Validate piece count consistency after placement
        #[cfg(debug_assertions)]
        {
            let new_piece_count = self.piece_count[pc as usize];
            let new_type_bb_count = self.by_type_bb[type_of(pc) as usize].count_ones();
            let new_color_bb_count = self.by_color_bb[color_of(pc) as usize].count_ones();

            if new_piece_count != original_piece_count + 1 {
                panic!("put_piece: Piece count inconsistency");
            } else if new_type_bb_count != original_type_bb_count + 1 {
                panic!("put_piece: Type bitboard count inconsistency");
            } else if new_color_bb_count != original_color_bb_count + 1 {
                panic!("put_piece: Color bitboard count inconsistency");
            }
        }
    }

    /// Remove a piece from the board. Updates board array, bitboards, and piece counts.
    /// Matches C++ position.h. Includes debug validation for consistency.
    #[inline(always)]
    pub(crate) fn remove_piece(&mut self, s: Square) {
        // s is Square (0-63), board has 64 elements
        let pc = self.board[s as usize];

        // Debug validation before removing piece
        debug_assert!(
            pc != Piece::NoPiece,
            "Cannot remove piece from empty square {:?}",
            s
        );

        // Note: We don't validate kings here because during castling, kings are temporarily removed
        // The validation will happen at the move level to ensure the position remains valid overall

        let s_bb = 1u64 << (s as u8);

        // Record original counts for validation in debug mode
        #[cfg(debug_assertions)]
        let original_piece_count = self.piece_count[pc as usize];
        #[cfg(debug_assertions)]
        let original_type_bb_count = self.by_type_bb[type_of(pc) as usize].count_ones();
        #[cfg(debug_assertions)]
        let original_color_bb_count = self.by_color_bb[color_of(pc) as usize].count_ones();

        // Perform actual piece removal
        // SAFETY: ALL_PIECES is 0 (NoPieceType), type_of returns 0-7, by_type_bb has 8 elements
        // color_of returns 0 or 1, by_color_bb has 2 elements
        // pc is Piece (0-15), piece_count has 16 elements
        unsafe {
            *self.by_type_bb.get_unchecked_mut(ALL_PIECES as usize) ^= s_bb;
            *self.by_type_bb.get_unchecked_mut(type_of(pc) as usize) ^= s_bb;
            *self.by_color_bb.get_unchecked_mut(color_of(pc) as usize) ^= s_bb;
            *self.board.get_unchecked_mut(s as usize) = Piece::NoPiece;
            *self.piece_count.get_unchecked_mut(pc as usize) -= 1;
        }
        // Use raw index calculation to avoid invalid Piece enum value for index 8
        // SAFETY: color_of returns 0 or 1, so (color << 3) is 0 or 8, piece_count has 16 elements
        unsafe {
            *self
                .piece_count
                .get_unchecked_mut((color_of(pc) as usize) << 3) -= 1;
        }

        // Validate piece count consistency after removal
        #[cfg(debug_assertions)]
        {
            let new_piece_count = self.piece_count[pc as usize];
            let new_type_bb_count = self.by_type_bb[type_of(pc) as usize].count_ones();
            let new_color_bb_count = self.by_color_bb[color_of(pc) as usize].count_ones();

            if new_piece_count != original_piece_count - 1 {
                panic!("remove_piece: Piece count inconsistency");
            } else if new_type_bb_count != original_type_bb_count - 1 {
                panic!("remove_piece: Type bitboard count inconsistency");
            } else if new_color_bb_count != original_color_bb_count - 1 {
                panic!("remove_piece: Color bitboard count inconsistency");
            }
        }
    }

    /// Move a piece from one square to another, toggling the from and to bits with XOR.
    #[inline(always)]
    pub(crate) fn move_piece(&mut self, from: Square, to: Square) {
        // from and to are Square (0-63), board has 64 elements
        let pc = self.board[from as usize];
        debug_assert!(pc != Piece::NoPiece);
        let from_to_bb = (1u64 << (from as u8)) | (1u64 << (to as u8));
        // SAFETY: ALL_PIECES is 0 (NoPieceType), type_of returns 0-7, by_type_bb has 8 elements
        unsafe {
            *self.by_type_bb.get_unchecked_mut(ALL_PIECES as usize) ^= from_to_bb;
            *self.by_type_bb.get_unchecked_mut(type_of(pc) as usize) ^= from_to_bb;
        }
        // SAFETY: color_of returns 0 or 1, by_color_bb has 2 elements
        unsafe {
            *self.by_color_bb.get_unchecked_mut(color_of(pc) as usize) ^= from_to_bb;
        }
        // SAFETY: from and to are Square (0-63), board has 64 elements
        unsafe {
            *self.board.get_unchecked_mut(from as usize) = Piece::NoPiece;
            *self.board.get_unchecked_mut(to as usize) = pc;
        }
    }

    // -------------------------------------------------------------------
    // Piece mutations that also record the NNUE threat-feature diff.
    // These mirror the `DirtyThreats* dts` overloads of the C++ mutators
    // (position.h). The threat update must happen with the *live*
    // occupancy: after a placement, before a removal.
    // -------------------------------------------------------------------

    /// `put_piece(pc, s, dts)` — C++ position.h.
    #[inline(always)]
    pub(crate) fn put_piece_dts(&mut self, pc: Piece, s: Square, dts: &mut DirtyThreats) {
        self.put_piece(pc, s);
        self.update_piece_threats::<true>(pc, s, dts, true, !0u64);
    }

    /// `remove_piece(s, dts)` — C++ position.h.
    #[inline(always)]
    pub(crate) fn remove_piece_dts(&mut self, s: Square, dts: &mut DirtyThreats) {
        let pc = self.piece_on(s);
        self.update_piece_threats::<true>(pc, s, dts, false, !0u64);
        self.remove_piece(s);
    }

    /// `move_piece(from, to, dts)` — C++ position.h.
    #[inline(always)]
    pub(crate) fn move_piece_dts(&mut self, from: Square, to: Square, dts: &mut DirtyThreats) {
        let pc = self.piece_on(from);
        let from_to = (1u64 << (from as u8)) | (1u64 << (to as u8));

        self.update_piece_threats::<true>(pc, from, dts, false, from_to);
        self.move_piece(from, to);
        self.update_piece_threats::<true>(pc, to, dts, true, from_to);
    }

    /// `swap_piece(s, pc, dts)` — C++ position.h. Used when a capture
    /// replaces the piece standing on `to` (including capture-promotions); the
    /// ray/discovered threats do not change, hence ComputeRay == false.
    #[inline(always)]
    pub(crate) fn swap_piece_dts(&mut self, s: Square, pc: Piece, dts: &mut DirtyThreats) {
        let old = self.piece_on(s);

        self.remove_piece(s);
        self.update_piece_threats::<false>(old, s, dts, false, !0u64);
        self.put_piece(pc, s);
        self.update_piece_threats::<false>(pc, s, dts, true, !0u64);
    }

    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------
    // Matches C++ `template<bool AfterMove> adjust_key50`. When AFTER_MOVE is true
    // (used by prefetch_key, before rule50 is incremented by do_move) the threshold
    // is lowered by one.
    #[inline(always)]
    fn adjust_key50<const AFTER_MOVE: bool>(&self, k: Key) -> Key {
        self.adjust_key50_with::<AFTER_MOVE>(self.st().rule50, k)
    }

    /// `adjust_key50` for a caller that already holds the current `rule50`.
    #[inline(always)]
    fn adjust_key50_with<const AFTER_MOVE: bool>(&self, rule50: i32, k: Key) -> Key {
        let threshold = 14 - AFTER_MOVE as i32;
        if rule50 < threshold {
            k
        } else {
            k ^ (make_key(((rule50 - threshold) / 8) as u64) << 1)
        }
    }

    /// Helper function used to set castling
    /// rights given the corresponding color and the rook starting square.
    fn set_castling_right(&mut self, c: Color, rfrom: Square) {
        let kfrom = self.square_of(c, PieceType::King);
        let cr = c & if (kfrom as u8) < (rfrom as u8) {
            CastlingRights::KING_SIDE
        } else {
            CastlingRights::QUEEN_SIDE
        };
        {
            self.st_mut().castling_rights |= cr;
        }
        self.castling_rights_mask[kfrom as usize] |= cr.bits() as i32;
        self.castling_rights_mask[rfrom as usize] |= cr.bits() as i32;
        self.castling_rook_square[cr.bits() as usize] = Some(rfrom);
        let kto = relative_square(
            c,
            if !(cr & CastlingRights::KING_SIDE).is_empty() {
                Square::SQ_G1
            } else {
                Square::SQ_C1
            },
        );
        let rto = relative_square(
            c,
            if !(cr & CastlingRights::KING_SIDE).is_empty() {
                Square::SQ_F1
            } else {
                Square::SQ_D1
            },
        );
        let path = (bitboard::between_bb_pub(rfrom, rto) | bitboard::between_bb_pub(kfrom, kto))
            & !((1u64 << (kfrom as u8)) | (1u64 << (rfrom as u8)));
        self.castling_path[cr.bits() as usize] = path;
    }

    /// Sets pinners and where stm can give check. Squares threatened by ~stm are
    /// computed lazily by `compute_threats()`.
    #[inline(always)]
    fn set_extra_bitboards(&mut self) {
        // SAFETY: `self.st` is live (see `st()`) and `&mut self` guarantees exclusive access.
        let st = unsafe { &mut *self.st };
        self.set_extra_bitboards_st(st);
    }

    /// `set_extra_bitboards` writing into `st`, which `do_move` holds in a local so the state
    /// pointer is not reloaded from `self` after every store.
    #[inline(always)]
    fn set_extra_bitboards_st(&self, st: &mut StateInfo) {
        self.update_slider_blockers_st(st, Color::White);
        self.update_slider_blockers_st(st, Color::Black);
        let ksq = self.square_of(!self.side_to_move, PieceType::King);
        st.check_squares[PieceType::Pawn as usize] =
            bitboard::pawn_attacks_from_square(ksq, !self.side_to_move);
        st.check_squares[PieceType::Knight as usize] =
            bitboard::attacks_bb(PieceType::Knight, ksq, 0);
        // Fetch both slider attack sets from the same dual entry, mirroring
        // C++ `both_attacks_bb(ksq, pieces())`.
        let (b_attacks, r_attacks) = bitboard::both_attacks_bb(ksq, self.pieces());
        st.check_squares[PieceType::Bishop as usize] = b_attacks;
        st.check_squares[PieceType::Rook as usize] = r_attacks;
        st.check_squares[PieceType::Queen as usize] = b_attacks | r_attacks;
        st.check_squares[PieceType::King as usize] = 0;

        st.threats.set([0; PIECE_TYPE_NB]);
    }

    /// Squares threatened by the side not to move, computed lazily on the first
    /// [`Position::threats_by`] (an empty `threats[ALL_PIECES]` marks them as not computed:
    /// the enemy king always attacks some square).
    fn compute_threats(&self) {
        // X-ray through our king so its destinations can be tested without recomputing attacks.
        let occupied = self.pieces()
            ^ bitboard::square_bb_pub(self.square_of(self.side_to_move, PieceType::King));
        let them = !self.side_to_move;
        let mut threats = [0; PIECE_TYPE_NB];
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            threats[pt as usize] = self.xattacks_by(pt, them, occupied);
            threats[ALL_PIECES as usize] |= threats[pt as usize];
        }
        // Like the C++ `const` method, this only caches data derived from the position.
        self.st().threats.set(threats);
    }

    /// Initialize all StateInfo fields from the current position.
    /// Called by set_fen(). Matches C++ position.cpp.
    ///
    /// Computes the hash keys of the position, and other
    /// data that once computed is updated incrementally as moves are made.
    /// The function is only used when a new position is set up
    fn set_state(&mut self) {
        {
            self.st_mut().key = self.side_to_move as Key;
            self.st_mut().material_key = 0;
            self.st_mut().minor_piece_key = 0;
            self.st_mut().non_pawn_key = [0; COLOR_NB];
            self.st_mut().pawn_key = 0;
            self.st_mut().non_pawn_material = [VALUE_ZERO; COLOR_NB];
            // repetition is read from previous StateInfo objects in the chain, so it must be initialized
            self.st_mut().repetition = 0;
            // set_fen rejects positions without exactly one king per side before this runs;
            // the guard only protects against a missing king.
            if self.count_of(self.side_to_move, PieceType::King) == 1 {
                let ksq = self.square_of(self.side_to_move, PieceType::King);
                self.st_mut().checkers_bb =
                    self.attackers_to(ksq) & self.pieces_of_color(!self.side_to_move);
            } else {
                self.st_mut().checkers_bb = 0; // No checkers if no king
            }
        }
        self.set_extra_bitboards();
        // Loop over all pieces and accumulate keys and material
        let mut b = self.pieces();
        while b != 0 {
            let s = bitboard::pop_lsb(&mut b);
            let pc = self.piece_on(s);
            // SAFETY: `pc` is the piece on an occupied square (< PIECE_NB), `s` a Square (< 64); co is color_of(pc) (0 or 1)
            // for the 2-element arrays; `st_mut()` follows the `st()` invariant.
            unsafe {
                self.st_mut().key ^= zobrist().psq[pc as usize][s as usize];
                if type_of(pc) == PieceType::Pawn {
                    self.st_mut().pawn_key ^= zobrist().psq[pc as usize][s as usize];
                } else {
                    let co = color_of(pc) as usize;
                    // SAFETY: co is color_of(pc) (0 or 1), arrays have 2 elements
                    *self.st_mut().non_pawn_key.get_unchecked_mut(co) ^=
                        zobrist().psq[pc as usize][s as usize];
                    if type_of(pc) != PieceType::King {
                        *self.st_mut().non_pawn_material.get_unchecked_mut(co) +=
                            PIECE_VALUE[pc as usize];
                        if type_of(pc) as u8 <= PieceType::Bishop as u8 {
                            self.st_mut().minor_piece_key ^= zobrist().psq[pc as usize][s as usize];
                        }
                    }
                }
            }
        }
        {
            if let Some(ep) = self.st().ep_square {
                self.st_mut().key ^= zobrist().enpassant[file_of(ep) as usize];
            }
            self.st_mut().key ^= zobrist().castling[self.st().castling_rights.bits() as usize];
            for pc in PIECES {
                for cnt in 0..(self.piece_count[pc as usize]) {
                    self.st_mut().material_key ^= zobrist().psq[pc as usize][(8 + cnt) as usize];
                }
            }
        }

        self.st_mut().piece_placement = pack_pieces(&self.board);
    }
}

// ---------------------------------------------------------------------------
// Helpers: piece <-> char mapping for FEN
// ---------------------------------------------------------------------------

const PIECE_TO_CHAR: &[u8] = b" PNBRQK  pnbrqk";

fn piece_to_char(pc: Piece) -> char {
    PIECE_TO_CHAR[pc as usize] as char
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_moves_restore_history_on_unwind_and_allow_state_reuse() {
        let mut pos = pos_from_fen(crate::uci::START_FEN);
        let original_fen = pos.fen();
        let original_key = pos.key();
        let e2e4 = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        let e7e5 = Move::from_to(Square::SQ_E7, Square::SQ_E5);
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: both moves are legal and guard nesting balances them.
            let mut outer = unsafe { ScopedMove::new(&mut pos, e2e4) };
            // SAFETY: e7e5 is legal after e2e4; the inner guard is dropped first.
            let inner = unsafe { ScopedMove::new(&mut outer, e7e5) };
            assert!(inner.pos_is_ok());
            panic!("nested move witness");
        }));
        assert!(failure.is_err());
        assert_eq!(pos.fen(), original_fen);
        assert_eq!(pos.key(), original_key);
        assert!(pos.pos_is_ok());
        for _ in 0..32 {
            // SAFETY: the guard restores the root before first is reused.
            let moved = unsafe { ScopedMove::new(&mut pos, e2e4) };
            assert!(moved.clone().pos_is_ok());
        }
        assert_eq!(pos.key(), original_key);
    }

    #[test]
    fn owned_moves_keep_history_and_reject_illegal_moves() {
        let mut pos = pos_from_fen(crate::uci::START_FEN);
        let start_key = pos.key();
        let parse = |pos: &Position, text| crate::uci::UciEngine::to_move(pos, text);

        // A move that is not legal here leaves the position untouched.
        let e7e5 = Move::from_to(Square::SQ_E7, Square::SQ_E5);
        assert!(!pos.do_move_owned(e7e5));
        assert!(!pos.do_move_owned(Move::none()));
        assert_eq!(pos.key(), start_key);

        // Knights out and back twice: the owned chain must see the repetitions.
        for _ in 0..2 {
            for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
                let mv = parse(&pos, text);
                assert!(pos.do_move_owned(mv), "{text}");
            }
        }
        assert_eq!(pos.key(), start_key);
        assert!(pos.is_draw(0));

        // Clones own a copy of the chain; the original can be reset independently.
        let copy = pos.clone();
        pos.set_fen(crate::uci::START_FEN, false).unwrap();
        assert!(copy.is_draw(0) && copy.pos_is_ok());
        assert!(!pos.is_draw(0));
    }

    fn pos_from_fen(fen: &str) -> Position {
        crate::bitboard::init();
        let mut p = Position::default();
        p.set_fen(fen, false).unwrap();
        p
    }

    #[test]
    fn fen_roundtrip_startpos() {
        crate::bitboard::init();
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut p = Position::default();
        p.set_fen(start, false).unwrap();
        assert_eq!(p.fen(), start);
        // Basic invariants
        assert_eq!(p.side_to_move(), Color::White);
        assert!(p.can_castle(CastlingRights::WHITE_OO));
        assert!(p.can_castle(CastlingRights::WHITE_OOO));
        assert!(p.can_castle(CastlingRights::BLACK_OO));
        assert!(p.can_castle(CastlingRights::BLACK_OOO));
    }

    #[test]
    fn enpassant_validation_kept_when_capturable() {
        // White pawn on d5, black pawn on e5, EP square e6 should be kept.
        // Both kings have to be on the board: `set_fen` validates the position
        // and rejects an incorrect number of kings.
        let fen = "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2";
        let p = pos_from_fen(fen);
        assert_eq!(p.ep_square(), Some(Square::SQ_E6));
        // Pseudo-legal EP capture d5xe6
        let mv = Move::make_en_passant(Square::SQ_D5, Square::SQ_E6);
        assert!(p.legal(mv));
        assert!(p.legal(mv));
    }

    #[test]
    fn clone_owns_history_and_survives_original() {
        crate::bitboard::init();
        let mut p = Position::default();
        p.set_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        )
        .unwrap();
        let moves = [
            Move::make_normal(Square::SQ_G1, Square::SQ_F3),
            Move::make_normal(Square::SQ_G8, Square::SQ_F6),
            Move::make_normal(Square::SQ_F3, Square::SQ_G1),
            Move::make_normal(Square::SQ_F6, Square::SQ_G8),
        ];
        for m in moves.iter() {
            // SAFETY: the test move is legal.
            unsafe { p.do_move(*m) };
        }
        let key = p.key();
        let fen = p.fen();
        let mut c = p.clone();
        assert_eq!(c.key(), key);
        assert!(c.has_repeated());
        assert_eq!(c.fen(), fen);
        // Mutate and drop the original: the clone is independent.
        // SAFETY: the test move is legal.
        unsafe { p.undo_move(moves[3]) };
        // SAFETY: the test move is legal.
        unsafe { p.undo_move(moves[2]) };
        drop(p);
        assert_eq!(c.key(), key);
        assert!(c.has_repeated());
        // do_move/undo_move on the clone still work, with the repetition seen through the copy.
        let m = Move::make_normal(Square::SQ_G1, Square::SQ_F3);
        // SAFETY: the test move is legal.
        unsafe { c.do_move(m) };
        assert!(c.has_repeated());
        // SAFETY: the test move is legal.
        unsafe { c.undo_move(m) };
        assert_eq!(c.key(), key);
        let c2 = c.clone();
        assert_eq!(c2.key(), key);
        let d = pos_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").clone();
        assert_eq!(d.fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    }

    #[test]
    fn pawn_push_and_check_flags() {
        // Simple pawn push e2e4 from startpos
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(
            p.piece_on(Square::SQ_E2),
            make_piece(Color::White, PieceType::Pawn)
        );
        assert_eq!(p.count_of(Color::White, PieceType::Pawn), 8);
        assert_eq!(p.square_of(Color::White, PieceType::King), Square::SQ_E1);
        assert!(p.empty(Square::SQ_E3));
        assert!(p.empty(Square::SQ_E4));
        // Verify bitboards align with piece_on
        let e2_bb = 1u64 << (Square::SQ_E2 as u8);
        let e4_bb = 1u64 << (Square::SQ_E4 as u8);
        assert_ne!(p.pieces_of_color(Color::White) & e2_bb, 0);
        assert_eq!(p.pieces_of_color(Color::White) & e4_bb, 0);
        assert_eq!(p.pieces() & e4_bb, 0);
        assert_eq!(bitboard::popcount(p.pieces_of_color(Color::Black)), 16);
        let at = p.attackers_to(Square::SQ_E1) & p.pieces_of_color(Color::Black);
        if at != 0 {
            let mut bb = at;
            let mut entries = vec![];
            while bb != 0 {
                let s = bitboard::pop_lsb(&mut bb);
                let pc = p.piece_on(s);
                entries.push((s as u8, piece_to_char(pc)));
            }
            panic!("unexpected attackers to E1: {:?}", entries);
        }
        assert_eq!(p.checkers(), 0);
        let m = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        assert!(p.legal(m));
        assert!(p.legal(m));
        assert!(!p.gives_check(m));
    }

    #[test]
    fn pos_is_ok_ep_and_kings() {
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(p.pos_is_ok());
        // Valid EP relative rank for side to move (white -> rank 6)
        let p_ok = pos_from_fen("4k3/8/8/8/3Pp3/8/8/4K3 w - e6 0 2");
        assert!(p_ok.pos_is_ok());
        // Invalid EP rank (white with e3) is rejected
        let mut p_bad = Position::default();
        assert!(
            p_bad
                .set_fen("4k3/8/8/8/3Pp3/8/8/4K3 w - e3 0 2", false)
                .is_err()
        );
    }

    #[test]
    fn flip_startpos() {
        // Test basic flip on starting position
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(p.flip().is_none());
        // Flipping swaps the sides to move and the castling rights; the FEN writer emits them
        // in canonical KQkq order.
        assert_eq!(
            p.fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        );
    }

    #[test]
    fn test_flip_comprehensive() {
        // Test 1: Flip starting position
        let mut p1 = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let initial_stm = p1.side_to_move();
        p1.flip();
        assert_eq!(p1.side_to_move(), !initial_stm, "Side to move should flip");
        // After flipping, the board is mirrored and colors are swapped via case transformation
        // This matches the C++ implementation exactly
        assert_eq!(
            p1.fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        );

        // Test 2: Flip with pieces in various positions
        let mut p2 =
            pos_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5");
        p2.flip();
        // After flipping, ranks are reversed and colors are swapped via case transformation
        let flipped_fen = p2.fen();
        // The C++ flip reverses ranks then applies case transformation to swap colors
        assert!(
            flipped_fen
                .starts_with("rnbqk2r/pppp1ppp/5n2/2b1p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq"),
            "FEN after flip was: {}",
            flipped_fen
        );

        // Test 3: Flip with en passant square
        let mut p3 = pos_from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        p3.flip();
        let flipped_fen3 = p3.fen();
        // En passant square f6 should become f3 after flip
        assert!(
            flipped_fen3.contains("f3"),
            "En passant should flip from f6 to f3, got: {}",
            flipped_fen3
        );

        // Test 4: Flip with no castling rights
        let mut p4 = pos_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 50 125");
        p4.flip();
        assert_eq!(p4.fen(), "4k3/8/8/8/8/8/8/4K3 b - - 50 125");

        // Test 5: Two flips restore the original FEN, including side to move
        let mut p5 =
            pos_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let original_fen = p5.fen();
        p5.flip();
        p5.flip();
        let double_flipped = p5.fen();
        assert_eq!(
            double_flipped, original_fen,
            "Double flip should restore original position"
        );
    }

    #[test]
    fn test_flip_castling_rights() {
        // Test flipping preserves castling rights correctly
        let mut p = pos_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        p.flip();
        // After flip, castling rights should be preserved but for opposite color
        assert!(
            p.can_castle(CastlingRights::WHITE_OO),
            "Should still have white O-O"
        );
        assert!(
            p.can_castle(CastlingRights::WHITE_OOO),
            "Should still have white O-O-O"
        );
        assert!(
            p.can_castle(CastlingRights::BLACK_OO),
            "Should still have black O-O"
        );
        assert!(
            p.can_castle(CastlingRights::BLACK_OOO),
            "Should still have black O-O-O"
        );

        // Test partial castling rights
        let mut p2 = pos_from_fen("r3k3/8/8/8/8/8/8/4K2R w K - 0 1");
        p2.flip();
        let fen = p2.fen();
        assert!(
            fen.contains(" k "),
            "Should preserve kingside castling for black after flip: {}",
            fen
        );
    }

    #[test]
    fn test_flip_without_en_passant() {
        let mut p = pos_from_fen("4k3/8/8/8/3Pp3/8/8/4K3 w - - 0 2");
        p.flip();
        let flipped_fen = p.fen();
        assert!(flipped_fen.contains("3pP"), "Basic flip should work");
    }

    #[test]
    fn test_flip_preserves_validity() {
        // Ensure flipped positions remain valid
        let test_positions = vec![
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];

        for fen in test_positions {
            let mut p = pos_from_fen(fen);
            assert!(p.pos_is_ok(), "Original position should be valid: {}", fen);
            p.flip();
            assert!(
                p.pos_is_ok(),
                "Flipped position should be valid. Original: {}",
                fen
            );
        }
    }

    #[test]
    fn repetition_cycle_knight_moves() {
        crate::bitboard::init();
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        // g1f3, g8f6, f3g1, f6g8
        let m1 = Move::from_to(Square::SQ_G1, Square::SQ_F3);
        let m2 = Move::from_to(Square::SQ_G8, Square::SQ_F6);
        let m3 = Move::from_to(Square::SQ_F3, Square::SQ_G1);
        let m4 = Move::from_to(Square::SQ_F6, Square::SQ_G8);
        assert!(p.legal(m1));
        // SAFETY: the test move is legal.
        unsafe { p.do_move(m1) };
        assert!(p.legal(m2));
        // SAFETY: the test move is legal.
        unsafe { p.do_move(m2) };
        assert!(p.legal(m3));
        // SAFETY: the test move is legal.
        unsafe { p.do_move(m3) };
        assert!(p.legal(m4));
        // SAFETY: the test move is legal.
        unsafe { p.do_move(m4) };
        // Now we should have repeated the initial position
        assert!(p.has_repeated());
        assert!(p.is_repetition(10));
    }

    #[test]
    fn draw_by_50_move_rule() {
        let p = pos_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 1");
        // Confirm FEN roundtrip retains halfmove count and fullmove
        let f = p.fen();
        assert!(f.ends_with("100 1"), "fen was: {}", f);
        assert_eq!(p.rule50_count(), 100);
        assert_eq!(p.checkers(), 0);
        assert!(p.is_draw(1));
    }

    #[test]
    fn startpos_piece_layout() {
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        // White back rank
        assert_eq!(
            p.piece_on(Square::SQ_A1),
            make_piece(Color::White, PieceType::Rook)
        );
        assert_eq!(
            p.piece_on(Square::SQ_B1),
            make_piece(Color::White, PieceType::Knight)
        );
        assert_eq!(
            p.piece_on(Square::SQ_C1),
            make_piece(Color::White, PieceType::Bishop)
        );
        assert_eq!(
            p.piece_on(Square::SQ_D1),
            make_piece(Color::White, PieceType::Queen)
        );
        assert_eq!(
            p.piece_on(Square::SQ_E1),
            make_piece(Color::White, PieceType::King)
        );
        assert_eq!(
            p.piece_on(Square::SQ_F1),
            make_piece(Color::White, PieceType::Bishop)
        );
        assert_eq!(
            p.piece_on(Square::SQ_G1),
            make_piece(Color::White, PieceType::Knight)
        );
        assert_eq!(
            p.piece_on(Square::SQ_H1),
            make_piece(Color::White, PieceType::Rook)
        );
        // White pawns
        for f in 0..8u8 {
            let sq = make_square(File::from(f), Rank::R2);
            assert_eq!(p.piece_on(sq), make_piece(Color::White, PieceType::Pawn));
        }
        // Black back rank
        assert_eq!(
            p.piece_on(Square::SQ_A8),
            make_piece(Color::Black, PieceType::Rook)
        );
        assert_eq!(
            p.piece_on(Square::SQ_B8),
            make_piece(Color::Black, PieceType::Knight)
        );
        assert_eq!(
            p.piece_on(Square::SQ_C8),
            make_piece(Color::Black, PieceType::Bishop)
        );
        assert_eq!(
            p.piece_on(Square::SQ_D8),
            make_piece(Color::Black, PieceType::Queen)
        );
        assert_eq!(
            p.piece_on(Square::SQ_E8),
            make_piece(Color::Black, PieceType::King)
        );
        assert_eq!(
            p.piece_on(Square::SQ_F8),
            make_piece(Color::Black, PieceType::Bishop)
        );
        assert_eq!(
            p.piece_on(Square::SQ_G8),
            make_piece(Color::Black, PieceType::Knight)
        );
        assert_eq!(
            p.piece_on(Square::SQ_H8),
            make_piece(Color::Black, PieceType::Rook)
        );
        // Black pawns
        for f in 0..8u8 {
            let sq = make_square(File::from(f), Rank::R7);
            assert_eq!(p.piece_on(sq), make_piece(Color::Black, PieceType::Pawn));
        }
    }

    /// Cross-checks `attackers_to()` against an independent decomposition (per-piece-type
    /// lookups plus a naive ray scan).
    #[test]
    fn attackers_to_e1_matches_manual_decomposition_at_startpos() {
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let s = Square::SQ_E1;
        let occ = p.pieces();
        let pawns_w = p.pieces_color_types(Color::White, &[PieceType::Pawn]);
        let pawns_b = p.pieces_color_types(Color::Black, &[PieceType::Pawn]);
        let pawn_w = bitboard::pawn_attacks(Color::Black, 1u64 << (s as u8)) & pawns_w;
        let pawn_b = bitboard::pawn_attacks(Color::White, 1u64 << (s as u8)) & pawns_b;
        let knights =
            bitboard::attacks_bb(PieceType::Knight, s, 0) & p.pieces_of_types(&[PieceType::Knight]);
        let kings =
            bitboard::attacks_bb(PieceType::King, s, 0) & p.pieces_of_types(&[PieceType::King]);
        let bishops = {
            // simple scan like in attackers_to()
            let mut bb = 0u64;
            for &d in &[NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST] {
                let mut cur = s;
                loop {
                    let (file, rank) = (file_of(cur) as i8, rank_of(cur) as i8);
                    let next = match d {
                        x if x == NORTH_EAST && file < 7 && rank < 7 => cur.offset(NORTH_EAST),
                        x if x == SOUTH_EAST && file < 7 && rank > 0 => cur.offset(SOUTH_EAST),
                        x if x == SOUTH_WEST && file > 0 && rank > 0 => cur.offset(SOUTH_WEST),
                        x if x == NORTH_WEST && file > 0 && rank < 7 => cur.offset(NORTH_WEST),
                        _ => None,
                    };
                    if next.is_none() {
                        break;
                    }
                    let t = next.unwrap();
                    let tobb = 1u64 << (t as u8);
                    if (occ & tobb) != 0 {
                        let pc = p.piece_on(t);
                        if matches!(type_of(pc), PieceType::Bishop | PieceType::Queen) {
                            bb |= tobb;
                        }
                        break;
                    }
                    cur = t;
                }
            }
            bb
        };
        let rooks = {
            let mut bb = 0u64;
            for &d in &[NORTH, SOUTH, EAST, WEST] {
                let mut cur = s;
                loop {
                    let (file, rank) = (file_of(cur) as i8, rank_of(cur) as i8);
                    let next = match d {
                        x if x == NORTH && rank < 7 => cur.offset(NORTH),
                        x if x == SOUTH && rank > 0 => cur.offset(SOUTH),
                        x if x == EAST && file < 7 => cur.offset(EAST),
                        x if x == WEST && file > 0 => cur.offset(WEST),
                        _ => None,
                    };
                    if next.is_none() {
                        break;
                    }
                    let t = next.unwrap();
                    let tobb = 1u64 << (t as u8);
                    if (occ & tobb) != 0 {
                        let pc = p.piece_on(t);
                        if matches!(type_of(pc), PieceType::Rook | PieceType::Queen) {
                            bb |= tobb;
                        }
                        break;
                    }
                    cur = t;
                }
            }
            bb
        };
        // All contributors must be white-only in startpos for E1
        let all = pawn_w | pawn_b | knights | kings | bishops | rooks;
        assert_eq!(
            all & p.pieces_of_color(Color::Black),
            0,
            "black attackers present at startpos to E1"
        );

        // The independent decomposition must agree with the real accessor.
        assert_eq!(
            all,
            p.attackers_to(s),
            "manual decomposition disagrees with attackers_to() at E1"
        );
    }

    #[test]
    fn castling_rights_and_impeded_path() {
        // From start position, kingside castling should have path clear and not in check
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(p.can_castle(CastlingRights::WHITE_OO));
        // Castling is encoded as the king capturing its own rook.
        let m = Move::make_castling(Square::SQ_E1, Square::SQ_H1);
        // Clear path, no attacks: castling is legal.
        let clear = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/R3K2R w KQkq - 0 1");
        assert!(clear.legal(m));
        // A black bishop on a6 attacks f1 along the open a6-f1 diagonal (e2 is empty),
        // so castling through f1 is illegal.
        let q = pos_from_fen("rnbqk1nr/pppppppp/b7/8/8/8/PPPP1PPP/R3K2R w KQkq - 0 1");
        // The cached enemy threats reject castling through f1.
        assert!(!q.legal(m));
    }

    // =========================================================================
    // COMPREHENSIVE UNIT TESTS
    // =========================================================================

    #[test]
    fn test_move_legality() {
        // Test legal move validation for various piece types
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Pawn moves
        let e2e4 = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        assert!(p.legal(e2e4), "e2-e4 should be legal");

        let e2e5 = Move::from_to(Square::SQ_E2, Square::SQ_E5);
        assert!(!p.legal(e2e5), "e2-e5 should not be legal");

        // Knight moves
        let b1c3 = Move::from_to(Square::SQ_B1, Square::SQ_C3);
        assert!(p.legal(b1c3), "Nb1-c3 should be legal");

        let b1d2 = Move::from_to(Square::SQ_B1, Square::SQ_D2);
        assert!(!p.legal(b1d2), "Nb1-d2 blocked by pawn");

        // After e2-e4
        // SAFETY: the test move is legal.
        unsafe { p.do_move(e2e4) };

        // Black pawn moves
        let e7e5 = Move::from_to(Square::SQ_E7, Square::SQ_E5);
        assert!(p.legal(e7e5), "e7-e5 should be legal for black");
    }

    #[test]
    fn test_en_passant_capture() {
        // Set up position where en passant is possible
        // Black just moved f7-f5, so white can capture en passant
        let mut p = pos_from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");

        // Verify EP square is set correctly
        assert_eq!(p.ep_square(), Some(Square::SQ_F6), "EP square should be f6");

        // White can capture f5 en passant by moving e5xf6
        let ep_move = Move::make_en_passant(Square::SQ_E5, Square::SQ_F6);
        assert!(p.legal(ep_move), "En passant should be legal");

        // Make the en passant capture
        let original_black_pawns = p.count_of(Color::Black, PieceType::Pawn);
        // SAFETY: the test move is legal.
        unsafe { p.do_move(ep_move) };

        // Verify pawn was captured
        assert_eq!(
            p.count_of(Color::Black, PieceType::Pawn),
            original_black_pawns - 1
        );
        assert_eq!(
            p.piece_on(Square::SQ_F5),
            Piece::NoPiece,
            "f5 should be empty"
        );
        assert_eq!(
            p.piece_on(Square::SQ_F6),
            Piece::W_Pawn,
            "White pawn should be on f6"
        );
    }

    #[test]
    fn test_promotion_moves() {
        // Position with white pawn on 7th rank that can promote
        let mut p = pos_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");

        // Test all promotion types
        let promotions = [
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
        ];

        for prom_type in promotions {
            let prom_move = Move::make_promotion(Square::SQ_A7, Square::SQ_A8, prom_type);
            assert!(
                p.legal(prom_move),
                "Promotion to {:?} should be legal",
                prom_type
            );
            assert!(
                p.legal(prom_move),
                "Promotion to {:?} should be legal",
                prom_type
            );
        }

        // Actually make a queen promotion
        let queen_prom = Move::make_promotion(Square::SQ_A7, Square::SQ_A8, PieceType::Queen);
        // SAFETY: the test move is legal.
        unsafe { p.do_move(queen_prom) };

        assert_eq!(
            p.piece_on(Square::SQ_A8),
            Piece::W_Queen,
            "Should have white queen on a8"
        );
        assert_eq!(
            p.piece_on(Square::SQ_A7),
            Piece::NoPiece,
            "a7 should be empty"
        );
    }

    #[test]
    fn test_castling_all_types() {
        // Test white kingside castling first
        let mut p = pos_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");

        // Verify initial positions
        assert_eq!(
            p.piece_on(Square::SQ_E1),
            Piece::W_King,
            "White king should be on e1"
        );
        assert_eq!(
            p.piece_on(Square::SQ_H1),
            Piece::W_Rook,
            "White rook should be on h1"
        );

        let wk_castle = Move::make_castling(Square::SQ_E1, Square::SQ_H1);
        assert!(p.legal(wk_castle), "White O-O should be legal");

        // SAFETY: the test move is legal.
        unsafe { p.do_move(wk_castle) };
        assert_eq!(
            p.piece_on(Square::SQ_G1),
            Piece::W_King,
            "King should be on g1 after O-O"
        );
        assert_eq!(
            p.piece_on(Square::SQ_F1),
            Piece::W_Rook,
            "Rook should be on f1 after O-O"
        );

        // Test black queenside castling separately with fresh position
        let mut p2 = pos_from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");

        assert_eq!(
            p2.piece_on(Square::SQ_E8),
            Piece::B_King,
            "Black king should be on e8"
        );
        assert_eq!(
            p2.piece_on(Square::SQ_A8),
            Piece::B_Rook,
            "Black rook should be on a8"
        );

        let bq_castle = Move::make_castling(Square::SQ_E8, Square::SQ_A8);
        assert!(p2.legal(bq_castle), "Black O-O-O should be legal");

        // SAFETY: the test move is legal.
        unsafe { p2.do_move(bq_castle) };
        assert_eq!(
            p2.piece_on(Square::SQ_C8),
            Piece::B_King,
            "King should be on c8 after O-O-O"
        );
        assert_eq!(
            p2.piece_on(Square::SQ_D8),
            Piece::B_Rook,
            "Rook should be on d8 after O-O-O"
        );
        assert_eq!(
            p2.piece_on(Square::SQ_E8),
            Piece::NoPiece,
            "e8 should be empty"
        );
        assert_eq!(
            p2.piece_on(Square::SQ_A8),
            Piece::NoPiece,
            "a8 should be empty"
        );
    }

    #[test]
    fn test_zobrist_key_consistency() {
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let initial_key = p.key();

        // Make and unmake a move
        let m = Move::from_to(Square::SQ_E2, Square::SQ_E4);
        // SAFETY: the test move is legal.
        unsafe { p.do_move(m) };
        let key_after_move = p.key();

        assert_ne!(initial_key, key_after_move, "Key should change after move");

        // SAFETY: the test move is legal.
        unsafe { p.undo_move(m) };
        let key_after_undo = p.key();

        assert_eq!(
            initial_key, key_after_undo,
            "Key should be restored after undo"
        );
    }

    #[test]
    fn test_material_tracking() {
        // Set up a capture position
        let mut p2 = pos_from_fen("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2");

        // Capture the knight with pawn
        let capture = Move::from_to(Square::SQ_E4, Square::SQ_F6);
        let black_material_before = p2.non_pawn_material_of(Color::Black);
        // SAFETY: the test move is legal.
        unsafe { p2.do_move(capture) };
        let black_material_after = p2.non_pawn_material_of(Color::Black);

        assert_eq!(
            black_material_after,
            black_material_before - KNIGHT_VALUE,
            "Black material should decrease by knight value"
        );
    }

    #[test]
    fn test_see_simple_exchanges() {
        // Position with simple pawn takes pawn
        let p = pos_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
        let pawn_takes = Move::from_to(Square::SQ_E4, Square::SQ_D5);
        assert!(p.see_ge(pawn_takes, 0), "PxP should win material");

        // Position where the queen takes an undefended pawn
        let p2 = pos_from_fen("rnbqkb1r/pppp1ppp/8/4p3/8/3Q4/PPPPPPPP/RNB1KBNR w KQkq - 0 4");
        let queen_takes_free = Move::from_to(Square::SQ_D3, Square::SQ_E5);
        assert!(
            p2.see_ge(queen_takes_free, 0),
            "QxP (undefended) should win material"
        );

        // Position where the knight takes an undefended pawn
        let p3 = pos_from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 3");
        let knight_takes = Move::from_to(Square::SQ_F3, Square::SQ_E5);
        assert!(p3.see_ge(knight_takes, 0), "NxP should be at least equal");
    }

    #[test]
    fn test_gives_check_negative() {
        let p2 = pos_from_fen("rnb1kbnr/pppp1ppp/8/4p3/4P3/3Q4/PPPP1PPP/RNB1KBNR w KQkq - 0 4");
        let check_move = Move::from_to(Square::SQ_D3, Square::SQ_H7);
        // Qd3-h7 does not give check here.
        assert!(
            !p2.gives_check(check_move),
            "Qd3-h7 doesn't give check here"
        );
    }

    #[test]
    fn test_null_move() {
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let initial_stm = p.side_to_move();
        let initial_ply = p.game_ply();

        // SAFETY: the test move is legal.
        unsafe { p.do_null_move() };

        assert_eq!(p.side_to_move(), !initial_stm, "Side to move should flip");
        assert_eq!(p.game_ply(), initial_ply + 1, "Game ply should increment");
        assert_eq!(p.ep_square(), None, "EP square should be cleared");

        // SAFETY: the test move is legal.
        unsafe { p.undo_null_move() };
        assert_eq!(
            p.side_to_move(),
            initial_stm,
            "Side to move should be restored"
        );
        assert_eq!(p.game_ply(), initial_ply, "Game ply should be restored");
    }

    #[test]
    fn test_repetition_detection() {
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Make moves that repeat position: Nf3, Nf6, Ng1, Ng8
        let moves = [
            Move::from_to(Square::SQ_G1, Square::SQ_F3),
            Move::from_to(Square::SQ_G8, Square::SQ_F6),
            Move::from_to(Square::SQ_F3, Square::SQ_G1),
            Move::from_to(Square::SQ_F6, Square::SQ_G8),
        ];

        for &m in moves.iter() {
            // SAFETY: the test move is legal.
            unsafe { p.do_move(m) };
        }

        assert!(p.has_repeated(), "Position should be repeated");
        assert!(
            p.is_repetition(10),
            "Should detect repetition within 10 plies"
        );
    }

    #[test]
    fn test_pos_is_ok_validation() {
        // Valid position should pass
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(p.pos_is_ok(), "Starting position should be valid");

        // Position with proper king counts
        let p2 = pos_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!(p2.pos_is_ok(), "Bare kings position should be valid");

        // Test EP square validation
        let p3 = pos_from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        assert!(p3.pos_is_ok(), "Position with valid EP square should be ok");
    }

    #[test]
    fn test_hash_key_operations() {
        // Test various hash key functions
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Test all key types return non-zero values for starting position
        assert_ne!(p.key(), 0, "Position key should not be zero");
        assert_ne!(p.pawn_key(), 0, "Pawn key should not be zero");
        assert_ne!(p.material_key(), 0, "Material key should not be zero");

        // Test that different positions have different keys
        let p2 = pos_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        assert_ne!(
            p.key(),
            p2.key(),
            "Different positions should have different keys"
        );

        // Test key consistency - same position should have same key
        let p3 = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(p.key(), p3.key(), "Same positions should have same keys");
        assert_eq!(
            p.pawn_key(),
            p3.pawn_key(),
            "Same positions should have same pawn keys"
        );
        assert_eq!(
            p.material_key(),
            p3.material_key(),
            "Same positions should have same material keys"
        );
    }

    #[test]
    fn test_piece_counting() {
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Test piece counts for starting position
        assert_eq!(p.count_of(Color::White, PieceType::Pawn), 8);
        assert_eq!(p.count_of(Color::Black, PieceType::Pawn), 8);
        assert_eq!(p.count_of(Color::White, PieceType::Knight), 2);
        assert_eq!(p.count_of(Color::Black, PieceType::Knight), 2);
        assert_eq!(p.count_of(Color::White, PieceType::Bishop), 2);
        assert_eq!(p.count_of(Color::Black, PieceType::Bishop), 2);
        assert_eq!(p.count_of(Color::White, PieceType::Rook), 2);
        assert_eq!(p.count_of(Color::Black, PieceType::Rook), 2);
        assert_eq!(p.count_of(Color::White, PieceType::Queen), 1);
        assert_eq!(p.count_of(Color::Black, PieceType::Queen), 1);
        assert_eq!(p.count_of(Color::White, PieceType::King), 1);
        assert_eq!(p.count_of(Color::Black, PieceType::King), 1);

        // Test total counts
        assert_eq!(p.count(PieceType::Pawn), 16);
        assert_eq!(p.count(PieceType::Knight), 4);
        assert_eq!(p.count(PieceType::Bishop), 4);
        assert_eq!(p.count(PieceType::Rook), 4);
        assert_eq!(p.count(PieceType::Queen), 2);
        assert_eq!(p.count(PieceType::King), 2);
    }

    #[test]
    fn test_blockers_and_pinners() {
        // White bishop on e2 is pinned to the king on e1 by the black rook on e8
        let p = pos_from_fen("4r1k1/8/8/8/8/8/4B3/4K3 w - - 0 1");
        let e2 = bitboard::square_bb_pub(Square::SQ_E2);
        let e8 = bitboard::square_bb_pub(Square::SQ_E8);

        assert_eq!(p.blockers_for_king(Color::White), e2);
        assert_eq!(p.blockers_for_king(Color::Black), 0);
        assert_eq!(p.pinners(Color::Black), e8);
        assert_eq!(p.pinners(Color::White), 0);
    }

    #[test]
    fn test_attacked_squares() {
        let p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Pawns attack the whole of the third (sixth) rank from the starting position
        assert_eq!(
            p.attacks_by(Color::White, PieceType::Pawn),
            bitboard::RANK_3_BB
        );
        assert_eq!(
            p.attacks_by(Color::Black, PieceType::Pawn),
            bitboard::RANK_6_BB
        );

        // Knights on b1 and g1 attack a3, c3, d2, e2, f3, h3
        let expected_knights = [
            Square::SQ_A3,
            Square::SQ_C3,
            Square::SQ_D2,
            Square::SQ_E2,
            Square::SQ_F3,
            Square::SQ_H3,
        ]
        .iter()
        .fold(0, |bb, &s| bb | bitboard::square_bb_pub(s));
        assert_eq!(
            p.attacks_by(Color::White, PieceType::Knight),
            expected_knights
        );
    }

    #[test]
    fn test_move_piece_operations() {
        let mut p = pos_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");

        // Add a piece
        p.put_piece(Piece::W_Queen, Square::SQ_D4);
        assert_eq!(p.piece_on(Square::SQ_D4), Piece::W_Queen);
        assert_eq!(p.count_of(Color::White, PieceType::Queen), 1);

        // Move the piece
        p.move_piece(Square::SQ_D4, Square::SQ_E5);
        assert_eq!(p.piece_on(Square::SQ_D4), Piece::NoPiece);
        assert_eq!(p.piece_on(Square::SQ_E5), Piece::W_Queen);

        // Remove the piece
        p.remove_piece(Square::SQ_E5);
        assert_eq!(p.piece_on(Square::SQ_E5), Piece::NoPiece);
        assert_eq!(p.count_of(Color::White, PieceType::Queen), 0);
    }

    #[test]
    fn test_check_detection_comprehensive() {
        // Quiet position: the queen on d2 attacks neither king, so with White to
        // move there are no checkers. (A queen on e2 would attack the black king
        // down the open e-file, which is not a legal position with White to move
        // -- `set_fen` rejects it as "King can be captured".)
        let p = pos_from_fen("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1");
        let checkers = p.checkers();
        assert_eq!(
            checkers, 0,
            "White to move and not in check, so no checkers"
        );

        // After white moves, black will be in check
        let p2 = pos_from_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let checkers2 = p2.checkers();
        assert_ne!(checkers2, 0, "Black king should be in check from queen");

        // Double check position - Queen on e2 attacks e8 vertically, Bishop on f7 attacks e8 diagonally
        let p3 = pos_from_fen("4k3/5B2/8/8/8/8/4Q3/4K3 b - - 0 1");
        let checkers3 = p3.checkers();
        assert_eq!(
            bitboard::popcount(checkers3),
            2,
            "Should be in double check"
        );
    }

    #[test]
    fn test_upcoming_repetition() {
        // Test the upcoming_repetition function
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(
            !p.upcoming_repetition(10),
            "Starting position shouldn't have upcoming repetition"
        );

        // Make moves that could lead to repetition

        // Nf3, Nf6, Ng1 (one move away from repetition)
        // SAFETY: the test move is legal.
        unsafe { p.do_move(Move::from_to(Square::SQ_G1, Square::SQ_F3)) };
        // SAFETY: the test move is legal.
        unsafe { p.do_move(Move::from_to(Square::SQ_G8, Square::SQ_F6)) };
        // SAFETY: the test move is legal.
        unsafe { p.do_move(Move::from_to(Square::SQ_F3, Square::SQ_G1)) };

        // Check if Ng8 would cause repetition
        assert!(
            p.upcoming_repetition(10),
            "Should detect upcoming repetition"
        );
    }

    #[test]
    fn const_pseudo_attacks_match_runtime_tables() {
        crate::bitboard::init();
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            for s in 0..SQUARE_NB {
                assert_eq!(
                    pseudo_attacks(pt, s),
                    bitboard::attacks_bb(pt, Square::from(s as u8), 0),
                    "{pt:?} on square {s}"
                );
            }
        }
    }

    #[test]
    fn repetition_table_finds_every_reversible_move() {
        crate::bitboard::init();
        let t = &REPETITION_TABLE;
        let mut found = 0;
        for pc in PIECES {
            if type_of(pc) == PieceType::Pawn {
                continue;
            }
            for s1 in 0..SQUARE_NB as u8 {
                let (sq1, attacks) = (
                    Square::from(s1),
                    bitboard::attacks_bb(type_of(pc), Square::from(s1), 0),
                );
                for s2 in s1 + 1..SQUARE_NB as u8 {
                    if attacks & (1u64 << s2) == 0 {
                        continue;
                    }
                    let key = zobrist().psq[pc as usize][s1 as usize]
                        ^ zobrist().psq[pc as usize][s2 as usize]
                        ^ ZOBRIST_SIDE;
                    // Probe exactly as `upcoming_repetition` does.
                    let (mut j, mut distance) = (repetition_hash(key), 1);
                    while t.distances[j] >= distance && t.keys[j] != key {
                        distance += 1;
                        j = (j + 1) & REPETITION_TABLE_MASK;
                    }
                    assert!(t.distances[j] >= distance, "{pc:?} {s1}-{s2} not found");
                    assert_eq!(t.moves[j], Move::from_to(sq1, Square::from(s2)));
                    found += 1;
                }
            }
        }
        assert_eq!(found, 3668);
        // A key that is not in the table stops at an empty or richer slot.
        let (mut j, mut distance) = (repetition_hash(0x1234_5678_9abc_def0), 1);
        while t.distances[j] >= distance {
            assert_ne!(t.keys[j], 0x1234_5678_9abc_def0);
            distance += 1;
            j = (j + 1) & REPETITION_TABLE_MASK;
        }
    }

    #[test]
    fn packed_piece_placement_tracks_the_board() {
        let p =
            pos_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let packed = pack_pieces(&p.board);
        for s in 0..SQUARE_NB {
            let nibble = (packed[s & 31] >> ((s & 0x20) >> 3)) & 0xF;
            assert_eq!(nibble, p.board[s] as u8, "square {s}");
        }
        assert_eq!(p.st().piece_placement, packed);

        // swap_nibble moves a piece exactly as the board would.
        for (a, b) in [
            (Square::SQ_E1, Square::SQ_E8),
            (Square::SQ_A1, Square::SQ_D4),
            (Square::SQ_B2, Square::SQ_C2),
        ] {
            let mut board = p.board;
            board.swap(a as usize, b as usize);
            let mut swapped = packed;
            swap_nibble(&mut swapped, a, b);
            assert_eq!(swapped, pack_pieces(&board), "{a:?}-{b:?}");
        }
    }

    /// A closed king tour of `len` distinct squares within files b-g of `ranks`.
    fn king_tour(ranks: [u8; 3], len: usize) -> Vec<Square> {
        let squares: Vec<(i32, i32)> = (1..7)
            .flat_map(|f| ranks.iter().map(move |&r| (f, i32::from(r))))
            .collect();
        fn dfs(path: &mut Vec<(i32, i32)>, squares: &[(i32, i32)], len: usize) -> bool {
            let last = *path.last().unwrap();
            let adjacent = |a: (i32, i32), b: (i32, i32)| {
                a != b && (a.0 - b.0).abs() <= 1 && (a.1 - b.1).abs() <= 1
            };
            if path.len() == len {
                return adjacent(last, path[0]);
            }
            for &next in squares {
                if adjacent(last, next) && !path.contains(&next) {
                    path.push(next);
                    if dfs(path, squares, len) {
                        return true;
                    }
                    path.pop();
                }
            }
            false
        }
        let mut path = vec![squares[0]];
        assert!(dfs(&mut path, &squares, len));
        path.iter()
            .map(|&(f, r)| Square::new((r * 8 + f) as u8).unwrap())
            .collect()
    }

    /// Plays `plies` king moves, white touring `lw` squares and black `lb` (period
    /// `2 * lw * lb` plies, all reversible), and returns the moves made.
    fn play_king_tours(p: &mut Position, lw: usize, lb: usize, plies: usize) -> Vec<Move> {
        let (w, b) = (king_tour([0, 1, 2], lw), king_tour([5, 6, 7], lb));
        let mut moves = Vec::new();
        for i in 0..plies {
            let (tour, j) = if i % 2 == 0 { (&w, i / 2) } else { (&b, i / 2) };
            let m = Move::from_to(tour[j % tour.len()], tour[(j + 1) % tour.len()]);
            assert!(p.legal(m), "tour move {i} illegal");
            // SAFETY: checked legal just above.
            unsafe { p.do_move(m) };
            moves.push(m);
        }
        moves
    }

    /// Histories longer than the `MAX_HIST` ring wrap as in C++: a repetition inside the ring is
    /// found at its distance, and the repetition walks stop at `MAX_HIST - 1` plies back, so the
    /// wrapped slot `MAX_HIST` plies back (the current state) is never compared. Undoing fewer
    /// than `MAX_HIST` plies across the wrap restores the position.
    #[test]
    fn state_ring_wraps_like_cpp() {
        let start = |lw: usize, lb: usize| {
            let (w, b) = (king_tour([0, 1, 2], lw)[0], king_tour([5, 6, 7], lb)[0]);
            let mut board = [[' '; 8]; 8];
            board[usize::from(w as u8 / 8)][usize::from(w as u8 % 8)] = 'K';
            board[usize::from(b as u8 / 8)][usize::from(b as u8 % 8)] = 'k';
            for (r, f, c) in [(3, 0, 'P'), (3, 7, 'P'), (4, 0, 'p'), (4, 7, 'p')] {
                board[r][f] = c;
            }
            let mut fen = String::new();
            for r in (0..8).rev() {
                let mut empty = 0;
                for f in 0..8 {
                    if board[r][f] == ' ' {
                        empty += 1;
                    } else {
                        if empty > 0 {
                            fen.push(char::from(b'0' + empty));
                        }
                        empty = 0;
                        fen.push(board[r][f]);
                    }
                }
                if empty > 0 {
                    fen.push(char::from(b'0' + empty));
                }
                if r > 0 {
                    fen.push('/');
                }
            }
            pos_from_fen(&format!("{fen} w - - 0 1"))
        };

        // Period 480 < MAX_HIST: the repetition is found 480 plies back.
        // `key()` also encodes the rule-50 count; compare the raw position key.
        let mut p = start(15, 16);
        let key = p.st().key;
        play_king_tours(&mut p, 15, 16, 479);
        assert_eq!(p.st().repetition, 0);
        // The next white king move returns to the start, 480 plies back: a draw only if that
        // position lies inside the search tree (ply > 479), as the start is not itself repeated.
        assert!(p.upcoming_repetition(480));
        assert!(!p.upcoming_repetition(1));
        let mut q = start(15, 16);
        play_king_tours(&mut q, 15, 16, 480);
        assert_eq!(q.st().key, key);
        assert_eq!(q.st().repetition, 480);

        // Period 544 > MAX_HIST: the repetition lies beyond the ring and is not seen.
        let mut p = start(16, 17);
        let (key, fen) = (p.key(), p.fen());
        let p_start_key = p.st().key;
        let moves = play_king_tours(&mut p, 16, 17, 511);
        assert!(!p.has_repeated());
        let mut q = start(16, 17);
        let all = play_king_tours(&mut q, 16, 17, 544);
        assert_eq!(q.st().key, p_start_key);
        assert_eq!(
            q.st().repetition,
            0,
            "slot 512 back, the current state, is not compared"
        );
        assert!(!q.has_repeated());
        // Undoing fewer than MAX_HIST plies restores every state exactly (further back, the
        // oldest slots have been reused, as in C++; nothing undoes that far).
        for &m in all[244..].iter().rev() {
            // SAFETY: undoing in reverse order.
            unsafe { q.undo_move(m) };
        }
        let mut r = start(16, 17);
        play_king_tours(&mut r, 16, 17, 244);
        assert_eq!(
            (q.key(), q.fen(), q.st().repetition),
            (r.key(), r.fen(), r.st().repetition)
        );
        for &m in moves.iter().rev() {
            // SAFETY: undoing in reverse order.
            unsafe { p.undo_move(m) };
        }
        assert_eq!((p.key(), p.fen()), (key, fen));
    }

    #[test]
    fn low_key_bit_is_the_side_to_move() {
        let mut p = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let moves = [
            Move::from_to(Square::SQ_E2, Square::SQ_E4),
            Move::from_to(Square::SQ_G8, Square::SQ_F6),
            Move::from_to(Square::SQ_G1, Square::SQ_F3),
        ];
        for m in moves.into_iter() {
            let stm = p.side_to_move() as Key;
            assert_eq!(p.key() & 1, stm);
            assert_eq!(p.pawn_key() & 1, stm);
            assert_eq!(p.minor_piece_key() & 1, stm);
            assert_eq!(p.non_pawn_key(Color::White) & 1, stm);
            assert_eq!(p.non_pawn_key(Color::Black) & 1, stm);
            // SAFETY: legal test moves.
            unsafe { p.do_move(m) };
        }
        assert_eq!(p.key() & 1, 1);
        // SAFETY: not in check.
        unsafe { p.do_null_move() };
        assert_eq!(p.key() & 1, 0);
        assert_eq!(p.pawn_key() & 1, 0);

        // The incremental keys agree with a fresh set-up, which puts the side to move in bit 0.
        let fresh = pos_from_fen(&p.fen());
        assert_eq!(p.key(), fresh.key());
        assert_eq!(p.pawn_key(), fresh.pawn_key());
        assert_eq!(p.minor_piece_key(), fresh.minor_piece_key());
        assert_eq!(p.st().piece_placement, fresh.st().piece_placement);
    }
}

fn char_to_piece(ch: char) -> Option<Piece> {
    let b = u8::try_from(ch).ok()?;
    PIECE_TO_CHAR
        .iter()
        .position(|&c| c == b)
        .map(|pos| Piece::from(pos as u8))
}

// Display implementation for Position - mirrors C++ operator<<
/// Returns an ASCII representation of the position
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n +---+---+---+---+---+---+---+---+")?;

        for r in (0..8).rev() {
            for file in 0..8 {
                let sq = make_square(File::from(file), Rank::from(r));
                write!(f, " | {}", piece_to_char(self.piece_on(sq)))?;
            }
            writeln!(f, " | {}\n +---+---+---+---+---+---+---+---+", r + 1)?;
        }

        writeln!(f, "   a   b   c   d   e   f   g   h\n")?;
        writeln!(f, "Fen: {}", self.fen())?;
        write!(f, "Key: {:016X}", self.key())?;
        write!(f, "\nCheckers: ")?;

        let mut checkers = self.checkers();
        while checkers != 0 {
            let sq = bitboard::pop_lsb(&mut checkers);
            write!(f, "{} ", crate::uci::UciEngine::square(sq))?;
        }

        // Tablebase probing for endgame positions (from position.cpp)
        // Check if tablebase probing is applicable
        if syzygy::MAX_CARDINALITY.load(core::sync::atomic::Ordering::Relaxed)
            >= bitboard::popcount(self.pieces()) as i32
            && !self.can_castle(CastlingRights::ANY_CASTLING)
        {
            // Create a new position for tablebase probing
            // In the C++ version they create a new Position object
            let mut probe_pos = Position::default();
            if probe_pos.set_fen(&self.fen(), self.is_chess960()).is_ok() {
                let mut probe_state_wdl = ProbeState::Fail;
                let mut probe_state_dtz = ProbeState::Fail;

                // Probe WDL (Win/Draw/Loss) tablebase
                let wdl = syzygy::probe_wdl(&mut probe_pos, &mut probe_state_wdl);

                // Probe DTZ (Distance to Zero) tablebase
                let dtz = syzygy::probe_dtz(&mut probe_pos, &mut probe_state_dtz);

                // Display tablebase results if probing succeeded
                writeln!(
                    f,
                    "\nTablebases WDL: {:4} ({:?})",
                    wdl as i32, probe_state_wdl
                )?;
                writeln!(f, "Tablebases DTZ: {:4} ({:?})", dtz, probe_state_dtz)?;
            }
        }

        Ok(())
    }
}
