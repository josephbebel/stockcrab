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

//! Syzygy tablebase probing. Port of upstream `syzygy/tbprobe.cpp`.

use core::cmp::{Ordering, max, min};
use core::ptr;
use core::sync::atomic::{AtomicI32, Ordering as AtomicOrdering};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::bitboard_fast::{edge_distance, pop_lsb};
use crate::movegen::{GenType, generate};
use crate::position::{Position, ScopedMove};
use crate::search::RootMoves;
use crate::types::{self, *};
use crate::ucioption::OptionsMap;

use std::fs::File as StdFile;

// Constants
const TBPIECES: usize = 7; // Max number of supported pieces
const MAX_DTZ: i32 = 1 << 18; // Max DTZ supported times 2
const SYM_COUNT: usize = 4096;

// Piece to char mapping for FEN notation
const PIECE_TO_CHAR: &str = " PNBRQK  pnbrqk";

// Global state
/// Largest number of men any loaded table covers.
///
/// Running maximum of men over loaded tables, updated with `fetch_max` as each
/// table is registered.
pub static MAX_CARDINALITY: AtomicI32 = AtomicI32::new(0);
static TB_TABLES: std::sync::LazyLock<Mutex<Arc<TBTables>>> =
    std::sync::LazyLock::new(|| Mutex::new(Arc::new(TBTables::new())));

// Mapping arrays - initialized in init()
static MAP_PAWNS: OnceLock<[i32; SQUARE_NB]> = OnceLock::new();
static MAP_B1H1H7: OnceLock<[i32; SQUARE_NB]> = OnceLock::new();
static MAP_A1D1D4: OnceLock<[i32; SQUARE_NB]> = OnceLock::new();
static MAP_KK: OnceLock<[[i32; SQUARE_NB]; 10]> = OnceLock::new();
static BINOMIAL: OnceLock<[[i32; SQUARE_NB]; 6]> = OnceLock::new();
static LEAD_PAWN_IDX: OnceLock<[[i32; SQUARE_NB]; 6]> = OnceLock::new();
static LEAD_PAWNS_SIZE: OnceLock<[[i32; 4]; 6]> = OnceLock::new();

/// Tablebase configuration for the current root position.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Maximum piece count for which tables are available.
    pub cardinality: i32,
    /// Whether the root position was found in the tables.
    pub root_in_tb: bool,
    /// Whether the 50-move rule is taken into account when scoring.
    pub use_rule50: bool,
    /// Minimum remaining search depth (in plies) at which the tables are probed.
    pub probe_depth: Depth,
    pub(crate) tables: TablebaseSnapshot,
}

/// WDL score values, from the point of view of the side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum WDLScore {
    WDLLoss = -2,        // Loss
    WDLBlessedLoss = -1, // Loss, but draw under 50-move rule
    #[default]
    WDLDraw = 0, // Draw
    WDLCursedWin = 1,    // Win, but draw under 50-move rule
    WDLWin = 2,          // Win
}

impl core::ops::Neg for WDLScore {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            WDLScore::WDLLoss => WDLScore::WDLWin,
            WDLScore::WDLBlessedLoss => WDLScore::WDLCursedWin,
            WDLScore::WDLDraw => WDLScore::WDLDraw,
            WDLScore::WDLCursedWin => WDLScore::WDLBlessedLoss,
            WDLScore::WDLWin => WDLScore::WDLLoss,
        }
    }
}

impl From<i32> for WDLScore {
    fn from(value: i32) -> Self {
        match value {
            -2 => WDLScore::WDLLoss,
            -1 => WDLScore::WDLBlessedLoss,
            0 => WDLScore::WDLDraw,
            1 => WDLScore::WDLCursedWin,
            2 => WDLScore::WDLWin,
            _ => WDLScore::WDLDraw,
        }
    }
}

// Probe state results
/// Possible states after a probing operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeState {
    Fail = 0,            // Probe failed (missing file table)
    Ok = 1,              // Probe successful
    ChangeSTM = -1,      // DTZ should check the other side
    ZeroingBestMove = 2, // Best move zeroes DTZ (capture or pawn move)
}

// Table types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TBType {
    WDL,
    DTZ,
}

// Table flags
/// Each table has a set of flags: all of them refer to DTZ tables, the last one to WDL tables
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum TBFlag {
    STM = 1,
    Mapped = 2,
    WinPlies = 4,
    LossPlies = 8,
    Wide = 16,
    SingleValue = 128,
}

// Sparse entry for indexing
/// Numbers in little-endian used by sparseIndex[] to point into blockLength[]
#[repr(C, packed)]
struct SparseEntry {
    block: [u8; 4],  // Number of block
    offset: [u8; 2], // Offset within the block
}

// Huffman symbol type
type Sym = u16;

// Left-Right tree entry for Huffman decoding
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
struct LR {
    lr: [u8; 3],
}

impl LR {
    fn get_left(&self) -> Sym {
        ((self.lr[1] as u16 & 0xF) << 8) | self.lr[0] as u16
    }

    fn get_right(&self) -> Sym {
        (self.lr[2] as u16) << 4 | (self.lr[1] as u16 >> 4)
    }
}

// Low-level indexing information for tablebase data
/// struct PairsData contains low-level indexing information to access TB data.
/// There are 8, 4, 2 or 1 PairsData records for each TBTable, according to the type
/// of table and if positions have pawns or not. It is populated at first access.
// Not `Clone`: `btree` points into this value's own `btree_storage`, so a clone would alias the
// original's heap buffer (and dangle once the original is dropped).
#[derive(Debug)]
struct PairsData {
    flags: u8,
    max_sym_len: u8,
    min_sym_len: u8,
    blocks_num: u32,
    sizeof_block: usize,
    span: usize,
    lowest_sym: *const Sym,
    btree: *const LR,
    block_length: *const u16,
    block_length_size: u32,
    sparse_index: *const SparseEntry,
    sparse_index_size: usize,
    data: *const u8,
    data_end: *const u8,
    base64: Vec<u64>,
    symlen: Vec<u8>,
    btree_storage: Vec<LR>,
    pieces: [Piece; TBPIECES],
    group_idx: [u64; TBPIECES + 1],
    group_len: [i32; TBPIECES + 1],
    map_idx: [u16; 4],
}

// SAFETY: PairsData's raw pointers target the read-only file mapping or the `btree_storage`
// owned by the same TBTable, and move together with it; no thread-affine resource is held.
unsafe impl Send for PairsData {}
// SAFETY: PairsData has no interior mutability. Its raw pointers point into the read-only file
// mapping owned by the same TBTable or into its own `btree_storage`, and are only written during
// table setup with exclusive access before OnceLock publication; shared access is read-only.
unsafe impl Sync for PairsData {}

impl Default for PairsData {
    fn default() -> Self {
        Self {
            flags: 0,
            max_sym_len: 0,
            min_sym_len: 0,
            blocks_num: 0,
            sizeof_block: 0,
            span: 0,
            lowest_sym: ptr::null(),
            btree: ptr::null(),
            block_length: ptr::null(),
            block_length_size: 0,
            sparse_index: ptr::null(),
            sparse_index_size: 0,
            data: ptr::null(),
            data_end: ptr::null(),
            base64: Vec::new(),
            symlen: Vec::new(),
            btree_storage: Vec::new(),
            pieces: [Piece::NoPiece; TBPIECES],
            group_idx: [0; TBPIECES + 1],
            group_len: [0; TBPIECES + 1],
            map_idx: [0; 4],
        }
    }
}

// Memory-mapped file wrapper
struct TBFile {
    mapping: Option<MappedFile>,
    path: PathBuf,
}

/// A read-only memory map of one tablebase file.
#[derive(Debug)]
struct MappedFile {
    map: memmap2::Mmap,
}

impl MappedFile {
    /// Start of the mapping. Never written through: the map is read-only.
    fn ptr(&self) -> *mut u8 {
        self.map.as_ptr().cast_mut()
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

impl TBFile {
    fn new(filename: &str, paths: &str) -> io::Result<Self> {
        let separator = if cfg!(windows) { ';' } else { ':' };

        // Empty segments of the search path are skipped, as upstream does.
        for path in paths.split(separator).filter(|p| !p.is_empty()) {
            let full_path = Path::new(path).join(filename);
            if full_path.exists() {
                return Ok(TBFile {
                    mapping: None,
                    path: full_path,
                });
            }
        }

        Err(io::Error::new(ErrorKind::NotFound, "TB file not found"))
    }

    fn map(&mut self, tb_type: TBType) -> io::Result<*mut u8> {
        let file = StdFile::open(&self.path)?;
        let size = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Tablebase file is too large"))?;

        if size % 64 != 16 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Corrupt tablebase file",
            ));
        }

        // SAFETY: as in C++, the mapping assumes the table file is not modified or truncated
        // while mapped (truncation would fault on access). It is only ever read.
        let map = unsafe { memmap2::Mmap::map(&file)? };

        // Verify magic bytes
        // C++ selects Magics[type == WDL]: WDL files start 71 E8 23 5D, DTZ files D7 66 0C A5.
        let magic: [u8; 4] = if tb_type == TBType::WDL {
            [0x71, 0xE8, 0x23, 0x5D]
        } else {
            [0xD7, 0x66, 0x0C, 0xA5]
        };
        // size >= 16 (size % 64 == 16), so the header is present.
        if map[..4] != magic {
            return Err(io::Error::new(ErrorKind::InvalidData, "Invalid magic"));
        }

        let mapping = MappedFile { map };
        let data = mapping.ptr();
        self.mapping = Some(mapping);
        // SAFETY: size >= 16, so `data + 4` is inside the mapping.
        Ok(unsafe { data.add(4) }) // Skip magic header
    }
}

// Table structure for WDL/DTZ
#[derive(Debug)]
struct TBTable<const TYPE: usize> {
    base_address: *mut u8,
    map: *mut u8,
    /// Bytes from `map` to the end of the live mapping; bounds every read through `map`.
    map_len: usize,
    mapping: Option<Box<MappedFile>>,
    key: Key,
    key2: Key,
    piece_count: i32,
    has_pawns: bool,
    has_unique_pieces: bool,
    pawn_count: [i32; 2],
    items: [[PairsData; 4]; 2], // [side][file]
}

// SAFETY: Each pointer targets this table's owned read-only mapping or heap allocation.
// Moving the table does not move these allocations. Initialization uses exclusive access;
// OnceLock publishes the completed table, after which all access is read-only.
unsafe impl<const TYPE: usize> Send for TBTable<TYPE> {}
// SAFETY: Published tables and their owned mapped bytes are immutable, as described above.
unsafe impl<const TYPE: usize> Sync for TBTable<TYPE> {}

impl<const TYPE: usize> TBTable<TYPE> {
    fn new() -> Self {
        TBTable {
            base_address: ptr::null_mut(),
            map: ptr::null_mut(),
            map_len: 0,
            mapping: None,
            key: 0,
            key2: 0,
            piece_count: 0,
            has_pawns: false,
            has_unique_pieces: false,
            pawn_count: [0; 2],
            items: Default::default(),
        }
    }

    /// Derive the table's material signature from a code such as "KRvK".
    ///
    /// Port of the C++ `TBTable<WDL>::TBTable(const std::string& code)`
    /// constructor (tbprobe.cpp), which sets up a `Position` from the code
    /// twice -- once with each side to move -- to obtain both material keys.
    ///
    /// The keys equal `Position::material_key()` of a position with this material.
    fn from_code(code: &str) -> Self {
        let mut table = Self::new();

        let mut pos = Position::default();

        // The code is not a legal position (e.g. the kings may be adjacent), so
        // `set_code` can report a validation error. C++ deliberately ignores it
        // for the same reason: only the material signature is wanted here.
        let _ = pos.set_code(code, Color::White);

        table.key = pos.material_key();
        table.piece_count = crate::bitboard_fast::popcount(pos.pieces()) as i32;
        table.has_pawns = pos.pieces_of_types(&[PieceType::Pawn]) != 0;

        table.has_unique_pieces = false;
        for c in [Color::White, Color::Black] {
            for pt in [
                PieceType::Pawn,
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Rook,
                PieceType::Queen,
            ] {
                if pos.count_of(c, pt) == 1 {
                    table.has_unique_pieces = true;
                }
            }
        }

        // Leading colour: when both sides have pawns it is the side with fewer
        // pawns, which compresses better.
        let wp = pos.count_of(Color::White, PieceType::Pawn);
        let bp = pos.count_of(Color::Black, PieceType::Pawn);
        let leading_white = bp == 0 || (wp > 0 && bp >= wp);

        table.pawn_count[0] = if leading_white { wp } else { bp };
        table.pawn_count[1] = if leading_white { bp } else { wp };

        let _ = pos.set_code(code, Color::Black);
        table.key2 = pos.material_key();

        table
    }

    fn get(&self, stm: usize, f: usize) -> &PairsData {
        let sides = if TYPE == 0 { 2 } else { 1 };
        let idx = if self.has_pawns { f } else { 0 };
        &self.items[stm % sides][idx]
    }

    fn get_mut(&mut self, stm: usize, f: usize) -> &mut PairsData {
        let sides = if TYPE == 0 { 2 } else { 1 }; // WDL has 2 sides, DTZ has 1
        let idx = if self.has_pawns { f } else { 0 };
        &mut self.items[stm % sides][idx]
    }
}

/// Own the path and mapping for one table independently of the current global set.
/// A probe keeps an Arc while decoding, so replacing SyzygyPath cannot unmap it.
#[derive(Debug)]
struct LazyTable<const TYPE: usize> {
    code: String,
    paths: String,
    table: OnceLock<Option<TBTable<TYPE>>>,
}

impl<const TYPE: usize> LazyTable<TYPE> {
    fn new(code: &str, paths: &str) -> Self {
        Self {
            code: code.to_owned(),
            paths: paths.to_owned(),
            table: OnceLock::new(),
        }
    }
}

impl LazyTable<0> {
    fn get(&self, pos: &Position) -> Option<&TBTable<0>> {
        self.table
            .get_or_init(|| {
                let mut table = TBTable::from_code(&self.code);
                map_table_wdl(&mut table, pos, &self.paths).then_some(table)
            })
            .as_ref()
    }
}

impl LazyTable<1> {
    fn get(&self, pos: &Position) -> Option<&TBTable<1>> {
        self.table
            .get_or_init(|| {
                let mut table = TBTable::from_code(&self.code);
                map_table_dtz(&mut table, pos, &self.paths).then_some(table)
            })
            .as_ref()
    }
}

/// A search retains one immutable table set. Lookup and decoding borrow its
/// tables without a global lock or an Arc reference-count operation per probe.
#[derive(Debug, Clone)]
pub(crate) struct TablebaseSnapshot(Arc<TBTables>);

impl Default for TablebaseSnapshot {
    fn default() -> Self {
        Self::current()
    }
}

impl TablebaseSnapshot {
    pub(crate) fn current() -> Self {
        Self(Arc::clone(&TB_TABLES.lock().unwrap()))
    }

    pub(crate) fn probe_wdl(&self, pos: &mut Position, result: &mut ProbeState) -> WDLScore {
        probe_wdl_with(&self.0, pos, result)
    }

    pub(crate) fn probe_dtz(&self, pos: &mut Position, result: &mut ProbeState) -> i32 {
        probe_dtz_with(&self.0, pos, result)
    }
}

// Main table storage and lookup
#[derive(Debug)]
struct TBTables {
    hash_table: Vec<Option<TBEntry>>,
    found_wdl_files: usize,
    found_dtz_files: usize,
    max_cardinality: i32,
}

#[derive(Clone, Debug)]
struct TBEntry {
    key: Key,
    wdl: Arc<LazyTable<0>>,
    dtz: Arc<LazyTable<1>>,
}

impl TBTables {
    const SIZE: usize = 1 << 12; // 4K table
    const OVERFLOW: usize = 1;

    fn new() -> Self {
        TBTables {
            hash_table: vec![None; Self::SIZE + Self::OVERFLOW],
            found_wdl_files: 0,
            found_dtz_files: 0,
            max_cardinality: 0,
        }
    }

    fn insert(&mut self, key: Key, wdl: Arc<LazyTable<0>>, dtz: Arc<LazyTable<1>>) {
        let home_bucket = (key as usize) & (Self::SIZE - 1);
        let entry = TBEntry { key, wdl, dtz };

        // Robin Hood hashing
        let mut current_entry = entry;
        let mut current_home = home_bucket;

        // Ensure last element is empty to avoid overflow when looking up
        for bucket in home_bucket..Self::SIZE + Self::OVERFLOW - 1 {
            match &self.hash_table[bucket] {
                None => {
                    self.hash_table[bucket] = Some(current_entry);
                    return;
                }
                Some(existing) if existing.key == current_entry.key => {
                    self.hash_table[bucket] = Some(current_entry);
                    return;
                }
                Some(existing) => {
                    // Robin Hood hashing: If we've probed for longer than this element,
                    // insert here and search for a new spot for the other element instead.
                    let other_home = (existing.key as usize) & (Self::SIZE - 1);
                    if other_home > current_home {
                        let old = self.hash_table[bucket].replace(current_entry);
                        current_entry = old.unwrap();
                        current_home = other_home;
                    }
                }
            }
        }

        panic!("TB hash table size too low!");
    }

    fn get_wdl(&self, key: Key) -> Option<&Arc<LazyTable<0>>> {
        let bucket = (key as usize) & (Self::SIZE - 1);

        for i in bucket..Self::SIZE + Self::OVERFLOW {
            match &self.hash_table[i] {
                Some(entry) if entry.key == key => return Some(&entry.wdl),
                None => return None,
                _ => continue,
            }
        }

        None
    }

    fn get_dtz(&self, key: Key) -> Option<&Arc<LazyTable<1>>> {
        let bucket = (key as usize) & (Self::SIZE - 1);

        for i in bucket..Self::SIZE + Self::OVERFLOW {
            match &self.hash_table[i] {
                Some(entry) if entry.key == key => return Some(&entry.dtz),
                None => return None,
                _ => continue,
            }
        }

        None
    }

    /// If the corresponding file exists two new objects `TBTable<WDL>` and `TBTable<DTZ>`
    /// are created and added to the lists and hash table. Called at init time.
    fn add(&mut self, pieces: Vec<PieceType>, paths: &str) {
        // Build code string from pieces
        let mut code = String::new();
        for pt in &pieces {
            code.push(PIECE_TO_CHAR.chars().nth(*pt as usize).unwrap());
        }

        // Insert 'v' before the second king, separating the two sides (KRK -> KRvK)
        if let Some(pos) = code[1..].find('K') {
            code.insert(pos + 1, 'v');
        }

        // Check for DTZ file
        let dtz_filename = format!("{}.rtbz", code);
        if TBFile::new(&dtz_filename, paths).is_ok() {
            self.found_dtz_files += 1;
        }

        // Check for WDL file
        let wdl_filename = format!("{}.rtbw", code);
        if TBFile::new(&wdl_filename, paths).is_err() {
            return; // Only WDL file is required
        }

        self.found_wdl_files += 1;

        // Update max cardinality (C++ tbprobe.cpp).
        self.max_cardinality = self.max_cardinality.max(pieces.len() as i32);

        let signature = TBTable::<0>::from_code(&code);
        let wdl = Arc::new(LazyTable::new(&code, paths));
        let dtz = Arc::new(LazyTable::new(&code, paths));
        self.insert(signature.key, Arc::clone(&wdl), Arc::clone(&dtz));
        self.insert(signature.key2, wdl, dtz);
    }

    fn info(&self) {
        let max_card = self.max_cardinality;
        crate::uci::UciEngine::write_stdout_bytes(
            format!(
                "info string Found {} WDL and {} DTZ tablebase files (up to {}-man).\n",
                self.found_wdl_files, self.found_dtz_files, max_card
            )
            .as_bytes(),
        );
    }
}

// Utility functions

fn off_a1h8(sq: Square) -> i32 {
    rank_of(sq) as i32 - file_of(sq) as i32
}

/// Comparison function to sort leading pawns in ascending MapPawns[] order
fn pawns_comp(a: &Square, b: &Square) -> Ordering {
    let map_pawns = MAP_PAWNS.get().unwrap();
    map_pawns[*a as usize].cmp(&map_pawns[*b as usize])
}

fn sign_of<T: PartialOrd + Default>(val: T) -> i32 {
    if val > T::default() {
        1
    } else if val < T::default() {
        -1
    } else {
        0
    }
}

/// DTZ tables don't store valid scores for moves that reset the rule50 counter
/// like captures and pawn moves but we can easily recover the correct dtz of the
/// previous move if we know the position's WDL score.
fn dtz_before_zeroing(wdl: WDLScore) -> i32 {
    match wdl {
        WDLScore::WDLWin => 1,
        WDLScore::WDLCursedWin => 101,
        WDLScore::WDLBlessedLoss => -101,
        WDLScore::WDLLoss => -1,
        WDLScore::WDLDraw => 0,
    }
}

/// Plain unsigned integers that can be decoded from raw file bytes.
trait FileInt: Copy {
    /// Read a little-endian value from `p`.
    ///
    /// # Safety
    /// `p` must be valid for reads of `size_of::<Self>()` bytes; no alignment is required.
    unsafe fn read_le(p: *const u8) -> Self;
    /// Read a big-endian value from `p`. Same contract as `read_le`.
    unsafe fn read_be(p: *const u8) -> Self;
}

macro_rules! impl_file_int {
    ($($t:ty),*) => {$(
        impl FileInt for $t {
            #[inline(always)]
            unsafe fn read_le(p: *const u8) -> Self {
                // SAFETY: the caller guarantees `p` is readable for size_of::<$t>() bytes;
                // read_unaligned has no alignment requirement and every bit pattern is valid.
                <$t>::from_le(unsafe { p.cast::<$t>().read_unaligned() })
            }
            #[inline(always)]
            unsafe fn read_be(p: *const u8) -> Self {
                // SAFETY: as for read_le.
                <$t>::from_be(unsafe { p.cast::<$t>().read_unaligned() })
            }
        }
    )*};
}
impl_file_int!(u16, u32, u64);

/// Read a little-endian integer from raw file memory.
///
/// # Safety
/// `addr` must be valid for reads of `size_of::<T>()` bytes (callers establish this with `fits`
/// against the mapping end, or by indexing an element already known to be in bounds). Alignment
/// is not required.
#[inline(always)]
unsafe fn number_le<T: FileInt>(addr: *const u8) -> T {
    // SAFETY: forwarded caller contract.
    unsafe { T::read_le(addr) }
}

/// Read a big-endian integer from raw file memory. Same contract as `number_le`.
///
/// # Safety
/// See `number_le`.
#[inline(always)]
unsafe fn number_be<T: FileInt>(addr: *const u8) -> T {
    // SAFETY: forwarded caller contract.
    unsafe { T::read_be(addr) }
}

/// View the file bytes in `[data, end)` as a slice.
///
/// # Safety
/// `data..end` must lie within one live mapping with `data <= end` (the load-time callers pass a
/// cursor already checked against the mapping end), and the mapping must outlive the returned
/// slice and not be written while it is borrowed. The slice is only used inside the table-load
/// functions, while the mapping is held by the table being built.
unsafe fn file_slice<'a>(data: *const u8, end: *const u8) -> Option<&'a [u8]> {
    let len = (end as usize).checked_sub(data as usize)?;
    // SAFETY: the caller guarantees `len` bytes from `data` are readable and live.
    Some(unsafe { core::slice::from_raw_parts(data, len) })
}

/// Check that `count` objects of `stride` bytes starting at `p` fit before
/// `end`. All pointers passed here refer to the same memory mapping.
fn fits(p: *const u8, count: usize, stride: usize, end: *const u8) -> bool {
    if (p as usize) > (end as usize) {
        return false;
    }

    let room = end as usize - p as usize;
    count == 0 || stride <= room / count
}

// Decompression function
/// TB tables are compressed with canonical Huffman code. The compressed data is divided into
/// blocks of size d->sizeofBlock, and each block stores a variable number of symbols.
/// Each symbol represents either a WDL or a (remapped) DTZ value, or a pair of other symbols
/// (recursively). If you keep expanding the symbols in a block, you end up with up to 65536
/// WDL or DTZ values. Each symbol represents up to 256 values and will correspond after
/// Huffman coding to at least 1 bit. So a block of 32 bytes corresponds to at most
/// 32 x 8 x 256 = 65536 values. This maximum is only reached for tables that consist mostly
/// of draws or mostly of wins, but such tables are actually quite common. In principle, the
/// blocks in WDL tables are 64 bytes long (and will be aligned on cache lines). But for
/// mostly-draw or mostly-win tables this can leave many 64-byte blocks only half-filled, so
/// in such cases blocks are 32 bytes long. The blocks of DTZ tables are up to 1024 bytes long.
/// The generator picks the size that leads to the smallest table. The "book" of symbols and
/// Huffman codes are the same for all blocks in the table. A non-symmetric pawnless TB file
/// will have one table for wtm and one for btm, a TB file with pawns will have tables per
/// file a,b,c,d also, in this case, one set for wtm and one for btm.
fn decompress_pairs(d: &PairsData, idx: u64) -> i32 {
    // Special case for single value
    // Special case where all table positions store the same value
    if d.flags & TBFlag::SingleValue as u8 != 0 {
        return d.min_sym_len as i32;
    }

    if d.span == 0
        || d.block_length_size == 0
        || d.blocks_num == 0
        || d.sparse_index.is_null()
        || d.block_length.is_null()
        || d.data.is_null()
        || d.data_end.is_null()
    {
        return 0;
    }

    // SAFETY: Every pointer here was bounds-checked at load time against the mapping end
    // (set_sizes, set_table_pointers, via `fits` and checked arithmetic), and entries are only
    // probed while `base_address` is non-null, i.e. the mapping is alive. Indices stay in range:
    // `k` < sparse_index_size is checked; `block` is clamped below block_length_size and every step
    // keeps it there, and `block` < blocks_num is checked before `data + block * sizeof_block` is
    // used (and re-checked with `fits`); `len` < base64.len() bounds lowest_sym; every symbol index
    // is masked or compared against SYM_COUNT, the length of btree_storage. Reads are bytewise via
    // number_le/number_be.
    unsafe {
        // Find the block containing our index
        let k = (idx / d.span as u64) as usize;
        if k >= d.sparse_index_size {
            return 0;
        }

        let sparse = &*d.sparse_index.add(k);
        let mut block = number_le::<u32>(sparse.block.as_ptr()) as usize;
        let mut offset = number_le::<u16>(sparse.offset.as_ptr()) as i32;

        // Now compute the difference idx - I(k). From the definition of k, we know that
        //
        // idx = k * d->span + idx % d->span    (2)
        //
        // So from (1) and (2) we can compute idx - I(K):
        let diff = (idx % d.span as u64) as i32 - (d.span / 2) as i32;
        offset += diff;

        let block_length = |block: usize| -> i32 {
            number_le::<u16>(d.block_length.add(block) as *const u8) as i32
        };

        if block >= d.block_length_size as usize {
            block = d.block_length_size as usize - 1;
        }
        // Find the correct block
        while offset < 0 && block > 0 {
            block -= 1;
            offset += block_length(block) + 1;
        }

        while offset > block_length(block) && block + 1 < d.block_length_size as usize {
            offset -= block_length(block) + 1;
            block += 1;
        }
        if offset < 0 || offset > block_length(block) || block >= d.blocks_num as usize {
            return 0;
        }

        // Get the data pointer
        let block_offset = match block.checked_mul(d.sizeof_block) {
            Some(offset) => offset,
            None => return 0,
        };
        // May lie outside the mapping for a corrupt `blocks_num`; `fits()` below bounds-checks
        // it before any read, so only address arithmetic is done here.
        let block_ptr = d.data.wrapping_add(block_offset);
        // Finally, we find the start address of our block of canonical Huffman symbols
        if !fits(block_ptr, 1, d.sizeof_block, d.data_end) {
            return 0;
        }

        // Read and decode the Huffman symbols
        if !fits(block_ptr, 8, 1, d.data_end) {
            return 0;
        }
        let mut buf64 = number_be::<u64>(block_ptr);
        let mut next_ptr = block_ptr.add(8);
        let mut buf64_size = 64;

        let sym = loop {
            let mut len = 0;

            // Find symbol length
            // Now get the symbol length. For any symbol s64 of length l right-padded
            // to 64 bits we know that d->base64[l-1] >= s64 >= d->base64[l] so we
            // can find the symbol length iterating through base64[].
            while len < d.base64.len() && buf64 < d.base64[len] {
                len += 1;
            }
            if len >= d.base64.len() {
                return 0;
            }

            // Decode symbol
            // All the symbols of a given length are consecutive integers (numerical
            // sequence property), so we can compute the offset of our symbol of
            // length len, stored at the beginning of buf64.
            let mut sym = ((buf64 - d.base64[len]) >> (64 - len - d.min_sym_len as usize)) as Sym;
            sym = sym.wrapping_add(number_le::<Sym>(d.lowest_sym.add(len) as *const u8))
                & (SYM_COUNT as Sym - 1);

            // Check if we found our value
            // If our offset is within the number of values represented by symbol sym,
            // we are done.
            if offset < d.symlen[sym as usize] as i32 + 1 {
                break sym;
            }

            // Continue searching
            // ...otherwise update the offset and continue to iterate
            offset -= d.symlen[sym as usize] as i32 + 1;
            let real_len = len + d.min_sym_len as usize;
            buf64 <<= real_len;
            buf64_size -= real_len;

            if buf64_size <= 32 {
                buf64_size += 32;
                if !fits(next_ptr, 4, 1, d.data_end) {
                    return 0;
                }
                buf64 |= (number_be::<u32>(next_ptr) as u64) << (64 - buf64_size);
                next_ptr = next_ptr.add(4);
            }
        };

        // Expand the symbol recursively
        let mut current_sym = sym;
        while d.symlen[current_sym as usize] > 0 {
            let lr = &*d.btree.add(current_sym as usize);
            let left = lr.get_left();
            if left as usize >= SYM_COUNT {
                return 0;
            }

            // If a symbol contains 36 sub-symbols (d->symlen[sym] + 1 = 36) and
            // expands in a pair (d->symlen[left] = 23, d->symlen[right] = 11), then
            // we know that, for instance, the tenth value (offset = 10) will be on
            // the left side because in Recursive Pairing child symbols are adjacent.
            if offset < d.symlen[left as usize] as i32 + 1 {
                current_sym = left;
            } else {
                offset -= d.symlen[left as usize] as i32 + 1;
                current_sym = lr.get_right();
                if current_sym as usize >= SYM_COUNT {
                    return 0;
                }
            }
        }

        let final_lr = &*d.btree.add(current_sym as usize);
        final_lr.get_left() as i32
    }
}

// WDL-specific mapping
// DTZ scores are sorted by frequency of occurrence and then assigned the
// values 0, 1, 2, ... in order of decreasing frequency. This is done for each
// of the four WDLScore values. The mapping information necessary to reconstruct
// the original values are stored in the TB file and read during map[] init.
fn map_score_wdl(_entry: &TBTable<0>, _f: types::File, value: i32, _wdl: WDLScore) -> WDLScore {
    WDLScore::from(value - 2)
}

// DTZ-specific mapping
fn map_score_dtz(entry: &TBTable<1>, f: types::File, value: i32, wdl: WDLScore) -> i32 {
    const WDL_MAP: [usize; 5] = [1, 3, 0, 2, 0];

    let flags = entry.get(0, f as usize).flags;
    let map = entry.map;
    let idx = &entry.get(0, f as usize).map_idx;

    let mut result = value;

    if flags & TBFlag::Mapped as u8 != 0 {
        // `idx[..]` was recorded in set_dtz_map as an element offset inside the mapping, but
        // `value` (a decoded symbol) is added on top, so bound the final read by `map_len`. A
        // table that fails the check is corrupt; report 0 like the other corrupt-table paths.
        let wide = flags & TBFlag::Wide as u8 != 0;
        let elem = idx[WDL_MAP[(wdl as i32 + 2) as usize]] as usize;
        let Some(i) = usize::try_from(value)
            .ok()
            .and_then(|v| v.checked_add(elem))
        else {
            return 0;
        };
        let (offset, width) = if wide {
            (i.checked_mul(2), 2)
        } else {
            (Some(i), 1)
        };
        let Some(offset) = offset else {
            return 0;
        };
        if map.is_null() || offset.checked_add(width).is_none_or(|e| e > entry.map_len) {
            return 0;
        }
        // SAFETY: `map` points at the start of `map_len` readable bytes of the live mapping
        // (set in set_dtz_map) and `offset + width <= map_len` was just checked. The u16 is read
        // unaligned, so no alignment of `map` is assumed.
        result = unsafe {
            if wide {
                u16::from_le(map.add(offset).cast::<u16>().read_unaligned()) as i32
            } else {
                *map.add(offset) as i32
            }
        };
    }

    // Convert to plies if needed
    // DTZ tables store distance to zero in number of moves or plies. We
    // want to return plies, so we have to convert to plies when needed.
    if (wdl == WDLScore::WDLWin && flags & TBFlag::WinPlies as u8 == 0)
        || (wdl == WDLScore::WDLLoss && flags & TBFlag::LossPlies as u8 == 0)
        || wdl == WDLScore::WDLCursedWin
        || wdl == WDLScore::WDLBlessedLoss
    {
        result *= 2;
    }

    result + 1
}

// Main probing functions
fn do_probe_table_wdl(pos: &Position, entry: &TBTable<0>, result: &mut ProbeState) -> WDLScore {
    do_probe_table_impl(pos, entry, WDLScore::WDLDraw, result, map_score_wdl)
}

fn do_probe_table_dtz(
    pos: &Position,
    entry: &TBTable<1>,
    wdl: WDLScore,
    result: &mut ProbeState,
) -> i32 {
    do_probe_table_impl(pos, entry, wdl, result, map_score_dtz)
}

fn do_probe_table_impl<const TYPE: usize, R, F>(
    pos: &Position,
    entry: &TBTable<TYPE>,
    wdl: WDLScore,
    result: &mut ProbeState,
    map_fn: F,
) -> R
where
    R: Default,
    F: Fn(&TBTable<TYPE>, File, i32, WDLScore) -> R,
{
    let mut squares = [Square::SQ_A1; TBPIECES];
    let mut pieces = [Piece::NoPiece; TBPIECES];
    let mut size = 0;
    let mut lead_pawns_cnt = 0;
    // The leading pawns in *board* coordinates. C++ keeps this
    // (`leadPawns = b = pos.pieces(color_of(pc), PAWN)`, tbprobe.cpp) before
    // any flipping, because it is later XORed against the equally unflipped
    // `pos.pieces()`.
    let mut lead_pawns_bb: u64 = 0;

    // Check if we need to flip colors/squares
    // A given TB entry like KRK has associated two material keys: KRvk and Kvkr.
    // If both sides have the same pieces keys are equal. In this case TB tables
    // only stores the 'white to move' case, so if the position to lookup has black
    // to move, we need to switch the color and flip the squares before to lookup.
    let symmetric_btm = entry.key == entry.key2 && pos.side_to_move() == Color::Black;
    // TB files are calculated for white as the stronger side. For instance, we
    // have KRvK, not KvKR. A position where the stronger side is white will have
    // its material key == entry->key, otherwise we have to switch the color and
    // flip the squares before to lookup.
    let black_stronger = pos.material_key() != entry.key;
    let flip_color = (symmetric_btm || black_stronger) as usize * 8;
    let flip_squares = (symmetric_btm || black_stronger) as usize * 56;
    let stm = ((symmetric_btm || black_stronger) as usize) ^ (pos.side_to_move() as usize);

    let mut tb_file = File::A;

    // Handle pawns
    // For pawns, TB files store 4 separate tables according if leading pawn is on
    // file a, b, c or d after reordering. The leading pawn is the one with maximum
    // MapPawns[] value, that is the one most toward the edges and with lowest rank.
    if entry.has_pawns {
        // In all the 4 tables, pawns are at the beginning of the piece sequence and
        // their color is the reference one. So we just pick the first one.
        let pc = (entry.get(0, 0).pieces[0] as u32) ^ (flip_color as u32);
        let pc_piece = Piece::from(pc as u8);

        if type_of(pc_piece) != PieceType::Pawn {
            *result = ProbeState::Fail;
            return R::default();
        }

        let mut b = pos.pieces_color_types(color_of(pc_piece), &[PieceType::Pawn]);
        lead_pawns_bb = b;
        while b != 0 {
            squares[size] = Square::from(pop_lsb(&mut b) as u8 ^ flip_squares as u8);
            size += 1;
        }

        lead_pawns_cnt = size;

        // Find leading pawn
        let mut max_idx = 0;
        let map_pawns = MAP_PAWNS.get().unwrap();
        for i in 1..lead_pawns_cnt {
            if map_pawns[squares[i] as usize] > map_pawns[squares[max_idx] as usize] {
                max_idx = i;
            }
        }
        squares.swap(0, max_idx);

        tb_file = types::File::from(edge_distance(file_of(squares[0])) as u8);
    }

    // Check DTZ STM
    if TYPE == 1 {
        // DTZ
        // DTZ tables are one-sided, i.e. they store positions only for white to
        // move or only for black to move, so check for side to move to be stm,
        // early exit otherwise.
        let flags = entry.get(stm, tb_file as usize).flags;
        let stm_matches = (flags & TBFlag::STM as u8) == stm as u8;
        let symmetric = entry.key == entry.key2 && !entry.has_pawns;

        if !stm_matches && !symmetric {
            *result = ProbeState::ChangeSTM;
            return R::default();
        }
    }

    // Get remaining pieces
    // `lead_pawns_bb` was captured before flipping and `pos.pieces()` is unflipped,
    // so the XOR removing the lead pawns uses the unflipped mask. Now we are ready
    // to get all the position pieces (but the lead pawns) and directly map them to
    // the correct color and square.
    let mut b = pos.pieces() ^ lead_pawns_bb;
    while b != 0 {
        let sq = pop_lsb(&mut b) as u8;
        squares[size] = Square::from(sq ^ flip_squares as u8);
        pieces[size] =
            Piece::from((pos.piece_on(Square::from(sq)) as u32 ^ flip_color as u32) as u8);
        size += 1;
    }

    assert!(size >= 2);

    let has_pawns = entry.has_pawns;
    let has_unique_pieces = entry.has_unique_pieces;
    let entry_pawn_count1 = entry.pawn_count[1];
    let d = entry.get(stm, tb_file as usize);

    // Reorder pieces to match compression sequence
    // Then we reorder the pieces to have the same sequence as the one stored
    // in pieces[i]: the sequence that ensures the best compression.
    for i in lead_pawns_cnt..size - 1 {
        for j in i + 1..size {
            if d.pieces[i] == pieces[j] {
                pieces.swap(i, j);
                squares.swap(i, j);
                break;
            }
        }
    }

    // Apply square mappings
    if file_of(squares[0]) > types::File::D {
        for i in 0..size {
            squares[i] = flip_file(squares[i]);
        }
    }

    // Calculate index
    // Both sides having pawns means the second group is the non-leading pawns,
    // which cannot stand on rank 1 (C++ `remainingPawns`).
    let remaining_pawns = has_pawns && entry_pawn_count1 != 0;
    let idx = if has_pawns {
        calculate_pawn_index(&mut squares, lead_pawns_cnt, d, remaining_pawns)
    } else {
        calculate_piece_index(has_unique_pieces, &mut squares, size, d)
    };

    // Decompress and return score
    let value = decompress_pairs(d, idx);
    map_fn(entry, tb_file, value, wdl)
}

/// Encode the non-leading groups into the table index.
///
/// Port of the `encode_remaining` tail of C++ `do_probe_table`
/// (tbprobe.cpp), shared by the pawn and the pawnless paths.
///
/// `remaining_pawns` shifts the first encoded group down by a rank, because when
/// both sides have pawns the second pawn group cannot occupy rank 1.
fn encode_remaining(
    mut idx: u64,
    squares: &mut [Square; TBPIECES],
    d: &PairsData,
    mut remaining_pawns: bool,
) -> u64 {
    let binomial = BINOMIAL.get().unwrap();

    idx *= d.group_idx[0];
    let mut group_sq = d.group_len[0] as usize;
    let mut next = 1usize;

    while next < d.group_len.len() && d.group_len[next] > 0 {
        let group_len = d.group_len[next] as usize;
        squares[group_sq..group_sq + group_len].sort();

        let mut n = 0u64;
        for i in 0..group_len {
            let sq_i = squares[group_sq + i];
            // Map a square down if it "comes later" than one already encoded.
            let adjust = squares[..group_sq].iter().filter(|&&s| sq_i > s).count();
            let k = sq_i as usize - adjust - 8 * remaining_pawns as usize;
            n += binomial[i + 1][k] as u64;
        }

        remaining_pawns = false;
        idx += n * d.group_idx[next];
        group_sq += group_len;
        next += 1;
    }

    idx
}

fn calculate_pawn_index(
    squares: &mut [Square; TBPIECES],
    lead_pawns_cnt: usize,
    d: &PairsData,
    remaining_pawns: bool,
) -> u64 {
    let lead_pawn_idx = LEAD_PAWN_IDX.get().unwrap();
    let map_pawns = MAP_PAWNS.get().unwrap();
    let binomial = BINOMIAL.get().unwrap();

    let mut idx = lead_pawn_idx[lead_pawns_cnt][squares[0] as usize] as u64;

    // Leading pawns are encoded in ascending MapPawns[] order after the first.
    squares[1..lead_pawns_cnt].sort_by(pawns_comp);

    for i in 1..lead_pawns_cnt {
        idx += binomial[i][map_pawns[squares[i] as usize] as usize] as u64;
    }

    // C++ falls through to `encode_remaining` from here.
    encode_remaining(idx, squares, d, remaining_pawns)
}

fn calculate_piece_index(
    has_unique_pieces: bool,
    squares: &mut [Square; TBPIECES],
    size: usize,
    d: &PairsData,
) -> u64 {
    // Apply additional square transformations for pawnless positions
    if rank_of(squares[0]) > Rank::R4 {
        for i in 0..size {
            squares[i] = flip_rank(squares[i]);
        }
    }

    // Check diagonal mappings
    for i in 0..d.group_len[0] as usize {
        if off_a1h8(squares[i]) == 0 {
            continue;
        }

        if off_a1h8(squares[i]) > 0 {
            for j in i..size {
                let sq = squares[j] as u8;
                squares[j] = Square::from(((sq >> 3) | (sq << 3)) & 63);
            }
        }
        break;
    }

    // Calculate index based on piece configuration
    let map_a1d1d4 = MAP_A1D1D4.get().unwrap();
    let map_b1h1h7 = MAP_B1H1H7.get().unwrap();
    let map_kk = MAP_KK.get().unwrap();

    // Encode the leading group.
    //
    // Suppose we have KRvK. Let's say the pieces are on square numbers wK, wR
    // and bK (each 0...63). The simplest way to map this position to an index
    // is like this:
    //
    // index = wK * 64 * 64 + wR * 64 + bK;
    //
    // But this way the TB is going to have 64*64*64 = 262144 positions, with
    // lots of positions being equivalent (because they are mirrors of each
    // other) and lots of positions being invalid (two pieces on one square,
    // adjacent kings, etc.).
    // Usually the first step is to take the wK and bK together. There are just
    // 462 ways legal and not-mirrored ways to place the wK and bK on the board.
    // Once we have placed the wK and bK, there are 62 squares left for the wR
    // Mapping its square from 0..63 to available squares 0..61 can be done like:
    //
    // wR -= (wR > wK) + (wR > bK);
    //
    // In words: if wR "comes later" than wK, we deduct 1, and the same if wR
    // "comes later" than bK. In case of two same pieces like KRRvK we want to
    // place the two Rs "together". If we have 62 squares left, we can place two
    // Rs "together" in 62 * 61 / 2 ways (we divide by 2 because rooks can be
    // swapped and still get the same position.)
    //
    // In case we have at least 3 unique pieces (including kings) we encode them
    // together.
    let idx = if has_unique_pieces {
        // Complex indexing for unique pieces
        let adjust1 = (squares[1] > squares[0]) as u64;
        let adjust2 = ((squares[2] > squares[0]) as u64) + ((squares[2] > squares[1]) as u64);

        if off_a1h8(squares[0]) != 0 {
            (map_a1d1d4[squares[0] as usize] as u64 * 63 + (squares[1] as u64 - adjust1)) * 62
                + squares[2] as u64
                - adjust2
        } else if off_a1h8(squares[1]) != 0 {
            (6 * 63 + rank_of(squares[0]) as u64 * 28 + map_b1h1h7[squares[1] as usize] as u64) * 62
                + squares[2] as u64
                - adjust2
        } else if off_a1h8(squares[2]) != 0 {
            6 * 63 * 62
                + 4 * 28 * 62
                + rank_of(squares[0]) as u64 * 7 * 28
                + (rank_of(squares[1]) as u64 - adjust1) * 28
                + map_b1h1h7[squares[2] as usize] as u64
        } else {
            6 * 63 * 62
                + 4 * 28 * 62
                + 4 * 7 * 28
                + rank_of(squares[0]) as u64 * 7 * 6
                + (rank_of(squares[1]) as u64 - adjust1) * 6
                + (rank_of(squares[2]) as u64 - adjust2)
        }
    } else {
        // Simple king mapping
        map_kk[map_a1d1d4[squares[0] as usize] as usize][squares[1] as usize] as u64
    };

    // Pawnless tables never have a second pawn group, hence `false`.
    encode_remaining(idx, squares, d, false)
}

// Search functions for probing
fn search_wdl(tables: &TBTables, pos: &mut Position, result: &mut ProbeState) -> WDLScore {
    search_impl::<false>(tables, pos, result)
}

fn search_dtz(tables: &TBTables, pos: &mut Position, result: &mut ProbeState) -> WDLScore {
    search_impl::<true>(tables, pos, result)
}

fn search_impl<const CHECK_ZEROING: bool>(
    tables: &TBTables,
    pos: &mut Position,
    result: &mut ProbeState,
) -> WDLScore {
    let mut best_value = WDLScore::WDLLoss;
    let moves = generate(pos, GenType::All);
    let total_count = moves.len();
    let mut move_count = 0;

    for &m in &moves {
        if !pos.capture(m) && (!CHECK_ZEROING || type_of(pos.moved_piece(m)) != PieceType::Pawn) {
            continue;
        }

        move_count += 1;

        // SAFETY: the move was generated as legal; the guard undoes it,
        // including on unwind.
        let mut moved = unsafe { ScopedMove::new(pos, m) };
        let pos = &mut *moved;
        let value = -search_impl::<false>(tables, pos, result);
        drop(moved);

        if *result == ProbeState::Fail {
            return WDLScore::WDLDraw;
        }

        if value > best_value {
            best_value = value;

            if value >= WDLScore::WDLWin {
                *result = ProbeState::ZeroingBestMove;
                return value;
            }
        }
    }

    let no_more_moves = move_count > 0 && move_count == total_count;

    let value = if no_more_moves {
        best_value
    } else {
        probe_table_wdl(tables, pos, result)
    };

    if *result == ProbeState::Fail {
        return WDLScore::WDLDraw;
    }

    if best_value >= value {
        *result = if best_value > WDLScore::WDLDraw || no_more_moves {
            ProbeState::ZeroingBestMove
        } else {
            ProbeState::Ok
        };
        return best_value;
    }

    *result = ProbeState::Ok;
    value
}

/// Memory-map a table's file and parse its header. Called once per table, at
/// first access. Port of C++ `mapped` (tbprobe.cpp).
///
/// Returns false when the file is missing or unreadable, which the caller turns
/// into `ProbeState::Fail`. The owning OnceLock retains success or failure,
/// so a missing file is not re-probed on every node.
fn map_table_wdl(entry: &mut TBTable<0>, pos: &Position, paths: &str) -> bool {
    let fname = tb_filename(entry.key, pos, true);

    let mut file = match TBFile::new(&fname, paths) {
        Ok(f) => f,
        Err(_) => {
            return false;
        }
    };

    let data = match file.map(TBType::WDL) {
        Ok(d) => d,
        Err(_) => {
            return false;
        }
    };

    entry.base_address = file.mapping.as_ref().map_or(ptr::null_mut(), |m| m.ptr());
    let end = match file.mapping.as_ref() {
        // SAFETY: `mapping.ptr() + mapping.len()` is one past the end of the mapping, which `add`
        // allows.
        Some(mapping) => unsafe { mapping.ptr().add(mapping.len()) },
        None => ptr::null_mut(),
    };
    let tail = match set_table(entry, data, end) {
        Some(tail) => tail,
        None => {
            entry.base_address = ptr::null_mut();
            return false;
        }
    };
    if !set_table_pointers(entry, tail, end) {
        entry.base_address = ptr::null_mut();
        return false;
    }
    entry.mapping = file.mapping.take().map(Box::new);

    true
}

/// DTZ counterpart of [`map_table_wdl`]. Differs only in the magic bytes and in
/// the value-remapping table that sits between the pairs data and the blocks.
fn map_table_dtz(entry: &mut TBTable<1>, pos: &Position, paths: &str) -> bool {
    let fname = tb_filename(entry.key, pos, false);

    let mut file = match TBFile::new(&fname, paths) {
        Ok(f) => f,
        Err(_) => {
            return false;
        }
    };

    let data = match file.map(TBType::DTZ) {
        Ok(d) => d,
        Err(_) => {
            return false;
        }
    };

    entry.base_address = file.mapping.as_ref().map_or(ptr::null_mut(), |m| m.ptr());
    let end = match file.mapping.as_ref() {
        // SAFETY: `mapping.ptr() + mapping.len()` is one past the end of the mapping, which `add`
        // allows.
        Some(mapping) => unsafe { mapping.ptr().add(mapping.len()) },
        None => ptr::null_mut(),
    };
    let mut tail = match set_table(entry, data, end) {
        Some(tail) => tail,
        None => {
            entry.base_address = ptr::null_mut();
            return false;
        }
    };
    let max_file = if entry.has_pawns { 3usize } else { 0usize };
    tail = match set_dtz_map(entry, tail, max_file, end) {
        Some(tail) => tail,
        None => {
            entry.base_address = ptr::null_mut();
            return false;
        }
    };
    if !set_table_pointers(entry, tail, end) {
        entry.base_address = ptr::null_mut();
        return false;
    }
    entry.mapping = file.mapping.take().map(Box::new);

    true
}

fn probe_table_wdl(tables: &TBTables, pos: &Position, result: &mut ProbeState) -> WDLScore {
    if crate::bitboard_fast::popcount(pos.pieces()) == 2 {
        return WDLScore::WDLDraw;
    }
    // Borrow from the retained table set; OnceLock synchronizes first mapping only.
    let owner = tables.get_wdl(pos.material_key());
    if let Some(owner) = owner {
        if let Some(entry) = owner.get(pos) {
            return do_probe_table_wdl(pos, entry, result);
        }
    }
    *result = ProbeState::Fail;
    WDLScore::WDLDraw
}

fn probe_table_dtz(
    tables: &TBTables,
    pos: &Position,
    wdl: WDLScore,
    result: &mut ProbeState,
) -> i32 {
    if crate::bitboard_fast::popcount(pos.pieces()) == 2 {
        return 0;
    }
    let owner = tables.get_dtz(pos.material_key());
    if let Some(owner) = owner {
        if let Some(entry) = owner.get(pos) {
            return do_probe_table_dtz(pos, entry, wdl, result);
        }
    }
    *result = ProbeState::Fail;
    0
}

/// Populate a table's `PairsData` records from the just-mapped file.
///
/// Port of C++ `set` (tbprobe.cpp). `TYPE` is 0 for WDL, 1 for DTZ; the DTZ
/// value map is handled by the caller since only DTZ tables have one.
fn set_table<const TYPE: usize>(
    entry: &mut TBTable<TYPE>,
    data: *const u8,
    end: *const u8,
) -> Option<*const u8> {
    const HAS_PAWNS: u8 = 2;
    const SPLIT: u8 = 1;

    // SAFETY: `data..end` is the unparsed remainder of the live mapping (`data` was produced by
    // the file-header check and `end` is the mapping end); `entry` holds the mapping for the
    // duration of the call.
    let buf = unsafe { file_slice(data, end)? };
    let mut pos = 0usize;

    let flags = *buf.first()?;
    if entry.has_pawns != (flags & HAS_PAWNS != 0)
        || (entry.key != entry.key2) != (flags & SPLIT != 0)
    {
        return None;
    }
    pos += 1; // first byte holds the flags

    // WDL tables store both sides when the two material keys differ.
    let sides = if TYPE == 0 && entry.key != entry.key2 {
        2
    } else {
        1
    };
    let max_file = if entry.has_pawns { 3usize } else { 0usize };

    // Encode remaining pawns and then pieces according to square, in ascending order
    let pp = entry.has_pawns && entry.pawn_count[1] != 0;
    debug_assert!(!pp || entry.pawn_count[0] != 0);

    let piece_count = usize::try_from(entry.piece_count).ok()?;
    let has_pawns = entry.has_pawns;
    let has_unique = entry.has_unique_pieces;
    let pawn_count = entry.pawn_count;

    if pp && pawn_count[0] == 0 {
        return None;
    }

    for f in 0..=max_file {
        for i in 0..sides {
            *entry.get_mut(i, f) = PairsData::default();
        }

        let hdr = buf.get(pos..pos + 1 + pp as usize + piece_count)?;
        let second = if pp { hdr[1] } else { 0xFF };
        let order = [
            [
                (hdr[0] & 0xF) as i32,
                if pp { (second & 0xF) as i32 } else { 0xF },
            ],
            [
                (hdr[0] >> 4) as i32,
                if pp { (second >> 4) as i32 } else { 0xF },
            ],
        ];
        let piece_bytes = &hdr[1 + pp as usize..];
        pos += hdr.len();

        for (k, &raw) in piece_bytes.iter().enumerate() {
            for i in 0..sides {
                entry.get_mut(i, f).pieces[k] =
                    Piece::from(if i != 0 { raw >> 4 } else { raw & 0xF });
            }
        }

        for i in 0..sides {
            if !set_groups(
                entry.get_mut(i, f),
                [order[i][0], order[i][1]],
                f,
                has_pawns,
                has_unique,
                piece_count as i32,
                pawn_count,
            ) {
                return None;
            }
        }
    }

    // Word alignment, by absolute address as in C++.
    let padding = (buf.as_ptr() as usize + pos) & 1;
    if pos + padding > buf.len() {
        return None;
    }
    pos += padding;

    let mut data = buf[pos..].as_ptr();
    for f in 0..=max_file {
        for i in 0..sides {
            data = set_sizes(entry.get_mut(i, f), data, end)?;
        }
    }

    Some(data)
}

/// Assign the sparse-index, block-length and data pointers that follow the
/// pairs data. Port of the tail of C++ `set` (tbprobe.cpp).
fn set_table_pointers<const TYPE: usize>(
    entry: &mut TBTable<TYPE>,
    mut data: *const u8,
    end: *const u8,
) -> bool {
    // SAFETY: Each `data.add(bytes)` is preceded by `fits(data, bytes, 1, end)` with checked
    // multiplication, so every pointer stored in PairsData lies within [mapping, end]. They are
    // only dereferenced under the same bounds in decompress_pairs; SparseEntry is packed (alignment
    // 1) and the u16 data is read bytewise.
    unsafe {
        let sides = if TYPE == 0 && entry.key != entry.key2 {
            2
        } else {
            1
        };
        let max_file = if entry.has_pawns { 3usize } else { 0usize };

        for f in 0..=max_file {
            for i in 0..sides {
                let d = entry.get_mut(i, f);
                let bytes = match d
                    .sparse_index_size
                    .checked_mul(core::mem::size_of::<SparseEntry>())
                {
                    Some(bytes) => bytes,
                    None => return false,
                };
                if !fits(data, bytes, 1, end) {
                    return false;
                }
                d.sparse_index = data as *const SparseEntry;
                data = data.add(bytes);
            }
        }

        for f in 0..=max_file {
            for i in 0..sides {
                let d = entry.get_mut(i, f);
                let bytes =
                    match (d.block_length_size as usize).checked_mul(core::mem::size_of::<u16>()) {
                        Some(bytes) => bytes,
                        None => return false,
                    };
                if !fits(data, bytes, 1, end) {
                    return false;
                }
                d.block_length = data as *const u16;
                data = data.add(bytes);
            }
        }

        for f in 0..=max_file {
            for i in 0..sides {
                // 64 byte alignment
                let pad = data.align_offset(64);
                if pad == usize::MAX || !fits(data, pad, 1, end) {
                    return false;
                }
                data = data.add(pad);
                let d = entry.get_mut(i, f);
                let bytes = match (d.blocks_num as usize).checked_mul(d.sizeof_block) {
                    Some(bytes) => bytes,
                    None => return false,
                };
                if !fits(data, bytes, 1, end) {
                    return false;
                }
                d.data = data;
                data = data.add(bytes);
                d.data_end = data;
            }
        }

        true
    }
}

/// Build the `.rtbw`/`.rtbz` filename for `pos`, matching C++ `mapped`
/// (tbprobe.cpp): pieces of each colour in decreasing order, with the side
/// whose material key matches the table's `key` written first.
fn tb_filename(entry_key: Key, pos: &Position, wdl: bool) -> String {
    const PIECE_CHARS: [char; 7] = [' ', 'P', 'N', 'B', 'R', 'Q', 'K'];
    let mut w = String::new();
    let mut b = String::new();

    for pt in [
        PieceType::King,
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
        PieceType::Pawn,
    ] {
        let ch = PIECE_CHARS[pt as usize];
        for _ in 0..pos.count_of(Color::White, pt) {
            w.push(ch);
        }
        for _ in 0..pos.count_of(Color::Black, pt) {
            b.push(ch);
        }
    }

    let body = if entry_key == pos.material_key() {
        format!("{w}v{b}")
    } else {
        format!("{b}v{w}")
    };

    format!("{body}{}", if wdl { ".rtbw" } else { ".rtbz" })
}

/// Colours for the DFS in `set_symlen`.
const SYM_WHITE: u8 = 0;
const SYM_GREY: u8 = 1;
const SYM_BLACK: u8 = 2;

/// Expand one Huffman symbol's length by walking the recursive-pairing tree.
///
/// Port of C++ `set_symlen` (tbprobe.cpp). Each symbol is a pair of child
/// symbols; a right child of 0xFFF marks a leaf.
fn set_symlen(btree: &[LR], symlen: &mut [u8], s: Sym, colour: &mut [u8], cyclic: &mut bool) -> u8 {
    let index = s as usize;
    if index >= SYM_COUNT {
        *cyclic = true;
        return 0;
    }

    colour[index] = SYM_GREY;

    let node = &btree[index]; // `btree` has SYM_COUNT entries and index < SYM_COUNT
    let sr = node.get_right();

    if sr == 0xFFF {
        colour[index] = SYM_BLACK;
        return 0;
    }

    let sl = node.get_left();
    let left = sl as usize;
    let right = sr as usize;
    if left >= SYM_COUNT || right >= SYM_COUNT {
        *cyclic = true;
        colour[index] = SYM_BLACK;
        return 0;
    }

    if colour[left] == SYM_GREY || colour[right] == SYM_GREY {
        *cyclic = true;
        colour[index] = SYM_BLACK;
        return 0;
    }

    if colour[left] == SYM_WHITE {
        symlen[left] = set_symlen(btree, symlen, sl, colour, cyclic);
    }
    if colour[right] == SYM_WHITE {
        symlen[right] = set_symlen(btree, symlen, sr, colour, cyclic);
    }

    colour[index] = SYM_BLACK;
    symlen[left].saturating_add(symlen[right]).saturating_add(1)
}

/// Compute the group lengths and per-group index multipliers.
///
/// Port of C++ `set_groups` (tbprobe.cpp). Takes the table's scalars
/// rather than the table itself so it can borrow one `PairsData` mutably.
///
/// Returns false when the file-derived group lengths or order cannot describe a valid
/// table (a group longer than the index tables allow, or more squares used than the board
/// has), so a corrupt file is rejected at load time instead of indexing out of bounds.
fn set_groups(
    d: &mut PairsData,
    order: [i32; 2],
    f: usize,
    has_pawns: bool,
    has_unique_pieces: bool,
    piece_count: i32,
    pawn_count: [i32; 2],
) -> bool {
    // Largest group length the binomial and leading-pawn tables have a row for.
    const MAX_GROUP_LEN: i32 = 5;

    let binomial = BINOMIAL.get().expect("binomial table initialized");
    let lead_pawns_size = LEAD_PAWNS_SIZE.get().expect("lead pawn sizes initialized");

    let mut n = 0usize;
    let mut first_len: i32 = if has_pawns {
        0
    } else if has_unique_pieces {
        3
    } else {
        2
    };
    d.group_len[0] = 1;

    for i in 1..piece_count as usize {
        first_len -= 1;
        if first_len > 0 || d.pieces[i] == d.pieces[i - 1] {
            d.group_len[n] += 1;
        } else {
            n += 1;
            d.group_len[n] = 1;
        }
    }
    n += 1;
    d.group_len[n] = 0; // zero-terminated

    // The sequence in pieces[] defines the groups, but not the order in which
    // they are encoded. If the pieces in a group g can be combined on the board
    // in N(g) different ways, then the position encoding will be of the form:
    //
    // g1 * N(g2) * N(g3) + g2 * N(g3) + g3
    //
    // This ensures unique encoding for the whole position. The order of the
    // groups is a per-table parameter and need not follow the canonical order of leading
    // pawns/pieces, then remaining pawns, then remaining pieces. In particular the
    // first group is at order[0] position and the remaining pawns, when present,
    // are at order[1] position.
    let pp = has_pawns && pawn_count[1] != 0;
    let mut next = if pp { 2usize } else { 1usize };
    if d.group_len[..=n].iter().any(|&len| len > MAX_GROUP_LEN) {
        return false;
    }
    let mut free_squares = 64 - d.group_len[0] - if pp { d.group_len[1] } else { 0 };
    let mut idx: u64 = 1;

    let mut k: i32 = 0;
    while next < n || k == order[0] || k == order[1] {
        if k == order[0] {
            // Leading pawns or pieces
            d.group_idx[0] = idx;
            idx *= if has_pawns {
                lead_pawns_size[d.group_len[0] as usize][f] as u64
            } else if has_unique_pieces {
                31332
            } else {
                462
            };
        } else if k == order[1] {
            // Remaining pawns
            d.group_idx[1] = idx;
            idx *= binomial[d.group_len[1] as usize][(48 - d.group_len[0]) as usize] as u64;
        } else {
            // Remaining pieces
            d.group_idx[next] = idx;
            if free_squares < 0 {
                return false;
            }
            idx *= binomial[d.group_len[next] as usize][free_squares as usize] as u64;
            free_squares -= d.group_len[next];
            next += 1;
        }
        k += 1;
    }

    d.group_idx[n] = idx;
    true
}

/// Read a `PairsData`'s compression parameters out of the mapped file.
///
/// Port of C++ `set_sizes` (tbprobe.cpp). Returns the pointer just past
/// the data it consumed, or `None` for a malformed/truncated table.
fn set_sizes(d: &mut PairsData, data: *const u8, end: *const u8) -> Option<*const u8> {
    // SAFETY: `data..end` is the unparsed remainder of the live mapping, held by the table whose
    // PairsData is being filled.
    let buf = unsafe { file_slice(data, end)? };
    let mut pos = 0usize;

    d.flags = *buf.first()?;
    pos += 1;

    if d.flags & TBFlag::SingleValue as u8 != 0 {
        d.blocks_num = 0;
        d.block_length_size = 0;
        d.span = 0;
        d.sparse_index_size = 0;
        d.min_sym_len = *buf.get(pos)?; // here it stores the single value
        return Some(buf[pos + 1..].as_ptr());
    }

    // group_len[] is zero-terminated; the matching group_idx[] entry holds
    // the largest index, i.e. the table size.
    let term = d.group_len.iter().take(7).position(|&x| x == 0)?;
    let tb_size = d.group_idx[term];

    // Fixed header: sizeof_block, span, padding, blocks_num (u32), max_sym_len, min_sym_len.
    let hdr = buf.get(pos..pos + 9)?;
    if hdr[0] >= usize::BITS as u8 || hdr[1] >= usize::BITS as u8 {
        return None;
    }
    d.sizeof_block = 1usize << hdr[0];
    d.span = 1usize << hdr[1];
    d.sparse_index_size = tb_size
        .checked_add(d.span as u64 - 1)?
        .checked_div(d.span as u64)? as usize;
    let padding = hdr[2];
    d.blocks_num = u32::from_le_bytes([hdr[3], hdr[4], hdr[5], hdr[6]]);
    // Padded so SparseIndex[] cannot point out of range.
    d.block_length_size = d.blocks_num.checked_add(padding as u32)?;
    d.max_sym_len = hdr[7];
    d.min_sym_len = hdr[8];
    pos += 9;
    if d.min_sym_len == 0 || d.min_sym_len > d.max_sym_len || d.max_sym_len > 32 {
        return None;
    }

    let base64_size = (d.max_sym_len - d.min_sym_len + 1) as usize;
    let lowest = buf.get(pos..pos + base64_size * core::mem::size_of::<Sym>())?;
    // Kept as a raw pointer for the hot path, which reads it with unaligned loads; the range
    // was just bounds-checked.
    d.lowest_sym = lowest.as_ptr().cast::<Sym>();
    let lowest_sym =
        |i: usize| -> u64 { Sym::from_le_bytes([lowest[2 * i], lowest[2 * i + 1]]) as u64 };
    d.base64 = vec![0u64; base64_size];

    // Canonical Huffman: longer codes have lower numeric values, so
    // base64[i] >= base64[i+1]. Build from the back.
    for i in (0..base64_size.saturating_sub(1)).rev() {
        let lo_i = lowest_sym(i);
        let lo_i1 = lowest_sym(i + 1);
        if lo_i < lo_i1 {
            return None;
        }
        d.base64[i] = (d.base64[i + 1] + lo_i - lo_i1) / 2;
        if d.base64[i].saturating_mul(2) < d.base64[i + 1] {
            return None;
        }
    }

    // Left-shift so each length is right-padded to 64 bits.
    for i in 0..base64_size {
        d.base64[i] <<= 64 - i - d.min_sym_len as usize;
    }

    pos += lowest.len();
    let symlen_len = u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?) as usize;
    pos += 2;
    if symlen_len > SYM_COUNT {
        return None;
    }
    let btree_bytes = buf.get(pos..pos + symlen_len * core::mem::size_of::<LR>())?;
    d.btree_storage = vec![
        LR {
            lr: [0x00, 0xF0, 0xFF]
        };
        SYM_COUNT
    ];
    let (nodes, _) = btree_bytes.as_chunks::<3>();
    for (node, raw) in d.btree_storage.iter_mut().zip(nodes) {
        node.lr = *raw;
    }
    d.btree = d.btree_storage.as_ptr();
    // The compression scheme used is "Recursive Pairing", that replaces the most
    // frequent adjacent pair of symbols in the source message by a new symbol,
    // reevaluating the frequencies of all of the symbol pairs with respect to
    // the extended alphabet, and then repeating the process.
    // See https://web.archive.org/web/20201106232444/http://www.larsson.dogma.net/dcc99.pdf
    d.symlen = vec![0u8; SYM_COUNT];

    // "Recursive Pairing": expand every symbol into its two children until
    // reaching leaves.
    let mut colour = vec![SYM_WHITE; SYM_COUNT];
    let mut cyclic = false;
    for sym in 0..symlen_len {
        if colour[sym] == SYM_WHITE {
            d.symlen[sym] = set_symlen(
                &d.btree_storage,
                &mut d.symlen,
                sym as Sym,
                &mut colour,
                &mut cyclic,
            );
        }
    }
    if cyclic {
        return None;
    }

    pos += btree_bytes.len();
    // Pad to a word boundary relative to the (word-aligned) table start.
    let padding = symlen_len & 1;
    if pos + padding > buf.len() {
        return None;
    }
    Some(buf[pos + padding..].as_ptr())
}

/// DTZ tables carry a value-remapping table after the pairs data; WDL tables do not.
///
/// Port of C++ `set_dtz_map` (tbprobe.cpp/1158).
fn set_dtz_map(
    entry: &mut TBTable<1>,
    data: *const u8,
    max_file: usize,
    end: *const u8,
) -> Option<*const u8> {
    // SAFETY: `data..end` is the unparsed remainder of the live mapping, held by `entry`.
    let buf = unsafe { file_slice(data, end)? };
    // `entry.map` addresses the start of `buf`; map_score_dtz bounds its reads by `map_len`.
    entry.map = data as *mut u8;
    entry.map_len = buf.len();
    let mut pos = 0usize;

    for f in 0..=max_file {
        let flags = entry.get_mut(0, f).flags;
        if flags & TBFlag::Mapped as u8 != 0 {
            if flags & TBFlag::Wide as u8 != 0 {
                // Word alignment; the table may be mixed.
                let padding = (buf.as_ptr() as usize + pos) & 1;
                if pos + padding > buf.len() {
                    return None;
                }
                pos += padding;
                for i in 0..4 {
                    let count = u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?);
                    entry.get_mut(0, f).map_idx[i] = u16::try_from(pos / 2 + 1).ok()?;
                    let step = 2usize.checked_mul(count as usize)?.checked_add(2)?;
                    if pos + step > buf.len() {
                        return None;
                    }
                    pos += step;
                }
            } else {
                for i in 0..4 {
                    let count = *buf.get(pos)?;
                    entry.get_mut(0, f).map_idx[i] = u16::try_from(pos + 1).ok()?;
                    let step = count as usize + 1;
                    if pos + step > buf.len() {
                        return None;
                    }
                    pos += step;
                }
            }
        }
    }

    let padding = (buf.as_ptr() as usize + pos) & 1;
    if pos + padding > buf.len() {
        return None;
    }
    Some(buf[pos + padding..].as_ptr()) // word alignment
}

// Public API

/// Publish a freshly built table set.
///
/// `init` runs again whenever `SyzygyPath` changes, so this cannot simply `set`
/// the `OnceLock` a second time: it replaces the contents of the existing one
/// instead. C++ likewise clears and refills its `TBTables` in place.
fn publish_tables(tables: TBTables) {
    let mut published = TB_TABLES.lock().unwrap();
    MAX_CARDINALITY.store(tables.max_cardinality, AtomicOrdering::Relaxed);
    *published = Arc::new(tables);
}

/// Called at startup and after every change to
/// "SyzygyPath" UCI option to publish a complete immutable table set. Existing
/// searches retain their previous set until they finish.
pub fn init(paths: &str) {
    let mut tables = TBTables::new();

    if paths.is_empty() {
        publish_tables(tables);
        return;
    }

    // Initialize mapping arrays
    initialize_maps();

    // Add all possible tablebase configurations
    // Add entries in TB tables if the corresponding ".rtbw" file exists
    for p1 in PieceType::Pawn as u8..PieceType::King as u8 {
        let p1 = PieceType::from(p1);
        tables.add(vec![PieceType::King, p1, PieceType::King], paths);

        for p2 in PieceType::Pawn as u8..=p1 as u8 {
            let p2 = PieceType::from(p2);
            tables.add(vec![PieceType::King, p1, p2, PieceType::King], paths);
            tables.add(vec![PieceType::King, p1, PieceType::King, p2], paths);

            for p3 in PieceType::Pawn as u8..PieceType::King as u8 {
                let p3 = PieceType::from(p3);
                tables.add(vec![PieceType::King, p1, p2, PieceType::King, p3], paths);
            }

            for p3 in PieceType::Pawn as u8..=p2 as u8 {
                let p3 = PieceType::from(p3);
                tables.add(vec![PieceType::King, p1, p2, p3, PieceType::King], paths);

                for p4 in PieceType::Pawn as u8..=p3 as u8 {
                    let p4 = PieceType::from(p4);
                    tables.add(
                        vec![PieceType::King, p1, p2, p3, p4, PieceType::King],
                        paths,
                    );

                    for p5 in PieceType::Pawn as u8..=p4 as u8 {
                        let p5 = PieceType::from(p5);
                        tables.add(
                            vec![PieceType::King, p1, p2, p3, p4, p5, PieceType::King],
                            paths,
                        );
                    }

                    for p5 in PieceType::Pawn as u8..PieceType::King as u8 {
                        let p5 = PieceType::from(p5);
                        tables.add(
                            vec![PieceType::King, p1, p2, p3, p4, PieceType::King, p5],
                            paths,
                        );
                    }
                }

                for p4 in PieceType::Pawn as u8..PieceType::King as u8 {
                    let p4 = PieceType::from(p4);
                    tables.add(
                        vec![PieceType::King, p1, p2, p3, PieceType::King, p4],
                        paths,
                    );

                    for p5 in PieceType::Pawn as u8..=p4 as u8 {
                        let p5 = PieceType::from(p5);
                        tables.add(
                            vec![PieceType::King, p1, p2, p3, PieceType::King, p4, p5],
                            paths,
                        );
                    }
                }
            }

            for p3 in PieceType::Pawn as u8..=p1 as u8 {
                let p3 = PieceType::from(p3);
                for p4 in PieceType::Pawn as u8..=if p1 == p3 { p2 as u8 } else { p3 as u8 } {
                    let p4 = PieceType::from(p4);
                    tables.add(
                        vec![PieceType::King, p1, p2, PieceType::King, p3, p4],
                        paths,
                    );
                }
            }
        }
    }

    tables.info();
    publish_tables(tables);
}

fn initialize_maps() {
    static READY: OnceLock<()> = OnceLock::new();
    READY.get_or_init(initialize_maps_inner);
}

fn initialize_maps_inner() {
    // Initialize MapB1H1H7
    let mut map_b1h1h7 = [0i32; SQUARE_NB];
    let mut code = 0;
    for sq in 0..SQUARE_NB {
        if off_a1h8(Square::from(sq as u8)) < 0 {
            map_b1h1h7[sq] = code;
            code += 1;
        }
    }
    MAP_B1H1H7.set(map_b1h1h7).unwrap();

    // Initialize MapA1D1D4
    let mut map_a1d1d4 = [0i32; SQUARE_NB];
    let mut diagonal = Vec::new();
    code = 0;

    for sq in Square::SQ_A1 as u8..=Square::SQ_D4 as u8 {
        let square = Square::from(sq);
        if off_a1h8(square) < 0 && file_of(square) <= File::D {
            map_a1d1d4[sq as usize] = code;
            code += 1;
        } else if off_a1h8(square) == 0 && file_of(square) <= File::D {
            diagonal.push(sq);
        }
    }

    for sq in diagonal {
        map_a1d1d4[sq as usize] = code;
        code += 1;
    }
    MAP_A1D1D4.set(map_a1d1d4).unwrap();

    // Initialize MapKK
    let mut map_kk = [[0i32; SQUARE_NB]; 10];
    let mut both_on_diagonal = Vec::new();

    code = 0;
    for idx in 0..10 {
        for s1 in Square::SQ_A1 as u8..=Square::SQ_D4 as u8 {
            let sq1 = Square::from(s1);
            if map_a1d1d4[s1 as usize] != idx as i32 || (idx == 0 && sq1 != Square::SQ_B1) {
                continue;
            }

            for s2 in Square::SQ_A1 as u8..=Square::SQ_H8 as u8 {
                let sq2 = Square::from(s2);

                // Check if positions are legal (kings not adjacent)
                if crate::bitboard_fast::distance_square(sq1, sq2) <= 1 {
                    continue;
                }

                if off_a1h8(sq1) == 0 && off_a1h8(sq2) > 0 {
                    continue; // First on diagonal, second above
                }

                if off_a1h8(sq1) == 0 && off_a1h8(sq2) == 0 {
                    both_on_diagonal.push((idx, s2));
                } else {
                    map_kk[idx][s2 as usize] = code;
                    code += 1;
                }
            }
        }
    }

    for (idx, s2) in both_on_diagonal {
        map_kk[idx][s2 as usize] = code;
        code += 1;
    }
    MAP_KK.set(map_kk).unwrap();

    // Initialize Binomial coefficients
    let mut binomial = [[0i32; SQUARE_NB]; 6];
    binomial[0][0] = 1;

    for n in 1..64 {
        for k in 0..6 {
            if k <= n {
                binomial[k][n] = if k > 0 { binomial[k - 1][n - 1] } else { 0 }
                    + if k < n { binomial[k][n - 1] } else { 0 };
            }
        }
    }
    BINOMIAL.set(binomial).unwrap();

    // Initialize MapPawns
    let mut map_pawns = [0i32; SQUARE_NB];
    let mut available_squares = 47;

    for f in File::A as u8..=File::D as u8 {
        for r in Rank::R2 as u8..=Rank::R7 as u8 {
            let sq = make_square(File::from(f), Rank::from(r));
            map_pawns[sq as usize] = available_squares;
            available_squares -= 1;
            map_pawns[flip_file(sq) as usize] = available_squares;
            available_squares -= 1;
        }
    }
    MAP_PAWNS.set(map_pawns).unwrap();

    // Initialize lead pawn indices
    let mut lead_pawn_idx = [[0i32; SQUARE_NB]; 6];
    let mut lead_pawns_size = [[0i32; 4]; 6];

    for lead_pawns_cnt in 1..=5 {
        for f in 0..4 {
            let mut idx = 0;

            for r in Rank::R2 as u8..=Rank::R7 as u8 {
                let sq = make_square(File::from(f as u8), Rank::from(r));
                lead_pawn_idx[lead_pawns_cnt][sq as usize] = idx;
                idx += binomial[lead_pawns_cnt - 1][map_pawns[sq as usize] as usize];
            }

            lead_pawns_size[lead_pawns_cnt][f] = idx;
        }
    }

    LEAD_PAWN_IDX.set(lead_pawn_idx).unwrap();
    LEAD_PAWNS_SIZE.set(lead_pawns_size).unwrap();
}

/// Probe the WDL table for a particular position.
/// If *result != FAIL, the probe was successful.
/// The return value is from the point of view of the side to move:
/// -2 : loss
/// -1 : loss, but draw under 50-move rule
/// 0 : draw
/// 1 : win, but draw under 50-move rule
/// 2 : win
pub fn probe_wdl(pos: &mut Position, result: &mut ProbeState) -> WDLScore {
    TablebaseSnapshot::current().probe_wdl(pos, result)
}

fn probe_wdl_with(tables: &TBTables, pos: &mut Position, result: &mut ProbeState) -> WDLScore {
    *result = ProbeState::Ok;
    search_wdl(tables, pos, result)
}

/// Probe the DTZ table for a particular position.
/// If *result != FAIL, the probe was successful.
/// The return value is from the point of view of the side to move:
/// n < -100 : loss, but draw under 50-move rule
/// -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
/// -1        : loss, the side to move is mated
/// 0        : draw
/// 1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
/// 100 < n        : win, but draw under 50-move rule
///
/// The return value n can be off by 1: a return value -n can mean a loss
/// in n+1 ply and a return value +n can mean a win in n+1 ply. This
/// cannot happen for tables with positions exactly on the "edge" of
/// the 50-move rule.
///
/// This implies that if dtz > 0 is returned, the position is certainly
/// a win if dtz + 50-move-counter <= 99. Care must be taken that the engine
/// picks moves that preserve dtz + 50-move-counter <= 99.
///
/// If n = 100 immediately after a capture or pawn move, then the position
/// is also certainly a win, and during the whole phase until the next
/// capture or pawn move, the inequality to be preserved is
/// dtz + 50-move-counter <= 100.
///
/// In short, if a move is available resulting in dtz + 50-move-counter <= 99,
/// then do not accept moves leading to dtz + 50-move-counter == 100.
pub fn probe_dtz(pos: &mut Position, result: &mut ProbeState) -> i32 {
    TablebaseSnapshot::current().probe_dtz(pos, result)
}

fn probe_dtz_with(tables: &TBTables, pos: &mut Position, result: &mut ProbeState) -> i32 {
    *result = ProbeState::Ok;
    let wdl = search_dtz(tables, pos, result);

    if *result == ProbeState::Fail || wdl == WDLScore::WDLDraw {
        return 0;
    }

    // DTZ stores a don't-care value here, or even a wrong one when the best move
    // is a losing en passant capture, so it cannot be probed.
    if *result == ProbeState::ZeroingBestMove {
        return dtz_before_zeroing(wdl);
    }

    let dtz = probe_table_dtz(tables, pos, wdl, result);

    if *result == ProbeState::Fail {
        return 0;
    }

    if *result != ProbeState::ChangeSTM {
        return (dtz
            + 100
                * if wdl == WDLScore::WDLBlessedLoss || wdl == WDLScore::WDLCursedWin {
                    1
                } else {
                    0
                })
            * sign_of(wdl as i32);
    }

    // Need to do 1-ply search for other side
    let mut min_dtz = 0xFFFF;
    let moves = generate(pos, GenType::All);

    for &m in &moves {
        let zeroing = pos.capture(m) || type_of(pos.moved_piece(m)) == PieceType::Pawn;

        // SAFETY: the move was generated as legal; the guard undoes it,
        // including on unwind.
        let mut moved = unsafe { ScopedMove::new(pos, m) };
        let pos = &mut *moved;

        let dtz = if zeroing {
            // For zeroing moves we want the dtz of the move _before_ doing it,
            // otherwise we will get the dtz of the next move sequence. Search the
            // position after the move to get the score sign (because even in a
            // winning position we could make a losing capture or go for a draw).
            -dtz_before_zeroing(search_wdl(tables, pos, result))
        } else {
            -probe_dtz_with(tables, pos, result)
        };

        // Check for mate
        if dtz == 1 && pos.checkers() != 0 && generate(pos, GenType::All).is_empty() {
            min_dtz = 1;
        }

        let mut final_dtz = dtz;
        if !zeroing {
            final_dtz += sign_of(dtz);
        }

        // Skip the draws and if we are winning only pick positive dtz
        if final_dtz < min_dtz && sign_of(final_dtz) == sign_of(wdl as i32) {
            min_dtz = final_dtz;
        }

        drop(moved);

        if *result == ProbeState::Fail {
            return 0;
        }
    }

    // When there are no legal moves, the position is mate: we return -1
    if min_dtz == 0xFFFF { -1 } else { min_dtz }
}

/// Use the DTZ tables to rank root moves.
///
/// A return value false indicates that not all probes were successful.
pub fn root_probe(
    pos: &mut Position,
    root_moves: &mut RootMoves,
    rule50: bool,
    rank_dtz: bool,
) -> bool {
    let tables = TablebaseSnapshot::current();
    root_probe_with(&tables.0, pos, root_moves, rule50, rank_dtz)
}

fn root_probe_with(
    tables: &TBTables,
    pos: &mut Position,
    root_moves: &mut RootMoves,
    rule50: bool,
    rank_dtz: bool,
) -> bool {
    let mut result = ProbeState::Ok;
    // Obtain 50-move counter for the root position
    let cnt50 = pos.rule50_count();
    // Check whether a position was repeated since the last zeroing move.
    let rep = pos.has_repeated();
    let bound = if rule50 { MAX_DTZ / 2 - 100 } else { 1 };

    for m in root_moves.iter_mut() {
        let Some(&mv) = m.pv.first() else {
            return false;
        };
        if !pos.legal(mv) {
            return false;
        }
        // SAFETY: legality is checked above; the guard restores the
        // position, including on unwind.
        let mut moved = unsafe { ScopedMove::new(pos, mv) };
        let pos = &mut *moved;

        // Calculate dtz for the current move counting from the root position
        let dtz = if pos.rule50_count() == 0 {
            // In case of a zeroing move, dtz is one of -101/-1/0/1/101
            let wdl = -probe_wdl_with(tables, pos, &mut result);
            dtz_before_zeroing(wdl)
        } else if (rule50 && pos.is_draw(1)) || pos.is_repetition(1) {
            0
        } else {
            // Otherwise, take dtz for the new position and correct by 1 ply
            let dtz = -probe_dtz_with(tables, pos, &mut result);
            if dtz > 0 {
                dtz + 1
            } else if dtz < 0 {
                dtz - 1
            } else {
                dtz
            }
        };

        // Check for mate
        let final_dtz = if pos.checkers() != 0 && dtz == 2 && generate(pos, GenType::All).is_empty()
        {
            1
        } else {
            dtz
        };

        drop(moved);

        if result == ProbeState::Fail {
            return false;
        }

        let r = if final_dtz > 0 {
            if final_dtz + cnt50 <= 99 && !rep {
                // Better moves are ranked higher. Certain wins are ranked equally.
                // Losing moves are ranked equally unless a 50-move draw is in sight.
                MAX_DTZ - if rank_dtz { final_dtz } else { 0 }
            } else {
                MAX_DTZ / 2 - (final_dtz + cnt50)
            }
        } else if final_dtz < 0 {
            if -final_dtz * 2 + cnt50 < 100 {
                -MAX_DTZ - if rank_dtz { final_dtz } else { 0 }
            } else {
                -MAX_DTZ / 2 + (-final_dtz + cnt50)
            }
        } else {
            0
        };

        m.tb_rank = r;

        m.tb_score = if r >= bound {
            VALUE_MATE - MAX_PLY - 1
        } else if r > 0 {
            Value::from(max(3, r - (MAX_DTZ / 2 - 200)) * PAWN_VALUE / 200)
        } else if r == 0 {
            VALUE_DRAW
        } else if r > -bound {
            Value::from(min(-3, r + (MAX_DTZ / 2 - 200)) * PAWN_VALUE / 200)
        } else {
            -VALUE_MATE + MAX_PLY + 1
        };
    }

    true
}

/// Use the WDL tables to rank root moves.
/// This is a fallback for the case that some or all DTZ tables are missing.
///
/// A return value false indicates that not all probes were successful.
pub fn root_probe_wdl(pos: &mut Position, root_moves: &mut RootMoves, rule50: bool) -> bool {
    let tables = TablebaseSnapshot::current();
    root_probe_wdl_with(&tables.0, pos, root_moves, rule50)
}

fn root_probe_wdl_with(
    tables: &TBTables,
    pos: &mut Position,
    root_moves: &mut RootMoves,
    rule50: bool,
) -> bool {
    const WDL_TO_RANK: [i32; 5] = [-MAX_DTZ, -MAX_DTZ + 101, 0, MAX_DTZ - 101, MAX_DTZ];
    const WDL_TO_VALUE: [Value; 5] = [
        -VALUE_MATE + MAX_PLY + 1,
        VALUE_DRAW - 2,
        VALUE_DRAW,
        VALUE_DRAW + 2,
        VALUE_MATE - MAX_PLY - 1,
    ];

    let mut result = ProbeState::Ok;

    for m in root_moves.iter_mut() {
        let Some(&mv) = m.pv.first() else {
            return false;
        };
        if !pos.legal(mv) {
            return false;
        }
        // SAFETY: legality is checked above; the guard restores the
        // position, including on unwind.
        let mut moved = unsafe { ScopedMove::new(pos, mv) };
        let pos = &mut *moved;

        let wdl = if pos.is_draw(1) {
            WDLScore::WDLDraw
        } else {
            -probe_wdl_with(tables, pos, &mut result)
        };

        drop(moved);

        if result == ProbeState::Fail {
            return false;
        }

        m.tb_rank = WDL_TO_RANK[(wdl as i32 + 2) as usize];

        let final_wdl = if !rule50 {
            if wdl > WDLScore::WDLDraw {
                WDLScore::WDLWin
            } else if wdl < WDLScore::WDLDraw {
                WDLScore::WDLLoss
            } else {
                WDLScore::WDLDraw
            }
        } else {
            wdl
        };

        m.tb_score = WDL_TO_VALUE[(final_wdl as i32 + 2) as usize];
    }

    true
}

/// Reads the Syzygy options, ranks the root moves by DTZ (falling back to WDL), and returns
/// the resulting [`Config`].
pub fn rank_root_moves(
    options: &OptionsMap,
    pos: &mut Position,
    root_moves: &mut RootMoves,
    mut rank_dtz: bool,
) -> Config {
    let mut config = Config::default();

    if root_moves.is_empty() {
        return config;
    }

    config.use_rule50 = options
        .get("Syzygy50MoveRule")
        .is_none_or(|opt| opt.is_true());
    config.probe_depth = options
        .get("SyzygyProbeDepth")
        .map_or(1, |opt| opt.value().parse().unwrap_or(1)) as Depth;
    config.cardinality = options
        .get("SyzygyProbeLimit")
        .map_or(7, |opt| opt.value().parse().unwrap_or(7));

    let tables = &config.tables.0;
    let max_cardinality = tables.max_cardinality;

    // Tables with fewer pieces than SyzygyProbeLimit are searched with
    // probeDepth == DEPTH_ZERO
    if config.cardinality > max_cardinality {
        config.cardinality = max_cardinality;
        config.probe_depth = 0;
    }

    let mut dtz_available = true;

    if config.cardinality >= crate::bitboard_fast::popcount(pos.pieces()) as i32
        && !pos.can_castle(CastlingRights::ANY_CASTLING)
    {
        // Use DTZ to rank the moves if checkmate is the only zeroing move
        rank_dtz = rank_dtz || pos.dtz_is_dtm();

        config.root_in_tb = root_probe_with(tables, pos, root_moves, config.use_rule50, rank_dtz);

        if !config.root_in_tb {
            dtz_available = false;
            config.root_in_tb = root_probe_wdl_with(tables, pos, root_moves, config.use_rule50);
        }
    }

    if config.root_in_tb {
        root_moves.sort_by_key(|m| core::cmp::Reverse(m.tb_rank));

        // Probe during search only if DTZ is not available and we are winning
        if dtz_available || root_moves[0].tb_score <= VALUE_DRAW {
            config.cardinality = 0;
        }
    } else {
        for m in root_moves.iter_mut() {
            m.tb_rank = 0;
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table may declare at most `SYM_COUNT` symbols, the size of `btree`, `symlen` and the
    /// DFS colours. A larger count is refused as corrupt even when its tree bytes are present.
    #[test]
    fn set_sizes_rejects_more_than_sym_count_symbols() {
        for (count, accepted) in [(SYM_COUNT, true), (SYM_COUNT + 1, false)] {
            // flags, sizeof_block, span, padding, blocks_num (u32), max_sym_len, min_sym_len,
            // one lowest_sym entry, the symbol count, then `count` leaf nodes (right = 0xFFF).
            let mut buf = vec![0u8, 5, 5, 0, 1, 0, 0, 0, 1, 1, 0, 0];
            buf.extend_from_slice(&(count as u16).to_le_bytes());
            for _ in 0..count {
                buf.extend_from_slice(&[0x00, 0xF0, 0xFF]);
            }
            buf.resize(buf.len() + (count & 1), 0);
            let range = buf.as_ptr_range();
            let mut d = PairsData::default();
            let end = set_sizes(&mut d, range.start, range.end);
            assert_eq!(end.is_some(), accepted, "{count} symbols");
            if accepted {
                assert_eq!(end, Some(range.end));
            }
        }
    }

    #[test]
    fn search_snapshot_probes_without_the_global_lock() {
        crate::bitboard_fast::init();
        initialize_maps();
        let mut tables = TBTables::new();
        tables.add(
            vec![PieceType::King, PieceType::Rook, PieceType::King],
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/syzygy"),
        );
        let snapshot = TablebaseSnapshot(Arc::new(tables));
        let (send, receive) = std::sync::mpsc::channel();
        let global = TB_TABLES.lock().unwrap();
        let child = std::thread::spawn(move || {
            for fen in [
                "7k/8/8/8/8/8/R7/K7 w - - 0 1",
                "8/8/8/8/8/3k4/3R4/7K b - - 0 1",
            ] {
                let mut position = Position::default();
                position.set_fen(fen, false).unwrap();
                for _ in 0..100 {
                    let mut result = ProbeState::Fail;
                    snapshot.probe_wdl(&mut position, &mut result);
                    assert!(matches!(
                        result,
                        ProbeState::Ok | ProbeState::ZeroingBestMove
                    ));
                }
            }
            send.send(()).unwrap();
        });
        let completed = receive.recv_timeout(std::time::Duration::from_secs(5));
        drop(global); // release even on failure, so the witness cannot deadlock the suite
        child.join().unwrap();
        completed.expect("retained search probes must not acquire the global table mutex");
    }

    #[test]
    fn concurrent_probes_retain_one_mapping_after_the_table_set_is_dropped() {
        crate::bitboard_fast::init();
        initialize_maps();
        let paths = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/syzygy");
        let mut tables = TBTables::new();
        tables.add(
            vec![PieceType::King, PieceType::Rook, PieceType::King],
            paths,
        );
        let signature = TBTable::<0>::from_code("KRvK");
        let wdl = tables
            .get_wdl(signature.key)
            .expect("required KRvK fixture")
            .clone();
        let dtz = tables.get_dtz(signature.key).unwrap().clone();
        drop(tables);
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let wdl = Arc::clone(&wdl);
                let dtz = Arc::clone(&dtz);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut pos = Position::default();
                    pos.set_fen("7k/8/8/8/8/8/R7/K7 w - - 0 1", false).unwrap();
                    barrier.wait();
                    let wdl = wdl.get(&pos).unwrap();
                    let dtz = dtz.get(&pos).unwrap();
                    for _ in 0..1000 {
                        let mut result = ProbeState::Ok;
                        assert_eq!(do_probe_table_wdl(&pos, wdl, &mut result), WDLScore::WDLWin);
                        assert_eq!(result, ProbeState::Ok);
                        assert!(do_probe_table_dtz(&pos, dtz, WDLScore::WDLWin, &mut result) > 0);
                        assert_eq!(result, ProbeState::Ok);
                    }
                    (wdl.base_address as usize, dtz.base_address as usize)
                })
            })
            .collect();
        let mappings: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(mappings.iter().all(|mapping| mapping == &mappings[0]));
    }
}
