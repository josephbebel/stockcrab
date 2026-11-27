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

//! History tables for move ordering - Rust port of Stockfish's history.h
//!
//! This module provides statistics tracking for move success/failure to improve
//! move ordering during search. Various history tables track different aspects
//! of move performance.
//!
//! Histories are shared between threads with atomic access.

use crate::misc::LargePageAllocator;
use crate::position::Position;
use crate::types::*;
use bytemuck::Zeroable;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicI16, Ordering};

// History table size constants
pub const PAWN_HISTORY_BASE_SIZE: usize = 8192; // Must be power of 2
pub const UINT_16_HISTORY_SIZE: usize = u16::MAX as usize + 1; // 65536
pub const CORRHIST_BASE_SIZE: usize = 2 * UINT_16_HISTORY_SIZE; // 131072
pub const CORRECTION_HISTORY_LIMIT: i32 = 1024;
pub const LOW_PLY_HISTORY_SIZE: usize = 5;

// Compile-time assertions for power of 2
const _: () = assert!(PAWN_HISTORY_BASE_SIZE.count_ones() == 1);
const _: () = assert!(CORRHIST_BASE_SIZE.count_ones() == 1);
const _: () = assert!(UINT_16_HISTORY_SIZE.count_ones() == 1);

// ---------------------------------------------------------------------------
// StatsEntry - Container for numerical statistics with bounded updates
// ---------------------------------------------------------------------------

/// One history counter (C++ `StatsEntry<T, D>`): updates use the history gravity formula,
/// which keeps the value within `[-D, D]`.
#[derive(Copy, Clone, Debug)]
pub struct StatsEntry<T, const D: i32> {
    entry: T,
}

impl<const D: i32> StatsEntry<i16, D> {
    pub const fn new() -> Self {
        StatsEntry { entry: 0 }
    }

    /// Update the stats entry with a bonus (clamped to [-D, D])
    pub fn update(&mut self, bonus: i32) {
        // Make sure that bonus is in range [-D, D]
        let clamped_bonus = bonus.clamp(-D, D);
        let entry_val = self.entry as i32;
        let new_val = entry_val + clamped_bonus - entry_val * clamped_bonus.abs() / D;
        // Like C++, rely on the invariant |entry| <= D <= i16::MAX being preserved by the update formula.
        debug_assert!((-32768..=32767).contains(&new_val));
        self.entry = new_val as i16;

        debug_assert!((self.entry as i32).abs() <= D);
    }

    /// Direct assignment of value
    pub fn set(&mut self, value: i16) {
        self.entry = value;
        debug_assert!((self.entry as i32).abs() <= D);
    }

    /// Get the current value
    pub fn get(&self) -> i16 {
        self.entry
    }
}

impl<T: Copy, const D: i32> Deref for StatsEntry<T, D> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.entry
    }
}

impl<T: Copy, const D: i32> DerefMut for StatsEntry<T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entry
    }
}

impl<T: Default + Copy, const D: i32> Default for StatsEntry<T, D> {
    fn default() -> Self {
        StatsEntry {
            entry: T::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// AtomicStatsEntry - Thread-safe version for shared histories
// Atomic access for thread-shared tables
// ---------------------------------------------------------------------------

/// Thread-safe stats entry using atomic operations
/// Used in SharedHistories for pawn history and correction history
#[derive(Debug, Zeroable)]
#[repr(transparent)]
pub struct AtomicStatsEntry<const D: i32> {
    entry: AtomicI16,
}

impl<const D: i32> AtomicStatsEntry<D> {
    pub const fn new() -> Self {
        AtomicStatsEntry {
            entry: AtomicI16::new(0),
        }
    }

    /// Get the current value (relaxed ordering for performance)
    #[inline]
    pub fn get(&self) -> i16 {
        self.entry.load(Ordering::Relaxed)
    }

    /// Set the value atomically
    #[inline]
    pub fn set(&self, value: i16) {
        self.entry.store(value, Ordering::Relaxed);
    }

    /// Update the stats entry with a bonus (clamped to [-D, D])
    /// Uses relaxed ordering as in C++ implementation
    #[inline]
    pub fn update(&self, bonus: i32) {
        let clamped_bonus = bonus.clamp(-D, D);
        let entry_val = self.get() as i32;
        let new_val = entry_val + clamped_bonus - entry_val * clamped_bonus.abs() / D;
        // Like C++, rely on the invariant |entry| <= D <= i16::MAX being preserved by the update formula.
        debug_assert!((-32768..=32767).contains(&new_val));
        self.set(new_val as i16);
    }
}

impl<const D: i32> Default for AtomicStatsEntry<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// `AtomicI16` cells per `UnifiedCorrectionEntry` (one bundle of five) and per
/// `AtomicPawnHistoryEntry`.
const CELLS_PER_CORRECTION_ENTRY: usize = 5;
const CELLS_PER_PAWN_ENTRY: usize = PIECE_NB * SQUARE_NB;
const _: () = assert!(
    core::mem::size_of::<UnifiedCorrectionEntry>()
        == CELLS_PER_CORRECTION_ENTRY * core::mem::size_of::<AtomicI16>()
);
const _: () = assert!(
    core::mem::size_of::<AtomicPawnHistoryEntry>()
        == CELLS_PER_PAWN_ENTRY * core::mem::size_of::<AtomicI16>()
);

/// Sets `n` consecutive `AtomicI16` cells starting at `first` to `value` with relaxed stores, so
/// a clear that overlaps a search or another clear is a logic error, not a data race.
///
/// # Safety
///
/// `first..first + n` must lie inside one allocation of `AtomicI16`-backed cells (an
/// `AtomicStatsEntry`, or a `CorrectionBundle`, which is `repr(C)` over five of them).
#[inline]
unsafe fn fill_cells(first: *const AtomicI16, n: usize, value: i16) {
    // SAFETY: the caller guarantees the range is `n` initialised `AtomicI16`s in one allocation;
    // shared references to atomics may coexist with any other access to them.
    let cells = unsafe { core::slice::from_raw_parts(first, n) };
    for cell in cells {
        cell.store(value, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// CorrectionBundle - Bundles pawn/minor/nonpawn corrections together
// Unified correction history structure
// ---------------------------------------------------------------------------

/// Bundle of atomic correction entries for a position
/// Groups pawn, minor, and non-pawn corrections together for cache efficiency
#[derive(Debug, Zeroable)]
#[repr(C)]
pub struct CorrectionBundle {
    /// Indexed by whether either side has a queen.
    pub pawn: [AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>; 2],
    pub minor: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
    pub non_pawn_white: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
    pub non_pawn_black: AtomicStatsEntry<CORRECTION_HISTORY_LIMIT>,
}

impl CorrectionBundle {
    pub const fn new() -> Self {
        CorrectionBundle {
            pawn: [AtomicStatsEntry::new(), AtomicStatsEntry::new()],
            minor: AtomicStatsEntry::new(),
            non_pawn_white: AtomicStatsEntry::new(),
            non_pawn_black: AtomicStatsEntry::new(),
        }
    }
}

impl Default for CorrectionBundle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Atomic pawn history entry - [piece][to] indexed
// Thread-safe pawn history
// ---------------------------------------------------------------------------

/// Atomic pawn history entry: [PIECE_NB][SQUARE_NB]
pub type AtomicPawnHistoryEntry = [[AtomicStatsEntry<8192>; SQUARE_NB]; PIECE_NB];

// ---------------------------------------------------------------------------
// Unified correction history entry - one CorrectionBundle
// Combined correction history
// ---------------------------------------------------------------------------

/// Unified correction entry. The side to move is not an index: it is the low bit of the
/// position keys that select the entry.
pub type UnifiedCorrectionEntry = CorrectionBundle;

/// Allocate `n` zero-filled `T`s as one contiguous block (so a table lookup is a single indexed
/// load instead of a pointer chase through per-entry boxes). `alloc_zeroed` underneath, so large
/// tables get fresh zero pages instead of a memset.
fn zeroed_vec<T: Zeroable>(n: usize) -> Vec<T> {
    bytemuck::allocation::try_zeroed_vec(n).expect("history table allocation failed")
}

// ---------------------------------------------------------------------------
// SharedHistories - Thread-shared history tables
// Shared between threads on a NUMA node
// ---------------------------------------------------------------------------

/// `thread_count` (at least 1) rounded up to a power of two, times `base` (a power of two): a
/// non-zero power of two, so that `key & (size - 1)` is always in bounds. Panics on overflow.
fn shared_table_size(thread_count: usize, base: usize) -> usize {
    debug_assert!(base.is_power_of_two());
    thread_count
        .max(1)
        .checked_next_power_of_two()
        .and_then(|n| n.checked_mul(base))
        .expect("shared history size overflows")
}

/// Shared pawn history: Vec of atomic pawn history entries
pub struct SharedPawnHistory {
    data: Vec<AtomicPawnHistoryEntry>,
    size: usize,
}

impl SharedPawnHistory {
    pub fn new(thread_count: usize) -> Self {
        // The size must be a power of two so that `key & (size - 1)` is always in bounds
        // (`SharedHistories` relies on it for unchecked lookups); round a stray thread count up.
        let size = shared_table_size(thread_count, PAWN_HISTORY_BASE_SIZE);
        let data: Vec<AtomicPawnHistoryEntry> = zeroed_vec(size);
        SharedPawnHistory { data, size }
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    /// Unchecked lookup for the hot path.
    ///
    /// # Safety
    ///
    /// `index < self.get_size()` (e.g. a key masked with `get_size() - 1`; the size is always a
    /// power of two).
    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, index: usize) -> &AtomicPawnHistoryEntry {
        debug_assert!(index < self.size);
        // SAFETY: by the caller's contract.
        unsafe { self.data.get_unchecked(index) }
    }
}

// Bounds-checked: the unchecked hot-path lookup is `get_unchecked`.
impl core::ops::Index<usize> for SharedPawnHistory {
    type Output = AtomicPawnHistoryEntry;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl core::ops::IndexMut<usize> for SharedPawnHistory {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Shared correction history: Vec of unified correction entries
pub struct SharedCorrectionHistory {
    data: Vec<UnifiedCorrectionEntry>,
    size: usize,
}

impl SharedCorrectionHistory {
    pub fn new(thread_count: usize) -> Self {
        // Power-of-two size: see `SharedPawnHistory::new`.
        let size = shared_table_size(thread_count, CORRHIST_BASE_SIZE);
        let data: Vec<UnifiedCorrectionEntry> = zeroed_vec(size);
        SharedCorrectionHistory { data, size }
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    /// Unchecked lookup for the hot path.
    ///
    /// # Safety
    ///
    /// `index < self.get_size()` (e.g. a key masked with `get_size() - 1`; the size is always a
    /// power of two).
    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, index: usize) -> &UnifiedCorrectionEntry {
        debug_assert!(index < self.size);
        // SAFETY: by the caller's contract.
        unsafe { self.data.get_unchecked(index) }
    }
}

// Bounds-checked: the unchecked hot-path lookup is `get_unchecked`.
impl core::ops::Index<usize> for SharedCorrectionHistory {
    type Output = UnifiedCorrectionEntry;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl core::ops::IndexMut<usize> for SharedCorrectionHistory {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Set of histories shared between groups of threads
/// To avoid excessive cross-node data transfer, histories are shared
/// only between threads on a given NUMA node.
pub struct SharedHistories {
    // Private: the lookups mask keys with sizes cached below, so the tables must not be replaced.
    correction_history: SharedCorrectionHistory,
    /// Continuation history is thread-shared and lives here (it is built on the
    /// atomic PieceToHistory). Indexed by \[in_check\]\[capture\].
    pub continuation_history_block: LargePageContinuationHistory,
    pawn_history: SharedPawnHistory,
    size_minus_1: usize,
    pawn_hist_size_minus_1: usize,
}

impl SharedHistories {
    /// Create new shared histories for a given thread count
    /// thread_count must be a power of 2
    pub fn new(thread_count: usize) -> Self {
        debug_assert!(
            thread_count > 0 && thread_count.count_ones() == 1,
            "thread_count must be a power of 2"
        );

        let correction_history = SharedCorrectionHistory::new(thread_count);
        let pawn_history = SharedPawnHistory::new(thread_count);

        let size_minus_1 = correction_history.get_size() - 1;
        let pawn_hist_size_minus_1 = pawn_history.get_size() - 1;

        SharedHistories {
            correction_history,
            continuation_history_block: LargePageContinuationHistory::new(),
            pawn_history,
            size_minus_1,
            pawn_hist_size_minus_1,
        }
    }

    /// Sets every entry of the four continuation-history tables to `value`
    /// (C++ `continuationHistory[inCheck][capture].fill(value)`). Only for the thread that owns
    /// the initialization of this block, while no search is running.
    pub fn fill_continuation_history(&self, value: i16) {
        const CELLS: usize =
            core::mem::size_of::<ContinuationHistoryBlock>() / core::mem::size_of::<AtomicI16>();
        // SAFETY: the block is `repr(C)` over nothing but `AtomicI16`-backed entries (`CELLS` of
        // them, checked below) and `table` borrows all of it, so the pointer covers the range.
        unsafe {
            fill_cells(
                core::ptr::from_ref(&self.continuation_history_block.table).cast::<AtomicI16>(),
                CELLS,
                value,
            );
        }
    }

    /// The correction-history table (bounds-checked `Index`).
    #[inline]
    pub fn correction_history(&self) -> &SharedCorrectionHistory {
        &self.correction_history
    }

    /// The pawn-history table (bounds-checked `Index`).
    #[inline]
    pub fn pawn_history(&self) -> &SharedPawnHistory {
        &self.pawn_history
    }

    /// Access the NUMA-local continuation-history block.
    #[inline]
    pub fn continuation_history(&self) -> &[[ContinuationHistory; 2]; 2] {
        &self.continuation_history_block.table
    }

    /// Get pawn history entry for a position
    #[inline]
    pub fn pawn_entry(&self, pos: &Position) -> &AtomicPawnHistoryEntry {
        let index = (pos.pawn_key() >> 1) as usize & self.pawn_hist_size_minus_1;
        // SAFETY: `pawn_hist_size_minus_1` is the (power-of-two) size minus one, so the masked
        // index is below the size.
        unsafe { self.pawn_history.get_unchecked(index) }
    }

    /// Get pawn correction entry for a position
    #[inline]
    pub fn pawn_correction_entry(&self, pos: &Position) -> &UnifiedCorrectionEntry {
        let index = (pos.pawn_key() as usize) & self.size_minus_1;
        // SAFETY: `size_minus_1` is the (power-of-two) size minus one, so the masked index is
        // below the size.
        unsafe { self.correction_history.get_unchecked(index) }
    }

    /// Get minor piece correction entry for a position
    #[inline]
    pub fn minor_piece_correction_entry(&self, pos: &Position) -> &UnifiedCorrectionEntry {
        let index = (pos.minor_piece_key() as usize) & self.size_minus_1;
        // SAFETY: `size_minus_1` is the (power-of-two) size minus one, so the masked index is
        // below the size.
        unsafe { self.correction_history.get_unchecked(index) }
    }

    /// Get non-pawn correction entry for white
    #[inline]
    pub fn nonpawn_correction_entry_white(&self, pos: &Position) -> &UnifiedCorrectionEntry {
        let index = (pos.non_pawn_key(Color::White) as usize) & self.size_minus_1;
        // SAFETY: `size_minus_1` is the (power-of-two) size minus one, so the masked index is
        // below the size.
        unsafe { self.correction_history.get_unchecked(index) }
    }

    /// Get non-pawn correction entry for black
    #[inline]
    pub fn nonpawn_correction_entry_black(&self, pos: &Position) -> &UnifiedCorrectionEntry {
        let index = (pos.non_pawn_key(Color::Black) as usize) & self.size_minus_1;
        // SAFETY: `size_minus_1` is the (power-of-two) size minus one, so the masked index is
        // below the size.
        unsafe { self.correction_history.get_unchecked(index) }
    }

    /// Clear a range of the shared histories (NUMA-aware)
    /// Each thread clears its portion to minimize cross-node traffic
    /// Mirrors the two C++ calls in `Search::Worker::clear()`:
    ///     sharedHistory.correctionHistory.clear_range(-5,    numaThreadIdx, numaTotal);
    ///     sharedHistory.pawnHistory.clear_range(-1338, numaThreadIdx, numaTotal);
    /// The fill values are parameters (as in C++) so upstream retunes are a call-site edit.
    ///
    /// Fills this thread's partition of the correction and pawn tables with `correction_value`
    /// and `pawn_value` respectively. Panics unless `thread_idx < numa_total`.
    pub fn clear_range(
        &self,
        correction_value: i16,
        pawn_value: i16,
        thread_idx: usize,
        numa_total: usize,
    ) {
        assert!(
            numa_total > 0 && thread_idx < numa_total,
            "invalid NUMA partition"
        );
        let partition = |size: usize| {
            let boundary =
                |index: usize| (index as u128 * size as u128 / numa_total as u128) as usize;
            boundary(thread_idx)..boundary(thread_idx + 1)
        };
        for bundle in &self.correction_history.data[partition(self.correction_history.get_size())] {
            for pawn in &bundle.pawn {
                pawn.set(correction_value);
            }
            bundle.minor.set(correction_value);
            bundle.non_pawn_white.set(correction_value);
            bundle.non_pawn_black.set(correction_value);
        }
        for entry in &self.pawn_history.data[partition(self.pawn_history.get_size())] {
            for row in entry {
                for cell in row {
                    cell.set(pawn_value);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// History table type definitions
// ---------------------------------------------------------------------------

/// ButterflyHistory records how often quiet moves have been successful or unsuccessful
/// during the current search, and is used for reduction and move ordering decisions.
/// Indexed by m.raw() (full 16-bit move value)
pub type ButterflyHistory = [[StatsEntry<i16, 7183>; UINT_16_HISTORY_SIZE]; COLOR_NB];

/// Indexed by ply and m.raw() (full 16-bit move value)
pub type LowPlyHistory = [[StatsEntry<i16, 7183>; UINT_16_HISTORY_SIZE]; LOW_PLY_HISTORY_SIZE];

/// CapturePieceToHistory is addressed by a move's \[piece\]\[to\][captured piece type]
pub type CapturePieceToHistory = [[[StatsEntry<i16, 10692>; PIECE_TYPE_NB]; SQUARE_NB]; PIECE_NB];

/// PieceToHistory is like ButterflyHistory but is addressed by a move's \[piece\]\[to\]
/// PieceToHistory is thread-shared and uses relaxed-atomic entries so that
/// ContinuationHistory can live in SharedHistories.
pub type PieceToHistory = [[AtomicStatsEntry<30000>; SQUARE_NB]; PIECE_NB];

/// ContinuationHistory is the combined history of a given pair of moves,
/// usually the current one given a previous one.
pub type ContinuationHistory = [[PieceToHistory; SQUARE_NB]; PIECE_NB];

/// The four continuation-history tables are allocated together so that the
/// large shared block can use Linux huge pages, matching Stockfish's
/// `ContinuationHistoryBlock` allocation.
#[repr(C)]
pub struct ContinuationHistoryBlock {
    pub table: [[ContinuationHistory; 2]; 2],
}

const _: () = assert!(
    core::mem::size_of::<ContinuationHistoryBlock>()
        .is_multiple_of(core::mem::size_of::<AtomicI16>())
);

/// A continuation-history block in a (large-page when available) allocation of its own.
pub struct LargePageContinuationHistory {
    // Kept alongside the typed pointer so the allocation remains owned for the
    // lifetime of the block. AtomicI16 has no destructor, so no typed drop is
    // required before LargePageAllocator releases the raw storage.
    _allocation: LargePageAllocator,
    ptr: *mut ContinuationHistoryBlock,
}

impl LargePageContinuationHistory {
    pub fn new() -> Self {
        let allocation =
            LargePageAllocator::allocate_zeroed(core::mem::size_of::<ContinuationHistoryBlock>())
                .expect("failed to allocate continuation history on large pages");
        let ptr = allocation.as_ptr() as *mut ContinuationHistoryBlock;
        Self {
            _allocation: allocation,
            ptr,
        }
    }
}

impl Deref for LargePageContinuationHistory {
    type Target = ContinuationHistoryBlock;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` points into `allocation`, which is zero-initialized and
        // remains alive for the duration of this reference.
        unsafe { &*self.ptr }
    }
}

// The block contains only atomics and is intentionally shared by workers in a
// NUMA group. The allocation is stable and the table entries are updated with
// relaxed atomic operations.
// SAFETY: the block contains only AtomicI16 entries (no interior pointers, no Drop), and the raw storage is owned by
// `_allocation` for the life of the value, so moving it to another thread is fine.
unsafe impl Send for LargePageContinuationHistory {}
// SAFETY: all access to the block's entries goes through relaxed atomic operations, so shared `&` access from several threads is race-free.
unsafe impl Sync for LargePageContinuationHistory {}

// ---------------------------------------------------------------------------
// Correction history types for static evaluation adjustments
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Convenience type aliases for correction histories using template pattern
// ---------------------------------------------------------------------------

/// Piece-to correction history - indexed by \[piece\]\[to\]
pub type PieceToCorrectionHistory = [[CorrectionStatsEntry; SQUARE_NB]; PIECE_NB];

/// Worker-local correction statistic. Cell permits updates through the shared
/// pointers cached in search stack entries without creating exclusive aliases.
#[repr(transparent)]
pub struct CorrectionStatsEntry(core::cell::Cell<i16>);

impl CorrectionStatsEntry {
    #[inline(always)]
    pub fn get(&self) -> i16 {
        self.0.get()
    }

    #[inline(always)]
    pub fn set(&self, value: i16) {
        debug_assert!(i32::from(value).abs() <= CORRECTION_HISTORY_LIMIT);
        self.0.set(value);
    }

    #[inline(always)]
    pub fn update(&self, bonus: i32) {
        let bonus = bonus.clamp(-CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
        let entry = i32::from(self.get());
        self.set((entry + bonus - entry * bonus.abs() / CORRECTION_HISTORY_LIMIT) as i16);
    }
}

/// Continuation correction history - indexed by \[prev_piece\]\[prev_to\]\[piece\]\[to\]
pub type ContinuationCorrectionHistory = [[PieceToCorrectionHistory; SQUARE_NB]; PIECE_NB];

/// Sole owner of the correction table with shared interior-mutable access.
/// The pointer field keeps recursive `&mut Worker` borrows from retagging a Box
/// pointee while stack entries retain pointers to its Cells. No exclusive table
/// borrow or cloned owner is exposed.
pub struct ContinuationCorrectionHistoryOwner {
    allocation: core::ptr::NonNull<ContinuationCorrectionHistory>,
}

impl Deref for ContinuationCorrectionHistoryOwner {
    type Target = ContinuationCorrectionHistory;

    fn deref(&self) -> &Self::Target {
        // SAFETY: this sole owner retains the allocation until Drop; all writes
        // use Cell, and it never grants an exclusive reference to the table.
        unsafe { self.allocation.as_ref() }
    }
}

impl Drop for ContinuationCorrectionHistoryOwner {
    fn drop(&mut self) {
        // SAFETY: created from a unique allocation with Box's exact layout.
        // Search stack pointers cannot outlive their worker; borrowed references
        // cannot outlive this owner. Drop is the only allocation reclamation.
        unsafe { drop(Box::from_raw(self.allocation.as_ptr())) };
    }
}

// SAFETY: ownership is unique and Cell<i16> is Send. Borrowed Cells cannot
// cross threads, and moving the owner leaves the allocation's address stable.
unsafe impl Send for ContinuationCorrectionHistoryOwner {}

/// TTMove history - tracks success rate of transposition table moves
pub type TTMoveHistory = StatsEntry<i16, 8192>;

// ---------------------------------------------------------------------------
// Helper functions for creating and initializing history tables
// ---------------------------------------------------------------------------

pub fn new_butterfly_history() -> Box<ButterflyHistory> {
    Box::new([[StatsEntry::new(); UINT_16_HISTORY_SIZE]; COLOR_NB])
}

pub fn new_low_ply_history() -> Box<LowPlyHistory> {
    Box::new([[StatsEntry::new(); UINT_16_HISTORY_SIZE]; LOW_PLY_HISTORY_SIZE])
}

pub fn new_capture_piece_to_history() -> Box<CapturePieceToHistory> {
    Box::new([[[StatsEntry::new(); PIECE_TYPE_NB]; SQUARE_NB]; PIECE_NB])
}

pub fn new_piece_to_history() -> Box<PieceToHistory> {
    // PieceToHistory holds atomic (non-Copy) entries, so it cannot be built with
    // an array-literal copy initialization. A zeroed allocation is valid because
    // AtomicI16 with all-zero bytes represents the value 0.
    // SAFETY: all-zero bytes are a valid PieceToHistory (AtomicI16 entries holding 0); a null result is diverted to
    // `handle_alloc_error`, and a non-null pointer came from the global allocator with this exact layout, as Box requires.
    unsafe {
        let layout = core::alloc::Layout::new::<PieceToHistory>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut PieceToHistory;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    }
}

pub fn new_continuation_history() -> Box<ContinuationHistory> {
    // Built on atomic PieceToHistory; zeroed allocation is valid and also avoids
    // stack overflow for this large table.
    // SAFETY: all-zero bytes are a valid ContinuationHistory (AtomicI16 entries holding 0); null is diverted to
    // `handle_alloc_error`, and the pointer came from the global allocator with this exact layout, as Box requires.
    unsafe {
        let layout = core::alloc::Layout::new::<ContinuationHistory>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut ContinuationHistory;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    }
}

// Create a single atomic pawn entry (for tests and direct use)
pub fn new_atomic_pawn_entry() -> Box<AtomicPawnHistoryEntry> {
    Box::new(core::array::from_fn(|_| {
        core::array::from_fn(|_| AtomicStatsEntry::new())
    }))
}

pub fn new_continuation_correction_history() -> ContinuationCorrectionHistoryOwner {
    // SAFETY: all-zero bytes are a valid ContinuationCorrectionHistory (Cell<i16> holding 0); null is
    // diverted to `handle_alloc_error`, and the pointer came from the global allocator with this exact layout, as Box
    // requires. Allocating zeroed memory avoids building the large array on the stack first.
    unsafe {
        let layout = core::alloc::Layout::new::<ContinuationCorrectionHistory>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut ContinuationCorrectionHistory;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        ContinuationCorrectionHistoryOwner {
            allocation: core::ptr::NonNull::new_unchecked(ptr),
        }
    }
}

pub fn new_tt_move_history() -> TTMoveHistory {
    StatsEntry::new()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_correction_statistics_preserve_the_upstream_update_formula() {
        for entry in -CORRECTION_HISTORY_LIMIT..=CORRECTION_HISTORY_LIMIT {
            for bonus in [-2048, -1024, -128, -1, 0, 1, 128, 1024, 2048] {
                let mut reference = StatsEntry::<i16, CORRECTION_HISTORY_LIMIT>::new();
                reference.set(entry as i16);
                let cell = CorrectionStatsEntry(core::cell::Cell::new(entry as i16));
                reference.update(bonus);
                cell.update(bonus);
                assert_eq!(cell.get(), reference.get());
            }
        }
    }

    #[test]
    fn cached_correction_entries_survive_recursive_owner_borrows() {
        fn reborrow(owner: &mut ContinuationCorrectionHistoryOwner) {
            owner[1][2][3][4].update(100);
        }
        let mut owner = new_continuation_correction_history();
        let entry = &owner[1][2][3][4] as *const CorrectionStatsEntry;
        reborrow(&mut owner);
        // SAFETY: the owner retains the same allocation. Reborrowing it never
        // creates an exclusive table reference; all mutation is through Cell.
        assert_eq!(unsafe { (*entry).get() }, 100);
        let mut moved = owner;
        reborrow(&mut moved);
        // SAFETY: moving the sole owner keeps its allocation and the Cell live.
        assert_eq!(unsafe { (*entry).get() }, 191);
    }

    #[test]
    fn shared_history_sizes_are_powers_of_two_and_index_is_checked() {
        for threads in [1usize, 2, 3, 6] {
            assert!(
                SharedCorrectionHistory::new(threads)
                    .get_size()
                    .is_power_of_two()
            );
            assert!(SharedPawnHistory::new(threads).get_size().is_power_of_two());
        }
        let h = SharedHistories::new(1);
        // The unchecked accessors are only reached through masked indices.
        let n = h.correction_history.get_size();
        let _ = &h.correction_history[n - 1];
        let _ = &h.pawn_history[h.pawn_history.get_size() - 1];
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn correction_history_index_out_of_range_panics() {
        let h = SharedHistories::new(1);
        let n = h.correction_history.get_size();
        let _ = &h.correction_history[n];
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn pawn_history_index_out_of_range_panics() {
        let h = SharedHistories::new(1);
        let n = h.pawn_history.get_size();
        let _ = &h.pawn_history[n];
    }

    #[test]
    fn test_stats_entry_update() {
        let mut entry: StatsEntry<i16, 100> = StatsEntry::new();

        // Test update with positive bonus
        entry.update(50);
        assert_eq!(*entry, 50);

        // Test update with negative bonus
        entry.update(-30);
        assert_eq!(*entry, 5);

        // Test clamping
        entry.update(200); // Should be clamped to 100
        assert_eq!(*entry, 100);
    }

    #[test]
    fn test_stats_entry_set_get() {
        let mut entry: StatsEntry<i16, 100> = StatsEntry::new();

        // Test set and get
        entry.set(75);
        assert_eq!(entry.get(), 75);
        assert_eq!(*entry, 75); // Test Deref

        entry.set(-50);
        assert_eq!(entry.get(), -50);
    }

    #[test]
    fn test_tt_move_history() {
        let mut tt_hist = new_tt_move_history();

        // Test initial value
        assert_eq!(tt_hist.get(), 0);

        // Test updates
        tt_hist.update(100);
        assert!(tt_hist.get() > 0);

        tt_hist.update(-50);
        let val = tt_hist.get();
        assert!(val > 0 && val < 100);
    }

    #[test]
    fn clear_range_fills_every_cell_for_any_partition() {
        for numa_total in 1..=4usize {
            let h = SharedHistories::new(1);
            for thread_idx in 0..numa_total {
                h.clear_range(-5, -1338, thread_idx, numa_total);
            }
            for bundle in h.correction_history.data.iter() {
                for cell in [
                    &bundle.pawn[0],
                    &bundle.pawn[1],
                    &bundle.minor,
                    &bundle.non_pawn_white,
                    &bundle.non_pawn_black,
                ] {
                    assert_eq!(cell.get(), -5);
                }
            }
            for entry in h.pawn_history.data.iter() {
                for row in entry {
                    for cell in row {
                        assert_eq!(cell.get(), -1338);
                    }
                }
            }
        }
    }

    #[test]
    fn clear_range_allows_overlapping_partitions_and_large_indices() {
        let h = SharedHistories::new(1);
        std::thread::scope(|scope| {
            scope.spawn(|| h.clear_range(-5, -1338, 0, 1));
            h.clear_range(-5, -1338, 0, 1);
        });
        h.clear_range(-6, -1339, usize::MAX - 2, usize::MAX);
        h.clear_range(-7, -1340, usize::MAX - 1, usize::MAX);
        for pawn in &h.correction_history.data.last().unwrap().pawn {
            assert_eq!(pawn.get(), -7);
        }
        assert_eq!(h.pawn_history.data.last().unwrap()[0][0].get(), -1340);
    }

    #[test]
    #[should_panic(expected = "invalid NUMA partition")]
    fn clear_range_rejects_out_of_range_thread_index() {
        let h = SharedHistories::new(1);
        h.clear_range(-5, -1338, 2, 2);
    }
}
