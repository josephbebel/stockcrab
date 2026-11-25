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

//! Transposition table. Port of upstream `tt.cpp`/`tt.h`.
//!
//! There is only one global hash table for the engine and all its threads.
//! For chess in particular, we even allow racy updates between threads to and
//! from the TT, as taking the time to synchronize access would cost thinking
//! time and thus Elo. In stockcrab every entry field is an atomic accessed with
//! `Ordering::Relaxed` (a plain `mov` on x86-64 and a plain `ldr`/`str` on
//! AArch64, no fences, no read-modify-write), so the races are defined
//! behaviour in the Rust memory model while the generated code and the
//! tolerated torn snapshots (fields of one entry from different writers) are
//! the same as in C++. As a hash table, collisions are possible and may cause
//! chess playing issues (bizarre blunders, faulty mate reports, etc). Fixing
//! these also loses Elo; however such risk decreases with larger TT size.
//!
//! We clearly separate TTData, a local copy of an entry, from TTWriter, which
//! writes to the global table.
//!
//! `probe(key)` is the primary method: given a board position, we lookup its
//!  entry in the table, and return a tuple of:
//!   1) whether the entry already had data on this position
//!   2) a copy of the prior data, if any (may be self-inconsistent due to races)
//!   3) a writer object to the entry

use crate::misc::LargePageAllocator;
use crate::types::{
    Bound, DEPTH_NONE, Depth, Key, Move, VALUE_INFINITE, VALUE_NONE, Value, is_decisive,
};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicI16, AtomicU8, AtomicU16, AtomicU64, Ordering};

/// A copy of the data already in an entry (possibly collided). Probes and reads
/// are racy (relaxed atomic loads of the individual fields), possibly resulting
/// in inconsistent data.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TTData {
    pub mv: Move,
    pub value: Value,
    pub eval: Value,
    pub depth: Depth,
    pub bound: Bound,
    pub is_pv: bool,
}

impl TTData {
    /// Creates a new `TTData`.
    pub fn new(
        mv: Move,
        value: Value,
        eval: Value,
        depth: Depth,
        bound: Bound,
        is_pv: bool,
    ) -> Self {
        Self {
            mv,
            value,
            eval,
            depth,
            bound,
            is_pv,
        }
    }
}

// This is used to make racy writes to the global TT (relaxed atomic stores of
// the entry fields). Writes are not "guaranteed": for chess reasons, we may
// decide the new data is less important than the old.
/// TTWriter is but a very thin wrapper around a reference to the cluster and the
/// index of the entry in it.
///
/// It borrows the table it came from (`'a`), so safe code cannot use it after the table is
/// resized, cleared or dropped (those take `&mut TranspositionTable`).
#[derive(Copy, Clone)]
pub struct TTWriter<'a> {
    cluster: &'a Cluster,
    /// Always `< CLUSTER_SIZE` (only `probe_for_search` and the tests construct writers).
    index: usize,
}

impl<'a> TTWriter<'a> {
    /// Creates a new writer for entry `index` of `cluster`.
    fn new(cluster: &'a Cluster, index: usize) -> Self {
        debug_assert!(index < CLUSTER_SIZE);
        Self { cluster, index }
    }

    #[inline(always)]
    fn entry(&self) -> &'a TTEntry {
        // SAFETY: `index < CLUSTER_SIZE` is an invariant of TTWriter (see `new`).
        unsafe { self.cluster.entry.get_unchecked(self.index) }
    }

    /// Populates the TTEntry and its key with a new node's data, possibly overwriting
    /// an old position. The update is racy (relaxed atomic accesses, no entry-wide
    /// atomicity). We convert external types to internal bitfields.
    // Inlined to match C++ (write is inlined into search)
    #[inline(always)]
    pub fn write(
        &self,
        k: Key,
        v: Value,
        pv: bool,
        b: Bound,
        d: Depth,
        m: Move,
        ev: Value,
        generation8: u8,
    ) {
        let shift = KEY_BITS * self.index;
        let keys = self.cluster.keys.load(Ordering::Relaxed);
        let key_diff = ((keys >> shift) ^ k) & KEY_MASK;
        let entry = self.entry();

        // Preserve the old ttmove if we don't have a new one
        if m.raw() != 0 || key_diff != 0 {
            entry.move16.store(m.0, Ordering::Relaxed);
        }

        // Overwrite less valuable entries (cheapest checks first). Relaxed loads are not
        // elided, so depth8/gen_bound8 are loaded only once the cheaper checks have not
        // already decided, as in the C++ short-circuit evaluation.
        // `no_overwrite` carries the loaded (depth8, gen_bound8) when the entry is kept.
        // Match C++ semantics: compare in signed space, no saturating sub
        let lhs = (d - DEPTH_NONE) + if pv { 2 } else { 0 };
        let no_overwrite = if b == Bound::EXACT || key_diff != 0 {
            None
        } else {
            let depth8 = entry.depth8.load(Ordering::Relaxed);
            if lhs > depth8 as i32 - 4 {
                None
            } else {
                let gen_bound8 = entry.gen_bound8.load(Ordering::Relaxed);
                if relative_age(gen_bound8, generation8) != 0 {
                    None
                } else {
                    Some((depth8, gen_bound8))
                }
            }
        };
        match no_overwrite {
            None => {
                debug_assert!(d > DEPTH_NONE);
                debug_assert!(d - DEPTH_NONE < 256);
                debug_assert!(generation8 <= GENERATION_MASK); // TT::new_search() plays nice

                self.cluster
                    .keys
                    .store(keys ^ (key_diff << shift), Ordering::Relaxed);
                entry
                    .depth8
                    .store((d - DEPTH_NONE) as u8, Ordering::Relaxed);
                entry.gen_bound8.store(
                    generation8 | (b.bits() << BOUND_SHIFT) | ((pv as u8) << PV_SHIFT),
                    Ordering::Relaxed,
                );
                entry.value16.store(v as i16, Ordering::Relaxed);
                entry.eval16.store(ev as i16, Ordering::Relaxed);
            }
            // Secondary aging. Important for elementary mate finding.
            // (*Scaler) Secondary aging on entries relevant to singular extensions
            // generally scales poorly and requires VVLTC verification.
            Some((depth8, gen_bound8)) => {
                if depth8 as i32 + DEPTH_NONE >= 5
                    && Bound::from_bits_truncate((gen_bound8 & BOUND_MASK) >> BOUND_SHIFT)
                        != Bound::EXACT
                {
                    let v16 = entry.value16.load(Ordering::Relaxed) as Value;
                    // Guard against racy underflows, default to "unoccupied"
                    if v16.abs() < VALUE_INFINITE && is_decisive(v16) {
                        entry
                            .depth8
                            .store((depth8 as i32 - 1).max(0) as u8, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Flag a TTEntry as useless (decrementing the stored depth by the given
    /// penalty). Guards against racy underflows, defaulting to "unoccupied".
    pub fn penalize(&self, penalty: i32) {
        let entry = self.entry();
        let d = entry.depth8.load(Ordering::Relaxed) as i32;
        entry
            .depth8
            .store((d - penalty).max(0) as u8, Ordering::Relaxed);
    }
}

// The transposition table entry: packed bitfields matching the C++ layout.
/// TTEntry struct is the 8 bytes transposition table entry, defined as:
///
/// depth       8 bit
/// pv node     1 bit
/// bound type  2 bit
/// generation  5 bit
/// move       16 bit
/// value      16 bit
/// evaluation 16 bit
///
/// Fields are laid out in the same order as the C++ TTEntry. The entry's key, its
/// low 21 bits, lives in the `keys` word of its cluster.
///
/// We use `bool(depth8)` as the cheap internal occupancy check, corresponding
/// to `depth == DEPTH_NONE` externally, so we offset the internal depth by DEPTH_NONE.
///
/// Pv, bound and generation are packed in a single byte.
///
/// Every field is an atomic accessed only with `Ordering::Relaxed`: the table is shared by all
/// search threads without synchronisation (tolerated races, as in C++), and relaxed atomics make
/// those races defined behaviour at the cost of nothing (plain `mov`s). An entry as a whole is
/// not atomic: a reader can see fields written by different threads.
#[repr(C)]
#[derive(Debug, Default)]
pub struct TTEntry {
    depth8: AtomicU8,
    gen_bound8: AtomicU8,
    move16: AtomicU16, // `Move.0`
    value16: AtomicI16,
    eval16: AtomicI16,
}

impl TTEntry {
    /// Extract data from the TT entry. We have to convert TT internal bitfields
    /// to external types.
    pub fn read(&self) -> TTData {
        self.read_occupied().0
    }

    /// `read` plus the occupancy check, from a single load of each field.
    #[inline(always)]
    fn read_occupied(&self) -> (TTData, bool) {
        let depth8 = self.depth8.load(Ordering::Relaxed);
        let gen_bound8 = self.gen_bound8.load(Ordering::Relaxed);
        (
            TTData::new(
                Move(self.move16.load(Ordering::Relaxed)),
                self.value16.load(Ordering::Relaxed) as Value,
                self.eval16.load(Ordering::Relaxed) as Value,
                DEPTH_NONE + depth8 as Depth,
                Bound::from_bits_truncate((gen_bound8 & BOUND_MASK) >> BOUND_SHIFT),
                (gen_bound8 & PV_MASK) != 0,
            ),
            depth8 != 0,
        )
    }

    /// Check if the TT entry is occupied.
    pub fn is_occupied(&self) -> bool {
        self.depth8.load(Ordering::Relaxed) != 0
    }

    /// Returns this entry's age. We count generations like clocks count hours,
    /// i.e. we require 0 - 1 == 31. Unsigned subtraction guarantees the required
    /// borrowing regardless of the upper pv/bound bits.
    pub fn relative_age(&self, generation8: u8) -> u8 {
        relative_age(self.gen_bound8.load(Ordering::Relaxed), generation8)
    }
}

#[inline(always)]
fn relative_age(gen_bound8: u8, generation8: u8) -> u8 {
    generation8.wrapping_sub(gen_bound8) & GENERATION_MASK
}

// Pv, bound and generation are packed in a single byte (`gen_bound8`):
// generation uses the 5 low bits, bound the next 2 bits, and pv the top bit.
const GENERATION_BITS: u8 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1;
const BOUND_SHIFT: u8 = GENERATION_BITS;
const BOUND_MASK: u8 = 0b11 << BOUND_SHIFT;
const PV_SHIFT: u8 = BOUND_SHIFT + 2;
const PV_MASK: u8 = 1 << PV_SHIFT;

// Number of entries per cluster (fits cache line)
pub const CLUSTER_SIZE: usize = 3;
const KEY_BITS: usize = 21;

/// A 32-byte transposition-table cluster: three entries whose 21-bit key fragments are packed
/// into one word, matched together with a SWAR test.
#[repr(C, align(32))]
#[derive(Debug, Default)]
pub struct Cluster {
    // keys contains the low 21 bits of the Zobrist keys of the 3 TT entries in this cluster,
    // packed together into 63 bits of an u64
    keys: AtomicU64,
    pub entry: [TTEntry; CLUSTER_SIZE],
}

// Ensure Cluster is 32 bytes
const _: [(); 32] = [(); core::mem::size_of::<Cluster>()];
const _: () = assert!(mem::align_of::<Cluster>() == 32);
// Ensure TTEntry is 8 bytes and has 2-byte alignment (matches C++)
const _: () = assert!(mem::size_of::<TTEntry>() == 8);
const _: () = assert!(mem::align_of::<TTEntry>() == 2);

const KEY_MASK: u64 = (1 << KEY_BITS) - 1;

/// Scheduling contract for clearing the transposition table.
///
/// # Safety
/// `run_on_thread` must execute each callback at most once, and `wait_on_thread`
/// must not return until every previously scheduled callback for that thread has
/// finished, including when it unwinds. Submission may unwind before accepting its
/// callback, or after accepting it if a subsequent wait still drains it. The table
/// catches submission/wait failures and drains every worker before returning or
/// propagating a panic. It lends its allocation only until that draining completes;
/// violating these requirements can cause data races or use-after-free.
///
/// Safe clients cannot implement this lifetime contract without acknowledging it:
/// ```compile_fail
/// struct DeferredPool;
/// impl stockcrab::tt::ThreadPool for DeferredPool {
///     fn num_threads(&self) -> usize { 1 }
///     fn run_on_thread<F: FnOnce() + Send + 'static>(&self, _: usize, _: F) {}
///     fn wait_on_thread(&self, _: usize) {}
/// }
/// ```
pub unsafe trait ThreadPool {
    fn num_threads(&self) -> usize;
    fn run_on_thread<F: FnOnce() + Send + 'static>(&self, i: usize, f: F);
    fn wait_on_thread(&self, i: usize);
    /// Number of NUMA nodes across which threads are bound. Defaults to 1.
    fn numa_nodes(&self) -> usize {
        1
    }
    /// For each thread index, the NUMA node it is bound to. Empty when threads
    /// are not bound to specific NUMA nodes.
    fn get_bound_thread_to_numa_node(&self) -> Vec<usize> {
        Vec::new()
    }
}

/// The transposition table itself.
/// Uses large page memory allocation for better performance when available.
///
/// A TranspositionTable is an array of Cluster, of size clusterCount. Each
/// cluster consists of ClusterSize number of TTEntry. Each non-empty TTEntry
/// contains information on exactly one position. The size of a Cluster should
/// divide the size of a cache line for best performance, as the cacheline is
/// prefetched when possible.
pub struct TranspositionTable {
    cluster_count: usize,
    table: Option<LargePageAllocator>, // Custom allocator for large pages
    table_ptr: *mut Cluster,           // Direct pointer to the table
    generation8: AtomicU8,
    /// Set once by `new_search` for internal search probes, and by public probes.
    /// Search uses a private probe path after establishing its session.
    /// While it is clear, the whole table is known to be zero, so `clear` has nothing to
    /// wipe and a `resize` to an identical layout has nothing to reallocate.
    dirty: AtomicBool,
    /// Allocation parameters of the live table (see `resize`).
    layout: TTLayout,
}

/// Everything that decides where the pages of the table live and who touched them first;
/// a `resize` that would produce the same layout over a still-zero table is a no-op.
#[derive(PartialEq, Eq, Default)]
struct TTLayout {
    size_bytes: usize,
    huge_page_hint: bool,
    thread_count: usize,
    thread_to_numa: Vec<usize>,
}

impl TranspositionTable {
    /// Creates an empty TT, memory will be allocated on `resize`.
    ///
    /// Until `resize` runs the table holds a single zeroed cluster, so `probe`/`get_cluster`
    /// are always in bounds (`cluster_count >= 1` and `table_ptr` non-null is an invariant of
    /// the struct) without any check on the hot path.
    pub fn new() -> Self {
        let allocator = LargePageAllocator::allocate_zeroed(mem::size_of::<Cluster>())
            .expect("Failed to allocate the placeholder transposition table cluster");
        Self {
            cluster_count: 1,
            table_ptr: allocator.as_ptr() as *mut Cluster,
            table: Some(allocator),
            generation8: AtomicU8::new(0),
            dirty: AtomicBool::new(false),
            layout: TTLayout::default(),
        }
    }

    /// Sets the size of the transposition table, measured in megabytes.
    /// Uses large page allocation for better TLB performance when available.
    ///
    /// Transposition table consists of clusters and each cluster consists
    /// of ClusterSize number of TTEntry.
    pub fn resize(&mut self, mb_size: usize, threads: &impl ThreadPool) {
        // At least one cluster, so the table is never empty (see `new`).
        let cluster_count = mb_size.saturating_mul(1024 * 1024) / core::mem::size_of::<Cluster>();
        let cluster_count = cluster_count.max(1);
        let size_bytes = cluster_count.saturating_mul(core::mem::size_of::<Cluster>());

        // Request 1 GiB huge pages only when each NUMA node would receive at
        // least eight. This is Stockfish's guard against oversubscribing the
        // reserved huge-page pool; the allocator falls back transparently when
        // the mapping is unavailable.
        const HUGE_PAGE_SIZE: usize = 1 << 30;
        let huge_page_hint = size_bytes
            >= threads
                .numa_nodes()
                .saturating_mul(HUGE_PAGE_SIZE)
                .saturating_mul(8);

        // Same layout over a table nobody has written since it was last wiped (the usual
        // `setoption Threads` + `setoption Hash` + `ucinewgame` start-up burst): freeing,
        // re-mapping and re-faulting the pages would reproduce exactly what is already there.
        let layout = TTLayout {
            size_bytes,
            huge_page_hint,
            thread_count: threads.num_threads(),
            thread_to_numa: threads.get_bound_thread_to_numa_node(),
        };
        if self.table.is_some() && !self.dirty.load(Ordering::Relaxed) && self.layout == layout {
            self.generation8.store(0, Ordering::Relaxed);
            return;
        }

        // Free the existing table first (as C++ does) so peak memory is not doubled. The struct
        // is only observable again after this function returns (it exits the process on failure).
        self.table = None;
        match LargePageAllocator::allocate_with_hint(size_bytes, huge_page_hint) {
            Some(allocator) => {
                self.cluster_count = cluster_count;
                self.table_ptr = allocator.as_ptr() as *mut Cluster;
                self.table = Some(allocator);
                self.layout = layout;
            }
            None => {
                // Match C++ error handling: print error and exit
                eprintln!("Failed to allocate {}MB for transposition table.", mb_size);
                std::process::exit(1);
            }
        }

        // The pages are fresh (zero) but still untouched: wipe them in `clear` so that the
        // pool threads fault them in, as C++ does.
        *self.dirty.get_mut() = true;
        self.clear(threads);
    }

    /// Initializes the entire transposition table to zero, in a multi-threaded way.
    pub fn clear(&mut self, threads: &impl ThreadPool) {
        self.generation8.store(0, Ordering::Relaxed);
        // Never written since the last wipe: already all zero (see `dirty`).
        if !*self.dirty.get_mut() && self.table.is_some() && self.layout.size_bytes != 0 {
            return;
        }
        let mut thread_count = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            thread_count = threads.num_threads();
            if thread_count == 0 {
                self.zero_all();
                return;
            }
            // Capture the raw pointer as usize so it can be sent safely
            let ptr_addr = self.table_ptr as usize;

            // To promote good NUMA distribution (esp. with huge pages), we permute
            // threads so that all threads in a NUMA node clear a contiguous region
            // of the TT.
            let thread_to_numa = threads.get_bound_thread_to_numa_node();
            let mut order: Vec<usize> = (0..thread_count).collect();
            if thread_to_numa.len() == thread_count {
                // Rust's sort_by is stable, matching C++ std::stable_sort.
                order.sort_by(|&t1, &t2| thread_to_numa[t1].cmp(&thread_to_numa[t2]));
            }

            for i in 0..thread_count {
                let start = (self.cluster_count / thread_count) * i;
                let len = if i + 1 != thread_count {
                    self.cluster_count / thread_count
                } else {
                    self.cluster_count - start
                };
                threads.run_on_thread(order[i], move || {
                    // Reconstruct the pointer inside the closure
                    let ptr = ptr_addr as *mut Cluster;
                    // SAFETY: `start + len <= cluster_count` (chunks partition `0..cluster_count`), so
                    // the written range lies inside the `cluster_count * 32`-byte allocation; chunks are
                    // disjoint across threads, and `clear` waits for all threads before returning while it
                    // holds `&mut self`, so no other access to the table (atomic or not) can overlap
                    // these plain writes and the table is not freed or resized while this runs.
                    unsafe {
                        // Zero the entire chunk in one go (memset), like C++
                        ptr::write_bytes(
                            ptr.add(start) as *mut u8,
                            0u8,
                            len * mem::size_of::<Cluster>(),
                        );
                    }
                });
            }
            for i in 0..thread_count {
                threads.wait_on_thread(i);
            }
        }));
        if let Err(payload) = result {
            // A worker can report an old panic during submission, or a current
            // panic during waiting. Keep the allocation live until ALL accepted
            // callbacks have finished, even if another wait also reports a panic.
            let mut secondary_panics = Vec::new();
            for i in 0..thread_count {
                if let Err(secondary) =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        threads.wait_on_thread(i);
                    }))
                {
                    // Even dropping a user-supplied panic payload can panic.
                    // Retain it until every callback has finished.
                    secondary_panics.push(secondary);
                }
            }
            // Submission may have skipped a region. Finish initialization even
            // on non-Linux allocations, whose initial bytes need not be zero.
            self.zero_all();
            *self.dirty.get_mut() = false;
            drop(secondary_panics);
            std::panic::resume_unwind(payload);
        }
        *self.dirty.get_mut() = false;
    }

    fn zero_all(&mut self) {
        // SAFETY: exclusive table access; callers have drained every callback.
        // The allocator owns cluster_count contiguous clusters until this returns.
        unsafe { ptr::write_bytes(self.table_ptr, 0, self.cluster_count) };
    }

    /// Returns an approximation of the hashtable occupation during a search.
    /// The hash is x permill full, as per the UCI protocol. Only counts entries
    /// whose relative age is at most `max_age`.
    pub fn hashfull(&self, max_age: usize) -> usize {
        let mut cnt = 0;
        let generation8 = self.generation8.load(Ordering::Relaxed);

        // Match C++ loop exactly: for (int i = 0; i < 1000; ++i)
        // The C++ code doesn't check clusterCount in the loop condition,
        // it relies on clusterCount being > 0 from resize()
        for i in 0..1000 {
            // Safety: Only access if within bounds
            if i >= self.cluster_count {
                break;
            }

            // SAFETY: `i < cluster_count` was checked above, so `table_ptr.add(i)` is an in-bounds,
            // 32-byte-aligned, initialized (zeroed in `resize`) cluster of the live allocation.
            // All its fields are atomics, so shared access from any thread is fine.
            let cluster = unsafe { &*self.table_ptr.add(i) };
            for entry in &cluster.entry {
                if entry.is_occupied() && (entry.relative_age(generation8) as usize) <= max_age {
                    cnt += 1;
                }
            }
        }
        cnt / CLUSTER_SIZE
    }

    /// Overload with default max_age = 0, matching C++ default parameter
    pub fn hashfull_default(&self) -> usize {
        self.hashfull(0)
    }

    /// Must be called at the beginning of each root search to track entry aging.
    pub fn new_search(&self) {
        self.mark_search_active();
        // Don't overflow into the other bits of TTEntry::gen_bound8
        let g = self.generation8.load(Ordering::Relaxed).wrapping_add(1) & GENERATION_MASK;
        self.generation8.store(g, Ordering::Relaxed);
    }

    /// Establish the write session also for direct public Worker entry points.
    /// Called once while taking the search's TT read lock, without advancing age.
    pub(crate) fn mark_search_active(&self) {
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Number of allocated clusters (`mbSize * 1024 * 1024 / sizeof(Cluster)`).
    /// Introspection accessor: used by `tests/regression_parity.rs` to verify
    /// that `setoption name Hash value N` actually resizes the table.
    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }

    /// The current age, used when writing new data to the TT.
    pub fn generation(&self) -> u8 {
        self.generation8.load(Ordering::Relaxed)
    }

    /// Looks up the current position in the transposition table. Calling probe(key)
    /// returns true if the key is found (which may be a collision) and has non-null
    /// data. Otherwise, it returns false and a pointer to an empty or least valuable
    /// TTEntry to be replaced later. The value of an entry is its depth minus 8 times
    /// its relative age.
    pub fn probe(&self, key: Key) -> (bool, TTData, TTWriter<'_>) {
        // Public writers do not have to start a search session first. Mark the
        // table conservatively at this boundary, never on the internal node path.
        self.dirty.store(true, Ordering::Relaxed);
        self.probe_for_search(key)
    }

    /// Internal callers establish the write session once with new_search().
    pub(crate) fn probe_for_search(&self, key: Key) -> (bool, TTData, TTWriter<'_>) {
        // SAFETY: `get_cluster` points inside the allocation made by `new`/`resize` (invariant:
        // `table_ptr` points at `cluster_count >= 1` clusters owned by `table`, which only
        // `&mut self` methods replace, so the borrow of `self` keeps it alive); clusters are
        // aligned to 32 bytes and zero-initialized. All their fields are atomics, so a shared
        // reference to them is valid while other threads access them concurrently.
        let cluster = unsafe { &*self.get_cluster(key) };
        let entries = &cluster.entry;
        let key21 = key & KEY_MASK; // Use the low 21 bits as key inside the cluster

        const KEY_LSB: u64 = 1 | (1 << KEY_BITS) | (1 << (2 * KEY_BITS));
        const KEY_MSB: u64 = KEY_LSB << (KEY_BITS - 1);
        let keys = cluster.keys.load(Ordering::Relaxed);
        let key_xors = keys ^ (KEY_LSB * key21);
        // This bit twiddling trick will produce a one in the last (=most significant) bit of
        // the first 21bit block matching key21. The bits before that first matching block will
        // be all zeros, but after the first matching block there could be garbage.
        let matches = key_xors.wrapping_sub(KEY_LSB) & !key_xors & KEY_MSB;

        if matches != 0 {
            const _: () = assert!(
                KEY_BITS == 21,
                "Index computation below assumes KeyBits == 21"
            );
            let i = (matches.trailing_zeros() as usize >> 4) - 1;
            // SAFETY: `matches` is a nonzero subset of KEY_MSB (bits 20, 41 and 62), so
            // `i` is (20 >> 4) - 1 = 0, (41 >> 4) - 1 = 1 or (62 >> 4) - 1 = 2.
            let entry = unsafe { entries.get_unchecked(i) };
            // This gap is the main place for read races.
            // After `read()` completes that copy is final, but may be self-inconsistent.
            let (data, occupied) = entry.read_occupied();
            return (occupied, data, TTWriter::new(cluster, i));
        }

        // Find an entry to replace (least valuable)
        let generation8 = self.generation8.load(Ordering::Relaxed);
        let score = |e: &TTEntry| {
            (e.depth8.load(Ordering::Relaxed) as i32) - 8 * (e.relative_age(generation8) as i32)
        };
        let mut replace_idx = 0;
        let mut best_score = score(&entries[0]);
        for i in 1..CLUSTER_SIZE {
            let s = score(&entries[i]);
            if best_score > s {
                replace_idx = i;
                best_score = s;
            }
        }

        let writer = TTWriter::new(cluster, replace_idx);
        let default_data = TTData::new(
            Move::none(),
            VALUE_NONE,
            VALUE_NONE,
            DEPTH_NONE,
            Bound::NONE,
            false,
        );
        (false, default_data, writer)
    }

    /// The hash function; its only external use is memory prefetching
    pub fn get_cluster(&self, key: Key) -> *const Cluster {
        let idx = mul_hi64(key, self.cluster_count as u64) as usize;
        // SAFETY: `idx = floor(key * cluster_count / 2^64) < cluster_count`, so the address is
        // inside the allocation; only an address is formed here, nothing is read.
        unsafe { self.table_ptr.add(idx) }
    }
}

// SAFETY: `table_ptr` points into the allocation owned by `table`, which moves with the struct,
// and is only replaced through `&mut self` (`resize`). Cross-thread access to the entries goes
// through relaxed atomics only (`clear` writes zeros through `&mut self`, with exclusive access).
unsafe impl Send for TranspositionTable {}
// SAFETY: see the `Send` impl: shared access goes through `probe`/`get_cluster`/`hashfull`, all
// of which stay in bounds and touch entries only via atomics, so concurrent `&self` use is free
// of data races.
unsafe impl Sync for TranspositionTable {}

/// Multiply using high 64 bits of the product.
/// Matches upstream `mul_hi64` in misc.h.
#[inline]
fn mul_hi64(a: u64, b: u64) -> u64 {
    // On x86-64 and AArch64 use the native 128-bit multiply.
    #[cfg(all(
        target_pointer_width = "64",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        ((a as u128).wrapping_mul(b as u128) >> 64) as u64
    }

    // Portable fallback for all other targets.
    #[cfg(not(all(
        target_pointer_width = "64",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    {
        let a_lo = a & 0xFFFF_FFFF;
        let a_hi = a >> 32;
        let b_lo = b & 0xFFFF_FFFF;
        let b_hi = b >> 32;

        let c1 = (a_lo.wrapping_mul(b_lo)) >> 32;
        let c2 = a_hi.wrapping_mul(b_lo).wrapping_add(c1);
        let c3 = a_lo.wrapping_mul(b_hi).wrapping_add(c2 & 0xFFFFFFFF);

        a_hi.wrapping_mul(b_hi)
            .wrapping_add(c2 >> 32)
            .wrapping_add(c3 >> 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::prefetch;
    use crate::types::{Bound, DEPTH_NONE, Depth, Key, Move, Value};

    /// Dummy single-threaded pool for testing
    struct DummyPool;
    // SAFETY: DummyPool executes each job inline; its wait has nothing pending.
    unsafe impl ThreadPool for DummyPool {
        fn num_threads(&self) -> usize {
            1
        }
        fn run_on_thread<F: FnOnce() + Send + 'static>(&self, _i: usize, f: F) {
            f();
        }
        fn wait_on_thread(&self, _i: usize) {}
    }

    /// Several threads probe, write, age and count on one small table at once. This is a defined
    /// (atomic) race now; the test checks that nothing panics and that every snapshot stays inside
    /// the ranges the writers produce per field.
    #[test]
    fn test_concurrent_probe_write_hashfull() {
        let mut tt = TranspositionTable::new();
        tt.resize(1, &DummyPool);
        let tt = &tt;
        std::thread::scope(|s| {
            for t in 0..4u64 {
                s.spawn(move || {
                    let mut x = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t + 1);
                    for i in 0..200_000u32 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let key = x % 4096; // few clusters, many collisions
                        let (hit, data, writer) = tt.probe(key);
                        assert!(!hit || (data.value.abs() <= 100 && data.eval.abs() <= 50));
                        assert!(data.depth >= DEPTH_NONE && data.depth <= DEPTH_NONE + 100);
                        writer.write(
                            key,
                            (x % 100) as Value,
                            i % 3 == 0,
                            Bound::LOWER,
                            DEPTH_NONE + 1 + (i % 100) as Depth,
                            Move::new((x >> 20) as u16),
                            (x % 50) as Value,
                            tt.generation(),
                        );
                        if i % 1000 == 0 {
                            writer.penalize(1);
                            assert!(tt.hashfull(0) <= 1000);
                            tt.new_search();
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn test_write_and_read() {
        let cluster = Cluster::default();
        let entry = &cluster.entry[1];
        assert!(!entry.is_occupied());
        let k: Key = 0xabcd_0012_3456;
        let v: Value = 42;
        let ev: Value = 7;
        let d: Depth = DEPTH_NONE + 5;
        TTWriter::new(&cluster, 1).write(k, v, true, Bound::EXACT, d, Move::new(0x55), ev, 8);
        assert!(entry.is_occupied());
        let data = entry.read();
        assert_eq!(data.mv, Move::new(0x55));
        assert_eq!(data.value, 42);
        assert_eq!(data.eval, 7);
        assert_eq!(data.depth, d);
        assert_eq!(data.bound, Bound::EXACT);
        assert!(data.is_pv);
        assert_eq!(
            cluster.keys.load(Ordering::Relaxed),
            (k & KEY_MASK) << KEY_BITS
        );
    }

    #[test]
    fn test_write_no_overwrite_preserves_depth() {
        let cluster = Cluster::default();
        let entry = &cluster.entry[2];
        // Seed with some data: non-EXACT bound, depth8 high enough
        let k: Key = 0x1f_1234;
        cluster.keys.store(k << (2 * KEY_BITS), Ordering::Relaxed);
        entry.depth8.store(10, Ordering::Relaxed); // depth = 10 + offset
        entry.gen_bound8.store(0, Ordering::Relaxed); // not pv, bound NONE
        // New save with same key, not meeting overwrite conditions
        let d: Depth = DEPTH_NONE + 1; // lhs = 1 (no pv)
        // rhs = depth8 - 4 = 6, lhs !> rhs; rel_age == 0; b != EXACT
        TTWriter::new(&cluster, 2).write(k, 0, false, Bound::NONE, d, Move::none(), 0, 0);
        // Secondary aging only triggers for decisive values; value 0 is not
        // decisive, so depth should NOT change.
        assert_eq!(entry.depth8.load(Ordering::Relaxed), 10);
    }

    /// A write replaces its own 21-bit key slot and leaves the other two, and the unused
    /// top bit, as they were.
    #[test]
    fn test_write_updates_only_its_key_slot() {
        let slots = [0x0a_bcdeu64, 0x1f_ffff, 0x00_0001];
        let packed = |s: [u64; 3]| s[0] | (s[1] << KEY_BITS) | (s[2] << (2 * KEY_BITS));
        for i in 0..CLUSTER_SIZE {
            let cluster = Cluster::default();
            cluster.keys.store(packed(slots), Ordering::Relaxed);
            let k: Key = 0xffff_ffff_ff15_5555;
            TTWriter::new(&cluster, i).write(
                k,
                1,
                false,
                Bound::EXACT,
                DEPTH_NONE + 1,
                Move::none(),
                0,
                0,
            );
            let mut expected = slots;
            expected[i] = k & KEY_MASK;
            assert_eq!(cluster.keys.load(Ordering::Relaxed), packed(expected));
        }
    }

    /// The SWAR key match finds the first slot whose 21-bit key equals the probe's, as a
    /// scan of the three slots would, whatever the other slots hold (including keys that
    /// differ from it by one bit, or by a borrow).
    #[test]
    fn test_probe_matches_first_equal_key_slot() {
        // The placeholder table has a single cluster: every key maps to it.
        let tt = TranspositionTable::new();
        // SAFETY: cluster 0 of the placeholder table, alive for the borrow of `tt`.
        let cluster = unsafe { &*tt.get_cluster(0) };
        for e in &cluster.entry {
            e.depth8.store(1, Ordering::Relaxed);
        }
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut hits = [0; CLUSTER_SIZE + 1];
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let key21 = x & KEY_MASK;
            let near = [
                key21,
                key21 ^ (1 << ((x >> 59) % KEY_BITS as u64)),
                key21.wrapping_sub(1) & KEY_MASK,
                (key21 + 1) & KEY_MASK,
                key21 ^ (1 << 20),
                0,
                KEY_MASK,
                (x >> 21) & KEY_MASK,
            ];
            let slots: [u64; CLUSTER_SIZE] =
                core::array::from_fn(|j| near[((x >> (3 * j + 42)) & 7) as usize]);
            let keys = slots[0] | (slots[1] << KEY_BITS) | (slots[2] << (2 * KEY_BITS));
            cluster.keys.store(keys, Ordering::Relaxed);

            let expected = slots.iter().position(|&s| s == key21);
            let (found, _, writer) = tt.probe(key21 | (x.rotate_left(17) & !KEY_MASK));
            assert_eq!(found, expected.is_some(), "keys {keys:#x} key21 {key21:#x}");
            if let Some(i) = expected {
                assert_eq!(writer.index, i, "keys {keys:#x} key21 {key21:#x}");
            }
            hits[expected.unwrap_or(CLUSTER_SIZE)] += 1;
        }
        // Every outcome is exercised.
        assert!(hits.iter().all(|&n| n > 1000), "{hits:?}");
    }

    /// Keys that agree in their low 16 bits but not in the low 21 no longer collide.
    #[test]
    fn test_keys_equal_in_low_16_bits_do_not_collide() {
        let tt = TranspositionTable::new();
        let a: Key = 0x0001_2345;
        let (_, _, writer) = tt.probe(a);
        writer.write(
            a,
            5,
            false,
            Bound::EXACT,
            DEPTH_NONE + 4,
            Move::none(),
            0,
            0,
        );
        assert!(tt.probe(a).0);
        assert!(!tt.probe(0x0010_2345).0);
        assert!(tt.probe(0xffff_ffff_ffe1_2345).0);
    }

    #[test]
    fn test_relative_age() {
        let entry = TTEntry::default();
        entry.gen_bound8.store(20, Ordering::Relaxed);
        let age = entry.relative_age(30);
        assert_eq!(age, 30u8.wrapping_sub(20) & GENERATION_MASK);
    }

    #[test]
    fn test_transposition_table_probe_and_hashfull() {
        let mut tt = TranspositionTable::new();
        let pool = DummyPool;
        tt.resize(1, &pool);
        let k: Key = 0xdead_beef;
        let (found, _, writer) = tt.probe(k);
        assert!(!found);
        writer.write(
            k,
            100,
            false,
            Bound::LOWER,
            DEPTH_NONE + 3,
            Move::new(0xAB),
            50,
            tt.generation(),
        );
        let (found2, data2, _) = tt.probe(k);
        assert!(found2);
        assert_eq!(data2.value, 100);
        assert_eq!(data2.eval, 50);
        assert_eq!(tt.hashfull(0), 0); // 1 entry / 3 => 0
        assert_eq!(tt.hashfull_default(), 0); // Test default parameter version
    }

    #[test]
    fn test_clear_and_resize_skip_only_pristine_tables() {
        let mut tt = TranspositionTable::new();
        let pool = DummyPool;
        tt.resize(1, &pool);
        let ptr = tt.table_ptr;
        // Pristine table, same layout: nothing to do, same allocation.
        tt.resize(1, &pool);
        assert_eq!(tt.table_ptr, ptr);
        tt.clear(&pool);
        assert_eq!(tt.table_ptr, ptr);

        // A search dirties the table: both `clear` and an identical `resize` must wipe it.
        for wipe in [false, true] {
            tt.new_search();
            let k = 0x1234_5678_9abc_def0u64;
            let (_, _, writer) = tt.probe(k);
            writer.write(
                k,
                100,
                false,
                Bound::LOWER,
                DEPTH_NONE + 3,
                Move::new(0xAB),
                50,
                tt.generation(),
            );
            assert!(tt.probe(k).0);
            if wipe {
                tt.resize(1, &pool);
            } else {
                tt.clear(&pool);
            }
            assert!(!tt.probe(k).0);
            assert_eq!(tt.generation(), 0);
            assert_eq!(tt.hashfull(0), 0);
        }
        // A different size always reallocates.
        tt.resize(2, &pool);
        assert_eq!(
            tt.cluster_count(),
            2 * 1024 * 1024 / core::mem::size_of::<Cluster>()
        );
    }

    #[test]
    fn test_fresh_table_is_probeable() {
        let tt = TranspositionTable::new();
        let (found, _, _) = tt.probe(0x1234_5678_9abc_def0);
        assert!(!found);
        assert!(!tt.get_cluster(u64::MAX).is_null());
        assert_eq!(tt.cluster_count(), 1);
    }

    #[test]
    fn test_new_search_generation() {
        let tt = TranspositionTable::new();
        assert_eq!(tt.generation(), 0);
        tt.new_search();
        assert_eq!(tt.generation(), 1);
        tt.new_search();
        assert_eq!(tt.generation(), 2);
    }

    #[test]
    fn test_replacement_policy_matches_cpp_logic() {
        let mut tt = TranspositionTable::new();
        let pool = DummyPool;
        tt.resize(1, &pool);
        tt.new_search();
        let generation = tt.generation();

        // Craft a specific cluster
        let idx = 0u64; // We'll probe with a nonzero key21 to avoid early match
        // SAFETY: cluster 0 of a 1 MiB table is in bounds and valid for the borrow of `tt`.
        let cluster = unsafe { &*tt.get_cluster(idx) };
        let entries = &cluster.entry;
        // entry 0: depth8=5, age=0 => score = 5 - 8*0 = 5
        // entry 1: depth8=7, age=3 => score = 7 - 8*3 = -17
        // entry 2: depth8=9, age=6 => score = 9 - 8*6 = -39 (smallest)
        for (e, (depth8, age)) in entries.iter().zip([(5, 0u8), (7, 3), (9, 6)]) {
            e.depth8.store(depth8, Ordering::Relaxed);
            e.gen_bound8.store(
                generation.wrapping_sub(age) & GENERATION_MASK,
                Ordering::Relaxed,
            );
        }

        let probe_key: Key = 1; // low 21-bits != 0 so no match in empty entries
        let (_found, _data, writer) = tt.probe(probe_key);
        // We expect replacement to pick entry 2 (lowest score)
        assert!(core::ptr::eq(writer.cluster, cluster));
        assert_eq!(writer.index, 2);
    }

    #[test]
    fn test_prefetch() {
        // Test that prefetch doesn't crash (it's mostly a no-op in tests)
        let data = vec![1u8; 64];
        prefetch(data.as_ptr());
    }

    #[test]
    fn test_mul_hi64() {
        // Test cases matching C++ implementation
        assert_eq!(mul_hi64(0, 0), 0);
        assert_eq!(mul_hi64(1, 1), 0);
        assert_eq!(mul_hi64(0xFFFFFFFF, 0xFFFFFFFF), 0);
        assert_eq!(mul_hi64(0xFFFFFFFFFFFFFFFF, 2), 1);
        assert_eq!(mul_hi64(0x8000000000000000, 2), 1);

        // Test with large numbers
        let a = 0x123456789ABCDEF0u64;
        let b = 0xFEDCBA9876543210u64;
        let expected = ((a as u128 * b as u128) >> 64) as u64;
        assert_eq!(mul_hi64(a, b), expected);
    }

    #[test]
    fn test_large_page_allocator() {
        // Test allocation and deallocation
        let size = 1024 * 1024; // 1MB
        if let Some(allocator) = LargePageAllocator::allocate_with_hint(size, true) {
            assert!(!allocator.as_ptr().is_null());
            assert!(allocator.size() >= size);
            // Allocator will be deallocated on drop
        }
        // If allocation fails (e.g., on systems without large pages), that's OK for tests
    }

    #[test]
    fn test_transposition_table_prefetch() {
        let mut tt = TranspositionTable::new();
        let pool = DummyPool;
        tt.resize(1, &pool);

        let key: Key = 0xDEADBEEF;
        // Test that prefetching the cluster, as search does, doesn't crash
        prefetch(tt.get_cluster(key));

        // Verify get_cluster returns valid pointer
        let cluster_ptr = tt.get_cluster(key);
        assert!(!cluster_ptr.is_null());
    }

    #[test]
    fn test_hashfull_matches_cpp() {
        let mut tt = TranspositionTable::new();
        let pool = DummyPool;
        tt.resize(1, &pool); // Small table for testing

        // 1 MiB is 32768 clusters, so `i << 49` maps to cluster `i` (within the 1000
        // sampled by hashfull). The low bit keeps key21 non-zero so the empty slot is
        // replaced.
        for i in 0u64..10 {
            let key = (i << 49) | 1;
            let (_, _, writer) = tt.probe(key);
            writer.write(
                key,
                100,
                false,
                Bound::LOWER,
                DEPTH_NONE + 3,
                Move::new(0xAB),
                50,
                tt.generation(),
            );
        }

        // 10 occupied entries in 10 distinct clusters: 10 / CLUSTER_SIZE (3) permille.
        assert_eq!(tt.hashfull(0), 3);
        assert_eq!(tt.hashfull(1), 3);

        // After a new search the entries are one generation old: excluded by
        // max_age 0, still counted by max_age 1.
        tt.new_search();
        assert_eq!(tt.hashfull(0), 0);
        assert_eq!(tt.hashfull(1), 3);
    }
}
