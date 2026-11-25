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

//! Thread module - Rust port of Stockfish's thread.h/thread.cpp
//!
//! This module manages the thread pool for parallel search in the chess engine.
//! It provides thread management, synchronization, and work distribution.

use core::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;
use std::thread::{Builder, JoinHandle};
use std::{sync::Arc, sync::Condvar, sync::Mutex, sync::RwLock};

use crate::evaluate::Networks;
use crate::history::SharedHistories;
use crate::movegen::{GenType, generate};
use crate::position::Position;
use crate::search::{LimitsType, Worker};
use crate::timeman::{TimeManagement, TimePoint};
use crate::tt::TranspositionTable;
use crate::types::{
    Color, Depth, Move, VALUE_INFINITE, VALUE_NONE, VALUE_ZERO, Value, is_decisive, is_loss,
};
use crate::uci::UciEngine;
use crate::ucioption::OptionsMap;

// Platform-specific thread creation with 8MB stack size
const THREAD_STACK_SIZE: usize = 8 * 1024 * 1024; // 8MB

/// Create a native thread with proper stack size
fn create_native_thread<F, T>(name: &str, f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    Builder::new()
        .name(name.to_owned())
        .stack_size(THREAD_STACK_SIZE)
        .spawn(f)
        .unwrap_or_else(|e| panic!("Failed to create thread {}: {}", name, e))
}

/// Optional thread to NUMA node binder
///
/// Sometimes we don't want to actually bind the threads, but the recipient still
/// needs to think it runs on *some* NUMA node, such that it can access structures
/// that rely on NUMA node knowledge. This class encapsulates this optional process
/// such that the recipient does not need to know whether the binding happened or not.
#[derive(Clone)]
pub struct OptionalThreadToNumaNodeBinder {
    numa_config: Option<Arc<NumaConfig>>,
    numa_id: NumaIndex,
}

impl OptionalThreadToNumaNodeBinder {
    pub fn new(numa_id: NumaIndex) -> Self {
        Self {
            numa_config: None,
            numa_id,
        }
    }

    pub fn with_config(numa_config: Arc<NumaConfig>, numa_id: NumaIndex) -> Self {
        Self {
            numa_config: Some(numa_config),
            numa_id,
        }
    }

    pub fn bind(&self) -> NumaReplicatedAccessToken {
        if let Some(ref config) = self.numa_config {
            config.bind_current_thread_to_numa_node(self.numa_id)
        } else {
            NumaReplicatedAccessToken::new(self.numa_id)
        }
    }
}

// ---------------------------------------------------------------------------
// NUMA support
// ---------------------------------------------------------------------------

pub type NumaIndex = usize;

/// Token representing access to NUMA-replicated resources
#[derive(Copy, Clone, Debug)]
pub struct NumaReplicatedAccessToken {
    numa_id: NumaIndex,
}

impl NumaReplicatedAccessToken {
    pub fn new(numa_id: NumaIndex) -> Self {
        Self { numa_id }
    }

    pub fn numa_id(&self) -> NumaIndex {
        self.numa_id
    }
}

/// NUMA configuration with platform-specific binding
#[derive(Clone, Debug)]
pub struct NumaConfig {
    nodes: Vec<Vec<usize>>, // CPU indices per NUMA node
    custom_affinity: bool,
}

impl NumaConfig {
    /// The system topology, honouring the process affinity (`NumaPolicy=auto`).
    pub fn new() -> Self {
        Self::from_system(true)
    }

    /// C++ `NumaConfig::from_system(DefaultNumaPolicy, respect_affinity)`: the OS NUMA nodes,
    /// with the L3 cache domains inside a node bundled up to [`Self::DEFAULT_L3_BUNDLE_SIZE`]
    /// CPUs (`BundledL3Policy{32}`). With `respect_affinity == false` (`NumaPolicy=hardware`) the
    /// process affinity mask is ignored and every CPU the OS reports is used.
    ///
    /// The L3 grouping gains history sharing between nearby cores at a small cost in cross-cache
    /// accesses (upstream PR 6526). It needs Linux sysfs; elsewhere, or if the cache topology is
    /// unreadable, this is the plain OS NUMA layout ([`Self::from_system_numa`]).
    pub fn from_system(respect_affinity: bool) -> Self {
        let system = Self::detect_cpu_topology(respect_affinity);
        let nodes =
            Self::try_get_l3_aware_config(&system, respect_affinity, Self::DEFAULT_L3_BUNDLE_SIZE)
                .unwrap_or(system);
        Self {
            nodes,
            custom_affinity: false,
        }
    }

    /// C++ `NumaConfig::from_system(SystemNumaPolicy{}, respect_affinity)`: one node per OS NUMA node.
    pub fn from_system_numa(respect_affinity: bool) -> Self {
        Self {
            nodes: Self::detect_cpu_topology(respect_affinity),
            custom_affinity: false,
        }
    }

    /// Upstream `engine.cpp`: "The default configuration will attempt to group L3 domains up to 32
    /// threads." (`BundledL3Policy{32}`).
    pub const DEFAULT_L3_BUNDLE_SIZE: usize = 32;

    /// C++ `try_get_l3_aware_config` + `from_l3_info` (Linux): read each CPU's L3 sharing list
    /// from sysfs, then within every system NUMA node merge neighbouring L3 domains while the
    /// merged domain stays within `bundle_size` CPUs. `None` if no L3 information is available.
    #[cfg(target_os = "linux")]
    fn try_get_l3_aware_config(
        system: &[Vec<usize>],
        respect_affinity: bool,
        bundle_size: usize,
    ) -> Option<Vec<Vec<usize>>> {
        use std::collections::{BTreeMap, BTreeSet};

        let allowed = if respect_affinity {
            Self::linux_allowed_cpus()
        } else {
            None
        };
        let is_cpu_allowed = |c: usize| allowed.as_ref().is_none_or(|set| set.contains(&c));

        // CPU -> index of its system NUMA node, in ascending CPU order (C++ `nodeByCpu`).
        let node_by_cpu: BTreeMap<usize, usize> = system
            .iter()
            .enumerate()
            .flat_map(|(n, cpus)| cpus.iter().map(move |&c| (c, n)))
            .collect();

        // (system NUMA index, CPUs) per L3 domain.
        let mut domains: Vec<(usize, BTreeSet<usize>)> = Vec::new();
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        for &next in node_by_cpu.keys() {
            if seen.contains(&next) {
                continue;
            }
            let path = format!("/sys/devices/system/cpu/cpu{next}/cache/index3/shared_cpu_list");
            let Some(siblings) = crate::misc::read_file_to_string(&path).filter(|s| !s.is_empty())
            else {
                continue;
            };
            let mut node = 0;
            let mut cpus = BTreeSet::new();
            for c in Self::indices_from_shortened_string(&siblings) {
                if is_cpu_allowed(c) {
                    if let Some(&n) = node_by_cpu.get(&c) {
                        node = n;
                        cpus.insert(c);
                    }
                }
                seen.insert(c);
            }
            if !cpus.is_empty() {
                domains.push((node, cpus));
            }
        }
        if domains.is_empty() {
            return None;
        }

        let mut by_node: BTreeMap<usize, Vec<BTreeSet<usize>>> = BTreeMap::new();
        for (node, cpus) in domains {
            by_node.entry(node).or_default().push(cpus);
        }
        let mut nodes = Vec::new();
        for (_, ds) in by_node {
            nodes.extend(
                Self::bundle_l3_domains(ds, bundle_size)
                    .into_iter()
                    .map(|d| d.into_iter().collect::<Vec<_>>()),
            );
        }
        Some(nodes)
    }

    /// C++ `from_l3_info` merge step for the L3 domains of one system NUMA node: scan through
    /// neighbouring pairs and merge them while the result stays within `bundle_size` CPUs.
    /// Only the Linux sysfs discovery calls it; the unit tests exercise it on every target.
    #[cfg(any(target_os = "linux", test))]
    fn bundle_l3_domains(
        mut ds: Vec<std::collections::BTreeSet<usize>>,
        bundle_size: usize,
    ) -> Vec<std::collections::BTreeSet<usize>> {
        loop {
            let mut changed = false;
            let mut j = 0;
            while j + 1 < ds.len() {
                if ds[j].len() + ds[j + 1].len() <= bundle_size {
                    changed = true;
                    let next = ds.remove(j + 1);
                    ds[j].extend(next);
                }
                j += 1;
            }
            if !changed {
                break;
            }
        }
        ds
    }

    /// The L3 grouping is only implemented for Linux (sysfs); other systems keep the OS layout.
    #[cfg(not(target_os = "linux"))]
    fn try_get_l3_aware_config(
        _system: &[Vec<usize>],
        _respect_affinity: bool,
        _bundle_size: usize,
    ) -> Option<Vec<Vec<usize>>> {
        None
    }

    /// C++ default-constructed `NumaConfig` (`NumaPolicy=none`): every CPU in a
    /// single node and no custom affinity, so threads are never bound.
    pub fn single_node() -> Self {
        let count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            nodes: vec![(0..count).collect()],
            custom_affinity: false,
        }
    }

    /// C++ `NumaConfig::from_string`: `:` separates NUMA nodes, `,` separates CPU
    /// indices and `a-b` is an inclusive range, e.g. `0-15,128-143:16-31`.
    /// Returns `None` if no node could be parsed or a CPU is assigned twice.
    /// Like C++, unparsable items (and ranges of 2^20 CPUs or more) are skipped, and a node
    /// with no valid index does not count.
    pub fn from_string(s: &str) -> Option<Self> {
        let mut nodes: Vec<Vec<usize>> = Vec::new();
        let mut assigned = std::collections::HashSet::new();
        for node_str in crate::misc::split(s, ":") {
            let mut cpus = Self::indices_from_shortened_string(node_str);
            if cpus.is_empty() {
                continue;
            }
            for &cpu in &cpus {
                if !assigned.insert(cpu) {
                    return None;
                }
            }
            cpus.sort_unstable();
            nodes.push(cpus);
        }
        if nodes.is_empty() {
            return None; // failed to parse any nodes
        }
        Some(Self {
            nodes,
            custom_affinity: true,
        })
    }

    /// C++ `NumaConfig::indices_from_shortened_string`.
    fn indices_from_shortened_string(s: &str) -> Vec<usize> {
        use crate::misc::{split, str_to_size_t};
        const MAX_INDICES: usize = 1 << 20; // prevent OOM
        let mut indices = Vec::new();
        for ss in split(s, ",") {
            if ss.is_empty() {
                continue;
            }
            let parts = split(ss, "-");
            if parts.len() == 1 {
                if let Some(c) = str_to_size_t(parts[0]) {
                    indices.push(c);
                }
            } else if parts.len() == 2 {
                if let (Some(first), Some(last)) =
                    (str_to_size_t(parts[0]), str_to_size_t(parts[1]))
                {
                    // An inverted range (first > last) is ignored, as is one that is too large.
                    if first <= last {
                        let span = last - first;
                        if span < MAX_INDICES {
                            indices.extend((0..=span).map(|k| first + k));
                        }
                    }
                }
            }
        }
        indices
    }

    /// Number of NUMA nodes (C++ `num_numa_nodes`).
    pub fn num_numa_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of CPUs in node `n`, or 0 if there is no such node.
    pub fn num_cpus_in_numa_node(&self, n: NumaIndex) -> usize {
        self.nodes.get(n).map_or(0, Vec::len)
    }

    fn detect_cpu_topology(respect_affinity: bool) -> Vec<Vec<usize>> {
        #[cfg(target_os = "linux")]
        {
            let allowed = if respect_affinity {
                Self::linux_allowed_cpus()
            } else {
                None
            };
            let mut dirs: Vec<_> = std::fs::read_dir("/sys/devices/system/node/")
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .filter(|entry| {
                    entry.file_name().to_str().is_some_and(|name| {
                        name.strip_prefix("node")
                            .is_some_and(|n| n.chars().all(|c| c.is_ascii_digit()))
                    })
                })
                .collect();
            // Numeric order, so that node10 follows node9 rather than node1.
            dirs.sort_by_key(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .and_then(|n| n.strip_prefix("node"))
                    .and_then(|n| n.parse::<usize>().ok())
            });
            let mut nodes = Vec::new();
            for entry in dirs {
                let mut cpus: Vec<_> = std::fs::read_to_string(entry.path().join("cpulist"))
                    .ok()
                    .map(|text| Self::parse_cpu_list(&text))
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|cpu| allowed.as_ref().is_none_or(|set| set.contains(cpu)))
                    .collect();
                cpus.sort_unstable();
                cpus.dedup();
                if !cpus.is_empty() {
                    nodes.push(cpus);
                }
            }
            if !nodes.is_empty() {
                return nodes;
            }
            let cpus: Vec<usize> =
                allowed
                    .map(|set| set.into_iter().collect())
                    .unwrap_or_else(|| {
                        (0..std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(1))
                            .collect()
                    });
            vec![cpus]
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = respect_affinity;
            let count = std::thread::available_parallelism().map_or(1, |n| n.get());
            vec![(0..count).collect()]
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_allowed_cpus() -> Option<std::collections::BTreeSet<usize>> {
        use nix::sched::{CpuSet, sched_getaffinity};
        use nix::unistd::Pid;
        // A fixed `cpu_set_t` (1024 CPUs), as before; pid 0 is the calling thread.
        let set = sched_getaffinity(Pid::from_raw(0)).ok()?;
        Some(
            (0..CpuSet::count())
                .filter(|&cpu| set.is_set(cpu) == Ok(true))
                .collect(),
        )
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_list(text: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        for item in text.trim().split(',').filter(|item| !item.is_empty()) {
            match item.split_once('-') {
                Some((start, end)) => {
                    if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                        if start <= end {
                            cpus.extend(start..=end);
                        }
                    }
                }
                None => {
                    if let Ok(cpu) = item.parse::<usize>() {
                        cpus.push(cpu);
                    }
                }
            }
        }
        cpus
    }

    /// Port of C++ `suggests_binding_threads`: bind only when the OS could not
    /// keep the threads inside the first node, ignoring small/empty nodes.
    pub fn suggests_binding_threads(&self, num_threads: usize) -> bool {
        // A user-specified affinity may differ from the OS one, so binding is needed.
        if self.custom_affinity {
            return true;
        }
        // A single thread cannot be distributed.
        if num_threads <= 1 {
            return false;
        }
        let largest = self.nodes.iter().map(Vec::len).max().unwrap_or(0);
        let not_small = self
            .nodes
            .iter()
            .filter(|cpus| cpus.len() as f64 / largest as f64 > 0.6)
            .count();
        (num_threads > largest / 2 || num_threads >= not_small * 4) && self.nodes.len() > 1
    }

    /// Port of C++ `distribute_threads_among_numa_nodes`: each thread goes to the
    /// node that would end up the least full.
    pub fn distribute_threads_among_numa_nodes(&self, num_threads: usize) -> Vec<NumaIndex> {
        if self.nodes.len() <= 1 {
            return vec![0; num_threads];
        }
        let mut occupation = vec![0usize; self.nodes.len()];
        let mut distribution = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let mut best = 0;
            let mut best_fill = f32::MAX;
            for (n, cpus) in self.nodes.iter().enumerate() {
                let fill = (occupation[n] + 1) as f32 / cpus.len() as f32;
                if fill < best_fill {
                    best = n;
                    best_fill = fill;
                }
            }
            distribution.push(best);
            occupation[best] += 1;
        }
        distribution
    }

    pub fn execute_on_numa_node<F>(&self, numa_id: NumaIndex, f: F)
    where
        F: FnOnce() + Send,
    {
        // The caller's affinity must remain intact, including during unwinding.
        // Allocations inherit the target affinity in a temporary scoped thread.
        #[cfg(target_os = "linux")]
        std::thread::scope(|scope| {
            let job = scope.spawn(|| {
                self.bind_current_thread_to_numa_node(numa_id);
                f();
            });
            if let Err(panic) = job.join() {
                std::panic::resume_unwind(panic);
            }
        });
        #[cfg(not(target_os = "linux"))]
        {
            let _ = numa_id;
            f();
        }
    }

    pub fn bind_current_thread_to_numa_node(
        &self,
        numa_id: NumaIndex,
    ) -> NumaReplicatedAccessToken {
        // Platform-specific thread binding. Like C++ `bind_current_thread_to_numa_node`, a request
        // that cannot be honoured (no such node, or the kernel rejects the mask, e.g. because none
        // of the CPUs exists) terminates the process with exit code 1 and no message.
        if numa_id >= self.nodes.len() || self.nodes[numa_id].is_empty() {
            std::process::exit(1);
        }

        #[cfg(target_os = "linux")]
        {
            use core::mem::size_of;

            // Dynamically sized mask (C++ `CPU_ALLOC(highestCpuIndex + 1)`), so CPU ids beyond
            // 1023 are honoured instead of skipped. This is why it is not `nix::sched`, whose
            // `CpuSet` is a fixed `cpu_set_t`.
            const BITS: usize = libc::c_ulong::BITS as usize;
            let highest = self.nodes[numa_id].iter().copied().max().unwrap_or(0);
            let words = highest / BITS + 1;
            let mut mask: Vec<libc::c_ulong> = Vec::new();
            if mask.try_reserve_exact(words).is_err() {
                std::process::exit(1);
            }
            mask.resize(words, 0);
            for &cpu in &self.nodes[numa_id] {
                mask[cpu / BITS] |= 1 << (cpu % BITS);
            }
            // SAFETY: `mask` holds `words * size_of::<c_ulong>()` initialised bytes, the size passed
            // to sched_setaffinity, and the kernel only reads from the pointer.
            let status = unsafe {
                libc::sched_setaffinity(0, words * size_of::<libc::c_ulong>(), mask.as_ptr().cast())
            };
            if status != 0 {
                std::process::exit(1);
            }
            // Defensive reschedule, as in C++ (not performance critical). On Unix
            // `yield_now` is `sched_yield`.
            std::thread::yield_now();
        }

        #[cfg(target_os = "windows")]
        {
            // Not ported: Windows processor groups and affinity masks
            // (`SetThreadGroupAffinity`, `GetNumaNodeProcessorMask`) from C++
            // thread.cpp / numa.cpp. NUMA thread binding is not implemented on
            // Windows; threads run unbound.
        }

        NumaReplicatedAccessToken::new(numa_id)
    }
}

/// C++ `NumaConfig::to_string`: nodes joined by `:`, consecutive CPUs collapsed to
/// `a-b`, ranges joined by `,`.
impl core::fmt::Display for NumaConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, cpus) in self.nodes.iter().enumerate() {
            if i > 0 {
                f.write_str(":")?;
            }
            let mut first_range = true;
            let mut k = 0;
            while k < cpus.len() {
                let start = cpus[k];
                let mut end = start;
                while k + 1 < cpus.len() && cpus[k + 1] == end + 1 {
                    k += 1;
                    end = cpus[k];
                }
                if !first_range {
                    f.write_str(",")?;
                }
                if end == start {
                    write!(f, "{start}")?;
                } else {
                    write!(f, "{start}-{end}")?;
                }
                first_range = false;
                k += 1;
            }
        }
        Ok(())
    }
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Search-related types
// ---------------------------------------------------------------------------

/// Root move with score and principal variation
///
/// RootMove struct is used for moves at the root of the tree. For each root move
/// we store a score and a PV (really a refutation in the case of moves which
/// fail low). Score is normally set at -VALUE_INFINITE for all non-pv moves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootMove {
    pub mv: Move,
    pub score: Value,
    pub pv: Vec<Move>,
    /// This move's PV as of the end of the previous iteration.
    ///
    /// Must live on the RootMove rather than in a positional side-table: the
    /// root move list is re-sorted during the MultiPV loop, so an index into a
    /// snapshot taken before sorting no longer refers to the same move. C++
    /// stores it here for exactly that reason (search.h).
    pub previous_pv: Vec<Move>,
    pub effort: u64,
    pub previous_score: Value,
    /// Whether `previous_score` came from a fully searched PV line of the previous
    /// iteration (`i < multi_pv` when it was saved), as opposed to a bound.
    pub previous_score_exact: bool,
    pub average_score: Value,
    pub mean_squared_score: Value,
    pub uci_score: Value,
    /// Root scores are exact unless one of these marks a one-sided bound.
    pub inexact_lower: bool,
    pub inexact_upper: bool,
    pub sel_depth: i32,
    pub tb_rank: i32,
    pub tb_score: Value,
}

impl RootMove {
    pub fn new(mv: Move) -> Self {
        Self {
            mv,
            score: -VALUE_INFINITE,
            pv: vec![mv],
            previous_pv: Vec::new(),
            effort: 0,
            previous_score: -VALUE_INFINITE,
            previous_score_exact: false,
            average_score: -VALUE_INFINITE,
            mean_squared_score: -VALUE_INFINITE * VALUE_INFINITE,
            uci_score: -VALUE_INFINITE,
            inexact_lower: false,
            inexact_upper: false,
            sel_depth: 0,
            tb_rank: 0,
            tb_score: VALUE_NONE,
        }
    }

    /// True when the stored score is only a bound (not exact).
    pub fn is_inexact(&self) -> bool {
        self.inexact_lower || self.inexact_upper
    }

    /// True when the score is an exact loss (not a bound).
    pub fn is_exact_loss(&self) -> bool {
        self.score != -VALUE_INFINITE && is_loss(self.score) && !self.is_inexact()
    }

    /// Clear the lower/upper inexact flags.
    pub fn unset_inexact(&mut self) {
        self.inexact_lower = false;
        self.inexact_upper = false;
    }

    /// Extract ponder move from transposition table
    /// Use the TT to extract a ponder move after the best move has been found.
    /// Faithful port of C++ `RootMove::extract_ponder_from_tt` (search.cpp).
    pub fn extract_ponder_from_tt(&mut self, tt: &TranspositionTable, pos: &Position) -> bool {
        if self.pv.len() != 1 || !pos.legal(self.pv[0]) {
            return false;
        }

        let mut pos_copy = (*pos).clone();

        // SAFETY: the PV move was just checked to be legal.
        unsafe { pos_copy.do_move(self.pv[0]) };

        if !pos_copy.is_draw(1) {
            let (found, data, _writer) = tt.probe(pos_copy.key());
            if found {
                let tt_move = data.mv;
                if tt_move != Move::none() && pos_copy.legal(tt_move) {
                    self.pv.push(tt_move);
                }
            }
        }

        // pos_copy is a local clone, so no undo_move is needed here.
        self.pv.len() > 1
    }
}

// Implement custom ordering for RootMove to match C++ behavior
// C++ sorts in descending order by score (highest score first)
impl PartialOrd for RootMove {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RootMove {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Sort in descending order by score (higher scores first)
        // This matches C++ operator< which returns m.score < score
        if self.score != other.score {
            other.score.cmp(&self.score) // Note: reversed order for descending
        } else {
            // Use previous_score as tiebreaker (also descending)
            other.previous_score.cmp(&self.previous_score)
        }
    }
}

pub type RootMoves = Vec<RootMove>;

/// Search control flags - extracted into Arc for lock-free access
/// These atomics control search state and are accessed on every node,
/// so they must not require lock acquisition.
///
/// `ponder` is the main `SearchManager`'s ponder flag (C++ `main_manager()->ponder`).
/// It is shared here because the main worker is checked out of its `Thread` for the
/// whole search, so `ponderhit` cannot reach the manager through the thread.
#[derive(Debug)]
pub struct SearchControl {
    pub stop: AtomicBool,
    pub increase_depth: AtomicBool,
    pub ponder: Arc<AtomicBool>,
}

impl SearchControl {
    pub fn new() -> Self {
        Self {
            stop: AtomicBool::new(false),
            increase_depth: AtomicBool::new(true),
            ponder: Arc::new(AtomicBool::new(false)),
        }
    }

    #[inline(always)]
    pub fn stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn set_stop(&self, value: bool) {
        self.stop.store(value, Ordering::SeqCst);
    }

    #[inline(always)]
    pub fn increase_depth(&self) -> bool {
        self.increase_depth.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn set_increase_depth(&self, value: bool) {
        self.increase_depth.store(value, Ordering::Relaxed);
    }

    /// Sets the main manager's ponder flag (UCI `ponderhit` clears it).
    pub fn set_ponder(&self, value: bool) {
        self.ponder.store(value, Ordering::Relaxed);
    }
}

impl Default for SearchControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared state between all search threads
///
/// The UCI stores the uci options, thread pool, and transposition table.
/// This struct is used to easily forward data to the Search::Worker class.
#[derive(Clone)]
pub struct SharedState {
    pub options: Arc<RwLock<OptionsMap>>,
    pub threads: Arc<RwLock<ThreadPool>>,
    pub tt: Arc<RwLock<TranspositionTable>>,
    pub networks: Arc<crate::engine::LazyNumaReplicated<Networks>>,
    pub search_control: Arc<SearchControl>,
    /// Shared history tables between threads (C++ SharedHistories).
    pub shared_histories: Arc<SharedHistories>,
}

impl SharedState {
    pub fn new(
        options: Arc<RwLock<OptionsMap>>,
        threads: Arc<RwLock<ThreadPool>>,
        tt: Arc<RwLock<TranspositionTable>>,
        networks: Arc<crate::engine::LazyNumaReplicated<Networks>>,
        search_control: Arc<SearchControl>,
        shared_histories: Arc<SharedHistories>,
    ) -> Self {
        Self {
            options,
            threads,
            tt,
            networks,
            search_control,
            shared_histories,
        }
    }

    pub fn from_components(
        options: &Arc<RwLock<OptionsMap>>,
        thread_pool: &Arc<RwLock<ThreadPool>>,
        tt: &Arc<RwLock<TranspositionTable>>,
        networks: &Arc<crate::engine::LazyNumaReplicated<Networks>>,
        search_control: &Arc<SearchControl>,
        shared_histories: &Arc<SharedHistories>,
    ) -> Self {
        Self {
            options: Arc::clone(options),
            threads: Arc::clone(thread_pool),
            tt: Arc::clone(tt),
            networks: Arc::clone(networks),
            search_control: Arc::clone(search_control),
            shared_histories: Arc::clone(shared_histories),
        }
    }
}

/// Search update callbacks
pub struct UpdateContext {
    pub on_update_no_moves: Box<dyn Fn(InfoShort) + Send + Sync>,
    pub on_update_full: Box<dyn Fn(InfoFull) + Send + Sync>,
    pub on_iter: Box<dyn Fn(InfoIteration) + Send + Sync>,
    pub on_bestmove: Box<dyn Fn(&str, &str) + Send + Sync>,
    /// C++ `UpdateStart onStart` - invoked once when the main thread starts a search
    pub on_start: Box<dyn Fn() + Send + Sync>,
}

/// Short search info
/// C++ `Search::InfoShort` (search.h) stores an already converted `Score`,
/// because the Value -> centipawn conversion is material (i.e. position) dependent.
pub struct InfoShort {
    pub depth: i32,
    pub score: crate::score::Score,
}

/// Full search info
pub struct InfoFull {
    pub depth: i32,
    pub score: crate::score::Score,
    pub sel_depth: i32,
    pub multi_pv: usize,
    pub wdl: String,
    pub bound: String,
    pub time_ms: u64,
    pub nodes: u64,
    pub nps: u64,
    pub tb_hits: u64,
    pub pv: String,
    pub hashfull: i32,
}

/// Iteration info
pub struct InfoIteration {
    pub depth: i32,
    pub currmove: String,
    pub currmovenumber: usize,
}

/// Main search manager
///
/// SearchManager manages the search from the main thread. It is responsible for
/// keeping track of the time, and storing data strictly related to the main thread.
pub struct SearchManager {
    pub best_previous_score: AtomicI32,
    pub best_previous_average_score: AtomicI32,
    pub previous_time_reduction: RwLock<f64>,
    pub calls_cnt: AtomicU32,
    pub original_time_adjust: RwLock<f64>,
    pub stop_on_ponderhit: AtomicBool,
    pub ponder: Arc<AtomicBool>,
    pub tm: RwLock<TimeManagement>,
    pub iter_value: RwLock<[Value; 4]>,
    pub id: usize,
    pub updates: Arc<RwLock<UpdateContext>>,
}

impl SearchManager {
    pub fn new(updates: Arc<RwLock<UpdateContext>>) -> Self {
        Self::with_ponder(updates, Arc::new(AtomicBool::new(false)))
    }

    /// Creates a manager whose ponder flag is the shared `ponder` atomic.
    pub fn with_ponder(updates: Arc<RwLock<UpdateContext>>, ponder: Arc<AtomicBool>) -> Self {
        Self {
            best_previous_score: AtomicI32::new(VALUE_INFINITE),
            best_previous_average_score: AtomicI32::new(VALUE_INFINITE),
            previous_time_reduction: RwLock::new(0.85),
            calls_cnt: AtomicU32::new(0),
            original_time_adjust: RwLock::new(-1.0),
            stop_on_ponderhit: AtomicBool::new(false),
            ponder,
            tm: RwLock::new(TimeManagement::new()),
            iter_value: RwLock::new([VALUE_NONE; 4]),
            id: 0,
            updates,
        }
    }
}

impl SearchManager {
    pub fn clear(&mut self) {
        self.calls_cnt.store(0, Ordering::Relaxed);
        self.best_previous_score
            .store(VALUE_INFINITE, Ordering::Relaxed);
        self.best_previous_average_score
            .store(VALUE_INFINITE, Ordering::Relaxed);
        *self.previous_time_reduction.write().unwrap() = 0.85;
        *self.original_time_adjust.write().unwrap() = -1.0;
        self.tm.write().unwrap().clear();
    }

    /// Stop the search when a limit is reached. Port of C++ `SearchManager::check_time`
    /// (search.cpp). Called on the main thread only.
    // The countdown fast path is inlined into every search node; the (rare) body is out of line
    // so the node does not pay for a call plus six callee-saved pushes/pops (LLVM does not
    // shrink-wrap them past the early return, GCC does for C++).
    #[inline(always)]
    pub fn check_time(&self, worker: &Worker) {
        // C++ counts down and re-arms with a period that shrinks when a node
        // limit is set, so the limit is never overshot by more than ~0.1%.
        // Only the main thread reaches here, so the load/store pair is race-free.
        // A plain load/store pair rather than `fetch_sub`: the latter is a `lock`-prefixed RMW
        // (~20 cycles) on every node, whereas C++ uses a plain `--callsCnt`. Only the main
        // thread's search calls this (the counter is also reset from the main thread while no
        // search is running), so there is no concurrent writer to lose an update to.
        let cnt = self.calls_cnt.load(Ordering::Relaxed);
        if cnt > 1 {
            self.calls_cnt.store(cnt - 1, Ordering::Relaxed);
            return;
        }
        self.check_time_slow(worker);
    }

    #[cold]
    #[inline(never)]
    fn check_time_slow(&self, worker: &Worker) {
        let period = if worker.limits.nodes != 0 {
            core::cmp::min(512, (worker.limits.nodes / 1024).max(1)) as u32
        } else {
            512
        };
        self.calls_cnt.store(period, Ordering::Relaxed);

        // Summing nodes across threads takes a read guard on the thread-pool
        // lock, so only do it when a limit actually needs the count. For plain
        // `movetime` and `wtime`/`btime` searches this stays lock-free, which
        // matters because this runs on every countdown expiry.
        let (uses_nodes_time, maximum) = {
            let tm = self.tm.read().unwrap();
            (tm.uses_nodes_time(), tm.maximum())
        };
        let nodes = if uses_nodes_time || worker.limits.nodes != 0 {
            worker
                .shared_state
                .threads
                .read()
                .unwrap()
                .nodes_searched_from_worker(worker)
        } else {
            0
        };

        // With `nodestime` this returns consumed nodes rather than milliseconds,
        // which is the whole point of that option.
        let elapsed = self.tm.read().unwrap().elapsed(|| nodes as i64);

        // We should not stop pondering until told so by the GUI.
        if self.ponder.load(Ordering::Relaxed) {
            return;
        }

        let hit_time_management = worker.limits.use_time_management()
            && (elapsed > maximum || self.stop_on_ponderhit.load(Ordering::Relaxed));
        let hit_movetime =
            worker.limits.movetime != 0 && elapsed >= worker.limits.movetime as TimePoint;
        let hit_nodes = worker.limits.nodes != 0 && nodes >= worker.limits.nodes;

        if hit_time_management || hit_movetime || hit_nodes {
            worker.shared_state.search_control.set_stop(true);
        }
    }

    // Main thread manager functionality implementations
    pub fn get_ponder(&self) -> bool {
        self.ponder.load(Ordering::Relaxed)
    }

    pub fn set_ponder(&self, value: bool) {
        self.ponder.store(value, Ordering::Relaxed);
    }

    pub fn get_best_previous_score(&self) -> Value {
        self.best_previous_score.load(Ordering::Relaxed)
    }

    pub fn set_iter_value(&self, index: usize, value: Value) {
        self.iter_value.write().unwrap()[index] = value;
    }

    pub fn get_iter_value(&self, index: usize) -> Value {
        self.iter_value.read().unwrap()[index]
    }

    pub fn fill_iter_value(&self, value: Value) {
        self.iter_value.write().unwrap().fill(value);
    }

    pub fn get_best_previous_average_score(&self) -> Value {
        self.best_previous_average_score.load(Ordering::Relaxed)
    }

    pub fn set_stop_on_ponderhit(&self, value: bool) {
        self.stop_on_ponderhit.store(value, Ordering::Relaxed);
    }

    pub fn get_stop_on_ponderhit(&self) -> bool {
        self.stop_on_ponderhit.load(Ordering::Relaxed)
    }

    pub fn set_previous_time_reduction(&self, value: f64) {
        *self.previous_time_reduction.write().unwrap() = value;
    }

    pub fn get_previous_time_reduction(&self) -> f64 {
        *self.previous_time_reduction.read().unwrap()
    }

    pub fn get_tm_optimum(&self) -> u64 {
        self.tm.read().unwrap().optimum() as u64
    }

    pub fn get_tm_maximum(&self) -> u64 {
        self.tm.read().unwrap().maximum() as u64
    }

    // Time management methods
    pub fn init_time_management(
        &self,
        limits: &mut LimitsType,
        us: Color,
        ply: i32,
        options: &OptionsMap,
        original_time_adjust: &mut f64,
    ) {
        *self.original_time_adjust.write().unwrap() = *original_time_adjust;
        self.tm
            .write()
            .unwrap()
            .init(limits, us, ply, options, original_time_adjust);
    }

    pub fn advance_nodes_time(&self, nodes_used: i64) {
        self.tm.write().unwrap().advance_nodes_time(nodes_used);
    }

    pub fn elapsed<F: Fn() -> i64>(&self, nodes_fn: F) -> u64 {
        self.tm.read().unwrap().elapsed(nodes_fn) as u64
    }

    pub fn elapsed_time(&self) -> u64 {
        self.tm.read().unwrap().elapsed_time() as u64
    }

    pub fn reset_calls_cnt(&self) {
        self.calls_cnt.store(0, Ordering::Relaxed);
    }

    // Score storage methods
    pub fn set_best_previous_score(&self, score: Value) {
        self.best_previous_score.store(score, Ordering::Relaxed);
    }

    pub fn set_best_previous_average_score(&self, score: Value) {
        self.best_previous_average_score
            .store(score, Ordering::Relaxed);
    }

    // UCI callback methods
    pub fn call_on_update_no_moves(&self, info: InfoShort) {
        (self.updates.read().unwrap().on_update_no_moves)(info);
    }

    pub fn call_on_bestmove(&self, bestmove: &str, ponder: &str) {
        (self.updates.read().unwrap().on_bestmove)(bestmove, ponder);
    }

    pub fn call_on_iter(&self, info: InfoIteration) {
        (self.updates.read().unwrap().on_iter)(info);
    }

    pub fn call_on_start(&self) {
        (self.updates.read().unwrap().on_start)();
    }
}

impl SearchManager {
    /// Output PV information for UCI.
    /// Faithful port of the C++ PV emitter (search.cpp).
    ///
    /// C++ `output_pv` also runs `syzygy_extend_pv` on each printed root move.
    /// That needs the worker mutably, so `Worker::extend_tb_pvs` does it just
    /// before this call and hands over the scores it left in `extended_scores`
    /// (indexed like the root moves; `None` or missing where no extension ran).
    pub fn pv(
        &self,
        worker: &Worker,
        threads: &ThreadPool,
        tt: &TranspositionTable,
        depth: Depth,
        extended_scores: &[Option<Value>],
    ) {
        // Always use the deadlock-safe versions that avoid locking the current worker
        let nodes = threads.nodes_searched_from_worker(worker);
        let pos = &worker.root_pos;
        let root_in_tb = worker.tb_config.root_in_tb;
        let tb_hits = threads.tb_hits_from_worker(worker)
            + if root_in_tb {
                worker.root_moves.len() as u64
            } else {
                0
            };

        let show_wdl = worker
            .shared_state
            .options
            .read()
            .unwrap()
            .get("UCI_ShowWDL")
            .and_then(|o| o.get_int())
            .map(|v| v != 0)
            .unwrap_or(false);
        let multi_pv = worker
            .shared_state
            .options
            .read()
            .unwrap()
            .get("MultiPV")
            .and_then(|o| o.get_int())
            .unwrap_or(1) as usize;
        let multi_pv = multi_pv.min(worker.root_moves.len());

        for i in 0..multi_pv {
            let rm = &worker.root_moves[i];
            let use_previous_score = rm.score == -VALUE_INFINITE;

            if depth == 1 && use_previous_score && i > 0 {
                continue;
            }

            let d = if use_previous_score {
                core::cmp::max(1, depth - 1)
            } else {
                depth
            };
            let mut v = if use_previous_score {
                rm.previous_score
            } else {
                rm.uci_score
            };

            if v == -VALUE_INFINITE {
                v = VALUE_ZERO;
            }

            let is_tb_score = root_in_tb && !crate::types::is_mate_or_mated(v);
            if is_tb_score {
                v = rm.tb_score;
            }

            // In exceptional cases the PV extension corrected v.
            if let Some(&Some(extended)) = extended_scores.get(i) {
                v = extended;
            }

            let pv = if use_previous_score {
                &rm.previous_pv
            } else {
                &rm.pv
            };
            let pv_string = pv
                .iter()
                .map(|&mv| crate::uci::UciEngine::move_str(mv, pos.is_chess960()))
                .collect::<Vec<String>>()
                .join(" ");

            let wdl = if show_wdl {
                crate::uci::UciEngine::wdl(v, pos)
            } else {
                String::new()
            };

            // TB and previous scores are exact, even though their bound flags may say otherwise.
            let bound = if is_tb_score || use_previous_score {
                String::new()
            } else if rm.inexact_lower {
                "lowerbound".to_string()
            } else if rm.inexact_upper {
                "upperbound".to_string()
            } else {
                String::new()
            };

            let time = core::cmp::max(1, self.tm.read().unwrap().elapsed_time()) as u64;

            let info = crate::thread::InfoFull {
                depth: d,
                sel_depth: rm.sel_depth,
                multi_pv: i + 1,
                // The centipawn conversion is material dependent, so it must be done
                // against the ROOT position (C++ `info.score = {v, pos}`).
                score: crate::score::Score::from_value(v, pos),
                nodes,
                nps: nodes * 1000 / time,
                tb_hits,
                time_ms: time,
                pv: pv_string,
                hashfull: tt.hashfull_default() as i32,
                wdl,
                bound,
            };

            (self.updates.read().unwrap().on_update_full)(info);
        }
    }
}

// Worker moves exclusively between the command thread and its search thread.
// SAFETY: Assumed, not machine-checked: a Worker is only touched by the thread that owns it,
// behind the `Mutex<Option<Worker>>` in its Thread; cross-thread observation goes through
// atomics. Its raw pointers are dereferenced only by that owning thread.
unsafe impl Send for Worker {}

/// Tablebase configuration
#[derive(Clone, Debug, Default)]
pub struct TablebaseConfig {
    pub cardinality: u32,
    pub root_in_tb: bool,
    pub probe_depth: Depth,
    pub use_rule50: bool,
    pub(crate) tables: crate::syzygy::TablebaseSnapshot,
}

// ---------------------------------------------------------------------------
// Thread management
// ---------------------------------------------------------------------------

/// Job type that can be executed by a thread
type Job = Box<dyn FnOnce() + Send>;

/// Thread state for synchronization
struct ThreadState {
    searching: bool,
    exit: bool,
    job: Option<Job>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

/// Return a worker even when a search callback unwinds. No lock is held while
/// search runs; restoration happens once per job, never per node.
struct TakenWorker {
    slot: Arc<Mutex<Option<Box<Worker>>>>,
    worker: Option<Box<Worker>>,
}

impl TakenWorker {
    fn new(slot: Arc<Mutex<Option<Box<Worker>>>>) -> Self {
        let worker = slot.lock().unwrap().take();
        Self { slot, worker }
    }
}

impl Drop for TakenWorker {
    fn drop(&mut self) {
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = self.worker.take();
    }
}

/// Per-thread statistics that other threads read while the search runs (C++ keeps these as
/// atomics in `Search::Worker` and `ThreadPool::accumulate` reads them relaxed). Cache-line
/// aligned so a helper's counter updates never share a line with another thread's data.
#[derive(Default)]
#[repr(align(128))]
pub struct WorkerCounters {
    pub nodes: AtomicU64,
    pub tb_hits: AtomicU64,
    pub best_move_changes: AtomicUsize,
}

/// Single search thread
pub struct Thread {
    counters: Arc<WorkerCounters>,
    /// Live control state remains accessible while the worker is searching.
    ponder: Option<Arc<AtomicBool>>,
    id: usize,
    handle: Option<JoinHandle<()>>,
    state: Arc<Mutex<ThreadState>>,
    cv: Arc<Condvar>,
    worker: Arc<Mutex<Option<Box<Worker>>>>,
}

impl Thread {
    pub fn new(
        shared_state: SharedState,
        manager: Option<Box<SearchManager>>,
        id: usize,
        binder: OptionalThreadToNumaNodeBinder,
    ) -> Arc<Self> {
        let total = shared_state
            .options
            .read()
            .unwrap()
            .get_int("Threads", 1)
            .max(1) as usize;
        Self::new_with_numa(shared_state, manager, id, binder, id, total)
    }

    fn new_with_numa(
        shared_state: SharedState,
        manager: Option<Box<SearchManager>>,
        id: usize,
        binder: OptionalThreadToNumaNodeBinder,
        numa_thread_idx: usize,
        numa_total: usize,
    ) -> Arc<Self> {
        let ponder = manager.as_ref().map(|m| Arc::clone(&m.ponder));
        let state = Arc::new(Mutex::new(ThreadState {
            searching: true, // Cleared by idle_loop once the thread is ready (as C++ Thread::Thread)
            exit: false,
            job: None,
            panic: None,
        }));
        let cv = Arc::new(Condvar::new());
        let worker = Arc::new(Mutex::new(None));

        let state_clone = Arc::clone(&state);
        let cv_clone = Arc::clone(&cv);
        let worker_clone = Arc::clone(&worker);
        let counters = Arc::new(WorkerCounters::default());
        let counters_clone = Arc::clone(&counters);
        let shared_state_clone = shared_state.clone();
        let search_control = Arc::clone(&shared_state.search_control);

        let thread_name = format!("Worker-{}", id);
        let handle = create_native_thread(&thread_name, move || {
            // Bind to NUMA node and create worker immediately
            let numa_token = binder.bind();
            let worker_instance =
                // Histories and clear ranges use indices local to this NUMA group.
                Box::new(Worker::new(
                    shared_state_clone,
                    manager,
                    id,
                    numa_thread_idx,
                    numa_total,
                    numa_token,
                ));
            let mut worker_instance = worker_instance;
            worker_instance.counters = counters_clone;
            *worker_clone.lock().unwrap() = Some(worker_instance);

            // Run idle loop
            Self::idle_loop(state_clone, cv_clone, search_control);
        });

        let thread = Arc::new(Self {
            counters,
            ponder,
            id,
            handle: Some(handle),
            state,
            cv,
            worker,
        });

        // Wait for thread to be ready
        thread.wait_for_search_finished();

        // The worker is created on the thread itself; this empty job is a barrier that
        // ensures creation finished before `new()` returns.
        thread.run_custom_job(move || {});
        thread.wait_for_search_finished();

        thread
    }

    // Thread idle loop implementation from thread.cpp
    /// Thread gets parked here, blocked on the condition variable
    /// when the thread has no work to do.
    fn idle_loop(
        state: Arc<Mutex<ThreadState>>,
        cv: Arc<Condvar>,
        search_control: Arc<SearchControl>,
    ) {
        loop {
            // Lock and set searching to false
            let mut s = state.lock().unwrap();
            s.searching = false;
            cv.notify_all(); // Every completion waiter must be released, including on panic.

            // Wait until we have work to do
            while !s.searching {
                s = cv.wait(s).unwrap();
            }

            // Check for exit condition
            if s.exit {
                return;
            }

            // Move job out of state and unlock before executing
            let job = s.job.take();
            drop(s); // Explicit unlock

            // Execute the job if we have one
            if let Some(job) = job {
                if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
                    // A failed main or helper search must also release the other
                    // workers. Save the failure for the waiting library caller.
                    search_control.set_stop(true);
                    state.lock().unwrap().panic = Some(payload);
                }
            }
        }
    }

    pub fn start_searching(&self) {
        let worker = Arc::clone(&self.worker);
        self.run_custom_job(move || {
            let mut local_worker = TakenWorker::new(worker);
            if let Some(ref mut w) = local_worker.worker {
                w.start_searching();
            }
        });
    }

    pub fn set_ponder(&self, value: bool) {
        if let Some(ponder) = &self.ponder {
            ponder.store(value, Ordering::Relaxed);
        }
    }

    /// Clears the histories for the thread worker (usually before a new game)
    pub fn clear_worker(&self) {
        let worker = Arc::clone(&self.worker);
        self.run_custom_job(move || {
            if let Some(ref mut w) = *worker.lock().unwrap() {
                w.clear();
            }
        });
    }

    /// Launching a function in the thread
    pub fn run_custom_job<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        {
            let mut state = self.state.lock().unwrap();
            // Wait if thread is busy
            while state.searching {
                state = self.cv.wait(state).unwrap();
            }
            if let Some(payload) = state.panic.take() {
                drop(state);
                std::panic::resume_unwind(payload);
            }
            state.job = Some(Box::new(f));
            state.searching = true;
        }
        // The worker and completion waiters share this condition variable;
        // waking only a waiter can leave the new job unstarted.
        self.cv.notify_all();
    }

    /// Blocks until completion, then propagates a job panic to the waiting caller.
    /// After catching that panic the caller can submit another job.
    pub fn wait_for_search_finished(&self) {
        if let Some(payload) = self.wait_for_completion() {
            std::panic::resume_unwind(payload);
        }
    }

    fn wait_for_completion(&self) -> Option<Box<dyn std::any::Any + Send>> {
        let mut state = self.state.lock().unwrap();
        while state.searching {
            state = self.cv.wait(state).unwrap();
        }
        state.panic.take()
    }

    /// Destructors must drain workers without causing a second unwind.
    pub(crate) fn wait_for_search_finished_without_unwind(&self) {
        let _ = self.wait_for_completion();
    }

    pub fn ensure_network_replicated(&self) {
        let worker = Arc::clone(&self.worker);
        self.run_custom_job(move || {
            if let Some(ref mut w) = *worker.lock().unwrap() {
                w.ensure_network_replicated();
            }
        });
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn worker(&self) -> Arc<Mutex<Option<Box<Worker>>>> {
        Arc::clone(&self.worker)
    }
}

/// Destructor wakes up the thread in idle_loop() and waits
/// for its termination. Thread should be already waiting.
impl Drop for Thread {
    fn drop(&mut self) {
        // C++ asserts the same before joining (thread.cpp).
        assert!(
            !self.state.lock().unwrap().searching,
            "Thread destroyed while still searching"
        );

        {
            let mut state = self.state.lock().unwrap();
            state.exit = true;
            state.searching = true;
        }
        self.cv.notify_all();

        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
    }
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

/// Thread pool managing all search threads
///
/// ThreadPool struct handles all the threads-related stuff like init, starting,
/// parking and, most importantly, launching a thread. All the access to threads
/// is done through this class.
pub struct ThreadPool {
    threads: Vec<Arc<Thread>>,
    bound_thread_to_numa_node: Vec<NumaIndex>,
    // Lock-free stop/increase_depth/ponder state shared with every Worker. It must not sit
    // behind the pool RwLock: polling stop would then need a read guard and could
    // deadlock against the engine write guard in Engine::go.
    search_control: Arc<SearchControl>,
    numa_config: Arc<NumaConfig>,
}

/// The root position as handed to the search threads, each of which clones it. `Position` is not
/// `Sync` (its threat cache is filled through `&self`), so this wrapper offers nothing but a clone.
struct SharedRoot(Position);

impl SharedRoot {
    fn clone_position(&self) -> Position {
        self.0.clone()
    }
}

// SAFETY: the only access through `&SharedRoot` is `Position::clone`, which reads the position
// (the threat cache's `Cell` included) and writes nothing, so concurrent clones cannot race.
unsafe impl Sync for SharedRoot {}

impl ThreadPool {
    pub fn new() -> Self {
        Self {
            threads: Vec::new(),
            bound_thread_to_numa_node: Vec::new(),
            search_control: Arc::new(SearchControl::new()),
            numa_config: Arc::new(NumaConfig::new()),
        }
    }

    /// The lock-free search flags shared with every Worker.
    pub fn search_control(&self) -> &Arc<SearchControl> {
        &self.search_control
    }

    pub fn stop(&self) -> bool {
        self.search_control.stop()
    }

    pub fn set_stop(&self, value: bool) {
        self.search_control.set_stop(value);
    }

    pub fn increase_depth(&self) -> bool {
        self.search_control.increase_depth()
    }

    pub fn set_increase_depth(&self, value: bool) {
        self.search_control.set_increase_depth(value);
    }

    /// Deliberate shutdown: waits for every thread to park, then drops them.
    ///
    /// `Thread::drop` wakes its native thread with `exit` set and JOINS it, and
    /// dropping the `Thread` releases its `Worker`. That also breaks the strong
    /// ownership cycle ThreadPool -> Thread -> Worker -> SharedState -> ThreadPool
    /// (each Worker holds an `Arc` to the pool), which would otherwise leak the
    /// pool, the threads and all their history tables for every dropped `Engine`.
    /// The caller must have stopped the search first (`SearchControl::set_stop`).
    pub fn shutdown(&mut self) {
        for thread in &self.threads {
            thread.wait_for_search_finished_without_unwind();
        }
        self.threads.clear();
        self.bound_thread_to_numa_node.clear();
    }

    /// Creates/destroys threads to match the requested number.
    ///
    /// Created and launched threads immediately go to sleep in `idle_loop`. On a
    /// resize the threads are recreated rather than reused, so that they can be
    /// bound to a NUMA node if necessary.
    pub fn set(
        &mut self,
        numa_config: Arc<NumaConfig>,
        shared_state: SharedState,
        update_context: Arc<RwLock<UpdateContext>>,
    ) {
        let num_threads = shared_state
            .options
            .read()
            .unwrap()
            .get("Threads")
            .and_then(|o| o.get_int())
            .unwrap_or(1) as usize;

        // Adopt the engine-wide SearchControl so the pool's `stop`/`increaseDepth`
        // and the Workers' lock-free flags are literally the same atomics.
        self.search_control = Arc::clone(&shared_state.search_control);

        // Clear existing threads
        if !self.threads.is_empty() {
            // Every thread, not just the main one: dropping an `Arc<Thread>` whose
            // worker is still searching asserts in `Thread::drop`.
            for thread in &self.threads {
                thread.wait_for_search_finished();
            }
            self.threads.clear();
            self.bound_thread_to_numa_node.clear();
        }

        if num_threads == 0 {
            return;
        }

        // Determine NUMA binding policy
        // Binding threads may be problematic when there's multiple NUMA nodes and
        // multiple Stockfish instances running. In particular, if each instance
        // runs a single thread then they would all be mapped to the first NUMA node.
        // This is undesirable, and so the default behaviour (i.e. when the user does not
        // change the NumaConfig UCI setting) is to not bind the threads to processors
        // unless we know for sure that we span NUMA nodes and replication is required.
        let numa_policy = shared_state
            .options
            .read()
            .unwrap()
            .get_string("NumaPolicy", "auto");

        let do_bind_threads = match numa_policy.as_str() {
            "none" => false,
            "auto" => numa_config.suggests_binding_threads(num_threads),
            _ => true, // "system" or user-specified
        };

        self.numa_config = numa_config.clone();

        // Left empty when threads are not bound; callers treat an empty vector as
        // no binding, matching C++.
        self.bound_thread_to_numa_node = if do_bind_threads {
            numa_config.distribute_threads_among_numa_nodes(num_threads)
        } else {
            Vec::new()
        };

        let mut counts = std::collections::BTreeMap::new();
        for thread in 0..num_threads {
            let node = self
                .bound_thread_to_numa_node
                .get(thread)
                .copied()
                .unwrap_or(0);
            *counts.entry(node).or_insert(0usize) += 1;
        }
        let mut histories = std::collections::BTreeMap::new();
        for (&node, &count) in &counts {
            let mut create = || {
                let history = if node == 0 && count == 1 {
                    Arc::clone(&shared_state.shared_histories)
                } else {
                    Arc::new(SharedHistories::new(count.next_power_of_two()))
                };
                histories.insert(node, history);
            };
            if do_bind_threads {
                numa_config.execute_on_numa_node(node, create);
            } else {
                create();
            }
        }
        let mut local_indices = std::collections::BTreeMap::new();

        // Create threads
        for thread_id in 0..num_threads {
            // Unbound threads all belong to node 0, as C++ assumes when the
            // binding vector is empty.
            let numa_id = self
                .bound_thread_to_numa_node
                .get(thread_id)
                .copied()
                .unwrap_or(0);

            let local_index = local_indices.entry(numa_id).or_insert(0usize);
            let numa_thread_idx = *local_index;
            *local_index += 1;
            let numa_total = counts[&numa_id];
            let mut shared_state_clone = shared_state.clone();
            shared_state_clone.shared_histories = Arc::clone(&histories[&numa_id]);
            let update_context_clone = update_context.clone();
            let numa_config_clone = numa_config.clone();

            let mut thread_opt = None;

            // Define creation logic
            let create_thread = move || {
                // The main thread has a SearchManager, the others have None
                let manager = (thread_id == 0).then(|| {
                    Box::new(SearchManager::with_ponder(
                        update_context_clone,
                        Arc::clone(&shared_state_clone.search_control.ponder),
                    ))
                });

                let binder = if do_bind_threads {
                    OptionalThreadToNumaNodeBinder::with_config(numa_config_clone, numa_id)
                } else {
                    // When not binding threads we want to force all access to happen
                    // from the same NUMA node, because in case of NUMA replicated memory
                    // accesses we don't want to trash cache in case the threads get scheduled
                    // on the same NUMA node.
                    OptionalThreadToNumaNodeBinder::new(numa_id)
                };

                Thread::new_with_numa(
                    shared_state_clone,
                    manager,
                    thread_id,
                    binder,
                    numa_thread_idx,
                    numa_total,
                )
            };

            // Ensure the worker thread inherits the intended NUMA affinity at creation.
            if do_bind_threads {
                // Execute creation on specific NUMA node to ensure Thread object allocation locality
                let thread_slot = &mut thread_opt;
                self.numa_config.execute_on_numa_node(numa_id, move || {
                    *thread_slot = Some(create_thread());
                });
            } else {
                thread_opt = Some(create_thread());
            }

            self.threads.push(thread_opt.unwrap());
        }

        self.clear();
        if let Some(main_thread) = self.main_thread() {
            main_thread.wait_for_search_finished();
        }
    }

    /// Clear thread pool state
    ///
    /// Sets threadPool data to initial values
    pub fn clear(&mut self) {
        if self.threads.is_empty() {
            return;
        }

        // Clear all workers
        for thread in &self.threads {
            thread.clear_worker();
        }

        // Wait for all to finish
        for thread in &self.threads {
            thread.wait_for_search_finished();
        }

        // Initialize main manager state from thread.cpp
        if let Some(main_thread) = self.main_thread() {
            if let Some(ref worker) = *main_thread.worker.lock().unwrap() {
                // Access the manager through the worker
                if let Some(manager) = worker.manager.as_deref() {
                    // These two affect the time taken on the first move of a game
                    manager
                        .best_previous_average_score
                        .store(VALUE_INFINITE, Ordering::Relaxed);
                    *manager.previous_time_reduction.write().unwrap() = 0.85;

                    manager.calls_cnt.store(0, Ordering::Relaxed);
                    manager
                        .best_previous_score
                        .store(VALUE_INFINITE, Ordering::Relaxed);
                    *manager.original_time_adjust.write().unwrap() = -1.0;
                    manager.tm.write().unwrap().clear();
                }
            }
        }
    }

    /// Get the main thread
    pub fn main_thread(&self) -> Option<&Arc<Thread>> {
        self.threads.first()
    }

    /// Call pv method on main search manager
    pub fn main_manager_pv(
        &self,
        worker: &crate::search::Worker,
        tt: &crate::tt::TranspositionTable,
        depth: crate::types::Depth,
        extended_scores: &[Option<crate::types::Value>],
    ) {
        // If this is the main thread worker, directly use its manager
        // Avoids deadlock by not trying to lock a worker that's already in use
        if worker.is_main_thread() {
            if let Some(manager) = worker.manager.as_deref() {
                manager.pv(worker, self, tt, depth, extended_scores);
            }
        } else {
            // For non-main threads, find the main thread's manager
            if let Some(thread) = self.threads.first() {
                // Use try_lock to avoid deadlock when worker is already locked by calling thread
                match thread.worker.try_lock() {
                    Ok(worker_guard) => {
                        if let Some(ref main_worker) = *worker_guard {
                            if let Some(manager) = main_worker.manager.as_deref() {
                                manager.pv(worker, self, tt, depth, extended_scores);
                            }
                        }
                    }
                    Err(_) => {
                        // Worker is locked (probably by the calling thread), skip PV output to avoid deadlock
                        // This prevents the hang but may result in less frequent UCI output updates
                    }
                }
            }
        }
    }

    /// Start thinking/searching (from thread.cpp)
    ///
    /// Takes `&self` (not `&mut self`) on purpose: the engine may only hold a
    /// *read* guard on the pool's RwLock here, because this function blocks on
    /// the worker threads' condition variables and those workers may in turn need
    /// to read the pool.
    ///
    /// Wakes up main thread waiting in idle_loop(). Returns once the main thread has been
    /// handed the search job; the search itself runs asynchronously. The main thread will
    /// wake up other threads and start the search.
    ///
    /// The root position is copied with its state history (C++ `set` + `copy_state_infos`).
    /// Every worker receives an owned copy.
    pub fn start_thinking(
        &self,
        options: &Arc<RwLock<OptionsMap>>,
        pos: &Position,
        limits: LimitsType,
    ) {
        // A main-thread panic may park it before its helpers have stopped.
        // Every old worker must finish before the new search clears stop.
        for thread in &self.threads {
            thread.wait_for_search_finished();
        }

        // Set search flags and ponder mode from thread.cpp
        self.search_control.set_stop(false);
        self.search_control.set_increase_depth(true);

        // Add memory fence to ensure all writes are visible
        core::sync::atomic::fence(Ordering::SeqCst);

        // Set ponder and stopOnPonderhit on main manager
        if let Some(main_thread) = self.main_thread() {
            if let Some(ref worker) = *main_thread.worker.lock().unwrap() {
                if let Some(manager) = worker.manager.as_deref() {
                    manager.stop_on_ponderhit.store(false, Ordering::Relaxed);
                    manager.ponder.store(limits.ponder_mode, Ordering::Relaxed);
                }
            }
        }

        // Generate root moves from thread.cpp
        let mut root_moves: RootMoves = Vec::new();

        // `to_move` already returns `Move::none()` for anything that is not a legal
        // move, so no extra membership test against the legal move list is needed.
        for uci_move in &limits.searchmoves {
            let mv = UciEngine::to_move(pos, uci_move);

            if mv != Move::none() {
                root_moves.push(RootMove::new(mv));
            }
        }

        if root_moves.is_empty() {
            for &m in &generate(pos, GenType::All) {
                root_moves.push(RootMove::new(m));
            }
        }

        // Give every worker its own copy of the root position and its history, so a later
        // position command cannot affect a running search.
        let root_position = Arc::new(SharedRoot(pos.clone()));
        let tb_config = match options.read() {
            Ok(opt) => {
                let mut probe_pos = root_position.clone_position();
                let cfg =
                    crate::syzygy::rank_root_moves(&opt, &mut probe_pos, &mut root_moves, false);
                TablebaseConfig {
                    cardinality: cfg.cardinality.max(0) as u32,
                    root_in_tb: cfg.root_in_tb,
                    probe_depth: cfg.probe_depth,
                    use_rule50: cfg.use_rule50,
                    tables: cfg.tables,
                }
            }
            Err(_) => TablebaseConfig::default(),
        };

        // Distribute work to all threads. The heap-owning inputs are shared through `Arc`s and
        // cloned inside each worker's own job: cloning them here (all on the main thread) made
        // every helper free memory from the main thread's malloc arena, which serialises on the
        // arena lock and showed up as a burst of futex waits at the start of every `go`.
        let limits_shared = Arc::new(limits);
        let root_moves_shared = Arc::new(root_moves);
        for thread in &self.threads {
            let limits_clone = Arc::clone(&limits_shared);
            let root_moves_clone = Arc::clone(&root_moves_shared);
            let tb_config_clone = tb_config.clone();
            let root_position_clone = Arc::clone(&root_position);

            let worker = Arc::clone(&thread.worker);
            thread.run_custom_job(move || {
                if let Some(ref mut w) = *worker.lock().unwrap() {
                    // Initialize worker state (thread.cpp)
                    w.limits = (*limits_clone).clone();
                    w.counters.nodes.store(0, Ordering::Relaxed);
                    w.counters.tb_hits.store(0, Ordering::Relaxed);
                    w.nmp_min_ply.store(0, Ordering::Relaxed);
                    w.counters.best_move_changes.store(0, Ordering::Relaxed);
                    w.root_depth = 0;
                    w.root_moves = (*root_moves_clone).clone();
                    w.root_pos = root_position_clone.clone_position();
                    w.tb_config = tb_config_clone;
                }
            });
        }

        // Wait for all threads to complete setup (thread.cpp)
        for thread in &self.threads {
            thread.wait_for_search_finished();
        }

        // Start main thread searching (thread.cpp)
        if let Some(main_thread) = self.main_thread() {
            main_thread.start_searching();
        }
    }

    /// Start searching on non-main threads
    ///
    /// Start non-main threads.
    /// Will be invoked by main thread after it has started searching.
    pub fn start_searching(&self) {
        if self.threads.len() > 1 {
            for thread in &self.threads[1..] {
                thread.start_searching();
            }
        }
    }

    /// Wait for search to finish on all threads
    pub fn wait_for_search_finished(&self) {
        if self.threads.len() > 1 {
            for thread in &self.threads[1..] {
                thread.wait_for_search_finished();
            }
        }
    }

    /// Owning snapshot of the non-main threads (C++ iterates `threads` directly).
    ///
    /// Used by the main worker to implement `threads.wait_for_search_finished()`
    /// (search.cpp) *without* keeping the pool's read guard alive while it
    /// blocks on the helpers' condition variables.
    pub fn helper_threads(&self) -> Vec<Arc<Thread>> {
        if self.threads.len() > 1 {
            self.threads[1..].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Get total nodes searched (thread.h). Lock-free: the counters live in
    /// `Arc`s shared with the threads, so this also works while helpers are searching.
    pub fn nodes_searched(&self) -> u64 {
        self.threads
            .iter()
            .map(|t| t.counters.nodes.load(Ordering::Relaxed))
            .sum()
    }

    /// Same as `nodes_searched` (kept for the existing call sites).
    pub fn nodes_searched_from_worker(&self, _current_worker: &Worker) -> u64 {
        self.nodes_searched()
    }

    /// Get total tablebase hits (thread.h)
    pub fn tb_hits(&self) -> u64 {
        self.threads
            .iter()
            .map(|t| t.counters.tb_hits.load(Ordering::Relaxed))
            .sum()
    }

    /// Same as `tb_hits` (kept for the existing call sites).
    pub fn tb_hits_from_worker(&self, _current_worker: &Worker) -> u64 {
        self.tb_hits()
    }

    /// Sum of all threads' best move changes, resetting each (search.cpp).
    pub fn take_best_move_changes(&self) -> u64 {
        self.threads
            .iter()
            .map(|t| t.counters.best_move_changes.swap(0, Ordering::Relaxed) as u64)
            .sum()
    }

    /// Get total best move changes from all threads and reset counters
    pub fn best_move_changes(&self) -> u64 {
        self.take_best_move_changes()
    }

    /// Get number of threads (alias for num_threads for compatibility)
    pub fn size(&self) -> usize {
        self.threads.len()
    }

    /// Select using stable result copies from all parked workers.
    pub fn get_best_thread(&self) -> &Arc<Thread> {
        self.best_thread_from_results(None)
    }

    /// Include the main worker, which is temporarily owned by its search job
    /// rather than present in the Thread's mutex while it performs final voting.
    pub fn get_best_thread_with_main(&self, main: &Worker) -> &Arc<Thread> {
        self.best_thread_from_results(Some(main))
    }

    fn best_thread_from_results(&self, main: Option<&Worker>) -> &Arc<Thread> {
        assert!(!self.threads.is_empty(), "No threads available");
        let mut results = Vec::with_capacity(self.threads.len());
        for (index, thread) in self.threads.iter().enumerate() {
            if index == 0
                && let Some(main) = main
            {
                if let Some(root) = main.root_moves.first() {
                    results.push((index, root.clone()));
                }
                continue;
            }
            if let Ok(worker) = thread.worker.try_lock()
                && let Some(worker) = worker.as_ref()
                && let Some(root) = worker.root_moves.first()
            {
                results.push((index, root.clone()));
            }
        }
        &self.threads[select_best_result(&results)]
    }

    /// Run a job on a specific thread
    pub fn run_on_thread<F>(&self, thread_id: usize, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if thread_id < self.threads.len() {
            self.threads[thread_id].run_custom_job(f);
        }
    }

    /// Wait for a specific thread
    pub fn wait_on_thread(&self, thread_id: usize) {
        if thread_id < self.threads.len() {
            self.threads[thread_id].wait_for_search_finished();
        }
    }

    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }

    /// Get the raw thread -> NUMA node binding vector (thread.cpp)
    pub fn get_bound_thread_to_numa_node(&self) -> Vec<NumaIndex> {
        self.bound_thread_to_numa_node.clone()
    }

    /// Number of distinct NUMA nodes threads are bound to, at least 1 (thread.cpp)
    pub fn numa_nodes(&self) -> usize {
        use std::collections::HashSet;
        let seen: HashSet<NumaIndex> = self.bound_thread_to_numa_node.iter().copied().collect();
        seen.len().max(1)
    }

    /// Get thread count by NUMA node
    pub fn get_bound_thread_count_by_numa_node(&self) -> Vec<usize> {
        if self.bound_thread_to_numa_node.is_empty() {
            return Vec::new();
        }

        let max_node = *self.bound_thread_to_numa_node.iter().max().unwrap();
        let mut counts = vec![0; max_node + 1];

        for &node in &self.bound_thread_to_numa_node {
            counts[node] += 1;
        }

        counts
    }

    /// Ensure network is replicated across threads.
    ///
    /// C++ `Thread::ensure_network_replicated` runs on the caller before returning; here the
    /// jobs are queued on every thread first (so replication stays parallel across NUMA nodes)
    /// and then awaited, so the replicas exist when this returns. The callers hold only a read
    /// guard on the pool and never run on a worker thread, and the jobs take just the worker
    /// mutex, so the wait cannot deadlock.
    pub fn ensure_network_replicated(&self) {
        for thread in &self.threads {
            thread.ensure_network_replicated();
        }
        for thread in &self.threads {
            thread.wait_for_search_finished();
        }
    }
}

/// Upstream thread.cpp voting over owned result snapshots, in thread order.
fn select_best_result(results: &[(usize, RootMove)]) -> usize {
    let Some((first_index, first_move)) = results.first() else {
        return 0;
    };
    let min_score = results.iter().map(|(_, root)| root.score).min().unwrap();
    let mut votes: HashMap<Move, i64> = HashMap::with_capacity(results.len() * 2);
    for (_, root) in results {
        *votes.entry(root.pv[0]).or_default() += i64::from(root.score - min_score + 14);
    }
    let mut best = (*first_index, first_move);
    for (index, root) in results {
        let best_decisive =
            best.1.score != -VALUE_INFINITE && is_decisive(best.1.score) && !best.1.is_inexact();
        let decisive =
            root.score != -VALUE_INFINITE && is_decisive(root.score) && !root.is_inexact();
        let replace = if best_decisive {
            decisive && root.score.abs() > best.1.score.abs()
        } else {
            decisive
                || (!is_loss(root.score)
                    && (votes[&root.pv[0]] > votes[&best.1.pv[0]]
                        || (votes[&root.pv[0]] == votes[&best.1.pv[0]]
                            && root.pv.len() > best.1.pv.len())))
        };
        if replace {
            best = (*index, root);
        }
    }
    best.0
}

// Iterator support for ThreadPool
impl ThreadPool {
    pub fn iter(&self) -> core::slice::Iter<'_, Arc<Thread>> {
        self.threads.iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, Arc<Thread>> {
        self.threads.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.threads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }
}

// Implement the ThreadPool trait for TranspositionTable
// SAFETY: idle_loop consumes jobs at most once. Submission either accepts the
// callback or reports a prior completed-job panic. Waiting consumes a panic only
// after completion, so TT clearing can catch it and drain all other workers.
unsafe impl crate::tt::ThreadPool for ThreadPool {
    fn num_threads(&self) -> usize {
        self.num_threads()
    }

    fn run_on_thread<F: FnOnce() + Send + 'static>(&self, i: usize, f: F) {
        self.run_on_thread(i, f)
    }

    fn wait_on_thread(&self, i: usize) {
        self.wait_on_thread(i)
    }

    fn numa_nodes(&self) -> usize {
        self.numa_nodes()
    }

    fn get_bound_thread_to_numa_node(&self) -> Vec<usize> {
        self.get_bound_thread_to_numa_node()
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    #[test]
    fn numa_config_from_string_matches_cpp_syntax() {
        let cfg = NumaConfig::from_string("0-3,8:4-7").expect("valid config");
        assert_eq!(cfg.num_numa_nodes(), 2);
        assert_eq!(cfg.num_cpus_in_numa_node(0), 5);
        assert_eq!(cfg.num_cpus_in_numa_node(1), 4);
        assert_eq!(cfg.num_cpus_in_numa_node(2), 0);
        assert_eq!(cfg.to_string(), "0-3,8:4-7");
        // A user-specified config always asks for binding, even for one thread.
        assert!(cfg.suggests_binding_threads(1));
    }

    #[test]
    fn numa_config_from_string_rejects_invalid_input() {
        assert!(NumaConfig::from_string("").is_none());
        assert!(NumaConfig::from_string("abc").is_none());
        assert!(NumaConfig::from_string("3-1").is_none());
        // CPU 2 assigned to two nodes.
        assert!(NumaConfig::from_string("0-2:2-4").is_none());
    }

    #[test]
    fn numa_policy_none_is_a_single_unbound_node() {
        let cfg = NumaConfig::single_node();
        assert_eq!(cfg.num_numa_nodes(), 1);
        assert!(!cfg.suggests_binding_threads(64));
        assert!(
            cfg.distribute_threads_among_numa_nodes(5)
                .iter()
                .all(|&n| n == 0)
        );
    }

    #[test]
    fn numa_distribution_fills_nodes_proportionally() {
        let cfg = NumaConfig::from_string("0-3:4-11").unwrap();
        let dist = cfg.distribute_threads_among_numa_nodes(12);
        let on = |n: usize| dist.iter().filter(|&&d| d == n).count();
        // The 8-CPU node receives twice as many threads as the 4-CPU node.
        assert_eq!((on(0), on(1)), (4, 8));
    }

    #[test]
    fn test_thread_pool_creation() {
        let options = Arc::new(RwLock::new(OptionsMap::new()));
        let threads = Arc::new(RwLock::new(ThreadPool::new()));
        let tt = Arc::new(RwLock::new(TranspositionTable::new()));
        let networks = Arc::new(crate::engine::LazyNumaReplicated::new(
            crate::evaluate::Networks::new(),
        ));
        let search_control = Arc::new(SearchControl::new());
        let shared_histories = Arc::new(SharedHistories::new(1)); // 1 thread for test
        let shared_state = SharedState::new(
            options,
            threads,
            tt,
            networks,
            search_control,
            shared_histories,
        );
        let numa_config = Arc::new(NumaConfig::new());
        let update_context = Arc::new(RwLock::new(UpdateContext {
            on_update_no_moves: Box::new(|_| {}),
            on_update_full: Box::new(|_| {}),
            on_iter: Box::new(|_| {}),
            on_bestmove: Box::new(|_, _| {}),
            on_start: Box::new(|| {}),
        }));

        let mut pool = ThreadPool::new();
        assert_eq!(pool.num_threads(), 0);

        // Create pool with default threads (1 when not set in options)
        pool.set(numa_config, shared_state, update_context);
        assert_eq!(pool.num_threads(), 1);
    }

    #[test]
    fn test_custom_job_execution() {
        let options = Arc::new(RwLock::new(OptionsMap::new()));
        let threads = Arc::new(RwLock::new(ThreadPool::new()));
        let tt = Arc::new(RwLock::new(TranspositionTable::new()));
        let networks = Arc::new(crate::engine::LazyNumaReplicated::new(
            crate::evaluate::Networks::new(),
        ));
        let search_control = Arc::new(SearchControl::new());
        let shared_histories = Arc::new(SharedHistories::new(1)); // 1 thread for test
        let shared_state = SharedState::new(
            options,
            threads,
            tt,
            networks,
            search_control,
            shared_histories,
        );
        let numa_config = Arc::new(NumaConfig::new());
        let update_context = Arc::new(RwLock::new(UpdateContext {
            on_update_no_moves: Box::new(|_| {}),
            on_update_full: Box::new(|_| {}),
            on_iter: Box::new(|_| {}),
            on_bestmove: Box::new(|_, _| {}),
            on_start: Box::new(|| {}),
        }));

        let mut pool = ThreadPool::new();
        pool.set(numa_config, shared_state, update_context);

        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = Arc::clone(&flag);

        pool.run_on_thread(0, move || {
            flag_clone.store(true, Ordering::Relaxed);
        });

        pool.wait_on_thread(0);
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn test_numa_distribution() {
        let numa_config = NumaConfig::new();
        let distribution = numa_config.distribute_threads_among_numa_nodes(8);

        assert_eq!(distribution.len(), 8);
        // Since we have only 1 NUMA node by default, all threads map to node 0
        for &node in &distribution {
            assert_eq!(node, 0);
        }
    }

    #[test]
    fn final_voting_includes_the_main_result_while_worker_is_taken() {
        let mut main = RootMove::new(Move::from_to(
            crate::types::Square::SQ_E2,
            crate::types::Square::SQ_E4,
        ));
        main.score = 100;
        let mut helper = RootMove::new(Move::from_to(
            crate::types::Square::SQ_D2,
            crate::types::Square::SQ_D4,
        ));
        helper.score = -100;
        assert_eq!(
            select_best_result(&[(0, main.clone()), (1, helper.clone())]),
            0
        );
        helper.score = 200;
        assert_eq!(select_best_result(&[(0, main), (1, helper)]), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn node_local_work_preserves_caller_affinity_even_after_unwinding() {
        let before = NumaConfig::linux_allowed_cpus().expect("read affinity");
        let cpu = *before.iter().next().unwrap();
        let config = NumaConfig::from_string(&cpu.to_string()).unwrap();
        config.execute_on_numa_node(0, || {
            assert_eq!(
                NumaConfig::linux_allowed_cpus().unwrap(),
                std::collections::BTreeSet::from([cpu])
            );
        });
        assert_eq!(NumaConfig::linux_allowed_cpus().unwrap(), before);
        let result =
            std::panic::catch_unwind(|| config.execute_on_numa_node(0, || panic!("probe")));
        assert!(result.is_err());
        assert_eq!(NumaConfig::linux_allowed_cpus().unwrap(), before);
    }

    #[test]
    fn l3_domains_are_bundled_like_cpp() {
        use std::collections::BTreeSet;
        let doms = |sizes: &[usize]| -> Vec<BTreeSet<usize>> {
            let mut next = 0;
            sizes
                .iter()
                .map(|&n| {
                    let d = (next..next + n).collect();
                    next += n;
                    d
                })
                .collect()
        };
        let sizes = |v: Vec<BTreeSet<usize>>| v.iter().map(BTreeSet::len).collect::<Vec<_>>();
        assert_eq!(
            sizes(NumaConfig::bundle_l3_domains(doms(&[16, 16, 16, 16]), 32)),
            [32, 32]
        );
        assert_eq!(
            sizes(NumaConfig::bundle_l3_domains(doms(&[20, 20, 20]), 32)),
            [20, 20, 20]
        );
        assert_eq!(
            sizes(NumaConfig::bundle_l3_domains(doms(&[8, 8, 8, 8, 8]), 32)),
            [32, 8]
        );
        assert_eq!(
            sizes(NumaConfig::bundle_l3_domains(doms(&[16, 16]), 0)),
            [16, 16]
        );
    }

    #[test]
    fn test_thread_synchronization() {
        let options = Arc::new(RwLock::new(OptionsMap::new()));
        let threads = Arc::new(RwLock::new(ThreadPool::new()));
        let tt = Arc::new(RwLock::new(TranspositionTable::new()));
        let networks = Arc::new(crate::engine::LazyNumaReplicated::new(
            crate::evaluate::Networks::new(),
        ));
        let search_control = Arc::new(SearchControl::new());
        let shared_histories = Arc::new(SharedHistories::new(1)); // 1 thread for test
        let shared_state = SharedState::new(
            options,
            threads,
            tt,
            networks,
            search_control,
            shared_histories,
        );
        let manager = None;
        let binder = OptionalThreadToNumaNodeBinder::new(0);

        let thread = Thread::new(shared_state, manager, 0, binder);

        // Thread should start idle
        thread.wait_for_search_finished();

        // Run a job
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        thread.run_custom_job(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        thread.wait_for_search_finished();
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
