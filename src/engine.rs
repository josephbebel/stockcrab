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

//! Engine module - Rust port of Stockfish's engine.h/engine.cpp
//!
//! This module provides the main UCI chess engine interface, managing thread pools,
//! position state, transposition tables, and neural network evaluations.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use crate::evaluate::{self, EVAL_FILE_DEFAULT_NAME};
use crate::history::SharedHistories;
use crate::misc::start_logger;
use crate::nnue::network::Network;
use crate::perft;
use crate::position::{Position, PositionSetError};
use crate::search::LimitsType;
use crate::shm::AllocationStatus;
use crate::syzygy;
use crate::thread::{
    SearchControl, SharedState as ThreadSharedState, ThreadPool,
    UpdateContext as ThreadUpdateContext,
};
use crate::tt::TranspositionTable;
use crate::types::{Depth, MAX_MOVES, Move};
use crate::uci::UciEngine;
use crate::ucioption::OptionsMap;

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MAX_HASH_MB: usize = if cfg!(target_pointer_width = "64") {
    33554432
} else {
    2048
};

static MAX_THREADS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    core::cmp::max(
        1024,
        4 * std::thread::available_parallelism().map_or(1, |n| n.get()),
    )
});

// ---------------------------------------------------------------------------
// NUMA SUPPORT
// ---------------------------------------------------------------------------

/// NUMA configuration context
#[derive(Clone, Debug)]
pub struct NumaReplicationContext {
    numa_config: NumaConfig,
}

impl NumaReplicationContext {
    pub fn new(numa_config: NumaConfig) -> Self {
        Self { numa_config }
    }

    pub fn from_system() -> Self {
        Self::new(NumaConfig::from_system(true))
    }

    pub fn set_numa_config(&mut self, config: NumaConfig) {
        self.numa_config = config;
    }

    pub fn get_numa_config(&self) -> &NumaConfig {
        &self.numa_config
    }
}

pub use crate::thread::NumaConfig;

// ---------------------------------------------------------------------------
// NETWORK SUPPORT
// ---------------------------------------------------------------------------

// The `Network` type comes from the nnue module (via `crate::nnue::network`).

/// Lazy NUMA-replicated container for the network.
///
/// Workers retain immutable snapshots. Node zero shares the canonical allocation;
/// other nodes receive local replicas when first used.
pub struct LazyNumaReplicated<T> {
    data: Arc<Mutex<Arc<T>>>,
    replicas: Arc<Mutex<std::collections::BTreeMap<usize, Arc<T>>>>,
}

pub(crate) trait NumaReplicable {
    fn replicate(&self, numa_node: usize) -> Self;
}

impl<T> LazyNumaReplicated<T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(Mutex::new(Arc::new(data))),
            replicas: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        }
    }

    pub fn modify_and_replicate<F>(&self, f: F)
    where
        F: FnOnce(&mut T),
        T: Clone,
    {
        let mut replicas = self.replicas.lock().unwrap();
        let mut data = self.data.lock().unwrap();
        // Drop cached owners before testing uniqueness. Live workers keep their
        // snapshots, so replacements cannot invalidate their weights.
        replicas.clear();
        f(Arc::make_mut(&mut data));
        // Workers retain immutable snapshots until refreshed. A replacement
        // never invalidates their pointers or mutates weights they are reading.
        replicas.clear();
    }

    pub(crate) fn get(&self, token: crate::thread::NumaReplicatedAccessToken) -> Arc<T>
    where
        T: NumaReplicable,
    {
        let mut replicas = self.replicas.lock().unwrap();
        Arc::clone(replicas.entry(token.numa_id()).or_insert_with(|| {
            let data = self.data.lock().unwrap();
            if token.numa_id() == 0 {
                Arc::clone(&data)
            } else {
                Arc::new(data.replicate(token.numa_id()))
            }
        }))
    }
}

impl<T> core::ops::Deref for LazyNumaReplicated<T> {
    type Target = Arc<Mutex<Arc<T>>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// SEARCH TYPES (re-export from thread module)
// ---------------------------------------------------------------------------

// Re-export search info types from thread module
pub use crate::thread::{InfoFull, InfoIteration, InfoShort};

// ---------------------------------------------------------------------------
// ENGINE
// ---------------------------------------------------------------------------

/// Main UCI chess engine
pub struct Engine {
    binary_directory: String,
    numa_context: NumaReplicationContext,

    // Position and game state
    pos: Position,

    // Engine components - with Arc wrappers for sharing
    options: Arc<RwLock<OptionsMap>>,
    tt: Arc<RwLock<TranspositionTable>>,
    // The NNUE network.
    network: LazyNumaReplicated<Network>,

    // Thread pool - shared with worker threads
    thread_pool_data: Arc<RwLock<ThreadPool>>,

    // Search control flags - lock-free access for hot path
    search_control: Arc<SearchControl>,

    // Shared history tables between threads (C++ SharedHistories)
    shared_histories: Arc<SharedHistories>,

    // Callbacks
    update_context: Arc<RwLock<ThreadUpdateContext>>,
    on_verify_network: Option<Box<dyn Fn(&str)>>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Stop first: an `infinite`/`ponder` search never ends by itself and would
        // make the wait below hang.
        self.search_control.set_stop(true);
        // Release the pool lock before waiting: workers may still need to read
        // it. Drain all workers and pending panics before taking the write lock.
        let threads: Vec<_> = self
            .thread_pool_data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect();
        for thread in &threads {
            thread.wait_for_search_finished_without_unwind();
        }
        // Join the native threads and release the Workers. Without this the
        // ThreadPool <-> Worker `Arc` cycle keeps everything alive forever. Do not
        // panic in drop on a poisoned lock (a search thread may have panicked).
        self.thread_pool_data
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .shutdown();
    }
}

impl Engine {
    /// Create a new engine instance
    pub fn new(path: Option<PathBuf>) -> Self {
        // Initialize bitboard tables, movegen tables, and position module
        crate::bitboard_fast::init();
        crate::movegen::init();

        // Equivalent of C++ `CommandLine::get_binary_directory` (engine.cpp).
        let binary_directory = path
            .or_else(|| std::env::current_exe().ok())
            .and_then(|p| p.parent().map(|d| d.to_string_lossy().to_string()))
            .unwrap_or_default();

        let numa_context = NumaReplicationContext::from_system();

        // C++ `networkFile{std::nullopt, ""}` and `network(numaContext)`; the
        // network is loaded further down via `get_default_network()`.
        let network = LazyNumaReplicated::new(Network::new());

        // Create Arc-wrapped components
        let options = Arc::new(RwLock::new(OptionsMap::new()));
        let tt = Arc::new(RwLock::new(TranspositionTable::new()));
        let thread_pool_data = Arc::new(RwLock::new(ThreadPool::new()));
        let search_control = Arc::new(SearchControl::new());

        // Create update context with default callbacks
        let update_context = Arc::new(RwLock::new(ThreadUpdateContext {
            on_update_no_moves: Box::new(|_| {}),
            on_update_full: Box::new(|_| {}),
            on_iter: Box::new(|_| {}),
            on_bestmove: Box::new(|_, _| {}),
            on_start: Box::new(|| {}),
        }));

        // Per 969285fa, 1a67ccc7: Create shared histories
        // Seed for the default one-thread group; ThreadPool::set allocates
        // larger groups per NUMA node when Threads or NumaPolicy changes.
        let shared_histories = Arc::new(SharedHistories::new(1));

        let mut engine = Self {
            binary_directory,
            numa_context,
            pos: Position::default(),
            options,
            tt,
            network,
            thread_pool_data,
            search_control,
            shared_histories,
            update_context,
            on_verify_network: None,
        };

        // Initialize position
        let _ = engine.pos.set_fen(START_FEN, false);

        // Add UCI options
        engine.init_options();

        // Build and install the default network, then (re)initialise threads.
        // Mirrors `network = get_default_network(); threads.clear();
        // threads.ensure_network_replicated(); resize_threads();`
        let default_network = engine.get_default_network();
        engine.network = LazyNumaReplicated::new(default_network);
        engine.thread_pool_data.write().unwrap().clear();
        engine
            .thread_pool_data
            .read()
            .unwrap()
            .ensure_network_replicated();

        // Resize threads to initialize the thread pool properly
        engine.resize_threads();

        engine
    }

    /// Initialize UCI options
    fn init_options(&mut self) {
        let mut options = self.options.write().unwrap();
        use crate::ucioption::UciOption;

        // Debug log file
        options.add(UciOption::new_string(
            "Debug Log File",
            "",
            Some(Arc::new(|opt| {
                if let Some(file) = opt.get_str() {
                    start_logger(file);
                }
                None
            })),
        ));

        // NUMA policy
        // C++ declares this as a *string*, not a combo: besides the "auto" /
        // "system" / "hardware" / "none" keywords it accepts explicit CPU lists
        // such as "0-7:8-15". A combo makes a GUI render a fixed dropdown, so a
        // user could not enter one.
        options.add(UciOption::new_string(
            "NumaPolicy",
            "auto",
            Some(Arc::new(move |_opt| {
                // C++ reports `numa_config_information_as_string()` and
                // `thread_allocation_information_as_string()` from this
                // on_change. Neither is reachable from a callback that cannot
                // capture the Engine, so `UciEngine::setoption` emits them.
                None
            })),
        ));

        // Threads
        options.add(UciOption::new_spin(
            "Threads",
            1,
            1,
            *MAX_THREADS as i32,
            Some(Arc::new(|_opt| {
                // UciEngine::setoption prints thread_allocation_information_as_string()
                // (incl. NUMA binding) since the Engine is not reachable from this callback.
                None
            })),
        ));

        // Hash table size
        options.add(UciOption::new_spin(
            "Hash",
            16,
            1,
            MAX_HASH_MB as i32,
            Some(Arc::new(move |_opt| {
                // Callback handled in set_tt_size
                None
            })),
        ));

        // Clear hash
        options.add(UciOption::new_button(
            "Clear Hash",
            Some(Arc::new(move |_opt| {
                // Callback handled in search_clear
                None
            })),
        ));

        // Ponder
        options.add(UciOption::new_check("Ponder", false, None));

        // MultiPV
        options.add(UciOption::new_spin("MultiPV", 1, 1, MAX_MOVES as i32, None));

        // Skill level
        options.add(UciOption::new_spin("Skill Level", 20, 0, 20, None));

        // Move overhead
        options.add(UciOption::new_spin("Move Overhead", 10, 0, 5000, None));

        // Nodes time
        options.add(UciOption::new_spin("nodestime", 0, 0, 10000, None));

        // Chess960
        options.add(UciOption::new_check("UCI_Chess960", false, None));

        // Limit strength
        options.add(UciOption::new_check("UCI_LimitStrength", false, None));

        // ELO - using proper Search::Skill bounds
        const LOWEST_ELO: i32 = 1320; // Search::Skill::LowestElo
        const HIGHEST_ELO: i32 = 3190; // Search::Skill::HighestElo
        options.add(UciOption::new_spin(
            "UCI_Elo",
            LOWEST_ELO,
            LOWEST_ELO,
            HIGHEST_ELO,
            None,
        ));

        // Show WDL
        options.add(UciOption::new_check("UCI_ShowWDL", false, None));

        // Syzygy tablebase settings. The `SyzygyPath` callback below is where
        // C++ calls `Tablebases::init(o)` (engine.cpp).
        options.add(UciOption::new_string(
            "SyzygyPath",
            "",
            Some(Arc::new(|opt| {
                if let Some(path) = opt.get_str() {
                    // Initialize Syzygy tablebases
                    syzygy::init(path);
                }
                // C++ returns std::nullopt here (engine.cpp): initialising
                // the tablebases is the whole side effect, nothing is reported.
                None
            })),
        ));

        options.add(UciOption::new_spin("SyzygyProbeDepth", 1, 1, 100, None));
        options.add(UciOption::new_check("Syzygy50MoveRule", true, None));
        options.add(UciOption::new_spin("SyzygyProbeLimit", 7, 0, 7, None));

        // The on_change callback cannot capture the Engine, so `UciEngine::setoption`
        // calls `load_network` when this option changes.
        options.add(UciOption::new_string(
            "EvalFile",
            EVAL_FILE_DEFAULT_NAME,
            Some(Arc::new(move |_opt| {
                // Handled in UciEngine::setoption
                None
            })),
        ));
    }

    /// Apply a UCI option update and all Engine-owned side effects.
    /// Returns informational records for the caller to emit.
    pub fn set_option(&mut self, line: &str) -> Vec<String> {
        self.wait_for_search_finished();
        let accepted = self.options.write().unwrap().setoption(line);
        let Some(name) = accepted else {
            return Vec::new();
        };
        match name.to_ascii_lowercase().as_str() {
            "hash" => {
                let mb = self.options.read().unwrap().get_int("Hash", 16) as usize;
                self.set_tt_size(mb);
            }
            "threads" => {
                self.resize_threads();
                return vec![self.thread_allocation_information_as_string()];
            }
            "clear hash" => self.search_clear(),
            "evalfile" => {
                let file = self
                    .options
                    .read()
                    .unwrap()
                    .get_string("EvalFile", EVAL_FILE_DEFAULT_NAME);
                self.load_network(&file);
            }
            "numapolicy" => {
                let value = self
                    .options
                    .read()
                    .unwrap()
                    .get_string("NumaPolicy", "auto");
                if self.set_numa_config_from_option(&value) {
                    return vec![
                        self.numa_config_information_as_string(),
                        self.thread_allocation_information_as_string(),
                    ];
                }
                return vec![format!(
                    "NumaPolicy: invalid value '{value}', keeping previous config."
                )];
            }
            _ => {}
        }
        Vec::new()
    }

    /// Run perft (performance test)
    ///
    /// Returns `Result<u64, PositionSetError>` (C++ `std::variant<u64,
    /// PositionSetError>`) so an invalid FEN surfaces as an error instead of a count.
    pub fn perft(
        &self,
        fen: &str,
        depth: Depth,
        is_chess960: bool,
    ) -> Result<u64, PositionSetError> {
        self.verify_network();

        // Create a position for perft calculation
        let mut pos = Position::default();
        pos.set_fen(fen, is_chess960)?;

        // Run perft and return node count
        Ok(perft::perft(&mut pos, depth))
    }

    /// Start searching (non-blocking)
    pub fn go(&mut self, limits: LimitsType) {
        assert_eq!(limits.perft, 0);
        self.verify_network();

        // Use the shared thread pool data that the Workers also use
        {
            // READ guard, not write. `start_thinking` and `set_stop` both take `&self`
            // so the engine never needs
            // exclusive access to the pool to launch a search.
            //
            // This is the deadlock that hung multi-threaded search: SharedState::threads
            // is Arc::clone'd from this same RwLock, so the search threads take READ
            // guards on it while running. std::sync::RwLock is writer-preferring, so the
            // engine requesting a WRITE guard here would queue ahead of them and block
            // every reader -- including the workers whose progress the engine was waiting
            // on. Readers do not exclude each other, so the cycle cannot form.
            let thread_pool = self.thread_pool_data.read().unwrap();
            // The pool waits for the previous search before resetting its stop
            // flag. Resetting here could erase the stop that it is waiting for.
            thread_pool.start_thinking(&self.options, &self.pos, limits);
        } // Release read guard here
    }

    /// Stop searching (non-blocking)
    pub fn stop(&mut self) {
        // Lock-free. ThreadPool holds an Arc to this same SearchControl (assigned in
        // ThreadPool::set), so there is a single source of truth and stopping never
        // touches the pool's RwLock.
        self.search_control.set_stop(true);
    }

    /// Wait for search to finish (blocking)
    pub fn wait_for_search_finished(&mut self) {
        if let Some(main_thread) = self.thread_pool_data.read().unwrap().main_thread() {
            main_thread.wait_for_search_finished();
        }
        // If no threads exist, there's nothing to wait for
    }

    /// Sets the position from a FEN and UCI moves. Returns Some(error) for a bad FEN or illegal move.
    pub fn set_position(&mut self, fen: &str, moves: &[String]) -> Option<PositionSetError> {
        // Set initial position
        let is_chess960 = self
            .options
            .read()
            .unwrap()
            .get("UCI_Chess960")
            .and_then(|o| o.get_int())
            .map(|v| v != 0)
            .unwrap_or(false);

        if let Err(e) = self.pos.set_fen(fen, is_chess960) {
            return Some(e);
        }

        // Apply moves; an unparsable move is reported as an illegal-move error.
        for move_str in moves {
            // Parse UCI move using proper UciEngine::to_move
            let mv = UciEngine::to_move(&self.pos, move_str);
            if mv == Move::none() || !self.pos.do_move_owned(mv) {
                return Some(PositionSetError::new(format!("Illegal move: {}", move_str)));
            }
        }

        None
    }

    /// Set NUMA configuration from option string.
    ///
    /// Returns `false` for an unrecognised custom config so the caller can keep the
    /// previous config and report an error.
    pub fn set_numa_config_from_option(&mut self, option: &str) -> bool {
        // `from_system` applies C++'s default `BundledL3Policy{32}` (L3 domains grouped up to 32 CPUs).
        let config = match option {
            "auto" | "system" => NumaConfig::from_system(true),
            // Ignore the affinity the OS gave this process.
            "hardware" => NumaConfig::from_system(false),
            "none" => NumaConfig::single_node(),
            // A failed parse returns false and leaves the previous config untouched.
            _ => match NumaConfig::from_string(option) {
                Some(parsed) => parsed,
                None => return false,
            },
        };

        self.numa_context.set_numa_config(config);
        self.resize_threads();
        self.thread_pool_data
            .read()
            .unwrap()
            .ensure_network_replicated();
        true
    }

    /// Resize thread pool
    pub fn resize_threads(&mut self) {
        // Wait for any ongoing search to finish
        self.wait_for_search_finished();

        // Create shared state with proper Arc wrappers
        let shared_state = ThreadSharedState::new(
            Arc::clone(&self.options),
            Arc::clone(&self.thread_pool_data),
            Arc::clone(&self.tt),
            {
                // Share the network: a LazyNumaReplicated wrapper over the same
                // Arc<Mutex<Network>> data.
                Arc::new(LazyNumaReplicated {
                    data: Arc::clone(&self.network.data),
                    replicas: Arc::clone(&self.network.replicas),
                })
            },
            Arc::clone(&self.search_control),
            Arc::clone(&self.shared_histories),
        );

        // Use the NUMA configuration stored in the context (set by `NumaPolicy`)
        let numa_config = Arc::new(self.numa_context.get_numa_config().clone());

        // Initialize thread pool
        self.thread_pool_data.write().unwrap().set(
            numa_config,
            shared_state,
            Arc::clone(&self.update_context),
        );

        // Reallocate hash with new thread pool size
        let hash_mb = self.options.read().unwrap().get_int("Hash", 16) as usize;
        self.set_tt_size(hash_mb);
        self.thread_pool_data
            .read()
            .unwrap()
            .ensure_network_replicated();
    }

    /// Set transposition table size
    pub fn set_tt_size(&mut self, mb: usize) {
        self.wait_for_search_finished();
        self.tt
            .write()
            .unwrap()
            .resize(mb, &*self.thread_pool_data.read().unwrap());
    }

    /// Current transposition table size in clusters.
    /// Introspection accessor: `tests/regression_parity.rs` uses it to check that
    /// the `Hash` option's on-change side effect actually happens.
    pub fn tt_cluster_count(&self) -> usize {
        self.tt.read().unwrap().cluster_count()
    }

    /// Set ponderhit flag
    pub fn set_ponderhit(&mut self, ponder: bool) {
        // Update the live manager control state without waiting behind search.
        if let Some(main_thread) = self.thread_pool_data.read().unwrap().main_thread() {
            main_thread.set_ponder(ponder);
        }
    }

    /// Clear search tables
    pub fn search_clear(&mut self) {
        self.wait_for_search_finished();

        // Clear the transposition table
        self.tt
            .write()
            .unwrap()
            .clear(&*self.thread_pool_data.read().unwrap());

        // Clear thread-specific tables
        self.thread_pool_data.write().unwrap().clear();

        // Reinitialise Syzygy tablebases if a path is set. Re-initialising frees the mapped
        // files, so several Engine instances in one process are not supported (same limitation
        // as upstream engine.cpp).
        let syzygy_path = self.options.read().unwrap().get_string("SyzygyPath", "");
        if !syzygy_path.is_empty() {
            syzygy::init(&syzygy_path);
        }
    }

    /// Set callback for position updates with no legal moves
    pub fn set_on_update_no_moves(&mut self, f: impl Fn(InfoShort) + Send + Sync + 'static) {
        let mut update_context = self.update_context.write().unwrap();
        update_context.on_update_no_moves = Box::new(f);
    }

    /// Set callback for full position updates
    pub fn set_on_update_full(&mut self, f: impl Fn(InfoFull) + Send + Sync + 'static) {
        let mut update_context = self.update_context.write().unwrap();
        update_context.on_update_full = Box::new(f);
    }

    /// Set callback for iteration updates
    pub fn set_on_iter(&mut self, f: impl Fn(InfoIteration) + Send + Sync + 'static) {
        let mut update_context = self.update_context.write().unwrap();
        update_context.on_iter = Box::new(f);
    }

    /// Set callback for best move
    pub fn set_on_bestmove(&mut self, f: impl Fn(&str, &str) + Send + Sync + 'static) {
        let mut update_context = self.update_context.write().unwrap();
        update_context.on_bestmove = Box::new(move |s1, s2| f(s1, s2));
    }

    /// Set callback invoked when a search starts (C++ `Engine::set_on_start`)
    pub fn set_on_start(&mut self, f: impl Fn() + Send + Sync + 'static) {
        let mut update_context = self.update_context.write().unwrap();
        update_context.on_start = Box::new(f);
    }

    /// Set callback for network verification
    pub fn set_on_verify_network(&mut self, f: impl Fn(&str) + 'static) {
        self.on_verify_network = Some(Box::new(f));
    }

    /// Verify the neural network
    ///
    /// The network is verified against the current `EvalFile` and the path
    /// currently held by the `EvalFile` UCI option.
    pub fn verify_network(&self) {
        let file = self
            .options
            .read()
            .unwrap()
            .get_string("EvalFile", EVAL_FILE_DEFAULT_NAME);
        // C++ hands `Network::verify` the same callback the UCI layer installed,
        // so every line it produces is wrapped in `info string`. It emits no
        // "Network verified" message.
        let network = self.network.lock().unwrap();
        match self.on_verify_network {
            Some(ref verify_fn) => network.verify(file, &|s: &str| verify_fn(s)),
            None => network.verify(file, &|_: &str| {}),
        }
        drop(network);

        let replicas: Vec<_> = self
            .network
            .replicas
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect();
        if let Some(ref verify_fn) = self.on_verify_network {
            for (index, network) in replicas.iter().enumerate() {
                let mut message = format!("Network replica {}: ", index + 1);
                message.push_str(match network.shared_weights_status() {
                    AllocationStatus::NoAllocation => "No allocation.",
                    AllocationStatus::LocalMemory => "Local memory.",
                    AllocationStatus::SharedMemory => "Shared memory.",
                });
                if let Some(err) = network.shared_weights_message() {
                    message.push(' ');
                    message.push_str(&err);
                }
                verify_fn(&message);
            }
        }
    }

    /// Build and load the default network.
    ///
    /// Mirrors C++ `get_default_network()`, which constructs a fresh `NN::Network`
    /// and loads it from the binary directory (using `EVAL_FILE_DEFAULT_NAME`).
    ///
    /// If no candidate path loads, a warning is printed to stderr and an unloaded
    /// Network is returned; `verify_network` will then fail.
    pub fn get_default_network(&mut self) -> Network {
        let mut network = Network::new();

        // Look next to the binary first (as C++ does), then in the working directory.
        let search_paths = vec![
            self.binary_directory.clone(),
            ".".to_string(),
            String::new(),
        ];

        // Probing the working directory too is a convenience C++ does not have (it looks
        // only next to the binary). Failures for the paths we are
        // merely *trying* are not errors, so they are collected and reported only
        // if every candidate fails.
        let mut loaded = false;
        let mut attempts = Vec::new();
        for path in &search_paths {
            match network.load(path, String::new()) {
                Ok(()) => {
                    loaded = true;
                    break;
                }
                Err(e) => attempts.push(format!(
                    "  {}: {}",
                    if path.is_empty() { "<cwd>" } else { path },
                    e
                )),
            }
        }

        if !loaded {
            eprintln!(
                "Warning: could not load the NNUE network from any of the standard paths:\n{}",
                attempts.join("\n")
            );
        }

        network
    }

    /// Load the network from `file`.
    pub fn load_network(&mut self, file: &str) {
        self.wait_for_search_finished();

        let path = file.to_string();
        let binary_directory = self.binary_directory.clone();
        self.network.modify_and_replicate(|network| {
            if let Err(e) = network.load(&binary_directory, path.clone()) {
                eprintln!(
                    "Warning: Failed to load NNUE network '{}': {}. Keeping the active network.",
                    path, e
                );
            }
        });

        self.thread_pool_data.write().unwrap().clear();
        self.thread_pool_data
            .read()
            .unwrap()
            .ensure_network_replicated();
    }

    /// Save the network. C++ takes `std::optional<std::filesystem::path>`.
    pub fn save_network(&self, file: &Option<PathBuf>) {
        let filename = file.as_ref().map(|p| p.to_string_lossy().into_owned());
        // `save` reports every failure on stdout and returns Ok(false), as C++ does.
        let _ = self.network.lock().unwrap().save(filename);
    }

    /// Trace evaluation
    pub fn trace_eval(&self) {
        self.verify_network();

        // Create a copy of the position for tracing
        let mut trace_pos = Position::default();
        let is_chess960 = self
            .options
            .read()
            .unwrap()
            .get("UCI_Chess960")
            .and_then(|o| o.get_int())
            .map(|v| v != 0)
            .unwrap_or(false);

        let _ = trace_pos.set_fen(&self.fen(), is_chess960);

        // Get the network for evaluation
        let network = self.network.lock().unwrap();

        // C++: `sync_cout << "\n" << Eval::trace(p, *network) << sync_endl`.
        // The leading newline belongs to the *caller*; `Eval::trace` itself also
        // starts with one, so C++ emits two blank lines before the table.
        let trace_output = evaluate::trace(&mut trace_pos, &network);

        // Output with synchronized cout
        crate::uci::UciEngine::write_stdout_bytes(format!("\n{trace_output}\n").as_bytes());
    }

    /// Shared handle to the options map.
    ///
    /// Needed by UCI callbacks, which outlive the borrow a `get_options()` guard
    /// could give them and must therefore read options at call time rather than
    /// capturing a snapshot. C++ gets this for free by capturing `this`.
    pub fn options_handle(&self) -> Arc<RwLock<OptionsMap>> {
        Arc::clone(&self.options)
    }

    /// Read guard over the options map.
    pub fn get_options(&self) -> std::sync::RwLockReadGuard<'_, OptionsMap> {
        self.options.read().unwrap()
    }

    /// Get options (mutable)
    pub fn get_options_mut(&mut self) -> std::sync::RwLockWriteGuard<'_, OptionsMap> {
        self.options.write().unwrap()
    }

    /// Get hash table fullness
    pub fn get_hashfull(&self, max_age: i32) -> i32 {
        self.tt.read().unwrap().hashfull(max_age as usize) as i32
    }

    /// Get current position FEN
    pub fn fen(&self) -> String {
        self.pos.fen()
    }

    /// Flip the board
    ///
    /// Returns `Option<PositionSetError>` (C++ `std::optional<PositionSetError>`).
    pub fn flip(&mut self) -> Option<PositionSetError> {
        // `Position::flip` re-sets the position, whose root state it keeps itself; the search
        // deep-copies `self.pos`, so there is no separate state list to update (C++ re-sets it
        // into `states.back()`).
        self.pos.flip()
    }

    /// Returns the board diagram for the current position.
    pub fn visualize(&self) -> String {
        self.pos.to_string()
    }

    /// Get thread count by NUMA node
    pub fn get_bound_thread_count_by_numa_node(&self) -> Vec<(usize, usize)> {
        let counts = self
            .thread_pool_data
            .read()
            .unwrap()
            .get_bound_thread_count_by_numa_node();
        let cfg = self.numa_context.get_numa_config();
        let mut ratios = Vec::new();

        for (n, &count) in counts.iter().enumerate() {
            ratios.push((count, cfg.num_cpus_in_numa_node(n)));
        }

        if !counts.is_empty() {
            for n in counts.len()..cfg.num_numa_nodes() {
                ratios.push((0, cfg.num_cpus_in_numa_node(n)));
            }
        }

        ratios
    }

    /// Get NUMA configuration as string
    pub fn get_numa_config_as_string(&self) -> String {
        self.numa_context.get_numa_config().to_string()
    }

    /// Get NUMA configuration information
    pub fn numa_config_information_as_string(&self) -> String {
        format!("Available processors: {}", self.get_numa_config_as_string())
    }

    /// Get thread binding information
    pub fn thread_binding_information_as_string(&self) -> String {
        let bound_threads = self.get_bound_thread_count_by_numa_node();
        if bound_threads.is_empty() {
            return String::new();
        }

        bound_threads
            .iter()
            .map(|(current, total)| format!("{}/{}", current, total))
            .collect::<Vec<_>>()
            .join(":")
    }

    /// Get thread allocation information
    pub fn thread_allocation_information_as_string(&self) -> String {
        let threads_size = self.thread_pool_data.read().unwrap().num_threads();
        let mut result = format!(
            "Using {} thread{}",
            threads_size,
            if threads_size > 1 { "s" } else { "" }
        );

        let binding_info = self.thread_binding_information_as_string();
        if !binding_info.is_empty() {
            result.push_str(&format!(" with NUMA node thread binding: {}", binding_info));
        }

        result
    }

    /// Get nodes searched in last search
    pub fn nodes_searched(&self) -> u64 {
        // No direct C++ counterpart: `uci.cpp` reads `threads.nodes_searched()`
        // itself. Exposed here because `uci.rs`, the bench and
        // `tests/invariants_search.rs` all need it.
        self.thread_pool_data.read().unwrap().nodes_searched()
    }
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// True when the default NNUE network cannot be found where `get_default_network`
    /// looks (next to the test binary, then the working directory). Tests that need the
    /// network call this first and return early: `Engine::verify_network` exits the
    /// whole process (as upstream does) when there is no network. With
    /// `STOCKCRAB_REQUIRE_NET` set a missing network panics instead of skipping, the
    /// same convention as `tests/common`.
    pub(crate) fn skip_without_network(test: &str) -> bool {
        let name = crate::evaluate::EVAL_FILE_DEFAULT_NAME;
        let beside_exe = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|d| d.join(name)));
        if std::path::Path::new(name).is_file() || beside_exe.is_some_and(|p| p.is_file()) {
            return false;
        }
        if std::env::var_os("STOCKCRAB_REQUIRE_NET").is_some() {
            panic!(
                "{test}: the default NNUE network is required but missing (run scripts/fetch-net.sh)"
            );
        }
        eprintln!("SKIP {test}: default NNUE network not found (run scripts/fetch-net.sh)");
        true
    }

    #[test]
    fn test_engine_creation() {
        let engine = Engine::new(None);
        assert_eq!(engine.fen(), START_FEN);
    }

    #[test]
    fn test_set_position() {
        let mut engine = Engine::new(None);
        assert!(
            engine
                .set_position(
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                    &[]
                )
                .is_none()
        );
        assert!(engine.fen().contains("4P3"));
    }

    #[test]
    fn test_options() {
        let engine = Engine::new(None);
        assert!(engine.get_options().get("Threads").is_some());
        assert!(engine.get_options().get("Hash").is_some());
        assert!(engine.get_options().get("Ponder").is_some());
    }

    #[test]
    fn test_numa_config() {
        let config = NumaConfig::new();
        assert!(config.num_numa_nodes() >= 1);
        assert!(config.num_cpus_in_numa_node(0) > 0);
    }

    #[test]
    fn test_perft_calculation() {
        if skip_without_network(module_path!()) {
            return;
        }
        let engine = Engine::new(None);
        let start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        let nodes = engine.perft(start_fen, 1, false).unwrap();
        assert_eq!(nodes, 20, "Perft(1) should return 20 for starting position");

        let nodes = engine.perft(start_fen, 2, false).unwrap();
        assert_eq!(
            nodes, 400,
            "Perft(2) should return 400 for starting position"
        );
    }

    #[test]
    fn test_uci_move_parsing() {
        let mut engine = Engine::new(None);
        let start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        // Test parsing a simple pawn move
        assert!(
            engine
                .set_position(start_fen, &["e2e4".to_string()])
                .is_none()
        );
        assert!(engine.fen().contains("4P3"), "Move e2e4 should be applied");

        // Test parsing multiple moves
        let err = engine.set_position(start_fen, &["e2e4".to_string(), "e7e5".to_string()]);
        assert!(err.is_none(), "Legal move sequence should not error");
        assert!(
            engine.fen().contains("4p3"),
            "Moves e2e4 e7e5 should be applied"
        );
    }

    #[test]
    fn test_ponderhit_releases_search() {
        if skip_without_network(module_path!()) {
            return;
        }
        use std::sync::mpsc;
        use std::time::Duration;

        let mut engine = Engine::new(None);
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        engine.set_on_bestmove(move |best, _ponder| {
            let _ = tx.lock().unwrap().send(best.to_string());
        });

        let mut limits = LimitsType::default();
        limits.depth = 4;
        limits.ponder_mode = true;
        engine.go(limits);

        // The depth limit is reached quickly, but a ponder search must not report
        // a bestmove until the GUI sends ponderhit or stop.
        assert!(rx.recv_timeout(Duration::from_millis(500)).is_err());

        engine.set_ponderhit(false);
        let best = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("bestmove after ponderhit");
        assert!(!best.is_empty());
        engine.wait_for_search_finished();
    }

    #[test]
    fn test_hash_table_operations() {
        let mut engine = Engine::new(None);

        // This should not panic
        engine.search_clear();
    }

    #[test]
    fn test_network_operations() {
        if skip_without_network(module_path!()) {
            return;
        }
        let mut engine = Engine::new(None);

        // These should not panic
        let net = engine.get_default_network();
        engine.network = LazyNumaReplicated::new(net);
        engine.verify_network();
    }

    #[test]
    fn test_thread_management() {
        let mut engine = Engine::new(None);

        {
            let mut opts = engine.get_options_mut();
            if let Some(opt) = opts.get_mut("Threads") {
                opt.set("2");
            }
        }
        engine.resize_threads();
        let thread_info = engine.thread_allocation_information_as_string();
        assert!(
            thread_info.starts_with("Using 2 thread"),
            "unexpected thread info: {thread_info}"
        );
    }

    #[test]
    fn history_groups_resize_and_use_local_worker_indices() {
        let mut engine = Engine::new(None);
        engine.set_option("name NumaPolicy value none");
        let base_size = engine.shared_histories.correction_history.get_size();
        for count in [1usize, 3, 5, 8] {
            engine.set_option(&format!("name Threads value {count}"));
            let pool = engine.thread_pool_data.read().unwrap();
            let mut group = None;
            for (index, thread) in pool.iter().enumerate() {
                thread.wait_for_search_finished();
                let worker_handle = thread.worker();
                let guard = worker_handle.lock().unwrap();
                let worker = guard.as_ref().unwrap();
                assert_eq!(worker.numa_thread_idx, index);
                assert_eq!(worker.numa_total, count);
                let history = &worker.shared_state.shared_histories;
                assert_eq!(
                    history.correction_history.get_size(),
                    base_size * count.next_power_of_two()
                );
                if let Some(first) = &group {
                    assert!(Arc::ptr_eq(first, history));
                } else {
                    group = Some(Arc::clone(history));
                }
            }
        }
    }

    #[test]
    fn production_voting_keeps_the_taken_main_worker_as_a_candidate() {
        let mut engine = Engine::new(None);
        engine.set_option("name NumaPolicy value none");
        engine.set_option("name Threads value 2");
        let pool = engine.thread_pool_data.read().unwrap();
        for thread in pool.iter() {
            thread.wait_for_search_finished();
        }
        let main_thread = pool.main_thread().unwrap();
        let mut main = main_thread.worker().lock().unwrap().take().unwrap();
        main.root_moves = vec![crate::thread::RootMove::new(Move::from_to(
            crate::types::Square::SQ_E2,
            crate::types::Square::SQ_E4,
        ))];
        main.root_moves[0].score = 100;
        let helper = pool.iter().nth(1).unwrap();
        {
            let worker_handle = helper.worker();
            let mut guard = worker_handle.lock().unwrap();
            let worker = guard.as_mut().unwrap();
            worker.root_moves = vec![crate::thread::RootMove::new(Move::from_to(
                crate::types::Square::SQ_D2,
                crate::types::Square::SQ_D4,
            ))];
            worker.root_moves[0].score = -100;
        }
        assert_eq!(pool.get_best_thread_with_main(&main).id(), 0);
        helper.worker().lock().unwrap().as_mut().unwrap().root_moves[0].score = 200;
        assert_eq!(pool.get_best_thread_with_main(&main).id(), 1);
        *main_thread.worker().lock().unwrap() = Some(main);
    }

    #[test]
    fn selected_helper_pv_uses_the_helpers_depth() {
        let mut engine = Engine::new(None);
        engine.set_option("name NumaPolicy value none");
        engine.set_option("name Threads value 2");
        let (send, receive) = std::sync::mpsc::channel();
        engine.set_on_update_full(move |info| send.send(info).unwrap());
        let pool = engine.thread_pool_data.read().unwrap();
        let main_thread = pool.main_thread().unwrap();
        let mut main = main_thread.worker().lock().unwrap().take().unwrap();
        main.root_pos = engine.pos.clone();
        main.root_depth = 3;
        main.completed_depth = 2;
        main.root_moves = vec![crate::thread::RootMove::new(Move::from_to(
            crate::types::Square::SQ_E2,
            crate::types::Square::SQ_E4,
        ))];
        main.root_moves[0].score = -100;
        let helper = pool.iter().nth(1).unwrap();
        {
            let handle = helper.worker();
            let mut held = handle.lock().unwrap();
            let selected = held.as_mut().unwrap();
            selected.root_depth = 8;
            selected.completed_depth = 7;
            selected.root_moves = vec![crate::thread::RootMove::new(Move::from_to(
                crate::types::Square::SQ_D2,
                crate::types::Square::SQ_D4,
            ))];
            selected.root_moves[0].score = 200;
        }
        let selected = pool.get_best_thread_with_main(&main);
        assert_eq!(selected.id(), 1);
        {
            let handle = selected.worker();
            let held = handle.lock().unwrap();
            main.adopt_thread_result(held.as_ref().unwrap());
        }
        main.manager.as_ref().unwrap().pv(
            &main,
            &pool,
            &engine.tt.read().unwrap(),
            main.root_depth,
            &[],
        );
        let info = receive
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        assert_eq!(info.depth, 8);
        assert_eq!(info.pv, "d2d4");
        assert_eq!(main.completed_depth, 7);
        *main_thread.worker().lock().unwrap() = Some(main);
    }

    /// C++ `output_pv` for a search stopped before its first iteration: the root
    /// move falls back on its previous score and previous PV, neither of which
    /// exists yet, so the line reads depth 1, score 0 and an empty PV.
    #[test]
    fn unsearched_root_move_prints_depth_one_and_an_empty_pv() {
        let mut engine = Engine::new(None);
        engine.set_option("name NumaPolicy value none");
        let (send, receive) = std::sync::mpsc::channel();
        engine.set_on_update_full(move |info| send.send(info).unwrap());
        let pool = engine.thread_pool_data.read().unwrap();
        let main_thread = pool.main_thread().unwrap();
        main_thread.wait_for_search_finished();
        let mut main = main_thread.worker().lock().unwrap().take().unwrap();
        main.root_pos = engine.pos.clone();
        main.root_depth = 0;
        main.root_moves = vec![crate::thread::RootMove::new(Move::from_to(
            crate::types::Square::SQ_A2,
            crate::types::Square::SQ_A3,
        ))];
        main.manager.as_ref().unwrap().pv(
            &main,
            &pool,
            &engine.tt.read().unwrap(),
            main.root_depth,
            &[],
        );
        let info = receive
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        assert_eq!(info.depth, 1);
        assert_eq!(info.pv, "");
        assert_eq!(main.root_moves[0].pv.len(), 1);
        *main_thread.worker().lock().unwrap() = Some(main);
    }

    #[test]
    fn production_search_history_is_owned_before_engine_storage_is_replaced() {
        if skip_without_network(module_path!()) {
            return;
        }
        let mut engine = Engine::new(None);
        assert!(
            engine
                .set_position(
                    START_FEN,
                    &["g1f3".into(), "g8f6".into(), "f3g1".into(), "f6g8".into()],
                )
                .is_none()
        );
        let expected = engine.pos.clone();
        engine.go(LimitsType {
            depth: 1,
            ..LimitsType::default()
        });
        engine.wait_for_search_finished();
        let pool = engine.thread_pool_data.read().unwrap();
        let main_thread = pool.main_thread().unwrap();
        let handle = main_thread.worker();
        let held = handle.lock().unwrap();
        let root = &held.as_ref().unwrap().root_pos;
        assert!(
            !core::ptr::eq(root.st(), engine.pos.st()),
            "search borrows Engine state"
        );
        assert_eq!(root.has_repeated(), expected.has_repeated());
        drop(held);
        drop(pool);
        assert!(
            engine
                .set_position(START_FEN, &["e2e4".into(), "e7e5".into()])
                .is_none()
        );
        let held = handle.lock().unwrap();
        let root = &held.as_ref().unwrap().root_pos;
        assert_eq!(root.key(), expected.key());
        assert_eq!(root.has_repeated(), expected.has_repeated());
    }

    #[test]
    fn canonical_numa_snapshot_shares_storage_and_replacements_preserve_readers() {
        #[derive(Clone)]
        struct Value(usize);
        impl NumaReplicable for Value {
            fn replicate(&self, _: usize) -> Self {
                self.clone()
            }
        }
        let network = LazyNumaReplicated::new(Value(7));
        let first = network.get(crate::thread::NumaReplicatedAccessToken::new(0));
        assert!(Arc::ptr_eq(&first, &network.lock().unwrap()));
        let same = network.get(crate::thread::NumaReplicatedAccessToken::new(0));
        assert!(Arc::ptr_eq(&first, &same));
        network.modify_and_replicate(|value| value.0 = 9);
        assert_eq!(first.0, 7);
        let next = network.get(crate::thread::NumaReplicatedAccessToken::new(0));
        assert_eq!(next.0, 9);
        assert!(Arc::ptr_eq(&next, &network.lock().unwrap()));
        let other = network.get(crate::thread::NumaReplicatedAccessToken::new(1));
        assert!(!Arc::ptr_eq(&next, &other));
    }

    #[test]
    fn test_elo_bounds() {
        let engine = Engine::new(None);
        let opts = engine.get_options();
        let elo_opt = opts.get("UCI_Elo").expect("UCI_Elo registered");
        let min = elo_opt.get_min().unwrap_or(0);
        let max = elo_opt.get_max().unwrap_or(0);
        assert_eq!(min, 1320, "Minimum ELO should be 1320");
        assert_eq!(max, 3190, "Maximum ELO should be 3190");
    }

    #[test]
    fn test_syzygy_options() {
        let mut engine = Engine::new(None);

        // Set Syzygy path (even if tables don't exist, setting should work)
        {
            let mut opts = engine.get_options_mut();
            if let Some(opt) = opts.get_mut("SyzygyPath") {
                opt.set("/tmp/syzygy");
            }
        }

        // Should not panic
    }

    #[test]
    fn test_debug_log_options() {
        let mut engine = Engine::new(None);

        {
            let mut opts = engine.get_options_mut();
            if let Some(opt) = opts.get_mut("Debug Log File") {
                opt.set("/tmp/stockfish_test.log");
            }
        }

        // Should not panic
    }

    #[test]
    fn test_numa_auto_config() {
        let mut engine = Engine::new(None);

        engine.set_numa_config_from_option("auto");
        let numa_info = engine.get_numa_config_as_string();
        assert!(!numa_info.is_empty());
    }
}
