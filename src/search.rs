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

//! Alpha-beta search: port of Stockfish search.h/search.cpp. `Worker` holds the per-thread search
//! state; `NonPV`/`PV`/`Root` implement `NodeType` in place of the C++ template parameter.

use core::cmp::{max, min};
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::bitboard_fast::*;
use crate::evaluate::{AccumulatorCaches, Networks, evaluate as nnue_evaluate};
use crate::history::*;
use crate::misc::{PRNG, now};
use crate::movegen::{GenType, generate};
use crate::movepick::MovePicker;
use crate::nnue::nnue_accumulator::AccumulatorStack;
use crate::position::Position;
use crate::syzygy::{ProbeState, WDLScore, rank_root_moves};
use crate::thread::{InfoIteration, InfoShort, RootMove, SharedState};
use crate::thread::{
    NumaReplicatedAccessToken, SearchControl, SearchManager, TablebaseConfig, WorkerCounters,
};
use crate::tt::TranspositionTable;
use crate::types::*;
use crate::uci::UciEngine;
use crate::ucioption::OptionsMap;

// Type aliases
type Ply = i32;

// Constants
const MAX_PLY: i32 = 246;
const SEARCHEDLIST_CAPACITY: usize = 32;

// Only start sending PV info when the search has gone beyond this node count.
const NODES_LIMIT_OUTPUT: u64 = 10_000_000;

// Reductions lookup table indexed by depth or move number.
// C++ builds it once at start-up as int(2872 / 128.0 * std::log(i)); `ln` is not a `const fn`,
// so the values are written out here and checked by
// `tests::reductions_table_matches_formula`.
static REDUCTIONS: [i32; crate::movegen::MAX_MOVES] = [
    0, 0, 15, 24, 31, 36, 40, 43, 46, 49, 51, 53, 55, 57, 59, 60, 62, 63, 64, 66, 67, 68, 69, 70,
    71, 72, 73, 73, 74, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 86,
    86, 87, 87, 88, 88, 89, 89, 89, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 94, 94, 94, 95, 95, 95,
    95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 98, 99, 99, 99, 99, 100, 100, 100, 100, 101, 101, 101,
    101, 102, 102, 102, 102, 103, 103, 103, 103, 103, 104, 104, 104, 104, 105, 105, 105, 105, 105,
    106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 109, 109, 109,
    109, 109, 109, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112,
    112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 115,
    115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 119,
    119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121,
    121, 121, 121, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123,
    123, 123, 123, 123, 123, 123, 124, 124, 124, 124,
];

/// `x / d` for a divisor the caller guarantees to be positive. Rust guards every signed division by
/// a variable with a divide-by-zero test and an `i32::MIN / -1` overflow test (two extra
/// branches); C++ has neither, so telling the compiler `d > 0` brings the hot-path divisions
/// back to a bare `idiv`.
///
/// # Safety
/// `d` must be greater than zero.
#[inline(always)]
unsafe fn div_pos(x: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    // SAFETY: the caller guarantees `d > 0`.
    unsafe { core::hint::assert_unchecked(d > 0) };
    x / d
}

// LMR history divisor, a quadratic in the depth (upstream `lmr_divisor`).
#[inline(always)]
fn lmr_divisor(depth: i32) -> i32 {
    let d = min(depth, 16);
    3000 + 7 * (d - 8) * (d - 8)
}

// Value constants
const VALUE_NONE: Value = 32002;
const VALUE_INFINITE: Value = 32001;
const VALUE_DRAW: Value = 0;
// Matches upstream types.h: VALUE_TB = VALUE_MATE_IN_MAX_PLY - 1 (31753).
const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;
const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY;
const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;
const VALUE_MATE: Value = 32000;
const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;

const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;
const VALUE_ZERO: Value = 0;

// These module-local constants MUST stay identical to the ones in `crate::types`
// (which uci.rs, tt.rs and syzygy.rs use); a mismatch would give search.rs its own
// is_win/is_loss/is_decisive thresholds. Checked at compile time, which is stronger than
// a unit test. See also tests/regression_parity.rs::bug8_value_tb_and_classification_thresholds.
const _: () = {
    assert!(VALUE_NONE == crate::types::VALUE_NONE);
    assert!(VALUE_INFINITE == crate::types::VALUE_INFINITE);
    assert!(VALUE_MATE == crate::types::VALUE_MATE);
    assert!(VALUE_MATE_IN_MAX_PLY == crate::types::VALUE_MATE_IN_MAX_PLY);
    assert!(VALUE_MATED_IN_MAX_PLY == crate::types::VALUE_MATED_IN_MAX_PLY);
    assert!(VALUE_TB == crate::types::VALUE_TB);
    assert!(VALUE_TB_WIN_IN_MAX_PLY == crate::types::VALUE_TB_WIN_IN_MAX_PLY);
    assert!(VALUE_TB_LOSS_IN_MAX_PLY == crate::types::VALUE_TB_LOSS_IN_MAX_PLY);
    assert!(VALUE_DRAW == crate::types::VALUE_DRAW);
    assert!(VALUE_ZERO == crate::types::VALUE_ZERO);
};

const DEPTH_QS: Depth = 0;

// Piece/Square constants
const NO_PIECE: Piece = Piece::NoPiece;

/// Non-PV node type (C++ `NodeType::NonPV`): a zero-window search.
pub struct NonPV;
/// PV node type (C++ `NodeType::PV`): a full-window search off the root.
pub struct PV;
/// The root node (C++ `NodeType::Root`), searched as a PV node.
pub struct Root;

/// Different node types, used as a template parameter
pub trait NodeType {
    const IS_PV: bool;
    const IS_ROOT: bool;
}

impl NodeType for NonPV {
    const IS_PV: bool = false;
    const IS_ROOT: bool = false;
}

impl NodeType for PV {
    const IS_PV: bool = true;
    const IS_ROOT: bool = false;
}

impl NodeType for Root {
    const IS_PV: bool = true;
    const IS_ROOT: bool = true;
}

// SearchedList for tracking searched moves - stack allocated to avoid heap allocation in hot path
#[derive(Clone)]
struct SearchedList {
    // Only `moves[..len]` is ever read, so the array needs no initialisation (C++ leaves it
    // uninitialised too); zero-filling it cost a 64-byte store per list on every node.
    moves: [core::mem::MaybeUninit<Move>; SEARCHEDLIST_CAPACITY],
    len: usize,
}

impl SearchedList {
    #[inline(always)]
    fn new() -> Self {
        Self {
            moves: [const { core::mem::MaybeUninit::uninit() }; SEARCHEDLIST_CAPACITY],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, mv: Move) {
        debug_assert!(self.len < SEARCHEDLIST_CAPACITY);
        // SAFETY: callers push each searched move at most once and only while
        // move_count <= SEARCHEDLIST_CAPACITY (see the gate in `search`), so len < 32 on entry.
        unsafe { self.moves.get_unchecked_mut(self.len).write(mv) };
        self.len += 1;
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a> IntoIterator for &'a SearchedList {
    type Item = &'a Move;
    type IntoIter = core::slice::Iter<'a, Move>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: len is always <= SEARCHEDLIST_CAPACITY (32), and `push` initialises
        // `moves[i]` before it increments `len` past `i`, so `moves[..len]` is initialised.
        unsafe {
            self.moves
                .get_unchecked(..self.len)
                .assume_init_ref()
                .iter()
        }
    }
}

// Stack struct for search tree state
/// Stack struct keeps track of the information we need to remember from nodes
/// shallower and deeper in the tree during the search. Each search thread has
/// its own array of Stack objects, indexed by the current ply.
#[repr(C)]
pub struct Stack {
    pub pv: *mut Move,
    pub continuation_history: *const PieceToHistory,
    /// Shared pointer into the worker-owned Cell correction table.
    pub continuation_correction_history: *const PieceToCorrectionHistory,
    pub ply: i32,
    pub current_move: Move,
    pub excluded_move: Move,
    pub static_eval: Value,
    pub stat_score: i32,
    pub move_count: i32,
    pub in_check: bool,
    pub tt_pv: bool,
    pub tt_hit: bool,
    pub follow_pv: bool,
    pub cutoff_cnt: i32,
    pub reduction: i32,
    pub prior_nmp_fail_high: i32,
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            pv: core::ptr::null_mut(),
            continuation_history: core::ptr::null(),
            continuation_correction_history: core::ptr::null(),
            ply: 0,
            current_move: Move::none(),
            excluded_move: Move::none(),
            static_eval: VALUE_NONE,
            stat_score: 0,
            move_count: 0,
            in_check: false,
            tt_pv: false,
            tt_hit: false,
            follow_pv: false,
            cutoff_cnt: 0,
            reduction: 0,
            prior_nmp_fail_high: 0,
        }
    }
}

impl Clone for Stack {
    fn clone(&self) -> Self {
        Stack {
            pv: self.pv,
            continuation_history: self.continuation_history,
            continuation_correction_history: self.continuation_correction_history,
            ply: self.ply,
            current_move: self.current_move,
            excluded_move: self.excluded_move,
            static_eval: self.static_eval,
            stat_score: self.stat_score,
            move_count: self.move_count,
            in_check: self.in_check,
            tt_pv: self.tt_pv,
            tt_hit: self.tt_hit,
            follow_pv: self.follow_pv,
            cutoff_cnt: self.cutoff_cnt,
            reduction: self.reduction,
            prior_nmp_fail_high: self.prior_nmp_fail_high,
        }
    }
}

pub type RootMoves = Vec<RootMove>;

// Limits structure for search constraints
/// LimitsType struct stores information sent by the caller about the analysis required.
#[derive(Clone, Debug, Default)]
pub struct LimitsType {
    pub searchmoves: Vec<String>,
    pub time: [i64; 2],    // Time for white and black (C++ TimePoint)
    pub inc: [i64; 2],     // Increment for white and black
    pub npmsec: i64,       // Nodes per millisecond
    pub movetime: i64,     // Fixed time per move
    pub start_time: i64,   // Search start time
    pub movestogo: i32,    // Moves until time control
    pub depth: Depth,      // Maximum depth
    pub mate: i32,         // Search for mate in N
    pub perft: i32,        // Perft depth
    pub infinite: i32,     // Infinite search mode
    pub nodes: u64,        // Maximum nodes
    pub ponder_mode: bool, // Ponder mode flag
}

impl LimitsType {
    /// True when the GUI sent wtime/btime.
    pub fn use_time_management(&self) -> bool {
        self.time[Color::White as usize] != 0 || self.time[Color::Black as usize] != 0
    }
}

/// Strength limiting for the `Skill Level` / `UCI_LimitStrength` options (C++ `Skill`): picks a
/// weaker best move from the MultiPV lines.
pub struct Skill {
    pub level: f64,
    pub best: Move,
}

impl Skill {
    const LOWEST_ELO: i32 = 1320;
    const HIGHEST_ELO: i32 = 3190;

    pub fn new(skill_level: i32, uci_elo: i32) -> Self {
        // Match C++ exactly: only clamp when uci_elo is set (UCI_LimitStrength mode)
        // When uci_elo is 0, use skill_level directly without clamping
        let level = if uci_elo != 0 {
            let e =
                (uci_elo - Self::LOWEST_ELO) as f64 / (Self::HIGHEST_ELO - Self::LOWEST_ELO) as f64;
            ((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438
        } else {
            skill_level as f64
        };

        Skill {
            // Only clamp in UCI_LimitStrength mode (when uci_elo != 0)
            level: if uci_elo != 0 {
                level.clamp(0.0, 19.0)
            } else {
                level
            },
            best: Move::none(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.level < 20.0
    }

    pub fn time_to_pick(&self, depth: Depth) -> bool {
        depth == 1 + self.level as i32
    }

    /// When playing with strength handicap, choose the best move among a set of
    /// RootMoves using a statistical rule dependent on 'level'.
    /// Idea by Heinz van Saanen.
    pub fn pick_best(&mut self, root_moves: &RootMoves, multi_pv: usize) -> Move {
        // RootMoves are already sorted by score in descending order
        let seed = (now() as u64).max(1); // Ensure seed is never 0
        let mut rng = PRNG::new(seed);

        // With tablebases at the root, rootMoves are ordered by tbRank rather than by
        // score, so compute the score range explicitly to keep 'delta' non-negative.
        let mut top_score = root_moves[0].score;
        let mut min_score = root_moves[0].score;
        for i in 1..multi_pv {
            top_score = max(top_score, root_moves[i].score);
            min_score = min(min_score, root_moves[i].score);
        }
        let delta = min(top_score - min_score, PAWN_VALUE);
        let mut max_score = -VALUE_INFINITE;
        let weakness = 120 - 2 * self.level as i32;

        // Choose best move. For each move score we add two terms, both dependent on
        // weakness. One is deterministic and bigger for weaker levels, and one is
        // random. Then we choose the move with the resulting highest score.
        for i in 0..multi_pv {
            // This is our magic formula - matching C++ exactly
            let push = (weakness * (top_score - root_moves[i].score)
                + delta * (rng.rand::<u64>() as u32 % weakness as u32) as i32)
                / 128;

            if root_moves[i].score + push >= max_score {
                max_score = root_moves[i].score + push;
                self.best = root_moves[i].pv[0];
            }
        }

        self.best
    }
}

/// Detect shuffling moves in order to limit search explosions. Added in upstream PR #6447 as
/// non-regression, so its parameters should not be tuned.
#[inline(always)]
fn is_shuffling(mv: Move, ss: *const Stack, pos: &Position) -> bool {
    // SAFETY: `ss` points into the Stack array allocated in iterative_deepening (ply 0 at index 7, so there
    // are 7 entries of front padding); `ss-2`/`ss-4` are only reached once ply >= 20, well inside the array.
    unsafe {
        // Early exit for captures or low rule50
        if pos.capture_stage(mv) || pos.rule50_count() < 10 {
            return false;
        }

        // Early exit for low plies from null or low ply
        if pos.plies_from_null() < 6 || (*ss).ply < 20 {
            return false;
        }

        // Check if the move is going back to where a previous piece came from
        let ss_m2 = ss.offset(-2);
        let ss_m4 = ss.offset(-4);

        mv.from_sq() == (*ss_m2).current_move.to_sq()
            && (*ss_m2).current_move.from_sq() == (*ss_m4).current_move.to_sq()
    }
}

/// Search::Worker is the class that does the actual search.
/// It is instantiated once per thread, and it is responsible for keeping track
/// of the search history, and storing data required for the search.
/// A worker moves between threads under exclusive ownership. Its Cell-backed
/// correction history cannot be shared for concurrent access.
/// ```compile_fail
/// fn requires_sync<T: Sync>() {}
/// requires_sync::<stockcrab::search::Worker>();
/// ```
pub struct Worker {
    pub shared_state: SharedState,
    // The main thread has a SearchManager, the others have None
    pub manager: Option<Box<SearchManager>>,
    pub thread_idx: usize, // Match C++ field name exactly
    // C++ `numaThreadIdx` / `numaTotal` (search.h). Needed by clear() to
    // split the shared histories between the threads of this NUMA node; keeping
    // them on the Worker avoids taking any lock while clearing.
    pub numa_thread_idx: usize,
    pub numa_total: usize,
    pub numa_access_token: NumaReplicatedAccessToken, // Match C++ field name exactly
    tt_ptr: *const TranspositionTable,
    network_snapshot: Arc<Networks>,
    search_control: Arc<SearchControl>, // Lock-free access to search flags

    // Search state
    pub limits: LimitsType,
    pub root_moves: RootMoves,
    pub root_pos: Position,
    pub tb_config: TablebaseConfig,

    // Statistics
    /// `nodes`, `tb_hits` and `best_move_changes`, shared with the owning `Thread` so that other
    /// threads can sum them while this worker is searching (the worker itself is moved out of its
    /// mutex for the duration of a search, so it cannot be reached through the pool).
    pub counters: Arc<WorkerCounters>,
    pub nmp_min_ply: AtomicUsize,
    pub root_depth: Depth,
    seek_mate: bool,
    pub completed_depth: Depth,
    pub pv_idx: usize,
    pub pv_last: usize,
    pub sel_depth: i32,

    // PV of the current (pvIdx) root move as of the previous iteration, used to
    // detect nodes that are still following the principal variation (followPV).
    pub last_iteration_idx_pv: Vec<Move>,

    // History tables
    pub main_history: Box<ButterflyHistory>,
    pub low_ply_history: Box<LowPlyHistory>,
    pub capture_history: Box<CapturePieceToHistory>,
    // Per SF18: pawn/minor/non_pawn correction histories are now stored in the
    // unified shared correction history (SharedHistories.correction_history).
    pub continuation_correction_history: ContinuationCorrectionHistoryOwner,
    pub tt_move_history: TTMoveHistory,

    // NNUE evaluation state
    pub accumulator_stack: AccumulatorStack,
    pub refresh_table: AccumulatorCaches,
}

impl Worker {
    pub fn new(
        shared_state: SharedState,
        manager: Option<Box<SearchManager>>,
        thread_idx: usize,
        numa_thread_idx: usize,
        numa_total: usize,
        numa_access_token: NumaReplicatedAccessToken,
    ) -> Self {
        // The TT pointer is only valid during a search: `start_searching` points it at the
        // table while holding the TT read lock and clears it again afterwards.
        let tt_ptr: *const TranspositionTable = core::ptr::null();
        let network_snapshot = shared_state.networks.get(numa_access_token);
        let search_control = Arc::clone(&shared_state.search_control);

        let worker = Self {
            shared_state,
            manager,
            thread_idx,
            numa_thread_idx,
            numa_total,
            numa_access_token,
            tt_ptr,
            network_snapshot,
            search_control,
            limits: LimitsType::default(),
            root_moves: Vec::new(),
            root_pos: Position::default(),
            tb_config: TablebaseConfig::default(),
            counters: Arc::new(WorkerCounters::default()),
            nmp_min_ply: AtomicUsize::new(0),
            root_depth: 0,
            seek_mate: false,
            completed_depth: 0,
            pv_idx: 0,
            pv_last: 0,
            sel_depth: 0,
            last_iteration_idx_pv: Vec::new(),
            // Public because they need to be updatable by the stats
            main_history: new_butterfly_history(),
            low_ply_history: new_low_ply_history(),
            capture_history: new_capture_piece_to_history(),
            continuation_correction_history: new_continuation_correction_history(),
            tt_move_history: new_tt_move_history(),

            accumulator_stack: AccumulatorStack::new(),
            refresh_table: AccumulatorCaches::new(),
        };

        // Histories are filled by clear(), which ThreadPool::clear() calls before any use.
        worker
    }

    #[inline]
    pub fn continuation_history(&self) -> &[[ContinuationHistory; 2]; 2] {
        self.shared_state.shared_histories.continuation_history()
    }

    #[inline(always)]
    fn continuation_correction_entry(
        &self,
        entry: *const PieceToCorrectionHistory,
    ) -> &PieceToCorrectionHistory {
        debug_assert!(!entry.is_null());
        // SAFETY: internal stack pointers target this worker's sole retained
        // correction-table allocation. Recursive Worker borrows do not create
        // exclusive table references; updates use Cell through shared borrows.
        unsafe { &*entry }
    }

    /// The shared transposition table. Only callable inside `start_searching`.
    ///
    /// The returned lifetime is deliberately not tied to `&self`: the search keeps `TTWriter`s
    /// (which borrow the table) alive across `&mut self` calls on the worker, which a
    /// `&self`-bound borrow would reject. It is instead bounded by the read guard that
    /// `start_searching` holds, see there.
    #[inline(always)]
    fn tt<'a>(&self) -> &'a TranspositionTable {
        debug_assert!(!self.tt_ptr.is_null(), "TT used outside of a search");
        // SAFETY: `tt_ptr` is non-null only between the start and the end of
        // `start_searching`, where it points at the table inside the `Arc<RwLock<..>>` of
        // `shared_state.tt` and a read guard on that lock is held. The table is moved only with
        // the `Arc` (which the worker keeps alive), and resized/cleared/dropped only through the
        // lock's write guard, which cannot be taken while the read guard is held. So the
        // pointee outlives every use inside the search, and the references handed out here never
        // outlive `start_searching` (they only live in its callees' locals).
        unsafe { &*self.tt_ptr }
    }

    /// Lock-free access to search control flags (hot path)
    #[inline(always)]
    fn should_stop(&self) -> bool {
        self.search_control.stop()
    }

    #[inline(always)]
    fn should_increase_depth(&self) -> bool {
        self.search_control.increase_depth()
    }

    /// C++ `threads.stop = value` (thread.h).
    ///
    /// `ThreadPool` does not own a second copy of the flag: it forwards to this
    /// very `SearchControl`, so no lock on the thread pool is needed (taking one here
    /// could deadlock against `Engine::go`'s guard).
    #[inline(always)]
    fn set_stop(&self, value: bool) {
        self.search_control.set_stop(value);
    }

    /// Reset histories, usually before a new game.
    pub fn clear(&mut self) {
        // Clear history tables with specific values from search.cpp Worker::clear()

        // mainHistory.fill(-5)
        for from in &mut *self.main_history {
            for to in from {
                to.set(-5);
            }
        }

        // captureHistory.fill(-742)
        for piece in &mut *self.capture_history {
            for to in piece {
                for captured in to {
                    captured.set(-742);
                }
            }
        }

        // Clear the shared pawn and correction histories for this worker's range.
        // C++:
        //   sharedHistory.correctionHistory.clear_range(-5,    numaThreadIdx, numaTotal);
        //   sharedHistory.pawnHistory.clear_range(-1338, numaThreadIdx, numaTotal);
        // No lock is taken here: ThreadPool::clear() already holds the pool lock.
        // SAFETY: `clear` runs while no search is active, and each worker clears its own partition
        // (`numa_thread_idx < numa_total`, the latter clamped to at least 1).
        unsafe {
            self.shared_state.shared_histories.clear_range(
                -5,
                -1338,
                self.numa_thread_idx,
                self.numa_total.max(1),
            );
        }

        // ttMoveHistory = 0
        self.tt_move_history.set(0);

        // continuationCorrectionHistory.fill(5)
        for piece1 in &*self.continuation_correction_history {
            for to1 in piece1 {
                for piece2 in to1 {
                    for to2 in piece2 {
                        to2.set(5);
                    }
                }
            }
        }

        // continuationHistory[inCheck][c].fill(-586)
        // This constant-size history is shared by the workers in one NUMA
        // node, so only that node's worker 0 may initialize it.
        if self.numa_thread_idx == 0 {
            // SAFETY: `clear` runs while no search is active, and only the worker with numa_thread_idx == 0 on this node writes this shared block (see above).
            unsafe {
                self.shared_state
                    .shared_histories
                    .fill_continuation_history(-586);
            }
        }

        // lowPlyHistory.fill(102) - this is done in iterative_deepening

        // Clear refresh table with networks
        self.refresh_table.clear(&self.network_snapshot);

        // Clear manager (only the main thread has one)
        if let Some(manager) = self.manager.as_deref_mut() {
            manager.clear();
        }

        // Reset statistics
        self.counters.nodes.store(0, Ordering::Relaxed);
        self.counters.tb_hits.store(0, Ordering::Relaxed);
        self.nmp_min_ply.store(0, Ordering::Relaxed);
        self.counters.best_move_changes.store(0, Ordering::Relaxed);
        self.root_depth = 0;
        self.completed_depth = 0;
    }

    pub fn ensure_network_replicated(&mut self) {
        self.network_snapshot = self.shared_state.networks.get(self.numa_access_token);
        self.refresh_table.clear(&self.network_snapshot);
    }

    pub fn is_main_thread(&self) -> bool {
        self.thread_idx == 0
    }
    /// Reference to the search manager; main thread only.
    pub fn main_manager(&self) -> &SearchManager {
        debug_assert!(self.is_main_thread());
        debug_assert!(self.manager.is_some());
        // Mirrors C++ `assert(manager.get() != nullptr)`: only the main thread is
        // constructed with a SearchManager, and every caller is already behind an
        // `is_main_thread()` guard.
        self.manager.as_deref().unwrap()
    }

    // Main search entry point
    /// Called when the program receives the UCI 'go' command.
    /// It searches from the root position and outputs the "bestmove".
    pub fn start_searching(&mut self) {
        self.with_tt_locked(Self::start_searching_locked);
    }

    /// Runs `f` with the TT read lock held and `tt_ptr` pointing at the table, so that
    /// `resize`/`clear` (write lock) cannot run or free the table while this worker uses it. The
    /// pointer is re-derived from the guard on every outermost call (so it is never stale) and
    /// cleared afterwards, also when `f` panics. Nested calls reuse the outer lock (taking a
    /// second read lock could deadlock against a waiting writer). Costs one uncontended lock per
    /// search, nothing on the hot path.
    fn with_tt_locked<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        if !self.tt_ptr.is_null() {
            return f(self);
        }
        let tt_lock = Arc::clone(&self.shared_state.tt);
        let tt_guard = tt_lock.read().unwrap();
        tt_guard.mark_search_active();
        self.tt_ptr = &*tt_guard;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(self)));
        self.tt_ptr = core::ptr::null();
        drop(tt_guard);
        match result {
            Ok(r) => r,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn start_searching_locked(&mut self) {
        self.accumulator_stack.reset();

        // Non-main threads go directly to iterative_deepening
        if !self.is_main_thread() {
            self.iterative_deepening();
            return;
        }

        // Main thread initialization - use trait method
        let mut limits_copy = self.limits.clone();
        let mut original_time_adjust_val = -1.0f64; // This will be set by SearchManager

        self.main_manager().init_time_management(
            &mut limits_copy,
            self.root_pos.side_to_move(),
            self.root_pos.game_ply(),
            &self.shared_state.options.read().unwrap(),
            &mut original_time_adjust_val,
        );
        // Store back the updated limits
        self.limits = limits_copy;
        self.tt().new_search();
        // C++: `main_manager()->updates.onStart();`
        self.main_manager().call_on_start();

        if self.root_moves.is_empty() {
            // Report no moves and immediately emit a null bestmove, then return.
            let info = InfoShort {
                depth: 0,
                score: crate::score::Score::from_value(
                    if self.root_pos.checkers() != 0 {
                        -VALUE_MATE
                    } else {
                        VALUE_DRAW
                    },
                    &self.root_pos,
                ),
            };
            self.main_manager().call_on_update_no_moves(info);
            let bestmove = UciEngine::move_str(Move::none(), self.root_pos.is_chess960());
            self.main_manager().call_on_bestmove(&bestmove, "");
            return;
        }

        // Main thread starts non-main threads, and begins own search.
        self.shared_state.threads.read().unwrap().start_searching();
        let mut uci_pv_sent = self.iterative_deepening();

        // Wait for stop signal only if in ponder or infinite mode
        if self.main_manager().get_ponder() || self.limits.infinite != 0 {
            while !self.should_stop()
                // When we reach the maximum depth, we can arrive here without a raise of
                // threads.stop. However, if we are pondering or in an infinite search,
                // the UCI protocol states that we shouldn't print the best move before the
                // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
                // until the GUI sends one of those commands.
                && (self.main_manager().get_ponder() || self.limits.infinite != 0)
            {
                std::thread::yield_now();
            }
        }

        // Set stop flag via lock-free SearchControl
        if !self.should_stop() {
            self.set_stop(true);
        }

        // C++ search.cpp `threads.wait_for_search_finished();`
        //
        // Without it the main worker would announce the bestmove while the helper
        // threads are still inside iterative_deepening, so (a) `get_best_thread()`
        // below could not see their results and (b) the next `go` would start on a
        // pool that still has live workers.
        //
        // The pool read guard is dropped *before* blocking on the helpers: never
        // hold a lock the awaited threads might need.
        let helpers = self.shared_state.threads.read().unwrap().helper_threads();
        for helper in helpers {
            // Wait until all threads have finished
            helper.wait_for_search_finished();
        }

        // Handle nodes as time mode
        // When playing in 'nodes as time' mode, subtract the searched nodes from the available
        // ones before exiting.
        if self.limits.npmsec > 0 {
            // SAFETY: side_to_move returns Color (0 or 1), inc has 2 elements
            let nodes_used = self
                .shared_state
                .threads
                .read()
                .unwrap()
                .nodes_searched_from_worker(self) as i64
                - unsafe {
                    *self
                        .limits
                        .inc
                        .get_unchecked(self.root_pos.side_to_move() as usize)
                };
            // Advance nodes time by subtracting the nodes used from available nodes
            self.main_manager().advance_nodes_time(nodes_used);
        }

        // Find best thread for result selection
        let options_handle = Arc::clone(&self.shared_state.options);
        let options = options_handle.read().unwrap();
        let skill = Skill::new(
            options
                .get("Skill Level")
                .and_then(|o| o.get_int())
                .unwrap_or(20),
            if options
                .get("UCI_LimitStrength")
                .map(|o| o.is_true())
                .unwrap_or(false)
            {
                options
                    .get("UCI_Elo")
                    .and_then(|o| o.get_int())
                    .unwrap_or(0)
            } else {
                0
            },
        );

        // Released here: the PV output below reads the options again.
        drop(options);
        let mut best_thread_used = false;

        // Select best thread when appropriate conditions are met
        if self.limits.depth == 0 && !skill.enabled() && !self.root_moves.is_empty() {
            // Get the best thread based on score/depth comparison
            let threads_handle = Arc::clone(&self.shared_state.threads);
            let threads_guard = threads_handle.read().unwrap();
            let best_thread = threads_guard.get_best_thread_with_main(self);

            // Access the best thread's worker results if it's not the main thread
            if best_thread.id() != 0 {
                if let Some(ref worker) = *best_thread.worker().lock().unwrap() {
                    // Copy best thread's search results to main thread
                    if !worker.root_moves.is_empty() {
                        self.adopt_thread_result(worker);
                        best_thread_used = true;
                        // The PV is re-sent below (C++ search.cpp handles the
                        // `bestThread != this` case together with `!uciPvSent`).
                    }
                }
            }
        }

        // Update scores for next iteration
        self.main_manager()
            .set_best_previous_score(self.root_moves[0].score);
        self.main_manager()
            .set_best_previous_average_score(self.root_moves[0].average_score);

        // C++ search.cpp: a single-move PV that gets a ponder move from the
        // TT changes the printed PV, so the last PV output is no longer up to date.
        let tt = self.tt();
        if !self.root_moves.is_empty()
            && self.root_moves[0].pv.len() == 1
            && self.root_moves[0].extract_ponder_from_tt(tt, &self.root_pos)
        {
            uci_pv_sent = false;
        }

        // Send PV info if it has changed since the last output in iterative_deepening().
        if !uci_pv_sent || best_thread_used {
            let depth = self.root_depth;
            let extended_scores = self.extend_tb_pvs();
            self.main_manager().pv(
                self,
                &self.shared_state.threads.read().unwrap(),
                self.tt(),
                depth,
                &extended_scores,
            );
        }

        // Extract ponder move
        // In rare cases, output_pv() may change the ponder move through syzygy_extend_pv()
        let mut ponder = String::new();
        if !self.root_moves.is_empty() && self.root_moves[0].pv.len() > 1 {
            ponder = UciEngine::move_str(self.root_moves[0].pv[1], self.root_pos.is_chess960());
        }

        let bestmove = UciEngine::move_str(self.root_moves[0].pv[0], self.root_pos.is_chess960());
        // Send best move through UCI callback
        self.main_manager().call_on_bestmove(&bestmove, &ponder);
    }

    /// Adopt the selected worker's result, including the depth used by final PV output.
    pub(crate) fn adopt_thread_result(&mut self, selected: &Worker) {
        self.root_moves = selected.root_moves.clone();
        self.completed_depth = selected.completed_depth;
        self.root_depth = selected.root_depth;
    }

    /// Correct and extend the PVs of the root moves about to be printed.
    ///
    /// Port of the `syzygy_extend_pv` call inside C++ `output_pv`
    /// (search.cpp), which runs for every printed root move on every
    /// PV emission, so this is called immediately before each `pv` call. The
    /// root does not have to be in the tablebases: a decisive score from
    /// in-search probes is extended as well.
    ///
    /// Not cosmetic: C++ mutates `rootMoves[i].pv` in place and that PV becomes
    /// `previousPV` for the next iteration, which drives `followPV` move
    /// ordering. A missing extension changes what the next iteration searches
    /// first, and with it the move finally chosen in a tablebase-decided
    /// position.
    ///
    /// Returns, per printed root move, the score left by `syzygy_extend_pv`
    /// when it ran: in exceptional cases the extension corrects the score (a
    /// draw by the 50-move rule found along the extended PV), and C++ prints
    /// that corrected value without storing it in the root move.
    fn extend_tb_pvs(&mut self) -> Vec<Option<Value>> {
        // C++ `multiPV = std::min(usize(worker.options["MultiPV"]), rootMoves.size())`.
        let multi_pv = self
            .shared_state
            .options
            .read()
            .unwrap()
            .get("MultiPV")
            .and_then(|o| o.get_int())
            .unwrap_or(1) as usize;
        let multi_pv = multi_pv.min(self.root_moves.len());
        let mut extended_scores = vec![None; multi_pv];

        for i in 0..multi_pv {
            // C++ skips moves that fall back on the previous iteration's score.
            if self.root_moves[i].score == -VALUE_INFINITE {
                continue;
            }

            let mut v = self.root_moves[i].uci_score;
            let is_tb_score = self.tb_config.root_in_tb && !crate::types::is_mate_or_mated(v);
            if is_tb_score {
                v = self.root_moves[i].tb_score;
            }

            if is_decisive(v)
                && !crate::types::is_mate_or_mated(v)
                && (!self.root_moves[i].is_inexact() || is_tb_score)
            {
                let options = self.shared_state.options.read().unwrap();
                syzygy_extend_pv(
                    &options,
                    &self.limits,
                    &self.root_pos,
                    &mut self.root_moves[i],
                    &mut v,
                    multi_pv,
                );
                extended_scores[i] = Some(v);
            }
        }

        extended_scores
    }

    /// MultiPV bookkeeping for an aborted search of PV line `pv_idx` (> 0).
    ///
    /// In multiPV analysis we do not let aborted searches spoil mated-in/TB loss
    /// scores from a completed search in an earlier PV line. Hence we guard against
    /// an aborted `pv_idx` line overtaking `pv_idx - 1` when `pv_idx - 1` is a proven
    /// loss. Moreover, we do not trust an exact loss score from an aborted search.
    pub fn guard_aborted_multi_pv_line(&mut self, pv_idx: usize, multi_pv: usize) {
        let prev_score = self.root_moves[pv_idx - 1].score;

        if (is_loss(prev_score) && self.root_moves[pv_idx] < self.root_moves[pv_idx - 1])
            || self.root_moves[pv_idx].is_exact_loss()
        {
            let rm = &mut self.root_moves[pv_idx];

            // If previousScore is exact and worse than pvIdx - 1, we can safely use it.
            // If it is equal, we make sure it cannot overtake pvIdx - 1.
            if rm.previous_score != -VALUE_INFINITE
                && rm.previous_score_exact
                && rm.previous_score <= prev_score
            {
                rm.score = rm.previous_score;
                rm.uci_score = rm.previous_score;
                rm.previous_score = -VALUE_INFINITE;
                rm.pv = rm.previous_pv.clone();
                rm.unset_inexact();
            }
            // Otherwise, if we can, we cap the score to the best possible, and mark
            // the score as inexact (also a valid excuse for the incomplete PV).
            else {
                if is_loss(prev_score) {
                    rm.score = prev_score;
                    rm.uci_score = prev_score;
                    rm.previous_score = -VALUE_INFINITE;
                    rm.pv.truncate(1);
                    rm.inexact_upper = true;
                } else {
                    rm.inexact_upper = false;
                }

                rm.inexact_lower = !rm.inexact_upper;
            }
        }

        // Finally, we mark all loss scores from partially searched moves as inexact.
        for rm in self.root_moves.iter_mut().take(multi_pv).skip(pv_idx + 1) {
            if rm.is_exact_loss() {
                rm.inexact_lower = true;
            }
        }
    }

    /// Main iterative deepening loop. It calls search() repeatedly with increasing
    /// depth until the allocated thinking time has been consumed, the user stops
    /// the search, or the maximum search depth is reached.
    /// Returns whether the last UCI PV line that was printed is up to date
    /// (C++ `bool Search::Worker::iterative_deepening()`, search.cpp).
    pub fn iterative_deepening(&mut self) -> bool {
        self.with_tt_locked(Self::iterative_deepening_locked)
    }

    fn iterative_deepening_locked(&mut self) -> bool {
        let is_main = self.is_main_thread();

        let mut pv = vec![Move::none(); MAX_PLY as usize + 1];
        let mut last_best_move_depth = 0;
        let mut last_best_score = -VALUE_INFINITE;
        let mut last_best_pv: Vec<Move> = Vec::new();

        let mut alpha: Value;
        let mut beta: Value;
        let mut best_value: Value;
        let mut time_reduction = 1.0;
        let mut tot_best_move_changes = 0.0;
        let mut delta;
        let mut iter_idx = 0;
        // C++ `uciPvSent` (search.cpp): true while the last PV printed by the
        // main thread still describes the current best line.
        let mut uci_pv_sent = false;
        // Nodes searched when the first PV line of the iteration completed; the effort spent
        // on the best move is measured against this rather than the MultiPV total.
        let mut total_effort: u64 = 0;

        // Allocate stack with extra size
        // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
        // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
        // (ss + 2) is needed for initialization of cutOffCnt.
        let mut stack = vec![Stack::default(); MAX_PLY as usize + 10];
        let ss = &mut stack[7] as *mut Stack;

        // Initialize stack sentinels
        // SAFETY: `stack` has MAX_PLY + 10 entries and `ss` is index 7, so ss-7..=ss+MAX_PLY+2 (indices 0..=MAX_PLY+9) are
        // in bounds, and `stack` outlives this function's uses; the history pointers stored refer to tables owned by
        // the shared histories / this worker, which outlive the stack.
        unsafe {
            for i in 1..=7 {
                (*ss.offset(-i)).continuation_history =
                    &self.continuation_history()[0][0][NO_PIECE as usize][0] as *const _;
                (*ss.offset(-i)).continuation_correction_history =
                    &self.continuation_correction_history[NO_PIECE as usize][0];
                (*ss.offset(-i)).static_eval = VALUE_NONE;
            }

            for i in 0..=(MAX_PLY + 2) {
                (*ss.offset(i as isize)).ply = i;
            }

            (*ss).pv = pv.as_mut_ptr();
        }

        // Initialize iteration values
        if is_main {
            let mgr = self.main_manager();
            if mgr.get_best_previous_score() == VALUE_INFINITE {
                mgr.fill_iter_value(VALUE_ZERO);
            } else {
                let score = mgr.get_best_previous_score();
                mgr.fill_iter_value(score);
            }
        }

        let (multi_pv, mut skill) = {
            let options = self.shared_state.options.read().unwrap();
            let multi_pv = options
                .get("MultiPV")
                .and_then(|o| o.get_int())
                .unwrap_or(1) as usize;
            let skill = Skill::new(
                options
                    .get("Skill Level")
                    .and_then(|o| o.get_int())
                    .unwrap_or(20), // Default to 20 (full strength) matching C++
                if options
                    .get("UCI_LimitStrength")
                    .map(|o| o.is_true())
                    .unwrap_or(false)
                {
                    options
                        .get("UCI_Elo")
                        .and_then(|o| o.get_int())
                        .unwrap_or(0)
                } else {
                    0
                },
            );

            // When playing with strength handicap enable MultiPV search that we will use behind-the-scenes
            // to retrieve a set of possible moves.
            let multi_pv = if skill.enabled() {
                max(multi_pv, 4)
            } else {
                multi_pv
            };
            let multi_pv = min(multi_pv, self.root_moves.len());
            (multi_pv, skill)
        };

        let mut search_again_counter = 0;
        let mut fail_high_recovery = 0;

        // Initialize low ply history
        for ply in &mut *self.low_ply_history {
            for entry in ply {
                entry.set(102);
            }
        }

        // Dampen main history with every new root position
        for color in 0..2 {
            // SAFETY: color is 0 or 1 and main_history has 2 rows.
            let hist = unsafe { self.main_history.get_unchecked_mut(color) };
            for i in 0..hist.len() {
                // SAFETY: i < hist.len() by the loop bound.
                let entry = unsafe { hist.get_unchecked_mut(i) };
                let val = entry.get() as i32;
                entry.set((val * 729 / 1024) as i16);
            }
        }

        // Iterative deepening loop until requested to stop or the target depth is reached
        self.root_depth = 0;

        // Iterative deepening loop - matches C++ search.cpp
        // C++: while (++rootDepth < MAX_PLY && !threads.stop && !(limits.depth && mainThread && rootDepth > limits.depth))
        while {
            // Lock-free stop check
            let stop = self.should_stop();
            let depth_limit = self.limits.depth > 0
                && self.is_main_thread()
                && self.root_depth >= self.limits.depth;
            self.root_depth + 1 < MAX_PLY && !stop && !depth_limit
        } {
            self.root_depth += 1;

            // Do not tune these values. They are not intended for playing strength.
            let seek_mate_threshold = 750 + 220000 / (self.root_depth * self.root_depth);

            // Signal the start of a new iteration
            if self.is_main_thread() {
                uci_pv_sent = false;
            }

            // Save the last iteration's scores before the first PV line is searched and
            // all the move scores except the (new) PV are set to -VALUE_INFINITE.
            // We also snapshot each root move's PV so that nodes still following the
            // principal variation can be detected (followPV).
            for (i, rm) in self.root_moves.iter_mut().enumerate() {
                rm.previous_score = rm.score;
                rm.previous_pv = rm.pv.clone();
                rm.previous_score_exact = i < multi_pv;
            }

            let mut pv_first = 0;
            self.pv_last = 0;

            // Lock-free access to search control flags (hot path optimization)
            let increase_depth = self.should_increase_depth();

            if !increase_depth {
                search_again_counter += 1;
            }

            // MultiPV loop
            // MultiPV loop: we perform a full root search for each PV line
            for pv_idx in 0..multi_pv {
                self.pv_idx = pv_idx;

                if pv_idx == self.pv_last {
                    pv_first = self.pv_last;
                    self.pv_last += 1;
                    while self.pv_last < self.root_moves.len() {
                        if self.root_moves[self.pv_last].tb_rank
                            != self.root_moves[pv_first].tb_rank
                        {
                            break;
                        }
                        self.pv_last += 1;
                    }
                }

                // PV of the current root move from the previous iteration, used to
                // track nodes still following the principal variation (followPV).
                // Read from the RootMove itself: the list is re-sorted inside the
                // MultiPV loop, so a positional lookup would attribute one move's
                // previous PV to a different move and perturb the node counts.
                self.last_iteration_idx_pv = self.root_moves[pv_idx].previous_pv.clone();

                // Reset UCI info selDepth
                // Reset UCI info selDepth for each depth and each PV line
                self.sel_depth = 0;

                // Reset aspiration window
                delta = 5
                    + (self.thread_idx % 8) as i32
                    + self.root_moves[pv_idx].mean_squared_score.abs() / 10193;
                let avg = self.root_moves[pv_idx].average_score;
                alpha = max(avg - delta, -VALUE_INFINITE);
                beta = min(avg + delta, VALUE_INFINITE);

                self.seek_mate = self.root_moves[pv_idx].previous_score != -VALUE_INFINITE
                    && self.root_moves[pv_idx].previous_score.abs() >= seek_mate_threshold;

                // Aspiration window search loop
                // Start with a small aspiration window and, in the case of a fail
                // high/low, enlarge the window progressively.
                let mut failed_high_cnt = 0;
                if pv_idx == 0 {
                    fail_high_recovery = max(0, fail_high_recovery - 2);
                }
                loop {
                    // Adjust the effective depth searched, but ensure at least one effective increment for every
                    // four searchAgain steps (see issue #2717).
                    let adjusted_depth = max(
                        1,
                        self.root_depth
                            - failed_high_cnt
                            - fail_high_recovery
                            - 3 * (search_again_counter + 1) / 4,
                    );

                    // Move the position out so the search receives disjoint
                    // mutable borrows of the worker and the board.
                    let mut root_position = std::mem::take(&mut self.root_pos);
                    best_value = self.search::<Root>(
                        &mut root_position,
                        ss,
                        alpha,
                        beta,
                        adjusted_depth,
                        false,
                    );
                    self.root_pos = root_position;

                    // Sort moves
                    // Bring the best move to the front. It is critical that sorting
                    // is done with a stable algorithm because all the values but the
                    // first and eventually the new best one is set to -VALUE_INFINITE
                    // and we want to keep the same order for all the moves except the
                    // new PV that goes to the front. Note that in the case of MultiPV
                    // search the already searched PV lines are preserved.
                    self.root_moves[pv_idx..self.pv_last].sort();

                    // If search has been stopped, we break immediately. Sorting is safe because RootMoves is
                    // still valid, although it refers to the previous iteration.
                    if self.should_stop() {
                        break;
                    }

                    // Report PV for failing high/low
                    // When failing high/low give some update before a re-search. To avoid
                    // excessive output that could hang GUIs like Fritz 19, only start
                    // at nodes > 10M (matches C++ search.cpp)
                    if self.is_main_thread()
                        && multi_pv == 1
                        && (best_value <= alpha || best_value >= beta)
                        && self.counters.nodes.load(Ordering::Relaxed) > NODES_LIMIT_OUTPUT
                    {
                        let extended_scores = self.extend_tb_pvs();
                        self.shared_state.threads.read().unwrap().main_manager_pv(
                            self,
                            self.tt(),
                            self.root_depth,
                            &extended_scores,
                        );
                    }

                    // Adjust aspiration window
                    // In case of failing low/high increase aspiration window and re-search, otherwise exit the loop.
                    if best_value <= alpha {
                        beta = alpha;
                        alpha = max(best_value - delta, -VALUE_INFINITE);
                        failed_high_cnt = 0;
                        if is_main {
                            let mgr = self.main_manager();
                            mgr.set_stop_on_ponderhit(false);
                        }
                    } else if best_value >= beta {
                        // Fail high: move alpha up toward beta and widen beta around best_value
                        alpha = max(beta - delta, alpha);
                        beta = min(best_value + delta, VALUE_INFINITE);
                        failed_high_cnt += 1;
                    } else {
                        break;
                    }

                    delta += 47 * delta / 128;
                }

                // Gradually increase depth after reduced depth search
                if failed_high_cnt > 0 && pv_idx == 0 {
                    fail_high_recovery = (failed_high_cnt + 1) / 2 + 2;
                }

                if self.should_stop() && pv_idx > 0 {
                    self.guard_aborted_multi_pv_line(pv_idx, multi_pv);
                }

                // Sort PV lines
                // Sort the PV lines searched so far and update the GUI
                self.root_moves[pv_first..=pv_idx].sort();

                // Report PV. Matches C++: send PV info if not stopped, and either this is
                // the last PV line or we have searched a large number of nodes.
                if self.is_main_thread()
                    && !self.should_stop()
                    && (pv_idx + 1 == multi_pv
                        || self.counters.nodes.load(Ordering::Relaxed) > NODES_LIMIT_OUTPUT)
                {
                    let extended_scores = self.extend_tb_pvs();
                    self.shared_state.threads.read().unwrap().main_manager_pv(
                        self,
                        self.tt(),
                        self.root_depth,
                        &extended_scores,
                    );
                    uci_pv_sent = pv_idx + 1 == multi_pv;
                }

                if self.is_main_thread() && pv_idx == 0 {
                    total_effort = self.counters.nodes.load(Ordering::Relaxed);
                }

                if self.should_stop() {
                    break;
                }
            }

            let stop_now = self.should_stop();

            // C++ has no completedDepth (it uses rootDepth), but the port's best-thread
            // selection and time management read completed_depth, so keep it updated
            // when the search was not aborted.
            if !stop_now {
                self.completed_depth = self.root_depth;
            }

            // Do not replace (shorter) mate scores from a previous iteration.
            let forgotten_mate = last_best_score != -VALUE_INFINITE
                && is_mate_or_mated(last_best_score)
                && (self.root_moves[0].score.abs() < last_best_score.abs()
                    || self.root_moves[0].is_inexact());

            if !stop_now {
                if last_best_pv.is_empty() || last_best_pv[0] != self.root_moves[0].pv[0] {
                    last_best_move_depth = self.root_depth;
                }

                if !forgotten_mate {
                    last_best_pv = self.root_moves[0].pv.clone();
                    last_best_score = self.root_moves[0].score;
                }
            }

            let aborted_loss_search =
                stop_now && self.pv_idx == 0 && self.root_moves[0].is_exact_loss();

            // An exact mated-in/TB-loss score from an aborted search cannot be trusted:
            // the loss could be delayed or refuted upon exploring the remaining root-moves.
            // Thus here we roll back to the score from the previous iteration. We do the
            // same if a search has failed to recover a mate score from a previous iteration.
            if aborted_loss_search
                || (self.root_moves[0].score != -VALUE_INFINITE && forgotten_mate)
            {
                if !last_best_pv.is_empty() {
                    // Bring the last best move to the front for best thread selection.
                    if let Some(idx) = self
                        .root_moves
                        .iter()
                        .position(|rm| rm.pv[0] == last_best_pv[0])
                    {
                        self.root_moves.swap(0, idx);
                    }
                    self.root_moves[0].score = last_best_score;
                    self.root_moves[0].uci_score = last_best_score;
                    self.root_moves[0].pv = last_best_pv.clone();
                    self.root_moves[0].unset_inexact();
                }
                // For an aborted d1 search we label the loss score as a lower bound.
                else if aborted_loss_search {
                    self.root_moves[0].inexact_lower = true;
                }
            }

            // Have we found a "mate in x" after a completed iteration?
            if self.limits.mate > 0
                && !stop_now
                && is_mate_or_mated(self.root_moves[0].score)
                && VALUE_MATE - self.root_moves[0].score.abs() <= self.limits.mate.saturating_mul(2)
            {
                // Use the mirroring setter rather than taking a WRITE guard on the shared
                // thread pool: a pending writer is what turns the per-node read guards
                // into a deadlock (see should_stop() above).
                self.set_stop(true);
            }

            if !self.is_main_thread() {
                continue;
            }

            // Skill level move selection
            // If the skill level is enabled and time is up, pick a sub-optimal best move
            if skill.enabled() && skill.time_to_pick(self.root_depth) {
                skill.pick_best(&self.root_moves, multi_pv);
            }

            // Time management
            // Do we have time for the next iteration? Can we stop searching now?
            if self.limits.use_time_management()
                && !self.should_stop()
                && !self.main_manager().get_stop_on_ponderhit()
            {
                let nodes_effort = self.root_moves[0].effort * 100000 / max(1, total_effort);

                let main_mgr = self.main_manager();
                let best_score = self.root_moves[0].score;
                let falling_eval = (11.48
                    + 2.30 * (main_mgr.get_best_previous_average_score() - best_score) as f64
                    + 1.1 * (main_mgr.get_iter_value(iter_idx) - best_score) as f64)
                    / 100.0;
                let falling_eval = falling_eval.clamp(0.576, 1.728);

                // If the bestMove is stable over several iterations, reduce time accordingly
                time_reduction = interpolate(
                    (self.root_depth - last_best_move_depth) as f64,
                    4.96,
                    18.79,
                    0.639,
                    1.712,
                )
                .clamp(0.629, 1.544);
                let reduction =
                    (1.468 + main_mgr.get_previous_time_reduction()) / (2.284 * time_reduction);
                // Age out and update PV variability metric
                tot_best_move_changes /= 2.0;
                // Scope the lock to ensure it's dropped before any potential write lock
                let thread_count = {
                    let threads = self.shared_state.threads.read().unwrap();
                    // C++ sums `bestMoveChanges` over every worker and resets them.
                    tot_best_move_changes += threads.take_best_move_changes() as f64;
                    threads.size() as f64
                }; // read lock is dropped here

                let best_move_instability = 1.077 + 2.229 * tot_best_move_changes / thread_count;
                let high_best_move_effort =
                    interpolate(nodes_effort as f64, 75800.0, 104510.0, 0.969, 0.714)
                        .clamp(0.693, 0.838);

                let total_time = main_mgr.get_tm_optimum() as f64
                    * falling_eval
                    * reduction
                    * best_move_instability
                    * high_best_move_effort;

                // Cap used time to 0.5s for a better viewer experience
                let total_time = if self.root_moves.len() == 1 {
                    total_time.min(500.0 * self.limits.npmsec.max(1) as f64)
                } else {
                    total_time
                };

                let elapsed_time = self.elapsed() as f64;

                // Stop the search if we have exceeded totalTime or maximum time, or if we
                // know that there are no better moves in the analysed line(s).
                if elapsed_time > total_time.min(main_mgr.get_tm_maximum() as f64)
                    || self.root_moves[multi_pv - 1].score >= mate_in(3)
                    || self.root_moves[0].score == mated_in(2)
                {
                    // If we are allowed to ponder do not stop the search now but keep pondering until the GUI sends "ponderhit" or "stop".
                    if main_mgr.get_ponder() {
                        main_mgr.set_stop_on_ponderhit(true);
                    } else {
                        self.set_stop(true);
                    }
                } else {
                    let increase_depth_value =
                        main_mgr.get_ponder() || elapsed_time <= total_time * 0.50;
                    self.search_control.set_increase_depth(increase_depth_value);
                }
            }

            if is_main {
                let mgr = self.main_manager();
                mgr.set_iter_value(iter_idx, self.root_moves[0].score);
            }
            iter_idx = (iter_idx + 1) & 3;
        }

        if !self.is_main_thread() {
            return uci_pv_sent;
        }

        if is_main {
            let mgr = self.main_manager();
            mgr.set_previous_time_reduction(time_reduction);
        }

        // Apply skill level
        // If the skill level is enabled, swap the best PV line with the sub-optimal one
        if skill.enabled() {
            let best_move = if skill.best != Move::none() {
                skill.best
            } else {
                skill.pick_best(&self.root_moves, multi_pv)
            };

            if best_move != self.root_moves[0].pv[0] {
                let best_score = self.root_moves[0].score;
                if let Some(idx) = self.root_moves.iter().position(|rm| rm.pv[0] == best_move) {
                    self.root_moves.swap(0, idx);
                }
                if is_loss(self.root_moves[0].score) && self.root_moves[0].score < best_score {
                    self.root_moves[0].inexact_lower = true;
                }
                uci_pv_sent = false;
            }
        }

        uci_pv_sent
    }

    /// Main search function for both PV and non-PV nodes
    fn search<NT: NodeType>(
        &mut self,
        pos: &mut Position,
        ss: *mut Stack,
        mut alpha: Value,
        mut beta: Value,
        mut depth: Depth,
        cut_node: bool,
    ) -> Value {
        let pv_node = NT::IS_PV;
        let root_node = NT::IS_ROOT;
        let all_node = !(pv_node || cut_node);
        let seek_mate = self.seek_mate;

        // Quiescence search at depth 0
        // Dive into quiescence search when the depth reaches zero
        if depth <= 0 {
            let result = if pv_node {
                self.qsearch::<PV>(pos, ss, alpha, beta)
            } else {
                self.qsearch::<NonPV>(pos, ss, alpha, beta)
            };
            return result;
        }

        // Limit depth
        // Limit the depth if extensions made it too large
        depth = min(depth, MAX_PLY - 1);

        // Nodes are counted in do_move(), as in C++.

        // Check for upcoming repetition
        // SAFETY: `ss` is this node's Stack entry (see the allocation in iterative_deepening; ply <= MAX_PLY keeps it in bounds).
        unsafe {
            // Check if we have an upcoming move that draws by repetition
            if !root_node && alpha < VALUE_DRAW && pos.upcoming_repetition((*ss).ply) {
                alpha = value_draw(self.counters.nodes.load(Ordering::Relaxed));
                if alpha >= beta {
                    return alpha;
                }
            }
        }

        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || (alpha == beta - 1));
        debug_assert!(0 < depth && depth < MAX_PLY);
        debug_assert!(!(pv_node && cut_node));

        // The PV buffer is deliberately uninitialised (as in C++), so no ~500-byte fill per node.
        // Invariant: only the child's PV terminator protocol reads it: slot 0 is set to
        // `Move::none()` here, and every later read (`update_pv`, root PV extraction) stops at the
        // first `Move::none()` that `update_pv` always writes after the copied moves.
        let mut pv = [core::mem::MaybeUninit::<Move>::uninit(); MAX_PLY as usize + 1];
        pv[0] = core::mem::MaybeUninit::new(Move::none());
        let pv_ptr = pv.as_mut_ptr() as *mut Move;
        let pos_key: Key;
        let mut move_: Move;
        let excluded_move: Move;
        let mut best_move: Move;
        let mut extension: Depth;
        let mut new_depth: Depth;
        let mut best_value: Value;
        let mut value: Value;
        let mut eval: Value;
        let mut max_value: Value;
        let mut prob_cut_beta: Value;
        let mut gives_check: bool;
        let mut improving: bool;
        let prior_capture: bool;
        let opponent_worsening: bool;
        let mut capture: bool;
        let tt_capture: bool;
        let prior_reduction: i32;
        let mut moved_piece: Piece;

        let mut captures_searched = SearchedList::new();
        let mut quiets_searched = SearchedList::new();

        // SAFETY: `ss` is this node's entry of the Stack array allocated in iterative_deepening (ply 0 at index 7, length
        // MAX_PLY + 10, ply <= MAX_PLY because nodes at ply >= MAX_PLY return before recursing), so ss-7..=ss+2 are in
        // bounds; entries below ss were set by the ancestors (or are sentinels), and only this node touches ss/ss+1/ss+2
        // until it returns. Other raw accesses in this block (pv buffers, histories) rely on the invariants noted at each use.
        unsafe {
            // Step 1. Initialize node
            (*ss).in_check = pos.checkers() != 0;
            prior_capture = pos.captured_piece() != NO_PIECE;
            let us = pos.side_to_move();
            (*ss).move_count = 0;
            best_value = -VALUE_INFINITE;
            max_value = VALUE_INFINITE;

            // Determine whether this node is still following the principal variation
            // from the previous iteration (used to gate IIR and some pruning).
            (*ss).follow_pv = root_node
                || ((*ss.offset(-1)).follow_pv
                    && ((*ss).ply as usize - 1) < self.last_iteration_idx_pv.len()
                    && (*ss.offset(-1)).current_move
                        == self.last_iteration_idx_pv[(*ss).ply as usize - 1]);

            // Check for time
            if self.is_main_thread() {
                self.main_manager().check_time(self);
            }

            // Update selective depth
            // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
            if pv_node && self.sel_depth < (*ss).ply + 1 {
                self.sel_depth = (*ss).ply + 1;
            }

            // Step 2. Check for aborted search or immediate draw (not at the root, like C++)
            if !root_node && self.should_stop() {
                return value_draw(self.counters.nodes.load(Ordering::Relaxed));
            }

            if !root_node {
                if pos.is_draw((*ss).ply) || (*ss).ply >= MAX_PLY {
                    return if (*ss).ply >= MAX_PLY && !(*ss).in_check {
                        self.evaluate(pos)
                    } else {
                        value_draw(self.counters.nodes.load(Ordering::Relaxed))
                    };
                }

                // Step 3. Mate distance pruning. Even if we mate at the next move our score
                // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
                // a shorter mate was found upward in the tree then there is no need to search
                // because we will never beat the current alpha. Equal and opposite logic applies
                // when being mated. In either case, return a fail-high score.
                alpha = max(mated_in((*ss).ply), alpha);
                beta = min(mate_in((*ss).ply + 1), beta);
                if alpha >= beta {
                    return alpha;
                }
            }

            debug_assert!(0 <= (*ss).ply && (*ss).ply < MAX_PLY);

            let prev_sq = if (*ss.offset(-1)).current_move.is_ok() {
                Some((*ss.offset(-1)).current_move.to_sq())
            } else {
                None
            };

            best_move = Move::none();
            prior_reduction = (*ss.offset(-1)).reduction;
            (*ss.offset(-1)).reduction = 0;
            (*ss).stat_score = 0;
            (*ss.offset(2)).cutoff_cnt = 0;
            (*ss.offset(1)).prior_nmp_fail_high = 0;

            // Step 4. Transposition table lookup
            excluded_move = (*ss).excluded_move;
            pos_key = pos.key();
            let (tt_hit, mut tt_data, tt_writer) = self.tt().probe_for_search(pos_key);

            (*ss).tt_hit = tt_hit;

            tt_data.mv = if root_node {
                self.root_moves[self.pv_idx].pv[0]
            } else if tt_hit {
                tt_data.mv
            } else {
                Move::none()
            };
            tt_data.value = if tt_hit {
                value_from_tt(tt_data.value, (*ss).ply, pos.rule50_count())
            } else {
                VALUE_NONE
            };
            (*ss).tt_pv = if excluded_move != Move::none() {
                (*ss).tt_pv
            } else {
                pv_node || (tt_hit && tt_data.is_pv)
            };
            tt_capture = tt_data.mv.is_ok() && pos.capture_stage(tt_data.mv);

            // Step 5. Static evaluation of the position
            let mut unadjusted_static_eval = VALUE_NONE;
            let correction_value = self.correction_value(pos, ss);

            if (*ss).in_check {
                // Skip early pruning when in check
                (*ss).static_eval = if (*ss).ply >= 2 {
                    (*ss.offset(-2)).static_eval
                } else {
                    VALUE_NONE
                };
                eval = (*ss).static_eval;
            } else if excluded_move != Move::none() {
                unadjusted_static_eval = (*ss).static_eval;
                eval = (*ss).static_eval;
            } else if (*ss).tt_hit {
                // Never assume anything about values stored in TT
                unadjusted_static_eval = tt_data.eval;
                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.evaluate(pos);
                }

                (*ss).static_eval =
                    to_corrected_static_eval(unadjusted_static_eval, correction_value);
                eval = (*ss).static_eval;

                // ttValue can be used as a better position evaluation
                if is_valid(tt_data.value)
                    && (tt_data.bound
                        & if tt_data.value > eval {
                            Bound::LOWER
                        } else {
                            Bound::UPPER
                        })
                        != Bound::NONE
                {
                    eval = tt_data.value;
                }
            } else {
                unadjusted_static_eval = self.evaluate(pos);
                (*ss).static_eval =
                    to_corrected_static_eval(unadjusted_static_eval, correction_value);
                eval = (*ss).static_eval;

                // Static evaluation is saved as it was before adjustment by correction history
                tt_writer.write(
                    pos_key,
                    VALUE_NONE,
                    (*ss).tt_pv,
                    Bound::NONE,
                    DEPTH_UNSEARCHED,
                    Move::none(),
                    unadjusted_static_eval,
                    self.tt().generation(),
                );
            }

            // Set up the improving flag, which is true if current static evaluation is
            // bigger than the previous static evaluation at our turn (if we were in
            // check at our previous move we go back until we weren't in check) and is
            // false otherwise. The improving flag is used in various pruning heuristics.
            // Similarly, opponentWorsening is true if our static evaluation is better
            // for us than at the last ply.
            improving = (*ss).static_eval > (*ss.offset(-2)).static_eval;
            opponent_worsening = (*ss).static_eval > -(*ss.offset(-1)).static_eval;

            // Hindsight adjustment of reductions based on static evaluation difference
            if prior_reduction >= 3 && !opponent_worsening {
                depth += 1;
            }
            if prior_reduction >= 2
                && depth >= 2
                && (*ss).static_eval + (*ss.offset(-1)).static_eval > 166
            {
                depth -= 1;
            }

            // Step 6. At non-PV nodes we check for an early TT cutoff. Note that we
            //         always check the validity of the TT value because of access races.
            if !pv_node
                && excluded_move == Move::none()
                && is_valid(tt_data.value)
                && tt_data.depth > depth - (tt_data.value <= beta) as i32
            {
                // Case A: TT entry can produce a cutoff
                if (tt_data.bound
                    & if tt_data.value >= beta {
                        Bound::LOWER
                    } else {
                        Bound::UPPER
                    })
                    != Bound::NONE
                    && (cut_node == (tt_data.value >= beta) || depth > 4)
                {
                    // If the ttMove is quiet, update move sorting heuristics on TT hit
                    if tt_data.mv.is_ok() && tt_data.value >= beta {
                        // Bonus for a quiet ttMove that fails high
                        if !tt_capture {
                            self.update_quiet_histories(pos, ss, tt_data.mv, 131 * depth);
                        }

                        // Extra penalty for early quiet moves of the previous ply
                        if let Some(prev_sq) = prev_sq
                            && (*ss.offset(-1)).move_count < 5
                            && !prior_capture
                        {
                            self.update_continuation_histories(
                                ss.offset(-1),
                                pos.piece_on(prev_sq),
                                prev_sq,
                                -2210,
                            );
                        }
                    }

                    // Partial workaround for the graph history interaction problem.
                    // For high rule50 counts don't produce transposition table cutoffs.
                    if pos.rule50_count() < 96 {
                        if depth >= 7
                            && tt_data.mv.is_ok()
                            && pos.legal(tt_data.mv)
                            && !is_decisive(tt_data.value)
                        {
                            pos.do_move(tt_data.mv);
                            let next_key = pos.key();
                            let (_next_hit, next_data, _) = self.tt().probe_for_search(next_key);
                            pos.undo_move(tt_data.mv);

                            // Check that the ttValue after the tt move would also trigger a cutoff
                            if !is_valid(next_data.value) {
                                return tt_data.value;
                            }
                            if (tt_data.value >= beta) == (-next_data.value >= beta) {
                                return tt_data.value;
                            }
                        } else {
                            return tt_data.value;
                        }
                    }
                }
                // Case B: No cutoff, but depth was sufficient, so the bound points the wrong way.
                else if depth > 5 {
                    debug_assert!(
                        tt_data.bound
                            & if tt_data.value >= beta {
                                Bound::LOWER
                            } else {
                                Bound::UPPER
                            }
                            == Bound::NONE
                    );

                    // If such a mismatch is the only reason cutoff failed, the TT entry
                    // is now useless.
                    // (*Scaler) This is very close to secondary aging, a known antiscaler. The depth
                    // and window conditions seem to be key to decent scaling.
                    tt_writer.penalize(1);
                }
            }

            // Step 7. Tablebases probe
            if !root_node && excluded_move == Move::none() && self.tb_config.cardinality > 0 {
                let pieces_count = (pos.pieces_of_color(Color::White)
                    | pos.pieces_of_color(Color::Black))
                .count_ones() as i32;

                if pieces_count <= self.tb_config.cardinality as i32
                    && (pieces_count < self.tb_config.cardinality as i32
                        || depth >= self.tb_config.probe_depth)
                    && pos.rule50_count() == 0
                    && pos.castling_rights(Color::White) == CastlingRights::empty()
                    && pos.castling_rights(Color::Black) == CastlingRights::empty()
                {
                    let mut err = ProbeState::Fail;
                    let wdl = self.tb_config.tables.probe_wdl(pos, &mut err);

                    // Force check of time on next occasion
                    if self.is_main_thread() {
                        // Reset call counter to ensure time check on next occasion
                        self.main_manager().reset_calls_cnt();
                    }

                    if err != ProbeState::Fail {
                        self.counters.tb_hits.fetch_add(1, Ordering::Relaxed);

                        let draw_score = if self.tb_config.use_rule50 { 1 } else { 0 };

                        // Convert WDL to score
                        let draw_wdl_pos = if draw_score == 0 {
                            WDLScore::WDLDraw
                        } else {
                            WDLScore::WDLCursedWin
                        };
                        let draw_wdl_neg = if draw_score == 0 {
                            WDLScore::WDLDraw
                        } else {
                            WDLScore::WDLBlessedLoss
                        };

                        // C++ search.cpp: Value tbValue = VALUE_TB - ss->ply;
                        let tb_base = VALUE_TB - (*ss).ply;
                        // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to score
                        let tb_value = if wdl < draw_wdl_neg {
                            -tb_base
                        } else if wdl > draw_wdl_pos {
                            tb_base
                        } else {
                            VALUE_DRAW + 2 * wdl as i32 * draw_score
                        };

                        let tb_bound = if wdl < draw_wdl_neg {
                            Bound::UPPER
                        } else if wdl > draw_wdl_pos {
                            Bound::LOWER
                        } else {
                            Bound::EXACT
                        };

                        // The TB bound is conclusive for this window: store it and return.
                        if tb_bound == Bound::EXACT
                            || (tb_bound == Bound::LOWER && tb_value >= beta)
                            || (tb_bound == Bound::UPPER && tb_value <= alpha)
                        {
                            // C++ uses the TTWriter returned by the earlier probe so
                            // that the cluster replacement policy is respected; writing
                            // to the cluster's first entry would always clobber slot 0.
                            tt_writer.write(
                                pos_key,
                                value_to_tt(tb_value, (*ss).ply),
                                (*ss).tt_pv,
                                tb_bound,
                                min(MAX_PLY - 1, depth + 6),
                                Move::none(),
                                VALUE_NONE,
                                self.tt().generation(),
                            );

                            return tb_value;
                        }

                        if pv_node {
                            if tb_bound == Bound::LOWER {
                                best_value = tb_value;
                                alpha = max(alpha, best_value);
                            } else {
                                max_value = tb_value;
                            }
                        }
                    }
                }
            }

            // Upstream jumps directly to the move loop for checked nodes. Keep
            // the pruning stages under one branch so the common unchecked path
            // does not carry a separate in-check test at every stage.
            if !(*ss).in_check {
                // Use static evaluation difference to improve quiet move ordering
                // Skip when in check (C++ skips via goto moves_loop)
                if let Some(prev_sq) = prev_sq
                    && !(*ss.offset(-1)).in_check
                    && !prior_capture
                {
                    let eval_diff =
                        (-((*ss.offset(-1)).static_eval + (*ss).static_eval)).clamp(-189, 194) + 60;
                    // SAFETY: !us is 0 or 1, raw() is u16 (0-65535)
                    {
                        self.main_history
                            .get_unchecked_mut(!us as usize)
                            .get_unchecked_mut((*ss.offset(-1)).current_move.raw() as usize)
                            .update(eval_diff * 11);
                    }

                    if !tt_hit
                        && piece_type(pos.piece_on(prev_sq)) != PieceType::Pawn
                        && (*ss.offset(-1)).current_move.move_type() != MoveType::Promotion
                    {
                        // SAFETY: piece is 0-15, square is 0-63
                        {
                            self.shared_state
                                .shared_histories
                                .pawn_entry(pos)
                                .get_unchecked(pos.piece_on(prev_sq) as usize)
                                .get_unchecked(prev_sq as usize)
                                .update(eval_diff * 13);
                        }
                    }
                }

                // Skip when in check (C++ skips via goto moves_loop)
                // Matches C++ search.cpp: allNode && eval < alpha - 342 * depth && !seekMate
                // Step 8. Razoring
                // If eval is really low, skip search entirely and return the qsearch value
                if all_node && eval < alpha - 342 * depth && !seek_mate {
                    return self.qsearch::<NonPV>(pos, ss, alpha, beta);
                }

                // Step 9. Futility pruning: child node
                // The depth condition is important for mate finding. It should NOT be tuned.
                // Skip when in check (C++ skips via goto moves_loop)
                if !(*ss).tt_pv
                    && eval >= beta
                    && (tt_data.mv == Move::none() || tt_capture)
                    && !is_loss(beta)
                    && !is_win(eval)
                    && depth < if seek_mate { 6 } else { 19 }
                {
                    let mut futility_mult = min(45 + depth * 4, 85);
                    futility_mult -= 20 * !(*ss).tt_hit as i32;

                    let futility_margin = futility_mult * depth
                        - (2789 * improving as i32 + 335 * opponent_worsening as i32)
                            * futility_mult
                            / 1024
                        + correction_value.abs() / 198435;

                    if eval - futility_margin >= beta {
                        return (661 * beta + 363 * eval) / 1024;
                    }
                }

                // Step 10. Null move search with verification search
                if cut_node
                    && (*ss).static_eval + 50 * (*ss).prior_nmp_fail_high
                        >= beta - 13 * depth - 47 * improving as i32 + 365
                    && excluded_move == Move::none()
                    && pos.non_pawn_material_of(us) > 0
                    && (*ss).ply >= self.nmp_min_ply.load(Ordering::Relaxed) as i32
                    && beta >= -2000
                {
                    debug_assert!((*ss.offset(-1)).current_move != Move::null());

                    // Null move dynamic reduction based on depth
                    // Odd threads reduce one ply less
                    let mut reduction =
                        7 + depth / 3 + (((*ss).static_eval - beta) / 256).clamp(0, (*ss).ply / 2)
                            - (self.thread_idx & 1) as i32;
                    reduction -= seek_mate as i32 * reduction / 2;

                    self.do_null_move(pos, ss);

                    value = -self.search::<NonPV>(
                        pos,
                        ss.offset(1),
                        -beta,
                        -beta + 1,
                        depth - reduction,
                        false,
                    );

                    self.undo_null_move(pos);

                    // Do not return unproven mate or TB scores
                    if value >= beta && !is_win(value) {
                        if self.nmp_min_ply.load(Ordering::Relaxed) > 0 || depth < 16 {
                            (*ss).prior_nmp_fail_high += 1;
                            return value;
                        }

                        debug_assert!(self.nmp_min_ply.load(Ordering::Relaxed) == 0);

                        // Do verification search at high depths, with null move pruning
                        // disabled until ply exceeds nmpMinPly.
                        self.nmp_min_ply.store(
                            ((*ss).ply + 3 * (depth - reduction) / 4) as usize,
                            Ordering::Relaxed,
                        );
                        let v =
                            self.search::<NonPV>(pos, ss, beta - 1, beta, depth - reduction, false);
                        self.nmp_min_ply.store(0, Ordering::Relaxed);

                        if v >= beta {
                            (*ss).prior_nmp_fail_high += 1;
                            return value;
                        }
                    }
                }

                // Skip when in check (C++ skips via goto moves_loop)
                improving = improving || (*ss).static_eval >= beta;

                // Skip when in check (C++ skips via goto moves_loop)
                // Step 11. Internal iterative reductions
                // At sufficient depth, reduce depth for PV/Cut nodes without a TTMove.
                // (*Scaler) Making IIR more aggressive scales poorly.
                if !(*ss).follow_pv && !all_node && depth >= 6 && tt_data.mv == Move::none() {
                    depth -= 1;
                }

                // Step 12. ProbCut
                // If we have a good enough capture (or queen promotion) and a reduced search
                // returns a value much above beta, we can (almost) safely prune the previous move.
                prob_cut_beta = beta + 241 - 64 * improving as i32;
                if depth >= 3
                    && !is_decisive(beta)
                    && !(is_valid(tt_data.value) && tt_data.value < prob_cut_beta)
                {
                    debug_assert!(prob_cut_beta < VALUE_INFINITE && prob_cut_beta > beta);

                    let mut mp = MovePicker::new_probcut(
                        pos,
                        tt_data.mv,
                        prob_cut_beta - (*ss).static_eval,
                        self.capture_history.as_ref() as *const _,
                    );
                    let prob_cut_depth = depth - if improving { 5 } else { 3 };

                    loop {
                        let mv = mp.next_move();
                        if mv == Move::none() {
                            break;
                        }
                        debug_assert!(mv.is_ok());

                        if mv == excluded_move {
                            continue;
                        }

                        debug_assert!(pos.capture_stage(mv));

                        let gives_check = pos.gives_check(mv);
                        capture = true;
                        self.do_move(pos, mv, gives_check, capture, ss);

                        value = -self.qsearch::<NonPV>(
                            pos,
                            ss.offset(1),
                            -prob_cut_beta,
                            -prob_cut_beta + 1,
                        );

                        // If the qsearch held, perform the regular search
                        if value >= prob_cut_beta && prob_cut_depth > 0 {
                            value = -self.search::<NonPV>(
                                pos,
                                ss.offset(1),
                                -prob_cut_beta,
                                -prob_cut_beta + 1,
                                prob_cut_depth,
                                !cut_node,
                            );
                        }

                        self.undo_move(pos, mv);

                        if value >= prob_cut_beta {
                            tt_writer.write(
                                pos_key,
                                value_to_tt(value, (*ss).ply),
                                (*ss).tt_pv,
                                Bound::LOWER,
                                prob_cut_depth + 1,
                                mv,
                                unadjusted_static_eval,
                                self.tt().generation(),
                            );

                            if !is_decisive(value) {
                                return value - (prob_cut_beta - beta);
                            }
                        }
                    }
                }
            }

            // Step 13. A small ProbCut idea
            prob_cut_beta = beta + 428;
            if (tt_data.bound & Bound::LOWER) != Bound::NONE
                && tt_data.depth >= depth - 4
                && tt_data.value >= prob_cut_beta
                && !is_decisive(beta)
                && is_valid(tt_data.value)
                && !is_decisive(tt_data.value)
                && !seek_mate
            {
                return prob_cut_beta;
            }

            // Prepare continuation histories - single array to avoid intermediate allocation
            let cont_hist: [&PieceToHistory; 6] = {
                [
                    &*(*ss.offset(-1)).continuation_history,
                    &*(*ss.offset(-2)).continuation_history,
                    &*(*ss.offset(-3)).continuation_history,
                    &*(*ss.offset(-4)).continuation_history,
                    &*(*ss.offset(-5)).continuation_history,
                    &*(*ss.offset(-6)).continuation_history,
                ]
            };

            // The pawn entry comes directly from the shared histories
            let pawn_entry = self.shared_state.shared_histories.pawn_entry(pos);
            let mut mp = MovePicker::new_main(
                pos,
                tt_data.mv,
                depth,
                self.main_history.as_ref() as *const _,
                self.low_ply_history.as_ref() as *const _,
                self.capture_history.as_ref() as *const _,
                &cont_hist[..],
                pawn_entry as *const _,
                (*ss).ply,
            );

            value = best_value;
            let mut move_count = 0;

            // Step 14. Loop through all legal moves until no moves remain
            // or a beta cutoff occurs.
            loop {
                move_ = mp.next_move();
                if move_ == Move::none() {
                    break;
                }
                debug_assert!(move_.is_ok());

                if move_ == excluded_move {
                    continue;
                }

                // At root obey searchmoves
                if root_node {
                    // At root obey the "searchmoves" option and skip moves not listed in Root
                    // Move List. In MultiPV mode we also skip PV moves that have been already
                    // searched and those of lower "TB rank" if we are in a TB root position.
                    let in_root_moves = self.root_moves[self.pv_idx..self.pv_last]
                        .iter()
                        .any(|rm| rm.pv[0] == move_);
                    if !in_root_moves {
                        continue;
                    }
                }

                move_count += 1;
                (*ss).move_count = move_count;

                if root_node
                    && self.is_main_thread()
                    && self.counters.nodes.load(Ordering::Relaxed) > NODES_LIMIT_OUTPUT
                {
                    let info = InfoIteration {
                        depth,
                        currmove: UciEngine::move_str(move_, pos.is_chess960()),
                        currmovenumber: move_count as usize + self.pv_idx,
                    };
                    // Report current move being searched for long searches
                    self.main_manager().call_on_iter(info);
                }

                if pv_node {
                    (*ss.offset(1)).pv = core::ptr::null_mut();
                }

                extension = 0;
                capture = pos.capture_stage(move_);
                moved_piece = pos.moved_piece(move_);
                gives_check = pos.gives_check(move_);

                // Calculate new depth
                // Calculate new depth for this move
                new_depth = depth - 1;

                let mut r = self.reduction(improving, depth, move_count);

                // Decrease reduction for PvNodes
                if pv_node {
                    r -= 512;
                }

                // Increase reduction for ttPv nodes
                if (*ss).tt_pv {
                    r += 929;
                }

                // Step 15. Pruning at shallow depths.
                // Depth conditions are important for mate finding.
                if !root_node && pos.non_pawn_material_of(us) > 0 && !is_loss(best_value) {
                    // Skip quiet moves if movecount exceeds our threshold
                    // div_pos: the divisor is 1 or 2.
                    if move_count >= div_pos(3 + depth * depth, 2 - improving as i32) {
                        mp.skip_quiet_moves();
                    }

                    // Reduced depth of the next LMR search
                    let lmr_depth = new_depth - r / 1024;

                    if capture || gives_check {
                        let captured_piece = pos.piece_on(move_.to_sq());
                        // SAFETY: piece is 0-15, square is 0-63, piece_type is 0-7
                        let capt_hist = {
                            self.capture_history
                                .get_unchecked(moved_piece as usize)
                                .get_unchecked(move_.to_sq() as usize)
                                .get_unchecked(piece_type(captured_piece) as usize)
                        };

                        // Futility pruning for captures
                        if !gives_check && lmr_depth < 8 {
                            let futility_value = (*ss).static_eval
                                + 234
                                + 247 * lmr_depth
                                + piece_value(captured_piece)
                                + 134 * capt_hist.get() as i32 / 1024;

                            if futility_value <= alpha {
                                continue;
                            }
                        }

                        // SEE based pruning for captures and checks
                        // Avoid pruning sacrifices of our last piece for stalemate
                        let margin = 177 * depth + capt_hist.get() as i32 * 34 / 1024;
                        if (alpha >= VALUE_DRAW
                            || pos.non_pawn_material_of(us) != piece_value(moved_piece))
                            && !pos.see_ge(move_, -margin)
                        {
                            continue;
                        }
                    } else if !(*ss).follow_pv || !pv_node {
                        // SAFETY: cont_hist has 6 elements (0-5 valid), piece is 0-15, square is 0-63
                        let history = {
                            (*cont_hist.get_unchecked(0))
                                .get_unchecked(moved_piece as usize)
                                .get_unchecked(move_.to_sq() as usize)
                                .get() as i32
                                + (*cont_hist.get_unchecked(1))
                                    .get_unchecked(moved_piece as usize)
                                    .get_unchecked(move_.to_sq() as usize)
                                    .get() as i32
                                + self
                                    .shared_state
                                    .shared_histories
                                    .pawn_entry(pos)
                                    .get_unchecked(moved_piece as usize)
                                    .get_unchecked(move_.to_sq() as usize)
                                    .get() as i32
                        };

                        // Continuation history based pruning
                        if history < -4136 * depth {
                            continue;
                        }

                        // SAFETY: us is 0 or 1, raw() is u16 (0-65535)
                        let history_boost = {
                            69 * self
                                .main_history
                                .get_unchecked(us as usize)
                                .get_unchecked(move_.raw() as usize)
                                .get() as i32
                                / 32
                        };
                        // (*Scaler): Generally, lower divisors scale well
                        // div_pos: lmr_divisor(depth) is at least 3000 for depth >= 1 (min(depth, 16)
                        // keeps (d - 8)^2 small, and depth > 0 in the move loop).
                        let adjusted_lmr_depth =
                            lmr_depth + div_pos(history + history_boost, lmr_divisor(depth));

                        let futility_value = (*ss).static_eval
                            + 119 * adjusted_lmr_depth
                            + 90 * ((*ss).static_eval > alpha) as i32
                            + 164;

                        // Futility pruning: parent node
                        // (*Scaler): Generally, more frequent futility pruning scales well
                        if !(*ss).in_check && adjusted_lmr_depth < 12 && futility_value <= alpha {
                            if best_value <= futility_value && !is_win(futility_value) {
                                best_value = futility_value;
                            }
                            continue;
                        }

                        let adjusted_lmr_depth = max(adjusted_lmr_depth, 0);

                        // Prune moves with negative SEE
                        if !pos.see_ge(move_, -23 * adjusted_lmr_depth * adjusted_lmr_depth) {
                            continue;
                        }
                    }
                }

                // Step 16. Singular Extensions
                //
                // We check for "only moves": if one move fails high on (alpha, beta) but all
                // others fail low on (alpha-s, beta-s), then that move is singular. Singular
                // moves are extended to better estimate the result of the (putatively)
                // forced line. If it's non-singular, we may do some pruning or reduction.
                //
                // Recursive excluded search is avoided. The `excluded_move` mechanism
                // was historically a hack and remains a bit fragile.
                //
                // (*Scaler) Generally, higher singular_beta (i.e closer to tt_value)
                // and lower extension margins scale well.
                if !root_node
                    && move_ == tt_data.mv
                    && excluded_move == Move::none()
                    && depth >= 6 + (*ss).tt_pv as i32
                    && is_valid(tt_data.value)
                    && !is_decisive(tt_data.value)
                    && (tt_data.bound & Bound::LOWER) != Bound::NONE
                    && tt_data.depth >= depth - 3
                    && !is_shuffling(move_, ss, pos)
                    && !seek_mate
                {
                    let singular_beta =
                        tt_data.value - (59 + 66 * ((*ss).tt_pv && !pv_node) as i32) * depth / 63;
                    let singular_depth = new_depth / 2;

                    (*ss).excluded_move = move_;
                    value = self.search::<NonPV>(
                        pos,
                        ss,
                        singular_beta - 1,
                        singular_beta,
                        singular_depth,
                        cut_node,
                    );
                    (*ss).excluded_move = Move::none();

                    if value < singular_beta {
                        let corr_val_adj = correction_value.abs() / 198368;
                        let tt_move_history = self.tt_move_history;
                        let double_margin = -2 + 204 * pv_node as i32
                            - 152 * !tt_capture as i32
                            - corr_val_adj
                            - 1175 * tt_move_history.get() as i32 / 114178
                            - ((*ss).ply > self.root_depth) as i32 * 38;
                        let triple_margin = 70 + 279 * pv_node as i32 - 188 * !tt_capture as i32
                            + 81 * (*ss).tt_pv as i32
                            - corr_val_adj
                            - ((*ss).ply > self.root_depth) as i32 * 43;

                        extension = 1
                            + (value < singular_beta - double_margin) as i32
                            + (value < singular_beta - triple_margin) as i32;

                        // C++ `depth++` (by 1, not by the extension);
                        // extension is applied to newDepth later
                        depth += 1;
                    } else if value >= beta && !is_decisive(value) {
                        // Multi-cut pruning
                        // Our ttMove is assumed to fail high based on the bound of the TT entry,
                        // and if after excluding the ttMove with a reduced search we fail high
                        // over the original beta, we assume this expected cut-node is not
                        // singular (multiple moves fail high), and we can prune the whole
                        // subtree by returning a softbound.
                        // Penalty to TT move history on MultiCut
                        self.tt_move_history.update(-421 - 110 * depth);

                        if !(*ss).in_check && value > (*ss).static_eval {
                            let bonus = ((value - (*ss).static_eval) * 664 / 1024)
                                .clamp(-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                            self.update_correction_history(pos, ss, bonus);
                        }

                        return value;
                    }
                    // Negative extensions
                    // If other moves failed high over (ttValue - margin) without the
                    // ttMove on a reduced search, but we cannot do multi-cut because
                    // (ttValue - margin) is lower than the original beta, we do not know
                    // if the ttMove is singular or can do a multi-cut, so we reduce the
                    // ttMove in favor of other moves based on some conditions:
                    //
                    // If the ttMove is assumed to fail high over current beta or
                    // if we are on a cutNode
                    else if tt_data.value >= beta || cut_node {
                        extension = -3;
                    }
                }

                let node_count = if root_node {
                    self.counters.nodes.load(Ordering::Relaxed)
                } else {
                    0
                };

                // Step 17. Make the move
                self.do_move(pos, move_, gives_check, capture, ss);

                // Add extension
                new_depth += extension;

                // Decrease reduction for PvNodes
                if (*ss).tt_pv {
                    r -= 3023
                        + pv_node as i32 * 1004
                        + (tt_data.value > alpha) as i32 * 885
                        + (tt_data.depth >= depth) as i32 * (816 + cut_node as i32 * 940);
                }

                r += 697; // Base reduction offset to compensate for other tweaks
                r -= move_count * 65;
                r -= correction_value.abs() / 26310;

                // Increase reduction for cut nodes
                if cut_node {
                    r += 4026 + 933 * (tt_data.mv == Move::none()) as i32;
                }

                // Increase reduction if ttMove is a capture
                if tt_capture {
                    r += 1079;
                }

                // Increase reduction if next ply has a lot of fail high
                if (*ss.offset(1)).cutoff_cnt > 1 {
                    r += 264
                        + 1095 * ((*ss.offset(1)).cutoff_cnt > 2) as i32
                        + 1138 * all_node as i32;
                }
                // For first picked move (ttMove) reduce reduction
                else if move_ == tt_data.mv {
                    r -= 2179;
                }

                if capture {
                    // SAFETY: moved_piece is Piece (0-15), to_sq is Square (0-63), piece_type is PieceType (0-7)
                    (*ss).stat_score = {
                        873 * piece_value(pos.captured_piece()) as i32 / 128
                            + (**self
                                .capture_history
                                .get_unchecked(moved_piece as usize)
                                .get_unchecked(move_.to_sq() as usize)
                                .get_unchecked(piece_type(pos.captured_piece()) as usize))
                                as i32
                    };
                } else {
                    // SAFETY: us is 0 or 1, raw() is u16, moved_piece is Piece, to_sq is Square
                    (*ss).stat_score = {
                        (2252
                            * self
                                .main_history
                                .get_unchecked(us as usize)
                                .get_unchecked(move_.raw() as usize)
                                .get() as i32
                            + 1126
                                * (*cont_hist.get_unchecked(0))
                                    .get_unchecked(moved_piece as usize)
                                    .get_unchecked(move_.to_sq() as usize)
                                    .get() as i32
                            + 1093
                                * (*cont_hist.get_unchecked(1))
                                    .get_unchecked(moved_piece as usize)
                                    .get_unchecked(move_.to_sq() as usize)
                                    .get() as i32)
                            / 1024
                    };
                }

                // Decrease/increase reduction for moves with a good/bad history
                r -= (*ss).stat_score * 439 / 4096;

                // Reduce less aggressively in loose alpha windows.
                if !capture && !is_decisive(alpha) {
                    r += 3 * (alpha - eval).clamp(-64, 96);
                }

                // Scale up reductions for expected ALL nodes
                if all_node {
                    // div_pos: depth >= 1 here (the node is past the depth <= 0 qsearch dive), so
                    // the divisor is at least 524.
                    r += div_pos(r * 276, 256 * depth + 268);
                }

                // When seeking a mate, reduce non-checking moves more and checking moves less.
                if seek_mate && r > 0 {
                    r += if gives_check { -r } else { r } / 4;
                }

                // Step 18. Compute and apply late moves reductions/extensions (LMR)
                if depth >= 2 && move_count > 1 {
                    // In general we want to cap the LMR depth search at new_depth, but when
                    // reduction is negative, we allow this move a limited search extension
                    // beyond the first move depth. To avoid search explosion, extensions
                    // are not allowed deep, relative to root_depth, in the search tree.
                    let extension_cap = if (*ss).ply < 2 * self.root_depth {
                        2
                    } else {
                        0
                    };
                    let d = max(1, new_depth + min(-r / 1024, extension_cap)) + pv_node as i32;

                    (*ss).reduction = new_depth - d;
                    value = -self.search::<NonPV>(pos, ss.offset(1), -(alpha + 1), -alpha, d, true);
                    (*ss).reduction = 0;

                    if value > alpha {
                        // Adjust full-depth search based on LMR results
                        // Adjust full-depth search based on LMR results - if the result was
                        // good enough search deeper, if it was bad enough search shallower.
                        let do_deeper_search = d < new_depth && value > best_value + 53;
                        let do_shallower_search = value < best_value + 8;

                        new_depth += do_deeper_search as i32 - do_shallower_search as i32;

                        if new_depth > d {
                            value = -self.search::<NonPV>(
                                pos,
                                ss.offset(1),
                                -(alpha + 1),
                                -alpha,
                                new_depth,
                                !cut_node,
                            );
                        }

                        self.update_continuation_histories(ss, moved_piece, move_.to_sq(), 1334);
                    }
                }
                // Step 19. Full-depth search when LMR is skipped
                else if !pv_node || move_count > 1 {
                    // Increase reduction if ttMove is not present
                    if tt_data.mv == Move::none() {
                        r += 1127;
                    }

                    value = -self.search::<NonPV>(
                        pos,
                        ss.offset(1),
                        -(alpha + 1),
                        -alpha,
                        new_depth - (r > 5234) as i32 - (r > 5487 && new_depth > 2) as i32,
                        !cut_node,
                    );
                }

                // For PV nodes, do full PV search on first move or after fail high
                // Step 20. For PV nodes only, do a full PV search on the first move
                // or after a fail high, otherwise let the parent node fail low with
                // value <= alpha and try another move.
                if pv_node && (move_count == 1 || value > alpha) {
                    (*ss.offset(1)).pv = pv_ptr;
                    (*(*ss.offset(1)).pv) = Move::none();

                    // Extend move from transposition table if we are about to dive into qsearch.
                    // Decisive score handling improves mate finding and retrograde analysis.
                    // See: https://github.com/official-stockfish/Stockfish/pull/6386
                    if move_ == tt_data.mv
                        && ((is_valid(tt_data.value)
                            && is_decisive(tt_data.value)
                            && tt_data.depth > 0)
                            || tt_data.depth > 1)
                    {
                        new_depth = max(new_depth, 1);
                    }

                    value = -self.search::<PV>(pos, ss.offset(1), -beta, -alpha, new_depth, false);
                }

                // Step 21. Undo move
                self.undo_move(pos, move_);

                debug_assert!(value > -VALUE_INFINITE && value < VALUE_INFINITE);

                // Step 22. Check for a new best move
                // If a stop occurred, the value of the search cannot be trusted,
                // and we return immediately without updating the best move,
                // principal variation or transposition table.
                // should_stop() reads the stop AtomicBool directly. Taking an RwLock read
                // guard per node could deadlock behind a pending writer, since std RwLock
                // is writer-preferring.
                if self.should_stop() {
                    return VALUE_ZERO;
                }

                if root_node {
                    let rm = self.root_moves[self.pv_idx..self.pv_last]
                        .iter_mut()
                        .find(|rm| rm.pv[0] == move_)
                        .unwrap();

                    let nodes_now = self.counters.nodes.load(Ordering::Relaxed);
                    rm.effort += nodes_now - node_count;

                    let n = nodes_now - node_count;
                    let e_prev = (rm.effort - n).max(1);

                    // Dynamic EMA parameters for root move
                    // Exponential moving average parameters for root move
                    const SCALE: u64 = 32;
                    const CHI_NUMERATOR: u64 = 3;
                    const CHI_DENOMINATOR: u64 = 2; // Chi = 3/2 = 1.5
                    const MIN_WEIGHT: u64 = 12; // 37.5% minimum weight
                    const MAX_WEIGHT: u64 = 24; // 75% maximum weight

                    let w = ((SCALE * n * CHI_DENOMINATOR)
                        / (n * CHI_DENOMINATOR + CHI_NUMERATOR * e_prev))
                        .clamp(MIN_WEIGHT, MAX_WEIGHT);
                    let w_mss = w.min(16);
                    let v2 = value as i64 * value.abs() as i64;

                    // The following expressions intentionally use wrapping u64 arithmetic
                    // to bit-for-bit match C++ (int * unsigned long) semantics.
                    if rm.average_score == -VALUE_INFINITE {
                        rm.average_score = value;
                    } else {
                        rm.average_score = ((value as u64)
                            .wrapping_mul(w)
                            .wrapping_add((rm.average_score as u64).wrapping_mul(SCALE - w))
                            / SCALE) as i32;
                    }

                    if rm.mean_squared_score == -VALUE_INFINITE * VALUE_INFINITE {
                        rm.mean_squared_score = value * value.abs();
                    } else {
                        rm.mean_squared_score = ((v2 as u64).wrapping_mul(w_mss).wrapping_add(
                            (rm.mean_squared_score as i64 as u64).wrapping_mul(SCALE - w_mss),
                        ) / SCALE) as i32;
                    }

                    // Always update score for root moves (matches C++ search.cpp)
                    // Only update PV if move beats alpha or is first move
                    if move_count == 1 || value > alpha {
                        rm.score = value;
                        rm.uci_score = value;
                        rm.sel_depth = self.sel_depth;
                        rm.unset_inexact();

                        if value >= beta {
                            rm.inexact_lower = true;
                            rm.uci_score = beta;
                        } else if value <= alpha {
                            rm.inexact_upper = true;
                            rm.uci_score = alpha;
                        }

                        rm.pv.clear();
                        rm.pv.push(move_);

                        debug_assert!(!(*ss.offset(1)).pv.is_null());

                        let mut m = (*ss.offset(1)).pv;
                        while *m != Move::none() {
                            rm.pv.push(*m);
                            m = m.offset(1);
                        }

                        if move_count > 1 && self.pv_idx == 0 {
                            self.counters
                                .best_move_changes
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        // All other moves but the PV are set to lowest value: this
                        // is not a problem when sorting because the sort is stable and the
                        // move position in the list is preserved - just the PV is pushed up.
                        rm.score = -VALUE_INFINITE;
                    }
                }

                // Check for new best value
                let inc = (value == best_value
                    && (*ss).ply + 2 >= self.root_depth
                    && (self.counters.nodes.load(Ordering::Relaxed) & 14) == 0
                    && !is_win(value.abs() + 1)) as i32;

                if value + inc > best_value {
                    best_value = value;

                    if value + inc > alpha {
                        best_move = move_;

                        if pv_node && !root_node {
                            update_pv((*ss).pv, move_, (*ss.offset(1)).pv);
                        }

                        if value >= beta {
                            // (*Scaler) Infrequent and small updates scale well
                            (*ss).cutoff_cnt += ((extension < 2) || pv_node) as i32;
                            debug_assert!(value >= beta);
                            break;
                        }

                        // Reduce other moves if we have found at least one score improvement
                        if depth > 3 && depth < 12 && !is_decisive(value) {
                            depth -= 3;
                        }

                        debug_assert!(depth > 0);
                        alpha = value;
                    }
                }

                // If the move is worse than some previously searched move,
                // remember it, to update its stats later.
                // C++ (search.cpp) gates on the MOVE COUNT, not on how many
                // moves have been recorded so far.
                if move_ != best_move && move_count <= SEARCHEDLIST_CAPACITY as i32 {
                    if capture {
                        captures_searched.push(move_);
                    } else {
                        quiets_searched.push(move_);
                    }
                }
            }

            // Step 23. Check for mate and stalemate, otherwise update bestmove/countermove stats
            debug_assert!(
                move_count > 0
                    || (*ss).in_check
                    || excluded_move != Move::none()
                    || generate(pos, GenType::All).is_empty()
            );

            // Adjust best value for fail high cases
            if best_value >= beta && !is_decisive(best_value) && !is_decisive(alpha) {
                // SAFETY: depth >= 1 in search, so depth + 1 >= 2: no division by zero and
                // no `i32::MIN / -1` overflow. The hint lets LLVM drop both checks.
                core::hint::assert_unchecked(depth >= 1);
                best_value = (best_value * depth + beta) / (depth + 1);
            }

            if move_count == 0 {
                best_value = if excluded_move != Move::none() {
                    alpha
                } else if (*ss).in_check {
                    mated_in((*ss).ply)
                } else {
                    VALUE_DRAW
                };
            } else if best_move != Move::none() {
                self.update_all_stats(
                    pos,
                    ss,
                    best_move,
                    prev_sq,
                    &quiets_searched,
                    &captures_searched,
                    depth,
                    tt_data.mv,
                );
                if !pv_node {
                    self.tt_move_history
                        .update(if best_move == tt_data.mv { 918 } else { -747 });
                }
            } else if !prior_capture && let Some(prev_sq) = prev_sq {
                let mut bonus_scale = -241;
                bonus_scale -= (*ss.offset(-1)).stat_score / 98;
                bonus_scale += min(59 * depth, 420);
                bonus_scale += 186 * ((*ss.offset(-1)).move_count > 9) as i32;
                bonus_scale +=
                    142 * (!(*ss).in_check && best_value <= (*ss).static_eval - 106) as i32;
                bonus_scale += 159
                    * (!(*ss.offset(-1)).in_check
                        && best_value <= -(*ss.offset(-1)).static_eval - 68)
                        as i32;

                bonus_scale = max(bonus_scale, 0);

                // scaledBonus ranges from 0 to roughly 2.3M, overflows happen for
                // multipliers larger than 900
                let scaled_bonus = min(150 * depth - 85, 1337) * bonus_scale;

                self.update_continuation_histories(
                    ss.offset(-1),
                    pos.piece_on(prev_sq),
                    prev_sq,
                    scaled_bonus * 263 / 16384,
                );

                // SAFETY: !us is 0 or 1, raw() is u16 (0-65535)
                {
                    self.main_history
                        .get_unchecked_mut(!us as usize)
                        .get_unchecked_mut((*ss.offset(-1)).current_move.raw() as usize)
                        .update(scaled_bonus * 215 / 32768);
                }

                if piece_type(pos.piece_on(prev_sq)) != PieceType::Pawn
                    && (*ss.offset(-1)).current_move.move_type() != MoveType::Promotion
                {
                    // SAFETY: piece is 0-15, square is 0-63
                    {
                        self.shared_state
                            .shared_histories
                            .pawn_entry(pos)
                            .get_unchecked(pos.piece_on(prev_sq) as usize)
                            .get_unchecked(prev_sq as usize)
                            .update(scaled_bonus * 324 / 8192);
                    }
                }
            } else if let Some(prev_sq) = prev_sq {
                // Bonus for prior capture countermove that caused the fail low
                debug_assert!(prior_capture);

                let captured_piece = pos.captured_piece();
                debug_assert!(captured_piece != NO_PIECE);
                // SAFETY: piece is 0-15, square is 0-63, piece_type is 0-7
                {
                    self.capture_history
                        .get_unchecked_mut(pos.piece_on(prev_sq) as usize)
                        .get_unchecked_mut(prev_sq as usize)
                        .get_unchecked_mut(piece_type(captured_piece) as usize)
                        .update(892);
                }
            }

            if pv_node {
                best_value = min(best_value, max_value);
            }

            if best_value <= alpha {
                (*ss).tt_pv = (*ss).tt_pv || (*ss.offset(-1)).tt_pv;
            }

            // Write to transposition table
            // Step 24. Write gathered information in transposition table. Note that the
            // static evaluation is saved as it was before correction history.
            if excluded_move == Move::none() && !(root_node && self.pv_idx > 0) {
                tt_writer.write(
                    pos_key,
                    value_to_tt(best_value, (*ss).ply),
                    (*ss).tt_pv,
                    if best_value >= beta {
                        Bound::LOWER
                    } else if pv_node && best_move != Move::none() {
                        Bound::EXACT
                    } else {
                        Bound::UPPER
                    },
                    if move_count != 0 {
                        depth
                    } else {
                        min(MAX_PLY - 1, depth + 6)
                    },
                    best_move,
                    unadjusted_static_eval,
                    self.tt().generation(),
                );
            }

            // Adjust correction history if the best move is not a capture
            // and the error direction matches whether we are above/below bounds.
            if !(*ss).in_check
                && !(best_move != Move::none() && pos.capture(best_move))
                && (best_value > (*ss).static_eval) == (best_move != Move::none())
            {
                let bonus = ((best_value - (*ss).static_eval)
                    * depth
                    * (if best_move != Move::none() { 12 } else { 18 })
                    / 128)
                    .clamp(-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                self.update_correction_history(pos, ss, 1061 * bonus / 1024);
            }

            debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);

            best_value
        }
    }

    // Quiescence search
    /// Quiescence search function, which is called by the main search function with depth zero, or
    /// recursively with further decreasing depth. With depth <= 0, we "should" be using static eval only,
    /// but tactical moves may confuse the static eval. To fight this horizon effect, we implement this
    /// qsearch of tactical moves only.
    fn qsearch<NT: NodeType>(
        &mut self,
        pos: &mut Position,
        ss: *mut Stack,
        mut alpha: Value,
        beta: Value,
    ) -> Value {
        let pv_node = NT::IS_PV;
        debug_assert!(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || (alpha == beta - 1));

        // Nodes are counted in do_move(), as in C++.

        // SAFETY: `ss` is this node's entry of the Stack array allocated in iterative_deepening (ply <= MAX_PLY because
        // nodes at ply >= MAX_PLY return before recursing), so ss-7..=ss+2 are in bounds; entries below ss were set by
        // the ancestors or are sentinels. Other raw accesses in this block rely on the invariants noted at each use.
        unsafe {
            // Check for upcoming repetition
            // Check if we have an upcoming move that draws by repetition
            if alpha < VALUE_DRAW && pos.upcoming_repetition((*ss).ply) {
                alpha = value_draw(self.counters.nodes.load(Ordering::Relaxed));
                if alpha >= beta {
                    return alpha;
                }
            }

            // Uninitialised PV buffer; see `search`: slot 0 is the terminator, written here, and all
            // readers stop at the first `Move::none()`.
            let mut pv = [core::mem::MaybeUninit::<Move>::uninit(); MAX_PLY as usize + 1];
            pv[0] = core::mem::MaybeUninit::new(Move::none());
            let pv_ptr = pv.as_mut_ptr() as *mut Move;
            let mut move_: Move;
            let mut best_move: Move;
            let mut best_value: Value;
            let mut value: Value;
            let futility_base: Value;

            let mut gives_check: bool;
            let mut capture: bool;
            let mut move_count: i32;

            // Step 1. Initialize node
            if pv_node {
                (*ss.offset(1)).pv = pv_ptr;
                (*(*ss).pv) = Move::none();
            }

            best_move = Move::none();
            (*ss).in_check = pos.checkers() != 0;
            move_count = 0;

            // Update selective depth
            // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
            if pv_node && self.sel_depth < (*ss).ply + 1 {
                self.sel_depth = (*ss).ply + 1;
            }

            // Step 2. Check for an immediate draw or maximum ply reached
            if pos.is_draw((*ss).ply) || (*ss).ply >= MAX_PLY {
                return if (*ss).ply >= MAX_PLY && !(*ss).in_check {
                    self.evaluate(pos)
                } else {
                    VALUE_DRAW
                };
            }

            debug_assert!(0 <= (*ss).ply && (*ss).ply < MAX_PLY);

            // Step 3. Transposition table lookup
            let pos_key: Key = pos.key();
            let (tt_hit, mut tt_data, tt_writer) = self.tt().probe_for_search(pos_key);
            (*ss).tt_hit = tt_hit;
            tt_data.mv = if tt_hit { tt_data.mv } else { Move::none() };
            tt_data.value = if tt_hit {
                value_from_tt(tt_data.value, (*ss).ply, pos.rule50_count())
            } else {
                VALUE_NONE
            };
            let pv_hit: bool = tt_hit && tt_data.is_pv;

            // TT cutoff
            if !pv_node
                && tt_data.depth >= DEPTH_QS
                && is_valid(tt_data.value)
                && (tt_data.bound
                    & if tt_data.value >= beta {
                        Bound::LOWER
                    } else {
                        Bound::UPPER
                    })
                    != Bound::NONE
            {
                return tt_data.value;
            }

            // Step 4. Static evaluation
            let mut unadjusted_static_eval = VALUE_NONE;
            if (*ss).in_check {
                best_value = -VALUE_INFINITE;
                futility_base = -VALUE_INFINITE;
            } else {
                let correction_value = self.correction_value(pos, ss);

                if (*ss).tt_hit {
                    unadjusted_static_eval = tt_data.eval;
                    if !is_valid(unadjusted_static_eval) {
                        unadjusted_static_eval = self.evaluate(pos);
                    }
                    (*ss).static_eval =
                        to_corrected_static_eval(unadjusted_static_eval, correction_value);
                    best_value = (*ss).static_eval;

                    if is_valid(tt_data.value)
                        && !is_decisive(tt_data.value)
                        && (tt_data.bound
                            & if tt_data.value > best_value {
                                Bound::LOWER
                            } else {
                                Bound::UPPER
                            })
                            != Bound::NONE
                    {
                        best_value = tt_data.value;
                    }
                } else {
                    unadjusted_static_eval = self.evaluate(pos);
                    (*ss).static_eval =
                        to_corrected_static_eval(unadjusted_static_eval, correction_value);
                    best_value = (*ss).static_eval;
                }

                // Stand pat
                if best_value >= beta {
                    debug_assert!(!is_decisive(best_value));
                    best_value = (441 * best_value + 583 * beta) / 1024;

                    if !(*ss).tt_hit {
                        tt_writer.write(
                            pos_key,
                            VALUE_NONE,
                            false,
                            Bound::LOWER,
                            DEPTH_UNSEARCHED,
                            Move::none(),
                            unadjusted_static_eval,
                            self.tt().generation(),
                        );
                    }
                    return best_value;
                }

                if best_value > alpha {
                    alpha = best_value;
                }

                futility_base = (*ss).static_eval + 306;
            }

            // qsearch uses a single continuation-history element
            let cont_hist_ptrs: [*const PieceToHistory; 1] =
                [(*ss.offset(-1)).continuation_history];

            // Convert raw pointers to references - use fixed array instead of Vec
            let cont_hist: [&PieceToHistory; 1] = { [&*cont_hist_ptrs[0]] };

            let prev_sq = if (*ss.offset(-1)).current_move.is_ok() {
                Some((*ss.offset(-1)).current_move.to_sq())
            } else {
                None
            };

            // The pawn entry comes directly from the shared histories
            let pawn_entry = self.shared_state.shared_histories.pawn_entry(pos);
            let mut mp = MovePicker::new_main(
                pos,
                tt_data.mv,
                DEPTH_QS,
                self.main_history.as_ref() as *const _,
                self.low_ply_history.as_ref() as *const _,
                self.capture_history.as_ref() as *const _,
                &cont_hist[..],
                pawn_entry as *const _,
                (*ss).ply,
            );

            // Step 5. Loop through moves
            loop {
                move_ = mp.next_move();
                if move_ == Move::none() {
                    break;
                }
                debug_assert!(move_.is_ok());

                gives_check = pos.gives_check(move_);
                capture = pos.capture_stage(move_);

                move_count += 1;

                // Step 6. Pruning
                if !is_loss(best_value) {
                    // Futility pruning
                    if !gives_check
                        && Some(move_.to_sq()) != prev_sq
                        && !is_loss(futility_base)
                        && move_.move_type() != MoveType::Promotion
                    {
                        if move_count > 2 {
                            continue;
                        }

                        let futility_value =
                            futility_base + piece_value(pos.piece_on(move_.to_sq()));

                        // If static eval + value of piece we are going to capture is
                        // much lower than alpha, we can prune this move.
                        if futility_value <= alpha {
                            best_value = max(best_value, futility_value);
                            continue;
                        }

                        // If static exchange evaluation is low enough, we can prune
                        if !pos.see_ge(move_, alpha - futility_base) {
                            best_value = max(best_value, min(alpha, futility_base));
                            continue;
                        }
                    }

                    // Skip all non-captures
                    if !capture {
                        continue;
                    }

                    // Do not search moves with bad enough SEE values
                    if !pos.see_ge(move_, -74) {
                        continue;
                    }
                }

                // Step 7. Make and search move
                self.do_move(pos, move_, gives_check, capture, ss);

                value = -self.qsearch::<NT>(pos, ss.offset(1), -beta, -alpha);

                self.undo_move(pos, move_);

                debug_assert!(value > -VALUE_INFINITE && value < VALUE_INFINITE);

                // Step 8. Check for new best move
                if value > best_value {
                    best_value = value;

                    if value > alpha {
                        best_move = move_;

                        if pv_node {
                            update_pv((*ss).pv, move_, (*ss.offset(1)).pv);
                        }

                        if value < beta {
                            alpha = value;
                        } else {
                            break; // Fail high
                        }
                    }
                }
            }

            // Step 9. Check for mate and stalemate
            // All legal moves have been searched. A special case: if we are in check
            // and no legal moves were found, it is checkmate.
            if move_count == 0 {
                if (*ss).in_check {
                    debug_assert!(generate(pos, GenType::All).is_empty());
                    return mated_in((*ss).ply); // Plies to mate from the root
                }

                // Only check for stalemate under specific conditions
                let us = pos.side_to_move();
                let pawn_pushes = if us == Color::White {
                    shift(pos.pieces_c_pt::<{ PieceType::Pawn as u8 }>(us), NORTH)
                } else {
                    shift(pos.pieces_c_pt::<{ PieceType::Pawn as u8 }>(us), SOUTH)
                };
                if (pawn_pushes & !pos.pieces()) == 0
                    && pos.non_pawn_material_of(us) == 0
                    && piece_type(pos.captured_piece()) >= PieceType::Knight
                    && generate(pos, GenType::All).is_empty()
                {
                    best_value = VALUE_DRAW;
                }
            }

            if !is_decisive(best_value) && best_value > beta {
                best_value = (462 * best_value + 562 * beta) / 1024;
            }

            // Write to transposition table
            tt_writer.write(
                pos_key,
                value_to_tt(best_value, (*ss).ply),
                pv_hit,
                if best_value >= beta {
                    Bound::LOWER
                } else {
                    Bound::UPPER
                },
                DEPTH_QS,
                best_move,
                unadjusted_static_eval,
                self.tt().generation(),
            );

            debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);

            best_value
        }
    }

    // Move execution helpers
    // C++ (GCC, x86-64-avx512icl) inlines Worker::do_move into search/qsearch.
    #[inline(always)]
    fn do_move(
        &mut self,
        pos: &mut Position,
        mv: Move,
        gives_check: bool,
        capture: bool,
        ss: *mut Stack,
    ) {
        // prefetch_key does not model castling, en passant or promotion keys
        // exactly. The correction-history prefetches below have the same
        // approximation; for rare moves they land on unused lines.
        crate::misc::prefetch(self.tt().get_cluster(pos.prefetch_key(mv)));

        let moved_pc = pos.moved_piece(mv);

        // SAFETY: `ss` is non-null and a valid Stack entry with at least 5 entries before it (7 padding entries at ply 0);
        // the correction-history pointers of ss-1/-3/-5 target retained Cell entries set by an ancestor's do_move,
        // and moved_pc < 16, to < 64 index the [PIECE_NB][SQUARE_NB] table (prefetch only).
        unsafe {
            if !ss.is_null() {
                let to = mv.to_sq() as usize;
                let prev1 = (*ss.offset(-1)).continuation_correction_history;
                let prev3 = (*ss.offset(-3)).continuation_correction_history;
                let prev5 = (*ss.offset(-5)).continuation_correction_history;
                crate::misc::prefetch(
                    self.continuation_correction_entry(prev1)
                        .get_unchecked(moved_pc as usize)
                        .get_unchecked(to) as *const _,
                );
                crate::misc::prefetch(
                    self.continuation_correction_entry(prev3)
                        .get_unchecked(moved_pc as usize)
                        .get_unchecked(to) as *const _,
                );
                crate::misc::prefetch(
                    self.continuation_correction_entry(prev5)
                        .get_unchecked(moved_pc as usize)
                        .get_unchecked(to) as *const _,
                );
            }
        }

        // Increment node count for each move made
        // Use load+store instead of fetch_add to avoid locking instructions (per b083049f)
        self.counters.nodes.store(
            self.counters.nodes.load(Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );

        // SF18: `accumulatorStack.push(pos)` hands back the `Dirties` aggregate
        // (dirty piece, dirty threats and, since SFNNv16, the dirty pawn pairs)
        // that `Position::do_move` fills in as it makes the move.
        // The raw pointer dance keeps the `&mut self` borrow from `push()` from
        // conflicting with the shared continuation-history reads below; the
        // pointed-to `Dirties` lives inside `self.accumulator_stack` and is not
        // moved by anything in this function.
        // SAFETY: `push` requires its `Dirties` to be filled before the next evaluation; that is
        // the `do_move_internal` call right below, the only `push` in the engine.
        let dirties: *mut Dirties = unsafe { self.accumulator_stack.push(pos) };
        // SAFETY: `dirties` points at the freshly pushed accumulator state, which does not
        // alias `pos`.
        unsafe {
            pos.do_move_internal(mv, gives_check, &mut *dirties);
        }

        // C++ `Position::do_move` prefetches the shared-history lines the child node reads
        // (pawn history and the four correction-history entries) once the new keys are known;
        // `do_move_internal` has no history parameter, so do it here.
        {
            let shared = &self.shared_state.shared_histories;
            // SAFETY: moved_pc < PIECE_NB and to < SQUARE_NB index the [PIECE_NB][SQUARE_NB]
            // pawn-history entry (the pointer is only prefetched, never dereferenced).
            let pawn_slot = unsafe {
                shared
                    .pawn_entry(pos)
                    .get_unchecked(moved_pc as usize)
                    .get_unchecked(mv.to_sq() as usize) as *const _
            };
            crate::misc::prefetch(pawn_slot);
            crate::misc::prefetch(shared.pawn_correction_entry(pos) as *const _);
            crate::misc::prefetch(shared.minor_piece_correction_entry(pos) as *const _);
            crate::misc::prefetch(shared.nonpawn_correction_entry_white(pos) as *const _);
            crate::misc::prefetch(shared.nonpawn_correction_entry_black(pos) as *const _);
        }

        // SAFETY: `ss` is non-null and a valid Stack entry (see above); in_check/capture are bool (0/1), moved_pc < 16 and
        // to < 64 index the history tables, whose pointers stay valid as they live in shared state / this worker.
        unsafe {
            if !ss.is_null() {
                (*ss).current_move = mv;
                // SAFETY: in_check is bool (0-1), capture is bool (0-1), piece is 0-15, square is 0-63
                (*ss).continuation_history =
                    self.continuation_history()
                        .get_unchecked((*ss).in_check as usize)
                        .get_unchecked(capture as usize)
                        .get_unchecked(moved_pc as usize)
                        .get_unchecked(mv.to_sq() as usize) as *const _;
                (*ss).continuation_correction_history =
                    self.continuation_correction_history
                        .get_unchecked(moved_pc as usize)
                        .get_unchecked(mv.to_sq() as usize) as *const _;
            }
        }
    }

    #[inline(always)]
    /// # Safety
    /// `ss` must be a valid Stack entry of the current search with the padding/ancestor entries before it (see the SAFETY note in the body).
    unsafe fn do_null_move(&mut self, pos: &mut Position, ss: *mut Stack) {
        // SAFETY: the caller only makes a null move when not in check.
        unsafe { pos.do_null_move() };
        // SAFETY: `ss` is a valid Stack entry (the caller's node); the indices are 0 or NO_PIECE (< PIECE_NB), in bounds of
        // the history tables, which outlive the stack.
        unsafe {
            (*ss).current_move = Move::null();
            // SAFETY: indices are all 0 or NO_PIECE (valid piece index)
            (*ss).continuation_history = self
                .continuation_history()
                .get_unchecked(0)
                .get_unchecked(0)
                .get_unchecked(NO_PIECE as usize)
                .get_unchecked(0) as *const _;
            (*ss).continuation_correction_history =
                &self.continuation_correction_history[NO_PIECE as usize][0];
        }
    }

    #[inline(always)]
    fn undo_move(&mut self, pos: &mut Position, mv: Move) {
        // SAFETY: search undoes its moves in reverse order.
        unsafe { pos.undo_move(mv) };
        self.accumulator_stack.pop();
    }

    fn undo_null_move(&mut self, pos: &mut Position) {
        // SAFETY: search undoes its null move before any other move.
        unsafe { pos.undo_null_move() };
    }

    // Reduction calculation
    fn reduction(&self, improving: bool, depth: Depth, move_count: i32) -> i32 {
        // C++: reductionScale + !i * reductionScale * 197 / 512 + 948
        // The reductions table has MAX_MOVES (256) entries, clamp to valid indices.
        // Callers (LMR in `search`, depth > 0) pass depth >= 0 and move_count >= 1.
        debug_assert!(depth >= 0 && move_count >= 0);
        // SAFETY: depth and move_count are non-negative (above) and clamped to <= 255, REDUCTIONS has 256 entries
        let reduction_scale = unsafe {
            *REDUCTIONS.get_unchecked(depth.min(255) as usize)
                * *REDUCTIONS.get_unchecked(move_count.min(255) as usize)
        };
        reduction_scale + !improving as i32 * reduction_scale * 197 / 512 + 948
    }

    // Evaluate position
    /// Evaluate the current position of the game tree, from the point of view of
    /// the side to move.
    #[inline(always)]
    fn evaluate(&mut self, pos: &Position) -> Value {
        nnue_evaluate(
            &self.network_snapshot,
            pos,
            &mut self.accumulator_stack,
            &mut self.refresh_table,
        )
    }

    // Get correction value for static evaluation
    /// (*Scalers):
    /// The values with Scaler asterisks have proven non-linear scaling.
    /// They are optimized to time controls of 180 + 1.8 and longer,
    /// so changing them or adding conditions that are similar requires
    /// tests at these types of time controls.
    /// (*Scaler) All tuned parameters at time controls shorter than
    /// optimized for require verifications at longer time controls.
    #[inline(always)]
    /// # Safety
    /// `ss` must be a valid Stack entry of the current search with the padding/ancestor entries before it (see the SAFETY note in the body).
    unsafe fn correction_value(&self, pos: &Position, ss: *const Stack) -> Value {
        // SAFETY: `ss` is a valid Stack entry with at least 6 entries before it (7 padding entries at ply 0); the
        // correction-history indices of ss-2/-4/-6 are sentinels or set by ancestors' do_move;
        // pc < 16 / sq < 64 index the [PIECE_NB][SQUARE_NB] tables.
        unsafe {
            let m = (*ss.offset(-1)).current_move;
            let shared = &self.shared_state.shared_histories;

            // Read from the shared correction histories
            let pcv = shared.pawn_correction_entry(pos).pawn.get() as i32;
            let micv = shared.minor_piece_correction_entry(pos).minor.get() as i32;
            let wnpcv = shared
                .nonpawn_correction_entry_white(pos)
                .non_pawn_white
                .get() as i32;
            let bnpcv = shared
                .nonpawn_correction_entry_black(pos)
                .non_pawn_black
                .get() as i32;

            // Add 2-, 4- and 6-ply continuation correction history terms (weights 7885, 7885, 6307);
            // when there is no previous move the constant 80695 is used instead (matches C++).
            // Note: continuation correction history stays per-worker.
            // SAFETY: pc is Piece (0-15), sq is Square (0-63)
            let cntcv = if m.is_ok() {
                let pc = pos.piece_on(m.to_sq()) as usize;
                let sq = m.to_sq() as usize;
                let cnt2 = self
                    .continuation_correction_entry((*ss.offset(-2)).continuation_correction_history)
                    .get_unchecked(pc)
                    .get_unchecked(sq)
                    .get() as i32;
                let cnt4 = self
                    .continuation_correction_entry((*ss.offset(-4)).continuation_correction_history)
                    .get_unchecked(pc)
                    .get_unchecked(sq)
                    .get() as i32;
                let cnt6 = self
                    .continuation_correction_entry((*ss.offset(-6)).continuation_correction_history)
                    .get_unchecked(pc)
                    .get_unchecked(sq)
                    .get() as i32;
                7885 * (cnt2 + cnt4) + 6307 * cnt6
            } else {
                80695
            };

            13806 * pcv + 9512 * micv + 11615 * (wnpcv + bnpcv) + cntcv
        }
    }

    // Update correction history
    // NOT inlined to match C++ (update_correction_history is a separate function call)
    #[inline(never)]
    /// # Safety
    /// `ss` must be a valid Stack entry of the current search with the padding/ancestor entries before it (see the SAFETY note in the body).
    unsafe fn update_correction_history(&mut self, pos: &Position, ss: *const Stack, bonus: i32) {
        // SAFETY: same invariants as `correction_value`: `ss` has 6 valid entries before it, pc < 16, sq < 64.
        // Only the continuation correction tables are per-worker, and no other reference to them is live here.
        unsafe {
            let m = (*ss.offset(-1)).current_move;
            let shared = &self.shared_state.shared_histories;

            const NON_PAWN_WEIGHT: i32 = 186;

            // Use shared correction histories with atomic updates
            shared.pawn_correction_entry(pos).pawn.update(bonus);
            shared
                .minor_piece_correction_entry(pos)
                .minor
                .update(bonus * 150 / 128);
            shared
                .nonpawn_correction_entry_white(pos)
                .non_pawn_white
                .update(bonus * NON_PAWN_WEIGHT / 128);
            shared
                .nonpawn_correction_entry_black(pos)
                .non_pawn_black
                .update(bonus * NON_PAWN_WEIGHT / 128);

            // Continuation correction history remains per-worker
            // SAFETY: pc is Piece (0-15), sq is Square (0-63)
            if m.is_ok() {
                let pc = pos.piece_on(m.to_sq()) as usize;
                let sq = m.to_sq() as usize;
                self.continuation_correction_entry(
                    (*ss.offset(-2)).continuation_correction_history,
                )
                .get_unchecked(pc)
                .get_unchecked(sq)
                .update(bonus * 130 / 128);
                self.continuation_correction_entry(
                    (*ss.offset(-4)).continuation_correction_history,
                )
                .get_unchecked(pc)
                .get_unchecked(sq)
                .update(bonus * 70 / 128);
                self.continuation_correction_entry(
                    (*ss.offset(-6)).continuation_correction_history,
                )
                .get_unchecked(pc)
                .get_unchecked(sq)
                .update(bonus * 35 / 128);
            }
        }
    }

    // Update quiet move histories - matches C++ search.cpp
    #[inline(always)]
    /// # Safety
    /// `ss` must be a valid Stack entry of the current search with the padding/ancestor entries before it (see the SAFETY note in the body).
    unsafe fn update_quiet_histories(
        &mut self,
        pos: &Position,
        ss: *const Stack,
        mv: Move,
        bonus: i32,
    ) {
        // SAFETY: us is 0 or 1 and raw() is a u16 (tables are [2][65536]); ply is checked < LOW_PLY_HISTORY_SIZE; piece < 16
        // and to < 64 for the pawn entry; `ss` is a valid Stack entry.
        unsafe {
            let us = pos.side_to_move();
            // SAFETY: us is 0 or 1, raw() is u16 (0-65535)
            self.main_history
                .get_unchecked_mut(us as usize)
                .get_unchecked_mut(mv.raw() as usize)
                .update(bonus);

            // SAFETY: ply is checked < LOW_PLY_HISTORY_SIZE, raw() is u16
            if (*ss).ply < LOW_PLY_HISTORY_SIZE as i32 {
                self.low_ply_history
                    .get_unchecked_mut((*ss).ply as usize)
                    .get_unchecked_mut(mv.raw() as usize)
                    .update(bonus * 712 / 1024);
            }

            self.update_continuation_histories(
                ss,
                pos.moved_piece(mv),
                mv.to_sq(),
                bonus * 750 / 1024,
            );

            // SAFETY: piece is 0-15, square is 0-63
            {
                self.shared_state
                    .shared_histories
                    .pawn_entry(pos)
                    .get_unchecked(pos.moved_piece(mv) as usize)
                    .get_unchecked(mv.to_sq() as usize)
                    .update(bonus * if bonus > -4 { 1104 } else { 459 } / 1024);
            }
        }
    }

    // Update continuation histories
    // NOT inlined to match C++ (update_continuation_histories is a separate function call)
    /// Updates the continuation histories for the move pairs formed by
    /// the current move and the moves played in previous plies.
    #[inline(never)]
    fn update_continuation_histories(
        &mut self,
        ss: *const Stack,
        pc: Piece,
        to: Square,
        bonus: i32,
    ) {
        // SAFETY: `ss` has 6 valid entries before it (7 padding entries at ply 0), only entries with `current_move.is_ok()`
        // are used, and their `continuation_history` pointers were set by do_move/do_null_move into tables that
        // outlive the stack; pc < 16 and to < 64. Atomic history updates use shared references.
        unsafe {
            const CONTHIST_BONUSES: [(usize, i32); 5] =
                [(1, 520), (2, 390), (3, 145), (4, 251), (6, 209)];

            // Multipliers for positive history consistency
            const CMHC_MULTIPLIERS: [i32; 6] = [94, 103, 110, 106, 119, 126];
            let mut positive_count: usize = 0;

            for &(i, weight) in &CONTHIST_BONUSES {
                if (*ss).in_check && i > 2 {
                    break;
                }
                if (*ss.offset(-(i as isize))).current_move.is_ok() {
                    // SAFETY: pc is Piece (0-15), to is Square (0-63)
                    let entry = (*(*ss.offset(-(i as isize))).continuation_history)
                        .get_unchecked(pc as usize)
                        .get_unchecked(to as usize);
                    if entry.get() > 0 {
                        positive_count += 1;
                    }
                    let multiplier = CMHC_MULTIPLIERS[positive_count];
                    entry.update((bonus * weight * multiplier / 65536) + 73 * (i < 2) as i32);
                }
            }
        }
    }

    // Update all stats after finding best move
    // NOT inlined to match C++ (update_all_stats is a separate function call)
    /// Updates stats at the end of search() when a bestMove is found
    #[inline(never)]
    fn update_all_stats(
        &mut self,
        pos: &Position,
        ss: *const Stack,
        best_move: Move,
        prev_sq: Option<Square>,
        quiets_searched: &SearchedList,
        captures_searched: &SearchedList,
        depth: Depth,
        tt_move: Move,
    ) {
        // SAFETY: `ss` is a valid Stack entry with 1 entry before it; the history tables are indexed by moved/captured piece
        // (< 16), square (< 64) and PieceType (< 8); `ss-1` is passed on, and its reads down to ss-7 are covered by the 7 padding entries.
        unsafe {
            let moved_piece = pos.moved_piece(best_move);
            let captured_piece: PieceType;

            let mut bonus = min(133 * depth - 81, 1487)
                + 364 * (best_move == tt_move) as i32
                + (*ss.offset(-1)).stat_score / 28;
            let malus = min(968 * depth - 235, 2244);

            // Important: don't remove the cast to a 64-bit number, else the
            // multiplication can overflow on 32-bit platforms which would change
            // the bench signature.
            bonus += ((bonus as u64)
                .wrapping_mul((quiets_searched.len() + captures_searched.len()) as u64)
                / 256) as i32;

            if !pos.capture_stage(best_move) {
                self.update_quiet_histories(pos, ss, best_move, bonus * 899 / 1024);

                // Decrease stats for all non-best quiet moves
                let mut actual_malus = malus * 1159 / 1024;
                for &mv in quiets_searched.into_iter() {
                    actual_malus = actual_malus * 921 / 1024;
                    self.update_quiet_histories(pos, ss, mv, -actual_malus);
                }
            } else {
                // Increase stats for the best move in case it was a capture move
                captured_piece = piece_type(pos.piece_on(best_move.to_sq()));
                // SAFETY: moved_piece is Piece (0-15), to_sq is Square (0-63), captured_piece is PieceType (0-7)
                self.capture_history
                    .get_unchecked_mut(moved_piece as usize)
                    .get_unchecked_mut(best_move.to_sq() as usize)
                    .get_unchecked_mut(captured_piece as usize)
                    .update(bonus * 1427 / 1024);
            }

            // Extra penalty for a quiet early move that was not a TT move in the
            // previous ply when it gets refuted.
            if let Some(prev_sq) = prev_sq
                && (*ss.offset(-1)).move_count == 1 + (*ss.offset(-1)).tt_hit as i32
                && pos.captured_piece() == NO_PIECE
            {
                self.update_continuation_histories(
                    ss.offset(-1),
                    pos.piece_on(prev_sq),
                    prev_sq,
                    -malus * 713 / 1024,
                );
            }

            // Decrease stats for all non-best capture moves
            for &mv in captures_searched {
                let mp = pos.moved_piece(mv);
                let cp = piece_type(pos.piece_on(mv.to_sq()));
                // SAFETY: mp is Piece (0-15), to_sq is Square (0-63), cp is PieceType (0-7)
                self.capture_history
                    .get_unchecked_mut(mp as usize)
                    .get_unchecked_mut(mv.to_sq() as usize)
                    .get_unchecked_mut(cp as usize)
                    .update(-malus * 1489 / 1024);
            }
        }
    }

    // Time management helpers
    /// elapsed() returns the time elapsed since the search started. If the
    /// 'nodestime' option is enabled, it will return the count of nodes searched
    /// instead. This function is called to check whether the search should be
    /// stopped based on predefined thresholds like time limits or nodes searched.
    fn elapsed(&self) -> u64 {
        // Get nodes_searched value first to avoid nested lock acquisition
        let nodes = self.shared_state.threads.read().unwrap().nodes_searched() as i64;
        self.main_manager().elapsed(|| nodes)
    }

    pub fn elapsed_time(&self) -> u64 {
        // Returns elapsed time in milliseconds (always time, never nodes)
        self.main_manager().elapsed_time()
    }
}

// Helper functions

/// Add a small random component to draw evaluations to avoid 3-fold blindness
#[inline(always)]
fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 0x2) as Value
}

/// Adjusts a mate or TB score from "plies to mate from the root" to
/// "plies to mate from the current position". Standard scores are unchanged.
/// The function is called before storing a value in the transposition table.
#[inline(always)]
fn value_to_tt(v: Value, ply: Ply) -> Value {
    if is_win(v) {
        v + ply
    } else if is_loss(v) {
        v - ply
    } else {
        v
    }
}

/// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
/// table (which refers to the plies to mate/be mated from current position) to
/// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
/// potentially false mate or TB scores related to the 50 moves rule and the
/// graph history interaction, we return the highest non-TB score instead.
#[inline(always)]
fn value_from_tt(v: Value, ply: Ply, r50c: i32) -> Value {
    if !is_valid(v) {
        return VALUE_NONE;
    }

    if is_win(v) {
        if v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - r50c {
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }

        if VALUE_TB - v > 100 - r50c {
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }

        return v - ply;
    }

    if is_loss(v) {
        if v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - r50c {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }

        if VALUE_TB + v > 100 - r50c {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }

        return v + ply;
    }

    v
}

/// # Safety
/// `pv` must point at a writable buffer of MAX_PLY + 1 Moves with room for `1 + len(child_pv) + 1`
/// entries, and `child_pv` must be null or point at a `Move::none()`-terminated sequence.
unsafe fn update_pv(pv: *mut Move, mv: Move, child_pv: *const Move) {
    // SAFETY: `pv` points at a buffer of MAX_PLY + 1 Moves and `child_pv` (if non-null) at the child's buffer, which the child
    // terminated with Move::none(); the copy writes at most the child's length + 2 entries. ASSUMPTION: the child PV
    // is shorter than the parent's remaining capacity (as in C++), which this function does not itself check.
    unsafe {
        *pv = mv;
        let mut pv = pv.offset(1);

        if !child_pv.is_null() {
            let mut cpv = child_pv;
            while *cpv != Move::none() {
                *pv = *cpv;
                pv = pv.offset(1);
                cpv = cpv.offset(1);
            }
        }
        *pv = Move::none();
    }
}

fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

#[inline(always)]
fn is_mate(v: Value) -> bool {
    v >= VALUE_MATE_IN_MAX_PLY
}

#[inline(always)]
fn is_mated(v: Value) -> bool {
    v <= VALUE_MATED_IN_MAX_PLY
}

#[inline(always)]
fn is_mate_or_mated(v: Value) -> bool {
    is_mate(v) || is_mated(v)
}

// Linear interpolation matching C++ misc.h interpolate():
// y0 + (y1 - y0) * (x - x0) / (x1 - x0)
#[inline(always)]
fn interpolate(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

fn is_loss(v: Value) -> bool {
    v <= VALUE_TB_LOSS_IN_MAX_PLY
}

fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

fn mate_in(ply: Ply) -> Value {
    VALUE_MATE - ply
}

fn mated_in(ply: Ply) -> Value {
    -VALUE_MATE + ply
}

/// Add correctionHistory value to raw staticEval and guarantee evaluation
/// does not hit the tablebase range.
fn to_corrected_static_eval(unadjusted: Value, correction: Value) -> Value {
    (unadjusted + correction / 131072)
        .clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

fn piece_value(piece: Piece) -> Value {
    const PIECE_VALUES: [Value; 8] = [
        0,
        PAWN_VALUE,
        KNIGHT_VALUE,
        BISHOP_VALUE,
        ROOK_VALUE,
        QUEEN_VALUE,
        0,
        0,
    ];
    // SAFETY: piece_type returns 0-7, PIECE_VALUES has 8 elements
    unsafe { *PIECE_VALUES.get_unchecked(piece_type(piece) as usize) }
}

fn piece_type(piece: Piece) -> PieceType {
    PieceType::from((piece as u8) & 7)
}

/// Returns true when the PV-extension time budget is exhausted: under time management,
/// 2 * MultiPV * elapsed ms >= Move Overhead. Never true under nodes-time.
#[inline]
fn abort_syzygy_extension(
    limits: &LimitsType,
    multi_pv: usize,
    elapsed_ms: u128,
    move_overhead: u64,
) -> bool {
    limits.npmsec == 0
        && limits.use_time_management()
        && 2 * multi_pv.max(1) as u128 * elapsed_ms >= move_overhead as u128
}

/// Used to correct and extend PVs for moves that have a TB (but not a mate) score.
/// Keeps the search based PV for as long as it is verified to maintain the game
/// outcome, truncates afterwards. Finally, extends to mate the PV, providing a
/// possible continuation (but not a proven mating line).
pub fn syzygy_extend_pv(
    options: &OptionsMap,
    limits: &LimitsType,
    root: &Position,
    root_move: &mut RootMove,
    score: &mut Value,
    multi_pv: usize,
) {
    use std::time::Instant;

    let t_start = Instant::now();
    let move_overhead = options
        .get("Move Overhead")
        .and_then(|o| o.get_int())
        .unwrap_or(10) as u64;
    let rule50 = options
        .get("Syzygy50MoveRule")
        .and_then(|o| o.get_int())
        .map(|v| v != 0)
        .unwrap_or(true);

    // Budget: about moveOverhead / (2 * MultiPV) ms, if time management is active.
    // Under nodes-time the local Position::do_move calls come for free.
    let time_abort = || {
        abort_syzygy_extension(
            limits,
            multi_pv,
            t_start.elapsed().as_millis(),
            move_overhead,
        )
    };

    if time_abort() {
        return;
    }

    let Some(&root_mv) = root_move.pv.first() else {
        return;
    };
    if !root.legal(root_mv) {
        return;
    }

    // Walk the PV on a copy that is never undone: the PV may exceed the
    // MAX_HIST states a Position keeps.
    let mut walker = root.clone();
    let pos = &mut walker;

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV in TB
    // SAFETY: root legality is checked above.
    unsafe { pos.do_move(root_mv) };
    let mut ply = 1;

    // Step 1: Walk the PV to the last position in TB with correct decisive score
    while ply < root_move.pv.len() {
        let pv_move = root_move.pv[ply];

        let mut legal_moves = RootMoves::new();
        for m in generate(pos, GenType::All) {
            legal_moves.push(RootMove::new(m));
        }

        let config = rank_root_moves(options, pos, &mut legal_moves, false);
        let rm = legal_moves.iter().find(|m| m.pv[0] == pv_move);

        if let Some(rm) = rm {
            if legal_moves[0].tb_rank != rm.tb_rank {
                break;
            }
        } else {
            break;
        }

        ply += 1;

        // SAFETY: pv_move was verified against generated legal moves above.
        unsafe { pos.do_move(pv_move) };

        // Do not allow repetitions or drawing moves along the PV in TB regime
        if config.root_in_tb
            && ((rule50 && pos.is_draw(ply as i32)) || pos.is_repetition(ply as i32))
        {
            // SAFETY: pv_move is the last move made on the walker.
            unsafe { pos.undo_move(pv_move) };
            ply -= 1;
            break;
        }

        // If we cannot validate the full PV in time, we do not show it
        if config.root_in_tb && time_abort() {
            break;
        }
    }

    // Resize the PV to the correct part
    root_move.pv.resize(ply, Move::none());

    // Step 2: Extend the PV to mate using DTZ-optimal moves
    // Step 2, now extend the PV to mate, as if the user explored syzygy-tables.info
    // using top ranked moves (minimal DTZ), which gives optimal mates only for simple
    // endgames e.g. KRvK.
    while !(rule50 && pos.is_draw(0)) {
        if time_abort() {
            break;
        }

        let mut legal_moves = RootMoves::new();
        for m in generate(pos, GenType::All) {
            let mut rm = RootMove::new(m);
            // SAFETY: m is generated as legal; the scope undoes it, including if probing
            // unwinds.
            let mut moved = unsafe { crate::position::ScopedMove::new(pos, m) };
            let pos = &mut *moved;

            // Give a score of each move to break DTZ ties restricting opponent
            // mobility, but not giving the opponent a capture.
            for opp_m in generate(pos, GenType::All) {
                rm.tb_rank -= if pos.capture(opp_m) { 100 } else { 1 };
            }
            drop(moved);
            legal_moves.push(rm);
        }

        // No legal reply: the PV ends in mate or stalemate.
        if legal_moves.is_empty() {
            break;
        }

        // Sort moves according to their assigned rank
        // This will break ties for moves with equal DTZ
        legal_moves.sort_by_key(|m| core::cmp::Reverse(m.tb_rank));

        // The winning side tries to minimize DTZ, the losing side maximizes it
        let config = rank_root_moves(options, pos, &mut legal_moves, true);

        // If DTZ is not available we might not find a mate, so we bail out
        if !config.root_in_tb || config.cardinality > 0 {
            break;
        }

        let pv_move = legal_moves[0].pv[0];
        root_move.pv.push(pv_move);
        // SAFETY: pv_move was verified against generated legal moves above.
        unsafe { pos.do_move(pv_move) };
    }

    // Handle exceptional case where position is draw due to 50-move rule
    // This can happen if reached with non-optimal 50-move counter
    // Finding a draw in this function is an exceptional case, that cannot happen when rule50 is false or
    // during engine game play, since we have a winning score, and play correctly
    // with TB support. However, it can be that a position is draw due to the 50 move
    // rule if it has been reached on the board with a non-optimal 50 move counter
    // (e.g. 8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106 ) which TB with dtz counter rounding
    // cannot always correctly rank. See also
    // https://github.com/official-stockfish/Stockfish/issues/5175#issuecomment-2058893495
    // We adjust the score to match the found PV. Note that a TB loss score can be
    // displayed if the engine did not find a drawing move yet, but eventually search
    // will figure it out (e.g. 1kq5/q2r4/5K2/8/8/8/8/7Q w - - 96 1 )
    if pos.is_draw(0) {
        *score = VALUE_DRAW;
    }

    // Inform if we couldn't get a full extension in time
    if time_abort() {
        println!(
            "info string Syzygy based PV extension requires more time, increase Move Overhead as needed."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reductions_table_matches_formula() {
        assert_eq!(REDUCTIONS[0], 0);
        for (i, &r) in REDUCTIONS.iter().enumerate().skip(1) {
            assert_eq!(
                r,
                (2872.0 / 128.0 * (i as f64).ln()) as i32,
                "reductions[{i}]"
            );
        }
    }

    #[test]
    fn syzygy_extension_budget_handles_nodes_time_and_multipv() {
        let mut limits = LimitsType {
            time: [1000, 1000],
            ..LimitsType::default()
        };

        assert!(!abort_syzygy_extension(&limits, 4, 4, 40));
        assert!(abort_syzygy_extension(&limits, 4, 5, 40));

        limits.npmsec = 1000;
        assert!(!abort_syzygy_extension(&limits, 4, u128::MAX / 16, 40));
    }
}
