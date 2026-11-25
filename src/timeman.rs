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

//! Time management logic port – Rust version of C++ TimeManagement

use crate::misc::now;
use crate::search::LimitsType;
use crate::types::Color;
use crate::ucioption::OptionsMap;
use core::cmp;

/// Type alias for time in milliseconds (or nodes count when in nodes-as-time mode).
pub type TimePoint = i64;

#[inline]
fn maximum_move_horizon(limits: &LimitsType, scaled_time: TimePoint) -> TimePoint {
    let mut moves = if limits.movestogo != 0 {
        i64::from(cmp::min(limits.movestogo, 50))
    } else {
        50
    };

    // Cyclic controls must retain their real horizon even below one second;
    // shortening it there can spend too much and lose on time at the boundary.
    if scaled_time < 1000 && limits.movestogo == 0 {
        moves = (scaled_time as f64 * 0.05) as TimePoint;
    }
    moves
}

/// Computes optimal and maximum thinking time for a move based on remaining time,
/// move number, increments, and engine options.
#[derive(Debug)]
pub struct TimeManagement {
    start_time: TimePoint,
    optimum_time: TimePoint,
    maximum_time: TimePoint,
    // Related to 'nodes as time' mode:
    use_nodes_time: bool,
    available_nodes: i64,
    previous_moves_to_go: i32,
    cyclic_budget: i64,
}

/// The TimeManagement class computes the optimal time to think depending on
/// the maximum available time, the game move number, and other parameters.
impl TimeManagement {
    /// Sentinel used when no finite time bound was supplied for this side.
    pub(crate) const NO_BOUND: TimePoint = i64::MAX / 2;

    /// Create a default TimeManagement (uninitialized).
    pub fn new() -> Self {
        TimeManagement {
            start_time: 0,
            optimum_time: Self::NO_BOUND,
            maximum_time: Self::NO_BOUND,
            use_nodes_time: false,
            available_nodes: -1,
            previous_moves_to_go: 0,
            cyclic_budget: 0,
        }
    }

    /// Initialize based on limits, side to move, ply, options, and original time adjust.
    /// If `*original_time_adjust` is negative on entry it is computed from the time left and
    /// written back so later calls reuse it.
    ///
    /// Called at the beginning of the search and calculates
    /// the bounds of time allowed for the current game ply. We currently support:
    /// 1) x basetime (+ z increment)
    /// 2) x moves in y seconds (+ z increment)
    pub fn init(
        &mut self,
        limits: &mut LimitsType,
        us: Color,
        ply: i32,
        options: &OptionsMap,
        original_time_adjust: &mut f64,
    ) {
        // Option accessors; a missing or non-integer option reads as 0 / false.
        let opt_int = |name: &str| -> i64 {
            options
                .get(name)
                .and_then(|o| o.get_int())
                .map(|v| v as i64)
                .unwrap_or(0)
        };
        let opt_bool = |name: &str| -> bool {
            options
                .get(name)
                .and_then(|o| o.get_int())
                .map(|v| v != 0)
                .unwrap_or(false)
        };

        let npmsec = opt_int("nodestime") as TimePoint;
        // If we have no time, we don't need to fully initialize TM.
        // startTime is used by movetime and useNodesTime is used in elapsed calls.
        self.start_time = limits.start_time;
        self.use_nodes_time = npmsec != 0;

        // elapsed() reports nodes while nodestime is active, so movetime must
        // use the same unit before the search starts.
        if self.use_nodes_time {
            limits.movetime = limits.movetime.saturating_mul(npmsec);
        }

        if limits.time[us as usize] == 0 {
            self.optimum_time = Self::NO_BOUND;
            self.maximum_time = Self::NO_BOUND;
            return;
        }

        let mut move_overhead = opt_int("Move Overhead") as TimePoint;
        if self.use_nodes_time {
            if self.available_nodes == -1 {
                // First time limit includes increment (both are in milliseconds)
                self.available_nodes = npmsec.saturating_mul(limits.time[us as usize]);
                self.cyclic_budget = npmsec.saturating_mul(
                    limits.time[us as usize].saturating_sub(limits.inc[us as usize]),
                );
            } else if limits.movestogo > 0
                && limits.movestogo > self.previous_moves_to_go
                && self.cyclic_budget > 0
            {
                self.available_nodes = self.available_nodes.saturating_add(self.cyclic_budget);
            }

            self.previous_moves_to_go = limits.movestogo;

            limits.time[us as usize] = self.available_nodes;
            limits.inc[us as usize] = limits.inc[us as usize].saturating_mul(npmsec);
            limits.npmsec = npmsec;
            move_overhead = move_overhead.saturating_mul(npmsec);
        }

        // These numbers are used where multiplications, divisions,
        // or comparisons with constants are involved.
        let scale_factor = if self.use_nodes_time { npmsec } else { 1 };
        // Clamp from below to 1: `scaled_time` feeds `log10()` below, and a
        // zero would produce -inf and, ultimately, an out-of-range float cast.
        let scaled_time = cmp::max(1, limits.time[us as usize] / scale_factor);

        // Maximum move horizon
        let mtg = maximum_move_horizon(limits, scaled_time);

        // Make sure timeLeft is > 0 since we may use it as a divisor
        let time_left = cmp::max(
            1,
            limits.time[us as usize]
                .saturating_add(limits.inc[us as usize].saturating_mul(mtg - 1))
                .saturating_sub(move_overhead.saturating_mul(2 + mtg)),
        );

        let (opt_scale, max_scale) = if limits.movestogo == 0 {
            if *original_time_adjust < 0.0 {
                *original_time_adjust = 0.3272 * (time_left as f64).log10() - 0.4141;
            }
            // `opt_scale` is the percentage of the available time to spend on
            // this move; `max_scale` is a multiplier applied to `optimum_time`.
            // Calculate time constants based on current time left.
            let log_time_sec = (scaled_time as f64 / 1000.0).log10();
            let opt_const = f64::min(0.0029869 + 0.00033554 * log_time_sec, 0.004905);
            let max_const = f64::max(3.3744 + 3.0608 * log_time_sec, 3.1441);

            let opt_scale = f64::min(
                0.012112 + (ply as f64 + 3.22713).powf(0.46866) * opt_const,
                0.19404 * (limits.time[us as usize] as f64) / (time_left as f64),
            ) * (*original_time_adjust);

            let max_scale = f64::min(6.873, max_const + ply as f64 / 12.352);
            (opt_scale, max_scale)
        } else {
            let opt_scale = f64::min(
                (0.88 + ply as f64 / 116.4) / (mtg as f64),
                0.88 * (limits.time[us as usize] as f64) / (time_left as f64),
            );
            let max_scale = 1.3 + 0.11 * (mtg as f64);
            (opt_scale, max_scale)
        };

        // Decrease time usage if behind in time.
        // This is skipped in two cases:
        // - if the nodestime option is used we can't calculate the opponent nodes budget in
        //   a deterministic way.
        // - if we use a cyclic time management (like 40/10) calculating time advantage for
        //   the last move (movestogo = 1) can be vastly off, because if the opponent had done
        //   his last move before us his time budget includes already the next cycle time
        //   increment but our not. This leads to a unnecessary big decrease in time usage
        //   which favors blunders.
        // Warning: don't remove this conditions.
        let opt_scale = if !self.use_nodes_time && limits.movestogo != 1 {
            let them = !us;
            let time_advantage = (limits.time[us as usize] as f64
                - limits.time[them as usize] as f64)
                / (1.0 + limits.time[us as usize] as f64 + limits.time[them as usize] as f64);
            opt_scale * (1.0 + 0.9 * f64::min(time_advantage, 0.0))
        } else {
            opt_scale
        };

        // Limit the maximum possible time for this move
        self.optimum_time = f64::max(1.0, opt_scale * (time_left as f64)) as TimePoint;
        self.maximum_time = f64::max(
            self.optimum_time as f64,
            f64::min(
                0.8097 * (limits.time[us as usize] as f64) - move_overhead as f64,
                max_scale * (self.optimum_time as f64),
            ),
        ) as TimePoint;

        if opt_bool("Ponder") {
            self.optimum_time += self.optimum_time / 4;
        }
    }

    /// Get the optimum time.
    pub fn optimum(&self) -> TimePoint {
        self.optimum_time
    }

    /// Get the maximum time.
    pub fn maximum(&self) -> TimePoint {
        self.maximum_time
    }

    /// True when the `nodestime` option is active, i.e. when [`Self::elapsed`]
    /// measures consumed nodes instead of wall-clock milliseconds.
    #[inline]
    pub fn uses_nodes_time(&self) -> bool {
        self.use_nodes_time
    }

    /// Milliseconds since start, or the value of `nodes()` when nodestime is active. `nodes` is
    /// only called in nodes mode.
    pub fn elapsed<F>(&self, nodes: F) -> TimePoint
    where
        F: Fn() -> i64,
    {
        if self.use_nodes_time {
            nodes()
        } else {
            now() - self.start_time
        }
    }

    /// Forget the nodes budget so the next init() starts a fresh one.
    pub fn clear(&mut self) {
        self.available_nodes = -1;
        self.previous_moves_to_go = 0;
    }

    /// Elapsed time since start (always returns time, not nodes).
    pub fn elapsed_time(&self) -> TimePoint {
        now() - self.start_time
    }

    /// Advance nodes-time by subtracting consumed nodes.
    pub fn advance_nodes_time(&mut self, nodes: i64) {
        assert!(self.use_nodes_time);
        self.available_nodes = cmp::max(0, self.available_nodes - nodes);
    }
}

impl Default for TimeManagement {
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
    use crate::search::LimitsType;
    use crate::types::Color;
    use crate::ucioption::{OptionsMap, UciOption};

    // Helper function to create an OptionsMap with specific values
    fn create_options_map(values: Vec<(&str, i32)>) -> OptionsMap {
        let mut options = OptionsMap::new();
        for (name, value) in values {
            let opt = UciOption::new_spin(name, value, 0, 10000, None);
            options.add(opt);
        }
        options
    }

    // Helper function to create default limits
    fn create_default_limits() -> LimitsType {
        LimitsType {
            searchmoves: vec![],
            time: [60000, 60000], // 60 seconds each
            inc: [1000, 1000],    // 1 second increment
            npmsec: 0,
            movetime: 0,
            start_time: 1000,
            movestogo: 0,
            depth: 255,
            mate: 0,
            perft: 0,
            infinite: 0,
            nodes: 0,
            ponder_mode: false,
        }
    }

    // ========== Basic Functionality Tests ==========

    #[test]
    fn test_new_creates_default_instance() {
        let tm = TimeManagement::new();
        assert_eq!(tm.optimum(), TimeManagement::NO_BOUND);
        assert_eq!(tm.maximum(), TimeManagement::NO_BOUND);
        assert_eq!(tm.available_nodes, -1);
        assert!(!tm.use_nodes_time);
    }

    #[test]
    fn test_default_trait_implementation() {
        let tm = TimeManagement::default();
        assert_eq!(tm.optimum(), TimeManagement::NO_BOUND);
        assert_eq!(tm.maximum(), TimeManagement::NO_BOUND);
    }

    #[test]
    fn test_clear_resets_available_nodes() {
        let mut tm = TimeManagement::new();
        tm.available_nodes = 5000;
        tm.clear();
        assert_eq!(tm.available_nodes, -1);
    }

    // ========== Time Control Mode Tests ==========

    #[test]
    fn test_standard_time_control_white() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > tm.optimum());
        assert!(!tm.use_nodes_time);
        assert!(time_adjust > 0.0); // Should be calculated
    }

    #[test]
    fn test_standard_time_control_black() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::Black, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > tm.optimum());
        assert!(!tm.use_nodes_time);
    }

    #[test]
    fn test_tournament_time_control() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.movestogo = 40; // 40 moves in time control
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > tm.optimum());
        assert_eq!(time_adjust, -1.0); // Should not be modified in tournament mode
    }

    #[test]
    fn test_increment_time_control() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [5000, 5000]; // 5 seconds
        limits.inc = [3000, 3000]; // 3 second increment
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > 0);
    }

    #[test]
    fn test_no_time_early_return() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [0, 0]; // No time available
        let options = create_options_map(vec![]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert_eq!(tm.optimum(), TimeManagement::NO_BOUND);
        assert_eq!(tm.maximum(), TimeManagement::NO_BOUND);
    }

    // ========== Nodes-as-Time Mode Tests ==========

    #[test]
    fn test_nodes_as_time_mode_initialization() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [10000, 10000]; // 10 seconds
        let options = create_options_map(vec![("nodestime", 1000)]); // 1000 nodes per ms
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.use_nodes_time);
        assert_eq!(tm.available_nodes, 10000 * 1000); // time * npmsec
        assert_eq!(limits.npmsec, 1000);
    }

    #[test]
    fn test_nodes_as_time_mode_reuse() {
        let mut tm = TimeManagement::new();
        tm.available_nodes = 5000000; // Already set from previous search

        let mut limits = create_default_limits();
        limits.time = [10000, 10000];
        let options = create_options_map(vec![("nodestime", 1000)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert_eq!(tm.available_nodes, 5000000); // Should not be reset
        assert_eq!(limits.time[Color::White as usize], 5000000); // Should use available_nodes
    }

    #[test]
    fn test_nodes_as_time_cyclic_budget_tops_up_on_new_cycle() {
        // Upstream 9287f165: in `nodestime` mode a cyclic control (e.g. 40/10)
        // hands back a fresh time budget every cycle, which the wall-clock path
        // gets for free from the GUI but the node budget has to add itself.
        // The tell that a new cycle started is `movestogo` increasing.
        let mut tm = TimeManagement::new();
        let options = create_options_map(vec![("nodestime", 1000)]);
        let mut time_adjust = -1.0;

        // Game start: 10s + 1s increment at 1000 nodes/ms.
        let mut limits = create_default_limits();
        limits.time = [10000, 10000];
        limits.inc = [1000, 1000];
        limits.movestogo = 2;
        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);
        assert_eq!(tm.available_nodes, 10_000_000);
        assert_eq!(tm.cyclic_budget, 9_000_000, "budget excludes the increment");

        // Same cycle, one move further in: movestogo counts down, no top-up.
        tm.advance_nodes_time(1_000_000);
        let mut limits = create_default_limits();
        limits.movestogo = 1;
        tm.init(&mut limits, Color::White, 12, &options, &mut time_adjust);
        assert_eq!(tm.available_nodes, 9_000_000);

        // New cycle: movestogo jumps back up, so the cycle budget is added.
        tm.advance_nodes_time(1_000_000);
        let mut limits = create_default_limits();
        limits.movestogo = 2;
        tm.init(&mut limits, Color::White, 14, &options, &mut time_adjust);
        assert_eq!(tm.available_nodes, 8_000_000 + 9_000_000);
    }

    #[test]
    fn test_time_disadvantage_reduces_optimum() {
        // Upstream 5ca4bfe7: spend less when behind on the clock. Skipped in
        // `nodestime` mode and on the last move of a cycle (movestogo == 1).
        let options = create_options_map(vec![("nodestime", 0)]);

        let optimum_for = |ours: i64, theirs: i64, movestogo: i32| {
            let mut tm = TimeManagement::new();
            let mut limits = create_default_limits();
            limits.time = [ours, theirs];
            limits.movestogo = movestogo;
            let mut time_adjust = -1.0;
            tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);
            tm.optimum()
        };

        let even = optimum_for(60000, 60000, 0);
        let behind = optimum_for(60000, 120000, 0);
        let ahead = optimum_for(120000, 60000, 0);

        assert!(
            behind < even,
            "behind {} should spend less than even {}",
            behind,
            even
        );
        assert_eq!(
            optimum_for(120000, 120000, 0),
            ahead,
            "being ahead must not change anything (min(advantage, 0) == 0)"
        );
        assert_eq!(
            optimum_for(60000, 120000, 1),
            optimum_for(60000, 60000, 1),
            "movestogo == 1 is exempt: the opponent's clock may already hold the next cycle"
        );
    }

    #[test]
    fn test_nodes_as_time_mode_with_fewer_nodes_than_npmsec() {
        // Upstream 6e4c77f5: once `available_nodes` drops below `npmsec`, the
        // integer division makes the scaled time zero. Unclamped, its log10()
        // is -inf and the resulting scales are non-finite, so the computed
        // times are garbage. `scaled_time` is clamped to 1 to prevent this.
        let mut tm = TimeManagement::new();
        tm.available_nodes = 500; // Fewer than one millisecond's worth
        let mut limits = create_default_limits();
        limits.time = [10000, 10000];
        let options = create_options_map(vec![("nodestime", 1000)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() >= 1, "optimum must stay positive");
        assert!(
            tm.optimum() <= tm.maximum(),
            "optimum {} must not exceed maximum {}",
            tm.optimum(),
            tm.maximum()
        );
        assert!(
            tm.maximum() < TimePoint::MAX,
            "maximum must be finite, got {}",
            tm.maximum()
        );
    }

    #[test]
    fn test_elapsed_in_nodes_mode() {
        let mut tm = TimeManagement::new();
        tm.use_nodes_time = true;
        tm.start_time = 1000;

        let elapsed = tm.elapsed(|| 999);
        assert_eq!(elapsed, 999);
    }

    #[test]
    fn test_elapsed_in_time_mode() {
        let mut tm = TimeManagement::new();
        tm.use_nodes_time = false;
        tm.start_time = now() - 100; // 100ms ago

        let elapsed = tm.elapsed(|| 999);
        assert!((100..=200).contains(&elapsed)); // Allow some tolerance
    }

    #[test]
    fn test_advance_nodes_time() {
        let mut tm = TimeManagement::new();
        tm.use_nodes_time = true;
        tm.available_nodes = 10000;

        tm.advance_nodes_time(3000);
        assert_eq!(tm.available_nodes, 7000);

        tm.advance_nodes_time(8000); // Would go negative
        assert_eq!(tm.available_nodes, 0); // Clamped to 0
    }

    #[test]
    #[should_panic(expected = "use_nodes_time")]
    fn test_advance_nodes_time_panics_when_not_in_nodes_mode() {
        let mut tm = TimeManagement::new();
        tm.use_nodes_time = false;
        tm.advance_nodes_time(1000);
    }

    // ========== Ponder Mode Tests ==========

    #[test]
    fn test_ponder_mode_increases_optimum_time() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let options = create_options_map(vec![("Move Overhead", 30), ("Ponder", 1)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);
        let optimum_with_ponder = tm.optimum();

        // Initialize without ponder
        let mut tm_no_ponder = TimeManagement::new();
        let options_no_ponder = create_options_map(vec![("Move Overhead", 30), ("Ponder", 0)]);
        let mut time_adjust_no_ponder = -1.0;

        tm_no_ponder.init(
            &mut limits,
            Color::White,
            10,
            &options_no_ponder,
            &mut time_adjust_no_ponder,
        );
        let optimum_without_ponder = tm_no_ponder.optimum();

        assert_eq!(
            optimum_with_ponder,
            optimum_without_ponder + optimum_without_ponder / 4
        );
    }

    // ========== Edge Cases and Boundary Tests ==========

    #[test]
    fn test_very_low_time_remaining() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [500, 500]; // 500ms (less than 1 second)
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > 0);
    }

    #[test]
    fn low_time_shortens_only_sudden_death_horizon() {
        let mut cyclic = create_default_limits();
        cyclic.movestogo = 40;
        assert_eq!(maximum_move_horizon(&cyclic, 500), 40);

        let sudden_death = create_default_limits();
        assert_eq!(maximum_move_horizon(&sudden_death, 500), 25);
    }

    #[test]
    fn test_high_ply_count() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 100, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > tm.optimum());
    }

    #[test]
    fn test_maximum_moves_to_go() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.movestogo = 100; // Will be capped to 50
        assert_eq!(maximum_move_horizon(&limits, 60000), 50);
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > tm.optimum());
    }

    #[test]
    fn test_high_move_overhead() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [5000, 5000]; // 5 seconds
        let options = create_options_map(vec![("Move Overhead", 1000)]); // 1 second overhead
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        assert!(tm.optimum() > 0);
        assert!(tm.maximum() < 4000); // Should account for overhead
    }

    // ========== Floating-Point Precision Tests ==========

    #[test]
    fn test_log10_calculation_precision() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [100000, 100000]; // 100 seconds
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        // Verify time_adjust calculation matches expected formula
        // time_adjust = 0.3272 * log10(timeLeft) - 0.4141
        // with mtg = 50 and timeLeft = 100000 + 1000 * (mtg - 1) - 30 * (2 + mtg)
        let time_left = 100000.0 + 1000.0 * 49.0 - 30.0 * 52.0;
        let expected = 0.3272 * f64::log10(time_left) - 0.4141;
        assert!((time_adjust - expected).abs() < 1e-9);
    }

    #[test]
    fn test_pow_calculation_precision() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let options = create_options_map(vec![("Move Overhead", 30)]);
        let mut time_adjust = 1.0; // Pre-set to avoid recalculation

        // Test with different ply values to verify pow() calculation
        for ply in [0, 10, 20, 50, 100] {
            tm.init(&mut limits, Color::White, ply, &options, &mut time_adjust);
            assert!(tm.optimum() > 0);
        }
    }

    // ========== Option Parsing Tests ==========

    #[test]
    fn test_option_parsing_with_missing_options() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let options = OptionsMap::new(); // Empty options map
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        // Should use defaults (0) for missing options
        assert!(tm.optimum() > 0);
        assert!(!tm.use_nodes_time); // nodestime defaults to 0
    }

    #[test]
    fn test_option_parsing_with_bool_as_int() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        let mut options = OptionsMap::new();

        // Add Ponder as a check option (bool represented as 0/1)
        let ponder_opt = UciOption::new_check("Ponder", true, None);
        options.add(ponder_opt);

        let mut time_adjust = -1.0;
        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        // Ponder=true should increase optimum time by a quarter
        let optimum_with_ponder = tm.optimum();
        assert!(optimum_with_ponder > 0);

        let mut tm_no_ponder = TimeManagement::new();
        let mut limits_no_ponder = create_default_limits();
        let mut time_adjust_no_ponder = -1.0;
        tm_no_ponder.init(
            &mut limits_no_ponder,
            Color::White,
            10,
            &OptionsMap::new(),
            &mut time_adjust_no_ponder,
        );
        let base = tm_no_ponder.optimum();
        assert_eq!(optimum_with_ponder, base + base / 4);
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_elapsed_time_always_returns_time() {
        let mut tm = TimeManagement::new();
        tm.use_nodes_time = true; // Even in nodes mode
        tm.start_time = now() - 50;

        let elapsed = tm.elapsed_time();
        assert!((50..=150).contains(&elapsed)); // Allow tolerance
    }

    #[test]
    fn test_time_management_workflow() {
        // Simulate a complete time management workflow
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [60000, 60000];
        limits.inc = [1000, 1000];
        limits.movestogo = 0; // Sudden death

        let options = create_options_map(vec![("Move Overhead", 30), ("Ponder", 0)]);
        let mut time_adjust = -1.0;

        // Initialize for white
        tm.init(&mut limits, Color::White, 1, &options, &mut time_adjust);
        let initial_optimum = tm.optimum();
        let initial_maximum = tm.maximum();

        assert!(initial_optimum > 0);
        assert!(initial_maximum > initial_optimum);
        assert!(time_adjust > 0.0);

        // Simulate moving to next ply
        tm.init(&mut limits, Color::White, 2, &options, &mut time_adjust);

        // Time should still be allocated reasonably
        assert!(tm.optimum() > 0);
        assert!(tm.maximum() > tm.optimum());
    }

    #[test]
    fn test_nodes_as_time_complete_workflow() {
        let mut tm = TimeManagement::new();
        let mut limits = create_default_limits();
        limits.time = [10000, 10000]; // 10 seconds

        let options = create_options_map(vec![("nodestime", 1000)]);
        let mut time_adjust = -1.0;

        // First initialization
        tm.init(&mut limits, Color::White, 1, &options, &mut time_adjust);
        assert_eq!(tm.available_nodes, 10000000);
        assert!(tm.use_nodes_time);

        // Simulate search consuming nodes
        tm.advance_nodes_time(1000000);
        assert_eq!(tm.available_nodes, 9000000);

        // Check elapsed returns nodes
        let elapsed = tm.elapsed(|| 500000);
        assert_eq!(elapsed, 500000);

        // Second search with remaining nodes
        limits.time = [5000, 5000]; // Different time, but should use available_nodes
        tm.init(&mut limits, Color::White, 2, &options, &mut time_adjust);
        assert_eq!(tm.available_nodes, 9000000); // Not reset
        assert_eq!(limits.time[Color::White as usize], 9000000);
    }

    // ========== Behavioral Compatibility Tests ==========

    #[test]
    fn test_c_plus_plus_compatibility_standard_time() {
        // Test case that should produce similar results to C++ version
        let mut tm = TimeManagement::new();
        let mut limits = LimitsType {
            searchmoves: vec![],
            time: [60000, 60000], // 60 seconds
            inc: [0, 0],          // No increment
            npmsec: 0,
            movetime: 0,
            start_time: 1000,
            movestogo: 0, // Sudden death
            depth: 255,
            mate: 0,
            perft: 0,
            infinite: 0,
            nodes: 0,
            ponder_mode: false,
        };

        let options = create_options_map(vec![("Move Overhead", 10)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::White, 10, &options, &mut time_adjust);

        // Verify reasonable time allocation
        assert!(tm.optimum() > 500); // At least 500ms
        assert!(tm.optimum() < 10000); // But not more than 10 seconds
        assert!(tm.maximum() > tm.optimum());
        assert!(tm.maximum() < 50000); // Maximum should be reasonable
    }

    #[test]
    fn test_c_plus_plus_compatibility_tournament() {
        let mut tm = TimeManagement::new();
        let mut limits = LimitsType {
            searchmoves: vec![],
            time: [300000, 300000], // 5 minutes
            inc: [3000, 3000],      // 3 second increment
            npmsec: 0,
            movetime: 0,
            start_time: 1000,
            movestogo: 40, // 40 moves to time control
            depth: 255,
            mate: 0,
            perft: 0,
            infinite: 0,
            nodes: 0,
            ponder_mode: false,
        };

        let options = create_options_map(vec![("Move Overhead", 50)]);
        let mut time_adjust = -1.0;

        tm.init(&mut limits, Color::Black, 20, &options, &mut time_adjust);

        // In tournament mode with 40 moves to go
        assert!(tm.optimum() > 1000); // Should use reasonable time
        assert!(tm.optimum() < 20000); // But not too much
        assert_eq!(time_adjust, -1.0); // Should not be modified in tournament mode
    }
}
