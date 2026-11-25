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

//! Tests for `go` limit enforcement and MultiPV:
//!
//! * `go movetime N` and `go nodes N` terminate: `SearchManager::check_time` checks the
//!   `movetime` and `nodes` conditions as well as the time-management maximum, as C++ does
//!   (`SearchManager::check_time` in search.cpp).
//! * `nodestime` makes `check_time` count nodes (`TimeManagement::elapsed(nodes)`) instead
//!   of wall-clock time.
//! * With MultiPV, each root move keeps its own previous-iteration PV although the root move
//!   list is re-sorted inside the MultiPV loop.
//!
//! These assert termination and self-consistency, which is what can be checked
//! without a C++ engine present. Exact agreement with C++ is covered by
//! `scripts/parity.sh`.

mod common;

use core::time::Duration;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use stockcrab::engine::Engine;
use stockcrab::misc::now;
use stockcrab::search::LimitsType;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A search must never outlive this. Every limit under test should finish within
/// a few seconds; the margin is for loaded CI machines. A watchdog thread aborts
/// the process if a search is still running after this long, since a search that
/// never terminates would otherwise hang the test forever.
const WATCHDOG: Duration = Duration::from_secs(60);

struct Outcome {
    best_move: String,
    nodes: u64,
    elapsed: Duration,
}

fn run(configure: impl FnOnce(&mut Engine), limits: LimitsType) -> Outcome {
    let mut engine = Engine::new(None);
    configure(&mut engine);

    if let Some(err) = engine.set_position(START, &[]) {
        panic!("failed to set position: {err:?}");
    }

    let best = Arc::new(Mutex::new(String::new()));
    let cb = best.clone();
    engine.set_on_bestmove(move |bm, _| *cb.lock().unwrap() = bm.to_string());

    // Enforce the deadline for real: `wait_for_search_finished` blocks for as
    // long as the search runs, so the check must happen on another thread.
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if done_rx.recv_timeout(WATCHDOG) == Err(RecvTimeoutError::Timeout) {
            eprintln!(
                "search did not terminate within {WATCHDOG:?} -- a limit is not being enforced"
            );
            std::process::abort();
        }
    });

    let start = Instant::now();
    engine.go(limits);
    engine.wait_for_search_finished();
    let elapsed = start.elapsed();
    drop(done_tx);
    watchdog.join().unwrap();

    assert!(
        elapsed < WATCHDOG,
        "search did not terminate within {WATCHDOG:?} -- a limit is not being enforced"
    );

    Outcome {
        best_move: best.lock().unwrap().clone(),
        nodes: engine.nodes_searched(),
        elapsed,
    }
}

fn limits_with(f: impl FnOnce(&mut LimitsType)) -> LimitsType {
    let mut l = LimitsType::default();
    l.start_time = now();
    f(&mut l);
    l
}

#[test]
fn go_movetime_terminates() {
    if common::skip_without_network("go_movetime_terminates") {
        return;
    }
    for ms in [200, 1000] {
        let out = run(|_| {}, limits_with(|l| l.movetime = ms));
        assert!(
            !out.best_move.is_empty() && out.best_move != "(none)",
            "movetime {ms} produced no best move"
        );
        // Generous upper bound: the point is that it stops at all, not that it
        // stops to the millisecond.
        assert!(
            out.elapsed < Duration::from_millis((ms as u64) * 10 + 5000),
            "movetime {ms} took {:?}, far beyond the requested budget",
            out.elapsed
        );
    }
}

#[test]
fn go_nodes_terminates_and_respects_the_limit() {
    if common::skip_without_network("go_nodes_terminates_and_respects_the_limit") {
        return;
    }
    // 500k nodes take minutes in an unoptimized build and would trip the watchdog.
    let limits: &[u64] = if cfg!(debug_assertions) {
        &[50_000]
    } else {
        &[50_000, 500_000]
    };
    for &limit in limits {
        let out = run(|_| {}, limits_with(|l| l.nodes = limit));
        assert!(
            !out.best_move.is_empty() && out.best_move != "(none)",
            "nodes {limit} produced no best move"
        );
        // C++ re-arms its check counter so the limit is overshot by ~0.1%. Allow
        // 5% plus 10k nodes of absolute slack (polling granularity, helper
        // threads); a missing nodes check runs to millions of nodes, so this is
        // not a knife edge.
        assert!(
            out.nodes >= limit,
            "stopped early at {} nodes for a limit of {limit}",
            out.nodes
        );
        assert!(
            out.nodes < limit + limit / 20 + 10_000,
            "overshot the node limit badly: {} nodes for a limit of {limit}",
            out.nodes
        );
    }
}

#[test]
fn nodestime_terminates() {
    if common::skip_without_network("nodestime_terminates") {
        return;
    }
    // `nodestime` makes the time manager count nodes instead of milliseconds,
    // so a wall-clock elapsed value never reaches the computed maximum.
    let out = run(
        |e| {
            e.get_options_mut().setoption("name nodestime value 100");
        },
        limits_with(|l| {
            l.time[0] = 1000;
            l.time[1] = 1000;
        }),
    );
    assert!(
        !out.best_move.is_empty() && out.best_move != "(none)",
        "nodestime search produced no best move"
    );
}

#[test]
fn go_depth_and_time_management_still_terminate() {
    if common::skip_without_network("go_depth_and_time_management_still_terminate") {
        return;
    }
    // Covers the depth and wtime/btime paths, which share check_time with
    // movetime and nodes.
    let out = run(|_| {}, limits_with(|l| l.depth = 8));
    assert!(!out.best_move.is_empty(), "go depth produced no best move");

    let out = run(
        |_| {},
        limits_with(|l| {
            l.time[0] = 1000;
            l.time[1] = 1000;
        }),
    );
    assert!(
        !out.best_move.is_empty(),
        "go wtime/btime produced no best move"
    );
}

#[test]
fn multipv_reports_the_requested_number_of_distinct_lines() {
    if common::skip_without_network("multipv_reports_the_requested_number_of_distinct_lines") {
        return;
    }
    for multi_pv in [1usize, 3, 5] {
        let lines: Arc<Mutex<Vec<(usize, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = lines.clone();
        let best_move = Arc::new(Mutex::new(String::new()));
        let best_sink = best_move.clone();

        let mut engine = Engine::new(None);
        engine
            .get_options_mut()
            .setoption(&format!("name MultiPV value {multi_pv}"));
        if let Some(err) = engine.set_position(START, &[]) {
            panic!("failed to set position: {err:?}");
        }
        engine.set_on_update_full(move |info| {
            sink.lock().unwrap().push((info.multi_pv, info.pv));
        });
        engine.set_on_bestmove(move |best, _ponder| {
            *best_sink.lock().unwrap() = best.to_string();
        });

        let mut limits = LimitsType::default();
        limits.start_time = now();
        limits.depth = 8;
        engine.go(limits);
        engine.wait_for_search_finished();

        let collected = lines.lock().unwrap();
        let deepest: Vec<_> = collected
            .iter()
            .rev()
            .take(multi_pv)
            .map(|(idx, pv)| (*idx, pv.clone()))
            .collect();

        assert_eq!(
            deepest.len(),
            multi_pv,
            "expected {multi_pv} PV lines in the final iteration, got {}",
            deepest.len()
        );

        // multipv indices are 1-based and must cover 1..=multi_pv exactly once.
        let mut indices: Vec<usize> = deepest.iter().map(|(i, _)| *i).collect();
        indices.sort_unstable();
        assert_eq!(
            indices,
            (1..=multi_pv).collect::<Vec<_>>(),
            "multipv indices are not 1..={multi_pv}"
        );

        // Line 1 is the best line, so its first move is the reported best move.
        let best = best_move.lock().unwrap().clone();
        let (_, first_pv) = deepest
            .iter()
            .find(|(i, _)| *i == 1)
            .expect("no multipv 1 line in the final iteration");
        assert_eq!(
            first_pv.split_whitespace().next(),
            Some(best.as_str()),
            "MultiPV={multi_pv}: line 1 does not start with the reported best move"
        );

        // Distinct lines must start with distinct moves.
        let mut firsts: Vec<String> = deepest
            .iter()
            .map(|(i, pv)| {
                pv.split_whitespace()
                    .next()
                    .unwrap_or_else(|| panic!("MultiPV line {i} has an empty PV"))
                    .to_string()
            })
            .collect();
        firsts.sort();
        let before = firsts.len();
        firsts.dedup();
        assert_eq!(
            firsts.len(),
            before,
            "MultiPV={multi_pv} produced duplicate first moves: {firsts:?}"
        );
    }
}
