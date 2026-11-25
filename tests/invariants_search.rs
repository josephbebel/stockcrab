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

//! Search-level invariants that hold regardless of upstream tuning.
//!
//! Deliberately not golden node counts: node counts change with every upstream
//! tuning patch, so this file checks invariants that survive tuning. What does
//! not change is that a fixed-depth single-threaded search is reproducible, that
//! its best move is legal, and that every reported PV starts with a legal move.
//!
//! `scripts/parity.sh` covers exact agreement with C++; these cover the properties
//! that must hold even when no C++ engine is present.

mod common;

use std::sync::{Arc, Mutex};

use stockcrab::engine::Engine;
use stockcrab::misc::now;
use stockcrab::movegen::{GenType, generate};
use stockcrab::position::Position;
use stockcrab::search::LimitsType;
use stockcrab::types::Move;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const ENDGAME: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

struct Run {
    best_move: String,
    nodes: u64,
    info_lines: Vec<(usize, String)>,
}

fn search(fen: &str, depth: i32, configure: impl FnOnce(&mut Engine)) -> Run {
    let mut engine = Engine::new(None);
    configure(&mut engine);

    if let Some(err) = engine.set_position(fen, &[]) {
        panic!("failed to set position {fen}: {err:?}");
    }

    let best = Arc::new(Mutex::new(String::new()));
    let cb = best.clone();
    engine.set_on_bestmove(move |bm, _| *cb.lock().unwrap() = bm.to_string());

    let lines: Arc<Mutex<Vec<(usize, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = lines.clone();
    engine.set_on_update_full(move |info| {
        sink.lock().unwrap().push((info.multi_pv, info.pv));
    });

    let mut limits = LimitsType::default();
    limits.start_time = now();
    limits.depth = depth;

    engine.go(limits);
    engine.wait_for_search_finished();

    Run {
        best_move: best.lock().unwrap().clone(),
        nodes: engine.nodes_searched(),
        info_lines: lines.lock().unwrap().clone(),
    }
}

fn legal_moves(fen: &str) -> Vec<String> {
    stockcrab::bitboard::init();
    stockcrab::movegen::init();

    let mut pos = Position::default();
    pos.set_fen(fen, false).expect("valid FEN");
    let chess960 = pos.is_chess960();

    generate(&pos, GenType::All)
        .as_slice()
        .iter()
        .map(|&m: &Move| stockcrab::uci::UciEngine::move_str(m, chess960))
        .collect()
}

/// A single-threaded fixed-depth search must be reproducible: same best move and
/// same node count every time.
///
/// Non-determinism here would invalidate `scripts/parity.sh` itself, since that
/// compares exact node counts against C++.
#[test]
fn single_threaded_fixed_depth_search_is_reproducible() {
    if common::skip_without_network("single_threaded_fixed_depth_search_is_reproducible") {
        return;
    }
    for fen in [START, KIWIPETE, ENDGAME] {
        let a = search(fen, 9, |_| {});
        let b = search(fen, 9, |_| {});

        assert_eq!(
            a.best_move, b.best_move,
            "best move not reproducible for {fen}"
        );
        assert_eq!(
            a.nodes, b.nodes,
            "node count not reproducible for {fen}: {} vs {}",
            a.nodes, b.nodes
        );
    }
}

#[test]
fn best_move_is_always_legal() {
    if common::skip_without_network("best_move_is_always_legal") {
        return;
    }
    for fen in [START, KIWIPETE, ENDGAME] {
        let run = search(fen, 8, |_| {});
        let legal = legal_moves(fen);

        assert!(
            legal.contains(&run.best_move),
            "search returned {}, not a legal move in {fen}\nlegal: {legal:?}",
            run.best_move
        );
    }
}

/// Every move in every reported PV must be legal from the root.
///
#[test]
fn every_reported_pv_is_legal_when_replayed() {
    if common::skip_without_network("every_reported_pv_is_legal_when_replayed") {
        return;
    }
    for fen in [START, KIWIPETE, ENDGAME] {
        let run = search(fen, 8, |_| {});
        for (idx, pv) in &run.info_lines {
            let mut position = Position::default();
            position.set_fen(fen, false).unwrap();
            for (ply, text) in pv.split_whitespace().enumerate() {
                let mv = generate(&position, GenType::All)
                    .as_slice()
                    .iter()
                    .copied()
                    .find(|&mv| stockcrab::uci::UciEngine::move_str(mv, false) == text)
                    .unwrap_or_else(|| {
                        panic!("multipv {idx} illegal move {text} at ply {ply} in {fen}: {pv}")
                    });
                // SAFETY: the move is legal.
                unsafe { position.do_move(mv) };
            }
        }
    }
}

/// Searching deeper must not search fewer nodes.
///
/// Iterative deepening accumulates, so the total is monotonic. A regression that
/// aborts an iteration early shows up as a non-monotonic total.
#[test]
fn node_count_is_monotonic_in_depth() {
    if common::skip_without_network("node_count_is_monotonic_in_depth") {
        return;
    }
    for fen in [START, KIWIPETE] {
        let mut previous = 0u64;
        for depth in 4..=9 {
            let run = search(fen, depth, |_| {});
            assert!(
                run.nodes >= previous,
                "depth {depth} searched {} nodes, fewer than depth {} which searched {previous}\nFEN: {fen}",
                run.nodes,
                depth - 1
            );
            previous = run.nodes;
        }
    }
}

/// `MultiPV = N` must report exactly N lines with distinct first moves.
#[test]
fn multipv_reports_distinct_lines() {
    if common::skip_without_network("multipv_reports_distinct_lines") {
        return;
    }
    for n in [1usize, 3, 5] {
        let run = search(KIWIPETE, 8, |e| {
            e.get_options_mut()
                .setoption(&format!("name MultiPV value {n}"));
        });

        // The final iteration starts at the last line reporting PV 1.
        let start = run
            .info_lines
            .iter()
            .rposition(|(idx, _)| *idx == 1)
            .expect("no PV line with multi_pv == 1");
        let final_lines = &run.info_lines[start..];
        assert_eq!(final_lines.len(), n, "expected exactly {n} PV lines");
        for (expected, (idx, _)) in (1..=n).zip(final_lines) {
            assert_eq!(*idx, expected, "MultiPV={n} lines are out of order");
        }

        let mut firsts: Vec<&str> = final_lines
            .iter()
            .filter_map(|(_, pv)| pv.split_whitespace().next())
            .collect();
        firsts.sort_unstable();
        let before = firsts.len();
        firsts.dedup();
        assert_eq!(
            firsts.len(),
            before,
            "MultiPV={n} produced duplicate first moves"
        );
    }
}

/// Changing the hash size must not change which move is chosen at a fixed depth
/// on a position this shallow.
///
/// A TT keyed or indexed incorrectly tends to produce different results at
/// different sizes; identical results across sizes is weak but cheap evidence
/// that the table is coherent.
#[test]
fn best_move_is_stable_across_hash_sizes() {
    if common::skip_without_network("best_move_is_stable_across_hash_sizes") {
        return;
    }
    let mut moves = Vec::new();
    for mb in [1, 16, 64] {
        let run = search(KIWIPETE, 8, |e| {
            e.get_options_mut()
                .setoption(&format!("name Hash value {mb}"));
        });
        moves.push((mb, run.best_move));
    }

    let first = &moves[0].1;
    for (mb, mv) in &moves {
        assert_eq!(
            mv, first,
            "hash size {mb} changed the best move at depth 8 ({mv} vs {first})"
        );
    }
}

/// `ucinewgame` must fully reset search state: the same search run before and
/// after it must give the same answer.
#[test]
fn ucinewgame_restores_a_clean_search() {
    if common::skip_without_network("ucinewgame_restores_a_clean_search") {
        return;
    }
    let mut engine = Engine::new(None);

    let run_once = |engine: &mut Engine| -> (String, u64) {
        assert!(
            engine.set_position(KIWIPETE, &[]).is_none(),
            "failed to set KIWIPETE"
        );
        let best = Arc::new(Mutex::new(String::new()));
        let cb = best.clone();
        engine.set_on_bestmove(move |bm, _| *cb.lock().unwrap() = bm.to_string());

        let mut limits = LimitsType::default();
        limits.start_time = now();
        limits.depth = 8;
        engine.go(limits);
        engine.wait_for_search_finished();
        let bm = best.lock().unwrap().clone();
        (bm, engine.nodes_searched())
    };

    let first = run_once(&mut engine);
    engine.search_clear();
    let second = run_once(&mut engine);

    assert_eq!(
        first, second,
        "search after ucinewgame differs from a fresh one; state is leaking between games"
    );
}

/// A position with a single legal move must return it.
#[test]
fn forced_move_is_returned() {
    if common::skip_without_network("forced_move_is_returned") {
        return;
    }
    // Black is in check from Rh1 along the h-file. Kh7 stays on the file and
    // Kg7 is covered by Kf6, so Kg8 is the only legal move. (Verified with the
    // C++ engine: `go perft 1` reports exactly `h8g8: 1`.)
    let fen = "7k/8/5K2/8/8/8/8/7R b - - 0 1";
    let legal = legal_moves(fen);
    assert_eq!(
        legal.len(),
        1,
        "test position should have exactly one legal move, has {legal:?}"
    );

    let run = search(fen, 6, |_| {});
    assert_eq!(run.best_move, legal[0], "forced move not returned");
}

/// `flip` must leave the engine in exactly the state `position fen <flipped fen>`
/// would: the next `go` copies its root state (keys, repetition history) from the
/// engine's position, so a stale history entry silently changes the search.
#[test]
fn flip_matches_setting_the_flipped_fen() {
    if common::skip_without_network("flip_matches_setting_the_flipped_fen") {
        return;
    }
    let go = |engine: &mut Engine, depth: i32| -> (String, u64) {
        let best = Arc::new(Mutex::new(String::new()));
        let cb = best.clone();
        engine.set_on_bestmove(move |bm, _| *cb.lock().unwrap() = bm.to_string());
        let mut limits = LimitsType::default();
        limits.start_time = now();
        limits.depth = depth;
        engine.go(limits);
        engine.wait_for_search_finished();
        let bm = best.lock().unwrap().clone();
        (bm, engine.nodes_searched())
    };

    let lines: [&[&str]; 2] = [
        &[],
        &["e2e4", "e7e5", "g1f3", "b8c6", "f3g1", "c6b8", "g1f3"],
    ];
    for moves in lines {
        let moves: Vec<String> = moves.iter().map(|m| m.to_string()).collect();

        let mut flipped = Engine::new(None);
        assert!(flipped.set_position(START, &moves).is_none());
        go(&mut flipped, 2);
        assert!(flipped.flip().is_none());
        let flipped_fen = flipped.fen();
        let via_flip = go(&mut flipped, 3);

        let mut via_fen = Engine::new(None);
        assert!(via_fen.set_position(START, &moves).is_none());
        go(&mut via_fen, 2);
        assert!(via_fen.set_position(&flipped_fen, &[]).is_none());
        let expected = go(&mut via_fen, 3);

        assert_eq!(
            via_flip, expected,
            "flip then go differs from position fen {flipped_fen} then go (moves: {moves:?})"
        );
    }
}
