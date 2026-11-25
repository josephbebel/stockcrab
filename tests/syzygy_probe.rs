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

//! Syzygy tablebase probing tests.
//!
//! Coverage:
//!
//! * `MAX_CARDINALITY` is a running maximum over every registered table.
//! * `TBTable::from_code` and `Position::set_code` derive the material key that
//!   the table hash lookup matches against.
//! * The WDL and DTZ file magic bytes are checked and not interchangeable.
//! * Table initialisation (`set_groups` / `set_sizes` / `set_dtz_map` / the
//!   header walk) produces indices that give correct verdicts, including the
//!   leading-pawn encoding and 5+ man tables.
//! * `rank_root_moves` orders root moves by their tablebase value.
//! * Probing survives positions that need 5+ men (see
//!   `known_5_plus_man_regression_positions`) and corrupt table files.
//!
//! These tests need real tablebase files, which are not in the repository. Point
//! `SYZYGY_PATH` at a tablebase directory to run them; without it they skip.
//! Files come from <https://tablebase.lichess.ovh/tables/standard/>: the 3-4 man
//! subset is under 5 MB, the full 3-4-5 set is 1.0 GB, and individual 6/7-man
//! tables can be fetched selectively (KPPPPPvK is only 3.7 MB and is the only
//! way to exercise five leading pawns).

mod common;
#[path = "common/syzygy.rs"]
mod syzygy_coverage;

use stockcrab::position::Position;
use stockcrab::syzygy::{self, ProbeState, WDLScore};
use stockcrab::types::Color;
use syzygy_coverage::Coverage;

/// Serialises every test in this file.
///
/// Syzygy state is process-global -- `TB_TABLES`, `MAX_CARDINALITY` and
/// `TB_PATHS` are statics, matching the C++ layout -- and several tests here
/// deliberately re-`init` it: with an empty path, with a nonexistent one, with a
/// directory of corrupt files. `cargo test` runs test functions in parallel, so
/// without this one test's `init` pulls the tables out from under another and
/// the failures read as probe bugs in whichever test happened to lose the race.
static TB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the tablebase lock and initialise. The returned guard must be bound
/// for the whole test body -- `let _tb = init_all(..)`, never `let _ =`, which
/// would drop it immediately and restore the race.
#[must_use = "bind the guard for the test body, or the global state is unlocked again at once"]
fn init_all(tb: &str) -> std::sync::MutexGuard<'static, ()> {
    // A failing test poisons the lock; the tests after it still need to run.
    let guard = TB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    stockcrab::bitboard::init();
    stockcrab::movegen::init();
    syzygy::init(tb);
    guard
}

fn setup(fen: &str) -> Position {
    let mut pos = Position::default();
    pos.set_fen(fen, false).expect("valid test FEN");
    pos
}

/// `set_code` is how every table derives its material key, so a bug here breaks
/// all of Syzygy. Runs without tablebase files.
#[test]
fn set_code_matches_a_real_position_of_the_same_material() {
    let _coverage = Coverage::new("set_code_matches_a_real_position_of_the_same_material");
    stockcrab::bitboard::init();
    stockcrab::movegen::init();

    for (code, fen) in [
        ("KRvK", "8/8/8/8/8/4k3/8/R3K3 w - - 0 1"),
        ("KQvK", "8/8/8/8/8/3k4/8/3KQ3 w - - 0 1"),
        ("KPvK", "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"),
        ("KBNvK", "4k3/8/8/8/8/8/8/2BNK3 w - - 0 1"),
    ] {
        let from_fen = setup(fen).material_key();

        let mut pos = Position::default();
        pos.set_code(code, Color::White)
            .unwrap_or_else(|e| panic!("set_code({code}) rejected: {}", e.message()));

        assert_eq!(
            pos.material_key(),
            from_fen,
            "material key from code {code} does not match a real position of that material; \
             the tablebase hash lookup is keyed on this"
        );
    }
}

#[test]
fn probe_returns_known_results() {
    let mut coverage = Coverage::new("probe_returns_known_results");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    // Verdicts confirmed against the C++ reference engine.
    let cases: &[(&str, WDLScore)] = &[
        // Pawnless wins.
        ("8/8/8/8/8/4k3/8/R3K3 w - - 0 1", WDLScore::WDLWin),
        ("8/8/8/8/8/3k4/8/3KQ3 w - - 0 1", WDLScore::WDLWin),
        ("4k3/8/8/8/8/8/8/2BNK3 w - - 0 1", WDLScore::WDLWin),
        // Pawnless draw: two knights cannot force mate.
        ("8/8/8/3k4/8/8/8/3KNN2 w - - 0 1", WDLScore::WDLDraw),
        // Pawn endings. These exercise the leading-pawn encoding, a separate
        // index path from the pieces one: the leading-pawn index must fold in
        // the remaining pieces, or drawn KPvK positions probe as wins.
        ("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", WDLScore::WDLDraw),
        ("8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", WDLScore::WDLDraw),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", WDLScore::WDLWin),
        ("8/8/1k6/8/8/8/6P1/6K1 w - - 0 1", WDLScore::WDLWin),
    ];

    for (fen, expect) in cases {
        let mut pos = setup(fen);
        let mut err = ProbeState::Fail;
        let wdl = syzygy::probe_wdl(&mut pos, &mut err);

        assert_eq!(err, ProbeState::Ok, "probe failed outright for {fen}");
        assert_eq!(wdl, *expect, "wrong WDL verdict for {fen}");
    }
}

#[test]
fn max_cardinality_reflects_every_loaded_table() {
    let mut coverage = Coverage::new("max_cardinality_reflects_every_loaded_table");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let max = syzygy::MAX_CARDINALITY.load(core::sync::atomic::Ordering::Relaxed);
    // It is a running maximum over every registered table, so it must reach the
    // largest table's piece count, not stop at the first table's value.
    assert!(
        max >= 4,
        "MAX_CARDINALITY is {max}; with a 3-4-5 man set it must be at least 4, \
         so it is not being updated per table"
    );
}

#[test]
fn dtz_probe_agrees_with_wdl() {
    let mut coverage = Coverage::new("dtz_probe_agrees_with_wdl");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    for fen in [
        "8/8/8/8/8/4k3/8/R3K3 w - - 0 1",
        "8/8/8/8/8/3k4/8/3KQ3 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    ] {
        let mut pos = setup(fen);

        let mut werr = ProbeState::Fail;
        let wdl = syzygy::probe_wdl(&mut pos, &mut werr);
        assert_eq!(werr, ProbeState::Ok, "WDL probe failed for {fen}");

        let mut derr = ProbeState::Fail;
        let dtz = syzygy::probe_dtz(&mut pos, &mut derr);
        assert_ne!(derr, ProbeState::Fail, "DTZ probe failed for {fen}");

        // A won position must have a positive distance-to-zeroing.
        if wdl == WDLScore::WDLWin {
            assert!(dtz > 0, "won position {fen} reported dtz {dtz}");
        }
    }
}

/// Known 5+ man positions that exercise paths the 3-4 man tables cannot reach.
///
/// * `1R6/7p/8/4p3/7K/8/6k1/8 w` has black as the reference colour for the
///   leading-pawn bitboard. The pawns must be removed from the occupancy using
///   the same (flipped or unflipped) squares the bitboard was built from, or
///   extra bits survive and the piece loop runs past the end of `squares`.
/// * `4K3/8/8/8/P4P2/8/1P5p/k7 b` has a long tablebase PV. `syzygy_extend_pv`
///   makes and unmakes a long line of moves on the position.
///
/// Both cases are covered by the full 3–5-piece set: the six-piece root in the
/// second case wins through a capture into that set. Required release mode must
/// execute both; optional runs with missing material may skip. Successful probes
/// must agree with `GOLDEN` where it has an entry.
#[test]
fn known_5_plus_man_regression_positions() {
    let mut coverage = Coverage::new("known_5_plus_man_regression_positions");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    for fen in [
        "1R6/7p/8/4p3/7K/8/6k1/8 w - - 0 1",
        "4K3/8/8/8/P4P2/8/1P5p/k7 b - - 0 1",
    ] {
        let mut pos = setup(fen);
        let mut err = ProbeState::Fail;
        let wdl = syzygy::probe_wdl(&mut pos, &mut err);
        if !coverage.probe_available(&tb, fen, err != ProbeState::Fail, true) {
            continue;
        }
        // Both are successful upstream outcomes. With only 3–5-man tables,
        // the second (6-man) position wins through a capture into a known table;
        // pinned C++ returns ZEROING_BEST_MOVE (2), rather than OK (1).
        assert!(
            matches!(err, ProbeState::Ok | ProbeState::ZeroingBestMove),
            "probe of {fen} reported {err:?}"
        );
        if let Some(&(_, expect)) = GOLDEN.iter().find(|&&(f, _)| f == fen) {
            assert_eq!(wdl, expect, "WDL verdict for {fen}");
        }
    }
}

/// Exercise a real mate extension long enough to grow its state container.
#[test]
fn extend_pv_stateinfo_container_has_stable_addresses() {
    let _coverage = Coverage::new("extend_pv_stateinfo_container_has_stable_addresses");
    use stockcrab::search::{LimitsType, syzygy_extend_pv};
    use stockcrab::thread::RootMove;
    use stockcrab::types::{Move, VALUE_TB_WIN_IN_MAX_PLY};
    use stockcrab::ucioption::OptionsMap;

    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/syzygy");
    let _tb = init_all(fixtures);
    let pos = setup("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let original_fen = pos.fen();
    let original_key = pos.key();
    let root_mv = stockcrab::uci::UciEngine::to_move(&pos, "a2a7");
    assert_ne!(root_mv, Move::none());
    let mut root = RootMove::new(root_mv);
    let mut score = VALUE_TB_WIN_IN_MAX_PLY;
    syzygy_extend_pv(
        &OptionsMap::new(),
        &LimitsType::default(),
        &pos,
        &mut root,
        &mut score,
        1,
    );
    assert!(root.pv.len() > 8, "PV must be extended: {:?}", root.pv);
    assert_eq!(pos.fen(), original_fen);
    assert_eq!(pos.key(), original_key);
    assert!(pos.pos_is_ok());

    let mut replay = pos.clone();
    for &mv in &root.pv {
        assert!(replay.legal(mv));
        // SAFETY: legality is checked.
        unsafe { replay.do_move(mv) };
        assert!(replay.pos_is_ok());
    }
    assert!(stockcrab::movegen::generate(&replay, stockcrab::movegen::GenType::All).is_empty());
}

#[test]
fn tb_won_positions_get_a_mate_length_pv() {
    let mut coverage = Coverage::new("tb_won_positions_get_a_mate_length_pv");
    let Some(tb) = coverage.tables(false) else {
        return;
    };

    if !coverage.network() {
        return;
    }

    let exe = std::path::PathBuf::from(env!("CARGO_BIN_EXE_stockcrab"));

    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::{Duration, Instant};

    // KBNvK is a forced mate that takes ~30 plies of DTZ-optimal play, so a
    // correctly extended PV is far longer than a depth-4 search would produce.
    let mut child = Command::new(&exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn engine");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(
            format!(
                "setoption name SyzygyPath value {tb}\n\
                 setoption name Threads value 1\n\
                 position fen 4k3/8/8/8/8/8/8/2BNK3 w - - 0 1\n\
                 go depth 4\n"
            )
            .as_bytes(),
        )
        .expect("write");

    // Read on a helper thread so a hung engine fails the test instead of
    // blocking it forever.
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut longest = 0usize;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining) {
            Ok(line) => {
                if let Some(idx) = line.find(" pv ") {
                    longest = longest.max(line[idx + 4..].split_whitespace().count());
                }
                if line.starts_with("bestmove") {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("engine printed no bestmove within 60 s");
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();

    assert!(
        longest >= 20,
        "longest PV in a won KBNvK position was {longest} moves; the Syzygy PV \
         extension is not running (C++ reports ~29 here)"
    );
}

/// Golden WDL verdicts for 5-, 6- and 7-man positions.
///
/// Expectations are the C++ reference engine's tablebase verdict for each
/// position -- its depth-1 score with the root in TB: +20000 win, 0 draw,
/// -20000 loss -- captured against the same tablebase directory. Positions whose
/// score was ambiguous (cursed wins and blessed losses, which the 50-move rule
/// maps to small non-zero values) were excluded rather than guessed at.
///
/// The 3-4 man cases in `probe_returns_known_results` cannot exercise the 5+ man
/// paths (leading-pawn bitboard handling, `StateInfo` stability in PV
/// extension), which is why this table exists.
///
/// To regenerate after an upstream change, use `scripts/syzygy_diff.sh golden`.
/// Pair each FEN with its score by segmenting the engine's output on `bestmove`,
/// not by zipping two line lists: positions that are already mate emit no
/// `info depth 1` line, and a naive zip silently shifts every later expectation.
const GOLDEN: &[(&str, WDLScore)] = &[
    // ---- 5 men ----
    ("1K3k2/8/8/8/3q4/8/p3r3/8 b - - 0 1", WDLScore::WDLWin),
    ("1K6/6p1/2q5/8/3k4/8/4P3/8 w - - 0 1", WDLScore::WDLLoss),
    ("1K6/8/1b6/8/8/2r5/2k5/4b3 w - - 0 1", WDLScore::WDLLoss),
    ("1K6/8/2k5/6n1/1B6/4p3/8/8 b - - 0 1", WDLScore::WDLDraw),
    ("1K6/8/7p/8/1k4p1/8/7Q/8 b - - 0 1", WDLScore::WDLLoss),
    ("1K6/8/8/8/P3Q3/2p5/8/2k5 w - - 0 1", WDLScore::WDLWin),
    ("1K6/p7/8/3P4/8/N7/8/6k1 w - - 0 1", WDLScore::WDLWin),
    ("1N6/5K2/6n1/1k3q2/8/8/8/8 w - - 0 1", WDLScore::WDLLoss),
    ("1NQ5/8/2Q4k/8/7K/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("1R6/7p/8/4p3/7K/8/6k1/8 w - - 0 1", WDLScore::WDLWin),
    ("1k6/1q4Q1/8/8/2K3p1/8/8/8 w - - 0 1", WDLScore::WDLDraw),
    ("1k6/4P3/8/1P6/8/3P4/4K3/8 b - - 0 1", WDLScore::WDLLoss),
    ("1k6/5p2/4p3/6p1/8/8/8/6K1 w - - 0 1", WDLScore::WDLLoss),
    ("1k6/8/8/8/5K2/4P3/r5P1/8 b - - 0 1", WDLScore::WDLWin),
    ("1k6/Q7/8/4R3/5r2/7K/8/8 b - - 0 1", WDLScore::WDLDraw),
    ("1n6/8/1q6/4K3/8/6p1/8/1k6 w - - 0 1", WDLScore::WDLLoss),
    ("1r6/8/3r4/5k2/8/8/4P3/K7 w - - 0 1", WDLScore::WDLLoss),
    ("2B5/4K3/6b1/8/2k5/4p3/8/8 b - - 0 1", WDLScore::WDLWin),
    ("2K5/5p2/8/3n4/8/8/4r3/6k1 b - - 0 1", WDLScore::WDLWin),
    ("2K5/6Nk/3p4/8/8/8/6R1/8 w - - 0 1", WDLScore::WDLWin),
    ("2K5/8/3B4/2b3P1/8/7k/8/8 w - - 0 1", WDLScore::WDLWin),
    ("2K5/8/4P3/Q7/8/1p6/8/3k4 b - - 0 1", WDLScore::WDLLoss),
    ("2K5/8/5p2/4P3/5k2/8/4q3/8 w - - 0 1", WDLScore::WDLLoss),
    ("2K5/8/8/4P3/8/k7/8/5b1r w - - 0 1", WDLScore::WDLLoss),
    ("2K5/N2p4/7k/7p/8/8/8/8 w - - 0 1", WDLScore::WDLDraw),
    ("2N2r2/8/8/1k6/3K4/8/8/4N3 b - - 0 1", WDLScore::WDLDraw),
    ("2N5/8/1k4K1/6P1/8/8/8/5Q2 b - - 0 1", WDLScore::WDLLoss),
    ("2R5/K6k/8/8/8/5p2/1B6/8 b - - 0 1", WDLScore::WDLLoss),
    ("2k5/6K1/N5n1/8/2Q5/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("2k5/8/1p6/8/7P/6K1/3p4/8 w - - 0 1", WDLScore::WDLLoss),
    ("2k5/8/5K2/4P3/3p4/5P2/8/8 w - - 0 1", WDLScore::WDLWin),
    ("2k5/8/6b1/4B3/8/1p6/8/6K1 b - - 0 1", WDLScore::WDLDraw),
    ("2k5/8/7K/R7/5P2/8/4r3/8 b - - 0 1", WDLScore::WDLDraw),
    ("2k5/8/8/8/7K/4N3/4B3/5R2 b - - 0 1", WDLScore::WDLLoss),
    ("2k5/8/8/8/8/K7/N7/2n3q1 w - - 0 1", WDLScore::WDLLoss),
    ("2r5/3k4/8/8/4NK2/8/8/7r w - - 0 1", WDLScore::WDLLoss),
    ("2r5/8/3P4/1K6/8/6n1/3k4/8 w - - 0 1", WDLScore::WDLLoss),
    ("2r5/8/8/8/8/8/1K1p1B2/7k b - - 0 1", WDLScore::WDLWin),
    ("3B1r2/8/8/1k1K4/8/8/8/2n5 b - - 0 1", WDLScore::WDLWin),
    ("3K1k2/8/8/8/8/1q6/1R6/6N1 b - - 0 1", WDLScore::WDLWin),
    ("3K4/1N1Q4/3N4/8/4k3/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("3K4/4Q3/6k1/8/p7/4P3/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("3K4/8/2b5/8/5B2/8/8/1k3r2 b - - 0 1", WDLScore::WDLWin),
    ("3K4/8/4R3/4kPN1/8/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("3K4/8/8/7n/3P4/5p2/2k5/8 w - - 0 1", WDLScore::WDLLoss),
    ("3KN3/1k6/8/8/8/7B/8/7b b - - 0 1", WDLScore::WDLDraw),
    ("3N4/8/7p/2k4K/8/8/8/4b3 w - - 0 1", WDLScore::WDLDraw),
    ("3Q4/8/5k2/8/7B/8/7p/2K5 b - - 0 1", WDLScore::WDLLoss),
    ("3Q4/8/8/7K/k7/p4N2/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("3R4/B3k3/8/1P6/K7/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("3b4/3B1K2/8/2b1k3/8/8/8/8 w - - 0 1", WDLScore::WDLDraw),
    ("3k4/8/5P2/K7/8/p7/1P6/8 w - - 0 1", WDLScore::WDLWin),
    ("3k4/8/8/2R5/5K2/8/2r5/2n5 b - - 0 1", WDLScore::WDLWin),
    ("3k4/8/8/8/2q5/3b4/7K/6N1 b - - 0 1", WDLScore::WDLWin),
    ("3k4/8/8/8/8/1P6/R7/2K3N1 w - - 0 1", WDLScore::WDLWin),
    ("3n4/8/8/8/2p5/K6P/8/7k b - - 0 1", WDLScore::WDLDraw),
    ("3q2K1/8/8/8/3k4/8/1n1p4/8 w - - 0 1", WDLScore::WDLLoss),
    ("3q4/7p/7k/8/6r1/8/8/4K3 b - - 0 1", WDLScore::WDLWin),
    ("3r4/2K5/8/7k/1b6/8/2n5/8 b - - 0 1", WDLScore::WDLWin),
    // ---- 6 men ----
    ("1K6/1P3P2/7P/8/8/7P/2k5/8 b - - 0 1", WDLScore::WDLLoss),
    ("1K6/8/8/7P/7k/5p2/P5P1/8 w - - 0 1", WDLScore::WDLWin),
    ("1k6/1PP2P2/8/4P2K/8/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("1k6/7p/8/3P1P2/4K3/8/P7/8 w - - 0 1", WDLScore::WDLWin),
    ("2K5/8/3P4/2P5/P7/3k4/5p2/8 w - - 0 1", WDLScore::WDLWin),
    ("2K5/8/8/8/5P1P/3P4/1P6/5k2 b - - 0 1", WDLScore::WDLLoss),
    ("2k5/8/2P4P/4p3/8/2P5/8/4K3 b - - 0 1", WDLScore::WDLLoss),
    ("2k5/8/2PK4/7P/5p2/8/4P3/8 b - - 0 1", WDLScore::WDLLoss),
    ("3K4/8/4k1P1/P7/8/1P3p2/8/8 b - - 0 1", WDLScore::WDLWin),
    ("3k4/1P4P1/1P6/P7/8/8/8/K7 b - - 0 1", WDLScore::WDLLoss),
    ("4K3/2P5/8/4p3/8/2P1P3/5k2/8 b - - 0 1", WDLScore::WDLLoss),
    ("4K3/3P4/8/8/7P/5p2/4P3/5k2 b - - 0 1", WDLScore::WDLDraw),
    ("4K3/7P/4P2P/8/8/6k1/5P2/8 b - - 0 1", WDLScore::WDLLoss),
    ("4K3/8/8/3P3p/8/4P3/3P4/6k1 b - - 0 1", WDLScore::WDLLoss),
    ("4k3/2P5/2p5/5P2/8/2K5/7P/8 b - - 0 1", WDLScore::WDLLoss),
    ("4k3/8/K6p/4P3/8/4P3/1P6/8 w - - 0 1", WDLScore::WDLWin),
    ("5K2/1P6/k7/3P1P2/8/5p2/8/8 b - - 0 1", WDLScore::WDLDraw),
    ("5K2/2k2P2/8/8/5p2/8/1P4P1/8 b - - 0 1", WDLScore::WDLLoss),
    ("5K2/5p2/5P2/7P/6k1/4P3/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("5k2/5P2/8/4P3/7P/7P/1K6/8 w - - 0 1", WDLScore::WDLWin),
    ("5k2/5p2/4P1P1/7P/8/8/8/2K5 w - - 0 1", WDLScore::WDLWin),
    ("5k2/6P1/8/P7/3P4/1p6/6K1/8 b - - 0 1", WDLScore::WDLDraw),
    ("5k2/8/4K3/1P6/5P2/3P4/2P5/8 w - - 0 1", WDLScore::WDLWin),
    ("5k2/8/5P2/P7/1P6/2P5/8/3K4 b - - 0 1", WDLScore::WDLLoss),
    ("5k2/8/7P/8/8/6KP/p5P1/8 w - - 0 1", WDLScore::WDLLoss),
    ("5k2/8/8/8/KP6/3P3P/P7/8 b - - 0 1", WDLScore::WDLLoss),
    ("6K1/3k4/8/1p1P4/2P5/6P1/8/8 w - - 0 1", WDLScore::WDLWin),
    ("6K1/8/1P6/8/5p2/6P1/4Pk2/8 w - - 0 1", WDLScore::WDLWin),
    ("6K1/8/2k5/2P4p/8/P1P5/8/8 b - - 0 1", WDLScore::WDLWin),
    ("6K1/8/3P4/4P1P1/8/8/5p2/4k3 w - - 0 1", WDLScore::WDLWin),
    ("6K1/8/5k2/1P6/8/1P5P/7P/8 w - - 0 1", WDLScore::WDLWin),
    ("6K1/P1k5/5PP1/8/8/5P2/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("6k1/8/P6P/8/8/4P3/P1K5/8 b - - 0 1", WDLScore::WDLLoss),
    ("6k1/p1P5/8/8/8/1P1K4/6P1/8 b - - 0 1", WDLScore::WDLLoss),
    ("7k/6p1/3K4/2P5/3P4/8/1P6/8 w - - 0 1", WDLScore::WDLWin),
    ("7k/8/2p3PK/8/8/8/P1P5/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/1K1k4/8/3P4/2Pp4/8/4P3/8 w - - 0 1", WDLScore::WDLWin),
    ("8/1P6/1P3P2/1P6/8/7K/8/6k1 b - - 0 1", WDLScore::WDLLoss),
    ("8/1P6/2p4k/8/1P6/8/2P5/2K5 b - - 0 1", WDLScore::WDLLoss),
    ("8/1P6/4k3/8/p2K3P/3P4/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/1k6/8/4K3/4P3/2P4P/1P6/8 w - - 0 1", WDLScore::WDLWin),
    ("8/2K1k3/P7/3P4/6P1/8/6P1/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/2K5/8/PP5P/8/4Pk2/8/8 w - - 0 1", WDLScore::WDLWin),
    ("8/2P1P3/7P/4k3/P7/8/8/1K6 b - - 0 1", WDLScore::WDLLoss),
    ("8/2P1k3/8/8/P1P2K2/8/4P3/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/2k2P2/P2P4/3K4/8/8/5P2/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/2kP4/8/8/2P5/P5P1/8/3K4 b - - 0 1", WDLScore::WDLLoss),
    ("8/3P2P1/8/1P5k/1P6/8/8/K7 b - - 0 1", WDLScore::WDLLoss),
    ("8/3P4/7K/7P/7P/6k1/3P4/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/3PK1P1/8/8/3P1k2/8/1P6/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/3k2P1/8/6P1/P1P5/8/8/7K b - - 0 1", WDLScore::WDLLoss),
    ("8/4K3/3P4/4P3/P7/5P2/3k4/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/4K3/8/1P6/6P1/1P6/6kP/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/4P1P1/1k3P2/2P1K3/8/8/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/4P3/1P6/8/5K2/3P3P/8/1k6 b - - 0 1", WDLScore::WDLLoss),
    ("8/4P3/8/8/1P2K3/8/2PPk3/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/5P2/5k2/5P2/8/7P/2P5/2K5 b - - 0 1", WDLScore::WDLLoss),
    // ---- 7 men ----
    ("1K6/7P/7P/8/6P1/3k2P1/1P6/8 b - - 0 1", WDLScore::WDLLoss),
    ("1Q6/8/3k1K2/7Q/3Q2Q1/8/8/Q7 b - - 0 1", WDLScore::WDLLoss),
    ("1k6/4K1Q1/8/8/Q7/3Q4/5Q2/7Q b - - 0 1", WDLScore::WDLLoss),
    ("1k6/7P/4PP2/8/3K4/P7/7P/8 b - - 0 1", WDLScore::WDLLoss),
    ("2Q1K3/8/1Q6/8/6Q1/Q7/7Q/5k2 b - - 0 1", WDLScore::WDLLoss),
    ("2Q5/3Q4/2KQQ3/7Q/8/8/3k4/8 b - - 0 1", WDLScore::WDLLoss),
    ("2Qk4/8/1Q6/8/2K4Q/2Q5/Q7/8 b - - 0 1", WDLScore::WDLLoss),
    ("3K4/7P/8/8/8/2P3PP/P1k5/8 b - - 0 1", WDLScore::WDLLoss),
    ("3Q4/8/3Q4/8/1Q6/2Q4K/4k2Q/8 b - - 0 1", WDLScore::WDLLoss),
    ("3k1Q1Q/7K/8/7Q/8/8/5QQ1/8 b - - 0 1", WDLScore::WDLLoss),
    ("3k4/K2P3P/1P6/8/8/6P1/3P4/8 b - - 0 1", WDLScore::WDLLoss),
    ("3k4/K7/1P2P3/6P1/4P1P1/8/8/8 w - - 0 1", WDLScore::WDLWin),
    ("4K3/3P4/8/7P/1P6/5P2/1P6/1k6 b - - 0 1", WDLScore::WDLLoss),
    ("4K3/5P2/kP6/8/2PP4/8/7P/8 b - - 0 1", WDLScore::WDLLoss),
    ("4K3/8/2P5/2P5/4P3/8/2P2P2/5k2 b - - 0 1", WDLScore::WDLLoss),
    ("4Q1Q1/8/4k3/8/1Q5Q/5K2/4Q3/8 b - - 0 1", WDLScore::WDLLoss),
    ("4k1K1/3P2PP/2P5/8/8/8/4P3/8 b - - 0 1", WDLScore::WDLLoss),
    ("4k3/8/5PP1/8/8/3P4/2P3P1/3K4 w - - 0 1", WDLScore::WDLWin),
    ("4k3/8/P1P5/5P2/8/6K1/1P3P2/8 w - - 0 1", WDLScore::WDLWin),
    ("5K2/8/2P5/8/k3P3/4PP2/P7/8 w - - 0 1", WDLScore::WDLWin),
    ("5Q2/6k1/8/1Q6/5K2/3Q4/3Q2Q1/8 b - - 0 1", WDLScore::WDLLoss),
    ("5Q2/8/8/k3Q3/3Q4/2Q5/7K/7Q b - - 0 1", WDLScore::WDLLoss),
    ("5QQ1/6Q1/8/8/8/4Q1k1/3Q4/3K4 b - - 0 1", WDLScore::WDLLoss),
    ("5k2/5P2/5PP1/6P1/8/1P6/8/K7 b - - 0 1", WDLScore::WDLDraw),
    ("5k2/PP6/1P6/8/4P3/3K4/7P/8 b - - 0 1", WDLScore::WDLLoss),
    ("6K1/3P2P1/3k4/4P3/5P2/1P6/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("6Q1/4k3/1Q6/3Q3Q/6Q1/8/8/2K5 b - - 0 1", WDLScore::WDLDraw),
    ("6Q1/8/7Q/3Q4/5K2/3k4/5Q2/1Q6 b - - 0 1", WDLScore::WDLLoss),
    ("6k1/8/2P5/7P/6P1/K3P3/4P3/8 b - - 0 1", WDLScore::WDLLoss),
    ("6k1/8/3P4/3P4/1P1P4/8/K5P1/8 w - - 0 1", WDLScore::WDLWin),
    ("7K/6P1/2P1PP2/8/3P4/8/8/7k b - - 0 1", WDLScore::WDLLoss),
    ("7K/k7/3P4/1P2P1P1/3P4/8/8/8 w - - 0 1", WDLScore::WDLWin),
    ("7Q/3K4/5Q2/8/8/8/1k3Q1Q/2Q5 b - - 0 1", WDLScore::WDLLoss),
    ("7Q/4Q3/3K4/1Q6/k3Q3/8/5Q2/8 b - - 0 1", WDLScore::WDLLoss),
    ("7Q/7Q/7Q/1k6/Q7/K5Q1/8/8 b - - 0 1", WDLScore::WDLLoss),
    ("7k/2P5/8/7P/2P5/1PK5/4P3/8 b - - 0 1", WDLScore::WDLLoss),
    ("7k/7P/8/3P2P1/2P5/1P6/8/3K4 w - - 0 1", WDLScore::WDLWin),
    ("7k/8/6P1/P3K1P1/5P2/8/6P1/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/1P6/7P/6P1/5k1P/8/P6K/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/1PP5/8/3k4/P7/8/2K1P1P1/8 b - - 0 1", WDLScore::WDLLoss),
    ("8/1PP5/P7/8/P2PK3/8/8/1k6 b - - 0 1", WDLScore::WDLLoss),
    ("8/1k5P/8/8/1P1PP1P1/8/8/K7 b - - 0 1", WDLScore::WDLLoss),
    ("8/1k6/8/2KP4/1P4P1/5P2/1P6/8 w - - 0 1", WDLScore::WDLWin),
    ("8/2P1P3/1P6/5K2/3P4/1P6/k7/8 b - - 0 1", WDLScore::WDLLoss),
];

#[test]
fn probe_matches_cpp_verdicts_for_5_6_and_7_men() {
    let mut coverage = Coverage::new("probe_matches_cpp_verdicts_for_5_6_and_7_men");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let mut failures = Vec::new();
    let mut checked = 0;

    for &(fen, expect) in GOLDEN {
        let mut pos = setup(fen);
        let mut err = ProbeState::Fail;
        let wdl = syzygy::probe_wdl(&mut pos, &mut err);

        if !coverage.probe_available(&tb, fen, err != ProbeState::Fail, false) {
            continue;
        }
        checked += 1;
        if wdl != expect {
            failures.push(format!("  {fen}: expected {expect:?}, got {wdl:?}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {checked} probes disagree with the C++ engine:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // Guard against the whole table skipping silently because the directory
    // holds only 3-4 man files.
    assert!(
        checked >= 40,
        "only {checked} of {} golden positions had tables available; point \
         SYZYGY_PATH at a set including 5-man tables to exercise this",
        GOLDEN.len()
    );
}

/// `init` must be safe to call more than once.
///
/// GUIs re-send options routinely. `initialize_maps` published seven constant
/// tables through `OnceLock::set().unwrap()`, and `TB_TABLES` did the same, so a
/// second `setoption name SyzygyPath` aborted the engine outright. C++ simply
/// rebuilds. This calls `init` several times, including with an empty path and a
/// nonexistent one, and then checks probing still works.
#[test]
fn init_is_idempotent() {
    let mut coverage = Coverage::new("init_is_idempotent");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    // This test is the reason `TB_LOCK` exists: it deliberately tears the
    // global tables down and rebuilds them several times.
    let _tb = init_all(&tb);

    syzygy::init(&tb);
    syzygy::init("");
    syzygy::init("/nonexistent/definitely-not-a-tablebase-directory");
    syzygy::init(&tb);

    let mut pos = setup("8/8/8/8/8/4k3/8/R3K3 w - - 0 1");
    let mut err = ProbeState::Fail;
    let wdl = syzygy::probe_wdl(&mut pos, &mut err);

    assert_eq!(err, ProbeState::Ok, "probing broken after repeated init");
    assert_eq!(wdl, WDLScore::WDLWin, "wrong verdict after repeated init");
}

/// `MAX_CARDINALITY` must match the largest table actually present, not merely
/// be "at least 4".
#[test]
fn max_cardinality_matches_the_files_on_disk() {
    let mut coverage = Coverage::new("max_cardinality_matches_the_files_on_disk");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    // Largest man-count among the .rtbw files present, from their names:
    // "KRPvKR.rtbw" -> 6 men.
    let mut expected = 0usize;
    for entry in std::fs::read_dir(&tb).expect("tablebase dir").flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(stem) = name.strip_suffix(".rtbw") else {
            continue;
        };
        let men = stem.chars().filter(|c| c.is_ascii_uppercase()).count();
        expected = expected.max(men);
    }

    let actual = syzygy::MAX_CARDINALITY.load(core::sync::atomic::Ordering::Relaxed) as usize;
    assert_eq!(
        actual, expected,
        "MAX_CARDINALITY is {actual} but the largest table on disk has {expected} men"
    );
}

// ---------------------------------------------------------------------------
// Cursed wins and blessed losses
// ---------------------------------------------------------------------------

/// Positions that are won (or lost) on the board but drawn under the 50-move
/// rule, with the DTZ the tables record for them.
///
/// The `GOLDEN` table above deliberately excluded these, so the entire 50-move
/// branch of the probe -- the `+100` offset in `probe_dtz`, `dtz_before_zeroing`
/// returning `±101`, and the `WDLCursedWin`/`WDLBlessedLoss` arms of
/// `map_score_dtz` -- had no coverage at all. That is the subtlest code in the
/// file and it was the one part left untested.
///
/// Derived independently of this port, and confirmed twice:
///
/// 1. Found with the C++ engine by the *definitional* property rather than by
///    reading a score off a table: a position is a cursed win exactly when it is
///    drawish with `Syzygy50MoveRule true` and winning with it `false`. 3000
///    candidate positions in cursed-rich material (KNNvKP, KBBvKN and relatives)
///    yielded these 39 flips.
/// 2. Cross-checked against python-chess 1.11.2, a Syzygy implementation sharing
///    no code with either engine. All 39 agree on both the verdict and the DTZ,
///    and every |DTZ| exceeds 100 as a cursed result must.
///
/// The same cross-check was run over the 160-entry `GOLDEN` table: 160 agree,
/// 0 disagree.
const AMBIGUOUS: &[(&str, WDLScore, i32)] = &[
    (
        "2K3k1/8/2N5/8/8/8/3b2b1/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -106,
    ),
    (
        "2k5/8/7b/K7/2N5/8/8/3b4 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -110,
    ),
    (
        "3B2B1/8/n7/8/8/3k4/8/5K2 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -118,
    ),
    (
        "3b4/5K2/6bk/8/8/8/8/3N4 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -114,
    ),
    (
        "3k4/8/8/5nB1/8/1B6/5K2/8 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -106,
    ),
    (
        "4N3/8/1K4k1/8/2b5/8/1b6/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -102,
    ),
    (
        "5b2/8/N7/4K3/8/1b6/5k2/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -112,
    ),
    (
        "6B1/8/8/K7/5n2/8/7k/2B5 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -106,
    ),
    (
        "6N1/8/bk4K1/8/7b/8/8/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -104,
    ),
    (
        "6k1/8/4n3/6n1/8/P4K2/8/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -105,
    ),
    (
        "7k/2n5/8/8/4P3/8/3K2n1/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -111,
    ),
    (
        "8/1Kn5/8/B7/4k3/8/8/3B4 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -112,
    ),
    (
        "8/5N2/4b3/8/8/7k/1K1b4/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -106,
    ),
    (
        "8/6Nk/8/5K2/8/5b2/8/2b5 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -122,
    ),
    (
        "8/8/2k5/8/4n2K/B7/8/5B2 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -112,
    ),
    (
        "8/8/3kb3/8/5b2/2N5/2K5/8 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -102,
    ),
    (
        "8/8/6k1/3b4/8/8/K4b2/3N4 w - - 0 1",
        WDLScore::WDLBlessedLoss,
        -104,
    ),
    (
        "8/8/7B/8/8/1k3n2/6B1/K7 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -122,
    ),
    (
        "8/Bn6/6K1/8/8/6k1/2B5/8 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -104,
    ),
    (
        "8/p7/3K4/8/5N2/8/4N1k1/8 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -103,
    ),
    (
        "k7/2pN4/4N3/8/8/8/8/1K6 b - - 0 1",
        WDLScore::WDLBlessedLoss,
        -101,
    ),
    (
        "1K6/8/8/8/p7/N7/6k1/5N2 w - - 0 1",
        WDLScore::WDLCursedWin,
        118,
    ),
    (
        "1N6/4K3/7p/8/k3N3/8/8/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        104,
    ),
    (
        "1N6/5k2/8/7p/8/8/3K1N2/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        110,
    ),
    (
        "3k2K1/p7/8/8/1N4N1/8/8/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        104,
    ),
    (
        "3n4/8/8/8/8/3Kn2P/k7/8 b - - 0 1",
        WDLScore::WDLCursedWin,
        110,
    ),
    (
        "5Bk1/8/8/8/6n1/8/8/K4B2 w - - 0 1",
        WDLScore::WDLCursedWin,
        111,
    ),
    (
        "8/2n1n3/8/8/P6K/8/8/3k4 b - - 0 1",
        WDLScore::WDLCursedWin,
        102,
    ),
    (
        "8/2p1N3/8/k7/8/8/8/1K3N2 w - - 0 1",
        WDLScore::WDLCursedWin,
        102,
    ),
    (
        "8/5b1K/8/8/8/k3N3/8/b7 b - - 0 1",
        WDLScore::WDLCursedWin,
        105,
    ),
    (
        "8/7k/3n4/3P4/8/8/5K2/1n6 b - - 0 1",
        WDLScore::WDLCursedWin,
        128,
    ),
    (
        "8/8/3bN1K1/8/8/5b2/8/4k3 b - - 0 1",
        WDLScore::WDLCursedWin,
        105,
    ),
    (
        "8/8/4k3/K2N3N/7p/8/8/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        124,
    ),
    (
        "8/8/8/2N4p/6N1/k2K4/8/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        106,
    ),
    (
        "8/8/8/3n2B1/BK1k4/8/8/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        119,
    ),
    (
        "8/8/8/5b2/k7/5N2/8/2bK4 b - - 0 1",
        WDLScore::WDLCursedWin,
        103,
    ),
    (
        "8/8/8/7k/5n1b/8/7K/1N6 b - - 0 1",
        WDLScore::WDLCursedWin,
        134,
    ),
    (
        "k6N/8/8/1K6/1b6/8/6b1/8 b - - 0 1",
        WDLScore::WDLCursedWin,
        117,
    ),
    (
        "k7/8/4N3/2p3N1/5K2/8/8/8 w - - 0 1",
        WDLScore::WDLCursedWin,
        110,
    ),
];

/// Cursed wins and blessed losses must be reported as such, not rounded to a
/// win, a loss, or a draw.
#[test]
fn probe_reports_cursed_wins_and_blessed_losses() {
    let mut coverage = Coverage::new("probe_reports_cursed_wins_and_blessed_losses");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let mut failures = Vec::new();
    let mut checked = 0;

    for &(fen, expect, _) in AMBIGUOUS {
        let mut pos = setup(fen);
        let mut err = ProbeState::Fail;
        let wdl = syzygy::probe_wdl(&mut pos, &mut err);

        if !coverage.probe_available(&tb, fen, err != ProbeState::Fail, false) {
            continue;
        }
        checked += 1;
        if wdl != expect {
            failures.push(format!("  {fen}: expected {expect:?}, got {wdl:?}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {checked} ambiguous probes are wrong:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        checked >= 20,
        "only {checked} of {} ambiguous positions had tables available; these are \
         all 5-man, so point SYZYGY_PATH at a set including 5-man tables",
        AMBIGUOUS.len()
    );
}

/// `probe_dtz` must carry the 50-move offset for ambiguous positions.
///
/// A cursed win gets `+100` added and a blessed loss `-100`, which is what makes
/// |DTZ| > 100 the signature of "wins on the board, draws under the rule". Drop
/// the offset and the value silently becomes a legal-looking ordinary DTZ, so
/// nothing downstream would notice.
#[test]
fn dtz_carries_the_fifty_move_offset() {
    let mut coverage = Coverage::new("dtz_carries_the_fifty_move_offset");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let mut failures = Vec::new();
    let mut checked = 0;

    for &(fen, expect, expect_dtz) in AMBIGUOUS {
        let mut pos = setup(fen);
        let mut err = ProbeState::Fail;
        let dtz = syzygy::probe_dtz(&mut pos, &mut err);
        if !coverage.probe_available(&tb, fen, err != ProbeState::Fail, false) {
            continue;
        }
        checked += 1;

        if dtz != expect_dtz {
            failures.push(format!("  {fen}: DTZ {dtz}, tables say {expect_dtz}"));
            continue;
        }
        // The property that makes the value meaningful, stated separately so a
        // future table regeneration cannot quietly weaken it.
        if dtz.abs() <= 100 {
            failures.push(format!("  {fen}: |DTZ| = {} is not > 100", dtz.abs()));
        }
        let want_positive = expect == WDLScore::WDLCursedWin;
        if (dtz > 0) != want_positive {
            failures.push(format!(
                "  {fen}: DTZ {dtz} has the wrong sign for {expect:?}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} DTZ values disagree with the tables ({checked} checked):\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        checked >= 20,
        "only {checked} ambiguous positions had tables available"
    );
}

// ---------------------------------------------------------------------------
// Root move ranking
// ---------------------------------------------------------------------------

/// A minimal options map holding the three Syzygy options, with the same
/// defaults `Engine` installs (the defaults registered by `Engine::new`).
///
/// Built directly rather than through `Engine` so these tests do not need a
/// network file, and so the option values under test are visible right here.
fn syzygy_options(rule50: bool, probe_limit: i32) -> stockcrab::ucioption::OptionsMap {
    use stockcrab::ucioption::{OptionsMap, UciOption};
    let mut o = OptionsMap::new();
    o.add(UciOption::new_spin("SyzygyProbeDepth", 1, 1, 100, None));
    o.add(UciOption::new_check("Syzygy50MoveRule", rule50, None));
    o.add(UciOption::new_spin(
        "SyzygyProbeLimit",
        probe_limit,
        0,
        7,
        None,
    ));
    o
}

fn root_moves_of(pos: &Position) -> stockcrab::search::RootMoves {
    use stockcrab::movegen::{GenType, generate};
    use stockcrab::thread::RootMove;
    generate(pos, GenType::All)
        .as_slice()
        .iter()
        .map(|&m| RootMove::new(m))
        .collect()
}

/// The WDL of each root move's resulting position, from the mover's point of
/// view, probed independently of the ranking under test.
fn wdl_after_each_move(pos: &mut Position) -> Vec<(String, WDLScore)> {
    use stockcrab::movegen::{GenType, generate};
    let chess960 = pos.is_chess960();
    let moves: Vec<_> = generate(pos, GenType::All).as_slice().to_vec();

    moves
        .iter()
        .map(|&m| {
            // SAFETY: the move is legal.
            unsafe { pos.do_move(m) };
            let mut err = ProbeState::Fail;
            let wdl = -syzygy::probe_wdl(pos, &mut err);
            // SAFETY: the move is legal.
            unsafe { pos.undo_move(m) };
            // `ZeroingBestMove` and `ChangeSTM` are informational states the
            // probe sets on success; only `Fail` means no answer.
            assert_ne!(err, ProbeState::Fail, "probe failed for a root move");
            (stockcrab::uci::UciEngine::move_str(m, chess960), wdl)
        })
        .collect()
}

/// `rank_root_moves` must order the root moves by their true tablebase value.
///
/// A wrong ranking changes the move played, which `scripts/syzygy_diff.sh` would
/// notice, but that needs the C++ engine and a tablebase set; this test pins the
/// ranking itself.
///
/// The invariant asserted here is strict monotonicity in WDL class: if move `a`
/// reaches a strictly better outcome than move `b`, `a` must rank strictly
/// higher. Ordering *within* a class is a DTZ tiebreak and is deliberately not
/// asserted, since that is exactly what upstream tunes.
#[test]
fn root_ranking_is_monotonic_in_the_true_tablebase_value() {
    let mut coverage = Coverage::new("root_ranking_is_monotonic_in_the_true_tablebase_value");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    // Positions with a genuine mix of winning, drawing and losing root moves --
    // a position where every move wins would satisfy monotonicity vacuously.
    let fens = [
        // Chosen with python-chess so each has root moves spanning three WDL
        // classes, and so the set as a whole covers all five. Obvious endgames
        // are useless here: in KRvK every legal move still wins, which satisfies
        // monotonicity without testing anything.
        "8/K7/8/4Q3/1R6/8/5k2/7q w - - 0 1", // 44 moves: win / draw / loss
        "6r1/Q2q4/8/8/8/5k2/8/1K6 b - - 0 1", // 43 moves: win / draw / loss
        "1N4b1/1k6/8/8/8/4b3/7K/8 b - - 0 1", // 24 moves: win / cursed win / draw
        "8/8/b7/2N3K1/8/8/k7/b7 w - - 0 1",  // 15 moves: draw / blessed loss / loss
        "8/1K6/8/2k3B1/8/3B1n2/8/8 b - - 0 1", // 12 moves: draw / blessed loss / loss
    ];

    let mut seen_classes = std::collections::BTreeSet::new();
    let mut positions_checked = 0;

    for fen in fens {
        let mut pos = setup(fen);
        {
            // Skip cleanly if this material is not in the directory, rather
            // than failing a coverage test for a missing download.
            let mut err = ProbeState::Fail;
            syzygy::probe_wdl(&mut pos, &mut err);
            if !coverage.probe_available(&tb, fen, err != ProbeState::Fail, false) {
                continue;
            }
        }
        positions_checked += 1;
        let truth = wdl_after_each_move(&mut pos);

        let mut rm = root_moves_of(&pos);
        let options = syzygy_options(true, 7);
        let config = syzygy::rank_root_moves(&options, &mut pos, &mut rm, false);

        assert!(
            config.root_in_tb,
            "{fen}: position is in the tablebases but root_in_tb is false"
        );

        let chess960 = pos.is_chess960();
        let ranked: Vec<(String, i32, WDLScore)> = rm
            .iter()
            .map(|m| {
                let s = stockcrab::uci::UciEngine::move_str(m.pv[0], chess960);
                let wdl = truth
                    .iter()
                    .find(|(t, _)| *t == s)
                    .expect("move in both lists")
                    .1;
                (s, m.tb_rank, wdl)
            })
            .collect();

        // Something to actually distinguish.
        let distinct: std::collections::BTreeSet<_> = ranked.iter().map(|r| r.2).collect();
        assert!(
            distinct.len() >= 3,
            "{fen}: root moves span only {distinct:?}, too few classes to order"
        );
        seen_classes.extend(distinct);

        for a in &ranked {
            for b in &ranked {
                if a.2 > b.2 {
                    assert!(
                        a.1 > b.1,
                        "{fen}: {} reaches {:?} and {} only {:?}, but ranks {} <= {}",
                        a.0,
                        a.2,
                        b.0,
                        b.2,
                        a.1,
                        b.1
                    );
                }
            }
        }
    }

    assert!(
        positions_checked >= 3,
        "only {positions_checked} of the ranking positions had tables available"
    );
    // Without this, losing the ability to rank (say) cursed wins correctly would
    // go unnoticed as long as the remaining classes still ordered.
    assert!(
        seen_classes.len() == 5,
        "the ranking positions only exercised {seen_classes:?}; all five WDL \
         classes must appear or the ordering is only partly tested"
    );
}

/// After ranking, the move list must be sorted best-first.
///
/// The search takes `root_moves[0]` as the tablebase-preferred move, so an
/// unsorted list is played directly.
#[test]
fn root_ranking_sorts_the_move_list() {
    let mut coverage = Coverage::new("root_ranking_sorts_the_move_list");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let mut pos = setup("8/8/8/8/8/4k3/8/R3K3 w - - 0 1");
    let mut rm = root_moves_of(&pos);
    let options = syzygy_options(true, 7);
    let config = syzygy::rank_root_moves(&options, &mut pos, &mut rm, true);

    assert!(config.root_in_tb);
    for w in rm.windows(2) {
        assert!(
            w[0].tb_rank >= w[1].tb_rank,
            "root moves not sorted: rank {} before {}",
            w[0].tb_rank,
            w[1].tb_rank
        );
    }
    // A won position must rank its best move above the worst available.
    assert!(
        rm[0].tb_rank > rm[rm.len() - 1].tb_rank,
        "KRvK ranked every move equally; the rook can be given away here"
    );
    assert_eq!(
        config.cardinality, 0,
        "a resolved root position must switch tablebase probing off inside the search"
    );
}

/// `SyzygyProbeLimit` below the piece count must disable root probing entirely,
/// leaving every rank at zero rather than a stale or partial ordering.
#[test]
fn probe_limit_below_the_piece_count_disables_root_ranking() {
    let mut coverage = Coverage::new("probe_limit_below_the_piece_count_disables_root_ranking");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let mut pos = setup("8/8/8/8/8/4k3/8/R3K3 w - - 0 1"); // 3 men
    let mut rm = root_moves_of(&pos);
    let options = syzygy_options(true, 2); // below 3
    let config = syzygy::rank_root_moves(&options, &mut pos, &mut rm, false);

    assert!(
        !config.root_in_tb,
        "root_in_tb set despite SyzygyProbeLimit = 2"
    );
    assert!(
        rm.iter().all(|m| m.tb_rank == 0),
        "ranks left non-zero after root probing was declined"
    );
}

/// Castling rights must disable root probing.
///
/// Syzygy tables do not encode castling, so a position that still has rights is
/// not representable and C++ skips the probe outright (tbprobe.cpp:1585). Ignore
/// that and the engine plays a confidently wrong "tablebase" move.
#[test]
fn castling_rights_disable_root_ranking() {
    let mut coverage = Coverage::new("castling_rights_disable_root_ranking");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let mut pos = setup("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    let mut rm = root_moves_of(&pos);
    let options = syzygy_options(true, 7);
    let config = syzygy::rank_root_moves(&options, &mut pos, &mut rm, false);

    assert!(
        !config.root_in_tb,
        "root probed a position that still has castling rights"
    );
    assert!(rm.iter().all(|m| m.tb_rank == 0));

    // Same position without the rights must probe, so the check above is really
    // about castling and not about the position being unreachable.
    let mut pos = setup("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mut rm = root_moves_of(&pos);
    let config = syzygy::rank_root_moves(&options, &mut pos, &mut rm, false);
    assert!(
        config.root_in_tb,
        "the same position without castling rights failed to probe"
    );
}

/// Turning the 50-move rule off must promote cursed wins to full wins.
///
/// This is the ranking half of what `AMBIGUOUS` covers for probing, and it is
/// the behaviour that identified those positions in the first place: with
/// `Syzygy50MoveRule true` the C++ engine calls them drawish, with it `false`
/// it calls them won.
#[test]
fn disabling_the_fifty_move_rule_promotes_cursed_wins() {
    let mut coverage = Coverage::new("disabling_the_fifty_move_rule_promotes_cursed_wins");
    let Some(tb) = coverage.tables(false) else {
        return;
    };
    let _tb = init_all(&tb);

    let cursed: Vec<&str> = AMBIGUOUS
        .iter()
        .filter(|(_, w, _)| *w == WDLScore::WDLCursedWin)
        .map(|(f, _, _)| *f)
        .collect();

    let mut checked = 0;
    for fen in cursed {
        let mut pos = setup(fen);
        {
            let mut probe_err = ProbeState::Fail;
            syzygy::probe_wdl(&mut pos, &mut probe_err);
            if !coverage.probe_available(&tb, fen, probe_err != ProbeState::Fail, false) {
                continue;
            }
        }
        checked += 1;

        let mut with_rule = root_moves_of(&pos);
        let c1 = syzygy::rank_root_moves(&syzygy_options(true, 7), &mut pos, &mut with_rule, false);
        let mut without_rule = root_moves_of(&pos);
        let c2 = syzygy::rank_root_moves(
            &syzygy_options(false, 7),
            &mut pos,
            &mut without_rule,
            false,
        );

        assert!(c1.root_in_tb && c2.root_in_tb, "{fen}: root probe failed");
        assert!(
            c1.use_rule50 && !c2.use_rule50,
            "{fen}: the option did not reach Config"
        );

        // Won on the board either way, so the best rank stays positive...
        assert!(
            with_rule[0].tb_rank > 0 && without_rule[0].tb_rank > 0,
            "{fen}: a cursed win should still rank as the best available outcome"
        );
        // ...but only without the rule is it scored as an actual win.
        assert!(
            without_rule[0].tb_score > with_rule[0].tb_score,
            "{fen}: score {} with the 50-move rule, {} without -- disabling it changed nothing",
            with_rule[0].tb_score,
            without_rule[0].tb_score
        );
        assert!(
            with_rule[0].tb_score < stockcrab::types::Value::from(200),
            "{fen}: cursed win scored {} under the 50-move rule, which is not drawish",
            with_rule[0].tb_score
        );
    }

    assert!(
        checked >= 10,
        "only {checked} cursed positions had tables available"
    );
}

// ---------------------------------------------------------------------------
// Corrupt tablebase files
// ---------------------------------------------------------------------------

/// Write a directory of deliberately malformed tables; it is removed when the result drops.
///
/// Synthesised rather than derived from real files, so these tests need no
/// tablebase download and run in every configuration.
fn corrupt_tb_dir(tag: &str, undetectable: bool) -> tempfile::TempDir {
    // A real .rtbw begins 71 E8 23 5D; .rtbz begins D7 66 0C A5. The format's
    // only other integrity check is that the file length satisfies
    // `len % 64 == 16` (syzygy.rs `map`), which 208 does.
    const WDL_MAGIC: [u8; 4] = [0x71, 0xE8, 0x23, 0x5D];
    const GOOD_LEN: usize = 208;

    let dir = tempfile::Builder::new()
        .prefix(&format!("stockcrab-corrupt-tb-{tag}-"))
        .tempdir()
        .expect("create temp dir");

    let body = |n: usize| -> Vec<u8> {
        let mut v = WDL_MAGIC.to_vec();
        v.extend((4..n).map(|i| (i * 37 + 11) as u8));
        v
    };

    if undetectable {
        // Valid magic and valid length, garbage contents. Nothing in the format
        // can detect this.
        std::fs::write(dir.path().join("KRvK.rtbw"), body(GOOD_LEN)).unwrap();
    } else {
        // Truncated to nothing.
        std::fs::write(dir.path().join("KQvK.rtbw"), []).unwrap();
        // Right magic, length violating the `% 64 == 16` rule -- what an
        // interrupted download looks like.
        std::fs::write(dir.path().join("KRvK.rtbw"), body(1000)).unwrap();
        // Right length, wrong magic -- e.g. a DTZ file renamed to .rtbw, which
        // the magic check must reject.
        let mut wrong = body(GOOD_LEN);
        wrong[0] ^= 0xFF;
        std::fs::write(dir.path().join("KBvK.rtbw"), wrong).unwrap();
        // A .rtbz carrying the WDL magic instead of its own.
        std::fs::write(dir.path().join("KNvK.rtbz"), body(GOOD_LEN)).unwrap();
    }
    dir
}

/// Run the current test binary again, in a child process, with `var` set.
///
/// Isolation is not optional here. The release profile sets `panic = "abort"`
/// (the `[profile.release]` section of Cargo.toml), so nothing this code can do
/// is catchable in-process: a regression that dereferences a bad header takes
/// the whole test binary down with a bare signal and no indication of which test
/// did it. In a child, the same regression is an ordinary assertion failure
/// naming the cause.
fn run_child(test_name: &str, var: &str, value: &std::path::Path) -> std::process::Output {
    std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(var, value)
        .env_remove("SYZYGY_PATH")
        .output()
        .expect("spawn child test process")
}

/// Detectably corrupt files must be rejected, leaving the engine running.
///
/// A truncated or mistyped tablebase file is a routine accident -- the full
/// 6-man set is 72 GB, so interrupted downloads are the normal case. What must
/// not happen is the engine reading past the mapping or aborting: it should
/// behave exactly as if the files were absent.
///
/// Needs no tablebases; the malformed files are synthesised.
#[test]
fn corrupt_tablebase_files_are_rejected_cleanly() {
    let _coverage = Coverage::new("corrupt_tablebase_files_are_rejected_cleanly");
    const VAR: &str = "STOCKCRAB_CORRUPT_TB_DIR";

    if let Ok(dir) = std::env::var(VAR) {
        // Child role.
        stockcrab::bitboard::init();
        stockcrab::movegen::init();
        syzygy::init(&dir);

        // `MAX_CARDINALITY` counts tables by *filename*: mapping is deferred to
        // the first probe, in both engines (`TBTables::add` only checks that
        // the file exists). So it is expected to be non-zero here even though
        // every file is junk -- the rejection happens below, at probe time.
        let announced = syzygy::MAX_CARDINALITY.load(core::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            announced, 3,
            "expected the three synthetic 3-man names to be counted"
        );

        // Probing must report "no answer", not a fabricated one.
        for fen in [
            "8/8/8/8/8/4k3/8/R3K3 w - - 0 1",
            "8/8/8/8/8/4k3/8/Q3K3 w - - 0 1",
            "8/8/8/8/8/4k3/8/B3K3 w - - 0 1",
            "8/8/8/8/8/4k3/8/N3K3 w - - 0 1",
        ] {
            let mut pos = setup(fen);
            let mut err = ProbeState::Fail;
            let wdl = syzygy::probe_wdl(&mut pos, &mut err);
            assert_eq!(
                err,
                ProbeState::Fail,
                "{fen}: answered {wdl:?} from a corrupt table"
            );
        }

        // Reset: an empty path must leave the module in a clean, table-free state.
        syzygy::init("");
        println!("CHILD-OK");
        return;
    }

    let dir = corrupt_tb_dir("detectable", false);
    let out = run_child(
        "corrupt_tablebase_files_are_rejected_cleanly",
        VAR,
        dir.path(),
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("CHILD-OK"),
        "engine did not survive a directory of corrupt tablebase files\n\
         exit: {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Corruption the format cannot detect must never produce a silent verdict.
///
/// A file with the right magic and the right length but a garbage body passes
/// every check the Syzygy format has, and the header walk then computes
/// pointers from nonsense. The C++ engine dies here too (verified: it aborts on
/// the same file), so crashing is upstream behaviour and not a port defect --
/// this test deliberately does not demand that it be fixed, because hardening
/// it would mean diverging from the structure upstream patches land against.
///
/// What it does demand is the thing that would actually be dangerous: the probe
/// must not come back with a confident, wrong answer. Losing a won endgame
/// because a table was silently misread is far worse than a loud failure.
///
/// Accepted outcomes: the child may die by signal (the C++ engine aborts on the
/// same file). The parent fails only if the child returns a verdict, or if the
/// probe is never reached.
#[test]
fn undetectable_corruption_never_yields_a_silent_verdict() {
    let _coverage = Coverage::new("undetectable_corruption_never_yields_a_silent_verdict");
    const VAR: &str = "STOCKCRAB_GARBAGE_TB_DIR";

    if let Ok(dir) = std::env::var(VAR) {
        // Child role. This may abort or take a signal; that is an accepted
        // outcome, so nothing here needs to be recoverable.
        stockcrab::bitboard::init();
        stockcrab::movegen::init();
        syzygy::init(&dir);

        let mut pos = setup("8/8/8/8/8/4k3/8/R3K3 w - - 0 1");

        // Flushed before the probe so the parent can tell "declined or died
        // trying" apart from "never got here", which would make the assertion
        // below pass for the wrong reason.
        use std::io::Write;
        println!("CHILD-PROBING");
        std::io::stdout().flush().unwrap();

        let mut err = ProbeState::Fail;
        let wdl = syzygy::probe_wdl(&mut pos, &mut err);
        if err != ProbeState::Fail {
            println!("CHILD-ANSWERED {wdl:?}");
        } else {
            println!("CHILD-DECLINED");
        }
        return;
    }

    let dir = corrupt_tb_dir("garbage", true);
    let out = run_child(
        "undetectable_corruption_never_yields_a_silent_verdict",
        VAR,
        dir.path(),
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("CHILD-PROBING"),
        "the child never reached the probe, so this proves nothing:\nexit: {:?}\n{stdout}",
        out.status
    );
    assert!(
        !stdout.contains("CHILD-ANSWERED"),
        "a table with a garbage body produced a verdict rather than declining:\n{stdout}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("panicked"),
        "probe panicked on a garbage table:\n{stderr}"
    );
}
