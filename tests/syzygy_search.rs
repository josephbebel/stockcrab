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

//! Search-level Syzygy regression tests.
//!
//! `tests/syzygy_probe.rs` checks what a probe returns; this file checks what
//! the search does with it, by driving the built binary and comparing the last
//! `info` line and the `bestmove` with the C++ engine's output.
//!
//! The roots here are deliberately NOT in the tablebases (six men against
//! 3-4-5 tables, or five men with `SyzygyProbeLimit 4`), so the decisive score
//! comes from probes inside the search. C++ `output_pv` extends such a PV with
//! `syzygy_extend_pv` exactly as it does for a root that is in the tables. The
//! extended PV becomes `previousPV` and steers the next iteration's move
//! ordering, so skipping the extension changes the node count, `tbhits` and
//! `seldepth`, not only the printed line.
//!
//! The expected values were taken from the C++ reference at the commit pinned
//! in `config/upstream.env`, with one thread and the full 3-4-5 table set.
//! Point `SYZYGY_PATH` at that set (`scripts/fetch-syzygy.sh`) to run these
//! tests; without it, or with only a subset, developer runs skip. Setting
//! `STOCKCRAB_REQUIRE_SYZYGY=1` makes missing prerequisites fail instead.

mod common;
#[path = "common/syzygy.rs"]
mod syzygy_coverage;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

/// Longest the engine may go without producing a line before the test fails.
const LINE_TIMEOUT: Duration = Duration::from_secs(60);

/// What the C++ engine prints for one position.
struct Golden {
    fen: &'static str,
    /// Extra `setoption` arguments, e.g. `name SyzygyProbeLimit value 4`.
    options: &'static [&'static str],
    depth: u32,
    seldepth: &'static str,
    score: &'static str,
    nodes: &'static str,
    tbhits: &'static str,
    pv: &'static str,
    bestmove: &'static str,
}

/// The token after `key` in a UCI `info` line.
fn field<'a>(line: &'a str, key: &str) -> &'a str {
    let mut tokens = line.split_whitespace();
    tokens.find(|&t| t == key);
    tokens
        .next()
        .unwrap_or_else(|| panic!("no `{key}` in: {line}"))
}

/// Search `golden.fen` with one thread and return the stdout lines up to `bestmove`.
fn search(tb: &str, golden: &Golden) -> Vec<String> {
    let mut input =
        format!("setoption name Threads value 1\nsetoption name SyzygyPath value {tb}\n");
    for option in golden.options {
        input.push_str(&format!("setoption {option}\n"));
    }
    input.push_str(&format!(
        "position fen {}\ngo depth {}\n",
        golden.fen, golden.depth
    ));

    let mut child = Command::new(env!("CARGO_BIN_EXE_stockcrab"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn engine");
    // Stdin stays open until the bestmove has been read: closing it makes the
    // engine quit, which would stop the search early.
    let mut stdin = child.stdin.take().expect("stdin");
    stdin.write_all(input.as_bytes()).expect("write");

    // Read on a helper thread so a hung engine fails the test instead of
    // blocking it forever.
    let reader = BufReader::new(child.stdout.take().expect("stdout"));
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut lines = Vec::new();
    loop {
        match rx.recv_timeout(LINE_TIMEOUT) {
            Ok(line) => {
                let done = line.starts_with("bestmove");
                lines.push(line);
                if done {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("engine produced no output for {LINE_TIMEOUT:?}; lines so far: {lines:?}");
            }
        }
    }

    let _ = stdin.write_all(b"quit\n");
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    lines
}

fn check(test: &'static str, golden: &Golden) {
    let mut coverage = syzygy_coverage::Coverage::new(test);
    let Some(tb) = coverage.tables(true) else {
        return;
    };
    if !coverage.network() {
        return;
    }

    let lines = search(&tb, golden);
    let prefix = format!("info depth {} ", golden.depth);
    let info = lines
        .iter()
        .rfind(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line for {}; lines: {lines:?}", golden.fen));
    let fen = golden.fen;

    assert_eq!(field(info, "cp"), golden.score, "score for {fen}: {info}");
    assert_eq!(
        field(info, "nodes"),
        golden.nodes,
        "nodes for {fen}: {info}"
    );
    assert_eq!(
        field(info, "tbhits"),
        golden.tbhits,
        "tbhits for {fen}: {info}"
    );
    assert_eq!(
        field(info, "seldepth"),
        golden.seldepth,
        "seldepth for {fen}: {info}"
    );
    let pv = info
        .split_once(" pv ")
        .map(|(_, pv)| pv.trim_end())
        .unwrap_or_else(|| panic!("no pv for {fen}: {info}"));
    assert_eq!(pv, golden.pv, "pv for {fen}");
    assert_eq!(
        lines.last().map(|l| l.trim_end()),
        Some(golden.bestmove),
        "bestmove for {fen}"
    );
}

/// Six men, one capture away from the tables. The PV extension affects subsequent
/// move ordering, so compare the full search output with the pinned reference.
#[test]
fn six_man_root_extends_the_pv_and_keeps_the_node_count() {
    check(
        "six_man_root_extends_the_pv_and_keeps_the_node_count",
        &Golden {
            fen: "5K2/5Q2/4q3/6P1/8/2p3k1/8/8 w - - 0 1",
            options: &[],
            depth: 12,
            seldepth: "14",
            score: "19999",
            nodes: "8781",
            tbhits: "496",
            pv: "f7e6 g3h4 g5g6 h4h5 g6g7 c3c2 g7g8q h5h4 e6g4",
            bestmove: "bestmove f7e6 ponder g3h4",
        },
    );
}

/// Six men requiring a tablebase extension beyond the searched PV. Check both
/// the extended line and the search statistics against the pinned reference.
#[test]
fn six_man_root_prints_the_extended_pv() {
    check(
        "six_man_root_prints_the_extended_pv",
        &Golden {
            fen: "1k5K/8/8/5B2/5Rn1/8/7p/8 b - - 0 1",
            options: &[],
            depth: 14,
            seldepth: "15",
            score: "19993",
            nodes: "4686",
            tbhits: "243",
            pv: "h2h1q f5h7 h1a1 h8g8 g4f6 g8f8 f6h7 f8e7 a1e5 e7d7 e5g7 f4f7 g7f7 d7d6 f7f5 d6c6 \
                 h7g5 c6b6 f5d5 b6a6 b8c7 a6a7 d5a2",
            bestmove: "bestmove h2h1q ponder f5h7",
        },
    );
}

/// Five men with the probe limit below the man count: the tables hold the root
/// position, but `SyzygyProbeLimit` keeps it out of the root probe (without the limit the
/// root probe gives `cp 20000` with 6 tbhits). The search reaches the four-man tables and
/// the 25-ply PV is the tablebase extension of the seldepth-15 search line.
#[test]
fn probe_limit_below_the_root_man_count_still_extends_the_pv() {
    check(
        "probe_limit_below_the_root_man_count_still_extends_the_pv",
        &Golden {
            // From SFNNv19 on, the former position (8/8/5r2/8/8/P2p4/3k4/K7 w) is either
            // undecided or a mate score, which the PV extension skips, at every depth.
            fen: "8/8/8/8/2k5/8/1pK5/1R1r4 w - - 0 1",
            options: &["name SyzygyProbeLimit value 4"],
            depth: 11,
            seldepth: "12",
            score: "19999",
            nodes: "1774",
            tbhits: "82",
            pv: "b1d1 c4c5 c2b2 c5c4 d1d2 c4b4 d2c2 b4a4 c2c4 a4a5 b2a3 a5b5 a3b3 b5a5 c4b4 a5a6 \
                 b3a4 a6a7 a4a5 a7a8 a5b6 a8b8 b4c4 b8a8 c4c8",
            bestmove: "bestmove b1d1 ponder c4c5",
        },
    );
}

/// The PV extension can correct the score: here the root is a tablebase win,
/// but with the 50-move counter at 54 the winning line runs into a draw. C++
/// prints the corrected score (`cp 0`), not the stored one (`cp 20000`).
#[test]
fn pv_extension_that_finds_a_fifty_move_draw_corrects_the_printed_score() {
    check(
        "pv_extension_that_finds_a_fifty_move_draw_corrects_the_printed_score",
        &Golden {
            fen: "8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106",
            options: &[],
            depth: 8,
            seldepth: "9",
            score: "0",
            nodes: "1248",
            tbhits: "26",
            pv: "d4e5 g6g7 e3f5 g7g6 d5e6 g6g5 f5e7 g5h6 e5f6 h6h7 e7g6 h7h6 e6g8 h6h5 g6e5 h5h4 \
                 g8d5 h4h5 d5e4 h5h6 e5g4 h6h5 f6f5 h5h4 f5f4 h4h5 e4b1 h5h4 b1g6 h4h3 g4e3 h3h4 \
                 e3g2 h4h3 f4f3 h3h2 g2f4 h2g1 g6d3 g1h2 f3f2 h2h1 f2g3 h1g1 f4h3 g1h1",
            bestmove: "bestmove d4e5 ponder g6g7",
        },
    );
}
