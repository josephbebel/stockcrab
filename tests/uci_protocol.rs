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

//! Protocol-level regression tests for the UCI output stream.
//!
//! These drive the built binary the way a GUI does and check the bytes it
//! writes, which is the only way to catch framing bugs: the engine can compute
//! everything correctly and still be unusable if a line is malformed.
//!
//! Each response ends with a newline; a line must never contain two responses
//! (for example an `option` line followed by `uciok`, as in
//! `option name EvalFile type string default nn-….nnueuciok`), or a GUI waiting
//! for a line equal to `uciok` never sees it and the handshake hangs forever.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

/// Longest the engine may go without producing a line before the test fails.
const LINE_TIMEOUT: Duration = Duration::from_secs(60);

/// Feed `input` to the engine and collect stdout lines until `sentinel` appears
/// (or, with `None`, until the engine exits and closes stdout).
fn run_until(input: &str, sentinel: Option<&str>) -> Vec<String> {
    run_until_line(input, |line| sentinel.is_some_and(|s| line == s))
}

/// Like [`run_until`], but stops at the first line (trailing whitespace trimmed) for which
/// `is_last` returns true.
fn run_until_line(input: &str, is_last: impl Fn(&str) -> bool) -> Vec<String> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_stockcrab"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn engine");

    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write");

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
                let done = is_last(line.trim_end());
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

    let _ = child.kill();
    let _ = child.wait();
    lines
}

#[test]
fn uci_handshake_ends_with_a_bare_uciok_line() {
    let lines = run_until("uci\nquit\n", Some("uciok"));
    assert!(!lines.is_empty(), "the engine produced no output");

    assert!(
        lines.iter().any(|l| l.trim_end() == "uciok"),
        "no line was exactly \"uciok\"; the handshake would hang a GUI.\n\
         last lines: {:?}",
        &lines[lines.len().saturating_sub(3)..]
    );

    // No line may contain both an option declaration and uciok.
    for l in &lines {
        assert!(
            !(l.contains("option name") && l.contains("uciok")),
            "an option line and uciok share a line: {l:?}"
        );
    }

    // Every advertised option must be one line of the documented shape.
    let options: Vec<&String> = lines
        .iter()
        .filter(|l| l.starts_with("option name"))
        .collect();
    assert!(
        options.len() >= 15,
        "expected the full option list, saw {}",
        options.len()
    );
    for o in options {
        assert!(
            o.contains(" type "),
            "malformed option line (no ` type `): {o:?}"
        );
    }
}

/// C++ runs perft for any nonzero `go perft` value (`if (limits.perft)`); a negative depth
/// counts the root moves. Routing it to the search instead hit `Engine::go`'s
/// `limits.perft == 0` assertion.
#[test]
fn go_perft_with_a_negative_depth_counts_root_moves_like_cpp() {
    if common::skip_without_network("go_perft_with_a_negative_depth_counts_root_moves_like_cpp") {
        return;
    }
    let lines = run_until(
        "uci\nposition startpos\ngo perft -1 depth 3\nisready\nquit\n",
        Some("readyok"),
    );
    let root_moves = lines.iter().filter(|l| l.ends_with(": 1")).count();
    assert_eq!(root_moves, 20, "lines: {lines:?}");
    assert!(
        lines.iter().any(|l| l.trim_end() == "Nodes searched: 20"),
        "lines: {lines:?}"
    );
    assert!(
        lines.iter().all(|l| !l.starts_with("bestmove")),
        "lines: {lines:?}"
    );
}

#[test]
fn every_non_protocol_line_is_an_info_string() {
    // A GUI parses stdout strictly. Anything that is not a known UCI response
    // must be wrapped in `info string`, or it is protocol noise.
    let lines = run_until("uci\nposition startpos\neval\nquit\n", None);
    assert!(!lines.is_empty(), "the engine produced no output");

    const KNOWN_PREFIXES: &[&str] = &[
        "id ",
        "option name",
        "uciok",
        "readyok",
        "info ",
        "bestmove",
        // The `eval` trace is printed as a plain table, exactly as C++ does.
        "NNUE network contributions",
        "NNUE evaluation",
        "Final evaluation",
        "+--",
        "|",
    ];

    for line in &lines {
        let l = line.trim_end();
        if l.is_empty() {
            continue;
        }
        // The startup banner is printed by C++ too, before any UCI exchange.
        if l.starts_with("Stockcrab") {
            continue;
        }
        assert!(
            KNOWN_PREFIXES.iter().any(|p| l.starts_with(p)),
            "unrecognised line on the UCI stream: {l:?}"
        );
    }
}

/// `setoption name EvalFile` must load the named network. Without that, the option string
/// changes but the loaded net does not, the next `go` fails `verify_network` and the process
/// exits before printing a bestmove.
#[test]
fn setoption_evalfile_loads_the_named_network() {
    if common::skip_without_network("setoption_evalfile_loads_the_named_network") {
        return;
    }
    let name = stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME;
    let source = if std::path::Path::new(name).is_file() {
        std::path::PathBuf::from(name)
    } else {
        std::env::current_exe()
            .expect("test binary path")
            .parent()
            .expect("test binary directory")
            .join(name)
    };
    let dir = tempfile::tempdir().unwrap();
    let copy = dir.path().join("evalfile.nnue");
    std::fs::copy(&source, &copy).expect("copy the default network");

    let input = format!(
        "setoption name EvalFile value {}\nposition startpos\ngo depth 1\n",
        copy.display()
    );
    let lines = run_until_line(&input, |line| line.starts_with("bestmove"));

    assert!(
        lines.iter().any(|l| l.starts_with("bestmove")),
        "no bestmove after setoption EvalFile; lines: {lines:?}"
    );
}

/// A `stop` that arrives before the first iteration starts must be answered the
/// way C++ answers it: `output_pv` runs at root depth 0 with every root move
/// still unsearched, so it prints depth `max(1, depth - 1)`, the fallback score
/// 0 and the (empty) previous PV, and the bestmove is the first legal move.
///
/// `go` and `stop` are written together so that `stop` is already buffered
/// when `go` is handled, but the search thread can still win that race and
/// complete an iteration, so the test asserts on the first attempt in which no
/// node was searched.
#[test]
fn stop_before_the_first_iteration_prints_depth_one_and_an_empty_pv() {
    if common::skip_without_network(
        "stop_before_the_first_iteration_prints_depth_one_and_an_empty_pv",
    ) {
        return;
    }

    const ATTEMPTS: usize = 50;
    for _ in 0..ATTEMPTS {
        let lines = run_until_line("position startpos\ngo infinite\nstop\n", |line| {
            line.starts_with("bestmove")
        });
        let infos: Vec<&str> = lines
            .iter()
            .map(|l| l.trim_end_matches(['\r', '\n']))
            .filter(|l| l.starts_with("info depth "))
            .collect();
        let Some(info) = infos.iter().find(|l| l.contains(" nodes 0 ")) else {
            continue;
        };

        assert_eq!(infos.len(), 1, "one info line expected; lines: {lines:?}");
        assert!(
            info.starts_with(
                "info depth 1 seldepth 0 multipv 1 score cp 0 nodes 0 nps 0 hashfull 0 tbhits 0 time "
            ),
            "unexpected info line: {info:?}"
        );
        let pv = info.split_once(" pv").map(|(_, pv)| pv);
        assert_eq!(pv, Some(" "), "the pv must be empty: {info:?}");
        assert_eq!(
            lines.last().map(|l| l.trim_end()),
            Some("bestmove a2a3"),
            "lines: {lines:?}"
        );
        return;
    }
    panic!("`stop` never arrived before the first iteration in {ATTEMPTS} attempts");
}

/// The default network, found where the engine finds it.
fn default_network_path() -> std::path::PathBuf {
    let name = stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME;
    if std::path::Path::new(name).is_file() {
        std::path::PathBuf::from(name)
    } else {
        std::env::current_exe()
            .expect("test binary path")
            .parent()
            .expect("test binary directory")
            .join(name)
    }
}

/// C++ `Thread::ensure_network_replicated` runs synchronously on the caller, so the replica
/// line printed by the next `eval` is deterministic. The Rust pool queues the work on its worker
/// threads, so it must wait for that work: otherwise `eval` races the replication and prints
/// the `Network replica 1` line only some of the time.
#[test]
fn network_replica_line_is_printed_every_time_after_replication() {
    if common::skip_without_network("network_replica_line_is_printed_every_time_after_replication")
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let copy = dir.path().join("replica.nnue");
    std::fs::copy(default_network_path(), &copy).expect("copy the default network");

    let count_replica_lines = |input: &str| {
        run_until(input, None)
            .iter()
            .filter(|l| l.contains("Network replica 1"))
            .count()
    };
    let mut failures = Vec::new();
    for attempt in 0..20 {
        for (what, input) in [
            (
                "EvalFile",
                format!(
                    "setoption name EvalFile value {}\neval\nquit\n",
                    copy.display()
                ),
            ),
            (
                "Threads",
                "setoption name Threads value 2\neval\nquit\n".to_string(),
            ),
        ] {
            let count = count_replica_lines(&input);
            if count != 1 {
                failures.push(format!("attempt {attempt}, {what}: {count} replica lines"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "expected exactly one line: {failures:?}"
    );
}
