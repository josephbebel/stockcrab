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

//! UCI session tests driving the engine binary: pondering, stop/go sequences, `EvalFile`,
//! Syzygy root ranking, search limits and `Debug Log File`. Every blocking read has a
//! deadline; Drop kills the child even when a bug makes the command loop hang.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

struct UciProcess {
    child: Child,
    lines: Receiver<String>,
}

impl UciProcess {
    fn new() -> Self {
        assert!(
            std::path::Path::new(stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME).is_file(),
            "these tests require the default network; run scripts/fetch-net.sh"
        );
        let mut child = Command::new(env!("CARGO_BIN_EXE_stockcrab"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn engine");
        let stdout = child.stdout.take().unwrap();
        let (send, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if send.send(line).is_err() {
                    break;
                }
            }
        });
        let mut engine = Self { child, lines };
        engine.send("isready");
        engine.until(|line| line == "readyok");
        engine
    }

    fn send(&mut self, commands: &str) {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{commands}").expect("send UCI command");
        stdin.flush().unwrap();
    }

    fn until(&self, mut predicate: impl FnMut(&str) -> bool) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut seen = Vec::new();
        loop {
            let wait = deadline.saturating_duration_since(Instant::now());
            let line = self.lines.recv_timeout(wait).unwrap_or_else(|error| {
                panic!(
                    "engine output deadline/EOF: {error}; last lines: {:?}",
                    &seen[seen.len().saturating_sub(8)..]
                )
            });
            let done = predicate(&line);
            seen.push(line);
            if done {
                return seen;
            }
        }
    }

    fn quit(&mut self) {
        self.send("quit");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "engine quit failed: {status}");
                return;
            }
            assert!(Instant::now() < deadline, "quit deadline exceeded");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn ponderhit_after_depth_completion_releases_search() {
    let mut engine = UciProcess::new();
    engine.send("position startpos\ngo ponder depth 1");
    engine.until(|line| line.starts_with("info depth 1 "));
    engine.send("ponderhit\nisready");
    let mut ready = false;
    let mut bestmove = false;
    engine.until(|line| {
        ready |= line == "readyok";
        bestmove |= line.starts_with("bestmove ");
        ready && bestmove
    });
    // No stop may release the pondering search for this assertion. Only after
    // its bestmove arrives can the next search verify command-loop reuse.
    engine.send("go depth 1");
    engine.until(|line| line.starts_with("bestmove "));
    engine.send("stop\nisready");
    engine.until(|line| line == "readyok");
    engine.quit();
}

#[test]
fn ponderhit_during_search_obeys_time_control() {
    let mut engine = UciProcess::new();
    engine.send("position startpos\ngo ponder wtime 1000 btime 1000");
    engine.until(|line| line.starts_with("info depth 1 "));
    engine.send("ponderhit");
    engine.until(|line| line.starts_with("bestmove "));
    engine.send("stop\nisready");
    engine.until(|line| line == "readyok");
    engine.quit();
}

#[test]
fn rapid_stop_go_keeps_the_previous_stop_request() {
    let mut engine = UciProcess::new();
    // Pin both the command reader and workers to one CPU to exercise the
    // reported scheduling interleaving reliably on Linux.
    #[cfg(target_os = "linux")]
    {
        // SAFETY: masks are fully initialized, have the expected libc size,
        // and the process id denotes our live child.
        unsafe {
            let mut allowed: libc::cpu_set_t = std::mem::zeroed();
            assert_eq!(
                libc::sched_getaffinity(
                    engine.child.id() as i32,
                    std::mem::size_of_val(&allowed),
                    &mut allowed
                ),
                0
            );
            let cpu = (0..libc::CPU_SETSIZE as usize)
                .find(|&cpu| libc::CPU_ISSET(cpu, &allowed))
                .unwrap();
            libc::CPU_ZERO(&mut allowed);
            libc::CPU_SET(cpu, &mut allowed);
            assert_eq!(
                libc::sched_setaffinity(
                    engine.child.id() as i32,
                    std::mem::size_of_val(&allowed),
                    &allowed
                ),
                0
            );
        }
        // Recreate workers so they inherit the command reader's affinity.
        engine.send("setoption name Threads value 1\nisready");
        engine.until(|line| line == "readyok");
    }
    for _ in 0..20 {
        engine.send("position startpos\ngo infinite depth 1");
        engine.until(|line| line.starts_with("info depth 1 "));
        engine.send("stop\ngo depth 1\nisready");
        let mut ready = false;
        let mut bestmoves = 0;
        engine.until(|line| {
            ready |= line == "readyok";
            bestmoves += usize::from(line.starts_with("bestmove "));
            ready && bestmoves == 2
        });
    }
    engine.quit();
}

#[test]
fn evalfile_absolute_path_loads_the_requested_network() {
    let mut engine = UciProcess::new();
    let network = std::fs::canonicalize(stockcrab::evaluate::EVAL_FILE_DEFAULT_NAME).unwrap();
    engine.send(&format!(
        "setoption name EvalFile value {}\nisready",
        network.display()
    ));
    engine.until(|line| line == "readyok");
    engine.send("position startpos\ngo depth 1");
    engine.until(|line| line.starts_with("bestmove "));
    engine.quit();
}

#[test]
fn syzygy_root_ranking_takes_the_repetition_draw() {
    let mut engine = UciProcess::new();
    let tables = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/syzygy");
    engine.send(&format!(
        "setoption name SyzygyPath value {}\nisready",
        tables.display()
    ));
    engine.until(|line| line == "readyok");
    engine.send("position fen 7k/8/8/8/8/8/R7/K7 w - - 0 1 moves a2b2 h8g8 b2a2 g8h8 a2b2 h8g8 b2a2\ngo depth 4");
    let lines = engine.until(|line| line.starts_with("bestmove "));
    assert!(
        lines.last().unwrap().starts_with("bestmove g8h8"),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|line| line.contains(" score cp 0 ")),
        "{lines:?}"
    );
    engine.quit();
}

#[test]
fn position_command_during_ponder_accepts_the_next_search() {
    let mut engine = UciProcess::new();
    engine.send("position startpos moves g1f3 g8f6 f3g1 f6g8\ngo infinite depth 1");
    engine.until(|line| line.starts_with("info depth 1 "));
    engine.send("position startpos moves e2e4 e7e5\nstop");
    engine.until(|line| line.starts_with("bestmove "));
    engine.send("go depth 2");
    engine.until(|line| line.starts_with("bestmove "));
    engine.quit();
}

#[test]
fn syzygy_winner_avoids_repeating_and_keeps_mate_at_rule50_boundary() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/syzygy");
    let mut engine = UciProcess::new();
    engine.send(&format!(
        "setoption name SyzygyPath value {}",
        fixtures.display()
    ));
    engine.send("position fen 7k/8/8/8/8/8/R7/K7 w - - 0 1 moves a2b2 h8g8 b2a2 g8h8 a2b2 h8g8 b2a2 g8h8\ngo depth 4");
    let lines = engine.until(|line| line.starts_with("bestmove "));
    assert!(
        !lines.last().unwrap().starts_with("bestmove a2b2"),
        "{lines:?}"
    );
    engine.send("position fen 7k/R7/6K1/8/8/8/8/8 w - - 99 1\ngo depth 4");
    let lines = engine.until(|line| line.starts_with("bestmove "));
    assert!(
        lines.last().unwrap().starts_with("bestmove a7a8"),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|line| line.contains("score mate 1")),
        "{lines:?}"
    );
    engine.quit();
}

#[test]
fn search_limits_terminate_with_subprocess_deadlines() {
    let mut engine = UciProcess::new();
    for command in [
        "go movetime 100",
        "go nodes 20000",
        "go depth 4",
        "go wtime 1000 btime 1000",
    ] {
        engine.send(&format!("position startpos\n{command}"));
        engine.until(|line| line.starts_with("bestmove "));
    }
    engine.send("setoption name nodestime value 100\nposition startpos\ngo wtime 1000 btime 1000");
    engine.until(|line| line.starts_with("bestmove "));
    engine.quit();
}

#[test]
fn debug_log_records_protocol_and_can_be_disabled() {
    let mut engine = UciProcess::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("protocol.log");
    engine.send(&format!(
        "setoption name Debug Log File value {}\nisready",
        path.display()
    ));
    engine.until(|line| line == "readyok");
    engine.send("position startpos\ngo depth 1");
    engine.until(|line| line.starts_with("bestmove "));
    // isready follows the output log write and confirms it is on disk.
    engine.send("eval\ngo perft 1\nsetoption name Missing Review Option value 1\nisready");
    engine.until(|line| line == "readyok");
    let log = std::fs::read_to_string(&path).unwrap();
    assert!(log.contains("<< position startpos\n"), "{log}");
    assert!(log.contains("<< go depth 1\n"), "{log}");
    assert!(log.contains(">> info depth 1 "), "{log}");
    assert!(log.contains(">> bestmove "), "{log}");
    assert!(log.contains(">> readyok\n"), "{log}");
    assert!(log.contains(">> Final evaluation"), "{log}");
    assert!(log.contains(">> e2e4: 1\n"), "{log}");
    assert!(
        log.contains(">> No such option: Missing Review Option\n"),
        "{log}"
    );
    engine.send("setoption name Debug Log File value <empty>\nisready");
    engine.until(|line| line == "readyok");
    let disabled = std::fs::read(&path).unwrap();
    engine.send("isready");
    engine.until(|line| line == "readyok");
    engine.quit();
    assert_eq!(std::fs::read(&path).unwrap(), disabled);
}
