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

//! Bounded UCI gameplay checks for Lazy SMP. Thread scheduling can change the
//! chosen move, score and node count, so these tests check legal output and the
//! command lifecycle rather than a fixed search result.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use stockcrab::movegen::{GenType, generate};
use stockcrab::position::Position;
use stockcrab::types::Move;
use stockcrab::uci::UciEngine;

const SESSION_TIMEOUT: Duration = Duration::from_secs(100);
const PHASE_TIMEOUT: Duration = Duration::from_secs(25);
const MAX_PHASE_LINES: usize = 4096;

type WriteRequest = (String, Sender<Result<(), String>>);

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Owns the process so an assertion, timeout or broken pipe always kills and reaps it.
/// Output and input have separate helper threads: neither pipe can block the test
/// thread past a deadline, including if the child stops reading stdin.
struct UciProcess {
    child: ChildGuard,
    lines: Receiver<String>,
    writer: Sender<WriteRequest>,
    session_end: Instant,
}

impl UciProcess {
    fn new() -> Self {
        let mut engine = Self::spawn(Command::new(env!("CARGO_BIN_EXE_stockcrab")));
        engine.send("isready");
        engine.until("initial readyok", |line| line == "readyok", true);
        engine
    }

    fn spawn(mut command: Command) -> Self {
        let mut child = ChildGuard(
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn UCI subprocess"),
        );

        let stdout = child.0.stdout.take().expect("engine stdout");
        let (line_tx, lines) = mpsc::sync_channel(1024);
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if line_tx.send(line).is_err() {
                    break;
                }
            }
        });

        let mut stdin = child.0.stdin.take().expect("engine stdin");
        let (writer, requests) = mpsc::channel::<WriteRequest>();
        std::thread::spawn(move || {
            for (command, ack) in requests {
                let result = writeln!(stdin, "{command}")
                    .and_then(|()| stdin.flush())
                    .map_err(|error| error.to_string());
                let failed = result.is_err();
                let _ = ack.send(result);
                if failed {
                    break;
                }
            }
        });

        Self {
            child,
            lines,
            writer,
            session_end: Instant::now() + SESSION_TIMEOUT,
        }
    }

    fn deadline(&self) -> Instant {
        (Instant::now() + PHASE_TIMEOUT).min(self.session_end)
    }

    fn send(&mut self, command: &str) {
        let deadline = self.deadline();
        let (ack_tx, ack_rx) = mpsc::channel();
        self.writer
            .send((command.to_owned(), ack_tx))
            .expect("engine writer thread closed");
        let remaining = remaining(deadline).expect("UCI command write deadline expired");
        ack_rx
            .recv_timeout(remaining)
            .unwrap_or_else(|error| panic!("UCI write deadline/EOF for {command:?}: {error}"))
            .unwrap_or_else(|error| panic!("UCI write failed for {command:?}: {error}"));
    }

    /// A stream of `info` lines cannot restart the deadline or grow memory
    /// without bound. The result form also supports the timeout negative control.
    fn receive_until(
        &self,
        label: &str,
        deadline: Instant,
        mut done: impl FnMut(&str) -> bool,
        forbid_bestmove: bool,
    ) -> Result<Vec<String>, String> {
        let mut seen = Vec::new();
        loop {
            let wait = remaining(deadline).ok_or_else(|| {
                format!("{label}: deadline expired; last lines: {:?}", tail(&seen))
            })?;
            let line = self.lines.recv_timeout(wait).map_err(|error| match error {
                RecvTimeoutError::Timeout => {
                    format!("{label}: deadline expired; last lines: {:?}", tail(&seen))
                }
                RecvTimeoutError::Disconnected => {
                    format!("{label}: output EOF; last lines: {:?}", tail(&seen))
                }
            })?;
            if forbid_bestmove && line.starts_with("bestmove") {
                return Err(format!(
                    "{label}: bestmove arrived before release: {line:?}"
                ));
            }
            let complete = done(&line);
            seen.push(line);
            if seen.len() > MAX_PHASE_LINES {
                return Err(format!("{label}: excessive output"));
            }
            if complete {
                return Ok(seen);
            }
        }
    }

    fn until(
        &self,
        label: &str,
        done: impl FnMut(&str) -> bool,
        forbid_bestmove: bool,
    ) -> Vec<String> {
        self.receive_until(label, self.deadline(), done, forbid_bestmove)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn ready_lines_without_bestmove(&mut self) -> Vec<String> {
        self.send("isready");
        self.until("readyok without bestmove", |line| line == "readyok", true)
    }

    fn ready_without_bestmove(&mut self) {
        let _ = self.ready_lines_without_bestmove();
    }

    fn search_result(&mut self, label: &str, root: &Position) -> usize {
        let lines = self.until(label, |line| line.starts_with("bestmove"), false);
        let pv_count = validate_search_output(&lines, root)
            .unwrap_or_else(|error| panic!("{label}: {error}; last lines: {:?}", tail(&lines)));
        // A barrier after each completed search also catches a second bestmove
        // before the next `go`, which could otherwise be mistaken for its reply.
        self.ready_without_bestmove();
        pv_count
    }

    fn quit(&mut self) {
        self.send("quit");
        let deadline = self.deadline();
        // Drain through stdout EOF. A delayed second bestmove after the last
        // `readyok` must fail rather than disappear during child shutdown.
        loop {
            let wait = remaining(deadline).expect("engine stdout EOF deadline expired");
            match self.lines.recv_timeout(wait) {
                Ok(line) => assert!(
                    !line.starts_with("bestmove"),
                    "unsolicited bestmove during quit: {line:?}"
                ),
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => panic!("engine stdout EOF deadline expired"),
            }
        }
        loop {
            if let Some(status) = self.child.0.try_wait().expect("poll engine exit") {
                assert!(status.success(), "engine quit returned {status}");
                return;
            }
            assert!(Instant::now() < deadline, "engine quit deadline expired");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn tail(lines: &[String]) -> &[String] {
    &lines[lines.len().saturating_sub(8)..]
}

fn remaining(deadline: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
}

fn legal_move(position: &mut Position, token: &str) -> Result<(), String> {
    let mv = UciEngine::to_move(position, token);
    let legal: stockcrab::movegen::MoveList = generate(position, GenType::All);
    if mv == Move::none() || !legal.contains(mv) || !position.do_move_owned(mv) {
        return Err(format!("illegal UCI move {token:?}"));
    }
    Ok(())
}

fn validate_search_output(lines: &[String], root: &Position) -> Result<usize, String> {
    let mut bestmoves = 0;
    let mut pv_count = 0;
    for line in lines {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.first() == Some(&"bestmove") {
            bestmoves += 1;
            if fields.len() != 2 && (fields.len() != 4 || fields[2] != "ponder") {
                return Err(format!("malformed bestmove response: {line:?}"));
            }
            let mut position = root.clone();
            legal_move(&mut position, fields[1])?;
            if fields.len() == 4 {
                legal_move(&mut position, fields[3])?;
            }
        } else if line.starts_with("info depth ")
            && let Some(pv) = fields.iter().position(|&field| field == "pv")
            && pv + 1 < fields.len()
        {
            let mut position = root.clone();
            for token in &fields[pv + 1..] {
                legal_move(&mut position, token)?;
            }
            pv_count += 1;
        }
    }
    if bestmoves != 1 {
        return Err(format!("expected one bestmove, saw {bestmoves}"));
    }
    Ok(pv_count)
}

fn root_after(moves: &[&str]) -> Position {
    let mut root = Position::default();
    for &token in moves {
        legal_move(&mut root, token).expect("legal setup move");
    }
    root
}

#[test]
fn timed_smp_searches_and_thread_changes_return_legal_play() {
    if common::skip_without_network("timed_smp_searches_and_thread_changes_return_legal_play") {
        return;
    }
    let mut engine = UciProcess::new();
    let mut checked_pvs = 0;
    for (threads, setup, moves) in [
        (2, "position startpos", &[][..]),
        (
            4,
            "position startpos moves e2e4 e7e5 g1f3 b8c6",
            &["e2e4", "e7e5", "g1f3", "b8c6"][..],
        ),
        (8, "position startpos", &[][..]),
    ] {
        engine.send(&format!("setoption name Threads value {threads}"));
        engine.ready_without_bestmove();
        engine.send(setup);
        engine.send("go movetime 200"); // depth remains unset: exercise SMP final voting.
        checked_pvs += engine.search_result(
            &format!("Threads={threads} timed search"),
            &root_after(moves),
        );
    }
    assert!(
        checked_pvs > 0,
        "timed SMP searches produced no PV to validate"
    );
    engine.quit();
}

#[test]
fn ponderhit_stop_and_restart_keep_one_reply_per_search() {
    if common::skip_without_network("ponderhit_stop_and_restart_keep_one_reply_per_search") {
        return;
    }
    let mut engine = UciProcess::new();
    engine.send("setoption name Threads value 4");
    engine.ready_without_bestmove();
    engine.send("position startpos");
    engine.send("go ponder movetime 200");
    let mut pondering = engine.until(
        "ponder depth 1",
        |line| line.starts_with("info depth 1 "),
        true,
    );
    pondering.extend(engine.ready_lines_without_bestmove());
    engine.send("ponderhit");
    let result = engine.until(
        "ponderhit bestmove",
        |line| line.starts_with("bestmove"),
        false,
    );
    pondering.extend(result);
    assert!(validate_search_output(&pondering, &root_after(&[])).unwrap() > 0);
    engine.ready_without_bestmove();

    engine.send("setoption name Threads value 8");
    engine.ready_without_bestmove();
    engine.send("position startpos moves e2e4 e7e5");
    engine.send("go infinite");
    let mut infinite = engine.until(
        "infinite depth 1",
        |line| line.starts_with("info depth 1 "),
        true,
    );
    infinite.extend(engine.ready_lines_without_bestmove());
    engine.send("stop");
    let result = engine.until(
        "stopped bestmove",
        |line| line.starts_with("bestmove"),
        false,
    );
    infinite.extend(result);
    assert!(validate_search_output(&infinite, &root_after(&["e2e4", "e7e5"])).unwrap() > 0);
    engine.ready_without_bestmove();

    engine.send("setoption name Threads value 2");
    engine.ready_without_bestmove();
    engine.send("position startpos");
    engine.send("go nodes 2000");
    engine.search_result("restarted node-limited search", &root_after(&[]));
    engine.quit();
}

#[test]
fn output_validator_rejects_duplicate_and_illegal_moves() {
    let root = root_after(&[]);
    let lines = vec!["bestmove e2e4".into(), "bestmove d2d4".into()];
    assert!(
        validate_search_output(&lines, &root)
            .unwrap_err()
            .contains("expected one bestmove")
    );
    let lines = vec!["bestmove e2e5".into()];
    assert!(
        validate_search_output(&lines, &root)
            .unwrap_err()
            .contains("illegal UCI move")
    );
    let lines = vec!["info depth 1 pv e2e5".into(), "bestmove e2e4".into()];
    assert!(
        validate_search_output(&lines, &root)
            .unwrap_err()
            .contains("illegal UCI move")
    );
}

#[test]
fn chatty_child_mode() {
    if std::env::var_os("STOCKCRAB_SMP_CHATTY_CHILD").is_none() {
        return;
    }
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    loop {
        writeln!(output, "info depth 1 pv e2e4").expect("write fake info");
        output.flush().expect("flush fake info");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn chatty_child_cannot_extend_the_absolute_deadline() {
    let mut command = Command::new(std::env::current_exe().expect("test executable path"));
    command
        .args(["--exact", "chatty_child_mode", "--nocapture"])
        .env("STOCKCRAB_SMP_CHATTY_CHILD", "1");
    let child = UciProcess::spawn(command);
    child.until(
        "chatty child startup",
        |line| line.starts_with("info depth "),
        false,
    );

    let deadline = Instant::now() + Duration::from_millis(100);
    let error = child
        .receive_until(
            "chatty child without bestmove",
            deadline,
            |line| line.starts_with("bestmove"),
            false,
        )
        .expect_err("info spam must not satisfy a bestmove wait");
    assert!(error.contains("deadline expired"), "{error}");
    assert!(
        error.contains("info depth"),
        "the child did not keep sending info: {error}"
    );
    // ChildGuard kills and reaps the still-chatty test subprocess here.
}
