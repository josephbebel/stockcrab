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

//! A GUI that closes the engine's stdout must kill it the way C++ Stockfish dies:
//! silently, by SIGPIPE. Not by a "Broken pipe" panic turned into SIGABRT (and a
//! core dump or crash report) by `panic = "abort"`, and not by carrying on with
//! the write errors ignored, which leaves an engine searching for nobody.
//!
//! Only where `main.rs` restores SIG_DFL for SIGPIPE; see `restore_default_sigpipe`.

#![cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
))]

use std::io::{Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, ChildStderr, Command, ExitStatus, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Held while a test creates pipes and spawns its engine. A child spawned while another test's
/// pipe descriptors are not yet close-on-exec inherits them, and an unrelated engine holding a
/// read end open means the write never fails: no SIGPIPE, and the test fails or times out. On
/// macOS std creates its pipes (`Stdio::piped()` and `io::pipe()`) with `pipe()` and sets
/// `FD_CLOEXEC` afterwards; Linux uses `pipe2(O_CLOEXEC)`.
static SPAWN: Mutex<()> = Mutex::new(());

fn spawn_lock() -> MutexGuard<'static, ()> {
    // A test that panicked while holding the lock must not fail the others.
    SPAWN.lock().unwrap_or_else(|e| e.into_inner())
}

/// Wait for `child` to terminate by itself, killing it and failing after 30 s.
fn wait_for_exit(child: &mut Child, what: &str) -> ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            return st;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("engine did not terminate after {what}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The engine must have been killed by SIGPIPE, with nothing resembling a panic.
fn assert_died_of_sigpipe(status: ExitStatus, stderr: Option<ChildStderr>) {
    let mut err = String::new();
    if let Some(mut stderr) = stderr {
        let _ = stderr.read_to_string(&mut err);
    }
    assert!(!err.contains("panicked"), "engine panicked: {err}");
    assert_ne!(
        status.signal(),
        Some(libc::SIGABRT),
        "engine aborted: {err}"
    );
    assert!(!status.core_dumped(), "engine dumped core");
    assert_eq!(
        status.signal(),
        Some(libc::SIGPIPE),
        "expected death by SIGPIPE, got {status:?}; stderr: {err}"
    );
}

#[test]
fn closed_stdout_terminates_with_sigpipe() {
    let mut child = {
        let _spawn = spawn_lock();
        Command::new(env!("CARGO_BIN_EXE_stockcrab"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine")
    };

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Read a few bytes (so the engine is demonstrably running), then hang up.
    let mut buf = [0u8; 8];
    stdout.read_exact(&mut buf).expect("engine banner");
    drop(stdout);

    // Plenty of output still to come. Write errors are fine once it has died.
    let _ = stdin.write_all(b"uci\nisready\nd\nd\nd\nuci\nuci\nquit\n");
    let _ = stdin.flush();
    drop(stdin);

    let status = wait_for_exit(&mut child, "stdout was closed");
    assert_died_of_sigpipe(status, Some(stderr));
}

/// The very first write, the banner, goes through a different code path from the
/// UCI output.
#[test]
fn stdout_closed_before_the_banner_terminates_with_sigpipe() {
    // `io::pipe` is not atomically close-on-exec on macOS either, so it is created under the lock.
    let mut child = {
        let _spawn = spawn_lock();
        // A pipe whose read end is already closed, so that the engine's first write
        // fails; hanging up on a `Stdio::piped()` after the spawn would race with it.
        let (read_end, write_end) = std::io::pipe().expect("pipe");
        drop(read_end);

        // stdin stays open and silent, so only the closed stdout can end the process. The
        // parent's `write_end` goes away with the `Command` at the end of this statement.
        Command::new(env!("CARGO_BIN_EXE_stockcrab"))
            .stdin(Stdio::piped())
            .stdout(Stdio::from(write_end))
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine")
    };

    let stdin = child.stdin.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let status = wait_for_exit(&mut child, "being started with a closed stdout");
    drop(stdin);
    assert_died_of_sigpipe(status, Some(stderr));
}

/// With stdout gone and stdin still open, a search must not run on unobserved:
/// its first `info` line ends the process.
#[test]
fn closed_stdout_stops_an_infinite_search() {
    let mut child = {
        let _spawn = spawn_lock();
        Command::new(env!("CARGO_BIN_EXE_stockcrab"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine")
    };

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let mut buf = [0u8; 8];
    stdout.read_exact(&mut buf).expect("engine banner");
    drop(stdout);

    let _ = stdin.write_all(b"go infinite\n");
    let _ = stdin.flush();

    // `stdin` is still open here, so EOF on it cannot be what stops the engine.
    let status = wait_for_exit(&mut child, "stdout was closed during go infinite");
    drop(stdin);
    assert_died_of_sigpipe(status, Some(stderr));
}

/// `bench` reports on stderr through `eprintln!`, which panics on a failed write
/// unless SIGPIPE gets there first.
#[test]
fn closed_stderr_terminates_bench_with_sigpipe() {
    let mut child = {
        let _spawn = spawn_lock();
        Command::new(env!("CARGO_BIN_EXE_stockcrab"))
            .arg("bench")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine")
    };

    let mut stderr = child.stderr.take().unwrap();
    let mut buf = [0u8; 8];
    stderr.read_exact(&mut buf).expect("bench output on stderr");
    drop(stderr);

    let status = wait_for_exit(&mut child, "stderr was closed");
    assert_died_of_sigpipe(status, None);
}
