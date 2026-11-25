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

//! Same-process `Engine` lifecycle: creating and dropping engines must release
//! their native threads and memory.
//!
//! The ownership graph ThreadPool -> Thread -> Worker -> SharedState -> ThreadPool
//! was a strong `Arc` cycle, so a dropped `Engine` leaked its search threads
//! (parked forever in `idle_loop`), Workers and history tables. A UCI process
//! exits after `quit`, which hid it; a library user creating engines in a loop
//! did not.
//!
//! The two tests share process-wide counters (thread count, RSS), so they are
//! serialised through a mutex.

mod common;

use core::time::Duration;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use stockcrab::engine::Engine;
use stockcrab::misc::now;
use stockcrab::search::LimitsType;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Number of native threads in this process (0 if the platform offers no cheap way to count).
#[cfg(target_os = "linux")]
fn thread_count() -> usize {
    std::fs::read_dir("/proc/self/task").map_or(0, |d| d.count())
}

/// Resident set size in bytes (0 if unavailable).
#[cfg(target_os = "linux")]
fn rss_bytes() -> usize {
    // `VmRSS` is reported in kibibytes, independent of the kernel page size.
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<usize>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// Output of `ps <args> -p <this pid>`, or an empty string when `ps` is unavailable.
#[cfg(all(unix, not(target_os = "linux")))]
fn ps(args: &[&str]) -> String {
    std::process::Command::new("ps")
        .args(args)
        .args(["-p", &std::process::id().to_string()])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// macOS and the BSDs: `ps -M` prints a header and then one line per thread.
#[cfg(all(unix, not(target_os = "linux")))]
fn thread_count() -> usize {
    ps(&["-M"]).lines().count().saturating_sub(1)
}

/// `ps -o rss=` reports kibibytes.
#[cfg(all(unix, not(target_os = "linux")))]
fn rss_bytes() -> usize {
    ps(&["-o", "rss="])
        .trim()
        .parse::<usize>()
        .map_or(0, |kb| kb * 1024)
}

#[cfg(not(unix))]
fn thread_count() -> usize {
    0
}

#[cfg(not(unix))]
fn rss_bytes() -> usize {
    0
}

fn make_engine(threads: i32, hash_mb: i32) -> Engine {
    let mut engine = Engine::new(None);
    {
        let mut opts = engine.get_options_mut();
        opts.get_mut("Threads").unwrap().set(&threads.to_string());
        opts.get_mut("Hash").unwrap().set(&hash_mb.to_string());
    }
    engine.resize_threads();
    assert!(engine.set_position(START, &[]).is_none());
    engine
}

fn limits_with(f: impl FnOnce(&mut LimitsType)) -> LimitsType {
    let mut l = LimitsType::default();
    l.start_time = now();
    f(&mut l);
    l
}

/// Waits (bounded) for the thread count to fall back to `target`.
fn wait_for_threads(target: usize) -> usize {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let n = thread_count();
        if n <= target || Instant::now() > deadline {
            return n;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn repeated_create_search_drop_does_not_leak_threads_or_memory() {
    if common::skip_without_network("repeated_create_search_drop_does_not_leak_threads_or_memory") {
        return;
    }
    let _g = serial();
    if thread_count() == 0 {
        return; // no way to count threads on this platform: nothing to measure
    }
    const CYCLES: usize = 20;

    // Warm-up cycles so one-time allocations (tables, network, the per-thread worker state of
    // the largest configuration used below) are not counted.
    let before_warmup = thread_count();
    for threads in [1, 2] {
        let mut e = make_engine(threads, 4);
        e.go(limits_with(|l| l.depth = 4));
        e.wait_for_search_finished();
    }
    let base_threads = wait_for_threads(before_warmup);
    let base_rss = rss_bytes();

    for i in 0..CYCLES {
        let threads = if i % 2 == 0 { 1 } else { 2 };
        let mut e = make_engine(threads, 4);
        e.go(limits_with(|l| l.nodes = 20_000));
        e.wait_for_search_finished();
        drop(e);
    }

    let end_threads = wait_for_threads(base_threads);
    let growth = rss_bytes().saturating_sub(base_rss);
    assert!(
        end_threads <= base_threads,
        "native threads leaked: {base_threads} before, {end_threads} after {CYCLES} engines"
    );
    assert!(
        growth < 50 * 1024 * 1024,
        "RSS grew by {} MiB over {CYCLES} create/search/drop cycles",
        growth >> 20
    );
}

#[test]
fn dropping_an_engine_during_a_running_search_does_not_hang() {
    if common::skip_without_network("dropping_an_engine_during_a_running_search_does_not_hang") {
        return;
    }
    let _g = serial();
    let base_threads = thread_count();

    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        for threads in [1, 2] {
            let mut e = make_engine(threads, 4);
            e.go(limits_with(|l| l.infinite = 1));
            std::thread::sleep(Duration::from_millis(200));
            // Dropped while the search is still running.
            drop(e);
        }
        tx.send(()).ok();
    });

    match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => panic!("dropping an Engine during `go infinite` hung"),
        Err(RecvTimeoutError::Disconnected) => {
            // The worker exited without reporting: surface its panic.
            worker.join().unwrap();
            panic!("worker exited without reporting");
        }
    }
    worker.join().unwrap();

    if base_threads != 0 {
        let end = wait_for_threads(base_threads);
        assert!(
            end <= base_threads,
            "native threads leaked after dropping a searching engine: {base_threads} -> {end}"
        );
    }
}
