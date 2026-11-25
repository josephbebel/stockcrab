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

//! Exercise unwind recovery under an external deadline: a broken worker must
//! fail the child, rather than hang the entire test runner.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use stockcrab::engine::{Engine, LazyNumaReplicated};
use stockcrab::evaluate::Networks;
use stockcrab::history::SharedHistories;
use stockcrab::search::LimitsType;
use stockcrab::thread::{
    NumaConfig, OptionalThreadToNumaNodeBinder, SearchControl, SharedState, Thread, ThreadPool,
    UpdateContext,
};
use stockcrab::tt::TranspositionTable;
use stockcrab::ucioption::{OptionsMap, UciOption};

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn worker_panics_are_reported_and_workers_remain_usable() {
    const CHILD_MODE: &str = "STOCKCRAB_WORKER_PANIC_CHILD";
    if let Ok(mode) = std::env::var(CHILD_MODE) {
        if mode.starts_with("tt-") {
            tt_batch_probe(&mode);
        } else if mode.starts_with("custom") {
            custom_job_probe(mode == "custom-resubmit");
        } else {
            search_callback_probe(&mode);
        }
        return;
    }

    for mode in [
        "custom-wait",
        "custom-resubmit",
        "search-start",
        "search-info",
        "search-drop",
        "tt-submit",
        "tt-resize",
        "tt-wait",
    ] {
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "worker_panics_are_reported_and_workers_remain_usable",
                "--nocapture",
            ])
            .env(CHILD_MODE, mode)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let mut child = ChildGuard(child);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success(), "{mode} failed: {status}");
                break;
            }
            assert!(Instant::now() < deadline, "{mode} hung");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Hold the first clearing callback open while the real pool reports another
/// worker's panic. The clear must keep borrowing its allocation until release.
struct GatedPool<'a> {
    pool: &'a ThreadPool,
    gate: std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    entered: std::sync::mpsc::Sender<()>,
    drained: Arc<std::sync::atomic::AtomicBool>,
    panic_in_callback: bool,
}
// SAFETY: submission and waits delegate to the real pool. Each wrapper calls
// its callback once; waits drain it even when the worker reports a panic.
unsafe impl stockcrab::tt::ThreadPool for GatedPool<'_> {
    fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }
    fn run_on_thread<F: FnOnce() + Send + 'static>(&self, i: usize, f: F) {
        if i == 0 {
            let gate = self.gate.lock().unwrap().take().unwrap();
            let entered = self.entered.clone();
            let drained = self.drained.clone();
            self.pool.run_on_thread(i, move || {
                entered.send(()).unwrap();
                gate.recv().unwrap();
                f();
                drained.store(true, std::sync::atomic::Ordering::SeqCst);
            });
        } else if self.panic_in_callback {
            self.pool.run_on_thread(i, move || {
                f();
                std::panic::panic_any("batch witness");
            });
        } else {
            self.pool.run_on_thread(i, f);
        }
    }
    fn wait_on_thread(&self, i: usize) {
        self.pool.wait_on_thread(i);
    }
}

fn tt_batch_probe(mode: &str) {
    let mut opts = OptionsMap::new();
    opts.add(UciOption::new_spin("Threads", 2, 1, 2, None));
    opts.add(UciOption::new_string("NumaPolicy", "none", None));
    let state = SharedState::new(
        Arc::new(RwLock::new(opts)),
        Arc::new(RwLock::new(ThreadPool::new())),
        Arc::new(RwLock::new(TranspositionTable::new())),
        Arc::new(LazyNumaReplicated::new(Networks::new())),
        Arc::new(SearchControl::new()),
        Arc::new(SharedHistories::new(2)),
    );
    let updates = Arc::new(RwLock::new(UpdateContext {
        on_update_no_moves: Box::new(|_| {}),
        on_update_full: Box::new(|_| {}),
        on_iter: Box::new(|_| {}),
        on_bestmove: Box::new(|_, _| {}),
        on_start: Box::new(|| {}),
    }));
    let mut pool = ThreadPool::new();
    pool.set(Arc::new(NumaConfig::single_node()), state, updates);
    let mut tt = TranspositionTable::new();
    tt.resize(1, &pool);
    tt.new_search();
    if mode != "tt-wait" {
        pool.run_on_thread(1, || std::panic::panic_any("batch witness"));
        // Leave this completed panic unconsumed: the batched clear or resize below must
        // still drain every region before it reports it.
    }
    let (release, gate) = std::sync::mpsc::channel();
    let (entered, started) = std::sync::mpsc::channel();
    let drained = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let failed = std::sync::atomic::AtomicBool::new(false);
    let gated = GatedPool {
        pool: &pool,
        gate: std::sync::Mutex::new(Some(gate)),
        entered,
        drained: drained.clone(),
        panic_in_callback: mode == "tt-wait",
    };
    std::thread::scope(|scope| {
        let failed = &failed;
        scope.spawn(move || {
            started.recv_timeout(Duration::from_secs(5)).unwrap();
            std::thread::sleep(Duration::from_millis(100));
            assert!(
                !failed.load(std::sync::atomic::Ordering::SeqCst),
                "clear unwound with a callback still borrowing the TT"
            );
            release.send(()).unwrap();
        });
        let failure = catch_unwind(AssertUnwindSafe(|| {
            if mode == "tt-resize" {
                tt.resize(2, &gated);
            } else {
                tt.clear(&gated);
            }
        }))
        .expect_err("worker panic must be reported after draining the batch");
        failed.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(failure.downcast_ref::<&str>(), Some(&"batch witness"));
        assert!(drained.load(std::sync::atomic::Ordering::SeqCst));
    });
    // Both submitted and skipped regions must be initialized, and usable after catch.
    for i in 1..=64u64 {
        assert!(!tt.probe(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).0);
    }
    tt.clear(&pool);
    pool.run_on_thread(1, || {});
    pool.wait_on_thread(1);
}

fn custom_job_probe(report_on_resubmit: bool) {
    let state = SharedState::new(
        Arc::new(RwLock::new(OptionsMap::new())),
        Arc::new(RwLock::new(ThreadPool::new())),
        Arc::new(RwLock::new(TranspositionTable::new())),
        Arc::new(LazyNumaReplicated::new(Networks::new())),
        Arc::new(SearchControl::new()),
        Arc::new(SharedHistories::new(1)),
    );
    let thread = Thread::new(state, None, 0, OptionalThreadToNumaNodeBinder::new(0));
    thread.run_custom_job(|| std::panic::panic_any("custom job witness"));
    let failure = catch_unwind(AssertUnwindSafe(|| {
        if report_on_resubmit {
            thread.run_custom_job(|| {});
        } else {
            thread.wait_for_search_finished();
        }
    }))
    .expect_err("job panic must reach the caller");
    assert_eq!(failure.downcast_ref::<&str>(), Some(&"custom job witness"));

    let (tx, rx) = std::sync::mpsc::channel();
    thread.run_custom_job(move || {
        tx.send(()).unwrap();
    });
    thread.wait_for_search_finished();
    rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(thread.worker().lock().unwrap().is_some());

    // Multiple waiters must all finish, and new jobs must wake the idle worker.
    for _ in 0..25 {
        let (release, blocked) = std::sync::mpsc::channel();
        thread.run_custom_job(move || {
            blocked.recv().unwrap();
        });
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| thread.wait_for_search_finished());
            }
            release.send(()).unwrap();
        });
    }
}

fn search_callback_probe(mode: &str) {
    let mut engine = Engine::new(None);
    engine.set_option("name NumaPolicy value none");
    engine.set_option("name Threads value 2");
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    if mode == "search-start" {
        engine.set_on_start(|| std::panic::panic_any("search witness"));
    } else {
        engine.set_on_update_full(move |_| {
            entered_tx.send(()).unwrap();
            std::panic::panic_any("search witness")
        });
    }
    engine.go(LimitsType {
        depth: 4,
        ..LimitsType::default()
    });
    if mode == "search-drop" {
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(engine);
        return;
    }
    let failure = catch_unwind(AssertUnwindSafe(|| engine.wait_for_search_finished()))
        .expect_err("search callback panic must reach the caller");
    assert_eq!(failure.downcast_ref::<&str>(), Some(&"search witness"));
    engine.set_on_start(|| {});
    engine.set_on_update_full(|_| {});
    assert!(
        engine
            .set_position(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                &[]
            )
            .is_none()
    );
    engine.go(LimitsType {
        depth: 4,
        ..LimitsType::default()
    });
    engine.wait_for_search_finished();
    assert!(
        engine.nodes_searched() > 0,
        "restored worker must search again"
    );
}
