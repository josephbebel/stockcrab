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

#[cfg(not(miri))]
use stockcrab::thread::ThreadPool;
use stockcrab::tt::{ThreadPool as ClearPool, TranspositionTable};
use stockcrab::types::{Bound, Move};

struct InlinePool;
// SAFETY: callbacks complete inline; none can survive submission or a wait.
unsafe impl ClearPool for InlinePool {
    fn num_threads(&self) -> usize {
        1
    }
    fn run_on_thread<F: FnOnce() + Send + 'static>(&self, _: usize, f: F) {
        f();
    }
    fn wait_on_thread(&self, _: usize) {}
}

const KEY: u64 = 0x1234_5678_9abc_def0;

// Miri cannot emulate OS topology detection. The native regression above uses
// the real empty ThreadPool; Miri exercises the same zero-worker clear branch
// through this synchronous scheduler, including the non-Linux heap allocator.
#[cfg(miri)]
struct EmptyPool;
#[cfg(miri)]
// SAFETY: no callbacks can be scheduled or retained by an empty pool.
unsafe impl ClearPool for EmptyPool {
    fn num_threads(&self) -> usize {
        0
    }
    fn run_on_thread<F: FnOnce() + Send + 'static>(&self, _: usize, _: F) {
        unreachable!();
    }
    fn wait_on_thread(&self, _: usize) {
        unreachable!();
    }
}
fn write_without_search(tt: &TranspositionTable) {
    let (_, _, writer) = tt.probe(KEY);
    writer.write(
        KEY,
        100,
        false,
        Bound::EXACT,
        5,
        Move::none(),
        50,
        tt.generation(),
    );
    assert!(tt.probe(KEY).0);
}

#[test]
fn empty_pool_initializes_and_clears() {
    #[cfg(not(miri))]
    let empty = ThreadPool::new();
    #[cfg(miri)]
    let empty = EmptyPool;
    let mut tt = TranspositionTable::new();
    tt.resize(1, &empty);
    assert!(!tt.probe(KEY).0);
    write_without_search(&tt);
    tt.clear(&empty);
    assert!(!tt.probe(KEY).0);
    assert_eq!(tt.hashfull_default(), 0);
}

#[test]
fn public_writes_invalidate_pristine_clear_and_resize() {
    let mut tt = TranspositionTable::new();
    tt.resize(1, &InlinePool);
    write_without_search(&tt);
    tt.clear(&InlinePool);
    assert!(!tt.probe(KEY).0);
    write_without_search(&tt);
    tt.resize(1, &InlinePool);
    assert!(!tt.probe(KEY).0);
}
