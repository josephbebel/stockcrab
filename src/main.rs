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

use std::env;
use std::io::{self, Write};

use stockcrab::misc::{build_performance_warning, engine_info};
use stockcrab::{bitboard, movegen, tune, uci};

/// Restore the default SIGPIPE disposition, which the Rust runtime replaces with
/// SIG_IGN before `main`. With SIG_IGN a GUI that closes our stdout turns the next
/// `println!` into a "failed printing to stdout: Broken pipe" panic, which under
/// `panic = "abort"` is SIGABRT and a core dump. C++ Stockfish just dies of
/// SIGPIPE, silently, and so do we now.
///
/// Leaving SIG_IGN in place is not merely noisy: the regular UCI output path ignores write
/// errors, so an engine whose stdout is gone but whose stdin is still open would search on
/// forever instead of terminating.
///
/// Linux, Android, macOS and iOS only. The shared-memory fd server (`shm_unix.rs`) must not be
/// killed by a client that vanished before `accept` returned, and on these platforms it is
/// protected at socket level: `MSG_NOSIGNAL` on the `sendmsg`, and on Darwin also
/// `SO_NOSIGPIPE` on the listening socket, which accepted sockets inherit
/// (`tests/shm_sigpipe.rs`). Neither has been verified on FreeBSD, NetBSD, OpenBSD or
/// DragonFly, so there the disposition is left alone and the fd server keeps relying on the
/// ignored signal.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
))]
fn restore_default_sigpipe() {
    // SAFETY: called first thing in `main`, before any thread exists; installing SIG_DFL for
    // SIGPIPE with `signal` has no preconditions and only changes the process-wide disposition.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn restore_default_sigpipe() {}

fn main() {
    restore_default_sigpipe();

    let args: Vec<String> = env::args().collect();

    // Print engine info. Not `println!`: where SIGPIPE stays ignored (see
    // `restore_default_sigpipe`) a closed stdout must end the process quietly, not as a panic.
    if writeln!(io::stdout(), "{}", engine_info(false)).is_err() {
        std::process::exit(1);
    }

    // GUIs ignore stderr, so this never disturbs the UCI stream.
    if let Some(warning) = build_performance_warning() {
        eprintln!("{warning}");
    }

    // Initialize bitboards, movegen tables, and position tables
    bitboard::init();
    movegen::init();

    // Create UCI engine and run loop
    let mut uci = uci::UciEngine::new(&args);

    // Initialize tuning parameters
    tune::Tune::init(&uci.engine_options());

    // Run UCI loop
    uci.loop_();
}
