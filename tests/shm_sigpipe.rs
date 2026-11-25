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

//! The shared-memory fd server must survive a client that vanishes mid-handshake.
//!
//! Once a peer is accepted, the server sends it the region's descriptor with
//! `sendmsg`. If that peer has already gone -- crashed, or simply hit its own
//! one-second join timeout -- the send fails with `EPIPE` and, by default, first
//! raises `SIGPIPE`. Killing the *server* is the worst possible response: it owns
//! the region every other process is attaching to, so one impatient client takes
//! the whole network down. Upstream fixed this in `762dd1da`.
//!
//! What protects this port, by platform:
//!
//! * `MSG_NOSIGNAL` (Linux, Android, and current macOS/iOS SDKs) is a per-send
//!   flag and always applies. Upstream uses it wherever the macro is defined.
//! * `SO_NOSIGPIPE` (macOS, BSD) is a socket option, and on a connected socket
//!   `setsockopt` returns EINVAL once the peer has closed, which makes
//!   upstream's placement just before the send ineffective for the
//!   early-disconnect case. The port sets it on the listening socket instead:
//!   on Darwin an accepted socket inherits it, even when the client is gone
//!   before `accept` returns. It is set again at accept time as a second layer.
//! * On FreeBSD, NetBSD, OpenBSD and DragonFly neither of those has been
//!   verified for a client that is gone before `accept` returns. There the
//!   protection is the process-wide `SIG_IGN` the Rust runtime installs for
//!   SIGPIPE at startup, which turns the signal into a plain `EPIPE` return,
//!   and the engine leaves that disposition alone.
//!
//! So this test runs under the engine's real signal disposition -- SIG_DFL
//! where `main.rs` restores it, which a test binary has to do for itself -- and
//! asserts the end-to-end property: hammer the server with clients that connect
//! and vanish, and it must still be alive and serving an intact region
//! afterwards.
//!
//! (The server deliberately does not block SIGPIPE on its thread: a blocked
//! signal is deferred rather than discarded, and it kills the process when the
//! thread exits. See `src/shm_unix.rs`.)
//!
//! Everything runs in a child process. The release profile sets
//! `panic = "abort"` (the `[profile.release]` section of Cargo.toml), so a
//! regression here would otherwise take the whole test binary down with a bare
//! signal and no indication of the cause.

#![cfg(unix)]

use std::path::PathBuf;

use stockcrab::shm::{AllocationStatus, SystemWideSharedConstant};

const CHILD_ENV: &str = "STOCKCRAB_SHM_SIGPIPE_CHILD";
const REGION_LEN: usize = 64 * 1024;

/// Every `<pid>-<instance>.sock` under this process's shared-memory directories.
///
/// The layout is `/tmp/stockfish-<uid>/sfshm_<hash>/<pid>-<instance>.sock` (see the module
/// docs in `src/shm_unix.rs`). Matching on our own pid keeps a concurrently
/// running test's server out of the way.
fn own_sockets() -> Vec<PathBuf> {
    // Hardcoded `/tmp`, not `std::env::temp_dir()`: `shm_unix.rs` builds this
    // path literally, and on macOS `temp_dir()` returns `$TMPDIR`
    // (`/var/folders/...`), which would silently find nothing and skip.
    // SAFETY: getuid has no preconditions and cannot fail.
    let root = PathBuf::from(format!("/tmp/stockfish-{}", unsafe { libc::getuid() }));
    let prefix = format!("{}-", std::process::id());
    let mut out = Vec::new();
    let Ok(dirs) = std::fs::read_dir(&root) else {
        return out;
    };
    for d in dirs.flatten() {
        if !d.path().is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(d.path()) else {
            continue;
        };
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".sock") {
                out.push(e.path());
            }
        }
    }
    out
}

/// Connect to `path`, then close at once without reading the descriptor.
/// Returns whether the `connect` succeeded.
fn connect_and_vanish(path: &std::path::Path) -> bool {
    let bytes = path.as_os_str().as_encoded_bytes();
    // SAFETY: sockaddr_un is a plain C struct (all-zero is valid) and the path length is
    // checked against sun_path before copying; `addr` outlives connect, and `fd` is a socket
    // created and closed exactly once here.
    unsafe {
        let mut addr: libc::sockaddr_un = core::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as _;
        if bytes.len() >= core::mem::size_of_val(&addr.sun_path) {
            panic!("socket path too long for sun_path: {}", path.display());
        }
        for (dst, &src) in addr.sun_path.iter_mut().zip(bytes) {
            *dst = src as _;
        }

        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0);
        if fd < 0 {
            return false;
        }
        let len = core::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
        let connected = libc::connect(fd, (&addr as *const libc::sockaddr_un).cast(), len) == 0;
        // No read: go away while the server is still trying to hand over the fd.
        libc::close(fd);
        connected
    }
}

#[test]
fn fd_server_survives_a_client_that_disconnects_mid_handshake() {
    if std::env::var(CHILD_ENV).is_ok() {
        child();
        return;
    }

    let out = std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            "--exact",
            "fd_server_survives_a_client_that_disconnects_mid_handshake",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn child test process");

    let stdout = String::from_utf8_lossy(&out.stdout);
    if stdout.contains("CHILD-SKIP") {
        eprintln!("SKIP: shared memory unavailable in this environment");
        return;
    }
    assert!(
        stdout.contains("CHILD-PROBING"),
        "the child never reached the disconnect storm, so this proves nothing\n\
         exit: {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success() && stdout.contains("CHILD-OK"),
        "the fd server did not survive clients disconnecting mid-handshake \
         (exit by signal 13 means SIGPIPE killed it)\n\
         exit: {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn child() {
    use std::io::Write;

    // The engine binary restores SIG_DFL for SIGPIPE on these platforms
    // (`restore_default_sigpipe` in `main.rs`). This is a test binary, which the
    // Rust runtime starts with SIG_IGN, so match the engine explicitly: only then
    // does surviving the storm prove the socket-level guard rather than the
    // ignored signal.
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    ))]
    // SAFETY: installing SIG_DFL for SIGPIPE with `signal` has no preconditions and only
    // changes the process-wide disposition; the child runs this one test and nothing else.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    // Elsewhere the engine keeps the runtime's SIG_IGN and the fd server relies
    // on it. Pin that assumption rather than leaving it inherited and unstated:
    // if a future toolchain stops installing SIG_IGN by default, this is where
    // it becomes visible.
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        // SAFETY: libc::sigaction is a plain C struct (all-zero is valid); a null new-action
        // pointer makes the call a pure query that writes only to the live local `old`.
        let sigpipe_ignored = unsafe {
            let mut old: libc::sigaction = core::mem::zeroed();
            libc::sigaction(libc::SIGPIPE, core::ptr::null(), &mut old);
            old.sa_sigaction == libc::SIG_IGN
        };
        assert!(
            sigpipe_ignored,
            "expected the Rust runtime to install SIG_IGN for SIGPIPE; without it the \
             fd server is only protected where SO_NOSIGPIPE/MSG_NOSIGNAL can be applied"
        );
    }

    let region = SystemWideSharedConstant::new(0x5163_9A1E_0000_0001, REGION_LEN, 0, |buf| {
        buf.fill(0xA5);
    });

    if region.status() != AllocationStatus::SharedMemory {
        println!("CHILD-SKIP {:?}", region.status());
        return;
    }

    let sockets = own_sockets();
    if sockets.is_empty() {
        println!("CHILD-SKIP no server socket found");
        return;
    }

    println!("CHILD-PROBING {} socket(s)", sockets.len());
    std::io::stdout().flush().unwrap();

    // Repeat: whether a given attempt loses the race and reaches `sendmsg`
    // after the peer is gone is timing-dependent, so one connection proves
    // little. Fifty reliably hit it.
    let mut connected = 0usize;
    for _ in 0..50 {
        for s in &sockets {
            if connect_and_vanish(s) {
                connected += 1;
            }
        }
    }
    // Not every connect has to succeed (the backlog may legitimately fill), but
    // if none did the storm never reached the server.
    assert!(
        connected > 0,
        "no connect() succeeded, so the test proved nothing"
    );

    // Give the server thread time to finish reacting to the last of them.
    std::thread::sleep(core::time::Duration::from_millis(300));

    // Still alive, and the region is still intact and serving.
    let slice = region.as_slice().expect("region still mapped");
    assert_eq!(slice.len(), REGION_LEN);
    assert!(
        slice.iter().all(|&b| b == 0xA5),
        "region contents were damaged"
    );

    println!("CHILD-OK");
}
