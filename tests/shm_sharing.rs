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

//! Cross-process tests for the shared-memory backend.
//!
//! The unit tests in `src/shm.rs` only prove a region round-trips within one
//! process, which the local-memory fallback also satisfies. These tests spawn a
//! real second process and check that it maps the *same physical pages* — the
//! only thing that distinguishes working shared memory from a private copy.
//!
//! Both halves live in this one test binary. The parent re-executes the test
//! binary with `--exact <child test>` and an environment variable, which is why
//! each test begins by checking whether it is running as parent or child.

use std::process::Command;

use stockcrab::shm::{AllocationStatus, SystemWideSharedConstant};

/// Set in the child process; carries the parent's pid.
const CHILD_ENV: &str = "STOCKCRAB_SHM_CHILD_PID";

const REGION_LEN: usize = 1 << 16;

/// Byte the *child* writes if it ends up creating its own region rather than
/// attaching to the parent's. Seeing this in the child means sharing failed.
const CHILD_OWN_FILL: u8 = 0xFF;

/// Content and region name are both derived from the parent's pid, so concurrent
/// or repeated test runs cannot collide with each other.
fn content_hash_for(pid: u32) -> u64 {
    0x5f5f_0000_0000_0000 ^ u64::from(pid)
}

/// Removes the per-name directory of a region minted by this test run when dropped.
///
/// The backend keeps `<temp root>/sfshm_<hash>/init_lock` on purpose (as C++ does: removing it
/// would race with a concurrent creator). The names here are unique to one run, so every run
/// would otherwise leave a directory behind. Declare the guard before the region so that it is
/// dropped after it, and only in the parent, once the child has exited.
struct RegionDirCleanup(Option<std::path::PathBuf>);

impl RegionDirCleanup {
    /// Mirrors how `SystemWideSharedConstant::new` and `SharedMemory::open` derive the directory;
    /// callers assert the directory exists while the region is shared, which keeps this honest.
    #[cfg(unix)]
    fn new(content_hash: u64, discriminator: u64) -> Self {
        use stockcrab::misc::hash_string;
        let name = format!(
            "Local\\sf_{}${}${}",
            content_hash,
            hash_string(&stockcrab::shm::executable_path()),
            discriminator
        );
        let shm_name = format!("/sf_{:016x}", hash_string(&name));
        Self(stockcrab::shm_unix::TempRoot::get().map(|root| {
            root.prefix
                .join(format!("sfshm_{:016}", hash_string(&shm_name)))
        }))
    }

    #[cfg(not(unix))]
    fn new(_content_hash: u64, _discriminator: u64) -> Self {
        Self(None)
    }

    fn assert_matches(&self, region: &SystemWideSharedConstant) {
        if region.status() == AllocationStatus::SharedMemory {
            let dir = self.0.as_ref().expect("shared region without a temp root");
            assert!(
                dir.is_dir(),
                "{} is not the region's directory",
                dir.display()
            );
        }
    }
}

impl Drop for RegionDirCleanup {
    fn drop(&mut self) {
        if let Some(dir) = &self.0 {
            let _ = std::fs::remove_file(dir.join("init_lock"));
            let _ = std::fs::remove_dir(dir);
        }
    }
}

fn expected_byte(pid: u32, i: usize) -> u8 {
    (u32::from(i as u8).wrapping_add(pid) & 0xFF) as u8
}

fn child_pid_env() -> Option<u32> {
    std::env::var(CHILD_ENV).ok()?.parse().ok()
}

#[test]
fn shared_region_is_mapped_by_a_second_process() {
    if child_pid_env().is_some() {
        return; // running as the child; the child body is the other test
    }

    let pid = std::process::id();
    let cleanup = RegionDirCleanup::new(content_hash_for(pid), 0);
    let region = SystemWideSharedConstant::new(content_hash_for(pid), REGION_LEN, 0, |buf| {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = expected_byte(pid, i);
        }
    });
    cleanup.assert_matches(&region);

    let status = region.status();
    if status != AllocationStatus::SharedMemory {
        // No shared-memory support in this environment (for example /tmp is not
        // usable). Skip loudly rather than silently passing on the fallback.
        eprintln!("SKIP: shared memory unavailable, status = {status:?}");
        return;
    }
    assert!(region.is_creator(), "parent should have created the region");

    // The parent must stay alive, holding the region and serving its fd, for the
    // whole lifetime of the child.
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args([
            "--exact",
            "child_attaches_to_parent_region",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, pid.to_string())
        .output()
        .expect("failed to spawn child test process");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "child process failed\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("CHILD_ATTACHED_OK"),
        "child did not report a successful attach\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );

    drop(region);
}

/// The child half of [`shared_region_is_mapped_by_a_second_process`].
///
/// Runs as a no-op unless the parent set the environment variable, so a plain
/// `cargo test` does not execute it standalone.
#[test]
fn child_attaches_to_parent_region() {
    let Some(parent_pid) = child_pid_env() else {
        return;
    };

    let region =
        SystemWideSharedConstant::new(content_hash_for(parent_pid), REGION_LEN, 0, |buf| {
            // Only reached if we failed to find the parent and had to create the
            // region ourselves -- a distinctive fill makes that unmistakable.
            buf.fill(CHILD_OWN_FILL);
        });

    assert_eq!(
        region.status(),
        AllocationStatus::SharedMemory,
        "child did not get a shared mapping"
    );
    assert!(
        !region.is_creator(),
        "child created its own region instead of attaching to the parent's"
    );

    let bytes = region.as_slice().expect("child must have a mapping");
    assert_eq!(bytes.len(), REGION_LEN);
    assert_ne!(
        bytes[0], CHILD_OWN_FILL,
        "child is looking at its own region, not the parent's"
    );

    for (i, &b) in bytes.iter().enumerate() {
        assert_eq!(
            b,
            expected_byte(parent_pid, i),
            "byte {i} does not match what the parent wrote"
        );
    }

    println!("CHILD_ATTACHED_OK");
}

/// Two regions with different content hashes must not alias, even though they
/// are created back to back by the same process.
#[test]
fn distinct_content_does_not_alias() {
    if child_pid_env().is_some() {
        return;
    }

    let pid = std::process::id();
    let cleanup_a = RegionDirCleanup::new(content_hash_for(pid) ^ 0xAAAA, 0);
    let cleanup_b = RegionDirCleanup::new(content_hash_for(pid) ^ 0xBBBB, 0);
    let a =
        SystemWideSharedConstant::new(content_hash_for(pid) ^ 0xAAAA, 4096, 0, |b| b.fill(0x11));
    let b =
        SystemWideSharedConstant::new(content_hash_for(pid) ^ 0xBBBB, 4096, 0, |b| b.fill(0x22));
    cleanup_a.assert_matches(&a);
    cleanup_b.assert_matches(&b);

    assert_eq!(a.as_slice().unwrap()[0], 0x11);
    assert_eq!(b.as_slice().unwrap()[0], 0x22);
    assert_ne!(
        a.as_slice().unwrap().as_ptr(),
        b.as_slice().unwrap().as_ptr(),
        "different content must map to different regions"
    );
}
