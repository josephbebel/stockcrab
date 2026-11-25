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

//! System-wide shared constants.
//!
//! Port of C++ `src/shm.h`: the platform-independent wrapper that picks a
//! shared-memory backend when one is available and falls back to a private
//! allocation otherwise. The Unix backend lives in [`crate::shm_unix`].
//!
//! The value shared here is the NNUE feature-transformer weights (the large part of
//! the network; the layer stacks stay private). Deduplicating it
//! matters when several engine processes run at once — a tournament, an SPRT
//! run, or one process per NUMA node — where the alternative is one private
//! copy each.
//!
//! Naming: a region is addressed by the hash of its *contents* combined with the
//! hash of the executable path, plus an optional discriminator (C++ uses the
//! NUMA node). Content addressing is what makes this safe to do implicitly:
//! two processes can only ever share a region if they would have built
//! byte-identical data anyway, so there is no way to attach to the wrong
//! weights. Including the executable path additionally prevents two different
//! builds from sharing, since their in-memory layouts may differ even when the
//! net file does not.

use crate::misc::{LargePageAllocator, hash_string};

#[cfg(unix)]
use crate::shm_unix::{NAME_MAX, SharedMemory};

/// How a [`SystemWideSharedConstant`] ended up being stored.
///
/// Mirrors C++ `SystemWideSharedConstantAllocationStatus`; the engine reports it
/// verbatim from `verify_network`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationStatus {
    /// Nothing is allocated (construction failed).
    NoAllocation,
    /// A private allocation: correct, but not shared with other processes.
    LocalMemory,
    /// Backed by a region shared with every other process using the same data.
    SharedMemory,
}

/// Path of the running executable, or an empty string if it cannot be determined.
///
/// Port of C++ `getExecutablePathHash` (which, despite the name, returns the
/// path; the caller hashes it). `argv[0]` is deliberately not used: it holds
/// whatever command invoked us, which need not be a full path.
pub fn executable_path() -> String {
    std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// A large constant deduplicated across every process of the same user.
///
/// The stored bytes are treated as immutable once published: other processes map
/// the same pages, so mutating them after construction would corrupt their view.
pub struct SystemWideSharedConstant {
    backend: Backend,
}

enum Backend {
    /// Nothing allocated.
    None,
    /// Shared with other processes.
    #[cfg(unix)]
    Shared(SharedMemory),
    /// Private fallback, used when the OS has no support or the protocol failed.
    Local(LargePageAllocator, usize),
}

impl SystemWideSharedConstant {
    /// Publish (or attach to) `len` bytes of content initialised by `init`.
    ///
    /// `content_hash` must be derived from exactly the bytes `init` writes:
    /// it is what decides whether two processes share a region. `discriminator`
    /// separates otherwise-identical content that must not be shared (C++ uses
    /// it for the NUMA node).
    ///
    /// `init` is invoked only when this process creates the region. If another
    /// process already published matching content, its pages are mapped and
    /// `init` is never called.
    pub fn new<F>(content_hash: u64, len: usize, discriminator: u64, init: F) -> Self
    where
        F: FnOnce(&mut [u8]),
    {
        let executable_hash = hash_string(&executable_path());
        // Same shape as the C++ name before it is re-hashed for POSIX.
        let name = format!(
            "Local\\sf_{}${}${}",
            content_hash, executable_hash, discriminator
        );
        #[cfg(not(unix))]
        let _ = &name;

        #[cfg(unix)]
        {
            // POSIX shared memory names must start with a slash, hash ensures it is short enough everywhere.
            let shm_name = format!("/sf_{:016x}", hash_string(&name));
            debug_assert!(shm_name.len() <= NAME_MAX);

            // `init` is FnOnce, so it can only be handed to one of the two
            // paths. Try shared first and only fall back if it declines.
            let mut init_slot = Some(init);
            let taken = &mut init_slot;
            if let Some(shm) = SharedMemory::open(&shm_name, len, |buf| {
                (taken.take().expect("init called at most once"))(buf)
            }) {
                return Self {
                    backend: Backend::Shared(shm),
                };
            }

            if let Some(init) = init_slot {
                return Self::local(len, init);
            }

            // Unreachable today: open() only consumes `init` after committing to
            // return Some. Kept as a guard so a future change cannot hand back
            // uninitialised memory.
            Self {
                backend: Backend::None,
            }
        }

        #[cfg(not(unix))]
        {
            Self::local(len, init)
        }
    }

    fn local<F>(len: usize, init: F) -> Self
    where
        F: FnOnce(&mut [u8]),
    {
        match LargePageAllocator::allocate_zeroed(len) {
            Some(alloc) => {
                // SAFETY: `alloc` is a live allocation of at least `len` bytes (LargePageAllocator
                // rounds sizes up, never down), zero-initialised by allocate_zeroed so every byte
                // is a valid u8, and uniquely owned here until it is moved into the backend below.
                let slice = unsafe { core::slice::from_raw_parts_mut(alloc.as_ptr(), len) };
                init(slice);
                Self {
                    backend: Backend::Local(alloc, len),
                }
            }
            None => Self {
                backend: Backend::None,
            },
        }
    }

    /// The published bytes, or `None` if nothing was allocated.
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.backend {
            Backend::None => None,
            #[cfg(unix)]
            Backend::Shared(shm) => Some(shm.as_slice()),
            Backend::Local(alloc, len) => {
                // SAFETY: `alloc` is kept alive by &self for the whole borrow and holds at least
                // `*len` initialised bytes (zeroed, then filled by `init` in local()); nothing
                // mutates it after construction, so a shared slice is sound.
                Some(unsafe { core::slice::from_raw_parts(alloc.as_ptr() as *const u8, *len) })
            }
        }
    }

    pub fn status(&self) -> AllocationStatus {
        match &self.backend {
            Backend::None => AllocationStatus::NoAllocation,
            #[cfg(unix)]
            Backend::Shared(_) => AllocationStatus::SharedMemory,
            Backend::Local(..) => AllocationStatus::LocalMemory,
        }
    }

    /// Human-readable note about the backing, matching the C++ messages.
    pub fn error_message(&self) -> Option<String> {
        match &self.backend {
            Backend::None => Some("Not initialized".to_string()),
            #[cfg(unix)]
            Backend::Shared(_) => None,
            Backend::Local(..) => {
                Some("Shared memory not available. Local allocation fallback.".to_string())
            }
        }
    }

    /// True when this process created the shared region rather than joining an
    /// existing one. Diagnostics and tests only. Non-shared backends (Local/None)
    /// have no peers, so they report true.
    pub fn is_creator(&self) -> bool {
        match &self.backend {
            #[cfg(unix)]
            Backend::Shared(shm) => shm.is_creator(),
            _ => true,
        }
    }

    /// Guard removing this region's directory once dropped, for tests whose region name is
    /// unique to one run. `None` when the region is not shared.
    #[cfg(all(test, unix))]
    pub(crate) fn dir_cleanup(&self) -> Option<crate::shm_unix::RegionDirCleanup> {
        match &self.backend {
            Backend::Shared(shm) => Some(crate::shm_unix::RegionDirCleanup::for_region(shm)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_path_is_non_empty_under_test() {
        assert!(!executable_path().is_empty());
    }

    #[test]
    fn round_trips_content() {
        let len = 4096;
        let c = SystemWideSharedConstant::new(0xdead_beef_1234_5678, len, 0, |buf| {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        });

        let got = c.as_slice().expect("must allocate somehow");
        assert_eq!(got.len(), len);
        // Whether we created it or attached to a peer from an earlier test run,
        // the content is defined by the hash, so it must match either way.
        for (i, &b) in got.iter().enumerate() {
            assert_eq!(b, (i % 251) as u8, "byte {i} differs");
        }
        assert_ne!(c.status(), AllocationStatus::NoAllocation);
    }

    #[test]
    fn distinct_hashes_do_not_collide() {
        let a = SystemWideSharedConstant::new(1, 64, 0, |b| b.fill(0xAA));
        let b = SystemWideSharedConstant::new(2, 64, 0, |b| b.fill(0xBB));
        assert_eq!(a.as_slice().unwrap()[0], 0xAA);
        assert_eq!(b.as_slice().unwrap()[0], 0xBB);
    }

    #[test]
    fn discriminator_separates_identical_content() {
        let a = SystemWideSharedConstant::new(7, 64, 0, |b| b.fill(1));
        let b = SystemWideSharedConstant::new(7, 64, 1, |b| b.fill(2));
        assert_eq!(a.as_slice().unwrap()[0], 1);
        assert_eq!(b.as_slice().unwrap()[0], 2);
    }
}
