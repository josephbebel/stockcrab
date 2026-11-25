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

//! Unix shared-memory backend.
//!
//! Port of C++ `src/shm_unix.h`. The goal is to have a single physical copy of
//! a large constant (in practice the NNUE weights, ~110 MiB) shared by every
//! Stockfish process belonging to the same user and built from the same
//! executable, instead of one private copy each.
//!
//! There is no name-based POSIX object involved. The region is an anonymous
//! file (`memfd_create` on Linux, Android and FreeBSD, an unlinked `mkstemp` file elsewhere)
//! and processes hand each other a *file descriptor* for it over a Unix domain
//! socket using `SCM_RIGHTS`. Descriptor passing is what makes this work without
//! a persistent name in the filesystem or in `/dev/shm`: the fd the receiver
//! gets may have a different number but refers to the same open file, so
//! `mmap(MAP_SHARED)` on it maps the same physical pages.
//!
//! Trust model: the processes sharing a region trust each other. The seals fix the region's size,
//! but any process holding the descriptor can write to it, and a write while another process
//! evaluates would change weights under its feet (as in C++). The descriptor is only handed to
//! peers of the same user, through a 0700 directory.
//!
//! The protocol, per region:
//!
//! 1. `TempRoot` — `/tmp/stockfish-<uid>`, created 0700 and re-validated on
//!    every startup (owner and mode must match) so another user cannot pre-create
//!    it and have us hand descriptors into their directory.
//! 2. A per-content directory `<temp_root>/sfshm_<hash>` holds one `.sock` per
//!    participating process plus an `init_lock` file.
//! 3. `flock(LOCK_EX)` on `init_lock` serialises startup, so exactly one process
//!    creates the region while the others wait and then find its socket.
//! 4. The joiner scans for `*.sock` peers and asks each for the fd. Sockets whose
//!    owner has died refuse the connection and are unlinked.
//! 5. If nobody answers, this process creates the region, sizes it, maps it and
//!    initialises the contents.
//! 6. Either way it then binds its own `.sock` and serves the fd to later
//!    arrivals from a background thread, which also watches a shutdown pipe.
//!
//! Cleanup: on a normal drop the socket is unlinked and the server thread joined.
//! On process exit an `atexit` hook unlinks the sockets but deliberately does
//! *not* unmap, because search threads may still be running and reading the
//! weights. Like C++, an abnormal exit (SIGKILL, Ctrl-C) can leave a stale
//! `.sock` behind; that is harmless, since a stale socket refuses connections
//! and gets unlinked by the next process that tries it.
//!
//! The system calls go through `std` where it has them (files, `flock`, directories, the
//! shutdown pipe, `accept`) and through `nix` otherwise (sockets, descriptor passing, `poll`,
//! `memfd_create`, seals). Only the mapping itself, `atexit` and `SO_NOSIGPIPE` call `libc`.

use core::sync::atomic::{AtomicU64, Ordering};
use std::ffi::OsStr;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, IoSlice, IoSliceMut, PipeReader, PipeWriter};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

use nix::errno::Errno;
#[cfg(not(target_os = "linux"))]
use nix::fcntl::FdFlag;
#[cfg(any(target_os = "linux", target_os = "android"))]
use nix::fcntl::SealFlag;
use nix::fcntl::{FcntlArg, fcntl};
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::sys::socket::{
    AddressFamily, Backlog, ControlMessage, ControlMessageOwned, MsgFlags, SockFlag, SockType,
    UnixAddr, bind, connect, listen, recvmsg, sendmsg, setsockopt, socket, sockopt,
};
use nix::sys::time::TimeVal;
use nix::unistd::getuid;

use crate::misc::hash_string;
#[cfg(target_os = "linux")]
use crate::misc::mmap_huge_aligned;

/// Longest name we will produce for a shared-memory directory entry.
///
/// `NAME_MAX`, which is 255 on both Linux and macOS. The names are generated hash strings,
/// far shorter, so this is a safety net rather than a real constraint.
pub const NAME_MAX: usize = 255;

// ---------------------------------------------------------------------------
// Small fd helpers
// ---------------------------------------------------------------------------

/// Set `FD_CLOEXEC` on `fd`, ignoring errors as C++ `set_cloexec` does.
///
/// Needed on platforms (macOS) lacking the `SOCK_CLOEXEC` / `MSG_CMSG_CLOEXEC`
/// atomic variants. Without it a descriptor could leak into a child process.
#[cfg(not(target_os = "linux"))]
fn set_cloexec(fd: BorrowedFd<'_>) {
    if let Ok(flags) = fcntl(fd, FcntlArg::F_GETFD) {
        let _ = fcntl(
            fd,
            FcntlArg::F_SETFD(FdFlag::from_bits_retain(flags) | FdFlag::FD_CLOEXEC),
        );
    }
}

/// Ask the kernel to report `EPIPE` from sends on `fd` instead of raising
/// SIGPIPE, on the platforms that offer the socket option. Linux has no such
/// option and uses the `MSG_NOSIGNAL` send flag in `send_fd` instead.
///
/// Best-effort, and applied in two places, neither of which is where upstream
/// `762dd1da` puts it (immediately before the send). On macOS
/// `setsockopt(SO_NOSIGPIPE)` returns EINVAL on a connected socket whose peer
/// has already closed, so by send time it can no longer be applied to the very
/// client it guards against. Measured on macOS 15 and 26 -- healthy socket:
/// succeeds, and the signal is suppressed; peer already closed: EINVAL, and
/// `sendmsg` raises SIGPIPE.
///
/// * `create_unix_socket` sets it on every socket at creation, which includes
///   the listening socket. On Darwin an accepted socket inherits the option
///   from its listener, even when the peer closed before `accept` returned
///   (measured on macOS 26). This is the layer that covers that case.
/// * The server sets it again on each accepted socket, which is what covers a
///   client still connected at accept time on a platform where inheritance has
///   not been verified (the other BSDs).
///
/// On macOS and iOS `send_fd` also passes `MSG_NOSIGNAL`, as upstream does
/// through its `#ifdef`. The engine binary restores SIG_DFL for SIGPIPE on
/// those two (`main.rs`), so these are what keep the server alive there;
/// `tests/shm_sigpipe.rs` pins that end to end. On the other BSDs a peer that
/// is gone before `accept` returns is still covered only by the process-wide
/// `SIG_IGN` that the Rust runtime installs at startup, which is why `main.rs`
/// leaves the disposition alone there.
///
/// Blocking SIGPIPE on the server thread instead was tried and is worse than
/// doing nothing: the signal is not discarded, only deferred, and it kills the
/// process when the thread exits.
///
/// `nix` has no `SO_NOSIGPIPE` option, hence the direct `setsockopt`.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
fn set_nosigpipe(fd: BorrowedFd<'_>) {
    let yes: libc::c_int = 1;
    // SAFETY: `yes` is a live c_int local and the length passed is its size_of; `fd` is open for
    // the call (borrowed). A failure is ignored (best effort).
    unsafe {
        libc::setsockopt(
            fd.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_NOSIGPIPE,
            (&yes as *const libc::c_int).cast(),
            core::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
)))]
fn set_nosigpipe(_fd: BorrowedFd<'_>) {}

/// `socket(AF_UNIX, SOCK_STREAM)`, close-on-exec, and with SIGPIPE suppressed
/// where that is a socket option (see `set_nosigpipe`).
///
/// Not `std::os::unix::net`: the client needs its timeouts set before `connect`
/// (see `try_receive_fd`), and the listener keeps C++'s backlog of 5.
fn create_unix_socket() -> Option<OwnedFd> {
    #[cfg(target_os = "linux")]
    let flags = SockFlag::SOCK_CLOEXEC;
    #[cfg(not(target_os = "linux"))]
    let flags = SockFlag::empty();

    let fd = socket(AddressFamily::Unix, SockType::Stream, flags, None).ok()?;
    #[cfg(not(target_os = "linux"))]
    {
        set_cloexec(fd.as_fd());
        // On the listening socket this is what protects the server from a
        // client that is gone before `accept` returns: accepted sockets
        // inherit it. A no-op where there is no such option.
        set_nosigpipe(fd.as_fd());
    }
    Some(fd)
}

/// The socket address for `path`.
///
/// Returns `None` if the path does not fit: `sun_path` is only 104 bytes on
/// macOS and 108 on Linux, and silently truncating would make us bind or connect
/// to the wrong socket. `UnixAddr::new` refuses such paths.
fn socket_addr(path: &Path) -> Option<UnixAddr> {
    UnixAddr::new(path).ok()
}

// ---------------------------------------------------------------------------
// Temp root
// ---------------------------------------------------------------------------

/// `/tmp/stockfish-<uid>`, created with mode 0700.
///
/// Port of C++ `TempRoot`. If the directory already exists it is only accepted
/// when it is a real directory owned by us with exactly mode 0700 — otherwise
/// another user could have pre-created it and would see our sockets.
pub struct TempRoot {
    pub prefix: PathBuf,
}

impl TempRoot {
    pub fn get() -> Option<&'static TempRoot> {
        static ROOT: OnceLock<Option<TempRoot>> = OnceLock::new();
        ROOT.get_or_init(TempRoot::probe).as_ref()
    }

    fn probe() -> Option<TempRoot> {
        let uid = getuid();
        let proposed = PathBuf::from(format!("/tmp/stockfish-{uid}"));

        match DirBuilder::new().mode(0o700).create(&proposed) {
            Ok(()) => Some(TempRoot { prefix: proposed }),
            // Already there: only use it if it is ours and not group/world
            // accessible. lstat, not stat, so a symlink cannot redirect us.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                let st = fs::symlink_metadata(&proposed).ok()?;
                (st.is_dir() && st.uid() == uid.as_raw() && st.mode() & 0o7777 == 0o700)
                    .then_some(TempRoot { prefix: proposed })
            }
            Err(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Init lock
// ---------------------------------------------------------------------------

/// `flock(LOCK_EX)` held on a lock file for as long as this value lives.
///
/// Serialises region creation: whoever wins creates and publishes, everyone else
/// blocks here and then discovers the winner's socket. The kernel releases the
/// lock if the holder dies, so a crash mid-initialisation cannot wedge the rest.
/// `File::lock` is `flock` on every Unix this backend runs on.
struct InitLock {
    file: File,
}

impl InitLock {
    fn acquire(path: &Path) -> Option<InitLock> {
        // std opens with O_CLOEXEC.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .ok()?;

        loop {
            match file.lock() {
                Ok(()) => return Some(InitLock { file }),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => return None,
            }
        }
    }
}

impl Drop for InitLock {
    fn drop(&mut self) {
        // Closing the file releases the lock as well; unlock first, as C++ does.
        let _ = self.file.unlock();
    }
}

// ---------------------------------------------------------------------------
// Exit-time cleanup registry
// ---------------------------------------------------------------------------

/// Socket paths to unlink when the process exits.
///
/// Mirrors C++ `SharedMemoryRegistry`. Only the sockets are removed: the mapping
/// is deliberately left in place because search threads may still be reading the
/// weights while `atexit` handlers run, and unmapping under them would segfault.
static ATEXIT_SOCKETS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
static ATEXIT_REGISTERED: OnceLock<()> = OnceLock::new();

extern "C" fn cleanup_at_exit() {
    // A poisoned mutex here would mean a panic while holding it; the paths are
    // still valid to unlink, so recover rather than abort during exit.
    let paths = match ATEXIT_SOCKETS.lock() {
        Ok(g) => g.clone(),
        Err(e) => e.into_inner().clone(),
    };
    for p in paths {
        let _ = fs::remove_file(p);
    }
}

fn register_socket_for_cleanup(path: &Path) {
    ATEXIT_REGISTERED.get_or_init(|| {
        // SAFETY: `cleanup_at_exit` is an `extern "C" fn()` of the signature atexit expects and
        // lives for the whole program; the OnceLock registers it at most once.
        unsafe { libc::atexit(cleanup_at_exit) };
    });
    if let Ok(mut g) = ATEXIT_SOCKETS.lock() {
        g.push(path.to_path_buf());
    }
}

fn unregister_socket(path: &Path) {
    if let Ok(mut g) = ATEXIT_SOCKETS.lock() {
        g.retain(|p| p != path);
    }
}

// ---------------------------------------------------------------------------
// fd passing
// ---------------------------------------------------------------------------

/// The most descriptors one `SCM_RIGHTS` message can carry on Linux (`SCM_MAX_FD`,
/// unix(7)).
///
/// The receiver has room for this many, so on Linux no reply can be truncated.
const SCM_MAX_FD: usize = 253;

/// Ask the peer listening on `their_path` to send us the region's descriptor.
///
/// Returns `None` if the peer is gone (in which case its stale socket is
/// unlinked), or if it answered without attaching exactly one descriptor.
fn try_receive_fd(their_path: &Path) -> Option<File> {
    let addr = socket_addr(their_path)?;
    let peer = create_unix_socket()?;

    // Bound how long a wedged peer can stall our startup. Set before `connect`:
    // on Linux a connect to a listener whose backlog is full waits for up to the
    // send timeout. Failure is ignored (best effort).
    let timeout = TimeVal::new(1, 0);
    let _ = setsockopt(&peer, sockopt::SendTimeout, &timeout);
    let _ = setsockopt(&peer, sockopt::ReceiveTimeout, &timeout);

    loop {
        match connect(peer.as_raw_fd(), &addr) {
            Ok(()) => break,
            Err(Errno::EINTR) => continue,
            // Owner is dead, or the entry is gone: drop the stale socket so the
            // directory does not accumulate them.
            Err(Errno::ECONNREFUSED | Errno::ENOENT) => {
                let _ = fs::remove_file(their_path);
                return None;
            }
            Err(_) => return None,
        }
    }

    let mut byte = [0u8; 1];
    let mut iov = [IoSliceMut::new(&mut byte)];
    // C++ offers only CMSG_SPACE(sizeof(int)). Offering room for SCM_MAX_FD descriptors means
    // that a Linux reply is never truncated, so every descriptor of a malformed one (several
    // descriptors) is owned below and closed again. `cmsg_space!` allocates the buffer on the
    // heap, which aligns it for `cmsghdr`.
    let mut control = nix::cmsg_space!([RawFd; SCM_MAX_FD]);

    #[cfg(target_os = "linux")]
    let flags = MsgFlags::MSG_CMSG_CLOEXEC;
    #[cfg(not(target_os = "linux"))]
    let flags = MsgFlags::empty();

    let msg = loop {
        match recvmsg::<()>(peer.as_raw_fd(), &mut iov, Some(&mut control), flags) {
            Err(Errno::EINTR) => continue,
            result => break result.ok()?,
        }
    };

    // Own every descriptor the kernel reported, so that a malformed message carrying several of
    // them leaks none: they are closed when `received` is dropped. A truncated reply
    // (`MSG_CTRUNC`) is refused without being decoded (`cmsgs` fails): XNU leaves the sender's
    // untruncated length in the last header, so walking it would read past what the kernel wrote.
    // Its descriptors stay open. On Linux that cannot happen (see `SCM_MAX_FD`); on XNU it takes
    // a reply of more than SCM_MAX_FD descriptors.
    let mut received = Vec::new();
    for cmsg in msg.cmsgs().ok()? {
        if let ControlMessageOwned::ScmRights(fds) = cmsg {
            for fd in fds.into_iter().filter(|&fd| fd >= 0) {
                // SAFETY: the kernel installed `fd` in our descriptor table for this message, so
                // nothing else owns it; the OwnedFd closes it exactly once.
                let fd = unsafe { OwnedFd::from_raw_fd(fd) };
                #[cfg(not(target_os = "linux"))]
                set_cloexec(fd.as_fd());
                received.push(fd);
            }
        }
    }

    let complete = msg.bytes > 0
        && !msg
            .flags
            .intersects(MsgFlags::MSG_TRUNC | MsgFlags::MSG_CTRUNC);
    if complete && received.len() == 1 {
        received.pop().map(File::from)
    } else {
        None
    }
}

/// Validate the object before mmap: mapping beyond EOF can succeed and then
/// SIGBUS on access. Region identity includes its exact size, not just its name.
fn region_file_matches(file: &File, len: usize) -> bool {
    let Ok(st) = file.metadata() else {
        return false;
    };
    if !st.is_file() || usize::try_from(st.len()).ok() != Some(len) {
        return false;
    }
    // Freeze the size across the interval between fstat and mapped access.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let fixed_size = SealFlag::F_SEAL_SHRINK | SealFlag::F_SEAL_GROW;
        match fcntl(file, FcntlArg::F_GET_SEALS) {
            Ok(seals) if SealFlag::from_bits_retain(seals).contains(fixed_size) => {}
            _ => return false,
        }
    }
    true
}

/// Send `fd` over `sock` as an `SCM_RIGHTS` control message.
///
/// The receiver gets a descriptor that may have a different number but refers to
/// the same open file, so mapping it `MAP_SHARED` shares physical pages with us.
fn send_fd(sock: BorrowedFd<'_>, fd: BorrowedFd<'_>) {
    let fds = [fd.as_raw_fd()];
    let cmsgs = [ControlMessage::ScmRights(&fds)];
    let iov = [IoSlice::new(&[0])];

    // Second half of the SIGPIPE guard; the first half is `set_nosigpipe`.
    // `MSG_NOSIGNAL` is a per-send flag, so unlike the socket option it
    // cannot be too late to apply. Upstream passes it under
    // `#ifdef MSG_NOSIGNAL`, which current macOS SDKs satisfy as well as
    // Linux. Rust has no such test, so the condition is the targets where
    // the flag has been checked to exist and to work; the other BSDs are
    // not among them. (`nix` names the flag only for non-Apple targets.)
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    ))]
    let flags = MsgFlags::from_bits_retain(libc::MSG_NOSIGNAL);
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    let flags = MsgFlags::empty();

    // Errors are deliberately ignored: the peer then tries the next socket or
    // creates the region itself.
    while sendmsg::<()>(sock.as_raw_fd(), &iov, &cmsgs, flags, None) == Err(Errno::EINTR) {}
}

// ---------------------------------------------------------------------------
// Shared memory region
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
const MADV_COLLAPSE: libc::c_int = 25;

/// Map a large shared object at a 2 MiB-aligned address when possible.
///
/// File-backed huge pages require both the file offset and virtual address to
/// have the required alignment. The anonymous reservation makes the later
/// MAP_FIXED replacement safe: it can only replace the address range reserved
/// by this function, never an unrelated mapping.
#[cfg(target_os = "linux")]
unsafe fn map_shared(fd: RawFd, len: usize) -> *mut libc::c_void {
    // SAFETY: The caller passes `len` > 0 and an open `fd` whose file is at least `len` bytes (the
    // creator sized it; a joiner checked its size with region_file_matches).
    // mmap_huge_aligned only replaces address space it reserved itself.
    unsafe { mmap_huge_aligned(len, libc::MAP_SHARED, fd, 0) }
}

#[cfg(not(target_os = "linux"))]
unsafe fn map_shared(fd: RawFd, len: usize) -> *mut libc::c_void {
    // SAFETY: plain anonymous/file mmap with a null hint; the caller owns the result.
    unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    }
}

/// A read-write `MAP_SHARED` mapping of a region's file, unmapped when dropped.
struct Mapping {
    ptr: *mut u8,
    len: usize,
}

impl Mapping {
    /// Map `len` > 0 bytes of `file`, which must be at least that long (the creator sized it, a
    /// joiner checked it with `region_file_matches`).
    fn new(file: &File, len: usize) -> Option<Mapping> {
        // SAFETY: `len` is non-zero and `file` is open and at least `len` bytes long, as
        // map_shared requires (see above). The result is compared with MAP_FAILED right below.
        let ptr = unsafe { map_shared(file.as_raw_fd(), len) };
        (ptr != libc::MAP_FAILED).then(|| Mapping {
            ptr: ptr.cast(),
            len,
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` are exactly the mapping created in `new`, unmapped only here. Its
        // owner hands out borrows of the bytes only through `&self` (`SharedMemory::as_slice`),
        // so none of them outlives this drop.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

/// A mapped region plus the server that hands its descriptor to later processes.
pub struct SharedMemory {
    mapping: Mapping,
    socket_path: PathBuf,
    /// Write end of the shutdown pipe; closing it wakes and stops the server.
    shutdown: Option<PipeWriter>,
    server: Option<JoinHandle<()>>,
    /// True when we created and initialised the region rather than joining one.
    creator: bool,
}

// SAFETY: the mapping is owned by this value and unmapped only when it is dropped; after
// initialisation it is read-only shared data handed out only as `&[u8]`, and the server thread
// holds file descriptors, not the pointer.
unsafe impl Send for SharedMemory {}
// SAFETY: Same argument as for `Send` above: the mapping is owned by this value and unmapped only
// when it is dropped, joiners never write, and the server thread holds file descriptors, not the
// pointer.
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Directory name for `name`, matching C++ `make_sentinel_base`.
    fn sentinel_base(name: &str) -> String {
        format!("sfshm_{:016}", hash_string(name))
    }

    /// Attach to (or create) the region identified by `name`.
    ///
    /// `len` is the region size in bytes. `init` is called with the freshly
    /// mapped region **only when this process is the creator**; joiners must not
    /// write, since the pages are already populated and shared with others.
    pub fn open<F>(name: &str, len: usize, init: F) -> Option<SharedMemory>
    where
        F: FnOnce(&mut [u8]),
    {
        if len == 0 || len > isize::MAX as usize {
            return None;
        }

        let temp_root = TempRoot::get()?;
        let mut base = Self::sentinel_base(name);
        base.truncate(NAME_MAX.min(base.len()));

        let shared_dir = temp_root.prefix.join(&base);
        let init_lock_path = shared_dir.join("init_lock");
        // More than one Engine/replica can own this region in a process. Each
        // needs an independently owned socket so dropping one cannot unlink another.
        static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(0);
        let instance = NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed);
        let socket_path = shared_dir.join(format!("{}-{instance}.sock", std::process::id()));

        // Bail before creating anything if our socket path would not fit in a
        // sockaddr_un, rather than binding a truncated path.
        let addr = socket_addr(&socket_path)?;

        if let Err(e) = DirBuilder::new().mode(0o700).create(&shared_dir)
            && e.kind() != io::ErrorKind::AlreadyExists
        {
            return None;
        }

        // Everything below happens under the init lock, including publishing our
        // own socket: releasing earlier would let a joiner see the directory
        // without a servable socket and needlessly create a second region.
        let _lock = InitLock::acquire(&init_lock_path)?;

        let mut file = None;
        for peer in Self::peer_sockets(&shared_dir) {
            if peer == socket_path {
                continue;
            }
            if let Some(f) = try_receive_fd(&peer) {
                if !region_file_matches(&f, len) {
                    return None;
                }
                file = Some(f);
                break;
            }
        }

        let creator = file.is_none();
        let file = match file {
            Some(f) => f,
            None => Self::create_region_file(len)?,
        };

        // From here on, an early return unmaps the region again (`Mapping::drop`).
        let mapping = Mapping::new(&file, len)?;

        // Best-effort: ask for transparent huge pages, as C++ does. The weights
        // are large and TLB-sensitive.
        #[cfg(target_os = "linux")]
        // SAFETY: `ptr`/`len` are the mapping created above; MADV_HUGEPAGE is advisory and cannot
        // invalidate it. Failure is ignored (best effort).
        unsafe {
            libc::madvise(mapping.ptr.cast(), len, libc::MADV_HUGEPAGE);
        }

        // Set up every fallible serving resource before consuming the FnOnce
        // initializer. The init lock prevents another process from discovering
        // our socket until this function returns, so the server cannot expose
        // uninitialized bytes. If setup fails, the caller can still initialize
        // its private-memory fallback.
        let (shutdown_r, shutdown_w) = io::pipe().ok()?;

        let server_fd = create_unix_socket()?;
        let _ = fs::remove_file(&socket_path);
        let listening =
            bind(server_fd.as_raw_fd(), &addr).and_then(|()| listen(&server_fd, Backlog::new(5)?));
        if listening.is_err() {
            // A successful bind followed by a failed listen leaves the socket file behind;
            // unlinking a path that was never bound is harmless.
            let _ = fs::remove_file(&socket_path);
            return None;
        }

        let Some(server) = Self::spawn_server(file, shutdown_r, UnixListener::from(server_fd))
        else {
            let _ = fs::remove_file(&socket_path);
            return None;
        };
        register_socket_for_cleanup(&socket_path);

        // Own every resource before calling user code. If init unwinds, Drop
        // withdraws the socket and unmaps the unpublished region under the init lock.
        let region = SharedMemory {
            mapping,
            socket_path,
            shutdown: Some(shutdown_w),
            server: Some(server),
            creator,
        };

        if creator {
            // SAFETY: the mapping is a fresh read-write mapping of `len` bytes owned by `region`,
            // and only the creator reaches this branch. The server thread serves the fd but never
            // touches the mapping, and the init lock (held until return) keeps joiners from
            // attaching before `init` finishes. Any bit pattern is a valid u8.
            init(unsafe { core::slice::from_raw_parts_mut(region.mapping.ptr, len) });

            // Collapse the initialized mapping into transparent huge pages.
            // This is best-effort: older kernels simply reject MADV_COLLAPSE.
            #[cfg(target_os = "linux")]
            // SAFETY: Same mapping as above; MADV_COLLAPSE is advisory and older kernels reject it
            // with EINVAL, which is ignored.
            unsafe {
                libc::madvise(region.mapping.ptr.cast(), len, MADV_COLLAPSE);
            }
        }

        Some(region)
    }

    /// Create the anonymous file backing a new region and size it to `len`.
    fn create_region_file(len: usize) -> Option<File> {
        // Upstream `4c37700c` changed this test from `__linux__ || __FreeBSD__` to
        // `defined(MFD_CLOEXEC)`, because the first spelling claims platforms that
        // define `__linux__` but ship no `memfd_create` (issue #7026).
        //
        // Rust has no `cfg(defined(...))`, so the condition is instead the set of
        // targets for which `nix` (and the `libc` crate under it) declares
        // `memfd_create`/`MFD_CLOEXEC` -- which is the same question, asked of the
        // same headers. Android was missing here: it defines `__linux__`, so C++
        // always covered it, but the port's `target_os = "linux"` does not match it.
        //
        // As in C++ there is no runtime fallback: if the call fails the whole
        // region creation fails, which disables sharing and leaves each thread
        // with its own copy of the weights. The engine still plays.
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
        let file = {
            use nix::sys::memfd::{MFdFlags, memfd_create};
            let flags = MFdFlags::MFD_CLOEXEC | MFdFlags::MFD_ALLOW_SEALING;
            File::from(memfd_create(c"replicated_data", flags).ok()?)
        };
        // macOS and the BSDs without memfd_create: an unlinked temp file is
        // equivalent for our purposes. It has no name once unlinked, so it
        // is reachable only through the descriptors we pass around, and it
        // disappears when the last one closes.
        #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
        let file = {
            let (fd, path) = nix::unistd::mkstemp("/tmp/stockfish_replicated_data.XXXXXX").ok()?;
            set_cloexec(fd.as_fd());
            // Unlink immediately: the fd keeps the inode alive.
            let _ = fs::remove_file(path);
            File::from(fd)
        };

        file.set_len(len as u64).ok()?;
        // These seals freeze only the file's size, allowing initialization and
        // normal shared mappings. Failure disables sharing.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        fcntl(
            &file,
            FcntlArg::F_ADD_SEALS(
                SealFlag::F_SEAL_SHRINK | SealFlag::F_SEAL_GROW | SealFlag::F_SEAL_SEAL,
            ),
        )
        .ok()?;
        Some(file)
    }

    fn peer_sockets(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension() == Some(OsStr::new("sock")) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Background thread that serves `file` to any process that connects.
    ///
    /// Exits when the write end of the shutdown pipe is closed, which `Drop`
    /// does. `poll` watches both so shutdown is immediate rather than waiting
    /// for the next connection.
    fn spawn_server(
        file: File,
        shutdown_r: PipeReader,
        listener: UnixListener,
    ) -> Option<JoinHandle<()>> {
        std::thread::Builder::new()
            .name("sf-shm-server".into())
            .spawn(move || {
                const SERVER: usize = 0;
                const SHUTDOWN: usize = 1;

                loop {
                    let mut fds = [
                        PollFd::new(listener.as_fd(), PollFlags::POLLIN),
                        PollFd::new(shutdown_r.as_fd(), PollFlags::POLLIN),
                    ];

                    match poll(&mut fds, PollTimeout::NONE) {
                        Ok(_) => {}
                        Err(Errno::EINTR) => continue,
                        Err(_) => break,
                    }

                    // Any event at all (`None` is one nix cannot name).
                    if fds[SHUTDOWN].any() != Some(false) {
                        break;
                    }

                    if fds[SERVER]
                        .revents()
                        .is_some_and(|events| events.contains(PollFlags::POLLIN))
                    {
                        // std accepts with close-on-exec set (`accept4` where there is one) and
                        // retries EINTR.
                        let Ok((client, _)) = listener.accept() else {
                            continue;
                        };

                        // A client that was accepted but has already gone --
                        // crashed, or just past its own 1 s join timeout --
                        // makes the `sendmsg` below fail with EPIPE, and the
                        // default disposition of the accompanying SIGPIPE kills
                        // this process: the one that owns the region everybody
                        // else is attaching to. One impatient peer would take
                        // the whole network down.
                        //
                        // Upstream `762dd1da` sets `SO_NOSIGPIPE` immediately
                        // before `sendmsg`. That is too late on macOS, where the
                        // `setsockopt` returns EINVAL once the peer has closed,
                        // so for precisely the client this guards against the
                        // option never gets applied. Here the listening socket
                        // already carries it (`create_unix_socket`) and Darwin
                        // hands it on to every accepted socket, and `send_fd`
                        // passes `MSG_NOSIGNAL` where the platform has it.
                        //
                        // Setting it again now, while the connection may still
                        // be healthy, is a second layer for the BSDs where that
                        // inheritance has not been verified. It cannot help a
                        // peer that is already gone. `tests/shm_sigpipe.rs`
                        // covers the whole arrangement.
                        set_nosigpipe(client.as_fd());

                        send_fd(client.as_fd(), file.as_fd());
                    }
                }
            })
            .ok()
    }

    /// The mapped bytes. Read-only by contract: other processes share these pages.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapping lives as long as `self` (it is unmapped only when `self` is
        // dropped, so no borrow from here survives it). It was fully initialised by the creator
        // before any joiner could attach and is never written afterwards, so a shared &[u8] is
        // sound.
        unsafe { core::slice::from_raw_parts(self.mapping.ptr, self.mapping.len) }
    }

    pub fn len(&self) -> usize {
        self.mapping.len
    }

    pub fn is_empty(&self) -> bool {
        self.mapping.len == 0
    }

    /// True when this process created the region rather than joining an existing
    /// one. Only useful for diagnostics and tests.
    pub fn is_creator(&self) -> bool {
        self.creator
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        // Stop serving first: unlink so nobody new finds us, then close the
        // shutdown pipe to wake the poll loop, then join. The region is unmapped
        // afterwards, when the `mapping` field is dropped.
        let _ = fs::remove_file(&self.socket_path);
        unregister_socket(&self.socket_path);

        drop(self.shutdown.take());
        if let Some(h) = self.server.take() {
            let _ = h.join();
        }
    }
}

/// Test-only guard that removes a region's per-name directory when dropped.
///
/// `SharedMemory` keeps the directory and its `init_lock` on purpose (as C++ does: removing them
/// would race with a concurrent creator). Tests that mint a name unique to one run would otherwise
/// leave a directory behind on every run; nobody else can be using such a name. Declare the guard
/// before the regions (or first in a destructured tuple) so that it is dropped after them.
#[cfg(test)]
pub(crate) struct RegionDirCleanup(PathBuf);

#[cfg(test)]
impl RegionDirCleanup {
    fn for_name(name: &str) -> Self {
        let root = TempRoot::get().expect("temp root");
        Self(root.prefix.join(SharedMemory::sentinel_base(name)))
    }

    pub(crate) fn for_region(region: &SharedMemory) -> Self {
        Self(
            region
                .socket_path
                .parent()
                .expect("socket directory")
                .to_path_buf(),
        )
    }
}

#[cfg(test)]
impl Drop for RegionDirCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0.join("init_lock"));
        let _ = fs::remove_dir(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn region_size_is_sealed_and_unsealed_peers_are_rejected() {
        use nix::sys::memfd::{MFdFlags, memfd_create};

        let file = SharedMemory::create_region_file(4096).unwrap();
        for size in [0, 8192] {
            let err = file.set_len(size).unwrap_err();
            assert_eq!(err.raw_os_error(), Some(libc::EPERM));
        }
        assert!(region_file_matches(&file, 4096));
        let unsealed = File::from(memfd_create(c"unsealed-peer", MFdFlags::MFD_CLOEXEC).unwrap());
        unsealed.set_len(4096).unwrap();
        assert!(!region_file_matches(&unsealed, 4096));
    }

    #[test]
    fn multiple_descriptor_messages_are_rejected_without_leaks() {
        use std::io::Read;

        // On Linux also the largest message the kernel accepts, which must arrive untruncated for
        // its descriptors to be closed (see SCM_MAX_FD). Not elsewhere: macOS defaults to 256 open
        // files, which that many descriptors could exhaust for the tests running alongside.
        let counts: &[usize] = if cfg!(target_os = "linux") {
            &[2, 4, 16, SCM_MAX_FD]
        } else {
            &[2, 4, 16]
        };
        for &count in counts {
            let path = std::env::temp_dir().join(format!(
                "sc-fds-{}-{:?}-{count}.sock",
                std::process::id(),
                std::thread::current().id()
            ));
            let listener = UnixListener::bind(&path).unwrap();

            // The descriptor sent is the write end of a pipe: once the sender has closed its own
            // copy, the read end reports end-of-file exactly when the receiver closed every copy
            // it was given.
            let (mut pipe_read, pipe_write) = io::pipe().unwrap();

            let server = std::thread::spawn(move || {
                let (client, _) = listener.accept().unwrap();
                let fds = vec![pipe_write.as_raw_fd(); count];
                let cmsgs = [ControlMessage::ScmRights(&fds)];
                let iov = [IoSlice::new(&[1])];
                let sent = sendmsg::<()>(client.as_raw_fd(), &iov, &cmsgs, MsgFlags::empty(), None);
                assert_eq!(sent, Ok(1));
            });

            assert!(try_receive_fd(&path).is_none(), "{count} descriptors");
            server.join().unwrap();
            fs::remove_file(path).unwrap();

            // Every received copy was owned and closed again, and the sender's copy went away
            // with the server thread, so no writer is left. Non-blocking, so that a leaked writer
            // fails the assertion (EAGAIN) instead of hanging the test.
            fcntl(&pipe_read, FcntlArg::F_SETFL(nix::fcntl::OFlag::O_NONBLOCK)).unwrap();
            let mut byte = [0u8];
            assert_eq!(
                pipe_read.read(&mut byte).ok(),
                Some(0),
                "{count} descriptors: a received descriptor leaked"
            );
        }
    }

    #[test]
    fn sentinel_base_is_stable_and_prefixed() {
        let a = SharedMemory::sentinel_base("hello");
        let b = SharedMemory::sentinel_base("hello");
        let c = SharedMemory::sentinel_base("world");
        assert_eq!(a, b, "name hashing must be deterministic within a build");
        assert_ne!(a, c);
        assert!(a.starts_with("sfshm_"));
        assert!(a.len() <= NAME_MAX);
    }

    #[test]
    fn socket_addr_rejects_overlong_paths() {
        // sun_path is 104 bytes on macOS, 108 on Linux; anything at or past the
        // limit must be refused rather than silently truncated.
        let long = PathBuf::from("/".to_string() + &"a".repeat(200));
        assert!(socket_addr(&long).is_none());
        assert!(socket_addr(Path::new("/tmp/x.sock")).is_some());
    }

    fn unique_name(label: &str) -> String {
        format!(
            "stockcrab-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
    }

    #[test]
    fn received_region_requires_exact_backing_size() {
        let name = unique_name("size");
        let cleanup = RegionDirCleanup::for_name(&name);
        let owner = SharedMemory::open(&name, 4096, |buf| buf.fill(42)).unwrap();
        assert!(cleanup.0.is_dir());
        assert!(region_file_matches(
            &SharedMemory::create_region_file(4096).unwrap(),
            4096
        ));
        for len in [1, 4095, 4097, 8192] {
            assert!(
                SharedMemory::open(&name, len, |_| panic!("invalid peer must be rejected"))
                    .is_none()
            );
        }
        assert!(SharedMemory::open(&name, 0, |_| {}).is_none());
        assert!(SharedMemory::open(&name, usize::MAX, |_| {}).is_none());
        assert!(owner.as_slice().iter().all(|&byte| byte == 42));
    }

    #[test]
    fn same_process_instances_share_and_own_independent_sockets() {
        let name = unique_name("instances");
        let cleanup = RegionDirCleanup::for_name(&name);
        let owner = SharedMemory::open(&name, 4096, |buf| buf.fill(77)).unwrap();
        assert!(cleanup.0.is_dir());
        let joined =
            SharedMemory::open(&name, 4096, |_| panic!("must join existing owner")).unwrap();
        assert!(!joined.is_creator());
        assert_ne!(owner.socket_path, joined.socket_path);
        assert!(joined.as_slice().iter().all(|&byte| byte == 77));
        let survivor_path = joined.socket_path.clone();
        drop(owner);
        assert!(survivor_path.exists());
        let next =
            SharedMemory::open(&name, 4096, |_| panic!("survivor must still serve")).unwrap();
        assert!(!next.is_creator());
        assert!(next.as_slice().iter().all(|&byte| byte == 77));
        drop(joined);
        assert!(next.socket_path.exists());
    }

    #[test]
    fn initializer_unwind_withdraws_unpublished_region() {
        let name = unique_name("unwind");
        let cleanup = RegionDirCleanup::for_name(&name);
        assert!(
            std::panic::catch_unwind(|| {
                SharedMemory::open(&name, 4096, |buf| {
                    buf.fill(99);
                    panic!("initializer witness");
                })
            })
            .is_err()
        );
        let region = SharedMemory::open(&name, 4096, |buf| buf.fill(23)).unwrap();
        assert!(cleanup.0.is_dir());
        assert!(region.is_creator());
        assert!(region.as_slice().iter().all(|&byte| byte == 23));
    }
}
