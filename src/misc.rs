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

//! Miscellaneous engine utilities – Rust port of Stockfish misc.cpp/h
//!
//! Focuses on correctness for key helpers, structured for performance and safety.

use core::sync::atomic::{AtomicI64, Ordering};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Version & compiler information
// ---------------------------------------------------------------------------

/// Crate version (from `Cargo.toml`).
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Short git hash of the source tree, captured at compile time by `build.rs`
/// (`nogit` when built outside a repository).
const GIT_HASH: &str = env!("STOCKCRAB_GIT_HASH");

/// Upstream Stockfish commit this port is synced to (`config/upstream.env`),
/// captured at compile time by `build.rs`.
const UPSTREAM_REF: &str = env!("STOCKCRAB_UPSTREAM_REF");

/// Returns the full name of this engine build:
///
/// ```text
/// Stockcrab 0.1.0 (git abcd1234)
/// ```
///
/// Everything comes from compile-time metadata (crate version and git hash via
/// `build.rs`), so the identity never changes between runs of one binary and
/// no wall-clock date is embedded (reproducible builds).
pub fn engine_version_info() -> String {
    format!("Stockcrab {VERSION} (git {GIT_HASH})")
}

/// The upstream Stockfish commit this port tracks.
pub fn upstream_reference() -> &'static str {
    UPSTREAM_REF
}

/// Returns extended engine info, optionally formatted for UCI.
pub fn engine_info(to_uci: bool) -> String {
    let sep = if to_uci { "\nid author " } else { " by " };
    format!(
        "{}{}the Stockfish developers (see AUTHORS file)",
        engine_version_info(),
        sep
    )
}

/// Warns when this binary was compiled without the CPU features the NNUE
/// kernels need. The SIMD paths are selected at compile time (like the C++
/// `ARCH=` Makefile variable), so a plain `cargo build --release` on x86-64
/// targets baseline SSE2 and runs the scalar affine layers, roughly 10x slower.
/// On AArch64 a target without `dotprod` (e.g. the Linux baseline) gets the
/// plain NEON kernels unless the `neon-dotprod` Cargo feature is enabled.
pub fn build_performance_warning() -> Option<&'static str> {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "ssse3")))]
    {
        Some(
            "warning: this build targets baseline x86-64 and uses scalar NNUE code \
             (about 10x slower). Build with `make build-x86-64-v3` or `make build-native`.",
        )
    }
    #[cfg(all(target_arch = "aarch64", not(stockcrab_dotprod)))]
    {
        Some(
            "note: this build uses the plain NEON kernels; on ARMv8.2+ CPUs with \
             dot-product support build with `--features neon-dotprod` (about 20% faster).",
        )
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", not(target_feature = "ssse3")),
        all(target_arch = "aarch64", not(stockcrab_dotprod))
    )))]
    {
        None
    }
}

/// Returns the text of the `compiler` command: engine identity with the upstream
/// Stockfish attribution, the build target, the compile-time CPU features and the
/// NNUE kernel families that will run (see [`isa_report`]).
pub fn compiler_info() -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{}, based on Stockfish {}\n",
        engine_version_info(),
        UPSTREAM_REF
    ));
    s.push_str(&format!(
        "Compiled by {} for {}-{}, {}-bit\n",
        env!("STOCKCRAB_RUSTC_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
        usize::BITS
    ));
    s.push_str(&isa_report());
    s
}

/// Compile-time target features of interest on the current architecture.
fn compile_time_features() -> Vec<&'static str> {
    let mut f = Vec::new();
    macro_rules! feat {
        ($($name:literal),*) => { $( if cfg!(target_feature = $name) { f.push($name); } )* };
    }
    feat!(
        "sse2",
        "ssse3",
        "sse4.1",
        "avx2",
        "avxvnni",
        "avx512f",
        "avx512bw",
        "avx512vnni",
        "neon",
        "dotprod"
    );
    // The SDOT kernels follow `cfg(stockcrab_dotprod)` (see build.rs): the
    // target feature above, or this Cargo feature asserting the hardware.
    if cfg!(all(target_arch = "aarch64", feature = "neon-dotprod")) {
        f.push("neon-dotprod(cargo feature)");
    }
    f
}

/// Kernel families selected for this build and CPU, one line per stage.
///
/// This MIRRORS the dispatch rules in the NNUE sources (it does not call into
/// them), so it must be updated when they change:
/// * accumulator update (`nnue_accumulator.rs`, x86-64): RUNTIME choice,
///   AVX-512 (needs avx512f+bw) > AVX2 > scalar. Not gated at compile time, so
///   a build labelled x86-64-v3 still runs AVX-512 kernels on AVX-512 hardware.
/// * feature-transformer output (`nnue_feature_transformer.rs`): AVX-512 only
///   if compiled with avx512f+bw AND detected at run time; otherwise AVX2 /
///   SSE2 / scalar chosen at compile time.
/// * paired sqr-clipped-ReLU (`sqr_clipped_relu.rs`): same rule as the
///   transformer (AVX-512 compile+runtime, else compile-time AVX2).
/// * clipped ReLU (`clipped_relu.rs`): runtime AVX2 > SSE4.1 > SSE2.
/// * affine / sparse affine layers (`simd.rs`, `affine_transform*.rs`): purely
///   compile-time: AVX-512 (avx512f+bw) > AVX2 > SSE (ssse3 for the SIMD kernels)
///   > scalar. Kernel width follows `target_feature`, never the CPU.
/// * AArch64: every kernel is NEON, fixed at compile time; the dot-product
///   kernels (`simd.rs`, `affine_transform*.rs`) follow `cfg(stockcrab_dotprod)`.
pub fn isa_report() -> String {
    let mut s = String::new();
    let ct = compile_time_features();
    #[cfg(target_arch = "x86_64")]
    s.push_str(
        "NNUE kernel selection: adaptive (accumulator/transform/clipped-ReLU kernels are \
         chosen at run time on x86-64; affine layers are fixed at compile time)\n",
    );
    #[cfg(not(target_arch = "x86_64"))]
    s.push_str("NNUE kernel selection: fixed at compile time\n");
    s.push_str(&format!(
        "Compile-time features: {}\n",
        if ct.is_empty() {
            "baseline".to_string()
        } else {
            ct.join(" ")
        }
    ));

    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
        let sse2 = std::arch::is_x86_feature_detected!("sse2");
        let mut rt = Vec::new();
        for (n, on) in [
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
            ("avx512bw", std::arch::is_x86_feature_detected!("avx512bw")),
            ("avx2", avx2),
            ("sse4.1", sse41),
        ] {
            if on {
                rt.push(n);
            }
        }
        s.push_str(&format!(
            "Runtime-detected features: {}\n",
            if rt.is_empty() {
                "none of avx512f/avx512bw/avx2/sse4.1".to_string()
            } else {
                rt.join(" ")
            }
        ));

        let ct_avx512 = cfg!(all(target_feature = "avx512f", target_feature = "avx512bw"));
        let ct_avx2 = cfg!(target_feature = "avx2");
        let ct_sse2 = cfg!(target_feature = "sse2");

        let accumulator = if avx512 {
            "AVX-512 (runtime)"
        } else if avx2 {
            "AVX2 (runtime)"
        } else {
            "scalar"
        };
        let transform = if ct_avx512 && avx512 {
            "AVX-512 (compile-time + runtime)"
        } else if ct_avx2 {
            "AVX2 (compile-time)"
        } else if ct_sse2 {
            "SSE2 (compile-time)"
        } else {
            "scalar"
        };
        let paired = if ct_avx512 && avx512 {
            "AVX-512 (compile-time + runtime)"
        } else if ct_avx2 {
            "AVX2 (compile-time)"
        } else {
            "scalar"
        };
        let clipped = if avx2 {
            "AVX2 (runtime)"
        } else if sse41 {
            "SSE4.1 (runtime)"
        } else if sse2 {
            "SSE2 (runtime)"
        } else {
            "scalar"
        };
        let affine = if ct_avx512 {
            "AVX-512 (compile-time)"
        } else if ct_avx2 {
            "AVX2 (compile-time)"
        } else if cfg!(target_feature = "ssse3") {
            "SSSE3 (compile-time)"
        } else {
            "scalar (compile-time; about 10x slower)"
        };
        s.push_str(&format!(
            "Accumulator update kernel: {accumulator}\n\
             Feature-transformer output kernel: {transform}\n\
             Paired sqr-clipped-ReLU kernel: {paired}\n\
             Clipped-ReLU kernel: {clipped}\n\
             Affine/sparse layer kernels: {affine}\n"
        ));
        if avx512 && !ct_avx512 {
            s.push_str(
                "Note: AVX-512 hardware detected but this build was not compiled with AVX-512; \
                 only the accumulator kernels use it (mixed-ISA).\n",
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        s.push_str(&format!(
            "Accumulator/transform/clipped-ReLU/affine kernels: NEON{} (compile-time)\n",
            if cfg!(stockcrab_dotprod) {
                " with DOTPROD"
            } else {
                ""
            }
        ));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    s.push_str("All NNUE kernels: scalar\n");
    s
}

/// Check if large pages are available (platform-specific)
pub fn has_large_pages() -> bool {
    // Linux supports transparent huge-page advice, which `LargePageAllocator`
    // issues for its 2 MiB-aligned allocation with a regular-memory fallback.
    #[cfg(all(target_os = "linux", not(target_os = "android")))]
    {
        true
    }

    #[cfg(not(all(target_os = "linux", not(target_os = "android"))))]
    false
}

// ---------------------------------------------------------------------------
// File & string utilities
// ---------------------------------------------------------------------------

/// Parses a decimal string into usize, returning None on overflow or invalid input.
/// Matches C++ `str_to_size_t` which returns `std::optional<usize>`, built on `strtoull`.
///
/// Leading C-locale whitespace and an optional `+` are accepted, and the number may be
/// followed by whitespace (the rest of the string is then ignored). A `-` as the first
/// byte returns None; a `-` after leading whitespace wraps modulo 2^64 as `strtoull`
/// does (mirrors C++).
pub fn str_to_size_t(s: &str) -> Option<usize> {
    if s.is_empty() || s.starts_with('-') {
        return None;
    }
    // C++ calls strtoull: it skips leading whitespace, takes an optional sign and a run of
    // digits, and reports where it stopped. The parse is accepted if that is the end of the
    // string or a whitespace character, so an all-whitespace string yields 0 (no digits:
    // endptr is the start of the string) and a leading `+` is fine.
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && is_space(b[i] as char) {
        i += 1;
    }
    let negative = i < b.len() && b[i] == b'-';
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    let mut value: u64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        // ERANGE: the magnitude does not fit unsigned long long.
        value = value.checked_mul(10)?.checked_add(u64::from(b[i] - b'0'))?;
        i += 1;
    }
    let end = if i == digits_start { 0 } else { i };
    if end < b.len() && !is_space(b[end] as char) {
        return None;
    }
    if i == digits_start {
        value = 0;
    } else if negative {
        value = value.wrapping_neg(); // strtoull negates modulo 2^64
    }
    usize::try_from(value).ok()
}

/// Emulates `std::istream::operator>>` for a built-in integer type (libstdc++ `num_get`, base 10)
/// on a byte buffer, advancing `*pos` the way the stream would.
///
/// Leading whitespace is skipped, an optional `+`/`-` sign and a run of decimal digits are
/// consumed, and anything after the digits (e.g. the `abc` of `5abc`) is left unread. Unsigned
/// targets accept a minus sign and negate modulo max+1 of the target type (so `-1` yields max). `Ok(value)` is a
/// successful extraction. `Err(value)` is a failed one (`failbit`): `0` if no digits were found
/// (or the input ended), the saturated limit if the value does not fit `[min, max]`.
/// The caller decides what a failed stream means; the stream stays unusable afterwards in C++.
pub fn stream_extract_int(
    s: &[u8],
    pos: &mut usize,
    min: i128,
    max: i128,
    unsigned: bool,
) -> Result<i128, i128> {
    while *pos < s.len() && is_space(s[*pos] as char) {
        *pos += 1;
    }
    let mut neg = false;
    if *pos < s.len() && (s[*pos] == b'+' || s[*pos] == b'-') {
        neg = s[*pos] == b'-';
        *pos += 1;
    }
    let start = *pos;
    let mut mag: u128 = 0;
    let mut overflow = false;
    while *pos < s.len() && s[*pos].is_ascii_digit() {
        match mag
            .checked_mul(10)
            .and_then(|m| m.checked_add(u128::from(s[*pos] - b'0')))
        {
            Some(m) if m <= u128::from(u64::MAX) * 2 => mag = m,
            _ => overflow = true,
        }
        *pos += 1;
    }
    if *pos == start {
        return Err(0);
    }
    let saturated = if neg { min } else { max };
    if overflow {
        return Err(saturated);
    }
    if unsigned {
        // strtoull semantics: a negative value wraps, only a magnitude beyond the type overflows.
        if mag > max as u128 {
            return Err(max);
        }
        return Ok(if neg {
            (max as u128 + 1 - mag) as i128 % (max + 1)
        } else {
            mag as i128
        });
    }
    let v = if neg { -(mag as i128) } else { mag as i128 };
    if v < min || v > max {
        Err(saturated)
    } else {
        Ok(v)
    }
}

/// Reads the file at `path` as UTF-8 text, or None on any I/O or decoding error.
pub fn read_file_to_string(path: &str) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// Splits `s` on `delimiter`, returning the segments (no allocation except the Vec).
pub fn split<'a>(s: &'a str, delimiter: &str) -> Vec<&'a str> {
    if s.is_empty() {
        return vec![];
    }
    if delimiter.is_empty() {
        return vec![s];
    }
    s.split(delimiter).collect()
}

/// Mirrors C++ `std::isspace` on an `unsigned char` in the default "C" locale:
/// only the six ASCII space characters count, and no byte >= 0x80 does. Rust's
/// `char::is_whitespace` is Unicode-aware and would also accept NBSP and
/// friends, so it is deliberately not used here.
fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r')
}

/// Splits on the C-locale `isspace` set that C++ `istream >> std::string` uses (ASCII only,
/// including the vertical tab). `str::split_whitespace` would also split on Unicode spaces
/// such as U+00A0, and `split_ascii_whitespace` misses `\v`.
pub fn split_cpp_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|t| !t.is_empty())
}

/// `str::trim` restricted to the C-locale `isspace` set (see [`split_cpp_whitespace`]).
pub fn trim_cpp_whitespace(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// Removes all spaces from `s` in-place.
///
/// Remove all spaces from a string
pub fn remove_whitespace(s: &mut String) {
    s.retain(|c| !is_space(c));
}

/// Returns true if `s` has only spaces.
///
/// Test if a string has only spaces
pub fn is_whitespace(s: &str) -> bool {
    s.chars().all(is_space)
}

// ---------------------------------------------------------------------------
// Memory utilities
// ---------------------------------------------------------------------------

use core::alloc::Layout;
use std::alloc::dealloc;
#[cfg(not(target_os = "linux"))]
use std::alloc::{alloc, alloc_zeroed};

/// Large page memory allocation for better TLB performance.
/// Matches C++ aligned_large_pages_alloc() from src/memory.cpp
pub struct LargePageAllocator {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
    backing: LargePageBacking,
}

enum LargePageBacking {
    // Constructed only where there is no mmap path (not Linux).
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    Heap,
    #[cfg(target_os = "linux")]
    Mmap,
    #[cfg(all(
        target_os = "linux",
        target_arch = "x86_64",
        not(target_os = "android")
    ))]
    ExplicitHuge,
}

/// Map `size` bytes at a 2 MiB-aligned virtual address when possible.
///
/// # Safety
///
/// `fd`/`offset`/`flags` must form a valid mmap request for `size` bytes (or
/// MAP_ANONYMOUS with fd -1, offset 0). The caller owns the returned mapping (unless
/// MAP_FAILED) and must munmap it with the same size.
///
/// Reserving the address range first makes the subsequent `MAP_FIXED`
/// replacement safe: it can only replace pages owned by this allocation.
#[cfg(target_os = "linux")]
#[allow(unsafe_op_in_unsafe_fn)] // the whole body is libc mmap/munmap calls
pub(crate) unsafe fn mmap_huge_aligned(
    size: usize,
    flags: libc::c_int,
    fd: libc::c_int,
    offset: libc::off_t,
) -> *mut libc::c_void {
    const ALIGNMENT: usize = 2 * 1024 * 1024;
    let page_size = libc::sysconf(libc::_SC_PAGESIZE);

    if size >= ALIGNMENT && page_size > 0 {
        let page_size = page_size as usize;
        let mapping_size = size.div_ceil(page_size) * page_size;
        if let Some(reservation_size) = mapping_size.checked_add(ALIGNMENT) {
            let reservation = libc::mmap(
                core::ptr::null_mut(),
                reservation_size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if reservation != libc::MAP_FAILED {
                let base = reservation as usize;
                let Some(aligned) = base
                    .checked_add(ALIGNMENT - 1)
                    .map(|address| address & !(ALIGNMENT - 1))
                else {
                    libc::munmap(reservation, reservation_size);
                    return libc::MAP_FAILED;
                };
                let mapped = libc::mmap(
                    aligned as *mut libc::c_void,
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags | libc::MAP_FIXED,
                    fd,
                    offset,
                );

                if mapped != libc::MAP_FAILED {
                    let prefix = aligned - base;
                    let suffix = reservation_size - prefix - mapping_size;
                    if prefix != 0 {
                        libc::munmap(reservation, prefix);
                    }
                    if suffix != 0 {
                        libc::munmap((aligned + mapping_size) as *mut libc::c_void, suffix);
                    }
                    return mapped;
                }

                libc::munmap(reservation, reservation_size);
            }
        }
    }

    libc::mmap(
        core::ptr::null_mut(),
        size,
        libc::PROT_READ | libc::PROT_WRITE,
        flags,
        fd,
        offset,
    )
}

impl LargePageAllocator {
    /// Allocate memory with large page support if available, otherwise fall back to aligned allocation.
    /// Returns None on allocation failure.
    pub fn allocate(size: usize) -> Option<Self> {
        Self::allocate_impl(size, false, false)
    }

    /// Allocate zero-initialized memory with large page support.
    pub fn allocate_zeroed(size: usize) -> Option<Self> {
        Self::allocate_impl(size, true, false)
    }

    /// Allocate with an explicit-huge-page hint. On Linux x86-64, sufficiently
    /// large allocations first try 1 GiB `MAP_HUGETLB` pages; all failures fall
    /// through to the regular aligned allocation and transparent-huge-page hint.
    pub fn allocate_with_hint(size: usize, huge_page_hint: bool) -> Option<Self> {
        Self::allocate_impl(size, false, huge_page_hint)
    }

    fn allocate_impl(size: usize, zeroed: bool, huge_page_hint: bool) -> Option<Self> {
        let _ = huge_page_hint; // only the Linux x86-64 path uses it
        #[cfg(target_os = "linux")]
        let _ = zeroed; // Anonymous mappings are zero-initialized by the kernel.

        #[cfg(all(
            target_os = "linux",
            target_arch = "x86_64",
            not(target_os = "android")
        ))]
        if huge_page_hint {
            const HUGE_PAGE_SIZE: usize = 1 << 30;
            const MAP_HUGE_SHIFT: i32 = 26; // Linux UAPI's HUGETLB_FLAG_ENCODE_SHIFT
            let huge_size = size.checked_add(HUGE_PAGE_SIZE - 1)? / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE;
            if huge_size >= HUGE_PAGE_SIZE {
                let layout = Layout::from_size_align(huge_size, HUGE_PAGE_SIZE).ok()?;
                // SAFETY: Anonymous private mapping with a null hint, so no existing memory can be
                // affected. `huge_size` is a non-zero multiple of 1 GiB (rounded with checked_add
                // above) and failure (MAP_FAILED) is handled right below.
                let ptr = unsafe {
                    libc::mmap(
                        core::ptr::null_mut(),
                        huge_size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE
                            | libc::MAP_ANONYMOUS
                            | libc::MAP_HUGETLB
                            | (30 << MAP_HUGE_SHIFT),
                        -1,
                        0,
                    )
                };
                if ptr != libc::MAP_FAILED {
                    return Some(Self {
                        ptr: ptr as *mut u8,
                        size: huge_size,
                        layout,
                        backing: LargePageBacking::ExplicitHuge,
                    });
                }
            }
        }

        // Linux uses direct anonymous mappings. Unlike glibc malloc arenas,
        // these cannot survive a free and later be reused by a worker bound to
        // another NUMA node.
        #[cfg(target_os = "linux")]
        {
            const ALIGNMENT: usize = 2 * 1024 * 1024;
            let aligned_size = if size == 0 {
                ALIGNMENT
            } else {
                size.checked_next_multiple_of(ALIGNMENT)?
            };
            let layout = Layout::from_size_align(aligned_size, ALIGNMENT).ok()?;
            // SAFETY: `aligned_size` is a non-zero multiple of 2 MiB (the rounding is
            // checked, so it did not overflow) and fd -1 with MAP_ANONYMOUS is the
            // anonymous-mapping form. MAP_FAILED is checked next.
            let ptr = unsafe {
                mmap_huge_aligned(aligned_size, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            if ptr == libc::MAP_FAILED {
                return None;
            }

            // SAFETY: `ptr`/`aligned_size` describe the mapping just returned (not MAP_FAILED);
            // madvise is advisory and its failure is ignored.
            unsafe {
                libc::madvise(ptr, aligned_size, libc::MADV_HUGEPAGE);
            }
            Some(Self {
                ptr: ptr.cast(),
                size: aligned_size,
                layout,
                backing: LargePageBacking::Mmap,
            })
        }

        #[cfg(not(target_os = "linux"))]
        const ALIGNMENT: usize = 4096;

        // Round up size to alignment boundary
        #[cfg(not(target_os = "linux"))]
        let aligned_size = if size == 0 {
            ALIGNMENT
        } else {
            size.checked_next_multiple_of(ALIGNMENT)?
        };

        // Create layout with proper alignment
        #[cfg(not(target_os = "linux"))]
        let layout = Layout::from_size_align(aligned_size, ALIGNMENT).ok()?;

        #[cfg(not(target_os = "linux"))]
        // SAFETY: `layout` has a non-zero size (aligned_size >= ALIGNMENT), as alloc/alloc_zeroed
        // require, and a null result is handled before the pointer is used.
        unsafe {
            let ptr = if zeroed {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            };

            if ptr.is_null() {
                return None;
            }

            Some(Self {
                ptr,
                size: aligned_size,
                layout,
                backing: LargePageBacking::Heap,
            })
        }
    }

    /// Get the allocated memory pointer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get the allocation size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the layout used for allocation
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl Drop for LargePageAllocator {
    fn drop(&mut self) {
        // SAFETY: `ptr`, `size` and `layout` are exactly what the constructor recorded for this
        // `backing` (alloc(layout) for Heap, the mmap'd range for Mmap/ExplicitHuge), and Drop runs
        // once with unique access, so each allocation is released exactly once.
        unsafe {
            match self.backing {
                LargePageBacking::Heap => dealloc(self.ptr, self.layout),
                #[cfg(target_os = "linux")]
                LargePageBacking::Mmap => {
                    libc::munmap(self.ptr.cast(), self.size);
                }
                #[cfg(all(
                    target_os = "linux",
                    target_arch = "x86_64",
                    not(target_os = "android")
                ))]
                LargePageBacking::ExplicitHuge => {
                    libc::munmap(self.ptr as *mut libc::c_void, self.size);
                }
            }
        }
    }
}

// SAFETY: LargePageAllocator raw pointer is managed exclusively by this struct
// and is valid for the lifetime of the struct. The underlying memory is never
// accessed concurrently without synchronization at a higher level.
unsafe impl Send for LargePageAllocator {}
// SAFETY: No interior mutability: &LargePageAllocator only exposes the pointer value, size and
// layout. Synchronising access to the memory behind the pointer is the caller's job, as for Send
// above.
unsafe impl Sync for LargePageAllocator {}

// ---------------------------------------------------------------------------
// Time utilities
// ---------------------------------------------------------------------------

/// A TimePoint is a count of milliseconds on the clock returned by `now()` (process-relative,
/// monotonic).
pub type TimePoint = i64;

/// Returns current monotonic time in milliseconds (process-relative).
pub fn now() -> TimePoint {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_millis() as i64
}

// ---------------------------------------------------------------------------
// Debug statistics (32 slots)
// ---------------------------------------------------------------------------

// Debug functions used mainly to collect run-time statistics
const MAX_DEBUG_SLOTS: usize = 32;
static HIT: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static MEAN: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static STDEV: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static EXTREMES_MIN: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static EXTREMES_MAX: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static CORREL_SUM1: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static CORREL_SQ1: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static CORREL_SUM2: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static CORREL_SQ2: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static CORREL_CROSS: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();
static COUNTS: OnceLock<[AtomicI64; MAX_DEBUG_SLOTS]> = OnceLock::new();

fn init_debug() {
    HIT.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    MEAN.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    STDEV.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    EXTREMES_MIN.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(i64::MAX)));
    EXTREMES_MAX.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(i64::MIN)));
    CORREL_SUM1.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    CORREL_SQ1.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    CORREL_SUM2.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    CORREL_SQ2.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    CORREL_CROSS.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
    COUNTS.get_or_init(|| core::array::from_fn(|_| AtomicI64::new(0)));
}

/// Counts one sample for `slot`, and a hit for `slot` if `cond` is true. Returns `cond`, so
/// the call can wrap the inspected expression in place.
pub fn dbg_hit_on(cond: bool, slot: usize) -> bool {
    init_debug();
    COUNTS.get().unwrap()[slot].fetch_add(1, Ordering::Relaxed);
    if cond {
        HIT.get().unwrap()[slot].fetch_add(1, Ordering::Relaxed);
    }

    cond
}

/// Record sum for mean calculations. Returns `value`.
pub fn dbg_mean_of(value: i64, slot: usize) -> i64 {
    init_debug();
    MEAN.get().unwrap()[slot].fetch_add(value, Ordering::Relaxed);
    COUNTS.get().unwrap()[slot].fetch_add(1, Ordering::Relaxed);

    value
}

/// Record sum of squares for standard deviation. Returns `value`.
pub fn dbg_stdev_of(value: i64, slot: usize) -> i64 {
    init_debug();
    STDEV.get().unwrap()[slot].fetch_add(value * value, Ordering::Relaxed);
    MEAN.get().unwrap()[slot].fetch_add(value, Ordering::Relaxed);
    COUNTS.get().unwrap()[slot].fetch_add(1, Ordering::Relaxed);

    value
}

/// Track min/max extremes. Returns `value`.
pub fn dbg_extremes_of(value: i64, slot: usize) -> i64 {
    init_debug();
    EXTREMES_MIN.get().unwrap()[slot].fetch_min(value, Ordering::Relaxed);
    EXTREMES_MAX.get().unwrap()[slot].fetch_max(value, Ordering::Relaxed);

    value
}

/// Correlation sums.
pub fn dbg_correl_of(value1: i64, value2: i64, slot: usize) {
    init_debug();
    COUNTS.get().unwrap()[slot].fetch_add(1, Ordering::Relaxed);
    CORREL_SUM1.get().unwrap()[slot].fetch_add(value1, Ordering::Relaxed);
    CORREL_SQ1.get().unwrap()[slot].fetch_add(value1 * value1, Ordering::Relaxed);
    CORREL_SUM2.get().unwrap()[slot].fetch_add(value2, Ordering::Relaxed);
    CORREL_SQ2.get().unwrap()[slot].fetch_add(value2 * value2, Ordering::Relaxed);
    CORREL_CROSS.get().unwrap()[slot].fetch_add(value1 * value2, Ordering::Relaxed);
}

/// Prints debug stats to stderr.
pub fn dbg_print() {
    init_debug();
    let counts = COUNTS.get().unwrap();
    // Hits
    for i in 0..MAX_DEBUG_SLOTS {
        let n = counts[i].load(Ordering::Relaxed);
        if n > 0 {
            let hits = HIT.get().unwrap()[i].load(Ordering::Relaxed);
            let rate = (hits as f64) * 100.0 / (n as f64);
            eprintln!(
                "Hit #{}: Total {} Hits {} Hit Rate (%) {:.3}",
                i, n, hits, rate
            );
        }
    }
    // Mean / Stdev
    for i in 0..MAX_DEBUG_SLOTS {
        let n = counts[i].load(Ordering::Relaxed);
        if n > 0 {
            let sum = MEAN.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            eprintln!("Mean #{}: Total {} Mean {:.6}", i, n, sum / n as f64);
        }
    }
    for i in 0..MAX_DEBUG_SLOTS {
        let n = counts[i].load(Ordering::Relaxed);
        if n > 0 {
            let sum = MEAN.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let sumsq = STDEV.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let mean = sum / n as f64;
            let var = (sumsq / n as f64) - (mean * mean);
            let stddev = if var > 0.0 { var.sqrt() } else { 0.0 };
            eprintln!("Stdev #{}: Total {} Stdev {:.6}", i, n, stddev);
        }
    }
    for i in 0..MAX_DEBUG_SLOTS {
        let n = counts[i].load(Ordering::Relaxed);
        if n > 0 {
            let mn = EXTREMES_MIN.get().unwrap()[i].load(Ordering::Relaxed);
            let mx = EXTREMES_MAX.get().unwrap()[i].load(Ordering::Relaxed);
            eprintln!("Extremity #{}: Total {} Min {} Max {}", i, n, mn, mx);
        }
    }
    for i in 0..MAX_DEBUG_SLOTS {
        let n = counts[i].load(Ordering::Relaxed);
        if n > 0 {
            let s1 = CORREL_SUM1.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let s2 = CORREL_SUM2.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let sq1 = CORREL_SQ1.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let sq2 = CORREL_SQ2.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let sc = CORREL_CROSS.get().unwrap()[i].load(Ordering::Relaxed) as f64;
            let n = n as f64;
            let cov = sc / n - (s1 / n) * (s2 / n);
            let var1 = (sq1 / n) - (s1 / n).powi(2);
            let var2 = (sq2 / n) - (s2 / n).powi(2);
            let denom = (var1.max(0.0).sqrt()) * (var2.max(0.0).sqrt());
            let r = if denom > 0.0 { cov / denom } else { 0.0 };
            eprintln!("Correl. #{}: Total {} Coefficient {:.6}", i, n as i64, r);
        }
    }
}

/// Clears all debug stats.
pub fn dbg_clear() {
    init_debug();
    for a in HIT.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in MEAN.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in STDEV.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in EXTREMES_MIN.get().unwrap() {
        a.store(i64::MAX, Ordering::Relaxed)
    }
    for a in EXTREMES_MAX.get().unwrap() {
        a.store(i64::MIN, Ordering::Relaxed)
    }
    for a in CORREL_SUM1.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in CORREL_SQ1.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in CORREL_SUM2.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in CORREL_SQ2.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in CORREL_CROSS.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
    for a in COUNTS.get().unwrap() {
        a.store(0, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Pseudo-random number generator (xorshift64*)
// ---------------------------------------------------------------------------

/// xorshift64star Pseudo-Random Number Generator
/// This class is based on original code written and dedicated
/// to the public domain by Sebastiano Vigna (2014).
/// It has the following characteristics:
///
///  -  Outputs 64-bit numbers
///  -  Passes Dieharder and SmallCrush test batteries
///  -  Does not require warm-up, no zeroland to escape
///  -  Internal state is a single 64-bit integer
///  -  Period is 2^64 - 1
///  -  Speed: 1.60 ns/call (Core i7 @3.40GHz)
///
/// For further analysis see
///   <http://vigna.di.unimi.it/ftp/papers/xorshift.pdf>
#[derive(Clone, Copy, Debug)]
pub struct PRNG {
    state: u64,
}

impl PRNG {
    /// Creates a new generator; `seed` must be non-zero.
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "Seed must be non-zero");
        PRNG { state: seed }
    }

    #[inline]
    fn rand64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(2685821657736338717u64)
    }

    /// Returns the next random `T` by casting the random bits.
    pub fn rand<T>(&mut self) -> T
    where
        T: From<u64>,
    {
        T::from(self.rand64())
    }
}

// ---------------------------------------------------------------------------
// High-precision multiplication: returns high 64 bits of a*b
// ---------------------------------------------------------------------------

#[inline]
pub fn mul_hi64(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) >> 64) as u64
}

// ---------------------------------------------------------------------------
// Byte hashing (MurmurHash64A)
// ---------------------------------------------------------------------------

/// Hash function based on public domain MurmurHash64A, by Austin Appleby.
///
/// Ported from C++ `hash_bytes`. Tail bytes are interpreted as unsigned
/// values, matching the current upstream implementation.
pub fn hash_bytes(data: &[u8]) -> u64 {
    const M: u64 = 0xc6a4a7935bd1e995;
    const R: u32 = 47;

    let size = data.len();
    let mut h: u64 = (size as u64).wrapping_mul(M);

    let n = size & !7usize;
    let mut i = 0;
    while i < n {
        let mut k = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());

        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h ^= k;
        h = h.wrapping_mul(M);
        i += 8;
    }

    if size & 7 != 0 {
        let mut k: u64 = 0;
        for j in (0..(size & 7)).rev() {
            k = (k << 8) | data[n + j] as u64;
        }
        h ^= k;
        h = h.wrapping_mul(M);
    }

    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;

    h
}

/// Hashes a string's bytes using [`hash_bytes`]. Ported from C++ `hash_string`.
#[inline]
pub fn hash_string(s: &str) -> u64 {
    hash_bytes(s.as_bytes())
}

// ---------------------------------------------------------------------------
// Command line utilities
// ---------------------------------------------------------------------------

/// The process arguments (C++ `CommandLine`), used for the `bench`-style CLI commands and to
/// find the binary's directory.
pub struct CommandLine {
    pub args: Vec<String>,
}

impl CommandLine {
    pub fn new() -> Self {
        CommandLine {
            args: std::env::args_os()
                .map(|a| a.to_string_lossy().into_owned())
                .collect(),
        }
    }

    /// Directory of the running binary
    ///
    /// Return the directory where our Stockfish binary sits. This is useful,
    /// because when the NNUE network is not embedded in the binary, this directory
    /// is one of the locations where we look for the NNUE file.
    pub fn get_binary_directory(&self) -> String {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "".into())
    }

    /// Current working directory
    pub fn get_working_directory() -> String {
        std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "".into())
    }
}

/// No-op in this port. C++ `set_console_utf8` switches the Windows console code
/// page to UTF-8; Stockcrab does not (Windows consoles are not switched).
pub fn set_console_utf8() {}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Moves first element matching `pred` to front of `vec`.
pub fn move_to_front<T, F>(vec: &mut [T], pred: F)
where
    F: FnMut(&T) -> bool,
{
    if let Some(pos) = vec.iter().position(pred) {
        vec.rotate_left(pos);
    }
}

// ---------------------------------------------------------------------------
// Prefetch utility (best-effort)
// ---------------------------------------------------------------------------

/// Preloads the given address into cache. This is a non-blocking
/// function that doesn't stall the CPU waiting for data to be loaded from memory,
/// which can be quite slow.
#[inline(always)]
pub fn prefetch<T>(addr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: _mm_prefetch is a hint that never faults, even for invalid or unmapped addresses, and
    // does not read or write through `addr` in the abstract machine.
    unsafe {
        use core::arch::x86_64::_mm_prefetch;
        const _MM_HINT_T0: i32 = 3;
        _mm_prefetch(addr as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    // SAFETY: _mm_prefetch is a hint that never faults, even for invalid or unmapped addresses, and
    // does not read or write through `addr` in the abstract machine.
    unsafe {
        use core::arch::x86::_mm_prefetch;
        const _MM_HINT_T0: i32 = 3;
        _mm_prefetch(addr as *const i8, _MM_HINT_T0);
    }
    // C++ `__builtin_prefetch(addr, READ, HIGH)` is `prfm pldl1keep` on AArch64.
    // (The `_prefetch` intrinsic is still unstable, so use inline asm.)
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm pldl1keep` is a hint that never faults, even for invalid addresses; the asm
    // block is declared readonly/nostack and touches no memory or flags.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, preserves_flags, readonly)
        );
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = addr; // no-op on other architectures
    }
}

/// SyncCout struct for synchronized console output
pub struct SyncCout;

impl SyncCout {
    pub fn start() {}

    pub fn end() {
        // A failed flush is ignored. On Linux, Android, macOS and iOS a closed pipe never gets
        // this far: SIGPIPE has its default disposition (`main.rs`) and ends the process, as it
        // does C++ Stockfish. The ignored error only matters where SIGPIPE stays ignored (the
        // other BSDs) or does not exist (Windows).
        let _ = io::stdout().flush();
    }
}

// ---------------------------------------------------------------------------
// Logging functionality
// ---------------------------------------------------------------------------

static LOG_FILE: Mutex<Option<String>> = Mutex::new(None);

/// Enables logging to `filename`, truncating it and writing a start-time header. Later calls
/// to `log_message` append to it.
pub fn start_logger(filename: &str) {
    let mut log_file = LOG_FILE.lock().unwrap();
    if filename.is_empty() {
        *log_file = None;
        return;
    }
    if log_file.as_deref() == Some(filename) {
        return;
    }
    *log_file = Some(filename.to_string());

    // Create or truncate the log file
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)
    {
        writeln!(
            file,
            "Stockcrab logging started at {}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        )
        .ok();
    }
}

/// Log a message to the log file if logging is enabled
pub fn log_message(msg: &str) {
    let log_file = LOG_FILE.lock().unwrap();
    if let Some(ref filename) = *log_file {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(filename) {
            writeln!(file, "{}", msg).ok();
        }
    }
}

// ---------------------------------------------------------------------------
// Fixed-capacity list ValueList<T, N>
// ---------------------------------------------------------------------------

/// A fixed-capacity, stack-allocated list (C++ `ValueList<T, N>`).
#[derive(Debug, Clone)]
pub struct ValueList<T: Copy, const N: usize> {
    data: [T; N],
    size: usize,
}

impl<T: Copy + Default, const N: usize> ValueList<T, N> {
    pub fn new() -> Self {
        ValueList {
            data: [T::default(); N],
            size: 0,
        }
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn push_back(&mut self, value: T) {
        assert!(self.size < N);
        self.data[self.size] = value;
        self.size += 1;
    }
    pub fn begin(&self) -> *const T {
        self.data.as_ptr()
    }
    pub fn end(&self) -> *const T {
        // SAFETY: `size` never exceeds N (push_back and push_back_if_lt assert size < N before
        // incrementing by at most 1), so `add(size)` is at most one past the end of `data`.
        unsafe { self.data.as_ptr().add(self.size) }
    }
    pub fn get(&self, index: usize) -> T {
        self.data[index]
    }
}

impl<T: Copy + Default + PartialOrd, const N: usize> ValueList<T, N> {
    /// Pushes back `value` only if `value < max`. Matches C++
    /// `ValueList::push_back_if_lt`.
    pub fn push_back_if_lt(&mut self, value: T, max: T) {
        assert!(self.size < N);
        self.data[self.size] = value;
        self.size += (value < max) as usize;
    }
}

// ---------------------------------------------------------------------------
// UNIT TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_stockcrab_and_stable() {
        let v = engine_version_info();
        assert!(v.starts_with("Stockcrab "), "{v}");
        assert!(v.contains(env!("CARGO_PKG_VERSION")));
        assert!(!v.contains("Stockfish"));
        assert_eq!(v, engine_version_info());
        assert!(!upstream_reference().is_empty());
        assert!(compiler_info().contains(&format!("based on Stockfish {}", upstream_reference())));
    }

    #[test]
    fn isa_report_names_kernel_families() {
        let r = isa_report();
        assert!(r.contains("Compile-time features:"), "{r}");
        #[cfg(target_arch = "x86_64")]
        {
            for key in [
                "Runtime-detected features:",
                "Accumulator update kernel:",
                "Feature-transformer output kernel:",
                "Clipped-ReLU kernel:",
                "Affine/sparse layer kernels:",
            ] {
                assert!(r.contains(key), "missing {key} in {r}");
            }
            if cfg!(target_feature = "avx2") {
                assert!(r.contains("avx2"), "{r}");
            }
            if cfg!(all(target_feature = "avx512f", target_feature = "avx512bw")) {
                assert!(r.contains("Affine/sparse layer kernels: AVX-512"), "{r}");
            } else if cfg!(target_feature = "avx2") {
                assert!(r.contains("Affine/sparse layer kernels: AVX2"), "{r}");
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(
                r.contains("NNUE kernel selection: fixed at compile time"),
                "{r}"
            );
            assert!(!r.contains("x86-64"), "{r}");
            // The report and the start-up note must agree with the kernels that
            // were compiled in (`cfg(stockcrab_dotprod)`, set by build.rs).
            assert_eq!(
                r.contains("NEON with DOTPROD"),
                cfg!(stockcrab_dotprod),
                "{r}"
            );
            assert_eq!(
                build_performance_warning().is_none(),
                cfg!(stockcrab_dotprod)
            );
            // A target that guarantees dot-product support (Apple Silicon by
            // default) or the Cargo feature must select the SDOT kernels.
            if cfg!(any(target_feature = "dotprod", feature = "neon-dotprod")) {
                assert!(cfg!(stockcrab_dotprod), "{r}");
                assert!(r.contains("dotprod"), "{r}");
            }
        }
    }

    #[test]
    fn compiler_info_names_the_compiler_version() {
        let info = compiler_info();
        let line = info
            .lines()
            .find(|line| line.starts_with("Compiled by "))
            .unwrap_or_else(|| panic!("no compiler line in {info}"));
        // build.rs records `rustc --version`, e.g. `rustc 1.99.0 (...)`.
        let version = env!("STOCKCRAB_RUSTC_VERSION");
        assert!(
            line.starts_with(&format!("Compiled by {version} for ")),
            "{line}"
        );
        assert!(version.starts_with("rustc"), "{version}");
    }

    #[test]
    fn test_str_to_size_t() {
        assert_eq!(str_to_size_t("42"), Some(42));
        // Trailing whitespace is tolerated.
        assert_eq!(str_to_size_t("42 "), Some(42));
        assert_eq!(str_to_size_t("42\n"), Some(42));
        assert_eq!(str_to_size_t("42\t\r\n"), Some(42));
    }

    #[test]
    fn test_str_to_size_t_invalid() {
        assert_eq!(str_to_size_t("not a number"), None);
        assert_eq!(str_to_size_t("-5"), None);
        assert_eq!(str_to_size_t(""), None);
        assert_eq!(str_to_size_t("42x"), None);
    }

    #[test]
    fn test_read_file_to_string() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_read_file_to_string.txt");
        let path_str = path.to_string_lossy().to_string();
        // Write and read back
        fs::write(&path, "hello").unwrap();
        assert_eq!(read_file_to_string(&path_str).unwrap(), "hello");

        // Nonexistent file should return None
        assert!(read_file_to_string("nonexistent_test_read_file").is_none());
    }

    #[test]
    fn test_split() {
        assert_eq!(split("a,b,c", ","), vec!["a", "b", "c"]);
        assert_eq!(split("", ","), Vec::<&str>::new());
        assert_eq!(split("abc", ""), vec!["abc"]);
    }

    #[test]
    fn test_whitespace_funcs() {
        let mut s = String::from(" a b\tc\n");
        assert!(is_whitespace(" \n\t"));
        remove_whitespace(&mut s);
        assert_eq!(s, "abc");
    }

    #[test]
    fn test_now() {
        let t1 = now();
        let t2 = now();
        assert!(t2 >= t1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn large_page_allocator_uses_aligned_mmap() {
        const ALIGNMENT: usize = 2 * 1024 * 1024;
        let allocation = LargePageAllocator::allocate_zeroed(4096).unwrap();
        assert_eq!(allocation.as_ptr() as usize % ALIGNMENT, 0);
        assert!(allocation.size() >= ALIGNMENT);
        // SAFETY: `allocation` is a live zeroed mapping of at least 2 MiB (asserted just above) and
        // outlives `bytes`, so 4096 initialised bytes are valid for reads and writes.
        let bytes = unsafe { core::slice::from_raw_parts_mut(allocation.as_ptr(), 4096) };
        assert!(bytes.iter().all(|&byte| byte == 0));
        bytes[0] = 0x5a;
        bytes[4095] = 0xa5;
        assert_eq!((bytes[0], bytes[4095]), (0x5a, 0xa5));
    }

    #[test]
    fn test_prng() {
        let mut pr = PRNG::new(1);
        let a = pr.rand::<u64>();
        let b = pr.rand::<u64>();
        assert_ne!(a, b);
    }

    #[test]
    fn test_mul_hi64() {
        let high = mul_hi64(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
        // (2^64-1)^2 high 64 bits = 0xFFFF_FFFF_FFFF_FFFE
        assert_eq!(high, 0xFFFF_FFFF_FFFF_FFFE);
    }

    #[test]
    fn test_move_to_front() {
        let mut v = vec![1, 2, 3, 4];
        move_to_front(&mut v, |&x| x == 3);
        assert_eq!(v[0], 3);
    }

    #[test]
    fn test_value_list() {
        let mut vl: ValueList<i32, 4> = ValueList::new();
        vl.push_back(10);
        vl.push_back(20);
        assert_eq!(vl.size(), 2);
        assert_eq!(vl.get(1), 20);
    }

    #[test]
    fn test_value_list_push_back_if_lt() {
        let mut vl: ValueList<i32, 4> = ValueList::new();
        vl.push_back_if_lt(3, 5); // 3 < 5 -> pushed
        vl.push_back_if_lt(7, 5); // 7 >= 5 -> not pushed
        assert_eq!(vl.size(), 1);
        assert_eq!(vl.get(0), 3);
    }

    #[test]
    fn test_hash_bytes_stable() {
        // hash_bytes is deterministic and hash_string routes through it.
        assert_eq!(hash_bytes(b"stockfish"), hash_string("stockfish"));
        assert_ne!(hash_string("a"), hash_string("b"));
        // Empty input hashes to 0 under MurmurHash64A finalization.
        assert_eq!(hash_bytes(&[]), 0);
    }

    #[test]
    fn test_str_to_size_t_overflow() {
        let big = (usize::MAX as u128 + 1).to_string();
        assert_eq!(str_to_size_t(&big), None);
    }

    #[test]
    fn test_debug_stats() {
        dbg_clear();
        dbg_hit_on(true, 0);
        dbg_hit_on(false, 0);
        dbg_mean_of(10, 1);
        dbg_mean_of(30, 1);
        dbg_stdev_of(2, 2);
        dbg_stdev_of(4, 2);
        dbg_extremes_of(5, 3);
        dbg_extremes_of(1, 3);
        dbg_extremes_of(9, 3);
        dbg_correl_of(1, 1, 4);
        dbg_correl_of(2, 2, 4);
        dbg_correl_of(3, 3, 4);
        // We just ensure dbg_print runs without panicking.
        dbg_print();
    }
}
