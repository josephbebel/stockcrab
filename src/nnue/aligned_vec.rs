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

//! Cache-line aligned Vec for SIMD performance.
//!
//! Provides a Vec-like container with guaranteed alignment suitable for SIMD operations.
//! Uses LargePageAllocator to support huge pages (2MB) on Linux for better TLB performance,
//! which is critical for NNUE weights.

use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::sync::Arc;

use bytemuck::Pod;

use crate::misc::LargePageAllocator;
use crate::shm::SystemWideSharedConstant;

/// Allocations below this size come from the ordinary heap instead of a
/// dedicated 2 MiB-aligned, `MADV_HUGEPAGE` mapping. The large-page allocator
/// rounds every request up to 2 MiB, so with transparent huge pages a few KiB of
/// layer weights costs a whole huge page once touched (the 8 layer stacks alone
/// were ~96 MiB resident); only the big transformer arrays benefit from them.
const SMALL_ALLOC_LIMIT: usize = 1 << 20;

/// A small zero-initialized heap block, 64-byte aligned for SIMD loads.
struct HeapBlock {
    ptr: NonNull<u8>,
    layout: core::alloc::Layout,
}

impl HeapBlock {
    fn new_zeroed(size_bytes: usize) -> Option<Self> {
        let layout = core::alloc::Layout::from_size_align(size_bytes.max(1), 64).ok()?;
        // SAFETY: `layout` has non-zero size.
        let ptr = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        Some(Self { ptr, layout })
    }
}

impl Drop for HeapBlock {
    fn drop(&mut self) {
        // SAFETY: allocated in `new_zeroed` with exactly this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: a HeapBlock uniquely owns its allocation.
unsafe impl Send for HeapBlock {}
// SAFETY: as above; the block is never accessed through `&self` beyond handing out the raw pointer.
unsafe impl Sync for HeapBlock {}

/// What owns the memory an [`AlignedVec`] points at.
enum Backing {
    /// A small private heap allocation (see [`SMALL_ALLOC_LIMIT`]).
    Heap(HeapBlock),
    /// Zero length; `ptr` is dangling.
    Empty,
    /// A private allocation, released when this value drops.
    Owned(LargePageAllocator),
    /// A window into a region owned by someone else -- in practice a shared
    /// mapping backed by [`SystemWideSharedConstant`], whose pages other
    /// processes have mapped too.
    ///
    /// The `Arc` keeps the mapping alive for as long as any `AlignedVec` borrows from it;
    /// this is enforced by reference counting rather than by the borrow checker.
    ///
    /// Data behind this backing is immutable. The mutating entry points
    /// (`DerefMut`, `as_mut_ptr`, `resize`, `copy_from_slice`) panic rather than
    /// silently corrupting every other process's view of the weights.
    // The field is never read: holding the `Arc` is what keeps the mapping alive.
    #[allow(dead_code)]
    Shared(Arc<SystemWideSharedConstant>),
}

/// A Vec-like container with guaranteed alignment and huge page support.
///
/// The constructors require `T: Pod`: `Copy` (no drop glue), valid for the all-zeroes pattern
/// (the container hands out zero-filled memory) and for every bit pattern (it reinterprets bytes
/// read from a network file or shared region). `NOT_ZST` additionally rejects zero-sized and
/// over-aligned element types at compile time.
pub struct AlignedVec<T> {
    backing: Backing,
    ptr: NonNull<T>,
    len: usize,
}

impl<T> AlignedVec<T> {
    /// Compile-time guard: zero-sized or over-aligned (>64 bytes) elements would break the capacity
    /// and alignment arithmetic.
    const NOT_ZST: () = {
        assert!(
            core::mem::size_of::<T>() != 0,
            "AlignedVec elements must not be zero-sized"
        );
        assert!(
            core::mem::align_of::<T>() <= 64,
            "AlignedVec elements may be at most 64-byte aligned (the heap block alignment)"
        );
    };

    /// `len * size_of::<T>()`, panicking on overflow.
    fn bytes_for(len: usize) -> usize {
        let () = Self::NOT_ZST;
        len.checked_mul(core::mem::size_of::<T>())
            .expect("AlignedVec length overflows usize")
    }

    /// Zeroed backing store for `size_bytes` (non-zero) bytes and its base pointer.
    fn alloc_backing(size_bytes: usize, what: &str) -> (Backing, NonNull<T>) {
        if size_bytes < SMALL_ALLOC_LIMIT {
            if let Some(block) = HeapBlock::new_zeroed(size_bytes) {
                let ptr = block.ptr.cast::<T>();
                return (Backing::Heap(block), ptr);
            }
        }
        let allocator =
            LargePageAllocator::allocate_zeroed(size_bytes).unwrap_or_else(|| panic!("{what}"));
        // SAFETY: a successful allocation is never null.
        let ptr = unsafe { NonNull::new_unchecked(allocator.as_ptr() as *mut T) };
        (Backing::Owned(allocator), ptr)
    }

    /// Create a new AlignedVec with the given length, zero-initialized.
    pub fn new_zeroed(len: usize) -> Self
    where
        T: Pod,
    {
        if len == 0 {
            return Self {
                backing: Backing::Empty,
                ptr: NonNull::dangling(),
                len: 0,
            };
        }

        let size_bytes = Self::bytes_for(len);
        // Large arrays use LargePageAllocator for huge page support; small ones
        // use the heap. Both are at least cache-line aligned.
        let (backing, ptr) = Self::alloc_backing(size_bytes, "Failed to allocate AlignedVec");

        Self { backing, ptr, len }
    }

    /// Borrow `len` elements starting at `byte_offset` inside a shared region.
    ///
    /// Both the bounds of `byte_offset .. byte_offset + len * size_of::<T>()` and the alignment
    /// of `byte_offset` for `T` are checked here and panic if violated, since getting this
    /// wrong would produce misaligned SIMD loads rather than a clean error.
    pub fn from_shared(
        region: &Arc<SystemWideSharedConstant>,
        byte_offset: usize,
        len: usize,
    ) -> Self
    where
        T: Pod,
    {
        if len == 0 {
            return Self {
                backing: Backing::Empty,
                ptr: NonNull::dangling(),
                len: 0,
            };
        }

        let bytes = region.as_slice().expect("shared region has no mapping");
        let size_bytes = Self::bytes_for(len);
        let window = byte_offset
            .checked_add(size_bytes)
            .and_then(|end| bytes.get(byte_offset..end))
            .unwrap_or_else(|| {
                panic!(
                    "shared slice {byte_offset}..+{size_bytes} out of bounds for region of {} bytes",
                    bytes.len()
                )
            });
        // The length is a whole number of `T`s, so the only possible failure is alignment.
        let window: &[T] = bytemuck::try_cast_slice(window).unwrap_or_else(|_| {
            panic!(
                "shared slice at offset {byte_offset} is misaligned for {}",
                core::any::type_name::<T>()
            )
        });

        Self {
            backing: Backing::Shared(Arc::clone(region)),
            ptr: NonNull::from(window).cast(),
            len,
        }
    }

    /// True when this view borrows shared memory rather than owning its bytes.
    #[inline]
    pub fn is_shared(&self) -> bool {
        matches!(self.backing, Backing::Shared(_))
    }

    /// Panic if this view is read-only shared memory.
    #[inline]
    fn assert_mutable(&self) {
        assert!(
            !self.is_shared(),
            "attempted to mutate an AlignedVec backed by shared memory; those \
             pages are mapped by other processes and must stay immutable"
        );
    }

    /// Returns the length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the capacity (elements)
    #[inline]
    pub fn capacity(&self) -> usize {
        let () = Self::NOT_ZST;
        match &self.backing {
            Backing::Owned(alloc) => alloc.size() / core::mem::size_of::<T>(),
            Backing::Heap(block) => block.layout.size() / core::mem::size_of::<T>(),
            // A shared window is exactly as large as it is long: there is no
            // spare capacity to grow into.
            Backing::Shared(_) => self.len,
            Backing::Empty => 0,
        }
    }

    /// Returns true if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the data
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.assert_mutable();
        self.ptr.as_ptr()
    }

    /// Resize to new length, zero-filling any new elements.
    /// If shrinking, data is truncated.
    ///
    /// # Panics
    /// If `self` is shared (read-only).
    pub fn resize(&mut self, new_len: usize)
    where
        T: Pod,
    {
        if new_len == self.len {
            return;
        }
        self.assert_mutable();

        if new_len == 0 {
            self.backing = Backing::Empty;
            self.ptr = NonNull::dangling();
            self.len = 0;
            return;
        }

        let current_cap = self.capacity();

        if new_len <= current_cap {
            // Just update length, zero new elements if growing
            if new_len > self.len {
                // SAFETY: `new_len <= capacity()` = allocation bytes / size_of::<T>(), so
                // `len..new_len` lies inside our owned allocation (Shared never gets here:
                // `assert_mutable` panicked, and Empty has capacity 0).
                unsafe {
                    core::ptr::write_bytes(self.ptr.as_ptr().add(self.len), 0, new_len - self.len);
                }
            }
            self.len = new_len;
            return;
        }

        // Need to reallocate
        let size_bytes = Self::bytes_for(new_len);
        let (new_backing, new_ptr) =
            Self::alloc_backing(size_bytes, "Failed to reallocate AlignedVec");

        // Copy old data
        if self.len > 0 {
            // SAFETY: the old region holds `len` initialized elements, the new allocation holds
            // `new_len > current_cap >= len` elements, and the two allocations are distinct.
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }

        // Replace the backing store (the old one is dropped and released)
        self.backing = new_backing;
        self.ptr = new_ptr;
        self.len = new_len;
    }

    /// Copy `src` into the first `src.len()` elements; the rest are untouched.
    ///
    /// # Panics
    /// If `src` is longer than `self`, or `self` is shared (read-only).
    pub fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        self.assert_mutable();
        assert!(src.len() <= self.len, "source slice too long");
        // SAFETY: `src.len() <= self.len` was just asserted, `self` is owned (not Shared) so it is
        // writable, and `&mut self` guarantees `src` (borrowed separately) cannot overlap it.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr(), src.len());
        }
    }

    /// Returns a slice of the entire AlignedVec
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Returns a mutable slice of the entire AlignedVec
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Returns an iterator over the elements
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the elements
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is non-null and aligned for `T` (Empty: dangling with len 0; Heap:
        // 64-byte-aligned block of >= len*size_of::<T>() bytes; Owned: page aligned allocation of
        // >= len*size_of::<T>() bytes, zeroed or written; Shared: bounds and alignment asserted
        // in `from_shared`), and the backing outlives `&self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.assert_mutable();
        // SAFETY: as in `deref`; additionally the backing is Heap/Owned/Empty (checked above) and
        // `&mut self` gives exclusive access.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

// Drop is handled automatically by `Backing`: an owned allocation frees itself,
// a shared window just releases its Arc handle on the mapping.

impl<T: Pod> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self {
                backing: Backing::Empty,
                ptr: NonNull::dangling(),
                len: 0,
            };
        }

        // A clone always owns its bytes, even when the source borrows shared
        // memory: the copy is independently mutable.
        let size_bytes = Self::bytes_for(self.len);
        let (new_backing, new_ptr) =
            Self::alloc_backing(size_bytes, "Failed to allocate AlignedVec clone");

        // `T: Pod` is `Copy`, so cloning is a plain byte copy.
        // SAFETY: the source holds `len` initialized elements and the new allocation has room for
        // `len` (distinct allocations, both aligned for `T`).
        unsafe {
            core::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
        }

        Self {
            backing: new_backing,
            ptr: new_ptr,
            len: self.len,
        }
    }
}

// SAFETY: AlignedVec uniquely owns its allocation (or holds an immutable Arc-kept shared window),
// so it behaves like `Vec<T>`: sending it sends the `T`s. `NonNull` is merely not auto-Send.
unsafe impl<T: Send> Send for AlignedVec<T> {}
// SAFETY: `&AlignedVec` only exposes `&[T]` (all mutation needs `&mut self`), like `Vec<T>`.
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<'a, T> IntoIterator for &'a AlignedVec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AlignedVec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_aligned64<T>(v: &AlignedVec<T>) -> bool {
        (v.as_ptr() as usize).is_multiple_of(64)
    }

    /// Element counts below and above the 1 MiB large-page threshold for `T`.
    fn small_and_large<T>() -> [usize; 2] {
        let sz = core::mem::size_of::<T>();
        [100, SMALL_ALLOC_LIMIT / sz + 3]
    }

    fn check_zeroed<T: Pod + PartialEq + core::fmt::Debug + Default>() {
        for len in small_and_large::<T>() {
            let v = AlignedVec::<T>::new_zeroed(len);
            assert_eq!(v.len(), len);
            assert_eq!(v.as_slice().len(), len);
            assert!(!v.is_empty());
            assert!(is_aligned64(&v), "{} not 64-byte aligned", len);
            assert!(v.capacity() >= len);
            assert!(!v.is_shared());
            assert!(v.iter().all(|x| *x == T::default()));
        }
    }

    #[test]
    fn new_zeroed_all_element_types() {
        check_zeroed::<i8>();
        check_zeroed::<u8>();
        check_zeroed::<i16>();
        check_zeroed::<u16>();
        check_zeroed::<i32>();
        check_zeroed::<u32>();
        check_zeroed::<i64>();
        check_zeroed::<u64>();
    }

    #[test]
    fn small_uses_heap_and_large_uses_large_pages() {
        let small = AlignedVec::<u8>::new_zeroed(SMALL_ALLOC_LIMIT - 1);
        assert!(matches!(small.backing, Backing::Heap(_)));
        let large = AlignedVec::<u8>::new_zeroed(SMALL_ALLOC_LIMIT);
        assert!(matches!(large.backing, Backing::Owned(_)));
        assert!(is_aligned64(&large));
    }

    #[test]
    fn zero_length() {
        let mut v = AlignedVec::<i32>::new_zeroed(0);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 0);
        assert!(v.as_slice().is_empty());
        assert!(v.as_mut_slice().is_empty());
        assert!(v.clone().is_empty());
        assert_eq!(v.iter().count(), 0);
        // Growing from empty works and zero-fills.
        v.resize(10);
        assert_eq!(v.as_slice(), &[0; 10]);
        v.resize(0);
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn resize_growth_preserves_contents_and_zero_fills() {
        for start in small_and_large::<u16>() {
            let mut v = AlignedVec::<u16>::new_zeroed(start);
            for (i, x) in v.iter_mut().enumerate() {
                *x = (i % 65521) as u16 + 1;
            }
            // Within capacity (heap blocks are exact; large ones are rounded up) and beyond it.
            for new_len in [start + 1, start * 2 + 7, start * 5] {
                let old = v.as_slice()[..start].to_vec();
                v.resize(new_len);
                assert_eq!(v.len(), new_len);
                assert!(v.capacity() >= new_len);
                assert!(is_aligned64(&v));
                assert_eq!(&v.as_slice()[..start], &old[..]);
                assert!(v.as_slice()[start..].iter().all(|&x| x == 0));
            }
        }
    }

    #[test]
    fn resize_shrink_then_regrow_rezeroes() {
        let mut v = AlignedVec::<u32>::new_zeroed(64);
        v.as_mut_slice().fill(0xDEAD_BEEF);
        v.resize(10);
        assert_eq!(v.len(), 10);
        assert!(v.as_slice().iter().all(|&x| x == 0xDEAD_BEEF));
        v.resize(64); // within capacity: stale data must not reappear
        assert!(v.as_slice()[..10].iter().all(|&x| x == 0xDEAD_BEEF));
        assert!(v.as_slice()[10..].iter().all(|&x| x == 0));
    }

    #[test]
    fn clone_is_equal_and_independent() {
        for len in small_and_large::<i8>() {
            let mut a = AlignedVec::<i8>::new_zeroed(len);
            for (i, x) in a.iter_mut().enumerate() {
                *x = (i as i8).wrapping_mul(7);
            }
            let mut b = a.clone();
            assert_eq!(a.as_slice(), b.as_slice());
            assert_ne!(a.as_ptr(), b.as_ptr());
            assert!(is_aligned64(&b));
            b.as_mut_slice()[0] = 99;
            b.as_mut_slice()[len - 1] = -99;
            assert_ne!(a.as_slice()[0], 99);
            assert_ne!(a.as_slice()[len - 1], -99);
            a.as_mut_slice().fill(1);
            assert_eq!(b.as_slice()[0], 99);
        }
    }

    #[test]
    fn copy_from_slice_and_iteration() {
        let mut v = AlignedVec::<i32>::new_zeroed(8);
        v.copy_from_slice(&[1, 2, 3]);
        assert_eq!(v.as_slice(), &[1, 2, 3, 0, 0, 0, 0, 0]);
        let mut sum = 0;
        for x in &v {
            sum += *x;
        }
        for x in &mut v {
            *x += 1;
        }
        assert_eq!(sum, 6);
        assert_eq!(v.as_ptr(), v.as_mut_ptr() as *const i32);
        assert_eq!(v[0], 2);
    }

    #[test]
    #[should_panic(expected = "source slice too long")]
    fn copy_from_slice_too_long_panics() {
        AlignedVec::<u8>::new_zeroed(2).copy_from_slice(&[1, 2, 3]);
    }

    /// A shared region of `len` bytes whose content byte `i` is `(i * 3) as u8`. The hash is
    /// unique per call so concurrent tests and earlier runs never share a region. The first
    /// element removes the region's per-name directory; bind it so that it outlives the region.
    fn shared_region(len: usize) -> (impl Sized, Arc<SystemWideSharedConstant>) {
        use core::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let hash = nanos
            ^ (u64::from(std::process::id()) << 40)
            ^ COUNTER
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let region = Arc::new(SystemWideSharedConstant::new(hash, len, 0xA11C, |buf| {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i * 3) as u8;
            }
        }));
        #[cfg(unix)]
        let cleanup = region.dir_cleanup();
        #[cfg(not(unix))]
        let cleanup = ();
        (cleanup, region)
    }

    #[test]
    fn from_shared_is_a_read_only_window() {
        let (_cleanup, region) = shared_region(4096);
        let bytes = region.as_slice().expect("mapping");
        let v = AlignedVec::<u8>::from_shared(&region, 16, 100);
        assert!(v.is_shared());
        assert_eq!(v.len(), 100);
        assert_eq!(v.capacity(), 100);
        assert_eq!(v.as_slice(), &bytes[16..116]);
        assert_eq!(v.as_ptr(), bytes[16..].as_ptr());

        let w = AlignedVec::<u32>::from_shared(&region, 64, 8);
        let expect: Vec<u32> = (0..8)
            .map(|i| u32::from_ne_bytes(bytes[64 + 4 * i..68 + 4 * i].try_into().unwrap()))
            .collect();
        assert_eq!(w.as_slice(), &expect[..]);

        // The window holds a handle on the region.
        assert_eq!(Arc::strong_count(&region), 3);
        drop(v);
        drop(w);
        assert_eq!(Arc::strong_count(&region), 1);

        // A clone owns private, mutable memory with the same contents.
        let v = AlignedVec::<u8>::from_shared(&region, 0, 4096);
        let mut c = v.clone();
        assert!(!c.is_shared());
        assert_eq!(c.as_slice(), v.as_slice());
        c.as_mut_slice()[5] ^= 0xFF;
        assert_ne!(c.as_slice()[5], v.as_slice()[5]);
        // Zero-length window.
        assert!(AlignedVec::<u8>::from_shared(&region, 0, 0).is_empty());
        // Exactly at the end is allowed.
        assert_eq!(AlignedVec::<u8>::from_shared(&region, 4090, 6).len(), 6);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn from_shared_out_of_bounds_panics() {
        let (_cleanup, region) = shared_region(256);
        let _ = AlignedVec::<u8>::from_shared(&region, 250, 7);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn from_shared_offset_overflow_panics() {
        let (_cleanup, region) = shared_region(256);
        let _ = AlignedVec::<u8>::from_shared(&region, usize::MAX, 2);
    }

    #[test]
    #[should_panic(expected = "misaligned")]
    fn from_shared_misaligned_panics() {
        let (_cleanup, region) = shared_region(256);
        let _ = AlignedVec::<u32>::from_shared(&region, 2, 4);
    }

    #[test]
    #[should_panic(expected = "shared memory")]
    fn shared_deref_mut_panics() {
        let (_cleanup, region) = shared_region(256);
        let mut v = AlignedVec::<u8>::from_shared(&region, 0, 16);
        v.as_mut_slice()[0] = 1;
    }

    #[test]
    #[should_panic(expected = "shared memory")]
    fn shared_resize_panics() {
        let (_cleanup, region) = shared_region(256);
        let mut v = AlignedVec::<u8>::from_shared(&region, 0, 16);
        v.resize(8);
    }

    #[test]
    #[should_panic(expected = "shared memory")]
    fn shared_copy_from_slice_panics() {
        let (_cleanup, region) = shared_region(256);
        let mut v = AlignedVec::<u8>::from_shared(&region, 0, 16);
        v.copy_from_slice(&[1]);
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn length_overflow_panics() {
        let _ = AlignedVec::<u64>::new_zeroed(usize::MAX / 4);
    }
}
