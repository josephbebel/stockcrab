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

//! Neural network implementation for NNUE evaluation
//!
//! This module implements the main NNUE network structure, including
//! support for loading embedded and external network files, evaluation,
//! and network management.

use core::alloc::Layout;
use core::ptr;
use std::alloc::{alloc_zeroed, dealloc};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use crate::bitboard::{FILE_A_BB, FILE_B_BB, FILE_C_BB, FILE_D_BB, popcount};
use crate::position::Position;
use crate::shm::{AllocationStatus, SystemWideSharedConstant};
use crate::types::{Bitboard, Color, PieceType, Value};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
use super::layers::NnzInfo;
use super::nnue_accumulator::{AccumulatorCaches, AccumulatorStack};
use super::nnue_architecture::{L1, LAYER_STACKS, NetworkArchitecture, RAW_LAYER_STACKS};
use super::nnue_common::{CACHE_LINE_SIZE, OUTPUT_SCALE, TransformedFeatureType};
use super::nnue_feature_transformer::FeatureTransformer;
use super::nnue_misc::{EvalFile, NnueEvalTrace};

// NNUE file version constant (SF18)
const VERSION: u32 = 0x6A448AFA;

// Default network name matching C++ `EvalFileDefaultName` in evaluate.h.
// Single source of truth lives in `evaluate` (mirrors evaluate.h); re-exported here
// so this module and its callers can never drift to different nets.
pub use crate::evaluate::EVAL_FILE_DEFAULT_NAME;

// No network is embedded in this build: the placeholder below makes `load_internal` fail so
// the net is always loaded from disk. Kept to preserve upstream's load order (embedded,
// working dir, root dir).
mod embedded {
    pub static EMBEDDED_NNUE_DATA: &[u8] = &[0x0];
}

// Helper functions for endian conversion
fn read_little_endian_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_little_endian_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_little_endian_u16<R: Read>(reader: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

/// Tag of the layer-stack table that follows the network header (C++ `LayerStackTableHash`).
const LAYER_STACK_TABLE_HASH: u32 = 0x5A7A_B1E5;

/// Read little-endian i16 values (SFNNv18: the transformer biases are stored uncompressed).
pub(super) fn read_little_endian_i16<R: Read>(reader: &mut R, out: &mut [i16]) -> io::Result<()> {
    let mut buf = [0u8; 8192];
    for chunk in out.chunks_mut(buf.len() / 2) {
        let bytes = &mut buf[..chunk.len() * 2];
        reader.read_exact(bytes)?;
        for (v, b) in chunk.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *v = i16::from_le_bytes(*b);
        }
    }
    Ok(())
}

/// Write little-endian i16 values.
pub(super) fn write_little_endian_i16<W: Write>(writer: &mut W, values: &[i16]) -> io::Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    writer.write_all(&bytes)
}

/// Read raw i8 values (little-endian/direct bytes) - used for threat weights in SFNNv10
pub(super) fn read_little_endian_i8<R: Read>(reader: &mut R, out: &mut [i8]) -> io::Result<()> {
    // Read directly as bytes (i8 is just signed interpretation of u8)
    reader.read_exact(bytemuck::cast_slice_mut(out))
}

/// Write raw i8 values (little-endian/direct bytes) - used for threat weights in SFNNv10
pub(super) fn write_little_endian_i8<W: Write>(writer: &mut W, values: &[i8]) -> io::Result<()> {
    // Write directly as bytes (i8 is just signed interpretation of u8)
    writer.write_all(bytemuck::cast_slice(values))
}

/// Owning pointer to one heap-allocated `T` (upstream's `LargePagePtr`; large pages are not
/// requested here, the shared weights live in shm instead).
pub struct LargePagePtr<T> {
    ptr: *mut T,
    layout: Layout,
}

impl<T> LargePagePtr<T> {
    /// Create a new large page allocated object
    pub fn new(value: T) -> Self {
        let layout = Layout::new::<T>();
        let ptr = if layout.size() == 0 {
            // Zero-sized `T`: nothing to allocate (`alloc_zeroed` forbids zero-size layouts).
            let ptr = core::ptr::NonNull::<T>::dangling().as_ptr();
            // SAFETY: writing a ZST through a dangling, aligned pointer is valid.
            unsafe { ptr::write(ptr, value) };
            ptr
        } else {
            // SAFETY: `layout` has non-zero size (checked above), so `alloc_zeroed` is allowed;
            // the result is checked for null and then initialized by `ptr::write`, which is valid
            // for freshly allocated, aligned memory.
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                if ptr.is_null() {
                    panic!("Failed to allocate memory for LargePagePtr");
                }
                ptr::write(ptr, value);
                ptr
            }
        };

        Self { ptr, layout }
    }
}

impl<T> Drop for LargePagePtr<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` holds one initialized `T` (written in `new`) and, when the layout is
        // non-zero-sized, came from `alloc_zeroed(self.layout)`; zero-sized layouts were never
        // allocated and are not freed. Drop runs once, so everything is dropped/freed once.
        unsafe {
            ptr::drop_in_place(self.ptr);
            if self.layout.size() != 0 {
                dealloc(self.ptr as *mut u8, self.layout);
            }
        }
    }
}

impl<T> core::ops::Deref for LargePagePtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` is non-null, aligned and points to an initialized `T` owned by `self`.
        unsafe { &*self.ptr }
    }
}

impl<T> core::ops::DerefMut for LargePagePtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: as in `deref`; `&mut self` guarantees exclusive access.
        unsafe { &mut *self.ptr }
    }
}

/// Aligned pointer for network layers
pub struct AlignedPtr<T> {
    ptr: *mut T,
    layout: Layout,
    len: usize,
}

impl<T: Default> AlignedPtr<T> {
    /// Create a cache-line aligned array of `len` default-constructed elements.
    ///
    /// Every element is genuinely constructed with `ptr::write` (not merely
    /// zeroed memory typed as `T`), so assigning to an element drops a live `T`
    /// and `Drop` can run the element destructors that free each element's
    /// weight allocations.
    pub fn new_array(len: usize) -> Self {
        let layout = Layout::array::<T>(len)
            .unwrap()
            .align_to(CACHE_LINE_SIZE)
            .unwrap();

        let ptr = if layout.size() == 0 {
            // `len == 0` or zero-sized `T`: `alloc_zeroed` forbids zero-size layouts, so use a
            // dangling aligned pointer; ZST elements are still constructed (and later dropped).
            let ptr = core::ptr::NonNull::<T>::dangling().as_ptr();
            for i in 0..len {
                // SAFETY: `T` is zero-sized here (len > 0 implies size 0), so every slot is the
                // dangling aligned address and writing a ZST there is valid.
                unsafe { ptr::write(ptr.add(i), T::default()) };
            }
            ptr
        } else {
            // SAFETY: `layout` has non-zero size (checked above) as `alloc_zeroed` requires; the
            // result is null-checked, and each of the `len` slots is initialized with
            // `ptr::write` before `Self` can be observed.
            unsafe {
                let ptr = alloc_zeroed(layout) as *mut T;
                if ptr.is_null() {
                    panic!("Failed to allocate aligned memory");
                }
                for i in 0..len {
                    ptr::write(ptr.add(i), T::default());
                }
                ptr
            }
        };

        Self { ptr, layout, len }
    }
}

impl<T> Drop for AlignedPtr<T> {
    fn drop(&mut self) {
        // SAFETY: all `len` slots were initialized in `new_array` and are dropped once; a non-zero
        // sized block was allocated with `self.layout` (zero-sized ones were never allocated).
        unsafe {
            for i in 0..self.len {
                ptr::drop_in_place(self.ptr.add(i));
            }
            if self.layout.size() != 0 {
                dealloc(self.ptr as *mut u8, self.layout);
            }
        }
    }
}

impl<T> core::ops::Index<usize> for AlignedPtr<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len);
        // SAFETY: `index < len` (asserted above), so the slot is inside the allocation and
        // initialized.
        unsafe { &*self.ptr.add(index) }
    }
}

impl<T> core::ops::IndexMut<usize> for AlignedPtr<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len);
        // SAFETY: as in `index`; `&mut self` gives exclusive access.
        unsafe { &mut *self.ptr.add(index) }
    }
}

// SAFETY: LargePagePtr and AlignedPtr uniquely own their allocation (like Box/Vec); sending one
// sends the owned `T`s, so `T: Send` suffices; mutation requires `&mut self`.
unsafe impl<T: Send> Send for LargePagePtr<T> {}
// SAFETY: see the `Send` impl above; `&self` only hands out `&T`, so `T: Sync` suffices.
unsafe impl<T: Sync> Sync for LargePagePtr<T> {}

// SAFETY: see the LargePagePtr impls above; same ownership model.
unsafe impl<T: Send> Send for AlignedPtr<T> {}
// SAFETY: see above; `&self` only hands out `&T`.
unsafe impl<T: Sync> Sync for AlignedPtr<T> {}

/// Single unified NNUE network.
///
/// One non-generic network: a single `FeatureTransformer` and `stored_layer_stacks` copies of
/// `NetworkArchitecture`. Each of the `RAW_LAYER_STACKS` configurations (see `layer_stack`) uses
/// one of the stored stacks, as listed in the file's layer-stack table.
pub struct Network {
    /// Feature transformer (input converter).
    feature_transformer: Option<LargePagePtr<FeatureTransformer>>,

    /// Evaluation network layers (`stored_layer_stacks` of them).
    network: Option<AlignedPtr<NetworkArchitecture>>,

    /// For each raw layer-stack configuration, the configuration whose stack it shares.
    layer_stack_fallback: [u16; RAW_LAYER_STACKS],
    /// For each raw layer-stack configuration, the index of its stack in `network`.
    layer_stack_index: [u16; RAW_LAYER_STACKS],
    /// Number of layer stacks stored in the network file.
    stored_layer_stacks: usize,

    /// Evaluation file information.
    eval_file: EvalFile,

    /// Hash value for network structure verification.
    hash: u32,

    /// Shared region backing the feature transformer's weights, when the
    /// weights were successfully published system-wide.
    ///
    /// Held here so the mapping outlives the transformer that borrows it. It is
    /// also what `Engine::verify_network` reports, mirroring C++'s
    /// "Network replica N: ..." lines.
    shared_weights: Option<Arc<SystemWideSharedConstant>>,
}

impl Clone for Network {
    fn clone(&self) -> Self {
        let feature_transformer = self.feature_transformer.as_ref().map(|ft| {
            let copied = match &self.shared_weights {
                Some(region) => FeatureTransformer::from_shared_region(region),
                None => (**ft).clone(),
            };
            LargePagePtr::new(copied)
        });
        let network = self.network.as_ref().map(|layers| {
            let mut copied = AlignedPtr::new_array(self.stored_layer_stacks);
            for i in 0..self.stored_layer_stacks {
                copied[i] = layers[i].clone();
            }
            copied
        });
        Self {
            feature_transformer,
            network,
            layer_stack_fallback: self.layer_stack_fallback,
            layer_stack_index: self.layer_stack_index,
            stored_layer_stacks: self.stored_layer_stacks,
            eval_file: self.eval_file.clone(),
            hash: self.hash,
            shared_weights: self.shared_weights.clone(),
        }
    }
}

impl crate::engine::NumaReplicable for Network {
    fn replicate(&self, numa_node: usize) -> Self {
        let mut replica = self.clone();
        if numa_node != 0
            && let Some(ft) = self.feature_transformer.as_ref()
        {
            let region = Arc::new(SystemWideSharedConstant::new(
                ft.content_hash(),
                FeatureTransformer::shared_image_len(),
                numa_node as u64,
                |bytes| ft.write_shared_image(bytes),
            ));
            if region.as_slice().is_some() {
                replica.feature_transformer = Some(LargePagePtr::new(
                    FeatureTransformer::from_shared_region(&region),
                ));
                replica.shared_weights = Some(region);
            }
        }
        replica
    }
}

impl Network {
    /// Create a new, unloaded network with the SF18 default eval file.
    pub fn new() -> Self {
        let hash = FeatureTransformer::get_hash_value()
            ^ NetworkArchitecture::get_hash_value()
            ^ LAYER_STACK_TABLE_HASH;

        let eval_file = EvalFile {
            default_name: EVAL_FILE_DEFAULT_NAME.to_string(),
            current: String::new(),
            net_description: String::new(),
        };

        // As in C++, a new network owns no weights: a load path allocates them. Building a
        // zeroed transformer here would cost ~107 MB per `Network::new` call, dirtied page by
        // page on platforms without lazily-zeroed anonymous mappings (macOS).
        Self {
            feature_transformer: None,
            network: None,
            layer_stack_fallback: [0; RAW_LAYER_STACKS],
            layer_stack_index: [0; RAW_LAYER_STACKS],
            stored_layer_stacks: 0,
            eval_file,
            hash,
            shared_weights: None,
        }
    }

    /// Initialize the network structures
    fn initialize(&mut self) {
        self.feature_transformer = Some(LargePagePtr::new(FeatureTransformer::default()));
        self.shared_weights = None;
        // The layer stacks are allocated by `read_layers`, once the layer-stack table has
        // said how many the file stores.
        self.network = None;
    }

    /// Load network from file or embedded data
    pub fn load(&mut self, root_directory: &str, evalfile_path: String) -> io::Result<()> {
        let mut path = evalfile_path;
        if path.is_empty() {
            path = self.eval_file.default_name.clone();
        }

        // Upstream treats an already loaded name as a successful no-op.
        if self.eval_file.current == path {
            return Ok(());
        }

        // Parse into temporary owned state. Publish weights and identity together
        // only after every component, including the end of file, is validated.
        let mut candidate = Self::new();

        // Try different directories
        let dirs = vec![
            "<internal>".to_string(),
            String::new(),
            root_directory.to_string(),
        ];

        let mut last_error = io::Error::new(io::ErrorKind::NotFound, "Failed to load network");
        for dir in dirs {
            let result = if dir == "<internal>" {
                if path != self.eval_file.default_name {
                    continue;
                }
                candidate.load_internal()
            } else {
                candidate.load_user_net(&dir, &path)
            };
            match result {
                Ok(()) => {
                    *self = candidate;
                    return Ok(());
                }
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }

    /// Load user network from file
    fn load_user_net(&mut self, dir: &str, evalfile_path: &str) -> io::Result<()> {
        let path = if dir.is_empty() {
            PathBuf::from(evalfile_path)
        } else {
            PathBuf::from(dir).join(evalfile_path)
        };

        let file = File::open(&path)?;

        #[cfg(unix)]
        if let Some(description) = self.load_into_shared(&file)? {
            self.eval_file.current = evalfile_path.to_string();
            self.eval_file.net_description = description;
            return Ok(());
        }

        #[cfg(unix)]
        {
            use std::io::{Seek, SeekFrom};
            (&file).seek(SeekFrom::Start(0))?;
        }

        let mut reader = BufReader::with_capacity(1 << 20, file);

        if let Some(description) = self.load_from_stream(&mut reader)? {
            self.eval_file.current = evalfile_path.to_string();
            self.eval_file.net_description = description;
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid network file",
            ))
        }
    }

    /// Load internal embedded network
    fn load_internal(&mut self) -> io::Result<()> {
        let data = embedded::EMBEDDED_NNUE_DATA;

        // Check if embedded data is just placeholder (single byte with value 0)
        if data.len() <= 1 {
            // Return error to force loading from external files instead of using default networks
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Embedded network data not available (placeholder only)",
            ));
        }

        let mut cursor = Cursor::new(data);

        if let Some(description) = self.load_from_stream(&mut cursor)? {
            self.eval_file.current = self.eval_file.default_name.clone();
            self.eval_file.net_description = description;
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid embedded network",
            ))
        }
    }

    /// Load the network from `file`, parsing the feature transformer straight
    /// into its shared region.
    ///
    /// The generic path (`load_from_stream`) has to parse into a private copy
    /// first, because the region is addressed by the hash of the parsed weights,
    /// and then copy it across: two ~110 MiB copies live at the peak. Here the
    /// region is addressed by a hash of the *file* instead (same file, same
    /// executable => same parsed weights), so the creator parses directly into
    /// the region and a process that finds it already published skips the
    /// transformer entirely.
    ///
    /// A trailer after the weight image records where the layer stacks start in
    /// the file; it is written last and stays zero if parsing failed, so a failed
    /// parse is never mistaken for a usable region by a later process.
    ///
    /// Returns `Ok(None)` when this path does not apply (header mismatch, no
    /// shared or local region available, or a region left behind by a failed
    /// parse) so the caller can fall back to the generic path, which reports the
    /// error exactly as before.
    #[cfg(unix)]
    fn load_into_shared(&mut self, file: &File) -> io::Result<Option<String>> {
        use std::io::{Seek, SeekFrom};

        const TRAILER: usize = CACHE_LINE_SIZE;
        let mut file = file;

        // Validate the headers exactly as `read_parameters` does.
        let (description, ft_start) = {
            let mut reader = BufReader::new(file);
            let (hash_value, description) = self.read_header(&mut reader)?;
            if hash_value != self.hash || !self.read_layer_stack_table(&mut reader)? {
                return Ok(None);
            }
            if read_little_endian_u32(&mut reader)? != FeatureTransformer::get_hash_value() {
                return Ok(None);
            }
            let ft_start = reader.stream_position()?;
            (description, ft_start)
        };

        // Content key: hash of the file, chunked so it streams in constant space.
        file.seek(SeekFrom::Start(0))?;
        let mut key: u64 = 0;
        let mut chunk = vec![0u8; 1 << 20];
        loop {
            let mut n = 0;
            while n < chunk.len() {
                match file.read(&mut chunk[n..])? {
                    0 => break,
                    m => n += m,
                }
            }
            if n == 0 {
                break;
            }
            let x = crate::misc::hash_bytes(&chunk[..n]);
            key ^= x
                .wrapping_add(0x9e37_79b9)
                .wrapping_add(key << 6)
                .wrapping_add(key >> 2);
            if n < chunk.len() {
                break;
            }
        }
        drop(chunk);

        // The region holds weights permuted for this build's SIMD layout, so the key also
        // identifies that layout.
        key ^= FeatureTransformer::image_layout_key();

        let image_len = FeatureTransformer::shared_image_len();
        let region = Arc::new(SystemWideSharedConstant::new(
            key,
            image_len + TRAILER,
            0,
            |buf| {
                let parsed = (|| -> io::Result<u64> {
                    let mut f = file;
                    f.seek(SeekFrom::Start(ft_start))?;
                    let mut reader = BufReader::with_capacity(1 << 20, f);
                    FeatureTransformer::read_parameters_into_image(
                        &mut reader,
                        &mut buf[..image_len],
                    )?;
                    reader.stream_position()
                })();
                if let Ok(end) = parsed {
                    buf[image_len..image_len + 8].copy_from_slice(&end.to_le_bytes());
                }
            },
        ));

        let layers_start = match region.as_slice() {
            Some(bytes) if bytes.len() == image_len + TRAILER => {
                u64::from_le_bytes(bytes[image_len..image_len + 8].try_into().expect("8 bytes"))
            }
            _ => 0,
        };
        if layers_start == 0 {
            return Ok(None);
        }

        self.feature_transformer = Some(LargePagePtr::new(FeatureTransformer::from_shared_region(
            &region,
        )));
        self.shared_weights = Some(region);

        file.seek(SeekFrom::Start(layers_start))?;
        let mut reader = BufReader::new(file);
        self.read_layers(&mut reader)?;
        Ok(Some(description))
    }

    /// Load network from a stream
    fn load_from_stream<R: Read>(&mut self, reader: &mut R) -> io::Result<Option<String>> {
        self.initialize();

        let mut description = String::new();
        match self.read_parameters(reader, &mut description) {
            Ok(true) => {
                self.publish_shared_weights();
                Ok(Some(description))
            }
            Ok(false) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Move the freshly-loaded feature transformer weights into shared memory.
    ///
    /// The weights are parsed into a private transformer first (there is no way
    /// to know their hash before reading them), then published; if another
    /// process already published byte-identical weights we attach to its region
    /// instead and the private copy is dropped. Either way the transformer is
    /// rebuilt to point at the region, so the private copy does not linger.
    ///
    /// This is best-effort: if the region cannot be created the private
    /// transformer is simply kept, which is exactly the C++ fallback behaviour.
    fn publish_shared_weights(&mut self) {
        let Some(ft) = self.feature_transformer.as_ref() else {
            return;
        };

        let content_hash = ft.content_hash();
        let len = FeatureTransformer::shared_image_len();

        // `init` runs only if we are the creator.
        let region = Arc::new(SystemWideSharedConstant::new(content_hash, len, 0, |buf| {
            ft.write_shared_image(buf);
        }));

        if region.status() != AllocationStatus::SharedMemory {
            // Local fallback: publishing into a private allocation would just be
            // a second copy of what we already have, so keep the original.
            return;
        }

        let shared_ft = FeatureTransformer::from_shared_region(&region);
        self.feature_transformer = Some(LargePagePtr::new(shared_ft));
        self.shared_weights = Some(region);
    }

    /// Allocation status of the shared weights, for `verify_network` reporting.
    pub fn shared_weights_status(&self) -> AllocationStatus {
        match &self.shared_weights {
            Some(r) => r.status(),
            None => AllocationStatus::LocalMemory,
        }
    }

    /// Whether this process created the shared weight region or attached to one
    /// published by another process.
    ///
    /// Diagnostics and tests only: attaching is the whole point of the feature,
    /// so this is how a test distinguishes real cross-process deduplication from
    /// two processes that each happened to allocate their own region.
    pub fn shared_weights_is_creator(&self) -> Option<bool> {
        self.shared_weights.as_ref().map(|r| r.is_creator())
    }

    /// Explanatory note to accompany [`Self::shared_weights_status`].
    pub fn shared_weights_message(&self) -> Option<String> {
        match &self.shared_weights {
            Some(r) => r.error_message(),
            None => Some("Shared memory not available. Local allocation fallback.".to_string()),
        }
    }

    /// Read network header
    fn read_header<R: Read>(&self, reader: &mut R) -> io::Result<(u32, String)> {
        let version = read_little_endian_u32(reader)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid version",
            ));
        }

        let hash_value = read_little_endian_u32(reader)?;
        let mut remaining = read_little_endian_u32(reader)? as usize;
        let mut description = Vec::new();
        let mut buffer = [0u8; 4096];

        while remaining > 0 {
            let want = remaining.min(buffer.len());
            reader.read_exact(&mut buffer[..want])?;
            description.extend_from_slice(&buffer[..want]);
            remaining -= want;
        }

        Ok((
            hash_value,
            String::from_utf8_lossy(&description).to_string(),
        ))
    }

    /// Write network header
    fn write_header<W: Write>(
        &self,
        writer: &mut W,
        hash_value: u32,
        description: &str,
    ) -> io::Result<()> {
        write_little_endian_u32(writer, VERSION)?;
        write_little_endian_u32(writer, hash_value)?;
        write_little_endian_u32(writer, description.len() as u32)?;
        writer.write_all(description.as_bytes())?;
        Ok(())
    }

    /// Read network parameters
    fn read_parameters<R: Read>(
        &mut self,
        reader: &mut R,
        description: &mut String,
    ) -> io::Result<bool> {
        let (hash_value, desc) = self.read_header(reader)?;
        *description = desc;

        if hash_value != self.hash || !self.read_layer_stack_table(reader)? {
            return Ok(false);
        }

        // Read feature transformer
        if let Some(ref mut transformer) = self.feature_transformer {
            // First read the transformer hash header
            let transformer_hash = read_little_endian_u32(reader)?;
            let expected_transformer_hash = FeatureTransformer::get_hash_value();

            if transformer_hash != expected_transformer_hash {
                return Ok(false);
            }

            // Read the transformer parameters
            transformer.read_parameters(reader)?;
        }

        self.read_layers(reader)?;

        Ok(true)
    }

    /// Read the layer-stack table that follows the header: its tag, the number of raw
    /// configurations, the number of stored stacks and, per raw configuration, the
    /// configuration it falls back to. A configuration that falls back to itself owns a stored
    /// stack (numbered in order); every other one must fall back to such a configuration.
    /// Returns `Ok(false)` for a table this engine cannot use, as C++ `read_parameters` does.
    fn read_layer_stack_table<R: Read>(&mut self, reader: &mut R) -> io::Result<bool> {
        if read_little_endian_u32(reader)? != LAYER_STACK_TABLE_HASH
            || read_little_endian_u32(reader)? != RAW_LAYER_STACKS as u32
        {
            return Ok(false);
        }
        self.stored_layer_stacks = read_little_endian_u32(reader)? as usize;
        if self.stored_layer_stacks == 0 || self.stored_layer_stacks > LAYER_STACKS {
            return Ok(false);
        }
        for f in self.layer_stack_fallback.iter_mut() {
            *f = read_little_endian_u16(reader)?;
            if usize::from(*f) >= RAW_LAYER_STACKS {
                return Ok(false);
            }
        }
        let mut stored = 0usize;
        for s in 0..RAW_LAYER_STACKS {
            if usize::from(self.layer_stack_fallback[s]) == s {
                self.layer_stack_index[s] = stored as u16;
                stored += 1;
            }
        }
        if stored != self.stored_layer_stacks {
            return Ok(false);
        }
        for s in 0..RAW_LAYER_STACKS {
            let f = usize::from(self.layer_stack_fallback[s]);
            if usize::from(self.layer_stack_fallback[f]) != f {
                return Ok(false);
            }
            self.layer_stack_index[s] = self.layer_stack_index[f];
        }
        Ok(true)
    }

    /// Read the `stored_layer_stacks` layer stacks that follow the feature transformer.
    fn read_layers<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        let network = self
            .network
            .insert(AlignedPtr::new_array(self.stored_layer_stacks));
        for i in 0..self.stored_layer_stacks {
            if read_little_endian_u32(reader)? != NetworkArchitecture::get_hash_value() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid layer hash",
                ));
            }
            network[i].read_parameters(reader)?;
        }
        // The stored stacks end the file.
        if reader.read(&mut [0u8; 1])? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Trailing network data",
            ));
        }
        Ok(())
    }

    /// Write network parameters
    fn write_parameters<W: Write>(&self, writer: &mut W, description: &str) -> io::Result<bool> {
        self.write_header(writer, self.hash, description)?;
        write_little_endian_u32(writer, LAYER_STACK_TABLE_HASH)?;
        write_little_endian_u32(writer, RAW_LAYER_STACKS as u32)?;
        write_little_endian_u32(writer, self.stored_layer_stacks as u32)?;
        for f in self.layer_stack_fallback {
            writer.write_all(&f.to_le_bytes())?;
        }

        // Write feature transformer.
        // C++ Detail::write_parameters (network.cpp) emits the component's
        // hash value before the component itself; read_parameters consumes it again.
        if let Some(ref transformer) = self.feature_transformer {
            write_little_endian_u32(writer, FeatureTransformer::get_hash_value())?;
            if let Err(e) = transformer.write_parameters(writer) {
                eprintln!("NNUE save error: Feature transformer write failed: {}", e);
                return Ok(false);
            }
        }

        // Write network layers
        if let Some(ref network) = self.network {
            for i in 0..self.stored_layer_stacks {
                write_little_endian_u32(writer, NetworkArchitecture::get_hash_value())?;
                if let Err(e) = network[i].write_parameters(writer) {
                    eprintln!("NNUE save error: Network layer {} write failed: {}", i, e);
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Save network to file
    pub fn save(&self, filename: Option<String>) -> io::Result<bool> {
        // `current` is empty until a load succeeds (C++ `!evalFile.current.has_value()`).
        // Stockcrab embeds no network, so this is the state whenever the file was not found.
        if self.eval_file.current.is_empty() {
            let msg = "Failed to export a net. No network file is currently loaded. Please load a network file first.";
            crate::uci::UciEngine::write_stdout_bytes(format!("{msg}\n").as_bytes());
            return Ok(false);
        }

        let actual_filename = if let Some(name) = filename {
            name
        } else {
            if self.eval_file.current != self.eval_file.default_name {
                let msg = "Failed to export a net. A non-embedded net can only be saved if the filename is specified";
                crate::uci::UciEngine::write_stdout_bytes(format!("{msg}\n").as_bytes());
                return Ok(false);
            }
            self.eval_file.default_name.clone()
        };

        // C++ opens the stream unchecked and reports any failure only through `save`'s boolean
        // result, so I/O errors fold into the same "Failed to export a net" path.
        let saved = File::create(&actual_filename).is_ok_and(|file| {
            let mut writer = BufWriter::new(file);
            // Surface a failure of the final buffered write instead of dropping it with the
            // writer.
            matches!(
                self.write_parameters(&mut writer, &self.eval_file.net_description),
                Ok(true)
            ) && writer.flush().is_ok()
        });

        let msg = if saved {
            format!("Network saved successfully to {}", actual_filename)
        } else {
            "Failed to export a net".to_string()
        };

        crate::uci::UciEngine::write_stdout_bytes(format!("{msg}\n").as_bytes());
        Ok(saved)
    }

    /// Verify the network is loaded, reporting through `f`.
    ///
    /// Everything is emitted via the callback rather than printed directly,
    /// matching C++ `Network::verify`. The caller (the UCI layer) wraps it in
    /// `info string`, so printing here would put a bare, non-UCI line on stdout
    /// that a GUI cannot parse.
    pub fn verify(&self, evalfile_path: String, f: &dyn Fn(&str)) {
        let path = if evalfile_path.is_empty() {
            self.eval_file.default_name.clone()
        } else {
            evalfile_path
        };

        if self.eval_file.current != path {
            let msg = format!(
                "ERROR: Network evaluation parameters compatible with the engine must be available.\n\
                 ERROR: The network file {path} was not loaded successfully.\n\
                 ERROR: The UCI option EvalFile might need to specify the full path, including the directory name, to the network file.\n\
                 ERROR: The default net can be downloaded from: https://tests.stockfishchess.org/api/nn/{}\n\
                 ERROR: The engine will be terminated now.\n",
                self.eval_file.default_name
            );
            f(&msg);
            std::process::exit(1);
        }

        // C++ prints `sizeof(featureTransformer) + sizeof(NetworkArchitecture) *
        // storedLayerStacks`, the real weight footprint because its members are inline
        // arrays. The Rust structs hold `AlignedVec`s, so `size_of` would only count a
        // few pointers; the transformer's weight image stands in for the first term and
        // the C++ `sizeof(NetworkArchitecture)` (35,392 bytes for this architecture, with
        // its padding) for the second. The layer stacks (~0.27 MiB) matter: they push
        // the total over the next MiB boundary for the default network, and the message
        // must read the same as upstream's ("107MiB").
        const CPP_SIZEOF_NETWORK_ARCHITECTURE: usize = 35_392;
        let size = FeatureTransformer::shared_image_len()
            + CPP_SIZEOF_NETWORK_ARCHITECTURE * self.stored_layer_stacks;
        f(&format!(
            "NNUE evaluation using {} ({}MiB, ({}, {}, {}, {}, 1))",
            path,
            size / (1024 * 1024),
            FeatureTransformer::INPUT_DIMENSIONS,
            NetworkArchitecture::TRANSFORMED_FEATURE_DIMENSIONS,
            NetworkArchitecture::FC_0_OUTPUTS,
            NetworkArchitecture::FC_1_OUTPUTS,
        ));
    }

    /// Get the feature transformer (used by the accumulator caches and benchmarks).
    #[inline]
    pub fn feature_transformer(&self) -> Option<&FeatureTransformer> {
        self.feature_transformer.as_deref()
    }

    /// Get the network layers (for benchmarking); there are `layer_stack_count()` of them.
    #[inline]
    pub fn network_arch(&self) -> Option<&AlignedPtr<NetworkArchitecture>> {
        self.network.as_ref()
    }

    /// Number of layer stacks the loaded network stores (C++ `storedLayerStacks`).
    #[inline]
    pub fn layer_stack_count(&self) -> usize {
        self.stored_layer_stacks
    }

    /// SFNNv20 layer-stack index: the stored stack of the raw configuration given by the
    /// piece-count band, the queens (side to move has a queen: 2, opponent has a queen: 1), the
    /// pawn layout and whether either side has two or more officers.
    #[inline(always)]
    fn layer_stack(&self, pos: &Position) -> usize {
        let stm = pos.side_to_move();
        let band = (pos.count_all_pieces() - 1) / 4;
        let has_queen = |c| pos.pieces_c_pt::<{ PieceType::Queen as u8 }>(c) != 0;
        let queens = 2 * usize::from(has_queen(stm)) + usize::from(has_queen(!stm));

        // Pawn layout: 0 if pawns stand on both board halves, 1 if on one half only, 2 if there are no pawns
        const QUEEN_SIDE: Bitboard = FILE_A_BB | FILE_B_BB | FILE_C_BB | FILE_D_BB;
        let pawns = pos.pieces_pt::<{ PieceType::Pawn as u8 }>();
        let layout = if pawns == 0 {
            2
        } else if (pawns & QUEEN_SIDE) != 0 && (pawns & !QUEEN_SIDE) != 0 {
            0
        } else {
            1
        };

        // Officers are the pieces other than pawns and kings
        let pawns_and_kings =
            pos.pieces_pt2::<{ PieceType::Pawn as u8 }, { PieceType::King as u8 }>();
        let officers = |c| popcount(pos.pieces_of_color(c) & !pawns_and_kings);
        let many = officers(Color::White).max(officers(Color::Black)) >= 2;

        let raw = ((band * 4 + queens) * 3 + layout) * 2 + usize::from(many);
        debug_assert!(raw < RAW_LAYER_STACKS);
        usize::from(self.layer_stack_index[raw])
    }

    /// Evaluate a position using the network.
    ///
    /// Mirrors C++ `Network::evaluate` (single net, threats folded into the
    /// accumulation).
    pub fn evaluate(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> Value {
        // Stack-allocated, cache-line-aligned buffer for the transformed features.
        // The buffer stays `MaybeUninit` (no reference to uninitialised `u8`s is ever formed)
        // until `transform*` has written all L1 bytes; only then is it viewed as `&[u8]`.
        let mut transformed_features_storage =
            core::mem::MaybeUninit::<AlignedTransformedFeatures<{ L1 }>>::uninit();
        let transformed_features_ptr = transformed_features_storage
            .as_mut_ptr()
            .cast::<core::mem::MaybeUninit<TransformedFeatureType>>();
        // SAFETY: the pointer is valid, aligned and exclusively ours for L1 elements of
        // `MaybeUninit<u8>` (same layout as the struct's `[u8; L1]` field at offset 0, which is
        // `repr(C)`); `MaybeUninit` elements may be uninitialised.
        let transformed_features_out =
            unsafe { core::slice::from_raw_parts_mut(transformed_features_ptr, L1) };

        let bucket = self.layer_stack(pos);

        // Evaluation is only reachable after a successful load (`verify_network` terminates the
        // engine otherwise), and neither Option is ever reset to None afterwards.
        let transformer = self
            .feature_transformer
            .as_ref()
            .expect("network not loaded");
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        let mut nnz = NnzInfo::<256>::new();
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        transformer.transform_with_nnz(
            pos,
            accumulator_stack,
            Some(cache),
            transformed_features_out,
            &mut nnz,
        );
        // C++ `NNZInfo::bitset`: one bit per four-byte chunk of the transformed features.
        #[cfg(target_arch = "aarch64")]
        let mut nnz_bitset = NnzBitset([core::mem::MaybeUninit::uninit(); L1 / 32]);
        #[cfg(target_arch = "aarch64")]
        transformer.transform_with_bitset(
            pos,
            accumulator_stack,
            cache,
            transformed_features_out,
            &mut nnz_bitset.0,
        );
        #[cfg(not(any(
            target_arch = "aarch64",
            all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )
        )))]
        transformer.transform_uninit(
            pos,
            accumulator_stack,
            Some(cache),
            transformed_features_out,
        );
        // SAFETY: `transform*` wrote all L1 bytes of the buffer (see `transform_uninit`), so they
        // are initialised; the earlier `&mut` view is no longer used.
        let transformed_features_slice: &[TransformedFeatureType] =
            unsafe { core::slice::from_raw_parts(transformed_features_ptr.cast(), L1) };

        let network = self.network.as_ref().expect("network not loaded");
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        let positional = network[bucket].propagate_with_nnz(transformed_features_slice, &nnz);
        #[cfg(target_arch = "aarch64")]
        let positional = {
            // SAFETY: `transform_with_bitset` wrote all `L1 / 32` bytes of the bitset, so they
            // are initialised; `MaybeUninit<u8>` has the layout of `u8`.
            let bitset: &[u8] =
                unsafe { core::slice::from_raw_parts(nnz_bitset.0.as_ptr().cast(), L1 / 32) };
            network[bucket].propagate_with_bitset(transformed_features_slice, bitset)
        };
        #[cfg(not(any(
            target_arch = "aarch64",
            all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )
        )))]
        let positional = network[bucket].propagate(transformed_features_slice);

        (positional / OUTPUT_SCALE) as Value
    }

    /// Evaluate a position and return per-bucket trace information.
    pub fn trace_evaluate(
        &self,
        pos: &Position,
        accumulator_stack: &mut AccumulatorStack,
        cache: &mut AccumulatorCaches,
    ) -> NnueEvalTrace {
        let mut trace = NnueEvalTrace::default();
        trace.correct_bucket = self.layer_stack(pos);
        trace.layer_stacks = self.stored_layer_stacks;

        let mut transformed_features = Box::new(AlignedTransformedFeatures([0u8; L1]));

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        let mut nnz = NnzInfo::<256>::new();
        // The transform does not depend on the bucket, so do it once.
        if let Some(ref transformer) = self.feature_transformer {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ))]
            transformer.transform_with_nnz(
                pos,
                accumulator_stack,
                Some(cache),
                crate::nnue::nnue_feature_transformer::as_uninit_bytes(&mut transformed_features.0),
                &mut nnz,
            );
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )))]
            transformer.transform(
                pos,
                accumulator_stack,
                Some(cache),
                &mut transformed_features.0,
            );
        }

        for bucket in 0..self.stored_layer_stacks {
            let positional = if let Some(ref network) = self.network {
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512bw"
                ))]
                {
                    network[bucket].propagate_with_nnz(&transformed_features.0, &nnz)
                }
                #[cfg(not(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512bw"
                )))]
                network[bucket].propagate(&transformed_features.0)
            } else {
                0
            };

            trace.positional[bucket] = (positional / OUTPUT_SCALE) as Value;
        }

        trace
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C, align(64))]
struct AlignedTransformedFeatures<const N: usize>([TransformedFeatureType; N]);

/// C++ `NNZInfo::bitset` (nnz_helper.h): `alignas(8) u8 bitset[(Dimensions + 31) / 32]`.
#[cfg(target_arch = "aarch64")]
#[repr(C, align(8))]
struct NnzBitset([core::mem::MaybeUninit<u8>; L1 / 32]);

/// Alias for [`Network`], kept so upstream-shaped call sites that name `Networks` still read the
/// same.
pub type Networks = Network;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_little_endian_i16_roundtrip() {
        let values: Vec<i16> = (-5000..5000).map(|v| (v * 7) as i16).collect();
        let mut buffer = Vec::new();
        write_little_endian_i16(&mut buffer, &values).unwrap();
        assert_eq!(buffer.len(), values.len() * 2);
        assert_eq!(&buffer[..2], &values[0].to_le_bytes());

        let mut read_values = vec![0i16; values.len()];
        read_little_endian_i16(&mut Cursor::new(buffer.clone()), &mut read_values).unwrap();
        assert_eq!(values, read_values);

        buffer.pop();
        assert!(read_little_endian_i16(&mut Cursor::new(buffer), &mut read_values).is_err());
    }

    #[test]
    fn test_network_creation() {
        // A new network owns no weights (as in C++); `initialize` allocates the transformer and
        // `read_layers` the layer stacks, once the table has given their number.
        let mut network = Network::new();
        assert!(network.feature_transformer.is_none());
        assert!(network.network.is_none());
        network.initialize();
        assert!(network.feature_transformer.is_some());
        assert!(network.network.is_none());
    }

    #[test]
    fn test_network_header_roundtrip() {
        let network = Network::new();
        let mut buffer = Vec::new();

        // Write header
        network
            .write_header(&mut buffer, 0x12345678, "Test Network")
            .unwrap();

        // Read header back
        let mut cursor = Cursor::new(buffer);
        let (hash, description) = network.read_header(&mut cursor).unwrap();

        assert_eq!(hash, 0x12345678);
        assert_eq!(description, "Test Network");
    }

    #[test]
    fn layer_reader_requires_every_hash_stack_and_eof() {
        const STORED: usize = 3;
        let mut network = Network::new();
        network.stored_layer_stacks = STORED;
        let mut serialized = Vec::new();
        for _ in 0..STORED {
            write_little_endian_u32(&mut serialized, NetworkArchitecture::get_hash_value())
                .unwrap();
            NetworkArchitecture::default()
                .write_parameters(&mut serialized)
                .unwrap();
        }
        let stack_size = serialized.len() / STORED;
        network.read_layers(&mut Cursor::new(&serialized)).unwrap();
        for i in 0..STORED {
            serialized[i * stack_size] ^= 1;
            assert!(
                network.read_layers(&mut Cursor::new(&serialized)).is_err(),
                "hash {i}"
            );
            serialized[i * stack_size] ^= 1;
            assert!(
                network
                    .read_layers(&mut Cursor::new(&serialized[..i * stack_size + 20]))
                    .is_err(),
                "truncated stack {i}"
            );
        }
        serialized.push(0);
        assert!(network.read_layers(&mut Cursor::new(&serialized)).is_err());
    }

    /// Serialize a layer-stack table as C++ `write_parameters` does.
    fn layer_stack_table(stored: u32, fallback: &[u16; RAW_LAYER_STACKS]) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_little_endian_u32(&mut bytes, LAYER_STACK_TABLE_HASH).unwrap();
        write_little_endian_u32(&mut bytes, RAW_LAYER_STACKS as u32).unwrap();
        write_little_endian_u32(&mut bytes, stored).unwrap();
        for f in fallback {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn layer_stack_table_maps_fallbacks_to_stored_stacks() {
        // Every third configuration owns a stack; the others use the owner before them.
        let mut fallback = [0u16; RAW_LAYER_STACKS];
        for (s, f) in fallback.iter_mut().enumerate() {
            *f = (s - s % 3) as u16;
        }
        let stored = RAW_LAYER_STACKS.div_ceil(3) as u32;
        let mut network = Network::new();
        let table = layer_stack_table(stored, &fallback);
        assert!(
            network
                .read_layer_stack_table(&mut Cursor::new(&table))
                .unwrap()
        );
        assert_eq!(network.stored_layer_stacks, stored as usize);
        for s in 0..RAW_LAYER_STACKS {
            assert_eq!(
                usize::from(network.layer_stack_index[s]),
                s / 3,
                "configuration {s}"
            );
        }

        // Round trip through the writer's table layout.
        let mut written = Vec::new();
        network.write_parameters(&mut written, "").unwrap();
        let header_len = 12;
        assert_eq!(&written[header_len..header_len + table.len()], &table[..]);

        let rejects = |stored: u32, fallback: &[u16; RAW_LAYER_STACKS]| {
            let table = layer_stack_table(stored, fallback);
            !Network::new()
                .read_layer_stack_table(&mut Cursor::new(&table))
                .unwrap()
        };
        assert!(rejects(stored + 1, &fallback), "stored count mismatch");
        assert!(rejects(0, &fallback), "no stored stacks");
        assert!(
            rejects(RAW_LAYER_STACKS as u32 + 1, &fallback),
            "too many stacks"
        );
        let mut chained = fallback;
        chained[4] = 5; // 5 falls back to 3, so 4 would reach a non-owner
        assert!(rejects(stored, &chained), "fallback to a non-owner");
        let mut out_of_range = fallback;
        out_of_range[7] = RAW_LAYER_STACKS as u16;
        assert!(rejects(stored, &out_of_range), "fallback out of range");

        let mut bad_tag = table.clone();
        bad_tag[0] ^= 1;
        assert!(
            !Network::new()
                .read_layer_stack_table(&mut Cursor::new(&bad_tag))
                .unwrap()
        );
        assert!(
            Network::new()
                .read_layer_stack_table(&mut Cursor::new(&table[..table.len() - 1]))
                .is_err()
        );
    }

    #[test]
    fn test_large_page_ptr() {
        let value = 42i32;
        let ptr = LargePagePtr::new(value);
        assert_eq!(*ptr, 42);
    }

    #[test]
    fn test_aligned_ptr() {
        let mut ptr = AlignedPtr::<i32>::new_array(10);
        ptr[0] = 100;
        ptr[9] = 200;
        assert_eq!(ptr[0], 100);
        assert_eq!(ptr[9], 200);
    }

    #[test]
    fn test_networks_container() {
        let network = Network::new();
        assert_eq!(network.eval_file.default_name, EVAL_FILE_DEFAULT_NAME);
    }

    #[test]
    fn test_position_piece_counting() {
        // Public construction starts from a valid position, including both kings.
        let default_pos = Position::default();
        assert_eq!(default_pos.count_all_pieces(), 32);

        crate::bitboard_fast::init();
        let mut pos = Position::default();
        pos.set_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false,
        )
        .unwrap();
        assert_eq!(pos.count_all_pieces(), 32);
    }
}
