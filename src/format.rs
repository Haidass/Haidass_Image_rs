use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"HIMG0001";
pub const ALIGN: usize = 64;
pub const GROUP_SIZE: usize = 128;
pub const NR_TILE: usize = 16;

/// Size in bytes of one (16 output channels x 128 K-elements) packed block for Q8K128.
/// Layout: [2048 bytes i8 weights (32 x 16 x 4)] [64 bytes i32 w_bias (128*sum)] [64 bytes f32 scales]
pub const Q8K128_BLOCK_BYTES: usize = 2048 + 64 + 64; // 2176 (64-byte aligned)

/// Size in bytes of one (16 output channels x 128 K-elements) packed block for Q4K128.
/// Layout: [1024 bytes packed nibbles (16 x 2 x 32)] [64 bytes i32 w_bias (128*sum)] [64 bytes f32 scales]
pub const Q4K128_BLOCK_BYTES: usize = 1024 + 64 + 64; // 1152 (64-byte aligned)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDType {
    /// Raw little-endian f32 tensor
    F32,
    /// Packed VNNI Q8K128 linear/conv weight [N_pad/16, K_pad/128, 2176 bytes]
    Q8K128,
    /// Packed VNNI Q4K128 linear/conv weight [N_pad/16, K_pad/128, 1152 bytes]
    Q4K128,
    /// Row-wise symmetric INT8 embedding table: [rows * cols] i8 followed by [rows] f32 scales
    EmbQ8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: TensorDType,
    /// Logical shape (e.g., [out_features, in_features] for linear, [C_out, C_in, kH, kW] for conv)
    pub shape: Vec<usize>,
    /// Byte offset from start of file (always 64-byte aligned)
    pub offset: usize,
    /// Byte length
    pub nbytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHeader {
    pub version: u32,
    pub quant_mode: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub tensors: Vec<TensorMeta>,
}

pub struct HimgWriter {
    writer: BufWriter<File>,
    pending: Vec<(String, TensorDType, Vec<usize>, Vec<u8>)>,
    metadata: HashMap<String, serde_json::Value>,
    quant_mode: String,
}

impl HimgWriter {
    pub fn create<P: AsRef<Path>>(path: P, quant_mode: &str) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::with_capacity(8 * 1024 * 1024, file),
            pending: Vec::new(),
            metadata: HashMap::new(),
            quant_mode: quant_mode.to_string(),
        })
    }

    pub fn set_meta(&mut self, key: &str, value: serde_json::Value) {
        self.metadata.insert(key.to_string(), value);
    }

    pub fn add_f32(&mut self, name: &str, shape: Vec<usize>, data: &[f32]) {
        let bytes = bytemuck::cast_slice(data).to_vec();
        self.pending
            .push((name.to_string(), TensorDType::F32, shape, bytes));
    }

    pub fn add_raw(&mut self, name: &str, dtype: TensorDType, shape: Vec<usize>, bytes: Vec<u8>) {
        self.pending.push((name.to_string(), dtype, shape, bytes));
    }

    pub fn finish(mut self) -> Result<()> {
        // Compute exact header size by two-pass serialization with fixed offsets
        let mut metas = Vec::with_capacity(self.pending.len());
        for (name, dtype, shape, bytes) in &self.pending {
            metas.push(TensorMeta {
                name: name.clone(),
                dtype: *dtype,
                shape: shape.clone(),
                offset: 0,
                nbytes: bytes.len(),
            });
        }

        // Reserve header space generously, compute exact offsets
        // Format: [8B MAGIC][8B u64 header_len][header_json][padding to 64B][tensor data...]
        let dummy_header = FileHeader {
            version: 1,
            quant_mode: self.quant_mode.clone(),
            metadata: self.metadata.clone(),
            tensors: metas.clone(),
        };
        let est_len = serde_json::to_vec(&dummy_header)?.len() + metas.len() * 16 + 256;
        let data_start = align_up(16 + est_len, ALIGN);

        let mut cursor = data_start;
        for meta in &mut metas {
            cursor = align_up(cursor, ALIGN);
            meta.offset = cursor;
            cursor += meta.nbytes;
        }

        let header = FileHeader {
            version: 1,
            quant_mode: self.quant_mode,
            metadata: self.metadata,
            tensors: metas,
        };
        let header_bytes = serde_json::to_vec(&header)?;
        if 16 + header_bytes.len() > data_start {
            bail!("Header exceeded reserved space");
        }

        self.writer.write_all(MAGIC)?;
        self.writer
            .write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        self.writer.write_all(&header_bytes)?;
        let pad_len = data_start - (16 + header_bytes.len());
        if pad_len > 0 {
            self.writer.write_all(&vec![0u8; pad_len])?;
        }

        let mut pos = data_start;
        for (i, (_, _, _, bytes)) in self.pending.iter().enumerate() {
            let target = header.tensors[i].offset;
            if target > pos {
                self.writer.write_all(&vec![0u8; target - pos])?;
                pos = target;
            }
            self.writer.write_all(bytes)?;
            pos += bytes.len();
        }
        self.writer.flush()?;
        Ok(())
    }
}

/// Streaming writer that writes tensor payloads directly to a temporary file or two-pass
/// to avoid keeping all quantized tensors in RAM at once.
pub struct StreamingHimgWriter {
    path: std::path::PathBuf,
    temp_path: std::path::PathBuf,
    data_writer: BufWriter<File>,
    rel_offset: usize,
    metas: Vec<TensorMeta>,
    metadata: HashMap<String, serde_json::Value>,
    quant_mode: String,
}

impl StreamingHimgWriter {
    pub fn create<P: AsRef<Path>>(path: P, quant_mode: &str) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temp_path = path.with_extension("himg.tmp");
        let file = File::create(&temp_path)?;
        Ok(Self {
            path,
            temp_path,
            data_writer: BufWriter::with_capacity(8 * 1024 * 1024, file),
            rel_offset: 0,
            metas: Vec::new(),
            metadata: HashMap::new(),
            quant_mode: quant_mode.to_string(),
        })
    }

    pub fn set_meta(&mut self, key: &str, value: serde_json::Value) {
        self.metadata.insert(key.to_string(), value);
    }

    pub fn add_f32(&mut self, name: &str, shape: Vec<usize>, data: &[f32]) -> Result<()> {
        self.add_raw(name, TensorDType::F32, shape, bytemuck::cast_slice(data))
    }

    pub fn add_raw(
        &mut self,
        name: &str,
        dtype: TensorDType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<()> {
        let aligned = align_up(self.rel_offset, ALIGN);
        if aligned > self.rel_offset {
            let pad = [0u8; ALIGN];
            self.data_writer.write_all(&pad[..aligned - self.rel_offset])?;
            self.rel_offset = aligned;
        }
        self.metas.push(TensorMeta {
            name: name.to_string(),
            dtype,
            shape,
            offset: self.rel_offset,
            nbytes: bytes.len(),
        });
        self.data_writer.write_all(bytes)?;
        self.rel_offset += bytes.len();
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.data_writer.flush()?;
        drop(self.data_writer);

        // Compute header size with final offsets
        let est_header = FileHeader {
            version: 1,
            quant_mode: self.quant_mode.clone(),
            metadata: self.metadata.clone(),
            tensors: self.metas.clone(),
        };
        let est_len = serde_json::to_vec(&est_header)?.len() + self.metas.len() * 16 + 512;
        let data_start = align_up(16 + est_len, ALIGN);

        for m in &mut self.metas {
            m.offset += data_start;
        }

        let header = FileHeader {
            version: 1,
            quant_mode: self.quant_mode,
            metadata: self.metadata,
            tensors: self.metas,
        };
        let header_bytes = serde_json::to_vec(&header)?;
        if 16 + header_bytes.len() > data_start {
            bail!("Header exceeded reserved space");
        }

        let mut out = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&self.path)?);
        out.write_all(MAGIC)?;
        out.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        out.write_all(&header_bytes)?;
        let pad_len = data_start - (16 + header_bytes.len());
        if pad_len > 0 {
            out.write_all(&vec![0u8; pad_len])?;
        }
        let mut src = File::open(&self.temp_path)?;
        std::io::copy(&mut src, &mut out)?;
        out.flush()?;
        drop(src);
        let _ = std::fs::remove_file(&self.temp_path);
        Ok(())
    }
}

pub struct HimgFile {
    mmap: Mmap,
    pub header: FileHeader,
    index: HashMap<String, usize>,
}

impl HimgFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open {}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 16 || &mmap[0..8] != MAGIC {
            bail!("Invalid HIMG file magic");
        }
        let hdr_len = u64::from_le_bytes(mmap[8..16].try_into()?) as usize;
        let header: FileHeader = serde_json::from_slice(&mmap[16..16 + hdr_len])?;
        let mut index = HashMap::with_capacity(header.tensors.len());
        for (i, t) in header.tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }
        Ok(Self {
            mmap,
            header,
            index,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn meta(&self, name: &str) -> Result<&TensorMeta> {
        let &idx = self
            .index
            .get(name)
            .with_context(|| format!("Missing tensor '{name}'"))?;
        Ok(&self.header.tensors[idx])
    }

    pub fn raw_bytes(&self, name: &str) -> Result<(&TensorMeta, &[u8])> {
        let meta = self.meta(name)?;
        let slice = &self.mmap[meta.offset..meta.offset + meta.nbytes];
        Ok((meta, slice))
    }

    pub fn f32_slice(&self, name: &str) -> Result<(&TensorMeta, &[f32])> {
        let (meta, bytes) = self.raw_bytes(name)?;
        if meta.dtype != TensorDType::F32 {
            bail!("Tensor '{name}' is {:?}, expected F32", meta.dtype);
        }
        Ok((meta, bytemuck::cast_slice(bytes)))
    }

    pub fn quantized_matrix(&self, name: &str) -> Result<QuantMatrixView<'_>> {
        let (meta, bytes) = self.raw_bytes(name)?;
        if meta.shape.len() < 2 {
            bail!("Tensor '{name}' shape {:?} is not 2D+", meta.shape);
        }
        let n = meta.shape[0];
        let k: usize = meta.shape[1..].iter().product();
        Ok(QuantMatrixView {
            dtype: meta.dtype,
            n,
            k,
            data: bytes,
        })
    }
}

#[derive(Clone, Copy)]
pub struct QuantMatrixView<'a> {
    pub dtype: TensorDType,
    pub n: usize,
    pub k: usize,
    pub data: &'a [u8],
}

#[inline]
pub const fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}
