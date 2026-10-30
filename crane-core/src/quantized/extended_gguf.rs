// SPDX-License-Identifier: MIT

//! GGUF probe for tensor encodings not known by Candle.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

use candle_core::Result;
use candle_core::quantized::gguf_file;

use super::iquant::{IQuantType, ggml_type_name};
use super::ternary::TernaryEncoding;

#[derive(Clone, Debug)]
pub struct ExtendedTensorInfo {
    pub encoding: TernaryEncoding,
    pub shape: Vec<usize>,
    pub offset: u64,
}

/// A tensor stored in a llama.cpp i-quant type (see [`super::iquant`]).
#[derive(Clone, Debug)]
pub struct IQuantTensorInfo {
    pub ty: IQuantType,
    pub shape: Vec<usize>,
    pub offset: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ExtendedGgufInfo {
    /// Prism ternary (`PTQ1_0`/`PQ2_0`) tensors.
    pub tensors: HashMap<String, ExtendedTensorInfo>,
    /// llama.cpp i-quant (and `Q2_0`) tensors, see [`IQuantType`].
    pub iquant: HashMap<String, IQuantTensorInfo>,
}

impl ExtendedGgufInfo {
    #[must_use]
    pub fn is_ternary(&self) -> bool {
        !self.tensors.is_empty()
    }

    #[must_use]
    pub fn has_iquant(&self) -> bool {
        !self.iquant.is_empty()
    }

    /// Whether any tensor needs Crane-side decoding.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty() && self.iquant.is_empty()
    }
}

/// # Errors
///
/// Returns an error if `bytes` is not a well-formed GGUF file or contains a
/// tensor type that neither Candle nor Crane's extended probe recognizes.
pub fn read_content(bytes: &[u8]) -> Result<(gguf_file::Content, ExtendedGgufInfo)> {
    let (patches, info) = probe(&mut Cursor::new(bytes))?;
    if patches.is_empty() {
        let mut cursor = Cursor::new(bytes);
        return Ok((gguf_file::Content::read(&mut cursor)?, info));
    }
    let mut reader = PatchedReader::new(bytes, patches);
    let content = gguf_file::Content::read(&mut reader)?;
    Ok((content, info))
}

/// Parse a GGUF header like Candle's `Content::read`, but tolerate tensor
/// types Crane decodes itself. Load the result through
/// [`Gguf`](super::gguf_file::Gguf), which re-probes the file for those
/// tensors; reading them straight from the `Content` would misread them.
///
/// # Errors
///
/// Returns an error if the header is malformed or uses a tensor type neither
/// Candle nor Crane can decode.
pub fn read_content_lenient(bytes: &[u8]) -> Result<gguf_file::Content> {
    read_content(bytes).map(|(content, _)| content)
}

/// Re-scan an already-parsed GGUF for tensors Crane decodes itself, from any
/// reader over the whole file. Leaves the reader at an unspecified position.
///
/// # Errors
///
/// Returns an error if the header is malformed.
pub fn probe_reader<R: Read + Seek>(reader: &mut R) -> Result<ExtendedGgufInfo> {
    reader.seek(SeekFrom::Start(0))?;
    probe(reader).map(|(_, info)| info)
}

fn probe<R: Read + Seek>(reader: &mut R) -> Result<(HashMap<u64, [u8; 4]>, ExtendedGgufInfo)> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        candle_core::bail!("invalid GGUF magic")
    }
    let version = read_u32(reader)?;
    if version != 3 {
        candle_core::bail!("extended GGUF reader supports version 3, got {version}")
    }
    let tensor_count = read_u64(reader)?;
    let metadata_count = read_u64(reader)?;
    for _ in 0..metadata_count {
        skip_string(reader, file_len)?;
        let ty = read_u32(reader)?;
        skip_value(reader, ty, file_len, 0)?;
    }

    let mut patches = HashMap::new();
    let mut tensors = HashMap::new();
    let mut iquant = HashMap::new();
    for _ in 0..tensor_count {
        let name = read_string(reader, file_len)?;
        let rank = read_u32(reader)? as usize;
        if rank > 4 {
            candle_core::bail!("GGUF tensor {name} has unsupported rank {rank}")
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u64(reader)? as usize);
        }
        shape.reverse();
        let type_offset = reader.stream_position()?;
        let raw_type = read_u32(reader)?;
        let offset = read_u64(reader)?;
        if let Some(encoding) = TernaryEncoding::from_ggml_type_id(raw_type) {
            patches.insert(type_offset, 30u32.to_le_bytes());
            tensors.insert(
                name,
                ExtendedTensorInfo {
                    encoding,
                    shape,
                    offset,
                },
            );
        } else if let Some(ty) = IQuantType::from_ggml_type_id(raw_type) {
            // Present the tensor to Candle as BF16 so its header parse
            // succeeds; the data is only ever read through `iquant`.
            patches.insert(type_offset, 30u32.to_le_bytes());
            iquant.insert(name, IQuantTensorInfo { ty, shape, offset });
        } else if let Some(type_name) = ggml_type_name(raw_type) {
            candle_core::bail!(
                "GGUF tensor {name} uses {type_name} (ggml type {raw_type}), which Crane cannot load yet"
            )
        }
    }
    Ok((patches, ExtendedGgufInfo { tensors, iquant }))
}

fn skip_value<R: Read + Seek>(reader: &mut R, ty: u32, file_len: u64, depth: usize) -> Result<()> {
    if depth > 64 {
        candle_core::bail!("GGUF metadata nesting is too deep")
    }
    match ty {
        0 | 1 | 7 => skip(reader, 1, file_len),
        2 | 3 => skip(reader, 2, file_len),
        4..=6 => skip(reader, 4, file_len),
        8 => skip_string(reader, file_len),
        9 => {
            let elem_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            for _ in 0..count {
                skip_value(reader, elem_type, file_len, depth + 1)?;
            }
            Ok(())
        },
        10..=12 => skip(reader, 8, file_len),
        _ => candle_core::bail!("unknown GGUF metadata type {ty}"),
    }
}

fn read_string<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<String> {
    let len = read_u64(reader)?;
    if len > file_len.saturating_sub(reader.stream_position()?) {
        candle_core::bail!("GGUF string exceeds file bounds")
    }
    let mut data = vec![0u8; len as usize];
    reader.read_exact(&mut data)?;
    Ok(String::from_utf8_lossy(&data).into_owned())
}

fn skip_string<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<()> {
    let len = read_u64(reader)?;
    skip(reader, len, file_len)
}

fn skip<R: Read + Seek>(reader: &mut R, len: u64, file_len: u64) -> Result<()> {
    let end = reader
        .stream_position()?
        .checked_add(len)
        .ok_or_else(|| candle_core::Error::Msg("GGUF offset overflow".into()))?;
    if end > file_len {
        candle_core::bail!("GGUF value exceeds file bounds")
    }
    reader.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

struct PatchedReader<'a> {
    inner: Cursor<&'a [u8]>,
    patches: HashMap<u64, [u8; 4]>,
}

impl<'a> PatchedReader<'a> {
    fn new(bytes: &'a [u8], patches: HashMap<u64, [u8; 4]>) -> Self {
        Self {
            inner: Cursor::new(bytes),
            patches,
        }
    }
}

impl Read for PatchedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let start = self.inner.position();
        let read = self.inner.read(buf)?;
        let end = start + read as u64;
        for (&offset, patch) in &self.patches {
            let patch_end = offset + 4;
            if offset < end && patch_end > start {
                let copy_start = offset.max(start);
                let copy_end = patch_end.min(end);
                for pos in copy_start..copy_end {
                    buf[(pos - start) as usize] = patch[(pos - offset) as usize];
                }
            }
        }
        Ok(read)
    }
}

impl Seek for PatchedReader<'_> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_external_ternary_gguf_when_configured() -> Result<()> {
        let Ok(path) = std::env::var("CRANE_TEST_TERNARY_GGUF") else {
            return Ok(());
        };
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (content, extended) = read_content(&mmap)?;
        assert!(extended.is_ternary());
        assert_eq!(
            content.metadata["general.architecture"].to_string()?,
            "qwen35"
        );
        assert!(extended.tensors.contains_key("token_embd.weight"));
        let mut gg = crate::quantized::gguf_file::Gguf::new_extended(
            content,
            Cursor::new(mmap.as_ref()),
            candle_core::Device::Cpu,
            candle_core::DType::F32,
            extended,
        )?;
        let layer = gg.linear("blk.0.attn_gate.weight")?;
        let crate::ops::linear::LinearLayer::Ternary(layer) = layer else {
            panic!("expected a ternary linear")
        };
        assert_eq!(layer.weight().cols(), 5120);
        assert!(layer.weight().size_in_bytes() > 0);
        Ok(())
    }
}
