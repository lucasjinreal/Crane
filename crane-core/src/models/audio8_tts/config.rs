// SPDX-License-Identifier: MIT
//! Deserializes Audio8-TTS's `runtime_manifest.json` and parses the bundled
//! `reference_codes.npy` reference-voice codec codes.
//!
//! There is no NPY-reading crate elsewhere in Crane; `reference_codes.npy` is
//! a single, known-shape `NumPy` array (`<i8` dtype meaning little-endian
//! 8-byte signed integers, i.e. `i64`, 2-D, C-order), so a minimal
//! special-purpose parser is implemented here rather than pulling in a
//! general-purpose NPY crate for one file.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// The fields of `runtime_manifest.json` that Audio8-TTS's ONNX wrapper
/// reads at load time and during generation.
#[derive(Debug, Deserialize)]
pub(crate) struct RuntimeManifest {
    /// Output waveform sample rate, in Hz (44100 for the 0.1B INT8 package).
    pub sample_rate: u32,
    /// Number of codec codebooks generated per audio frame by the fast AR.
    pub num_codebooks: usize,
    /// Number of entries per codebook.
    pub codebook_size: usize,
    /// First vocabulary ID of the semantic-token range.
    pub semantic_begin_id: i64,
    /// Last vocabulary ID of the semantic-token range (inclusive).
    pub semantic_end_id: i64,
    /// Vocabulary ID that signals end-of-generation.
    pub im_end_id: i64,
    /// Maximum slow-AR sequence length (prompt + generated frames).
    pub max_seq_len: usize,
    /// Number of slow AR transformer/Mamba layers.
    pub num_layers: usize,
    /// Number of fast AR transformer layers.
    pub num_fast_layers: usize,
    /// Filename, relative to the model directory, of the bundled reference
    /// voice's pre-encoded codec codes.
    pub reference_codes: String,
    /// Transcript of the bundled reference voice's audio.
    pub reference_text: String,
    /// Quantization precision selected by default (e.g. `"int8"`), used to
    /// key into `slow_decode_models`/`fast_models`.
    pub default_precision: String,
    /// Codec precision selected by default (e.g. `"fp16"`), used to key into
    /// `codec_models`.
    pub default_codec_precision: String,
    /// Maps precision name to the slow AR ONNX filename for that precision.
    pub slow_decode_models: HashMap<String, String>,
    /// Maps precision name to the fast AR ONNX filename for that precision.
    pub fast_models: HashMap<String, String>,
    /// Maps codec precision name to the codec decoder ONNX filename.
    pub codec_models: HashMap<String, String>,
}

/// Reads and deserializes `runtime_manifest.json` from `model_dir`.
///
/// # Errors
///
/// Returns an error if the file can't be read or doesn't match the expected
/// schema.
pub(crate) fn load_manifest(model_dir: &Path) -> Result<RuntimeManifest> {
    let path = model_dir.join("runtime_manifest.json");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

/// Locates the `'key':` marker in a `NumPy` header dict literal and returns the
/// byte offset immediately after its colon, where the value starts.
fn field_value_start(header: &str, key: &str) -> Result<usize> {
    let marker = format!("'{key}'");
    let key_pos = header
        .find(&marker)
        .ok_or_else(|| anyhow::anyhow!("NPY header missing '{key}' field"))?;
    let colon_offset = header[key_pos..]
        .find(':')
        .ok_or_else(|| anyhow::anyhow!("NPY header '{key}' field has no ':'"))?;
    Ok(key_pos + colon_offset + 1)
}

/// Extracts the single-quoted string value of `key` from a `NumPy` header dict
/// literal (e.g. `descr` or `fortran_order`'s literal text).
fn extract_quoted(header: &str, key: &str) -> Result<String> {
    let start = field_value_start(header, key)?;
    let rest = &header[start..];
    let quote_start = rest
        .find('\'')
        .ok_or_else(|| anyhow::anyhow!("NPY header '{key}' value isn't quoted"))?
        + 1;
    let quote_end = rest[quote_start..]
        .find('\'')
        .ok_or_else(|| anyhow::anyhow!("NPY header '{key}' value has no closing quote"))?;
    Ok(rest[quote_start..quote_start + quote_end].to_string())
}

/// Extracts the shape tuple's dimensions from a `NumPy` header dict literal,
/// e.g. `(10, 110)` -> `[10, 110]`.
fn extract_shape(header: &str) -> Result<Vec<usize>> {
    let start = field_value_start(header, "shape")?;
    let rest = &header[start..];
    let paren_start = rest
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("NPY header 'shape' value has no '('"))?
        + 1;
    let paren_end = rest[paren_start..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("NPY header 'shape' value has no ')'"))?;
    rest[paren_start..paren_start + paren_end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}

/// Parses a `NumPy` `.npy` v1.x file holding a 2-D, C-order, `<i8` (i.e.
/// little-endian `i64`) array, returning its rows and `(nrows, ncols)`.
///
/// This is deliberately narrow: it supports exactly the format Audio8-TTS's
/// `reference_codes.npy` uses, not the general NPY spec.
///
/// # Errors
///
/// Returns an error if the file isn't a valid v1.x NPY file, isn't `<i8`
/// dtype, isn't C-order, isn't 2-D, or its data length doesn't match its
/// declared shape.
pub(crate) fn load_npy_i64_2d(path: &Path) -> Result<(Vec<Vec<i64>>, usize, usize)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        bail!("{} is not a NumPy .npy file (bad magic)", path.display());
    }
    let major = bytes[6];
    if major != 1 {
        bail!(
            "{} uses unsupported NPY version {major}.{}, only 1.x is supported",
            path.display(),
            bytes[7]
        );
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_start = 10;
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        bail!("{} has a truncated NPY header", path.display());
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .with_context(|| format!("{} has a non-UTF8 NPY header", path.display()))?;

    let descr = extract_quoted(header, "descr")?;
    if descr != "<i8" {
        bail!(
            "{} has dtype {descr:?}, only '<i8' (little-endian i64) is supported",
            path.display()
        );
    }
    // `fortran_order`'s literal value is the bareword `True`/`False`, not a
    // quoted string, so it needs its own (non-`extract_quoted`) check.
    let fortran_order_start = field_value_start(header, "fortran_order")?;
    let is_fortran = header[fortran_order_start..]
        .trim_start()
        .starts_with("True");
    if is_fortran {
        bail!(
            "{} is stored in Fortran order, only C order is supported",
            path.display()
        );
    }

    let shape = extract_shape(header)?;
    let [rows, cols] = shape[..] else {
        bail!(
            "{} has shape {shape:?}, expected a 2-D array",
            path.display()
        );
    };

    let data_start = header_end;
    let expected_len = rows * cols * 8;
    let actual_len = bytes.len() - data_start;
    if actual_len != expected_len {
        bail!(
            "{} has {actual_len} data bytes, expected {expected_len} for shape {rows}x{cols} of i64",
            path.display()
        );
    }

    let mut out = Vec::with_capacity(rows);
    let mut offset = data_start;
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            let value = i64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
            row.push(value);
            offset += 8;
        }
        out.push(row);
    }
    Ok((out, rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The manifest fields the model wrapper actually reads must round-trip
    // through serde from the real package's JSON shape.
    #[test]
    fn parse_manifest_roundtrip() {
        let json = r#"{
            "sample_rate": 44100,
            "num_codebooks": 10,
            "codebook_size": 4096,
            "semantic_begin_id": 65537,
            "semantic_end_id": 69632,
            "im_end_id": 4096,
            "max_seq_len": 2048,
            "num_layers": 24,
            "num_fast_layers": 4,
            "reference_codes": "reference_codes.npy",
            "reference_text": "hello",
            "default_precision": "int8",
            "default_codec_precision": "fp16",
            "slow_decode_models": {"int8": "slow_ar_int8.onnx"},
            "fast_models": {"int8": "fast_ar_int8.onnx"},
            "codec_models": {"fp16": "codec_decoder_fp16.onnx"}
        }"#;
        let manifest: RuntimeManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.sample_rate, 44100);
        assert_eq!(manifest.semantic_begin_id, 65537);
        assert_eq!(
            manifest.slow_decode_models.get("int8").unwrap(),
            "slow_ar_int8.onnx"
        );
    }

    fn write_npy_i64(
        dir: &std::path::Path,
        rows: usize,
        cols: usize,
        values: &[i64],
    ) -> std::path::PathBuf {
        let header_body =
            format!("{{'descr': '<i8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        // NumPy pads the header so header_start + header_len is a multiple
        // of 64 and the whole thing ends with '\n'; not required for
        // correctness here, just padding with a trailing newline suffices.
        let mut header_body = header_body;
        header_body.push('\n');
        let header_len = header_body.len() as u16;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1); // major version
        bytes.push(0); // minor version
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header_body.as_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let path = dir.join("test.npy");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    // A valid <i8 (int64), C-order, 2-D array parses into the right rows.
    #[test]
    fn load_npy_parses_valid_i64_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_npy_i64(dir.path(), 2, 3, &[1, 2, 3, 4, 5, 6]);
        let (rows, nrows, ncols) = load_npy_i64_2d(&path).unwrap();
        assert_eq!(nrows, 2);
        assert_eq!(ncols, 3);
        assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    // A dtype other than <i8 must be a clear load-time error, not silently
    // misinterpreted bytes.
    #[test]
    fn load_npy_rejects_wrong_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let header_body = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1), }\n";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header_body.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header_body.as_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes());
        let path = dir.path().join("test.npy");
        std::fs::write(&path, bytes).unwrap();

        let err = load_npy_i64_2d(&path).unwrap_err();
        assert!(err.to_string().contains("dtype"));
    }

    // Fortran-ordered arrays aren't supported and must error, not silently
    // return transposed data.
    #[test]
    fn load_npy_rejects_fortran_order() {
        let dir = tempfile::tempdir().unwrap();
        let header_body = "{'descr': '<i8', 'fortran_order': True, 'shape': (1, 1), }\n";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header_body.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header_body.as_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        let path = dir.path().join("test.npy");
        std::fs::write(&path, bytes).unwrap();

        let err = load_npy_i64_2d(&path).unwrap_err();
        assert!(err.to_string().contains("Fortran"));
    }
}
