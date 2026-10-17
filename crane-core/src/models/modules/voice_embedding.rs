// SPDX-License-Identifier: MIT

//! Shared reader for tensor bytes stored inside `PyTorch` `.pt` ZIP archives.
//!
//! `torch.save` writes each file under a top-level directory named after a
//! fresh temp dir, which differs per file. The tensor storages inside live at
//! `<top-level-dir>/data/<index>`, so callers must match on the `data/<index>`
//! suffix rather than a fixed path.

use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result};

/// Read the raw bytes of the `index`-th tensor storage from a `PyTorch` `.pt`
/// ZIP archive at `path`.
///
/// Matches any ZIP entry ending in `data/{index}`, regardless of the
/// archive's top-level directory name.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the ZIP structure is
/// invalid, or no entry matches `data/{index}`.
pub fn load_pt_tensor_bytes(path: &Path, index: usize) -> Result<Vec<u8>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("failed to read ZIP in {}", path.display()))?;

    let suffix = format!("data/{index}");
    let dir_suffix = format!("/{suffix}");
    let mut match_index = None;
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .with_context(|| format!("failed to read ZIP entry {i} in {}", path.display()))?;
        let name = entry.name().with_context(|| {
            format!("failed to read name of ZIP entry {i} in {}", path.display())
        })?;
        let name = name.as_ref();
        if name == suffix || name.ends_with(&dir_suffix) {
            match_index = Some(i);
            break;
        }
    }
    let match_index = match_index
        .with_context(|| format!("could not find a '{suffix}' entry in {}", path.display()))?;

    // `by_index`'s returned entry borrows `archive` mutably, which conflicts
    // with the loop above reusing `archive.by_index(i)` on the next
    // iteration, so the match is looked up again here instead of kept.
    let mut entry = archive
        .by_index(match_index)
        .with_context(|| format!("failed to read '{suffix}' entry in {}", path.display()))?;
    let mut bytes = Vec::new();
    entry
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read '{suffix}' from {}", path.display()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an in-memory `.pt`-style ZIP with `data/0` and `data/1` entries
    /// under an arbitrary top-level directory, matching `torch.save`'s
    /// temp-dir-prefixed layout.
    fn make_pt_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write as _;
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, data) in entries {
                zip.start_file(*name, opts).unwrap();
                zip.write_all(data).unwrap();
            }
            zip.finish().unwrap();
        }
        buf
    }

    // Verifies entries are matched by their data/<index> suffix regardless of
    // the random top-level directory name torch.save produces.
    #[test]
    fn test_load_pt_tensor_bytes_matches_suffix() {
        let data0: &[u8] = b"acoustic-bytes";
        let data1: &[u8] = b"semantic-bytes";
        let pt_bytes =
            make_pt_bytes(&[("tmpwqrdrmpm/data/0", data0), ("tmpwqrdrmpm/data/1", data1)]);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write as _;
        tmp.as_file().write_all(&pt_bytes).unwrap();

        assert_eq!(load_pt_tensor_bytes(tmp.path(), 0).unwrap(), data0);
        assert_eq!(load_pt_tensor_bytes(tmp.path(), 1).unwrap(), data1);
    }

    // Verifies a missing index produces an error instead of silently
    // returning the wrong entry.
    #[test]
    fn test_load_pt_tensor_bytes_missing_index() {
        let pt_bytes = make_pt_bytes(&[("tmpabc/data/0", b"only-entry")]);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write as _;
        tmp.as_file().write_all(&pt_bytes).unwrap();

        assert!(load_pt_tensor_bytes(tmp.path(), 99).is_err());
    }
}
