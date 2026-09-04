// SPDX-License-Identifier: MIT
//! Resolves ONNX's "external data" mechanism (`TensorProto.data_location ==
//! EXTERNAL`), where a tensor's bytes are stored in a sidecar file next to
//! the `.onnx` protobuf instead of inline. Needed for exports past the
//! ~2GB inline-protobuf limit, which store large initializers (e.g.
//! quantized weights) in a sibling `.onnx.data` file.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::{Result, bail};
use memmap2::Mmap;

use super::proto::tensor_proto::DataLocation;
use super::proto::{GraphProto, ModelProto, SparseTensorProto, TensorProto};

/// Sidecar files read so far, keyed by their canonicalized path and
/// memory-mapped so a large sidecar isn't fully copied into the heap up
/// front, only the byte ranges each tensor actually references.
type ExternalFiles = HashMap<PathBuf, Mmap>;

/// Inlines every external-data tensor in `model`'s graph (and subgraphs,
/// e.g. an `If` node's `then_branch`/`else_branch`) from sidecar files
/// resolved relative to `onnx_path`'s directory.
///
/// # Errors
///
/// Returns an error if `onnx_path`'s directory can't be resolved, an
/// external-data tensor's `location` resolves outside that directory, its
/// sidecar file is missing, it has an unparseable `offset`/`length`, or it
/// doesn't contain the referenced byte range.
pub(crate) fn inline(model: &mut ModelProto, onnx_path: &Path) -> Result<()> {
    let Some(graph) = model.graph.as_mut() else {
        return Ok(());
    };
    let base_dir = onnx_path.parent().unwrap_or_else(|| Path::new("."));
    let base_dir = base_dir.canonicalize().map_err(|err| {
        candle_core::Error::Msg(format!(
            "resolving model directory {}: {err}",
            base_dir.display()
        ))
    })?;
    inline_graph(graph, &base_dir, &mut HashMap::new())
}

/// Walks every initializer, sparse initializer, and node-attribute tensor
/// of `graph` (and its subgraphs), inlining any external-data reference
/// found. `external_files` caches each sidecar's mapped bytes by
/// canonicalized path across the whole walk, since a single sidecar is
/// typically referenced by hundreds of tensors. `base_dir` must already be
/// canonicalized, so a `location` escaping it can be detected by prefix.
fn inline_graph(
    graph: &mut GraphProto,
    base_dir: &Path,
    external_files: &mut ExternalFiles,
) -> Result<()> {
    for initializer in &mut graph.initializer {
        inline_tensor(initializer, base_dir, external_files)?;
    }
    for sparse_initializer in &mut graph.sparse_initializer {
        inline_sparse_tensor(sparse_initializer, base_dir, external_files)?;
    }
    for node in &mut graph.node {
        for attribute in &mut node.attribute {
            if let Some(tensor) = attribute.t.as_mut() {
                inline_tensor(tensor, base_dir, external_files)?;
            }
            for tensor in &mut attribute.tensors {
                inline_tensor(tensor, base_dir, external_files)?;
            }
            if let Some(sparse_tensor) = attribute.sparse_tensor.as_mut() {
                inline_sparse_tensor(sparse_tensor, base_dir, external_files)?;
            }
            for sparse_tensor in &mut attribute.sparse_tensors {
                inline_sparse_tensor(sparse_tensor, base_dir, external_files)?;
            }
            if let Some(subgraph) = attribute.g.as_mut() {
                inline_graph(subgraph, base_dir, external_files)?;
            }
            for subgraph in &mut attribute.graphs {
                inline_graph(subgraph, base_dir, external_files)?;
            }
        }
    }
    Ok(())
}

/// Inlines a sparse tensor's `values` and `indices` tensors, either of
/// which may independently reference external data per the ONNX spec.
fn inline_sparse_tensor(
    sparse_tensor: &mut SparseTensorProto,
    base_dir: &Path,
    external_files: &mut ExternalFiles,
) -> Result<()> {
    if let Some(values) = sparse_tensor.values.as_mut() {
        inline_tensor(values, base_dir, external_files)?;
    }
    if let Some(indices) = sparse_tensor.indices.as_mut() {
        inline_tensor(indices, base_dir, external_files)?;
    }
    Ok(())
}

/// Replaces `tensor`'s external-data reference with its bytes read from the
/// sidecar file it names (resolved relative to `base_dir`, which must
/// already be canonicalized), using `external_files` to map each sidecar
/// from disk only once. No-op for a tensor that already stores its data
/// inline (`data_location == DEFAULT`).
fn inline_tensor(
    tensor: &mut TensorProto,
    base_dir: &Path,
    external_files: &mut ExternalFiles,
) -> Result<()> {
    if tensor.data_location != DataLocation::External as i32 {
        return Ok(());
    }
    let mut location = None;
    let mut offset = 0usize;
    let mut length = None;
    for entry in &tensor.external_data {
        match entry.key.as_str() {
            "location" => location = Some(entry.value.as_str()),
            "offset" => {
                offset = entry.value.parse().map_err(|err| {
                    candle_core::Error::Msg(format!(
                        "external tensor {:?} has an invalid 'offset' {:?}: {err}",
                        tensor.name, entry.value
                    ))
                })?;
            },
            "length" => {
                length = Some(entry.value.parse::<usize>().map_err(|err| {
                    candle_core::Error::Msg(format!(
                        "external tensor {:?} has an invalid 'length' {:?}: {err}",
                        tensor.name, entry.value
                    ))
                })?);
            },
            _ => {},
        }
    }
    let Some(location) = location else {
        bail!(
            "external tensor {:?} is missing the required 'location' entry",
            tensor.name
        );
    };

    let path = base_dir.join(location);
    let path = path.canonicalize().map_err(|err| {
        candle_core::Error::Msg(format!(
            "reading external data file {} for tensor {:?}: {err}",
            path.display(),
            tensor.name
        ))
    })?;
    if !path.starts_with(base_dir) {
        bail!(
            "external tensor {:?} location {location:?} resolves to {}, which escapes the model directory {}",
            tensor.name,
            path.display(),
            base_dir.display()
        );
    }
    if let std::collections::hash_map::Entry::Vacant(entry) = external_files.entry(path.clone()) {
        let file = File::open(&path).map_err(|err| {
            candle_core::Error::Msg(format!(
                "reading external data file {} for tensor {:?}: {err}",
                path.display(),
                tensor.name
            ))
        })?;
        // SAFETY: the sidecar isn't expected to be modified by another
        // process while the model loads; a concurrent truncation would
        // surface as a SIGBUS on access, not memory unsafety this can
        // prevent.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            candle_core::Error::Msg(format!(
                "memory-mapping external data file {} for tensor {:?}: {err}",
                path.display(),
                tensor.name
            ))
        })?;
        entry.insert(mmap);
    }
    let file = &external_files[&path];
    let end = match length {
        Some(length) => offset.checked_add(length).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "external tensor {:?} byte range offset {offset} + length {length} overflows",
                tensor.name
            ))
        })?,
        None => file.len(),
    };
    if offset > file.len() || end > file.len() || offset > end {
        bail!(
            "external tensor {:?} references byte range {offset}..{end} outside {} ({} bytes)",
            tensor.name,
            path.display(),
            file.len()
        );
    }
    tensor.raw_data = file[offset..end].to_vec();
    tensor.data_location = DataLocation::Default as i32;
    tensor.external_data.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;
    use crate::onnx::proto::tensor_proto::DataType;
    use crate::onnx::proto::{AttributeProto, NodeProto, StringStringEntryProto, attribute_proto};
    use crate::onnx::{eval, read_file};

    fn external_data_entry(key: &str, value: &str) -> StringStringEntryProto {
        StringStringEntryProto {
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    fn external_f32_tensor(
        name: &str,
        dims: Vec<i64>,
        external_data: Vec<(&str, &str)>,
    ) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims,
            data_type: DataType::Float as i32,
            data_location: DataLocation::External as i32,
            external_data: external_data
                .into_iter()
                .map(|(key, value)| external_data_entry(key, value))
                .collect(),
            ..Default::default()
        }
    }

    // A sidecar file with a 4-byte "header" ahead of the real payload —
    // `offset`/`length` must select exactly the payload, not the whole file.
    #[test]
    fn read_file_inlines_external_data_with_offset_and_length() {
        let dir = tempfile::tempdir().unwrap();
        let mut sidecar = vec![0xAAu8; 4];
        let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        for value in values {
            sidecar.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(dir.path().join("data.bin"), &sidecar).unwrap();

        let tensor = external_f32_tensor(
            "weight",
            vec![4],
            vec![("location", "data.bin"), ("offset", "4"), ("length", "16")],
        );
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let loaded = read_file(&onnx_path).unwrap();
        let initializer = &loaded.graph.unwrap().initializer[0];
        assert_eq!(initializer.data_location, DataLocation::Default as i32);
        assert!(initializer.external_data.is_empty());
        let decoded = eval::get_tensor(initializer, "weight").unwrap();
        assert_eq!(decoded.to_vec1::<f32>().unwrap(), values);
    }

    // No `offset`/`length` entries means the tensor's data is the sidecar
    // file's entire contents.
    #[test]
    fn read_file_inlines_external_data_without_offset_or_length() {
        let dir = tempfile::tempdir().unwrap();
        let values: [f32; 2] = [5.0, 6.0];
        let mut sidecar = Vec::new();
        for value in values {
            sidecar.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(dir.path().join("data.bin"), &sidecar).unwrap();

        let tensor = external_f32_tensor("weight", vec![2], vec![("location", "data.bin")]);
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let loaded = read_file(&onnx_path).unwrap();
        let initializer = &loaded.graph.unwrap().initializer[0];
        let decoded = eval::get_tensor(initializer, "weight").unwrap();
        assert_eq!(decoded.to_vec1::<f32>().unwrap(), values);
    }

    // Two external tensors sharing one sidecar file, at disjoint byte
    // ranges — exercises the `external_files` cache hitting on the second
    // tensor and must still return the correct, distinct bytes for each.
    #[test]
    fn read_file_inlines_multiple_tensors_from_shared_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let mut sidecar = Vec::new();
        sidecar.extend_from_slice(&1.0f32.to_le_bytes());
        sidecar.extend_from_slice(&2.0f32.to_le_bytes());

        std::fs::write(dir.path().join("data.bin"), &sidecar).unwrap();
        let tensor_a = external_f32_tensor(
            "a",
            vec![1],
            vec![("location", "data.bin"), ("offset", "0"), ("length", "4")],
        );
        let tensor_b = external_f32_tensor(
            "b",
            vec![1],
            vec![("location", "data.bin"), ("offset", "4"), ("length", "4")],
        );
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor_a, tensor_b],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let loaded = read_file(&onnx_path).unwrap();
        let graph = loaded.graph.unwrap();
        assert_eq!(
            eval::get_tensor(&graph.initializer[0], "a")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [1.0]
        );
        assert_eq!(
            eval::get_tensor(&graph.initializer[1], "b")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [2.0]
        );
    }

    // A `Constant` node's `value` attribute tensor can also carry external
    // data per the ONNX spec, not just `graph.initializer` entries.
    #[test]
    fn read_file_inlines_external_data_in_node_attribute() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.bin"), 7.0f32.to_le_bytes()).unwrap();

        let tensor = external_f32_tensor("value", vec![1], vec![("location", "data.bin")]);
        let node = NodeProto {
            op_type: "Constant".to_string(),
            output: vec!["out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                t: Some(tensor),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(GraphProto {
                node: vec![node],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let loaded = read_file(&onnx_path).unwrap();
        let graph = loaded.graph.unwrap();
        let tensor = graph.node[0].attribute[0].t.as_ref().unwrap();
        assert_eq!(tensor.data_location, DataLocation::Default as i32);
        assert_eq!(
            eval::get_tensor(tensor, "value")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [7.0]
        );
    }

    // Missing the required 'location' entry must be a clear load-time
    // error, not a panic or silent empty tensor.
    #[test]
    fn read_file_errors_on_missing_location() {
        let dir = tempfile::tempdir().unwrap();
        let tensor = external_f32_tensor("weight", vec![1], vec![("offset", "0")]);
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let err = read_file(&onnx_path).unwrap_err();
        assert!(err.to_string().contains("location"));
    }

    // A byte range past the end of the sidecar file must be a clear
    // load-time error, not an out-of-bounds panic.
    #[test]
    fn read_file_errors_on_out_of_range_byte_range() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.bin"), [0u8; 4]).unwrap();

        let tensor = external_f32_tensor(
            "weight",
            vec![4],
            vec![("location", "data.bin"), ("offset", "0"), ("length", "16")],
        );
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let err = read_file(&onnx_path).unwrap_err();
        assert!(err.to_string().contains("byte range"));
    }

    // An external tensor's `location` must not be able to escape the
    // model's directory via `..` components — this would otherwise let a
    // crafted `.onnx` file read arbitrary files off disk.
    #[test]
    fn read_file_errors_on_location_escaping_model_directory() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("secret.bin"), [0u8; 4]).unwrap();
        let model_dir = root.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();

        let tensor = external_f32_tensor("weight", vec![1], vec![("location", "../secret.bin")]);
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = model_dir.join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let err = read_file(&onnx_path).unwrap_err();
        assert!(err.to_string().contains("escapes"));
    }

    // `offset + length` must not panic on overflow when both are large,
    // untrusted values parsed straight from the model file.
    #[test]
    fn read_file_errors_on_offset_length_overflow() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.bin"), [0u8; 4]).unwrap();

        let tensor = external_f32_tensor(
            "weight",
            vec![1],
            vec![
                ("location", "data.bin"),
                ("offset", &usize::MAX.to_string()),
                ("length", "1"),
            ],
        );
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let err = read_file(&onnx_path).unwrap_err();
        assert!(err.to_string().contains("overflows"));
    }

    // A non-numeric 'offset' must be a clear load-time error, not a panic.
    #[test]
    fn read_file_errors_on_unparseable_offset() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.bin"), [0u8; 4]).unwrap();

        let tensor = external_f32_tensor(
            "weight",
            vec![1],
            vec![("location", "data.bin"), ("offset", "not-a-number")],
        );
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let err = read_file(&onnx_path).unwrap_err();
        assert!(err.to_string().contains("offset"));
    }

    // A `location` naming a sidecar file that doesn't exist must be a clear
    // load-time error, not a panic.
    #[test]
    fn read_file_errors_on_missing_sidecar_file() {
        let dir = tempfile::tempdir().unwrap();
        let tensor = external_f32_tensor("weight", vec![1], vec![("location", "missing.bin")]);
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let err = read_file(&onnx_path).unwrap_err();
        assert!(err.to_string().contains("weight"));
    }

    // An `If` node's `then_branch` subgraph can hold its own initializers,
    // which must also be walked for external-data resolution.
    #[test]
    fn read_file_inlines_external_data_in_subgraph() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.bin"), 9.0f32.to_le_bytes()).unwrap();

        let tensor = external_f32_tensor("weight", vec![1], vec![("location", "data.bin")]);
        let then_branch = GraphProto {
            initializer: vec![tensor],
            ..Default::default()
        };
        let node = NodeProto {
            op_type: "If".to_string(),
            output: vec!["out".to_string()],
            attribute: vec![AttributeProto {
                name: "then_branch".to_string(),
                r#type: attribute_proto::AttributeType::Graph as i32,
                g: Some(then_branch),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(GraphProto {
                node: vec![node],
                ..Default::default()
            }),
            ..Default::default()
        };
        let onnx_path = dir.path().join("model.onnx");
        std::fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        let loaded = read_file(&onnx_path).unwrap();
        let graph = loaded.graph.unwrap();
        let subgraph = graph.node[0].attribute[0].g.as_ref().unwrap();
        let initializer = &subgraph.initializer[0];
        assert_eq!(initializer.data_location, DataLocation::Default as i32);
        assert_eq!(
            eval::get_tensor(initializer, "weight")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [9.0]
        );
    }
}
