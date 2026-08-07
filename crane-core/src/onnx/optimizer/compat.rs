// SPDX-License-Identifier: MIT

//! Load-time ONNX graph rewrites that work around gaps in
//! `crate::onnx::eval`'s vendored evaluator, for models (currently Kokoro
//! TTS) whose export hits an op `eval.rs` either can't run at all or runs
//! incorrectly. Run as part of `crate::onnx::optimizer::optimize`, before
//! its constant-folding/alias-elimination passes so their output (e.g. the
//! `Constant` nodes this module's rewrites introduce) gets simplified too.
//!
//! Two different kinds of gap exist, and only one of them belongs here:
//!
//! - **Ops `eval.rs` runs *incorrectly*** — the op is implemented, but its
//!   implementation is wrong for some input shape, dtype, or attribute
//!   combination. These are fixable by rewriting the graph, at load time,
//!   into a decomposition of ops `eval.rs` already runs *correctly*. That
//!   rewrite is [`rewrite_unsupported_ops`]'s job.
//! - **Ops `eval.rs` can't run at all** — no dispatch arm exists for the op
//!   (e.g. `Resize` `mode="linear"`, a real DSP computation with no
//!   decomposition into other ONNX ops `eval.rs` already runs). Those need
//!   a native Rust implementation added directly to `eval.rs`, not a graph
//!   rewrite.
//!
//! This module intentionally never modifies `crate::onnx::eval` itself:
//! upstream Crane doesn't want per-model workarounds baked into the shared
//! evaluator, so every fix here is a graph transformation applied once,
//! before `crate::onnx::simple_eval` ever sees the graph.
//!
//! # Gaps fixed here
//!
//! - **`Trilu` produces NaN on `+/-inf` inputs.** `eval.rs`'s `Trilu`
//!   computes `input * mask`, and `0 * inf` is `NaN` in IEEE 754 — so any
//!   masked-out entry of an `f32::INFINITY`/`f32::NEG_INFINITY`-valued
//!   input (e.g. an additive attention mask before a softmax) becomes NaN
//!   instead of the intended value. [`expand_trilu`] rewrites `Trilu` into
//!   a `Where`-based selection between `data` and a same-dtype zero
//!   tensor, which never multiplies and so never produces this NaN.
//! - **`ReduceSum` doesn't normalize negative axes.** `eval.rs` casts a
//!   negative axis directly via `x as usize`, producing an out-of-range
//!   index instead of wrapping. [`fix_reduce_sum_negative_axes`] rewrites
//!   `axes` to a normalized form, either at rewrite time (when `axes` is a
//!   compile-time constant with a declared rank) or via a small runtime
//!   subgraph otherwise.
//! - **`ReduceMean` doesn't read the opset-18+ `axes` input.** `eval.rs`'s
//!   `ReduceMean` only ever reads the older `axes` *attribute*, so an
//!   axes-as-input export silently reduces over every axis instead of the
//!   intended ones. [`fix_reduce_mean_axes_input`] folds a compile-time
//!   constant axes input into the attribute form.
//! - **`CumSum` doesn't support `int64`, and doesn't normalize negative
//!   axes either.** `eval.rs`'s `CumSum` is a matmul-based, float-only
//!   implementation, and (like `ReduceSum`) never normalizes a negative
//!   `axis` input. [`fix_int_cumsum`] wraps `data` in a `Double` round-trip
//!   cast (cast back via `CastLike` against `data` itself, so the original
//!   dtype never needs to be statically known) and normalizes `axis`
//!   dynamically.
//! - **`LSTM` only implements `direction == "forward"`.** `eval.rs` bails
//!   immediately on `"bidirectional"`. [`expand_bidirectional_lstm`] splits
//!   a bidirectional `LSTM` into two independent forward-direction `LSTM`
//!   nodes (one fed the reversed sequence) and recombines their outputs.

use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::DType;

use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::{AttributeProto, GraphProto, NodeProto, TensorProto};

/// Whether a rewrite function actually rewrote its node in place, so
/// [`rewrite_unsupported_ops`] knows whether to also keep the original
/// node or discard it in favor of the rewrite's replacement(s).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rewritten {
    Yes,
    No,
}

/// Rewrites every node in `graph` that unmodified `crate::onnx::eval`
/// handles incorrectly into a decomposition it handles correctly. Runs
/// once, at model load time, before the graph (or its segments) is passed
/// to `crate::onnx::simple_eval`.
pub(super) fn rewrite_unsupported_ops(graph: &mut GraphProto) -> Result<()> {
    let constant_tensors = collect_constant_tensors(graph);
    let constants = collect_constant_i64_values(&constant_tensors);
    let orig_nodes = std::mem::take(&mut graph.node);
    let mut new_nodes = Vec::with_capacity(orig_nodes.len());

    for node in &orig_nodes {
        match node.op_type.as_str() {
            "Trilu" => expand_trilu(node, graph, &mut new_nodes)?,
            "ReduceSum" => {
                if let Rewritten::No =
                    fix_reduce_sum_negative_axes(node, graph, &constants, &mut new_nodes)
                {
                    new_nodes.push(node.clone());
                }
            },
            "ReduceMean" => {
                if let Rewritten::No = fix_reduce_mean_axes_input(node, &constants, &mut new_nodes)
                {
                    new_nodes.push(node.clone());
                }
            },
            "CumSum" => fix_int_cumsum(node, &mut new_nodes),
            "LSTM" if is_bidirectional(node) => {
                if let Rewritten::No = expand_bidirectional_lstm(node, &mut new_nodes) {
                    new_nodes.push(node.clone());
                }
            },
            _ => new_nodes.push(node.clone()),
        }
    }

    graph.node = new_nodes;
    Ok(())
}

/// Collects every graph initializer and `Constant`-node output's full
/// `TensorProto`, keyed by name. Must run before `rewrite_unsupported_ops`
/// takes `graph.node` out of the graph to build its replacement node list,
/// since this scans the original (pre-rewrite) node list for `Constant`
/// nodes.
fn collect_constant_tensors(graph: &GraphProto) -> HashMap<String, TensorProto> {
    let mut constants = HashMap::new();
    for initializer in &graph.initializer {
        constants.insert(initializer.name.clone(), initializer.clone());
    }
    for node in &graph.node {
        if node.op_type != "Constant" || node.output.len() != 1 {
            continue;
        }
        let Some(value_attr) = node.attribute.iter().find(|attr| attr.name == "value") else {
            continue;
        };
        let Some(tensor_proto) = &value_attr.t else {
            continue;
        };
        constants.insert(node.output[0].clone(), tensor_proto.clone());
    }
    constants
}

/// Decodes every constant tensor in `constant_tensors` that holds an
/// integer type, flattened to `Vec<i64>` via
/// [`crate::onnx::eval::get_tensor`] (which also upcasts `Int32` to `i64`).
/// Used to resolve small integer inputs (like `axes`) to compile-time
/// constants when possible, so rewrites can fall back to a (larger, but
/// universally correct) dynamic subgraph only when genuinely needed.
fn collect_constant_i64_values(
    constant_tensors: &HashMap<String, TensorProto>,
) -> HashMap<String, Vec<i64>> {
    constant_tensors
        .iter()
        .filter_map(|(name, tensor_proto)| {
            tensor_proto_to_i64_vec(tensor_proto, name).map(|values| (name.clone(), values))
        })
        .collect()
}

/// Decodes `tensor_proto` and flattens it to a `Vec<i64>`, or `None` if it
/// can't be decoded or isn't an integer tensor.
fn tensor_proto_to_i64_vec(tensor_proto: &TensorProto, name: &str) -> Option<Vec<i64>> {
    let tensor = crate::onnx::eval::get_tensor(tensor_proto, name).ok()?;
    let tensor = tensor.flatten_all().ok()?;
    tensor.to_dtype(DType::I64).ok()?.to_vec1::<i64>().ok()
}

/// Rewrites `node` (a `Trilu`) in place into a small subgraph that avoids
/// the NaN `crate::onnx::eval`'s `Trilu` produces via `data * mask` when
/// `data` contains `+/-inf` and a masked-out entry is 0.
///
/// Instead of multiplying, the rewrite selects between `data` and a
/// same-dtype zero tensor with `Where`, driven by a 0/1 mask computed by
/// applying the *same* `Trilu` node — diagonal offset (`k`) and `upper`
/// attribute preserved verbatim, including a non-constant `k` input, since
/// nothing here needs `k`'s value at rewrite time — to an all-ones tensor
/// of `data`'s shape. `Trilu` applied to an all-ones/all-finite tensor can
/// never produce NaN, since it has no infinities to mishandle. Since the
/// mask is computed by `eval.rs`'s own unmodified `Trilu`, this rewrite
/// doesn't change (or fix) that op's existing rank-2-only mask broadcast —
/// a batched (rank > 2) `Trilu` input already fails in unmodified
/// `eval.rs` today, NaN or not, and continues to fail the same way here.
///
/// The zero-fill's data type is read from `graph.value_info` when `data`'s
/// declared type is present there; falls back to `Float` otherwise
/// (matching `eval.rs`'s own float-only defaults elsewhere, e.g.
/// `ConstantOfShape`'s default `value`). See [`scalar_value_attribute`] for
/// which dtypes are natively represented.
fn expand_trilu(node: &NodeProto, graph: &GraphProto, new_nodes: &mut Vec<NodeProto>) -> Result<()> {
    let data = node
        .input
        .first()
        .cloned()
        .context("Trilu node has no data input")?;
    let output = node
        .output
        .first()
        .cloned()
        .context("Trilu node has no output")?;

    let shape_name = format!("{output}__onnx_compat_shape");
    new_nodes.push(unary_node("Shape", &data, &shape_name));

    let ones_name = format!("{output}__onnx_compat_ones");
    new_nodes.push(constant_of_shape_node(
        &shape_name,
        &ones_name,
        scalar_value_attribute(DataType::Float as i32, 1.0),
    ));

    let mask_name = format!("{output}__onnx_compat_mask");
    let mut mask_node = node.clone();
    mask_node.input[0] = ones_name;
    mask_node.output = vec![mask_name.clone()];
    new_nodes.push(mask_node);

    let mask_bool_name = format!("{output}__onnx_compat_mask_bool");
    new_nodes.push(cast_node(&mask_name, &mask_bool_name, DataType::Bool));

    let data_dtype = declared_dtype(graph, &data).unwrap_or(DataType::Float as i32);
    let zero_name = format!("{output}__onnx_compat_zero");
    new_nodes.push(constant_of_shape_node(
        &shape_name,
        &zero_name,
        scalar_value_attribute(data_dtype, 0.0),
    ));

    new_nodes.push(NodeProto {
        name: node.name.clone(),
        op_type: "Where".to_string(),
        input: vec![mask_bool_name, data, zero_name],
        output: vec![output],
        ..Default::default()
    });

    Ok(())
}

/// Looks up `name`'s declared tensor type among `graph`'s `value_info`,
/// `input`, and `output` lists.
fn tensor_type_info<'a>(
    graph: &'a GraphProto,
    name: &str,
) -> Option<&'a crate::onnx::proto::type_proto::Tensor> {
    graph
        .value_info
        .iter()
        .chain(graph.input.iter())
        .chain(graph.output.iter())
        .find(|value_info| value_info.name == name)
        .and_then(|value_info| value_info.r#type.as_ref())
        .and_then(|type_proto| type_proto.value.as_ref())
        .and_then(|value| match value {
            crate::onnx::proto::type_proto::Value::TensorType(tensor_type) => Some(tensor_type),
            _ => None,
        })
}

/// Looks up `name`'s declared ONNX element type (as a raw `TensorProto`
/// `DataType` value) among `graph`'s `value_info`, `input`, and `output`
/// lists. Returns `None` when no declaration is present, which is common —
/// exporters often only populate `value_info` for a subset of tensors.
fn declared_dtype(graph: &GraphProto, name: &str) -> Option<i32> {
    tensor_type_info(graph, name).map(|tensor_type| tensor_type.elem_type)
}

/// Looks up `name`'s declared rank (number of dimensions) among `graph`'s
/// `value_info`, `input`, and `output` lists. Returns `None` when no
/// declaration is present, or the declaration has no shape at all.
fn declared_rank(graph: &GraphProto, name: &str) -> Option<usize> {
    tensor_type_info(graph, name)?
        .shape
        .as_ref()
        .map(|shape| shape.dim.len())
}

/// Rewrites a `ReduceSum` node whose `axes` input may contain negative
/// values into one whose `axes` input is guaranteed non-negative, working
/// around `eval.rs`'s `ReduceSum` casting axes directly via `x as usize`
/// with no negative-axis normalization at all (unlike `ReduceMean`, which
/// does normalize negative axes, but only in its older attribute form).
///
/// When `axes` resolves to a compile-time constant *and* `data`'s rank is
/// declared in `graph.value_info`, normalizes `axes` directly and swaps in
/// a replacement `Constant` node — cheaper, and produces a smaller graph.
/// Otherwise (axes computed dynamically elsewhere in the graph, or a
/// constant axes value whose rank isn't declared) emits a small subgraph
/// that normalizes at runtime: `fixed = axes < 0 ? axes + rank : axes`,
/// using `Size(Shape(data))` for `rank` — this covers every case, just
/// with a larger graph than the constant-resolution path needs.
///
/// Returns [`Rewritten::No`] (leaving `node` untouched) when `axes` is
/// absent, empty, or already proven non-negative by a resolved constant.
fn fix_reduce_sum_negative_axes(
    node: &NodeProto,
    graph: &GraphProto,
    constants: &HashMap<String, Vec<i64>>,
    new_nodes: &mut Vec<NodeProto>,
) -> Rewritten {
    let Some(axes_name) = node.input.get(1).filter(|name| !name.is_empty()) else {
        return Rewritten::No;
    };
    let data = &node.input[0];
    let output = &node.output[0];

    if let Some(axes_values) = constants.get(axes_name) {
        if axes_values.iter().all(|&axis| axis >= 0) {
            return Rewritten::No;
        }
        if let Some(rank) = declared_rank(graph, data) {
            #[allow(clippy::cast_possible_wrap)]
            let rank = rank as i64;
            let normalized = axes_values
                .iter()
                .map(|&axis| if axis < 0 { axis + rank } else { axis })
                .collect::<Vec<_>>();
            let fixed_axes_name = format!("{output}__onnx_compat_axes_fixed");
            #[allow(clippy::cast_possible_wrap)]
            let dims = vec![normalized.len() as i64];
            new_nodes.push(int64_constant_node(&fixed_axes_name, dims, normalized));
            let mut rewritten = node.clone();
            rewritten.input[1] = fixed_axes_name;
            new_nodes.push(rewritten);
            return Rewritten::Yes;
        }
    }

    let fixed_axes_name =
        push_dynamic_axis_normalization(data, axes_name, output, "axes", new_nodes);
    let mut rewritten = node.clone();
    rewritten.input[1] = fixed_axes_name;
    new_nodes.push(rewritten);
    Rewritten::Yes
}

/// Pushes nodes computing `axis < 0 ? axis + rank : axis` at runtime —
/// `rank` via `Size(Shape(data))` — and returns the name of the resulting
/// non-negative axis/axes tensor. Shared by every rewrite that needs to
/// normalize a negative axis input dynamically, since `data`'s rank isn't
/// known until the graph actually runs.
///
/// `label` only distinguishes this call site's intermediate tensor names
/// from another rewrite's in the same graph (e.g. `"axes"` vs `"axis"`);
/// it has no effect on the computation.
fn push_dynamic_axis_normalization(
    data: &str,
    axis_name: &str,
    output: &str,
    label: &str,
    new_nodes: &mut Vec<NodeProto>,
) -> String {
    let zero_name = format!("{output}__onnx_compat_zero_{label}");
    new_nodes.push(int64_constant_node(&zero_name, vec![], vec![0]));

    let shape_name = format!("{output}__onnx_compat_data_shape_{label}");
    new_nodes.push(unary_node("Shape", data, &shape_name));

    let rank_name = format!("{output}__onnx_compat_rank_{label}");
    new_nodes.push(unary_node("Size", &shape_name, &rank_name));

    let is_negative_name = format!("{output}__onnx_compat_{label}_negative");
    new_nodes.push(binary_node("Less", axis_name, &zero_name, &is_negative_name));

    let adjusted_name = format!("{output}__onnx_compat_{label}_adjusted");
    new_nodes.push(binary_node("Add", axis_name, &rank_name, &adjusted_name));

    let fixed_name = format!("{output}__onnx_compat_{label}_fixed");
    new_nodes.push(NodeProto {
        op_type: "Where".to_string(),
        input: vec![is_negative_name, adjusted_name, axis_name.to_string()],
        output: vec![fixed_name.clone()],
        ..Default::default()
    });
    fixed_name
}

/// Rewrites a `ReduceMean` node passing `axes` as an opset-18+ input into
/// the older attribute form, working around `eval.rs`'s `ReduceMean`
/// reading *only* the `axes` attribute and never an `axes` input at all —
/// an axes-as-input node silently reduces over every axis instead of the
/// intended ones.
///
/// Unlike [`fix_reduce_sum_negative_axes`], this can only resolve `axes`
/// to a compile-time constant: `eval.rs`'s `ReduceMean` has no path to
/// accept a non-attribute axes value, so a dynamic-subgraph fallback
/// (which still has to produce *some* form eval.rs reads) isn't possible
/// here — that's inherent to targeting unmodified `eval.rs`, not a bug in
/// this rewrite. Negative entries in `axes` don't need normalizing before
/// becoming the attribute: `eval.rs`'s `ReduceMean` already normalizes
/// negative axes in its attribute-reading path.
///
/// Known limitation inherited from `eval.rs`, not fixed here: `eval.rs`'s
/// `ReduceMean` never reads `noop_with_empty_axes` at all, so a node with
/// no axes input/attribute and `noop_with_empty_axes=1` still incorrectly
/// reduces every axis instead of being a no-op. Not fixed because doing so
/// means rewriting an absent/empty-axes node into an `Identity`, and
/// Kokoro's export always passes non-empty axes as an input for every one
/// of its `ReduceMean` nodes, so the gap doesn't manifest in practice.
///
/// Returns [`Rewritten::No`] (leaving `node` untouched) when `axes` is
/// absent, empty, or not a compile-time constant.
fn fix_reduce_mean_axes_input(
    node: &NodeProto,
    constants: &HashMap<String, Vec<i64>>,
    new_nodes: &mut Vec<NodeProto>,
) -> Rewritten {
    let Some(axes_name) = node.input.get(1).filter(|name| !name.is_empty()) else {
        return Rewritten::No;
    };
    let Some(axes_values) = constants.get(axes_name) else {
        return Rewritten::No;
    };

    let mut rewritten = node.clone();
    rewritten.input.truncate(1);
    rewritten.attribute.push(AttributeProto {
        name: "axes".to_string(),
        r#type: AttributeType::Ints as i32,
        ints: axes_values.clone(),
        ..Default::default()
    });
    new_nodes.push(rewritten);
    Rewritten::Yes
}

/// Rewrites every `CumSum` node to fix two independent gaps in `eval.rs`'s
/// `CumSum`, unconditionally (there's no "already fine" case worth
/// detecting, unlike this module's other rewrites):
///
/// - `candle_core::Tensor::cumsum` is a matmul-based implementation that
///   only supports floating-point dtypes, so an int64 `data` input fails
///   outright. Wrapping `data` in `Cast(to=Double)` before, and back to its
///   original dtype after, routes every dtype through the same working
///   float path. `Double` (not `Float`) matches the precision the dropped
///   `ops/cumsum.rs` implementation used, since `Float`/f32 loses exactness
///   above 2^24 — plausible for cumulative sums longer than a couple
///   thousand terms. The "back to its original dtype" cast uses
///   `CastLike(cumsum_out, data)` (`eval.rs`'s `CastLike`), not a static
///   `Cast(to=...)` resolved from `graph.value_info`: a real-world export
///   (e.g. `torch.onnx.export(dynamo=False)`) can leave `value_info`
///   completely empty, and a rewrite that guessed the dtype from it used to
///   silently leave the output at `Double` when the guess failed —
///   `CastLike` reads `data`'s actual runtime dtype instead, so there's
///   nothing to guess.
/// - `axis` is an `eval.rs` *input* (not an attribute), cast via
///   `to_dtype(DType::U32)` then to `usize` with no negative-axis
///   normalization at all — the same wraparound bug
///   [`fix_reduce_sum_negative_axes`] fixes for `ReduceSum`. Always
///   normalized dynamically via [`push_dynamic_axis_normalization`], since
///   (unlike `ReduceSum`) there's no meaningfully cheaper constant-
///   resolution path worth adding just for this.
///
/// `exclusive`/`reverse` attributes, if present, are preserved verbatim on
/// the rewritten node — this rewrite only touches `data` and `axis`.
fn fix_int_cumsum(node: &NodeProto, new_nodes: &mut Vec<NodeProto>) {
    let data = &node.input[0];
    let axis = &node.input[1];
    let output = &node.output[0];

    let cast_in_name = format!("{output}__onnx_compat_cumsum_in");
    new_nodes.push(cast_node(data, &cast_in_name, DataType::Double));

    let fixed_axis = push_dynamic_axis_normalization(data, axis, output, "axis", new_nodes);

    let cumsum_out_name = format!("{output}__onnx_compat_cumsum_out");
    let mut rewritten = node.clone();
    rewritten.input = vec![cast_in_name, fixed_axis];
    rewritten.output = vec![cumsum_out_name.clone()];
    new_nodes.push(rewritten);

    new_nodes.push(binary_node("CastLike", &cumsum_out_name, data, output));
}

/// Whether `node` (an `LSTM`) declares `direction == "bidirectional"`.
fn is_bidirectional(node: &NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "direction")
        .is_some_and(|attr| attr.s == b"bidirectional")
}

/// Splits a bidirectional `LSTM` node into two independent forward-
/// direction `LSTM` nodes — one processing the sequence normally, one
/// processing it reversed along the time axis — recombining their outputs
/// to match what a single bidirectional node would have produced.
///
/// `eval.rs`'s `LSTM` bails immediately on any `direction` other than
/// `"forward"`, so bidirectional support has to come from decomposing it
/// into ops `eval.rs` already runs correctly, same as this module's other
/// rewrites.
///
/// Unlike the earlier abandoned attempt at this rewrite, this doesn't
/// inspect `sequence_lens`/peephole (`P`)/`activations` values at rewrite
/// time to decide whether to bail. Splitting is *unconditionally*
/// structurally correct regardless of those values: `sequence_lens` isn't
/// direction-shaped, so it's forwarded to both split nodes unchanged;
/// `P` is direction-shaped, so it's sliced like `W`/`R`/`initial_h`/
/// `initial_c`; a present `activations` attribute (which must have exactly
/// 6 entries — two triples — for a bidirectional node) is split 3-and-3
/// between the two directions. Each split node is still a single-direction
/// `LSTM`, so `eval.rs`'s own existing runtime checks (`seq_lens_is_default`,
/// `p_is_zeros`, `activations != activations_default`) independently
/// accept or reject each direction's values exactly as they would for any
/// other single-direction `LSTM` node — nothing here needs to duplicate
/// that validation.
///
/// The one place this *does* need care: `Y`'s backward-direction output is
/// produced in time-reversed order (since its input was reversed) and must
/// be reversed back before concatenating with the forward direction. `Y_h`
/// and `Y_c` (the final hidden/cell states) must **not** be reversed —
/// each split node's final state already corresponds to having consumed
/// the *original* sequence's first timestep last (for the backward
/// direction), which is exactly the state ONNX's backward-direction output
/// is defined to be.
///
/// Returns [`Rewritten::No`] (leaving `node` untouched) only when
/// `activations` is present with a length other than 6 — a malformed
/// bidirectional `LSTM` this rewrite can't meaningfully split.
fn expand_bidirectional_lstm(node: &NodeProto, new_nodes: &mut Vec<NodeProto>) -> Rewritten {
    let activations = node.attribute.iter().find(|attr| attr.name == "activations");
    let (fwd_activations, bwd_activations) = match activations {
        None => (None, None),
        Some(attr) if attr.strings.len() == 6 => {
            (Some(attr.strings[0..3].to_vec()), Some(attr.strings[3..6].to_vec()))
        },
        Some(_) => return Rewritten::No,
    };

    let output = node.output.first().map_or("lstm", String::as_str);
    let x = node.input[0].clone();
    let x_reversed = reverse_along_axis(&x, 0, output, "x", new_nodes);

    let mut fwd_inputs = vec![x];
    let mut bwd_inputs = vec![x_reversed];
    for (idx, name) in node.input.iter().enumerate().skip(1) {
        if idx == 4 {
            // sequence_lens isn't direction-shaped (it's per-batch-item,
            // shared across directions) -- forward unchanged.
            fwd_inputs.push(name.clone());
            bwd_inputs.push(name.clone());
        } else if name.is_empty() {
            fwd_inputs.push(String::new());
            bwd_inputs.push(String::new());
        } else {
            fwd_inputs.push(slice_direction(name, 0, idx, output, new_nodes));
            bwd_inputs.push(slice_direction(name, 1, idx, output, new_nodes));
        }
    }

    let mut fwd_attributes = base_lstm_attributes(node);
    if let Some(activations) = fwd_activations {
        fwd_attributes.push(strings_attribute("activations", activations));
    }
    let mut bwd_attributes = base_lstm_attributes(node);
    if let Some(activations) = bwd_activations {
        bwd_attributes.push(strings_attribute("activations", activations));
    }

    let fwd_outputs = node
        .output
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if name.is_empty() {
                String::new()
            } else {
                format!("{output}__onnx_compat_lstm_fwd_out{idx}")
            }
        })
        .collect::<Vec<_>>();
    let bwd_outputs = node
        .output
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if name.is_empty() {
                String::new()
            } else {
                format!("{output}__onnx_compat_lstm_bwd_out{idx}")
            }
        })
        .collect::<Vec<_>>();

    new_nodes.push(NodeProto {
        name: format!("{}/onnx_compat_fwd", node.name),
        op_type: "LSTM".to_string(),
        input: fwd_inputs,
        output: fwd_outputs.clone(),
        attribute: fwd_attributes,
        ..Default::default()
    });
    new_nodes.push(NodeProto {
        name: format!("{}/onnx_compat_bwd", node.name),
        op_type: "LSTM".to_string(),
        input: bwd_inputs,
        output: bwd_outputs.clone(),
        attribute: bwd_attributes,
        ..Default::default()
    });

    if let Some(y_name) = node.output.first().filter(|name| !name.is_empty()) {
        let bwd_y_reversed = reverse_along_axis(&bwd_outputs[0], 0, output, "y_bwd", new_nodes);
        new_nodes.push(NodeProto {
            op_type: "Concat".to_string(),
            input: vec![fwd_outputs[0].clone(), bwd_y_reversed],
            output: vec![y_name.clone()],
            attribute: vec![axis_attribute(1)],
            ..Default::default()
        });
    }
    for idx in 1..=2 {
        if let Some(name) = node.output.get(idx).filter(|name| !name.is_empty()) {
            new_nodes.push(NodeProto {
                op_type: "Concat".to_string(),
                input: vec![fwd_outputs[idx].clone(), bwd_outputs[idx].clone()],
                output: vec![name.clone()],
                attribute: vec![axis_attribute(0)],
                ..Default::default()
            });
        }
    }

    Rewritten::Yes
}

/// Copies every attribute from an `LSTM` node except `direction` (the
/// split nodes are implicitly forward-direction, `eval.rs`'s default) and
/// `activations` (split 3-and-3 between directions by the caller).
fn base_lstm_attributes(node: &NodeProto) -> Vec<AttributeProto> {
    node.attribute
        .iter()
        .filter(|attr| attr.name != "direction" && attr.name != "activations")
        .cloned()
        .collect()
}

/// Builds an `Ints`-free `axis` `INT` attribute (as used by `Concat`).
fn axis_attribute(axis: i64) -> AttributeProto {
    AttributeProto {
        name: "axis".to_string(),
        r#type: AttributeType::Int as i32,
        i: axis,
        ..Default::default()
    }
}

/// Builds a `STRINGS` attribute.
fn strings_attribute(name: &str, values: Vec<Vec<u8>>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Strings as i32,
        strings: values,
        ..Default::default()
    }
}

/// Pushes a `Gather(name, [index], axis=0)` node selecting direction
/// `index`'s slice (keeping the leading dimension, at size 1) from a
/// `[num_directions, ...]`-shaped `LSTM` input, returning the new tensor's
/// name.
///
/// `input_idx` is the sliced input's ordinal position among the `LSTM`
/// node's own inputs (e.g. 5 for `initial_h`, 6 for `initial_c`) and is
/// used, not `name`, to keep the generated node names unique: `initial_h`
/// and `initial_c` are both zero-initialized with the same shape in
/// Kokoro's export, so they can share one `ConstantOfShape` tensor as
/// `name` — keying on `name` alone would then emit two `Constant`/`Gather`
/// pairs with identical output names, an SSA violation that surfaces
/// downstream as a "cannot find output" error when both copies are
/// requested as a segment output.
fn slice_direction(
    name: &str,
    index: i64,
    input_idx: usize,
    output: &str,
    new_nodes: &mut Vec<NodeProto>,
) -> String {
    let indices_name = format!("{output}__onnx_compat_lstm_dir{index}_indices_in{input_idx}");
    new_nodes.push(int64_constant_node(&indices_name, vec![1], vec![index]));
    let sliced_name = format!("{output}__onnx_compat_lstm_dir{index}_in{input_idx}");
    new_nodes.push(NodeProto {
        op_type: "Gather".to_string(),
        input: vec![name.to_string(), indices_name],
        output: vec![sliced_name.clone()],
        attribute: vec![axis_attribute(0)],
        ..Default::default()
    });
    sliced_name
}

/// Pushes a `Slice` node reversing `name` along `axis`, returning the new
/// tensor's name. Uses the standard ONNX "reverse an entire axis" idiom:
/// `starts=[-1]` (the last index), `ends=[i64::MIN]` (a sentinel that,
/// even after `eval.rs`'s negative-value normalization, clamps to exactly
/// one-before-the-first index), `steps=[-1]`.
fn reverse_along_axis(
    name: &str,
    axis: i64,
    output: &str,
    label: &str,
    new_nodes: &mut Vec<NodeProto>,
) -> String {
    let starts_name = format!("{output}__onnx_compat_{label}_rev_starts");
    new_nodes.push(int64_constant_node(&starts_name, vec![1], vec![-1]));
    let ends_name = format!("{output}__onnx_compat_{label}_rev_ends");
    new_nodes.push(int64_constant_node(&ends_name, vec![1], vec![i64::MIN]));
    let axes_name = format!("{output}__onnx_compat_{label}_rev_axes");
    new_nodes.push(int64_constant_node(&axes_name, vec![1], vec![axis]));
    let steps_name = format!("{output}__onnx_compat_{label}_rev_steps");
    new_nodes.push(int64_constant_node(&steps_name, vec![1], vec![-1]));

    let reversed_name = format!("{output}__onnx_compat_{label}_reversed");
    new_nodes.push(NodeProto {
        op_type: "Slice".to_string(),
        input: vec![
            name.to_string(),
            starts_name,
            ends_name,
            axes_name,
            steps_name,
        ],
        output: vec![reversed_name.clone()],
        ..Default::default()
    });
    reversed_name
}

/// Builds a single-input, single-output node with no attributes.
fn unary_node(op_type: &str, input: &str, output: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        ..Default::default()
    }
}

/// Builds a two-input, single-output node with no attributes.
fn binary_node(op_type: &str, a: &str, b: &str, output: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: vec![a.to_string(), b.to_string()],
        output: vec![output.to_string()],
        ..Default::default()
    }
}

/// Builds a `Constant` node holding an `int64` tensor. Unlike
/// [`scalar_value_attribute`] (used for `ConstantOfShape`'s `"value"`,
/// which `eval.rs` decodes via a `raw_data`-only path), `eval.rs`'s
/// `Constant` handler decodes its `"value"` attribute with
/// [`crate::onnx::eval::get_tensor`], which reads the type-specific
/// `int64_data` field directly.
fn int64_constant_node(output: &str, dims: Vec<i64>, values: Vec<i64>) -> NodeProto {
    NodeProto {
        op_type: "Constant".to_string(),
        output: vec![output.to_string()],
        attribute: vec![AttributeProto {
            name: "value".to_string(),
            r#type: AttributeType::Tensor as i32,
            t: Some(TensorProto {
                data_type: DataType::Int64 as i32,
                dims,
                int64_data: values,
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Builds a `ConstantOfShape` node reading its shape from `shape_input`
/// and filling with `value` (the node's `"value"` attribute).
fn constant_of_shape_node(shape_input: &str, output: &str, value: AttributeProto) -> NodeProto {
    NodeProto {
        op_type: "ConstantOfShape".to_string(),
        input: vec![shape_input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![value],
        ..Default::default()
    }
}

/// Builds a `Cast` node converting `input` to ONNX element type `to`.
fn cast_node(input: &str, output: &str, to: DataType) -> NodeProto {
    NodeProto {
        op_type: "Cast".to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![AttributeProto {
            name: "to".to_string(),
            r#type: AttributeType::Int as i32,
            i: to as i64,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Builds a scalar-tensor `"value"` attribute (as used by
/// `ConstantOfShape`) holding `value` encoded as `data_type`.
///
/// `eval.rs`'s `TENSOR`-attribute decoder only reads a tensor's `raw_data`
/// bytes (never the type-specific `float_data`/`int32_data`/`int64_data`
/// fields `crate::onnx::eval::get_tensor` accepts for initializers), and
/// has no `Int32` case at all — so a declared `Int32` dtype is represented
/// here as `Int64` instead, matching how `get_tensor` already promotes
/// ONNX `Int32` *tensors* (as opposed to this *attribute* path) to
/// candle's `I64` dtype elsewhere in this evaluator. Any other declared
/// type falls back to `Float`, since `value` only ever needs to
/// distinguish "zero" from "one" for the rewrites in this module.
fn scalar_value_attribute(data_type: i32, value: f64) -> AttributeProto {
    let (resolved_type, raw_data) = match DataType::try_from(data_type) {
        Ok(DataType::Int64 | DataType::Int32) => {
            #[allow(clippy::cast_possible_truncation)]
            let bytes = (value as i64).to_le_bytes().to_vec();
            (DataType::Int64, bytes)
        },
        Ok(DataType::Double) => (DataType::Double, value.to_le_bytes().to_vec()),
        _ => {
            #[allow(clippy::cast_possible_truncation)]
            let bytes = (value as f32).to_le_bytes().to_vec();
            (DataType::Float, bytes)
        },
    };
    AttributeProto {
        name: "value".to_string(),
        r#type: AttributeType::Tensor as i32,
        t: Some(TensorProto {
            data_type: resolved_type as i32,
            raw_data,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::onnx::proto::{ModelProto, ValueInfoProto, type_proto};

    fn trilu_node(data: &str, k: Option<&str>, upper: i64, output: &str) -> NodeProto {
        let mut input = vec![data.to_string()];
        if let Some(k) = k {
            input.push(k.to_string());
        }
        NodeProto {
            op_type: "Trilu".to_string(),
            input,
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "upper".to_string(),
                r#type: AttributeType::Int as i32,
                i: upper,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn run_graph(graph: GraphProto, inputs: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        crate::onnx::simple_eval(&model, inputs).expect("simple_eval should succeed")
    }

    fn declare_tensor_type(name: &str, elem_type: DataType) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(crate::onnx::proto::TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: elem_type as i32,
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    // Upper-triangular Trilu on a +/-inf input must zero the masked-out
    // lower triangle instead of producing NaN via `-inf * 0`.
    #[test]
    fn trilu_upper_zeroes_masked_entries_without_nan() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", None, 1, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(
            vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 2.0],
            (2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![f32::INFINITY, 1.0], vec![0.0, 2.0]]);
    }

    // Lower-triangular Trilu (upper=0) must keep the diagonal and below,
    // zeroing the strictly-upper entries.
    #[test]
    fn trilu_lower_zeroes_masked_entries_without_nan() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", None, 0, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(
            vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 2.0],
            (2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![1.0, 0.0], vec![f32::NEG_INFINITY, 2.0]]);
    }

    // A non-zero diagonal offset `k` (here as a graph input, i.e. not
    // resolvable at rewrite time) must still be honored, since the
    // rewrite forwards `k` verbatim into the mask's Trilu instead of
    // requiring it to be a compile-time constant.
    #[test]
    fn trilu_diagonal_offset_from_dynamic_input_is_honored() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", Some("k"), 1, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(
            vec![f32::INFINITY, 1.0, 2.0, f32::NEG_INFINITY, 3.0, 4.0],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("k".to_string(), Tensor::new(1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        // upper=1, k=1 keeps j >= i+1: row0 keeps col>=1, row1 keeps col>=2.
        assert_eq!(out, vec![vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 4.0]]);
    }

    // The motivating fix: a non-f32 (int32) Trilu input must not have its
    // zero-fill hardcoded to Float, which would produce a dtype-mismatched
    // `Where` node. `data`'s declared type comes from `graph.value_info`.
    #[test]
    fn trilu_non_f32_input_uses_declared_dtype_for_zero_fill() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", None, 0, "out")],
            value_info: vec![declare_tensor_type("data", DataType::Int32)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(vec![5i64, 6, 7, 8], (2, 2), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::I64);
        assert_eq!(out.to_vec2::<i64>().unwrap(), vec![vec![5, 0], vec![7, 8]]);
    }

    // A graph with no Trilu nodes must pass through unchanged.
    #[test]
    fn rewrite_is_a_no_op_without_trilu_nodes() {
        let mut graph = GraphProto {
            node: vec![unary_node("Identity", "x", "y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Identity");
    }

    fn reduce_sum_node(data: &str, axes: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: "ReduceSum".to_string(),
            input: vec![data.to_string(), axes.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn int64_initializer(name: &str, dims: Vec<i64>, values: Vec<i64>) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Int64 as i32,
            dims,
            int64_data: values,
            ..Default::default()
        }
    }

    fn declare_rank(name: &str, rank: usize) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(crate::onnx::proto::TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DataType::Float as i32,
                    shape: Some(crate::onnx::proto::TensorShapeProto {
                        dim: vec![Default::default(); rank],
                    }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn data_2x3() -> Tensor {
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap()
    }

    // Kokoro's actual case: a constant negative axis (`[-1]`), with data's
    // rank declared in value_info, takes the cheap constant-resolution
    // path (a replacement `Constant` node, not a dynamic subgraph).
    #[test]
    fn reduce_sum_constant_negative_axis_with_declared_rank_uses_constant_path() {
        let mut graph = GraphProto {
            node: vec![reduce_sum_node("data", "axes", "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![-1])],
            value_info: vec![declare_rank("data", 2)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        // The cheap path swaps in a `Constant` node instead of the dynamic
        // Shape/Size/Less/Add/Where subgraph.
        assert!(
            graph
                .node
                .iter()
                .any(|n| n.op_type == "Constant" && !n.output.is_empty())
        );
        assert!(!graph.node.iter().any(|n| n.op_type == "Shape"));

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![6.0], vec![15.0]]);
    }

    // A constant, already non-negative axis must be left untouched.
    #[test]
    fn reduce_sum_constant_positive_axis_is_a_no_op() {
        let mut graph = GraphProto {
            node: vec![reduce_sum_node("data", "axes", "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![1])],
            value_info: vec![declare_rank("data", 2)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "ReduceSum");
        assert_eq!(graph.node[0].input[1], "axes");
    }

    // A dynamic (non-constant) axes input containing a negative value must
    // still be normalized correctly, proving the runtime Shape/Size/Less/
    // Add/Where subgraph — not just the constant-resolution path — works.
    #[test]
    fn reduce_sum_dynamic_negative_axis_is_normalized_at_runtime() {
        let mut graph = GraphProto {
            node: vec![reduce_sum_node("data", "axes", "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        // No declared rank and no constant axes forces the dynamic path.
        assert!(graph.node.iter().any(|n| n.op_type == "Shape"));

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        inputs.insert(
            "axes".to_string(),
            Tensor::from_vec(vec![-1i64], 1, &Device::Cpu).unwrap(),
        );
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![6.0], vec![15.0]]);
    }

    fn reduce_mean_node(data: &str, axes: Option<&str>, output: &str) -> NodeProto {
        let mut input = vec![data.to_string()];
        if let Some(axes) = axes {
            input.push(axes.to_string());
        }
        NodeProto {
            op_type: "ReduceMean".to_string(),
            input,
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    // Kokoro's actual case: a constant positive axes input must be
    // converted to the attribute form eval.rs's ReduceMean actually reads.
    #[test]
    fn reduce_mean_constant_positive_axis_input_becomes_attribute() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", Some("axes"), "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![1])],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input.len(), 1);
        assert!(graph.node[0].attribute.iter().any(|a| a.name == "axes"));

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![2.0], vec![5.0]]);
    }

    // A constant negative axes input must also convert correctly — eval.rs
    // already normalizes negative axes once they're in the attribute form.
    #[test]
    fn reduce_mean_constant_negative_axis_input_becomes_attribute() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", Some("axes"), "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![-1])],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![2.0], vec![5.0]]);
    }

    // A ReduceMean with no axes input at all (attribute-only form, or a
    // deliberate full reduction) must be left untouched.
    #[test]
    fn reduce_mean_without_axes_input_is_a_no_op() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", None, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input.len(), 1);
        assert!(graph.node[0].attribute.is_empty());
    }

    // A non-constant axes input can't be resolved at rewrite time, since
    // eval.rs's ReduceMean has no path to accept axes as anything but an
    // attribute — the node must be left untouched (conservative, documented
    // limitation) rather than guessing.
    #[test]
    fn reduce_mean_non_constant_axes_input_is_a_no_op() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", Some("axes"), "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input, vec!["data".to_string(), "axes".to_string()]);
    }

    fn cumsum_node(
        data: &str,
        axis: &str,
        output: &str,
        exclusive: Option<i64>,
        reverse: Option<i64>,
    ) -> NodeProto {
        let mut attribute = vec![];
        if let Some(value) = exclusive {
            attribute.push(AttributeProto {
                name: "exclusive".to_string(),
                r#type: AttributeType::Int as i32,
                i: value,
                ..Default::default()
            });
        }
        if let Some(value) = reverse {
            attribute.push(AttributeProto {
                name: "reverse".to_string(),
                r#type: AttributeType::Int as i32,
                i: value,
                ..Default::default()
            });
        }
        NodeProto {
            op_type: "CumSum".to_string(),
            input: vec![data.to_string(), axis.to_string()],
            output: vec![output.to_string()],
            attribute,
            ..Default::default()
        }
    }

    // Kokoro's actual case: int64 data with a negative axis. Both bugs
    // (float-only cumsum, un-normalized negative axis) are exercised at
    // once, and the output must come back as int64, not float.
    #[test]
    fn cumsum_int64_negative_axis_produces_correct_int64_output() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", None, None)],
            value_info: vec![declare_tensor_type("data", DataType::Int64)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(-1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::I64);
        assert_eq!(out.to_vec2::<i64>().unwrap(), vec![vec![1, 3, 6], vec![4, 9, 15]]);
    }

    // Float data with a positive axis must round-trip through the Double
    // intermediate cast without changing the result or the output dtype.
    #[test]
    fn cumsum_float_data_round_trips_through_double_precision() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", None, None)],
            value_info: vec![declare_tensor_type("data", DataType::Float)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::F32);
        assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![1.0, 3.0, 6.0], vec![
            4.0, 9.0, 15.0
        ]]);
    }

    // Real-world exports (e.g. `torch.onnx.export(dynamo=False)`) can leave
    // `graph.value_info` completely empty, so `data`'s dtype is never
    // statically declared anywhere. The old `declared_dtype`-guessing
    // rewrite silently left `out` as `Double` in exactly this case; the
    // `CastLike`-based rewrite must still recover the correct `F32` output
    // dtype from `data`'s actual runtime value.
    #[test]
    fn cumsum_float_data_without_declared_dtype_still_casts_back_to_f32() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", None, None)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::F32);
        assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![1.0, 3.0, 6.0], vec![
            4.0, 9.0, 15.0
        ]]);
    }

    // An `exclusive`/`reverse` attribute must survive the rewrite verbatim
    // rather than being silently dropped — eval.rs still rejects
    // `exclusive != 0` explicitly, so the error proves the attribute made
    // it onto the rewritten node instead of being lost.
    #[test]
    fn cumsum_exclusive_attribute_is_preserved_and_still_rejected() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", Some(1), None)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(0i64, &Device::Cpu).unwrap());

        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let err = crate::onnx::simple_eval(&model, inputs).unwrap_err();
        assert!(err.to_string().contains("exclusive"));
    }

    const LSTM_HIDDEN: usize = 2;
    const LSTM_INPUT: usize = 2;
    const LSTM_SEQ_LEN: usize = 3;

    fn out(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn lstm_node(inputs: &[&str], outputs: &[&str], bidirectional: bool) -> NodeProto {
        let mut attribute = vec![AttributeProto {
            name: "hidden_size".to_string(),
            r#type: AttributeType::Int as i32,
            #[allow(clippy::cast_possible_wrap)]
            i: LSTM_HIDDEN as i64,
            ..Default::default()
        }];
        if bidirectional {
            attribute.push(AttributeProto {
                name: "direction".to_string(),
                r#type: AttributeType::String as i32,
                s: b"bidirectional".to_vec(),
                ..Default::default()
            });
        }
        NodeProto {
            op_type: "LSTM".to_string(),
            input: inputs.iter().map(|s| (*s).to_string()).collect(),
            output: outputs.iter().map(|s| (*s).to_string()).collect(),
            attribute,
            ..Default::default()
        }
    }

    // Deterministic, non-symmetric pseudo-random floats -- small enough to
    // avoid sigmoid/tanh saturation, distinct enough between calls (via
    // `seed`) that forward and backward weights can never accidentally
    // produce identical results.
    fn deterministic_values(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.7 + seed).sin() * 0.3).collect()
    }

    fn reverse_time_major(values: &[f32], step: usize) -> Vec<f32> {
        let mut chunks: Vec<&[f32]> = values.chunks(step).collect();
        chunks.reverse();
        chunks.concat()
    }

    // The motivating fix: eval.rs's LSTM bails outright on
    // direction=="bidirectional". A bidirectional LSTM's Y must match
    // concatenating an independent forward pass over the original sequence
    // with an independent forward pass over the reversed sequence (itself
    // reversed back), and Y_h/Y_c must match those two passes' final
    // states *without* reversing them -- the classic bidirectional-LSTM
    // gotcha this rewrite has to get right.
    #[test]
    fn bidirectional_lstm_matches_two_independent_forward_passes() {
        let w_fwd = deterministic_values(4 * LSTM_HIDDEN * LSTM_INPUT, 0.0);
        let r_fwd = deterministic_values(4 * LSTM_HIDDEN * LSTM_HIDDEN, 1.0);
        let b_fwd = deterministic_values(8 * LSTM_HIDDEN, 2.0);
        let w_bwd = deterministic_values(4 * LSTM_HIDDEN * LSTM_INPUT, 3.0);
        let r_bwd = deterministic_values(4 * LSTM_HIDDEN * LSTM_HIDDEN, 4.0);
        let b_bwd = deterministic_values(8 * LSTM_HIDDEN, 5.0);

        let x_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_vec(x_data.clone(), (LSTM_SEQ_LEN, 1, LSTM_INPUT), &Device::Cpu).unwrap();
        let x_rev_data = reverse_time_major(&x_data, LSTM_INPUT);
        let x_rev =
            Tensor::from_vec(x_rev_data, (LSTM_SEQ_LEN, 1, LSTM_INPUT), &Device::Cpu).unwrap();

        // Reference pass 1: plain forward LSTM over the original sequence.
        let mut fwd_graph = GraphProto {
            node: vec![lstm_node(&["x", "w", "r", "b"], &["y", "y_h", "y_c"], false)],
            output: vec![out("y"), out("y_h"), out("y_c")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut fwd_graph).unwrap();
        let mut fwd_inputs = HashMap::new();
        fwd_inputs.insert("x".to_string(), x.clone());
        fwd_inputs.insert(
            "w".to_string(),
            Tensor::from_vec(w_fwd.clone(), (1, 4 * LSTM_HIDDEN, LSTM_INPUT), &Device::Cpu)
                .unwrap(),
        );
        fwd_inputs.insert(
            "r".to_string(),
            Tensor::from_vec(r_fwd.clone(), (1, 4 * LSTM_HIDDEN, LSTM_HIDDEN), &Device::Cpu)
                .unwrap(),
        );
        fwd_inputs.insert(
            "b".to_string(),
            Tensor::from_vec(b_fwd.clone(), (1, 8 * LSTM_HIDDEN), &Device::Cpu).unwrap(),
        );
        let fwd_ref = run_graph(fwd_graph, fwd_inputs);

        // Reference pass 2: plain forward LSTM over the reversed sequence.
        let mut bwd_graph = GraphProto {
            node: vec![lstm_node(&["x", "w", "r", "b"], &["y", "y_h", "y_c"], false)],
            output: vec![out("y"), out("y_h"), out("y_c")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut bwd_graph).unwrap();
        let mut bwd_inputs = HashMap::new();
        bwd_inputs.insert("x".to_string(), x_rev);
        bwd_inputs.insert(
            "w".to_string(),
            Tensor::from_vec(w_bwd.clone(), (1, 4 * LSTM_HIDDEN, LSTM_INPUT), &Device::Cpu)
                .unwrap(),
        );
        bwd_inputs.insert(
            "r".to_string(),
            Tensor::from_vec(r_bwd.clone(), (1, 4 * LSTM_HIDDEN, LSTM_HIDDEN), &Device::Cpu)
                .unwrap(),
        );
        bwd_inputs.insert(
            "b".to_string(),
            Tensor::from_vec(b_bwd.clone(), (1, 8 * LSTM_HIDDEN), &Device::Cpu).unwrap(),
        );
        let bwd_ref = run_graph(bwd_graph, bwd_inputs);

        let fwd_y = fwd_ref.get("y").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bwd_y = bwd_ref.get("y").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bwd_y_reversed = reverse_time_major(&bwd_y, LSTM_HIDDEN);
        let mut expected_y = Vec::new();
        for t in 0..LSTM_SEQ_LEN {
            expected_y.extend_from_slice(&fwd_y[t * LSTM_HIDDEN..(t + 1) * LSTM_HIDDEN]);
            expected_y.extend_from_slice(&bwd_y_reversed[t * LSTM_HIDDEN..(t + 1) * LSTM_HIDDEN]);
        }

        let mut expected_y_h =
            fwd_ref.get("y_h").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        expected_y_h
            .extend(bwd_ref.get("y_h").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let mut expected_y_c =
            fwd_ref.get("y_c").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        expected_y_c
            .extend(bwd_ref.get("y_c").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap());

        // The bidirectional node under test: W/R/B concatenate the two
        // directions' weights along axis 0, exactly as the ONNX LSTM spec
        // requires.
        let mut graph = GraphProto {
            node: vec![lstm_node(&["x", "w", "r", "b"], &["y", "y_h", "y_c"], true)],
            output: vec![out("y"), out("y_h"), out("y_c")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();
        assert!(graph.node.iter().filter(|n| n.op_type == "LSTM").count() == 2);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);
        inputs.insert(
            "w".to_string(),
            Tensor::from_vec(
                [w_fwd, w_bwd].concat(),
                (2, 4 * LSTM_HIDDEN, LSTM_INPUT),
                &Device::Cpu,
            )
            .unwrap(),
        );
        inputs.insert(
            "r".to_string(),
            Tensor::from_vec(
                [r_fwd, r_bwd].concat(),
                (2, 4 * LSTM_HIDDEN, LSTM_HIDDEN),
                &Device::Cpu,
            )
            .unwrap(),
        );
        inputs.insert(
            "b".to_string(),
            Tensor::from_vec([b_fwd, b_bwd].concat(), (2, 8 * LSTM_HIDDEN), &Device::Cpu).unwrap(),
        );
        let values = run_graph(graph, inputs);

        let actual_y = values.get("y").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_y_h =
            values.get("y_h").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_y_c =
            values.get("y_c").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(actual_y, expected_y);
        assert_eq!(actual_y_h, expected_y_h);
        assert_eq!(actual_y_c, expected_y_c);
    }

    // A forward-only (or default-direction) LSTM must never enter this
    // rewrite at all -- the dispatch guard is `is_bidirectional`, not the
    // rewrite function itself.
    #[test]
    fn forward_only_lstm_is_left_untouched() {
        let mut graph = GraphProto {
            node: vec![lstm_node(&["x", "w", "r"], &["y"], false)],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "LSTM");
    }

    fn minimal_bidirectional_lstm_inputs() -> HashMap<String, Tensor> {
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::from_vec(
                deterministic_values(LSTM_SEQ_LEN * LSTM_INPUT, 0.0),
                (LSTM_SEQ_LEN, 1, LSTM_INPUT),
                &Device::Cpu,
            )
            .unwrap(),
        );
        inputs.insert(
            "w".to_string(),
            Tensor::from_vec(
                deterministic_values(2 * 4 * LSTM_HIDDEN * LSTM_INPUT, 1.0),
                (2, 4 * LSTM_HIDDEN, LSTM_INPUT),
                &Device::Cpu,
            )
            .unwrap(),
        );
        inputs.insert(
            "r".to_string(),
            Tensor::from_vec(
                deterministic_values(2 * 4 * LSTM_HIDDEN * LSTM_HIDDEN, 2.0),
                (2, 4 * LSTM_HIDDEN, LSTM_HIDDEN),
                &Device::Cpu,
            )
            .unwrap(),
        );
        inputs
    }

    // A trivial (all-entries-equal) sequence_lens must be accepted: it's
    // forwarded unchanged to both split nodes, and eval.rs's own
    // `seq_lens_is_default` check on each accepts it.
    #[test]
    fn bidirectional_lstm_with_trivial_sequence_lens_is_accepted() {
        let mut graph = GraphProto {
            node: vec![lstm_node(
                &["x", "w", "r", "", "sequence_lens"],
                &["y"],
                true,
            )],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut inputs = minimal_bidirectional_lstm_inputs();
        #[allow(clippy::cast_possible_wrap)]
        let seq_len = LSTM_SEQ_LEN as i64;
        inputs.insert(
            "sequence_lens".to_string(),
            Tensor::from_vec(vec![seq_len], 1, &Device::Cpu).unwrap(),
        );
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        crate::onnx::simple_eval(&model, inputs).expect("trivial sequence_lens should be accepted");
    }

    // A non-trivial sequence_lens (an entry shorter than the actual
    // sequence length) must still be rejected -- eval.rs's own
    // `seq_lens_is_default` check on each split node catches it, with no
    // rewrite-time inspection needed.
    #[test]
    fn bidirectional_lstm_with_non_trivial_sequence_lens_is_rejected() {
        let mut graph = GraphProto {
            node: vec![lstm_node(
                &["x", "w", "r", "", "sequence_lens"],
                &["y"],
                true,
            )],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut inputs = minimal_bidirectional_lstm_inputs();
        inputs.insert(
            "sequence_lens".to_string(),
            Tensor::from_vec(vec![1i64], 1, &Device::Cpu).unwrap(),
        );
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let err = crate::onnx::simple_eval(&model, inputs).unwrap_err();
        assert!(err.to_string().contains("seq_lens"));
    }

    // An all-zero peephole (`P`) must be accepted: it's sliced per
    // direction like W/R, and eval.rs's own `p_is_zeros` check on each
    // split node's half accepts an all-zero slice.
    #[test]
    fn bidirectional_lstm_with_zero_peephole_is_accepted() {
        let mut graph = GraphProto {
            node: vec![lstm_node(
                &["x", "w", "r", "", "", "", "", "p"],
                &["y"],
                true,
            )],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut inputs = minimal_bidirectional_lstm_inputs();
        inputs.insert(
            "p".to_string(),
            Tensor::zeros((2, 3 * LSTM_HIDDEN), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        crate::onnx::simple_eval(&model, inputs).expect("all-zero peephole should be accepted");
    }

    // A non-zero peephole must still be rejected, via eval.rs's own
    // `p_is_zeros` check on the split node whose half is non-zero.
    #[test]
    fn bidirectional_lstm_with_non_zero_peephole_is_rejected() {
        let mut graph = GraphProto {
            node: vec![lstm_node(
                &["x", "w", "r", "", "", "", "", "p"],
                &["y"],
                true,
            )],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut inputs = minimal_bidirectional_lstm_inputs();
        let mut p_values = vec![0.0f32; 2 * 3 * LSTM_HIDDEN];
        p_values[0] = 1.0;
        inputs.insert(
            "p".to_string(),
            Tensor::from_vec(p_values, (2, 3 * LSTM_HIDDEN), &Device::Cpu).unwrap(),
        );
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let err = crate::onnx::simple_eval(&model, inputs).unwrap_err();
        assert!(err.to_string().contains('p'));
    }

    // Regression test: Kokoro's export gives `initial_h` and `initial_c`
    // the same shape, so both can point at one shared `ConstantOfShape`
    // zero-tensor. `slice_direction` must still emit uniquely-named nodes
    // for each -- keying its generated names only on the shared tensor
    // name (rather than on each input's ordinal position) previously
    // produced two `Constant`/`Gather` pairs with identical output names,
    // which `Model::run_segment` (which requests every segment node's
    // output as a graph output) turned into a "cannot find output" error
    // on the second, already-consumed occurrence.
    #[test]
    fn bidirectional_lstm_with_shared_initial_state_tensor_is_supported() {
        let mut graph = GraphProto {
            node: vec![lstm_node(
                &["x", "w", "r", "", "", "shared_init", "shared_init"],
                &["y"],
                true,
            )],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut seen = std::collections::HashSet::new();
        for node in &graph.node {
            for output in node.output.iter().filter(|o| !o.is_empty()) {
                assert!(seen.insert(output.clone()), "duplicate node output name {output:?}");
            }
        }

        let mut inputs = minimal_bidirectional_lstm_inputs();
        inputs.insert(
            "shared_init".to_string(),
            Tensor::zeros((2, 1, LSTM_HIDDEN), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        // Mirrors `Model::run_segment`, which requests every node's output
        // in the segment as a graph output -- this is what originally
        // surfaced the bug.
        graph.output =
            graph.node.iter().flat_map(|n| n.output.iter()).filter(|o| !o.is_empty()).map(|o| out(o)).collect();
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        crate::onnx::simple_eval(&model, inputs)
            .expect("shared initial_h/initial_c tensor should not produce duplicate output names");
    }

    fn default_activations_x6() -> Vec<Vec<u8>> {
        vec![
            b"Sigmoid".to_vec(),
            b"Tanh".to_vec(),
            b"Tanh".to_vec(),
            b"Sigmoid".to_vec(),
            b"Tanh".to_vec(),
            b"Tanh".to_vec(),
        ]
    }

    // An explicit `activations` attribute that spells out the default
    // triple twice (once per direction) must be accepted and split 3-and-3
    // between the two directions, each still matching eval.rs's default.
    #[test]
    fn bidirectional_lstm_with_default_activations_is_accepted() {
        let mut node = lstm_node(&["x", "w", "r"], &["y"], true);
        node.attribute.push(strings_attribute("activations", default_activations_x6()));
        let mut graph = GraphProto {
            node: vec![node],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let inputs = minimal_bidirectional_lstm_inputs();
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        crate::onnx::simple_eval(&model, inputs).expect("default activations should be accepted");
    }

    // A non-default activations triple on one direction must still be
    // rejected by that split node's own eval.rs check.
    #[test]
    fn bidirectional_lstm_with_non_default_activations_is_rejected() {
        let mut node = lstm_node(&["x", "w", "r"], &["y"], true);
        let activations = vec![
            b"Tanh".to_vec(),
            b"Tanh".to_vec(),
            b"Tanh".to_vec(),
            b"Sigmoid".to_vec(),
            b"Tanh".to_vec(),
            b"Tanh".to_vec(),
        ];
        node.attribute.push(strings_attribute("activations", activations));
        let mut graph = GraphProto {
            node: vec![node],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let inputs = minimal_bidirectional_lstm_inputs();
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let err = crate::onnx::simple_eval(&model, inputs).unwrap_err();
        assert!(err.to_string().contains("activations"));
    }

    // A malformed (neither absent nor length-6) activations attribute on a
    // bidirectional node can't be meaningfully split -- must be left
    // untouched rather than guessing.
    #[test]
    fn bidirectional_lstm_with_malformed_activations_is_left_untouched() {
        let mut node = lstm_node(&["x", "w", "r"], &["y"], true);
        node.attribute.push(strings_attribute("activations", vec![b"Sigmoid".to_vec()]));
        let mut graph = GraphProto {
            node: vec![node],
            output: vec![out("y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "LSTM");
        assert!(is_bidirectional(&graph.node[0]));
    }
}
