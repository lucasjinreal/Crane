// SPDX-License-Identifier: MIT

//! Crane Added 20260804: evaluator-loop-wide bookkeeping for ONNX subgraph
//! (e.g. "If" branch) value capture and eviction, shared by `eval.rs`'s
//! main per-node cleanup loop and by its `"If"` op handler.

use super::eval::Value;
use super::proto::GraphProto;
use std::collections::{HashMap, HashSet};

// Crane Added 20260804: every name a subgraph's own nodes reference as an
// input but don't themselves produce — i.e. a value captured from whatever
// scope the subgraph is nested in, per ONNX subgraph scoping rules.
// Occurrences are not deduplicated: a name referenced N times contributes N
// entries, matching how the caller's `remaining_uses` counts every
// occurrence individually.
pub(crate) fn captured_names(subgraph: &GraphProto) -> Vec<&str> {
    let locally_produced: HashSet<&str> =
        subgraph.node.iter().flat_map(|node| node.output.iter().map(String::as_str)).collect();
    subgraph
        .node
        .iter()
        .flat_map(|node| node.input.iter())
        .filter(|input| !input.is_empty())
        .map(String::as_str)
        .filter(|input| !locally_produced.contains(input))
        .collect()
}

// Crane Added 20260806: every captured reference (see `captured_names`) in
// `subgraph` itself plus every subgraph nested anywhere inside it (e.g. an
// inner "If" node's then_branch/else_branch). This is the single traversal
// shared by both the up-front counting in `count_nested_subgraph_captures`
// and the post-run release in the "If" op handler, so a name counted for a
// given subgraph tree is always released against that same tree.
pub(crate) fn collect_all_captures<'a>(subgraph: &'a GraphProto, result: &mut Vec<&'a str>) {
    result.extend(captured_names(subgraph));
    for node in &subgraph.node {
        for attribute in &node.attribute {
            let Some(nested) = &attribute.g else { continue };
            collect_all_captures(nested, result);
        }
    }
}

// Crane Added 20260804: recursively adds, into `counts`, one entry per
// captured reference (see `captured_names`) in every subgraph nested
// anywhere under `graph` (e.g. an "If" node's then_branch/else_branch,
// including branches nested inside other branches). Counted once per
// subgraph a capture appears in, even across mutually-exclusive branches of
// the same "If" (at most one of which actually executes) — an intentional
// over-count, never an under-count, since which branch gets taken isn't
// known until evaluation. The "If" op handler releases the matching count
// for both the taken branch (once it has run) and the untaken branch (since
// it never runs at all) via the same `collect_all_captures` traversal used
// here, so every increment added below has exactly one release.
pub(crate) fn count_nested_subgraph_captures<'a>(
    graph: &'a GraphProto,
    counts: &mut HashMap<&'a str, usize>,
) {
    for node in &graph.node {
        for attribute in &node.attribute {
            let Some(subgraph) = &attribute.g else { continue };
            let mut captures = Vec::new();
            collect_all_captures(subgraph, &mut captures);
            for name in captures {
                *counts.entry(name).or_default() += 1;
            }
        }
    }
}

// Crane Added 20260804: decrements `remaining_uses` for each of `names`,
// evicting a value from `values` once its count reaches zero — unless it's
// a graph output or was inherited from an enclosing scope. Shared by the
// per-node cleanup loop in `simple_eval_` and by "If"'s handling of its
// taken branch's captured names (see `captured_names`), so both paths
// agree on exactly when a value is safe to free.
pub(crate) fn release_names_if_done<'a>(
    names: impl IntoIterator<Item = &'a str>,
    remaining_uses: &mut HashMap<&'a str, usize>,
    graph_outputs: &HashSet<&str>,
    inherited_values: &HashSet<String>,
    values: &mut HashMap<String, Value>,
) {
    for name in names {
        if let Some(count) = remaining_uses.get_mut(name) {
            *count -= 1;
            if *count == 0 && !graph_outputs.contains(name) && !inherited_values.contains(name) {
                values.remove(name);
            }
        }
    }
}
