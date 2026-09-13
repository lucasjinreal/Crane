//! Conservative graph simplification performed when an ONNX session is built.

mod compat;
mod constant_fold;
mod eliminate;
pub(crate) mod fuse_atan2;
pub(crate) mod fuse_quantized_linear;
pub(crate) mod fuse_snake;

use std::collections::{HashMap, HashSet};

use candle_core::{Result, Tensor};

use super::proto::{GraphProto, NodeProto};

#[derive(Clone, Debug)]
pub struct SessionOptions {
    /// Simplify the graph before preparing its initializer tensors.
    pub optimize: bool,
    /// Refuse to retain a newly folded constant larger than this many elements.
    pub max_folded_elements: usize,
    /// Stop fixed-point optimization after this many iterations.
    pub max_optimization_passes: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            optimize: true,
            max_folded_elements: 1_000_000,
            max_optimization_passes: 8,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OptimizationReport {
    pub original_nodes: usize,
    pub final_nodes: usize,
    pub folded_nodes: usize,
    pub removed_alias_nodes: usize,
    pub removed_dead_nodes: usize,
    pub removed_initializers: usize,
    /// Number of decomposed `atan2(y, x)` patterns fused into single nodes.
    pub fused_atan2_nodes: usize,
    /// Number of decomposed `Snake` activation patterns fused into single nodes.
    pub fused_snake_nodes: usize,
    /// Number of `DynamicQuantizeLinear`/`MatMulInteger`/`DequantizeLinear`
    /// patterns erased in favor of a pre-dequantized `Gather`/`MatMul`.
    pub dequantized_int8_nodes: usize,
    /// DCE is skipped when graph-valued attributes may capture outer values.
    pub skipped_dce_for_subgraphs: bool,
}

pub(crate) fn optimize(
    graph: &mut GraphProto,
    constants: &mut HashMap<String, Tensor>,
    options: &SessionOptions,
) -> Result<OptimizationReport> {
    let mut report = OptimizationReport {
        original_nodes: graph.node.len(),
        ..Default::default()
    };

    // Correctness fixes for gaps in `eval.rs`, not a size optimization —
    // run unconditionally, even when `options.optimize` disables the
    // simplification passes below.
    compat::rewrite_unsupported_ops(graph)
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?;

    // Also unconditional: without this, `eval.rs`'s missing
    // `DynamicQuantizeLinear`/`MatMulInteger`/`DequantizeLinear` support
    // would make INT8-quantized graphs (e.g. Audio8-TTS) fail to evaluate
    // at all, optimizations disabled or not.
    report.dequantized_int8_nodes +=
        fuse_quantized_linear::fuse_quantized_linear(graph, constants)?;

    if !options.optimize {
        report.final_nodes = graph.node.len();
        return Ok(report);
    }

    report.removed_alias_nodes += eliminate::eliminate_alias_nodes(graph);
    report.fused_atan2_nodes = fuse_atan2::fuse_atan2_decomposition(graph);
    report.fused_snake_nodes = fuse_snake::fuse_snake_decomposition(graph);

    for _ in 0..options.max_optimization_passes {
        let folded = constant_fold::fold_constants(graph, constants, options.max_folded_elements)?;
        let aliases = eliminate::eliminate_alias_nodes(graph);
        // Re-run: constant folding/alias elimination can turn an
        // indirection (Identity/Transpose/Cast) between a static
        // initializer and a quantized op into a direct reference, which
        // this pass otherwise only sees on its first, pre-fold pass.
        let dequantized = fuse_quantized_linear::fuse_quantized_linear(graph, constants)?;
        report.folded_nodes += folded;
        report.removed_alias_nodes += aliases;
        report.dequantized_int8_nodes += dequantized;
        if folded == 0 && aliases == 0 && dequantized == 0 {
            break;
        }
    }

    if eliminate::contains_subgraphs(graph) {
        report.skipped_dce_for_subgraphs = true;
    } else {
        report.removed_dead_nodes = eliminate::eliminate_dead_nodes(graph);
    }
    report.removed_initializers = prune_unused_constants(graph, constants);
    report.final_nodes = graph.node.len();
    Ok(report)
}

/// Maps each node output name to a clone of its producing node, so a
/// backward-walking fusion pass (e.g. [`fuse_atan2`] or [`fuse_snake`]) can
/// trace a subgraph's inputs back to their producers without a second pass
/// over the node list.
pub(super) fn collect_producers(nodes: &[NodeProto]) -> HashMap<String, NodeProto> {
    let mut producers = HashMap::with_capacity(nodes.len());
    for node in nodes {
        for output in &node.output {
            if !output.is_empty() {
                producers.insert(output.clone(), node.clone());
            }
        }
    }
    producers
}

fn prune_unused_constants(
    graph: &mut GraphProto,
    constants: &mut HashMap<String, Tensor>,
) -> usize {
    let mut used = graph
        .node
        .iter()
        .flat_map(|node| node.input.iter())
        .filter(|name| !name.is_empty())
        .cloned()
        .collect::<HashSet<_>>();
    used.extend(graph.output.iter().map(|output| output.name.clone()));
    // Older ONNX files also list overridable initializers as graph inputs.
    used.extend(graph.input.iter().map(|input| input.name.clone()));

    let before = constants.len();
    constants.retain(|name, _| used.contains(name));
    graph
        .initializer
        .retain(|initializer| constants.contains_key(&initializer.name));
    before - constants.len()
}
