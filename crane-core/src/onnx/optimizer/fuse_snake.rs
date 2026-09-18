// SPDX-License-Identifier: MIT
//! Fuses the decomposed BigVGAN-style `Snake` periodic activation into a
//! single `Snake` node.
//!
//! ONNX exporters emit `snake(x, alpha) = x + sin(alpha * x)^2 / alpha`
//! (Liu et al., used in Kokoro's HiFiGAN-style vocoder decoder, and in
//! Audio8-TTS's codec decoder) as a five-op decomposition:
//!
//! ```text
//! Mul(alpha, x) → Sin → Pow(_, 2) → Mul(inv_alpha, _) → Add(x, _) → result
//! ```
//!
//! `inv_alpha` is either a constant baked in at export time (Kokoro), or
//! computed dynamically as `Reciprocal(alpha)` (optionally behind an
//! `Identity` passthrough, as Audio8-TTS's codec decoder does) — this pass
//! recognizes both forms.
//!
//! Each op is a full read-and-write pass over the whole tensor; on the
//! decoder's largest resblocks this becomes memory-bandwidth-bound once the
//! intermediate tensors exceed cache, causing synthesis time to blow up
//! superlinearly with input length (see `ONNX_SPEEDUP.md`). This pass
//! recognizes the terminal `Add` node of that decomposition and replaces the
//! entire subgraph with a single `Snake(x, alpha)` node, evaluated by the
//! single-pass `CustomOp2` kernel in [`crate::ops::fused_ops::snake`]. Dead
//! intermediate nodes (`Mul`, `Sin`, `Pow`, `Reciprocal`, `Identity`) are
//! left for the existing DCE pass in [`super::eliminate`] to clean up.

use std::collections::HashMap;

use candle_core::DType;

use super::super::eval::{get_tensor, to_scalar_flexible};
use super::super::proto::{GraphProto, NodeProto, TensorProto};
use super::collect_producers;

/// Bound on how many `Identity` hops [`resolve_identity`] will follow before
/// giving up — guards against a pathological cycle in a malformed graph.
const MAX_IDENTITY_DEPTH: usize = 8;

/// Fuses every decomposed `Snake` pattern in `graph` into a single `Snake`
/// node, returning the number of fusions performed.
///
/// Intended to run once at optimization time (before constant folding),
/// not per inference call.
pub(crate) fn fuse_snake_decomposition(graph: &mut GraphProto) -> usize {
    let producers = collect_producers(&graph.node);
    let initializers = &graph.initializer;
    let mut fused = 0;

    for node in &mut graph.node {
        if node.op_type == "Add" && try_fuse_add(node, initializers, &producers) {
            fused += 1;
        }
    }

    fused
}

/// Given a commutative binary node's two inputs, returns
/// `(matched_input, other_input)` where `matched_input` is produced by a
/// node of type `op_type`. Tries both operand positions since exporters may
/// emit either order, returning the first match if both operands happen to
/// qualify. Returns `None` if neither input matches or `inputs` isn't
/// exactly 2 elements.
fn find_producer_input<'a>(
    inputs: &'a [String],
    producers: &HashMap<String, NodeProto>,
    op_type: &str,
) -> Option<(&'a String, &'a String)> {
    let [a, b] = inputs else {
        return None;
    };
    if producers.get(a).is_some_and(|p| p.op_type == op_type) {
        return Some((a, b));
    }
    if producers.get(b).is_some_and(|p| p.op_type == op_type) {
        return Some((b, a));
    }
    None
}

/// Given a commutative binary node's two inputs, returns the input that
/// isn't `target`, or `None` if `target` isn't among exactly 2 inputs.
fn other_input<'a>(inputs: &'a [String], target: &str) -> Option<&'a String> {
    let [a, b] = inputs else {
        return None;
    };
    if a == target {
        return Some(b);
    }
    if b == target {
        return Some(a);
    }
    None
}

/// Attempts to match `node` (an `Add`) against the `Snake` decomposition
/// and, if it matches, rewrites `node` in place to a `Snake(x, alpha)` node
/// with the same output name. Returns `true` on a successful rewrite.
///
/// The matched shape (backward from the terminal `Add`, trying both operand
/// orders since `Add`/`Mul` are commutative):
///
/// 1. `Add(x, mul2_out)` — `x` is one operand, the other is the tail of the
///    `sin^2/alpha` chain.
/// 2. `mul2_out` is produced by `Mul(inv_alpha, pow_out)`.
/// 3. `pow_out` is produced by `Pow(sin_out, exponent)`.
/// 4. `sin_out` is produced by `Sin(mul1_out)`.
/// 5. `mul1_out` is produced by `Mul(alpha, x)` — one operand is the same
///    `x` tensor as the outer `Add`'s.
fn try_fuse_add(
    node: &mut NodeProto,
    initializers: &[TensorProto],
    producers: &HashMap<String, NodeProto>,
) -> bool {
    let [in0, in1] = node.input.as_slice() else {
        return false;
    };
    if node.output.len() != 1 {
        return false;
    }

    let matched = try_match_chain(in0, in1, initializers, producers)
        .or_else(|| try_match_chain(in1, in0, initializers, producers));
    let Some((x, alpha)) = matched else {
        return false;
    };

    node.op_type = "Snake".to_string();
    node.input = vec![x, alpha];
    node.name = if node.name.is_empty() {
        "fused_snake".to_string()
    } else {
        format!("{}/fused_snake", node.name)
    };
    true
}

/// Walks backward from `mul2_name` (the `Add`'s non-`x` operand) through
/// `Mul → Pow → Sin → Mul` looking for a `Mul` whose operands are `alpha`
/// and `x`. The `Pow` exponent and `alpha` itself must each be static —
/// either a raw graph initializer (absent from `producers`) or a `Constant`
/// op node's output (some exporters, including Audio8-TTS's, emit scalar
/// literals like the exponent `2` as `Constant` nodes rather than
/// initializers) — rather than dynamically computed, since the fused kernel
/// hardcodes the square. The exponent's value is also checked to actually
/// *be* `2` (via [`is_exponent_two`]): the fused kernel hardcodes
/// `sin(alpha * x)^2`, so a static-but-different exponent must not fuse
/// either. `inv_alpha` may be static in either of those same two forms, or
/// `Reciprocal(alpha)` (optionally behind an `Identity` passthrough on
/// either side, per [`resolve_identity`], as Audio8-TTS's codec decoder
/// emits) — anything else (e.g. an `inv_alpha` unrelated to `alpha`) is
/// rejected, since the fused kernel hardcodes division by `alpha`. Returns
/// `(x, alpha)` on a match.
fn try_match_chain(
    x: &str,
    mul2_name: &str,
    initializers: &[TensorProto],
    producers: &HashMap<String, NodeProto>,
) -> Option<(String, String)> {
    let mul2 = producers.get(mul2_name)?;
    if mul2.op_type != "Mul" {
        return None;
    }
    let (pow_name, inv_alpha) = find_producer_input(&mul2.input, producers, "Pow")?;

    let pow = &producers[pow_name];
    let (sin_name, exponent) = find_producer_input(&pow.input, producers, "Sin")?;
    if is_dynamic(exponent, producers) || !is_exponent_two(exponent, initializers, producers) {
        return None;
    }

    let sin = &producers[sin_name];
    let [mul1_name] = sin.input.as_slice() else {
        return None;
    };

    let mul1 = producers.get(mul1_name)?;
    if mul1.op_type != "Mul" {
        return None;
    }
    let alpha = other_input(&mul1.input, x)?;
    if is_dynamic(alpha, producers) {
        return None;
    }

    let inv_alpha_ok = match producers.get(inv_alpha) {
        None => true,
        Some(node) if node.op_type == "Constant" => true,
        Some(node) => {
            node.op_type == "Reciprocal"
                && matches!(node.input.as_slice(), [reciprocal_input]
                    if resolve_identity(reciprocal_input, producers) == resolve_identity(alpha, producers))
        },
    };
    if !inv_alpha_ok {
        return None;
    }

    Some((x.to_string(), alpha.clone()))
}

/// Whether `name` is computed by a node other than `Constant` — i.e. a
/// value that can genuinely vary per forward pass, as opposed to a raw
/// graph initializer (absent from `producers` entirely) or a `Constant`
/// node's baked-in literal (both of which are static for every inference).
fn is_dynamic(name: &str, producers: &HashMap<String, NodeProto>) -> bool {
    producers
        .get(name)
        .is_some_and(|node| node.op_type != "Constant")
}

/// Returns the `TensorProto` backing `name`, whether it's a raw graph
/// initializer or a `Constant` node's `"value"` attribute — the two static
/// forms [`is_dynamic`] accepts.
fn resolve_constant_tensor<'a>(
    name: &str,
    initializers: &'a [TensorProto],
    producers: &'a HashMap<String, NodeProto>,
) -> Option<&'a TensorProto> {
    if let Some(node) = producers.get(name) {
        return node
            .attribute
            .iter()
            .find(|attr| attr.name == "value")
            .and_then(|attr| attr.t.as_ref());
    }
    initializers
        .iter()
        .find(|initializer| initializer.name == name)
}

/// Whether `name` (already confirmed static by [`is_dynamic`]) holds the
/// scalar value `2`. The fused kernel hardcodes `sin(alpha * x)^2`, so a
/// `Pow` exponent of any other value must not be fused; if the tensor can't
/// be resolved or decoded at all, this conservatively refuses to fuse.
fn is_exponent_two(
    name: &str,
    initializers: &[TensorProto],
    producers: &HashMap<String, NodeProto>,
) -> bool {
    let Some(tensor_proto) = resolve_constant_tensor(name, initializers, producers) else {
        return false;
    };
    let Ok(tensor) = get_tensor(tensor_proto, name) else {
        return false;
    };
    let Ok(tensor) = tensor.to_dtype(DType::F64) else {
        return false;
    };
    let Ok(value) = to_scalar_flexible::<f64>(&tensor) else {
        return false;
    };
    (value - 2.0).abs() < 1e-6
}

/// Follows a chain of `Identity` nodes back to the original source tensor
/// name, stopping at the first non-`Identity` producer or a graph leaf
/// (initializer/input). Bounded depth guards against a pathological cycle
/// in a malformed graph.
fn resolve_identity<'a>(mut name: &'a str, producers: &'a HashMap<String, NodeProto>) -> &'a str {
    for _ in 0..MAX_IDENTITY_DEPTH {
        let Some(node) = producers.get(name) else {
            return name;
        };
        if node.op_type != "Identity" {
            return name;
        }
        let [only_input] = node.input.as_slice() else {
            return name;
        };
        name = only_input;
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::proto::AttributeProto;
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::tensor_proto::DataType;

    /// A raw graph initializer holding the scalar `value`, e.g. the Pow
    /// exponent `2` most exporters bake in as an initializer rather than a
    /// `Constant` node.
    fn scalar_initializer(name: &str, value: f32) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Float as i32,
            float_data: vec![value],
            ..Default::default()
        }
    }

    fn binary_node(op_type: &str, a: &str, b: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: vec![a.to_string(), b.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn unary_node(op_type: &str, input: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: vec![input.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    /// Builds the full `snake(x, alpha)` decomposition:
    /// `Mul(alpha, x) -> Sin -> Pow(_, 2) -> Mul(inv_alpha, _) -> Add(x, _)`.
    /// `suffix` disambiguates intermediate tensor names across multiple
    /// independent instances in the same graph.
    fn snake_decomposition(
        x: &str,
        alpha: &str,
        inv_alpha: &str,
        output: &str,
        suffix: &str,
    ) -> Vec<NodeProto> {
        vec![
            binary_node("Mul", alpha, x, &format!("mul1{suffix}")),
            unary_node("Sin", &format!("mul1{suffix}"), &format!("sin{suffix}")),
            binary_node(
                "Pow",
                &format!("sin{suffix}"),
                "two",
                &format!("pow{suffix}"),
            ),
            binary_node(
                "Mul",
                inv_alpha,
                &format!("pow{suffix}"),
                &format!("mul2{suffix}"),
            ),
            binary_node("Add", x, &format!("mul2{suffix}"), output),
        ]
    }

    // The motivating case: the full Snake decomposition is recognized and
    // the terminal Add is rewritten to a single Snake(x, alpha) node.
    #[test]
    fn fuses_full_snake_decomposition() {
        let mut graph = GraphProto {
            node: snake_decomposition("x", "alpha", "inv_alpha", "result", ""),
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
        assert_eq!(snake_node.output, vec!["result"]);
    }

    // Exporters may emit the Add/Mul operands in either order since both
    // ops are commutative; the reversed order must fuse identically.
    #[test]
    fn fuses_with_reversed_commutative_operands() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "x", "alpha", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "pow", "inv_alpha", "mul2"),
                binary_node("Add", "mul2", "x", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
    }

    // Unrelated Add nodes must be left completely unchanged.
    #[test]
    fn leaves_unrelated_add_unchanged() {
        let mut graph = GraphProto {
            node: vec![binary_node("Add", "a", "b", "y")],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 0);
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Add");
    }

    // Multiple independent Snake decompositions in the same graph should
    // each be fused.
    #[test]
    fn fuses_multiple_decompositions() {
        let mut nodes = snake_decomposition("x1", "alpha1", "inv_alpha1", "result1", "_a");
        nodes.extend(snake_decomposition(
            "x2",
            "alpha2",
            "inv_alpha2",
            "result2",
            "_b",
        ));

        let mut graph = GraphProto {
            node: nodes,
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 2);
    }

    // A Pow node whose non-exponent input is not a Sin output must not be
    // fused — the chain is broken.
    #[test]
    fn does_not_fuse_wrong_pow_input() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                binary_node("Add", "mul1", "other", "not_sin"),
                binary_node("Pow", "not_sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // The innermost Mul must actually reference the same `x` tensor as the
    // outer Add — a decomposition for a *different* tensor must not fuse.
    #[test]
    fn does_not_fuse_when_x_mismatch() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "unrelated_x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // The Pow exponent must be a constant (absent from producers), not a
    // dynamically computed value — the fused kernel hardcodes the square.
    #[test]
    fn does_not_fuse_dynamic_exponent() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                unary_node("Identity", "dynamic_source", "two"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // inv_alpha must be a constant (absent from producers), not derived
    // from an unrelated dynamic computation — the fused kernel hardcodes
    // division by `alpha`, so a dynamic inv_alpha could silently diverge.
    #[test]
    fn does_not_fuse_dynamic_inv_alpha() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                unary_node("Identity", "dynamic_source", "inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // Audio8-TTS's codec decoder computes inv_alpha dynamically as
    // Reciprocal(alpha) rather than baking it in as a constant. Must still
    // fuse, since it's mathematically identical to the constant case.
    #[test]
    fn fuses_with_dynamic_reciprocal_inv_alpha() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                unary_node("Reciprocal", "alpha", "inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
    }

    // The real Audio8-TTS graph wraps alpha in a redundant Identity before
    // feeding Reciprocal, an exporter artifact. Must still fuse, resolving
    // through the Identity on both sides.
    #[test]
    fn fuses_with_reciprocal_of_identity_wrapped_alpha() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                unary_node("Identity", "alpha", "alpha_identity"),
                unary_node("Reciprocal", "alpha_identity", "inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
    }

    // A Reciprocal of a tensor unrelated to alpha must not fuse — the fused
    // kernel hardcodes division by alpha specifically.
    #[test]
    fn does_not_fuse_reciprocal_of_unrelated_tensor() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                unary_node("Reciprocal", "unrelated", "inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    fn constant_node(output: &str) -> NodeProto {
        constant_node_with_value(output, 2.0)
    }

    /// A `Constant` node holding the scalar `value` in its `"value"`
    /// attribute, matching how `eval.rs`'s `Constant` handler
    /// ([`get_tensor`]) decodes it.
    fn constant_node_with_value(output: &str, value: f32) -> NodeProto {
        NodeProto {
            op_type: "Constant".to_string(),
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor as i32,
                t: Some(TensorProto {
                    data_type: DataType::Float as i32,
                    float_data: vec![value],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // Audio8-TTS's codec decoder exports the Pow exponent as a `Constant`
    // op node (a PyTorch-traced scalar literal) rather than a raw graph
    // initializer. Must still fuse.
    #[test]
    fn fuses_with_constant_node_exponent() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                constant_node("two"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 1);
    }

    // inv_alpha represented as a `Constant` op node (rather than a raw
    // initializer) must also fuse.
    #[test]
    fn fuses_with_constant_node_inv_alpha() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                constant_node("inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            initializer: vec![scalar_initializer("two", 2.0)],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 1);
    }

    // The real Audio8-TTS shape end-to-end: exponent as a Constant node,
    // inv_alpha as Reciprocal(Identity(alpha)) — both relaxations combined.
    #[test]
    fn fuses_audio8_shaped_decomposition() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                constant_node("two"),
                binary_node("Pow", "sin", "two", "pow"),
                unary_node("Identity", "alpha", "alpha_identity"),
                unary_node("Reciprocal", "alpha_identity", "inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
    }

    // A Pow exponent that's static but not actually 2 must not fuse — the
    // fused kernel hardcodes squaring, so this would silently miscompute.
    #[test]
    fn does_not_fuse_non_two_exponent() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                constant_node_with_value("three", 3.0),
                binary_node("Pow", "sin", "three", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }
}
