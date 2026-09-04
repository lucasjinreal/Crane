// SPDX-License-Identifier: MIT
//! Erases ONNX Runtime's dynamic INT8 quantization ops
//! (`DynamicQuantizeLinear`, `MatMulInteger`, `DequantizeLinear`) at
//! session-load time. `crate::onnx::eval` doesn't implement any of the
//! three, and candle has no `Int8` dtype to build a competitive integer
//! GEMM around. In both patterns below, the quantized weight/table
//! operand (and its scale/zero-point) is always a static graph
//! initializer, so the whole chain can be replaced once, at load time, by
//! dequantizing that operand to F32 and using a plain `Gather`/`MatMul`:
//!
//! ```text
//! Gather(quantized_table) -> DequantizeLinear(scale, zero_point)
//!
//! DynamicQuantizeLinear(x) -> MatMulInteger(x_q, W_q, x_zp, W_zp)
//!     -> Cast(to=Float) -> Mul(x_scale * W_scale)
//! ```
//!
//! No runtime int8 arithmetic is needed at all; the activation is never
//! actually quantized.

use std::collections::{HashMap, HashSet};

use candle_core::{DType, Result, Tensor};

use super::super::proto::tensor_proto::DataType;
use super::super::proto::{GraphProto, NodeProto};
use super::collect_producers;

/// Erases every `Gather -> DequantizeLinear` and `DynamicQuantizeLinear ->
/// MatMulInteger -> Cast -> Mul` pattern in `graph` whose quantized
/// weight/table operand is a static entry in `constants`, replacing each
/// with a plain `Gather`/`MatMul` against a load-time-dequantized F32
/// tensor inserted into `constants`. Returns the number of patterns erased.
///
/// The dead intermediate `Gather` left behind by the first pattern is a
/// supported op, so the existing DCE pass in [`super::eliminate`] can clean
/// it up whenever it next runs. `DynamicQuantizeLinear`/`MatMulInteger` are
/// *not* supported by `crate::onnx::eval`, so any left dangling by the
/// second pattern are removed here directly by
/// [`remove_dead_quantization_ops`], rather than deferred to DCE. DCE only
/// runs when `super::SessionOptions::optimize` is set.
pub(crate) fn fuse_quantized_linear(
    graph: &mut GraphProto,
    constants: &mut HashMap<String, Tensor>,
) -> Result<usize> {
    let producers = collect_producers(&graph.node);
    let mut fused = 0;
    let mut new_constants = HashMap::new();

    for node in &mut graph.node {
        let rewritten = match node.op_type.as_str() {
            "DequantizeLinear" => {
                try_fuse_gather_dequantize(node, &producers, constants, &mut new_constants)?
            },
            "Mul" => try_fuse_quantized_matmul(node, &producers, constants, &mut new_constants)?,
            _ => false,
        };
        if rewritten {
            fused += 1;
        }
    }

    constants.extend(new_constants);
    remove_dead_quantization_ops(graph);
    Ok(fused)
}

/// Deletes any `DynamicQuantizeLinear`/`MatMulInteger` node unreachable
/// backward from `graph.output`. `crate::onnx::eval` implements neither op
/// and evaluates every node in `graph.node` unconditionally, so one left
/// behind by [`try_fuse_quantized_matmul`] would make the graph fail to
/// evaluate even though nothing actually needs it. Its own consuming `Mul`
/// got rewritten away, but a still-present, now-otherwise-dead `Cast`/`Mul`
/// between it and that `Mul` keeps referencing its output.
///
/// Other now-dead nodes (that `Cast`, a scale-combining `Mul`, the original
/// quantized `Gather`) are left in place for the existing DCE pass in
/// [`super::eliminate`] to remove, same as before this function existed.
/// Only `DynamicQuantizeLinear`/`MatMulInteger` are unconditionally
/// unsupported, so only they need removing regardless of whether that DCE
/// pass runs (it's skipped when `super::SessionOptions::optimize` is unset).
fn remove_dead_quantization_ops(graph: &mut GraphProto) {
    let mut needed: HashSet<String> = graph
        .output
        .iter()
        .map(|output| output.name.clone())
        .collect();
    let mut reachable = vec![false; graph.node.len()];
    for (index, node) in graph.node.iter().enumerate().rev() {
        if node.output.iter().any(|output| needed.contains(output)) {
            reachable[index] = true;
            needed.extend(node.input.iter().filter(|name| !name.is_empty()).cloned());
        }
    }
    let mut reachable = reachable.into_iter();
    graph.node.retain(|node| {
        reachable.next().unwrap_or(true)
            || !matches!(
                node.op_type.as_str(),
                "DynamicQuantizeLinear" | "MatMulInteger"
            )
    });
}

/// Computes `(quantized - zero_point) * scale` in F32, broadcasting
/// `zero_point`/`scale` against `quantized`'s shape. Scalar for embedding
/// tables, per-output-channel `[N]` for `MatMulInteger` weights (`[K, N]`,
/// broadcasting against the last dimension).
fn dequantize(quantized: &Tensor, zero_point: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let quantized = quantized.to_dtype(DType::F32)?;
    let zero_point = zero_point.to_dtype(DType::F32)?;
    let scale = scale.to_dtype(DType::F32)?;
    quantized.broadcast_sub(&zero_point)?.broadcast_mul(&scale)
}

/// Dequantizes `quantized` and records it under `name` in `new_constants`,
/// unless `name` is already present there or in `constants`. A quantized
/// weight/table shared by more than one consumer (e.g. tied weights) is
/// dequantized at most once per pass.
fn dequantize_cached(
    name: String,
    quantized: &Tensor,
    zero_point: &Tensor,
    scale: &Tensor,
    constants: &HashMap<String, Tensor>,
    new_constants: &mut HashMap<String, Tensor>,
) -> Result<()> {
    if constants.contains_key(&name) || new_constants.contains_key(&name) {
        return Ok(());
    }
    let dequantized = dequantize(quantized, zero_point, scale)?;
    new_constants.insert(name, dequantized);
    Ok(())
}

/// Matches `node` (a `DequantizeLinear`) against `Gather(quantized_table)
/// -> DequantizeLinear(scale, zero_point)`, where `quantized_table` is a
/// static entry in `constants`. On a match, rewrites `node` in place into a
/// `Gather(dequantized_table, indices)` node with the same output name.
/// This copies the original `Gather`'s `axis` attribute, if any, and
/// records the dequantized table for insertion into `constants`.
fn try_fuse_gather_dequantize(
    node: &mut NodeProto,
    producers: &HashMap<String, NodeProto>,
    constants: &HashMap<String, Tensor>,
    new_constants: &mut HashMap<String, Tensor>,
) -> Result<bool> {
    let inputs = node.input.clone();
    let [quantized_out, scale_name, zero_point_name] = inputs.as_slice() else {
        return Ok(false);
    };
    let Some(gather) = producers.get(quantized_out) else {
        return Ok(false);
    };
    if gather.op_type != "Gather" || gather.input.len() != 2 {
        return Ok(false);
    }
    let table_name = gather.input[0].clone();
    let indices_name = gather.input[1].clone();
    let (Some(table), Some(scale), Some(zero_point)) = (
        constants.get(&table_name),
        constants.get(scale_name),
        constants.get(zero_point_name),
    ) else {
        return Ok(false);
    };

    let dequantized_name = format!("{table_name}/dequantized");
    dequantize_cached(
        dequantized_name.clone(),
        table,
        zero_point,
        scale,
        constants,
        new_constants,
    )?;

    node.op_type = "Gather".to_string();
    node.input = vec![dequantized_name, indices_name];
    node.attribute.clone_from(&gather.attribute);
    node.name = if node.name.is_empty() {
        "dequantized_gather".to_string()
    } else {
        format!("{}/dequantized_gather", node.name)
    };

    Ok(true)
}

/// Matches `node` (a `Mul`) against `DynamicQuantizeLinear(x) ->
/// MatMulInteger(x_q, W_q, x_zp, W_zp) -> Cast(to=Float) -> Mul(cast_out,
/// combined_scale)`, where `W_q`/`W_zp` are static entries in `constants`,
/// `MatMulInteger`'s `a_zero_point` input is `DynamicQuantizeLinear`'s own
/// zero-point output, and `combined_scale` resolves to a static weight
/// scale (see [`resolve_weight_scale`]). The zero-point equality is
/// required for the rewrite's math to hold; see below for why. On a match,
/// rewrites `node` in place into a `MatMul(x, dequantized_weight)` node
/// with the same output name, and records the dequantized weight for
/// insertion into `constants`.
///
/// The rewrite computes `x @ ((W_q - W_zp) * W_scale)`, which only equals
/// the original `(x_q - x_zp) @ (W_q - W_zp) * x_scale * W_scale` chain if
/// `MatMulInteger`'s `a_zero_point` operand is exactly the `x_zp` that
/// `DynamicQuantizeLinear` derived from `x`. A different (e.g. constant)
/// zero-point there would silently break that cancellation.
fn try_fuse_quantized_matmul(
    node: &mut NodeProto,
    producers: &HashMap<String, NodeProto>,
    constants: &HashMap<String, Tensor>,
    new_constants: &mut HashMap<String, Tensor>,
) -> Result<bool> {
    let inputs = node.input.clone();
    let [in0, in1] = inputs.as_slice() else {
        return Ok(false);
    };
    let Some((cast_out, combined_scale)) = try_match_float_cast(in0, in1, producers)
        .or_else(|| try_match_float_cast(in1, in0, producers))
    else {
        return Ok(false);
    };
    let cast = &producers[cast_out];
    let Some(mmi) = producers.get(&cast.input[0]) else {
        return Ok(false);
    };
    // ONNX also allows `MatMulInteger` with 2 or 3 inputs (implicit
    // zero-points of 0); only the 4-input form observed in Audio8-TTS's
    // export is handled, matching this module's narrow scope.
    if mmi.op_type != "MatMulInteger" || mmi.input.len() != 4 {
        return Ok(false);
    }
    let x_q_name = mmi.input[0].clone();
    let w_q_name = mmi.input[1].clone();
    let a_zero_point_name = mmi.input[2].clone();
    let w_zp_name = mmi.input[3].clone();

    let Some(dql) = producers.get(&x_q_name) else {
        return Ok(false);
    };
    if dql.op_type != "DynamicQuantizeLinear"
        || dql.input.len() != 1
        || dql.output.len() != 3
        || dql.output[0] != x_q_name
        || dql.output[2] != a_zero_point_name
    {
        return Ok(false);
    }
    let x_name = dql.input[0].clone();
    let x_scale_name = &dql.output[1];

    let Some(w_scale) = resolve_weight_scale(combined_scale, x_scale_name, producers, constants)
    else {
        return Ok(false);
    };
    let (Some(w_q), Some(w_zp)) = (constants.get(&w_q_name), constants.get(&w_zp_name)) else {
        return Ok(false);
    };

    let dequantized_name = format!("{w_q_name}/dequantized");
    dequantize_cached(
        dequantized_name.clone(),
        w_q,
        w_zp,
        w_scale,
        constants,
        new_constants,
    )?;

    node.op_type = "MatMul".to_string();
    node.input = vec![x_name, dequantized_name];
    node.name = if node.name.is_empty() {
        "dequantized_matmul".to_string()
    } else {
        format!("{}/dequantized_matmul", node.name)
    };

    Ok(true)
}

/// Given a commutative binary node's two inputs, returns `(cast_out,
/// other)` if `a` is produced by a single-input `Cast` node whose `to`
/// attribute targets a floating-point type. Anything else means `a` isn't
/// the `Cast(MatMulInteger_output, to=Float)` half of the pattern this pass
/// looks for.
fn try_match_float_cast<'a>(
    a: &'a str,
    b: &'a str,
    producers: &HashMap<String, NodeProto>,
) -> Option<(&'a str, &'a str)> {
    let cast = producers.get(a)?;
    if cast.op_type != "Cast" || cast.input.len() != 1 {
        return None;
    }
    let to = *crate::onnx::eval::get_attr_opt::<i64>(cast, "to")
        .ok()
        .flatten()?;
    match DataType::try_from(i32::try_from(to).ok()?) {
        Ok(DataType::Float | DataType::Float16 | DataType::Double) => Some((a, b)),
        _ => None,
    }
}

/// Resolves the static per-output-channel weight scale from the terminal
/// `Mul`'s non-`Cast` operand (`combined_scale`): either that operand is
/// itself a static constant, or it's `Mul(x_scale, w_scale)` with one
/// operand being the `DynamicQuantizeLinear`'s own scale output
/// (`x_scale_name`) and the other a static constant. This is the shape
/// observed in Audio8-TTS's export, since the activation scale is dynamic
/// per call and must be combined with the static weight scale after the
/// matmul.
fn resolve_weight_scale<'a>(
    combined_scale: &str,
    x_scale_name: &str,
    producers: &HashMap<String, NodeProto>,
    constants: &'a HashMap<String, Tensor>,
) -> Option<&'a Tensor> {
    if let Some(w_scale) = constants.get(combined_scale) {
        return Some(w_scale);
    }
    let scale_mul = producers.get(combined_scale)?;
    if scale_mul.op_type != "Mul" {
        return None;
    }
    let [a, b] = scale_mul.input.as_slice() else {
        return None;
    };
    let w_scale_name = if a == x_scale_name {
        b
    } else if b == x_scale_name {
        a
    } else {
        return None;
    };
    constants.get(w_scale_name)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::super::super::proto::attribute_proto::AttributeType;
    use super::super::super::proto::{AttributeProto, ValueInfoProto};
    use super::*;

    fn graph_output(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    // `Cast(to=Float)`, matching the `to`-attribute check
    // `try_match_float_cast` requires before treating a `Cast` as the
    // `MatMulInteger` -> float half of the fused pattern.
    fn cast_to_float_node(input: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: "Cast".to_string(),
            input: vec![input.to_string()],
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "to".to_string(),
                r#type: AttributeType::Int as i32,
                i: DataType::Float as i64,
                ..Default::default()
            }],
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

    // Gather(quantized_table) -> DequantizeLinear(scale, zero_point) with a
    // scalar scale/zero-point, matching Audio8-TTS's embedding tables. The
    // rewritten graph must evaluate to the same result as manually
    // dequantizing the table and gathering from it.
    #[test]
    fn fuses_gather_dequantize() -> Result<()> {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Gather", "table_quantized", "indices", "gathered_quantized"),
                NodeProto {
                    op_type: "DequantizeLinear".to_string(),
                    input: vec![
                        "gathered_quantized".to_string(),
                        "scale".to_string(),
                        "zero_point".to_string(),
                    ],
                    output: vec!["result".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut constants = HashMap::from([
            (
                "table_quantized".to_string(),
                Tensor::new(&[10u8, 20, 30, 40], &Device::Cpu)?.reshape((2, 2))?,
            ),
            ("scale".to_string(), Tensor::new(0.5f32, &Device::Cpu)?),
            ("zero_point".to_string(), Tensor::new(10u8, &Device::Cpu)?),
        ]);

        let fused = fuse_quantized_linear(&mut graph, &mut constants)?;

        assert_eq!(fused, 1);
        assert!(!graph.node.iter().any(|n| n.op_type == "DequantizeLinear"));
        let gather = graph
            .node
            .iter()
            .find(|n| n.output == ["result"])
            .expect("rewritten node should still produce 'result'");
        assert_eq!(gather.op_type, "Gather");
        let table = constants
            .get(&gather.input[0])
            .expect("dequantized table constant");
        assert_eq!(
            table.to_vec2::<f32>()?,
            vec![vec![0.0, 5.0], vec![10.0, 15.0]]
        );
        Ok(())
    }

    // DynamicQuantizeLinear(x) -> MatMulInteger(x_q, W_q, x_zp, W_zp) ->
    // Cast -> Mul(x_scale * W_scale), with a per-output-channel W_scale,
    // matching Audio8-TTS's linear layers. The rewritten graph must
    // evaluate to the same result as manually dequantizing the weight and
    // matmul-ing against it.
    #[test]
    fn fuses_quantized_matmul() -> Result<()> {
        let mut graph = GraphProto {
            node: vec![
                NodeProto {
                    op_type: "DynamicQuantizeLinear".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["x_q".to_string(), "x_scale".to_string(), "x_zp".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "MatMulInteger".to_string(),
                    input: vec![
                        "x_q".to_string(),
                        "w_q".to_string(),
                        "x_zp".to_string(),
                        "w_zp".to_string(),
                    ],
                    output: vec!["mmi_out".to_string()],
                    ..Default::default()
                },
                cast_to_float_node("mmi_out", "cast_out"),
                binary_node("Mul", "cast_out", "combined_scale", "combined_scale_out"),
                binary_node("Mul", "x_scale", "w_scale", "combined_scale"),
            ],
            output: vec![graph_output("combined_scale_out")],
            ..Default::default()
        };
        // Swap the two Mul node positions so the scale-combining Mul is
        // discovered via `producers`, not list order.
        graph.node.swap(3, 4);

        let mut constants = HashMap::from([
            (
                "w_q".to_string(),
                Tensor::new(&[1i16, 2, 3, 4], &Device::Cpu)?.reshape((2, 2))?,
            ),
            ("w_zp".to_string(), Tensor::new(&[0i16, 1], &Device::Cpu)?),
            (
                "w_scale".to_string(),
                Tensor::new(&[2.0f32, 4.0], &Device::Cpu)?,
            ),
        ]);

        let fused = fuse_quantized_linear(&mut graph, &mut constants)?;

        assert_eq!(fused, 1);
        let matmul = graph
            .node
            .iter()
            .find(|n| n.output == ["combined_scale_out"])
            .expect("rewritten node should still produce 'combined_scale_out'");
        assert_eq!(matmul.op_type, "MatMul");
        assert_eq!(matmul.input[0], "x");
        let weight = constants
            .get(&matmul.input[1])
            .expect("dequantized weight constant");
        // (w_q - w_zp) * w_scale, per column: col0 zp=0 scale=2, col1 zp=1 scale=4.
        assert_eq!(
            weight.to_vec2::<f32>()?,
            vec![vec![2.0, 4.0], vec![6.0, 12.0]]
        );
        // The now-unreachable `DynamicQuantizeLinear`/`MatMulInteger` must
        // be gone. `eval.rs` implements neither, so leaving them behind
        // would break evaluation even when `SessionOptions::optimize` skips
        // the general DCE pass that would otherwise remove them.
        assert!(!graph.node.iter().any(|n| matches!(
            n.op_type.as_str(),
            "DynamicQuantizeLinear" | "MatMulInteger"
        )));
        Ok(())
    }

    // Unrelated Mul/DequantizeLinear nodes must be left completely
    // unchanged.
    #[test]
    fn leaves_unrelated_nodes_unchanged() -> Result<()> {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "a", "b", "y"),
                NodeProto {
                    op_type: "DequantizeLinear".to_string(),
                    input: vec!["a".to_string(), "scale".to_string(), "zp".to_string()],
                    output: vec!["z".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut constants = HashMap::new();

        let fused = fuse_quantized_linear(&mut graph, &mut constants)?;

        assert_eq!(fused, 0);
        assert_eq!(graph.node[0].op_type, "Mul");
        assert_eq!(graph.node[1].op_type, "DequantizeLinear");
        Ok(())
    }

    // A MatMulInteger whose weight operand is dynamic (not in `constants`)
    // must not be fused. This pass only erases *static* quantized weights.
    #[test]
    fn does_not_fuse_dynamic_weight() -> Result<()> {
        let mut graph = GraphProto {
            node: vec![
                NodeProto {
                    op_type: "DynamicQuantizeLinear".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["x_q".to_string(), "x_scale".to_string(), "x_zp".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "MatMulInteger".to_string(),
                    input: vec![
                        "x_q".to_string(),
                        "w_q".to_string(),
                        "x_zp".to_string(),
                        "w_zp".to_string(),
                    ],
                    output: vec!["mmi_out".to_string()],
                    ..Default::default()
                },
                cast_to_float_node("mmi_out", "cast_out"),
                binary_node("Mul", "x_scale", "w_scale", "combined_scale"),
                binary_node("Mul", "cast_out", "combined_scale", "result"),
            ],
            output: vec![graph_output("result")],
            ..Default::default()
        };
        let mut constants = HashMap::new(); // w_q/w_zp deliberately absent

        let fused = fuse_quantized_linear(&mut graph, &mut constants)?;

        assert_eq!(fused, 0);
        assert!(graph.node.iter().any(|n| n.op_type == "MatMulInteger"));
        Ok(())
    }

    // A `MatMulInteger` whose chain is genuinely dead post-fusion (its
    // consuming `Mul` was rewritten away, and nothing else in the graph,
    // per `graph.output`, still needs it) must be deleted outright, not
    // merely left for the general DCE pass, since `eval.rs` can't evaluate
    // it. A second, independent `MatMulInteger` chain that a live graph
    // output still depends on must be left untouched.
    #[test]
    fn removes_unreachable_matmul_integer_after_fusion() -> Result<()> {
        let dead_chain = vec![
            NodeProto {
                op_type: "DynamicQuantizeLinear".to_string(),
                input: vec!["dead_x".to_string()],
                output: vec![
                    "dead_x_q".to_string(),
                    "dead_x_scale".to_string(),
                    "dead_x_zp".to_string(),
                ],
                ..Default::default()
            },
            NodeProto {
                op_type: "MatMulInteger".to_string(),
                input: vec![
                    "dead_x_q".to_string(),
                    "dead_w_q".to_string(),
                    "dead_x_zp".to_string(),
                    "dead_w_zp".to_string(),
                ],
                output: vec!["dead_mmi_out".to_string()],
                ..Default::default()
            },
            cast_to_float_node("dead_mmi_out", "dead_cast_out"),
            binary_node("Mul", "dead_x_scale", "dead_w_scale", "dead_combined_scale"),
            binary_node("Mul", "dead_cast_out", "dead_combined_scale", "dead_out"),
        ];
        let live_chain = vec![
            NodeProto {
                op_type: "DynamicQuantizeLinear".to_string(),
                input: vec!["live_x".to_string()],
                output: vec![
                    "live_x_q".to_string(),
                    "live_x_scale".to_string(),
                    "live_x_zp".to_string(),
                ],
                ..Default::default()
            },
            NodeProto {
                op_type: "MatMulInteger".to_string(),
                input: vec![
                    "live_x_q".to_string(),
                    "live_w_q".to_string(),
                    "live_x_zp".to_string(),
                    "live_w_zp".to_string(),
                ],
                output: vec!["live_mmi_out".to_string()],
                ..Default::default()
            },
            cast_to_float_node("live_mmi_out", "live_cast_out"),
            binary_node("Mul", "live_x_scale", "live_w_scale", "live_combined_scale"),
            binary_node("Mul", "live_cast_out", "live_combined_scale", "result"),
        ];
        let mut node = dead_chain;
        node.extend(live_chain);
        let mut graph = GraphProto {
            node,
            output: vec![graph_output("dead_out"), graph_output("result")],
            ..Default::default()
        };
        // `dead_out`'s own `Mul` gets rewritten into a `MatMul` (fused),
        // dropping its reference to `dead_combined_scale`/`dead_cast_out`
        // and making the rest of its chain unreachable; `result`'s
        // `MatMulInteger`/`w_q`/`w_zp` are deliberately absent from
        // `constants` so its chain is left untouched and stays reachable.
        let mut constants = HashMap::from([
            ("dead_w_q".to_string(), Tensor::new(&[1i16], &Device::Cpu)?),
            ("dead_w_zp".to_string(), Tensor::new(0i16, &Device::Cpu)?),
            (
                "dead_w_scale".to_string(),
                Tensor::new(2.0f32, &Device::Cpu)?,
            ),
        ]);

        fuse_quantized_linear(&mut graph, &mut constants)?;

        assert!(
            !graph
                .node
                .iter()
                .any(|n| n.input.iter().any(|i| i.starts_with("dead_"))
                    && matches!(
                        n.op_type.as_str(),
                        "DynamicQuantizeLinear" | "MatMulInteger"
                    ))
        );
        assert!(
            graph
                .node
                .iter()
                .any(|n| n.output == ["live_mmi_out"] && n.op_type == "MatMulInteger")
        );
        assert!(
            graph
                .node
                .iter()
                .any(|n| n.output == ["live_x_q", "live_x_scale", "live_x_zp"]
                    && n.op_type == "DynamicQuantizeLinear")
        );
        Ok(())
    }
}
