//! Reusable execution state for Crane's ONNX evaluator.

use std::collections::HashMap;

use candle_core::{Result, Tensor};

use super::{
    eval, optimizer,
    optimizer::{OptimizationReport, SessionOptions},
    proto,
};

/// An ONNX model whose initializer tensors are decoded once at load time.
///
/// The graph is still executed by [`eval::simple_eval`], keeping Crane's
/// reusable state separate from the vendored evaluator implementation.
pub struct Session {
    model: proto::ModelProto,
    initializers: HashMap<String, Tensor>,
    optimization_report: OptimizationReport,
}

impl Session {
    pub fn new(model: proto::ModelProto) -> Result<Self> {
        Self::with_options(model, SessionOptions::default())
    }

    pub fn with_options(mut model: proto::ModelProto, options: SessionOptions) -> Result<Self> {
        let graph = model
            .graph
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("no graph defined in proto".to_string()))?;
        let mut initializers = HashMap::with_capacity(graph.initializer.len());

        // Crane Added 20260731: cache parameter tensors instead of decoding
        // the protobuf initializer payload again on every forward pass.
        for initializer in &graph.initializer {
            let tensor = eval::get_tensor(initializer, &initializer.name)?;
            initializers.insert(initializer.name.clone(), tensor);
        }
        // Crane Added 20260731: simplify only the in-memory session graph.
        // The source ONNX file and the vendored evaluator remain unchanged.
        let optimization_report = optimizer::optimize(graph, &mut initializers, &options)?;
        if std::env::var_os("CRANE_ONNX_OPT_REPORT").is_some() {
            eprintln!(
                "Crane ONNX optimized '{}': {} -> {} nodes (folded {}, aliases {}, dead {}, initializers {}, fused_atan2 {}, fused_snake {}, dequantized_int8 {})",
                graph.name,
                optimization_report.original_nodes,
                optimization_report.final_nodes,
                optimization_report.folded_nodes,
                optimization_report.removed_alias_nodes,
                optimization_report.removed_dead_nodes,
                optimization_report.removed_initializers,
                optimization_report.fused_atan2_nodes,
                optimization_report.fused_snake_nodes,
                optimization_report.dequantized_int8_nodes,
            );
        }
        graph.initializer.clear();

        Ok(Self {
            model,
            initializers,
            optimization_report,
        })
    }

    pub fn optimization_report(&self) -> &OptimizationReport {
        &self.optimization_report
    }

    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut values = inputs;
        values.extend(
            self.initializers
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.clone())),
        );
        eval::simple_eval(&self.model, values)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Result, Tensor};

    use super::Session;
    use crate::onnx::proto::{
        GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto::DataType,
    };

    #[test]
    fn folds_small_constant_chains() -> Result<()> {
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![scalar("left", 2.0), scalar("right", 3.0)],
                node: vec![
                    node("Add", &["left", "right"], "sum"),
                    node("Identity", &["sum"], "result"),
                ],
                output: vec![value("result")],
                ..Default::default()
            }),
            ..Default::default()
        };

        let session = Session::new(model)?;
        assert_eq!(session.optimization_report().original_nodes, 2);
        assert_eq!(session.optimization_report().folded_nodes, 2);
        assert_eq!(session.optimization_report().final_nodes, 0);
        let outputs = session.run(HashMap::new())?;
        assert_eq!(outputs["result"].to_scalar::<f32>()?, 5.0);
        Ok(())
    }

    #[test]
    fn removes_dynamic_aliases_and_dead_nodes() -> Result<()> {
        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![value("input")],
                node: vec![
                    node("Identity", &["input"], "alias"),
                    node("Identity", &["alias"], "result"),
                    node("Add", &["input", "input"], "dead"),
                ],
                output: vec![value("result")],
                ..Default::default()
            }),
            ..Default::default()
        };

        let session = Session::new(model)?;
        let report = session.optimization_report();
        assert_eq!(report.removed_alias_nodes, 1);
        assert_eq!(report.removed_dead_nodes, 1);
        assert_eq!(report.final_nodes, 1);
        let input = Tensor::new(7.0f32, &Device::Cpu)?;
        let outputs = session.run(HashMap::from([("input".to_string(), input)]))?;
        assert_eq!(outputs["result"].to_scalar::<f32>()?, 7.0);
        Ok(())
    }

    fn scalar(name: &str, value: f32) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Float as i32,
            float_data: vec![value],
            ..Default::default()
        }
    }

    fn node(operator: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(|input| (*input).to_string()).collect(),
            output: vec![output.to_string()],
            op_type: operator.to_string(),
            ..Default::default()
        }
    }

    fn value(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }
}
