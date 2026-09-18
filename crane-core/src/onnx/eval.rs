use super::ops;
use super::proto::attribute_proto::AttributeType;
use super::proto::tensor_proto::DataType;
use super::proto::{self as onnx, GraphProto};
use super::utils::{collect_all_captures, count_nested_subgraph_captures, release_names_if_done};
use crate::ops::fused_ops;
use candle::{DType, Device, IndexOp, Result, Tensor, bail};
use candle_core as candle;
use candle_core::Module;
use candle_nn::activation::PReLU;
use std::collections::{HashMap, HashSet};

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint32 => Some(DType::U32),
        DataType::Int64 => Some(DType::I64),
        DataType::Float16 => Some(DType::F16),
        // Crane Added 20260827: candle's `from_raw_buffer` already decodes
        // `DType::BF16` via `half::bf16`, matching ONNX's raw byte layout
        // exactly; only the dtype mapping itself was missing. Audio8-TTS's
        // fast/slow AR graphs have bf16 `Constant` nodes.
        DataType::Bfloat16 => Some(DType::BF16),
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        DataType::Bool => Some(DType::U8),
        _ => None,
    }
}

pub(crate) trait Attr {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self>;
}

trait AttrOwned: Sized {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<Self>;
}

impl Attr for i64 {
    const TYPE: AttributeType = AttributeType::Int;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(&attr.i)
    }
}

impl Attr for f32 {
    const TYPE: AttributeType = AttributeType::Float;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(&attr.f)
    }
}

impl Attr for [i64] {
    const TYPE: AttributeType = AttributeType::Ints;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(attr.ints.as_slice())
    }
}

impl Attr for str {
    const TYPE: AttributeType = AttributeType::String;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        std::str::from_utf8(&attr.s).map_err(candle::Error::wrap)
    }
}

impl Attr for GraphProto {
    const TYPE: AttributeType = AttributeType::Graph;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        attr.g
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("attribute does not contain graph".to_string()))
    }
}

impl AttrOwned for Vec<String> {
    const TYPE: AttributeType = AttributeType::Strings;
    fn get(attr: &onnx::AttributeProto) -> Result<Self> {
        let mut ret = vec![];
        for bytes in attr.strings.iter() {
            let s = String::from_utf8(bytes.clone()).map_err(candle::Error::wrap)?;
            ret.push(s);
        }
        Ok(ret)
    }
}

impl AttrOwned for Tensor {
    const TYPE: AttributeType = AttributeType::Tensor;
    fn get(attr: &onnx::AttributeProto) -> Result<Self> {
        let tensor_proto = match &attr.t {
            Some(value) => value,
            None => bail!(
                "attribute {} was of type TENSOR, but no tensor was found",
                attr.name
            ),
        };

        let data_type = match DataType::try_from(tensor_proto.data_type) {
            Ok(value) => value,
            Err(_) => bail!(
                "attribute {} of type TENSOR was an invalid data_type number {}",
                attr.name,
                tensor_proto.data_type
            ),
        };

        let dtype = match dtype(data_type) {
            Some(value) => value,
            None => bail!(
                "attribute {} of type TENSOR has an unsupported data_type {}",
                attr.name,
                data_type.as_str_name()
            ),
        };

        let mut dims = Vec::with_capacity(tensor_proto.dims.len());
        for dim in &tensor_proto.dims {
            if dim < &0 {
                bail!(
                    "attribute {} of type TENSOR has a negative dimension, which is unsupported",
                    attr.name
                )
            }
            dims.push(*dim as usize)
        }

        Tensor::from_raw_buffer(&tensor_proto.raw_data, dtype, &dims, &Device::Cpu)
    }
}

fn get_attr_<'a>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a onnx::AttributeProto> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => {
            bail!(
                "cannot find the '{name}' attribute in '{}' for {}",
                node.op_type,
                node.name
            )
        },
        Some(dt) => Ok(dt),
    }
}

fn get_attr<'a, T: Attr + ?Sized>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a T> {
    let attr = get_attr_(node, name)?;
    if attr.r#type() != T::TYPE {
        bail!(
            "unsupported type {:?} for '{name}' attribute in '{}' for {}",
            attr.r#type,
            node.op_type,
            node.name
        )
    }
    T::get(attr)
}

pub(crate) fn get_attr_opt<'a, T: Attr + ?Sized>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Result<Option<&'a T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        },
    }
}

fn get_attr_opt_owned<T: AttrOwned>(node: &onnx::NodeProto, name: &str) -> Result<Option<T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        },
    }
}

// Crane Added 20260731: ONNX SAME_UPPER padding for Conv exports such as PaddleOCR.
fn same_upper_conv_pads(
    input: &Tensor,
    weight: &Tensor,
    strides: Option<&[i64]>,
    dilations: Option<&[i64]>,
) -> Result<Vec<i64>> {
    let spatial_dims = weight.rank().checked_sub(2).ok_or_else(|| {
        candle::Error::Msg("Conv weight must have at least two dimensions".to_string())
    })?;
    if input.rank() != spatial_dims + 2 {
        bail!(
            "Conv input rank {} does not match weight rank {}",
            input.rank(),
            weight.rank()
        );
    }
    let default_strides = vec![1; spatial_dims];
    let default_dilations = vec![1; spatial_dims];
    let strides = strides.unwrap_or(&default_strides);
    let dilations = dilations.unwrap_or(&default_dilations);
    if strides.len() != spatial_dims || dilations.len() != spatial_dims {
        bail!("Conv SAME_UPPER has invalid strides or dilations")
    }
    let mut begin = Vec::with_capacity(spatial_dims);
    let mut end = Vec::with_capacity(spatial_dims);
    for axis in 0..spatial_dims {
        let stride = strides[axis];
        let dilation = dilations[axis];
        if stride <= 0 || dilation <= 0 {
            bail!("Conv SAME_UPPER requires positive strides and dilations")
        }
        let input_size = input.dims()[axis + 2] as i64;
        let kernel_size = weight.dims()[axis + 2] as i64;
        let output_size = (input_size + stride - 1) / stride;
        let total =
            ((output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size).max(0);
        begin.push(total / 2);
        end.push(total - total / 2);
    }
    begin.extend(end);
    Ok(begin)
}

pub fn get_tensor(t: &onnx::TensorProto, name: &str) -> Result<Tensor> {
    let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
    match DataType::try_from(t.data_type) {
        // Crane Added 20260827: candle has no `I8` dtype, so decode ONNX
        // `Int8` (used by Audio8-TTS's `MatMulInteger` weight operands) into
        // the narrowest signed dtype candle does have, widening losslessly.
        Ok(DataType::Int8) => {
            if t.int32_data.is_empty() {
                let data = t
                    .raw_data
                    .iter()
                    .map(|&b| i16::from(b.cast_signed()))
                    .collect::<Vec<_>>();
                Tensor::from_vec(data, dims.as_slice(), &Device::Cpu)
            } else {
                // Values are ONNX int8, packed one per `int32_data` entry.
                #[allow(clippy::cast_possible_truncation)]
                let data = t.int32_data.iter().map(|&v| v as i16).collect::<Vec<_>>();
                Tensor::from_vec(data, dims.as_slice(), &Device::Cpu)
            }
        },
        Ok(DataType::Int32) => {
            if t.int32_data.is_empty() {
                let len = t.raw_data.len() / 4;
                let data: &[i32] =
                    unsafe { std::slice::from_raw_parts(t.raw_data.as_ptr() as *const i32, len) };
                let data = data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Tensor::from_vec(data, len, &Device::Cpu)
            } else {
                let data = t.int32_data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Tensor::from_vec(data, t.int32_data.len(), &Device::Cpu)
            }
        },
        Ok(dt) => match dtype(dt) {
            Some(dt) => {
                if dt == DType::F32 && !t.float_data.is_empty() {
                    Tensor::from_slice(&t.float_data, dims.as_slice(), &Device::Cpu)
                } else if dt == DType::F64 && !t.double_data.is_empty() {
                    Tensor::from_slice(&t.double_data, dims.as_slice(), &Device::Cpu)
                } else if dt == DType::I64 && !t.int64_data.is_empty() {
                    Tensor::from_slice(&t.int64_data, dims.as_slice(), &Device::Cpu)
                } else {
                    Tensor::from_raw_buffer(
                        t.raw_data.as_slice(),
                        dt,
                        dims.as_slice(),
                        &Device::Cpu,
                    )
                }
            },
            None => {
                bail!("unsupported 'value' data-type {dt:?} for {name}")
            },
        },
        Err(_) => {
            bail!("unsupported 'value' data-type {} for {name}", t.data_type,)
        },
    }
}

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
pub fn simple_eval(
    model: &onnx::ModelProto,
    mut inputs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    let graph = match &model.graph {
        None => bail!("no graph defined in proto"),
        Some(graph) => graph,
    };
    simple_eval_(graph, &mut inputs)
}

pub(crate) fn simple_eval_(
    graph: &onnx::GraphProto,
    values: &mut HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    for t in graph.initializer.iter() {
        let tensor = get_tensor(t, t.name.as_str())?;
        values.insert(t.name.to_string(), tensor);
    }
    for input in graph.input.iter() {
        let input_type = match &input.r#type {
            Some(input_type) => input_type,
            None => continue,
        };
        let input_type = match &input_type.value {
            Some(input_type) => input_type,
            None => continue,
        };
        let tensor_type = match input_type {
            onnx::type_proto::Value::TensorType(tt) => tt,
            _ => continue,
        };

        let tensor = match values.get(&input.name) {
            None => bail!("missing input {}", input.name),
            Some(tensor) => tensor,
        };
        let dt = match DataType::try_from(tensor_type.elem_type) {
            Ok(dt) => match dtype(dt) {
                Some(dt) => dt,
                None => {
                    bail!("unsupported 'value' data-type {dt:?} for {}", input.name)
                },
            },
            type_ => bail!("unsupported input type {type_:?}"),
        };
        match &tensor_type.shape {
            None => continue,
            Some(shape) => {
                if shape.dim.len() != tensor.rank() {
                    bail!(
                        "unexpected rank for {}, got {:?}, expected {:?}",
                        input.name,
                        shape.dim,
                        tensor.shape()
                    )
                }
                for (idx, (d, &dim)) in shape.dim.iter().zip(tensor.dims().iter()).enumerate() {
                    match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            if *v as usize != dim {
                                bail!(
                                    "unexpected dim {idx} for {}, got {:?}, expected {:?}",
                                    input.name,
                                    shape.dim,
                                    tensor.shape()
                                )
                            }
                        },
                        // We do not check equality constraints for the DimParam dimensions for now.
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None => (),
                    }
                }
            },
        };
        if dt != tensor.dtype() {
            bail!(
                "unexpected dtype for {}, got {:?}, expected {dt:?}",
                input.name,
                tensor.dtype()
            )
        }
    }
    // Crane Added 20260731: release intermediates after their last consumer.
    // Upstream simple_eval retained the full graph, which can OOM on image models.
    let graph_outputs = graph
        .output
        .iter()
        .map(|output| output.name.as_str())
        .collect::<HashSet<_>>();
    let mut remaining_uses = HashMap::<&str, usize>::new();
    for node in &graph.node {
        for input in node.input.iter().filter(|input| !input.is_empty()) {
            *remaining_uses.entry(input.as_str()).or_default() += 1;
        }
    }
    // Crane Added 20260804: also count references a nested subgraph (e.g.
    // an "If" node's then_branch/else_branch) makes to a name from this
    // scope. ONNX lets a subgraph reference any name visible in its
    // enclosing scope without declaring it as an explicit node input, so
    // the flat scan just above never sees these — without this, a value
    // could hit a remaining-use count of zero (from its top-level
    // consumers alone) and be evicted before a later "If" node's
    // not-yet-evaluated branch gets to use it. See count_nested_subgraph_captures.
    count_nested_subgraph_captures(graph, &mut remaining_uses);
    // Crane Added 20260804: never evict a value this invocation didn't
    // itself produce. "If" (below) recursively calls simple_eval_ on a
    // branch's subgraph while sharing the same `values` map with the
    // enclosing graph. That recursive call's own last-use bookkeeping is
    // scoped only to the branch's own node list, so without this guard it
    // would free an outer-scope value the moment the branch's *local* view
    // of its uses hits zero — even though the outer graph still needs that
    // value after the "If" node returns.
    let inherited_values: HashSet<String> = values.keys().cloned().collect();

    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => bail!("cannot find {input_name} for op '{}'", node.name),
        };
        let get_opt = |i: usize| {
            node.input
                .get(i)
                .filter(|s: &&String| !s.is_empty())
                .map(|s| get(s))
        };

        // TODO: Validate node.input for each operator.
        match node.op_type.as_str() {
            "Add" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_add(input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "Sub" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_sub(input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "Mul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_mul(input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "Div" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_div(input1)?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260731: implementation lives in ops/modulo.rs.
            "Mod" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = ops::modulo::modulo(node, input0, input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "Pow" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                // HACK: current implementation of broadcast_pow cannot handle negative base,
                // so we use powf where we can, which *does* correctly handle negative base.
                if let Ok(exp) = to_scalar_flexible::<f64>(&input1.to_dtype(DType::F64)?) {
                    let output = input0.powf(exp)?;
                    values.insert(node.output[0].clone(), output);
                } else {
                    let output = input0.broadcast_pow(input1)?;
                    values.insert(node.output[0].clone(), output);
                }
            },
            "Exp" => {
                let xs = get(&node.input[0])?;
                let output = xs.exp()?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260804: new op, needed by a fine-tuned Kokoro
            // backbone's ISTFTNet vocoder — its noise-residual block
            // computes `1 / alpha` (Snake activation) via `x.__rdiv__`,
            // which the ONNX exporter lowers to `Reciprocal`.
            "Reciprocal" => {
                let xs = get(&node.input[0])?;
                let output = ops::reciprocal::reciprocal(xs)?;
                values.insert(node.output[0].clone(), output);
            },
            "Equal" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_eq(input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "Not" => {
                let xs = get(&node.input[0])?;
                let xs = xs.eq(&xs.zeros_like()?)?;
                values.insert(node.output[0].clone(), xs);
            },
            "IsNaN" => {
                let xs = get(&node.input[0])?;
                let output = ops::is_nan::is_nan(xs)?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260731: implementation lives in ops/nonzero.rs.
            "NonZero" => {
                let input = get(&node.input[0])?;
                let output = ops::nonzero::nonzero(node, input)?;
                values.insert(node.output[0].clone(), output);
            },
            "MatMul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_matmul(input1)?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260827: implementation lives in ops/einsum.rs.
            "Einsum" => {
                let equation = get_attr::<str>(node, "equation")?;
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = ops::einsum::einsum(equation, input0, input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "Reshape" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?.to_vec1::<i64>()?;
                // Crane Added 20260731: correctly infer dynamic batched shapes.
                let output = ops::reshape::reshape(input0, &input1)?;
                values.insert(node.output[0].clone(), output);
            },
            "LogSoftmax" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(&axis) => {
                        let axis = input.normalize_axis(axis)?;
                        candle_nn::ops::log_softmax(input, axis)?
                    },
                };
                values.insert(node.output[0].clone(), output);
            },
            "Softmax" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(&axis) => {
                        let axis = input.normalize_axis(axis)?;
                        candle_nn::ops::softmax(input, axis)?
                    },
                };
                values.insert(node.output[0].clone(), output);
            },
            "Transpose" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<[i64]>(node, "perm")? {
                    None => input.t()?,
                    Some(perm) => {
                        let perm = perm.iter().map(|&v| v as usize).collect::<Vec<_>>();
                        input
                            .permute(perm.clone())
                            .and_then(|output| output.contiguous())
                            .map_err(|error| {
                                candle::Error::Msg(format!(
                                    "Transpose node '{}' failed for shape {:?} and perm {:?}: {error}",
                                    node.name,
                                    input.dims(),
                                    perm
                                ))
                            })?
                    },
                };
                values.insert(node.output[0].clone(), output);
            },
            "Dropout" => {
                let input = get(&node.input[0])?;
                // Do not apply dropout at the moment, consider that we're only doing inference.
                values.insert(node.output[0].clone(), input.clone());
            },
            "MaxPool" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let kernel_shape = get_attr::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") | Some("SAME_UPPER") => (),
                    Some(s) => bail!("unsupported MaxPool auto_pad {s} for {}", node.name),
                };
                if let Some(d) = dilations {
                    if d.iter().any(|&v| v != 1) {
                        bail!("MaxPool with dilation != 1, {dilations:?}")
                    }
                }
                if let Some(d) = pads {
                    if d.iter().any(|&v| v != 0) {
                        bail!("MaxPool with pads != 0, {pads:?}")
                    }
                }
                let xs = get(&node.input[0])?;
                let (k1, k2) = match kernel_shape {
                    [k1, k2] => (*k1 as usize, *k2 as usize),
                    _ => bail!("only 2d MaxPool is supported, kernel shape {kernel_shape:?}"),
                };
                let xs = if auto_pad == Some("SAME_UPPER") {
                    let (pad_h_before, pad_h_after, pad_w_before, pad_w_after) =
                        same_upper_pool2d_pads(xs, k1, k2, strides, dilations)?;
                    xs.pad_with_zeros(2, pad_h_before, pad_h_after)?
                        .pad_with_zeros(3, pad_w_before, pad_w_after)?
                } else {
                    xs.clone()
                };
                let ys = match strides {
                    None => xs.max_pool2d((k1, k2))?,
                    Some([s1, s2]) => {
                        xs.max_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?
                    },
                    Some(strides) => bail!("only 2d MaxPool is supported, strides {strides:?}"),
                };
                values.insert(node.output[0].clone(), ys);
            },
            "AveragePool" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let kernel_shape = get_attr::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => bail!("unsupported AveragePool auto_pad {s} for {}", node.name),
                };
                if let Some(d) = dilations {
                    if d.iter().any(|&v| v != 1) {
                        bail!("AvgPool with dilation != 1, {dilations:?}")
                    }
                }
                if let Some(d) = pads {
                    if d.iter().any(|&v| v != 0) {
                        bail!("AvgPool with pads != 0, {pads:?}")
                    }
                }
                let xs = get(&node.input[0])?;
                let (k1, k2) = match kernel_shape {
                    [k1, k2] => (*k1 as usize, *k2 as usize),
                    _ => bail!("only 2d AvgPool is supported, kernel shape {kernel_shape:?}"),
                };
                let ys = match strides {
                    None => xs.avg_pool2d((k1, k2))?,
                    Some([s1, s2]) => {
                        xs.avg_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?
                    },
                    Some(strides) => bail!("only 2d AvgPool is supported, strides {strides:?}"),
                };
                values.insert(node.output[0].clone(), ys);
            },
            // Crane Added 20260731: implementation lives in ops/pooling.rs.
            "GlobalAveragePool" => {
                let input = get(&node.input[0])?;
                let output = ops::pooling::global_average_pool(node, input)?;
                values.insert(node.output[0].clone(), output);
            },
            "BatchNormalization" => {
                let training_mode = get_attr_opt::<i64>(node, "training_mode")?;
                if training_mode.copied().unwrap_or(0) != 0 {
                    bail!("training mode is not supported for BatchNorm")
                }
                let eps = get_attr_opt::<f32>(node, "epsilon")?
                    .copied()
                    .unwrap_or(1e-5);
                let xs = get(&node.input[0])?;
                let weight = get(&node.input[1])?;
                let bias = get(&node.input[2])?;
                let running_mean = get(&node.input[3])?;
                let running_var = get(&node.input[4])?;
                let target_shape: Vec<usize> = xs
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| if idx == 1 { *v } else { 1 })
                    .collect();
                let target_shape = target_shape.as_slice();
                let xs = xs
                    .broadcast_sub(&running_mean.reshape(target_shape)?)?
                    .broadcast_div(&(running_var.reshape(target_shape)? + eps as f64)?.sqrt()?)?;
                let weight = weight.reshape(target_shape)?;
                let bias = bias.reshape(target_shape)?;
                let xs = xs.broadcast_mul(&weight)?.broadcast_add(&bias)?;
                values.insert(node.output[0].clone(), xs);
            },
            // Crane Added 20260731: implementation lives in ops/layer_norm.rs.
            "LayerNormalization" => {
                let x = get(&node.input[0])?;
                let scale = get(&node.input[1])?;
                let bias = get_opt(2).transpose()?;
                let output = ops::layer_norm::layer_norm(node, x, scale, bias)?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260804: implementation lives in
            // ops/instance_norm.rs — needed by a fine-tuned Kokoro
            // backbone's ISTFTNet decoder (F0Ntrain's residual blocks use
            // InstanceNorm1d).
            "InstanceNormalization" => {
                let x = get(&node.input[0])?;
                let scale = get(&node.input[1])?;
                let bias = get(&node.input[2])?;
                let output = ops::instance_norm::instance_normalization(node, x, scale, bias)?;
                values.insert(node.output[0].clone(), output);
            },
            "Squeeze" => {
                let xs = get(&node.input[0])?;
                let axes = get_opt(1).transpose()?;
                // Crane Added 20260731: support axes attributes used by PaddleOCR.
                let output = ops::squeeze::squeeze(node, xs, axes)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape
            "ConstantOfShape" => {
                let input = get(&node.input[0])?;
                let value = get_attr_opt_owned::<Tensor>(node, "value")?.unwrap_or(Tensor::zeros(
                    (),
                    DType::F32,
                    &Device::Cpu,
                )?);

                let shape_vec: Vec<usize> = input
                    .to_vec1::<i64>()?
                    .iter()
                    .map(|&x| x as usize)
                    .collect();

                let xs = Tensor::ones(shape_vec, value.dtype(), input.device())?
                    .broadcast_mul(&value)?;
                values.insert(node.output[0].clone(), xs);
            },
            "Unsqueeze" => {
                let xs = get(&node.input[0])?;
                let axes = match get_attr_opt::<[i64]>(node, "axes")? {
                    Some(axis) => axis.to_vec(),
                    None => get(&node.input[1])?.to_vec1::<i64>()?,
                };
                let mut axes = axes
                    .iter()
                    .map(|&i| {
                        if i == xs.rank() as i64 {
                            Ok(xs.rank())
                        } else if i < 0 {
                            // normalize_axis doesn't work correctly here
                            // because we actually want normalized with respect
                            // to the final size, not the current (off by one)
                            Ok(xs.rank() - (-i as usize) + 1)
                        } else {
                            xs.normalize_axis(i)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                axes.sort();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.unsqueeze(axis)?
                }
                values.insert(node.output[0].clone(), xs);
            },
            "Clip" => {
                let xs = get(&node.input[0])?;
                let xs = if let Some(mins) = get_opt(1) {
                    xs.broadcast_maximum(mins?)?
                } else {
                    xs.clone()
                };
                let xs = if let Some(maxs) = get_opt(2) {
                    xs.broadcast_minimum(maxs?)?
                } else {
                    xs.clone()
                };
                values.insert(node.output[0].clone(), xs);
            },
            "TopK" => {
                let xs = get(&node.input[0])?;
                let k = get(&node.input[1])?;
                let (out_values, out_indices) = ops::topk::top_k(node, xs, k)?;
                values.insert(node.output[0].clone(), out_values);
                if let Some(indices_name) = node.output.get(1).filter(|s| !s.is_empty()) {
                    values.insert(indices_name.clone(), out_indices);
                }
            },
            "Gather" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
                let xs = get(&node.input[0])?;
                let indices = get(&node.input[1])?;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = xs.normalize_axis(axis)?;

                // Crane Added 20260805: normalize indices to I64 up front.
                // index_select's CPU backend only accepts U8/U32/I64, but
                // mid-graph indices (e.g. from Cast/Where/mask ops) can be
                // I32, which would otherwise fail with UnsupportedDTypeForOp.
                let indices = &indices.to_dtype(DType::I64)?;

                // index_select does not support negative indices, so normalize them
                // to positive indices.
                let indices = &{
                    let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
                    let max = Tensor::new(xs.dims()[axis] as i64, indices.device())?
                        .to_dtype(indices.dtype())?;
                    let mask = indices.lt(&zeros)?;
                    mask.to_dtype(indices.dtype())?
                        .broadcast_mul(&max)?
                        .add(indices)?
                };

                // In Pytorch or Numpy this can be done by indexing the xs tensor using the indices
                // tensor directly, but candle does not support tensor indexing at the moment, so
                // some workarounds must be done.
                //
                // Crane Added 20260804: generalize to a single implementation
                // that handles any index rank (previously hit `todo!()` for
                // ranks above 2, needed by a fine-tuned Kokoro backbone that
                // gathers with a 4-D index tensor). index_select only accepts
                // a 1-D index tensor, so flatten `indices`, gather along
                // `axis`, then reshape the result to the ONNX-spec output
                // shape (`data.shape[:axis] + indices.shape + data.shape[axis+1:]`).
                // This also subsumes the previous rank-0/1/2 special cases,
                // which produced identical results through this same path.
                let flat_indices = indices.flatten_all()?;
                let gathered = xs.index_select(&flat_indices, axis)?;
                let mut out_shape = xs.dims()[..axis].to_vec();
                out_shape.extend(indices.dims());
                out_shape.extend(&xs.dims()[axis + 1..]);
                let xs = gathered.reshape(out_shape)?;
                values.insert(node.output[0].clone(), xs);
            },
            // https://onnx.ai/onnx/operators/onnx__GatherElements.html#gatherelements
            // A Note to fellow lurkers:
            // The numpy based `gather_elements` implementation in `onnx` tests [here](https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gatherelements.py)
            // and examples is incorrect.
            // Use `torch.gather` for the validating/ verifying against the proper behaviour
            "GatherElements" => {
                let data = get(&node.input[0])?;
                let indices = get(&node.input[1])?;

                let rank = data.rank();
                if rank != indices.rank() {
                    bail!(
                        "indices must have same rank as input data. Data rank [{}] != indices rank [{}]",
                        data.rank(),
                        indices.rank()
                    );
                }

                let axis = {
                    let axis_i64 = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                    let axis = data.normalize_axis(axis_i64)?;

                    if axis >= rank {
                        bail!(
                            "axis ({}) out of accepted range [-rank, rank-1] which was [-{rank}, {}]",
                            axis_i64,
                            rank - 1
                        )
                    }

                    axis
                };

                // index_select does not support negative indices, so normalize them
                // to positive indices.
                let indices = &{
                    let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
                    let max = Tensor::new(data.dims()[axis] as i64, indices.device())?
                        .to_dtype(indices.dtype())?;
                    let mask = indices.lt(&zeros)?;
                    mask.to_dtype(indices.dtype())?
                        .broadcast_mul(&max)?
                        .add(indices)?
                };

                values.insert(node.output[0].clone(), data.gather(indices, axis)?);
            },
            "Shape" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
                let xs = get(&node.input[0])?;
                let start = get_attr_opt::<i64>(node, "start")?.copied().unwrap_or(0);
                let end = get_attr_opt::<i64>(node, "end")?.copied().unwrap_or(-1);
                let start = xs.normalize_axis(start)?;
                let end = xs.normalize_axis(end)?;
                let mut dims = vec![];
                for idx in start..=end {
                    dims.push(xs.dim(idx)? as i64)
                }
                // Crane Changed 20260827: the output tensor's length is
                // `dims.len()` (the number of dims actually collected by the
                // `start..=end` loop above), not `xs.rank()` — those only
                // coincide when `start`/`end` default to the full range.
                // With explicit `start`/`end` attributes selecting a shorter
                // slice, the old code passed `xs.rank()` as the shape,
                // mismatching `dims`'s actual element count.
                let dims_len = dims.len();
                let dims = Tensor::from_vec(dims, dims_len, xs.device())?;
                values.insert(node.output[0].clone(), dims);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
            "Size" => {
                let data = get(&node.input[0])?;
                let size: usize = data.dims().iter().product();
                let output = Tensor::from_slice(&[size as i64], (), data.device())?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
            "Sqrt" => {
                let xs = get(&node.input[0])?;
                let output = xs.sqrt()?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260731: implementation lives in ops/stft.rs.
            "STFT" => {
                let signal = get(&node.input[0])?;
                let frame_step = get(&node.input[1])?;
                let window = get_opt(2).transpose()?;
                let frame_length = get_opt(3).transpose()?;
                let output = ops::stft::stft(node, signal, frame_step, window, frame_length)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
            "Range" => {
                let start = get(&node.input[0])?;
                let limit = get(&node.input[1])?;
                let delta = get(&node.input[2])?;

                macro_rules! arange_step {
                    ($t: ty) => {
                        Tensor::arange_step(
                            to_vec0_flexible::<$t>(start)?,
                            to_vec0_flexible::<$t>(limit)?,
                            to_vec0_flexible::<$t>(delta)?,
                            &Device::Cpu,
                        )?
                    };
                }

                let output = match start.dtype() {
                    DType::U8 => arange_step!(u8),
                    DType::U32 => arange_step!(u32),
                    DType::I64 => arange_step!(i64),
                    DType::BF16 => arange_step!(f32),
                    DType::F16 => arange_step!(f32),
                    DType::F32 => arange_step!(f32),
                    DType::F64 => arange_step!(f64),
                    DType::F8E4M3 => arange_step!(f32),
                    DType::I32
                    | DType::I16
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F4
                    | DType::F8E8M0 => {
                        bail!("unsupported Range type i32/i16/f6e2m3/f6e3m2/f4/f8e8m0")
                    },
                    dt => bail!("unsupported Range dtype {dt:?}"),
                };

                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
            "Greater" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_gt(b)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
            "Less" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_lt(b)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual
            "LessOrEqual" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_le(b)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual
            "GreaterOrEqual" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_ge(b)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
            "Log" => {
                let a = get(&node.input[0])?;

                let output = a.log()?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
            "Min" => {
                let mut output = get(&node.input[0])?.clone();
                for input in node.input.iter() {
                    let input = get(input)?;
                    output = output.broadcast_minimum(input)?
                }

                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
            "Where" => {
                let cond = get(&node.input[0])?;
                let a = get(&node.input[1])?;
                let b = get(&node.input[2])?;

                // where_cond requires that all inputs are the same shape.
                // In contrast, the Where op in ONNX only requires that they are broadcastable.
                let shape = broadcast_shape_from_many(&[cond.dims(), a.dims(), b.dims()])?;
                let cond = cond.broadcast_as(shape.clone())?;
                let a = a.broadcast_as(shape.clone())?;
                let b = b.broadcast_as(shape)?;
                let output = cond.where_cond(&a, &b)?;
                values.insert(node.output[0].clone(), output);
            },
            "Conv" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let groups = get_attr_opt::<i64>(node, "group")?.copied().unwrap_or(1);
                let _kernel_shape = get_attr_opt::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                let xs = get(&node.input[0])?;
                let ws = get(&node.input[1])?;
                // Crane Added 20260731: PaddleOCR exports Conv with SAME_UPPER.
                let pads = match auto_pad {
                    None | Some("NOTSET") => pads.map(|pads| pads.to_vec()),
                    Some("SAME_UPPER") => Some(same_upper_conv_pads(xs, ws, strides, dilations)?),
                    Some(s) => bail!("unsupported Conv auto_pad {s} for {}", node.name),
                };
                let ys = match ws.rank() {
                    3 => {
                        let (pads, xs) = match pads.as_deref() {
                            None => (0, xs.clone()),
                            Some([p]) => (*p as usize, xs.clone()),
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    (0usize, xs.pad_with_zeros(2, *p1 as usize, *p2 as usize)?)
                                } else {
                                    (*p1 as usize, xs.clone())
                                }
                            },
                            Some(pads) => {
                                bail!("more pads than expected in conv1d {pads:?} {}", node.name)
                            },
                        };
                        let strides = match strides {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                bail!("more strides than expected in conv1d {s:?} {}", node.name)
                            },
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                bail!("more dilations than expected in conv1d {s:?} {}", node.name)
                            },
                        };
                        xs.conv1d(ws, pads, strides, dilations, groups as usize)?
                    },
                    4 => {
                        let (pads, xs) = match pads.as_deref() {
                            None => (0, xs.clone()),
                            Some([p]) => (*p as usize, xs.clone()),
                            Some(&[p1, p2, p3, p4]) => {
                                let p1 = p1 as usize;
                                let p2 = p2 as usize;
                                let p3 = p3 as usize;
                                let p4 = p4 as usize;
                                if p1 != p2 || p1 != p3 || p1 != p4 {
                                    (0, xs.pad_with_zeros(2, p1, p3)?.pad_with_zeros(3, p2, p4)?)
                                } else {
                                    (p1, xs.clone())
                                }
                            },
                            Some(pads) => {
                                bail!("more pads than expected in conv2d {pads:?} {}", node.name)
                            },
                        };
                        // Crane Added 20260731: Candle's kernel has one stride value.
                        // For ONNX anisotropic stride, run stride=1 then select output rows/cols.
                        let (stride, output_stride) = match strides {
                            None => (1, None),
                            Some([p]) => (*p as usize, None),
                            Some([p1, p2]) if p1 == p2 => (*p1 as usize, None),
                            Some([p1, p2]) if *p1 > 0 && *p2 > 0 => {
                                (1, Some((*p1 as usize, *p2 as usize)))
                            },
                            Some(s) => {
                                bail!("more strides than expected in conv2d {s:?} {}", node.name)
                            },
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    bail!(
                                        "dilations have to be the same on both axis {pads:?} {}",
                                        node.name
                                    )
                                }
                                *p1 as usize
                            },
                            Some(s) => {
                                bail!("more dilations than expected in conv2d {s:?} {}", node.name)
                            },
                        };
                        let ys = xs.conv2d(ws, pads, stride, dilations, groups as usize)?;
                        match output_stride {
                            None => ys,
                            Some((stride_h, stride_w)) => {
                                let rows = Tensor::arange_step(
                                    0u32,
                                    ys.dim(2)? as u32,
                                    stride_h as u32,
                                    ys.device(),
                                )?;
                                let cols = Tensor::arange_step(
                                    0u32,
                                    ys.dim(3)? as u32,
                                    stride_w as u32,
                                    ys.device(),
                                )?;
                                ys.index_select(&rows, 2)?.index_select(&cols, 3)?
                            },
                        }
                    },
                    rank => bail!(
                        "unsupported rank for weight matrix {rank} in conv {}",
                        node.name
                    ),
                };
                let ys = if node.input.len() > 2 {
                    let bs = get(&node.input[2])?;
                    let mut bs_shape = vec![1; ys.rank()];
                    bs_shape[1] = bs.elem_count();
                    ys.broadcast_add(&bs.reshape(bs_shape)?)?
                } else {
                    ys
                };
                values.insert(node.output[0].clone(), ys);
            },
            // Crane Added 20260731: implementation lives in ops/conv_transpose.rs.
            "ConvTranspose" => {
                let input = get(&node.input[0])?;
                let weight = get(&node.input[1])?;
                let bias = get_opt(2).transpose()?;
                let output = ops::conv_transpose::conv_transpose(node, input, weight, bias)?;
                values.insert(node.output[0].clone(), output);
            },
            "Concat" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
                let inputs = node
                    .input
                    .iter()
                    .map(|n| Ok(get(n.as_str())?.clone()))
                    .collect::<Result<Vec<Value>>>()?;
                let axis: i64 = *get_attr(node, "axis")?;
                if inputs.is_empty() {
                    bail!("empty concat")
                };
                // Find minimum rank among inputs and squeeze trailing singleton dims to match
                let min_rank = inputs.iter().map(|t| t.rank()).min().unwrap();
                let inputs: Vec<_> = inputs
                    .into_iter()
                    .map(|t| {
                        let mut t = t;
                        while t.rank() > min_rank {
                            let last_dim = t.rank() - 1;
                            if t.dims()[last_dim] == 1 {
                                t = t.squeeze(last_dim).unwrap_or(t);
                            } else {
                                break;
                            }
                        }
                        t
                    })
                    .collect();
                let axis = inputs[0].normalize_axis(axis)?;
                let output = Tensor::cat(&inputs, axis).map_err(|e| {
                    let shapes: Vec<_> = inputs.iter().map(|t| format!("{:?}", t.dims())).collect();
                    candle::Error::Msg(format!(
                        "Concat failed for node '{}': {} (input shapes: {:?})",
                        node.name, e, shapes
                    ))
                })?;
                values.insert(node.output[0].clone(), output);
            },
            "Abs" => {
                let input = get(&node.input[0])?;
                let output = input.abs()?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260731: implementation lives in ops/atan.rs.
            "Atan" => {
                let input = get(&node.input[0])?;
                let output = ops::atan::atan(input)?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260731: fused atan2(y, x) — produced by the
            // optimizer's atan2-decomposition fusion pass, not present in
            // the original ONNX model.
            "Atan2" => {
                let y = get(&node.input[0])?;
                let x = get(&node.input[1])?;
                let shape = broadcast_shape(y.dims(), x.dims())?;
                let y = y.broadcast_as(shape.clone())?;
                let x = x.broadcast_as(shape)?;
                let output = fused_ops::atan2::atan2(&y, &x)?;
                values.insert(node.output[0].clone(), output);
            },
            "Cos" => {
                let input = get(&node.input[0])?;
                let output = input.cos()?;
                values.insert(node.output[0].clone(), output);
            },
            "Sin" => {
                let input = get(&node.input[0])?;
                let output = input.sin()?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260806: fused snake(x, alpha) — produced by the
            // optimizer's snake-decomposition fusion pass, not present in
            // the original ONNX model.
            "Snake" => {
                let x = get(&node.input[0])?;
                let alpha = get(&node.input[1])?;
                let shape = broadcast_shape(x.dims(), alpha.dims())?;
                let x = x.broadcast_as(shape.clone())?;
                let alpha = alpha.broadcast_as(shape)?;
                let output = fused_ops::snake::snake(&x, &alpha)?;
                values.insert(node.output[0].clone(), output);
            },
            "Neg" => {
                let input = get(&node.input[0])?;
                // neg() not implemented for i64, work around with multiply by -1
                let output = if input.dtype() == DType::I64 {
                    let minus_one =
                        Tensor::new(&[-1i64], input.device())?.broadcast_as(input.shape())?;
                    input.mul(&minus_one)?
                } else {
                    input.neg()?
                };
                values.insert(node.output[0].clone(), output);
            },
            "Erf" => {
                let input = get(&node.input[0])?;
                let output = input.erf()?;
                values.insert(node.output[0].clone(), output);
            },
            "Tanh" => {
                let input = get(&node.input[0])?;
                let output = input.tanh()?;
                values.insert(node.output[0].clone(), output);
            },
            "Sigmoid" => {
                let input = get(&node.input[0])?;
                let output = candle_nn::ops::sigmoid(input)?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260731: implementation lives in ops/activation.rs.
            "HardSigmoid" => {
                let input = get(&node.input[0])?;
                let output = ops::activation::hard_sigmoid(node, input)?;
                values.insert(node.output[0].clone(), output);
            },
            "Gelu" => {
                let input = get(&node.input[0])?;
                let output = input.gelu_erf()?;
                values.insert(node.output[0].clone(), output);
            },
            // Crane Added 20260827: implementation lives in ops/softplus.rs.
            "Softplus" => {
                let input = get(&node.input[0])?;
                let output = ops::softplus::softplus(input)?;
                values.insert(node.output[0].clone(), output);
            },
            "Relu" => {
                let input = get(&node.input[0])?;
                let output = input.relu()?;
                values.insert(node.output[0].clone(), output);
            },
            "PRelu" => {
                // https://onnx.ai/onnx/operators/onnx__PRelu.html
                let input = get(&node.input[0])?;
                let slope = get(&node.input[1])?;

                let output = PReLU::new(slope.clone(), false).forward(input)?;
                values.insert(node.output[0].clone(), output);
            },
            "Ceil" => {
                let input = get(&node.input[0])?;
                let output = input.ceil()?;
                values.insert(node.output[0].clone(), output);
            },
            "Floor" => {
                let input = get(&node.input[0])?;
                let output = input.floor()?;
                values.insert(node.output[0].clone(), output);
            },
            // https://onnx.ai/onnx/operators/onnx__Round.html
            "Round" => {
                let input = get(&node.input[0])?;
                let output = input.round()?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
            "Constant" => {
                let value = match node.attribute.iter().find(|attr| attr.name == "value") {
                    None => {
                        // TODO: support sparse_value etc.
                        bail!("cannot find 'value' attr in 'Constant' for {}", node.name)
                    },
                    Some(value) => value,
                };
                let output = match value.r#type() {
                    AttributeType::Tensor => {
                        let t = value.t.as_ref().unwrap();
                        get_tensor(t, &node.name)?
                    },
                    rtype => bail!("unsupported 'value' type {rtype:?} for {}", node.name),
                };

                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
            "Cast" => {
                let input = get(&node.input[0])?;
                let dt: i64 = *get_attr(node, "to")?;
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(DataType::Int32) => DType::I64,
                    Ok(dt) => match dtype(dt) {
                        Some(dt) => dt,
                        None => {
                            bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                        },
                    },
                    Err(_) => {
                        bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                    },
                };
                let output = input.to_dtype(dtype)?;
                values.insert(node.output[0].clone(), output);
            },
            "CastLike" => {
                let input = get(&node.input[0])?;
                let target_type = get(&node.input[1])?;
                let output = ops::cast_like::cast_like(input, target_type)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
            "CumSum" => {
                let exclusive = get_attr_opt::<i64>(node, "exclusive")?
                    .copied()
                    .unwrap_or(0);
                let reverse = get_attr_opt::<i64>(node, "reverse")?.copied().unwrap_or(0);
                if exclusive != 0 {
                    bail!("only exclusive == 0 is supported in CumSum")
                }
                if reverse != 0 {
                    bail!("only reverse == 0 is supported in CumSum")
                }
                let input = get(&node.input[0])?;
                let axis = to_vec0_flexible::<u32>(&get(&node.input[1])?.to_dtype(DType::U32)?)?;
                let output = input.cumsum(axis as usize)?;
                values.insert(node.output[0].clone(), output);
            },
            //  https://github.com/onnx/onnx/blob/main/docs/Operators.md#flatten
            "Flatten" => {
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(1) as usize;
                let input = get(&node.input[0])?;
                let first_part: usize = input.shape().dims().iter().take(axis).product();
                let end_index = input.shape().dims().iter().product::<usize>();
                let new_shape = (first_part, end_index / first_part);
                let output = input.reshape(new_shape)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
            "Identity" => {
                let input = get(&node.input[0])?;
                values.insert(node.output[0].clone(), input.clone());
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#if
            "If" => {
                // protobuf encodes boolean false as 0 and true as 1
                let cond = to_scalar_flexible::<u8>(&get(&node.input[0])?.get(0)?)?;
                let (taken_attr, untaken_attr) = if cond != 0 {
                    ("then_branch", "else_branch")
                } else {
                    ("else_branch", "then_branch")
                };
                let sub_graph = get_attr::<GraphProto>(node, taken_attr)?;
                if sub_graph.output.len() != node.output.len() {
                    bail!(
                        "If node {:?} is malformed: branch outputs ({}) don't match node outputs ({})",
                        node.name,
                        sub_graph.output.len(),
                        node.output.len()
                    );
                }
                let branch_out = simple_eval_(sub_graph, values)?;
                for (i, out) in node.output.iter().enumerate() {
                    values.insert(
                        out.clone(),
                        branch_out.get(&sub_graph.output[i].name).unwrap().clone(),
                    );
                }
                // Crane Added 20260806: release the taken branch's captured
                // (outer-scope) references, recursively through any nested
                // subgraphs it contains (e.g. an inner "If"), now that it
                // has actually run — matches the up-front over-count in
                // count_nested_subgraph_captures, which counts both
                // branches (and nested subgraphs within them) since the
                // taken one isn't known until here.
                let mut taken_captures = Vec::new();
                collect_all_captures(sub_graph, &mut taken_captures);
                release_names_if_done(
                    taken_captures,
                    &mut remaining_uses,
                    &graph_outputs,
                    &inherited_values,
                    values,
                );
                // Crane Added 20260806: the untaken branch's captures were
                // also counted up front (count_nested_subgraph_captures
                // can't know which branch will be taken) but that branch
                // never runs, so its counts would otherwise never be
                // released. The attribute may be absent (an "If" is valid
                // with only one branch set), hence get_attr_opt.
                if let Some(untaken_graph) = get_attr_opt::<GraphProto>(node, untaken_attr)? {
                    let mut untaken_captures = Vec::new();
                    collect_all_captures(untaken_graph, &mut untaken_captures);
                    release_names_if_done(
                        untaken_captures,
                        &mut remaining_uses,
                        &graph_outputs,
                        &inherited_values,
                        values,
                    );
                }
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#pad
            "Pad" => {
                let mode = get_attr_opt(node, "mode")?.unwrap_or("constant");
                let data = get(&node.input[0])?;
                let pads = get(&node.input[1])?;
                // Crane Changed 20260827: this arm used to bail on a 3rd
                // input at all. `constant_value` (input 2) is optional per
                // the ONNX spec and only meaningful for `mode="constant"`;
                // `get_opt` already treats an empty input name (the standard
                // "omitted" marker) as absent, so a 3-input node with an
                // omitted `constant_value` (as Audio8-TTS's codec decoder
                // emits) no longer needs to be rejected.
                let constant_value = get_opt(2).transpose()?;
                let output = ops::pad::pad(node, data, pads, mode, constant_value)?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice
            "Slice" => {
                let data = get(&node.input[0])?;
                let starts = get(&node.input[1])?;
                let ends = get(&node.input[2])?;
                let default_axes;
                let default_steps;
                let axes: &Tensor;
                let steps: &Tensor;
                // If axes are omitted, they are set to [0, ..., r-1]. If steps are omitted,
                // they are set to [1, ..., 1] of length len(starts)
                match node.input.len() {
                    3 => {
                        let len = starts.dims()[0];
                        default_axes = Some(Tensor::arange(0, len as i64, starts.device())?);
                        axes = default_axes.as_ref().unwrap();
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    },
                    4 => {
                        let len = starts.dims()[0];
                        axes = get(&node.input[3])?;
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    },
                    5 => {
                        steps = get(&node.input[4])?;
                        axes = get(&node.input[3])?;
                    },
                    _ => bail!(
                        "Slice node is invalid, expected 3-5 inputs, got {}: {:?}",
                        node.input.len(),
                        node
                    ),
                }

                let mut out = data.clone();
                for (i, axis) in axes.to_vec1::<i64>()?.into_iter().enumerate() {
                    // All negative elements of axes are made non-negative by
                    // adding r to them, where r = rank(input).
                    let axis = if axis < 0 {
                        axis + data.rank() as i64
                    } else {
                        axis
                    } as usize;

                    let data_dim = data.dims()[axis] as i64;
                    let mut s = to_scalar_flexible::<i64>(&starts.get(i)?)?;
                    let mut e = to_scalar_flexible::<i64>(&ends.get(i)?)?;
                    // All negative values in starts[i] and ends[i] have
                    // dims[axes[i]] added to them, where dims are the
                    // dimensions of input.
                    if s < 0 {
                        s += data_dim;
                    }
                    if e < 0 {
                        e += data_dim;
                    }

                    let p = to_scalar_flexible::<i64>(&steps.get(i)?)?;
                    // starts[i] is clamped into the range [0, dims[axes[i]]]
                    // for positive stepping and [0, dims[axes[i]]-1] for
                    // negative stepping.
                    // for positive stepping ends[axes[i]] is clamped to
                    // [0, dims[axes[i]]], while for negative stepping it is
                    // clamped to [-1, dims[axes[i]]-1].
                    if p >= 0 {
                        s = s.clamp(0, data_dim);
                        e = e.clamp(0, data_dim);
                    } else {
                        s = s.clamp(0, data_dim - 1);
                        e = e.clamp(-1, data_dim - 1);
                    }

                    let indexes = Tensor::arange_step(s, e, p, data.device())?;
                    out = out.contiguous()?.index_select(&indexes, axis)?
                }
                values.insert(node.output[0].clone(), out);
            },
            // https://onnx.ai/onnx/operators/onnx__ReduceMax.html#reducemax
            "ReduceMax" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1) == 1;

                let axes = if let Some(Ok(axes)) = axes {
                    // Satisfies version 18+
                    axes.to_vec1::<i64>().ok()
                } else if let Ok(Some(axes)) = get_attr_opt::<[i64]>(node, "axes") {
                    // Backward compatibility with version 13 and below
                    Some(axes.to_vec())
                } else {
                    None
                };

                let axes = if let Some(axes) = axes {
                    let rank = input.rank();
                    let mut axes_set = HashSet::new();

                    let mut axes = axes
                        .iter()
                        .map(|a| {
                            let axis = if *a < 0 {
                                (rank as i64 + *a) as usize
                            } else {
                                *a as usize
                            };

                            axes_set.insert(axis);
                            axis
                        })
                        .collect::<Vec<_>>();

                    if axes_set.len() < axes.len() {
                        bail!("Duplicate value in 'axes'");
                    }

                    if axes.len() > 1 {
                        axes.sort();
                    }

                    Some(axes)
                } else {
                    None
                };

                // TODO: Handle empty set
                // Definition:
                // "Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise"
                // For now, this will throw an error
                if input.elem_count() == 0 {
                    bail!("reduction over zero-size tensor not supported");
                }

                let output = if let Some(axes) = axes {
                    let mut result = input.clone();
                    for &axis in axes.iter().rev() {
                        result = if keepdims {
                            result.max_keepdim(axis)?
                        } else {
                            result.max(axis)?
                        }
                    }

                    result
                } else {
                    // If `axes` is empty and `noop_with_empty_axes` is set to `true (1)`
                    // ""input tensor will not be reduced,and the output tensor would be equivalent to input tensor.""
                    if get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied() == Some(1) {
                        input.clone()
                    } else {
                        let mut result = input.flatten_all()?;
                        if keepdims {
                            result = result.max_keepdim(0)?;
                            // If keepdims is true, reshape to match input dimensions
                            let shape = vec![1; input.rank()];
                            result.reshape(shape)?
                        } else {
                            result.max(0)?
                        }
                    }
                };

                values.insert(node.output[0].clone(), output);
            },
            // https://onnx.ai/onnx/operators/onnx__ReduceMean.html#reducemean-13
            // TODO: This version is only compatible with ReduceMean V13 and below.
            "ReduceMean" => {
                let input = get(&node.input[0])?;
                let axes = get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);

                let n_dims = input.dims().len();

                let axes: Vec<usize> = if let Some(axes) = axes {
                    axes.iter()
                        .map(|e| (if e < &0 { (n_dims as i64) + *e } else { *e }) as usize)
                        .collect()
                } else {
                    (0..n_dims).collect()
                };
                let output = if keepdims == 1 {
                    input.mean_keepdim(axes)?
                } else {
                    input.mean(axes)?
                };
                values.insert(node.output[0].clone(), output);
            },
            // https://onnx.ai/onnx/operators/onnx__ReduceMin.html#reducemin
            "ReduceMin" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1) == 1;

                let axes = if let Some(Ok(axes)) = axes {
                    // Satisfies version 18+
                    axes.to_vec1::<i64>().ok()
                } else if let Ok(Some(axes)) = get_attr_opt::<[i64]>(node, "axes") {
                    // Backward compatibility with version 13 and below
                    Some(axes.to_vec())
                } else {
                    None
                };

                let axes = if let Some(axes) = axes {
                    let rank = input.rank();
                    let mut axes_set = HashSet::new();

                    let mut axes = axes
                        .iter()
                        .map(|a| {
                            let axis = if *a < 0 {
                                (rank as i64 + *a) as usize
                            } else {
                                *a as usize
                            };

                            axes_set.insert(axis);
                            axis
                        })
                        .collect::<Vec<_>>();

                    if axes_set.len() < axes.len() {
                        bail!("Duplicate value in 'axes'");
                    }

                    if axes.len() > 1 {
                        axes.sort();
                    }

                    Some(axes)
                } else {
                    None
                };

                // TODO: Handle empty set
                // Definition:
                // "Reduction over an empty set of values yields positive infinity (if supported by the datatype) or the max value of the data type otherwise"
                // For now, this will throw an error
                if input.elem_count() == 0 {
                    bail!("reduction over zero-size tensor not supported");
                }

                let output = if let Some(axes) = axes {
                    let mut result = input.clone();
                    for &axis in axes.iter().rev() {
                        result = if keepdims {
                            result.min_keepdim(axis)?
                        } else {
                            result.min(axis)?
                        }
                    }

                    result
                } else {
                    // If `axes` is empty and `noop_with_empty_axes` is set to `true (1)`
                    // ""input tensor will not be reduced,and the output tensor would be equivalent to input tensor.""
                    if get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied() == Some(1) {
                        input.clone()
                    } else {
                        let mut result = input.flatten_all()?;
                        if keepdims {
                            result = result.min_keepdim(0)?;
                            // If keepdims is true, reshape to match input dimensions
                            let shape = vec![1; input.rank()];
                            result.reshape(shape)?
                        } else {
                            result.min(0)?
                        }
                    }
                };

                values.insert(node.output[0].clone(), output);
            },
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
            // Version 18 impl
            "Split" => {
                let input_tensor = get(&node.input[0])?;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = input_tensor.normalize_axis(axis)?;

                // Determine split sizes
                let splits = if node.input.len() > 1 {
                    // If the split tensor is provided, use it to determine sizes
                    let split_tensor = get(&node.input[1])?.to_vec1::<i64>()?;
                    split_tensor.iter().map(|&x| x as usize).collect::<Vec<_>>()
                } else {
                    let num_outputs = if let Some(&num_outputs_attrib) =
                        get_attr_opt::<i64>(node, "num_outputs")?
                    {
                        num_outputs_attrib as usize
                    } else {
                        node.output.len()
                    };

                    let input_dim = input_tensor.dim(axis)?;

                    let mut split_sizes =
                        vec![input_dim / num_outputs as usize; num_outputs as usize];
                    let remainder = input_dim % num_outputs as usize;
                    if remainder > 0 {
                        // If there's a remainder, add it to the last split size
                        split_sizes[num_outputs as usize - 1] += remainder;
                    }

                    split_sizes
                };

                // Perform the split operation
                let mut outputs = vec![];
                let mut start = 0;
                for &size in &splits {
                    let end = start + size;
                    let slice = input_tensor.narrow(axis, start, size)?;
                    outputs.push(slice);
                    start = end;
                }

                // Insert the split outputs into the values map
                for (output, slice) in node.output.iter().zip(outputs.into_iter()) {
                    values.insert(output.clone(), slice);
                }
            },
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
            // Version 13 impl
            "Expand" => {
                // unlike broadcast_to, expand allows for the output shape to
                // be different from the specified shape.
                let input_tensor = get(&node.input[0])?;
                let input_shape = get(&node.input[1])?;

                // Check that the shape tensor is 1D
                if input_shape.rank() != 1 {
                    bail!(
                        "Expand expects 'shape' input to be 1D tensor: {:?}",
                        input_shape
                    );
                }
                let input_tensor_dims = input_tensor.dims();
                let input_shape_dims = input_shape
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<_>>();

                let target_shape = broadcast_shape(input_tensor_dims, input_shape_dims.as_slice())?;

                let expanded_tensor = input_tensor.broadcast_as(target_shape)?;

                values.insert(node.output[0].clone(), expanded_tensor);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile
            "Tile" => {
                let input = get(&node.input[0])?;
                let repeats = get(&node.input[1])?.to_vec1::<i64>()?;

                let mut result = input.clone();
                for (dim, &repeat) in repeats.iter().enumerate() {
                    if repeat > 1 {
                        let repeat = repeat as usize;
                        let tensors: Vec<_> = (0..repeat).map(|_| result.clone()).collect();
                        result = Tensor::cat(&tensors, dim)?;
                    }
                }
                values.insert(node.output[0].clone(), result);
            },
            "ReduceProd" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1).and_then(Result::ok);
                let output = ops::reduce_prod::reduce_prod(node, input, axes)?;
                values.insert(node.output[0].clone(), output);
            },
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
            // Version 13 impl
            "ReduceSum" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let axes = match axes {
                    Some(Ok(axes)) => axes
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    Some(Err(_)) | None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank()).collect()
                        }
                    },
                };

                let output = if keepdims == 1 {
                    input.sum_keepdim(axes)?
                } else {
                    input.sum(axes)?
                };

                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
            // Version 18 impl
            "ReduceL2" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let input_sq = input.sqr()?;

                let axes = match axes {
                    Some(axes) => axes?
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input_sq.rank()).collect()
                        }
                    },
                };

                let output = if keepdims == 1 {
                    input_sq.sum_keepdim(axes)?.sqrt()?
                } else {
                    input_sq.sum(axes)?.sqrt()?
                };

                values.insert(node.output[0].clone(), output);
            },
            random_type @ ("RandomUniform" | "RandomNormal") => {
                let dtype = ops::random::parse_random_float_dtype(node, random_type, DType::F32)?;
                ops::random::reject_random_seed(node, random_type)?;
                let shape: Vec<usize> = get_attr::<[i64]>(node, "shape")?
                    .iter()
                    .map(|x| *x as usize)
                    .collect();
                let output = if random_type == "RandomUniform" {
                    let low: f32 = get_attr_opt(node, "low")?.copied().unwrap_or(0.0);
                    let high: f32 = get_attr_opt(node, "high")?.copied().unwrap_or(1.0);
                    Tensor::rand(low, high, shape, &Device::Cpu)?.to_dtype(dtype)?
                } else {
                    let mean: f32 = get_attr_opt(node, "mean")?.copied().unwrap_or(0.0);
                    let scale: f32 = get_attr_opt(node, "scale")?.copied().unwrap_or(1.0);
                    Tensor::randn(mean, scale, shape, &Device::Cpu)?.to_dtype(dtype)?
                };
                values.insert(node.output[0].clone(), output);
            },
            "RandomUniformLike" => {
                let input = get(&node.input[0])?;
                let output = ops::random::random_uniform_like(node, input)?;
                values.insert(node.output[0].clone(), output);
            },
            "RandomNormalLike" => {
                let input = get(&node.input[0])?;
                let output = ops::random::random_normal_like(node, input)?;
                values.insert(node.output[0].clone(), output);
            },
            "ArgMin" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    bail!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = get_attr_opt(node, "keepdims")?.copied().unwrap_or(1);
                let select_last_index: i64 = get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    bail!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmin_keepdim(axis)?
                } else {
                    input.argmin(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            },
            "ArgMax" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    bail!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = get_attr_opt(node, "keepdims")?.copied().unwrap_or(1);
                let select_last_index: i64 = get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    bail!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmax_keepdim(axis)?
                } else {
                    input.argmax(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            },
            "LeakyRelu" => {
                let input = get(&node.input[0])?;
                let dt = input.dtype();
                match dt {
                    DType::U8
                    | DType::U32
                    | DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F4
                    | DType::F8E8M0 => {
                        bail!(
                            "unsupported dtype {}, only float types are allowed for LeakyRelu",
                            dt.as_str()
                        )
                    },
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 | DType::F8E4M3 => {},
                    dt => bail!("unsupported dtype {dt:?} for LeakyRelu"),
                }
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(0.01);
                let output = candle_nn::ops::leaky_relu(input, alpha.into())?;
                values.insert(node.output[0].clone(), output);
            },
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
            "Gemm" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;
                let c = get(&node.input[2])?;

                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                let beta = get_attr_opt::<f32>(node, "beta")?.copied().unwrap_or(1.0);

                let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
                let beta = Tensor::full(beta, c.shape(), &Device::Cpu)?;

                let trans_a = get_attr_opt::<i64>(node, "transA")?.copied().unwrap_or(0);
                let trans_b = get_attr_opt::<i64>(node, "transB")?.copied().unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t()? };
                let b = if trans_b == 0 { b.clone() } else { b.t()? };

                let output = a
                    .broadcast_mul(&alpha)?
                    .broadcast_matmul(&b)?
                    .broadcast_add(&c.broadcast_mul(&beta)?)?;
                values.insert(node.output[0].clone(), output);
            },
            "LSTM" => {
                let direction = get_attr_opt(node, "direction")?.unwrap_or("forward");
                if direction != "forward" {
                    bail!("LSTM currently only supports direction == \"forward\"");
                }
                let num_directions = if direction == "bidirectional" { 2 } else { 1 };
                let hidden_size: i64 = get_attr(node, "hidden_size").copied()?;
                let input_forget = get_attr_opt(node, "input_forget")?.copied().unwrap_or(0);
                if input_forget != 0 {
                    bail!("LSTM currently only supports input_forget == 0");
                }
                let activations_default = vec![
                    "Sigmoid".to_string(),
                    "Tanh".to_string(),
                    "Tanh".to_string(),
                ];
                let activations = get_attr_opt_owned::<Vec<String>>(node, "activations")?
                    .unwrap_or(activations_default.clone());
                if activations != activations_default {
                    bail!(
                        "LSTM currently only supports default activations ({activations_default:?})"
                    );
                }
                // activation_alpha and activation_beta don't apply to (Sigmoid, Tanh, Tanh) so ignoring them is okay
                if get_attr_opt::<f32>(node, "clip")?.is_some() {
                    bail!("LSTM does not currently support clip attribute");
                }

                // The shape format of inputs X, initial_h and outputs Y, Y_h.
                // If 0, the following shapes are expected:
                //     X.shape = [seq_length, batch_size, input_size],
                //     Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                //     initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
                // If 1, the following shapes are expected:
                //     X.shape = [batch_size, seq_length, input_size],
                //     Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                //     initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
                let layout = get_attr_opt(node, "layout")?.copied().unwrap_or(0);
                if layout != 0 {
                    bail!("LSTM currently only supports layout == 0");
                }

                // The input sequences packed (and potentially padded) into one 3-D tensor
                // with the shape of `[seq_length, batch_size, input_size]`.
                let x = get(&node.input[0])?;
                // XXX: depends on layout
                let (seq_length, batch_size, input_size) = x.dims3()?;
                // The weight tensor for the gates.
                // Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along dimension 0.
                // The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
                let w = get(&node.input[1])?;
                // The recurrence weight tensor.
                // Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0.
                // This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
                let r = get(&node.input[2])?;

                // The bias tensor for input gate.
                // Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0.
                // This tensor has shape `[num_directions, 8*hidden_size]`.
                // Optional: If not specified - assumed to be 0.
                let b_default: Tensor;
                let b = match get_opt(3) {
                    Some(n) => n?,
                    None => {
                        b_default = Tensor::zeros(
                            (num_directions, 8 * hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &b_default
                    },
                };

                // Optional tensor specifying lengths of the sequences in a batch.
                // If not specified - assumed all sequences in the batch to have length `seq_length`.
                // It has shape `[batch_size]`.
                let seq_lens_default: Tensor;
                let seq_lens = match get_opt(4) {
                    Some(n) => n?,
                    None => {
                        seq_lens_default =
                            Tensor::full(seq_length as i64, (batch_size,), x.device())?;
                        &seq_lens_default
                    },
                };
                let seq_lens_is_default =
                    (seq_lens.to_vec1::<i64>()?.iter()).all(|e| *e as usize == seq_length);
                if !seq_lens_is_default {
                    bail!("LSTM currently only supports default value of seq_lens");
                }

                // Optional initial value of the hidden. If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_h_default: Tensor;
                let initial_h = match get_opt(5) {
                    Some(n) => n?,
                    _ => {
                        initial_h_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_h_default
                    },
                };

                // Optional initial value of the cell.
                // If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_c_default: Tensor;
                let initial_c = match node.input.get(6) {
                    Some(n) if !n.is_empty() => get(n)?,
                    _ => {
                        initial_c_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_c_default
                    },
                };

                // The weight tensor for peepholes.
                // Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0.
                // It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
                let p_default = Tensor::zeros(
                    (num_directions, 3 * hidden_size as usize),
                    DType::F32,
                    x.device(),
                )?;
                let p = get_opt(7).unwrap_or(Ok(&p_default))?;
                let p_is_zeros = (p.to_vec2::<f32>()?.iter()).all(|v| v.iter().all(|e| *e == 0.0));
                if !p_is_zeros {
                    bail!(
                        "LSTM currently only supports default value of p (a Tensor of all zeroes)"
                    );
                }

                // these all have [num_directions, ...] shapes
                let w = w.get(0)?; // w[iofc] has shape [4*hidden_size, input_size]
                let r = r.get(0)?; // r[iofc] has shape [4*hidden_size, hidden_size]
                let b = b.get(0)?; // concat of [wb[iofc],rb[iofc]] has shape [8*hidden_size]
                let idx_wb = Tensor::arange(0, 4 * hidden_size, x.device())?;
                let idx_rb = Tensor::arange(4 * hidden_size, 8 * hidden_size, x.device())?;
                let wb = b.index_select(&idx_wb, 0)?;
                let rb = b.index_select(&idx_rb, 0)?;
                let c = initial_c.get(0)?;
                let h = initial_h.get(0)?;

                // w, r, wb, rb are all iofc but lstm expects ifco
                // so we need to move some stuff around
                let idx_i = Tensor::arange(0, hidden_size, x.device())?;
                let idx_o = Tensor::arange(hidden_size, 2 * hidden_size, x.device())?;
                let idx_f = Tensor::arange(2 * hidden_size, 3 * hidden_size, x.device())?;
                let idx_c = Tensor::arange(3 * hidden_size, 4 * hidden_size, x.device())?;
                let idx_ifco = Tensor::cat(&[&idx_i, &idx_f, &idx_c, &idx_o], 0)?;
                let w = w.index_select(&idx_ifco, 0)?;
                let r = r.index_select(&idx_ifco, 0)?;
                let wb = wb.index_select(&idx_ifco, 0)?;
                let rb = rb.index_select(&idx_ifco, 0)?;
                let vmap = candle_nn::VarMap::new();
                vmap.data().lock().unwrap().extend([
                    ("weight_ih_l0".to_string(), candle::Var::from_tensor(&w)?),
                    ("weight_hh_l0".to_string(), candle::Var::from_tensor(&r)?),
                    ("bias_ih_l0".to_string(), candle::Var::from_tensor(&wb)?),
                    ("bias_hh_l0".to_string(), candle::Var::from_tensor(&rb)?),
                ]);
                use candle_nn::rnn::RNN as _;
                let lstm = candle_nn::rnn::lstm(
                    input_size,
                    hidden_size as usize,
                    candle_nn::rnn::LSTMConfig::default(),
                    candle_nn::VarBuilder::from_varmap(&vmap, w.dtype(), w.device()),
                )?;

                let mut lstm_state = candle_nn::rnn::LSTMState::new(h, c);
                let mut h_acc = if node.output.first().map(String::as_str).unwrap_or("") != "" {
                    Some(vec![])
                } else {
                    None
                };
                for t in 0..seq_length {
                    let x = x.get(t)?;
                    lstm_state = lstm.step(&x, &lstm_state)?;
                    if let Some(h_acc) = &mut h_acc {
                        h_acc.push(lstm_state.clone());
                    }
                }

                assert_eq!(
                    num_directions, 1,
                    "if support for bidirectional is ever added, outputs will have to be concatenated, not simply reshaped"
                );
                if let Some(name) = node.output.first() {
                    let h_acc = h_acc.as_ref().unwrap();
                    let h_acc = lstm.states_to_tensor(h_acc)?;
                    let h_acc = h_acc.reshape((
                        seq_length,
                        num_directions,
                        batch_size,
                        hidden_size as usize,
                    ))?;
                    values.insert(name.clone(), h_acc);
                }
                if let Some(name) = node.output.get(1) {
                    values.insert(
                        name.clone(),
                        lstm_state.h().reshape((
                            num_directions,
                            batch_size,
                            hidden_size as usize,
                        ))?,
                    );
                }
                if let Some(name) = node.output.get(2) {
                    values.insert(
                        name.clone(),
                        lstm_state.c().reshape((
                            num_directions,
                            batch_size,
                            hidden_size as usize,
                        ))?,
                    );
                }
            },
            "RNN" => {
                // activation_alpha and activation_beta don't apply to (Tanh, Tanh) so ignoring them is okay
                let activations_default = vec!["Tanh".to_string(), "Tanh".to_string()];
                let activations = get_attr_opt_owned::<Vec<String>>(node, "activations")?
                    .unwrap_or(activations_default.clone());
                let clip = get_attr_opt::<f32>(node, "clip")?.copied();
                if clip.is_some() {
                    bail!("RNN does not currently support clip attribute");
                }
                let direction = get_attr_opt(node, "direction")?.unwrap_or("forward");
                if direction != "forward" {
                    bail!("RNN currently only supports direction == \"forward\"");
                }
                let num_directions = if direction == "bidirectional" { 2 } else { 1 };
                let hidden_size: i64 = get_attr(node, "hidden_size").copied()?;

                // The shape format of inputs X, initial_h and outputs Y, Y_h.
                // If 0, the following shapes are expected:
                //    X.shape = [seq_length, batch_size, input_size],
                //    Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                //    initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
                // If 1, the following shapes are expected:
                //    X.shape = [batch_size, seq_length, input_size],
                //    Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                //    initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
                let layout = get_attr_opt(node, "layout")?.copied().unwrap_or(0);
                if layout != 0 {
                    bail!("RNN currently only supports layout == 0");
                }

                // The input sequences packed (and potentially padded) into one 3-D tensor
                // with the shape of `[seq_length, batch_size, input_size]`.
                let x = get(&node.input[0])?;
                // XXX: depends on layout
                let (seq_length, batch_size, _) = x.dims3()?;
                // The weight tensor for the input gate.
                // Concatenation of `Wi` and `WBi` (if bidirectional).
                // The tensor has shape `[num_directions, hidden_size, input_size]`.
                let w = get(&node.input[1])?;
                // The recurrence weight tensor.
                // Concatenation of `Ri` and `RBi` (if bidirectional).
                // This tensor has shape `[num_directions, hidden_size, hidden_size]`.
                let r = get(&node.input[2])?;

                // The bias tensor for input gate.
                // Concatenation of `[Wbi, Rbi]` and `[WBbi, RBbi]` (if bidirectional).
                // This tensor has shape `[num_directions, 2*hidden_size]`.
                // Optional: If not specified - assumed to be 0.
                let b_default: Tensor;
                let b = match get_opt(3) {
                    Some(n) => n?,
                    None => {
                        b_default = Tensor::zeros(
                            (num_directions, 2 * hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &b_default
                    },
                };

                // Optional tensor specifying lengths of the sequences in a batch.
                // If not specified - assumed all sequences in the batch to have length `seq_length`.
                // It has shape `[batch_size]`.
                let seq_lens_default: Tensor;
                let seq_lens = match get_opt(4) {
                    Some(n) => n?,
                    None => {
                        seq_lens_default =
                            Tensor::full(seq_length as i64, (batch_size,), x.device())?;
                        &seq_lens_default
                    },
                };
                let seq_lens_is_default =
                    (seq_lens.to_vec1::<i64>()?.iter()).all(|e| *e as usize == seq_length);
                if !seq_lens_is_default {
                    bail!(
                        "RNN currently does not support variable-length sequences. All sequences must use the full sequence length of {}",
                        seq_length
                    );
                }

                // Optional initial value of the hidden. If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_h_default: Tensor;
                let initial_h = match get_opt(5) {
                    Some(n) => n?,
                    _ => {
                        initial_h_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_h_default
                    },
                };

                fn choose_activation(activation: &str, x: &Tensor) -> Result<Tensor> {
                    match activation {
                        "Tanh" => x.tanh(),
                        _ => bail!("unsupported activation {activation}"),
                    }
                }

                // these all have [num_directions, ...] shapes
                let w = w.get(0)?;
                let r = r.get(0)?;
                let b = b.get(0)?;
                let idx_wb = Tensor::arange(0, hidden_size, x.device())?;
                let idx_rb = Tensor::arange(hidden_size, 2 * hidden_size, x.device())?;
                let wb = b.index_select(&idx_wb, 0)?;
                let rb = b.index_select(&idx_rb, 0)?;
                let mut h_t = initial_h.get(0)?;
                let mut h_list: Vec<Tensor> = vec![];
                for i in 0..seq_length {
                    let xs = x.get(i)?;
                    let h = xs
                        .matmul(&w.t()?)?
                        .add(&h_t.matmul(&r.t()?)?)?
                        .add(&wb.unsqueeze(0)?)?
                        .add(&rb.unsqueeze(0)?)?;
                    let h = choose_activation(&activations[0], &h)?;
                    h_list.push(h.to_owned());
                    h_t = h;
                }
                let h = Tensor::stack(&h_list, 0)?;
                let h =
                    h.reshape((seq_length, num_directions, batch_size, hidden_size as usize))?;
                values.insert(node.output[0].clone(), h);
                values.insert(
                    node.output[1].clone(),
                    h_t.reshape((num_directions, batch_size, hidden_size as usize))?,
                );
            },
            // https://onnx.ai/onnx/operators/onnx__Xor.html
            "Xor" => {
                // Since we don't have a `DType::Bool` yet, this ensures that we are working with `0`(False) & `1`(True)
                let a = get(&node.input[0])?.gt(0_u8)?;
                let b = get(&node.input[1])?.gt(0_u8)?;

                let out = a.broadcast_add(&b)?.eq(1_u8)?;

                values.insert(node.output[0].clone(), out);
            },
            // https://onnx.ai/onnx/operators/onnx__And.html
            "And" => {
                let a = get(&node.input[0])?.gt(0_u8)?;
                let b = get(&node.input[1])?.gt(0_u8)?;

                let out = a.broadcast_mul(&b)?;

                values.insert(node.output[0].clone(), out);
            },
            // https://onnx.ai/onnx/operators/onnx__Or.html
            "Or" => {
                let a = get(&node.input[0])?.gt(0_u8)?;
                let b = get(&node.input[1])?.gt(0_u8)?;

                let out = a.broadcast_add(&b)?.gt(0_u8)?;

                values.insert(node.output[0].clone(), out);
            },
            // https://onnx.ai/onnx/operators/onnx__Sign.html
            "Sign" => {
                let input = get(&node.input[0])?;
                let output = input.sign()?;
                values.insert(node.output[0].clone(), output);
            },
            // https://onnx.ai/onnx/operators/onnx__Selu.html
            "Selu" => {
                let input = get(&node.input[0])?;
                let alpha = get_attr_opt::<f32>(node, "alpha")?
                    .copied()
                    .unwrap_or(1.6732632);
                let gamma = get_attr_opt::<f32>(node, "gamma")?
                    .copied()
                    .unwrap_or(1.050701);
                let out = candle_nn::ops::selu(input, alpha as f32, gamma as f32)?;
                values.insert(node.output[0].clone(), out);
            },

            // https://onnx.ai/onnx/operators/onnx__OneHot.html
            "OneHot" => {
                let indices = get(&node.input[0])?;
                let orig_shape = get(&node.input[0])?.dims().to_vec();
                let depth_tensor = get(&node.input[1])?;
                let values_tensor = get(&node.input[2])?;

                let depth = to_scalar_flexible::<i64>(depth_tensor)? as usize;
                let values_vec = values_tensor.to_vec1::<f32>()?;
                if values_vec.len() != 2 {
                    return Err(candle::Error::Msg(
                        "OneHot: expected 2-element values tensor".to_string(),
                    ));
                }
                let off_value = values_vec[0];
                let on_value = values_vec[1];

                let mut axis = node
                    .attribute
                    .iter()
                    .find(|attr| attr.name == "axis")
                    .map(|attr| attr.i)
                    .unwrap_or(-1);

                let rank = indices.rank();
                if axis < -((rank as i64) + 1) || axis > (rank as i64) {
                    return Err(candle::Error::Msg(format!(
                        "OneHot: invalid axis {axis} for rank {rank}"
                    )));
                }
                if axis < 0 {
                    axis += rank as i64 + 1;
                }

                let indices = indices.flatten_all()?;
                let indices_vec = indices.to_vec1::<i64>()?;
                let mut out = vec![off_value; depth * indices.elem_count()];
                for (i, &index) in indices_vec.iter().enumerate() {
                    let idx = if index < 0 {
                        (index + depth as i64) as usize
                    } else {
                        index as usize
                    };
                    if idx >= depth {
                        continue;
                    }
                    out[i * depth + idx] = on_value;
                }

                let mut target_shape = orig_shape;
                target_shape.push(depth);
                let output = Tensor::from_vec(out, target_shape, indices.device())?;

                let final_output = if axis as usize == output.rank() - 1 {
                    output
                } else {
                    fn move_axis_to(rank: usize, from: usize, to: usize) -> Vec<usize> {
                        let mut dims: Vec<usize> = (0..rank).collect();
                        let axis = dims.remove(from);
                        dims.insert(to, axis);
                        dims
                    }

                    let perm = move_axis_to(output.rank(), output.rank() - 1, axis as usize);
                    output.permute(&*perm)?
                };
                values.insert(node.output[0].clone(), final_output);
            },
            "HardSwish" => {
                let input = get(&node.input[0])?;
                let hard_sigmoid = candle_nn::ops::hard_sigmoid(&input)?;
                let output = input * hard_sigmoid;
                values.insert(node.output[0].clone(), output?);
            },
            // Crane Added 20260802: support rank-3 `[N, C, L]` inputs via
            // `upsample_nearest1d`, in addition to the existing rank-4
            // `[N, C, H, W]` support.
            "Resize" => {
                let input = get(&node.input[0])?;

                let scales = if node.input.len() > 2 && !node.input[2].is_empty() {
                    Some(get(&node.input[2])?)
                } else {
                    None
                };

                let sizes = if node.input.len() > 3 && !node.input[3].is_empty() {
                    Some(get(&node.input[3])?)
                } else {
                    None
                };

                let output_dims = match (scales, sizes) {
                    (Some(_), Some(_)) => {
                        bail!("Scales and sizes cannot both be set for Resize operation")
                    },
                    (Some(scales_tensor), None) => {
                        let scale_values = scales_tensor.to_vec1::<f32>()?;
                        input
                            .dims()
                            .iter()
                            .enumerate()
                            .map(|(i, &d)| (d as f32 * scale_values[i]) as usize)
                            .collect::<Vec<_>>()
                    },
                    (None, Some(sizes_tensor)) => sizes_tensor
                        .to_vec1::<i64>()?
                        .iter()
                        .map(|&d| d as usize)
                        .collect::<Vec<_>>(),
                    (None, None) => bail!("Either scales or sizes should be present"),
                };

                let coordinate_transformation_mode =
                    get_attr_opt::<str>(node, "coordinate_transformation_mode")?
                        .unwrap_or("half_pixel");
                // Interpolation mode: nearest, linear, or cubic.
                let mode = get_attr_opt::<str>(node, "mode")?.unwrap_or("nearest");

                // Crane Added 20260802: mode="linear" half-pixel single-axis
                // interpolation, e.g. resampling a vocoder's [N, C, L]
                // waveform along its length. Only a `scales` input with
                // exactly one non-1.0-scaled axis is supported.
                let output = if mode == "linear" {
                    if coordinate_transformation_mode != "half_pixel" {
                        bail!(
                            "Unsupported coordinate_transformation_mode for linear resize: {}",
                            coordinate_transformation_mode
                        );
                    }
                    let Some(scales_tensor) = scales else {
                        bail!("Resize mode=\"linear\" requires a 'scales' input (not 'sizes')");
                    };
                    let scale_values = scales_tensor.to_vec1::<f32>()?;
                    let mut axis = None;
                    for (i, &s) in scale_values.iter().enumerate() {
                        if (s - 1.0).abs() > 1e-6 {
                            if axis.is_some() {
                                bail!(
                                    "Resize mode=\"linear\" only supports a single scaled axis, got scales {scale_values:?}"
                                );
                            }
                            axis = Some(i);
                        }
                    }
                    let Some(axis) = axis else {
                        bail!("Resize mode=\"linear\": no axis has a non-1.0 scale");
                    };

                    let in_len = input.dim(axis)?;
                    let out_len = output_dims[axis];
                    let scale = scale_values[axis];
                    let max_idx = in_len as i64 - 1;

                    let mut floor_idx = Vec::with_capacity(out_len);
                    let mut ceil_idx = Vec::with_capacity(out_len);
                    let mut fracs = Vec::with_capacity(out_len);
                    for i in 0..out_len {
                        let coord = (i as f32 + 0.5) / scale - 0.5;
                        let f = coord.floor();
                        let f_i64 = f as i64;
                        floor_idx.push(f_i64.clamp(0, max_idx));
                        ceil_idx.push((f_i64 + 1).clamp(0, max_idx));
                        fracs.push(coord - f);
                    }

                    let dev = input.device();
                    let floor_t = Tensor::new(floor_idx, dev)?;
                    let ceil_t = Tensor::new(ceil_idx, dev)?;

                    let left = input.index_select(&floor_t, axis)?;
                    let right = input.index_select(&ceil_t, axis)?;

                    let mut frac_shape = vec![1usize; input.rank()];
                    frac_shape[axis] = out_len;
                    let frac_t = Tensor::new(fracs, dev)?.reshape(frac_shape)?;

                    left.broadcast_add(&right.broadcast_sub(&left)?.broadcast_mul(&frac_t)?)?
                } else {
                    // How to determine the "nearest" pixel in nearest interpolation mode.
                    let nearest_mode =
                        get_attr_opt::<str>(node, "nearest_mode")?.unwrap_or("round_prefer_floor");

                    if mode != "nearest" {
                        bail!("Unsupported resize mode: {}", mode);
                    }

                    if nearest_mode != "floor" {
                        bail!("Unsupported nearest_mode for resize: {}", nearest_mode);
                    }

                    if coordinate_transformation_mode != "asymmetric" {
                        bail!(
                            "Unsupported coordinate_transformation_mode for resize: {}",
                            coordinate_transformation_mode
                        );
                    }

                    match input.rank() {
                        3 => input.upsample_nearest1d(output_dims[2])?,
                        4 => input.upsample_nearest2d(output_dims[2], output_dims[3])?,
                        rank => bail!("Unsupported rank for nearest resize: {rank}"),
                    }
                };

                values.insert(node.output[0].clone(), output);
            },
            "Trilu" => {
                let input = get(&node.input[0])?;

                // Get the diagonal offset 'k' from the second input if provided
                let k = if node.input.len() > 1 && !node.input[1].is_empty() {
                    to_vec0_flexible::<i64>(get(&node.input[1])?)?
                } else {
                    0
                };

                // Get the 'upper' attribute
                let upper = get_attr_opt::<i64>(node, "upper")?.copied().unwrap_or(1);

                // For batched inputs, we need to handle each matrix separately
                let dims = input.dims();
                if dims.len() < 2 {
                    bail!("Trilu expects input with at least 2 dimensions: {:?}", dims);
                }

                // Get the last two dimensions which represent the matrix
                let n = dims[dims.len() - 2];
                let m = dims[dims.len() - 1];
                let max_dim = std::cmp::max(n, m);

                // Handle the diagonal offset k
                let mask = if k != 0 {
                    let mut data = vec![0u32; n * m];
                    for i in 0..n {
                        for j in 0..m {
                            if (upper != 0 && (j as i64) >= (i as i64) + k)
                                || (upper == 0 && (j as i64) <= (i as i64) + k)
                            {
                                data[i * m + j] = 1u32;
                            }
                        }
                    }
                    Tensor::from_vec(data, (n, m), input.device())?.to_dtype(input.dtype())?
                } else if upper == 0 {
                    Tensor::tril2(max_dim, input.dtype(), input.device())?
                } else {
                    Tensor::triu2(max_dim, input.dtype(), input.device())?
                };

                let final_mask = if n != m {
                    mask.narrow(0, 0, n)?.narrow(1, 0, m)?
                } else {
                    mask
                };

                let output = (input * &final_mask)?;

                values.insert(node.output[0].clone(), output);
            },
            "ScatterElements" => {
                let data = get(&node.input[0])?;
                let indices = get(&node.input[1])?;
                let updates = get(&node.input[2])?;
                let output = ops::scatter_elements::scatter_elements(node, data, indices, updates)?;
                values.insert(node.output[0].clone(), output);
            },
            "ScatterND" => {
                let data = get(&node.input[0])?;

                let indices = get(&node.input[1])?;
                let indices = indices.to_dtype(DType::I64)?;

                let updates = get(&node.input[2])?;

                let reduction = get_attr_opt::<str>(node, "reduction")?.unwrap_or("none");

                let indices_shape = indices.dims();
                let data_shape = data.dims();
                let _updates_shape = updates.dims();

                // Last dimension of indices represents the depth of indexing
                let k = indices_shape.last().unwrap().clone();

                if k > data.rank() {
                    bail!("ScatterND expects k (indices.shape[-1]) to be at most the rank of data");
                }

                let num_updates = indices_shape[..indices_shape.len() - 1]
                    .iter()
                    .product::<usize>();

                let flat_indices = if indices.rank() == 1 && k == 1 {
                    indices.unsqueeze(0)?
                } else {
                    indices.reshape((num_updates, k))?
                };

                // Calculate the shape of each update element
                let update_element_shape = if k < data_shape.len() {
                    data_shape[k..].to_vec()
                } else {
                    vec![]
                };

                // Expected shape for updates based on indices and target tensor
                let expected_updates_shape = {
                    let mut shape = indices_shape[..indices_shape.len() - 1].to_vec();
                    shape.extend(&update_element_shape);
                    shape
                };

                // Validate or reshape updates to expected shape
                let updates = if updates.dims() != expected_updates_shape {
                    if updates.rank() == 0 {
                        // Handle scalar updates
                        let mut target_shape = vec![num_updates];
                        target_shape.extend(&update_element_shape);
                        updates.broadcast_as(target_shape)?
                    } else {
                        // Try to broadcast or reshape updates to expected shape
                        let flat_shape =
                            vec![num_updates, update_element_shape.iter().product::<usize>()];
                        let flattened = updates.reshape(flat_shape)?;
                        flattened.reshape(expected_updates_shape)?
                    }
                } else {
                    updates.clone()
                };

                let mut output = data.clone();

                // convert indices to flat indices
                let mut flat_output = output.flatten_all()?;
                let flat_updates = if update_element_shape.is_empty() {
                    updates.reshape(num_updates)?
                } else {
                    let product = update_element_shape.iter().product::<usize>();
                    updates.reshape((num_updates, product))?
                };

                // Calculate strides for the output tensor
                let mut strides: Vec<usize> = vec![1];
                for i in (0..data_shape.len() - 1).rev() {
                    strides.push(strides.last().unwrap() * data_shape[i + 1]);
                }
                strides.reverse();

                // Process each update
                for i in 0..num_updates {
                    let index_slice = flat_indices.narrow(0, i, 1)?;
                    let indices_vec = index_slice.squeeze(0)?.to_vec1::<i64>()?;

                    // Convert multi-dimensional indices to flat index
                    let mut flat_idx: usize = 0;
                    for (dim, &idx) in indices_vec.iter().enumerate() {
                        let dim_size = data_shape[dim] as i64;
                        let norm_idx = if idx < 0 { dim_size + idx } else { idx };

                        if norm_idx < 0 || norm_idx >= dim_size {
                            bail!(
                                "Index {} out of bounds for dimension {} with size {}",
                                idx,
                                dim,
                                dim_size
                            );
                        }

                        flat_idx += (norm_idx as usize) * strides[dim];
                    }

                    // Extract current update. `flat_updates` is `(num_updates,)`
                    // when each update is a scalar, or `(num_updates, product)`
                    // when each update is itself a `product`-sized slice; either
                    // way, squeezing dim 0 drops the narrowed size-1 leading
                    // dim so `update_slice`'s rank matches `flat_output`'s
                    // (rank 0 for a scalar update, rank 1 for a slice update).
                    // Crane Changed 20260827: this squeeze was missing for the
                    // slice-update case, leaving `update_slice` at rank 2
                    // (`(1, product)`) and making every `slice_scatter` below
                    // fail with a rank mismatch against the rank-1
                    // `flat_output`.
                    let update_slice = flat_updates.narrow(0, i, 1)?.squeeze(0)?;

                    match reduction {
                        "add" => {
                            if update_element_shape.is_empty() {
                                let existing = flat_output.narrow(0, flat_idx, 1)?;
                                let new_value = existing.add(&update_slice.unsqueeze(0)?)?;
                                flat_output = flat_output.slice_scatter(&new_value, 0, flat_idx)?;
                            } else {
                                let slice_size = update_element_shape.iter().product::<usize>();
                                let existing = flat_output.narrow(0, flat_idx, slice_size)?;
                                let new_value = existing.add(&update_slice)?;
                                flat_output = flat_output.slice_scatter(&new_value, 0, flat_idx)?;
                            }
                        },
                        "none" | _ => {
                            if update_element_shape.is_empty() {
                                flat_output = flat_output.slice_scatter(
                                    &update_slice.unsqueeze(0)?,
                                    0,
                                    flat_idx,
                                )?;
                            } else {
                                flat_output =
                                    flat_output.slice_scatter(&update_slice, 0, flat_idx)?;
                            }
                        },
                    }
                }

                // Reshape flat output back to original shape
                output = flat_output.reshape(data_shape.to_vec())?;

                values.insert(node.output[0].clone(), output);
            },
            op_type => bail!("unsupported op_type {op_type} for op {node:?}"),
        }

        // Crane Added 20260807: detach every output this node just produced
        // from its computation graph. `simple_eval_` is a forward-only
        // evaluator that never calls `Tensor::backward()`, but candle's
        // `Tensor::track_op()` is `is_variable || op.is_some()` and
        // `BackpropOp::new1/2/3` check it transitively on their operands —
        // so once any tensor anywhere upstream is a `Var` (the "LSTM" op
        // above wraps its weights in one, since `candle_nn::rnn::lstm()`
        // requires a `VarBuilder`), the taint propagates forward forever:
        // every tensor computed from it, and everything computed from
        // those, keeps retaining full backward-provenance for the rest of
        // the graph. On a model whose LSTM feeds into a large downstream
        // network (e.g. a TTS vocoder), that produces a real, deep
        // computation graph, and dropping the final output recurses through
        // the whole thing in one native call-stack frame per node —
        // confirmed via gdb backtrace to overflow the stack on Kokoro
        // synthesis. Detaching per node keeps every tensor's `Drop` O(1)
        // regardless of what fed into it, and as a side effect lets
        // `release_names_if_done` below actually free memory — previously
        // an evicted name's storage could still be kept alive by a later
        // tensor's retained op chain even after its `values` entry was
        // removed.
        for output_name in node.output.iter().filter(|name| !name.is_empty()) {
            if let Some(value) = values.get_mut(output_name.as_str()) {
                *value = value.detach();
            }
        }

        release_names_if_done(
            node.input
                .iter()
                .filter(|input| !input.is_empty())
                .map(String::as_str),
            &mut remaining_uses,
            &graph_outputs,
            &inherited_values,
            values,
        );
    }
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => bail!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}

// Crane Added 20260731: ONNX SAME_UPPER padding for PaddleOCR MaxPool.
fn same_upper_pool2d_pads(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    strides: Option<&[i64]>,
    dilations: Option<&[i64]>,
) -> Result<(usize, usize, usize, usize)> {
    let stride_h = strides.and_then(|v| v.first()).copied().unwrap_or(1);
    let stride_w = strides.and_then(|v| v.get(1)).copied().unwrap_or(stride_h);
    let dilation_h = dilations.and_then(|v| v.first()).copied().unwrap_or(1);
    let dilation_w = dilations
        .and_then(|v| v.get(1))
        .copied()
        .unwrap_or(dilation_h);
    if stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 {
        bail!("MaxPool SAME_UPPER requires positive strides and dilations")
    }
    let one_axis = |input_size: usize, kernel: usize, stride: i64, dilation: i64| {
        let output_size = (input_size as i64 + stride - 1) / stride;
        let total = ((output_size - 1) * stride + (kernel as i64 - 1) * dilation + 1
            - input_size as i64)
            .max(0) as usize;
        (total / 2, total - total / 2)
    };
    let (h0, h1) = one_axis(input.dim(2)?, kernel_h, stride_h, dilation_h);
    let (w0, w1) = one_axis(input.dim(3)?, kernel_w, stride_w, dilation_w);
    Ok((h0, h1, w0, w1))
}

fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let (longest, shortest) = if shape_a.len() > shape_b.len() {
        (shape_a, shape_b)
    } else {
        (shape_b, shape_a)
    };
    let diff = longest.len() - shortest.len();
    let mut target_shape = longest[0..diff].to_vec();
    for (dim1, dim2) in longest[diff..].iter().zip(shortest.iter()) {
        if *dim1 == *dim2 || *dim2 == 1 || *dim1 == 1 {
            target_shape.push(usize::max(*dim1, *dim2));
        } else {
            bail!(
                "Expand: incompatible shapes for broadcast, {:?} and {:?}",
                shape_a,
                shape_b
            );
        }
    }
    Ok(target_shape)
}

fn broadcast_shape_from_many(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape_out = shapes[0].to_vec();
    for shape in shapes[1..].iter() {
        shape_out = broadcast_shape(&shape_out, shape)?;
    }
    Ok(shape_out)
}

/// Extract scalar from tensors that may be wrapped in extra dimensions.
/// Some ONNX exports use shape [1] or [1,1] where scalars are expected.
/// Only accepts single-element tensors; multi-element tensors still fail.
pub(crate) fn to_scalar_flexible<T: candle::WithDType>(t: &Tensor) -> Result<T> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_scalar::<T>()
    } else {
        t.to_scalar::<T>()
    }
}

/// Same as to_scalar_flexible but returns via to_vec0 for types that need it.
fn to_vec0_flexible<T: candle::WithDType>(t: &Tensor) -> Result<T> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_vec0::<T>()
    } else {
        t.to_vec0::<T>()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Result};

    use super::{Value, simple_eval, simple_eval_};
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};

    #[test]
    fn round_rounds_half_away_from_zero() -> Result<()> {
        // ONNX Round: nearest integer per element; candle's round() (and
        // this evaluator) breaks exact .5 ties away from zero rather than
        // to even, matching the tie-breaking already used elsewhere in
        // Crane's ONNX graph rewrites.
        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![ValueInfoProto {
                    name: "x".to_string(),
                    ..Default::default()
                }],
                node: vec![NodeProto {
                    op_type: "Round".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "y".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let x = Value::new(&[0.5f32, 2.5, -0.5, -2.5, 1.4, 1.6], &Device::Cpu)?;
        let outputs = simple_eval(&model, [("x".to_string(), x)].into())?;

        assert_eq!(
            outputs["y"].to_vec1::<f32>()?,
            vec![1.0, 3.0, -1.0, -3.0, 1.0, 2.0]
        );
        Ok(())
    }

    fn resize_node(mode: &str, coordinate_transformation_mode: &str) -> NodeProto {
        NodeProto {
            op_type: "Resize".to_string(),
            input: vec!["x".to_string(), String::new(), "scales".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![
                AttributeProto {
                    name: "mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: mode.as_bytes().to_vec(),
                    ..Default::default()
                },
                AttributeProto {
                    name: "coordinate_transformation_mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: coordinate_transformation_mode.as_bytes().to_vec(),
                    ..Default::default()
                },
                AttributeProto {
                    name: "nearest_mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: b"floor".to_vec(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn resize_model(node: NodeProto) -> ModelProto {
        ModelProto {
            graph: Some(GraphProto {
                input: vec![ValueInfoProto {
                    name: "x".to_string(),
                    ..Default::default()
                }],
                node: vec![node],
                output: vec![ValueInfoProto {
                    name: "y".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn resize_linear_half_pixel_upsamples_single_axis() -> Result<()> {
        // Verifies mode="linear" scales the one non-1.0-scaled axis using
        // half-pixel coordinates: coord(i) = (i + 0.5) / scale - 0.5, so
        // i=0 -> -0.25 (clamped to index 0, value 0.0) and i=1 -> 0.25 ->
        // lerp(0, 10, 0.25) = 2.5.
        let model = resize_model(resize_node("linear", "half_pixel"));
        let x = Value::new(&[[0f32, 10., 20., 30.]], &Device::Cpu)?;
        let scales = Value::new(&[1.0f32, 2.0], &Device::Cpu)?;

        let outputs = simple_eval(
            &model,
            [("x".to_string(), x), ("scales".to_string(), scales)].into(),
        )?;

        let got = outputs["y"].to_vec2::<f32>()?[0].clone();
        let expected = [0.0f32, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {got:?}, expected {expected:?}");
        }
        Ok(())
    }

    #[test]
    fn resize_linear_half_pixel_downsamples_single_axis() -> Result<()> {
        // Verifies mode="linear" also handles a scale factor below 1.0
        // (downsampling), producing floor(input_len * scale) outputs.
        let model = resize_model(resize_node("linear", "half_pixel"));
        let x = Value::new(&[[0f32, 10., 20., 30.]], &Device::Cpu)?;
        let scales = Value::new(&[1.0f32, 0.5], &Device::Cpu)?;

        let outputs = simple_eval(
            &model,
            [("x".to_string(), x), ("scales".to_string(), scales)].into(),
        )?;

        assert_eq!(outputs["y"].dims(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn resize_nearest_asymmetric_supports_rank3() -> Result<()> {
        // Verifies mode="nearest" now dispatches rank-3 [N, C, L] inputs to
        // upsample_nearest1d instead of hard-bailing, alongside the
        // existing rank-4 upsample_nearest2d path.
        let model = resize_model(resize_node("nearest", "asymmetric"));
        let x = Value::new(&[[[0f32, 10.]]], &Device::Cpu)?;
        let scales = Value::new(&[1.0f32, 1.0, 2.0], &Device::Cpu)?;

        let outputs = simple_eval(
            &model,
            [("x".to_string(), x), ("scales".to_string(), scales)].into(),
        )?;

        assert_eq!(outputs["y"].dims(), &[1, 1, 4]);
        Ok(())
    }
}
