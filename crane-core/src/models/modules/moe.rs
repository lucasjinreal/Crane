// SPDX-License-Identifier: MIT

use crate::device::{format_budget, query_gpu_memory};
use crate::models::hunyuan_dense::modeling::Gguf;
use crate::ops::linear::LinearLayer;
use crate::ops::prof::{self, Span};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor, ggml_file::qtensor_from_ggml};
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder, linear_no_bias};
use ribo::utils::log;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Configuration for Mixture-of-Experts feed-forward layers.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts per `MoE` layer.
    pub num_experts: usize,
    /// Number of experts activated per token (top-K).
    pub num_experts_per_tok: usize,
    /// Hidden dimension of each expert's feed-forward network.
    pub moe_intermediate_size: usize,
    /// Whether to renormalize the top-K routing weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Every Nth layer is `MoE`; the rest stay dense MLP.
    /// `None` means every layer is `MoE` (matches HF's default of `1`).
    /// Only used by the safetensors path; the GGUF path detects
    /// MoE-vs-dense per layer by tensor presence.
    pub decoder_sparse_step: Option<usize>,
}

/// A single Mixture-of-Experts feed-forward expert.
///
/// A `SiLU`-gated MLP identical in shape to a dense Qwen3 `Mlp`, but sized to
/// `moe_intermediate_size` rather than the model's dense `intermediate_size`.
#[allow(clippy::struct_field_names)]
pub struct MoeExpert {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl MoeExpert {
    /// Create an expert from a safetensors checkpoint.
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension (input/output size)
    /// * `intermediate_size` - Expert feed-forward hidden dimension
    /// * `vb` - `VarBuilder` scoped to this expert
    ///
    /// # Errors
    /// Returns an error if any of the expert's weight tensors are missing or
    /// have the wrong shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: LinearLayer::Standard(gate_proj),
            up_proj: LinearLayer::Standard(up_proj),
            down_proj: LinearLayer::Standard(down_proj),
        })
    }

    /// Create an expert from a GGUF checkpoint using the per-expert tensor
    /// layout (`blk.{layer_idx}.ffn_gate.{expert_idx}.weight`, etc.).
    ///
    /// # Arguments
    /// * `gg` - GGUF reader
    /// * `layer_idx` - Decoder layer index
    /// * `expert_idx` - Expert index within the layer
    /// * `device` - Device to load the expert's weights onto (supports
    ///   expert offloading to a device other than the rest of the model)
    ///
    /// # Errors
    /// Returns an error if the expert's tensors are missing from the GGUF file.
    pub fn new_from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        expert_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_proj = gg.linear_on(&format!("{prefix}.ffn_gate.{expert_idx}.weight"), device)?;
        let up_proj = gg.linear_on(&format!("{prefix}.ffn_up.{expert_idx}.weight"), device)?;
        let down_proj = gg.linear_on(&format!("{prefix}.ffn_down.{expert_idx}.weight"), device)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Build an expert directly from already-loaded projections.
    ///
    /// Used by [`SparseMoeBlock::new_from_gguf`] for the packed GGUF expert
    /// layout, where the three projections come from byte-slicing a shared
    /// 3D tensor rather than loading per-expert tensors.
    fn from_layers(gate_proj: LinearLayer, up_proj: LinearLayer, down_proj: LinearLayer) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Moves all three projections to `device`, in `dtype`. See
    /// [`LinearLayer::to_device`] for the `Quantized` dequantization
    /// tradeoff this implies and why `dtype` must match the model's
    /// compute dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if any projection's device transfer fails.
    pub fn to_device(&self, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            gate_proj: self.gate_proj.to_device(device, dtype)?,
            up_proj: self.up_proj.to_device(device, dtype)?,
            down_proj: self.down_proj.to_device(device, dtype)?,
        })
    }
}

impl Module for MoeExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let gate = Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Mixture-of-Experts routing + dispatch block.
///
/// Routes each token to its top-K experts (by router-gate logits), runs only
/// those experts, and combines their outputs weighted by routing probability.
/// Experts may live on a different device than the router and input
/// (`expert_device`), so that expert weights can be offloaded (e.g. to CPU)
/// while the rest of the model stays on GPU.
pub struct SparseMoeBlock {
    gate: LinearLayer,
    experts: Vec<MoeExpert>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    expert_device: Device,
}

impl SparseMoeBlock {
    /// Create a `MoE` block from a safetensors checkpoint.
    ///
    /// # Arguments
    /// * `config` - `MoE` layer configuration
    /// * `hidden_size` - Model hidden dimension
    /// * `vb` - `VarBuilder` scoped to this block (holds `gate` and `experts.{i}`)
    /// * `expert_device` - Device to place expert weights on
    ///
    /// # Errors
    /// Returns an error if the router or any expert's weight tensors are
    /// missing or have the wrong shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        config: &MoeConfig,
        hidden_size: usize,
        vb: VarBuilder,
        expert_device: &Device,
    ) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, config.num_experts, vb.pp("gate"))?;
        // Router logits are always computed in F32 (see `forward`); re-wrap the
        // weight in F32 here so a BF16/F16 checkpoint doesn't hit a dtype
        // mismatch on the first router matmul.
        let gate_weight = gate.weight().to_dtype(DType::F32)?;
        let gate = Linear::new(gate_weight, None);
        let experts = (0..config.num_experts)
            .map(|i| {
                MoeExpert::new(
                    hidden_size,
                    config.moe_intermediate_size,
                    vb.pp(format!("experts.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate: LinearLayer::Standard(gate),
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_device: expert_device.clone(),
        })
    }

    /// Create a `MoE` block from a GGUF checkpoint.
    ///
    /// Supports both packed (`blk.{i}.ffn_gate_exps.weight`, all experts
    /// stacked in one 3D tensor) and per-expert
    /// (`blk.{i}.ffn_gate.{j}.weight`) tensor layouts, auto-detected by
    /// tensor presence.
    ///
    /// # Arguments
    /// * `config` - `MoE` layer configuration
    /// * `gg` - GGUF reader
    /// * `layer_idx` - Decoder layer index
    /// * `expert_device` - Device to place expert weights on (supports
    ///   expert offloading to a device other than the rest of the model)
    ///
    /// # Errors
    /// Returns an error if the router or any expert's tensors are missing
    /// from the GGUF file.
    pub fn new_from_gguf<R: Read + Seek>(
        config: &MoeConfig,
        gg: &mut Gguf<R>,
        layer_idx: usize,
        expert_device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_weight = gg
            .dequant_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
            .to_dtype(DType::F32)?;
        let gate = LinearLayer::Standard(Linear::new(gate_weight, None));

        let packed_name = format!("{prefix}.ffn_gate_exps.weight");
        let experts = if gg.contains_tensor(&packed_name) {
            Self::load_packed_experts(gg, &prefix, config.num_experts, expert_device)?
        } else {
            (0..config.num_experts)
                .map(|expert_idx| {
                    MoeExpert::new_from_gguf(gg, layer_idx, expert_idx, expert_device)
                })
                .collect::<Result<Vec<_>>>()?
        };

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_device: expert_device.clone(),
        })
    }

    /// Load the packed (Unsloth-style) `_exps` expert tensor layout: each
    /// projection is a single 3D tensor `[num_experts, out, in]` covering all
    /// experts, loaded onto `expert_device` and byte-sliced per expert
    /// without dequantizing.
    ///
    /// Expert boundaries align with quantization block boundaries for every
    /// standard GGML block size (a Qwen3-Coder-30B-A3B expert is
    /// 768x2048 = 1,572,864 elements, divisible by block sizes 1/32/256), so
    /// [`slice_packed_qtensor`] can carve each expert's raw bytes out of the
    /// packed tensor and hand them to `qtensor_from_ggml` directly. This
    /// keeps packed experts quantized on load, matching the per-expert
    /// layout (already quantized via [`MoeExpert::new_from_gguf`]).
    ///
    /// Each packed tensor's bytes are fetched via `QTensor::data()` once,
    /// up front, rather than once per expert inside the loop: on CUDA,
    /// `data()` does a full device-to-host memcpy of the whole buffer, so a
    /// per-expert call would cost `num_experts` redundant full-tensor
    /// copies instead of one.
    fn load_packed_experts<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        expert_device: &Device,
    ) -> Result<Vec<MoeExpert>> {
        let gate_packed = gg.tensor_on(&format!("{prefix}.ffn_gate_exps.weight"), expert_device)?;
        let up_packed = gg.tensor_on(&format!("{prefix}.ffn_up_exps.weight"), expert_device)?;
        let down_packed = gg.tensor_on(&format!("{prefix}.ffn_down_exps.weight"), expert_device)?;

        // Shape is [num_experts, out_dim, in_dim]; gate/up share one shape,
        // down has its own (in_dim/out_dim swapped relative to gate/up).
        let gate_dims = gate_packed.shape().dims();
        if gate_dims[0] != num_experts {
            candle_core::bail!(
                "{prefix}: config num_experts={num_experts} does not match packed tensor's expert dim {}",
                gate_dims[0]
            );
        }
        let (gate_up_out, gate_up_in) = (gate_dims[1], gate_dims[2]);
        let down_dims = down_packed.shape().dims();
        let (down_out, down_in) = (down_dims[1], down_dims[2]);

        let gate_dtype = gate_packed.dtype();
        let up_dtype = up_packed.dtype();
        let down_dtype = down_packed.dtype();
        let gate_raw = gate_packed.data()?;
        let up_raw = up_packed.data()?;
        let down_raw = down_packed.data()?;

        (0..num_experts)
            .map(|i| {
                let gate_qt = slice_packed_qtensor(
                    &gate_raw,
                    gate_dtype,
                    i,
                    gate_up_out,
                    gate_up_in,
                    expert_device,
                )?;
                let up_qt = slice_packed_qtensor(
                    &up_raw,
                    up_dtype,
                    i,
                    gate_up_out,
                    gate_up_in,
                    expert_device,
                )?;
                let down_qt = slice_packed_qtensor(
                    &down_raw,
                    down_dtype,
                    i,
                    down_out,
                    down_in,
                    expert_device,
                )?;
                Ok(MoeExpert::from_layers(
                    LinearLayer::Quantized(QMatMul::from_arc(Arc::new(gate_qt))?),
                    LinearLayer::Quantized(QMatMul::from_arc(Arc::new(up_qt))?),
                    LinearLayer::Quantized(QMatMul::from_arc(Arc::new(down_qt))?),
                ))
            })
            .collect()
    }

    /// Moves every expert's weights to `device`, in `dtype`, updating
    /// `expert_device`. No-op if experts are already on `device`. The
    /// router gate always stays on the main device and is never moved.
    ///
    /// Used to promote a CPU-placed layer to GPU once real post-warmup VRAM
    /// headroom is known (see `Qwen3Model::from_gguf`'s promotion pass) —
    /// the inverse of the offloading `expert_device` already supports at
    /// load time. `dtype` must be the model's compute dtype: see
    /// [`LinearLayer::to_device`] for why a mismatched dtype breaks the
    /// next forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error if any expert's device transfer fails.
    pub fn promote_experts_to(&mut self, device: &Device, dtype: DType) -> Result<()> {
        if self.expert_device.location() == device.location() {
            return Ok(());
        }
        match batched_promote(&self.experts, device, dtype) {
            Ok(Some(moved)) => {
                self.experts = moved;
                self.expert_device = device.clone();
                return Ok(());
            },
            Ok(None) => {},
            Err(err) => {
                if let Some((free, mem_total)) = query_gpu_memory(device) {
                    log::warn!(
                        "batched promotion to {:?} failed, falling back to per-expert: free={}, total={}, err={err}",
                        device.location(),
                        format_budget(free),
                        format_budget(mem_total),
                    );
                }
            },
        }
        // Collect into a new Vec first and only commit on full success: if
        // e.g. expert 50 of 128 fails (an out-of-memory device transfer),
        // an in-place per-expert overwrite would leave `self.experts` with
        // a mix of old- and new-device tensors while `self.expert_device`
        // still names the old device — `forward()`'s `same_device` check
        // would then be wrong for the already-moved experts, causing a
        // tensor-device mismatch on the next real forward pass. Partial
        // failure here must leave the block exactly as it was.
        //
        // Uses an explicit indexed loop (not `.map().collect()`) so a
        // failure can be pinpointed to the exact expert index and paired
        // with a live VRAM query — a whole-layer promotion is ~1GB but made
        // of ~3*num_experts small per-projection transfers, and a plain
        // per-layer before/after query can't tell a gradual drain from a
        // single-transfer cliff.
        let total = self.experts.len();
        let mut moved: Vec<MoeExpert> = Vec::with_capacity(total);
        for (idx, expert) in self.experts.iter().enumerate() {
            let promoted = match expert.to_device(device, dtype) {
                Ok(promoted) => promoted,
                Err(err) => {
                    if let Some((free, mem_total)) = query_gpu_memory(device) {
                        log::warn!(
                            "expert {idx}/{total}: to_device failed on {:?}: free={}, total={}",
                            device.location(),
                            format_budget(free),
                            format_budget(mem_total),
                        );
                    }
                    return Err(err);
                },
            };
            moved.push(promoted);
            if (idx % 16 == 0 || idx + 1 == total)
                && let Some((free, mem_total)) = query_gpu_memory(device)
            {
                log::debug!(
                    "expert {}/{total} moved to {:?}: free={}, total={}",
                    idx + 1,
                    device.location(),
                    format_budget(free),
                    format_budget(mem_total),
                );
            }
        }
        self.experts = moved;
        self.expert_device = device.clone();
        Ok(())
    }
}

/// Byte-slice one expert's 2D weight out of a packed `[num_experts, out, in]`
/// tensor's raw bytes, without dequantizing.
///
/// Expert boundaries are quantization-block-aligned (verified at the call
/// site's doc comment), so `expert_idx`'s raw bytes can be carved directly
/// out of `raw` and handed to `qtensor_from_ggml` to build an independent,
/// still-quantized `QTensor` for that expert alone. Takes `raw` (rather than
/// the source `QTensor`) so callers can fetch the packed tensor's bytes once
/// and reuse them across all experts — see `load_packed_experts`'s doc
/// comment on why fetching it per-expert would be expensive on CUDA.
///
/// # Errors
///
/// Returns an error if `expert_rows * expert_cols` isn't a whole number of
/// quantization blocks, or if the computed byte range falls outside `raw`.
fn slice_packed_qtensor(
    raw: &[u8],
    ggml_dtype: GgmlDType,
    expert_idx: usize,
    expert_rows: usize,
    expert_cols: usize,
    device: &Device,
) -> Result<QTensor> {
    let expert_elems = expert_rows * expert_cols;
    let block_size = ggml_dtype.block_size();
    if !expert_elems.is_multiple_of(block_size) {
        candle_core::bail!(
            "expert element count {expert_elems} not divisible by block size {block_size}"
        );
    }
    let bytes_per_expert = expert_elems / block_size * ggml_dtype.type_size();
    let start = expert_idx * bytes_per_expert;
    let end = start + bytes_per_expert;
    if end > raw.len() {
        candle_core::bail!(
            "expert {expert_idx} byte range {start}..{end} exceeds packed tensor ({} bytes)",
            raw.len(),
        );
    }
    qtensor_from_ggml(
        ggml_dtype,
        &raw[start..end],
        vec![expert_rows, expert_cols],
        device,
    )
}

/// Copies a [`LinearLayer`] to `device`, preserving quantization.
///
/// Unlike [`LinearLayer::to_device`] (which dequantizes `Quantized`
/// variants to promote an expert permanently), this rebuilds a
/// `QMatMul::QTensor` from its raw quantized bytes on `device` using the
/// same `qtensor_from_ggml` reconstruction technique as
/// [`slice_packed_qtensor`], so a routed expert's temporary copy still runs
/// through the target device's quantized matmul kernels instead of a dense
/// GEMM. Used by [`SparseMoeBlock::forward`] to copy only the routed
/// experts' weights to the input device for each call, rather than moving
/// the hidden state to `expert_device`.
///
/// Unlike [`LinearLayer::to_device`], this does not cast to a target dtype:
/// the `Quantized` path's `forward` handles F32 conversion internally, and
/// the `Standard` path assumes weights are already in the model's compute
/// dtype (true for all current callers, since experts are loaded in the
/// compute dtype at model-load time).
///
/// Note: `QTensor::data()` on CUDA storage copies bytes to the host, so for
/// GPU-to-GPU offload this results in a D2H2D round-trip through host
/// memory. Acceptable for the primary use case (CPU-offloaded experts
/// copied to GPU).
///
/// # Errors
///
/// Returns an error if the device transfer or `QTensor` reconstruction
/// fails.
fn copy_linear_to_device(layer: &LinearLayer, device: &Device) -> Result<LinearLayer> {
    match layer {
        LinearLayer::Standard(l) => {
            let weight = l.weight().to_device(device)?;
            let bias = l.bias().map(|b| b.to_device(device)).transpose()?;
            Ok(LinearLayer::Standard(Linear::new(weight, bias)))
        },
        LinearLayer::Quantized(QMatMul::QTensor(qt)) => {
            let raw = qt.data()?;
            let new_qt = qtensor_from_ggml(qt.dtype(), &raw, qt.shape().dims().to_vec(), device)?;
            Ok(LinearLayer::Quantized(QMatMul::from_arc(Arc::new(new_qt))?))
        },
        LinearLayer::Quantized(QMatMul::Tensor(t)) => Ok(LinearLayer::Quantized(QMatMul::Tensor(
            t.to_device(device)?,
        ))),
        LinearLayer::Quantized(QMatMul::TensorF16(t)) => Ok(LinearLayer::Quantized(
            QMatMul::TensorF16(t.to_device(device)?),
        )),
    }
}

/// Copies all three of an [`MoeExpert`]'s projections to `device` via
/// [`copy_linear_to_device`], for use as a short-lived GPU-resident copy
/// of a CPU-offloaded expert (see [`SparseMoeBlock::forward`]).
///
/// # Errors
///
/// Returns an error if any projection's device transfer fails.
fn copy_expert_to_device(expert: &MoeExpert, device: &Device) -> Result<MoeExpert> {
    Ok(MoeExpert::from_layers(
        copy_linear_to_device(&expert.gate_proj, device)?,
        copy_linear_to_device(&expert.up_proj, device)?,
        copy_linear_to_device(&expert.down_proj, device)?,
    ))
}

/// Narrows one expert's view out of an already-promoted, single-owner
/// `[num_experts, out, in]` tensor, without forcing a copy.
///
/// The source here (`batched_promote`'s freshly-transferred
/// `gate_all`/`up_all`/`down_all`) is never handed to another `to_device`
/// call, so sharing storage across all `num_experts` views back into one
/// buffer is safe and avoids re-introducing the exact per-allocation
/// overhead `batched_promote` exists to eliminate.
fn narrow_packed_expert(packed: &Tensor, expert_idx: usize) -> Result<Tensor> {
    packed.narrow(0, expert_idx, 1)?.squeeze(0)
}

/// Stacks `experts`' projection selected by `proj` into one
/// `[num_experts, out, in]` tensor and transfers it to `device` in `dtype`
/// with a single allocation.
///
/// Returns `Ok(None)` if any expert's projection isn't a bias-free
/// `Standard` layer (the per-expert GGUF layout keeps `Quantized` layers,
/// which can't be stacked this way). `Tensor::stack` builds the stacked
/// tensor on the experts' current device before `to_device` transfers it,
/// so this transiently doubles host-side memory for the stacked
/// projection's size when promoting from CPU.
fn stack_projection(
    experts: &[MoeExpert],
    device: &Device,
    dtype: DType,
    proj: impl Fn(&MoeExpert) -> &LinearLayer,
) -> Result<Option<Tensor>> {
    let mut weights = Vec::with_capacity(experts.len());
    for expert in experts {
        match proj(expert) {
            LinearLayer::Standard(l) if l.bias().is_none() => weights.push(l.weight()),
            _ => return Ok(None),
        }
    }
    Ok(Some(
        Tensor::stack(&weights, 0)?
            .to_device(device)?
            .to_dtype(dtype)?,
    ))
}

/// Attempts single-shot batched promotion of `experts` to `device` in
/// `dtype`: all experts' same projection stacked into one tensor and
/// transferred with one device allocation per projection (3 total)
/// instead of one per expert projection (`3 * experts.len()`).
///
/// Exists because transferring `3 * num_experts` small tensors
/// individually (`SparseMoeBlock::promote_experts_to`'s per-expert
/// fallback loop below) measured a fixed ~1MB overhead per allocation on
/// ROCm — ~0.4G of pure overhead on top of the tensors' actual combined
/// size for a 128-expert layer. Returns `Ok(None)` (fall back to the
/// per-expert loop) if any projection isn't stackable, i.e. the
/// checkpoint uses the per-expert (still-quantized) GGUF layout rather
/// than the packed one, or if `experts` is empty.
fn batched_promote(
    experts: &[MoeExpert],
    device: &Device,
    dtype: DType,
) -> Result<Option<Vec<MoeExpert>>> {
    if experts.is_empty() {
        return Ok(None);
    }
    let Some(gate_all) = stack_projection(experts, device, dtype, |e| &e.gate_proj)? else {
        return Ok(None);
    };
    let Some(up_all) = stack_projection(experts, device, dtype, |e| &e.up_proj)? else {
        return Ok(None);
    };
    let Some(down_all) = stack_projection(experts, device, dtype, |e| &e.down_proj)? else {
        return Ok(None);
    };
    if let Some((free, mem_total)) = query_gpu_memory(device) {
        log::debug!(
            "batched promotion to {:?}: {} experts transferred in 3 allocations, free={}, total={}",
            device.location(),
            experts.len(),
            format_budget(free),
            format_budget(mem_total),
        );
    }
    (0..experts.len())
        .map(|i| {
            Ok(MoeExpert::from_layers(
                LinearLayer::Standard(Linear::new(narrow_packed_expert(&gate_all, i)?, None)),
                LinearLayer::Standard(Linear::new(narrow_packed_expert(&up_all, i)?, None)),
                LinearLayer::Standard(Linear::new(narrow_packed_expert(&down_all, i)?, None)),
            ))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

impl Module for SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dims = xs.dims().to_vec();
        let Some(&hidden_size) = original_dims.last() else {
            candle_core::bail!("SparseMoeBlock input must have at least one dimension");
        };
        let original_dtype = xs.dtype();
        let xs_flat = xs.reshape(((), hidden_size))?;
        let xs_f32 = xs_flat.to_dtype(DType::F32)?;

        let (topk_ids, topk_weights) =
            prof::timed(Span::MoeRouter, || -> Result<(Tensor, Tensor)> {
                let logits = self.gate.forward(&xs_f32)?;
                let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                let topk_ids = probs
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                    .contiguous()?;
                let mut topk_weights = probs.gather(&topk_ids, D::Minus1)?;
                if self.norm_topk_prob {
                    let sum = topk_weights.sum_keepdim(D::Minus1)?;
                    topk_weights = topk_weights.broadcast_div(&sum)?;
                }
                Ok((topk_ids, topk_weights))
            })?;

        // Routing dispatch is CPU-side: topk indices and weights are pulled to
        // the host each forward call, and per-expert token/weight lists below
        // are heap-allocated fresh each call. A fused GPU MoE kernel (as in
        // candle's moe_gemm_gguf) would eliminate both the sync and the
        // allocations, at the cost of losing the packed/per-expert layout
        // flexibility this dispatch loop gets for free.
        let topk_ids = topk_ids.to_vec2::<u32>()?;
        let topk_weights = topk_weights.to_vec2::<f32>()?;

        let mut token_lists: Vec<Vec<u32>> = vec![Vec::new(); self.experts.len()];
        let mut weight_lists: Vec<Vec<f32>> = vec![Vec::new(); self.experts.len()];
        for (token_idx, (ids, weights)) in topk_ids.iter().zip(topk_weights.iter()).enumerate() {
            // Token counts (batch * seq_len) never approach u32::MAX.
            #[allow(clippy::cast_possible_truncation)]
            let token_idx = token_idx as u32;
            for (&expert_idx, &weight) in ids.iter().zip(weights.iter()) {
                token_lists[expert_idx as usize].push(token_idx);
                weight_lists[expert_idx as usize].push(weight);
            }
        }

        // `Device::Cpu` is a unit variant, so this cross-device branch is only
        // exercised (and only exercisable in tests) on multi-device hardware.
        // All unit tests here run router and experts on the same CPU device.
        let same_device = xs_flat.device().location() == self.expert_device.location();
        let input_device = xs_flat.device();

        // The hidden state stays on `input_device` throughout: rather than
        // moving it to `expert_device` (Root Cause #4's device-transfer
        // cost, and a total mismatch for CPU-offloaded experts since GPU
        // matmuls vastly outrun CPU ones), only the handful of routed
        // experts' weights are copied to `input_device` per call, each
        // copy dropped at the end of its loop iteration.
        //
        // `MoeToDevice` and `MoeExpert` are timed as separate, non-nested
        // spans per expert (rather than one span wrapping the whole loop):
        // both are Tier2b siblings, and `prof::timed` never subtracts a
        // span's children, so nesting one inside the other would double-
        // count the copy time into `MoeExpert`'s bucket too.
        let mut output = Tensor::zeros(xs_flat.dims(), xs_flat.dtype(), input_device)?;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let tokens = &token_lists[expert_idx];
            if tokens.is_empty() {
                continue;
            }
            let copied_expert;
            let expert = if same_device {
                expert
            } else {
                copied_expert = prof::timed(Span::MoeToDevice, || {
                    copy_expert_to_device(expert, input_device)
                })?;
                &copied_expert
            };
            output = prof::timed(Span::MoeExpert, || -> Result<Tensor> {
                let token_ids = Tensor::new(tokens.as_slice(), input_device)?;
                let selected = xs_flat.index_select(&token_ids, 0)?;
                let expert_out = expert.forward(&selected)?;
                let weights = Tensor::new(weight_lists[expert_idx].as_slice(), input_device)?
                    .reshape((tokens.len(), 1))?
                    .to_dtype(expert_out.dtype())?;
                let scaled = expert_out.broadcast_mul(&weights)?;
                output.index_add(&token_ids, &scaled, 0)
            })?;
        }
        if output.dtype() != original_dtype {
            candle_core::bail!(
                "MoE output dtype {:?} differs from input dtype {:?}",
                output.dtype(),
                original_dtype,
            );
        }
        output.reshape(original_dims)
    }
}

/// Dense MLP or Mixture-of-Experts feed-forward layer.
///
/// Generic over the model's own dense MLP type `M`, so each model plugs in
/// its existing MLP struct unchanged (e.g. Qwen3's `Mlp`, Qwen3.5's `Mlp`).
/// `DecoderLayer` holds `MlpOrMoe<Mlp>` and calls `.forward()` without
/// branching on the layer type.
pub enum MlpOrMoe<M: Module> {
    /// Standard dense MLP (non-MoE layer).
    Dense(M),
    /// Mixture-of-Experts routing + dispatch block.
    Moe(SparseMoeBlock),
}

impl<M: Module> Module for MlpOrMoe<M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device, Tensor};

    // Keys use dot-separated format ("gate_proj.weight") to match VarBuilder::pp("gate_proj").
    fn make_vb(
        hidden: usize,
        intermediate: usize,
        gate_data: Vec<f32>,
        up_data: Vec<f32>,
        down_data: Vec<f32>,
    ) -> candle_nn::VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate_proj.weight".to_string(),
            Tensor::from_vec(gate_data, (intermediate, hidden), device).expect("gate weight"),
        );
        tensors.insert(
            "up_proj.weight".to_string(),
            Tensor::from_vec(up_data, (intermediate, hidden), device).expect("up weight"),
        );
        tensors.insert(
            "down_proj.weight".to_string(),
            Tensor::from_vec(down_data, (hidden, intermediate), device).expect("down weight"),
        );
        candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    fn zeros_expert(hidden: usize, intermediate: usize) -> MoeExpert {
        let gate = vec![0.0f32; hidden * intermediate];
        let up = vec![0.0f32; hidden * intermediate];
        let down = vec![0.0f32; intermediate * hidden];
        let vb = make_vb(hidden, intermediate, gate, up, down);
        MoeExpert::new(hidden, intermediate, vb).expect("MoeExpert::new")
    }

    fn identity_vb(hidden: usize) -> candle_nn::VarBuilder<'static> {
        let eye: Vec<f32> = (0..hidden * hidden)
            .map(|idx| {
                if idx / hidden == idx % hidden {
                    1.0_f32
                } else {
                    0.0
                }
            })
            .collect();
        make_vb(hidden, hidden, eye.clone(), eye.clone(), eye)
    }

    #[test]
    fn test_output_shape_2d() {
        let expert = zeros_expert(8, 16);
        let x = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[4, 8]);
    }

    #[test]
    fn test_output_shape_3d() {
        let expert = zeros_expert(16, 32);
        let x = Tensor::zeros((2, 5, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn test_zero_weights_give_zero_output() {
        let expert = zeros_expert(8, 16);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let y = expert.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero weights, got max={max_val}"
        );
    }

    #[test]
    fn test_zero_input_gives_zero_output() {
        let vb = identity_vb(8);
        let expert = MoeExpert::new(8, 8, vb).expect("new");
        let x = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero input, got max={max_val}"
        );
    }

    #[test]
    fn test_intermediate_size_larger_than_hidden() {
        let expert = zeros_expert(32, 128);
        let x = Tensor::zeros((1, 32), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[1, 32]);
    }

    #[test]
    fn test_formula_manual_verification() {
        // hidden=2, intermediate=2
        // gate_proj.weight = [[1, 0], [0, 1]]  (I, row-major so output[i] = x[i])
        // up_proj.weight   = [[2, 0], [0, 2]]  (2*I)
        // down_proj.weight = [[1, 0], [0, 1]]  (I)
        // x = [1.0, 0.5]
        //
        // Linear stores weight as [out, in] and computes x @ weight.T:
        //   gate = x @ I.T = [1.0, 0.5]
        //   gate_activated = silu([1.0, 0.5])
        //   up = x @ (2*I).T = [2.0, 1.0]
        //   product = gate_activated * up
        //   output = product @ I.T = product
        let device = &Device::Cpu;
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let doubled = vec![2.0f32, 0.0, 0.0, 2.0];
        let vb = make_vb(2, 2, identity.clone(), doubled, identity);
        let expert = MoeExpert::new(2, 2, vb).expect("new");

        let x = Tensor::new(&[1.0f32, 0.5], device)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = expert.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let silu = |v: f32| v / (1.0 + (-v).exp());
        let expected = [silu(1.0_f32) * 2.0, silu(0.5_f32) * 1.0];

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "output[{i}]: got {g}, expected {e}");
        }
    }

    #[test]
    fn test_batch_consistency() {
        // Identical rows in a batch should produce identical output rows.
        let device = &Device::Cpu;
        let hidden = 8usize;
        let intermediate = 16usize;
        let gate: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let up: Vec<f32> = gate.iter().map(|v| v * 0.5).collect();
        let down: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let vb = make_vb(hidden, intermediate, gate, up, down);
        let expert = MoeExpert::new(hidden, intermediate, vb).expect("new");

        let row: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let single = Tensor::from_vec(row, (1, hidden), device).expect("single row");
        let batch = Tensor::cat(&[&single, &single, &single], 0).expect("cat");

        let out_batch = expert.forward(&batch).expect("batch forward");
        let out_single = expert.forward(&single).expect("single forward");

        for b in 0..3 {
            let row_b = out_batch.narrow(0, b, 1).expect("narrow");
            let diff: f32 = (&row_b - &out_single)
                .expect("sub")
                .abs()
                .expect("abs")
                .max_all()
                .expect("max_all")
                .to_scalar()
                .expect("scalar");
            assert!(
                diff < 1e-6,
                "batch row {b} differs from single-row output, diff={diff}"
            );
        }
    }

    // GgmlDType::F32 quantization is lossless (block_size=1, no bit-packing),
    // so this exercises the real quantize -> byte-slice -> qtensor_from_ggml
    // path with exact value comparisons instead of tolerating quantization
    // error from a lossy dtype.
    #[test]
    fn test_slice_packed_qtensor() {
        use candle_core::quantized::GgmlDType;

        // [num_experts=3, out=2, in=4]; expert i's slice is filled with (i+1).
        let device = &Device::Cpu;
        let data: Vec<f32> = (0..3 * 2 * 4)
            .map(|idx| ((idx / (2 * 4)) + 1) as f32)
            .collect();
        let packed_tensor = Tensor::from_vec(data, (3, 2, 4), device).expect("packed tensor");
        let packed = QTensor::quantize(&packed_tensor, GgmlDType::F32).expect("quantize");
        let raw = packed.data().expect("raw bytes");

        for expert_idx in 0..3 {
            let sliced = slice_packed_qtensor(&raw, GgmlDType::F32, expert_idx, 2, 4, device)
                .expect("slice");
            assert_eq!(sliced.shape().dims(), &[2, 4]);
            assert_eq!(sliced.dtype(), GgmlDType::F32);
            let vals = sliced
                .dequantize(device)
                .expect("dequantize")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to_vec1");
            let expected_val = (expert_idx + 1) as f32;
            assert!(
                vals.iter().all(|&v| (v - expected_val).abs() < 1e-6),
                "expert {expert_idx}: got {vals:?}, expected all {expected_val}"
            );
        }
    }

    // GgmlDType::Q8_0 has block_size=32, unlike the F32 test above
    // (block_size=1) where the `expert_elems / block_size * type_size`
    // division is a no-op -- this exercises that division with a real
    // divisor, so a swapped block_size/type_size bug would be caught.
    #[test]
    fn test_slice_packed_qtensor_q8_0() {
        use candle_core::quantized::GgmlDType;

        // [num_experts=3, out=32, in=64]; expert i's slice is filled with (i+1).
        // out*in = 2048, a multiple of Q8_0's block_size=32.
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3, 32, 64);
        let data: Vec<f32> = (0..num_experts * out_dim * in_dim)
            .map(|idx| ((idx / (out_dim * in_dim)) + 1) as f32)
            .collect();
        let packed_tensor =
            Tensor::from_vec(data, (num_experts, out_dim, in_dim), device).expect("packed tensor");
        let packed = QTensor::quantize(&packed_tensor, GgmlDType::Q8_0).expect("quantize");
        let raw = packed.data().expect("raw bytes");

        for expert_idx in 0..num_experts {
            let sliced =
                slice_packed_qtensor(&raw, GgmlDType::Q8_0, expert_idx, out_dim, in_dim, device)
                    .expect("slice");
            assert_eq!(sliced.shape().dims(), &[out_dim, in_dim]);
            assert_eq!(sliced.dtype(), GgmlDType::Q8_0);
            let vals = sliced
                .dequantize(device)
                .expect("dequantize")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to_vec1");
            let expected_val = (expert_idx + 1) as f32;
            assert!(
                vals.iter().all(|&v| (v - expected_val).abs() < 0.1),
                "expert {expert_idx}: got {vals:?}, expected all ~{expected_val}"
            );
        }
    }

    // `load_packed_experts` must produce `Quantized` layers, not `Standard`
    // -- the whole point of Phase 3 is to keep packed experts quantized
    // instead of eagerly dequantizing them at load time.
    #[test]
    fn load_packed_experts_produces_quantized_layers() {
        use candle_core::quantized::GgmlDType;
        use std::io::Cursor;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let (num_experts, out_dim, in_dim) = (2usize, 4usize, 8usize);

        let mut writer = Cursor::new(Vec::new());
        let gate = QTensor::quantize(
            &Tensor::ones((num_experts, out_dim, in_dim), dtype, &device).expect("gate data"),
            GgmlDType::F32,
        )
        .expect("quantize gate");
        let up = QTensor::quantize(
            &Tensor::ones((num_experts, out_dim, in_dim), dtype, &device).expect("up data"),
            GgmlDType::F32,
        )
        .expect("quantize up");
        let down = QTensor::quantize(
            &Tensor::ones((num_experts, in_dim, out_dim), dtype, &device).expect("down data"),
            GgmlDType::F32,
        )
        .expect("quantize down");
        candle_core::quantized::gguf_file::write(
            &mut writer,
            &[],
            &[
                ("blk.0.ffn_gate_exps.weight", &gate),
                ("blk.0.ffn_up_exps.weight", &up),
                ("blk.0.ffn_down_exps.weight", &down),
            ],
        )
        .expect("write gguf");

        writer.set_position(0);
        let ct = candle_core::quantized::gguf_file::Content::read(&mut writer).expect("read gguf");
        let mut gg = Gguf::new(ct, writer, device.clone(), dtype);

        let experts = SparseMoeBlock::load_packed_experts(&mut gg, "blk.0", num_experts, &device)
            .expect("load_packed_experts");

        assert_eq!(experts.len(), num_experts);
        for expert in &experts {
            assert!(
                matches!(expert.gate_proj, LinearLayer::Quantized(_)),
                "gate_proj must stay Quantized"
            );
            assert!(
                matches!(expert.up_proj, LinearLayer::Quantized(_)),
                "up_proj must stay Quantized"
            );
            assert!(
                matches!(expert.down_proj, LinearLayer::Quantized(_)),
                "down_proj must stay Quantized"
            );
        }
    }

    // Verifies `MoeExpert::to_device` preserves forward-pass output. Only
    // CPU->CPU is exercisable without real GPU hardware (matching this
    // file's existing note that cross-device dispatch is CPU-only in unit
    // tests), but this still exercises the actual per-projection transfer
    // loop, unlike `SparseMoeBlock::promote_experts_to`'s same-device
    // early-return short-circuit tested separately below.
    #[test]
    fn moe_expert_to_device_preserves_output() {
        let vb = identity_vb(8);
        let expert = MoeExpert::new(8, 8, vb).expect("new");
        let x = Tensor::arange(0f32, 8f32, &Device::Cpu)
            .expect("arange")
            .reshape((1, 8))
            .expect("reshape");
        let before = expert.forward(&x).expect("forward");

        let moved = expert
            .to_device(&Device::Cpu, DType::F32)
            .expect("to_device");
        let after = moved.forward(&x).expect("forward");

        assert_eq!(
            before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    // Verifies `copy_linear_to_device` on a `Standard` layer (CPU->CPU
    // here, same constructibility limitation noted throughout this file)
    // preserves forward-pass output.
    #[test]
    fn copy_linear_to_device_standard_preserves_output() {
        let weight = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let layer = LinearLayer::Standard(Linear::new(weight, None));
        let x = Tensor::new(&[1.0f32, 0.5], &Device::Cpu)
            .unwrap()
            .reshape((1, 2))
            .unwrap();
        let before = layer.forward(&x).unwrap();

        let copied = copy_linear_to_device(&layer, &Device::Cpu).unwrap();
        assert!(matches!(copied, LinearLayer::Standard(_)));
        let after = copied.forward(&x).unwrap();

        assert_eq!(
            before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    // Unlike `LinearLayer::to_device`, `copy_linear_to_device` must keep a
    // `Quantized(QMatMul::QTensor)` layer quantized rather than
    // dequantizing it to `Standard` -- that's the entire point of Phase 6's
    // routed-only expert copy.
    #[test]
    fn copy_linear_to_device_quantized_stays_quantized() {
        use candle_core::quantized::GgmlDType;

        // Q8_0 (block_size=32), unlike F32, is not auto-dequantized by
        // `QMatMul::from_arc` -- F32/F16/BF16 always dequantize to
        // `QMatMul::Tensor`, which would make this test pass vacuously.
        let weight = Tensor::from_vec(vec![1.0f32; 2 * 32], (2, 32), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap();
        let layer = LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());
        let x = Tensor::from_vec(vec![1.0f32; 32], (1, 32), &Device::Cpu).unwrap();
        let before = layer.forward(&x).unwrap();

        let copied = copy_linear_to_device(&layer, &Device::Cpu).unwrap();
        assert!(
            matches!(copied, LinearLayer::Quantized(QMatMul::QTensor(_))),
            "copy must stay Quantized(QTensor), not dequantize"
        );
        let after = copied.forward(&x).unwrap();

        let before_vals = before.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let after_vals = after.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (b, a) in before_vals.iter().zip(after_vals.iter()) {
            assert!((b - a).abs() < 1e-4, "before={b} after={a}");
        }
    }

    // Verifies `copy_expert_to_device` (CPU->CPU) preserves forward-pass
    // output across all three projections.
    #[test]
    fn copy_expert_to_device_preserves_output() {
        let vb = identity_vb(8);
        let expert = MoeExpert::new(8, 8, vb).expect("new");
        let x = Tensor::arange(0f32, 8f32, &Device::Cpu)
            .expect("arange")
            .reshape((1, 8))
            .expect("reshape");
        let before = expert.forward(&x).expect("forward");

        let copied = copy_expert_to_device(&expert, &Device::Cpu).expect("copy");
        let after = copied.forward(&x).expect("forward");

        assert_eq!(
            before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    // Verifies `copy_expert_to_device` preserves quantization on all three
    // projections rather than dequantizing any of them.
    #[test]
    fn copy_expert_to_device_quantized_stays_quantized() {
        use candle_core::quantized::GgmlDType;

        // Q8_0 (block_size=32), unlike F32, is not auto-dequantized by
        // `QMatMul::from_arc` -- F32/F16/BF16 always dequantize to
        // `QMatMul::Tensor`, which would make this test pass vacuously.
        let make_quantized = || {
            let weight = Tensor::from_vec(vec![1.0f32; 2 * 32], (2, 32), &Device::Cpu).unwrap();
            let qt = QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap();
            LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap())
        };
        let expert = MoeExpert::from_layers(make_quantized(), make_quantized(), make_quantized());

        let copied = copy_expert_to_device(&expert, &Device::Cpu).expect("copy");
        assert!(matches!(
            copied.gate_proj,
            LinearLayer::Quantized(QMatMul::QTensor(_))
        ));
        assert!(matches!(
            copied.up_proj,
            LinearLayer::Quantized(QMatMul::QTensor(_))
        ));
        assert!(matches!(
            copied.down_proj,
            LinearLayer::Quantized(QMatMul::QTensor(_))
        ));
    }

    // Router weight key is "gate.weight" (VarBuilder::pp("gate")); expert keys
    // are "experts.{i}.{gate,up,down}_proj.weight" (VarBuilder::pp("experts.{i}")).
    fn make_sparse_moe_vb(
        hidden: usize,
        num_experts: usize,
        gate_data: Vec<f32>,
        expert_data: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> candle_nn::VarBuilder<'static> {
        let device = &Device::Cpu;
        let intermediate = expert_data[0].0.len() / hidden;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate.weight".to_string(),
            Tensor::from_vec(gate_data, (num_experts, hidden), device).expect("gate weight"),
        );
        for (i, (gate, up, down)) in expert_data.iter().enumerate() {
            tensors.insert(
                format!("experts.{i}.gate_proj.weight"),
                Tensor::from_vec(gate.clone(), (intermediate, hidden), device)
                    .expect("expert gate weight"),
            );
            tensors.insert(
                format!("experts.{i}.up_proj.weight"),
                Tensor::from_vec(up.clone(), (intermediate, hidden), device)
                    .expect("expert up weight"),
            );
            tensors.insert(
                format!("experts.{i}.down_proj.weight"),
                Tensor::from_vec(down.clone(), (hidden, intermediate), device)
                    .expect("expert down weight"),
            );
        }
        candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    fn make_sparse_moe(
        hidden: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        gate_data: Vec<f32>,
        expert_data: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> SparseMoeBlock {
        let vb = make_sparse_moe_vb(hidden, num_experts, gate_data, expert_data);
        let config = MoeConfig {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob,
            decoder_sparse_step: None,
        };
        SparseMoeBlock::new(&config, hidden, vb, &Device::Cpu).expect("SparseMoeBlock::new")
    }

    fn zeros_sparse_moe(
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
    ) -> SparseMoeBlock {
        let gate_data = vec![0.0f32; num_experts * hidden];
        let expert_data: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..num_experts)
            .map(|_| {
                (
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; intermediate * hidden],
                )
            })
            .collect();
        make_sparse_moe(
            hidden,
            intermediate,
            num_experts,
            num_experts_per_tok,
            false,
            gate_data,
            &expert_data,
        )
    }

    #[test]
    fn test_sparse_moe_output_shape_2d() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::zeros((3, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[3, 8]);
    }

    #[test]
    fn test_sparse_moe_output_shape_3d() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[2, 3, 8]);
    }

    #[test]
    fn test_single_token_routing() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::zeros((1, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[1, 8]);
    }

    // Verifies `promote_experts_to` is a safe no-op when already on the
    // target device (the `expert_device.location() == device.location()`
    // early return) — output and forward behavior stay identical. The
    // actual cross-device transfer loop is exercised at the `MoeExpert`/
    // `LinearLayer` unit level instead, since a second distinct device
    // isn't constructible without real GPU hardware (see this file's
    // existing note on `SparseMoeBlock::forward`'s `same_device` check).
    #[test]
    fn promote_experts_to_same_device_is_noop() {
        let (mut moe, _, _) = routing_test_setup(false, 2);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let before = moe.forward(&x).expect("forward");

        moe.promote_experts_to(&Device::Cpu, DType::F32)
            .expect("promote");
        let after = moe.forward(&x).expect("forward");

        assert_eq!(
            before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    // Verifies `batched_promote`'s stack-transfer-narrow path preserves
    // each expert's distinct forward-pass output and index order (CPU->CPU
    // here, same constructibility limitation as `promote_experts_to_*`
    // above). Each expert's `up_proj` is scaled differently so a bug that
    // mixed up narrowed slices across experts would be caught.
    #[test]
    fn batched_promote_preserves_per_expert_output_and_order() {
        let hidden = 2;
        let intermediate = 2;
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let scaled = |c: f32| vec![c, 0.0, 0.0, c];
        let experts: Vec<MoeExpert> = (0..3)
            .map(|i| {
                let vb = make_vb(
                    hidden,
                    intermediate,
                    identity.clone(),
                    scaled((i + 1) as f32),
                    identity.clone(),
                );
                MoeExpert::new(hidden, intermediate, vb).expect("new")
            })
            .collect();

        let promoted = batched_promote(&experts, &Device::Cpu, DType::F32)
            .expect("batched_promote")
            .expect("packed-style experts should take the batched path");
        assert_eq!(promoted.len(), 3);

        let x = Tensor::new(&[1.0f32, 0.5], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        for (i, expert) in promoted.iter().enumerate() {
            let got = expert
                .forward(&x)
                .expect("forward")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let scale = (i + 1) as f32;
            let silu = |v: f32| v / (1.0 + (-v).exp());
            // gate = identity(x) = x; up = scale * x; down = identity, so
            // output = silu(x) * (scale * x) elementwise.
            let expected = [silu(1.0f32) * scale * 1.0, silu(0.5f32) * scale * 0.5];
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!(
                    (g - e).abs() < 1e-5,
                    "expert {i}: got {got:?}, expected {expected:?}"
                );
            }
        }
    }

    // Verifies the per-expert quantized GGUF layout (which can't be stacked
    // into one tensor) makes `batched_promote` decline rather than silently
    // dropping data, so `promote_experts_to` falls back to its per-expert
    // loop instead.
    #[test]
    fn batched_promote_declines_quantized_experts() {
        use candle_core::quantized::GgmlDType;

        let weight = Tensor::from_vec(vec![1.0f32; 4], (2, 2), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&weight, GgmlDType::F32).unwrap();
        let quantized = LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());
        let expert = MoeExpert::from_layers(
            quantized,
            LinearLayer::Standard(Linear::new(weight.clone(), None)),
            LinearLayer::Standard(Linear::new(weight, None)),
        );

        let result = batched_promote(std::slice::from_ref(&expert), &Device::Cpu, DType::F32)
            .expect("batched_promote should not error, just decline");
        assert!(result.is_none());
    }

    // Verifies a bias-present `Standard` layer (distinct from the
    // `Quantized`-variant decline above) also makes `batched_promote`
    // decline rather than silently dropping the bias.
    #[test]
    fn batched_promote_declines_biased_experts() {
        let weight = Tensor::from_vec(vec![1.0f32; 4], (2, 2), &Device::Cpu).unwrap();
        let bias = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
        let biased = LinearLayer::Standard(Linear::new(weight.clone(), Some(bias)));
        let expert = MoeExpert::from_layers(
            biased,
            LinearLayer::Standard(Linear::new(weight.clone(), None)),
            LinearLayer::Standard(Linear::new(weight, None)),
        );

        let result = batched_promote(std::slice::from_ref(&expert), &Device::Cpu, DType::F32)
            .expect("batched_promote should not error, just decline");
        assert!(result.is_none());
    }

    #[test]
    fn test_all_experts_zero_weights_gives_zero_output() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let y = moe.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero weights, got max={max_val}"
        );
    }

    // 4 experts, top-2. Router logits (from x=[1,0] and gate rows
    // [2,0],[-2,0],[1.5,0],[-1.5,0]) are [2,-2,1.5,-1.5] — close enough
    // together that normalization measurably changes the combined weight,
    // while still unambiguously ranking experts 0 and 2 highest. Each expert
    // e has gate_proj=I, up_proj=c_e*I, down_proj=I, so
    // expert_e(x) = c_e * x * silu(x) elementwise. Experts 1 and 3 (not
    // selected) use a huge c=100 so any routing bug that includes them is
    // trivially detectable.
    fn routing_test_setup(norm_topk_prob: bool, top_k: usize) -> (SparseMoeBlock, f32, f32) {
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let scaled = |c: f32| vec![c, 0.0, 0.0, c];
        let expert_data = vec![
            (identity.clone(), scaled(1.0), identity.clone()),
            (identity.clone(), scaled(100.0), identity.clone()),
            (identity.clone(), scaled(2.0), identity.clone()),
            (identity.clone(), scaled(100.0), identity.clone()),
        ];
        let gate_data = vec![
            2.0, 0.0, //
            -2.0, 0.0, //
            1.5, 0.0, //
            -1.5, 0.0,
        ];
        let moe = make_sparse_moe(2, 2, 4, top_k, norm_topk_prob, gate_data, &expert_data);

        let logits = [2.0f32, -2.0, 1.5, -1.5];
        let exps: Vec<f32> = logits.iter().map(|l| l.exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        (moe, probs[0], probs[2])
    }

    #[test]
    fn test_routing_selects_top_k_experts() {
        let (moe, weight0, weight2) = routing_test_setup(false, 2);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = moe.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expected0 = (weight0 * 1.0 + weight2 * 2.0) * silu_1;
        assert!(
            (got[0] - expected0).abs() < 1e-4,
            "got {got:?}, expected [{expected0}, 0.0] (only experts 0 and 2 should contribute)"
        );
        assert!(
            got[1].abs() < 1e-5,
            "got {got:?}, expected second element 0"
        );
    }

    // Degenerate case: top-1 routing, so only expert 0 (the highest router
    // logit) has a non-zero routing weight. Reuses routing_test_setup's
    // poison-weighted experts 1 and 3 so any bug that leaks them into the
    // output is trivially detectable.
    #[test]
    fn test_single_expert_routing() {
        let (moe, weight0, _) = routing_test_setup(false, 1);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = moe.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expected0 = weight0 * 1.0 * silu_1;
        assert!(
            (got[0] - expected0).abs() < 1e-4,
            "got {got:?}, expected [{expected0}, 0.0] (only expert 0 should contribute)"
        );
        assert!(
            got[1].abs() < 1e-5,
            "got {got:?}, expected second element 0"
        );
    }

    // Degenerate case: top-1 with norm_topk_prob=true divides the single
    // selected expert's raw softmax weight by itself, so the applied routing
    // weight must always be exactly 1.0 regardless of the raw router logits.
    #[test]
    fn test_single_expert_routing_norm_topk_prob_forces_weight_one() {
        let (moe, _, _) = routing_test_setup(true, 1);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = moe.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!(
            (got[0] - silu_1).abs() < 1e-4,
            "got {got:?}, expected [{silu_1}, 0.0] (normalized top-1 weight must be 1.0)"
        );
        assert!(
            got[1].abs() < 1e-5,
            "got {got:?}, expected second element 0"
        );
    }

    #[test]
    fn test_norm_topk_prob_normalizes_weights() {
        let (moe, weight0, weight2) = routing_test_setup(true, 2);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = moe.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let norm0 = weight0 / (weight0 + weight2);
        let norm2 = weight2 / (weight0 + weight2);
        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expected0 = (norm0 * 1.0 + norm2 * 2.0) * silu_1;
        assert!(
            (got[0] - expected0).abs() < 1e-4,
            "got {got:?}, expected [{expected0}, 0.0] with normalized top-k weights"
        );

        // Normalized weights must sum to 1, so this must differ from the
        // un-normalized (raw softmax) result.
        let raw_expected0 = (weight0 * 1.0 + weight2 * 2.0) * silu_1;
        assert!(
            (expected0 - raw_expected0).abs() > 1e-4,
            "normalized and raw-weighted outputs should differ"
        );
    }

    // Regression test: a checkpoint whose weights load in a dtype other than
    // F32 (e.g. F16/BF16 in production) must not have its gate weight
    // forwarded against `forward`'s F32-upcast router input, which used to
    // hard-error on a dtype mismatch in the router matmul. Uses F16 rather
    // than BF16 because candle's CPU matmul backend (no mkl/accelerate)
    // only supports F16/F32/F64. That is an unrelated candle limitation,
    // not something this test is meant to exercise.
    #[test]
    fn test_sparse_moe_non_f32_checkpoint_does_not_crash() {
        let device = &Device::Cpu;
        let hidden = 8;
        let num_experts = 4;
        let intermediate = 4;
        let gate_data = vec![0.0f32; num_experts * hidden];
        let expert_data: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..num_experts)
            .map(|_| {
                (
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; intermediate * hidden],
                )
            })
            .collect();

        let to_f16 = |data: Vec<f32>, shape: (usize, usize)| {
            Tensor::from_vec(data, shape, device)
                .expect("tensor")
                .to_dtype(DType::F16)
                .expect("to f16")
        };
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate.weight".to_string(),
            to_f16(gate_data, (num_experts, hidden)),
        );
        for (i, (gate, up, down)) in expert_data.into_iter().enumerate() {
            tensors.insert(
                format!("experts.{i}.gate_proj.weight"),
                to_f16(gate, (intermediate, hidden)),
            );
            tensors.insert(
                format!("experts.{i}.up_proj.weight"),
                to_f16(up, (intermediate, hidden)),
            );
            tensors.insert(
                format!("experts.{i}.down_proj.weight"),
                to_f16(down, (hidden, intermediate)),
            );
        }
        let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F16, device);
        let config = MoeConfig {
            num_experts,
            num_experts_per_tok: 2,
            moe_intermediate_size: intermediate,
            norm_topk_prob: false,
            decoder_sparse_step: None,
        };
        let moe = SparseMoeBlock::new(&config, hidden, vb, device).expect("SparseMoeBlock::new");

        let x = Tensor::zeros((3, hidden), DType::F16, device).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dtype(), DType::F16);
        assert_eq!(y.dims(), &[3, hidden]);
    }

    #[test]
    fn test_mlp_or_moe_dense_passthrough() {
        let expert = zeros_expert(8, 16);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let direct = expert.forward(&x).expect("direct forward");

        let wrapped: MlpOrMoe<MoeExpert> = MlpOrMoe::Dense(zeros_expert(8, 16));
        let via_enum = wrapped.forward(&x).expect("enum forward");

        let diff: f32 = (&direct - &via_enum)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-8,
            "MlpOrMoe::Dense output should match direct expert forward, diff={diff}"
        );
    }
}
