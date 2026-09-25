// SPDX-License-Identifier: MIT

use crate::device::{format_budget, query_gpu_memory};
use crate::models::hunyuan_dense::modeling::Gguf;
use crate::ops::linear::LinearLayer;
use crate::utils::DeviceExt;
use crate::utils::prof::{self, Span};
use candle_core::quantized::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8_1, BlockQ8K,
};
use candle_core::quantized::{GgmlDType, GgmlType, QMatMul, QTensor, ggml_file::qtensor_from_ggml};
use candle_core::utils::barrier_pool;
use candle_core::{D, DType, Device, Module, Result, Tensor};
#[cfg(test)]
use candle_nn::Activation;
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use half::{bf16, f16};
use ribo::utils::log;
use std::io::{Read, Seek};
use std::sync::{Arc, Mutex};

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
pub struct MoeExpert {
    /// Fused gate+up projection, `(2 * moe_intermediate_size, hidden_size)`.
    /// The first half of the output is the gate projection, the second half
    /// is up (see [`fuse_quantized_qtensors`]). Fusing the two into one
    /// matmul halves the kernel launches this step needs.
    gate_up_proj: LinearLayer,
    down_proj: LinearLayer,
    /// Output size of the gate/up half, i.e. `moe_intermediate_size`. Needed
    /// to split `gate_up_proj`'s fused output back into gate and up halves.
    intermediate_size: usize,
}

impl MoeExpert {
    /// Create an expert from a safetensors checkpoint.
    ///
    /// Fuses the separately-stored `gate_proj` and `up_proj` weights into a
    /// single `gate_up_proj` via `Tensor::cat`.
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
        let gate_up_weight = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        Ok(Self {
            gate_up_proj: LinearLayer::Standard(Linear::new(gate_up_weight, None)),
            down_proj: LinearLayer::Standard(down_proj),
            intermediate_size,
        })
    }

    /// Create an expert from a GGUF checkpoint using the per-expert tensor
    /// layout (`blk.{layer_idx}.ffn_gate.{expert_idx}.weight`, etc.).
    ///
    /// Fuses the separately-stored gate and up `QTensor`s into a single
    /// `gate_up_proj` via [`fuse_quantized_qtensors`].
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
        let gate_qt = gg.tensor_on(&format!("{prefix}.ffn_gate.{expert_idx}.weight"), device)?;
        let up_qt = gg.tensor_on(&format!("{prefix}.ffn_up.{expert_idx}.weight"), device)?;
        let intermediate_size = gate_qt.shape().dims()[0];
        let down_proj = gg.linear_on(&format!("{prefix}.ffn_down.{expert_idx}.weight"), device)?;
        let gate_up_qt = fuse_quantized_qtensors(&gate_qt, &up_qt, device)?;
        let gate_up_proj = LinearLayer::quantized(QMatMul::from_arc(Arc::new(gate_up_qt))?);
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    /// Build an expert directly from already-loaded projections.
    ///
    /// Used by [`SparseMoeBlock::new_from_gguf`] for the packed GGUF expert
    /// layout, where the projections come from byte-slicing a shared 3D
    /// tensor rather than loading per-expert tensors, and by
    /// `batched_promote` for the same narrow-from-a-stacked-tensor pattern
    /// when promoting experts to another device.
    fn from_layers(
        gate_up_proj: LinearLayer,
        down_proj: LinearLayer,
        intermediate_size: usize,
    ) -> Self {
        Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        }
    }

    /// Moves both projections to `device`, in `dtype`. See
    /// [`LinearLayer::to_device`] for the `Quantized` dequantization
    /// tradeoff this implies and why `dtype` must match the model's
    /// compute dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if either projection's device transfer fails.
    pub fn to_device(&self, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            gate_up_proj: self.gate_up_proj.to_device(device, dtype)?,
            down_proj: self.down_proj.to_device(device, dtype)?,
            intermediate_size: self.intermediate_size,
        })
    }

    /// Runs the expert's forward pass entirely in F32, regardless of the
    /// stored weight dtype. If `xs` is not F32, each projection casts it
    /// internally via [`LinearLayer::forward_f32`].
    ///
    /// Same gate/up split and `SiLU` logic as [`Module::forward`], routed
    /// through [`LinearLayer::forward_f32`] instead of `forward` so no
    /// intermediate BF16/F16 casts happen inside either projection.
    ///
    /// # Errors
    ///
    /// Returns an error if either projection's matmul fails.
    pub fn forward_f32(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward_f32(xs)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let hidden = prof::timed(Span::MoeActivation, || {
            crate::ops::fused_ops::swiglu::swiglu(&gate, &up)
        })?;
        self.down_proj.forward_f32(&hidden)
    }

    /// Whether this expert's projections are stored as quantized weights.
    fn is_quantized(&self) -> bool {
        matches!(self.gate_up_proj, LinearLayer::Quantized(_))
    }
}

impl Module for MoeExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let hidden = prof::timed(Span::MoeActivation, || {
            crate::ops::fused_ops::swiglu::swiglu(&gate, &up)
        })?;
        self.down_proj.forward(&hidden)
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
    /// Decoder layer index, used only to identify this block in dispatch
    /// placement logging (see [`Self::log_dispatch_decision`]).
    layer_idx: usize,
    gate: LinearLayer,
    experts: Vec<MoeExpert>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    expert_device: Device,
    /// Packed `(num_experts, 2 * moe_intermediate, hidden)` fused gate+up
    /// projection weights, for either the GPU `indexed_moe_forward` fused
    /// dispatch path (CUDA/ROCm) or the CPU batched dispatch path
    /// ([`cpu_indexed_moe_forward`]). `Some` when [`supports_fused_moe`]
    /// held for every packed projection at load time (GPU), or when
    /// `expert_device` is CPU; `experts` is left empty in either case
    /// since the batched path replaces the per-expert loop entirely
    /// (see [`load_packed_experts`]'s doc comment on why both aren't kept).
    packed_gate_up_exps: Option<Arc<QTensor>>,
    /// Packed `(num_experts, hidden, moe_intermediate)` down-projection
    /// weights. See [`Self::packed_gate_up_exps`].
    packed_down_exps: Option<Arc<QTensor>>,
    /// Reused token-to-expert routing buffers for the CPU batched dispatch
    /// path. `Mutex` because [`Module::forward`] takes `&self`, but
    /// [`MoeRouting::compute`] needs `&mut self` to refill its buffers in
    /// place across calls (see [`MoeRouting`]'s doc comment on why reuse
    /// matters on the decode hot path). `Mutex` rather than `RefCell` so
    /// `SparseMoeBlock` stays `Sync`; lock contention is a non-issue since
    /// nothing calls `forward` concurrently on the same block today.
    routing: Mutex<MoeRouting>,
}

impl SparseMoeBlock {
    /// Create a `MoE` block from a safetensors checkpoint.
    ///
    /// # Arguments
    /// * `config` - `MoE` layer configuration
    /// * `layer_idx` - Decoder layer index (identifies this block in
    ///   dispatch placement logging only)
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
        layer_idx: usize,
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
        // Redirect the builder before loading experts so their weights land
        // on `expert_device` rather than `vb`'s (main) device.
        let expert_vb = vb.set_device(expert_device.clone());
        let experts = (0..config.num_experts)
            .map(|i| {
                MoeExpert::new(
                    hidden_size,
                    config.moe_intermediate_size,
                    expert_vb.pp(format!("experts.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let block = Self {
            layer_idx,
            gate: LinearLayer::Standard(gate),
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_device: expert_device.clone(),
            packed_gate_up_exps: None,
            packed_down_exps: None,
            routing: Mutex::new(MoeRouting::new()),
        };
        block.log_dispatch_decision();
        Ok(block)
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
        let fused_packed_name = format!("{prefix}.ffn_gate_up_exps.weight");
        let (experts, packed_gate_up_exps, packed_down_exps) =
            if gg.contains_tensor(&packed_name) || gg.contains_tensor(&fused_packed_name) {
                Self::load_packed_experts(gg, &prefix, config.num_experts, expert_device)?
            } else {
                let experts = (0..config.num_experts)
                    .map(|expert_idx| {
                        MoeExpert::new_from_gguf(gg, layer_idx, expert_idx, expert_device)
                    })
                    .collect::<Result<Vec<_>>>()?;
                (experts, None, None)
            };

        let block = Self {
            layer_idx,
            gate,
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_device: expert_device.clone(),
            packed_gate_up_exps,
            packed_down_exps,
            routing: Mutex::new(MoeRouting::new()),
        };
        block.log_dispatch_decision();
        Ok(block)
    }

    /// Load the packed (Unsloth-style) `_exps` expert tensor layout: each
    /// projection is a single 3D tensor `[num_experts, out, in]` covering all
    /// experts, loaded onto `expert_device`.
    ///
    /// Gate and up are always fused into one `[num_experts, 2*out, in]`
    /// tensor before further processing: either loaded directly from a
    /// pre-fused `ffn_gate_up_exps` tensor (some GGUF converters emit this),
    /// or by fusing separate `ffn_gate_exps`/`ffn_up_exps` tensors via
    /// [`fuse_packed_qtensors`]. This halves the matmul kernel launches
    /// `SparseMoeBlock::forward` needs for that step.
    ///
    /// When [`supports_fused_moe`] holds for the fused gate+up and down
    /// projections' quant types (CUDA/`ROCm`, `Q2K`-`Q6K` or `Q8_0`), the
    /// packed tensors are kept intact and returned directly for the GPU
    /// `indexed_moe_forward` dispatch, with an empty `Vec<MoeExpert>`.
    /// When `expert_device` is CPU, the packed tensors are likewise kept
    /// intact for the CPU batched dispatch path
    /// ([`cpu_indexed_moe_forward`]), which supports every `GgmlDType`
    /// (not just the CUDA/`ROCm`-eligible subset). Keeping both the packed
    /// and per-expert copies would double memory for expert weights in
    /// either case. Only Metal or an unsupported quant type on a
    /// non-CPU/non-fused-eligible device falls back to byte-slicing each
    /// packed tensor into per-expert 2D `QTensor`s without dequantizing.
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
    #[allow(clippy::type_complexity)]
    fn load_packed_experts<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        expert_device: &Device,
    ) -> Result<(Vec<MoeExpert>, Option<Arc<QTensor>>, Option<Arc<QTensor>>)> {
        let down_packed = gg.tensor_on(&format!("{prefix}.ffn_down_exps.weight"), expert_device)?;
        let down_dims = down_packed.shape().dims().to_vec();
        let (down_out, down_in) = (down_dims[1], down_dims[2]);
        let down_dtype = down_packed.dtype();

        let fused_name = format!("{prefix}.ffn_gate_up_exps.weight");
        let gate_up_packed = if gg.contains_tensor(&fused_name) {
            gg.tensor_on(&fused_name, expert_device)?
        } else {
            let gate_packed =
                gg.tensor_on(&format!("{prefix}.ffn_gate_exps.weight"), expert_device)?;
            let up_packed = gg.tensor_on(&format!("{prefix}.ffn_up_exps.weight"), expert_device)?;
            fuse_packed_qtensors(&gate_packed, &up_packed, expert_device)?
        };

        // Shape is [num_experts, out_dim, in_dim].
        let gate_up_dims = gate_up_packed.shape().dims();
        if gate_up_dims[0] != num_experts {
            candle_core::bail!(
                "{prefix}: config num_experts={num_experts} does not match packed tensor's expert dim {}",
                gate_up_dims[0]
            );
        }
        let (gate_up_out, gate_up_in) = (gate_up_dims[1], gate_up_dims[2]);
        let gate_up_dtype = gate_up_packed.dtype();

        if supports_fused_moe(expert_device, gate_up_dtype)
            && supports_fused_moe(expert_device, down_dtype)
        {
            return Ok((
                Vec::new(),
                Some(Arc::new(gate_up_packed)),
                Some(Arc::new(down_packed)),
            ));
        }

        // CPU batched dispatch (`cpu_indexed_moe_forward`) supports every
        // `GgmlDType`, so unlike the GPU fused path above there is no
        // quant-type eligibility check here: any CPU-resident packed tensor
        // qualifies.
        if expert_device.is_cpu() {
            return Ok((
                Vec::new(),
                Some(Arc::new(gate_up_packed)),
                Some(Arc::new(down_packed)),
            ));
        }

        let gate_up_raw = gate_up_packed.data()?;
        let down_raw = down_packed.data()?;

        let experts = unpack_experts_from_packed(
            &gate_up_raw,
            gate_up_dtype,
            gate_up_out,
            gate_up_in,
            &down_raw,
            down_dtype,
            down_out,
            down_in,
            num_experts,
            expert_device,
        )?;
        Ok((experts, None, None))
    }

    /// Moves this block's expert weights to `device`, in `dtype`. No-op if
    /// experts already live on `device`. Used to promote CPU-offloaded `MoE`
    /// experts to GPU once real VRAM headroom is known post-construction
    /// (see `Qwen3Model::promote_experts_to_gpu`), rather than deciding
    /// placement while the model is still loading.
    ///
    /// If this block holds CPU-resident packed tensors (the CPU batched
    /// dispatch path), promotes them directly: to another packed tensor on
    /// `device` if [`supports_fused_moe`] holds there (staying on the GPU
    /// fused path), or by byte-slicing into per-expert `QTensor`s otherwise
    /// (falling back to the per-expert loop, matching what
    /// [`load_packed_experts`] would have produced for a non-fused-eligible
    /// device at load time).
    ///
    /// Otherwise, tries [`batched_promote`] first (one device transfer per
    /// projection instead of one per expert); falls back to a per-expert
    /// loop if the checkpoint's expert layout isn't stackable (e.g.
    /// still-quantized per-expert GGUF tensors).
    ///
    /// # Errors
    ///
    /// Returns an error if any expert's device transfer fails.
    pub fn promote_experts_to(&mut self, device: &Device, dtype: DType) -> Result<()> {
        if self.expert_device.location() == device.location() {
            return Ok(());
        }
        if let (Some(gate_up), Some(down)) = (&self.packed_gate_up_exps, &self.packed_down_exps) {
            // Compute the promoted result into locals first and only commit
            // to `self` on full success: a mid-way failure (e.g. an
            // out-of-memory device transfer for the down projection after
            // the gate+up one already succeeded) must leave the block
            // exactly as it was, matching the atomicity discipline the
            // per-expert fallback below already follows.
            let gu_dtype = gate_up.dtype();
            let dn_dtype = down.dtype();
            let promoted =
                if supports_fused_moe(device, gu_dtype) && supports_fused_moe(device, dn_dtype) {
                    let new_gate_up = upload_qtensor(gate_up, device)?;
                    let new_down = upload_qtensor(down, device)?;
                    (
                        Vec::new(),
                        Some(Arc::new(new_gate_up)),
                        Some(Arc::new(new_down)),
                    )
                } else {
                    let gu_dims = gate_up.shape().dims().to_vec();
                    let dn_dims = down.shape().dims().to_vec();
                    let (num_experts, gu_out, gu_in) = (gu_dims[0], gu_dims[1], gu_dims[2]);
                    let (dn_out, dn_in) = (dn_dims[1], dn_dims[2]);
                    let gu_raw = gate_up.data()?;
                    let dn_raw = down.data()?;
                    let experts = unpack_experts_from_packed(
                        &gu_raw,
                        gu_dtype,
                        gu_out,
                        gu_in,
                        &dn_raw,
                        dn_dtype,
                        dn_out,
                        dn_in,
                        num_experts,
                        device,
                    )?;
                    (experts, None, None)
                };
            self.experts = promoted.0;
            self.packed_gate_up_exps = promoted.1;
            self.packed_down_exps = promoted.2;
            self.expert_device = device.clone();
            self.log_dispatch_decision();
            return Ok(());
        }
        match batched_promote(&self.experts, device, dtype) {
            Ok(Some(moved)) => {
                self.experts = moved;
                self.expert_device = device.clone();
                self.log_dispatch_decision();
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
        }
        self.experts = moved;
        self.expert_device = device.clone();
        self.log_dispatch_decision();
        Ok(())
    }

    /// Returns the device that this block's expert weights currently live on.
    #[must_use]
    pub fn expert_device(&self) -> &Device {
        &self.expert_device
    }

    /// Fused GPU `MoE` dispatch via `indexed_moe_forward` (Phase 8a/10): two
    /// kernel launches (gate+up, down) against the packed 3D expert
    /// tensors, instead of a per-expert `index_select`/`forward`/`index_add`
    /// loop. `topk_ids`/`topk_weights` stay on-device throughout.
    ///
    /// `xs_f32` is `(num_tokens, hidden_size)`. For the gate+up projection,
    /// every routed expert of a token shares the same input row
    /// (`indexed_moe_forward`'s `input_dim1 == 1` case), so `xs_f32` is
    /// unsqueezed to `(num_tokens, 1, hidden_size)`. Its fused output,
    /// `(num_tokens, topk, 2*moe_intermediate_size)`, is split back into
    /// gate and up halves. For the down projection, each routed expert has
    /// its own intermediate activation (`input_dim1 == topk`), which the
    /// `(num_tokens, topk, moe_intermediate_size)` `hidden` tensor already
    /// matches.
    ///
    /// # Errors
    ///
    /// Returns an error if any `indexed_moe_forward` call fails (e.g. a
    /// device/dtype mismatch) or if a tensor op fails.
    fn fused_forward(
        xs_f32: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        gate_up_exps: &QTensor,
        down_exps: &QTensor,
        original_dtype: DType,
        original_dims: &[usize],
    ) -> Result<Tensor> {
        let intermediate_size = gate_up_exps.shape().dims()[1] / 2;
        let xs_3d = prof::timed(Span::MoeInputPrep, || xs_f32.unsqueeze(1)?.contiguous())?;
        let gate_up_out = prof::timed(Span::MoeGateUp, || {
            gate_up_exps.indexed_moe_forward(&xs_3d, topk_ids)
        })?;
        let gate = gate_up_out.narrow(D::Minus1, 0, intermediate_size)?;
        let up = gate_up_out.narrow(D::Minus1, intermediate_size, intermediate_size)?;
        let hidden = prof::timed(Span::MoeActivation, || {
            crate::ops::fused_ops::swiglu::swiglu(&gate, &up)
        })?;
        let down_out = prof::timed(Span::MoeDownProj, || {
            down_exps.indexed_moe_forward(&hidden, topk_ids)
        })?;

        prof::timed(Span::MoeCombine, || {
            Self::combine_expert_outputs(&down_out, topk_weights, original_dtype, original_dims)
        })
    }

    /// CPU-native equivalent of [`Self::fused_forward`], using
    /// [`cpu_indexed_moe_forward`] (the CPU batched kernel) instead of
    /// `QTensor::indexed_moe_forward` (CUDA/`ROCm`-only). Two batched
    /// dispatches (gate+up, then down) replace the per-expert loop's up to
    /// 336 individually-dispatched matmuls.
    ///
    /// `xs_f32`, `topk_ids`, and `topk_weights` may live on any device
    /// (e.g. GPU, when the rest of the model runs there and only `MoE`
    /// experts are CPU-offloaded); this method moves only the small
    /// activation and routing-weight tensors to CPU and moves the small
    /// result back to `xs_f32`'s original device, mirroring Phase 1's
    /// activation-not-weights strategy for the cross-device case.
    ///
    /// `self.routing` is reused across the gate+up and down calls within
    /// one forward pass (both route the same `topk_ids`), and across
    /// forward calls, to avoid reallocating its internal buffers on the
    /// decode hot path.
    ///
    /// Timed internally via the same `MoeToDevice`/`MoeExpert`/`MoeMisc`
    /// spans the per-expert loop below uses (not `MoeFused`, which stays
    /// reserved for the true GPU `indexed_moe_forward` dispatch), so this
    /// path's `to_dev`/`expert`/`misc` cost breaks down the same way the
    /// per-expert loop's does -- e.g. to isolate whether a slowdown is in
    /// the CPU/GPU activation transfer or in the batched kernel's own
    /// compute. One difference from the per-expert loop: `MoeActivation`
    /// (timing the `swiglu()` call below) runs between the two
    /// `MoeExpert` spans here rather than nested inside one, so unlike the
    /// per-expert loop, its time is additional on top of `MoeExpert`'s, not
    /// a subset of it. Each of `MoeToDevice`'s three calls is additionally
    /// wrapped in its own `MoeCpuXsDev`/`MoeCpuWeightsDev`/`MoeCpuOutDev`
    /// span, since on `--features rocm` each crossing independently pays a
    /// full stream-drain-then-copy (see candle-rocm's
    /// `SendSyncDeviceMemory::copy_from_host`/`copy_to_host`) regardless of
    /// the few-KB payload -- this breakdown is what tells apart a
    /// per-call-overhead-bound `to_dev` (all three roughly equal despite
    /// `topk_weights` being far smaller than `xs_f32`/the combined output)
    /// from a bytes-bound one.
    ///
    /// # Errors
    ///
    /// Returns an error if the routing mutex is poisoned, if routing
    /// computation fails, if any `cpu_indexed_moe_forward` call fails, or
    /// if a tensor op fails.
    #[allow(clippy::too_many_arguments)]
    fn cpu_batched_forward(
        &self,
        xs_f32: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        gate_up_exps: &QTensor,
        down_exps: &QTensor,
        original_dtype: DType,
        original_dims: &[usize],
    ) -> Result<Tensor> {
        let input_device = xs_f32.device();
        let num_experts = gate_up_exps.shape().dims()[0];
        let intermediate_size = gate_up_exps.shape().dims()[1] / 2;
        let xs_cpu = prof::timed(Span::MoeToDevice, || {
            prof::timed(Span::MoeCpuXsDev, || xs_f32.to_device(&Device::Cpu))
        })?;
        let topk_weights_cpu = prof::timed(Span::MoeToDevice, || {
            prof::timed(Span::MoeCpuWeightsDev, || {
                topk_weights.to_device(&Device::Cpu)
            })
        })?;

        let Ok(mut routing) = self.routing.lock() else {
            candle_core::bail!("SparseMoeBlock::routing mutex poisoned");
        };
        prof::timed(Span::MoeMisc, || routing.compute(topk_ids, num_experts))?;
        let xs_3d = xs_cpu.unsqueeze(1)?.contiguous()?;
        let gate_up_out = prof::timed(Span::MoeExpert, || {
            cpu_indexed_moe_forward(gate_up_exps, &xs_3d, &routing)
        })?;
        let gate = gate_up_out.narrow(D::Minus1, 0, intermediate_size)?;
        let up = gate_up_out.narrow(D::Minus1, intermediate_size, intermediate_size)?;
        let hidden = prof::timed(Span::MoeActivation, || {
            crate::ops::fused_ops::swiglu::swiglu(&gate, &up)
        })?;
        let down_out = prof::timed(Span::MoeExpert, || {
            cpu_indexed_moe_forward(down_exps, &hidden, &routing)
        })?;
        drop(routing);

        let combined = prof::timed(Span::MoeMisc, || {
            Self::combine_expert_outputs(
                &down_out,
                &topk_weights_cpu,
                original_dtype,
                original_dims,
            )
        })?;
        prof::timed(Span::MoeToDevice, || {
            prof::timed(Span::MoeCpuOutDev, || combined.to_device(input_device))
        })
    }

    /// GPU-offloaded equivalent of [`Self::cpu_batched_forward`], for a
    /// CPU-resident packed expert pair whose caller has decided the batch is
    /// large enough that a one-time upload to `device` is worth paying to run
    /// the batched matmul through [`Self::fused_forward`] instead of the CPU
    /// kernel.
    ///
    /// The uploaded `QTensor` copies (`gate_up_gpu`/`down_gpu`) are locals:
    /// they live only for this call and are dropped on return, so the
    /// transient VRAM they occupy (one layer's packed gate+up and down
    /// tensors) is freed immediately rather than held for the model's
    /// lifetime.
    ///
    /// `xs_f32`, `topk_ids`, and `topk_weights` may already live on `device`
    /// (the common case, when the rest of the model runs there and only
    /// `MoE` experts are CPU-offloaded) or on any other device; either way
    /// they're moved to `device` before dispatch, and the result is moved
    /// back to `xs_f32`'s original device before returning, matching
    /// `cpu_batched_forward`'s device contract.
    ///
    /// # Errors
    ///
    /// Returns an error if either upload, any device transfer, or
    /// `fused_forward` fails.
    #[allow(clippy::too_many_arguments)]
    fn gpu_offload_forward(
        xs_f32: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        gate_up_exps: &QTensor,
        down_exps: &QTensor,
        device: &Device,
        original_dtype: DType,
        original_dims: &[usize],
    ) -> Result<Tensor> {
        debug_assert!(
            gate_up_exps.device().is_cpu() && down_exps.device().is_cpu(),
            "gpu_offload_forward expects CPU-resident packed experts"
        );
        let input_device = xs_f32.device().clone();
        let (gate_up_gpu, down_gpu) = prof::timed(Span::MoeToDevice, || -> Result<_> {
            let gate_up_gpu = upload_qtensor(gate_up_exps, device)?;
            let down_gpu = upload_qtensor(down_exps, device)?;
            Ok((gate_up_gpu, down_gpu))
        })?;

        let xs_gpu = prof::timed(Span::MoeToDevice, || xs_f32.to_device(device))?;
        let topk_ids_gpu = prof::timed(Span::MoeToDevice, || topk_ids.to_device(device))?;
        let topk_weights_gpu = prof::timed(Span::MoeToDevice, || topk_weights.to_device(device))?;

        let out = prof::timed(Span::MoeFused, || {
            Self::fused_forward(
                &xs_gpu,
                &topk_ids_gpu,
                &topk_weights_gpu,
                &gate_up_gpu,
                &down_gpu,
                original_dtype,
                original_dims,
            )
        })?;

        prof::timed(Span::MoeToDevice, || out.to_device(&input_device))
    }

    /// Dispatches to the packed-tensor batched `MoE` path when this block
    /// holds packed tensors, returning `None` otherwise so the caller falls
    /// through to the per-expert loop. `load_packed_experts` only keeps
    /// both packed tensors (leaving `experts` empty) when the GPU fused
    /// path is eligible ([`supports_fused_moe`]) or `expert_device` is CPU,
    /// so this covers both. Three outcomes: GPU fused via
    /// [`Self::fused_forward`] when experts already live on GPU; GPU
    /// offload via [`Self::gpu_offload_forward`] when experts are
    /// CPU-resident but the batch is large enough
    /// ([`moe_offload_min_batch`]) and the input device/quant type support
    /// it; otherwise CPU batched via [`Self::cpu_batched_forward`].
    fn dispatch_packed(
        &self,
        xs_f32: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        original_dtype: DType,
        original_dims: &[usize],
    ) -> Option<Result<Tensor>> {
        let gate_up_exps = self.packed_gate_up_exps.as_ref()?;
        let down_exps = self.packed_down_exps.as_ref()?;
        if self.expert_device.is_cuda() || self.expert_device.is_rocm() {
            // Fused GPU dispatch (CUDA/ROCm only): `topk_ids`/
            // `topk_weights` stay on-device throughout, no CPU sync.
            return Some(prof::timed(Span::MoeFused, || {
                Self::fused_forward(
                    xs_f32,
                    topk_ids,
                    topk_weights,
                    gate_up_exps,
                    down_exps,
                    original_dtype,
                    original_dims,
                )
            }));
        }
        // GPU offload dispatch: CPU-resident experts, but the batch (e.g. a
        // prefill chunk) is large enough that a one-time upload to the
        // input's GPU device and a fused GPU matmul beats the CPU batched
        // kernel. Requires the input to already be on a `MoE`-eligible GPU
        // device (only reachable when the model's non-expert layers run on
        // CUDA/ROCm and just the experts were placed on CPU) and both packed
        // tensors' quant types to be `indexed_moe_forward`-eligible.
        let input_device = xs_f32.device();
        let tokens = xs_f32.dims()[0];
        if self.expert_device.is_cpu()
            && tokens >= moe_offload_min_batch()
            && (input_device.is_cuda() || input_device.is_rocm())
            && supports_fused_moe(input_device, gate_up_exps.dtype())
            && supports_fused_moe(input_device, down_exps.dtype())
        {
            return Some(Self::gpu_offload_forward(
                xs_f32,
                topk_ids,
                topk_weights,
                gate_up_exps,
                down_exps,
                input_device,
                original_dtype,
                original_dims,
            ));
        }
        // CPU batched dispatch: routing is pulled to host (unavoidable,
        // the kernel runs on CPU), but the per-expert loop's up to 336
        // individually-dispatched `to_device`/matmul calls collapse into 2
        // merged dispatches. Timed internally by `cpu_batched_forward`
        // itself (not wrapped in `Span::MoeFused` here), so its cost
        // breaks down into the same `to_dev`/`expert`/`misc` spans the
        // per-expert loop below uses.
        Some(self.cpu_batched_forward(
            xs_f32,
            topk_ids,
            topk_weights,
            gate_up_exps,
            down_exps,
            original_dtype,
            original_dims,
        ))
    }

    /// Describes this block's `MoE` dispatch *placement policy*: which
    /// device the packed experts live on, and (for CPU-resident packed
    /// experts) the per-call token threshold above which
    /// [`Self::dispatch_packed`] routes through [`Self::gpu_offload_forward`]
    /// instead of [`Self::cpu_batched_forward`]. The placement itself
    /// (`expert_device` and whether packed tensors are present) is fixed at
    /// construction and only changed by [`Self::promote_experts_to`], so
    /// it's safe to compute and log once rather than on every
    /// decode/prefill step -- but note the actual per-call outcome for
    /// CPU-resident packed experts still varies with `xs_f32`'s token
    /// count, which this label surfaces rather than resolves.
    fn dispatch_kind(&self) -> String {
        if self.packed_gate_up_exps.is_none() || self.packed_down_exps.is_none() {
            "per-expert loop".to_string()
        } else if self.expert_device.is_cuda() || self.expert_device.is_rocm() {
            "GPU fused".to_string()
        } else {
            format!(
                "CPU batched (GPU offload >= {} tokens)",
                moe_offload_min_batch()
            )
        }
    }

    /// Logs this block's `MoE` dispatch placement policy, identified by
    /// `layer_idx` so the decision for a specific decoder layer can be
    /// distinguished from the other layers' -- called once at construction
    /// and again whenever [`Self::promote_experts_to`] changes it, rather
    /// than per forward call, since the placement policy itself doesn't
    /// vary per-call (see [`Self::dispatch_kind`] for the per-call caveat
    /// on CPU-resident packed experts). Fires only at construction and on
    /// the rare re-promotion call, not per decode/prefill step, so `trace`
    /// is fine here.
    fn log_dispatch_decision(&self) {
        let dtype = self
            .packed_gate_up_exps
            .as_ref()
            .map_or_else(|| "n/a".to_string(), |t| format!("{:?}", t.dtype()));
        log::trace!(
            "MoE layer {}: dispatch={}, expert_device={:?}, dtype={dtype}",
            self.layer_idx,
            self.dispatch_kind(),
            self.expert_device,
        );
    }

    /// Weighted sum of per-expert outputs back to the original sequence shape.
    ///
    /// `expert_out` is `(tokens, top_k, hidden)`, `topk_weights` is
    /// `(tokens, top_k)`. Returns a tensor of shape `original_dims` in
    /// `original_dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor op fails.
    fn combine_expert_outputs(
        expert_out: &Tensor,
        topk_weights: &Tensor,
        original_dtype: DType,
        original_dims: &[usize],
    ) -> Result<Tensor> {
        let weights = topk_weights.unsqueeze(D::Minus1)?;
        let weighted = expert_out.broadcast_mul(&weights)?;
        let summed = weighted.sum(1)?;
        summed.to_dtype(original_dtype)?.reshape(original_dims)
    }
}

/// Reinterprets a byte slice as a slice of `T`, without copying.
///
/// `data` always originates from a `QTensor`'s CPU-resident storage (see
/// `QTensor::data`), which is backed by a `Vec<T>` -- so its pointer is
/// already aligned for `T` and its length is a whole number of `T`s. Both
/// are asserted rather than assumed.
fn as_quantized_slice<T: GgmlType>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "quantized data length {} is not a multiple of element size {size}",
        data.len()
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "quantized data pointer is not aligned for the target element type"
    );
    // SAFETY: length and alignment checked above; `data` is a read-only
    // borrow so the returned slice cannot outlive it or alias a mutable
    // reference.
    unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), data.len() / size) }
}

/// Validated shape dimensions extracted from `cpu_indexed_moe_forward`'s
/// inputs by [`validate_moe_forward_shapes`].
struct MoeForwardDims {
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    tokens: usize,
    input_dim1: usize,
    topk: usize,
}

/// Validates `packed_weights`/`xs` shapes against an already-known
/// `(tokens, topk)` pair, typically sourced from a [`MoeRouting`].
///
/// # Errors
///
/// Returns an error if `xs` is not F32; if `packed_weights` is not 3-D or
/// `xs` is not 3-D; if `xs`'s hidden dim does not match the packed weight
/// tensor's `in_dim`; if `xs` dim\[0\] does not match `tokens`; or if `xs`
/// dim\[1\] is neither `1` nor `topk`.
fn validate_moe_forward_dims(
    packed_weights: &QTensor,
    xs: &Tensor,
    tokens: usize,
    topk: usize,
) -> Result<MoeForwardDims> {
    if xs.dtype() != DType::F32 {
        candle_core::bail!(
            "cpu_indexed_moe_forward: xs must be F32, got {:?}",
            xs.dtype()
        );
    }

    let weight_dims = packed_weights.shape().dims();
    if weight_dims.len() != 3 {
        candle_core::bail!(
            "cpu_indexed_moe_forward: packed_weights must be 3-D \
             [num_experts, out_dim, in_dim], got {}-D {:?}",
            weight_dims.len(),
            weight_dims
        );
    }
    let (num_experts, out_dim, in_dim) = (weight_dims[0], weight_dims[1], weight_dims[2]);

    let xs_dims = xs.dims();
    if xs_dims.len() != 3 {
        candle_core::bail!(
            "cpu_indexed_moe_forward: xs must be 3-D [tokens, input_dim1, in_dim], \
             got {}-D {:?}",
            xs_dims.len(),
            xs_dims
        );
    }
    let (xs_tokens, input_dim1, xs_hidden) = (xs_dims[0], xs_dims[1], xs_dims[2]);
    if xs_hidden != in_dim {
        candle_core::bail!(
            "cpu_indexed_moe_forward: xs hidden dim {xs_hidden} does not match weight in_dim {in_dim}"
        );
    }
    if xs_tokens != tokens {
        candle_core::bail!(
            "cpu_indexed_moe_forward: xs dim[0] ({xs_tokens}) does not match \
             topk_ids dim[0] ({tokens})"
        );
    }
    if input_dim1 != 1 && input_dim1 != topk {
        candle_core::bail!(
            "cpu_indexed_moe_forward: xs dim[1] must be 1 (shared input) or \
             topk ({topk}, per-slot input), got {input_dim1}",
        );
    }

    Ok(MoeForwardDims {
        num_experts,
        out_dim,
        in_dim,
        tokens,
        input_dim1,
        topk,
    })
}

/// Per-expert lists of routed `(token, slot)` pairs, indexed by expert id.
type MoeTokenLists = Vec<Vec<(usize, usize)>>;

/// Pre-computed token-to-expert routing for [`cpu_indexed_moe_forward`].
///
/// Built once per `topk_ids` via [`MoeRouting::compute`] and shared across
/// both the `gate_up` and down projection calls of a decode step, since both
/// route the same `topk_ids` and would otherwise redo identical grouping
/// work. Its buffers keep their heap capacity across calls to `compute`, so
/// reusing one `MoeRouting` across steps avoids reallocating on the decode
/// hot path.
#[derive(Default)]
pub struct MoeRouting {
    token_lists: MoeTokenLists,
    active_experts: Vec<usize>,
    topk_ids_flat: Vec<u32>,
    /// Cumulative sum of `token_lists[e].len()` over `active_experts`, one
    /// longer than `active_experts` (a leading `0`). Shared by both
    /// `dispatch_moe_quads` calls of a decode step instead of being rebuilt
    /// per call.
    pair_prefix: Vec<usize>,
    num_experts: usize,
    tokens: usize,
    topk: usize,
}

impl MoeRouting {
    /// Creates an empty routing with no buffers allocated yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Groups `topk_ids` (`[tokens, topk]` `u32`) by routed expert.
    ///
    /// # Errors
    ///
    /// Returns an error if `topk_ids` is not 2-D, if any of its values is
    /// `>= num_experts`, or if any tensor op fails.
    pub fn compute(&mut self, topk_ids: &Tensor, num_experts: usize) -> Result<()> {
        if topk_ids.dims().len() != 2 {
            candle_core::bail!(
                "MoeRouting::compute: topk_ids must be 2-D [tokens, topk], got {}-D {:?}",
                topk_ids.dims().len(),
                topk_ids.dims()
            );
        }
        let (tokens, topk) = (topk_ids.dims()[0], topk_ids.dims()[1]);

        self.topk_ids_flat.clear();
        // `to_vec1` copies into a fresh Vec (candle has no zero-copy
        // accessor for a contiguous CPU tensor's raw slice); only
        // `topk_ids_flat`'s own capacity is reused across calls.
        self.topk_ids_flat
            .extend(topk_ids.flatten_all()?.to_vec1::<u32>()?);

        group_tokens_by_expert(
            &self.topk_ids_flat,
            num_experts,
            tokens,
            topk,
            &mut self.token_lists,
            &mut self.active_experts,
        )?;

        self.pair_prefix.clear();
        self.pair_prefix.push(0usize);
        let mut prev = 0usize;
        for &e in &self.active_experts {
            prev += self.token_lists[e].len();
            self.pair_prefix.push(prev);
        }

        self.num_experts = num_experts;
        self.tokens = tokens;
        self.topk = topk;
        Ok(())
    }
}

/// CPU-side batched `MoE` dispatch: the CPU equivalent of
/// `QTensor::indexed_moe_forward`, which is CUDA/`ROCm`-only. Replaces the
/// per-expert loop's serialized matmuls (one `BarrierPool` dispatch per
/// expert) with a single merged dispatch across every routed expert's
/// output columns.
///
/// `packed_weights` is a CPU-resident `[num_experts, out_dim, in_dim]`
/// `QTensor`. `xs` is F32 with shape `[tokens, input_dim1, in_dim]`:
/// `input_dim1` is `1` when every routed expert of a token shares the same
/// input row (the gate+up projection), or `topk` when each routed expert has
/// its own row (the down projection, fed the per-expert intermediate
/// activation). `routing` must already be computed (via
/// [`MoeRouting::compute`]) for `packed_weights`' expert count; callers
/// doing both projections for one decode step should compute it once and
/// pass it to both calls.
///
/// Returns an F32 tensor of shape `[tokens, topk, out_dim]`.
///
/// # Errors
///
/// Returns an error if `xs` is not F32; if `packed_weights` is not 3-D or
/// `xs` is not 3-D; if `xs` dim\[1\] is neither `1` nor `topk`; if `xs`'s
/// hidden dim does not match the packed weight tensor's `in_dim`; if `xs`
/// dim\[0\] does not match `routing`'s token count; if `routing` was computed
/// for a different expert count than `packed_weights` has; if `packed_weights`
/// or `xs` is not CPU-resident; if `in_dim` is not a multiple of the
/// quantization block size; if the packed weight data length is inconsistent
/// with the computed per-expert byte count; or if any tensor op fails.
///
/// # Panics
///
/// May panic if the packed weight tensor's raw bytes are not correctly
/// aligned for the underlying quantization block type, or if the byte
/// count per expert is not an exact multiple of the block's in-memory
/// size. These are internal invariants of `QTensor` construction and
/// should not be triggered by well-formed inputs.
pub fn cpu_indexed_moe_forward(
    packed_weights: &QTensor,
    xs: &Tensor,
    routing: &MoeRouting,
) -> Result<Tensor> {
    let dims = validate_moe_forward_dims(packed_weights, xs, routing.tokens, routing.topk)?;
    if dims.num_experts != routing.num_experts {
        candle_core::bail!(
            "cpu_indexed_moe_forward: routing was computed for {} experts, \
             but packed_weights has {} experts",
            routing.num_experts,
            dims.num_experts
        );
    }
    if !packed_weights.device().is_cpu() {
        candle_core::bail!(
            "cpu_indexed_moe_forward: packed_weights must be CPU-resident, got {:?}",
            packed_weights.device()
        );
    }
    if !xs.device().is_cpu() {
        candle_core::bail!(
            "cpu_indexed_moe_forward: xs must be CPU-resident, got {:?}",
            xs.device()
        );
    }

    // Small per-step copy (decode-step `xs` is `[tokens, 1|topk, in_dim]`,
    // typically a single token); candle has no zero-copy accessor for a
    // contiguous CPU tensor's raw `f32` slice.
    let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;

    let dtype = packed_weights.dtype();
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    if !dims.in_dim.is_multiple_of(block_size) {
        candle_core::bail!(
            "cpu_indexed_moe_forward: in_dim {} is not a multiple of block size {block_size}",
            dims.in_dim,
        );
    }
    let bytes_per_expert = dims.out_dim * dims.in_dim / block_size * type_size;
    let k_in_blocks = dims.in_dim.div_ceil(block_size);

    let weight_data = packed_weights.data()?;
    if weight_data.len() != bytes_per_expert * dims.num_experts {
        candle_core::bail!(
            "cpu_indexed_moe_forward: packed weight data length {} does not match \
             expected {} ({} experts x {bytes_per_expert} bytes)",
            weight_data.len(),
            bytes_per_expert * dims.num_experts,
            dims.num_experts,
        );
    }

    macro_rules! dispatch {
        ($block:ty) => {
            batched_moe_matmul::<$block>(
                &weight_data,
                bytes_per_expert,
                k_in_blocks,
                &xs_flat,
                routing,
                &dims,
            )
        };
    }
    let dst = match dtype {
        GgmlDType::F32 => dispatch!(f32),
        GgmlDType::F16 => dispatch!(f16),
        GgmlDType::BF16 => dispatch!(bf16),
        GgmlDType::Q4_0 => dispatch!(BlockQ4_0),
        GgmlDType::Q4_1 => dispatch!(BlockQ4_1),
        GgmlDType::Q5_0 => dispatch!(BlockQ5_0),
        GgmlDType::Q5_1 => dispatch!(BlockQ5_1),
        GgmlDType::Q8_0 => dispatch!(BlockQ8_0),
        GgmlDType::Q8_1 => dispatch!(BlockQ8_1),
        GgmlDType::Q2K => dispatch!(BlockQ2K),
        GgmlDType::Q3K => dispatch!(BlockQ3K),
        GgmlDType::Q4K => dispatch!(BlockQ4K),
        GgmlDType::Q5K => dispatch!(BlockQ5K),
        GgmlDType::Q6K => dispatch!(BlockQ6K),
        GgmlDType::Q8K => dispatch!(BlockQ8K),
    };

    Tensor::from_vec(dst, (dims.tokens, dims.topk, dims.out_dim), &Device::Cpu)
}

/// Groups every `(token, slot)` pair by its routed expert into `token_lists`
/// (indexed by expert id, empty for unrouted experts) and `active_experts`
/// (the sorted list of experts with at least one routed pair). Both are
/// cleared and refilled in place, reusing their existing heap capacity
/// across calls instead of reallocating.
///
/// # Errors
///
/// Returns an error if any `topk_ids` value is `>= num_experts`.
fn group_tokens_by_expert(
    topk_ids: &[u32],
    num_experts: usize,
    tokens: usize,
    topk: usize,
    token_lists: &mut MoeTokenLists,
    active_experts: &mut Vec<usize>,
) -> Result<()> {
    if token_lists.len() != num_experts {
        token_lists.resize_with(num_experts, Vec::new);
    }
    for list in token_lists.iter_mut() {
        list.clear();
    }
    for t in 0..tokens {
        for s in 0..topk {
            let expert_idx = topk_ids[t * topk + s] as usize;
            if expert_idx >= num_experts {
                candle_core::bail!(
                    "cpu_indexed_moe_forward: topk_ids[{t}, {s}] = {expert_idx} \
                     exceeds num_experts ({num_experts})",
                );
            }
            token_lists[expert_idx].push((t, s));
        }
    }
    active_experts.clear();
    active_experts.extend((0..num_experts).filter(|&e| !token_lists[e].is_empty()));
    Ok(())
}

/// Quantizes every unique input row into `T::VecDotType` blocks once, into
/// `scratch` (resized as needed), so experts sharing an input row (relevant
/// when `input_dim1 == 1`, i.e. the gate+up projection) reuse the same
/// quantized copy instead of re-quantizing per routed expert.
///
/// Rows are independent, so the block-quantized path (`DIRECT_COPY ==
/// false`, i.e. every K-quant weight type) splits them across
/// [`barrier_pool`] the same way [`dispatch_moe_quads`] splits output
/// columns. Left serial, this loop runs on one thread while every other
/// pool worker sits idle waiting for the matmul dispatch that follows it,
/// which is what caps overall CPU utilization well under 100% despite the
/// matmul itself being fully parallel.
fn quantize_lhs_rows<'a, T: GgmlType>(
    xs: &[f32],
    num_rows: usize,
    in_dim: usize,
    k_in_blocks: usize,
    scratch: &'a mut Vec<u64>,
) -> &'a [T::VecDotType] {
    let elem_size = std::mem::size_of::<T::VecDotType>();
    let scratch_len = (num_rows * k_in_blocks * elem_size).div_ceil(8);
    if scratch.len() < scratch_len {
        scratch.resize(scratch_len, 0);
    }
    let lhs_ptr = scratch.as_mut_ptr().cast::<T::VecDotType>() as usize;
    if T::DIRECT_COPY {
        // SAFETY: u64 alignment (8 bytes) covers every `VecDotType` block
        // type; the buffer holds exactly `num_rows * k_in_blocks` elements.
        let lhs_b: &'a mut [T::VecDotType] = unsafe {
            std::slice::from_raw_parts_mut(lhs_ptr as *mut T::VecDotType, num_rows * k_in_blocks)
        };
        T::VecDotType::direct_copy(xs, lhs_b);
    } else if num_rows > 0 {
        let pool = barrier_pool();
        let n_total = pool.n_workers() + 1;
        let rows_per_thread = num_rows.div_ceil(n_total);
        pool.execute(|tid| {
            let start = tid * rows_per_thread;
            if start >= num_rows {
                return;
            }
            let end = num_rows.min(start + rows_per_thread);
            // SAFETY: each thread only ever touches rows in its own
            // exclusively-owned `start..end` range, computed from `tid`
            // the same way `dispatch_moe_quads` partitions its flat index
            // space, so no two threads ever write the same row.
            let lhs_b: &mut [T::VecDotType] = unsafe {
                std::slice::from_raw_parts_mut(
                    lhs_ptr as *mut T::VecDotType,
                    num_rows * k_in_blocks,
                )
            };
            for row_idx in start..end {
                let src = &xs[row_idx * in_dim..(row_idx + 1) * in_dim];
                let dst_row = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
                T::VecDotType::from_float(src, dst_row);
            }
        });
    }
    // SAFETY: every row in `0..num_rows` has now been written, either by
    // `direct_copy` or by the parallel loop above.
    unsafe { std::slice::from_raw_parts(lhs_ptr as *const T::VecDotType, num_rows * k_in_blocks) }
}

/// Shared read-only context for the batched `MoE` matmul dispatch
/// functions. Bundles the parameters that [`dispatch_moe_quads`] and
/// [`moe_tail_columns`] both need, so those functions don't need a long
/// individually-listed argument list.
struct MoeMatmulCtx<'a> {
    weight_data: &'a [u8],
    bytes_per_expert: usize,
    k_in_blocks: usize,
    token_lists: &'a [Vec<(usize, usize)>],
    active_experts: &'a [usize],
    /// Precomputed by [`MoeRouting::compute`]; see that field's doc comment.
    pair_prefix: &'a [usize],
    input_dim1: usize,
    topk: usize,
    in_dim: usize,
    out_dim: usize,
}

/// Single static-merged [`barrier_pool`] dispatch across every routed
/// `(token, slot)` pair's output-column quads (`out_dim & !3` columns, in
/// groups of 4).
///
/// The flat work space is `total_routed_pairs * quads_per_expert`, where
/// `total_routed_pairs` is `tokens * topk` -- fixed regardless of how
/// routing is distributed across experts, since every routed pair belongs
/// to exactly one expert. Splitting this space evenly across threads
/// therefore gives every thread the same amount of real `vec_dot_4` work
/// no matter how skewed real-world routing is (a handful of "hot" experts
/// taking far more tokens than the rest, which is the common case, not
/// the exception, once routing comes from a real trained gate rather than
/// a synthetic uniform distribution). Partitioning by
/// `active_experts.len() * quads_per_expert` instead would implicitly
/// treat every active expert as equal-sized work regardless of how many
/// tokens it actually received, leaving some threads with far more real
/// work than others under skewed routing while the rest sat idle.
fn dispatch_moe_quads<T: GgmlType>(
    ctx: &MoeMatmulCtx<'_>,
    lhs_b: &[T::VecDotType],
    quads_per_expert: usize,
    dst: &mut [f32],
) {
    if ctx.active_experts.is_empty() || quads_per_expert == 0 {
        return;
    }
    let pair_prefix = ctx.pair_prefix;
    let total_pairs = pair_prefix[ctx.active_experts.len()];
    let total_items = total_pairs * quads_per_expert;
    if total_items == 0 {
        return;
    }
    let pool = barrier_pool();
    let n_total = pool.n_workers() + 1;
    let items_per_thread = total_items.div_ceil(n_total);
    let dst_ptr = dst.as_mut_ptr() as usize;

    pool.execute(|tid| {
        let start = tid * items_per_thread;
        if start >= total_items {
            return;
        }
        let end = total_items.min((tid + 1) * items_per_thread);
        let dst_ptr = dst_ptr as *mut f32;

        // Find the active-expert segment containing this thread's first
        // pair once via binary search; subsequent iterations only ever
        // advance forward through `pair_prefix` (checked below), since
        // `pair_idx` increases monotonically within one thread's range.
        // `start_pair < total_pairs` always holds here (`start <
        // total_items` was checked above), so `Ok(i)` can only land on one
        // of the first `active_experts.len()` prefix entries, never the
        // final (`total_pairs`) one.
        let start_pair = start / quads_per_expert;
        let mut active_idx = match pair_prefix.binary_search(&start_pair) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let mut expert_idx = ctx.active_experts[active_idx];
        let mut expert_rhs: &[T] = as_quantized_slice(
            &ctx.weight_data
                [expert_idx * ctx.bytes_per_expert..(expert_idx + 1) * ctx.bytes_per_expert],
        );

        for flat_idx in start..end {
            let pair_idx = flat_idx / quads_per_expert;
            let quad_idx = flat_idx % quads_per_expert;
            let col = quad_idx * 4;

            while pair_idx >= pair_prefix[active_idx + 1] {
                active_idx += 1;
                expert_idx = ctx.active_experts[active_idx];
                expert_rhs = as_quantized_slice(
                    &ctx.weight_data[expert_idx * ctx.bytes_per_expert
                        ..(expert_idx + 1) * ctx.bytes_per_expert],
                );
            }
            let local_pair = pair_idx - pair_prefix[active_idx];
            let (t, s) = ctx.token_lists[expert_idx][local_pair];

            let w0 = &expert_rhs[col * ctx.k_in_blocks..(col + 1) * ctx.k_in_blocks];
            let w1 = &expert_rhs[(col + 1) * ctx.k_in_blocks..(col + 2) * ctx.k_in_blocks];
            let w2 = &expert_rhs[(col + 2) * ctx.k_in_blocks..(col + 3) * ctx.k_in_blocks];
            let w3 = &expert_rhs[(col + 3) * ctx.k_in_blocks..(col + 4) * ctx.k_in_blocks];

            let lhs_row_idx = if ctx.input_dim1 == 1 {
                t
            } else {
                t * ctx.topk + s
            };
            let lhs_row =
                &lhs_b[lhs_row_idx * ctx.k_in_blocks..(lhs_row_idx + 1) * ctx.k_in_blocks];
            let (d0, d1, d2, d3) = T::vec_dot_4(ctx.in_dim, w0, w1, w2, w3, lhs_row);

            let out_offset = (t * ctx.topk + s) * ctx.out_dim + col;
            // SAFETY: each (t, s, col) triple is written by exactly one
            // thread. (t, s) is routed to exactly one expert (this loop's
            // `expert_idx`), and `col` falls in this thread's
            // exclusively-owned `flat_idx` range, so no other thread's
            // iteration ever targets this `out_offset`.
            unsafe {
                let base = dst_ptr.add(out_offset);
                *base = d0;
                *base.add(1) = d1;
                *base.add(2) = d2;
                *base.add(3) = d3;
            }
        }
    });
}

/// Handles the 0..=3 output columns left over when `out_dim` isn't a
/// multiple of 4, sequentially on the calling thread -- same
/// `vec_dot_2`/`vec_dot` split as `k_quants::matmul`'s tail handling.
fn moe_tail_columns<T: GgmlType>(
    ctx: &MoeMatmulCtx<'_>,
    lhs_b: &[T::VecDotType],
    n_quad: usize,
    dst: &mut [f32],
) {
    let n_tail = ctx.out_dim - n_quad;
    if n_tail == 0 {
        return;
    }
    for &expert_idx in ctx.active_experts {
        let expert_bytes = &ctx.weight_data
            [expert_idx * ctx.bytes_per_expert..(expert_idx + 1) * ctx.bytes_per_expert];
        let expert_rhs: &[T] = as_quantized_slice(expert_bytes);
        for &(t, s) in &ctx.token_lists[expert_idx] {
            let lhs_row_idx = if ctx.input_dim1 == 1 {
                t
            } else {
                t * ctx.topk + s
            };
            let lhs_row =
                &lhs_b[lhs_row_idx * ctx.k_in_blocks..(lhs_row_idx + 1) * ctx.k_in_blocks];
            let out_offset = (t * ctx.topk + s) * ctx.out_dim + n_quad;
            if n_tail >= 2 {
                let col = n_quad;
                let w0 = &expert_rhs[col * ctx.k_in_blocks..(col + 1) * ctx.k_in_blocks];
                let w1 = &expert_rhs[(col + 1) * ctx.k_in_blocks..(col + 2) * ctx.k_in_blocks];
                let (d0, d1) = T::vec_dot_2(ctx.in_dim, w0, w1, lhs_row);
                dst[out_offset] = d0;
                dst[out_offset + 1] = d1;
            }
            if n_tail & 1 == 1 {
                let col = ctx.out_dim - 1;
                let w = &expert_rhs[col * ctx.k_in_blocks..(col + 1) * ctx.k_in_blocks];
                dst[out_offset + n_tail - 1] = T::vec_dot(ctx.in_dim, w, lhs_row);
            }
        }
    }
}

// Thread-local scratch buffer reused across calls to avoid per-matmul heap
// allocation of the quantized LHS. Mirrors `k_quants::matmul`'s pattern.
// Using u64 ensures sufficient alignment regardless of `T::VecDotType`.
thread_local! {
    static MOE_LHS_SCRATCH: std::cell::RefCell<Vec<u64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Core batched matmul for one projection across every routed expert,
/// dispatched from [`cpu_indexed_moe_forward`] after `GgmlDType` matching.
///
/// Mirrors `k_quants::matmul`'s LHS-quantization and `vec_dot_4` pattern,
/// but flattens every active expert's output-column quads into a single
/// index space and issues one [`barrier_pool`] dispatch for the whole
/// projection, instead of one dispatch per expert. LHS rows are quantized
/// once and shared across every expert a row is routed to (relevant when
/// `input_dim1 == 1`, i.e. the gate+up projection, where all `topk` routed
/// experts for a token share the same input row). `routing` supplies the
/// token-to-expert grouping, already computed by the caller.
fn batched_moe_matmul<T: GgmlType>(
    weight_data: &[u8],
    bytes_per_expert: usize,
    k_in_blocks: usize,
    xs: &[f32],
    routing: &MoeRouting,
    dims: &MoeForwardDims,
) -> Vec<f32> {
    let tokens = routing.tokens;
    let topk = routing.topk;
    let (input_dim1, in_dim, out_dim) = (dims.input_dim1, dims.in_dim, dims.out_dim);

    let num_rows = if input_dim1 == 1 {
        tokens
    } else {
        tokens * topk
    };

    // Output is zero-initialized so unrouted (token, slot) pairs stay zero.
    let mut dst = vec![0.0f32; tokens * topk * out_dim];
    let n_quad = out_dim & !3;
    let quads_per_expert = n_quad / 4;

    MOE_LHS_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let lhs_b = quantize_lhs_rows::<T>(xs, num_rows, in_dim, k_in_blocks, &mut scratch);

        let ctx = MoeMatmulCtx {
            weight_data,
            bytes_per_expert,
            k_in_blocks,
            token_lists: &routing.token_lists,
            active_experts: &routing.active_experts,
            pair_prefix: &routing.pair_prefix,
            input_dim1,
            topk,
            in_dim,
            out_dim,
        };

        dispatch_moe_quads::<T>(&ctx, lhs_b, quads_per_expert, &mut dst);
        moe_tail_columns::<T>(&ctx, lhs_b, n_quad, &mut dst);
    });

    dst
}

/// Minimum token count for [`SparseMoeBlock::dispatch_packed`] to route
/// CPU-resident packed experts through [`SparseMoeBlock::gpu_offload_forward`]
/// instead of [`SparseMoeBlock::cpu_batched_forward`]. Configurable via
/// `CRANE_MOE_OFFLOAD_MIN_BATCH`; defaults to 32, matching the batch size at
/// which GPU matmul throughput starts to outweigh the one-time `PCIe`
/// upload cost of the packed expert tensors. Zero, negative, or unparseable
/// values silently fall back to the default rather than disabling the
/// threshold.
fn moe_offload_min_batch() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| parse_min_batch(std::env::var("CRANE_MOE_OFFLOAD_MIN_BATCH").ok().as_deref()))
}

/// Parses [`moe_offload_min_batch`]'s env var value, defaulting to 32 when
/// `raw` is absent, unparseable, or not a positive integer. Split out from
/// `moe_offload_min_batch` so this pure logic is unit-testable without
/// mutating process env state against a `OnceLock`-cached value.
fn parse_min_batch(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(32)
}

/// Whether `QTensor::indexed_moe_forward` (Phase 8a's fused `MoE` dispatch) is
/// available for `device` and `ggml_dtype`. `indexed_moe_forward` is
/// implemented on CUDA and `ROCm` only (see the xmiksay/candle fork's
/// `quantized/{cuda,rocm}.rs`). It only supports `Q2K`-`Q6K` and `Q8_0`.
/// Those are the same dtypes MMVQ supports, since the kernel reuses its
/// `vec_dot_q*_q8_1` inner loop.
fn supports_fused_moe(device: &Device, ggml_dtype: GgmlDType) -> bool {
    (device.is_cuda() || device.is_rocm())
        && matches!(
            ggml_dtype,
            GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8_0
        )
}

/// Uploads a packed `QTensor`'s raw quantized bytes to `device`, without
/// dequantizing. Shared by [`SparseMoeBlock::promote_experts_to`]'s
/// fused-eligible branch and [`SparseMoeBlock::gpu_offload_forward`], which
/// both need to move a whole packed gate+up or down tensor onto a GPU
/// device.
///
/// # Errors
///
/// Returns an error if fetching `src`'s raw bytes or `qtensor_from_ggml`
/// fails.
fn upload_qtensor(src: &QTensor, device: &Device) -> Result<QTensor> {
    let raw = src.data()?;
    qtensor_from_ggml(src.dtype(), &raw, src.shape().dims().to_vec(), device)
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

/// Byte-slices a packed `(gate_up, down)` `QTensor` pair's raw data into
/// per-expert [`MoeExpert`]s, without dequantizing. Shared by
/// [`SparseMoeBlock::load_packed_experts`]'s and
/// [`SparseMoeBlock::promote_experts_to`]'s non-fused-eligible fallback,
/// which both need to unpack the same packed layout into a per-expert loop.
///
/// # Errors
///
/// Returns an error if any [`slice_packed_qtensor`] call fails.
#[allow(clippy::too_many_arguments)]
fn unpack_experts_from_packed(
    gate_up_raw: &[u8],
    gate_up_dtype: GgmlDType,
    gate_up_out: usize,
    gate_up_in: usize,
    down_raw: &[u8],
    down_dtype: GgmlDType,
    down_out: usize,
    down_in: usize,
    num_experts: usize,
    device: &Device,
) -> Result<Vec<MoeExpert>> {
    (0..num_experts)
        .map(|i| {
            let gate_up_qt = slice_packed_qtensor(
                gate_up_raw,
                gate_up_dtype,
                i,
                gate_up_out,
                gate_up_in,
                device,
            )?;
            let down_qt = slice_packed_qtensor(down_raw, down_dtype, i, down_out, down_in, device)?;
            Ok(MoeExpert::from_layers(
                LinearLayer::quantized(QMatMul::from_arc(Arc::new(gate_up_qt))?),
                LinearLayer::quantized(QMatMul::from_arc(Arc::new(down_qt))?),
                gate_up_out / 2,
            ))
        })
        .collect::<Result<Vec<_>>>()
}

/// Fuses two 2D `[rows, cols]` `QTensor`s (e.g. one expert's gate and up
/// projections) into a single `[2*rows, cols]` `QTensor` by concatenating
/// their raw quantized bytes, without dequantizing. Valid because each
/// tensor's rows are laid out as whole quantization blocks: appending `b`'s
/// bytes after `a`'s produces exactly the byte layout `qtensor_from_ggml`
/// expects for a `[a_rows + b_rows, cols]` tensor.
///
/// # Errors
///
/// Returns an error if `a` and `b` have different quant dtypes or column
/// counts, or if `qtensor_from_ggml` fails.
fn fuse_quantized_qtensors(a: &QTensor, b: &QTensor, device: &Device) -> Result<QTensor> {
    if a.dtype() != b.dtype() {
        candle_core::bail!(
            "fuse_quantized_qtensors: mismatched dtypes {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        );
    }
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims[1] != b_dims[1] {
        candle_core::bail!(
            "fuse_quantized_qtensors: mismatched column counts {} vs {}",
            a_dims[1],
            b_dims[1]
        );
    }
    let a_raw = a.data()?;
    let b_raw = b.data()?;
    let mut fused = Vec::with_capacity(a_raw.len() + b_raw.len());
    fused.extend_from_slice(&a_raw);
    fused.extend_from_slice(&b_raw);
    qtensor_from_ggml(
        a.dtype(),
        &fused,
        vec![a_dims[0] + b_dims[0], a_dims[1]],
        device,
    )
}

/// Fuses two packed 3D `[num_experts, n, k]` `QTensor`s (e.g. `gate_exps`
/// and `up_exps`) into a single `[num_experts, 2*n, k]` `QTensor`, without
/// dequantizing.
///
/// Unlike [`fuse_quantized_qtensors`], the two source tensors' bytes can't
/// simply be concatenated end-to-end: that would order all of `a`'s experts
/// before all of `b`'s, giving a `[2*num_experts, n, k]` layout instead of
/// the `[num_experts, 2*n, k]` one `indexed_moe_forward`'s per-expert stride
/// arithmetic expects. Instead, each expert's bytes are interleaved: `a`'s
/// slice for expert `i` followed by `b`'s slice for expert `i`, for every
/// `i`.
///
/// # Errors
///
/// Returns an error if `a` and `b` have different quant dtypes, expert
/// counts, or `k` dimensions; if either tensor's per-expert element count
/// isn't a whole number of quantization blocks; or if `qtensor_from_ggml`
/// fails.
fn fuse_packed_qtensors(a: &QTensor, b: &QTensor, device: &Device) -> Result<QTensor> {
    if a.dtype() != b.dtype() {
        candle_core::bail!(
            "fuse_packed_qtensors: mismatched dtypes {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        );
    }
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims[0] != b_dims[0] || a_dims[2] != b_dims[2] {
        candle_core::bail!("fuse_packed_qtensors: shape mismatch {a_dims:?} vs {b_dims:?}");
    }
    let num_experts = a_dims[0];
    let block_size = a.dtype().block_size();
    let a_expert_elems = a_dims[1] * a_dims[2];
    let b_expert_elems = b_dims[1] * b_dims[2];
    if !a_expert_elems.is_multiple_of(block_size) || !b_expert_elems.is_multiple_of(block_size) {
        candle_core::bail!(
            "fuse_packed_qtensors: expert element counts {a_expert_elems}/{b_expert_elems} \
             not divisible by block size {block_size}"
        );
    }
    let a_raw = a.data()?;
    let b_raw = b.data()?;
    if !a_raw.len().is_multiple_of(num_experts) || !b_raw.len().is_multiple_of(num_experts) {
        candle_core::bail!(
            "fuse_packed_qtensors: raw byte lengths {}/{} not divisible by expert count {num_experts}",
            a_raw.len(),
            b_raw.len(),
        );
    }
    let bytes_per_expert_a = a_raw.len() / num_experts;
    let bytes_per_expert_b = b_raw.len() / num_experts;
    let mut fused = Vec::with_capacity(a_raw.len() + b_raw.len());
    for i in 0..num_experts {
        fused.extend_from_slice(&a_raw[i * bytes_per_expert_a..(i + 1) * bytes_per_expert_a]);
        fused.extend_from_slice(&b_raw[i * bytes_per_expert_b..(i + 1) * bytes_per_expert_b]);
    }
    qtensor_from_ggml(
        a.dtype(),
        &fused,
        vec![num_experts, a_dims[1] + b_dims[1], a_dims[2]],
        device,
    )
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
/// transferred with one device allocation per projection (2 total)
/// instead of one per expert projection (`2 * experts.len()`).
///
/// Exists because transferring `2 * num_experts` small tensors
/// individually (`SparseMoeBlock::promote_experts_to`'s per-expert
/// fallback loop above) measured a fixed ~1MB overhead per allocation on
/// `ROCm` — a significant amount of pure overhead on top of the tensors'
/// actual combined size for a large-expert-count layer. Returns `Ok(None)`
/// (fall back to the per-expert loop) if any projection isn't stackable,
/// i.e. the checkpoint uses the per-expert (still-quantized) GGUF layout
/// rather than the packed one, or if `experts` is empty.
fn batched_promote(
    experts: &[MoeExpert],
    device: &Device,
    dtype: DType,
) -> Result<Option<Vec<MoeExpert>>> {
    if experts.is_empty() {
        return Ok(None);
    }
    let Some(gate_up_all) = stack_projection(experts, device, dtype, |e| &e.gate_up_proj)? else {
        return Ok(None);
    };
    let Some(down_all) = stack_projection(experts, device, dtype, |e| &e.down_proj)? else {
        return Ok(None);
    };
    if let Some((free, mem_total)) = query_gpu_memory(device) {
        log::debug!(
            "batched promotion to {:?}: {} experts transferred in 2 allocations, free={}, total={}",
            device.location(),
            experts.len(),
            format_budget(free),
            format_budget(mem_total),
        );
    }
    let intermediate_size = experts[0].intermediate_size;
    (0..experts.len())
        .map(|i| {
            Ok(MoeExpert::from_layers(
                LinearLayer::Standard(Linear::new(narrow_packed_expert(&gate_up_all, i)?, None)),
                LinearLayer::Standard(Linear::new(narrow_packed_expert(&down_all, i)?, None)),
                intermediate_size,
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
        let (xs_flat, xs_f32) = prof::timed(Span::MoeMisc, || -> Result<(Tensor, Tensor)> {
            let xs_flat = xs.reshape(((), hidden_size))?;
            let xs_f32 = xs_flat.to_dtype(DType::F32)?;
            Ok((xs_flat, xs_f32))
        })?;

        let (topk_ids, topk_weights) =
            prof::timed(Span::MoeRouter, || -> Result<(Tensor, Tensor)> {
                let logits = self.gate.forward(&xs_f32)?;
                crate::ops::fused_ops::topk_moe::topk_moe_routing(
                    &logits,
                    self.num_experts_per_tok,
                    self.norm_topk_prob,
                )
            })?;

        // Batched packed-tensor MoE dispatch (GPU fused or CPU batched): 2
        // dispatches (gate+up, down) instead of the per-expert loop below.
        // See `dispatch_packed`'s doc comment for when this applies.
        if let Some(result) = self.dispatch_packed(
            &xs_f32,
            &topk_ids,
            &topk_weights,
            original_dtype,
            &original_dims,
        ) {
            return result;
        }

        // Routing dispatch is CPU-side: topk indices and weights are pulled to
        // the host each forward call, and per-expert token/weight lists below
        // are heap-allocated fresh each call. This is the fallback path for
        // CPU, Metal, and unsupported quant types; the fused path above
        // avoids both the sync and the allocations on CUDA/ROCm.
        let (token_lists, weight_lists) = prof::timed(Span::MoeMisc, || -> Result<_> {
            let topk_ids = topk_ids.to_vec2::<u32>()?;
            let topk_weights = topk_weights.to_vec2::<f32>()?;

            let mut token_lists: Vec<Vec<u32>> = vec![Vec::new(); self.experts.len()];
            let mut weight_lists: Vec<Vec<f32>> = vec![Vec::new(); self.experts.len()];
            for (token_idx, (ids, weights)) in topk_ids.iter().zip(topk_weights.iter()).enumerate()
            {
                // Token counts (batch * seq_len) never approach u32::MAX.
                #[allow(clippy::cast_possible_truncation)]
                let token_idx = token_idx as u32;
                for (&expert_idx, &weight) in ids.iter().zip(weights.iter()) {
                    token_lists[expert_idx as usize].push(token_idx);
                    weight_lists[expert_idx as usize].push(weight);
                }
            }
            Ok((token_lists, weight_lists))
        })?;

        // `Device::Cpu` is a unit variant, so this cross-device branch is only
        // exercised (and only exercisable in tests) on multi-device hardware.
        // All unit tests here run router and experts on the same CPU device.
        let same_device = xs_flat.device().location() == self.expert_device.location();
        let input_device = xs_flat.device();

        // For a cross-device expert (CPU-resident weights, GPU input, or
        // vice versa), only the tiny per-expert activation slice
        // (`selected` -- 1-8 rows during decode) moves to `expert_device`
        // and back; the matmul itself runs where the weights already
        // live, rather than copying the full weight tensors to
        // `input_device` every call.
        //
        // `MoeToDevice` and `MoeExpert` are timed as separate, non-nested
        // spans per expert (rather than one span wrapping the whole loop):
        // both are Tier2b siblings, and `prof::timed` never subtracts a
        // span's children, so nesting one inside the other would double-
        // count time into the wrong bucket.
        //
        // Accumulation happens in F32 regardless of `original_dtype`, with a
        // single cast back to `original_dtype` once after the loop instead
        // of per-expert. Quantized experts route through `forward_f32`
        // (their input is already `xs_f32`, so no per-expert cast is
        // needed either way it's dispatched) since `QMatMul` computes in F32
        // internally and `forward_f32` skips the intermediate cast back to
        // `original_dtype` that `LinearLayer::forward` would otherwise do.
        // Standard (unquantized) experts stay on `forward` in their native
        // dtype and only the small output gets cast to F32: `forward_f32`
        // would instead recast the *entire weight matrix* to F32 on every
        // call, which is far more expensive than the round-trip it's meant
        // to avoid.
        let quantized_experts = self.experts.first().is_some_and(MoeExpert::is_quantized);
        let xs_input = if quantized_experts { &xs_f32 } else { &xs_flat };
        let mut output = Tensor::zeros(xs_flat.dims(), DType::F32, input_device)?;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let tokens = &token_lists[expert_idx];
            if tokens.is_empty() {
                continue;
            }
            let (token_ids, selected) =
                prof::timed(Span::MoeMisc, || -> Result<(Tensor, Tensor)> {
                    let token_ids = Tensor::new(tokens.as_slice(), input_device)?;
                    let selected = xs_input.index_select(&token_ids, 0)?;
                    Ok((token_ids, selected))
                })?;
            let expert_out = if same_device {
                prof::timed(Span::MoeExpert, || -> Result<Tensor> {
                    if quantized_experts {
                        expert.forward_f32(&selected)
                    } else {
                        expert.forward(&selected)?.to_dtype(DType::F32)
                    }
                })?
            } else {
                let expert_in = prof::timed(Span::MoeToDevice, || {
                    selected.to_device(&self.expert_device)
                })?;
                let expert_out = prof::timed(Span::MoeExpert, || -> Result<Tensor> {
                    if quantized_experts {
                        expert.forward_f32(&expert_in)
                    } else {
                        expert.forward(&expert_in)?.to_dtype(DType::F32)
                    }
                })?;
                prof::timed(Span::MoeToDevice, || expert_out.to_device(input_device))?
            };
            output = prof::timed(Span::MoeExpert, || -> Result<Tensor> {
                let weights = Tensor::new(weight_lists[expert_idx].as_slice(), input_device)?
                    .reshape((tokens.len(), 1))?;
                let scaled = expert_out.broadcast_mul(&weights)?;
                output.index_add(&token_ids, &scaled, 0)
            })?;
        }
        prof::timed(Span::MoeMisc, || -> Result<Tensor> {
            let output = if output.dtype() == original_dtype {
                output
            } else {
                output.to_dtype(original_dtype)?
            };
            output.reshape(original_dims)
        })
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

    // Verifies `unpack_experts_from_packed` (the per-expert byte-slicing
    // helper shared by `load_packed_experts` and `promote_experts_to`'s
    // non-fused-eligible fallback) produces `MoeExpert`s whose forward
    // output matches the closed-form expected value for constant-per-expert
    // weights. `promote_experts_to`'s packed-tensor slicing branch cannot be
    // exercised directly without a second real device, so this gives it
    // indirect coverage by testing the shared slicing logic in isolation.
    #[test]
    fn unpack_experts_from_packed_matches_per_expert() {
        use candle_core::quantized::GgmlDType;

        let device = Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3usize, 4usize, 8usize);

        let mut gate_up_data = Vec::with_capacity(num_experts * 2 * out_dim * in_dim);
        let mut down_data = Vec::with_capacity(num_experts * in_dim * out_dim);
        for expert_idx in 0..num_experts {
            let gate_val = (expert_idx + 1) as f32 * 0.1;
            let up_val = gate_val * 3.0;
            gate_up_data.extend(std::iter::repeat_n(gate_val, out_dim * in_dim));
            gate_up_data.extend(std::iter::repeat_n(up_val, out_dim * in_dim));
            down_data.extend(std::iter::repeat_n(gate_val, in_dim * out_dim));
        }

        let gate_up_tensor =
            Tensor::from_vec(gate_up_data, (num_experts, 2 * out_dim, in_dim), &device)
                .expect("gate_up tensor");
        let gate_up_qt =
            QTensor::quantize(&gate_up_tensor, GgmlDType::F32).expect("quantize gate_up");
        let down_tensor = Tensor::from_vec(down_data, (num_experts, in_dim, out_dim), &device)
            .expect("down tensor");
        let down_qt = QTensor::quantize(&down_tensor, GgmlDType::F32).expect("quantize down");

        let gate_up_raw = gate_up_qt.data().expect("gate_up data");
        let down_raw = down_qt.data().expect("down data");

        let experts = unpack_experts_from_packed(
            &gate_up_raw,
            gate_up_qt.dtype(),
            2 * out_dim,
            in_dim,
            &down_raw,
            down_qt.dtype(),
            in_dim,
            out_dim,
            num_experts,
            &device,
        )
        .expect("unpack_experts_from_packed");

        assert_eq!(experts.len(), num_experts);

        let x = Tensor::ones((1, in_dim), DType::F32, &device).expect("input");
        for (expert_idx, expert) in experts.iter().enumerate() {
            assert_eq!(expert.intermediate_size, out_dim);
            let gate_val = (expert_idx + 1) as f32 * 0.1;
            let up_val = gate_val * 3.0;
            let down_val = gate_val;
            let gate_out = gate_val * in_dim as f32;
            let up_out = up_val * in_dim as f32;
            let silu_gate = gate_out / (1.0 + (-gate_out).exp());
            let expected = out_dim as f32 * silu_gate * up_out * down_val;

            let got = expert
                .forward(&x)
                .expect("forward")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to_vec1");
            for v in got {
                assert!(
                    (v - expected).abs() < 1e-4,
                    "expert {expert_idx}: got {v}, expected {expected}"
                );
            }
        }
    }

    // GgmlDType::F32 quantization is lossless, so this exercises the real
    // byte-concat path with exact value comparisons.
    #[test]
    fn test_fuse_quantized_qtensors() {
        use candle_core::quantized::GgmlDType;

        // gate: [rows=2, cols=4] filled with 1.0; up: [rows=3, cols=4] filled
        // with 2.0. Different row counts to catch a bug that assumes equal
        // sizes.
        let device = &Device::Cpu;
        let gate_tensor = Tensor::from_vec(vec![1.0f32; 2 * 4], (2, 4), device).expect("gate");
        let up_tensor = Tensor::from_vec(vec![2.0f32; 3 * 4], (3, 4), device).expect("up");
        let gate = QTensor::quantize(&gate_tensor, GgmlDType::F32).expect("quantize gate");
        let up = QTensor::quantize(&up_tensor, GgmlDType::F32).expect("quantize up");

        let fused = fuse_quantized_qtensors(&gate, &up, device).expect("fuse");
        assert_eq!(fused.shape().dims(), &[5, 4]);
        assert_eq!(fused.dtype(), GgmlDType::F32);

        let vals = fused
            .dequantize(device)
            .expect("dequantize")
            .to_vec2::<f32>()
            .expect("to_vec2");
        for row in &vals[0..2] {
            assert!(row.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }
        for row in &vals[2..5] {
            assert!(row.iter().all(|&v| (v - 2.0).abs() < 1e-6));
        }
    }

    // GgmlDType::F32 quantization is lossless, so this exercises the real
    // per-expert byte-interleaving path with exact value comparisons.
    #[test]
    fn test_fuse_packed_qtensors() {
        use candle_core::quantized::GgmlDType;

        // [num_experts=3, out=2, in=4]; expert i's gate slice is filled with
        // (i+1), up slice with 10*(i+1), so a mis-interleaved expert
        // boundary is trivially detectable.
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3, 2, 4);
        let gate_data: Vec<f32> = (0..num_experts * out_dim * in_dim)
            .map(|idx| ((idx / (out_dim * in_dim)) + 1) as f32)
            .collect();
        let up_data: Vec<f32> = gate_data.iter().map(|v| v * 10.0).collect();
        let gate_tensor =
            Tensor::from_vec(gate_data, (num_experts, out_dim, in_dim), device).expect("gate");
        let up_tensor =
            Tensor::from_vec(up_data, (num_experts, out_dim, in_dim), device).expect("up");
        let gate = QTensor::quantize(&gate_tensor, GgmlDType::F32).expect("quantize gate");
        let up = QTensor::quantize(&up_tensor, GgmlDType::F32).expect("quantize up");

        let fused = fuse_packed_qtensors(&gate, &up, device).expect("fuse");
        assert_eq!(fused.shape().dims(), &[num_experts, 2 * out_dim, in_dim]);
        let raw = fused.data().expect("raw bytes");

        for expert_idx in 0..num_experts {
            let sliced = slice_packed_qtensor(
                &raw,
                GgmlDType::F32,
                expert_idx,
                2 * out_dim,
                in_dim,
                device,
            )
            .expect("slice");
            let vals = sliced
                .dequantize(device)
                .expect("dequantize")
                .to_vec2::<f32>()
                .expect("to_vec2");
            let expected_gate = (expert_idx + 1) as f32;
            let expected_up = expected_gate * 10.0;
            for row in &vals[0..out_dim] {
                assert!(
                    row.iter().all(|&v| (v - expected_gate).abs() < 1e-6),
                    "expert {expert_idx} gate half: got {row:?}, expected all {expected_gate}"
                );
            }
            for row in &vals[out_dim..2 * out_dim] {
                assert!(
                    row.iter().all(|&v| (v - expected_up).abs() < 1e-6),
                    "expert {expert_idx} up half: got {row:?}, expected all {expected_up}"
                );
            }
        }
    }

    // `combine_expert_outputs` is plain candle-core tensor arithmetic (no
    // CUDA/ROCm dependency), so it can be verified on CPU even though
    // `indexed_moe_forward` itself cannot.
    #[test]
    fn combine_expert_outputs_weights_and_reshapes() {
        // 2 tokens, top_k=3, hidden=4.
        let expert_out =
            Tensor::from_vec(vec![1.0f32; 2 * 3 * 4], (2, 3, 4), &Device::Cpu).unwrap();
        let topk_weights =
            Tensor::from_vec(vec![0.5f32, 0.3, 0.2, 0.6, 0.1, 0.3], (2, 3), &Device::Cpu).unwrap();
        let result =
            SparseMoeBlock::combine_expert_outputs(&expert_out, &topk_weights, DType::F32, &[2, 4])
                .unwrap();
        assert_eq!(result.dims(), &[2, 4]);
        // Each token's per-hidden-dim output is the sum of its top-k weights
        // (since every expert output element is 1.0): both rows sum to 1.0.
        let vals = result.to_vec2::<f32>().unwrap();
        let eps = 1e-6;
        for row in &vals {
            for &v in row {
                assert!((v - 1.0).abs() < eps, "expected ~1.0, got {v}");
            }
        }
    }

    // `indexed_moe_forward` is CUDA/ROCm-only in the candle fork, so the CPU
    // unit test process can never construct an eligible device, but the
    // eligibility check itself is still a plain function callable here.
    #[test]
    fn supports_fused_moe_cpu_always_false() {
        use candle_core::quantized::GgmlDType;

        for dtype in [
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
        ] {
            assert!(
                !supports_fused_moe(&Device::Cpu, dtype),
                "CPU must never be fused-eligible, dtype={dtype:?}"
            );
        }
    }

    // Quant types outside Q2K-Q6K/Q8_0 have no `indexed_moe_forward` kernel
    // in the candle fork, so they must stay ineligible even if some future
    // change makes `Device::Cpu` erroneously report as CUDA/ROCm-like.
    #[test]
    fn supports_fused_moe_unsupported_dtype_always_false() {
        use candle_core::quantized::GgmlDType;

        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_1,
            GgmlDType::Q8K,
        ] {
            assert!(
                !supports_fused_moe(&Device::Cpu, dtype),
                "unsupported dtype must never be fused-eligible: {dtype:?}"
            );
        }
    }

    // Verifies the absent-env-var case falls back to the documented default.
    #[test]
    fn parse_min_batch_default() {
        assert_eq!(parse_min_batch(None), 32);
    }

    // Verifies a well-formed positive value is used as-is.
    #[test]
    fn parse_min_batch_valid() {
        assert_eq!(parse_min_batch(Some("64")), 64);
    }

    // Verifies zero falls back to the default rather than disabling the threshold.
    #[test]
    fn parse_min_batch_zero_falls_back() {
        assert_eq!(parse_min_batch(Some("0")), 32);
    }

    // Verifies a negative value falls back to the default.
    #[test]
    fn parse_min_batch_negative_falls_back() {
        assert_eq!(parse_min_batch(Some("-1")), 32);
    }

    // Verifies an unparseable value falls back to the default.
    #[test]
    fn parse_min_batch_invalid_falls_back() {
        assert_eq!(parse_min_batch(Some("abc")), 32);
    }

    // Verifies surrounding whitespace is trimmed before parsing.
    #[test]
    fn parse_min_batch_whitespace_trimmed() {
        assert_eq!(parse_min_batch(Some("  128  ")), 128);
    }

    /// Verifies a packed CPU `gate_up`/`down` tensor pair against the
    /// closed-form expected output for an all-ones input of width
    /// `hidden_size`, where expert `i`'s weights are the constants
    /// `gate_val = (i+1) * 0.1`, `up_val = 3 * gate_val`, `down_val =
    /// gate_val`. Routes token `i` to expert `i` (identity routing) and
    /// calls [`cpu_indexed_moe_forward`] directly, since `experts` is
    /// empty when the packed-CPU path is taken (no per-expert `MoeExpert`
    /// to call `forward` on).
    fn assert_packed_constant_experts_forward_correct(
        packed_gate_up: &QTensor,
        packed_down: &QTensor,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) {
        let device = Device::Cpu;
        let topk_ids = Tensor::from_vec(
            (0..num_experts as u32).collect::<Vec<_>>(),
            (num_experts, 1),
            &device,
        )
        .expect("topk_ids");
        let mut routing = MoeRouting::new();
        routing.compute(&topk_ids, num_experts).expect("routing");

        let x = Tensor::ones((num_experts, 1, hidden_size), DType::F32, &device).expect("input");
        let gate_up_out =
            cpu_indexed_moe_forward(packed_gate_up, &x, &routing).expect("gate_up forward");
        let gate = gate_up_out
            .narrow(D::Minus1, 0, intermediate_size)
            .expect("gate narrow");
        let up = gate_up_out
            .narrow(D::Minus1, intermediate_size, intermediate_size)
            .expect("up narrow");
        let hidden = (Activation::Silu.forward(&gate).unwrap() * up).unwrap();
        let down_out =
            cpu_indexed_moe_forward(packed_down, &hidden, &routing).expect("down forward");
        let got = down_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for expert_idx in 0..num_experts {
            let gate_val = (expert_idx + 1) as f32 * 0.1;
            let up_val = gate_val * 3.0;
            let down_val = gate_val;
            let gate_out = gate_val * hidden_size as f32;
            let up_out = up_val * hidden_size as f32;
            let silu_gate = gate_out / (1.0 + (-gate_out).exp());
            let expected = intermediate_size as f32 * silu_gate * up_out * down_val;
            for col in 0..hidden_size {
                let v = got[expert_idx * hidden_size + col];
                assert!(
                    (v - expected).abs() < 1e-4,
                    "expert {expert_idx}: got {v}, expected {expected}"
                );
            }
        }
    }

    // `load_packed_experts` must keep CPU-resident packed tensors intact
    // (still quantized, not sliced/dequantized into per-expert
    // `MoeExpert`s), so `experts` stays empty and the CPU batched dispatch
    // path (`cpu_indexed_moe_forward`) can use the packed tensors directly.
    #[test]
    fn load_packed_experts_keeps_packed_tensors_on_cpu() {
        use candle_core::quantized::GgmlDType;
        use std::io::Cursor;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let (num_experts, out_dim, in_dim) = (2usize, 4usize, 8usize);

        let mut gate_data = Vec::with_capacity(num_experts * out_dim * in_dim);
        let mut up_data = Vec::with_capacity(num_experts * out_dim * in_dim);
        let mut down_data = Vec::with_capacity(num_experts * in_dim * out_dim);
        for expert_idx in 0..num_experts {
            let gate_val = (expert_idx + 1) as f32 * 0.1;
            let up_val = gate_val * 3.0;
            gate_data.extend(std::iter::repeat_n(gate_val, out_dim * in_dim));
            up_data.extend(std::iter::repeat_n(up_val, out_dim * in_dim));
            down_data.extend(std::iter::repeat_n(gate_val, in_dim * out_dim));
        }

        let mut writer = Cursor::new(Vec::new());
        let gate = QTensor::quantize(
            &Tensor::from_vec(gate_data, (num_experts, out_dim, in_dim), &device)
                .expect("gate data"),
            GgmlDType::F32,
        )
        .expect("quantize gate");
        let up = QTensor::quantize(
            &Tensor::from_vec(up_data, (num_experts, out_dim, in_dim), &device).expect("up data"),
            GgmlDType::F32,
        )
        .expect("quantize up");
        let down = QTensor::quantize(
            &Tensor::from_vec(down_data, (num_experts, in_dim, out_dim), &device)
                .expect("down data"),
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

        let (experts, packed_gate_up, packed_down) =
            SparseMoeBlock::load_packed_experts(&mut gg, "blk.0", num_experts, &device)
                .expect("load_packed_experts");

        assert!(
            experts.is_empty(),
            "CPU keeps packed tensors for the batched dispatch path; experts must stay empty"
        );
        let packed_gate_up = packed_gate_up.expect("packed gate_up must be Some on CPU");
        let packed_down = packed_down.expect("packed down must be Some on CPU");
        assert_eq!(
            packed_gate_up.shape().dims(),
            &[num_experts, 2 * out_dim, in_dim]
        );
        assert_eq!(packed_down.shape().dims(), &[num_experts, in_dim, out_dim]);

        assert_packed_constant_experts_forward_correct(
            &packed_gate_up,
            &packed_down,
            num_experts,
            out_dim,
            in_dim,
        );
    }

    // A pre-fused `ffn_gate_up_exps` tensor (some GGUF converters emit this)
    // must be detected and used directly, without also requiring separate
    // `ffn_gate_exps`/`ffn_up_exps` tensors to be present. Expert i's gate
    // half is filled with a distinct value and its up half with another, so
    // a gate/up half swap (or any other ordering mismatch) in the externally
    // pre-fused tensor changes the forward-pass output instead of silently
    // passing, unlike an all-ones fill.
    #[test]
    fn load_packed_experts_uses_pre_fused_gate_up_tensor() {
        use candle_core::quantized::GgmlDType;
        use std::io::Cursor;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let (num_experts, out_dim, in_dim) = (2usize, 4usize, 8usize);

        let mut gate_up_data = Vec::with_capacity(num_experts * 2 * out_dim * in_dim);
        let mut down_data = Vec::with_capacity(num_experts * in_dim * out_dim);
        for expert_idx in 0..num_experts {
            let gate_val = (expert_idx + 1) as f32 * 0.1;
            let up_val = gate_val * 3.0;
            gate_up_data.extend(std::iter::repeat_n(gate_val, out_dim * in_dim));
            gate_up_data.extend(std::iter::repeat_n(up_val, out_dim * in_dim));
            down_data.extend(std::iter::repeat_n(gate_val, in_dim * out_dim));
        }

        let mut writer = Cursor::new(Vec::new());
        // Pre-fused: out dim is already 2*out_dim (gate half + up half).
        let gate_up_tensor =
            Tensor::from_vec(gate_up_data, (num_experts, 2 * out_dim, in_dim), &device)
                .expect("gate_up tensor");
        let gate_up = QTensor::quantize(&gate_up_tensor, GgmlDType::F32).expect("quantize gate_up");
        let down_tensor = Tensor::from_vec(down_data, (num_experts, in_dim, out_dim), &device)
            .expect("down tensor");
        let down = QTensor::quantize(&down_tensor, GgmlDType::F32).expect("quantize down");
        candle_core::quantized::gguf_file::write(
            &mut writer,
            &[],
            &[
                ("blk.0.ffn_gate_up_exps.weight", &gate_up),
                ("blk.0.ffn_down_exps.weight", &down),
            ],
        )
        .expect("write gguf");

        writer.set_position(0);
        let ct = candle_core::quantized::gguf_file::Content::read(&mut writer).expect("read gguf");
        let mut gg = Gguf::new(ct, writer, device.clone(), dtype);

        let (experts, packed_gate_up, packed_down) =
            SparseMoeBlock::load_packed_experts(&mut gg, "blk.0", num_experts, &device)
                .expect("load_packed_experts");

        assert!(experts.is_empty());
        let packed_gate_up = packed_gate_up.expect("packed gate_up must be Some on CPU");
        let packed_down = packed_down.expect("packed down must be Some on CPU");

        assert_packed_constant_experts_forward_correct(
            &packed_gate_up,
            &packed_down,
            num_experts,
            out_dim,
            in_dim,
        );
    }

    // Verifies `MoeExpert::new_from_gguf` fuses the per-expert gate/up
    // tensors into a numerically correct `gate_up_proj`: the fused forward
    // pass must match manually computing `silu(gate(x)) * up(x)` from the
    // separately-loaded, unfused weights. GgmlDType::F32 quantization is
    // lossless, so this is an exact comparison.
    #[test]
    fn moe_expert_new_from_gguf_fuses_gate_up_correctly() {
        use candle_core::quantized::GgmlDType;
        use std::io::Cursor;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let (hidden, intermediate) = (4usize, 2usize);

        let gate_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let up_data: Vec<f32> = gate_data.iter().map(|v| v * 3.0).collect();
        let down_data: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let gate_tensor =
            Tensor::from_vec(gate_data, (intermediate, hidden), &device).expect("gate");
        let up_tensor = Tensor::from_vec(up_data, (intermediate, hidden), &device).expect("up");
        let down_tensor =
            Tensor::from_vec(down_data, (hidden, intermediate), &device).expect("down");
        let gate_qt = QTensor::quantize(&gate_tensor, GgmlDType::F32).expect("quantize gate");
        let up_qt = QTensor::quantize(&up_tensor, GgmlDType::F32).expect("quantize up");
        let down_qt = QTensor::quantize(&down_tensor, GgmlDType::F32).expect("quantize down");

        let mut writer = Cursor::new(Vec::new());
        candle_core::quantized::gguf_file::write(
            &mut writer,
            &[],
            &[
                ("blk.0.ffn_gate.0.weight", &gate_qt),
                ("blk.0.ffn_up.0.weight", &up_qt),
                ("blk.0.ffn_down.0.weight", &down_qt),
            ],
        )
        .expect("write gguf");
        writer.set_position(0);
        let ct = candle_core::quantized::gguf_file::Content::read(&mut writer).expect("read gguf");
        let mut gg = Gguf::new(ct, writer, device.clone(), dtype);

        let expert = MoeExpert::new_from_gguf(&mut gg, 0, 0, &device).expect("new_from_gguf");

        let x = Tensor::new(&[1.0f32, 0.5, -0.5, 2.0], &device)
            .expect("tensor")
            .reshape((1, hidden))
            .expect("reshape");
        let got = expert
            .forward(&x)
            .expect("forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Manually compute silu(gate(x)) * up(x) @ down using the unfused weight
        // layers directly.
        let gate_layer = LinearLayer::quantized(QMatMul::from_arc(Arc::new(gate_qt)).unwrap());
        let up_layer = LinearLayer::quantized(QMatMul::from_arc(Arc::new(up_qt)).unwrap());
        let down_layer = LinearLayer::quantized(QMatMul::from_arc(Arc::new(down_qt)).unwrap());
        let gate_out = Activation::Silu
            .forward(&gate_layer.forward(&x).unwrap())
            .unwrap();
        let up_out = up_layer.forward(&x).unwrap();
        let expected = down_layer
            .forward(&(gate_out * up_out).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {got:?}, expected {expected:?}");
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

    // Verifies `MoeExpert::forward_f32` matches `Module::forward` output
    // when given F32 input.
    #[test]
    fn forward_f32_matches_forward_on_f32_input() {
        let vb = identity_vb(8);
        let expert = MoeExpert::new(8, 8, vb).expect("new");
        let x = Tensor::arange(0f32, 8f32, &Device::Cpu)
            .expect("arange")
            .reshape((1, 8))
            .expect("reshape");

        let via_forward = expert.forward(&x).expect("forward");
        let via_forward_f32 = expert.forward_f32(&x).expect("forward_f32");

        assert_eq!(via_forward_f32.dtype(), DType::F32);
        assert_eq!(
            via_forward.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            via_forward_f32
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    // Manual perf comparison for the same-device quantized-expert dispatch
    // fixed alongside `quantized_experts` in `SparseMoeBlock::forward`:
    // `forward` (cast to F32, matmul, cast back to `original_dtype`) versus
    // `forward_f32` (cast to F32, matmul, stay F32) on a BF16/F16-sized
    // expert. Runs on CUDA/ROCm when built with that feature (matching
    // `Qwen3Model::from_gguf_with_kv_kind`'s device -> dtype selection),
    // CPU otherwise. Timing-based, so `#[ignore]`d by default; run with:
    //   cargo test -p crane-core --release --features rocm \
    //     same_device_quantized_forward_f32_avoids_round_trip -- --ignored --nocapture
    #[test]
    #[ignore]
    fn same_device_quantized_forward_f32_avoids_round_trip() {
        use candle_core::quantized::GgmlDType;
        use std::io::Cursor;
        use std::time::Instant;

        #[cfg(feature = "cuda")]
        let (device, activation_dtype) = (Device::new_cuda(0).expect("cuda device"), DType::BF16);
        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        let (device, activation_dtype) = (Device::new_rocm(0).expect("rocm device"), DType::F16);
        // CPU never runs this dtype in production (`from_gguf_with_kv_kind`
        // always picks F32 there); F16 here only to exercise the same
        // round-trip this benchmark is measuring.
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let (device, activation_dtype) = (Device::Cpu, DType::F16);

        let cpu = Device::Cpu;
        let dtype = DType::F32;
        let (hidden, intermediate, batch) = (2048usize, 768usize, 8usize);

        let gate_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let up_data: Vec<f32> = gate_data.iter().map(|v| v * 0.5).collect();
        let down_data: Vec<f32> = (0..hidden * intermediate)
            .map(|i| ((i as f32) * 0.001).cos())
            .collect();

        // Block quantization runs on CPU; `MoeExpert::new_from_gguf` below
        // loads the resulting GGUF bytes onto `device`, same as production
        // GGUF loading assigning an expert to an arbitrary device.
        let gate_tensor = Tensor::from_vec(gate_data, (intermediate, hidden), &cpu).expect("gate");
        let up_tensor = Tensor::from_vec(up_data, (intermediate, hidden), &cpu).expect("up");
        let down_tensor = Tensor::from_vec(down_data, (hidden, intermediate), &cpu).expect("down");
        let gate_qt = QTensor::quantize(&gate_tensor, GgmlDType::Q8_0).expect("quantize gate");
        let up_qt = QTensor::quantize(&up_tensor, GgmlDType::Q8_0).expect("quantize up");
        let down_qt = QTensor::quantize(&down_tensor, GgmlDType::Q8_0).expect("quantize down");

        let mut writer = Cursor::new(Vec::new());
        candle_core::quantized::gguf_file::write(
            &mut writer,
            &[],
            &[
                ("blk.0.ffn_gate.0.weight", &gate_qt),
                ("blk.0.ffn_up.0.weight", &up_qt),
                ("blk.0.ffn_down.0.weight", &down_qt),
            ],
        )
        .expect("write gguf");
        writer.set_position(0);
        let ct = candle_core::quantized::gguf_file::Content::read(&mut writer).expect("read gguf");
        let mut gg = Gguf::new(ct, writer, device.clone(), dtype);
        let expert = MoeExpert::new_from_gguf(&mut gg, 0, 0, &device).expect("new_from_gguf");

        // Per-expert token count during decode: top-8 routing selects up to
        // `num_experts_per_tok` tokens per expert per step.
        let x = Tensor::randn(0f32, 1.0, (batch, hidden), &device)
            .expect("randn")
            .to_dtype(activation_dtype)
            .expect("to_dtype activation_dtype");

        let old_path = || -> Result<Tensor> { expert.forward(&x)?.to_dtype(DType::F32) };
        let new_path = || -> Result<Tensor> { expert.forward_f32(&x) };

        for _ in 0..10 {
            let _ = old_path().expect("warmup old_path");
            let _ = new_path().expect("warmup new_path");
        }
        device.synchronize().expect("sync after warmup");

        let iters = 200;
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(old_path().expect("old_path"));
        }
        device.synchronize().expect("sync after old_path");
        let old_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(new_path().expect("new_path"));
        }
        device.synchronize().expect("sync after new_path");
        let new_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

        println!(
            "same_device quantized expert (hidden={hidden}, intermediate={intermediate}, batch={batch}, device={device:?}): \
             forward+to_dtype(F32) = {old_ms:.4} ms/iter, forward_f32 = {new_ms:.4} ms/iter"
        );
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
        SparseMoeBlock::new(&config, 0, hidden, vb, &Device::Cpu).expect("SparseMoeBlock::new")
    }

    /// Builds a packed-CPU `SparseMoeBlock` from the same `(gate, up,
    /// down)` per-expert weight layout `make_sparse_moe` uses, but stored
    /// as packed 3D `QTensor`s (`F32`-quantized, which is lossless)
    /// instead of per-expert `MoeExpert`s -- exercising
    /// `SparseMoeBlock::cpu_batched_forward` instead of the per-expert loop.
    fn make_packed_sparse_moe(
        hidden: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        gate_data: Vec<f32>,
        expert_data: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> SparseMoeBlock {
        let device = &Device::Cpu;
        let gate_weight =
            Tensor::from_vec(gate_data, (num_experts, hidden), device).expect("gate weight");
        let gate = LinearLayer::Standard(Linear::new(gate_weight, None));

        // Pack gate+up per expert into [2*intermediate, hidden] (gate rows
        // first, then up rows, matching MoeExpert::new's Tensor::cat
        // order), then stack every expert into
        // [num_experts, 2*intermediate, hidden].
        let mut gate_up_rows: Vec<f32> =
            Vec::with_capacity(num_experts * 2 * moe_intermediate_size * hidden);
        let mut down_rows: Vec<f32> =
            Vec::with_capacity(num_experts * hidden * moe_intermediate_size);
        for (gate_w, up_w, down_w) in expert_data {
            gate_up_rows.extend_from_slice(gate_w);
            gate_up_rows.extend_from_slice(up_w);
            down_rows.extend_from_slice(down_w);
        }
        let gate_up_packed = Tensor::from_vec(
            gate_up_rows,
            (num_experts, 2 * moe_intermediate_size, hidden),
            device,
        )
        .expect("packed gate_up");
        let down_packed = Tensor::from_vec(
            down_rows,
            (num_experts, hidden, moe_intermediate_size),
            device,
        )
        .expect("packed down");

        SparseMoeBlock {
            layer_idx: 0,
            gate,
            experts: Vec::new(),
            num_experts_per_tok,
            norm_topk_prob,
            expert_device: Device::Cpu,
            packed_gate_up_exps: Some(Arc::new(
                QTensor::quantize(&gate_up_packed, GgmlDType::F32).expect("quantize gate_up"),
            )),
            packed_down_exps: Some(Arc::new(
                QTensor::quantize(&down_packed, GgmlDType::F32).expect("quantize down"),
            )),
            routing: Mutex::new(MoeRouting::new()),
        }
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

    // Verifies `SparseMoeBlock::cpu_batched_forward` (the packed-CPU
    // dispatch path, taken when `packed_gate_up_exps`/`packed_down_exps`
    // are `Some` and `expert_device` is CPU) produces the same output as
    // the per-expert fallback loop, given identical weight data. This is
    // the integration test confirming `cpu_indexed_moe_forward` is wired
    // correctly into `SparseMoeBlock::forward`'s dispatch, not just correct
    // in isolation (already covered by the `test_cpu_indexed_moe_forward_*`
    // tests below).
    #[test]
    fn cpu_batched_forward_matches_per_expert() {
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let scaled = |c: f32| vec![c, 0.0, 0.0, c];
        let expert_data = vec![
            (identity.clone(), scaled(1.0), identity.clone()),
            (identity.clone(), scaled(3.0), identity.clone()),
            (identity.clone(), scaled(2.0), identity.clone()),
            (identity.clone(), scaled(5.0), identity.clone()),
        ];
        let gate_data = vec![
            2.0, 0.0, //
            -2.0, 0.0, //
            1.5, 0.0, //
            -1.5, 0.0,
        ];
        let per_expert = make_sparse_moe(2, 2, 4, 2, false, gate_data.clone(), &expert_data);
        let packed = make_packed_sparse_moe(2, 2, 4, 2, false, gate_data, &expert_data);

        let x = Tensor::new(&[1.0f32, 0.5, -0.5, 2.0], &Device::Cpu)
            .expect("tensor")
            .reshape((2, 2))
            .expect("reshape");

        let expected = per_expert
            .forward(&x)
            .expect("per-expert forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = packed
            .forward(&x)
            .expect("packed forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (e, g) in expected.iter().zip(got.iter()) {
            assert!((e - g).abs() < 1e-4, "expected {expected:?}, got {got:?}");
        }
    }

    // Verifies `SparseMoeBlock::gpu_offload_forward` (uploading a
    // CPU-resident packed expert pair to a GPU device and dispatching
    // through `fused_forward`) produces the same output as
    // `cpu_batched_forward`, given identical weight data and routing.
    // Requires a real CUDA/ROCm device (the underlying `indexed_moe_forward`
    // has no CPU implementation), so this is only compiled with one of those
    // features enabled and `#[ignore]`d by default; run with:
    //   cargo test -p crane-core --features cuda \
    //     gpu_offload_forward_matches_cpu_batched -- --ignored --nocapture
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[test]
    #[ignore]
    fn gpu_offload_forward_matches_cpu_batched() {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).expect("cuda device");
        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        let device = Device::new_rocm(0).expect("rocm device");

        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let scaled = |c: f32| vec![c, 0.0, 0.0, c];
        let expert_data = vec![
            (identity.clone(), scaled(1.0), identity.clone()),
            (identity.clone(), scaled(3.0), identity.clone()),
            (identity.clone(), scaled(2.0), identity.clone()),
            (identity.clone(), scaled(5.0), identity.clone()),
        ];
        let gate_data = vec![
            2.0, 0.0, //
            -2.0, 0.0, //
            1.5, 0.0, //
            -1.5, 0.0,
        ];
        let moe = make_packed_sparse_moe(2, 2, 4, 2, false, gate_data, &expert_data);

        let x = Tensor::new(&[1.0f32, 0.5, -0.5, 2.0], &Device::Cpu)
            .expect("tensor")
            .reshape((2, 2))
            .expect("reshape");

        let expected = moe
            .forward(&x)
            .expect("cpu batched forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let xs_f32 = x.to_dtype(DType::F32).expect("to_dtype");
        let logits = moe.gate.forward(&xs_f32).expect("gate forward");
        let (topk_ids, topk_weights) =
            crate::ops::fused_ops::topk_moe::topk_moe_routing(&logits, 2, false).expect("routing");
        let gate_up_exps = moe.packed_gate_up_exps.as_ref().expect("packed gate_up");
        let down_exps = moe.packed_down_exps.as_ref().expect("packed down");

        let got = SparseMoeBlock::gpu_offload_forward(
            &xs_f32,
            &topk_ids,
            &topk_weights,
            gate_up_exps,
            down_exps,
            &device,
            DType::F32,
            &[2, 2],
        )
        .expect("gpu_offload_forward")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        for (e, g) in expected.iter().zip(got.iter()) {
            assert!((e - g).abs() < 1e-4, "expected {expected:?}, got {got:?}");
        }
    }

    // Verifies the packed-CPU dispatch path handles both 2D and 3D inputs,
    // mirroring `test_sparse_moe_output_shape_2d`/`_3d` for the per-expert
    // path.
    #[test]
    fn cpu_batched_forward_shape_2d_and_3d() {
        let hidden = 8;
        let intermediate = 4;
        let num_experts = 4;
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
        let moe = make_packed_sparse_moe(
            hidden,
            intermediate,
            num_experts,
            2,
            false,
            gate_data,
            &expert_data,
        );

        let x2d = Tensor::zeros((3, hidden), DType::F32, &Device::Cpu).expect("zeros 2d");
        let y2d = moe.forward(&x2d).expect("forward 2d");
        assert_eq!(y2d.dims(), &[3, hidden]);

        let x3d = Tensor::zeros((2, 3, hidden), DType::F32, &Device::Cpu).expect("zeros 3d");
        let y3d = moe.forward(&x3d).expect("forward 3d");
        assert_eq!(y3d.dims(), &[2, 3, hidden]);
    }

    // Verifies a packed-CPU `SparseMoeBlock`'s `promote_experts_to` early
    // return (already on the target device) leaves the packed tensors
    // intact rather than accidentally clearing them, mirroring
    // `promote_experts_to_same_device_is_noop` for the per-expert path.
    #[test]
    fn promote_packed_cpu_to_cpu_is_noop() {
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let scaled = |c: f32| vec![c, 0.0, 0.0, c];
        let expert_data = vec![
            (identity.clone(), scaled(1.0), identity.clone()),
            (identity.clone(), scaled(3.0), identity.clone()),
        ];
        let gate_data = vec![2.0, 0.0, -2.0, 0.0];
        let mut moe = make_packed_sparse_moe(2, 2, 2, 1, false, gate_data, &expert_data);

        let x = Tensor::new(&[1.0f32, 0.5], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let before = moe
            .forward(&x)
            .expect("forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        moe.promote_experts_to(&Device::Cpu, DType::F32)
            .expect("promote");
        assert!(moe.packed_gate_up_exps.is_some());
        assert!(moe.packed_down_exps.is_some());

        let after = moe
            .forward(&x)
            .expect("forward")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(before, after);
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
        let quantized = LinearLayer::quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());
        let expert = MoeExpert::from_layers(
            quantized,
            LinearLayer::Standard(Linear::new(weight, None)),
            2,
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
        let expert =
            MoeExpert::from_layers(biased, LinearLayer::Standard(Linear::new(weight, None)), 2);

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
        let moe = SparseMoeBlock::new(&config, 0, hidden, vb, device).expect("SparseMoeBlock::new");

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

    /// Builds a packed `[num_experts, out_dim, in_dim]` `QTensor`, quantized
    /// as `dtype`, where expert `i`'s weight matrix is filled with
    /// `(i + 1) * 0.1`.
    fn make_packed_experts(
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
        dtype: GgmlDType,
        device: &Device,
    ) -> QTensor {
        let mut data = Vec::with_capacity(num_experts * out_dim * in_dim);
        for e in 0..num_experts {
            let val = (e + 1) as f32 * 0.1;
            data.extend(std::iter::repeat_n(val, out_dim * in_dim));
        }
        let tensor = Tensor::from_vec(data, (num_experts, out_dim, in_dim), device)
            .expect("packed weight tensor");
        QTensor::quantize(&tensor, dtype).expect("quantize packed weights")
    }

    /// Computes a [`MoeRouting`] for `topk_ids`, for tests that only need
    /// routing for a single `cpu_indexed_moe_forward` call.
    fn make_routing(topk_ids: &Tensor, num_experts: usize) -> MoeRouting {
        let mut routing = MoeRouting::new();
        routing
            .compute(topk_ids, num_experts)
            .expect("compute routing");
        routing
    }

    /// Reference implementation: for each `(token, slot)`, runs expert
    /// `topk_ids[token, slot]`'s matmul directly against its input row via
    /// `slice_packed_qtensor` + `QMatMul`, without any of
    /// `cpu_indexed_moe_forward`'s batching. Returns a flat
    /// `[tokens * topk * out_dim]` `Vec<f32>`.
    fn reference_moe_forward(
        packed: &QTensor,
        xs: &[f32],
        topk_ids: &[u32],
        tokens: usize,
        input_dim1: usize,
        topk: usize,
        in_dim: usize,
        out_dim: usize,
        device: &Device,
    ) -> Vec<f32> {
        let dtype = packed.dtype();
        let raw = packed.data().expect("packed data");
        let mut out = vec![0.0f32; tokens * topk * out_dim];
        for t in 0..tokens {
            for s in 0..topk {
                let expert_idx = topk_ids[t * topk + s] as usize;
                let row_idx = if input_dim1 == 1 { t } else { t * topk + s };
                let row = Tensor::from_vec(
                    xs[row_idx * in_dim..(row_idx + 1) * in_dim].to_vec(),
                    (1, in_dim),
                    device,
                )
                .expect("input row");
                let sliced = slice_packed_qtensor(&raw, dtype, expert_idx, out_dim, in_dim, device)
                    .expect("slice expert");
                let qmm = QMatMul::from_arc(Arc::new(sliced)).expect("QMatMul");
                let expected_row = qmm
                    .forward(&row)
                    .expect("expert forward")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("to_vec1");
                out[(t * topk + s) * out_dim..(t * topk + s + 1) * out_dim]
                    .copy_from_slice(&expected_row);
            }
        }
        out
    }

    // GgmlDType::F32 quantization is lossless, so `cpu_indexed_moe_forward`'s
    // output can be compared exactly against per-expert reference matmuls.
    // Exercises the `input_dim1 == 1` case (gate/up projection): every
    // routed slot for a token shares the same input row.
    #[test]
    fn test_cpu_indexed_moe_forward_f32_exact_shared_input() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (4usize, 8usize, 4usize);
        let (tokens, topk) = (3usize, 2usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..tokens * in_dim)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = (0..tokens)
            .flat_map(|t| [(t % num_experts) as u32, ((t + 2) % num_experts) as u32])
            .collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        assert_eq!(got.dims(), &[tokens, topk, out_dim]);
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got_vals.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "got {got_vals:?}, expected {expected:?}"
            );
        }
    }

    // Exercises the `input_dim1 == topk` case (down projection): each
    // routed slot has its own input row (the per-expert intermediate
    // activation).
    #[test]
    fn test_cpu_indexed_moe_forward_f32_exact_per_slot_input() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3usize, 6usize, 4usize);
        let (tokens, topk) = (2usize, 3usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..tokens * topk * in_dim)
            .map(|i| (i as f32) * 0.03 + 0.1)
            .collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, topk, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = (0..tokens * topk)
            .map(|i| (i % num_experts) as u32)
            .collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        assert_eq!(got.dims(), &[tokens, topk, out_dim]);
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            topk,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got_vals.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "got {got_vals:?}, expected {expected:?}"
            );
        }
    }

    // GgmlDType::Q8_0 introduces real quantization error, unlike the lossless
    // F32 tests above, so this uses a wider tolerance while still exercising
    // the real quantize/dequantize `vec_dot_4` path with `BLCK_SIZE=32`.
    #[test]
    fn test_cpu_indexed_moe_forward_q8_0_tolerance() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (4usize, 32usize, 64usize);
        let (tokens, topk) = (2usize, 2usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::Q8_0, device);

        let xs_data: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = (0..tokens)
            .flat_map(|t| [(t % num_experts) as u32, ((t + 1) % num_experts) as u32])
            .collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got_vals.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 0.05,
                "got {got_vals:?}, expected {expected:?}"
            );
        }
    }

    // GgmlDType::Q4K is a K-quant super-block format (`BLCK_SIZE=256`,
    // `DIRECT_COPY == false`), unlike Q8_0's flat block layout -- this
    // exercises `quantize_lhs_rows`'s parallel `from_float` path and Q4K's
    // super-block `vec_dot_4`, which is the format CPU-offloaded MoE experts
    // most commonly ship as in practice.
    #[test]
    fn test_cpu_indexed_moe_forward_q4k_tolerance() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (4usize, 32usize, 256usize);
        let (tokens, topk) = (2usize, 2usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::Q4K, device);

        let xs_data: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = (0..tokens)
            .flat_map(|t| [(t % num_experts) as u32, ((t + 1) % num_experts) as u32])
            .collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got_vals.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 0.1,
                "got {got_vals:?}, expected {expected:?}"
            );
        }
    }

    // An expert with no routed tokens must leave its would-be output slots
    // at zero, since `dst` is zero-initialized and unrouted experts never
    // appear in `active_experts`.
    #[test]
    fn test_cpu_indexed_moe_forward_unrouted_expert_stays_zero() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3usize, 8usize, 4usize);
        let (tokens, topk) = (2usize, 1usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..tokens * in_dim).map(|i| (i as f32) * 0.1).collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        // Every token routes to expert 0 only; experts 1 and 2 are unrouted.
        let topk_ids_data: Vec<u32> = vec![0u32; tokens * topk];
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got_vals.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "got {got_vals:?}, expected {expected:?}"
            );
        }
        // Sanity: expert 0's output must actually be non-zero (otherwise
        // this test would trivially pass with a broken kernel that always
        // returns zeros).
        assert!(got_vals.iter().any(|&v| v.abs() > 1e-6));
    }

    // The primary decode use case: a single token routed to `topk` experts,
    // all sharing the same input row.
    #[test]
    fn test_cpu_indexed_moe_forward_single_token_decode() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (8usize, 16usize, 8usize);
        let (tokens, topk) = (1usize, 8usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        // All experts routed exactly once, in order.
        let topk_ids_data: Vec<u32> = (0..num_experts as u32).collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        assert_eq!(got.dims(), &[tokens, topk, out_dim]);
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got_vals.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "got {got_vals:?}, expected {expected:?}"
            );
        }
    }

    // `out_dim` not a multiple of 4 exercises the tail-column path
    // (`vec_dot_2` for 2 remaining columns, `vec_dot` for 1) separately from
    // the main quad loop. Tested at tail lengths 1, 2, and 3.
    #[test]
    fn test_cpu_indexed_moe_forward_tail_columns() {
        let device = &Device::Cpu;
        for out_dim in [5usize, 6, 7] {
            let (num_experts, in_dim) = (3usize, 4usize);
            let (tokens, topk) = (2usize, 2usize);

            let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

            let xs_data: Vec<f32> = (0..tokens * in_dim).map(|i| (i as f32) * 0.07).collect();
            let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

            let topk_ids_data: Vec<u32> = (0..tokens)
                .flat_map(|t| [(t % num_experts) as u32, ((t + 1) % num_experts) as u32])
                .collect();
            let topk_ids =
                Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

            let routing = make_routing(&topk_ids, num_experts);
            let got =
                cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
            let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let expected = reference_moe_forward(
                &packed,
                &xs_data,
                &topk_ids_data,
                tokens,
                1,
                topk,
                in_dim,
                out_dim,
                device,
            );
            for (g, e) in got_vals.iter().zip(expected.iter()) {
                assert!(
                    (g - e).abs() < 1e-5,
                    "out_dim={out_dim}: got {got_vals:?}, expected {expected:?}"
                );
            }
        }
    }

    // Large enough (64 experts, out_dim=256 -> 4096 work items) to force
    // `dispatch_moe_quads`'s barrier pool to split work across multiple
    // worker threads, actually exercising the unsafe cross-thread `dst`
    // write's exclusivity guarantee rather than running single-threaded.
    #[test]
    fn test_cpu_indexed_moe_forward_large_concurrency() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (64usize, 256usize, 128usize);
        let (tokens, topk) = (4usize, 8usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = (0..tokens * topk)
            .map(|i| (i % num_experts) as u32)
            .collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        assert_eq!(got.dims(), &[tokens, topk, out_dim]);
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (i, (g, e)) in got_vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "mismatch at index {i}: got {g}, expected {e}"
            );
        }
    }

    // `dispatch_moe_quads` partitions its flat work space by actual
    // routed-pair count rather than by `active_experts.len()`, so it must
    // stay correct when routing is unbalanced (a handful of experts
    // taking far more tokens than the rest) and pairs are visited out of
    // per-expert order. This test's skewed routing (all of slot 0 routed
    // to expert 0, the rest spread thinly) exercises that every pair
    // still gets visited exactly once and lands in the correct output row.
    #[test]
    fn test_cpu_indexed_moe_forward_skewed_routing_matches_reference() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (16usize, 32usize, 16usize);
        let (tokens, topk) = (40usize, 4usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i as f32) * 0.037).cos())
            .collect();
        let xs = Tensor::from_vec(xs_data.clone(), (tokens, 1, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = (0..tokens)
            .flat_map(|t| {
                (0..topk).map(move |s| {
                    if s == 0 {
                        0
                    } else {
                        (1 + (t * 5 + s * 3) % (num_experts - 1)) as u32
                    }
                })
            })
            .collect();
        let topk_ids =
            Tensor::from_vec(topk_ids_data.clone(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        assert_eq!(got.dims(), &[tokens, topk, out_dim]);
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected = reference_moe_forward(
            &packed,
            &xs_data,
            &topk_ids_data,
            tokens,
            1,
            topk,
            in_dim,
            out_dim,
            device,
        );
        for (i, (g, e)) in got_vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "mismatch at index {i}: got {g}, expected {e}"
            );
        }
    }

    // Zero tokens means `active_experts` is empty, exercising the
    // `total_items == 0` early return in `dispatch_moe_quads` directly
    // (distinct from the unrouted-expert test, which still has active
    // experts and non-empty token lists).
    #[test]
    fn test_cpu_indexed_moe_forward_zero_tokens() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3usize, 8usize, 4usize);
        let (tokens, topk) = (0usize, 2usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs = Tensor::from_vec(Vec::<f32>::new(), (tokens, 1, in_dim), device).expect("xs");
        let topk_ids =
            Tensor::from_vec(Vec::<u32>::new(), (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let got = cpu_indexed_moe_forward(&packed, &xs, &routing).expect("cpu_indexed_moe_forward");
        assert_eq!(got.dims(), &[tokens, topk, out_dim]);
        let got_vals = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(got_vals.is_empty());
    }

    // A `topk_ids` row count that doesn't match `xs`'s token count must
    // return an error instead of `group_tokens_by_expert` indexing past the
    // end of the flattened `topk_ids` buffer.
    #[test]
    fn test_cpu_indexed_moe_forward_mismatched_tokens() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3usize, 8usize, 4usize);
        let (xs_tokens, topk_ids_tokens, topk) = (3usize, 2usize, 2usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);

        let xs_data: Vec<f32> = (0..xs_tokens * in_dim).map(|i| i as f32).collect();
        let xs = Tensor::from_vec(xs_data, (xs_tokens, 1, in_dim), device).expect("xs");

        let topk_ids_data: Vec<u32> = vec![0u32; topk_ids_tokens * topk];
        let topk_ids =
            Tensor::from_vec(topk_ids_data, (topk_ids_tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts);
        let err = cpu_indexed_moe_forward(&packed, &xs, &routing)
            .expect_err("mismatched token counts must error, not panic");
        assert!(
            err.to_string().contains("does not match"),
            "unexpected error: {err}"
        );
    }

    // `MoeRouting` computed for a different expert count than `packed_weights`
    // has must error, rather than let the matmul index the weight buffer
    // with an expert id that was valid at grouping time but isn't now.
    #[test]
    fn test_cpu_indexed_moe_forward_routing_expert_count_mismatch() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (3usize, 8usize, 4usize);
        let (tokens, topk) = (2usize, 2usize);

        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);
        let xs_data: Vec<f32> = (0..tokens * in_dim).map(|i| i as f32).collect();
        let xs = Tensor::from_vec(xs_data, (tokens, 1, in_dim), device).expect("xs");
        let topk_ids_data: Vec<u32> = vec![0u32; tokens * topk];
        let topk_ids = Tensor::from_vec(topk_ids_data, (tokens, topk), device).expect("topk_ids");

        let routing = make_routing(&topk_ids, num_experts + 1);
        let err = cpu_indexed_moe_forward(&packed, &xs, &routing)
            .expect_err("expert count mismatch must error");
        assert!(
            err.to_string().contains("experts"),
            "unexpected error: {err}"
        );
    }

    // A single `MoeRouting` reused across two different `topk_ids` (mirroring
    // the gate_up/down-projection reuse it exists for) must produce correct
    // results both times: `compute` must fully overwrite the prior call's
    // grouping rather than leaking stale `token_lists`/`active_experts`
    // entries left over from reusing their heap capacity.
    #[test]
    fn test_moe_routing_reuse() {
        let device = &Device::Cpu;
        let (num_experts, out_dim, in_dim) = (4usize, 8usize, 4usize);
        let packed = make_packed_experts(num_experts, out_dim, in_dim, GgmlDType::F32, device);
        let mut routing = MoeRouting::new();

        let (tokens1, topk1) = (3usize, 2usize);
        let xs1_data: Vec<f32> = (0..tokens1 * in_dim)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        let xs1 = Tensor::from_vec(xs1_data.clone(), (tokens1, 1, in_dim), device).expect("xs1");
        let topk_ids1_data: Vec<u32> = (0..tokens1)
            .flat_map(|t| [(t % num_experts) as u32, ((t + 2) % num_experts) as u32])
            .collect();
        let topk_ids1 =
            Tensor::from_vec(topk_ids1_data.clone(), (tokens1, topk1), device).expect("topk_ids1");
        routing.compute(&topk_ids1, num_experts).expect("compute 1");
        let got1 = cpu_indexed_moe_forward(&packed, &xs1, &routing).expect("forward 1");
        let got1_vals = got1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected1 = reference_moe_forward(
            &packed,
            &xs1_data,
            &topk_ids1_data,
            tokens1,
            1,
            topk1,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got1_vals.iter().zip(expected1.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "call 1: got {got1_vals:?}, expected {expected1:?}"
            );
        }

        let (tokens2, topk2) = (2usize, 1usize);
        let xs2_data: Vec<f32> = (0..tokens2 * in_dim)
            .map(|i| (i as f32) * 0.1 + 0.3)
            .collect();
        let xs2 = Tensor::from_vec(xs2_data.clone(), (tokens2, 1, in_dim), device).expect("xs2");
        let topk_ids2_data: Vec<u32> = vec![num_experts as u32 - 1; tokens2 * topk2];
        let topk_ids2 =
            Tensor::from_vec(topk_ids2_data.clone(), (tokens2, topk2), device).expect("topk_ids2");
        routing.compute(&topk_ids2, num_experts).expect("compute 2");
        let got2 = cpu_indexed_moe_forward(&packed, &xs2, &routing).expect("forward 2");
        let got2_vals = got2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected2 = reference_moe_forward(
            &packed,
            &xs2_data,
            &topk_ids2_data,
            tokens2,
            1,
            topk2,
            in_dim,
            out_dim,
            device,
        );
        for (g, e) in got2_vals.iter().zip(expected2.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "call 2: got {got2_vals:?}, expected {expected2:?}"
            );
        }
    }
}
