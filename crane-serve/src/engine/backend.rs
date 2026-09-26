//! Model backend abstraction for the inference engine.
//!
//! The [`ModelBackend`] trait decouples the engine from specific model implementations,
//! allowing any compatible LLM to be served through the OpenAI-compatible API.
//!
//! # Capability Levels
//!
//! | Capability        | Required | Effect when absent                            |
//! |-------------------|----------|-----------------------------------------------|
//! | `forward_step`    | Yes      | —                                             |
//! | KV cache swap     | No       | `max_concurrent` capped to 1                  |
//! | Batch decode      | No       | Sequences decoded sequentially per step        |

use anyhow::Result;
use crane_core::device::DeviceAssignment;
use crane_core::{DType, Device, Tensor, bail};

/// Per-layer KV cache for one sequence: `(K, V)` per layer, or `None` for
/// layers with no cached state yet.
pub type SequenceKvCaches = Vec<Option<(Tensor, Tensor)>>;

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Core abstraction over different model backends.
///
/// All models must support single-sequence forward passes and KV cache clearing.
/// Optionally, models can support KV cache extraction/restoration (for concurrent
/// sequence serving) and batched decoding (for GPU-efficient parallel generation).
pub trait ModelBackend: Send + 'static {
    /// Run a forward pass for a single sequence.
    ///
    /// * `input_ids` — token IDs to process
    /// * `start_pos` — KV cache position (0 for a fresh sequence)
    ///
    /// Returns logits tensor, typically `[1, seq_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor>;

    /// Clear all KV caches.
    ///
    /// # Errors
    ///
    /// Returns an error if a cache could not be reset: on an accelerator this
    /// issues device memsets, so it fails for the same reasons any other
    /// device op does.
    fn clear_kv_cache(&mut self) -> Result<()>;

    /// Capture the layer state so a later request sharing this token prefix
    /// can resume from it. `None` when the backend cannot (the default).
    ///
    /// The capture is only valid until something resets the caches.
    fn snapshot_state(&self) -> Option<crane_core::models::qwen3_5::StateSnapshot> {
        None
    }

    /// Restore a [`snapshot_state`](Self::snapshot_state) capture.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot restore, or the snapshot does
    /// not describe this model.
    fn restore_state(&mut self, _snap: &crane_core::models::qwen3_5::StateSnapshot) -> Result<()> {
        anyhow::bail!("this backend does not support state snapshots")
    }

    /// Number of transformer layers (for KV cache vector sizing).
    fn num_layers(&self) -> usize;

    /// Device the model is running on.
    fn device(&self) -> &Device;

    /// Data type of model weights.
    fn dtype(&self) -> DType;

    /// Reference to the underlying tokenizer.
    fn tokenizer(&self) -> &tokenizers::Tokenizer;

    /// The model's end-of-sequence token ID(s).
    fn eos_token_id(&self) -> Vec<u32>;

    /// Warm up the model with a small forward pass.
    fn warmup(&mut self);

    // ── KV cache swap (for concurrent sequence serving) ───────

    /// Whether this backend supports extracting and restoring KV caches.
    fn supports_kv_swap(&self) -> bool {
        false
    }

    /// Extract per-layer KV caches from the model.
    fn get_kv_caches(&self) -> SequenceKvCaches {
        vec![]
    }

    /// Restore per-layer KV caches into the model.
    fn set_kv_caches(&mut self, _caches: SequenceKvCaches) {}

    /// Compute bytes held by the model's active KV caches without copying.
    /// Used for memory tracking without the overhead of `get_kv_caches()`.
    fn active_kv_cache_bytes(&self) -> u64 {
        0
    }

    /// Bytes of KV cache one sequence consumes per generated token, if this
    /// backend's cache layout is a simple per-layer rate. `None` for
    /// architectures where a generic formula would be wrong (hybrid
    /// recurrent/linear-attention layers, cross-layer KV sharing) or not yet
    /// computed for this backend.
    fn kv_bytes_per_token(&self) -> Option<u64> {
        None
    }

    /// Re-run expert promotion with an updated policy (e.g. after
    /// `derive_safe_max_seq_len` tightens `max_seq_len`). The default
    /// implementation is a no-op — only `MoE` backends override this.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying promotion fails for reasons
    /// unrelated to out-of-memory (which is caught inside the model).
    fn re_promote_experts(&mut self, _policy: &ExpertPromotionPolicy) -> Result<()> {
        Ok(())
    }

    // ── Batch decode (GPU-efficient concurrent serving) ───────

    /// Whether this backend supports batched decoding.
    fn supports_batch_decode(&self) -> bool {
        false
    }

    /// Pad and load per-sequence KV caches for batched decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if this backend doesn't support batch decode.
    fn setup_batch_decode(
        &mut self,
        _seq_kv_caches: &[SequenceKvCaches],
        _extra_room: usize,
    ) -> crane_core::Result<(Vec<usize>, usize)> {
        bail!("Batch decode not supported by this backend")
    }

    /// Run one batched decode step.
    ///
    /// # Errors
    ///
    /// Returns an error if this backend doesn't support batch decode.
    fn step_batch_decode(
        &mut self,
        _input_ids: &Tensor,
        _positions: &[usize],
        _attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> crane_core::Result<Tensor> {
        bail!("Batch decode not supported by this backend")
    }

    /// Extract per-sequence KV caches from batched state.
    ///
    /// # Errors
    ///
    /// Returns an error if this backend doesn't support batch decode.
    fn extract_batch_kv(
        &mut self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _rounds_done: usize,
    ) -> crane_core::Result<Vec<SequenceKvCaches>> {
        bail!("Batch decode not supported by this backend")
    }

    /// Build attention mask for batched decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if this backend doesn't support batch decode.
    fn build_batch_decode_mask(
        &self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _max_total_width: usize,
    ) -> crane_core::Result<Option<Tensor>> {
        bail!("Batch decode not supported by this backend")
    }
}

// ─────────────────────────────────────────────────────────────
//  Gemma 4 Backend
// ─────────────────────────────────────────────────────────────

pub struct Gemma4Backend {
    pub model: crane_core::models::gemma4::Model,
}

impl Gemma4Backend {
    /// # Errors
    ///
    /// Returns an error if the model fails to load from `model_path`.
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::gemma4::ModelFormat,
    ) -> Result<Self> {
        let model =
            crane_core::models::gemma4::Model::new_with_format(model_path, device, dtype, format)?;
        Ok(Self { model })
    }
}

impl ModelBackend for Gemma4Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        let tok = &self.model.tokenizer.tokenizer;
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<end_of_turn>") {
            ids.push(id);
        }
        if let Some(id) = tok.token_to_id("<eos>") {
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(1); // Gemma default EOS
        }
        ids
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // KV swap and batch decode not supported due to KV cache sharing architecture.
    // supports_kv_swap() defaults to false, capping OAI server to 1 concurrent sequence.
    // supports_batch_decode() defaults to false.
}

// ─────────────────────────────────────────────────────────────
//  Hunyuan Dense Backend
// ─────────────────────────────────────────────────────────────

pub struct HunyuanBackend {
    pub model: crane_core::models::hunyuan_dense::Model,
}

impl HunyuanBackend {
    /// # Errors
    ///
    /// Returns an error if the model fails to load from `model_path`.
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::hunyuan_dense::ModelFormat,
    ) -> Result<Self> {
        let model = crane_core::models::hunyuan_dense::Model::new_with_format(
            model_path, device, dtype, format,
        )?;
        Ok(Self { model })
    }
}

impl ModelBackend for HunyuanBackend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        vec![120_020]
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> crane_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> crane_core::Result<Tensor> {
        self.model.step_batch_decode_with_input_ids(
            input_ids,
            positions,
            attention_mask,
            batch_kv_info,
        )
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> crane_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> crane_core::Result<Option<Tensor>> {
        crane_core::models::hunyuan_dense::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 2.5 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen25Backend {
    pub model: crane_core::models::qwen25::Model,
    dtype: DType,
}

impl Qwen25Backend {
    /// # Errors
    ///
    /// Returns an error if the model fails to load from `model_path`.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen25::Model::new(model_path, device, dtype)?;
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl ModelBackend for Qwen25Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }

    fn num_layers(&self) -> usize {
        0 // KV swap not supported; vector is unused
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        self.model
            .tokenizer
            .tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.model.tokenizer.tokenizer.token_to_id("<|im_end|>"))
            .map_or_else(|| vec![151_643], |id| vec![id])
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }
}

// ─────────────────────────────────────────────────────────────
//  MiniCPM5 Backend
// ─────────────────────────────────────────────────────────────

/// Engine wrapper around `crane_core::models::minicpm5::Model`.
///
/// Only single-sequence forward is supported (no KV swap, no batch decode) —
/// same tier as [`Qwen25Backend`].
pub struct Minicpm5Backend {
    pub model: crane_core::models::minicpm5::Model,
    dtype: DType,
}

impl Minicpm5Backend {
    /// # Errors
    ///
    /// Returns an error if the model fails to load from `model_path`.
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::minicpm5::ModelFormat,
    ) -> Result<Self> {
        let model = crane_core::models::minicpm5::Model::new_with_format(
            model_path, device, dtype, format,
        )?;
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl ModelBackend for Minicpm5Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }

    fn num_layers(&self) -> usize {
        0 // KV swap not supported; vector is unused
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        let ids = self.model.eos_token_ids();
        if !ids.is_empty() {
            return ids.to_vec();
        }
        let tok = &self.model.tokenizer.tokenizer;
        tok.token_to_id("<|im_end|>")
            .or_else(|| tok.token_to_id("<|endoftext|>"))
            .map_or_else(|| vec![1], |id| vec![id])
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 3.5 Backend
// ─────────────────────────────────────────────────────────────

/// Engine wrapper around `crane_core::models::qwen3_5::Model`.
///
/// Only single-sequence forward is supported (no KV swap, no batch decode).
/// The hybrid full-attention / GDN layers share a single set of caches; the
/// engine treats them as opaque per-request state.
pub struct Qwen3_5Backend {
    pub model: crane_core::models::qwen3_5::Model,
    #[allow(dead_code)]
    dtype: DType,
}

impl Qwen3_5Backend {
    /// # Errors
    ///
    /// Returns an error if the model fails to load from `model_path`.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_options(
            model_path,
            device,
            dtype,
            crane_core::models::qwen3_5::ModelFormat::Auto,
            None,
        )
    }

    /// `quant` requests in-situ quantization (`--quant` / `CRANE_ISQ`);
    /// `None` falls back to the `CRANE_ISQ` env var inside the model loader.
    pub fn new_with_options(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::qwen3_5::ModelFormat,
        quant: Option<crane_core::GgmlDType>,
    ) -> Result<Self> {
        let model = match quant {
            Some(dt) => crane_core::models::qwen3_5::Model::new_with_options(
                model_path,
                device,
                dtype,
                format,
                Some(dt),
            )?,
            None => crane_core::models::qwen3_5::Model::new_with_format(
                model_path, device, dtype, format,
            )?,
        };
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl ModelBackend for Qwen3_5Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        // The underlying model already returns anyhow::Result — no conversion needed.
        self.model.forward_step(input_ids, start_pos)
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache()
    }

    fn snapshot_state(&self) -> Option<crane_core::models::qwen3_5::StateSnapshot> {
        Some(self.model.snapshot_state())
    }

    fn restore_state(&mut self, snap: &crane_core::models::qwen3_5::StateSnapshot) -> Result<()> {
        self.model.restore_state(snap)
    }

    fn num_layers(&self) -> usize {
        // The KV-cache vector sizing is a no-op for this backend — we always
        // reset full-attention state implicitly via GDN-cache reset, and the
        // engine caps `max_concurrent` to 1 since `supports_kv_swap` is false.
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        // Prefer the ids the model read from generation_config.json / GGUF
        // metadata (Qwen 3.5 uses a multi-id EOS set, e.g. [248044, 248046]).
        let ids = self.model.eos_token_ids().to_vec();
        if !ids.is_empty() {
            return ids;
        }
        // Fallback: resolve the ChatML stop tokens through the tokenizer.
        let tok = &self.model.tokenizer.tokenizer;
        ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .collect()
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // supports_kv_swap defaults to false → engine caps max_concurrent to 1.
    // Batch decode is not yet implemented for hybrid layer types.
}

// ─────────────────────────────────────────────────────────────
//  Qwen 3 Backend
// ─────────────────────────────────────────────────────────────

/// `MoE` expert GPU-promotion policy, resolved from CLI flags in
/// `crane-serve/src/lib.rs`. Never enters `crane-core` — the model only
/// sees the resolved scalars via `Model::promote_experts_to_gpu`.
pub struct ExpertPromotionPolicy {
    /// VRAM ceiling for expert weights, in bytes.
    pub vram_ceiling_bytes: u64,
    /// Maximum concurrent sequences the engine will serve, used to
    /// estimate KV cache VRAM. `None` falls back to a conservative default.
    pub max_concurrent: Option<usize>,
    /// Maximum tokens (prompt + completion) per sequence, used to estimate
    /// KV cache VRAM. `None` or `Some(0)` falls back to a conservative
    /// default.
    pub max_seq_len: Option<usize>,
}

pub struct Qwen3Backend {
    pub model: crane_core::models::qwen3::Model,
    #[allow(dead_code)]
    dtype: DType,
}

impl Qwen3Backend {
    /// `promotion` is `Some` only when `devices.expert` starts on CPU and
    /// the runtime wants the model to attempt promoting experts to GPU
    /// after construction, once real VRAM headroom is known.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load from `model_path`, or if
    /// expert promotion fails outright (not counting an out-of-memory
    /// promotion attempt, which is caught and logged inside the model).
    pub fn new(
        model_path: &str,
        devices: &DeviceAssignment,
        dtype: &DType,
        promotion: Option<&ExpertPromotionPolicy>,
    ) -> Result<Self> {
        let mut model = crane_core::models::qwen3::Model::loader(model_path, &devices.main, dtype)
            .expert_device(&devices.expert)
            .build()?;
        if let Some(p) = promotion {
            model.promote_experts_to_gpu(p.vram_ceiling_bytes, p.max_concurrent, p.max_seq_len)?;
        }
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl ModelBackend for Qwen3Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        // Qwen3 chat models stop at <|im_end|> (151645).
        // Also include <|endoftext|> (151643) as a fallback.
        let tok = &self.model.tokenizer.tokenizer;
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") {
            ids.push(id);
        }
        if let Some(id) = tok.token_to_id("<|endoftext|>") {
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(151_645);
            ids.push(151_643);
        }
        ids
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn kv_bytes_per_token(&self) -> Option<u64> {
        Some(self.model.kv_bytes_per_token())
    }

    fn re_promote_experts(&mut self, policy: &ExpertPromotionPolicy) -> Result<()> {
        self.model.promote_experts_to_gpu(
            policy.vram_ceiling_bytes,
            policy.max_concurrent,
            policy.max_seq_len,
        )
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> crane_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> crane_core::Result<Tensor> {
        self.model.step_batch_decode_with_input_ids(
            input_ids,
            positions,
            attention_mask,
            batch_kv_info,
        )
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> crane_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> crane_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}
