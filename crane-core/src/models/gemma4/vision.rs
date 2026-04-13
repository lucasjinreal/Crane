//! Gemma 4 vision encoder.
//!
//! Architecture:
//! - Patch embedding with learned 2D position table
//! - 16-layer bidirectional transformer with 2D RoPE
//! - Gemma-style 4 norms per layer, QK+V norms
//! - Clipped linears (clip bounds stored but only applied during training)
//! - Spatial average pooling
//! - RMSNorm + Linear projection to text hidden_size

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::rotary_emb::rope;
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

use super::modeling::rms_normalize;

// ── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub patch_size: usize,
    #[serde(default = "default_position_embedding_size")]
    pub position_embedding_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default = "default_output_length")]
    pub default_output_length: usize,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    #[serde(default)]
    pub rope_parameters: Option<VisionRopeParams>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionRopeParams {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_head_dim() -> usize { 64 }
fn default_position_embedding_size() -> usize { 10240 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_output_length() -> usize { 280 }
fn default_pooling_kernel_size() -> usize { 3 }
fn default_rope_theta() -> f64 { 100.0 }

impl Gemma4VisionConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|rp| rp.rope_theta)
            .unwrap_or(100.0)
    }
}

// ── Clipped Linear ──────────────────────────────────────────────────────
// Weights are at `.linear.weight`; clip bounds are stored but unused at inference.

fn clipped_linear(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<Linear> {
    linear_no_bias(in_d, out_d, vb.pp("linear"))
}

// ── 2D RoPE for Vision ─────────────────────────────────────────────────

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(config: &Gemma4VisionConfig, device: &Device) -> Result<Self> {
        let theta = config.rope_theta();
        let spatial_dim = config.head_dim / 2; // half head_dim per spatial dimension
        let inv: Vec<f32> = (0..spatial_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / spatial_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;
        Ok(Self { inv_freq })
    }

    /// Compute 2D cos/sin from pixel position IDs.
    ///
    /// position_ids: [B, num_patches, 2] (x, y positions as i64)
    /// Returns: (cos, sin) each [B, num_patches, head_dim//2] per spatial dim,
    ///          concatenated to [B, num_patches, head_dim//2] for each spatial half.
    fn forward(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let freq_dim = self.inv_freq.dim(0)?; // spatial_dim / 2

        let mut all_cos = Vec::new();
        let mut all_sin = Vec::new();

        for dim_idx in 0..2 {
            // Extract positions for this spatial dimension: [B, num_patches]
            let dim_pos = position_ids.narrow(D::Minus1, dim_idx, 1)?.squeeze(D::Minus1)?;
            let dim_pos = dim_pos.to_dtype(DType::F32)?;

            // freqs = positions × inv_freq: [B, num_patches, freq_dim]
            let inv = self.inv_freq.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, freq_dim]
            let pos = dim_pos.unsqueeze(D::Minus1)?; // [B, num_patches, 1]
            let freqs = pos.broadcast_mul(&inv)?; // [B, num_patches, freq_dim]

            let cos = freqs.cos()?.to_dtype(dtype)?;
            let sin = freqs.sin()?.to_dtype(dtype)?;
            all_cos.push(cos);
            all_sin.push(sin);
        }

        // Concatenate X and Y: [B, num_patches, freq_dim*2 = head_dim/2]
        let cos = Tensor::cat(&all_cos, D::Minus1)?.contiguous()?;
        let sin = Tensor::cat(&all_sin, D::Minus1)?.contiguous()?;
        Ok((cos, sin))
    }
}

/// Apply 2D RoPE to Q or K.
///
/// x: [B, H, num_patches, head_dim]
/// cos/sin: [B, num_patches, head_dim/2]
///
/// Split head_dim into two halves (X, Y). Apply standard RoPE to each half
/// independently using the corresponding spatial frequencies.
fn apply_2d_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;

    // Split into X and Y halves
    let x_part = x.narrow(D::Minus1, 0, half)?;
    let y_part = x.narrow(D::Minus1, half, half)?;

    // Split cos/sin for each spatial dim: each is [B, S, freq_dim]
    let freq_dim = cos.dim(D::Minus1)? / 2;
    let cos_x = cos.narrow(D::Minus1, 0, freq_dim)?;
    let sin_x = sin.narrow(D::Minus1, 0, freq_dim)?;
    let cos_y = cos.narrow(D::Minus1, freq_dim, freq_dim)?;
    let sin_y = sin.narrow(D::Minus1, freq_dim, freq_dim)?;

    // Apply standard RoPE to each half (ensure all inputs contiguous for rope kernel)
    let x_rot = rope(&x_part.contiguous()?, &cos_x.contiguous()?, &sin_x.contiguous()?)?;
    let y_rot = rope(&y_part.contiguous()?, &cos_y.contiguous()?, &sin_y.contiguous()?)?;

    Tensor::cat(&[&x_rot, &y_rot], D::Minus1)
}

// ── Patch Embeddings ────────────────────────────────────────────────────

struct VisionPatchEmbedder {
    input_proj: Linear,
    position_embedding_table: Tensor, // [2, position_embedding_size, hidden_size]
    position_embedding_size: usize,
}

impl VisionPatchEmbedder {
    fn new(config: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let input_proj = linear_no_bias(patch_dim, config.hidden_size, vb.pp("input_proj"))?;
        let position_embedding_table = vb.get(
            (2, config.position_embedding_size, config.hidden_size),
            "position_embedding_table",
        )?;
        Ok(Self {
            input_proj,
            position_embedding_table,
            position_embedding_size: config.position_embedding_size,
        })
    }

    /// Compute 2D position embeddings from position IDs.
    ///
    /// pixel_position_ids: [B, num_patches, 2] (i64)
    /// padding_positions: [B, num_patches] (bool)
    /// Compute 2D position embeddings using index_select from the learned table.
    ///
    /// pixel_position_ids: [B, num_patches, 2] (i64)
    /// padding_positions: [B, num_patches] (bool-like)
    fn position_embeddings(
        &self,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, num_patches, _) = pixel_position_ids.dims3()?;
        let clamped = pixel_position_ids.clamp(0i64, self.position_embedding_size as i64 - 1)?;

        // table shape: [2, position_embedding_size, hidden_size]
        // For each spatial dim, gather embeddings by position index
        let mut pos_emb_sum: Option<Tensor> = None;
        for dim_idx in 0..2usize {
            // table_dim: [position_embedding_size, hidden_size]
            let table_dim = self.position_embedding_table.get(dim_idx)?;
            // pos_ids: [B, num_patches] as i64
            let pos_ids = clamped
                .narrow(D::Minus1, dim_idx, 1)?
                .squeeze(D::Minus1)?
                .to_dtype(candle_core::DType::U32)?;
            // Flatten batch for index_select, then reshape
            let pos_flat = pos_ids.flatten_all()?; // [B * num_patches]
            let emb = table_dim.index_select(&pos_flat, 0)?; // [B*num_patches, hidden_size]
            let emb = emb.reshape((b_sz, num_patches, ()))?;

            pos_emb_sum = Some(match pos_emb_sum {
                Some(acc) => (acc + emb)?,
                None => emb,
            });
        }
        let pos_emb = pos_emb_sum.unwrap();

        // Zero out padding positions
        let mask = padding_positions
            .unsqueeze(D::Minus1)?
            .to_dtype(pos_emb.dtype())?;
        pos_emb.broadcast_mul(&(1.0 - mask)?)
    }

    fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        // Normalize: 2 * (x - 0.5)
        let pixel_values = ((pixel_values * 2.0)? - 1.0)?;
        let hidden_states = self.input_proj.forward(
            &pixel_values.to_dtype(self.position_embedding_table.dtype())?,
        )?;
        let pos_emb = self.position_embeddings(pixel_position_ids, padding_positions)?;
        hidden_states + pos_emb
    }
}

// ── Vision Attention ────────────────────────────────────────────────────

struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rms_norm_eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(config: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hd = config.head_dim;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;

        let q_proj = clipped_linear(config.hidden_size, nh * hd, vb.pp("q_proj"))?;
        let k_proj = clipped_linear(config.hidden_size, nkv * hd, vb.pp("k_proj"))?;
        let v_proj = clipped_linear(config.hidden_size, nkv * hd, vb.pp("v_proj"))?;
        let o_proj = clipped_linear(nh * hd, config.hidden_size, vb.pp("o_proj"))?;

        let q_norm = candle_nn::rms_norm(hd, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(hd, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rms_norm_eps: config.rms_norm_eps,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(hidden_states)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(hidden_states)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // QK norms before RoPE
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        // V norm (no learnable scale)
        let v = rms_normalize(&v, self.rms_norm_eps)?;

        // 2D RoPE
        let q = apply_2d_rope(&q, cos, sin)?;
        let k = apply_2d_rope(&k, cos, sin)?;

        // GQA expand
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            v
        };

        // Bidirectional attention (no causal mask), scaling=1.0
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

// ── Vision MLP ──────────────────────────────────────────────────────────

struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionMlp {
    fn new(config: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = clipped_linear(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = clipped_linear(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = clipped_linear(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Vision Encoder Layer ────────────────────────────────────────────────

struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl VisionEncoderLayer {
    fn new(config: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(config, vb.pp("self_attn"))?;
        let mlp = VisionMlp::new(config, vb.pp("mlp"))?;
        let eps = config.rms_norm_eps;
        let hs = config.hidden_size;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("post_attention_layernorm"))?,
            pre_feedforward_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("pre_feedforward_layernorm"))?,
            post_feedforward_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("post_feedforward_layernorm"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Pre-norm → attention → post-norm → residual
        let residual = hidden_states;
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self.self_attn.forward(&h, cos, sin)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let hidden_states = (residual + h)?;

        // Pre-norm → MLP → post-norm → residual
        let residual = &hidden_states;
        let h = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        residual + h
    }
}

// ── Vision Pooler ───────────────────────────────────────────────────────

struct VisionPooler {
    root_hidden_size: f64,
}

impl VisionPooler {
    fn new(config: &Gemma4VisionConfig) -> Self {
        Self {
            root_hidden_size: (config.hidden_size as f64).sqrt(),
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
        output_length: usize,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;

        // Zero out padding positions
        let mask = padding_positions.unsqueeze(D::Minus1)?.to_dtype(hidden_states.dtype())?;
        let hidden_states = hidden_states.broadcast_mul(&(1.0 - mask)?)?;

        let hidden_states = if seq_len != output_length {
            self.avg_pool_by_positions(&hidden_states, pixel_position_ids, output_length)?
        } else {
            hidden_states
        };

        // Scale by sqrt(hidden_size)
        hidden_states * self.root_hidden_size
    }

    fn avg_pool_by_positions(
        &self,
        hidden_states: &Tensor,
        pixel_position_ids: &Tensor,
        length: usize,
    ) -> Result<Tensor> {
        let input_len = hidden_states.dim(1)?;
        let k = ((input_len as f64 / length as f64).sqrt()) as usize;
        let k_squared = k * k;

        // Compute kernel indices from position IDs
        let clamped = pixel_position_ids.clamp(0i64, i64::MAX)?;

        // Get max X position per batch
        let pos_x = clamped.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let max_x = pos_x.max_keepdim(D::Minus1)?;
        let max_x = (max_x + 1.0)?;

        // kernel_idx = floor(x/k) + floor(max_x/k) * floor(y/k)
        let pos_y = clamped.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
        let pos_x = pos_x.to_dtype(DType::F32)?;
        let pos_y = pos_y.to_dtype(DType::F32)?;
        let max_x = max_x.to_dtype(DType::F32)?;
        let k_f = k as f64;

        let kx = (pos_x / k_f)?.floor()?;
        let ky = (pos_y / k_f)?.floor()?;
        let stride = (max_x / k_f)?.floor()?;
        let kernel_idx = (kx + ky.broadcast_mul(&stride)?)?.to_dtype(DType::I64)?;

        // Build pooling weights via scatter: [B, input_len, length]
        let b_sz = hidden_states.dim(0)?;
        let idx = kernel_idx.clamp(0i64, length as i64 - 1)?.to_dtype(DType::U32)?;
        let mut weights_data = vec![0f32; b_sz * input_len * length];
        let idx_flat: Vec<u32> = idx.flatten_all()?.to_vec1()?;
        for b in 0..b_sz {
            for s in 0..input_len {
                let i = idx_flat[b * input_len + s] as usize;
                if i < length {
                    weights_data[b * input_len * length + s * length + i] =
                        1.0 / k_squared as f32;
                }
            }
        }
        let weights = Tensor::from_vec(
            weights_data,
            (b_sz, input_len, length),
            hidden_states.device(),
        )?;

        // Pool: [B, length, hidden_size] = weights^T @ hidden_states
        let pooled = weights
            .transpose(1, 2)?
            .matmul(&hidden_states.to_dtype(DType::F32)?)?;

        pooled.to_dtype(hidden_states.dtype())
    }
}

// ── Full Vision Model ───────────────────────────────────────────────────

pub struct Gemma4VisionModel {
    patch_embedder: VisionPatchEmbedder,
    layers: Vec<VisionEncoderLayer>,
    rotary_emb: VisionRotaryEmbedding,
    pooler: VisionPooler,
    config: Gemma4VisionConfig,
    dtype: DType,
}

impl Gemma4VisionModel {
    pub fn new(config: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let patch_embedder = VisionPatchEmbedder::new(config, vb.pp("patch_embedder"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = vb.pp("encoder").pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(VisionEncoderLayer::new(config, layers_vb.pp(i))?);
        }

        let rotary_emb = VisionRotaryEmbedding::new(config, vb.device())?;
        let pooler = VisionPooler::new(config);

        Ok(Self {
            patch_embedder,
            layers,
            rotary_emb,
            pooler,
            config: config.clone(),
            dtype,
        })
    }

    /// Forward pass.
    ///
    /// * `pixel_values`: [B, num_patches, 3*patch_size^2] — flattened patch pixels
    /// * `pixel_position_ids`: [B, num_patches, 2] — (x, y) positions as i64
    /// * `padding_positions`: [B, num_patches] — bool mask (true = padding)
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.patch_embedder.forward(
            pixel_values,
            pixel_position_ids,
            padding_positions,
        )?;

        // Compute 2D RoPE cos/sin from position IDs
        let (cos, sin) = self.rotary_emb.forward(pixel_position_ids, self.dtype)?;

        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin)?;
        }

        // Pool: output_length = num_real_patches / pooling_kernel²
        let num_patches = hidden_states.dim(1)?;
        let padding_sum: f32 = padding_positions.sum_all()?.to_scalar()?;
        let num_real = num_patches - padding_sum as usize;
        let pk2 = self.config.pooling_kernel_size.pow(2);
        let output_length = num_real / pk2;

        self.pooler.forward(
            &hidden_states,
            pixel_position_ids,
            padding_positions,
            output_length,
        )
    }
}

// ── Multimodal Embedder (vision → text projection) ──────────────────────

pub struct Gemma4MultimodalEmbedder {
    projection: Linear,
    rms_norm_eps: f64,
}

impl Gemma4MultimodalEmbedder {
    pub fn new(
        vision_hidden_size: usize,
        text_hidden_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let projection = linear_no_bias(
            vision_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;
        Ok(Self {
            projection,
            rms_norm_eps,
        })
    }

    pub fn forward(&self, vision_features: &Tensor) -> Result<Tensor> {
        // RMSNorm without learnable scale
        let normed = rms_normalize(vision_features, self.rms_norm_eps)?;
        self.projection.forward(&normed)
    }
}

// ── Image Preprocessing ─────────────────────────────────────────────────

/// Preprocessing parameters for Gemma4 vision.
pub struct ImagePreprocessConfig {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
    pub rescale_factor: f64,
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            rescale_factor: 1.0 / 255.0,
        }
    }
}

impl ImagePreprocessConfig {
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

/// Result of preprocessing an image for Gemma4 vision.
pub struct PreprocessedImage {
    /// Flattened patches: [1, num_patches_padded, patch_size² * 3]
    pub pixel_values: Tensor,
    /// 2D position IDs: [1, num_patches_padded, 2] as i64
    pub pixel_position_ids: Tensor,
    /// Padding mask: [1, num_patches_padded] (1.0 = padding, 0.0 = real)
    pub padding_positions: Tensor,
    /// Number of image placeholder tokens to insert in the prompt
    pub num_image_tokens: usize,
}

/// Compute target size preserving aspect ratio for Gemma4 vision.
///
/// Ensures the image is resized so that:
/// - Dimensions are multiples of patch_size
/// - Total patches <= max_patches
fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
) -> (usize, usize) {
    let aspect = width as f64 / height as f64;

    // Start from max_patches and find dimensions
    let total_pixels = max_patches * patch_size * patch_size;
    let mut target_h = (total_pixels as f64 / aspect).sqrt();
    let mut target_w = target_h * aspect;

    // Round to nearest patch_size multiple
    target_h = ((target_h / patch_size as f64).floor()) * patch_size as f64;
    target_w = ((target_w / patch_size as f64).floor()) * patch_size as f64;

    let mut h = target_h.max(patch_size as f64) as usize;
    let mut w = target_w.max(patch_size as f64) as usize;

    // Ensure total patches doesn't exceed max
    while (h / patch_size) * (w / patch_size) > max_patches {
        if h >= w {
            h -= patch_size;
        } else {
            w -= patch_size;
        }
    }

    (h.max(patch_size), w.max(patch_size))
}

/// Preprocess an image for the Gemma4 vision encoder.
///
/// Takes a raw image tensor [C, H, W] with values in [0, 1] and returns
/// patchified, padded tensors ready for the vision model.
pub fn preprocess_image(
    image: &Tensor, // [3, H, W] float, values in [0, 1]
    config: &ImagePreprocessConfig,
    device: &Device,
) -> Result<PreprocessedImage> {
    let (channels, height, width) = image.dims3()?;
    assert_eq!(channels, 3);

    let max_patches = config.max_patches();
    let ps = config.patch_size;

    // Compute target size
    let (target_h, target_w) = get_aspect_ratio_preserving_size(height, width, ps, max_patches);

    // Resize to target
    let resized = image
        .unsqueeze(0)?
        .upsample_bilinear2d(target_h, target_w, false)?
        .squeeze(0)?; // [3, target_h, target_w]

    // Patchify: [3, H, W] → [num_patches, patch_size² * 3]
    let num_patches_h = target_h / ps;
    let num_patches_w = target_w / ps;
    let num_patches = num_patches_h * num_patches_w;

    let patched = resized
        .reshape((3, num_patches_h, ps, num_patches_w, ps))?
        .permute((1, 3, 2, 4, 0))? // [H/P, W/P, P, P, 3]
        .contiguous()?
        .reshape((num_patches, ps * ps * 3))?; // [num_patches, P²*3]

    // Create 2D position IDs: [num_patches, 2]
    let mut pos_data = vec![0i64; num_patches * 2];
    for py in 0..num_patches_h {
        for px in 0..num_patches_w {
            let idx = py * num_patches_w + px;
            pos_data[idx * 2] = px as i64;     // x position
            pos_data[idx * 2 + 1] = py as i64; // y position
        }
    }
    let positions = Tensor::from_vec(pos_data, (num_patches, 2), device)?;

    // Pad to max_patches
    let pad_len = max_patches - num_patches;
    let (padded_patches, padded_positions, padding_mask) = if pad_len > 0 {
        let patch_pad = Tensor::zeros((pad_len, ps * ps * 3), patched.dtype(), device)?;
        let padded_patches = Tensor::cat(&[&patched, &patch_pad], 0)?;

        let pos_pad = Tensor::full(-1i64, (pad_len, 2), device)?;
        let padded_positions = Tensor::cat(&[&positions, &pos_pad], 0)?;

        // Padding mask: 0.0 for real patches, 1.0 for padding
        let mut mask_data = vec![0f32; max_patches];
        for i in num_patches..max_patches {
            mask_data[i] = 1.0;
        }
        let padding_mask = Tensor::from_vec(mask_data, max_patches, device)?;

        (padded_patches, padded_positions, padding_mask)
    } else {
        let mask = Tensor::zeros(num_patches, DType::F32, device)?;
        (patched, positions, mask)
    };

    // Add batch dimension
    let pixel_values = padded_patches.unsqueeze(0)?;
    let pixel_position_ids = padded_positions.unsqueeze(0)?;
    let padding_positions = padding_mask.unsqueeze(0)?;

    // Number of output tokens after pooling
    // Output tokens = real patches / pooling_kernel² (not max_soft_tokens)
    let pk2 = config.pooling_kernel_size * config.pooling_kernel_size;
    let num_image_tokens = num_patches / pk2;

    Ok(PreprocessedImage {
        pixel_values,
        pixel_position_ids,
        padding_positions,
        num_image_tokens,
    })
}

/// Load and preprocess an image file for Gemma4 vision.
pub fn load_and_preprocess_image(
    path: &std::path::Path,
    config: &ImagePreprocessConfig,
    device: &Device,
) -> Result<PreprocessedImage> {
    let img = image::ImageReader::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open image: {e}")))?
        .decode()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to decode image: {e}")))?
        .to_rgb8();

    let (w, h) = img.dimensions();
    let raw = img.into_raw();

    // Convert to [3, H, W] float tensor in [0, 1]
    let tensor = (Tensor::from_vec(raw, (h as usize, w as usize, 3), device)?
        .to_dtype(DType::F32)?
        .permute((2, 0, 1))?
        * config.rescale_factor)?;

    preprocess_image(&tensor, config, device)
}
