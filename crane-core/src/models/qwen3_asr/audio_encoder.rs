//! Conv2d frontend, windowed transformer, and sinusoidal positional encoding
//! for the Qwen3-ASR audio encoder ("`AuT`").
//!
//! Converts a log-mel spectrogram `[1, n_mels, n_frames]` from
//! [`WhisperFeatureExtractor`](super::feature_extractor::WhisperFeatureExtractor)
//! into `[1, total_tokens, output_dim]`-ready token embeddings (the encoder's
//! own output is `[1, total_tokens, d_model]`; the multi-modal projector,
//! not yet implemented, maps `d_model` to `output_dim`):
//!
//! 1. Split mel frames into non-overlapping chunks of
//!    [`FRAMES_PER_WINDOW`](super::feature_extractor) (100) frames along the
//!    time axis.
//! 2. Each chunk passes independently through three stride-2 `Conv2d` layers
//!    with GELU, reducing `[1, 1, n_mels, time]` to
//!    `[1, downsample_hidden_size, mel', time']`.
//! 3. Permute and flatten to `[1, time', downsample_hidden_size * mel']`,
//!    then project via `conv_out` to `[1, time', d_model]`.
//! 4. Add a fixed sinusoidal positional encoding (Whisper-style, not
//!    learned), sliced to the chunk's actual token count.
//! 5. Concatenate all chunk outputs along the time axis ([`AudioEncoderFrontend`]).
//! 6. Run the concatenated sequence through `encoder_layers` pre-norm
//!    LayerNorm+GELU transformer blocks, with self-attention masked into
//!    block-diagonal ~8s windows (`n_window_infer` raw frames per block —
//!    a hard correctness requirement, not an optional streaming mode), then
//!    a final `ln_post` ([`AudioEncoder`]).

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Conv2dConfig, VarBuilder};

use super::config::AudioConfig;
use super::feature_extractor::{FRAMES_PER_WINDOW, TOKENS_PER_WINDOW, chunk_split, conv_output_len};
use crate::models::with_tracing;

/// `LayerNorm` epsilon for the audio encoder's transformer layers and
/// `ln_post`. Not present in the `-hf` checkpoint's `audio_config` JSON;
/// hardcoded to `1e-5`, matching `torch.nn.LayerNorm`'s default and the
/// Whisper/BART-style encoder this architecture is based on (§3).
const LAYER_NORM_EPS: f64 = 1e-5;

/// Builds a fixed sinusoidal positional encoding table of shape
/// `[max_positions, d_model]`: `[sin(scaled_time) || cos(scaled_time)]`
/// concatenated (not interleaved) along the last dimension, matching the
/// Whisper/BART-style encoding the Qwen3-ASR reference computes as a
/// non-persistent buffer (never saved to the checkpoint).
fn build_sinusoidal_table(
    max_positions: usize,
    d_model: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    debug_assert!(
        d_model >= 4 && d_model.is_multiple_of(2),
        "d_model must be even and >= 4, got {d_model}"
    );
    let half_dim = d_model / 2;
    // half_dim is at most a few hundred (d_model <= a few thousand in
    // practice), well within f64's exact integer range.
    #[allow(clippy::cast_precision_loss)]
    let log_timescale_incr = 10_000_f64.ln() / (half_dim as f64 - 1.0);

    let mut table = vec![0f32; max_positions * d_model];
    for pos in 0..max_positions {
        for i in 0..half_dim {
            // pos/i are bounded by max_positions/half_dim, both tiny
            // relative to f64's exact integer range.
            #[allow(clippy::cast_precision_loss)]
            let scaled = pos as f64 * (-log_timescale_incr * i as f64).exp();
            // sin/cos are bounded to [-1, 1], safe to narrow to f32.
            #[allow(clippy::cast_possible_truncation)]
            let sin_val = scaled.sin() as f32;
            #[allow(clippy::cast_possible_truncation)]
            let cos_val = scaled.cos() as f32;
            table[pos * d_model + i] = sin_val;
            table[pos * d_model + half_dim + i] = cos_val;
        }
    }
    Tensor::from_vec(table, (max_positions, d_model), device)?.to_dtype(dtype)
}

/// Conv2d frontend and sinusoidal positional encoding for the Qwen3-ASR
/// audio encoder.
///
/// Processes a log-mel spectrogram through three stride-2 `Conv2d` layers,
/// a bias-free linear projection, and per-chunk sinusoidal positional
/// encoding, producing token embeddings for the (not-yet-implemented)
/// transformer encoder layers.
///
/// The supplied [`VarBuilder`] must be scoped to the audio tower prefix
/// (checkpoint path `model.audio_tower`). Loaded tensors:
/// - `conv2d1.{weight,bias}` — `[downsample_hidden_size, 1, 3, 3]`, `[downsample_hidden_size]`
/// - `conv2d2.{weight,bias}` — `[downsample_hidden_size, downsample_hidden_size, 3, 3]`, `[downsample_hidden_size]`
/// - `conv2d3.{weight,bias}` — `[downsample_hidden_size, downsample_hidden_size, 3, 3]`, `[downsample_hidden_size]`
/// - `conv_out.weight` — `[d_model, downsample_hidden_size * post_conv_mel_bins]` (no bias)
#[derive(Debug)]
pub struct AudioEncoderFrontend {
    /// First conv: 1 input channel -> `downsample_hidden_size` channels.
    conv2d1: with_tracing::Conv2d,
    /// Second conv: `downsample_hidden_size` -> `downsample_hidden_size`.
    conv2d2: with_tracing::Conv2d,
    /// Third conv: `downsample_hidden_size` -> `downsample_hidden_size`.
    conv2d3: with_tracing::Conv2d,
    /// Bias-free linear projection from the flattened conv output to `d_model`.
    conv_out: with_tracing::Linear,
    /// Activation applied after each conv layer (always GELU on released checkpoints).
    activation: Activation,
    /// Precomputed sinusoidal positional encoding, shape `[max_position_embeddings, d_model]`.
    pos_embed: Tensor,
    /// Flattened conv output width (`downsample_hidden_size * post_conv_mel_bins`),
    /// i.e. the input dimension to `conv_out`.
    conv_flat_dim: usize,
    /// Maximum number of full-size chunks batched through the conv stack in
    /// one call, matching `AudioConfig::conv_chunksize` (memory bound only).
    conv_chunksize: usize,
}

impl AudioEncoderFrontend {
    /// Builds the conv frontend from checkpoint weights.
    ///
    /// `vb` must already be scoped to the audio tower prefix (e.g.
    /// `vb.pp("model").pp("audio_tower")`).
    ///
    /// # Errors
    ///
    /// Returns a candle error if a required weight tensor is missing or has
    /// an unexpected shape.
    // `VarBuilder` is conventionally passed by value throughout this codebase
    // (its `pp`/`device`/`dtype` accessors take `&self` and are cheap to
    // call repeatedly); matching that convention here rather than switching
    // to `&VarBuilder` for this one constructor.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let dhs = config.downsample_hidden_size;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };

        let conv2d1 = with_tracing::conv2d(1, dhs, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = with_tracing::conv2d(dhs, dhs, 3, conv_cfg, vb.pp("conv2d2"))?;
        let conv2d3 = with_tracing::conv2d(dhs, dhs, 3, conv_cfg, vb.pp("conv2d3"))?;

        // 3x stride-2 conv reduces the mel-bin axis: 128 -> 64 -> 32 -> 16.
        let post_conv_mel_bins =
            conv_output_len(conv_output_len(conv_output_len(config.num_mel_bins)));
        let conv_flat_dim = dhs * post_conv_mel_bins;

        let conv_out =
            with_tracing::linear_no_bias(conv_flat_dim, config.d_model, vb.pp("conv_out"))?;

        let pos_embed = build_sinusoidal_table(
            config.max_position_embeddings,
            config.d_model,
            vb.device(),
            vb.dtype(),
        )?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            activation: config.activation_function,
            pos_embed,
            conv_flat_dim,
            conv_chunksize: config.conv_chunksize,
        })
    }

    /// Runs the conv stack on a batch of chunks.
    ///
    /// `chunk` is `[B, 1, n_mels, chunk_frames]` (B chunks as 1-channel
    /// 2D images). Returns `[B, tokens_in_chunk, d_model]`.
    fn conv_forward_chunk(&self, chunk: &Tensor) -> Result<Tensor> {
        let x = self.activation.forward(&self.conv2d1.forward(chunk)?)?;
        let x = self.activation.forward(&self.conv2d2.forward(&x)?)?;
        let x = self.activation.forward(&self.conv2d3.forward(&x)?)?;
        // x: [1, downsample_hidden_size, post_conv_mels, post_conv_time]

        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        let (b, t, _c, _f) = x.dims4()?;
        let x = x.reshape((b, t, self.conv_flat_dim))?;

        self.conv_out.forward(&x)
    }

    /// Processes a log-mel spectrogram into token embeddings.
    ///
    /// Takes the raw mel output of
    /// [`WhisperFeatureExtractor::extract`](super::feature_extractor::WhisperFeatureExtractor::extract),
    /// shape `[1, n_mels, n_frames]`, and returns token embeddings of shape
    /// `[1, total_tokens, d_model]`, where `total_tokens` matches
    /// [`get_feat_extract_output_lengths`](super::feature_extractor::get_feat_extract_output_lengths)`(n_frames)`.
    ///
    /// Full-size chunks are batched (up to `conv_chunksize` at a time)
    /// through the conv stack to reduce per-chunk kernel-launch overhead on
    /// GPU. A trailing partial chunk is processed separately.
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor operations fail.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (_batch, _mels, n_frames) = mel.dims3()?;

        if n_frames == 0 {
            let d_model = self.pos_embed.dims()[1];
            return Tensor::zeros(
                (1, 0, d_model),
                self.pos_embed.dtype(),
                self.pos_embed.device(),
            );
        }

        let (n_full_chunks, remainder) = chunk_split(n_frames);

        let mut outputs = Vec::new();

        // Full-size chunks: batch up to `conv_chunksize` at a time through
        // the conv stack in a single call.
        let mut chunk_start = 0;
        while chunk_start < n_full_chunks {
            let batch_size = (n_full_chunks - chunk_start).min(self.conv_chunksize);

            let batch: Vec<Tensor> = (0..batch_size)
                .map(|j| {
                    let start = (chunk_start + j) * FRAMES_PER_WINDOW;
                    // [1, n_mels, len] -> [1, 1, n_mels, len] for Conv2d.
                    mel.narrow(2, start, FRAMES_PER_WINDOW)?.unsqueeze(1)
                })
                .collect::<Result<_>>()?;
            let batch_input = Tensor::cat(&batch, 0)?;

            let embedded = self.conv_forward_chunk(&batch_input)?;
            let (b, seq_len, d) = embedded.dims3()?;
            let pos = self.pos_embed.narrow(0, 0, seq_len)?;
            let with_pos = embedded.broadcast_add(&pos)?;
            outputs.push(with_pos.reshape((1, b * seq_len, d))?);

            chunk_start += batch_size;
        }

        // Trailing partial chunk, if any (different frame count, always
        // processed alone).
        if remainder > 0 {
            let start = n_full_chunks * FRAMES_PER_WINDOW;
            let chunk = mel.narrow(2, start, remainder)?.unsqueeze(1)?;
            let embedded = self.conv_forward_chunk(&chunk)?;

            let (_, seq_len, _) = embedded.dims3()?;
            let pos = self.pos_embed.narrow(0, 0, seq_len)?;
            outputs.push(embedded.broadcast_add(&pos)?);
        }

        if outputs.len() == 1 {
            Ok(outputs.remove(0))
        } else {
            Tensor::cat(&outputs, 1)
        }
    }
}

/// Per-chunk output token counts for an `n_frames`-long mel input, in chunk
/// order — full [`FRAMES_PER_WINDOW`]-sized chunks each contribute
/// [`TOKENS_PER_WINDOW`] tokens, and a trailing partial chunk contributes
/// whatever the triple stride-2 conv-output-length formula gives its
/// (smaller) frame count. Mirrors [`AudioEncoderFrontend::forward`]'s own
/// chunking exactly, so the block-diagonal mask built from these counts
/// lines up with the frontend's actual token boundaries.
fn chunk_token_counts(n_frames: usize) -> Vec<usize> {
    if n_frames == 0 {
        return Vec::new();
    }
    let (n_full_chunks, remainder) = chunk_split(n_frames);

    let mut counts = vec![TOKENS_PER_WINDOW; n_full_chunks];
    if remainder > 0 {
        counts.push(conv_output_len(conv_output_len(conv_output_len(remainder))));
    }
    counts
}

/// Builds the additive block-diagonal attention mask that restricts each
/// encoder token to attending only within its own `chunks_per_block`-sized
/// group of conv chunks (an ~8s window, per `n_window_infer`) — a hard
/// correctness requirement of the reference implementation, not an optional
/// streaming mode (§3).
///
/// Returns `None` when the whole clip fits in a single block (all chunks
/// mutually visible, no masking needed).
///
/// This mirrors the reference implementation's own approach: a full
/// `total_tokens x total_tokens` dense mask applied to full dense attention,
/// rather than reshaping into per-block batches. That is O(n^2) in both
/// memory and attention matmul cost — for very long clips this is
/// significant (e.g. ~10 minutes of audio is ~7,800 tokens, so a
/// `[1, n_heads, 7800, 7800]` f32 attention-weights tensor is already ~3.4GB
/// per layer). Fine for typical utterance-length ASR input; a true windowed
/// (block-batched) attention implementation would be needed to process
/// long-form audio efficiently.
fn build_block_diagonal_mask(
    n_frames: usize,
    chunks_per_block: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    let counts = chunk_token_counts(n_frames);
    if counts.len() <= chunks_per_block {
        return Ok(None);
    }

    let total_tokens: usize = counts.iter().sum();
    let block_sizes: Vec<usize> = counts
        .chunks(chunks_per_block)
        .map(|group| group.iter().sum())
        .collect();

    // Fill everything as masked, then zero out each block's on-diagonal
    // square in one slice fill per row, rather than branching per cell.
    let mut mask = vec![f32::NEG_INFINITY; total_tokens * total_tokens];
    let mut offset = 0;
    for &block_len in &block_sizes {
        for i in 0..block_len {
            let row_start = (offset + i) * total_tokens + offset;
            mask[row_start..row_start + block_len].fill(0.0);
        }
        offset += block_len;
    }

    let mask = Tensor::from_vec(mask, (1, 1, total_tokens, total_tokens), device)?;
    Ok(Some(mask.to_dtype(dtype)?))
}

/// Self-attention for one audio encoder transformer layer.
///
/// Plain multi-head attention (`num_key_value_heads == encoder_attention_heads`
/// on both released checkpoint sizes, §3) with bias on every projection, no
/// rotary embeddings, and no `KV` cache — the encoder processes a complete
/// utterance in a single non-causal forward pass, so there is no
/// autoregressive state to accumulate across calls.
///
/// The supplied [`VarBuilder`] must be scoped to one layer's `self_attn`
/// prefix. Loaded tensors: `q_proj.{weight,bias}`, `k_proj.{weight,bias}`,
/// `v_proj.{weight,bias}`, `out_proj.{weight,bias}`, each `[d_model, d_model]`
/// (checkpoint uses `out_proj`, not `o_proj`).
#[derive(Debug, Clone)]
struct AudioEncoderAttention {
    q_proj: with_tracing::Linear,
    k_proj: with_tracing::Linear,
    v_proj: with_tracing::Linear,
    out_proj: with_tracing::Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl AudioEncoderAttention {
    // `VarBuilder` is conventionally passed by value throughout this
    // codebase (its `pp`/`device`/`dtype` accessors take `&self` and are
    // cheap to call repeatedly); matching that convention here.
    #[allow(clippy::needless_pass_by_value)]
    fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        debug_assert_eq!(
            config.num_key_value_heads, config.encoder_attention_heads,
            "AudioEncoderAttention only implements plain multi-head attention, not GQA"
        );

        let d_model = config.d_model;
        let n_heads = config.encoder_attention_heads;
        let head_dim = d_model / n_heads;

        let q_proj = with_tracing::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = with_tracing::linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = with_tracing::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = with_tracing::linear(d_model, d_model, vb.pp("out_proj"))?;

        // head_dim is at most a few hundred in practice, exact in f64.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1f64 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            head_dim,
            scale,
        })
    }

    /// `hidden_states`: `[batch, seq_len, d_model]`. `attention_mask`, if
    /// given, is an additive mask broadcastable to
    /// `[batch, n_heads, seq_len, seq_len]` (`f32::NEG_INFINITY` for blocked
    /// positions).
    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, _d) = hidden_states.dims3()?;
        let shape = (batch, seq_len, self.n_heads, self.head_dim);

        let query = self
            .q_proj
            .forward(hidden_states)?
            .reshape(shape)?
            .transpose(1, 2)?;
        let key = self
            .k_proj
            .forward(hidden_states)?
            .reshape(shape)?
            .transpose(1, 2)?;
        let value = self
            .v_proj
            .forward(hidden_states)?
            .reshape(shape)?
            .transpose(1, 2)?;

        let attn_weights = (query
            .contiguous()?
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            * self.scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        // Softmax in f32 for numerical stability, matching GqaAttention's
        // convention, then cast back to the working dtype.
        let attn_weights = candle_nn::ops::softmax(&attn_weights.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(value.dtype())?;

        let attn_output = attn_weights.matmul(&value.contiguous()?)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

/// Feed-forward block for one audio encoder transformer layer: `fc1` ->
/// activation (GELU on released checkpoints) -> `fc2`, both linear layers
/// with bias.
///
/// The supplied [`VarBuilder`] must be scoped to one layer's prefix
/// directly (checkpoint tensors `fc1.{weight,bias}`, `fc2.{weight,bias}`
/// sit directly under `layers.N`, not nested under a `mlp` submodule).
#[derive(Debug, Clone)]
struct AudioEncoderMlp {
    fc1: with_tracing::Linear,
    fc2: with_tracing::Linear,
    activation: Activation,
}

impl AudioEncoderMlp {
    // See `AudioEncoderAttention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = with_tracing::linear(config.d_model, config.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = with_tracing::linear(config.encoder_ffn_dim, config.d_model, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation: config.activation_function,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.activation.forward(&self.fc1.forward(xs)?)?;
        self.fc2.forward(&xs)
    }
}

/// One pre-norm audio encoder transformer layer (standard Whisper/BART
/// block shape, §3):
///
/// ```text
/// h = h + self_attn(self_attn_layer_norm(h), mask)
/// h = h + mlp(final_layer_norm(h))
/// ```
///
/// The supplied [`VarBuilder`] must be scoped to one layer's prefix (e.g.
/// `layers.N`).
#[derive(Debug, Clone)]
struct AudioEncoderLayer {
    self_attn: AudioEncoderAttention,
    self_attn_layer_norm: with_tracing::LayerNorm,
    mlp: AudioEncoderMlp,
    final_layer_norm: with_tracing::LayerNorm,
}

impl AudioEncoderLayer {
    // See `AudioEncoderAttention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = AudioEncoderAttention::new(config, vb.pp("self_attn"))?;
        let self_attn_layer_norm = with_tracing::layer_norm(
            config.d_model,
            LAYER_NORM_EPS,
            vb.pp("self_attn_layer_norm"),
        )?;
        let mlp = AudioEncoderMlp::new(config, vb.clone())?;
        let final_layer_norm =
            with_tracing::layer_norm(config.d_model, LAYER_NORM_EPS, vb.pp("final_layer_norm"))?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            mlp,
            final_layer_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = hidden_states;
        let hs = self.self_attn_layer_norm.forward(hidden_states)?;
        let hs = self.self_attn.forward(&hs, attention_mask)?;
        let hidden_states = (residual + hs)?;

        let residual = &hidden_states;
        let hs = self.final_layer_norm.forward(&hidden_states)?;
        let hs = self.mlp.forward(&hs)?;
        residual + hs
    }
}

/// Full Qwen3-ASR audio encoder ("`AuT`"): conv frontend
/// ([`AudioEncoderFrontend`]) + `encoder_layers` transformer blocks with
/// block-diagonal windowed attention + a final `ln_post`.
///
/// The supplied [`VarBuilder`] must be scoped to the audio tower prefix
/// (checkpoint path `model.audio_tower`). Loaded tensors, beyond the
/// frontend's own (documented on [`AudioEncoderFrontend`]):
/// `layers.N.*` (§[`AudioEncoderLayer`]) for `N` in `0..encoder_layers`, and
/// `ln_post.{weight,bias}` — `[d_model]`.
#[derive(Debug)]
pub struct AudioEncoder {
    frontend: AudioEncoderFrontend,
    layers: Vec<AudioEncoderLayer>,
    ln_post: with_tracing::LayerNorm,
    /// Number of conv chunks per block-diagonal attention window
    /// (`n_window_infer / (n_window * 2)`, i.e. `n_window_infer / FRAMES_PER_WINDOW`).
    chunks_per_block: usize,
}

impl AudioEncoder {
    /// Builds the full audio encoder from checkpoint weights.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a required weight tensor is missing or has
    /// an unexpected shape.
    // See `AudioEncoderAttention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let frontend = AudioEncoderFrontend::new(config, vb.clone())?;

        let layers_vb = vb.pp("layers");
        let layers = (0..config.encoder_layers)
            .map(|i| AudioEncoderLayer::new(config, layers_vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;

        let ln_post = with_tracing::layer_norm(config.d_model, LAYER_NORM_EPS, vb.pp("ln_post"))?;

        debug_assert_eq!(
            FRAMES_PER_WINDOW, config.n_window * 2,
            "FRAMES_PER_WINDOW must equal n_window * 2"
        );
        debug_assert_eq!(
            config.n_window_infer % FRAMES_PER_WINDOW, 0,
            "n_window_infer must be a multiple of FRAMES_PER_WINDOW"
        );
        let chunks_per_block = config.n_window_infer / FRAMES_PER_WINDOW;

        Ok(Self {
            frontend,
            layers,
            ln_post,
            chunks_per_block,
        })
    }

    /// Processes a log-mel spectrogram into audio token embeddings.
    ///
    /// Takes the same input as [`AudioEncoderFrontend::forward`] (`[1,
    /// n_mels, n_frames]`) and returns `[1, total_tokens, d_model]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor operations fail.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (_batch, _mels, n_frames) = mel.dims3()?;

        let mut hidden_states = self.frontend.forward(mel)?;
        let mask = build_block_diagonal_mask(
            n_frames,
            self.chunks_per_block,
            hidden_states.device(),
            hidden_states.dtype(),
        )?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
        }

        self.ln_post.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::DType;

    use super::super::feature_extractor::get_feat_extract_output_lengths;
    use super::*;

    fn test_audio_config() -> AudioConfig {
        AudioConfig {
            d_model: 896,
            encoder_layers: 18,
            encoder_attention_heads: 14,
            num_key_value_heads: 14,
            encoder_ffn_dim: 3584,
            output_dim: 1024,
            num_mel_bins: 128,
            downsample_hidden_size: 480,
            max_position_embeddings: 13,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: Activation::Gelu,
            scale_embedding: false,
        }
    }

    /// Builds a `VarBuilder` with zero-valued weights for the audio encoder
    /// frontend, matching `AudioEncoderFrontend`'s expected checkpoint keys.
    fn make_frontend_vb(config: &AudioConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let dtype = DType::F32;
        let dhs = config.downsample_hidden_size;
        let post_conv_mel = conv_output_len(conv_output_len(conv_output_len(config.num_mel_bins)));
        let conv_flat = dhs * post_conv_mel;

        let zeros4 = |shape: (usize, usize, usize, usize)| {
            Tensor::zeros(shape, dtype, device).expect("zeros weight")
        };
        let zeros1 = |n: usize| Tensor::zeros(n, dtype, device).expect("zeros bias");
        let zeros2 =
            |shape: (usize, usize)| Tensor::zeros(shape, dtype, device).expect("zeros weight");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("conv2d1.weight".into(), zeros4((dhs, 1, 3, 3)));
        t.insert("conv2d1.bias".into(), zeros1(dhs));
        t.insert("conv2d2.weight".into(), zeros4((dhs, dhs, 3, 3)));
        t.insert("conv2d2.bias".into(), zeros1(dhs));
        t.insert("conv2d3.weight".into(), zeros4((dhs, dhs, 3, 3)));
        t.insert("conv2d3.bias".into(), zeros1(dhs));
        t.insert(
            "conv_out.weight".into(),
            zeros2((config.d_model, conv_flat)),
        );

        VarBuilder::from_tensors(t, dtype, device)
    }

    #[test]
    fn sinusoidal_table_shape() {
        let table = build_sinusoidal_table(13, 896, &Device::Cpu, DType::F32).expect("build table");
        assert_eq!(table.dims(), &[13, 896]);
    }

    #[test]
    fn sinusoidal_table_values_are_bounded() {
        let table = build_sinusoidal_table(13, 896, &Device::Cpu, DType::F32).expect("build table");
        let data = table
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        for &v in &data {
            assert!(
                (-1.0..=1.0).contains(&v),
                "sin/cos value {v} out of [-1, 1]"
            );
        }
    }

    #[test]
    fn sinusoidal_table_position_zero_is_predictable() {
        // At position 0: sin(0) = 0 for all frequencies, cos(0) = 1 for all frequencies.
        let d_model = 16;
        let table =
            build_sinusoidal_table(4, d_model, &Device::Cpu, DType::F32).expect("build table");
        let row0 = table
            .narrow(0, 0, 1)
            .expect("narrow")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        let half = d_model / 2;
        for i in 0..half {
            assert!(
                row0[i].abs() < 1e-6,
                "sin at pos 0 should be 0, got {}",
                row0[i]
            );
            assert!(
                (row0[half + i] - 1.0).abs() < 1e-6,
                "cos at pos 0 should be 1, got {}",
                row0[half + i]
            );
        }
    }

    #[test]
    fn post_conv_mel_bins_is_16() {
        assert_eq!(conv_output_len(conv_output_len(conv_output_len(128))), 16);
    }

    #[test]
    fn full_window_produces_13_tokens() {
        assert_eq!(
            conv_output_len(conv_output_len(conv_output_len(FRAMES_PER_WINDOW))),
            TOKENS_PER_WINDOW
        );
    }

    #[test]
    fn frontend_output_shape_single_full_chunk() {
        let config = test_audio_config();
        let vb = make_frontend_vb(&config);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 128, 100), DType::F32, &Device::Cpu).expect("mel");
        let out = frontend.forward(&mel).expect("forward");
        assert_eq!(out.dims(), &[1, 13, 896]);
    }

    #[test]
    fn frontend_output_shape_multiple_chunks() {
        let config = test_audio_config();
        let vb = make_frontend_vb(&config);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        // 250 frames = 2 full chunks (26 tokens) + 50-frame remainder (7 tokens).
        let mel = Tensor::zeros((1, 128, 250), DType::F32, &Device::Cpu).expect("mel");
        let out = frontend.forward(&mel).expect("forward");
        let expected_tokens =
            2 * TOKENS_PER_WINDOW + conv_output_len(conv_output_len(conv_output_len(50)));
        assert_eq!(out.dims(), &[1, expected_tokens, 896]);
    }

    #[test]
    fn frontend_output_matches_get_feat_extract_output_lengths() {
        let config = test_audio_config();
        let vb = make_frontend_vb(&config);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        for n_frames in [50, 99, 100, 150, 200, 830] {
            let mel = Tensor::zeros((1, 128, n_frames), DType::F32, &Device::Cpu).expect("mel");
            let out = frontend.forward(&mel).expect("forward");
            let actual_tokens = out.dims()[1];
            let expected = get_feat_extract_output_lengths(n_frames);
            assert_eq!(
                actual_tokens, expected,
                "n_frames={n_frames}: conv frontend produced {actual_tokens} tokens, expected {expected}"
            );
        }
    }

    #[test]
    fn frontend_zero_weights_output_is_pos_embed_only() {
        // With all-zero conv/conv_out weights (and zero conv bias), conv_out's
        // output is exactly zero, so the forward output should equal the
        // positional embedding alone.
        let config = test_audio_config();
        let vb = make_frontend_vb(&config);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 128, 100), DType::F32, &Device::Cpu).expect("mel");
        let out = frontend.forward(&mel).expect("forward");

        let expected = frontend
            .pos_embed
            .narrow(0, 0, 13)
            .expect("narrow")
            .unsqueeze(0)
            .expect("unsqueeze");
        let diff: f32 = (&out - &expected)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-5,
            "zero-weight output should equal pos_embed, max_diff={diff}"
        );
    }

    #[test]
    fn frontend_zero_frame_input() {
        let config = test_audio_config();
        let vb = make_frontend_vb(&config);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 128, 0), DType::F32, &Device::Cpu).expect("mel");
        let out = frontend.forward(&mel).expect("forward");
        assert_eq!(out.dims(), &[1, 0, 896]);
    }

    #[test]
    fn frontend_multi_chunk_positions_restart() {
        // With zero weights, forward() output is purely the positional
        // embedding. A 200-frame input yields 2 full chunks of 100 frames
        // (13 tokens each); since positions restart at 0 every chunk, both
        // halves of the output must be identical.
        let config = test_audio_config();
        let vb = make_frontend_vb(&config);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 128, 200), DType::F32, &Device::Cpu).expect("mel");
        let out = frontend.forward(&mel).expect("forward");
        assert_eq!(out.dims(), &[1, 26, 896]);

        let first_chunk = out.narrow(1, 0, 13).expect("narrow first");
        let second_chunk = out.narrow(1, 13, 13).expect("narrow second");
        let diff: f32 = (&first_chunk - &second_chunk)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-6,
            "positions should restart each chunk, max_diff={diff}"
        );
    }

    /// Config where `conv_flat_dim == d_model == 6`, so `conv_out` can be an
    /// identity matrix, making the conv's permute+flatten ordering directly
    /// observable in the output.
    fn permute_test_config() -> AudioConfig {
        AudioConfig {
            d_model: 6,
            encoder_layers: 1,
            encoder_attention_heads: 1,
            num_key_value_heads: 1,
            encoder_ffn_dim: 6,
            output_dim: 6,
            num_mel_bins: 16,
            downsample_hidden_size: 3,
            max_position_embeddings: 13,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: Activation::Gelu,
            scale_embedding: false,
        }
    }

    #[test]
    fn frontend_permute_ordering_is_channel_major() {
        // With zero conv weights, each conv layer's output is
        // GELU(its_bias) at every spatial position (the input is ignored).
        // conv2d3.bias = [1, 2, 3] (one value per of the 3 channels), so
        // after conv3+GELU: shape [1, 3, post_conv_mels, T], channel c is
        // constant GELU(bias3[c]).
        //
        // Correct permute (0,3,1,2) -> [1, T, 3, post_conv_mels] -> flatten
        // to [1, T, 6] is channel-major: [g1, g1, g2, g2, g3, g3]. A wrong
        // permute (e.g. keeping mel before channel) would instead produce
        // the mel-major [g1, g2, g3, g1, g2, g3].
        //
        // conv_out is an identity matrix, so the flattened vector passes
        // through unchanged and the ordering is directly observable.
        let config = permute_test_config();
        let dhs = config.downsample_hidden_size;
        let post_conv_mel = conv_output_len(conv_output_len(conv_output_len(config.num_mel_bins)));
        let conv_flat = dhs * post_conv_mel;

        let device = &Device::Cpu;
        let dtype = DType::F32;
        let zeros4 = |shape: (usize, usize, usize, usize)| {
            Tensor::zeros(shape, dtype, device).expect("zeros")
        };
        let zeros1 = |n: usize| Tensor::zeros(n, dtype, device).expect("zeros");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("conv2d1.weight".into(), zeros4((dhs, 1, 3, 3)));
        t.insert("conv2d1.bias".into(), zeros1(dhs));
        t.insert("conv2d2.weight".into(), zeros4((dhs, dhs, 3, 3)));
        t.insert("conv2d2.bias".into(), zeros1(dhs));
        t.insert("conv2d3.weight".into(), zeros4((dhs, dhs, 3, 3)));
        t.insert(
            "conv2d3.bias".into(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], dhs, device).expect("bias"),
        );
        t.insert(
            "conv_out.weight".into(),
            Tensor::eye(conv_flat, dtype, device).expect("eye"),
        );

        let vb = VarBuilder::from_tensors(t, dtype, device);
        let frontend = AudioEncoderFrontend::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 16, 8), dtype, device).expect("mel");
        let out = frontend.forward(&mel).expect("forward");

        // Subtract pos_embed to isolate the conv stack's contribution.
        let (_, seq_len, _) = out.dims3().expect("dims3");
        let pos = frontend
            .pos_embed
            .narrow(0, 0, seq_len)
            .expect("narrow")
            .unsqueeze(0)
            .expect("unsqueeze");
        let conv_only = (&out - &pos).expect("sub");

        let row = conv_only
            .narrow(1, 0, 1)
            .expect("narrow")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let g1 = row[0];
        let g2 = row[2];
        let g3 = row[4];
        assert!(
            (g2 - g1).abs() > 0.5,
            "channels 1 and 2 should differ: g1={g1}, g2={g2}"
        );
        assert!(
            (g3 - g2).abs() > 0.5,
            "channels 2 and 3 should differ: g2={g2}, g3={g3}"
        );

        assert!(
            (row[0] - row[1]).abs() < 1e-6,
            "channel-major: row[0]={} should equal row[1]={}",
            row[0],
            row[1]
        );
        assert!(
            (row[2] - row[3]).abs() < 1e-6,
            "channel-major: row[2]={} should equal row[3]={}",
            row[2],
            row[3]
        );
        assert!(
            (row[4] - row[5]).abs() < 1e-6,
            "channel-major: row[4]={} should equal row[5]={}",
            row[4],
            row[5]
        );
    }

    #[test]
    fn block_diagonal_mask_single_block() {
        // 200 frames = 2 full chunks; chunks_per_block=8 means both chunks
        // fit in one block, so no masking is needed.
        let mask = build_block_diagonal_mask(200, 8, &Device::Cpu, DType::F32).expect("mask");
        assert!(mask.is_none());
    }

    #[test]
    fn block_diagonal_mask_shape_and_values() {
        // 300 frames = 3 full chunks (13 tokens each -> 39 total). With
        // chunks_per_block=2: chunks {0,1} form block 0 (26 tokens), chunk
        // {2} forms block 1 (13 tokens).
        let mask = build_block_diagonal_mask(300, 2, &Device::Cpu, DType::F32)
            .expect("mask")
            .expect("mask should be Some for 3 chunks over a 2-chunk block");
        assert_eq!(mask.dims(), &[1, 1, 39, 39]);

        let data = mask
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        let at = |i: usize, j: usize| data[i * 39 + j];

        // Within block 0 (tokens 0..26): unmasked.
        assert_eq!(at(0, 25), 0.0);
        // Within block 1 (tokens 26..39): unmasked.
        assert_eq!(at(26, 38), 0.0);
        // Across the block 0 / block 1 boundary: masked.
        assert_eq!(at(0, 26), f32::NEG_INFINITY);
        assert_eq!(at(25, 26), f32::NEG_INFINITY);
        assert_eq!(at(26, 0), f32::NEG_INFINITY);
    }

    #[test]
    fn block_diagonal_mask_isolates_windows() {
        // Same 3-chunk / 2-chunks-per-block layout as above: exhaustively
        // check every (i, j) pair agrees with the expected block membership,
        // proving no attention leaks across the ~8s (here, 2-chunk) boundary.
        let mask = build_block_diagonal_mask(300, 2, &Device::Cpu, DType::F32)
            .expect("mask")
            .expect("mask should be Some");
        let data = mask
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let counts = chunk_token_counts(300);
        let mut block_of_token = Vec::new();
        for (chunk_idx, &count) in counts.iter().enumerate() {
            block_of_token.extend(std::iter::repeat_n(chunk_idx / 2, count));
        }
        let total = block_of_token.len();

        for i in 0..total {
            for j in 0..total {
                let v = data[i * total + j];
                if block_of_token[i] == block_of_token[j] {
                    assert_eq!(v, 0.0, "expected unmasked at ({i}, {j})");
                } else {
                    assert_eq!(v, f32::NEG_INFINITY, "expected masked at ({i}, {j})");
                }
            }
        }
    }

    /// A tiny `AudioConfig` for encoder-layer/full-encoder tests: small
    /// enough that per-layer weight tensors are trivial to allocate, while
    /// keeping `encoder_attention_heads` dividing `d_model` evenly.
    fn small_encoder_config() -> AudioConfig {
        AudioConfig {
            d_model: 8,
            encoder_layers: 2,
            encoder_attention_heads: 2,
            num_key_value_heads: 2,
            encoder_ffn_dim: 16,
            output_dim: 8,
            num_mel_bins: 16,
            downsample_hidden_size: 3,
            max_position_embeddings: 13,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: Activation::Gelu,
            scale_embedding: false,
        }
    }

    /// Builds a `VarBuilder` for one `AudioEncoderLayer` with all projection
    /// weights/biases zeroed and identity-ish (`weight=1, bias=0`) layer
    /// norms.
    fn make_layer_vb(config: &AudioConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let dtype = DType::F32;
        let d = config.d_model;
        let ffn = config.encoder_ffn_dim;

        let zeros2 = |shape: (usize, usize)| Tensor::zeros(shape, dtype, device).expect("zeros");
        let zeros1 = |n: usize| Tensor::zeros(n, dtype, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, dtype, device).expect("ones");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            t.insert(format!("self_attn.{proj}.weight"), zeros2((d, d)));
            t.insert(format!("self_attn.{proj}.bias"), zeros1(d));
        }
        t.insert("self_attn_layer_norm.weight".into(), ones1(d));
        t.insert("self_attn_layer_norm.bias".into(), zeros1(d));
        t.insert("fc1.weight".into(), zeros2((ffn, d)));
        t.insert("fc1.bias".into(), zeros1(ffn));
        t.insert("fc2.weight".into(), zeros2((d, ffn)));
        t.insert("fc2.bias".into(), zeros1(d));
        t.insert("final_layer_norm.weight".into(), ones1(d));
        t.insert("final_layer_norm.bias".into(), zeros1(d));

        VarBuilder::from_tensors(t, dtype, device)
    }

    /// Builds a `VarBuilder` for one `AudioEncoderLayer` with small non-zero
    /// attention weights (so `V` is non-zero and attention actually mixes
    /// information across tokens) but zeroed MLP weights, isolating any
    /// cross-token mixing in the test below to the attention sub-block.
    fn make_nonzero_attn_layer_vb(config: &AudioConfig, scale: f32) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let dtype = DType::F32;
        let d = config.d_model;
        let ffn = config.encoder_ffn_dim;

        let fill2 = |rows: usize, cols: usize| {
            let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * scale).collect();
            Tensor::from_vec(data, (rows, cols), device).expect("weight")
        };
        let zeros2 = |shape: (usize, usize)| Tensor::zeros(shape, dtype, device).expect("zeros");
        let zeros1 = |n: usize| Tensor::zeros(n, dtype, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, dtype, device).expect("ones");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            t.insert(format!("self_attn.{proj}.weight"), fill2(d, d));
            t.insert(format!("self_attn.{proj}.bias"), zeros1(d));
        }
        t.insert("self_attn_layer_norm.weight".into(), ones1(d));
        t.insert("self_attn_layer_norm.bias".into(), zeros1(d));
        t.insert("fc1.weight".into(), zeros2((ffn, d)));
        t.insert("fc1.bias".into(), zeros1(ffn));
        t.insert("fc2.weight".into(), zeros2((d, ffn)));
        t.insert("fc2.bias".into(), zeros1(d));
        t.insert("final_layer_norm.weight".into(), ones1(d));
        t.insert("final_layer_norm.bias".into(), zeros1(d));

        VarBuilder::from_tensors(t, dtype, device)
    }

    #[test]
    fn encoder_layer_mask_isolates_blocks() {
        // With non-zero attention weights (V is non-zero, so attention
        // actually mixes information across tokens), varying block 1's
        // input must leave block 0's output untouched — proving the mask
        // is wired through `AudioEncoderLayer::forward`, not just
        // structurally correct in isolation (unlike the `v_proj=0` layer
        // tests above, which would pass identically even if the mask were
        // silently dropped).
        let config = small_encoder_config();
        let vb = make_nonzero_attn_layer_vb(&config, 0.01);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        // 200 frames = 2 chunks; chunks_per_block=1 puts each chunk in its
        // own block, so tokens 0..13 and 13..26 must be mutually invisible.
        let block_len = TOKENS_PER_WINDOW;
        let total = 2 * block_len;
        let mask = build_block_diagonal_mask(200, 1, &Device::Cpu, DType::F32)
            .expect("mask")
            .expect("mask should be Some for 2 blocks");

        let base = Tensor::randn(0f32, 1f32, (1, total, config.d_model), &Device::Cpu)
            .expect("randn base");
        let perturbation = Tensor::randn(0f32, 1f32, (1, block_len, config.d_model), &Device::Cpu)
            .expect("randn perturbation");

        let block0 = base.narrow(1, 0, block_len).expect("narrow block0");
        let block1 = base.narrow(1, block_len, block_len).expect("narrow block1");
        let block1_perturbed = (&block1 + &perturbation).expect("add");
        let perturbed = Tensor::cat(&[&block0, &block1_perturbed], 1).expect("cat");

        let out_base = layer.forward(&base, Some(&mask)).expect("forward base");
        let out_perturbed = layer
            .forward(&perturbed, Some(&mask))
            .expect("forward perturbed");

        let max_abs_diff = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b)
                .expect("sub")
                .abs()
                .expect("abs")
                .max_all()
                .expect("max_all")
                .to_scalar()
                .expect("scalar")
        };

        let diff_block0 = max_abs_diff(
            &out_base.narrow(1, 0, block_len).expect("narrow"),
            &out_perturbed.narrow(1, 0, block_len).expect("narrow"),
        );
        assert!(
            diff_block0 < 1e-5,
            "block 0 output must be unaffected by block 1's input, diff={diff_block0}"
        );

        let diff_block1 = max_abs_diff(
            &out_base.narrow(1, block_len, block_len).expect("narrow"),
            &out_perturbed.narrow(1, block_len, block_len).expect("narrow"),
        );
        assert!(
            diff_block1 > 1e-4,
            "block 1 output must change when its own input changes, diff={diff_block1}"
        );
    }

    #[test]
    fn encoder_layer_output_shape() {
        let config = small_encoder_config();
        let vb = make_layer_vb(&config);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        let hidden =
            Tensor::randn(0f32, 1f32, (1, 5, config.d_model), &Device::Cpu).expect("randn");
        let out = layer.forward(&hidden, None).expect("forward");
        assert_eq!(out.dims(), hidden.dims());
    }

    #[test]
    fn encoder_layer_is_prenorm_residual() {
        // With every attention/MLP projection zeroed, both sub-blocks
        // contribute exactly zero, so the pre-norm residual pattern
        // (h = h + f(norm(h))) must leave the input unchanged.
        let config = small_encoder_config();
        let vb = make_layer_vb(&config);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        let hidden =
            Tensor::randn(0f32, 1f32, (1, 5, config.d_model), &Device::Cpu).expect("randn");
        let out = layer.forward(&hidden, None).expect("forward");

        let diff: f32 = (&out - &hidden)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-5,
            "zero-weight layer should be identity, max_diff={diff}"
        );
    }

    /// Builds a `VarBuilder` for a full `AudioEncoder` (frontend + layers +
    /// `ln_post`), all zero-valued except identity-ish layer norms.
    fn make_encoder_vb(config: &AudioConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let dtype = DType::F32;
        let dhs = config.downsample_hidden_size;
        let post_conv_mel = conv_output_len(conv_output_len(conv_output_len(config.num_mel_bins)));
        let conv_flat = dhs * post_conv_mel;

        let zeros4 = |shape: (usize, usize, usize, usize)| {
            Tensor::zeros(shape, dtype, device).expect("zeros")
        };
        let zeros2 = |shape: (usize, usize)| Tensor::zeros(shape, dtype, device).expect("zeros");
        let zeros1 = |n: usize| Tensor::zeros(n, dtype, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, dtype, device).expect("ones");

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("conv2d1.weight".into(), zeros4((dhs, 1, 3, 3)));
        t.insert("conv2d1.bias".into(), zeros1(dhs));
        t.insert("conv2d2.weight".into(), zeros4((dhs, dhs, 3, 3)));
        t.insert("conv2d2.bias".into(), zeros1(dhs));
        t.insert("conv2d3.weight".into(), zeros4((dhs, dhs, 3, 3)));
        t.insert("conv2d3.bias".into(), zeros1(dhs));
        t.insert(
            "conv_out.weight".into(),
            zeros2((config.d_model, conv_flat)),
        );

        let d = config.d_model;
        let ffn = config.encoder_ffn_dim;
        for layer_idx in 0..config.encoder_layers {
            let p = format!("layers.{layer_idx}");
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                t.insert(format!("{p}.self_attn.{proj}.weight"), zeros2((d, d)));
                t.insert(format!("{p}.self_attn.{proj}.bias"), zeros1(d));
            }
            t.insert(format!("{p}.self_attn_layer_norm.weight"), ones1(d));
            t.insert(format!("{p}.self_attn_layer_norm.bias"), zeros1(d));
            t.insert(format!("{p}.fc1.weight"), zeros2((ffn, d)));
            t.insert(format!("{p}.fc1.bias"), zeros1(ffn));
            t.insert(format!("{p}.fc2.weight"), zeros2((d, ffn)));
            t.insert(format!("{p}.fc2.bias"), zeros1(d));
            t.insert(format!("{p}.final_layer_norm.weight"), ones1(d));
            t.insert(format!("{p}.final_layer_norm.bias"), zeros1(d));
        }
        t.insert("ln_post.weight".into(), ones1(d));
        t.insert("ln_post.bias".into(), zeros1(d));

        VarBuilder::from_tensors(t, dtype, device)
    }

    #[test]
    fn encoder_output_shape() {
        let config = small_encoder_config();
        let vb = make_encoder_vb(&config);
        let encoder = AudioEncoder::new(&config, vb).expect("new");

        for n_frames in [50, 99, 100, 150, 200, 830] {
            let mel = Tensor::zeros((1, config.num_mel_bins, n_frames), DType::F32, &Device::Cpu)
                .expect("mel");
            let out = encoder.forward(&mel).expect("forward");
            let expected_tokens = get_feat_extract_output_lengths(n_frames);
            assert_eq!(out.dims(), &[1, expected_tokens, config.d_model]);
        }
    }

    #[test]
    fn encoder_single_chunk_no_mask() {
        // 100 frames = 1 chunk, well within a single ~8s attention block, so
        // the encoder must run without ever constructing a mask.
        let config = small_encoder_config();
        let vb = make_encoder_vb(&config);
        let encoder = AudioEncoder::new(&config, vb).expect("new");

        let mel =
            Tensor::zeros((1, config.num_mel_bins, 100), DType::F32, &Device::Cpu).expect("mel");
        let out = encoder.forward(&mel).expect("forward");
        assert_eq!(out.dims(), &[1, TOKENS_PER_WINDOW, config.d_model]);
    }
}
