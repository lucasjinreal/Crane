//! Conv2d frontend, windowed transformer, and sinusoidal positional encoding
//! for the Qwen3-ASR audio encoder ("`AuT`").
//!
//! Converts a log-mel spectrogram `[1, n_mels, n_frames]` from
//! [`WhisperFeatureExtractor`](super::feature_extractor::WhisperFeatureExtractor)
//! into `[1, total_tokens, output_dim]`-ready token embeddings (the encoder's
//! own output is `[1, total_tokens, d_model]`; the multi-modal projector
//! ([`super::modeling::MultiModalProjector`]) maps `d_model` to
//! `output_dim`):
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
//!    LayerNorm+GELU transformer blocks, with self-attention restricted via
//!    `cu_seqlens` boundaries to independent ~8s windows (`n_window_infer`
//!    raw frames per block — a hard correctness requirement, not an
//!    optional streaming mode), then a final `ln_post` ([`AudioEncoder`]).

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::attention::{AttnMask, flash_attn_varlen_cpu};
use candle_nn::{Activation, Conv2dConfig, VarBuilder};

use super::config::AudioConfig;
use super::feature_extractor::{
    FRAMES_PER_WINDOW, TOKENS_PER_WINDOW, chunk_split, conv_output_len,
};
use crate::models::modules::flash_attn::dispatch_flash_attn;
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
/// chunking exactly, so the `cu_seqlens` boundaries built from these counts
/// line up with the frontend's actual token boundaries.
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

/// Builds cumulative sequence-length boundaries that restrict each encoder
/// token to attending only within its own `chunks_per_block`-sized group of
/// conv chunks (an ~8s window, per `n_window_infer`) — a hard correctness
/// requirement of the reference implementation, not an optional streaming
/// mode (§3).
///
/// Returns `[0, block_0_len, block_0_len + block_1_len, ..., total_tokens]`
/// (always at least `[0, total_tokens]`, even for a single block) — a window
/// over consecutive pairs gives each block's `(start, end)` token range.
/// Attention is computed independently per block (`AudioEncoderAttention::
/// forward`), so this replaces a dense `total_tokens x total_tokens` mask
/// (O(n^2) memory and matmul cost — e.g. ~10 minutes of audio is ~7,800
/// tokens, making a `[1, n_heads, 7800, 7800]` f32 attention-weights tensor
/// ~3.4GB per layer) with O(n) total attention cost across blocks.
fn build_cu_seqlens(n_frames: usize, chunks_per_block: usize) -> Vec<usize> {
    let counts = chunk_token_counts(n_frames);
    let block_sizes: Vec<usize> = counts
        .chunks(chunks_per_block)
        .map(|group| group.iter().sum())
        .collect();

    let mut cu_seqlens = Vec::with_capacity(block_sizes.len() + 1);
    cu_seqlens.push(0);
    let mut acc = 0;
    for &block_len in &block_sizes {
        acc += block_len;
        cu_seqlens.push(acc);
    }
    cu_seqlens
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
/// (checkpoint uses `out_proj`, not `o_proj`). `q_proj`/`k_proj`/`v_proj` are
/// concatenated into a single `qkv_proj` at construction time (mirroring
/// `qwen3::Attention`'s merged QKV) so `forward` dispatches one gemm instead
/// of three.
#[derive(Debug, Clone)]
struct AudioEncoderAttention {
    qkv_proj: with_tracing::Linear,
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
        if config.num_key_value_heads != config.encoder_attention_heads {
            candle_core::bail!(
                "AudioEncoderAttention only implements plain multi-head attention, not GQA \
                 (num_key_value_heads={}, encoder_attention_heads={})",
                config.num_key_value_heads,
                config.encoder_attention_heads
            );
        }

        let d_model = config.d_model;
        let n_heads = config.encoder_attention_heads;
        let head_dim = d_model / n_heads;

        let q_proj = with_tracing::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = with_tracing::linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = with_tracing::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = with_tracing::linear(d_model, d_model, vb.pp("out_proj"))?;

        // Merge q/k/v into one gemm + narrow() splits, same pattern as
        // qwen3::Attention::new's qkv_proj.
        let qkv_w = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
        let qkv_b = match (q_proj.bias(), k_proj.bias(), v_proj.bias()) {
            (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[qb, kb, vb], 0)?),
            // `with_tracing::linear` above always attaches a bias, so this
            // arm is unreachable in practice.
            _ => None,
        };
        let qkv_proj = with_tracing::Linear::from_weights(qkv_w, qkv_b);

        // head_dim is at most a few hundred in practice, exact in f64.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1f64 / (head_dim as f64).sqrt();

        Ok(Self {
            qkv_proj,
            out_proj,
            n_heads,
            head_dim,
            scale,
        })
    }

    /// `hidden_states`: `[batch, seq_len, d_model]`. `cu_seqlens` is a list
    /// of cumulative sequence-length boundaries (as built by
    /// [`build_cu_seqlens`]); each consecutive pair `(cu_seqlens[i],
    /// cu_seqlens[i + 1])` is one attention block, computed independently of
    /// every other block — no cross-block masking needed since blocks are
    /// never mixed. `cu_seqlens` must have at least 2 elements to define any
    /// block; fewer (i.e. a 0-token sequence) short-circuits to an empty
    /// output.
    fn forward(&self, hidden_states: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        if cu_seqlens.len() < 2 {
            // No tokens to attend over; `hidden_states` is already
            // correctly shaped (0-length), nothing further to compute.
            debug_assert_eq!(
                hidden_states.dim(1)?,
                0,
                "cu_seqlens has fewer than 2 entries but hidden_states has non-zero seq_len"
            );
            return self.out_proj.forward(hidden_states);
        }

        let (batch, seq_len, _d) = hidden_states.dims3()?;

        // One gemm for q/k/v, then reshape to [B, S, 3H, D] *before*
        // splitting so narrow() operates on the head axis rather than the
        // last axis — the accumulated non-contiguity is absorbed by the
        // .contiguous() calls already required below, without adding any
        // extra copies.
        let qkv = self.qkv_proj.forward(hidden_states)?;
        let qkv = qkv.reshape((batch, seq_len, 3 * self.n_heads, self.head_dim))?;

        let query = qkv.narrow(2, 0, self.n_heads)?.transpose(1, 2)?;
        let key = qkv
            .narrow(2, self.n_heads, self.n_heads)?
            .transpose(1, 2)?;
        let value = qkv
            .narrow(2, 2 * self.n_heads, self.n_heads)?
            .transpose(1, 2)?;

        // Run attention independently per block (narrowed along the seq_len
        // axis) instead of one dense [seq_len, seq_len] matmul + mask — each
        // block only ever attends within itself, so this is O(n) total
        // matmul/softmax cost across blocks rather than O(n^2) (see
        // `build_cu_seqlens`'s doc comment).
        let n_blocks = cu_seqlens.len() - 1;

        let attn_output = if hidden_states.device().is_cpu() {
            self.forward_attn_cpu(&query, &key, &value, cu_seqlens, n_blocks)?
        } else {
            self.forward_attn_gpu(&query, &key, &value, batch, cu_seqlens, n_blocks)?
        };

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }

    /// CPU-only attention core. `query`/`key`/`value` are `[B, H, S, D]`
    /// (BHSD); each block runs through `candle_nn::attention::flash_attn`'s
    /// online-softmax kernel, which streams K/V with an `O(head_dim)` working
    /// set instead of materializing an `O(block_len^2)` score tensor, needs no
    /// f32 dtype round-trip for softmax, and parallelizes internally across
    /// the block's `H * block_len` independent (head, position) pairs —
    /// unlike `forward_attn_gpu`'s batched matmul, which dispatches one
    /// `gemm` call per (head, block) slice in a serial loop that only
    /// requests thread-parallelism per call, not across calls. Not available
    /// on GPU: candle's `flash_attn` kernels read CPU storage directly and
    /// error otherwise, so GPU callers use `forward_attn_gpu` instead.
    ///
    /// When `batch == 1` and the dtype is `F32`/`F16`, delegates to
    /// [`Self::forward_attn_cpu_varlen`], which fuses every block into a
    /// single `flash_attn_varlen_cpu` call parallelized across all blocks
    /// and heads at once, rather than issuing one `flash_attn` call (each
    /// with its own internal rayon dispatch) per block. `B > 1` or `BF16`
    /// (unsupported by the varlen kernel) fall back to the per-block loop.
    fn forward_attn_cpu(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cu_seqlens: &[usize],
        n_blocks: usize,
    ) -> Result<Tensor> {
        // BHSD [B, H, S, D] -> BSHD [B, S, H, D], as flash_attn expects.
        let q_bshd = query.transpose(1, 2)?;
        let k_bshd = key.transpose(1, 2)?;
        let v_bshd = value.transpose(1, 2)?;

        // self.scale is 1/sqrt(head_dim), always small and positive; f64->f32
        // here only drops precision flash_attn's own f32 accumulator would
        // discard anyway.
        #[allow(clippy::cast_possible_truncation)]
        let scale = self.scale as f32;

        let batch = q_bshd.dim(0)?;
        if batch == 1 && matches!(q_bshd.dtype(), DType::F32 | DType::F16) {
            return Self::forward_attn_cpu_varlen(&q_bshd, &k_bshd, &v_bshd, cu_seqlens, scale);
        }

        let mut outputs = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let start = cu_seqlens[i];
            let len = cu_seqlens[i + 1] - start;
            let q_block = q_bshd.narrow(1, start, len)?;
            let k_block = k_bshd.narrow(1, start, len)?;
            let v_block = v_bshd.narrow(1, start, len)?;
            // Each block attends only within itself (no cross-block mixing),
            // so an unmasked kernel is correct here.
            outputs.push(dispatch_flash_attn(
                &q_block,
                &k_block,
                &v_block,
                scale,
                AttnMask::None,
            )?);
        }

        if n_blocks == 1 {
            Ok(outputs.into_iter().next().expect("n_blocks == 1 checked above"))
        } else {
            Tensor::cat(&outputs, 2)
        }
    }

    /// Fused CPU path for `forward_attn_cpu`: `q_bshd`/`k_bshd`/`v_bshd` are
    /// `[1, S, H, D]` (BSHD, batch always 1 since `cu_seqlens`-based
    /// blocking is inherently a single packed sequence). Drops the batch
    /// dim to `[S, H, D]` and calls `flash_attn_varlen_cpu` once, passing
    /// each `cu_seqlens` block's length as an independent "batch" entry —
    /// matching this module's packed-sequence semantics rather than a
    /// padded-batch layout — instead of one `flash_attn` call per block.
    fn forward_attn_cpu_varlen(
        q_bshd: &Tensor,
        k_bshd: &Tensor,
        v_bshd: &Tensor,
        cu_seqlens: &[usize],
        scale: f32,
    ) -> Result<Tensor> {
        let (_batch, seq_len, n_heads, head_dim) = q_bshd.dims4()?;
        let q_packed = q_bshd.reshape((seq_len, n_heads, head_dim))?;
        let k_packed = k_bshd.reshape((seq_len, n_heads, head_dim))?;
        let v_packed = v_bshd.reshape((seq_len, n_heads, head_dim))?;

        let seqlens_vec: Vec<u32> = cu_seqlens
            .windows(2)
            .map(|w| {
                // Block lengths are bounded by `seq_len` audio tokens per
                // chunk (at most a few thousand), well within u32 range.
                #[allow(clippy::cast_possible_truncation)]
                let len = (w[1] - w[0]) as u32;
                len
            })
            .collect();
        let max_seqlen = seqlens_vec.iter().copied().max().unwrap_or(0) as usize;
        let n_blocks = seqlens_vec.len();
        let seqlens = Tensor::from_vec(seqlens_vec, n_blocks, q_packed.device())?;

        let out = flash_attn_varlen_cpu(
            &q_packed,
            &k_packed,
            &v_packed,
            None,
            &seqlens,
            &seqlens,
            max_seqlen,
            max_seqlen,
            scale,
            false,
            None,
            None,
        )?;

        // [S, H, D] -> [1, S, H, D] -> [1, H, S, D] (BHSD, matching
        // `forward_attn_gpu`'s and the per-block loop's return convention).
        out.reshape((1, seq_len, n_heads, head_dim))?.transpose(1, 2)
    }

    /// GPU attention core (also used for any non-CPU device): explicit
    /// matmul + softmax + matmul SDPA, since candle's `flash_attn` kernels
    /// are CPU-only. Most blocks share the same length
    /// (`TOKENS_PER_WINDOW`-derived), so all uniform-length blocks are
    /// batched into one cuBLAS-batched matmul call — reducing N separate
    /// kernel launches to one for the common case. Only a trailing shorter
    /// block (if any) falls back to the per-block path.
    fn forward_attn_gpu(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        batch: usize,
        cu_seqlens: &[usize],
        n_blocks: usize,
    ) -> Result<Tensor> {
        let block_len = cu_seqlens[1] - cu_seqlens[0];
        let mut n_full = n_blocks;
        for i in 1..n_blocks {
            if cu_seqlens[i + 1] - cu_seqlens[i] != block_len {
                n_full = i;
                break;
            }
        }
        // `build_cu_seqlens` groups chunks via `chunks(chunks_per_block)`,
        // which can only make the *last* group shorter than the rest — so
        // at most one trailing block may differ in length. The fallback
        // path below only handles a single trailing block; if this
        // invariant is ever violated (e.g. `cu_seqlens` built some other
        // way), the batched reshape below would silently drop all but the
        // first non-uniform block instead of processing every block.
        debug_assert!(
            n_full >= n_blocks - 1,
            "expected at most one trailing non-uniform block, got {} \
             non-uniform blocks",
            n_blocks - n_full
        );

        // Softmax in f32 for numerical stability, matching GqaAttention's
        // convention, then cast back to the working dtype.
        let softmax_f32 = |weights: &Tensor, out_dtype: DType| -> Result<Tensor> {
            candle_nn::ops::softmax(&weights.to_dtype(DType::F32)?, D::Minus1)?
                .to_dtype(out_dtype)
        };

        let mut outputs: Vec<Tensor> = Vec::with_capacity(2);

        // Batched path: reshape all uniform blocks into an extra batch
        // dimension and run a single matmul/softmax.
        if n_full > 0 {
            let total_full_len = n_full * block_len;
            let v_dtype = value.dtype();
            let q_all = query
                .narrow(2, 0, total_full_len)?
                .contiguous()?
                .reshape((batch * self.n_heads * n_full, block_len, self.head_dim))?;
            let k_all = key
                .narrow(2, 0, total_full_len)?
                .contiguous()?
                .reshape((batch * self.n_heads * n_full, block_len, self.head_dim))?;
            let v_all = value
                .narrow(2, 0, total_full_len)?
                .contiguous()?
                .reshape((batch * self.n_heads * n_full, block_len, self.head_dim))?;

            let attn_weights = (q_all.matmul(&k_all.transpose(1, 2)?)? * self.scale)?;
            let attn_weights = softmax_f32(&attn_weights, v_dtype)?;
            let batched = attn_weights
                .matmul(&v_all)?
                .reshape((batch, self.n_heads, n_full * block_len, self.head_dim))?;
            outputs.push(batched);
        }

        // Fallback: trailing shorter block, if any.
        if n_full < n_blocks {
            let start = cu_seqlens[n_full];
            let len = cu_seqlens[n_full + 1] - start;

            let q_block = query.narrow(2, start, len)?.contiguous()?;
            let k_block = key.narrow(2, start, len)?.contiguous()?;
            let v_block = value.narrow(2, start, len)?.contiguous()?;

            let attn_weights = (q_block.matmul(&k_block.transpose(2, 3)?)? * self.scale)?;
            let attn_weights = softmax_f32(&attn_weights, v_block.dtype())?;
            outputs.push(attn_weights.matmul(&v_block)?);
        }

        Tensor::cat(&outputs, 2)
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
/// h = h + self_attn(self_attn_layer_norm(h), cu_seqlens)
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

    fn forward(&self, hidden_states: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let residual = hidden_states;
        let hs = self.self_attn_layer_norm.forward(hidden_states)?;
        let hs = self.self_attn.forward(&hs, cu_seqlens)?;
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
            FRAMES_PER_WINDOW,
            config.n_window * 2,
            "FRAMES_PER_WINDOW must equal n_window * 2"
        );
        debug_assert_eq!(
            config.n_window_infer % FRAMES_PER_WINDOW,
            0,
            "n_window_infer must be a multiple of FRAMES_PER_WINDOW"
        );
        let chunks_per_block = config.n_window_infer / FRAMES_PER_WINDOW;
        debug_assert!(
            chunks_per_block >= 1,
            "n_window_infer must be at least FRAMES_PER_WINDOW ({FRAMES_PER_WINDOW})"
        );

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
        let cu_seqlens = build_cu_seqlens(n_frames, self.chunks_per_block);

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cu_seqlens)?;
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
    fn cu_seqlens_single_block() {
        // 200 frames = 2 full chunks; chunks_per_block=8 means both chunks
        // fit in one block, so cu_seqlens has a single [0, total] window.
        let cu_seqlens = build_cu_seqlens(200, 8);
        assert_eq!(cu_seqlens, vec![0, 26]);
    }

    #[test]
    fn cu_seqlens_multiple_blocks() {
        // 300 frames = 3 full chunks (13 tokens each -> 39 total). With
        // chunks_per_block=2: chunks {0,1} form block 0 (26 tokens), chunk
        // {2} forms block 1 (13 tokens).
        let cu_seqlens = build_cu_seqlens(300, 2);
        assert_eq!(cu_seqlens, vec![0, 26, 39]);
    }

    #[test]
    fn cu_seqlens_zero_frames() {
        let cu_seqlens = build_cu_seqlens(0, 8);
        assert_eq!(cu_seqlens, vec![0]);
    }

    #[test]
    #[should_panic(expected = "chunk size must be non-zero")]
    fn cu_seqlens_zero_chunks_per_block_panics() {
        // `chunks_per_block == 0` is an invalid config (guarded against by
        // the `debug_assert!` in `AudioEncoder::new`); document that
        // `build_cu_seqlens` panics rather than silently misbehaving.
        let _ = build_cu_seqlens(200, 0);
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
        // input must leave block 0's output untouched — proving the
        // cu_seqlens block boundaries are wired through
        // `AudioEncoderLayer::forward`, not just structurally correct in
        // isolation (unlike the `v_proj=0` layer tests above, which would
        // pass identically even if the boundaries were silently dropped).
        let config = small_encoder_config();
        let vb = make_nonzero_attn_layer_vb(&config, 0.01);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        // 200 frames = 2 chunks; chunks_per_block=1 puts each chunk in its
        // own block, so tokens 0..13 and 13..26 must be mutually invisible.
        let block_len = TOKENS_PER_WINDOW;
        let total = 2 * block_len;
        let cu_seqlens = build_cu_seqlens(200, 1);
        assert_eq!(cu_seqlens, vec![0, block_len, total]);

        let base = Tensor::randn(0f32, 1f32, (1, total, config.d_model), &Device::Cpu)
            .expect("randn base");
        let perturbation = Tensor::randn(0f32, 1f32, (1, block_len, config.d_model), &Device::Cpu)
            .expect("randn perturbation");

        let block0 = base.narrow(1, 0, block_len).expect("narrow block0");
        let block1 = base.narrow(1, block_len, block_len).expect("narrow block1");
        let block1_perturbed = (&block1 + &perturbation).expect("add");
        let perturbed = Tensor::cat(&[&block0, &block1_perturbed], 1).expect("cat");

        let out_base = layer.forward(&base, &cu_seqlens).expect("forward base");
        let out_perturbed = layer
            .forward(&perturbed, &cu_seqlens)
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
            &out_perturbed
                .narrow(1, block_len, block_len)
                .expect("narrow"),
        );
        assert!(
            diff_block1 > 1e-4,
            "block 1 output must change when its own input changes, diff={diff_block1}"
        );
    }

    #[test]
    fn encoder_layer_cu_seqlens_isolates_three_blocks() {
        // 300 frames = 3 chunks; chunks_per_block=1 puts each chunk in its
        // own block, giving 3 blocks of `TOKENS_PER_WINDOW` tokens each.
        // Unlike `encoder_layer_mask_isolates_blocks` (2 blocks, so neither
        // block has a neighbor on both sides), this exercises the middle
        // block flanked on both sides — the case most likely to expose an
        // off-by-one in `cu_seqlens.windows(2)`'s start/end computation.
        let config = small_encoder_config();
        let vb = make_nonzero_attn_layer_vb(&config, 0.01);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        let block_len = TOKENS_PER_WINDOW;
        let total = 3 * block_len;
        let cu_seqlens = build_cu_seqlens(300, 1);
        assert_eq!(cu_seqlens, vec![0, block_len, 2 * block_len, total]);

        let base = Tensor::randn(0f32, 1f32, (1, total, config.d_model), &Device::Cpu)
            .expect("randn base");

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

        let out_base = layer.forward(&base, &cu_seqlens).expect("forward base");

        for perturbed_block in 0..3 {
            let perturbation =
                Tensor::randn(0f32, 1f32, (1, block_len, config.d_model), &Device::Cpu)
                    .expect("randn perturbation");

            let mut blocks: Vec<Tensor> = (0..3)
                .map(|b| base.narrow(1, b * block_len, block_len).expect("narrow"))
                .collect();
            blocks[perturbed_block] = (&blocks[perturbed_block] + &perturbation).expect("add");
            let perturbed = Tensor::cat(&blocks.iter().collect::<Vec<_>>(), 1).expect("cat");

            let out_perturbed = layer
                .forward(&perturbed, &cu_seqlens)
                .expect("forward perturbed");

            for other_block in 0..3 {
                let diff = max_abs_diff(
                    &out_base
                        .narrow(1, other_block * block_len, block_len)
                        .expect("narrow"),
                    &out_perturbed
                        .narrow(1, other_block * block_len, block_len)
                        .expect("narrow"),
                );
                if other_block == perturbed_block {
                    assert!(
                        diff > 1e-4,
                        "block {other_block} output must change when its own input \
                         changes, diff={diff}"
                    );
                } else {
                    assert!(
                        diff < 1e-5,
                        "block {other_block} output must be unaffected by block \
                         {perturbed_block}'s input, diff={diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn encoder_layer_output_shape() {
        let config = small_encoder_config();
        let vb = make_layer_vb(&config);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        let hidden =
            Tensor::randn(0f32, 1f32, (1, 5, config.d_model), &Device::Cpu).expect("randn");
        let out = layer.forward(&hidden, &[0, 5]).expect("forward");
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
        let out = layer.forward(&hidden, &[0, 5]).expect("forward");

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
    fn encoder_single_chunk_single_block() {
        // 100 frames = 1 chunk, well within a single ~8s attention block, so
        // cu_seqlens is just [0, total_tokens] (one block, all tokens
        // mutually visible).
        let config = small_encoder_config();
        let vb = make_encoder_vb(&config);
        let encoder = AudioEncoder::new(&config, vb).expect("new");

        let mel =
            Tensor::zeros((1, config.num_mel_bins, 100), DType::F32, &Device::Cpu).expect("mel");
        let out = encoder.forward(&mel).expect("forward");
        assert_eq!(out.dims(), &[1, TOKENS_PER_WINDOW, config.d_model]);
    }

    #[test]
    fn encoder_batched_attention_with_trailing_shorter_block() {
        // Shape-level sanity check that multi-block attention is wired
        // correctly through `AudioEncoderLayer::forward`: 1350 frames = 13
        // full chunks (13 tokens each) + 50-frame remainder (7 tokens) =
        // 176 total tokens. With chunks_per_block=8, block 0 has 8 chunks
        // (104 tokens), block 1 has 5 full + remainder (72 tokens). These
        // are CPU tensors, so both blocks run through `forward_attn_cpu`'s
        // per-block flash-attention loop (`forward_attn_gpu`'s batched-vs-
        // fallback matmul split only applies to non-CPU devices). Numeric
        // correctness is covered separately by
        // `cpu_flash_attn_matches_per_block_reference` and
        // `gpu_attention_matches_per_block_reference` below.
        let config = small_encoder_config();
        let vb = make_layer_vb(&config);
        let layer = AudioEncoderLayer::new(&config, vb).expect("new");

        let n_frames = 1350;
        let (n_full_chunks, remainder) = chunk_split(n_frames);
        assert_eq!(n_full_chunks, 13);
        assert_eq!(remainder, 50);

        let cu_seqlens = build_cu_seqlens(n_frames, 8);
        let total_tokens = get_feat_extract_output_lengths(n_frames);
        assert_eq!(*cu_seqlens.last().unwrap(), total_tokens);
        assert_eq!(cu_seqlens.len(), 3);
        // Block 0 = 8 full chunks = 8 * 13 tokens.
        assert_eq!(cu_seqlens[1] - cu_seqlens[0], 8 * TOKENS_PER_WINDOW);

        let hidden =
            Tensor::randn(0f32, 1f32, (1, total_tokens, config.d_model), &Device::Cpu)
                .expect("randn");
        let out = layer.forward(&hidden, &cu_seqlens).expect("forward");
        assert_eq!(out.dims(), &[1, total_tokens, config.d_model]);
    }

    /// Shared fixture for `gpu_attention_matches_per_block_reference` and
    /// `cpu_flash_attn_matches_per_block_reference` below: builds an
    /// `AudioEncoderAttention` with small non-zero weights, the BHSD
    /// query/key/value tensors `AudioEncoderAttention::forward` would
    /// compute internally, and a from-scratch per-block matmul/softmax/
    /// matmul reference output — the only way to catch a `batch * n_heads *
    /// n_full` reshape/ordering regression, which a shape-only assertion
    /// would miss.
    ///
    /// 350 frames = 3 full chunks (13 tokens each) + 50-frame remainder (7
    /// tokens). With chunks_per_block=1, each full chunk gets its own
    /// block: cu_seqlens = [0, 13, 26, 39, 46], i.e. 3 uniform full blocks
    /// (n_full=3, enough to exercise `forward_attn_gpu`'s batched path)
    /// followed by one shorter 7-token block.
    fn attention_reference_fixture() -> (
        AudioEncoderAttention,
        Tensor,
        Tensor,
        Tensor,
        Vec<usize>,
        usize,
        usize,
        Tensor,
    ) {
        let config = small_encoder_config();
        let vb = make_nonzero_attn_layer_vb(&config, 0.01);
        let attn = AudioEncoderAttention::new(&config, vb.pp("self_attn")).expect("new");

        let n_frames = 350;
        let cu_seqlens = build_cu_seqlens(n_frames, 1);
        assert_eq!(cu_seqlens, vec![0, 13, 26, 39, 46]);
        let n_full = 3;
        assert_eq!(cu_seqlens[n_full + 1] - cu_seqlens[n_full], 7);

        let total_tokens = *cu_seqlens.last().unwrap();
        let hidden =
            Tensor::randn(0f32, 1f32, (1, total_tokens, config.d_model), &Device::Cpu)
                .expect("randn hidden");
        let (batch, seq_len, _d) = hidden.dims3().expect("dims3");

        // query/key/value BHSD via attn's own merged `qkv_proj`, mirroring
        // `AudioEncoderAttention::forward`'s internal split.
        let qkv = attn.qkv_proj.forward(&hidden).expect("qkv_proj");
        let qkv = qkv
            .reshape((batch, seq_len, 3 * attn.n_heads, attn.head_dim))
            .expect("reshape qkv");
        let query = qkv
            .narrow(2, 0, attn.n_heads)
            .expect("narrow q")
            .transpose(1, 2)
            .expect("transpose q");
        let key = qkv
            .narrow(2, attn.n_heads, attn.n_heads)
            .expect("narrow k")
            .transpose(1, 2)
            .expect("transpose k");
        let value = qkv
            .narrow(2, 2 * attn.n_heads, attn.n_heads)
            .expect("narrow v")
            .transpose(1, 2)
            .expect("transpose v");

        // From-scratch per-block reference: `make_nonzero_attn_layer_vb` is
        // deterministic, so a second call rebuilds identical q/k/v weights
        // independent of `attn`'s merged `qkv_proj` — then run each block's
        // attention independently via plain narrow/matmul/softmax/matmul,
        // exactly mirroring the pre-optimization loop that this commit
        // replaced.
        let vb_ref = make_nonzero_attn_layer_vb(&config, 0.01);
        let sa_ref = vb_ref.pp("self_attn");
        let q_proj_ref = with_tracing::linear(config.d_model, config.d_model, sa_ref.pp("q_proj"))
            .expect("q_proj_ref");
        let k_proj_ref = with_tracing::linear(config.d_model, config.d_model, sa_ref.pp("k_proj"))
            .expect("k_proj_ref");
        let v_proj_ref = with_tracing::linear(config.d_model, config.d_model, sa_ref.pp("v_proj"))
            .expect("v_proj_ref");

        let shape = (batch, seq_len, attn.n_heads, attn.head_dim);
        let q_ref = q_proj_ref
            .forward(&hidden)
            .expect("q_proj")
            .reshape(shape)
            .expect("reshape q")
            .transpose(1, 2)
            .expect("transpose q");
        let k_ref = k_proj_ref
            .forward(&hidden)
            .expect("k_proj")
            .reshape(shape)
            .expect("reshape k")
            .transpose(1, 2)
            .expect("transpose k");
        let v_ref = v_proj_ref
            .forward(&hidden)
            .expect("v_proj")
            .reshape(shape)
            .expect("reshape v")
            .transpose(1, 2)
            .expect("transpose v");

        let mut ref_outputs = Vec::with_capacity(cu_seqlens.len() - 1);
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let len = window[1] - start;

            let q_block = q_ref
                .narrow(2, start, len)
                .expect("narrow q")
                .contiguous()
                .expect("contig q");
            let k_block = k_ref
                .narrow(2, start, len)
                .expect("narrow k")
                .contiguous()
                .expect("contig k");
            let v_block = v_ref
                .narrow(2, start, len)
                .expect("narrow v")
                .contiguous()
                .expect("contig v");

            let attn_weights = (q_block.matmul(&k_block.transpose(2, 3).expect("transpose k_block"))
                .expect("matmul qk")
                * attn.scale)
                .expect("scale");
            let attn_weights = candle_nn::ops::softmax(
                &attn_weights.to_dtype(DType::F32).expect("to f32"),
                D::Minus1,
            )
            .expect("softmax")
            .to_dtype(v_block.dtype())
            .expect("to dtype");
            ref_outputs.push(attn_weights.matmul(&v_block).expect("matmul av"));
        }
        let ref_attn_output = Tensor::cat(&ref_outputs, 2).expect("cat");
        let ref_out = finish_attn_output(&attn, &ref_attn_output, batch, seq_len);

        (attn, query, key, value, cu_seqlens, batch, seq_len, ref_out)
    }

    /// Mirrors `AudioEncoderAttention::forward`'s post-attention transpose
    /// + reshape + `out_proj`, so tests can invoke `forward_attn_cpu`/
    /// `forward_attn_gpu` directly and still compare like-for-like against
    /// `attn.forward`'s output shape.
    fn finish_attn_output(
        attn: &AudioEncoderAttention,
        attn_output: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Tensor {
        let out = attn_output
            .transpose(1, 2)
            .expect("transpose out")
            .contiguous()
            .expect("contig out")
            .reshape((batch, seq_len, attn.n_heads * attn.head_dim))
            .expect("reshape out");
        attn.out_proj.forward(&out).expect("out_proj")
    }

    #[test]
    fn gpu_attention_matches_per_block_reference() {
        // Numeric-equivalence check for `forward_attn_gpu`'s batched-matmul
        // optimization, invoked directly since `attn.forward` would
        // otherwise dispatch CPU tensors to `forward_attn_cpu` instead (see
        // `attention_reference_fixture`'s doc comment for the block
        // layout).
        let (attn, query, key, value, cu_seqlens, batch, seq_len, ref_out) =
            attention_reference_fixture();
        let n_blocks = cu_seqlens.len() - 1;

        let attn_output = attn
            .forward_attn_gpu(&query, &key, &value, batch, &cu_seqlens, n_blocks)
            .expect("forward_attn_gpu");
        let out = finish_attn_output(&attn, &attn_output, batch, seq_len);

        let max_abs_diff = (&out - &ref_out)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            max_abs_diff < 1e-5,
            "forward_attn_gpu output must match per-block reference, \
             max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn cpu_flash_attn_matches_per_block_reference() {
        // Numeric-equivalence check for `forward_attn_cpu`'s flash-attention
        // path, invoked directly and compared against the same reference as
        // `gpu_attention_matches_per_block_reference` above.
        let (attn, query, key, value, cu_seqlens, batch, seq_len, ref_out) =
            attention_reference_fixture();
        let n_blocks = cu_seqlens.len() - 1;

        let attn_output = attn
            .forward_attn_cpu(&query, &key, &value, &cu_seqlens, n_blocks)
            .expect("forward_attn_cpu");
        let out = finish_attn_output(&attn, &attn_output, batch, seq_len);

        let max_abs_diff = (&out - &ref_out)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            max_abs_diff < 1e-5,
            "forward_attn_cpu output must match per-block reference, \
             max_abs_diff={max_abs_diff}"
        );
    }
}
