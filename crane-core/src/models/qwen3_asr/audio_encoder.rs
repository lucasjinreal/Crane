//! Conv2d frontend, chunking, and sinusoidal positional encoding for the
//! Qwen3-ASR audio encoder ("`AuT`").
//!
//! Converts a log-mel spectrogram `[1, n_mels, n_frames]` from
//! [`WhisperFeatureExtractor`](super::feature_extractor::WhisperFeatureExtractor)
//! into `[1, total_tokens, d_model]` token embeddings:
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
//! 5. Concatenate all chunk outputs along the time axis.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Conv2dConfig, VarBuilder};

use super::config::AudioConfig;
use super::feature_extractor::{FRAMES_PER_WINDOW, conv_output_len};
use crate::models::with_tracing;

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

        let n_full_chunks = n_frames / FRAMES_PER_WINDOW;
        let remainder = n_frames % FRAMES_PER_WINDOW;

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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::DType;

    use super::super::feature_extractor::{TOKENS_PER_WINDOW, get_feat_extract_output_lengths};
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
        let post_conv_mel =
            conv_output_len(conv_output_len(conv_output_len(config.num_mel_bins)));
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
        t.insert("conv_out.weight".into(), zeros2((config.d_model, conv_flat)));

        VarBuilder::from_tensors(t, dtype, device)
    }

    #[test]
    fn sinusoidal_table_shape() {
        let table =
            build_sinusoidal_table(13, 896, &Device::Cpu, DType::F32).expect("build table");
        assert_eq!(table.dims(), &[13, 896]);
    }

    #[test]
    fn sinusoidal_table_values_are_bounded() {
        let table =
            build_sinusoidal_table(13, 896, &Device::Cpu, DType::F32).expect("build table");
        let data = table
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        for &v in &data {
            assert!((-1.0..=1.0).contains(&v), "sin/cos value {v} out of [-1, 1]");
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
            assert!(row0[i].abs() < 1e-6, "sin at pos 0 should be 0, got {}", row0[i]);
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
        assert!(diff < 1e-5, "zero-weight output should equal pos_embed, max_diff={diff}");
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
        assert!(diff < 1e-6, "positions should restart each chunk, max_diff={diff}");
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
        let post_conv_mel =
            conv_output_len(conv_output_len(conv_output_len(config.num_mel_bins)));
        let conv_flat = dhs * post_conv_mel;

        let device = &Device::Cpu;
        let dtype = DType::F32;
        let zeros4 =
            |shape: (usize, usize, usize, usize)| Tensor::zeros(shape, dtype, device).expect("zeros");
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
        assert!((g2 - g1).abs() > 0.5, "channels 1 and 2 should differ: g1={g1}, g2={g2}");
        assert!((g3 - g2).abs() > 0.5, "channels 2 and 3 should differ: g2={g2}, g3={g3}");

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
}
