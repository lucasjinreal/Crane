use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    conv1d, conv1d_no_bias, conv_transpose1d, layer_norm, linear, linear_no_bias, Conv1d,
    Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, RmsNorm, VarBuilder,
};
use serde::Deserialize;

fn default_codebook_size() -> usize {
    2048
}
fn default_hidden_size() -> usize {
    1024
}
fn default_latent_dim() -> usize {
    1024
}
fn default_codebook_dim() -> usize {
    1024
}
fn default_max_position_embeddings() -> usize {
    8000
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_num_heads() -> usize {
    16
}
fn default_head_dim() -> usize {
    64
}
fn default_attention_bias() -> bool {
    false
}
fn default_sliding_window() -> usize {
    72
}
fn default_intermediate_size() -> usize {
    3072
}
fn default_layer_scale_init() -> f64 {
    0.01
}
fn default_rms_eps() -> f64 {
    1e-5
}
fn default_num_layers() -> usize {
    8
}
fn default_num_quantizers() -> usize {
    16
}
fn default_upsample_rates() -> Vec<usize> {
    vec![8, 5, 4, 3]
}
fn default_upsampling_ratios() -> Vec<usize> {
    vec![2, 2]
}
fn default_decoder_dim() -> usize {
    1536
}
fn default_output_sample_rate() -> u32 {
    24000
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_enc_num_filters")]
    pub num_filters: usize,
    #[serde(default = "default_enc_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_enc_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_enc_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_enc_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_enc_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_enc_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_enc_kernel_size")]
    pub kernel_size: usize,
    #[serde(default = "default_enc_last_kernel_size")]
    pub last_kernel_size: usize,
    #[serde(default = "default_enc_residual_kernel_size")]
    pub residual_kernel_size: usize,
    #[serde(default = "default_enc_num_residual_layers")]
    pub num_residual_layers: usize,
    #[serde(default = "default_enc_upsampling_ratios")]
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "default_enc_codebook_dim")]
    pub codebook_dim: usize,
    #[serde(default = "default_enc_codebook_size")]
    pub codebook_size: usize,
    #[serde(default = "default_enc_num_quantizers")]
    pub num_quantizers: usize,
    #[serde(default = "default_enc_num_semantic_quantizers")]
    pub num_semantic_quantizers: usize,
    #[serde(default = "default_enc_layer_scale_init")]
    pub layer_scale_initial_scale: f64,
    #[serde(default = "default_enc_rms_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_enc_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_enc_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_enc_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_enc_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_enc_upsample_groups")]
    pub upsample_groups: usize,
    #[serde(default = "default_enc_vq_hidden_dim")]
    pub vector_quantization_hidden_dimension: usize,
}
fn default_enc_num_filters() -> usize { 64 }
fn default_enc_hidden_size() -> usize { 512 }
fn default_enc_intermediate_size() -> usize { 2048 }
fn default_enc_num_hidden_layers() -> usize { 8 }
fn default_enc_num_attention_heads() -> usize { 8 }
fn default_enc_num_key_value_heads() -> usize { 8 }
fn default_enc_head_dim() -> usize { 64 }
fn default_enc_kernel_size() -> usize { 7 }
fn default_enc_last_kernel_size() -> usize { 3 }
fn default_enc_residual_kernel_size() -> usize { 3 }
fn default_enc_num_residual_layers() -> usize { 1 }
fn default_enc_upsampling_ratios() -> Vec<usize> { vec![8, 6, 5, 4] }
fn default_enc_codebook_dim() -> usize { 256 }
fn default_enc_codebook_size() -> usize { 2048 }
fn default_enc_num_quantizers() -> usize { 32 }
fn default_enc_num_semantic_quantizers() -> usize { 1 }
fn default_enc_layer_scale_init() -> f64 { 0.01 }
fn default_enc_rms_eps() -> f64 { 1e-5 }
fn default_enc_rope_theta() -> f64 { 10000.0 }
fn default_enc_max_pos() -> usize { 8000 }
fn default_enc_sliding_window() -> usize { 250 }
fn default_enc_attention_bias() -> bool { false }
fn default_enc_upsample_groups() -> usize { 512 }
fn default_enc_vq_hidden_dim() -> usize { 256 }

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            num_filters: default_enc_num_filters(),
            hidden_size: default_enc_hidden_size(),
            intermediate_size: default_enc_intermediate_size(),
            num_hidden_layers: default_enc_num_hidden_layers(),
            num_attention_heads: default_enc_num_attention_heads(),
            num_key_value_heads: default_enc_num_key_value_heads(),
            head_dim: default_enc_head_dim(),
            kernel_size: default_enc_kernel_size(),
            last_kernel_size: default_enc_last_kernel_size(),
            residual_kernel_size: default_enc_residual_kernel_size(),
            num_residual_layers: default_enc_num_residual_layers(),
            upsampling_ratios: default_enc_upsampling_ratios(),
            codebook_dim: default_enc_codebook_dim(),
            codebook_size: default_enc_codebook_size(),
            num_quantizers: default_enc_num_quantizers(),
            num_semantic_quantizers: default_enc_num_semantic_quantizers(),
            layer_scale_initial_scale: default_enc_layer_scale_init(),
            norm_eps: default_enc_rms_eps(),
            rope_theta: default_enc_rope_theta(),
            max_position_embeddings: default_enc_max_pos(),
            sliding_window: default_enc_sliding_window(),
            attention_bias: default_enc_attention_bias(),
            upsample_groups: default_enc_upsample_groups(),
            vector_quantization_hidden_dimension: default_enc_vq_hidden_dim(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerV2Config {
    pub decoder_config: DecoderConfig,
    #[serde(default)]
    pub encoder_config: EncoderConfig,
    #[serde(default = "default_output_sample_rate")]
    pub output_sample_rate: u32,
    #[serde(default = "default_encoder_valid_num_quantizers")]
    pub encoder_valid_num_quantizers: usize,
}
fn default_encoder_valid_num_quantizers() -> usize { 16 }

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,
    #[serde(default = "default_codebook_dim")]
    pub codebook_dim: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default = "default_layer_scale_init")]
    pub layer_scale_initial_scale: f64,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<usize>,
    #[serde(default = "default_upsampling_ratios")]
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "default_decoder_dim")]
    pub decoder_dim: usize,
}

impl DecoderConfig {
    fn total_upsample(&self) -> usize {
        self.upsample_rates
            .iter()
            .chain(self.upsampling_ratios.iter())
            .product()
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_pos: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        Ok(Self {
            cos_table: freqs.cos()?,
            sin_table: freqs.sin()?,
        })
    }

    fn forward(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, 0, seq_len)?;
        let sin = self.sin_table.narrow(0, 0, seq_len)?;
        Ok((cos, sin))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_h, t, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, kv_h, n_rep, t, d))?
        .reshape((b, kv_h * n_rep, t, d))?)
}

fn causal_sliding_mask(seq_len: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let blocked = j > i || i.saturating_sub(j) >= window;
            if blocked {
                data[i * seq_len + j] = -1e9;
            }
        }
    }
    Ok(Tensor::new(data.as_slice(), device)?.reshape((1, 1, seq_len, seq_len))?)
}

#[derive(Debug, Clone)]
struct TokenizerAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    rotary: RotaryEmbedding,
}

impl TokenizerAttention {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let make = |in_d, out_d, name: &str| -> Result<Linear> {
            if cfg.attention_bias {
                Ok(linear(in_d, out_d, vb.pp(name))?)
            } else {
                Ok(linear_no_bias(in_d, out_d, vb.pp(name))?)
            }
        };
        Ok(Self {
            q_proj: make(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                "q_proj",
            )?,
            k_proj: make(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                "k_proj",
            )?,
            v_proj: make(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                "v_proj",
            )?,
            o_proj: make(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                "o_proj",
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            sliding_window: cfg.sliding_window,
            rotary: RotaryEmbedding::new(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                vb.device(),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b, t, c) = hidden.dims3()?;
        let q = hidden
            .apply(&self.q_proj)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = hidden
            .apply(&self.k_proj)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = hidden
            .apply(&self.v_proj)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (cos, sin) = self.rotary.forward(t)?;
        let target_dtype = q.dtype();
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?
            .to_dtype(target_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?
            .to_dtype(target_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?.contiguous()?;
        let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?.contiguous()?;

        let rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, rep)?.contiguous()?;
        let v = repeat_kv(&v, rep)?.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q
            .matmul(&k_t)
            .map_err(|e| anyhow::anyhow!(
                "TokenizerAttention q@k^T matmul failed (q={:?}, k_t={:?}): {e}",
                q.dims(),
                k_t.dims()
            ))?
            * scale)?;
        let mask = causal_sliding_mask(t, self.sliding_window, hidden.device())?.to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.contiguous()?;
        let out = probs
            .matmul(&v)
            .map_err(|e| anyhow::anyhow!(
                "TokenizerAttention attn@v matmul failed (attn={:?}, v={:?}): {e}",
                probs.dims(),
                v.dims()
            ))?
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(channels: usize, init: f64, vb: VarBuilder) -> Result<Self> {
        let scale = match vb.get(channels, "scale") {
            Ok(v) => v,
            Err(_) => {
                let v = vec![init as f32; channels];
                Tensor::new(v.as_slice(), vb.device())?
            }
        };
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.scale.dims1()?;
        Ok(x.broadcast_mul(&self.scale.reshape((1, 1, c))?)?)
    }
}

#[derive(Debug, Clone)]
struct TokenizerMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TokenizerMlp {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = x.apply(&self.gate_proj)?.silu()?;
        let u = x.apply(&self.up_proj)?;
        Ok(g.broadcast_mul(&u)?.apply(&self.down_proj)?)
    }
}

#[derive(Debug, Clone)]
struct TokenizerTransformerLayer {
    self_attn: TokenizerAttention,
    mlp: TokenizerMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
}

impl TokenizerTransformerLayer {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: TokenizerAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: TokenizerMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            self_attn_layer_scale: LayerScale::new(
                cfg.hidden_size,
                cfg.layer_scale_initial_scale,
                vb.pp("self_attn_layer_scale"),
            )?,
            mlp_layer_scale: LayerScale::new(
                cfg.hidden_size,
                cfg.layer_scale_initial_scale,
                vb.pp("mlp_layer_scale"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let residual = hidden.clone();
        let hidden = hidden.apply(&self.input_layernorm)?;
        let hidden = self.self_attn.forward(&hidden)?;
        let hidden = (residual + self.self_attn_layer_scale.forward(&hidden)?)?;

        let residual = hidden.clone();
        let hidden = hidden.apply(&self.post_attention_layernorm)?;
        let hidden = self.mlp.forward(&hidden)?;
        Ok((residual + self.mlp_layer_scale.forward(&hidden)?)?)
    }
}

#[derive(Debug, Clone)]
struct TokenizerTransformer {
    input_proj: Linear,
    output_proj: Linear,
    layers: Vec<TokenizerTransformerLayer>,
    norm: RmsNorm,
}

impl TokenizerTransformer {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TokenizerTransformerLayer::new(cfg, vb.pp("layers").pp(i))?);
        }
        Ok(Self {
            input_proj: linear(cfg.latent_dim, cfg.hidden_size, vb.pp("input_proj"))?,
            output_proj: linear(cfg.hidden_size, cfg.latent_dim, vb.pp("output_proj"))?,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
        })
    }

    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let mut hidden = inputs.apply(&self.input_proj)?;
        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden.apply(&self.norm)?.apply(&self.output_proj)?)
    }
}

#[derive(Debug, Clone)]
struct CausalConvNet {
    conv: Conv1d,
    stride: usize,
    kernel_size: usize,
    padding: usize,
}

impl CausalConvNet {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig { stride, dilation, groups, ..Default::default() };
        let conv = conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let effective = (kernel_size - 1) * dilation + 1;
        Ok(Self { conv, stride, kernel_size: effective, padding: effective.saturating_sub(stride) })
    }

    fn new_no_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig { stride, dilation, groups, ..Default::default() };
        let conv = conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let effective = (kernel_size - 1) * dilation + 1;
        Ok(Self { conv, stride, kernel_size: effective, padding: effective.saturating_sub(stride) })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let length = hidden.dim(D::Minus1)?;
        let n_frames = (length as f64 - self.kernel_size as f64 + self.padding as f64)
            / self.stride as f64
            + 1.0;
        let ideal_len = ((n_frames.ceil() as usize).saturating_sub(1)) * self.stride
            + (self.kernel_size.saturating_sub(self.padding));
        let extra_padding = ideal_len.saturating_sub(length);
        let hidden = hidden.pad_with_zeros(D::Minus1, self.padding, extra_padding)?;
        Ok(hidden.apply(&self.conv)?)
    }
}

#[derive(Debug, Clone)]
struct CausalTransConvNet {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalTransConvNet {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            ConvTranspose1dConfig {
                stride,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self {
            conv,
            right_pad: kernel_size.saturating_sub(stride),
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = hidden.apply(&self.conv)?;
        if self.right_pad == 0 {
            Ok(hidden)
        } else {
            let t = hidden.dim(D::Minus1)?;
            Ok(hidden.narrow(D::Minus1, 0, t.saturating_sub(self.right_pad))?)
        }
    }
}

#[derive(Debug, Clone)]
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get(channels, "alpha")?,
            beta: vb.get(channels, "beta")?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let c = self.alpha.dims1()?;
        let alpha = self.alpha.reshape((1, c, 1))?.exp()?;
        let beta = self.beta.reshape((1, c, 1))?.exp()?;
        let periodic = hidden.broadcast_mul(&alpha)?.sin()?.sqr()?;
        let gain = (&beta + 1e-9)?.recip()?;
        let mut periodic_gain = gain.broadcast_mul(&periodic)?;
        if periodic_gain.dtype() != hidden.dtype() {
            periodic_gain = periodic_gain.to_dtype(hidden.dtype())?;
        }
        Ok((hidden + periodic_gain)?)
    }
}

#[derive(Debug, Clone)]
struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConvNet,
    act2: SnakeBeta,
    conv2: CausalConvNet,
}

impl DecoderResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, vb.pp("act1"))?,
            conv1: CausalConvNet::new(dim, dim, 7, 1, dilation, 1, vb.pp("conv1"))?,
            act2: SnakeBeta::new(dim, vb.pp("act2"))?,
            conv2: CausalConvNet::new(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let residual = hidden.clone();
        let hidden = self.act1.forward(hidden)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;
        Ok((hidden + residual)?)
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    first_act: SnakeBeta,
    upsample: CausalTransConvNet,
    res1: DecoderResidualUnit,
    res2: DecoderResidualUnit,
    res3: DecoderResidualUnit,
}

impl DecoderBlock {
    fn new(cfg: &DecoderConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let in_dim = cfg.decoder_dim / (1usize << layer_idx);
        let out_dim = cfg.decoder_dim / (1usize << (layer_idx + 1));
        let up = cfg.upsample_rates[layer_idx];
        let block = vb.pp("block");
        Ok(Self {
            first_act: SnakeBeta::new(in_dim, block.pp(0))?,
            upsample: CausalTransConvNet::new(in_dim, out_dim, 2 * up, up, block.pp(1))?,
            res1: DecoderResidualUnit::new(out_dim, 1, block.pp(2))?,
            res2: DecoderResidualUnit::new(out_dim, 3, block.pp(3))?,
            res3: DecoderResidualUnit::new(out_dim, 9, block.pp(4))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.first_act.forward(hidden)?;
        let hidden = self.upsample.forward(&hidden)?;
        let hidden = self.res1.forward(&hidden)?;
        let hidden = self.res2.forward(&hidden)?;
        self.res3.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv: CausalConvNet,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dwconv: CausalConvNet::new(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?,
            norm: layer_norm(dim, 1e-6, vb.pp("norm"))?,
            pwconv1: linear(dim, 4 * dim, vb.pp("pwconv1"))?,
            pwconv2: linear(4 * dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let residual = hidden.clone();
        let hidden = self.dwconv.forward(hidden)?;
        let hidden = hidden.transpose(1, 2)?;
        let hidden = hidden.apply(&self.norm)?;
        let hidden = hidden.apply(&self.pwconv1)?.gelu_erf()?;
        let hidden = hidden.apply(&self.pwconv2)?;
        let c = self.gamma.dims1()?;
        let hidden = hidden.broadcast_mul(&self.gamma.reshape((1, 1, c))?)?;
        let hidden = hidden.transpose(1, 2)?;
        Ok((hidden + residual)?)
    }
}

#[derive(Debug, Clone)]
struct EuclideanCodebook {
    cluster_usage: Tensor,
    embedding_sum: Tensor,
    epsilon: f64,
}

impl EuclideanCodebook {
    fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        let embedding_sum = vb.get((codebook_size, dim), "embedding_sum")?;
        let cluster_usage = match vb.get(codebook_size, "cluster_usage") {
            Ok(v) => v,
            Err(_) => {
                let ones = vec![1f32; codebook_size];
                Tensor::new(ones.as_slice(), vb.device())?.to_dtype(embedding_sum.dtype())?
            }
        };

        Ok(Self {
            cluster_usage,
            embedding_sum,
            epsilon: 1e-5,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (b, t) = codes.dims2()?;
        let usage = self
            .cluster_usage
            .clamp(self.epsilon, f64::INFINITY)?
            .reshape((self.cluster_usage.dims1()?, 1))?;
        let embedding = self.embedding_sum.broadcast_div(&usage)?;
        let flat_codes = codes.flatten_all()?;
        let quantized = embedding.embedding(&flat_codes)?;
        Ok(quantized.reshape((b, t, self.embedding_sum.dims2()?.1))?)
    }
}

#[derive(Debug, Clone)]
struct VectorQuantization {
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            codebook: EuclideanCodebook::new(dim, codebook_size, vb.pp("_codebook"))?,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let quantized = self.codebook.decode(codes)?;
        Ok(quantized.transpose(1, 2)?)
    }
}

#[derive(Debug, Clone)]
struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
    dim: usize,
}

impl ResidualVectorQuantization {
    fn new(num_quantizers: usize, dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_quantizers);
        for i in 0..num_quantizers {
            layers.push(VectorQuantization::new(dim, codebook_size, vb.pp("layers").pp(i))?);
        }
        Ok(Self { layers, dim })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (_, k, _) = codes.dims3()?;
        if k > self.layers.len() {
            anyhow::bail!("Too many quantizers: got {k}, model has {}", self.layers.len());
        }
        let mut sum: Option<Tensor> = None;
        for idx in 0..k {
            let layer_codes = codes.i((.., idx, ..))?;
            let mut q = self.layers[idx].decode(&layer_codes)?;
            sum = Some(match sum {
                Some(acc) => {
                    if q.dtype() != acc.dtype() {
                        q = q.to_dtype(acc.dtype())?;
                    }
                    (acc + q)?
                }
                None => q,
            });
        }
        sum.ok_or_else(|| anyhow::anyhow!("No quantizer outputs to decode"))
    }
}

#[derive(Debug, Clone)]
struct ResidualVectorQuantizer {
    vq: ResidualVectorQuantization,
    output_proj: PointwiseProjNoBias,
}

#[derive(Debug, Clone)]
struct PointwiseProjNoBias {
    weight_t: Tensor, // [in, out]
}

impl PointwiseProjNoBias {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w = match vb.get((out_dim, in_dim, 1), "weight") {
            Ok(w) => w.squeeze(2)?,
            Err(_) => vb.get((out_dim, in_dim), "weight")?,
        };
        Ok(Self {
            weight_t: w.transpose(0, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?; // [B, T, C]
        let (b, t, c) = x.dims3()?;
        let o = self.weight_t.dims2()?.1;
        let x2 = x.reshape((b * t, c))?;
        let y2 = x2.matmul(&self.weight_t)?; // [B*T, O]
        let y = y2.reshape((b, t, o))?;
        Ok(y.transpose(1, 2)?) // [B, O, T]
    }
}

impl ResidualVectorQuantizer {
    fn new(n_q: usize, dimension: usize, bins: usize, output_dimension: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            vq: ResidualVectorQuantization::new(n_q, dimension, bins, vb.pp("vq"))?,
            output_proj: PointwiseProjNoBias::new(dimension, output_dimension, vb.pp("output_proj"))?,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        self.output_proj.forward(&self.vq.decode(codes)?)
    }
}

#[derive(Debug, Clone)]
struct SplitResidualVectorQuantizer {
    n_q_semantic: usize,
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
}

impl SplitResidualVectorQuantizer {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let n_q_semantic = 1usize;
        let dimension = cfg.codebook_dim / 2;
        Ok(Self {
            n_q_semantic,
            rvq_first: ResidualVectorQuantizer::new(
                n_q_semantic,
                dimension,
                cfg.codebook_size,
                cfg.codebook_dim,
                vb.pp("rvq_first"),
            )?,
            rvq_rest: ResidualVectorQuantizer::new(
                cfg.num_quantizers - n_q_semantic,
                dimension,
                cfg.codebook_size,
                cfg.codebook_dim,
                vb.pp("rvq_rest"),
            )?,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (_, k, _) = codes.dims3()?;
        let mut first = self
            .rvq_first
            .decode(&codes.narrow(1, 0, self.n_q_semantic)?)?;
        if k > self.n_q_semantic {
            let mut rest = self
                .rvq_rest
                .decode(&codes.narrow(1, self.n_q_semantic, k - self.n_q_semantic)?)?;
            if rest.dtype() != first.dtype() {
                rest = rest.to_dtype(first.dtype())?;
            }
            Ok((first + rest)?)
        } else {
            if first.dtype() != codes.dtype() {
                first = first.to_dtype(codes.dtype())?;
            }
            Ok(first)
        }
    }
}

#[derive(Debug, Clone)]
enum DecoderTailLayer {
    CausalConv(CausalConvNet),
    DecoderBlock(DecoderBlock),
    Snake(SnakeBeta),
}

impl DecoderTailLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::CausalConv(m) => m.forward(x),
            Self::DecoderBlock(m) => m.forward(x),
            Self::Snake(m) => m.forward(x),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Mimi Encoder (audio → codec codes)
// ═══════════════════════════════════════════════════════════════════════

/// EnCodec-style residual unit: two causal convs with residual connection.
#[derive(Debug, Clone)]
struct EncoderResidualUnit {
    block: Vec<CausalConvNet>,
}

impl EncoderResidualUnit {
    fn new(channels: usize, kernel_size: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let mut block = Vec::new();
        block.push(CausalConvNet::new(channels, channels / 2, kernel_size, 1, dilation, 1, vb.pp("block").pp(1))?);
        block.push(CausalConvNet::new(channels / 2, channels, 1, 1, 1, 1, vb.pp("block").pp(3))?);
        Ok(Self { block })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for conv in &self.block {
            h = conv.forward(&h)?.elu(1.0)?;
        }
        Ok((x + h)?)
    }
}

/// EnCodec encoder block: residual units + downsampling conv.
#[derive(Debug, Clone)]
struct EncoderBlock {
    residuals: Vec<EncoderResidualUnit>,
    downsample: CausalConvNet,
}

impl EncoderBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        num_residual: usize,
        residual_kernel: usize,
        res_layer_idx: usize,
        down_layer_idx: usize,
        layers_vb: &VarBuilder,
    ) -> Result<Self> {
        let mut residuals = Vec::new();
        for i in 0..num_residual {
            let dilation = 2usize.pow(i as u32);
            residuals.push(EncoderResidualUnit::new(
                in_channels, residual_kernel, dilation,
                layers_vb.pp(res_layer_idx),
            )?);
        }
        let downsample = CausalConvNet::new(
            in_channels, out_channels, stride * 2, stride, 1, 1,
            layers_vb.pp(down_layer_idx),
        )?;
        Ok(Self { residuals, downsample })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for res in &self.residuals {
            h = res.forward(&h)?;
        }
        h = h.elu(1.0)?;
        self.downsample.forward(&h)
    }
}

/// Encoder transformer layer (same architecture as decoder, reused).
/// Uses LayerNorm (not RMSNorm) and GELU activation.
#[derive(Debug, Clone)]
struct EncoderTransformerLayer {
    input_ln: LayerNorm,
    post_attn_ln: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    fc1: Linear,
    fc2: Linear,
    attn_scale: Tensor,
    mlp_scale: Tensor,
    num_heads: usize,
    head_dim: usize,
    sliding_window: usize,
}

impl EncoderTransformerLayer {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let h = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        Ok(Self {
            input_ln: layer_norm(d, cfg.norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln: layer_norm(d, cfg.norm_eps, vb.pp("post_attention_layernorm"))?,
            q_proj: linear_no_bias(d, h * hd, vb.pp("self_attn").pp("q_proj"))?,
            k_proj: linear_no_bias(d, h * hd, vb.pp("self_attn").pp("k_proj"))?,
            v_proj: linear_no_bias(d, h * hd, vb.pp("self_attn").pp("v_proj"))?,
            o_proj: linear_no_bias(h * hd, d, vb.pp("self_attn").pp("o_proj"))?,
            fc1: linear_no_bias(d, cfg.intermediate_size, vb.pp("mlp").pp("fc1"))?,
            fc2: linear_no_bias(cfg.intermediate_size, d, vb.pp("mlp").pp("fc2"))?,
            attn_scale: vb.get(d, "self_attn_layer_scale.scale")?,
            mlp_scale: vb.get(d, "mlp_layer_scale.scale")?,
            num_heads: h,
            head_dim: hd,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryEmbedding) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let h = self.num_heads;
        let hd = self.head_dim;

        // Self-attention with sliding window
        let normed = x.apply(&self.input_ln)?;
        let q = normed.apply(&self.q_proj)?.reshape((b, t, h, hd))?.transpose(1, 2)?;
        let k = normed.apply(&self.k_proj)?.reshape((b, t, h, hd))?.transpose(1, 2)?;
        let v = normed.apply(&self.v_proj)?.reshape((b, t, h, hd))?.transpose(1, 2)?;

        // Apply RoPE using existing rotate_half approach
        let (cos, sin) = rotary.forward(t)?;
        // cos/sin: [t, hd/2] → expand to [1, 1, t, hd] by repeating
        let cos2 = Tensor::cat(&[&cos, &cos], 1)?.reshape((1, 1, t, hd))?;
        let sin2 = Tensor::cat(&[&sin, &sin], 1)?.reshape((1, 1, t, hd))?;
        let q = (q.broadcast_mul(&cos2)? + rotate_half(&q)?.broadcast_mul(&sin2)?)?;
        let k = (k.broadcast_mul(&cos2)? + rotate_half(&k)?.broadcast_mul(&sin2)?)?;

        // Sliding window causal attention
        let scale = (hd as f64).sqrt().recip();
        let attn_w = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let mask = causal_sliding_mask(t, self.sliding_window, x.device())?
            .to_dtype(x.dtype())?;
        let attn_w = attn_w.broadcast_add(&mask)?;
        let attn_w = candle_nn::ops::softmax_last_dim(&attn_w)?;

        let attn_out = attn_w.matmul(&v.contiguous()?)?
            .transpose(1, 2)?.contiguous()?.reshape((b, t, h * hd))?;
        let attn_out = attn_out.apply(&self.o_proj)?;
        let attn_out = attn_out.broadcast_mul(&self.attn_scale.reshape((1, 1, d))?)?;
        let x_post_attn = (x + &attn_out)?;
        let normed2 = x_post_attn.apply(&self.post_attn_ln)?;

        // MLP (GELU)
        let mlp_out = normed2.apply(&self.fc1)?.gelu()?.apply(&self.fc2)?;
        let mlp_out = mlp_out.broadcast_mul(&self.mlp_scale.reshape((1, 1, d))?)?;
        Ok((x_post_attn + mlp_out)?)
    }
}

/// EuclideanCodebook encode: nearest-neighbor lookup.
impl EuclideanCodebook {
    fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, dim] or [T, dim]
        let usage = self.cluster_usage.clamp(self.epsilon, f64::INFINITY)?
            .reshape((self.cluster_usage.dims1()?, 1))?;
        let embedding = self.embedding_sum.broadcast_div(&usage)?; // [codebook_size, dim]

        // Compute squared Euclidean distances: ||x - e||^2 = ||x||^2 - 2*x·e + ||e||^2
        let x_flat = if x.dims().len() == 3 {
            let (b, t, d) = x.dims3()?;
            x.reshape((b * t, d))?
        } else {
            x.clone()
        };
        let (n, _d) = x_flat.dims2()?;
        let x_sq = x_flat.sqr()?.sum(1)?; // [N]
        let e_sq = embedding.sqr()?.sum(1)?; // [C]
        let dot = x_flat.matmul(&embedding.t()?)?; // [N, C]
        // dist = x_sq[:, None] - 2*dot + e_sq[None, :]
        let dist = (x_sq.reshape((n, 1))?.broadcast_sub(&(dot * 2.0)?)? + e_sq.reshape((1, embedding.dims2()?.0))?)?;
        // argmin over codebook dim
        let codes = dist.argmin(1)?; // [N]

        if x.dims().len() == 3 {
            let (b, t, _) = x.dims3()?;
            Ok(codes.reshape((b, t))?)
        } else {
            Ok(codes)
        }
    }
}

impl VectorQuantization {
    fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, D, T] → transpose to [B, T, D] for codebook lookup
        let x_btd = x.transpose(1, 2)?;
        let codes = self.codebook.encode(&x_btd)?;
        Ok(codes)
    }
}

/// Encoder codebook: loads from `codebook.embed_sum` / `codebook.cluster_usage`.
struct EncEuclideanCodebook {
    embedding_sum: Tensor,
    cluster_usage: Tensor,
    epsilon: f64,
}

impl EncEuclideanCodebook {
    fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        let embedding_sum = vb.get((codebook_size, dim), "embed_sum")?;
        let cluster_usage = match vb.get(codebook_size, "cluster_usage") {
            Ok(v) => v,
            Err(_) => {
                let ones = vec![1f32; codebook_size];
                Tensor::new(ones.as_slice(), vb.device())?.to_dtype(embedding_sum.dtype())?
            }
        };
        Ok(Self { embedding_sum, cluster_usage, epsilon: 1e-5 })
    }

    fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let usage = self.cluster_usage.clamp(self.epsilon, f64::INFINITY)?
            .reshape((self.cluster_usage.dims1()?, 1))?;
        let embedding = self.embedding_sum.broadcast_div(&usage)?; // [C, D]

        let x_flat = if x.dims().len() == 3 {
            let (b, t, d) = x.dims3()?;
            x.reshape((b * t, d))?
        } else {
            x.clone()
        };
        let (n, _d) = x_flat.dims2()?;
        let c = embedding.dims2()?.0;
        let x_sq = x_flat.sqr()?.sum(1)?;  // [N]
        let e_sq = embedding.sqr()?.sum(1)?; // [C]
        let dot = x_flat.matmul(&embedding.t()?)?; // [N, C]
        // dist = x_sq[N,1] - 2*dot[N,C] + e_sq[1,C]
        let dist = x_sq.reshape((n, 1))?.broadcast_sub(&dot.affine(2.0, 0.0)?)?
            .broadcast_add(&e_sq.reshape((1, c))?)?;
        let codes = dist.argmin(1)?;

        if x.dims().len() == 3 {
            let (b, t, _) = x.dims3()?;
            Ok(codes.reshape((b, t))?)
        } else {
            Ok(codes)
        }
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let usage = self.cluster_usage.clamp(self.epsilon, f64::INFINITY)?
            .reshape((self.cluster_usage.dims1()?, 1))?;
        let embedding = self.embedding_sum.broadcast_div(&usage)?;
        let flat = codes.flatten_all()?;
        let q = embedding.embedding(&flat.contiguous()?)?;
        let (b, t) = codes.dims2()?;
        Ok(q.reshape((b, t, self.embedding_sum.dims2()?.1))?)
    }
}

struct EncVectorQuantization {
    codebook: EncEuclideanCodebook,
}

impl EncVectorQuantization {
    fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { codebook: EncEuclideanCodebook::new(dim, codebook_size, vb.pp("codebook"))? })
    }

    fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, D, T] → [B, T, D]
        self.codebook.encode(&x.transpose(1, 2)?)
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // codes: [B, T] → quantized [B, D, T]
        Ok(self.codebook.decode(codes)?.transpose(1, 2)?)
    }
}

/// Encoder-side SplitRVQ: encode audio latent → [B, T, n_q] codes.
struct EncoderSplitRVQ {
    semantic_input_proj: PointwiseProjNoBias,
    acoustic_input_proj: PointwiseProjNoBias,
    semantic_layers: Vec<EncVectorQuantization>,
    acoustic_layers: Vec<EncVectorQuantization>,
}

impl EncoderSplitRVQ {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let n_q_semantic = cfg.num_semantic_quantizers;
        let n_q_acoustic = cfg.num_quantizers - n_q_semantic;
        let dim = cfg.vector_quantization_hidden_dimension;
        let codebook_size = cfg.codebook_size;

        let sem_vb = vb.pp("semantic_residual_vector_quantizer");
        let aco_vb = vb.pp("acoustic_residual_vector_quantizer");

        let semantic_input_proj = PointwiseProjNoBias::new(cfg.hidden_size, dim, sem_vb.pp("input_proj"))?;
        let acoustic_input_proj = PointwiseProjNoBias::new(cfg.hidden_size, dim, aco_vb.pp("input_proj"))?;

        let mut semantic_layers = Vec::new();
        for i in 0..n_q_semantic {
            semantic_layers.push(EncVectorQuantization::new(dim, codebook_size, sem_vb.pp("layers").pp(i))?);
        }
        let mut acoustic_layers = Vec::new();
        for i in 0..n_q_acoustic {
            acoustic_layers.push(EncVectorQuantization::new(dim, codebook_size, aco_vb.pp("layers").pp(i))?);
        }

        Ok(Self { semantic_input_proj, acoustic_input_proj, semantic_layers, acoustic_layers })
    }

    /// Encode: x [B, D, T] → codes [B, T, n_q_total].
    fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let sem_x = self.semantic_input_proj.forward(x)?; // [B, dim, T]
        let aco_x = self.acoustic_input_proj.forward(x)?;

        let mut all_codes: Vec<Tensor> = Vec::new();
        let mut residual = sem_x.clone();
        for layer in &self.semantic_layers {
            let codes = layer.encode(&residual)?; // [B, T]
            let quantized = layer.decode(&codes)?; // [B, D, T]
            residual = (residual - quantized)?;
            all_codes.push(codes);
        }

        let mut residual = aco_x.clone();
        for layer in &self.acoustic_layers {
            let codes = layer.encode(&residual)?;
            let quantized = layer.decode(&codes)?;
            residual = (residual - quantized)?;
            all_codes.push(codes);
        }

        // Stack: [B, T, n_q]
        Ok(Tensor::stack(&all_codes, 2)?)
    }
}

/// Full Mimi encoder: audio [B, 1, N] → codes [B, T, n_q].
#[allow(dead_code)]
pub struct MimiEncoder {
    encoder_layers: Vec<EncoderBlock>,
    first_conv: CausalConvNet,
    last_conv: CausalConvNet,
    downsample: CausalConvNet,
    transformer_layers: Vec<EncoderTransformerLayer>,
    rotary: RotaryEmbedding,
    quantizer: EncoderSplitRVQ,
    valid_num_quantizers: usize,
    config: EncoderConfig,
}

impl MimiEncoder {
    pub fn new(cfg: &EncoderConfig, valid_n_q: usize, vb: VarBuilder) -> Result<Self> {
        // vb is already at the "encoder" prefix (caller does vb.pp("encoder"))
        let layers_vb = vb.pp("encoder").pp("layers");

        // Layer 0: first conv [1 → num_filters, k=kernel_size]
        let first_conv = CausalConvNet::new(1, cfg.num_filters, cfg.kernel_size, 1, 1, 1, layers_vb.pp(0))?;

        // Encoder blocks. The actual layer indices in the safetensors are non-sequential.
        // upsampling_ratios = [8, 6, 5, 4] are the DECODER ratios (largest first).
        // The ENCODER uses them reversed: [4, 5, 6, 8] (smallest stride first).
        // Actual kernel sizes from weights: 8, 10, 12, 16 = stride*2 for strides 4, 5, 6, 8.
        // Layer structure: 0=first_conv, 1=res(4), 3=down(4), 4=res(5), 6=down(5), 7=res(6), 9=down(6), 10=res(8), 12=down(8), 14=last_conv
        let ratios: Vec<usize> = cfg.upsampling_ratios.iter().rev().cloned().collect();
        let mut encoder_layers = Vec::new();
        let mut in_ch = cfg.num_filters;
        // residual layer indices: 1, 4, 7, 10
        // downsample layer indices: 3, 6, 9, 12
        let res_indices = [1usize, 4, 7, 10];
        let down_indices = [3usize, 6, 9, 12];
        for (block_i, &stride) in ratios.iter().enumerate() {
            let out_ch = in_ch * 2;
            let res_idx = res_indices[block_i];
            let down_idx = down_indices[block_i];
            encoder_layers.push(EncoderBlock::new(
                in_ch, out_ch, stride,
                cfg.num_residual_layers,
                cfg.residual_kernel_size,
                res_idx, down_idx,
                &layers_vb,
            )?);
            in_ch = out_ch;
        }

        // Layer 14: last conv [in_ch → hidden_size, k=last_kernel_size]
        let last_conv = CausalConvNet::new(in_ch, cfg.hidden_size, cfg.last_kernel_size, 1, 1, 1, layers_vb.pp(14))?;

        // Downsample: stride-4 conv, no bias (compress=2 → stride=4)
        // Weight: encoder.downsample.conv.weight [512, 512, 4] (no bias key)
        let downsample = CausalConvNet::new_no_bias(cfg.hidden_size, cfg.hidden_size, 4, 4, 1, 1, vb.pp("downsample"))?;

        // Encoder transformer
        let mut transformer_layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            transformer_layers.push(EncoderTransformerLayer::new(cfg, vb.pp("encoder_transformer").pp("layers").pp(i))?);
        }
        let rotary = RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, vb.device())?;

        // Quantizer
        let quantizer = EncoderSplitRVQ::new(cfg, vb.pp("quantizer"))?;

        Ok(Self {
            encoder_layers,
            first_conv,
            last_conv,
            downsample,
            transformer_layers,
            rotary,
            quantizer,
            valid_num_quantizers: valid_n_q,
            config: cfg.clone(),
        })
    }

    /// Encode audio [B, 1, N] → codes [B, T, valid_n_q].
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        // audio: [B, 1, N], CausalConvNet expects [B, C, T]
        // Ensure audio is F32 to match encoder weights (loaded in F32)
        let mut x = if audio.dtype() != DType::F32 {
            audio.to_dtype(DType::F32)?
        } else {
            audio.clone()
        };

        // First conv
        x = self.first_conv.forward(&x)?.elu(1.0)?;

        // Encoder blocks
        for block in &self.encoder_layers {
            x = block.forward(&x)?;
        }

        // Last conv
        x = self.last_conv.forward(&x)?;

        // Downsample
        x = self.downsample.forward(&x)?;

        // Transformer: [B, D, T] → [B, T, D]
        x = x.transpose(1, 2)?;
        for layer in &self.transformer_layers {
            x = layer.forward(&x, &self.rotary)?;
        }
        // Back to [B, D, T]
        x = x.transpose(1, 2)?;

        // Quantize: [B, D, T] → [B, T, n_q]
        let codes = self.quantizer.encode(&x)?;

        // Trim to valid_num_quantizers
        let n_q = codes.dim(2)?;
        let valid = self.valid_num_quantizers.min(n_q);
        Ok(codes.narrow(2, 0, valid)?)
    }
}

pub struct NativeSpeechTokenizerDecoder {
    config: DecoderConfig,
    sample_rate: u32,
    total_upsample: usize,
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConvNet,
    pre_transformer: TokenizerTransformer,
    upsample: Vec<(CausalTransConvNet, ConvNeXtBlock)>,
    decoder: Vec<DecoderTailLayer>,
    encoder: Option<MimiEncoder>,
    device: Device,
    dtype: DType,
}

impl NativeSpeechTokenizerDecoder {
    pub fn new(model_dir: &str, device: &Device, _dtype: DType) -> Result<Self> {
        // Always load speech tokenizer weights in F32 for numerical stability.
        // The decoder pipeline uses SnakeBeta (exp/sin/sqr), LayerNorm, and GELU
        // which require F32 precision to produce correct audio waveforms.
        let dtype = DType::F32;
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let cfg_data = std::fs::read(&config_path)?;
        let cfg: TokenizerV2Config = serde_json::from_slice(&cfg_data)?;
        let dcfg = cfg.decoder_config.clone();

        let filenames = crate::utils::utils::get_safetensors_files(model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device) }?;
        let decoder_vb = vb.pp("decoder");

        let quantizer = SplitResidualVectorQuantizer::new(&dcfg, decoder_vb.pp("quantizer"))?;
        let pre_conv = CausalConvNet::new(
            dcfg.codebook_dim,
            dcfg.latent_dim,
            3,
            1,
            1,
            1,
            decoder_vb.pp("pre_conv"),
        )?;
        let pre_transformer = TokenizerTransformer::new(&dcfg, decoder_vb.pp("pre_transformer"))?;

        let mut upsample = Vec::with_capacity(dcfg.upsampling_ratios.len());
        for (i, &factor) in dcfg.upsampling_ratios.iter().enumerate() {
            let up = CausalTransConvNet::new(
                dcfg.latent_dim,
                dcfg.latent_dim,
                factor,
                factor,
                decoder_vb.pp("upsample").pp(i).pp(0),
            )?;
            let block = ConvNeXtBlock::new(dcfg.latent_dim, decoder_vb.pp("upsample").pp(i).pp(1))?;
            upsample.push((up, block));
        }

        let mut decoder = Vec::new();
        decoder.push(DecoderTailLayer::CausalConv(CausalConvNet::new(
            dcfg.latent_dim,
            dcfg.decoder_dim,
            7,
            1,
            1,
            1,
            decoder_vb.pp("decoder").pp(0),
        )?));

        for i in 0..dcfg.upsample_rates.len() {
            decoder.push(DecoderTailLayer::DecoderBlock(DecoderBlock::new(
                &dcfg,
                i,
                decoder_vb.pp("decoder").pp(i + 1),
            )?));
        }

        let output_dim = dcfg.decoder_dim / (1usize << dcfg.upsample_rates.len());
        decoder.push(DecoderTailLayer::Snake(SnakeBeta::new(
            output_dim,
            decoder_vb.pp("decoder").pp(dcfg.upsample_rates.len() + 1),
        )?));
        decoder.push(DecoderTailLayer::CausalConv(CausalConvNet::new(
            output_dim,
            1,
            7,
            1,
            1,
            1,
            decoder_vb.pp("decoder").pp(dcfg.upsample_rates.len() + 2),
        )?));

        let encoder = match MimiEncoder::new(&cfg.encoder_config, cfg.encoder_valid_num_quantizers, vb.pp("encoder")) {
            Ok(enc) => Some(enc),
            Err(e) => {
                eprintln!("[speech_tokenizer] Warning: encoder not loaded: {e}");
                None
            }
        };

        Ok(Self {
            total_upsample: dcfg.total_upsample(),
            config: dcfg,
            sample_rate: cfg.output_sample_rate,
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            decoder,
            encoder,
            device: device.clone(),
            dtype,
        })
    }

    /// Encode audio `[B, 1, N]` → codes `[B, T, n_q]`.
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Speech tokenizer encoder not loaded"))?;
        enc.encode(audio)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn tts_debug() -> bool {
        std::env::var("CRANE_TTS_DEBUG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    }

    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let debug = Self::tts_debug();
        let (_, k, _) = codes.dims3()?;
        if k != self.config.num_quantizers {
            anyhow::bail!(
                "Expected {} code layers, got {}",
                self.config.num_quantizers,
                k
            );
        }
        if debug { eprintln!("[DECODER] input codes: {:?}", codes.dims()); }

        // Ensure codes are on the right device (I64 indices don't need dtype cast)
        let mut hidden = self.quantizer.decode(codes)?;
        // Force F32 for the entire decoder pipeline for numerical stability
        if hidden.dtype() != DType::F32 {
            hidden = hidden.to_dtype(DType::F32)?;
        }
        if debug { eprintln!("[DECODER] after quantizer.decode: {:?}", hidden.dims()); }

        hidden = self.pre_conv.forward(&hidden)?;
        if debug { eprintln!("[DECODER] after pre_conv: {:?}", hidden.dims()); }

        hidden = hidden.transpose(1, 2)?;
        hidden = self.pre_transformer.forward(&hidden)?;
        hidden = hidden.transpose(1, 2)?;
        if debug { eprintln!("[DECODER] after pre_transformer: {:?}", hidden.dims()); }

        for (i, (up, block)) in self.upsample.iter().enumerate() {
            hidden = up.forward(&hidden)?;
            if debug { eprintln!("[DECODER] after upsample[{i}].up: {:?}", hidden.dims()); }
            hidden = block.forward(&hidden)?;
            if debug { eprintln!("[DECODER] after upsample[{i}].block: {:?}", hidden.dims()); }
        }

        let mut wav = hidden;
        for (i, block) in self.decoder.iter().enumerate() {
            wav = block.forward(&wav)?;
            if debug { eprintln!("[DECODER] after decoder[{i}]: {:?}", wav.dims()); }
        }

        if debug { eprintln!("[DECODER] final output (before clamp): {:?}", wav.dims()); }
        Ok(wav.clamp(-1.0, 1.0)?)
    }

    pub fn chunked_decode(&self, codes: &Tensor, chunk_size: usize, left_context_size: usize) -> Result<Tensor> {
        let debug = Self::tts_debug();
        let (_, _, t) = codes.dims3()?;
        if debug {
            eprintln!("[DECODER] chunked_decode: codes={:?} chunk_size={chunk_size} total_upsample={}", codes.dims(), self.total_upsample);
        }
        let mut wavs = Vec::new();
        let mut start = 0usize;

        while start < t {
            let end = usize::min(start + chunk_size, t);
            let context = if start > left_context_size {
                left_context_size
            } else {
                start
            };
            let chunk = codes.narrow(D::Minus1, start - context, end - (start - context))?;
            let wav_chunk = self.forward(&chunk)?;
            let trim = context * self.total_upsample;
            let (_, _, tw) = wav_chunk.dims3()?;
            let wav_chunk = wav_chunk.narrow(D::Minus1, trim, tw.saturating_sub(trim))?;
            wavs.push(wav_chunk);
            start = end;
        }

        let refs: Vec<&Tensor> = wavs.iter().collect();
        Ok(Tensor::cat(&refs, D::Minus1)?)
    }
}
