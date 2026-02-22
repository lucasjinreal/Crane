use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    conv1d, conv_transpose1d, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, RmsNorm, VarBuilder,
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
pub struct TokenizerV2Config {
    pub decoder_config: DecoderConfig,
    #[serde(default = "default_output_sample_rate")]
    pub output_sample_rate: u32,
}

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
        let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
        let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;

        let rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, rep)?;
        let v = repeat_kv(&v, rep)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let mask = causal_sliding_mask(t, self.sliding_window, hidden.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, c))?
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
        let conv = conv1d(
            in_channels,
            out_channels,
            kernel_size,
            Conv1dConfig {
                stride,
                dilation,
                groups,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        let effective = (kernel_size - 1) * dilation + 1;
        Ok(Self {
            conv,
            stride,
            kernel_size: effective,
            padding: effective.saturating_sub(stride),
        })
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
        Ok((hidden + gain.broadcast_mul(&periodic)?)?)
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
        let hidden = hidden.apply(&self.pwconv1)?.gelu()?;
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
        let (b, k, t) = codes.dims3()?;
        if k > self.layers.len() {
            anyhow::bail!("Too many quantizers: got {k}, model has {}", self.layers.len());
        }
        let mut sum = Tensor::zeros((b, self.dim, t), DType::F32, codes.device())?;
        for idx in 0..k {
            let layer_codes = codes.i((.., idx, ..))?;
            let q = self.layers[idx].decode(&layer_codes)?;
            sum = (sum + q)?;
        }
        Ok(sum)
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
        let y = x.matmul(&self.weight_t)?; // [B, T, O]
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
        let first = self
            .rvq_first
            .decode(&codes.narrow(1, 0, self.n_q_semantic)?)?;
        if k > self.n_q_semantic {
            let rest = self
                .rvq_rest
                .decode(&codes.narrow(1, self.n_q_semantic, k - self.n_q_semantic)?)?;
            Ok((first + rest)?)
        } else {
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

#[derive(Debug, Clone)]
pub struct NativeSpeechTokenizerDecoder {
    config: DecoderConfig,
    sample_rate: u32,
    total_upsample: usize,
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConvNet,
    pre_transformer: TokenizerTransformer,
    upsample: Vec<(CausalTransConvNet, ConvNeXtBlock)>,
    decoder: Vec<DecoderTailLayer>,
}

impl NativeSpeechTokenizerDecoder {
    pub fn new(model_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
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

        Ok(Self {
            total_upsample: dcfg.total_upsample(),
            config: dcfg,
            sample_rate: cfg.output_sample_rate,
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            decoder,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let (_, k, _) = codes.dims3()?;
        if k != self.config.num_quantizers {
            anyhow::bail!(
                "Expected {} code layers, got {}",
                self.config.num_quantizers,
                k
            );
        }

        let mut hidden = self.quantizer.decode(codes)?;
        hidden = self.pre_conv.forward(&hidden)?;

        hidden = hidden.transpose(1, 2)?;
        hidden = self.pre_transformer.forward(&hidden)?;
        hidden = hidden.transpose(1, 2)?;

        for (up, block) in self.upsample.iter() {
            hidden = up.forward(&hidden)?;
            hidden = block.forward(&hidden)?;
        }

        let mut wav = hidden;
        for block in self.decoder.iter() {
            wav = block.forward(&wav)?;
        }

        Ok(wav.clamp(-1.0, 1.0)?)
    }

    pub fn chunked_decode(&self, codes: &Tensor, chunk_size: usize, left_context_size: usize) -> Result<Tensor> {
        let (_, _, t) = codes.dims3()?;
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
