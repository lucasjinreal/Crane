// implement Siglip2 model
// to support Namo-500M-v2
// Note that this Siglip2 doesn't contains original text part.
// Just vision part.

use crate::utils::utils;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv2d_no_bias, embedding, func, layer_norm, linear, Activation, Conv2d, Conv2dConfig,
    Embedding, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Siglip2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f64,
    pub attention_dropout: f32,
    pub hidden_act: Activation,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub projection_dim: usize,
    pub num_patches: usize,
    pub vision_use_head: bool,
}

#[derive(Clone, Debug)]
pub struct Siglip2MLP {
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl Siglip2MLP {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(config.hidden_size, config.intermediate_size, vb.pp("fc2"))?;

        Ok(Self {
            fc1,
            fc2,
            activation: config.hidden_act.clone(),
        })
    }
}

impl Module for Siglip2MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.activation.forward(&xs)?;
        self.fc2.forward(&xs)
    }
}

#[derive(Clone, Debug)]
pub struct Siglip2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    dropout: f32,
}

impl Siglip2Attention {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;

        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            dropout: config.attention_dropout,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_size, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_size, q_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b_size, q_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b_size, q_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        let mut attn_weights = (q.matmul(&k.t()?)? * self.scale)?;

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
            b_size,
            q_len,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Clone, Debug)]
pub struct Siglip2EncoderLayer {
    self_attn: Siglip2Attention,
    layer_norm1: LayerNorm,
    mlp: Siglip2MLP,
    layer_norm2: LayerNorm,
}

impl Siglip2EncoderLayer {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Siglip2Attention::new(config, vb.pp("self_attn"))?;

        let layer_norm1 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm1"),
        )?;
        let mlp = Siglip2MLP::new(config, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm2"),
        )?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

#[derive(Debug)]
pub struct Siglip2Encoder {
    layers: Vec<Siglip2EncoderLayer>,
    gradient_checkpointing: bool,
}

impl Siglip2Encoder {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Siglip2EncoderLayer::new(config, vb.pp(&format!("layers.{i}")))?;
            layers.push(layer);
        }
        Ok(Self {
            layers,
            gradient_checkpointing: false,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = xs.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

#[derive(Debug)]
pub struct Siglip2VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    position_embedding_size: usize,
    num_patches: usize,
    patch_size: usize,
}

impl Siglip2VisionEmbeddings {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let conv2dconfig = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };

        let patch_embedding = conv2d_no_bias(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            conv2dconfig,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = embedding(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
            vb.pp("patch_embedding"),
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            position_embedding_size: (config.image_size / config.patch_size) as usize,
            num_patches: config.num_patches,
            patch_size: config.patch_size,
        })
    }

    fn resize_positional_embeddings(
        &self,
        positional_embeddings: &Tensor,
        spatial_shapes: &Tensor,
        max_length: usize,
    ) -> Result<Tensor> {
        let (b_size, _) = spatial_shapes.dims2()?;
        let embed_dim = positional_embeddings.dim(D::Minus1)?;
        let device = positional_embeddings.device();

        // 转换为图像格式 (H, W, C) -> (C, H, W)
        let embeddings = positional_embeddings.permute((2, 0, 1))?.unsqueeze(0)?;

        let mut resized = Vec::with_capacity(b_size);
        for i in 0..b_size {
            // let (h, w) = spatial_shapes.i((i, 0..2))?.to_vec2::<usize>()?;
            let row_tensor = spatial_shapes.narrow(0, i, 1)?; // 提取第i行
            let hw_tensor = row_tensor.narrow(1, 0, 2)?; // 提取前两列
            let hw_vec = hw_tensor.to_vec1::<i64>()?; // 转换为i64类型
            let (h, w) = (hw_vec[0] as usize, hw_vec[1] as usize);

            // 双线性插值
            // candle not support bilinear?
            let resized_emb = embeddings.upsample_nearest2d(h, w)?;

            // 转换回序列格式 (C, H, W) -> (H*W, C)
            let seq_emb = resized_emb
                .squeeze(0)?
                .permute((1, 2, 0))?
                .reshape((h * w, embed_dim))?;
            resized.push(seq_emb);
        }

        // 填充到最大长度
        let mut padded = Vec::with_capacity(b_size);
        for emb in resized {
            let len = emb.dim(0)?;
            if len < max_length {
                let pad = Tensor::zeros((max_length - len, embed_dim), emb.dtype(), device)?;
                padded.push(Tensor::cat(&[emb, pad], 0)?);
            } else {
                padded.push(emb);
            }
        }

        Tensor::stack(&padded, 0)
    }

    fn forward(&self, pixel_values: &Tensor, spatial_shapes: &Tensor) -> Result<Tensor> {
        // 投影到嵌入空间
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

        // 获取原始位置编码
        let positions = Tensor::arange(0u32, self.num_patches as u32, pixel_values.device())?;
        let mut pos_embeddings = self.position_embedding.forward(&positions)?;

        // 调整形状用于插值 (num_patches, dim) -> (H, W, dim)
        pos_embeddings = pos_embeddings.reshape((
            self.position_embedding_size,
            self.position_embedding_size,
            self.position_embedding.hidden_size(),
        ))?;

        // 调整位置编码尺寸
        let resized_pos = self.resize_positional_embeddings(
            &pos_embeddings,
            spatial_shapes,
            pixel_values.dim(1)?,
        )?;

        // 合并嵌入
        patch_embeds.add(&resized_pos)
    }
}

// 视觉Transformer模块
#[derive(Debug)]
pub struct Siglip2VisionTransformer {
    embeddings: Siglip2VisionEmbeddings,
    encoder: Siglip2Encoder,
    post_layernorm: LayerNorm,
    use_head: bool,
    head: Option<Siglip2MultiheadAttentionPoolingHead>,
}

impl Siglip2VisionTransformer {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let embeddings = Siglip2VisionEmbeddings::new(config, vb.pp("embeddings"))?;
        let encoder = Siglip2Encoder::new(config, vb.pp("encoder"))?;

        let post_layernorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_layernorm"),
        )?;

        let head = if config.vision_use_head {
            Some(Siglip2MultiheadAttentionPoolingHead::new(
                config,
                vb.pp("head"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            use_head: config.vision_use_head,
            head,
        })
    }

    fn forward(
        &self,
        pixel_values: &Tensor,
        attention_mask: Option<&Tensor>,
        spatial_shapes: &Tensor,
    ) -> Result<Tensor> {
        // 嵌入处理
        let mut hidden_states = self.embeddings.forward(pixel_values, spatial_shapes)?;

        // 编码器处理
        hidden_states = self.encoder.forward(&hidden_states, attention_mask)?;

        // 后层归一化
        hidden_states = self.post_layernorm.forward(&hidden_states)?;

        // 池化头
        if let Some(head) = &self.head {
            head.forward(&hidden_states, attention_mask)
        } else {
            Ok(hidden_states)
        }
    }
}

// 注意力池化头（示例实现）
#[derive(Debug)]
pub struct Siglip2MultiheadAttentionPoolingHead {
    attn: Siglip2Attention,
    output_proj: Linear,
}

impl Siglip2MultiheadAttentionPoolingHead {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let attn = Siglip2Attention::new(config, vb.pp("attn"))?;
        let output_proj = linear(
            config.hidden_size,
            config.projection_dim,
            vb.pp("output_proj"),
        )?;
        Ok(Self { attn, output_proj })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let pooled = self.attn.forward(xs, mask)?;
        self.output_proj.forward(&pooled)
    }
}

#[derive(Debug)]
pub struct Siglip2VisionModel {
    vision_model: Siglip2VisionTransformer,
}

impl Siglip2VisionModel {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let vision_model = Siglip2VisionTransformer::new(config, vb.pp("vision_model"))?;
        Ok(Self { vision_model })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_mask: &Tensor,
        spatial_shapes: &Tensor,
    ) -> Result<Tensor> {
        self.vision_model
            .forward(pixel_values, Some(pixel_attention_mask), spatial_shapes)
    }

    pub fn from_pretrained(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let config: Siglip2Config =
            serde_json::from_slice(&config_data).expect("siglip2 config parse error.");

        let filenames = utils::get_safetensors_files(model_path).unwrap();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
        let vision_model = Siglip2VisionTransformer::new(&config, vb.pp("vision_model"))?;

        Ok(Self { vision_model })
    }
}

// Siglip2 Processor?


pub fn test_main() {
    // let device = Device::Cpu;
    // let vb = VarBuilder::from_pretrained("model.safetensors", DType::F32, &device)?;

    // let config = Siglip2Config {
    //     hidden_size: 768,
    //     num_attention_heads: 12,
    //     intermediate_size: 3072,
    //     num_hidden_layers: 12,
    //     layer_norm_eps: 1e-5,
    //     attention_dropout: 0.1,
    //     hidden_act: Activation::Gelu,
    //     image_size: 224,
    //     patch_size: 16,
    //     num_channels: 3,
    //     projection_dim: 512,
    //     vision_use_head: true,
    //     num_patches: 1024,
    // };

    // let model = Siglip2VisionModel::new(&config, vb.pp("vision_model"))?;

    // // 示例输入
    // let pixel_values = Tensor::randn(0f32, 1.0, (2, 196, 768), &device)?; // [batch, seq, dim]
    // let attention_mask = Tensor::ones((2, 196), DType::U8, &device)?;
    // let spatial_shapes = Tensor::new(&[[14i64, 14], [16, 12]], &device)?; // 示例空间形状

    // let output = model.forward(&pixel_values, &attention_mask, &spatial_shapes)?;
}
