//! DSpark speculative-decoding drafter (MiniCPM-V-4.7-1B-DSpark).
//!
//! DSpark ([arXiv 2607.05147](https://arxiv.org/abs/2607.05147)) extends
//! DFlash's block drafter. One forward of a small Qwen3-style stack proposes a
//! whole block in parallel; a low-rank *Markov head* then restores the
//! dependency between neighbouring draft tokens, and a *confidence head*
//! estimates how likely each position is to survive verification.
//!
//! * **Context injection.** Hidden states of the target at `target_layer_ids`
//!   are concatenated, projected by `fc` and RMS-normalised once
//!   (`hidden_norm`); every draft layer's K/V projections are applied to that
//!   shared context as well as to the block, so the block attends to both.
//!   Context K/V (post-RoPE) persists across rounds in [`DSparkDrafter`];
//!   block K/V is recomputed every round.
//! * **Block.** `[anchor, <mask> x (block_size-1)]` embedded with the
//!   *target's* embedding table; attention is fully bidirectional (no mask).
//!   `block_size` positions yield `block_size` draft tokens (the anchor
//!   position predicts the token after the anchor), logits come from the
//!   *target's* `lm_head`.
//! * **Markov head.** `logits_k = U_k + W2(W1[prev_k])` where `prev_k` is the
//!   token drafted at `k-1` (the anchor for `k = 0`), sampled left to right.
//! * **Confidence head.** `sigmoid(w . [h_k ; W1[prev_k]] + b)`.
//!
//! Tap layers follow DFlash's convention: id `l` is the output of 0-based
//! decoder layer `l` (`hidden_states[l + 1]`).

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod metal;
pub mod policy;
#[cfg(all(feature = "sycl", not(any(feature = "cuda", feature = "rocm"))))]
mod sycl;

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias, ops::rms_norm, rotary_emb::rope};
use serde::Deserialize;

/// Candidate tokens per position in [`DSparkDrafter::markov_draft`].
pub const MARKOV_CANDIDATES: usize = 256;

#[derive(Debug, Clone, Deserialize)]
struct DflashSection {
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
}

/// The subset of the drafter's `config.json` the forward pass needs.
#[derive(Debug, Clone, Deserialize)]
pub struct DSparkConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub block_size: usize,
    pub vocab_size: usize,
    pub markov_rank: usize,
    rope_parameters: RopeParameters,
    dflash_config: DflashSection,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
}

impl DSparkConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    /// Target decoder layers whose outputs feed the drafter's context.
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.dflash_config.target_layer_ids
    }

    pub fn mask_token_id(&self) -> u32 {
        self.dflash_config.mask_token_id
    }
}

/// llama.cpp `dflash` tensor name -> the HF name [`DSparkDrafter::new`] loads.
fn gguf_to_hf_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let (idx, field) = rest.split_once('.')?;
        let mapped = match field {
            "attn_norm.weight" => "input_layernorm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            _ => return None,
        };
        return Some(format!("layers.{idx}.{mapped}"));
    }
    Some(
        match name {
            "fc.weight" => "fc.weight",
            "enc.output_norm.weight" => "hidden_norm.weight",
            "output_norm.weight" => "norm.weight",
            "markov_w1.weight" => "markov_head.markov_w1.weight",
            "markov_w2.weight" => "markov_head.markov_w2.weight",
            "conf_proj.weight" => "confidence_head.proj.weight",
            "conf_proj.bias" => "confidence_head.proj.bias",
            _ => return None,
        }
        .to_string(),
    )
}

struct Layer {
    input_ln: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    post_ln: Tensor,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// Context keys (post-norm, post-RoPE) `[kv_heads, ctx_len, head_dim]`.
    ctx_k: Option<Tensor>,
    ctx_v: Option<Tensor>,
}

/// Per-position Markov candidate sets (see [`DSparkDrafter::markov_candidates`]).
pub struct MarkovCandidates {
    /// `[block, k]` token ids, best base logit first.
    pub ids: Tensor,
    /// `[block, k]` base logits at those ids (f32).
    pub logits: Tensor,
    /// `[block, k, rank]` the matching `W2` rows.
    pub w2: Tensor,
}

/// Output of one drafted block.
#[derive(Debug, Clone)]
pub struct Draft {
    /// `block_size` proposed tokens, in order after the anchor.
    pub tokens: Vec<u32>,
    /// Per-position acceptance probability from the confidence head.
    pub confidence: Vec<f32>,
}

pub struct DSparkDrafter {
    cfg: DSparkConfig,
    fc: Linear,
    hidden_norm: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    markov_w1: Tensor,
    /// `[vocab, rank]`; rows are gathered per candidate set.
    markov_w2: Tensor,
    conf_w: Tensor,
    conf_b: Tensor,
    inv_freq: Vec<f32>,
    device: Device,
    dtype: DType,
    ctx_len: usize,
}

impl DSparkDrafter {
    /// Load from a HF-format directory (`config.json` + `model.safetensors`).
    ///
    /// # Errors
    ///
    /// Returns an error if the config or weights are missing or malformed.
    pub fn from_dir(dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        let cfg_path = std::path::Path::new(dir).join("config.json");
        let cfg: DSparkConfig = serde_json::from_slice(
            &std::fs::read(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?,
        )
        .with_context(|| format!("parse {}", cfg_path.display()))?;
        let files = crate::utils::utils::get_safetensors_files(dir)?;
        // Cast on the CPU and upload from heap memory. The published
        // `model.safetensors` stores its tensors at 2-byte-aligned offsets, and
        // on SYCL (Level Zero, Arc B70) a host->device copy straight from a
        // misaligned mmap address issued behind any queued kernel loses the
        // device.
        let vb = unsafe {
            crate::utils::cast_var_builder::from_mmaped_safetensors_cpu_cast(&files, dtype, device)
        }?;
        Self::new(cfg, &vb, device, dtype)
    }

    /// Load from a llama.cpp `dflash`-architecture GGUF
    /// (`dspark-MiniCPM-V-4_7-1B-*.gguf`). Every tensor is dequantized to
    /// `dtype` once at load; the config comes from the `dflash.*` metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the file isn't a `dflash` GGUF or lacks a tensor.
    pub fn from_gguf(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        use crate::quantized::gguf_file::Gguf;
        use crate::quantized::gguf_metadata::GgufMetadata;

        let mmap = crate::quantized::gguf_file::mmap_gguf_file(path)
            .with_context(|| format!("mmap {path}"))?;
        let (ct, _) = crate::quantized::extended_gguf::read_content(mmap.as_ref())?;
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let mut gg = Gguf::new(ct, &mut cursor, device.clone(), dtype);
        let meta = gg.metadata().clone();
        let md = GgufMetadata::new(&meta);
        let arch = md.opt_string("general.architecture").unwrap_or_default();
        if arch != "dflash" {
            anyhow::bail!(
                "{path} is not a DSpark/DFlash drafter (general.architecture = {arch:?})"
            );
        }
        let vocab_size = gg
            .ct
            .tensor_infos
            .get("markov_w1.weight")
            .context("GGUF missing markov_w1.weight (not a DSpark drafter)")?
            .shape
            .dims()[0];
        let markov_rank = gg.ct.tensor_infos["markov_w1.weight"].shape.dims()[1];
        // llama.cpp counts layers as "number applied"; DFlash ids are the
        // 0-based layer index, i.e. one less.
        let target_layer_ids: Vec<usize> = md
            .usizes("dflash.target_layers")?
            .into_iter()
            .map(|l| l.saturating_sub(1))
            .collect();
        let cfg = DSparkConfig {
            hidden_size: md.usize("dflash.embedding_length")?,
            intermediate_size: md.usize("dflash.feed_forward_length")?,
            num_attention_heads: md.usize("dflash.attention.head_count")?,
            num_key_value_heads: md.usize("dflash.attention.head_count_kv")?,
            head_dim: md.usize("dflash.attention.key_length")?,
            num_hidden_layers: md.usize("dflash.block_count")?,
            rms_norm_eps: f64::from(md.f32("dflash.attention.layer_norm_rms_epsilon")?),
            block_size: md.usize("dflash.block_size")?,
            vocab_size,
            markov_rank,
            rope_parameters: RopeParameters {
                rope_theta: f64::from(md.f32("dflash.rope.freq_base")?),
            },
            dflash_config: DflashSection {
                target_layer_ids,
                mask_token_id: md.u32("tokenizer.ggml.mask_token_id")?,
            },
        };

        let mut tensors = std::collections::HashMap::new();
        let names: Vec<String> = gg.ct.tensor_infos.keys().cloned().collect();
        for name in names {
            let Some(hf) = gguf_to_hf_name(&name) else {
                continue;
            };
            let mut t = gg.dequant_tensor(&name)?;
            if hf == "confidence_head.proj.weight" {
                t = t.reshape((1, ()))?; // stored 1-D in the GGUF
            }
            tensors.insert(hf, t);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        Self::new(cfg, &vb, device, dtype)
    }

    /// Build from an already-opened `VarBuilder` (safetensors or a tensor map
    /// dequantized from GGUF).
    ///
    /// # Errors
    ///
    /// Returns an error if a weight is missing or has an unexpected shape.
    pub fn new(cfg: DSparkConfig, vb: &VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = vb.pp("layers").pp(i);
            let a = p.pp("self_attn");
            let m = p.pp("mlp");
            layers.push(Layer {
                input_ln: p.get(h, "input_layernorm.weight")?,
                q_proj: linear_no_bias(h, nh * hd, a.pp("q_proj"))?,
                k_proj: linear_no_bias(h, nkv * hd, a.pp("k_proj"))?,
                v_proj: linear_no_bias(h, nkv * hd, a.pp("v_proj"))?,
                o_proj: linear_no_bias(nh * hd, h, a.pp("o_proj"))?,
                q_norm: a.get(hd, "q_norm.weight")?,
                k_norm: a.get(hd, "k_norm.weight")?,
                post_ln: p.get(h, "post_attention_layernorm.weight")?,
                gate_proj: linear_no_bias(h, cfg.intermediate_size, m.pp("gate_proj"))?,
                up_proj: linear_no_bias(h, cfg.intermediate_size, m.pp("up_proj"))?,
                down_proj: linear_no_bias(cfg.intermediate_size, h, m.pp("down_proj"))?,
                ctx_k: None,
                ctx_v: None,
            });
        }
        let taps = cfg.target_layer_ids().len();
        let r = cfg.markov_rank;
        let markov = vb.pp("markov_head");
        let markov_w1 = markov.get((cfg.vocab_size, r), "markov_w1.weight")?;
        let markov_w2 = markov.get((cfg.vocab_size, r), "markov_w2.weight")?;
        let conf = vb.pp("confidence_head").pp("proj");
        let inv_freq = (0..hd / 2)
            .map(|i| (cfg.rope_theta().powf(-(2.0 * i as f64) / hd as f64)) as f32)
            .collect();
        Ok(Self {
            fc: linear_no_bias(taps * h, h, vb.pp("fc"))?,
            hidden_norm: vb.get(h, "hidden_norm.weight")?,
            norm: vb.get(h, "norm.weight")?,
            markov_w1,
            markov_w2,
            conf_w: conf.get((1, h + r), "weight")?,
            conf_b: conf.get(1, "bias")?,
            layers,
            inv_freq,
            device: device.clone(),
            dtype,
            ctx_len: 0,
            cfg,
        })
    }

    pub fn config(&self) -> &DSparkConfig {
        &self.cfg
    }

    /// Number of context positions currently cached.
    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    /// Forget all context (new request).
    pub fn reset(&mut self) {
        self.ctx_len = 0;
        for l in &mut self.layers {
            l.ctx_k = None;
            l.ctx_v = None;
        }
    }

    /// `cos`/`sin` tables `[len, head_dim/2]` for positions `start..start+len`.
    fn rope_tables(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let pos = Tensor::arange(start as u32, (start + len) as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((len, 1))?;
        let inv = Tensor::from_slice(&self.inv_freq, (1, self.inv_freq.len()), &self.device)?;
        let freqs = pos.broadcast_mul(&inv)?;
        Ok((
            freqs.cos()?.to_dtype(self.dtype)?,
            freqs.sin()?.to_dtype(self.dtype)?,
        ))
    }

    fn norm(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        Ok(rms_norm(&x.contiguous()?, w, self.cfg.rms_norm_eps as f32)?)
    }

    /// Append target features for the next `n` token positions to the cached
    /// context. `taps`: `[n, target_layer_ids.len() * hidden]`, positions
    /// `ctx_len .. ctx_len + n`.
    ///
    /// # Errors
    ///
    /// Returns an error if a projection or concat fails.
    pub fn append_context(&mut self, taps: &Tensor) -> Result<()> {
        let n = taps.dim(0)?;
        if n == 0 {
            return Ok(());
        }
        let (nkv, hd) = (self.cfg.num_key_value_heads, self.cfg.head_dim);
        let ctx = self.fc.forward(&taps.to_dtype(self.dtype)?)?;
        let ctx = self.norm(&ctx, &self.hidden_norm)?;
        let (cos, sin) = self.rope_tables(self.ctx_len, n)?;
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let k = layer.k_proj.forward(&ctx)?.reshape((n, nkv, hd))?;
            let k = self.norm(&k, &layer.k_norm)?;
            let k =
                rope(&k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?.squeeze(0)?;
            let v = layer
                .v_proj
                .forward(&ctx)?
                .reshape((n, nkv, hd))?
                .transpose(0, 1)?;
            let layer = &mut self.layers[i];
            layer.ctx_k = Some(match layer.ctx_k.take() {
                Some(old) => Tensor::cat(&[&old, &k], 1)?.contiguous()?,
                None => k,
            });
            layer.ctx_v = Some(match layer.ctx_v.take() {
                Some(old) => Tensor::cat(&[&old, &v], 1)?.contiguous()?,
                None => v.contiguous()?,
            });
        }
        self.ctx_len += n;
        Ok(())
    }

    /// Run the block through the stack. `block_embeds`: `[block, hidden]`,
    /// the target-embedded `[anchor, mask...]`; the anchor sits at position
    /// `ctx_len`. Returns the final-normed hidden states `[block, hidden]`.
    ///
    /// # Errors
    ///
    /// Returns an error if any layer fails.
    pub fn block_hidden(&self, block_embeds: &Tensor) -> Result<Tensor> {
        let (blk, _) = block_embeds.dims2()?;
        let (nh, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let groups = nh / nkv;
        let (cos, sin) = self.rope_tables(self.ctx_len, blk)?;
        let scale = (hd as f64).powf(-0.5);
        let mut x = block_embeds.to_dtype(self.dtype)?;
        for layer in &self.layers {
            let h = self.norm(&x, &layer.input_ln)?;
            let q = layer.q_proj.forward(&h)?.reshape((blk, nh, hd))?;
            let q = self.norm(&q, &layer.q_norm)?;
            let q =
                rope(&q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?.squeeze(0)?;
            let k = layer.k_proj.forward(&h)?.reshape((blk, nkv, hd))?;
            let k = self.norm(&k, &layer.k_norm)?;
            let k =
                rope(&k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?.squeeze(0)?;
            let v = layer
                .v_proj
                .forward(&h)?
                .reshape((blk, nkv, hd))?
                .transpose(0, 1)?;
            let (k, v) = match (&layer.ctx_k, &layer.ctx_v) {
                (Some(ck), Some(cv)) => (
                    Tensor::cat(&[ck, &k], 1)?.contiguous()?,
                    Tensor::cat(&[cv, &v.contiguous()?], 1)?.contiguous()?,
                ),
                _ => (k, v.contiguous()?),
            };
            // Group query heads under their KV head: head h uses kv h / groups.
            let q = q.reshape((nkv, groups * blk, hd))?;
            let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                .to_dtype(self.dtype)?;
            let attn = probs.matmul(&v)?.reshape((nh, blk, hd))?;
            let attn = attn.transpose(0, 1)?.reshape((blk, nh * hd))?;
            x = (x + layer.o_proj.forward(&attn)?)?;
            let h = self.norm(&x, &layer.post_ln)?;
            let mlp = (candle_nn::ops::silu(&layer.gate_proj.forward(&h)?)?
                * layer.up_proj.forward(&h)?)?;
            x = (x + layer.down_proj.forward(&mlp)?)?;
        }
        self.norm(&x, &self.norm)
    }

    /// Markov-conditioned greedy drafting. `base_logits`: `[block, vocab]` f32
    /// from the target's `lm_head` over [`Self::block_hidden`]; `hidden`: the
    /// same call's output (for the confidence head).
    ///
    /// Greedy decoding is sequential (`prev_k` is the token chosen at `k-1`).
    /// Doing it over the full vocabulary reads all of `W2` (127 MB) per step,
    /// so each position is instead restricted to its [`MARKOV_CANDIDATES`]
    /// highest base logits: the bias is a gather of a few hundred `W2` rows
    /// and the sequential loop touches almost no memory. 256 candidates
    /// reproduce the full-vocabulary result on the reference prompts (64
    /// already costs ~9% acceptance, 16 ~16%). A different draft can only
    /// change the acceptance rate, never the output: verification decides.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor op or the device->host read fails.
    pub fn markov_draft(
        &self,
        base_logits: &Tensor,
        hidden: &Tensor,
        anchor: u32,
    ) -> Result<Draft> {
        let cands = self.markov_candidates(base_logits)?;
        let (tokens, prevs) = self.markov_chain(&cands, anchor)?;
        self.finish_draft(hidden, &tokens, &prevs)
    }

    /// Per-position top-[`MARKOV_CANDIDATES`] base-logit candidates and their
    /// `W2` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor op fails.
    pub fn markov_candidates(&self, base_logits: &Tensor) -> Result<MarkovCandidates> {
        let (blk, vocab) = base_logits.dims2()?;
        let k = MARKOV_CANDIDATES.min(vocab);
        let r = self.cfg.markov_rank;
        let ids = crate::ops::fused_ops::topk_indices_rows(&base_logits.contiguous()?, k)?; // [blk, k] u32
        let logits = base_logits.gather(&ids, 1)?; // [blk, k] f32
        let w2 = self
            .markov_w2
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((blk, k, r))?;
        Ok(MarkovCandidates { ids, logits, w2 })
    }

    /// The sequential greedy chain over the candidates. Returns the drafted
    /// tokens and, per position, the previous token each was conditioned on.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor op fails.
    pub fn markov_chain(&self, c: &MarkovCandidates, anchor: u32) -> Result<(Vec<u32>, Tensor)> {
        // One launch on CUDA / Metal / SYCL instead of ~10 per position (see
        // `kernels/{cuda,metal,sycl}/dspark.*`).
        #[cfg(feature = "cuda")]
        if self.device.is_cuda()
            && let Some((tokens, prevs)) =
                cuda::markov_chain(&c.ids, &c.logits, &c.w2, &self.markov_w1, anchor)?
        {
            return Ok((tokens.to_vec1::<u32>()?, prevs));
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if self.device.is_metal()
            && let Some((tokens, prevs)) =
                metal::markov_chain(&c.ids, &c.logits, &c.w2, &self.markov_w1, anchor)?
        {
            return Ok((tokens.to_vec1::<u32>()?, prevs));
        }
        #[cfg(all(feature = "sycl", not(any(feature = "cuda", feature = "rocm"))))]
        if self.device.is_sycl()
            && let Some((tokens, prevs)) =
                sycl::markov_chain(&c.ids, &c.logits, &c.w2, &self.markov_w1, anchor)?
        {
            return Ok((tokens.to_vec1::<u32>()?, prevs));
        }
        markov_chain_ops(c, &self.markov_w1, anchor)
    }

    /// Confidence head over the final tokens.
    fn finish_draft(&self, hidden: &Tensor, tokens: &[u32], prevs: &Tensor) -> Result<Draft> {
        let e = self.markov_w1.index_select(prevs, 0)?;
        let feat = Tensor::cat(&[hidden.to_dtype(self.dtype)?, e], 1)?;
        let conf = candle_nn::ops::sigmoid(
            &(feat
                .matmul(&self.conf_w.t()?)?
                .broadcast_add(&self.conf_b)?)
            .to_dtype(DType::F32)?,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
        Ok(Draft {
            tokens: tokens.to_vec(),
            confidence: conf,
        })
    }
}

/// [`DSparkDrafter::markov_chain`] as a chain of tensor ops (~10 per
/// position): the path for devices without the fused kernel, and the reference
/// the kernels are tested against.
fn markov_chain_ops(c: &MarkovCandidates, w1: &Tensor, anchor: u32) -> Result<(Vec<u32>, Tensor)> {
    let blk = c.ids.dim(0)?;
    let mut prev = Tensor::from_vec(vec![anchor], 1, w1.device())?;
    let mut prevs = Vec::with_capacity(blk);
    let mut toks = Vec::with_capacity(blk);
    for i in 0..blk {
        let e = w1.index_select(&prev, 0)?; // [1, r]
        let bias = e.matmul(&c.w2.narrow(0, i, 1)?.squeeze(0)?.t()?)?; // [1, k]
        // Reduce along the last dim: candle's Metal argmax over dim 0 of a
        // `[k, 1]` tensor returns 0 whatever the values.
        let scores = (c.logits.narrow(0, i, 1)? + bias.to_dtype(DType::F32)?)?;
        let pick = scores.argmax(D::Minus1)?.reshape(1)?;
        prevs.push(prev);
        prev = c
            .ids
            .narrow(0, i, 1)?
            .flatten_all()?
            .index_select(&pick, 0)?;
        toks.push(prev.clone());
    }
    Ok((
        Tensor::cat(&toks, 0)?.to_vec1::<u32>()?,
        Tensor::cat(&prevs, 0)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::gguf_to_hf_name;

    /// A fused Markov-chain kernel against the op chain (and the op chain
    /// against the CPU), on random candidates with a W1/W2 bias large enough
    /// to reorder them.
    #[cfg(any(
        all(feature = "metal", not(feature = "cuda")),
        all(feature = "sycl", not(any(feature = "cuda", feature = "rocm")))
    ))]
    fn check_markov_kernel(
        dev: &candle_core::Device,
        kernel: impl Fn(
            &candle_core::Tensor,
            &candle_core::Tensor,
            &candle_core::Tensor,
            &candle_core::Tensor,
            u32,
        )
            -> candle_core::Result<Option<(candle_core::Tensor, candle_core::Tensor)>>,
    ) {
        use super::{MarkovCandidates, markov_chain_ops};
        use candle_core::{DType, Device, Tensor};

        let (blk, k, r, vocab) = (15usize, 256usize, 256usize, 4096usize);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            // Candidate ids: distinct per row, all inside the vocabulary.
            let ids: Vec<u32> = (0..blk * k).map(|i| ((i * 7919) % vocab) as u32).collect();
            let ids = Tensor::from_vec(ids, (blk, k), dev).unwrap();
            let logits = Tensor::randn(0f32, 1.0, (blk, k), dev).unwrap();
            let w1 = Tensor::randn(0f32, 0.3, (vocab, r), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let w2 = Tensor::randn(0f32, 0.3, (blk, k, r), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let c = MarkovCandidates { ids, logits, w2 };
            let (want_tok, want_prev) = markov_chain_ops(&c, &w1, 17).unwrap();
            // The CPU has no BF16 matmul.
            if dtype != DType::BF16 {
                let cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap();
                let on_cpu = MarkovCandidates {
                    ids: cpu(&c.ids),
                    logits: cpu(&c.logits),
                    w2: cpu(&c.w2),
                };
                let (cpu_tok, _) = markov_chain_ops(&on_cpu, &cpu(&w1), 17).unwrap();
                assert_eq!(want_tok, cpu_tok, "{dtype:?} op chain: GPU vs CPU");
            }
            let (got_tok, got_prev) = kernel(&c.ids, &c.logits, &c.w2, &w1, 17)
                .unwrap()
                .expect("shapes fit the kernel");
            assert_eq!(
                got_tok.to_vec1::<u32>().unwrap(),
                want_tok,
                "{dtype:?} tokens"
            );
            assert_eq!(
                got_prev.to_vec1::<u32>().unwrap(),
                want_prev.to_vec1::<u32>().unwrap(),
                "{dtype:?} prevs"
            );
        }
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    #[test]
    fn metal_markov_chain_matches_op_chain() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let dev = candle_core::Device::new_metal(0).unwrap();
        check_markov_kernel(&dev, super::metal::markov_chain);
    }

    #[cfg(all(feature = "sycl", not(any(feature = "cuda", feature = "rocm"))))]
    #[test]
    fn sycl_markov_chain_matches_op_chain() {
        let Ok(dev) = candle_core::Device::new_sycl(0) else {
            return;
        };
        check_markov_kernel(&dev, super::sycl::markov_chain);
    }

    #[test]
    fn maps_gguf_tensor_names() {
        assert_eq!(
            gguf_to_hf_name("blk.3.attn_output.weight").unwrap(),
            "layers.3.self_attn.o_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_norm.weight").unwrap(),
            "layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("enc.output_norm.weight").unwrap(),
            "hidden_norm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("markov_w2.weight").unwrap(),
            "markov_head.markov_w2.weight"
        );
        assert!(gguf_to_hf_name("something.else").is_none());
    }
}
