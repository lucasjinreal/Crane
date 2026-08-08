//! Flow-matching sampler: Euler ODE integration with sway-sampling timestep
//! warping and "CFG-Zero-star" classifier-free guidance. Inference-only port
//! of `locdit/unified_cfm.py`'s `UnifiedCFM` (`forward`/`solve_euler`/
//! `optimized_scale`) — no `compute_loss`, this crate never trains.
//!
//! The single most algorithmically distinct piece of VoxCPM2 relative to
//! the rest of Crane: `crate::models::voxtral_tts::modeling::flow_match_inference`
//! is a real precedent for the overall shape (batched cond/uncond forward
//! through the same estimator, `x = x - dt*v`) but has neither sway sampling
//! nor the CFG-Zero-star optimal-scale step, so its exact math doesn't
//! transfer — this is a fresh port of the VoxCPM2-specific algorithm.

use candle_core::{DType, Device, Result, Tensor};

use super::local_dit::VoxCpmLocDit;

pub struct UnifiedCfm {
    estimator: VoxCpmLocDit,
    in_channels: usize,
    /// `dit_config.mean_mode` — always `false` for the current VoxCPM2
    /// checkpoint, which zeroes `dt` unconditionally; implemented for
    /// fidelity with configs that do set it.
    mean_mode: bool,
}

impl UnifiedCfm {
    pub fn new(estimator: VoxCpmLocDit, mean_mode: bool) -> Self {
        let in_channels = estimator.in_channels();
        Self { estimator, in_channels, mean_mode }
    }

    /// `mu`: `[B, 2*hidden_size]` (LM context). `cond`: `[B, in_channels, T']`
    /// (previous patch). Returns `[B, in_channels, patch_size]` — the
    /// sampled next patch.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        mu: &Tensor,
        n_timesteps: usize,
        patch_size: usize,
        cond: &Tensor,
        cfg_value: f64,
        temperature: f64,
        sway_sampling_coef: f64,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        let (b, _) = mu.dims2()?;
        let device = mu.device();
        let dtype = mu.dtype();

        let z = Tensor::randn(0f32, 1f32, (b, self.in_channels, patch_size), device)?;
        let z = (z.to_dtype(dtype)? * temperature)?;

        let n = n_timesteps + 1;
        let t_span: Vec<f64> = (0..n)
            .map(|i| {
                let t = 1.0 - i as f64 / n_timesteps as f64;
                t + sway_sampling_coef * ((std::f64::consts::PI / 2.0 * t).cos() - 1.0 + t)
            })
            .collect();

        self.solve_euler(z, &t_span, mu, cond, cfg_value, use_cfg_zero_star, b, device, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_euler(
        &mut self,
        mut x: Tensor,
        t_span: &[f64],
        mu: &Tensor,
        cond: &Tensor,
        cfg_value: f64,
        use_cfg_zero_star: bool,
        b: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let n = t_span.len();
        let zero_init_steps = ((0.04 * n as f64) as usize).max(1);

        let mut t = t_span[0];
        let mut dt = t_span[0] - t_span[1];

        for step in 1..n {
            let dphi_dt = if use_cfg_zero_star && step <= zero_init_steps {
                x.zeros_like()?
            } else {
                let x2 = Tensor::cat(&[&x, &x], 0)?;
                let mu_zero = mu.zeros_like()?;
                let mu2 = Tensor::cat(&[mu, &mu_zero], 0)?;
                let cond2 = Tensor::cat(&[cond, cond], 0)?;
                let t2 = Tensor::full(t as f32, 2 * b, device)?.to_dtype(dtype)?;
                let dt2 = if self.mean_mode {
                    Tensor::full(dt as f32, 2 * b, device)?.to_dtype(dtype)?
                } else {
                    Tensor::zeros(2 * b, dtype, device)?
                };

                let out = self.estimator.forward(&x2, &mu2, &t2, &cond2, &dt2)?; // [2b, C, T]
                let cond_out = out.narrow(0, 0, b)?;
                let uncond_out = out.narrow(0, b, b)?;

                let scale = if use_cfg_zero_star {
                    let pos_flat = cond_out.flatten_from(1)?.to_dtype(DType::F32)?; // [b, C*T]
                    let neg_flat = uncond_out.flatten_from(1)?.to_dtype(DType::F32)?;
                    let dot = (&pos_flat * &neg_flat)?.sum(1)?; // [b]
                    let sq_norm = ((&neg_flat * &neg_flat)?.sum(1)? + 1e-8)?;
                    (dot / sq_norm)?.to_dtype(dtype)? // [b]
                } else {
                    Tensor::ones(b, dtype, device)?
                };
                let scale = scale.reshape((b, 1, 1))?;
                let uncond_scaled = uncond_out.broadcast_mul(&scale)?;
                let diff = (cond_out - &uncond_scaled)?;
                (uncond_scaled + diff.affine(cfg_value, 0.0)?)?
            };

            x = (x - dphi_dt.affine(dt, 0.0)?)?;
            t -= dt;
            if step < n - 1 {
                dt = t - t_span[step + 1];
            }
        }

        Ok(x)
    }
}

#[cfg(test)]
mod hf_diff {
    use super::*;
    use crate::models::voxcpm2::config::load_config;
    use crate::models::voxcpm2::local_dit::VoxCpmLocDit;
    use candle_core::DType;
    use candle_nn::VarBuilder;

    // Opt-in: cross-checks `solve_euler` (sway sampling + CFG-Zero-star)
    // against a real HF dump with shared mu/cond/z/t_span inputs (see
    // voxcpm_src_cfm_diff.py). `cargo test --release -p crane-core
    // cfm_solve_euler_matches_python -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint + a matching HF dump"]
    fn cfm_solve_euler_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/voxcpm_diff";
        let device = Device::Cpu;
        let dtype = DType::F32;

        let cfg = load_config(&format!("{model_path}/config.json")).unwrap();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[format!("{model_path}/model.safetensors")], dtype, &device) }.unwrap();

        let dit_cfg = cfg.lm_config.derive(&cfg.dit_config.shape);
        let estimator =
            VoxCpmLocDit::new(&dit_cfg, cfg.feat_dim, cfg.max_length, vb.pp("feat_decoder").pp("estimator")).unwrap();
        let mut cfm = UnifiedCfm::new(estimator, cfg.dit_config.mean_mode);

        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/meta.json")).unwrap()).unwrap();
        let (b, patch_size, feat_dim, dit_hidden, n_timesteps) = (
            meta["b"].as_u64().unwrap() as usize,
            meta["patch_size"].as_u64().unwrap() as usize,
            meta["feat_dim"].as_u64().unwrap() as usize,
            meta["dit_hidden"].as_u64().unwrap() as usize,
            meta["n_timesteps"].as_u64().unwrap() as usize,
        );

        let load = |name: &str, shape: (usize, usize, usize)| -> Tensor {
            let raw = std::fs::read(format!("{diff_dir}/{name}.bin")).unwrap();
            let floats: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            Tensor::from_vec(floats, shape, &device).unwrap()
        };
        let mu = load("mu", (b, 2 * dit_hidden, 1)).reshape((b, 2 * dit_hidden)).unwrap();
        let cond = load("cond", (b, feat_dim, patch_size));
        let z = load("z", (b, feat_dim, patch_size));

        let n = n_timesteps + 1;
        let sway = 1.0;
        let t_span: Vec<f64> = (0..n)
            .map(|i| {
                let t = 1.0 - i as f64 / n_timesteps as f64;
                t + sway * ((std::f64::consts::PI / 2.0 * t).cos() - 1.0 + t)
            })
            .collect();

        let result = cfm.solve_euler(z, &t_span, &mu, &cond, 2.0, true, b, &device, dtype).unwrap();
        eprintln!("result shape: {:?}", result.dims());

        let hf: Vec<f32> = {
            let raw = std::fs::read(format!("{diff_dir}/hf_cfm_result.bin")).unwrap();
            raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let rust: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(hf.len(), rust.len());

        let diff: f32 = hf.iter().zip(&rust).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let cos = {
            let dot: f32 = hf.iter().zip(&rust).map(|(a, b)| a * b).sum();
            let na: f32 = hf.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nb: f32 = rust.iter().map(|a| a * a).sum::<f32>().sqrt();
            dot / (na * nb + 1e-12)
        };
        eprintln!("max abs diff: {diff}, cosine: {cos}");
        assert!(cos > 0.999, "cosine similarity too low: {cos}");
    }
}
