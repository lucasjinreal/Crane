//! DPM-Solver++ (SDE) multistep scheduler — the noise/velocity-to-latent
//! sampler driving [`super::diffusion_head::DiffusionHead`].
//!
//! `kugelaudio_open.schedule.dpm_solver` vendors HuggingFace `diffusers`'
//! `DPMSolverMultistepScheduler` verbatim (its own doc string:
//! "This file is strongly influenced by
//! <https://github.com/LuChengTHU/dpm-solver>"). That scheduler supports a
//! large matrix of options (`dpmsolver`/`dpmsolver++`/`sde-dpmsolver`/
//! `sde-dpmsolver++`, `epsilon`/`sample`/`v_prediction`, Karras/Lu sigma
//! schedules, dynamic thresholding, solver orders 1-3, …), but the reference
//! checkpoint's `diffusion_head_config` only ever selects one corner of it:
//!
//! - `ddpm_beta_schedule: "cosine"` (Glide/`squaredcos_cap_v2`)
//! - `ddpm_algorithm_type: "sde-dpmsolver++"`
//! - `prediction_type: "v_prediction"`
//! - solver order 2 (`diffusers`' own default — not a `config.json` field)
//! - `solver_type: "midpoint"`, `lower_order_final: true`,
//!   `euler_at_final: false`, `use_karras_sigmas: false`,
//!   `use_lu_lambdas: false`, `final_sigmas_type: "zero"`,
//!   `timestep_spacing: "linspace"`, `thresholding: false`,
//!   `lambda_min_clipped: -inf` (all `diffusers` defaults — none of these
//!   are `config.json` fields either)
//!
//! This port only implements that corner — [`DpmSolverScheduler::new`] bails
//! if `beta_schedule`/`algorithm_type`/`prediction_type` don't match, and
//! [`DpmSolverScheduler::step`] only implements first- and second-order
//! updates (matching the fixed solver order 2; a third-order update is
//! never reached at that order and is not ported). With
//! `final_sigmas_type == "zero"` always true, `diffusers`' `lower_order_final`
//! condition — an OR of four terms, one of which is exactly
//! `self.config.final_sigmas_type == "zero"` — collapses to just
//! `is_last_or_past`; see [`DpmSolverScheduler::step`]'s comment.
//!
//! Deviation from the Python: `step()` upcasts both `model_output` and
//! `sample` to F32 for the *entire* step (Python only upcasts `sample`, and
//! only after `convert_model_output`). Strictly more precision, never less;
//! done for implementation simplicity (see that method's doc comment).

use candle_core::{DType, Result, Tensor};
use std::f64::consts::PI;

use super::config::DiffusionHeadConfig;

/// `betas_for_alpha_bar` (`alpha_transform_type="cosine"`): Glide's cosine
/// noise schedule, discretized into `num_train_timesteps` betas.
fn betas_cosine(num_train_timesteps: usize) -> Vec<f64> {
    let alpha_bar = |t: f64| ((t + 0.008) / 1.008 * PI / 2.0).cos().powi(2);
    (0..num_train_timesteps)
        .map(|i| {
            let t1 = i as f64 / num_train_timesteps as f64;
            let t2 = (i + 1) as f64 / num_train_timesteps as f64;
            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999)
        })
        .collect()
}

/// `np.round`'s round-half-to-even, needed for [`DpmSolverScheduler::set_timesteps`]'s
/// `linspace(...).round()` to land on the exact same integer timesteps as
/// the Python — Rust's `f64::round()` rounds half away from zero instead.
/// (This codebase has hit real bugs from exactly this Rust/NumPy rounding
/// mismatch before, in VoxCPM2's FSQ layer.)
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    if (x - floor - 0.5).abs() < 1e-9 {
        if (floor as i64).rem_euclid(2) == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

fn sigma_to_alpha_sigma_t(sigma: f64) -> (f64, f64) {
    let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
    let sigma_t = sigma * alpha_t;
    (alpha_t, sigma_t)
}

/// `x * scale`, `scale` a host-side f64 (mirrors the Python's scalar
/// `torch.Tensor`-times-tensor broadcasts, computed here as a plain affine
/// — `sigma`/`alpha`/`lambda` are themselves host scalars throughout).
fn scale(x: &Tensor, s: f64) -> Result<Tensor> {
    x.affine(s, 0.0)
}

/// Fixed solver order — see module doc comment. `diffusers`' own default;
/// not read from `config.json`.
const SOLVER_ORDER: usize = 2;

pub struct DpmSolverScheduler {
    /// `sigmas[t] = sqrt((1 - alphas_cumprod[t]) / alphas_cumprod[t])` for
    /// `t` in `0..num_train_timesteps`, indexed by raw (untouched) timestep.
    sigmas_full: Vec<f64>,
    num_train_timesteps: usize,

    // Set by `set_timesteps`; empty/zeroed before the first call.
    timesteps: Vec<i64>,
    /// `sigmas[0..num_inference_steps]` interpolated from `sigmas_full` at
    /// each `timesteps[i]`, plus a final `0.0` (`final_sigmas_type == "zero"`).
    sigmas: Vec<f64>,
    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: usize,
    step_index: usize,
}

impl DpmSolverScheduler {
    /// Build a scheduler from `diffusion_head_config`. Bails if the config
    /// selects anything outside the one corner this port implements (see
    /// module doc comment).
    pub fn new(cfg: &DiffusionHeadConfig) -> Result<Self> {
        if cfg.ddpm_beta_schedule != "cosine" {
            candle_core::bail!(
                "kugelaudio dpm_solver: unsupported beta_schedule {:?} (only \"cosine\" is implemented)",
                cfg.ddpm_beta_schedule
            );
        }
        if cfg.ddpm_algorithm_type != "sde-dpmsolver++" {
            candle_core::bail!(
                "kugelaudio dpm_solver: unsupported algorithm_type {:?} (only \"sde-dpmsolver++\" is implemented)",
                cfg.ddpm_algorithm_type
            );
        }
        if cfg.prediction_type != "v_prediction" {
            candle_core::bail!(
                "kugelaudio dpm_solver: unsupported prediction_type {:?} (only \"v_prediction\" is implemented)",
                cfg.prediction_type
            );
        }
        let num_train_timesteps = cfg.ddpm_num_steps;
        let betas = betas_cosine(num_train_timesteps);
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut acc = 1.0f64;
        for b in &betas {
            acc *= 1.0 - b;
            alphas_cumprod.push(acc);
        }
        let sigmas_full = alphas_cumprod
            .iter()
            .map(|ac| ((1.0 - ac) / ac).sqrt())
            .collect();
        Ok(Self {
            sigmas_full,
            num_train_timesteps,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            model_outputs: vec![None; SOLVER_ORDER],
            lower_order_nums: 0,
            step_index: 0,
        })
    }

    /// Convenience: build directly from `ddpm_num_inference_steps` in the
    /// config and call [`Self::set_timesteps`].
    pub fn new_with_default_steps(cfg: &DiffusionHeadConfig) -> Result<Self> {
        let mut s = Self::new(cfg)?;
        s.set_timesteps(cfg.ddpm_num_inference_steps);
        Ok(s)
    }

    /// Sets the `num_inference_steps` discrete timesteps used for the
    /// denoising chain, and resets all per-generation solver state
    /// (`model_outputs`, `lower_order_nums`, `step_index`). Must be called
    /// before [`Self::step`]; call again to start a fresh denoising chain
    /// (e.g. for the next generated patch).
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        // `lambda_min_clipped = -inf` (never overridden) ⇒ `last_timestep`
        // is always `num_train_timesteps` — the `searchsorted`-based clipping
        // in the Python is dead code for this config and is not ported.
        let last_timestep = self.num_train_timesteps;
        // `timestep_spacing == "linspace"`:
        // `np.linspace(0, last_timestep-1, n+1).round()[::-1][:-1]`.
        let n = num_inference_steps;
        let mut ts: Vec<i64> = (0..=n)
            .map(|i| {
                let v = if n == 0 {
                    0.0
                } else {
                    i as f64 * (last_timestep - 1) as f64 / n as f64
                };
                round_half_even(v) as i64
            })
            .collect();
        ts.reverse();
        ts.pop(); // drop what was index 0 pre-reverse (the "[:-1]")

        let mut sigmas: Vec<f64> = ts.iter().map(|&t| self.sigmas_full[t as usize]).collect();
        sigmas.push(0.0); // final_sigmas_type == "zero"

        self.timesteps = ts;
        self.sigmas = sigmas;
        self.model_outputs = vec![None; SOLVER_ORDER];
        self.lower_order_nums = 0;
        self.step_index = 0;
    }

    /// The timesteps to iterate `step()` over, in order, matching Python's
    /// `for t in scheduler.timesteps: ...` usage
    /// (`kugelaudio_inference.py`). The `t` values themselves aren't
    /// consumed by [`Self::step`] (which tracks its own `step_index`
    /// counter instead, driven purely by call order) — exposed for parity
    /// with the Python call site and so callers can report progress.
    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    /// One denoising step: `model_output` is the diffusion head's raw
    /// (v-prediction) output for `sample` at the *current* `step_index`'s
    /// timestep; returns the sample at the *next* (less noisy) timestep.
    /// Must be called exactly `timesteps().len()` times per generation
    /// (matching a `for t in timesteps() { sample = step(...) }` loop) —
    /// `step_index` advances by one on every call, sourced purely from call
    /// order (see [`Self::timesteps`]'s doc comment).
    ///
    /// `model_output`/`sample` may be any shape, so long as they match
    /// (typically `[1, latent_size]` for one generated patch, or
    /// `[N, latent_size]` when denoising `N` positions' latents in
    /// lockstep — e.g. the `ddpm_batch_mul` training path this port doesn't
    /// use, or a batched inference path a future caller might add).
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let device = model_output.device().clone();
        let orig_dtype = sample.dtype();
        let model_output = model_output.to_dtype(DType::F32)?;
        let sample = sample.to_dtype(DType::F32)?;

        let num_inference_steps = self.timesteps.len();
        let is_last_or_past = self.step_index >= num_inference_steps.saturating_sub(1);
        // `final_sigmas_type == "zero"` is always true for this config, which
        // makes the Python's 4-way OR unconditionally true whenever
        // `is_last_or_past` — see module doc comment.
        let lower_order_final = is_last_or_past;
        // Python's second-order branch is `elif self.config.solver_order == 2
        // or self.lower_order_nums < 2 or lower_order_second`, and `or`
        // short-circuits on the first (always-true, since `SOLVER_ORDER` is
        // fixed at 2) term — so `lower_order_nums`/`lower_order_second` never
        // actually get consulted here and are not computed at all.

        // convert_model_output (sde-dpmsolver++, v_prediction):
        // x0_pred = alpha_t * sample - sigma_t * model_output
        let safe_idx = self.step_index.min(self.sigmas.len() - 1);
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma_t(self.sigmas[safe_idx]);
        let converted = (scale(&sample, alpha_t)? - scale(&model_output, sigma_t)?)?;

        for i in 0..SOLVER_ORDER - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
        }
        let last = SOLVER_ORDER - 1;
        self.model_outputs[last] = Some(converted.clone());

        let noise = Tensor::randn(0f32, 1f32, model_output.shape(), &device)?;

        let prev_sample = if self.lower_order_nums < 1 || lower_order_final {
            self.first_order_update(&converted, &sample, &noise)?
        } else {
            self.second_order_update(&sample, &noise)?
        };

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;

        prev_sample.to_dtype(orig_dtype)
    }

    fn first_order_update(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let current_index = self.step_index.min(self.sigmas.len() - 1);
        let next_index = (self.step_index + 1).min(self.sigmas.len() - 1);
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma_t(self.sigmas[next_index]);
        let (alpha_s, sigma_s) = sigma_to_alpha_sigma_t(self.sigmas[current_index]);
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s = alpha_s.ln() - sigma_s.ln();
        let h = lambda_t - lambda_s;

        let term_sample = scale(sample, sigma_t / sigma_s * (-h).exp())?;
        let term_model = scale(model_output, alpha_t * (1.0 - (-2.0 * h).exp()))?;
        let term_noise = scale(noise, sigma_t * (1.0 - (-2.0 * h).exp()).sqrt())?;
        (term_sample + term_model)? + term_noise
    }

    fn second_order_update(&self, sample: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let current_index = self.step_index.min(self.sigmas.len() - 1);
        let next_index = (self.step_index + 1).min(self.sigmas.len() - 1);
        let prev_index = self.step_index.saturating_sub(1);
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma_t(self.sigmas[next_index]);
        let (alpha_s0, sigma_s0) = sigma_to_alpha_sigma_t(self.sigmas[current_index]);
        let (alpha_s1, sigma_s1) = sigma_to_alpha_sigma_t(self.sigmas[prev_index]);
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s0 = alpha_s0.ln() - sigma_s0.ln();
        let lambda_s1 = alpha_s1.ln() - sigma_s1.ln();

        // m0 = most recent converted output, m1 = the one before it.
        let m0 = self.model_outputs[SOLVER_ORDER - 1]
            .as_ref()
            .expect("model_outputs[-1] must be set by step() before calling second_order_update");
        let m1 = self.model_outputs[SOLVER_ORDER - 2]
            .as_ref()
            .expect("model_outputs[-2] must be set (lower_order_nums >= 1 guards this call)");

        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let r0 = h0 / h;
        let d0 = m0;
        let d1 = scale(&(m0 - m1)?, 1.0 / r0)?;

        let coeff_d = alpha_t * (1.0 - (-2.0 * h).exp());
        let term_sample = scale(sample, sigma_t / sigma_s0 * (-h).exp())?;
        let term_d0 = scale(d0, coeff_d)?;
        let term_d1 = scale(&d1, 0.5 * coeff_d)?;
        let term_noise = scale(noise, sigma_t * (1.0 - (-2.0 * h).exp()).sqrt())?;
        ((term_sample + term_d0)? + term_d1)? + term_noise
    }
}

/// Same math as [`DpmSolverScheduler`], run entirely on host-side `f64`
/// scalars — used by tests to cross-check the tensor-based scheduler
/// without depending on it, and available for a future caller that only
/// needs the scalar schedule (e.g. to size a batched denoise loop).
#[cfg(test)]
fn sigmas_for_inference_steps(cfg: &DiffusionHeadConfig, num_inference_steps: usize) -> Vec<f64> {
    let mut s = DpmSolverScheduler::new(cfg).expect("scheduler");
    s.set_timesteps(num_inference_steps);
    s.sigmas.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cfg() -> DiffusionHeadConfig {
        DiffusionHeadConfig {
            hidden_size: 16,
            latent_size: 8,
            head_layers: 2,
            head_ffn_ratio: 3.0,
            rms_norm_eps: 1e-5,
            prediction_type: "v_prediction".to_string(),
            ddpm_beta_schedule: "cosine".to_string(),
            ddpm_num_steps: 1000,
            ddpm_num_inference_steps: 20,
            ddpm_algorithm_type: "sde-dpmsolver++".to_string(),
        }
    }

    #[test]
    fn rejects_unsupported_config() {
        let mut c = cfg();
        c.prediction_type = "epsilon".to_string();
        assert!(DpmSolverScheduler::new(&c).is_err());

        let mut c = cfg();
        c.ddpm_algorithm_type = "dpmsolver++".to_string();
        assert!(DpmSolverScheduler::new(&c).is_err());

        let mut c = cfg();
        c.ddpm_beta_schedule = "linear".to_string();
        assert!(DpmSolverScheduler::new(&c).is_err());
    }

    #[test]
    fn timesteps_are_20_descending_values_in_range() {
        let mut s = DpmSolverScheduler::new(&cfg()).expect("new");
        s.set_timesteps(20);
        let ts = s.timesteps();
        assert_eq!(ts.len(), 20);
        assert!(
            ts.windows(2).all(|w| w[0] > w[1]),
            "timesteps must be strictly descending: {ts:?}"
        );
        assert!(ts.iter().all(|&t| (0..1000).contains(&t)));
        // linspace(0, 999, 21) round-trip: first (highest) timestep should be
        // very close to 999, matching diffusers' behavior for this config.
        assert!(*ts.first().unwrap() >= 990);
    }

    #[test]
    fn sigmas_length_is_steps_plus_one_and_ends_at_zero() {
        let sigmas = sigmas_for_inference_steps(&cfg(), 20);
        assert_eq!(sigmas.len(), 21);
        assert_eq!(*sigmas.last().unwrap(), 0.0);
        // sigmas must be strictly decreasing toward 0 (denoising = decreasing noise level).
        assert!(sigmas.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn round_half_even_matches_numpy_on_known_ties() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(0.3), 0.0);
        assert_eq!(round_half_even(0.7), 1.0);
    }

    #[test]
    fn full_denoise_loop_produces_finite_output_of_correct_shape() {
        let device = Device::Cpu;
        let mut s = DpmSolverScheduler::new(&cfg()).expect("new");
        s.set_timesteps(20);

        let latent_size = 8usize;
        let mut sample = Tensor::rand(-1f32, 1f32, (1, latent_size), &device).unwrap();
        let steps = s.timesteps().len();
        for _ in 0..steps {
            // Stand-in "model output": in real use this comes from
            // `DiffusionHead::forward`; a fixed function of `sample` here is
            // enough to exercise the scheduler's control flow and numerics.
            let fake_model_output = scale(&sample, 0.1).unwrap();
            sample = s.step(&fake_model_output, &sample).expect("step");
            assert_eq!(sample.dims(), &[1, latent_size]);
        }
        let max_abs: f32 = sample
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(max_abs.is_finite());
    }

    /// Same denoising loop on Metal (skipped where Metal isn't available) —
    /// `step`'s scalar-coefficient `cos`/`sin`/`exp` math is plain candle
    /// tensor ops with no CUDA-only path, but this is the only place that
    /// actually runs the scheduler on the Metal backend.
    #[test]
    fn full_denoise_loop_produces_finite_output_of_correct_shape_on_metal() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let device = Device::new_metal(0).expect("metal device");
        let mut s = DpmSolverScheduler::new(&cfg()).expect("new");
        s.set_timesteps(20);

        let latent_size = 8usize;
        let mut sample = Tensor::rand(-1f32, 1f32, (1, latent_size), &device).unwrap();
        let steps = s.timesteps().len();
        for _ in 0..steps {
            let fake_model_output = scale(&sample, 0.1).unwrap();
            sample = s.step(&fake_model_output, &sample).expect("step");
            assert_eq!(sample.dims(), &[1, latent_size]);
        }
        let max_abs: f32 = sample
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(max_abs.is_finite());
    }

    #[test]
    fn step_index_exhausted_after_full_loop() {
        let mut s = DpmSolverScheduler::new(&cfg()).expect("new");
        s.set_timesteps(20);
        let device = Device::Cpu;
        let mut sample = Tensor::zeros((1, 8), DType::F32, &device).unwrap();
        for _ in 0..20 {
            sample = s.step(&sample, &sample).expect("step");
        }
        assert_eq!(s.step_index, 20);
    }
}
