//! CosyVoice2's `CausalConditionalCFM` — the Euler ODE sampler driving
//! [`super::dit::DiT`]. Ported from `stepaudio2.cosyvoice2.flow.flow_matching`
//! (real reference source). Simpler than VoxCPM2's CFM: plain cosine-warped
//! timesteps (no sway-sampling) and plain linear classifier-free guidance
//! (no CFG-Zero-star) — see the module doc on `Token2wav` for the summary.
//! Non-streaming (`forward`/`solve_euler`) only.

use candle_core::{Result, Tensor};

use super::dit::DiT;

/// `x`: initial noise `[1, 80, T]`. `mu`: encoder output `[1, 80, T]`.
/// `spks`: speaker embedding `[1, 80]`. `cond`: prompt-mel conditioning
/// `[1, 80, T]` (zero beyond the prompt length). `n_timesteps`: Euler steps.
/// `cfg_rate`: classifier-free guidance rate (`0.7` for this checkpoint).
///
/// # Errors
///
/// Returns an error if any `DiT` forward pass fails.
pub fn solve_euler(estimator: &DiT, x: &Tensor, mu: &Tensor, spks: &Tensor, cond: &Tensor, n_timesteps: usize, cfg_rate: f64) -> Result<Tensor> {
    let device = x.device();
    let dtype = x.dtype();

    // Cosine-scheduled timesteps: t_span = 1 - cos(linspace(0,1,n+1) * pi/2).
    let mut t_span = Vec::with_capacity(n_timesteps + 1);
    for i in 0..=n_timesteps {
        let lin = i as f64 / n_timesteps as f64;
        t_span.push(1.0 - (lin * std::f64::consts::FRAC_PI_2).cos());
    }

    let zeros_mu = mu.zeros_like()?;
    let zeros_spks = spks.zeros_like()?;
    let zeros_cond = cond.zeros_like()?;
    let mu_in = Tensor::cat(&[mu, &zeros_mu], 0)?;
    let spks_in = Tensor::cat(&[spks, &zeros_spks], 0)?;
    let cond_in = Tensor::cat(&[cond, &zeros_cond], 0)?;

    let mut x = x.clone();
    let mut t = t_span[0];
    let mut dt = t_span[1] - t_span[0];

    for step in 1..=n_timesteps {
        let x_in = Tensor::cat(&[&x, &x], 0)?;
        let t_in = Tensor::from_vec(vec![t as f32, t as f32], 2, device)?.to_dtype(dtype)?;

        let dphi_dt = estimator.forward(&x_in, &mu_in, &t_in, &spks_in, &cond_in)?;
        let dphi_dt_cond = dphi_dt.narrow(0, 0, 1)?;
        let dphi_dt_uncond = dphi_dt.narrow(0, 1, 1)?;
        let dphi_dt = ((1.0 + cfg_rate) * dphi_dt_cond)? - (cfg_rate * dphi_dt_uncond)?;
        let dphi_dt = dphi_dt?;

        x = (x + (dphi_dt * dt)?)?;
        t += dt;
        if step < n_timesteps {
            dt = t_span[step + 1] - t;
        }
    }

    Ok(x)
}
