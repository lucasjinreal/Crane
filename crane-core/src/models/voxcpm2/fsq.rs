//! Scalar-quantization bottleneck applied to `base_lm`'s hidden state at
//! audio positions — VoxCPM2's "tokenizer-free" discretization: a
//! differentiable-in-training, hard-round-in-inference bottleneck instead of
//! a VQ codebook lookup. Port of `layers/scalar_quantization_layer.py`
//! (inference path only — no straight-through estimator, this crate never
//! trains).

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct ScalarQuantizationLayer {
    in_proj: Linear,
    out_proj: Linear,
    scale: f64,
}

impl ScalarQuantizationLayer {
    pub fn new(in_dim: usize, out_dim: usize, latent_dim: usize, scale: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            in_proj: linear(in_dim, latent_dim, vb.pp("in_proj"))?,
            out_proj: linear(latent_dim, out_dim, vb.pp("out_proj"))?,
            scale: scale as f64,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.in_proj.forward(hidden)?.tanh()?;
        let hidden = (round_half_to_even(&(hidden * self.scale)?)? / self.scale)?;
        self.out_proj.forward(&hidden)
    }
}

/// `Tensor::round` is "round half away from zero" (Rust's `f32::round`
/// convention) — `torch.round`/Python's builtin `round` is IEEE 754 "round
/// half to even" (banker's rounding), and they disagree at every exact
/// `.5` boundary that isn't already even (confirmed directly: `0.5→1` vs
/// `→0`, `2.5→3` vs `→2`, `4.5→5` vs `→4`, ...). This bottleneck's whole
/// purpose is snapping values to a small discrete grid (`scale=9` → ties
/// roughly every `1/18` of the tanh-bounded range), and a
/// straight-through-estimator-trained quantizer like this one plausibly
/// learns to push many dimensions right up against those boundaries — so
/// the tie-breaking convention isn't a rounding-error footnote here, it's
/// a real, discrete per-dimension divergence source. Found live: root cause
/// of a multi-turn VoxCPM2 generation bug where autoregressively re-quantizing
/// this layer's own prior output compounded these per-step discrete jumps
/// into complete generation collapse after a few steps, reproducible
/// identically in both bf16 and f32 (ruling out ordinary precision loss) —
/// see `model.rs`'s module docs / `AGENTS.md` for the full investigation.
///
/// `2 * round_away(x / 2)` gives the *even*-rounded result exactly at tie
/// points (verified: `round_half_even(2.5)=2`, `round_half_even(-2.5)=-2`,
/// etc.) but is wrong elsewhere (e.g. `round_half_even(2.7)` must be `3`,
/// not `2`) — so it's only used to correct the ordinary "away from zero"
/// result at positions where `x` is *exactly* `.5` away from its own
/// away-from-zero rounding (the only case where the two conventions can
/// differ), leaving every non-tie value untouched.
fn round_half_to_even(x: &Tensor) -> Result<Tensor> {
    let away = x.round()?;
    let is_tie = (x - &away)?.abs()?.eq(0.5)?;
    let even = (x / 2.0)?.round()?.affine(2.0, 0.0)?;
    is_tie.where_cond(&even, &away)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn round_half_to_even_matches_torch_round_at_ties() {
        let device = Device::Cpu;
        // Every exact `.5` tie in [-4.5, 4.5], plus non-tie values that must
        // round normally (both conventions agree there, so any regression
        // in the tie-detection mask would show up as *these* breaking, not
        // just the ties).
        let input = Tensor::new(
            &[0.5f32, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5, -3.5, -4.5, 2.3, 2.7, -2.3, -2.7, 0.0, 3.0],
            &device,
        )
        .unwrap();
        // `torch.round()` on this exact input (verified live against PyTorch).
        let expected = [
            0.0f32, 2.0, 2.0, 4.0, 4.0, -0.0, -2.0, -2.0, -4.0, -4.0, 2.0, 3.0, -2.0, -3.0, 0.0, 3.0,
        ];
        let got: Vec<f32> = round_half_to_even(&input).unwrap().to_vec1().unwrap();
        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() < 1e-6, "index {i}: got {g}, expected {e} (input {})", got[i]);
        }
    }

    #[test]
    fn round_half_to_even_differs_from_plain_round_at_ties() {
        // Documents *why* this function exists: plain `Tensor::round` (Rust's
        // round-half-away-from-zero) disagrees with `round_half_to_even` at
        // every tie that isn't already even.
        let device = Device::Cpu;
        let input = Tensor::new(&[0.5f32, 2.5, 4.5], &device).unwrap();
        let plain: Vec<f32> = input.round().unwrap().to_vec1().unwrap();
        let even: Vec<f32> = round_half_to_even(&input).unwrap().to_vec1().unwrap();
        assert_eq!(plain, vec![1.0, 3.0, 5.0]);
        assert_eq!(even, vec![0.0, 2.0, 4.0]);
    }
}
