use candle_core::{Result, Tensor, D};
use candle_nn::{Activation, Module, VarBuilder};

use crate::models::with_tracing::{linear_no_bias, Linear};

/// `SwiGLU` feed-forward network with merged gate/up projection.
///
/// Computes `activation(x @ gate) * (x @ up) @ down` with a single fused
/// matmul for the gate+up path. Uses `fused_silu_mul` on CUDA when the
/// activation is `Silu`.
#[derive(Debug, Clone)]
pub struct SwiGluFfn {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
    activation: Activation,
}

impl SwiGluFfn {
    /// Create a `SwiGLU` FFN: `activation(x @ gate_proj) * (x @ up_proj) @ down_proj`.
    ///
    /// Gate and up weights are merged into a single `[2*intermediate_size, hidden_size]`
    /// matrix at construction time, enabling a single matmul per forward pass and the
    /// `fused_silu_mul` CUDA kernel when the activation is `Silu`.
    ///
    /// # Arguments
    /// * `hidden_size` - Input and output dimension
    /// * `intermediate_size` - Intermediate (gate/up output) dimension
    /// * `activation` - Activation applied to the gate branch (e.g. `Activation::Silu`)
    /// * `vb` - `VarBuilder` scoped to this layer; loads weights at `gate_proj`, `up_proj`, `down_proj`
    ///
    /// # Errors
    ///
    /// Returns a candle error if weight loading fails.
    // VarBuilder::pp() consumes self, so it must be taken by value.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let gate_up_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        // gate_proj and up_proj are dropped here — their memory is freed.
        let gate_up_proj = Linear::from_weights(gate_up_w, None);
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
            activation,
        })
    }
}

impl Module for SwiGluFfn {
    /// Compute `activation(x @ gate) * (x @ up) @ down`.
    ///
    /// On CUDA with `Silu` activation, uses `fused_silu_mul` for a single kernel launch.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gu = x.apply(&self.gate_up_proj)?;

        #[cfg(feature = "cuda")]
        if matches!(self.activation, Activation::Silu) && gu.device().is_cuda() {
            let activated =
                crate::ops::fused_silu_mul(&gu.contiguous()?, self.intermediate_size)?;
            return self.down_proj.forward(&activated);
        }

        let gate = gu.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gu.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = gate.apply(&self.activation)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device, Tensor};

    // Keys use dot-separated format ("gate_proj.weight") to match VarBuilder::pp("gate_proj").
    fn make_vb(
        hidden: usize,
        intermediate: usize,
        gate_data: Vec<f32>,
        up_data: Vec<f32>,
        down_data: Vec<f32>,
    ) -> candle_nn::VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate_proj.weight".to_string(),
            Tensor::from_vec(gate_data, (intermediate, hidden), device).expect("gate weight"),
        );
        tensors.insert(
            "up_proj.weight".to_string(),
            Tensor::from_vec(up_data, (intermediate, hidden), device).expect("up weight"),
        );
        tensors.insert(
            "down_proj.weight".to_string(),
            Tensor::from_vec(down_data, (hidden, intermediate), device).expect("down weight"),
        );
        candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    fn zeros_ffn(hidden: usize, intermediate: usize, act: Activation) -> SwiGluFfn {
        let gate = vec![0.0f32; hidden * intermediate];
        let up = vec![0.0f32; hidden * intermediate];
        let down = vec![0.0f32; intermediate * hidden];
        let vb = make_vb(hidden, intermediate, gate, up, down);
        SwiGluFfn::new(hidden, intermediate, act, vb).expect("SwiGluFfn::new")
    }

    fn identity_vb(hidden: usize) -> candle_nn::VarBuilder<'static> {
        let eye: Vec<f32> = (0..hidden * hidden)
            .map(|idx| if idx / hidden == idx % hidden { 1.0_f32 } else { 0.0 })
            .collect();
        make_vb(hidden, hidden, eye.clone(), eye.clone(), eye)
    }

    #[test]
    fn test_output_shape_2d() {
        let ffn = zeros_ffn(8, 16, Activation::Silu);
        let x = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = ffn.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[4, 8]);
    }

    #[test]
    fn test_output_shape_3d() {
        let ffn = zeros_ffn(16, 32, Activation::Silu);
        let x = Tensor::zeros((2, 5, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = ffn.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn test_zero_weights_give_zero_output() {
        let ffn = zeros_ffn(8, 16, Activation::Silu);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let y = ffn.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero weights, got max={max_val}"
        );
    }

    #[test]
    fn test_zero_input_gives_zero_output() {
        let ffn = identity_vb(8);
        let ffn = SwiGluFfn::new(8, 8, Activation::Silu, ffn).expect("new");
        let x = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = ffn.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero input, got max={max_val}"
        );
    }

    #[test]
    fn test_intermediate_size_larger_than_hidden() {
        let ffn = zeros_ffn(32, 128, Activation::Silu);
        let x = Tensor::zeros((1, 32), DType::F32, &Device::Cpu).expect("zeros");
        let y = ffn.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[1, 32]);
    }

    #[test]
    fn test_activation_changes_output() {
        // Same weights and input; SiLU and GELU must produce different values on non-trivial input.
        let device = &Device::Cpu;
        let hidden = 4usize;
        let intermediate = 8usize;
        let gate: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let up: Vec<f32> = gate.iter().map(|v| v * 0.5).collect();
        let down: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let vb_silu = make_vb(hidden, intermediate, gate.clone(), up.clone(), down.clone());
        let vb_gelu = make_vb(hidden, intermediate, gate, up, down);
        let ffn_silu = SwiGluFfn::new(hidden, intermediate, Activation::Silu, vb_silu).expect("silu");
        let ffn_gelu =
            SwiGluFfn::new(hidden, intermediate, Activation::GeluPytorchTanh, vb_gelu).expect("gelu");

        let x = Tensor::ones((1, hidden), DType::F32, device).expect("ones");
        let out_silu = ffn_silu.forward(&x).expect("silu forward");
        let out_gelu = ffn_gelu.forward(&x).expect("gelu forward");

        let diff: f32 = (&out_silu - &out_gelu)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff > 1e-4,
            "SiLU and GeluPytorchTanh should produce different outputs; got max_diff={diff}"
        );
    }

    #[test]
    fn test_formula_manual_verification() {
        // hidden=2, intermediate=2
        // gate_proj.weight = [[1, 0], [0, 1]]  (I, row-major so output[i] = x[i])
        // up_proj.weight   = [[2, 0], [0, 2]]  (2*I)
        // down_proj.weight = [[1, 0], [0, 1]]  (I)
        // x = [1.0, 0.5]
        //
        // Linear stores weight as [out, in] and computes x @ weight.T:
        //   gate = x @ I.T = [1.0, 0.5]
        //   gate_activated = silu([1.0, 0.5])
        //   up = x @ (2*I).T = [2.0, 1.0]
        //   product = gate_activated * up
        //   output = product @ I.T = product
        let device = &Device::Cpu;
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let doubled = vec![2.0f32, 0.0, 0.0, 2.0];
        let vb = make_vb(2, 2, identity.clone(), doubled, identity);
        let ffn = SwiGluFfn::new(2, 2, Activation::Silu, vb).expect("new");

        let x = Tensor::new(&[1.0f32, 0.5], device)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = ffn.forward(&x).expect("forward");
        let got = y.squeeze(0).expect("squeeze").to_vec1::<f32>().expect("to_vec1");

        let silu = |v: f32| v / (1.0 + (-v).exp());
        let expected = [silu(1.0_f32) * 2.0, silu(0.5_f32) * 1.0];

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "output[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn test_batch_consistency() {
        // Identical rows in a batch should produce identical output rows.
        let device = &Device::Cpu;
        let hidden = 8usize;
        let intermediate = 16usize;
        let gate: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let up: Vec<f32> = gate.iter().map(|v| v * 0.5).collect();
        let down: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let vb = make_vb(hidden, intermediate, gate, up, down);
        let ffn = SwiGluFfn::new(hidden, intermediate, Activation::Silu, vb).expect("new");

        let row: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let single = Tensor::from_vec(row, (1, hidden), device).expect("single row");
        let batch = Tensor::cat(&[&single, &single, &single], 0).expect("cat");

        let out_batch = ffn.forward(&batch).expect("batch forward");
        let out_single = ffn.forward(&single).expect("single forward");

        for b in 0..3 {
            let row_b = out_batch.narrow(0, b, 1).expect("narrow");
            let diff: f32 = (&row_b - &out_single)
                .expect("sub")
                .abs()
                .expect("abs")
                .max_all()
                .expect("max_all")
                .to_scalar()
                .expect("scalar");
            assert!(diff < 1e-6, "batch row {b} differs from single: max_diff={diff}");
        }
    }
}
