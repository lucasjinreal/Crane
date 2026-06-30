use candle_core::{Device, Result, Tensor};
use candle_nn::rotary_emb::rope;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Pre-computed cosine table with shape `[max_pos, dim/2]`.
    ///
    /// Half-width because inv-frequency pairs cover indices `0, 2, …, dim-2`.
    /// The fused `candle_nn::rotary_emb::rope()` kernel expects this format
    /// and handles the duplication internally.
    cos_table: Tensor,
    /// Pre-computed sine table with shape `[max_pos, dim/2]`. Same layout as `cos_table`.
    sin_table: Tensor,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding with pre-computed cos/sin tables.
    ///
    /// # Arguments
    /// * `dim` - Head dimension; inv\_freq is computed for `dim/2` pairs
    /// * `max_pos` - Maximum sequence position (table rows)
    /// * `theta` - `RoPE` base frequency (e.g. 10\_000.0 or 1\_000\_000.0)
    /// * `device` - Device to place the tables on
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor allocation or arithmetic fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn new(dim: usize, max_pos: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1.0_f64 / theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;

        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;
        Ok(Self {
            cos_table,
            sin_table,
        })
    }

    /// Apply rotary position embedding to query and key tensors.
    ///
    /// Uses the fused `candle_nn::rotary_emb::rope()` kernel. Cos/sin tables are
    /// cast to `q.dtype()` automatically so callers working in BF16/F16 do not
    /// need an explicit dtype conversion.
    ///
    /// # Arguments
    /// * `q` - Query tensor `[batch, n_heads, seq_len, head_dim]`
    /// * `k` - Key tensor `[batch, n_kv_heads, seq_len, head_dim]`
    /// * `start_pos` - Starting position index into the cos/sin tables
    /// * `seq_len` - Number of positions to apply
    ///
    /// # Errors
    ///
    /// Returns a candle error if shapes are incompatible or `start_pos + seq_len`
    /// exceeds `max_pos`.
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.forward(start_pos, seq_len)?;
        let cos = cos.to_dtype(q.dtype())?;
        let sin = sin.to_dtype(q.dtype())?;
        let q = rope(&q.contiguous()?, &cos, &sin)?;
        let k = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }

    /// Return cos/sin slices for positions `[start_pos..start_pos + seq_len]`.
    ///
    /// Both `narrow()` calls are zero-copy views over the pre-computed tables.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `start_pos + seq_len` exceeds `max_pos`.
    pub fn forward(&self, start_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, start_pos, seq_len)?;
        let sin = self.sin_table.narrow(0, start_pos, seq_len)?;
        Ok((cos, sin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // --- existing tests ---

    #[test]
    fn test_shape() {
        let rope = RotaryEmbedding::new(64, 128, 10000.0, &Device::Cpu)
            .expect("failed to construct RotaryEmbedding");
        let (cos, sin) = rope.forward(0, 32).expect("forward failed");
        assert_eq!(cos.dims(), &[32, 32]);
        assert_eq!(sin.dims(), &[32, 32]);
    }

    #[test]
    fn test_values_at_position_zero() {
        // At position 0 all angles are 0, so cos == 1 and sin == 0
        let rope = RotaryEmbedding::new(8, 4, 10000.0, &Device::Cpu)
            .expect("failed to construct RotaryEmbedding");
        let (cos, sin) = rope.forward(0, 1).expect("forward failed");
        for &c in rope_vec2(&cos)[0].iter() {
            assert!((c - 1.0).abs() < 1e-6, "expected cos=1 at pos 0, got {c}");
        }
        for &s in rope_vec2(&sin)[0].iter() {
            assert!(s.abs() < 1e-6, "expected sin=0 at pos 0, got {s}");
        }
    }

    #[test]
    fn test_apply() {
        let rope = RotaryEmbedding::new(64, 128, 10000.0, &Device::Cpu)
            .expect("failed to construct RotaryEmbedding");
        let q = Tensor::zeros((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)
            .expect("zeros failed");
        let k = Tensor::zeros((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)
            .expect("zeros failed");
        let (q_rot, k_rot) = rope.apply(&q, &k, 0, 4).expect("apply failed");
        assert_eq!(q_rot.dims(), &[1, 2, 4, 64]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 64]);
    }

    #[test]
    fn test_start_pos_offset() {
        let rope = RotaryEmbedding::new(64, 128, 10000.0, &Device::Cpu)
            .expect("failed to construct RotaryEmbedding");
        let (full_cos, full_sin) = rope.forward(0, 10).expect("forward failed");
        let (offset_cos, offset_sin) = rope.forward(5, 5).expect("forward failed");
        let expected_cos = full_cos.narrow(0, 5, 5).expect("narrow failed");
        let expected_sin = full_sin.narrow(0, 5, 5).expect("narrow failed");
        let max_cos_diff: f32 = (offset_cos - expected_cos)
            .expect("sub failed")
            .abs()
            .expect("abs failed")
            .max_all()
            .expect("max_all failed")
            .to_scalar()
            .expect("to_scalar failed");
        let max_sin_diff: f32 = (offset_sin - expected_sin)
            .expect("sub failed")
            .abs()
            .expect("abs failed")
            .max_all()
            .expect("max_all failed")
            .to_scalar()
            .expect("to_scalar failed");
        assert!(max_cos_diff < 1e-6, "cos offset mismatch: {max_cos_diff}");
        assert!(max_sin_diff < 1e-6, "sin offset mismatch: {max_sin_diff}");
    }

    // --- table correctness ---

    // dim=8, theta=10000 => inv_freq = [1.0, 0.1, 0.01, 0.001]
    // At position 1 freqs equal inv_freq, so we can verify each cos/sin value.
    #[test]
    fn test_inv_freq_values() {
        let rope = RotaryEmbedding::new(8, 2, 10000.0, &Device::Cpu)
            .expect("new failed");
        let (cos, sin) = rope.forward(1, 1).expect("forward failed");
        let cos_vals = rope_vec2(&cos);
        let sin_vals = rope_vec2(&sin);
        let expected_freqs = [1.0_f32, 0.1, 0.01, 0.001];
        for (i, &freq) in expected_freqs.iter().enumerate() {
            assert!(
                (cos_vals[0][i] - freq.cos()).abs() < 1e-5,
                "cos[{i}] at freq {freq}: got {}, expected {}",
                cos_vals[0][i],
                freq.cos()
            );
            assert!(
                (sin_vals[0][i] - freq.sin()).abs() < 1e-5,
                "sin[{i}] at freq {freq}: got {}, expected {}",
                sin_vals[0][i],
                freq.sin()
            );
        }
    }

    // Higher-index frequency pairs have smaller inv_freq values.  At position 1
    // all freqs are in (0, 1] ⊂ [0, π/2] where sin is monotonically increasing,
    // so the sin row at position 1 must be strictly decreasing across columns.
    #[test]
    fn test_inv_freq_monotonic_decay() {
        let rope = RotaryEmbedding::new(64, 4, 10000.0, &Device::Cpu)
            .expect("new failed");
        let (_, sin) = rope.forward(1, 1).expect("forward failed");
        let sin_row = rope_vec2(&sin);
        let vals = &sin_row[0];
        for i in 1..vals.len() {
            assert!(
                vals[i] < vals[i - 1],
                "sin not monotonically decreasing: vals[{i}]={} >= vals[{}]={}",
                vals[i],
                i - 1,
                vals[i - 1]
            );
        }
    }

    // dim=4, theta=100 => inv_freq = [1.0, 0.1]; verify cos/sin at several positions.
    #[test]
    fn test_table_values_at_specific_positions() {
        let rope = RotaryEmbedding::new(4, 16, 100.0, &Device::Cpu)
            .expect("new failed");
        for &pos in &[0usize, 1, 5, 10] {
            let (cos, sin) = rope.forward(pos, 1).expect("forward failed");
            let cos_vals = rope_vec2(&cos);
            let sin_vals = rope_vec2(&sin);
            let freqs = [pos as f32 * 1.0, pos as f32 * 0.1];
            for (i, &freq) in freqs.iter().enumerate() {
                assert!(
                    (cos_vals[0][i] - freq.cos()).abs() < 1e-5,
                    "pos={pos}, col={i}: cos got {}, expected {}",
                    cos_vals[0][i],
                    freq.cos()
                );
                assert!(
                    (sin_vals[0][i] - freq.sin()).abs() < 1e-5,
                    "pos={pos}, col={i}: sin got {}, expected {}",
                    sin_vals[0][i],
                    freq.sin()
                );
            }
        }
    }

    // Two embeddings with different theta must produce different cos/sin at a
    // non-zero position, and the higher-theta embedding should be closer to the
    // identity (cos≈1, sin≈0) because its frequencies are lower.
    #[test]
    fn test_different_theta_changes_frequencies() {
        let rope_a = RotaryEmbedding::new(64, 32, 10_000.0, &Device::Cpu)
            .expect("new (a) failed");
        let rope_b = RotaryEmbedding::new(64, 32, 1_000_000.0, &Device::Cpu)
            .expect("new (b) failed");
        let (cos_a, _) = rope_a.forward(10, 1).expect("forward (a) failed");
        let (cos_b, _) = rope_b.forward(10, 1).expect("forward (b) failed");
        assert!(
            max_abs_diff(&cos_a, &cos_b) > 0.01,
            "expected theta to affect cos values"
        );
        // Higher theta (rope_b) → smaller frequencies → cos closer to 1
        let ones = Tensor::ones_like(&cos_a).expect("ones_like failed");
        let dist_a = (&cos_a - &ones)
            .expect("sub a")
            .abs()
            .expect("abs a")
            .sum_all()
            .expect("sum_all a")
            .to_scalar::<f32>()
            .expect("scalar a");
        let dist_b = (&cos_b - &ones)
            .expect("sub b")
            .abs()
            .expect("abs b")
            .sum_all()
            .expect("sum_all b")
            .to_scalar::<f32>()
            .expect("scalar b");
        assert!(
            dist_b < dist_a,
            "higher theta should produce cos closer to 1: dist_a={dist_a}, dist_b={dist_b}"
        );
    }

    // Table width must equal dim/2 for each supported dim.
    #[test]
    fn test_dim_controls_table_width() {
        for dim in [8usize, 16, 32, 64, 128] {
            let rope = RotaryEmbedding::new(dim, 4, 10000.0, &Device::Cpu)
                .expect("new failed");
            let (cos, sin) = rope.forward(0, 4).expect("forward failed");
            assert_eq!(cos.dims(), &[4, dim / 2], "dim={dim}: cos width mismatch");
            assert_eq!(sin.dims(), &[4, dim / 2], "dim={dim}: sin width mismatch");
        }
    }

    // --- rotation correctness ---

    // apply() on a non-zero input at a non-zero position must change the values.
    #[test]
    fn test_apply_nonzero_changes_values() {
        let rope = RotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu)
            .expect("new failed");
        let q = Tensor::ones((1, 1, 1, 8), candle_core::DType::F32, &Device::Cpu)
            .expect("ones failed");
        let k = q.clone();
        let (q_rot, k_rot) = rope.apply(&q, &k, 5, 1).expect("apply failed");
        assert!(
            max_abs_diff(&q, &q_rot) > 0.01,
            "apply at pos 5 must change non-zero query"
        );
        assert!(
            max_abs_diff(&k, &k_rot) > 0.01,
            "apply at pos 5 must change non-zero key"
        );
    }

    // RoPE is a rotation so it preserves the total squared norm of the input.
    #[test]
    fn test_apply_preserves_norm() {
        let rope = RotaryEmbedding::new(64, 128, 10000.0, &Device::Cpu)
            .expect("new failed");
        let data: Vec<f32> = (0..1 * 2 * 8 * 64)
            .map(|i| (i as f32 + 1.0) * 0.001)
            .collect();
        let q = Tensor::from_vec(data, (1usize, 2, 8, 64), &Device::Cpu)
            .expect("from_vec failed");
        let k = q.clone();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0, 8).expect("apply failed");
        let sq_sum = |t: &Tensor| -> f32 {
            (t * t)
                .expect("mul")
                .sum_all()
                .expect("sum_all")
                .to_scalar::<f32>()
                .expect("to_scalar")
        };
        let q_sq = sq_sum(&q);
        let q_rot_sq = sq_sum(&q_rot);
        let rel_q = ((q_sq - q_rot_sq) / q_sq).abs();
        assert!(rel_q < 1e-3, "q norm not preserved: rel_diff={rel_q}");
        let k_sq = sq_sum(&k);
        let k_rot_sq = sq_sum(&k_rot);
        let rel_k = ((k_sq - k_rot_sq) / k_sq).abs();
        assert!(rel_k < 1e-3, "k norm not preserved: rel_diff={rel_k}");
    }

    // Identical input at different positions must produce different outputs.
    #[test]
    fn test_apply_position_matters() {
        let rope = RotaryEmbedding::new(8, 32, 10000.0, &Device::Cpu)
            .expect("new failed");
        let q = Tensor::ones((1, 1, 1, 8), candle_core::DType::F32, &Device::Cpu)
            .expect("ones failed");
        let k = q.clone();
        let (q0, _) = rope.apply(&q, &k, 0, 1).expect("apply pos 0");
        let (q5, _) = rope.apply(&q, &k, 5, 1).expect("apply pos 5");
        let (q10, _) = rope.apply(&q, &k, 10, 1).expect("apply pos 10");
        assert!(max_abs_diff(&q0, &q5) > 0.01, "pos 0 and 5 should differ");
        assert!(max_abs_diff(&q5, &q10) > 0.01, "pos 5 and 10 should differ");
        assert!(max_abs_diff(&q0, &q10) > 0.01, "pos 0 and 10 should differ");
    }

    // At position 0 all cos=1 and sin=0, so apply must be the identity.
    #[test]
    fn test_apply_at_position_zero_is_identity() {
        let rope = RotaryEmbedding::new(64, 16, 10000.0, &Device::Cpu)
            .expect("new failed");
        let data: Vec<f32> = (0..1 * 2 * 1 * 64)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let q = Tensor::from_vec(data.clone(), (1usize, 2, 1, 64), &Device::Cpu)
            .expect("from_vec q");
        let k = Tensor::from_vec(data, (1usize, 2, 1, 64), &Device::Cpu)
            .expect("from_vec k");
        let (q_rot, k_rot) = rope.apply(&q, &k, 0, 1).expect("apply failed");
        assert!(
            max_abs_diff(&q, &q_rot) < 1e-5,
            "apply at pos 0 must be identity for q"
        );
        assert!(
            max_abs_diff(&k, &k_rot) < 1e-5,
            "apply at pos 0 must be identity for k"
        );
    }

    // dim=4, theta=100 => inv_freq=[1.0, 0.1].  At position 3 the contiguous
    // half-split formula gives exact expected values we can compute by hand.
    #[test]
    fn test_apply_rotation_formula_manual() {
        let rope = RotaryEmbedding::new(4, 8, 100.0, &Device::Cpu)
            .expect("new failed");
        // q = [1.0, 2.0, 3.0, 4.0], shape [1, 1, 1, 4]
        let q = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)
            .expect("new tensor")
            .reshape((1usize, 1, 1, 4))
            .expect("reshape");
        let (q_rot, _) = rope.apply(&q, &q, 3, 1).expect("apply failed");
        let result = q_rot
            .squeeze(0)
            .expect("sq0")
            .squeeze(0)
            .expect("sq1")
            .squeeze(0)
            .expect("sq2")
            .to_vec1::<f32>()
            .expect("to_vec1");
        // freqs at pos 3: [3.0, 0.3]
        // out[0] = q[0]*cos(3) - q[2]*sin(3)
        // out[1] = q[1]*cos(0.3) - q[3]*sin(0.3)
        // out[2] = q[0]*sin(3) + q[2]*cos(3)
        // out[3] = q[1]*sin(0.3) + q[3]*cos(0.3)
        let (c0, s0) = (3.0_f32.cos(), 3.0_f32.sin());
        let (c1, s1) = (0.3_f32.cos(), 0.3_f32.sin());
        let expected = [
            1.0 * c0 - 3.0 * s0,
            2.0 * c1 - 4.0 * s1,
            1.0 * s0 + 3.0 * c0,
            2.0 * s1 + 4.0 * c1,
        ];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "result[{i}]: got {got}, expected {exp}"
            );
        }
    }

    // --- integration properties ---

    // Token-by-token application with advancing start_pos must match
    // full-sequence application.  This is the KV-cache decode invariant.
    #[test]
    fn test_sequential_vs_batch_equivalence() {
        let seq_len = 8usize;
        let rope = RotaryEmbedding::new(64, 128, 10000.0, &Device::Cpu)
            .expect("new failed");
        let q_data: Vec<f32> = (0..1 * 2 * seq_len * 64)
            .map(|i| (i as f32 + 1.0) * 0.001)
            .collect();
        let q = Tensor::from_vec(q_data, (1usize, 2, seq_len, 64), &Device::Cpu)
            .expect("from_vec q");
        let k = q.clone();
        let (q_batch, k_batch) = rope.apply(&q, &k, 0, seq_len).expect("batch apply");
        for t in 0..seq_len {
            let q_t = q.narrow(2, t, 1).expect("narrow q_t");
            let k_t = k.narrow(2, t, 1).expect("narrow k_t");
            let (q_t_rot, k_t_rot) = rope.apply(&q_t, &k_t, t, 1).expect("sequential apply");
            let q_batch_t = q_batch.narrow(2, t, 1).expect("narrow q_batch_t");
            let k_batch_t = k_batch.narrow(2, t, 1).expect("narrow k_batch_t");
            let q_diff = max_abs_diff(&q_batch_t, &q_t_rot);
            let k_diff = max_abs_diff(&k_batch_t, &k_t_rot);
            assert!(q_diff < 1e-5, "token {t}: q sequential/batch mismatch: {q_diff}");
            assert!(k_diff < 1e-5, "token {t}: k sequential/batch mismatch: {k_diff}");
        }
    }

    // Identical sequences in different batch slots must receive identical embeddings.
    #[test]
    fn test_multi_batch_consistency() {
        let rope = RotaryEmbedding::new(64, 64, 10000.0, &Device::Cpu)
            .expect("new failed");
        let single_data: Vec<f32> = (0..1 * 2 * 4 * 64)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let single = Tensor::from_vec(single_data, (1usize, 2, 4, 64), &Device::Cpu)
            .expect("from_vec");
        let batched = Tensor::cat(&[&single, &single, &single], 0).expect("cat");
        let (q_rot, _) = rope.apply(&batched, &batched, 0, 4).expect("apply");
        let b0 = q_rot.narrow(0, 0, 1).expect("narrow b0");
        let b1 = q_rot.narrow(0, 1, 1).expect("narrow b1");
        let b2 = q_rot.narrow(0, 2, 1).expect("narrow b2");
        assert!(max_abs_diff(&b0, &b1) < 1e-6, "batch 0 and 1 differ");
        assert!(max_abs_diff(&b0, &b2) < 1e-6, "batch 0 and 2 differ");
    }

    // GQA: q and k may have different head counts; shapes must be preserved.
    #[test]
    fn test_apply_different_q_k_heads() {
        let rope = RotaryEmbedding::new(64, 32, 10000.0, &Device::Cpu)
            .expect("new failed");
        let q_data: Vec<f32> = (0..1 * 8 * 4 * 64)
            .map(|i| (i as f32 + 1.0) * 0.001)
            .collect();
        let k_data: Vec<f32> = (0..1 * 2 * 4 * 64)
            .map(|i| (i as f32 + 1.0) * 0.001)
            .collect();
        let q = Tensor::from_vec(q_data, (1usize, 8, 4, 64), &Device::Cpu)
            .expect("from_vec q");
        let k = Tensor::from_vec(k_data, (1usize, 2, 4, 64), &Device::Cpu)
            .expect("from_vec k");
        let (q_rot, k_rot) = rope.apply(&q, &k, 0, 4).expect("apply");
        assert_eq!(q_rot.dims(), &[1, 8, 4, 64]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 64]);
        let sq_sum = |t: &Tensor| -> f32 {
            (t * t)
                .expect("mul")
                .sum_all()
                .expect("sum_all")
                .to_scalar::<f32>()
                .expect("to_scalar")
        };
        let q_sq = sq_sum(&q);
        let q_rot_sq = sq_sum(&q_rot);
        assert!(
            ((q_sq - q_rot_sq) / q_sq).abs() < 1e-3,
            "q norm not preserved in GQA"
        );
    }

    // Both usage patterns (apply() vs forward() + manual rope()) must agree.
    // Models like Qwen2.5 use the latter; SpeechTokenizer uses the former.
    #[test]
    fn test_forward_then_manual_rope_matches_apply() {
        let rope_emb = RotaryEmbedding::new(64, 64, 10000.0, &Device::Cpu)
            .expect("new failed");
        let q_data: Vec<f32> = (0..1 * 4 * 8 * 64)
            .map(|i| (i as f32 + 1.0) * 0.001)
            .collect();
        let q = Tensor::from_vec(q_data, (1usize, 4, 8, 64), &Device::Cpu)
            .expect("from_vec q");
        let k = q.clone();
        let (q1, k1) = rope_emb.apply(&q, &k, 5, 8).expect("apply");
        let (cos, sin) = rope_emb.forward(5, 8).expect("forward");
        let cos = cos.to_dtype(q.dtype()).expect("to_dtype cos");
        let sin = sin.to_dtype(q.dtype()).expect("to_dtype sin");
        let q2 = rope(&q.contiguous().expect("contiguous q"), &cos, &sin).expect("rope q");
        let k2 = rope(&k.contiguous().expect("contiguous k"), &cos, &sin).expect("rope k");
        assert!(
            max_abs_diff(&q1, &q2) < 1e-6,
            "q: apply vs forward+rope mismatch"
        );
        assert!(
            max_abs_diff(&k1, &k2) < 1e-6,
            "k: apply vs forward+rope mismatch"
        );
    }

    // --- edge cases ---

    // dim=2 is the smallest valid dimension (one frequency pair).
    #[test]
    fn test_dim_2_minimal() {
        let rope = RotaryEmbedding::new(2, 4, 10000.0, &Device::Cpu)
            .expect("new with dim=2 failed");
        let (cos, sin) = rope.forward(0, 4).expect("forward failed");
        assert_eq!(cos.dims(), &[4, 1]);
        assert_eq!(sin.dims(), &[4, 1]);
        let q = Tensor::ones((1, 1, 2, 2), candle_core::DType::F32, &Device::Cpu)
            .expect("ones failed");
        let (q_rot, _) = rope.apply(&q, &q, 0, 2).expect("apply failed");
        assert_eq!(q_rot.dims(), &[1, 1, 2, 2]);
    }

    // forward() must return Err when start_pos + seq_len > max_pos.
    #[test]
    fn test_forward_out_of_bounds_errors() {
        let rope = RotaryEmbedding::new(8, 10, 10000.0, &Device::Cpu)
            .expect("new failed");
        assert!(rope.forward(8, 3).is_err(), "8+3=11 > 10 should error");
        assert!(rope.forward(10, 1).is_err(), "10+1=11 > 10 should error");
        assert!(rope.forward(0, 11).is_err(), "0+11=11 > 10 should error");
        assert!(rope.forward(9, 1).is_ok(), "9+1=10 should succeed");
        assert!(rope.forward(0, 10).is_ok(), "0+10=max_pos should succeed");
    }

    // --- helpers ---

    fn rope_vec2(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_vec2::<f32>().expect("to_vec2 conversion failed")
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .expect("sub failed")
            .abs()
            .expect("abs failed")
            .max_all()
            .expect("max_all failed")
            .to_scalar::<f32>()
            .expect("to_scalar failed")
    }
}
