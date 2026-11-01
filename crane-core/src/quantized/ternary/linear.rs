// SPDX-License-Identifier: MIT

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, bail};

#[cfg(not(feature = "rocm"))]
use crate::utils::DeviceExt;

use super::codec::{TernaryEncoding, dot_row};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadamardMode {
    None,
    Forward,
    Inverse,
}

/// Feature-axis permutation used by Qwen3.5 GDN output projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnPermutation {
    pub head_dim: usize,
    pub key_heads: usize,
    pub repeats: usize,
}

#[derive(Clone)]
pub struct TernaryWeight {
    pub(crate) encoding: TernaryEncoding,
    pub(crate) packed_cpu: Arc<Vec<u8>>,
    #[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
    pub(crate) packed_device: Option<Tensor>,
    device: Device,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl TernaryWeight {
    /// # Errors
    ///
    /// Returns an error if `cols` is not a multiple of 128, if `rows * cols`
    /// overflows, if `packed`'s length does not match the size implied by
    /// `rows`/`cols`/`encoding`, or if allocating the device copy fails.
    pub fn new(
        encoding: TernaryEncoding,
        packed: Vec<u8>,
        rows: usize,
        cols: usize,
        device: &Device,
    ) -> Result<Self> {
        if cols % 128 != 0 {
            bail!("ternary weight columns must be divisible by 128, got {cols}")
        }
        let expected = rows
            .checked_mul(cols / 128)
            .and_then(|n| n.checked_mul(encoding.block_bytes()))
            .ok_or_else(|| candle_core::Error::Msg("ternary weight size overflow".into()))?;
        if packed.len() != expected {
            bail!(
                "invalid {:?} tensor length {}, expected {expected} for [{rows}, {cols}]",
                encoding,
                packed.len()
            )
        }
        // Keep a single host copy on CPU. CUDA needs its own device allocation, but
        // cloning multi-gigabyte packed weights into a second CPU tensor would double
        // the model's resident memory for no benefit.
        let packed_device = if device.is_cuda() || device.is_rocm() {
            Some(Tensor::from_vec(packed.clone(), (expected,), device)?)
        } else {
            None
        };
        Ok(Self {
            encoding,
            packed_cpu: Arc::new(packed),
            packed_device,
            device: device.clone(),
            rows,
            cols,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn size_in_bytes(&self) -> usize {
        self.packed_cpu.len()
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// # Errors
    ///
    /// Returns an error if `ids` contains a value outside the vocabulary
    /// (`>= rows`), if decoding a packed ternary block fails, or if the
    /// output tensor cannot be built or cast to `dtype`.
    pub fn embedding(
        &self,
        ids: &Tensor,
        dtype: DType,
        signs: &[f32],
        block_size: usize,
    ) -> Result<Tensor> {
        let ids_cpu = ids
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let mut output = Vec::with_capacity(ids_cpu.len() * self.cols);
        let mut row = vec![0f32; self.cols];
        for id in ids_cpu {
            let id = id as usize;
            if id >= self.rows {
                bail!("embedding id {id} is outside vocabulary {}", self.rows)
            }
            for block_idx in 0..self.cols / 128 {
                let row_bytes = (self.cols / 128) * self.encoding.block_bytes();
                let off = id * row_bytes + block_idx * self.encoding.block_bytes();
                let mut values = [0f32; 128];
                super::codec::decode_block(
                    self.encoding,
                    &self.packed_cpu[off..off + self.encoding.block_bytes()],
                    &mut values,
                )?;
                row[block_idx * 128..(block_idx + 1) * 128].copy_from_slice(&values);
            }
            fwht_blocks(&mut row, block_size);
            for (value, sign) in row.iter_mut().zip(signs) {
                *value *= sign;
            }
            output.extend_from_slice(&row);
        }
        let mut dims = ids.dims().to_vec();
        dims.push(self.cols);
        Tensor::from_vec(output, dims, ids.device())?.to_dtype(dtype)
    }
}

#[derive(Clone)]
pub struct TernaryLinear {
    weight: Arc<TernaryWeight>,
    signs: Arc<Vec<f32>>,
    #[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
    signs_device: Tensor,
    block_size: usize,
    mode: HadamardMode,
    gdn_permutation: Option<GdnPermutation>,
}

impl TernaryLinear {
    /// # Errors
    ///
    /// Returns an error if `mode` requires Hadamard transform parameters
    /// (`block_size`, `signs`) that are invalid or mismatched with the
    /// weight's width, if `gdn_permutation`'s geometry does not match the
    /// weight's width, or if allocating the signs tensor on the weight's
    /// device fails.
    pub fn new(
        weight: Arc<TernaryWeight>,
        signs: Arc<Vec<f32>>,
        block_size: usize,
        mode: HadamardMode,
        gdn_permutation: Option<GdnPermutation>,
    ) -> Result<Self> {
        if mode != HadamardMode::None {
            if block_size == 0 || !block_size.is_power_of_two() || weight.cols % block_size != 0 {
                bail!(
                    "invalid Hadamard block size {block_size} for width {}",
                    weight.cols
                )
            }
            if signs.len() != weight.cols {
                bail!(
                    "Hadamard signs length {} does not match width {}",
                    signs.len(),
                    weight.cols
                )
            }
        }
        if let Some(perm) = gdn_permutation
            && perm.head_dim * perm.key_heads * perm.repeats != weight.cols
        {
            bail!("invalid GDN permutation geometry for width {}", weight.cols)
        }
        let signs_device =
            Tensor::from_vec(signs.as_ref().clone(), (signs.len(),), &weight.device)?;
        Ok(Self {
            weight,
            signs,
            signs_device,
            block_size,
            mode,
            gdn_permutation,
        })
    }

    pub fn weight(&self) -> &Arc<TernaryWeight> {
        &self.weight
    }

    fn transform_forward(&self, values: &mut [f32]) {
        if self.mode == HadamardMode::None {
            return;
        }
        if let Some(perm) = self.gdn_permutation {
            permute_gdn(values, perm);
        }
        for (value, sign) in values.iter_mut().zip(self.signs.iter()) {
            *value *= sign;
        }
        fwht_blocks(values, self.block_size);
    }

    /// # Errors
    ///
    /// Returns an error if `xs`'s last dimension does not match the weight's
    /// column count, if `xs` is on an unsupported device, if a CUDA/`ROCm`
    /// weight has no device allocation, or if the underlying dot-product
    /// computation fails.
    pub fn forward_f32(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32)?.contiguous()?;
        let dims = xs.dims();
        let last = *dims
            .last()
            .ok_or_else(|| candle_core::Error::Msg("ternary linear expects rank >= 1".into()))?;
        if last != self.weight.cols {
            bail!(
                "ternary linear input width {last} does not match {}",
                self.weight.cols
            )
        }
        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() {
            let packed_device = self.weight.packed_device.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("ternary CUDA weight has no device allocation".into())
            })?;
            return super::super::super::ops::quant_ternary::cuda::linear_f32(
                &xs,
                packed_device,
                self.weight.encoding,
                self.weight.rows,
                self.weight.cols,
                &self.signs_device,
                self.block_size,
                self.mode,
                self.gdn_permutation,
            );
        }
        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        if xs.device().is_rocm() {
            let packed_device = self.weight.packed_device.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("ternary ROCm weight has no device allocation".into())
            })?;
            return crate::ops::quant_ternary::rocm::linear_f32(
                &xs,
                packed_device,
                self.weight.encoding,
                self.weight.rows,
                self.weight.cols,
                &self.signs_device,
                self.block_size,
                self.mode,
                self.gdn_permutation,
            );
        }
        if !xs.device().is_cpu() {
            bail!("TernaryLinear currently supports CPU, CUDA, and ROCm")
        }
        let rows = xs.elem_count() / last;
        let input = xs.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0f32; rows * self.weight.rows];
        for row in 0..rows {
            let mut x = input[row * last..(row + 1) * last].to_vec();
            self.transform_forward(&mut x);
            parallel_dot_rows(
                &mut output[row * self.weight.rows..(row + 1) * self.weight.rows],
                &self.weight,
                &x,
            )?;
        }
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = self.weight.rows;
        Tensor::from_vec(output, out_dims, xs.device())
    }

    /// # Errors
    ///
    /// Returns an error if [`Self::forward_f32`] fails or the result cannot
    /// be cast back to `xs`'s dtype.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let out = self.forward_f32(xs)?;
        if dtype == DType::F32 {
            Ok(out)
        } else {
            out.to_dtype(dtype)
        }
    }
}

fn parallel_dot_rows(output: &mut [f32], weight: &TernaryWeight, input: &[f32]) -> Result<()> {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(output.len().max(1));
    let chunk_size = output.len().div_ceil(threads);
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(threads);
        for (chunk_idx, chunk) in output.chunks_mut(chunk_size).enumerate() {
            handles.push(scope.spawn(move || -> Result<()> {
                let start = chunk_idx * chunk_size;
                for (index, value) in chunk.iter_mut().enumerate() {
                    *value = dot_row(
                        weight.encoding,
                        &weight.packed_cpu,
                        start + index,
                        weight.cols,
                        input,
                    )?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| candle_core::Error::Msg("ternary CPU worker panicked".into()))??;
        }
        Ok(())
    })
}

fn permute_gdn(values: &mut [f32], perm: GdnPermutation) {
    let source = values.to_vec();
    for key in 0..perm.key_heads {
        for repeat in 0..perm.repeats {
            for dim in 0..perm.head_dim {
                let src = dim + perm.head_dim * (key + perm.key_heads * repeat);
                let dst = dim + perm.head_dim * (repeat + perm.repeats * key);
                values[dst] = source[src];
            }
        }
    }
}

pub(crate) fn fwht_blocks(values: &mut [f32], block_size: usize) {
    let scale = 1.0 / (block_size as f32).sqrt();
    for block in values.chunks_exact_mut(block_size) {
        for value in block.iter_mut() {
            *value *= scale;
        }
        let mut len = 1;
        while len < block_size {
            for base in (0..block_size).step_by(2 * len) {
                for j in 0..len {
                    let x = block[base + j];
                    let y = block[base + len + j];
                    block[base + j] = x + y;
                    block[base + len + j] = x - y;
                }
            }
            len *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GdnPermutation, permute_gdn};
    #[cfg(feature = "cuda")]
    use super::{HadamardMode, TernaryLinear, TernaryWeight};
    #[cfg(feature = "cuda")]
    use crate::quantized::ternary::TernaryEncoding;
    #[cfg(feature = "cuda")]
    use candle_core::{Device, Result, Tensor};
    #[cfg(feature = "cuda")]
    use std::sync::Arc;

    #[test]
    fn gdn_permutation_matches_grouped_layout() {
        let mut values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        permute_gdn(
            &mut values,
            GdnPermutation {
                head_dim: 2,
                key_heads: 2,
                repeats: 3,
            },
        );
        assert_eq!(values, [0., 1., 4., 5., 8., 9., 2., 3., 6., 7., 10., 11.]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu_for_both_ternary_encodings() -> Result<()> {
        if !candle_core::utils::cuda_is_available() {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        for encoding in [TernaryEncoding::Ptq1_0, TernaryEncoding::Pq2_0] {
            let rows = 7;
            let cols = 128;
            let mut packed = vec![0u8; rows * encoding.block_bytes()];
            let scale = half::f16::from_f32(0.25).to_bits().to_le_bytes();
            for (block_index, block) in packed.chunks_exact_mut(encoding.block_bytes()).enumerate()
            {
                match encoding {
                    TernaryEncoding::Pq2_0 => {
                        block[..2].copy_from_slice(&scale);
                        block[2..].fill((block_index as u8).wrapping_mul(37));
                    },
                    TernaryEncoding::Ptq1_0 => {
                        block[..26].fill(97u8.wrapping_add(block_index as u8 * 11));
                        block[26..].copy_from_slice(&scale);
                    },
                }
            }
            let signs = Arc::new(
                (0..cols)
                    .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
            );
            let input_rows = 5;
            let input = (0..input_rows * cols)
                .map(|index| (index as f32 * 0.03125).sin())
                .collect::<Vec<_>>();
            let cpu = TernaryLinear::new(
                Arc::new(TernaryWeight::new(
                    encoding,
                    packed.clone(),
                    rows,
                    cols,
                    &Device::Cpu,
                )?),
                signs.clone(),
                128,
                HadamardMode::Forward,
                None,
            )?;
            let gpu = TernaryLinear::new(
                Arc::new(TernaryWeight::new(encoding, packed, rows, cols, &cuda)?),
                signs,
                128,
                HadamardMode::Forward,
                None,
            )?;
            let expected = cpu
                .forward_f32(&Tensor::from_vec(
                    input.clone(),
                    (input_rows, cols),
                    &Device::Cpu,
                )?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = gpu
                .forward_f32(&Tensor::from_vec(input, (input_rows, cols), &cuda)?)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_error = expected
                .iter()
                .zip(actual.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_error < 1e-3, "{encoding:?} max error {max_error}");
        }
        Ok(())
    }
}
