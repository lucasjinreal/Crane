// SPDX-License-Identifier: MIT

use candle_core::{Result, bail};

pub const TERNARY_BLOCK_SIZE: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TernaryEncoding {
    Pq2_0,
    Ptq1_0,
}

impl TernaryEncoding {
    #[must_use]
    pub const fn ggml_type_id(self) -> u32 {
        match self {
            Self::Pq2_0 => 142,
            Self::Ptq1_0 => 143,
        }
    }

    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Pq2_0 => 34,
            Self::Ptq1_0 => 28,
        }
    }

    #[must_use]
    pub fn from_ggml_type_id(id: u32) -> Option<Self> {
        match id {
            142 => Some(Self::Pq2_0),
            143 => Some(Self::Ptq1_0),
            _ => None,
        }
    }
}

fn f16_le(bytes: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

/// # Errors
///
/// Returns an error if `block`'s length does not match `encoding`'s expected
/// block size.
pub fn decode_block(encoding: TernaryEncoding, block: &[u8], dst: &mut [f32; 128]) -> Result<()> {
    if block.len() != encoding.block_bytes() {
        bail!(
            "invalid {:?} block length {}, expected {}",
            encoding,
            block.len(),
            encoding.block_bytes()
        )
    }
    match encoding {
        TernaryEncoding::Pq2_0 => {
            let d = f16_le(&block[..2]);
            for (i, out) in dst.iter_mut().enumerate() {
                let q = (block[2 + i / 4] >> (2 * (i % 4))) & 0x3;
                *out = (q as f32 - 1.0) * d;
            }
        },
        TernaryEncoding::Ptq1_0 => {
            let d = f16_le(&block[26..28]);
            for (i, out) in dst.iter_mut().enumerate() {
                let (mut q, n) = if i < 80 {
                    (block[i & 15], i >> 4)
                } else if i < 120 {
                    let t = i - 80;
                    (block[16 + (t & 7)], t >> 3)
                } else {
                    let t = i - 120;
                    (block[24 + (t & 1)], t >> 1)
                };
                for _ in 0..n {
                    q = q.wrapping_mul(3);
                }
                let trit = ((u16::from(q) * 3) >> 8) as i16 - 1;
                *out = trit as f32 * d;
            }
        },
    }
    Ok(())
}

pub(crate) fn dot_row(
    encoding: TernaryEncoding,
    packed: &[u8],
    row: usize,
    cols: usize,
    x: &[f32],
) -> Result<f32> {
    if cols % TERNARY_BLOCK_SIZE != 0 || x.len() != cols {
        bail!("ternary dot expects a row width divisible by 128")
    }
    let blocks_per_row = cols / TERNARY_BLOCK_SIZE;
    let row_bytes = blocks_per_row * encoding.block_bytes();
    let start = row
        .checked_mul(row_bytes)
        .ok_or_else(|| candle_core::Error::Msg("ternary row offset overflow".into()))?;
    let row_data = packed
        .get(start..start + row_bytes)
        .ok_or_else(|| candle_core::Error::Msg("ternary row is outside packed storage".into()))?;
    let mut values = [0f32; TERNARY_BLOCK_SIZE];
    let mut sum = 0f32;
    for block_idx in 0..blocks_per_row {
        let off = block_idx * encoding.block_bytes();
        decode_block(
            encoding,
            &row_data[off..off + encoding.block_bytes()],
            &mut values,
        )?;
        let x = &x[block_idx * TERNARY_BLOCK_SIZE..(block_idx + 1) * TERNARY_BLOCK_SIZE];
        sum += values.iter().zip(x).map(|(w, v)| w * v).sum::<f32>();
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq2_zero_code_decodes_to_negative_scale() -> Result<()> {
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        let mut out = [0f32; 128];
        decode_block(TernaryEncoding::Pq2_0, &block, &mut out)?;
        assert!(out.iter().all(|&v| v == -2.0));
        Ok(())
    }

    #[test]
    fn ptq1_reference_zero_trits() -> Result<()> {
        let mut block = [0u8; 28];
        block[..26].fill(128);
        block[26..].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let mut out = [0f32; 128];
        decode_block(TernaryEncoding::Ptq1_0, &block, &mut out)?;
        assert!(out.iter().all(|&v| v == 0.0));
        Ok(())
    }
}
