//! Dimension calculations and config trait for Gated Delta Net layers.

use serde::Deserialize;

/// Per-head dimensions derived from a GDN config.
///
/// `key_dim = num_k_heads * head_k_dim`, `value_dim = num_v_heads * head_v_dim`,
/// `conv_dim = 2 * key_dim + value_dim` (Q/K/V concatenated for the causal conv1d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnDims {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    /// Number of value heads per key head (GQA ratio).
    pub v_per_group: usize,
}

impl GdnDims {
    pub fn new(cfg: &dyn GdnConfig) -> Self {
        let hidden_size = cfg.hidden_size();
        let num_k_heads = cfg.linear_num_key_heads();
        let num_v_heads = cfg.linear_num_value_heads();
        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let conv_kernel_size = cfg.linear_conv_kernel_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let v_per_group = num_v_heads / num_k_heads;

        Self {
            hidden_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
            conv_dim,
            v_per_group,
        }
    }

    /// Output dimension of the QKV/Z fused projection (Q + K + V + Z concatenated).
    pub fn qkvz_out_dim(&self) -> usize {
        self.key_dim * 2 + self.value_dim * 2
    }

    /// Output dimension of the B/A fused projection (B + A concatenated).
    pub fn ba_out_dim(&self) -> usize {
        self.num_v_heads * 2
    }
}

/// Trait implemented by model configs that contain linear-attention parameters.
///
/// Default impls compute derived dims (`linear_key_dim`, `linear_value_dim`,
/// `linear_conv_dim`) from the per-head fields.
pub trait GdnConfig {
    fn hidden_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn linear_key_head_dim(&self) -> usize;
    fn linear_value_head_dim(&self) -> usize;
    fn linear_num_key_heads(&self) -> usize;
    fn linear_num_value_heads(&self) -> usize;

    fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads() * self.linear_key_head_dim()
    }

    fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim()
    }

    fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

/// Default values for HF `config.json` fields that are sometimes missing.
#[allow(dead_code)]
pub mod defaults {
    pub fn conv_kernel() -> usize {
        4
    }

    pub fn partial_rotary_factor() -> f64 {
        0.25
    }

    pub fn rope_theta() -> f64 {
        10_000_000.0
    }

    pub fn full_attention_interval() -> usize {
        4
    }
}

/// Marker type used by model configs that deserialize nested blocks with
/// default values (e.g. an empty `rope_parameters` wrapper).
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Empty {}