use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use ribo::utils::log::tracing;

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    /// # Errors
    ///
    /// Returns an error if the embedding weight can't be loaded from `vb`
    /// (missing tensor or shape mismatch).
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    /// # Errors
    ///
    /// Returns an error if `weights` doesn't have exactly 2 dimensions.
    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = candle_nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    #[must_use]
    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }

    #[must_use]
    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }

    /// Returns the layer's bias tensor, if any.
    #[must_use]
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias()
    }
}

/// # Errors
///
/// Returns an error if the layer's weight (or bias, when `b` is true) can't
/// be loaded from `vb`.
pub fn linear_b(d1: usize, d2: usize, b: bool, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_b(d1, d2, b, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

/// # Errors
///
/// Returns an error if the layer's weight or bias can't be loaded from `vb`.
pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

/// # Errors
///
/// Returns an error if the layer's weight can't be loaded from `vb`.
pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    span: tracing::Span,
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

/// # Errors
///
/// Returns an error if the convolution's weight or bias can't be loaded
/// from `vs`.
pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    /// # Errors
    ///
    /// Returns an error if the quantized weight can't be loaded from `vb`,
    /// or if it isn't a quantized dtype `QMatMul` supports.
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: &crate::utils::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// # Errors
    ///
    /// Returns an error if `ws` isn't a quantized dtype `QMatMul` supports.
    pub fn from_weights(ws: std::sync::Arc<candle_core::quantized::QTensor>) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let inner = candle_nn::LayerNorm::new(weight, bias, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self { inner, span }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// # Errors
///
/// Returns an error if the layer's weight or bias can't be loaded from `vb`.
pub fn layer_norm<C: Into<candle_nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = candle_nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    /// # Errors
    ///
    /// Returns an error if the weight tensor can't be loaded from `vb`.
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    /// # Errors
    ///
    /// Returns an error if `x`'s last dimension doesn't match the norm's size.
    pub fn forward_diff(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward_diff(x)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}
