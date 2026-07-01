use thiserror::Error;

/// Result type for Crane SDK operations
pub type CraneResult<T> = Result<T, CraneError>;

/// Error type for Crane SDK operations
#[derive(Error, Debug)]
pub enum CraneError {
    /// Error from the underlying model
    #[error("Model error: {0}")]
    ModelError(String),

    /// Error from tokenization
    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Anyhow error
    #[error("Anyhow error: {0}")]
    AnyhowError(#[from] anyhow::Error),

    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}
