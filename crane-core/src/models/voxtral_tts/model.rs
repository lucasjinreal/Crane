//! Voxtral TTS model configuration, tokenizer integration, and voice embedding loader.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use tekken::Tekkenizer;

// ── Token IDs ──────────────────────────────────────────────────────────────

/// BOS token (id=1, `<s>`).
pub const BOS: u32 = 1;
/// Audio token placeholder fed as LLM input at each decoding step (id=24).
pub const AUDIO_TOKEN: u32 = 24;
/// Marks the start of an audio section (id=25).
pub const BEGIN_AUDIO: u32 = 25;
/// `[INST]` instruction start marker (id=35).
pub const INST: u32 = 35;
/// `[/INST]` instruction end marker (id=36).
pub const INST_END: u32 = 36;

/// End-of-audio semantic code (special value returned by the semantic head to
/// signal that generation is complete).
pub const END_AUDIO_CODE: u32 = 1;

// ── Config types ───────────────────────────────────────────────────────────

/// Top-level Voxtral TTS configuration parsed from `params.json`.
#[derive(Debug, Deserialize)]
pub struct VoxtralConfig {
    /// Hidden dimension of the LLM backbone (3072).
    pub dim: usize,
    /// Number of transformer layers in the LLM backbone (26).
    pub n_layers: usize,
    /// Attention head dimension (128).
    pub head_dim: usize,
    /// `SwiGLU` FFN intermediate dimension (9216).
    pub hidden_dim: usize,
    /// Number of query heads (32).
    pub n_heads: usize,
    /// Number of key-value heads (GQA, 8).
    pub n_kv_heads: usize,
    /// `RoPE` base frequency for the LLM backbone (1 000 000.0).
    pub rope_theta: f64,
    /// `RMSNorm` epsilon (1e-5).
    pub norm_eps: f64,
    /// Tekken vocabulary size (131 072).
    pub vocab_size: usize,
    /// Whether the token embedding and output projection share weights.
    pub tied_embeddings: bool,
    /// Multimodal (audio) configuration.
    pub multimodal: MultimodalConfig,
}

/// Multimodal section of `params.json`.
#[derive(Debug, Deserialize)]
pub struct MultimodalConfig {
    /// BOS token id (1).
    pub bos_token_id: u32,
    /// Audio model (LLM + acoustic transformer) configuration.
    pub audio_model_args: AudioModelArgs,
    /// Audio codec (encoder/decoder) configuration.
    pub audio_tokenizer_args: AudioTokenizerArgs,
}

/// Configuration for the LLM backbone's audio head and the acoustic transformer.
#[derive(Debug, Deserialize)]
pub struct AudioModelArgs {
    /// Number of VQ codes in the semantic codebook (8192).
    pub semantic_codebook_size: usize,
    /// Number of FSQ levels for acoustic codes (21).
    pub acoustic_codebook_size: usize,
    /// Number of acoustic codebooks (36).
    pub n_acoustic_codebook: usize,
    /// Token id used as the audio placeholder token in the LLM input (24).
    pub audio_token_id: u32,
    /// Token id that marks the beginning of an audio section (25).
    pub begin_audio_token_id: u32,
    /// Flow-matching acoustic transformer configuration.
    pub acoustic_transformer_args: AcousticTransformerArgs,
}

/// Configuration for the 3-layer flow-matching acoustic transformer.
#[derive(Debug, Deserialize)]
pub struct AcousticTransformerArgs {
    /// Hidden dimension (3072).
    pub dim: usize,
    /// Number of transformer layers (3).
    pub n_layers: usize,
    /// Attention head dimension (128).
    pub head_dim: usize,
    /// `SwiGLU` FFN intermediate dimension (9216).
    pub hidden_dim: usize,
    /// Number of query heads (32).
    pub n_heads: usize,
    /// Number of key-value heads (8).
    pub n_kv_heads: usize,
    /// `RoPE` base frequency — present in `params.json` but unused (acoustic
    /// transformer does not use `RoPE`).
    #[serde(default)]
    pub rope_theta: f64,
}

/// Configuration for the audio codec decoder.
#[derive(Debug, Deserialize)]
pub struct AudioTokenizerArgs {
    /// Output sample rate in Hz (24 000).
    pub sampling_rate: usize,
    /// Number of audio samples per codec frame (240).
    pub pretransform_patch_size: usize,
    /// Number of entries in the semantic VQ codebook (8192).
    pub semantic_codebook_size: usize,
    /// Dimension of each semantic codebook embedding (256).
    pub semantic_dim: usize,
    /// Number of FSQ levels for acoustic codes (21).
    pub acoustic_codebook_size: usize,
    /// Number of acoustic FSQ dimensions (36).
    pub acoustic_dim: usize,
    /// Codec transformer hidden dimension (1024).
    pub dim: usize,
    /// Codec `SwiGLU` FFN intermediate dimension (4096).
    pub hidden_dim: usize,
    /// Codec attention head dimension (128).
    pub head_dim: usize,
    /// Number of query heads in the codec transformer (8).
    pub n_heads: usize,
    /// Number of key-value heads in the codec transformer (8).
    pub n_kv_heads: usize,
    /// QK norm epsilon (1e-6).
    pub qk_norm_eps: f64,
    /// `RMSNorm` epsilon for the codec transformer (0.01).
    pub norm_eps: f64,
    /// Initial value for layer scale parameters (0.01).
    pub layer_scale_init: f64,
    /// Transformer layer counts per codec stage (from `"2,2,2,2"` in `params.json`).
    #[serde(
        rename = "decoder_transformer_lengths_str",
        deserialize_with = "deserialize_comma_separated"
    )]
    pub decoder_transformer_lengths: Vec<usize>,
    /// Convolution kernel sizes per codec stage (from `"3,4,4,4"` in `params.json`).
    #[serde(
        rename = "decoder_convs_kernels_str",
        deserialize_with = "deserialize_comma_separated"
    )]
    pub decoder_convs_kernels: Vec<usize>,
    /// Convolution strides per codec stage (from `"1,2,2,2"` in `params.json`).
    #[serde(
        rename = "decoder_convs_strides_str",
        deserialize_with = "deserialize_comma_separated"
    )]
    pub decoder_convs_strides: Vec<usize>,
    /// Map of voice name to integer voice ID.
    pub voice: HashMap<String, usize>,
}

fn deserialize_comma_separated<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

impl VoxtralConfig {
    /// Load configuration from the `params.json` file in `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is malformed.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("params.json");
        let data =
            std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&data).with_context(|| format!("failed to parse {}", path.display()))
    }
}

// ── Prompt construction ────────────────────────────────────────────────────

/// A segment in the Voxtral TTS prompt embedding sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptSegment {
    /// A single token to be embedded via the LLM token embedding table.
    Token(u32),
    /// Placeholder for the pre-computed voice embedding tensor `[N, 3072]`.
    VoiceEmbeddings,
}

/// Build the ordered sequence of prompt segments for Voxtral TTS.
///
/// The prompt layout (per the reference implementation) is:
/// ```text
/// [BOS=1] [BEGIN_AUDIO=25] <voice_embeddings> [INST_END=36]
///   text_token_0 ... text_token_M [INST=35] [BEGIN_AUDIO=25]
/// ```
///
/// The returned `Vec` uses `PromptSegment::VoiceEmbeddings` as a placeholder;
/// callers replace it with the actual `[N, 3072]` BF16 tensor in Step 2.
///
/// # Note
///
/// `text_token_ids` must be encoded **without** BOS or EOS; this function
/// inserts its own `BOS` at the start of the sequence.
#[must_use]
pub fn build_prompt_segments(text_token_ids: &[u32]) -> Vec<PromptSegment> {
    let mut segments = Vec::with_capacity(6 + text_token_ids.len());
    segments.push(PromptSegment::Token(BOS));
    segments.push(PromptSegment::Token(BEGIN_AUDIO));
    segments.push(PromptSegment::VoiceEmbeddings);
    segments.push(PromptSegment::Token(INST_END));
    for &id in text_token_ids {
        segments.push(PromptSegment::Token(id));
    }
    segments.push(PromptSegment::Token(INST));
    segments.push(PromptSegment::Token(BEGIN_AUDIO));
    segments
}

// ── Voice embedding loader ─────────────────────────────────────────────────

/// Internal dimension of all Voxtral embeddings; must equal `VoxtralConfig::dim`.
const EMBED_DIM: usize = 3072;

/// Load a pre-computed voice embedding from a `PyTorch` `.pt` file.
///
/// The file is a `PyTorch` ZIP archive containing a single BF16 tensor of
/// shape `[N, 3072]`, where N is the number of reference audio frames for
/// the chosen voice.  The archive entry is tried at several conventional
/// `PyTorch` paths.  Returns the tensor on `device` in BF16.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the ZIP structure is invalid,
/// no known tensor entry is found, or tensor construction fails.
///
pub fn load_voice_embedding(path: &Path, device: &Device) -> Result<Tensor> {
    use std::io::Read as _;
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("failed to read ZIP in {}", path.display()))?;

    // PyTorch saves the raw tensor data at one of these paths inside the archive.
    let candidates = ["voice_embed/data/0", "archive/data/0", "data/0"];
    let mut bytes = Vec::new();
    let mut found = false;
    for &name in &candidates {
        if let Ok(mut entry) = archive.by_name(name) {
            entry
                .read_to_end(&mut bytes)
                .with_context(|| format!("failed to read '{}' from {}", name, path.display()))?;
            found = true;
            break;
        }
    }
    anyhow::ensure!(
        found,
        "could not find tensor data entry in {}; tried {:?}",
        path.display(),
        candidates
    );

    // Each BF16 element is 2 bytes; the tensor is [N, EMBED_DIM].
    anyhow::ensure!(
        bytes.len() % (EMBED_DIM * 2) == 0,
        "voice embedding size {} is not a multiple of {} (EMBED_DIM * 2 bytes) in {}",
        bytes.len(),
        EMBED_DIM * 2,
        path.display()
    );
    let n_frames = bytes.len() / (EMBED_DIM * 2);

    Tensor::from_raw_buffer(&bytes, DType::BF16, &[n_frames, EMBED_DIM], device).with_context(
        || {
            format!(
                "failed to create voice embedding tensor from {}",
                path.display()
            )
        },
    )
}

// ── Tekken tokenizer ───────────────────────────────────────────────────────

/// Load the Tekken tokenizer from `tekken.json` in `model_dir`.
///
/// # Errors
///
/// Returns an error if the file cannot be read or the tokenizer JSON is invalid.
pub fn load_tokenizer(model_dir: &Path) -> Result<Tekkenizer> {
    let path = model_dir.join("tekken.json");
    Tekkenizer::from_file(&path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load Tekken tokenizer from {}: {}",
            path.display(),
            e
        )
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Synthetic helpers ─────────────────────────────────────────────────

    fn minimal_params_json() -> &'static str {
        r#"{
            "dim": 3072, "n_layers": 26, "head_dim": 128, "hidden_dim": 9216,
            "n_heads": 32, "n_kv_heads": 8, "rope_theta": 1000000.0,
            "norm_eps": 1e-05, "vocab_size": 131072, "tied_embeddings": true,
            "multimodal": {
                "bos_token_id": 1,
                "audio_model_args": {
                    "semantic_codebook_size": 8192, "acoustic_codebook_size": 21,
                    "n_acoustic_codebook": 36, "audio_token_id": 24,
                    "begin_audio_token_id": 25,
                    "acoustic_transformer_args": {
                        "dim": 3072, "n_layers": 3, "head_dim": 128, "hidden_dim": 9216,
                        "n_heads": 32, "n_kv_heads": 8, "rope_theta": 10000.0
                    }
                },
                "audio_tokenizer_args": {
                    "sampling_rate": 24000, "pretransform_patch_size": 240,
                    "semantic_codebook_size": 8192, "semantic_dim": 256,
                    "acoustic_codebook_size": 21, "acoustic_dim": 36,
                    "dim": 1024, "hidden_dim": 4096, "head_dim": 128,
                    "n_heads": 8, "n_kv_heads": 8, "qk_norm_eps": 1e-06,
                    "norm_eps": 0.01, "layer_scale_init": 0.01,
                    "decoder_transformer_lengths_str": "2,2,2,2",
                    "decoder_convs_kernels_str": "3,4,4,4",
                    "decoder_convs_strides_str": "1,2,2,2",
                    "voice": {
                        "neutral_female": 0, "casual_male": 1, "ar_male": 2,
                        "de_male": 3, "de_female": 4, "es_male": 5,
                        "es_female": 6, "fr_male": 7, "fr_female": 8,
                        "it_male": 9, "it_female": 10, "nl_male": 11,
                        "nl_female": 12, "pt_male": 13, "pt_female": 14,
                        "hi_male": 15, "hi_female": 16, "cheerful_female": 17,
                        "casual_female": 18, "neutral_male": 19
                    }
                }
            }
        }"#
    }

    /// Build a minimal in-memory PyTorch `.pt` (ZIP) file containing a BF16
    /// tensor of shape `[n_frames, 3072]` at the path `voice_embed/data/0`.
    fn make_pt_bytes(n_frames: usize) -> Vec<u8> {
        use std::io::Write as _;
        let raw: Vec<u8> = vec![0u8; n_frames * EMBED_DIM * 2]; // all-zero BF16
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("voice_embed/data/0", opts).unwrap();
            zip.write_all(&raw).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    // ── Unit tests (no external files) ───────────────────────────────────

    #[test]
    fn test_config_parse() {
        let cfg: VoxtralConfig =
            serde_json::from_str(minimal_params_json()).expect("config should parse");
        assert_eq!(cfg.dim, 3072);
        assert_eq!(cfg.n_layers, 26);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.vocab_size, 131_072);
        assert!(cfg.tied_embeddings);
        assert_eq!(cfg.multimodal.audio_model_args.semantic_codebook_size, 8192);
        assert_eq!(cfg.multimodal.audio_model_args.n_acoustic_codebook, 36);
        assert_eq!(cfg.multimodal.audio_tokenizer_args.voice.len(), 20);
        assert_eq!(
            cfg.multimodal
                .audio_tokenizer_args
                .decoder_transformer_lengths,
            vec![2, 2, 2, 2]
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.decoder_convs_kernels,
            vec![3, 4, 4, 4]
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.decoder_convs_strides,
            vec![1, 2, 2, 2]
        );
    }

    #[test]
    fn test_voice_embedding_load() {
        let n_frames = 4usize;
        let pt_bytes = make_pt_bytes(n_frames);

        // Write to a temp file so load_voice_embedding can open it.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write as _;
        tmp.as_file().write_all(&pt_bytes).unwrap();

        let emb =
            load_voice_embedding(tmp.path(), &Device::Cpu).expect("voice embedding should load");
        assert_eq!(emb.dims(), &[n_frames, EMBED_DIM]);
        assert_eq!(emb.dtype(), DType::BF16);
    }

    #[test]
    fn test_prompt_segments() {
        let text_ids = [1000u32, 2000, 3000];
        let segs = build_prompt_segments(&text_ids);

        assert_eq!(segs[0], PromptSegment::Token(BOS));
        assert_eq!(segs[1], PromptSegment::Token(BEGIN_AUDIO));
        assert_eq!(segs[2], PromptSegment::VoiceEmbeddings);
        assert_eq!(segs[3], PromptSegment::Token(INST_END));
        assert_eq!(segs[4], PromptSegment::Token(1000));
        assert_eq!(segs[5], PromptSegment::Token(2000));
        assert_eq!(segs[6], PromptSegment::Token(3000));
        let n = segs.len();
        assert_eq!(segs[n - 2], PromptSegment::Token(INST));
        assert_eq!(segs[n - 1], PromptSegment::Token(BEGIN_AUDIO));
    }

    #[test]
    fn test_prompt_segments_empty() {
        let segs = build_prompt_segments(&[]);
        assert_eq!(segs.len(), 6);
        assert_eq!(segs[0], PromptSegment::Token(BOS));
        assert_eq!(segs[1], PromptSegment::Token(BEGIN_AUDIO));
        assert_eq!(segs[2], PromptSegment::VoiceEmbeddings);
        assert_eq!(segs[3], PromptSegment::Token(INST_END));
        assert_eq!(segs[4], PromptSegment::Token(INST));
        assert_eq!(segs[5], PromptSegment::Token(BEGIN_AUDIO));
    }

    #[test]
    fn test_load_tokenizer_missing_file() {
        let result = load_tokenizer(std::path::Path::new("/nonexistent/voxtral/path"));
        assert!(result.is_err());
    }

    // ── Integration tests (require local checkpoint, skipped in CI) ───────

    fn checkpoint_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .join("checkpoints/Voxtral-4B-TTS-2603");
        p.is_dir().then_some(p)
    }

    #[test]
    #[ignore = "requires local checkpoint at checkpoints/Voxtral-4B-TTS-2603"]
    fn test_config_parse_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let cfg = VoxtralConfig::from_model_dir(&dir).expect("config should parse");
        assert_eq!(cfg.dim, 3072);
        assert_eq!(cfg.multimodal.audio_tokenizer_args.voice.len(), 20);
    }

    #[test]
    #[ignore = "requires local checkpoint at checkpoints/Voxtral-4B-TTS-2603"]
    fn test_tekken_encode_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let tok = load_tokenizer(&dir).expect("tokenizer should load");
        let ids = tok
            .encode("Hello, world!", false, false)
            .expect("encode should succeed");
        assert!(!ids.is_empty());
        let vocab = tok.vocab_size();
        for &id in &ids {
            assert!((id as usize) < vocab);
        }
    }

    #[test]
    #[ignore = "requires local checkpoint at checkpoints/Voxtral-4B-TTS-2603"]
    fn test_voice_embedding_load_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let emb =
            load_voice_embedding(&dir.join("voice_embedding/neutral_female.pt"), &Device::Cpu)
                .expect("voice embedding should load");
        assert_eq!(emb.dims()[1], EMBED_DIM);
        assert_eq!(emb.dtype(), DType::BF16);
    }
}
