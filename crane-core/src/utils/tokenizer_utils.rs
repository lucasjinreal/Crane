use ahash::AHashMap;
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Content;
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::{AddedToken, Tokenizer};

#[derive(Debug, serde::Deserialize)]
struct TokenObj {
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum MaybeToken {
    Str(String),
    Obj(TokenObj),
}

#[derive(Debug, serde::Deserialize, Default)]
struct TokenizerConfigLike {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, TokenObj>,
    #[serde(default)]
    bos_token: Option<MaybeToken>,
    #[serde(default)]
    eos_token: Option<MaybeToken>,
    #[serde(default)]
    pad_token: Option<MaybeToken>,
    #[serde(default)]
    unk_token: Option<MaybeToken>,
}

fn as_added_token(tok: &TokenObj) -> AddedToken {
    AddedToken::from(tok.content.clone(), tok.special)
        .lstrip(tok.lstrip)
        .rstrip(tok.rstrip)
        .normalized(tok.normalized)
        .single_word(tok.single_word)
}

fn maybe_token_to_obj(tok: &MaybeToken) -> TokenObj {
    match tok {
        MaybeToken::Str(s) => TokenObj {
            content: s.clone(),
            lstrip: false,
            rstrip: false,
            normalized: true,
            single_word: false,
            special: true,
        },
        MaybeToken::Obj(o) => TokenObj {
            content: o.content.clone(),
            lstrip: o.lstrip,
            rstrip: o.rstrip,
            normalized: o.normalized,
            single_word: o.single_word,
            special: o.special,
        },
    }
}

fn load_hf_bpe_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    if !vocab.exists() || !merges.exists() {
        anyhow::bail!(
            "No supported tokenizer files in {}. Need tokenizer.json or vocab.json + merges.txt",
            dir.display()
        );
    }

    let vocab_str = vocab
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path: {}", vocab.display()))?;
    let merges_str = merges
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path: {}", merges.display()))?;

    let mut bpe_builder = BPE::from_file(vocab_str, merges_str);

    let tok_cfg_path = dir.join("tokenizer_config.json");
    if tok_cfg_path.exists() {
        let cfg_data = std::fs::read_to_string(&tok_cfg_path)
            .with_context(|| format!("Failed to read {}", tok_cfg_path.display()))?;
        if let Ok(cfg) = serde_json::from_str::<TokenizerConfigLike>(&cfg_data) {
            if let Some(unk) = cfg.unk_token {
                let unk_obj = maybe_token_to_obj(&unk);
                bpe_builder = bpe_builder.unk_token(unk_obj.content);
            }
        }
    }

    let model = bpe_builder
        .build()
        .map_err(anyhow::Error::msg)
        .context("Failed to build BPE tokenizer from vocab.json/merges.txt")?;
    let mut tokenizer = Tokenizer::new(model);

    tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::default()));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));

    // Load special/added tokens when available.
    if tok_cfg_path.exists() {
        let cfg_data = std::fs::read_to_string(&tok_cfg_path)
            .with_context(|| format!("Failed to read {}", tok_cfg_path.display()))?;
        if let Ok(cfg) = serde_json::from_str::<TokenizerConfigLike>(&cfg_data) {
            let mut tokens: Vec<AddedToken> = cfg
                .added_tokens_decoder
                .values()
                .map(as_added_token)
                .collect();

            for maybe in [cfg.bos_token, cfg.eos_token, cfg.pad_token, cfg.unk_token]
                .into_iter()
                .flatten()
            {
                let t = maybe_token_to_obj(&maybe);
                tokens.push(as_added_token(&t));
            }

            if !tokens.is_empty() {
                tokenizer
                    .add_special_tokens(tokens)
                    .map_err(anyhow::Error::msg)?;
            }
        }
    }

    Ok(tokenizer)
}

pub fn load_tokenizer_from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Tokenizer> {
    let model_dir = model_dir.as_ref();
    let tokenizer_json = model_dir.join("tokenizer.json");

    if tokenizer_json.exists() {
        return Tokenizer::from_file(&tokenizer_json)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("Failed to load {}", tokenizer_json.display()));
    }

    load_hf_bpe_tokenizer(model_dir)
}

// ─────────────────────────────────────────────────────────────
//  GGUF-embedded tokenizer
// ─────────────────────────────────────────────────────────────

/// True if `ct` carries enough metadata to build a tokenizer from the GGUF
/// itself (`tokenizer.ggml.tokens` + `tokenizer.ggml.merges`).
///
/// Newer llama.cpp quantizers always emit these; older / third-party ones may
/// not, in which case callers should fall back to a sibling `tokenizer.json`.
pub fn gguf_has_embedded_tokenizer(ct: &Content) -> bool {
    ct.metadata.contains_key("tokenizer.ggml.tokens")
        && ct.metadata.contains_key("tokenizer.ggml.merges")
}

/// Read `tokenizer.chat_template` (the Jinja string llama.cpp embeds alongside
/// the weights) if present.
pub fn extract_chat_template_from_gguf(ct: &Content) -> Option<String> {
    ct.metadata
        .get("tokenizer.chat_template")
        .and_then(|v| v.to_string().ok())
        .cloned()
}

/// Map llama.cpp `tokenizer.ggml.token_type` values to the categories the
/// `tokenizers` crate cares about.
///
/// ```text
/// 0 = UNDEFINED     -> normal vocab (no special handling)
/// 1 = NORMAL        -> normal vocab
/// 2 = UNKNOWN       -> normal vocab
/// 3 = CONTROL       -> special token (chat-template boundaries, etc.)
/// 4 = USER_DEFINED  -> added token (tool-calling, reasoning markers, …)
/// 5 = UNUSED        -> padding placeholders, ignore
/// 6 = BYTE          -> byte fallback (always present in GPT-2 byte-level BPE)
/// ```
fn is_gguf_added_token(token_type: i32) -> bool {
    matches!(token_type, 3 | 4)
}

/// Build a HF-compatible `tokenizers::Tokenizer` directly from the BPE
/// vocab, merges, and added/special token metadata embedded in a GGUF file.
///
/// This reproduces what llama.cpp ships in its tokenizer (`qwen2`,
/// `qwen35`, etc.): a GPT-2 byte-level BPE with `Ġ`-prefixed space tokens,
/// wired up with `ByteLevel` for both pre-tokenization and decoding.
///
/// Special tokens are detected two ways, in order:
/// 1. The optional `tokenizer.ggml.added_tokens` array (used by recent
///    llama.cpp builds) gives an explicit list with `special: true`.
/// 2. The per-id `tokenizer.ggml.token_type` array — tokens with type 3
///    (CONTROL) or 4 (USER_DEFINED) are registered as special. Type 5
///    placeholders (`[PADxxx]`) and type 6 byte fallbacks are skipped.
pub fn build_tokenizer_from_gguf(ct: &Content) -> Result<Tokenizer> {
    let tokens_arr = ct
        .metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF is missing `tokenizer.ggml.tokens`; cannot reconstruct the tokenizer"
            )
        })?
        .to_vec()
        .map_err(anyhow::Error::msg)?;
    let merges_arr = ct
        .metadata
        .get("tokenizer.ggml.merges")
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF is missing `tokenizer.ggml.merges`; cannot reconstruct the tokenizer"
            )
        })?
        .to_vec()
        .map_err(anyhow::Error::msg)?;

    let tokens: Vec<String> = tokens_arr
        .iter()
        .map(|v| v.to_string().map(|s| s.clone()).map_err(anyhow::Error::msg))
        .collect::<Result<_>>()?;

    // GGUF merges are stored as `"Ġ a"` style strings; the BPE model wants
    // `("Ġ", "a")` pairs (split on the first space).
    let merges: Vec<(String, String)> = merges_arr
        .iter()
        .map(|v| {
            let s = v.to_string().map_err(anyhow::Error::msg)?;
            let (l, r) = s
                .split_once(' ')
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GGUF merge entry {s:?} is not in `<left> <right>` form"
                    )
                })?;
            Ok((l.to_string(), r.to_string()))
        })
        .collect::<Result<_>>()?;

    // `BpeBuilder::vocab_and_merges` (and `BPE::new`) wants an `AHashMap`.
    let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }

    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(anyhow::Error::msg)
        .context("Failed to build BPE tokenizer from GGUF metadata")?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::default()));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));

    // Register special / added tokens. The HF tokenizer.json uses
    // `AddedToken { content, special: true, normalized: true }` by default;
    // `normalized: true` is what allows e.g. `<tool_call>` to be matched even
    // when the user types `< tool_call >` with extra spaces (the byte-level
    // pre-tokenizer is whitespace-sensitive).
    //
    // We derive the list from `tokenizer.ggml.token_type`: tokens whose type
    // is 3 (CONTROL, e.g. `<|im_start|>`, `<|im_end|>`) or 4 (USER_DEFINED,
    // e.g. `<tool_call>`, `<think>`) are registered as special. Type 5
    // (`[PADxxx]` padding placeholders added by llama.cpp to align vocab
    // sizes) and type 6 (byte fallbacks) are skipped.
    let mut added: Vec<AddedToken> = Vec::new();
    if let Some(tt_val) = ct.metadata.get("tokenizer.ggml.token_type") {
        if let Ok(tt) = tt_val.to_vec() {
            for (i, v) in tt.iter().enumerate() {
                let n = v.to_i32().unwrap_or(1);
                if is_gguf_added_token(n)
                    && let Some(tok) = tokens.get(i)
                {
                    let mut at = AddedToken::from(tok.clone(), true);
                    at.normalized = true;
                    if !added.iter().any(|a| a.content == tok.as_str()) {
                        added.push(at);
                    }
                }
            }
        }
    }

    if !added.is_empty() {
        tokenizer
            .add_special_tokens(added)
            .map_err(anyhow::Error::msg)
            .context("Failed to register added/special tokens from GGUF metadata")?;
    }

    Ok(tokenizer)
}

/// Convenience: open a `.gguf` file and build a tokenizer from its metadata.
///
/// Returns `Ok(None)` if the GGUF lacks `tokenizer.ggml.tokens` /
/// `tokenizer.ggml.merges` (caller should fall back to a sibling
/// `tokenizer.json`); errors out on real I/O / parse problems.
pub fn build_tokenizer_from_gguf_path<P: AsRef<Path>>(path: P) -> Result<Option<Tokenizer>> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open GGUF file {}", path.display()))?;
    let ct = Content::read(&mut file)
        .with_context(|| format!("parse GGUF header of {}", path.display()))?;
    if !gguf_has_embedded_tokenizer(&ct) {
        return Ok(None);
    }
    Ok(Some(build_tokenizer_from_gguf(&ct)?))
}
