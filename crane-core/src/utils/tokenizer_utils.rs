use anyhow::{Context, Result};
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
                tokenizer.add_special_tokens(&tokens);
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
