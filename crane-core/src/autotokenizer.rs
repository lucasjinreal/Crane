use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::{EncodeInput, Tokenizer};

/// Defines the aditional parameters available for the `from_pretrained` function
#[derive(Debug, Clone)]
pub struct FromPretrainedParameters {
    pub revision: String,
    pub user_agent: HashMap<String, String>,
    pub token: Option<String>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            revision: "main".into(),
            user_agent: HashMap::new(),
            token: None,
        }
    }
}

/// Downloads and cache the identified tokenizer if it exists on
/// the Hugging Face Hub, and returns a local path to the file
pub fn from_pretrained<S: AsRef<str>>(
    identifier: S,
    params: Option<FromPretrainedParameters>,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let identifier: String = identifier.as_ref().to_string();

    let valid_chars = ['-', '_', '.', '/'];
    let is_valid_char = |x: char| x.is_alphanumeric() || valid_chars.contains(&x);

    let valid = identifier.chars().all(is_valid_char);
    let valid_chars_stringified = valid_chars
        .iter()
        .fold(vec![], |mut buf, x| {
            buf.push(format!("'{}'", x));
            buf
        })
        .join(", "); // "'/', '-', '_', '.'"
    if !valid {
        return Err(format!(
            "Model \"{}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}",
            identifier
        )
        .into());
    }
    let params = params.unwrap_or_default();

    let revision = &params.revision;
    let valid_revision = revision.chars().all(is_valid_char);
    if !valid_revision {
        return Err(format!(
            "Revision \"{}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}",
            revision
        )
        .into());
    }

    let mut builder = ApiBuilder::new();
    if let Some(token) = params.token {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;
    let repo = Repo::with_revision(identifier, RepoType::Model, params.revision);
    let api = api.repo(repo);
    Ok(api.get("tokenizer_config.json")?)
}

use minijinja::context;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct TokenObj {
    #[serde(rename = "__type")]
    pub token_type: String,
    pub content: String,
    pub lstrip: bool,
    pub normalized: bool,
    pub rstrip: bool,
    pub single_word: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum Token {
    String(String),
    TokenObj(TokenObj),
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct AutoTokenizerConfig {
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub clean_up_tokenization_spaces: bool,
    pub legacy: Option<bool>,
    pub tokenizer_class: String,
    pub model_max_length: usize,
    pub bos_token: Option<Token>,
    pub eos_token: Option<Token>,
    pub pad_token: Option<Token>,
    pub unk_token: Option<Token>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutoTokenizer {
    pub config: AutoTokenizerConfig,
    pub tokenizer: Tokenizer,
}

impl AutoTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(
        file: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = file.as_ref();
        let content = std::fs::read_to_string(file)?;
        let mut config: AutoTokenizerConfig = serde_json::from_str(&content)?;

        // Fall back to a standalone chat_template.jinja if the field is absent.
        if config.chat_template.is_none() {
            let jinja_path = file.parent().unwrap_or(std::path::Path::new(".")).join("chat_template.jinja");
            if jinja_path.exists() {
                config.chat_template = std::fs::read_to_string(&jinja_path).ok();
            }
        }

        // Load actual tokenizer model
        let d = file.parent();
        let tokenizer = Tokenizer::from_file(d.unwrap().join("tokenizer.json"))?;
        Ok(Self { config, tokenizer })
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, tokenizers::Error> {
        let encoding = self
            .tokenizer
            .encode(EncodeInput::from(text), add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, tokenizers::Error> {
        self.tokenizer.decode(ids, skip_special_tokens)
    }

    pub fn from_pretrained(
        identifier: &str,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let try_path = std::path::Path::new(identifier);

        if try_path.exists() {
            if try_path.is_file() {
                AutoTokenizer::from_file(identifier)
            } else {
                let tokenizer_config_file = try_path.join("tokenizer_config.json");
                AutoTokenizer::from_file(tokenizer_config_file)
            }
        } else {
            let tokenizer_file = from_pretrained(identifier, params)?;
            AutoTokenizer::from_file(tokenizer_file)
        }
    }

    pub fn apply_chat_template<S: serde::Serialize>(
        &self,
        ctx: S,
        add_generation_prompt: bool,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let template_str = self.config.chat_template.as_deref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "missing field `chat_template`",
            )) as Box<dyn std::error::Error + Send + Sync>
        })?;
        let mut env = minijinja::Environment::new();
        env.add_template("default", template_str).unwrap();
        let tmpl = env.get_template("default").unwrap();
        let eos = if let Some(eos) = &self.config.eos_token {
            match eos {
                Token::String(realeos) => realeos,
                Token::TokenObj(token_obj) => &token_obj.content,
            }
        } else {
            &String::new()
        };
        let bos = if let Some(bos) = &self.config.bos_token {
            match bos {
                Token::String(realbos) => realbos,
                Token::TokenObj(token_obj) => &token_obj.content,
            }
        } else {
            &String::new()
        };
        let pad = if let Some(pad) = &self.config.pad_token {
            match pad {
                Token::String(realpad) => realpad,
                Token::TokenObj(token_obj) => &token_obj.content,
            }
        } else {
            &String::new()
        };
        let unk: &String = if let Some(unk) = &self.config.unk_token {
            match unk {
                Token::String(realunk) => realunk,
                Token::TokenObj(token_obj) => &token_obj.content,
            }
        } else {
            &String::new()
        };

        match tmpl.render(context! {
            messages=> ctx,
            unk_token=> *unk,
            pad_token=> *pad,
            bos_token=> *bos,
            eos_token=> *eos,
            add_generation_prompt=> add_generation_prompt
        }) {
            Ok(result) => Ok(result),
            Err(e) => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))),
        }
    }
}
