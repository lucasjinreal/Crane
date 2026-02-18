use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::{EncodeInput, Tokenizer};
use minijinja_contrib::add_to_environment;

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
}

/// Rewrite Python-style string method calls and indexing to minijinja filter syntax.
///
/// Handles:
/// - `s.startswith(x)` → `s | startswith(x)`
/// - `s.split(x)[-1]` → `s | split(x) | last`
/// - `s.split(x)[0]`  → `s | split(x) | first`
fn rewrite_python_str_methods(template: &str) -> String {
    // Step 1: rewrite `.split(...)[-1]` → ` | split(...) | last`
    //         and    `.split(...)[0]`  → ` | split(...) | first`
    // Use a simple char-by-char scan to find these patterns.
    let template = rewrite_split_index(template);

    // Step 2: rewrite remaining `.method(` → ` | method(`
    const METHODS: &[&str] = &["startswith", "endswith", "split", "lstrip", "rstrip", "strip"];
    let mut out = template;
    for method in METHODS {
        let pat = format!(".{}(", method);
        let repl = format!(" | {}(", method);
        out = out.replace(&pat, &repl);
    }
    out
}

/// Rewrite `.split(SEP)[N]` to ` | split(SEP) | first/last`.
/// Handles `[0]` → `| first` and `[-1]` → `| last`.
fn rewrite_split_index(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // Look for `.split(`
        if template[i..].starts_with(".split(") {
            // Find matching closing paren
            let start = i + 1; // points to 's' in 'split'
            let args_start = i + 7; // after `.split(`
            if let Some(close) = find_matching_paren(&template[args_start..]) {
                let after_close = args_start + close + 1; // index after ')'
                // Check for [0] or [-1]
                if template[after_close..].starts_with("[0]") {
                    out.push_str(" | ");
                    out.push_str(&template[start..args_start + close + 1]);
                    out.push_str(" | first");
                    i = after_close + 3;
                    continue;
                } else if template[after_close..].starts_with("[-1]") {
                    out.push_str(" | ");
                    out.push_str(&template[start..args_start + close + 1]);
                    out.push_str(" | last");
                    i = after_close + 4;
                    continue;
                }
            }
        }
        out.push(template[i..].chars().next().unwrap());
        i += template[i..].chars().next().unwrap().len_utf8();
    }
    out
}

/// Find the index of the closing `)` that matches the opening `(` assumed at position -1.
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 1usize;
    let mut in_single = false;
    let mut in_double = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' if !in_single && !in_double => depth += 1,
            ')' if !in_single && !in_double => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

impl AutoTokenizer {
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
        // Rewrite Python-style method calls to minijinja filter syntax.
        // e.g. `content.startswith('x')` → `content | startswith('x')`
        let template_str = rewrite_python_str_methods(template_str);

        let mut env = minijinja::Environment::new();
        // Register Python-compatible builtins (tojson, namespace, etc.)
        add_to_environment(&mut env);

        // Additional Python-style string filters.
        env.add_filter("startswith", |s: &str, prefix: &str| s.starts_with(prefix));
        env.add_filter("endswith", |s: &str, suffix: &str| s.ends_with(suffix));
        env.add_filter("split", |s: String, sep: String| -> Vec<String> {
            s.split(sep.as_str()).map(|p| p.to_string()).collect()
        });
        env.add_filter("lstrip", |s: String, chars: Option<String>| -> String {
            match chars {
                None => s.trim_start().to_string(),
                Some(c) => s.trim_start_matches(c.chars().collect::<Vec<_>>().as_slice()).to_string(),
            }
        });
        env.add_filter("rstrip", |s: String, chars: Option<String>| -> String {
            match chars {
                None => s.trim_end().to_string(),
                Some(c) => s.trim_end_matches(c.chars().collect::<Vec<_>>().as_slice()).to_string(),
            }
        });
        env.add_filter("strip", |s: String, chars: Option<String>| -> String {
            match chars {
                None => s.trim().to_string(),
                Some(c) => {
                    let ch: Vec<char> = c.chars().collect();
                    s.trim_matches(ch.as_slice()).to_string()
                }
            }
        });

        env.add_template("default", &template_str).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── rewrite_split_index ──────────────────────────────────────────────

    #[test]
    fn split_index_first() {
        let input = r#"content.split('<tool_response>')[0]"#;
        let output = rewrite_split_index(input);
        assert_eq!(output, r#"content | split('<tool_response>') | first"#);
    }

    #[test]
    fn split_index_last() {
        let input = r#"content.split('<tool_response>')[-1]"#;
        let output = rewrite_split_index(input);
        assert_eq!(output, r#"content | split('<tool_response>') | last"#);
    }

    #[test]
    fn split_index_unchanged_middle() {
        // [1] is not handled — should be left as-is
        let input = r#"content.split('/')[1]"#;
        let output = rewrite_split_index(input);
        assert_eq!(output, r#"content.split('/')[1]"#);
    }

    #[test]
    fn split_index_no_match() {
        let input = "no split here";
        let output = rewrite_split_index(input);
        assert_eq!(output, input);
    }

    // ── rewrite_python_str_methods ───────────────────────────────────────

    #[test]
    fn method_startswith() {
        let input = r#"message.role.startswith('tool')"#;
        let output = rewrite_python_str_methods(input);
        assert_eq!(output, r#"message.role | startswith('tool')"#);
    }

    #[test]
    fn method_endswith() {
        let input = r#"filename.endswith('.json')"#;
        let output = rewrite_python_str_methods(input);
        assert_eq!(output, r#"filename | endswith('.json')"#);
    }

    #[test]
    fn method_strip() {
        let input = "s.strip()";
        let output = rewrite_python_str_methods(input);
        assert_eq!(output, "s | strip()");
    }

    #[test]
    fn method_lstrip_rstrip() {
        let input = "s.lstrip(' ').rstrip(' ')";
        let output = rewrite_python_str_methods(input);
        assert_eq!(output, "s | lstrip(' ') | rstrip(' ')");
    }

    #[test]
    fn method_split_and_index_combined() {
        // Full pipeline: .split(...)[0] AND .startswith(...)
        let input = r#"msg.content.split('<tool_response>')[0].strip()"#;
        let output = rewrite_python_str_methods(input);
        assert_eq!(
            output,
            r#"msg.content | split('<tool_response>') | first | strip()"#
        );
    }

    #[test]
    fn method_split_then_index_last() {
        let input = r#"text.split('\n')[-1]"#;
        let output = rewrite_python_str_methods(input);
        assert_eq!(output, r#"text | split('\n') | last"#);
    }

    #[test]
    fn no_false_positives() {
        // Variables that look like method names but aren't preceded by '.'
        let input = "startswith is not a method call here";
        let output = rewrite_python_str_methods(input);
        assert_eq!(output, input);
    }
}
