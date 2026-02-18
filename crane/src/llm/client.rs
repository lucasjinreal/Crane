use crate::common::{
    CraneError, CraneResult,
    config::{CommonConfig, DataType, DeviceConfig},
};
use crate::llm::{GenerationConfig, LlmModelType};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::streamer::{AsyncTextStreamer, StreamerMessage};

enum LoadedModel {
    Qwen25 {
        model: crane_core::models::qwen25::Model,
        tokenizer: crane_core::autotokenizer::AutoTokenizer,
    },
    HunyuanDense {
        model: crane_core::models::hunyuan_dense::Model,
    },
}

/// LLM client for various language models
pub struct LlmClient {
    config: CommonConfig,
    model: LoadedModel,
}

impl LlmClient {
    /// Create a new LLM client with the given configuration
    pub fn new(config: CommonConfig) -> CraneResult<Self> {
        let device = match &config.device {
            DeviceConfig::Cpu => crane_core::models::Device::Cpu,
            DeviceConfig::Cuda(gpu_id) => crane_core::models::Device::cuda_if_available(*gpu_id as usize)
                .map_err(|e| CraneError::ModelError(e.to_string()))?,
            DeviceConfig::Metal => {
                #[cfg(target_os = "macos")]
                {
                    crane_core::models::Device::new_metal(0)
                        .map_err(|e| CraneError::ModelError(e.to_string()))?
                }
                #[cfg(not(target_os = "macos"))]
                {
                    return Err(CraneError::ConfigError(
                        "Metal device not available on this platform".to_string(),
                    ));
                }
            }
        };

        let dtype = match (&config.device, &config.dtype) {
            (DeviceConfig::Cpu, _) => crane_core::models::DType::F32,
            (_, DataType::F16) => crane_core::models::DType::F16,
            (_, DataType::F32) => crane_core::models::DType::F32,
            (_, DataType::BF16) => crane_core::models::DType::BF16,
        };

        let model = match config.model_type {
            LlmModelType::Qwen25 => {
                let tokenizer = crane_core::autotokenizer::AutoTokenizer::from_pretrained(
                    &config.model_path,
                    None,
                )
                .map_err(|e| CraneError::TokenizationError(e.to_string()))?;

                let mut model = crane_core::models::qwen25::Model::new(&config.model_path, &device, &dtype)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;
                model.warmup();
                LoadedModel::Qwen25 { model, tokenizer }
            }
            LlmModelType::HunyuanDense => {
                let mut model = crane_core::models::hunyuan_dense::Model::new(
                    &config.model_path,
                    &device,
                    &dtype,
                )
                .map_err(|e| CraneError::ModelError(e.to_string()))?;
                model.warmup();
                LoadedModel::HunyuanDense { model }
            }
            _ => {
                return Err(CraneError::ConfigError(
                    format!("Unsupported model type: {:?}", config.model_type),
                ));
            }
        };

        Ok(Self { config, model })
    }

    /// Generate text using the model
    pub fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> CraneResult<String> {
        let messages = [crane_core::chat::Message {
            role: crane_core::chat::Role::User,
            content: prompt.to_string(),
        }];
        self.generate_chat(&messages, config)
    }

    /// Generate text with streaming support
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        callback: F,
    ) -> CraneResult<String>
    where
        F: Fn(&str),
    {
        let messages = [crane_core::chat::Message {
            role: crane_core::chat::Role::User,
            content: prompt.to_string(),
        }];
        self.generate_chat_streaming(&messages, config, callback)
    }
}

impl LlmClient {
    pub fn generate_chat(
        &mut self,
        messages: &[crane_core::chat::Message],
        config: &GenerationConfig,
    ) -> CraneResult<String> {
        let gen_config = crane_core::generation::GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            repeat_last_n: config.repeat_last_n,
            do_sample: config.do_sample,
            pad_token_id: config.pad_token_id,
            eos_token_id: config.eos_token_id,
            report_speed: config.report_speed,
        };

        match &mut self.model {
            LoadedModel::Qwen25 { model, tokenizer } => {
                let prompt = tokenizer
                    .apply_chat_template(messages, true)
                    .map_err(|e| CraneError::TokenizationError(e.to_string()))?;
                let input_ids = model
                    .prepare_inputs(&prompt)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;

                let output_ids = model
                    .generate(&input_ids, &gen_config, None)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;

                let generated_ids = output_ids.get(input_ids.len()..).unwrap_or(&[]);
                let result = tokenizer
                    .decode(generated_ids, true)
                    .map_err(|e| CraneError::TokenizationError(e.to_string()))?;
                Ok(result)
            }
            LoadedModel::HunyuanDense { model } => {
                let prompt = hunyuan_apply_chat_template(messages);
                let input_ids = model
                    .prepare_inputs(&prompt)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;

                let output_ids = model
                    .generate(
                        &input_ids,
                        &crane_core::generation::GenerationConfig {
                            eos_token_id: gen_config.eos_token_id.or(Some(120020)),
                            ..gen_config
                        },
                        None,
                    )
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;

                let generated_ids = output_ids.get(input_ids.len()..).unwrap_or(&[]);
                let result = model
                    .tokenizer
                    .tokenizer
                    .decode(generated_ids, true)
                    .map_err(|e| CraneError::TokenizationError(e.to_string()))?;
                Ok(result)
            }
        }
    }

    pub fn generate_chat_streaming<F>(
        &mut self,
        messages: &[crane_core::chat::Message],
        config: &GenerationConfig,
        callback: F,
    ) -> CraneResult<String>
    where
        F: Fn(&str),
    {
        let mut gen_config = crane_core::generation::GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            repeat_last_n: config.repeat_last_n,
            do_sample: config.do_sample,
            pad_token_id: config.pad_token_id,
            eos_token_id: config.eos_token_id,
            report_speed: config.report_speed,
        };

        let (input_ids, tokenizer_for_stream) = match &self.model {
            LoadedModel::Qwen25 { tokenizer, model } => {
                let prompt = tokenizer
                    .apply_chat_template(messages, true)
                    .map_err(|e| CraneError::TokenizationError(e.to_string()))?;
                let input_ids = model
                    .prepare_inputs(&prompt)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;
                (input_ids, StreamTokenizer::Auto(tokenizer.clone()))
            }
            LoadedModel::HunyuanDense { model } => {
                gen_config.eos_token_id = gen_config.eos_token_id.or(Some(120020));
                let prompt = hunyuan_apply_chat_template(messages);
                let input_ids = model
                    .prepare_inputs(&prompt)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?;
                (input_ids, StreamTokenizer::Tokenizer(model.tokenizer.tokenizer.clone()))
            }
        };

        let (mut streamer, receiver) = match tokenizer_for_stream {
            StreamTokenizer::Auto(t) => AsyncTextStreamer::with_tokenizer(t),
            StreamTokenizer::Tokenizer(t) => AsyncTextStreamer::with_tokenizer(t),
        };

        let mut response_text = String::new();
        std::thread::scope(|scope| {
            let gen_handle = scope.spawn(|| match &mut self.model {
                LoadedModel::Qwen25 { model, .. } => model.generate(&input_ids, &gen_config, Some(&mut streamer)),
                LoadedModel::HunyuanDense { model } => model.generate(&input_ids, &gen_config, Some(&mut streamer)),
            });

            for message in receiver {
                if let StreamerMessage::Token(token_text) = message {
                    callback(&token_text);
                    response_text.push_str(&token_text);
                }
            }

            match gen_handle.join() {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(e)) => Err(CraneError::ModelError(e.to_string())),
                Err(_) => Err(CraneError::ModelError("Generation thread panicked".to_string())),
            }
        })?;

        Ok(response_text)
    }
}

enum StreamTokenizer {
    Auto(crane_core::autotokenizer::AutoTokenizer),
    Tokenizer(tokenizers::Tokenizer),
}

fn hunyuan_apply_chat_template(messages: &[crane_core::chat::Message]) -> String {
    const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
    const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
    const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
    const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
    const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

    let mut result = String::new();
    result.push_str(BOS);

    let (system_msg, loop_messages) = match messages.first() {
        Some(first) if matches!(first.role, crane_core::chat::Role::System) => (Some(&first.content), &messages[1..]),
        _ => (None, messages),
    };

    if let Some(sys) = system_msg {
        result.push_str(sys);
        result.push_str(SEP);
    }

    for msg in loop_messages {
        match msg.role {
            crane_core::chat::Role::User => {
                result.push_str(USER);
                result.push_str(&msg.content);
            }
            crane_core::chat::Role::Assistant => {
                result.push_str(ASSISTANT);
                result.push_str(&msg.content);
                result.push_str(EOS);
            }
            _ => {}
        }
    }

    result.push_str(ASSISTANT);
    result
}
