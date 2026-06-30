use std::io::{self, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

const RESET: &str = "\x1b[0m";
const BLUE: &str = "\x1b[34m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const GREY: &str = "\x1b[90m";
const ITALIC: &str = "\x1b[3m";
const BOLD: &str = "\x1b[1m";

fn arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn model_path() -> String {
    arg("--model-path")
        .or_else(|| std::env::var("MODEL_PATH").ok())
        .unwrap_or_else(|| "checkpoints/Qwen3.5-0.8B".to_string())
}

fn model_type() -> LlmModelType {
    match arg("--model-type")
        .or_else(|| std::env::var("MODEL_TYPE").ok())
        .as_deref()
    {
        Some("qwen25") => LlmModelType::Qwen25,
        Some("qwen3") => LlmModelType::Qwen3,
        Some("hunyuan") => LlmModelType::HunyuanDense,
        _ => LlmModelType::Qwen35,
    }
}

fn max_new_tokens() -> usize {
    arg("--max-new-tokens")
        .or_else(|| std::env::var("MAX_NEW_TOKENS").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

fn pick_device_dtype() -> (DeviceConfig, DataType) {
    #[cfg(feature = "cuda")]
    {
        return (DeviceConfig::Cuda(0), DataType::F16);
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    {
        return (DeviceConfig::Metal, DataType::F16);
    }
    #[allow(unreachable_code)]
    (DeviceConfig::Cpu, DataType::F32)
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn gpu_mem_usage() -> Option<(u64, u64)> {
    None
}

fn grey(text: impl AsRef<str>) -> String {
    format!("{GREY}{}{RESET}", text.as_ref())
}

fn grey_italic(text: impl AsRef<str>) -> String {
    format!("{GREY}{ITALIC}{}{RESET}", text.as_ref())
}

fn blue_bold(text: impl AsRef<str>) -> String {
    format!("{BLUE}{BOLD}{}{RESET}", text.as_ref())
}

fn green_bold(text: impl AsRef<str>) -> String {
    format!("{GREEN}{BOLD}{}{RESET}", text.as_ref())
}

fn yellow(text: impl AsRef<str>) -> String {
    format!("{YELLOW}{}{RESET}", text.as_ref())
}

fn print_help() {
    println!("{} {}", grey("Commands:"), "/help /history /clear /stats /exit");
}

fn main() -> Result<()> {
    let model_path = model_path();
    let model_type = model_type();
    let (device, dtype) = pick_device_dtype();
    let config = ChatConfig {
        common: CommonConfig {
            model_path: model_path.clone(),
            model_type,
            device,
            dtype,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: max_new_tokens(),
            temperature: Some(0.7),
            report_speed: false,
            ..Default::default()
        },
        max_history_turns: 12,
        enable_streaming: true,
    };

    println!("{} {}", grey("Loading model from"), model_path);
    let gpu_before = gpu_mem_usage();
    let load_started_at = Instant::now();
    let mut chat_client = ChatClient::new(config).context("create chat client")?;
    let load_elapsed = load_started_at.elapsed().max(chat_client.load_duration());
    let gpu_after = gpu_mem_usage();

    println!("{} {:.2}s", grey("Loaded in"), load_elapsed.as_secs_f64());
    println!(
        "{} {} | {} {} | {} {}",
        grey("Backend:"),
        chat_client.config().common.device,
        grey("dtype:"),
        chat_client.config().common.dtype,
        grey("max_new_tokens:"),
        chat_client.config().generation.max_new_tokens,
    );
    match (gpu_before, gpu_after) {
        (Some((before_used, _)), Some((after_used, total))) => {
            println!(
                "{} {} / {} (model +{})",
                grey("GPU memory:"),
                format_bytes(after_used),
                format_bytes(total),
                format_bytes(after_used.saturating_sub(before_used)),
            );
        }
        _ => println!("{} {}", grey("GPU memory:"), yellow("unavailable")),
    }
    print_help();

    let mut rl = DefaultEditor::new().context("init line editor")?;
    loop {
        let line = rl.readline(&format!("{} ", blue_bold(">")));
        let message = match line {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err).context("read input"),
        };
        if message.is_empty() {
            continue;
        }
        rl.add_history_entry(message.as_str())
            .context("store history entry")?;

        match message.as_str() {
            "/exit" | "/quit" => break,
            "/help" => {
                print_help();
                continue;
            }
            "/clear" => {
                chat_client.clear_history();
                println!("{}", grey("Conversation cleared."));
                continue;
            }
            "/history" => {
                for (idx, msg) in chat_client.get_history().iter().enumerate() {
                    println!("{} {:?}: {}", idx + 1, msg.role, msg.content);
                }
                continue;
            }
            "/stats" => {
                let turns = chat_client.get_history().len() / 2;
                println!("{} {} | {} {}", grey("Turns:"), turns, grey("messages:"), chat_client.get_history().len());
                continue;
            }
            _ => {}
        }

        print!("{} ", green_bold("AI:"));
        io::stdout().flush()?;

        let started_at = Instant::now();
        let response = chat_client
            .send_message_streaming(&message, |token| {
                print!("{}", token.replace('\u{fffd}', ""));
                let _ = io::stdout().flush();
            })
            .context("generate response")?;
        let elapsed = started_at.elapsed();
        let tokens = response.split_whitespace().count().max(1);
        let avg_tokens_per_sec = tokens as f64 / elapsed.as_secs_f64().max(0.001);

        print!("\r\n");
        io::stdout().flush()?;
        println!(
            "{} {} {} {} {} {}",
            grey_italic("avg"),
            grey_italic(format!("{avg_tokens_per_sec:.2} token/s")),
            grey_italic("chars"),
            grey_italic(response.chars().count().to_string()),
            grey_italic("history"),
            grey_italic(chat_client.get_history().len().to_string()),
        );
    }

    println!("{}", grey("Bye."));
    Ok(())
}
