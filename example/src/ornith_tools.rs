//! Ornith-1.0-9B tool-calling demo.
//!
//! Ornith is an agentic coding model (Qwen3.5 architecture). This example walks
//! the full tool-use loop end to end:
//!
//!   1. render the prompt with a set of tool specs (`# Tools` system block),
//!   2. let the model reason (`think…`) and emit a `tool_call`,
//!   3. parse the call, run the matching local function,
//!   4. feed the result back as a `tool` turn and let the model answer.
//!
//! Build (CUDA): `cargo run --release -p crane-examples --bin ornith_tools --features cuda`
//! Ornith-9B in bf16 needs ~20 GB of free VRAM; falls back to CPU/f32 otherwise.
//! Override the model path via `--model-path` or the `MODEL_PATH` env var; defaults
//! to `checkpoints/Ornith-1.0-9B`.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use crane_core::autotokenizer::AutoTokenizer;
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::qwen3_5::Model;
use crane_core::models::{DType, Device};
use serde_json::{json, Value};

fn model_path() -> String {
    arg("--model-path")
        .or_else(|| std::env::var("MODEL_PATH").ok())
        .unwrap_or_else(|| "checkpoints/Ornith-1.0-9B".to_string())
}

fn arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

// ── A tiny local tool registry ───────────────────────────────────────────

/// Run a named tool with string arguments, returning a result string (JSON).
fn run_tool(name: &str, args: &BTreeMap<String, String>) -> String {
    match name {
        "get_weather" => {
            // Canned, deterministic "weather service".
            let city = args.get("city").map(String::as_str).unwrap_or("unknown");
            let (temp, cond) = match city.to_lowercase().as_str() {
                "paris" => (15, "light rain"),
                "tokyo" => (22, "clear"),
                "reykjavik" => (6, "windy"),
                _ => (18, "partly cloudy"),
            };
            json!({ "city": city, "temperature_c": temp, "conditions": cond }).to_string()
        }
        "list_directory" => {
            let path = args.get("path").cloned().unwrap_or_else(|| ".".to_string());
            match std::fs::read_dir(&path) {
                Ok(rd) => {
                    let mut names: Vec<String> = rd
                        .filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect();
                    names.sort();
                    json!({ "path": path, "entries": names }).to_string()
                }
                Err(e) => json!({ "path": path, "error": e.to_string() }).to_string(),
            }
        }
        other => json!({ "error": format!("unknown tool '{other}'") }).to_string(),
    }
}

// ── Parse the model's <tool_call> XML ─────────────────────────────────────

struct ToolCall {
    name: String,
    args: BTreeMap<String, String>,
}

/// Extract every `<tool_call>…</tool_call>` block the model emitted.
fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(s) = rest.find("<tool_call>") {
        let after = &rest[s + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else { break };
        if let Some(call) = parse_one(&after[..end]) {
            calls.push(call);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    calls
}

/// Parse one `<function=NAME> <parameter=K>\nV\n</parameter> … </function>`.
fn parse_one(block: &str) -> Option<ToolCall> {
    let fs = block.find("<function=")? + "<function=".len();
    let fe = block[fs..].find('>')? + fs;
    let name = block[fs..fe].trim().to_string();

    let mut args = BTreeMap::new();
    let mut rest = &block[fe..];
    while let Some(ps) = rest.find("<parameter=") {
        let a = &rest[ps + "<parameter=".len()..];
        let Some(pe) = a.find('>') else { break };
        let pname = a[..pe].trim().to_string();
        let val_region = &a[pe + 1..];
        let Some(ve) = val_region.find("</parameter>") else { break };
        // The template wraps the value in newlines: `>\nVALUE\n</parameter>`.
        args.insert(pname, val_region[..ve].trim().to_string());
        rest = &val_region[ve + "</parameter>".len()..];
    }
    Some(ToolCall { name, args })
}

/// Split a generated assistant turn into (reasoning, answer) on `</think>`.
fn split_think(text: &str) -> (&str, &str) {
    match text.find("</think>") {
        Some(i) => (text[..i].trim(), text[i + "</think>".len()..].trim()),
        None => ("", text.trim()),
    }
}

// ── Device / model setup ──────────────────────────────────────────────────

fn pick_device() -> Result<(Device, DType)> {
    #[cfg(feature = "cuda")]
    {
        return Ok((Device::new_cuda(0).context("init CUDA")?, DType::BF16));
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    {
        return Ok((Device::new_metal(0).context("init Metal")?, DType::F16));
    }
    #[allow(unreachable_code)]
    Ok((Device::Cpu, DType::F32))
}

fn generate(model: &mut Model, prompt: &str) -> Result<String> {
    let ids = model.prepare_inputs(prompt).context("tokenize")?;
    let cfg = GenerationConfig {
        max_new_tokens: 768,
        temperature: Some(0.6),
        top_p: Some(0.95),
        report_speed: false,
        ..Default::default()
    };
    let out = model.generate(&ids, &cfg, None).context("generate")?;
    let new_ids = &out[ids.len()..];
    model
        .tokenizer
        .tokenizer
        .decode(new_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
}

fn main() -> Result<()> {
    let (device, dtype) = pick_device()?;
    let path = model_path();
    eprintln!("Loading Ornith-1.0-9B ({dtype:?}) from {path} … this needs ~20 GB on GPU.");
    let mut model = Model::new(&path, &device, &dtype).context("load model")?;
    let tok = AutoTokenizer::from_pretrained(&path, None)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // The tools the model may call this session.
    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string", "description": "City name" } },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List the entries in a local directory.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string", "description": "Directory path" } },
                    "required": ["path"]
                }
            }
        }
    ]);

    let mut messages: Vec<Value> = vec![json!({
        "role": "user",
        "content": "What's the weather in Paris right now? Then tell me what to wear."
    })];
    println!("\nUSER: {}\n", messages[0]["content"].as_str().unwrap());

    // Agentic loop: reason → call tools → observe → answer.
    for round in 1..=4 {
        let prompt = tok
            .apply_chat_template_with_tools(&Value::Array(messages.clone()), Some(&tools), true)
            .map_err(|e| anyhow::anyhow!("render: {e}"))?;
        let text = generate(&mut model, &prompt)?;

        let (reasoning, answer) = split_think(&text);
        if !reasoning.is_empty() {
            println!("── round {round}: model reasoning ──\n{reasoning}\n");
        }

        let calls = parse_tool_calls(&text);
        if calls.is_empty() {
            println!("ASSISTANT (final):\n{answer}");
            return Ok(());
        }

        // Record the assistant turn verbatim, then run each requested tool and
        // append its result as a `tool` message for the next round.
        messages.push(json!({ "role": "assistant", "content": text }));
        for call in &calls {
            let result = run_tool(&call.name, &call.args);
            println!("🔧 tool call: {}({:?})", call.name, call.args);
            println!("   ↳ result: {result}\n");
            messages.push(json!({ "role": "tool", "content": result }));
        }
    }

    println!("(stopped after max rounds without a final answer)");
    Ok(())
}
