//! Simple ASR (Automatic Speech Recognition) Example
//! 
//! This example shows how to transcribe audio to text using the Crane SDK.
//! Note: This requires the 'onnx' feature to be enabled.

#[cfg(feature = "onnx")]
use crane::common::config::{CommonConfig, DataType, DeviceConfig};

#[cfg(feature = "onnx")]
use crane::llm::LlmModelType;

#[cfg(feature = "onnx")]
use crane::prelude::*;

#[cfg(feature = "onnx")]
fn main() -> CraneResult<()> {
    // Create an ASR configuration
    let config = CommonConfig {
        model_path: "checkpoints/moonshine".to_string(), // Update this path to your ASR model
        model_type: LlmModelType::MoonshineAsr,
        device: DeviceConfig::Cpu,
        dtype: DataType::F16,
        max_memory: None,
    };
    
    // Create a new ASR client
    let asr_client = AsrClient::new(config)?;
    
    // Transcribe an audio file
    let transcription = asr_client.transcribe_from_file("data/audio_sample.wav")?;
    println!("Transcription: {}", transcription);
    
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    println!("ASR example requires the 'onnx' feature to be enabled.");
    println!("Run with: cargo run --features onnx --bin asr_simple");
}
