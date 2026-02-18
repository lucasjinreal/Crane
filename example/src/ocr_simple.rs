use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::LlmModelType;
use crane::prelude::*;
use std::env;

fn main() -> CraneResult<()> {
    let args: Vec<String> = env::args().collect();
    let image_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("data/images/test_chart.png");

    let config = CommonConfig {
        // model_path: "checkpoints/PaddleOCR-VL".to_string(),
        model_path: "checkpoints/PaddleOCR-VL-1.5".to_string(),
        model_type: LlmModelType::PaddleOcrVl,
        // device: DeviceConfig::Cpu,
        device: DeviceConfig::Cuda(0),
        // dtype: DataType::BF16,
        dtype: DataType::F32,
        max_memory: None,
    };

    let mut ocr_client = OcrClient::new(config)?;

    println!("Performing OCR on image: {}", image_path);

    // let response = ocr_client.extract_text_from_image(image_path)?;
    // println!("OCR result: {}", response);

    // streaming way
    let _ = ocr_client.extract_text_from_image_stream(image_path)?;
    println!("");

    Ok(())
}
