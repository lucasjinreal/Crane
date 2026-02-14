use anyhow::Result;
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::hunyuan_dense::Model;

/// Special tokens (shared by Hunyuan / Orion):
///   BOS       = 120000  <｜hy_begin▁of▁sentence｜>
///   User      = 120006  <｜hy_User｜>
///   Assistant = 120007  <｜hy_Assistant｜>
///   EOS       = 120020  <｜hy_place▁holder▁no▁2｜>

/// Build single-line translation prompt per README spec.
///
/// ```text
/// 将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：
///
/// {source_text}
/// ```
fn build_single_prompt(source_text: &str) -> String {
    format!(
        "将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{}",
        source_text,
    )
}

/// Build JSONLINE translation prompt per README spec.
///
/// ```text
/// 将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，注意只需要输出翻译后的结果，不要额外解释：
///
///
/// {"1":"line1"}
/// {"2":"line2"}
/// ...
/// ```
fn build_jsonline_prompt(lines: &[&str]) -> String {
    let mut body = String::new();
    for (i, line) in lines.iter().enumerate() {
        // Escape JSON special characters in the source text
        let escaped = line
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        body.push_str(&format!("{{\"{}\":\"{}\"}}\n", i + 1, escaped));
    }
    format!(
        "将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，注意只需要输出翻译后的结果，不要额外解释：\n\n\n{}",
        body.trim_end(),
    )
}

/// Wrap user content with the Hunyuan-family chat template.
///   <BOS><User>{content}<Assistant>
fn apply_chat_template(user_content: &str) -> String {
    format!(
        "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>\
         <\u{ff5c}hy_User\u{ff5c}>{}\
         <\u{ff5c}hy_Assistant\u{ff5c}>",
        user_content,
    )
}

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model/Orion-HYMT1.5-1.8B-SFT-v2601".to_string());

    println!("Loading model from: {}", model_path);

    #[cfg(feature = "cuda")]
    let device = crane_core::models::Device::cuda_if_available(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = crane_core::models::Device::Cpu;

    // BF16 on CUDA (4090 native), F32 on CPU
    #[cfg(feature = "cuda")]
    let dtype = crane_core::models::DType::BF16;
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    println!("Device: {:?}, dtype: {:?}", device, dtype);

    let mut model = Model::new(&model_path, &device, &dtype)?;
    println!("Model loaded successfully!");

    model.warmup();
    println!("Model warmed up.");

    // ── JSONLINE long-text test ──
    let source_lines: Vec<&str> = vec![
        "「子どものオシラサマは、絹になる生糸を吐けるのよ。とっても上質なのよ。私もたくさん吐いたわ。でもね、思春期――大人になる過程で、糸は吐けなくなってしまうのよ。そういう風に、人間様が作ったのよ」",
        "「そういうことに、なるわね。知らなかった？ 知らなかったのね。戦争で、私たちの一族は人間領を追われてしまった。ずっと人間様と一緒にいたのに。人間様といるために、身体がこうなったのに。戦争が始まったら、人間様と一緒にいられなくなったの。そう聞いてるわ。追われた時の別れは、それは惨いものだったと――聞いているわ」",
        "「聞いているということは……フソゥさんは、その時代を知らないのですね」",
        "「知らないわ。開戦の直前、百年近く前だと聞いているもの。でもね、でもね、人間様のことは聞いているの。一緒に過ごした時代の話を、おばあ様からたくさん聞かされたの。人間様は優しいのね、子どものときはオシラサマから糸をとって、大人になったらたくさんもてなしてくれるんですって。身の回りの世話をなんでもしてくれるんですって」",
        "「私も話を聞いててとっても羨ましかったわ。人間様はとても優しいんだって。なんでもしてくれるんだって。だから私はね、人間様に会いたかったの。ずっとずっと会いたかったの。でも戦争で、魔族領に逃れた私たちには、再会の術がなかったの――今日まさに今この瞬間、そうまさに、この瞬間までは……！」",
        "人間の生徒は、このネメア・アカデミーではグレンただ一人。",
        "「あの……それで、僕になにを、求めているのでしょうか……」",
    ];

    let user_content = build_jsonline_prompt(&source_lines);
    let formatted = apply_chat_template(&user_content);

    let input_ids = model.prepare_inputs(&formatted)?;

    println!("Mode: JSONLINE ({} lines)", source_lines.len());
    println!("Input tokens: {}", input_ids.len());
    for (i, line) in source_lines.iter().enumerate() {
        let preview: String = line.chars().take(40).collect();
        println!("  [{}] {}...", i + 1, preview);
    }

    // Generation config from Orion's generation_config.json
    let gen_config = GenerationConfig {
        max_new_tokens: 1024,
        temperature: Some(0.7),
        top_p: Some(0.8),
        repetition_penalty: 1.05,
        repeat_last_n: 64,
        eos_token_id: Some(120020),
        report_speed: true,
        ..Default::default()
    };

    println!("\nTranslating...");
    let output_ids = model.generate(&input_ids, &gen_config, None)?;

    // Decode only the generated tokens (skip the prompt)
    let generated_ids = &output_ids[input_ids.len()..];
    let translation = model
        .tokenizer
        .tokenizer
        .decode(generated_ids, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("\n── Translation ──\n{}", translation);

    Ok(())
}
