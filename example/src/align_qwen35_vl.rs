//! Alignment + speed harness for Crane's Qwen3.5-2B multimodal model.
//!
//! Reads the reference inputs/logits/tokens produced by
//! `tests/align_qwen35_vl_reference.py` (PyTorch / HuggingFace Transformers)
//! from a directory, runs the SAME forward + greedy decode with Crane on Metal,
//! and reports:
//!   * vision-embedding cosine similarity (identical pixel inputs)
//!   * prefill logits cosine similarity + top-1 argmax agreement
//!   * greedy-decoded token agreement (longest common prefix, accuracy)
//!   * prefill & decode throughput, compared against PyTorch's numbers.
//!
//! Usage:
//!   cargo run --release --features metal,accelerate --bin align_qwen35_vl -- \
//!       checkpoints/Qwen3.5-2B tests/align_out

use anyhow::{Context, Result};
use crane_core::candle_core::{self, DType, Device, Tensor};
use crane_core::models::qwen3_5::Qwen3_5VLModel;
use std::path::{Path, PathBuf};

/// Read a binary array dump written by the Python `write_array` helper:
///   i32 rank, then i32 dims[rank], then raw little-endian data.
fn read_raw(path: &Path) -> Result<(Vec<usize>, Vec<u8>)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut off = 0;
    let take_i32 = |b: &[u8], o: &mut usize| -> i32 {
        let v = i32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]);
        *o += 4;
        v
    };
    let rank = take_i32(&bytes, &mut off) as usize;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(take_i32(&bytes, &mut off) as usize);
    }
    Ok((dims, bytes[off..].to_vec()))
}

fn read_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let (dims, raw) = read_raw(path)?;
    let n = raw.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(f32::from_le_bytes([
            raw[i * 4],
            raw[i * 4 + 1],
            raw[i * 4 + 2],
            raw[i * 4 + 3],
        ]));
    }
    Ok((dims, v))
}

fn read_i32(path: &Path) -> Result<(Vec<usize>, Vec<i32>)> {
    let (dims, raw) = read_raw(path)?;
    let n = raw.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(i32::from_le_bytes([
            raw[i * 4],
            raw[i * 4 + 1],
            raw[i * 4 + 2],
            raw[i * 4 + 3],
        ]));
    }
    Ok((dims, v))
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn top_k_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(v.len());
    idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
    idx.truncate(k);
    idx
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "checkpoints/Qwen3.5-2B".to_string());
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "tests/align_out".to_string()));
    let read = |name: &str| read_f32(&out_dir.join(name)).map(|(d, v)| (d, v));

    // ---- device / dtype ----
    let (device, dtype) = match Device::new_metal(0) {
        Ok(d) => (d, DType::F16),
        Err(_) => (Device::Cpu, DType::F32),
    };
    eprintln!("[crane] device={device:?} dtype={dtype:?}");

    // ---- load model ----
    let t_load = std::time::Instant::now();
    let mut model = Qwen3_5VLModel::new(&model_path, &device, &dtype)?;
    let load_ms = t_load.elapsed().as_millis();

    // ---- read reference inputs ----
    let (pv_dims, pv) = read_f32(&out_dir.join("pixel_values.bin"))?;
    let (_gd, grid_i32) = read_i32(&out_dir.join("grid_thw.bin"))?;
    let (_id_dims, ids_i32) = read_i32(&out_dir.join("input_ids.bin"))?;
    let (_rl_dims, ref_logits) = read("ref_logits.bin")?;
    let (_eb_dims, emb_torch) = read("emb_torch.bin")?;
    let (_g_dims, ref_gen) = read_i32(&out_dir.join("ref_gen_ids.bin"))?;

    let (p, d) = (pv_dims[0], pv_dims[1]);
    eprintln!(
        "[crane] patches={p} feat={d} prompt_len={} gen_ref={}",
        ids_i32.len(),
        ref_gen.len()
    );

    // ---- build tensors ----
    let pv_t = Tensor::from_vec(pv, (p, d), &device)?;
    let grid_t = Tensor::from_vec(
        grid_i32.iter().map(|x| *x as u32).collect::<Vec<_>>(),
        (1, 3),
        &device,
    )?;
    let ids_u32: Vec<u32> = ids_i32.iter().map(|x| *x as u32).collect();
    let s = ids_u32.len();
    let ids_t = Tensor::from_vec(ids_u32.clone(), (1, s), &device)?;

    // eos set matching PyTorch reference
    let eos: Vec<u32> = [248044u32, 248045, 248046].to_vec();

    // ===== 1. vision embedding alignment (identical pixel inputs) =====
    let emb_crane = model.encode_images(&pv_t, &grid_t)?;
    let emb_crane_f = emb_crane
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rows = emb_crane_f.len() / 2048;
    let mut row_cos = Vec::with_capacity(rows);
    for r in 0..rows {
        let a = &emb_crane_f[r * 2048..(r + 1) * 2048];
        let b = &emb_torch[r * 2048..(r + 1) * 2048];
        row_cos.push(cosine(a, b));
    }
    let mean_row_cos: f64 = row_cos.iter().sum::<f64>() / rows as f64;
    let min_row_cos = row_cos.iter().cloned().fold(f64::INFINITY, f64::min);
    let overall_emb_cos = cosine(&emb_crane_f, &emb_torch);
    let mut max_abs_diff = 0.0f32;
    for (a, b) in emb_crane_f.iter().zip(emb_torch.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }

    // ===== 2. prefill logits alignment =====
    // Warm up so Metal shader compilation is not counted in the timed pass.
    model.clear_kv_cache()?;
    let mut warm_logits = model.forward(&ids_t, Some(&pv_t), Some(&grid_t), 0)?;
    let mut warm_pos = s;
    for _ in 0..4 {
        let nx = warm_logits
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .argmax(candle_core::D::Minus1)?
            .to_scalar::<u32>()?;
        warm_logits = model.decode_step(nx, warm_pos)?;
        warm_pos += 1;
    }
    model.clear_kv_cache()?;

    // Isolate vision-tower cost.
    let t_vision = std::time::Instant::now();
    let _ = model.encode_images(&pv_t, &grid_t)?;
    let vision_ms = t_vision.elapsed().as_millis() as f64;

    // Timed prefill (multimodal): best of 3 after warmup.
    let mut best_prefill_ms = f64::INFINITY;
    let mut prefill_logits = model.forward(&ids_t, Some(&pv_t), Some(&grid_t), 0)?;
    for _ in 0..3 {
        model.clear_kv_cache()?;
        let t0 = std::time::Instant::now();
        prefill_logits = model.forward(&ids_t, Some(&pv_t), Some(&grid_t), 0)?;
        let ms = t0.elapsed().as_millis() as f64;
        if ms < best_prefill_ms {
            best_prefill_ms = ms;
        }
    }
    let prefill_ms = best_prefill_ms;
    let prefill_f = prefill_logits
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let logits_cos = cosine(&prefill_f, &ref_logits);
    let crane_argmax = top_k_indices(&prefill_f, 1)[0];
    let ref_argmax = top_k_indices(&ref_logits, 1)[0];
    let crane_top5 = top_k_indices(&prefill_f, 5);
    let ref_top5 = top_k_indices(&ref_logits, 5);
    let top5_overlap = crane_top5.iter().filter(|x| ref_top5.contains(x)).count();

    // ===== 3. greedy decode token agreement =====
    // Reuse the prefill forward's cache: prefill once (section 2), then continue
    // with decode_step (mirrors `generate`). Re-running `forward` would re-fill
    // the already-populated KV cache and double the context length.
    let mut generated: Vec<u32> = Vec::new();
    let mut cur_pos = s;
    let decode_start = std::time::Instant::now();
    let mut logits = prefill_logits; // logits from the prefill forward above
    let max_new = ref_gen.len().max(96);
    for _ in 0..max_new {
        let next = logits
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .argmax(candle_core::D::Minus1)?
            .to_scalar::<u32>()?;
        if eos.contains(&next) {
            break;
        }
        generated.push(next);
        logits = model.decode_step(next, cur_pos)?;
        cur_pos += 1;
    }
    let decode_ms = decode_start.elapsed().as_millis() as f64;

    // compare generated vs reference
    let n = generated.len().min(ref_gen.len());
    let mut lcp = 0;
    for i in 0..n {
        if generated[i] == ref_gen[i] as u32 {
            lcp += 1;
        } else {
            break;
        }
    }
    let mut matches = 0;
    for i in 0..n {
        if generated[i] == ref_gen[i] as u32 {
            matches += 1;
        }
    }
    let acc = if n > 0 {
        matches as f64 / n as f64
    } else {
        0.0
    };

    // also produce decoded text for eyeballing
    let text = model
        .tokenizer
        .tokenizer
        .decode(&generated, true)
        .unwrap_or_default();

    // ===== PyTorch reference numbers for comparison =====
    let summary_path = out_dir.join("summary.json");
    let (torch_prefill_tps, torch_decode_tps) = std::fs::read_to_string(&summary_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .map(|v| {
            let pt = v["prefill_tps"].as_f64().unwrap_or(0.0);
            let dt = v["decode_tps"].as_f64().unwrap_or(0.0);
            (pt, dt)
        })
        .unwrap_or((0.0, 0.0));

    // ===== print report =====
    println!("\n========== ALIGNMENT: Crane vs PyTorch (Qwen3.5-2B-VL) ==========");
    println!(
        "device={device:?} dtype={dtype:?}  prompt_len={s} gen_len(crane)={}",
        generated.len()
    );
    println!();
    println!("--- vision tower (identical pixel_values) ---");
    println!("  vision emb mean row cosine : {mean_row_cos:.6}");
    println!("  vision emb min  row cosine : {min_row_cos:.6}");
    println!("  vision emb overall cosine : {overall_emb_cos:.6}");
    println!("  vision emb max |Δ| (f32)   : {max_abs_diff:.4}");
    println!();
    println!("--- prefill logits (last position) ---");
    println!("  logits cosine             : {logits_cos:.6}");
    println!(
        "  top-1 argmax match        : {} (crane={crane_argmax} ref={ref_argmax})",
        crane_argmax == ref_argmax
    );
    println!("  top-5 overlap             : {top5_overlap}/5");
    println!();
    println!("--- greedy decode (same prompt + pixels) ---");
    println!("  longest common prefix     : {lcp}/{n} tokens");
    println!("  token accuracy            : {acc:.4}");
    println!("  crane decoded text        : {text}");
    println!();
    println!("--- speed (same machine) ---");
    println!("  load model                : {load_ms} ms (crane)");
    println!(
        "  prefill (multimodal)     : {:.1} ms  -> {:.1} t/s (crane)   | pytorch {:.1} t/s",
        prefill_ms,
        s as f64 / prefill_ms * 1000.0,
        torch_prefill_tps
    );
    println!(
        "    - of which vision tower : {:.1} ms  ({:.1}% of prefill)",
        vision_ms,
        vision_ms / prefill_ms * 100.0
    );
    println!(
        "  decode                   : {:.1} ms  -> {:.1} t/s (crane)   | pytorch {:.1} t/s",
        decode_ms,
        generated.len() as f64 / decode_ms * 1000.0,
        torch_decode_tps
    );
    let ct = generated.len() as f64 / decode_ms * 1000.0;
    if torch_decode_tps > 0.0 {
        println!(
            "  => crane decode speedup  : {:.1}x vs PyTorch/MPS",
            ct / torch_decode_tps
        );
    }
    println!("===============================================================");
    Ok(())
}
