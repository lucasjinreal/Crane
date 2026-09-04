// SPDX-License-Identifier: MIT
//! Audio8-TTS model wrapper: loads the three ONNX graphs (slow AR, fast AR,
//! codec decoder), builds the `ChatML` prompt, and runs the `DualAR` generation
//! loop (one semantic token per frame from the slow AR, `num_codebooks`
//! codec entries per frame from the fast AR) to produce a waveform.
//!
//! CPU-only: Crane's ONNX evaluator has no `Device` concept (every
//! initializer is materialized on `Device::Cpu`), so there is no GPU path
//! for this model.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, Tensor};

use super::config::{self, RuntimeManifest};
use super::prompt::PromptBuilder;
use super::sampling::{
    SamplingParams, SemanticVocab, SplitMix64, sample_semantic, sample_topk_topp,
};
use crate::generation::SpeechOptions;
use crate::onnx::Session;
use crate::onnx::proto::{self as onnx_proto, GraphProto};

/// Nucleus sampling top-k, matching the reference implementation's default.
const DEFAULT_TOP_K: usize = 50;
/// Default nucleus sampling top-p when `SpeechOptions::top_p` is `None`.
const DEFAULT_TOP_P: f64 = 0.95;
/// Default sampling temperature when `SpeechOptions::temperature` is `None`.
/// This 0.1B Preview model is prone to sampling a bad token at
/// mid-utterance pauses (e.g. a comma) and needing Repetition-Avoidance
/// Sampling to recover; empirically, that recovery is far noisier at high
/// temperature (measured high-frequency burst energy dropped roughly 18x
/// going from 0.9 to 0.3 on the same input/seed). `0.3` is also this
/// model family's own reference service's default, unlike `0.9`, which
/// has no source in this package's `config.json`.
const DEFAULT_TEMPERATURE: f64 = 0.3;
/// PRNG seed, matching the reference implementation's default.
const DEFAULT_SEED: u64 = 42;
/// Number of recent semantic tokens considered for repetition avoidance.
const RAS_WINDOW_SIZE: usize = 10;

/// Resolves `model_dir.join(map[key])`, the ONNX filename `manifest` names
/// for one of the three graphs (slow AR, fast AR, codec decoder) under a
/// given precision key.
///
/// # Errors
///
/// Returns an error if `map` has no entry for `key`.
fn manifest_model_path(
    map: &HashMap<String, String>,
    key: &str,
    what: &str,
    model_dir: &Path,
) -> Result<std::path::PathBuf> {
    let filename = map
        .get(key)
        .ok_or_else(|| anyhow!("manifest has no {what} entry for {key:?}"))?;
    Ok(model_dir.join(filename))
}

/// The slow AR's persistent state tensor shapes, read from its ONNX graph.
struct SlowStateShapes {
    cache_keys: Vec<usize>,
    cache_values: Vec<usize>,
    conv_states: Vec<usize>,
    ssm_states: Vec<usize>,
}

/// Reads the slow AR graph's `cache_keys`/`cache_values`/`conv_states`/
/// `ssm_states` input shapes, cross-checked against `manifest.num_layers`.
///
/// # Errors
///
/// Returns an error if any of the four inputs is missing or non-fixed-shape,
/// or `cache_keys`'s layer count doesn't match `manifest.num_layers`.
fn load_slow_state_shapes(
    graph: &GraphProto,
    manifest: &RuntimeManifest,
) -> Result<SlowStateShapes> {
    let cache_keys = input_shape(graph, "cache_keys")?;
    let cache_values = input_shape(graph, "cache_values")?;
    let conv_states = input_shape(graph, "conv_states")?;
    let ssm_states = input_shape(graph, "ssm_states")?;
    if cache_keys.first() != Some(&manifest.num_layers) {
        bail!(
            "slow AR cache_keys has {:?} layers, manifest says num_layers={}",
            cache_keys.first(),
            manifest.num_layers
        );
    }
    Ok(SlowStateShapes {
        cache_keys,
        cache_values,
        conv_states,
        ssm_states,
    })
}

/// Reads the fully-static shape of `graph`'s input named `name`.
///
/// # Errors
///
/// Returns an error if no such input exists, it has no tensor type, or any
/// of its dimensions isn't a fixed value.
fn input_shape(graph: &GraphProto, name: &str) -> Result<Vec<usize>> {
    let input = graph
        .input
        .iter()
        .find(|input| input.name == name)
        .ok_or_else(|| anyhow!("ONNX graph has no input named {name:?}"))?;
    let Some(onnx_proto::type_proto::Value::TensorType(tensor_type)) =
        input.r#type.as_ref().and_then(|t| t.value.as_ref())
    else {
        bail!("input {name:?} has no tensor type");
    };
    let shape = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| anyhow!("input {name:?} has no declared shape"))?;
    shape
        .dim
        .iter()
        .map(|dim| match &dim.value {
            Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(value)) => {
                usize::try_from(*value)
                    .with_context(|| format!("input {name:?} has a negative dimension"))
            },
            _ => bail!("input {name:?} has a non-fixed dimension"),
        })
        .collect()
}

/// Audio8-TTS 0.1B ONNX INT8: a `DualAR` text-to-speech model. A slow AR
/// transformer/Mamba-hybrid backbone autoregressively predicts one semantic
/// token per audio frame; a fast AR transformer expands each frame's hidden
/// state into `num_codebooks` codec codebook entries; a bundled neural codec
/// decodes the codebooks into a waveform.
pub struct Model {
    slow_session: Session,
    fast_session: Session,
    codec_session: Session,
    manifest: RuntimeManifest,
    prompt_builder: PromptBuilder,
    reference_codes: Vec<Vec<i64>>,
    // Slow AR state, threaded through every `slow_step` call and reset at
    // the start of `generate_speech`.
    cache_keys: Tensor,
    cache_values: Tensor,
    conv_states: Tensor,
    ssm_states: Tensor,
    // Fast AR state, reset at the start of every `generate_frame` call.
    fast_cache_keys: Vec<Tensor>,
    fast_cache_values: Vec<Tensor>,
}

impl Model {
    /// Loads an Audio8-TTS ONNX INT8 package from `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if `runtime_manifest.json`, the tokenizer, any of
    /// the three ONNX graphs, or the bundled reference voice codes are
    /// missing, malformed, or internally inconsistent (e.g. the reference
    /// codes' codebook count doesn't match the manifest).
    pub fn new(model_dir: &str) -> Result<Self> {
        let model_dir = Path::new(model_dir);
        let manifest = config::load_manifest(model_dir)?;

        if manifest.semantic_end_id
            != manifest.semantic_begin_id + i64::try_from(manifest.codebook_size)? - 1
        {
            bail!(
                "manifest semantic_end_id ({}) is inconsistent with semantic_begin_id ({}) + codebook_size ({})",
                manifest.semantic_end_id,
                manifest.semantic_begin_id,
                manifest.codebook_size
            );
        }
        if manifest.num_codebooks == 0 || manifest.codebook_size == 0 {
            bail!(
                "manifest num_codebooks ({}) and codebook_size ({}) must both be non-zero",
                manifest.num_codebooks,
                manifest.codebook_size
            );
        }

        let slow_path = manifest_model_path(
            &manifest.slow_decode_models,
            &manifest.default_precision,
            "slow_decode_models",
            model_dir,
        )?;
        let fast_path = manifest_model_path(
            &manifest.fast_models,
            &manifest.default_precision,
            "fast_models",
            model_dir,
        )?;
        let codec_path = manifest_model_path(
            &manifest.codec_models,
            &manifest.default_codec_precision,
            "codec_models",
            model_dir,
        )?;

        let slow_proto = crate::onnx::read_file(&slow_path)
            .with_context(|| format!("loading {}", slow_path.display()))?;
        let slow_graph = slow_proto
            .graph
            .as_ref()
            .ok_or_else(|| anyhow!("{} has no graph", slow_path.display()))?;
        let slow_state_shapes = load_slow_state_shapes(slow_graph, &manifest)?;
        let slow_session = Session::new(slow_proto)?;

        let fast_proto = crate::onnx::read_file(&fast_path)
            .with_context(|| format!("loading {}", fast_path.display()))?;
        let fast_graph = fast_proto
            .graph
            .as_ref()
            .ok_or_else(|| anyhow!("{} has no graph", fast_path.display()))?;
        let fast_cache_key_shape = input_shape(fast_graph, "cache_key_0")?;
        let fast_cache_value_shape = input_shape(fast_graph, "cache_value_0")?;
        let fast_session = Session::new(fast_proto)?;

        let codec_proto = crate::onnx::read_file(&codec_path)
            .with_context(|| format!("loading {}", codec_path.display()))?;
        let codec_session = Session::new(codec_proto)?;

        let prompt_builder = PromptBuilder::new(
            &model_dir.join("tokenizer"),
            manifest.semantic_begin_id,
            manifest.num_codebooks,
        )?;

        let reference_codes_path = model_dir.join(&manifest.reference_codes);
        let (reference_codes, rows, _cols) = config::load_npy_i64_2d(&reference_codes_path)?;
        if rows != manifest.num_codebooks {
            bail!(
                "{} has {rows} codebooks, expected {}",
                reference_codes_path.display(),
                manifest.num_codebooks
            );
        }

        let device = Device::Cpu;
        let cache_keys = Tensor::zeros(slow_state_shapes.cache_keys, DType::F32, &device)?;
        let cache_values = Tensor::zeros(slow_state_shapes.cache_values, DType::F32, &device)?;
        let conv_states = Tensor::zeros(slow_state_shapes.conv_states, DType::F32, &device)?;
        let ssm_states = Tensor::zeros(slow_state_shapes.ssm_states, DType::F32, &device)?;
        let fast_cache_keys = (0..manifest.num_fast_layers)
            .map(|_| Tensor::zeros(fast_cache_key_shape.as_slice(), DType::F32, &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let fast_cache_values = (0..manifest.num_fast_layers)
            .map(|_| Tensor::zeros(fast_cache_value_shape.as_slice(), DType::F32, &device))
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self {
            slow_session,
            fast_session,
            codec_session,
            manifest,
            prompt_builder,
            reference_codes,
            cache_keys,
            cache_values,
            conv_states,
            ssm_states,
            fast_cache_keys,
            fast_cache_values,
        })
    }

    /// The waveform sample rate this model's codec decoder produces, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.manifest.sample_rate
    }

    /// Generates speech for `text` using the bundled reference voice.
    ///
    /// Audio8-TTS's Preview release doesn't vary generation by `_language`
    /// (documentation-only coverage claim, not runtime-enforced) or by
    /// `_voice` (a single bundled reference voice ships with the package);
    /// both parameters are accepted for API parity with Crane's other TTS
    /// models.
    ///
    /// # Errors
    ///
    /// Returns an error if prompt construction fails, the prompt already
    /// meets or exceeds the model's maximum sequence length, or any ONNX
    /// session invocation fails.
    pub fn generate_speech(
        &mut self,
        text: &str,
        _language: &str,
        _voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<(Tensor, u32)> {
        let device = Device::Cpu;
        let prompt = self.prompt_builder.build(
            text,
            &self.manifest.reference_text,
            &self.reference_codes,
        )?;
        let prompt_len = prompt.dim(2)?;
        if prompt_len >= self.manifest.max_seq_len {
            bail!(
                "prompt length {prompt_len} meets or exceeds max sequence length {}",
                self.manifest.max_seq_len
            );
        }
        let max_new = opts
            .max_new_tokens
            .min(self.manifest.max_seq_len - prompt_len);

        self.cache_keys = self.cache_keys.zeros_like()?;
        self.cache_values = self.cache_values.zeros_like()?;
        self.conv_states = self.conv_states.zeros_like()?;
        self.ssm_states = self.ssm_states.zeros_like()?;

        let mut rng = SplitMix64::new(DEFAULT_SEED);
        let sampling_params = SamplingParams {
            temperature: opts.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: opts.top_p.unwrap_or(DEFAULT_TOP_P),
            top_k: DEFAULT_TOP_K,
        };
        let vocab = SemanticVocab {
            semantic_begin_id: self.manifest.semantic_begin_id,
            im_end_id: self.manifest.im_end_id,
            codebook_size: self.manifest.codebook_size,
        };

        let mut state: Option<(Vec<f64>, Tensor)> = None;
        for pos in 0..prompt_len {
            let column = prompt.narrow(2, pos, 1)?.contiguous()?;
            // pos < max_seq_len (2048 in the shipped package), well within
            // i64 range.
            #[allow(clippy::cast_possible_wrap)]
            let position_value = pos as i64;
            let position = Tensor::from_vec(vec![position_value], (1,), &device)?;
            state = Some(self.slow_step(&column, &position)?);
        }
        let (mut logits, mut hidden) =
            state.ok_or_else(|| anyhow!("prompt produced an empty token sequence"))?;

        let mut previous: VecDeque<i64> = VecDeque::with_capacity(RAS_WINDOW_SIZE);
        let mut frames: Vec<Vec<i64>> = Vec::new();

        for step in 0..max_new {
            let semantic = sample_semantic(&logits, &previous, &sampling_params, &vocab, &mut rng);
            if semantic == self.manifest.im_end_id {
                break;
            }
            previous.push_back(semantic);
            if previous.len() > RAS_WINDOW_SIZE {
                previous.pop_front();
            }

            let frame = self.generate_frame(&hidden, semantic, &sampling_params, &mut rng)?;

            if step + 1 >= max_new {
                frames.push(frame);
                break;
            }

            let mut column_values = Vec::with_capacity(self.manifest.num_codebooks + 1);
            column_values.push(semantic);
            column_values.extend_from_slice(&frame);
            frames.push(frame);
            let column = Tensor::from_vec(
                column_values,
                (1, self.manifest.num_codebooks + 1, 1),
                &device,
            )?;
            // prompt_len + step < max_seq_len (2048), well within i64 range.
            #[allow(clippy::cast_possible_wrap)]
            let position_value = (prompt_len + step) as i64;
            let position = Tensor::from_vec(vec![position_value], (1,), &device)?;
            (logits, hidden) = self.slow_step(&column, &position)?;
        }

        if frames.is_empty() {
            bail!("Audio8-TTS generated no frames for the given input text");
        }

        let num_codebooks = self.manifest.num_codebooks;
        let mut codes_values = Vec::with_capacity(num_codebooks * frames.len());
        for codebook in 0..num_codebooks {
            for frame in &frames {
                codes_values.push(frame[codebook]);
            }
        }
        let codes = Tensor::from_vec(codes_values, (1, num_codebooks, frames.len()), &device)?;

        let audio = self.decode_codes(&codes)?;
        Ok((audio, self.manifest.sample_rate))
    }

    /// Runs one slow AR step: `codes` is a single `(1, num_codebooks + 1, 1)`
    /// column and `position` is that column's `(1,)` sequence position.
    /// Updates the persistent KV-cache and Mamba state in place and returns
    /// the step's logits (as `f64`, ready for [`sample_semantic`]) and
    /// hidden state (fed straight into [`Self::generate_frame`]).
    fn slow_step(&mut self, codes: &Tensor, position: &Tensor) -> Result<(Vec<f64>, Tensor)> {
        // `Tensor::clone` is an `Arc` refcount bump, not a data copy; it's
        // required here because `Session::run` takes ownership of its inputs.
        let inputs = HashMap::from([
            ("codes".to_string(), codes.clone()),
            ("position".to_string(), position.clone()),
            ("cache_keys".to_string(), self.cache_keys.clone()),
            ("cache_values".to_string(), self.cache_values.clone()),
            ("conv_states".to_string(), self.conv_states.clone()),
            ("ssm_states".to_string(), self.ssm_states.clone()),
        ]);
        let outputs = self
            .slow_session
            .run(inputs)
            .context("running slow AR session")?;

        let logits = output(&outputs, "logits")?;
        let logits: Vec<f64> = logits
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let hidden = output(&outputs, "hidden")?.clone();

        // `key_delta`/`value_delta` lack the sequence axis `cache_keys`'s
        // dim 3 has (they cover exactly the one new position), so it's
        // inserted before scattering into the persistent cache.
        let key_delta = output(&outputs, "key_delta")?.contiguous()?.unsqueeze(3)?;
        let value_delta = output(&outputs, "value_delta")?
            .contiguous()?
            .unsqueeze(3)?;
        let position_value = position.to_vec1::<i64>()?[0];
        let offset = usize::try_from(position_value)
            .with_context(|| format!("slow AR position {position_value} is negative"))?;
        self.cache_keys.slice_set(&key_delta, 3, offset)?;
        self.cache_values.slice_set(&value_delta, 3, offset)?;

        self.conv_states = output(&outputs, "next_conv_states")?.contiguous()?;
        self.ssm_states = output(&outputs, "next_ssm_states")?.contiguous()?;

        Ok((logits, hidden))
    }

    /// Expands one slow-AR frame's hidden state into `num_codebooks` fast AR
    /// codec codebook indices. Codebook 0 is derived deterministically from
    /// `semantic` (matching the reference implementation); codebooks 1.. are
    /// sampled autoregressively from the fast AR.
    fn generate_frame(
        &mut self,
        hidden: &Tensor,
        semantic: i64,
        sampling_params: &SamplingParams,
        rng: &mut SplitMix64,
    ) -> Result<Vec<i64>> {
        for cache in self
            .fast_cache_keys
            .iter_mut()
            .chain(self.fast_cache_values.iter_mut())
        {
            *cache = cache.zeros_like()?;
        }

        // Priming step: primes the fast AR's cache with the slow hidden
        // state; its logits are unused.
        self.fast_step(hidden, 0, true, 0)?;

        let codebook_size = self.manifest.codebook_size;
        // codebook_size (4096 in the shipped package) - 1 is well within i64
        // range.
        #[allow(clippy::cast_possible_wrap)]
        let max_codebook_index = codebook_size as i64 - 1;
        let mut token = (semantic - self.manifest.semantic_begin_id).clamp(0, max_codebook_index);
        let mut codebooks = vec![token];

        for pos in 1..self.manifest.num_codebooks {
            // pos < num_codebooks (10), well within i64 range.
            #[allow(clippy::cast_possible_wrap)]
            let position = pos as i64;
            let logits = self.fast_step(hidden, token, false, position)?;
            let sampled = sample_topk_topp(&logits, sampling_params, rng);
            // sampled < codebook_size (4096), well within i64 range.
            #[allow(clippy::cast_possible_wrap)]
            {
                token = sampled as i64;
            }
            codebooks.push(token);
        }
        Ok(codebooks)
    }

    /// Runs one fast AR step. `use_hidden = true` primes the fast AR with
    /// `hidden` (the caller must discard the returned logits in that case);
    /// otherwise `token_id` is the previous codebook's sampled value.
    /// Updates the persistent fast-AR KV-cache in place and returns the
    /// step's logits as `f64`.
    fn fast_step(
        &mut self,
        hidden: &Tensor,
        token_id: i64,
        use_hidden: bool,
        position: i64,
    ) -> Result<Vec<f64>> {
        let device = Device::Cpu;
        // `Tensor::clone` is an `Arc` refcount bump, not a data copy; it's
        // required here because `Session::run` takes ownership of its inputs.
        let mut inputs = HashMap::from([
            ("slow_hidden".to_string(), hidden.clone()),
            (
                "token_id".to_string(),
                Tensor::from_vec(vec![token_id], (1, 1), &device)?,
            ),
            (
                "use_slow_hidden".to_string(),
                Tensor::from_vec(vec![u8::from(use_hidden)], (1,), &device)?,
            ),
            (
                "input_pos".to_string(),
                Tensor::from_vec(vec![position], (1,), &device)?,
            ),
        ]);
        for i in 0..self.manifest.num_fast_layers {
            inputs.insert(format!("cache_key_{i}"), self.fast_cache_keys[i].clone());
            inputs.insert(
                format!("cache_value_{i}"),
                self.fast_cache_values[i].clone(),
            );
        }
        let outputs = self
            .fast_session
            .run(inputs)
            .context("running fast AR session")?;

        let logits: Vec<f64> = output(&outputs, "logits")?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| f64::from(v))
            .collect();

        let offset = usize::try_from(position)
            .with_context(|| format!("fast AR position {position} is negative"))?;
        for i in 0..self.manifest.num_fast_layers {
            let key_delta = output(&outputs, &format!("key_delta_{i}"))?.contiguous()?;
            let value_delta = output(&outputs, &format!("value_delta_{i}"))?.contiguous()?;
            self.fast_cache_keys[i].slice_set(&key_delta, 2, offset)?;
            self.fast_cache_values[i].slice_set(&value_delta, 2, offset)?;
        }
        Ok(logits)
    }

    /// Decodes `codes` (shape `(1, num_codebooks, T)`) into a waveform via
    /// the bundled neural codec.
    fn decode_codes(&self, codes: &Tensor) -> Result<Tensor> {
        // `Tensor::clone` is an `Arc` refcount bump, not a data copy; it's
        // required here because `Session::run` takes ownership of its inputs.
        let inputs = HashMap::from([("codes".to_string(), codes.clone())]);
        let outputs = self
            .codec_session
            .run(inputs)
            .context("running codec decoder session")?;
        Ok(output(&outputs, "audio")?.clone())
    }
}

/// Looks up `name` in an ONNX session's output map with a clear error
/// instead of a `HashMap` panic if the graph's output names ever change.
fn output<'a>(outputs: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    outputs
        .get(name)
        .ok_or_else(|| anyhow!("ONNX session is missing expected output {name:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_type_input(name: &str, dims: &[i64]) -> onnx_proto::ValueInfoProto {
        onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                value: Some(onnx_proto::type_proto::Value::TensorType(
                    onnx_proto::type_proto::Tensor {
                        elem_type: 1,
                        shape: Some(onnx_proto::TensorShapeProto {
                            dim: dims
                                .iter()
                                .map(|&d| onnx_proto::tensor_shape_proto::Dimension {
                                    value: Some(
                                        onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                                            d,
                                        ),
                                    ),
                                    ..Default::default()
                                })
                                .collect(),
                        }),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    // A fixed-shape input's dimensions are read back as declared.
    #[test]
    fn input_shape_reads_fixed_dims() {
        let graph = GraphProto {
            input: vec![tensor_type_input("cache_keys", &[4, 1, 8])],
            ..Default::default()
        };
        assert_eq!(input_shape(&graph, "cache_keys").unwrap(), vec![4, 1, 8]);
    }

    // No input with the requested name is an error, not a panic.
    #[test]
    fn input_shape_missing_input_errors() {
        let graph = GraphProto::default();
        assert!(input_shape(&graph, "cache_keys").is_err());
    }

    // An input with no declared type (e.g. no tensor_type) is an error.
    #[test]
    fn input_shape_no_tensor_type_errors() {
        let graph = GraphProto {
            input: vec![onnx_proto::ValueInfoProto {
                name: "cache_keys".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(input_shape(&graph, "cache_keys").is_err());
    }

    // A symbolic (non-fixed) dimension is an error.
    #[test]
    fn input_shape_non_fixed_dim_errors() {
        let graph = GraphProto {
            input: vec![onnx_proto::ValueInfoProto {
                name: "cache_keys".to_string(),
                r#type: Some(onnx_proto::TypeProto {
                    value: Some(onnx_proto::type_proto::Value::TensorType(
                        onnx_proto::type_proto::Tensor {
                            elem_type: 1,
                            shape: Some(onnx_proto::TensorShapeProto {
                                dim: vec![onnx_proto::tensor_shape_proto::Dimension {
                                    value: Some(
                                        onnx_proto::tensor_shape_proto::dimension::Value::DimParam(
                                            "batch".to_string(),
                                        ),
                                    ),
                                    ..Default::default()
                                }],
                            }),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(input_shape(&graph, "cache_keys").is_err());
    }

    // A present key resolves to `model_dir` joined with its filename.
    #[test]
    fn manifest_model_path_resolves_present_key() {
        let map = HashMap::from([("fp32".to_string(), "slow.onnx".to_string())]);
        let path =
            manifest_model_path(&map, "fp32", "slow_decode_models", Path::new("/models")).unwrap();
        assert_eq!(path, Path::new("/models/slow.onnx"));
    }

    // A missing key is an error, not a panic.
    #[test]
    fn manifest_model_path_missing_key_errors() {
        let map = HashMap::new();
        assert!(
            manifest_model_path(&map, "fp32", "slow_decode_models", Path::new("/models")).is_err()
        );
    }

    // The feedback column fed back into the slow AR after each frame must
    // be shaped (1, num_codebooks + 1, 1) with row 0 = semantic token and
    // rows 1.. = the frame's codebook values, in order.
    #[test]
    fn feedback_column_shape_and_layout() {
        let semantic = 65540i64;
        let frame = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut column_values = Vec::with_capacity(frame.len() + 1);
        column_values.push(semantic);
        column_values.extend(frame.iter().copied());
        let column =
            Tensor::from_vec(column_values, (1, frame.len() + 1, 1), &Device::Cpu).unwrap();

        assert_eq!(column.dims(), &[1, 11, 1]);
        let flat = column.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(flat[0], semantic);
        assert_eq!(&flat[1..], frame.as_slice());
    }

    // Codebook 0 is derived deterministically from the semantic token,
    // clamped to the valid codebook index range.
    #[test]
    fn deterministic_codebook_0_clamps_to_range() {
        let semantic_begin_id = 65537i64;
        let codebook_size = 4096i64;
        let clamp = |semantic: i64| (semantic - semantic_begin_id).clamp(0, codebook_size - 1);

        assert_eq!(clamp(semantic_begin_id), 0);
        assert_eq!(clamp(semantic_begin_id - 100), 0);
        assert_eq!(
            clamp(semantic_begin_id + codebook_size - 1),
            codebook_size - 1
        );
        assert_eq!(
            clamp(semantic_begin_id + codebook_size + 100),
            codebook_size - 1
        );
    }

    fn model_dir() -> Option<std::path::PathBuf> {
        let dir = if let Ok(dir) = std::env::var("CRANE_AUDIO8_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()?
                .join("models/tts/Audio8-TTS-0.1B-ONNX-INT8")
        };
        dir.is_dir().then_some(dir)
    }

    // Loads the real downloaded package and checks basic model metadata.
    #[test]
    #[ignore = "requires the downloaded Audio8-TTS-0.1B-ONNX-INT8 package"]
    fn new_loads_real_model() {
        let dir = model_dir().expect("set CRANE_AUDIO8_DIR or download the model");
        let model = Model::new(dir.to_str().unwrap()).unwrap();
        assert_eq!(model.sample_rate(), 44100);
    }

    // End-to-end smoke test: generate speech for a short sentence and check
    // the result is a non-empty waveform.
    #[test]
    #[ignore = "requires the downloaded Audio8-TTS-0.1B-ONNX-INT8 package"]
    fn generate_speech_smoke() {
        let dir = model_dir().expect("set CRANE_AUDIO8_DIR or download the model");
        let mut model = Model::new(dir.to_str().unwrap()).unwrap();
        let opts = SpeechOptions {
            max_new_tokens: 32,
            ..Default::default()
        };
        let (audio, sample_rate) = model
            .generate_speech("Hello from Crane.", "en", None, &opts)
            .unwrap();
        assert_eq!(sample_rate, 44100);
        assert!(audio.elem_count() > 0);
    }
}
