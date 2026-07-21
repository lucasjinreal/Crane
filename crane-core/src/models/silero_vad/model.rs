//! This module implements voice activity detection (VAD) using a pre-trained Silero VAD model.
//! It provides functionalities to segment audio into speech and non-speech segments.
//!
//! If the local model file path is not specified, the model will be downloaded from the Hugging Face Hub by default.

use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    fmt,
    mem::take,
};

/// Result type used by Candle core.
pub use candle_core::Result;
use candle_core::{DType, Device, Error, Tensor, bail, utils};
use candle_onnx::onnx::ModelProto;
use ribo::utils::log;

use crate::utils::select_device;

/// Supported sample rates.
pub const SAMPLE_RATES: [usize; 2] = [8000, 16000];
/// Chunk sizes for 8kHz sample rate.
pub const CHUNKS_SR8K: usize = 256;
/// Chunk sizes for 16kHz sample rate.
pub const CHUNKS_SR16K: usize = 512;

static DEFAULT_MODEL_NAME: &str = "onnx-community/silero-vad";
static DEFAULT_MODEL_FILE: &str = "onnx/model.onnx";
static DEFAULT_SAMPLE_RATE: usize = 16000;
static DEFAULT_SILENCE: usize = 400; // all times are in milliseconds.

/// Configuration for voice activity detection.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Force inference to use CPU.
    pub use_cpu: bool,
    /// Sample rate, default is 16000.
    pub sample_rate: usize,
    /// Minimum speech duration in samples. Segments shorter than this will be discarded.
    pub min_speech: usize,
    /// Maximum speech duration in samples. Segments longer than this will be split.
    pub max_speech: usize,
    /// Minimum silence duration in samples. This value is crucial and affects the granularity of speech segments.
    pub min_silence: usize,
    /// Length of the last silence segment to consider when maximum speech length is reached, in samples.
    pub min_silence_at_max_speech: usize,
    /// Padding duration in samples added before and after each speech segment.
    pub speech_pad: usize,
    /// Speech threshold. Probabilities above this value are considered speech.
    pub threshold: f32,
    /// Hysteresis value for speech detection. Default is 0.15, corresponding to a threshold of 0.5 - 0.15 = 0.35.
    pub hysteresis: f32,
    /// Whether to return timestamps in milliseconds. Default is false, returns sample offsets.
    pub timestamp_offset: bool,
    /// Context size in samples (64 for 16 kHz, 32 for 8 kHz).
    pub context_size: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self::new(DEFAULT_SILENCE, DEFAULT_SAMPLE_RATE)
    }
}

impl VadConfig {
    /// Creates a new `VadConfig` with the given minimum silence duration (in
    /// milliseconds) and sample rate, using default values for all other
    /// fields.
    #[must_use]
    pub fn new(min_silence: usize, sample_rate: usize) -> Self {
        let context_size = if sample_rate == 8000 { 32 } else { 64 };
        VadConfig {
            min_silence,
            sample_rate,
            context_size,
            use_cpu: false,
            threshold: 0.5,
            hysteresis: 0.15,
            min_speech: 250,
            max_speech: 60_000,
            min_silence_at_max_speech: 98,
            speech_pad: min_silence,
            timestamp_offset: false,
        }
    }
}

/// Voice Activity Detection (VAD) struct.
pub struct Vad {
    /// The current configuration.
    pub config: VadConfig,
    /// Sample rate, default is 16000.
    sample_rate: usize,
    /// Chunk size for audio frames, in samples.
    chunk_size: usize,
    /// Minimum speech duration in samples. Segments shorter than this will be discarded.
    min_speech: usize,
    /// Maximum speech duration in samples. Segments longer than this will be split.
    max_speech: usize,
    /// Minimum silence duration in samples. This value is crucial and affects the granularity of speech segments.
    min_silence: usize,
    /// Length of the last silence segment to consider when maximum speech length is reached, in samples.
    min_silence_at_max_speech: usize,
    /// Padding duration in samples added before and after each speech segment.
    speech_pad: usize,
    /// Speech threshold. Probabilities above this value are considered speech.
    threshold: f32,
    /// Hysteresis value for speech detection. Default is 0.15, corresponding to a threshold of 0.5 - 0.15 = 0.35.
    hysteresis: f32,
    /// Whether to return timestamps in milliseconds. Default is false, returns sample offsets.
    timestamp_offset: bool,
    // Context size.
    context_size: usize,
    // Speech threshold in hysteresis state, corresponding to threshold - hysteresis.
    neg_threshold: f32,
    // Processor device for inference.
    device: Device,
    // ONNX inference model.
    model: Option<ModelProto>,
    // Input data vector.
    state: Vec<Tensor>,
    // Is speaking.
    triggered: bool,
    // Current stream end position.
    head: usize,
    // Current stream start position.
    tail: usize,
    // To save potential segment end (and tolerate some silence).
    temp_end: usize,
    // Longest recorded silence gap (position, duration) in the current speech
    // run, for longest-gap max-speech splitting.
    longest_silence_gap: Option<(usize, usize)>,
    // Current active segment start position.
    current_start: usize,
    // Current active segment stop position.
    current_end: usize,
    // Whether if both sides of speech segments padded.
    padded: bool,
    // Start-end pairs of active segments.
    segments: VecDeque<(usize, usize)>,
    // Buffer for remaining audio data.
    buffer: Vec<f32>,
}

// Debug information for Vad.
impl fmt::Debug for Vad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vad")
            .field("config", &self.config)
            .field("sample_rate", &self.sample_rate)
            .field("chunk_size", &self.chunk_size)
            .field("threshold", &self.threshold)
            .field("hysteresis", &self.hysteresis)
            .field("neg_threshold", &self.neg_threshold)
            .field("min_speech", &self.min_speech)
            .field("max_speech", &self.max_speech)
            .field("min_silence", &self.min_silence)
            .field("min_silence_at_max_speech", &self.min_silence_at_max_speech)
            .field("speech_pad", &self.speech_pad)
            .field("timestamp_offset", &self.timestamp_offset)
            .field("context_size", &self.context_size)
            .field("device", &self.device)
            .field("model", &self.model)
            .field("triggered", &self.triggered)
            .field("head", &self.head)
            .field("tail", &self.tail)
            .field("temp_end", &self.temp_end)
            .field("longest_silence_gap", &self.longest_silence_gap)
            .field("current_start", &self.current_start)
            .field("current_end", &self.current_end)
            .field("padded", &self.padded)
            .field("segments", &self.segments.len())
            .field("buffer", &self.buffer.len())
            .finish()
    }
}

impl Default for Vad {
    fn default() -> Self {
        Vad::new(VadConfig::default()).expect("default VadConfig is always valid")
    }
}

impl Vad {
    /// Creates a new `Vad` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if device selection fails.
    pub fn new(config: VadConfig) -> Result<Self> {
        let sr = config.sample_rate;
        let chunk_size = if sr == 8000 {
            CHUNKS_SR8K
        } else {
            CHUNKS_SR16K
        };
        let min_speech = sr * config.min_speech / 1000;
        let speech_pad = sr * config.speech_pad / 1000;
        // Refine max_speech to avoid cutting speech. Saturates instead of
        // underflowing on an invalid config; `verify()` rejects the config
        // before this value is ever used for inference.
        let max_speech = (sr * config.max_speech / 1000)
            .saturating_sub(chunk_size)
            .saturating_sub(2 * speech_pad);
        let min_silence = sr * config.min_silence / 1000;
        let min_silence_at_max_speech = sr * config.min_silence_at_max_speech / 1000;
        let device = select_device(config.use_cpu).map_err(Error::wrap)?;
        let neg_threshold = (config.threshold - config.hysteresis).max(0.01);

        Ok(Vad {
            sample_rate: sr,
            chunk_size,
            min_speech,
            max_speech,
            min_silence,
            min_silence_at_max_speech,
            speech_pad,
            device,
            neg_threshold,
            threshold: config.threshold,
            hysteresis: config.hysteresis,
            context_size: config.context_size,
            timestamp_offset: config.timestamp_offset,
            config,
            model: None,
            state: vec![],
            triggered: false,
            head: 0,
            tail: 0,
            temp_end: 0,
            longest_silence_gap: None,
            current_start: 0,
            current_end: 0,
            padded: true,
            segments: VecDeque::new(),
            buffer: vec![],
        })
    }

    /// Recomputes derived fields (chunk size, padding, thresholds, etc.)
    /// from `self.config` and applies them.
    ///
    /// Validates the config first; if validation fails, `self` is left
    /// unchanged. If `sample_rate` or `context_size` differ from their
    /// current values, this also calls [`Self::reset`], clearing all
    /// internal state (segments, buffer, triggered flag, head/tail
    /// positions).
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid.
    pub fn apply_config(&mut self) -> Result<()> {
        let sr = self.config.sample_rate;
        let chunk_size = if sr == 8000 {
            CHUNKS_SR8K
        } else {
            CHUNKS_SR16K
        };
        let min_speech = sr * self.config.min_speech / 1000;
        let speech_pad = sr * self.config.speech_pad / 1000;
        let max_speech = (sr * self.config.max_speech / 1000)
            .saturating_sub(chunk_size)
            .saturating_sub(2 * speech_pad);
        let min_silence = sr * self.config.min_silence / 1000;
        let min_silence_at_max_speech = sr * self.config.min_silence_at_max_speech / 1000;
        let neg_threshold = (self.config.threshold - self.config.hysteresis).max(0.01);
        let context_size = self.config.context_size;

        // Validate the new values before touching `self`, so a rejected
        // config leaves the instance unchanged instead of half-applied.
        Self::validate(
            sr,
            context_size,
            chunk_size,
            self.config.hysteresis,
            self.config.threshold,
            max_speech,
        )?;

        // `state[2]`'s shape is derived from `context_size`, and `reset()`
        // rebuilds it from `self.sample_rate`/`self.context_size`; reset
        // whenever either changes so that shape never goes stale.
        let needs_reset = self.sample_rate != sr || self.context_size != context_size;

        self.sample_rate = sr;
        self.chunk_size = chunk_size;
        self.min_speech = min_speech;
        self.speech_pad = speech_pad;
        self.max_speech = max_speech;
        self.min_silence = min_silence;
        self.min_silence_at_max_speech = min_silence_at_max_speech;
        self.neg_threshold = neg_threshold;
        self.threshold = self.config.threshold;
        self.hysteresis = self.config.hysteresis;
        self.context_size = context_size;
        self.timestamp_offset = self.config.timestamp_offset;

        if needs_reset {
            self.reset()?;
        }

        Ok(())
    }

    /// Loads the VAD model.
    pub fn load(&mut self, model_file: impl AsRef<str>) -> Result<()> {
        self.verify()?;
        self.reset()?;
        let model_file = model_file.as_ref();
        let model = if model_file.is_empty() {
            let model_file = hf_hub::api::sync::Api::new()
                .and_then(|api| api.model(DEFAULT_MODEL_NAME.into()).get(DEFAULT_MODEL_FILE))
                .map_err(Error::wrap)?;
            candle_onnx::read_file(model_file)?
        } else {
            candle_onnx::read_file(model_file)?
        };
        self.model = Some(model);
        log::info!(
            "silero vad is loaded with capibilites \
      [avx: {}, neon: {}, simd128: {}, f16c: {}]",
            utils::with_avx(),
            utils::with_neon(),
            utils::with_simd128(),
            utils::with_f16c()
        );
        Ok(())
    }

    /// Flushes any remaining audio and returns the segments.
    pub fn flush(&mut self) -> Result<&[(usize, usize)]> {
        let buffer = take(&mut self.buffer);
        if !buffer.is_empty() {
            self.feed_chunk(buffer)?;
        }
        if (self.current_end > 0 || self.current_start > 0)
            && self.head - self.current_start > self.min_speech
        {
            self.current_end = self.head;
            self.push_segment()
        }
        self.finish_padding(true);
        self.padded = true;
        self.triggered = false;
        self.current_start = 0;
        self.current_end = 0;
        self.temp_end = 0;
        self.longest_silence_gap = None;
        self.tail = self.head;
        self.buffer.clear();
        self.state[1] = Tensor::zeros_like(&self.state[1])?;
        self.state[2] = Tensor::zeros_like(&self.state[2])?;

        Ok(self.segments.make_contiguous())
    }

    /// Yields the next available segment.
    pub fn yield_segment(&mut self) -> Option<(usize, usize)> {
        if self.segments.is_empty() {
            return None;
        }
        if self.segments.len() == 1 && !self.padded {
            return None;
        }
        let segment = self.segments.pop_front()?;
        self.tail = segment.1;
        Some(segment)
    }

    /// Segments the provided audio data.
    pub fn segment_audio(&mut self, audio: &[f32]) -> Result<usize> {
        let mut count = 0;
        let chunk_size = self.chunk_size;
        let audio = if !self.buffer.is_empty() {
            let appendix = chunk_size - self.buffer.len();
            log::debug!("Fill previous buffer with {appendix} of {}", audio.len());
            if appendix > audio.len() {
                self.buffer.extend_from_slice(audio);
                return Ok(0);
            }
            self.buffer.extend_from_slice(&audio[..appendix]);
            &audio[appendix..]
        } else {
            audio
        };
        debug_assert!(self.buffer.is_empty() || self.buffer.len() == chunk_size);
        if self.buffer.len() == chunk_size {
            let buffer = take(&mut self.buffer);
            self.feed_chunk(buffer)?;
            count += 1;
        }

        for chunk in audio.chunks(chunk_size) {
            if chunk.len() < chunk_size {
                self.buffer.extend_from_slice(chunk);
                break;
            }
            self.feed_chunk(chunk.to_vec())?;
            count += 1;
        }
        Ok(count)
    }

    /// Returns the list of detected speech segments.
    #[must_use]
    pub fn get_segments(&self) -> Cow<'_, [(usize, usize)]> {
        if self.timestamp_offset {
            Cow::Owned(
                self.segments
                    .iter()
                    .map(|(start, end)| {
                        (
                            start * 1000 / self.sample_rate,
                            end * 1000 / self.sample_rate,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        } else if let (front, []) = self.segments.as_slices() {
            Cow::Borrowed(front)
        } else {
            Cow::Owned(self.segments.iter().copied().collect())
        }
    }

    /// Returns the number of detected speech segments.
    #[must_use]
    pub fn count(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` if there are no active speech segments.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.segments.is_empty() && !self.triggered
    }

    /// Resets the VAD state.
    pub fn reset(&mut self) -> Result<()> {
        #[allow(clippy::cast_possible_wrap)] // 8000 or 16000, well within i64 range
        let sr = Tensor::new(self.sample_rate as i64, &self.device)?;
        let previous = Tensor::zeros((2, 1, 128), DType::F32, &self.device)?;
        let context = Tensor::zeros((1, self.context_size), DType::F32, &self.device)?;
        self.state = vec![sr, previous, context];
        self.triggered = false;
        self.head = 0;
        self.tail = 0;
        self.temp_end = 0;
        self.longest_silence_gap = None;
        self.current_start = 0;
        self.current_end = 0;
        self.padded = true;
        self.segments.clear();
        self.buffer.clear();
        Ok(())
    }

    /// Sets the offset for the audio data.
    pub fn set_offset(&mut self, offset: usize) {
        self.tail = offset;
        self.head = offset.max(self.head);
    }

    fn validate(
        sample_rate: usize,
        context_size: usize,
        chunk_size: usize,
        hysteresis: f32,
        threshold: f32,
        max_speech: usize,
    ) -> Result<()> {
        if !SAMPLE_RATES.contains(&sample_rate) {
            bail!(
                "invalid sample rate: {}, only support 8000 or 16000.",
                sample_rate
            );
        }

        if context_size == 0 || context_size >= chunk_size {
            bail!(
                "invalid context_size: {}, must be greater than 0 and less than chunk_size ({})",
                context_size,
                chunk_size
            );
        }

        if hysteresis < 0.0 || hysteresis > threshold {
            bail!(
                "invalid hysteresis: {}, must be within [0, threshold={}]",
                hysteresis,
                threshold
            );
        }

        // `max_speech` is already the saturating-subtracted value computed in
        // `new()`/`apply_config()`; checking it directly (instead of
        // re-deriving the same arithmetic here) means this can't drift out of
        // sync with that computation.
        if max_speech == 0 {
            bail!(
                "invalid max_speech: computed 0 samples after subtracting chunk_size and speech_pad padding, increase max_speech or decrease speech_pad"
            );
        }

        Ok(())
    }

    fn verify(&self) -> Result<()> {
        Self::validate(
            self.sample_rate,
            self.context_size,
            self.chunk_size,
            self.hysteresis,
            self.threshold,
            self.max_speech,
        )
    }

    #[inline]
    fn model(&self) -> Result<&ModelProto> {
        self.model.as_ref().ok_or_else(|| {
            Error::Msg("VAD model is not loaded, call `load` method first.".into()).bt()
        })
    }

    fn inputs(&self, chunk: Tensor) -> HashMap<String, Tensor> {
        HashMap::from([
            ("input".to_string(), chunk),
            ("sr".to_string(), self.state[0].clone()),
            ("state".to_string(), self.state[1].clone()),
        ])
    }

    fn update_state(&mut self, output: &Tensor, context: Tensor) {
        debug_assert_eq!(output.dims(), &[2, 1, 128]);
        self.state[1] = output.clone();
        self.state[2] = context;
    }

    // Determine voice activity probability for a single chunk of audio.
    fn feed_chunk(&mut self, mut chunk: Vec<f32>) -> Result<f32> {
        self.head += chunk.len();
        if chunk.len() < self.chunk_size {
            chunk.resize(self.chunk_size, 0f32);
        } else {
            chunk.truncate(self.chunk_size);
        }
        let next_context = Tensor::from_slice(
            &chunk[self.chunk_size - self.context_size..],
            (1, self.context_size),
            &self.device,
        )?;
        let chunk = Tensor::from_vec(chunk, (1, self.chunk_size), &self.device)?;
        let chunk = Tensor::cat(&[&self.state[2], &chunk], 1)?;
        let model = self.model()?;
        let out = candle_onnx::simple_eval(model, self.inputs(chunk))?;
        let out_names = &model
            .graph
            .as_ref()
            .ok_or_else(|| Error::Msg("VAD model has no graph".into()).bt())?
            .output;
        if out_names.len() < 2 {
            bail!(
                "VAD model graph has {} output(s), expected at least 2",
                out_names.len()
            );
        }
        let output = out
            .get(&out_names[0].name)
            .ok_or_else(|| Error::Msg(format!("missing VAD output '{}'", out_names[0].name)).bt())?
            .clone();
        let state_output = out.get(&out_names[1].name).ok_or_else(|| {
            Error::Msg(format!("missing VAD output '{}'", out_names[1].name)).bt()
        })?;
        self.update_state(state_output, next_context);

        let output = output.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(output.len(), 1);
        let output = output[0];
        log::info!("vad chunk prediction: {output}");
        // Make segmentation;
        self.make_segment(output);
        Ok(output)
    }

    fn make_segment(&mut self, prob: f32) {
        let offset = self.head;
        // Straightforward detection.
        if prob >= self.threshold {
            if self.temp_end > 0 {
                let sil_dur = offset - self.temp_end;
                if sil_dur > self.min_silence_at_max_speech
                    && self
                        .longest_silence_gap
                        .is_none_or(|(_, best_dur)| sil_dur > best_dur)
                {
                    self.longest_silence_gap = Some((self.temp_end, sil_dur));
                }
                self.temp_end = 0;
            }
            if !self.triggered {
                self.finish_padding(true);
                self.triggered = true;
                self.current_start = offset;
                return;
            }
        }
        // Maximum active segment size reached.
        if self.triggered && offset - self.current_start > self.max_speech {
            log::debug!("max speech reached: {}", offset - self.current_start);
            if let Some((end_pos, dur)) = self.longest_silence_gap {
                // Use the longest recorded silence gap in the current speech run.
                self.current_end = end_pos;
                self.push_segment();
                let next = end_pos + dur;
                if next < end_pos + offset {
                    // previously reached silence (< neg_thres) and is still not speech (< thres)
                    self.current_start = next;
                } else {
                    self.triggered = false;
                }
                self.temp_end = 0;
                self.longest_silence_gap = None;
            } else {
                self.current_end = offset;
                self.push_segment();
                self.temp_end = 0;
                self.longest_silence_gap = None;
                self.triggered = false;
                return;
            }
        }

        // Silence detected.
        if self.triggered && prob < self.neg_threshold {
            if self.temp_end == 0 {
                self.temp_end = offset;
            }
            if offset - self.temp_end < self.min_silence {
                return;
            } else {
                self.current_end = self.temp_end;
                if self.current_end - self.current_start > self.min_speech {
                    self.push_segment();
                }
                self.current_start = 0;
                self.current_end = 0;
                self.temp_end = 0;
                self.longest_silence_gap = None;
                self.triggered = false;
                return;
            }
        }

        // State unchanged, try padding.
        self.finish_padding(false);
    }

    fn finish_padding(&mut self, triggering: bool) {
        if self.padded || (self.triggered && !triggering) {
            return;
        }
        if let Some(s) = self.segments.back_mut() {
            let silence = self.head - s.1;
            let pad = if silence > self.speech_pad * 2 {
                self.speech_pad
            } else if triggering {
                silence / 2
            } else {
                return;
            };
            s.1 += pad;
            self.padded = true;
        }
    }

    fn push_segment(&mut self) {
        debug_assert!(self.current_start < self.current_end);
        let mut start = self.current_start;
        if let Some(&(_, last)) = self.segments.back() {
            start = if start > last + self.speech_pad {
                start - self.speech_pad
            } else {
                last
            };
        } else {
            start = if start > self.tail + self.speech_pad {
                start - self.speech_pad
            } else {
                self.tail
            };
        }
        self.segments.push_back((start, self.current_end));
        self.current_start = 0;
        self.current_end = 0;
        self.padded = false;
    }
}

/// Audio segment information.
#[derive(Default, Clone)]
pub struct Segment {
    /// Position of the segment start in samples.
    pub position: usize,
    /// Timestamp of the segment start in milliseconds.
    pub timestamp: usize,
    /// Duration of the segment in milliseconds.
    pub duration: usize,
    /// Audio data of the segment as 32-bit float samples.
    pub audio: Vec<f32>,
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("position", &self.position)
            .field("timestamp", &self.timestamp)
            .field("duration", &self.duration)
            .field("audio", &self.audio.len())
            .finish()
    }
}

impl Segment {
    /// Creates a new `Segment` from audio data.
    #[must_use]
    pub fn from_audio(audio: Vec<f32>, offset: usize, sample_rate: usize) -> Self {
        Segment {
            position: offset,
            timestamp: offset * 1000 / sample_rate,
            duration: audio.len() * 1000 / sample_rate,
            audio,
        }
    }

    /// Creates a new empty `Segment` with the specified offset and sample count.
    #[must_use]
    pub fn new(offset: usize, samples: usize, sample_rate: usize) -> Self {
        Segment {
            audio: vec![],
            position: offset,
            timestamp: offset * 1000 / sample_rate,
            duration: samples * 1000 / sample_rate,
        }
    }
}

/// Audio buffer for storing audio data.
pub struct AudioBuffer {
    queue: VecDeque<Vec<f32>>,
    length: usize,
    start: usize,
    offset: usize,
    sample_rate: usize,
}

impl AudioBuffer {
    /// Creates a new `AudioBuffer` with the specified sample rate.
    #[must_use]
    pub fn new(sample_rate: usize) -> Self {
        AudioBuffer {
            queue: VecDeque::new(),
            length: 0,
            start: 0,
            offset: 0,
            sample_rate,
        }
    }

    /// Sets the start position for the audio buffer.
    pub fn start_point(&mut self, start_position: usize) {
        self.start = start_position;
    }

    /// Returns the length of the audio data.
    #[must_use]
    pub fn audio_length(&self) -> usize {
        self.length - self.offset
    }

    /// Inputs audio data into the buffer.
    pub fn input(&mut self, audio: Vec<f32>) {
        self.length += audio.len();
        self.queue.push_back(audio);
    }

    /// Outputs a segment of audio data from the buffer.
    pub fn output(&mut self, from: usize, to: usize) -> Option<Segment> {
        if self.queue.is_empty()
            || from < self.start + self.offset
            || to > self.start + self.length
            || to < from
        {
            return None;
        }
        let chunk_size = to - from;
        let mut audio = Vec::with_capacity(chunk_size);
        let mut filled = 0;
        while !self.queue.is_empty() {
            let first_len = self.queue[0].len();
            if self.start + first_len < from {
                self.start += first_len;
                self.offset = 0;
                self.queue.pop_front();
                self.length -= first_len;
                continue;
            }
            let avlen = first_len - self.offset;
            let rest_len = chunk_size - filled;
            if avlen >= rest_len {
                audio.extend_from_slice(&self.queue[0][self.offset..self.offset + rest_len]);
                self.offset += rest_len;
                filled = chunk_size;
                break;
            } else {
                audio.extend_from_slice(&self.queue[0][self.offset..]);
                filled += avlen;
                self.queue.pop_front();
                self.start += first_len;
                self.length -= first_len;
                self.offset = 0;
            }
        }
        debug_assert!(chunk_size == audio.len());
        if filled != chunk_size {
            return None;
        }
        Some(Segment::from_audio(audio, from, self.sample_rate))
    }

    /// Clears the buffer, keeping the specified amount of data.
    pub fn clear(&mut self, keep: usize) -> Option<Segment> {
        if self.length - self.offset <= keep {
            return None;
        }
        let position = self.start + self.offset;
        while !self.queue.is_empty() {
            let first_len = self.queue[0].len();
            if self.length + self.offset - first_len > keep {
                self.start += first_len;
                self.offset = 0;
                self.queue.pop_front();
                self.length -= first_len;
                continue;
            }
            self.offset = self.length - keep;
            break;
        }
        let chunk = self.start + self.offset - position;
        Some(Segment::new(position, chunk, self.sample_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_config() -> VadConfig {
        let mut config = VadConfig::new(400, 16000);
        config.use_cpu = true;
        config
    }

    #[test]
    fn test_verify_default_config() {
        let mut vad = Vad::new(cpu_config()).unwrap();
        assert!(vad.apply_config().is_ok());
    }

    #[test]
    fn test_verify_rejects_invalid_sample_rate() {
        let mut config = cpu_config();
        config.sample_rate = 44100;
        let mut vad = Vad::new(config).unwrap();
        assert!(vad.apply_config().is_err());
    }

    #[test]
    fn test_verify_rejects_context_size_zero() {
        let mut config = cpu_config();
        config.context_size = 0;
        let mut vad = Vad::new(config).unwrap();
        assert!(vad.apply_config().is_err());
    }

    #[test]
    fn test_verify_rejects_context_size_ge_chunk() {
        let mut config = cpu_config();
        config.context_size = CHUNKS_SR16K;
        let mut vad = Vad::new(config).unwrap();
        assert!(vad.apply_config().is_err());
    }

    #[test]
    fn test_verify_rejects_hysteresis_above_threshold() {
        let mut config = cpu_config();
        config.hysteresis = config.threshold + 0.1;
        let mut vad = Vad::new(config).unwrap();
        assert!(vad.apply_config().is_err());
    }

    #[test]
    fn test_verify_rejects_negative_hysteresis() {
        let mut config = cpu_config();
        config.hysteresis = -0.1;
        let mut vad = Vad::new(config).unwrap();
        assert!(vad.apply_config().is_err());
    }

    #[test]
    fn test_verify_rejects_small_max_speech() {
        let mut config = cpu_config();
        config.max_speech = 1;
        let mut vad = Vad::new(config).unwrap();
        assert!(vad.apply_config().is_err());
    }

    #[test]
    fn test_apply_config_no_mutation_on_failure() {
        let mut vad = Vad::new(cpu_config()).unwrap();
        vad.apply_config().unwrap();
        let good_max_speech = vad.max_speech;

        vad.config.max_speech = 1;
        assert!(vad.apply_config().is_err());
        assert_eq!(vad.max_speech, good_max_speech);
    }

    #[test]
    fn test_make_segment_cuts_at_longest_silence_gap() {
        // max_speech (100ms) - chunk_size (512) - 2*speech_pad (0) = 1088 samples.
        let mut config = VadConfig::new(50, 16000);
        config.use_cpu = true;
        config.speech_pad = 0;
        config.min_silence_at_max_speech = 10;
        config.max_speech = 100;
        let mut vad = Vad::new(config).unwrap();

        // Trigger speech at offset 100.
        vad.head = 100;
        vad.make_segment(0.9);
        assert!(vad.triggered);
        assert_eq!(vad.current_start, 100);

        // Longer silence gap: 300 -> 700 (duration 400).
        vad.head = 300;
        vad.make_segment(0.1);
        vad.head = 700;
        vad.make_segment(0.9);
        assert_eq!(vad.longest_silence_gap, Some((300, 400)));

        // Shorter silence gap recorded afterwards: 900 -> 1100 (duration 200).
        // Must not replace the already-recorded longer gap.
        vad.head = 900;
        vad.make_segment(0.1);
        vad.head = 1100;
        vad.make_segment(0.9);
        assert_eq!(vad.longest_silence_gap, Some((300, 400)));

        // Push past max_speech (1088 samples from current_start=100),
        // triggering the max-speech cut. It must land on the longer gap
        // (300), not the shorter, more recent one (900).
        vad.head = 1200;
        vad.make_segment(0.9);

        assert_eq!(vad.segments.make_contiguous(), [(100, 300)]);
        assert_eq!(vad.current_start, 700);
        assert!(vad.triggered);
    }

    /// `Vad` with small, easy-to-reason-about sample counts for exercising
    /// the state machine: `min_speech`=320, `speech_pad`=0, `min_silence`=800,
    /// `min_silence_at_max_speech`=160, `max_speech`=1088 (all in samples).
    fn segment_test_vad() -> Vad {
        let mut config = VadConfig::new(50, 16000);
        config.use_cpu = true;
        config.speech_pad = 0;
        config.min_silence_at_max_speech = 10;
        config.max_speech = 100;
        config.min_speech = 20;
        Vad::new(config).unwrap()
    }

    #[test]
    fn test_vad_config_new_16khz() {
        let config = VadConfig::new(400, 16000);
        assert_eq!(config.context_size, 64);
        assert_eq!(config.speech_pad, 400);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.min_silence, 400);
        assert!((config.threshold - 0.5).abs() < 1e-6);
        assert!((config.hysteresis - 0.15).abs() < 1e-6);
        assert_eq!(config.min_speech, 250);
        assert_eq!(config.max_speech, 60_000);
        assert_eq!(config.min_silence_at_max_speech, 98);
        assert!(!config.use_cpu);
        assert!(!config.timestamp_offset);
    }

    #[test]
    fn test_vad_config_new_8khz() {
        let config = VadConfig::new(400, 8000);
        assert_eq!(config.context_size, 32);
    }

    #[test]
    fn test_vad_config_default() {
        let default = VadConfig::default();
        let expected = VadConfig::new(400, 16000);
        assert_eq!(default.sample_rate, expected.sample_rate);
        assert_eq!(default.min_silence, expected.min_silence);
        assert_eq!(default.context_size, expected.context_size);
        assert_eq!(default.speech_pad, expected.speech_pad);
        assert_eq!(default.min_speech, expected.min_speech);
        assert_eq!(default.max_speech, expected.max_speech);
        assert_eq!(
            default.min_silence_at_max_speech,
            expected.min_silence_at_max_speech
        );
    }

    #[test]
    fn test_vad_new_derived_fields() {
        let vad = Vad::new(cpu_config()).unwrap();
        assert_eq!(vad.chunk_size, CHUNKS_SR16K);
        assert_eq!(vad.min_speech, 4000);
        assert_eq!(vad.speech_pad, 6400);
        assert_eq!(vad.min_silence, 6400);
        assert_eq!(vad.min_silence_at_max_speech, 1568);
        assert_eq!(vad.max_speech, 960_000 - 512 - 12_800);
        assert!((vad.neg_threshold - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_vad_new_neg_threshold_clamped() {
        let mut config = cpu_config();
        config.hysteresis = 0.9; // exceeds threshold; would go negative without the clamp.
        let vad = Vad::new(config).unwrap();
        assert!((vad.neg_threshold - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_make_segment_speech_onset() {
        let mut vad = segment_test_vad();
        vad.head = 100;
        vad.make_segment(0.9);
        assert!(vad.triggered);
        assert_eq!(vad.current_start, 100);
    }

    #[test]
    fn test_make_segment_silence_ends_segment() {
        let mut vad = segment_test_vad();
        vad.max_speech = 5000; // large enough that this timeline never trips the max-speech cut
        vad.head = 100;
        vad.make_segment(0.9);
        assert!(vad.triggered);

        vad.head = 500;
        vad.make_segment(0.1);
        assert!(vad.triggered); // within the silence-tolerance window

        vad.head = 1300;
        vad.make_segment(0.1);
        assert!(!vad.triggered);
        assert_eq!(vad.segments.make_contiguous(), [(100, 500)]);
    }

    #[test]
    fn test_make_segment_short_speech_discarded() {
        let mut vad = segment_test_vad();
        vad.head = 100;
        vad.make_segment(0.9);

        vad.head = 150;
        vad.make_segment(0.1);
        vad.head = 950;
        vad.make_segment(0.1);

        assert!(!vad.triggered);
        assert!(vad.segments.is_empty());
    }

    #[test]
    fn test_make_segment_short_silence_tolerated() {
        let mut vad = segment_test_vad();
        vad.head = 100;
        vad.make_segment(0.9);

        vad.head = 300;
        vad.make_segment(0.1);
        assert!(vad.triggered);

        vad.head = 350;
        vad.make_segment(0.9);
        assert!(vad.triggered);
        assert_eq!(vad.current_start, 100);
        assert!(vad.segments.is_empty());
        assert_eq!(vad.temp_end, 0);
    }

    #[test]
    fn test_make_segment_max_speech_hard_cut() {
        let mut vad = segment_test_vad();
        vad.head = 0;
        vad.make_segment(0.9);
        assert!(vad.triggered);

        vad.head = 1089;
        vad.make_segment(0.9);

        assert!(!vad.triggered);
        assert_eq!(vad.segments.make_contiguous(), [(0, 1089)]);
    }

    #[test]
    fn test_make_segment_max_speech_untriggers_on_large_gap() {
        let mut vad = segment_test_vad();
        vad.triggered = true;
        vad.current_start = 0;
        vad.longest_silence_gap = Some((50, 2000));
        vad.head = 1089;

        vad.make_segment(0.9);

        assert!(!vad.triggered);
        assert_eq!(vad.segments.make_contiguous(), [(0, 50)]);
    }

    #[test]
    fn test_push_segment_start_padding() {
        let mut vad = segment_test_vad();
        vad.speech_pad = 100;
        vad.current_start = 500;
        vad.current_end = 600;
        vad.push_segment();
        assert_eq!(vad.segments.make_contiguous(), [(400, 600)]);
        assert_eq!(vad.current_start, 0);
        assert_eq!(vad.current_end, 0);
        assert!(!vad.padded);
    }

    #[test]
    fn test_push_segment_clamped_to_tail() {
        let mut vad = segment_test_vad();
        vad.speech_pad = 100;
        vad.tail = 450;
        vad.current_start = 500;
        vad.current_end = 600;
        vad.push_segment();
        assert_eq!(vad.segments.make_contiguous(), [(450, 600)]);
    }

    #[test]
    fn test_push_segment_clamped_to_prev_segment() {
        let mut vad = segment_test_vad();
        vad.speech_pad = 100;
        vad.segments.push_back((0, 500));
        vad.current_start = 550;
        vad.current_end = 700;
        vad.push_segment();
        assert_eq!(vad.segments.make_contiguous(), [(0, 500), (500, 700)]);
    }

    #[test]
    fn test_finish_padding_end_pad() {
        let mut vad = segment_test_vad();
        vad.speech_pad = 50;
        vad.segments.push_back((0, 200));
        vad.padded = false;
        vad.head = 200 + 101; // silence (101) > 2 * speech_pad (100)
        vad.finish_padding(false);
        assert_eq!(vad.segments.back().unwrap().1, 250);
        assert!(vad.padded);
    }

    #[test]
    fn test_finish_padding_half_on_trigger() {
        let mut vad = segment_test_vad();
        vad.speech_pad = 50;
        vad.segments.push_back((0, 200));
        vad.padded = false;
        vad.head = 200 + 40; // silence (40) <= 2 * speech_pad (100)
        vad.finish_padding(true);
        assert_eq!(vad.segments.back().unwrap().1, 220);
        assert!(vad.padded);
    }

    #[test]
    fn test_finish_padding_noop_when_padded() {
        let mut vad = segment_test_vad();
        vad.segments.push_back((0, 200));
        vad.padded = true;
        vad.head = 500;
        vad.finish_padding(true);
        assert_eq!(vad.segments.back().unwrap().1, 200);
    }

    #[test]
    fn test_yield_segment_empty() {
        let mut vad = segment_test_vad();
        assert_eq!(vad.yield_segment(), None);
    }

    #[test]
    fn test_yield_segment_single_unpadded() {
        let mut vad = segment_test_vad();
        vad.segments.push_back((0, 100));
        vad.padded = false;
        assert_eq!(vad.yield_segment(), None);
    }

    #[test]
    fn test_yield_segment_single_padded() {
        let mut vad = segment_test_vad();
        vad.segments.push_back((0, 100));
        vad.padded = true;
        assert_eq!(vad.yield_segment(), Some((0, 100)));
        assert_eq!(vad.tail, 100);
    }

    #[test]
    fn test_yield_segment_multiple() {
        let mut vad = segment_test_vad();
        vad.segments.push_back((0, 100));
        vad.segments.push_back((100, 200));
        vad.padded = false;
        assert_eq!(vad.yield_segment(), Some((0, 100)));
        assert_eq!(vad.tail, 100);
    }

    #[test]
    fn test_get_segments_sample_offsets() {
        let mut vad = segment_test_vad();
        vad.segments.push_back((100, 200));
        vad.segments.push_back((300, 400));
        assert_eq!(
            vad.get_segments().into_owned(),
            vec![(100, 200), (300, 400)]
        );
    }

    #[test]
    fn test_get_segments_timestamp_offsets() {
        let mut vad = segment_test_vad();
        vad.timestamp_offset = true;
        vad.segments.push_back((16000, 32000));
        assert_eq!(vad.get_segments().into_owned(), vec![(1000, 2000)]);
    }

    #[test]
    fn test_count_and_is_idle() {
        let mut vad = segment_test_vad();
        assert_eq!(vad.count(), 0);
        assert!(vad.is_idle());

        vad.triggered = true;
        assert!(!vad.is_idle());
        vad.triggered = false;

        vad.segments.push_back((0, 100));
        assert_eq!(vad.count(), 1);
        assert!(!vad.is_idle());
    }

    #[test]
    fn test_set_offset() {
        let mut vad = segment_test_vad();
        vad.head = 500;
        vad.set_offset(200);
        assert_eq!(vad.tail, 200);
        assert_eq!(vad.head, 500);

        vad.set_offset(800);
        assert_eq!(vad.tail, 800);
        assert_eq!(vad.head, 800);
    }

    #[test]
    fn test_segment_from_audio() {
        let audio = vec![0.0f32; 800];
        let segment = Segment::from_audio(audio.clone(), 1600, 16000);
        assert_eq!(segment.position, 1600);
        assert_eq!(segment.timestamp, 100);
        assert_eq!(segment.duration, 50);
        assert_eq!(segment.audio, audio);
    }

    #[test]
    fn test_segment_new() {
        let segment = Segment::new(1600, 800, 16000);
        assert_eq!(segment.position, 1600);
        assert_eq!(segment.timestamp, 100);
        assert_eq!(segment.duration, 50);
        assert!(segment.audio.is_empty());
    }

    #[test]
    fn test_segment_from_audio_8khz() {
        let audio = vec![0.0f32; 400];
        let segment = Segment::from_audio(audio, 800, 8000);
        assert_eq!(segment.timestamp, 100);
        assert_eq!(segment.duration, 50);
    }

    #[test]
    fn test_audio_buffer_input_and_length() {
        let mut buf = AudioBuffer::new(16000);
        buf.input(vec![0.0; 100]);
        buf.input(vec![0.0; 50]);
        assert_eq!(buf.audio_length(), 150);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_audio_buffer_output_basic() {
        let mut buf = AudioBuffer::new(16000);
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        buf.input(data.clone());
        let segment = buf.output(0, 100).unwrap();
        assert_eq!(segment.audio, data);
        assert_eq!(segment.position, 0);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_audio_buffer_output_cross_chunk() {
        let mut buf = AudioBuffer::new(16000);
        let a: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let b: Vec<f32> = (50..100).map(|i| i as f32).collect();
        buf.input(a);
        buf.input(b);
        let segment = buf.output(0, 75).unwrap();
        let expected: Vec<f32> = (0..75).map(|i| i as f32).collect();
        assert_eq!(segment.audio, expected);
        assert_eq!(segment.position, 0);
    }

    #[test]
    fn test_audio_buffer_output_none_empty() {
        let mut buf = AudioBuffer::new(16000);
        assert!(buf.output(0, 10).is_none());
    }

    #[test]
    fn test_audio_buffer_output_none_out_of_bounds() {
        let mut buf = AudioBuffer::new(16000);
        buf.input(vec![0.0; 100]);
        assert!(buf.output(0, 200).is_none());
    }

    #[test]
    fn test_audio_buffer_output_none_reversed_range() {
        let mut buf = AudioBuffer::new(16000);
        buf.input(vec![0.0; 100]);
        assert!(buf.output(50, 10).is_none());
    }

    #[test]
    fn test_audio_buffer_clear() {
        let mut buf = AudioBuffer::new(16000);
        buf.input(vec![0.0; 100]);
        let segment = buf.clear(40).unwrap();
        assert_eq!(segment.position, 0);
        assert_eq!(segment.duration, 60 * 1000 / 16000);
        assert_eq!(buf.audio_length(), 40);
    }

    #[test]
    fn test_audio_buffer_clear_nothing_to_discard() {
        let mut buf = AudioBuffer::new(16000);
        buf.input(vec![0.0; 40]);
        assert!(buf.clear(40).is_none());
    }

    #[test]
    fn test_audio_buffer_start_point() {
        let mut buf = AudioBuffer::new(16000);
        buf.start_point(500);
        buf.input(vec![0.0; 10]);
        let segment = buf.output(500, 510).unwrap();
        assert_eq!(segment.position, 500);
    }
}
