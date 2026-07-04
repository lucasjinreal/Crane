//! Protocol-independent model runtime.
//!
//! [`ModelRuntime`] owns the full model lifecycle (LLM engine, TTS models)
//! independently of any transport (HTTP, Wyoming). Consumers load models via
//! [`ModelRuntime::load_tts`], then send generation requests through the
//! returned [`TtsHandle`]. Each TTS model runs on its own dedicated OS thread
//! and is addressed through channels, so [`ModelRuntime`] can be shared via
//! `Arc` across async tasks without locking.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot};

use crane_core::generation::SpeechOptions;

use crate::audio::tts::{AudioInfo, Tts, VoiceInfo};
#[cfg(feature = "tts-cache")]
use crate::engine::cache::{CacheKey, TtsCache};
use crate::engine::model_factory::{self, ModelType};
use crate::engine::types::EngineHandle;
use crate::engine::vlm_types::{Gemma4VlmRequest, VlmRequest};
use crate::engine::{backend::ModelBackend, InferenceEngine, MemoryConfig};

/// A request to generate speech, sent to a TTS model's dedicated thread.
///
/// This type is transport-agnostic: it carries no HTTP- or Wyoming-specific
/// fields. The response is a raw f32 PCM [`Tensor`]; encoding to WAV, PCM
/// bytes, or any other wire format is the caller's responsibility.
pub struct TtsGenerateRequest {
    /// Text to synthesize.
    pub text: String,
    /// Target language (e.g. "english", "auto").
    pub language: String,
    /// Voice name from [`TtsHandle::voices`], or `None` for the model default.
    pub voice: Option<String>,
    /// Generation parameters (temperature, max tokens, etc.).
    pub opts: SpeechOptions,
    /// Reference audio path for voice cloning. `None` for a predefined voice.
    pub reference_audio: Option<PathBuf>,
    /// Transcript of the reference audio (required by some models).
    pub reference_text: Option<String>,
    /// Channel to send back the generated audio tensor.
    pub response_tx: oneshot::Sender<Result<Tensor>>,
}

/// Handle to a TTS model running on its dedicated thread.
///
/// Cloneable metadata (audio format, voices) is queried once at load time,
/// before the model is moved to its thread, so it can be read without
/// blocking on the generation queue.
pub struct TtsHandle {
    tx: mpsc::UnboundedSender<TtsGenerateRequest>,
    audio_info: AudioInfo,
    voices: Vec<VoiceInfo>,
    supports_voice_cloning: bool,
    model_type_name: &'static str,
    pending_count: Arc<AtomicU64>,
    /// Identity used in TTS cache keys. Defaults to the registration name,
    /// but [`ModelRuntime::load_tts`] overrides it to the full model path so
    /// two directories that merely share a final path component (different
    /// checkpoint, dtype, or quantization) don't collide in the cache.
    cache_model_id: String,
}

impl TtsHandle {
    /// Returns the audio format this model produces.
    #[must_use]
    pub fn audio_info(&self) -> AudioInfo {
        self.audio_info
    }

    /// Returns the voices available for service discovery.
    #[must_use]
    pub fn voices(&self) -> &[VoiceInfo] {
        &self.voices
    }

    /// Returns true if this model supports voice cloning from reference audio.
    #[must_use]
    pub fn supports_voice_cloning(&self) -> bool {
        self.supports_voice_cloning
    }

    /// Returns the model type name (e.g. "`qwen3_tts`", "`voxtral_tts`").
    #[must_use]
    pub fn model_type_name(&self) -> &'static str {
        self.model_type_name
    }

    /// Returns the number of requests currently queued or being processed.
    #[must_use]
    pub fn pending_count(&self) -> u64 {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Send a generation request to the model's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's thread has stopped.
    pub fn send(&self, req: TtsGenerateRequest) -> Result<()> {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(req).map_err(|_| {
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            anyhow::anyhow!("TTS thread has stopped")
        })
    }
}

/// Protocol-independent model runtime.
///
/// Owns the LLM continuous-batching engine handle (if any) and all loaded
/// TTS models, keyed by name. crane-serve and crane-wyoming build one
/// `ModelRuntime` at startup and share it via `Arc`.
pub struct ModelRuntime {
    engine: Option<EngineHandle>,
    tts: HashMap<String, TtsHandle>,
    default_tts: Option<String>,
    #[cfg(feature = "tts-cache")]
    tts_cache: Option<Arc<TtsCache>>,
    vlm_tx: Option<mpsc::UnboundedSender<VlmRequest>>,
    gemma4_vlm_tx: Option<mpsc::UnboundedSender<Gemma4VlmRequest>>,
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: Vec<u32>,
    model_name: String,
    model_type: ModelType,
    dtype_name: String,
    device_name: String,
}

impl ModelRuntime {
    /// Create an empty runtime with the given metadata.
    #[must_use]
    pub fn new(
        model_name: String,
        model_type: ModelType,
        dtype_name: String,
        device_name: String,
        tokenizer: tokenizers::Tokenizer,
        eos_token_id: Vec<u32>,
    ) -> Self {
        Self {
            engine: None,
            tts: HashMap::new(),
            default_tts: None,
            #[cfg(feature = "tts-cache")]
            tts_cache: None,
            vlm_tx: None,
            gemma4_vlm_tx: None,
            tokenizer,
            eos_token_id,
            model_name,
            model_type,
            dtype_name,
            device_name,
        }
    }

    /// Enable disk caching for TTS responses.
    ///
    /// Once set, [`generate_speech`](Self::generate_speech) checks the
    /// cache before dispatching to a model's thread and returns a cached
    /// waveform on a hit. Disabled by default. Requests with
    /// `reference_audio` set (voice cloning) always bypass the cache.
    #[cfg(feature = "tts-cache")]
    pub fn set_tts_cache(&mut self, cache: TtsCache) {
        self.tts_cache = Some(Arc::new(cache));
    }

    /// Register an already-constructed TTS model under `name`.
    ///
    /// Queries audio format, voices, and voice-cloning support from the
    /// model before moving it to a dedicated thread. This ordering lets
    /// tests inject a mock [`Tts`] implementation without touching disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's dedicated thread fails to spawn.
    pub fn register_tts(
        &mut self,
        name: String,
        model_type_name: &'static str,
        tts: Box<dyn Tts + Send>,
    ) -> Result<()> {
        let audio_info = tts.audio_info();
        let voices = tts.voices();
        let supports_voice_cloning = tts.supports_voice_cloning();
        let pending_count = Arc::new(AtomicU64::new(0));

        let (tx, rx) = mpsc::unbounded_channel::<TtsGenerateRequest>();

        let thread_name = format!("tts-{name}");
        let log_name = name.clone();
        let thread_pending_count = Arc::clone(&pending_count);
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || run_tts_thread(rx, tts, &log_name, &thread_pending_count))
            .map_err(|e| anyhow::anyhow!("Failed to spawn TTS thread: {e}"))?;

        if self.default_tts.is_none() {
            self.default_tts = Some(name.clone());
        }

        self.tts.insert(
            name.clone(),
            TtsHandle {
                tx,
                audio_info,
                voices,
                supports_voice_cloning,
                model_type_name,
                pending_count,
                cache_model_id: name,
            },
        );
        Ok(())
    }

    /// Load a TTS model from disk and register it.
    ///
    /// Detects the model type from `model_path`, constructs the model, and
    /// registers it under a name derived from the path's final component.
    /// Returns the registration name.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load.
    pub fn load_tts(&mut self, model_path: &str, device: &Device, dtype: &DType) -> Result<String> {
        let resolved_type = model_factory::resolve(self.model_type, model_path);
        let tts = model_factory::create_tts(resolved_type, model_path, device, dtype)?;
        let name = extract_model_name(model_path);

        tracing::info!(
            name = %name,
            model_type = %resolved_type.display_name(),
            "TTS model loaded, spawning thread",
        );

        self.register_tts(name.clone(), resolved_type.display_name(), tts)?;
        // Cache keys should discriminate by the full on-disk path, not just
        // its final component -- two directories with the same file name
        // (different checkpoint, dtype, or quantization) must not collide.
        if let Some(handle) = self.tts.get_mut(&name) {
            handle.cache_model_id = model_path.to_string();
        }
        Ok(name)
    }

    /// Returns the TTS handle registered under `name`, if any.
    #[must_use]
    pub fn tts_handle(&self, name: &str) -> Option<&TtsHandle> {
        self.tts.get(name)
    }

    /// Returns an arbitrary loaded TTS handle.
    ///
    /// Useful when only one TTS model is loaded (the common case for a
    /// Wyoming server started with a single `--model` flag).
    #[must_use]
    pub fn default_tts_handle(&self) -> Option<&TtsHandle> {
        self.default_tts.as_ref().and_then(|name| self.tts.get(name))
    }

    /// Returns the registration name of the default TTS model, if any.
    #[must_use]
    pub fn default_tts_name(&self) -> Option<&str> {
        self.default_tts.as_deref()
    }

    /// Dispatch a TTS request, consulting the cache first if one is configured.
    ///
    /// This is the preferred entry point for TTS generation over calling
    /// [`TtsHandle::send`] directly:
    ///
    /// 1. If no cache is configured, delegates straight to `TtsHandle::send`.
    /// 2. If `req.reference_audio` is set (voice cloning), delegates
    ///    straight to `TtsHandle::send` -- voice-cloned audio is never
    ///    cached (see [`crate::engine::cache::TtsCache`]).
    /// 3. On a cache hit, sends the cached tensor back immediately without
    ///    running inference.
    /// 4. On a cache miss, tees the model's response through the cache: the
    ///    result is forwarded to the caller and written to disk on a
    ///    spawned task so the write never blocks generation.
    ///
    /// Concurrent requests for the same not-yet-cached text each miss and
    /// run inference independently -- there is no in-flight de-duplication.
    /// Atomic writes (see [`crate::engine::cache::TtsCache::put`]) keep this safe,
    /// just not maximally efficient under that access pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    #[cfg(feature = "tts-cache")]
    pub fn generate_speech(&self, model_name: &str, req: TtsGenerateRequest) -> Result<()> {
        let handle = self
            .tts_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown TTS model: {model_name}"))?;

        let Some(cache) = &self.tts_cache else {
            return handle.send(req);
        };
        if req.reference_audio.is_some() {
            return handle.send(req);
        }

        let digest = CacheKey::from_request(&handle.cache_model_id, &req).digest();
        if let Some(audio) = cache.get(&digest) {
            tracing::debug!(model = %model_name, "TTS cache hit");
            let _ = req.response_tx.send(Ok(audio));
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        let original_tx = req.response_tx;
        let cache = Arc::clone(cache);
        tokio::spawn(async move {
            if let Ok(result) = rx.await {
                if let Ok(ref audio) = result
                    && let Err(e) = cache.put(&digest, audio)
                {
                    tracing::warn!("Failed to write TTS cache entry: {e}");
                }
                let _ = original_tx.send(result);
            }
        });
        handle.send(TtsGenerateRequest { response_tx: tx, ..req })
    }

    /// Dispatch a TTS request directly to the model's thread.
    ///
    /// Built without the `tts-cache` feature, so there is no cache to
    /// consult -- this always delegates straight to [`TtsHandle::send`].
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    #[cfg(not(feature = "tts-cache"))]
    pub fn generate_speech(&self, model_name: &str, req: TtsGenerateRequest) -> Result<()> {
        let handle = self
            .tts_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown TTS model: {model_name}"))?;
        handle.send(req)
    }

    /// Returns the LLM engine handle, if a continuous-batching model is loaded.
    #[must_use]
    pub fn engine(&self) -> Option<&EngineHandle> {
        self.engine.as_ref()
    }

    /// Set the LLM engine handle.
    pub fn set_engine(&mut self, engine: EngineHandle) {
        self.engine = Some(engine);
    }

    /// Create a continuous-batching engine for `backend`, spawn its dedicated
    /// thread, and store the resulting handle.
    ///
    /// `backend` should already be warmed up (see
    /// [`ModelBackend::warmup`](crate::engine::backend::ModelBackend::warmup)) before
    /// calling this method.
    pub fn load_llm_engine(
        &mut self,
        backend: Box<dyn ModelBackend>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
        memory_config: MemoryConfig,
    ) {
        let (engine, handle) =
            InferenceEngine::new(backend, max_concurrent, decode_tokens_per_seq, memory_config);
        std::thread::Builder::new()
            .name("inference-engine".into())
            .spawn(move || engine.run())
            .expect("Failed to spawn engine thread");
        self.set_engine(handle);
    }

    /// Returns the VLM request channel, if a PaddleOCR-VL model is loaded.
    #[must_use]
    pub fn vlm_tx(&self) -> Option<&mpsc::UnboundedSender<VlmRequest>> {
        self.vlm_tx.as_ref()
    }

    /// Set the VLM request channel.
    pub fn set_vlm_tx(&mut self, tx: mpsc::UnboundedSender<VlmRequest>) {
        self.vlm_tx = Some(tx);
    }

    /// Returns the Gemma4 VLM request channel, if a Gemma4VL model is loaded.
    #[must_use]
    pub fn gemma4_vlm_tx(&self) -> Option<&mpsc::UnboundedSender<Gemma4VlmRequest>> {
        self.gemma4_vlm_tx.as_ref()
    }

    /// Set the Gemma4 VLM request channel.
    pub fn set_gemma4_vlm_tx(&mut self, tx: mpsc::UnboundedSender<Gemma4VlmRequest>) {
        self.gemma4_vlm_tx = Some(tx);
    }

    /// Returns the tokenizer for the loaded model.
    #[must_use]
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Returns the end-of-sequence token IDs for the loaded model.
    #[must_use]
    pub fn eos_token_id(&self) -> &[u32] {
        &self.eos_token_id
    }

    /// Returns the model's display name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the detected/configured model type.
    #[must_use]
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Returns the dtype display name (e.g. "F32", "BF16").
    #[must_use]
    pub fn dtype_name(&self) -> &str {
        &self.dtype_name
    }

    /// Returns the device display name (e.g. "Cpu", "Cuda(0)").
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// Derive a registration name from a model path's final path component.
fn extract_model_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .map_or_else(|| "tts".to_string(), |n| n.to_string_lossy().into_owned())
}

/// Run the blocking generation loop for one TTS model on its dedicated thread.
fn run_tts_thread(
    mut rx: mpsc::UnboundedReceiver<TtsGenerateRequest>,
    mut tts: Box<dyn Tts + Send>,
    model_name: &str,
    pending_count: &AtomicU64,
) {
    tracing::info!(model = %model_name, "TTS thread started");
    while let Some(req) = rx.blocking_recv() {
        let text_len = req.text.chars().count();

        if req.response_tx.is_closed() {
            tracing::warn!(model = %model_name, text_len, "Caller disconnected, skipping");
            pending_count.fetch_sub(1, Ordering::Relaxed);
            continue;
        }

        let t0 = std::time::Instant::now();

        let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            if let Some(ref ref_audio) = req.reference_audio {
                let ref_text = req.reference_text.as_deref().unwrap_or("");
                tts.generate_voice_clone(&req.text, &req.language, ref_audio, ref_text, &req.opts)
            } else {
                tts.generate_speech(&req.text, &req.language, req.voice.as_deref(), &req.opts)
            }
        }));

        let result = match panic_result {
            Ok(result) => result,
            Err(panic_payload) => {
                let msg = panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                tracing::error!(model = %model_name, text_len, panic = %msg, "TTS generation panicked");
                Err(anyhow::anyhow!("TTS generation panicked: {msg}"))
            }
        };

        let elapsed_ms = t0.elapsed().as_millis();
        match &result {
            Ok(tensor) => {
                tracing::info!(
                    model = %model_name,
                    text_len,
                    samples = tensor.elem_count(),
                    elapsed_ms,
                    "TTS generation complete",
                );
            }
            Err(e) => {
                tracing::error!(model = %model_name, text_len, elapsed_ms, error = %e, "TTS generation failed");
            }
        }

        let _ = req.response_tx.send(result);
        pending_count.fetch_sub(1, Ordering::Relaxed);
    }
    tracing::info!(model = %model_name, "TTS thread stopped (channel closed)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[cfg(feature = "tts-cache")]
    use std::fs;

    struct MockTts {
        audio_info: AudioInfo,
        voices: Vec<VoiceInfo>,
        supports_cloning: bool,
    }

    impl MockTts {
        fn new() -> Self {
            Self {
                audio_info: AudioInfo {
                    sample_rate: 24000,
                    channels: 1,
                    bits_per_sample: 16,
                },
                voices: vec![
                    VoiceInfo {
                        name: "alice".into(),
                        languages: vec!["en".into()],
                    },
                    VoiceInfo {
                        name: "bob".into(),
                        languages: vec!["en".into(), "fr".into()],
                    },
                ],
                supports_cloning: false,
            }
        }

        fn with_cloning(mut self) -> Self {
            self.supports_cloning = true;
            self
        }

        fn with_sample_rate(mut self, sample_rate: u32) -> Self {
            self.audio_info.sample_rate = sample_rate;
            self
        }
    }

    impl Tts for MockTts {
        fn audio_info(&self) -> AudioInfo {
            self.audio_info
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            self.voices.clone()
        }

        fn supports_voice_cloning(&self) -> bool {
            self.supports_cloning
        }

        fn generate_speech(
            &mut self,
            text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            let n = text.chars().count().max(1);
            Tensor::new(vec![0.5f32; n], &Device::Cpu).map_err(Into::into)
        }

        fn generate_voice_clone(
            &mut self,
            text: &str,
            _language: &str,
            _ref_audio: &Path,
            _ref_text: &str,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            let n = text.chars().count().max(1);
            Tensor::new(vec![-0.5f32; n], &Device::Cpu).map_err(Into::into)
        }
    }

    fn new_runtime() -> ModelRuntime {
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        ModelRuntime::new(
            "test-model".into(),
            ModelType::Qwen3TTS,
            "F32".into(),
            "Cpu".into(),
            tokenizer,
            vec![2],
        )
    }

    #[test]
    fn test_register_tts_stores_handle() {
        let mut rt = new_runtime();
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();

        let handle = rt.tts_handle("m1").expect("handle should be registered");
        assert_eq!(handle.audio_info().sample_rate, 24000);
        assert_eq!(handle.voices().len(), 2);
        assert!(!handle.supports_voice_cloning());
        assert_eq!(handle.model_type_name(), "qwen3_tts");
    }

    #[test]
    fn test_generate_speech_roundtrip() {
        let mut rt = new_runtime();
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();
        let handle = rt.tts_handle("m1").unwrap();

        let (tx, rx) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx,
            })
            .unwrap();

        let tensor = rx.blocking_recv().unwrap().unwrap();
        let samples: Vec<f32> = tensor.to_vec1().unwrap();
        assert_eq!(samples, vec![0.5f32; 5]);
    }

    #[test]
    fn test_generate_voice_clone_roundtrip() {
        let mut rt = new_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_cloning()),
        )
        .unwrap();
        let handle = rt.tts_handle("m1").unwrap();

        let (tx, rx) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "hi".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: Some(PathBuf::from("/ref.wav")),
                reference_text: Some("hi there".into()),
                response_tx: tx,
            })
            .unwrap();

        let tensor = rx.blocking_recv().unwrap().unwrap();
        let samples: Vec<f32> = tensor.to_vec1().unwrap();
        assert_eq!(samples, vec![-0.5f32; 2]);
    }

    #[test]
    fn test_multiple_tts_models() {
        let mut rt = new_runtime();
        rt.register_tts(
            "a".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_sample_rate(24000)),
        )
        .unwrap();
        rt.register_tts(
            "b".into(),
            "voxtral_tts",
            Box::new(MockTts::new().with_sample_rate(16000)),
        )
        .unwrap();

        assert_eq!(rt.tts_handle("a").unwrap().audio_info().sample_rate, 24000);
        assert_eq!(rt.tts_handle("b").unwrap().audio_info().sample_rate, 16000);
    }

    #[test]
    fn test_default_tts_handle() {
        let mut rt = new_runtime();
        assert!(rt.default_tts_handle().is_none());

        rt.register_tts("a".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();
        assert!(rt.default_tts_handle().is_some());
    }

    #[test]
    fn test_tts_handle_not_found() {
        let rt = new_runtime();
        assert!(rt.tts_handle("nonexistent").is_none());
    }

    #[test]
    fn test_model_runtime_new_metadata() {
        let rt = new_runtime();
        assert_eq!(rt.model_name(), "test-model");
        assert_eq!(rt.model_type(), ModelType::Qwen3TTS);
        assert_eq!(rt.dtype_name(), "F32");
        assert_eq!(rt.device_name(), "Cpu");
        assert!(rt.engine().is_none());
        assert!(rt.default_tts_handle().is_none());
        assert!(rt.default_tts_name().is_none());
        assert_eq!(rt.eos_token_id(), &[2]);
        assert!(rt.vlm_tx().is_none());
        assert!(rt.gemma4_vlm_tx().is_none());
    }

    #[test]
    fn test_vlm_tx_roundtrip() {
        let mut rt = new_runtime();
        assert!(rt.vlm_tx().is_none());
        let (tx, _rx) = mpsc::unbounded_channel::<VlmRequest>();
        rt.set_vlm_tx(tx);
        assert!(rt.vlm_tx().is_some());
    }

    #[test]
    fn test_gemma4_vlm_tx_roundtrip() {
        let mut rt = new_runtime();
        assert!(rt.gemma4_vlm_tx().is_none());
        let (tx, _rx) = mpsc::unbounded_channel::<Gemma4VlmRequest>();
        rt.set_gemma4_vlm_tx(tx);
        assert!(rt.gemma4_vlm_tx().is_some());
    }

    #[test]
    fn test_default_tts_name() {
        let mut rt = new_runtime();
        assert!(rt.default_tts_name().is_none());
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();
        assert_eq!(rt.default_tts_name(), Some("m1"));
    }

    #[test]
    fn test_extract_model_name() {
        assert_eq!(
            extract_model_name("/models/Qwen3-TTS-12Hz-0.6B"),
            "Qwen3-TTS-12Hz-0.6B"
        );
        assert_eq!(extract_model_name("/models/voxtral/"), "voxtral");
        assert_eq!(extract_model_name(""), "tts");
    }

    #[test]
    fn test_send_after_thread_stopped() {
        let (tx, rx) = mpsc::unbounded_channel::<TtsGenerateRequest>();
        drop(rx);

        let handle = TtsHandle {
            tx,
            audio_info: AudioInfo {
                sample_rate: 24000,
                channels: 1,
                bits_per_sample: 16,
            },
            voices: vec![],
            supports_voice_cloning: false,
            model_type_name: "qwen3_tts",
            pending_count: Arc::new(AtomicU64::new(0)),
            cache_model_id: "test-model".into(),
        };

        let (resp_tx, _resp_rx) = oneshot::channel();
        let result = handle.send(TtsGenerateRequest {
            text: "hello".into(),
            language: "en".into(),
            voice: None,
            opts: SpeechOptions::default(),
            reference_audio: None,
            reference_text: None,
            response_tx: resp_tx,
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stopped"));
        assert_eq!(handle.pending_count(), 0);
    }

    #[test]
    fn test_set_engine_roundtrip() {
        use crate::engine::stats::EngineStats;
        use crate::engine::types::EngineHandle;

        let mut rt = new_runtime();
        assert!(rt.engine().is_none());

        let (tx, _rx) = mpsc::unbounded_channel();
        let engine = EngineHandle {
            request_tx: tx,
            stats: Arc::new(EngineStats::new()),
        };
        rt.set_engine(engine);

        assert!(rt.engine().is_some());
    }

    #[test]
    fn test_register_duplicate_name() {
        let mut rt = new_runtime();
        rt.register_tts(
            "dup".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_sample_rate(24000)),
        )
        .unwrap();
        rt.register_tts(
            "dup".into(),
            "voxtral_tts",
            Box::new(MockTts::new().with_sample_rate(16000)),
        )
        .unwrap();

        let handle = rt.tts_handle("dup").unwrap();
        assert_eq!(handle.audio_info().sample_rate, 16000);
        assert_eq!(handle.model_type_name(), "voxtral_tts");
    }

    struct PanickingTts {
        call_count: u32,
    }

    impl PanickingTts {
        fn new() -> Self {
            Self { call_count: 0 }
        }
    }

    impl Tts for PanickingTts {
        fn audio_info(&self) -> AudioInfo {
            AudioInfo {
                sample_rate: 24000,
                channels: 1,
                bits_per_sample: 16,
            }
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            vec![]
        }

        fn generate_speech(
            &mut self,
            text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            self.call_count += 1;
            if self.call_count == 1 {
                panic!("simulated model panic");
            }
            let n = text.chars().count().max(1);
            Tensor::new(vec![0.1f32; n], &Device::Cpu).map_err(Into::into)
        }
    }

    #[test]
    fn test_panic_in_generate_is_caught() {
        let mut rt = new_runtime();
        rt.register_tts("panic_model".into(), "qwen3_tts", Box::new(PanickingTts::new()))
            .unwrap();
        let handle = rt.tts_handle("panic_model").unwrap();

        let (tx1, rx1) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "boom".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx1,
            })
            .unwrap();

        let result1 = rx1.blocking_recv().unwrap();
        assert!(result1.is_err());
        assert!(result1.unwrap_err().to_string().contains("panicked"));

        let (tx2, rx2) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "ok".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx2,
            })
            .unwrap();

        let result2 = rx2.blocking_recv().unwrap();
        assert!(result2.is_ok());
        let samples: Vec<f32> = result2.unwrap().to_vec1().unwrap();
        assert_eq!(samples, vec![0.1f32; 2]);
    }

    #[test]
    fn test_generate_speech_no_cache() {
        let mut rt = new_runtime();
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();

        let (tx, rx) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx,
            },
        )
        .unwrap();

        let tensor = rx.blocking_recv().unwrap().unwrap();
        let samples: Vec<f32> = tensor.to_vec1().unwrap();
        assert_eq!(samples, vec![0.5f32; 5]);
    }

    #[test]
    fn test_generate_speech_unknown_model() {
        let rt = new_runtime();
        let (tx, _rx) = oneshot::channel();
        let result = rt.generate_speech(
            "nonexistent",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx,
            },
        );
        assert!(result.is_err());
    }

    #[cfg(feature = "tts-cache")]
    #[tokio::test]
    async fn test_generate_speech_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let mut rt = new_runtime();
        rt.set_tts_cache(TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap());
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();

        // First request: cache miss, generates and caches.
        let (tx1, rx1) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx1,
            },
        )
        .unwrap();
        let first = rx1.await.unwrap().unwrap();
        assert_eq!(first.to_vec1::<f32>().unwrap(), vec![0.5f32; 5]);

        // The tee task writes the cache entry before forwarding the
        // response, so by the time `rx1.await` resolves above the entry
        // is already on disk -- no extra synchronization needed here.

        // Second identical request: cache hit, same result without
        // depending on the (still-registered) model thread.
        let (tx2, rx2) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx2,
            },
        )
        .unwrap();
        let second = rx2.await.unwrap().unwrap();
        assert_eq!(second.to_vec1::<f32>().unwrap(), vec![0.5f32; 5]);
    }

    #[cfg(feature = "tts-cache")]
    #[tokio::test]
    async fn test_generate_speech_reference_audio_bypasses_cache() {
        let dir = tempfile::tempdir().unwrap();
        let mut rt = new_runtime();
        rt.set_tts_cache(TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap());
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_cloning()),
        )
        .unwrap();

        let (tx, rx) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: Some(PathBuf::from("/ref.wav")),
                reference_text: Some("hi".into()),
                response_tx: tx,
            },
        )
        .unwrap();

        let result = rx.await.unwrap().unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![-0.5f32; 5]);

        // Nothing should have been written to the cache directory.
        let has_entries = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .any(|e| e.path().is_dir());
        assert!(!has_entries);
    }

    struct MockModelBackend {
        device: Device,
        tokenizer: tokenizers::Tokenizer,
    }

    impl MockModelBackend {
        fn new() -> Self {
            Self {
                device: Device::Cpu,
                tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            }
        }
    }

    impl ModelBackend for MockModelBackend {
        fn forward_step(&mut self, _input_ids: &[u32], _start_pos: usize) -> Result<Tensor> {
            Tensor::zeros(1, DType::F32, &self.device).map_err(Into::into)
        }

        fn clear_kv_cache(&mut self) {}

        fn num_layers(&self) -> usize {
            0
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn tokenizer(&self) -> &tokenizers::Tokenizer {
            &self.tokenizer
        }

        fn eos_token_id(&self) -> Vec<u32> {
            vec![2]
        }

        fn warmup(&mut self) {}
    }

    #[test]
    fn test_load_llm_engine() {
        let mut rt = new_runtime();
        assert!(rt.engine().is_none());

        rt.load_llm_engine(
            Box::new(MockModelBackend::new()),
            1,
            1,
            MemoryConfig {
                max_seq_len: 0,
                gpu_memory_limit_bytes: 0,
                baseline_gpu_bytes: 0,
            },
        );

        assert!(rt.engine().is_some());
    }
}
