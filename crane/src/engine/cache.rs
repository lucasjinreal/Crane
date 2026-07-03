//! Optional disk cache for TTS audio responses.
//!
//! Home Assistant workloads are dominated by repeated phrases ("the front
//! door is open", "timer finished"). On CPU, regenerating audio for
//! previously-seen text is pure waste. [`TtsCache`] stores generated
//! waveforms on disk, keyed by every input that affects the output, so a
//! repeated request can skip inference entirely.
//!
//! Caching is wired in at the [`crate::engine::ModelRuntime`] level (see
//! [`crate::engine::ModelRuntime::generate_speech`]), not in
//! [`crate::engine::TtsHandle`], so direct handle callers are unaffected by
//! whether a cache is configured.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::engine::runtime::TtsGenerateRequest;

/// Disambiguates concurrent `put` temp-file names for the same cache key.
static TMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Deterministic cache key for a TTS generation request.
///
/// Captures every input that affects the generated waveform. Hashed with
/// blake3 to produce a stable, cross-process-restart digest used as the
/// on-disk filename. Floats are converted to their bit patterns because
/// they do not implement `Hash`/`Eq`.
///
/// Requests with `reference_audio` set (voice cloning) must never be
/// turned into a `CacheKey` -- [`crate::engine::ModelRuntime::generate_speech`]
/// enforces this bypass.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CacheKey {
    model_name: String,
    voice: Option<String>,
    language: String,
    text: String,
    max_new_tokens: usize,
    temperature_bits: u64,
    top_p_bits: Option<u64>,
    repetition_penalty_bits: u32,
}

impl CacheKey {
    /// Build a cache key from a model name and a generation request.
    #[must_use]
    pub(crate) fn from_request(model_name: &str, req: &TtsGenerateRequest) -> Self {
        Self {
            model_name: model_name.to_string(),
            voice: req.voice.clone(),
            language: req.language.clone(),
            text: req.text.clone(),
            max_new_tokens: req.opts.max_new_tokens,
            temperature_bits: req.opts.temperature.to_bits(),
            top_p_bits: req.opts.top_p.map(f64::to_bits),
            repetition_penalty_bits: req.opts.repetition_penalty.to_bits(),
        }
    }

    /// Hash all fields into a stable 64-character hex digest.
    ///
    /// Each field is fed to the hasher with a preceding length or
    /// discriminant byte so that, e.g., `text: "ab"` followed by
    /// `voice: "c"` cannot collide with `text: "a"` followed by
    /// `voice: "bc"`.
    #[must_use]
    pub(crate) fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hash_str(&mut hasher, &self.model_name);
        hash_opt_str(&mut hasher, self.voice.as_deref());
        hash_str(&mut hasher, &self.language);
        hash_str(&mut hasher, &self.text);
        hasher.update(&(self.max_new_tokens as u64).to_le_bytes());
        hasher.update(&self.temperature_bits.to_le_bytes());
        match self.top_p_bits {
            Some(bits) => {
                hasher.update(&[1u8]);
                hasher.update(&bits.to_le_bytes());
            }
            None => {
                hasher.update(&[0u8]);
            }
        }
        hasher.update(&self.repetition_penalty_bits.to_le_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

/// Feed a length-prefixed string into the hasher.
fn hash_str(hasher: &mut blake3::Hasher, s: &str) {
    hasher.update(&(s.len() as u64).to_le_bytes());
    hasher.update(s.as_bytes());
}

/// Feed a length-prefixed optional string into the hasher.
fn hash_opt_str(hasher: &mut blake3::Hasher, s: Option<&str>) {
    match s {
        Some(s) => {
            hasher.update(&[1u8]);
            hash_str(hasher, s);
        }
        None => {
            hasher.update(&[0u8]);
        }
    }
}

/// On-disk metadata sidecar for a cached TTS audio entry.
///
/// Kept as a small separate file (rather than an embedded header) so
/// hit-count/last-accessed updates only rewrite a few bytes instead of
/// the (potentially large) PCM file.
#[derive(Serialize, Deserialize)]
struct CacheMeta {
    hits: u64,
    created_unix: u64,
    last_accessed_unix: u64,
    bytes: u64,
}

/// Optional disk cache for TTS audio responses.
///
/// Stores raw f32 little-endian PCM samples on disk, keyed by a blake3
/// hash of the generation parameters (see [`CacheKey`]).
///
/// # On-disk layout
///
/// ```text
/// <cache_dir>/
///   ab/
///     ab34...ef.pcm    # raw f32 LE samples, no header
///     ab34...ef.meta   # JSON: { hits, created_unix, last_accessed_unix, bytes }
/// ```
///
/// The first byte of the hex digest forms a subdirectory so no single
/// directory accumulates too many entries.
///
/// # Eviction
///
/// When total cached bytes exceed `max_bytes`, entries with the lowest
/// `hits` are removed first; ties are broken by the oldest
/// `last_accessed_unix`. Eviction runs after every [`TtsCache::put`].
///
/// # Voice cloning
///
/// Voice-cloned text is typically novel per call, and correctly keying on
/// arbitrary reference-audio bytes would require hashing the whole file
/// every request, defeating the purpose of caching. Callers must not
/// build a [`CacheKey`] for requests with `reference_audio` set --
/// [`crate::engine::ModelRuntime::generate_speech`] enforces this.
pub struct TtsCache {
    dir: PathBuf,
    max_bytes: u64,
}

impl TtsCache {
    /// Open (creating if needed) a disk cache rooted at `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dir` cannot be created.
    pub fn new(dir: PathBuf, max_bytes: u64) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir, max_bytes })
    }

    /// Look up a cached result by digest. Updates `hits` and
    /// `last_accessed_unix` on hit.
    ///
    /// Returns `None` on a miss, on any I/O error while reading, or if the
    /// stored PCM data is corrupt (empty or not a whole number of `f32`
    /// samples) -- caching is a performance optimization, not a correctness
    /// requirement, so failures degrade to a cache miss rather than
    /// propagating or returning truncated audio.
    #[must_use]
    pub(crate) fn get(&self, digest: &str) -> Option<Tensor> {
        let (pcm_path, meta_path) = self.entry_paths(digest);

        let pcm_bytes = fs::read(&pcm_path).ok()?;
        if pcm_bytes.is_empty() || pcm_bytes.len() % 4 != 0 {
            return None;
        }
        let mut meta: CacheMeta = fs::read(&meta_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())?;

        meta.hits += 1;
        meta.last_accessed_unix = now_unix();
        if let Ok(bytes) = serde_json::to_vec(&meta)
            && let Err(e) = fs::write(&meta_path, bytes)
        {
            tracing::warn!("Failed to update TTS cache metadata: {e}");
        }

        let samples: Vec<f32> = pcm_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        // Cached audio is always decoded onto the CPU; callers that need it
        // on another device are responsible for moving it there.
        Tensor::new(samples, &Device::Cpu).ok()
    }

    /// Store a generated result under `digest`, then evict if `max_bytes` is
    /// exceeded.
    ///
    /// The `.pcm` file is written atomically (temp file + rename) so a
    /// concurrent `get` never observes a partial write. A failure during
    /// eviction is logged but does not make this call return an error --
    /// the entry itself was written successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if the audio or metadata cannot be written.
    pub(crate) fn put(&self, digest: &str, audio: &Tensor) -> Result<()> {
        let samples = audio
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let (pcm_path, meta_path) = self.entry_paths(digest);

        let subdir = pcm_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("cache entry path has no parent directory"))?;
        fs::create_dir_all(subdir)?;

        let bytes: Vec<u8> = samples.iter().flat_map(|sample| sample.to_le_bytes()).collect();

        // Unique per-call temp name: concurrent `put`s for the same key
        // must not share a temp path, or one thread's rename can remove
        // the file out from under another's.
        let nonce = TMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = pcm_path.with_extension(format!("pcm.{nonce}.tmp"));
        fs::write(&tmp_path, &bytes)?;
        fs::rename(&tmp_path, &pcm_path)?;

        let now = now_unix();
        let meta = CacheMeta {
            hits: 0,
            created_unix: now,
            last_accessed_unix: now,
            bytes: bytes.len() as u64,
        };
        fs::write(&meta_path, serde_json::to_vec(&meta)?)?;

        if let Err(e) = self.evict_if_needed() {
            tracing::warn!("Failed to evict TTS cache entries: {e}");
        }
        Ok(())
    }

    /// Path pair `(pcm_path, meta_path)` for a given digest.
    fn entry_paths(&self, digest: &str) -> (PathBuf, PathBuf) {
        let subdir = self.dir.join(&digest[..2]);
        (
            subdir.join(format!("{digest}.pcm")),
            subdir.join(format!("{digest}.meta")),
        )
    }

    /// Delete entries with the lowest `hits` (ties broken by oldest
    /// `last_accessed_unix`) until total cached size is under `max_bytes`.
    fn evict_if_needed(&self) -> Result<()> {
        let mut entries = Vec::new();
        let mut total_bytes: u64 = 0;

        for subdir in fs::read_dir(&self.dir)?.filter_map(std::result::Result::ok) {
            if !subdir.path().is_dir() {
                continue;
            }
            for entry in fs::read_dir(subdir.path())?.filter_map(std::result::Result::ok) {
                let path = entry.path();
                if path.extension().and_then(std::ffi::OsStr::to_str) != Some("meta") {
                    continue;
                }
                let Ok(bytes) = fs::read(&path) else { continue };
                let Ok(meta) = serde_json::from_slice::<CacheMeta>(&bytes) else {
                    continue;
                };
                total_bytes += meta.bytes;
                entries.push((path, meta));
            }
        }

        if total_bytes <= self.max_bytes {
            return Ok(());
        }

        entries.sort_by_key(|(_, meta)| (meta.hits, meta.last_accessed_unix));

        for (meta_path, meta) in entries {
            if total_bytes <= self.max_bytes {
                break;
            }
            let pcm_path = meta_path.with_extension("pcm");
            let _ = fs::remove_file(&pcm_path);
            let _ = fs::remove_file(&meta_path);
            total_bytes = total_bytes.saturating_sub(meta.bytes);
            tracing::debug!(path = %pcm_path.display(), "Evicted TTS cache entry");
        }

        Ok(())
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crane_core::generation::SpeechOptions;
    use tokio::sync::oneshot;

    fn make_request(text: &str) -> TtsGenerateRequest {
        let (tx, _rx) = oneshot::channel();
        TtsGenerateRequest {
            text: text.to_string(),
            language: "en".into(),
            voice: Some("alice".into()),
            opts: SpeechOptions::default(),
            reference_audio: None,
            reference_text: None,
            response_tx: tx,
        }
    }

    fn make_key(text: &str) -> CacheKey {
        CacheKey::from_request("model-a", &make_request(text))
    }

    #[test]
    fn test_cache_key_determinism() {
        let k1 = make_key("hello");
        let k2 = make_key("hello");
        assert_eq!(k1.digest(), k2.digest());

        let k3 = make_key("world");
        assert_ne!(k1.digest(), k3.digest());
    }

    #[test]
    fn test_cache_key_differs_per_field() {
        let base = make_key("hello");

        let mut voice_changed = base.clone();
        voice_changed.voice = Some("bob".into());
        assert_ne!(base.digest(), voice_changed.digest());

        let mut lang_changed = base.clone();
        lang_changed.language = "fr".into();
        assert_ne!(base.digest(), lang_changed.digest());

        let mut model_changed = base.clone();
        model_changed.model_name = "model-b".into();
        assert_ne!(base.digest(), model_changed.digest());

        let mut temp_changed = base.clone();
        temp_changed.temperature_bits = 1.0_f64.to_bits();
        assert_ne!(base.digest(), temp_changed.digest());

        let mut top_p_changed = base.clone();
        top_p_changed.top_p_bits = Some(0.9_f64.to_bits());
        assert_ne!(base.digest(), top_p_changed.digest());

        let mut rep_changed = base.clone();
        rep_changed.repetition_penalty_bits = 2.0_f32.to_bits();
        assert_ne!(base.digest(), rep_changed.digest());

        let mut tokens_changed = base.clone();
        tokens_changed.max_new_tokens = 1;
        assert_ne!(base.digest(), tokens_changed.digest());
    }

    #[test]
    fn test_cache_key_from_request() {
        let req = make_request("hi there");
        let key = CacheKey::from_request("m1", &req);
        assert_eq!(key.model_name, "m1");
        assert_eq!(key.text, "hi there");
        assert_eq!(key.voice.as_deref(), Some("alice"));
        assert_eq!(key.language, "en");
    }

    #[test]
    fn test_miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap();
        let key = make_key("nope");
        assert!(cache.get(&key.digest()).is_none());
    }

    #[test]
    fn test_put_then_get_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap();
        let key = make_key("hello");
        let audio = Tensor::new(vec![0.25f32, -0.5, 1.0], &Device::Cpu).unwrap();

        cache.put(&key.digest(), &audio).unwrap();
        let result = cache.get(&key.digest()).unwrap();
        let samples: Vec<f32> = result.to_vec1().unwrap();
        assert_eq!(samples, vec![0.25f32, -0.5, 1.0]);
    }

    #[test]
    fn test_hit_updates_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap();
        let key = make_key("hello");
        let audio = Tensor::new(vec![0.5f32; 10], &Device::Cpu).unwrap();
        cache.put(&key.digest(), &audio).unwrap();

        let (_, meta_path) = cache.entry_paths(&key.digest());
        let read_meta = || -> CacheMeta {
            let bytes = fs::read(&meta_path).unwrap();
            serde_json::from_slice(&bytes).unwrap()
        };

        assert_eq!(read_meta().hits, 0);

        cache.get(&key.digest()).unwrap();
        assert_eq!(read_meta().hits, 1);
        let first_access = read_meta().last_accessed_unix;

        cache.get(&key.digest()).unwrap();
        assert_eq!(read_meta().hits, 2);
        assert!(read_meta().last_accessed_unix >= first_access);
    }

    #[test]
    fn test_eviction_lowest_hits_first() {
        let dir = tempfile::tempdir().unwrap();
        // Each entry is 100 f32 samples = 400 bytes; cap fits only one.
        let cache = TtsCache::new(dir.path().to_path_buf(), 500).unwrap();
        let audio = Tensor::new(vec![0.5f32; 100], &Device::Cpu).unwrap();

        let popular = make_key("popular");
        let unpopular = make_key("unpopular");

        cache.put(&popular.digest(), &audio).unwrap();
        cache.get(&popular.digest()).unwrap();
        cache.get(&popular.digest()).unwrap(); // 2 hits, unambiguously more than 0.

        // Adding a second entry pushes total bytes over the cap. The
        // freshly-written (0-hit) entry has strictly fewer hits than
        // `popular`, so it is evicted immediately.
        cache.put(&unpopular.digest(), &audio).unwrap();

        assert!(cache.get(&unpopular.digest()).is_none());
        assert!(cache.get(&popular.digest()).is_some());
    }

    #[test]
    fn test_eviction_ties_broken_by_oldest_access() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 500).unwrap();
        let audio = Tensor::new(vec![0.5f32; 100], &Device::Cpu).unwrap();

        let older = make_key("older");
        let newer = make_key("newer");

        cache.put(&older.digest(), &audio).unwrap();

        // Both entries have 0 hits, so real-time creation order (which can
        // tie at one-second resolution) is not reliable for this test.
        // Force `older`'s timestamp further into the past to make the
        // tie-break deterministic.
        let (_, meta_path) = cache.entry_paths(&older.digest());
        let mut meta: CacheMeta = serde_json::from_slice(&fs::read(&meta_path).unwrap()).unwrap();
        meta.last_accessed_unix = meta.last_accessed_unix.saturating_sub(1000);
        fs::write(&meta_path, serde_json::to_vec(&meta).unwrap()).unwrap();

        // Adding `newer` pushes total bytes over the cap. Both entries have
        // 0 hits, so the tie is broken by oldest `last_accessed_unix`.
        cache.put(&newer.digest(), &audio).unwrap();

        assert!(cache.get(&older.digest()).is_none());
        assert!(cache.get(&newer.digest()).is_some());
    }

    #[test]
    fn test_concurrent_get_put() {
        let dir = tempfile::tempdir().unwrap();
        let cache =
            std::sync::Arc::new(TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap());
        let key = make_key("concurrent");
        let audio = Tensor::new(vec![1.0f32; 100], &Device::Cpu).unwrap();

        // Seed one entry before racing get/put so gets have something to find.
        cache.put(&key.digest(), &audio).unwrap();

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cache = std::sync::Arc::clone(&cache);
                let digest = key.digest();
                let audio = audio.clone();
                std::thread::spawn(move || {
                    if i % 2 == 0 {
                        cache.put(&digest, &audio).unwrap();
                    } else {
                        let _ = cache.get(&digest);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let result = cache.get(&key.digest()).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap().len(), 100);
    }

    #[test]
    fn test_get_returns_none_for_truncated_pcm() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap();
        let key = make_key("truncated");
        let audio = Tensor::new(vec![0.5f32; 10], &Device::Cpu).unwrap();
        cache.put(&key.digest(), &audio).unwrap();

        let (pcm_path, _) = cache.entry_paths(&key.digest());
        let mut bytes = fs::read(&pcm_path).unwrap();
        bytes.pop(); // Leave a length that isn't a multiple of 4.
        fs::write(&pcm_path, bytes).unwrap();

        assert!(cache.get(&key.digest()).is_none());
    }

    #[test]
    fn test_get_returns_none_for_empty_pcm() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap();
        let key = make_key("empty");
        let audio = Tensor::new(vec![0.5f32; 10], &Device::Cpu).unwrap();
        cache.put(&key.digest(), &audio).unwrap();

        let (pcm_path, _) = cache.entry_paths(&key.digest());
        fs::write(&pcm_path, []).unwrap();

        assert!(cache.get(&key.digest()).is_none());
    }

    #[test]
    fn test_get_returns_none_for_invalid_meta() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap();
        let key = make_key("bad-meta");
        let audio = Tensor::new(vec![0.5f32; 10], &Device::Cpu).unwrap();
        cache.put(&key.digest(), &audio).unwrap();

        let (_, meta_path) = cache.entry_paths(&key.digest());
        fs::write(&meta_path, b"not json").unwrap();

        assert!(cache.get(&key.digest()).is_none());
    }
}
