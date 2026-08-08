//! Live full-duplex demo for MiniCPM-o-4.5 (phase 7's final acceptance
//! check — see the plan doc and `AGENTS.md`'s MiniCPM-o section).
//!
//! Connects to crane-serve's `/v1/audio/duplex` WebSocket endpoint, holds a
//! real spoken conversation: captures microphone audio via `cpal`, streams
//! it in ~1s chunks (matching the server's own chunking expectation),
//! prints the model's listen/speak arbitration state per chunk, and plays
//! back any returned speech audio live through the default output device.
//!
//! Two modes:
//! - **Live mic** (default): real-time capture + playback, Ctrl+C to stop.
//! - `--wav <path>`: reads a file instead of the mic — for automated/
//!   headless testing without a live microphone (this is how the pipeline
//!   was actually validated during development). Still plays back
//!   response audio live if an output device is available.
//!
//! Either mode always writes the session's accumulated response audio to
//! `--output-wav` (24kHz mono, the vocoder's native rate) so the round
//! trip can be inspected offline regardless of whether live playback
//! worked on the machine running it.
//!
//! Start the server first, e.g.:
//! `cargo run -p crane-serve --features cuda --bin crane-serve -- -m /path/to/MiniCPM-o-4_5 --model-type minicpmo`

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use base64::Engine as _;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::tungstenite::Message;

/// Matches the server's own "called once per second" chunking convention
/// (`handlers::duplex`'s module doc) — the client controls actual cadence,
/// but 1s chunks is what the model was tuned/tested against.
const CHUNK_SAMPLES: usize = 16_000;
const INPUT_SAMPLE_RATE: u32 = 16_000;
/// `Token2Wav::sample_rate()` — fixed, matches `handlers::duplex`'s
/// `DuplexChunkEvent::audio_sample_rate`.
const SERVER_AUDIO_SAMPLE_RATE: u32 = 24_000;

#[derive(Parser, Debug)]
#[command(about = "MiniCPM-o-4.5 full-duplex live audio chat demo")]
struct Args {
    #[arg(long, default_value = "ws://127.0.0.1:8080/v1/audio/duplex")]
    server: String,
    #[arg(long)]
    system_prompt: Option<String>,
    /// Read audio from a WAV file instead of the live microphone (for
    /// automated/headless testing).
    #[arg(long)]
    wav: Option<String>,
    /// Where to write the session's accumulated response audio.
    #[arg(long, default_value = "/tmp/minicpmo_duplex_demo_output.wav")]
    output_wav: String,
    /// Skip live playback through the default output device (still writes
    /// `--output-wav` either way).
    #[arg(long)]
    no_playback: bool,
    /// Only in `--wav` mode: stop after this many chunks (0 = whole file).
    #[arg(long, default_value_t = 0)]
    max_chunks: usize,
}

#[derive(Debug, Serialize)]
struct PrepareMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DuplexChunkEvent {
    is_listen: bool,
    text: String,
    end_of_turn: bool,
    audio_base64: Option<String>,
    audio_sample_rate: Option<u32>,
}

type WsStream = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = futures_util::stream::SplitSink<WsStream, Message>;
type WsSource = futures_util::stream::SplitStream<WsStream>;
type PlaybackQueue = Arc<Mutex<VecDeque<f32>>>;
type CaptureBuffer = Arc<Mutex<VecDeque<f32>>>;

/// Naive linear-interpolation resampler — not a general-purpose DSP
/// primitive, just good enough for mic/speaker rate conversion in a demo
/// (same technique already used by this codebase's offline duplex tests).
fn linear_resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = (samples.len() as f64 * ratio).round() as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = (src_pos - idx as f64) as f32;
            let a = samples.get(idx).copied().unwrap_or(0.0);
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved.chunks_exact(channels as usize).map(|frame| frame.iter().sum::<f32>() / f32::from(channels)).collect()
}

async fn send_chunk(write: &mut WsSink, samples: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        bytes.extend_from_slice(&((clamped * f32::from(i16::MAX)) as i16).to_le_bytes());
    }
    write.send(Message::Binary(bytes.into())).await.context("failed to send audio chunk")?;
    Ok(())
}

async fn recv_event(read: &mut WsSource) -> Result<DuplexChunkEvent> {
    loop {
        match read.next().await {
            Some(Ok(Message::Text(text))) => {
                let value: serde_json::Value = serde_json::from_str(&text)?;
                if let Some(err) = value.get("error").and_then(serde_json::Value::as_str) {
                    anyhow::bail!("server error: {err}");
                }
                return Ok(serde_json::from_value(value)?);
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("connection closed by server"),
            Some(Ok(_)) => continue,
            Some(Err(e)) => anyhow::bail!("WebSocket error: {e}"),
        }
    }
}

fn print_event(i: usize, event: &DuplexChunkEvent) {
    println!(
        "chunk {i}: is_listen={} end_of_turn={} text={:?} audio_bytes={}",
        event.is_listen,
        event.end_of_turn,
        event.text,
        event.audio_base64.as_ref().map_or(0, String::len)
    );
}

/// Decodes any audio in `event`, accumulates it (at the server's native
/// 24kHz) into `accumulated` for the final `--output-wav`, and — if a
/// playback device is available — resamples and queues it for live
/// playback too.
fn handle_audio(event: &DuplexChunkEvent, playback_queue: &PlaybackQueue, playback_rate: u32, accumulated: &mut Vec<f32>) {
    let Some(b64) = &event.audio_base64 else {
        return;
    };
    let server_rate = event.audio_sample_rate.unwrap_or(SERVER_AUDIO_SAMPLE_RATE);
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
        eprintln!("warning: failed to decode audio_base64");
        return;
    };
    let samples: Vec<f32> = bytes.chunks_exact(2).map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0).collect();
    accumulated.extend_from_slice(&samples);

    if playback_rate > 0 {
        let resampled = linear_resample(&samples, server_rate, playback_rate);
        let mut q = playback_queue.lock().unwrap();
        q.extend(resampled);
    }
}

/// Starts a live-playback output stream reading from `queue`. Returns
/// `(None, 0)` (not an error) if no output device is available — playback
/// is best-effort, the demo should still run headless.
fn start_playback_stream(queue: PlaybackQueue) -> Result<(Option<cpal::Stream>, u32)> {
    let host = cpal::default_host();
    let Some(device) = host.default_output_device() else {
        eprintln!("no default output device found; skipping live playback (--output-wav will still be written)");
        return Ok((None, 0));
    };
    let config = device.default_output_config().context("no default output config")?;
    let sample_rate = config.sample_rate();
    let channels = usize::from(config.channels());
    let stream_config: cpal::StreamConfig = config.into();
    let stream = device.build_output_stream(
        stream_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut q = queue.lock().unwrap();
            for frame in data.chunks_mut(channels) {
                let sample = q.pop_front().unwrap_or(0.0);
                for out in frame {
                    *out = sample;
                }
            }
        },
        move |err| eprintln!("output stream error: {err}"),
        None,
    )?;
    stream.play()?;
    println!("playback device ready: {sample_rate} Hz, {channels} channel(s)");
    Ok((Some(stream), sample_rate))
}

/// Starts live microphone capture into `buffer`. Returns the stream handle
/// (must be kept alive for capture to continue) plus the device's actual
/// sample rate/channel count, since a demo shouldn't assume the default
/// input device runs at exactly 16kHz mono.
fn start_capture_stream(buffer: CaptureBuffer) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = host.default_input_device().context("no default input device found")?;
    let config = device.default_input_config().context("no default input config")?;
    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let stream_config: cpal::StreamConfig = config.into();
    let stream = device.build_input_stream(
        stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buf = buffer.lock().unwrap();
            buf.extend(data.iter().copied());
        },
        move |err| eprintln!("input stream error: {err}"),
        None,
    )?;
    stream.play()?;
    Ok((stream, sample_rate, channels))
}

fn write_output_wav(path: &str, samples: &[f32]) -> Result<()> {
    if samples.is_empty() {
        println!("no response audio received — nothing to write to {path}");
        return Ok(());
    }
    let spec = hound::WavSpec { channels: 1, sample_rate: SERVER_AUDIO_SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)?;
    }
    writer.finalize()?;
    println!("wrote {path} ({:.2}s)", samples.len() as f32 / SERVER_AUDIO_SAMPLE_RATE as f32);
    Ok(())
}

fn load_wav_as_16k_mono(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("failed to open {path}"))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / max)).collect::<std::result::Result<_, _>>()?
        }
    };
    let mono = downmix_to_mono(&raw, spec.channels);
    Ok(linear_resample(&mono, spec.sample_rate, INPUT_SAMPLE_RATE))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("connecting to {}", args.server);
    let (ws_stream, _) = tokio_tungstenite::connect_async(&args.server).await.context("failed to connect to duplex WS endpoint — is crane-serve running with a MiniCPM-o-4.5 checkpoint?")?;
    let (mut write, mut read) = ws_stream.split();

    let prepare = PrepareMessage { system_prompt: args.system_prompt.clone() };
    write.send(Message::Text(serde_json::to_string(&prepare)?.into())).await?;

    loop {
        match read.next().await {
            Some(Ok(Message::Text(text))) => {
                if text.contains("\"ready\"") {
                    break;
                }
                if text.contains("\"error\"") {
                    anyhow::bail!("server rejected the session: {text}");
                }
            }
            Some(Ok(_)) => continue,
            Some(Err(e)) => anyhow::bail!("WebSocket error while waiting for ready: {e}"),
            None => anyhow::bail!("connection closed before session became ready"),
        }
    }
    println!("session ready");

    let playback_queue: PlaybackQueue = Arc::new(Mutex::new(VecDeque::new()));
    let (_playback_stream, playback_rate) = if args.no_playback {
        (None, 0)
    } else {
        match start_playback_stream(playback_queue.clone()) {
            Ok((stream, rate)) => (stream, rate),
            Err(e) => {
                eprintln!("playback unavailable ({e}); continuing without live playback");
                (None, 0)
            }
        }
    };

    let mut all_response_audio: Vec<f32> = Vec::new();

    if let Some(wav_path) = &args.wav {
        let resampled = load_wav_as_16k_mono(wav_path)?;
        let chunks: Vec<&[f32]> = resampled.chunks(CHUNK_SAMPLES).collect();
        let total = if args.max_chunks > 0 { args.max_chunks.min(chunks.len()) } else { chunks.len() };
        println!("streaming {total} chunk(s) from {wav_path}");
        for (i, chunk) in chunks.iter().take(total).enumerate() {
            send_chunk(&mut write, chunk).await?;
            let event = recv_event(&mut read).await?;
            print_event(i, &event);
            handle_audio(&event, &playback_queue, playback_rate, &mut all_response_audio);
        }
    } else {
        let capture_buffer: CaptureBuffer = Arc::new(Mutex::new(VecDeque::new()));
        let (_capture_stream, capture_rate, capture_channels) = start_capture_stream(capture_buffer.clone())?;
        println!("capturing from the default input device: {capture_rate} Hz, {capture_channels} channel(s) — speak now (Ctrl+C to stop)");
        let raw_samples_per_chunk = capture_rate as usize * usize::from(capture_channels);
        let mut i = 0usize;
        loop {
            loop {
                let len = capture_buffer.lock().unwrap().len();
                if len >= raw_samples_per_chunk {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            let raw_chunk: Vec<f32> = capture_buffer.lock().unwrap().drain(..raw_samples_per_chunk).collect();
            let mono = downmix_to_mono(&raw_chunk, capture_channels);
            let resampled = linear_resample(&mono, capture_rate, INPUT_SAMPLE_RATE);
            send_chunk(&mut write, &resampled).await?;
            let event = recv_event(&mut read).await?;
            print_event(i, &event);
            handle_audio(&event, &playback_queue, playback_rate, &mut all_response_audio);
            i += 1;
        }
    }

    write_output_wav(&args.output_wav, &all_response_audio)?;
    Ok(())
}
