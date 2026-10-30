//! Minimal Standard MIDI File (SMF) writer for `MuScriptor` output.
//!
//! Implements just enough of SMF type-1 to round-trip the notes
//! `TranscriptionModel` produces:
//!   * a `MThd` header with `ticks_per_beat = 480`
//!   * a conductor track carrying tempo + (optional) time signature
//!   * one `MTrk` per program (or one drum track), holding `program_change`
//!     followed by absolute-tick `note_on` / `note_off` messages
//!
//! Tempo defaults to `500_000` µs/quarter = 120 BPM. Multi-track layout
//! matches the upstream `note_event2midi` so a downstream DAW (Ableton,
//! Logic, `MuseScore`) sees the same per-instrument splits.
//!
//! Out of scope: SMPTE timecode, mid-track tempo changes, control
//! changes, sysex. `MuScriptor` never produces those, and Crane doesn't
//! have a use case for them.

use crate::utils::float_cast::f64_to_u32;
use std::io::Write;

/// Default tempo: 120 BPM (`500_000` µs / quarter).
pub const DEFAULT_TEMPO: u32 = 500_000;
/// Default ticks per quarter note. 480 is the de-facto DAW standard and
/// matches the upstream `note_event2midi` default.
pub const DEFAULT_TICKS_PER_BEAT: u16 = 480;
/// Velocity baked into every emitted `note_on` (the `MuScriptor` vocab
/// doesn't carry dynamics).
pub const DEFAULT_VELOCITY: u8 = 100;

/// One decoded note ready to be serialized.
#[derive(Debug, Clone, Copy)]
pub struct MidiNote {
    /// `program == DRUM_PROGRAM` ⇒ drum track; otherwise the MIDI
    /// program number to use both for track routing and as a
    /// `program_change` at track open.
    pub program: u8,
    /// MIDI pitch, 0-127.
    pub pitch: u8,
    /// Onset in seconds (relative to the start of the audio).
    pub onset: f32,
    /// Offset in seconds; for drum notes, set equal to onset.
    pub offset: f32,
    /// Optional human-readable track name (program or "drums"). When
    /// `None`, the writer falls back to `program_<n>` / `"drums"`.
    pub instrument: Option<&'static str>,
}

/// All the per-track material a writer needs after the notes are sorted
/// into their per-program buckets.
#[derive(Debug)]
struct TrackBuf {
    program: u8,
    name: String,
    /// Absolute tick of the last event emitted on this track. Used for
    /// the delta-tick field on each subsequent message.
    last_tick: u32,
    /// Sorted by absolute tick, then `note_on` before `note_off` at the
    /// same tick (the standard tie-breaking for SMT).
    events: Vec<TrackEvent>,
    /// Channel assigned to this track; assigned by the writer at finalization.
    channel: u8,
}

#[derive(Debug, Clone, Copy)]
enum TrackEvent {
    NoteOn { tick: u32, pitch: u8, velocity: u8 },
    NoteOff { tick: u32, pitch: u8 },
}

/// Builder for a multi-track Standard MIDI File. Drop-in replacement for
/// the upstream `notes_to_midi(...)` call inside `MuScriptor`'s
/// `transcribe_to_midi`.
pub struct MidiWriter {
    tempo: u32,
    ticks_per_beat: u16,
    time_signature: Option<(u8, u8)>,
    velocity: u8,
}

impl Default for MidiWriter {
    fn default() -> Self {
        Self {
            tempo: DEFAULT_TEMPO,
            ticks_per_beat: DEFAULT_TICKS_PER_BEAT,
            time_signature: None,
            velocity: DEFAULT_VELOCITY,
        }
    }
}

impl MidiWriter {
    /// Create a writer with the default tempo/tick-density/velocity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_tempo(mut self, tempo_us_per_quarter: u32) -> Self {
        self.tempo = tempo_us_per_quarter;
        self
    }

    #[must_use]
    pub fn with_time_signature(mut self, numerator: u8, denominator: u8) -> Self {
        self.time_signature = Some((numerator, denominator));
        self
    }

    #[must_use]
    pub fn with_velocity(mut self, velocity: u8) -> Self {
        self.velocity = velocity;
        self
    }

    /// Serialize `notes` (any order) into a complete SMF byte buffer.
    /// Splits drums onto channel 9 and per-program notes onto the lowest
    /// unused non-drum channel, matching the upstream `note_event2midi`
    /// channel-allocation policy.
    #[must_use]
    pub fn write(&self, notes: &[MidiNote]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 1024);
        self.write_to(&mut out, notes);
        out
    }

    /// Same as [`Self::write`] but streams into a caller-owned `Write`.
    pub fn write_to<W: Write>(&self, out: &mut W, notes: &[MidiNote]) {
        let tracks = self.prepare_tracks(notes);
        let num_tracks = tracks.len() + 1; // +1 for the conductor
        let header = build_header(num_tracks as u16, self.ticks_per_beat);
        out.write_all(&header)
            .expect("writing to Vec/BytesMut never fails");

        // Conductor: tempo (+ optional time signature) then `EndOfTrack`.
        let mut conductor = Vec::with_capacity(64);
        write_meta(&mut conductor, 0, 0x51, &encode_tempo(self.tempo));
        if let Some((num, den)) = self.time_signature {
            write_meta(&mut conductor, 0, 0x58, &encode_time_signature(num, den));
        }
        // EndOfTrack meta.
        push_varlen(&mut conductor, 0);
        conductor.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        write_track(out, &conductor);

        for t in &tracks {
            let mut buf = Vec::with_capacity(64);
            // track_name meta at delta 0
            let name_bytes = t.name.as_bytes();
            push_varlen(&mut buf, 0);
            buf.push(0xFF);
            buf.push(0x03);
            push_varlen(&mut buf, name_bytes.len() as u32);
            buf.extend_from_slice(name_bytes);

            // set_tempo (MuseScore ignores set_tempo in a conductor
            // track that has no notes, so repeat it here — harmless for
            // hosts that read the meta track).
            write_meta(&mut buf, 0, 0x51, &encode_tempo(self.tempo));

            // program_change (drums: gm program 0; otherwise the program id)
            push_varlen(&mut buf, 0);
            let program_id = if t.program == crate::models::muscriptor::DRUM_PROGRAM {
                0
            } else {
                t.program
            };
            buf.push(0xC0 | (t.channel & 0x0F));
            buf.push(program_id & 0x7F);

            // Emit note events with delta tick encoding.
            // TrackEvent sorts `note_on` before `note_off` at the same
            // tick by construction (within the same tick, sorting puts
            // NoteOn earlier — see `prepare_tracks`).
            let mut last_tick: u32 = 0;
            for ev in &t.events {
                let delta = (ev.tick_u32()).saturating_sub(last_tick);
                push_varlen(&mut buf, delta);
                match ev {
                    TrackEvent::NoteOn {
                        pitch, velocity, ..
                    } => {
                        buf.push(0x90 | (t.channel & 0x0F));
                        buf.push(*pitch & 0x7F);
                        buf.push(*velocity & 0x7F);
                    },
                    TrackEvent::NoteOff { pitch, .. } => {
                        buf.push(0x80 | (t.channel & 0x0F));
                        buf.push(*pitch & 0x7F);
                        buf.push(0);
                    },
                }
                last_tick = ev.tick_u32();
            }

            // EndOfTrack.
            push_varlen(&mut buf, 0);
            buf.extend_from_slice(&[0xFF, 0x2F, 0x00]);
            write_track(out, &buf);
        }
    }

    /// Bucket notes into per-program tracks with sorted, normalized
    /// events. Doesn't write anything — pure data-shaping pass.
    fn prepare_tracks(&self, notes: &[MidiNote]) -> Vec<TrackBuf> {
        const DRUM: u8 = crate::models::muscriptor::DRUM_PROGRAM;
        let mut tracks: std::collections::BTreeMap<u8, TrackBuf> =
            std::collections::BTreeMap::new();

        for n in notes {
            let track_name: String =
                n.instrument
                    .map(|s| s.replace('_', " "))
                    .unwrap_or_else(|| {
                        if n.program == DRUM {
                            "drums".to_string()
                        } else {
                            format!("program {}", n.program)
                        }
                    });
            let entry = tracks.entry(n.program).or_insert_with(|| TrackBuf {
                program: n.program,
                name: track_name,
                last_tick: 0,
                events: Vec::new(),
                channel: 0, // assigned below
            });
            let onset_tick = secs_to_tick(n.onset as f64, self.ticks_per_beat, self.tempo);
            let offset_tick = secs_to_tick(
                n.offset.max(n.onset) as f64,
                self.ticks_per_beat,
                self.tempo,
            );
            entry.events.push(TrackEvent::NoteOn {
                tick: onset_tick,
                pitch: n.pitch,
                velocity: self.velocity,
            });
            // Drums only carry an onset; their offset is silently dropped
            // (the upstream emits a 0-velocity `note_off` 10 ms after the
            // onset for DAW round-trip — we skip that to match the
            // "drums are onset-only" contract documented in the HF
            // model card).
            if n.program != DRUM && offset_tick > onset_tick {
                entry.events.push(TrackEvent::NoteOff {
                    tick: offset_tick,
                    pitch: n.pitch,
                });
            }
        }

        // Assign channels: drums get 9; everyone else gets 0..8, 10..15
        // in order of first appearance (same allocation policy as the
        // upstream `note_event2midi`).
        let mut next_channel: u8 = 0;
        for (_, t) in tracks.iter_mut() {
            t.channel = if t.program == DRUM {
                9
            } else {
                // Skip channel 9 (drum channel) for non-drums.
                if next_channel == 9 {
                    next_channel += 1;
                }
                let c = next_channel;
                next_channel += 1;
                c
            };
        }

        // Sort each track's events by (tick, NoteOn before NoteOff).
        for (_, t) in tracks.iter_mut() {
            t.events.sort_by(|a, b| {
                let ta = a.tick_u32();
                let tb = b.tick_u32();
                ta.cmp(&tb).then_with(|| match (a, b) {
                    (TrackEvent::NoteOn { .. }, TrackEvent::NoteOff { .. }) => {
                        std::cmp::Ordering::Less
                    },
                    (TrackEvent::NoteOff { .. }, TrackEvent::NoteOn { .. }) => {
                        std::cmp::Ordering::Greater
                    },
                    _ => std::cmp::Ordering::Equal,
                })
            });
        }

        tracks.into_values().collect()
    }
}

impl TrackEvent {
    fn tick_u32(self) -> u32 {
        match self {
            TrackEvent::NoteOn { tick, .. } | TrackEvent::NoteOff { tick, .. } => tick,
        }
    }
}

// ── MIDI-level helpers (variable-length quantity, track emission, etc.) ─

fn secs_to_tick(secs: f64, ticks_per_beat: u16, tempo_us: u32) -> u32 {
    if !secs.is_finite() || secs < 0.0 {
        return 0;
    }
    // ticks = secs * 1e6 / tempo * ticks_per_beat
    let ticks = secs * 1_000_000.0 * (ticks_per_beat as f64) / (tempo_us as f64);
    f64_to_u32(ticks.round().max(0.0))
}

fn build_header(num_tracks: u16, ticks_per_beat: u16) -> Vec<u8> {
    let mut h = Vec::with_capacity(14);
    h.extend_from_slice(b"MThd");
    h.extend_from_slice(&6u32.to_be_bytes()); // length
    h.extend_from_slice(&1u16.to_be_bytes()); // format = type 1
    h.extend_from_slice(&num_tracks.to_be_bytes());
    h.extend_from_slice(&ticks_per_beat.to_be_bytes());
    h
}

fn write_track<W: Write>(out: &mut W, bytes: &[u8]) {
    out.write_all(b"MTrk").expect("writing never fails");
    out.write_all(&(bytes.len() as u32).to_be_bytes())
        .expect("writing never fails");
    out.write_all(bytes).expect("writing never fails");
}

/// Push a MIDI variable-length quantity (1-4 bytes).
fn push_varlen(buf: &mut Vec<u8>, mut value: u32) {
    let mut tmp = [0u8; 4];
    let mut n = 0;
    tmp[3] = (value & 0x7F) as u8;
    value >>= 7;
    while value > 0 {
        n += 1;
        tmp[3 - n] = ((value & 0x7F) as u8) | 0x80;
        value >>= 7;
    }
    buf.extend_from_slice(&tmp[3 - n..=3]);
}

fn write_meta(buf: &mut Vec<u8>, delta: u32, meta_type: u8, data: &[u8]) {
    push_varlen(buf, delta);
    buf.push(0xFF);
    buf.push(meta_type & 0x7F);
    push_varlen(buf, data.len() as u32);
    buf.extend_from_slice(data);
}

fn encode_tempo(tempo_us_per_quarter: u32) -> [u8; 3] {
    [
        ((tempo_us_per_quarter >> 16) & 0xFF) as u8,
        ((tempo_us_per_quarter >> 8) & 0xFF) as u8,
        (tempo_us_per_quarter & 0xFF) as u8,
    ]
}

fn encode_time_signature(numerator: u8, log2_denominator: u8) -> [u8; 4] {
    [numerator & 0x7F, log2_denominator & 0x7F, 24, 8]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(pitch: u8, onset: f32, offset: f32, program: u8) -> MidiNote {
        MidiNote {
            program,
            pitch,
            onset,
            offset,
            instrument: None,
        }
    }

    #[test]
    fn empty_writes_valid_smf() {
        let bytes = MidiWriter::new().write(&[]);
        // "MThd" magic + (4-byte length=6) + format(2) + ntracks(2) +
        // division(2) = 14 bytes for the header.
        assert_eq!(&bytes[..4], b"MThd");
        assert_eq!(&bytes[8..10], &1u16.to_be_bytes()); // format 1
        assert_eq!(&bytes[10..12], &1u16.to_be_bytes()); // 1 track (conductor only)
    }

    #[test]
    fn one_piano_note_round_trip() {
        let notes = vec![n(60, 0.0, 0.5, 0)];
        let bytes = MidiWriter::new().write(&notes);
        // Sanity: starts with MThd
        assert_eq!(&bytes[..4], b"MThd");
        // Should contain MTrk twice (header + conductor + program track)
        let trk_count = bytes.windows(4).filter(|w| *w == b"MTrk").count();
        assert_eq!(trk_count, 2);
    }

    #[test]
    fn drum_track_uses_channel_nine() {
        let notes = vec![
            MidiNote {
                program: 128,
                pitch: 36,
                onset: 0.0,
                offset: 0.0,
                instrument: Some("drums"),
            },
            n(60, 0.0, 0.5, 0),
        ];
        let bytes = MidiWriter::new().write(&notes);
        // The drum track contains 0x99 (note_on ch9) somewhere.
        assert!(bytes.contains(&0x99), "drums should be on channel 9");
        // And a non-drum note_on uses channel 0 (0x90).
        assert!(bytes.contains(&0x90), "non-drum should use channel 0");
    }

    #[test]
    fn secs_to_tick_120bpm() {
        // 120 BPM ⇒ tempo = 500_000 µs / quarter = 0.5 s / quarter, so one
        // quarter note (480 ticks) is half a second and one full second is
        // *two* quarter notes = 960 ticks. (This test previously asserted
        // 480/240 — off by 2x; `secs_to_tick` itself was already correct.)
        assert_eq!(secs_to_tick(1.0, 480, 500_000), 960);
        assert_eq!(secs_to_tick(0.5, 480, 500_000), 480);
        assert_eq!(secs_to_tick(0.0, 480, 500_000), 0);
    }
}
