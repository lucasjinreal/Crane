//! Env-gated forward-pass profiling (`CRANE_PROF=1`).
//!
//! Answers one question the usual counters cannot: at decode's `seq_len == 1`
//! every op is a separate kernel launch, so the forward pass can be limited by
//! how fast the *CPU* enqueues work rather than by how fast the GPU runs it.
//! `rocm-smi`'s "GPU busy" reads high in both cases — it samples whether a
//! kernel is resident, not whether the queue ever starves — so it cannot tell
//! them apart.
//!
//! Each pass is therefore timed twice: `enqueue` is wall time to the point the
//! last op has been submitted, `wall` is after an explicit device sync. If the
//! two are close the pass is dispatch-bound and no kernel will fix it; if
//! `enqueue` is a small fraction of `wall` the GPU is the bottleneck.
//!
//! Span timings are CPU-side only and take no syncs, so they measure submission
//! cost without perturbing it — which is exactly the quantity of interest when
//! the answer is "dispatch-bound". They do *not* attribute GPU time; use
//! `rocprof` for that.
//!
//! Everything here compiles to a predictable-branch no-op when the variable is
//! unset: [`timed`] calls the closure directly and [`pass`] returns `None`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use candle_core::Device;

/// One measured region of the forward pass.
///
/// The variants form three non-overlapping tiers: [`Span::Embed`]..=[`Span::Head`]
/// partition the whole pass, [`Span::GdnProj`]..=[`Span::GdnFinish`] partition
/// [`Span::Gdn`], and [`Span::GdnPrep`]..=[`Span::GdnPost`] partition
/// [`Span::GdnRecur`]. Each tier is reported on its own line and should sum to
/// its parent.
#[derive(Clone, Copy)]
pub enum Span {
    // Tier 1 — the whole pass.
    Embed,
    BlockNorm,
    Attn,
    Gdn,
    Mlp,
    Resid,
    Head,
    // Tier 2 — inside `Gdn`.
    GdnProj,
    GdnConv,
    GdnQkv,
    GdnRecur,
    GdnFinish,
    // Tier 3 — inside `GdnRecur`.
    GdnPrep,
    GdnLaunch,
    GdnPost,
}

const NUM_SPANS: usize = 15;
const TIER1: std::ops::Range<usize> = 0..7;
const TIER2: std::ops::Range<usize> = 7..12;
const TIER3: std::ops::Range<usize> = 12..15;

const NAMES: [&str; NUM_SPANS] = [
    "embed", "norm", "attn", "gdn", "mlp", "resid", "head", //
    "proj", "conv", "qkv", "recur", "finish", //
    "prep", "launch", "post",
];

static SPAN_NS: [AtomicU64; NUM_SPANS] = [const { AtomicU64::new(0) }; NUM_SPANS];

/// Whether `CRANE_PROF` is set to something other than `0`.
#[must_use]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("CRANE_PROF") {
        Ok(v) => !matches!(v.trim(), "" | "0"),
        Err(_) => false,
    })
}

/// Passes per summary line; `CRANE_PROF_EVERY`, default 64.
fn report_every() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CRANE_PROF_EVERY")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(64)
    })
}

/// Run `f`, charging its wall time to `span`.
///
/// Nesting is allowed as long as the nested span belongs to a lower tier; a
/// span never subtracts its children, so overlapping spans within one tier
/// would double-count.
#[inline]
pub fn timed<T>(span: Span, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    #[allow(clippy::cast_possible_truncation)]
    SPAN_NS[span as usize].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

/// Running totals for one kind of pass (decode or prefill).
#[derive(Default, Clone, Copy)]
struct Totals {
    passes: u64,
    tokens: u64,
    enqueue_ns: u64,
    wall_ns: u64,
    spans: [u64; NUM_SPANS],
}

static TOTALS: Mutex<[Totals; 2]> = Mutex::new([
    Totals {
        passes: 0,
        tokens: 0,
        enqueue_ns: 0,
        wall_ns: 0,
        spans: [0; NUM_SPANS],
    },
    Totals {
        passes: 0,
        tokens: 0,
        enqueue_ns: 0,
        wall_ns: 0,
        spans: [0; NUM_SPANS],
    },
]);

/// A forward pass being measured. Created by [`pass`], closed by
/// [`PassTimer::finish`].
pub struct PassTimer {
    start: Instant,
    seq_len: usize,
    spans_at_start: [u64; NUM_SPANS],
}

/// Begin timing a forward pass over `seq_len` tokens, or `None` when profiling
/// is off.
#[must_use]
pub fn pass(seq_len: usize) -> Option<PassTimer> {
    if !enabled() {
        return None;
    }
    Some(PassTimer {
        start: Instant::now(),
        seq_len,
        spans_at_start: std::array::from_fn(|i| SPAN_NS[i].load(Ordering::Relaxed)),
    })
}

impl PassTimer {
    /// Close the pass: record enqueue time, synchronize `device`, record wall
    /// time, and emit a summary every `CRANE_PROF_EVERY` passes of this kind.
    ///
    /// The sync is what separates submission cost from execution cost, so it is
    /// deliberate rather than incidental — it only ever runs under
    /// `CRANE_PROF`.
    pub fn finish(self, device: &Device) {
        let enqueue = self.start.elapsed();
        // A sync failure here is a profiling artifact, not a model error, and
        // the caller has no useful response to it — the pass itself succeeded.
        if let Err(e) = device.synchronize() {
            eprintln!("[crane-prof] device sync failed, dropping sample: {e}");
            return;
        }
        let wall = self.start.elapsed();

        // Decode is a single token; anything longer is a prefill (chunk).
        let kind = usize::from(self.seq_len > 1);
        let Ok(mut totals) = TOTALS.lock() else {
            return;
        };
        let t = &mut totals[kind];
        t.passes += 1;
        t.tokens += self.seq_len as u64;
        t.enqueue_ns += enqueue.as_nanos() as u64;
        t.wall_ns += wall.as_nanos() as u64;
        for ((total, span), start) in t
            .spans
            .iter_mut()
            .zip(&SPAN_NS)
            .zip(&self.spans_at_start)
        {
            *total += span.load(Ordering::Relaxed) - *start;
        }

        if t.passes % report_every() == 0 {
            let snapshot = *t;
            *t = Totals::default();
            drop(totals);
            report(kind, &snapshot);
        }
    }
}

/// Emit one window's summary, normalized per pass.
fn report(kind: usize, t: &Totals) {
    let label = if kind == 0 { "decode" } else { "prefill" };
    #[allow(clippy::cast_precision_loss)]
    let per = |ns: u64| ns as f64 / t.passes as f64 / 1e6;
    let (enqueue, wall) = (per(t.enqueue_ns), per(t.wall_ns));
    let ratio = if wall > 0.0 { enqueue / wall * 100.0 } else { 0.0 };

    let line = |range: std::ops::Range<usize>| {
        range
            .map(|i| format!("{} {:.2}", NAMES[i], per(t.spans[i])))
            .collect::<Vec<_>>()
            .join("  ")
    };
    #[allow(clippy::cast_precision_loss)]
    let sum = |range: std::ops::Range<usize>| range.map(|i| per(t.spans[i])).sum::<f64>();

    // `eprintln!`, not a log macro: crane-core has no logging facade, and an
    // explicitly opted-in diagnostic that silently needs a second variable
    // (`RUST_LOG`) to appear is the trap `CRANE_SAMPLE_TRACE` already fell into.
    eprintln!(
        "[crane-prof] {label} n={} tokens={} | enqueue {enqueue:.2} ms  wall {wall:.2} ms  \
         enqueue/wall {ratio:.0}%",
        t.passes, t.tokens,
    );
    eprintln!("[crane-prof]   pass:  {} | sum {:.2} ms", line(TIER1), sum(TIER1));
    eprintln!("[crane-prof]   gdn:   {} | sum {:.2} ms", line(TIER2), sum(TIER2));
    eprintln!("[crane-prof]   recur: {} | sum {:.2} ms", line(TIER3), sum(TIER3));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With `CRANE_PROF` unset, `timed` must still return the closure's value
    /// and `pass` must hand back nothing to time.
    #[test]
    fn disabled_is_transparent() {
        assert!(!enabled(), "CRANE_PROF must be unset in the test environment");
        assert_eq!(timed(Span::Embed, || 7), 7);
        assert!(pass(1).is_none());
        assert_eq!(SPAN_NS[Span::Embed as usize].load(Ordering::Relaxed), 0);
    }

    /// The three tiers must partition the span list exactly — a span left out
    /// of every tier would be recorded and never reported.
    #[test]
    fn tiers_cover_every_span() {
        assert_eq!(TIER1.start, 0);
        assert_eq!(TIER1.end, TIER2.start);
        assert_eq!(TIER2.end, TIER3.start);
        assert_eq!(TIER3.end, NUM_SPANS);
        assert_eq!(NAMES.len(), NUM_SPANS);
    }
}
