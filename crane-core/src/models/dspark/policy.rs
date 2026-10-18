//! When is speculative decoding worth it?
//!
//! A speculative round costs `round_secs` and yields `tokens` tokens; a plain
//! decode step costs `plain_secs` and yields one. Speculation wins exactly when
//! `round_secs / tokens < plain_secs`, i.e. when the acceptance (tokens per
//! round) beats the **break-even** `round_secs / plain_secs`. On BF16 weights
//! that is ~1.6 and the drafter essentially always clears it; on quantized GGUF
//! weights a plain step is so cheap (~5 ms) that the break-even is ~2.8, which
//! prose does not reach (1.7-2.4) while code and math easily do (4-9).
//!
//! [`AdaptivePolicy`] watches the acceptance, compares it to the measured
//! break-even and hands out *plain stretches*: when speculation is losing it
//! decodes `stretch_len()` tokens the plain way, then probes again. A switch
//! is not free (a round trip costs ~1.5-2 rounds: two batched passes), so it
//! only happens when the saving over the coming stretch,
//! `(round / tokens - plain) * stretch`, exceeds that measured cost; a
//! noisy dip near break-even is ridden out instead. A failed
//! probe doubles the next stretch (so steady prose costs little), a successful
//! one resets it (so a code block that starts mid-answer is picked up quickly).
//!
//! The policy only decides *when*; the state handling around a switch lives in
//! `minicpm_v::vlm`.

/// Rounds judged before the first switch of a generation.
const FIRST_MIN_ROUNDS: usize = 4;
/// Rounds judged after resuming from a plain stretch (a probe).
const PROBE_MIN_ROUNDS: usize = 3;
/// Tokens in the first plain stretch.
const BASE_STRETCH: usize = 32;
/// Tokens in the one-off calibration stretch that times a plain decode step.
const CALIBRATION_TOKENS: usize = 8;
/// Longest plain stretch between probes.
const MAX_STRETCH: usize = 128;
/// Weight of the newest round in the running average of the round cost.
const ALPHA: f64 = 0.25;
/// Rounds of acceptance history the decision looks at.
const WINDOW: usize = 16;
/// Standard errors added to the mean acceptance before judging it: a switch
/// must pay off even on an optimistic reading, because rounds yield anywhere
/// from 1 to 16 tokens and a short window is mostly noise.
const OPTIMISM: f64 = 0.75;
/// Margin applied to the break-even reported by [`AdaptivePolicy::threshold`].
const MARGIN: f64 = 1.05;
/// The saving over a stretch must beat the switch cost by this factor: the
/// acceptance estimate is noisy (a round yields 1 to 16 tokens), and a switch
/// that merely looks profitable loses once the probes that follow are counted.
const SAFETY: f64 = 1.25;
/// Cost of a switch round trip, in speculative rounds, until one is measured.
const DEFAULT_SWITCH_ROUNDS: f64 = 1.8;
/// Break-even assumed until a plain step has been timed: between BF16's ~1.6
/// and GGUF's ~2.8.
const PRIOR_BREAK_EVEN: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct AdaptivePolicy {
    enabled: bool,
    /// Seconds per plain decode step, once measured (kept across requests).
    plain_secs: Option<f64>,
    /// Tokens produced by the last [`WINDOW`] rounds since the last resume.
    window: std::collections::VecDeque<usize>,
    ewma_round_secs: Option<f64>,
    /// Rounds since the generation started or last resumed from plain.
    rounds_since_resume: usize,
    min_rounds: usize,
    stretch: usize,
    /// Testing aid: deterministic `(spec rounds, plain tokens)` cycle that
    /// replaces the cost-based decision.
    schedule: Option<(usize, usize)>,
    /// Measured cost of one speculation -> plain -> speculation round trip.
    switch_secs: Option<f64>,
    /// The coming plain stretch is the one-off calibration of `plain_secs`.
    calibrating: bool,
}

impl AdaptivePolicy {
    /// `plain_secs`: a previously measured plain decode step, if any.
    #[must_use]
    pub fn new(enabled: bool, plain_secs: Option<f64>) -> Self {
        Self {
            enabled,
            plain_secs,
            window: std::collections::VecDeque::with_capacity(WINDOW),
            ewma_round_secs: None,
            rounds_since_resume: 0,
            min_rounds: FIRST_MIN_ROUNDS,
            stretch: BASE_STRETCH,
            schedule: None,
            switch_secs: None,
            calibrating: false,
        }
    }

    /// Replace the cost model with a fixed cycle: speculate for `spec_rounds`
    /// rounds, then decode `plain_tokens` plainly, and repeat. For tests that
    /// need to exercise the switching deterministically.
    #[must_use]
    pub fn with_schedule(mut self, spec_rounds: usize, plain_tokens: usize) -> Self {
        self.enabled = true;
        self.schedule = Some((spec_rounds.max(1), plain_tokens.max(1)));
        self
    }

    /// Tokens per round speculation must reach to beat plain decoding.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        let break_even = match (self.ewma_round_secs, self.plain_secs) {
            (Some(round), Some(plain)) if plain > 0.0 => round / plain,
            _ => PRIOR_BREAK_EVEN,
        };
        (break_even * MARGIN).max(1.0)
    }

    /// Record a finished speculative round that produced `tokens` tokens in
    /// `secs`. Returns `true` if the generation should switch to plain decoding.
    pub fn on_round(&mut self, tokens: usize, secs: f64) -> bool {
        self.rounds_since_resume += 1;
        let blend = |old: Option<f64>, new: f64| old.map_or(new, |o| o + ALPHA * (new - o));
        if self.window.len() == WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(tokens);
        self.ewma_round_secs = Some(blend(self.ewma_round_secs, secs));
        if !self.enabled {
            return false;
        }
        if let Some((spec_rounds, _)) = self.schedule {
            return self.rounds_since_resume >= spec_rounds;
        }
        if self.rounds_since_resume < self.min_rounds {
            return false;
        }
        let losing = self.switching_pays();
        if losing {
            // A probe that fails right away backs off; a long successful run
            // before this starts over from the short stretch.
            let was_probe = self.rounds_since_resume <= PROBE_MIN_ROUNDS + 1
                && self.min_rounds == PROBE_MIN_ROUNDS;
            self.stretch = if was_probe {
                (self.stretch * 2).min(MAX_STRETCH)
            } else {
                BASE_STRETCH
            };
        }
        losing
    }

    /// Whether decoding the coming stretch plainly would save more than the
    /// switch round trip costs.
    fn switching_pays(&self) -> bool {
        let Some(round) = self.ewma_round_secs else {
            return false;
        };
        let n = self.window.len();
        if n < 2 {
            return false;
        }
        let mean = self.window.iter().sum::<usize>() as f64 / n as f64;
        let var = self
            .window
            .iter()
            .map(|&t| (t as f64 - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        let tokens = (mean + OPTIMISM * (var / n as f64).sqrt()).max(1.0);
        let plain = self.plain_secs.unwrap_or(round / PRIOR_BREAK_EVEN);
        let loss_per_token = round / tokens - plain;
        let switch_cost = self.switch_secs.unwrap_or(DEFAULT_SWITCH_ROUNDS * round);
        loss_per_token * self.stretch as f64 > SAFETY * switch_cost
    }

    /// Whether the plain decode step has never been timed, so the break-even
    /// is only a guess. If so, the caller should start the generation with a
    /// short plain stretch (its tokens are real output) and then call
    /// [`Self::on_plain_done`]; true at most once per process once the cost is
    /// remembered across requests.
    pub fn begin_calibration(&mut self) -> bool {
        self.calibrating = self.enabled && self.schedule.is_none() && self.plain_secs.is_none();
        self.calibrating
    }

    /// How many tokens the coming plain stretch should decode.
    #[must_use]
    pub fn stretch_len(&self) -> usize {
        if self.calibrating {
            return CALIBRATION_TOKENS;
        }
        self.schedule.map_or(self.stretch, |(_, plain)| plain)
    }

    /// A plain stretch finished; `step_secs` is the median plain step time.
    /// Prepares the probe that follows.
    pub fn on_plain_done(&mut self, step_secs: Option<f64>) {
        if let Some(s) = step_secs.filter(|s| *s > 0.0) {
            self.plain_secs = Some(self.plain_secs.map_or(s, |old| old + 0.5 * (s - old)));
        }
        self.rounds_since_resume = 0;
        // Calibration happens before the first judgement, so that one still
        // gets the full window; a probe after a real stretch gets the short one.
        if !self.calibrating {
            self.min_rounds = PROBE_MIN_ROUNDS;
        }
        self.calibrating = false;
        // Judge the probe on its own rounds, not on the stale history.
        self.window.clear();
    }

    /// A full round trip (speculation -> plain -> speculation) cost `secs`
    /// of switching overhead; remembered to price the next switch.
    pub fn on_switch_cost(&mut self, secs: f64) {
        if secs > 0.0 {
            self.switch_secs = Some(self.switch_secs.map_or(secs, |o| o + 0.5 * (secs - o)));
        }
    }

    /// The measured plain decode step (seconds), to carry to the next request.
    #[must_use]
    pub fn plain_secs(&self) -> Option<f64> {
        self.plain_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `n` identical rounds; return the 1-based round that asked to switch.
    fn first_switch(p: &mut AdaptivePolicy, n: usize, tokens: usize, secs: f64) -> Option<usize> {
        (1..=n)
            .find(|_| p.on_round(tokens, secs))
            .map(|_| p.rounds_since_resume)
    }

    #[test]
    fn disabled_never_switches() {
        let mut p = AdaptivePolicy::new(false, Some(0.005));
        assert_eq!(first_switch(&mut p, 50, 1, 0.015), None);
    }

    #[test]
    fn good_acceptance_stays_speculative() {
        // GGUF-like: 5 ms plain step, 14.6 ms round, break-even 2.9 -> 6 tokens wins.
        let mut p = AdaptivePolicy::new(true, Some(0.005));
        assert_eq!(first_switch(&mut p, 100, 6, 0.0146), None);
    }

    #[test]
    fn poor_acceptance_switches_after_the_judging_window() {
        let mut p = AdaptivePolicy::new(true, Some(0.005));
        // 1.7 tokens/round at 14.6 ms: 8.6 ms/token vs 5 ms plain -> lose.
        assert_eq!(first_switch(&mut p, 10, 2, 0.0146), Some(FIRST_MIN_ROUNDS));
    }

    #[test]
    fn bf16_break_even_is_low() {
        // 8.5 ms plain, 13.6 ms round: break-even 1.6; 1.8 tokens/round wins.
        let mut p = AdaptivePolicy::new(true, Some(0.0085));
        assert_eq!(first_switch(&mut p, 100, 2, 0.0136), None);
        assert!(p.threshold() < 1.8, "{}", p.threshold());
    }

    /// Feed a repeating, noisy acceptance pattern; true if it ever switches.
    fn ever_switches(p: &mut AdaptivePolicy, pattern: &[usize], round_secs: f64) -> bool {
        (0..200).any(|i| p.on_round(pattern[i % pattern.len()], round_secs))
    }

    #[test]
    fn noisy_acceptance_above_break_even_is_ridden_out() {
        // BF16 prose: ~2.6 tokens/round on average with plenty of 1-token
        // rounds, against a break-even of ~1.6. Dips must not trigger a switch.
        let noisy = [1, 2, 4, 1, 3, 2, 5, 1, 1, 3, 4, 2];
        let mut bf16 = AdaptivePolicy::new(true, Some(0.0085));
        bf16.on_switch_cost(0.025);
        assert!(!ever_switches(&mut bf16, &noisy, 0.0136));
    }

    #[test]
    fn quantized_prose_switches_but_quantized_code_does_not() {
        let prose = [1, 2, 1, 3, 2, 1, 2, 2, 1, 3]; // ~1.8 tokens/round
        let code = [6, 9, 4, 12, 7, 8, 5, 10]; // ~7.6
        let mut a = AdaptivePolicy::new(true, Some(0.005));
        a.on_switch_cost(0.025);
        assert!(ever_switches(&mut a, &prose, 0.0146));
        let mut b = AdaptivePolicy::new(true, Some(0.005));
        b.on_switch_cost(0.025);
        assert!(!ever_switches(&mut b, &code, 0.0146));
    }

    #[test]
    fn a_measured_expensive_switch_raises_the_bar() {
        let tokens_that_switch = |switch_secs: f64| {
            let mut p = AdaptivePolicy::new(true, Some(0.005));
            p.on_switch_cost(switch_secs);
            // highest whole token count per round that still switches
            (1..=6)
                .filter(|&t| first_switch(&mut p.clone(), 12, t, 0.0146).is_some())
                .max()
        };
        assert!(tokens_that_switch(0.200) < tokens_that_switch(0.010));
    }

    #[test]
    fn unknown_plain_cost_uses_the_prior() {
        let p = AdaptivePolicy::new(true, None);
        assert!((p.threshold() - PRIOR_BREAK_EVEN * MARGIN).abs() < 1e-9);
    }

    #[test]
    fn failed_probes_back_off_and_a_good_one_resets() {
        let mut p = AdaptivePolicy::new(true, Some(0.005));
        assert!(first_switch(&mut p, 10, 1, 0.0146).is_some());
        assert_eq!(p.stretch_len(), BASE_STRETCH);
        // Each probe fails after PROBE_MIN_ROUNDS rounds: 32 -> 64 -> 128 -> 128.
        for want in [64, 128, 128] {
            p.on_plain_done(Some(0.005));
            assert!(first_switch(&mut p, 10, 1, 0.0146).is_some());
            assert_eq!(p.stretch_len(), want);
        }
        // A probe that works stays speculative for a long while...
        p.on_plain_done(Some(0.005));
        assert_eq!(first_switch(&mut p, 30, 7, 0.0146), None);
        // ...and when it eventually degrades, the stretch is short again.
        assert!(first_switch(&mut p, 60, 1, 0.0146).is_some());
        assert_eq!(p.stretch_len(), BASE_STRETCH);
    }

    #[test]
    fn plain_cost_estimate_is_remembered_and_smoothed() {
        let mut p = AdaptivePolicy::new(true, None);
        p.on_plain_done(Some(0.006));
        assert_eq!(p.plain_secs(), Some(0.006));
        p.on_plain_done(Some(0.004));
        assert_eq!(p.plain_secs(), Some(0.005));
        p.on_plain_done(None);
        assert_eq!(p.plain_secs(), Some(0.005));
    }

    #[test]
    fn calibration_runs_once_until_the_plain_cost_is_known() {
        let mut p = AdaptivePolicy::new(true, None);
        assert!(p.begin_calibration());
        assert_eq!(p.stretch_len(), CALIBRATION_TOKENS);
        p.on_plain_done(Some(0.005));
        // The calibration stretch must not shrink the first judging window.
        assert_eq!(first_switch(&mut p, 10, 1, 0.0146), Some(FIRST_MIN_ROUNDS));
        // Known cost, remembered across requests: no further calibration.
        assert!(!AdaptivePolicy::new(true, p.plain_secs()).begin_calibration());
        // Never when disabled or scheduled.
        assert!(!AdaptivePolicy::new(false, None).begin_calibration());
        assert!(
            !AdaptivePolicy::new(false, None)
                .with_schedule(1, 1)
                .begin_calibration()
        );
    }

    #[test]
    fn schedule_overrides_the_cost_model() {
        let mut p = AdaptivePolicy::new(false, None).with_schedule(2, 7);
        assert!(!p.on_round(9, 0.01));
        assert!(p.on_round(9, 0.01));
        assert_eq!(p.stretch_len(), 7);
        p.on_plain_done(None);
        assert!(!p.on_round(9, 0.01));
        assert!(p.on_round(9, 0.01));
    }
}
