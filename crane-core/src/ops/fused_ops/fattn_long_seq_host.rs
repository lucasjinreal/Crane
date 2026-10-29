// SPDX-License-Identifier: MIT
//! Host-side helpers of the long-sequence flash-attention tests: tolerance
//! checks and the sweep override parser. They need no GPU, so this module is
//! gated on `test` alone and CPU CI runs its unit tests.

/// Mixed absolute/relative tolerance, `atol + rtol * |want|`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tolerance {
    pub(crate) atol: f32,
    pub(crate) rtol: f32,
}

impl Tolerance {
    /// The largest error allowed against the reference value `want`.
    fn limit(self, want: f32) -> f32 {
        self.atol + self.rtol * want.abs()
    }
}

/// Result of [`compare`].
#[derive(Debug, Default)]
pub(crate) struct Comparison {
    pub(crate) max_err: f32,
    /// Largest `err / limit`. Above `1.0` means failure.
    pub(crate) max_ratio: f32,
    pub(crate) failures: Vec<RowFailure>,
}

/// One `(row, head)` with at least one element out of tolerance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowFailure {
    pub(crate) row: usize,
    pub(crate) head: usize,
    pub(crate) max_err: f32,
    /// Number of elements out of tolerance.
    pub(crate) elems: usize,
    /// Index within `head_dim` of the element furthest outside tolerance
    /// (largest error over its limit), and the two values compared there.
    /// A NaN counts as the furthest.
    pub(crate) elem: usize,
    pub(crate) got: f32,
    pub(crate) want: f32,
}

/// Checks `got` against `want`, both flat `[rows, heads, d]`. A NaN in `got`
/// counts as out of tolerance.
pub(crate) fn compare(
    got: &[f32],
    want: &[f32],
    heads: usize,
    d: usize,
    tol: Tolerance,
) -> Comparison {
    assert_eq!(got.len(), want.len());
    let mut cmp = Comparison::default();
    for (c, (g, w)) in got.chunks_exact(d).zip(want.chunks_exact(d)).enumerate() {
        let mut row_err = 0f32;
        let mut worst = (0, 0f32, 0f32);
        let mut worst_ratio = 0f32;
        let mut elems = 0;
        for (e, (&g, &w)) in g.iter().zip(w).enumerate() {
            let err = (g - w).abs();
            let ratio = err / tol.limit(w);
            if err.is_nan() || ratio > 1.0 {
                elems += 1;
            }
            if (ratio.is_nan() && !worst_ratio.is_nan()) || ratio > worst_ratio {
                worst = (e, g, w);
                worst_ratio = ratio;
            }
            row_err = nan_max(row_err, err);
            cmp.max_ratio = nan_max(cmp.max_ratio, ratio);
        }
        cmp.max_err = nan_max(cmp.max_err, row_err);
        if elems > 0 {
            cmp.failures.push(RowFailure {
                row: c / heads,
                head: c % heads,
                max_err: row_err,
                elems,
                elem: worst.0,
                got: worst.1,
                want: worst.2,
            });
        }
    }
    cmp
}

/// `a.max(b)`, except that a NaN in either stays NaN. `f32::max` would drop
/// it and hide the failure.
pub(crate) fn nan_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// Parses `1024, 2048` into `[1024, 2048]`.
pub(crate) fn parse_seq_list(s: &str) -> Result<Vec<usize>, std::num::ParseIntError> {
    s.split(',').map(|part| part.trim().parse()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADS: usize = 8;

    // `compare` reports the failing row and head, and keeps a NaN visible
    // even when later elements are finite.
    #[test]
    fn compare_flags_nan_and_locates_the_row() {
        let d = 64;
        let want = vec![1.0f32; 2 * HEADS * d];
        let mut got = want.clone();
        let tol = Tolerance {
            atol: 2e-3,
            rtol: 1e-2,
        };
        assert!(compare(&got, &want, HEADS, d, tol).failures.is_empty());

        // Row 1, head 2: a NaN followed by finite elements.
        got[(HEADS + 2) * d] = f32::NAN;
        let cmp = compare(&got, &want, HEADS, d, tol);
        assert!(cmp.max_err.is_nan());
        let [f] = cmp.failures[..] else {
            panic!("expected one failing row, got {:?}", cmp.failures);
        };
        assert_eq!((f.row, f.head, f.elems), (1, 2, 1));
        assert!(f.max_err.is_nan());
    }

    // `compare` names the element furthest outside tolerance, and the values
    // it compared there.
    #[test]
    fn compare_reports_the_worst_element() {
        let d = 64;
        let want = vec![1.0f32; HEADS * d];
        let mut got = want.clone();
        got[5] += 0.5;
        got[9] += 2.0;
        got[11] -= 1.0;
        let tol = Tolerance {
            atol: 1e-2,
            rtol: 2e-2,
        };
        let cmp = compare(&got, &want, HEADS, d, tol);
        let [f] = cmp.failures[..] else {
            panic!("expected one failing row, got {:?}", cmp.failures);
        };
        assert_eq!((f.row, f.head, f.elems, f.elem), (0, 0, 3, 9));
        assert!((f.got - 3.0).abs() < 1e-6 && (f.want - 1.0).abs() < 1e-6);
    }

    // The sequence-length override accepts spaces and rejects junk.
    #[test]
    fn parse_seq_list_accepts_spaces_and_rejects_junk() {
        assert_eq!(
            parse_seq_list("1024, 2048 ,4096").unwrap(),
            [1024, 2048, 4096]
        );
        assert!(parse_seq_list("1024,abc").is_err());
        assert!(parse_seq_list("").is_err());
    }
}
