//! Canvas M-RoPE position ids for MiniCPM-V-4.7.
//!
//! 4.6 feeds the text decoder plain sequential positions. 4.7 instead lays
//! every image on a 2D "canvas": the thumbnail is stretched over the whole
//! slice grid, each slice sits at its grid offset, and the structural markers
//! (`<image>`, `</image>`, `<slice>`, `</slice>`, row-ending `\n`) are pinned
//! to the canvas corners/edges. Text keeps plain 1-D positions and resumes
//! after the canvas.
//!
//! Direct port of HF's `MiniCPMV4_7Model.get_rope_index` /
//! `_group_visual_frames` / `get_vision_position_ids`, restricted to a single
//! unpadded sequence of images (no video, no batching). The output rows are
//! `[T, H, W]`, matching what `Qwen3_5TextModel::forward_embeds` expects.
//!
//! Everything here is integer bookkeeping with one numeric trap:
//! `torch.linspace(..).round()` is evaluated in f32 and rounds ties to even,
//! so [`linspace_round`] reproduces both exactly (see the VoxCPM2 FSQ note in
//! `AGENTS.md` for the same trap).

/// Token ids of the markers the canvas layout is anchored on
/// (`config.json`: `image_start_id`, `image_end_id`, `slice_start_id`,
/// `slice_end_id`, `newline_id`).
#[derive(Debug, Clone, Copy)]
pub struct CanvasTokenIds {
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

impl CanvasTokenIds {
    fn is_marker(&self, id: u32) -> bool {
        id == self.slice_start
            || id == self.slice_end
            || id == self.image_start
            || id == self.image_end
            || id == self.newline
    }
}

/// Result of [`canvas_position_ids`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanvasPositions {
    /// `[T, H, W]` rows, each of length `seq_len`.
    pub position_ids: [Vec<i64>; 3],
    /// `max(position) + 1 - seq_len`: the offset decode adds to the cache
    /// length to get the next token's position.
    pub rope_deltas: i64,
}

/// One frame: a thumbnail crop plus the detail slices that tile it.
struct Frame {
    thumb: (usize, usize),
    slices: Vec<(usize, usize)>,
    end: usize,
}

struct Group {
    start: usize,
    frames: Vec<Frame>,
}

/// `torch.linspace(0, canvas - 1, n).round().long()`, bit-faithful: f32 math,
/// torch's symmetric fill order, ties-to-even rounding.
fn linspace_round(canvas: usize, n: usize) -> Vec<i64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let end = (canvas as f32) - 1.0;
    let step = end / ((n - 1) as f32);
    (0..n)
        .map(|i| {
            let v = if i < n / 2 {
                step * (i as f32)
            } else {
                end - step * ((n - i - 1) as f32)
            };
            v.round_ties_even() as i64
        })
        .collect()
}

/// `get_vision_position_ids`: `[T, H, W]` rows (row-major over the crop's LLM
/// tokens) for one crop. `canvas == (0, 0)` means "use the crop's own grid".
fn vision_position_ids(
    start: i64,
    grid: (usize, usize),
    canvas: (usize, usize),
    offset: (i64, i64),
    merge: usize,
) -> [Vec<i64>; 3] {
    let llm_h = grid.0 / merge;
    let llm_w = grid.1 / merge;
    let canvas_h = if canvas.0 == 0 { llm_h } else { canvas.0 };
    let canvas_w = if canvas.1 == 0 { llm_w } else { canvas.1 };
    let h = linspace_round(canvas_h, llm_h);
    let w = linspace_round(canvas_w, llm_w);
    let n = llm_h * llm_w;
    let mut rows = [vec![start; n], Vec::with_capacity(n), Vec::with_capacity(n)];
    for &hv in &h {
        for &wv in &w {
            rows[1].push(hv + offset.0 + start);
            rows[2].push(wv + offset.1 + start);
        }
    }
    rows
}

/// Split a sequence into visual groups (`_group_visual_frames`).
///
/// `is_visual[i]` marks tokens that are image placeholders. Each maximal run
/// of them is one crop; a crop preceded by `<slice>` is a detail slice of the
/// open frame, otherwise it is a new thumbnail. Frames separated only by
/// marker tokens share a group.
fn group_visual_frames(input_ids: &[u32], is_visual: &[bool], ids: &CanvasTokenIds) -> Vec<Group> {
    let seq_len = input_ids.len();
    let mut groups: Vec<Group> = Vec::new();
    let mut previous_end = 0usize;
    let mut i = 0usize;
    while i < seq_len {
        if !is_visual[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < seq_len && is_visual[i] {
            i += 1;
        }
        let crop = (start, i);
        let opens_slice =
            !groups.is_empty() && start > 0 && input_ids[start - 1] == ids.slice_start;
        if opens_slice {
            let group = groups.last_mut().expect("checked non-empty");
            let frame = group.frames.last_mut().expect("group has a frame");
            frame.slices.push(crop);
        } else {
            let gap_lo = previous_end.min(start.saturating_sub(1));
            let gap = &input_ids[gap_lo..start.saturating_sub(1)];
            let adjacent = !groups.is_empty() && gap.iter().all(|&t| ids.is_marker(t));
            let frame = Frame {
                thumb: crop,
                slices: Vec::new(),
                end: 0,
            };
            if adjacent {
                groups
                    .last_mut()
                    .expect("checked non-empty")
                    .frames
                    .push(frame);
            } else {
                groups.push(Group {
                    start: start.saturating_sub(1),
                    frames: vec![frame],
                });
            }
        }
        previous_end = i;
    }

    for group in &mut groups {
        for frame in &mut group.frames {
            let mut end = frame.slices.last().map_or(frame.thumb.1, |s| s.1);
            while end < seq_len && ids.is_marker(input_ids[end]) {
                end += 1;
            }
            frame.end = end;
        }
    }
    groups
}

fn fill_arange(row: &mut [i64], from: usize, to: usize, base: i64) {
    for (k, slot) in row[from..to].iter_mut().enumerate() {
        *slot = base + k as i64;
    }
}

/// Compute canvas M-RoPE positions for one sequence.
///
/// * `input_ids`: the full prompt, placeholders already expanded.
/// * `image_token_id`: id of `<|image_pad|>` (defines the visual spans).
/// * `target_sizes`: patch grid `(h, w)` of every crop in appearance order
///   (per image: thumbnail, then each slice row-major).
/// * `merge`: patches per LLM token per axis — 4 for `"16x"`, 2 for `"4x"`.
///
/// # Panics
///
/// Panics if `target_sizes` has fewer entries than the prompt has crops.
#[must_use]
pub fn canvas_position_ids(
    input_ids: &[u32],
    image_token_id: u32,
    target_sizes: &[(usize, usize)],
    ids: &CanvasTokenIds,
    merge: usize,
) -> CanvasPositions {
    let seq_len = input_ids.len();
    let is_visual: Vec<bool> = input_ids.iter().map(|&t| t == image_token_id).collect();
    let mut rows: [Vec<i64>; 3] = std::array::from_fn(|_| (0..seq_len as i64).collect());

    let mut grids = target_sizes.iter().copied();
    let mut next_grid = || grids.next().expect("target_sizes shorter than crop count");

    let mut current_pos = 0i64;
    let mut current_idx = 0usize;
    for group in group_visual_frames(input_ids, &is_visual, ids) {
        let group_end = group.frames.last().map_or(group.start, |f| f.end);

        // Plain 1-D text in front of the group.
        if group.start > current_idx {
            for row in &mut rows {
                fill_arange(row, current_idx, group.start, current_pos);
            }
            current_pos += (group.start - current_idx) as i64;
        }

        let mut frame_cursor = group.start;
        for frame in &group.frames {
            let thumb_grid = next_grid();
            let (thumb_start, thumb_end) = frame.thumb;
            let frame_start = thumb_start - 1;

            // Text between two frames of a clip; the extra step keeps the next
            // frame's halo from landing on the last of those tokens.
            if frame_start > frame_cursor {
                for row in &mut rows {
                    fill_arange(row, frame_cursor, frame_start, current_pos);
                }
                current_pos += (frame_start - frame_cursor) as i64 + 1;
            }

            let frame_end = frame.end.min(group_end);
            let origin = current_pos;
            let halo = (origin - 1).max(0);

            let mut canvas_h = thumb_grid.0 / merge;
            let mut canvas_w = thumb_grid.1 / merge;
            let (mut slice_h, mut slice_w) = (0usize, 0usize);
            let mut cols = 0usize;
            let mut first_slice_grid = (0usize, 0usize);
            if !frame.slices.is_empty() {
                // A gap wider than `</slice><slice>` is the newline ending a
                // row, which fixes the column count.
                cols = frame.slices.len();
                for k in 0..frame.slices.len() - 1 {
                    if frame.slices[k + 1].0 - frame.slices[k].1 > 2 {
                        cols = k + 1;
                        break;
                    }
                }
                let rows_n = frame.slices.len() / cols;
                if rows_n * cols != frame.slices.len() {
                    cols = frame.slices.len();
                }
                let rows_n = frame.slices.len() / cols;
                first_slice_grid = next_grid();
                slice_h = first_slice_grid.0 / merge;
                slice_w = first_slice_grid.1 / merge;
                canvas_h = rows_n * slice_h;
                canvas_w = cols * slice_w;
            }

            // Base coat for the whole frame, then `<image>` -> halo.
            for row in &mut rows {
                for slot in &mut row[frame_start..frame_end] {
                    *slot = origin;
                }
            }
            rows[1][frame_start] = halo;
            rows[2][frame_start] = halo;

            // `</image>` closes the thumbnail on the far corner of the canvas.
            if thumb_end < frame_end {
                rows[1][thumb_end] = origin + canvas_h as i64;
                rows[2][thumb_end] = origin + canvas_w as i64;
            }

            // Thumbnail stretched over the full canvas.
            let thumb_pos =
                vision_position_ids(origin, thumb_grid, (canvas_h, canvas_w), (0, 0), merge);
            for (row, vals) in rows.iter_mut().zip(&thumb_pos) {
                row[thumb_start..thumb_end].copy_from_slice(vals);
            }

            // Slices plus the markers pinning each crop's corners.
            for (k, &(s_start, s_end)) in frame.slices.iter().enumerate() {
                let h_off = ((k / cols) * slice_h) as i64;
                let w_off = ((k % cols) * slice_w) as i64;
                let grid = if k == 0 {
                    first_slice_grid
                } else {
                    next_grid()
                };
                let (sh, sw) = (grid.0 / merge, grid.1 / merge);

                let marker = s_start - 1;
                if marker >= frame_start {
                    rows[1][marker] = origin + h_off;
                    rows[2][marker] = origin + w_off;
                }
                if s_end < frame_end {
                    rows[1][s_end] = origin + h_off + sh as i64 - 1;
                    rows[2][s_end] = origin + w_off + sw as i64 - 1;
                }
                let pos = vision_position_ids(origin, grid, (0, 0), (h_off, w_off), merge);
                for (row, vals) in rows.iter_mut().zip(&pos) {
                    row[s_start..s_end].copy_from_slice(vals);
                }
            }

            // The "\n" ending a slice row sits just past that row's right edge.
            for k in 0..frame.slices.len().saturating_sub(1) {
                let (gap_start, gap_end) = (frame.slices[k].1, frame.slices[k + 1].0);
                if gap_end - gap_start <= 2 {
                    continue;
                }
                let boundary_h = (((k / cols) + 1) * slice_h) as i64 - 1;
                let right_edge_w = (cols * slice_w) as i64;
                for idx in gap_start + 1..gap_end - 1 {
                    rows[1][idx] = origin + boundary_h;
                    rows[2][idx] = origin + right_edge_w;
                }
            }

            current_pos = origin + canvas_h.max(canvas_w) as i64 + 1;
            frame_cursor = frame_end;
        }

        if frame_cursor < group_end {
            for row in &mut rows {
                fill_arange(row, frame_cursor, group_end, current_pos);
            }
            current_pos += (group_end - frame_cursor) as i64;
        }
        current_idx = group_end;
    }

    if current_idx < seq_len {
        for row in &mut rows {
            fill_arange(row, current_idx, seq_len, current_pos);
        }
    }

    let max_pos = rows
        .iter()
        .flat_map(|r| r.iter().copied())
        .max()
        .unwrap_or(0);
    CanvasPositions {
        position_ids: rows,
        rope_deltas: max_pos + 1 - seq_len as i64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_degenerate_cases() {
        assert_eq!(linspace_round(5, 0), Vec::<i64>::new());
        assert_eq!(linspace_round(5, 1), vec![0]);
        assert_eq!(linspace_round(4, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn linspace_rounds_ties_to_even() {
        // linspace(0, 3, 3) = [0, 1.5, 3] -> 1.5 rounds to 2 (even), like torch.
        assert_eq!(linspace_round(4, 3), vec![0, 2, 3]);
        // linspace(0, 5, 3) = [0, 2.5, 5] -> 2.5 rounds to 2 (even).
        assert_eq!(linspace_round(6, 3), vec![0, 2, 5]);
    }

    const IDS: CanvasTokenIds = CanvasTokenIds {
        image_start: 100,
        image_end: 101,
        slice_start: 102,
        slice_end: 103,
        newline: 198,
    };

    #[test]
    fn text_only_is_sequential() {
        let p = canvas_position_ids(&[1, 2, 3, 4], 9, &[], &IDS, 4);
        assert_eq!(p.position_ids[0], vec![0, 1, 2, 3]);
        assert_eq!(p.position_ids[1], vec![0, 1, 2, 3]);
        assert_eq!(p.rope_deltas, 0);
    }

    #[test]
    fn single_thumbnail_without_slices() {
        // text, <image> 2x2 tokens </image>, text. 8x8-patch grid, merge 4 -> 2x2.
        let ids = [1, 100, 9, 9, 9, 9, 101, 2];
        let p = canvas_position_ids(&ids, 9, &[(8, 8)], &IDS, 4);
        // `<image>` is the halo (origin-1 on H/W, origin on T): text token 1 is
        // at 0, so origin = 1 and the halo row/col is 0.
        assert_eq!(p.position_ids[0], vec![0, 1, 1, 1, 1, 1, 1, 4]);
        assert_eq!(p.position_ids[1], vec![0, 0, 1, 1, 2, 2, 3, 4]);
        assert_eq!(p.position_ids[2], vec![0, 0, 1, 2, 1, 2, 3, 4]);
    }

    /// Cross-check against HF `get_rope_index` on a real 1182x883 chart:
    /// 1 thumbnail + 6 slices in `"16x"` mode (487 tokens), and the same image
    /// in `"4x"` mode (1828 tokens). Fixtures produced by
    /// `ai/scripts47/hf_ref.py` (transformers 5.20.1).
    fn check_against_hf(fixture: &str, merge: usize) {
        let fx: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let ids: Vec<u32> = serde_json::from_value(fx["input_ids"].clone()).unwrap();
        let sizes: Vec<(usize, usize)> =
            serde_json::from_value::<Vec<[usize; 2]>>(fx["target_sizes"].clone())
                .unwrap()
                .into_iter()
                .map(|[h, w]| (h, w))
                .collect();
        let want: Vec<Vec<i64>> = serde_json::from_value(fx["pos"].clone()).unwrap();
        let want_delta = fx["rope_deltas"][0][0].as_i64().unwrap();
        let image_token_id = fx["image_token_id"].as_u64().unwrap() as u32;
        let tok = &fx["token_ids"];
        let marker = |k: &str| tok[k].as_u64().unwrap() as u32;
        let cids = CanvasTokenIds {
            image_start: marker("image_start"),
            image_end: marker("image_end"),
            slice_start: marker("slice_start"),
            slice_end: marker("slice_end"),
            newline: marker("newline"),
        };
        let got = canvas_position_ids(&ids, image_token_id, &sizes, &cids, merge);
        for (r, name) in ["T", "H", "W"].iter().enumerate() {
            assert_eq!(got.position_ids[r], want[r], "row {name} differs");
        }
        assert_eq!(got.rope_deltas, want_delta);
    }

    #[test]
    fn matches_hf_chart_16x() {
        check_against_hf(
            include_str!("../../../tests/fixtures/minicpm_v47_chart16x.json"),
            4,
        );
    }

    #[test]
    fn matches_hf_chart_4x() {
        check_against_hf(
            include_str!("../../../tests/fixtures/minicpm_v47_chart4x.json"),
            2,
        );
    }
}
