//! `topk_indices_rows` (one pair of launches for all rows) must equal running
//! `topk_indices` on each row separately, ties included. On Metal and SYCL
//! the per-row reference is the host sort.

#![cfg(any(feature = "cuda", feature = "metal", feature = "sycl"))]

use crane_core::candle_core::{DType, Device, Tensor};
use crane_core::ops::fused_ops::{topk_indices, topk_indices_rows};

#[test]
#[ignore = "needs a CUDA, Metal or SYCL GPU"]
fn batched_topk_matches_per_row() {
    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(0).unwrap();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let dev = Device::new_metal(0).unwrap();
    #[cfg(all(feature = "sycl", not(any(feature = "cuda", feature = "metal"))))]
    let dev = Device::new_sycl(0).unwrap();
    for (rows, n, k) in [
        (1usize, 5000usize, 16usize),
        (15, 248_094, 256),
        (7, 70_000, 40),
        (3, 600, 512),
    ] {
        // Quantised values force plenty of exact ties.
        let x = (Tensor::randn(0f32, 1.0, (rows, n), &dev).unwrap() * 8.0)
            .unwrap()
            .round()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let batched = topk_indices_rows(&x, k).unwrap().to_vec2::<u32>().unwrap();
        assert_eq!(batched.len(), rows);
        for (i, got) in batched.iter().enumerate() {
            let row = x
                .narrow(0, i, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .contiguous()
                .unwrap();
            let want = topk_indices(&row, k).unwrap().to_vec1::<u32>().unwrap();
            assert_eq!(got, &want, "row {i} of {rows}x{n} (k={k})");
        }
    }
}
