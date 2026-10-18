//! candle 0.11's Metal `argmax`/`argmin` along a dim that is not the last one
//! returns wrong indices when every dim after it has size 1, or when the
//! reduced dim itself has size 1 (see `data/docs/candle-metal-argmax.md`).
//! `crane_core::ops::arg_reduce` works around it; this checks the workaround
//! against the CPU on the shapes that break, and records the candle bug.

#![cfg(feature = "metal")]

use crane_core::candle_core::{DType, Device, Tensor};
use crane_core::ops::arg_reduce;

/// Shapes and dims candle 0.11 gets wrong on Metal, plus neighbours it gets
/// right (they must stay right through the workaround).
const BROKEN: &[(&[usize], usize)] = &[
    (&[300, 1], 0),
    (&[1, 300], 0),
    (&[4, 300, 1], 1),
    (&[5, 1, 7], 1),
];
const FINE: &[(&[usize], usize)] = &[
    (&[300, 2], 0),
    (&[5, 300], 0),
    (&[300, 1], 1),
    (&[3, 4, 300], 1),
    (&[300], 0),
];

fn metal() -> Option<Device> {
    crane_core::candle_core::utils::metal_is_available().then(|| Device::new_metal(0).unwrap())
}

/// A random tensor on Metal in `dtype`, and the same values on the CPU in f32.
fn pair(dev: &Device, shape: &[usize], dtype: DType) -> (Tensor, Tensor) {
    let gpu = Tensor::randn(0f32, 1.0, shape, dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let cpu = gpu
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    (gpu, cpu)
}

fn ids(t: Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_vec1::<u32>().unwrap()
}

#[test]
fn arg_reduce_matches_cpu_on_metal() {
    let Some(dev) = metal() else {
        return;
    };
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for &(shape, dim) in BROKEN.iter().chain(FINE) {
            let (gpu, cpu) = pair(&dev, shape, dtype);
            let cases: [(&str, fn(&Tensor, usize) -> _, fn(&Tensor, usize) -> _); 4] = [
                ("argmax", arg_reduce::argmax, Tensor::argmax::<usize>),
                (
                    "argmax_keepdim",
                    arg_reduce::argmax_keepdim,
                    Tensor::argmax_keepdim::<usize>,
                ),
                ("argmin", arg_reduce::argmin, Tensor::argmin::<usize>),
                (
                    "argmin_keepdim",
                    arg_reduce::argmin_keepdim,
                    Tensor::argmin_keepdim::<usize>,
                ),
            ];
            for (name, ours, cpu_op) in cases {
                let got = ours(&gpu, dim).unwrap();
                let want = cpu_op(&cpu, dim).unwrap();
                assert_eq!(
                    got.dims(),
                    want.dims(),
                    "{dtype:?} {shape:?} {name}({dim}) shape"
                );
                assert_eq!(ids(got), ids(want), "{dtype:?} {shape:?} {name}({dim})");
            }
        }
    }
}

/// Fails once candle fixes its Metal arg-reductions; then
/// `ops::arg_reduce` can go.
#[test]
#[ignore = "documents a candle 0.11 Metal bug"]
fn candle_metal_arg_reduce_bug_reproduces() {
    let Some(dev) = metal() else {
        return;
    };
    for &(shape, dim) in BROKEN {
        let (gpu, cpu) = pair(&dev, shape, DType::F32);
        let got = ids(gpu.argmax(dim).unwrap());
        let want = ids(cpu.argmax(dim).unwrap());
        println!(
            "{shape:?}.argmax({dim}): cpu {:?} metal {:?}",
            &want[..want.len().min(4)],
            &got[..got.len().min(4)]
        );
        assert_ne!(got, want, "{shape:?}.argmax({dim}) is right on Metal now");
    }
}
