# Crane on Intel SYCL / oneAPI (proof-of-concept)

candle 0.11 on crates.io has no SYCL backend, so the root `Cargo.toml`
`[patch.crates-io]` pins the three `candle-*` crates to
[`Hahihula/candle`](https://github.com/Hahihula/candle) rev `c91b22a` — candle
0.11.0 plus an **off-by-default** `sycl` feature. With `sycl` off it is stock
candle, so CPU/CUDA/Metal builds are unchanged. Not for `main`.

The fork adds, behind `sycl`:

- native SYCL kernels for `candle-nn`'s fused ops — `rms_norm`, `softmax`,
  `rope`, `sigmoid` (no CPU round-trip);
- a matmul "densify non-contiguous operands" fix (grouped-query attention);
- a small `pub` launch surface — `SyclStorage::{buf, elems, from_buffer}`,
  `SyclDevice::alloc_bytes`, `Queue::native_ptr()` — so an out-of-tree kernel can
  submit onto candle's in-order queue (`candle-sycl-matmul-fix.patch` here is the
  older standalone form of the matmul fix).

## What's wired in Crane

- `sycl` feature on `crane-core` / `crane` / `crane-serve` / `example` →
  `candle-*/sycl`.
- `DeviceConfig::Sycl(u32)` + `Device::new_sycl` in every device selector; the
  examples / `crane-serve` device ladder pick `Sycl(0)` when built `--features sycl`.
- **Fused Gated Delta Net recurrence** — `crane-core/kernels/sycl/gdn.cpp` (icpx →
  `libcrane_gdn_sycl.so` via `crane-core/build.rs`) + `ops/gdn/sycl_backend.rs`
  launcher, dispatched by `apply_recurrence` for `is_sycl()`. The SYCL
  counterpart of `kernels/cuda/gdn.cu`. `CRANE_GDN_PORTABLE=1` forces the
  op-by-op path.

## Requirements

The Intel oneAPI toolchain (`icpx`, oneMKL) plus the Level-Zero GPU runtime. The
`intel/oneapi-basekit` image has all of it; with `--device /dev/dri` an Intel GPU
is visible out of the box (`sycl-ls` lists a `level_zero:gpu` entry). The fork's
SYCL runtime **only enumerates GPU devices**, so a usable Intel GPU is required.

## One-shot container recipe

```bash
# build the image (oneAPI basekit + Rust + libssl/alsa/cmake)
docker build -t crane-sycl:dev -f contrib/sycl/Dockerfile contrib/sycl

contrib/sycl/run.sh build              # compile
contrib/sycl/run.sh test               # cargo test --test sycl_kernels (kernel vs portable)
CRANE_SYCL_MODELS=/path/to/models \
  contrib/sycl/run.sh chat -m /models/Qwen3.5-0.8B --max-new-tokens 200
```

`run.sh <cmd>` runs an arbitrary command in the container. It mounts the repo,
`~/.cargo/{registry,git}`, and a persistent `../crane-sycl-docker-target/` (kept
outside the repo — the container writes it as root). It sets `LD_LIBRARY_PATH` to
the two build.rs `OUT_DIR`s **prepended to** the image's baked oneAPI paths — do
not clobber those (they carry the split `libmkl_sycl_*` deps and the Level-Zero
adapter that GPU discovery needs).

## Status

Verified on an Intel Arc iGPU (Meteor Lake), safetensors + F16:

| Model | Arch | Notes |
|---|---|---|
| `Qwen3-0.6B` | dense | ~7–8 tok/s |
| `Qwen3.5-0.8B` | hybrid GDN + attention | fused GDN kernel; ~4.4 tok/s |

Both generate coherent text and exit cleanly. The fused GDN kernel matches the
portable reference exactly (`cos = 1.0` for K=128/64 × prefill/decode) and is
~15% faster at decode than the op-by-op path (3.8 → 4.4 tok/s).

### Not yet done

- The GDN kernel is a **naive v0** — no shared-memory staging of `k_t`/`q_t`, and
  the per-work-item state column (`Scol[128]`) spills to scratch on Intel. SLM
  tiling of the state is the next optimisation; `kernels/cuda/gdn.cu` is the
  reference for the math and the 4-way ILP split.
- GGUF path leaves SYCL on F32 side tensors (`qwen3_5/model.rs` `from_gguf` only
  gives CUDA/Metal/ROCm F16/BF16); add `|| device.is_sycl()` there for the F16
  memory win. Untested with a quantized Qwen3.5.
- Multi-Intel-GPU, `crane-serve` continuous batching, and vision towers untried.
