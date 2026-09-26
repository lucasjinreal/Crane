# Crane on Intel SYCL / oneAPI (proof-of-concept)

candle 0.11 on crates.io has no SYCL backend, so the root `Cargo.toml`
`[patch.crates-io]` points the three `candle-*` crates at the `sycle-support`
branch of [`Hahihula/candle`](https://github.com/Hahihula/candle) — candle
0.11.0 plus an **off-by-default** `sycl` feature. With `sycl` off it is stock
candle, so CPU/CUDA/Metal builds are unchanged.

The fork adds, behind `sycl`:

- native SYCL kernels for `candle-nn`'s fused ops — `rms_norm`, `softmax`,
  `rope`, `sigmoid`;
- an integer quantized mat-vec (`candle_sycl_mmvq_q8`, Q4_K/Q5_K/Q6_K/Q4_0/
  Q8_0) taking f16 or f32 activations, plus `candle_sycl_get_rows` for
  quantized embedding lookups;
- a matmul "densify non-contiguous operands" fix (grouped-query attention);
- `CANDLE_SYCL_TRACE_SYNC=1` / `CANDLE_SYCL_TRACE_DEQ=1` diagnostics;
- a small `pub` launch surface — `SyclStorage::{buf, elems, from_buffer}`,
  `SyclDevice::alloc_bytes`, `Queue::native_ptr()` — so an out-of-tree kernel
  can submit onto candle's in-order queue.

## What's wired in Crane

- `sycl` feature on `crane-core` / `crane` / `crane-serve` / `example`,
  forwarding to `candle-*/sycl`.
- `DeviceConfig::Sycl(u32)` + `Device::new_sycl` in every device selector; the
  examples and the `crane-serve` device ladder pick `Sycl(0)` when built
  `--features sycl`.
- **Fused Gated Delta Net recurrence** — `crane-core/kernels/sycl/gdn.cpp`
  (built by `crane-core/build.rs` with `icpx` into `libcrane_gdn_sycl.so`) plus
  the `ops/gdn/sycl_backend.rs` launcher, dispatched by `apply_recurrence` for
  `is_sycl()`. The SYCL counterpart of `kernels/cuda/gdn.cu`;
  `CRANE_GDN_PORTABLE=1` forces the op-by-op path.
- **Fused `SiLU(gate) * up`** — `crane-core/kernels/sycl/fused_ops.cpp` (same
  `.so`) plus `ops/fused_ops/sycl_impl.rs`, mirroring `rocm_impl.rs`. Wired
  into `Qwen3`, `HunyuanDense` and `SwiGluFfn`'s merged gate/up path. F32 and
  F16 only; BF16 bails with an error rather than truncating.

Everything else falls back to `ops/fused_ops/portable.rs`.

## Requirements

The Intel oneAPI toolchain (`icpx`, oneMKL) plus the Level-Zero GPU runtime.
The `intel/oneapi-basekit` image has all of it; with `--device /dev/dri` an
Intel GPU is visible out of the box (`sycl-ls` lists a `level_zero:gpu` entry).
The fork's SYCL runtime only enumerates GPU devices, so a usable Intel GPU is
required.

```bash
source /opt/intel/oneapi/setvars.sh          # skip inside the oneAPI container
cargo build --release --features sycl
```

The binaries bake an rpath to the kernel libraries and re-exec once with the
oneAPI runtime on `LD_LIBRARY_PATH` and the Level-Zero V2 adapter workaround
applied (`crane_core::utils::sycl_env::ensure_sycl_runtime_env`), so they run
straight from `target/release/` with nothing sourced — see "Driver notes"
below for what that workaround is and why it's needed.

## Docker

Two images, for two different jobs. Both are verified end to end (build,
`sycl-ls` enumerates the GPU, `crane-serve`/`chat_cli` select it and run) on
a discrete Arc Pro B70 — see "Status" below.

- **`Dockerfile.dev`** — a dev/build container. Nothing is baked in; mount the
  repo and build/run/test inside on each invocation via `run.sh`, so the
  compiled binary stays in sync with the mounted source.
- **`Dockerfile`** — a deployable image with `crane-serve` baked in, built
  multi-stage (oneAPI basekit `builder` → a slim `runtime` stage carrying only
  the SYCL/oneMKL/TBB runtime libraries and the Level-Zero GPU driver stack —
  2.9 GB vs. `Dockerfile.dev`'s 14 GB), mirroring `docker/rocm/Dockerfile`.

Both replace the oneAPI basekit's/Ubuntu's own bundled Level-Zero GPU driver
with a newer, version-matched set from GitHub releases — required for
Battlemage (Arc B-series, including the B70); see "Driver notes" below.

```bash
docker build -t crane-sycl:dev -f docker/sycl/Dockerfile.dev docker/sycl

docker/sycl/run.sh build              # compile
docker/sycl/run.sh test               # cargo test --test sycl_kernels
CRANE_SYCL_MODELS=/path/to/models \
  docker/sycl/run.sh chat -m /models/Qwen3.5-0.8B --max-new-tokens 200
```

`run.sh <cmd>` runs an arbitrary command in the container. It mounts the repo,
`~/.cargo/{registry,git}` and a persistent `../crane-sycl-docker-target/` (kept
outside the repo — the container writes it as root).

```bash
docker build -f docker/sycl/Dockerfile -t localhost/crane-serve:sycl .
docker run --rm --device /dev/dri:/dev/dri -p 8080:8080 \
  -v /path/to/models:/models:ro \
  localhost/crane-serve:sycl --model-path /models/Qwen3.5-0.8B
```

## Status

Verified on an Intel Arc iGPU (Meteor Lake) and a discrete Arc Pro B70
(Battlemage, `xe` driver, compute-runtime 26.31 / IGC 2.40, oneAPI 2026.1) on
bare metal, and separately in both `docker/sycl/Dockerfile.dev` and
`docker/sycl/Dockerfile` under rootless podman on the B70 (`sycl-ls` lists
`Intel(R) Arc(TM) Pro B70 Graphics` under `level_zero:gpu`; `crane-serve`
starts and `chat_cli` selects the device and gets past model loading).
Rootless-container GPU passthrough needs `--group-add keep-groups` (or the
exact host gid, e.g. `--group-add $(getent group render | cut -d: -f3)`) —
`--device /dev/dri` alone is not sufficient. Decode figures below are from the
bare-metal run: `CRANE_PROF=1` per-token wall time; the `avg tok/s` the CLI
prints includes prefill and reads lower on short runs.

| Model | Arch / format | Meteor Lake | B70 |
|---|---|---|---|
| `Qwen3-0.6B` | dense, safetensors F16 | ~7–8 tok/s | ~37 tok/s |
| `Qwen3.5-0.8B` | hybrid GDN, safetensors F16 | ~4.4 tok/s | ~35 tok/s |
| `Qwen3.5-0.8B` | hybrid GDN, GGUF Q4_K_M | — | ~40 tok/s |
| `Qwen3.8-27B` | hybrid GDN, GGUF Q4_K_M | — | ~11.5 tok/s |

The fused GDN kernel matches the portable reference exactly on both
(`cos = 1.0` for K=128/64 × prefill/decode, `--test sycl_kernels`) and is ~15%
faster at decode than the op-by-op path.

## Known gaps

- Decode is GPU-bound, not dispatch-bound: on the 27B, `enqueue/wall` sits at
  ~66%, so further launch cutting will not show up until the kernels get
  faster. ~3.7k launches per decoded token remain; the obvious next fusions are
  `compute_beta_g` and Qwen3.5's own MLP (which calls `candle_nn::ops::silu`
  rather than `crate::ops::fused_silu_mul` — true on CUDA/ROCm too).
- `gpu_argmax` and `topk_indices` still fall through to `portable.rs`.
  `topk_indices` copies the whole vocab to the host and sorts there on every
  sampled token when `top_k`/`top_p` is active; a native SYCL top-k mirroring
  `kernels/cuda/topk.cu` is the fix.
- The GDN kernel is a naive v0 — no shared-memory staging of `k_t`/`q_t`, and
  the per-work-item state column spills to scratch on Intel. SLM tiling is the
  next optimisation.
- Prefill beyond the mat-vec threshold dequantizes whole weights; an MMQ-style
  quantized GEMM is what that needs.
- Multi-GPU, `crane-serve` continuous batching, and vision towers untried.
- The pinned driver versions in `docker/sycl/Dockerfile`/`Dockerfile.dev`
  (`COMPUTE_RUNTIME_VERSION`, `IGC_VERSION`, `LEVEL_ZERO_VERSION` build args)
  need bumping — and re-verifying on real hardware — as newer GPU generations
  ship; see "Driver notes" below for why they're pinned at all.

## Driver notes (not code)

- **Battlemage (Arc B-series, including the B70) needs compute-runtime
  25.27+.** Both the oneAPI basekit image's bundled driver and Intel's own
  apt repo (`repositories.intel.com/gpu/ubuntu noble/unified`, checked at the
  time this was written) are older than that, so on Battlemage they enumerate
  zero GPU devices — no error, `sycl-ls`/`zeInit` just silently report none.
  Both `docker/sycl/Dockerfile` and `Dockerfile.dev` replace the driver with a
  version-matched set (`libze1`, `intel-igc-core-2`/`intel-igc-opencl-2`,
  `libigdgmm12`, `libze-intel-gpu1`, `intel-opencl-icd`) pulled from GitHub
  releases instead. The loader and driver share an ABI, so a newer driver
  paired with an older `libze1` (e.g. Ubuntu 24.04's stock 1.16.1) segfaults
  during enumeration rather than working or failing cleanly — bump the four
  version build args together, never independently.
- **oneAPI 2026.x defaults to the Level-Zero V2 adapter**, which fails on the
  first kernel submission from an in-order USM queue on Battlemage with
  `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` (44). It reproduces with a bare 10-line
  SYCL program, so it is an adapter/driver limitation.
  `crane_core::utils::sycl_env::ensure_sycl_runtime_env` sets
  `UR_LOADER_USE_LEVEL_ZERO_V2=0` automatically (forcing the legacy adapter)
  before anything touches SYCL, so this needs no manual step; set it yourself
  before launching if you're driving the SYCL runtime some other way (e.g. a
  bare `sycl-ls`).
- **Resizable BAR must be on.** Without it the compute runtime does not
  enumerate the GPU at all — `sycl-ls` shows only the CPU device and warns
  `Resizable BAR not detected`. Check with `lspci -v -s <bus> | grep "Memory at"`
  (the big prefetchable BAR should be VRAM-sized, not 256M). It can turn itself
  off across a reboot if the firmware cannot place the 64-bit window;
  `pci=realloc` on the kernel command line fixes the case where Linux, not the
  BIOS, gave up.
