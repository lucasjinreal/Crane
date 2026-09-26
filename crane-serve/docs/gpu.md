# GPU Deployment

CUDA support requires the `cuda` feature flag at build time:

```bash
cargo build -p crane-serve --release --features cuda
```

The server automatically uses the first available CUDA device.

## Basic CUDA inference

```bash
crane-serve --model-path /path/to/Qwen3-8B-Instruct
```

`model_info` reports the device as `Cuda(0)` (or `Cuda(1)`, etc.).

## GPU memory control

GPU memory grows as KV caches accumulate. Use `--gpu-memory-limit` to keep
usage bounded:

```bash
# Hard cap at 8 GB — recommended starting point for a 12 GB GPU
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 8G \
    --context 4K

# Cap at 5 GB for 8 GB VRAM cards
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 5G \
    --context 2K \
    --max-concurrent 4

# Use 75% of total VRAM
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 0.75
```

When the KV memory budget is exceeded, the engine evicts the longest-output
sequence, preserving its state, and tightens the concurrency cap. It resumes
the evicted sequence automatically once load subsides. This avoids OOM
without crashing the server.

**Recommended values by GPU size:**

| GPU VRAM | `--gpu-memory-limit` | `--context` |
|----------|---------------------|-------------|
| 8 GB     | `6G` or `0.7`       | `2K`        |
| 12 GB    | `8G` or `0.7`       | `4K`        |
| 24 GB    | `20G` or `0.8`      | `8K`        |
| 48 GB+   | *(omit)*            | *(omit)*    |

The startup log prints `kv_bytes` and `kv_budget`. Monitor these to
validate your `--gpu-memory-limit` headroom.

### Auto-derived context length

If `--context`/`--max-seq-len` is left unset while `--gpu-memory-limit` is
set, and the loaded model supports it (currently Qwen3), crane-serve
computes a safe cap after loading from measured GPU memory, actual weight
usage, and the model's per-token memory cost — protecting a single
long-running session's own KV cache growth, which eviction alone doesn't
cover (see above). Check the startup log for `auto-derived N tokens...`;
if it instead logs a warning that no safe value could be derived
(unsupported model/device), set `--context` explicitly.

## GGUF quantized models on CUDA

GGUF quantization roughly halves VRAM usage compared to FP16:

```bash
crane-serve --model-path /path/to/Qwen3-8B-Q4_K_M.gguf \
    --format gguf \
    --gpu-memory-limit 8G
```

GGUF quantization is supported for Hunyuan Dense and Qwen 3. Qwen 2.5
requires the Safetensors format.

## Multi-GPU

crane-serve runs on a single CUDA device (device 0). Multi-GPU tensor
parallelism is not yet supported.

## AMD ROCm in Container

`container/rocm/Containerfile` builds `crane-serve --features rocm` against
Fedora's native ROCm 7.1 packages (builder: `fedora:44`) and ships it on
`fedora-minimal:44`. Fedora's `rocblas` package ships kernels for gfx900
through gfx950, covering AMD's officially-supported architecture list (e.g.
Strix Halo/gfx1151, RDNA4/gfx1201) rather than one GPU family. The root
`compose.yaml` runs it as service `crane-serve-rocm` under the `rocm`
profile, with the GPU devices mapped. Examples below use `podman`; swap in
`docker` if that's what you have installed:

```bash
COMPOSE_PROFILES=rocm MODEL_DIR=/path/to/models MODEL=Qwen3-4B \
    podman compose up --build
```

- The ROCm backend compiles its HIP kernels with `hipcc` on first use, so the
  runtime image includes the ROCm LLVM toolchain (~2.9 GB). Code objects are
  cached in `CANDLE_ROCM_CACHE_DIR` (`/var/cache/candle-rocm`); the compose
  file keeps it in a named volume so restarts skip recompilation.
- The container needs `/dev/kfd` and `/dev/dri`, plus the host's `render` and
  `video` GIDs (`getent group render video`); set them in `group_add`.
- Do not set `HSA_OVERRIDE_GFX_VERSION`; your GPU's real architecture is
  supported natively.
- Build without `-Z build-std=core`: it collides with the prebuilt `std`
  (E0152). The builder sets `RUSTC_BOOTSTRAP=1` and installs `rust-src` for
  the nested amdgcn kernel build that `rocm-rs` runs itself.
- Kernel micro-benchmarks: the Containerfile's optional `bench` target ships
  `gdn_bench [BH S K V iters]` and `topk_bench [N K iters]`; run them with
  `podman compose run --rm crane-bench-rocm [gdn_bench|topk_bench ARGS]`
  (profile `rocm-bench`). With no arguments it runs both at their defaults.

## Environment variables

These tune GPU-side sampling. They rarely need changing.

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU top-k even for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | k value for GPU top-k fallback |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Sample on CPU after GPU top-k |
