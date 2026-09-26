# Container Images

Containerfiles for building `crane-serve` container images, one directory
per backend (`container/<backend>/`). Currently just `container/rocm/`;
more may follow (e.g. `container/cuda/`). Each backend directory's
`Containerfile` also has a `Dockerfile` symlink pointing at it: Podman
looks for `Containerfile` before falling back to `Dockerfile`, but Docker
only recognizes `Dockerfile` by default.

All variants build and run through the root [`compose.yaml`](../compose.yaml);
see the comment block at the top of that file for the full list of
services and profiles. Examples below use `podman`; swap in `docker` if
that's what you have installed, both read the same `compose.yaml`.

## ROCm

Two Containerfiles cover AMD ROCm:

- **`container/rocm/Containerfile`** (default, profile `rocm`) — builds
  against Fedora's native ROCm 7.1 packages, covering AMD's
  officially-supported architecture list (e.g. Strix Halo/gfx1151, RDNA4/
  gfx1201) from one image.
- **`container/rocm/Containerfile.strix-halo`** (profile `rocm-strixhalo`) —
  builds against the community `kyuz0/amd-strix-halo-toolboxes` image
  instead. Strix Halo (gfx1151) only, kept in case that toolbox ships newer
  or better-tuned kernels for that one chip than Fedora's own ROCm packages
  do at any given time.

Both produce the same `crane-serve --features rocm` binary and accept the
same environment knobs; only the build differs.

### Build & run

```bash
# MODEL is a model directory or single .gguf file under MODEL_DIR;
# CRANE_PORT sets the host port (default 8080).
COMPOSE_PROFILES=rocm MODEL_DIR=~/models MODEL=Qwen3-4B/Qwen3-4B-Q6_K.gguf \
    podman compose up -d --build
curl localhost:8080/v1/models

# Strix-Halo-toolbox variant instead:
COMPOSE_PROFILES=rocm-strixhalo MODEL_DIR=~/models MODEL=Qwen3-4B/Qwen3-4B-Q6_K.gguf \
    podman compose up -d --build
```

Put those variables in a `.env` file next to `compose.yaml` (git-ignored)
and a plain `podman compose up -d` does the same. `.env` values can
reference your environment, so no home directory is hard-coded:

```bash
# .env
COMPOSE_PROFILES=rocm
MODEL_DIR=${HOME}/.cache/huggingface/hub
MODEL=models--unsloth--Qwen3-4B-128K-GGUF/snapshots/<rev>/Qwen3-4B-128K-UD-Q6_K_XL.gguf
CRANE_PORT=8080
```

With no profile selected, nothing starts. To build only the image:
`podman build -f container/rocm/Containerfile -t localhost/crane-serve:rocm10 .`
(the context is the repo root; swap in `Containerfile.strix-halo` for that
variant).

### Micro-benchmarks

Each Containerfile's optional `bench` target builds `gdn_bench` and
`topk_bench` (see the root [README.md](../README.md)'s Qwen 3.5 section for
what each measures). Compose runs them as `crane-bench-rocm`/
`crane-bench-rocm-strixhalo` under their own `rocm-bench`/
`rocm-strixhalo-bench` profiles, so `podman compose up` under `rocm`/
`rocm-strixhalo` never starts them:

```bash
podman compose run --rm crane-bench-rocm                    # both, default args
podman compose run --rm crane-bench-rocm gdn_bench 16 512 128 128 100  # BH S K V iters
podman compose run --rm crane-bench-rocm topk_bench 248320 40 200      # N K iters
CRANE_TOPK_HOST=1 podman compose run --rm crane-bench-rocm topk_bench  # A/B: both arms host sort
```

`MODEL_TYPE`, `FORMAT`, `EXTRA_ARGS`, `RUST_LOG`, `CRANE_PROF`,
`CRANE_PROF_EVERY`, `CRANE_TOPK_HOST`, `CRANE_GDN_PORTABLE`,
`CRANE_GRAMMAR_TRACE`, `CRANE_VRAM_TRACE`, `HIP_LAUNCH_BLOCKING` and
`AMD_SERIALIZE_KERNEL` are passed through to `crane-serve-rocm`/
`crane-serve-rocm-strixhalo` when set in the shell or `.env` (see
`compose.yaml`'s `environment:` block); the bench services only forward
`CRANE_TOPK_HOST`. `gdn_bench` calls the fused kernel directly, so
`CRANE_GDN_PORTABLE` and `CRANE_PROF` do not change its numbers.

### Notes

- `group_add` uses numeric host GIDs (render=105, video=39 here); check
  yours with `getent group render video`. Do not set
  `HSA_OVERRIDE_GFX_VERSION`; your GPU's real architecture is supported
  natively.
- The runtime image (~2.9 GB) carries `hipcc` and the ROCm LLVM toolchain,
  because kernels are compiled on first use. They are cached in
  `/var/cache/candle-rocm` (a named volume in the compose file), so only
  the first start pays that cost. The Strix-Halo-toolbox variant uses a
  separate `candle-rocm-cache-strixhalo` volume, since a different
  toolchain compiled those code objects.
- `ROCBLAS_LAYER=4` in the compose file is rocBLAS profile logging. Drop
  it for quieter logs.
- Do not build with `-Z build-std=core`: it collides with the prebuilt
  `std` (E0152). `RUSTC_BOOTSTRAP=1` and `rust-src` are enough for
  `rocm-rs`'s nested amdgcn build.
- On SELinux hosts (Podman), keep `label=disable`.
- The compose file passes `--host [::]` so the server also listens on
  IPv6. With the default `0.0.0.0`, `http://localhost:…` fails under
  rootless podman (pasta forwards `::1` as IPv6) while `127.0.0.1` works.
- A Hugging Face cache works as `MODEL_DIR` (`MODEL_DIR=~/.cache/huggingface/hub
  MODEL=models--<org>--<repo>/snapshots/<rev>/<file>.gguf`). Mount the
  whole `hub/` directory: snapshot files are symlinks into `../../blobs`.
- More detail: [crane-serve/docs/gpu.md](../crane-serve/docs/gpu.md).
