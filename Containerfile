# Runs crane-serve on an AMD ROCm GPU (see CONTAINER.md for the
# research/decisions behind every choice below). The model itself is not
# baked into the image -- mount a host directory with the model files to
# /models and point MODEL_PATH at the file/dir to load (see compose.yaml).
#
# Build:
#   podman build -t crane-serve-rocm .
#
# Run (needs the host's ROCm device nodes and, on rootless Podman, group
# access to them -- adjust --group-add for your host's render group if
# `keep-groups` isn't enough):
#   podman run --rm -it \
#     --device /dev/kfd --device /dev/dri --group-add keep-groups \
#     -p 8080:8080 \
#     -v /path/to/models:/models:ro \
#     -v crane-rocm-kernel-cache:/cache/candle-rocm \
#     -e MODEL_PATH=/models/llm/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q5_K_M.gguf \
#     -e RUST_LOG=crane_serve=debug,crane_core=debug \
#     crane-serve-rocm

# ---------------------------------------------------------------------
# Builder: compiles crane-serve against the pinned candle ROCm fork.
# ---------------------------------------------------------------------
FROM rocm/dev-ubuntu-24.04:7.14.0-full AS builder

ENV ROCM_PATH=/opt/rocm
ENV DEBIAN_FRONTEND=noninteractive

# libssl-dev/pkg-config: reqwest/hf-hub's default-tls (native-tls) backend
# links system OpenSSL and needs headers to build openssl-sys from source
# -- unrelated to ROCm, needed for any crane-serve build.
# libclang-dev: rocm-rs's build.rs runs bindgen over the HIP headers and
# needs libclang.so at *build* time (distinct from the clang inside
# /opt/rocm/llvm that hipcc uses).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libssl-dev \
        pkg-config \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . .

RUN cargo build --release -p crane-serve --features rocm

# ---------------------------------------------------------------------
# Runtime: same ROCm base as the builder, deliberately not a slimmer
# runtime-only image. candle's rocm feature JIT-compiles kernels via
# hipcc/clang-offload-bundler on first use, at *runtime*, not just at
# build time -- a runtime-libs-only image would be missing the compiler
# the first inference request needs.
# ---------------------------------------------------------------------
FROM rocm/dev-ubuntu-24.04:7.14.0-full

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV DEBIAN_FRONTEND=noninteractive
# rocm-rs's build.rs links crane-serve against /opt/rocm/lib's libraries
# (libamdhip64.so, librocblas.so, ...) via an explicit `-L` search path at
# *build* time, but that doesn't add an rpath to the binary or register
# the path with the dynamic linker for *run* time -- without this, the
# binary fails at startup with e.g. "libamdhip64.so.7: cannot open shared
# object file", even inside this same ROCm image.
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# No default set here: compose.yaml sets it (overridable via the host's
# RUST_LOG); pass -e RUST_LOG=... explicitly for a direct `podman run`
# (see example above) since tracing_subscriber otherwise falls back to
# "info" (crane-serve/src/lib.rs).

RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/target/release/crane-serve /usr/local/bin/crane-serve
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Kernels JIT-compile on first use and are cached on disk (see
# CONTAINER.md); mount a volume here so a container restart doesn't pay
# the compile cost again.
ENV CANDLE_ROCM_CACHE_DIR=/cache/candle-rocm
VOLUME ["/cache/candle-rocm"]

# crane-serve's own CLI defaults are already --host 0.0.0.0 --port 8080,
# i.e. reachable from outside the container once the port is published.
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
