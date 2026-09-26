# syntax=docker/dockerfile:1
#
# crane-serve with the Intel oneAPI / SYCL backend (proof-of-concept), targeting
# Intel Arc GPUs (verified on a Meteor Lake iGPU and a discrete Arc Pro B70 /
# Battlemage — see docker/sycl/README.md).
#
#   docker build -f docker/sycl/Dockerfile -t localhost/crane-serve:sycl .
#
# Stages:
#   builder   - oneAPI basekit (icpx, oneMKL, Level-Zero) + `crane-serve` build
#   runtime   - slim base + just the oneAPI/Level-Zero *runtime* libraries the
#               binary dlopens, copied out of the builder image
#
# The build context is the repository root; the root .dockerignore applies
# (if present).
#
# Unlike docker/rocm/Dockerfile, the runtime stage here does NOT need a
# compiler toolchain: `crane-core/kernels/sycl/*.cpp` are compiled by `icpx`
# at `cargo build` time (see `crane-core/build.rs`), not JIT-compiled on first
# use like candle's ROCm backend and Crane's CUDA kernels under ROCm. So the
# runtime stage only needs the SYCL/oneMKL/TBB shared libraries and the
# Level-Zero GPU driver stack, not the full basekit image.
#
# Built and run end to end against a discrete Arc Pro B70 (rootless podman —
# `sycl-ls` lists the device, `crane-serve` starts, all shared libraries
# resolve) — see docker/sycl/README.md's "Status" section.

ARG BUILDER_IMAGE=docker.io/intel/oneapi-basekit:2025.1.0-0-devel-ubuntu24.04
ARG RUNTIME_IMAGE=docker.io/library/ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BUILDER_IMAGE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config git \
        libssl-dev cmake libasound2-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Rust (edition 2024 needs >= 1.85; pin to match the host stable toolchain).
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    CARGO_TARGET_DIR=/build/target \
    PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.93.0

WORKDIR /src
COPY . .

# The basekit image bakes oneAPI's env (MKLROOT, CPATH, LIBRARY_PATH, ...) in
# as plain image-level `ENV`, so it's already present for any `RUN` here
# regardless of shell — no need to source `setvars.sh` (which, if it detects
# it already ran, exits non-zero and aborts the build under `set -e`).
#
# The out-of-tree SYCL kernel libraries it dlopens (libcandle_sycl.so,
# libcrane_gdn_sycl.so — built into their own crates' OUT_DIRs by
# candle-sycl-kernels' and crane-core's build scripts) are copied out to
# /out/lib here too: they're NOT covered by the binary's own rpath
# (build_sycl_rpath.rs's comment: "a dependency's cargo::rustc-link-arg does
# not reach the final binary's link"; same reason docker/sycl/run.sh's
# RUN_PRE finds and adds them to LD_LIBRARY_PATH for the dev container), and
# their OUT_DIR only exists inside this RUN's `/build/target` cache mount, so
# they must be pulled out to a normal layer path now rather than in a later
# RUN.
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    --mount=type=cache,target=/build/target <<-EOF
    set -eux
    cargo build --release -p crane-serve --bin crane-serve \
        --no-default-features --features sycl
    install -Dm755 /build/target/release/crane-serve /out/crane-serve
    mkdir -p /out/lib
    find /build/target/release/build -name 'libcandle_sycl.so' -o -name 'libcrane_gdn_sycl.so' \
        | xargs -I{} cp {} /out/lib/
EOF

# Trimmed copy of the oneAPI runtime libraries crane-serve dlopens: exactly
# the directory list `build_sycl_rpath.rs` (`oneapi_runtime_dirs`) bakes into
# the binary's own rpath — the SYCL runtime + Unified Runtime adapters
# (compiler), oneMKL, the task/completion-manager and unified-memory-framework
# libs (tcm, umf) and TBB. Headers, static archives, CMake/pkg-config files
# and the compiler binaries (icpx, icx, ...) are dropped: the runtime image
# never compiles anything.
RUN <<-EOF
    set -eux
    mkdir -p /oneapi-rt/lib
    for d in \
        compiler/latest/lib \
        mkl/latest/lib \
        mkl/latest/lib/intel64 \
        tcm/latest/lib \
        umf/latest/lib \
        tbb/latest/lib/intel64/gcc4.8 \
    ; do
        src="/opt/intel/oneapi/$d"
        [ -d "$src" ] && cp -a "$src/." /oneapi-rt/lib/
    done
    find /oneapi-rt -name '*.a' -delete
    rm -rf /oneapi-rt/lib/cmake /oneapi-rt/lib/pkgconfig
    # The out-of-tree kernel libs the previous RUN pulled out to /out/lib —
    # into the same directory as the oneAPI runtime libs above so one
    # LD_LIBRARY_PATH covers both.
    cp /out/lib/*.so /oneapi-rt/lib/
EOF

# ---------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE} AS runtime

# Level-Zero loader + Intel Graphics Compiler + compute-runtime (the GPU
# driver `sycl-ls` needs to enumerate a `level_zero:gpu` device), installed
# from GitHub releases rather than Intel's own apt repo, and version-pinned as
# a matched set rather than individually latest:
#
# - Battlemage device IDs 0xE220-0xE223 (Arc B-series, including the Arc Pro
#   B70 this was verified against — see docker/sycl/README.md's "Status")
#   were only recognized starting in compute-runtime 25.27; Intel's apt repo
#   (repositories.intel.com/gpu/ubuntu noble/unified, checked at the time
#   this was written) tops out at 25.18, so it silently produces a driver
#   that enumerates zero GPU devices on this hardware rather than erroring.
# - Ubuntu 24.04's stock `libze1` (1.16.1) is old enough that a newer driver
#   segfaults inside it during device enumeration — the loader and driver
#   share an ABI and must move together, not just "driver newest wins".
#
# Bump these four ARGs together (matched release), not independently, and
# re-verify on real hardware after any bump.
ARG COMPUTE_RUNTIME_VERSION=26.35.39758.10
ARG IGC_VERSION=2.41.5
ARG IGC_BUILD=22716
ARG LEVEL_ZERO_VERSION=1.33.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget libstdc++6 ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    for pkg in \
        "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/libze1_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb" \
        "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
        "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
        "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_22.10.0_amd64.deb" \
        "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
        "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
        ; do wget -q "$pkg"; done && \
    dpkg -i ./*.deb && \
    rm -rf /tmp/*.deb

COPY --from=builder /oneapi-rt/lib /opt/intel/oneapi-rt/lib
ENV LD_LIBRARY_PATH=/opt/intel/oneapi-rt/lib

COPY --from=builder /out/crane-serve /usr/local/bin/crane-serve
RUN mkdir -p /models

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/crane-serve"]
CMD ["--help"]
