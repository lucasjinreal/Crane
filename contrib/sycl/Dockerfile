# Crane SYCL (Intel oneAPI) build + run environment — proof-of-concept.
#
# Base image ships the icpx SYCL compiler, oneMKL, and the Level-Zero GPU
# runtime, so an Intel GPU passed in with `--device /dev/dri` is usable out of
# the box (`sycl-ls` lists a `level_zero:gpu` entry).
FROM intel/oneapi-basekit:2025.1.0-0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config git \
        libssl-dev cmake libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust (edition 2024 needs >= 1.85; pin to match the host stable toolchain).
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.93.0 \
    && rustc --version

WORKDIR /src
