#!/usr/bin/env bash
# Build and run Crane on the Intel SYCL backend inside the oneAPI dev
# container (docker/sycl/Dockerfile.dev).
#
#   docker/sycl/run.sh build
#   docker/sycl/run.sh chat  -m /models/<dir>
#   docker/sycl/run.sh test          # cargo test --test sycl_kernels
#   docker/sycl/run.sh <any bash command run inside the container>
#
# Env overrides:
#   CRANE_SYCL_MODELS  host dir of models, mounted read-only at /models
#   CRANE_SYCL_TARGET  host dir for the container's CARGO_TARGET_DIR
#   CRANE_CANDLE       host path of a Hahihula/candle checkout to mount at
#                      /candle (only needed if the root Cargo.toml [patch] uses
#                      a `path = "../candle/..."` dep instead of the git branch)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=crane-sycl:dev
MODELS="${CRANE_SYCL_MODELS:-}"
# Kept outside the repo: the container writes it as root.
TARGET="${CRANE_SYCL_TARGET:-$REPO/../crane-sycl-docker-target}"
CANDLE="${CRANE_CANDLE:-}"
mkdir -p "$TARGET"

docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  docker build -t "$IMAGE" -f "$REPO/docker/sycl/Dockerfile.dev" "$REPO/docker/sycl"

BUILD='cargo build --release -p crane-examples --bin chat_cli --features sycl'
# crane_core::utils::sycl_env::ensure_sycl_runtime_env() puts the oneAPI
# runtime on LD_LIBRARY_PATH itself (re-exec'ing once), so RUN_PRE only needs
# to cover the out-of-tree kernel libraries (libcandle_sycl.so /
# libcrane_gdn_sycl.so — their rpath does not reach the final binary).
RUN_PRE='export LD_LIBRARY_PATH="$(find /target -name "libcandle_sycl.so" -o -name "libcrane_gdn_sycl.so" | xargs -rn1 dirname | sort -u | paste -sd:):${LD_LIBRARY_PATH}"'
case "${1:-build}" in
  build) CMD="$BUILD" ;;
  chat)  shift; CMD="$BUILD && $RUN_PRE && /target/release/chat_cli $*" ;;
  test)  shift; CMD="$RUN_PRE && cargo test -p crane-core --release --features sycl --test sycl_kernels -- --nocapture $*" ;;
  *)     CMD="$*" ;;
esac

ARGS=(
  --rm -i
  --device /dev/dri:/dev/dri
  -v "$REPO":/src
  -v "$TARGET":/target
  -v "$HOME/.cargo/registry":/opt/cargo/registry
  -v "$HOME/.cargo/git":/opt/cargo/git
  -e CARGO_TARGET_DIR=/target
  -w /src
)
[ -n "$MODELS" ] && ARGS+=( -v "$MODELS":/models:ro )
[ -n "$CANDLE" ] && ARGS+=( -v "$CANDLE":/candle:ro )

exec docker run "${ARGS[@]}" "$IMAGE" bash -lc "$CMD"
