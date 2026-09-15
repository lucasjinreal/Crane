#!/usr/bin/env bash
# Build and run Crane on the Intel SYCL backend inside the oneAPI container.
#
#   contrib/sycl/run.sh build
#   contrib/sycl/run.sh chat  -m /models/<dir>
#   contrib/sycl/run.sh test          # cargo test --test sycl_kernels
#   contrib/sycl/run.sh <any bash command run inside the container>
#
# Env overrides:
#   CRANE_SYCL_MODELS  host dir of models, mounted read-only at /models
#   CRANE_SYCL_TARGET  host dir for the container's CARGO_TARGET_DIR
#   CRANE_CANDLE       host path of a Hahihula/candle checkout to mount at
#                      /candle (only needed if the root Cargo.toml [patch] uses
#                      a `path = "../candle/..."` dep instead of the git rev)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=crane-sycl:dev
MODELS="${CRANE_SYCL_MODELS:-}"
# Kept outside the repo: the container writes it as root.
TARGET="${CRANE_SYCL_TARGET:-$REPO/../crane-sycl-docker-target}"
CANDLE="${CRANE_CANDLE:-}"
mkdir -p "$TARGET"

docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  docker build -t "$IMAGE" -f "$REPO/contrib/sycl/Dockerfile" "$REPO/contrib/sycl"

BUILD='cargo build --release -p crane-examples --bin chat_cli --features sycl'
# Prepend both build.rs OUT_DIRs (libcandle_sycl.so / libcrane_gdn_sycl.so —
# their rpath does not reach the final binary) to the image's baked oneAPI
# LD_LIBRARY_PATH. Do NOT clobber it: it carries mkl / umf / pti / level-zero
# adapter dirs that both the split libmkl_sycl_* deps and GPU discovery need.
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
