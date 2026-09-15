#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BIN_PATH="$ROOT_DIR/target/release/crane-serve"
CHAT_SIMPLE_BIN_PATH="$ROOT_DIR/target/release/chat_simple"
CHAT_CLI_BIN_PATH="$ROOT_DIR/target/release/chat_cli"
BUILD_FEATURES=()
PLATFORM="unknown"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

say() {
  printf "%b\n" "$1"
}

if [[ "$(uname -s)" == "Darwin" ]]; then
  PLATFORM="macos"
  BUILD_FEATURES+=("metal")
  BUILD_FEATURES+=("accelerate")
  say "${YELLOW}${BOLD}Detected platform:${NC} macOS"
  say "${YELLOW}Enabled Cargo features:${NC} metal,accelerate"
elif [[ "$(uname -s)" == "Linux" ]]; then
  if have_cmd nvidia-smi; then
    PLATFORM="linux-cuda"
    BUILD_FEATURES+=("cuda")
    say "${YELLOW}${BOLD}Detected platform:${NC} Linux with NVIDIA CUDA"
    say "${YELLOW}Enabled Cargo features:${NC} cuda"
  elif have_cmd icpx || have_cmd sycl-ls; then
    # Intel oneAPI / SYCL (proof-of-concept). Uses the candle fork wired in via
    # [patch.crates-io] in the root Cargo.toml.
    PLATFORM="linux-sycl"
    BUILD_FEATURES+=("sycl")
    say "${YELLOW}${BOLD}Detected platform:${NC} Linux with Intel oneAPI / SYCL"
    say "${YELLOW}Enabled Cargo features:${NC} sycl"
  else
    PLATFORM="linux"
    say "${YELLOW}${BOLD}Detected platform:${NC} Linux (CPU build)"
  fi
else
  say "${RED}Unsupported platform: $(uname -s)${NC}"
  exit 1
fi

BUILD_CMD=(cargo build --release -p crane-serve -p crane-examples --bin crane-serve --bin chat_simple --bin chat_cli)
if [[ ${#BUILD_FEATURES[@]} -gt 0 ]]; then
  FEATURES_CSV=$(IFS=,; printf '%s' "${BUILD_FEATURES[*]}")
  BUILD_CMD+=(--features "$FEATURES_CSV")
fi

say "${BLUE}${BOLD}Building crane-serve, chat_simple, and chat_cli...${NC}"
(
  cd "$ROOT_DIR"
  "${BUILD_CMD[@]}"
)

say "${GREEN}${BOLD}Build complete.${NC}"
say "${GREEN}Server:${NC} $SERVER_BIN_PATH"
say "${GREEN}Example:${NC} $CHAT_SIMPLE_BIN_PATH"
say "${GREEN}CLI:${NC} $CHAT_CLI_BIN_PATH"
say ""
say "${BLUE}${BOLD}Start a server (one model per process):${NC}"
say "  $SERVER_BIN_PATH -m /path/to/model -p 8080"
say ""
say "${BLUE}${BOLD}Model examples (model type is auto-detected when possible):${NC}"
say "  # LLM"
say "  $SERVER_BIN_PATH -m /path/to/Qwen3-0.6B -p 8080"
say "  # VLM (chat and image requests share the OpenAI chat endpoint)"
say "  $SERVER_BIN_PATH -m /path/to/Qwen3.5-VL --model-type qwen3_5_vl -p 8080"
say "  # ASR (Qwen3-ASR-0.6B)"
say "  $SERVER_BIN_PATH -m /path/to/Qwen3-ASR-0.6B --model-type qwen3_asr -p 8080"
say "  # TTS (Qwen3-TTS)"
say "  $SERVER_BIN_PATH -m /path/to/Qwen3-TTS-12Hz-0.6B-Base --model-type qwen3_tts -p 8080"
say ""
say "${BLUE}${BOLD}API endpoints:${NC}"
say "  LLM / VLM: POST http://127.0.0.1:8080/v1/chat/completions"
say "  ASR:       POST http://127.0.0.1:8080/v1/audio/transcriptions"
say "  TTS:       POST http://127.0.0.1:8080/v1/audio/speech"
say "  Health:    GET  http://127.0.0.1:8080/health"
say ""
say "${BLUE}${BOLD}LLM request example:${NC}"
say "  curl http://127.0.0.1:8080/v1/chat/completions \\\n    -H 'Content-Type: application/json' \\\n    -d '{\n      \"model\": \"your-model\",\n      \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],\n      \"stream\": false\n    }'"
