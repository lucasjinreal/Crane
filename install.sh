#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="$ROOT_DIR/target/release/crane"
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
  say "${YELLOW}${BOLD}Detected platform:${NC} macOS"
  say "${YELLOW}Note:${NC} Metal backend will be used automatically at runtime when available."
elif [[ "$(uname -s)" == "Linux" ]]; then
  if have_cmd nvidia-smi; then
    PLATFORM="linux-cuda"
    BUILD_FEATURES+=("cuda")
    say "${YELLOW}${BOLD}Detected platform:${NC} Linux with NVIDIA CUDA"
    say "${YELLOW}Enabled Cargo features:${NC} cuda"
  else
    PLATFORM="linux"
    say "${YELLOW}${BOLD}Detected platform:${NC} Linux (CPU build)"
  fi
else
  say "${RED}Unsupported platform: $(uname -s)${NC}"
  exit 1
fi

BUILD_CMD=(cargo build --release -p crane)
if [[ ${#BUILD_FEATURES[@]} -gt 0 ]]; then
  FEATURES_CSV=$(IFS=,; printf '%s' "${BUILD_FEATURES[*]}")
  BUILD_CMD+=(--features "$FEATURES_CSV")
fi

say "${BLUE}${BOLD}Building crane...${NC}"
(
  cd "$ROOT_DIR"
  "${BUILD_CMD[@]}"
)

say "${GREEN}${BOLD}Build complete.${NC}"
say "${GREEN}Binary:${NC} $BIN_PATH"
say ""
say "${BLUE}${BOLD}Serve a model:${NC}"
say "  $BIN_PATH -m /path/to/model -p 8080"
say ""
say "${BLUE}${BOLD}OpenAI API:${NC}"
say "  GET  http://127.0.0.1:8080/health"
say "  POST http://127.0.0.1:8080/v1/chat/completions"
say ""
say "${BLUE}${BOLD}Example:${NC}"
say "  curl http://127.0.0.1:8080/v1/chat/completions \\\n    -H 'Content-Type: application/json' \\\n    -d '{\n      \"model\": \"your-model\",\n      \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],\n      \"stream\": false\n    }'"
