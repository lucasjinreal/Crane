#!/bin/bash
# Test script for Gemma4 VLM (vision-language) — starts server, sends image+text, captures output.
#
# Usage:
#   ./scripts/test_gemma4_vlm.sh [model_path] [image_url] [prompt] [max_tokens]
#
# Examples:
#   ./scripts/test_gemma4_vlm.sh /path/to/gemma-4-E2B-it/
#   ./scripts/test_gemma4_vlm.sh /path/to/gemma-4-E2B-it/ "https://example.com/image.jpg" "What is in this image?"
#
# NOTE: Use the instruct model (-it), not the base model. The base model
# doesn't understand the chat template and will produce garbage output.

# cd to project root (scripts may be run from any directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

MODEL_PATH="${1:?Usage: $0 <model_path> [image_url] [prompt] [max_tokens]}"
IMAGE_URL="${2:-https://placecats.com/300/200}"
PROMPT="${3:-Describe this image in detail.}"
MAX_TOKENS="${4:-50}"
PORT=8080
LOG_FILE="/tmp/crane-gemma4-vlm-debug.log"

# Kill any existing server
pkill -f "crane-oai" 2>/dev/null
sleep 1

echo "=== Gemma4 VLM Test ==="
echo "Model: $MODEL_PATH"
echo "Image: $IMAGE_URL"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Build
echo "Building..."
if ! cargo build -p crane-oai --release 2>&1 | tail -3; then
    echo "Build may have warnings, continuing..."
fi

# Start server (auto-detect should pick gemma4_vl)
echo "Starting server..."
target/release/crane-oai \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    2>"$LOG_FILE" &
SERVER_PID=$!

# Wait for server to be ready
echo -n "Waiting for server"
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo " ready!"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo " FAILED (timeout)"
    echo "--- Server log ---"
    tail -20 "$LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test: send image + text via chat completions
echo ""
echo "--- VLM Request ---"
RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"gemma4-vl\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"$IMAGE_URL\"}},
                {\"type\": \"text\", \"text\": \"$PROMPT\"}
            ]
        }],
        \"max_tokens\": $MAX_TOKENS
    }")
echo "Response: $RESPONSE"

# Wait for async output
sleep 1

# Show errors from log
echo ""
ERRORS=$(grep -i "error\|panic\|Failed" "$LOG_FILE" 2>/dev/null | tail -10)
if [ -n "$ERRORS" ]; then
    echo "--- Errors from log ---"
    echo "$ERRORS"
fi

# Cleanup
echo ""
echo "--- Stopping server (PID $SERVER_PID) ---"
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
