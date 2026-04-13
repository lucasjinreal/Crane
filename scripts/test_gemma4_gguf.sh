#!/bin/bash
# Test script for Gemma4 GGUF model — starts server, makes a request, captures output.
#
# Usage:
#   ./scripts/test_gemma4_gguf.sh <gguf_file_path> [prompt] [max_tokens]

# cd to project root (scripts may be run from any directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

GGUF_PATH="${1:?Usage: $0 <gguf_file_path> [prompt] [max_tokens]}"
PROMPT="${2:-The capital of France is}"
MAX_TOKENS="${3:-20}"
PORT=8080
LOG_FILE="/tmp/crane-gemma4-gguf-debug.log"

# Kill any existing server
pkill -f "crane-oai" 2>/dev/null
sleep 1

echo "=== Gemma4 GGUF Test ==="
echo "Model: $GGUF_PATH"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Check file exists
if [ ! -f "$GGUF_PATH" ]; then
    echo "Error: GGUF file not found: $GGUF_PATH"
    exit 1
fi

# Check tokenizer.json exists nearby
GGUF_DIR=$(dirname "$GGUF_PATH")
if [ ! -f "$GGUF_DIR/tokenizer.json" ]; then
    echo "Warning: tokenizer.json not found in $GGUF_DIR"
    echo "Place tokenizer.json in the same directory as the GGUF file."
    echo ""
fi

# Build
echo "Building..."
if ! cargo build -p crane-oai --release 2>&1 | tail -3; then
    echo "Build may have warnings, continuing..."
fi

# Start server
echo "Starting server..."
target/release/crane-oai \
    --model-path "$GGUF_PATH" \
    --model-type gemma4 \
    --format gguf \
    --port $PORT \
    2>"$LOG_FILE" &
SERVER_PID=$!

# Wait for server to be ready
echo -n "Waiting for server"
for i in $(seq 1 60); do
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
    cat "$LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test completion
echo ""
echo "--- Completion: prompt=\"$PROMPT\" max_tokens=$MAX_TOKENS ---"
RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"gemma4\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS}")
echo "Response: $RESPONSE"

# Wait for async output
sleep 1

# Show any errors from log
echo ""
ERRORS=$(grep -i "error\|panic\|cannot find" "$LOG_FILE" 2>/dev/null)
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
