#!/bin/bash
# Test script for Gemma4 model — starts server, makes a request, captures debug output.
#
# Usage:
#   ./scripts/test_gemma4.sh <model_path> [prompt] [max_tokens]

MODEL_PATH="${1:?Usage: $0 <model_path> [prompt] [max_tokens]}"
PROMPT="${2:-What is a sun?}"
PORT=8080
MAX_TOKENS="${3:-20}"
LOG_FILE="/tmp/crane-gemma4-debug.log"

# Kill any existing server
pkill -f "crane-oai.*gemma" 2>/dev/null
sleep 1

echo "=== Gemma4 Debug Test ==="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Log: $LOG_FILE"
echo ""

# Build if needed
echo "Building..."
if ! cargo build -p crane-oai --release 2>&1 | tail -5; then
    echo "Build failed!"
    exit 1
fi

# Start server, capture stderr (debug output) to log file
echo "Starting server..."
target/release/crane-oai \
    --model-path "$MODEL_PATH" \
    --model-type gemma4 \
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
    kill $SERVER_PID 2>/dev/null
    cat "$LOG_FILE"
    exit 1
fi

# Make test request
echo ""
echo "--- Request: prompt=\"$PROMPT\" max_tokens=$MAX_TOKENS ---"
RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"gemma-4-E2B\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS}")
echo "Response: $RESPONSE"

# Wait a moment for any async debug output
sleep 1

# Show debug output
echo ""
echo "--- Debug Output (from stderr) ---"
grep "\[DEBUG\]" "$LOG_FILE" | tail -20

# Cleanup
echo ""
echo "--- Stopping server (PID $SERVER_PID) ---"
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
