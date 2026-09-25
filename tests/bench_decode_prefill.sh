#!/bin/bash
# Synthetic decode/prefill throughput probes against an already-running
# crane-serve instance. Unlike a real coding-agent session, both requests
# use a fixed, hardcoded payload with no tools offered, so prompt size and
# token count are identical on every run — needed to compare timing/profiler
# output across runs (or against another server) without agentic tool-call
# branching changing the input on every attempt.
#
# This script never starts or stops crane-serve itself — point it at a
# server you already have running (e.g. via `podman compose up`).
#
# Usage:
#   ./tests/bench_decode_prefill.sh [decode|prefill|default|sweep] [host:port]
#
# Requires: curl, jq, bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

MODE="${1:-default}"
HOST="${2:-localhost:8080}"
URL="http://$HOST/v1/chat/completions"

# Files concatenated into the prefill benchmark's fixed prompt. Sized to land
# in the same few-thousand-to-low-five-figure token range real coding-agent
# requests hit in practice, since a coding agent's system prompt commonly
# includes the repo's own agent-instructions file plus whatever source it
# read. Adjust this list to change the prefill benchmark's prompt size.
PREFILL_SOURCES=(
    "AGENTS.md"
    "crane-core/src/device.rs"
)

# Target prompt sizes (in repeated filler words, not exact tokens) for the
# `sweep` mode. Chosen to straddle CRANE_MOE_OFFLOAD_MIN_BATCH's default of
# 32 tokens so a paired run with the threshold forced above vs. below each
# size shows the real crossover instead of guessing from a single data
# point.
SWEEP_WORD_COUNTS=(8 16 32 64 128)

for dep in curl jq bc; do
    command -v "$dep" >/dev/null 2>&1 || {
        echo "Missing dependency: $dep" >&2
        exit 1
    }
done

run_decode_bench() {
    echo "=== Decode benchmark ==="
    echo "Fixed short prompt, forced long generation, no tools, greedy decoding."

    local payload response t0 t1 completion_tokens elapsed
    payload=$(jq -n '{
        model: "default",
        messages: [{
            role: "user",
            content: "Count from 1 to 1000. Output only the numbers, one per line, no other text."
        }],
        max_tokens: 300,
        temperature: 0,
        top_p: 1,
        stream: false
    }')

    t0=$(date +%s.%N)
    response=$(curl -s "$URL" -H "Content-Type: application/json" -d "$payload")
    t1=$(date +%s.%N)

    completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // empty')
    elapsed=$(echo "$t1 - $t0" | bc)
    echo "Wall time: ${elapsed}s"
    if [ -n "$completion_tokens" ]; then
        echo "Completion tokens: $completion_tokens"
        echo "Client-side tok/s: $(echo "scale=2; $completion_tokens / $elapsed" | bc)"
    else
        echo "Response (no usage field found):"
        echo "$response"
    fi
    echo "Cross-check against the server log's own decode_tok_s for this request,"
    echo "and any [crane-prof] decode spans if CRANE_PROF=1 is set server-side."
}

run_prefill_bench() {
    echo "=== Prefill benchmark ==="
    echo "Fixed large prompt (pinned repo files), minimal generation."

    local file payload response t0 t1 elapsed
    for file in "${PREFILL_SOURCES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "Prefill source file not found: $file" >&2
            exit 1
        fi
    done

    payload=$(
        {
            echo -n '{"parts": ['
            local first=1
            for file in "${PREFILL_SOURCES[@]}"; do
                [ "$first" -eq 1 ] || echo -n ","
                first=0
                jq -Rs '.' <"$file"
            done
            echo -n ']}'
        } | jq '{
            model: "default",
            messages: [{
                role: "user",
                content: ("Summarize the following in one word.\n\n" + (.parts | join("\n\n")))
            }],
            max_tokens: 4,
            temperature: 0,
            stream: false
        }'
    )

    t0=$(date +%s.%N)
    response=$(curl -s "$URL" -H "Content-Type: application/json" -d "$payload")
    t1=$(date +%s.%N)

    elapsed=$(echo "$t1 - $t0" | bc)
    echo "Wall time: ${elapsed}s"
    echo "Response: $response"
    echo "Cross-check against the server log's 'Prefill complete' / prefill_tok_s"
    echo "line for this request."
}

run_sweep_bench() {
    echo "=== MoE offload threshold sweep ==="
    echo "Fires one small prefill request per target size below. Start the"
    echo "server once with CRANE_MOE_OFFLOAD_MIN_BATCH forced below every"
    echo "size and once forced above every size (CRANE_PROF=1 CRANE_PROF_EVERY=1),"
    echo "then compare tokens=/prefill_tok_s for matching sizes across the"
    echo "two server logs."

    local count payload response t0 t1 elapsed
    for count in "${SWEEP_WORD_COUNTS[@]}"; do
        payload=$(jq -n --argjson n "$count" '
            {
                model: "default",
                messages: [{
                    role: "user",
                    content: (
                        "Summarize the following in one word.\n\n"
                        + ([range(0; $n)] | map("word") | join(" "))
                    )
                }],
                max_tokens: 4,
                temperature: 0,
                stream: false
            }')

        t0=$(date +%s.%N)
        response=$(curl -s "$URL" -H "Content-Type: application/json" -d "$payload")
        t1=$(date +%s.%N)
        elapsed=$(echo "$t1 - $t0" | bc)

        echo "--- target words: $count ---"
        echo "Wall time: ${elapsed}s"
        echo "Response: $response"
    done
}

case "$MODE" in
decode) run_decode_bench ;;
prefill) run_prefill_bench ;;
default)
    run_decode_bench
    echo
    run_prefill_bench
    ;;
sweep) run_sweep_bench ;;
*)
    echo "Usage: $0 [decode|prefill|default|sweep] [host:port]" >&2
    exit 1
    ;;
esac
