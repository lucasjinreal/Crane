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
#   ./tests/bench_llm.sh [decode|prefill|prefill+decode|decode-batched|decode-longctx|sweep] [host:port]
#
# Requires: curl, jq, bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

MODE="${1:-prefill+decode}"
HOST="${2:-localhost:8080}"
URL="http://$HOST/v1/chat/completions"

# Files concatenated into the prefill benchmark's fixed prompt. Sized to land
# in the same few-thousand-to-low-five-figure token range real coding-agent
# requests hit in practice, since a coding agent's system prompt commonly
# includes a repo-level doc plus whatever source it read. Must be files
# tracked in the repo, so the prompt is reproducible across checkouts.
# Adjust this list to change the prefill benchmark's prompt size.
PREFILL_SOURCES=(
    "README.md"
    "crane-core/src/device.rs"
)

# Target prompt sizes (in repeated filler words, not exact tokens) for the
# `sweep` mode. The >=32 sizes straddle CRANE_MOE_OFFLOAD_MIN_BATCH's default
# of 32 tokens so a paired run with the threshold forced above vs. below each
# size shows the real crossover instead of guessing from a single data point.
# The <32 sizes stay on the CPU-batched dispatch path (cpu_batched_forward /
# dispatch_moe_quads) at every default threshold, and are fine-grained so a
# paired run across two server builds (same threshold, same everything else)
# isolates a CPU dispatch-order change's effect from request-size noise.
SWEEP_WORD_COUNTS=(4 8 12 16 20 24 28 32 64 128)

# Concurrency levels for `decode-batched`. Spans below, at, and above
# crane-serve's default --max-concurrent (16) so a batched-decode kernel
# path (vec ncols=2 / MMA's gqa_opt_applies) and parallel-blocks' SM
# occupancy search -- both unreachable from a single in-flight request --
# get exercised at more than one scheduler load point.
BATCH_CONCURRENCY_LEVELS=(1 4 8 16)

# How many times to repeat PREFILL_SOURCES' concatenated contents for
# `decode-longctx`, to push the KV length well past a single pass of those
# files and into the range (tens of thousands of tokens) where
# parallel-blocks has enough KV tiles to split, and where a mandatory mask
# materialization on the decode path has a real cost.
LONGCTX_REPEATS=40

for dep in curl jq bc; do
    command -v "$dep" >/dev/null 2>&1 || {
        echo "Missing dependency: $dep" >&2
        exit 1
    }
done

run_decode_bench() {
    echo "=== Decode benchmark ==="
    echo "Fixed short prompt, forced long generation, no tools, greedy decoding."

    local payload response t0 t1 completion_tokens finish_reason elapsed
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
    finish_reason=$(echo "$response" | jq -r '.choices[0].finish_reason // empty')
    elapsed=$(echo "$t1 - $t0" | bc)
    echo "Wall time: ${elapsed}s"
    if [ -n "$completion_tokens" ]; then
        echo "Completion tokens: $completion_tokens"
        echo "Client-side tok/s: $(echo "scale=2; $completion_tokens / $elapsed" | bc)"
        if [ "$finish_reason" != "length" ]; then
            echo "Warning: finish_reason=$finish_reason (expected 'length') -- the model" >&2
            echo "stopped before max_tokens, so completion_tokens isn't fixed across runs" >&2
            echo "and tok/s isn't comparable to other runs." >&2
        fi
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

    local file payload response t0 t1 elapsed prompt_tokens
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

    prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // empty')
    elapsed=$(echo "$t1 - $t0" | bc)
    echo "Wall time: ${elapsed}s"
    if [ -n "$prompt_tokens" ]; then
        echo "Prompt tokens: $prompt_tokens"
        echo "Client-side tok/s (wall time, includes the 4-token decode): $(echo "scale=2; $prompt_tokens / $elapsed" | bc)"
    else
        echo "Response (no usage field found):"
        echo "$response"
    fi
    echo "Cross-check against the server log's 'Prefill complete' / prefill_tok_s"
    echo "line for this request -- it excludes the decode step and client-side"
    echo "request overhead that the wall-time figure above includes."
}

run_sweep_bench() {
    echo "=== MoE offload threshold sweep ==="
    echo "Fires one small prefill request per target size below."
    echo "Offload-threshold crossover: start the server once with"
    echo "CRANE_MOE_OFFLOAD_MIN_BATCH forced below every size and once forced"
    echo "above every size (CRANE_PROF=1 CRANE_PROF_EVERY=1), then compare"
    echo "tokens=/prefill_tok_s for matching sizes across the two server logs."
    echo "CPU dispatch-order comparison (e.g. before/after a"
    echo "dispatch_moe_quads change): leave CRANE_MOE_OFFLOAD_MIN_BATCH at"
    echo "its default (32) on both builds, run this sweep against each build"
    echo "in turn with CRANE_PROF=1 CRANE_PROF_EVERY=1, then diff the 'moe:'"
    echo "line's expert/moe-sum values for the <32-word sizes across the two"
    echo "server logs -- those sizes always take the CPU-batched path"
    echo "regardless of build."

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

run_decode_batched_bench() {
    echo "=== Batched decode benchmark ==="
    echo "Same fixed short prompt and forced long generation as the single-request"
    echo "decode benchmark, fired concurrently at each of: ${BATCH_CONCURRENCY_LEVELS[*]}"
    echo "requests. A single in-flight request always takes the seq_q==1/batch==1"
    echo "decode path; this sweep is what exercises the scheduler's batched-decode"
    echo "path and any SM-occupancy-driven kernel selection instead."

    local payload
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

    local concurrency responses_dir i pid pids t0 t1 elapsed
    local total_tokens all_finished_length completion_tokens finish_reason
    for concurrency in "${BATCH_CONCURRENCY_LEVELS[@]}"; do
        echo "--- concurrency: $concurrency ---"
        responses_dir=$(mktemp -d)
        pids=()

        t0=$(date +%s.%N)
        for ((i = 0; i < concurrency; i++)); do
            curl -s "$URL" -H "Content-Type: application/json" -d "$payload" \
                >"$responses_dir/$i.json" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        t1=$(date +%s.%N)
        elapsed=$(echo "$t1 - $t0" | bc)

        total_tokens=0
        all_finished_length=1
        for ((i = 0; i < concurrency; i++)); do
            completion_tokens=$(jq -r '.usage.completion_tokens // 0' "$responses_dir/$i.json")
            finish_reason=$(jq -r '.choices[0].finish_reason // empty' "$responses_dir/$i.json")
            total_tokens=$((total_tokens + completion_tokens))
            [ "$finish_reason" = "length" ] || all_finished_length=0
        done
        rm -rf "$responses_dir"

        echo "Wall time: ${elapsed}s"
        echo "Total completion tokens across $concurrency requests: $total_tokens"
        echo "Aggregate client-side tok/s: $(echo "scale=2; $total_tokens / $elapsed" | bc)"
        if [ "$all_finished_length" -ne 1 ]; then
            echo "Warning: at least one request's finish_reason != 'length' -- token" >&2
            echo "counts aren't fixed across runs at this concurrency." >&2
        fi
    done
    echo "Cross-check against the server's per-request decode_tok_s log lines (and"
    echo "[crane-prof] decode spans with CRANE_PROF=1) for per-sequence throughput"
    echo "inside the batch -- the aggregate figures above hide per-sequence variance."
}

run_decode_longctx_bench() {
    echo "=== Long-context decode benchmark ==="
    echo "Prefills a long fixed context (pinned repo files, repeated"
    echo "$LONGCTX_REPEATS times) then forces a long generation, to measure decode"
    echo "throughput at a KV length with enough tiles for parallel-blocks to split,"
    echo "and to surface the cost of materializing a mask on the decode path."

    local file payload response t0 t1 elapsed context_text rep
    local prompt_tokens completion_tokens finish_reason
    for file in "${PREFILL_SOURCES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "Prefill source file not found: $file" >&2
            exit 1
        fi
    done

    context_text=""
    for ((rep = 0; rep < LONGCTX_REPEATS; rep++)); do
        for file in "${PREFILL_SOURCES[@]}"; do
            context_text+=$(cat "$file")
            context_text+=$'\n\n'
        done
    done

    payload=$(jq -n --arg ctx "$context_text" '{
        model: "default",
        messages: [{
            role: "user",
            content: ($ctx + "\n\nCount from 1 to 1000. Output only the numbers, one per line, no other text.")
        }],
        max_tokens: 300,
        temperature: 0,
        top_p: 1,
        stream: false
    }')

    t0=$(date +%s.%N)
    response=$(curl -s "$URL" -H "Content-Type: application/json" -d "$payload")
    t1=$(date +%s.%N)
    elapsed=$(echo "$t1 - $t0" | bc)

    prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // empty')
    completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // empty')
    finish_reason=$(echo "$response" | jq -r '.choices[0].finish_reason // empty')

    echo "Wall time: ${elapsed}s (prefill + decode combined)"
    if [ -n "$prompt_tokens" ]; then
        echo "Prompt tokens: $prompt_tokens"
    fi
    if [ -n "$completion_tokens" ]; then
        echo "Completion tokens: $completion_tokens"
        if [ "$finish_reason" != "length" ]; then
            echo "Warning: finish_reason=$finish_reason (expected 'length') -- the model" >&2
            echo "stopped before max_tokens, so completion_tokens isn't fixed across runs" >&2
            echo "and decode tok/s isn't comparable to other runs." >&2
        fi
    else
        echo "Response (no usage field found):"
        echo "$response"
    fi
    echo "This wall time includes prefill -- read the server log's decode_tok_s for"
    echo "this request, not the client-side figure above, to isolate decode"
    echo "throughput at this KV length from the prefill cost."
}

case "$MODE" in
decode) run_decode_bench ;;
prefill) run_prefill_bench ;;
prefill+decode)
    run_decode_bench
    echo
    run_prefill_bench
    ;;
decode-batched) run_decode_batched_bench ;;
decode-longctx) run_decode_longctx_bench ;;
sweep) run_sweep_bench ;;
*)
    echo "Usage: $0 [decode|prefill|prefill+decode|decode-batched|decode-longctx|sweep] [host:port]" >&2
    exit 1
    ;;
esac
