#!/bin/sh
# SPDX-License-Identifier: MIT
# Entrypoint for the ROCm crane-serve image (see CONTAINER.md).
#
# The model is not baked into the image -- mount a host model directory to
# /models (see compose.yaml) and set MODEL_PATH to the file/dir inside it
# to load. MODEL_TYPE/FORMAT default to crane-serve's own "auto" detection;
# set them if auto-detection picks the wrong architecture. EXTRA_ARGS is a
# space-separated string of additional flags (e.g. "--dtype f32"), for
# `podman-compose`/`podman compose` setups that have no other way to append
# arguments; it is intentionally unquoted below to word-split. Extra
# arguments passed directly to `podman run` are appended after that, so
# they can add flags (e.g. --max-concurrent 8) or override one by repeating
# it (e.g. --model-path /a/different/model.gguf).
set -eu

: "${MODEL_PATH:?MODEL_PATH must be set to the model file/dir path, e.g. /models/llm/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q5_K_M.gguf}"

# shellcheck disable=SC2086 # EXTRA_ARGS is meant to word-split.
exec crane-serve \
    --model-path "${MODEL_PATH}" \
    --model-type "${MODEL_TYPE:-auto}" \
    --format "${FORMAT:-auto}" \
    ${EXTRA_ARGS:-} \
    "$@"
