#!/bin/sh
# SPDX-License-Identifier: MIT
# Entrypoint for the ROCm crane-serve image (see compose.yaml).
#
# MODEL_TYPE/FORMAT default to crane-serve's own "auto" detection; set them
# on the host to override auto-detection. EXTRA_ARGS is a space-separated
# string of additional flags (e.g. "--dtype f32"); intentionally unquoted
# below to word-split, since compose has no other way to append arbitrary
# arguments. Arguments after the image name (compose's `command:`, or extra
# args to `podman run`/`docker run`) are appended last via "$@", so they can
# add flags or override one of the above by repeating it.
set -eu

# Show set variables set
export | grep -E 'CRANE|AMD|ROCM'

# shellcheck disable=SC2086 # EXTRA_ARGS is meant to word-split.
exec crane-serve \
    --model-type "${MODEL_TYPE:-auto}" \
    --format "${FORMAT:-auto}" \
    ${EXTRA_ARGS:-} \
    "$@"
