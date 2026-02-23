#!/usr/bin/env bash
# Example script to demonstrate PaddleOCR-VL-1.5 multimodal inference using crane-oai

HOST="http://127.0.0.1:8000"
IMAGE_URL="https://i0.hdslb.com/bfs/new_dyn/1824ac967aca31d7ac9da4fdda678c4639471072.png"

echo "=================================================="
echo "Testing /v1/chat/completions (OpenAI Compatible API)"
echo "=================================================="
curl -s -X POST $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "paddleocr_vl-1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "'"$IMAGE_URL"'"}},
          {"type": "text", "text": "OCR:"}
        ]
      }
    ],
    "max_tokens": 1024
  }' | jq '{id, choices: .choices[0].message.content}'
echo ""
echo ""

echo "=================================================="
echo "Testing /generate (SGLang Compatible API)"
echo "=================================================="
curl -s -X POST $HOST/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "OCR:",
    "image_url": "'"$IMAGE_URL"'",
    "sampling_params": {
      "max_new_tokens": 1024
    }
  }' | jq '{text: .text, meta: .meta_info}'
echo ""
echo ""

echo "=================================================="
echo "Testing /v1/chat/completions with Streaming"
echo "=================================================="
# We don't pipe through jq since it's an SSE stream
curl -N -s -X POST $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "paddleocr_vl-1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "'"$IMAGE_URL"'"}},
          {"type": "text", "text": "OCR:"}
        ]
      }
    ],
    "max_tokens": 1024,
    "stream": true
  }'
echo ""
echo "=================================================="
