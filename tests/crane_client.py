#!/usr/bin/env python3
"""Shared HTTP client for crane-serve's TTS, ASR, and chat completion endpoints."""

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid


def speech(
    text: str,
    voice: str | None = None,
    language: str | None = None,
    server_url: str = "http://localhost:8000",
    timeout: float = 180,
    **opts,
) -> bytes:
    """POST to /v1/audio/speech and return the raw audio response bytes."""
    payload = {"model": "default", "input": text}
    if voice is not None:
        payload["voice"] = voice
    if language is not None:
        payload["language"] = language
    payload.update(opts)  # allows callers to pass extra API fields

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        message = e.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(message)["error"]["message"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        raise RuntimeError(f"speech request failed ({e.code}): {message}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"speech request failed: {e.reason}") from e


def chat(
    message: str,
    system: str | None = None,
    model: str = "default",
    server_url: str = "http://localhost:8000",
    max_tokens: int = 512,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    timeout: float = 180,
    **opts,
) -> str:
    """POST to /v1/chat/completions and return the assistant's reply text."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    payload.update(opts)  # allows callers to pass extra API fields

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Long prompts and long generations can take minutes on CPU/small GPUs.
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_message = e.read().decode("utf-8", errors="replace")
        try:
            err_message = json.loads(err_message)["error"]["message"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        raise RuntimeError(f"chat request failed ({e.code}): {err_message}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"chat request failed: {e.reason}") from e

    try:
        response = json.loads(response_body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"chat response was not valid JSON: {response_body}") from e

    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"chat response missing content: {response}") from e


def transcribe(
    wav_path: str,
    language: str | None = None,
    server_url: str = "http://localhost:8000",
    timeout: float = 180,
    **opts,
) -> str:
    """POST a WAV file to /v1/audio/transcriptions and return the transcript text."""
    fields = {"model": "default"}
    if language is not None:
        fields["language"] = language
    for key, value in opts.items():  # allows callers to pass extra API fields
        fields[key] = value if isinstance(value, str) else json.dumps(value)

    boundary = uuid.uuid4().hex
    content_type, body = _encode_multipart(boundary, fields, wav_path)

    req = urllib.request.Request(
        f"{server_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        message = e.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(message)["error"]["message"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        raise RuntimeError(f"transcribe request failed ({e.code}): {message}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"transcribe request failed: {e.reason}") from e

    try:
        response = json.loads(response_body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"transcribe response was not valid JSON: {response_body}") from e

    if "text" not in response:
        raise RuntimeError(f"transcribe response missing 'text' field: {response}")
    return response["text"]


def _positive_float(value: str) -> float:
    """Parse a CLI argument as a strictly positive float, for use as an argparse type."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value!r}")
    return parsed


def _escape_header_value(value: str) -> str:
    """Escape a value for safe use inside a quoted multipart header parameter."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def _encode_multipart(boundary: str, fields: dict[str, str], file_path: str) -> tuple[str, bytes]:
    """Encode form fields and a file as multipart/form-data."""
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    filename = _escape_header_value(os.path.basename(file_path))
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except OSError as e:
        raise RuntimeError(f"could not read {file_path}: {e}") from e

    parts = []
    for name, value in fields.items():
        name = _escape_header_value(name)
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())

    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _cmd_speech(args):
    print(f"POST {args.url}/v1/audio/speech", file=sys.stderr)
    audio = speech(
        args.text,
        voice=args.voice,
        language=args.language,
        server_url=args.url,
        timeout=args.timeout,
    )
    if args.output:
        with open(args.output, "wb") as f:
            f.write(audio)
        print(f"Wrote {len(audio)} bytes to {args.output}", file=sys.stderr)
    else:
        sys.stdout.buffer.write(audio)


_SPEECH_EXAMPLES = """examples:
  crane_client.py speech "hello world" -o out.wav
  crane_client.py speech "hello world" | pw-play -
  crane_client.py speech "hello world" --voice ash --language en | pw-play -
  crane_client.py speech "hello world" -u http://localhost:9000 | pw-play -
  crane_client.py speech "hello world" -t 120 -o out.wav"""

_TRANSCRIBE_EXAMPLES = """examples:
  crane_client.py transcribe speech.wav
  crane_client.py transcribe speech.wav --language en
  crane_client.py transcribe speech.wav -u http://localhost:9000
  crane_client.py transcribe speech.wav -t 120"""

_CHAT_EXAMPLES = """examples:
  crane_client.py chat "what model is suggested for Qwen3 on a 16GB VRAM GPU?"
  crane_client.py chat "summarize this" --file long_prompt.txt
  crane_client.py chat "hi" --system "You are terse." --max-tokens 64
  crane_client.py chat "hi" -u http://localhost:9000
  crane_client.py chat "hi" -t 600"""


def _cmd_chat(args):
    message = args.message
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as f:
                # Filler text before the actual question, to construct long
                # prompts that force chunked prefill.
                message = f"{f.read()}\n\n{message}"
        except OSError as e:
            raise RuntimeError(f"could not read {args.file}: {e}") from e
    print(f"POST {args.url}/v1/chat/completions", file=sys.stderr)
    text = chat(
        message,
        system=args.system,
        model=args.model,
        server_url=args.url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        timeout=args.timeout,
    )
    print(text)


def _cmd_transcribe(args):
    print(f"POST {args.url}/v1/audio/transcriptions", file=sys.stderr)
    text = transcribe(
        args.wav_path,
        language=args.language,
        server_url=args.url,
        timeout=args.timeout,
    )
    print(text)


def main():
    url_parser = argparse.ArgumentParser(add_help=False)
    url_parser.add_argument(
        "--url", "-u", default="http://localhost:8000", help="crane-serve base URL"
    )

    parser = argparse.ArgumentParser(
        description="Crane HTTP client",
        epilog=f"{_SPEECH_EXAMPLES}\n\n{_TRANSCRIBE_EXAMPLES}\n\n{_CHAT_EXAMPLES}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    speech_parser = subparsers.add_parser(
        "speech",
        help="synthesize speech from text",
        epilog=_SPEECH_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[url_parser],
    )
    speech_parser.add_argument("text", help="text to synthesize")
    speech_parser.add_argument("--voice", default=None, help="voice/speaker preset")
    speech_parser.add_argument("--language", default=None, help="language hint")
    speech_parser.add_argument(
        "--output", "-o", default=None, help="write audio to this file instead of stdout"
    )
    speech_parser.add_argument(
        "--timeout",
        "-t",
        type=_positive_float,
        default=180,
        help="request timeout in seconds (default: 180; large models can take longer)",
    )
    speech_parser.set_defaults(func=_cmd_speech)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="transcribe a WAV file to text",
        epilog=_TRANSCRIBE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[url_parser],
    )
    transcribe_parser.add_argument("wav_path", help="path to the WAV file to transcribe")
    transcribe_parser.add_argument("--language", default=None, help="language hint")
    transcribe_parser.add_argument(
        "--timeout",
        "-t",
        type=_positive_float,
        default=180,
        help="request timeout in seconds (default: 180)",
    )
    transcribe_parser.set_defaults(func=_cmd_transcribe)

    chat_parser = subparsers.add_parser(
        "chat",
        help="send a chat completion request",
        epilog=_CHAT_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[url_parser],
    )
    chat_parser.add_argument("message", help="user message (the question to ask)")
    chat_parser.add_argument(
        "--file", default=None, help="prepend this file's contents to the message as context"
    )
    chat_parser.add_argument("--system", default=None, help="system prompt")
    chat_parser.add_argument("--model", default="default", help="model name")
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=None)
    chat_parser.add_argument("--top-p", type=float, default=None)
    chat_parser.add_argument("--top-k", type=int, default=None)
    chat_parser.add_argument("--repetition-penalty", type=float, default=None)
    chat_parser.add_argument(
        "--timeout",
        "-t",
        type=_positive_float,
        default=180,
        help="request timeout in seconds (default: 180; long prompts/generations can take minutes)",
    )
    chat_parser.set_defaults(func=_cmd_chat)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
