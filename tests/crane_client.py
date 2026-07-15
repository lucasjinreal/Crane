#!/usr/bin/env python3
"""Shared HTTP client for crane-serve's TTS and ASR endpoints."""

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
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        if isinstance(e, urllib.error.HTTPError):
            message = e.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(message)["error"]["message"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            raise RuntimeError(f"speech request failed ({e.code}): {message}") from e
        raise RuntimeError(f"speech request failed: {e.reason}") from e


def transcribe(
    wav_path: str,
    language: str | None = None,
    server_url: str = "http://localhost:8000",
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        if isinstance(e, urllib.error.HTTPError):
            message = e.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(message)["error"]["message"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            raise RuntimeError(f"transcribe request failed ({e.code}): {message}") from e
        raise RuntimeError(f"transcribe request failed: {e.reason}") from e

    try:
        response = json.loads(response_body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"transcribe response was not valid JSON: {response_body}") from e

    if "text" not in response:
        raise RuntimeError(f"transcribe response missing 'text' field: {response}")
    return response["text"]


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
            f"{value}\r\n".encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _cmd_speech(args):
    print(f"POST {args.url}/v1/audio/speech", file=sys.stderr)
    audio = speech(
        args.text,
        voice=args.voice,
        language=args.language,
        server_url=args.url,
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
  crane_client.py speech "hello world" -u http://localhost:9000 | pw-play -"""

_TRANSCRIBE_EXAMPLES = """examples:
  crane_client.py transcribe speech.wav
  crane_client.py transcribe speech.wav --language en
  crane_client.py transcribe speech.wav -u http://localhost:9000"""


def _cmd_transcribe(args):
    print(f"POST {args.url}/v1/audio/transcriptions", file=sys.stderr)
    text = transcribe(
        args.wav_path,
        language=args.language,
        server_url=args.url,
    )
    print(text)


def main():
    url_parser = argparse.ArgumentParser(add_help=False)
    url_parser.add_argument(
        "--url", "-u", default="http://localhost:8000", help="crane-serve base URL"
    )

    parser = argparse.ArgumentParser(
        description="Crane HTTP client",
        epilog=f"{_SPEECH_EXAMPLES}\n\n{_TRANSCRIBE_EXAMPLES}",
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
    transcribe_parser.set_defaults(func=_cmd_transcribe)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
