#!/usr/bin/env python3
"""Shared HTTP client for crane-serve's TTS and ASR endpoints."""

import argparse
import json
import sys
import urllib.error
import urllib.request


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


def main():
    url_parser = argparse.ArgumentParser(add_help=False)
    url_parser.add_argument(
        "--url", "-u", default="http://localhost:8000", help="crane-serve base URL"
    )

    parser = argparse.ArgumentParser(
        description="Crane HTTP client",
        epilog=_SPEECH_EXAMPLES,
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

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
