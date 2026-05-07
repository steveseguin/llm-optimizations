#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://localmaxxing.com/api/benchmarks"


def post_payload(key: str, payload: dict) -> tuple[int, str, int | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, response.read().decode("utf-8"), None
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        retry_after_ms = None
        try:
            parsed = json.loads(text)
            retry_after_ms = parsed.get("retryAfterMs")
        except Exception:
            pass
        return exc.code, text, retry_after_ms


def print_success_response(text: str) -> None:
    try:
        parsed = json.loads(text)
        print(json.dumps({"id": parsed.get("id"), "status": parsed.get("status")}))
    except Exception:
        print(text[:500])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payloads",
        default="/home/steve/localmaxxing_payloads.json",
        help="JSON payload queue",
    )
    parser.add_argument("--label", action="append", help="Submit only this label")
    parser.add_argument("--limit", type=int, help="Submit at most N payloads")
    parser.add_argument("--sleep-on-429", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue = json.loads(Path(args.payloads).read_text())
    if args.label:
        labels = set(args.label)
        queue = [item for item in queue if item["label"] in labels]
    if args.limit is not None:
        queue = queue[: args.limit]

    if args.dry_run:
        print(json.dumps(queue, indent=2))
        return 0

    key = os.environ.get("LMX_API_KEY")
    if not key:
        print("LMX_API_KEY is required", file=sys.stderr)
        return 2

    for index, item in enumerate(queue, start=1):
        label = item["label"]
        status, text, retry_after_ms = post_payload(key, item["payload"])
        print(f"{index}/{len(queue)} {label}: HTTP {status}")
        if 200 <= status < 300:
            print_success_response(text)
            continue

        print(text[:1000], file=sys.stderr)
        if status == 429 and args.sleep_on_429 and retry_after_ms:
            sleep_s = max(1, int(retry_after_ms / 1000) + 2)
            print(f"rate limited; sleeping {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)
            status, text, _ = post_payload(key, item["payload"])
            print(f"{index}/{len(queue)} {label} retry: HTTP {status}")
            if not (200 <= status < 300):
                print(text[:1000], file=sys.stderr)
                return 1
            print_success_response(text)
        else:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
