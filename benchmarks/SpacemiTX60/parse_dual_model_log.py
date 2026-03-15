#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

STATS_RE = re.compile(
    r"^\[stats\]\s+"
    r"dronet_hz=(?P<dronet_hz>[0-9.]+)\s+"
    r"mlp_hz=(?P<mlp_hz>[0-9.]+)\s+"
    r"mlp_misses=(?P<mlp_misses>[0-9]+)\s+"
    r"dronet_total=(?P<dronet_total>[0-9]+)\s+"
    r"mlp_total=(?P<mlp_total>[0-9]+)\s+"
    r"dronet_fresh=(?P<dronet_fresh>[0-9]+)\s+"
    r"mlp_fresh=(?P<mlp_fresh>[0-9]+)\s+"
    r"dronet_sensor_generated=(?P<dronet_sensor_generated>[0-9]+)\s+"
    r"mlp_sensor_generated=(?P<mlp_sensor_generated>[0-9]+)"
)

KV_RE = re.compile(r"^\s*(?P<key>[a-zA-Z0-9_]+)\s*=\s*(?P<value>.+?)\s*$")


def maybe_number(text: str):
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_log(path: Path):
    summary = {}
    last_stats = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        m = STATS_RE.match(line)
        if m:
            last_stats = {k: maybe_number(v) for k, v in m.groupdict().items()}
            continue

        kv = KV_RE.match(line)
        if kv:
            key = kv.group("key")
            value = maybe_number(kv.group("value"))
            summary[key] = value

    return {
        "last_stats": last_stats,
        "run_complete": summary,
        "line_count": len(lines),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to runtime log")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    args = parser.parse_args()

    parsed = parse_log(Path(args.log))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
