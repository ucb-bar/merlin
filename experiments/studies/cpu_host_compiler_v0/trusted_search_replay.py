#!/usr/bin/env python3
"""Replay one trusted CPU-host search observation from a private broker ledger.

This helper is never staged into an agent workspace.  The driver uses it after the agent exits to
rerun the frozen beam-search state machine without touching the board a second time.  A missing or
different candidate/split observation fails closed, which makes a hand-written ``search_record``
insufficient for promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-policy", type=Path, required=True)
    parser.add_argument("--capsules", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    parent_policy = json.loads(args.parent_policy.read_text(encoding="utf-8"))
    candidate = str(policy.get("candidate_sha256", ""))
    parent = str(parent_policy.get("candidate_sha256", ""))
    index_path = args.ledger / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index.get("evaluations", {}).get(
        f"{parent}:{candidate}:{args.split}:{args.phase}")
    if not isinstance(entry, dict):
        raise SystemExit(f"trusted ledger has no {candidate}/{args.split}/{args.phase} evaluation")
    if int(entry.get("measurement_repeats", -1)) != args.repeats:
        raise SystemExit("trusted ledger repeat count differs from frozen search")
    if entry.get("policy_sha256") != _sha256(args.policy):
        raise SystemExit("candidate policy bytes differ from the trusted evaluation request")
    if (entry.get("parent_candidate_sha256") != parent or
            entry.get("parent_policy_sha256") != _sha256(args.parent_policy)):
        raise SystemExit("parent policy bytes differ from the trusted evaluation request")
    if entry.get("capsules_sha256") != _sha256(args.capsules):
        raise SystemExit("capsule sample differs from the trusted evaluation request")
    observations = args.ledger / str(entry.get("observations", ""))
    if not observations.is_file() or entry.get("observations_sha256") != _sha256(observations):
        raise SystemExit("trusted observation artifact is absent or changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(observations, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
