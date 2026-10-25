#!/usr/bin/env python3
"""Qualify TinyLlama operational typing, preserving native counterexamples and pending claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.scalar_square_typing import check, replay_typing, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    base = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square"
    destination = base / "typing/SY_model_tiny_llama"
    preserved = json.loads((base / "composition/counterexamples-preserved.json").read_text())["files"]
    if any(_file_ref(Path(ref["path"])) != ref for ref in preserved):
        raise ValueError("a preserved numerical counterexample or original review record changed")
    if args.replay:
        import uuid

        work = destination / "replays" / uuid.uuid4().hex
        result = replay_typing(destination / "receipt.json", work, compiler_python=args.compiler_python)
        if not result["fresh"] or not result["matches_recorded"]:
            raise ValueError(result)
    else:
        print("Checking complete TinyLlama operational typing with the Lean-sound checker", flush=True)
        result = check(base / "SY_model_tiny_llama/receipt.json", destination, compiler_python=args.compiler_python)
    write(base / "typing/counterexamples-preserved.json", {"files": preserved})
    if any(_file_ref(Path(ref["path"])) != ref for ref in preserved):
        raise ValueError("a preserved numerical counterexample changed")
    print(json.dumps(result, indent=2), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
