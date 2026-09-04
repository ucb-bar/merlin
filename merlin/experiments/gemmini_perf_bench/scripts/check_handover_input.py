#!/usr/bin/env python3
"""Refuse an input the compiler would accept and then emit nothing for.

THE FAILURE THIS EXISTS TO STOP. The shipped compiler consumes one closed interface dialect. Handed
a module written in anything else -- the linalg-on-tensors that a real model capture actually
produces -- it does not fail. It exits 0, emits a command buffer with an EMPTY command list and a
``declined`` note, and the toolchain downstream happily builds and runs that: a kernel that does no
work, finishes almost immediately, and reports a cycle count far below the baseline.

Reproduced on the sealed package: a four-line module with no interface op returned exit 0, zero
commands, and nothing on stderr.

For someone benchmarking the compiler on their own models this is the worst possible behaviour,
because the artefact of the failure is a SPECTACULAR speedup. Anyone comparing cycles would conclude
the compiler is enormously fast on exactly the inputs it cannot compile at all. So this checks, ahead
of any measurement, that the compiler actually emitted work -- and says what is missing when it did
not.

Run it on every input before measuring:

    check_handover_input.py --package <submission> --input model.interface.mlir
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

#: The manifest key naming the compiler entrypoint, and the command that emits a command buffer.
TOOL_KEY = "tool"
EMIT_COMMAND = "emit_command_buffer"
#: A command buffer carrying this key was declined: the compiler read the module and routed none of
#: it. Its presence is the compiler's own admission, and it is the signal this refuses on.
DECLINED = "declined"


def _entrypoint(package: Path) -> Path:
    import yaml

    manifest = package / "manifest.yaml"
    document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    tool = (document.get("entrypoints") or {}).get(TOOL_KEY)
    if not isinstance(tool, str) or not tool:
        raise SystemExit(f"{manifest} declares no {TOOL_KEY!r} entrypoint")
    resolved = package / tool
    if not resolved.is_file():
        raise SystemExit(f"the declared entrypoint does not exist: {resolved}")
    return resolved


def inspect(package: Path, source: Path) -> dict:
    """Compile one input to a command buffer and report whether it carries any work."""
    tool = _entrypoint(package)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "command_buffer.json"
        proc = subprocess.run(
            [sys.executable, str(tool), "--convert-iface-to-gemmini",
             f"--emit-command-buffer={out}", str(source)],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return {"status": "refused_by_compiler", "returncode": proc.returncode,
                    "stderr": (proc.stderr or "")[-2000:],
                    "reason": "the compiler rejected this input, which is an honest failure"}
        if not out.is_file():
            return {"status": "no_command_buffer",
                    "reason": "the compiler exited 0 but emitted no command buffer"}
        buffer = json.loads(out.read_text(encoding="utf-8"))
    commands = buffer.get("commands")
    declined = buffer.get(DECLINED)
    n = len(commands) if isinstance(commands, list) else 0
    if declined or n == 0:
        return {
            "status": "empty_kernel", "commands": n, "declined": declined,
            "reason": ("the compiler ACCEPTED this input and emitted no work. Measuring it would "
                       "report a near-zero cycle count and read as an enormous speedup. The input "
                       "is not in the interface dialect this compiler consumes, or uses an "
                       "operation it does not route."),
        }
    return {"status": "ok", "commands": n,
            "reason": f"the compiler emitted {n} command(s); this input is measurable"}


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path, help="the submission tree to check")
    ap.add_argument("--input", required=True, nargs="+", type=Path, help="interface MLIR file(s)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = []
    worst = 0
    for source in args.input:
        if not source.is_file():
            rows.append({"input": str(source), "status": "missing"})
            worst = 2
            continue
        row = {"input": str(source), **inspect(args.package.resolve(), source)}
        rows.append(row)
        if row["status"] == "empty_kernel":
            worst = max(worst, 2)
        elif row["status"] != "ok":
            worst = max(worst, 1)

    for row in rows:
        mark = {"ok": "  OK      ", "empty_kernel": "  REFUSED "}.get(row["status"], "  WARN    ")
        print(f"{mark}{Path(row['input']).name}: {row.get('reason', row['status'])}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if worst == 2:
        print("\nDO NOT MEASURE the refused input(s): an empty kernel reports a fast cycle count "
              "for work it never did.")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
