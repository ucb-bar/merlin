"""The custom-instruction scan of a large linked program, split by address over worker processes.

:func:`merlin.analysis.perf.isa_prohibition.custom_instructions` reads the disassembler's whole listing in one
interpreter loop: on SmolVLA's 640 MB open-model program that is 30 million lines and 9 million custom
instructions, one to two minutes of every build. The listing is a concatenation of function bodies, so
it splits exactly at function starts: each worker disassembles one contiguous address range with the
same disassembler (``--start-address``/``--stop-address`` at symbol boundaries, the first range open
below and the last open above) and parses it with the same rules, returning only what the check keeps
-- a count per (function, selector) in first-seen order and the first rows, in address order, of the
selectors asked for. :func:`merge` folds the ranges back in address order, so every count, every first-seen
order and every reported row is the one a single pass gives. Nothing here names a target.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: Rows kept per range, in address order, of the selectors asked for; the check reports at most this many.
ROWS_KEPT = 200


def boundaries(elf: Path, *, nm: Path, parts: int) -> list[int]:
    """Up to ``parts - 1`` split addresses, each the start of a code symbol, spread over the code."""
    listing = subprocess.run(
        [str(nm), "-n", "--defined-only", str(elf)], capture_output=True, text=True, check=True
    ).stdout
    starts = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] in ("T", "t"):
            try:
                starts.append(int(fields[0], 16))
            except ValueError:
                continue
    starts = sorted(set(starts))
    if parts <= 1 or len(starts) < 2:
        return []
    low, high = starts[0], starts[-1]
    chosen: list[int] = []
    for k in range(1, parts):
        goal = low + (high - low) * k // parts
        nearest = min(starts, key=lambda a, goal=goal: abs(a - goal))
        if nearest > low and (not chosen or nearest > chosen[-1]):
            chosen.append(nearest)
    return chosen


def scan_range(
    elf: Path, *, objdump: Path, opcode: int, start: int | None, stop: int | None, keep: Sequence[int]
) -> dict[str, Any]:
    """One address range, parsed exactly as ``custom_instructions`` parses the whole listing."""
    from merlin.analysis.perf.isa_prohibition import MAJOR_OPCODE_MASK, SELECTOR_SHIFT

    argv = [str(objdump), "-d"]
    if start is not None:
        argv.append(f"--start-address={hex(start)}")
    if stop is not None:
        argv.append(f"--stop-address={hex(stop)}")
    listing = subprocess.run([*argv, str(elf)], capture_output=True, text=True, check=True).stdout
    kept = {int(s) for s in keep}
    pairs: dict[tuple[str | None, int], int] = {}
    rows: list[dict[str, Any]] = []
    function = None
    for line in listing.splitlines():
        stripped = line.strip()
        if stripped.endswith(">:") and "<" in stripped:
            function = stripped[stripped.index("<") + 1 : -2]
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0].strip().endswith(":"):
            continue
        word_text = fields[1].strip()
        if len(word_text) != 8:
            continue
        try:
            word = int(word_text, 16)
        except ValueError:
            continue
        if word & MAJOR_OPCODE_MASK != opcode:
            continue
        selector = word >> SELECTOR_SHIFT
        pairs[(function, selector)] = pairs.get((function, selector), 0) + 1
        if selector in kept and len(rows) < ROWS_KEPT:
            rows.append(
                {"address": fields[0].strip()[:-1], "word": word_text, "selector": selector, "function": function}
            )
    return {
        "pairs": [[f, s, n] for (f, s), n in pairs.items()],
        "rows": rows,
        "opened_without_label": function is None and bool(pairs),
    }


def scan(elf: Path, *, objdump: Path, nm: Path, opcode: int, keep: Sequence[int], parts: int) -> list[dict[str, Any]]:
    """Every range's result, in address order, each computed in a worker process of its own."""
    cuts = boundaries(Path(elf), nm=nm, parts=parts)
    edges = [None, *cuts, None]
    ranges = list(zip(edges[:-1], edges[1:], strict=True))

    def one(span: tuple[int | None, int | None]) -> dict[str, Any]:
        start, stop = span
        argv = [
            sys.executable,
            "-m",
            __name__,
            json.dumps(
                {
                    "elf": str(elf),
                    "objdump": str(objdump),
                    "opcode": int(opcode),
                    "start": start,
                    "stop": stop,
                    "keep": [int(k) for k in keep],
                }
            ),
        ]
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"an instruction-scan worker failed: {done.stderr[-1500:]}")
        return json.loads(done.stdout)

    with ThreadPoolExecutor(max_workers=max(1, len(ranges))) as pool:
        results = list(pool.map(one, ranges))
    # A range after the first opens at a symbol start, so its first instruction has a label; one that
    # did not would attribute instructions to no function where a single pass names the previous one.
    if any(r["opened_without_label"] for r in results[1:]):
        raise RuntimeError("an instruction-scan range began without its function's label")
    return results


def _main(argv: list[str]) -> int:
    spec = json.loads(argv[0])
    result = scan_range(
        Path(spec["elf"]),
        objdump=Path(spec["objdump"]),
        opcode=int(spec["opcode"]),
        start=spec["start"],
        stop=spec["stop"],
        keep=spec["keep"],
    )
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
