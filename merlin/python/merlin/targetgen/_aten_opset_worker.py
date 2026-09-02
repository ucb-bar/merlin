#!/usr/bin/env python3
"""Emit the PyTorch Core ATen IR opset. Runs in the m2m venv, where torch lives.

Separate process for the same reason the capture worker is: torch is not installed in the merlin venv,
so anything that needs it has to be asked rather than imported.

⚠ THE OBVIOUS API IS THE WRONG ONE. ``torch._decomp.core_aten_decompositions()`` looks like the Core
ATen opset and is not -- it is the DECOMPOSITION TABLE, i.e. the ops that get decomposed AWAY on the
path to Core ATen. Measured: 1004 entries against 188 core-tagged overloads. Using it as a denominator
would report coverage of roughly a fifth of the true figure, against a set that is close to the
complement of the one meant.

The authority is ``torch.Tag.core``, which torch stamps on the overloads that survive into Core ATen.

Prints one JSON object on stdout: ``{"torch": version, "n_core": int, "ops": [...]}``.
"""
from __future__ import annotations

import json
import sys


def core_opset() -> dict:
    import torch

    ops: set[str] = set()
    aten = torch.ops.aten
    for name in dir(aten):
        packet = getattr(aten, name, None)
        overloads = getattr(packet, "overloads", None)
        if overloads is None:
            continue
        try:
            names = list(overloads())
        except Exception:                          # noqa: BLE001 -- a packet with no overloads is not an op
            continue
        for overload in names:
            op = getattr(packet, overload, None)
            tags = getattr(op, "tags", ()) or ()
            if torch.Tag.core in tags:
                ops.add(str(op))
    return {"torch": torch.__version__, "n_core": len(ops), "ops": sorted(ops)}


def main(argv=None) -> int:
    try:
        print(json.dumps(core_opset()))
    except Exception as exc:                       # noqa: BLE001 -- report, never a partial opset
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
