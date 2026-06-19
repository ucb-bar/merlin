"""Framework attainment — the MEASURED gap to the expert kernel itself (not just structural).

``attainment = ceiling_cycles / our_cycles`` for a matched ``(op, dtype, shape, target)``: 1.0 means
we match the curated kernel's cycles; <1 means slower. The ceiling comes from running the curated
kernel STANDALONE (``kernels.bench_ceiling`` -> ceiling.jsonl); ours from the runner's results.yaml.
Joined on the CCA key so the report shows how close our codegen is to the expert, per target.

Honest by construction (not_run discipline): if no ceiling was measured for a key, attainment is
``None`` with reason "ceiling_not_measured" — never a fabricated number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def cca_key(op: str, dtype: str, shape: tuple | list, target: str) -> str:
    sh = "x".join(str(x) for x in shape) if shape else "?"
    return f"{op}|{dtype}|{sh}|{target}"


@dataclass
class Attainment:
    key: str
    ceiling_cycles: int | None
    our_cycles: int | None
    attainment: float | None        # ceiling/ours; None if either side missing
    reason: str = ""


def _load_ceilings(ceiling_jsonl: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not ceiling_jsonl.is_file():
        return out
    for line in ceiling_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = cca_key(r.get("op", "?"), r.get("dtype", "?"),
                    (r.get("M"), r.get("N"), r.get("K")), r.get("target", "?"))
        if r.get("cycles") is not None:
            out[k] = int(r["cycles"])
    return out


def _our_cycles(runs_root: Path) -> dict[str, int]:
    """Our measured cycles per (op,dtype,shape,target) from the runner results.yaml files."""
    out: dict[str, int] = {}
    for rd in sorted(runs_root.glob("*/results.yaml")):
        try:
            r = yaml.safe_load(rd.read_text())
        except Exception:  # noqa: BLE001
            continue

        def find(d, key):
            # Recurse dicts AND lists: the runner nests cycles/target inside a
            # ``measurement:`` LIST, so a dict-only walk silently drops them.
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    f = find(v, key)
                    if f is not None:
                        return f
            elif isinstance(d, (list, tuple)):
                for v in d:
                    f = find(v, key)
                    if f is not None:
                        return f
            return None
        cyc = find(r, "cycles")
        wl = find(r, "workload") or rd.parent.name
        # workload bundle name encodes op + shape, e.g. matmul_f32_64x64x64
        parts = str(wl).split("_")
        op = parts[0] if parts else "?"
        dtype = parts[1] if len(parts) > 1 else "?"
        shape = tuple(parts[2].split("x")) if len(parts) > 2 and "x" in parts[2] else ()
        # Run target lives in the measurement entry ("spike"/"k1"); the top-level
        # ``target`` is the backend family ("rvv"), which is NOT the join axis.
        meas = r.get("measurement") if isinstance(r, dict) else None
        target = None
        if isinstance(meas, list) and meas and isinstance(meas[0], dict):
            target = meas[0].get("target")
        target = target or find(r, "target") or ("spike" if not find(r, "vlen") else "k1")
        if cyc is not None:
            out[cca_key(op, dtype, shape, target)] = int(cyc)
    return out


def compute(ceiling_jsonl: str | Path, runs_root: str | Path) -> list[Attainment]:
    """Join measured ceiling vs ours per key; honest None when ceiling not measured."""
    ceil = _load_ceilings(Path(ceiling_jsonl))
    ours = _our_cycles(Path(runs_root))
    out: list[Attainment] = []
    for k in sorted(set(ceil) | set(ours)):
        c, o = ceil.get(k), ours.get(k)
        if c is None:
            out.append(Attainment(k, None, o, None, "ceiling_not_measured"))
        elif o is None:
            out.append(Attainment(k, c, None, None, "ours_not_measured"))
        else:
            out.append(Attainment(k, c, o, round(c / o, 3) if o else None))
    return out
