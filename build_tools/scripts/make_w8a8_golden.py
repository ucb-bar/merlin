#!/usr/bin/env python3
"""Write the **W8A8** reference (`golden_w8a8.npy`) for int8 capture bundles.

Why this exists. `golden.npy` in an `*_int8_*` recapture is a **weight-only-int8**
reference: model2MLIR quantizes with torchAO ``int8_weight_only``, so the weights are int8
and **the activations stay fp32**. Merlin's int8 path computes **W8A8** (activations
quantized too). Those are different computations. Grading a W8A8 run against `golden.npy`
measures activation-quantization error rather than correctness, and produces a large,
entirely expected cosine drop that looks exactly like a codegen defect.

Every consumer (`compile_cli`, `rvvgen/runner`, the baselines arms, `test_zephyr_*`) prefers
`golden_w8a8.npy` when present and silently drops to the weaker fp32 tier when it is absent.
Until now **nothing in the repo wrote it** — it was produced ad hoc, which is how whole
batches of `_full` recaptures shipped without one. That cost a multi-hour hunt for a
TinyLlama int8 "board defect" that did not exist: measured against its W8A8 reference the
board matched the host at rel 0.0.

**What this reference can and cannot decide.** It is computed by *merlin's own* int8 datapath
(`dispatch_runtime.run_model(int8_compute=True)`), so it is an EXECUTION reference, not an
independent one: it answers "did the device reproduce what the host compiler computes", which is
what the TinyLlama hunt actually needed. It cannot answer "is our int8 arithmetic right", because
against it our arithmetic is right by construction — a host run scores `cos 1.0 / rel 0.0` no
matter what the datapath does. A W8A8 tier pass on a golden written here is therefore not
evidence about the arithmetic, and a `rel` of exactly 0.0 is the tell that the two sides are the
same program. Deciding the arithmetic needs a reference from OUTSIDE the compiler (a torchao
`int8_dyn_act_int8_weight` instance of the same seeded model, weights asserted equal to the
bundle's bit-for-bit). Each golden written here records which kind it is in
`golden_w8a8.provenance.json` beside it, so the two are never confused again.

Usage:
    make_w8a8_golden.py                     # every int8 bundle that lacks one
    make_w8a8_golden.py tiny_llama_int8_full small_llama_int8_full
    make_w8a8_golden.py --force <bundle>    # regenerate even if present
    make_w8a8_golden.py --list              # report coverage, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from merlin.common.artifacts import recaptures_dir
from merlin.runtime.dispatch_runtime import run_model


def int8_bundles() -> list[Path]:
    root = recaptures_dir()
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and "_int8" in d.name and (d / "model.mlir").is_file())


def cos(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n else float("nan")


def generate(bundle: Path, *, force: bool = False) -> tuple[bool, str]:
    target = bundle / "golden_w8a8.npy"
    if target.is_file() and not force:
        return True, "already present"
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"w8a8_{bundle.name}_") as work:
        res = run_model(str(bundle), work, int8_compute=True)
    out = np.asarray(res["output"], dtype=np.float32)
    np.save(target, out)
    # Say in the bundle what kind of reference this is (see the module docstring): a consumer that
    # reads a W8A8 tier pass off this file is reading a self-comparison, not a verdict.
    (bundle / "golden_w8a8.provenance.json").write_text(json.dumps({
        "producer": "build_tools/scripts/make_w8a8_golden.py",
        "computed_by": "merlin.runtime.dispatch_runtime.run_model(int8_compute=True)",
        "independent_of_merlin": False,
        "decides": "device-vs-host execution agreement",
        "does_not_decide": "whether merlin's int8 arithmetic is correct (rel is 0 by construction)",
        "created": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }, indent=2) + "\n", encoding="utf-8")
    gold = bundle / "golden.npy"
    note = ""
    if gold.is_file():
        # A LOW number here is the normal, expected weight-only-vs-W8A8 gap on a real
        # pretrained model (TinyLlama measures ~0.976); a random-init `_consistent` capture
        # has no activation outliers and sits near 1.0. Neither is a defect signal.
        note = f", cos vs weight-only golden {cos(out, np.load(gold)):.6f}"
    return True, f"wrote {out.size} elems in {time.time() - t0:.0f}s{note}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="*", help="bundle names (default: all int8 bundles)")
    ap.add_argument("--force", action="store_true", help="regenerate even if one exists")
    ap.add_argument("--list", action="store_true", help="report coverage and exit")
    args = ap.parse_args()

    root = recaptures_dir()
    targets = [root / b for b in args.bundles] if args.bundles else int8_bundles()

    if args.list:
        for b in int8_bundles():
            print(f"{'YES' if (b / 'golden_w8a8.npy').is_file() else 'no ':4} {b.name}")
        return 0

    rc = 0
    for b in targets:
        if not (b / "model.mlir").is_file():
            print(f"{b.name}: SKIP (no model.mlir)", flush=True)
            continue
        try:
            _, msg = generate(b, force=args.force)
            print(f"{b.name}: {msg}", flush=True)
        except Exception as exc:  # noqa: BLE001 — one bad bundle must not stop the batch
            print(f"{b.name}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
