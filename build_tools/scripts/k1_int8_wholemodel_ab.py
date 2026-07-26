#!/usr/bin/env python
"""Whole-model A/B of the int8 (W8A8) code-generation RECIPE on the real K1 board.

The kernel-region A/B (``k1_int8_recipe_ab.py``) showed the incumbent int8 recipe
(``accumulator_resident_wholemodel_vf``) is 1.6-1.8x off the v3 micro-kernel + ``erase_self_copy``
path the f32 arm uses. A kernel win that vanishes whole-model is not a win, so this runs the SAME
recipes on a whole int8 model, N reps each, gated with the repo's MULTI-TIER int8 gate (T1 W8A8
cos/rel vs golden_w8a8, T2 fp32 cos + argmax) -- not the fp32-only cos>=0.9999 gate, which int8
legitimately fails.

Fail-closed: a config whose gate does not pass carries no timing.

Usage:
    build_tools/scripts/k1_int8_wholemodel_ab.py --models bitvla,openvla -n 3
"""
from __future__ import annotations

import argparse, json, shutil, tempfile, time
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.baselines import bundle as _bundle
from merlin.common.paths import artifacts_dir, repo_root
from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm

#: The recipes under test. ``microkernel`` is the knob-block spelling of the SHARED micro-kernel
#: capability (registry._resolve_features -> from_strategy._rvv_microkernel_resolver), which is the
#: whole point: int8 and f32 name the same recipe the same way and differ only in dtype_strategy.
CONFIGS = {
    "int8_vf":       {"compiler_features": ["accumulator_resident_wholemodel_vf"]},   # incumbent
    "int8_v3_knob":  {"microkernel": {"MR": 4, "NR": 16, "KC": 16}},                  # shared path
    "int8_baseline": {},                                                              # hand_v0_int8
}


def _pkg_for(base, tag: str, cfg: dict):
    """Build the package for one config. A ``microkernel`` block must go through the knob resolver
    (that IS the shared capability), so it is applied to knobs and re-resolved, not hand-listed."""
    if "microkernel" in cfg:
        from merlin.rvvgen.registry import _resolve_features
        knobs = {**base.knobs, "microkernel": cfg["microkernel"]}
        return replace(base, run_id=tag, knobs=knobs,
                       compiler_features=_resolve_features(knobs, base.manifest))
    return replace(base, run_id=tag, compiler_features=list(cfg.get("compiler_features", [])))


def _append(out: Path, rec: dict) -> None:
    """Append one finished config to the ledger IMMEDIATELY.

    A whole-model int8 config costs many minutes of host clang plus a board run, so a campaign that
    only writes at the end throws away every completed config when a later one hangs or the process
    is interrupted. Configs that DID finish are results and are kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**rec, "t": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}) + "\n")


def run_model(model: str, base, n: int, timeout: int, out: Path,
              only: set[str] | None = None) -> list[dict]:
    b = _bundle.resolve(model, "int8")
    mdir = b.root
    refs = {"fp32": np.load(mdir / "golden.npy")}
    w8 = mdir / "golden_w8a8.npy"
    if w8.is_file():
        refs["w8a8"] = np.load(w8)
    rows = []
    for tag, cfg in CONFIGS.items():
        if only and tag not in only:
            continue
        pkg = _pkg_for(base, tag, cfg)
        rec = {"model": model, "bundle": mdir.name, "config": tag,
               "compiler_features": list(pkg.compiler_features or []),
               "tiers": list(refs), "board": "k1_spacemit", "vlen": k1.VLEN}
        walls, gate, blocker = [], None, None
        print(f"=== {model} / {tag} {rec['compiler_features']} ===", flush=True)
        # BUILD ONCE, RUN n TIMES. clang is deterministic, so rebuilding per repeat puts a
        # byte-identical object under each measurement while tripling the campaign's cost --
        # and a whole-model int8 compile here is ~25 min at ~10 GB RSS. The variance the repeats
        # are FOR is board-side (thermal, scheduling, cache), which needs the same binary run
        # again, not a new one.
        work = Path(tempfile.mkdtemp(prefix=f"k1i8ab_{tag}_"))
        try:
            binary = k1.build_k1_binary(mdir, work, pkg, inputs_npz=mdir / "inputs.npz")
            for i in range(n):
                try:
                    res = k1.run_binary_on_k1(mdir, work, pkg, binary, timeout=timeout)
                    gate = zm._gate(res["prefix"], refs)
                    w = res.get("metrics", {}).get("wall_ns")
                    if w:
                        walls.append(w)
                    print(f"  run {i}: wall_ns={w} gate_ok={gate.get('ok')} cos={gate.get('cos')}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    blocker = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                    print(f"  run {i}: BLOCKED -- {blocker}", flush=True)
                    break
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            print(f"  build BLOCKED -- {blocker}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        ok = bool(gate and gate.get("ok")) and bool(walls)
        rec.update(gate_ok=bool(gate and gate.get("ok")),
                   cos=(gate or {}).get("cos"), rel=(gate or {}).get("rel"),
                   w8a8_cos=(gate or {}).get("w8a8_cos"),
                   fp32_cos=(gate or {}).get("fp32_cos"),
                   # FAIL-CLOSED: no gate pass => no timing leaves this script.
                   min_wall_ns=(min(walls) if ok else None),
                   walls_ns=(sorted(walls) if ok else []), n=len(walls),
                   status=("pass" if ok else "not_run"),
                   blocker=blocker or (None if ok else "gate did not pass"))
        print(f"  -> {rec['status']} min_wall_ns={rec['min_wall_ns']}", flush=True)
        _append(out, rec)
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="bitvla")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--baseline", default="out/artifacts/targets/rvv/hand_v0_int8")
    ap.add_argument("--configs", default=None,
                    help=f"comma list of {sorted(CONFIGS)} (default: all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    only = {c.strip() for c in a.configs.split(",")} if a.configs else None
    if only and (bad := only - set(CONFIGS)):
        ap.error(f"unknown config(s) {sorted(bad)}; known: {sorted(CONFIGS)}")
    base = load_rvv_package(Path(repo_root()) / a.baseline)
    out = Path(a.out) if a.out else (artifacts_dir() / "measurements" / "k1_spacemit" /
                                     "int8_wholemodel_recipe_ab.jsonl")
    rows = []
    # NO board lock here on purpose. `k1.run_on_k1` already takes it, and takes it around the
    # deploy+run ONLY -- it deliberately cross-compiles outside the lock so concurrent agents can
    # build in parallel while just the physical board serializes. Wrapping the campaign in a second
    # board_lock() is wrong twice over: it holds the single K1 through hours of host clang that does
    # not touch the board, and because fcntl.flock is per open-file-description it DEADLOCKS the
    # campaign against itself the moment run_on_k1 opens the same lockfile on its own fd. That
    # deadlock is silent -- the process just sleeps in locks_lock_inode_wait looking like a slow
    # build -- and it stayed hidden while an earlier campaign was timing out in clang before ever
    # reaching the run phase.
    for m in a.models.split(","):
        if m:
            rows += run_model(m.strip(), base, a.n, a.timeout, out, only)
    print(f"\nwrote {out}")
    for r in rows:
        print(f"  {r['model']:12s} {r['config']:14s} {r['status']:8s} "
              f"min_wall_ns={r['min_wall_ns']} cos={r['cos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
