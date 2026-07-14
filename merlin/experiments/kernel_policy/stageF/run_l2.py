#!/usr/bin/env python3
"""Stage-F L2 runner: compile + run the slate ablations under Spike, count events.

Each insight is a paired ablation (decision on vs off). Programs run under
``spike --extension=gemmini -l`` (or plain ``--isa=rv64gcv`` for vector insights) and
events are counted from the commit log: Gemmini commands by decoding the custom-3 opcode
(0x7B; funct: 0 config, 1 mvin2/B, 2 mvin/A, 3 mvout, 4/5 compute, 6 preload, 7 flush),
fences textually. Spike is functional — counts are events, never cycles.

Usage:
  MERLIN_CHIPYARD=/path/to/chipyard \
  python run_l2.py --insight resident_rhs|accumulator_commit|dispatch_batching|vl_tail|all
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the ablation .c kernels are a library-consumed benchmark input (compiled by cost_model.calibrate too),
# so they live under merlin/benchmarks/cost_calib/. HERE=stageF -> parents[2]=merlin/.
CALIB_SRC = HERE.parents[2] / "benchmarks" / "cost_calib"
FUNCT_NAMES = {0: "config", 1: "mvin2_B", 2: "mvin_A", 3: "mvout", 4: "compute_preloaded",
               5: "compute_accumulated", 6: "preload", 7: "flush", 14: "mvin3_bias"}
_LOGLINE = re.compile(r"\(0x([0-9a-f]{8})\)")
DIM, ELEM, ACC_ELEM = 16, 1, 4

INSIGHTS = {
    "resident_rhs": {
        "harness": "resident_rhs_ablation.c",
        "variants": ("baseline", "hoisted", "oracle"),
        "sweep": ("REPS", [1, 2, 4, 8, 16]),
        "march": "rv64gc", "isa": None,
    },
    "accumulator_commit": {
        "harness": "accumulator_commit_ablation.c",
        "variants": ("baseline", "fused"),
        "sweep": ("REPS", [1, 4, 8]),
        "march": "rv64gc", "isa": None,
    },
    "dispatch_batching": {
        "harness": "dispatch_batching_ablation.c",
        "variants": ("baseline", "batched"),
        "sweep": ("TILES", [4, 16, 39]),   # 39 = corpus median dispatches/kernel
        "march": "rv64gc", "isa": None,
    },
    "vl_tail": {
        "harness": "vl_tail_ablation.c",
        "variants": ("fixed", "vla"),
        "sweep": ("N_ELEMS", [64, 70, 1024, 1030]),
        "march": "rv64gcv", "isa": "rv64gcv",
    },
}


def env_paths() -> dict[str, Path]:
    cy = Path(os.environ.get("MERLIN_CHIPYARD", "/path/to/chipyard"))
    tools = cy / ".conda-env" / "riscv-tools"
    return {
        "gcc": tools / "bin" / "riscv64-unknown-elf-gcc",
        "spike": tools / "bin" / "spike",
        "pk": tools / "riscv64-unknown-elf" / "bin" / "pk",
        "rocc": cy / "generators" / "gemmini" / "software" / "gemmini-rocc-tests",
    }


def compile_variant(paths, spec, variant: str, n: int, out_dir: Path) -> Path:
    out = out_dir / f"{spec['harness'][:-2]}_{variant}_{n}"
    cmd = [str(paths["gcc"]), "-O2", "-std=gnu99", f"-march={spec['march']}",
           "-mcmodel=medany", "-fno-common", "-fno-builtin-printf",
           f"-DVARIANT_{variant.upper()}", f"-D{spec['sweep'][0]}={n}",
           f"-I{paths['rocc']}", f"-I{paths['rocc']}/include",
           str(CALIB_SRC / spec["harness"]), "-o", str(out), "-lm"]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def run_and_count(paths, spec, binary: Path) -> dict:
    cmd = [str(paths["spike"])]
    cmd += [f"--isa={spec['isa']}"] if spec["isa"] else ["--extension=gemmini"]
    cmd += ["-l", str(paths["pk"]), str(binary)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    counts: collections.Counter = collections.Counter()
    instret = 0
    for line in proc.stderr:
        m = _LOGLINE.search(line)
        if not m:
            continue
        instret += 1
        word = int(m.group(1), 16)
        if word & 0x7F == 0x7B:
            counts[FUNCT_NAMES.get(word >> 25, f"funct{word >> 25}")] += 1
        elif word == 0x0FF0000F:
            counts["fence"] += 1
    stdout = proc.stdout.read()
    proc.wait()
    if "PASS" not in stdout:
        raise RuntimeError(f"{binary.name}: correctness check failed: {stdout!r}")
    counts["instret"] = instret
    counts["gemmini_cmds"] = sum(v for k, v in counts.items()
                                 if k not in ("instret", "fence"))
    return dict(counts)


def sweep(insight: str) -> list[dict]:
    spec = INSIGHTS[insight]
    paths = env_paths()
    for k, p in paths.items():
        if not p.exists():
            sys.exit(f"missing {k}: {p} (set MERLIN_CHIPYARD)")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for n in spec["sweep"][1]:
            for variant in spec["variants"]:
                ev = run_and_count(paths, spec, compile_variant(paths, spec, variant, n, Path(td)))
                rows.append({"insight": insight, "variant": variant,
                             spec["sweep"][0]: n, "events": ev})
                print(f"{insight} {spec['sweep'][0]}={n:>4} {variant:<8} "
                      + " ".join(f"{k}={v}" for k, v in sorted(ev.items())))
    return rows


def decide(insight: str, rows: list[dict]) -> dict:
    """Apply the slate's act_if rule; returns the summary dict."""
    key = INSIGHTS[insight]["sweep"][0]
    nmax = max(r[key] for r in rows)
    at = {r["variant"]: r["events"] for r in rows if r[key] == nmax}
    s: dict = {"slate": insight, "fidelity": "L2_event_counts (events, not cycles)",
               "at": {key: nmax}, "rows": rows}
    if insight == "resident_rhs":
        base, hoist, oracle = (at[v].get("mvin2_B", 0) for v in ("baseline", "hoisted", "oracle"))
        s["rhs_traffic_ratio"] = base / hoist
        s["exploitability"] = (base - hoist) / (base - oracle) if base != oracle else 1.0
        s["decision"] = "act" if s["rhs_traffic_ratio"] >= 2 and s["exploitability"] >= 0.5 else "park"
    elif insight == "accumulator_commit":
        b, f = at["baseline"], at["fused"]
        s["mvout_bytes_ratio"] = (b["mvout"] * ACC_ELEM) / (f["mvout"] * ELEM)
        s["instret_saved"] = b["instret"] - f["instret"]
        s["decision"] = "act" if s["mvout_bytes_ratio"] >= 1.8 else "park"
    elif insight == "dispatch_batching":
        b, g = at["baseline"], at["batched"]
        s["config_fraction_baseline"] = round((b["config"] + b["fence"]) / b["gemmini_cmds"], 3)
        s["commands_removed_fraction"] = round(1 - g["gemmini_cmds"] / b["gemmini_cmds"], 3)
        s["decision"] = ("act" if s["config_fraction_baseline"] > 0.3
                         and s["commands_removed_fraction"] >= 0.3 else "park")
    elif insight == "vl_tail":
        tail = {r[key]: r for r in rows if r["variant"] == "fixed"}
        vla = {r[key]: r for r in rows if r["variant"] == "vla"}
        s["tail_overhead_fraction"] = {
            n: round(tail[n]["events"]["instret"] / vla[n]["events"]["instret"] - 1, 4)
            for n in tail}
        worst = max(s["tail_overhead_fraction"].values())
        s["decision"] = ("act" if worst >= 0.1 else
                         "park (no measurable instret win at VLEN=128; portability rationale only)")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insight", default="all",
                    choices=[*INSIGHTS, "all"])
    ap.add_argument("--out-dir", default="out/artifacts/kernel-mining/stageF")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for insight in INSIGHTS if args.insight == "all" else [args.insight]:
        summary = decide(insight, sweep(insight))
        out = out_dir / f"{insight}_l2.json"
        out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        keys = [k for k in summary if k not in ("rows", "slate", "fidelity", "at")]
        print(f"\n{insight}: " + ", ".join(f"{k}={summary[k]}" for k in keys))
        print(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
