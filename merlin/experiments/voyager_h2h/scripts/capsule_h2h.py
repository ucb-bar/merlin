"""Plane A capsule head-to-head: the same capsules, the same simulator binary, two or more packages.

Each (arm, capsule) is one ``merlin.targetgen.oot_runner`` run -- contract, the four entrypoints, the
command-buffer semantic check, then the chosen simulator -- and the result is read from the runner's own
YAML verdict, never re-derived. Cycles are compared only between arms that both PASSED on the same
engine, and only when that engine is cycle-accurate; everything else is listed with its reason.

Every arm shares the reference package's command buffer (the bridge copies it), so an oracle pass means
the arm's output equals the reference semantics the reference package was certified against.

Provenance written with the table: the hardware pins as verified right now (drift is RECORDED, not
fixed -- other sessions use those checkouts), the simulator binary's digest, and each package's
manifest and schedule digests.

Usage (merlin venv):
    python merlin/experiments/voyager_h2h/scripts/capsule_h2h.py --target <target> \
        --arm reference=<pkg> --arm voyager=<bridge pkg> --simulator verilator --from-bridge <bridge pkg>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from merlin.common import provenance
from merlin.common.artifacts import new_product
from merlin.common.paths import env as dotenv
from merlin.common.paths import merlin_dir, runs_dir


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capsule_inputs(target: str, profile: str, names: list[str] | None) -> dict[str, Path]:
    root = merlin_dir() / "contract" / "capsules"
    doc = yaml.safe_load((root / "profiles" / f"{profile or target}.yaml").read_text())
    out = {}
    for entry in doc["capsules"]:
        if names is not None and entry["name"] not in names:
            continue
        path = root / entry["cat"] / entry["name"] / "capsule.interface.mlir"
        if path.is_file():
            out[entry["name"]] = path
    return out


def _lowered_in_bridge(package: Path) -> list[str]:
    doc = json.loads((package / "mlir_oot" / "lowering" / "voyager_schedules.json").read_text())
    return sorted(name for entry in doc["schedules"].values() if "ops" in entry
                  for name in entry["capsules"])


def _run(arm: str, package: Path, capsule: str, interface: Path, simulator: str,
         runs_root: Path, timeout: int, env: dict) -> dict:
    run_id = f"{arm}__{capsule}__{simulator}"
    cmd = [sys.executable, "-m", "merlin.targetgen.oot_runner", "--package", str(package),
           "--input", str(interface), "--run-id", run_id, "--simulator", simulator,
           "--runs-root", str(runs_root)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"arm": arm, "capsule": capsule, "status": "timeout", "run_id": run_id}
    try:
        verdict = yaml.safe_load(proc.stdout) or {}
    except yaml.YAMLError:
        verdict = {}
    oracle = verdict.get("oracle") or {}
    failure = verdict.get("failure") or {}
    return {"arm": arm, "capsule": capsule, "run_id": run_id, "rc": proc.returncode,
            "status": verdict.get("status", "unparsed"), "oracle_result": oracle.get("result"),
            "engine": oracle.get("engine"), "cycle_accurate": oracle.get("cycle_accurate"),
            "derived_from_rtl": oracle.get("derived_from_rtl"), "cycles": oracle.get("cycles"),
            "failure": {k: failure.get(k) for k in ("plane", "category", "detail")} if failure else None,
            "stderr_tail": (proc.stderr or "").splitlines()[-3:] if proc.returncode else []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PACKAGE")
    parser.add_argument("--simulator", default="verilator", choices=("spike", "gsim", "verilator"))
    parser.add_argument("--capsules", nargs="*")
    parser.add_argument("--from-bridge", type=Path, help="grade the capsules this bridge lowered")
    parser.add_argument("--profile")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    arms = dict(a.split("=", 1) for a in args.arm)
    arms = {name: Path(path).resolve() for name, path in arms.items()}
    names = args.capsules
    if args.from_bridge is not None:
        names = _lowered_in_bridge(args.from_bridge)
    inputs = _capsule_inputs(args.target, args.profile, names)
    env = dict(os.environ)
    for key in ("MERLIN_CHIPYARD",):
        if key not in env and dotenv(key):
            env[key] = dotenv(key)
    runs_root = runs_dir() / args.target / "voyager-h2h"

    jobs = [(arm, pkg, cap, path) for cap, path in inputs.items() for arm, pkg in arms.items()]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda j: _run(j[0], j[1], j[2], j[3], args.simulator, runs_root,
                                               args.timeout, env), jobs))

    by = {(r["arm"], r["capsule"]): r for r in results}
    arm_names = list(arms)
    base = arm_names[0]
    rows, ratios = [], {a: [] for a in arm_names[1:]}
    for cap in inputs:
        row = {"capsule": cap}
        for arm in arm_names:
            r = by[(arm, cap)]
            row[arm] = {"status": r["status"], "cycles": r["cycles"], "engine": r["engine"],
                        "cycle_accurate": r["cycle_accurate"], "failure": r["failure"]}
        for arm in arm_names[1:]:
            a, b = by[(arm, cap)], by[(base, cap)]
            comparable = (a["status"] == b["status"] == "pass" and a["cycle_accurate"]
                          and b["cycle_accurate"] and a["engine"] == b["engine"]
                          and a["cycles"] and b["cycles"])
            if comparable:
                ratio = a["cycles"] / b["cycles"]
                ratios[arm].append(ratio)
                row[f"{arm}/{base}"] = round(ratio, 6)
        rows.append(row)
    summary = {}
    for arm, rs in ratios.items():
        summary[arm] = {
            "matched": len(rs),
            "geomean_ratio_vs_" + base: (math.exp(sum(math.log(x) for x in rs) / len(rs))
                                         if rs else None),
            "wins": sum(1 for x in rs if x < 1), "losses": sum(1 for x in rs if x > 1),
            "ties": sum(1 for x in rs if x == 1),
            "passes": sum(1 for c in inputs if by[(arm, c)]["status"] == "pass")}
    summary[base] = {"passes": sum(1 for c in inputs if by[(base, c)]["status"] == "pass")}

    pins = {}
    for name in ("gemmini_rtl", "voyager_compiler"):
        try:
            pins[name] = provenance.verify(name)
        except Exception as exc:  # noqa: BLE001 -- recorded, not fatal: the drift IS the finding
            print(f"pin {name}: {exc}")
    sim_binaries = {}
    chipyard = env.get("MERLIN_CHIPYARD")
    if args.simulator == "verilator" and chipyard:
        from merlin.targetgen.target_experiment import load_capability_manifest
        cfg = (load_capability_manifest(args.target).contract.get("runtime") or {}).get(
            "rtl_sim_config")
        sim_binaries["verilator"] = Path(chipyard) / "sims" / "verilator" / \
            f"simulator-chipyard.harness-{cfg}"
    record = provenance.record(pins=pins, artifacts=sim_binaries)
    packages = {arm: {"path": str(pkg), "manifest_sha256": _sha256(pkg / "manifest.yaml"),
                      "schedules_sha256": _sha256(pkg / "mlir_oot" / "lowering"
                                                  / "voyager_schedules.json")}
                for arm, pkg in arms.items()}

    product = new_product("compare", version=1, target=args.target,
                          notes=f"voyager_h2h capsule head-to-head on {args.simulator}")
    doc = {"target": args.target, "simulator": args.simulator, "arms": packages,
           "baseline_arm": base, "summary": summary, "rows": rows, "runs": results,
           "provenance": record}
    (product.path / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    lines = [f"# Capsule head-to-head on {args.simulator} ({args.target})", "",
             f"Baseline arm: `{base}`. Cycles compared only where both arms pass on the same "
             "cycle-accurate engine.", "",
             "| capsule | " + " | ".join(arm_names) + " | " +
             " | ".join(f"{a}/{base}" for a in arm_names[1:]) + " |",
             "|---|" + "---|" * (len(arm_names) * 2 - 1)]
    for row in rows:
        cells = []
        for arm in arm_names:
            cell = row[arm]
            cells.append(f"{cell['status']} {cell['cycles'] or ''}".strip())
        ratios_cells = [str(row.get(f"{a}/{base}", "—")) for a in arm_names[1:]]
        lines.append(f"| {row['capsule']} | " + " | ".join(cells) + " | "
                     + " | ".join(ratios_cells) + " |")
    lines += ["", "## Summary", "", "```", json.dumps(summary, indent=1, default=str), "```", "",
              "## Pin state at run time", ""]
    for name, block in (record.get("hardware_pins") or {}).items():
        lines.append(f"- `{name}` ok={block.get('ok')} drift={block.get('drift')}")
    (product.path / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"product": str(product.path), "summary": summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
