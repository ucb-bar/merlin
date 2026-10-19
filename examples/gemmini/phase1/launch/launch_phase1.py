"""Operator-only fresh launch from one config; reject incomplete input preparation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from merlin_experiments.phase1 import launch_config as L

from merlin.model_check import bundle as B
from merlin.model_check import seal as S


def save(path: Path, value: object) -> None:
    staging = path.with_suffix(path.suffix + ".next")
    staging.write_text(json.dumps(value, indent=2) + "\n")
    staging.replace(path)


def operator_module(name: str):
    """Load the selected sibling bytes even under Python's safe-path startup."""
    source = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"operator helper is unavailable: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def memory(records: dict[int, tuple[int, int]]) -> tuple[int, int]:
    charged = 0
    for pid in records:
        try:
            rows = (Path("/proc") / str(pid) / "smaps_rollup").read_text().splitlines()
            charged += next(int(row.split()[1]) * 1024 for row in rows if row.startswith("Pss:"))
        except (OSError, ValueError, StopIteration):
            # Match the campaign guard: inaccessible PSS is charged at full RSS.
            try:
                charged += int((Path("/proc") / str(pid) / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
    available = next(
        int(row.split()[1]) * 1024
        for row in Path("/proc/meminfo").read_text().splitlines()
        if row.startswith("MemAvailable:")
    )
    return charged, available


def prepare(launch: L.Launch, prep: Path) -> dict:
    doc = launch.document
    origin = doc["origin"]
    if (
        origin["seed_submission"] is not None
        or origin["resume"]
        or not origin["initial_submission"].startswith("empty")
    ):
        raise ValueError("this launcher requires an empty submission with no seed or resume")
    repo = launch.path(doc["inputs"]["repo"])
    baseline = launch.variables["SOURCE_REVISION"]
    # Operator metadata can be committed without changing the selected compiler
    # harness. Comparing the tree avoids an impossible self-referential HEAD pin
    # in the tracked config itself; the launcher bytes have their own hash pin.
    scope = ["--", ".", ":(exclude)examples/gemmini/phase1/launch"]
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", baseline, "HEAD"], cwd=repo)
    changed = subprocess.run(["git", "diff", "--quiet", baseline, *scope], cwd=repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", *scope], cwd=repo, text=True
    )
    if ancestry.returncode or changed.returncode or untracked.strip():
        raise ValueError(f"source tree differs from selected revision {baseline}")
    required = {"private_bundle_manifest", "private_bundle_sums"}
    if not required <= set(doc.get("runtime_pins") or {}):
        raise ValueError(
            "private model inputs are not ready: config lacks pinned private_bundle_manifest/private_bundle_sums; "
            "finish and verify input preparation first"
        )
    selected = Path(str(launch.environment["MERLIN_MODEL_CHECK_BUNDLE"]))
    for key, name in [("private_bundle_manifest", "MANIFEST.json"), ("private_bundle_sums", "SHA256SUMS")]:
        if launch.path(doc["runtime_pins"][key]["path"]).resolve() != (selected / name).resolve():
            raise ValueError(f"{key} does not select the configured private bundle")
    private_spec = launch.path(doc["inputs"]["private_full_model_spec"])
    import yaml

    spec = yaml.safe_load(private_spec.read_text())
    if Path(spec["numerics_bundle"]["path"]).resolve() != selected.resolve():
        raise ValueError("full-model selection and configured private input bundle differ")
    wanted = {row["id"] for row in spec["models"]}
    manifest = B.Bundle(selected).manifest()
    if wanted != set(manifest.get("models") or {}):
        raise ValueError("private bundle does not contain exactly the selected full models")
    out = launch.path(doc["launcher"]["environment_file"]).parent
    if launch.path(doc["launcher"]["plan"]).parent != out:
        raise ValueError("rendered environment and plan must use one directory")
    errata = doc.get("errata")
    if errata:
        content = (launch.config.path.parent / errata["source"]).read_bytes()
        target = launch.path(errata["path"])
        if target.exists() and target.read_bytes() != content:
            raise ValueError("existing served errata differs; use a fresh preparation directory")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    L.render(launch, out)
    plan = launch.path(doc["launcher"]["plan"])
    desired = L.launch_plan(launch, {})
    if plan.exists() and json.loads(plan.read_text()) != desired:
        raise ValueError("existing operator plan differs; use a fresh preparation directory")
    save(plan, desired)
    report = L.verify(launch)
    record = L.report_record(launch, report)
    save(out / "config-verification.json", record)
    if report.drifts:
        raise ValueError(f"launch config drift: see {out / 'config-verification.json'}")
    # This checks every selected byte, not just that a manifest exists. Never seal
    # or repair here: incomplete data must stop before any author is started.
    verified = S.verify(B.Bundle(selected), jobs=min(8, len(launch.manager_cpus)))
    save(out / "private-input-verification.json", verified)
    if verified["status"] != "pass":
        raise ValueError("private full-model input seal failed verification")
    if verified["manifest_sha256"] != doc["runtime_pins"]["private_bundle_manifest"]["sha256"]:
        raise ValueError("private input manifest changed during preparation")
    references = {int(cpu) for cpu in launch.variables["REFERENCE_CPUS"].split(",")}
    author = set(doc["cpu"]["author"]["cpus"])
    compute = set(doc["cpu"]["compute_pool"]["cpus"])
    if (references & author) or (references & compute) or (author & compute):
        raise ValueError("reference, author and compute CPU assignments overlap")
    assigned = references | author | compute
    if len(assigned) != 18 or not assigned <= os.sched_getaffinity(0):
        raise ValueError("the full selected 18-CPU allocation is unavailable")
    readiness = operator_module("full_model_verification").validate_readiness(launch)
    return {
        "status": "ready",
        "assigned_cpus": sorted(assigned),
        "config": record["config"],
        "private_inputs": verified,
        "full_model_readiness": readiness,
        "readiness_scope": "infrastructure_preflight"
        if readiness and readiness.get("schema") == "merlin.phase1.infrastructure_readiness.v2"
        else "retained_full_model_qualification" if readiness else "launch_configuration",
        "phase1_complete": False,
    }


def run(launch: L.Launch, prep: Path, started: float) -> int:
    operator_module("full_model_verification")
    return operator_module("launch_supervisor").run(launch, prep, started, memory=memory)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("phase1_el4_config.yaml"))
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--var", action="append", default=[])
    parser.add_argument(
        "--run",
        action="store_true",
        help="start a fresh native controller after every readiness check passes; default checks only",
    )
    args = parser.parse_args()
    try:
        overrides = L._overrides(args.var)
        launch = L.build(
            L.load(args.config), task_root=args.task_root, prep=args.prep, run_id=args.run_id, overrides=overrides
        )
        repo = launch.path(launch.document["inputs"]["repo"])
        if args.run and (repo / "out/runs" / launch.document["identity"]["target"] / "phase1" / args.run_id).exists():
            raise ValueError("run ID already exists; fresh launches require a new ID")
        args.prep.mkdir(parents=True, exist_ok=True)
        ready = prepare(launch, args.prep)
        save(args.prep / "launch-readiness.json", ready)
        print(json.dumps({"status": "ready", "readiness_scope": ready["readiness_scope"], "phase1_complete": False,
                          "config": str(launch.config.path), "run_id": args.run_id}), flush=True)
        return run(launch, args.prep, started) if args.run else 0
    except (ValueError, OSError, B.BundleError, RuntimeError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
