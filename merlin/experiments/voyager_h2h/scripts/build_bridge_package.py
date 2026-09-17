"""Build the Voyager bridge package for a target (plane A: same hardware, compiler vs compiler).

For every resident-matmul capsule in the target's profile this runs the pinned Voyager compiler on the
equivalent int8 linear (in Voyager's own venv, through ``voyager_export.py``), replays its compiled
program (``merlin.baselines.voyager_ir``), lowers the schedule onto the target's derived geometry
(``merlin.baselines.voyager_schedule``), proves the lowered arithmetic with the numpy executor, and
stores the result. The package is the certified reference backend with one substitution: its
``build_trace`` replays these schedules (``package_template/voyager_replay.py``). Voyager decides the
schedule; the capsule fixes the arithmetic (readout, epilogue) -- see AGENT.md, concessions C1-C3.

A capsule Voyager cannot compile, or whose schedule the bridge cannot lower exactly, is recorded as
refused with its reason and stage. It is never filled in with another compiler's schedule.

Usage (merlin venv):
    python merlin/experiments/voyager_h2h/scripts/build_bridge_package.py \
        --target <target> --reference-package <path to the certified package> [--only A2_single_tile_matmul]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from merlin.baselines.voyager import accelerator_config_for, geometry_for
from merlin.baselines.voyager_ir import UnsupportedConstruct, load_model, replay
from merlin.baselines.voyager_schedule import Compute, Mvin, Mvout, Preload, execute, lower_gemm
from merlin.common import provenance
from merlin.common.paths import artifacts_dir, build_dir, merlin_dir

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "package_template" / "voyager_replay.py"
EXPORT = HERE / "voyager_export.py"
ARTIFACT_IMPORT = "from lowering.isa import Address, Instruction, build_trace"
ARTIFACT_IMPORT_REPLACED = (
    "from lowering.isa import Address, Instruction\nfrom lowering.voyager_replay import build_trace"
)


def schedule_key(m: int, k: int, n: int, epilogue: list[str], out_dtype: str) -> str:
    # Must equal package_template/voyager_replay.schedule_key.
    return f"{m}x{k}x{n}|{'+'.join(epilogue) or 'none'}|{out_dtype}"


def _serialize(op) -> list:
    if isinstance(op, Mvin):
        if op.row_step != 1 or op.role == "zero":
            # The replay template packs every load at the operand's own row stride.
            raise ValueError(f"a {op.role} load with row step {op.row_step} has no packing here")
        return ["mvin", op.role, op.dram_row, op.dram_col, op.rows, op.cols, op.spad_row]
    if isinstance(op, Preload):
        return ["preload", op.weight_row, op.acc_row, op.accumulate, op.rows, op.cols]
    if isinstance(op, Compute):
        return ["compute", op.input_row, op.rows, op.fresh_weights]
    if isinstance(op, Mvout):
        return ["mvout", op.role, op.dram_row, op.dram_col, op.rows, op.cols, op.acc_row]
    raise TypeError(op)


def _gemm_capsules(profile: dict, capsules_root: Path) -> list[dict]:
    out = []
    for entry in profile["capsules"]:
        if entry.get("op") != "matmul":
            continue
        path = capsules_root / entry["cat"] / entry["name"] / "capsule.yaml"
        doc = yaml.safe_load(path.read_text())
        attrs = (doc.get("operation") or {}).get("attributes", {})
        shapes = {i["name"]: (tuple(i["shape"]), i["dtype"]) for i in doc.get("inputs", ())}
        (m, k), lhs_dtype = shapes[attrs["lhs"]]
        (k2, n), w_dtype = shapes[attrs["weight"]]
        out.append(
            {
                "name": doc["name"],
                "label": doc.get("label"),
                "path": str(path),
                "M": m,
                "K": k,
                "N": n,
                "K_weight": k2,
                "dtypes": (lhs_dtype, w_dtype),
                "epilogue": list(attrs.get("epilogue", [])),
                "output_dtype": attrs.get("output_dtype"),
            }
        )
    return out


def _refusal_before_compile(cap: dict, dim: int) -> str | None:
    if cap["K"] != cap["K_weight"]:
        return f"contraction mismatch {cap['K']} vs {cap['K_weight']}"
    if cap["dtypes"] != ("i8", "i8"):
        return f"operand dtypes {cap['dtypes']}: the int8 Voyager flow is the arm under test"
    if "maxpool" in cap["epilogue"]:
        return "maxpool-on-store epilogue: Voyager's linear flow has no pooling readout"
    if any(x % dim for x in (cap["M"], cap["K"], cap["N"])):
        return (
            f"{cap['M']}x{cap['K']}x{cap['N']} is not whole {dim}-blocks: the bridge does not yet "
            "lower Voyager's padded edge tiles"
        )
    return None


def _export(cap: dict, config: dict, out: Path, voyager_python: Path, env: dict) -> float:
    out.mkdir(parents=True, exist_ok=True)
    workload = {
        "kind": "linear",
        "name": cap["name"],
        "M": cap["M"],
        "K": cap["K"],
        "N": cap["N"],
        "bias": False,
        "relu": "relu" in cap["epilogue"],
    }
    (out / "workload.json").write_text(json.dumps(workload, indent=1))
    (out / "config.json").write_text(json.dumps(config, indent=1))
    start = time.monotonic()
    proc = subprocess.run(
        [
            str(voyager_python),
            str(EXPORT),
            "--workload",
            str(out / "workload.json"),
            "--config",
            str(out / "config.json"),
            "--out",
            str(out / "voyager"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    wall = time.monotonic() - start
    (out / "export.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        tail = [line for line in (proc.stderr or proc.stdout).splitlines() if line.strip()][-1:]
        raise RuntimeError(f"voyager compile failed (rc={proc.returncode}): {tail}")
    return wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference-package", required=True, type=Path)
    parser.add_argument("--package-id", default="voyager_bridge_v0")
    parser.add_argument("--profile", help="profile name (defaults to the target)")
    parser.add_argument("--only", nargs="*", help="restrict to these capsule names")
    parser.add_argument("--voyager-python", type=Path, default=build_dir() / "voyager-venv" / "bin" / "python")
    parser.add_argument("--force", action="store_true", help="replace an existing package dir")
    args = parser.parse_args(argv)

    os.environ.setdefault("MERLIN_EXT_VOYAGER_COMPILER", str(build_dir() / "external" / "voyager-compiler"))
    verification = provenance.verify("voyager_compiler")
    if not verification.ok:
        print(f"refusing: the Voyager compiler checkout does not match its pin: {verification.to_dict()}")
        return 2
    compiler_pin = provenance.record(pins={"voyager_compiler": verification})
    compiler_commit = provenance.pin("voyager_compiler").commit
    derived = accelerator_config_for(args.target)
    geometry = geometry_for(args.target)
    capsules_root = merlin_dir() / "contract" / "capsules"
    profile = yaml.safe_load((capsules_root / "profiles" / f"{args.profile or args.target}.yaml").read_text())
    capsules = _gemm_capsules(profile, capsules_root)
    if args.only:
        capsules = [c for c in capsules if c["name"] in set(args.only)]

    pkg = artifacts_dir() / "targets" / args.target / args.package_id
    if pkg.exists():
        if not args.force:
            print(f"refusing to overwrite {pkg} (pass --force)")
            return 2
        shutil.rmtree(pkg)
    exports = pkg / "voyager_exports"
    env = {**os.environ, "TMPDIR": os.environ.get("TMPDIR", str(build_dir()))}

    schedules: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    for cap in capsules:
        key = schedule_key(cap["M"], cap["K"], cap["N"], cap["epilogue"], cap["output_dtype"])
        entry: dict = {"capsules": [cap["name"]]}
        if key in schedules:
            schedules[key]["capsules"].append(cap["name"])
            continue
        reason = _refusal_before_compile(cap, geometry.dim)
        if reason:
            entry.update(refused=reason, stage="bridge_admission")
        else:
            try:
                wall = _export(cap, derived.to_json(), exports / cap["name"], args.voyager_python, env)
                trace = replay(load_model(exports / cap["name"] / "voyager" / "model.json"))
                schedule = lower_gemm(trace, geometry)
                lhs = rng.integers(-128, 128, size=(cap["M"], cap["K"]))
                weight = rng.integers(-128, 128, size=(cap["K"], cap["N"]))
                if not np.array_equal(execute(schedule, lhs, weight), lhs @ weight):
                    raise UnsupportedConstruct("lowered schedule does not reproduce lhs @ weight")
                entry.update(
                    ops=[_serialize(op) for op in schedule.ops],
                    notes=schedule.notes,
                    voyager_compile_seconds=round(wall, 3),
                    counts={k.__name__: schedule.count(k) for k in (Mvin, Preload, Compute, Mvout)},
                )
            except (RuntimeError, UnsupportedConstruct, subprocess.TimeoutExpired) as exc:
                stage = (
                    "voyager_compile"
                    if isinstance(exc, (RuntimeError, subprocess.TimeoutExpired))
                    else "bridge_lowering"
                )
                entry.update(refused=str(exc), stage=stage)
        schedules[key] = entry
        state = "REFUSED " + entry.get("stage", "") if "refused" in entry else "ok"
        print(f"{cap['name']:32s} {key:28s} {state}  {entry.get('refused', entry.get('counts'))}")

    # Materialize: the certified reference package, with build_trace replaced by the replay.
    shutil.copytree(args.reference_package / "mlir_oot", pkg / "mlir_oot", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(TEMPLATE, pkg / "mlir_oot" / "lowering" / "voyager_replay.py")
    artifact = pkg / "mlir_oot" / "targetgen" / "generate" / "llvm_artifact.py"
    text = artifact.read_text()
    if text.count(ARTIFACT_IMPORT) != 1:
        raise SystemExit(f"{artifact}: expected exactly one `{ARTIFACT_IMPORT}` line to substitute")
    artifact.write_text(text.replace(ARTIFACT_IMPORT, ARTIFACT_IMPORT_REPLACED))
    document = {
        "target": args.target,
        "voyager_compiler": compiler_pin,
        "accelerator_config": derived.fields,
        "config_sources": derived.sources,
        "not_modelled": derived.not_modelled,
        "geometry": geometry.__dict__,
        "schedules": schedules,
    }
    (pkg / "mlir_oot" / "lowering" / "voyager_schedules.json").write_text(json.dumps(document, indent=1, default=str))

    manifest = yaml.safe_load((args.reference_package / "manifest.yaml").read_text())
    manifest.pop("publication", None)  # the reference's certification is not this package's
    manifest["package_id"] = args.package_id
    # The schema's own vocabulary: generated deterministically from a specification (Voyager's
    # compiled program), by no agent. What that specification was is the voyager_bridge block below.
    manifest["authoring"] = {
        "author": "merlin voyager_h2h bridge builder",
        "generated_by_agent": False,
        "mode": "deterministic_generated_from_spec",
    }
    manifest["voyager_bridge"] = {
        "reference_package": str(args.reference_package),
        "compiler_commit": compiler_commit,
        "schedules": sum(1 for e in schedules.values() if "ops" in e),
        "refused": {k: e["refused"] for k, e in schedules.items() if "refused" in e},
    }
    (pkg / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    ok = sum(1 for e in schedules.values() if "ops" in e)
    print(f"package: {pkg}\nschedules: {ok} lowered, {len(schedules) - ok} refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
