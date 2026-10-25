"""Isolated actual pinned planner/reader probes for new source-domain variants.

No production implementation is copied or changed. Captured-constant source
reading and connected-region planning are different APIs, with distinct results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def reference(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def load_projection():
    path = Path(__file__).with_name("lean_graph_regions_projection.py")
    spec = importlib.util.spec_from_file_location("primary_source_bridge_projection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("primary source projector unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    root = args.reference.resolve(strict=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != REVISION:
        raise ValueError("different production revision")
    from merlin.compile.connected_regions import plan_connected_region
    from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput

    projection = load_projection()
    module = projection.parse_source(args.source.read_text())
    operations = tuple(op for op in module.walk() if op.name == "linalg.generic")[:3]
    before = projection.ir_text(module)
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema="merlin.graph-regions.source-production-probe.v1",
        revision=revision,
        source_input=reference(args.source),
        worker=reference(Path(__file__)),
    )
    try:
        plan = plan_connected_region(module, operations)
        plan.revalidate()
        projected, ordinals, ids = projection.project(module)
        result["connected_plan"] = dict(status="accepted")
        (destination / "plan.json").write_text(json.dumps(projection.planner_record(plan, ordinals, ids, projected)))
        (destination / "native-plan.json").write_text(json.dumps(plan.record(), indent=2))
    except ValueError as error:
        result["connected_plan"] = dict(status="refused", diagnostic=str(error))
    try:
        source = read_source_multioutput(module, operations)
        result["multioutput_source_reader"] = dict(
            status="accepted", input_count=len(source.inputs), output_count=len(source.outputs)
        )
    except ValueError as error:
        result["multioutput_source_reader"] = dict(status="refused", diagnostic=str(error))
    if projection.ir_text(module) != before:
        raise ValueError("production probes changed source")
    (destination / "source.mlir").write_text(before)
    (destination / "retained-source.mlir").write_text(projection.ir_text(module))
    sources = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("merlin.") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        relative = path.relative_to(root)
        expected = subprocess.check_output(["git", "show", f"{REVISION}:{relative.as_posix()}"], cwd=root)
        if path.read_bytes() != expected:
            raise ValueError("loaded production source is not exact pinned content")
        sources[name] = reference(path)
    result.update(
        production_sources=sources,
        source=reference(destination / "source.mlir"),
        after=reference(destination / "retained-source.mlir"),
        python=reference(Path(sys.executable)),
        native_implementation_contract_proved=False,
    )
    (destination / "PROBE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def capture(source: Path, reference_root: Path, destination: Path):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(reference_root / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--source",
        str(source.resolve()),
        "--reference",
        str(reference_root.resolve()),
        "--destination",
        str(destination.resolve()),
    ]
    result = subprocess.run(command, cwd=reference_root, env=env, capture_output=True, text=True, timeout=90)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / (destination.name + "-process.json")).write_text(
        json.dumps(
            dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2
        )
        + "\n"
    )
    result.check_returncode()
    return json.loads((destination / "PROBE.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        print(json.dumps(capture(args.source, args.reference, args.destination)))


if __name__ == "__main__":
    main()
