"""Primary-owned actual-IR intake using a separately pinned, read-only producer.

This subprocess supplies bound evidence, not proof authority. Complete IR and
effect correspondence are checked by the pinned original projector; the new
primary checker separately checks the composed mathematical certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def external_runtime():
    """External parser dependencies remain trusted Python, with measured identities."""
    from importlib.metadata import version

    return {
        "python": file_ref(sys.executable),
        "python_version": sys.version,
        "xdsl_version": version("xdsl"),
        "loaded_sources": sorted(
            [
                file_ref(module.__file__)
                for name, module in tuple(sys.modules.items())
                if name.split(".")[0] in ("xdsl", "immutabledict", "typing_extensions")
                and getattr(module, "__file__", None)
            ],
            key=lambda row: row["path"],
        ),
    }


def capture(reference, source_path, target_path, destination, *, order=None, contract_path=None):
    reference, destination = Path(reference).resolve(), Path(destination).resolve()
    actual = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != REVISION:
        raise ValueError("producer revision differs from the selected graph source")
    subprocess.run(
        ["git", "-C", str(reference), "diff", "--exit-code", REVISION, "--", "src/merlin"],
        check=True,
        capture_output=True,
    )
    sys.dont_write_bytecode = True
    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.planning import apply_order, blocks_in
    from merlin.verify.lean_state_effects import (
        EffectOperation,
        EffectScheduleContract,
        derive_effect_schedule,
        effect_semantic_identity,
    )

    for name, module in tuple(sys.modules.items()):
        if name == "merlin" or name.startswith("merlin."):
            path = getattr(module, "__file__", None)
            if path is not None and not Path(path).resolve().is_relative_to(reference / "src"):
                raise ValueError(f"producer import escaped selected source: {name}")
    if contract_path is None:
        contract = EffectScheduleContract(
            standard_memory_sequential=True, standard_allocations_successful=True, index_bits=64
        )
    else:
        record = json.loads(Path(contract_path).read_text())
        operations = tuple(
            EffectOperation(
                row["name"],
                tuple(row["operand_types"]),
                tuple(row["result_types"]),
                tuple(row["reads"]),
                tuple(row["writes"]),
                row["semantic_sha256"],
                row["deterministic_total"],
                row["complete_read_write_frames"],
                tuple(row.get("attribute_names", ())),
                row.get("observable", False),
                row["sequential_atomic_complete"],
            )
            for row in record.pop("operations")
        )
        for key in ("disjoint_resources", "isa_records"):
            if key in record:
                record[key] = (
                    tuple(tuple(pair) for pair in record[key]) if key == "disjoint_resources" else tuple(record[key])
                )
        contract = EffectScheduleContract(operations=operations, **record)
    contract.validate()
    source = Path(source_path).read_text()
    module = parse_ir(source)
    functions = [op for op in module.body.block.ops if getattr(getattr(op, "sym_name", None), "data", None) == "kernel"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("example needs the exact single-block kernel entry")
    block = blocks_in(module).index(functions[0].body.block)
    producer = "supplied-actual-target"
    if order is not None:
        apply_order(module, block, tuple(order))
        target = ir_text(module)
        producer = "merlin.transforms.planning.apply_order"
    else:
        target = Path(target_path).read_text()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "before.mlir").write_text(source)
    (destination / "after.mlir").write_text(target)
    from dataclasses import asdict

    report = {
        "schema": "merlin.graph-state.producer-intake.v1",
        "producer": producer,
        "producer_revision": REVISION,
        "public_acceptance_issued": False,
        "projection_trust": "pinned original Python actual whole-IR correspondence and effect derivation",
        "source_contract_approved": False,
        "native_fidelity_proved": False,
        "contract": asdict(contract),
        "contract_sha256": contract.identity(),
    }
    try:
        projection = derive_effect_schedule(source, target, entry="kernel", contract=contract)
        report.update(
            {
                "projection_checked": True,
                "projection": projection.record(),
                "projection_sha256": projection.identity(),
                "effect_semantics_sha256": effect_semantic_identity(),
            }
        )
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        report.update({"projection_checked": False, "reason": {"type": type(error).__name__, "detail": str(error)}})
    report["producer_sources"] = sorted(
        [
            file_ref(module.__file__)
            for name, module in tuple(sys.modules.items())
            if (name == "merlin" or name.startswith("merlin.")) and getattr(module, "__file__", None)
        ],
        key=lambda row: row["path"],
    )
    report["owned_generator"] = file_ref(__file__)
    report["external_runtime"] = external_runtime()
    report["inputs"] = [file_ref(destination / "before.mlir"), file_ref(destination / "after.mlir")]
    (destination / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--order", type=int, nargs="+")
    args = parser.parse_args()
    if args.order is None and args.target is None:
        parser.error("actual target or actual production order required")
    try:
        report = capture(
            args.reference, args.source, args.target, args.destination, order=args.order, contract_path=args.contract
        )
    except (ValueError, TypeError, KeyError, IndexError) as error:
        # Admission refusals are distinct from an unavailable import or tool.
        args.destination.mkdir(parents=True, exist_ok=True)
        inputs = []
        for role, path in (("before", args.source), ("after", args.target)):
            if path is not None:
                target = args.destination / f"{role}.mlir"
                target.write_bytes(path.read_bytes())
                inputs.append(file_ref(target))
        report = {
            "schema": "merlin.graph-state.producer-intake.v1",
            "producer": "refused-actual-projection-input",
            "producer_revision": REVISION,
            "projection_checked": False,
            "public_acceptance_issued": False,
            "failure_phase": "input_or_semantic_projection_refused",
            "reason": {"type": type(error).__name__, "detail": str(error)},
            "projection_trust": "pinned original Python projection",
            "contract_sha256": None,
            "inputs": inputs,
            "producer_sources": [],
            "owned_generator": file_ref(__file__),
        }
        (args.destination / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"projection_checked": report["projection_checked"], "producer": report["producer"]}))


if __name__ == "__main__":
    main()
