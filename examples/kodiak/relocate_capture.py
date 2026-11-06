"""Materialize a saved capture at its final path without losing source provenance.

Model2MLIR records an absolute ``prov.weights_file`` in ``model.mlir``. A copied
capture can verify against its receipt yet still name the original machine's
weight path. This command verifies the original bytes, copies every receipt
artifact into a fresh destination, retargets only that module attribute, and
issues a derived receipt. It never claims source closure or numerical parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from merlin.capture.rewrite import retarget_weights_file
from merlin.common import mlir_query as query
from merlin.targetgen.application_inventory import exact_int_mm_generic_operation, verify_capture_receipt


def _digest(path: Path) -> dict[str, int | str]:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def _integer_roots(path: Path) -> tuple[str | None, list[tuple[str, str, str]]]:
    module = query.parse(path)
    scheme = query.attr_str(module, "prov.quantization")
    roots = []
    for op in query.walk(module, "linalg.generic"):
        if query.attr_str(op, "prov.aten") != "aten._int_mm.default":
            continue
        if not exact_int_mm_generic_operation(op):
            raise ValueError(f"malformed signed i8×i8→i32 _int_mm root in {path}")
        roots.append(tuple(str(value.type) for value in (*op.operands[:2], *op.results)))
    return scheme, roots


def relocate(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() or not (source / "model.mlir").is_file():
        raise ValueError(f"saved capture is absent: {source}")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination must be fresh: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("destination must be outside the source capture")

    original_check = verify_capture_receipt(source / "model.mlir")
    if original_check.get("status") != "verified_materialized":
        raise ValueError(f"source capture receipt failed: {original_check.get('errors')}")
    source_receipt_path = source / "capture_receipt.json"
    original_receipt = json.loads(source_receipt_path.read_bytes())
    artifact_records = original_receipt["artifacts"]
    original_mlir = source / "model.mlir"
    original_text = original_mlir.read_text(encoding="utf-8")
    source_scheme, source_roots = _integer_roots(original_mlir)
    source_module = query.parse(original_mlir)
    old_weight_path = query.attr_str(source_module, "prov.weights_file")
    if not old_weight_path or Path(old_weight_path).name != "weights.safetensors":
        raise ValueError("capture must name its weights.safetensors in prov.weights_file")
    old_attr = f'prov.weights_file = "{old_weight_path}"'
    if original_text.count(old_attr) != 1:
        raise ValueError("capture has an ambiguous weight-path attribute")

    destination_weights = destination / "weights.safetensors"
    updated_text, changed = retarget_weights_file(original_text, destination_weights)
    new_attr = f'prov.weights_file = "{destination_weights}"'
    exact_rewrite = original_text.replace(old_attr, new_attr, 1)
    # The shared helper always appends a final newline. Preserve the source's
    # original newline policy so the module attribute is the only changed byte.
    helper_rewrite = exact_rewrite if original_text.endswith("\n") else exact_rewrite + "\n"
    if not changed or updated_text != helper_rewrite:
        raise ValueError("retargeting changed more than the weight-path attribute")
    updated_text = exact_rewrite
    if old_weight_path in updated_text or updated_text.count(new_attr) != 1:
        raise ValueError("relocated model retains its original weight path")

    destination.mkdir(parents=True)
    for name, expected in sorted(artifact_records.items()):
        original = source / name
        copied = destination / name
        shutil.copyfile(original, copied)
        if _digest(copied) != expected:
            raise ValueError(f"copied artifact differs from the verified source: {name}")
    shutil.copyfile(original_mlir, destination / "model.source.mlir")
    shutil.copyfile(source_receipt_path, destination / "capture_receipt.source.json")
    (destination / "model.mlir").write_text(updated_text, encoding="utf-8")

    derived_receipt = dict(original_receipt)
    derived_receipt["source_closure_verified"] = False
    derived_receipt["artifacts"] = dict(artifact_records)
    for name in ("model.mlir", "model.source.mlir", "capture_receipt.source.json"):
        derived_receipt["artifacts"][name] = _digest(destination / name)
    (destination / "capture_receipt.json").write_text(
        json.dumps(derived_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    derived_check = verify_capture_receipt(destination / "model.mlir")
    if derived_check.get("status") != "verified_materialized":
        raise ValueError(f"derived capture receipt failed: {derived_check.get('errors')}")
    destination_scheme, destination_roots = _integer_roots(destination / "model.mlir")
    if (destination_scheme, destination_roots) != (source_scheme, source_roots):
        raise ValueError("relocation changed the typed integer contraction inventory")
    if _digest(destination_weights) != artifact_records["weights.safetensors"]:
        raise ValueError("destination weight file differs from source receipt")

    report = {
        "schema": "merlin.capture-relocation.v1",
        "status": "verified_materialized",
        "source_closure_verified": False,
        "source_receipt": _digest(destination / "capture_receipt.source.json"),
        "source_mlir": _digest(destination / "model.source.mlir"),
        "destination_receipt": _digest(destination / "capture_receipt.json"),
        "destination_mlir": _digest(destination / "model.mlir"),
        "weights": _digest(destination_weights),
        "source_scheme": source_scheme,
        "integer_matmuls": len(source_roots),
        "only_rewrite": "prov.weights_file",
        "original_weight_path_still_in_model": False,
    }
    (destination / "relocation_receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="untouched, extracted capture with its original receipt")
    parser.add_argument("destination", type=Path, help="fresh final bundle directory")
    args = parser.parse_args(argv)
    print(json.dumps(relocate(args.source, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
