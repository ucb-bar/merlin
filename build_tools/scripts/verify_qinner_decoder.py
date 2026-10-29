#!/usr/bin/env python3
"""Retain audited decoder proofs, specialized Qinner receipts and archive/native bit observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.qinner import actual, audit, check_boundary, replay, write
from merlin.verify.lean.qinner_decoder import decode_records, replay_decode


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument("--stage", nargs=3, action="append", default=[], metavar=("LABEL", "PROPAGATION", "LIFT"))
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    sys.path.insert(0, str(repo_root() / "merlin/tests/ir"))
    import numpy as np
    from lean_qinner_cases import fixture
    from lean_qinner_decoder_cases import KINDS, array_from_words, native_compare, patterns

    base = (options.output or artifacts_dir() / "verification/merlin-passes/frontend/qinner-decoder").resolve()
    run_id = uuid.uuid4().hex
    run = base / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    proof_audit = audit()
    write(run / "proof-audit.json", proof_audit)
    cases = []
    for label, propagation, lift in options.stage:
        cases.append((label, Path(propagation) / "before.mlir", Path(lift) / "after.mlir", None))
    for kind, shape, nested in [*((kind, (2, 8), kind in {"f32", "i1"}) for kind in KINDS), ("f32", (0, 3), True)]:
        label = f"{kind}-{'empty' if 0 in shape else 'bits'}"
        directory = run / label
        directory.mkdir()
        source = directory / "before.mlir"
        target = directory / "after.mlir"
        source.write_text(fixture(kind, shape, nested=nested))
        target.write_text(actual(source.read_text())["lifted"])
        left, _ = array_from_words(kind, shape)
        right, _ = array_from_words(kind, shape, list(reversed(patterns(kind))))
        extra = directory / "extra.npz"
        np.savez(extra, **{"qinner::left": left, "qinner::right": right})
        cases.append((label, source, target, extra))
    boundaries = []
    for label, source, target, extra in cases:
        print(f"checking {label}", flush=True)
        folder = base / "boundaries" / label
        attempt = folder / "attempts" / run_id
        receipt = check_boundary(source, target, attempt, compiler_python=options.compiler_python, extra_path=extra)
        assert receipt["checked"] and not receipt["approved"], (label, receipt)
        write(folder / "receipt.json", receipt)
        replayed = replay(folder / "receipt.json", attempt / "replay", compiler_python=options.compiler_python)
        assert replayed["fresh"] and replayed["checked"] and replayed["matches_recorded"], (label, replayed)
        boundaries.append(
            {
                "label": label,
                "kind": receipt["boundary_kind"],
                "sites": receipt["covered_sites"],
                "receipt": str(folder / "receipt.json"),
                "original_inputs": [_file_ref(source), _file_ref(target)],
                "replay": replayed,
            }
        )
    native = []
    for kind in KINDS:
        for shape in ((2, 8), (0, 3)):
            label = f"{kind}-{'empty' if 0 in shape else 'bits'}"
            print(f"archive/native {label}", flush=True)
            directory = base / "native" / run_id / label
            observed = native_compare(kind, shape, options.compiler_python, directory)
            replayed = replay_decode(directory / "decoder/receipt.json", directory / "decoder/replay")
            assert all(replayed[key] for key in ("fresh", "checked", "matches_recorded", "archive_bytes_match"))
            native.append(
                {
                    "kind": kind,
                    "shape": shape,
                    "samples": observed["samples"],
                    "all_output_bits_match": observed["all_output_bits_match"],
                    "comparison": str(directory / "native-comparison.json"),
                    "decoder_receipt": str(directory / "decoder/receipt.json"),
                    "replay": replayed,
                }
            )
    # Retain an actual archive that the production resolver admits but the concrete decoder refuses.
    refusal = base / "counterexamples" / run_id
    refusal.mkdir(parents=True)
    invalid = np.array([2, 128, 255, 0, 1], dtype=np.uint8).view(np.bool_)
    np.savez(refusal / "invalid-bool.npz", **{"qinner::values": invalid})
    write(
        refusal / "invalid-bool.json",
        {
            "raw_hex": invalid.tobytes().hex(),
            "reason": "Host i1 decoder accepts exactly byte 00 or 01; production resolve checks dtype/shape only",
            "production_modified": False,
        },
    )
    boolean_type = {"kind": "tensor", "shape": [5], "element": {"kind": "int", "width": 1}, "encoding": None}
    rejected = decode_records(
        [{"key": "values", "type": boolean_type, "expected_type": boolean_type, "hex": invalid.tobytes().hex()}],
        refusal / "decoder",
    )
    assert not rejected["payloads"][0]["checked"]
    summary = {
        "run_id": run_id,
        "approved": False,
        "verdict": "abstracted",
        "boundaries": boundaries,
        "native": native,
        "concrete_decoder": "little_endian_row_major_host_i1_00_01",
        "decoded_binding_typing_proved": True,
        "decoded_binding_initialized_proved": True,
        "exact_decoded_bits_proved": True,
        "archive_authenticity_proved": False,
        "host_implements_decoder_proved": False,
        "actual_runtime_delivery_proved": False,
        "whole_model_definedness_proved": False,
        "numerical_laws_used": [],
    }
    write(run / "summary.json", summary)
    write(base / "summary.json", summary)
    owned = [
        *list((repo_root() / "merlin/lean/MerlinVerify").rglob("Qinner*.lean")),
        repo_root() / "merlin/lean/QINNER_REVIEW.md",
        *[
            repo_root() / name
            for name in (
                "src/merlin/verify/lean/qinner.py",
                "src/merlin/verify/lean/qinner_decoder.py",
                "build_tools/scripts/verify_qinner.py",
                "build_tools/scripts/verify_qinner_decoder.py",
                "merlin/tests/ir/lean_qinner_cases.py",
                "merlin/tests/ir/lean_qinner_decoder_cases.py",
                "merlin/tests/ir/test_lean_qinner.py",
                "merlin/tests/ir/test_lean_qinner_decoder.py",
            )
        ],
    ]
    files = sorted([*owned, *(path for path in base.rglob("*") if path.is_file() and path.name != "manifest.json")])
    write(
        base / "manifest.json",
        [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "boundary_receipts": len(boundaries),
                "decoder_receipts": len(native),
                "native_samples": sum(entry["samples"] for entry in native),
                "approved": False,
            }
        )
    )


if __name__ == "__main__":
    main()
