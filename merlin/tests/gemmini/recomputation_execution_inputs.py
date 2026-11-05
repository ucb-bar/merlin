"""Truthful fixed public recomputation calibration inputs, never a capture grant."""

from __future__ import annotations

import hashlib
import json
import runpy
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import recomputation_host_work as W
from packing_execution_inputs import pin, recheck_artifacts

from merlin.common.paths import repo_root
from merlin.compile.materialized_model_inputs import MaterializedModelInputs
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_digest, ir_text

SOURCE_SHA = "be3df891d1c03c7f48702a9ef30c699d65acce915aa10e5ad81d8589bc7eb54b"


def _types(rows):
    return tuple(f"tensor<{'x'.join(map(str, shape))}x{dtype}>" for _, shape, dtype in rows)


@dataclass(frozen=True)
class PublicRecomputationCase:
    source: str
    candidate: str
    reference: object
    plan: object
    source_pins: tuple[dict, ...]
    callbacks: tuple
    plan_check: object
    plan_code: object
    reference_abi: tuple

    def revalidate(self):
        assert hashlib.sha256(self.source.encode()).hexdigest() == SOURCE_SHA
        recheck_artifacts(self.source_pins)
        for name, function, code in self.callbacks:
            assert getattr(self.reference, name) is function and function.__code__ is code
            assert function.__globals__[name] is function
            assert (function.__globals__["INPUTS"], function.__globals__["OUTPUTS"]) == self.reference_abi
        assert (self.reference.INPUTS, self.reference.OUTPUTS) == self.reference_abi
        assert self.plan.revalidate.__func__ is self.plan_check.__func__
        assert self.plan_check.__func__.__code__ is self.plan_code
        self.plan_check()
        assert ir_text(self.plan.candidate) == self.candidate

    def source_for(self, arm):
        self.revalidate()
        assert arm in ("baseline", "candidate"), "unknown calibration arm"
        return self.source if arm == "baseline" else self.candidate


def original_case():
    """Bind the fixed original source before any runtime/provider callbacks."""
    root = repo_root()
    paths = (
        root / "merlin/tests/fixtures/recomputation_connected.mlir",
        root / "merlin/tests/ir/test_recomputation_connected.py",
        root / "merlin/tests/gemmini/recomputation_execution_inputs.py",
        root / "merlin/tests/gemmini/test_public_recomputation_execution.py",
        root / "merlin/tests/gemmini/packing_execution_inputs.py",
        root / "merlin/tests/gemmini/test_public_packing_execution.py",
        root / "merlin/tests/gemmini/recomputation_host_work.py",
        root / "merlin/tests/gemmini/test_recomputation_host_work.py",
    )
    pins = tuple(pin(path) for path in paths)
    assert pins[0]["sha256"] == SOURCE_SHA
    reference = SimpleNamespace(**runpy.run_path(str(paths[1])))
    source = paths[0].read_text()
    plan = reference.checked_plan()
    module = parse_mlir_text(source)
    module.verify()
    assert ir_digest(module) == plan.stamp.source_sha256
    entry = module.body.block.first_op
    assert len(tuple(module.body.block.ops)) == 1 and entry.sym_name.data == "forward"
    assert tuple(map(str, entry.function_type.inputs)) == _types(reference.INPUTS)
    assert tuple(map(str, entry.function_type.outputs)) == _types(reference.OUTPUTS)
    assert len(reference.INPUTS) == 6 and len(reference.OUTPUTS) == 9
    pins += tuple(pin(path) for path in plan.implementation_paths())
    pins += tuple(pin(path) for path in W.implementation_paths())
    check = plan.revalidate
    result = PublicRecomputationCase(
        source,
        ir_text(plan.candidate),
        reference,
        plan,
        pins,
        tuple(
            (name, getattr(reference, name), getattr(reference, name).__code__)
            for name in ("inputs", "reference", "initial_state", "checked_plan", "signed", "source_path")
        ),
        check,
        check.__func__.__code__,
        (reference.INPUTS, reference.OUTPUTS),
    )
    result.revalidate()
    return result


def materialize(destination, *, case, arm, epoch, state):
    """Validate first, then bind actual original/candidate bytes and complete ABI."""
    assert type(case) is PublicRecomputationCase
    case.revalidate()
    assert type(epoch) is int and 0 <= epoch < 3
    assert type(state) is np.ndarray and state.dtype == np.int32 and state.shape == (2, 5)
    source = case.source_for(arm)
    module = parse_mlir_text(source)
    module.verify()
    assert ir_digest(module) == ir_digest(case.plan.source if arm == "baseline" else case.plan.candidate)
    assert tuple(map(str, module.body.block.first_op.function_type.inputs)) == _types(case.reference.INPUTS)
    assert tuple(map(str, module.body.block.first_op.function_type.outputs)) == _types(case.reference.OUTPUTS)
    arrays = case.reference.inputs(epoch, state.copy())
    expected = case.reference.reference(*[value.copy() for value in arrays])
    for values, rows in ((arrays, case.reference.INPUTS), (expected, case.reference.OUTPUTS)):
        assert len(values) == len(rows)
        assert all(
            value.shape == shape and value.dtype == np.dtype("int" + dtype[1:])
            for value, (_, shape, dtype) in zip(values, rows, strict=True)
        )
    case.revalidate()
    destination.mkdir()
    (destination / "model.mlir").write_text(source)
    (destination / "weights.safetensors").write_bytes(struct.pack("<Q", 8) + b"{}      ")
    names = tuple(name for name, _, _ in case.reference.INPUTS)
    (destination / "weights.safetensors.manifest.json").write_text(
        json.dumps({str(i): {"kind": "input", "name": name} for i, name in enumerate(names)})
    )
    np.savez(destination / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
    (destination / "input_order.json").write_text(json.dumps(dict(zip(names, range(len(names)), strict=True))))
    np.save(destination / "golden.npy", expected[0])
    outputs = tuple(name for name, _, _ in case.reference.OUTPUTS)
    np.savez(destination / "goldens.npz", **dict(zip(outputs, expected, strict=True)))
    (destination / "output_order.json").write_text(json.dumps(outputs))
    artifacts = {
        path.name: {"sha256": pin(path)["sha256"], "size_bytes": path.stat().st_size} for path in destination.iterdir()
    }
    case.revalidate()
    bound = MaterializedModelInputs.bind(
        destination,
        artifacts=artifacts,
        validator_identity={
            "worker_sha256": pin(Path(__file__).resolve())["sha256"],
            "source_and_reference": case.source_pins,
            "scope": "fixed public source/ABI and independent ordered integer reference; calibration only",
        },
        source_identity={
            "content_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "original_content_sha256": SOURCE_SHA,
            "selected_ir_sha256": ir_digest(module),
            "arm": arm,
            "plan": case.plan.record(),
            "artifacts": case.source_pins,
            "role": "explicit checked-source calibration; no capture, cost or target qualification",
        },
    )
    assert not (destination / "capture_receipt.json").exists()
    bound.verify(destination)
    case.revalidate()
    return bound, arrays, expected
