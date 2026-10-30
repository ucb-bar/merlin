"""Actual compiler qualification for default-all memory composition."""

import ctypes
import json
from pathlib import Path

import pytest
from buffer_rank_one_inputs import source
from test_compiler_formal_transforms import shared_library

from merlin.llvmlower import lower
from merlin.verify import compiler_transforms as transforms
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import SpeculativeGate


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


class Returned(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int32), ("dimension", ctypes.c_int64)]


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("tile_size,extent", [(None, 3), (32, 257)], ids=["untiled3", "tiled32-tail257"])
def test_actual_default_all_memory_gate_keeps_DCE_proof_and_observed_CSE_identity(
    tmp_path,
    pass_gate_tools,
    tile_size,
    extent,
):
    from merlin.verify import compiler_memory_dce, compiler_memory_origin

    raw = source(extent)
    options = transforms.Options(buffer_allocator=transforms.ALLOCATOR, tile_size=tile_size)
    original = lower.lower_model(
        raw, tmp_path / "original", targets=(), formal_transforms=transforms.Options(memory_source=True)
    )
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=1800)
    selected = pass_validation.PassValidation(worker, pass_gate_tools)  # All registered checks, no subset rewrite.
    assert selected.passes == pass_validation.DEFAULT_CHECKS
    checked = lower.lower_model(
        raw, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 2 and all(row["verdict"] == "accepted" for row in worker.verdicts)
    coverage = json.loads((worker.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is True
    assert coverage["verified_boundaries"] == 2 and coverage["unverified_boundaries"] == 1
    excluded = [row for row in coverage["boundaries"] if row["status"] == "unverified"]
    assert len(excluded) == 1 and excluded[0]["pass"] == "lower_quant_ext"
    assert excluded[0]["source_sha256"] == excluded[0]["final_sha256"] and excluded[0]["proof_credit"] is False
    observed_cse = [row for row in coverage["boundaries"] if row["pass"] == "cse"]
    assert observed_cse and all(row["status"] == "unchanged" and row["proof_credit"] is False for row in observed_cse)
    verified = [row for row in coverage["boundaries"] if row["proof_credit"]]
    assert {row["check"] for row in verified} == {"compiler_buffer_refinement", compiler_memory_dce.CHECK}
    labels = {row["label"] for row in worker.verdicts}
    assert "prune_dead_pure_tensor_ops" in labels
    assert not any(row["label"] == "cse" for row in worker.verdicts)  # No proof/optimization credit for identity.
    dce = next(row["detail"] for row in worker.verdicts if row["label"] == "prune_dead_pure_tensor_ops")
    assert dce["schema"] == "merlin.compiler-memory-dce.gate.v1"
    assert dce["domain"] == compiler_memory_dce.DOMAIN
    assert dce["theorems"] == list(compiler_memory_dce.ROOTS)
    assert dce["changed_sites"] == 1 and dce["ordinary_same_mode"] is True
    assert dce["normalization_mode_used"] is False and dce["allocator_policy_required"] is False
    report = json.loads(Path(dce["receipt"]).read_text())
    assert report["audit"]["violations"] == []
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Quot.sound", "Classical.choice"}
    captures = json.loads((tmp_path / "checked/leading-cse/index.json").read_text())
    assert captures and all(row["schema"] == "merlin.original-memory-cse.v2" for row in captures)
    for number, capture in enumerate(captures, 1):
        before = (tmp_path / "checked/leading-cse" / f"{number:04d}.before.mlir").read_bytes()
        after = (tmp_path / "checked/leading-cse" / f"{number:04d}.after.mlir").read_bytes()
        assert before == after and capture["before_sha256"] == capture["after_sha256"]
        assert "checked" not in capture
    assert compiler_memory_origin.current() is None
    observations = []
    for arm, result in (("original", original), ("baseline", baseline), ("checked", checked)):
        library = shared_library(result, tmp_path / arm)
        function = library._mlir_ciface_kernel
        function.argtypes = [ctypes.POINTER(Returned), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(Descriptor)]
        function.restype = None
        for first, second, initial in ((7, -3, 11), (0, 0, 5), (2**31 - 4, 17, 2**31 - 1), (-(2**31), -23, -9)):
            storage = (ctypes.c_int32 * 1)(initial)
            pointer = ctypes.cast(storage, ctypes.c_void_p)
            descriptor = Descriptor(pointer, pointer, 0, 1, 1)
            returned = Returned()
            function(ctypes.byref(returned), first, second, ctypes.byref(descriptor))
            middle = ctypes.c_int32(initial + first).value
            final = ctypes.c_int32(middle + second).value
            actual = [returned.value, returned.dimension, storage[0]]
            assert actual == [ctypes.c_int32(middle + final).value, extent, final]
            observations.append({"arm": arm, "inputs": [first, second, initial], "observed": actual})
        library._mlir_ciface_untouched.argtypes = [ctypes.c_int32]
        library._mlir_ciface_untouched.restype = ctypes.c_int32
        assert library._mlir_ciface_untouched(-(2**31)) == -(2**31)
    (tmp_path / "NATIVE-OBSERVATIONS.json").write_text(json.dumps(observations, indent=2) + "\n")
