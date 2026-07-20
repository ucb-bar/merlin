"""Board-free tests for the TRUE ExecuTorch whole-model COLUMN of the autonomous beam experiment.

Exercises merlin.compare.executorch_column (ingest + honest labeling) and the dtype-appropriate
reference logic in build_tools/scripts/run_autonomous_beam_experiment.py — the int8-reference-bug fix.
Named with "experiment"/"compare" so the pre-commit -k filter selects them. No board, no ET venv:
fixtures are hand-written BaselineResult JSONs (pass / not_run / int8 variant-mismatch).
"""
from __future__ import annotations

import importlib.util

from merlin.baselines.contract import BaselineResult
from merlin.common.paths import repo_root
from merlin.compare.executorch_column import (EXECUTORCH_LABEL, XNNPACK_KERNELS_LABEL,
                                              dtype_comparability, executorch_cell, gate_basis)


def _write_result(root, model, variant, *, built, ran, cos=None, rel=None, wall_ns=None,
                  cos_threshold=0.9999, rel_threshold=2e-3, gap_reason="", ts="20260101T000000Z"):
    """Write one baseline_result.json into the measurements tree the ingester reads."""
    res = BaselineResult(framework="executorch", model=model, variant=variant,
                         substrate="k1_spacemit", built=built, ran=ran, cos=cos, rel=rel,
                         e2e_wall_ns=wall_ns, cos_threshold=cos_threshold,
                         rel_threshold=rel_threshold, gap_reason=gap_reason, timestamp=ts)
    d = root / "out/artifacts/measurements/k1_spacemit" / model / f"{model}_{variant}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    res.write(d)
    return res


def _load_experiment_module():
    """Import the (non-package) experiment driver by path so we can test its reference logic."""
    p = repo_root() / "build_tools" / "scripts" / "run_autonomous_beam_experiment.py"
    spec = importlib.util.spec_from_file_location("_autobeam_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- executorch_cell ingest/labeling
def test_experiment_executorch_column_passing_cell_is_measured(tmp_path):
    # bitvla is random-init -> gate_basis must say lowering-exactness, not semantic.
    _write_result(tmp_path, "bitvla", "fp32", built=True, ran=True, cos=0.99999999,
                  rel=1.6e-6, wall_ns=146_000_000)
    cell = executorch_cell("bitvla", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "measured"
    assert cell["executorch_wall_ns"] == 146_000_000.0
    assert cell["variant"] == "fp32"
    assert "lowering-exactness" in cell["gate_basis"]
    assert cell["label"] == EXECUTORCH_LABEL


def test_experiment_executorch_column_semantic_gate_label(tmp_path):
    # rdt2's captured golden IS reproducible -> semantic gate label (not lowering-exactness).
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.99999,
                  rel=1e-4, wall_ns=855_000_000)
    cell = executorch_cell("rdt2", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "measured"
    assert cell["gate_basis"].startswith("semantic")
    assert "lowering-exactness" not in cell["gate_basis"]


def test_experiment_executorch_column_not_run_is_not_measured(tmp_path):
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=False,
                  gap_reason="K1 board unavailable (MERLIN_K1_HOST unset)")
    cell = executorch_cell("rdt2", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert "not_run" in cell["reason"] and "board unavailable" in cell["reason"]


def test_experiment_executorch_column_fail_never_reported_as_number(tmp_path):
    # An executed-but-failed run (missed tolerance) is a gap, NEVER a wall number.
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.5, rel=0.9,
                  wall_ns=999_000_000, gap_reason="")
    cell = executorch_cell("rdt2", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert "fail" in cell["reason"] and "cos=0.5" in cell["reason"]


def test_experiment_executorch_column_int8_never_borrows_fp32_number(tmp_path):
    # A passing fp32 exists but int8 was NOT run -> the int8 cell must be not_measured, never the
    # fp32 wall (variant-mismatch honesty).
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.99999, rel=1e-4,
                  wall_ns=855_000_000)
    cell = executorch_cell("rdt2", "int8", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert cell["variant"] == "int8"
    assert "int8" in cell["reason"]


def test_experiment_executorch_column_absent_tree_is_not_measured(tmp_path):
    cell = executorch_cell("bitvla", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None


def test_experiment_executorch_column_ram_infeasible_reason(tmp_path):
    # openvla has no ET result AND is RAM-infeasible -> reason must say so honestly.
    cell = executorch_cell("openvla", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert "RAM-infeasible" in cell["reason"]


def test_experiment_gate_basis_matches_random_init_set():
    assert "lowering-exactness" in gate_basis("bitvla")   # random-init
    assert gate_basis("rdt2").startswith("semantic")      # reproducible golden


# ---------------------------------------------------------------- int8 reference-bug fix
def test_experiment_reference_int8_uses_executorch_never_fp32_xnnpack(tmp_path):
    mod = _load_experiment_module()
    # A passing ExecuTorch int8 result exists -> it is the int8 reference (int8-vs-int8).
    _write_result(tmp_path, "rdt2", "int8", built=True, ran=True, cos=0.995, rel=1e-2,
                  wall_ns=700_000_000, cos_threshold=0.99, rel_threshold=5e-2)
    ref = mod._reference("rdt2", "int8", tmp_path)
    assert ref["kind"] == "executorch_external"
    assert ref["wall_ns"] == 700_000_000.0
    assert ref["dtype"] == "int8"
    assert ref["label"] == EXECUTORCH_LABEL


def test_experiment_reference_int8_without_et_is_none_never_fp32(tmp_path):
    mod = _load_experiment_module()
    # Only a passing fp32 ET result exists -> the int8 reference must be None, NEVER the fp32 wall.
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.99999, rel=1e-4,
                  wall_ns=855_000_000)
    ref = mod._reference("rdt2", "int8", tmp_path)
    assert ref["kind"] is None
    assert ref["wall_ns"] is None
    assert "NEVER" in ref["note"] and "fp32 XNNPACK" in ref["note"]


def test_experiment_reference_fp32_is_xnnpack_in_runtime_label(tmp_path):
    mod = _load_experiment_module()
    ref = mod._reference("rdt2", "fp32", tmp_path)   # wall_ns may be None if no local four-way cache
    assert ref["kind"] == "xnnpack_kernels_in_runtime"
    assert ref["label"] == XNNPACK_KERNELS_LABEL
    assert ref["dtype"] == "fp32"


def test_experiment_comparability_labels_are_distinct():
    # The two arms MUST NOT read as the same thing.
    assert "OUR runtime" in XNNPACK_KERNELS_LABEL
    assert "TRUE external" in EXECUTORCH_LABEL
    assert XNNPACK_KERNELS_LABEL != EXECUTORCH_LABEL


def test_dtype_comparability_states_the_per_dtype_caveats():
    """Same storage dtype is necessary but NOT sufficient — the contract must SAY what differs."""
    fp32, int8, fp16 = (dtype_comparability(d) for d in ("fp32", "int8", "fp16"))
    # fp32 is the only like-for-like row (storage + accumulate both f32).
    assert "like-for-like" in fp32 and "f32" in fp32
    # int8: no in-runtime kernel-swap arm; external ref is ExecuTorch whole-system.
    assert "NO in-runtime" in int8 and "ExecuTorch" in int8
    # fp16: same storage, DIFFERENT accumulate — must be flagged, never presented as a clean match.
    assert "ACCUMULATE-ASYMMETRIC" in fp16 and "vfwmacc" in fp16
    assert "not a like-for-like" in fp16 and "caveated" in fp16   # explicitly negated, never clean
    # an unknown dtype fails loud rather than implying a match.
    assert "UNKNOWN" in dtype_comparability("fp8")
    # every measured/not-measured cell carries the field.
    for st in (executorch_cell("bitvla", "int8"), executorch_cell("bitvla", "fp16")):
        assert "dtype_comparability" in st
