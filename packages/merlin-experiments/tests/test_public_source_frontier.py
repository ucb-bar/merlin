"""Source-only diagnostics retain pending effects and cannot waive full admission."""

import copy
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import connected_integer_qualification as Q
from merlin_experiments.phase0 import public_programs as PUBLIC
from merlin_experiments.phase0 import public_source_frontier as F


@pytest.mark.parametrize("protocol,target", [("unknown", "source_only"), (F.PROTOCOL, "gemmini")])
def test_frontier_needs_explicit_protocol_and_source_only_binding(tmp_path, protocol, target):
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="source-only diagnostic"):
        F.publish({}, SimpleNamespace(target=target), destination, protocol=protocol, operator_exclusion=None)
    assert not destination.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "single"},
        {"source_role": "pytorch_model_slice"},
        {"label": "private"},
        {"model_qualification": {"status": "qualified"}},
        {"integer_partial_sum_bound": {"status": "proven_safe"}},
    ],
)
def test_frontier_cannot_downgrade_or_relabel_target_qualification(change):
    cap = dict(kind="model", label="dev", source_role="materialized_performance_capture")
    with pytest.raises(ValueError, match="unqualified complete public"):
        F.arithmetic({}, dict(cap, **change), Path("unused"), protocol=F.PROTOCOL)


@pytest.fixture
def original(tmp_path):
    source = os.environ.get("MERLIN_TEST_FRONTIER_CAPSULE")
    if not source:
        pytest.skip("explicit original sealed public frontier fixture not selected")
    directory = tmp_path / "capsule"
    shutil.copytree(source, directory)
    return yaml.safe_load((directory / "capsule.yaml").read_bytes()), directory


def test_actual_sealed_frontier_keeps_complete_source_and_pending_target_scope(original):
    cap, directory = original
    PUBLIC.verify_written(cap, directory)
    record = F.verify_written(cap, directory)
    assert record["target_effects"] == "PENDING" and record["formal_admission"] is False
    assert "model_qualification" not in cap and "integer_partial_sum_bound" not in cap
    assert record["source_proof"]["status"] == "source_arithmetic_proven"
    with pytest.raises(ValueError, match="host-selected realization owner"):
        Q.qualify({"source_integer_proof": record["source_integer_proof"]}, cap, directory)


@pytest.mark.parametrize(
    "fault",
    [
        "effects",
        "qualification",
        "arithmetic",
        "selection",
        "source",
        "reference",
        "frontend_trace_sha256",
        "original_graph_sha256",
        "source_mode",
        "selection_sha256",
        "source_scope",
    ],
)
def test_actual_frontier_reopens_complete_bytes_arithmetic_and_original_scope(original, fault):
    cap, directory = original
    changed = copy.deepcopy(cap)
    if fault == "effects":
        changed["public_source_frontier"]["target_effects"] = "PASS"
    elif fault == "qualification":
        changed["model_qualification"] = {"status": "qualified"}
    elif fault == "selection":
        changed["public_source_frontier"]["source_integer_proof"] = "unknown"
    elif fault == "arithmetic":
        row = next(
            r
            for r in changed["public_source_frontier"]["source_proof"]["operations"]
            if r.get("kind") == "multiply_accumulate"
        )
        row["reduction_extent"] += 1
    elif fault in {"frontend_trace_sha256", "original_graph_sha256", "source_mode", "selection_sha256", "source_scope"}:
        changed["public_source_frontier"]["source_proof"][fault] = "changed"
    else:
        path = directory / ("capsule.interface.mlir" if fault == "source" else "source-capture-golden.json")
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(Exception):
        PUBLIC.verify_written(changed, directory)
