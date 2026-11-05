"""Pure controls for the fixed public recomputation input boundary."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from recomputation_execution_inputs import materialize, original_case

pytestmark = pytest.mark.target("gemmini")


def test_original_and_candidate_materializations_bind_actual_distinct_sources(tmp_path):
    case = original_case()
    records = []
    for arm in ("baseline", "candidate"):
        bound, arrays, expected = materialize(
            tmp_path / arm, case=case, arm=arm, epoch=0, state=case.reference.initial_state()
        )
        assert len(bound.verify(bound.root)) == 8
        source = (bound.root / "model.mlir").read_text()
        identity = json.loads(bound.source_identity_bytes)
        assert identity["content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        assert identity["arm"] == arm and not (bound.root / "capture_receipt.json").exists()
        assert len(expected) == 9 and sum(value.size for value in expected) == 106
        assert [value.tobytes() for value in expected[3:]] == [value.tobytes() for value in arrays]
        with np.load(bound.root / "goldens.npz", allow_pickle=False) as goldens:
            assert goldens.files == [row[0] for row in case.reference.OUTPUTS]
            assert all(
                goldens[name].tobytes() == value.tobytes() for name, value in zip(goldens.files, expected, strict=True)
            )
        records.append((source, [value.tobytes() for value in arrays], [value.tobytes() for value in expected]))
    assert records[0][0] != records[1][0]
    assert records[0][1:] == records[1][1:]
    case.revalidate()


def test_unknown_arm_or_mutated_fixed_source_cannot_be_materialized(tmp_path):
    case = original_case()
    for changed, arm in ((case, "other"), (replace(case, source=case.source + "\n"), "candidate")):
        with pytest.raises(AssertionError):
            materialize(tmp_path / arm, case=changed, arm=arm, epoch=0, state=case.reference.initial_state())
        assert not (tmp_path / arm).exists()


def test_reference_code_change_refuses_before_materialization(tmp_path):
    case = original_case()
    function, old = case.reference.reference, case.reference.reference.__code__
    function.__code__ = (lambda *args: ()).__code__
    try:
        with pytest.raises(AssertionError):
            materialize(tmp_path / "changed", case=case, arm="candidate", epoch=0, state=case.reference.initial_state())
        assert not (tmp_path / "changed").exists()
    finally:
        function.__code__ = old
    case.revalidate()


@pytest.mark.parametrize("field", ["INPUTS", "OUTPUTS", "signed"])
def test_reference_global_or_abi_replacement_refuses_before_materialization(tmp_path, monkeypatch, field):
    case = original_case()
    changed = (lambda value, width: 0) if field == "signed" else ()
    monkeypatch.setitem(case.reference.reference.__globals__, field, changed)
    with pytest.raises(AssertionError):
        materialize(tmp_path / "changed", case=case, arm="candidate", epoch=0, state=case.reference.initial_state())
    assert not (tmp_path / "changed").exists()


def test_missing_complete_cost_retains_the_original_source():
    from merlin.llvmlower.source_recomputation import SourceRecomputationPreparation, SourceRecomputationSelection

    case = original_case()

    def verify_current(**current):
        assert current["comparison"] is None
        assert current["source"] == case.source

    prepared = SourceRecomputationPreparation(
        case.source, SourceRecomputationSelection(case.plan, verify_current=verify_current), settings={}
    )
    assert prepared.selected_source == case.source
    assert not prepared.report["selected"] and prepared.report["status"] == "MISSING_COST"
    prepared.revalidate()
