"""Disclosed vacuous qualification: opt-in per entry, additive, default refusal unchanged."""

from types import SimpleNamespace

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import group_capsule_entries as G
from merlin_experiments.phase0 import model_qualification as MQ
from merlin_experiments.phase0.requirements import NONFINITE_CONTRACT_LABELS

OPT = {MQ.VACUOUS_OPT_IN: MQ.VACUOUS_SELECTION}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    (tmp_path / "capsule.interface.mlir").write_text("module {}\n")
    monkeypatch.setattr(mq, "parse", lambda path: object())
    box = {"value": {"entries": [], "accelerator_groups": 0, "unstated": {}}}
    monkeypatch.setattr(G, "entries", lambda *a, **k: box["value"])
    monkeypatch.setattr(MQ, "_weight_args", lambda *a: [])
    return tmp_path, box


def run(tmp, entry):
    return MQ.qualify({"name": "m", **entry}, {}, tmp, SimpleNamespace(target="t"), tmp / "out")


def test_default_refuses_a_model_without_device_group(stub):
    tmp, _ = stub
    with pytest.raises(ValueError, match=MQ.REFUSAL):
        run(tmp, {"numerical_semantics": {"internal_arithmetic": {}}})


def test_opt_in_records_a_disclosed_vacuous_qualification(stub):
    tmp, _ = stub
    record = run(tmp, OPT)
    assert record["status"] == "qualified" and record["qualification"] == "vacuous"
    assert record["device_groups"] == [] and record["group_capsules"] == []
    assert "no device group; qualification vacuous" in record["scope"]


@pytest.mark.parametrize("value", ["", "vacuous", None, True])
def test_only_the_exact_selection_opts_in(stub, value):
    tmp, _ = stub
    with pytest.raises(ValueError):
        run(tmp, {MQ.VACUOUS_OPT_IN: value})


def test_opt_in_never_hides_unstated_or_present_groups(stub):
    tmp, box = stub
    box["value"] = {"entries": [], "accelerator_groups": 2, "unstated": {}}
    with pytest.raises(ValueError):
        run(tmp, OPT)
    box["value"] = {"entries": [], "accelerator_groups": 0, "unstated": {"x": 1}}
    with pytest.raises(ValueError):
        run(tmp, OPT)


def test_only_the_four_coverage_labels_opt_in():
    assert len(NONFINITE_CONTRACT_LABELS) == 4
