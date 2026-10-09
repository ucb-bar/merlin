"""Ordinary source preparation is consumable but incomplete originals refuse.

Native fixtures observe only independent synthetic RTL and source programs.
No compiler authoring, target runtime qualification or physical jobs execute.
"""

import copy
import importlib.util
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import source_preparation_release as P
from merlin_experiments.phase0.component_generation import digest
from merlin_experiments.phase1 import component_generation_admission as A
from merlin_experiments.phase1.component_origin import FreshPhase1Inputs
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common.jsonio import canonical_json
from merlin.targetgen import golden_store
from merlin.targetgen.target_experiment import load_target_experiment


def _fixtures():
    path = Path(__file__).with_name("test_source_requirement_ledger.py")
    spec = importlib.util.spec_from_file_location("source_preparation_real_generation_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _fixtures()
automatic = F.automatic
independent = F.independent
selected = F.selected
inputs = F.inputs
BUDGET = P.SourcePreparationBudget(10000000)


@pytest.fixture
def prepared(inputs, tmp_path):
    coverage = inputs["root"] / "_evidence/coverage/component-coverage.json"
    return P.prepare(
        root=inputs["root"],
        coverage=coverage,
        hardware=inputs["hardware"],
        software=inputs["software"],
        semantic_cases=None,
        budget=BUDGET,
        forbidden_roots=(),
        destination=tmp_path / "source-preparation",
    )


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_fixed_ordinary_producer_consumer_retains_every_original_and_pending_candidate(prepared, inputs):
    record = prepared.record()
    assert record["schema"] == P.SCHEMA
    assert record["original_required_ids"] == [row["id"] for row in inputs["coverage"]["obligations"]]
    assert record["original_mandatory_ids"] == [
        row["id"] for row in inputs["coverage"]["obligations"] if row["mandatory"]
    ]
    assert record["checked_source_witnesses"] and record["mandatory_source_blockers"]
    assert record["status"] == "source_preparation_incomplete"
    assert len(record["candidate_predicates"]) == len(record["original_required_ids"])
    assert all(row["candidate_verdict"] == "not_evaluated" for row in record["candidate_predicates"])
    for row in record["requirements"]:
        if row["kind"] in {"physical_interaction", "effect_domain", "resource_role"}:
            assert row["missing_source_producers"] and row["source_input_state"] == "unavailable"
    with pytest.raises(P.SourcePreparationRefusal) as refusal:
        prepared.require_complete()
    assert refusal.value.original_ids == tuple(record["mandatory_source_blockers"])
    assert refusal.value.blockers == record["requirements"]
    assert "candidate correctness/runtime/isolation/hardware authority" in record["scope"]


def test_explicit_phase1_source_selection_cannot_bypass_original_blockers(prepared):
    with pytest.raises(StageGateError, match="unresolved original mandatory"):
        A.verify_generation_inputs(
            prepared.root, preparation=prepared, hardware=prepared.hardware, software=prepared.software
        )
    # The unselected legacy path still rejects source_generated itself.
    with pytest.raises(StageGateError, match="source-only preparation is not concrete"):
        A.verify_generation_inputs(
            prepared.root, preparation=None, hardware=prepared.hardware, software=prepared.software
        )


def test_source_selection_does_not_remove_prior_runtime_gate(prepared, monkeypatch, tmp_path):
    values = {field.name: None for field in fields(FreshPhase1Inputs)}
    descriptor = Path(next(pin.path for pin in prepared.hardware.source_pins if pin.role == "target-descriptor"))
    values.update(
        hardware=prepared.hardware,
        software=prepared.software,
        corpus_root=prepared.root,
        source_preparation=prepared,
        target_experiment=load_target_experiment(descriptor, source_root=tmp_path),
    )
    from merlin_experiments.phase1 import component_origin as O

    monkeypatch.setattr(O, "verify_generation_inputs", lambda *_args, **_kwargs: pytest.fail("runtime gate bypassed"))
    with pytest.raises(StageGateError, match="independently issued target runtime support"):
        FreshPhase1Inputs(**values).verify()


@pytest.mark.parametrize("defect", ["missing", "reordered", "state", "cohort", "predicate", "blockers"])
def test_saved_or_resigned_preparation_cannot_drop_original_slots(prepared, defect):
    document = json.loads(prepared.receipt_json)
    if defect == "missing":
        document["requirements"].pop()
    elif defect == "reordered":
        document["original_required_ids"].reverse()
    elif defect == "state":
        document["status"] = "source_inputs_complete"
    elif defect == "cohort":
        document["requirements"][0]["cohort"] = "development"
    elif defect == "predicate":
        document["candidate_predicates"][0]["candidate_verdict"] = "passed"
    else:
        document["mandatory_source_blockers"] = []
    forged = replace(prepared, receipt_json=canonical_json(document))
    with pytest.raises(ValueError, match="actual live"):
        forged.require_complete()
    with pytest.raises(ValueError, match="actual live"):
        copy.copy(prepared).verify()


@pytest.mark.parametrize("defect", ["missing", "cohort", "mandatory", "expectation", "declaration", "selected_source"])
def test_actual_ordinary_coverage_drift_refuses_even_after_resigning(inputs, tmp_path, defect):
    coverage = copy.deepcopy(inputs["coverage"])
    if defect == "missing":
        coverage["obligations"].pop()
    elif defect == "cohort":
        coverage["obligations"][0]["cohort"] = "development"
    elif defect == "mandatory":
        coverage["obligations"][0]["mandatory"] = False
    elif defect == "expectation":
        coverage["obligations"][0]["expectation"] = "compile_only"
    elif defect == "declaration":
        coverage["obligations"][0]["declaration_sha256"] = "f" * 64
    else:
        path = Path(coverage["generation_identity"]["recipe"]["path"])
        path.write_bytes(path.read_bytes() + b"\n")
    coverage["sha256"] = digest({key: value for key, value in coverage.items() if key != "sha256"})
    selected_report = tmp_path / "coverage.json"
    selected_report.write_bytes(canonical_json(coverage))
    with pytest.raises(ValueError):
        P.prepare(
            root=inputs["root"],
            coverage=selected_report,
            hardware=inputs["hardware"],
            software=inputs["software"],
            semantic_cases=None,
            budget=BUDGET,
            forbidden_roots=(),
            destination=tmp_path / "refused",
        )


@pytest.mark.parametrize("product", ["source", "last_reference", "coverage", "private_preparation"])
def test_actual_source_complete_reference_and_selected_product_mutations_refuse(prepared, product):
    if product == "coverage":
        path = prepared.coverage
    elif product == "private_preparation":
        path = prepared.output
    else:
        row = json.loads(prepared.receipt_json)["checked_source_witnesses"][-1]
        if product == "source":
            path = Path(row["source"]["path"])
        else:
            directory = prepared.root / row["member"]
            original = golden_store.load_golden(directory)
            name = sorted(original["outputs"])[-1]
            original["outputs"][name][-1][-1] += 1
            golden_store.write_golden(directory, original)
            with pytest.raises(ValueError):
                prepared.verify()
            return
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        prepared.verify()


@pytest.mark.parametrize("selection", [None, {}, object()])
def test_saved_data_cannot_select_versioned_phase1_admission(tmp_path, selection):
    if selection is None:
        with pytest.raises(StageGateError):
            A.verify_generation_inputs(tmp_path, preparation=None, hardware=None, software=None)
    else:
        with pytest.raises(StageGateError, match="actual live versioned"):
            A.verify_generation_inputs(tmp_path, preparation=selection, hardware=None, software=None)


@pytest.mark.parametrize("wrong", ["root", "hardware", "software"])
def test_source_preparation_must_match_exact_phase1_membership(prepared, wrong, tmp_path):
    arguments = dict(preparation=prepared, hardware=prepared.hardware, software=prepared.software)
    root = tmp_path if wrong == "root" else prepared.root
    if wrong != "root":
        arguments[wrong] = object()
    with pytest.raises(StageGateError, match="exact original corpus/hardware/software"):
        A.verify_generation_inputs(root, **arguments)


def test_source_reader_budget_precedes_json_decoding_and_accepts_no_assumed_limits(tmp_path):
    path = tmp_path / "huge-report.json"
    path.write_bytes(b"{" * 200)
    with pytest.raises(ValueError, match="before decoding"):
        P._coverage(path, budget=P.SourcePreparationBudget(100), forbidden=())
    for limit in (True, 0, -1, 1.0):
        with pytest.raises(ValueError, match="positive integer"):
            P.SourcePreparationBudget(limit).verify()


def test_report_growth_after_stat_keeps_the_selected_read_bound(tmp_path, monkeypatch):
    path = tmp_path / "growing-report.json"
    path.write_bytes(b"{}")
    original_open = Path.open

    def growing_open(selected, mode="r", *args, **kwargs):
        if selected == path and mode == "rb":
            with original_open(selected, "ab") as stream:
                stream.write(b" " * 200)
        return original_open(selected, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="limit"):
        P._coverage(path, budget=P.SourcePreparationBudget(100), forbidden=())
