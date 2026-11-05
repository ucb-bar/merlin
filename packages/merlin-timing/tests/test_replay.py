"""Synthetic replay custody checks; no simulator or qualification verdicts."""

import copy
import json

import pytest
from merlin_timing import archive as A
from merlin_timing import contracts as C
from merlin_timing import replay as R
from test_archive import prepared, reports_for


def inputs(tmp_path):
    plan, _, _, _ = prepared(tmp_path)
    libraries = [tmp_path / "functional.so", tmp_path / "provider.so"]
    for index, path in enumerate(libraries):
        path.write_bytes(f"synthetic library {index}".encode())
    executable = tmp_path / "spike"
    executable.write_bytes(b"synthetic executable")
    model = {
        "schema": "merlin.annotated_spike_model.v1",
        "profile": "phase1-public",
        "domain": {
            key: value for key, value in plan["dataset_template"]["domain"].items() if key != "target_model_sha256"
        },
        "scope": {"kind": "counter_pc_pair"},
        "libraries": [A.file_identity(path) for path in libraries],
        "parameters": {},
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    return plan, model_path, executable, libraries


def test_replay_preserves_labels_reviews_and_missing_denominator(tmp_path):
    plan, model, executable, libraries = inputs(tmp_path)
    plan["observations"][0]["scope_binding"] = None
    original = copy.deepcopy(plan)
    result = R.prepare(plan, model, executable, libraries)
    replay = result["observation_plan"]
    assert plan == original
    assert replay["observations"] == original["observations"]
    assert replay["dataset_template"]["cases"] == original["dataset_template"]["cases"]
    assert replay["dataset_template"]["domain"]["target_model_sha256"] == A.file_identity(model)
    assert result["replayable_records"] == len(plan["observations"]) - 1
    assert result["unavailable_counter_reviews"] == [plan["observations"][0]["record_id"]]
    assert result["prediction_authority"] == "NONE"
    assert result["launches"] == []
    assert not result["launch_authority"]
    with pytest.raises(C.ContractError, match="exact frozen plan"):
        A.attach(
            replay, {"schema": "merlin.phase1.spike_observations.v1", "plan_sha256": C.digest(plan), "reports": {}}
        )


@pytest.mark.parametrize(
    "field", ["device_sha256", "feature_engine_sha256", "memory_regime_sha256", "scope_sha256", "label_engine_sha256"]
)
def test_domain_cannot_be_silently_changed(tmp_path, field):
    plan, model, executable, libraries = inputs(tmp_path)
    content = json.loads(model.read_text())
    content["domain"][field] = C.digest("different")
    model.write_text(json.dumps(content))
    with pytest.raises(C.ContractError, match="cannot change original"):
        R.prepare(plan, model, executable, libraries)


def test_load_order_is_pinned(tmp_path):
    plan, model, executable, libraries = inputs(tmp_path)
    with pytest.raises(C.ContractError, match="exact load order"):
        R.prepare(plan, model, executable, list(reversed(libraries)))


def test_input_binding_cannot_change(tmp_path):
    plan, model, executable, libraries = inputs(tmp_path)
    plan["observations"][0]["scope_binding"]["input_sha256"] = C.digest("another input")
    with pytest.raises(C.ContractError, match="original ELF"):
        R.prepare(plan, model, executable, libraries)


def test_old_reports_cannot_be_reused_even_with_new_plan_digest(tmp_path):
    plan, model, executable, libraries = inputs(tmp_path)
    replay = R.prepare(plan, model, executable, libraries)["observation_plan"]
    attachment = reports_for(plan)
    attachment["plan_sha256"] = C.digest(replay)
    with pytest.raises(C.ContractError):
        A.attach(replay, attachment)


def test_new_reports_join_through_existing_archive_attach(tmp_path):
    plan, model, executable, libraries = inputs(tmp_path)
    replay = R.prepare(plan, model, executable, libraries)["observation_plan"]
    dataset = A.attach(replay, reports_for(replay))
    assert len(dataset["records"]) == len(plan["observations"])
    assert [row["label"] for row in dataset["records"]] == [row["label"] for row in plan["observations"]]
