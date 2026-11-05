"""Recorded measurements cannot replace semantic or attribution qualification."""

import hashlib
import json

import pytest

from merlin.mining.reconstruction_history import load_history, read_histories, read_history


def _selector(pointer, **kwargs):
    return dict(artifact="result", pointer=pointer, **kwargs)


def _binding(tmp_path, result=None):
    result = result or dict(
        before=dict(compiler_sha256="a" * 64, program_sha256="b" * 64, settings=dict(block=[1, 1, 8], retained="none")),
        after=dict(compiler_sha256="c" * 64, program_sha256="d" * 64, settings=dict(block=[1, 1, 8], retained="lhs")),
        common=dict(
            source="source",
            input="input",
            shape=[1, 80, 48],
            roi="complete_call",
            environment=dict(engine="simulator", revision="e" * 64),
        ),
        before_cycles=1067,
        after_cycles=1514,
        numeric="pass",
        placement="pass",
        attribution="pass",
        actual_drains=dict(before=3001, after=1126),
    )
    path = tmp_path / "experiment.json"
    data = json.dumps(result).encode()
    path.write_bytes(data)
    fields = {
        side: {
            field: _selector("/" + side + "/" + field) for field in ("compiler_sha256", "program_sha256", "settings")
        }
        for side in ("before", "after")
    }
    fields["measurement"] = {name: _selector("/" + name) for name in ("before_cycles", "after_cycles")}
    fields["metadata"] = dict(actual_drains=_selector("/actual_drains", optional=True))
    return dict(
        schema="merlin.reconstruction_history_binding.v1",
        id="recorded_trial",
        artifacts=dict(result=dict(path=path.name, sha256=hashlib.sha256(data).hexdigest())),
        fields=fields,
        measurement_kind="measured",
        identities={
            axis: dict(shared=_selector("/common/" + axis))
            for axis in ("source", "input", "shape", "roi", "environment")
        },
        gates={gate: dict(selector=_selector("/" + gate)) for gate in ("numeric", "placement", "attribution")},
    )


def _rewrite(tmp_path, binding, update):
    path = tmp_path / binding["artifacts"]["result"]["path"]
    document = json.loads(path.read_text())
    update(document)
    path.write_text(json.dumps(document))
    binding["artifacts"]["result"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_measured_regression_preserves_realized_reduction_without_promoting_it(tmp_path):
    binding = _binding(tmp_path)
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["outcome"] == "measured_regression"
    assert row["metadata"]["actual_drains"] == dict(before=3001, after=1126)
    assert row["measurement"]["delta_percent"] == pytest.approx(41.893158388)
    assert row["measurement"]["same_domain"] is True
    assert row["ranking_eligible"] is True  # Eligible to rank as a regression.
    assert row["changed_factors"] == [
        dict(name="retained", before="none", after="lhs", before_present=True, after_present=True)
    ]
    assert row["trial_identity"] is not None
    assert row["next_action"] == "controlled_factor_comparison"


def test_missing_context_remains_prior_evidence_and_cannot_suppress_future_trial(tmp_path):
    binding = _binding(tmp_path)
    del binding["identities"]["input"]
    del binding["gates"]["attribution"]
    del binding["fields"]["before"]["settings"]
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["outcome"] == "measured_regression"
    assert row["measurement"]["same_domain"] is None
    assert row["measurement"]["domain_sha256"] is None
    assert row["ranking_eligible"] is False
    assert row["trial_identity"] is None
    assert row["changed_factors"] is None
    assert row["before"]["settings"] is None
    assert "identities.input" in row["unknowns"]


def test_different_input_or_numeric_json_type_does_not_match_domain(tmp_path):
    binding = _binding(tmp_path)
    _rewrite(tmp_path, binding, lambda result: result.update(before_input=True, after_input=1))
    binding["identities"]["input"] = dict(before=_selector("/before_input"), after=_selector("/after_input"))
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["measurement"]["same_domain"] is False
    assert row["ranking_eligible"] is False


def test_numeric_pass_does_not_override_placement_failure(tmp_path):
    binding = _binding(tmp_path)
    _rewrite(
        tmp_path, binding, lambda result: result.update(after_cycles=500, failure=dict(category="payload_on_host"))
    )
    binding["gates"]["placement"] = dict(selector=_selector("/failure/category"), fail_values=["payload_on_host"])
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["measurement"]["direction"] == "improvement"
    assert row["gates"]["numeric"] == "pass"
    assert row["outcome"] == "rejected_gate"
    assert row["next_action"] == "placement_repair"
    assert row["ranking_eligible"] is False


def test_attribution_failure_is_evidence_repair_not_schedule_optimization(tmp_path):
    binding = _binding(tmp_path)
    _rewrite(tmp_path, binding, lambda result: result.update(attribution="fail"))
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["outcome"] == "rejected_gate"
    assert row["next_action"] == "attribution_evidence_repair"


def test_unresolved_attribution_requests_evidence_without_inventing_failure(tmp_path):
    binding = _binding(tmp_path)
    _rewrite(tmp_path, binding, lambda result: result.update(attribution="unresolved_audit"))
    binding["gates"]["attribution"]["unresolved_values"] = ["unresolved_audit"]
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["gates"]["attribution"] == "unknown"
    assert row["next_action"] == "attribution_evidence_repair"
    assert row["ranking_eligible"] is False


def test_provisional_measurements_and_unknown_gates_stay_explicit(tmp_path):
    binding = _binding(tmp_path)
    binding["measurement_kind"] = "provisional"
    _rewrite(tmp_path, binding, lambda result: result.update(numeric="pending"))
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["outcome"] == "provisional_regression"
    assert row["gates"]["numeric"] == "unknown"
    assert row["ranking_eligible"] is False


def test_artifact_and_later_revalidation_reject_changed_historical_bytes(tmp_path):
    binding = _binding(tmp_path)
    original = read_history(binding, root=tmp_path)
    (tmp_path / "experiment.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        read_history(binding, root=tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        original.revalidate()


def test_exact_duplicate_records_deduplicate_without_case_or_label_matching(tmp_path):
    binding = _binding(tmp_path)
    duplicate = dict(binding, id="different_operator_label")
    collection = read_histories([binding, duplicate], root=tmp_path)
    assert len(collection["records"]) == 1
    assert len(collection["duplicate_records"]) == 1
    selection = dict(schema="merlin.reconstruction_history_selection.v1", bindings=[binding, duplicate])
    assert len(load_history(selection, root=tmp_path)) == 1


def test_structural_selector_supports_arrays_escaped_keys_and_optional_unknown(tmp_path):
    binding = _binding(tmp_path)
    _rewrite(tmp_path, binding, lambda result: result.update(**{"a/b~c": [17]}))
    binding["fields"]["metadata"]["selected"] = _selector("/a~1b~0c/0")
    binding["fields"]["metadata"]["absent"] = _selector("/missing", optional=True)
    row = read_history(binding, root=tmp_path).to_dict()
    assert row["metadata"]["selected"] == 17
    assert row["metadata"]["absent"] is None
    binding["fields"]["metadata"]["selected"] = _selector("/a~")
    with pytest.raises(ValueError, match="escape"):
        read_history(binding, root=tmp_path)


@pytest.mark.parametrize("cycles", [True, -1, float("inf"), float("nan")])
def test_invalid_cycles_cannot_become_a_measurement(tmp_path, cycles):
    binding = _binding(tmp_path)
    _rewrite(tmp_path, binding, lambda result: result.update(after_cycles=cycles))
    with pytest.raises(ValueError, match="cycles"):
        read_history(binding, root=tmp_path)


def test_bounded_inputs_and_required_fields_fail_closed(tmp_path):
    binding = _binding(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        read_history(binding, root=tmp_path, max_artifact_bytes=10)
    binding["fields"]["measurement"]["after_cycles"] = _selector("/not_recorded")
    with pytest.raises(ValueError, match="required history field"):
        read_history(binding, root=tmp_path)
