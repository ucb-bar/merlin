"""Final CCA replies reuse public details across both real broker decorations."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean_contract import document_digest
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.brokers.selfcheck_sources import pinned_source, request_label
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from merlin_experiments.phase1.feedback.result_details import load_row
from test_reference_experiment_feedback import _live_fixture


def _large(tmp_path, count=400):
    fixture, report = _live_fixture(tmp_path)
    # One compared function/axis with a large witness: a lower page limit does
    # not shrink this payload, as in the closed study's seven size refusals.
    report["cca"]["divergences"][0]["witness"] = ["retained loop and operand evidence " * 8] * count
    report["operand_comparison"] = dict(status="compared_recovered_operands",
        order_change="reordered_same_recovered_operands", semantic_alignment="unknown")
    base = dict(schema=analysis.SCHEMA, all_pass=None, numeric_assessment="not_run", official=False,
                qualification_issued=False, semantic_equivalence="unknown", performance="unknown")
    return fixture, report, base


def _result(fixture, report, base):
    return analysis._response(base, fixture["packet"], copy.deepcopy(report), fixture["compiler"],
                              workspace=fixture["workspace"])


def test_huge_single_witness_preserves_full_detail_identity_and_requests(tmp_path):
    fixture, report, base = _large(tmp_path)
    selection_bytes = fixture["path"].read_bytes()
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        result = _result(fixture, report, base)
        requests = reference.pending_requests(fixture["compiler"])
        detail = tmp_path / result["detail"]["path"]
        before = detail.stat().st_mtime_ns
        repeated = _result(fixture, report, base)
        assert requests == reference.pending_requests(fixture["compiler"])
    assert result == repeated and requests
    assert result["status"] == "analyzed" and len(json.dumps(result).encode()) < 64 << 10
    assert result["report"]["kind"] == report["kind"] and result["report_in_detail"] is True
    assert result["report"]["operand_comparison"] == report["operand_comparison"]
    assert detail.stat().st_mtime_ns == before
    assert len(list((tmp_path / "qa/cases/reference_analysis").glob("*.json"))) == 1
    raw = detail.read_bytes()
    assert result["detail"]["sha256"] == hashlib.sha256(raw).hexdigest() and len(raw) > 64 << 10
    full = load_row(result, workspace=tmp_path)
    assert full["report"]["cca"] == report["cca"]
    assert full["report_identity"] == result["report_identity"] == document_digest(full["report"])
    assert {row["request_id"] for task in full["report"]["experiment"]["tasks"] for row in task["requests"]} == {
        row["request_id"] for row in requests}
    assert result["report"]["experiment"]["identity"] == full["report"]["experiment"]["identity"]
    assert result["all_pass"] is None and result["qualification_issued"] is False
    assert result["semantic_equivalence"] == result["performance"] == "unknown"
    assert fixture["path"].read_bytes() == selection_bytes
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("witnesses", [80, 400])
def test_lean_double_decoration_bounds_transport_without_republishing(tmp_path, monkeypatch, witnesses):
    fixture, report, _ = _large(tmp_path, witnesses)
    path = repo_root() / "examples/gemmini/phase1/studies/kernel_abstractions/run.py"
    spec = importlib.util.spec_from_file_location("bounded_lean_runner", path)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    run.CH, run.WS, run.F = channel, tmp_path, repo_root()
    run.C = dict(reference_guidance="on", facts=str(tmp_path / "facts"), descriptor=str(tmp_path / "descriptor"))
    run.shim = SimpleNamespace(_submission_digest=broker._submission_digest)
    run.request_label, run.pinned_source = request_label, pinned_source
    run.reference_analysis = analysis
    run.selected = lambda: reference.invocation(fixture["path"], fixture["sha"], tasks=False)
    # No compiler/evaluator: use one closed report through the actual planner,
    # response owner, two decorators and original lean channel writer.
    monkeypatch.setattr(analysis, "_analyze", lambda *a, **kw: (copy.deepcopy(report), None))
    original_record, publications = reference.record_analysis, []

    def record(*args, **kwargs):
        publications.append(1)
        return original_record(*args, **kwargs)

    monkeypatch.setattr(reference, "record_analysis", record)
    original_decorate, sizes, pointers = broker._decorate_response, [], []

    def decorate(raw, *args, **kwargs):
        body = original_decorate(raw, *args, **kwargs)
        sizes.append((len(raw.encode()), len(json.dumps(body, indent=2).encode()) + 1))
        pointers.append(body.get("detail"))
        return body

    monkeypatch.setattr(broker, "_decorate_response", decorate)
    run._decorate_response = decorate
    request = channel / "req_large.json"
    request.write_text(json.dumps(dict(action="reference-analysis", submission_sha256=fixture["compiler"],
        reference_analysis=dict(operation="candidate", path="artifacts/views.json", format="artifact_views"))))
    run.serve(request)
    raw = (channel / "resp_large.json").read_bytes()
    response = json.loads(raw)
    assert response["status"] == "analyzed", response
    assert response["protocol"] == 3 and response["selfcheck_request_id"] == "large"
    assert response["submission_sha256"] == fixture["compiler"]
    assert response["reference_experiments"]["required_selfchecks"][0]["request_id"]
    assert isinstance(response["reference_experiments"]["next_action"]["gaps"], list)
    assert isinstance(response["next_steps"]["lines"], list)
    assert publications == [1] and len(sizes) == 2 and len(raw) < 64 << 10
    if witnesses == 80:
        assert sizes[0][0] < 64 << 10 < sizes[0][1]  # feedback alone crosses the bound
    else:
        assert pointers[0] == pointers[1] == response["detail"]
    assert sizes[1][1] > 64 << 10  # proves the final lean projection is needed
    full = load_row(response, workspace=tmp_path)
    assert full["report"]["cca"] == report["cca"]
    assert response["report_identity"] == document_digest(full["report"])
    assert response["qualification_issued"] is False and response["all_pass"] is None
    assert (channel / "done_large").read_text() == "ok\n"


def test_large_request_field_keeps_minimal_analyzed_reply_with_exact_detail(tmp_path):
    fixture, report, base = _large(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        initial = _result(fixture, report, base)
    response = dict(initial, reference_experiments=dict(packet_identity="p", task_id="current-task",
        required_selfchecks=[dict(request_id="current-request", argv=["x" * (70 << 10)])]))
    result = analysis._bounded_response(response, workspace=tmp_path)
    assert result["status"] == "analyzed" and len(json.dumps(result).encode()) < 64 << 10
    assert result["detail"] == initial["detail"] and result["report_identity"] == initial["report_identity"]
    assert result["reference_experiments"]["request_ids"] == ["current-request"]
    assert "exceeded" in result["report"]["projection"]
    assert load_row(result, workspace=tmp_path)["report"]["cca"] == report["cca"]
    assert result["all_pass"] is None and result["qualification_issued"] is False


@pytest.mark.parametrize("remove", [False, True])
def test_missing_or_changed_detail_cannot_be_replaced_with_its_summary(tmp_path, remove):
    fixture, report, base = _large(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        initial = _result(fixture, report, base)
    detail = tmp_path / initial["detail"]["path"]
    if remove:
        detail.unlink()
    else:
        detail.write_text("changed retained evidence")
    expanded = dict(initial, reference_experiments=dict(observed=["x" * (70 << 10)]))
    result = analysis._bounded_response(expanded, workspace=tmp_path)
    assert result["status"] == "unknown" and result["reason"] == "analysis_report_detail_unavailable"
    assert result["detail"] == initial["detail"] and result["report_identity"] == initial["report_identity"]
    assert result["all_pass"] is None and result["qualification_issued"] is False
    assert len(list((tmp_path / "qa/cases/reference_analysis").glob("*.json"))) == (0 if remove else 1)
