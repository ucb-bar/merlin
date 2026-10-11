"""Synthetic normal routing and evidence refusals; no compiler or engine starts."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import test_ordinary_search_policy as NORMAL
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2 import ordinary_calibration as CAL
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import search_inputs as SI
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.retained_revisions import executing_revision

revisions = NORMAL.revisions


@pytest.fixture
def ordinary_inputs(tmp_path, monkeypatch):
    return NORMAL.ordinary_inputs.__wrapped__(tmp_path, monkeypatch)


def extend_corpus(fixture, tmp_path):
    source = tmp_path / "generated"
    member = source / "_tuning/held"
    member.mkdir()
    descriptor = json.loads((source / "_tuning/member/capsule.yaml").read_text())
    descriptor["name"] = "held"
    (member / "capsule.yaml").write_text(json.dumps(descriptor))
    manifest = source / "MANIFEST.yaml"
    document = json.loads(manifest.read_text())
    document["generated"].append("_tuning/held")
    manifest.write_text(json.dumps(document))
    public = source / "public"
    live = C.discover_performance_corpus(
        SimpleNamespace(target="fixture", capsule_corpus=public, graded_roots=lambda: [public])
    )
    fixture.arguments["corpus"] = C.freeze_performance_corpus(live, tmp_path / "extended_frozen")
    return fixture


@pytest.fixture
def cheap_inputs(ordinary_inputs, monkeypatch):
    calls = []

    def selected_adapter(engine, target):
        assert engine == "spike"
        return "synthetic-selected-spike"

    def run(capsule, package, **kwargs):
        issued = executing_revision(package)
        assert issued is not None
        assert kwargs["oracle_adapters"] == {"L2": "synthetic-selected-spike"}
        assert kwargs["adapter_managed_tiers"] == ("L2",)
        assert capsule["oracle_tier_ceiling"] == "L2"
        calls.append((capsule, package, kwargs))
        return {"tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}, "status": "incomplete"}

    monkeypatch.setattr(FAST.CR, "load_capsule", lambda path, **kw: {"required_oracle_tiers": ["L0", "L1", "L2", "L3"]})
    monkeypatch.setattr(FAST.CR, "run_capsule", run)
    monkeypatch.setattr(FAST.CR, "simulator_adapter", selected_adapter)
    monkeypatch.setattr(FAST.PC, "package_sandbox_policy", lambda *args: "synthetic-original-policy")
    monkeypatch.setattr(FAST.PC, "boxed_entrypoints", lambda policy: nullcontext())
    return ordinary_inputs, calls


def test_retained_cheap_route_executes_only_original_spike_and_keeps_rtl_pending(cheap_inputs, tmp_path):
    inputs, calls = cheap_inputs
    policy = OS.prepare_ordinary_search_policy(**inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    policy.candidate.joinpath("compiler.py").write_text("unfinished: [")
    result = policy.evaluate_retained(
        candidate, baseline, kind=FAST.KIND, work_root=tmp_path / "cheap", call_index=0, timeout_s=10
    )
    assert len(calls) == 2 * len(policy.search_corpus.capsules)
    assert {package for _, package, _ in calls} == {candidate.source, baseline.source}
    assert not inputs.executions  # The original synthetic RTL spy never ran.
    assert result["screening"]["status"] == "UNKNOWN"
    assert result["promotion"] == "SCREENING_ONLY"
    assert str(tmp_path) not in json.dumps(result)
    for cell in result["cells"]:
        for arm in ("baseline", "candidate"):
            assert cell[arm]["functional_status"] == "PASS"
            assert cell[arm]["pending_tiers"] == ["L3"]
            assert cell[arm]["compiled"]["status"] == "UNKNOWN"
            assert cell[arm]["timing_cycles"] is None
    inputs = SI.SearchInputs(policy)
    assert inputs.kinds == (FAST.KIND, SI.ORDINARY_KIND)
    assert inputs.validate_result(result, candidate, FAST.KIND) == result
    with pytest.raises(StageGateError, match="another evaluator"):
        inputs.validate_result(result, candidate, "arbitrary")


@pytest.mark.parametrize("drift", ["member", "traffic", "timing", "configuration", "roster"])
def test_cheap_transport_cannot_gain_timing_or_other_input_identity(cheap_inputs, tmp_path, drift):
    fixture, _ = cheap_inputs
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    result = policy.evaluate_retained(
        candidate, baseline, kind=FAST.KIND, work_root=tmp_path / "cheap", call_index=0, timeout_s=10
    )
    result = copy.deepcopy(result)
    if drift == "member":
        result["cells"][0]["source_sha256"] = "d" * 64
    elif drift in ("traffic", "timing"):
        result["cells"][0]["candidate"]["physical_memory_traffic" if drift == "traffic" else "timing_cycles"] = 1
    elif drift == "configuration":
        result["configuration_sha256"] = "d" * 64
    else:
        result["cells"].pop()
    with pytest.raises(StageGateError):
        SI.SearchInputs(policy).validate_result(result, candidate, FAST.KIND)


def test_public_partition_requires_disjoint_complete_original_members(ordinary_inputs, tmp_path):
    extend_corpus(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    members = [[r.family, r.capsule] for r in policy.search_corpus.capsules]
    partition = {"schema": CAL.PARTITION_SCHEMA, "calibration": members[:1], "held_validation": members[1:]}
    assert len(CAL.public_partition(policy, partition)) == len(members)
    partition["held_validation"].append(members[0])
    with pytest.raises(StageGateError, match="leaks"):
        CAL.public_partition(policy, partition)


def test_saved_calibration_and_incomplete_native_join_cannot_qualify(ordinary_inputs, tmp_path):
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    foreign = CAL.OrdinaryCalibration(CAL.domain(policy), b"{}", b"[]", b'{"exposable":true}', (), object())
    with pytest.raises(StageGateError, match="not live"):
        policy.install_screen(foreign)
    with pytest.raises(StageGateError, match="fresh joined"):
        CAL._join(tmp_path, {"family": "public", "capsule": "member"}, "candidate", {}, "a" * 64)
    assert policy.publish_completed_artifacts(tmp_path)["status"] == "UNAVAILABLE"


def test_compiled_error_paths_stay_in_private_record(tmp_path, monkeypatch):
    secret = str(tmp_path / "private-compiler-product")

    def unavailable(*args):
        raise ValueError("missing " + secret)

    monkeypatch.setattr(FAST, "_unique", unavailable)
    result = FAST.compiled_observation(tmp_path, target="synthetic")
    assert secret not in json.dumps(result)
    assert secret in (tmp_path / "compiled_failure_private.json").read_text()


def test_calibration_reopens_deterministic_private_evidence_and_rejects_returned_path(tmp_path):
    from merlin_experiments.phase2.contracts import sha256_file

    root = tmp_path / "cheap"
    evidence = root / "m000_candidate/ordinary_observation.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({"compiled": {"status": "UNKNOWN"}, "original_grade": {"private": "details"}}))
    cell = {"candidate": {"compiled": {"status": "UNKNOWN"}, "evidence_sha256": sha256_file(evidence)}}
    private, pin = CAL._cheap_observation(root, cell, "candidate", 0)
    assert private == {"compiled": {"status": "UNKNOWN"}}
    assert pin[0] == str(evidence)
    cell["candidate"]["evidence"] = {"path": str(tmp_path / "caller-file")}
    with pytest.raises(StageGateError, match="actual retained"):
        CAL._cheap_observation(root, cell, "candidate", 0)
    cell["candidate"].pop("evidence")
    evidence.write_text('{"changed":true}')
    with pytest.raises(StageGateError, match="changed"):
        CAL._cheap_observation(root, cell, "candidate", 0)


def test_different_shapes_cannot_supply_same_member_ranking_denominator():
    from merlin_experiments.phase2.contracts import document_sha256

    from merlin.perf.fast_estimate_validation import Observation

    rows = []
    for index in range(20):
        x = index + 1
        rows.append(
            Observation(
                document_sha256(["train", index]),
                document_sha256(["training-member", index]),
                "calibration",
                "a" * 64,
                {FAST.FEATURES[0]: float(x), FAST.FEATURES[1]: float(x * x)},
                float(20 + 3 * x + 2 * x * x),
                ("b" * 64,),
            )
        )
    for index in range(24):
        x = index % 8 + 1
        rows.append(
            Observation(
                document_sha256(["held", index]),
                document_sha256(["different-member", index]),
                "held-" + str(index % 3),
                "a" * 64,
                {FAST.FEATURES[0]: float(x), FAST.FEATURES[1]: float(x * x)},
                float(20 + 3 * x + 2 * x * x),
                ("b" * 64,),
            )
        )
    report = CAL._qualify(rows, "c" * 64)
    assert report["ranking"]["overall"]["decided"] == 0
    assert report["exposable"] is False
    assert report["policy"]["minimum_decided"] == 100
