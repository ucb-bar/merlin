"""Synthetic normal routing and evidence refusals; no compiler or engine starts."""

import copy
import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
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

from merlin.common import invocation_record as I

revisions = NORMAL.revisions


@pytest.fixture
def ordinary_inputs(tmp_path, monkeypatch, request):
    return NORMAL.ordinary_inputs.__wrapped__(tmp_path, monkeypatch, request)


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


def test_selected_cheap_route_runs_only_selected_members_and_preserves_pending_roster(cheap_inputs, tmp_path):
    fixture, calls = cheap_inputs
    extend_corpus(fixture, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    selected = tuple(sorted(policy.validate_search_inputs()))[:1]
    result = policy.evaluate_retained(
        candidate,
        baseline,
        kind=FAST.SELECTED_KIND,
        selected_members=selected,
        work_root=tmp_path / "selected-cheap",
        call_index=0,
        timeout_s=10,
    )
    assert len(calls) == 2 and not fixture.executions
    assert result["promotion"] == "DIAGNOSTIC_ONLY" and result["screening"]["status"] == "UNKNOWN"
    assert result["selected_members"] == [list(key) for key in selected]
    assert len(result["cells"]) == 2
    for cell in result["cells"]:
        if (cell["family"], cell["capsule"]) in selected:
            assert cell["pending_reason"] is None
            for arm in ("baseline", "candidate"):
                assert cell[arm]["functional_status"] == "PASS" and cell[arm]["pending_tiers"] == ["L3"]
        else:
            assert cell["baseline"] is cell["candidate"] is None and cell["pending_reason"]
    inputs = SI.SearchInputs(policy)
    assert inputs.validate_result(result, candidate, FAST.SELECTED_KIND) == result
    FAST.validate_selected_feedback(result, members=selected)
    with pytest.raises(StageGateError):
        inputs.validate_result(result, candidate, FAST.KIND)
    with pytest.raises(StageGateError, match="queued selection"):
        FAST.validate_selected_feedback(result, members=tuple(sorted(policy.validate_search_inputs()))[1:])
    with pytest.raises(CAL.CalibrationUnavailable, match="complete original"):
        object.__new__(CAL.OrdinaryCalibration).screen(policy, result)


@pytest.mark.parametrize("drift", ["unselected_arm", "selected_missing", "screening", "promotion", "selection"])
def test_selected_cheap_transport_cannot_claim_unselected_work_or_complete_screening(cheap_inputs, tmp_path, drift):
    fixture, _calls = cheap_inputs
    extend_corpus(fixture, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    selected = tuple(sorted(policy.validate_search_inputs()))[:1]
    result = policy.evaluate_retained(
        candidate,
        baseline,
        kind=FAST.SELECTED_KIND,
        selected_members=selected,
        work_root=tmp_path / "selected-cheap",
        call_index=0,
        timeout_s=10,
    )
    result = copy.deepcopy(result)
    observed = next(row for row in result["cells"] if row["baseline"] is not None)
    pending = next(row for row in result["cells"] if row["baseline"] is None)
    if drift == "unselected_arm":
        pending["candidate"] = observed["candidate"]
    elif drift == "selected_missing":
        observed["baseline"] = None
    elif drift == "selection":
        result["selected_members"].append(["private", "not-in-public-roster"])
    elif drift == "promotion":
        result["promotion"] = "SCREENING_ONLY"
    else:
        result["screening"]["status"] = "qualified_public_screening"
    with pytest.raises(StageGateError):
        SI.SearchInputs(policy).validate_result(result, candidate, FAST.SELECTED_KIND)


def test_partial_kind_cannot_enter_complete_measured_or_provisional_ranking(ordinary_inputs, tmp_path):
    from merlin_experiments.phase2 import provisional_search, search_coordinator

    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    roster = tuple(sorted(policy.validate_search_inputs()))
    observation = {"kind": FAST.SELECTED_KIND, "state": "completed"}
    assert (
        search_coordinator.rank_measured_contenders(
            [observation],
            roster=roster,
            corpus_sha256=policy.search_corpus.capsules_sha256,
            configuration_sha256=policy.feedback_evaluator.certificate.sha256,
            timing_kind=SI.ORDINARY_KIND,
        )["ranked"]
        == []
    )
    pool = SimpleNamespace(
        inputs=SimpleNamespace(ordinary=True),
        revisions=SimpleNamespace(get=lambda *_a: pytest.fail("partial parent is not ranked")),
    )
    assert provisional_search.project(SimpleNamespace(pool=pool, roster=roster), [observation])["candidates"] == []


@pytest.mark.parametrize("kind,members", [(FAST.SELECTED_KIND, ()), (FAST.KIND, (("private", "absent"),))])
def test_selected_route_requires_explicit_public_members_without_narrowing_full_jobs(
    cheap_inputs, tmp_path, kind, members
):
    fixture, calls = cheap_inputs
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    with pytest.raises(StageGateError):
        policy.evaluate_retained(
            candidate,
            baseline,
            kind=kind,
            selected_members=members,
            work_root=tmp_path / "refused",
            call_index=0,
            timeout_s=10,
        )
    assert not calls


def test_fixed_worker_and_broker_preserve_selected_functional_scope_without_running_a_pool(
    cheap_inputs, tmp_path, monkeypatch
):
    from merlin_experiments.phase2 import evaluation_pool as EP
    from merlin_experiments.phase2 import queued_search_actions as ACTIONS

    fixture, calls = cheap_inputs
    extend_corpus(fixture, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    members = tuple(sorted(policy.validate_search_inputs()))[:1]
    root = tmp_path / "worker"
    root.mkdir()
    # Exercise real dispatch; OS allocation and queue edges remain explicit stubs.
    selected_cpu = min(EP.os.sched_getaffinity(0))
    monkeypatch.setattr(EP.os, "sched_setaffinity", lambda *_a: None)
    monkeypatch.setattr(EP.os, "sched_getaffinity", lambda *_a: {selected_cpu})
    monkeypatch.setattr(EP.resource, "setrlimit", lambda *_a: None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("MERLIN_PERF_SWEEP_WORKERS", "2")
    result = EP._evaluate_fixed_routes(
        policy=policy,
        revision=candidate,
        baseline_revision=baseline,
        kind=FAST.SELECTED_KIND,
        selected_members=members,
        timeout_s=10,
        root=root,
        grant=SimpleNamespace(cores=1, engine_slots=1, ram_bytes=1 << 30, disk_bytes=1 << 20),
        cpus=tuple(sorted(EP.os.sched_getaffinity(0)))[:1],
        index=0,
    )
    assert result["selected_members"] == [list(key) for key in members] and len(calls) == 2
    submitted = []
    policy.evaluation_pool = SimpleNamespace(submit=lambda *args, **kwargs: submitted.append((args, kwargs)) or {})
    policy.search_coordinator = SimpleNamespace(submit=policy.evaluation_pool.submit)
    ACTIONS.execute(
        policy,
        FAST.SELECTED_ACTION,
        {"revision_id": candidate.revision_id, "members": "/".join(members[0])},
        10,
        time.monotonic(),
    )
    assert submitted == [
        ((candidate.revision_id,), {"kind": FAST.SELECTED_KIND, "selected_members": members, "timeout_s": 10})
    ]
    assert FAST.SELECTED_ACTION in {action.name for action in ACTIONS.actions()}


def test_census_uses_original_selected_facts_and_restores_ambient_context(cheap_inputs, tmp_path, monkeypatch):
    fixture, _ = cheap_inputs
    selected_facts = {"facts": {"source": {"config": "original-board"}}, "observation": "selected"}
    path = fixture.arguments["rtl_facts_path"]
    path.write_text(json.dumps(selected_facts))
    seen = []

    def census(root, **kwargs):
        # This is the actual accessor used by the nested descriptive observer;
        # no target lookup, extraction, engine or namespace runs in this control.
        seen.append(FAST.RTL_FACTS.load_facts(kwargs["target"], regenerate=False))
        return {"status": "UNKNOWN", "reason": "synthetic control"}

    monkeypatch.setattr(FAST, "observe_passed_census", census)
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    ambient = {"observation": "ambient, not the selected original facts"}
    with FAST.RTL_FACTS.observed_facts("fixture", ambient, tmp_path / "ambient.json"):
        policy.evaluate_retained(
            candidate, baseline, kind=FAST.KIND, work_root=tmp_path / "cheap", call_index=0, timeout_s=10
        )
        assert FAST.RTL_FACTS.load_facts("fixture", regenerate=False) == ambient
    assert seen == [selected_facts] * (2 * len(policy.search_corpus.capsules))


def test_optional_census_budget_cannot_block_original_cheap_grade(cheap_inputs, tmp_path, monkeypatch):
    fixture, _ = cheap_inputs
    # A bounded observer does not load an oversized fact document. The source
    # policy still pins it; sparse bytes here avoid an unnecessary allocation.
    with fixture.arguments["rtl_facts_path"].open("wb") as stream:
        stream.truncate((8 << 20) + 1)
    calls = []
    monkeypatch.setattr(FAST, "observe_passed_census", lambda *args, **kwargs: calls.append(kwargs))
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    result = policy.evaluate_retained(
        candidate, baseline, kind=FAST.KIND, work_root=tmp_path / "cheap", call_index=0, timeout_s=10
    )
    assert not calls
    for cell in result["cells"]:
        for arm in ("candidate", "baseline"):
            assert cell[arm]["functional_status"] == "PASS"
            assert cell[arm]["compiled"]["dynamic_census"]["status"] == "UNKNOWN"
            assert cell[arm]["pending_tiers"] == ["L3"]


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


@pytest.mark.parametrize("stage", ["elf", "original_build_process", "object"])
def test_cheap_features_require_original_completed_elf_producer(tmp_path, monkeypatch, stage):
    """File-only producer selection; synthetic bytes are never linked or run."""
    assert Path(I.__file__).resolve().is_relative_to(Path.cwd() / "src")
    assert Path(FAST.__file__).resolve().is_relative_to(Path.cwd() / "packages/merlin-experiments/src")
    llvm, elf = tmp_path / "lowered.llvm.mlir", tmp_path / FAST.PACKAGE_ELF_NAME
    llvm.write_text("synthetic emitted source")
    elf.write_bytes(b"synthetic linked bytes")
    tool = tmp_path / "linker"
    tool.write_bytes(b"fake executable; never loaded")
    tool.chmod(0o555)
    with I.observe(tmp_path, stage=stage, argv=[str(tool)], inputs=(llvm,), outputs=(elf,), env={}) as recorded:
        recorded.complete(subprocess.CompletedProcess([str(tool)], 0, b"", b""))
    recipe = SimpleNamespace(require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="entry"))
    monkeypatch.setattr(FAST, "get_backend", lambda target: SimpleNamespace(harness_build_recipe=lambda: recipe))
    monkeypatch.setattr(FAST.S, "linked_entry_extent", lambda data, entry: (0, len(data)))
    monkeypatch.setattr(
        FAST.S,
        "derive_compiled_static_features",
        lambda **kwargs: {
            "emitted": {"action_sites": {"synthetic": 3}},
            "linked": {"executable_bytes": len(elf.read_bytes())},
        },
    )
    result = FAST.compiled_observation(tmp_path, target="synthetic")
    assert result["status"] == ("observed" if stage == "elf" else "UNKNOWN")
    if stage == "elf":
        assert result["features"][FAST.FEATURES[0]] == 3
        elf.write_bytes(b"changed after completed link")
        assert FAST.compiled_observation(tmp_path, target="synthetic")["status"] == "UNKNOWN"


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
