"""Diagnostic stage wiring controls; synthetic owners grant no experiment authority.

The ordinary final controller must retain original owners and evaluate actual
edited bytes before qualification. These routing mocks cannot issue origin,
compile/static, numerical, runtime, isolation or final acceptance authority.
"""

import contextlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_final_qualification as FQ
from merlin_experiments.phase2 import component_launch as CL
from merlin_experiments.phase2 import component_stage as CS
from merlin_experiments.phase2.contracts import StageGateError


@pytest.fixture
def diagnostic_stage(tmp_path, monkeypatch):
    candidate, stage, seed = tmp_path / "candidate", tmp_path / "stage", tmp_path / "seed"
    candidate.mkdir()
    stage.mkdir()
    (candidate / "compiler.py").write_text("# synthetic transport fixture\n")
    shutil.copytree(candidate, seed)
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    target = SimpleNamespace(path=descriptor, target="synthetic")
    corpus = SimpleNamespace(manifest_sha256="1" * 64, capsules_sha256="2" * 64, capsules=())
    roster, build_service, instruction_check, baseline_admission, runtime_authority = (object() for _ in range(5))
    original_evaluation = SimpleNamespace(
        roster=roster,
        build_service=build_service,
        instruction_check=instruction_check,
        readelf=Path("/usr/bin/readelf"),
    )
    origin = SimpleNamespace(receipt_sha256="3" * 64, inputs=SimpleNamespace(compile_roster=roster))
    observed, lineage = [], SimpleNamespace(receipt_sha256="5" * 64)

    def verify_baseline(*, candidate):
        assert candidate == seed and (seed / "compiler.py").read_text() == "# synthetic transport fixture\n"
        observed.append("baseline_verify")

    qualification = SimpleNamespace(
        corpus_root=tmp_path / "corpus",
        contract_root=tmp_path / "contract",
        source_root=tmp_path,
        runtime=(object(),),
        compiler_origin=origin,
        runtime_authority=runtime_authority,
        compile_role_evaluation=original_evaluation,
        verify=verify_baseline,
    )

    def policy(**kwargs):
        # Catch the original constructor omission before model authoring.
        if (
            kwargs.get("baseline_admission") is not baseline_admission
            or kwargs.get("independent_runtime") is not runtime_authority
        ):
            raise StageGateError("diagnostic original owner omission")
        return SimpleNamespace(
            **kwargs, workflow_id="synthetic-transport", build_registry=lambda: (), verify_receipts=lambda *a, **kw: {}
        )

    initial = policy(
        candidate=candidate,
        target_experiment=target,
        receipt_path=stage / "probe/receipts.jsonl",
        component_corpus=corpus,
        component_analytical=object(),
        component_cca=None,
        component_rtl=None,
        services=object(),
        baseline_admission=baseline_admission,
        independent_runtime=runtime_authority,
    )
    inputs = CL.ComponentLaunchInputs(
        SimpleNamespace(root=tmp_path / "view"),
        candidate,
        initial,
        SimpleNamespace(seed=seed),
        qualification,
        qualification.runtime,
        (SimpleNamespace(destination="/usr/bin/bwrap", source=Path("/usr/bin/bwrap"), verify=lambda: None),),
        (),
        Path("/usr/bin/client"),
        "/usr/bin/client",
        tmp_path / "private_auth",
        stage,
        tmp_path / "prices.json",
    )
    launch = CL.QualifiedComponentLaunch(inputs, tmp_path / "qualification.json", "4" * 64, "fixture", "low", object())
    monkeypatch.setattr(CL.QualifiedComponentLaunch, "verify", lambda self: None)
    monkeypatch.setattr(CL.ComponentLaunchInputs, "verify", lambda self, **kwargs: None)
    # This routing fixture bypasses transport admission and never runs a tool.
    # Real outer-sandbox/origin checks are exercised by the separate launch controls.
    monkeypatch.setattr(
        CL.ComponentLaunchInputs, "sandbox_binary", property(lambda self: Path("/diagnostic/outer-sandbox"))
    )
    monkeypatch.setattr(CS, "ComponentOnlyPolicy", policy)
    monkeypatch.setattr(CS, "strict_tool_policy", lambda *args, **kwargs: ())
    monkeypatch.setattr(CS.SI, "prepare_component_prompt_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(CS.SP, "render_component_prompt", lambda *args: "synthetic transport control")
    monkeypatch.setattr(CL.B, "stage_broker_shim", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        CL.B,
        "Broker",
        lambda *args, **kwargs: SimpleNamespace(token="fixture", serving=lambda **kwargs: contextlib.nullcontext()),
    )
    monkeypatch.setattr(CS.TEL, "prepare", lambda **kwargs: {})
    monkeypatch.setattr(CS.TEL, "collect_round", lambda *args, **kwargs: {})
    monkeypatch.setattr(CS.TEL, "finalize", lambda *args, **kwargs: {})
    monkeypatch.setattr(CS, "audit_codex_transcript", lambda *args: {"clean": True})

    def author(actual_launch, authored_inputs, **kwargs):
        assert actual_launch is launch and authored_inputs.qualification is qualification
        assert authored_inputs.policy.receipt_path == stage / "control-authoring/receipts.jsonl"
        for name in (
            "component_corpus",
            "component_analytical",
            "component_cca",
            "component_rtl",
            "services",
            "baseline_admission",
            "independent_runtime",
            "target_experiment",
        ):
            assert getattr(authored_inputs.policy, name) is getattr(initial, name)
        (candidate / "compiler.py").write_text("# synthetic observed descendant edit\n")
        transcript = stage / "synthetic_transcript.json"
        transcript.write_text("{}\n")
        observed.append("author")
        return 0, transcript, lineage

    def evaluate(**kwargs):
        assert kwargs["candidate"] == candidate and CS.hash_tree(candidate) != CS.hash_tree(seed)
        assert kwargs["compiler_origin"] is origin and kwargs["compiler_lineage"] is lineage
        assert kwargs["roster"] is roster and kwargs["build_service"] is build_service
        assert kwargs["instruction_check"] is instruction_check and kwargs["readelf"] is original_evaluation.readelf
        assert kwargs["contract_root"] == qualification.contract_root and kwargs["timeout_s"] == 10
        owner = kwargs["evidence_root"]
        assert owner == stage / "final_compile_role_evaluation" and not owner.exists()
        owner.mkdir()
        (owner / "diagnostic.json").write_text('{"scope":"routing mock only; no static authority"}')
        observed.append("compile")

        def complete(*, candidate):
            assert candidate == inputs.candidate
            observed.append("static_verify")

        return SimpleNamespace(require_complete=complete)

    fresh_evaluations = []

    def recorded_evaluate(**kwargs):
        evaluation = evaluate(**kwargs)
        fresh_evaluations.append(evaluation)
        return evaluation

    def requalify(actual_candidate, **kwargs):
        assert actual_candidate == candidate and kwargs["compiler_origin"] is origin
        assert kwargs["compiler_lineage"] is lineage and kwargs["runtime_authority"] is runtime_authority
        assert kwargs["compile_role_evaluation"] is fresh_evaluations[-1]
        assert kwargs["compile_role_evaluation"] is not original_evaluation
        assert kwargs["runtime"] is qualification.runtime and kwargs["view"] is inputs.view
        assert kwargs["target_experiment"] is target
        observed.append("qualify")
        return SimpleNamespace(receipt_sha256="6" * 64, verify=lambda: observed.append("verify"))

    monkeypatch.setattr(CS, "run_component_origin_round", author)
    monkeypatch.setattr(FQ, "evaluate_component_compile_roles", recorded_evaluate)
    monkeypatch.setattr(FQ, "qualify_component_compiler", requalify)
    return SimpleNamespace(
        launch=launch,
        observed=observed,
        inputs=inputs,
        lineage=lineage,
        original_evaluation=original_evaluation,
        fresh_evaluations=fresh_evaluations,
    )


def _run(launch):
    return CL.run_component_stage(
        launch,
        model="fixture",
        effort="low",
        wall_budget_seconds=10,
        max_tool_calls=2,
        tool_timeout_seconds=10,
        suite="synthetic-transport",
    )


def test_stage_requalifies_only_the_observed_descendant(diagnostic_stage, monkeypatch):
    fixture = diagnostic_stage
    document = CL.C.mapping_file(_run(fixture.launch))
    assert fixture.observed == ["author", "baseline_verify", "compile", "static_verify", "qualify", "verify"]
    assert document["phase1_origin_sha256"] == fixture.inputs.qualification.compiler_origin.receipt_sha256
    assert document["phase2_lineage_sha256"] == fixture.lineage.receipt_sha256
    assert document["admission"]["consumable"] is True
    assert document["final_acceptance"] == "NOT_ESTABLISHED"

    def refused_author(*args, **kwargs):
        raise StageGateError("observed descendant refused")

    monkeypatch.setattr(CS, "run_component_origin_round", refused_author)
    with pytest.raises(StageGateError, match="descendant refused"):
        _run(fixture.launch)
    assert fixture.observed == ["author", "baseline_verify", "compile", "static_verify", "qualify", "verify"]


@pytest.mark.parametrize("failure", ["unresolved", "baseline_reuse", "compile_crash", "baseline_drift"])
def test_stage_cannot_promote_missing_or_unresolved_descendant_static_roles(diagnostic_stage, monkeypatch, failure):
    fixture = diagnostic_stage
    if failure == "baseline_drift":

        def refused_baseline(**kwargs):
            raise StageGateError("diagnostic original selection drift")

        fixture.inputs.qualification.verify = refused_baseline
    elif failure == "baseline_reuse":
        monkeypatch.setattr(FQ, "evaluate_component_compile_roles", lambda **kwargs: fixture.original_evaluation)
    elif failure == "compile_crash":

        def failed_compile(**kwargs):
            raise RuntimeError("diagnostic actual compilation unavailable")

        monkeypatch.setattr(FQ, "evaluate_component_compile_roles", failed_compile)
    else:
        original = FQ.evaluate_component_compile_roles

        def unresolved_compile(**kwargs):
            evaluation = original(**kwargs)

            def refuse(**kwargs):
                raise StageGateError("required original compilation/static roles remain incomplete")

            evaluation.require_complete = refuse
            return evaluation

        monkeypatch.setattr(FQ, "evaluate_component_compile_roles", unresolved_compile)
    document = CL.C.mapping_file(_run(fixture.launch))
    assert document["admission"]["consumable"] is False and document["admission"]["refusal"]
    assert document["functional_qualification_sha256"] is None
    assert document["final_acceptance"] == "NOT_ESTABLISHED"
    assert "qualify" not in fixture.observed and "verify" not in fixture.observed
    assert (
        fixture.inputs.stage_root / "sealed_candidate/compiler.py"
    ).read_text() == "# synthetic observed descendant edit\n"
    if failure == "unresolved":
        assert (fixture.inputs.stage_root / "final_compile_role_evaluation/diagnostic.json").is_file()
