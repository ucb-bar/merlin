"""Invocation-local loop grading, lazy policy and frozen implementation identity."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import loop_grading as G

from merlin.common.paths import module_source_path, python_import_roots


@pytest.mark.parametrize("missing", ["snapshot", "current"])
def test_missing_source_identity_never_qualifies_snapshot_pass(tmp_path, monkeypatch, missing):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "compiler.py").write_text("VALUE = 1\n")
    snapshot = G.hash_tree(submission)["sha256"]
    if missing == "snapshot":
        snapshot = None
    else:
        monkeypatch.setattr(G, "hash_tree", lambda path: (_ for _ in ()).throw(OSError("unavailable")))
    result = G._attribute_source(
        {"all_pass": True, "gradeable": True, "n_passed": 153, "n_capsules": 153},
        tmp_path,
        snapshot,
        stage="loop_screen",
    )
    assert result["snapshot_result"]["all_pass"] is True
    assert result["snapshot_result"]["n_passed"] == 153
    assert result["all_pass"] is None and result["gradeable"] is False
    assert result["snapshot_current"] is False and result["current_source_eligible"] is False
    assert result["current_source_n_passed"] is None


@pytest.mark.parametrize("changed", [False, True])
def test_archived_snapshot_pass_never_qualifies_different_current_bytes(tmp_path, monkeypatch, changed):
    from contextlib import nullcontext

    from merlin_experiments.phase1.feedback import promotion

    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    compiler = ws / "submission/compiler.py"
    compiler.write_text("VALUE = 1\n")
    inputs = G.GradingInputs(context(tmp_path), "raw_baseline", tmp_path / "public", (), None)
    monkeypatch.setattr(G, "compiler_execution", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(G.OH, "commit_graded", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: None)
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(
        G, "_attach_shape_generalization", lambda v, *a, **kw: v.update(shape_coverage={"all_covered": True})
    )
    monkeypatch.setattr(G, "_attach_codegen_scalability", lambda *a, **kw: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: (None, None, None))
    monkeypatch.setenv("MERLIN_BACKGROUND_FINISH_SNAPSHOT", "1")

    def qa(candidate, *args, **kwargs):
        assert (Path(candidate) / "compiler.py").read_text() == "VALUE = 1\n"
        if changed:
            compiler.write_text("VALUE = 2\n")
        return {
            "all_pass": True,
            "gradeable": True,
            "n_passed": 2,
            "n_capsules": 2,
            "structural_evidence_complete": True,
            "per_capsule": [{"capsule": n, "status": "pass"} for n in ("a", "b")],
        }

    result = G.grade(
        ws,
        run,
        901,
        False,
        10,
        label="inturn",
        scratch_key="r0_t1",
        inputs=inputs,
        qa_runner=qa,
        advisory_background=True,
    )
    archived = json.loads((run / "qa_history/verdict_inturn_r0_t1.json").read_text())
    completed = json.loads((run / "qa_history/completed_inturn_r0_t1.json").read_text())
    visible = json.loads((ws / "qa/verdict.json").read_text())
    assert "shape_coverage" not in archived
    for record in (archived, completed, visible, result):
        assert record["snapshot_result"]["all_pass"] is True
        assert record["snapshot_result"]["n_passed"] == record["n_passed"] == 2
        assert record["count_scope"] == "immutable_snapshot"
        assert record["snapshot_current"] is (not changed)
        assert record["current_source_eligible"] is (not changed)
        assert record["all_pass"] is (None if changed else True)
        assert record["current_source_n_passed"] == (None if changed else 2)
        assert (record["graded_source_sha256"] == record["current_source_sha256"]) is (not changed)
    plateau = json.loads((run / "plateau.json").read_text())
    assert plateau["snapshot_history"]["n_grades"] == 1  # Completed replaces pre-shape, not an extra grade.
    assert plateau["snapshot_history"]["best_passed"] == 2
    assert plateau["current_source_grade_available"] is (not changed)
    assert plateau["latest_passed"] == (None if changed else 2)
    if changed:
        assert "No grade" in plateau["sentence"]


def test_plateau_does_not_promote_legacy_or_other_source_snapshot_counts(tmp_path):
    history = tmp_path / "qa_history"
    history.mkdir()
    for i, record in enumerate(
        [
            {"all_pass": True, "n_passed": 153, "n_capsules": 153},
            {
                "all_pass": None,
                "n_passed": 153,
                "n_capsules": 153,
                "graded_source_sha256": "a" * 64,
                "snapshot_current": False,
            },
        ]
    ):
        (history / f"verdict_inturn_{i}.json").write_text(json.dumps(record))
    G._record_plateau(tmp_path, current_source_sha256="b" * 64)
    result = json.loads((tmp_path / "plateau.json").read_text())
    assert result["latest_passed"] is None and result["best_passed"] is None
    assert result["snapshot_history"]["best_passed"] == 153
    assert not result["current_source_grade_available"] and not result["stuck"]


def test_shape_pending_progress_drops_current_credit_after_last_capsule(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from merlin_experiments.phase1.feedback import promotion

    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    compiler = ws / "submission/compiler.py"
    compiler.write_text("VALUE = 1\n")
    inputs = G.GradingInputs(context(tmp_path), "raw_baseline", tmp_path / "public", (), None)
    monkeypatch.setattr(G, "compiler_execution", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(G.OH, "commit_graded", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: None)
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(G, "_attach_codegen_scalability", lambda *a, **kw: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: (None, None, None))
    monkeypatch.setenv("MERLIN_BACKGROUND_PROGRESS", "1")

    def qa(candidate, *args, **kwargs):
        kwargs["on_capsule_complete"]({"capsule": "a", "status": "pass", "tiers": {"L2": "pass"}})
        return {"all_pass": True, "n_passed": 1, "n_capsules": 1, "per_capsule": []}

    observed = []

    def shape(verdict, *args, **kwargs):
        compiler.write_text("VALUE = 2\n")
        kwargs["before_probe"]()
        observed.append(json.loads((ws / "qa/progress.json").read_text()))

    monkeypatch.setattr(G, "_attach_shape_generalization", shape)
    G.grade(ws, run, 901, False, 10, inputs=inputs, qa_runner=qa, advisory_background=True)
    assert len(observed) == 1
    assert observed[0]["state"] == "shape_coverage_pending"
    assert observed[0]["n_passed"] == 1 and observed[0]["snapshot_current"] is False
    assert observed[0]["current_source_n_passed"] is None


def context(root, target="fixture"):
    return InvocationContext(
        root, root / "descriptor.yaml", root, target, root / "runs", root / "reports", root / "bundles", ()
    )


@pytest.mark.parametrize("required", [True, False])
def test_selected_public_resources_publish_before_simulation_without_changing_capsule_counts(
    tmp_path,
    monkeypatch,
    required,
):
    from merlin_experiments.phase1.feedback import promotion, public_scalability_gate

    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    owner, facts = object(), tmp_path / "selected-facts.json"
    inputs = G.GradingInputs(
        context(tmp_path),
        "raw_baseline",
        tmp_path / "public",
        (),
        None,
        compiler_profile_service=owner,
        rtl_facts=facts,
    )
    from contextlib import nullcontext

    monkeypatch.setattr(G, "compiler_execution", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(G.OH, "commit_graded", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_record_plateau", lambda *a, **kw: None)
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(G, "_attach_shape_generalization", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_attach_codegen_scalability", lambda *a, **kw: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: (None, None, None))
    events = []
    observed = {
        "required": required,
        "resource_complete": False,
        "status": "incomplete",
        "samples": [{"member": "public_a", "scale": 64, "lowering_status": "unavailable"}],
    }

    def stress(**kwargs):
        assert kwargs["owner"] is owner and kwargs["facts_path"] == facts
        assert kwargs["candidate"] != ws / "submission"
        assert (kwargs["candidate"] / "manifest.yaml").is_file()
        assert kwargs["capsules_root"] == inputs.public_root
        events.append("selected public source stress")
        return observed

    monkeypatch.setattr(public_scalability_gate, "observe_authoring", stress)

    def numeric(*args, **kwargs):
        # This file exists before any simulator would be admitted, while its
        # resource result remains separate from the real capsule denominator.
        assert json.loads((ws / "qa/public_codegen_resources.json").read_text()) == observed
        events.append("numeric capsules")
        return {"all_pass": True, "n_capsules": 1, "n_passed": 1, "per_capsule": []}

    result = G.grade(ws, run, 0, False, 5, inputs=inputs, qa_runner=numeric)
    assert events == ["selected public source stress", "numeric capsules"]
    assert result["n_capsules"] == result["n_passed"] == 1
    assert result["all_pass"] is (not required)
    assert result["public_codegen_resources"] == observed


def test_cold_import_is_native_and_target_inert(tmp_path):
    script = """
import importlib.abc, os, subprocess, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_common", "run_baseline_qa_loop", "tooling_readiness"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoNative())
def refused(*args, **kwargs):
    raise AssertionError("import launched a process")
subprocess.run = subprocess.Popen = refused
before = dict(os.environ)
from merlin_experiments.phase1.feedback import loop_grading
assert dict(os.environ) == before
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
        MERLIN_TARGET_EXPERIMENT=str(tmp_path / "absent.yaml"),
        MERLIN_REPO_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", [False, True])
def test_cert_policy_does_not_materialize_without_admitted_extra_tiers(tmp_path, monkeypatch, failure):
    from merlin.targetgen import capsule_runner as runner

    monkeypatch.setattr(G, "load_target_experiment", lambda _: SimpleNamespace(target="fixture", sim_via="fixture"))
    monkeypatch.setattr(G, "declared_loop_tiers", lambda roots: {"L2"})

    def checkpoint(*args):
        if failure:
            raise ValueError("descriptor admission failed")
        return {"L2": object()}

    monkeypatch.setattr(runner, "qa_checkpoint_adapters", checkpoint)
    monkeypatch.setattr(runner, "qa_loop_adapters", lambda *a, **kw: {"L2": object()})
    assert G.cert_tiers_beyond_loop(
        context=context(tmp_path), policy_roots=(), public_roots=lambda: pytest.fail("eager materialization")
    ) == (set(), set())


def test_cert_policy_intersects_all_public_roots_after_derivation(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner as runner

    monkeypatch.setattr(G, "load_target_experiment", lambda _: SimpleNamespace(target="fixture", sim_via="fixture"))
    roots = tuple(tmp_path / name for name in ("a", "b"))
    for root, tier in zip(roots, ("L2", "L3")):
        root.mkdir()
        (root / "capsule.yaml").write_text(f"required_oracle_tiers: [{tier}]\n")
    assert G.declared_loop_tiers(roots) == {"L2", "L3"}
    events = []
    monkeypatch.setattr(
        runner, "qa_checkpoint_adapters", lambda *a: events.append("checkpoint") or {"L2": 1, "L3": 2, "L4": 3}
    )
    monkeypatch.setattr(
        runner, "qa_loop_adapters", lambda *a, **kw: events.append(("loop", kw["declared_tiers"])) or {"L2": 1}
    )
    result = G.cert_tiers_beyond_loop(
        context=context(tmp_path), policy_roots=roots, public_roots=lambda: events.append("public") or roots
    )
    assert result == ({"L3", "L4"}, {"L3"})
    assert events == ["checkpoint", ("loop", {"L2", "L3"}), "public"]


def test_interleaved_invocations_keep_snapshot_policy_order_and_publication(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import promotion

    monkeypatch.setenv("MERLIN_BACKGROUND_CODEGEN_SCALABILITY", "0")

    events = []
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: events.append("ledger"))

    def shape(verdict, *args, context, **kwargs):
        assert kwargs["additional_forbidden"] == config.additional_forbidden
        events.append(("shape", context.target))
        verdict["shape_coverage"] = {"all_covered": True}

    monkeypatch.setattr(G, "_attach_shape_generalization", shape)

    def scalability(verdict, *args, context, **kwargs):
        assert kwargs["additional_forbidden"] == config.additional_forbidden
        events.append(("scalability", context.target))
        verdict["codegen_scalability"] = {"ran": True, "scope": "public_emit_only"}

    monkeypatch.setattr(G, "_attach_codegen_scalability", scalability)
    monkeypatch.setattr(G, "_record_plateau", lambda *a, **kw: events.append("plateau"))
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: events.append("health"))

    def tiers(ws, *, context, capsules_root):
        events.append(("policy", context.target, capsules_root))
        return "L2", "L3", None

    monkeypatch.setattr(promotion, "resolve_tiers", tiers)

    def promote(ws, channel, verdict, *args, source_ws, context, capsules_root, **kwargs):
        assert verdict["shape_coverage"]["all_covered"]
        assert not (ws / "qa/verdict.json").exists()
        assert (source_ws / "submission/manifest.yaml").is_file()
        events.append(("promote", context.target))

    monkeypatch.setattr(promotion, "promote", promote)
    configs = [
        G.GradingInputs(
            context(tmp_path / name, name),
            "raw_baseline",
            tmp_path / name / "public",
            (tmp_path / name / "policy",),
            tmp_path / name / "contract",
            tmp_path / name / "policy",
            ("outputs_match",) if name == "first" else (),
        )
        for name in ("first", "second")
    ]
    with pytest.raises(FrozenInstanceError):
        configs[0].public_root = tmp_path
    for config in (configs[1], configs[0]):
        ws, run = config.context.repo / "ws", config.context.runs
        (ws / "submission/build").mkdir(parents=True)
        (ws / "submission/manifest.yaml").write_text("language: python\n")
        (ws / "submission/build/private").write_text("old-build-secret")

        def grade(candidate, public, runs, labels, no_oracle, timeout, **kwargs):
            from pathlib import Path

            assert Path(candidate) != ws / "submission"
            assert not (Path(candidate) / "build").exists()
            assert public == str(config.public_root)
            expected = {"contract": config.contract}
            if config.additional_forbidden:
                expected["additional_forbidden"] = config.additional_forbidden
            assert kwargs == expected
            from merlin.targetgen.package_runtime import CertFailure, integrity_scan

            (Path(candidate) / "marker.py").write_text("# outputs_match\n")
            package = SimpleNamespace(directory=Path(candidate), integrity_exempt=False)
            if config.additional_forbidden:
                with pytest.raises(CertFailure, match="outputs_match"):
                    integrity_scan(package, additional_forbidden=kwargs["additional_forbidden"])
            else:
                integrity_scan(package)
            events.append(("grade", config.context.target))
            return {"all_pass": True, "n_passed": 1, "n_capsules": 1, "per_capsule": [{"status": "pass"}]}

        resolutions = []
        lazy = replace(config, public_root=lambda: resolutions.append(config.public_root) or config.public_root)
        result = G.grade(ws, run, 901, False, 5, label="inturn", scratch_key="r0_t1", inputs=lazy, qa_runner=grade)
        assert resolutions == [config.public_root]
        public = json.loads((ws / "qa/verdict.json").read_text())
        assert public == result and public["graded_at"]
        archive = json.loads((run / "qa_history/verdict_inturn_r0_t1.json").read_text())
        assert "shape_coverage" not in archive  # Preserve historical pre-shape archive order.
        assert "old-build-secret" not in json.dumps(public)
        assert events == [
            ("grade", config.context.target),
            "ledger",
            ("shape", config.context.target),
            ("scalability", config.context.target),
            "plateau",
            "health",
            ("policy", config.context.target, config.promotion_root),
            ("promote", config.context.target),
        ]
        events.clear()


@pytest.mark.parametrize("finish_snapshot", [False, True])
def test_in_turn_grade_reaches_agent_after_source_changes(tmp_path, monkeypatch, finish_snapshot):
    from merlin_experiments.phase1.feedback import promotion

    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    (ws / "submission/compiler.py").write_text("VALUE = 1\n")
    inputs = G.GradingInputs(context(tmp_path), "raw_baseline", tmp_path / "public", (), None)
    monkeypatch.setenv("MERLIN_BACKGROUND_FINISH_SNAPSHOT", "1" if finish_snapshot else "0")
    monkeypatch.setenv("MERLIN_BACKGROUND_PROGRESS", "1")
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_attach_shape_generalization", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_attach_codegen_scalability", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_record_plateau", lambda *a, **kw: None)
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: (None, None, None))

    def grade_snapshot(candidate, *args, **kwargs):
        assert (Path(candidate) / "compiler.py").read_text() == "VALUE = 1\n"
        (ws / "submission/compiler.py").write_text("VALUE = 2\n")
        kwargs["on_capsule_complete"]({"capsule": "completed", "status": "pass", "tiers": {"L2": "pass"}})
        progress = json.loads((ws / "qa/progress.json").read_text())
        assert progress["n_completed"] == 1 and progress["n_passed"] == 1
        assert progress["all_pass"] is None and progress["certified_complete"] is False
        assert progress["snapshot_current"] is False
        assert progress["count_scope"] == "immutable_snapshot" and progress["current_source_n_passed"] is None
        assert not (ws / "qa/verdict.json").exists()
        assert kwargs["defer_pending"]() == "background source superseded"
        assert kwargs["defer_running"] is False
        return {"all_pass": True, "n_passed": 1, "n_capsules": 1, "per_capsule": []}

    G.grade(ws, run, 900, False, 10, label="inturn", inputs=inputs, qa_runner=grade_snapshot, advisory_background=True)
    visible = json.loads((ws / "qa/verdict.json").read_text())
    assert visible["snapshot_current"] is False
    assert visible["all_pass"] is None
    assert visible["gradeable"] is False
    assert visible["background_grade"]["reason"] == "background source superseded"
    assert len(visible["graded_source_sha256"]) == 64
    progress = json.loads((ws / "qa/progress.json").read_text())
    assert progress["state"] == "complete" and progress["all_pass"] is None


def test_selected_object_build_reaches_grade_and_candidate_matched_continuous_brief(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import brief, promotion
    from merlin_experiments.phase1.feedback import codegen_scalability as S

    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    service = object()
    private_selection = {"selected_tool": "/host/private/tool", "sha256": "a" * 64}
    seen = []
    monkeypatch.setattr(S, "_selected_build_inputs", lambda actual, target: private_selection)

    def observe(candidate, **kwargs):
        seen.append(kwargs)
        assert kwargs["build_service"] is service
        assert kwargs["build_timeout_s"] == 7
        return {
            "schema": "merlin.codegen_scalability_probe.v2",
            "samples": [
                {
                    "extent_scale": scale,
                    "outcome": "lowered",
                    "build_only": {
                        "status": "compiled",
                        "build_selection": private_selection,
                        "artifact_bytes": 100 * scale,
                        "object_bytes": 20 * scale,
                        "compile_wall_s": 0.25,
                    },
                }
                for scale in (1, 2, 4, 8)
            ],
        }

    monkeypatch.setattr(S, "run", observe)
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_attach_shape_generalization", lambda *a, **kw: None)
    monkeypatch.setattr(G, "_record_plateau", lambda *a, **kw: None)
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: (None, None, None))
    inputs = G.GradingInputs(
        context(tmp_path),
        "raw_baseline",
        tmp_path / "public",
        (),
        None,
        public_build_service=service,
        public_build_selection=private_selection,
        public_build_budget_s=7,
    )
    result = G.grade(
        ws,
        run,
        0,
        False,
        10,
        scratch_key="r0000_t000007",
        inputs=inputs,
        qa_runner=lambda *a, **kw: {"all_pass": False, "n_passed": 0, "n_capsules": 1, "per_capsule": []},
    )
    archive = json.loads((run / "qa_history/verdict_round_r0000_t000007.json").read_text())
    host = json.loads((run / "qa_history/codegen_scalability_round_r0000_t000007.json").read_text())
    public = json.loads((run / "qa_history/codegen_scalability_public_round_r0000_t000007.json").read_text())
    assert len(seen) == 1 and archive["graded_candidate_sha256"] == public["candidate_sha256"]
    assert host["graded_candidate_sha256"] == public["candidate_sha256"]
    assert private_selection["selected_tool"] in json.dumps(host)
    assert private_selection["selected_tool"] not in json.dumps(result)
    assert result["codegen_scalability"] == public
    text = brief.build(run, ws, 0)
    assert "Public object-build scalability (advisory only)" in text
    assert "| 8 | lowered | compiled | 800 | 160 | 0.25 |" in text
    assert private_selection["selected_tool"] not in text

    # A grader-side mutation invalidates the pre-grade candidate identity;
    # the later round cannot borrow this candidate's earlier projection.
    def mutate(candidate, *args, **kwargs):
        (Path(candidate) / "postgrade.py").write_text("changed after candidate selection\n")
        return {"all_pass": False, "n_passed": 0, "n_capsules": 1, "per_capsule": []}

    later = G.grade(ws, run, 0, False, 10, scratch_key="r0000_t000008", inputs=inputs, qa_runner=mutate)
    assert len(seen) == 1
    assert later["codegen_scalability"]["ran"] is False
    assert "Public object-build scalability" not in brief.build(run, ws, 0)


def test_source_receipt_rejects_loop_grading_drift(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    package = tmp_path / "phase1"
    shutil.copytree(
        module_source_path("merlin_experiments.phase1").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    original = SI._source

    def copied_source(module):
        if module == "merlin_experiments.phase1":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            path = package.joinpath(*module.split(".")[2:])
            return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        return original(module)

    monkeypatch.setattr(SI, "_source", copied_source)
    arguments = {"repo": tmp_path, "entrypoint": tmp_path / "transport.py"}
    receipt = SI.record(**arguments)
    member = package / "feedback/loop_grading.py"
    assert receipt["inputs"]["phase1:source:feedback/loop_grading.py"]["path"] == str(member)
    SI.verify(receipt, **arguments)
    member.write_text(member.read_text() + "\n# grading drift\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(receipt, **arguments)


@pytest.mark.parametrize("manifest", [None, "language: cpp\n"])
def test_submission_refusal_does_not_resolve_public_corpus(tmp_path, monkeypatch, manifest):
    from merlin_experiments.phase1.feedback import promotion

    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    if manifest is not None:
        (ws / "submission/manifest.yaml").write_text(manifest)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: (None, None, None))
    config = G.GradingInputs(
        context(tmp_path), "merlin_assisted", lambda: pytest.fail("materialized refused submission"), (), None
    )
    verdict = G.grade(ws, tmp_path / "run", 0, False, 3, inputs=config)
    assert verdict["n_capsules"] == 0 and verdict["all_pass"] is None
    assert verdict["snapshot_result"]["all_pass"] is False
    assert verdict["current_source_eligible"] is False
    assert verdict["package_failure"]["plane"] == ("schema" if manifest is None else "language")
    if manifest is None:
        with pytest.raises(RuntimeError, match="no submission"):
            G.fast_grade(ws, tmp_path / "run", 0, 3, inputs=config)


def test_native_lazy_selection_binds_descriptor_and_cache_per_invocation(tmp_path, monkeypatch):
    import importlib

    from merlin_experiments.corpus import admission

    from merlin.common.paths import checkout_root
    from merlin.targetgen import target_experiment

    monkeypatch.syspath_prepend(str(checkout_root() / "merlin/experiments/capsule_bench/harness"))
    native = importlib.import_module("run_baseline_qa_loop")
    monkeypatch.setattr(native, "PILOT_SUBSET", None)
    seen = []
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda descriptor: descriptor)
    monkeypatch.setattr(
        admission, "public_capsules_for", lambda descriptor: seen.append(descriptor) or descriptor.parent / "public"
    )
    first, second = context(tmp_path / "first"), context(tmp_path / "second")
    from merlin_experiments.phase1 import authoring

    a, b = authoring._grading_public_root(first), authoring._grading_public_root(second)
    monkeypatch.setattr(native, "PILOT_SUBSET", tmp_path / "later-global-override")
    assert seen == []
    assert b() == second.repo / "public"
    assert a() == a() == first.repo / "public"
    assert seen == [second.descriptor, first.descriptor]
    explicit = tmp_path / "admitted"
    assert authoring._grading_public_root(first, explicit) == explicit


@pytest.mark.parametrize("shared_compute", [False, True])
def test_fast_grade_uses_explicit_target_roots_and_withheld_tier(tmp_path, monkeypatch, shared_compute):

    if shared_compute:
        cpu = min(os.sched_getaffinity(0))
        monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
        monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "compute"))
        actual_copytree = G.shutil.copytree

        def copied(*args, **kwargs):
            import fcntl

            with (tmp_path / "compute" / f"cpu-{cpu}.lock").open("r+") as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return actual_copytree(*args, **kwargs)

        monkeypatch.setattr(G.shutil, "copytree", copied)
    from merlin_experiments.phase1.feedback import qa

    from merlin.targetgen import capsule_grade, capsule_runner

    inputs = G.GradingInputs(
        context(tmp_path, "isolated"),
        "raw_baseline",
        tmp_path / "public",
        (tmp_path / "policy",),
        tmp_path / "contract",
    )
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    seen = []

    def descriptor(path):
        assert path == inputs.context.descriptor
        return SimpleNamespace(target="isolated", sim_via="explicit")

    monkeypatch.setattr(G, "load_target_experiment", descriptor)
    monkeypatch.setattr(
        G, "declared_loop_tiers", lambda roots: {"L2"} if roots == inputs.policy_roots else pytest.fail("wrong policy")
    )
    monkeypatch.setattr(
        capsule_runner,
        "qa_loop_adapters",
        lambda target, via, **kw: (
            {"L2": "loop"} if (target, via) == ("isolated", "explicit") else pytest.fail("wrong target")
        ),
    )
    monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *a: {"L2": "loop", "L3": "cert"})
    inputs.public_root.mkdir()
    (inputs.public_root / "capsule.yaml").write_text("required_oracle_tiers: [L2, L3]\n")

    def grade(candidate, **kwargs):
        # The parent copy/discovery leases have closed before this grade can
        # invoke independent compiler/child owners requiring the same core.
        if shared_compute:
            import fcntl

            with (tmp_path / "compute" / f"cpu-{cpu}.lock").open("r+") as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen.append(kwargs)
        assert candidate != str(ws / "submission")

    monkeypatch.setattr(capsule_grade, "grade", grade)
    monkeypatch.setattr(
        qa, "_per_capsule_from_results", lambda root: {"public": {"status": "pass", "tiers": {"L2": "pass"}}}
    )
    from merlin_experiments.phase1.feedback import promotion

    promoted = []
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: ("L2", "L3", {"public"}))

    def promote(*args, source_ws, context, capsules_root):
        assert (ws / "qa/verdict.json").is_file()
        assert source_ws != ws
        assert (source_ws / "submission/manifest.yaml").read_text() == "language: python\n"
        assert context == inputs.context
        assert capsules_root == inputs.promotion_root
        promoted.append(args[2]["per_capsule"])

    monkeypatch.setattr(promotion, "promote", promote)
    verdict = G.fast_grade(ws, tmp_path / "run", 900, 10, inputs=inputs)
    assert promoted and promoted[0][0]["tiers"] == {"L2": "pass"}
    assert verdict["all_pass"] is None
    assert seen[0]["target"] == "isolated"
    assert seen[0]["capsules_root"] == str(inputs.public_root)
    assert seen[0]["contract"] == str(inputs.contract)
    assert seen[0]["oracle_adapters"] == {"L2": "loop"}
    assert json.loads((ws / "qa/verdict.json").read_text()) == verdict


def test_nested_policy_cert_cover_keeps_public_dev_exclusions_and_real_empty(tmp_path, monkeypatch):
    import yaml
    from merlin_experiments.phase1.feedback import promotion

    from merlin.targetgen import capsule_common, target_experiment

    policy = tmp_path / "policy"
    for index, (name, label, family) in enumerate(
        (
            ("public_case", "public", "contraction"),
            ("dev_case", "dev", "elementwise_map"),
            ("hidden_case", "hidden", "reduction"),
        )
    ):
        leaf = policy / str(index) / name
        leaf.mkdir(parents=True)
        (leaf / "capsule.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "label": label,
                    "inputs": [{"dtype": "int8", "shape": [16, 16]}],
                    "semantic": {"semantic_family": family},
                }
            )
        )
    monkeypatch.setattr(
        capsule_common,
        "load_capsule",
        lambda directory, **kw: {
            **yaml.safe_load((Path(directory) / "capsule.yaml").read_text()),
            "__dir__": str(directory),
        },
    )
    excluded = set()
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda _: SimpleNamespace(
            target="fixture",
            effective_exclusions=lambda _: excluded,
        ),
    )
    assert promotion._cert_cover(tmp_path, context=context(tmp_path), capsules_root=policy) == {
        "public_case",
        "dev_case",
    }
    excluded.update(("public_case", "dev_case"))
    assert promotion._cert_cover(tmp_path, context=context(tmp_path), capsules_root=policy) == set()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert promotion._cert_cover(tmp_path, context=context(tmp_path), capsules_root=empty) == set()


def test_mandatory_cert_members_respect_selected_cohort_labels_and_exclusions(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import promotion

    from merlin.targetgen import capsule_common, target_experiment

    caps = [
        {"name": "mandatory", "label": "public", "required_oracle_tiers": ["L2", "L3"]},
        {"name": "optional", "label": "dev", "required_oracle_tiers": ["L2"]},
        {"name": "excluded", "label": "public", "required_oracle_tiers": ["L3"]},
    ]
    selected = tmp_path / "selected-functional"

    def discover(roots, **kw):
        assert roots == (selected,)
        assert kw["labels"] == {"public", "dev"}
        return caps

    monkeypatch.setattr(capsule_common, "discover_capsules", discover)
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda _: SimpleNamespace(
            effective_exclusions=lambda names: {"excluded"},
            graded_roots=lambda: pytest.fail("selected root was ignored"),
            perf_roots=lambda: pytest.fail("performance roots were inspected"),
        ),
    )
    assert promotion._mandatory_cert_members(context=context(tmp_path), capsules_root=selected, cert_tier="L3") == {
        "mandatory"
    }


@pytest.mark.parametrize("producer", ["fast_grade", "selfcheck"])
def test_completed_loop_pass_queues_before_batch_finishes_from_frozen_source(tmp_path, monkeypatch, producer):
    import threading

    from merlin_experiments.phase1.feedback import promotion, qa

    from merlin.targetgen import capsule_common, capsule_scheduling

    ws, frozen, runs = (tmp_path / name for name in ("ws", "frozen", "runs"))
    for root in (ws, frozen):
        (root / "submission").mkdir(parents=True)
        (root / "submission/manifest.yaml").write_text("x: 1\n")
    (ws / ".qa_channel").mkdir()
    ctx = context(tmp_path)
    ctx.descriptor.write_text("{}\n")
    inputs = G.GradingInputs(
        context=ctx,
        arm="raw_baseline",
        public_root=tmp_path / "public",
        policy_roots=(),
        promotion_root=tmp_path / "policy",
        contract=tmp_path / "contract",
    )
    monkeypatch.setattr(
        G,
        "load_target_experiment",
        lambda _: SimpleNamespace(target="fixture", effective_exclusions=lambda names: {"excluded"}),
    )
    monkeypatch.setattr(
        capsule_common,
        "discover_capsules",
        lambda *a, **kw: [{"name": "first"}, {"name": "second"}, {"name": "failed"}, {"name": "excluded"}],
    )
    monkeypatch.setattr(G, "_suite_for", lambda _: "fixture-suite")
    monkeypatch.setattr(qa, "_execution_digest_from_result", lambda _: None)
    monkeypatch.setattr(promotion, "affordable_cert_items", lambda want, **kw: (want, {}))
    monkeypatch.setattr(promotion, "_mandatory_cert_members", lambda **kw: {"first", "second"})
    if producer == "fast_grade":
        callback = G._completed_loop_promoter(ws, frozen, runs, "L2", "L3", {"first"}, inputs=inputs)
    else:
        from merlin_experiments.phase1.brokers.selfcheck import _submission_digest
        from merlin_experiments.phase1.feedback import selfcheck

        monkeypatch.setenv("MERLIN_SELFCHECK_PROMOTION_WS", str(ws))
        monkeypatch.setenv("MERLIN_SELFCHECK_PROMOTION_ROOT", str(inputs.promotion_root))
        monkeypatch.setenv("MERLIN_SELFCHECK_PROMOTION_SHA256", _submission_digest(frozen / "submission"))
        monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **kw: ("L2", "L3", {"first"}))
        callback = selfcheck._streaming_completion(
            context=ctx,
            capsules_root=inputs.public_root,
            contract=inputs.contract,
            source=frozen / "submission",
            runs=runs,
            adapters={"L2": object()},
        )
    second_started, release_second = threading.Event(), threading.Event()
    rows = {}

    def run_one(cap, workers):
        name = cap["name"]
        if name == "second":
            second_started.set()
            assert release_second.wait(5)
        row = {"capsule": name, "tiers": {"L2": {"status": "fail" if name == "failed" else "pass"}}}
        path = runs / "runs/fixture-suite" / name / "capsule_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))
        rows[name] = row
        return row

    def completed(row):
        callback(row)
        if row["capsule"] == "first":
            assert second_started.wait(5)
            reqs = list((ws / ".qa_channel").glob("simreq_*.json"))
            assert len(reqs) == 1, "first pass must queue while second capsule is unfinished"
            req = json.loads(reqs[0].read_text())
            assert req["submission_digest"] == promotion._submission_digest(frozen)
            (ws / "submission/manifest.yaml").write_text("live edit: true\n")
            assert req["submission_digest"] != promotion._submission_digest(ws)
            callback(row)  # The same completed identity cannot enqueue twice.
            assert len(list((ws / ".qa_channel").glob("simreq_*.json"))) == 1
            release_second.set()

    capsule_scheduling.run_pipelined(
        [{"name": name} for name in ("first", "second", "failed", "hidden", "excluded")],
        run_one,
        max_workers=2,
        on_complete=completed,
    )
    # The second frozen result is retained but cannot replenish native work
    # after the author genuinely changes the reachable source.
    assert len(list((ws / ".qa_channel").glob("simreq_*.json"))) == 1
    # A result that has not been durably published cannot unlock a certificate.
    callback({"capsule": "first", "tiers": {"L2": {"status": "pass"}}, "tampered": True})
    assert len(list((ws / ".qa_channel").glob("simreq_*.json"))) == 1
