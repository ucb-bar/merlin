"""Invocation-local loop grading, lazy policy and frozen implementation identity."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import loop_grading as G

from merlin.common.paths import module_source_path, python_import_roots


def context(root, target="fixture"):
    return InvocationContext(
        root, root / "descriptor.yaml", root, target, root / "runs", root / "reports", root / "bundles", ()
    )


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

    events = []
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **kw: events.append("ledger"))

    def shape(verdict, *args, context, **kwargs):
        assert kwargs["additional_forbidden"] == config.additional_forbidden
        events.append(("shape", context.target))
        verdict["shape_coverage"] = {"all_covered": True}

    monkeypatch.setattr(G, "_attach_shape_generalization", shape)
    monkeypatch.setattr(G, "_record_plateau", lambda *a: events.append("plateau"))
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
            "plateau",
            "health",
            ("policy", config.context.target, config.promotion_root),
            ("promote", config.context.target),
        ]
        events.clear()


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
    assert verdict["n_capsules"] == 0 and verdict["all_pass"] is False
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


def test_fast_grade_uses_explicit_target_roots_and_withheld_tier(tmp_path, monkeypatch):
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
        seen.append(kwargs)
        assert candidate != str(ws / "submission")

    monkeypatch.setattr(capsule_grade, "grade", grade)
    monkeypatch.setattr(
        qa, "_per_capsule_from_results", lambda root: {"public": {"status": "pass", "tiers": {"L2": "pass"}}}
    )
    verdict = G.fast_grade(ws, tmp_path / "run", 900, 10, inputs=inputs)
    assert verdict["all_pass"] is None
    assert seen[0]["target"] == "isolated"
    assert seen[0]["capsules_root"] == str(inputs.public_root)
    assert seen[0]["contract"] == str(inputs.contract)
    assert seen[0]["oracle_adapters"] == {"L2": "loop"}
    assert json.loads((ws / "qa/verdict.json").read_text()) == verdict
