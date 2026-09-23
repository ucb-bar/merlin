"""Invocation-local bundle selection through real native callers, without agent execution.

Admission is intercepted only where explicitly noted; the separate controller integration
fixture qualifies fresh/resumed admission. These tests do not claim concurrent host environments.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import session, treatments, workspace_transport
from merlin_experiments.phase1.providers import execution

from merlin.common.digest import sha256_file
from merlin.common.paths import repo_root

HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"


@pytest.fixture
def loop(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(HARNESS))
    module = importlib.import_module("run_baseline_qa_loop")
    from dataclasses import replace

    from merlin_experiments.phase1 import task_staging as TS

    # Actual main still owns process environment setup; isolate those writes in this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(module.C, "BUNDLES", tmp_path / "bundles")
    monkeypatch.setattr(module.C, "EXP", tmp_path / "experiment")
    task = module.C.EXP / "task/TASK_realistic.md"
    task.parent.mkdir(parents=True)
    task.write_text("Shared authored task\n")
    monkeypatch.setattr(module.C, "CONTEXT", replace(module.C.CONTEXT, experiment=module.C.EXP))
    monkeypatch.setattr(
        TS,
        "load_target_experiment",
        lambda *_: SimpleNamespace(
            target="fixture",
            resource_path=lambda member: module.C.EXP / member,
        ),
    )
    monkeypatch.setattr(TS, "task_runtime_scope_block", lambda *a, **k: "\nRuntime scope\n")
    monkeypatch.setattr(workspace_transport.subprocess, "run", lambda *a, **k: pytest.fail("unexpected child"))
    return module


def bundle(root: Path, name: str, tools: str = "") -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "tools.txt").write_text(tools)
    (path / "input_bundle_manifest.yaml").write_text(yaml.safe_dump({"bundle_id": name, "allowed": []}))
    (path / "STARTER_PROMPT.md").write_text("starter " + name)
    (path / "TASK_ADDENDUM.md").write_text("addendum " + name)
    (path / "allowed_files.txt").write_text("merlin/" + name + "\n")
    (path / "denied_files.txt").write_text("private/" + name + "\n")
    return path


def test_main_callbacks_keep_selected_bundle_after_another_invocation(loop, monkeypatch, tmp_path, capsys):
    first = bundle(loop.C.BUNDLES, "first", "isa_tools\ncca_tools\n")
    second = bundle(loop.C.BUNDLES, "second", "rtl_facts\n")
    captured = []

    def prepare(request, transport, stage_task, **kwargs):
        captured.append((request, stage_task))
        return 37  # Stop before admission/agent work, retaining the real main's callbacks.

    monkeypatch.setattr(session, "prepare", prepare)
    defaults = dict(loop.RX.ARM_BUNDLE)
    for selected, flags in (
        (first, ["--with-tool", "eqsat_seam", "--without-tool", "cca_tools"]),
        (second, []),
    ):
        assert (
            loop.main(
                [
                    "--run-id",
                    selected.name,
                    "--arm",
                    "raw_baseline",
                    "--experiment",
                    "realistic",
                    "--bundle",
                    str(selected),
                    *flags,
                ]
            )
            == 37
        )
    # Regression: the banner used to inspect the missing default BEFORE applying --bundle.
    assert "tools ['isa_tools', 'eqsat_seam']" in capsys.readouterr().out
    assert dict(loop.RX.ARM_BUNDLE) == defaults
    assert captured[0][0].resolved_tools() == ("isa_tools", "eqsat_seam")
    assert captured[1][0].resolved_tools() == ("rtl_facts",)
    from merlin_experiments.phase1 import task_staging as TS

    sources = {"inputs": {"staging": {"path": TS.__file__, "sha256": sha256_file(Path(TS.__file__))}}}
    for (request, stage), selected in zip(captured, (first, second), strict=True):
        assert request.bundle_manifest == selected / "input_bundle_manifest.yaml"
        for callback in (request.resolved_tools, stage):
            assert treatments.callback_reference(callback, sources, label="test")["source_input"] == "staging"
        workspace, run = tmp_path / (selected.name + "-ws"), tmp_path / (selected.name + "-run")
        workspace.mkdir()
        run.mkdir()
        stage("raw_baseline", workspace, run, sandbox="bwrap", task_scope={}, policy_root=None)
        text = (workspace / "TASK.md").read_text()
        assert "starter " + selected.name in text
        assert (run / "TASK.md").read_text() == text
        assert TS.granted_merlin_tools(selected) == {"merlin/" + selected.name}
    # Preserve observation timing: callback selection is fixed, file contents are not cached.
    (first / "tools.txt").write_text("cca_tools\nverify_seam\n")
    assert captured[0][0].resolved_tools() == ("verify_seam", "eqsat_seam")
    assert captured[1][0].resolved_tools() == ("rtl_facts",)
    default = bundle(loop.C.BUNDLES, defaults["raw_baseline"], "cca_tools\n")
    assert loop.main(["--run-id", "default", "--arm", "raw_baseline"]) == 37
    assert captured[2][0].bundle_manifest == default / "input_bundle_manifest.yaml"
    assert captured[2][0].resolved_tools() == ("cca_tools",)
    with pytest.raises(TypeError):
        loop.RX.ARM_BUNDLE["raw_baseline"] = "mutated"


@pytest.mark.parametrize("option", [[], ["--bundle", "operator"], ["--bundle=operator"], ["--bund=operator"]])
def test_eqsat_default_is_local_and_explicit_bundle_wins(loop, monkeypatch, option):
    defaults = dict(loop.RX.ARM_BUNDLE)
    eqsat = importlib.import_module("run_eqsat_qa_loop")
    calls = []
    monkeypatch.setattr(loop, "main", lambda argv: calls.append(argv) or 12)
    assert eqsat.main(["--run-id", "run", *option]) == 12
    from merlin_experiments.phase1.options import parse_options

    parsed = parse_options(calls[0], default_arm="raw_baseline")
    assert parsed.bundle == ("operator" if option else eqsat._EQSAT_BUNDLE)
    assert parsed.arm == "merlin_assisted"
    assert dict(loop.RX.ARM_BUNDLE) == defaults


def test_canary_loads_explicit_bundle_without_changing_default(loop, monkeypatch, tmp_path):
    selected = bundle(loop.C.BUNDLES, "canary")
    canary = importlib.import_module("codex_canary")
    from merlin.common import artifacts

    monkeypatch.setattr(artifacts, "cache_dir", lambda *a: tmp_path / "canary-output")
    defaults = dict(loop.RX.ARM_BUNDLE)

    def stop(document, workspace, **kwargs):
        assert document["bundle_id"] == selected.name
        raise RuntimeError("fixture stops before provider")

    monkeypatch.setattr(workspace_transport, "assemble_copy_workspace", stop)
    with pytest.raises(RuntimeError, match="fixture stops"):
        canary.run_canary(arm="raw_baseline", bundle_id=str(selected))
    assert dict(loop.RX.ARM_BUNDLE) == defaults


def test_fullsuite_task_uses_explicit_bundle_and_preserves_bytes(loop, tmp_path, monkeypatch):
    full = importlib.import_module("run_fullsuite")
    selected = bundle(loop.C.BUNDLES, "fullsuite")
    monkeypatch.setattr(full, "TASK_FULL", tmp_path / "TASK_full.md")
    full.TASK_FULL.write_text("full task\n")
    workspace, run = tmp_path / "ws", tmp_path / "run"
    workspace.mkdir()
    run.mkdir()
    full._full_build("merlin_assisted", workspace, run, bundle_dir=selected)
    assert (workspace / "TASK.md").read_text() == "full task\n\n\n---\n\naddendum fullsuite"
    assert (run / "TASK.md").read_bytes() == (workspace / "TASK.md").read_bytes()


def test_smoke_resolves_its_own_bundle_before_any_provider(loop, monkeypatch):
    smoke = importlib.import_module("smoke_agent_check")
    selected = bundle(loop.C.BUNDLES, smoke.ARM_BUNDLE["baseline"], "isa_tools\n")
    from merlin.targetgen import target_experiment

    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda *a: SimpleNamespace(
            target="fixture",
            sim_via="",
            capsule_corpus="missing-corpus",
        ),
    )
    monkeypatch.setattr(smoke.WT, "assemble_workspace", lambda doc, ws, **kwargs: ws.mkdir())
    monkeypatch.setattr(execution, "sandbox_command", lambda *a, **k: "unused")

    def stop(workspace, config):
        assert config.tools == ("isa_tools",)
        assert config.timing_file == loop.C.HARNESS / ".oracle_timing.json"
        assert selected.is_dir()
        raise RuntimeError("fixture stops before broker/provider")

    monkeypatch.setattr(smoke.FL, "start_brokers", stop)
    with pytest.raises(RuntimeError, match="fixture stops"):
        smoke.main(["--arm", "baseline"])


@pytest.mark.parametrize("arm_arg", [["--arm", "raw_baseline"], ["--arm=raw_baseline"]])
def test_eqsat_preserves_explicit_other_arm_default(loop, monkeypatch, arm_arg):
    eqsat = importlib.import_module("run_eqsat_qa_loop")
    calls = []
    monkeypatch.setattr(loop, "main", lambda argv: calls.append(argv) or 0)
    assert eqsat.main(["--run-id", "run", *arm_arg]) == 0
    from merlin_experiments.phase1.options import parse_options

    selected = parse_options(calls[0], default_arm="raw_baseline")
    assert selected.arm == "raw_baseline"
    assert selected.bundle == ""


def test_finalize_audits_selected_bundle_not_default(loop, tmp_path, monkeypatch):
    from merlin_experiments.phase1 import authoring
    from merlin_experiments.phase1.audit import AnswerAudit

    selected = bundle(loop.C.BUNDLES, "selected")
    workspace, run = tmp_path / "ws", tmp_path / "run"
    (workspace / "submission").mkdir(parents=True)
    (workspace / "qa").mkdir()
    run.mkdir()
    observed = []

    policy = AnswerAudit((), (), (), selected)

    def audit(self, path, arm, *, workspace):
        observed.append(self._bundle_grants(arm))
        return {"clean": True, "hits": []}

    monkeypatch.setattr(AnswerAudit, "audit_transcript", audit)
    authoring.finalize_report(
        workspace,
        run,
        "unused",
        "low",
        "none",
        {},
        "raw_baseline",
        {"all_pass": True},
        1,
        provider=execution.ProviderConfig(driver="codex"),
        context=loop.C.CONTEXT,
        audit=policy,
        grade_callback=lambda *a: {"all_pass": True},
    )
    assert observed == [(("merlin/selected",), ("private/selected",))]


@pytest.mark.parametrize("token", ["plain", "merlin_assisted_rtlchecks_public_v0", "/rtlchecks-parent/plain"])
def test_checkpoint_preserves_original_token_advisory_selection(loop, monkeypatch, tmp_path, token):
    from merlin_experiments.phase1 import authoring

    tree = ast.parse(Path(authoring.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    branch = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "bundle_id.find('rtlchecks') >= 0"
    )
    calls = []
    gate = SimpleNamespace(gated_adapter=lambda adapter, **kwargs: calls.append(kwargs) or adapter)
    monkeypatch.setitem(sys.modules, "merlin.targetgen.circt_gate", gate)
    from merlin import targetgen

    monkeypatch.setattr(targetgen, "circt_gate", gate, raising=False)
    scope = {
        "bundle_id": token,
        "adapters": {"L3": object()},
        "_te_ck": SimpleNamespace(target="fixture"),
        "run_dir": tmp_path,
        "attempt": 1,
    }
    exec(compile(ast.Module(body=[branch], type_ignores=[]), loop.__file__, "exec"), scope)
    assert bool(calls) == ("rtlchecks" in token)
