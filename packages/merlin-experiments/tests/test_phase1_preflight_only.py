"""No-author continuation controls; native admission owners are substituted.

These tests establish selection and dispatch, not real sandbox/oracle readiness.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import preflight as P
from merlin_experiments.phase1 import session as S
from merlin_experiments.phase1.options import parse_options


def _options(tmp_path):
    seal = tmp_path / "seal.json"
    seal.write_text("owned source-control seal\n")
    return parse_options(
        ["--run-id", "preflight", "--preflight-only", "--corpus-seal", str(seal), "--bundle", "selected"],
        environ={},
    )


def test_legacy_selection_and_receipt_mode_are_unchanged():
    options = parse_options(["--run-id", "ordinary"], environ={})
    assert not options.preflight_only and not options.corpus_seal
    S.validate_preflight_options(options)
    request = S.RunRequest(
        context=SimpleNamespace(readback_policy=None),
        options=options,
        treatment=None,
        bundle_manifest=Path("bundle"),
        launcher_argv=(),
        source_entrypoint=Path("source"),
        require_native_source=False,
        account={},
        resolved_tools=lambda: (),
    )
    assert request.run_config["session_mode"] == "certified_continuous"


@pytest.mark.parametrize(
    "changed",
    [
        {"resume": True},
        {"seed_submission": "/candidate"},
        {"qualify_submission": "/candidate"},
        {"seal_current": True},
        {"continuous": True},
        {"no_oracle": True},
        {"skip_hidden": True},
        {"sandbox": "none"},
        {"allow_unsandboxed": True},
        {"corpus_seal": ""},
        {"bundle": ""},
        {"corpus_seal": "relative-seal.json"},
    ],
)
def test_missing_or_bypass_selection_refuses_before_native_probe(tmp_path, monkeypatch, changed):
    from merlin.targetgen.sandbox import preflight

    monkeypatch.setattr(preflight, "require_working_sandbox", lambda **kw: pytest.fail("native probe started"))
    with pytest.raises(ValueError):
        S.validate_options(dataclasses.replace(_options(tmp_path), **changed))


def test_symlink_and_absent_seal_refuse(tmp_path):
    options = _options(tmp_path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(options.corpus_seal)
    for source in (alias, tmp_path / "absent.json"):
        with pytest.raises(ValueError, match="canonical corpus seal"):
            S.validate_preflight_options(dataclasses.replace(options, corpus_seal=str(source)))


def test_explicit_seal_must_agree_with_existing_environment_selection(tmp_path, monkeypatch):
    options = _options(tmp_path)
    monkeypatch.setenv("MERLIN_CORPUS_SEAL", options.corpus_seal)
    S.validate_preflight_options(options)
    monkeypatch.setenv("MERLIN_CORPUS_SEAL", str(tmp_path / "another-seal.json"))
    with pytest.raises(ValueError, match="environment selection"):
        S.validate_preflight_options(options)
    # An absent explicit selection preserves the legacy environment-owned route.
    S.validate_preflight_options(parse_options(["--run-id", "legacy"], environ={}))


def test_operator_account_and_errata_remain_normal_verified_inputs(tmp_path):
    S.validate_preflight_options(
        dataclasses.replace(
            _options(tmp_path),
            account_config_dir="/operator/account",
            operator_errata="/operator/correction",
        )
    )


@pytest.mark.parametrize("bundle", [{"bundle_id": "other"}, [], {"bundle_id": "selected"}])
def test_actual_prepare_checks_selected_manifest_before_normal_review(tmp_path, monkeypatch, bundle):
    options = _options(tmp_path)
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    request = S.RunRequest(
        context=SimpleNamespace(readback_policy=None, descriptor=tmp_path / "descriptor.yaml"),
        options=options,
        treatment=None,
        bundle_manifest=manifest,
        launcher_argv=(),
        source_entrypoint=Path(S.__file__),
        require_native_source=False,
        account={},
        resolved_tools=lambda: (),
    )
    target = object()
    calls = []
    # Stop at the actual normal review boundary, before any mutation or native work.
    monkeypatch.setattr(S, "validate_options", S.validate_preflight_options)
    monkeypatch.setattr(S, "load_target_experiment", lambda path: target)
    monkeypatch.setattr(S, "phase_run_dir", lambda *a, **kw: pytest.fail("workspace preparation started"))

    def review(actual_target, actual_manifest, actual_bundle):
        assert (actual_target, actual_manifest, actual_bundle) == (target, manifest, bundle)
        calls.append("review")
        raise RuntimeError("normal reviewed-bundle boundary reached")

    monkeypatch.setattr(S.CI, "require_reviewed_bundle", review)
    if bundle == {"bundle_id": "selected"}:
        with pytest.raises(RuntimeError, match="normal reviewed-bundle boundary"):
            S.prepare(request, None, None, workspace_leases=[])
        assert calls == ["review"]
    else:
        with pytest.raises(ValueError, match="bundle identity"):
            S.prepare(request, None, None, workspace_leases=[])
        assert not calls


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    options = _options(tmp_path)
    run, ws = tmp_path / "private-run", tmp_path / "workspace"
    run.mkdir()
    ws.mkdir()
    bundle = {"bundle_id": "selected", "allowed": [], "denied": []}
    manifest = run / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    request = SimpleNamespace(
        options=options,
        source_context={"repo": tmp_path},
        context=SimpleNamespace(
            repo=tmp_path,
            descriptor=tmp_path / "descriptor.yaml",
        ),
    )
    snapshot = {"content_sha256": "b" * 64}
    environment = {
        "implementation_sources": {"owned": "source"},
        "bundle_input_snapshot": snapshot,
        "bundle_manifest_sha256": P.run_inputs.bundle_manifest_identity(manifest, bundle),
    }
    value = SimpleNamespace(
        request=request,
        workspace=ws,
        run_dir=run,
        bundle=bundle,
        environment=environment,
        resuming=False,
        reviewed_roots=(tmp_path / "frozen-corpus",),
    )
    calls = []
    value.verify_inputs = lambda: calls.append(("inputs",))
    monkeypatch.setattr(P.bwrap, "verify_snapshot_binding", lambda *args, **kw: calls.append(("snapshot", args, kw)))
    monkeypatch.setattr(P, "verify_snapshot_for_phase1", lambda *args, **kw: calls.append(("seal", args, kw)))
    return value, calls


def test_complete_reopens_exact_ordinary_sources_manifest_snapshot_and_seal(prepared):
    selected, calls = prepared
    assert P.complete(selected) == 0
    assert [row[0] for row in calls] == ["inputs", "snapshot", "seal"]
    assert calls[1][1] == (selected.workspace, selected.bundle, selected.environment["bundle_input_snapshot"])
    assert calls[2][1] == (
        Path(selected.request.options.corpus_seal),
        selected.request.context.descriptor,
        selected.workspace,
        selected.bundle,
    )
    result = json.loads((selected.run_dir / "preflight_result.json").read_bytes())
    assert result["status"] == "startup_checks_completed" and result["provider_started"] is False
    assert result["formal_complete"] is False and "no author/client isolation" in result["scope"]
    assert "all_pass" not in result and "compiler_qualified" not in result
    assert (selected.run_dir / "preflight_result.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change", ["manifest", "source", "snapshot", "seal", "unreviewed", "resume"])
def test_drift_or_unreviewed_inputs_cannot_emit_preflight_success(prepared, monkeypatch, change):
    selected, _ = prepared

    def refused(*args, **kwargs):
        raise ValueError("owned live producer refused changed inputs")

    if change == "manifest":
        (selected.run_dir / "input_bundle_manifest.yaml").write_text("bundle_id: substituted\n")
    elif change == "source":
        selected.verify_inputs = refused
    elif change == "snapshot":
        monkeypatch.setattr(P.bwrap, "verify_snapshot_binding", refused)
    elif change == "seal":
        monkeypatch.setattr(P, "verify_snapshot_for_phase1", refused)
    elif change == "unreviewed":
        selected.reviewed_roots = None
    elif change == "resume":
        selected.resuming = True
    with pytest.raises((ValueError, RuntimeError)):
        P.complete(selected)
    assert not (selected.run_dir / "preflight_result.json").exists()


@pytest.mark.parametrize("prepared_result", [0, 3, "prepared"])
def test_actual_controller_stops_after_normal_prepare_without_authoring(tmp_path, monkeypatch, prepared_result):
    import merlin_experiments.phase1.preflight as preflight
    from merlin_experiments.phase1 import authoring, controller, runtime_environment, task_staging

    options = _options(tmp_path)
    context = SimpleNamespace(readback_policy=None, descriptor=tmp_path / "descriptor", target="owned")
    calls = []
    monkeypatch.setattr(S, "validate_options", lambda selected: None)
    from merlin_experiments.phase1 import timing

    monkeypatch.setattr(timing, "requires_chipyard_timing", lambda path: False)
    monkeypatch.setattr(
        task_staging,
        "callbacks",
        lambda config: SimpleNamespace(
            resolved_tools=lambda: (),
            stage_task=lambda *a, **k: None,
        ),
    )
    runtime = SimpleNamespace(refusal=None, account={})
    monkeypatch.setattr(runtime_environment, "prepare_runtime_environment", lambda *a, **kw: runtime)
    monkeypatch.setattr(runtime_environment, "applied_environment", lambda selected: nullcontext())
    marker = object()

    def prepare(request, transport, stage_task, **kwargs):
        calls.append(("prepare", request, transport))
        assert (
            request.options is options and request.bundle_manifest == tmp_path / "selected/input_bundle_manifest.yaml"
        )
        return marker if prepared_result == "prepared" else prepared_result

    def complete(selected):
        assert selected is marker
        calls.append(("preflight",))
        return 0

    monkeypatch.setattr(S, "prepare", prepare)
    monkeypatch.setattr(preflight, "complete", complete)
    monkeypatch.setattr(authoring, "execute", lambda *a, **kw: pytest.fail("authoring/completion reached"))
    result = controller.run(
        context,
        options,
        bundle_manifest=tmp_path / "selected/input_bundle_manifest.yaml",
        bundle_id="selected",
        oracle_timing=tmp_path / "timing.json",
        base_environment={},
    )
    assert result == (0 if prepared_result == "prepared" else prepared_result)
    assert [row[0] for row in calls] == (["prepare", "preflight"] if prepared_result == "prepared" else ["prepare"])


def test_installed_cli_propagates_explicit_selection_without_native_defaults(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import __main__ as CLI
    from merlin_experiments.phase1 import context, controller

    options = _options(tmp_path)
    selected = object()
    observed = {}
    monkeypatch.setattr(context, "load_context", lambda descriptor, **kw: selected)

    def run(actual, actual_options, **kwargs):
        assert actual is selected and actual_options.preflight_only
        assert actual_options.corpus_seal == options.corpus_seal
        assert kwargs["bundle_manifest"] == tmp_path / "input_bundle_manifest.yaml"
        observed.update(kwargs)
        return 3

    monkeypatch.setattr(controller, "run", run)
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text("bundle_id: selected\n")
    assert (
        CLI.main(
            [
                "--descriptor",
                str(tmp_path / "descriptor.yaml"),
                "--repo",
                str(tmp_path),
                "--bundle-manifest",
                str(manifest),
                "--bundle",
                "selected",
                "--oracle-timing",
                str(tmp_path / "timing.json"),
                "--preflight-only",
                "--corpus-seal",
                options.corpus_seal,
                "--run-id",
                "preflight",
                "--account-config-dir",
                "",
            ]
        )
        == 3
    )
    assert observed["oracle_timing"] == tmp_path / "timing.json"
