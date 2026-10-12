"""The pre-spend gate binds the selected bundle and never launches an agent."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from merlin_experiments.phase1 import tooling_readiness as R


def _inputs(tmp_path):
    context = SimpleNamespace(target="synthetic", repo=tmp_path, descriptor=tmp_path / "descriptor.yaml")
    te = SimpleNamespace(target="synthetic", rtl_facts_pin="target/contracts/rtl_facts/", sim_via="")
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    return context, te, ws


def test_baseline_receipt_is_private_and_explicitly_skipped(tmp_path):
    context, te, ws = _inputs(tmp_path)
    run_dir = tmp_path / "receipts"
    run_dir.mkdir()
    receipt = R.run(
        context,
        te,
        ws,
        {"bundle_id": "raw_baseline_fixture", "arm": "raw_baseline"},
        (),
        tmp_path,
        run_dir,
        {"content_sha256": "frozen"},
        "manifest",
    )
    saved = yaml.safe_load((run_dir / "tooling_readiness.yaml").read_text())
    assert receipt == saved
    assert saved["status"] == "skipped"
    assert saved["snapshot_content_sha256"] == "frozen"
    assert (run_dir / "tooling_readiness.yaml").stat().st_mode & 0o077 == 0


def test_assisted_without_frozen_snapshot_refuses_with_private_receipt(tmp_path):
    context, te, ws = _inputs(tmp_path)
    run_dir = tmp_path / "receipts"
    run_dir.mkdir()
    receipt = R.run(
        context,
        te,
        ws,
        {"bundle_id": "merlin_assisted_rtlchecks_fixture", "arm": "merlin_rtlchecks"},
        R.TR.ARM_TOOLS["merlin_rtlchecks"],
        tmp_path,
        run_dir,
        None,
        "manifest",
    )
    assert receipt["status"] == "no_go"
    assert receipt["checks"] == [{"name": "frozen_snapshot", "ok": False, "detail": "no verified snapshot"}]
    assert (run_dir / "tooling_readiness.yaml").stat().st_mode & 0o077 == 0


def test_assisted_missing_selected_grant_fails_before_sandbox(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    monkeypatch.setattr(R, "_live_probe", lambda *args: (_ for _ in ()).throw(AssertionError("sandbox launched")))
    result = R.assess(
        context,
        te,
        ws,
        {"bundle_id": "merlin_assisted_rtlchecks_fixture", "arm": "merlin_rtlchecks", "allowed": []},
        R.TR.ARM_TOOLS["merlin_rtlchecks"],
        tmp_path,
    )
    assert result["status"] == "no_go"
    assert any(row["name"].startswith("grant:xdsl_kit:") and not row["ok"] for row in result["checks"])


def test_selected_source_grants_survive_a_different_installed_python(tmp_path, monkeypatch):
    _, te, _ = _inputs(tmp_path)
    source = tmp_path / "source"
    allowed = []
    for logical in R.TR.spec("merlin_infra").bundle_paths:
        selected = source / "merlin" / logical.removeprefix("merlin/python/merlin/")
        allowed.append({"path": str(selected) + ("/" if logical.endswith("/") else "")})
    monkeypatch.setattr(R, "python_source_dir", lambda: tmp_path / "unrelated-install")
    checks = R._grant_checks(te, {"allowed": allowed}, ("merlin_infra",))
    assert checks and all(check["ok"] for check in checks)


def test_selected_source_grants_still_fail_when_one_is_missing(tmp_path, monkeypatch):
    _, te, _ = _inputs(tmp_path)
    source = tmp_path / "source"
    logical = R.TR.spec("merlin_infra").bundle_paths
    allowed = [
        {
            "path": str(source / "merlin" / item.removeprefix("merlin/python/merlin/"))
            + ("/" if item.endswith("/") else "")
        }
        for item in logical[:-1]
    ]
    monkeypatch.setattr(R, "python_source_dir", lambda: tmp_path / "unrelated-install")
    checks = R._grant_checks(te, {"allowed": allowed}, ("merlin_infra",))
    assert sum(not check["ok"] for check in checks) == 1


def test_new_verification_profile_accepts_exact_legacy_client_leaves(tmp_path, monkeypatch):
    _, te, _ = _inputs(tmp_path)
    root = tmp_path / "selected-src" / "merlin"
    monkeypatch.setattr(R, "python_source_dir", lambda: tmp_path / "unrelated-install")
    allowed = [
        {"path": str(root / "common") + "/"},
        {"path": str(root / "verify/cli.py")},
        {"path": str(root / "xdsl_dialects/opt.py")},
    ]
    bundle = {"allowed": allowed, "compiler_tool_profile": {"schema": "merlin.compiler_tool_profile.v1"}}
    checks = R._grant_checks(te, bundle, ("verify_seam",))
    assert len(checks) == 2 and all(check["ok"] for check in checks)
    assert "verify/cli.py" in checks[0]["detail"]
    allowed.remove({"path": str(root / "verify/cli.py")})
    assert not R._grant_checks(te, bundle, ("verify_seam",))[0]["ok"]


def test_historical_verification_profile_still_requires_its_directory_grant(tmp_path, monkeypatch):
    _, te, _ = _inputs(tmp_path)
    root = tmp_path / "selected-src" / "merlin"
    monkeypatch.setattr(R, "python_source_dir", lambda: tmp_path / "unrelated-install")
    allowed = [
        {"path": str(root / "common") + "/"},
        {"path": str(root / "verify/cli.py")},
        {"path": str(root / "xdsl_dialects/opt.py")},
    ]
    assert not R._grant_checks(te, {"allowed": allowed}, ("verify_seam",))[0]["ok"]
    allowed.append({"path": str(root / "verify") + "/"})
    assert all(check["ok"] for check in R._grant_checks(te, {"allowed": allowed}, ("verify_seam",)))


def test_assisted_selected_snapshot_probe_failure_is_no_go(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    monkeypatch.setattr(R, "_grant_checks", lambda *args: [R._check("grants", True, "frozen")])
    monkeypatch.setattr(R, "_live_probe", lambda *args: R._check("selected_sandbox_authoring", False, "broker failed"))
    result = R.assess(
        context,
        te,
        ws,
        {"bundle_id": "merlin_assisted_fixture", "arm": "merlin_assisted", "allowed": []},
        R.TR.ARM_TOOLS["merlin_assisted"],
        tmp_path,
    )
    assert result["status"] == "no_go"
    assert result["checks"][-1]["detail"] == "broker failed"


def test_el4_missing_rtl_tools_is_no_go_even_when_manifest_has_base_grants(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    monkeypatch.setattr(R, "_grant_checks", lambda *args: [R._check("grants", True, "frozen")])
    monkeypatch.setattr(R, "_live_probe", lambda *args: (_ for _ in ()).throw(AssertionError("sandbox launched")))
    result = R.assess(
        context,
        te,
        ws,
        {"bundle_id": "merlin_assisted_rtlchecks_fixture", "arm": "merlin_rtlchecks"},
        R.TR.ARM_TOOLS["merlin_assisted"],
        tmp_path,
    )
    assert result["status"] == "no_go"
    missing = next(row for row in result["checks"] if row["name"] == "selected_tool_set")
    assert "rtl_generators" in missing["detail"] and "rtl_facts" in missing["detail"]


def test_full_authoring_probe_is_valid_python():
    script = R._sandbox_probe(
        "synthetic",
        ("merlin_infra", "xdsl_kit", "cca_spine", "rtl_generators", "rtl_facts", "isa_tools", "cca_tools"),
        "FENCE",
    )
    compile(script, "<readiness probe>", "exec")
    assert "AUTHORING_AND_BROKER_ROUNDTRIPS_OK" in script


def test_probe_sibling_mounts_the_candidates_frozen_snapshot(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "tool.py"
    source.parent.mkdir()
    source.write_text("FROZEN = True\n")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": str(source), "mode": "ro"}], "denied": []}
    R.BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    probe_ws = ws.parent / "tooling-readiness-probe"
    probe_ws.mkdir()

    assert R.BW.bundle_snapshot_root(probe_ws) == R.BW.bundle_snapshot_root(ws)
    args = R.BW.base_argv(probe_ws, bundle, repo=repo, include_claude_home=False)
    frozen = R.BW.bundle_snapshot_root(ws) / "repo/tool.py"
    assert ["--ro-bind", str(frozen), str(source)] == args[args.index(str(frozen)) - 1 : args.index(str(frozen)) + 2]


def test_rtlchecks_compiles_from_frozen_facts_and_capsule(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    frozen = tmp_path / "run" / "bundle_inputs" / "repo" / "target" / "contracts" / "rtl_facts"
    frozen.mkdir(parents=True)
    (frozen / "facts.json").write_text('{"facts": {"interfaces": [{"name": "decode"}]}}')
    capsule = tmp_path / "public" / "sample"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: sample\n")
    from merlin.targetgen import rtl_check_compiler as compiler
    from merlin.targetgen import rtl_check_runner as runner

    seen = []
    monkeypatch.setattr(R.BW, "resolve_grant", lambda path, repo: tmp_path / "target/contracts/rtl_facts")
    monkeypatch.setattr(R.BW, "snapshot_input_paths", lambda ws, bundle, sources, repo: [frozen])
    monkeypatch.setattr(
        compiler,
        "compile_checks",
        lambda facts, cap, target: (
            seen.append((facts, cap, target)) or {"kernel": "CHECK: op", "endpoint_status": "resolved"}
        ),
    )
    monkeypatch.setattr(runner, "find_filecheck", lambda *args: "/bin/FileCheck")
    checks = R._frozen_rtl_checks(te, ws, {}, capsule.parent, repo=context.repo)
    assert all(row["ok"] for row in checks)
    assert seen == [({"facts": {"interfaces": [{"name": "decode"}]}}, {"name": "sample"}, "synthetic")]


def test_filecheck_lookup_includes_declared_sim_family(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    te.sim_via = "chipyard"
    frozen = tmp_path / "run" / "bundle_inputs" / "facts.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{"facts": {"interfaces": [{"name": "decode"}]}}')
    capsule = tmp_path / "public" / "sample"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: sample\n")
    from merlin.targetgen import rtl_check_compiler as compiler
    from merlin.targetgen import rtl_check_runner as runner
    from merlin.targetgen.sandbox import toolchain as toolchain

    sim_bin = tmp_path / "sim-tools" / "bin"
    sim_bin.mkdir(parents=True)
    (sim_bin / "FileCheck").write_text("tool")
    monkeypatch.setattr(toolchain, "_sim", lambda selected: toolchain.SimToolchain(path_dirs=(str(sim_bin),)))
    monkeypatch.setattr(R.BW, "resolve_grant", lambda path, repo: tmp_path / "facts.json")
    monkeypatch.setattr(R.BW, "snapshot_input_paths", lambda ws, bundle, sources, repo: [frozen])
    monkeypatch.setattr(compiler, "compile_checks", lambda *args: {"kernel": "CHECK: op"})
    monkeypatch.setattr(
        runner,
        "find_filecheck",
        lambda candidates: next((str(path) for path in candidates if Path(path).is_file()), None),
    )
    checks = R._frozen_rtl_checks(te, ws, {}, capsule.parent, repo=context.repo)
    assert next(check for check in checks if check["name"] == "FileCheck")["detail"] == str(sim_bin / "FileCheck")


def test_rtlchecks_finds_a_checkable_capsule_after_an_uncheckable_one(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    frozen = tmp_path / "run" / "bundle_inputs" / "facts.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{"facts": {"interfaces": [{"name": "decode"}]}}')
    public = tmp_path / "public"
    for name in ("a_host_only", "b_device"):
        member = public / name
        member.mkdir(parents=True)
        (member / "capsule.yaml").write_text(f"name: {name}\n")
    from merlin.targetgen import rtl_check_compiler as compiler
    from merlin.targetgen import rtl_check_runner as runner

    monkeypatch.setattr(R.BW, "resolve_grant", lambda path, repo: tmp_path / "facts.json")
    monkeypatch.setattr(R.BW, "snapshot_input_paths", lambda ws, bundle, sources, repo: [frozen])
    monkeypatch.setattr(
        compiler,
        "compile_checks",
        lambda facts, cap, target: {"kernel": "CHECK: op"} if cap["name"] == "b_device" else {},
    )
    monkeypatch.setattr(runner, "find_filecheck", lambda *args: "/bin/FileCheck")
    checks = R._frozen_rtl_checks(te, ws, {}, public, repo=context.repo)
    assert all(row["ok"] for row in checks)
    assert "capsule=b_device" in checks[-1]["detail"]


def test_selected_release_facts_replace_the_legacy_pin(tmp_path, monkeypatch):
    context, te, ws = _inputs(tmp_path)
    selected = tmp_path / "phase0-selected" / "facts.json"
    frozen = tmp_path / "run" / "bundle_inputs" / "external" / "phase0-selected" / "facts.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{"facts": {"target": "synthetic", "interfaces": [{"name": "decode"}]}}')
    capsule = tmp_path / "public" / "sample"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: sample\n")
    bundle = {
        "allowed": [{"path": str(selected.parent) + "/"}],
        "selected_rtl_facts_file": str(selected),
    }
    assert all(row["ok"] for row in R._grant_checks(te, bundle, ("rtl_facts",)))

    def reject_legacy(*args, **kwargs):
        raise AssertionError("legacy pin used")

    def frozen_selected(ws, bundle, sources, repo):
        assert sources == [selected]
        return [frozen]

    monkeypatch.setattr(R.BW, "resolve_grant", reject_legacy)
    monkeypatch.setattr(R.BW, "snapshot_input_paths", frozen_selected)
    from merlin.targetgen import rtl_check_compiler as compiler
    from merlin.targetgen import rtl_check_runner as runner

    monkeypatch.setattr(compiler, "compile_checks", lambda *args: {"kernel": "CHECK: op"})
    monkeypatch.setattr(runner, "find_filecheck", lambda *args: "/bin/FileCheck")
    assert all(row["ok"] for row in R._frozen_rtl_checks(te, ws, bundle, capsule.parent, repo=context.repo))
