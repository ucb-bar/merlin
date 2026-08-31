"""The launch preflight must be able to see what it checks, and its verdict must stop the launch.

Two independent failures occurred on a live launch:

  * ``_run_preflight`` chmod-000-locked every answer surface. The host anti-cheat and hidden grader run as
    the same user, so this blinded both of them. The real isolation boundary is bwrap; host trees must stay
    owner-readable while the sandbox mount table masks them from the agent.
  * ``chia_ab_batch`` called ``LB._run_preflight()`` and discarded the return code, so the run printed
    "VERIFY_NO_CHEAT: FAIL -- DO NOT launch" and launched.

A gate that cannot see, whose verdict is thrown away, is decoration.
"""
from __future__ import annotations

import ast
import importlib
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root

HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"


def _src(name: str) -> str:
    return (HARNESS / name).read_text(encoding="utf-8")


def _fn(name: str, mod: str) -> ast.FunctionDef:
    tree = ast.parse(_src(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {mod}")


def _launcher():
    sys.path.insert(0, str(HARNESS))
    return importlib.import_module("launch_ab_batch")


def _args(experiment: str):
    return SimpleNamespace(
        model="test-model", effort="high", max_rounds=1, max_rate_limit_waits=1,
        round_timeout=60, min_rounds=0, schedule="rounds", max_wall_s=0,
        plateau_rounds=None, model_budget_s=None, driver="codex", subagent_model="",
        background_model="", experiment=experiment, skip_hidden=False, sandbox="bwrap",
        provider="subscription", aws_region="us-east-1", aws_profile="", with_tool=[],
        without_tool=[])


def test_full_rtlchecks_command_pins_the_wrapper_bundle():
    """The full Arm4 wrapper swaps to rtlchecks_public; preflight must see that exact bundle."""
    launcher = _launcher()
    command = launcher._arm_cmd("merlin_rtlchecks", "test-run", _args("full"))

    assert command[command.index("--experiment") + 1] == "full"
    assert command[command.index("--bundle") + 1] == "merlin_assisted_rtlchecks_public_v0"


def test_realistic_rtlchecks_commands_pin_each_condition_bundle():
    launcher = _launcher()
    expected = {
        "kernels": "merlin_assisted_rtlchecks_hwbringup_v0",
        "no-kernels": "merlin_assisted_rtlchecks_hwbringup_nokernel_v0",
    }
    for condition, bundle_id in expected.items():
        command = launcher._arm_cmd(
            "merlin_rtlchecks", f"test-{condition}", _args("realistic"), condition)
        assert command[command.index("--bundle") + 1] == bundle_id


def test_preflight_checks_only_the_manifests_named_by_planned_commands(tmp_path, monkeypatch):
    launcher = _launcher()
    bundles = tmp_path / "bundles"
    selected = bundles / "selected" / "input_bundle_manifest.yaml"
    selected.parent.mkdir(parents=True)
    selected.write_text("bundle_id: selected\n", encoding="utf-8")
    # An unrelated malformed bundle must not enter a one-cell preflight.
    unrelated = bundles / "unrelated" / "input_bundle_manifest.yaml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("bundle_id: wrong\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "target_experiment.yaml").write_text("target: test\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify_no_cheat.py").write_text("", encoding="utf-8")

    seen = []
    import merlin.targetgen.target_experiment as target_experiment
    monkeypatch.setattr(launcher.C, "BUNDLES", bundles)
    monkeypatch.setattr(launcher.C, "EXP", experiment)
    monkeypatch.setattr(launcher.C, "REPO", tmp_path)
    monkeypatch.setattr(launcher.C, "require_scaffolding", lambda: None)
    monkeypatch.setattr(launcher, "SCRIPTS", scripts)
    monkeypatch.setattr(launcher, "_host_answer_surfaces", lambda _te: [])
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _path: object())
    monkeypatch.setattr(
        target_experiment, "bundles_match_descriptor",
        lambda _te, manifests: seen.extend(manifests) or [])
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    assert launcher._run_preflight([["driver", "--bundle", "selected"]]) == 0
    assert seen == [selected]


def test_selected_bundle_must_exist_and_declare_the_requested_identity(tmp_path, monkeypatch):
    launcher = _launcher()
    monkeypatch.setattr(launcher.C, "BUNDLES", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        launcher._planned_bundle_manifests([["driver", "--bundle", "missing"]])

    manifest = tmp_path / "selected" / "input_bundle_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("bundle_id: another-cell\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        launcher._planned_bundle_manifests([["driver", "--bundle", "selected"]])


def test_host_access_is_restored_before_verification():
    """A stale mode-000 run must be repaired before the host tries to enumerate hidden capsules."""
    fn = _fn("_run_preflight", "launch_ab_batch.py")
    prepare_lines, vnc_lines = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_make_host_owner_only":
                prepare_lines.append(node.lineno)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "verify_no_cheat.py":
                vnc_lines.append(node.lineno)
    assert prepare_lines, "the host-readable answer-surface preparation disappeared"
    assert vnc_lines, "verify_no_cheat is no longer invoked by the preflight"
    assert min(prepare_lines) < min(vnc_lines), (
        "host access must be restored before verify_no_cheat walks hidden/*/capsule.yaml")


def test_host_protection_is_owner_only_never_mode_zero():
    """The host grader retains access; the agent is isolated by the separately tested bwrap masks."""
    fn = _fn("_make_host_owner_only", "launch_ab_batch.py")
    modes = {node.value for node in ast.walk(fn)
             if isinstance(node, ast.Constant) and isinstance(node.value, int)}
    assert 0o700 in modes and 0o600 in modes
    assert 0 not in modes, "mode 000 blinds same-UID host grading and is not an agent security boundary"


def test_the_preflight_verdict_gates_the_launch():
    """The return value must be bound and returned, not called for its side effects."""
    tree = ast.parse(_src("chia_ab_batch.py"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_run_preflight"]
    assert calls, "chia_ab_batch no longer runs the preflight at all"
    # every call must be part of an assignment (its value used), never a bare expression statement
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "_run_preflight"]
    assert not bare, ("_run_preflight's return code is discarded — a failed preflight would print "
                      "'DO NOT launch' and then launch")


def test_bundle_lock_hashes_file_grants_and_caches_shared_paths(tmp_path, monkeypatch):
    """File bytes are pinned, and one shared large grant is read once per pass."""
    sys.path.insert(0, str(HARNESS))
    preflight = importlib.import_module("preflight")
    repo = tmp_path / "repo"
    bundles = repo / "bundles"
    grant = repo / "isa.h"
    repo.mkdir()
    grant.write_bytes(b"first")
    for bundle_id in ("arm_a", "arm_b"):
        bundle_dir = bundles / bundle_id
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "input_bundle_manifest.yaml").write_text(
            f"bundle_id: {bundle_id}\nallowed:\n  - path: isa.h\n", encoding="utf-8")
    monkeypatch.setattr(preflight.C, "REPO", repo)
    monkeypatch.setattr(preflight.C, "BUNDLES", bundles)
    real_hash = preflight._hash_granted_path
    calls = []

    def counted(path):
        calls.append(path)
        return real_hash(path)

    monkeypatch.setattr(preflight, "_hash_granted_path", counted)
    result = preflight.check_bundle_hash_repro()
    before = preflight.yaml.safe_load(
        (bundles / "arm_a" / "bundle_lock.yaml").read_text(encoding="utf-8"))

    assert all(cell["reproducible"] for cell in result.values())
    assert len(calls) == 2, "one shared path must be hashed once in each independent pass"
    assert before["allowed_tree_sha256"]["isa.h"] == real_hash(grant)["sha256"]

    grant.write_bytes(b"second")
    calls.clear()
    preflight.check_bundle_hash_repro()
    after = preflight.yaml.safe_load(
        (bundles / "arm_a" / "bundle_lock.yaml").read_text(encoding="utf-8"))

    assert len(calls) == 2
    assert before["allowed_tree_sha256"]["isa.h"] != after["allowed_tree_sha256"]["isa.h"]
