"""Original one-ELF policy routing controls; no namespace or engine executes."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import census_launch as L

from merlin.common import invocation_record as I
from merlin.perf import dynamic_census as C


@pytest.fixture
def original(tmp_path, monkeypatch):
    workspace, package, tools = (tmp_path / name for name in ("workspace", "compiler", "public-tools"))
    for path in (workspace, package, tools):
        path.mkdir()
    elf, source = workspace / "package_kernel.elf", workspace / "kernel.o"
    elf.write_bytes(b"synthetic original linked product; never loaded")
    source.write_bytes(b"synthetic original object")
    tool = tools / "linker"
    tool.write_bytes(b"synthetic original tool; never loaded")
    tool.chmod(0o555)
    prefix = (
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-net",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        str(tools),
        str(tools),
        "--tmpfs",
        "/scratch",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        str(workspace),
        "--bind",
        str(workspace),
        str(workspace),
        "--ro-bind",
        str(package),
        str(package),
    )
    policy = PC.PackageSandboxPolicy(prefix, (), (), workspace, package, None)
    with I.observe(workspace, stage="elf", argv=[str(tool)], inputs=(source,), outputs=(elf,), env={}) as record:
        record.complete(subprocess.CompletedProcess([str(tool)], 0, b"", b""))
    monkeypatch.setattr(L.R.resources, "active_resource_boundary", lambda: None)
    yield SimpleNamespace(**locals())


def test_only_exact_readonly_elf_and_original_tools_reach_census(original, monkeypatch):
    assert Path(L.__file__).resolve().is_relative_to(Path.cwd() / "packages/merlin-experiments/src")
    calls = []

    def issue(**kwargs):
        launch = SimpleNamespace(**kwargs, closed=True, _issuer=object())
        calls.append(launch)
        return launch

    monkeypatch.setattr(C, "_issue_launch", issue)
    monkeypatch.setattr(C, "profile_elf", lambda *args, **kwargs: C.unknown("synthetic routing control"))
    result = L.observe_dynamic_census(
        original.elf,
        target="synthetic",
        directory=original.workspace / "census",
        remaining_s=20,
        policy=original.policy,
    )
    assert result["status"] == "UNKNOWN" and len(calls) == 1
    launch = calls[0]
    assert launch.executed_elf != original.elf
    assert str(original.package) not in launch.prefix
    assert str(original.workspace) not in launch.prefix
    assert str(original.tools) in launch.prefix
    assert "--tmpfs" not in launch.prefix
    assert launch.writable_roots[0] == launch.executed_elf.parent
    assert not launch.executed_elf.parent.parent.exists()  # Proven no process started.


@pytest.mark.parametrize("damage", ["absent_policy", "wrong_stage", "changed_elf", "foreign_output", "linked_elf"])
def test_missing_or_changed_original_owner_never_launches(original, monkeypatch, damage):
    policy, directory = original.policy, original.workspace / "census"
    if damage == "absent_policy":
        policy = None
    elif damage == "wrong_stage":
        row = json.loads(original.record.path.read_text())
        row["stage"] = "original_build_process"
        original.record.path.write_text(json.dumps(row))
    elif damage == "changed_elf":
        original.elf.write_bytes(b"not the completed linked bytes")
    elif damage == "foreign_output":
        directory = original.tmp_path / "foreign"
    else:
        original.elf.unlink()
        original.elf.symlink_to(original.source)
    monkeypatch.setattr(C, "_issue_launch", lambda **kwargs: pytest.fail("foreign census gained a live launch"))
    result = L.observe_dynamic_census(
        original.elf, target="synthetic", directory=directory, remaining_s=20, policy=policy
    )
    assert result["status"] == "UNKNOWN" and result["observations"] is None
    assert str(original.tmp_path) not in str(result)


def test_cleanup_unknown_aborts_optional_feedback_and_retains_input_backing(original, monkeypatch):
    calls = []

    def issue(**kwargs):
        launch = SimpleNamespace(**kwargs, closed=False, _issuer=object())
        calls.append(launch)
        return launch

    def unresolved(*args, **kwargs):
        raise C.CensusCleanupUnknown("synthetic unresolved ownership control")

    monkeypatch.setattr(C, "_issue_launch", issue)
    monkeypatch.setattr(C, "profile_elf", unresolved)
    with pytest.raises(C.CensusCleanupUnknown):
        L.observe_dynamic_census(
            original.elf,
            target="synthetic",
            directory=original.workspace / "census",
            remaining_s=20,
            policy=original.policy,
        )
    assert calls[0].executed_elf.exists()
    assert calls[0].writable_roots[1].is_dir()


def test_completed_actual_input_is_retained_for_invocation_reopening(original, monkeypatch):
    calls = []

    def issue(**kwargs):
        launch = SimpleNamespace(**kwargs, closed=True, started=True, _issuer=object())
        calls.append(launch)
        return launch

    monkeypatch.setattr(C, "_issue_launch", issue)
    monkeypatch.setattr(C, "profile_elf", lambda *args, **kwargs: C.unknown("synthetic joined completion"))
    L.observe_dynamic_census(
        original.elf,
        target="synthetic",
        directory=original.workspace / "census",
        remaining_s=20,
        policy=original.policy,
    )
    assert calls[0].executed_elf.read_bytes() == original.elf.read_bytes()
    assert not calls[0].writable_roots[1].exists()
