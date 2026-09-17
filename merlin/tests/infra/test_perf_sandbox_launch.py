"""The two ways a derived answer mask stops a run from ever launching.

Both failures were measured driving the real gemmini performance stage, and both scale with the
capsule corpus, so they get worse silently as capsules are added:

1. **The policy outgrows one argv entry.** The outer bwrap policy is handed to ``bash -c`` as a single
   argument, and Linux caps one argument at ``MAX_ARG_STRLEN`` (131072). At 866 derived answer
   surfaces the inlined policy is ~133 KB and the round dies at spawn with
   ``OSError: [Errno 7] Argument list too long: 'bash'`` -- a message that names nothing.
   :func:`bwrap_launch_command` spills the policy to a NUL-separated file read via ``bwrap --args FD``.

2. **The mask is enumerated from a tree that is not the one bound.** The frozen-input harness binds an
   immutable SNAPSHOT of ``merlin/contract`` over that tree's own live path. A capsule created after
   the freeze exists live but not in the snapshot, so masking it asks bwrap to mkdir a parent inside a
   read-only mount and every launch dies with ``Can't mkdir parents ...: Read-only file system``.
   ``bwrap.is_exposed`` now maps a path through the controlling bind's SOURCE and asks about the bytes
   actually served.

The direction that matters is the LEAK direction: a surface that IS present in the bound tree must
still be masked. "Skip what is inconvenient" would silently hand the agent the answer key, so every
test here asserts the mask fires as well as that the absent one is skipped, and each carries a vacuity
check -- this suite has a history of negative assertions passing for the wrong reason.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def _load_stage():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("_perf_agent_stage_launch", scripts
                                                  / "perf_agent_stage.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PS = _load_stage()
_HAS_BWRAP = shutil.which("bwrap") is not None
_SECRET = "answer: 1234-do-not-leak\n"


def _system_binds() -> list[str]:
    """The smallest set that lets ``bash``/``cat`` run inside the box."""
    argv = ["bwrap", "--die-with-parent", "--proc", "/proc", "--dev", "/dev"]
    for root in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(root).exists():
            argv += ["--ro-bind", root, root]
    return argv


@pytest.fixture()
def frozen_over_live(tmp_path: Path) -> dict:
    """A snapshot bound over its own live tree, with one capsule created AFTER the freeze."""
    live, snap = tmp_path / "live", tmp_path / "snap"
    for corpus in (live, snap):
        (corpus / "kept").mkdir(parents=True)
        (corpus / "kept" / "golden.yaml").write_text(_SECRET, encoding="utf-8")
    # created after the freeze: present live, absent from the snapshot the sandbox actually binds
    (live / "postfreeze").mkdir()
    (live / "postfreeze" / "golden.yaml").write_text(_SECRET, encoding="utf-8")
    kept = live / "kept" / "golden.yaml"
    postfreeze = live / "postfreeze" / "golden.yaml"
    surfaces = [AnswerSurface(f"golden:{p.name}", p, "file", "golden") for p in (kept, postfreeze)]
    argv = _system_binds() + ["--ro-bind", str(snap), str(live)]
    return {"live": live, "snap": snap, "kept": kept, "postfreeze": postfreeze,
            "surfaces": surfaces, "argv": argv}


# --------------------------------------------------------------------------- the mask enumeration
def test_mask_enumeration_follows_the_bound_tree_not_the_live_one(frozen_over_live):
    """LEAK DIRECTION FIRST: a surface present in the bound tree is masked; only the one that is
    genuinely absent from it is skipped."""
    argv, surfaces = frozen_over_live["argv"], frozen_over_live["surfaces"]
    kept, postfreeze = frozen_over_live["kept"], frozen_over_live["postfreeze"]

    # vacuity: both surfaces exist on the LIVE host, so a live-tree enumeration would list both and the
    # assertions below are not passing merely because there was nothing to enumerate.
    assert kept.is_file() and postfreeze.is_file()
    assert len(surfaces) == 2

    assert BW.is_exposed(argv, kept), "the snapshot serves this golden -- it IS reachable"
    assert not BW.is_exposed(argv, postfreeze), "the snapshot has no such path -- nothing is served"

    masked = BW.apply_answer_masks(argv, surfaces)
    added = masked[len(argv):]
    assert added == ["--ro-bind", "/dev/null", str(kept)], (
        "the present golden must be masked and the absent one must not be bound")
    assert BW.coverage_gap(masked, surfaces) == [], "no surface may remain reachable"


def test_a_masked_surface_is_never_skipped_when_the_snapshot_carries_it(frozen_over_live):
    """The fail-closed rule, stated as its own case: adding the file to the SNAPSHOT flips the skip
    back into a mask. Nothing about the live tree changes."""
    snap, postfreeze = frozen_over_live["snap"], frozen_over_live["postfreeze"]
    argv, surfaces = frozen_over_live["argv"], frozen_over_live["surfaces"]
    assert BW.apply_answer_masks(argv, surfaces)[len(argv):].count(str(postfreeze)) == 0

    (snap / "postfreeze").mkdir()
    (snap / "postfreeze" / "golden.yaml").write_text(_SECRET, encoding="utf-8")

    added = BW.apply_answer_masks(argv, surfaces)[len(argv):]
    assert added.count(str(postfreeze)) == 1, "now the bound tree carries it, so it MUST be masked"
    assert BW.coverage_gap(argv, surfaces) == surfaces, "unmasked, both are reachable"


@pytest.mark.skipif(not _HAS_BWRAP, reason="bwrap is not installed on this host")
def test_masked_launch_succeeds_and_the_unmapped_mask_is_what_broke_it(frozen_over_live):
    """The real launch. Positive: bwrap starts and the masked golden serves no content (bwrap mounts
    the /dev/null overlay nodev, so the read is refused rather than answered).
    Vacuity/regression: the pre-fix mask set -- the one enumerated from the live tree -- makes the very
    same launch fail with the measured 'Can't mkdir parents' error."""
    argv, surfaces = frozen_over_live["argv"], frozen_over_live["surfaces"]
    kept, postfreeze = frozen_over_live["kept"], frozen_over_live["postfreeze"]
    read = f"cat {kept} 2>&1; echo MARK; test -e {postfreeze} && echo PRESENT || echo ABSENT"

    # vacuity: WITHOUT the mask the secret is readable, so a passing assertion below means the mask worked
    bare = subprocess.run([*argv, "bash", "-c", read], capture_output=True, text=True, timeout=120)
    assert bare.returncode == 0, bare.stderr
    assert "do-not-leak" in bare.stdout, "unmasked, the golden must be readable or this proves nothing"

    masked = BW.apply_answer_masks(argv, surfaces)
    proc = subprocess.run([*masked, "bash", "-c", read], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"the masked policy must launch; stderr={proc.stderr!r}"
    served = proc.stdout.split("MARK")[0]
    assert "do-not-leak" not in served and "answer:" not in served, (
        f"the masked golden served content: {served!r}")
    assert "ABSENT" in proc.stdout, "a post-freeze capsule does not exist inside the sandbox at all"

    # the defect this pins: masking the post-freeze path anyway kills the launch outright
    overreach = [*masked, "--ro-bind", "/dev/null", str(postfreeze)]
    broken = subprocess.run([*overreach, "bash", "-c", read], capture_output=True, text=True,
                            timeout=120)
    assert broken.returncode != 0
    assert "Can't mkdir parents" in broken.stderr, broken.stderr


# --------------------------------------------------------------------------- the argv-length spill
def _oversized_policy() -> list[str]:
    """A real, launchable bwrap argv whose inlined join exceeds one argv entry."""
    argv = _system_binds()
    index = 0
    while len(" ".join(argv)) <= PS._ARG_STRLEN_MAX:
        argv += ["--setenv", f"PERF_PAD_{index:06d}", "p" * 240]
        index += 1
    return argv


def test_small_policy_is_inlined_byte_for_byte(tmp_path: Path):
    """Below the cap nothing changes -- no file is created and the string is the plain join."""
    argv = ["bwrap", "--ro-bind", "/usr", "/usr"]
    spilled: list[Path] = []
    spill_dir = tmp_path / "spill"
    assert PS.bwrap_launch_command(argv, " bash -c 'true'", spill_dir, spilled) == \
        "bwrap --ro-bind /usr /usr bash -c 'true'"
    assert spilled == []
    assert not spill_dir.exists(), "the small path must not touch the filesystem"


@pytest.mark.skipif(not _HAS_BWRAP, reason="bwrap is not installed on this host")
def test_oversized_policy_spills_to_an_args_fd_and_actually_launches(tmp_path: Path):
    """Positive case that fires: the spill path is taken AND the resulting command runs bwrap.
    Vacuity: the inlined form of the same policy really does raise E2BIG on this host."""
    argv = _oversized_policy()
    payload = " bash -c 'echo SPILLED_LAUNCH_OK'"

    inline = " ".join(argv) + payload
    assert len(inline) > PS._ARG_STRLEN_MAX, "the fixture must exceed the cap or it proves nothing"
    with pytest.raises(OSError) as excinfo:              # this is the measured production failure
        subprocess.run(["bash", "-c", inline], capture_output=True, timeout=120)
    assert excinfo.value.errno == 7                      # E2BIG

    spilled: list[Path] = []
    command = PS.bwrap_launch_command(argv, payload, tmp_path / "spill", spilled)
    assert len(spilled) == 1, "the oversized policy must have taken the spill path"
    assert len(command) < 4096, "the whole point is a short command line"
    assert command.startswith(f"bwrap --args {PS._ARGS_SPILL_FD} ")

    proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert "SPILLED_LAUNCH_OK" in proc.stdout

    spill = spilled[0]
    assert oct(spill.stat().st_mode & 0o777) == "0o600"
    assert spill.read_bytes().split(b"\0") == [v.encode() for v in argv[1:]]
    spill.unlink()


def test_spill_file_refuses_a_policy_it_cannot_represent(tmp_path: Path):
    with pytest.raises(PS.StageGateError, match="sandbox executable"):
        PS._spill_bwrap_argv(["notbwrap", "--ro-bind", "/usr", "/usr"], tmp_path / "s")
    with pytest.raises(PS.StageGateError, match="NUL"):
        PS._spill_bwrap_argv(["bwrap", "--setenv", "K", "a\0b"], tmp_path / "s")


def test_spill_files_are_removed_when_the_round_ends(tmp_path: Path):
    """The cleanup contract _codex_round relies on: every recorded spill is unlinkable and gone."""
    argv = _oversized_policy()
    spilled: list[Path] = []
    PS.bwrap_launch_command(argv, " bash -c 'true'", tmp_path / "spill", spilled)
    PS.bwrap_launch_command(argv, " bash -c 'true'", tmp_path / "spill", spilled)
    assert len(spilled) == 2 and len(set(spilled)) == 2, "each launch gets its own file"
    assert all(path.is_file() for path in spilled)
    for path in spilled:                                  # exactly what _codex_round's finally does
        path.unlink(missing_ok=True)
    assert not any(path.exists() for path in spilled)
    assert list((tmp_path / "spill").iterdir()) == []


# --------------------------------------------------------------------------- the refusal's evidence
def test_a_refusing_tool_probe_reports_the_stderr_it_refused_on(monkeypatch, tmp_path: Path):
    """A gate that swallows the evidence turns a one-line cause into an all-day hunt: the measured
    bwrap error must reach the raised message."""
    class _Probe:
        label, cmd, bind = "python3", "python3 -c 'pass'", "/usr/bin/python3"

    monkeypatch.setattr(PS.TC, "required_tool_probes", lambda te: [_Probe()])
    monkeypatch.setattr(PS.TC, "sandbox_env", lambda te, ws: "")
    monkeypatch.setattr(PS.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0] if a else [], 1, "", "bwrap: Can't mkdir parents for /x/golden.yaml: Read-only file system"))
    policy = PS.AgentSandboxPolicy(("bwrap",), (), "available_not_an_isolation_claim", True, True, True)

    with pytest.raises(PS.StageGateError) as excinfo:
        PS.run_required_tool_probes(policy, object(), tmp_path)
    message = str(excinfo.value)
    assert "rc=1" in message
    assert "Can't mkdir parents" in message, "the refusal must carry the evidence, not just an rc"
    assert "python3 -c" in message
