"""A gate must never report success for something it did not examine.

MEASURED, and the reason this file exists: five gates in the pre-commit hook took their entire work
list from `git` and never checked whether `git` ran. A failed `git` yields an empty work list, an
empty work list yields no findings, and no findings printed OK::

    GIT_DIR=/nonexistent/x.git build_tools/scripts/check_no_regex.py --staged
      ->  [  ok] no-regex: ... no stray regex in scope.     rc=0

That is a green that could not have gone red. `check_no_answer_keys.py` fixed exactly this shape
first and says why -- "we could not look" is not "there is nothing to find" -- and its siblings in
the same hook never got the fix. These tests hold the fix in place for all of them at once, in the
only way that proves anything: by reproducing the failure and demanding a refusal.

Every assertion here fails if its fix is reverted. Verified by mutation, not by inspection.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root

#: Gates whose work list (or whose repo root) comes from `git`. Each is invoked the way the
#: pre-commit hook invokes it. `check_fact_provenance` is here too even though it is not in the hook:
#: the defect is in the script, not in who calls it.
_GIT_FED_GATES = [
    "check_no_regex.py",
    "check_no_target_name.py",
    "check_no_assumed_constants.py",
    "check_artifact_layout.py",
    "check_provenance.py",
    "check_fact_provenance.py",
]

#: Gates that also speak the Claude Code Stop-hook dialect, where a BLOCK is
#: ``{"decision": "block"}`` on stdout and a non-zero exit is a NON-blocking error.
_STOP_HOOK_GATES = [
    "check_no_regex.py",
    "check_no_target_name.py",
    "check_no_assumed_constants.py",
    "check_artifact_layout.py",
    "check_provenance.py",
]


def _script(name: str):
    return repo_root() / "build_tools" / "scripts" / name


def _run(name: str, args: list[str], *, broken_git: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if broken_git:
        # The exact reproduction from the audit: a GIT_DIR that does not exist. `git` exits non-zero
        # and prints nothing on stdout, which is indistinguishable from "no files matched" unless the
        # gate checks the status.
        env["GIT_DIR"] = "/nonexistent/x.git"
        env.pop("GIT_WORK_TREE", None)
    return subprocess.run([sys.executable, str(_script(name)), *args],
                          cwd=repo_root(), capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.parametrize("gate", _GIT_FED_GATES)
def test_an_unreadable_work_list_is_a_refusal_not_a_pass(gate: str):
    """With `git` unusable the gate examined nothing, so it must not exit 0."""
    got = _run(gate, ["--staged"], broken_git=True)
    combined = got.stdout + got.stderr
    assert got.returncode != 0, (
        f"{gate} exited 0 with an unreadable index -- it examined NOTHING and reported success:\n"
        f"{combined}")
    assert "not the same as clean" in combined, (
        f"{gate} failed, but without saying that nothing was examined:\n{combined}")


@pytest.mark.parametrize("gate", _STOP_HOOK_GATES)
def test_the_refusal_reaches_a_stop_hook_in_its_own_dialect(gate: str):
    """A Stop hook BLOCKS via JSON on stdout; exiting non-zero there is merely a non-blocking error."""
    got = _run(gate, ["--staged", "--stop-hook"], broken_git=True)
    assert got.returncode == 0, f"{gate} --stop-hook must signal via JSON, not the exit code"
    payload = json.loads(got.stdout.strip().splitlines()[-1])
    assert payload.get("decision") == "block", f"{gate} did not BLOCK the session:\n{got.stdout}"
    assert "not the same as clean" in payload.get("reason", "")


def _n_checked(out: str) -> int:
    """The report count out of ``provenance: OK (N verdict-claiming report(s) checked...)``."""
    for line in out.splitlines():
        head, sep, rest = line.partition("provenance: OK (")
        if sep:
            return int(rest.split(" ", 1)[0])
    raise AssertionError(f"no provenance OK line in:\n{out}")


def test_staged_provenance_scans_the_untracked_reports_it_exists_for():
    """`--staged` must cover `out/`, which is the ONLY place the reports it gates actually live.

    Measured before the fix: `--staged` reported 0 verdict-claiming reports checked and the bare
    invocation reported 86. The pre-commit hook is the only caller that passes `--staged`, so that
    check had never run there once. The scan is affordable -- 1.4 s against 0.09 s -- so the fix is
    to cover the population, not to narrow the claim.
    """
    bare = _run("check_provenance.py", [])
    staged = _run("check_provenance.py", ["--staged"])
    assert bare.returncode == 0 and staged.returncode == 0, (bare.stderr, staged.stderr)
    n_bare, n_staged = _n_checked(bare.stdout), _n_checked(staged.stdout)
    if n_bare == 0:
        pytest.skip("no verdict-claiming reports on this host; nothing to establish")
    assert n_staged >= n_bare, (
        f"--staged checked {n_staged} verdict-claiming report(s) but the bare run found {n_bare}; "
        f"the untracked `out/` scan is exactly what --staged must not skip")


def test_the_mesh_gate_exit_code_reflects_its_finding_without_a_flag():
    """A gate whose finding only reached the exit code behind an opt-in flag cannot enforce.

    It printed 25 un-ratcheted weakenings and exited 0, and was wired into no hook and no CI job.
    """
    got = _run("check_mesh_assertion_not_weakened.py", ["--no-ratchet"])
    assert got.returncode != 0, (
        "the mesh gate reported findings and exited 0 with no ratchet applied:\n"
        + got.stdout[-2000:])
    assert _run("check_mesh_assertion_not_weakened.py", ["--no-ratchet", "--advisory"]).returncode == 0


def test_the_mesh_gate_resolves_the_default_target_not_a_corpus_category():
    """`_target_of` returned the CATEGORY for a capsule that carries no target directory.

    `merlin/contract/capsules/isa/<capsule>/` is the DEFAULT target's, and `isa` is not a target: no
    contract resolves under that name, so all 133 such capsules landed in `unresolved_targets` and
    established nothing while the gate exited 0.
    """
    got = _run("check_mesh_assertion_not_weakened.py", ["--json", "--no-ratchet"])
    rep = json.loads(got.stdout)["report"]
    assert rep["unresolved_targets"] == {}, (
        f"unresolved: {sorted(rep['unresolved_targets'])} -- a corpus category is not a target")
    assert rep["n_admitted_capsules_checked"] > 400, (
        f"only {rep['n_admitted_capsules_checked']} capsules examined; the blind gate saw 199")


@pytest.mark.parametrize("flag", ["--staged", "--stop-hook"])
def test_contract_copies_reads_the_flags_it_advertises(flag: str):
    """Both were parsed, advertised in the hardware-pins skill, and read by nothing."""
    plain = _run("check_contract_copies.py", [])
    got = _run("check_contract_copies.py", [flag])
    assert got.returncode == 0 and plain.returncode == 0, (plain.stderr, got.stderr)
    assert got.stdout != plain.stdout, (
        f"{flag} produced byte-identical output to no flag at all, which is what a dead argparse "
        f"destination looks like")
    if flag == "--stop-hook":
        json.loads(got.stdout.strip())      # stdout must be JSON ONLY in hook mode
    else:
        assert got.stdout.strip() == ""     # pre-commit mode: same checks, no success chatter


# --- artifact-layout: both git calls, individually --------------------------------------------------
# The end-to-end test above is killed by reverting EITHER of this gate's two git calls, because
# whichever one still raises stops the run -- so it cannot tell which fix is in place. These two pin
# each call on its own, which is what a mutation table needs to be honest.

def _layout_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_layout_gate", _script("check_artifact_layout.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_layout_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_layout_repo_root_refuses_rather_than_falling_back_to_the_cwd(monkeypatch):
    """A wrong root checks the wrong files. `git rev-parse` failing used to mean `Path.cwd()`."""
    gate = _layout_gate()
    monkeypatch.setenv("GIT_DIR", "/nonexistent/x.git")
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        gate._repo_root()


def test_layout_tracked_refuses_rather_than_returning_an_empty_work_list(monkeypatch):
    gate = _layout_gate()
    monkeypatch.setenv("GIT_DIR", "/nonexistent/x.git")
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    for staged in (True, False):
        with pytest.raises((OSError, subprocess.CalledProcessError)):
            gate._tracked(repo_root(), staged)
