"""A gate wired into the Stop hook must be able to BLOCK, not merely to complain.

A Claude Code Stop hook signals a refusal through ``{"decision": "block"}`` on stdout. A non-zero exit
status there is a NON-blocking error: the message is shown and the session stops anyway. Two of the
seven gates wired with ``--stop-hook`` in ``.claude/settings.json`` never implemented that contract --
``check_no_answer_keys.py`` did not parse the flag at all, and ``check_provenance.py`` parsed it only to
decorate its success line -- so both detected real violations, exited 1, and enforced nothing. Measured
before the fix: a tracked ``golden.yaml`` produced ``EXIT=1`` and plain text, while the five working
gates produced ``{"decision": "block", ...}``.

The roster is READ FROM THE SETTINGS FILE, never listed here: a gate added to the Stop hook tomorrow is
covered by these tests without editing them, which is the whole point of catching this class once.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root

REPO = repo_root()
SETTINGS = REPO / ".claude" / "settings.json"


def _stop_hook_scripts() -> list[str]:
    """Every ``build_tools/scripts/check_*.py`` the Stop hook invokes with ``--stop-hook``."""
    cfg = json.loads(SETTINGS.read_text(encoding="utf-8"))
    out = []
    for group in cfg.get("hooks", {}).get("Stop", []):
        for hook in group.get("hooks", []):
            cmd = hook.get("command", "")
            if "--stop-hook" not in cmd:
                continue
            for tok in cmd.split():
                if tok.startswith("build_tools/scripts/") and tok.endswith(".py"):
                    out.append(tok)
    return sorted(set(out))


def _run(script: str, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([str(REPO / ".venv" / "bin" / "python"), str(REPO / script), *args],
                          capture_output=True, text=True, cwd=str(REPO), env=env, timeout=900)


def test_the_settings_file_actually_wires_stop_hook_gates():
    """Guard the guard: if the roster is empty every test below passes vacuously."""
    scripts = _stop_hook_scripts()
    assert len(scripts) >= 5, f"expected the Stop hook to carry several gates, found {scripts}"


@pytest.mark.parametrize("script", _stop_hook_scripts())
def test_a_stop_hook_gate_speaks_the_hook_protocol(script):
    """stdout must be a JSON object -- that is the only channel a Stop hook can refuse through.

    Anything else (a ``[FAIL] ...`` line, a notes dump) means the gate can report but not enforce.
    Diagnostics belong on stderr, which the hook surfaces either way.
    """
    r = _run(script, "--stop-hook")
    assert r.returncode == 0, (
        f"{script} --stop-hook exited {r.returncode}; a Stop hook signals through JSON on stdout, and "
        f"a non-zero exit is a NON-blocking error there.\nstdout: {r.stdout[:400]}")
    try:
        payload = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"{script} --stop-hook wrote non-JSON to stdout ({e}); it cannot block.\n"
            f"stdout: {r.stdout[:400]}") from None
    assert isinstance(payload, dict), f"{script} --stop-hook must emit a JSON object, got {type(payload)}"
    if payload:
        assert payload.get("decision") == "block", (
            f"{script} --stop-hook emitted a non-empty payload without decision=block: {payload}")
        assert payload.get("reason"), f"{script} blocks without saying why: {payload}"


def _index_with(tmp_path, path_in_repo: str, content: str) -> dict:
    """A THROWAWAY git index containing ``path_in_repo``.

    Built with ``hash-object`` + ``update-index --cacheinfo`` so nothing is written to the worktree and
    the real ``.git/index`` -- shared with concurrent sessions in this checkout -- is never touched.
    """
    idx = tmp_path / "throwaway.index"
    shutil.copy(REPO / ".git" / "index", idx)
    env = {"GIT_INDEX_FILE": str(idx)}
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=content, text=True,
                          capture_output=True, cwd=str(REPO), check=True).stdout.strip()
    subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path_in_repo}"],
                   cwd=str(REPO), env={**__import__("os").environ, **env}, check=True,
                   capture_output=True)
    return env


def test_a_tracked_answer_key_blocks_the_stop_hook(tmp_path):
    """The invariant the gate exists for: a published answer key must stop the session.

    Before the fix this exited 1 with plain text and the session stopped regardless -- on a PUBLIC repo
    whose whole benchmark depends on the goldens staying untracked.
    """
    leak = "merlin/contract/capsules/conformance/_stop_hook_probe/golden.yaml"
    env = _index_with(tmp_path, leak, "answer: 42\n")
    r = _run("build_tools/scripts/check_no_answer_keys.py", "--stop-hook", env_extra=env)
    payload = json.loads(r.stdout)
    assert payload.get("decision") == "block", (
        f"a TRACKED answer key did not block the Stop hook; payload={payload!r} rc={r.returncode}")
    assert leak in payload.get("reason", ""), "the block does not name the leaked path"


def test_a_tracked_answer_key_fails_the_plain_gate(tmp_path):
    """Same violation, pre-commit/CI dialect: non-zero exit."""
    leak = "merlin/contract/capsules/conformance/_stop_hook_probe/hidden/answers.yaml"
    env = _index_with(tmp_path, leak, "answer: 42\n")
    r = _run("build_tools/scripts/check_no_answer_keys.py", env_extra=env)
    assert r.returncode != 0, f"a tracked answer key exited 0; stdout={r.stdout!r}"


def test_an_unreadable_index_is_a_refusal_not_a_pass(tmp_path):
    """"We could not look" must never read as "there is nothing to find".

    The gate used to return 0 when ``git ls-files`` failed, making an unexaminable tree indistinguishable
    from a clean one -- a green that cannot fail, on the repo's highest-severity invariant.
    """
    corrupt = tmp_path / "corrupt.index"
    corrupt.write_text("GARBAGE-NOT-AN-INDEX")
    env = {"GIT_INDEX_FILE": str(corrupt)}
    r = _run("build_tools/scripts/check_no_answer_keys.py", env_extra=env)
    assert r.returncode != 0, (
        "the answer-key gate passed while it could not read the index; an unexamined surface is a "
        f"refusal, not a clean bill of health. stdout={r.stdout!r} stderr={r.stderr[:300]!r}")
    r = _run("build_tools/scripts/check_no_answer_keys.py", "--stop-hook", env_extra=env)
    assert json.loads(r.stdout).get("decision") == "block", "unreadable index did not block the hook"
