"""A tracked file may not name one person's home or scratch directory.

This repo is public, and an absolute path into someone's account is two defects at once: a
disclosure, and a default no other clone can follow -- so the feature it configures is silently
unavailable everywhere else. The sweep that added this gate found a compiler wrapper that ``exec``'d
a clang inside ONE worktree of ONE checkout, two library modules pointing at a directory that no
longer exists on this machine either, and a test whose temp root sat outside any git repository,
which made the gate it was exercising die rather than assert.
"""

from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import repo_root

ROOT = repo_root()
GATE = ROOT / "build_tools" / "scripts" / "check_no_local_paths.py"


def _run(cwd=None):
    return subprocess.run([sys.executable, str(GATE)], cwd=cwd or ROOT, capture_output=True, text=True, timeout=300)


def test_the_tree_is_clean_or_ratcheted():
    r = _run()
    assert r.returncode == 0, r.stderr + r.stdout


def _repo(tmp_path, body: str, name: str = "thing.py"):
    import shutil

    scripts = tmp_path / "build_tools" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(GATE, scripts / GATE.name)
    (tmp_path / name).write_text(body, encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-qm", "c"],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return subprocess.run([sys.executable, str(scripts / GATE.name)], cwd=tmp_path, capture_output=True, text=True)


#: Assembled at run time, never written out as one literal -- a contiguous spelling in this file
#: would be a personal path in a tracked file, and the gate would (correctly) fail on its own test.
_ACCOUNT = "some" + "person"
_PERSONAL = "/scratch/" + _ACCOUNT


def test_a_personal_path_fails_and_the_ratchet_excuses_it(tmp_path):
    r = _repo(tmp_path, f'HOME_DIR = "{_PERSONAL}/projects/thing"\n')
    assert r.returncode == 1 and _PERSONAL in r.stderr, r.stdout + r.stderr

    (tmp_path / "build_tools" / "scripts" / "no_local_paths_ratchet.txt").write_text(
        "# debt\nthing.py\n", encoding="utf-8"
    )
    r2 = subprocess.run(
        [sys.executable, str(tmp_path / "build_tools/scripts" / GATE.name)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr


def test_the_replacements_the_gate_asks_for_all_pass(tmp_path):
    """The three fixes named in the failure message must actually satisfy it."""
    body = (
        'A = env("MERLIN_EXT_THING")\n'
        'B = repo_root().parent / "sibling-checkout"\n'
        'C = Path(os.environ.get("TMPDIR", "/tmp")) / "work"\n'
        'D = "/path/to/thing"      # documentation placeholder\n'
        'E = "$HOME/thing"\n'
    )
    assert _repo(tmp_path, body).returncode == 0


def test_a_relative_path_containing_scratch_is_not_a_personal_path(tmp_path):
    """`out/scratch/whatever.json` is a fixture path, not an account -- the match must be anchored
    at the start of an absolute path."""
    assert _repo(tmp_path, 'FIXTURE = "out/scratch/whatever.json"\n').returncode == 0


def test_a_shared_service_directory_is_not_a_person(tmp_path):
    """/scratch/firesim_queue is the queue every session submits to."""
    assert _repo(tmp_path, 'QUEUE = "/scratch/firesim_queue/bin/firesim-queue"\n').returncode == 0


def test_a_one_letter_segment_is_a_test_placeholder(tmp_path):
    """`/home/x` in a fabricated environment dict is noise; a ledger full of noise stops being read."""
    assert _repo(tmp_path, 'ENV = {"HOME": "/home/x"}\n').returncode == 0


def test_it_refuses_to_pass_when_it_could_not_look(tmp_path):
    """ "We could not look" is not "there is nothing to find" -- the same shape check_no_answer_keys
    and check_artifact_layout were both hardened against."""
    import shutil

    scripts = tmp_path / "build_tools" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(GATE, scripts / GATE.name)  # no `git init` here
    r = subprocess.run([sys.executable, str(scripts / GATE.name)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0, "a non-repository reported OK having examined nothing"
