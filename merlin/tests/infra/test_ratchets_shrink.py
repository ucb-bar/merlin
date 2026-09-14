"""check_ratchets_shrink: a may-only-shrink ledger that grows must fail, in every mode it runs in."""
from __future__ import annotations

import subprocess
import sys

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_ratchets_shrink.py"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _gate(cwd, *args):
    r = subprocess.run([sys.executable, str(GATE), *args], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    led = tmp_path / "build_tools" / "scripts" / "x_ratchet.txt"
    led.parent.mkdir(parents=True)
    led.write_text("# header comment\na.py\nb.py  # rationale\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path, led


def test_unchanged_ledger_passes(repo):
    root, _ = repo
    rc, out = _gate(root)
    assert rc == 0 and "OK" in out, out


def test_growth_fails_and_names_the_new_entry(repo):
    root, led = repo
    led.write_text("# header comment\na.py\nb.py  # rationale\nc.py\n")
    rc, out = _gate(root)
    assert rc == 1 and "c.py" in out and "2 -> 3" in out, out


def test_comment_and_blank_lines_are_not_entries(repo):
    root, led = repo
    led.write_text("# header comment\n\n# another note\na.py\nb.py  # rationale changed\n\n")
    rc, out = _gate(root)
    assert rc == 0, out


def test_shrink_passes(repo):
    root, led = repo
    led.write_text("# header comment\na.py\n")
    rc, out = _gate(root)
    assert rc == 0 and "shrank" in out, out


def test_same_count_swap_passes_but_is_reported(repo):
    root, led = repo
    led.write_text("# header comment\na.py\nmoved/b.py\n")
    rc, out = _gate(root)
    assert rc == 0 and "moved/b.py" in out, out


def test_staged_mode_reads_the_index_not_the_working_tree(repo):
    root, led = repo
    led.write_text("a.py\nb.py\nc.py\n")          # grown on disk, not staged
    assert _gate(root, "--staged")[0] == 0
    _git(root, "add", str(led))
    rc, out = _gate(root, "--staged")
    assert rc == 1 and "c.py" in out, out


def test_growth_passes_only_with_an_accept_marker_added_in_the_same_change(repo):
    root, led = repo
    led.write_text("# header comment\n# growth-accepted: gate widened its scan\na.py\nb.py\nc.py\n")
    rc, out = _gate(root)
    assert rc == 0 and "ACCEPTED" in out and "gate widened its scan" in out, out
    # the marker is spent once committed: growing again needs a NEW reason
    _git(root, "commit", "-qam", "accepted")
    led.write_text(led.read_text() + "d.py\n")
    assert _gate(root)[0] == 1


def test_a_new_ledger_starts_its_own_baseline(repo):
    root, _ = repo
    (root / "build_tools" / "scripts" / "y_allowlist.txt").write_text("p.py\nq.py\n")
    rc, out = _gate(root)
    assert rc == 0 and "new ledger" in out, out


def test_unresolvable_base_fails_closed(repo):
    root, _ = repo
    rc, out = _gate(root, "--base", "deadbeefdeadbeef")
    assert rc == 1 and "NOTHING was compared" in out, out


def test_all_zero_base_is_unmeasured_not_ok(repo):
    root, _ = repo
    rc, out = _gate(root, "--base", "0" * 40)
    assert rc == 0 and "UNMEASURED" in out and "OK" not in out, out
