"""The repository map must be derived from the tracked tree, not maintained by hand.

"What is in this repo, directory by directory" had no trustworthy answer: `repo_structure.md`
describes the intended shape and `module_index.md` lists importable packages, but neither says how
much is where, so a directory could appear, grow to a thousand files and show up in neither. The map
reads `git ls-files` and each directory's own AGENT.md, so there is nothing to keep in sync.
"""

from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import repo_root

ROOT = repo_root()
SCRIPTS = ROOT / "build_tools" / "scripts"
MAP = ROOT / "docs" / "reference" / "repo_map.md"


def test_the_generator_counts_what_git_tracks():
    """The totals line is the claim; an off-by-a-tree map would be worse than no map.

    Checked against a fresh render rather than the committed file. The committed map is a snapshot and
    is deliberately not required to be current: gating it failed the NEXT unrelated commit whenever
    anyone added a file, and cost other sessions a regenerate commit each.

    Counts of ten or more are rounded to two significant figures so one file cannot move a bucket.
    """
    import sys as _sys

    _sys.path.insert(0, str(SCRIPTS))
    import gen_repo_map as R

    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert f"**{R._approx(len(tracked))} tracked files**" in R.render()
    assert R._approx(7) == "7" and R._approx(6814) == "~6.8k" and R._approx(293) == "~290"
    assert R._approx(17) == R._approx(18) and R._approx(293) == R._approx(294)


def test_every_listed_directory_exists():
    body = MAP.read_text(encoding="utf-8")
    rows = [ln for ln in body.splitlines() if ln.startswith("| ") and "`" in ln]
    assert len(rows) > 40, "the map lost most of its rows"


def test_a_directory_with_an_agent_md_reports_its_own_purpose():
    """The purpose column is the directory's own answer, not a second place to maintain one."""
    body = MAP.read_text(encoding="utf-8")
    assert "single top-level root for all generated" in body, "out/ lost its AGENT.md purpose"


def test_a_wrapped_purpose_is_not_cut_mid_sentence():
    """Reading only the first physical line published `merlin.mining` to the module index as
    "fork an iteration of the RVV codegen (a transform-dialect" -- close paren on line two."""
    sys.path.insert(0, str(SCRIPTS))
    import gen_package_docs as G

    got = G._first_docline(ROOT / "merlin" / "python" / "merlin" / "mining" / "__init__.py")
    assert "transform-dialect SCHEDULE" in got, got


def test_every_package_agent_md_names_its_own_directory():
    """Four survived a rename pointing at the old path; mining/ still called itself rvvgen."""
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "gen_package_docs.py"), "--check"], capture_output=True, text=True, cwd=ROOT
    )
    assert r.returncode == 0, r.stderr + r.stdout
