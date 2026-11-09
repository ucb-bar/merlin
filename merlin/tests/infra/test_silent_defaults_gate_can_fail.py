"""The silent-default gate must be able to tell a fabricated datum from an absence.

``.get("explicit_cycles", 0)`` could not distinguish "the run did not record cycles" from "the run
recorded zero cycles", and the zero went into a ratio that was then cited. The gate exists to find
that shape on the paths where a wrong answer becomes a verdict.

The mutation each test applies is a single silent default in otherwise clean source. It removes every
satisfier for that finding at once, because the finding has exactly one: the shape is either in the
``ast`` or it is not. The pairs below are the load-bearing part -- for every shape the gate flags,
there is a neighbouring spelling it must NOT flag, because a scanner that reports everything passes a
one-sided test just as well as a scanner that reports the right things.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SD = _load("check_silent_defaults")


def _ownership(root, analysis=()):
    components = {
        owner: {"namespaces": [], "modules": []}
        for owner in ("merlin-experiments", "merlin-dse", "merlin-mining", "merlin-analysis")
    }
    components["merlin-analysis"]["modules"] = list(analysis)
    components["merlin-experiments"]["namespaces"] = ["merlin_experiments"]
    path = root / "build_tools/logical_components.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "merlin.logical_components.v1", "research": components}))


def _shapes(source: str) -> list[tuple[str, str]]:
    return [(f.shape, f.subject) for f in SD.findings_for_source(source, "probe.py")]


# --------------------------------------------------------------------- the four flagged shapes

FLAGGED = {
    "get-default (a number)": ("def grade(r):\n    return r.get('explicit_cycles', 0)\n", "get-default"),
    "get-default (a verdict)": ("def grade(r):\n    return r.get('status', 'ok')\n", "get-default"),
    "get-default (a bool)": ("def grade(r):\n    return r.get('cycle_accurate', True)\n", "get-default"),
    "swallowed-error": (
        "import json\n\n\ndef load(p):\n    try:\n        return json.loads(p.read_text())\n"
        "    except ValueError:\n        pass\n",
        "swallowed-error",
    ),
    "unchecked-subprocess": (
        "import subprocess\n\n\ndef go():\n    subprocess.run(['make'])\n",
        "unchecked-subprocess",
    ),
    "shell-pipe-status": (
        "import subprocess\n\n\ndef go():\n    subprocess.run('make | tee log', shell=True, check=True)\n",
        "shell-pipe-status",
    ),
}


@pytest.mark.parametrize("name", sorted(FLAGGED))
def test_each_shape_reaches_the_verdict(name):
    source, shape = FLAGGED[name]
    found = _shapes(source)
    assert shape in [s for s, _ in found], f"{name} produced no {shape} finding; got {found}"


# ------------------------------------------------- the neighbouring spellings it must NOT flag

CLEAN = {
    "a read with no default": "def grade(r):\n    return r['explicit_cycles']\n",
    "a one-argument get": "def grade(r):\n    return r.get('explicit_cycles')\n",
    "an explicit None": "def grade(r):\n    return r.get('explicit_cycles', None)\n",
    "an empty container": "def grade(r):\n    return r.get('regions', [])\n",
    "a default that SAYS unknown": "def grade(r):\n    return r.get('status', 'unmeasured')\n",
    "a default that is derived, not fabricated": "def grade(r, d):\n    return r.get('cycles', d.derive())\n",
    "an except that records": (
        "import json\n\n\ndef load(p, log):\n    try:\n        return json.loads(p.read_text())\n"
        "    except ValueError as exc:\n        log.append(exc)\n"
    ),
    "a checked subprocess": "import subprocess\n\n\ndef go():\n    subprocess.run(['make'], check=True)\n",
    "an inspected subprocess": (
        "import subprocess\n\n\ndef go():\n    done = subprocess.run(['make'])\n    return done.returncode\n"
    ),
}


@pytest.mark.parametrize("name", sorted(CLEAN))
def test_the_right_spelling_is_not_flagged(name):
    assert _shapes(CLEAN[name]) == [], f"{name} was flagged; the scanner reports things it should not"


def test_the_unknown_marker_exemption_is_the_whole_discrimination():
    """`get('status', 'ok')` and `get('status', 'unmeasured')` differ only in the fallback's MEANING.

    The first asserts a verdict nobody read; the second records that nobody read it. A gate that
    cannot separate those two would either flag the fix or miss the defect, and either way it is the
    same could-not-discriminate bug it exists to catch.
    """
    assert _shapes("def g(r):\n    return r.get('status', 'ok')\n")
    assert not _shapes("def g(r):\n    return r.get('status', 'unmeasured')\n")
    assert SD.is_unknown_marker("(UNKNOWN)") and not SD.is_unknown_marker("ok")


# --------------------------------------------------------------------- failing closed, and the ledger


def test_unparseable_source_is_a_finding_not_a_clean_answer():
    """An evidence file that will not parse must not scan as having nothing wrong with it."""
    found = SD.findings_for_source("def broken(:\n", "probe.py")
    assert [f.subject for f in found] == ["unparseable source"]


def test_a_ratcheted_finding_is_forgiven_and_a_new_one_is_not():
    finding = SD.Finding("probe.py", "grade", "get-default", "explicit_cycles")
    assert SD.verdict([finding], set())[0] == [finding]
    assert SD.verdict([finding], {finding.key})[0] == []


def test_a_ledger_entry_whose_finding_is_gone_fails():
    """Otherwise the ledger rots into an allowlist and stops being a may-only-shrink ledger."""
    assert SD.verdict([], {"probe.py::grade::get-default::explicit_cycles"})[1] == [
        "probe.py::grade::get-default::explicit_cycles"
    ]


def test_a_roster_entry_that_matches_nothing_fails_closed(monkeypatch):
    """A file moves, the roster still names the old path, and the gate quietly stops covering it.

    That is this gate committing the defect it is for, so it raises instead of scanning less.
    """
    monkeypatch.setattr(SD, "EVIDENCE_ROOTS", ("build_tools/scripts/check_a_gate_that_never_existed.py",))
    with pytest.raises(FileNotFoundError):
        SD._scanned_files()


def test_the_live_scan_covers_real_files():
    """A scan over an empty file set reports the same clean answer as a scan that found nothing."""
    files = SD._scanned_files()
    assert len(files) > 50, f"the evidence roster resolved to {len(files)} files; it is not scanning the repo"


def test_moved_verifier_body_keeps_its_debt_but_new_defaults_still_fail(tmp_path, monkeypatch):
    _ownership(tmp_path)
    owner = tmp_path / "src/merlin/verification/probe.py"
    owner.parent.mkdir(parents=True)
    owner.write_text("def grade(r):\n    return r.get('cycles', 0), r.get('status', 'ok')\n")
    former = tmp_path / "src/merlin/verify/probe.py"
    former.parent.mkdir(parents=True)
    former.write_text(
        '"""Compatibility identity."""\nimport sys\n'
        "from merlin.verification import probe as _owner\nsys.modules[__name__] = _owner\n"
    )
    monkeypatch.setattr(SD, "ROOT", tmp_path)
    monkeypatch.setattr(SD, "EVIDENCE_ROOTS", ("src/merlin/verify",))
    (owner.parent / "outside_declared_scope.py").write_text("def other(r): return r.get('result', True)\n")
    found = SD.scan()
    assert len(found) == 2 and all(f.path == "src/merlin/verification/probe.py" for f in found)
    known = {"src/merlin/verify/probe.py::grade::get-default::cycles"}
    new, stale = SD.verdict(found, known)
    assert [finding.subject for finding in new] == ["status"] and stale == []
    # A real operation in the old file makes it a separate implementation, not an alias.
    former.write_text(former.read_text() + "EXTRA = True\n")
    assert len(SD.verdict(found, known)[0]) == 2


def test_relocated_research_debt_normalizes_both_sides():
    finding = SD.Finding("src/merlin_experiments/phase1/audit.py", "audit", "get-default", "status")
    known = {"packages/merlin-experiments/src/merlin_experiments/phase1/audit.py::audit::get-default::status"}
    assert SD.verdict([finding], known) == ([], [])


def test_core_directory_scope_does_not_expand_when_research_joins_the_wheel(tmp_path, monkeypatch):
    _ownership(tmp_path, analysis=("merlin.verify.plots",))
    sources = {
        "src/merlin/verify/proof.py": "def grade(r): return r.get('cycles', 0)\n",
        "src/merlin/verify/plots.py": "def plot(r): return r.get('bars', 0)\n",
        "src/merlin_experiments/phase1/feedback/grader.py": "def grade(r): return r.get('status', 'ok')\n",
    }
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(SD, "ROOT", tmp_path)
    selected = ("src/merlin/verify", "src/merlin_experiments/phase1/feedback")
    monkeypatch.setattr(SD, "EVIDENCE_ROOTS", selected)
    assert {f.path for f in SD.scan()} == {
        "src/merlin/verify/proof.py",
        "src/merlin_experiments/phase1/feedback/grader.py",
    }
    # An explicit file selection still scans research; ownership is not a blanket exemption.
    monkeypatch.setattr(SD, "EVIDENCE_ROOTS", (*selected, "src/merlin/verify/plots.py"))
    assert {f.path for f in SD.scan()} == set(sources)
    (tmp_path / "build_tools/logical_components.json").unlink()
    with pytest.raises(FileNotFoundError):
        SD._scanned_files()
