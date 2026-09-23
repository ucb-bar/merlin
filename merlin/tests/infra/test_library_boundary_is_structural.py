"""The library/experiments boundary scan reads syntax, not lines.

The rule is real: library code (``merlin/python/merlin``) reads its inputs from ``benchmarks/`` and
``contract/`` and reaches a corpus still under ``experiments/`` only through
``merlin.targetgen.corpora``. The SCAN was not. It split each line at ``#`` and looked for the text
``experiments/`` anywhere in what remained, so a docstring that EXPLAINS the boundary counted as
breaking it -- eleven of thirty reported violations were prose, including the docstring of the very
resolver that implements the shorthand. And the fix it advised was to use ``merlin.common.corpora``,
a module that has never existed in this repo.

A gate whose census is a third false and whose remedy names nothing is worse than no gate: it is read
once, disbelieved, and thereafter ignored. These tests hold the replacement to being structural (it
parses the module and asks whether a STRING LITERAL names ``experiments`` as a path component), to
failing closed on a module it cannot parse, and to naming a module that exists.
"""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_structure.py"
RATCHET = repo_root() / "build_tools" / "scripts" / "library_boundary_ratchet.txt"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("check_structure_under_test", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan(gate, tmp_path, monkeypatch, source, *, ratchet=None):
    """Run the boundary check over a one-module tree standing in for the library."""
    package = tmp_path / "merlin" / "python" / "merlin"
    package.mkdir(parents=True, exist_ok=True)
    (package / "probe.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", str(tmp_path))
    monkeypatch.setattr(gate, "BOUNDARY_RATCHET", str(ratchet) if ratchet else str(tmp_path / "absent.txt"))
    errors: list[str] = []
    gate.check_library_boundary(errors)
    return errors


def test_a_docstring_that_explains_the_rule_does_not_break_it(gate, tmp_path, monkeypatch):
    """The regression. This is the real docstring shape that was reported eleven times."""
    source = '''"""Grants are repo-root-relative, with one documented shorthand: a leading
``experiments/...`` resolves under ``merlin/``, so the prefix may be elided."""


def resolve(rel):
    """An ``experiments/`` grant resolves under merlin/ when only that path exists."""
    return rel
'''
    assert _scan(gate, tmp_path, monkeypatch, source) == []


def test_a_refusal_message_is_prose_not_a_path(gate, tmp_path, monkeypatch):
    """Not every non-path string is a docstring. A message explaining WHY the library refuses to read
    a corpus necessarily names the directory it is refusing, and must not itself be the violation."""
    source = 'REASON = "the only schedule contract lives under merlin/experiments/, which library code may not read"\n'
    assert _scan(gate, tmp_path, monkeypatch, source) == []


def test_a_path_literal_is_still_caught(gate, tmp_path, monkeypatch):
    """The gate must not have been softened into uselessness: both spellings still fail."""
    joined = _scan(gate, tmp_path, monkeypatch, 'P = os.path.join(root, "experiments", "x")\n')
    direct = _scan(gate, tmp_path, monkeypatch, 'P = root / "experiments/cpu_host_v0/grader.py"\n')
    assert len(joined) == 1 and len(direct) == 1
    assert "probe.py:1" in joined[0] and "probe.py:1" in direct[0]


def test_the_remedy_names_a_module_that_exists(gate, tmp_path, monkeypatch):
    """The old message advised `merlin.common.corpora`, which is not importable and never was."""
    errors = _scan(gate, tmp_path, monkeypatch, 'P = "experiments"\n')
    assert "merlin.targetgen.corpora" in errors[0]
    assert "merlin.common.corpora" not in errors[0]
    importlib.import_module("merlin.targetgen.corpora")


def test_a_component_is_compared_whole(gate, tmp_path, monkeypatch):
    """`experiments` as a path COMPONENT, not as a substring: a sibling directory or a file whose name
    happens to contain the word is not a boundary crossing."""
    source = 'A = "out/artifacts/experiments_archive/x"\nB = "my_experiments.yaml"\nC = "preexperiments"\n'
    assert _scan(gate, tmp_path, monkeypatch, source) == []


def test_a_module_that_does_not_parse_is_reported_not_skipped(gate, tmp_path, monkeypatch):
    """Fail closed. An unscanned module is not a clean one, and a scan that silently drops the files
    it could not read reports a census that is quietly short."""
    errors = _scan(gate, tmp_path, monkeypatch, 'def broken(:\n    "experiments/x"\n')
    assert len(errors) == 1
    assert "unscannable" in errors[0] and "probe.py" in errors[0]


def test_a_ratcheted_module_is_excused_and_nothing_else_is(gate, tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("# debt\nmerlin/python/merlin/probe.py  # 1\n", encoding="utf-8")
    source = 'P = "experiments"\n'
    assert _scan(gate, tmp_path, monkeypatch, source, ratchet=ledger) == []
    assert len(_scan(gate, tmp_path, monkeypatch, source)) == 1


def test_the_recorded_debt_is_what_the_tree_actually_has(gate):
    """The ratchet is a measurement of this tree, not a wish. Every entry must still be a real module,
    so an entry cannot outlive the file it excuses and quietly widen the gate."""
    entries = [
        line.split("#", 1)[0].strip()
        for line in RATCHET.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    assert entries, "the ledger must list the debt it claims to hold"
    assert entries == sorted(entries), "sorted, so a diff shows what changed"
    import _source_layout as layout

    physical = {
        layout.policy_path(path.as_posix()): repo_root() / path
        for path in layout.python_files(repo_root(), layout.SOURCE_SCAN_ROOTS)
    }
    for entry in entries:
        assert entry in physical and physical[entry].is_file(), f"ratchet names a module that is gone: {entry}"
