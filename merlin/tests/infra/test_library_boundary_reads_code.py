"""The library-boundary gate must read CODE, not the prose that explains it.

`check_library_boundary` forbids a library module from naming ``experiments/`` as a path. It scanned
each line with the ``#`` comment stripped -- but a docstring is not a comment, so every sentence that
mentioned the layout was reported as a read of it. Measured on this tree: 29 violations, of which 10
were docstrings, including all four in ``targetgen/sandbox/bwrap.py``, a file that constructs sandbox
bind paths and describes them in prose while importing nothing from experiments/.

A false positive in a gate is worse than a missing one here: the next author cannot tell which of the 29
lines is the real defect, and the message told them to use ``merlin.common.corpora`` -- a module that has
never existed. The sanctioned indirection is ``merlin.targetgen.corpora``, and the message now derives
its name from the allowlist entry rather than restating it.

The same defect then showed up one level down: the remaining scan still matched the WORD. Of the 13 it
reported, four were an argparse subcommand named ``experiments``, two equality tests against a parent
directory's name, and a ``startswith`` over declared-path prefixes -- none of which builds a path. The
scan now asks what the string is USED FOR (``_experiments_refs``), and these tests pin both directions:
a construction stays a violation in every spelling, and a non-path use is not one.
"""

from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import repo_root

_SPEC = importlib.util.spec_from_file_location(
    "check_structure", repo_root() / "build_tools/scripts/check_structure.py"
)
_CS = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("check_structure", _CS)
_SPEC.loader.exec_module(_CS)


def test_a_docstring_mentioning_the_layout_is_not_a_read():
    src = '''\
"""Bind the corpus, which lives under merlin/experiments/... for historical reasons."""


def f():
    """A second docstring naming "experiments" as a word."""
    return 1
'''
    assert _CS._prose_lines(src), "the docstrings must be located"
    assert 1 in _CS._prose_lines(src)


def test_a_string_in_code_is_still_a_read():
    """The mutation: only a BARE string expression is prose. A path built in code is a violation."""
    src = 'from pathlib import Path\np = Path("experiments") / "x"\n'
    assert 2 not in _CS._prose_lines(src), "a string used as a value is code, not documentation"


def test_an_unparsable_file_is_scanned_exactly_as_before():
    """Fail-closed: a syntax error must not silently exempt a file from the boundary rule."""
    assert _CS._prose_lines("def broken(:\n") == set()


def _refs(src):
    return _CS._experiments_refs(src)


def test_a_path_built_from_the_directory_is_flagged():
    """The gate's whole job. A real construction must stay a violation, in every spelling."""
    cases = [
        'from merlin.common.paths import merlin_dir\np = merlin_dir() / "experiments" / "capsule_bench"\n',
        'import os\np = os.path.join(root, "experiments", "capsule_bench")\n',
        'p = repo_root() / "merlin/experiments/capsule_bench/targets/x/target_experiment.yaml"\n',
        'roots = ["experiments/capsule_bench/targets"]\n',
        'txt = open("merlin/experiments/x/grader.py").read()\n',
        'p = f"experiments/{exp}/task/"\n',
    ]
    for src in cases:
        assert _refs(src), f"a path built under experiments/ must be reported: {src!r}"


def test_the_word_used_as_something_other_than_a_path_is_not_a_read():
    """The false positives this gate used to report: the token, not a path."""
    cases = [
        'exp = sub.add_parser("experiments", help="what ONE experiment costs")\n',  # a CLI subcommand
        'if args.command == "experiments":\n    pass\n',  # a parsed arg
        'if anc.name == "experiments":\n    pass\n',  # a parent directory's NAME
        'ok = p.startswith(("merlin/", "experiments/"))\n',  # a declared-path prefix test
        'if name in {"experiments", "benchmarks"}:\n    pass\n',  # a membership test
    ]
    for src in cases:
        assert _refs(src) == [], f"not a path construction, must not be reported: {src!r}"


def test_a_different_directory_that_merely_starts_with_the_word_is_not_a_read():
    """``experiments`` must be matched as a whole path COMPONENT, not as a substring."""
    assert _refs('p = out_dir() / "experiments_summary.json"\n') == []
    assert _refs('p = merlin_dir() / "benchmarks/experiments_v2"\n') == []


def test_an_unparsable_file_falls_back_to_the_text_scan():
    """Fail-closed: no AST means the textual scan, not an exemption."""
    assert _CS._experiments_refs("def broken(:\n") is None


def test_the_literal_allowlist_is_a_ratchet_over_real_lines():
    """Every allowlisted (file, literal) must still occur in that file — a stale entry is a hole."""
    import _source_layout as layout

    physical = {
        layout.policy_path(path.as_posix()): repo_root() / path
        for path in layout.python_files(repo_root(), layout.SOURCE_SCAN_ROOTS)
    }
    for (rel, literal), reason in _CS._BOUNDARY_LITERAL_ALLOW.items():
        assert rel in physical, f"allowlisted file is gone: {rel}"
        p = physical[rel]
        assert p.is_file(), f"allowlisted file is gone: {rel}"
        assert literal in p.read_text(encoding="utf-8"), f"stale allowlist entry: {rel} {literal!r}"
        assert reason.strip(), f"allowlist entry without a reason: {rel} {literal!r}"


def test_the_message_names_a_module_that_exists():
    """The instruction has to be followable. `merlin.common.corpora` never existed."""
    allow = sorted(_CS._BOUNDARY_ALLOW)[0]
    assert allow.endswith("corpora.py")
    assert (repo_root() / allow).is_file(), "the allowlisted indirection must be a real file"
    errors: list[str] = []
    _CS.check_library_boundary(errors)
    for e in errors:
        assert "merlin.targetgen.corpora" in e, e
        assert "merlin.common.corpora" not in e
