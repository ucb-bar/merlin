"""Regression tests proving the Phase-1 overfit GATES actually bite.

The three de-overfit gates (``check_no_assumed_constants`` / ``check_no_target_name`` / ``check_no_regex``)
are the automated enforcement of CLAUDE.md's cardinal rule: no baked ISA constants, no hardcoded target
names, no regex in core library code. A gate that silently stopped flagging (a refactor broke its AST
visitor; a marker check inverted) would let the whole class of overfit back in un-noticed — so we plant a
KNOWN violation of each and assert (a) the gate's own ``_scan_file`` flags it, and (b) the sanctioned
inline suppression marker silences it. A benign file trips none of the three.

The gates scan fixed SCAN_ROOTS at ``main()``; we bypass that and call each gate's ``_scan_file`` on a
crafted temp file directly, so the test is hermetic (no dependence on the live tree's contents) and
targets the exact detection logic each gate exposes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load_gate(name: str):
    """Import a build_tools gate script by file path (they are standalone scripts, not a package)."""
    path = _SCRIPTS / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"gate script missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- assumed ISA constants


def test_assumed_constant_gate_flags_and_marker_silences(tmp_path):
    """A baked ISA-identity constant (an ``*opcode*``-named target bound to a numeric literal) is flagged
    by ``check_no_assumed_constants``, and a ``# derived-ok:`` marker on that line silences it — proving
    the gate bites AND its sanctioned escape hatch works."""
    gate = _load_gate("check_no_assumed_constants")

    flagged = _write(tmp_path, "planted.py", "CUSTOM_OPCODE = 0x7B\n")
    hits = gate._scan_file(flagged)
    assert hits, "gate failed to flag a baked CUSTOM_OPCODE literal"
    assert any("opcode" in label.lower() for _ln, label, _val in hits)

    silenced = _write(tmp_path, "marked.py",
                      "CUSTOM_OPCODE = 0x7B  # derived-ok: RISC-V custom-3 standard, documented fallback\n")
    assert gate._scan_file(silenced) == [], "# derived-ok: marker did not silence the baked constant"


def test_assumed_constant_gate_ignores_a_benign_mask(tmp_path):
    """The gate stays high-signal: a bit-mask constant (not an ISA-identity name) is NOT flagged, so the
    gate is not a false-positive machine that would get everything allowlisted."""
    gate = _load_gate("check_no_assumed_constants")
    benign = _write(tmp_path, "benign.py", "MASK32 = 0xFFFFFFFF\nrows = (v >> 48) & 0xFFFF\n")
    assert gate._scan_file(benign) == []


# --------------------------------------------------------------------------- hardcoded target names


def test_target_name_gate_flags_import_and_filename_and_marker_silences(tmp_path):
    """A hardcoded target-name IMPORT and a target-named module FILENAME are both flagged by
    ``check_no_target_name``, and a ``# target-ok:`` marker silences the import line — proving the gate
    catches the two structural coupling surfaces and honors its escape hatch."""
    gate = _load_gate("check_no_target_name")

    body = "import merlin.targets.gemmini.backend\n"
    flagged = _write(tmp_path, "planted.py", body)
    # Surface (2) import + surface (3) filename (the rel path itself names a target).
    hits = gate._scan_file(flagged, "merlin/python/merlin/cost_model/gemmini.py")
    kinds = {kind for _ln, _name, _snip, kind in hits}
    assert "import" in kinds, f"import coupling not flagged: {hits}"
    assert "filename" in kinds, f"target-named filename not flagged: {hits}"
    assert all(name == "gemmini" for _ln, name, _snip, _k in hits)

    silenced_body = "import merlin.targets.gemmini.backend  # target-ok: pending eviction reference\n"
    silenced = _write(tmp_path, "marked.py", silenced_body)
    # rel path has no target name now, so only the (silenced) import surface is in play.
    assert gate._scan_file(silenced, "merlin/python/merlin/shared_ok.py") == []


def test_target_name_gate_ignores_a_generic_module(tmp_path):
    """A module with no target name in its imports/literals/filename trips nothing — the gate does not
    flag ordinary code."""
    gate = _load_gate("check_no_target_name")
    benign = _write(tmp_path, "benign.py", "import os\nTARGET = resolve_target()\n")
    assert gate._scan_file(benign, "merlin/python/merlin/generic_thing.py") == []


# --------------------------------------------------------------------------- regex in core library


def test_regex_gate_flags_and_marker_silences(tmp_path):
    """A ``re.compile(...)`` call site is flagged by ``check_no_regex``, and a ``# regex-ok:`` marker on
    that line silences it — proving the no-regex gate bites AND honors its escape hatch."""
    gate = _load_gate("check_no_regex")

    flagged = _write(tmp_path, "planted.py", "import re\nPAT = re.compile('x')\n")
    hits = gate._scan_file(flagged)
    assert hits and any("compile" in what for _ln, what in hits), f"regex call not flagged: {hits}"

    silenced = _write(tmp_path, "marked.py",
                      "import re\nPAT = re.compile('x')  # regex-ok: opaque inline-asm parse\n")
    assert gate._scan_file(silenced) == [], "# regex-ok: marker did not silence the regex call"


def test_regex_gate_ignores_a_file_with_no_regex(tmp_path):
    """A file that never touches the ``re`` module trips nothing — the gate flags call sites, not the
    substring ``re`` in unrelated identifiers."""
    gate = _load_gate("check_no_regex")
    benign = _write(tmp_path, "benign.py", "def resolve(x):\n    return x.split('_')\n")
    assert gate._scan_file(benign) == []


# --------------------------------------------------------------------------- a fully benign file is clean


def test_benign_file_passes_all_three_gates(tmp_path):
    """One innocuous module passes ALL three gates — the negative control that ensures the gates above
    flagged the PLANTED violation, not merely any file handed to them."""
    a = _load_gate("check_no_assumed_constants")
    t = _load_gate("check_no_target_name")
    r = _load_gate("check_no_regex")
    benign = _write(tmp_path, "clean.py",
                    "def add(a, b):\n"
                    "    '''Add two numbers.'''\n"
                    "    return a + b\n")
    assert a._scan_file(benign) == []
    assert t._scan_file(benign, "merlin/python/merlin/clean.py") == []
    assert r._scan_file(benign) == []
