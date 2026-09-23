"""One structural scanner, distinct repository and candidate exemption policies."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

from merlin.common.paths import module_source_path, repo_root


def _conformance():
    from merlin_experiments.phase1 import conformance

    return conformance


def _submission(tmp_path, source="import re\nre.search('x', 'x')\n"):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "compiler.py").write_text(source)
    return submission


def test_missing_scanner_is_unavailable_not_clean(tmp_path, monkeypatch):
    conformance = _conformance()
    monkeypatch.setattr(conformance, "_load_scan_file", lambda: None)
    result = conformance.compute(
        tmp_path / "absent.jsonl", _submission(tmp_path), "fixture", "", resolved_tools=["xdsl_kit"]
    )
    assert result["checks"]["no_regex_ok"] is False
    assert result["regex_scan_error"]
    assert result["conformant"] is False


def test_candidate_cannot_self_exempt_with_repository_marker(tmp_path):
    conformance = _conformance()
    result = conformance.compute(
        tmp_path / "absent.jsonl",
        _submission(tmp_path, "import re\nre.search('x', 'x')  # regex-ok: candidate says allowed\n"),
        "fixture",
        "",
        resolved_tools=["xdsl_kit"],
    )
    assert result["checks"]["no_regex_ok"] is False
    assert result["regex_hits"] == [{"file": "compiler.py", "line": 2, "kind": "re.search"}]


@pytest.mark.parametrize("failure", [OSError("unreadable"), SyntaxError("malformed candidate")])
def test_failed_scan_is_unavailable_not_clean(tmp_path, monkeypatch, failure):
    conformance = _conformance()

    def scan(path):
        raise failure

    monkeypatch.setattr(conformance, "_load_scan_file", lambda: scan)
    result = conformance.compute(
        tmp_path / "absent.jsonl", _submission(tmp_path), "fixture", "", resolved_tools=["xdsl_kit"]
    )
    assert result["checks"]["no_regex_ok"] is False
    assert type(failure).__name__ in result["regex_scan_error"]


def test_non_xdsl_treatment_does_not_require_scanner(tmp_path, monkeypatch):
    conformance = _conformance()
    monkeypatch.setattr(conformance, "_load_scan_file", lambda: pytest.fail("inapplicable scanner invoked"))
    result = conformance.compute(tmp_path / "absent", _submission(tmp_path), "fixture", "", resolved_tools=[])
    assert result["checks"]["no_regex_ok"] is None
    assert "regex_scan_error" not in result


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import re\nre.search('x', 'x')\n", [(2, "re.search")]),
        ("import re as rx\nrx.compile('x')\n", [(2, "rx.compile")]),
        ("from re import fullmatch\nfullmatch('x','x')\n", [(2, "fullmatch")]),
        ("import re\ndef f():\n    return re.sub('x','y','x')\n", [(3, "re.sub")]),
        ("import re\nre.search(re.escape('x'), 'x')\n", [(2, "re.escape"), (2, "re.search")]),
        ("import re\ntext = 're.search(x)'\n", []),
        ("import ast\nast.parse('x')\n", []),
        ("from re import search as find\nfind('x','x')\n", [(2, "find")]),
        ("from re import compile as make_pattern\nmake_pattern('x')\n", [(2, "make_pattern")]),
        ("from re import search as split\nsplit('x','x')\n", [(2, "split")]),
        (
            "from re import search as find, sub as replace\nfind('x',replace('y','x','y'))\n",
            [(2, "find"), (2, "replace")],
        ),
        ("from re import purge\npurge()\n", []),
        ("from re import purge as search\nsearch()\n", []),
        ("from other_module import search as find\nfind('x','x')\n", []),
    ],
)
def test_shared_ast_findings_preserve_existing_detection(source, expected):
    from merlin.common.regex_scan import scan_source

    assert scan_source(source) == expected


def test_candidate_renamed_regex_import_cannot_bypass_policy(tmp_path):
    result = _conformance().compute(
        tmp_path / "absent",
        _submission(tmp_path, "from re import search as find\nfind('x', 'x')\n"),
        "fixture",
        "",
        resolved_tools=["xdsl_kit"],
    )
    assert result["checks"]["no_regex_ok"] is False
    assert result["regex_hits"] == [{"file": "compiler.py", "line": 2, "kind": "find"}]


def test_build_policy_keeps_marker_allowlist_and_syntax_handling(tmp_path, monkeypatch):
    source = repo_root() / "build_tools/scripts/check_no_regex.py"
    spec = importlib.util.spec_from_file_location("_regex_gate", source)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    candidate = tmp_path / "compiler.py"
    candidate.write_text("import re\nre.search('x', 'x')  # regex-ok: reviewed repository exception\n")
    assert gate._scan_file(candidate) == []
    candidate.write_text("import re\nre.search('x', 'x')\n")
    assert gate._scan_file(candidate) == [(2, "re.search")]
    candidate.write_text("from re import search as find\nfind('x', 'x')\n")
    assert gate._scan_file(candidate) == [(2, "find")]
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_iter_targets", lambda staged: [candidate.relative_to(tmp_path)])
    monkeypatch.setattr(gate, "_load_allowlist", lambda: {"compiler.py"})
    assert gate.main([]) == 0
    monkeypatch.setattr(gate, "_load_allowlist", set)
    assert gate.main([]) == 1
    candidate.write_text("def broken(:\n")
    assert gate._scan_file(candidate) == []
    from merlin.common.regex_scan import scan_file

    with pytest.raises(SyntaxError):
        scan_file(candidate)


def test_conformance_uses_core_without_checkout_build_tools(tmp_path):
    """Copy only the policy owner; use installed-shaped core imports under -I -S."""
    source = module_source_path("merlin_experiments.phase1.conformance")
    copied = tmp_path / "conformance.py"
    copied.write_bytes(source.read_bytes())
    submission = _submission(tmp_path)
    code = """
import importlib.util, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
os.environ['MERLIN_REPO_ROOT'] = str(Path.cwd())
spec = importlib.util.spec_from_file_location('conformance', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module._load_scan_file().__module__ == 'merlin.common.regex_scan'
print(json.dumps(module.compute(Path('absent'), Path(sys.argv[3]), 'fixture', '', resolved_tools=['xdsl_kit'])))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            code,
            str(copied),
            str(module_source_path("merlin").parent.parent),
            str(submission),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checks"]["no_regex_ok"] is False


def test_scanner_bytes_belong_to_phase1_source_inventory(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    scanner = tmp_path / "scanner.py"
    scanner.write_bytes(module_source_path("merlin.common.regex_scan").read_bytes())
    original = SI._source
    monkeypatch.setattr(SI, "_source", lambda name: scanner if name == "merlin.common.regex_scan" else original(name))
    context = {"repo": tmp_path, "entrypoint": tmp_path / "synthetic_entrypoint.py"}
    record = SI.record(**context)
    assert record["inputs"]["phase1:startup:python_regex_scanner"]["path"] == str(scanner)
    SI.verify(record, **context)
    scanner.write_text("# changed scanner\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **context)


def test_candidate_under_generated_build_parent_is_still_scanned(tmp_path):
    parent = tmp_path / "build/work"
    parent.mkdir(parents=True)
    result = _conformance().compute(
        tmp_path / "absent", _submission(parent), "fixture", "", resolved_tools=["xdsl_kit"]
    )
    assert result["checks"]["no_regex_ok"] is False


def test_candidate_generated_descendants_remain_excluded(tmp_path):
    submission = _submission(tmp_path, "import ast\nast.parse('x')\n")
    for name in ("build", "__pycache__"):
        generated = submission / name
        generated.mkdir()
        (generated / "generated.py").write_text("import re\nre.search('x','x')\n")
    result = _conformance().compute(tmp_path / "absent", submission, "fixture", "", resolved_tools=["xdsl_kit"])
    assert result["checks"]["no_regex_ok"] is True
    assert result["regex_hits"] == []
