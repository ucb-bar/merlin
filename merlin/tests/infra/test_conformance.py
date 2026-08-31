"""Dev-conformance verdict (capsule-bench harness): the completion gate that catches a run scoring 0 by NOT using
the mandated tooling (invented encoding / regex parser / no lever enumeration / partial self-check),
which the numeric grade alone cannot distinguish from a real capability wall. Hermetic — synthetic
transcript + submission, no live run, no target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))
import conformance as C  # noqa: E402


def _transcript(path: Path, calls: list[tuple[str, dict]], *, failed: set[int] | None = None,
                missing: set[int] | None = None) -> Path:
    """Write a Codex-compatible transcript with correlated tool requests and results."""
    failed, missing = failed or set(), missing or set()
    ev = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": f"t{i}", "name": name, "input": inp} for i, (name, inp) in enumerate(calls)]}}
    results = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok" if i not in failed else "failed",
         "is_error": i in failed}
        for i in range(len(calls)) if i not in missing]}}
    path.write_text("\n".join((
        json.dumps({"type": "system", "subtype": "init", "driver": "codex"}),
        json.dumps(ev), json.dumps(results), "")))
    return path


def _submission(root: Path, py: dict[str, str]) -> Path:
    sub = root / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "manifest.yaml").write_text("entrypoints: {}\n")
    for name, text in py.items():
        (sub / name).write_text(text)
    return sub


def test_fully_conformant_run(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast  # xdsl pipeline, no regex\n"})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python cca_contract.py check-bijection atlas"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder atlas"}),
        ("bash", {"command": "python isa_tools.py asm ops.txt"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.S"}),
        ("bash", {"command": "python3 agent_selfcheck.py --submission submission --capsules all"}),
    ])
    v = C.compute(tp, sub, "merlin_rtlchecks", "external_backend")
    assert v["conformant"] is True
    assert C.failing_checks(v) == []


def test_regex_and_missing_tools_flagged(tmp_path):
    sub = _submission(tmp_path, {"iface_parser.py": "import re\nx = re.search('a', 'b')\n"})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python isa_tools.py disasm submission/kernel.S"}),  # isa_tools used, but no asm/cca
        ("bash", {"command": "python3 agent_selfcheck.py --capsules AT0"}),        # only one capsule
    ])
    v = C.compute(tp, sub, "merlin_rtlchecks", "external_backend")
    assert v["conformant"] is False
    bad = set(C.failing_checks(v))
    assert {"no_regex_ok", "asm_used", "cca_used", "full_selfcheck"} <= bad
    assert v["checks"]["isa_tools_used"] is True                                   # disasm counts
    assert any(h["kind"] == "re.search" for h in v["regex_hits"])


def test_prose_mention_is_not_a_tool_use(tmp_path):
    """The key false-positive guard: an agent that WRITES 'run isa_tools.py asm' into its plan but never
    executes it is NOT credited with using asm (only the executed command counts, not written content)."""
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("write_file", {"path": "docs/PLAN.md", "content": "step 5: run `isa_tools.py asm` and cca_contract"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.S"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    v = C.compute(tp, sub, "merlin_rtlchecks", "external_backend")
    assert v["checks"]["asm_used"] is False                                        # mentioned, never run
    assert v["checks"]["cca_used"] is False


def test_failed_or_unanswered_required_commands_do_not_satisfy_conformance(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python isa_tools.py asm ops.txt"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    tp = _transcript(tmp_path / "t.jsonl", calls, failed={1}, missing={0})
    v = C.compute(tp, sub, "merlin_rtlchecks", "external_backend")
    assert v["conformant"] is False
    assert v["checks"]["cca_used"] is False
    assert v["checks"]["asm_used"] is False
    assert v["checks"]["isa_tools_used"] is True  # the separate successful lint call still counts
    assert v["tool_evidence"] == {
        "n_calls": 4, "n_successful": 2, "n_failed": 1, "n_missing_results": 1,
    }


def test_asm_not_applicable_off_external_backend(tmp_path):
    """asm_used is external_backend-only (a RoCC/.insn arm emits MLIR, not hand-assembled words), so it is
    N/A (None) there and never a false violation."""
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder gemmini"}),
        ("bash", {"command": "python isa_tools.py lint submission/x.mlir"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    v = C.compute(tp, sub, "merlin_rtlchecks", "inline_asm_insn")
    assert v["checks"]["asm_used"] is None                                         # not applicable
    assert v["conformant"] is True


def test_raw_baseline_only_held_to_selfcheck(tmp_path):
    """raw_baseline grants no assisted tools, so cca/asm/isa are N/A — only the full self-check applies."""
    sub = _submission(tmp_path, {})                                               # no python authored
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    v = C.compute(tp, sub, "raw_baseline", "external_backend")
    assert v["checks"]["cca_used"] is None and v["checks"]["asm_used"] is None
    assert v["checks"]["no_regex_ok"] is None                                      # no .py to scan
    assert v["conformant"] is True
