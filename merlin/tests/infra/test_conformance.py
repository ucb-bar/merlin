"""Dev-conformance verdict (capsule-bench harness): the completion gate that catches a run scoring 0 by NOT using
the mandated tooling (invented encoding / regex parser / no lever enumeration / partial self-check),
which the numeric grade alone cannot distinguish from a real capability wall. Hermetic — synthetic
transcript + submission, no live run, no target.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))
import conformance as C  # noqa: E402
import run_baseline_qa_loop as LOOP  # noqa: E402


def _transcript(path: Path, calls: list[tuple[str, dict]], *, failed: set[int] | None = None,
                missing: set[int] | None = None, outputs: dict[int, str] | None = None) -> Path:
    """Write a Codex-compatible transcript with correlated tool requests and results."""
    failed, missing = failed or set(), missing or set()
    outputs = outputs or {}
    ev = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": f"t{i}", "name": name, "input": inp} for i, (name, inp) in enumerate(calls)]}}
    results = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": f"t{i}",
         "content": outputs.get(i, "ok" if i not in failed else "failed"),
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
    v = C.compute(tp, sub, "merlin_assisted", "external_backend", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
    })
    assert v["conformant"] is True
    assert C.failing_checks(v) == []


def test_regex_and_missing_tools_flagged(tmp_path):
    sub = _submission(tmp_path, {"iface_parser.py": "import re\nx = re.search('a', 'b')\n"})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python isa_tools.py disasm submission/kernel.S"}),  # isa_tools used, but no asm/cca
        ("bash", {"command": "python3 agent_selfcheck.py --capsules AT0"}),        # only one capsule
    ])
    v = C.compute(tp, sub, "merlin_assisted", "external_backend", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
    })
    assert v["conformant"] is False
    bad = set(C.failing_checks(v))
    assert {"no_regex_ok", "asm_used", "cca_used", "full_selfcheck"} <= bad
    assert v["checks"]["isa_tools_used"] is True                                   # disasm counts
    assert any(h["kind"] == "re.search" for h in v["regex_hits"])


def test_byte_identical_vendored_xdsl_regex_is_attributed_not_charged(
        tmp_path, monkeypatch: pytest.MonkeyPatch):
    vendored = "import re\nx = re.search('a', 'b')\n"
    digest = hashlib.sha256(vendored.encode()).hexdigest()
    monkeypatch.setattr(C, "_vendored_xdsl_hashes", lambda: frozenset({digest}))
    sub = _submission(tmp_path, {"compiler.py": "import ast\n"})
    (sub / "xdsl").mkdir()
    (sub / "xdsl/parser.py").write_text(vendored)
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn",
                  resolved_tools={"xdsl_kit"})
    assert v["checks"]["no_regex_ok"] is True
    assert v["regex_hits"] == []
    assert v["vendored_regex_files"] == [{
        "file": "xdsl/parser.py", "n_hits": 1,
        "attribution": "byte_identical_granted_xdsl",
    }]


def test_xdsl_directory_name_does_not_exempt_modified_regex(
        tmp_path, monkeypatch: pytest.MonkeyPatch):
    upstream = "import re\nx = re.search('a', 'b')\n"
    digest = hashlib.sha256(upstream.encode()).hexdigest()
    monkeypatch.setattr(C, "_vendored_xdsl_hashes", lambda: frozenset({digest}))
    sub = _submission(tmp_path, {})
    (sub / "xdsl").mkdir()
    (sub / "xdsl/parser.py").write_text(upstream + "agent_change = True\n")
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn",
                  resolved_tools={"xdsl_kit"})
    assert v["checks"]["no_regex_ok"] is False
    assert any(hit["file"] == "xdsl/parser.py" for hit in v["regex_hits"])
    assert v["vendored_regex_files"] == []


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


def test_arm4_is_selected_by_resolved_tools_not_the_stale_arm_name(tmp_path):
    """The production Arm4 launcher spells its driver arm ``merlin_assisted``; the bundle tool set is
    the treatment and must still activate every RTL-specific requirement."""
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder spatial.dataflow gemmini"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    v = C.compute(_transcript(tmp_path / "t.jsonl", calls), sub, "merlin_assisted",
                  "inline_asm_insn", resolved_tools={
                      "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
                      "rtl_generators", "rtl_facts",
                  })
    assert v["arm4_tooling_required"] is True
    assert {
        "rtl_derived_levers_used", "rtl_facts_used", "scaffold_generators_used", "rtl_checks_read",
    } <= set(C.failing_checks(v))


def test_arm3_tool_set_does_not_gain_arm4_requirements_from_a_misleading_name(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder spatial.dataflow gemmini"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    v = C.compute(_transcript(tmp_path / "t.jsonl", calls), sub, "merlin_rtlchecks",
                  "inline_asm_insn", resolved_tools={
                      "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
                  })
    assert v["conformant"] is True
    assert v["arm4_tooling_required"] is False
    assert v["checks"]["rtl_derived_levers_used"] is None
    assert v["checks"]["rtl_checks_read"] is None


def test_fully_conformant_arm4_requires_successful_rtl_generator_and_readback_evidence(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder spatial.dataflow gemmini"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python -c \"from merlin.targetgen import rtl_backend as R; "
                              "print(R.derived_levers(R.target_profile('gemmini')))\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.rtl.facts import load_facts; "
                              "import json; print(json.dumps(load_facts('gemmini')['facts']))\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.generate import mlir_scaffold; "
                              "print(mlir_scaffold.generate(plan))\""}),
        ("bash", {"command": "jq '.rtl_checks' qa/verdict.json"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    tp = _transcript(tmp_path / "t.jsonl", calls, outputs={
        3: "['spatial.dataflow']",
        4: '{"target":"gemmini","arrays":[{"name":"mesh"}],"memories":[{"name":"spad"}]}',
        5: "[Artifact(relpath='dialect.py', content='...')]",
        6: '{"rtl_checks": {"verdict": "accept"}}',
    })
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
        "rtl_generators", "rtl_facts",
    })
    assert v["conformant"] is True
    assert v["checks"]["arm4_toolset_complete"] is True
    assert v["checks"]["rtl_derived_levers_used"] is True
    assert v["checks"]["rtl_facts_used"] is True
    assert v["checks"]["scaffold_generators_used"] is True
    assert v["checks"]["rtl_checks_read"] is True
    assert v["checks"]["arm4_discovery_before_submission_mutation"] is True


def test_arm4_required_commands_fail_closed_on_error_missing_result_and_stale_readback(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python -c \"import merlin.targetgen.rtl_backend as R; "
                              "print(R.derived_levers(R.target_profile('gemmini')))\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.rtl.facts import load_facts; "
                              "import json; print(json.dumps(load_facts('gemmini')['facts']))\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.generate import mlir_scaffold; "
                              "print(mlir_scaffold.generate(plan))\""}),
        ("bash", {"command": "cat qa/verdict.json"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    tp = _transcript(tmp_path / "t.jsonl", calls, failed={0, 2}, missing={1},
                     outputs={3: '{"all_pass": false}'})
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
        "rtl_generators", "rtl_facts",
    })
    assert v["checks"]["rtl_derived_levers_used"] is False
    assert v["checks"]["rtl_facts_used"] is False
    assert v["checks"]["scaffold_generators_used"] is False
    assert v["checks"]["rtl_checks_read"] is False
    assert v["conformant"] is False

    # ``jq`` exits zero when a key is absent, so a successful process with a null readback also fails.
    null_tp = _transcript(tmp_path / "null.jsonl", [
        ("bash", {"command": "jq '.rtl_checks' qa/verdict.json"}),
    ], outputs={0: "null"})
    null_v = C.compute(null_tp, sub, "merlin_assisted", "inline_asm_insn", resolved_tools={
        "rtl_generators", "rtl_facts",
    })
    assert null_v["checks"]["rtl_checks_read"] is False


def test_partial_rtl_tool_surface_cannot_silently_downgrade_to_arm3(tmp_path):
    sub = _submission(tmp_path, {})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn",
                  resolved_tools={"rtl_generators"})
    assert v["arm4_tooling_required"] is True
    assert v["checks"]["arm4_toolset_complete"] is False
    assert v["conformant"] is False


def test_arm4_echo_and_comment_spoofs_do_not_satisfy_semantic_tool_evidence(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder spatial.dataflow gemmini"}),
        # Each result has the right superficial shape, but the API name occurs only as inert text.
        ("bash", {"command": "true # R.derived_levers(profile)"}),
        ("bash", {"command": "python -c \"print('facts') # load_facts('gemmini')\""}),
        ("bash", {"command": "python -c \"print('artifact') # mlir_scaffold.generate(plan)\""}),
        ("bash", {"command": "echo '{\"rtl_checks\":[{\"verdict\":\"accept\"}]}' qa/verdict.json"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    tp = _transcript(tmp_path / "spoof.jsonl", calls, outputs={
        2: "['spatial.dataflow']",
        3: '{"target":"gemmini","arrays":[{}],"memories":[{}]}',
        4: "[Artifact(relpath='dialect.py', content='...')]",
        5: '{"rtl_checks":[{"verdict":"accept"}]}',
    })
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
        "rtl_generators", "rtl_facts",
    })
    assert v["checks"]["rtl_derived_levers_used"] is False
    assert v["checks"]["rtl_facts_used"] is False
    assert v["checks"]["scaffold_generators_used"] is False
    assert v["checks"]["rtl_checks_read"] is False
    assert v["conformant"] is False


def test_arm4_discovery_must_precede_first_submission_mutation(tmp_path):
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder spatial.dataflow gemmini"}),
        ("bash", {"command": "python -c \"from merlin.targetgen import rtl_backend as R; "
                              "print(R.derived_levers(R.target_profile('gemmini')))\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.rtl.facts import load_facts; "
                              "import json; print(json.dumps(load_facts('gemmini')['facts']))\""}),
        # The first four discovery calls are timely, but the required scaffold is too late.
        ("write_file", {"path": "submission/backend.py", "content": "import ast\n"}),
        ("bash", {"command": "python -c \"from merlin.targetgen.generate import mlir_scaffold; "
                              "print(mlir_scaffold.generate(plan))\""}),
        ("bash", {"command": "jq '.rtl_checks' qa/verdict.json"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    tp = _transcript(tmp_path / "late.jsonl", calls, outputs={
        2: "['spatial.dataflow']",
        3: '{"target":"gemmini","arrays":[{}],"interfaces":[{}]}',
        5: "[Artifact(relpath='dialect.py', content='...')]",
        6: '[{"capsule":"C0","verdict":"accept"}]',
    })
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
        "rtl_generators", "rtl_facts",
    })
    assert v["checks"]["rtl_derived_levers_used"] is True
    assert v["checks"]["rtl_facts_used"] is True
    assert v["checks"]["scaffold_generators_used"] is True
    assert v["checks"]["rtl_checks_read"] is True
    assert v["checks"]["arm4_discovery_before_submission_mutation"] is False
    assert v["conformant"] is False


def test_l3_fix_recompute_replaces_an_earlier_conformant_verdict(tmp_path):
    """Regression: the L3 loop used to retain the pre-L3 boolean while ignoring each fix transcript."""
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    tools = {
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
        "rtl_generators", "rtl_facts",
    }
    good_calls = [
        ("bash", {"command": "python cca_contract.py check-bijection gemmini"}),
        ("bash", {"command": "python action_catalog.py escalation-ladder spatial.dataflow gemmini"}),
        ("bash", {"command": "python isa_tools.py lint submission/kernel.mlir"}),
        ("bash", {"command": "python -c \"import merlin.targetgen.rtl_backend as R; "
                              "R.derived_levers(R.target_profile('gemmini'))\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.rtl.facts import load_facts; "
                              "load_facts('gemmini')\""}),
        ("bash", {"command": "python -c \"from merlin.targetgen.generate import mlir_scaffold; "
                              "mlir_scaffold.generate(plan)\""}),
        ("bash", {"command": "jq '.rtl_checks' qa/verdict.json"}),
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ]
    good_tp = _transcript(tmp_path / "good.jsonl", good_calls, outputs={
        3: "['spatial.dataflow']",
        4: '{"target":"gemmini","arrays":[{"name":"mesh"}],"interfaces":[{"name":"rocc"}]}',
        5: "[Artifact(relpath='dialect.py', content='...')]",
        6: '{"rtl_checks": {"verdict": "accept"}}',
    })
    _, was_conformant = LOOP._workflow_conformance(
        good_tp, sub, "merlin_assisted", "inline_asm_insn", tools)
    assert was_conformant is True

    # The final fix self-checks but omits all Arm4 authoring evidence.  Recomputing from this transcript
    # must return false; callers assign this result rather than OR-ing it with ``was_conformant``.
    fix_tp = _transcript(tmp_path / "fix.jsonl", [
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ])
    final, is_conformant = LOOP._workflow_conformance(
        fix_tp, sub, "merlin_assisted", "inline_asm_insn", tools)
    assert is_conformant is False
    assert set(C.failing_checks(final)) >= {
        "rtl_derived_levers_used", "rtl_facts_used", "scaffold_generators_used", "rtl_checks_read",
    }


def test_l3_fix_verdict_keeps_arm4_rtl_readback_available():
    vv = {
        "n_passed": 2, "n_capsules": 3,
        "per_capsule": [{"capsule": "C0", "l3_status": "fail"}],
        "rtl_checks": [{"capsule": "C0", "verdict": "reject"}],
    }
    verdict = LOOP._l3_fix_verdict(vv, 4)
    assert verdict["stage"] == "verilator_checkpoint"
    assert verdict["attempt"] == 4
    assert verdict["rtl_checks"] == vv["rtl_checks"]
    assert "answer-free rtl structural feedback" in verdict["rtl_checks_note"].lower()
