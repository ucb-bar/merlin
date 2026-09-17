"""Dev-conformance verdict (capsule-bench harness): the completion gate that catches a run scoring 0 by NOT using
the mandated tooling (invented encoding / regex parser / no lever enumeration / partial self-check),
which the numeric grade alone cannot distinguish from a real capability wall. Hermetic — synthetic
transcript + submission, no live run, no target.
"""
from __future__ import annotations

import json
import hashlib
import shlex
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
         "content": outputs.get(i, _default_output(calls[i], i in failed)),
         "is_error": i in failed}
        for i in range(len(calls)) if i not in missing]}}
    path.write_text("\n".join((
        json.dumps({"type": "system", "subtype": "init", "driver": "codex"}),
        json.dumps(ev), json.dumps(results), "")))
    return path


# What agent_selfcheck.py actually prints.  The conformance check reads the REPORT rather than the exit
# status, because the script exits non-zero whenever a capsule is still failing -- gating on success made
# "did you run the self-check?" a restatement of the score.
_SELFCHECK_REPORT = '{"n_capsules": 20, "n_passed": 3, "all_pass": false, "per_capsule": []}'


def _default_output(call: tuple[str, dict], failed: bool) -> str:
    command = str(call[1].get("command") or call[1].get("cmd") or "")
    if "agent_selfcheck" in command:
        return _SELFCHECK_REPORT
    return "ok" if not failed else "failed"


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


def test_full_selfcheck_counts_complete_nonzero_grade_response(tmp_path):
    """The CLI exits nonzero for a real all-capsule run with screened/failed tiers; it still ran all."""
    sub = _submission(tmp_path, {})
    calls = [("bash", {"command":
              "python3 agent_selfcheck.py --submission submission --sim spike --capsules all"})]
    complete = json.dumps({
        "sim": "spike", "n_capsules": 2, "n_passed": 1, "all_pass": False,
        "per_capsule": [
            {"capsule": "C0", "pass": True},
            {"capsule": "C1", "pass": False},
        ],
    })
    tp = _transcript(tmp_path / "nonzero.jsonl", calls, failed={0}, outputs={0: complete})

    verdict = C.compute(tp, sub, "raw_baseline", "inline_asm_insn", resolved_tools=set())

    assert verdict["checks"]["full_selfcheck"] is True
    assert verdict["conformant"] is True


def test_full_selfcheck_does_not_credit_nonzero_error_or_partial_response(tmp_path):
    sub = _submission(tmp_path, {})
    calls = [("bash", {"command": "python3 agent_selfcheck.py --capsules all"})]
    for name, output in (
        ("error", '{"error":"broker timed out"}'),
        ("partial", '{"n_capsules":2,"per_capsule":[{"capsule":"C0"}]}'),
    ):
        tp = _transcript(tmp_path / f"{name}.jsonl", calls, failed={0}, outputs={0: output})
        verdict = C.compute(tp, sub, "raw_baseline", "inline_asm_insn", resolved_tools=set())
        assert verdict["checks"]["full_selfcheck"] is False
        assert verdict["conformant"] is False


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


def test_native_codex_shell_wrapper_preserves_arm4_semantic_evidence(tmp_path):
    """Codex persists Bash calls with a transport wrapper; conformance must inspect its one inner command."""
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})

    def codex(command: str) -> str:
        return f"/bin/bash -lc {shlex.quote(command)}"

    calls = [
        ("Bash", {"command": codex("python cca_contract.py check-bijection gemmini")}),
        ("Bash", {"command": codex(
            "python action_catalog.py escalation-ladder spatial.dataflow gemmini")}),
        ("Bash", {"command": codex("python isa_tools.py lint submission/kernel.mlir")}),
        ("Bash", {"command": codex(
            "python -c \"from merlin.targetgen import rtl_backend as R; "
            "print(R.derived_levers(R.target_profile('gemmini')))\"")}),
        ("Bash", {"command": codex(
            "python -c \"from merlin.targetgen.rtl.facts import load_facts; import json; "
            "print(json.dumps(load_facts('gemmini')['facts']))\"")}),
        ("Bash", {"command": codex(
            "python -c \"from merlin.targetgen.generate import target_repo; "
            "print([x.relpath for x in target_repo.generate_skeleton('gemmini')])\"")}),
        ("Bash", {"command": codex("jq '.rtl_checks' qa/verdict.json")}),
        ("Bash", {"command": codex("python3 agent_selfcheck.py --capsules all")}),
    ]
    tp = _transcript(tmp_path / "codex.jsonl", calls, outputs={
        3: "['spatial.dataflow']",
        4: '{"target":"gemmini","arrays":[{}],"memories":[{}]}',
        5: "['README.md', 'dialect.py', 'CMakeLists.txt']",
        6: '[{"capsule":"C0","verdict":"ok"}]',
    })
    v = C.compute(tp, sub, "merlin_assisted", "inline_asm_insn", resolved_tools={
        "merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools",
        "rtl_generators", "rtl_facts",
    })

    assert v["conformant"] is True
    assert v["checks"]["rtl_derived_levers_used"] is True
    assert v["checks"]["rtl_facts_used"] is True
    assert v["checks"]["scaffold_generators_used"] is True
    assert v["checks"]["rtl_checks_read"] is True
    assert v["checks"]["arm4_discovery_before_submission_mutation"] is True


def test_native_codex_shell_wrapper_does_not_enable_spoof_composition_or_null_readback(tmp_path):
    sub = _submission(tmp_path, {})

    def codex(command: str) -> str:
        return f"/bin/bash -lc {shlex.quote(command)}"

    spoofed = _transcript(tmp_path / "spoofed.jsonl", [
        ("Bash", {"command": codex("true # R.derived_levers(profile)")}),
        ("Bash", {"command": codex("python -c \"print(['spatial.dataflow'])\"; true")}),
        ("Bash", {"command": codex("echo load_facts gemmini")}),
        ("Bash", {"command": codex("echo generate_skeleton")}),
        ("Bash", {"command": codex("echo '{\"rtl_checks\":[{}]}' qa/verdict.json")}),
    ], outputs={
        0: "['spatial.dataflow']",
        1: "['spatial.dataflow']",
        2: '{"target":"gemmini","arrays":[{}],"memories":[{}]}',
        3: "['dialect.py']",
        4: '{"rtl_checks":[{}]}',
    })
    tools = {"rtl_generators", "rtl_facts"}
    v = C.compute(spoofed, sub, "merlin_assisted", "inline_asm_insn", resolved_tools=tools)
    assert v["checks"]["rtl_derived_levers_used"] is False
    assert v["checks"]["rtl_facts_used"] is False
    assert v["checks"]["scaffold_generators_used"] is False
    assert v["checks"]["rtl_checks_read"] is False

    null_read = _transcript(tmp_path / "null.jsonl", [
        ("Bash", {"command": codex("jq '.rtl_checks' qa/verdict.json")}),
    ], outputs={0: "null"})
    null_v = C.compute(
        null_read, sub, "merlin_assisted", "inline_asm_insn", resolved_tools=tools)
    assert null_v["checks"]["rtl_checks_read"] is False


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


def test_a_target_is_resolved_by_its_declared_name_not_its_directory():
    """A descriptor sits in a directory that need not match the target its contract is registered
    under -- one target declares a configuration-qualified name beside a short directory. Resolving by
    the directory finds no contract at all, which is why that target had no derived requirement: not
    because none could be derived, but because nobody was asking about the right name.

    Two call sites resolved this independently and only one was right, so `audit` produced a
    requirement while `--write` -- the path that actually creates the file -- crashed."""
    import sys

    from merlin.common.paths import repo_root

    sys.path.insert(0, str(repo_root() / "build_tools" / "scripts"))
    from check_conformance_coverage import _contract_target

    roots = repo_root() / "merlin/experiments/capsule_bench/targets"
    mismatched = []
    for d in sorted(p for p in roots.iterdir() if (p / "target_experiment.yaml").is_file()):
        resolved = _contract_target(d.name)
        assert resolved, d.name
        if resolved != d.name:
            mismatched.append((d.name, resolved))
    # Not an assertion about WHICH targets differ -- that is a naming choice and may change. The point
    # is that the resolver reads the descriptor rather than assuming the directory, so a mismatch
    # resolves instead of raising.
    assert all(r for _, r in mismatched)


# --- What the harness must MEASURE, not the shape it hoped the agent would type -------------------
# Every check below failed on the atlas arm-4 run of 2026-09-04 while the agent had in fact done the
# work.  The run scored 3/57 and reported NOT CONFORMANT on five checks; four of the five were the
# harness failing to see evidence that was present in its own transcript.

def test_a_composed_command_still_yields_its_python_evidence(tmp_path):
    """`bash -lc "A; python - <<'PY' ... PY"` is how agents actually work.

    The extractor accepted only a bare, uncomposed `python -c` or a lone heredoc, so a wrapper or a
    `;` made the RTL discovery invisible and five arm-4 checks reported False."""
    sub = _submission(tmp_path, {"backend.py": "import ast\n"})
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "/bin/bash -lc \"python cca_contract.py check-bijection t; python - <<'PY'\n"
                             "from rtl import facts\nx = facts.load_facts('t')\nprint(x)\nPY\""}),
    ], outputs={0: "{'target': 't', 'arrays': [{'name': 'mesh'}]}"})
    calls = C._tool_calls(tp)
    assert "facts.load_facts" in C._python_call_names(calls[0])
    assert C._rtl_facts_evidence(calls[0]) is True


def test_echoed_api_text_is_still_not_evidence(tmp_path):
    """Decomposing composition must not make forgery pass: echo is not a Python invocation."""
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "/bin/bash -lc \"echo 'facts.load_facts(1)'; true # derived_levers(p)\""}),
    ], outputs={0: "{'target': 't', 'arrays': []}"})
    calls = C._tool_calls(tp)
    assert C._python_call_names(calls[0]) == ()
    assert C._rtl_facts_evidence(calls[0]) is False


def test_facts_evidence_does_not_depend_on_print_style(tmp_path):
    """`print(facts)` yields a Python repr; `jq` yields JSON.  Both are the same evidence."""
    for body in ('{"target": "t", "arrays": [1]}', "{'target': 't', 'arrays': [1]}"):
        tp = _transcript(tmp_path / f"t{hash(body)}.jsonl", [
            ("bash", {"command": "python -c \"from rtl import facts; print(facts.load_facts('t'))\""}),
        ], outputs={0: body})
        assert C._rtl_facts_evidence(C._tool_calls(tp)[0]) is True, body


def test_running_the_selfcheck_is_not_the_same_as_passing_it(tmp_path):
    """agent_selfcheck exits non-zero while capsules fail, so requiring rc==0 encoded the score."""
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "python3 agent_selfcheck.py --capsules all"}),
    ], failed={0})
    assert C._full_selfcheck(C._tool_calls(tp)) is True


def test_a_selfcheck_that_produced_no_report_is_not_evidence(tmp_path):
    """Dropping the exit-status requirement must not credit a bare mention of the command."""
    tp = _transcript(tmp_path / "t.jsonl", [
        ("bash", {"command": "echo python3 agent_selfcheck.py --capsules all"}),
    ], outputs={0: "python3 agent_selfcheck.py --capsules all"})
    assert C._full_selfcheck(C._tool_calls(tp)) is False
