"""A conformance check may only require what the PROMPT actually mandates.

The dev-conformance flag exists to catch an agent that ignored its arm's tooling. That only means
something while the checks and the enforced-workflow prompt agree about what was mandated. They drifted:
the ISA-tool checks keyed on ``endpoint_kind == "external_backend"`` alone, while the prompt splits that
endpoint in two by the 4th artifact's form —

  * a WORD-STREAM target is told to assemble with ``isa_tools asm`` and check with ``lint``/``disasm``;
  * an MLIR/fork-free target is told to emit ``llvm.func`` and explicitly **"no prints, no `.insn`/`.word`"**.

So a target on the MLIR path was marked NOT CONFORMANT, every round, for not running a tool its own
instructions forbid. Measured on a live 3-repeat run: ``asm_used`` false in 13/13 rounds and
``isa_tools_used`` false in 10/13, on an agent that was following its prompt correctly.

These tests pin the two sides to ONE derived fact so they cannot drift again.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_HARNESS = repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"


def _load(name: str):
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    spec = importlib.util.spec_from_file_location(name, _HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("conformance")


#: (endpoint_kind, fourth_output_name) -> does the agent hand-author instruction words?
_CASES = [
    ("external_backend", "kernel.S", True),
    ("external_backend", "lowered.llvm.mlir", False),
    ("inline_asm_insn", "lowered.llvm.mlir", False),
    ("inline_asm_insn", "command_buffer.json", False),
    ("", "", False),
]


@pytest.mark.parametrize("endpoint,fourth,expected", _CASES)
def test_word_stream_is_derived_from_the_artifact_not_the_endpoint_alone(endpoint, fourth, expected):
    assert C.emits_word_stream(endpoint, fourth) is expected


def _transcript(tmp_path: Path, *commands: str) -> Path:
    """A minimal driver transcript whose assistant turns ran *commands* through a bash tool_use."""
    lines = []
    for cmd in commands:
        lines.append(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "bash", "input": {"command": cmd}}]}}))
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    return p


#: The two mandates that apply to an assisted arm REGARDLESS of the artifact form, so a test about the
#: ISA tools can hold them satisfied and isolate the thing it is actually asserting.
_ALWAYS_MANDATED = ("python cca_contract.py check-bijection t",
                    "python3 agent_selfcheck.py --submission submission --capsules all")


def test_the_mlir_path_makes_both_isa_tool_checks_not_applicable(tmp_path):
    """The check that regressed: an MLIR-path target must report N/A, never False, for the ISA tools."""
    sub = tmp_path / "submission"
    sub.mkdir()
    tpath = _transcript(tmp_path, *_ALWAYS_MANDATED)   # never touches isa_tools
    v = C.compute(tpath, sub, "merlin_rtlchecks", "external_backend", "lowered.llvm.mlir")
    assert v["checks"]["isa_tools_used"] is None, v["checks"]
    assert v["checks"]["asm_used"] is None, v["checks"]
    assert v["emits_word_stream"] is False
    # With the mandates that DO apply satisfied, never calling the ISA tools is fully conformant here.
    assert v["conformant"] is True, v


def test_the_isa_tools_are_not_what_makes_an_mlir_round_non_conformant(tmp_path):
    """Guard the inverse: an MLIR-path round that skips a mandate which DOES apply still fails, so this
    fix cannot be mistaken for switching conformance off on that path."""
    sub = tmp_path / "submission"
    sub.mkdir()
    tpath = _transcript(tmp_path)                      # nothing ran at all
    v = C.compute(tpath, sub, "merlin_rtlchecks", "external_backend", "lowered.llvm.mlir")
    assert v["conformant"] is False
    assert v["checks"]["cca_used"] is False and v["checks"]["full_selfcheck"] is False


def test_a_word_stream_target_still_requires_the_isa_tools(tmp_path):
    """The fix must not silently disable the check everywhere — a self-hosted kernel still owes them."""
    sub = tmp_path / "submission"
    sub.mkdir()
    tpath = tmp_path / "transcript.jsonl"
    tpath.write_text("")
    v = C.compute(tpath, sub, "merlin_rtlchecks", "external_backend", "kernel.S")
    assert v["checks"]["isa_tools_used"] is False, v["checks"]
    assert v["checks"]["asm_used"] is False, v["checks"]
    assert v["conformant"] is False


def test_the_prompt_and_the_check_gate_on_the_same_fact():
    """The anti-drift invariant: the prompt mandates `isa_tools asm` on exactly the targets the checker
    requires it on. Both sides are asked here, so changing one without the other fails this test."""
    from merlin.targetgen import generate_prompt as GP

    class _Manifest:
        def __init__(self, endpoint, fourth):
            self.endpoint_kind = endpoint
            self.fourth_output_name = fourth

    for endpoint, fourth, expect_words in _CASES:
        prompt_mandates_asm = not GP._is_simt_mlir(_Manifest(endpoint, fourth)) and \
            endpoint == "external_backend"
        checker_requires_asm = C.emits_word_stream(endpoint, fourth)
        assert prompt_mandates_asm == checker_requires_asm, (
            f"drift for endpoint={endpoint!r} fourth={fourth!r}: "
            f"prompt mandates asm={prompt_mandates_asm}, checker requires it={checker_requires_asm}")
        assert checker_requires_asm is expect_words
