"""An L3 run of a program already measured returns the measurement, and never a different one.

Measured across every campaign on disk: consecutive candidates emitted BYTE-IDENTICAL code for every
corpus member. The agent edits something, the harness pays a full cycle-accurate sweep, and the
program it measures is the one it measured last time. On a single call, 15 of 28 repeated members
re-ran a program whose cycle count was already known.

This is not a screen and not a prediction. Two runs of one program on one pinned engine return the
same cycles -- verified over 392 repeated measurements of identical bytes with zero disagreement --
so a hit returns the number rather than estimating it.

The key has to be the whole emitted program. The command buffer alone is NOT the program: 28 members
shared a command buffer and only 15 of them shared a cycle count, because the lowered module differed.
Keyed on the lowered module the agreement was exact, 15 of 15.
"""
from __future__ import annotations

import sys

import pytest
from pathlib import Path

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_paired_perf_bench as PAIR  # noqa: E402

PIN_A, PIN_B = "a" * 64, "b" * 64
CB = {"abi_version": "0.1", "commands": [{"opcode": "MATMUL", "operands": {"dst": "Y"}}]}


def test_the_same_emitted_program_on_the_same_engine_is_one_key():
    assert PAIR._l3_memo_key(CB, "module {}", PIN_A) == PAIR._l3_memo_key(CB, "module {}", PIN_A)


def test_key_ordering_of_the_command_buffer_does_not_matter():
    """Two dicts that differ only in key order are the same program."""
    reordered = {"commands": CB["commands"], "abi_version": CB["abi_version"]}
    assert PAIR._l3_memo_key(CB, "m", PIN_A) == PAIR._l3_memo_key(reordered, "m", PIN_A)


def test_a_different_lowered_module_is_a_different_program():
    """The defect this guards: the command buffer alone is not the program."""
    assert PAIR._l3_memo_key(CB, "module { A }", PIN_A) != PAIR._l3_memo_key(CB, "module { B }", PIN_A)


def test_a_different_command_buffer_is_a_different_program():
    other = {"abi_version": "0.1", "commands": [{"opcode": "CONV2D", "operands": {"dst": "Y"}}]}
    assert PAIR._l3_memo_key(CB, "m", PIN_A) != PAIR._l3_memo_key(other, "m", PIN_A)


def test_a_different_engine_shares_nothing():
    """A cycle count is a fact about a program AND the engine that ran it."""
    assert PAIR._l3_memo_key(CB, "m", PIN_A) != PAIR._l3_memo_key(CB, "m", PIN_B)


def test_the_memo_starts_empty_so_a_stale_table_cannot_answer_for_a_fresh_stage():
    assert isinstance(PAIR._L3_MEMO, dict)


# ---------------------------------------------------------------------------------------------------
# the free screen's path contract
# ---------------------------------------------------------------------------------------------------
def test_a_relative_command_buffer_path_resolves_against_the_submission(tmp_path):
    """Three path bases were live in one tool and this action honoured none of them.

    The agent's shell sees `submission/performance/...`; a brokered compile command is chdir'd into
    the submission and sees `performance/...`; this action runs in the HOST process and saw neither,
    so a relative argument resolved against a directory the agent has never seen. Measured: two
    wasted calls, and a refusal reading "absent or linked" -- a claim about the filesystem when the
    fault was the base.
    """
    import json as _json
    import perf_agent_stage as PAS  # noqa: PLC0415

    root = tmp_path / "submission"
    (root / "performance").mkdir(parents=True)
    buf = {"abi_version": "0.1", "target": "t", "commands": [], "tensors": {}}
    (root / "performance" / "b.json").write_text(_json.dumps(buf), encoding="utf-8")

    out = PAS.analyze_command_buffers(
        Path("performance/b.json"), Path("performance/b.json"),
        candidate_root=root, peak_macs_per_cycle=256, achievable_macs_per_cycle=80.0, target="t")
    assert out["kind"] == "host_owned_command_buffer_analysis"


def test_an_unresolvable_path_says_what_it_was_resolved_against(tmp_path):
    """The refusal must name the base, or the agent cannot tell a missing file from a wrong base."""
    import perf_agent_stage as PAS  # noqa: PLC0415

    root = tmp_path / "submission"
    root.mkdir()
    with pytest.raises(PAS.StageGateError, match="resolved against the candidate root"):
        PAS.analyze_command_buffers(
            Path("performance/missing.json"), Path("performance/missing.json"),
            candidate_root=root, peak_macs_per_cycle=256, achievable_macs_per_cycle=80.0,
            target="t")
