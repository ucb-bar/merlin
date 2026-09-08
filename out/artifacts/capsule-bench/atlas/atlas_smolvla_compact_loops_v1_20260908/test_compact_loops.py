"""Focused compile/code-size regressions for the Atlas SmolVLA recovery.

These tests establish emission, bounded IMEM size, runtime batch reuse, and valid
decoded control-flow targets.  They deliberately do not claim simulator or numeric
correctness.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "submission/mlir_oot/atlas-opt"
INTERFACES = ROOT / "interfaces_before_reference"
IMEM_WORDS = 32768


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _branch_immediate(word: int) -> int:
    value = (
        ((word >> 31) & 1) << 12
        | ((word >> 7) & 1) << 11
        | ((word >> 25) & 0x3F) << 5
        | ((word >> 8) & 0xF) << 1
    )
    return _signed(value, 13)


def _jump_immediate(word: int) -> int:
    value = (
        ((word >> 31) & 1) << 20
        | ((word >> 12) & 0xFF) << 12
        | ((word >> 20) & 1) << 11
        | ((word >> 21) & 0x3FF) << 1
    )
    return _signed(value, 21)


def _compile(path: Path) -> list[int]:
    result = subprocess.run(
        [str(TOOL), "--emit-target-artifact", str(path)],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return [
        int(line.split()[1], 16)
        for line in result.stdout.splitlines()
        if line.lstrip().startswith(".word")
    ]


def _assert_control_targets(words: list[int]) -> int:
    backward_edges = 0
    for index, word in enumerate(words):
        opcode = word & 0x7F
        if opcode == 0x63:
            immediate = _branch_immediate(word)
        elif opcode == 0x6F:
            immediate = _jump_immediate(word)
        else:
            continue
        assert immediate % 2 == 0
        target = index + immediate // 2
        assert 0 <= target < len(words), (index, immediate, target, len(words))
        backward_edges += target < index
    return backward_edges


def test_rank2_k_and_n_tails_fit_and_have_valid_runtime_loops() -> None:
    # K=720 has a 16-element tail; this shape was the 2,774,543-word maximum.
    words = _compile(INTERFACES / "matmul_50_720_2048.mlir")
    assert len(words) == 12605
    assert len(words) <= IMEM_WORDS
    assert _assert_control_targets(words) >= 3


def test_partial_n_tail_and_batch_count_reuse_one_body() -> None:
    # N=113 has a 17-element tail and B=15.  The old emitter produced 443,625
    # words by repeating the batch/tile bodies.
    source = INTERFACES / "matmul_batched_15_50_64_113.mlir"
    batch15 = _compile(source)
    assert len(batch15) == 32458
    assert len(batch15) <= IMEM_WORDS
    assert _assert_control_targets(batch15) >= 4

    # Only the immediate batch trip count may change.  Code size must remain
    # invariant, proving that the body is emitted once rather than 15 times.
    batch1_text = source.read_text(encoding="utf-8").replace("15x", "1x").replace(
        "batch = 15 : i64", "batch = 1 : i64"
    )
    with tempfile.TemporaryDirectory() as directory:
        batch1_path = Path(directory) / "batch1.mlir"
        batch1_path.write_text(batch1_text, encoding="utf-8")
        batch1 = _compile(batch1_path)
    assert len(batch1) == len(batch15)


if __name__ == "__main__":
    test_rank2_k_and_n_tails_fit_and_have_valid_runtime_loops()
    test_partial_n_tail_and_batch_count_reuse_one_body()
    print("ok: compact K/N tails, runtime batch reuse, IMEM bounds, control targets")
