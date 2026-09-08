"""Focused compile/code-size regressions for the Atlas SmolVLA recovery.

These tests establish emission, bounded IMEM size, runtime batch reuse, and valid
decoded control-flow targets.  They deliberately do not claim simulator or numeric
correctness.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "submission/mlir_oot/atlas-opt"
INTERFACES = ROOT / "interfaces_before_reference"
IMEM_WORDS = 32768
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot import encoder as e  # noqa: E402


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


def test_compact_epilogues_fail_closed_instead_of_emitting_ecall_only() -> None:
    source = (ROOT / "cases/smolvla_state_proj_1_32_960/two_matmuls.mlir").read_text()
    source = source.replace('["bias_add"]', '["bias_add", "relu"]')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "unsupported_combo.mlir"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [str(TOOL), "--emit-target-artifact", str(path)],
            text=True, capture_output=True, timeout=30,
        )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "unsupported Atlas emission for matmul" in result.stderr


def test_static_bias_precedes_relu_in_the_encoded_epilogue() -> None:
    source = '''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x8xf8E4M3FN>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<1x32xf8E4M3FN>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<8xbf16>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x8xf8E4M3FN>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<1x32xf8E4M3FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["bias_add", "relu"], output_dtype = "bf16", bias = "B"} : (!merlin_iface.acc<bf16>) -> tensor<1x8xbf16>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
'''
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "static_combo.mlir"
        path.write_text(source, encoding="utf-8")
        words = _compile(path)
    assert words.index(e.vadd(2, 2, 6)) < words.index(e.vrelu(2, 2))


def test_emitted_pair_writers_use_even_banks_and_validator_can_fail() -> None:
    words = _compile(ROOT / "cases/smolvla_state_proj_1_32_960/two_matmuls.mlir")
    e.validate_pair_banks(words)
    assert e.vli_all(62, 0) in words
    assert e.vli_all(63, 0) not in words
    # Odd VPU pair destinations fail, while odd single-bank VLOAD is legal.
    e.validate_pair_banks([e.vload(7, 6)])
    try:
        e.validate_pair_banks([e.vli_all(63, 0)])
    except ValueError as error:
        assert "pair-write" in str(error) and "odd MREG bank 63" in str(error)
    else:
        raise AssertionError("odd VLI_ALL pair destination was accepted")
    for bad_word, expected_role in (
        (e.vrelu(2, 3), "pair-read-primary"),
        (e.vadd(2, 2, 3), "pair-read-secondary"),
        (e.pop_bf16(63, 0, mxu=1), "pair-write"),
    ):
        try:
            e.validate_pair_banks([bad_word])
        except ValueError as error:
            assert expected_role in str(error)
        else:
            raise AssertionError(f"odd {expected_role} base was accepted")


if __name__ == "__main__":
    test_rank2_k_and_n_tails_fit_and_have_valid_runtime_loops()
    test_partial_n_tail_and_batch_count_reuse_one_body()
    test_compact_epilogues_fail_closed_instead_of_emitting_ecall_only()
    test_static_bias_precedes_relu_in_the_encoded_epilogue()
    test_emitted_pair_writers_use_even_banks_and_validator_can_fail()
    print("ok: compact K/N tails, runtime batch reuse, IMEM bounds, control targets")
