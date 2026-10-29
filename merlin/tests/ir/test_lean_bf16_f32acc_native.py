"""Numerical behavior of ``lower_bf16_matmul_f32acc`` output under the compiler's own MLIR JIT.

Executed bits are compared with independent NumPy references of the f32-accumulation contract
(``lean_bf16_f32acc_cases``). Inputs are chosen so each clause is observable: reordering,
initialization and contraction each change at least one bit, and 16-bit accumulation differs
from the contract. The retained SmolVLA boundary's own rewritten operations are executed at
their full captured shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from lean_bf16_f32acc_cases import (
    bmm,
    contract_reference,
    contract_reference_fast,
    from_f32,
    half_accumulation_reference,
    line_with,
    matmul,
    random_bits,
    replace_once,
    reverse_reduction,
    run_pass,
)

from merlin.common.paths import repo_root
from merlin.llvmlower import toolchain

RETAINED = (
    "out/artifacts/verification/merlin-passes/w8a8/model-boundaries/20261020T215617Z/SY_model_smolvla/"
    "lean-preparation/3d2dd40bbd9740f1868470172e4d0aa5/stages/0008_prepare-bf16-matmul-f32acc"
)


@pytest.fixture(scope="module")
def compiler():
    selected = os.environ.get("MERLIN_M2M_PYTHON")
    python = Path(selected) if selected else toolchain.compiler_python()
    if not python.exists():
        reason = f"compiler interpreter unavailable: {python}"
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return python


def execute(
    compiler: Path, work: Path, text: str, inputs: list[tuple[np.ndarray, str]], shape, fmt: str, opt_level: int = 0
) -> np.ndarray:
    work.mkdir(parents=True, exist_ok=True)
    source = work / "subject.mlir"
    source.write_text(text)
    specs = []
    for index, (bits, element) in enumerate(inputs):
        np.save(work / f"in{index}.npy", np.asarray(bits, dtype=np.uint16))
        specs.append({"shape": list(np.shape(bits)), "format": element, "path": str(work / f"in{index}.npy")})
    request = {
        "source": str(source),
        "function": "subject",
        "inputs": specs,
        "output": {"shape": list(shape), "format": fmt, "path": str(work / "out.npy")},
        "opt_level": opt_level,
        "lowered": str(work / "lowered.mlir"),
    }
    (work / "request.json").write_text(json.dumps(request))
    oracle = repo_root() / "merlin/tests/ir/lean_bf16_native_oracle.py"
    proc = subprocess.run(
        [str(compiler), str(oracle), str(work / "request.json"), str(repo_root() / "src/merlin/verify/lean")],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    answer = json.loads(proc.stdout.strip().splitlines()[-1])
    assert answer["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert answer["opt_level"] == opt_level
    return np.load(work / "out.npy")


@pytest.mark.parametrize("opt_level", [0, 3])
@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_actual_output_executes_the_contract(compiler, tmp_path, fmt, opt_level):
    rng = np.random.default_rng(20261021)
    source = bmm(fmt, depth=37)
    output, count = run_pass(source)
    assert count == 1
    for sample in range(3):
        left, right = random_bits(rng, (2, 3, 37), fmt), random_bits(rng, (2, 37, 4), fmt)
        expected = contract_reference(left, right, fmt)
        # Every product of extended 16-bit operands here is exact in f32, so the contraction
        # license cannot change these bits; the comparison is unambiguous.
        assert np.array_equal(expected, contract_reference(left, right, fmt, fused=True))
        got = execute(compiler, tmp_path / f"s{sample}", output, [(left, fmt), (right, fmt)], (2, 3, 4), fmt, opt_level)
        assert np.array_equal(got, expected), (fmt, opt_level, sample)


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_source_accumulates_in_sixteen_bits_and_differs(compiler, tmp_path, fmt):
    rng = np.random.default_rng(7)
    source = bmm(fmt, depth=64)
    output, _ = run_pass(source)
    left, right = random_bits(rng, (2, 3, 64), fmt), random_bits(rng, (2, 64, 4), fmt)
    before = execute(compiler, tmp_path / "before", source, [(left, fmt), (right, fmt)], (2, 3, 4), fmt)
    after = execute(compiler, tmp_path / "after", output, [(left, fmt), (right, fmt)], (2, 3, 4), fmt)
    assert np.array_equal(before, half_accumulation_reference(left, right, fmt))
    assert np.array_equal(after, contract_reference(left, right, fmt))
    assert np.count_nonzero(before != after) > 0


def _ordered_case():
    """Products 2^24, 1, 1, -2^24, 0: forward 2^24 absorbs both 1s (ties to even), reverse keeps 2."""
    left = np.zeros((2, 3, 5), dtype=np.float32)
    right = np.zeros((2, 5, 4), dtype=np.float32)
    left[0, 0, :4] = [4096, 1, 1, -4096]
    right[0, :4, 0] = [4096, 1, 1, 4096]
    return from_f32(left, "bf16"), from_f32(right, "bf16")


def test_reduction_order_is_observable(compiler, tmp_path):
    left, right = _ordered_case()
    output, _ = run_pass(bmm())
    forward = contract_reference(left, right, "bf16")
    backward = contract_reference(left, right, "bf16", order=reversed(range(5)))
    assert forward[0, 0, 0] != backward[0, 0, 0]
    got = execute(compiler, tmp_path / "actual", output, [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16")
    reordered = execute(
        compiler, tmp_path / "reversed", reverse_reduction(output), [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16"
    )
    assert np.array_equal(got, forward) and np.array_equal(reordered, backward)


def test_initialization_is_observable(compiler, tmp_path):
    rng = np.random.default_rng(11)
    left, right = random_bits(rng, (2, 3, 5), "bf16"), random_bits(rng, (2, 5, 4), "bf16")
    output, _ = run_pass(bmm())
    one = replace_once(output, "<{value = 0.000000e+00 : f32}>", "<{value = 1.000000e+00 : f32}>")
    got = execute(compiler, tmp_path, one, [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16")
    assert np.array_equal(got, contract_reference(left, right, "bf16", init=1.0))
    assert not np.array_equal(got, contract_reference(left, right, "bf16"))


def _license_case():
    """Products -2^127 then 2^128: unfused overflows to inf; one fused rounding gives 2^127."""
    left = np.zeros((2, 3, 2), dtype=np.float32)
    right = np.zeros((2, 2, 4), dtype=np.float32)
    left[0, 0, :] = [-(2.0**63), 2.0**64]
    right[0, :, 0] = [2.0**64, 2.0**64]
    return from_f32(left, "bf16"), from_f32(right, "bf16")


@pytest.mark.parametrize("opt_level", [0, 3])
def test_contraction_license_admits_exactly_fused_or_unfused(compiler, tmp_path, opt_level):
    left, right = _license_case()
    unfused = contract_reference(left, right, "bf16")
    fused = contract_reference(left, right, "bf16", fused=True)
    assert unfused[0, 0, 0] == 0x7F80 and fused[0, 0, 0] == 0x7F00  # +inf versus 2^127
    output, _ = run_pass(bmm(depth=2))
    got = execute(
        compiler, tmp_path / "contract", output, [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16", opt_level
    )
    assert np.array_equal(got, unfused) or np.array_equal(got, fused)
    unlicensed = output.replace("fastmath<contract>", "fastmath<none>")
    strict = execute(
        compiler, tmp_path / "none", unlicensed, [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16", opt_level
    )
    assert np.array_equal(strict, unfused)


def test_f16_accumulation_overflow_is_removed(compiler, tmp_path):
    left = np.zeros((2, 3, 4), dtype=np.float32)
    right = np.zeros((2, 4, 4), dtype=np.float32)
    left[0, 0, :] = [200, 200, -200, -200]
    right[0, :, 0] = [200, 200, 200, 200]
    left, right = from_f32(left, "f16"), from_f32(right, "f16")
    source = bmm("f16", depth=4)
    before = execute(compiler, tmp_path / "before", source, [(left, "f16"), (right, "f16")], (2, 3, 4), "f16")
    after = execute(
        compiler, tmp_path / "after", run_pass(source)[0], [(left, "f16"), (right, "f16")], (2, 3, 4), "f16"
    )
    assert before[0, 0, 0] == 0x7C00  # 40000 + 40000 overflows f16 before the subtractions
    assert after[0, 0, 0] == 0 and np.array_equal(after, contract_reference(left, right, "f16"))


def test_nonzero_source_is_refused_and_zero_init_fault_remains_observable(compiler, tmp_path):
    source = bmm(init="1.000000e+00")
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        run_pass(source)
    left, right = np.zeros((2, 3, 5), dtype=np.uint16), np.zeros((2, 5, 4), dtype=np.uint16)
    original = execute(compiler, tmp_path / "source", source, [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16")
    wrong, _ = run_pass(bmm())
    got = execute(compiler, tmp_path / "old-fault", wrong, [(left, "bf16"), (right, "bf16")], (2, 3, 4), "bf16")
    assert np.array_equal(got, contract_reference(left, right, "bf16"))
    assert not np.array_equal(got, original)
    assert np.array_equal(original, contract_reference(left, right, "bf16", init=1.0))


def test_pass_executes_original_transposed_maps_and_old_fault_differs(compiler, tmp_path):
    rng = np.random.default_rng(5)
    left, right = random_bits(rng, (4, 4), "bf16"), random_bits(rng, (4, 4), "bf16")
    source = matmul(m=4, n=4, depth=4, transposed=True)
    output, _ = run_pass(source)
    got = execute(compiler, tmp_path / "correct", output, [(left, "bf16"), (right, "bf16")], (4, 4), "bf16")
    untransposed = contract_reference(left[None], right[None], "bf16")[0]
    intended = contract_reference(left[None], right.T[None], "bf16")[0]
    assert not np.array_equal(untransposed, intended)
    assert np.array_equal(got, intended)
    wrong = replace_once(output, "-> (d1, d2)>", "-> (d2, d1)>")
    old = execute(compiler, tmp_path / "old-fault", wrong, [(left, "bf16"), (right, "bf16")], (4, 4), "bf16")
    assert np.array_equal(old, untransposed)


def extract_site(text: str, ordinal: int) -> tuple[str, list[str], str]:
    """The six operations that replaced one contraction, as a function of its two operands."""
    lines = text.splitlines()
    heads = [
        i
        for i, line in enumerate(lines)
        if '"linalg.generic"' in line and "iterator_type<reduction>" in line and '"arith.extf"' in lines[i + 2]
    ]
    head = heads[ordinal]
    start, stop = head - 6, head + 14
    block = lines[start:stop]
    assert '"tensor.empty"' in block[0] and '"linalg.fill"' in block[2] and '"arith.truncf"' in block[17]
    left, right, _ = [part.strip() for part in block[6].split('"(', 1)[1].split(")", 1)[0].split(",")]
    types = block[13].rsplit(" : (", 1)[1].split(")", 1)[0].split(", ")[:2]
    out_type = block[19].rsplit("-> ", 1)[1].strip()
    result = block[15].split("=", 1)[0].strip()
    body = "\n".join(block)
    module = (
        f'"builtin.module"() ({{\n  "func.func"() <{{function_type = ({types[0]}, {types[1]}) -> {out_type}, '
        f'sym_name = "subject"}}> ({{\n  ^bb0({left}: {types[0]}, {right}: {types[1]}):\n{body}\n'
        f'    "func.return"({result}) : ({out_type}) -> ()\n  }}) : () -> ()\n}}) : () -> ()\n'
    )
    return module, types, out_type


def _shape(ty: str) -> tuple[int, ...]:
    return tuple(int(d) for d in ty.removeprefix("tensor<").split("x")[:-1])


@pytest.mark.parametrize("ordinal", [0, 23])
def test_retained_smolvla_site_executes_the_contract(compiler, tmp_path, ordinal):
    stage = repo_root() / RETAINED
    if not (stage / "after.mlir").is_file():
        pytest.skip("retained SmolVLA boundary is not present in this checkout")
    meta = json.loads((stage / "meta.json").read_text())
    assert hashlib.sha256((stage / "after.mlir").read_bytes()).hexdigest() == meta["after_sha256"]
    module, types, out_type = extract_site((stage / "after.mlir").read_text(), ordinal)
    assert line_with(module, '"arith.mulf"').count("fastmath<contract>") == 1
    rng = np.random.default_rng(ordinal)
    left, right = random_bits(rng, _shape(types[0]), "bf16", 2.0), random_bits(rng, _shape(types[1]), "bf16", 2.0)
    got = execute(compiler, tmp_path, module, [(left, "bf16"), (right, "bf16")], _shape(out_type), "bf16")
    assert np.array_equal(got, contract_reference_fast(left, right, "bf16"))
