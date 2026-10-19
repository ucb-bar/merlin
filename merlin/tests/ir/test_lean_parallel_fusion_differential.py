"""Independent native references for real bounded fusion and a retained capture gap."""

import json
import math

import pytest
from test_lean_parallel_fusion import source, stage_for
from test_lean_reshape_compose import _verify_mlir

from merlin.verify.lean import differential as diff


def _compare(program, samples, expected, work, tools, bits):
    diff.compare(program, samples, work, tools, function="forward", index_bits=bits)
    assert json.loads((work / "native-results.json").read_bytes())["results"] == expected


def _reference(values, width, producer, consumer):
    mask = (1 << width) - 1
    produced = [((x + x) if producer == "addi" else (x * x)) & mask for x in values]
    return [str(((x + x) if consumer == "addi" else (x * x)) & mask) for x in produced]


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize(
    "shape,width,producer,consumer",
    [
        ((2,), 13, "addi", "muli"),
        ((), 8, "muli", "addi"),
        ((0, 3), 32, "addi", "muli"),
        ((2, 3), 64, "muli", "addi"),
    ],
)
def test_real_fused_integer_bodies_match_original_and_independent_reference(
    tmp_path, fidelity_tools, bits, shape, width, producer, consumer
):
    program = source(shape, width, producer, consumer)
    _, _, after = stage_for(tmp_path / "actual", fidelity_tools.compiler_python, program)
    n = math.prod(shape)
    mask = (1 << width) - 1
    cases = [[0, 1, mask, 1 << (width - 1), 7, 13], [mask, 3, 11, 5, 17, 19]]
    samples = [[values[:n], [17 + i for i in range(n)], [73 + i for i in range(n)]] for values in cases]
    expected = [_reference(values[0], width, producer, consumer) for values in samples]
    for label, text in (("original", program), ("actual-fused", after.read_text())):
        _compare(text, samples, expected, tmp_path / label, fidelity_tools, bits)


def test_changed_fused_primitive_has_observable_wrong_output(tmp_path, fidelity_tools):
    program = source()
    _, _, after = stage_for(tmp_path / "actual", fidelity_tools.compiler_python, program)
    assert '"arith.addi"' in after.read_text()
    mutant = after.read_text().replace('"arith.addi"', '"arith.subi"', 1)
    path = tmp_path / "mutant.mlir"
    path.write_text(mutant)
    _verify_mlir(path, fidelity_tools.compiler_python)
    samples = [[[2, 3], [11, 13], [17, 19]]]
    _compare(program, samples, [["16", "36"]], tmp_path / "original", fidelity_tools, 64)
    _compare(mutant, samples, [["0", "0"]], tmp_path / "wrong-target", fidelity_tools, 64)


def test_captured_scalar_fusion_native_matches_without_claiming_proof(tmp_path, fidelity_tools):
    program = source(captured=True)
    _, _, after = stage_for(tmp_path / "actual", fidelity_tools.compiler_python, program)
    samples = [
        [[2, 3], [11, 13], [17, 19], [7]],
        [[1, 2], [23, 29], [31, 37], [4294967295]],
    ]
    expected = [[str(((x + capture[0]) & 4294967295) ** 2 & 4294967295) for x in xs]
                for xs, _, _, capture in samples]
    # The binding suite requires this changed boundary to abstain. Equal sampled
    # executions are empirical evidence and do not discharge its missing proof.
    for label, text in (("original", program), ("actual-fused", after.read_text())):
        _compare(text, samples, expected, tmp_path / label, fidelity_tools, 64)
