from __future__ import annotations

from verification_witnesses.quantized_pair_source import (
    CG,
    CS,
    deepcopy,
    parse_interface_mlir,
    pytest,
    source,
    validate_capsule,
)


@pytest.mark.parametrize("bias,skip,expected,side", [(3, 1, 3, 0), (-257, 127, 0, 1)])
def test_public_builder_and_independent_original_stage_evaluators(bias, skip, expected, side):
    (entry, binding) = source(bias, skip, side)
    original = deepcopy(entry)
    (cap, interface) = CS.build(entry, binding)
    validate_capsule(cap)
    cb = parse_interface_mlir(interface)
    from merlin.compiler.lower.model.quantized_sum_source import (
        plan_quantized_sum_commands,
        split_quantized_sum_commands,
    )

    bound_source = plan_quantized_sum_commands(cb)
    bound_source.revalidate()
    (p, q) = split_quantized_sum_commands(bound_source)
    assert p["commands"] == cb["commands"][:-1] and q["commands"] == cb["commands"][-1:]
    conv = next((c for c in cb["commands"] if c["opcode"] == "CONV2D"))
    residual = cb["commands"][-1]
    assert residual["operands"][("lhs", "rhs")[side]] == conv["operands"]["dst"]
    assert conv["attributes"]["acc_scale"] == 0.5 and residual["attributes"]["source_scales"] == [1.0] * 3
    assert CG.golden(cap) == {"sum_Y0": [[expected]] * 4}
    (original_final, _) = CS.build(entry["components"][1], binding)
    assert cap["numeric_policy"] == original_final["numeric_policy"]
    assert entry == original
