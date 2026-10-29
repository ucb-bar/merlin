"""Source-proven CPU packets are independent of any accelerator provider."""

from dataclasses import replace

import pytest

from merlin.llvmlower.constant_fma_packet import analyze, rewrite
from merlin.runtime.host_fma_packet import RV64GCScalarFmaCapability

POLICY = dict(ordinary_nontrapping=True, exception_flags_unobserved=True)


def source(width, mode):
    values = ("%a", "%b", "1.0")
    if mode == "constant_rhs":
        values = ("%a", "1.0", "%b")
    elif mode == "constant_lhs_addend":
        values = ("1.0", "%a", "2.0")
    lines = [
        f"  %r{i} = call float @llvm.fma.f32(" + ", ".join("float " + value for value in values) + ")"
        for i in range(width)
    ]
    return (
        "define float @probe(float %a,float %b) {\n"
        + "\n".join(lines)
        + "\n  ret float %r0\n}\n"
        + "declare float @llvm.fma.f32(float,float,float)\n"
    )


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("mode", ["constant_rhs", "constant_addend", "constant_lhs_addend"])
def test_explicit_source_packet_and_early_clobber(width, mode):
    original = source(width, mode)
    result, report = rewrite(original, width=width, emitter=RV64GCScalarFmaCapability().emit, **POLICY)
    assert len(report["packets"]) == 1
    assert result.count("fmadd.s") == width
    assert result.count("=&f") == width
    assert "~{ft0}" in result
    assert "ret float %r0" in result
    assert "extractvalue" in result
    assert "csrw" not in result and "fcvt" not in result


@pytest.mark.parametrize(
    "field,value", [("isa", "rv64gcv"), ("abi", "lp64"), ("ieee_f32_fma", False), ("ieee_gradual_underflow", False)]
)
def test_capability_refusals(field, value):
    (packet,) = analyze(source(2, "constant_addend"), width=2, **POLICY)
    capability = replace(RV64GCScalarFmaCapability(), **{field: value})
    with pytest.raises(ValueError, match="capability"):
        capability.emit(packet, "%temporary")


def test_mutated_source_constant_and_mode_refuse():
    (packet,) = analyze(source(2, "constant_addend"), width=2, **POLICY)
    with pytest.raises(ValueError, match="constant witness"):
        RV64GCScalarFmaCapability().emit(replace(packet, constant_words=(0x40000000,)), "%t")
    with pytest.raises(ValueError, match="mode"):
        RV64GCScalarFmaCapability().emit(replace(packet, mode="unknown"), "%t")
