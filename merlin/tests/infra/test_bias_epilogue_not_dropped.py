"""A declared bias stage must reach the arithmetic — in every engine, or loudly not at all.

The fusion (PF) performance family compares a fused matmul+bias against the matmul and the bias
add it replaces. Every one of its members was ungradeable, for two reasons that both presented as
"the capsule is wrong":

* the FUSED members failed L0/L1 with 224 of 256 elements off and ``max_abs == 3``, which reads
  like a rounding or requant-ordering defect. It was neither: the bias never ran. The name of the
  bias tensor is carried in the COMMIT *attributes* by a buffer that came through the interface
  grammar and in the COMMIT *operands* by one an emitter built directly. The golden engine read
  only the attributes; the reference/simulate engines read only the operands and then skipped the
  stage when they found nothing — silently. Every element differed from the golden by exactly its
  own column's bias, and ``max_abs`` was just ``max(B)``.
* the UNFUSED bias_add parts were rejected before grading: ``BIAS_ADD`` is an opcode the canonical
  interface grammar emits, but the command-buffer schema's opcode enum did not list it, so a
  conformant buffer was scored a PROTOCOL_VIOLATION.

The tests below fire on the real corpus capsules rather than on synthetic buffers, and each
asserts the bias is ACTUALLY LOAD-BEARING first — a bias of all zeros would let a dropped-bias
engine pass this file and prove nothing.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.runtime.commandbuffer import bias_tensor_name
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

_PERF = merlin_dir() / "contract/capsules/_perf"

#: The fusion group: a fused member plus the two parts it replaces, at both declared shapes.
_FUSION_GROUP = [
    "PF00_fused_matmul_bias_m16k16n16",
    "PF01_matmul_m16k16n16",
    "PF02_bias_add_m16k16n16",
    "PF03_fused_matmul_bias_m16k32n16",
    "PF04_matmul_m16k32n16",
    "PF05_bias_add_m16k32n16",
]
_FUSED = ["PF00_fused_matmul_bias_m16k16n16", "PF03_fused_matmul_bias_m16k32n16"]


def _capsule(name: str):
    d = _PERF / name
    if not d.is_dir():
        pytest.skip(f"{name} is not present in this checkout")
    cb = parse_interface_mlir((d / "capsule.interface.mlir").read_text(encoding="utf-8"))
    golden = yaml.safe_load((d / "golden.yaml").read_text(encoding="utf-8"))["outputs"]
    return cb, golden


def _flat(x):
    if isinstance(x, list):
        return [v for item in x for v in _flat(item)]
    return [x]


# --------------------------------------------------------------------------------------------
# The bug itself: the bias must reach the arithmetic, on every capsule in the group.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", _FUSION_GROUP)
def test_fusion_group_member_is_bit_exact_at_the_math_floor(name):
    """L0 (reference) and L1 (simulate) reproduce the independent golden EXACTLY.

    `exact_int` is the capsules' declared policy, so this asserts equality — not a tolerance. The
    fused members returned 224/256 mismatches here before the bias resolver was shared.
    """
    cb, golden = _capsule(name)
    ref = reference_outputs(cb)
    sim = simulate(cb)["outputs"]
    for out, want in golden.items():
        assert out in ref, f"{name}: reference produced no {out!r}"
        assert out in sim, f"{name}: simulate produced no {out!r}"
        assert _flat(ref[out]) == _flat(want), f"{name}: reference does not match golden for {out}"
        assert _flat(sim[out]) == _flat(want), f"{name}: simulate does not match golden for {out}"


@pytest.mark.parametrize("fused,part", [(_FUSED[0], "PF01_matmul_m16k16n16"),
                                        (_FUSED[1], "PF04_matmul_m16k32n16")])
def test_the_bias_is_load_bearing_so_the_exactness_test_can_fail(fused, part):
    """Guard against a vacuous suite: the fused golden must actually DIFFER from the unfused matmul.

    If the corpus ever generated an all-zero bias, an engine that drops the stage entirely would
    satisfy the test above while proving nothing about the stage the capsule declares.
    """
    _, fused_golden = _capsule(fused)
    _, part_golden = _capsule(part)
    f, p = _flat(fused_golden["Y0"]), _flat(part_golden["Y0"])
    assert len(f) == len(p)
    differing = sum(1 for a, b in zip(f, p) if a != b)
    assert differing > 0, (
        f"{fused} and {part} have identical goldens: the bias contributes nothing, so no engine "
        f"can be caught dropping it")
    # The exact signature the defect produced, asserted as the thing this test is sensitive to.
    assert differing >= len(f) // 2, (
        f"only {differing}/{len(f)} elements carry the bias — too weak to detect a dropped stage")


@pytest.mark.parametrize("name", _FUSED)
def test_dropping_the_bias_is_detected_rather_than_tolerated(name):
    """Strip the bias name from the buffer and the engines must NOT quietly return the unbiased value.

    This is the regression's negative control. Before the fix this mutation was indistinguishable
    from the real buffer — which is precisely why the stage could vanish without any engine
    complaining.
    """
    cb, _ = _capsule(name)
    for cmd in cb["commands"]:
        if cmd["opcode"] == "COMMIT":
            cmd.get("operands", {}).pop("bias", None)
            cmd.get("attributes", {}).pop("bias", None)
    with pytest.raises(ValueError):
        reference_outputs(cb)
    with pytest.raises(Exception):
        simulate(cb)


# --------------------------------------------------------------------------------------------
# The shared resolver.
# --------------------------------------------------------------------------------------------

def test_bias_name_resolves_from_either_spelling():
    """Both placements are legitimate and both are produced by this tree's own emitters."""
    assert bias_tensor_name({}, {"bias": "B"}, op="t") == "B"           # interface-grammar buffer
    assert bias_tensor_name({"bias": "B"}, {}, op="t") == "B"           # directly-built buffer
    assert bias_tensor_name({"bias": "B"}, {"bias": "B"}, op="t") == "B"


def test_bias_name_fails_closed_when_no_tensor_is_named():
    """A declared stage with no operand is unanswerable — never a silent no-op."""
    with pytest.raises(ValueError) as exc:
        bias_tensor_name({}, {}, op="COMMIT 'Y0'")
    assert "COMMIT 'Y0'" in str(exc.value)


# --------------------------------------------------------------------------------------------
# The protocol gate that rejected the unfused parts.
# --------------------------------------------------------------------------------------------

def test_schema_accepts_every_opcode_the_interface_grammar_can_emit():
    """DERIVED coverage, so this cannot drift again.

    The grammar's own operand table is the source of truth for what a conformant buffer may
    contain. An opcode it emits that the schema rejects is a contract contradicting itself, and it
    surfaces as a PROTOCOL_VIOLATION blaming the submission for the harness's drift.
    """
    from merlin.targetgen.contract import interface_emit as IE
    from merlin.targetgen.contract.schemas import load_schema

    schema = load_schema("command_buffer")
    enum = set(schema["properties"]["commands"]["items"]["properties"]["opcode"]["enum"])
    emittable = set(IE._NAMED_OP_TO_OPCODE.values())
    assert not (emittable - enum), (
        f"the interface grammar emits {sorted(emittable - enum)}, which the command_buffer schema "
        f"rejects; a conformant buffer would be scored a protocol violation")


@pytest.mark.parametrize("name", ["PF02_bias_add_m16k16n16", "PF05_bias_add_m16k32n16"])
def test_standalone_bias_add_is_schema_valid(name):
    """The unfused half of the fusion comparison has to survive the protocol gate to be graded."""
    jsonschema = pytest.importorskip("jsonschema")
    from merlin.targetgen.contract.schemas import load_schema

    cb, _ = _capsule(name)
    assert any(c["opcode"] == "BIAS_ADD" for c in cb["commands"]), \
        f"{name} no longer lowers to a standalone BIAS_ADD"
    jsonschema.validate(cb, load_schema("command_buffer"))


@pytest.mark.parametrize("name", ["PF02_bias_add_m16k16n16", "PF05_bias_add_m16k32n16"])
def test_standalone_bias_add_is_modeled_by_both_math_floor_engines(name):
    """Neither engine may report BIAS_ADD as an opcode it has no definition for."""
    cb, _ = _capsule(name)
    assert "Y0" in reference_outputs(cb)
    assert "Y0" in simulate(cb)["outputs"]
