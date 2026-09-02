"""The two capsules an L5 fusion claim compares, built from one shape statement.

The claim is that `matmul+bias` fused costs less than the `matmul` and the `bias_add` it replaces. Two
things blocked expressing it at a target's own datapath dtype:

* the fused member had nowhere to declare the bias. ``epilogue = ["bias_add"]`` and ``role = "bias"``
  were both already in the interface contract, but nothing named WHICH operand the stage adds -- so a
  module could say a bias happens without saying what to add, and the golden engine's bias stage
  (`capsule_golden._apply_epilogue`) reached for a tensor no builder declared.
* the unfused half was not expressible at all, so the comparison group had one member. The generator
  already refuses a group of one, correctly: a group that cannot be compared is a declaration with no
  content.

The dtype is the load-bearing detail. The bias lands on the ACCUMULATOR, before any requant, so on an
``i8 x i8 -> i32`` datapath the bias vector is i32 and the standalone member's operands are i32. Declared
in the operand dtype instead, the two members would be adding different numbers and their cycles would
not be summable.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import semantic_families as SF
from merlin.targetgen.corpora import descriptor_path
from merlin.targetgen.target_experiment import load_target_experiment

sys.path.insert(0, str(merlin_dir() / "contract" / "capsules"))

_TARGETS = ("gemmini", "atlas")


def _binding(target: str):
    import generate_corpus as GC
    prof = GC.load_profile(target)
    te = load_target_experiment(descriptor_path(target))
    return CS.derive_binding(te, prof.get("datapath") or {})


def _entry(name: str, op: str, **kw) -> dict:
    return {"name": name, "kind": "model_slice", "cat": "_perf", "op": op,
            "source_role": "derived_sweep", "source_reference": "fusion pair test",
            "out": "Y0", **kw}


@pytest.fixture(params=_TARGETS, ids=_TARGETS)
def binding(request):
    try:
        return _binding(request.param)
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"{request.param} does not resolve a binding: {type(exc).__name__}: {exc}")


def _role(cap, role):
    return next((i for i in cap["inputs"] if i.get("role") == role), None)


def test_the_fused_member_declares_the_bias_it_adds(binding):
    cap, mlir = CS.build(_entry("F", "fused_matmul_bias", M=16, K=16, N=16), binding)
    attrs = cap["operation"]["attributes"]
    assert "bias_add" in attrs["epilogue"], "the op name implies the stage; it must appear"
    assert attrs.get("bias"), "the stage is declared but not the operand it consumes"
    bias = _role(cap, "bias")
    assert bias is not None and bias["name"] == attrs["bias"]
    assert f'bias = "{attrs["bias"]}"' in mlir, "the emitted module does not name the bias operand"
    assert f'role = "bias"' in mlir


def test_the_bias_is_declared_in_the_accumulator_dtype(binding):
    """Not the operand dtype: the addition happens on the accumulator, before any requant."""
    accum = binding.cap_dtype(binding.accum_dtype)
    operand = binding.cap_dtype(binding.operand_dtype)
    cap, _ = CS.build(_entry("F", "fused_matmul_bias", M=16, K=16, N=16), binding)
    bias = _role(cap, "bias")
    assert bias["dtype"] == accum, (
        f"bias declared {bias['dtype']!r}; the stage runs in the accumulator domain ({accum!r})")
    if accum != operand:
        assert bias["dtype"] != operand, "bias must not be declared in the operand dtype"
    assert bias["shape"] == [16], "a per-column bias is a length-N vector"


def test_the_two_members_share_the_stages_arithmetic(binding):
    """The fused golden must exceed the plain matmul golden by EXACTLY the bias.

    Not a smoke test: the golden engine's terminal branch used to skip an unrecognised epilogue stage
    silently, and the failure that produces is a capsule shipping a golden without the arithmetic it
    declared -- which then agrees with a backend that dropped the same stage.
    """
    shape = dict(M=8, K=16, N=8)
    plain, _ = CS.build(_entry("P", "matmul", **shape), binding)
    fused, _ = CS.build(_entry("F", "fused_matmul_bias", **shape), binding)
    if CG.is_independent_float_golden(fused, ""):
        pytest.skip("this target's golden is read from an independent oracle, not recomputed here")
    gp = CG.golden({**plain, "__dir__": ""})["Y0"]
    gf = CG.golden({**fused, "__dir__": ""})["Y0"]
    bias = CG.materialize_capsule_leaves(fused)[fused["operation"]["attributes"]["bias"]].to_list()
    for r, (rp, rf) in enumerate(zip(gp, gf)):
        assert [b - a for a, b in zip(rp, rf)] == list(bias), (
            f"row {r}: the fused golden does not differ from the matmul golden by the bias")


def test_the_standalone_member_computes_the_same_addition(binding):
    cap, mlir = CS.build(_entry("B", "bias_add", M=4, N=4), binding)
    assert "merlin_iface.bias_add" in mlir
    accum = binding.cap_dtype(binding.accum_dtype)
    assert [i["dtype"] for i in cap["inputs"]] == [accum, accum], (
        "the unfused half operates in the accumulator domain, like the stage it is")
    if CG.is_independent_float_golden(cap, ""):
        pytest.skip("this target's golden is read from an independent oracle, not recomputed here")
    env = CG.materialize_capsule_leaves(cap)
    got = CG.golden({**cap, "__dir__": ""})["Y0"]
    x, b = env["X"].to_list(), env["B"].to_list()
    assert got == [[xv + bv for xv, bv in zip(row, b)] for row in x]


def test_neither_member_owes_an_instruction_its_target_lacks(binding):
    """Derived, not asserted: the standalone add owes no multiply, and no vector class is invented.

    A target that folds the bias into its accumulator read-out has no separate class for it, and the
    honest demand is then the data motion alone -- not a fabricated vector instruction, and not the
    systolic sequence, which is what a RoCC target used to be told it owed for every op.
    """
    part, _ = CS.build(_entry("B", "bias_add", M=16, N=16), binding)
    classes = part["expected"]["instruction_classes"]
    assert classes, "an empty requirement is satisfied by emitting nothing at all"
    contraction = [c for c in classes if SF.from_isa_class(c) == "contraction"]
    assert not contraction, f"a standalone bias add is required to issue {contraction}"

    fused, _ = CS.build(_entry("F", "fused_matmul_bias", M=16, K=16, N=16), binding)
    assert fused["expected"]["instruction_classes"], "the fused member owes nothing at all"
