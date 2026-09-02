"""Capsule entries synthesized from a target's own derived requirement.

The corpus pipeline was deterministic everywhere except at its input: every per-capsule field is derived
and regeneration is byte-stable, but WHICH capsules exist was a hand-written list of ~180 entries across
six profiles -- the one input a new target's owner cannot reasonably be asked to produce. These tests
hold the loop closed, and hold it closed HONESTLY: the failure mode being guarded against is not a wrong
capsule but a missing one, because a requirement that quietly produces nothing is indistinguishable from
a requirement that is met.
"""

from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import corpus_synth as CS


def _spec(target: str) -> dict:
    p = merlin_dir() / "contract/capsules/conformance" / f"{target}.yaml"
    if not p.is_file():
        pytest.skip(f"no tracked conformance spec for {target}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _specs() -> list[str]:
    root = merlin_dir() / "contract/capsules/conformance"
    return sorted(p.stem for p in root.glob("*.yaml")) if root.is_dir() else []


@pytest.mark.parametrize("target", _specs())
def test_every_required_cell_becomes_an_entry(target):
    """One entry per required cell, and the cell it came from is written on it. A synthesizer that
    silently skipped a cell would leave the requirement uncovered while looking like it had run."""
    doc = _spec(target)
    res = CS.synthesize(doc)
    cells = [c["cell"] for c in (doc.get("cells") or ())]
    assert len(res["capsules"]) == len(cells), (
        f"{len(cells)} required cell(s) but {len(res['capsules'])} entry/entries")
    for entry, cell in zip(res["capsules"], sorted(cells)):
        assert cell in entry["source_reference"], "an entry must name the cell it was synthesized for"


def test_a_family_no_op_expresses_raises_rather_than_dropping_the_cell():
    """The core honesty property. `synchronization` is a declared semantic family with no op in either
    the builder set or the PyTorch bodies, so a requirement naming it cannot be met -- and must say so
    instead of emitting five entries for six cells."""
    doc = {"target": "t", "boundaries": {"extent_probes": [{"edge": 16, "points": [15, 16]}]},
           "cells": [{"cell": "synchronization/i8/aligned", "family": "synchronization",
                      "dtype": "i8", "alignment": "aligned"}]}
    with pytest.raises(CS.SynthesisError) as e:
        CS.synthesize(doc)
    assert "synchronization" in str(e.value) and "do not drop the cell" in str(e.value)


def test_the_budget_raises_rather_than_truncating():
    """A silently dropped point reads downstream as a covered one, which is the whole failure this
    pipeline exists to prevent."""
    doc = {"target": "t", "boundaries": {"extent_probes": [{"edge": 16, "points": [15, 16]}]},
           "cells": [{"cell": f"contraction/i8/{a}", "family": "contraction", "dtype": "i8",
                      "alignment": a} for a in ("aligned", "partial")]}
    with pytest.raises(CS.SynthesisError) as e:
        CS.synthesize(doc, budget=1)
    assert "never" in str(e.value) and "truncate" in str(e.value)


def test_a_fused_only_family_is_carried_as_an_epilogue_not_a_standalone_op():
    """The one case where the obvious choice is a DEFECT.

    A family declared `composed_with: [contraction]` is fused-only: the eligibility oracle refuses a
    standalone capsule for it as a false fallback, so the only capsule that can ever evidence that cell
    is a contraction carrying it as an epilogue. The conformance ratchet spells this out for gemmini in
    exactly those terms.
    """
    doc = _spec("gemmini")
    res = CS.synthesize(doc)
    fused = [e for e in res["capsules"] if "elementwise_map" in e["name"]]
    if not fused:
        pytest.skip("this target's requirement has no fused-only family")
    for entry in fused:
        assert entry["op"] == "matmul", "a fused-only family must ride a contraction"
        assert entry.get("epilogue"), "…and must actually carry an epilogue stage"


def test_alignment_decides_the_shape():
    doc = _spec("gemmini")
    res = CS.synthesize(doc)
    by = {e["name"]: e for e in res["capsules"]}
    aligned = next(e for n, e in by.items() if n.endswith("_aligned"))
    partial = next(e for n, e in by.items() if n.endswith("_partial"))
    assert "tile" in str(aligned["N"]) and "-" not in str(aligned["N"])
    assert str(partial["N"]).endswith("-1"), "a partial cell must rag an axis off the boundary"


def test_extents_are_tile_relative_not_baked_integers():
    """The synthesized file must describe a shape RELATIVE to whatever edge the hardware has. Baking
    integers would make one target's geometry a property of a file that is supposed to be portable."""
    res = CS.synthesize(_spec("gemmini"))
    for entry in res["capsules"]:
        for axis in ("M", "K", "N"):
            assert isinstance(entry[axis], str) and "tile" in entry[axis], (
                f"{entry['name']}.{axis} = {entry[axis]!r} is not tile-relative")


def test_a_preference_cannot_widen_what_the_target_admits():
    """A preference RANKS the dtypes the hardware already has. A token that does not survive the filter
    is reported rather than silently ignored, because "we preferred int8 and this target has no int8
    datapath" is a fact the reader of a synthesized corpus needs."""
    res = CS.synthesize(_spec("gemmini"),
                        workload_spec={"precision_preference": ["int8", "fp8_e4m3", "bf16"]})
    prov = res["provenance"]
    assert prov["precision_preference_kept"] == ["i8"], (
        "the preference must be compared in the CAPSULE dtype spelling the cells use")
    assert set(prov["precision_preference_dropped"]) == {"fp8_e4m3", "bf16"}
    dtypes = {e["operand_dtype"] for e in res["capsules"]}
    assert dtypes <= {"i8"}, "no entry may use a dtype the requirement does not admit"


def test_synthesis_is_deterministic():
    doc = _spec("gemmini")
    assert CS.synthesize(doc) == CS.synthesize(doc)


def test_every_chosen_op_can_actually_be_materialized():
    """An op in the schema's enum with neither a builder nor a PyTorch body is dead vocabulary; naming
    one would produce an entry nothing can write."""
    pool = CS.available_ops()
    for target in _specs():
        for entry in CS.synthesize(_spec(target))["capsules"]:
            assert entry["op"] in pool, f"{entry['name']} names unmaterializable op {entry['op']!r}"


class TestQuantizedCapture:
    """A capsule meant to exercise an INTEGER datapath must contain integer arithmetic.

    The default int8 scheme is weight-only, which emits a float matmul over dequantized weights. That
    is the right capture for a model ladder and the wrong PROGRAM for a capsule about a systolic
    integer mesh -- and it cannot be repaired by substituting a golden, because the program contains no
    integer contraction to grade. These pin the mechanism that lets an entry ask for the other one.
    """

    def test_a_quantized_capture_needs_a_weight_parameter(self):
        """torchAO quantizes module WEIGHTS. Every default op body takes both operands as INPUTS
        (`a @ w`), so a scheme has nothing to bind to and is silently a no-op -- measured: a capture
        requested at W8A8 still emitted `aten.mm.default` over f32 while faithfully recording
        `prov.quantization = "int8_dyn_act_int8_weight"`. The declared scheme was true and the program
        was not."""
        from merlin.targetgen.capsule_source import build_loader_src

        plain = build_loader_src({"op": "linear", "dtype": "int8", "seed": 1,
                                  "M": 16, "K": 32, "N": 16})
        assert "nn.Linear" not in plain, "the default body is parameter-free by design"

        quant = build_loader_src({"op": "linear", "dtype": "int8", "seed": 1, "M": 16, "K": 32,
                                  "N": 16, "quant_scheme": "int8_dyn_act_int8_weight"})
        assert "nn.Linear" in quant, "a quantized capture needs a parameter for the scheme to bind to"

    def test_asking_to_quantize_a_non_contraction_is_refused(self):
        """Only a contraction carries a weight a torchAO scheme can act on. An unquantized program
        shipped under a quantized name is worse than a refusal."""
        import pytest as _pytest

        from merlin.targetgen.capsule_source import build_loader_src

        with _pytest.raises(ValueError, match="worse than a refusal"):
            build_loader_src({"op": "softmax", "dtype": "int8", "seed": 1, "M": 16, "K": 32, "N": 16,
                              "quant_scheme": "int8_dyn_act_int8_weight"})

    def test_the_interface_marker_accepts_an_integer_contraction(self):
        """A contraction arrives tagged `matmul` (float `linalg.matmul`) or `int_matmul` (the
        `aten._int_mm` form W8A8 emits). Matching only the float spelling refused the very capture
        carrying the arithmetic a systolic mesh runs."""
        from merlin.targetgen.capsule_source import _OP_MARKER

        for op in ("matmul", "linear", "attention_qk"):
            assert "int_matmul" in _OP_MARKER[op], f"{op} must accept an integer contraction"
            assert "matmul" in _OP_MARKER[op], f"{op} must still accept a float contraction"

    def test_the_scheme_travels_from_entry_to_capture_spec(self):
        from merlin.targetgen.capsule_source import _capture_spec

        class _B:
            tile_dim = 16
            operand_dtype = "int8"

        spec = _capture_spec({"name": "x", "op": "linear", "M": 16, "K": 32, "N": 16,
                              "quant_scheme": "int8_dyn_act_int8_weight"}, _B())
        assert spec["quant_scheme"] == "int8_dyn_act_int8_weight"
        assert "quant_scheme" not in _capture_spec(
            {"name": "y", "op": "linear", "M": 16, "K": 32, "N": 16}, _B())
