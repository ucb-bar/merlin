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
    """Every required cell produces an entry that NAMES it, and every other entry is attributable to a
    declared axis. A synthesizer that silently skipped a cell would leave the requirement uncovered
    while looking like it had run; one that emitted an entry no axis asked for would inflate the corpus
    with capsules nothing demands.

    Counting entries against cells was the original form of this check, and it stopped being right once
    the requirement grew axes beyond family/dtype/alignment -- memory regime and the host-only lane each
    produce entries of their own. Attribution is the stronger property anyway: it survives a new axis,
    where a count does not."""
    doc = _spec(target)
    res = CS.synthesize(doc)
    cells = sorted(c["cell"] for c in (doc.get("cells") or ()))

    by_cell = {}
    other = []
    for entry in res["capsules"]:
        ref = entry.get("source_reference") or ""
        hit = next((c for c in cells if c in ref), None)
        if hit:
            by_cell.setdefault(hit, []).append(entry)
        else:
            other.append(entry)
    # A cell with no entry is acceptable ONLY when it is named as unwritable. That is a real capability
    # gap -- an op with no direct-MLIR builder at a dtype the PyTorch writer cannot express -- and the
    # honest handling is an uncovered cell to argue about, not a silently absent one.
    reported = " ".join(res["provenance"].get("cells_no_writer_can_express") or ())
    missing = [c for c in cells if c not in by_cell and c not in reported]
    assert not missing, f"required cell(s) neither synthesized nor reported unwritable: {missing}"

    #: Every non-cell entry must say which axis asked for it. These are the axis markers the
    #: synthesizer writes into `source_reference`; an entry matching none of them is unattributable.
    axes = ("memory regime", "host-only family", "composition axis", "roster axis",
            "rank axis", "layout axis", "host lane", "epilogue axis")
    unattributed = [e["name"] for e in other
                    if not any(a in (e.get("source_reference") or "") for a in axes)]
    assert not unattributed, f"entries no declared axis asked for: {unattributed}"


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
    # CELL entries only. The host-lane axis also emits an `elementwise_map` capsule, and it is the exact
    # inverse of this rule: it exists because the target does NOT admit that family at that dtype, so it
    # must NOT ride a contraction the hardware would then be entitled to accelerate.
    fused = [e for e in res["capsules"]
             if "elementwise_map" in e["name"]
             and (e.get("semantic") or {}).get("generalization_axis") != "host_lane"]
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
    # OP-LEVEL entries only. A model capsule carries no M/K/N -- its extents live in the derived
    # inventory that writes its loader, and demanding them here would demand a shape it does not have.
    for entry in [e for e in res["capsules"] if e.get("op")]:
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
    # ACCELERATOR entries only. A host-only entry is exempt BY CONSTRUCTION: its family is one the
    # manifest admits no capability for, so it has no admitted dtype to draw on and takes the dtype the
    # captures actually carry instead. Holding it to the admitted set would forbid the very capsule that
    # proves the compiler leaves unadmitted work on the host.
    accel = [e for e in res["capsules"]
             if e.get("operand_dtype")
             and "on_mesh" not in ((e.get("lanes") or {}).get("forbid") or ())]
    assert {e["operand_dtype"] for e in accel} <= {"i8"}, (
        "no accelerator entry may use a dtype the requirement does not admit")


def test_synthesis_is_deterministic():
    doc = _spec("gemmini")
    assert CS.synthesize(doc) == CS.synthesize(doc)


def test_every_chosen_op_can_actually_be_materialized():
    """An op in the schema's enum with neither a builder nor a PyTorch body is dead vocabulary; naming
    one would produce an entry nothing can write."""
    pool = CS.available_ops()
    for target in _specs():
        # A model capsule names no op: its program is the derived micro model, whose per-layer ops
        # come from this same pool by construction (micro_model.statement_for).
        for entry in [e for e in CS.synthesize(_spec(target))["capsules"] if e.get("op")]:
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


# --- the negative lane -------------------------------------------------------------------------------
# Families a real capture CONTAINS and the manifest does NOT admit. The requirement derived that set and
# nothing demanded it, so the negative lane was covered only where a hand-authored capsule asserted it.

_HOST_SPEC = {
    "target": "t",
    "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
               "alignment": "aligned", "basis": "observed"}],
    "boundaries": {"extent_probes": [{"boundary": "tile_edge", "edge": 16, "points": [15, 16, 17, 32]}]},
    "host_only": {"families": ["normalization", "reduction"],
                  "dtypes": {"normalization": "bf16", "reduction": "bf16"}},
}


def test_a_host_only_family_becomes_a_forbidding_capsule():
    out = CS.synthesize(_HOST_SPEC)
    made = {e["name"]: e for e in out["capsules"]}
    for family in ("normalization", "reduction"):
        e = made[f"{CS.SYNTH_PREFIX}_host_only_{family}"]
        assert e["lanes"] == {"forbid": ["on_mesh"]}
        assert e["semantic"]["must_accelerate"] is False
        assert e["operand_dtype"] == "bf16"


def test_a_host_only_capsule_requires_no_lane_it_cannot_be_graded_on():
    """Requiring the host lane too would add a demand no op-path grade can measure, turning the capsule
    into a permanent `incomplete` instead of a test. The forbid alone IS enforceable."""
    out = CS.synthesize(_HOST_SPEC)
    e = next(x for x in out["capsules"] if x["name"].endswith("_host_only_normalization"))
    assert "require" not in e["lanes"]


def test_a_host_family_with_no_observed_dtype_is_reported_not_defaulted():
    """A host capsule emitted at a dtype no model uses tests a program nobody runs."""
    spec = {**_HOST_SPEC, "host_only": {"families": ["normalization"], "dtypes": {}}}
    out = CS.synthesize(spec)
    assert any("normalization" in x for x in out["provenance"]["host_only_unsynthesizable"])
    assert not any("host_only" in e["name"] for e in out["capsules"])


def test_a_target_with_no_host_only_families_synthesizes_none():
    spec = {**_HOST_SPEC, "host_only": {"families": [], "dtypes": {}}}
    out = CS.synthesize(spec)
    assert out["provenance"]["host_only_unsynthesizable"] == []
    assert not any("host_only" in e["name"] for e in out["capsules"])


def test_op_choice_prefers_an_op_that_can_actually_be_written_at_the_dtype():
    """Ranking by cost alone picked the cheapest op in the ABSTRACT and then discovered no writer could
    express it: an elementwise cell chose `gelu` -- one operand, no direct-MLIR builder -- over
    `bias_add`, which has one, and then failed at an fp8 dtype the PyTorch writer cannot take. Both are
    elementwise_map, so the cell had a writable representative all along."""
    pool = CS.available_ops()
    # A float dtype the PyTorch writer CAN express: the cheapest op wins, builder or not.
    assert CS.op_for_family("elementwise_map", admitted_ops=pool, dtype="bf16") == "gelu"
    # A dtype it cannot: the choice must fall to an op with a direct-MLIR builder.
    chosen = CS.op_for_family("elementwise_map", admitted_ops=pool, dtype="fp8_e4m3")
    from merlin.targetgen.corpus_spec import BUILDERS
    assert chosen in BUILDERS, f"{chosen!r} has no builder and the PyTorch writer cannot take fp8"


def test_no_required_cell_is_left_without_a_writer():
    """The end state this axis is for: every cell every target requires has an entry something can
    write. A cell with no writer is still REPORTED rather than dropped, so this asserts the count."""
    for target in _specs():
        prov = CS.synthesize(_spec(target))["provenance"]
        assert not prov.get("cells_no_writer_can_express"), (
            f"{target}: {prov['cells_no_writer_can_express']}")


# --------------------------------------------------------------------- the roster axis

def _ws(target: str) -> dict:
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment
    return dict(getattr(load_target_experiment(descriptor_path(target)), "workload_spec", None) or {})


def _roster_entries(target: str) -> list[dict]:
    res = CS.synthesize(_spec(target), workload_spec=_ws(target))
    return [e for e in res["capsules"]
            if (e.get("semantic") or {}).get("generalization_axis") == "roster"]


@pytest.mark.parametrize("target", ["gemmini", "atlas"])
def test_every_declared_roster_model_gets_a_whole_model_capsule(target):
    """The roster is the one thing the workload spec declares that nothing consumed. Every capsule the
    other axes emit is a SLICE; the claim the experiment builds toward is about the roster's real
    networks, and a roster nobody synthesizes a capsule for is a claim nobody can make."""
    declared = [str(m) for m in (_ws(target).get("models") or ())]
    if not declared:
        pytest.skip(f"{target} declares no roster")
    got = {e.get("model") for e in _roster_entries(target)}
    assert got == set(declared), f"{target}: roster {declared} but capsules for {sorted(got)}"


@pytest.mark.parametrize("target", ["gemmini", "atlas"])
def test_the_roster_capsule_compiles_at_the_format_the_target_admits(target):
    """Not at a format someone typed. `precision_policy.best_format` composes the declared preference
    with what the manifest admits, and the capsule carries its answer -- so a target whose hardware has
    no int8 datapath gets its models at the next format it does have, without anything being edited."""
    from merlin.targetgen.precision_policy import best_format

    entries = _roster_entries(target)
    if not entries:
        pytest.skip(f"{target} declares no roster")
    admitted = {str(c["dtype"]) for c in (_spec(target).get("cells") or ())
                if c.get("dtype") and str(c.get("family")) == "contraction"}
    want = best_format(target, preference=(_ws(target).get("precision_preference") or None),
                       admitted=admitted)["chosen"]["capsule_dtype"]
    assert {e["operand_dtype"] for e in entries} == {want}


@pytest.mark.parametrize("target", ["gemmini", "atlas"])
def test_the_roster_capsule_declares_the_scheme_its_arithmetic_needs(target):
    """A capture asked for a quantized format WEIGHT-ONLY emits a float matmul over dequantized weights
    -- the wrong program for a datapath that consumes the narrow format on both operands, and one no
    golden substitution can repair. The scheme is derived from the format, so the entry cannot declare a
    precision without also declaring the arithmetic that produces it."""
    from merlin.targetgen.capsule_source import activation_quantizing_scheme

    entries = _roster_entries(target)
    if not entries:
        pytest.skip(f"{target} declares no roster")
    for e in entries:
        want = activation_quantizing_scheme(e["operand_dtype"])
        assert e.get("quant_scheme") == want, (
            f"{e['name']}: declares {e.get('quant_scheme')!r} for {e['operand_dtype']}, needs {want!r}")


@pytest.mark.parametrize("target", ["gemmini", "atlas"])
def test_the_roster_capsule_requires_the_mesh_rather_than_only_the_numbers(target):
    """A whole-model capsule graded on numerics alone passes a submission that ran the entire network on
    the host. That vacuity was removed from the op capsules and left in the capstones; a synthesized
    roster capsule must not reintroduce it."""
    for e in _roster_entries(target):
        assert "on_mesh" in ((e.get("lanes") or {}).get("require") or ())


def test_a_roster_whose_preference_names_nothing_admitted_reports_it_rather_than_synthesizing():
    """Fail closed. Compiling a roster model in a format the hardware lacks is not a weaker result, it
    is a different one -- so the axis raises naming the roster, instead of quietly picking whatever the
    target happens to admit."""
    doc = _spec("gemmini")
    with pytest.raises(CS.SynthesisError, match="roster axis"):
        CS.synthesize(doc, workload_spec={"models": ["tiny_llama"],
                                          "precision_preference": ["mxfp4"]})
