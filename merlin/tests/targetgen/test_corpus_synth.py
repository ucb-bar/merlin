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
            "rank axis", "layout axis", "host lane", "epilogue axis", "application axis",
            "accumulation-depth axis")
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
    #
    # And NOT the application axis, whose whole content is a shape a real model contains. That one is
    # deliberately not portable across tile edges -- `968` has no tile-relative spelling at tile 16 --
    # which is exactly why it carries `source_role: model_derived` instead of `derived_sweep`. Scoping
    # by that role rather than by name keeps this test meaning "a SWEEP's shapes track the geometry".
    for entry in [e for e in res["capsules"]
                  if e.get("op") and e.get("source_role") != "model_derived"]:
        for axis in ("M", "K", "N"):
            assert isinstance(entry[axis], str) and "tile" in entry[axis], (
                f"{entry['name']}.{axis} = {entry[axis]!r} is not tile-relative")


def test_an_application_capsule_carries_a_concrete_shape_and_says_where_it_came_from():
    """The inverse of the rule above, pinned so the exemption cannot quietly widen: only a
    `model_derived` entry may bake integers, and it must name the class and the model it represents."""
    doc = _spec("gemmini")
    doc["application_shapes"] = {
        "required": [{"class": "contraction/i8/aligned/spills/rank2/squareish_gemm",
                      "M": 64, "K": 64, "N": 64, "batch": 1, "tier": "L3", "extends": None,
                      "basis": {"sized_by": "measured_cost_model", "representative_of": 12,
                                "source": "an_app"}}],
        "cert_budget_s": 300.0,
    }
    made = [e for e in CS.synthesize(doc)["capsules"]
            if (e.get("semantic") or {}).get("generalization_axis") == "application"]
    assert len(made) == 1
    entry = made[0]
    assert entry["source_role"] == "model_derived"
    assert (entry["M"], entry["K"], entry["N"]) == (64, 64, 64)
    assert "an_app" in entry["source_reference"] and "squareish_gemm" in entry["source_reference"]


def test_an_l2_only_application_capsule_names_the_sibling_it_extends():
    """A large capsule resting on nothing is the failure this axis exists to avoid. The cap and the
    sibling travel together on the entry so the generator can enforce both."""
    doc = _spec("gemmini")
    cls = "contraction/i8/aligned/spills/rank2/wide_skinny"
    doc["application_shapes"] = {
        "required": [
            # The sibling has to be here: an L2 capsule with no certified sibling is dropped, which is
            # the property the next test pins.
            {"class": cls, "M": 32, "K": 64, "N": 32, "batch": 1, "tier": "L3", "extends": None,
             "basis": {"sized_by": "measured_cost_model", "representative_of": 12,
                       "source": "an_app"}},
            {"class": cls, "M": 256, "K": 64, "N": 784, "batch": 1, "tier": "L2", "extends": cls,
             "basis": {"sized_by": "application_shape", "representative_of": 12,
                       "source": "an_app"}},
        ],
        "cert_budget_s": 300.0,
    }
    entry = [e for e in CS.synthesize(doc)["capsules"]
             if (e.get("semantic") or {}).get("generalization_axis") == "application"][-1]
    assert entry["max_oracle_tier"] == "L2"
    assert entry["extends"], "an L2-only capsule must say what carries its cycle-accurate guarantee"
    assert "extends" in entry["source_reference"]


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


def test_an_unsized_depth_is_not_reported_as_a_missing_writer():
    """Two absences that license opposite actions must not share a channel.

    A cell with no writer needs somebody to build a builder. A depth that could not be sized needs a
    CERTIFICATION RUN -- the writer is there, and the measured history to size it against is not. A
    target that has simply never been timed would otherwise read as one whose corpus cannot express a
    reduction, which is both false and the more alarming of the two."""
    for target in _specs():
        res = CS.synthesize(_spec(target))
        prov = res["provenance"]
        joined = " ".join(prov.get("cells_no_writer_can_express") or ())
        assert "certifiable" not in joined and "certification history" not in joined, (
            f"{target}: an unsized depth is being reported as a missing writer")
        # THE INVARIANT, stated over what was EMITTED rather than over which target it is: a target
        # that produced no depth capsule owes a reason, and one that produced some owes nothing. Keying
        # this on a target name instead was wrong the moment the cost law stopped being per-target --
        # mx_gemmini has no measured history of its own and still sizes a depth, because the law prices
        # the output tile and the tile edge is a fact about the target rather than about its run log.
        emitted = [e for e in res["capsules"]
                   if (e.get("semantic") or {}).get("generalization_axis") == "accumulation_depth"]
        if not emitted:
            assert prov.get("accumulation_depth_unsizable"), (
                f"{target} synthesized no depth capsule and said nothing about why")


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


def test_an_l2_application_capsule_with_no_certified_sibling_is_dropped():
    """The `extends` relation is only worth anything if it is enforced. A large capsule whose
    cycle-accurate sibling was never emitted rests on nothing — and an L2 pass on a shape nothing
    ever certified is exactly the "read tier_reached, never a bare score" failure this corpus has
    scar tissue for."""
    doc = _spec("gemmini")
    cls = "contraction/i8/aligned/spills/rank2/wide_skinny"
    doc["application_shapes"] = {
        "required": [{"class": cls, "M": 256, "K": 64, "N": 784, "batch": 1, "tier": "L2",
                      "extends": cls,
                      "basis": {"sized_by": "application_shape", "representative_of": 5,
                                "source": "app"}}],
        "cert_budget_s": 300.0,
    }
    res = CS.synthesize(doc)
    made = [e for e in res["capsules"]
            if (e.get("semantic") or {}).get("generalization_axis") == "application"]
    assert made == [], "an L2 capsule with no certified sibling may not ship"
    holes = " ".join(res["provenance"].get("cells_no_writer_can_express") or ())
    assert "rest on nothing" in holes, "the dropped capsule must say why, not vanish"


def test_an_l2_application_capsule_ships_when_its_sibling_does():
    """The other half: the relation must not become a blanket refusal."""
    doc = _spec("gemmini")
    cls = "contraction/i8/aligned/spills/rank2/wide_skinny"
    doc["application_shapes"] = {
        "required": [
            {"class": cls, "M": 32, "K": 64, "N": 32, "batch": 1, "tier": "L3", "extends": None,
             "basis": {"sized_by": "measured_cost_model", "representative_of": 5, "source": "app"}},
            {"class": cls, "M": 256, "K": 64, "N": 784, "batch": 1, "tier": "L2", "extends": cls,
             "basis": {"sized_by": "application_shape", "representative_of": 5, "source": "app"}},
        ],
        "cert_budget_s": 300.0,
    }
    made = [e for e in CS.synthesize(doc)["capsules"]
            if (e.get("semantic") or {}).get("generalization_axis") == "application"]
    assert [e.get("max_oracle_tier") for e in made] == [None, "L2"]


# ------------------------------------------------------------------- the accumulation-depth axis

def _depth_axis(res):
    return [e for e in res["capsules"]
            if (e.get("semantic") or {}).get("generalization_axis") == "accumulation_depth"]


def _with_depth(doc, *, certified=True, refusal=None, regimes=True, ceiling=862):
    """A spec carrying an affordability ceiling and the residency depths a real target derives."""
    doc = dict(doc)
    doc["cert_affordability"] = {"max_elements": ceiling, "budget_s": 300.0,
                                 "metric": "written_output_elements"}
    mm = dict(doc.get("memory_mapping") or {})
    mm["regime_dtype"] = mm.get("regime_dtype") or "i8"
    mm["reduction_depth"] = {
        "certified": ({"M": 16, "K": 176, "N": 16, "K_tiles": 11, "predicted_seconds": 81.0,
                       "budget_s": 300.0, "sized_by": "measured_cost_model",
                       "cost_fit": {"n_samples": 32}} if certified else None),
        "certified_refusal": refusal,
        "by_regime": ({
            "fits_single": {"points": [{"M": 16, "K": 4112, "N": 16, "K_tiles": 257,
                                        "fraction_of_capacity": 0.5},
                                       {"M": 16, "K": 8192, "N": 16, "K_tiles": 512,
                                        "fraction_of_capacity": 1.0}]},
            "spills": {"points": [{"M": 16, "K": 16384, "N": 16, "K_tiles": 1024,
                                   "fraction_of_capacity": 2.0}]},
        } if regimes else {}),
    }
    doc["memory_mapping"] = mm
    return doc


def _passes(entry):
    """The tile multiple, which IS the number of accumulation passes this capsule performs."""
    assert entry["M"] == "tile" and entry["N"] == "tile"
    return int(str(entry["K"]).partition("*tile")[0])


def test_a_deep_reduction_is_certified_because_it_writes_one_tile():
    """The measurement that reversed this axis's design.

    Sized by OPERANDS, a K=1024-tile reduction looks like the most expensive capsule in the corpus and
    was capped at the loop tier. Sized by WRITTEN OUTPUT -- which is what the cost law was calibrated
    on, and what refitting this target's own certifications confirms (r2 0.924 against 0.226 for the
    largest operand) -- it drains a single result tile and certifies for about the price of the
    shallowest capsule there is. Measured on this target: `PK03_k128` took 161.5 s against 121.1 s for
    the same shape at K=16, so eight times the reduction cost a third more time.

    That is the sweet spot the axis exists to find: maximum accumulation depth at minimum simulation
    cost. Capping it would throw the cycle-accurate guarantee away on the behaviour hardest to get
    right, which is exactly what the wrong metric did."""
    made = _depth_axis(CS.synthesize(_with_depth(_spec("gemmini"))))
    assert made, "the residency depths must be exercised"
    deepest = max(made, key=_passes)
    assert _passes(deepest) >= 512, "the deepest reduction the store admits must be reached"
    for e in made:
        assert not e.get("max_oracle_tier"), (
            f"{e['name']} was capped below the cert tier, but it writes one output tile")
        assert _passes(e) >= 2, "a single-pass K exercises no accumulation at all"


def test_a_capsule_whose_OUTPUT_is_too_large_is_capped_and_names_its_sibling():
    """The other half of the same rule, and the one that keeps the corpus runnable. Depth is cheap;
    PARALLEL EXTENT is not, because that is what the result tile is made of. A ceiling low enough to
    bite must cap and must name what the capped capsule rests on."""
    doc = _with_depth(_spec("gemmini"), ceiling=4)   # below one 16x16 tile, so everything is too big
    made = _depth_axis(CS.synthesize(doc))
    assert made
    for e in made:
        assert e["max_oracle_tier"] == "L2"
        assert e["extends"], "a capped capsule that names nothing rests on nothing"


def test_the_anchor_is_dropped_when_a_regime_depth_already_certifies():
    """A shallower duplicate of a capsule already in the corpus is one more certification buying
    nothing -- and this corpus is paid for in simulator hours."""
    with_regimes = {e["name"] for e in _depth_axis(CS.synthesize(_with_depth(_spec("gemmini"))))}
    assert not any("certified" in n for n in with_regimes), (
        "the anchor duplicates a deeper regime capsule that already certifies")
    # ...but a target whose regimes yield no depth still gets its multi-pass guarantee.
    without = {e["name"] for e in _depth_axis(
        CS.synthesize(_with_depth(_spec("gemmini"), regimes=False)))}
    assert any("certified" in n for n in without), (
        "with no regime depth, nothing else guarantees a multi-pass reduction is certified"
    )


def test_an_unsizable_depth_is_reported_rather_than_dropped():
    """A target that could size no depth at all must say so. Silence here is indistinguishable from
    a target whose accumulator was actually exercised."""
    res = CS.synthesize(_with_depth(_spec("gemmini"), certified=False, regimes=False,
                                    refusal="no measured certification history"))
    assert _depth_axis(res) == []
    holes = " ".join(res["provenance"].get("accumulation_depth_unsizable") or ())
    assert "no measured certification history" in holes
    # ...and NOT in the missing-writer channel, which would send a reader after a builder that exists.
    assert not res["provenance"].get("cells_no_writer_can_express")


def test_a_target_deriving_no_depth_synthesizes_none():
    """Inertness. The axis must not manufacture a depth on a target whose store geometry it could
    not read -- an underivable depth is not a shallow one."""
    doc = dict(_spec("gemmini"))
    mm = dict(doc.get("memory_mapping") or {})
    mm["reduction_depth"] = {"unavailable": "RuntimeError: no operand store"}
    doc["memory_mapping"] = mm
    assert _depth_axis(CS.synthesize(doc)) == []
