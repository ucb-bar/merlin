"""A contraction and its residual add as TWO regions, and the naming gap that made them unrefusable.

The capsule this builds exists to ask a question every other fused-family capsule answers for the
compiler: a normal epilogue capsule hands over an interface with the stage already written into the
contraction's ``epilogue`` attribute, so the backend is TOLD to fuse. Here the two regions are
separate and the placement is the backend's own decision, which is the decision a deployable
compiler was measured getting wrong.

The second half of this file is about why that decision could never be graded before. A capture tags
each region with a coarse ``prov.family``; three of the four tags a captured ResNet-50's host regions
carry (``quantize``, ``minmax``, ``pool``) were absent from the taxonomy, so the region resolved to
no family at all and the capability check refused it as ``undeclared_family`` -- indistinguishable,
in any aggregate, from work the hardware genuinely cannot do.
"""

from __future__ import annotations

import copy

import pytest

from merlin.runtime import reference, simulator
from merlin.runtime.commandbuffer import STIMULUS_RANGE_KEY
from merlin.targetgen import capsule_golden
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import eligibility as E
from merlin.targetgen import semantic_families as SF
from merlin.targetgen.contract import interface_emit as IE
from merlin.targetgen.contract.schemas import validate_capsule

#: The four (prov.family, prov.op) pairs a captured ResNet-50's contradicted host regions carry, with
#: the region count the census measured for each. Named as DATA because every assertion below is over
#: all four -- a fix that closes three of them is a fix that leaves the fourth silently unnameable.
CENSUS = (
    ("quantize", "quantize_per_tensor", 50),
    ("minmax", "minmax", 49),
    ("elementwise", "add", 16),
    ("pool", "max_pool2d", 1),
)


def _binding(**over):
    kw = dict(
        target="t", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact", requant_output_dtype="int8",
    )  # fmt: skip
    kw.update(over)
    return CS.CorpusBinding(**kw)


def _entry(**over):
    entry = {
        "name": "T_seam", "kind": "layer", "source_role": "derived_sweep", "source_reference": "test",
        "op": "residual_seam", "M": 16, "K": 32, "N": 16, "epilogue": ["requant"], "requant_shift": 8,
        "lhs_scale": 1.0, "rhs_scale": 1.0, "bound_lsb": 0, "add_epilogue": ["relu"],
    }  # fmt: skip
    entry.update(over)
    return entry


def _built(binding=None, **over):
    capsule, text = CS.build(_entry(**over), binding or _binding())
    capsule[STIMULUS_RANGE_KEY] = [-128, 127]
    cb = IE.parse_interface_mlir(text)
    cb.setdefault("params", {})[STIMULUS_RANGE_KEY] = [-128, 127]
    return capsule, cb


# ---------------------------------------------------------------------------------------------------
# the capsule itself
# ---------------------------------------------------------------------------------------------------


def test_the_capsule_validates_and_its_golden_is_reproducible() -> None:
    capsule, cb = _built()
    validate_capsule(capsule)
    golden = capsule_golden.golden(capsule, None)["Y0"]
    assert golden == reference.reference_outputs(cb)["Y0"] == simulator.simulate(cb)["outputs"]["Y0"]
    # Recomputing it changes nothing: the stimulus is derived from the declared names, not sampled.
    assert golden == capsule_golden.golden(capsule, None)["Y0"]
    values = [v for row in golden for v in row]
    assert min(values) == 0 and max(values) == 127  # the relu and the saturation both did something


def test_the_two_regions_stay_two_regions() -> None:
    """The point of the capsule. If the builder emitted one fused command there would be nothing for a
    backend to decide, and the capsule would test the emitter instead of the placement."""
    _capsule, cb = _built()
    opcodes = [c["opcode"] for c in cb["commands"]]
    assert opcodes == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "RESIDUAL_ADD", "EVICT"]


def test_the_add_is_a_separate_region_over_the_contraction_s_own_result() -> None:
    """...and it consumes the COMMIT's output, not a third leaf. A seam whose add reads an independent
    input is not a seam -- the two regions would be unrelated work that happens to be adjacent."""
    _capsule, cb = _built()
    commit = next(c for c in cb["commands"] if c["opcode"] == "COMMIT")
    add = next(c for c in cb["commands"] if c["opcode"] == "RESIDUAL_ADD")
    assert add["operands"]["lhs"] == commit["operands"]["dst"] == "Yc"
    assert add["operands"]["rhs"] == "X1"  # the residual leaf, in the operand dtype


@pytest.mark.parametrize("missing", ["lhs_scale", "rhs_scale", "bound_lsb"])
def test_nothing_about_the_add_s_arithmetic_has_a_default(missing) -> None:
    with pytest.raises(ValueError, match=missing):
        CS.build(_entry(**{missing: None}), _binding())


def test_a_readout_that_does_not_narrow_is_refused_naming_the_stage() -> None:
    """The add's operands have to meet in one domain. Without the narrowing readout the capsule would
    still build -- and its golden would add an i32 accumulator to an i8 leaf, which is not the
    operation the entry named."""
    with pytest.raises(ValueError, match="meet in one domain"):
        CS.build(_entry(epilogue=[], requant_shift=None), _binding())


def test_a_scale_with_no_stage_and_a_stage_with_no_scale_are_both_refused() -> None:
    with pytest.raises(ValueError, match="acc_scale"):
        CS.build(_entry(epilogue=["acc_scale"], requant_shift=None), _binding())
    with pytest.raises(ValueError, match="read by nothing"):
        CS.build(_entry(acc_scale=0.0625), _binding())


def test_a_stage_the_add_cannot_carry_is_refused() -> None:
    with pytest.raises(ValueError, match="carries only"):
        CS.build(_entry(add_epilogue=["acc_scale"]), _binding())


def test_the_declared_bound_is_the_comparison() -> None:
    capsule, _cb = _built(bound_lsb=2)
    assert capsule["numeric_policy"] == {"compare": "bounded_int", "dtype": "i8", "atol": 2, "rtol": 0}
    golden = capsule_golden.golden(capsule, None)["Y0"]
    off = copy.deepcopy(golden)
    off[0][0] = min(127, off[0][0] + 2)
    policy = capsule["numeric_policy"]
    assert capsule_golden.compare({"Y0": golden}, {"Y0": off}, policy)["status"] == "pass"
    assert capsule_golden.compare({"Y0": golden}, {"Y0": off}, {**policy, "atol": 0})["status"] == "fail"


def test_the_op_is_in_the_grammar_and_its_family_is_the_contraction_it_contains() -> None:
    assert "residual_seam" in CS.BUILDERS
    assert {"matmul", "commit", "residual_add"} <= IE.defined_mnemonics()
    # Its PRIMARY family is the contraction. Claiming the add's family here would count the seam as
    # evidence for a standalone elementwise capability that this target does not declare.
    assert SF.from_op("residual_seam") == "contraction"


# ---------------------------------------------------------------------------------------------------
# zero core lines: a second target
# ---------------------------------------------------------------------------------------------------


def test_a_second_target_needs_no_edit_to_the_builder() -> None:
    """A different array edge, a different name and a different SPELLING of the dtypes, through the
    same builder. Every extent and dtype below is read from the binding, so a second target arrives as
    a descriptor rather than a patch -- no literal shape, name or dtype token in the control flow."""
    other = _binding(target="u", tile_dim=8, operand_dtype="i8", accum_dtype="i32",
                     requant_output_dtype="i8")  # fmt: skip
    capsule, cb = _built(other, M=8, K=16, N=8)
    validate_capsule(capsule)
    assert [s["shape"] for s in capsule["inputs"]] == [[16, 8], [8, 16], [8, 8]]
    assert capsule_golden.golden(capsule, None)["Y0"] == reference.reference_outputs(cb)["Y0"]
    assert 'merlin_iface.target = "u"' in CS.build(_entry(M=8, K=16, N=8), other)[1]
    # The shapes really did follow the binding rather than the entry's defaults.
    first, _cb = _built()
    assert [s["shape"] for s in first["inputs"]] == [[32, 16], [16, 32], [16, 16]]


def test_the_extents_default_to_the_second_target_s_own_tile() -> None:
    """An entry that declares no extents at all is sized by the binding's tile, so the same profile
    entry produces a legal capsule on both targets -- the point of tile-relative authoring."""
    entry = {k: v for k, v in _entry().items() if k not in ("M", "K", "N")}
    for tile in (8, 16, 32):
        capsule, _text = CS.build(entry, _binding(tile_dim=tile))
        assert [s["shape"] for s in capsule["inputs"]] == [[tile, tile], [tile, tile], [tile, tile]]


# ---------------------------------------------------------------------------------------------------
# THE MUTATION. A backend that leaves one of these families on the host must be refusable.
# ---------------------------------------------------------------------------------------------------


def test_every_census_family_resolves_to_a_canonical_family() -> None:
    """The naming gap itself. Before this, three of these four returned None from both tables."""
    for prov_family, op, _n in CENSUS:
        assert SF.from_prov(prov_family, op) is not None, f"prov.family {prov_family!r} unnameable"
        assert SF.from_op(op) is not None, f"prov.op {op!r} unnameable"
        # And the two agree -- a tag that resolved one way through the family and another through the
        # op would place the same region in two different capabilities depending on the caller.
        assert SF.from_prov(prov_family, op) == SF.from_op(op)


def test_the_two_capture_vocabularies_do_not_drift_apart() -> None:
    """``llvmlower.op_profile.FAMILY_CATEGORY`` says in its own docstring that its keys are the families
    the study models' captures emit. ``semantic_families._PROV_FAMILY`` answers a different question
    about the same words. They were grown independently and drifted: the first knew ``quantize``,
    ``minmax`` and ``pool`` and the second did not, so a region the cost tool could categorise was a
    region the eligibility oracle could not name. Holding them together is what stops the next tag from
    resolving in one and not the other."""
    from merlin.llvmlower.op_profile import FAMILY_CATEGORY

    unnameable = sorted(
        tag for tag in FAMILY_CATEGORY if SF.from_prov(tag) is None and tag not in SF.PROV_FAMILY_UNMAPPED
    )
    assert not unnameable, (
        f"prov.family tag(s) {unnameable} are categorised by op_profile but resolve to no semantic "
        f"family, so a region carrying one is refused as undeclared rather than judged. Either map it "
        f"or declare it in semantic_families.PROV_FAMILY_UNMAPPED with the reason it has no primitive"
    )
    # An exemption has to state a reason, so the list cannot be used to silence the check.
    assert all(str(why).strip() for why in SF.PROV_FAMILY_UNMAPPED.values())
    # ...and it may not be used to exempt a tag that DOES resolve, which would be a stale entry
    # claiming an unmapped state the table has since left.
    assert not [t for t in SF.PROV_FAMILY_UNMAPPED if SF.from_prov(t) is not None]


def test_a_capture_s_raw_tag_is_canonicalized_rather_than_trusted() -> None:
    """A census passes ``prov.family`` verbatim. Returning it unchanged sent a nameable region to the
    capability check under a word no capability uses, where it was refused as undeclared."""
    for prov_family, op, _n in CENSUS:
        raw = E.RegionDescriptor(source="c", op=op, family=prov_family, in_dtype="int8", rank=2)
        assert raw.resolved_family() == SF.from_op(op)
    # An already-canonical family is returned untouched, so no existing caller changes...
    canon = E.RegionDescriptor(source="c", op="add", family="elementwise_map", in_dtype="int8", rank=2)
    assert canon.resolved_family() == "elementwise_map"
    # ...and a tag neither table knows is PRESERVED, so the refusal still names what the capture said.
    unknown = E.RegionDescriptor(source="c", op="???", family="not_a_family", in_dtype="int8", rank=2)
    assert unknown.resolved_family() == "not_a_family"


def test_the_refusal_changes_from_unnameable_to_the_honest_one(census_caps) -> None:
    """THE MUTATION THAT FIRES, half one: the verdict for every census family is now a statement about
    the HARDWARE (``fused_only`` -- this target's manifest declares the family reachable only in
    composition) rather than about the vocabulary (``undeclared_family``). The distinction is the
    whole point: a fused-only family is coverable by a capsule, an unnameable one is not."""
    cap_map, undetermined = census_caps
    for prov_family, op, _n in CENSUS:
        region = E.RegionDescriptor(source="c", op=op, family=prov_family, in_dtype="int8", rank=2)
        verdict = E.is_eligible(region, cap_map, undetermined=undetermined)
        assert verdict.refusal == "fused_only", f"{op}: {verdict.refusal} -- {verdict.reason}"
        assert verdict.family in cap_map


def test_a_region_fused_with_the_seam_s_contraction_is_eligible_and_so_refusable(census_caps) -> None:
    """THE MUTATION THAT FIRES, half two, and the reason the seam capsule exists.

    The seam supplies exactly what these families need to be legal: a contraction to compose with.
    With it the verdict is ELIGIBLE -- and an eligible region routed to the host is precisely what
    ``coverage_report`` counts as a false fallback and what the capsule runner turns into
    ``FALLBACK_ON_ELIGIBLE_REGION``. So a backend that computes the seam's add on the host produces
    the right numbers and still fails, which is what it means for the capsule to DEMAND the work.

    Without both halves this assertion is decoration: an ineligible region left on the host is a
    legitimate placement, so the gate cannot fire, however loudly the capsule declares itself.
    """
    cap_map, undetermined = census_caps
    for prov_family, op, _n in CENSUS:
        region = E.RegionDescriptor(source="c", op=op, family=prov_family, in_dtype="int8", rank=2)
        verdict = E.is_eligible(region, cap_map, undetermined=undetermined, fused_with=frozenset({"contraction"}))
        assert verdict.eligible, f"{op}: {verdict.refusal} -- {verdict.reason}"


def test_the_seam_capsule_is_the_composition_those_regions_need(census_caps) -> None:
    """...and the seam really is a contraction, by the same table the oracle just used. A capsule whose
    own family were the elementwise map would not supply the composition it is meant to license."""
    cap_map, _undetermined = census_caps
    capsule, _cb = _built()
    assert capsule["semantic"]["semantic_family"] == "contraction"
    assert "contraction" in cap_map


@pytest.fixture(scope="module")
def census_caps():
    """The capability map of a target that declares these families fused-only, read from its own
    contract. Skipped rather than faked when the target's contract is not resolvable in this tree:
    a hand-built cap_map would be this test asserting against its own fixture."""
    target = "gemmini"
    try:
        cap_map = E.capability_map_for_target(target)
        undetermined = E.undetermined_families_for_target(target)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no resolvable capability map for {target!r}: {exc}")
    composed = {f for f, c in cap_map.items() if getattr(c, "composed_with", ())}
    if not composed:
        pytest.skip(f"{target!r} declares no fused-only family, so there is no refusal to observe")
    return cap_map, undetermined
