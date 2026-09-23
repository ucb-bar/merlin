"""The corpus must be able to ASK for the work that is the host-side performance gap.

A capsule corpus fails an agent only for work some capsule demands. On a captured ResNet-50 the
compiler left 116 regions on the host under four op families -- ``quantize`` (50), ``minmax`` (49),
``elementwise`` (16) and ``pool`` (1) -- and for a long time none of the four could be demanded at all,
for two compounding reasons:

* the TAXONOMY could not name three of them, so the region resolved to no family, was refused as
  ``undeclared_family``, never reached a conformance cell, and no capsule could be synthesized for it;
* the BUILDER vocabulary could not write the fourth on an integer target, because the only writer that
  knew ``add`` was the PyTorch one and the generator refuses that outside the float regime.

Nothing here is a list of the four. Every requirement below is DERIVED from committed vocabularies --
the ABI contract's own ``host_refusal_evidence`` map, and the capture-tag tables ``op_profile`` keeps --
because a literal list is precisely how this bug survived its own warning comment once already:
``maxpool2d`` was added to the taxonomy under a note explaining that a missing pooling spelling had
sent every pooling capsule to ``family = None``, and the snake_case ``max_pool2d`` from a second
capture's vocabulary was still not added.
"""

from __future__ import annotations

import pytest

from merlin.runtime import route_quality as RQ
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth as CSY
from merlin.targetgen import semantic_families as SF

# ---------------------------------------------------------------------------------------------------
# the vocabularies, read rather than restated
# ---------------------------------------------------------------------------------------------------


def _host_refusal_evidence() -> dict:
    rows, cause = RQ.host_refusal_evidence()
    assert rows, (
        f"the backend ABI contract declares no admission.route_quality.host_refusal_evidence "
        f"({cause}). That map is what says WHICH device rung adjudicates each host-refused op family, "
        f"and without it nothing here can state what the corpus must be able to demand"
    )
    return rows


def _capture_family_tags() -> dict[str, str]:
    """The ``prov.family`` tags the study models' own captures emit, from the table whose docstring
    says so. Read, not copied: a tag added there must resolve here without a second edit."""
    from merlin.llvmlower.op_profile import FAMILY_CATEGORY

    return dict(FAMILY_CATEGORY)


def _capture_op_tags() -> dict[str, str]:
    """The ``prov.op`` / ``prov.role`` spellings the captures emit, from the same module's own
    override tables. ``FAMILY_CATEGORY`` is already cross-held against the taxonomy; these two are the
    OP-level half of the same vocabulary and were not, which is the half the pooling spelling lived
    in."""
    from merlin.llvmlower.op_profile import OP_CATEGORY, ROLE_CATEGORY

    return {**OP_CATEGORY, **ROLE_CATEGORY}


# ---------------------------------------------------------------------------------------------------
# 1. every family the contract adjudicates can be NAMED
# ---------------------------------------------------------------------------------------------------


def test_every_host_refused_family_resolves_to_a_canonical_family() -> None:
    """A family the ABI contract knows how to adjudicate is a family the taxonomy must be able to name.
    An unnameable one is refused as ``undeclared_family`` -- a sentence that reads as a claim about the
    hardware and is a claim about our vocabulary."""
    unnameable = sorted(fam for fam in _host_refusal_evidence() if SF.from_prov(fam) is None)
    assert not unnameable, (
        f"the ABI contract declares device evidence for host-refused family/families {unnameable}, "
        f"and semantic_families cannot name them. A region carrying one resolves to no family, is "
        f"refused as undeclared_family, and never reaches a conformance cell -- so no requirement "
        f"demands it and no capsule can be synthesized for it"
    )


def test_every_capture_op_spelling_resolves() -> None:
    """The OP-level half of the capture vocabulary. ``op_profile`` classifies each of these spellings
    for cost; the taxonomy must classify the same spelling for capability, or the two tables disagree
    about the same word again."""
    unnameable = sorted(op for op in _capture_op_tags() if SF.from_op(op) is None)
    assert not unnameable, (
        f"prov.op/prov.role spelling(s) {unnameable} are categorised by op_profile but resolve to no "
        f"semantic family. Map them in semantic_families._OP_FAMILY -- a spelling that resolves in one "
        f"table and not the other is how the same region gets a cost and no capability"
    )


# ---------------------------------------------------------------------------------------------------
# 2. A SPELLING IS NOT A CAPABILITY. The recurrence guard.
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("table_name", ["_OP_FAMILY", "_PROV_FAMILY"])
def test_punctuating_a_declared_spelling_differently_still_resolves(table_name: str) -> None:
    """THE MUTATION FOR THE TAXONOMY LOOKUP. Every declared key, re-punctuated the way another
    capture's vocabulary would punctuate it, must resolve to the SAME family.

    This is the assertion that a literal key list cannot make. ``maxpool2d`` and ``max_pool2d`` are one
    window under two spellings; so are ``quantize_per_tensor`` and ``quantizepertensor``. Listing both
    is what was tried, and the next capture punctuated something else.
    """
    table = getattr(SF, table_name)
    lookup = SF.from_op if table_name == "_OP_FAMILY" else SF.from_prov
    for key, family in table.items():
        folded = SF.fold_spelling(key)
        underscored = "_".join(part for part in key.split("_") if part)
        for spelling in {folded, underscored, key.upper(), f" {key} "}:
            assert lookup(spelling) == family, (
                f"{key!r} resolves to {family!r} but its re-punctuation {spelling!r} resolves to "
                f"{lookup(spelling)!r}. One capability under two spellings is one capability"
            )


def test_a_deliberately_unmapped_tag_is_not_resolved_by_the_fold() -> None:
    """The fold must not hand a family to a tag the taxonomy declares has none. ``spectral`` is
    declared unmapped with a reason; folding it must not quietly borrow a neighbour's family."""
    for tag in SF.PROV_FAMILY_UNMAPPED:
        assert SF.from_prov(tag) is None
        assert SF.from_prov(SF.fold_spelling(tag)) is None


def test_the_taxonomy_declares_no_folded_collision() -> None:
    """Two declared keys that fold together while naming DIFFERENT families would make one resolve to
    the other's capability -- worse than not resolving. ``check()`` refuses that."""
    assert SF.check() == []


# ---------------------------------------------------------------------------------------------------
# 3. every family the contract adjudicates can be BUILT INTO A CAPSULE
# ---------------------------------------------------------------------------------------------------


def _demand_route(family: str, row: dict) -> tuple[str, str]:
    """``(kind, what)`` for the capsule route the contract declares for ``family``.

    ``epilogue_stage`` -- the device runs this family on its READOUT path, so the only capsule that can
    demand it is a contraction carrying that stage; a standalone capsule would declare a capability no
    target here evidences and the eligibility oracle would refuse it as a false fallback.
    ``facet_field`` -- the device runs it as its own region, so it earns a standalone builder.
    """
    if row.get("epilogue_stage"):
        return "epilogue_stage", str(row["epilogue_stage"])
    if row.get("facet_field"):
        return "facet_field", str(row["facet_field"])
    raise AssertionError(f"host_refusal_evidence for {family!r} declares neither rung: {row!r}")


def test_a_capsule_can_be_written_for_every_host_refused_family() -> None:
    """THE GATE. For each family the contract adjudicates, the corpus vocabulary must contain the
    capsule that demands it -- the epilogue stage a contraction builder can carry, or a standalone
    builder of that family. A family with neither is work no capsule can ask for, which is work no
    backend can be failed for leaving on the host.

    THE MUTATION THAT FIRES: remove the stage from ``BUILDER_EPILOGUE_STAGES``, or remove the
    standalone builder from ``BUILDERS``, and the family it covers becomes undemandable here.
    """
    op_families = {op: SF.from_op(op) for op in CS.BUILDERS}
    missing: list[str] = []
    for family, row in sorted(_host_refusal_evidence().items()):
        canonical = SF.from_prov(family)
        kind, what = _demand_route(family, row)
        if kind == "epilogue_stage":
            if what not in CS.BUILDER_EPILOGUE_STAGES:
                missing.append(
                    f"{family}: the contract says a device runs it as the {what!r} readout stage, and "
                    f"no capsule builder can carry that stage onto a contraction"
                )
            elif SF.from_op(what) != canonical:
                missing.append(
                    f"{family}: its readout stage {what!r} resolves to family {SF.from_op(what)!r}, "
                    f"not to {canonical!r}, so a capsule carrying it is credited for a different family"
                )
        else:
            if canonical not in op_families.values():
                missing.append(
                    f"{family}: the contract adjudicates it with the standalone rung {what!r}, and no "
                    f"builder writes a standalone {canonical!r} region"
                )
    assert not missing, "the corpus cannot ask for work the ABI contract says a device can do:\n  " + "\n  ".join(
        missing
    )


def test_the_captures_own_spelling_of_the_standalone_family_is_buildable() -> None:
    """...and it must be buildable under the name the CAPTURE uses, not only under the corpus's own.

    A captured model tags its residual connections ``add``. Before this, ``add`` reached a capsule only
    through the PyTorch writer, which the generator refuses outside the float regime -- so on an integer
    target the spelling a deployable model actually contains had no writer at all.
    """
    assert "add" in CS.BUILDERS
    assert SF.from_op("add") == SF.from_op("residual_add")
    assert CSY.source_for_op("add") is None, "an op with a direct-MLIR builder must not route to torch"
    # ...and it is expressible at an INTEGER dtype, which is the whole point: the PyTorch path is not.
    assert "add" in CSY.ops_gradeable_at("i8", CSY.available_ops())


# ---------------------------------------------------------------------------------------------------
# 4. the synthesizer can produce a COMPLETE entry for the op it chose
# ---------------------------------------------------------------------------------------------------


def _binding(**over):
    kw = dict(
        target="t", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact", requant_output_dtype="int8",
    )  # fmt: skip
    kw.update(over)
    return CS.CorpusBinding(**kw)


def _entry(op: str = "add", **over):
    entry = {
        "name": "SY_elementwise_map_i8_aligned", "kind": "isa", "source_role": "derived_sweep",
        "source_reference": "test", "op": op, "M": 16, "N": 16,
    }  # fmt: skip
    entry.update(over)
    return entry


def test_an_op_the_chooser_may_pick_has_an_entry_the_builder_accepts() -> None:
    """THE MUTATION FOR THE NEW BUILDER. ``build_residual_add`` gives its arithmetic no default -- a
    bound chosen after seeing a result is not a gate -- so an entry that names the op without stating
    it raises. An op the synthesizer may CHOOSE and cannot COMPLETE is a generation failure dressed as
    a covered cell, so the synthesizer declares the parameters, and without that declaration the build
    below fails."""
    bare = _entry()
    with pytest.raises(ValueError):
        CS.build(bare, _binding())
    complete = CSY.declare_residual_add_params(_entry())
    capsule, text = CS.build(complete, _binding())
    assert capsule["operation"]["op"] == "residual_add"
    assert "merlin_iface.residual_add" in text


def test_the_declared_parameters_make_the_capsule_the_plain_sum_of_its_operands() -> None:
    """...and they are not arbitrary: unity multipliers are what make the golden the integer sum, so
    the capsule tests the ADD rather than a scale the synthesizer chose, and a zero bound is what stops
    it admitting a wrong answer."""
    from merlin.runtime import reference, simulator
    from merlin.targetgen import capsule_golden
    from merlin.targetgen.contract import interface_emit as IE
    from merlin.targetgen.contract.schemas import validate_capsule

    entry = CSY.declare_residual_add_params(_entry())
    capsule, text = CS.build(entry, _binding())
    validate_capsule(capsule)
    assert capsule["numeric_policy"]["atol"] == 0
    cb = IE.parse_interface_mlir(text)
    golden = capsule_golden.golden(capsule, None)["Y0"]
    assert golden == reference.reference_outputs(cb)["Y0"] == simulator.simulate(cb)["outputs"]["Y0"]


def test_the_declaration_is_inert_on_every_other_op() -> None:
    """It may not sprinkle residual-add parameters onto entries that have no add in them: the builders
    refuse a parameter read by nothing, which is the same silent-wrong-answer guard pointing the other
    way."""
    other = CSY.declare_residual_add_params(_entry(op="matmul", K=16))
    assert "lhs_scale" not in other and "bound_lsb" not in other


def test_a_synthesized_entry_that_chose_the_add_is_declared_by_the_synthesizer() -> None:
    """The wiring, not just the helper: the entry the synthesizer HANDS to the builder carries the
    parameters. Asserted through the same function the axes call, so removing a call site fails."""
    entry = _entry()
    CSY.declare_pool_window(entry)
    CSY.declare_residual_add_params(entry)
    assert entry["lhs_scale"] == entry["rhs_scale"] == CSY.SYNTH_RESIDUAL_SCALE
    assert entry["bound_lsb"] == CSY.SYNTH_RESIDUAL_BOUND_LSB
