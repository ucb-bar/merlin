"""A model's quantization, and which layers get it, follow the target — and a second target differs.

Two properties are gated here.

**Zero core lines.** A second accelerator arrives as its own facts (RTL-extracted), its own
capability contract and its own operand formats. Nothing below edits a dispatch table, a registry
or a shared code path: both targets go through the same public entry points, and the derived
quantization tracks whichever one it was asked about. If adding a target ever needs an edit to
``quant_recipe`` or ``quant_layer_plan``, one of these assertions is the thing that should fail.

**The mutation fires.** A gate that would pass whichever way the hardware came out is not a gate.
So the granularity is perturbed at the FACET — the hardware rung, not the recipe JSON — and the
test pins the value on both sides of the perturbation: what it was, and what it became. A recipe
that ignored the change, or a layer plan that placed the same layers either way, fails here.

The layer plan is deliberately framework-free, so all of this runs in the ordinary test
interpreter. The realisation of a plan as TorchAO's own ``AOBaseConfig`` needs torch and is
exercised in the capture venv (``test_quant_recipe.py``); what must never depend on that venv is
the decision itself, because a test that skips is a test that passes either way.
"""

from __future__ import annotations

import pytest

from merlin.common import quant_formats as qf
from merlin.targetgen import quant_layer_plan as QLP
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import readout_facet as RF

# --- two targets, each described only by its own hardware -----------------------------------------

#: A scalar readout contract a backend derives from its own generated header.
_ABI = {
    "schema": RF.SCALAR_ABI_SCHEMA,
    "accumulator_dtype": "i32",
    "output_dtype": "i8",
    "scale_dtype": "f32",
    "clamp_min": -128,
    "clamp_max": 127,
    "provenance": {"scope": "test"},
}
_READOUTS = [{"selector": "i8", "applies": ["acc_scale", "relu", "bias_add"]}]


def _register_facts(*, scale_fields: tuple[str, ...] = ("out_scale",)) -> dict:
    """RTL facts for a unit whose scales are command-register fields — one value per command."""
    return {
        "facts": {
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
            "memories": [{"name": "accumulator", "bytes": 4096}],
            "interfaces": [
                {
                    "name": "register_bundle_layouts",
                    "unresolved": {},
                    "bundles": {
                        "StoreConfig": {
                            "width": 64,
                            "fields": {
                                name: {"offset": 32, "width": None, "width_param": "scale_bits", "slot_width": 32}
                                for name in scale_fields
                            },
                        }
                    },
                }
            ],
        }
    }


def _mesh_target(name: str = "mesh_alpha") -> RF.ReadoutFacet:
    """Target A: an integer mesh whose store path holds one scale per command."""
    return RF.derive(
        name,
        facts=_register_facts(),
        unit={"name": "mac_array", "dtypes": ["int8"]},
        scalar_abi=_ABI,
        readouts=_READOUTS,
    )


def _block_format() -> qf.QuantFormat:
    return next(f for _, f in sorted(qf.registry().items()) if f.kind == "mx_block" and f.element_bits == 8)


def _block_target(name: str = "blockunit_beta") -> RF.ReadoutFacet:
    """Target B: a unit whose OPERAND FORMAT is block-scaled, so the format fixes the granularity."""
    return RF.derive(name, unit={"name": "block_mac", "dtypes": [_block_format().name]})


#: One model, described the way :func:`_recipe_quantizer.layer_inventory` describes a real one.
#: ``narrow`` exists to be a layer whose reduced extent is not a whole number of blocks.
_LAYERS = (
    {"fqn": "encoder.proj", "kind": "Linear", "weight_shape": [64, 128]},
    {"fqn": "encoder.narrow", "kind": "Linear", "weight_shape": [16, 24]},
    {"fqn": "stem.conv", "kind": "Conv2d", "weight_shape": [32, 3, 3, 3]},
    {"fqn": "head.pool", "kind": "AdaptiveAvgPool2d", "weight_shape": None},
    {"fqn": "stem.act", "kind": "ReLU", "weight_shape": None},
)


def _placements(recipe) -> dict[str, tuple[str, str | None]]:
    plan = QLP.plan(recipe.to_dict(), _LAYERS)
    return {d.fqn: (d.placement, d.refusal) for d in plan.decisions}


# --- zero core lines ------------------------------------------------------------------------------


def test_a_second_target_derives_its_own_quantization_with_no_shared_edit() -> None:
    """Two targets, two hardware descriptions, two different derived numeric forms."""
    mesh, block = QR.derive(_mesh_target()), QR.derive(_block_target())
    assert mesh.status == QR.DERIVED and block.status == QR.DERIVED

    # The recipes disagree about the thing the hardware disagrees about, and say so in their own
    # words -- neither is a default, and neither was written down for a named target.
    assert mesh.weight.granularity == "tensor", "a register-held scale is one value per command"
    assert block.weight.granularity == "block", "a block-scaled format fixes its own granularity"
    assert block.weight.block == _block_format().scale.block
    assert mesh.weight.dtype == "int8" and block.weight.dtype == _block_format().name
    assert (mesh.activation.mode, block.activation.mode) == ("static", "dynamic")
    assert mesh.to_dict()["recipe_sha256"] != block.to_dict()["recipe_sha256"]

    # Each recipe names the target it came from, so a plan can never be attributed to the other.
    assert (mesh.target, block.target) == ("mesh_alpha", "blockunit_beta")


def test_the_same_model_is_planned_differently_for_each_target() -> None:
    """The per-layer decision tracks the target; one model, two plans, for derived reasons."""
    mesh_plan = QLP.plan(QR.derive(_mesh_target()).to_dict(), _LAYERS)
    block_plan = QLP.plan(QR.derive(_block_target()).to_dict(), _LAYERS)
    assert mesh_plan.to_dict()["plan_sha256"] != block_plan.to_dict()["plan_sha256"]

    mesh, block = _placements(QR.derive(_mesh_target())), _placements(QR.derive(_block_target()))

    # A per-tensor scale covers any shape, so every contraction is in the mesh's numeric form.
    assert mesh["encoder.proj"] == (QLP.DEVICE, None)
    assert mesh["encoder.narrow"] == (QLP.DEVICE, None)
    assert mesh["stem.conv"] == (QLP.DEVICE, None)

    # The block unit takes the same layers only where its block tiles their reduced extent.
    assert block["encoder.proj"] == (QLP.DEVICE, None), "128 is a whole number of 32-element blocks"
    assert block["encoder.narrow"] == (QLP.HOST, QLP.GRANULARITY_DOES_NOT_TILE)
    why = next(d.why for d in block_plan.decisions if d.fqn == "encoder.narrow")
    assert "24" in why and str(_block_format().scale.block) in why

    # A pooling layer rides on a readout only where the recipe lists the family; the mesh's static
    # per-tensor scale can carry 1/count, and the block unit's per-block dynamic one cannot.
    assert mesh["head.pool"] == (QLP.DEVICE, None)
    assert block["head.pool"] == (QLP.HOST, QLP.FAMILY_NOT_ABSORBED)

    # And neither target calls an activation host arithmetic: it stores nothing to scale, so it is
    # no quantization site of its own -- a different finding from a refusal, under its own code.
    assert mesh["stem.act"] == (QLP.HOST, QLP.NO_QUANTIZATION_SITE)
    assert block["stem.act"] == (QLP.HOST, QLP.NO_QUANTIZATION_SITE)


def test_an_underivable_target_quantizes_nothing_and_says_which_field(monkeypatch) -> None:
    """Fail closed: no readout, no recipe, no quantized layer -- and the reason survives to the plan."""
    # A readout whose scale field the reader never placed: the granularity is unknown, not per-tensor.
    facet = RF.derive(
        "unread_gamma",
        facts=_register_facts(scale_fields=("out_gain",)),
        unit={"name": "mac_array", "dtypes": ["int8"]},
        scalar_abi=_ABI,
        readouts=_READOUTS,
    )
    recipe = QR.derive(facet)
    assert recipe.status == QR.UNDERIVABLE and recipe.weight is None

    plan = QLP.plan(recipe.to_dict(), _LAYERS)
    assert plan.on_device == (), "an underivable recipe must quantize nothing at all"
    assert set(plan.refusals()) == {QLP.RECIPE_UNDERIVABLE}
    assert "scale role" in plan.unknown["recipe"]
    # every layer carries the hardware reason, not a generic one
    assert all("scale role" in d.why for d in plan.decisions)


# --- the mutation fires ---------------------------------------------------------------------------


def test_perturbing_the_derived_granularity_changes_the_configuration() -> None:
    """Change what the READOUT holds; the recipe and the per-layer plan must both follow."""
    facet = _mesh_target()
    before = QR.derive(facet)
    assert before.weight.granularity == "tensor", "the unperturbed derivation is per-tensor"
    before_plan = QLP.plan(before.to_dict(), _LAYERS).to_dict()

    # THE MUTATION: the same hardware, described as holding one scale per accumulator column.
    facet.scale_granularities = ("tensor", "column")
    after = QR.derive(facet)

    assert after.weight.granularity == "channel", "a readout that holds a finer scale must be used"
    assert after.to_dict()["recipe_sha256"] != before.to_dict()["recipe_sha256"]
    # The recipe's own account of itself moved with it, so the change is reviewable.
    assert "per column" in after.why["weight.granularity"]

    after_plan = QLP.plan(after.to_dict(), _LAYERS).to_dict()
    changed = [
        (b["fqn"], b["why"], a["why"])
        for b, a in zip(before_plan["layers"], after_plan["layers"])
        if b["why"] != a["why"]
    ]
    assert changed, "the plan repeated itself under a different granularity: it is not deriving"
    # Each contraction's stated reason moved from "one scale covers the whole weight" to one that
    # names the output-channel axis it now indexes.
    proj_before, proj_after = next((b, a) for fqn, b, a in changed if fqn == "encoder.proj")
    assert "whole weight" in proj_before and "output channel" in proj_after


def test_a_granularity_the_layer_cannot_tile_is_refused_not_rounded() -> None:
    """The finer granularity must be able to REFUSE, or 'it changed' proves nothing."""
    recipe = QR.derive(_mesh_target()).to_dict()
    assert all(d.placement == QLP.DEVICE for d in QLP.plan(recipe, _LAYERS).decisions if d.family == QLP.CONTRACTION)

    # THE MUTATION: one scale per 32-element block. Two of the three contractions stop fitting.
    blocked = {**recipe, "weight": {**recipe["weight"], "granularity": "block", "block": 32}}
    decisions = {d.fqn: d for d in QLP.plan(blocked, _LAYERS).decisions}
    assert decisions["encoder.proj"].placement == QLP.DEVICE, "128 reduces to 4 whole blocks"
    for fqn, extent in (("encoder.narrow", 24), ("stem.conv", 27)):
        assert decisions[fqn].placement == QLP.HOST
        assert decisions[fqn].refusal == QLP.GRANULARITY_DOES_NOT_TILE
        assert str(extent) in decisions[fqn].why, "the refusal states the extent it measured"

    # A convolution reduces over ALL the axes behind its output one (3x3x3 = 27), not over the last.
    assert "3x3x3" in decisions["stem.conv"].why


def test_a_plan_digest_is_of_the_placements_and_not_of_the_prose() -> None:
    """The digest a capture keys on must move for a placement and stay for a rewording."""
    recipe = QR.derive(_mesh_target()).to_dict()
    plan = QLP.plan(recipe, _LAYERS).to_dict()

    reworded = {**plan, "layers": [{**layer, "why": "reworded"} for layer in plan["layers"]]}
    assert QLP.digest(reworded) == plan["plan_sha256"]

    moved = {**plan, "layers": [{**layer, "placement": QLP.HOST} for layer in plan["layers"]]}
    assert QLP.digest(moved) != plan["plan_sha256"]


def test_every_declared_target_plans_the_same_model_with_no_code_of_its_own() -> None:
    """The roster, on real hardware descriptions: one model, one code path, a plan for each target.

    What each target derives depends on what has been extracted on this machine, so the expected
    granularity is not pinned here — the invariants are. A target that derives a recipe places
    contractions; one that does not places nothing and says which fact was missing. Two targets
    that derive DIFFERENT numeric forms must not produce the same plan, which is the property that
    fails the moment a shared code path starts deciding for everyone.
    """
    from merlin.common.paths import merlin_dir
    from merlin.targetgen.target_experiment import load_capability_manifest

    seen: dict[str, tuple[str, str | None]] = {}
    examined = 0
    for contract in sorted((merlin_dir() / "targets").glob("*/contracts/target_contract.yaml")):
        target = contract.parent.parent.name
        try:
            load_capability_manifest(target)
            recipe = QR.select(QR.for_target(target))
        except Exception:  # noqa: BLE001 -- a manifest that does not load is another test's
            continue
        if recipe is None:
            continue
        body = recipe.to_dict()
        plan = QLP.plan(body, _LAYERS)
        examined += 1

        assert [d.fqn for d in plan.decisions] == [layer["fqn"] for layer in _LAYERS], target
        assert all(d.refusal and d.why for d in plan.on_host), f"{target}: a host layer states no reason"

        placed = [d for d in plan.on_device if d.family == QLP.CONTRACTION]
        if body["status"] == QR.DERIVED:
            assert placed, f"{target} derives {body['weight']} yet places no contraction"
            seen[target] = (body["weight"]["granularity"], plan.to_dict()["plan_sha256"])
        else:
            assert not plan.on_device, f"{target} is underivable yet quantized something"
            assert set(plan.refusals()) == {QLP.RECIPE_UNDERIVABLE}, target
            assert body["underivable"], f"{target} is underivable with no reason recorded"

    assert examined >= 3, f"the roster went vacuous: only {examined} target(s) reached a plan"

    # Two real targets whose readouts hold different granularities must plan differently. Which
    # targets derive depends on what this machine has extracted, so this bites when it applies and
    # the synthetic pair above is the always-on gate.
    granularities = {granularity for granularity, _digest in seen.values()}
    digests = {digest for _granularity, digest in seen.values()}
    assert len(digests) >= len(granularities), f"targets deriving {granularities} produced one plan: {seen}"


@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_every_contraction_the_recipe_covers_is_decided_never_skipped(granularity: str) -> None:
    """No layer is silently absent: a plan accounts for every layer it was given, either way."""
    recipe = QR.derive(_mesh_target()).to_dict()
    recipe = {**recipe, "weight": {**recipe["weight"], "granularity": granularity}}
    plan = QLP.plan(recipe, _LAYERS)
    assert [d.fqn for d in plan.decisions] == [layer["fqn"] for layer in _LAYERS]
    assert all(d.placement in (QLP.DEVICE, QLP.HOST) for d in plan.decisions)
    assert all(d.refusal and d.why for d in plan.on_host), "a host layer always states its reason"
