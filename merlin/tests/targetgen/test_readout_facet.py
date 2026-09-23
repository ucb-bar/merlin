"""A readout's scale granularity is derived from the design, and a contract that promises more is named."""

from __future__ import annotations

import pytest

from merlin.common import quant_formats as qf
from merlin.targetgen import readout_facet as RF
from merlin.targetgen.rtl.circt_introspect import extract_register_bundle_layouts, unresolved_register_bundles

# The Chisel idiom a generator uses for a command register whose field widths are parameters: a
# spacer that subtracts the parameter above a field of that width, so the pair is a fixed slot.
_ISA = """
  val STORE_SCALE_WIDTH = 32
  val STORE_STRIDE_WIDTH = 32
  val MODE_WIDTH = 2
  class StoreConfig(scale_bits: Int, stride_bits: Int) extends Bundle {
    val _spacer1 = UInt((STORE_SCALE_WIDTH - scale_bits).W)
    val out_scale = UInt(scale_bits.W)
    val _spacer0 = UInt((STORE_STRIDE_WIDTH - stride_bits).W)
    val stride = UInt(stride_bits.W)
  }
  class ModeConfig extends Bundle {
    val activation = UInt(MODE_WIDTH.W)
    val pool_size = UInt(MODE_WIDTH.W)
  }
  class Opaque extends Bundle {
    val head = UInt(MODE_WIDTH.W)
    val flag = Bool()
    val tail = UInt(MODE_WIDTH.W)
  }
"""
_ABI = {
    "schema": RF.SCALAR_ABI_SCHEMA,
    "accumulator_dtype": "i32",
    "output_dtype": "i8",
    "scale_dtype": "f32",
    "clamp_min": -128,
    "clamp_max": 127,
    "provenance": {"scope": "synthetic header"},
}
_READOUTS = [{"selector": "i8", "applies": ["acc_scale", "relu", "bias_add"]}, {"selector": "i32", "applies": []}]


def _facts(isa: str = _ISA, *, with_unresolved: bool = True) -> dict:
    record = {"name": "register_bundle_layouts", "bundles": extract_register_bundle_layouts(isa)}
    if with_unresolved:
        record["unresolved"] = unresolved_register_bundles(isa)
    return {
        "facts": {
            "datapaths": [
                {"name": "input", "dtype": "i8", "evidence": "element memory"},
                {"name": "accumulator", "dtype": "i32", "evidence": "accumulator memory"},
            ],
            "memories": [{"name": "accumulator", "bytes": 4096}],
            "interfaces": [record],
        }
    }


def test_a_parametric_register_field_is_placed_and_bounded_not_dropped() -> None:
    layouts = extract_register_bundle_layouts(_ISA)
    assert layouts["StoreConfig"] == {
        "width": 64,
        "parameters": ["scale_bits", "stride_bits"],
        "fields": {
            "stride": {"offset": 0, "width": None, "width_param": "stride_bits", "slot_width": 32},
            "out_scale": {"offset": 32, "width": None, "width_param": "scale_bits", "slot_width": 32},
        },
    }
    assert layouts["ModeConfig"]["fields"]["activation"] == {"offset": 2, "width": 2}


def test_a_field_the_reader_cannot_size_withholds_the_bundle_and_says_so() -> None:
    # The field used to be skipped, which shifted `head` down by its width and reported the shifted
    # offset as derived.
    assert "Opaque" not in extract_register_bundle_layouts(_ISA)
    assert "flag" in unresolved_register_bundles(_ISA)["Opaque"]


def test_one_bounded_scale_register_per_command_is_a_per_tensor_readout() -> None:
    facet = RF.derive("t", facts=_facts(), unit={"name": "u", "dtypes": ["int8"]}, scalar_abi=_ABI, readouts=_READOUTS)
    assert facet.scale_granularities == ("tensor",)
    assert facet.rungs_for("scale_granularities") == ("register_layout",)
    assert (facet.element_dtype, facet.accumulator_dtype, facet.accumulator_kind) == ("i8", "i32", "addressable")
    assert set(facet.register_stages) == {"activation", "pool"}
    assert facet.zero_point_carried is False and not facet.unknown


def test_without_the_scale_field_the_granularity_is_unknown_never_a_default() -> None:
    facet = RF.derive("t", facts=_facts(_ISA.replace("out_scale", "out_gain")))
    assert facet.scale_granularities is None
    assert "no command-register field carries the scale role" in facet.unknown["scale_granularities"]


def test_layouts_from_the_reader_that_dropped_parametric_bundles_decide_nothing() -> None:
    facet = RF.derive("t", facts=_facts(with_unresolved=False))
    # The carrier IS visible here; what the old record cannot support is a claim of absence.
    assert facet.zero_point_carried is None
    dropped = _facts(with_unresolved=False)
    del dropped["facts"]["interfaces"][0]["bundles"]["StoreConfig"]
    facet = RF.derive("t", facts=dropped)
    assert facet.scale_granularities is None
    assert "re-derive" in facet.unknown["scale_granularities"]


def test_a_scale_passed_by_reference_is_not_called_per_tensor() -> None:
    facet = RF.derive("t", facts=_facts(_ISA.replace("out_scale", "scale_addr")))
    assert facet.scale_granularities is None
    assert "by reference" in facet.unknown["scale_granularities"]


def test_a_block_scaled_operand_format_fixes_the_granularity_by_definition() -> None:
    name, fmt = next((n, f) for n, f in sorted(qf.registry().items()) if f.scale.is_block)
    facet = RF.derive("t", facts={}, unit={"name": "u", "dtypes": ["int8", name]})
    assert facet.scale_granularities == ("block",)
    assert (facet.scale_block, facet.scale_dtype) == (fmt.scale.block, fmt.scale.dtype)
    assert facet.rungs_for("scale_granularities") == ("format",)


def test_a_contract_that_promises_more_than_the_readout_holds_is_named() -> None:
    facet = RF.derive("t", facts=_facts(), scalar_abi=_ABI)
    (finding,) = RF.reconcile({"name": "u", "scaling": "per_channel"}, facet)
    assert finding["kind"] == "scaling_exceeds_readout"
    assert (finding["declared_granularity"], finding["derived"]) == ("column", ["tensor"])
    assert RF.reconcile({"name": "u", "scaling": "per_tensor"}, facet) == []
    assert RF.reconcile({"name": "u", "scaling": "none"}, facet) == []
    (unaudited,) = RF.reconcile({"name": "u", "scaling": "per_channel"}, RF.derive("t", facts={}))
    assert unaudited["kind"] == "unaudited_scaling"  # "could not look" is not "cannot"


def test_the_epilogue_capability_is_built_from_derived_limits_or_not_at_all() -> None:
    facet = RF.derive("t", facts=_facts(), unit={"name": "u"}, scalar_abi=_ABI, readouts=_READOUTS)
    capability = RF.epilogue_capability(facet)
    assert capability.scale_granularities == ("per_tensor",)
    assert capability.saturations == ((-128, 127),) and capability.output_dtypes == ("i8",)
    assert ("bias_i32", "scale_f32", "round_to_nearest_even", "clamp", "relu") in (capability.ordered_stage_templates)
    # No header-verified rounding: nothing licenses one, so no capability exists to plan against.
    with pytest.raises(ValueError, match="rounding"):
        RF.epilogue_capability(RF.derive("t", facts=_facts(), readouts=_READOUTS))


def test_the_scalar_contract_reaches_a_numerics_contract_builder_unchanged() -> None:
    facet = RF.derive("t", facts=_facts(), scalar_abi=_ABI)
    assert facet.scalar_readout_facts() == _ABI
    assert RF.derive("t", facts=_facts()).scalar_readout_facts() is None


def test_every_declared_target_derives_a_facet_with_no_code_of_its_own() -> None:
    from merlin.common.paths import merlin_dir
    from merlin.targetgen.target_experiment import load_capability_manifest

    derived = 0
    for contract_path in sorted((merlin_dir() / "targets").glob("*/contracts/target_contract.yaml")):
        target = contract_path.parent.parent.name
        try:
            contract = load_capability_manifest(target).contract
        except Exception:  # noqa: BLE001 -- a target whose manifest does not load is another test's
            continue
        for unit, facet in zip(contract.get("compute_units") or (), RF.for_target(target, contract=contract)):
            record = facet.to_dict()
            assert record["schema"] == RF.SCHEMA
            # Every field is either derived with evidence or unknown with a reason: never neither.
            for name in ("element_dtype", "accumulator_dtype", "rounding"):
                assert (record[name] is None) == (name in record["unknown"]), (target, name)
            assert (record["scale"]["granularities"] is None) == ("scale_granularities" in record["unknown"]), target
            for finding in RF.reconcile(unit, facet):
                assert finding["kind"] in ("scaling_exceeds_readout", "unaudited_scaling", "unmapped_scaling")
            derived += 1
    assert derived >= 3


def test_layouts_cached_by_the_old_reader_are_re_read_from_their_recorded_source(tmp_path) -> None:
    source = tmp_path / "ISA.scala"
    source.write_text(_ISA, encoding="utf-8")
    stale = _facts(with_unresolved=False)
    record = stale["facts"]["interfaces"][0]
    del record["bundles"]["StoreConfig"]  # what the old reader dropped
    record["source"] = str(source)
    assert RF.derive("t", facts=stale).scale_granularities is None

    current = RF.with_current_register_layouts(stale)
    assert "StoreConfig" not in record["bundles"]  # the caller's bundle is not mutated
    facet = RF.derive("t", facts=current)
    assert facet.scale_granularities == ("tensor",)
    assert any("re-read from the recorded source" in e.observed for e in facet.evidence)

    # Already current, or a source this machine does not have: returned as it is.
    assert RF.with_current_register_layouts(_facts()) == _facts()
    record["source"] = str(tmp_path / "absent.scala")
    assert RF.with_current_register_layouts(stale) is stale


def _taxonomy(scaled_fields, plain_fields):
    member = lambda role, fields: {"mnemonic": role.upper(), "role": role, "fields": {f: [0] for f in fields}}  # noqa: E731
    return {
        "by_class": {
            "ScaledPop": [member("acc_readout_scaled", scaled_fields)],
            "Pop": [member("acc_readout", plain_fields)],
        }
    }


def test_a_scaled_readout_that_cannot_address_part_of_a_drain_holds_a_per_tensor_scale() -> None:
    # Same operand fields as the plain readout: nothing selects a part of what is drained, so one
    # scale per drain, and a scale constant over the tensor is expressible. A LOWER BOUND: finer
    # is unknown, and unknown is not a refusal.
    facet = RF.derive("synthetic", facts={}, taxonomy=_taxonomy(("vd", "vs2"), ("vd", "vs2")))
    assert facet.scale_granularities == ("tensor",) and facet.scale_granularities_complete is False
    assert facet.admits_granularity("tensor") is True
    assert facet.admits_granularity("row") is None
    assert "scale_granularities" not in facet.unknown and "scale_granularities_finer" in facet.unknown
    assert facet.to_dict()["scale"]["granularities_complete"] is False


def test_a_scaled_readout_with_an_operand_the_plain_one_lacks_derives_nothing() -> None:
    # The mutation: an extra operand could be a row or lane selector. The bound is not claimed.
    facet = RF.derive("synthetic", facts={}, taxonomy=_taxonomy(("vd", "vs2", "lane"), ("vd", "vs2")))
    assert facet.scale_granularities is None and facet.admits_granularity("tensor") is None
    assert "scale_granularities" in facet.unknown


def test_the_numerics_handoff_is_one_digest_over_what_a_scheduler_may_assume() -> None:
    abi = {
        "schema": RF.SCALAR_ABI_SCHEMA,
        "accumulator_dtype": "i32",
        "output_dtype": "i8",
        "scale_dtype": "f32",
        "clamp_min": -128,
        "clamp_max": 127,
        "provenance": {"scope": "synthetic"},
    }
    facet = RF.derive("synthetic", facts={}, scalar_abi=abi, taxonomy=_taxonomy(("vd", "vs2"), ("vd", "vs2")))
    handoff = facet.numerics_handoff()
    # Exactly the record a numerics contract is built from, and the granularities it may be built at.
    assert handoff["readout_facts"]["schema"] == RF.SCALAR_ABI_SCHEMA
    assert (handoff["granularities"], handoff["granularities_complete"]) == (["tensor"], False)
    assert handoff["sha256"] == facet.numerics_handoff()["sha256"]
    # A different readout is a different digest: the two sides cannot agree by accident.
    other = RF.derive("synthetic", facts={}, scalar_abi={**abi, "clamp_max": 126}, taxonomy=None)
    assert other.numerics_handoff()["sha256"] != handoff["sha256"]


_SUM = {
    "schema": RF.OPERAND_SUM_SCHEMA,
    "operands": 2,
    "operand_dtype": "i8",
    "scale_dtype": "f32",
    "operand_rounding": "half_even",
    "operand_saturates": True,
    "provenance": {"scope": "synthetic header"},
}


def test_a_scaled_load_into_an_addressable_accumulator_licenses_an_operand_sum() -> None:
    facet = RF.derive("t", facts=_facts(), unit={"name": "u"}, scalar_abi=_ABI, operand_sum=_SUM)
    assert facet.operand_sum["operands"] == 2 and facet.rungs_for("operand_sum") == ("scalar_abi",)
    assert not facet.unknown  # a licence, or its absence, is never an unknown of the readout
    # The bound is computed: two per-operand roundings against one, then a factor the readout carries.
    assert facet.operand_sum_bound() == 1 and facet.operand_sum_bound((0.4, 1.0)) == 1
    assert facet.operand_sum_bound((1.5, 0.5)) == 2 and facet.operand_sum_bound((2.5, 0.5)) == 3
    assert facet.operand_sum_refusal((0.5, 0.25)) is None
    assert facet.operand_sum_refusal((1.5, 0.25)) is None  # the readout's derived scale carries 1.5
    assert "negative" in facet.operand_sum_refusal((-0.5, 0.25))
    assert "sums 2 operands" in facet.operand_sum_refusal((0.5, 0.25, 0.1))
    assert facet.to_dict()["operand_sum"]["bound_lsb"] == 1


def test_a_scaled_load_with_nowhere_to_meet_is_not_an_operand_sum() -> None:
    # Delete the accumulator store: the load still multiplies, and two loads have no place to add.
    facts = _facts()
    facts["facts"]["memories"] = []
    facet = RF.derive("t", facts=facts, unit={"name": "u"}, scalar_abi=_ABI, operand_sum=_SUM)
    assert facet.operand_sum is None and "addressable" in facet.operand_sum_absent
    assert facet.operand_sum_refusal((0.5, 0.5)) == facet.operand_sum_absent
    # No contract at all is the same answer with a different reason, and still not an unknown.
    silent = RF.derive("t", facts=_facts(), unit={"name": "u"}, scalar_abi=_ABI)
    assert silent.operand_sum is None and "no load-scale contract" in silent.operand_sum_absent
    assert "operand_sum" not in silent.unknown


def test_a_multiplier_above_one_needs_a_readout_scale_to_carry_it() -> None:
    bare = RF.ReadoutFacet(target="t", unit="u", accumulator_kind="addressable", operand_sum=dict(_SUM))
    assert "exceed one" in bare.operand_sum_refusal((1.5, 0.5))
    assert bare.operand_sum_refusal((1.0, 0.5)) is None
