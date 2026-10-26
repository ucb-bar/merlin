"""The output-epilogue record is DERIVED from a design's own structure, and fails closed when it cannot be."""

from __future__ import annotations

import hashlib

import pytest

from merlin.targetgen.rtl import output_epilogue as OE

_READOUT = """circuit Top :
  module Readout : @[Readout.scala 1:1]
    input clock : Clock @[Readout.scala 1:1]
    output io : { flip in : { data : SInt<32>[2], scale : UInt<32>, act : UInt<2>}, out : SInt<8>[2]}
    node _is_rect = eq(io.in.act, UInt<2>(0h1)) @[Readout.scala 10:1]
    node _gated = and(UInt<1>(0h1), _is_rect) @[Readout.scala 10:2]
    node _positive = geq(io.in.data[0], asSInt(UInt<1>(0h0))) @[Readout.scala 11:1]
    node _rectified = mux(_positive, io.in.data[0], asSInt(UInt<1>(0h0))) @[Readout.scala 11:2]
    node acted = mux(_gated, _rectified, io.in.data[0]) @[Readout.scala 12:1]
{extra}
    node s_sign = bits(io.in.scale, 31, 31) @[Float.scala 1:1]
    node s_exp = bits(io.in.scale, {exp_hi}, {frac}) @[Float.scala 2:1]
    node s_frac = bits(io.in.scale, {frac_hi}, 0) @[Float.scala 3:1]
    node s_rec = cat(s_sign, cat(s_exp, s_frac)) @[Float.scala 4:1]
    inst conv of IntToFloat @[Readout.scala 20:1]
    connect conv.io.in, asUInt(acted) @[Readout.scala 20:2]
    inst mul of FloatMul @[Readout.scala 21:1]
    connect mul.io.a, conv.io.out @[Readout.scala 21:2]
    connect mul.io.b, s_rec @[Readout.scala 21:3]
"""

_PROBE = {"module": "Readout", "value": "io.in.data", "scale": "io.in.scale", "activation": "io.in.act"}


def _fir(tmp_path, *, frac=23, extra=""):
    text = _READOUT.replace("{extra}", extra).replace("{frac}", str(frac))
    text = text.replace("{exp_hi}", "30").replace("{frac_hi}", str(frac - 1))
    path = tmp_path / "design.fir"
    path.write_text(text, encoding="utf-8")
    return path


def _facts(path, *, sha=None, pool=True):
    interfaces = [
        {
            "name": "register_bundle_layouts",
            "bundles": {
                "StoreConfig": {
                    "fields": {
                        "out_scale": {"width": 32},
                        "activation": {"width": 2},
                        "pool_size": {"width": 2},
                    }
                }
            },
            "unresolved": [],
        }
    ]
    if pool:
        interfaces.append(
            {"name": "elaborated_rtl_features", "status": "derived", "features": {"max_pool": True}, "evidence": "gate"}
        )
    return {
        "facts": {
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
            "interfaces": interfaces,
            "source": {
                "fir_path": str(path),
                "fir_sha256": sha or hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        }
    }


def _contract():
    return {
        "rtl_extraction": {OE.FIELD: dict(_PROBE)},
        "runner": {
            "chipyard_rocc": {
                "readout": {
                    "scalar_abi": {
                        "schema": "scalar_narrow_readout_contract_v1",
                        "scale_dtype": "f32",
                        "clamp_min": -128,
                        "clamp_max": 127,
                    }
                }
            }
        },
    }


def test_every_output_stage_is_read_from_the_design(tmp_path):
    record = OE.derive(_facts(_fir(tmp_path)), contract=_contract())
    assert record["stages"] == {"acc_scale": True, "relu": True, "maxpool": True}
    scale = record["scale"]
    assert scale["status"] == OE.DERIVED
    assert (scale["dtype"], scale["format"]["format"], scale["accumulator_dtype"]) == ("f32", "fp32", "i32")
    assert scale["output_dtypes"] == ["i8"] and scale["rounding"] == "half_even"
    assert scale["saturation"] == [-128, 127]
    assert scale["combined_by"] == {"instance": "mul", "of": "FloatMul", "value_through": ["conv"]}
    activation = record["activation"]
    assert activation["functions"]["relu"]["codes"] == [1]
    assert activation["unrecognized_codes"] == [] and activation["before_scale"] is True
    assert activation["identity"]["code"] == 0
    assert record["pool"]["reductions"] == ["max"]
    assert record["pool"]["padding"] == OE.UNKNOWN
    assert "not derived" in record["unknown"]["pool.padding"]
    assert record["output_path"]["readout"][-1]["padding"] == "unknown"
    assert record["accumulate"]["status"] == OE.UNKNOWN
    assert "accumulate" in record["unknown"]


def test_activation_identity_refuses_an_unbound_guard(tmp_path):
    path = _fir(tmp_path)
    path.write_text(path.read_text().replace("and(UInt<1>(0h1), _is_rect)", "or(io.unknown, _is_rect)"))
    record = OE.derive(_facts(path), contract=_contract())
    assert record["activation"]["status"] == OE.DERIVED
    assert "identity" not in record["activation"]


_POOL = """
  module Store :
    input clock : Clock
    input cfg : UInt<3>
    input bad : UInt<1>
    input row : UInt<8>
    input column : UInt<8>
    input rows : UInt<8>
    input columns : UInt<8>
    reg step : UInt<3>, clock
    connect step, cfg
    node enabled = neq(step, UInt<3>(0h7))
    node row_out = geq(row, rows)
    node column_out = geq(column, columns)
    node outside = or(row_out, column_out)
    wire address : { tag : UInt<3> }
    connect address.tag, UInt<3>(0h0)
    when outside :
      connect address.tag, UInt<3>(0h6)
  module Writer :
    input io : { address : { tag : UInt<3> }, configurable : UInt<8> }
    output data : UInt<8>
    node sentinel = eq(io.address.tag, UInt<3>(0h6))
    node tap = mux(sentinel, UInt<8>(0h0), io.configurable)
    connect data, tap
"""


def _pool_contract():
    contract = _contract()
    contract["rtl_extraction"][OE.FIELD].update(
        {
            "pool_disable": {"module": "Store", "selector": "step", "configured_by": "cfg", "enabled": "enabled"},
            "pool_padding": {
                "address": {
                    "module": "Store",
                    "out_of_bounds": "outside",
                    "value": "address",
                    "fields": ["tag"],
                    "out_of_bounds_checks": ["geq(row, rows)", "geq(column, columns)"],
                },
                "tap": {"module": "Writer", "address": "io.address", "sentinel": "sentinel", "data": "data"},
            },
        }
    )
    return contract


def test_pool_disable_and_zero_padding_follow_the_selected_rtl(tmp_path):
    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL)
    record = OE.derive(_facts(path), contract=_pool_contract())
    assert record["pool"]["disable"]["code"] == 7
    assert record["pool"]["padding"] == "zero_tap"
    assert record["pool"]["padding_evidence"]["sentinel"] == {"address.tag": 6}
    assert record["output_path"]["readout"][-1]["padding"] == "zero_tap"
    assert "pool.padding" not in record["unknown"]


@pytest.mark.parametrize(
    "replacement",
    [
        "bad",
        "row_out",
        "geq(row, columns)",
        "and(row_out, column_out)",
        "not(or(row_out, column_out))",
        "eq(row_out, column_out)",
        "xor(row_out, column_out)",
    ],
)
def test_padding_predicate_must_retain_both_image_bound_comparisons(tmp_path, replacement):
    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL.replace("or(row_out, column_out)", replacement))
    record = OE.derive(_facts(path), contract=_pool_contract())
    assert record["pool"]["padding"] == OE.UNKNOWN
    assert "bound comparisons" in record["unknown"]["pool.padding"]


def test_padding_temporary_can_be_renamed_with_its_verified_cone(tmp_path):
    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL.replace("outside", "_T_99"))
    contract = _pool_contract()
    address = contract["rtl_extraction"][OE.FIELD]["pool_padding"]["address"]
    address["out_of_bounds"] = "_T_99"
    assert OE.derive(_facts(path), contract=contract)["pool"]["padding"] == "zero_tap"
    del address["out_of_bounds_checks"]
    record = OE.derive(_facts(path), contract=contract)
    assert record["pool"]["padding"] == OE.UNKNOWN


def test_pool_evidence_bytes_ignore_contract_key_order(tmp_path):
    from merlin.common.jsonio import canonical_json

    def reordered(value):
        if isinstance(value, dict):
            return {key: reordered(item) for key, item in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reordered(item) for item in value]
        return value

    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL)
    contract = _pool_contract()
    reverse = reordered(contract)
    assert canonical_json(contract) == canonical_json(reverse)
    first = OE.derive(_facts(path), contract=contract)
    second = OE.derive(_facts(path), contract=reverse)
    assert first["pool"]["padding"] == second["pool"]["padding"] == "zero_tap"
    assert first["pool"]["evidence"] == second["pool"]["evidence"]
    assert canonical_json(first) == canonical_json(second)


@pytest.mark.parametrize("replacement", ["UInt<8>(0h1)", "io.configurable"])
def test_nonzero_or_configurable_padding_is_not_claimed_as_zero(tmp_path, replacement):
    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL.replace("mux(sentinel, UInt<8>(0h0)", "mux(sentinel, " + replacement))
    record = OE.derive(_facts(path), contract=_pool_contract())
    assert record["pool"]["padding"] == OE.UNKNOWN
    assert "constant zero" in record["unknown"]["pool.padding"]


@pytest.mark.parametrize("missing", ["pool_disable", "pool_padding"])
def test_missing_pool_probe_leaves_a_reason(tmp_path, missing):
    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL)
    contract = _pool_contract()
    del contract["rtl_extraction"][OE.FIELD][missing]
    record = OE.derive(_facts(path), contract=contract)
    assert "pool." + missing.removeprefix("pool_") in record["unknown"]


def test_a_pool_gate_without_a_disabling_value_refuses(tmp_path):
    path = _fir(tmp_path)
    path.write_text(path.read_text() + _POOL.replace("neq(step, UInt<3>(0h7))", "UInt<1>(0h1)"))
    record = OE.derive(_facts(path), contract=_pool_contract())
    assert "disable" not in record["pool"]
    assert "no compared" in record["unknown"]["pool.disable"]


def test_a_changed_padding_sentinel_refuses(tmp_path):
    path = _fir(tmp_path)
    path.write_text(
        path.read_text() + _POOL.replace("eq(io.address.tag, UInt<3>(0h6))", "eq(io.address.tag, UInt<3>(0h5))")
    )
    record = OE.derive(_facts(path), contract=_pool_contract())
    assert record["pool"]["padding"] == OE.UNKNOWN
    assert "pool.padding" in record["unknown"]


def test_without_a_probe_the_firrtl_stages_are_unknown(tmp_path):
    contract = _contract()
    contract.pop("rtl_extraction")
    record = OE.derive(_facts(_fir(tmp_path)), contract=contract)
    assert record["stages"] == {"acc_scale": False, "relu": False, "maxpool": True}
    assert "rtl_extraction.output_epilogue" in record["unknown"]["scale.format"]
    assert "which function each code selects is not derived" in record["unknown"]["activation"]
    # The register fields still say what is ENCODABLE; that alone is never a derived stage.
    assert record["activation"]["carriers"] == [{"bundle": "StoreConfig", "field": "activation", "bits": 2}]


def test_a_design_that_is_not_the_pinned_one_is_not_read(tmp_path):
    record = OE.derive(_facts(_fir(tmp_path), sha="0" * 64), contract=_contract())
    assert record["stages"]["acc_scale"] is False and record["stages"]["relu"] is False
    assert "not the one the facts pin" in record["unknown"]["activation"]


def test_an_unrecognized_activation_code_is_listed_not_claimed(tmp_path):
    extra = "\n".join(
        [
            "    node _is_other = eq(io.in.act, UInt<2>(0h2)) @[Readout.scala 13:1]",
            "    node other = mux(_is_other, io.in.data[1], acted) @[Readout.scala 13:2]",
        ]
    )
    record = OE.derive(_facts(_fir(tmp_path, extra=extra)), contract=_contract())
    assert record["activation"]["functions"] == {"relu": {**record["activation"]["functions"]["relu"], "codes": [1]}}
    assert record["activation"]["unrecognized_codes"] == [2]


def test_a_scale_split_no_registered_format_has_is_unknown(tmp_path):
    record = OE.derive(_facts(_fir(tmp_path, frac=20)), contract=_contract())
    assert record["scale"]["status"] == OE.UNKNOWN and record["stages"]["acc_scale"] is False
    assert "registered IEEE format" in record["unknown"]["scale.format"]


def test_pooling_needs_both_the_built_gate_and_a_command_field(tmp_path):
    record = OE.derive(_facts(_fir(tmp_path), pool=False), contract=_contract())
    assert record["pool"]["status"] == OE.UNKNOWN and record["stages"]["maxpool"] is False


def test_pool_geometry_does_not_admit_padded_pooling(tmp_path):
    from merlin.xdsl_dialects.lowering import output_fusion as fusion

    record = OE.derive(_facts(_fir(tmp_path)), contract=_contract())
    assert record["pool"]["status"] == OE.DERIVED
    chain = fusion.Chain(
        root=None,
        channels=2,
        float_dtype="f32",
        ops=[],
        stages=[fusion.Stage(kind, object()) for kind in ("convert", "pad", "pool", "quantize")],
    )
    decision = fusion.decide(chain, record["output_path"], fusion.Values(), {})
    assert decision.cut != "pool"
    assert any(row["cut"] == "pool" and row["code"] == fusion.PADDING for row in decision.refusals)


def test_accumulate_follows_the_derived_behavioural_fact(tmp_path):
    semantic = {"facts": {"accumulate_on_load": {"status": "derived", "value": {"accumulates_on_load": True}}}}
    record = OE.derive(_facts(_fir(tmp_path)), contract=_contract(), semantic_facts=semantic)
    assert record["accumulate"]["status"] == OE.DERIVED
    unread = {"facts": {"accumulate_on_load": {"status": "UNKNOWN"}}}
    assert OE.derive(_facts(_fir(tmp_path)), contract=_contract(), semantic_facts=unread)["accumulate"]["status"] == (
        OE.UNKNOWN
    )


_STORE_AND_LOAD = """  module Store :
    input clock : Clock
    output io : { flip write : { acc : UInt<1>, data : SInt<32>[2]}, flip sum : SInt<32>[2], wdata : SInt<32>[2]}
    node _T = mux(io.write.acc, io.sum, io.write.{arm})
    connect io.wdata, _T
  module Lanes :
    input clock : Clock
    output io : {{ flip in : SInt<{bits}>[2], out : SInt<{bits}>[2]}}
    reg hold : SInt<{bits}>[2], clock
    connect hold, io.in
{lanes}
  module Load :
    input clock : Clock
    inst lane of Lanes
"""
_COPY = "    connect io.out[0], hold[0]\n    connect io.out[1], hold[1]"
_SCALED = "    connect io.out[0], mul(hold[0], hold[1])\n    connect io.out[1], hold[1]"
_FULL_WIDTH = (
    "    wire keep : SInt<32>[2] @[Readout.scala 30:1]\n    connect keep, io.in.data @[Readout.scala 30:2]\n"
    "    connect io.full, keep @[Readout.scala 30:3]"
)


def _preload_fir(tmp_path, *, arm="data", bits=32, lanes=_COPY, full_width=True):
    text = _READOUT.replace("{extra}", _FULL_WIDTH if full_width else "").replace("{frac}", "23")
    text = text.replace("{exp_hi}", "30").replace("{frac_hi}", "22")
    if full_width:
        text = text.replace("out : SInt<8>[2]}", "out : SInt<8>[2], full : SInt<32>[2]}")
    store = _STORE_AND_LOAD.replace("{arm}", arm).replace("{{", "{").replace("}}", "}")
    store = store.replace("{bits}", str(bits)).replace("{lanes}", lanes)
    path = tmp_path / "design.fir"
    path.write_text(text + store, encoding="utf-8")
    return path


def _accumulates():
    value = {
        "store": {"select": "io.write.acc", "sum": "io.sum", "cite": {"module": "Store"}},
        "load_path": {"cite": {"module": "Load"}, "carried_by": ["lane.io.out"]},
    }
    return {"facts": {"accumulate_on_load": {"status": "derived", "value": value}}}


def test_a_bias_preload_is_an_exact_full_width_load_into_the_accumulator(tmp_path):
    record = OE.derive(_facts(_preload_fir(tmp_path)), contract=_contract(), semantic_facts=_accumulates())
    preload = record["bias_preload"]
    assert preload["status"] == OE.DERIVED
    assert (preload["dtype"], preload["bits"], preload["rounding"]) == ("i32", 32, "none")
    assert preload["range"] == [-(1 << 31), (1 << 31) - 1]
    assert preload["lanes"]["store_writes"] == "io.write.data" and preload["lanes"]["lanes"] == 2
    assert record["full_width_readout"]["status"] == OE.DERIVED
    assert record["scale"]["columns_per_scale"] == 2
    path = record["output_path"]
    assert [s["kind"] for s in path["accumulator_stages"]] == ["preload", "accumulate"]
    assert [s["kind"] for s in path["readout"]] == ["activation", "convert", "scale", "requant", "pool"]
    assert path["readout"][0]["domain"] == "accumulator" and path["exits"] == {"accumulator": True, "requant": True}


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"arm": "acc"}, "writes its own data"),  # the clear arm is not the arriving data
        ({"lanes": _SCALED}, "copies 32-bit accumulator lanes unchanged"),  # arithmetic on the lanes
        ({"bits": 8}, "copies 32-bit accumulator lanes unchanged"),  # a narrower load path
    ],
)
def test_a_load_that_changes_the_value_is_not_a_preload(tmp_path, change, why):
    record = OE.derive(_facts(_preload_fir(tmp_path, **change)), contract=_contract(), semantic_facts=_accumulates())
    assert record["bias_preload"]["status"] == OE.UNKNOWN and why in record["unknown"]["bias_preload"]
    assert [s["kind"] for s in record["output_path"]["accumulator_stages"]] == []


def test_a_readout_with_no_raw_exit_has_no_full_width_path(tmp_path):
    record = OE.derive(_facts(_preload_fir(tmp_path, full_width=False)), contract=_contract())
    assert record["full_width_readout"]["status"] == OE.UNKNOWN
    assert record["output_path"]["exits"]["accumulator"] is False


def test_a_record_whose_stages_disagree_with_its_capabilities_is_refused(tmp_path):
    record = OE.derive(_facts(_fir(tmp_path)), contract=_contract())
    broken = {**record, "stages": {**record["stages"], "relu": False}}
    with pytest.raises(ValueError, match="disagrees"):
        OE.validate(broken)
    with pytest.raises(ValueError, match="mapping"):
        OE.validate({"schema": "other"})


def test_a_malformed_probe_fails_closed(tmp_path):
    contract = _contract()
    contract["rtl_extraction"][OE.FIELD] = {"module": "Readout"}
    with pytest.raises(ValueError, match="missing"):
        OE.derive(_facts(_fir(tmp_path)), contract=contract)


def test_a_derived_contract_declares_its_output_epilogue(tmp_path):
    from merlin.targetgen import capability_manifests as cm

    residual = {
        "compute_units": [{"name": "array", "kind": "systolic", "ops": ["matmul"], "dtypes": ["int8"]}],
        **_contract(),
    }
    manifest = cm.derive_manifest({"target": "unbound"}, _facts(_fir(tmp_path)), residual=residual)
    record = OE.validate(manifest[OE.FIELD])
    assert record["stages"] == {"acc_scale": True, "relu": True, "maxpool": True}
    bare = cm.derive_manifest({"target": "unbound"}, {}, residual={"compute_units": residual["compute_units"]})
    assert not any(OE.validate(bare[OE.FIELD])["stages"].values())


def test_contract_validation_refuses_a_malformed_declaration(tmp_path):
    from merlin.targetgen import capability_manifests as cm

    residual = {"compute_units": [{"name": "array", "kind": "systolic", "ops": ["matmul"], "dtypes": ["int8"]}]}
    manifest = cm.derive_manifest({"target": "unbound"}, {}, residual=residual)
    manifest[OE.FIELD] = {**manifest[OE.FIELD], "stages": {"acc_scale": True}}
    with pytest.raises(ValueError, match="stages"):
        cm.validate(manifest)


def test_a_selected_contract_is_declared_once_from_its_own_facts(tmp_path):
    import json

    contract = _contract()
    contract.pop("rtl_extraction")
    contract_path, facts_path, out = tmp_path / "c.json", tmp_path / "f.json", tmp_path / "declared.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    facts_path.write_text(json.dumps(_facts(_fir(tmp_path))), encoding="utf-8")
    argv = ["--contract", str(contract_path), "--facts", str(facts_path), "--probe", json.dumps(_PROBE)]
    assert OE.main([*argv, "--out", str(out)]) == 0
    declared = json.loads(out.read_text(encoding="utf-8"))
    assert declared["rtl_extraction"][OE.FIELD] == _PROBE
    assert declared[OE.FIELD]["stages"] == {"acc_scale": True, "relu": True, "maxpool": True}
    assert "rtl_extraction" not in contract  # the selected contract itself is not edited
    with pytest.raises(SystemExit):
        OE.main([*argv, "--out", str(out)])  # never overwritten


def test_an_underived_record_is_well_formed_and_claims_nothing():
    record = OE.underived("no facts")
    assert record["stages"] == {"acc_scale": False, "relu": False, "maxpool": False}
    assert record["unknown"] == {"derivation": "no facts", "pool.padding": "no facts"}
    assert record["pool"]["padding"] == OE.UNKNOWN


def test_the_scalar_abi_is_read_from_the_contracts_readout_semantics(tmp_path):
    contract = _contract()
    abi = contract.pop("runner")["chipyard_rocc"]["readout"]["scalar_abi"]
    contract["readout_semantics"] = {"scalar_abi": abi}
    record = OE.derive(_facts(_fir(tmp_path)), contract=contract)
    assert record["scale"]["rounding"] == "half_even" and record["scale"]["saturation"] == [-128, 127]
    assert record["output_path"]["exits"]["requant"] is True
    del contract["readout_semantics"]
    assert "scale.rounding" in OE.derive(_facts(_fir(tmp_path)), contract=contract)["unknown"]
