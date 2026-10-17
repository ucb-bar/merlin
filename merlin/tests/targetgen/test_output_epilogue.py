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
    assert record["pool"]["reductions"] == ["max"]
    assert record["accumulate"]["status"] == OE.UNKNOWN
    assert "accumulate" in record["unknown"]


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


def test_accumulate_follows_the_derived_behavioural_fact(tmp_path):
    semantic = {"facts": {"accumulate_on_load": {"status": "derived", "value": {"accumulates_on_load": True}}}}
    record = OE.derive(_facts(_fir(tmp_path)), contract=_contract(), semantic_facts=semantic)
    assert record["accumulate"]["status"] == OE.DERIVED
    unread = {"facts": {"accumulate_on_load": {"status": "UNKNOWN"}}}
    assert OE.derive(_facts(_fir(tmp_path)), contract=_contract(), semantic_facts=unread)["accumulate"]["status"] == (
        OE.UNKNOWN
    )


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


def test_an_underived_record_is_well_formed_and_claims_nothing():
    record = OE.underived("no facts")
    assert record["stages"] == {"acc_scale": False, "relu": False, "maxpool": False}
    assert record["unknown"] == {"derivation": "no facts"}
