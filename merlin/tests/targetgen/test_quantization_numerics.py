"""Per-scheme quantization numerics are explicit, evidenced and separated by checked witnesses."""

import copy
import json
import os
from pathlib import Path

import pytest
import yaml
from test_composition_corpus import facts as corpus_facts
from test_quant_recipe import _scheme_facet

from merlin.common.paths import repo_root
from merlin.targetgen import composition_corpus as CC
from merlin.targetgen import functional_truth_compositions as FTC
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import quantization_numerics as QN
from merlin.targetgen.quantization_spec import resolve_quantization_schemes
from merlin.targetgen.software_spec import load_software_spec, validate_software_spec

SPEC = repo_root() / "examples/gemmini/target/software-spec.yaml"
ALL = ["q_pow2_tensor", "q_f32_tensor", "q_f32_column_block", "q_f32_per_operand_sum", "q_f32_packed_panel"]


@pytest.fixture
def document():
    return yaml.safe_load(SPEC.read_bytes())


def _numerics(document):
    return document["quantization"]["numerics"]


def test_every_declared_scheme_states_every_property_with_evidence(document):
    block = QN.validate(validate_software_spec(document))
    assert set(block["schemes"]) == {row["id"] for row in document["quantization"]["schemes"]["schemes"]}
    for properties in block["schemes"].values():
        assert set(properties) == set(QN.PROPERTIES)
        assert {record["evidence"] for record in properties.values()} <= QN.EVIDENCE
    assert block["unmodeled"]


def test_numerics_add_no_scheme_and_leave_scheme_rows_unchanged(document):
    stripped = copy.deepcopy(document)
    stripped["quantization"].pop("numerics")
    context = {"output_path": {"readout": [{"kind": "scale", "granularity": "per_command", "columns_per_scale": 3}]}}
    context.update(dtype="int8", max_k_extension=64, weight_panel_width=3)
    with_block, _ = resolve_quantization_schemes(validate_software_spec(document), context)
    without, _ = resolve_quantization_schemes(validate_software_spec(stripped), context)
    assert with_block == without


#: Digests at 3531ac020, before the numerics and operation-domain declarations existed.
#: The authored quantization contract excludes only the new numerics block.
V1_DIGESTS = {
    "scheme_row": "ebbbbf5e186272b83e1322c1354a0d309e85f2f27970fa2c8ff57228f46960c4",
    "recipe_sha256": "7539a6f1f18136d5ce7dcd4ac4e9f7f0e45000af02e6a357251296e05bb47797",
    "quantization_contract": "b4c2fcd8d6c70bb26db3fefa962f63cc9f2d0e960de21fcb50e1649fcb182fa2",
}

# eae0cdbee intentionally changed corpus selection and assigned new shape classes to pow2.
# These truth digests match 3531ac020's _case + expectation for contraction_readout;
# other chains have distinct members and instruction obligations even at the same shape.
# Construct these fixed cases directly, independently of corpus membership or order.
V1_COMPOSITION_TRUTH = {
    "aligned": "c20024abf3e6ca33b1cdabda37f5dd4142bc808caf4a9756ddea8cca8bd133e8",
    "n_tail": "d86eb1e49beedf81149c55fad1230885499bfecfacb50ca1b3f169ed52eb2157",
    "multi_block": "ef1333069e84ab5f68c10c376b2dfa029d890a379c73718950a979da6e19ec75",
    "fused_pool": "474c04c6cbf03d77091fc310e562342b5f38681b92ee49a5ec0c9b8e2074432a",
    "os_constant": "54452f5cde655322ecad3f7a14b92ade40da83bc55d0101dfbd9c2407aeb3533",
    "os_bound_outside": "d0214a2795ea35fc501e2c7a7c957494931a63eadd305ce7d9aed90e2131f070",
    "os_runtime": "1c9e1aa0ebf4eb7ea9a1dcbe80d0b45456829d8315bf7ffbbf7f6a78e4a5df0f",
}


def _digest(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def test_v1_scheme_artifacts_are_byte_identical():
    spec = load_software_spec(SPEC)
    context = {"output_path": {"readout": [{"kind": "scale", "granularity": "per_command", "columns_per_scale": 3}]}}
    context.update(dtype="int8", max_k_extension=64, weight_panel_width=3)
    resolved, _ = resolve_quantization_schemes(spec, context)
    v1 = next(row for row in resolved["schemes"] if row["id"] == "q_pow2_tensor")
    recipe = QR.derive(
        _scheme_facet(3), scheme=v1, calibration_domain=spec["numerical_semantics"]["calibration_domain"]
    ).to_dict()
    quantization = {key: value for key, value in spec["quantization"].items() if key != "numerics"}
    assert {
        "scheme_row": _digest(v1),
        "recipe_sha256": recipe["recipe_sha256"],
        "quantization_contract": _digest(quantization),
    } == V1_DIGESTS
    facts = corpus_facts.__wrapped__()
    chain = next(row for row in spec["compositions"]["chains"] if row["id"] == "contraction_readout")
    assert v1["readout_strategy"] == "exact_rational_host"
    for shape_class, expected_digest in V1_COMPOSITION_TRUTH.items():
        case = CC._case(chain, v1, shape_class, facts)["composition_instance"]
        assert case["multipliers"] == [0.0625]
        truth = FTC.expectation(chain, v1, case, facts=facts)
        assert _digest(truth) == expected_digest, (chain["id"], shape_class)


@pytest.mark.parametrize("boundary", ["readout_ties", "f32_multiplier_formation", "internal_partial_sum_width"])
def test_a_declared_result_that_the_order_does_not_produce_is_refused(document, boundary):
    row = next(b for b in _numerics(document)["boundaries"] if b["id"] == boundary)
    row["expected"] = row["rejected"]["result"]
    with pytest.raises(ValueError, match="gives"):
        validate_software_spec(document)


def test_a_witness_that_does_not_distinguish_the_regimes_is_refused(document):
    row = next(b for b in _numerics(document)["boundaries"] if b["id"] == "readout_ties")
    row["input"]["accumulators"] = [4, -4]
    row["expected"], row["rejected"]["result"] = [2, -2], [2, -2]
    with pytest.raises(ValueError, match="does not distinguish"):
        validate_software_spec(document)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda n: n["schemes"]["q_f32_tensor"].pop("bias"), "must declare exactly"),
        (lambda n: n["schemes"].pop("q_f32_packed_panel"), "exactly the declared schemes"),
        (lambda n: n["schemes"]["q_pow2_tensor"]["rounding"].update(evidence="assumed"), "evidence must be"),
        (lambda n: n["schemes"]["q_pow2_tensor"]["operand_type"].pop("fact"), "fact path"),
        (lambda n: n["schemes"]["q_pow2_tensor"]["rounding"].update(basis=""), "basis"),
        (lambda n: n["schemes"]["q_pow2_tensor"]["scale_value"].update(value="arbitrary_f32"), "axis point"),
        (lambda n: n["schemes"]["q_pow2_tensor"]["internal_accumulation"]["value"].update(bits=32), "internal"),
        (lambda n: n["schemes"]["q_f32_tensor"]["rounding"]["value"].update(ties="half_away_from_zero"), "rounding"),
        (lambda n: n.update(unmodeled=[]), "unmodeled"),
        (
            lambda n: n.update(boundaries=[b for b in n["boundaries"] if b["id"] == "per_operand_rounding"]),
            "without an audited",
        ),
    ],
)
def test_incomplete_or_drifting_numerics_are_refused(document, edit, message):
    edit(_numerics(document))
    with pytest.raises(ValueError, match=message):
        validate_software_spec(document)


def test_multiplier_formation_follows_the_scheme_readout(document):
    # The anchored f32 order is shared by f32 schemes; v1 must keep the exact rational multiplier.
    order = _numerics(document)["schemes"]["q_pow2_tensor"]["conversion_order"]["value"]["contraction"]
    next(step for step in order if step["step"] == "multiply")["multiplier"] = "rne_f32_of_exact_rational"
    with pytest.raises(ValueError, match="multiplier formation"):
        validate_software_spec(document)


@pytest.mark.parametrize("scheme_id", ALL)
@pytest.mark.parametrize("path", QN.PATHS)
def test_every_conversion_type_must_match_its_property(document, scheme_id, path):
    order = _numerics(document)["schemes"][scheme_id]["conversion_order"]["value"][path]
    checked = 0
    for index, step in enumerate(order):
        field = {"multiply_accumulate": "operand", "convert": "to", "operand_sum": "load_dtype"}.get(
            step["step"], "dtype"
        )
        if field not in step:
            continue
        changed = copy.deepcopy(document)
        changed_order = _numerics(changed)["schemes"][scheme_id]["conversion_order"]["value"][path]
        changed_order[index][field] = "uint8"
        with pytest.raises(ValueError, match="type.*differs"):
            validate_software_spec(changed)
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize("signedness", ["unsigned", "unknown"])
def test_operand_signedness_must_match_the_declared_type(document, signedness):
    _numerics(document)["schemes"]["q_pow2_tensor"]["signedness"]["value"] = signedness
    with pytest.raises(ValueError, match="signedness"):
        validate_software_spec(document)


def test_unsigned_operand_cannot_claim_signed_arithmetic(document):
    document["numerical_semantics"]["operand_dtype"] = "uint8"
    operand = _numerics(document)["schemes"]["q_pow2_tensor"]["operand_type"]["value"]
    operand.update(dtype="uint8", range=[0, 255])
    with pytest.raises(ValueError, match="signedness"):
        validate_software_spec(document)


def test_evaluator_separates_the_audited_regimes():
    f32 = [
        {"step": "convert", "to": "f32", "rounding": "half_even"},
        {
            "step": "multiply",
            "multiplier": "rne_f32_of_exact_rational",
            "product": "f32_half_even",
            "granularity": "tensor",
        },
        {"step": "round", "ties": "half_even"},
        {"step": "saturate", "dtype": "int8"},
    ]
    exact = [{**f32[1], "multiplier": "exact_rational", "product": "exact"}, *f32[2:]]
    assert QN.evaluate(f32, {"accumulators": [-399], "multipliers": ["1/14"]}) == [-29]
    assert QN.evaluate(exact, {"accumulators": [-399], "multipliers": ["1/14"]}) == [-28]
    # Above the f32 integer range a power-of-two scale still separates the two readouts.
    assert QN.evaluate(exact, {"accumulators": [16777217], "multipliers": ["1/33554432"]}) == [1]
    assert QN.evaluate(f32, {"accumulators": [16777217], "multipliers": ["1/33554432"]}) == [0]


FACTS = {"facts": {"datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]}}


def test_rtl_claims_are_confirmed_only_by_matching_facts():
    block = load_software_spec(SPEC)["quantization"]["numerics"]
    confirmed = QN.audit(block, FACTS)
    assert all(confirmed[s]["operand_type"] == confirmed[s]["transfer"] == "rtl_derived" for s in ALL)
    narrowed = copy.deepcopy(FACTS)
    narrowed["facts"]["datapaths"][1]["dtype"] = "i16"
    assert QN.audit(block, narrowed)["q_pow2_tensor"]["accumulator_type"] == QN.UNCONFIRMED
    assert all(status != "rtl_derived" for status in QN.audit(block, {})["q_f32_tensor"].values())


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RTL_FACTS") or not Path(os.environ["MERLIN_RTL_FACTS"]).is_file(),
    reason="the selected RTL facts (MERLIN_RTL_FACTS) are an operator input",
)
def test_selected_facts_confirm_every_rtl_claim():
    block = load_software_spec(SPEC)["quantization"]["numerics"]
    table = QN.audit(block, json.loads(Path(os.environ["MERLIN_RTL_FACTS"]).read_text()))
    assert QN.UNCONFIRMED not in {status for row in table.values() for status in row.values()}
