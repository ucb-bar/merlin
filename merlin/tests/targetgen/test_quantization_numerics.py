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


#: Digests of the v1 scheme's derived artifacts at the parent revision (3531ac020), before the
#: numerics and operation-domain declarations existed. They must not move.
V1_DIGESTS = {
    "scheme_row": "ebbbbf5e186272b83e1322c1354a0d309e85f2f27970fa2c8ff57228f46960c4",
    "recipe_sha256": "7539a6f1f18136d5ce7dcd4ac4e9f7f0e45000af02e6a357251296e05bb47797",
    "corpus_entries": "ca8214d0463ab365491e8261537e420c28d9f85e09a166c2860eae86c7244427",
    "composition_truth": "b82dffb93b9e4476687e96619943f9e1762c201f3b1688bdf00f92bd0c89763a",
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
    facts = corpus_facts.__wrapped__()
    entries = [e for e in CC.entries(spec, facts) if e["generalization"]["quantization_scheme_id"] == "q_pow2_tensor"]
    truth = [
        FTC.expectation(
            e["generalization"]["composition_chain"],
            e["generalization"]["quantization_scheme"],
            e["generalization"]["composition_instance"],
            facts=e["generalization"]["composition_facts"],
        )
        for e in entries
    ]
    assert {
        "scheme_row": _digest(v1),
        "recipe_sha256": recipe["recipe_sha256"],
        "corpus_entries": _digest(entries),
        "composition_truth": _digest(truth),
    } == V1_DIGESTS


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
