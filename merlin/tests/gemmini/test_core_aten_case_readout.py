"""A case's resolved quantization scheme selects the readout written into its admission."""

from __future__ import annotations

import copy
import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.core_aten_spec_int8 import F32_READOUT, case_readout
from merlin.targetgen.software_spec import load_software_spec, validate_readout


def _selection():
    spec = load_software_spec(repo_root() / "examples" / "gemmini" / "target" / "software-spec.yaml", target="gemmini")
    return {"spec": spec, "identities": {"software_spec": {"sha256": "fixture"}}}


def _scheme(selection, scheme_id):
    row = next(r for r in selection["spec"]["quantization"]["schemes"]["schemes"] if r["id"] == scheme_id)
    return {**copy.deepcopy(row), "status": "resolved", "unknown": []}


def _admission(selection, plan):
    # The bytes core_aten_numeric writes to spec-int8-admission.json.
    return json.dumps({"selection": selection, **plan}, indent=2, sort_keys=True) + "\n"


PLAN = {"status": "prepared", "quantized_contractions": 1, "source_sha256_after": "0" * 64}


@pytest.mark.parametrize("scheme_id", [None, "q_pow2_tensor"])
def test_exact_cases_keep_the_admission_byte_identical(scheme_id):
    selection = _selection()
    before = _admission(selection, PLAN)
    scheme = None if scheme_id is None else _scheme(selection, scheme_id)
    after = case_readout(selection, scheme)
    assert after is selection
    assert _admission(after, PLAN) == before
    assert (
        json.loads(before)["selection"]["spec"]["numerical_semantics"]["readout"]["strategy"] == "exact_rational_host"
    )


@pytest.mark.parametrize("scheme_id", ["q_f32_tensor", "q_f32_column_block"])
def test_an_f32_scheme_selects_its_validated_definition(scheme_id):
    selection = _selection()
    authored = copy.deepcopy(selection)
    after = case_readout(selection, _scheme(selection, scheme_id))
    readout = after["spec"]["numerical_semantics"]["readout"]
    assert readout["strategy"] == F32_READOUT
    assert readout == validate_readout(readout)
    assert after["case_quantization_scheme"]["id"] == scheme_id
    # Only the readout moves; the selection it was derived from is untouched.
    assert selection == authored
    rest = {k: v for k, v in after["spec"]["numerical_semantics"].items() if k != "readout"}
    assert rest == {k: v for k, v in authored["spec"]["numerical_semantics"].items() if k != "readout"}


def test_an_undeclared_or_unresolved_scheme_is_refused():
    selection = _selection()
    scheme = _scheme(selection, "q_f32_tensor")
    with pytest.raises(ValueError, match="not declared"):
        case_readout(selection, {**scheme, "id": "q_invented"})
    with pytest.raises(ValueError, match="not declared"):
        case_readout(selection, {**scheme, "readout_strategy": "exact_rational_host"})
    with pytest.raises(ValueError, match="resolved"):
        case_readout(selection, {**scheme, "status": "unresolved"})
    with pytest.raises(ValueError, match="resolved"):
        case_readout(selection, {**scheme, "unknown": ["readout_strategy"]})


def test_an_ambiguous_definition_is_refused():
    selection = _selection()
    transfers = selection["spec"]["transfer_contracts"]["declarations"]
    defined = next(t for t in transfers if "readout_definitions" in t["semantics"])
    clone = copy.deepcopy(defined)
    clone["id"] = "second_readout"
    clone["semantics"]["readout_definitions"][F32_READOUT]["exactness_bridge"] = "other"
    transfers.append(clone)
    with pytest.raises(ValueError, match="2 distinct"):
        case_readout(selection, _scheme(selection, "q_f32_tensor"))
