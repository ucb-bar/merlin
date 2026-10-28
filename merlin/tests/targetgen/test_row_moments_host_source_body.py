"""Row-moments host source-body screen: closed proof, simulated review admission, and refusals.

The fixture is a generic-form module holding the two-result row-moments
``linalg.generic`` (zero inputs, two ``[32, 1]`` f32 inits) whose body reads the
function argument through ``tensor.extract``. Every mutation below is a text
substitution guarded by an exactly-once count, so a refusal can only come from
the single deliberate edit.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

import pytest
import yaml

from merlin.common import mlir_query as mq
from merlin.frontends import linalg_row_moments_source_body as rm
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures" / "row_moments_32x32.mlir"
ROW_ID = "host_f32_layer_norm_two_result_moments"
DIGEST = "2caf020879e81e59b013922f88747a6b65507a6056e8c1bb187bc6f1b961e3b0"
FRONTEND_OP = "aten.native_layer_norm.default"
SCHEMA = "merlin.static_f32_row_moments_source_body.v1"
DATA_TYPE = "tensor<32x32xf32>"
OBSERVED = {
    "family": "reduction",
    "ordered_operand_dtypes": ["f32", "f32"],
    "ordered_result_dtypes": ["f32", "f32"],
    "rank": 2,
}
SOURCE_ROW = {"frontend_op": FRONTEND_OP, "mlir_operation": "linalg.generic", "count": 1}


def _profile_path() -> Path:
    found = sorted(ROOT.glob("examples/*/platform/host-capabilities-portable-v3.yaml"))
    assert len(found) == 1
    return found[0]


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(_profile_path().read_bytes())


@pytest.fixture(scope="module")
def fixture_text():
    return FIXTURE.read_text()


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def _sub(text: str, old: str, new: str, *, count: int = 1) -> str:
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


def _declaration(shape=(32, 32), digest=DIGEST) -> dict:
    return {
        "schema": SCHEMA,
        "operation": rm.OPERATION,
        "data_operand": {"dtype": "f32", "shape": list(shape), "reduction_axis": 1},
        "canonical_body_sha256": digest,
    }


def _simulated_reviewed(document: dict, *, mutate=None) -> dict:
    """In-memory only: flip just the row-moments row (and its contract) to the reviewed form."""
    document = copy.deepcopy(document)
    row = next(r for r in document["operations"] if r["id"] == ROW_ID)
    row["status"] = "reviewed"
    row["numerical_contract"]["status"] = "reviewed"
    row["numerical_contract"]["evidence"].pop("review_status", None)
    if mutate is not None:
        mutate(row)
    return document


def _selection(document: dict) -> dict:
    raw = yaml.safe_dump(document).encode()
    return {
        "int8_w8a8": {
            "capability_spec": document,
            "capability_spec_sha256": hashlib.sha256(raw).hexdigest(),
            "package_sha256": document["compiler"]["package_sha256"],
            "dtype_strategy": document["compiler"]["dtype_strategy"],
        }
    }


def _admit(document, ops, *, signature=None, row=None):
    return admit_host_operation(
        _selection(document),
        dict(SOURCE_ROW if row is None else row),
        dict(OBSERVED if signature is None else signature),
        source_operations=ops,
    )


# ---------------------------------------------------------------- 1. closed proof


def test_fixture_digest_is_the_declared_digest(fixture_text):
    assert rm.canonical_body_sha256(_generic(fixture_text)) == DIGEST


def test_prove_reports_shape_axis_column_reads_and_extract_sites(fixture_text):
    proof = rm.prove(_generic(fixture_text), _declaration())
    assert proof == {
        "data_operand": {"dtype": "f32", "shape": [32, 32], "reduction_axis": 1},
        "row_index": "linalg.index 0 over [R, 1]",
        "column_reads": 32,
        "extract_sites": 9,
        "canonical_body_sha256": DIGEST,
    }


def test_screen_admits_the_fixture_with_the_declared_digest(fixture_text):
    declaration = {"id": "neutral_row", "source_body": _declaration()}
    result = rm.screen(declaration, dict(SOURCE_ROW), dict(OBSERVED), (_generic(fixture_text),), None)
    assert result["status"] == "admitted"
    proof = result["proof"]
    assert proof["schema"] == SCHEMA and proof["declaration"] == "neutral_row"
    (pattern,) = proof["patterns"]
    assert pattern["data_operand"] == {"dtype": "f32", "shape": [32, 32], "reduction_axis": 1}
    assert pattern["column_reads"] == 32 and pattern["extract_sites"] == 9
    assert pattern["canonical_body_sha256"] == DIGEST
    assert "execution remain required" in proof["scope"]


def test_provenance_attributes_do_not_move_the_digest(fixture_text):
    noisy = _sub(fixture_text, '%360 = "linalg.index"() <{dim = 0 : i64}> :', '%360 = "linalg.index"() <{dim = 0 : i64}> {prov.note = "x"} :')
    assert rm.canonical_body_sha256(_generic(noisy)) == DIGEST


# ---------------------------------------------------------------- 2. end to end


def test_shipped_row_records_the_operator_review_and_its_declaration_is_valid(shipped):
    validate_host_capabilities(shipped)
    row = next(r for r in shipped["operations"] if r["id"] == ROW_ID)
    assert row["status"] == "reviewed" and row["numerical_contract"]["status"] == "reviewed"
    assert "OPERATOR-SIGNOFF-20261028.txt sha256 54ac3295" in row["numerical_contract"]["evidence"]["reviewed_by"]
    assert row["source_body"]["canonical_body_sha256"] == DIGEST
    assert row["source_body"]["data_operand"] == {"dtype": "f32", "shape": [32, 32], "reduction_axis": 1}
    assert row["signature"]["ranks"] == [2]


def test_simulated_reviewed_row_admits_the_fixture_with_proof(shipped, fixture_text):
    document = _simulated_reviewed(shipped)
    validate_host_capabilities(document)
    result = _admit(document, (_generic(fixture_text),))
    assert result["status"] == "admitted"
    (profile,) = result["profiles"]
    assert profile["profile"] == "int8_w8a8" and profile["status"] == "admitted"
    proof = profile["source_body_proof"]
    assert profile["decisions"][0]["source_body_proof"] == proof
    assert proof["schema"] == SCHEMA and proof["declaration"] == ROW_ID
    assert proof["operation"] == rm.OPERATION
    (pattern,) = proof["patterns"]
    assert pattern["column_reads"] == 32 and pattern["extract_sites"] == 9
    assert pattern["data_operand"]["shape"] == [32, 32]
    assert pattern["canonical_body_sha256"] == DIGEST


def test_without_parsed_source_operations_the_row_is_not_admitted(shipped):
    document = _simulated_reviewed(shipped)
    for ops in (None, ()):
        result = _admit(document, ops)
        assert result["status"] == "unknown"
        assert "source_body_proof" not in result["profiles"][0]
        assert "requires parsed source operations" in result["profiles"][0]["decisions"][0]["reason"]


def test_as_shipped_reviewed_row_admits_the_fixture_only_with_its_parsed_body(shipped, fixture_text):
    assert _admit(shipped, (_generic(fixture_text),))["status"] == "admitted"
    assert _admit(shipped, None)["status"] != "admitted"


def test_occurrence_roster_must_match_the_source_row(shipped, fixture_text):
    document = _simulated_reviewed(shipped)
    op = _generic(fixture_text)
    twin = _generic(fixture_text)
    assert _admit(document, (op, twin), row={**SOURCE_ROW, "count": 2})["status"] == "admitted"
    assert _admit(document, (op,), row={**SOURCE_ROW, "count": 2})["status"] != "admitted"
    assert _admit(document, (op, op), row={**SOURCE_ROW, "count": 2})["status"] != "admitted"
    assert _admit(document, [op])["status"] != "admitted"  # roster must be a tuple
    assert _admit(document, (op,), row={**SOURCE_ROW, "frontend_op": "aten.mean.dim"})["status"] != "admitted"


# ---------------------------------------------------------------- 3. refusals


@pytest.mark.parametrize(
    ("signature", "reason"),
    [
        pytest.param(
            {**OBSERVED, "ordered_result_dtypes": ["f32", "f32", "f32"]}, "two-init f32 rank-2 ABI", id="three_results"
        ),
        pytest.param(
            {**OBSERVED, "ordered_operand_dtypes": ["bf16", "bf16"], "ordered_result_dtypes": ["bf16", "bf16"]},
            "two-init f32 rank-2 ABI",
            id="bf16_signature",
        ),
        pytest.param({**OBSERVED, "rank": 3}, "two-init f32 rank-2 ABI", id="rank3"),
    ],
)
def test_wrong_abi_signature_is_refused(shipped, fixture_text, signature, reason):
    op = _generic(fixture_text)
    assert _admit(_simulated_reviewed(shipped), (op,), signature=signature)["status"] != "admitted"
    screened = rm.screen({"id": "neutral_row", "source_body": _declaration()}, dict(SOURCE_ROW), signature, (op,), None)
    assert screened["status"] == "unsupported" and reason in screened["reason"]


def _bf16_data(text):
    return _sub(text, DATA_TYPE, "tensor<32x32xbf16>", count=10)


def _ub_five(text):
    return _sub(text, '%363 = "arith.constant"() <{value = 4 : index}>', '%363 = "arith.constant"() <{value = 5 : index}>')


def _lane_offset_six(text):
    return _sub(text, "<{value = 7 : index}>", "<{value = 6 : index}>")


def _swapped_extract_indices(text):
    return _sub(text, '"tensor.extract"(%x, %360, %407)', '"tensor.extract"(%x, %407, %360)')


def _wider_data(text):
    return _sub(text, DATA_TYPE, "tensor<32x64xf32>", count=10)


def _changed_divisor(text):
    return _sub(
        text,
        '%648 = "arith.constant"() <{value = 3.200000e+01 : f32}>',
        '%648 = "arith.constant"() <{value = 3.300000e+01 : f32}>',
    )


def _second_captured_value(text):
    text = _sub(
        text,
        '    %354 = "tensor.empty"() : () -> tensor<32x1xf32>\n',
        '    %outer = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32\n'
        '    %354 = "tensor.empty"() : () -> tensor<32x1xf32>\n',
    )
    return _sub(text, '%554 = "arith.addf"(%361, %380)', '%554 = "arith.addf"(%outer, %380)')


# bf16 data: xDSL's generic parser accepts a bf16 data operand as long as the
# trailing function types agree with the argument type, so the real dtype check
# in prove() is reached without any monkeypatching. Only the data tensor type
# changes (function argument and the nine extract operand types); the extract
# result type stays f32, which keeps the body's arithmetic uses well typed.
BODY_MUTATIONS = [
    pytest.param(_bf16_data, (32, 32), "differs from the declared static f32 shape", id="bf16_data_operand"),
    pytest.param(_ub_five, (32, 32), r"column index 3\d is outside \[0, 32\)", id="chunk_bound_out_of_range"),
    pytest.param(_lane_offset_six, (32, 32), "does not read each column of its row exactly once", id="column_read_twice"),
    pytest.param(_swapped_extract_indices, (32, 32), "data row index is not linalg.index 0", id="swapped_axes"),
    pytest.param(_wider_data, (32, 32), "differs from the declared static f32 shape", id="data_wider_than_declared"),
    pytest.param(lambda t: t, (32, 64), "differs from the declared static f32 shape", id="declaration_wider_than_data"),
    pytest.param(_changed_divisor, (32, 32), "canonical body digest differs", id="digest_mismatch_constant"),
    pytest.param(
        _second_captured_value, (32, 32), "captures an outer value other than the extracted data operand", id="second_capture"
    ),
]


@pytest.mark.parametrize(("mutate", "shape", "reason"), BODY_MUTATIONS)
def test_mutated_body_or_declaration_is_refused_by_prove(fixture_text, mutate, shape, reason):
    op = _generic(mutate(fixture_text))
    with pytest.raises(rm.RowMomentsRefused, match=reason):
        rm.prove(op, _declaration(shape))


@pytest.mark.parametrize(("mutate", "shape", "reason"), BODY_MUTATIONS)
def test_mutated_body_or_declaration_is_not_admitted_end_to_end(shipped, fixture_text, mutate, shape, reason):
    def set_shape(row):
        row["source_body"]["data_operand"]["shape"] = list(shape)

    document = _simulated_reviewed(shipped, mutate=set_shape)
    validate_host_capabilities(document)
    result = _admit(document, (_generic(mutate(fixture_text)),))
    assert result["status"] == "unsupported"
    (profile,) = result["profiles"]
    assert "source_body_proof" not in profile
    # The shipped [2, 8] sibling is a PROPOSED row; its own proof refuses these bodies too, so both
    # declarations report the structural refusal and the unreviewed one cannot mask it as unknown.
    assert len(profile["decisions"]) == 3  # reviewed [32, 32] plus the proposed [2, 8] and rank-3 siblings
    for decision in profile["decisions"]:
        assert decision["status"] == "unsupported" and "source_body_proof" not in decision
        # the rank-3 sibling is excluded by its own signature before any body proof runs
        assert "row moments structural proof refused" in decision["reason"] or decision["reason"] == "rank 2 not in [3]"
    reviewed_decision = next(d for d in profile["decisions"] if d["declaration"] == ROW_ID)
    assert re.search(reason, reviewed_decision["reason"])


def test_unmutated_control_still_admits(shipped, fixture_text):
    """Guards the parametrized refusals: the same harness admits the untouched fixture."""
    assert rm.prove(_generic(fixture_text), _declaration())["column_reads"] == 32


# ---------------------------------------------------------------- (i) declaration validation


def _mut(path, value):
    def apply(declaration):
        node = declaration
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


def _drop(path):
    def apply(declaration):
        node = declaration
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return apply


BAD_DECLARATIONS = [
    pytest.param(_mut(["schema"], "merlin.static_f32_row_moments_source_body.v2"), id="schema"),
    pytest.param(_mut(["operation"], "row_mean"), id="operation"),
    pytest.param(_mut(["extra"], 1), id="extra_key"),
    pytest.param(_drop(["canonical_body_sha256"]), id="missing_digest"),
    pytest.param(_mut(["data_operand", "extra"], 1), id="extra_data_key"),
    pytest.param(_drop(["data_operand", "reduction_axis"]), id="missing_axis"),
    pytest.param(_mut(["data_operand", "reduction_axis"], 0), id="axis_zero"),
    pytest.param(_mut(["data_operand", "reduction_axis"], 2), id="axis_two"),
    pytest.param(_mut(["data_operand", "dtype"], "bf16"), id="dtype"),
    pytest.param(_mut(["data_operand", "shape"], [32, 32, 4]), id="shape_rank3"),
    pytest.param(_mut(["data_operand", "shape"], [32]), id="shape_rank1"),
    pytest.param(_mut(["data_operand", "shape"], [32, 0]), id="shape_zero"),
    pytest.param(_mut(["data_operand", "shape"], [32, True]), id="shape_bool"),
    pytest.param(_mut(["data_operand", "shape"], (32, 32)), id="shape_tuple"),
    pytest.param(_mut(["canonical_body_sha256"], "not-a-digest"), id="digest_not_sha"),
    pytest.param(_mut(["canonical_body_sha256"], DIGEST[:-1]), id="digest_short"),
]


def test_valid_declaration_validates_and_copies_the_shape():
    body = _declaration()
    checked = rm.validate(body)
    assert checked == body and checked["data_operand"]["shape"] is not body["data_operand"]["shape"]


def test_non_mapping_declaration_is_rejected():
    for body in (None, [], "x"):
        with pytest.raises(ValueError):
            rm.validate(body)


@pytest.mark.parametrize("mutate", BAD_DECLARATIONS)
def test_bad_declaration_is_a_value_error(mutate, shipped):
    body = _declaration()
    mutate(body)
    with pytest.raises(ValueError, match="row moments needs one closed rank-2 f32 data operand"):
        rm.validate(body)

    def apply(row):
        row["source_body"] = body

    document = _simulated_reviewed(shipped, mutate=apply)
    with pytest.raises(ValueError):
        validate_host_capabilities(document)


def _shipped_row(shipped) -> dict:
    return copy.deepcopy(next(r for r in shipped["operations"] if r["id"] == ROW_ID))


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"ops": ["aten.native_layer_norm.default", "aten.mean.dim"]}, id="extra_op"),
        pytest.param({"ops": ["aten.mean.dim"]}, id="other_op"),
        pytest.param({"ops": []}, id="no_ops"),
        pytest.param(
            {"signature": {**rm.SIGNATURE, "ranks": [2, 3]}},
            id="ranks_2_3",
        ),
        pytest.param({"signature": {**rm.SIGNATURE, "ranks": [3]}}, id="rank_3"),
        pytest.param({"signature": {**rm.SIGNATURE, "ordered_result_dtypes": ["f32", "f32", "f32"]}}, id="three_results"),
        pytest.param({"signature": {**rm.SIGNATURE, "ordered_operand_dtypes": ["bf16", "bf16"]}}, id="bf16_operands"),
        pytest.param({"numerical_contract": "reviewed"}, id="contract_not_mapping"),
    ],
)
def test_row_shape_is_closed_to_the_two_init_layer_norm_abi(shipped, changes):
    row = _shipped_row(shipped)
    rm.validate_row(row)
    row.update(changes)
    with pytest.raises(ValueError, match="row moments needs"):
        rm.validate_row(row)
    document = copy.deepcopy(shipped)
    index = next(i for i, r in enumerate(document["operations"]) if r["id"] == ROW_ID)
    document["operations"][index] = row
    with pytest.raises(ValueError):
        validate_host_capabilities(document)


# ---------------------------------------------------------------- sibling rows keep digests fixed

SIBLING_ID = "host_f32_layer_norm_two_result_moments_2x8"
SIBLING_DIGEST = "80ae7cf203308d57065fed1031a198251023bcb85812d9ed8014fdb77b12726c"


def test_reviewed_and_proposed_digests_are_byte_identical_to_the_recorded_ones(shipped):
    rows = {r["id"]: r for r in shipped["operations"]}
    assert rows[ROW_ID]["source_body"]["canonical_body_sha256"] == DIGEST
    sibling = rows[SIBLING_ID]
    assert sibling["source_body"]["canonical_body_sha256"] == SIBLING_DIGEST
    assert sibling["source_body"]["data_operand"]["shape"] == [2, 8]
    assert sibling["status"] == "reviewed" and sibling["numerical_contract"]["status"] == "reviewed"
    assert sibling["numerical_contract"]["evidence"]["reviewed_by"].startswith(
        "Reviewed by Loren Hung (approved in operator chat 2026-10-28)")
    assert "no numerical comparison against eager" in sibling["numerical_contract"]["evidence"]["reviewed_by"]
    assert SIBLING_ID in shipped["review"]["operator_chat_addendum_2"]
    # demote counterfactual: the in-memory unreviewed copy is what the old proposal looked like
    demoted = copy.deepcopy(shipped)
    row = next(r for r in demoted["operations"] if r["id"] == SIBLING_ID)
    row["status"] = row["numerical_contract"]["status"] = "unreviewed"
    assert next(r for r in demoted["operations"] if r["id"] == SIBLING_ID)["status"] == "unreviewed"
    assert rm.canonical_body_sha256(_generic(FIXTURE.read_text())) == DIGEST


def test_reviewed_sibling_refuses_the_other_shape_and_does_not_mask_a_reviewed_row(shipped, fixture_text):
    # The reviewed [32, 32] row admits its fixture; the reviewed [2, 8] row refuses it structurally.
    result = _admit(shipped, (_generic(fixture_text),))
    assert result["status"] == "admitted"
    by_id = {d["declaration"]: d for d in result["profiles"][0]["decisions"]}
    assert by_id[ROW_ID]["status"] == "admitted"
    assert by_id[SIBLING_ID]["status"] == "unsupported"
