"""Rank-3 row-moments source proof (multimodal_policy layer norm): acceptance, refusals, rank-2 stability."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.common import mlir_query as mq
from merlin.frontends import linalg_row_moments_source_body as rm
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures" / "row_moments_1x17x16.mlir"
RANK2_FIXTURE = Path(__file__).parent / "fixtures" / "row_moments_32x32.mlir"
ROW_ID = "host_f32_layer_norm_two_result_moments_rank3_1x17x16"
DIGEST = "b49e906cd89c993f1dabf4e39f128088d91b7c36b1bf7653b007e4f7894abfce"
RANK2_DIGEST = "2caf020879e81e59b013922f88747a6b65507a6056e8c1bb187bc6f1b961e3b0"
OBSERVED = {"family": "reduction", "ordered_operand_dtypes": ["f32", "f32"], "ordered_result_dtypes": ["f32", "f32"], "rank": 3}
SOURCE_ROW = {"frontend_op": rm.FRONTEND_OP, "mlir_operation": "linalg.generic", "count": 1}
DATA = "tensor<1x17x16xf32>"


def _generic(text):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def _sub(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


def _declaration(shape=(1, 17, 16), digest=DIGEST):
    return {
        "schema": rm.SCHEMA_RANK3,
        "operation": rm.OPERATION,
        "data_operand": {"dtype": "f32", "shape": list(shape), "reduction_axis": 2},
        "canonical_body_sha256": digest,
    }


@pytest.fixture(scope="module")
def text():
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load((ROOT / "examples/gemmini/platform/host-capabilities-portable-v3.yaml").read_bytes())


def _selection(document):
    raw = yaml.safe_dump(document).encode()
    return {
        "int8_w8a8": {
            "capability_spec": document,
            "capability_spec_sha256": hashlib.sha256(raw).hexdigest(),
            "package_sha256": document["compiler"]["package_sha256"],
            "dtype_strategy": document["compiler"]["dtype_strategy"],
        }
    }


def _reviewed(document):
    document = copy.deepcopy(document)
    row = next(r for r in document["operations"] if r["id"] == ROW_ID)
    row["status"] = row["numerical_contract"]["status"] = "reviewed"
    row["numerical_contract"]["evidence"].pop("review_status", None)
    return document


def test_digest_and_proof_of_the_real_body(text):
    assert rm.canonical_body_sha256(_generic(text), rm.SCHEMA_RANK3) == DIGEST
    proof = rm.prove(_generic(text), _declaration())
    assert proof["column_reads"] == 16 and proof["extract_sites"] == 9
    assert proof["data_operand"] == {"dtype": "f32", "shape": [1, 17, 16], "reduction_axis": 2}


def test_rank2_serialization_and_digest_are_unchanged():
    op = _generic(RANK2_FIXTURE.read_text())
    assert rm.canonical_body_sha256(op) == RANK2_DIGEST
    assert rm.canonical_body_sha256(op, rm.SCHEMA) == RANK2_DIGEST
    assert rm.SCHEMA == "merlin.static_f32_row_moments_source_body.v1"
    # the rank-3 prover refuses a rank-2 body and the rank-2 schema refuses a rank-3 declaration
    with pytest.raises(rm.RowMomentsRefused):
        rm.prove(op, _declaration())
    with pytest.raises(ValueError):
        rm.validate({**_declaration(), "schema": rm.SCHEMA})


def test_screen_admits_only_with_rank3_abi(text):
    declaration = {"id": "x", "source_body": _declaration()}
    ok = rm.screen(declaration, dict(SOURCE_ROW), dict(OBSERVED), (_generic(text),), None)
    assert ok["status"] == "admitted" and ok["proof"]["schema"] == rm.SCHEMA_RANK3
    bad = rm.screen(declaration, dict(SOURCE_ROW), {**OBSERVED, "rank": 2}, (_generic(text),), None)
    assert bad["status"] == "unsupported" and "rank-3 ABI" in bad["reason"]


def _demoted(document, *, contract_too=True):
    document = copy.deepcopy(document)
    row = next(r for r in document["operations"] if r["id"] == ROW_ID)
    row["status"] = "unreviewed"
    if contract_too:
        row["numerical_contract"]["status"] = "unreviewed"
    return document


def test_shipped_row_is_operator_reviewed_and_demotion_withdraws_admission(shipped, text):
    validate_host_capabilities(shipped)
    row = next(r for r in shipped["operations"] if r["id"] == ROW_ID)
    assert row["status"] == "reviewed" and row["numerical_contract"]["status"] == "reviewed"
    reviewed_by = row["numerical_contract"]["evidence"]["reviewed_by"]
    assert reviewed_by.startswith("Reviewed by Loren Hung (approved in operator chat 2026-10-28)")
    assert "NOT repeated" in reviewed_by and "no numerical comparison against eager" in reviewed_by
    assert ROW_ID in shipped["review"]["operator_chat_addendum_2"]
    assert row["source_body"]["canonical_body_sha256"] == DIGEST and row["signature"]["ranks"] == [3]
    result = admit_host_operation(_selection(shipped), dict(SOURCE_ROW), dict(OBSERVED), source_operations=(_generic(text),))
    assert result["status"] == "admitted"
    for document in (_demoted(shipped), _demoted(shipped, contract_too=False)):
        demoted = admit_host_operation(_selection(document), dict(SOURCE_ROW), dict(OBSERVED), source_operations=(_generic(text),))
        assert demoted["status"] == "unknown"


def test_simulated_review_admits_the_real_body(shipped, text):
    document = _reviewed(shipped)
    validate_host_capabilities(document)
    result = admit_host_operation(_selection(document), dict(SOURCE_ROW), dict(OBSERVED), source_operations=(_generic(text),))
    assert result["status"] == "admitted"
    assert result["profiles"][0]["source_body_proof"]["declaration"] == ROW_ID


MUTATIONS = [
    pytest.param(lambda t: _sub(t, DATA, "tensor<1x17x16xbf16>", count=10), (1, 17, 16), id="bf16_data"),
    pytest.param(lambda t: _sub(t, DATA, "tensor<1x17x32xf32>", count=10), (1, 17, 16), id="wider_data"),
    pytest.param(lambda t: t, (1, 17, 32), id="declaration_wider"),
    pytest.param(lambda t: t, (2, 17, 16), id="declaration_taller"),
    pytest.param(lambda t: _sub(t, "<{value = 7 : index}>", "<{value = 6 : index}>"), (1, 17, 16), id="column_twice"),
    pytest.param(
        lambda t: _sub(t, '"tensor.extract"(%x, %6, %7, %54)', '"tensor.extract"(%x, %7, %6, %54)'),
        (1, 17, 16),
        id="swapped_leading_indices",
    ),
    pytest.param(
        lambda t: _sub(t, '%7 = "linalg.index"() <{dim = 1 : i64}>', '%7 = "linalg.index"() <{dim = 2 : i64}>'),
        (1, 17, 16),
        id="index_on_reduction_dim",
    ),
]


@pytest.mark.parametrize(("mutate", "shape"), MUTATIONS)
def test_mutations_are_refused(text, mutate, shape):
    with pytest.raises(rm.RowMomentsRefused):
        rm.prove(_generic(mutate(text)), _declaration(shape))


def test_digest_mismatch_is_refused(text):
    changed = _sub(text, "3.200000e+01 : f32", "3.300000e+01 : f32") if "3.200000e+01 : f32" in text else None
    if changed is None:
        changed = _sub(text, "1.600000e+01 : f32", "1.700000e+01 : f32", count=text.count("1.600000e+01 : f32"))
    with pytest.raises(rm.RowMomentsRefused, match="canonical body digest differs"):
        rm.prove(_generic(changed), _declaration())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["data_operand"].__setitem__("reduction_axis", 1),
        lambda b: b["data_operand"].__setitem__("shape", [17, 16]),
        lambda b: b["data_operand"].__setitem__("shape", [1, 17, 16, 2]),
        lambda b: b.__setitem__("extra", 1),
    ],
)
def test_bad_rank3_declarations_are_rejected(mutate):
    body = _declaration()
    mutate(body)
    with pytest.raises(ValueError, match="row moments needs"):
        rm.validate(body)
