"""Seed overflow guard host row: closed proof tied to the consuming contraction's K, proposal status, refusals.

The fixture is the real model2MLIR capture of ``m2m_q.int_mm_seeded(a[4,8], w[8,6], seed[6])``: the rank-1 guard
generic, the ``linalg.broadcast`` and the seeded contraction. Every mutation is a text substitution guarded by an
exactly-once count, so a refusal can only come from the single deliberate edit.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.common import mlir_query as mq
from merlin.frontends import linalg_seed_guard_source_body as sg
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "examples" / "gemmini" / "platform" / "host-capabilities-portable-v3.yaml"
FIXTURE = Path(__file__).parent / "fixtures" / "int_mm_seeded_k8.mlir"
ROW_ID = "host_i32_int_mm_seed_overflow_guard_rank1"
LOW, HIGH = "-2147352576", "2147352575"  # -2^31 + 8*16384, 2^31 - 1 - 8*16384
OBSERVED = {"family": "elementwise_map", "ordered_operand_dtypes": ["i32", "i32"], "ordered_result_dtypes": ["i32"], "rank": 1}
SOURCE_ROW = {"frontend_op": "m2m_q.int_mm_seeded.default", "mlir_operation": "linalg.generic", "count": 1}


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(PROFILE.read_bytes())


@pytest.fixture(scope="module")
def text():
    return FIXTURE.read_text()


def _guard(source: str):
    return next(op for op in mq.walk(mq.parse(source), "linalg.generic") if len(op.operands) == 2 and len(list(op.regions[0].blocks[0].ops)) == 8)


def _sub(source: str, old: str, new: str, *, count: int = 1) -> str:
    assert source.count(old) == count, (old, source.count(old))
    return source.replace(old, new)


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
    row["status"] = "reviewed"
    return document


def _demoted(document, *, contract_too=True):
    document = copy.deepcopy(document)
    row = next(r for r in document["operations"] if r["id"] == ROW_ID)
    row["status"] = "unreviewed"
    if contract_too:
        row["numerical_contract"]["status"] = "unreviewed"
    return document


def _admit(document, ops):
    return admit_host_operation(_selection(document), dict(SOURCE_ROW), dict(OBSERVED), source_operations=ops)


def test_row_is_operator_reviewed_with_open_obligations_kept(shipped):
    validate_host_capabilities(shipped)
    row = next(r for r in shipped["operations"] if r["id"] == ROW_ID)
    assert row["status"] == "reviewed" and row["numerical_contract"]["status"] == "reviewed"
    reviewed_by = row["numerical_contract"]["evidence"]["reviewed_by"]
    assert reviewed_by.startswith("Reviewed by Loren Hung (approved in operator chat 2026-10-28)")
    assert "cf.assert inside a linalg.generic body is not proved" in reviewed_by
    assert "no Spike or host execution" in reviewed_by
    assert ROW_ID in shipped["review"]["operator_chat_addendum_2"]


def test_real_capture_guard_is_proved_against_its_contraction_k(text):
    proof = sg.prove(_guard(text))
    assert proof["seed_length"] == 6 and proof["reduction_extent"] == 8
    assert proof["half_width"] == 8 * 128 * 128 and proof["interval"] == [int(LOW), int(HIGH)]


def test_shipped_reviewed_row_admits_and_demotion_withdraws_the_admission(shipped, text):
    after = _admit(shipped, (_guard(text),))
    assert after["status"] == "admitted" and after["reviewed"] is True
    for document in (_demoted(shipped), _demoted(shipped, contract_too=False)):
        before = _admit(document, (_guard(text),))
        assert not (before["status"] == "admitted" and before["reviewed"] is True)


@pytest.mark.parametrize(
    "old,new",
    [
        (LOW, "-2147352577"),  # lower bound not matching K
        (HIGH, "2147352574"),  # upper bound asymmetric
        (LOW, "-2147483648"),  # no guard margin at all
        ('msg = "seeded contraction may overflow its int32 accumulator"', 'msg = "another message"'),
        ("<{predicate = 5 : i64}>", "<{predicate = 4 : i64}>"),  # sgt instead of sge
        ("<{predicate = 3 : i64}>", "<{predicate = 2 : i64}>"),  # slt instead of sle
    ],
)
def test_mutated_guards_are_refused_even_when_the_row_is_reviewed(shipped, text, old, new):
    mutated = _sub(text, old, new)
    verdict = _admit(_reviewed(shipped), (_guard(mutated),))
    assert verdict["status"] != "admitted"
    assert "seed guard structural proof refused" in str(verdict["profiles"][0]["decisions"])


def test_guard_must_be_bound_to_the_k_of_the_contraction_it_feeds(shipped, text):
    # Same guard, but the contraction reduces over K=16: the guard margin (K=8) no longer covers it.
    mutated = text.replace("tensor<4x8xi8>", "tensor<4x16xi8>").replace("tensor<8x6xi8>", "tensor<16x6xi8>")
    assert _admit(_reviewed(shipped), (_guard(mutated),))["status"] != "admitted"


def test_guard_that_does_not_feed_the_broadcast_is_refused(shipped, text):
    other = _sub(text, '"linalg.broadcast"(%4, %13)', '"linalg.broadcast"(%0, %13)')
    verdict = _admit(_reviewed(shipped), (_guard(other),))
    assert verdict["status"] != "admitted"
    assert "exactly one use" in str(verdict["profiles"][0]["decisions"])


def test_missing_source_operations_are_unknown_not_evidence(shipped):
    assert _admit(_reviewed(shipped), None)["status"] != "admitted"
