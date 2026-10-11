"""Neutral corpus roles cannot borrow a legacy schedule or omit an operation slot."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from merlin.targetgen.corpus_spec import _classes_source


def contract():
    roster = [
        {"class": name, "funct": i, "operands": {}}
        for i, name in enumerate(["DATA_IN", "WEIGHT_STAGE", "PRODUCT", "DATA_OUT"])
    ]
    return {
        "rocc_operand_roles": {"instructions": roster},
        "corpus_instruction_roles": {
            "version": 1,
            "instruction_roster_sha256": hashlib.sha256(
                json.dumps(roster, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "by_role": {
                "memory": ["DATA_IN"],
                "weight_load": ["WEIGHT_STAGE"],
                "matmul": ["PRODUCT"],
                "acc_readout": ["DATA_OUT"],
            },
        },
    }


def test_source_classes_match_without_legacy_issue_order():
    c = contract()
    select = _classes_source(SimpleNamespace(target="synthetic"), c)
    assert select(op="matmul") == ["DATA_IN", "WEIGHT_STAGE", "PRODUCT", "DATA_OUT"]
    assert select(op="movement", movement=True) == ["DATA_IN"]
    c["corpus_instruction_roles"]["by_role"]["matmul"][0] = "CHANGED"
    assert select(op="matmul")[2] == "PRODUCT"


@pytest.mark.parametrize("mutation", ["digest", "class", "role", "version", "roster"])
def test_changed_or_undeclared_selection_refuses(mutation):
    c = contract()
    s = c["corpus_instruction_roles"]
    if mutation == "digest":
        s["instruction_roster_sha256"] = "0" * 64
    elif mutation == "class":
        s["by_role"]["matmul"] = ["UNDECLARED"]
    elif mutation == "role":
        s["by_role"]["invented"] = ["PRODUCT"]
    elif mutation == "version":
        s["version"] = True
    else:
        c["rocc_operand_roles"]["instructions"][2]["funct"] = 99
    with pytest.raises(ValueError):
        _classes_source(SimpleNamespace(target="synthetic"), c)


def test_missing_operation_role_is_not_satisfied_by_memory():
    c = contract()
    del c["corpus_instruction_roles"]["by_role"]["matmul"]
    select = _classes_source(SimpleNamespace(target="synthetic"), c)
    with pytest.raises(ValueError, match="lack required role slot"):
        select(op="matmul")
    with pytest.raises(ValueError, match="lack required role slot"):
        select(op="softmax")


def test_unselected_legacy_route_still_requires_its_declaration():
    with pytest.raises(ValueError, match="corpus_issue_order"):
        _classes_source(SimpleNamespace(target="synthetic"), {}, taxonomy={})


def test_fused_epilogue_is_not_a_general_compute_role():
    c = contract()
    c["corpus_instruction_roles"]["epilogue_roles"] = {
        "relu": {"tensor_compute_unary": ["DATA_OUT"]},
        "bias": {"tensor_compute_binary": ["WEIGHT_STAGE"]},
    }
    select = _classes_source(SimpleNamespace(target="synthetic"), c)
    assert select(op="matmul", epilogue=("relu", "bias")) == ["DATA_IN", "WEIGHT_STAGE", "PRODUCT", "DATA_OUT"]
    for op in ("softmax", "residual_add"):
        with pytest.raises(ValueError, match="lack required role slot"):
            select(op=op, epilogue=("relu", "bias"))


def test_operation_roles_are_explicit_and_scoped():
    c = contract()
    c["corpus_instruction_roles"].update(version=2, operation_roles={
        "residual_add": {"tensor_compute_binary": ["DATA_IN", "DATA_OUT"]},
    })
    select = _classes_source(SimpleNamespace(target="synthetic"), c)
    assert select(op="residual_add") == ["DATA_IN", "DATA_OUT"]
    with pytest.raises(ValueError, match="lack required role slot"):
        select(op="softmax")
    c["corpus_instruction_roles"]["operation_roles"]["residual_add"]["tensor_compute_binary"][0] = "CHANGED"
    assert select(op="residual_add") == ["DATA_IN", "DATA_OUT"]


def test_operation_roles_cannot_change_legacy_selection():
    c = contract()
    c["corpus_instruction_roles"]["operation_roles"] = {
        "residual_add": {"tensor_compute_binary": ["DATA_IN"]},
    }
    with pytest.raises(ValueError, match="require.*version 2"):
        _classes_source(SimpleNamespace(target="synthetic"), c)
