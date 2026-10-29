"""Candidate evidence cannot discard stronger domains, premises or actual-prefix binding."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_lean_w8a8_numerical import evidence, invoke

from merlin.verify.lean import numerical_evidence


@pytest.fixture
def qualified(tmp_path, lean_tools, lean_oracle):
    work = tmp_path
    receipt, _ = invoke(work, lean_tools)
    report = evidence(receipt)
    before = json.loads(Path(report["program_exports"][0]["path"]).read_text())
    return report, before


@pytest.mark.parametrize(
    "changed",
    [
        "review",
        "concrete_context",
        "agreement",
        "domain",
        "theorem",
        "boolean_count",
        "unbound_quantizer",
        "unbound_dequantizer",
        "unchecked_prefix",
        "intervals",
        "duplicate_function",
    ],
)
def test_candidate_scope_and_prefix_binding_cannot_be_forged(qualified, changed):
    report, before = qualified
    candidate = copy.deepcopy(report["numerical"])
    function = candidate["functions"][0]
    prefix = function["source_prefix"]
    if changed == "review":
        candidate["whole_pass_verified"] = True
    elif changed == "concrete_context":
        candidate["concrete_context_inhabited"] = True
    elif changed == "agreement":
        candidate["arithmetic_agreement_assumed"] = False
    elif changed == "domain":
        candidate["input_domain"] = "finite_f32_and_declared_integer_intervals_per_function"
    elif changed == "theorem":
        candidate["theorem"] = "MerlinVerify.Check.PolicyW8A8BoundedChain.check_sound"
    elif changed == "boolean_count":
        prefix["operation_count"] = True
    elif changed == "unbound_quantizer":
        prefix["initialized_source_quantizers"] = [10**9]
    elif changed == "unbound_dequantizer":
        prefix["finite_source_dequantizers"] = [10**9]
    elif changed == "unchecked_prefix":
        prefix["checked"] = False
    elif changed == "intervals":
        function["input_intervals"] = []
    elif changed == "duplicate_function":
        candidate["functions"].append(copy.deepcopy(function))
    with pytest.raises(ValueError):
        numerical_evidence.validate(
            candidate,
            bounds=report["input_contracts"]["entries"],
            before=before,
            index_bits=64,
            scope="captured",
            suffix_count=0,
        )
