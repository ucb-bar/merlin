"""Every unresolved row the software-spec resolver emits becomes a diagnostic with a deterministic reason.

WHY THIS EXISTS. The derivation loop in phase0/evidence.py read ``row["reason"]`` from every unresolved row, but only
operation and format rows carry one; scheme, readout and composition rows are ``{field, value: "UNKNOWN"}`` (with an
optional owner). The first real derive on a spec that declares quantization schemes died with a KeyError. The
producer-coverage test below feeds the helper the rows the REAL resolver emits on the REAL spec, so a new row shape
fails here instead of in the middle of a release run.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from merlin.common.paths import repo_root
from merlin.targetgen import spec_fact_drift
from merlin_experiments.phase0.evidence import _unresolved_reason


def test_rows_with_a_reason_keep_it_verbatim():
    row = {"id": "op", "family": "matmul", "form": "f", "status": "unresolved", "reason": "no such form"}
    assert _unresolved_reason(row) == "no such form"
    assert _unresolved_reason({"format": "i8__i32", "reason": "nothing covers it"}) == "nothing covers it"


@pytest.mark.parametrize(
    ("row", "text"),
    [
        ({"field": "readout.activation_order", "value": "UNKNOWN"},
         "readout.activation_order is UNKNOWN: the selected facts do not establish it"),
        ({"scheme": "q_f32_tensor", "field": "columns_per_scale", "value": "UNKNOWN"},
         "scheme q_f32_tensor: columns_per_scale is UNKNOWN: the selected facts do not establish it"),
        ({"composition": "c1", "field": "k_extension", "value": "UNKNOWN"},
         "composition c1: k_extension is UNKNOWN: the selected facts do not establish it"),
    ],
)
def test_field_rows_get_a_reason_from_their_own_keys(row, text):
    assert _unresolved_reason(row) == text
    assert _unresolved_reason(dict(reversed(list(row.items())))) == text  # key order does not matter


@pytest.mark.parametrize(
    "row",
    [{}, {"foo": 1}, {"field": "x"}, {"field": "x", "value": "KNOWN"}, {"value": "UNKNOWN"}, {"field": "", "value": "UNKNOWN"},
     {"reason": ""}, {"reason": 3}, {"scheme": "s", "value": "UNKNOWN"}],
)
def test_an_unrecognised_row_shape_refuses(row):
    with pytest.raises(ValueError, match="software-spec derivation row"):
        _unresolved_reason(row)
