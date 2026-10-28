"""``unsupported_mlir:<type>`` operand observations compare as the raw MLIR element type."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.software_spec import precision_disjoint

ROW = {
    "id": "host_embedding_gather",
    "ops": ["aten.embedding.default"],
    "families": ["elementwise_map"],
    "placement": "host",
    "signature": {
        "ordered_operand_dtypes": ["i64", "f32"],
        "ordered_result_dtypes": ["f32"],
        "ranks": [3],
    },
}
OBS = {
    "family": "elementwise_map",
    "ordered_operand_dtypes": ["i64", "f32"],
    "ordered_result_dtypes": ["f32"],
    "operand_dtype": "unsupported_mlir:i64",
    "readout_dtype": "f32",
    "rank": 3,
    "layout": "row_major_contiguous",
    "aliasing": "disjoint_inputs_outputs",
}


def _obs(**kw):
    o = copy.deepcopy(OBS)
    o.update(kw)
    return o


def test_observed_signature_is_not_disjoint():
    assert precision_disjoint(ROW, OBS) == []


@pytest.mark.parametrize(
    "kw",
    [
        {"ordered_operand_dtypes": ["i64", "i32"]},  # i32 table
        {"ordered_operand_dtypes": ["f32", "f32"]},  # f32 index
        {"operand_dtype": "unsupported_mlir:i32"},
        {"operand_dtype": "unsupported_mlir:f16"},
        {"ordered_result_dtypes": ["f16"]},
    ],
)
def test_variants_are_disjoint(kw):
    assert precision_disjoint(ROW, _obs(**kw)) != []


def test_only_operand_dtype_mismatch_is_reported_for_i32_token():
    reasons = precision_disjoint(ROW, _obs(operand_dtype="unsupported_mlir:i32"))
    assert reasons and all("operand_dtype" in r for r in reasons)


def test_prefix_is_stripped_only_for_unsupported_token():
    # an unprefixed observed i64 behaves identically to the prefixed one
    assert precision_disjoint(ROW, _obs(operand_dtype="i64")) == []
    # a bare prefix with nothing behind it is not an executable dtype -> proves nothing
    assert precision_disjoint(ROW, _obs(operand_dtype="unsupported_mlir:")) == []


def test_input_signature_not_mutated():
    o = _obs()
    precision_disjoint(ROW, o)
    assert o == OBS


# ------------------------------------------------------------------ real portable-v3 profile
PROFILE = Path(__file__).resolve().parents[3] / "examples/gemmini/platform/host-capabilities-portable-v3.yaml"


def test_real_profile_row_matches_fixture_and_agrees():
    doc = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    (row,) = [r for r in doc["operations"] if r["id"] == "host_embedding_gather"]
    assert row["signature"]["ordered_operand_dtypes"] == ["i64", "f32"]
    assert precision_disjoint(row, OBS) == []
    assert precision_disjoint(row, _obs(ordered_operand_dtypes=["i64", "i32"])) != []
    assert precision_disjoint(row, _obs(ordered_operand_dtypes=["f32", "f32"])) != []
    assert precision_disjoint(row, _obs(operand_dtype="unsupported_mlir:i32")) != []


# ------------------------------------------------------------------ end-to-end host admission
CAP = Path("/scratch/loren/c17-compute1/rootp/seal4/captures/fp32/causal_decoder/capture/linalg.mlir")


@pytest.fixture(scope="module")
def embedding_gather():
    if not CAP.is_file():
        pytest.skip("causal_decoder fp32 capture is unavailable")
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen.application_inventory import application_demand_inventory
    from merlin.targetgen.operation_accounting import _observed_signature

    inv = application_demand_inventory({"cd": CAP}, "gemmini", detailed=True, include_graph=True)
    row = next(
        r
        for r in inv["applications"]["cd"]["signatures"]
        if r["operation"] == "aten.embedding.default" and r["mlir_operation"] == "linalg.generic"
    )
    walked = tuple(mq.walk(mq.parse(normalize_capture_mlir(CAP.read_text())[0])))
    return row, _observed_signature(row), tuple(walked[i] for i in row["ordinals"])


def _admit(row, obs, src_ops):
    from merlin.targetgen.host_capabilities import admit_host_operation

    doc = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    sel = {
        "portable-v3": {
            "capability_spec": doc,
            "package_sha256": doc["compiler"]["package_sha256"],
            "capability_spec_sha256": "0" * 64,
            "dtype_strategy": doc["compiler"]["dtype_strategy"],
        }
    }
    return admit_host_operation(sel, row, obs, source_operations=src_ops)


def test_portable_v3_admits_observed_embedding_gather(embedding_gather):
    row, obs, src = embedding_gather
    assert obs["operand_dtype"].startswith("unsupported_mlir:")
    assert _admit(row, obs, src)["status"] == "admitted"


@pytest.mark.parametrize(
    "kw",
    [
        {"ordered_operand_dtypes": ["i64", "i32"]},
        {"ordered_operand_dtypes": ["f32", "f32"], "operand_dtype": "fp32"},
    ],
)
def test_portable_v3_refuses_table_and_index_dtype_variants(embedding_gather, kw):
    row, obs, src = embedding_gather
    assert _admit(row, _obs_from(obs, **kw), src)["status"] != "admitted"


def _obs_from(obs, **kw):
    o = copy.deepcopy(obs)
    o.update(kw)
    return o
