"""An axis that is DERIVED, written into every spec, and measured by nothing is not a requirement.

Three axes were in that state. `host_lane` carried 12 obligations on one target and `epilogue` four,
both emitted by `conformance.spec` and read back by `uncovered()` nowhere; and the corpus-side
evidence for the negative lane read a whole model's entry tensor as an operand dtype, minting a
`contraction/i64` obligation for arithmetic nothing performs.
"""
from __future__ import annotations

import yaml

from merlin.targetgen import conformance as CF


def _capsule(tmp_path, name, **doc):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    base = {"name": name, "label": "public"}
    base.update(doc)
    (d / "capsule.yaml").write_text(yaml.safe_dump(base))
    return d


# --------------------------------------------------------------- the entry-tensor misread


def test_a_whole_models_entry_tensor_is_not_read_as_an_operand_dtype(tmp_path):
    """⚠️ REGRESSION. A language model is entered with a token-id vector -- i64 -- and computes in
    none of it; it declares its arithmetic separately. Reading `inputs[]` as operand dtypes minted a
    `contraction/i64` host-lane obligation on all three targets, for work no capsule performs and no
    hardware was ever asked to take. Measured: 20 of the 24 capsules declaring a compute dtype
    disagreed with their own `inputs[]`, and every one was `kind: model`."""
    _capsule(tmp_path, "M0_whole_model",
             kind="model",
             semantic={"semantic_family": "contraction"},
             operation={"op": "model", "attributes": {"dtype": "i8"}},
             inputs=[{"role": "input", "dtype": "i64", "shape": [1, 128]}])
    pairs = CF.corpus_presented_pairs([tmp_path])
    assert ("contraction", "i64") not in pairs, "the entry tensor is not an operand dtype"
    assert pairs[("contraction", "i8")] == 1, "the DECLARED compute dtype is what the model computes in"


def test_a_capsule_declaring_no_compute_dtype_still_reports_its_operands(tmp_path):
    """The fix must not blind the ordinary case: a single-region capsule declares no separate compute
    dtype BECAUSE its entry tensor is its operand."""
    _capsule(tmp_path, "A0_matmul",
             semantic={"semantic_family": "contraction"},
             operation={"op": "matmul", "attributes": {}},
             inputs=[{"role": "input", "dtype": "bf16", "shape": [16, 16]},
                     {"role": "weight", "dtype": "bf16", "shape": [16, 16]}])
    pairs = CF.corpus_presented_pairs([tmp_path])
    assert pairs[("contraction", "bf16")] == 2


# --------------------------------------------------------------- the epilogue axis


def _epi(*stages):
    return [{"stage": s, "family": "elementwise_map", "isa_classes": [],
             "evidenced_by": ["manifest_composed_with"]} for s in stages]


def test_an_epilogue_stage_no_capsule_demands_is_reported_uncovered(tmp_path):
    _capsule(tmp_path, "A5_relu",
             semantic={"semantic_family": "contraction"},
             operation={"op": "matmul", "attributes": {"epilogue": ["relu"]}},
             inputs=[{"role": "input", "dtype": "i8", "shape": [16, 16]}])
    gap = CF._epilogue_gap(_epi("relu", "requant"), [tmp_path])
    assert gap["status"] == "ok"
    assert gap["uncovered"] == ["requant"], (
        "a stage the ABI declares and no capsule demands means a backend that cannot emit it fails "
        "nothing here")
    assert gap["fused_by"]["relu"] == ["A5_relu"]


def test_a_standalone_stage_counts_and_is_kept_apart_from_a_fused_one(tmp_path):
    """A bare requant member catches a backend with NO lowering for the stage; a fused one catches a
    backend that cannot fuse it. Summing them would let either stand in for the other."""
    _capsule(tmp_path, "R0_requant",
             semantic={"semantic_family": "elementwise_map"},
             operation={"op": "requant", "attributes": {}},
             inputs=[{"role": "input", "dtype": "i32", "shape": [16, 16]}])
    gap = CF._epilogue_gap(_epi("requant"), [tmp_path])
    assert gap["uncovered"] == []
    assert gap["standalone_by"]["requant"] == ["R0_requant"]
    assert gap["standalone_only"] == ["requant"], (
        "evidenced standalone only: the lowering is tested and the FUSION is not")
    assert gap["fused_only"] == []


def test_a_stale_spec_with_no_epilogue_block_reads_not_measured_never_covered(tmp_path):
    gap_doc = CF.uncovered({"target": "t", "cells": []}, [tmp_path])
    assert gap_doc["epilogue"]["status"] == "not_measured", (
        "an axis a stale spec cannot express must not read as an axis with no gaps")
    assert "regenerate" in gap_doc["epilogue"]["detail"]


# --------------------------------------------------------------- the host-lane axis is wired


def test_uncovered_reports_the_host_lane_axis_at_all(tmp_path):
    """The narrow point of the whole change: `uncovered()` must SAY something about host_lane. It
    derived the obligation and returned a report that never mentioned it."""
    doc = {"target": "t", "cells": [],
           "host_lane": {"required": [{"family": "movement", "dtype": "f32", "n_regions": 4}]}}
    rep = CF.uncovered(doc, [tmp_path])
    assert "host_lane" in rep
    assert rep["host_lane"]["status"] == "ok"
    assert rep["host_lane"]["n_required"] == 1
    assert rep["host_lane"]["uncovered"] == ["movement/f32"]


def test_a_spec_with_no_host_lane_block_is_not_measured(tmp_path):
    rep = CF.uncovered({"target": "t", "cells": []}, [tmp_path])
    assert rep["host_lane"]["status"] == "not_measured"
