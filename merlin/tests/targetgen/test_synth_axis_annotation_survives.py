"""A synthesized capsule must carry the AXIS it was synthesized for, all the way to disk.

⚠️ REGRESSION, and it was total. `corpus_synth` annotated every derived axis as
``entry["semantic"]["generalization_axis"]``, and the generator reads the intent from
``entry["generalization"]`` -- ``semantic`` is already taken, for a free-form op-semantics label that
flows into ``operation.attributes.semantic``. So all EIGHT annotations were discarded and every
synthesized capsule was labelled by kind alone: measured on a regenerated corpus, 163 capsules read
``composition`` and not one read ``application``, ``conv_window``, ``epilogue`` or ``host_lane``.

The corpus-wide ``generalization_axis`` census was therefore blind to every derived axis -- it did not
report them as empty, it reported them as something else, which is worse.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import corpus_synth as CS

_SPEC = {
    "target": "t",
    "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
               "alignment": "aligned"}],
    "boundaries": {"tile_edge": 16,
                   "extent_probes": [{"boundary": "tile_edge", "edge": 16,
                                      "points": [1, 4, 8, 15, 16, 17, 32]}]},
    "epilogue": {"required": [{"stage": "relu", "family": "elementwise_map", "isa_classes": [],
                               "evidenced_by": ["manifest_composed_with"]}]},
    "conv_geometry": {"required": [
        {"signature": "k3x3/s2x2/d1x1/pad1x1", "kernel": [3, 3], "stride": [2, 2],
         "dilation": [1, 1], "pad_before": [1, 1], "pad_after": [1, 1], "pad_known": True,
         "n_regions": 6, "sources": ["a_capture"]}]},
}


def _entries():
    out = CS.synthesize(_SPEC, workload_spec={"models": [], "precision_preference": ["int8"]})
    return out["capsules"] if isinstance(out, dict) else out


def test_no_axis_annotation_uses_the_colliding_semantic_key():
    """`semantic` on a profile entry means an OP-semantics label ("quantized_linear"), not a
    generalization intent. Twelve existing entries use it that way; writing a dict there both loses the
    annotation and risks colliding with them."""
    import inspect

    src = inspect.getsource(CS)
    assert '"semantic": {"generalization_axis"' not in src, (
        "an axis annotated under `semantic` is silently discarded by the generator")
    assert '"generalization": {"generalization_axis"' in src


@pytest.mark.parametrize("axis", ["epilogue", "conv_window"])
def test_the_entry_declares_its_axis_under_the_key_the_generator_reads(axis):
    got = [e for e in _entries()
           if (e.get("generalization") or {}).get("generalization_axis") == axis]
    assert got, f"no entry annotated {axis!r} under `generalization`"
    for e in got:
        assert "semantic" not in e or not isinstance(e.get("semantic"), dict), (
            "the op-semantics key must not be reused for generalization intent")


def test_the_declared_obligation_reference_survives_into_the_capsule():
    """A `padUNKNOWN` window cannot be matched by re-deriving a signature from the member's
    attributes -- the member can only spell zero padding, so it would re-derive as `pad0x0` and the
    obligation it was built for could never be satisfied by it. The reference has to survive."""
    conv = [e for e in _entries()
            if (e.get("generalization") or {}).get("generalization_axis") == "conv_window"]
    assert conv
    assert conv[0]["generalization"]["conv_window"] == "k3x3/s2x2/d1x1/pad1x1"


def test_an_authored_key_the_semantic_block_does_not_consume_is_not_dropped():
    """The block consumed the keys it knew and silently dropped the rest, which is why a declared
    obligation reference could not reach disk."""
    from merlin.targetgen import corpus_spec as CSp

    src = __import__("inspect").getsource(CSp._semantic_block) if hasattr(CSp, "_semantic_block") \
        else __import__("inspect").getsource(CSp)
    assert "for key, value in authored.items()" in src, (
        "authored keys outside the known set must be carried, not discarded")
