"""C8 selection front-end: resolve a human "profile this section" spec -> prov.region_id set.

Runs against the committed real-architecture captures. Assertions are self-referential (derived from
the recognized menu itself, not hard-coded model shapes) so the test does not overfit to any workload.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.dse_guidance import section_select as S
from merlin.dse_guidance.attribution import REGION_ATTENTION

_RECAP = merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures"
_MODELS = [m for m in ("tiny_llama", "small_llama", "openvla")
           if (_RECAP / m / "model.mlir").is_file()]
pytestmark = pytest.mark.skipif(not _MODELS, reason="section-select corpus not present")


def _cap(m: str) -> str:
    return str(_RECAP / m)


def test_list_sections_is_the_menu():
    cap = _cap(_MODELS[0])
    secs = S.list_sections(cap)
    assert secs, "expected a non-empty section menu"
    assert any(s.label == REGION_ATTENTION for s in secs)     # attention is a selectable section
    assert all(s.region_ids for s in secs)                    # every listed section owns ≥1 region_id


def test_whole_selects_every_region():
    cap = _cap(_MODELS[0])
    all_ids = {rid for s in S.list_sections(cap) for rid in s.region_ids}
    assert S.resolve(cap, "whole") == all_ids
    assert S.resolve(cap, None) == all_ids
    assert S.resolve(cap, "*") == all_ids


def test_exact_region_id_and_fqn_substring():
    cap = _cap(_MODELS[0])
    secs = S.list_sections(cap)
    one = secs[0]
    rid = one.region_ids[0]
    assert S.resolve(cap, rid) == {rid}                       # exact region_id
    # an fqn substring selects (at least) that section's regions.
    assert set(one.region_ids) <= S.resolve(cap, one.fqn)


def test_attention_only_selection_is_a_strict_subset():
    cap = _cap(_MODELS[0])
    secs = S.list_sections(cap)
    attn = {rid for s in secs if s.label == REGION_ATTENTION for rid in s.region_ids}
    allids = {rid for s in secs for rid in s.region_ids}
    got = S.resolve(cap, "fqn:*attn*")
    assert attn <= got                                        # captures the attention regions
    assert got < allids                                       # but not the whole model


def test_layer_range_selection():
    cap = _cap(_MODELS[0])
    secs = S.list_sections(cap)
    # find any layer index present in this capture.
    idxs = sorted({li for s in secs if (li := S._layer_index(s.fqn)) is not None})
    if not idxs:
        pytest.skip("capture has no layer-indexed sections")
    lo = idxs[0]
    want = {rid for s in secs if S._layer_index(s.fqn) == lo for rid in s.region_ids}
    assert want <= S.resolve(cap, f"layers:{lo}")
    # a range spanning all present layers selects every layer-indexed region.
    ranged = S.resolve(cap, f"layers:{idxs[0]}-{idxs[-1]}")
    all_layered = {rid for s in secs if S._layer_index(s.fqn) is not None for rid in s.region_ids}
    assert all_layered <= ranged


def test_bad_selection_fails_closed():
    cap = _cap(_MODELS[0])
    with pytest.raises(ValueError):
        S.resolve(cap, "no_such_section_xyz")
