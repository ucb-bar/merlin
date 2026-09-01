"""A block-scale operand must not decide a capsule's dtype cell or its tile alignment.

`role: scale` is a shared-exponent stream — one e8m0 per fixed-length group of the operand it scales
(`scale_of`). It is metadata ABOUT the compute, not the compute. Counting it in
`cert_capsule_cover`'s two axes did two measurable things on 2026-09-01:

* invented cells named for the scale's own dtype (`contraction/e8m0/partial`), which no capsule can
  ever be "about"; and
* made every microscaling capsule PERMANENTLY `partial`. A scale plane is `[K/group, M]`, deliberately
  small — for radiance `[1, 16]` against tile 16, and `1 % 16 != 0` — so
  `contraction|attention/mxfp{4,6,8}/aligned` sat in the requirement and was uncoverable BY
  CONSTRUCTION: 6 cells no author could ever close.
"""
from __future__ import annotations

import json

from merlin.targetgen.contract import materialize as MZ


def _capsule(tmp_path, name: str, inputs: list[dict], *, family="contraction"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(json.dumps({
        "name": name, "kind": "isa", "label": "public",
        "operation": {"op": "matmul"},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": []},
        "required_oracle_tiers": ["L2"],
        "semantic": {"semantic_family": family},
        "inputs": inputs,
    }), encoding="utf-8")
    return d


def _cells(tmp_path, tile: int):
    """The (family, dtype, alignment) cells cert_capsule_cover attributes to this corpus."""
    cover = MZ.cert_capsule_cover([tmp_path], labels={"public", "dev"}, tile_dim=tile,
                                  exclude=set())
    return {str(c) for c in (cover or {}).get("cells") or []}


def test_a_scale_plane_does_not_make_a_whole_tile_capsule_partial(tmp_path):
    """The exact radiance geometry: tile-aligned operands plus a [1, 16] e8m0 scale plane."""
    _capsule(tmp_path, "R_mx_tile", [
        {"name": "A", "role": "input", "shape": [16, 32], "dtype": "mxfp8"},
        {"name": "W", "role": "weight", "shape": [32, 16], "dtype": "mxfp8"},
        {"name": "S", "role": "scale", "shape": [1, 16], "dtype": "e8m0",
         "scale_of": "A", "block": 32},
    ])
    cells = _cells(tmp_path, 16)
    assert any(c.endswith("/aligned") for c in cells), (
        f"a capsule whose COMPUTE operands are whole tiles was classified only partial; the [1, 16] "
        f"scale plane decided it. cells={sorted(cells)}")


def test_the_scale_dtype_does_not_become_a_cell(tmp_path):
    """`e8m0` is the scale's storage, never a compute dtype a capsule can be authored for."""
    _capsule(tmp_path, "R_mx_tile", [
        {"name": "A", "role": "input", "shape": [16, 32], "dtype": "mxfp8"},
        {"name": "S", "role": "scale", "shape": [1, 16], "dtype": "e8m0",
         "scale_of": "A", "block": 32},
    ])
    cells = _cells(tmp_path, 16)
    assert not any("e8m0" in c for c in cells), (
        f"the scale's own dtype became a required cell nothing can cover. cells={sorted(cells)}")


def test_a_genuinely_ragged_compute_operand_is_still_partial(tmp_path):
    """The fix must not make everything aligned: a real ragged extent still reads partial."""
    _capsule(tmp_path, "R_mx_tail", [
        {"name": "A", "role": "input", "shape": [15, 32], "dtype": "mxfp8"},
        {"name": "S", "role": "scale", "shape": [1, 15], "dtype": "e8m0",
         "scale_of": "A", "block": 32},
    ])
    cells = _cells(tmp_path, 16)
    assert any(c.endswith("/partial") for c in cells), (
        f"a ragged COMPUTE extent (15 vs tile 16) must still be partial. cells={sorted(cells)}")
