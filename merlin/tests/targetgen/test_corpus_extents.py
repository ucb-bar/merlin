"""A sweep axis resolves to a BARE name, and a builder that reads only ``_tiles`` drops it silently.

``expand_sweeps`` resolves ``axes: {K: ["4*tile", "8*tile"]}`` into ``entry["K"] = 64`` and
``entry["K"] = 128``. Two builders read only ``entry["K_tiles"]``, defaulted it to 1, and emitted the
tile edge for every point. Reproduced 2026-09-04: PC00_k64 and PC01_k128 both came out as a
``[tile, tile]`` weight -- byte-identical interfaces differing only in a name and a prose string.

That is the worst shape a corpus defect can take. The family's gate demands two separation regimes,
and it would have had one regime wearing two labels, so the paired differential built on it would
have measured the same program twice and reported agreement. Nothing failed; the names read right.

This pins the property directly: a declared extent must reach the emitted shape, whichever spelling
declared it, and two points of one axis must not collapse onto each other.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import corpus_spec as CS

TILE = 16


def _binding():
    return CS.CorpusBinding(target="t", tile_dim=TILE, operand_dtype="int8", accum_dtype="int32",
                            integer=True, tiers=["L2", "L3"], compare="exact")


def _weight_shape(mlir: str) -> str:
    for line in mlir.splitlines():
        if 'role = "weight"' in line:
            return line.rsplit(":", 1)[-1].strip()
    raise AssertionError("the built interface declares no weight tensor")


def _resident_entry(**over):
    entry = {"cat": "_perf", "name": "X", "kind": "model_slice", "op": "resident_reuse",
             "weight": "W", "label": "dev", "source_role": "derived_sweep",
             "source": "b", "source_reference": "r",
             "matmuls": [{"lhs": "A0", "out": "Y0", "M_tiles": 1},
                         {"lhs": "A1", "out": "Y1", "M_tiles": 1}]}
    entry.update(over)
    return entry


@pytest.mark.parametrize("spelling", ["bare", "tiles"])
def test_a_declared_extent_reaches_the_emitted_shape(spelling):
    """Both spellings must work; reading one and ignoring the other is the defect."""
    over = {"K": 4 * TILE, "N": 4 * TILE} if spelling == "bare" else {"K_tiles": 4, "N_tiles": 4}
    _doc, mlir = CS.build_resident_reuse(_resident_entry(**over), _binding())
    assert _weight_shape(mlir).startswith(f"tensor<{4 * TILE}x{4 * TILE}x"), (
        f"the {spelling} spelling did not reach the emitted weight shape")


def test_two_points_of_one_axis_do_not_collapse_onto_each_other():
    """The exact corruption: distinct declared depths emitting one identical program."""
    binding = _binding()
    shapes = set()
    for depth in (4 * TILE, 8 * TILE):
        _doc, mlir = CS.build_resident_reuse(
            _resident_entry(K=depth, N=4 * TILE), binding)
        shapes.add(_weight_shape(mlir))
    assert len(shapes) == 2, (
        f"two declared contraction depths emitted one shape {shapes}; a fit over them would "
        f"measure the same program twice and call it agreement")


def test_the_attention_builder_reads_both_spellings_too():
    entry = {"cat": "_perf", "name": "X", "kind": "model_slice", "op": "attention_qk",
             "label": "dev", "source_role": "derived_sweep", "source": "b",
             "source_reference": "r", "M": 2 * TILE, "K": 4 * TILE}
    _doc, mlir = CS.build_attention_qk(entry, _binding())
    assert f"{2 * TILE}x{4 * TILE}" in mlir, (
        "the bare M/K spelling did not reach the emitted attention shapes")
