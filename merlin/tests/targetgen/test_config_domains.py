"""A config VALUE out of its hardware's range must be detectable before the RTL disagrees.

The derived encoder makes an illegal opcode and a use-before-config structurally impossible, and says
nothing about whether a value is right. That is the gap a whole failure class lives in: the program
decodes cleanly, every instruction is of the right class, the functional tiers agree, and only the RTL
disagrees. One submission passed L0/L1/L2 and collapsed to 1/23 on the RTL with nothing in between to
point at.

These pin the two properties that make the checker worth citing: bounds come from the target's OWN
facts with the fact quoted, and a field the facts cannot bound is REPORTED rather than passed.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import config_domains as CD


def test_a_memory_depth_bounds_a_row_index():
    d = CD.derive_domains("gemmini")
    row = d.get("scratchpad.row")
    if row is None:
        pytest.skip("this checkout's facts discover no scratchpad depth")
    assert row.lo == 0 and row.hi >= 1
    assert "depth" in row.evidence, "a bound must quote the fact that fixes it"


def test_an_out_of_range_value_is_a_violation_naming_its_evidence():
    d = CD.derive_domains("gemmini")
    if "scratchpad.row" not in d:
        pytest.skip("no scratchpad row domain in this checkout")
    hi = d["scratchpad.row"].hi
    r = CD.check("gemmini", {"scratchpad.row": hi + 1})
    assert len(r["violations"]) == 1
    v = r["violations"][0]
    assert v["domain"] == [0, hi] and "depth" in v["evidence"]


def test_an_in_range_value_passes():
    d = CD.derive_domains("gemmini")
    if "scratchpad.row" not in d:
        pytest.skip("no scratchpad row domain in this checkout")
    r = CD.check("gemmini", {"scratchpad.row": d["scratchpad.row"].hi})
    assert r["violations"] == [] and len(r["ok"]) == 1


def test_a_field_with_no_derived_domain_is_reported_not_accepted():
    """The property that makes this citable: silence must not read as approval."""
    r = CD.check("gemmini", {"a.field.nothing.bounds": 1 << 40})
    assert r["violations"] == []
    assert [u["field"] for u in r["unbounded"]] == ["a.field.nothing.bounds"]


def test_a_target_whose_facts_bound_nothing_says_so():
    """A checker that returns a clean bill for a target it cannot check is worse than none."""
    gaps = CD.undecidable("saturn_opu")
    doms = CD.derive_domains("saturn_opu")
    if doms:
        pytest.skip("this checkout discovered facts for that target")
    assert gaps, "no domains AND no reported gaps would be a silent pass"


def test_integer_dtype_ranges_are_parsed_from_the_width_not_tabulated():
    assert CD._int_dtype_range("i8") == (-128, 127)
    assert CD._int_dtype_range("u8") == (0, 255)
    assert CD._int_dtype_range("i32") == (-(1 << 31), (1 << 31) - 1)


def test_a_non_integer_dtype_yields_no_bound_rather_than_a_guess():
    for t in ("bf16", "fp8_e4m3", "f32", "", "iX", "i0", "i999"):
        assert CD._int_dtype_range(t) is None, t


def test_an_array_bounds_an_index_not_a_tile_dimension():
    """A tile larger than the mesh is legal -- it gets tiled -- so the array bound must be named as an
    index bound. Mis-naming it would reject correct programs."""
    d = CD.derive_domains("gemmini")
    if "mesh.row" not in d:
        pytest.skip("no mesh array in this checkout's facts")
    assert d["mesh.row"].unit == "index"
    assert not any(k.endswith(".tile") or "tile" in k for k in d), "no tile-dimension domain may be derived"
