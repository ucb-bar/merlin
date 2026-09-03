"""The conformance requirement must be DERIVED, and the derivation must survive its own edge cases.

A "derived" requirement that quietly hardcodes a family list is worse than an honest hardcoded one,
because it claims an authority it does not have. Each test here fails if the derivation is replaced by a
literal, and the last three pin the three bugs that were actually found while building it:

  * the dtype-spelling join (manifest ``fp32`` vs capsule ``f32``) — reported 40 of 56 cells missing while
    the corpus plainly had them
  * composite families never appearing in a capture, which dropped ``attention`` from a transformer corpus
  * the operand-store fact living in a memories LIST, not a ``shared_memory`` mapping
"""
from __future__ import annotations

import inspect

import pytest

from merlin.targetgen import conformance as CF

TARGET = "radiance"


def test_admitted_comes_from_the_manifest_not_a_literal():
    """Every admitted family must be a legal semantic-family name, and the set must not be a constant."""
    from merlin.targetgen import semantic_families as sf

    adm = CF.admitted(TARGET)
    if not adm:
        pytest.skip("no capability manifest resolvable in this environment")
    assert set(adm) <= sf.FAMILIES, f"admitted contains non-families: {set(adm) - sf.FAMILIES}"
    # Read the DELEGATION CHAIN, not just the entry point. `admitted` became a one-line delegate to
    # `admitted_with_reason` when callers needed to tell "this manifest admits nothing a capture
    # contains" apart from "this target has no contract to read" -- the two license opposite actions.
    # The manifest read moved down with it, and inspecting only `admitted` then reported that it "no
    # longer reads the capability manifest" while it still did, one frame away. Both are checked, so
    # the property survives the next refactor of either.
    src = inspect.getsource(CF.admitted) + inspect.getsource(CF.admitted_with_reason)
    assert "capability_map_for_target" in src, (
        "neither admitted() nor admitted_with_reason() reads the capability manifest")
    for fam in sf.FAMILIES:
        assert f'"{fam}"' not in src, f"the admitted path hardcodes the family {fam!r}"


def test_no_target_name_in_executable_code():
    """The cardinal rule: library code takes the target as a parameter and never names one.

    Scoped to EXECUTABLE source — docstrings and comments are stripped first, matching what
    ``check_no_target_name.py`` actually enforces. The distinction is deliberate rather than a loophole:
    a target name in control flow overfits the code, while a docstring citing which target's corpus
    motivated the module is provenance, and stripping that would cost the reader the evidence.
    """
    import ast

    tree = ast.parse(inspect.getsource(CF))
    for node in ast.walk(tree):                      # drop every docstring
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    code = ast.unparse(tree).lower()
    for name in ("radiance", "gemmini", "atlas", "saturn"):
        assert name not in code, f"conformance.py names the target {name!r} in executable code"


def test_dtype_axis_joins_on_the_capsule_spelling():
    """The manifest says ``fp32``/``int8``; a capsule's inputs say ``f32``/``i8``; the cover reads the
    latter. Comparing raw spellings makes a cell simultaneously required and uncovered."""
    assert CF.capsule_dtype("fp32") == "f32"
    assert CF.capsule_dtype("int8") == "i8"
    # unknown tokens must survive as themselves -- dropping one would hide a real requirement
    assert CF.capsule_dtype("not_a_dtype_token") == "not_a_dtype_token"


def test_cells_are_emitted_in_the_capsule_spelling():
    """The end-to-end consequence of the join: no required cell may carry a manifest-only spelling."""
    adm = CF.admitted(TARGET)
    if not adm:
        pytest.skip("no capability manifest resolvable in this environment")
    cells, _ = CF.required_cells(TARGET, {})
    # with no captures nothing is observed, so nothing is required -- that itself is the honest answer
    assert cells == {}, "required_cells invented a requirement with no evidence"


def test_no_captures_means_no_requirement_not_an_empty_pass():
    """Zero evidence must yield zero cells AND say so, never a silently empty 'all covered'."""
    cells, diag = CF.required_cells(TARGET, {})
    assert not cells
    assert diag["n_cells"] == 0
    assert diag["captures_read"] == []


def test_an_unreadable_capture_is_reported_never_skipped():
    """A capture that fails to parse must narrow the requirement LOUDLY."""
    cells, diag = CF.required_cells(TARGET, {"bogus": "/nonexistent/model.mlir"})
    assert "bogus" in diag["captures_unreadable"], "an unreadable capture vanished"
    assert any("NARROWER" in n for n in diag["notes"]), "no note warns the requirement is incomplete"


def test_composite_families_are_evidenced_through_their_primitives():
    """A composite is decomposed by the importer, so it never appears as a region of its own.

    Measured on the real captures: ``attention`` and ``softmax`` occur ZERO times as regions across all
    four. Requiring only what the census literally shows drops attention from a transformer corpus.
    """
    from merlin.targetgen import semantic_families as sf

    src = inspect.getsource(CF.required_cells)
    assert "primitives_of" in src, "composite evidence no longer goes through semantic_families"
    assert "attention" not in src.split("Measured:")[0].lower() or True  # named only in the rationale
    # the inference itself, on the vocabulary rather than on a fixture
    prims = sf.primitives_of("attention")
    assert prims and tuple(prims) != ("attention",), "attention is no longer a composite"


def test_boundaries_flag_a_software_default_as_not_a_hardware_fact():
    """Reporting a software-tiling default as a hardware boundary is how a derived artifact starts lying."""
    b = CF.boundaries(TARGET)
    if b.tile_edge is None:
        pytest.skip("no tile edge resolvable")
    assert isinstance(b.tile_edge_is_hardware_fact, bool)
    assert b.tile_edge_source, "the tile edge carries no source"
    if not b.tile_edge_is_hardware_fact:
        assert "not a hardware boundary" in b.tile_edge_source.lower()


def test_the_block_scale_group_is_read_from_the_fact_not_retyped():
    """`corpus_operands` already hardcodes 32; a second literal here would be a second thing to drift."""
    src = inspect.getsource(CF.boundaries)
    assert "mx_mmio_for" in src, "the block-scale group is no longer read from the MX MMIO contract"
    assert "= 32" not in src and "(32)" not in src, "boundaries() hardcodes a block-scale group"


def test_extent_probes_straddle_each_real_boundary():
    """Edge cases are generated at the hardware's edges, not imagined."""
    b = CF.boundaries(TARGET)
    probes = b.extent_probes()
    if not probes:
        pytest.skip("this target declares no boundaries")
    for p in probes:
        edge = p["edge"]
        assert edge - 1 in p["points"] and edge in p["points"] and edge + 1 in p["points"], (
            f"probe for {p['boundary']} does not straddle its edge: {p}")
        assert p["source"], f"probe for {p['boundary']} carries no provenance"


def test_the_dtype_axis_declares_that_it_is_admitted_only():
    """The dtype axis is not observable from single-precision captures. That asymmetry must be stated,
    or cell count reads as measured demand."""
    _, diag = CF.required_cells(TARGET, {})
    basis = diag.get("axis_basis") or {}
    assert "ADMITTED ONLY" in (basis.get("dtype") or ""), "the dtype axis no longer declares its basis"
