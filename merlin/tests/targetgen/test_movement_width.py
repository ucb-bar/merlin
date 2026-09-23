"""A transfer that carries one array tile where the target admits several, and the bound that says so.

Measured, on 90 instruction traces (82,985 instructions) emitted by one live agent's compiler: the
memory-load classes were 19.2% of the stream and NOT ONE carried more than a single array tile, while
the same backend's generated package derived a four-tile transfer payload from the target's own DMA
bound and its dialect already permitted the wider form. Nothing could observe that, and nothing could
demand otherwise -- the pass-obligation vocabulary had four classes and none of them was about data
movement.

These tests pin the two halves of the fix:

* the DIAGNOSTIC (:func:`trace_check.movement_findings`) -- mode-free, like ``residency_findings``:
  it reads the emitted stream and says the declared width went unused, with every number in the
  message coming from the derived bound rather than from a constant in the checker;
* the BOUND (:func:`trace_check.movement_bound_for`) -- three facts, each from the place that owns it
  (the tile edge and the input element width from the target's RTL facts, the payload budget from its
  capability manifest), floored, and ABSENT rather than defaulted when any of them is missing.

The third test is the one that matters most for a repo that plugs in arbitrary targets: a target that
declares no transfer payload gets SILENCE. A guessed default would either invent an optimization the
hardware cannot perform or -- worse, because it reads as clean -- hide one it can. Nothing in this
file names a tile edge, a payload size or a block length: the bounds under test are constructed from
the derivation, or supplied as the data the derivation returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from merlin.targetgen import trace_check as TCK

# The load class the decoder derives for a memory-movement instruction, and a funct so the trace is a
# well-formed one. Neither is a target fact under test here: the class vocabulary is the shared
# human-owned one, and the funct only has to be non-null.
_LOAD = "MVIN"
_FUNCT = 2


def _trace(cols_per_transfer: list[int]) -> dict:
    """A stream of memory loads, each carrying ``cols`` columns, as the decoder reports them."""
    ins = []
    for index, cols in enumerate(cols_per_transfer):
        ins.append(
            {
                "index": index,
                "class": _LOAD,
                "funct": _FUNCT,
                "decoded": {
                    "dram": {"raw": None, "kind": "argbase", "arg_index": 0, "offset": index * cols},
                    "rows": 1,
                    "cols": cols,
                    "spad_addr": index,
                },
            }
        )
    return {"instructions": ins}


#: A target whose capability manifest declares a transfer payload. Named here because this is a test
#: OF that declaration; nothing in the library knows it.
_DECLARING_TARGET = "gemmini"  # target-ok: a target-specific test of that target's own declaration


@pytest.fixture(scope="module")
def bound() -> dict:
    """The derived bound for a target whose manifest declares a transfer payload.

    Derived, not written down: if any of the three facts were missing this would be ``{}`` and every
    assertion below would be vacuous. So the two halves are separated rather than both skipped —
    DELETING THE DECLARATION IS A FAILURE (it is a tracked file, and the tests that follow are the
    only thing that would notice it going), while a host that cannot derive the target's RTL facts is
    an honest skip and not a green tick on an empty bound.
    """
    import yaml

    example_contract = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / _DECLARING_TARGET
        / "target"
        / "contracts"
        / "target_contract.yaml"
    )
    contract = yaml.safe_load(example_contract.read_text(encoding="utf-8"))
    declared = ((contract.get("memory_model") or {}).get("dma") or {}).get("max_transfer_bytes")
    assert declared, "this target's manifest must declare memory_model.dma.max_transfer_bytes"
    derived = TCK.movement_bound_for(_DECLARING_TARGET)
    if not derived:
        pytest.skip("the declaration is present but this host cannot derive the target's tile edge / dtype")
    assert derived["max_transfer_bytes"] == declared, "the bound must carry the DECLARED payload, not another"
    return derived


def test_the_bound_is_derived_from_the_three_facts_and_floors(bound):
    """Payload budget / (tile edge x element width), FLOORED -- and the inputs kept beside the result.

    A payload that does not hold a whole extra tile holds no extra tile; rounding up would tell a
    backend to issue a transfer the hardware truncates.
    """
    assert bound["tiles_per_transfer"] == bound["max_transfer_bytes"] // (bound["tile_cols"] * bound["elem_bytes"])
    assert bound["transfer_cols"] == bound["tile_cols"] * bound["tiles_per_transfer"]
    assert bound["tiles_per_transfer"] >= 2, "a target that admits only one tile per transfer cannot coalesce"


def test_a_single_tile_stream_is_reported(bound):
    """THE MEASURED DEFECT: every transfer carries one tile while the payload holds several."""
    trace = _trace([bound["tile_cols"]] * 8)
    findings = TCK.movement_findings(trace, bound)
    assert len(findings) == 1, findings
    finding = findings[0]
    assert "movement width" in finding
    # The finding must cite what it was computed from, so a reader can check the claim rather than
    # trust it -- and must state the saving as a count of transfers, not as a speedup.
    assert str(bound["max_transfer_bytes"]) in finding
    assert str(bound["transfer_cols"]) in finding
    expected_coalesced = -(-8 // bound["tiles_per_transfer"])
    assert f"cost 8 transfers where {expected_coalesced} would carry them" in finding


def test_transfers_at_the_full_bound_are_silent(bound):
    """THE FIXED STREAM: the same bytes at the declared width. Nothing left to say."""
    assert TCK.movement_findings(_trace([bound["transfer_cols"]] * 2), bound) == []


def test_a_lone_narrow_transfer_has_nothing_to_coalesce_with(bound):
    """The finding is about a stream, not an instruction: one transfer cannot be widened by merging."""
    assert TCK.movement_findings(_trace([bound["tile_cols"]]), bound) == []


def test_no_declared_bound_is_silence_not_a_default():
    """FAIL CLOSED. A target that declares no transfer payload gets no finding and no guessed width.

    Both spellings of "unknown" are silent: no bound at all, and a bound whose payload holds exactly
    one tile (where coalescing is not a thing the hardware can do).
    """
    stream = _trace([16] * 8)
    assert TCK.movement_findings(stream, None) == []
    assert TCK.movement_findings(stream, {}) == []
    one_tile = {
        "tile_cols": 16,
        "elem_bytes": 1,
        "max_transfer_bytes": 16,
        "tiles_per_transfer": 1,
        "transfer_cols": 16,
    }
    assert TCK.movement_findings(stream, one_tile) == []


def test_a_target_with_no_declaration_derives_no_bound():
    """The same fail-closed answer at the DERIVATION, not only at the diagnostic.

    Every in-tree target that does not declare ``memory_model.dma.max_transfer_bytes`` must come back
    with no bound. This is the property that makes the diagnostic safe to run on any target: a missing
    declaration cannot become a width someone assumed.
    """
    import yaml

    from merlin.common.paths import merlin_dir

    checked = 0
    for contract_path in sorted((merlin_dir() / "targets").glob("*/contracts/target_contract.yaml")):
        doc = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        if ((doc.get("memory_model") or {}).get("dma") or {}).get("max_transfer_bytes"):
            continue
        checked += 1
        assert TCK.movement_bound_for(str(doc.get("name") or contract_path.parents[1].name)) == {}
    assert checked, "no undeclared target to check: this property would be vacuous"


def test_an_absent_target_derives_no_bound():
    """A name that resolves to no package is an unknown bound, never a raise and never a default."""
    assert TCK.movement_bound_for("") == {}
    assert TCK.movement_bound_for(None) == {}
    assert TCK.movement_bound_for("no_such_target_package") == {}


def test_a_capsule_can_now_demand_block_dma():
    """THE OTHER HALF: the obligation vocabulary can express the demand at all.

    Before this, ``pass_requirements_for`` could emit exactly four classes -- partition, tiling, ISA
    lowering, host seam -- and not one of them was about how much data a transfer carries, so no
    capsule could ask for a wide transfer and no gate could notice its absence.

    The class is DERIVED from the entry's own extents, like the other four: a transfer's width runs
    along the operand's contiguous axis, so K (the activation row) or N (the weight/result row) past
    the tile edge is the case where one transfer could carry the next tile's bytes too. M past the
    edge alone adds ROWS -- more transfers of the same width -- and demands tiling, nothing about
    movement.
    """
    from merlin.targetgen.corpus_synth import pass_requirements_for
    from merlin.xdsl_dialects.lowering import passes as P

    spec = {"boundaries": {"extent_probes": [{"boundary": "tile_edge", "edge": 16}]}}

    wide_reduction = pass_requirements_for({"M": "tile", "K": "2*tile", "N": "tile"}, spec)
    assert P.BLOCK_MOVEMENT in wide_reduction
    assert P.TILE_SCHEDULE in wide_reduction, "a wide operand still has to be tiled"

    wide_output = pass_requirements_for({"M": "tile", "K": "tile", "N": "2*tile"}, spec)
    assert P.BLOCK_MOVEMENT in wide_output

    tall_only = pass_requirements_for({"M": "2*tile", "K": "tile", "N": "tile"}, spec)
    assert P.TILE_SCHEDULE in tall_only
    assert P.BLOCK_MOVEMENT not in tall_only, "more rows is not a wider transfer"

    one_tile = pass_requirements_for({"M": "tile", "K": "tile", "N": "tile-1"}, spec)
    assert P.BLOCK_MOVEMENT not in one_tile


def test_the_new_class_discharges_a_declared_obligation():
    """A requirement class that maps onto no obligation is a label; the gate reads this mapping."""
    from merlin.xdsl_dialects.lowering import passes as P

    assert P.CLASS_OBLIGATION[P.BLOCK_MOVEMENT] in P.OBLIGATIONS
