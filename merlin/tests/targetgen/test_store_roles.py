"""Store ROLES are resolved by row width, never by the name an extractor gave a memory.

Every fixture here uses neutral store names, and declares the stores in an order that would mislead a
resolver leaning on declaration order. What is pinned: the narrowest-row store holds operands and the
widest holds accumulate results only where the address space is decided to be separate; a tie, an
unmeasurable width, or a single store is reported with its reason instead of picked; and the
``capacity_fit`` tier that used to look up ``"scratchpad"`` accepts only a store classified against a
second one.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import address_space as AS


def _space(memories, *, edge=16, datapaths=None):
    body = {
        "arrays": [{"name": "grid", "rows": edge, "cols": edge}],
        "datapaths": datapaths
        if datapaths is not None
        else [
            {"name": "in", "dtype": "i8", "evidence": "m_narrow smem UInt<8>"},
            {"name": "out", "dtype": "i32", "evidence": "m_wide smem SInt<32>"},
        ],
        "memories": memories,
    }
    return AS.derive_address_space("t_roles", facts={"schema_version": "2.0", "inputs": {}, "facts": body})


TWO = [{"name": "m_wide", "bytes": 65536, "depth": 512}, {"name": "m_narrow", "bytes": 262144, "depth": 4096}]


def test_the_narrowest_row_holds_operands_and_the_widest_accumulates_whatever_the_order():
    for memories in (TWO, list(reversed(TWO))):
        space = _space(memories)
        operand, accumulator = AS.operand_store(space), AS.accumulator_store(space)
        assert (operand.store.name, operand.basis) == ("m_narrow", AS.BY_WIDTH)
        assert (accumulator.store.name, accumulator.basis) == ("m_wide", AS.BY_WIDTH)
        assert operand.capacity_rows() == 16384 and accumulator.capacity_rows() == 1024


def test_a_sole_store_holds_operands_but_is_marked_unclassified_and_has_no_accumulator():
    space = _space([{"name": "m_narrow", "bytes": 262144, "depth": 4096}])
    operand, accumulator = AS.operand_store(space), AS.accumulator_store(space)
    assert (operand.store.name, operand.basis) == ("m_narrow", AS.SOLE_STORE)
    assert accumulator.store is None and "one store" in accumulator.reason


def test_a_tie_at_the_narrowest_row_is_refused_not_picked():
    same = [{"name": "m_narrow", "bytes": 262144, "depth": 4096}, {"name": "m_twin", "bytes": 131072, "depth": 2048}]
    datapaths = [
        {"name": "in", "dtype": "i8", "evidence": "m_narrow smem UInt<8>"},
        {"name": "in2", "dtype": "i8", "evidence": "m_twin smem UInt<8>"},
    ]
    space = _space(same, datapaths=datapaths)
    operand = AS.operand_store(space)
    assert operand.store is None and "tie" in operand.reason
    accumulator = AS.accumulator_store(space)
    assert accumulator.store is None and accumulator.reason, "equal widths do not decide a second space"


def test_an_unmeasurable_width_among_several_stores_is_refused():
    datapaths = [{"name": "in", "dtype": "i8", "evidence": "m_narrow smem UInt<8>"}]  # m_wide unlinked
    space = _space(TWO, datapaths=datapaths)
    operand = AS.operand_store(space)
    assert operand.store is None and "not measurable" in operand.reason
    assert AS.accumulator_store(space).store is None


def test_no_store_list_is_a_refusal_that_names_the_status():
    resolved = AS.operand_store(_space([]))
    assert resolved.store is None and AS.ABSENT in resolved.reason


@pytest.mark.parametrize(
    "target, operand, basis, accumulator",
    [
        ("gemmini", "scratchpad", AS.BY_WIDTH, "accumulator"),
        ("atlas", "lsu.vmemscalarwrite", AS.SOLE_STORE, None),
        ("radiance", "shared_memory", AS.SOLE_STORE, None),
    ],
)
def test_the_real_targets_resolve_as_their_facts_say(target, operand, basis, accumulator):
    space = AS.derive_address_space(target)
    if space.stores_status != AS.DERIVED:
        pytest.skip(f"no derived store list for {target!r} in this checkout")
    resolved = AS.operand_store(space, dtype="i8")
    assert (resolved.store.name, resolved.basis) == (operand, basis)
    got = AS.accumulator_store(space)
    assert (got.store.name if got.store else None) == accumulator
    if accumulator is None:
        assert got.reason


def test_capacity_accepts_only_a_store_classified_against_a_second(monkeypatch):
    from merlin.compile import capacity
    from merlin.targetgen.rtl import mlc_bridge

    monkeypatch.setattr(mlc_bridge, "discovered_capacities", lambda target: None)  # force the facts tier
    cases = [(_space(TWO), 262144), (_space([{"name": "m_narrow", "bytes": 262144, "depth": 4096}]), None)]
    for space, want in cases:  # both built BEFORE the derivation is patched below
        monkeypatch.setattr(AS, "derive_address_space", lambda target, facts=None, _s=space: _s)
        assert capacity._operand_store_bytes("t_roles") == want
