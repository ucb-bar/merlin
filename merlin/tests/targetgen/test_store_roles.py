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

#: The real-target cases pin what each target's own facts derive; the synthetic ones carry the rules.
pytestmark = pytest.mark.target("gemmini", "atlas", "radiance", "saturn_opu_mxv256d128")


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


# ------------------------------------------------------------------------------- accumulator kind


def _body(memories, datapaths, *, edge=16):
    return AS.derive_address_space(
        "t_kind",
        facts={
            "schema_version": "2.0",
            "inputs": {},
            "facts": {
                "arrays": [{"name": "grid", "rows": edge, "cols": edge}],
                "datapaths": datapaths,
                "memories": memories,
            },
        },
    )


LINKED = [
    {"name": "in", "dtype": "i8", "evidence": "m_narrow smem UInt<8>"},
    {"name": "accumulator", "dtype": "i32", "evidence": "m_wide smem SInt<32>"},
]


def test_a_wide_store_the_accumulate_datapath_fills_is_addressable():
    kind = AS.accumulator_kind(_body(TWO, LINKED))
    assert (kind.kind, kind.store.name, kind.rows, kind.dtype) == (AS.ADDRESSABLE, "m_wide", 1024, "i32")


def test_an_accumulate_type_with_no_store_to_fill_is_in_the_datapath_and_has_no_rows():
    """The facts say WHERE the accumulator is; they do not say how deep, so rows stay unknown."""
    one = [{"name": "m_narrow", "bytes": 262144, "depth": 4096}]
    datapaths = [LINKED[0], {"name": "accumulator", "dtype": "bf16", "module": "PE", "ports": ["mac"]}]
    kind = AS.accumulator_kind(_body(one, datapaths))
    assert (kind.kind, kind.dtype, kind.rows) == (AS.IN_DATAPATH, "bf16", None)
    assert kind.unknown.quantity == "accumulator_rows" and "not declare how many tiles" in kind.reason


def test_contradictory_accumulate_types_are_unknown_not_picked():
    """Two datapaths each carrying an accumulate type (i32 and f32) have no single type to schedule for."""
    one = [{"name": "m_narrow", "bytes": 16384}]
    datapaths = [
        {"name": "int8", "dtype": "i8", "accumulator": "i32"},
        {"name": "float8", "dtype": "f8", "accumulator": "f32"},
    ]
    kind = AS.accumulator_kind(_body(one, datapaths))
    assert kind.kind == AS.UNKNOWN_KIND and "2 different accumulate types" in kind.reason


def test_a_wide_store_no_datapath_fills_is_not_called_an_accumulator():
    """Width separates two address spaces; it does not say results accumulate in the wider one."""
    datapaths = [
        {"name": "in", "dtype": "i8", "evidence": "m_narrow smem UInt<8>"},
        {"name": "other", "dtype": "i32", "evidence": "m_wide smem SInt<32>"},
    ]
    kind = AS.accumulator_kind(_body(TWO, datapaths))  # linked, but nothing declares accumulation
    assert kind.kind == AS.UNKNOWN_KIND and "no datapath declares an accumulate type" in kind.reason
    unlinked = [{"name": "in", "dtype": "i8", "evidence": "m_narrow smem UInt<8>"}]
    kind = AS.accumulator_kind(_body(TWO, unlinked))
    assert kind.kind == AS.UNKNOWN_KIND and kind.reason


def test_a_store_whose_type_disagrees_with_the_declared_accumulate_type_is_unknown():
    datapaths = [
        LINKED[0],
        {"name": "wide", "dtype": "i32", "evidence": "m_wide smem SInt<32>"},
        {"name": "sum", "dtype": "i16", "accumulator": "i16"},
    ]
    kind = AS.accumulator_kind(_body(TWO, datapaths))
    assert kind.kind == AS.UNKNOWN_KIND and "declare an accumulate type of i16" in kind.reason


def test_no_declaration_at_all_is_unknown():
    kind = AS.accumulator_kind(_body([{"name": "m", "bytes": 131072}], []))
    assert kind.kind == AS.UNKNOWN_KIND and "no datapath declares" in kind.reason


@pytest.mark.parametrize(
    "target, kind",
    [
        ("gemmini", AS.ADDRESSABLE),
        ("atlas", AS.IN_DATAPATH),
        ("radiance", AS.UNKNOWN_KIND),
        ("saturn_opu_mxv256d128", AS.UNKNOWN_KIND),
    ],
)
def test_the_real_targets_accumulators_are_what_their_facts_say(target, kind):
    space = AS.derive_address_space(target)
    if space.stores_status != AS.DERIVED:
        pytest.skip(f"no derived store list for {target!r} in this checkout")
    got = AS.accumulator_kind(space)
    assert got.kind == kind, got.reason
    if kind != AS.ADDRESSABLE:
        assert got.reason and got.rows is None


def test_the_schedule_pass_refuses_an_in_datapath_accumulator_by_name():
    from merlin.compile.scheduling import BlockScheduleError, Geometry

    space = AS.derive_address_space("atlas")
    if space.stores_status != AS.DERIVED:
        pytest.skip("no derived store list for atlas in this checkout")
    with pytest.raises(BlockScheduleError, match="in_datapath"):
        Geometry.from_address_space(space)
