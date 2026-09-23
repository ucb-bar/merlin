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
    from merlin.compile.scheduling import BlockScheduleError, geometry_from_address_space

    space = AS.derive_address_space("atlas")
    if space.stores_status != AS.DERIVED:
        pytest.skip("no derived store list for atlas in this checkout")
    with pytest.raises(BlockScheduleError, match="in_datapath"):
        geometry_from_address_space(space)


# ---------------------------------------------------------------------------- declared counts


def _declared(memory):
    return AS.derive_address_space(
        "t_declared",
        facts={
            "schema_version": "2.0",
            "inputs": {},
            "facts": {
                "arrays": [{"name": "grid", "rows": 16, "cols": 16}],
                "datapaths": [{"name": "in", "dtype": "i8", "evidence": "m smem UInt<8>"}],
                "memories": [memory],
            },
        },
    )


def _unknowns(space, quantity):
    return [u.reason for u in space.unknowns if u.quantity == quantity]


def test_declared_counts_that_agree_are_recorded_and_change_nothing():
    base = {"name": "m", "bytes": 262144, "depth": 4096}
    plain, declared = _declared(base), _declared({**base, "rows": 16384, "banks": 4})
    a, b = plain.stores[0], declared.stores[0]
    assert (a.total_rows, a.banks) == (b.total_rows, b.banks) == (16384, 4)
    assert "total_rows=16384" in b.sources["declared_counts"] and not _unknowns(declared, "banks")


@pytest.mark.parametrize("field, value, quantity", [("rows", 16000, "total_rows"), ("banks", 8, "banks")])
def test_a_declared_count_that_disagrees_withholds_the_quantity(field, value, quantity):
    space = _declared({"name": "m", "bytes": 262144, "depth": 4096, field: value})
    assert getattr(space.stores[0], quantity) is None
    assert any("neither is used" in reason for reason in _unknowns(space, quantity))


def test_a_row_width_declared_in_bytes_and_in_bits_is_compared_in_one_unit():
    agree = _declared({"name": "m", "bytes": 262144, "depth": 4096, "row_bytes": 16, "row_bits_rtl": 128})
    assert agree.stores[0].row_bytes == 16 and not _unknowns(agree, "row_bytes")
    clash = AS.derive_address_space(
        "t_clash",
        facts={
            "schema_version": "2.0",
            "inputs": {},
            "facts": {
                "memories": [{"name": "m", "bytes": 262144, "depth": 4096, "row_bytes": 16, "row_bits_rtl": 256}]
            },
        },
    )
    assert any("disagree" in reason for reason in _unknowns(clash, "row_bytes"))


def test_a_declared_bank_count_agreeing_with_a_declared_row_width_is_accepted():
    """The atlas shape: row width from the RTL's declared bits, banks declared, no datapath link."""
    declared = {"name": "m", "bytes": 1572864, "depth": 8192, "row_bits_rtl": 256, "banks": 6}
    space = AS.derive_address_space(
        "t_fill", facts={"schema_version": "2.0", "inputs": {}, "facts": {"memories": [declared]}}
    )
    assert (space.stores[0].total_rows, space.stores[0].banks) == (49152, 6)
    assert not _unknowns(space, "banks")


def test_element_width_declarations_on_a_port_are_not_read_as_an_element_type():
    """A byte enable's lane width is not what the store holds; the extractor says so, so it stays UNKNOWN."""
    space = AS.derive_address_space(
        "t_port",
        facts={
            "schema_version": "2.0",
            "inputs": {},
            "facts": {
                "arrays": [{"name": "grid", "rows": 32, "cols": 32}],
                "datapaths": [{"name": "accumulator", "dtype": "bf16", "module": "PE"}],
                "memories": [
                    {
                        "name": "vmem",
                        "bytes": 1572864,
                        "depth": 8192,
                        "row_bits_rtl": 256,
                        "row_elems": 32,
                        "elem_bits": 8,
                        "banks": 6,
                        "source": "firrtl_port_geometry",
                        "row_element_note": "row_elems x elem_bits is the WRITE GRANULARITY",
                    }
                ],
            },
        },
    )
    store = space.stores[0]
    assert store.element_bits is None and store.element_dtype is None
    assert (store.row_bytes, store.total_rows, store.banks) == (32, 49152, 6)


def test_the_gemmini_family_geometry_does_not_move():
    for target in ("gemmini", "gemmini_universal"):
        space = AS.derive_address_space(target)
        if space.stores_status != AS.DERIVED:
            continue
        rows = {s.name: (s.row_bytes, s.total_rows, s.depth, s.banks) for s in space.stores}
        assert rows == {"scratchpad": (16, 16384, 4096, 4), "accumulator": (64, 1024, 512, 2)}, target


def test_an_sram_declaration_does_state_its_element_width():
    """The census reads ``smem buffer : UInt<16>[32] [32]`` -- a declared vector of 32 16-bit elements --
    so, where no datapath links to the store, 16 bits is the element width (the format is still unknown)."""
    memory = {
        "name": "buffer",
        "bytes": 2048,
        "depth": 32,
        "row_elems": 32,
        "elem_bits": 16,
        "row_bits_rtl": 512,
        "banks": 1,
        "source": "firrtl_census",
    }
    body = {
        "arrays": [{"name": "grid", "rows": 32, "cols": 32}],
        "datapaths": [{"name": "accumulator", "dtype": "bf16", "module": "PE"}],
        "memories": [memory],
    }
    space = AS.derive_address_space("t_sram", facts={"schema_version": "2.0", "inputs": {}, "facts": body})
    store = space.stores[0]
    assert (store.element_bits, store.element_dtype, store.row_bytes, store.total_rows) == (16, None, 64, 32)


def test_atlas_as_its_declared_elaboration_derives_a_schedulable_address_space():
    """From the SRAM declarations in AtlasRocketConfig (checked in as a fixture, so no RTL checkout is
    needed): the 1.5 MiB VMEM is the operand store; the matrix register file (one 256-bit word per row) and
    the instruction memory (one 32-bit word) cannot hold an array-edge row and take no array role; and the
    accumulator is ADDRESSABLE -- two identical 32-row bf16 buffers, the acc0/acc1 a compute selects --
    rather than state inside the PE. Every number here is a derivation from the RTL, none is a literal."""
    import json

    from merlin.common.paths import merlin_dir
    from merlin.compile.scheduling import geometry_from_address_space

    doc = json.loads((merlin_dir() / "tests/data/block_schedule/atlas_declared_elaboration_facts.json").read_text())
    space = AS.derive_address_space("atlas_declared", facts=doc)
    fed = {s.name: AS._array_fed(s, space) for s in space.stores}
    assert fed == {
        "mregfile.banks": False,
        "vmem.banks": True,
        "accumulationbuffers.buffer0": True,
        "accumulationbuffers.buffer1": True,
        "instrmem.mem": False,
    }
    operand = AS.operand_store(space)
    assert (operand.store.name, operand.basis, operand.store.total_rows) == ("vmem.banks", AS.BY_WIDTH, 49152)
    kind = AS.accumulator_kind(space)
    assert (kind.kind, kind.buffers, kind.rows, kind.dtype) == (AS.ADDRESSABLE, 2, 32, "bf16")
    geometry = geometry_from_address_space(space)
    assert (geometry.block, geometry.operand_rows, geometry.operand_bank_rows, geometry.accumulator_rows) == (
        32,
        49152,
        8192,
        32,
    )


def test_identical_stores_of_a_different_width_than_the_accumulate_type_stay_a_tie():
    """Twins are buffers only when they hold the declared accumulate type; otherwise the tie is a refusal."""
    fed = {"row_elems": 32, "depth": 32, "banks": 1, "source": "firrtl_census"}
    memories = [
        {"name": "op", "bytes": 32 * 32 * 64, "depth": 64, "row_elems": 32, "elem_bits": 8, "source": "firrtl_census"},
        {**fed, "name": "a", "bytes": 32 * 32 * 4, "elem_bits": 32},
        {**fed, "name": "b", "bytes": 32 * 32 * 4, "elem_bits": 32},
    ]
    body = {
        "arrays": [{"name": "grid", "rows": 32, "cols": 32}],
        "datapaths": [{"name": "accumulator", "dtype": "bf16", "module": "PE"}],
        "memories": memories,
    }
    space = AS.derive_address_space("t_twins", facts={"schema_version": "2.0", "inputs": {}, "facts": body})
    assert AS.accumulator_kind(space).kind != AS.ADDRESSABLE
