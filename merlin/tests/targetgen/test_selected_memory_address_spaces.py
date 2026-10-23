"""Selected role views preserve both banks and their independent physical census."""

import pytest

from merlin.targetgen.address_space import accumulator_store, derive_address_space, operand_store
from merlin.targetgen.rtl.circt_introspect import _add_selected_address_space


def test_selected_role_views_retain_operand_and_accumulator_capacity():
    physical = {"name": "bank.mem", "bytes": 1024, "depth": 16, "elem_bits": 8}
    facts = {
        "arrays": [{"name": "array", "rows": 4, "cols": 4}],
        "datapaths": [
            {"name": "operand", "dtype": "i8"},
            {"name": "accum", "dtype": "i32"},
        ],
        "memories": [physical],
    }
    _add_selected_address_space(
        facts, {"name": "operand", "bytes": 1024, "depth": 64}, source="selected_firrtl_role_probe"
    )
    _add_selected_address_space(facts, {"name": "accum", "bytes": 512, "depth": 16}, source="selected_hw_port_geometry")
    # The physical member remains exact; it must not become a third schedulable store.
    assert facts["memories"][0] == physical
    space = derive_address_space("demo", facts={"facts": facts})
    assert [store.name for store in space.stores] == ["operand", "accum"]
    assert operand_store(space).store.total_rows == 256
    assert accumulator_store(space).store.total_rows == 32
    assert space.separate_accumulator_space is True


def test_incomplete_probe_does_not_claim_a_logical_store():
    facts = {"memories": []}
    _add_selected_address_space(facts, {"name": "accum", "bytes": None}, source="probe")
    assert facts == {"memories": []}


def test_duplicate_logical_views_refuse_instead_of_double_counting():
    facts = {}
    memory = {"name": "buffer", "bytes": 128}
    _add_selected_address_space(facts, memory, source="ports")
    with pytest.raises(ValueError, match="duplicate selected memory address space"):
        _add_selected_address_space(facts, memory, source="probe")
