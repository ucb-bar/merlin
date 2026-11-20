"""Word-stage custody cannot replace actual primitive effect selection."""

from copy import deepcopy

import pytest

from merlin.frontends.kernel_word_source_effects import checked_word_source_effects


def fixture():
    from test_kernel_barrier_evidence import combined_fixture

    graph, _, _, protected = combined_fixture()
    memory = {"effect_profile": "private-invocation-typed-atomics-v2",
              "volatile_profile": "private-invocation-volatile-loads-v1"}
    return protected, {"schema": "t2m.source-word-stages.v4"}, memory, graph["operations"]


def test_actual_combined_primitives_keep_their_independent_selected_contracts():
    checked_word_source_effects(*fixture())


@pytest.mark.parametrize("schema", ["t2m.triton-shared-stage.v1", "t2m.source-word-stages.v2"])
def test_legacy_stage_versions_cannot_admit_actual_volatile_source(schema):
    source, stage, memory, rows = fixture()
    stage["schema"] = schema
    with pytest.raises(ValueError, match="actual volatile"):
        checked_word_source_effects(source, stage, memory, rows)


@pytest.mark.parametrize("damage", [
    "atomic-profile", "volatile-profile", "atomic", "volatile", "kind", "flag", "dtype"
])
def test_actual_primitives_do_not_inherit_permissions_from_changed_records(damage):
    source, stage, memory, rows = fixture()
    if damage == "atomic-profile":
        memory["effect_profile"] = "ordinary"
    elif damage == "volatile-profile":
        memory["volatile_profile"] = "ordinary"
    elif damage == "atomic":
        rows[2].pop("atomic")
    elif damage == "volatile":
        rows[4].pop("volatile_read")
    elif damage == "kind":
        source = source.replace(b"atomic_rmw_op=4:i32", b"atomic_rmw_op=10:i32")
    elif damage == "flag":
        source = source.replace(b"isVolatile=true", b"isVolatile=false")
    else:
        rows[4]["volatile_read"]["result_type"] = "i32"
    with pytest.raises(ValueError, match="atomic|volatile"):
        checked_word_source_effects(source, stage, memory, rows)


@pytest.mark.parametrize("schema", ["t2m.triton-shared-stage.v1", "t2m.source-word-stages.v2"])
def test_existing_f64_ieee_word_evidence_keeps_its_original_label(schema):
    from test_kernel_barrier_evidence import atomic_fixture

    graph, _, _, source = atomic_fixture()
    source = source.replace(b"value=4:i64", b"value=4.0:f64").replace(b"i64", b"f64")
    source = source.replace(b"atomic_rmw_op=4:i32", b"atomic_rmw_op=5:i32")
    row = graph["operations"][2]["atomic"]
    row.update(operation="fadd", standard_kind=0, value_dtype="f64", numeric_contract="ieee-f64-rne-add")
    before = deepcopy(graph)
    checked_word_source_effects(source, {"schema": schema},
        {"effect_profile": "private-invocation-typed-atomics-v2"}, graph["operations"])
    assert graph == before


def test_an_unexpanded_physical_operation_cannot_be_granted_by_custody():
    source, stage, memory, rows = fixture()
    source = source.replace(b'"tt.return"()', b'"ttg.barrier"() <{addrSpace=31:i32}>:()->() "tt.return"()')
    with pytest.raises(ValueError, match="unexpanded physical/effect"):
        checked_word_source_effects(source, stage, memory, rows)


def test_malformed_word_source_is_refused_before_it_can_be_staged():
    _, stage, memory, rows = fixture()
    with pytest.raises(ValueError, match="parseable typed"):
        checked_word_source_effects(b"this is not IR", stage, memory, rows)
