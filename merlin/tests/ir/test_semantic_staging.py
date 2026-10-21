"""Pure region cuts, completed-region lifetimes and temporary-buffer reuse."""

from dataclasses import replace

import pytest

from merlin.compile.semantic_regions import partition_semantic_dag, semantic_staging_buffers
from merlin.semantic_compiler.allocate import Reservation, StorageBank
from merlin.semantic_compiler.buffer_allocation import (
    Buffer,
    allocate_buffers,
    check_buffer_allocation,
    first_fit_buffer_assignment,
)
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def chain():
    tensor = TensorType((2,), "i32", "exact-i32")
    return KernelRequest(
        (SemanticNode("x", "input", (), tensor, effect="input"),) + tuple(
            SemanticNode(name, "add", (previous, "x"), tensor)
            for name, previous in zip(("a", "b", "c", "d"), ("x", "a", "b", "c"), strict=True)
        ),
        ("d",), ("external",), (("x", "external"),), "synthetic-staging",
    )


@pytest.mark.parametrize("max_nodes,capacity", [(1, 3), (2, 4)])
def test_completed_regions_reuse_storage_without_changing_values(max_nodes, capacity):
    source = chain()
    regions = partition_semantic_dag(source, max_nodes=max_nodes)
    buffers = semantic_staging_buffers(
        source, regions, storage="external", extents=dict.fromkeys((n.id for n in source.nodes), 1),
        fixed={"x": 0, "d": 1},
    )
    result = allocate_buffers(buffers, (StorageBank("external", "memory", capacity, "tensor"),))
    assert result.status == "sat", result.reason
    addresses = dict(result.addresses)
    if len(regions) > 2:
        assert len(set(addresses.values())) < len(buffers)
    for elements in ((1, 3), (-7, 11), (0, -19)):
        value = TensorValue(source.nodes[0].type, elements)
        memory = {addresses["x"]: value}
        for region in regions:
            inputs = {key: memory[addresses[key]] for key, _ in region.input_storages}
            assert not {addresses[key] for key in region.outputs}.intersection(
                addresses[key] for key in inputs
            )
            for key, output in zip(region.outputs, evaluate_graph(region, inputs), strict=True):
                memory[addresses[key]] = output
        assert memory[addresses["d"]] == evaluate_graph(source, {"x": value})[0]
        assert memory[addresses["x"]] == value


def test_staging_rejects_reordered_or_modified_slices_and_missing_computations():
    source = chain()
    regions = partition_semantic_dag(source, max_nodes=1)
    kwargs = {"storage": "external", "extents": dict.fromkeys((n.id for n in source.nodes), 1), "fixed": {"x": 0}}
    with pytest.raises(ValueError, match="before its producer completed"):
        semantic_staging_buffers(source, tuple(reversed(regions)), **kwargs)
    with pytest.raises(ValueError, match="differ"):
        changed = replace(regions[0], nodes=(*regions[0].nodes[:-1], replace(regions[0].nodes[-1], op="multiply")))
        semantic_staging_buffers(source, (changed, *regions[1:]), **kwargs)
    with pytest.raises(ValueError, match="omit source"):
        semantic_staging_buffers(source, regions[:-1], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        semantic_staging_buffers(source, (regions[0], *regions), **kwargs)


def test_staging_preserves_multiple_outputs_and_shared_producers():
    source = chain()
    source = replace(source, outputs=("d", "a", "d"), output_storages=("external",) * 3)
    regions = partition_semantic_dag(source, max_nodes=2)
    buffers = semantic_staging_buffers(
        source, regions, storage="external", extents=dict.fromkeys((n.id for n in source.nodes), 1), fixed={},
    )
    last = len(regions) + 1
    assert next(item for item in buffers if item.name == "a").last_event == last
    assert regions[0].outputs == ("a", "b")
    assert regions[1].outputs == ("d",)


def test_eligibility_shrinks_only_the_rejected_boundary_and_preserves_values():
    source = chain()
    seen = []

    def accept(region):
        pure = tuple(n.id for n in region.nodes if n.effect == "pure")
        seen.append(pure)
        return not ("a" in pure and "c" in pure)

    regions = partition_semantic_dag(source, max_nodes=3, accept=accept)
    assert [tuple(n.id for n in r.nodes if n.effect == "pure") for r in regions] == [("a", "b"), ("c", "d")]
    assert seen == [("a", "b", "c"), ("a", "b"), ("c", "d")]
    values = {"x": TensorValue(source.nodes[0].type, (3, -7))}
    for region in regions:
        results = evaluate_graph(region, {key: values[key] for key, _ in region.input_storages})
        values.update(zip(region.outputs, results, strict=True))
    assert values["d"] == evaluate_graph(source, {"x": values["x"]})[0]


@pytest.mark.parametrize("answer,reason", ((False, "no nonempty region"), (None, "definite Boolean")))
def test_rejected_or_unknown_eligibility_never_produces_a_region(answer, reason):
    with pytest.raises(ValueError, match=reason):
        partition_semantic_dag(chain(), max_nodes=3, accept=lambda region: answer)


def test_first_fit_proposal_is_checked_against_original_buffers_and_solver():
    source = chain()
    regions = partition_semantic_dag(source, max_nodes=1)
    buffers = semantic_staging_buffers(
        source, regions, storage="external", extents=dict.fromkeys((n.id for n in source.nodes), 1),
        fixed={"x": 0, "d": 1},
    )
    banks = (StorageBank("external", "memory", 3, "tensor"),)
    proposal = first_fit_buffer_assignment(buffers, banks)
    assert proposal is not None
    pinned = tuple(replace(item, fixed_start=proposal[item.name]) for item in buffers)
    result = allocate_buffers(pinned, banks)
    assert result.status == "sat", result.reason
    check_buffer_allocation(buffers, banks, dict(result.addresses))
    assert first_fit_buffer_assignment(buffers, (StorageBank("external", "memory", 2, "tensor"),)) is None


def test_first_fit_preserves_view_lifetimes_alignment_and_reservations():
    buffers = (
        Buffer("parent", "a", 3, 0, 0, alignment=2),
        Buffer("view", "a", 1, 0, 3, alignment=2, view_of="parent", view_offset=2),
        Buffer("other", "b", 2, 1, 2, alignment=2),
    )
    banks = (StorageBank("a", "memory", 8, "word"), StorageBank("b", "memory", 8, "word"))
    reservations = (Reservation("b", 0, 2),)
    proposal = first_fit_buffer_assignment(buffers, banks, reservations=reservations)
    assert proposal is not None
    assert proposal["view"] == proposal["parent"] + 2
    assert all(address % 2 == 0 and address >= 2 for address in proposal.values())
    check_buffer_allocation(buffers, banks, proposal, reservations=reservations)
    result = allocate_buffers(
        tuple(replace(item, fixed_start=proposal[item.name]) for item in buffers),
        banks, reservations=reservations,
    )
    assert result.status == "sat", result.reason
    check_buffer_allocation(buffers, banks, dict(result.addresses), reservations=reservations)
