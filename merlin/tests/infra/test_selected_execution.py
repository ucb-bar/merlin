"""Selected exact-source covers and conservative execution memory obligations."""

from dataclasses import replace

import pytest
import yaml
from test_offload_plan import example
from test_source_proposal_admission import original_check, proposal, topology

from merlin.compiler.backend.sourceproposal import admit_original_cpu, admit_proposal
from merlin.compiler.planning.execution import select_execution
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, ValueRepresentation
from merlin.xdsl_dialects.lowering.global_plan_emission import _component_boundary


def roster(tmp_path, *, capacity=4096, extra_domain=False, shape=(3, 7, 5)):
    legacy, old_alternatives, context = example(shape=shape)
    source = bind_source(legacy.module, entry=legacy.entry, context=context, graph_mode="ssa")
    alternatives = tuple(
        replace(
            source.cpu_alternatives[item.nodes[0]],
            id=item.id,
            implementation=item.implementation,
            placement=item.placement,
        )
        for item in old_alternatives
    )
    hardware = topology(tmp_path)
    document = yaml.safe_load(hardware.path.read_text())
    document["memory_domains"][0]["capacity_bytes"] = capacity
    if extra_domain:
        document["memory_domains"].append({"id": "other", "capacity_bytes": capacity, "alignment_bytes": 64})
        document["access"].append({**document["access"][1], "domain": "other"})
    hardware.path.write_text(yaml.safe_dump(document))
    hardware = select_topology(hardware.path)
    cpu = tuple(
        admit_original_cpu(source, i, hardware, resource="cpu0", memory_domain="dram")
        for i in range(len(source.operations))
    )
    candidates = [proposal(source, alt) for alt in alternatives]
    candidates[0] = replace(candidates[0], provider="vector", resource="cpu0", required_features=("rvv",))
    candidates[1] = replace(candidates[1], provider="array", memory_domain="other" if extra_domain else "dram")
    device = tuple(
        admit_proposal(source, item, hardware, validate_implementation=original_check) for item in candidates
    )
    return source, hardware, cpu, device


def test_exact_equal_shape_sites_complete_cover_and_shared_cpu_are_deterministic(tmp_path):
    source, hardware, cpu, device = roster(tmp_path)
    candidates = (*cpu, *device)
    result = select_execution(source, candidates, preferred_providers=("vector", "array"))
    reversed_result = select_execution(source, tuple(reversed(candidates)), preferred_providers=("vector", "array"))
    assert result.plan.digest == reversed_result.plan.digest
    assert tuple(n for item in result.selected for n in item.alternative.nodes) == tuple(range(8))
    selected = {item.alternative.nodes[0]: item for item in result.selected}
    assert selected[3] is device[0] and selected[7] is device[1]
    record = result.record()
    assert record["qualification"] == "not_established" and not result.plan.cycles.resolved
    assert {item["shared_group"] for item in record["execution_order"] if item["resource"] == "cpu0"} == {"cpu0"}
    assert len({item["slot"] for item in record["execution_order"]}) == len(result.selected)
    preferred_cpu = select_execution(source, candidates, preferred_providers=("merlin.scalar",))
    assert preferred_cpu.selected == cpu
    assert hardware.memory("dram").alignment_bytes == 64


def test_capacity_backtracking_keeps_unknown_costs_unknown(tmp_path):
    source, hardware, cpu, device = roster(tmp_path, capacity=640)
    costly = replace(device[1].proposal, alternative=replace(device[1].alternative, id="large", scratch_bytes=640))
    costly = admit_proposal(source, costly, hardware, validate_implementation=original_check)
    result = select_execution(source, (*cpu, costly), preferred_providers=("array",))
    assert result.selected == cpu and result.storage_bound_bytes <= 640
    assert any(identity == "large" and "capacity" in reason for identity, reason in result.rejected)
    assert result.plan.cycles.lo is None and result.plan.cycles.hi is None
    root = tmp_path / "tiny"
    root.mkdir()
    source, _, cpu, _ = roster(root, capacity=64)
    with pytest.raises(ValueError, match="exceeds selected available capacity"):
        select_execution(source, cpu)


def test_unmaterialized_domain_and_uncovered_or_overlapping_rosters_refuse(tmp_path):
    source, _, cpu, device = roster(tmp_path, extra_domain=True)
    # Forced array route leaves no same-domain alternative for node7.
    with pytest.raises(ValueError, match="memory-domain transition"):
        select_execution(source, (*cpu[:7], device[1]))
    with pytest.raises(ValueError, match="no complete source cover"):
        select_execution(source, cpu[:7])
    with pytest.raises(ValueError, match="IDs must be unique"):
        select_execution(source, (*cpu, cpu[0]))
    # Overlapping alternatives are choices, never emitted twice.
    chosen = select_execution(source, (*cpu, device[0]), preferred_providers=("vector",))
    assert len(chosen.plan.selected) == 8
    assert sum(3 in item.nodes for item in chosen.plan.selected) == 1


def test_selected_source_topology_and_report_custody(tmp_path):
    source, hardware, cpu, _ = roster(tmp_path)
    chosen = select_execution(source, cpu)
    with pytest.raises(ValueError, match="decision report changed"):
        replace(chosen, policy=("another",)).revalidate()
    hardware.path.write_text(hardware.path.read_text() + "\n# changed source\n")
    with pytest.raises(ValueError, match="topology changed"):
        chosen.revalidate()
    other = tmp_path / "other"
    other.mkdir()
    source, _, cpu, _ = roster(other)
    chosen = select_execution(source, cpu)
    from xdsl.dialects.builtin import StringAttr

    source.operations[0].attributes["prov.changed"] = StringAttr("different")
    with pytest.raises(ValueError, match="changed|stale"):
        chosen.revalidate()


def test_region_lifetimes_are_widened_before_capacity_admission(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms.pipeline_state import PipelineContext

    module = parse_mlir_text("""module {func.func @forward(%a:tensor<1xi32>,%b:tensor<1xi32>)->tensor<1xi32> {
      %x=tensor.cast %a:tensor<1xi32> to tensor<1xi32>
      %y=tensor.cast %x:tensor<1xi32> to tensor<1xi32>
      %z=tensor.cast %y:tensor<1xi32> to tensor<1xi32>
      func.return %z:tensor<1xi32>
    }}""")
    source = bind_source(module, entry="forward", context=PipelineContext(), graph_mode="ssa")
    hardware = topology(tmp_path)
    document = yaml.safe_load(hardware.path.read_text())
    document["memory_domains"][0]["capacity_bytes"] = 256
    hardware.path.write_text(yaml.safe_dump(document))
    hardware = select_topology(hardware.path)
    cpu = tuple(admit_original_cpu(source, i, hardware, resource="cpu0", memory_domain="dram") for i in range(3))
    assert select_execution(source, cpu).storage_bound_bytes == 256
    nodes = (0, 1, 2)
    ins, outs = _component_boundary(source.program, nodes)
    order = {name: index for index, name in enumerate(source.program.buffers)}

    def reps(names):
        return tuple(
            BufferRepresentation(name, ValueRepresentation("tensor_ssa", "logical", "i32"))
            for name in sorted(names, key=order.__getitem__)
        )

    alternative = replace(source.cpu_alternatives[0], id="combined", nodes=nodes, inputs=reps(ins), outputs=reps(outs))
    candidate = replace(proposal(source, alternative), resource="cpu0", required_features=())

    def source_casts_only(bound, offered):
        assert all(bound.operations[n].name == "tensor.cast" for n in offered.alternative.nodes)

    admitted = admit_proposal(source, candidate, hardware, validate_implementation=source_casts_only)
    # Whole-region storage overlaps conservatively, unlike three serial source steps.
    with pytest.raises(ValueError, match="storage bound 320.*capacity 256"):
        select_execution(source, (admitted,))
    with pytest.raises(ValueError, match="state budget"):
        select_execution(source, cpu, max_states=1)
