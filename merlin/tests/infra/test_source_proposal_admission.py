from dataclasses import replace

import pytest
import yaml
from corefixtures.offload_plan import example as legacy_example

from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    admit_original_cpu,
    admit_proposal,
    source_numerical_identity,
)
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


def example():
    legacy, alternatives, context = legacy_example()
    source = bind_source(legacy.module, entry=legacy.entry, context=context, graph_mode="ssa")
    alternatives = tuple(
        (
            replace(
                source.cpu_alternatives[item.nodes[0]],
                id=item.id,
                implementation=item.implementation,
                placement=item.placement,
            )
            for item in alternatives
        )
    )
    return (source, alternatives, context)


def topology(tmp_path, **edge_changes):
    document = {
        "schema": "merlin.execution_topology.v1",
        "resources": [
            {"id": "cpu0", "kind": "cpu", "features": ["rvv"], "shared_group": "cpu0"},
            {"id": "array0", "kind": "accelerator", "features": ["integer-matmul"], "shared_group": "array0"},
        ],
        "memory_domains": [{"id": "dram", "capacity_bytes": 4096, "alignment_bytes": 64}],
        "access": [
            dict(
                resource=resource,
                domain="dram",
                read=True,
                write=True,
                addressability="shared",
                visibility="coherent",
                completion="synchronous",
            )
            for resource in ("cpu0", "array0")
        ],
    }
    document["access"][1].update(edge_changes)
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(document))
    return select_topology(path)


def proposal(source, alternative):
    return SourceProposal(
        alternative,
        "test.checked.integer",
        "array0",
        "dram",
        "logical_tensor_v1",
        source.stamp.source_sha256,
        tuple((source.operation_ordinals[n] for n in alternative.nodes)),
        source_numerical_identity(source, alternative.nodes),
        ("integer-matmul",),
    )


def original_check(source, candidate):
    for index in candidate.alternative.nodes:
        operation = source.operations[index]
        dtypes = tuple((str(v.type.element_type) for v in (*operation.inputs, *operation.outputs)))
        recognize_signed_integer_matmul(operation, dtypes)


def test_exact_sites_shared_cpu_and_original_source_changes(tmp_path):
    source, alternatives, _ = example()
    selected = topology(tmp_path)
    admitted = admit_proposal(
        source, proposal(source, alternatives[1]), selected, validate_implementation=original_check
    )
    assert admitted.alternative.nodes == (7,)
    assert admitted.proposal.operation_ordinals != proposal(source, alternatives[0]).operation_ordinals
    cpu = tuple(
        (
            admit_original_cpu(source, n, selected, resource="cpu0", memory_domain="dram")
            for n in range(len(source.operations))
        )
    )
    assert all((p.proposal.resource == "cpu0" and p.proposal.abi == "inlined" for p in cpu))
    assert selected.resource("cpu0").shared_group == "cpu0"
    vector = replace(proposal(source, alternatives[0]), resource="cpu0", required_features=("rvv",))
    admit_proposal(source, vector, selected, validate_implementation=original_check).revalidate()
    selected.path.write_text(selected.path.read_text() + "\n# changed selected evidence\n")
    with pytest.raises(ValueError, match="topology changed"):
        admitted.revalidate()


@pytest.mark.parametrize("edge", [{"addressability": "separate"}, {"completion": "unknown"}])
def test_missing_memory_path_is_not_an_implicit_transfer(tmp_path, edge):
    source, alternatives, _ = example()
    with pytest.raises(ValueError, match="access|transfer, visibility or completion"):
        admit_proposal(
            source,
            proposal(source, alternatives[0]),
            topology(tmp_path, **edge),
            validate_implementation=original_check,
        )
