import yaml

from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    source_numerical_identity,
)
from merlin.system.topology import select_topology
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


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
