from dataclasses import replace

import yaml

from corefixtures.offload_plan import example
from corefixtures.source_proposal_admission import original_check, proposal, topology
from merlin.compiler.backend.sourceproposal import admit_original_cpu, admit_proposal
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology


def roster(tmp_path, *, capacity=4096, extra_domain=False, shape=(3, 7, 5)):
    legacy, old_alternatives, context = example(shape=shape)
    source = bind_source(legacy.module, entry=legacy.entry, context=context, graph_mode="ssa")
    alternatives = tuple(
        (
            replace(
                source.cpu_alternatives[item.nodes[0]],
                id=item.id,
                implementation=item.implementation,
                placement=item.placement,
            )
            for item in old_alternatives
        )
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
        (
            admit_original_cpu(source, i, hardware, resource="cpu0", memory_domain="dram")
            for i in range(len(source.operations))
        )
    )
    candidates = [proposal(source, alt) for alt in alternatives]
    candidates[0] = replace(candidates[0], provider="vector", resource="cpu0", required_features=("rvv",))
    candidates[1] = replace(candidates[1], provider="array", memory_domain="other" if extra_domain else "dram")
    device = tuple(
        (admit_proposal(source, item, hardware, validate_implementation=original_check) for item in candidates)
    )
    return (source, hardware, cpu, device)
