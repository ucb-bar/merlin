"""Public source-bound model build using the coordinator's fixed host provider.

This adapter never consumes a candidate whole-program LLVM artifact. Existing
model gates must independently inspect its ordinary device/source/native proof;
a completed build or numeric match does not authorize a tier by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.compile.materialized_model_inputs import MaterializedModelInputs
from merlin.compile.model_execution_inputs import file_sha256
from merlin.compile.owned_host_model import OwnedHostModelCompiler


def _selected_file(selection, role):
    record = selection.get(role)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"owned host build needs exact {role} selection")
    path = Path(record["path"])
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"owned host build {role} is indirect")
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"owned host build {role} changed")
    return path


def execute_owned_host_model(*, capsule, candidate_package, target, bundle,
                             provenance, verify, output, selection_record,
                             simulator, timeout):
    from merlin.targetgen.capsule_runner import _resolve_model_host_lane
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as coordinator
    from merlin_experiments.phase1.feedback import private_instruction_declaration as declaration

    if not isinstance(selection_record, dict):
        raise ValueError("public model needs coordinator-selected owned host build inputs")
    selection_path = _selected_file({"selection": selection_record}, "selection")
    selection = json.loads(selection_path.read_text())
    if set(selection) != {"board_catalog", "dts", "board", "arena_mb", "instruction_declaration", "rtl_facts"}:
        raise ValueError("owned host build selection fields are incomplete or unexpected")
    catalog = _selected_file(selection, "board_catalog")
    dts = _selected_file(selection, "dts")
    instruction = _selected_file(selection, "instruction_declaration")
    facts = _selected_file(selection, "rtl_facts")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("owned host model needs a fresh output directory")
    output.mkdir(parents=True)
    prepared = coordinator.prepare(declaration.read(instruction, target=target), target=target,
                                   output=(output / "instruction_selection").absolute())
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    experiment, host, identity = _resolve_model_host_lane(target, attrs.get("compile_dtype", "fp32"))
    service = OwnedHostModelCompiler.select(host_package=host, target=target,
                                           board_catalog=catalog, board=selection["board"], dts=dts)
    verify()
    materialized = MaterializedModelInputs.bind(
        bundle, artifacts=provenance["bundle"]["artifacts"],
        validator_identity=provenance["tool"], source_identity=provenance["source"],
    )
    receipt = service.compile(capture=bundle, candidate_package=candidate_package,
                              datapath=attrs.get("dtype"), linked_elf_admission=prepared.service,
                              output=output / "model", arena_mb=selection["arena_mb"],
                              timeout_s=timeout, run=simulator, rtl_facts=facts,
                              materialized_inputs=materialized)
    verify()
    audit = _audit_owned_host_device_build(bundle, receipt, target=target)
    prepared.verify()
    for role in ("board_catalog", "dts", "instruction_declaration", "rtl_facts"):
        _selected_file(selection, role)
    _selected_file({"selection": selection_record}, "selection")
    return {"status": "owned_host_complete_output_observed" if receipt["status"] == "verified_complete_output" else "compiled_not_run",
            "scope": "Merlin-owned host model with candidate closed device groups; mandatory source/device/tier gates remain independent",
            "host_provider": identity, "build": receipt, "source_device_join": audit,
            "instruction_selection": prepared.record(), "materialization": materialized.record()}


def _audit_owned_host_device_build(bundle, receipt, *, target):
    """Recompute original group identities and audit the actual linked objects."""
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api
    from merlin_experiments.phase1.feedback.private_group_provenance import (
        eligible_source_identity, join_routed_source_groups, routed_kernel_symbols,
    )
    from merlin_experiments.phase1.feedback.private_device_audit import audit_built_device_host_compute
    from merlin_experiments.phase1.feedback.private_full_models import _linked_symbols

    # Match the exact source passed to OwnedHostModelCompiler.plan_before_build.
    text = (Path(bundle) / "model.mlir").read_text()
    module = mq.parse(text)
    groups = groups_api.form_groups(module, target)
    groups_api.require_explained(groups_api.plan(module, target, groups=groups))
    source = {"eligible_groups": [], "eligible_group_provenance": [], "eligible_group_metrics": {}}
    for group in groups:
        if group.placement == groups_api.HOST:
            continue
        region, nodes = eligible_source_identity(group.root)
        source["eligible_groups"].append(group.index)
        source["eligible_group_provenance"].append({"source_group": group.index, "source_region_id": region, "source_node_ids": list(nodes)})
        values = (value for member in reversed(group.members) for value in member.results)
        for value in values:
            shape, _ = mq.type_shape_dtype(value.type)
            elements = 1
            for extent in shape:
                elements *= extent
            size = mq.value_bytes(value.type)
            if shape and elements > 0 and size > 0 and size % elements == 0:
                source["eligible_group_metrics"][group.index] = {"elements": elements, "element_bytes": size // elements}
                break
        if group.index not in source["eligible_group_metrics"]:
            raise ValueError("source-bound device group has no static output extent")
    output = receipt["output"]
    sidecar = output["device_sidecar"]
    path = Path(sidecar["path"])
    if file_sha256(path) != sidecar["sha256"]:
        raise ValueError("owned host device sidecar changed")
    emitted = json.loads(path.read_text())
    if emitted.get("skipped") or emitted.get("device") != target:
        raise ValueError("owned host model omitted eligible device groups")
    assigned = emitted.get("routed")
    joined = join_routed_source_groups(assigned, source)
    elf = Path(output["elf"])
    if file_sha256(elf) != output["elf_sha256"]:
        raise ValueError("owned host linked ELF changed")
    _linked_symbols(elf, routed_kernel_symbols(emitted))
    audits = audit_built_device_host_compute(elf.parent, assigned, source, target)
    return {"source": source, "joined_groups": joined, "static_device_host_compute": audits,
            "scope": "source identities joined to actual linked device objects; dynamic execution and required tiers remain separate"}
