"""Run every selected region through ordinary reconstruction or source hypotheses.

Batching changes orchestration, never source semantics, schedules or qualification.
An explicitly selected factory owns graph/reference alignment and compiler invocation.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from merlin.kernels.reference_inventory import digest, read_pinned

MAX_REGIONS = 4096


def reference_jobs(document, index, *, root, workspace):
    """Select one pinned reusable graph/evidence binding for the complete batch."""
    from .reconstruction_reference import _endpoint

    binding = document["region_binding"]
    if binding.get("factory") == __name__ + ":reference_jobs":
        raise ValueError("region binding cannot recursively select itself")
    if "artifact" in binding:
        artifact = binding["artifact"]
        read_pinned(Path(root) / artifact["path"], artifact["sha256"])
    factory, pin, _ = _endpoint(binding, root=Path(root))
    jobs = factory(document, index, root=Path(root), workspace=Path(workspace))
    if not isinstance(jobs, dict) or "regions" not in jobs:
        raise ValueError("region binding must return the complete indexed regions")
    return dict(jobs, batch_binding=pin)


def _pin(record, *, root):
    if not isinstance(record, dict):
        raise ValueError("region artifact needs a pinned ordinary file")
    path = (Path(root) / record["path"]).resolve()
    raw = read_pinned(path, record["sha256"])
    return dict(record, path=str(path), sha256=digest(raw))


def _gap(kind, goal, reason, action):
    return dict(kind=kind, goals=[goal], reason=reason, next_action=action)


def _tasks(result, *, context):
    from .reconstruction_learning import ROUTES, learning_tasks

    gaps = []
    for original in result.get("gaps", ()):
        gap = deepcopy(original)
        kind = gap["kind"]
        if kind not in ROUTES:
            gap.update(kind="missing_evidence", original_kind=kind)
        gap["goals"] = gap.get("goals") or result.get("goals") or ["evidence." + kind]
        witness = dict(region_id=context["region_id"])
        gap["witnesses"] = list(gap.get("witnesses", ()))
        if witness not in gap["witnesses"]:
            gap["witnesses"].append(witness)
        gaps.append(gap)
    result["gaps"] = gaps
    return [row.to_dict() for row in learning_tasks(
        dict(goals=result.get("goals", ()), gaps=gaps), context=context)]


def _hypothesis_qualification_gaps(goals):
    gaps = [
        _gap("missing_evidence", "source.reference_region_equivalence",
             "source-only compilation does not establish the complete expert/source region match",
             "prove_complete_graph_layout_and_reference_binding"),
        _gap("missing_evidence", "numeric.reference_boundary_equivalence",
             "source-only emission does not qualify reference quantization or numerical execution",
             "verify_original_numeric_boundaries_and_actual_emitted_output"),
        _gap("missing_evidence", "performance.complete_region",
             "source-only emission has no independently matched complete-region elapsed timing",
             "measure_independent_matched_complete_region"),
    ]
    if "schedule.consumed" in goals:
        gaps.append(_gap("adapter_gap", "schedule.consumed",
            "source-only compilation has no independently validated schedule.plan decision, "
            "existing consumer and actual emitted-program binding; provider claims cannot establish consumption",
            "bind_checked_schedule_plan_consumer_and_actual_program_with_independent_owner"))
    return gaps


def _hypothesis(command, job, *, root, workspace, context):
    """Compile independently selected source without crediting reference parity."""
    result = dict(status="unresolved", goals=list(job.get("goals", ())),
                  gaps=deepcopy(job.get("gaps", ())), reference_equivalence=False,
                  performance_qualification=False, established_mechanisms=[])
    if job.get("source_binding", {}).get("standalone") is False:
        result["gaps"].append(_gap("representation_gap", "source.standalone_region",
            "the pinned module is the complete graph, not an outlined source region",
            "outline_complete_source_region_with_original_boundary_values"))
        result["learning_tasks"] = _tasks(result, context=context)
        return result
    source = _pin(job["source"], root=root)
    result["source"] = source
    if command == "analyze":
        result["gaps"].extend(_hypothesis_qualification_gaps(job.get("goals", ())))
        result["status"] = "analyzed" if not result["gaps"] else "unresolved"
        result["learning_tasks"] = _tasks(result, context=context)
        return result
    provider = job["hypothesis"]
    if not callable(provider):
        raise ValueError("source hypothesis requires an explicitly selected callable")
    receipt = provider(source=source, workspace=workspace)
    # A provider cannot replace its selected input during compiler execution.
    _pin(source, root=root)
    if not isinstance(receipt, dict):
        raise ValueError("source compiler must return its artifact/reanalysis mapping")
    artifacts = [_pin(row, root=workspace) for row in receipt.get("artifacts", ())]
    result["hypothesis_artifacts"] = artifacts
    result["gaps"].extend(deepcopy(receipt.get("gaps", ())))
    program = receipt.get("program")
    if program is not None:
        program = _pin(program, root=workspace)
        if not Path(program["path"]).is_relative_to(workspace.resolve()):
            raise ValueError("source hypothesis program must be emitted in its owned workspace")
        result["program"] = program
        analysis = receipt.get("reanalysis")
        if (not isinstance(analysis, dict) or analysis.get("program_sha256") != program["sha256"]
                or not isinstance(analysis.get("observations"), dict) or not analysis["observations"]):
            result["gaps"].append(_gap("adapter_gap", "emission.actual_program_reanalysis",
                "emitted source program lacks reanalysis bound to its actual bytes",
                "reanalyze_actual_emitted_program"))
        else:
            analysis = deepcopy(analysis)
            analysis["artifacts"] = [_pin(row, root=workspace) for row in analysis.get("artifacts", ())]
            result.update(status="compiled_hypothesis", reanalysis=analysis)
    elif not result["gaps"]:
        result["gaps"].append(_gap("adapter_gap", "emission.actual_program",
            "source compiler returned no emitted program", "bind_actual_source_compiler_emission"))
    # Receipt status, qualification and mechanism assertions carry no authority.
    result["gaps"].extend(_hypothesis_qualification_gaps(job.get("goals", ())))
    result["learning_tasks"] = _tasks(result, context=context)
    return result


def execute_batch(command, batch, result, *, pin, runner, root, workspace, **options):
    """Keep every region and its own failures; unsupported neighbors do not gate it."""
    jobs = batch["regions"]
    if not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= MAX_REGIONS:
        raise ValueError("region batch requires a bounded complete nonempty sequence")
    identities = [job.get("region_id") if isinstance(job, dict) else None for job in jobs]
    if (any(type(value) is not str or not value for value in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("each batch region requires a distinct nonempty identity")
    if workspace is None:
        raise ValueError("region batches require an explicitly owned workspace")
    workspace = Path(workspace)
    pin = batch.get("batch_binding", pin)
    selected = result["inventory"]
    regions = []
    for ordinal, job in enumerate(jobs):
        region_id = identities[ordinal]
        directory = workspace / (str(ordinal).zfill(4) + "-" + digest(region_id.encode())[:16])
        directory.mkdir(parents=True, exist_ok=True)
        context = dict(manifest_sha256=selected["manifest_sha256"], adapter=pin, region_id=region_id)
        # Intake adapters may diagnose finer evidence gaps than the learning router.
        normalized = dict(goals=list(job.get("goals", ())), gaps=deepcopy(job.get("gaps", ())))
        _tasks(normalized, context=context)
        job = dict(job, gaps=normalized["gaps"])
        if "hypothesis" in job:
            row = _hypothesis(command, job, root=root, workspace=directory, context=context)
        else:
            inventory = deepcopy(selected)
            reference_ids = job.get("reference_ids")
            if reference_ids is not None:
                inventory["references"] = [value for value in inventory["references"]
                                           if value["id"] in reference_ids]
            row = runner(command, job, dict(schema=result["schema"], mode=result["mode"],
                         command=command, inventory=inventory, gaps=[], performance_qualification=False),
                         pin=dict(pin, region_id=region_id), workspace=directory, **options)
            # The complete parent inventory is retained once, not duplicated per region.
            row.pop("inventory", None)
            if row.get("gaps") and not row.get("learning_tasks"):
                row["learning_tasks"] = _tasks(row, context=context)
        for key in ("reference_ids", "source_binding", "reference_bindings", "role", "binding_evidence",
                    "source_region", "reference_region", "reference_cca", "evidence"):
            if key in job:
                value = job[key]
                row[key] = deepcopy(value.to_dict() if hasattr(value, "to_dict") else value)
        if "source" in job and "source" not in row:
            row["source"] = _pin(job["source"], root=root)
        row.update(region_id=region_id, ordinal=ordinal)
        regions.append(row)
    counts = dict(sorted(Counter(row["status"] for row in regions).items()))
    result.update(regions=regions, batch=dict(regions=len(regions), status_counts=counts,
                  complete_selection=True, binding=batch.get("batch_binding", pin)),
                  status=("reconstructed" if all(row["status"] == "reconstructed" for row in regions)
                          else "analyzed" if command == "analyze" else "unresolved"),
                  performance_qualification=False)
    result["adapter"] = pin
    return result


def experiment_packets(index, *, context, history=(), budget=None):
    """Shard task budgets deterministically, retaining all selected diagnoses."""
    from .reconstruction_experiments import build_experiment_packet

    packets, offset = [], 0
    while True:
        packet = build_experiment_packet(index, context=context, history=history,
                                         budget=budget, diagnosis_offset=offset)
        packets.append(packet)
        offset = packet["coverage"]["next_offset"]
        if offset is None:
            return packets
