"""Plan original-ELF counter intake without observations, execution or qualification."""

from __future__ import annotations

import copy

from . import archive as A
from . import contracts as C

SCHEMA = "merlin.phase1.original_count_intake_plan.v1"


def prepare(inventory: dict, binding: dict | None = None) -> dict:
    """Keep every inspected v3 record; missing reviews remain explicit UNKNOWN.

    CLI callers freshly inspect the immutable tar. A reviewed binding delegates to
    archive.prepare and can feed its unchanged observation_plan to archive-attach.
    Metadata here grants neither an execution lifetime nor timing authority.
    """
    if (
        inventory.get("schema") != "merlin.phase1.spike_archive_inventory.v1"
        or inventory.get("archive_schema") != A.TRAINING_ARCHIVE
    ):
        raise C.ContractError("count intake requires the inspected per-record v3 archive")
    C.sha(inventory.get("archive_sha256"), "archive")
    C.sha(inventory.get("index_sha256"), "index")
    records = inventory.get("records")
    if (
        not isinstance(records, list)
        or len(records) > C.MAX_RECORDS
        or inventory.get("capsule_records") != len(records)
    ):
        raise C.ContractError("count intake must retain the complete bounded record roster")
    observation_plan = A.prepare(inventory, binding) if binding is not None else None
    observations = {r["record_id"]: r for r in observation_plan["observations"]} if observation_plan else {}
    cases = {r["case_id"]: r for r in observation_plan["dataset_template"]["cases"]} if observation_plan else {}
    requests, groups, splits, join_groups, join_splits, seen, results = [], {}, {}, {}, {}, set(), set()
    for original in records:
        record_id = C.sha(original.get("record_id"), "original record")
        result_sha = C.sha(original.get("result_sha256"), "original result bytes")
        row = original.get("row")
        if not isinstance(row, dict):
            raise C.ContractError("count intake requires original record metadata")
        result = A._path(row.get("result"))
        if record_id in seen or result in results:
            raise C.ContractError("count intake cannot duplicate or replace original records")
        seen.add(record_id)
        results.add(result)
        request = {"record_id": record_id, "result_sha256": result_sha, "status": "UNKNOWN"}
        requests.append(request)
        if row.get("visibility") != "public":
            # Retain an opaque refusal, never publish hidden names or program pins.
            request.update(status="REFUSED", reason="not_public")
            continue
        request.update(result=result, capsule=row.get("capsule"))
        if not A._training_eligible(row):
            request.update(status="REFUSED", reason="not_training_eligible")
            continue
        artifacts = row.get("artifacts")
        A._artifacts(row)
        elf, source = artifacts["elf"], artifacts["input_interface"]
        if elf["path"] != row.get("elf") or elf["sha256"] != row.get("elf_sha256"):
            raise C.ContractError("count request ELF differs from its original artifact role")
        executed = (row.get("executions") or {}).get("L3")
        if executed and executed.get("elf_sha256") != elf["sha256"]:
            raise C.ContractError("count request differs from the originally executed ELF")
        source_group = C.sha(source["sha256"], "original source interface")
        groups.setdefault(source_group, []).append(record_id)
        join_group = (row.get("join") or {}).get("source_group_key")
        if join_group is not None:
            C.sha(join_group, "original join source group")
            join_groups.setdefault(join_group, []).append(record_id)
        observation = observations.get(record_id)
        case = cases[observation["case_id"]] if observation is not None else None
        if case is not None:
            if case["source"]["source_sha256"] != source_group or case["source_group_sha256"] != source_group:
                raise C.ContractError("reviewed source group differs from original source bytes")
            if splits.setdefault(source_group, case["split"]) != case["split"]:
                raise C.ContractError("compiler variants of one source cross dataset splits")
            if join_group is not None and join_splits.setdefault(join_group, case["split"]) != case["split"]:
                raise C.ContractError("original archive source variants cross dataset splits")
        missing = []
        if case is None:
            missing += ["reviewed_hardware_window_domain", "whole_source_split"]
        if observation is None or observation["scope_binding"] is None:
            missing.append("original_counter_pc_review")
        request.update(
            reason="missing_original_reviews" if missing else "ready_for_separate_original_observation",
            source_group_sha256=source_group,
            original_join_group_sha256=join_group,
            candidate_sha256=row.get("candidate_sha256"),
            shape=copy.deepcopy(row.get("shape")),
            split=case["split"] if case else None,
            elf_member="run/" + elf["path"],
            elf_sha256=elf["sha256"],
            source_interface=copy.deepcopy(source),
            command_buffer=copy.deepcopy(artifacts["command_buffer"]),
            original_harness=copy.deepcopy(artifacts["harness"]),
            original_rtl_console=copy.deepcopy(artifacts["rtl_console"]),
            timing_selection=copy.deepcopy(row["timing_selection"]),
            simulator_provenance=copy.deepcopy(row["sim_provenance"]),
            original_result_sha256=result_sha,
            original_label={
                key: copy.deepcopy(row.get(key))
                for key in ("engine", "cycles", "cycles_measured", "status", "measured_now", "cycle_scope")
            },
            scope_binding=copy.deepcopy(observation["scope_binding"]) if observation else None,
            retained_counter_review=copy.deepcopy(artifacts.get("annotated_spike_counter_review")),
            alternate_functional_execution=copy.deepcopy((row.get("executions") or {}).get("L2")),
            wrapper_policy="count_original_gsim_elf; alternate_ELF_is_not_a_substitute",
            missing=missing,
            fresh_count_report=None,
            qualified_timing=None,
        )
    return {
        "schema": SCHEMA,
        "archive_sha256": inventory["archive_sha256"],
        "index_sha256": inventory["index_sha256"],
        "inventory_sha256": C.digest(inventory),
        "requests": requests,
        "source_groups": [
            {"source_group_sha256": key, "records": sorted(value), "split": splits.get(key)}
            for key, value in sorted(groups.items())
        ],
        "original_join_groups": [
            {"source_group_sha256": key, "records": sorted(value), "split": join_splits.get(key)}
            for key, value in sorted(join_groups.items())
        ],
        "domain": copy.deepcopy(binding["domain"]) if binding is not None else None,
        "observation_plan": observation_plan,
        "total_records": len(records),
        "unresolved_records": sum(r["status"] == "UNKNOWN" for r in requests),
        "refused_records": sum(r["status"] == "REFUSED" for r in requests),
        "launch_authority": False,
        "qualification_authority": "NONE",
        "launches": [],
        "required_before_execution": [
            "Reinspect the exact archive and reopen original ELF/source/caller/console pins.",
            "Review original counter-PC window per record; preserve source splits and all refusals.",
            "Use original independently checked observer/runtime owners and bounded shared resources.",
            "Do not replace the original gSIM ELF by its differently wrapped functional sibling.",
        ],
        "required_before_ranking": [
            "Attach fresh original reports through archive-attach using the reviewed observation_plan.",
            "Keep paid RTL labels unchanged; qualify held groups and enabled capacity domains independently.",
        ],
    }
