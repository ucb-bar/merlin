"""Bind and validate captured public timing evidence without executing a simulator."""
# ruff: noqa: E402 -- Select the task's existing package roots before importing optional tooling.

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from pathlib import Path

import common as C

C.configure()
from merlin_timing import archive as A
from merlin_timing import contracts as T
from merlin_timing import design as D
from merlin_timing import fitting as F
from merlin_timing import validation as V

ARCHIVE = (
    C.P1
    / "merlin/out/runs/gemmini/phase1/el4-fire-aligned-phase1-20261017-v3"
    / "capsule_timing_training_d4c730b33e12.tar"
)
ARCHIVE_SHA = "f815a0d2f4d80415b1e89e6ccdd62a0eb7af039287f539c91bd235451f8cfd5a"


def mechanism(cb):
    """Describe the original command buffer; never select compiler behavior."""
    ops = {x["opcode"] for x in cb.get("commands", [])}
    if "CONV2D" in ops:
        return "conv"
    if ops & {"ATTENTION_QK", "ATTENTION_PV", "BATCHED_MATMUL"}:
        return "attention"
    if "RESIDUAL_ADD" in ops:
        return "residual"
    if ops == {"MOVEMENT"}:
        return "movement"
    if sum(x["opcode"] == "COMMIT" for x in cb.get("commands", [])) > 1:
        return "resident_chain"
    return "matmul"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=C.OUT / "calibration-offline-v2")
    root = parser.parse_args(argv).output_dir
    if root.exists():
        parser.error("retain the previous output and select a fresh --output-dir")
    inventory = A.inspect_archive(ARCHIVE, ARCHIVE_SHA)
    root.mkdir(parents=True)
    C.save(root / "archive-inventory.json", inventory)
    rows = [x["row"] for x in inventory["records"]]
    reports, status, unresolved, gaps, by_case = {}, collections.Counter(), collections.Counter(), [], {}
    with tarfile.open(ARCHIVE) as archive:

        def artifact(row, role):
            ref = row["artifacts"].get(role)
            return json.load(archive.extractfile("run/" + ref["path"])) if ref else None

        first = next(
            artifact(r, "annotated_spike_report") for r in rows if r["artifacts"].get("annotated_spike_report")
        )
        selection = rows[0]["timing_selection"]
        pin = selection["counter_semantics"]["source_pins"][0]
        if C.file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("original CSR accessor/ISA/clock source changed")
        review = {
            "timing_selection": selection,
            "cycle_scope": rows[0]["cycle_scope"],
            "sim_provenance_sha256": T.digest(rows[0]["sim_provenance"]),
            "cache_protocol": None,
            "counter": {
                "unit": "cycles",
                "width_bits": 64,
                "wrap": "modulo",
                "clock": selection["counter_semantics"]["clock"],
                "max_delta_cycles": selection["max_delta_cycles"],
                "sources": {role: pin["sha256"] for role in ("accessor", "isa", "clock")},
            },
        }
        for row in rows:
            source_sha = row["artifacts"]["input_interface"]["sha256"]
            split = ("train", "train", "train", "calibration", "test")[int(source_sha[:8], 16) % 5]
            case = {
                "case_id": T.digest([row["elf_sha256"], row["harness_sha256"], source_sha]),
                "source_group_sha256": source_sha,
                "mechanism_family": mechanism(artifact(row, "command_buffer")),
                "split": split,
                "source": {
                    "visibility": "public",
                    "phase": "phase1",
                    "manifest_sha256": inventory["index_sha256"],
                    "source_sha256": source_sha,
                },
                "elf_sha256": row["elf_sha256"],
                "input_sha256": T.digest(
                    {"schema": "merlin.embedded_input_context.v1", "harness_sha256": row["harness_sha256"]}
                ),
                "scope_sha256": first["domain"]["scope_sha256"],
                "device_sha256": first["domain"]["device_sha256"],
            }
            case["workload_sha256"] = T.workload(case)
            native = artifact(row, "annotated_spike_report")
            if native:
                reports[row["result"]] = {"sha256": T.digest(native), "report": native}
                status["complete" if native.get("timing_complete") else "incomplete"] += 1
                unresolved.update(native.get("unresolved", []))
            else:
                status["missing"] += 1
            counter = artifact(row, "annotated_spike_counter_review")
            scope = None
            if counter is not None:
                if counter["elf_sha256"] != row["elf_sha256"]:
                    raise ValueError("counter review belongs to a different executed ELF")
                scope = {k: counter[k] for k in ("begin_pc", "end_pc", "occurrence")}
                scope["evidence_sha256"] = T.digest(counter)
            else:
                gaps.append(
                    {
                        "result": row["result"],
                        "elf_sha256": row["elf_sha256"],
                        "reason": "original counter review absent; no replacement or boundary inferred",
                    }
                )
            bound = by_case.setdefault(
                case["case_id"],
                {
                    "case": case,
                    "results": [],
                    "candidate_sha256": [],
                    "shape_sha256": T.digest(row["shape"]),
                    "counter_scopes": {},
                },
            )
            if bound["case"] != case:
                raise ValueError("one execution changed its original source or split")
            bound["results"].append(row["result"])
            bound["candidate_sha256"].append(row["candidate_sha256"])
            bound["counter_scopes"][row["result"]] = scope
        for bound in by_case.values():
            bound["candidate_sha256"] = sorted(set(bound["candidate_sha256"]))
        binding = {
            "schema": A.BINDING_V2,
            "archive_sha256": ARCHIVE_SHA,
            "domain": first["domain"],
            "feature_names": [
                "cpu.integer",
                "cpu.load",
                "cpu.store",
                "cpu.branch",
                "target.commands",
                "target.requested_read_bytes",
            ],
            "review": review,
            "cases": list(by_case.values()),
        }
    C.save(root / "reviewed-binding.json", binding)
    if A.file_identity(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("original archive changed during offline binding")
    plan = A.prepare(inventory, binding)
    C.save(root / "observation-plan.json", plan)
    attachment = {
        "schema": "merlin.phase1.spike_observations.v1",
        "plan_sha256": T.digest(plan),
        "reports": {x["record_id"]: reports[x["result"]] for x in plan["observations"] if x["result"] in reports},
    }
    C.save(root / "original-observations.json", attachment)
    dataset = A.attach(plan, attachment)
    C.save(root / "dataset.json", dataset)
    design = D.check(plan)
    C.save(root / "design-check.json", design)
    native = V.validate(dataset, prediction_kind="event_timing")
    C.save(root / "event-validation.json", native)
    try:
        screen = F.fit(dataset)
        C.save(root / "screen.json", screen)
        empirical = V.validate(dataset, screen)
        C.save(root / "screen-validation.json", empirical)
    except T.ContractError as error:
        empirical = {"status": "UNKNOWN", "reason": str(error)}
    summary = {
        "schema": "phase2.offline_calibration_status.v2",
        "archive_sha256": ARCHIVE_SHA,
        "archive_prepare": "passed",
        "archive_attach": "passed",
        "cases": len(dataset["cases"]),
        "observations": len(plan["observations"]),
        "attached_records": len(dataset["records"]),
        "source_groups": len({c["source_group_sha256"] for c in dataset["cases"]}),
        "split_cases": dict(collections.Counter(c["split"] for c in dataset["cases"])),
        "reports": dict(status),
        "unresolved": dict(unresolved),
        "intake_gaps": gaps,
        "native_validation": native,
        "empirical_validation": empirical,
        "design_check": design,
        "prediction_authority": "NONE",
        "launches": [],
        "copied_verdicts_are_fresh_measurements": False,
        "capacity_extrapolation": "UNKNOWN; no qualified native model or capacity-scale observations",
        "boundary": "original kernel entry through completion; no complete-model performance claim",
        "artifacts": {p.name: C.file_hash(p) for p in root.glob("*.json")},
    }
    C.save(root / "status.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "archive_prepare",
                    "archive_attach",
                    "cases",
                    "observations",
                    "attached_records",
                    "source_groups",
                    "reports",
                    "prediction_authority",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
