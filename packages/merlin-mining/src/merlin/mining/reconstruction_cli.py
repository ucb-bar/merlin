"""Reference-guided intake, executable reconstruction, and automatic gap learning.

An explicitly selected adapter factory connects source semantics, generated target
dialect and evaluator ownership. The CLI never guesses a backend or starts a campaign.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

from merlin.kernels.reference_inventory import (
    digest,
    inventory,
    load_manifest,
    read_pinned,
    select_representatives,
)


def _write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(record, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _factory(selection):
    module_name, separator, name = selection.partition(":")
    if not separator or not name.isidentifier() or not all(part.isidentifier() for part in module_name.split(".")):
        raise ValueError("adapter selection requires module:factory")
    module = importlib.import_module(module_name)
    factory = getattr(module, name)
    source = inspect.getsourcefile(factory)
    if not callable(factory) or source is None:
        raise ValueError("selected adapter must be an inspectable factory")
    path = Path(source).resolve()
    return factory, {"entrypoint": selection, "source": str(path), "sha256": digest(path.read_bytes())}


def _record(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "summary"):
        return value.summary()
    if hasattr(value, "record"):
        return value.record()
    return value


def _regions(job):
    from merlin.kernels.kernel_evidence import SemanticRegion, compare_regions

    reference, source = job.get("reference_region"), job.get("source_region")
    if reference is None or source is None:
        return {
            "status": "unknown",
            "differences": [
                {
                    "kind": "missing_evidence",
                    "reason": "adapter has not reconstructed both expert and frontend semantic regions",
                }
            ],
        }
    if isinstance(reference, dict):
        reference = SemanticRegion.from_record(reference)
    if isinstance(source, dict):
        source = SemanticRegion.from_record(source)
    return _record(compare_regions(reference, source))


def analyze(document, *, root, per_family=2):
    """Pinned corpus intake plus structured evidence analysis when present."""
    index = inventory(document, root=root)
    from merlin.compile.scheduling.api_catalog import scheduling_capabilities
    from merlin.compile.scheduling.library_sources import library_modules

    version = document.get("library_version", 23)
    index["library"] = {
        "version": version,
        "modules": library_modules(version),
        "catalog": scheduling_capabilities(version),
        "scope": "discovery only; executable bindings supplied by selected adapter",
    }
    index["representatives"] = select_representatives(index, per_family=per_family)
    for row in index["references"]:
        if "semantic_region" in row:
            from merlin.kernels.kernel_evidence import SemanticRegion

            row["semantic_region"] = SemanticRegion.from_record(row["semantic_region"]).to_dict()
            row["semantic_region_authority"] = "selected declaration; source/ISA reconstruction still required"
        if row["format"] == "semantic_region":
            from merlin.kernels.kernel_evidence import SemanticRegion

            row["semantic_region"] = SemanticRegion.from_record(row["analysis"]["record"]).to_dict()
            row["gaps"] = [g for g in row["gaps"] if g["next_action"] != "recover_semantic_region"]
        if row["format"] == "normalized_commands":
            from merlin.kernels.kernel_evidence import (
                ArtifactPin,
                KernelEvidence,
                analyze_normalized_commands,
            )

            record = row["analysis"]["record"]
            commands = record.get("commands") if isinstance(record, dict) else record
            if not isinstance(commands, list) or not all(isinstance(v, dict) for v in commands):
                row["gaps"].append(
                    _gap("missing_evidence", "unsupported normalized capture shape", "repair_trace_projection")
                )
                continue
            selection = next(v for v in document["references"] if v["id"] == row["id"])
            evidence = KernelEvidence(
                row["id"],
                artifacts=tuple(ArtifactPin.from_record(v) for v in row["artifacts"]),
                capture_kind="normalized_commands",
                complete=selection.get("complete", False),
                visibility="public" if row["visibility"] == "public_reference" else row["visibility"],
            )
            row["evidence"] = evidence.to_dict()
            try:
                row["schedule_model"] = analyze_normalized_commands(
                    commands,
                    evidence_id=evidence.identity,
                    complete=evidence.complete,
                    source_kind=evidence.capture_kind,
                ).to_dict()
            except ValueError as error:
                row["gaps"].append(_gap("missing_evidence", str(error)[:512], "repair_trace_projection"))
    return index


def _gap(kind, reason, next_action):
    return {"kind": kind, "reason": reason, "next_action": next_action}


def _execute(
    command,
    document,
    *,
    root,
    adapter_selection=None,
    max_candidates=8,
    max_search_steps=4096,
    proposal_attempts=3,
    workspace=None,
    per_family=2,
):
    """One automatable invocation; unresolved requirements are data, not prompts."""
    index = analyze(document, root=root, per_family=per_family)
    result = {
        "schema": "merlin.kernel_reconstruction_run.v1",
        "mode": "reference_guided",
        "command": command,
        "inventory": index,
        "gaps": [],
        "performance_qualification": False,
    }
    selection = adapter_selection or document.get("adapter")
    automatic = selection is None and ("target_binding" in document or "region_binding" in document)
    if automatic:
        selection = ("merlin.mining.reconstruction_batch:reference_jobs" if "region_binding" in document
                     else "merlin.mining.reconstruction_reference:reference_job")
    if command == "analyze" and selection is None:
        result["status"] = "analyzed"
        return result
    if selection is None:
        result.update(
            status="unresolved",
            gaps=[
                _gap(
                    "adapter_gap",
                    "no executable frontend/target/library adapter selected",
                    "construct_reconstruction_adapter",
                )
            ],
        )
        return result
    factory, pin = _factory(selection)
    job = factory(document, index, root=root, workspace=workspace) if automatic else factory(document, index)
    if not isinstance(job, dict):
        raise ValueError("adapter factory must return its reconstruction job mapping")
    options = dict(max_candidates=max_candidates, max_search_steps=max_search_steps,
                   proposal_attempts=proposal_attempts, workspace=workspace)
    if "regions" in job:
        from .reconstruction_batch import execute_batch

        return execute_batch(command, job, result, pin=pin, runner=_execute_job, root=root, **options)
    return _execute_job(command, job, result, pin=pin, **options)


def _execute_job(command, job, result, *, pin, max_candidates, max_search_steps,
                 proposal_attempts, workspace):
    """Execute one ordinary reconstruction job, also reused by region batches."""
    index = result["inventory"]
    learning_context = dict(manifest_sha256=index["manifest_sha256"], adapter=pin)
    if "region_id" in pin:
        learning_context["region_id"] = pin["region_id"]
    result["adapter"] = pin
    if "automatic_reference" in job:
        result["automatic_reference"] = job["automatic_reference"]
        recovered = job["automatic_reference"].get("semantic_region")
        selected_reference = job["automatic_reference"].get("selected_reference")
        if isinstance(recovered, dict) and recovered.get("complete") is True:
            for reference in index["references"]:
                if reference["id"] != selected_reference:
                    continue
                remaining, resolved = [], []
                for gap in reference.get("gaps", ()):
                    if gap["next_action"] == "recover_semantic_region":
                        resolved.append(
                            dict(
                                gap,
                                resolution="complete source/trace semantic recovery",
                                evidence_id=recovered["region_id"],
                            )
                        )
                    else:
                        remaining.append(gap)
                reference["gaps"] = remaining
                reference["resolved_gaps"] = [*reference.get("resolved_gaps", ()), *resolved]
                reference["semantic_region"] = recovered
                reference["semantic_region_authority"] = "complete selected source/native trace reconstruction"
    if job.get("gaps"):
        from merlin.mining.reconstruction_learning import learning_tasks

        result.update(
            status="unresolved",
            gaps=job["gaps"],
            learning_tasks=[
                task.to_dict()
                for task in learning_tasks(
                    {"goals": tuple(job.get("goals", ("source.complete_semantic_region",))), "gaps": job["gaps"]},
                    context=learning_context,
                )
            ],
        )
        return result
    result["region_comparison"] = comparison = _regions(job)
    if comparison["status"] != "equivalent":
        result.update(
            status="unresolved",
            gaps=[
                _gap(
                    "semantic_mismatch" if comparison["status"] == "mismatch" else "missing_evidence",
                    "expert and frontend fused regions are not established equivalent",
                    "recover_or_match_semantic_region",
                )
            ],
        )
        return result
    if command == "analyze":
        result["status"] = "analyzed"
        return result
    if "adapter" not in job:
        result.update(status="unresolved", gaps=[_gap("adapter_gap",
            "region has no executable library/target binding", "construct_reconstruction_adapter")])
        return result
    if command == "reconstruct":
        from merlin.kernels.reconstruction import reconstruct

        reconstruction = reconstruct(
            job["adapter"],
            goals=tuple(job.get("goals", ())),
            reference_cca=job.get("reference_cca"),
            max_candidates=max_candidates,
            max_search_steps=max_search_steps,
        )
        summary = reconstruction.summary()
        artifacts = getattr(job["adapter"].emission, "artifact_records", None)
        if callable(artifacts):
            result["target_artifacts"] = artifacts()
        from merlin.mining.reconstruction_learning import learning_tasks

        result.update(
            status="reconstructed" if any(v["satisfied"] for v in summary["candidates"]) else "unresolved",
            reconstruction=summary,
            learning_tasks=[
                v.to_dict()
                for v in (
                    *learning_tasks(
                        reconstruction, context=learning_context
                    ),
                    *learning_tasks(
                        {"goals": ("dispatch.actual_instruction_projection",), "gaps": job.get("comparison_gaps", ())},
                        context=learning_context,
                    ),
                )
            ],
        )
        result["gaps"].extend(job.get("comparison_gaps", ()))
        return result
    if command != "learn":
        raise ValueError("unsupported reconstruction command")
    from merlin.mining.reconstruction_learning import LearningBudget, run_learning

    if "learning" not in job:
        result.update(
            status="unresolved",
            gaps=job.get(
                "learning_gaps",
                [
                    _gap(
                        "adapter_gap",
                        "selected adapter has no independent automatic learning gates",
                        "bind_independent_learning_owner",
                    )
                ],
            ),
        )
        return result
    options = dict(job["learning"])
    if workspace is not None:
        options["workspace"] = Path(workspace)
    options["budget"] = LearningBudget(
        max_candidates=max_candidates, max_search_steps=max_search_steps, proposal_attempts=proposal_attempts
    )
    result.update(status="learning_completed", learning=_record(run_learning(**options)))
    return result


def execute(command, document, *, root, **options):
    """Run reconstruction and, when selected, derive next experiments automatically."""
    binding = document.get("experiment_context")
    context, history, inputs = None, [], []
    if binding is not None:
        from .reconstruction_history import load_history

        def selected(value):
            path = (Path(root) / value["path"]).resolve()
            raw = read_pinned(path, value["sha256"])
            inputs.append({"path": str(path), "sha256": digest(raw)})
            return json.loads(raw)

        context = selected(binding)
        for selection in document.get("experiment_history", ()):
            history_root = (Path(root) / selection["path"]).resolve().parent
            history.extend(load_history(selected(selection), root=history_root))
    result = _execute(command, document, root=root, **options)
    if context is not None:
        from .reconstruction_experiments import build_experiment_packet

        if "regions" in result:
            from .reconstruction_batch import experiment_packets

            result["experiment_packets"] = experiment_packets(result, context=context, history=tuple(history))
            result["experiment_packet"] = result["experiment_packets"][0]
        else:
            result["experiment_packet"] = build_experiment_packet(result, context=context, history=tuple(history))
        result["experiment_inputs"] = inputs
    return result


def markdown_report(result):
    """Compact indexed report; JSON remains the automatic experiment interface."""
    if result.get("command") == "steer":
        packet = result["experiment_packet"]
        rows = [
            "# Reference-guided next experiments", "", "Exact requests and evidence are retained in packet.json.", "",
        ]
        if "experiment_packets" in result:
            rows.extend([f"All diagnoses are retained across {len(result['experiment_packets'])} packet shards.",
                         "Select each pinned shard from packet-shards.json; no experiment is launched here.", ""])
        for task in packet["tasks"]:
            rows.extend([
                f"- {task['kind']}: {task.get('summary', task.get('next_action', 'inspect evidence'))}",
                f"  Status: {task['status']}. Goals: {', '.join(task['goals'])}.",
            ])
        rows.extend(["", "Requests confer no numerical, timing, or compiler qualification.", ""])
        return "\n".join(rows)
    index = result.get("inventory", {})
    rows = [
        "# Reference-guided compiler reconstruction",
        "",
        f"Status: {result.get('status', 'unknown')}. Mode: reference-guided.",
        "",
        "| Reference | Family | Source functions | Evidence gaps |",
        "|---|---|---:|---:|",
    ]
    for reference in index.get("references", []):
        rows.append(
            f"| {reference['id']} | {reference['family']} | "
            f"{len(reference['analysis'].get('functions', []))} | {len(reference['gaps'])} |"
        )
    if "regions" in result:
        rows.extend(["", "| Region | Status | Gaps |", "|---|---|---:|"])
        rows.extend(f"| {row['region_id']} | {row['status']} | {len(row.get('gaps', ()))} |"
                    for row in result["regions"])
    rows.extend(["", "## Actionable gaps", ""])
    for gap in result.get("gaps", []):
        rows.append(f"- {gap['kind']}: {gap['reason']} → {gap['next_action']}")
    for reference in index.get("references", []):
        for gap in reference.get("gaps", []):
            rows.append(f"- {reference['id']}: {gap['reason']} → {gap['next_action']}")
    rows.extend(
        [
            "",
            "Actual emitted paths, observations, provenance, and learning results are indexed in the JSON record.",
            "Source structure and issue order alone establish no hardware timing or performance parity.",
            "",
        ]
    )
    return "\n".join(rows)


def steer(index, context, *, history=(), budget=None):
    """Generate bounded next experiments from selected observations and history."""
    from .reconstruction_experiments import build_experiment_packet

    result = {
        "schema": "merlin.kernel_reconstruction_run.v1",
        "mode": "reference_guided",
        "command": "steer",
        "status": "planned",
        "performance_qualification": False,
    }
    if "regions" in index:
        from .reconstruction_batch import experiment_packets

        result["experiment_packets"] = experiment_packets(index, context=context, history=history, budget=budget)
        result["experiment_packet"] = result["experiment_packets"][0]
    else:
        result["experiment_packet"] = build_experiment_packet(index, context=context, history=history, budget=budget)
    return result


def _steer_command(args):
    from .reconstruction_history import load_history

    context = json.loads(args.context.read_text())
    history = []
    inputs = []
    for path in (args.input, args.context, *args.history):
        inputs.append({"path": str(path.resolve()), "sha256": digest(path.read_bytes())})
    for path in args.history:
        history.extend(load_history(json.loads(path.read_text()), root=path.resolve().parent))
    result = steer(json.loads(args.input.read_text()), context, history=tuple(history))
    result["experiment_inputs"] = inputs
    return _publish_result(args.out, result)


def _publish_packet(output, packet, *, context_pins=()):
    packet_path = (output / "packet.json").resolve()
    _write(packet_path, packet)
    selection = {
        "schema": "merlin.reference_experiment_selection.v1",
        "packet": {"path": str(packet_path), "sha256": digest(packet_path.read_bytes())},
        "current_context": packet["context"],
        "context_pins": [*context_pins, *packet["context"].get("context_pins", ())],
    }
    selection_path = (output / "selection.json").resolve()
    _write(selection_path, selection)
    return {
        "packet": str(packet_path), "selection": str(selection_path),
        "selection_sha256": digest(selection_path.read_bytes()),
    }


def _publish_result(output, result):
    _write(output / "index.json", result)
    published = (_publish_packet(output, result["experiment_packet"], context_pins=result.get("experiment_inputs", ()))
                 if "experiment_packet" in result else {})
    if "experiment_packets" in result:
        shards = [published]
        for position, packet in enumerate(result["experiment_packets"][1:], 1):
            shards.append(_publish_packet(output / "shards" / str(position).zfill(4), packet,
                                          context_pins=result.get("experiment_inputs", ())))
        shard_path = output / "packet-shards.json"
        _write(shard_path, dict(schema="merlin.reference_experiment_shards.v1", complete=True,
                               tasks=sum(len(row["tasks"]) for row in result["experiment_packets"]),
                               shards=shards))
        published = dict(published, packet_shards=str(shard_path))
    report = output / "report.md"
    report.write_text(markdown_report(result))
    os.chmod(report, 0o600)
    print(json.dumps({"status": result["status"], "index": str(output / "index.json"),
                      "report": str(report), **published}, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "reconstruct", "learn"):
        child = commands.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
        child.add_argument("--adapter", help="explicit trusted operator module:factory")
        child.add_argument("--per-family", type=int, choices=(1, 2), default=2)
        child.add_argument("--max-candidates", type=int, default=8)
        child.add_argument("--max-search-steps", type=int, default=4096)
        child.add_argument("--proposal-attempts", type=int, default=3)
    report = commands.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    steering = commands.add_parser("steer", help="produce controlled experiment requests for the author feedback loop")
    steering.add_argument("--input", type=Path, required=True, help="retained reconstruction index")
    steering.add_argument(
        "--context", type=Path, required=True, help="selected compiler, evaluator and public tuning context",
    )
    steering.add_argument(
        "--history", type=Path, action="append", default=[], help="pinned structured history selection",
    )
    steering.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "steer":
        return _steer_command(args)
    if args.command == "report":
        result = json.loads(args.input.read_text())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown_report(result))
        os.chmod(args.out, 0o600)
        return 0
    document, root = load_manifest(args.manifest)
    result = execute(
        args.command,
        document,
        root=root,
        adapter_selection=args.adapter,
        max_candidates=args.max_candidates,
        max_search_steps=args.max_search_steps,
        proposal_attempts=args.proposal_attempts,
        workspace=args.out / "learning",
        per_family=args.per_family,
    )
    return _publish_result(args.out, result)


if __name__ == "__main__":
    raise SystemExit(main())
