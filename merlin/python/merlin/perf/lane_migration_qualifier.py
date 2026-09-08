"""Bounded numerical qualification for a source operation that changes execution lane.

This verifier is target-neutral: task kinds and exact source contraction semantics select it,
while a host-owned target adapter builds and executes the reduced programs.  It never executes,
recompiles, or extrapolates the full model.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile
from time import monotonic
from typing import Any, Callable

from merlin.frontends.linalg_mlir import parse_mlir_text
from .source_contraction_preparation import prepare_source_contraction
from .source_contraction_witness import _contract
from .source_program_pair import program_plan, text_digest


def _tasks(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks = program_plan(artifact.get("command_buffer") or {}).get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("lane migration needs an exact task list")
    return tasks


def _owners(tasks: list[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    owners: dict[int, Mapping[str, Any]] = {}
    for task in tasks:
        indices = task.get("source_op_indices")
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or index < 0 for index in indices)):
            raise ValueError("lane migration task has malformed source ownership")
        for index in indices:
            if index in owners:
                raise ValueError("lane migration source ownership is ambiguous")
            owners[index] = task
    return owners


def has_contraction_lane_migration(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """Detect a declared transition; the qualifier must still prove its source semantics."""
    try:
        before_owners = _owners(_tasks(before))
        after_tasks = _tasks(after)
    except (ValueError, KeyError, TypeError):
        return False
    for task in after_tasks:
        if task.get("kind") != "contraction":
            continue
        for index in task["source_op_indices"]:
            owner = before_owners.get(index)
            if owner is None or owner.get("kind") != "contraction":
                return True
    return False


def contraction_lane_migrations(source_text: str, before: Mapping[str, Any],
                                after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind each host→contraction task to one structurally recognized source operation."""
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("lane migration witness requires one single-block source entry")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    before_owners = _owners(_tasks(before))
    rows: list[dict[str, Any]] = []
    for task in _tasks(after):
        if task.get("kind") != "contraction":
            continue
        indices = task["source_op_indices"]
        old = [before_owners.get(index) for index in indices]
        if all(owner is not None and owner.get("kind") == "contraction" for owner in old):
            continue
        if (any(owner is None or owner.get("kind") != "host" for owner in old)
                or len({owner["task_index"] for owner in old}) != 1):
            raise ValueError("changed contraction task is not an exact single host-task migration")
        candidates = []
        for index in indices:
            if not 0 <= index < len(ops):
                raise ValueError("changed task source index lies outside the selected entry")
            try:
                _info, geometry = _contract(ops[index])
            except ValueError:
                continue
            candidates.append((index, geometry))
        if len(candidates) != 1:
            raise ValueError("changed contraction task does not bind one exact source contraction")
        index, geometry = candidates[0]
        batching = ("batched" if geometry["lhs_batched"] else "broadcast",
                    "batched" if geometry["rhs_batched"] else "broadcast")
        rows.append({
            "source_op_index": index,
            "before_task_index": old[0]["task_index"],
            "after_task_index": task["task_index"],
            "before_kind": "host", "after_kind": "contraction",
            "output_rank": geometry["output_rank"],
            "iteration_rank": geometry["iteration_rank"],
            "batch_rank": len(geometry["batch_shape"]),
            "operand_batching": {"lhs": batching[0], "rhs": batching[1]},
            "source_batch_shape": list(geometry["batch_shape"]),
            "source_geometry_mkn": [geometry["m"], geometry["k"], geometry["n"]],
        })
    if not rows:
        raise ValueError("no exact host-to-contraction source migration exists")
    return sorted(rows, key=lambda row: row["source_op_index"])


def representative_migrations(rows: list[Mapping[str, Any]], *, maximum: int = 4
                              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily cover every observed iterator rank and operand-batching mode."""
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("lane migration representative bound must lie in [1,8]")
    remaining_ranks = {row["iteration_rank"] for row in rows}
    remaining_batching = {tuple(row["operand_batching"][side] for side in ("lhs", "rhs"))
                          for row in rows}
    selected: list[dict[str, Any]] = []
    available = [dict(row) for row in rows]
    while remaining_ranks or remaining_batching:
        ranked = []
        for row in available:
            mode = tuple(row["operand_batching"][side] for side in ("lhs", "rhs"))
            coverage = int(row["iteration_rank"] in remaining_ranks) + int(mode in remaining_batching)
            ranked.append((coverage, -row["source_op_index"], row, mode))
        coverage, _order, winner, mode = max(ranked, key=lambda item: (item[0], item[1]))
        if coverage <= 0:
            break
        selected.append(winner)
        available.remove(winner)
        remaining_ranks.discard(winner["iteration_rank"])
        remaining_batching.discard(mode)
        if len(selected) == maximum and (remaining_ranks or remaining_batching):
            raise ValueError("observed lane-migration classes exceed the bounded representative budget")
    coverage = {
        "observed_iteration_ranks": sorted({row["iteration_rank"] for row in rows}),
        "qualified_iteration_ranks": sorted({row["iteration_rank"] for row in selected}),
        "observed_operand_batching": sorted({
            "/".join(row["operand_batching"][side] for side in ("lhs", "rhs")) for row in rows}),
        "qualified_operand_batching": sorted({
            "/".join(row["operand_batching"][side] for side in ("lhs", "rhs")) for row in selected}),
        "selection": "minimum greedy representatives covering source iterator rank and operand batching",
        "shape_policy": "source shapes classify only; independently bounded extents execute",
    }
    if (coverage["observed_iteration_ranks"] != coverage["qualified_iteration_ranks"]
            or coverage["observed_operand_batching"] != coverage["qualified_operand_batching"]):
        raise ValueError("selected representatives do not cover every observed migration class")
    return selected, coverage


class LaneMigrationContractionQualifier:
    """Execute exact reduced source witnesses through both bound compiler revisions."""

    def __init__(self, *, target: str, runtime_provider: Callable[..., Mapping[str, Any]],
                 abi_provenance: Mapping[str, Any], output: Path,
                 maximum_representatives: int = 4,
                 max_source_bytes: int = 8_000_000):
        if not target or not callable(runtime_provider) or not abi_provenance:
            raise ValueError("lane migration qualifier needs a target adapter, runtime, and ABI provenance")
        if type(max_source_bytes) is not int or not 1 <= max_source_bytes <= 16_000_000:
            raise ValueError("lane migration source byte bound must lie in [1,16000000]")
        self.target, self.runtime_provider = target, runtime_provider
        self.abi_provenance, self.output = dict(abi_provenance), Path(output)
        self.maximum_representatives = maximum_representatives
        self.max_source_bytes = max_source_bytes

    def __call__(self, *, candidate: Path, experiment: Any, timeout_s: float,
                 portfolio_member: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = monotonic()
        if (isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("lane migration qualification requires a finite positive budget")
        deadline = started + min(float(timeout_s), 300.0)
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="lane_migration_", dir=self.output))
        record: dict[str, Any] = {
            "schema": "lane_migration_contraction_qualification_v1", "status": "UNKNOWN",
            "abi_provenance": self.abi_provenance, "target": self.target,
            "full_model_executed": False, "full_model_recompiled": False,
            "full_shape_backend_correspondence": "UNPROVEN", "global_speedup_proven": False,
            "full_model_cycles": None, "bounded_programs_executed": 0,
            "proof_scope": "source-rank and operand-batching representatives at reduced extents",
            "representatives": [],
        }

        def remaining() -> float:
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("lane migration qualification exhausted its shared wall budget")
            return value

        try:
            if not callable(getattr(experiment, "selected_changed_portfolio_context", None)):
                raise ValueError("lane migration qualification requires member-aware artifacts")
            selected = experiment.selected_changed_portfolio_context(candidate, portfolio_member)
            member_binding = {"selection": selected["selection"],
                "previous": selected["previous"]["member_binding"],
                "current": selected["current"]["member_binding"]}
            record["portfolio_member_binding"] = member_binding
            before, after = (selected["previous"]["artifacts"], selected["current"]["artifacts"])
            prior_source, source = Path(before["interface"]), Path(after["interface"])
            if (source.is_symlink() or prior_source.is_symlink() or not source.is_file()
                    or not prior_source.is_file()
                    or source.stat().st_size > self.max_source_bytes
                    or prior_source.stat().st_size > self.max_source_bytes):
                raise ValueError("selected lane-migration source path is unsafe or oversized")
            source_text = source.read_text()
            if (prior_source.read_text() != source_text
                    or text_digest(source_text) != member_binding["current"]["source_sha256"]
                    or member_binding["previous"]["source_sha256"]
                    != member_binding["current"]["source_sha256"]):
                raise ValueError("selected lane-migration source or member identity changed")
            rows = contraction_lane_migrations(source_text, before, after)
            representatives, coverage = representative_migrations(
                rows, maximum=self.maximum_representatives)
            record.update(migrations=rows, representative_coverage=coverage,
                          source_sha256=member_binding["current"]["source_sha256"])
            entry = next(op.sym_name.data for op in parse_mlir_text(source_text).body.block.ops
                         if op.name == "func.func")
            for ordinal, row in enumerate(representatives):
                remaining()
                prepared = prepare_source_contraction(candidate=candidate, experiment=experiment,
                    comparison_arm="previous", source_op_index=row["source_op_index"], entry=entry,
                    max_m=2, max_n=2, max_k=3, max_batch_extent=2, max_macs=96,
                    expected_route_transition=("host", "contraction"),
                    expected_short_route_transition=(None, "contraction"),
                    portfolio_member=selected["selection"], output=work / f"prepare_{ordinal}",
                    max_source_bytes=self.max_source_bytes,
                    timeout_s=min(60.0, remaining()))
                representative = {"selection": row, "preparation": prepared}
                record["representatives"].append(representative)
                if prepared.get("status") != "prepared":
                    raise ValueError("lane-migration representative preparation refused: "
                                     + str(prepared.get("reason", "unknown")))
                extraction = prepared.get("extraction") or {}
                expected_extraction = {
                    "source_iteration_rank": row["iteration_rank"],
                    "operand_batching": row["operand_batching"],
                    "source_batch_shape": row["source_batch_shape"],
                    "source_geometry_mkn": row["source_geometry_mkn"],
                }
                if (any(extraction.get(key) != value for key, value in expected_extraction.items())
                        or prepared.get("expected_route_transition") != ["host", "contraction"]
                        or (prepared.get("arms") or {}).get("after", {}).get("declared_route")
                        != "contraction"
                        or (math.prod(row["source_batch_shape"]) > 1
                            and math.prod(extraction.get("probe_batch_shape") or []) <= 1)):
                    raise ValueError("reduced source does not preserve the selected rank, batching, or route")
                execution = dict(self.runtime_provider(candidate=candidate, experiment=experiment,
                    prepared=prepared, timeout_s=min(60.0, remaining())))
                representative["execution"] = execution
                if (execution.get("status") != "passed" or execution.get("correct") is not True
                        or any((execution.get("arms") or {}).get(arm, {}).get("correct") is not True
                               for arm in ("before", "after"))):
                    raise ValueError("lane-migration representative differs from the source oracle: "
                                     + str(execution.get("reason", "unknown")))
                record["bounded_programs_executed"] += 2
            verified = experiment.selected_changed_portfolio_context(
                candidate, selected["selection"])
            actual_binding = {"selection": verified["selection"],
                "previous": verified["previous"]["member_binding"],
                "current": verified["current"]["member_binding"]}
            if actual_binding != member_binding:
                raise ValueError("compiler, portfolio member, source, plan, or artifacts changed")
            record.update(status="passed_reduced_lane_migration_witness",
                          numerical_pass=True, bounded_runtime_executed=True,
                          representatives_qualified=len(record["representatives"]),
                          timing_scope="warm1/measured1 per reduced arm; cycles are not a model estimate")
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError,
                TimeoutError) as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = monotonic() - started
        receipt = work / "qualification.json"
        receipt.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        return {**record, "detail_path": str(receipt),
                "detail_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
