"""Exact built-artifact host-compute veto for private whole-model qualification."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

from merlin.development.backend.qualification.provenance.groups import (
    join_routed_source_groups as _join_routed_source_groups,
)
from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import (
    _graph_totality,
    _placement,
    _source_complete,
)
from merlin.runtime.execution.model_inputs import file_sha256, strict_tree_sha256
from merlin.verification.host.compute_audit import HostComputeVeto

# ---- private_device_audit ----------------------------------------------------------------------


def require_linked_symbols(elf: Path, symbols: set[str]) -> None:
    from merlin.compiler.lower.model import toolchain

    command = [str(toolchain.nm()), "--defined-only", "--extern-only", str(elf)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    defined = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    missing = sorted(symbols - defined)
    if missing:
        raise ValueError(f"linked ELF omits {len(missing)} routed device symbol(s): {missing[:8]}")


def require_static_build_inputs(
    receipt: Mapping[str, Any], *, target: str, board: str, catalog: Path, dts: Path
) -> None:
    """Bind a linked image to the selected build-only board inputs."""
    inputs = receipt.get("inputs") or {}
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("target") != target
        or inputs.get("board") != board
        or inputs.get("board_catalog") != str(catalog)
        or inputs.get("dts") != str(dts)
        or inputs.get("board_catalog_sha256") != file_sha256(catalog)
        or inputs.get("dts_sha256") != file_sha256(dts)
        or inputs.get("run") != "none"
    ):
        raise ValueError("linked image used another board or implied model execution")


def audit_built_device_host_compute(
    build: Path,
    assigned: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    target: str,
    *,
    diagnostics: Path | None = None,
    host_artifacts=None,
    source_composition=None,
) -> list[dict[str, Any]]:
    """Audit the package's actual per-group build inputs, not a fresh diagnostic emission.

    The trusted whole-model builder saves each package stdout as ``.device.mlir``
    before translating and compiling it to the neighboring linked object.  We
    read those exact files and the independently sourced group output extents.
    Identical artifacts and output budgets are audited once per invocation, but
    every source call keeps its own identity. No mutable or cross-run cache exists.
    All defined functions are judged as accelerator work; an extra helper
    cannot self-label as a permitted host group.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, func, llvm
    from xdsl.parser import Parser

    from merlin.compiler.lower.device.build import executable_kernel_entry
    from merlin.verification.host import compute_audit as HA
    from merlin.verification.host.llvm_symbols import defined_function

    entry_symbol = executable_kernel_entry(target)
    device_dir = build / "device"
    if device_dir.is_symlink() or not device_dir.is_dir():
        raise ValueError("linked build has no ordinary device-artifact directory")
    metrics = {**source.get("optional_group_metrics", {}), **source["eligible_group_metrics"]}
    if source_composition is None:
        source_groups = _join_routed_source_groups(assigned, source)
    else:
        # This is the separately rederived original-source projection, never
        # an assertion from the emitted program. Keep all original members in
        # the report and audit the one physical object with its full raw output.
        source_groups, composition_metrics = _composed_source_metrics(assigned, source, source_composition)
        metrics = composition_metrics
    preparations = source.get("conv_input_preparation", {})
    host_alignments, alignment_record = {}, None
    if preparations and host_artifacts is not None:
        from merlin.verification.host.llvm_symbols import inspect

        host_alignments, alignment_record = inspect(host_artifacts, assigned, preparations, diagnostics=diagnostics)
    records = []
    completed = {}
    for entry in assigned:
        symbol, index = entry["symbol"], entry["group"]
        if not isinstance(symbol, str) or not symbol or Path(symbol).name != symbol or symbol in {".", ".."}:
            raise ValueError("routed group has no safe built-artifact stem")
        source_group = source_groups[index]
        metric = metrics.get(source_group if source_composition is None else index)
        if not isinstance(metric, Mapping):
            raise ValueError(f"routed group {index} has no source-derived output extent")
        artifact, llvm_ir, obj = (
            device_dir / f"{symbol}.device.mlir",
            device_dir / f"{symbol}.ll",
            device_dir / f"{symbol}.o",
        )
        for path in (artifact, llvm_ir, obj):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"linked group {index} lacks its exact built artifact: {path}")
        digests = tuple(file_sha256(path) for path in (artifact, llvm_ir, obj))
        key = (symbol, metric["elements"], metric["element_bytes"], *digests)
        if key in completed:
            records.append(
                {
                    **completed[key],
                    "group": index,
                    "source_group": source_group if source_composition is None else None,
                    **({"source_groups": list(source_group)} if source_composition is not None else {}),
                }
            )
            continue
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, llvm.LLVM, func.Func):
            context.load_dialect(dialect)
        module = Parser(context, artifact.read_text(encoding="utf-8")).parse_module()
        functions = HA._functions(module)  # noqa: PLC2701 -- shared independent host-code reader
        try:
            kernel = defined_function(module, entry_symbol)
        except ValueError as error:
            raise ValueError(f"linked group {index} has no body for its target ABI kernel") from error
        sites = [
            HA.GroupSite(
                group=index,
                placement=target,
                symbol=name,
                elements=metric["elements"],
                element_bytes=metric["element_bytes"],
            )
            for name in functions
        ]
        preparation = preparations.get(symbol)
        if preparation is not None and symbol in host_alignments:
            preparation = {**preparation, "argument_alignments": host_alignments[symbol]}
        # The ABI may alias the real body. Scope proven transport to that one
        # resolved function; every other helper retains the ordinary veto.
        report = HA.audit(
            module,
            sites,
            input_preparation={kernel.sym_name.data: preparation} if preparation is not None else None,
            target=target,
        )
        # Retain the instrument's complete reasons on the exact built artifact.
        # A short exception alone used to discard unsupported-CFG diagnostics.
        # Revalidation of a sealed build remains read-only. Only the builder's
        # explicitly provided mutable diagnostic directory receives a report.
        if diagnostics is not None and report.get("proven_clean") is not True:
            diagnostics.mkdir(parents=True, exist_ok=True)
            diagnostic = {
                "schema": "merlin.built_device.host_compute_audit.v1",
                "group": index,
                "source_group": source_group if source_composition is None else None,
                **({"source_groups": list(source_group)} if source_composition is not None else {}),
                "symbol": symbol,
                "artifacts": {"llvm_mlir": digests[0], "llvm_ir": digests[1], "object": digests[2]},
                "report": report,
            }
            raw = (json.dumps(diagnostic, sort_keys=True, indent=2) + "\n").encode()
            diagnostic_path = diagnostics / f"{symbol}.host-compute-audit-{hashlib.sha256(raw).hexdigest()}.json"
            try:
                with diagnostic_path.open("xb") as stream:
                    stream.write(raw)
            except FileExistsError:
                if diagnostic_path.is_symlink() or diagnostic_path.read_bytes() != raw:
                    raise ValueError("built device audit diagnostic changed")
        if report.get("proven_clean") is not True or report.get("accelerator_groups_clean") != len(sites):
            if tuple(file_sha256(path) for path in (artifact, llvm_ir, obj)) != digests:
                raise ValueError("built device artifacts changed during host-compute audit")
            if alignment_record is not None and any(
                file_sha256(Path(host_artifacts[role]["path"])) != digest
                for role, digest in alignment_record["host_artifacts"].items()
            ):
                raise ValueError("compiled host alignment inputs changed during device audit")
        HA.require_clean(report)
        if report.get("proven_clean") is not True or report.get("accelerator_groups_clean") != len(sites):
            reasons = []
            for group in report.get("groups", []):
                if group.get("verdict") == "clean":
                    continue
                problems = group.get("activity_problems") or [group.get("why") or "proof detail unavailable"]
                reasons.append(str(group.get("symbol")) + ": " + "; ".join(map(str, problems[-8:])))

            raise ProgramCheckFailure(
                f"linked group {index} host-compute audit is unknown, not clean; " + "; ".join(reasons)[:1600],
                check="host_compute",
            )
        from merlin.compiler.lower.model import toolchain

        undefined = subprocess.run(
            [str(toolchain.nm()), "--undefined-only", "--extern-only", str(obj)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        if undefined.stdout.strip():
            raise ValueError(f"linked group {index} calls an unaudited external host helper")
        records.append(
            {
                "group": index,
                "source_group": source_group if source_composition is None else None,
                **({"source_groups": list(source_group)} if source_composition is not None else {}),
                "symbol": symbol,
                "defined_functions_audited": len(sites),
                "artifact_sha256": digests[0],
                "llvm_sha256": digests[1],
                "object_sha256": digests[2],
                "verdict": "clean_static_host_compute_audit",
                "audit": {
                    "scope": (
                        "all defined LLVM functions of the exact built device artifact; "
                        "command-scale budget with independently proven byte transport"
                    ),
                    "budget": report["budget"],
                    "groups": [
                        {
                            key: group.get(key)
                            for key in (
                                "symbol",
                                "verdict",
                                "elements",
                                "host_arithmetic",
                                "host_value_arithmetic",
                                "host_payload_bytes",
                                "host_total_arithmetic",
                                "host_raw_payload_bytes",
                                "input_preparation",
                                "arithmetic_per_element",
                                "value_arithmetic_per_element",
                                "payload_ratio",
                            )
                        }
                        for group in report["groups"]
                    ],
                },
            }
        )
        if tuple(file_sha256(path) for path in (artifact, llvm_ir, obj)) != digests:
            raise ValueError("built device artifacts changed during host-compute audit")
        if alignment_record is not None:
            records[-1]["host_argument_alignment"] = (
                {
                    "alignment_bytes": host_alignments[symbol],
                    "observation": alignment_record,
                }
                if symbol in host_alignments
                else None
            )
        completed[key] = records[-1]
    if alignment_record is not None and any(
        file_sha256(Path(host_artifacts[role]["path"])) != digest
        for role, digest in alignment_record["host_artifacts"].items()
    ):
        raise ValueError("compiled host alignment inputs changed during device audit")
    return records


def _composed_source_metrics(assigned, source, composition):
    """Budget one actual object from every independently rederived raw product."""
    if isinstance(composition, Mapping) and composition.get("schema") == "merlin.original_head_source_composition.v1":
        from merlin.development.backend.qualification.provenance.composition import source_metrics

        return source_metrics(assigned, source, composition)
    if isinstance(composition, Mapping) and composition.get("schema") == "merlin.original_sibling_source_components.v1":
        from merlin.development.backend.qualification.provenance.composition import source_subset

        groups, metrics, covered = {}, {}, set()
        for component in composition["components"]:
            members = set(component["covered_source_groups"])
            if covered.intersection(members):
                raise ValueError("component audit duplicates original work")
            covered.update(members)
            subsource = source_subset(
                source, set(source["eligible_groups"] + source.get("optional_groups", [])) - members
            )
            rows = [r for r in assigned if r["symbol"] == component["physical_symbol"]]
            subgroups, submetrics = _composed_source_metrics(rows, subsource, component)
            if set(groups).intersection(subgroups):
                raise ValueError("component audit duplicates physical work")
            groups.update(subgroups)
            metrics.update(submetrics)
        ordinary = [r for r in assigned if r["symbol"] not in {c["physical_symbol"] for c in composition["components"]}]
        joined = _join_routed_source_groups(ordinary, source_subset(source, covered))
        if joined != composition["ordinary_joined_groups"] or set(groups).intersection(joined):
            raise ValueError("unmatched original source join changed")
        original_metrics = {**source.get("optional_group_metrics", {}), **source["eligible_group_metrics"]}
        for physical, original in joined.items():
            groups[physical] = (original,)
            metrics[physical] = original_metrics[original]
        if not set(source["eligible_groups"]) <= covered | set(joined.values()):
            raise ValueError("component audit omits original required work")
        return groups, metrics
    if (
        not isinstance(composition, Mapping)
        or composition.get("schema") != "merlin.original_sibling_source_composition.v1"
        or composition.get("original_source_denominator_unchanged") is not True
        or len(assigned) != 1
        or assigned[0].get("group") != composition.get("physical_group")
        or assigned[0].get("symbol") != composition.get("physical_symbol")
    ):
        raise ValueError("device audit needs its independently rederived original source composition")
    members = composition.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("source composition has no complete original member roster")
    indices, elements = [], 0
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("source composition member is absent")
        index = member.get("source_group")
        original = source.get("source_group_contractions", {}).get(index)
        if (
            not isinstance(original, Mapping)
            or original.get("dtypes") != ["i8", "i8", "i32"]
            or len(original.get("parallel", [])) != 2
            or len(original.get("reduction", [])) != 1
        ):
            raise ValueError("composed device audit lacks original raw product geometry")
        m, n = original["parallel"]
        k = original["reduction"][0]
        if (
            any(type(v) is not int or v <= 0 for v in (m, n, k))
            or member.get("geometry_m_k_n") != [m, k, n]
            or member.get("raw_output_elements") != m * n
            or member.get("raw_output_element_bytes") != 4
            or member.get("source_useful_macs") != m * k * n
        ):
            raise ValueError("composed device budget differs from original product")
        indices.append(index)
        elements += m * n
    if (
        len(set(indices)) != len(indices)
        or not set(source["eligible_groups"]) <= set(indices)
        or composition.get("required_source_groups") != source["eligible_groups"]
        or composition.get("covered_source_groups") != indices
        or composition.get("raw_output_elements") != elements
        or composition.get("raw_output_element_bytes") != 4
    ):
        raise ValueError("composed device budget omits or duplicates original source work")
    physical = composition["physical_group"]
    return {physical: tuple(indices)}, {physical: {"elements": elements, "element_bytes": 4}}


# ---- model_routes ------------------------------------------------------------------------------
# Report exact whole-program host/device routing evidence without inventing a compiler.
#
# The Phase 0 ledger screens *candidate* placements. A host LLVM file and an OOT
# command buffer are separate products; neither proves the other was stitched into
# the model or that the result is numerically correct. Keep that boundary explicit
# until a compiler supplies per-operation lowering and typed crossing receipts.

SCHEMA = "merlin.phase1.model_routes.v1"


def _ledger_observation(application: dict | None, model_sha256: str) -> tuple[dict, list[str]]:
    blockers: list[str] = []
    summary = {"status": "absent", "n_operations": None, "candidate_lanes": {}, "unresolved_obligations": None}
    if not isinstance(application, dict):
        return summary, ["selected per-program operation ledger is absent"]
    if application.get("capture_sha256") != model_sha256:
        blockers.append("operation ledger belongs to different captured MLIR bytes")
    receipt = application.get("capture_receipt") or {}
    if receipt.get("status") != "verified_materialized" or receipt.get("source_closure_verified") is not True:
        blockers.append("capture source closure is not verified")
    completeness = application.get("completeness") or {}
    try:
        _, graph_reasons = _graph_totality(application, completeness)
    except (KeyError, TypeError, ValueError) as exc:
        graph_reasons = [f"malformed exact operation/SSA graph: {type(exc).__name__}: {exc}"]
    blockers.extend(graph_reasons)
    if not _source_complete(completeness.get("source_trace") or {}):
        blockers.append("frontend-to-normalized-MLIR source correspondence is incomplete")
    obligations = completeness.get("operation_obligations")
    if not isinstance(obligations, list):
        obligations = []
        blockers.append("per-operation placement obligations are absent")
    lanes = Counter()
    selected_lanes = {}
    unresolved = 0
    support_pending = 0
    for row in obligations:
        if not isinstance(row, dict):
            unresolved += 1
            continue
        if row.get("role") == "support_lowering":
            # The Phase 0 source observation is not an executable/compiler
            # lowering receipt. Support nodes have no independent lane or
            # arithmetic contract, and cannot be transfer endpoints here.
            support_pending += 1
            continue
        lane = _placement(row)
        numerical = ((row.get("precision") or {}).get("numerical_contracts") or {}).get(lane) or {}
        if (
            lane is None
            or row.get("status") != "resolved"
            or (row.get("precision") or {}).get("status") != "resolved"
            or numerical.get("status") != "resolved"
        ):
            unresolved += 1
        else:
            lanes[lane] += 1
            for operation_id in row.get("operation_ids") or []:
                selected_lanes[operation_id] = lane
    if unresolved:
        blockers.append(f"{unresolved} operation placement/precision/numerical obligations remain unresolved")
    if support_pending:
        blockers.append(f"{support_pending} typed support-lowering/shape obligations lack compiler verification")
    transfers = completeness.get("transfer_obligations")
    if not isinstance(transfers, list):
        blockers.append("typed host/device transfer obligations are absent")
        transfers = []
    # These are compute SSA edges, not necessarily crossings. Only reviewed,
    # resolved endpoint placements can turn an edge into a required transfer.
    transfer_counts = Counter()
    for edge in transfers:
        if not isinstance(edge, dict):
            transfer_counts["pending_placement"] += 1
            continue
        producer = selected_lanes.get(edge.get("producer_operation_id"))
        consumer = selected_lanes.get(edge.get("consumer_operation_id"))
        if producer is None or consumer is None:
            transfer_counts["pending_placement"] += 1
        elif producer == consumer:
            transfer_counts["same_lane"] += 1
        else:
            transfer_counts["required_crossing"] += 1
    if transfer_counts["required_crossing"]:
        blockers.append(
            f"{transfer_counts['required_crossing']} selected host/device crossings "
            "lack typed transfer lowering witnesses"
        )
    transfer_status = (
        "pending_placement"
        if transfer_counts["pending_placement"]
        else "requires_lowering"
        if transfer_counts["required_crossing"]
        else "same_lane_or_none"
    )
    support_edges = sum(
        edge.get("accounting") == "support_dependency"
        for edge in (completeness.get("graph_accounting") or {}).get("edges") or []
        if isinstance(edge, dict)
    )
    if support_edges:
        blockers.append(f"{support_edges} support-mediated SSA dependencies lack compiler-owned route evidence")
    summary = {
        "status": "screened_not_compiled" if not blockers else "incomplete",
        "n_operations": application.get("n_mlir_operations"),
        "candidate_lanes": dict(sorted(lanes.items())),
        "unresolved_obligations": unresolved,
        "support_lowering": {"pending": support_pending, "support_dependencies": support_edges},
        "conditional_ssa_edges": {
            "status": transfer_status,
            "count": len(transfers),
            "pending_placement": transfer_counts["pending_placement"],
            "same_lane": transfer_counts["same_lane"],
            "required_crossing": transfer_counts["required_crossing"],
        },
        "qualification": (
            "declaration screening only; support lowering and compiler-owned routes require separate evidence"
        ),
    }
    return summary, blockers


def summarize_model_routes(
    workflow: dict, accounting: dict | None, compiler_observations: list[dict], native_lowerings: list[dict]
) -> list[dict]:
    """Bind model-route diagnostics to each declared program's exact captured bytes.

    This is intentionally not an executable plan. An OOT package that declines a
    whole captured model cannot be turned into accelerator work by a host LLVM
    file, and a nonempty buffer alone cannot prove that every model operation or
    host/device edge was lowered.
    """
    applications = (accounting or {}).get("applications") or {}
    compiler = {(row.get("program"), row.get("entrypoint")): row for row in compiler_observations}
    native = {row.get("program"): row for row in native_lowerings}
    result = []
    for program in workflow["programs"]:
        name, model_sha256 = program["name"], program["model_sha256"]
        ledger, blockers = _ledger_observation(applications.get(name), model_sha256)
        command = (compiler.get((name, "emit_command_buffer")) or {}).get("command_buffer") or {}
        command_status = command.get("status", "not_emitted")
        route_status = {
            "emitted": "unverified_candidate",
            "declined": "observed_decline",
            "invalid": "invalid_observation",
        }.get(command_status, "unresolved")
        if command_status != "emitted":
            blockers.append(f"selected OOT compiler command buffer is {command_status}")
        else:
            blockers.append("emitted commands have no whole-model operation/transfer correspondence witness")
        host = native.get(name) or {}
        host_status = host.get("status", "not_requested")
        if host_status != "llvm_ir_emitted":
            blockers.append(f"complete native-host LLVM lowering is {host_status}")
        blockers.append("whole-program numerical execution and target-visible seam are not verified")
        result.append(
            {
                "schema": SCHEMA,
                "program": name,
                "capture_sha256": model_sha256,
                "status": route_status,
                "whole_model_compiler_verified": False,
                "ledger": ledger,
                "host": {
                    "status": host_status,
                    "llvm_ir_sha256": host.get("llvm_ir_sha256"),
                    "qualification": "native LLVM IR only; not selected target-host execution",
                },
                "accelerator": {
                    "status": command_status,
                    "commands_count": command.get("commands_count"),
                    "qualification": "OOT payload observation only; not complete-model offload",
                },
                "blockers": sorted(set(blockers)),
            }
        )
    return result


# ---- private_execution_checks ------------------------------------------------------------------
# Revalidate the complete declared numerical checks of a private execution record.
#
# These are structural checks over host-owned results, not a second numerical
# oracle or an authority for candidate-supplied receipts.


def _count(value):
    return type(value) is int and value >= 0


def _rows(value, indices):
    return (
        isinstance(value, list)
        and len(value) == len(indices)
        and all(
            isinstance(row, Mapping)
            and row.get("index") == index
            and type(row.get("index")) is int
            and row.get("passed") is True
            for row, index in zip(value, indices, strict=True)
        )
    )


def complete_checks(checks, entry):
    """Require the descriptor's checks and complete ordered result coverage."""
    required = {"completion", "integer_reference", "host_execution", "per_group_exactness"}
    sanity = entry.get("fp32_sanity")
    if sanity is not None:
        required.add("fp32_sanity")
    if not isinstance(checks, Mapping) or set(checks) != required:
        return False
    if any(
        not isinstance(checks[name], Mapping) or checks[name].get("passed") is not True
        for name in required - {"per_group_exactness"}
    ):
        return False
    scope = checks["per_group_exactness"]
    if scope.get("status") != "not_observable" or "passed" in scope:
        return False
    completion = checks["completion"]
    results, elements = completion.get("results"), completion.get("elements")
    if not _count(results) or results == 0 or not _count(elements):
        return False
    integer, host = checks["integer_reference"], checks["host_execution"]
    integer_rows = integer.get("results", [])
    if integer.get("status") == "compared":
        if (
            integer.get("reference") not in entry["integer_reference"]
            or not isinstance(integer_rows, list)
            or not 0 < len(integer_rows) <= results
            or not _rows(integer_rows, range(len(integer_rows)))
        ):
            return False
    elif integer.get("status") == "not_available":
        if (
            entry["require_integer_reference"] is not False
            or integer.get("looked_for") != entry["integer_reference"]
            or integer_rows != []
        ):
            return False
    else:
        return False
    covered, counts = set(), {}
    for row in integer_rows:
        if row.get("status") == "unconfirmed_reference":
            if (
                row.get("reference_confirmed_by_host_execution") is not False
                or not _count(row.get("host_mismatched_elements"))
                or row["host_mismatched_elements"] == 0
            ):
                return False
            continue
        if (
            row.get("status") is not None
            or type(row.get("mismatched_elements")) is not int
            or row["mismatched_elements"] != 0
            or not _count(row.get("of"))
        ):
            return False
        covered.add(row["index"])
        counts[row["index"]] = row["of"]
    judged = [index for index in range(results) if index not in covered]
    host_rows, tolerance = host.get("results"), entry.get("host_execution")
    if not _rows(host_rows, judged) or (judged and tolerance is None):
        return False
    for row in host_rows:
        if (
            not _count(row.get("of"))
            or type(row.get("within")) is not int
            or row["within"] != row["of"]
            or row.get("atol") != tolerance["atol"]
            or row.get("rtol") != tolerance["rtol"]
        ):
            return False
        counts[row["index"]] = row["of"]
    if len(counts) != results or sum(counts.values()) != elements:
        return False
    if sanity is None:
        return True
    check = checks["fp32_sanity"]
    digest = check.get("sha256")
    if (
        check.get("reference") != sanity["reference"]
        or type(check.get("receipt_bound")) is not bool
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or check.get("cosine_margin") != sanity["cosine_margin"]
        or not _rows(check.get("results"), range(results))
    ):
        return False
    from merlin.development.backend.qualification.full_models.execution_engine import COSINE_DISTANCE_FLOOR

    for row in check["results"]:
        if any(
            type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
            for key in ("cosine", "floor", "quantized_program_cosine")
        ):
            return False
        expected_floor = 1.0 - sanity["cosine_margin"] * max(
            1.0 - row["quantized_program_cosine"],
            COSINE_DISTANCE_FLOOR,
        )
        if row["floor"] != expected_floor or row["cosine"] < row["floor"]:
            return False
    top1 = check.get("top1")
    if not isinstance(top1, Mapping):
        return False
    if not sanity["top1"]:
        return top1 == {"status": "not_declared"}
    return (
        top1.get("passed") is True
        and all(_count(top1.get(key)) for key in ("rows", "applicable_rows", "agreeing_rows"))
        and top1["agreeing_rows"] == top1["applicable_rows"] <= top1["rows"]
    )


# ---- private_prebuilt_receipt ------------------------------------------------------------------
# Fail-closed reading of unbound prebuilt evidence for diagnostic model checks.


def _require_selected_rtl_facts(inputs: Mapping[str, Any], selected_rtl_facts: Mapping[str, str] | None) -> None:
    """Join an old path-only observation to current selected bytes, without producer credit."""
    observed = inputs.get("rtl_facts")
    identity = inputs.get("rtl_facts_identity")
    if observed is None:
        if identity is not None:
            raise ValueError("prebuilt diagnostic receipt has RTL facts identity without a selected path")
        return
    if not isinstance(observed, str) or selected_rtl_facts is None:
        raise ValueError("prebuilt diagnostic receipt has unselected RTL facts")
    selected_path = selected_rtl_facts.get("path")
    selected_sha256 = selected_rtl_facts.get("sha256")
    if not isinstance(selected_path, str) or not isinstance(selected_sha256, str):
        raise ValueError("prebuilt diagnostic receipt has no byte-pinned selected RTL facts")
    path = Path(observed)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or observed != selected_path
        or file_sha256(path) != selected_sha256
        or (identity is not None and identity != dict(selected_rtl_facts))
    ):
        raise ValueError("prebuilt diagnostic receipt differs from selected RTL facts")


def load_diagnostic_receipt(
    path: Path,
    *,
    capture: Path,
    host_package: Path,
    candidate: Path,
    host_package_tree: Mapping[str, Any],
    candidate_tree: Mapping[str, Any],
    arena_mb: int,
    target: str,
    device_selected: bool,
    selected_rtl_facts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read old build evidence without asserting its unrecorded producer identity."""
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or path.name != "baremetal_model.json"
        or not path.is_file()
    ):
        raise ValueError("prebuilt diagnostic receipt is absent or indirect")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != "merlin.baremetal-saved-model.v1":
        raise ValueError("prebuilt diagnostic receipt has another schema")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no immutable build inputs")
    device = inputs.get("device")
    if device_selected and not isinstance(device, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no selected device")
    _require_selected_rtl_facts(inputs, selected_rtl_facts)
    if (
        inputs.get("capture") != str(capture.resolve())
        or inputs.get("package") != str(host_package.resolve())
        or inputs.get("capture_tree") != strict_tree_sha256(capture)
        or host_package_tree != strict_tree_sha256(host_package)
        or inputs.get("package_tree") != host_package_tree
        or inputs.get("arena_mb") != arena_mb
        or inputs.get("reference_file") is not None
        or (device_selected and device.get("package") != str(candidate.resolve()))
        or (device_selected and device.get("name") != target)
        or (device_selected and candidate_tree != strict_tree_sha256(candidate))
        or (device_selected and device.get("package_tree") != candidate_tree)
        or (not device_selected and device is not None)
    ):
        raise ValueError("prebuilt diagnostic receipt differs from current immutable build inputs")
    output = receipt.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no build output")
    sidecar_record = output.get("device_sidecar")
    if device_selected and not isinstance(sidecar_record, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no selected device sidecar")
    if not device_selected and sidecar_record is not None:
        raise ValueError("prebuilt host-only diagnostic unexpectedly contains device sidecar")
    elf = Path(str(output.get("elf") or ""))
    sidecar = Path(str((sidecar_record or {}).get("path") or "")) if device_selected else None
    build = path.parent / "build"
    if elf.is_symlink() or not elf.is_file() or not elf.resolve().is_relative_to(build.resolve()):
        raise ValueError("prebuilt diagnostic receipt names an absent or external build artifact")
    if sidecar is not None and (
        sidecar.is_symlink() or not sidecar.is_file() or not sidecar.resolve().is_relative_to(build.resolve())
    ):
        raise ValueError("prebuilt diagnostic receipt names an absent or external device sidecar")
    return receipt


# ---- private_program_checks --------------------------------------------------------------------
# Retain explicit program-local static refusals without skipping sibling checks.
#
# Only typed semantic refusals may continue. A changed input, route, recipe or
# product, an unexpected exception and an interrupted process keep their original
# failure path. No failed program gains linked, numerical or full-model credit.

VERIFIED = "capture_lower_codegen_link_verified"

FAILED = "program_static_checks_failed"


class ProgramCheckFailure(ValueError):
    """One explicitly identified semantic checker refused this program."""

    def __init__(self, message, *, check):
        super().__init__(message)
        self.check = check


def verify(checker, receipt, *, revalidate, **arguments):
    """Run the ordinary checker; continue only after independent identity rechecks."""
    started = monotonic()
    try:
        result = checker(receipt, **arguments)
    except (ProgramCheckFailure, HostComputeVeto) as error:
        # Identity/route exceptions are intentionally not caught. A semantic
        # refusal also cannot mask simultaneous shared-input/product mutation.
        revalidate()
        from merlin.development.backend.qualification.full_models.private_compilation_inputs import (
            verify as verify_compilation,
        )

        output = receipt.get("output") or {}
        elf = Path(str(output.get("elf") or ""))
        if elf.is_symlink() or not elf.is_file() or file_sha256(elf) != output.get("elf_sha256"):
            raise ValueError("linked program changed during semantic verification") from error
        verify_compilation(
            output,
            elf,
            required=(arguments["require_compilation_recipe"] if "require_compilation_recipe" in arguments else True),
        )
        return {
            "program": arguments["program"],
            "status": FAILED,
            "linked_device_groups": 0,
            "check_stage": "linked_program_checks",
            "check": error.check if isinstance(error, ProgramCheckFailure) else "host_compute",
            "reason": f"{type(error).__name__}: {error}",
            "observed_elf": {"path": str(elf), "sha256": output["elf_sha256"]},
            "check_elapsed_s": monotonic() - started,
            "scope": "program-local static refusal; no linked or numerical credit",
        }
    return {**result, "check_stage": "linked_program_checks", "check_elapsed_s": monotonic() - started}


def status(programs):
    return VERIFIED if programs and all(program.get("status") == VERIFIED for program in programs) else FAILED


def require_complete(programs):
    if status(programs) != VERIFIED:
        failed = [program.get("program") for program in programs if program.get("status") != VERIFIED]
        raise ProgramCheckFailure(f"independent program checks failed: {failed}", check="program_roster")


def progress_stage(program):
    return "program_receipt_verified" if program.get("status") == VERIFIED else "program_checks_failed"
