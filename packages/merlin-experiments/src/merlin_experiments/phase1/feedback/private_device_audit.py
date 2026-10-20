"""Exact built-artifact host-compute veto for private whole-model qualification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.compile.model_execution_inputs import file_sha256
from merlin_experiments.phase1.feedback.private_group_provenance import (
    join_routed_source_groups as _join_routed_source_groups,
)


def require_linked_symbols(elf: Path, symbols: set[str]) -> None:
    from merlin.llvmlower import toolchain

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

    from merlin.llvmlower.device_build import executable_kernel_entry
    from merlin.verify import host_compute_audit as HA
    from merlin.verify.llvm_symbols import defined_function

    entry_symbol = executable_kernel_entry(target)
    device_dir = build / "device"
    if device_dir.is_symlink() or not device_dir.is_dir():
        raise ValueError("linked build has no ordinary device-artifact directory")
    metrics = {**source.get("optional_group_metrics", {}), **source["eligible_group_metrics"]}
    source_groups = _join_routed_source_groups(assigned, source)
    preparations = source.get("conv_input_preparation", {})
    host_alignments, alignment_record = {}, None
    if preparations and host_artifacts is not None:
        from merlin.verify.host_call_alignment import inspect

        host_alignments, alignment_record = inspect(host_artifacts, assigned, preparations, diagnostics=diagnostics)
    records = []
    completed = {}
    for entry in assigned:
        symbol, index = entry["symbol"], entry["group"]
        if not isinstance(symbol, str) or not symbol or Path(symbol).name != symbol or symbol in {".", ".."}:
            raise ValueError("routed group has no safe built-artifact stem")
        source_group = source_groups[index]
        metric = metrics.get(source_group)
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
            records.append({**completed[key], "group": index, "source_group": source_group})
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
                "source_group": source_group,
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
                problems = group.get("activity_problems") or [group.get("why", "no proof detail")]
                reasons.append(str(group.get("symbol")) + ": " + "; ".join(map(str, problems[-8:])))
            from .private_program_checks import ProgramCheckFailure

            raise ProgramCheckFailure(
                f"linked group {index} host-compute audit is unknown, not clean; " + "; ".join(reasons)[:1600],
                check="host_compute",
            )
        from merlin.llvmlower import toolchain

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
                "source_group": source_group,
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
