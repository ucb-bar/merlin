"""What a backend package puts on the accelerator, measured from what it emits.

The shape-coverage probe asks whether a package lowers one contraction at several extents. This asks
the question a passing capsule suite cannot: across the programs in a corpus, how much of the work
reached a unit, how much was placed on the host, and how much was declined. It runs ONLY the emit
half of the backend contract, so it needs no golden, no oracle and no simulator, and it reads nothing
but the package's own command buffer. That makes it safe to hand to the agent that wrote the package,
and cheap enough to run every round.

The report is per program and carries a denominator: a program that emits no command and declares no
decline is named, because at the numeric tier it arrives as an output of zeros and looks exactly like
arithmetic that ran and was wrong.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from merlin.perf import isa_utilization
from merlin.perf.offload import offload_report

from .capsule_common import discover_capsules, make_run_paths, run_entrypoints
from .oot_runner import BackendDeclined, CertFailure

SCHEMA = "offload_census_v1"


def _placed(buffer: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    params = buffer.get("params")
    rows = params.get("lane_placement") if isinstance(params, Mapping) else None
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def program_row(name: str, buffer: Mapping[str, Any]) -> dict[str, Any]:
    """One emitted program's offload accounting, from its own buffer."""
    report = offload_report(buffer)
    commands = buffer.get("commands") or []
    placed = _placed(buffer)
    declined = buffer.get("declined")
    row: dict[str, Any] = {
        "program": name,
        "commands": len(commands),
        "routed_macs": report.routed_macs,
        "routed_is_lower_bound": report.routed_is_lower_bound,
        "unit_regions": report.unit_regions,
        "host_regions": report.host_regions,
        "contraction_offload_fraction": report.contraction_offload_fraction,
        "placement_declared": bool(placed),
        "declined": bool(declined),
    }
    if declined:
        row["outcome"] = "declined"
        row["reason"] = str(declined.get("reason") if isinstance(declined, Mapping) else declined)
    elif not commands and not placed:
        row["outcome"] = "silent"  # nothing emitted, nothing placed, nothing said
    elif not commands:
        row["outcome"] = "host_only"  # a declared host-lane program
    elif report.host_regions and not report.unit_regions:
        row["outcome"] = "host_only"
    else:
        row["outcome"] = "offloaded"
    return row


def _payload_elements(buffer: Mapping[str, Any]) -> tuple[int, int]:
    """``(elements, element bytes)`` of the program's payload: what element-scale host work scales with.

    The tensors the commands write, where the buffer declares them. A buffer often declares only
    what it reads (the results live in handles the commands name), and then the largest tensor it
    reads stands in: a host loop over the payload is at least that long.
    """
    from merlin.common.mlir_query import _DTYPE_BYTES

    tensors = buffer.get("tensors") if isinstance(buffer.get("tensors"), Mapping) else {}
    written = {
        str(cmd.get("operands", {}).get("dst"))
        for cmd in buffer.get("commands") or ()
        if isinstance(cmd, Mapping) and isinstance(cmd.get("operands"), Mapping)
    }
    abi = buffer.get("kernel_abi") if isinstance(buffer.get("kernel_abi"), Mapping) else {}
    outputs = ({str(name) for name in abi.get("outputs") or ()} | written) & set(tensors)

    def size(name: str) -> int:
        total = 1
        for extent in tensors[name].get("shape") or ():
            total *= int(extent)
        return total

    chosen = sorted(outputs) or sorted(tensors, key=size)[-1:]
    elements = sum(size(name) for name in chosen) if outputs else max((size(name) for name in chosen), default=0)
    width = max((_DTYPE_BYTES.get(str(tensors[name].get("dtype")), 1) for name in chosen), default=1)
    return elements, width


def host_compute_row(name: str, buffer: Mapping[str, Any], artifact_text: str, outcome: str) -> dict[str, Any]:
    """The host-compute audit of one emitted program (:mod:`merlin.verify.host_compute_audit`).

    A program that reached a unit is judged: its host code may issue commands, and may not do the
    unit's arithmetic. One that declared itself host-only is reported and not judged. The audit
    reads host LLVM; an artifact that is a device's own program (a target whose host and device
    meet at addresses, not at a call) is not host code, and says so rather than passing.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, func, llvm
    from xdsl.parser import Parser

    from merlin.verify import host_compute_audit as HA

    try:
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, llvm.LLVM, func.Func):
            context.load_dialect(dialect)
        module = Parser(context, artifact_text).parse_module()
    except Exception as exc:  # noqa: BLE001 -- not LLVM-dialect MLIR: nothing here can count it
        return {
            "program": name,
            "verdict": HA.UNKNOWN,
            "why": f"the lowered artifact is not host LLVM this reader parses ({type(exc).__name__})",
        }
    functions = HA._functions(module)
    if not functions:
        return {"program": name, "verdict": HA.UNKNOWN, "why": "the lowered artifact defines no function with a body"}
    elements, width = _payload_elements(buffer)
    placement = HA.HOST if outcome in ("host_only", "declined", "silent") else "accelerator"
    rows = [
        HA.audit_group(
            HA.GroupSite(group=0, placement=placement, symbol=symbol, elements=elements, element_bytes=width), function
        )
        for symbol, function in functions.items()
    ]
    # The program's verdict is its worst function's: one host loop at element scale is enough.
    order = (HA.HOST_COMPUTE, HA.UNKNOWN, HA.NOT_EMITTED, HA.DECLARED_HOST, HA.CLEAN)
    worst = min(rows, key=lambda row: order.index(row["verdict"]))
    return {
        "program": name,
        **{
            key: worst.get(key)
            for key in (
                "verdict",
                "why",
                "symbol",
                "elements",
                "arithmetic_per_element",
                "value_arithmetic_per_element",
                "payload_ratio",
                "dominant_block",
            )
        },
    }


def instruction_use(artifacts: Iterable[str], *, target: str) -> dict[str, Any]:
    """Which of the target's DECLARED instructions the emitted programs ever use.

    The declared set is the target's own derived decode table. A target with no such table (a
    different endpoint, or facts not derived) reports why it cannot be measured; it does not report
    that everything was used.
    """
    from merlin.kernels.decode.rocc import funct_table_for

    table = funct_table_for(target)
    names, opcode = table.get("names"), table.get("custom_opcode")
    if not isinstance(names, Mapping) or not names or opcode is None:
        return {"status": "unavailable", "reason": "the target's derived facts carry no instruction decode table"}
    try:
        declared = {int(str(code), 0): str(name) for code, name in names.items()}
        opcode_value = int(str(opcode), 0)
    except ValueError:
        return {"status": "unavailable", "reason": "the derived decode table's codes are not integers"}
    report = isa_utilization.capability_utilization(
        "\n".join(artifacts), declared_functs=declared, custom_opcode=opcode_value
    )
    return {"status": "measured", **report}


def census(
    package: str | Path,
    *,
    target: str,
    capsules: Iterable[Mapping[str, Any]],
    contract: str | Path | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Emit every capsule that carries interface MLIR through ``package`` and account for the result."""
    rows: list[dict[str, Any]] = []
    artifacts: list[str] = []
    audits: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="offcensus_") as scratch:
        for capsule in capsules:
            if not capsule.get("interface_mlir"):
                continue
            name = str(capsule.get("name") or Path(str(capsule.get("__dir__", ""))).name)
            paths = make_run_paths(
                Path(scratch) / "runs",
                name,
                suite=f"{target}-offload-census",
                target=target,
                dtype=str(capsule.get("dtype") or "unknown"),
                benchmark=str(capsule.get("kind") or "capsule"),
            )
            try:
                _pkg, buffer, _artifact = run_entrypoints(
                    None,
                    package,
                    dict(capsule),
                    paths,
                    contract=contract,
                    timeout=timeout,
                    fourth_output_name="lowered.llvm.mlir",
                )
            except BackendDeclined as declined:
                rows.append(
                    {"program": name, "outcome": "declined", "reason": declined.reason, "commands": 0, "routed_macs": 0}
                )
                continue
            except CertFailure as failure:
                rows.append({"program": name, "outcome": "error", "reason": f"{failure.plane}: {failure.detail[:200]}"})
                continue
            except Exception as exc:  # noqa: BLE001 -- a harness fault is not an offload verdict
                rows.append({"program": name, "outcome": "error", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
                continue
            rows.append(program_row(name, buffer))
            artifacts.append(_artifact or "")
            audits.append(host_compute_row(name, buffer, _artifact or "", rows[-1]["outcome"]))

    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {
        "schema": SCHEMA,
        "target": target,
        "programs": rows,
        "outcomes": outcomes,
        "routed_macs": sum(int(row.get("routed_macs") or 0) for row in rows),
        "silent": [row["program"] for row in rows if row["outcome"] == "silent"],
        "placement_undeclared": [
            row["program"] for row in rows if row["outcome"] == "offloaded" and not row.get("placement_declared")
        ],
        "all_accounted": not any(row["outcome"] in ("silent", "error") for row in rows),
        # An instruction the hardware offers and no emitted program reaches for is invisible in every
        # per-program number: each buffer is well formed and each gate passes.
        "instruction_use": instruction_use(artifacts, target=target),
        # A program can reach the unit and still do the unit's arithmetic in a host loop. Numerics
        # and the command trace both pass on that program; only counting the host code shows it.
        "host_compute": audits,
        "host_compute_vetoed": [
            row["program"] for row in audits if row["verdict"] == "host_compute_in_accelerator_group"
        ],
    }


def census_of_corpus(
    package: str | Path,
    *,
    target: str,
    capsule_roots: Iterable[str | Path],
    contract: str | Path | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    return census(
        package,
        target=target,
        contract=contract,
        timeout=timeout,
        capsules=discover_capsules(list(capsule_roots), contract=contract),
    )


#: Set to ``0`` to report the host-compute audit beside a grade without letting it fail a capsule.
VETO_ENV = "MERLIN_HOST_COMPUTE_VETO"
VETO_CATEGORY = "host_compute_in_accelerator_group"


def apply_host_compute_veto(results: list[dict[str, Any]], suite_runs: Path) -> list[dict[str, Any]]:
    """Fail every PASSING capsule whose program does the unit's arithmetic on the host.

    A program can pass numerics and the instruction-trace check while a host loop loads, computes
    and stores every element of work it claims to put on a unit: the trace shows the unit was
    touched, the numbers are right, and the time is spent on the host. That was reported to the
    agent and left out of the verdict, so a backend could be certified on it.

    Read from what each graded run already left on disk (its command buffer and its lowered host
    artifact), so nothing is re-emitted. The verdict is WRITTEN BACK to that run's
    ``capsule_result.json``: the row an agent reads is rebuilt from that file, so a veto held only
    in memory reached it as a bare fail with no plane, no category and no sentence (measured on
    the first campaign it ran in: 22 capsules failed that way). Only a program that REACHED a unit is judged: one that
    declared itself host-only or declined says so and is reported, not penalised, and a whole-model
    capsule legitimately owns host regions. A program whose artifacts cannot be read is left as the
    grade found it; this can fail a pass, never manufacture one. Returns the rows it vetoed.
    """
    import json
    import os

    from merlin.verify import host_compute_audit as HA

    vetoed: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "pass" or result.get("kind") == "model":
            continue
        generated = Path(suite_runs) / str(result.get("capsule")) / "generated"
        buffer_path, artifact_path = generated / "command_buffer.json", generated / "lowered.llvm.mlir"
        if not (buffer_path.is_file() and artifact_path.is_file()):
            continue
        try:
            buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
            row = program_row(str(result.get("capsule")), buffer)
            audit = host_compute_row(
                str(result.get("capsule")), buffer, artifact_path.read_text(encoding="utf-8"), row["outcome"]
            )
        except Exception:  # noqa: BLE001 -- an unreadable artifact is not evidence of host compute
            continue
        result["host_compute"] = {key: audit.get(key) for key in ("verdict", "why", "symbol", "dominant_block")}
        if audit.get("verdict") != HA.HOST_COMPUTE:
            continue
        vetoed.append(audit)
        if os.environ.get(VETO_ENV, "1") == "0":
            result["host_compute"]["veto_suppressed_by"] = VETO_ENV
            continue
        result["status"] = "fail"
        result["failure"] = {
            "plane": "host_compute",
            "category": VETO_CATEGORY,
            "detail": (
                f"the program passes numerics and spends its work on the host: {audit.get('why')} "
                f"(function {audit.get('symbol')!r}). Work that scales with the payload belongs to "
                f"the unit: issue the commands that make the unit gather, accumulate or scale, in "
                f"place of a host loop over the elements. The host may compute addresses, issue "
                f"commands and branch. A program that DECLARES itself host-only is reported, not failed."
            ),
        }
        _record_veto(Path(suite_runs) / str(result.get("capsule")) / "capsule_result.json", result)
    return vetoed


def _record_veto(path: Path, result: Mapping[str, Any]) -> None:
    """Put a veto where every reader of the run looks for a failure. Never raises: a result file
    that cannot be rewritten leaves the in-memory verdict standing, which still fails the count."""
    import json

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            return
        stored["status_before_host_compute_veto"] = stored.get("status")
        stored.update({key: result[key] for key in ("status", "failure", "host_compute")})
        path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    except (OSError, ValueError, KeyError):
        return
