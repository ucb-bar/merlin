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

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any

from merlin.common import cpu_pool
from merlin.perf import isa_utilization
from merlin.perf.offload import offload_report

from .capsule_common import discover_capsules, make_run_paths, run_entrypoints
from .package_runtime import BackendDeclined, CertFailure

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


def host_compute_row(
    name: str, buffer: Mapping[str, Any], artifact_text: str, outcome: str,
    *, input_preparation=None, target=None,
) -> dict[str, Any]:
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
            HA.GroupSite(group=0, placement=placement, symbol=symbol, elements=elements, element_bytes=width), function,
            input_preparation=(input_preparation if input_preparation is not None
                               and input_preparation["entry_symbol"] == symbol else None),
            target=target,
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
                "arithmetic_basis",
                "host_arithmetic",
                "host_total_arithmetic",
                "host_address_arithmetic",
                "host_value_arithmetic",
                "host_raw_payload_bytes",
                "host_payload_bytes",
                "arithmetic_per_element",
                "value_arithmetic_per_element",
                "payload_ratio",
                "dominant_block",
                "input_preparation",
                "command_parameters",
                "command_only_refusal",
            )
        },
    }


def _source_host_compute(name, capsule, buffer, artifact, outcome, *, target):
    """Use the same original-source preparation proof in grading and diagnostic reports."""
    from .input_preparation import source_convolution

    try:
        preparation = source_convolution(capsule, target=target) if target is not None else None
    except (ValueError, KeyError, OSError):
        # An unavailable source binding grants no exemption from the original audit.
        preparation = None
    return host_compute_row(
        name, buffer, artifact, outcome,
        input_preparation=preparation, target=target,
    )


def instruction_use(artifacts: Iterable[str], *, target: str) -> dict[str, Any]:
    """Which of the target's DECLARED instructions the emitted programs ever use.

    The declared set is the target's own derived decode table. A target with no such table (a
    different endpoint, or facts not derived) reports why it cannot be measured; it does not report
    that everything was used.
    """
    from merlin.compile.analysis.rocc_commands import funct_table_for

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


def _summary(*, target: str, rows: list[dict[str, Any]], artifacts: list[str],
             audits: list[dict[str, Any]], execution: dict[str, Any]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {
        "schema": SCHEMA,
        "target": target,
        "execution": execution,
        "programs": rows,
        "outcomes": outcomes,
        "routed_macs": sum(int(row.get("routed_macs") or 0) for row in rows),
        "silent": [row["program"] for row in rows if row["outcome"] == "silent"],
        "placement_undeclared": [
            row["program"] for row in rows if row["outcome"] == "offloaded" and not row.get("placement_declared")
        ],
        "all_accounted": bool(rows) and not any(row["outcome"] in ("silent", "error") for row in rows),
        "instruction_use": instruction_use(artifacts, target=target),
        "host_compute": audits,
        "host_compute_vetoed": [
            row["program"] for row in audits if row["verdict"] == "host_compute_in_accelerator_group"
        ],
    }


def census_from_graded_artifacts(*, target: str, capsules: Iterable[Mapping[str, Any]],
                                 suite_runs: str | Path) -> dict[str, Any]:
    """Read the current public grade's own emitted buffers, without running a compiler again.

    The caller owns the grade's source identity and public-corpus selection. Missing or malformed
    artifacts stay named errors in the fixed denominator; they cannot become evidence of offload.
    Whole-model capstones are handled by their separate model placement audit, including those
    that carry an interface MLIR for their source model.
    """
    root = Path(suite_runs)
    all_capsules = [dict(capsule) for capsule in capsules]
    model_capsules = [str(capsule.get("name")) for capsule in all_capsules
                      if capsule.get("interface_mlir") and capsule.get("kind") == "model"]
    selected = [capsule for capsule in all_capsules
                if capsule.get("interface_mlir") and capsule.get("kind") != "model"]
    rows: list[dict[str, Any]] = []
    artifacts: list[str] = []
    audits: list[dict[str, Any]] = []
    for capsule in selected:
        name = str(capsule.get("name") or Path(str(capsule.get("__dir__", ""))).name)
        generated = root / name / "generated"
        buffer_path, artifact_path = generated / "command_buffer.json", generated / "lowered.llvm.mlir"
        try:
            result_path = root / name / "capsule_result.json"
            if (not name or name in (".", "..") or "/" in name or "\\" in name
                    or root.is_symlink() or (root / name).is_symlink() or generated.is_symlink()
                    or buffer_path.is_symlink() or artifact_path.is_symlink() or result_path.is_symlink()):
                raise ValueError("graded artifact path is indirect or invalid")
            if not result_path.is_file():
                raise FileNotFoundError("graded capsule result is absent")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (not isinstance(result, dict) or result.get("capsule") != name
                    or result.get("kind") != capsule.get("kind") or result.get("status") != "pass"):
                raise ValueError("graded capsule result does not prove a passing matching capsule")
            if result.get("kind") == "model_slice" and not buffer_path.is_file():
                # Provided host code has no candidate command buffer. Count it as host only
                # only when the grade proves source ineligibility, host execution, and no
                # device work in the linked ELF. A missing candidate artifact alone proves nothing.
                qualification = result.get("provided_host_qualification")
                eligibility = qualification.get("source_eligibility") if isinstance(qualification, dict) else None
                completeness = eligibility.get("denominator_completeness") if isinstance(eligibility, dict) else None
                host_check = result.get("host_execution_check")
                lane = result.get("lane_report")
                tiers = result.get("tiers")
                scan = lane.get("elf_scan") if isinstance(lane, dict) else None
                if (not isinstance(eligibility, dict) or eligibility.get("n_eligible") != 0
                        or not isinstance(completeness, dict)
                        or completeness.get("inventory_status") != "verified_against_parsed_contractions"
                        or completeness.get("unmatched_contraction_macs") != 0
                        or completeness.get("n_unpriceable_contractions") != 0
                        or completeness.get("unverified_operand_format_contractions") != []
                        or completeness.get("caveats") != []
                        or not isinstance(host_check, dict) or host_check.get("status") != "pass"
                        or not isinstance(lane, dict) or lane.get("required") != []
                        or lane.get("violated") != [] or "on_mesh" not in lane.get("forbidden", [])
                        or not isinstance(scan, dict) or scan.get("status") != "measured"
                        or scan.get("n_hits") != 0 or not isinstance(tiers, dict)
                        or any(not isinstance(tiers.get(tier), dict)
                               or tiers[tier].get("status") != "pass" for tier in ("L0", "L1", "L2", "L3"))):
                    raise ValueError("provided host result lacks passing source/placement/execution evidence")
                rows.append({"program": name, "outcome": "host_only", "commands": 0,
                             "routed_macs": 0, "host_regions": 1, "unit_regions": 0,
                             "contraction_offload_fraction": 0.0, "placement_declared": True,
                             "placement_basis": "provided_host_qualification"})
                audits.append({"program": name, "verdict": "declared_host",
                               "why": "passing provided-host qualification and linked ELF device-work scan"})
                continue
            if not buffer_path.is_file():
                raise FileNotFoundError("graded command buffer is absent")
            buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
            if not isinstance(buffer, dict):
                raise ValueError("graded command buffer is not an object")
            row = program_row(name, buffer)
            if row["outcome"] == "declined":
                refusal = result.get("expected_refusal")
                if (not isinstance(refusal, dict) or refusal.get("status") != "met"
                        or result.get("declined") != buffer.get("declined")):
                    raise ValueError("graded refusal is not a matching expected refusal")
                rows.append(row)
                continue
            if not artifact_path.is_file():
                raise FileNotFoundError("graded lowered host artifact is absent")
            artifact = artifact_path.read_text(encoding="utf-8")
            audit = _source_host_compute(name, capsule, buffer, artifact, row["outcome"], target=target)
        except Exception as failure:  # noqa: BLE001 -- an unreadable artifact cannot establish offload
            rows.append({
                "program": name, "outcome": "error",
                "reason": f"graded emission unavailable: {type(failure).__name__}: {str(failure)[:160]}",
            })
            continue
        rows.append(row)
        artifacts.append(artifact)
        audits.append(audit)
    return _summary(
        target=target, rows=rows, artifacts=artifacts, audits=audits,
        execution={"mode": "same_grade_artifacts", "programs_requested": len(selected),
                   "programs_observed": sum(row["outcome"] != "error" for row in rows),
                   "emitted_artifacts": len(artifacts),
                   "model_capsules_separate_gate": model_capsules},
    )


def census(
    package: str | Path,
    *,
    target: str,
    capsules: Iterable[Mapping[str, Any]],
    contract: str | Path | None = None,
    timeout: int = 300,
    workers: int = 0,
) -> dict[str, Any]:
    """Emit independent programs concurrently, returning accounting in input order.

    Zero workers selects ``MERLIN_GRADE_WORKERS`` (when unset, the shared compute
    pool size, else six), bounded by the shared compute roster. Build once before
    fan-out. Each case inherits the original execution context and owns an
    independent artifact directory; no parent CPU lease spans package subprocesses
    or their common-service RPCs.
    """
    from .package_runtime import build_package, integrity_scan, load_package

    if type(workers) is not int or workers < 0:
        raise ValueError("offload census workers must be a nonnegative integer")
    if workers == 0:
        raw = os.environ.get("MERLIN_GRADE_WORKERS", str(cpu_pool.compute_capacity() or 6))
        if not raw.isdecimal() or int(raw) < 1:
            raise ValueError("MERLIN_GRADE_WORKERS must be a positive integer")
        workers = int(raw)
    compute_cpus = cpu_pool.grading_cpus()
    preparation_cpus = cpu_pool.preparation_cpus()
    if cpu_pool.shared_compute_enabled():
        workers = min(workers, len(compute_cpus))
    selected = [dict(capsule) for capsule in capsules if capsule.get("interface_mlir")]
    workers = min(workers, len(selected)) if selected else 1
    rows: list[dict[str, Any]] = []
    artifacts: list[str] = []
    audits: list[dict[str, Any]] = []

    def name_of(capsule):
        return str(capsule.get("name") or Path(str(capsule.get("__dir__", ""))).name)

    def error_row(name, failure):
        if isinstance(failure, BackendDeclined):
            return {"program": name, "outcome": "declined", "reason": failure.reason, "commands": 0, "routed_macs": 0}
        detail = (f"{failure.plane}: {failure.detail[:200]}" if isinstance(failure, CertFailure)
                  else f"{type(failure).__name__}: {str(failure)[:200]}")
        return {"program": name, "outcome": "error", "reason": detail}

    with tempfile.TemporaryDirectory(prefix="offcensus_") as scratch:
        pkg = None
        preparation_failure = None
        if selected:
            try:
                with cpu_pool.shared_compute_pool():
                    pkg = load_package(package, contract=contract)
                    integrity_scan(pkg)
                # The original executor owns the build and its child invocations.
                # Holding our CPU lock here can deadlock a child common-service call.
                with cpu_pool.selected_pool(preparation_cpus):
                    build_package(pkg)
            except Exception as failure:  # noqa: BLE001 -- reported for every unexecuted program
                preparation_failure = failure

        def run_one(index, capsule):
            name = name_of(capsule)
            if preparation_failure is not None:
                return error_row(name, preparation_failure), None, None
            try:
                with cpu_pool.shared_compute_pool():
                    paths = make_run_paths(
                        Path(scratch) / f"case-{index:05d}" / "runs", name,
                        suite=f"{target}-offload-census", target=target,
                        dtype=str(capsule.get("dtype") or "unknown"),
                        benchmark=str(capsule.get("kind") or "capsule"),
                    )
                _pkg, buffer, _artifact = run_entrypoints(
                    pkg,
                    package,
                    capsule,
                    paths,
                    contract=contract,
                    timeout=timeout,
                    fourth_output_name="lowered.llvm.mlir",
                )
                with cpu_pool.shared_compute_pool():
                    row = program_row(name, buffer)
                    audit = _source_host_compute(name, capsule, buffer, _artifact or "", row["outcome"], target=target)
                return row, _artifact or "", audit
            except Exception as failure:  # noqa: BLE001 -- a harness fault is not an offload verdict
                return error_row(name, failure), None, None

        def owned_case(index, capsule):
            # Base affinity is a pool, not a held slot. Actual bounded Python
            # stages lease one CPU; child/RPC owners can acquire theirs freely.
            with cpu_pool.selected_pool(preparation_cpus):
                return run_one(index, capsule)

        if workers == 1:
            completed = [owned_case(index, capsule) for index, capsule in enumerate(selected)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = [pool.submit(copy_context().run, owned_case, index, capsule)
                           for index, capsule in enumerate(selected)]
                completed = [future.result() for future in pending]
        for row, artifact, audit in completed:
            rows.append(row)
            if artifact is not None:
                artifacts.append(artifact)
            if audit is not None:
                audits.append(audit)

    return _summary(
        target=target, rows=rows, artifacts=artifacts, audits=audits,
        execution={"workers": workers, "programs_requested": len(selected)},
    )


def census_of_corpus(
    package: str | Path,
    *,
    target: str,
    capsule_roots: Iterable[str | Path],
    contract: str | Path | None = None,
    timeout: int = 300,
    workers: int = 0,
) -> dict[str, Any]:
    return census(
        package,
        target=target,
        contract=contract,
        timeout=timeout,
        workers=workers,
        capsules=discover_capsules(list(capsule_roots), contract=contract),
    )


#: Set to ``0`` to report the host-compute audit beside a grade without letting it fail a capsule.
VETO_ENV = "MERLIN_HOST_COMPUTE_VETO"
VETO_CATEGORY = "host_compute_in_accelerator_group"


def apply_host_compute_veto(
    results: list[dict[str, Any]], suite_runs: Path, *, capsules=None, target=None,
) -> list[dict[str, Any]]:
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

    The audit runs in a short-lived child process that imports the code on disk
    (:mod:`.host_compute_veto_process`), so a long-lived grader applies the current check, not the
    one it imported at launch. Without a child answer it runs here, with the same result.
    """
    from . import host_compute_veto_process

    if host_compute_veto_process.enabled():
        vetoed = host_compute_veto_process.run(results, suite_runs, capsules=capsules, target=target)
        if vetoed is not None:
            return vetoed
    return _apply_host_compute_veto_here(results, suite_runs, capsules=capsules, target=target)


def _apply_host_compute_veto_here(
    results: list[dict[str, Any]], suite_runs: Path, *, capsules=None, target=None,
) -> list[dict[str, Any]]:
    """:func:`apply_host_compute_veto`, evaluated with the audit this process imported."""
    import json
    import os

    from merlin.verify import host_compute_audit as HA

    vetoed: list[dict[str, Any]] = []
    originals = {capsule["name"]: capsule for capsule in capsules or ()}
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
            audit = _source_host_compute(
                str(result.get("capsule")), originals.get(result.get("capsule"), {}),
                buffer, artifact_path.read_text(encoding="utf-8"), row["outcome"], target=target,
            )
        except Exception:  # noqa: BLE001 -- an unreadable artifact is not evidence of host compute
            continue
        result["host_compute"] = {
            key: audit.get(key) for key in ("verdict", "why", "symbol", "dominant_block", "input_preparation")
        }
        if audit.get("verdict") != HA.HOST_COMPUTE:
            continue
        vetoed.append(audit)
        if os.environ.get(VETO_ENV, "1") == "0":
            result["host_compute"]["veto_suppressed_by"] = VETO_ENV
            continue
        result["status"] = "fail"
        # With no value arithmetic and no memory traffic, nothing shows tensor compute: the veto
        # rests on integer work that could not be proven to be command orchestration.
        unproven = (audit.get("command_only_refusal") and audit.get("host_value_arithmetic") == 0
                    and audit.get("host_payload_bytes") == 0)
        lead = (
            "the program passes numerics; its host integer work is not proven to be command orchestration"
            if unproven
            else "the program passes numerics and spends its work on the host"
        )
        result["failure"] = {
            "plane": "host_compute",
            "category": VETO_CATEGORY,
            "detail": (
                f"{lead}: {audit.get('why')} "
                f"(function {audit.get('symbol')!r}). Work that scales with the payload belongs to "
                f"the unit: issue the commands that make the unit gather, accumulate or scale, in "
                f"place of a host loop over the elements. The host may compute addresses, issue "
                f"commands and branch. A program that DECLARES itself host-only is reported, not failed."
            ),
        }
        _record_veto(Path(suite_runs) / str(result.get("capsule")) / "capsule_result.json", result)
    return vetoed


#: The fields a recorded veto rewrites in ``capsule_result.json``; everything else stays as executed.
VETO_FIELDS = ("status", "failure", "host_compute")
VETO_PRIOR_STATUS = "status_before_host_compute_veto"


def is_recorded_veto(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """Whether ``after`` is exactly ``before`` with a host-compute veto recorded by :func:`_record_veto`.

    Readers that pinned a completed result before the score applied the veto use this to tell that
    rewrite apart from any other change: same execution evidence, a failed status, the veto category
    and the status it replaced."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    failure = after.get("failure")
    if (after.get("status") != "fail" or after.get(VETO_PRIOR_STATUS) != before.get("status")
            or not isinstance(failure, Mapping) or failure.get("category") != VETO_CATEGORY):
        return False
    untouched = (*VETO_FIELDS, VETO_PRIOR_STATUS)
    return ({k: v for k, v in before.items() if k not in untouched}
            == {k: v for k, v in after.items() if k not in untouched})


def _record_veto(path: Path, result: Mapping[str, Any]) -> None:
    """Put a veto where every reader of the run looks for a failure. Never raises: a result file
    that cannot be rewritten leaves the in-memory verdict standing, which still fails the count."""
    import json

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            return
        stored[VETO_PRIOR_STATUS] = stored.get("status")
        stored.update({key: result[key] for key in VETO_FIELDS})
        from merlin.common.jsonio import readable_json

        path.write_text(readable_json(stored), encoding="utf-8")
    except (OSError, ValueError, KeyError):
        return
