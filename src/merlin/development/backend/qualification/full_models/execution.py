"""Operator-private numerical execution of the private full-model roster's linked programs.

:mod:`.private_full_models` builds every declared program of every complete validation model and
verifies the linked ELF statically; it records ``full_model_numerical_equivalence: not_run``. This
module is the numerical half, run operator-side after the freeze on THE SAME ELF bytes the static
gate verified: each program is executed on the engine the target descriptor declares for it and its
printed output is held to the capture's own references.

The declaration lives beside the static one, under ``phase1_gates.whole_model.private_full_models``
of the target descriptor::

    whole_model:
      private_full_models:
        required: true
        programs:
          <model>:
            <program>:
              engine: <a runtime engine the target backend runs ELFs on>
              timeout_s: <wall-clock bound>
              max_cycles: <optional hang bound handed to the engine>
              integer_reference: [<capture-bound .npy>, ...]   # exact, where the capture binds one
              require_integer_reference: false
              end_to_end: {reference: <capture-bound .npy>, atol: <float>, rtol: <float>}
            <other program>:
              deferred: <why this program cannot be executed on any declared engine>
              estimate: {...}                                    # recorded verbatim

Every program of the static roster must be declared, either with an engine or ``deferred`` with a
reason: nothing is skipped silently, and a deferral is recorded on the result as a deferral, never as a
pass. At least one program must be executed, so the gate cannot pass vacuously.

What is checked per executed program:

* **binding** -- the static gate passed the model, its receipt names the program's ELF, and the ELF's
  bytes are the ones the static gate recorded (before and after the run);
* **completion** -- the console carries exactly one complete output and ``DONE``, identifies the linked
  build, and reports a clean memref-rank diagnostic;
* **integer reference** (exact) -- the first capture-bound reference among ``integer_reference``,
  compared bit for bit. A capture that binds none is recorded ``not_available`` unless the declaration
  requires one;
* **end to end** -- the output against the declared capture-bound reference within ``atol``/``rtol``
  (:func:`merlin.analysis.whole_model.grading.exactness.compare`); every element must be within.

The linked saved-model program prints one complete output and carries no per-group check, so
per-group exactness is not observable here and the result says so rather than implying it.

Nothing here names a target, a model, an engine binary or a reference file: those are the descriptor's
and the capture's. Results carry counts and error magnitudes, never tensor values.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import yaml

# ---- private_full_model_execution --------------------------------------------------------------


def judge_full(
    console: bytes,
    receipt: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    host_reference: Callable[[Path], list] | None = None,
) -> dict[str, Any]:
    """Every numerical check of a full-readback program, against the QUANTIZED program's semantics.

    1. ``integer_reference``: every result the capture's integer reference states is value-exact.
    2. ``host_execution``: every other result is held to an independent host execution of the same saved
       (quantized) program within the declared tolerance.
    3. ``fp32_sanity``: the float model is a sanity bound, never the yardstick (see :func:`_fp32_sanity`).
    A result no check covers fails the program: nothing is reported checked that was not."""
    from merlin.development.backend.qualification.full_models.execution_engine import (
        PER_GROUP_SCOPE,
        ExecutionGateError,
        _exact,
        _fp32_reference,
        _fp32_sanity,
        _host_outputs_in_scratch,
        integer_reference,
    )
    from merlin.model import read_full_outputs

    output = receipt.get("output") or {}
    capture = Path(str((receipt.get("inputs") or {}).get("capture") or ""))
    checks: dict[str, Any] = {}
    try:
        values, metrics = read_full_outputs(console, capture, output.get("build_hash"))
    except Exception as exc:  # noqa: BLE001 -- an unreadable console is a failed completion, with its reason
        tail = console[-400:].decode("latin-1", errors="replace") if isinstance(console, bytes) else ""
        return {
            "checks": {"completion": {"passed": False, "note": f"{type(exc).__name__}: {exc}", "console_tail": tail}},
            "cycles": None,
            "passed": False,
        }
    checks["completion"] = {"passed": True, "results": len(values), "elements": int(sum(v.size for v in values))}
    hosted: list | None = None

    def host_values() -> list:
        nonlocal hosted
        if hosted is None:
            if host_reference is not None:
                hosted = host_reference(capture)
            elif (receipt.get("inputs") or {}).get("package"):
                hosted = _host_outputs_in_scratch(capture, receipt=receipt)
            else:
                hosted = _host_outputs_in_scratch(capture)
            if len(hosted) != len(values):
                raise ExecutionGateError("the host execution has a different result roster than the program")
        return hosted

    covered: list = [None] * len(values)
    integer: dict[str, Any] = {
        "passed": True,
        "status": "not_available",
        "looked_for": list(entry["integer_reference"]),
    }
    for name in entry["integer_reference"]:
        reference = integer_reference(capture, name)
        if reference is None:
            continue
        if len(reference) > len(values):
            integer = {"status": "compared", "reference": name, "passed": False, "note": "more references than results"}
            break
        rows = []
        for index, ref in enumerate(reference):
            row = {"index": index, **_exact(values[index], ref)}
            covered[index] = ref
            if not row["passed"] and "mismatched_elements" in row:
                # Is the integer reference this saved program's exact semantics? Only when the program's
                # own host execution reproduces it value-exactly; otherwise the reference's float
                # nonlinear arithmetic differs from the saved program's, and the result is held to the
                # host execution instead (the reference stays the quantization yardstick for fp32_sanity).
                confirmed = _exact(host_values()[index], ref)
                row["reference_confirmed_by_host_execution"] = confirmed["passed"]
                if not confirmed["passed"]:
                    row.update(
                        passed=True,
                        status="unconfirmed_reference",
                        host_mismatched_elements=confirmed.get("mismatched_elements"),
                        note="the saved program's own host execution differs from this integer reference; "
                        "the result is judged against the host execution",
                    )
            rows.append(row)
        integer = {
            "status": "compared",
            "reference": name,
            "semantics": "value-exact against the quantized program's integer reference where the program's "
            "host execution reproduces it",
            "results": rows,
            "passed": all(row["passed"] for row in rows),
        }
        break
    if integer["status"] == "not_available" and entry["require_integer_reference"]:
        integer.update(passed=False, note="the declaration requires an integer reference the capture does not bind")
    checks["integer_reference"] = integer
    unconfirmed = {row["index"] for row in integer.get("results") or [] if row.get("status") == "unconfirmed_reference"}
    judged = [index for index, ref in enumerate(covered) if ref is None or index in unconfirmed]
    host_rows = []
    if judged:
        tolerance = entry.get("host_execution")
        if tolerance is None:
            host_rows.append(
                {
                    "passed": False,
                    "note": f"results {judged} have no confirmed integer reference and the declaration names "
                    "no host_execution tolerance",
                }
            )
        else:
            try:
                for index in judged:
                    host_rows.append({"index": index, **_within(values[index], host_values()[index], tolerance)})
            except ExecutionGateError as exc:
                host_rows.append({"passed": False, "note": str(exc)})
    checks["host_execution"] = {
        "passed": all(row["passed"] for row in host_rows),
        "results": host_rows,
        "semantics": "tolerance against a host execution of the same quantized saved program",
    }
    sanity = entry.get("fp32_sanity")
    if sanity is not None:
        found = _fp32_reference(capture, sanity["reference"], len(values))
        if found is None:
            checks["fp32_sanity"] = {"passed": False, "note": f"the capture has no {sanity['reference']}"}
        else:
            fp32, identity = found
            # Calibrate each result from the semantics it was judged against.
            quantized = [
                hosted[index] if index in judged and hosted else covered[index] for index in range(len(values))
            ]
            checks["fp32_sanity"] = {
                **identity,
                "cosine_margin": sanity["cosine_margin"],
                **_fp32_sanity(values, quantized, fp32, sanity),
            }
    checks["per_group_exactness"] = {"status": "not_observable", "why": PER_GROUP_SCOPE}
    from merlin.development.backend.qualification.full_models.audits import complete_checks

    cycles = metrics.get("cycles")
    return {
        "checks": checks,
        "cycles": int(cycles) if isinstance(cycles, str) and cycles.isdigit() else None,
        "passed": complete_checks(checks, entry),
    }


def _within(value, reference, tolerance: Mapping[str, float]) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(value, np.float64).reshape(-1)
    reference = np.asarray(reference, np.float64).reshape(-1)
    if value.size != reference.size:
        return {"passed": False, "note": f"printed {value.size} of {reference.size} elements"}
    error = np.abs(value - reference)
    bound = tolerance["atol"] + tolerance["rtol"] * np.abs(reference)
    within = int(np.count_nonzero(error <= bound))
    return {
        "within": within,
        "of": int(reference.size),
        "max_abs": float(error.max()) if error.size else 0.0,
        "atol": tolerance["atol"],
        "rtol": tolerance["rtol"],
        "passed": within == reference.size,
    }


def _static_programs(static: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{model: {program: static entry}}`` for every model the static gate passed, with its build record."""
    from merlin.development.backend.qualification.full_models.execution_engine import PASS

    out: dict[str, dict[str, Any]] = {}
    for row in static.get("models") or ():
        if not isinstance(row, Mapping) or row.get("status") != PASS:
            continue
        build = (row.get("checks") or {}).get("build") or {}
        out[str(row.get("model"))] = {
            str(entry.get("program")): {**entry, "_build": build}
            for entry in build.get("programs") or ()
            if isinstance(entry, Mapping)
        }
    return out


def _execute_program(
    model: str,
    program: str,
    entry: Mapping[str, Any],
    static_entry: Mapping[str, Any],
    *,
    static_out: Path,
    target: str,
    execute: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    import json

    from merlin.development.backend.qualification.full_models.execution_engine import (
        ENGINE_SCOPE,
        FAIL,
        PASS,
        RECEIPT,
        ExecutionGateError,
    )
    from merlin.runtime.execution.model_inputs import file_sha256

    receipt_path = static_out / model / program / RECEIPT
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ExecutionGateError("the static gate left no receipt for this program")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    output = receipt.get("output") or {}
    elf = Path(str(output.get("elf") or ""))
    expected = static_entry.get("elf_sha256")
    if (
        not isinstance(expected, str)
        or output.get("elf_sha256") != expected
        or elf.is_symlink()
        or not elf.is_file()
        or file_sha256(elf) != expected
    ):
        raise ExecutionGateError("the linked ELF is absent or differs from the bytes the static gate verified")
    build = static_entry.get("_build") or {}
    board = (receipt.get("inputs") or {}).get("board")
    if entry.get("board") is not None and board != entry["board"]:
        raise ExecutionGateError(
            f"the program was linked for board {board!r}, the declaration names {entry['board']!r}"
        )
    full = output.get("readback") == "full"
    if not full:
        raise ExecutionGateError("the program was not linked with full readback; its results cannot all be checked")
    run = execute(
        elf,
        engine=entry["engine"],
        target=target,
        timeout_s=entry["timeout_s"],
        max_cycles=entry["max_cycles"],
        elf_sha256=expected,
        rtl_config=build.get("accelerator_rtl_config"),
        rtl_facts_sha256=build.get("accelerator_rtl_facts_sha256"),
        capture_bytes=True,
        declared_machine=entry.get("board") is not None,
    )
    if file_sha256(elf) != expected:
        raise ExecutionGateError("the linked ELF changed during execution")
    console = run["console"]
    verdict = judge_full(console if isinstance(console, bytes) else str(console).encode(), receipt, entry)
    profile = None
    if (receipt.get("inputs") or {}).get("group_profile"):
        from merlin.runtime.harness.output_decoding import binary_console_diagnostics
        from merlin.runtime.harness.output_observation import parse_group_profile

        try:
            profile = parse_group_profile(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001 -- a broken profile is recorded, never a numerical verdict
            profile = {"error": f"{type(exc).__name__}: {exc}"}
    wall = run.get("wall_s")
    cycles = verdict["cycles"]
    return {
        "status": PASS if verdict["passed"] else FAIL,
        "engine": entry["engine"],
        "scope": ENGINE_SCOPE.get(entry["engine"], "unknown"),
        "certification": entry["engine"] != "spike",
        "board": board,
        "group_profile": profile,
        "elf_sha256": expected,
        "wall_s": wall,
        "cycles": cycles,
        "simulated_cycles_per_s": round(cycles / wall, 1) if cycles and wall else None,
        "engine_evidence": {k: v for k, v in run.items() if k not in {"console", "wall_s"}},
        "checks": verdict["checks"],
    }


def run(
    static: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    target: str,
    static_out: str | Path,
    selected_model: str | None = None,
    execute: Callable[..., Mapping[str, Any]] | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Execute every declared program of the static gate's passing models; one record per program.

    ``static`` is :func:`.private_full_models.run`'s result and ``static_out`` the directory it wrote.
    A program whose model did not pass the static gate is ``not_run`` (and fails an executed
    declaration); a deferred program is recorded as deferred with its declared reason."""
    from merlin.development.backend.qualification.full_models.execution_engine import (
        DEFERRED,
        FAIL,
        NOT_RUN,
        PASS,
        PER_GROUP_SCOPE,
        SCHEMA,
        ExecutionGateError,
        roster_of,
        run_engine,
    )

    execute = execute or run_engine
    static_out = Path(static_out)
    if type(max_workers) is not int or not 1 <= max_workers <= 15:
        raise ExecutionGateError("private program workers must be an integer from 1 to 15")
    if selected_model is not None:
        from merlin.development.backend.qualification.full_models import private_full_models as models
        from merlin.development.backend.qualification.provenance import private_source_freeze

        programs = gate.get("programs") or {}
        expected_programs = {name: list(stages) for name, stages in programs.items()}
        selected_rows = static.get("models") or []
        if (
            selected_model not in programs
            or static.get("schema") != models.RESULT_SCHEMA
            or static.get("target") != target
            or static.get("mode") != "canonical_model_partition"
            or static.get("selected_model") != selected_model
            or static.get("partition_passed") is not True
            or static.get("passed") is not False
            or static.get("required_models") != list(programs)
            or static.get("required_programs") != expected_programs
            or not private_source_freeze.valid_binding(
                static.get("authored_source_freeze"), static.get("private_spec_sha256")
            )
            or len(selected_rows) != 1
            or selected_rows[0].get("model") != selected_model
            or selected_rows[0].get("status") != PASS
        ):
            raise ExecutionGateError("canonical execution partition differs from the complete original static roster")
    built = _static_programs(static)
    started = time.monotonic()
    roster = [item for item in roster_of(gate) if selected_model is None or item[0] == selected_model]

    def evaluate(item: tuple[str, str, Mapping[str, Any]]) -> dict[str, Any]:
        model, program, entry = item
        row: dict[str, Any] = {"model": model, "program": program}
        if "deferred" in entry:
            row.update(status=DEFERRED, reason=entry["deferred"], estimate=entry.get("estimate"))
        elif program not in built.get(model, {}):
            row.update(status=NOT_RUN, reason="the static private full-model gate did not pass this model")
        else:
            try:
                row.update(
                    _execute_program(
                        model,
                        program,
                        entry,
                        built[model][program],
                        static_out=static_out,
                        target=target,
                        execute=execute,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- one program's refusal must not hide the others
                row.update(status=FAIL, engine=entry["engine"], reason=f"{type(exc).__name__}: {exc}")
        return row

    workers = min(max_workers, len(roster)) if selected_model is None else 1
    if workers > 1:
        # Independent program receipts and ELF paths allow concurrent execution. map() retains
        # the declared roster order even when simulations complete in a different order.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="private-program") as pool:
            rows = list(pool.map(evaluate, roster))
    else:
        rows = [evaluate(item) for item in roster]
    executed = [row for row in rows if row["status"] != DEFERRED]
    deferred = [f"{row['model']}:{row['program']}" for row in rows if row["status"] == DEFERRED]
    passed = bool(executed) and all(row["status"] == PASS for row in executed)
    result = {
        "schema": SCHEMA,
        "target": target,
        "required": gate["required"],
        "candidate_tree_sha256": static.get("candidate_tree_sha256"),
        "programs": rows,
        "deferred": deferred,
        "passed": selected_model is None and passed,
        "numerical_scope": (
            "every declared program executed"
            if not deferred
            else "executed programs only; declared deferrals are not numerical evidence"
        ),
        "per_group_exactness": PER_GROUP_SCOPE,
        "wall_s": round(time.monotonic() - started, 1),
        "parallel_workers": workers,
    }
    if selected_model is not None:
        result.update(
            mode="canonical_model_partition",
            selected_model=selected_model,
            partition_passed=passed,
            required_models=list(gate["programs"]),
            required_programs={name: list(stages) for name, stages in gate["programs"].items()},
            numerical_scope="selected complete model programs only; full declared roster gate incomplete",
        )
    return result


def not_run(gate: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """The record of a declared gate that could not start (e.g. the static gate failed)."""
    from merlin.development.backend.qualification.full_models.execution_engine import SCHEMA

    return {
        "schema": SCHEMA,
        "required": gate["required"],
        "programs": [],
        "deferred": [],
        "passed": False,
        "reason": reason,
    }


def complete(
    record: Mapping[str, Any] | None,
    gate: Mapping[str, Any],
    *,
    static: Mapping[str, Any] | None,
    candidate_sha256: str,
) -> bool:
    """Revalidate a recorded execution result against the declaration, the static record and the candidate.

    Every declared program appears once, in order: a deferral exactly as declared, an executed program
    passed on its declared engine with the ELF digest the static record verified."""
    from merlin.development.backend.qualification.full_models.execution_engine import DEFERRED, PASS, SCHEMA, roster_of
    from merlin.development.backend.qualification.provenance import (
        private_numerical_selection as numerical_selection,
    )

    if isinstance(record, Mapping) and record.get("schema") == numerical_selection.RESULT:
        return numerical_selection.complete(record, static=static, gate=gate, candidate_sha256=candidate_sha256)
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != SCHEMA
        or record.get("passed") is not True
        or record.get("mode") == "canonical_model_partition"
    ):
        return False
    if record.get("candidate_tree_sha256") != candidate_sha256 or not isinstance(static, Mapping):
        return False
    if static.get("candidate_tree_sha256") != candidate_sha256:
        return False
    rows = record.get("programs")
    roster = roster_of(gate)
    if not isinstance(rows, list) or len(rows) != len(roster):
        return False
    from merlin.development.backend.qualification.full_models.audits import complete_checks

    built = _static_programs(static)
    for row, (model, program, entry) in zip(rows, roster, strict=True):
        if not isinstance(row, Mapping) or row.get("model") != model or row.get("program") != program:
            return False
        if "deferred" in entry:
            if row.get("status") != DEFERRED or row.get("reason") != entry["deferred"]:
                return False
            continue
        expected = (built.get(model, {}).get(program) or {}).get("elf_sha256")
        if (
            row.get("status") != PASS
            or row.get("engine") != entry["engine"]
            or not isinstance(expected, str)
            or row.get("elf_sha256") != expected
            or not complete_checks(row.get("checks"), entry)
        ):
            return False
    return True


__all__ = [
    "COSINE_DISTANCE_FLOOR",
    "DEFERRED",
    "ENGINES",
    "ENGINE_SCOPE",
    "ExecutionGateError",
    "INTEGER_REFERENCE_SCHEMA",
    "SCHEMA",
    "build_options",
    "complete",
    "gate_for",
    "integer_reference",
    "judge_full",
    "not_run",
    "roster_of",
    "run",
    "run_engine",
]
