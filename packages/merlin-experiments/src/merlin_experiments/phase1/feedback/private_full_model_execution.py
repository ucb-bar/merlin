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
  (:func:`merlin.perf.float_accuracy.compare`); every element must be within.

The linked saved-model program prints one complete output and carries no per-group check, so
per-group exactness is not observable here and the result says so rather than implying it.

Nothing here names a target, a model, an engine binary or a reference file: those are the descriptor's
and the capture's. Results carry counts and error magnitudes, never tensor values.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "merlin.phase1.private_full_model_execution.v1"
#: ``phase1_gates.<GATE>.<SECTION>`` in the target descriptor.
GATE, SECTION = "whole_model", "private_full_models"
PASS, FAIL, DEFERRED, NOT_RUN = "pass", "fail", "deferred", "not_run"
#: Engines whose run is cited against the selected elaborated RTL (their facts must be the static gate's).
RTL_ENGINES = frozenset({"gsim", "verilator"})
#: Every engine a declaration may name; the target backend still decides whether it is available.
ENGINES = RTL_ENGINES | {"spike"}
#: The static gate's receipt for one program, below ``<static out>/<model>/<program>/``.
RECEIPT = "baremetal_model.json"
PER_GROUP_SCOPE = (
    "not observable: every result of the program is read back, group outputs are not; group cycles are recorded"
)
#: What a run on each engine can establish. The functional model is a numerics check only.
ENGINE_SCOPE = {
    "spike": "spike-functional: functional instruction-set model with the accelerator extension; numerics only, "
    "not RTL-derived, not certification, not timing",
    "gsim": "rtl-equivalent: elaborated-RTL engine; numerics and cycles",
    "verilator": "rtl-equivalent: elaborated-RTL engine; numerics and cycles",
}
_ENGINE_ENV_LOCK = threading.Lock()


class ExecutionGateError(ValueError):
    """The declaration or the evidence cannot support the numerical claim; the message says why."""


# ------------------------------------------------------------------------------------------ declaration


def _reference_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or not value.endswith((".npy", ".json")):
        raise ExecutionGateError(f"{where}: a reference is one plain capture file name ending .npy or .json")
    return value


def _positive_int(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExecutionGateError(f"{where} must be a positive integer")
    return value


def _tolerance(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ExecutionGateError(f"{where} must be a non-negative number")
    return float(value)


def _program_entry(entry: Any, where: str) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ExecutionGateError(f"{where}: declare an engine or a deferral")
    if "deferred" in entry:
        if "engine" in entry:
            raise ExecutionGateError(f"{where}: a program is either executed or deferred, not both")
        reason = entry["deferred"]
        if not isinstance(reason, str) or not reason.strip():
            raise ExecutionGateError(f"{where}: a deferral must state its reason")
        unknown = set(entry) - {"deferred", "estimate"}
        if unknown:
            raise ExecutionGateError(f"{where}: unknown deferral key(s) {sorted(unknown)}")
        return {"deferred": reason.strip(), "estimate": entry.get("estimate")}
    unknown = set(entry) - {
        "engine",
        "timeout_s",
        "max_cycles",
        "integer_reference",
        "require_integer_reference",
        "host_execution",
        "fp32_sanity",
        "board",
        "group_profile",
        "basis",
    }
    if unknown:
        raise ExecutionGateError(f"{where}: unknown key(s) {sorted(unknown)}")
    engine = entry.get("engine")
    if engine not in ENGINES:
        raise ExecutionGateError(f"{where}: engine must be one of {sorted(ENGINES)}")
    references = entry.get("integer_reference") or []
    if not isinstance(references, list):
        raise ExecutionGateError(f"{where}: integer_reference is a list of capture file names")
    required = entry.get("require_integer_reference", False)
    if type(required) is not bool or (required and not references):
        raise ExecutionGateError(f"{where}: require_integer_reference is a boolean that needs a named reference")
    max_cycles = entry.get("max_cycles")
    hosted = entry.get("host_execution")
    if hosted is not None and (not isinstance(hosted, Mapping) or set(hosted) != {"atol", "rtol"}):
        raise ExecutionGateError(f"{where}: host_execution declares exactly atol and rtol")
    if hosted is None and not required:
        raise ExecutionGateError(
            f"{where}: results without a required integer reference need a host_execution tolerance"
        )
    sanity = entry.get("fp32_sanity")
    if sanity is not None and (
        not isinstance(sanity, Mapping)
        or set(sanity) != {"reference", "cosine_margin", "top1"}
        or type(sanity["top1"]) is not bool
    ):
        raise ExecutionGateError(f"{where}: fp32_sanity declares exactly reference, cosine_margin and top1")
    if sanity is not None:
        margin = sanity["cosine_margin"]
        if isinstance(margin, bool) or not isinstance(margin, int | float) or margin < 1:
            raise ExecutionGateError(f"{where}.fp32_sanity.cosine_margin must be a number of at least 1")
    board = entry.get("board")
    if board is not None and (not isinstance(board, str) or not board):
        raise ExecutionGateError(f"{where}: board names one board of the selected catalog")
    profile = entry.get("group_profile", True)
    if type(profile) is not bool:
        raise ExecutionGateError(f"{where}: group_profile is a boolean")
    return {
        "board": board,
        "group_profile": profile,
        "engine": engine,
        "timeout_s": _positive_int(entry.get("timeout_s"), f"{where}.timeout_s"),
        "max_cycles": None if max_cycles is None else _positive_int(max_cycles, f"{where}.max_cycles"),
        "integer_reference": [_reference_name(name, f"{where}.integer_reference") for name in references],
        "require_integer_reference": required,
        "host_execution": None
        if hosted is None
        else {
            "atol": _tolerance(hosted["atol"], f"{where}.host_execution.atol"),
            "rtol": _tolerance(hosted["rtol"], f"{where}.host_execution.rtol"),
        },
        "fp32_sanity": None
        if sanity is None
        else {
            "reference": _reference_name(sanity["reference"], f"{where}.fp32_sanity.reference"),
            "cosine_margin": float(sanity["cosine_margin"]),
            "top1": sanity["top1"],
        },
        "basis": entry.get("basis"),
    }


def gate_for(descriptor: str | Path, *, required_programs: Mapping[str, Sequence[str]]) -> dict[str, Any] | None:
    """The validated execution declaration, or ``None`` when the descriptor declares none.

    ``required_programs`` is the static gate's roster (:func:`.private_full_models.program_requirements_for`);
    the declaration must name exactly its models and programs."""
    if not Path(descriptor).is_file():
        return None
    document = yaml.safe_load(Path(descriptor).read_bytes()) or {}
    gates = document.get("phase1_gates") if isinstance(document, Mapping) else None
    section = ((gates or {}).get(GATE) or {}).get(SECTION) if isinstance(gates, Mapping) else None
    if section is None:
        return None
    if not isinstance(section, Mapping) or set(section) - {"required", "programs"}:
        raise ExecutionGateError("private full-model execution declaration is malformed")
    required = section.get("required", True)
    if type(required) is not bool:
        raise ExecutionGateError("private full-model execution 'required' must be a boolean")
    programs = section.get("programs")
    if not required_programs:
        raise ExecutionGateError("an execution declaration needs the static private full-model roster")
    if not isinstance(programs, Mapping) or set(programs) != set(required_programs):
        raise ExecutionGateError("execution declaration names a different model roster than the static gate")
    roster: dict[str, dict[str, Any]] = {}
    for model, names in required_programs.items():
        declared = programs[model]
        if not isinstance(declared, Mapping) or list(declared) != list(names):
            raise ExecutionGateError(f"{model}: declare every program of the static roster, in its order")
        roster[model] = {name: _program_entry(declared[name], f"{model}.{name}") for name in names}
    if not any("engine" in entry for entries in roster.values() for entry in entries.values()):
        raise ExecutionGateError("every program is deferred; the execution gate would check nothing")
    return {"required": required, "programs": roster}


def build_options(gate: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """How the static gate links each program the declaration executes: every result printed in full,
    and group calls bracketed where declared. Deferred programs keep the default build."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model, program, entry in roster_of(gate):
        if "engine" in entry:
            out.setdefault(model, {})[program] = {
                "readback": "full",
                "group_profile": bool(entry.get("group_profile", True)),
            }
    return out


def roster_of(gate: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    return [(model, name, entry) for model, entries in gate["programs"].items() for name, entry in entries.items()]


# ------------------------------------------------------------------------------------------ execution


@contextmanager
def _engine_hang_bound(backend: Any, engine: str, max_cycles: int | None):
    """Hand ``max_cycles`` to an engine that takes its hang bound from the environment the backend names."""
    if max_cycles is None:
        yield
        return
    name = getattr(backend, f"{engine.upper()}_MAXCYCLES_ENV", None)
    if not isinstance(name, str) or not name:
        raise ExecutionGateError(f"the backend names no hang-bound variable for {engine}; max_cycles cannot apply")
    with _ENGINE_ENV_LOCK:
        previous = os.environ.get(name)
        os.environ[name] = str(max_cycles)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def run_engine(
    elf: Path,
    *,
    engine: str,
    target: str,
    timeout_s: int,
    max_cycles: int | None,
    elf_sha256: str,
    rtl_config: str | None,
    rtl_facts_sha256: str | None,
    capture_bytes: bool = False,
) -> dict[str, Any]:
    """Run one linked ELF on ``engine`` through the target backend; ``{console, wall_s, engine}``.

    ``capture_bytes`` keeps a full-readback console's raw payload bytes intact.

    An RTL engine is cited against the ambient selected facts, which must be the ones the static gate
    built against (same digest, same config), and its citation is revalidated around the run."""
    from merlin.compile import model_execution_inputs as MI
    from merlin.runtime.backends import base as backends

    evidence: dict[str, Any] = {"engine": engine}
    revalidate: Callable[[], None] = lambda: None  # noqa: E731
    command = None
    if engine in RTL_ENGINES:
        ambient = os.environ.get("MERLIN_RTL_FACTS", "").strip()
        if not ambient or not rtl_config:
            raise ExecutionGateError(f"{engine} execution needs the selected RTL facts and the static build's config")
        facts = MI.selected_firrtl(Path(ambient), target=target, config=rtl_config)
        if facts["sha256"] != rtl_facts_sha256:
            raise ExecutionGateError("ambient RTL facts differ from the facts the static gate built against")
        backend, citation, revalidate, prepare = MI.native_engine(target, engine, facts)
        evidence["engine_citation"] = citation
        evidence["rtl_facts_identity"] = facts
        if callable(prepare):
            command = prepare(
                elf, expected_elf_sha256=elf_sha256, expected_engine_provenance=citation, max_cycles=max_cycles
            )
            check = command.revalidate()
            evidence["native_command_sha256"] = check["command_sha256"]
            evidence["max_cycles"] = command.to_evidence().get("max_cycles")
    else:
        backend = backends.get_backend(target)
        if not backend.available(engine):
            raise ExecutionGateError(f"{target} backend reports {engine} unavailable")
    with _engine_hang_bound(backend, engine, max_cycles):
        revalidate()
        started = time.monotonic()
        console = backend.run_elf(elf, simulator=engine, timeout=timeout_s, capture_bytes=capture_bytes)
        wall = time.monotonic() - started
        if command is not None:
            command.revalidate()
        revalidate()
    if isinstance(console, bytes) and not capture_bytes:
        console = console.decode("utf-8", errors="replace")
    return {**evidence, "console": console, "wall_s": round(wall, 1)}


# ------------------------------------------------------------------------------------------ judgement


def _capture_file(capture: Path, name: str) -> tuple[Path, bool] | None:
    """``(path, bound)`` for a capture file, ``None`` when absent; ``bound`` when the capture receipt
    records its digest. A file the receipt records under different bytes is refused."""
    import json

    from merlin.compile.model_execution_inputs import file_sha256

    path = capture / name
    if not path.is_file() or path.is_symlink():
        return None
    receipt = json.loads((capture / "capture_receipt.json").read_text(encoding="utf-8"))
    recorded = (receipt.get("artifacts") or {}).get(name)
    if recorded is None:
        return path, False
    if recorded.get("sha256") != file_sha256(path):
        raise ExecutionGateError(f"capture reference {name} differs from the bytes the capture receipt binds")
    return path, True


def _bound_reference(capture: Path, name: str):
    """A ``.npy`` capture file the capture's own receipt binds by digest; ``None`` when absent or unbound."""
    import numpy as np

    found = _capture_file(capture, name)
    if found is None or not found[1]:
        return None
    return np.load(found[0], allow_pickle=False)


INTEGER_REFERENCE_SCHEMA = "merlin.capture.integer_reference.v1"
_ABI_DTYPES = {"f32": "float32", "f64": "float64", "f16": "float16", "i64": "int64", "i32": "int32", "i8": "int8"}


def integer_reference(capture: Path, name: str) -> list | None:
    """The quantized program's own results as the capture's integer reference states them (``None`` when the
    capture binds no such file): every result for ``integer-reference.json``, result 0 for a ``.npy``."""
    import json

    import numpy as np

    found = _capture_file(capture, name)
    if found is None:
        return None
    path, bound = found
    if not bound:
        raise ExecutionGateError(f"integer reference {name} is not bound by the capture receipt")
    if name.endswith(".npy"):
        return [np.load(path, allow_pickle=False)]
    document = json.loads(path.read_text(encoding="utf-8"))
    abi, outputs = document.get("output_abi"), document.get("outputs")
    if (
        document.get("schema") != INTEGER_REFERENCE_SCHEMA
        or not isinstance(abi, list)
        or not isinstance(outputs, list)
        or len(abi) != len(outputs)
    ):
        raise ExecutionGateError(f"integer reference {name} is not a {INTEGER_REFERENCE_SCHEMA} result list")
    out = []
    for row, value in zip(abi, outputs, strict=True):
        dtype = _ABI_DTYPES.get(str((row or {}).get("dtype")))
        if dtype is None:
            raise ExecutionGateError(f"integer reference {name} states an unsupported result dtype {row!r}")
        out.append(np.asarray(value, dtype=dtype).reshape([int(d) for d in row.get("shape") or []]))
    return out


def _fp32_reference(capture: Path, name: str, results: int) -> tuple[list, dict[str, Any]] | None:
    """The float model's results (one per program result) and how the file is identified."""
    import json

    import numpy as np

    from merlin.compile.model_execution_inputs import file_sha256

    found = _capture_file(capture, name)
    if found is None:
        return None
    path, bound = found
    identity = {"reference": name, "sha256": file_sha256(path), "receipt_bound": bound}
    if name.endswith(".npy"):
        return [np.load(path, allow_pickle=False)], identity
    document = json.loads(path.read_text(encoding="utf-8"))
    values = [document] if results == 1 else document
    if not isinstance(values, list) or len(values) != results:
        raise ExecutionGateError(f"fp32 reference {name} does not state one value per program result")
    return [np.asarray(value, dtype=np.float64) for value in values], identity


def _exact(values, reference) -> dict[str, Any]:
    """Value-exact agreement: every element equal as a value (+0 equals -0; NaN matches NaN)."""
    import numpy as np

    values = np.asarray(values).reshape(-1)
    reference = np.asarray(reference).reshape(-1)
    if values.size != reference.size:
        return {"passed": False, "note": f"program printed {values.size} of {reference.size} elements"}
    if values.dtype.kind == "f" or reference.dtype.kind == "f":
        if values.dtype.kind != "f":
            return {"passed": False, "note": f"output dtype {values.dtype} cannot be exact against {reference.dtype}"}
        left = values.astype(np.float64)
        right = reference.astype(values.dtype).astype(np.float64)
        same = (left == right) | (np.isnan(left) & np.isnan(right))
    elif values.dtype.kind in "iub" and reference.dtype.kind in "iub":
        same = values.astype(np.int64) == reference.astype(np.int64)
    else:
        return {"passed": False, "note": f"output dtype {values.dtype} cannot be exact against {reference.dtype}"}
    differ = int(np.count_nonzero(~same))
    row = {"passed": differ == 0, "mismatched_elements": differ, "of": int(reference.size)}
    if differ:
        row["max_abs"] = float(np.nanmax(np.abs(values.astype(np.float64) - reference.astype(np.float64))))
    return row


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else (1.0 if not a.any() and not b.any() else 0.0)


#: The smallest quantization distance a cosine floor is scaled from, so a quantized program whose results
#: equal the float model's (an unquantized program) still leaves float-order room.
COSINE_DISTANCE_FLOOR = 1e-6


def _fp32_sanity(values: list, quantized: list, fp32: list, sanity: Mapping[str, Any]) -> dict[str, Any]:
    """The float model as a sanity bound, calibrated from the quantized program's own distance to it.

    For each result: ``floor = 1 - cosine_margin * max(1 - cos(quantized, fp32), COSINE_DISTANCE_FLOOR)``,
    where ``quantized`` is the result the quantized program itself produces (its integer reference, or
    its host execution). Top-1 (result 0, last axis) is required to equal the float model's only on the
    rows where the quantized program's own top-1 does."""
    import numpy as np

    rows, passed = [], True
    for index, (value, own, float_value) in enumerate(zip(values, quantized, fp32, strict=True)):
        if own is None or np.asarray(float_value).size != np.asarray(value).size:
            rows.append({"index": index, "passed": False, "note": "no calibration or a different result size"})
            passed = False
            continue
        distance = max(1.0 - _cosine(own, float_value), COSINE_DISTANCE_FLOOR)
        floor = 1.0 - sanity["cosine_margin"] * distance
        observed = _cosine(value, float_value)
        rows.append(
            {
                "index": index,
                "cosine": observed,
                "floor": floor,
                "quantized_program_cosine": _cosine(own, float_value),
                "passed": observed >= floor,
            }
        )
        passed = passed and observed >= floor
    top1: dict[str, Any] = {"status": "not_declared"}
    if sanity["top1"]:
        width = int(np.asarray(fp32[0]).shape[-1]) if np.asarray(fp32[0]).ndim else 1
        device = np.asarray(values[0], np.float64).reshape(-1, width).argmax(axis=-1)
        own = (
            np.asarray(quantized[0], np.float64).reshape(-1, width).argmax(axis=-1)
            if quantized[0] is not None
            else None
        )
        ref = np.asarray(fp32[0], np.float64).reshape(-1, width).argmax(axis=-1)
        if own is None or device.shape != ref.shape:
            top1 = {"passed": False, "note": "result 0 has no calibration or a different shape"}
        else:
            applicable = own == ref
            agree = int(np.count_nonzero(device[applicable] == ref[applicable]))
            top1 = {
                "rows": int(ref.size),
                "applicable_rows": int(np.count_nonzero(applicable)),
                "agreeing_rows": agree,
                "passed": agree == int(np.count_nonzero(applicable)),
            }
            if not applicable.all():
                top1["note"] = "rows where the quantized program's own top-1 differs from the float model are excluded"
        passed = passed and top1["passed"]
    return {"passed": passed, "results": rows, "top1": top1}


def host_execution_outputs(capture: Path, workdir: Path) -> list:
    """Every result of the saved program executed on the host, the independent reference for results
    the capture binds no reference for."""
    import numpy as np

    from merlin.runtime.dispatch_runtime import run_model

    result = run_model(capture, workdir)
    return [np.asarray(value) for value in (result.get("outputs") or [result["output"]])]


def _host_outputs_in_scratch(capture: Path) -> list:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="merlin-host-reference-") as scratch:
        return host_execution_outputs(capture, Path(scratch))


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
    from merlin.compile.baremetal_model import read_full_outputs

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
            hosted = (host_reference or _host_outputs_in_scratch)(capture)
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
    cycles = metrics.get("cycles")
    return {
        "checks": checks,
        "cycles": int(cycles) if isinstance(cycles, str) and cycles.isdigit() else None,
        "passed": all(check.get("passed", True) for check in checks.values()),
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

    from merlin.compile.model_execution_inputs import file_sha256

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
    )
    if file_sha256(elf) != expected:
        raise ExecutionGateError("the linked ELF changed during execution")
    console = run["console"]
    verdict = judge_full(console if isinstance(console, bytes) else str(console).encode(), receipt, entry)
    profile = None
    if (receipt.get("inputs") or {}).get("group_profile"):
        from merlin.runtime.out_bin import binary_console_diagnostics
        from merlin.runtime.whole_model_readback import parse_group_profile

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
    execute: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute every declared program of the static gate's passing models; one record per program.

    ``static`` is :func:`.private_full_models.run`'s result and ``static_out`` the directory it wrote.
    A program whose model did not pass the static gate is ``not_run`` (and fails an executed
    declaration); a deferred program is recorded as deferred with its declared reason."""
    execute = execute or run_engine
    static_out = Path(static_out)
    built = _static_programs(static)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for model, program, entry in roster_of(gate):
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
        rows.append(row)
    executed = [row for row in rows if row["status"] != DEFERRED]
    deferred = [f"{row['model']}:{row['program']}" for row in rows if row["status"] == DEFERRED]
    passed = bool(executed) and all(row["status"] == PASS for row in executed)
    return {
        "schema": SCHEMA,
        "target": target,
        "required": gate["required"],
        "candidate_tree_sha256": static.get("candidate_tree_sha256"),
        "programs": rows,
        "deferred": deferred,
        "passed": passed,
        "numerical_scope": (
            "every declared program executed"
            if not deferred
            else "executed programs only; declared deferrals are not numerical evidence"
        ),
        "per_group_exactness": PER_GROUP_SCOPE,
        "wall_s": round(time.monotonic() - started, 1),
    }


def not_run(gate: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """The record of a declared gate that could not start (e.g. the static gate failed)."""
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
    if not isinstance(record, Mapping) or record.get("schema") != SCHEMA or record.get("passed") is not True:
        return False
    if record.get("candidate_tree_sha256") != candidate_sha256 or not isinstance(static, Mapping):
        return False
    if static.get("candidate_tree_sha256") != candidate_sha256:
        return False
    rows = record.get("programs")
    roster = roster_of(gate)
    if not isinstance(rows, list) or len(rows) != len(roster):
        return False
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
            or not isinstance(row.get("checks"), Mapping)
            or not all(isinstance(c, Mapping) and c.get("passed", True) for c in row["checks"].values())
        ):
            return False
    return True


__all__ = [
    "DEFERRED",
    "ENGINE_SCOPE",
    "COSINE_DISTANCE_FLOOR",
    "INTEGER_REFERENCE_SCHEMA",
    "integer_reference",
    "build_options",
    "judge_full",
    "ENGINES",
    "SCHEMA",
    "ExecutionGateError",
    "complete",
    "gate_for",
    "not_run",
    "roster_of",
    "run",
    "run_engine",
]
