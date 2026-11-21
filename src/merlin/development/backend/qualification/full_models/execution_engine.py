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

SCHEMA = "merlin.phase1.private_full_model_execution.v1"

GATE, SECTION = "whole_model", "private_full_models"

PASS, FAIL, DEFERRED, NOT_RUN = "pass", "fail", "deferred", "not_run"

RTL_ENGINES = frozenset({"gsim", "verilator"})

ENGINES = RTL_ENGINES | {"spike"}

RECEIPT = "baremetal_model.json"

PER_GROUP_SCOPE = (
    "not observable: every result of the program is read back, group outputs are not; group cycles are recorded"
)

ENGINE_SCOPE = {
    "spike": "spike-functional: functional instruction-set model with the accelerator extension; numerics only, "
    "not RTL-derived, not certification, not timing",
    "gsim": "rtl-equivalent: elaborated-RTL engine; numerics and cycles",
    "verilator": "rtl-equivalent: elaborated-RTL engine; numerics and cycles",
}

_ENGINE_ENV_LOCK = threading.Lock()


class ExecutionGateError(ValueError):
    """The declaration or the evidence cannot support the numerical claim; the message says why."""


def _reference_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or not value.endswith((".npy", ".json")):
        raise ExecutionGateError(f"{where}: a reference is one plain capture file name ending .npy or .json")
    return value


def _positive_int(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExecutionGateError(f"{where} must be a positive integer")
    return value


def _tolerance(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
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
        "device_ir_budget",
        "basis",
        "host_llvm_bitcode",
    }
    if unknown:
        raise ExecutionGateError(f"{where}: unknown key(s) {sorted(unknown)}")
    engine = entry.get("engine")
    if engine not in ENGINES:
        raise ExecutionGateError(f"{where}: engine must be one of {sorted(ENGINES)}")
    references = entry.get("integer_reference") or []
    if not isinstance(references, list):
        raise ExecutionGateError(f"{where}: integer_reference is a list of capture file names")
    # Author policy: integer references are opt-in, with explicit host tolerances required otherwise.
    required = entry["require_integer_reference"] if "require_integer_reference" in entry else False
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
        if isinstance(margin, bool) or not isinstance(margin, int | float) or not math.isfinite(margin) or margin < 1:
            raise ExecutionGateError(f"{where}.fp32_sanity.cosine_margin must be a number of at least 1")
    board = entry.get("board")
    if board is not None and (not isinstance(board, str) or not board):
        raise ExecutionGateError(f"{where}: board names one board of the selected catalog")
    # Profiling is enabled unless the authored build policy explicitly disables it.
    profile = entry["group_profile"] if "group_profile" in entry else True
    if type(profile) is not bool:
        raise ExecutionGateError(f"{where}: group_profile is a boolean")
    # Retaining bitcode is an optional build product, not a proof claim.
    bitcode = entry["host_llvm_bitcode"] if "host_llvm_bitcode" in entry else False
    if type(bitcode) is not bool:
        raise ExecutionGateError(f"{where}: host_llvm_bitcode is a boolean")
    budget = entry.get("device_ir_budget")
    if budget is not None:
        from merlin.compiler.lower.device.selection import DeviceIRBudget

        if type(budget) is not dict:
            raise ExecutionGateError(f"{where}: device_ir_budget is explicit original-caller policy data")
        try:
            budget = DeviceIRBudget(**budget).record()
        except (TypeError, ValueError) as error:
            raise ExecutionGateError(f"{where}: invalid device_ir_budget: {error}") from error
    return {
        **({"host_llvm_bitcode": True} if bitcode else {}),
        "board": board,
        "group_profile": profile,
        **({"device_ir_budget": budget} if budget is not None else {}),
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
    # Declaring this gate makes it required unless the operator explicitly opts out.
    required = section["required"] if "required" in section else True
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
                "group_profile": entry["group_profile"],
            }
            if entry.get("board") is not None:
                out[model][program]["board"] = entry["board"]
            if entry.get("host_llvm_bitcode"):
                out[model][program]["host_llvm_bitcode"] = True
            if entry.get("device_ir_budget") is not None:
                out[model][program]["device_ir_budget"] = dict(entry["device_ir_budget"])
    return out


def roster_of(gate: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    return [(model, name, entry) for model, entries in gate["programs"].items() for name, entry in entries.items()]


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
    declared_machine: bool = False,
) -> dict[str, Any]:
    """Run one linked ELF on ``engine`` through the target backend; ``{console, wall_s, engine}``.

    ``capture_bytes`` keeps a full-readback console's raw payload bytes intact. ``declared_machine``
    runs a whole-model Spike image on the machine its program declaration names (its ``board``)
    instead of the simulator's default machine.

    An RTL engine is cited against the ambient selected facts, which must be the ones the static gate
    built against (same digest, same config), and its citation is revalidated around the run."""
    from merlin.common.process.ownership import ExecutionDeadline
    from merlin.development.backend.campaign.control import require_full_model
    from merlin.development.backend.campaign.control import selected as joint_campaign
    from merlin.development.targetgen.execution.runner.compile import _selected_run_timeout
    from merlin.runtime.backends import registry as backends
    from merlin.runtime.execution import model_inputs as MI

    if joint_campaign() is not None:
        timeout_s = min(timeout_s, require_full_model(engine=engine))
    deadline = ExecutionDeadline.start(timeout_s)
    evidence: dict[str, Any] = {"engine": engine}
    revalidate: Callable[[], None] = lambda: None  # noqa: E731
    command = None
    if engine in RTL_ENGINES:
        ambient = os.environ.get("MERLIN_RTL_FACTS", "").strip()
        if not ambient or not rtl_config:
            raise ExecutionGateError(f"{engine} execution needs the selected RTL facts and the static build's config")
        built_facts = MI.selected_firrtl(Path(ambient), target=target, config=rtl_config)
        facts = MI.selected_execution_firrtl(Path(ambient), target=target)
        if built_facts["sha256"] != rtl_facts_sha256 or facts["sha256"] != rtl_facts_sha256:
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
    from merlin.runtime.backends.rocc.bound_backend import BoundRoCCBackend

    private_bound_gsim = engine == "gsim" and max_cycles is not None and isinstance(backend, BoundRoCCBackend)
    if private_bound_gsim:
        evidence["max_cycles"] = max_cycles
        evidence["max_cycle_source"] = "frozen_private_execution_declaration"
    with nullcontext() if private_bound_gsim else _engine_hang_bound(backend, engine, max_cycles):
        revalidate()
        started = time.monotonic()
        bounded = _selected_run_timeout(backend, engine, timeout_s, deadline)
        if engine == "spike" and declared_machine:
            run_declared = getattr(backend, "run_declared_model_elf", None)
            if not callable(run_declared):
                raise ExecutionGateError("selected backend cannot run the linked model's declared machine")
            console = run_declared(elf, timeout=bounded, capture_bytes=capture_bytes)
        else:
            options = {"private_max_cycles": max_cycles} if private_bound_gsim else {}
            console = backend.run_elf(elf, simulator=engine, timeout=bounded, capture_bytes=capture_bytes, **options)
        wall = time.monotonic() - started
        if command is not None:
            command.revalidate()
        revalidate()
    if isinstance(console, bytes) and not capture_bytes:
        console = console.decode("utf-8", errors="replace")
    return {**evidence, "console": console, "wall_s": round(wall, 1)}


def _capture_file(capture: Path, name: str) -> tuple[Path, bool] | None:
    """``(path, bound)`` for a capture file, ``None`` when absent; ``bound`` when the capture receipt
    records its digest. A file the receipt records under different bytes is refused."""
    import json

    from merlin.runtime.execution.model_inputs import file_sha256

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

    from merlin.runtime.execution.model_inputs import file_sha256

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


def host_execution_outputs(capture: Path, workdir: Path, *, receipt: Mapping[str, Any] | None = None) -> list:
    """Every result of the saved program executed on the host, the independent reference for results
    the capture binds no reference for."""
    import numpy as np

    from merlin.runtime.dispatch.whole_model import run_model

    options: dict[str, Any] = {}
    if receipt is not None:
        from merlin.compiler.backend.host import load_host_package as load_rvv_package
        from merlin.runtime.execution import model_inputs

        inputs = receipt.get("inputs") or {}
        package_path = Path(str(inputs.get("package") or ""))
        expected_tree = inputs.get("package_tree")
        if not package_path.is_dir() or model_inputs.strict_tree_sha256(package_path) != expected_tree:
            raise ExecutionGateError("the host package changed since the linked program was built")
        package = load_rvv_package(package_path)
        if package.is_int8:
            options["int8_compute"] = True
            if inputs.get("respect_captured_quantization_scope"):
                from merlin.compiler.lower.options.quantization_scope import captured_quantized_contraction

                options["quant_passes"] = ["contraction_int8"]
                options["quant_select"] = captured_quantized_contraction
        elif inputs.get("respect_captured_quantization_scope"):
            raise ExecutionGateError("captured quantization scope requires an INT8 host package")
    result = run_model(capture, workdir, **options)
    return [np.asarray(value) for value in (result.get("outputs") or [result["output"]])]


def _host_outputs_in_scratch(capture: Path, *, receipt: Mapping[str, Any] | None = None) -> list:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="merlin-host-reference-") as scratch:
        return host_execution_outputs(capture, Path(scratch), receipt=receipt)
