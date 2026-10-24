"""Resumable bounded adapter for exported full-contract boundary observations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from merlin.targetgen.core_aten_capture import loader_source


def atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(document, dict) and any(
        isinstance(v, Mapping) and not isinstance(v, dict) for v in document.values()
    ):
        # A complete response/report can exceed memory. Keep its ordinary JSON
        # protocol while materializing at most one independently graded case.
        with temporary.open("w") as stream:
            stream.write("{\n")
            for index, (key, value) in enumerate(sorted(document.items())):
                if index:
                    stream.write(",\n")
                stream.write(json.dumps(key) + ": ")
                if isinstance(value, Mapping) and not isinstance(value, dict):
                    stream.write("{\n")
                    for position, identifier in enumerate(value):
                        if position:
                            stream.write(",\n")
                        stream.write(json.dumps(identifier) + ": ")
                        json.dump(value[identifier], stream, sort_keys=True)
                    stream.write("\n}")
                else:
                    json.dump(value, stream, sort_keys=True)
            stream.write("\n}\n")
    else:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class JsonRecordMapping(MutableMapping):
    """Read/write complete per-case JSON records without retaining their payloads."""

    def __init__(self, directory, paths=None, *, field=None, missing_value=None):
        self.directory = Path(directory)
        self.paths = dict(paths or {})
        self.field = field
        self.missing_value = missing_value

    def __getitem__(self, key):
        path = self.paths[key]
        if not path.exists() and self.missing_value is not None:
            return copy.deepcopy(self.missing_value)
        value = json.loads(path.read_text())
        return value[self.field] if self.field else value

    def __setitem__(self, key, value):
        if self.field:
            raise TypeError("a projected record mapping is read-only")
        path = self.directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        atomic_json(path, value)
        self.paths[key] = path

    def __delitem__(self, key):
        if self.field:
            raise TypeError("a projected record mapping is read-only")
        self.paths.pop(key).unlink()

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)


def case_digest(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def bundle_admission_reason(capture: dict, mlir_source: str | None = None) -> str | None:
    """Expose the native batch admission error before its empty-bundle wrapper."""
    from merlin.targetgen.core_aten_batch import _function_parts, _tensor_type

    meta = capture.get("capture_meta", {})
    try:
        for abi in [*meta.get("input_abi", []), *meta.get("output_abi", [])]:
            _tensor_type(abi)
        if mlir_source is not None:
            names, _, returned, _, _ = _function_parts(mlir_source)
            if len(names) != len(meta.get("input_abi", [])):
                raise ValueError("captured argument ABI or input count mismatch")
            contract = meta.get("result_contract")
            result_abi = contract["results"] if contract is not None else meta.get("output_abi", [])
            if len(returned) != len(result_abi):
                raise ValueError("captured result ABI count mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def bounded_loader_source(case: dict) -> str:
    """Retain recorded input layouts and seed in the standalone capture loader."""
    return loader_source(case, rng_seed=int(case["rng_seed"]))


def semantic_observation(verdict: dict, evidence: dict) -> dict:
    status = verdict["status"]
    reason = verdict.get("reason") or json.dumps(verdict.get("results", []), sort_keys=True)
    if verdict.get("semantic_scope") in ("full", "values+mutation+observed-metadata") and status == "pass":
        return {
            **verdict["observation"],
            "semantic_scope": verdict["semantic_scope"],
            "imposed_by_harness": verdict.get("imposed_by_harness", []),
            "evidence": {**verdict["observation"].get("evidence", {}), **evidence},
        }
    if status == "pass":
        status = "ungradable"
        reason = (
            "numeric output passes; byte readback cannot observe post-arguments, mutations, aliases, "
            "or full output metadata"
        )
    return {"status": status, "reason": reason, "evidence": evidence}


def grade_portable_outputs(batch: dict, output_bytes: list[bytes], *, output_shapes=None) -> dict:
    """Decode portable nonfinite leaves, then use the unmodified byte comparator.

    The sealed case and its digest stay untouched. Only the transient numeric
    oracle representation is decoded, exactly as the portable value codec does.
    """
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    def decode(value):
        if isinstance(value, dict):
            if value.get("kind") == "float":
                return float(value["value"])
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    transient = copy.deepcopy(batch)
    for record in transient["cases"]:
        record.pop("semantic_boundary", None)
        record["case"]["expected"] = decode(record["case"]["expected"])
    return grade_core_aten_batch(transient, output_bytes, output_shapes=output_shapes)


class SavedResponseAdapter:
    """The same response document accepted by JsonCommandSuiteAdapter."""

    name = "bounded-spike-semantic-adapter"

    def __init__(self, response: dict):
        self.response = response

    def execute(self, corpus):
        return self.response["results"]


def failure_family(status: str, reason: str, overload: str) -> tuple[str, str, str]:
    """Merge varying operands/locations while retaining distinct compiler boundaries.

    Ownership and causes are diagnostic hypotheses, never compiler dispatch policy.
    """
    if "byte readback cannot observe" in reason:
        return "legacy numeric-only boundary", "Merlin", "Baseline runner did not measure the full contract."
    if "unsupported batch ABI dtype" in reason or "unsupported ABI dtype" in reason:
        dtype = reason.rsplit("dtype", 1)[-1].strip()
        return f"unsupported capture/batch ABI: {dtype}", "Merlin", "Capture or native bundle ABI lacks this dtype."
    if "non-clean program" in reason:
        if "Opaque ops:" in reason and "(none reported)" not in reason:
            ops = reason.partition("Opaque ops:")[2].strip()
            names = sorted({part.partition("=")[0] for part in ops.split()})
            return "opaque capture: " + ", ".join(names), "m2m", "No clean lowering for the captured operation/dtype."
        return "capture rejected without opaque ops", "m2m", "Conversion validation rejects the emitted program."
    if "undefined reference" in reason:
        symbols = set()
        for line in reason.splitlines():
            if "undefined reference to" in line:
                symbols.add(line.partition("undefined reference to")[2].strip().strip("`'"))
        return (
            "missing runtime symbols: " + ", ".join(sorted(symbols)),
            "Merlin",
            "Bare-metal runtime does not link emitted helper calls.",
        )
    if "returncode=-11" in reason:
        return "upstream lowering segmentation fault", "Merlin", "Upstream MLIR lowering crashes on the captured IR."
    if "undefined global" in reason:
        return "missing captured global", "Merlin", "Batch extraction omits an MLIR global required by the function."
    if "captured result ABI count mismatch" in reason or "result roles differ" in reason:
        return (
            "capture/result-role ABI mismatch",
            "Merlin",
            "Emitted results do not agree with the captured result contract.",
        )
    if "operand is used with type" in reason:
        return (
            "inconsistent SSA tensor types",
            "m2m",
            "Capture emits an SSA use whose type differs from its definition.",
        )
    if "metadata mismatch" in reason:
        boundary = "post-argument" if "post" in reason else "output"
        return f"{boundary} metadata", "Merlin", "Compiled boundary layout/shape/dtype/grad state differs from eager."
    if "output_input_aliases mismatch" in reason:
        return "output/input storage alias", "Merlin", "Compiled boundary storage identity differs from eager."
    if "mutated_arguments mismatch" in reason:
        return (
            "argument mutation classification",
            "Merlin",
            "Compiled post-call bytes or metadata expose a different mutation set.",
        )
    if status == "mismatch":
        operation = overload.split(".")[1] if "." in overload else overload
        return (
            "numeric semantics: " + operation,
            "m2m",
            "Captured operation arithmetic/order/RNG differs from eager; owner is provisional.",
        )
    lines = [line.strip() for line in reason.splitlines() if line.strip()]
    diagnostic = lines[-1] if lines else "no diagnostic"
    if status == "compile_lowering_failed":
        return (
            "native lowering: " + diagnostic[:160],
            "Merlin",
            "Native lowering cannot legalize the captured operation.",
        )
    if status == "capture_unavailable":
        return (
            "capture: " + diagnostic[:160],
            "m2m",
            "Capture/conversion fails before producing clean Linalg; owner is provisional.",
        )
    return (
        status + ": " + diagnostic[:160],
        "Merlin",
        "Runner/bundle/readback boundary refuses the case; inspect retained diagnostic.",
    )
