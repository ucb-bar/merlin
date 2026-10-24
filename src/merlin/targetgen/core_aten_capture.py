"""model2MLIR adapter for the compiler-independent Core ATen case corpus.

The source corpus is authoritative and remains complete when this adapter fails.  Each adapter result
is a separate observation: exact provenance retained, exact overload lost, or capture failed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from merlin.targetgen.capsule_source import PytorchRefSource
from merlin.targetgen.capture_refusals import InputStorageUnavailable
from merlin.targetgen.core_aten_cover import _provenance_ops
from merlin.targetgen.core_aten_input_storage import (
    apply_native_input_storage,
    native_argument_projection,
    native_input_storage_plan,
)


def _split_tensor_documents(value: Any, tensors: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict) and value.get("kind") == "tensor":
        index = len(tensors)
        tensors.append(value)
        return {"input_index": index}
    if isinstance(value, dict):
        return {key: _split_tensor_documents(child, tensors) for key, child in value.items()}
    if isinstance(value, list):
        return [_split_tensor_documents(child, tensors) for child in value]
    return value


def loader_source(case: dict[str, Any], *, rng_seed: int | None = None) -> str:
    """Render a self-contained PyTorch module for one portable case document.

    Every tensor argument, including tensors nested in lists and keyword ``out`` values, is an
    external model input.  Scalars, dtype objects, memory formats, and container structure stay as
    static schema arguments.  The generated loader imports neither Merlin nor model2MLIR.
    """

    storage_plan = native_input_storage_plan(case["overload"], case["arguments"])
    seed_statement = "" if rng_seed is None else f"torch.manual_seed({int(rng_seed)})"
    tensors: list[dict[str, Any]] = []
    arguments = _split_tensor_documents(case["arguments"], tensors)
    if storage_plan is not None:
        tensors = [dict(tensor) for tensor in tensors]
        for read in storage_plan["raw_reads"]:
            tensors[read["input_index"]]["minimum_storage_elements"] = read["minimum_storage_elements"]
    argument_projection = native_argument_projection(storage_plan, tensors)
    parameters = ", ".join(f"input_{index}" for index in range(len(tensors)))
    inputs = ", ".join(f"input_{index}" for index in range(len(tensors)))
    if len(tensors) == 1:
        inputs += ","
    expected = case["expected"]
    structured_result = isinstance(expected, list) or (
        isinstance(expected, dict) and expected.get("kind") in {"tensor", "tuple"}
    )
    scalar_result = not structured_result
    scalar_wrap = "result = torch.ops.aten.scalar_tensor.default(result)" if scalar_result else ""
    if isinstance(expected, dict) and expected.get("kind") == "complex":
        # scalar_tensor defaults to float32 even for a Python complex value.
        # Preserve the source component precision in this tensor ABI wrapper.
        dtypes = {tensor["dtype"] for tensor in tensors if tensor["dtype"] in {"complex64", "complex128"}}
        if len(dtypes) != 1:
            raise ValueError("complex scalar capture requires unambiguous source component precision")
        scalar_wrap = f"result = torch.ops.aten.scalar_tensor.default(result, dtype=torch.{dtypes.pop()})"
    document = json.dumps(
        {"overload": case["overload"], "arguments": arguments, "tensor_inputs": tensors,
         "input_storage_convention": storage_plan},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f'''"""Generated from the versioned Core ATen case corpus; do not hand-edit."""
import itertools
import json
import torch
from torch import nn

_DOCUMENT = json.loads({document!r})


def _decode(value, inputs=()):
    if isinstance(value, list):
        return [_decode(item, inputs) for item in value]
    if not isinstance(value, dict):
        return value
    if "input_index" in value:
        return inputs[value["input_index"]]
    kind = value.get("kind")
    if kind == "tuple":
        return tuple(_decode(item, inputs) for item in value["items"])
    if kind == "dtype" or kind == "memory_format":
        return getattr(torch, value["value"])
    if kind == "device":
        return torch.device(value["value"])
    if kind == "complex":
        return complex(_decode(value["real"]), _decode(value["imag"]))
    if kind == "float":
        return float(value["value"])
    if kind == "tensor":
        dtype = getattr(torch, value["dtype"])
        logical = torch.tensor(_decode(value["values"]), dtype=dtype).reshape(value["shape"])
        shape, stride = value["shape"], value["stride"]
        offset = value["storage_offset"]
        span = offset + (1 + sum((n - 1) * s for n, s in zip(shape, stride)) if all(shape) else 0)
        storage = torch.zeros(max(span, value.get("minimum_storage_elements", 0)), dtype=dtype)
        tensor = storage.as_strided(shape, stride, offset)
        if torch._debug_has_internal_overlap(tensor) == 0:
            tensor.copy_(logical)
        else:
            # Expanded inputs require scalar writes. Repeated physical addresses
            # must carry identical recorded bytes, including signed zero and NaN.
            seen = {{}}
            for index in itertools.product(*(range(extent) for extent in shape)):
                address = offset + sum(i * step for i, step in zip(index, stride))
                scalar = logical[index]
                if address in seen:
                    previous = seen[address].contiguous().reshape(-1).view(torch.uint8)
                    current = scalar.contiguous().reshape(-1).view(torch.uint8)
                    if not torch.equal(previous, current):
                        raise ValueError("recorded logical values are inconsistent with overlapping strides")
                seen[address] = scalar
                storage[address] = scalar
        return tensor.requires_grad_(value["requires_grad"])
    return {{key: _decode(child, inputs) for key, child in value.items()}}


def _op(name):
    namespace, packet, overload = name.split(".", 2)
    assert namespace == "aten"
    return getattr(getattr(torch.ops.aten, packet), overload)


# A program with results, all of them int/bool, may be a descriptor query;
# the capture worker folds it only if it evaluates on data-free meta tensors.
_RETURNS = _op(_DOCUMENT["overload"])._schema.returns
STATIC_METADATA_ONLY = bool(_RETURNS) and all(
    isinstance(result.type, (torch._C.IntType, torch._C.BoolType, torch._C.SymIntType, torch._C.SymBoolType))
    for result in _RETURNS
)


class Model(nn.Module):
    def forward(self, {parameters}):
        inputs = ({inputs})
        {seed_statement}
        arguments = _decode(_DOCUMENT["arguments"], inputs)
        {argument_projection}
        operation = _op(_DOCUMENT["overload"])
        result = operation(*arguments["args"], **arguments["kwargs"])
        {scalar_wrap}
        return result


def get_model_and_inputs():
    inputs = tuple(_decode(value) for value in _DOCUMENT["tensor_inputs"])
    return Model(), inputs
'''


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return process.stdout.strip() if process.returncode == 0 and process.stdout.strip() else None


def case_capture_name(case: dict[str, Any]) -> str:
    """Stable filesystem key; additive variants must never overwrite another capture."""
    import hashlib

    identifier = case.get("case_id")
    if identifier:
        return "case-" + hashlib.sha256(str(identifier).encode()).hexdigest()
    return str(case["overload"]).replace(".", "__")


def descriptor_specialization_matches(case: dict, capture: dict) -> bool:
    """Recognize a folded query only under its recorded static input descriptors.

    This is capture authority, not fabricated operator provenance on the constant:
    the frontend recorded that the program's result is a fold of the inputs'
    static descriptors. No source overload name is consulted here, and tensor
    values and the case's expected result are never read.
    """
    from merlin.targetgen.core_aten_batch import _tensor_documents

    proof = capture.get("capture_meta", {}).get("static_metadata_specialization") or {}
    descriptors = [
        {key: tensor[key] for key in ("shape", "stride", "storage_offset")}
        for tensor in _tensor_documents(case["arguments"])
    ]
    return (
        proof.get("authority") == "static_capture_input_descriptors"
        and proof.get("inputs") == descriptors
        and bool(proof.get("queries"))
    )


def capture_case_corpus(
    corpus: dict[str, Any],
    output_root: str | Path,
    *,
    overloads: set[str] | None = None,
    m2m_dir: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    """Capture canonical cases independently, retaining failures as auditable result records."""

    identifiers = [str(case.get("case_id") or case["overload"]) for case in corpus["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate case identifiers in capture corpus")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source = PytorchRefSource(
        m2m_dir=Path(m2m_dir) if m2m_dir else None,
        python=Path(python) if python else None,
    )
    results: dict[str, dict[str, Any]] = {}
    for case in corpus["cases"]:
        overload = case["overload"]
        if overloads is not None and overload not in overloads:
            continue
        identifier = str(case.get("case_id") or overload)
        destination = root / case_capture_name(case)
        destination.mkdir(parents=True, exist_ok=True)
        loader = destination / "capsule.pytorch.py"
        mlir = destination / "capsule.linalg.mlir"
        try:
            loader.write_text(loader_source(case), encoding="utf-8")
            artifact = source.capture_loader(loader, "fp32", workdir=destination / "capture-work")
            mlir.write_text(apply_native_input_storage(artifact.linalg_mlir, case), encoding="utf-8")
            actual = sorted(_provenance_ops(mlir))
            specialization = artifact.meta.get("static_metadata_specialization") or {}
            metadata_exact = (
                specialization.get("authority") == "static_capture_input_descriptors"
                and any(query.get("operator") == overload for query in specialization.get("queries", []))
            )
            exact = overload in actual or metadata_exact
            record = {
                "schema_version": 1,
                "status": "captured_exact" if exact else "captured_lost_exact_provenance",
                "overload": overload,
                "case_id": identifier,
                "captured_overloads": actual,
                "pytorch_version": artifact.meta.get("torch_version"),
                "model2mlir_revision": _git_revision(source.m2m_dir),
                "capture_meta": artifact.meta,
                "exact_provenance_kind": "static_metadata_specialization" if metadata_exact else "source_operator",
                "input_storage_convention": native_input_storage_plan(overload, case["arguments"]),
            }
            (destination / "inputs.json").write_text(json.dumps(artifact.inputs, indent=2) + "\n", encoding="utf-8")
            (destination / "golden.json").write_text(json.dumps(artifact.golden, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 -- every frontend failure is a result, not an omission
            mlir.unlink(missing_ok=True)
            record = {
                "schema_version": 1,
                "status": "capture_failed",
                "overload": overload,
                "case_id": identifier,
                "captured_overloads": [],
                "pytorch_version": corpus["pytorch_version"],
                "model2mlir_revision": _git_revision(source.m2m_dir),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, InputStorageUnavailable):
                record.update(refusal_code=exc.code, refusal_details=exc.details)
        (destination / "capture.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results[identifier] = record
    counts: dict[str, int] = {}
    for record in results.values():
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "case_count": len(results),
        "status_counts": dict(sorted(counts.items())),
        "results": dict(sorted(results.items())),
    }
