"""Resolve a source-bound operand to an immutable declared model parameter.

This path only reads the compiler-only staging directory. Runtime samples and
reference outputs are not inputs. A view is replayed from parsed IR; unknown
producers remain runtime values rather than invented compile-time constants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import TensorType as XdslTensorType

from merlin.frontends.compile_inputs import load_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text

from .precision_intake import SourceValueRef
from .precision_transform import DiagnosticMatmulTile


@dataclass(frozen=True, eq=False)
class DeclaredStaticTensor:
    role: str
    argument_index: int
    weight_key: str
    views: tuple[tuple[int, ...], ...]
    value: np.ndarray
    weights_sha256: str
    tensor_sha256: str

    def record(self) -> dict:
        return {
            "role": self.role,
            "argument_index": self.argument_index,
            "weight_key": self.weight_key,
            "views": [list(view) for view in self.views],
            "shape": list(self.value.shape),
            "dtype": "f32",
            "weights_sha256": self.weights_sha256,
            "tensor_sha256": self.tensor_sha256,
        }


@dataclass(frozen=True)
class PreparedStaticInput:
    """One target-packed parameter or a concrete numerical admission blocker."""

    role: str
    address: int
    source: DeclaredStaticTensor
    payload: bytes | None
    blocker: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.role.isidentifier() or type(self.address) is not int or self.address < 0
            or not isinstance(self.source, DeclaredStaticTensor)
            or self.source.role != self.role
            or (self.payload is None) == (self.blocker is None)
            or self.payload is not None and (not isinstance(self.payload, bytes) or not self.payload)
            or self.blocker is not None and (not isinstance(self.blocker, str) or not self.blocker)
        ):
            raise ValueError("prepared static input needs exactly one payload or blocker")

    def record(self) -> dict:
        row = {
            "role": self.role,
            "address": self.address,
            "source": self.source.record(),
        }
        if self.payload is not None:
            return {
                **row, "status": "packed", "path": f"constants/{self.role}.bin",
                "bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest(),
            }
        return {**row, "status": "blocked", "reason": self.blocker}


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("weight manifest has duplicate JSON fields")
    return result


def _f32_shape(value) -> tuple[int, ...]:
    type_ = value.type
    if not isinstance(type_, XdslTensorType) or not type_.has_static_shape() or str(type_.element_type) != "f32":
        raise ValueError("static weight trace requires a static f32 tensor")
    shape = tuple(int(size) for size in type_.get_shape())
    if len(shape) < 2 or any(size <= 0 for size in shape):
        raise ValueError("static weight trace requires a positive tensor of rank at least two")
    return shape


def _trace_argument(
    program: str, entry: str, source_ref: SourceValueRef,
) -> tuple[int, tuple[tuple[int, ...], ...]] | None:
    module = parse_mlir_text(program)
    module.verify()
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("declared weight trace needs one parsed entry block")
    block = functions[0].body.block
    operations = tuple(block.ops)
    if source_ref.kind == "argument":
        if not 0 <= source_ref.index < len(block.args) or source_ref.result_index != 0:
            raise ValueError("source argument reference is out of range")
        value = block.args[source_ref.index]
    elif source_ref.kind == "operation_result":
        if not 0 <= source_ref.index < len(operations):
            raise ValueError("source operation reference is out of range")
        results = operations[source_ref.index].results
        if not 0 <= source_ref.result_index < len(results):
            raise ValueError("source result reference is out of range")
        value = results[source_ref.result_index]
    else:
        raise ValueError("source reference has an unknown kind")
    views: list[tuple[int, ...]] = []
    for _ in range(len(operations) + 1):
        _f32_shape(value)
        if value.owner is block:
            return tuple(block.args).index(value), tuple(reversed(views))
        owner = value.owner
        if owner.name != "linalg.transpose" or len(owner.inputs) != 1 or len(owner.results) != 1:
            return None
        permutation = owner.properties.get("permutation")
        axes = tuple(permutation.iter_values()) if permutation is not None else ()
        if sorted(axes) != list(range(len(_f32_shape(value)))) or _f32_shape(value) != tuple(
            _f32_shape(owner.inputs[0])[axis] for axis in axes
        ):
            raise ValueError("source transpose has an invalid shape or permutation")
        views.append(axes)
        value = owner.inputs[0]
    raise ValueError("source view trace did not terminate")


def load_declared_static_operand(
    staged_directory: str | Path, plan: DiagnosticMatmulTile, role: str,
) -> DeclaredStaticTensor | None:
    """Load a declared parameter for one tile operand, or None for a runtime producer.

    The returned array has the *source operand* shape. Target conversion and
    physical packing are separate steps owned by the selected target binding.
    """
    if not isinstance(plan, DiagnosticMatmulTile) or role not in {
        item.source_role for item in plan.inputs
    }:
        raise ValueError("declared operand needs a source-bound tile input role")
    staged = load_compile_inputs(staged_directory)
    if (
        staged.program_sha256 != plan.staged_program_sha256
        or staged.original_source_sha256 != plan.original_source_sha256
        or staged.precision_policy_sha256 != plan.precision_policy_sha256
    ):
        raise ValueError("declared weight stage differs from the source-bound tile")
    conversion = next(item for item in plan.inputs if item.source_role == role)
    if conversion.source_role != role:
        raise ValueError("source-bound tile operand order differs")
    traced = _trace_argument(staged.program, staged.entry, conversion.source_value)
    if traced is None:
        return None
    index, views = traced
    root = Path(staged_directory).absolute()
    manifest = json.loads((root / "weights.safetensors.manifest.json").read_bytes(), object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict) or str(index) not in manifest:
        raise ValueError("source argument has no declared weight or runtime kind")
    entry = manifest[str(index)]
    if not isinstance(entry, dict):
        raise ValueError("weight manifest argument entry is malformed")
    if entry.get("kind") == "input":
        return None
    if entry.get("kind") != "param":
        raise ValueError("unsupported static argument kind")
    key = entry.get("weight")
    shape = entry.get("shape")
    if (
        not isinstance(key, str) or not key
        or entry.get("dtype") not in {"float32", "f32"}
        or not isinstance(shape, list) or len(shape) < 2
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError("declared parameter has an invalid key, dtype or shape")
    weights_path = root / "weights.safetensors"
    stage_manifest = json.loads((root / "compile-inputs.json").read_bytes(), object_pairs_hook=_unique_object)
    expected_sha = stage_manifest["members"]["weights.safetensors"]
    from safetensors import safe_open

    with safe_open(weights_path, framework="np", device="cpu") as weights:
        if key not in weights.keys():
            raise ValueError("declared parameter key is absent from staged weights")
        tensor = weights.get_tensor(key)
    if tensor.dtype != np.float32 or list(tensor.shape) != entry.get("shape"):
        raise ValueError("staged parameter differs from its declared dtype or shape")
    with weights_path.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("staged weights changed while resolving a parameter")
    value = np.array(tensor, copy=True)
    for axes in views:
        value = np.transpose(value, axes)
    value = np.ascontiguousarray(value)
    if value.shape != conversion.source_type.shape:
        raise ValueError("declared parameter views differ from source operand shape")
    value.setflags(write=False)
    return DeclaredStaticTensor(
        role, index, key, views, value, actual_sha, hashlib.sha256(value.tobytes()).hexdigest(),
    )
