"""Array oracle for an explicitly selected, shared-source binary32 libm policy.

Only explicit portable policies select this adapter. Native remains NumPy/platform
owned and returns no adapter, without building or writing anything. The portable
oracle and generated runtime compile the same reviewed C implementation; this is
arithmetic correspondence, not model, device or performance qualification.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from merlin.runtime.host_math import host_math_recipe

_SCHEMA = "merlin.host_math.reference.v1"
_POLICIES = ("portable_exp_trig", "portable_neural_f32")
_FLAGS = ("-std=c11", "-O2", "-fPIC")
_OPERATIONS = {"math.exp": 0, "math.sin": 1, "math.cos": 2}
_NEURAL_OPERATIONS = {**_OPERATIONS, "math.tanh": 3, "math.powf": 4}
_WRAPPER = r"""
#include <fenv.h>
#include <stddef.h>
/* The scalar bodies above are identical to the generated runtime recipe. */
int merlin_reference_math_array(int op, const float *input, float *output, size_t count) {
  if (fegetround() != FE_TONEAREST) return -2;
  if (op < 0 || op > 2) return -1;
  for (size_t i = 0; i < count; ++i) {
    if (op == 0) output[i] = __wrap_expf(input[i]);
    else if (op == 1) output[i] = __wrap_sinf(input[i]);
    else output[i] = __wrap_cosf(input[i]);
  }
  return 0;
}
"""
_NEURAL_WRAPPER = (
    _WRAPPER.replace("op > 2", "op > 3").replace(
        "else output[i] = __wrap_cosf(input[i]);",
        "else if (op == 2) output[i] = __wrap_cosf(input[i]);\n    else output[i] = __wrap_tanhf(input[i]);",
    )
    + r"""
int merlin_reference_math_binary(const float *left, const float *right,
                                float *output, size_t count) {
  if (fegetround() != FE_TONEAREST) return -2;
  for (size_t i = 0; i < count; ++i) output[i] = __wrap_powf(left[i], right[i]);
  return 0;
}
"""
)
_ACTIVE: ContextVar[HostMathEvaluator | None] = ContextVar("merlin_host_math_reference", default=None)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class HostMathEvaluator:
    """One byte-bound compiled oracle, with one C call per entire input array."""

    def __init__(self, directory: Path, receipt: dict):
        self._directory = directory
        self._receipt = receipt
        self.policy = receipt["inputs"]["policy"]
        self._verify()
        self._library = ctypes.CDLL(str(directory / "host_math_reference.so"))
        self._function = self._library.merlin_reference_math_array
        pointer = ctypes.POINTER(ctypes.c_float)
        self._function.argtypes = [ctypes.c_int, pointer, pointer, ctypes.c_size_t]
        self._function.restype = ctypes.c_int
        self._binary = None
        if self.policy == "portable_neural_f32":
            self._binary = self._library.merlin_reference_math_binary
            self._binary.argtypes = [pointer, pointer, pointer, ctypes.c_size_t]
            self._binary.restype = ctypes.c_int
        self._verify()

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(_NEURAL_OPERATIONS if self.policy == "portable_neural_f32" else _OPERATIONS)

    def _verify(self) -> None:
        for name, expected in self._receipt["files"].items():
            path = self._directory / name
            if not path.is_file() or _sha(path) != expected:
                raise RuntimeError(f"portable host-math artifact changed: {name}")
        recipe = host_math_recipe(self.policy)
        if hashlib.sha256(recipe.source.encode()).hexdigest() != self._receipt["inputs"]["scalar_source_sha256"]:
            raise RuntimeError("portable host-math source changed")
        if _sha(Path(__file__)) != self._receipt["inputs"]["adapter_sha256"]:
            raise RuntimeError("portable host-math reference adapter changed")

    def identity(self) -> dict:
        """Direct source/compiler/object/library identity, independent of work pathname."""
        self._verify()
        selected = self._receipt["inputs"]
        compiler = Path(selected["compiler"]["path"])
        if _sha(compiler) != selected["compiler"]["sha256"]:
            raise RuntimeError("portable host-math compiler changed")
        return {
            "schema": _SCHEMA,
            "policy": self.policy,
            "operations": list(self.operations),
            "element_type": "f32",
            "rounding": "round_to_nearest_ties_to_even",
            "scope": "returned values; errno and floating exception flags unclaimed",
            **json.loads(json.dumps(selected)),
            "object_sha256": self._receipt["files"]["host_math_reference.o"],
            "library_sha256": self._receipt["files"]["host_math_reference.so"],
        }

    def evaluate(self, op_name: str, array: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
        if op_name not in self.operations:
            raise ValueError(f"portable host-math does not select {op_name!r}")
        value = np.asarray(array)
        if value.dtype != np.dtype(np.float32):
            raise ValueError("portable host-math requires native binary32 input")
        right = None
        if op_name == "math.powf":
            if other is None:
                raise ValueError("portable powf needs two binary32 operands")
            right = np.asarray(other)
            if right.dtype != np.dtype(np.float32):
                raise ValueError("portable host-math requires native binary32 input")
            value, right = np.broadcast_arrays(value, right)
        elif other is not None:
            raise ValueError("unary portable host-math takes one operand")
        self._verify()
        contiguous = np.ascontiguousarray(value)
        output = np.empty(contiguous.shape, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        if right is not None:
            rhs = np.ascontiguousarray(right)
            status = self._binary(
                contiguous.ctypes.data_as(pointer),
                rhs.ctypes.data_as(pointer),
                output.ctypes.data_as(pointer),
                contiguous.size,
            )
        else:
            operations = _NEURAL_OPERATIONS if self.policy == "portable_neural_f32" else _OPERATIONS
            status = self._function(
                operations[op_name], contiguous.ctypes.data_as(pointer), output.ctypes.data_as(pointer), contiguous.size
            )
        if status == -2:
            raise RuntimeError("portable host-math requires round-to-nearest ties-to-even")
        if status != 0:
            raise RuntimeError(f"portable host-math array evaluator refused: {status}")
        self._verify()
        return output.reshape(value.shape)


def reference_math(policy: str, work: Path, compiler: str | Path = "cc") -> HostMathEvaluator | None:
    """Build/reuse the explicit portable policy; native performs no I/O or tool resolution.

    Cache receipts bind actual source, selected compiler and every produced byte.
    The direct compiler driver and products are recorded, without claiming a
    hermetic closure of the operating system or dynamic loader.
    """
    if policy == "native":
        return None
    if policy not in _POLICIES:
        raise ValueError("reference host_math_policy must be 'native', 'portable_exp_trig', or 'portable_neural_f32'")
    recipe = host_math_recipe(policy)
    resolved = Path(shutil.which(str(compiler)) or compiler).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("reference host-math compiler must be a regular executable")
    scalar = recipe.source
    assert scalar is not None
    wrapper = _NEURAL_WRAPPER if policy == "portable_neural_f32" else _WRAPPER
    source = scalar + "\n" + wrapper
    inputs = {
        "policy": policy,
        "scalar_source_sha256": hashlib.sha256(scalar.encode()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(wrapper.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapter_sha256": _sha(Path(__file__)),
        "compiler": {"path": str(resolved), "sha256": _sha(resolved)},
        "compile_flags": [*_FLAGS, *recipe.compile_flags],
        "link_flags": ["-shared", "-Wl,-z,defs", "-lm"],
    }
    directory = Path(work).resolve() / policy / _digest(inputs)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = directory / "receipt.json"
        try:
            receipt = json.loads(record.read_text())
            if receipt.get("schema") == _SCHEMA and receipt.get("inputs") == inputs:
                return HostMathEvaluator(directory, receipt)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            pass  # Missing, legacy or byte-changed products require an observed fresh build.
        (directory / "host_math_reference.c").write_text(source)
        commands = [
            [
                str(resolved),
                *inputs["compile_flags"],
                "-c",
                "host_math_reference.c",
                "-o",
                "host_math_reference.o.pending",
            ],
            [
                str(resolved),
                "-shared",
                "-Wl,-z,defs",
                "host_math_reference.o.pending",
                "-lm",
                "-o",
                "host_math_reference.so.pending",
            ],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(
                    "portable host-math compiler failed: " + result.stderr.decode(errors="replace")[-2000:]
                )
        # Never overwrite an inode whose code another evaluator may have mapped.
        for name in ("host_math_reference.o", "host_math_reference.so"):
            (directory / (name + ".pending")).replace(directory / name)
        receipt = {
            "schema": _SCHEMA,
            "inputs": inputs,
            "commands": commands,
            "files": {
                name: _sha(directory / name)
                for name in ("host_math_reference.c", "host_math_reference.o", "host_math_reference.so")
            },
        }
        if inputs["compiler"]["sha256"] != _sha(resolved) or source != host_math_recipe(policy).source + "\n" + wrapper:
            raise RuntimeError("portable host-math producer inputs changed during compilation")
        pending = directory / "receipt.pending"
        pending.write_text(json.dumps(receipt, indent=2) + "\n")
        pending.replace(record)
        return HostMathEvaluator(directory, receipt)


@contextmanager
def host_math_context(evaluator: HostMathEvaluator | None) -> Iterator[None]:
    """Select one task-local arithmetic policy, restoring it even after refusal."""
    if evaluator is not None and not isinstance(evaluator, HostMathEvaluator):
        raise TypeError("host math context needs a reference_math evaluator or None")
    token = _ACTIVE.set(evaluator)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_host_math() -> HostMathEvaluator | None:
    return _ACTIVE.get()
