"""Differential fidelity checks of the proved Lean denotation against native upstream MLIR.

This is empirical evidence about the model, not a proof that MLIR implements it. The caller
supplies independently chosen, defined inputs. UB, poison, unsupported float models, unavailable
tools, failed exports, and JIT failures cannot count as matching execution.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify.lean.export import export_file, round_trip_exported


@dataclass(frozen=True)
class ValueType:
    """Observable scalar or static unencoded tensor; shape=None distinguishes a scalar from rank 0."""

    element: str
    shape: tuple[int, ...] | None = None

    @property
    def count(self) -> int:
        return math.prod(self.shape) if self.shape is not None else 1

    def width(self, index_bits: int) -> int:
        return index_bits if self.element == "index" else int(self.element[1:])

    @property
    def mlir(self) -> str:
        if self.shape is None:
            return self.element
        dims = "".join(f"{d}x" for d in self.shape)
        return f"tensor<{dims}{self.element}>"


@dataclass(frozen=True)
class Tools:
    compiler_python: Path
    lean_exe: Path
    lake: Path
    runner_utils: Path | None = None


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} exited {result.returncode}:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
        )
    return result


def _type(ty: dict) -> ValueType:
    shape = None
    if ty["kind"] == "tensor":
        if ty.get("encoding") is not None or any(d is None for d in ty["shape"]):
            raise ValueError("differential execution requires static unencoded tensors")
        shape, ty = tuple(ty["shape"]), ty["element"]
    if ty["kind"] == "int" and 1 <= ty["width"] <= 64:
        element = f"i{ty['width']}"
    elif ty["kind"] == "float" and ty["format"] in {"f32", "f64"}:
        element = ty["format"]
    elif ty["kind"] == "index":
        element = "index"
    else:
        raise ValueError(f"unsupported fidelity value type: {ty}")
    return ValueType(element, shape)


def signature(doc: dict, function: str) -> tuple[list[ValueType], list[ValueType]]:
    for op in doc["module"]["ops"]:
        attrs = {**op["attrs"], **op["props"]}
        if op["name"] == "func.func" and attrs["sym_name"]["value"] == function:
            ty = attrs["function_type"]["value"]
            return [_type(t) for t in ty["inputs"]], [_type(t) for t in ty["results"]]
    raise ValueError(f"function {function!r} missing")


def _samples(samples: list, inputs: list[ValueType], index_bits: int) -> list:
    if not samples:
        raise ValueError("a differential run requires at least one sample")
    normalized = []
    for sample in samples:
        if len(sample) != len(inputs):
            raise ValueError("sample has a different argument count than the MLIR signature")
        args = []
        for values, ty in zip(sample, inputs, strict=True):
            if len(values) != ty.count:
                raise ValueError("sample has a different element count than the MLIR signature")
            bits = []
            for value in values:
                if type(value) is not int:
                    raise ValueError("inputs must be integer bit patterns")
                bits.append(str(value % (1 << ty.width(index_bits))))
            args.append(bits)
        normalized.append(args)
    return normalized


class _Wrapper:
    def __init__(self):
        self.lines: list[str] = []
        self.serial = 0

    def emit(self, expression: str) -> str:
        name = f"%v{self.serial}"
        self.serial += 1
        self.lines.append(f"{name} = {expression}")
        return name

    def index(self, value: int) -> str:
        return self.emit(f"arith.constant {value} : index")

    def unpack(self, word: str, ty: ValueType, index_bits: int) -> str:
        width = ty.width(index_bits)
        if ty.element == "index":
            return self.emit(f"arith.index_castui {word} : i64 to index")
        bits = word if width == 64 else self.emit(f"arith.trunci {word} : i64 to i{width}")
        return self.emit(f"arith.bitcast {bits} : i{width} to {ty.element}") if ty.element[0] == "f" else bits

    def pack(self, value: str, ty: ValueType, index_bits: int) -> str:
        width = ty.width(index_bits)
        if ty.element == "index":
            # Zero extend to 64 so the observation is an unsigned index-width bit pattern.
            return self.emit(f"arith.index_castui {value} : index to i64")
        bits = self.emit(f"arith.bitcast {value} : {ty.element} to i{width}") if ty.element[0] == "f" else value
        return bits if width == 64 else self.emit(f"arith.extui {bits} : i{width} to i64")


def wrapper(inputs: list[ValueType], outputs: list[ValueType], function: str, index_bits: int) -> str:
    """Observe exact result bits via memrefs, avoiding platform-specific scalar/tensor return ABIs."""
    w = _Wrapper()
    in_size, out_size = sum(t.count for t in inputs), sum(t.count for t in outputs)
    args, position = [], 0
    for ty in inputs:
        elements = []
        for _ in range(ty.count):
            at = w.index(position)
            position += 1
            word = w.emit(f"memref.load %inputs[{at}] : memref<{in_size}xi64>")
            elements.append(w.unpack(word, ty, index_bits))
        if ty.shape is None:
            args.append(elements[0])
        elif elements:
            args.append(w.emit(f"tensor.from_elements {', '.join(elements)} : {ty.mlir}"))
        else:
            args.append(w.emit(f"tensor.empty() : {ty.mlir}"))
    in_sig = ", ".join(t.mlir for t in inputs)
    out_sig = ", ".join(t.mlir for t in outputs)
    call = f"func.call @{function}({', '.join(args)}) : ({in_sig}) -> ({out_sig})"
    if outputs:
        w.lines.append(f"%results:{len(outputs)} = {call}")
    else:
        w.lines.append(call)
    position = 0
    for i, ty in enumerate(outputs):
        result = f"%results#{i}"
        coordinates = [None] if ty.shape is None else itertools.product(*(range(d) for d in ty.shape))
        for coordinate in coordinates:
            value = result
            if coordinate is not None:
                indices = ", ".join(w.index(c) for c in coordinate)
                value = w.emit(f"tensor.extract {result}[{indices}] : {ty.mlir}")
            word, at = w.pack(value, ty, index_bits), w.index(position)
            position += 1
            w.lines.append(f"memref.store {word}, %outputs[{at}] : memref<{out_size}xi64>")
    w.lines.append("return")
    return (
        f"func.func private @{function}({in_sig}) -> ({out_sig})\n"
        f"func.func @__merlin_fidelity_probe(%inputs: memref<{in_size}xi64>, "
        f"%outputs: memref<{out_size}xi64>) attributes {{llvm.emit_c_interface}} {{\n" + "\n".join(w.lines) + "\n}"
    )


def evaluate_lean(
    exported: Path, samples: list, work: Path, tools: Tools, *, function: str = "subject", index_bits: int = 64
) -> list:
    work.mkdir(parents=True, exist_ok=True)
    request = work / "lean-request.json"
    request.write_text(
        json.dumps({"export": exported.read_text(), "function": function, "index_bits": index_bits, "samples": samples})
    )
    root = repo_root() / "merlin/lean"
    result = _run(
        [str(tools.lake), "env", "lean", "--run", str(root / "Differential.lean"), str(request.resolve())], cwd=root
    )
    answer = json.loads(result.stdout)
    (work / "lean-results.json").write_text(json.dumps(answer, indent=2))
    return answer


def execute_mlir(
    source: Path,
    samples: list,
    work: Path,
    tools: Tools,
    *,
    inputs: list[ValueType],
    outputs: list[ValueType],
    function: str = "subject",
    index_bits: int = 64,
) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    request = work / "native-request.json"
    request.write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "samples": samples,
                "wrapper": wrapper(inputs, outputs, function, index_bits),
                "index_bits": index_bits,
                "output_words": sum(t.count for t in outputs),
                "lowered": str((work / "llvm.mlir").resolve()),
                "runner_utils": str(tools.runner_utils.resolve()) if tools.runner_utils else None,
            }
        )
    )
    result = _run([str(tools.compiler_python), str(Path(__file__).with_name("mlir_execute.py")), str(request)])
    answer = json.loads(result.stdout)
    (work / "native-results.json").write_text(json.dumps(answer, indent=2))
    if answer["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("native oracle executed another source snapshot")
    return answer


def _nan(bits: int, width: int) -> bool:
    fraction = 23 if width == 32 else 52
    exponent = 8 if width == 32 else 11
    return ((bits >> fraction) & ((1 << exponent) - 1)) == (1 << exponent) - 1 and bits % (1 << fraction) != 0


def assert_equal(
    lean: list, native: list, outputs: list[ValueType], samples: list, *, arithmetic_nans: bool = False
) -> None:
    if len(lean) != len(native) or len(lean) != len(samples):
        raise AssertionError("oracle returned a different number of samples")
    for n, (left, right) in enumerate(zip(lean, native, strict=True)):
        if left["status"] != "ok":
            raise AssertionError(f"sample {n} {samples[n]}: Lean returned {left}; no defined result to compare")
        if len(left["values"]) != len(outputs):
            raise AssertionError("Lean returned a different number of result values")
        position = 0
        for result, ty in zip(left["values"], outputs, strict=True):
            shape = list(ty.shape) if ty.shape is not None else None
            if result["shape"] != shape or len(result["bits"]) != ty.count:
                raise AssertionError("Lean result shape differs from the MLIR signature")
            for element, expected in enumerate(result["bits"]):
                if expected is None:
                    raise AssertionError(f"sample {n}: Lean result contains poison; native UB is not comparable")
                actual = right[position]
                position += 1
                if actual == expected:
                    continue
                # Arithmetic cases can explicitly allow unspecified NaN payloads/signs. Data movement
                # and bitcasts keep the default bitwise comparison, including NaN payloads.
                if (
                    arithmetic_nans
                    and ty.element in {"f32", "f64"}
                    and _nan(int(expected), ty.width(64))
                    and _nan(int(actual), ty.width(64))
                ):
                    continue
                raise AssertionError(
                    f"sample {n} {samples[n]}, {ty.mlir} element {element}: "
                    f"Lean bits={expected}, native MLIR bits={actual}"
                )
        if position != len(right):
            raise AssertionError("native oracle returned a different number of result elements")


def compare(
    source_text: str,
    samples: list,
    work: Path,
    tools: Tools,
    *,
    function: str = "subject",
    index_bits: int = 64,
    arithmetic_nans: bool = False,
) -> dict:
    """Compare every supplied defined sample and retain both source and lowered IR on failure."""
    if index_bits not in {32, 64}:
        raise ValueError("native fidelity tests support 32- or 64-bit index")
    work.mkdir(parents=True, exist_ok=True)
    source, exported = work / "source.mlir", work / "export.json"
    source.write_text(source_text)
    export_file(source, exported, backend="mlir", m2m_python=tools.compiler_python)
    trip = round_trip_exported(
        source, exported, work / "roundtrip", lean_exe=tools.lean_exe, backend="mlir", m2m_python=tools.compiler_python
    )
    if trip.status != "identical":
        raise AssertionError(f"source export did not round trip: {trip.status}: {trip.detail}")
    inputs, outputs = signature(json.loads(exported.read_text()), function)
    samples = _samples(samples, inputs, index_bits)
    lean = evaluate_lean(exported, samples, work, tools, function=function, index_bits=index_bits)
    # Refuse undefined/unsupported model outcomes BEFORE executing native code. The corpus must
    # additionally choose safe inputs independently; the Lean model alone is not a UB oracle.
    for result in lean:
        if result["status"] != "ok" or any(x is None for v in result.get("values", []) for x in v["bits"]):
            raise AssertionError(f"case requires independently defined execution; Lean returned {result}")
    native = execute_mlir(
        source, samples, work, tools, inputs=inputs, outputs=outputs, function=function, index_bits=index_bits
    )
    evidence = {
        "lean": lean,
        "native": native,
        "nan_comparison": "classification" if arithmetic_nans else "bits",
        "lean_driver_sha256": hashlib.sha256((repo_root() / "merlin/lean/Differential.lean").read_bytes()).hexdigest(),
        "spec_lock_sha256": hashlib.sha256((repo_root() / "merlin/lean/SPEC_LOCK").read_bytes()).hexdigest(),
        "semantics_sha256": {
            str(path.relative_to(repo_root() / "merlin/lean")): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((repo_root() / "merlin/lean/MerlinVerify/Sem").rglob("*.lean"))
        },
    }
    (work / "results.json").write_text(json.dumps(evidence, indent=2))
    assert_equal(lean, native["results"], outputs, samples, arithmetic_nans=arithmetic_nans)
    return {
        "samples": len(samples),
        "source_sha256": native["source_sha256"],
        "pipeline": native["pipeline"],
        "mlir_revision": native["mlir_revision"],
        "index_bits": index_bits,
    }
