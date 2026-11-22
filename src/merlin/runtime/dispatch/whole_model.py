"""Exact retained whole-model transform stages and independent structural replay.

The runtime executes the normalized/outlined module; these helpers only archive
and verify the same stages. Their public compatibility imports remain in
``dispatch_runtime`` so existing callers retain one entry point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from merlin.runtime.dispatch.errors import DispatchRuntimeError
from merlin.runtime.execution.model_inputs import _ModelInput, plan_forward_args

# ---- dispatch_transform_audit ------------------------------------------------------------------


def _validate_normalization_recipe(recipe: dict) -> dict:
    if not isinstance(recipe, dict) or set(recipe) != {
        "int8_compute",
        "quant_passes",
        "prequant_gather",
        "selection_policy",
    }:
        raise ValueError("model normalization recipe is malformed")
    if (
        type(recipe["int8_compute"]) is not bool
        or type(recipe["prequant_gather"]) is not bool
        or recipe["selection_policy"] not in {"all", "custom_unreplayable"}
        or (
            recipe["quant_passes"] is not None
            and (
                not isinstance(recipe["quant_passes"], list)
                or any(not isinstance(name, str) or not name for name in recipe["quant_passes"])
            )
        )
        or (recipe["prequant_gather"] and not recipe["int8_compute"])
    ):
        raise ValueError("model normalization recipe has invalid options")
    return dict(recipe)


def record_model_transform_audit(
    source: Path,
    workdir: Path,
    normalized,
    outlined,
    *,
    enabled: bool | str,
    expected_source_sha256: str | None = None,
    normalization_recipe: dict | None = None,
) -> Path | None:
    """Retain exact captured, normalized, and outlined IR for later validation.

    The audit binds bytes and a source-to-output sequence; it does not claim
    value equivalence. Semantic receipts must cite these exact stages and be
    replayed independently before they can make that stronger claim.
    """
    if enabled is False:
        return None
    if enabled == "compact":
        raise ValueError("compact model transform audit keeps hashes but not exact IR, so it cannot be replayed")

    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.ir.lowering_audit import IrAudit

    source_text = source.read_text(encoding="utf-8")
    if (
        expected_source_sha256 is not None
        and hashlib.sha256(source_text.encode("utf-8")).hexdigest() != expected_source_sha256
    ):
        raise ValueError("model source changed between parse and transformation audit")
    with IrAudit(
        workdir,
        enabled=enabled,
        producer="merlin.runtime.dispatch.whole_model.run_model",
        source=__file__,
        sidecars=(source,),
    ) as audit:
        if normalization_recipe is not None:
            audit.accounting(
                "normalization-recipe",
                {
                    "schema": "merlin.model_normalization_recipe.v1",
                    **_validate_normalization_recipe(normalization_recipe),
                },
            )
        audit.stage("captured-model", source_text)
        # Generic MLIR is the replayable serialization: xDSL's custom printer
        # emits operations this permissive parser does not implement.
        audit.stage("normalized-model", to_text(normalized, generic=True))
        audit.stage("outlined-model", to_text(outlined.module, generic=True))
    return audit.directory / "index.json" if audit.directory is not None else None


def qualify_model_transform_audit(index_path: Path) -> dict[str, Any]:
    """Replay the archived normalization and outline when their recipe is available.

    This checks archival integrity, MLIR validity, and deterministic pass output.
    It does not prove that a pass preserves values.
    """
    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.partition.outline import outline_dispatches

    index_path = Path(index_path)
    if index_path.is_symlink() or not index_path.is_file() or index_path.name != "index.json":
        raise ValueError("model transform audit index is absent or indirect")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("producer") != "merlin.runtime.dispatch.whole_model.run_model"
        or index.get("outcome") != "completed"
        or index.get("mode") not in {"exact", "both"}
    ):
        raise ValueError("model transform audit did not complete with exact IR")
    stages = index.get("stages")
    expected = ("captured-model", "normalized-model", "outlined-model")
    if (
        not isinstance(stages, list)
        or tuple(stage.get("name") for stage in stages if isinstance(stage, dict)) != expected
        or len(stages) != 3
    ):
        raise ValueError("model transform audit has no exact three-stage sequence")
    sidecars = index.get("sidecars")
    if (
        not isinstance(sidecars, list)
        or len(sidecars) != 1
        or not isinstance(sidecars[0], dict)
        or sidecars[0].get("sha256") != stages[0].get("sha256")
    ):
        raise ValueError("model transform audit does not bind the captured source bytes")
    contents = []
    for ordinal, stage in enumerate(stages):
        name = stage["name"]
        file = stage.get("file")
        digest = stage.get("sha256")
        size = stage.get("bytes")
        if (
            file != f"{ordinal:03d}-{name}.mlir"
            or stage.get("representation") != "exact-ir"
            or stage.get("format") != "mlir"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(f"model transform audit has malformed {name} stage")
        path = index_path.parent / file
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"model transform audit {name} stage is absent or indirect")
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"model transform audit {name} stage differs from its index")
        contents.append(raw.decode("utf-8"))
    for name, content in zip(expected, contents, strict=True):
        try:
            parse_mlir_text(content).verify()
        except Exception as exc:
            raise ValueError(f"model transform audit {name} stage is not valid MLIR: {exc}") from exc
    normalization_replay = "not_recorded"
    receipts = index.get("accounting_receipts", [])
    if not isinstance(receipts, list):
        raise ValueError("model transform audit receipts are malformed")
    recipe_descriptors = [
        row for row in receipts if isinstance(row, dict) and row.get("name") == "normalization-recipe"
    ]
    if len(recipe_descriptors) > 1:
        raise ValueError("model transform audit has duplicate normalization recipes")
    if recipe_descriptors:
        descriptor = recipe_descriptors[0]
        if (
            descriptor.get("file") != "normalization-recipe.json"
            or descriptor.get("schema") != "merlin.model_normalization_recipe.v1"
            or type(descriptor.get("bytes")) is not int
            or not isinstance(descriptor.get("sha256"), str)
        ):
            raise ValueError("model transform audit normalization recipe descriptor is malformed")
        recipe_path = index_path.parent / descriptor["file"]
        if recipe_path.is_symlink() or not recipe_path.is_file():
            raise ValueError("model transform audit normalization recipe is absent or indirect")
        raw = recipe_path.read_bytes()
        if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("model transform audit normalization recipe differs from its index")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("schema") != descriptor["schema"]:
            raise ValueError("model transform audit normalization recipe schema differs from its index")
        recipe = _validate_normalization_recipe({key: val for key, val in payload.items() if key != "schema"})
        if recipe["selection_policy"] == "all":
            # The replay invokes the same runtime normalizer, but only on
            # independently re-parsed captured bytes, never on archived IR.
            from merlin.compiler.lower.quant.int8_ops import known

            replay_normalized = parse_mlir_text(contents[0])
            # Archived v1 None selected all six. New receipts retain their
            # effective pass selection explicitly; never reinterpret old bytes.
            replay_passes = (
                list(known()) if recipe["int8_compute"] and recipe["quant_passes"] is None else recipe["quant_passes"]
            )
            _normalize_model_module(
                replay_normalized,
                int8_compute=recipe["int8_compute"],
                quant_passes=replay_passes,
                prequant_gather=recipe["prequant_gather"],
            )
            if to_text(replay_normalized, generic=True) != contents[1]:
                raise ValueError("archived normalization differs from replay on the exact captured MLIR")
            normalization_replay = "matched"
        else:
            normalization_replay = "unsupported_custom_selection"
    replay = to_text(outline_dispatches(parse_mlir_text(contents[1])).module, generic=True)
    if replay != contents[2]:
        raise ValueError("archived outline differs from replay on the exact normalized MLIR")
    return {
        "schema": "merlin.model_transform_audit_qualification.v1",
        "status": "structural_replay_matched",
        "stage_sha256": {name: stage["sha256"] for name, stage in zip(expected, stages, strict=True)},
        "normalization_replay": normalization_replay,
        "semantic_equivalence": "not_proven",
    }


# ---- dispatch_runtime --------------------------------------------------------------------------
# Execute a whole model through the Merlin dispatch table (host reference runtime).
#
# This is the capstone of the outliner path: it runs an **entire** captured model by
#
# 1. outlining it into per-dispatch kernels (``merlin-outline-dispatches``),
# 2. compiling each kernel func in isolation to a host ``.so`` (``llvmlower.kernel_backend``,
#    deduplicated by kernel body so identical-shape layers compile once), and
# 3. walking the driver, evaluating the cheap view/glue ops (``expand_shape``,
#    ``collapse_shape``, ``extract_slice``, ``concat``, ``splat``, ``constant``) directly in
#    numpy and invoking each compiled kernel symbol in order.
#
# The forward arguments (input activations + weights) are bound exactly as the C runtime
# binds them — inputs from ``inputs.npz``, weights read from the ``safetensors`` blob via the
# manifest ``data_offsets`` (no safetensors library needed). The whole-model output is then
# gated against the captured torch golden, proving the outlined dispatch table is faithful
# end to end — the same numbers the monolithic compile produces, but through the unified
# per-kernel path.
#
# This is the Python reference executor; the deployable C runtime walks the same dispatch
# program. Only the view-op semantics needed by the captured models are implemented; an
# unrecognized op raises (no silent skips).

_NP = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.int8,
}


# bf16 has no native numpy dtype; we store it as the raw 16-bit pattern (uint16) so the
# compiled kernels (which read/write bf16) see the right bytes, and the numpy view ops
# (reshape/concat/slice — pure data movement) stay byte-correct. Float arithmetic on bf16
# only happens inside compiled kernels, never in the numpy view evaluator.
def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF  # round to nearest even
    return ((u + bias) >> 16).astype(np.uint16)


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f8e4m3fn_to_f32(u8: np.ndarray) -> np.ndarray:
    """Decode OCP float8_e4m3fn (1 sign, 4 exp bias-7, 3 mantissa; no inf, NaN = S.1111.111).

    torchao's ``float8_weight_only_e4m3`` stores weights as ``F8_E4M3`` in the safetensors;
    numpy has no fp8 dtype, so the bytes must be decoded to f32 at load (the MLIR forward arg
    is f32 — the graph multiplies by a separate f32 scale, weight-only style).
    """
    u = np.ascontiguousarray(u8, np.uint8).astype(np.uint32)
    sign = np.where((u >> 7) & 1 == 1, np.float32(-1.0), np.float32(1.0))
    exp = (u >> 3) & 0xF
    man = (u & 0x7).astype(np.float32)
    sub = (man / 8.0) * np.float32(2.0**-6)  # exp == 0: subnormal
    nrm = (1.0 + man / 8.0) * np.exp2((exp.astype(np.float32)) - 7.0)
    val = (sign * np.where(exp == 0, sub, nrm)).astype(np.float32)
    return np.where((exp == 0xF) & (u & 0x7 == 0x7), np.float32(np.nan), val)


def _np_dtype(elem: str):
    """numpy storage dtype for an MLIR element type (bf16 stored as uint16 bit pattern)."""
    if elem == "bf16":
        return np.uint16
    return _NP.get(elem, np.float32)


# --- forward-argument binding (inputs + weights), mirroring llvmlower.c_runtime ----------


def resolve_forward_args(model_dir: str | Path, *, cache: ForwardArgsCache | None = None) -> list[np.ndarray]:
    """Numpy array for every ``func @forward`` argument, in signature order.

    With ``cache``, the stage's weights, buffers and lifted constants are read once per cache and the
    same (read-only) arrays are returned on every call; only the model inputs are read from
    ``inputs.npz`` each time.
    """
    model_dir = Path(model_dir)
    inputs = np.load(model_dir / "inputs.npz")
    files = tuple(inputs.files)
    plan = cache.plan(model_dir, files) if cache is not None else _forward_plan(model_dir, files)
    args: list[np.ndarray] = []
    for entry in plan:
        if isinstance(entry, _ModelInput):
            arr = np.ascontiguousarray(inputs[entry.key])
            if entry.element == "bf16":
                args.append(f32_to_bf16(arr).reshape(entry.shape))
            else:
                args.append(arr.astype(_np_dtype(entry.element)).reshape(entry.shape))
        else:
            args.append(entry)
    return args


def _forward_plan(model_dir: Path, input_files: tuple[str, ...]) -> list[np.ndarray | _ModelInput]:
    """Preserve the dispatch binding entrypoint and its selected numeric converters."""
    return plan_forward_args(
        model_dir,
        input_files,
        _np_dtype=_np_dtype,
        f32_to_bf16=f32_to_bf16,
        f8e4m3fn_to_f32=f8e4m3fn_to_f32,
    )


class ForwardArgsCache:
    """The constant ``@forward`` arguments of each stage (weights, buffers, lifted constants), read once.

    A stage is keyed by the resolved path, size and modification time of every file its constants and
    its input binding are read from, and by the members of its ``inputs.npz``; a shadow directory whose
    files link to a stage's shares that stage's entry. Every cached array is read-only, so the same
    objects can be handed to every call: :meth:`is_constant` names them.
    """

    FILES = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json", "extra.npz", "input_order.json")

    def __init__(self) -> None:
        self._plans: dict[tuple[Any, ...], list[np.ndarray | _ModelInput]] = {}
        self._constants: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._plans)

    def plan(self, model_dir: Path, input_files: tuple[str, ...]) -> list[np.ndarray | _ModelInput]:
        key = (*(self._identity(Path(model_dir) / name) for name in self.FILES), input_files)
        if key not in self._plans:
            plan = _forward_plan(Path(model_dir), input_files)
            for entry in plan:
                if isinstance(entry, np.ndarray):
                    entry.flags.writeable = False
                    self._constants[id(entry)] = entry
            self._plans[key] = plan
        return self._plans[key]

    def is_constant(self, value: Any) -> bool:
        """``value`` is one of the cached constant arguments (the very object)."""
        return id(value) in self._constants

    @staticmethod
    def _identity(path: Path) -> tuple[str, int, int] | None:
        if not path.exists():
            return None
        resolved = path.resolve()
        stat = resolved.stat()
        return str(resolved), stat.st_size, stat.st_mtime_ns


# --- view-op evaluation ------------------------------------------------------------------

_DYN = -9223372036854775808  # xDSL dynamic-dim sentinel


def _tensor_shape(t) -> tuple[int, ...]:
    from xdsl.dialects.builtin import TensorType

    return tuple(int(d) for d in t.get_shape()) if isinstance(t, TensorType) else ()


def _reshape_shape(rtype) -> tuple[int, ...]:
    """Result shape with dynamic dims as -1 (numpy infers the single unknown)."""
    return tuple(-1 if d <= _DYN // 2 else d for d in _tensor_shape(rtype))


def _elem_str(t) -> str:
    from xdsl.dialects.builtin import TensorType

    et = t.element_type if isinstance(t, TensorType) else t
    return str(et)


def _scalar_dtype(t):
    return _np_dtype(_elem_str(t))


def _fill_value(value, elem: str):
    """Scalar fill value in the storage dtype (bf16 -> its 16-bit pattern as an int)."""
    if elem == "bf16":
        return int(f32_to_bf16(np.array([float(value)], np.float32))[0])
    return value


def _const_value(op):
    val = op.properties["value"]
    data = getattr(val, "value", None)
    if data is not None and hasattr(data, "data"):
        return data.data
    # dense elements constant
    rtype = op.results[0].type
    arr = np.array(list(val.get_values())).reshape(_tensor_shape(rtype))
    if _elem_str(rtype) == "bf16":
        return f32_to_bf16(arr)
    return arr.astype(_scalar_dtype(rtype))


def _pad_value(op, env):
    """The scalar constant a tensor.pad fills with (its region's tensor.yield)."""
    for region in op.regions:
        for block in region.blocks:
            for sub in block.ops:
                if sub.name in ("tensor.yield", "linalg.yield"):
                    v = sub.operands[0]
                    if id(v) in env:
                        return env[id(v)]
                    owner = getattr(v, "owner", None)
                    if owner is not None and owner.name == "arith.constant":
                        return _const_value(owner)
    return 0


def _int_array(attr) -> list[int]:
    return [int(v) for v in attr.get_values()]


def _pad_rank(a: list[int], fill: int, rank: int) -> list[int]:
    """Left-pad a rank-reduced static offset/size/stride array to the source rank."""
    return [fill] * (rank - len(a)) + a if len(a) < rank else a


# Scalar arith evaluated directly (driver-level index/condition glue, not compute kernels).
_CMPI_PRED = {
    0: lambda a, b: a == b,
    1: lambda a, b: a != b,
    2: lambda a, b: a < b,
    3: lambda a, b: a <= b,
    4: lambda a, b: a > b,
    5: lambda a, b: a >= b,
    6: lambda a, b: a < b,
    7: lambda a, b: a <= b,
    8: lambda a, b: a > b,
    9: lambda a, b: a >= b,
}
_SCALAR_ARITH = {
    "arith.addi",
    "arith.subi",
    "arith.muli",
    "arith.divsi",
    "arith.remsi",
    "arith.maxsi",
    "arith.minsi",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.cmpi",
    "arith.select",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.sitofp",
    "arith.fptosi",
}


def _eval_scalar_arith(op, env):
    name = op.name
    a = env[id(op.operands[0])] if op.operands else None
    if name == "arith.select":
        return env[id(op.operands[1])] if a else env[id(op.operands[2])]
    if name in ("arith.extsi", "arith.extui", "arith.trunci", "arith.fptosi"):
        return int(a)
    if name == "arith.sitofp":
        return float(a)
    b = env[id(op.operands[1])]
    if name == "arith.cmpi":
        return bool(_CMPI_PRED[int(op.properties["predicate"].value.data)](a, b))
    return {
        "arith.addi": lambda: a + b,
        "arith.subi": lambda: a - b,
        "arith.muli": lambda: a * b,
        "arith.divsi": lambda: int(a / b),
        "arith.remsi": lambda: int(a - b * int(a / b)),
        "arith.maxsi": lambda: max(a, b),
        "arith.minsi": lambda: min(a, b),
        "arith.andi": lambda: a & b,
        "arith.ori": lambda: a | b,
        "arith.xori": lambda: a ^ b,
        "arith.addf": lambda: a + b,
        "arith.subf": lambda: a - b,
        "arith.mulf": lambda: a * b,
        "arith.divf": lambda: a / b,
    }[name]()


def _eval_view(op, env: dict[int, Any]):
    name = op.name
    rtype = op.results[0].type
    rshape = _tensor_shape(rtype)
    if name == "arith.constant":
        return _const_value(op)
    if name == "tensor.empty":
        # dynamic dims (tensor<?x...>) take their size from the op's SSA operands, in order
        dyn = list(op.operands)
        j = 0
        shape = []
        for d in _tensor_shape(rtype):
            if d <= _DYN // 2:
                shape.append(int(env[id(dyn[j])]))
                j += 1
            else:
                shape.append(d)
        return np.zeros(shape, _scalar_dtype(rtype))
    if name in ("tensor.splat", "linalg.fill"):
        elem = _elem_str(rtype)
        return np.full(rshape, _fill_value(env[id(op.operands[0])], elem), dtype=_np_dtype(elem))
    if name in ("tensor.collapse_shape", "tensor.expand_shape"):
        return np.ascontiguousarray(env[id(op.operands[0])]).reshape(_reshape_shape(rtype))
    if name == "tensor.concat":
        dim = int(op.properties["dim"].value.data)
        return np.concatenate([env[id(o)] for o in op.operands], axis=dim)
    if name == "tensor.pad":
        src = env[id(op.operands[0])]
        low = _int_array(op.properties["static_low"])
        high = _int_array(op.properties["static_high"])
        if op.operands[1:]:  # dynamic pad amounts not supported
            raise DispatchRuntimeError("dynamic tensor.pad amounts not supported")
        return np.pad(src, list(zip(low, high)), mode="constant", constant_values=_pad_value(op, env))
    if name == "tensor.extract_slice":
        seg = _int_array(op.properties["operandSegmentSizes"])
        if seg[1:] != [0, 0, 0]:
            raise DispatchRuntimeError("dynamic extract_slice not supported")
        src = env[id(op.operands[0])]
        rank = src.ndim
        # model2MLIR can emit rank-reduced static arrays (only the kept dims). Left-pad to
        # source rank — offsets 0, sizes 1, strides 1 — matching passes_xdsl._fix_extract_slice.
        offs = _pad_rank(_int_array(op.properties["static_offsets"]), 0, rank)
        sizes = _pad_rank(_int_array(op.properties["static_sizes"]), 1, rank)
        strides = _pad_rank(_int_array(op.properties["static_strides"]), 1, rank)
        idx = tuple(slice(o, o + s * st, st) for o, s, st in zip(offs, sizes, strides))
        return np.ascontiguousarray(src[idx]).reshape(_reshape_shape(rtype))
    if name == "tensor.insert_slice":
        seg = _int_array(op.properties["operandSegmentSizes"])
        if seg[2:] != [0, 0, 0]:
            raise DispatchRuntimeError("dynamic insert_slice not supported")
        src = env[id(op.operands[0])]
        dst = np.array(env[id(op.operands[1])], copy=True)
        rank = dst.ndim
        offs = _pad_rank(_int_array(op.properties["static_offsets"]), 0, rank)
        sizes = _pad_rank(_int_array(op.properties["static_sizes"]), 1, rank)
        strides = _pad_rank(_int_array(op.properties["static_strides"]), 1, rank)
        idx = tuple(slice(o, o + s * st, st) for o, s, st in zip(offs, sizes, strides))
        dst[idx] = np.asarray(src).reshape(dst[idx].shape)
        return dst
    if name in ("arith.index_cast", "arith.index_castui"):
        return int(env[id(op.operands[0])])
    # scalar integer/float arith (loop-index / mask / condition glue in the driver)
    if name in _SCALAR_ARITH:
        return _eval_scalar_arith(op, env)
    if name == "tensor.extract":
        src = env[id(op.operands[0])]
        idx = tuple(int(env[id(o)]) for o in op.operands[1:])
        return src[idx]
    if name == "tensor.insert":
        scalar = env[id(op.operands[0])]
        dst = np.array(env[id(op.operands[1])], copy=True)
        idx = tuple(int(env[id(o)]) for o in op.operands[2:])
        dst[idx] = scalar
        return dst
    if name == "tensor.from_elements":
        elem = _elem_str(rtype)
        vals = [env[id(o)] for o in op.operands]
        if elem == "bf16":
            return f32_to_bf16(np.array(vals, np.float32)).reshape(_reshape_shape(rtype))
        return np.array(vals, dtype=_np_dtype(elem)).reshape(_reshape_shape(rtype))
    raise DispatchRuntimeError(f"no runtime evaluation for view op {name}")


# --- the executor ------------------------------------------------------------------------


def _kernel_io(call):
    """(callee symbol, output (shape,dtype) list) for a func.call."""
    callee = call.callee.string_value() if hasattr(call.callee, "string_value") else call.callee.root_reference.data
    outs = [(_tensor_shape(r.type), _scalar_dtype(r.type)) for r in call.results]
    return callee, outs


def _compile_kernel_so(args: tuple) -> str:
    """Process-pool worker: lower a kernel's MLIR text to a host ``.so`` at ``so_path``.

    Runs in a separate process (xDSL's parser/lowering keeps mutable global state and is NOT
    thread-safe — concurrent parses race and corrupt op-building), so each kernel compiles in
    an isolated interpreter. Takes/returns only strings (picklable)."""
    import shutil
    from pathlib import Path

    from merlin.compiler.lower.model.lower import lower_model

    text, workdir_str, so_path = args
    wd = Path(workdir_str)
    lower_model(text, wd, targets=("host",))
    shutil.copy2(wd / "model_host.so", so_path)
    return so_path


def mesh_datapath(target: str, *, numeric_policy: dict | None = None):
    """Combine the selected target contract and caller-declared numerical policy into a binding.

    The mesh boundary below used to be written as ``operand_dtype="int8", accum_dtype="i32"`` with an
    element-type filter of exactly ``("f32", "i8")``. Both are facts about ONE target's datapath, and on a
    float-datapath accelerator they are wrong in the two ways that cancel out into silence: the filter
    never classifies an fp8 layer, so no layer is ever routed, so the hardcoded dtypes are never reached
    and never observed to be wrong. The whole model runs on the host and the run looks clean.

    This RETURNS THE BINDING rather than a hand-picked tuple of four of its fields, and that is the point.
    The projection it used to perform is how ``subnormal_operand_flush`` was dropped between the
    authored numerical declaration and the consumer that feeds the mesh real model operands.
    The fact existed and was correct; nothing carried it the last few lines. Every lossy projection of a
    derived-facts object is that same bug waiting to be written again, so the projection is gone: callers
    take the fields they need off the binding, and a field added later reaches them without a signature
    change. The MLIR spelling the filter wants is ``b.mlir_dtype(b.operand_dtype)``.
    """
    from merlin.compiler.selection.kernels import _mesh_tile_binding

    return _mesh_tile_binding(target, None, None, numeric_policy=numeric_policy)


def _has_contraction(kfn) -> bool:
    """True when a kernel func contains a contraction op at all — used to count the matmuls the mesh
    classifier declined, structurally (no name matching on the symbol)."""
    for op in kfn.walk():
        nm = getattr(op, "name", "")
        if nm in ("linalg.matmul", "linalg.batch_matmul", "linalg.matmul_transpose_b", "linalg.quantized_matmul"):
            return True
    return False


def _classify_mesh_matmul(kfn, accept: tuple[str, ...] = ("f32",)) -> dict | None:
    """Classify a kernel for the MESH route: a plain 2-D ``linalg.matmul`` whose two
    ``ins`` are kernel block args, with an element type this target's mesh can take.

    ``accept`` is derived from the target's own declared operand format (plus ``f32``, the host lane's
    spelling, which the boundary converts). It was the literal pair ``("f32", "i8")``, which silently
    routed nothing at all on any target whose datapath is not int8. Returns ``{"a", "b", "in_dtype"}`` so
    the boundary knows whether the layer arrives already in the mesh's format or needs converting."""
    from xdsl.dialects.builtin import TensorType

    block = kfn.body.blocks[0]
    arg_ids = {id(a): i for i, a in enumerate(block.args)}
    ret = next((o for o in block.ops if o.name == "func.return"), None)
    if ret is None or len(ret.operands) != 1:
        return None
    mm = getattr(ret.operands[0], "owner", None)
    if mm is None or getattr(mm, "name", None) != "linalg.matmul" or len(mm.operands) < 3:
        return None
    a_val, b_val = mm.operands[0], mm.operands[1]

    def _et(v):
        t = v.type
        return str(t.element_type) if isinstance(t, TensorType) and len(t.get_shape()) == 2 else None

    ea, eb = _et(a_val), _et(b_val)
    if ea is None or ea != eb or ea not in accept:
        return None
    if id(a_val) not in arg_ids or id(b_val) not in arg_ids:
        return None  # an operand is computed inside the kernel
    return {"a": arg_ids[id(a_val)], "b": arg_ids[id(b_val)], "in_dtype": ea}


#: the canonical 2-D contraction iterator signature: (i, j) parallel over a k reduction.
_MATMUL_ITERS = ("parallel", "parallel", "reduction")


def _is_widening_mac_body(gen) -> bool:
    """True when a ``linalg.generic`` body is a (widening) multiply-accumulate: optional operand
    extensions, then one multiply, one add, and the yield.

    Structural, over the IR's own op vocabulary — not a pattern match on text. Accepts the integer
    (``muli``/``addi``) and float (``mulf``/``addf``) spellings alike, so the same check covers a
    quantized i8xi8->i32 contraction and an f32 one."""
    blocks = getattr(getattr(gen, "body", None), "blocks", None)
    if not blocks:
        return False
    names = [o.name for o in blocks[0].ops]
    if not names or names[-1] != "linalg.yield":
        return False
    core = [n for n in names[:-1] if not n.startswith("arith.ext")]
    return core in (["arith.muli", "arith.addi"], ["arith.mulf", "arith.addf"])


def _classify_mesh_contraction(kfn, accept: tuple[str, ...] = ("f32",)) -> dict | None:
    """The ``linalg.generic`` form of :func:`_classify_mesh_matmul`.

    Quantization rewrites every ``linalg.matmul`` into a generic — measured on an int8 whole model, 0 of
    325 outlined kernels still return a ``linalg.matmul``, and the matmul-only classifier therefore routed
    NOTHING to the mesh: 0 layers on the accelerator, 0 recorded as fallbacks, and a capstone that read as
    a pass. A quantized contraction is exactly what an integer mesh exists for, so it must be recognised.

    Requires the canonical matmul indexing maps, not merely the right iterator shape: a transposed or
    otherwise permuted contraction has the same iterators and computes something else, and routing it to
    the mesh would silently produce the wrong answer. Fails closed when the maps cannot be read."""
    from xdsl.dialects.builtin import TensorType

    block = kfn.body.blocks[0]
    arg_ids = {id(a): i for i, a in enumerate(block.args)}
    ret = next((o for o in block.ops if o.name == "func.return"), None)
    if ret is None or len(ret.operands) != 1:
        return None
    gen = getattr(ret.operands[0], "owner", None)
    if gen is None or getattr(gen, "name", None) != "linalg.generic" or len(gen.inputs) != 2:
        return None
    if tuple(str(getattr(a, "data", a)).rsplit(".", 1)[-1].lower() for a in gen.iterator_types) != _MATMUL_ITERS:
        return None
    if not _is_widening_mac_body(gen):
        return None

    a_val, b_val = gen.inputs[0], gen.inputs[1]

    def _et(v):
        t = v.type
        return str(t.element_type) if isinstance(t, TensorType) and len(t.get_shape()) == 2 else None

    ea, eb = _et(a_val), _et(b_val)
    if ea is None or ea != eb or ea not in accept:
        return None
    if isinstance(gen.results[0].type, TensorType) and len(gen.results[0].type.get_shape()) != 2:
        return None
    if id(a_val) not in arg_ids or id(b_val) not in arg_ids:
        return None

    # the maps must be exactly (d0,d2), (d2,d1) -> (d0,d1)
    try:
        maps = [m.data for m in gen.indexing_maps]
        got = tuple(tuple(str(r) for r in mp.results) for mp in maps)
    except Exception:  # noqa: BLE001 — unreadable maps: fail closed
        return None
    if got != (("d0", "d2"), ("d2", "d1"), ("d0", "d1")):
        return None
    return {"a": arg_ids[id(a_val)], "b": arg_ids[id(b_val)], "in_dtype": ea, "via": "generic"}


def _oracle_unreachable(decline: str) -> bool:
    """Whether a mesh decline describes our MEASUREMENT failing rather than the backend refusing.

    A timed-out simulator, a missing binary, a build that did not produce an oracle: none of these say
    anything about whether the mesh can run the layer. Treating them as refusals turned a whole model
    whose every layer the mesh executes correctly into "15 of 15 fell back to the host", and that number
    then failed a must_accelerate gate -- a compiler blamed for a simulator budget.

    Matched structurally on the oracle's own words (no regex, repo rule); anything unrecognized is
    treated as a genuine refusal, which is the conservative direction: an unknown cause must not be able
    to excuse a real fallback."""
    d = (decline or "").lower()
    if not d:
        return False
    # The worker never reached the compiler or mesh when the host denied
    # bubblewrap's loopback setup. This is an execution-environment failure,
    # not evidence that a target layer cannot run.
    if "bwrap: loopback:" in d and "netlink_route socket" in d:
        return True
    return any(
        tok in d
        for tok in (
            "timed out",
            "timeout",
            "invocation failed",
            "no such file",
            "not found",
            "unreachable",
            "build failed",
            "oracle unavailable",
        )
    )


def _propagate_quant_inner(module) -> int:
    """Re-attach quant-inner binding keys onto the (elided, uninitialized) torchao subclass
    inner-tensor ``tensor.empty`` ops, in memory, so ``execute`` can bind the real quantized
    data. xDSL's printer drops attributes on tensor.empty, so m2m parks the keys on the
    CONSUMER ops instead: every op carries ``prov.quant_inner_<i>`` for operand ``i`` whose
    source is an inner-tensor empty. Covers int8 (dequant) and fp8 (mul x scale) patterns.
    Also honors the legacy ``prov.quant_inner_{w,s}`` on the dequant op (operands 0/1)."""

    def _tag_src_empty(val, tag):
        op = getattr(val, "owner", None)
        for _ in range(8):  # walk through transpose/cast wrappers
            if op is None or not hasattr(op, "name"):
                return
            if op.name == "tensor.empty":
                op.attributes["prov.quant_inner"] = tag
                return
            ins = getattr(op, "operands", ())
            op = getattr(ins[0], "owner", None) if ins else None

    n = 0
    for op in module.walk():
        operands = getattr(op, "operands", ())
        for key, tag in op.attributes.items():
            idx = None
            if key.startswith("prov.quant_inner_"):
                suffix = key[len("prov.quant_inner_") :]
                idx = {"w": 0, "s": 1}.get(suffix)  # legacy dequant aliases
                if idx is None and suffix.isdigit():
                    idx = int(suffix)
            if idx is not None and idx < len(operands):
                _tag_src_empty(operands[idx], tag)
                n += 1
    return n


def outlined_dispatch_inventory(outlined) -> list[dict[str, str | None]]:
    """Exact static dispatch identities carried beside a completed-run ledger."""
    return [
        {
            "symbol": dispatch.symbol,
            "root_op": dispatch.root_op,
            "region_id": dispatch.prov.get("prov.region_id"),
            "prov_op": dispatch.prov.get("prov.op"),
            "prov_role": dispatch.prov.get("prov.role"),
            "prov_rewrite": dispatch.prov.get("prov.rewrite"),
        }
        for dispatch in outlined.dispatches
    ]


def _normalize_model_module(
    module,
    *,
    int8_compute: bool,
    quant_passes: list[str] | None,
    prequant_gather: bool,
    quant_select=None,
) -> None:
    """The one normalization sequence used by execution and exact audit replay."""
    from merlin.compiler.quantization.linalg import lower_quant_ext
    from merlin.compiler.quantization.torchao.affine import lower_torchao_affine_quant
    from merlin.compiler.transforms.canonicalization import (
        collapse_overrank_matmul,
        fix_bool_fptosi,
        fix_bool_sitofp,
        lower_bf16_matmul_f32acc,
    )

    lower_torchao_affine_quant(module)
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)
    if int8_compute:
        from merlin.compiler.lower.options.quantization_scope import captured_quantized_contraction
        from merlin.compiler.lower.quant.int8_ops import apply_quant, compute_passes

        # Match the compiled captured-scope path: a BF16 dequantized operand
        # needs an F32 accumulator before the INT8 contraction selector runs.
        if quant_select is captured_quantized_contraction:
            lower_bf16_matmul_f32acc(module)

        extra = {"prequant_gather": True} if prequant_gather else {}
        apply_quant(module, list(compute_passes(quant_passes)), select=quant_select, **extra)
    lower_quant_ext(module)
    lower_bf16_matmul_f32acc(module)
    fix_bool_sitofp(module)
    fix_bool_fptosi(module)


def run_model(
    model_dir: str | Path,
    workdir: str | Path,
    cache_dir: str | Path | None = None,
    tap=None,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    quant_select=None,
    prequant_gather: bool = False,
    kernel_backend: str | None = None,
    mesh_target: str | None = None,
    mesh_package: str | None = None,
    numeric_policy: dict | None = None,
    transform_audit: bool | str | None = None,
) -> dict[str, Any]:
    """Outline + bind + execute a captured model; gate against ``golden.npy``.

    Returns ``{output, golden, cos, rel, ok, n_kernels, n_unique_kernels}``.

    ``int8_compute=True`` runs the integer (W8A8) datapath: each ``dequant(weight)→f32 matmul``
    becomes ``quantize(act)→ i8×i8→i32 matmul → requant`` (native integer reference contraction),
    instead of dequantizing the weight to f32 (the default weight-only path).

    ``prequant_gather=True`` additionally moves each gathered activation's quantization to BEFORE
    the gather with a per-tensor scale (the ``quantize_before_gather`` feature). It is meaningful
    only with ``int8_compute=True`` and it is NOT bit-exact against the default per-row scheme.

    ``quant_passes`` / ``quant_select`` narrow that datapath and are meaningful only when
    ``int8_compute=True``. ``quant_passes`` is a subset of ``quant_passes.known()`` (default None =
    contraction/conv only); nonlinear approximations require explicit selection. ``quant_select``
    is an ``(op) -> bool`` predicate restricting which ops those passes may rewrite. Together they
    make the datapath's REACH a variable. It separates "our arithmetic is wrong" from "we quantize
    more operations than the reference does" when grading against a reference whose quantization
    policy is narrower (e.g. torchao quantizes ``nn.Linear`` only).
    """
    import hashlib
    import os as _os

    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.compiler.lower.quant.int8_ops import compute_passes
    from merlin.compiler.partition.outline import outline_dispatches
    from merlin.runtime.dispatch.executor import execute

    if not int8_compute and quant_passes is not None:
        raise ValueError("quant_passes requires int8_compute=True")
    effective_quant_passes = list(compute_passes(quant_passes)) if int8_compute else None

    model_dir = Path(model_dir)
    model_source = model_dir / "model.mlir"
    source_sha256 = hashlib.sha256(model_source.read_bytes()).hexdigest()
    module = parse_mlir_file(model_source)
    if hashlib.sha256(model_source.read_bytes()).hexdigest() != source_sha256:
        raise DispatchRuntimeError("model source changed while it was parsed")
    # Normalize before outlining so quantized/bf16/over-rank models compute correctly:
    #  - 3-D `aten.linear` matmuls (invalid 2-D-map matmul) -> batched linalg.generic;
    #  - int8 weights: dequantize_per_channel -> linalg.generic (weights stay i8 in memory);
    #  - bf16 matmuls: accumulate in f32 then truncate (matches torch), else ~bf16 round-off;
    #  - bool->float casts: sitofp(i1) (true -> -1.0) -> uitofp (true -> 1.0), matching torch
    #    (fixes the eager-attention causal-mask sign flip; molmoact decoder).
    # An activation-quant capture leaves torchao's choose_qparams/quantize as opaque calls to
    # externs nothing defines; without this the module cannot even be outlined.
    _normalize_model_module(
        module,
        int8_compute=int8_compute,
        quant_passes=effective_quant_passes,
        prequant_gather=prequant_gather,
        quant_select=quant_select,
    )
    outlined = outline_dispatches(module)
    requested_audit = (
        _os.environ.get("MERLIN_MODEL_TRANSFORM_AUDIT", "") if transform_audit is None else transform_audit
    )
    transform_audit_index = record_model_transform_audit(
        model_source,
        Path(workdir),
        module,
        outlined,
        enabled={"": False, "1": True, "exact": "exact", "both": "both", "compact": "compact"}.get(
            requested_audit, requested_audit
        ),
        expected_source_sha256=source_sha256,
        normalization_recipe={
            "int8_compute": int8_compute,
            "quant_passes": effective_quant_passes,
            "prequant_gather": prequant_gather,
            "selection_policy": "all" if quant_select is None else "custom_unreplayable",
        },
    )
    transform_audit_qualification = (
        qualify_model_transform_audit(transform_audit_index) if transform_audit_index is not None else None
    )
    # Static inventory from the exact normalized module whose dispatches this
    # invocation executes.  A source region may contain multiple linalg roots;
    # the dynamic ledger alone cannot tell whether one of them disappeared.
    outlined_dispatches = outlined_dispatch_inventory(outlined)
    driver = next(op for op in outlined.module.walk() if op.name == "func.func" and "$kernel_" not in op.sym_name.data)
    out_types = list(driver.function_type.outputs.data)
    if not out_types:
        raise DispatchRuntimeError("whole-model forward has no results")
    args = resolve_forward_args(model_dir)
    # Quantized-subclass inner tensors (int_data/scale) the capture extracted under qinner::
    # keys; bind them to the prov.quant_inner-tagged empties m2m left uninitialized.
    qinner = None
    extra_path = model_dir / "extra.npz"
    if extra_path.is_file():
        ex = np.load(extra_path)
        qinner = {k[len("qinner::") :]: ex[k] for k in ex.files if k.startswith("qinner::")}
    # Per-run mesh counters. A whole-model verdict is decided by these, so they must not
    # live on a module-global function attribute that a concurrent grade can clobber.
    _mesh_counts: dict = {}
    _mesh_counts["outlined_dispatches"] = outlined_dispatches
    if transform_audit_index is not None:
        _mesh_counts["transform_audit_index"] = str(transform_audit_index)
        _mesh_counts["transform_audit_qualification"] = transform_audit_qualification
    if kernel_backend == "mesh" and mesh_target:
        # Compare the grouping the target admits with the outline THIS run will execute.  The
        # whole-model runtime still outlines one linalg operation per kernel; a separate planning
        # report built from the capture is not evidence that an epilogue stayed with its producer.
        # Keep the result separate from the dynamic dispatch ledger: this proves only emitted
        # structure, not where a completed call ran.
        from merlin.analysis.optimization.candidates import placement_census as _placement_census

        try:
            _mesh_counts["planned_outlined_alignment"] = _placement_census.planned_outlined_alignment(
                module, mesh_target, outlined
            )
        except Exception as error:  # noqa: BLE001 -- an unavailable comparison cannot certify placement
            _mesh_counts["planned_outlined_alignment"] = {
                "schema": "planned_outlined_alignment_v1",
                "status": "incomplete",
                "reason": f"{type(error).__name__}: {error}",
            }
    results = execute(
        outlined,
        args,
        Path(workdir),
        cache_dir=cache_dir,
        tap=tap,
        qinner=qinner,
        kernel_backend=kernel_backend,
        mesh_target=mesh_target,
        mesh_package=mesh_package,
        numeric_policy=numeric_policy,
        counters=_mesh_counts,
    )
    if len(results) != len(out_types):
        raise DispatchRuntimeError(f"whole-model execution returned {len(results)} values for {len(out_types)} results")
    # Widen each result independently.  Concatenating results would erase the declared ABI and can let
    # a correct first result hide an untouched recurrent-state result.
    widened = [
        (bf16_to_f32(raw) if _elem_str(typ) == "bf16" else np.asarray(raw, dtype=np.float32)).ravel()
        for raw, typ in zip(results, out_types, strict=True)
    ]

    res: dict[str, Any] = {
        "output": results[0],
        "outputs": list(results),
        "n_kernels": outlined.n_kernels,
        "n_unique_kernels": getattr(execute, "last_unique_kernels", None),
        "kernel_backend": kernel_backend,
        "quantization_passes": effective_quant_passes,
        **_mesh_counts,
    }
    goldens_path = model_dir / "goldens.npz"
    order_path = model_dir / "output_order.json"
    if goldens_path.is_file() and order_path.is_file():
        names = [str(x) for x in json.loads(order_path.read_text(encoding="utf-8"))]
        archive = np.load(goldens_path, allow_pickle=False)
        if len(names) != len(widened) or set(names) != set(archive.files):
            raise DispatchRuntimeError("goldens.npz/output_order.json disagree with model results")
        golden_arrays = [np.asarray(archive[name], dtype=np.float32).ravel() for name in names]
    else:
        gpath = model_dir / "golden.npy"
        golden_arrays = [np.load(gpath).astype(np.float32).ravel()] if gpath.is_file() else []
        names = ["Y0"] if golden_arrays else []
    if golden_arrays:
        checks = []
        # A capture with only golden.npy references result 0; the further results have no golden here.
        for name, out, gold in zip(names, widened[: len(golden_arrays)], golden_arrays, strict=True):
            if len(out) != len(gold):
                raise DispatchRuntimeError(f"model result {name!r} has {len(out)} values, golden has {len(gold)}")
            gmax = float(np.abs(gold).max())
            abs_err = float(np.abs(out - gold).max())
            rel = abs_err / max(1e-9, gmax)
            cos = float((out @ gold) / (np.linalg.norm(out) * np.linalg.norm(gold) + 1e-12))
            ok = rel < 1e-3 if gmax < 1e-6 else (cos > 0.9999 and rel < 1e-3)
            checks.append({"name": name, "cos": cos, "rel": rel, "abs_err": abs_err, "gold_absmax": gmax, "ok": ok})
        res.update(
            golden=golden_arrays[0],
            cos=min(x["cos"] for x in checks),
            rel=max(x["rel"] for x in checks),
            abs_err=max(x["abs_err"] for x in checks),
            gold_absmax=max(x["gold_absmax"] for x in checks),
            ok=all(x["ok"] for x in checks),
            output_checks=checks,
        )
    if hashlib.sha256(model_source.read_bytes()).hexdigest() != source_sha256:
        raise DispatchRuntimeError("model source changed during whole-model execution")
    return res
