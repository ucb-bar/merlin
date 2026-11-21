"""The expected result of one device-kernel call: the group's interface program evaluated in numpy.

A device group is built from its *stated program* (:func:`merlin.compiler.lower.device.build.kernel_entry`):
the entry is restated as an interface module (``corpus_spec.build``), and the package lowers that
module. The kernel is therefore correct exactly when, on the host call's operands, it returns what the
interface program computes. :func:`merlin.runtime.engine.reference.reference_outputs` defines that computation
on Merlin's integer ``Tensor``; this module computes the same function with numpy so that a whole
model's calls can be answered in seconds:

- contractions (``MATMUL``/``MATMUL_RESIDENT``/``CONV2D``/``BATCHED_MATMUL``) sum int8 x int8 exactly
  (float64 products and sums are exact far beyond any reduction length a call carries);
- ``acc_scale`` multiplies in float32 and rounds to nearest, ties to even, as
  ``Tensor.requant_acc_scale``; ``requant`` is the rounding arithmetic shift of ``Tensor.requant``;
- pooling and anything outside this vocabulary are computed by the reference engine itself, so no
  second definition of those exists.

The two engines are kept equal by test (``tests/runtime/test_model_check_kernel_reference.py``)
and, per model, by evaluating the device-rewritten program with these answers against the unrewritten
INT8 reference (:mod:`merlin.evaluation.model.bundles.program_catalog`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from merlin.runtime.engine import reference as REF
from merlin.runtime.engine.command_buffer import (
    BIAS_STAGES,
    DERIVATION_RECIPE_KEYS,
    Tensor,
    apply_pool_stage,
    bias_tensor_name,
    conv_out_dims,
    declared_acc_scale,
    declared_acc_scale_by_column,
)

# ---- kernel_reference --------------------------------------------------------------------------

#: Opcodes evaluated in numpy here. A buffer using any other opcode is evaluated whole by the
#: reference engine instead (correct, slower).
FAST_OPCODES = frozenset(
    {"RES_PACK", "EVICT", "MATMUL", "MATMUL_RESIDENT", "COMMIT", "CONV2D", "BIAS_ADD", "MOVEMENT", "BATCHED_MATMUL"}
)

_EXACT_F64_LIMIT = float(2**53)


class KernelReferenceError(ValueError):
    """The interface program or the call's operands cannot be evaluated as stated."""


def interface_for(
    entry: Mapping[str, Any],
    *,
    target: str,
    operand_dtype: str,
    accum_dtype: str,
    numeric_policy: dict | None = None,
) -> str:
    """The interface module a device build hands the package for one stated group program."""
    from merlin.compiler.selection.kernels import _mesh_tile_binding
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1

    binding = _mesh_tile_binding(target, operand_dtype, accum_dtype, numeric_policy=numeric_policy)
    _capsule, interface = _merlin_model_builders_1.build(dict(entry), binding)
    return interface

    # ------------------------------------------------------------------------------------- arithmetic


def _int_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Exact integer (m, k) x (k, n) -> int64."""
    a = np.asarray(lhs, dtype=np.int64)
    b = np.asarray(rhs, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise KernelReferenceError(f"matmul shape mismatch: {a.shape} x {b.shape}")
    bound = float(np.abs(a).max(initial=0)) * float(np.abs(b).max(initial=0)) * a.shape[1]
    if bound < _EXACT_F64_LIMIT:
        return np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)
    return a @ b


def _acc_scale(x: np.ndarray, scale: float | list[float]) -> np.ndarray:
    prod = np.asarray(x, dtype=np.int64).astype(np.float32) * np.asarray(scale, dtype=np.float32)
    return np.rint(prod).astype(np.int64)


def _requant(x: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return np.asarray(x, dtype=np.int64)
    return (np.asarray(x, dtype=np.int64) + (1 << (shift - 1))) >> shift


def _narrow(x: np.ndarray, dtype: str) -> np.ndarray:
    """``reference._narrow_int_readout``: saturate into a declared integer container below 32 bits."""
    kind, digits = (dtype[:1], dtype[1:]) if dtype else ("", "")
    if kind not in ("i", "u") or not digits.isdigit():
        return x
    bits = int(digits)
    if bits >= 32:
        return x
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if kind == "i" else (0, (1 << bits) - 1)
    return np.clip(x, lo, hi)


def _pool(x: np.ndarray, attrs: Mapping[str, Any], op: str) -> np.ndarray:
    t = apply_pool_stage(Tensor(tuple(x.shape), [int(v) for v in x.reshape(-1)], "i32"), "maxpool", dict(attrs), op=op)
    return np.asarray(t.data, dtype=np.int64).reshape(t.shape)


def _im2col(ifm: np.ndarray, *, kh, kw, ci, stride, padding, dilation) -> np.ndarray:
    """``commandbuffer.conv_im2col``: [N*Ho*Wo, Kh*Kw*Ci], columns in (kh, kw, ci) order, zero pad."""
    n, h, w, c = ifm.shape
    if c != ci:
        raise KernelReferenceError(f"conv channel mismatch: ifm C={c} != ci={ci}")
    sh, sw = stride
    pt, pl, pb, pr = padding
    dh, dw = dilation
    ho, wo = conv_out_dims(h, w, kh, kw, stride, padding, dilation)
    padded = np.zeros((n, h + pt + pb + kh * dh, w + pl + pr + kw * dw, c), dtype=np.int64)
    padded[:, pt : pt + h, pl : pl + w, :] = ifm
    cols = np.empty((n, ho, wo, kh, kw, c), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            y0, x0 = ky * dh, kx * dw
            cols[:, :, :, ky, kx, :] = padded[:, y0 : y0 + sh * (ho - 1) + 1 : sh, x0 : x0 + sw * (wo - 1) + 1 : sw, :]
    return cols.reshape(n * ho * wo, kh * kw * c)


def _epilogue(x, stages, attrs, env, ops, default_shift, op) -> np.ndarray:
    for stage in stages:
        if stage in BIAS_STAGES:
            bias = np.asarray(env[bias_tensor_name(ops, attrs, op=op)], dtype=np.int64).reshape(-1)
            if bias.shape != (x.shape[-1],):
                raise KernelReferenceError(f"{op}: bias {bias.shape} != ({x.shape[-1]},)")
            x = x + bias[None, :]
        elif stage == "acc_scale":
            scale = (
                declared_acc_scale_by_column(attrs, x.shape[-1], op=op)
                if "acc_scale_by_column" in attrs
                else declared_acc_scale(attrs, op=op)
            )
            x = _acc_scale(x, scale)
        elif stage == "requant":
            x = _requant(x, int(attrs.get("requant_shift", default_shift)))
        elif stage == "relu":
            x = np.maximum(x, 0)
        elif stage == "maxpool":
            x = _pool(x, attrs, op)
        else:
            raise KernelReferenceError(f"{op} declares epilogue stage {stage!r}, which the reference does not define")
    return _narrow(x, str(attrs.get("output_dtype", "i32")))


def _conv(cb, env, ops, attrs, default_shift) -> np.ndarray:
    unknown = sorted(set(attrs) - REF._CONV2D_ATTRS)
    if unknown:
        raise KernelReferenceError(f"CONV2D attribute(s) {unknown} are not defined by the reference")
    if str(attrs.get("layout", "nhwc")) != "nhwc":
        raise KernelReferenceError("CONV2D layout must be nhwc")
    kh, kw, ci, co = (int(v) for v in attrs["kernel"])
    stride = REF._conv_geom(attrs, "stride", 2, [1, 1])
    padding = REF._conv_geom(attrs, "padding", 4, [0, 0, 0, 0])
    dilation = REF._conv_geom(attrs, "dilation", 2, [1, 1])
    ifm = np.asarray(env[ops["ifm"]], dtype=np.int64)
    declared = ((cb.get("tensors") or {}).get(ops["ifm"]) or {}).get("shape")
    if ifm.ndim == 2 and declared is not None:
        want = [int(v) for v in declared]
        if len(want) != 4 or int(np.prod(want[:-1])) != ifm.shape[0] or want[-1] != ifm.shape[1]:
            raise KernelReferenceError(f"CONV2D flat activation {ifm.shape} is not the declared {want}")
        ifm = ifm.reshape(want)
    if ifm.ndim != 4:
        raise KernelReferenceError(f"CONV2D activation {ifm.shape} is not rank-4 NHWC")
    weight = np.asarray(env[ops["weight"]], dtype=np.int64)
    if weight.shape != (kh * kw * ci, co):
        raise KernelReferenceError(f"CONV2D weight {weight.shape} is not [Kh*Kw*Ci, Co] = [{kh * kw * ci}, {co}]")
    cols = _im2col(ifm, kh=kh, kw=kw, ci=ci, stride=stride, padding=padding, dilation=dilation)
    result = _int_matmul(cols, weight)
    stages = list(attrs.get("epilogue", []))
    if "maxpool" in stages:
        ho, wo = conv_out_dims(ifm.shape[1], ifm.shape[2], kh, kw, stride, padding, dilation)
        if tuple(int(v) for v in attrs.get("pool_in_dims", ())) != (ho, wo):
            raise KernelReferenceError("CONV2D pool_in_dims disagrees with the convolution's output extent")
    return _epilogue(result, stages, attrs, env, ops, default_shift, f"CONV2D {ops['dst']!r}")


def _fast(cb: Mapping[str, Any], env: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    resident: dict[str, str] = {}
    accumulators: dict[str, np.ndarray] = {}
    outputs: dict[str, np.ndarray] = {}
    default_shift = int((cb.get("params") or {}).get("requant_shift", 4))
    for cmd in cb.get("commands", []):
        op, ops, attrs = cmd["opcode"], cmd.get("operands", {}), cmd.get("attributes", {})
        if op == "RES_PACK":
            if "scale" in ops:
                raise KernelReferenceError("dequantizing RES_PACK is outside the fast vocabulary")
            resident[ops["dst"]] = ops["src"]
        elif op == "EVICT":
            handle = ops.get("handle", ops.get("src"))
            if isinstance(handle, str):
                resident.pop(handle, None)
        elif op in ("MATMUL", "MATMUL_RESIDENT"):
            rhs = env[resident.get(ops["rhs"], ops["rhs"])]
            accumulators[ops["dst"]] = _int_matmul(env[ops["lhs"]], rhs)
        elif op == "COMMIT":
            dst = ops["dst"]
            value = _epilogue(
                accumulators[ops["src"]],
                list(attrs.get("epilogue", [])),
                attrs,
                env,
                ops,
                default_shift,
                f"COMMIT {dst!r}",
            )
            outputs[dst] = env[dst] = value
        elif op == "BIAS_ADD":
            dst = ops["dst"]
            value = _epilogue(
                np.asarray(env[ops["src"]], dtype=np.int64),
                ["bias_add"],
                attrs,
                env,
                ops,
                default_shift,
                f"BIAS_ADD {dst!r}",
            )
            outputs[dst] = env[dst] = value
        elif op == "CONV2D":
            if ops["weight"] in resident:
                ops = {**ops, "weight": resident[ops["weight"]]}
            outputs[ops["dst"]] = env[ops["dst"]] = _conv(cb, env, ops, attrs, default_shift)
        elif op == "BATCHED_MATMUL":
            from merlin.runtime.engine.command_buffer import batched_matmul_geometry

            a, w, dst = np.asarray(env[ops["a"]]), np.asarray(env[ops["w"]]), ops["dst"]
            spec = (cb.get("tensors") or {}).get(dst)
            if not isinstance(spec, dict):
                raise KernelReferenceError(f"BATCHED_MATMUL {dst!r}: destination must name a declared tensor")
            g = batched_matmul_geometry(a.shape, w.shape, spec.get("shape"), op=f"BATCHED_MATMUL {dst!r}")
            av = a.reshape(g.batch_count, g.m, g.k)
            wv = w.reshape(g.batch_count, g.k, g.n)
            value = np.stack([_int_matmul(av[i], wv[i]) for i in range(g.batch_count)]).reshape(g.output_shape)
            outputs[dst] = env[dst] = value
        elif op == "MOVEMENT":
            outputs[ops["dst"]] = env[ops["dst"]] = np.asarray(env[ops["src"]])
        else:  # pragma: no cover - guarded by FAST_OPCODES
            raise KernelReferenceError(f"opcode {op} is outside the fast vocabulary")
    for name, spec in (cb.get("tensors") or {}).items():
        if spec.get("role") == "output" and name in env and name not in outputs:
            outputs[name] = np.asarray(env[name])
    declared = cb.get("outputs")
    if declared:
        outputs = {k: v for k, v in outputs.items() if k in set(declared)}
    return outputs


def evaluate_kernel(cb: Mapping[str, Any], inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every committed output of an interface command buffer, on fully supplied leaf inputs."""
    params = cb.get("params") or {}
    opcodes = {c.get("opcode") for c in cb.get("commands", [])}
    env = {name: np.asarray(value) for name, value in inputs.items()}
    if opcodes <= FAST_OPCODES and not any(params.get(key) for key in DERIVATION_RECIPE_KEYS):
        return _fast(cb, env)
    slow = REF.reference_outputs(dict(cb), {k: np.asarray(v).tolist() for k, v in env.items()})
    return {k: np.asarray(v, dtype=np.int64) for k, v in slow.items()}


def call_outputs(
    interface: str, operands: Sequence[np.ndarray], roles: Sequence[str] | None = None
) -> list[np.ndarray]:
    """What the kernel for ``interface`` must return for one host call's operands, in result order.

    ``roles`` names each operand's logical role in the host call's order (the rewrite's
    ``call_buffers``: ``input_<k>``, ``weight_<k>``, ``bias_<k>``, ``out_<k>``); without it the operands
    are taken in the kernel's own argument order
    (:func:`merlin.compiler.lower.device.device_shim.logical_buffers`). Destination values are ignored.
    """
    from merlin.compiler.lower.device.device_shim import logical_buffers
    from merlin.evaluation.model.linalg_semantics import storage_dtype
    from merlin.target.contract.interfaces.grammar import parse_interface_mlir

    buffers, why = logical_buffers(interface)
    if buffers is None:
        raise KernelReferenceError(why)
    if len(operands) != len(buffers):
        raise KernelReferenceError(f"the call passes {len(operands)} operand(s); the interface has {len(buffers)}")
    cb = parse_interface_mlir(interface)
    if roles is not None:
        by_role = {b.role: b for b in buffers}
        if len(roles) != len(operands) or set(roles) != set(by_role):
            raise KernelReferenceError(f"call roles {list(roles)} do not match the interface's {sorted(by_role)}")
        pairs = [(by_role[role], value) for role, value in zip(roles, operands, strict=True)]
    else:
        pairs = list(zip(buffers, operands, strict=True))
    inputs: dict[str, np.ndarray] = {}
    for buffer, value in pairs:
        if buffer.kind != "input":
            continue
        array = np.asarray(value)
        if tuple(array.shape) != buffer.shape and array.size == int(np.prod(buffer.shape)):
            array = array.reshape(buffer.shape)
        if tuple(array.shape) != buffer.shape:
            raise KernelReferenceError(f"operand {buffer.name}: shape {array.shape} != interface {buffer.shape}")
        inputs[buffer.name] = array
    produced = evaluate_kernel(cb, inputs)
    results: list[np.ndarray] = []
    for buffer in buffers:
        if buffer.kind != "output":
            continue
        if buffer.name not in produced:
            raise KernelReferenceError(f"interface result {buffer.name!r} was not committed")
        value = np.asarray(produced[buffer.name]).reshape(buffer.shape)
        results.append(value.astype(storage_dtype(buffer.dtype)))
    return results


# ---- catalog -----------------------------------------------------------------------------------
# Everything about one captured program that does not depend on a submission, computed once.
#
# Device grouping is Merlin's: which source groups are routed, the interface each one is stated as, and
# the host call that passes it its operands are all fixed by the capture and the target, never by the
# package that lowers them. A program's catalog therefore records:
#
# - the INT8 reference (the prepared quantized program, :func:`merlin.evaluation.model.linalg_evaluator.evaluate`d)
#   and the device-rewritten program, both prepared exactly as a whole-model build prepares them;
# - the rewrite's sidecar (signatures, stated entries, call roles) and every routed symbol's interface;
# - the offload census over the ORIGINAL captured contractions (fixed denominator);
# - per input, every device call's operands and expected results, content-addressed so a constant
#   weight shared by every input and step is stored once.
#
# A consistency check rides along: the rewritten program, with each device call answered by the
# interface reference (:mod:`merlin.evaluation.model.bundles.program_catalog`), must reproduce the INT8 reference
# bit for bit. A mismatch is a Merlin grouping/rewrite defect, reported as such and never charged to a
# submission.

if TYPE_CHECKING:
    from merlin.evaluation.model.linalg_evaluator import ConstantCache
    from merlin.runtime.dispatch.whole_model import ForwardArgsCache

CATALOG_SCHEMA = "merlin.evaluation.model.bundles.program_catalog.v1"
CALLS_SCHEMA = "merlin.evaluation.model.calls.v1"
CONTRACTION_OPS = ("linalg.matmul", "linalg.batch_matmul")


class CatalogError(RuntimeError):
    """A program could not be prepared or its reference is inconsistent."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    """Content-addressed arrays (``<root>/<sha256>.npy``): one copy of every distinct operand."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, array: np.ndarray) -> str:
        array = np.ascontiguousarray(array)
        header = f"{array.dtype.str}|{array.shape}".encode()
        key = _sha(header + array.tobytes())
        path = self.root / f"{key}.npy"
        if not path.exists():
            # A writer's own temporary name: several processes may store the same operand at once.
            tmp = path.with_name(f"{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy")
            np.save(tmp, array, allow_pickle=False)
            tmp.replace(path)
        return key

    def get(self, key: str) -> np.ndarray:
        return np.load(self.root / f"{key}.npy", allow_pickle=False)


@dataclass
class ProgramCatalog:
    """The submission-independent facts of one captured program (see the module docstring)."""

    target: str
    stage: Path
    ref_prepared: Path
    dev_prepared: Path
    routing: dict[str, Any]
    sidecar: dict[str, Any]
    interfaces: dict[str, str]

    def roles(self, symbol: str) -> tuple[str, ...]:
        return tuple(b["role"] for b in self.sidecar["call_buffers"][symbol])

    def batch_shape(self, symbol: str) -> tuple[int, ...]:
        """The leading batch of ``symbol``'s host call; its interface states ONE slice (``()``: unbatched)."""
        program = (self.sidecar.get("programs") or {}).get(symbol) or {}
        return tuple(int(v) for v in program.get("batch_shape") or ())

    def record(self) -> dict[str, Any]:
        return {
            "schema": CATALOG_SCHEMA,
            "target": self.target,
            "stage": str(self.stage),
            "ref_prepared": str(self.ref_prepared),
            "dev_prepared": str(self.dev_prepared),
            "routing": self.routing,
            "interfaces": self.interfaces,
        }

    @classmethod
    def load(cls, directory: Path) -> ProgramCatalog:
        doc = json.loads((directory / "catalog.json").read_text())
        if doc.get("schema") != CATALOG_SCHEMA:
            raise CatalogError(f"{directory}: not a {CATALOG_SCHEMA} catalog")
        sidecar = json.loads((directory / "dev" / "device_signatures.json").read_text())
        return cls(
            target=doc["target"],
            stage=Path(doc["stage"]),
            ref_prepared=Path(doc["ref_prepared"]),
            dev_prepared=Path(doc["dev_prepared"]),
            routing=doc["routing"],
            sidecar=sidecar,
            interfaces=doc["interfaces"],
        )


def prepare(stage: Path, directory: Path, *, target: str) -> ProgramCatalog:
    """Prepare ``stage`` (a capture directory with ``model.mlir``) with and without device groups."""
    from merlin.compiler.build.model_preparation import _prepare_model_mlir
    from merlin.compiler.planning.complete import routing

    stage, directory = Path(stage), Path(directory)
    (directory / "ref").mkdir(parents=True, exist_ok=True)
    (directory / "dev").mkdir(parents=True, exist_ok=True)
    ref = _prepare_model_mlir(
        stage / "model.mlir", directory / "ref", int8_compute=True, respect_captured_quantization_scope=True
    )
    try:
        route = routing(
            target,
            (stage / "linalg.mlir").read_text(),
            str(stage),
            capture=str(stage),
            model=stage.name,
            linked_elf_admission=None,
        )
    except ValueError as error:
        if "no admitted accelerator groups" not in str(error):
            raise
            # A program with nothing to route (a pure slice, say) runs on the host only: its rewritten form IS
            # its reference, and it has no device call to judge.
        sidecar = {"signatures": {}, "entries": {}, "call_buffers": {}, "routed": [], "host_only": str(error)}
        (directory / "dev" / "device_signatures.json").write_text(json.dumps(sidecar, indent=1) + "\n")
        catalog = ProgramCatalog(
            target,
            stage,
            Path(ref),
            Path(ref),
            {"operand_dtype": None, "accum_dtype": None, "numeric_policy": None},
            sidecar,
            {},
        )
        (directory / "catalog.json").write_text(json.dumps(catalog.record(), indent=1) + "\n")
        return catalog
    dev = _prepare_model_mlir(
        stage / "model.mlir",
        directory / "dev",
        int8_compute=True,
        respect_captured_quantization_scope=True,
        complete_group_device=route,
    )
    sidecar = json.loads((directory / "dev" / "device_signatures.json").read_text())
    routing_record = {
        "operand_dtype": route.operand_dtype,
        "accum_dtype": route.accum_dtype,
        "numeric_policy": route.numeric_policy,
    }
    interfaces = {
        symbol: interface_for(entry, target=target, **routing_record) for symbol, entry in sidecar["entries"].items()
    }
    catalog = ProgramCatalog(target, stage, Path(ref), Path(dev), routing_record, sidecar, interfaces)
    (directory / "catalog.json").write_text(json.dumps(catalog.record(), indent=1) + "\n")
    return catalog


def _function_arity(module) -> int:
    fn = next(op for op in module.walk() if op.name == "func.func" and op.body.blocks)
    return len(fn.body.blocks[0].args)


@dataclass
class Evaluation:
    reference: list[np.ndarray]
    rewritten: list[np.ndarray]
    calls: list[dict[str, Any]]

    def consistent(self) -> list[dict[str, Any]]:
        rows = []
        for index, (a, b) in enumerate(zip(self.reference, self.rewritten, strict=True)):
            same = a.shape == b.shape and np.array_equal(a, b)
            rows.append({"output": index, "shape": list(a.shape), "exact": bool(same)})
        return rows


def _answers(
    catalog: ProgramCatalog,
    calls: list[dict[str, Any]],
    *,
    blobs: BlobStore | None,
    keep_call: Callable[[str, int], bool] | None,
) -> dict[str, Callable]:
    """Each routed symbol's host call answered by its interface reference, recording the device calls.

    A BATCHED host call is a loop over its slices, one device call per slice: the interface states one
    slice and the host shim calls the per-slice kernel once for each (``device_shim``'s logical ABI). So
    each slice is evaluated, recorded and numbered as its own device call, and the host receives the
    slices' results stacked back under the batch.
    """
    counts: dict[str, int] = {}

    def answer(symbol: str):
        roles = catalog.roles(symbol)
        batch = catalog.batch_shape(symbol)

        def device_call(values):
            results = call_outputs(catalog.interfaces[symbol], values, roles)
            ordinal = counts.get(symbol, 0)
            counts[symbol] = ordinal + 1
            if blobs is not None and (keep_call is None or keep_call(symbol, ordinal)):
                calls.append(
                    {
                        "symbol": symbol,
                        "ordinal": ordinal,
                        "roles": list(roles),
                        "operands": [blobs.put(np.asarray(v)) for v in values],
                        "expected": [blobs.put(np.asarray(r)) for r in results],
                    }
                )
            return results

        def call(values):
            if not batch:
                return device_call(values)
            count = math.prod(batch)
            arrays = [np.asarray(v) for v in values]
            if any(tuple(a.shape[: len(batch)]) != batch for a in arrays):
                raise CatalogError(f"{symbol}: a host operand does not lead with the batch {list(batch)}")
            sliced = [a.reshape(count, *a.shape[len(batch) :]) for a in arrays]
            per_slice = [device_call([a[index] for a in sliced]) for index in range(count)]
            return [
                np.stack([results[j] for results in per_slice]).reshape(*batch, *per_slice[0][j].shape)
                for j in range(len(per_slice[0]))
            ]

        return call

    return {symbol: answer(symbol) for symbol in catalog.interfaces}


def evaluate(
    catalog: ProgramCatalog,
    arguments: Sequence[np.ndarray],
    *,
    blobs: BlobStore | None = None,
    keep_call: Callable[[str, int], bool] | None = None,
) -> Evaluation:
    """Both programs on ``arguments``; every (kept) device call's operands and expected results."""
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.evaluation.model import linalg_evaluator as LN

    ref = parse_mlir_file(catalog.ref_prepared)
    dev = parse_mlir_file(catalog.dev_prepared)
    reference = LN.evaluate(ref, list(arguments)[: _function_arity(ref)])
    calls: list[dict[str, Any]] = []
    rewritten = LN.evaluate(
        dev,
        list(arguments)[: _function_arity(dev)],
        calls=_answers(catalog, calls, blobs=blobs, keep_call=keep_call),
    )
    return Evaluation([np.asarray(v) for v in reference], [np.asarray(v) for v in rewritten], calls)


def capture_arguments(
    stage: Path,
    inputs: Mapping[str, np.ndarray] | None,
    scratch: Path,
    *,
    weights_cache: ForwardArgsCache | None = None,
) -> list[np.ndarray]:
    """Every ``@forward`` argument of ``stage``, with its model inputs replaced by ``inputs`` (``in0``...).

    Weights, buffers and lifted constants still come from the capture: the call binds a shadow of the
    capture directory whose files are links to the capture's, except ``inputs.npz``. With
    ``weights_cache`` they are read once per cache, and every call returns the same read-only arrays.
    """
    from merlin.runtime.dispatch.whole_model import resolve_forward_args

    stage = Path(stage)
    if inputs is None:
        return resolve_forward_args(stage, cache=weights_cache)
    shadow = Path(scratch)
    shadow.mkdir(parents=True, exist_ok=True)
    for item in stage.iterdir():
        link = shadow / item.name
        if item.name == "inputs.npz" or link.exists() or link.is_symlink():
            continue
        link.symlink_to(item.resolve())
    expected = np.load(stage / "inputs.npz")
    missing = sorted(set(expected.files) - set(inputs))
    if missing:
        raise CatalogError(f"inputs for {stage.name} lack {missing}")
    for key in expected.files:
        if tuple(np.asarray(inputs[key]).shape) != tuple(expected[key].shape):
            raise CatalogError(f"input {key}: shape {np.asarray(inputs[key]).shape} != capture {expected[key].shape}")
    np.savez(shadow / "inputs.npz", **{k: np.asarray(inputs[k]) for k in expected.files})
    return resolve_forward_args(shadow, cache=weights_cache)


def record_calls(
    catalog: ProgramCatalog,
    arguments: Sequence[np.ndarray],
    *,
    blobs: BlobStore,
    keep_call: Callable[[str, int], bool] | None = None,
    module: Any = None,
    constants: ConstantCache | None = None,
) -> list[dict[str, Any]]:
    """The device calls of the rewritten program on ``arguments`` (operands and expected results).

    ``module`` is the parsed ``dev_prepared`` when the caller keeps it; ``constants`` is passed to
    :func:`linalg_evaluator.evaluate` (a device call is never hoisted, so every call is still recorded)."""
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.evaluation.model import linalg_evaluator as LN

    dev = module if module is not None else parse_mlir_file(catalog.dev_prepared)
    calls: list[dict[str, Any]] = []
    LN.evaluate(
        dev,
        list(arguments)[: _function_arity(dev)],
        calls=_answers(catalog, calls, blobs=blobs, keep_call=keep_call),
        constants=constants,
    )
    return calls

    # ------------------------------------------------------------------------------------------- census


def _node_ids(op) -> tuple[str, ...]:
    attr = op.attributes.get("prov.source_node_ids")
    return tuple(str(getattr(item, "data", item)) for item in getattr(attr, "data", ()) or ())


def _is_contraction(op) -> bool:
    if op.name in CONTRACTION_OPS or op.name.startswith("linalg.conv_"):
        return True
    if op.name != "linalg.generic":
        return False
    iterators = str(op.properties.get("iterator_types", ""))
    if "reduction" not in iterators:
        return False
    body = [inner.name for inner in op.regions[0].blocks[0].ops]
    return any(n in ("arith.mulf", "arith.muli") for n in body) and any(n in ("arith.addf", "arith.addi") for n in body)


def _macs(op) -> int:
    shapes = [list(getattr(v.type, "get_shape", lambda: [])()) for v in op.operands]
    out = list(getattr(op.results[0].type, "get_shape", lambda: [])()) if op.results else []
    if op.name == "linalg.matmul" and len(shapes) >= 2 and len(shapes[0]) == 2:
        return int(shapes[0][0] * shapes[0][1] * shapes[1][-1])
    if op.name == "linalg.batch_matmul" and len(shapes[0]) == 3:
        return int(shapes[0][0] * shapes[0][1] * shapes[0][2] * shapes[1][-1])
    if op.name.startswith("linalg.conv_") and len(shapes) >= 2 and out:
        # Output elements times the filter's reduced extent (every filter dim but its output channels).
        filt = shapes[1]
        return int(math.prod(out)) * int(math.prod(filt)) // max(1, int(out[-1] if "hwcf" in op.name else filt[0]))
        # A generic: the size of its iteration domain, each loop dim's extent read off the operand that
        # indexes it directly (one multiply-accumulate per point of the domain).
    maps = getattr(op.properties.get("indexing_maps"), "data", ())
    extent: dict[int, int] = {}
    for amap, shape in zip(maps, shapes, strict=False):
        for axis, expr in enumerate(getattr(amap.data, "results", ())):
            position = getattr(expr, "position", None)
            if position is not None and axis < len(shape):
                extent[position] = int(shape[axis])
    return int(math.prod(extent.values())) if extent else 0


def _dtype(value) -> str:
    element = (
        getattr(getattr(value, "type", None), "element_type", None)
        or getattr(value.type, "get_element_type", lambda: None)()
    )
    return str(element)


def census(catalog: ProgramCatalog, *, refused: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every contraction of the original capture, classified ``device`` or ``host:<reason>``."""
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.compiler.lower.options.quantization_scope import captured_quantized_contraction

    module = parse_mlir_file(catalog.stage / "model.mlir")
    routed: dict[str, dict[str, Any]] = {}
    for row in catalog.sidecar.get("routed") or ():
        for node in row.get("source_node_ids") or ():
            routed[str(node)] = row
    refused = dict(refused or {})
    rows: list[dict[str, Any]] = []
    for op in module.walk():
        if not _is_contraction(op):
            continue
        nodes = _node_ids(op)
        dtypes = [_dtype(v) for v in list(op.operands)[:2]]
        row = next((routed[n] for n in nodes if n in routed), None)
        # A contraction the capture already integerized (int x int operands) was quantized by the recipe
        # even though no dequantize feeds it; left on the host, that is a policy decision, not the recipe's.
        integer = all(d[:1] in ("i", "u") and d[1:].isdigit() for d in dtypes)
        if row is not None and str(row.get("symbol")) not in refused:
            placement = "device"
        elif row is not None:
            placement = "host:compiler_refused"
        elif not (integer or captured_quantized_contraction(op)):
            placement = "host:not_quantized_by_recipe"
        else:
            placement = "host:policy_refused"
        rows.append(
            {
                "op": op.name,
                "nodes": list(nodes),
                "dtypes": dtypes,
                "macs": _macs(op),
                "placement": placement,
                "symbol": row.get("symbol") if row else None,
                "source_region_id": row.get("source_region_id") if row else None,
            }
        )
    total_macs = sum(r["macs"] for r in rows) or 1
    counts: dict[str, int] = {}
    macs: dict[str, int] = {}
    for r in rows:
        counts[r["placement"]] = counts.get(r["placement"], 0) + 1
        macs[r["placement"]] = macs.get(r["placement"], 0) + r["macs"]
    return {
        "denominator": len(rows),
        "device": counts.get("device", 0),
        "host": {k: v for k, v in sorted(counts.items()) if k != "device"},
        "device_mac_fraction": round(macs.get("device", 0) / total_macs, 6),
        "host_mac_fraction": {k: round(v / total_macs, 6) for k, v in sorted(macs.items()) if k != "device"},
        "contractions": rows,
    }


def write_calls(path: Path, calls: Sequence[Mapping[str, Any]], **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": CALLS_SCHEMA, **meta, "calls": list(calls)}, indent=1) + "\n")


def read_calls(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != CALLS_SCHEMA:
        raise CatalogError(f"{path}: not a {CALLS_SCHEMA} document")
    return doc
