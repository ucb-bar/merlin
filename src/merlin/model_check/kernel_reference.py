"""The expected result of one device-kernel call: the group's interface program evaluated in numpy.

A device group is built from its *stated program* (:func:`merlin.llvmlower.device_build.kernel_entry`):
the entry is restated as an interface module (``corpus_spec.build``), and the package lowers that
module. The kernel is therefore correct exactly when, on the host call's operands, it returns what the
interface program computes. :func:`merlin.runtime.reference.reference_outputs` defines that computation
on Merlin's integer ``Tensor``; this module computes the same function with numpy so that a whole
model's calls can be answered in seconds:

- contractions (``MATMUL``/``MATMUL_RESIDENT``/``CONV2D``/``BATCHED_MATMUL``) sum int8 x int8 exactly
  (float64 products and sums are exact far beyond any reduction length a call carries);
- ``acc_scale`` multiplies in float32 and rounds to nearest, ties to even, as
  ``Tensor.requant_acc_scale``; ``requant`` is the rounding arithmetic shift of ``Tensor.requant``;
- pooling and anything outside this vocabulary are computed by the reference engine itself, so no
  second definition of those exists.

The two engines are kept equal by test (``merlin/tests/runtime/test_model_check_kernel_reference.py``)
and, per model, by evaluating the device-rewritten program with these answers against the unrewritten
INT8 reference (:mod:`merlin.model_check.catalog`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from merlin.runtime import reference as REF
from merlin.runtime.commandbuffer import (
    BIAS_STAGES,
    DERIVATION_RECIPE_KEYS,
    apply_pool_stage,
    bias_tensor_name,
    conv_out_dims,
    declared_acc_scale,
)
from merlin.runtime.tensor import Tensor

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
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.targetgen import corpus_spec as CS

    binding = _mesh_tile_binding(target, operand_dtype, accum_dtype, numeric_policy=numeric_policy)
    _capsule, interface = CS.build(dict(entry), binding)
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


def _acc_scale(x: np.ndarray, scale: float) -> np.ndarray:
    prod = np.asarray(x, dtype=np.int64).astype(np.float32) * np.float32(scale)
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
            scale = declared_acc_scale(attrs, op=op) if op.startswith("CONV2D") else float(attrs.get("acc_scale", 1.0))
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
            from merlin.runtime.commandbuffer import batched_matmul_geometry

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


def evaluate(cb: Mapping[str, Any], inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
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
    (:func:`merlin.llvmlower.device_shim.logical_buffers`). Destination values are ignored.
    """
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.runtime.linalg_numpy import storage_dtype
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

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
    produced = evaluate(cb, inputs)
    results: list[np.ndarray] = []
    for buffer in buffers:
        if buffer.kind != "output":
            continue
        if buffer.name not in produced:
            raise KernelReferenceError(f"interface result {buffer.name!r} was not committed")
        value = np.asarray(produced[buffer.name]).reshape(buffer.shape)
        results.append(value.astype(storage_dtype(buffer.dtype)))
    return results
