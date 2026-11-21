"""Offline prepack: what a model's closed groups need from its weights, computed once.

A closed integer group adds its bias in the accumulator's domain, so the bias the model stores in
floating point has to become ``roundeven(b / (s_x * s_w))``. That is a function of the weights file
and two compile-time scales. It does not depend on the input, so it does not belong in the
inference: today such work runs on the host on every inference, per element, for every layer.

This stage reads the group plan and the capture's weights (the safetensors payload, through the
same torch-free reader the weight packer uses) and produces, per closed group, the integer bias
and the requantization multiplier's exact single-precision bits. A value that would not fit the
accumulator is a refusal naming the group, never a wrapped integer.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

from xdsl.dialects import builtin, func, tensor
from xdsl.dialects.linalg.ops import TransposeOp, YieldOp

from merlin.compiler.ir import mlir_query as mq
from merlin.compiler.partition.groups import compute_groups as CG
from merlin.compiler.partition.groups import group_numerics as GN

# ---- group_prepack -----------------------------------------------------------------------------

SCHEMA = "group_prepack_v1"
_NUMPY_DTYPE = {"F32": "<f4", "F64": "<f8", "F16": "<f2"}


_INTEGER_DTYPE = {"I8": "<i1", "U8": "<u1", "I16": "<i2", "I32": "<i4"}


def _stored_raw(arg_index: int, manifest: dict, safetensors_path: Path):
    """``(name, array)`` of the stored tensor behind a model argument, in its own dtype and shape."""
    import numpy as np

    from merlin.compiler.lower.c.weights import load_safetensors_header

    meta = manifest.get(str(arg_index))
    if not isinstance(meta, dict) or meta.get("kind") == "input":
        raise GN.GroupNumericsError(f"model argument {arg_index} is not a stored tensor")
    header, payload_offset = load_safetensors_header(safetensors_path)
    name = meta.get("weight") or meta.get("name")
    if name not in header:
        raise GN.GroupNumericsError(f"stored tensor {name!r} is not in the weights file")
    spelled = header[name]["dtype"]
    dtype = _NUMPY_DTYPE.get(spelled) or _INTEGER_DTYPE.get(spelled)
    if dtype is None:
        raise GN.GroupNumericsError(f"stored tensor {name!r} has dtype {spelled}, which this reader does not read")
    begin, end = header[name]["data_offsets"]
    with open(safetensors_path, "rb") as handle:
        handle.seek(payload_offset + begin)
        raw = handle.read(end - begin)
    array = np.frombuffer(raw, dtype=dtype)
    shape = [int(v) for v in header[name].get("shape") or ()]
    return name, (array.reshape(shape) if shape else array), spelled


def _stored_tensor(arg_index: int, manifest: dict, safetensors_path: Path):
    """The stored FLOATING-POINT tensor behind a model argument, flat, or a reason it cannot be read."""
    import numpy as np

    name, array, spelled = _stored_raw(arg_index, manifest, safetensors_path)
    if spelled not in _NUMPY_DTYPE:
        raise GN.GroupNumericsError(
            f"stored tensor {name!r} is {spelled}; a bias is folded from a floating-point tensor"
        )
    return name, array.reshape(-1).astype(np.float64)


def as_the_contraction_reads_it(group: CG.Group, operand_index: int, stored):
    """``stored`` carried through the views and permutations between it and the contraction.

    Evaluated, not pattern-matched: each reshape and each permutation on the way is applied to the
    array, so the result is the matrix the capture's contraction multiplies, whatever the framework
    did to get there. Anything on the way that is not a view or a permutation is refused.
    """
    import numpy as np

    operand = list(group.root.operands)[operand_index]
    adapters, dequantize, _dtype = CG._input_chain(operand)
    value = np.asarray(stored)
    between = []  # from the stored tensor toward the dequantize: views and movement only
    # A contraction that reads the INTEGERS directly (an int8 capture, no dequantize in between) still
    # reads them through whatever relayout the capture put there -- a torch linear's [out, in] weight
    # transposed to [in, out] -- so the walk starts at the operand itself. Skipping it laid the weight
    # out as stored: the right bytes for a square matrix only by accident, and the wrong shape otherwise.
    cursor = dequantize.operands[0] if dequantize is not None else operand
    while cursor is not None and getattr(cursor, "owner", None) is not None and hasattr(cursor.owner, "operands"):
        if not cursor.owner.operands:
            break
        between.append(cursor.owner)
        cursor = cursor.owner.operands[0]
    for op in [*reversed(between), *reversed(adapters)]:
        name = mq.op_name(op)
        shape, _ = mq.type_shape_dtype(op.results[0].type)
        if name in CG._VIEW_OPS:
            value = value.reshape([int(v) for v in shape])
        elif name == "linalg.transpose":
            value = np.transpose(value, CG._int_array(op, "permutation"))
        else:
            raise GN.GroupNumericsError(f"{name} between a stored tensor and its contraction is not a layout step")
        if list(value.shape) != [int(v) for v in shape]:
            raise GN.GroupNumericsError(f"{name} did not produce the shape the capture declares")
    return value


def device_weight(group: CG.Group, stated, stored):
    """The stored tensor laid out the way the group's device program says a unit holds it.

    ``stated`` is the group's :class:`~.group_command.GroupProgram`. A unit's contraction is
    ``A[M, K] @ W[K, N]``: the weight is transposed when the capture had it on the left, and a
    convolution's reduced axis is reordered from the capture's gather order to the command's
    ``[tap_h, tap_w, channel]`` packing. Done once, offline; per inference this is work the host
    repeats today.
    """
    import numpy as np

    matrix = as_the_contraction_reads_it(group, stated.stored_operand, stored)
    if matrix.ndim != 2:
        raise GN.GroupNumericsError("the contraction's stored operand is not a matrix")
    device = matrix.T if stated.transposed else matrix  # [K, N]
    entry = stated.entry
    if entry["op"] == "conv2d" and stated.column_order is not None:
        extent = {"channel": int(entry["ci"]), "tap_h": int(entry["kh"]), "tap_w": int(entry["kw"])}
        have = list(stated.column_order)
        want = ["tap_h", "tap_w", "channel"]
        if sorted(have) != sorted(want):
            raise GN.GroupNumericsError(f"unknown gather order {have}")
        split = device.reshape([extent[axis] for axis in have] + [device.shape[1]])
        device = np.transpose(split, [have.index(axis) for axis in want] + [3]).reshape(device.shape)
    return np.ascontiguousarray(device)


def _f32_bits(value: float) -> int:
    """The bits a scale register takes: one IEEE-754 single, no re-rounding downstream."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _scalar_row(group: CG.Group) -> dict[str, Any]:
    """A group with nothing to fold: an integer sum or a window mean, stated as register values.

    A sum's two multipliers go through the LOADS after the factor its readout carries is divided
    out of them, which is how the group's program is issued; a mean has one readout multiplier.
    """
    row: dict[str, Any] = {"group": group.index, "stages": list(group.stages)}
    if group.operand_sum is not None:
        factor = float(group.operand_sum.get("readout_factor") or 1.0)
        loads = [float(group.operand_sum[key]) / factor for key in ("lhs_scale", "rhs_scale")]
        row["operand_sum"] = {
            "load_multipliers": loads,
            "load_multipliers_f32_bits": [_f32_bits(value) for value in loads],
            "readout_multiplier": factor,
            "readout_multiplier_f32_bits": _f32_bits(factor),
            "bound_lsb": int(group.operand_sum["bound_lsb"]),
            "activation": "relu" if group.operand_sum.get("relu") else "none",
        }
        return row
    multiplier = float(group.window_mean["multiplier"])
    row["window_mean"] = {
        "rows": int(group.window_mean["rows"]),
        "window": int(group.window_mean["window"]),
        "multiplier": multiplier,
        "multiplier_f32_bits": _f32_bits(multiplier),
        "stored_operand": "a constant one for every element of the window",
        "bound_lsb": int(group.window_mean["bound_lsb"]),
    }
    return row


def module_extents(groups: Sequence[CG.Group]) -> dict[int, tuple] | None:
    """The contraction-extent table of the one module ``groups`` were formed over, read ONCE, so a
    caller stating every group does not walk the whole module per group. ``None`` (each group reads
    its own) when the groups do not name one module."""
    from merlin.compiler.ir.contraction_shapes import contraction_extents

    modules = {
        id(module): module
        for module in (
            g.root.parent_op().parent_op() for g in groups if g.root is not None and g.root.parent_op() is not None
        )
    }
    if len(modules) != 1:
        return None
    return contraction_extents(next(iter(modules.values())))


def prepack(
    groups: Sequence[CG.Group],
    manifest_path: str | Path,
    safetensors_path: str | Path,
    *,
    accumulator_bits: int = 32,
    device_layout: bool = False,
) -> dict[str, Any]:
    """Fold every closed group's bias and state its multiplier. Returns the record and the arrays.

    With ``device_layout`` each group's stored integer weight is also laid out the way its device
    program holds it (:func:`device_weight`); a group that cannot be restated, or whose weight is
    not a stored integer tensor, says why in its row and keeps its folded bias.
    """
    import numpy as np

    from merlin.compiler.partition import memory_plan as stream_plan
    from merlin.compiler.partition.groups import group_command

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    weight_args = stream_plan.weight_args_of(manifest)
    extents = module_extents(groups)
    limit = (1 << (accumulator_bits - 1)) - 1
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    arrays: dict[str, Any] = {}
    for group in groups:
        if group.placement == CG.HOST:
            continue
        if group.fusion and group.fusion.get("cut") != "bare":
            from merlin.compiler.partition.groups import group_output_program as GOP

            try:
                if accumulator_bits != 32:
                    raise GN.GroupNumericsError("the selected output statement requires its declared i32 accumulator")
                stated = group_command.program(group, weight_args=weight_args, extents=extents)
                row, folded = GOP.prepack(group, stated, manifest, Path(safetensors_path), device_layout=device_layout)
                rows.append(row)
                arrays.update(folded)
            except (CG.NoCapsuleForm, GN.GroupNumericsError) as refusal:
                skipped.append({"group": group.index, "reason": str(refusal)})
            continue
        if CG.QUANTIZE not in group.stages:
            # A GROUP THAT LEAVES AS THE ACCUMULATOR STILL HAS A DEVICE WEIGHT. This guard skipped
            # it entirely, conflating "has a quantized readout" with "has a stored operand the unit
            # reads in its own layout" -- and the second is a property of the TARGET, not of how the
            # group ends. A model's final classifier is exactly this group: it commits i32, so it
            # folds no bias by a multiplier and states no clamp, but its weight is permuted like any
            # other. Consumers that asked the prepack for it were told no row existed and declared
            # the capture's [out, in] where the contraction reads [in, out].
            if device_layout:
                row = {"group": group.index, "stages": list(group.stages), "leaves_as": "accumulator"}
                try:
                    stated = group_command.program(group, weight_args=weight_args, extents=extents)
                    name, stored, spelled = _stored_raw(stated.stored_arg, manifest, Path(safetensors_path))
                    if spelled not in _INTEGER_DTYPE:
                        raise GN.GroupNumericsError(f"stored tensor {name!r} is {spelled}, not an integer weight")
                    laid_out = device_weight(group, stated, stored)
                    key = f"group_{group.index}_weight_device"
                    arrays[key] = laid_out
                    row["weight"] = {
                        "stored_tensor": name,
                        "arg_index": stated.stored_arg,
                        "array": key,
                        "shape": [int(v) for v in laid_out.shape],
                        "transposed": stated.transposed,
                        "column_order": list(stated.column_order) if stated.column_order else None,
                        "sha256": hashlib.sha256(laid_out.tobytes()).hexdigest(),
                    }
                    # ITS BIAS IS FOLDED BY THE SAME RULE A CLOSED GROUP'S IS, and stated here so
                    # the prepack is the ONE definition of it. The group commits the accumulator, so
                    # the divisor is the product of the two operand scales rather than a readout
                    # multiplier; a consumer that folded it for itself would be a second definition
                    # free to drift from this one.
                    if stated.bias_arg is not None:
                        sources = [
                            GN._scale_source(member)
                            for member in group.members
                            if CG.classify(member).kind == CG.DEQUANTIZE
                        ]
                        if len(sources) != 2 or any(s.value is None or s.zero_point for s in sources):
                            raise GN.GroupNumericsError(
                                "this group leaves as the accumulator and its scales are not static, "
                                "so nothing can say what its bias is in the accumulator's units"
                            )
                        divisor = float(sources[0].value) * float(sources[1].value)
                        bias_name, bias = _stored_tensor(stated.bias_arg, manifest, Path(safetensors_path))
                        folded = np.rint(bias / divisor).astype(np.int32)
                        bias_key = f"group_{group.index}_bias_q"
                        arrays[bias_key] = folded
                        row["bias"] = {
                            "stored_tensor": bias_name,
                            "arg_index": stated.bias_arg,
                            "divisor": divisor,
                            "array": bias_key,
                            "elements": int(folded.size),
                            "sha256": hashlib.sha256(folded.tobytes()).hexdigest(),
                        }
                except (CG.NoCapsuleForm, GN.GroupNumericsError, TypeError) as refusal:
                    row["weight"] = {"refused": str(refusal)}
                rows.append(row)
            continue
        if group.window_pool is not None:
            rows.append({"group": group.index, "stages": list(group.stages), "window_pool": dict(group.window_pool)})
            continue
        if group.operand_sum is not None or group.window_mean is not None:
            scalar = _scalar_row(group)
            # A MEAN'S STORED OPERAND IS A REAL ARRAY, not only a sentence about one. The row has
            # always SAID the operand is "a constant one for every element of the window"; it just
            # never supplied it, so a consumer resolving `arrays[row["weight"]["array"]]` found
            # nothing and the group fell back with the prepack blamed for holding no device weight.
            # The contraction is the window summed and then scaled by the readout's reciprocal
            # count, so the operand is a column of ones -- derived from the group's own window
            # extent, and stated in the same words every other weight row uses.
            if device_layout and group.window_mean is not None:
                import numpy as np

                window = int(group.window_mean["window"])
                ones = np.ones((window, 1), dtype=np.int8)
                key = f"group_{group.index}_weight_device"
                arrays[key] = ones
                scalar["weight"] = {
                    "stored_tensor": None,
                    "arg_index": None,
                    "array": key,
                    "shape": [window, 1],
                    "transposed": False,
                    "column_order": None,
                    "sha256": hashlib.sha256(ones.tobytes()).hexdigest(),
                    "derived": "a constant one per window element; the mean's divisor is the readout's",
                }
            rows.append(scalar)
            continue
        try:
            numerics = GN.numerics_of(group)
            if numerics.multiplier is None:
                raise GN.GroupNumericsError(
                    "a scale of this group is a stored tensor; a per-axis multiplier is not folded here"
                )
            row: dict[str, Any] = {
                "group": group.index,
                "stages": list(group.stages),
                "multiplier": numerics.multiplier,
                # The bits a readout register takes: one IEEE-754 single, no re-rounding downstream.
                "multiplier_f32_bits": struct.unpack("<I", struct.pack("<f", numerics.multiplier))[0],
                "clamp": list(numerics.clamp),
                "activation": numerics.activation,
            }
            if numerics.bias_arg_index is not None:
                name, bias = _stored_tensor(numerics.bias_arg_index, manifest, Path(safetensors_path))
                from merlin.compiler.partition.groups import group_exact_float as GEF

                exact_float = GEF.needs_proof(group)
                if exact_float:
                    # The route must split a converted float chain when this
                    # source/weight-bound proof fails. A surviving fused group
                    # with no proof is a compiler error, never a silent fold.
                    proof = GEF.prove(group, manifest, safetensors_path, weight_args=weight_args)
                    if not proof["exact"]:
                        raise ValueError(
                            f"group {group.index} has no exact float readout on "
                            f"{proof['channels'] - proof['accepted_channels']} channels"
                        )
                    folded = np.asarray(proof["integer_bias"], dtype=np.int64)
                    row["multiplier"] = proof["readout_scale"]
                    row["multiplier_f32_bits"] = struct.unpack("<I", struct.pack("<f", proof["readout_scale"]))[0]
                    row["exact_float_proof"] = {
                        "schema": "group_exact_float_v1",
                        "source_weight": proof["weight"],
                        "source_bias": proof["bias"],
                        "input_bound": proof["input_bound"],
                        "channels": proof["channels"],
                        "method": proof["proof"],
                    }
                else:
                    folded = np.rint(bias / numerics.bias_divisor)
                if not np.all(np.isfinite(folded)) or np.abs(folded).max(initial=0) > limit:
                    raise GN.GroupNumericsError(
                        f"bias {name!r} folds to a value outside the {accumulator_bits}-bit "
                        f"accumulator (max |b_q| = {float(np.abs(folded).max(initial=0)):.3g})"
                    )
                folded = folded.astype(np.int64)
                key = f"group_{group.index}_bias_q"
                arrays[key] = folded.astype(np.int32)
                row["bias"] = {
                    "stored_tensor": name,
                    "arg_index": numerics.bias_arg_index,
                    "divisor": numerics.bias_divisor,
                    "fold_rule": (
                        "source_bound_exact_float_transition" if exact_float else "roundeven_bias_over_divisor"
                    ),
                    "array": key,
                    "elements": int(folded.size),
                    "max_abs": int(np.abs(folded).max(initial=0)),
                    "sha256": hashlib.sha256(arrays[key].tobytes()).hexdigest(),
                }
            if device_layout:
                try:
                    stated = group_command.program(group, weight_args=weight_args, extents=extents)
                    name, stored, spelled = _stored_raw(stated.stored_arg, manifest, Path(safetensors_path))
                    if spelled not in _INTEGER_DTYPE:
                        raise GN.GroupNumericsError(f"stored tensor {name!r} is {spelled}, not an integer weight")
                    laid_out = device_weight(group, stated, stored)
                    key = f"group_{group.index}_weight_device"
                    arrays[key] = laid_out
                    row["weight"] = {
                        "stored_tensor": name,
                        "arg_index": stated.stored_arg,
                        "array": key,
                        "shape": [int(v) for v in laid_out.shape],
                        "transposed": stated.transposed,
                        "column_order": list(stated.column_order) if stated.column_order else None,
                        "sha256": hashlib.sha256(laid_out.tobytes()).hexdigest(),
                    }
                except (CG.NoCapsuleForm, GN.GroupNumericsError) as refusal:
                    row["weight"] = {"refused": str(refusal)}
            rows.append(row)
        except GN.GroupNumericsError as refusal:
            skipped.append({"group": group.index, "reason": str(refusal)})
    return {
        "record": {
            "schema": SCHEMA,
            "accumulator_bits": accumulator_bits,
            "groups": rows,
            "skipped": skipped,
            "folded_bias_elements": sum(r.get("bias", {}).get("elements", 0) for r in rows),
            "device_weights": sum(1 for r in rows if "array" in (r.get("weight") or {})),
        },
        "arrays": arrays,
    }


def write(result: dict[str, Any], out_dir: str | Path) -> Path:
    """``prepack.json`` beside ``prepack.npz`` under ``out_dir``; returns the JSON path."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prepack.npz", **result["arrays"])
    path = out / "prepack.json"
    path.write_text(json.dumps(result["record"], indent=1, sort_keys=True), encoding="utf-8")
    return path


# ---- weight_layout -----------------------------------------------------------------------------
# Weights the model re-lays-out at run time, and whether that work can be hoisted offline.
#
# A `linalg.transpose` whose input is a function argument is not computation — it is the model paying,
# on every inference, to convert a weight into the layout its consumer wanted all along. The packer
# could have stored it that way once.
#
# MEASURED, and why this module exists. On `gemma2_2b_int8_full_seq8` 183 such transposes move
# **2,493.0 MiB per inference** against a total weight blob of 2,505 MiB: essentially every weight in
# the model, re-laid-out every time it runs. It is not only a throughput tax. The largest one, the
# 562.5 MiB int8 tied head, is what killed a whole-model FireSim run — `FAIL alloc bytes=589824064` at
# op 11,494 of 11,526, with the arena unable to serve it. Hoisting all 183 into the bundle took the
# count to 0 with the MAC total and the matrix-unit routing both unchanged.
#
# HOISTABLE MEANS SOLE-USE. Storing a weight pre-transposed is safe only when the transpose is that
# argument's ONLY consumer. If anything else reads the argument, it would silently start seeing
# transposed data. :func:`weight_layout_report` therefore splits the two cases and never merges them —
# on the model above the split was 183 sole-use and 0 mixed, but a model with a weight used both
# transposed and untransposed must not be hoisted blindly.
#
# The rewrite itself is bit-exact: a transpose moves elements, it does not compute. Every value the
# consumer sees is the value it saw before, at the same index, so goldens survive untouched.
#
# This is a structural opportunity analysis, not permission to change an argument's stored bytes.
# The caller must separately establish immutable invocation ownership and storage identity.

#: bytes per element, by the dtype spelling `mlir_query.type_shape_dtype` returns.
_WIDTH = {"i8": 1, "u8": 1, "ui8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}


@dataclass(frozen=True)
class WeightLayoutStep:
    """One checked, row-major, element-preserving source layout operation."""

    operation: str
    shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    permutation: tuple[int, ...] = ()
    reassociation: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class WeightRelayout:
    """A transpose and its maximal sole-use static layout chain from an argument."""

    arg: int
    shape: list[int]
    dtype: str
    result_shape: list[int]
    permutation: tuple[int, ...]
    steps: tuple[WeightLayoutStep, ...]
    #: True when the transpose is the argument's only consumer, so the layout can be pre-applied.
    hoistable: bool
    #: why not, when it is not
    reason: str = ""

    @property
    def bytes_moved(self) -> int:
        width = _WIDTH.get(self.dtype)
        if width is None:
            return 0
        return sum(prod(step.result_shape) * width for step in self.steps if step.operation == "linalg.transpose")

    @property
    def transposes(self) -> int:
        return sum(step.operation == "linalg.transpose" for step in self.steps)


@dataclass
class WeightLayoutReport:
    relayouts: list[WeightRelayout] = field(default_factory=list)
    #: transposes whose element width is unknown, so their cost could not be priced. Surfaced rather
    #: than counted as zero: an unpriceable op must not read as a free one.
    unpriceable: list[str] = field(default_factory=list)

    @property
    def hoistable(self) -> list[WeightRelayout]:
        return [r for r in self.relayouts if r.hoistable]

    @property
    def blocked(self) -> list[WeightRelayout]:
        return [r for r in self.relayouts if not r.hoistable]

    @property
    def hoistable_bytes(self) -> int:
        """Weight traffic per inference that storing the transposed layout would remove."""
        return sum(r.bytes_moved for r in self.hoistable)

    @property
    def total_bytes(self) -> int:
        return sum(r.bytes_moved for r in self.relayouts)


def _func(module: Any, func_name: str):
    """Select one exact top-level symbol, never a nested function or substring."""
    if not isinstance(module, builtin.ModuleOp) or type(func_name) is not str or not func_name:
        raise ValueError("weight layout requires a module and an exact top-level function name")
    matches = [op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == func_name]
    if len(matches) != 1 or len(matches[0].body.blocks) != 1:
        raise ValueError(f"weight layout requires one defined top-level function {func_name!r}")
    return matches[0]


def _tensor(type):
    return (
        isinstance(type, builtin.TensorType)
        and isinstance(type.encoding, builtin.NoneAttr)
        and all(d >= 0 for d in type.get_shape())
    )


def _transpose_problem(op, arg, fn):
    if op.parent is not fn.body.block:
        return "transpose is not directly owned by the selected function"
    if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1 or op.inputs[0] is not arg:
        return "argument must be operand zero of one tensor transpose"
    source, result = arg.type, op.results[0].type
    if not _tensor(source) or not _tensor(result) or op.outputs[0].type != result:
        return "transpose requires static unencoded ranked tensor input, init and result"
    permutation = tuple(op.permutation.get_values())
    rank = len(source.get_shape())
    if rank == 0 or sorted(permutation) != list(range(rank)):
        return "transpose requires an explicit complete rank-N permutation"
    if (
        source.element_type != result.element_type
        or tuple(source.get_shape()[i] for i in permutation) != result.get_shape()
    ):
        return "transpose permutation, shape or element types disagree"
    body = op.body.block
    operations = list(body.ops)
    if (
        len(body.args) != 2
        or any(a.type != source.element_type for a in body.args)
        or len(operations) != 1
        or not isinstance(operations[0], YieldOp)
        or tuple(operations[0].operands) != (body.args[0],)
        or set(op.properties) != {"permutation"}
        or any(not key.startswith("prov.") for operation in (op, operations[0]) for key in operation.attributes)
        or operations[0].properties
    ):
        return "transpose has unsupported executable fields or scalar body"
    return ""


def _step(op, value, fn):
    if isinstance(op, TransposeOp):
        if _transpose_problem(op, value, fn):
            return None
        return WeightLayoutStep(
            op.name, value.type.get_shape(), op.results[0].type.get_shape(), tuple(op.permutation.get_values())
        )
    if not isinstance(op, tensor.CollapseShapeOp | tensor.ExpandShapeOp):
        return None
    if (
        op.parent is not fn.body.block
        or len(op.operands) != 1
        or op.operands[0] is not value
        or not _tensor(value.type)
        or not _tensor(op.results[0].type)
        or value.type.element_type != op.results[0].type.element_type
        or any(not key.startswith("prov.") for key in op.attributes)
    ):
        return None
    source, result = value.type.get_shape(), op.results[0].type.get_shape()
    groups = tuple(tuple(axis.value.data for axis in group) for group in op.reassociation)
    if isinstance(op, tensor.CollapseShapeOp):
        allowed, expanded, collapsed = {"reassociation"}, source, result
    else:
        allowed, expanded, collapsed = {"reassociation", "static_output_shape"}, result, source
        if tuple(op.static_output_shape.get_values()) != result:
            return None
    if (
        set(op.properties) != allowed
        or not source
        or not result
        or len(groups) != len(collapsed)
        or any(not group for group in groups)
        or tuple(axis for group in groups for axis in group) != tuple(range(len(expanded)))
        or tuple(prod(expanded[axis] for axis in group) for group in groups) != collapsed
    ):
        return None
    return WeightLayoutStep(op.name, source, result, reassociation=groups)


def layout_chain(value, fn):
    """The maximal supported sole-use chain; unknown operations and extra readers stop it."""
    operations, steps = [], []
    while len(uses := list(value.uses)) == 1 and uses[0].index == 0:
        op = uses[0].operation
        step = _step(op, value, fn)
        if step is None:
            break
        operations.append(op)
        steps.append(step)
        value = op.results[0]
    return tuple(operations), tuple(steps)


def apply_layout_steps(value, steps):
    """Evaluate checked views/permutations, as group_prepack does, without numeric conversion.

    The caller chooses the element representation. Capture rewriting uses fixed-width raw
    byte records, so BF16 and floating NaN payloads never pass through arithmetic.
    """
    import numpy as np

    for step in steps:
        if tuple(value.shape) != step.shape:
            raise ValueError("stored tensor shape differs from the checked layout step")
        if step.operation == "linalg.transpose":
            if sorted(step.permutation) != list(range(value.ndim)):
                raise ValueError("invalid stored tensor permutation")
            value = np.transpose(value, step.permutation)
        elif step.operation in {"tensor.collapse_shape", "tensor.expand_shape"}:
            value = value.reshape(step.result_shape)
        else:
            raise ValueError("unsupported stored tensor layout step")
        if tuple(value.shape) != step.result_shape:
            raise ValueError("stored tensor layout did not produce the checked shape")
    return np.ascontiguousarray(value)


def weight_layout_report(module: Any, func_name: str = "forward") -> WeightLayoutReport:
    """Every run-time re-layout of a weight in `func_name`, split by whether it can be hoisted."""
    try:
        module.verify()
    except Exception as exc:
        raise ValueError(f"weight layout requires verified source IR: {exc}") from exc
    fn = _func(module, func_name)
    rep = WeightLayoutReport()

    for i, arg in enumerate(fn.body.blocks[0].args):
        uses = list(arg.uses)
        transposes = [u for u in uses if isinstance(u.operation, TransposeOp)]
        if not transposes:
            continue
        op = transposes[0].operation
        shape = list(arg.type.get_shape()) if isinstance(arg.type, builtin.TensorType) else []
        dtype = str(arg.type.element_type) if isinstance(arg.type, builtin.TensorType) else "unknown"
        result = op.results[0].type if op.results else None
        res_shape = list(result.get_shape()) if isinstance(result, builtin.TensorType) else []

        if dtype not in _WIDTH:
            rep.unpriceable.append(f"arg {i}: unknown element width for dtype {dtype!r}")

        sole = len(uses) == 1 and len(transposes) == 1 and transposes[0].index == 0
        reason = (
            _transpose_problem(op, arg, fn)
            if sole
            else (
                f"argument has {len(uses)} consumers ({len(transposes)} transpose(s)); pre-transposing "
                "would change what the other readers see"
            )
        )
        first = _step(op, arg, fn)
        _operations, steps = layout_chain(arg, fn) if sole and not reason else ((), (first,) if first else ())
        if steps:
            res_shape = list(steps[-1].result_shape)
        rep.relayouts.append(
            WeightRelayout(
                arg=i,
                shape=shape,
                dtype=dtype,
                result_shape=res_shape,
                permutation=tuple(op.permutation.get_values()),
                steps=steps,
                hoistable=sole and not reason,
                reason=reason,
            )
        )
    return rep
