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
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq

from . import compute_groups as CG
from . import group_numerics as GN

SCHEMA = "group_prepack_v1"
_NUMPY_DTYPE = {"F32": "<f4", "F64": "<f8", "F16": "<f2"}


_INTEGER_DTYPE = {"I8": "<i1", "U8": "<u1", "I16": "<i2", "I32": "<i4"}


def _stored_raw(arg_index: int, manifest: dict, safetensors_path: Path):
    """``(name, array)`` of the stored tensor behind a model argument, in its own dtype and shape."""
    import numpy as np

    from merlin.llvmlower.weights_pack import load_safetensors_header

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

    adapters, dequantize, _dtype = CG._input_chain(list(group.root.operands)[operand_index])
    value = np.asarray(stored)
    between = []  # from the stored tensor toward the dequantize: views and movement only
    cursor = dequantize.operands[0] if dequantize is not None else None
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

    from . import group_command, stream_plan

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    weight_args = stream_plan.weight_args_of(manifest)
    limit = (1 << (accumulator_bits - 1)) - 1
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    arrays: dict[str, Any] = {}
    for group in groups:
        if group.placement == CG.HOST or CG.QUANTIZE not in group.stages:
            continue
        if group.operand_sum is not None or group.window_mean is not None:
            rows.append(_scalar_row(group))
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
                    "array": key,
                    "elements": int(folded.size),
                    "max_abs": int(np.abs(folded).max(initial=0)),
                    "sha256": hashlib.sha256(arrays[key].tobytes()).hexdigest(),
                }
            if device_layout:
                try:
                    stated = group_command.program(group, weight_args=weight_args)
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
