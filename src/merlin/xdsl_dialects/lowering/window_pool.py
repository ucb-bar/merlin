"""Recognize a closed quantized spatial maximum as movement plus readout.

The finite input domain proves the scale composition, not model calibration
quality. Layout and padding are read from the source, never from a model name.
"""

from __future__ import annotations

import math

import numpy as np

from merlin.common import mlir_query as mq


def match(group, oracle):
    from . import compute_groups as CG
    from . import contraction_coverage as CC
    from . import group_command as GC
    from . import group_numerics as GN

    if len(group.members) not in (3, 4):
        return None
    dq, q = group.members[0], group.members[-1]
    if mq.op_name(dq) != "quant_ext.dequantize_per_tensor" or mq.op_name(q) != "quant_ext.quantize_per_tensor":
        return None
    with CC.select_max_is_max():
        stages = [CG.classify(member).kind for member in group.members]
    if stages not in ([CG.DEQUANTIZE, CG.POOL, CG.QUANTIZE], [CG.DEQUANTIZE, CG.PAD, CG.POOL, CG.QUANTIZE]):
        return None
    image_shape, image_dtype = mq.type_shape_dtype(dq.operands[0].type)
    output_shape, output_dtype = mq.type_shape_dtype(q.results[0].type)
    if (
        len(image_shape) != 4
        or image_shape[0] != 1
        or len(output_shape) != 4
        or output_shape[:2] != image_shape[:2]
        or min(*image_shape, *output_shape) < 1
        or CG._dtype_token(image_dtype) != "int8"
        or CG._dtype_token(output_dtype) != "int8"
    ):
        return None
    scales = [GN._constant_of(op.operands[1]) for op in (dq, q)]
    zeros = [GN._constant_of(op.operands[2]) for op in (dq, q)]
    if zeros != [0, 0] or any(not isinstance(s, (float, int)) or not math.isfinite(s) or s <= 0 for s in scales):
        return None

    # Quantization bounds belong to the source operation, not merely its type.
    def bound(op, key):
        attr = op.properties.get(key) or op.attributes.get(key)
        return int(attr.value.data) if attr is not None else None

    if any((bound(op, "quant_min"), bound(op, "quant_max")) != (-128, 127) for op in (dq, q)):
        return None
    movement = oracle.ask(op="movement", family="movement", in_dtype="int8", attached=False, form="copy")
    if not movement.admitted:
        return None
    unit = movement.units[0] if movement.units else "accelerator"
    if not oracle.absorbs_sequence([CG.SCALE, CG.POOL], unit=unit).admitted:
        return None
    source = np.arange(-128, 128, dtype=np.float32)
    scale_in, inverse = np.float32(scales[0]), np.float32(1.0 / scales[1])
    gain = np.float32(scale_in * inverse)
    if not np.isfinite(gain) or gain <= 0:
        return None
    expected = np.clip(np.rint((source * scale_in) * inverse), -128, 127)
    actual = np.clip(np.rint(source * gain), -128, 127)
    if not np.array_equal(expected, actual):
        return None
    pool = group.members[stages.index(CG.POOL)]
    initial = GN._constant_of(pool.operands[-1])
    if not isinstance(initial, float) or not (math.isinf(initial) and initial < 0):
        return None
    # A connected, single-use path; no unrelated member can be consumed.
    value = dq.results[0]
    for member in group.members[1:]:
        if not member.operands or member.operands[0] is not value or any(u.operation is not member for u in value.uses):
            return None
        value = member.results[0]
    try:
        with CC.select_max_is_max():
            geometry = GC._pool(group, clamp=(-128, 127))
    except CG.NoCapsuleForm:
        return None
    from merlin.runtime.tensor import pool_out_dims

    h, w = image_shape[2:]
    ho, wo = pool_out_dims(h, w, geometry["pool_size"], geometry["pool_stride"], geometry["pool_padding"])
    if output_shape[2:] != [ho, wo]:
        return None
    return (
        pool,
        stages,
        unit,
        {
            **geometry,
            "pool_in_dims": [h, w],
            "M": h * w,
            "N": image_shape[1],
            "acc_scale": float(gain),
            "output_shape": [ho * wo, image_shape[1]],
            "proof": "equal requantized codes for every signed i8 input; positive monotone scales commute with maximum",
        },
    )
