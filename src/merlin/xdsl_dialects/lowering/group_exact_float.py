"""Prove when an integer readout exactly implements a captured float epilogue.

The proof is source and weight bound. A failed proof changes the compute-group
boundary, leaving the raw contraction on the accelerator and the original float
operations on the host. It never uses model names or sample outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import mlir_query as mq
from merlin.llvmlower import requantization as RQ

from . import compute_groups as CG
from . import group_command as GC
from . import group_numerics as GN
from . import group_prepack as GP
from . import stream_plan as SP


class ExactFloatProofUnavailable(ValueError):
    """The stated device readout has no proof of captured float equivalence."""


def needs_proof(group) -> bool:
    """A cast i32 accumulator with a stored float bias and a quantized sink."""
    if group.root is None or group.placement == CG.HOST:
        return False
    kinds = list(group.stages)
    if not {CG.CAST, CG.SCALE, CG.BIAS_ADD, CG.QUANTIZE}.issubset(kinds):
        return False
    _, dtype = mq.type_shape_dtype(group.root.results[0].type)
    return dtype == 'i32'


def _order(group) -> list[float]:
    kinds = list(group.stages)
    if kinds.count(CG.CAST) != 1 or kinds.count(CG.BIAS_ADD) != 1 or kinds.count(CG.QUANTIZE) != 1:
        raise ExactFloatProofUnavailable('the float readout is not one cast, bias and quantize')
    cast, bias, quantize = (kinds.index(kind) for kind in (CG.CAST, CG.BIAS_ADD, CG.QUANTIZE))
    if not cast < bias < quantize == len(kinds) - 1:
        raise ExactFloatProofUnavailable('the float readout stage order is not proved')
    if any(kind == CG.SCALE and not cast < index < bias for index, kind in enumerate(kinds)):
        raise ExactFloatProofUnavailable('a scale after the float bias has no exact readout proof')
    if any(kind not in {CG.CONTRACTION, CG.CAST, CG.SCALE, CG.VIEW, CG.MOVEMENT,
                        CG.BIAS_ADD, CG.RELU, CG.QUANTIZE}
           for kind in kinds):
        raise ExactFloatProofUnavailable('the float readout contains a stage the proof does not model')
    if CG.RELU in kinds and (kinds.count(CG.RELU) != 1 or not bias < kinds.index(CG.RELU) < quantize):
        raise ExactFloatProofUnavailable('the float activation order is not proved')
    # The transition proof models ordered binary32 arithmetic. A captured
    # f64/f16 intermediate or output scale has different rounding boundaries.
    for member in group.members:
        stage = CG.classify(member)
        if stage is not None and stage.kind in {CG.CAST, CG.SCALE, CG.BIAS_ADD, CG.RELU}:
            if not member.results or mq.type_shape_dtype(member.results[0].type)[1] != 'f32':
                raise ExactFloatProofUnavailable('the staged float arithmetic is not binary32')
    closing = next(member for member in group.members if CG.classify(member).kind == CG.QUANTIZE)
    if (mq.type_shape_dtype(closing.operands[0].type)[1] != 'f32'
            or mq.type_shape_dtype(closing.operands[1].type)[1] != 'f32'):
        raise ExactFloatProofUnavailable('the closing quantize is not binary32')
    scales = [GN._scale_constant(member) for member in group.members
              if CG.classify(member).kind == CG.SCALE]
    if not scales or any(scale is None for scale in scales):
        raise ExactFloatProofUnavailable('the float scale operands are not fixed scalar values')
    return [float(scale) for scale in scales]


def prove(group, manifest: dict, weights_path: str | Path, *, weight_args=None) -> dict:
    """Return a whole-group exact proof, or an explicit unproved result.

    The accumulator interval covers every signed int8 input against the exact
    stored integer weights. Output channels share its maximum bound so the
    proof never depends on the captured sample input.
    """
    import numpy as np

    if not needs_proof(group):
        raise ExactFloatProofUnavailable('group is not a converted float-bias readout')
    scales = _order(group)
    numerics = GN.numerics_of(group)
    if numerics.clamp != (-128, 127) or numerics.output.zero_point != 0:
        raise ExactFloatProofUnavailable('the proven readout requires signed i8 and zero output offset')
    if numerics.output.value is None or numerics.output.value <= 0:
        raise ExactFloatProofUnavailable('the output quantization scale is not one positive constant')
    arguments = weight_args if weight_args is not None else SP.weight_args_of(manifest)
    stated = GC.program(group, weight_args=arguments)
    if stated.stored_arg is None or numerics.bias_arg_index is None:
        raise ExactFloatProofUnavailable('the integer weight or float bias is not source-bound')
    bias_name, bias, bias_dtype = GP._stored_raw(numerics.bias_arg_index, manifest, Path(weights_path))
    if bias_dtype != 'F32':
        raise ExactFloatProofUnavailable('the stored float bias is not binary32')
    bias = bias.reshape(-1)
    weight_name, weight, spelled = GP._stored_raw(stated.stored_arg, manifest, Path(weights_path))
    if spelled != 'I8':
        raise ExactFloatProofUnavailable('the proof requires stored signed i8 weights')
    op = stated.entry.get('op')
    if op == 'conv2d':
        channel_axis = 0
    elif op in {'matmul', 'matmul_batched'}:
        channel_axis = -2 if stated.transposed else -1
    else:
        raise ExactFloatProofUnavailable('the stored-weight output axis is not proved')
    channel_count = int(weight.shape[channel_axis])
    if bias.size != channel_count or stated.entry.get('N') != channel_count:
        raise ExactFloatProofUnavailable('stored weight and float bias channel counts differ')
    columns = np.moveaxis(weight.astype(np.int64), channel_axis, 0).reshape(channel_count, -1)
    bound = int(np.abs(columns).sum(axis=1).max(initial=0)) * 128
    if bound >= 1 << 31:
        raise ExactFloatProofUnavailable('the stated input range can overflow the i32 accumulator')
    result = RQ.synthesize_bias(
        scales, bias, RQ.f32(1.0 / numerics.output.value), -bound, bound,
        relu=numerics.activation == 'relu',
    )
    if result['scale'] != RQ.f32(numerics.multiplier):
        raise ExactFloatProofUnavailable('the proven scale differs from the emitted command scale')
    return {
        'group': group.index,
        'exact': result['accepted_channels'] == result['channels'],
        'weight': weight_name,
        'bias': bias_name,
        'input_bound': [-bound, bound],
        'channels': result['channels'],
        'accepted_channels': result['accepted_channels'],
        'integer_bias': result['integer_bias'],
        'readout_scale': result['scale'],
        'refused_channels': result['refused_channels'],
        'proof': result['proof'],
    }


def refusals(groups, capture: str | Path, *, weight_args=None) -> dict[int, str]:
    """Return root-identity refusals for converted stores lacking exact proof."""
    beside = Path(capture)
    beside = beside if beside.is_dir() else beside.parent
    manifest_path = next(iter(sorted(beside.glob('*.manifest.json'))), None)
    weights_path = next(iter(sorted(beside.glob('*.safetensors'))), None)
    out = {}
    for group in groups:
        if not needs_proof(group):
            continue
        if manifest_path is None or weights_path is None:
            out[id(group.root)] = 'the captured weight and bias files are absent, so exact float fusion is unproved'
            continue
        try:
            evidence = prove(group, json.loads(manifest_path.read_text()), weights_path,
                             weight_args=weight_args)
            if not evidence['exact']:
                out[id(group.root)] = (
                    f"the staged f32 bias/scale cannot be represented exactly by an integer "
                    f"readout on {evidence['channels'] - evidence['accepted_channels']} of "
                    f"{evidence['channels']} channels over its input bound"
                )
        except (ExactFloatProofUnavailable, GN.GroupNumericsError, CG.NoCapsuleForm, ValueError) as error:
            out[id(group.root)] = f'exact float readout cannot be proved: {error}'
    return out
