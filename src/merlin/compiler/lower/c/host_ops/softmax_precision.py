"""Bind complete source softmax regions to the host scalar implementation.

The implementation order is internal policy. Admission comes from recognition
of the actual semantic request, including its masks and exact constant bytes.
It does not claim to reproduce a CPU vector kernel's reduction order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import prod

from merlin.compiler.semantic.model import IndexMap, KernelRequest, TensorType, pointwise_identity_maps
from merlin.compiler.semantic.linalg.softmax_source import (
    RowSoftmaxSource,
    recognize_masked_row_softmax,
    recognize_row_softmax,
)

ORDER = 'host-softmax-f32-left-fold-div-rne-v1'
ORDER_RECORD = {
    'order': ORDER,
    'input': 'dense binary32 scores; additive mask rounded once before maximum',
    'select_mask': 'keep score when predicate equals source keep_when; otherwise exact source binary32 fill_bits',
    'maximum': 'left fold from negative infinity, propagate canonical NaN',
    'shift': 'binary32 score minus maximum, round once',
    'exponential': 'merlin_host_expf through admitted shared host-math policy',
    'sum': 'left fold from positive zero, round each binary32 addition',
    'normalize': 'binary32 exponential divided by sum, round once',
    'nan': 'canonical binary32 0x7fc00000',
    'environment': 'RNE, gradual underflow, no contraction or reassociation',
}


@dataclass(frozen=True)
class _SelectRowSoftmaxSource(RowSoftmaxSource):
    mask: str
    mask_map: IndexMap
    masked_scores: str
    mask_shape: tuple[int, ...]
    fill_bits: int
    keep_when: bool

    def record(self):
        return {
            **super().record(),
            'mask': self.mask,
            'mask_map': self.mask_map.record(),
            'masked_scores': self.masked_scores,
            'mask_shape': list(self.mask_shape),
            'fill_bits': self.fill_bits,
            'keep_when': self.keep_when,
        }


def _check_map(mapping, source_shape, loop_shape):
    if (
        not isinstance(mapping, IndexMap)
        or mapping.loop_rank != len(loop_shape)
        or len(mapping.coefficients) != len(source_shape)
    ):
        raise ValueError('host softmax: mask or fill map has incompatible rank')
    for row, offset, extent in zip(mapping.coefficients, mapping.offsets, source_shape, strict=True):
        low = offset + sum(min(0, coefficient * (n - 1)) for coefficient, n in zip(row, loop_shape, strict=True))
        high = offset + sum(max(0, coefficient * (n - 1)) for coefficient, n in zip(row, loop_shape, strict=True))
        if not 0 <= low <= high < extent:
            raise ValueError('host softmax: mask or fill map leaves its source')


def _compose_map(inner, outer):
    """Compose loop -> gathered value -> underlying tensor."""
    return IndexMap(
        outer.loop_rank,
        tuple(tuple(sum(a * b for a, b in zip(row, column, strict=True))
                    for column in zip(*outer.coefficients)) for row in inner.coefficients),
        tuple(offset + sum(a * b for a, b in zip(row, outer.offsets, strict=True))
              for row, offset in zip(inner.coefficients, inner.offsets, strict=True)),
    )


def _recognize_select(request, core):
    selected = request.node(core.scores)
    rank = len(core.shape)
    identity = pointwise_identity_maps(rank, 1)[0]
    if (
        selected.effect != 'pure'
        or len(selected.inputs) != 3
        or selected.attrs
        or len(selected.index_maps) != 4
        or selected.index_maps[-1] != identity
    ):
        raise ValueError('host softmax: unsupported source select')
    computations = set(core.computations) | {selected.id}

    def unwrap(key, mapping):
        value = request.node(key)
        _check_map(mapping, value.type.shape, core.shape)
        if value.op == 'gather':
            gather_identity = pointwise_identity_maps(len(value.type.shape), 1)[0]
            if (
                value.effect != 'pure' or len(value.inputs) != 1 or value.attrs
                or len(value.index_maps) != 2 or value.index_maps[-1] != gather_identity
            ):
                raise ValueError('host softmax: unsupported mask or fill gather')
            source = request.node(value.inputs[0])
            if source.type.dtype != value.type.dtype or source.type.numerical_policy != value.type.numerical_policy:
                raise ValueError('host softmax: gather changes mask or fill type')
            _check_map(value.index_maps[0], source.type.shape, value.type.shape)
            computations.add(value.id)
            mapping = _compose_map(value.index_maps[0], mapping)
            value = source
        return value, mapping

    def literal(key, mapping):
        before = set(computations)
        try:
            value, _ = unwrap(key, mapping)
            binding = next((item for item in request.constants if item.node_id == value.id), None)
            if (
                value.op != 'constant' or value.effect != 'constant' or value.attrs or value.index_maps
                or value.type != TensorType(value.type.shape, 'f32', 'raw-bit-pattern')
                or binding is None or binding.encoding != 'f32-le-raw'
            ):
                raise ValueError('host softmax: select fill needs an exact f32 constant')
            data = bytes.fromhex(binding.data_hex)
            if len(data) != 4 * prod(value.type.shape) or data != data[:4] * prod(value.type.shape):
                raise ValueError('host softmax: select fill is not one uniform binary32 value')
            return value.id, int.from_bytes(data[:4], 'little')
        except ValueError:
            computations.intersection_update(before)
            raise

    try:
        fill, fill_bits = literal(selected.inputs[2], selected.index_maps[2])
        score_index, keep_when = 1, True
    except ValueError:
        fill, fill_bits = literal(selected.inputs[1], selected.index_maps[1])
        score_index, keep_when = 2, False
    scores = request.node(selected.inputs[score_index])
    mask, mask_map = unwrap(selected.inputs[0], selected.index_maps[0])
    if (
        scores.type != selected.type or selected.index_maps[score_index] != identity
        or scores.effect not in {'input', 'constant', 'pure'}
        or mask.type != TensorType(mask.type.shape, 'i1', 'boolean')
        or mask.effect not in {'input', 'constant', 'pure'}
    ):
        raise ValueError('host softmax: select scores or Boolean mask have incompatible types')
    _check_map(mask_map, mask.type.shape, core.shape)
    hidden = computations - {core.result}
    if hidden.intersection(request.outputs) or any(
        node.id not in computations and hidden.intersection(node.inputs) for node in request.nodes
    ):
        raise ValueError('host softmax: a select intermediate escapes its complete region')
    return _SelectRowSoftmaxSource(
        core.source_digest, scores.id, core.result, core.shape,
        tuple(node.id for node in request.nodes if node.id in computations),
        tuple(node.id for node in request.nodes if node.id in {*core.initializers, fill}),
        mask.id, mask_map, selected.id, mask.type.shape, fill_bits, keep_when,
    )


def _recognize(request, result_id):
    core = recognize_row_softmax(request, result_id)
    producer = request.node(core.scores)
    if producer.op == 'select':
        return _recognize_select(request, core), 'select'
    if producer.op == 'f32_add':
        return recognize_masked_row_softmax(request, result_id), 'additive'
    return core, 'none'


def _signature(region):
    """Capture every region binding independently of the supplied object."""
    if not isinstance(region, RowSoftmaxSource):
        raise ValueError('host softmax: expected a complete recognized source region')
    return tuple((item.name, getattr(region, item.name)) for item in fields(region))


_BINDING_TOKEN = object()


@dataclass(frozen=True)
class HostSoftmaxContract:
    region: RowSoftmaxSource
    dtype: str
    axis: int
    computations: tuple[str, ...]
    initializers: tuple[str, ...]
    mask_kind: str
    fill_bits: int | None
    keep_when: bool | None
    source_semantics: str
    source_digest: str
    _request: KernelRequest = field(repr=False, compare=False)
    _binding: tuple = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self):
        if self._token is not _BINDING_TOKEN:
            raise ValueError('host softmax: bind an actual semantic source request')

    def record(self):
        self.validate_region(self.region)
        return {
            'source_request_digest': self.source_digest,
            'source_semantics': self.source_semantics,
            'dtype': self.dtype,
            'axis': self.axis,
            'computations': list(self.computations),
            'initializers': list(self.initializers),
            'mask_kind': self.mask_kind,
            'fill_bits': self.fill_bits,
            'keep_when': self.keep_when,
        }

    def validate_region(self, region):
        """Replay source recognition and reject changed or incomplete bindings."""
        if self._token is not _BINDING_TOKEN or self.source_semantics != 'torch.softmax':
            raise ValueError('host softmax: unsupported source contract')
        if self._request.digest() != self.source_digest:
            raise ValueError('host softmax: source request digest changed')
        try:
            recognized, mask_kind = _recognize(self._request, dict(self._binding)['result'])
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('host softmax: incomplete bound source region') from error
        signature = _signature(recognized)
        if (
            signature != self._binding or _signature(self.region) != signature or _signature(region) != signature
            or self.dtype != 'f32' or type(self.axis) is not int or self.axis != len(recognized.shape) - 1
            or self.computations != recognized.computations or self.initializers != recognized.initializers
            or self.mask_kind != mask_kind or self.source_digest != recognized.source_digest
            or self.fill_bits != getattr(recognized, 'fill_bits', None)
            or self.keep_when != getattr(recognized, 'keep_when', None)
            or type(self.fill_bits) is not type(getattr(recognized, 'fill_bits', None))
            or type(self.keep_when) is not type(getattr(recognized, 'keep_when', None))
        ):
            raise ValueError('host softmax: region or source contract bindings differ')


def bind_source_contract(request, result_id, *, source_semantics='torch.softmax'):
    """Admit the complete last-axis f32 source graph, with no caller order."""
    if not isinstance(request, KernelRequest):
        raise ValueError('host softmax: an actual semantic KernelRequest is required')
    if source_semantics != 'torch.softmax':
        raise ValueError('host softmax: unsupported source semantics')
    try:
        region, mask_kind = _recognize(request, result_id)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('host softmax: incomplete source region') from error
    contract = HostSoftmaxContract(
        region, 'f32', len(region.shape) - 1, region.computations, region.initializers,
        mask_kind, getattr(region, 'fill_bits', None), getattr(region, 'keep_when', None),
        source_semantics, request.digest(), request, _signature(region), _BINDING_TOKEN,
    )
    contract.validate_region(region)
    return contract


def admit_host_math(host_math, policy_path):
    """Use the shared host-math module and its original policy without fallback."""
    policy = host_math.load_policy(policy_path, required=('expf',))
    if policy['functions']['expf']['symbol'] != 'merlin_host_expf':
        raise ValueError('unexpected host-math exponential symbol')
    return policy


POLICY_SCHEMA = 'merlin.host_softmax.v1'


def host_softmax_policy():
    """Declare the scalar family and its order; the exponential is embedded from the pinned supplier."""
    return validate_policy({'schema': POLICY_SCHEMA, 'order': ORDER})


def validate_policy(value):
    """Admit only this family schema and its declared order."""
    if not isinstance(value, dict) or value != {'schema': POLICY_SCHEMA, 'order': ORDER}:
        raise ValueError('host softmax policy needs its schema and declared order')
    return dict(value)
