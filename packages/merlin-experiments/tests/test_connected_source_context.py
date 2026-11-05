"""Conditional width facts cannot be reused across changed source projections."""
from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.connected_region import extract_window, text
from merlin_experiments.phase0.connected_source_context import ConnectedSourceContext, require_context

SOURCE = '''builtin.module {
  func.func @original(%x: i64, %y: i64) -> i64 {
    %a = arith.addi %x, %y : i64
    func.return %a : i64
  }
}'''


def _selected():
    return dict(schema='merlin.selected-index-lowering.v1', compiler_requested='fixture',
                compiler_resolved='/fixture', compiler_sha256='a'*64, cross_flags=['--target=fixture'],
                data_layout='e-p:64:64', index_bits=64, scope='synthetic test observation')


def _make(tmp_path):
    source, projected = tmp_path/'source.mlir', tmp_path/'cut.mlir'
    source.write_text(SOURCE)
    projected.write_text(text(extract_window(mq.parse(SOURCE), 0, 0).module))
    observation = _selected()
    context = ConnectedSourceContext(source, projected, source_sha256=sha256(source.read_bytes()).hexdigest(),
        projected_sha256=sha256(projected.read_bytes()).hexdigest(), start=0, stop=0,
        selected_index_observation=observation)
    return source, projected, context, observation


def test_context_bound_to_exact_sources_and_owned_observation(tmp_path):
    source, projected, context, observation = _make(tmp_path)
    observation['index_bits'] = 8
    assert context.for_program(projected)['selected_index_observation']['index_bits'] == 64
    require_context(context, {'cut': projected})
    with pytest.raises(ValueError, match='another projection'):
        context.for_program(source)
    source.write_text(SOURCE.replace('arith.addi', 'arith.muli'))
    with pytest.raises(ValueError, match='bytes changed'):
        context.for_program(projected)


def test_serialized_or_unissued_context_cannot_supply_admission_premise(tmp_path):
    _, projected, context, _ = _make(tmp_path)
    for invalid in [context.for_program(projected), object.__new__(ConnectedSourceContext)]:
        with pytest.raises(ValueError, match='issued'):
            require_context(invalid, {'cut':projected})
    projected.write_text(projected.read_text().replace('arith.addi','arith.muli'))
    with pytest.raises(ValueError, match='bytes changed'):
        require_context(context, {'cut':projected})


def test_index_observation_must_be_complete_and_consistent(tmp_path):
    source, projected, _, observation = _make(tmp_path)
    observation['index_bits'] = 32
    with pytest.raises(ValueError, match='compiler index observation'):
        ConnectedSourceContext(source, projected, source_sha256=sha256(source.read_bytes()).hexdigest(),
            projected_sha256=sha256(projected.read_bytes()).hexdigest(), start=0, stop=0,
            selected_index_observation=observation)
