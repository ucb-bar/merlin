"""Exact cut facts require fresh literals and source IDs, never copied ordinals."""
from hashlib import sha256

import pytest
from xdsl.dialects.func import FuncOp

from merlin.common import mlir_query as mq
from merlin.frontends.connected_region import extract_window, text
from merlin_experiments.phase0.connected_index_context import prove
from merlin_experiments.phase0.connected_source_context import ConnectedSourceContext
from test_connected_source_context import _selected
from test_private_index_source import _capture


def _fixture(tmp_path, trace_change=None):
    capture = _capture(tmp_path, trace_change=trace_change)
    source = capture / 'model.mlir'
    parsed = mq.parse(source.read_text())
    function = next(o for o in parsed.walk() if isinstance(o, FuncOp))
    stop = len(tuple(function.body.block.ops)) - 2
    cut = tmp_path / 'cut.mlir'
    cut.write_text(text(extract_window(parsed, 0, stop).module))
    context = ConnectedSourceContext(source, cut,
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        projected_sha256=sha256(cut.read_bytes()).hexdigest(),
        start=0, stop=stop, selected_index_observation=_selected())
    pins = dict(zip(('capture_model_sha256', 'frontend_trace_sha256', 'capture_receipt_sha256'),
        (sha256((capture / name).read_bytes()).hexdigest()
         for name in ('model.mlir', 'frontend-trace.json', 'capture_receipt.json')), strict=True))
    return context, cut, capture, pins


def test_reproves_cut_bounds_and_pins_without_granting_capture_equivalence(tmp_path):
    context, cut, capture, pins = _fixture(tmp_path)
    result = prove(context, cut, capture, **pins)
    assert result['records'][0]['kind'] == 'literal_indexed_extract'
    assert len(result['ranges']) == 2
    assert result['source_sha256'] == pins['capture_model_sha256']
    assert result['projected_sha256'] == sha256(cut.read_bytes()).hexdigest()
    assert result['status'] == 'source_proved_pending_capture_preparation_join_host_and_build'
    # No caller-supplied saved proof is consumed; a changed pinned trace refuses.
    trace = capture / 'frontend-trace.json'
    trace.write_text(trace.read_text() + '\n')
    with pytest.raises(ValueError, match='pin differs'):
        prove(context, cut, capture, **pins)


@pytest.mark.parametrize('change', ['target', 'literal', 'duplicate'])
def test_self_consistent_receipt_cannot_hide_wrong_source_context(tmp_path, change):
    def modify(trace):
        nodes = trace['graphs']['prepared']['nodes']
        if change == 'target':
            nodes[-1]['target'] = 'aten.other.default'
        elif change == 'literal':
            nodes[0]['args'] = [1, 3, 1]  # Same extent; actual source starts at zero.
        else:
            nodes.append(dict(nodes[0]))
    context, cut, capture, pins = _fixture(tmp_path, modify)
    with pytest.raises(ValueError):
        prove(context, cut, capture, **pins)


def test_cut_mutation_cannot_reuse_context(tmp_path):
    context, cut, capture, pins = _fixture(tmp_path)
    cut.write_text(cut.read_text().replace('g:prepared:root:index', 'g:prepared:root:other'))
    with pytest.raises(ValueError, match='bytes changed'):
        prove(context, cut, capture, **pins)
