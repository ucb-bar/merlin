"""Early layout consumption retains original software and actual binding order."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_host_graph_optimizations import _execute, _source
from test_host_graph_policy import software

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.device_preparation import freeze_device_offload_selection, require_device_offload_selection
from merlin.llvmlower.host_graph_policy import (
    bind_group_layout_artifacts,
    prepare_group_layout,
    read_host_graph_policy,
)
from merlin.runtime.backends.zephyr_model import _prepare_model_mlir
from merlin.xdsl_dialects._common import text


@pytest.mark.parametrize('shape', [(2, 3), (3, 5)])
def test_layout_is_checked_and_executed_before_group_binding(tmp_path, monkeypatch, shape):
    from merlin.llvmlower import device_offload

    source = tmp_path/'source.mlir'
    source.write_text(_source(*shape))
    policy = read_host_graph_policy(software(tmp_path), target='neutral', entry='kernel')
    device = SimpleNamespace(pre_group_layout=policy, device='neutral', select=None, capture=None, model='')
    seen = []

    def bind(module, *args, **kwargs):
        seen.append(text(module, generic=True))
        assert not any(op.name == 'linalg.transpose' for op in module.walk())
        return SimpleNamespace(signatures={}, arg_access={}, moved=0, write_sidecar=lambda work: None)

    monkeypatch.setattr(device_offload, 'rewrite_groups_to_device', bind)
    work = tmp_path/'prepared'
    work.mkdir()
    result = _prepare_model_mlir(source, work, complete_group_device=device)
    assert len(seen) == 1
    x = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    bias = np.arange(shape[1], dtype=np.int32)
    np.testing.assert_array_equal(_execute(source.read_text(), x, bias)[0], x+bias)
    np.testing.assert_array_equal(_execute(result.read_text(), x, bias)[0], x+bias)
    report = json.loads((work/'pre_group_layout/host-graph-optimization.json').read_text())
    assert report['automatic_selection']['status'] == 'applied'
    assert report['layout']['before_transpose_bytes'] > report['layout']['after_transpose_bytes']
    pins = {}
    compilation = SimpleNamespace(bind_preparation=lambda role, path: pins.update({role: path}))
    bind_group_layout_artifacts(device, work, compilation)
    assert len(pins) == 5 and all(path.is_file() for path in pins.values())


def test_missing_permission_retains_same_module_and_explicit_reason(tmp_path):
    policy = read_host_graph_policy(software(tmp_path, independent_element_order_unobserved=False),
                                    target='neutral', entry='kernel')
    original = parse_mlir_text(_source())
    result = prepare_group_layout(original, policy, tmp_path/'early')
    assert result is original
    report = json.loads((tmp_path/'early/host-graph-optimization.json').read_text())
    assert report['automatic_selection']['reason'] == 'permission_not_admitted'


def test_policy_and_realized_source_reopen_at_ordinary_consumption(tmp_path):
    policy = read_host_graph_policy(software(tmp_path), target='neutral', entry='kernel')
    device = SimpleNamespace(device='neutral', pre_group_layout=policy)
    frozen = freeze_device_offload_selection(device)
    prepare_group_layout(parse_mlir_text(_source()), policy, tmp_path/'pre_group_layout')
    require_device_offload_selection(device, frozen)
    device.pre_group_layout = read_host_graph_policy(software(tmp_path), target='neutral', entry='kernel')
    with pytest.raises(ValueError, match='selection changed'):
        require_device_offload_selection(device, frozen)
    device.pre_group_layout = policy
    selected = tmp_path/'pre_group_layout/host-graph-optimized.mlir'
    selected.write_text(selected.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'):
        bind_group_layout_artifacts(device, tmp_path, SimpleNamespace(bind_preparation=lambda *args: None))


def projected_source():
    # A real call boundary, including a public observation of its complete
    # result. The helper body is original source, never interpreted by layout
    # propagation across the call.
    source = _source().replace('@kernel', '@forward')
    source = source.replace('builtin.module {', '''builtin.module {
 func.func private @selected_device(%a:tensor<2x3xi32>)->tensor<2x3xi32> {
  func.return %a:tensor<2x3xi32>
 }
''', 1)
    source = source.replace('  %e =', '''  %device = func.call @selected_device(%x)
    :(tensor<2x3xi32>)->tensor<2x3xi32>
  %e =''', 1).replace('ins(%x:', 'ins(%device:')
    source = source.replace('-> tensor<2x3xi32> {', '-> (tensor<2x3xi32>,tensor<2x3xi32>) {')
    source = source.replace('func.return %r : tensor<2x3xi32>',
                            'func.return %r,%device : tensor<2x3xi32>,tensor<2x3xi32>')
    return source


def test_projected_host_layout_reaches_normal_preparation_and_retains_opaque_call(tmp_path):
    from merlin.runtime.linalg_numpy import evaluate

    source = tmp_path/'projected.mlir'
    source.write_text(text(parse_mlir_text(projected_source()), generic=True)+'\n')
    policy = read_host_graph_policy(software(tmp_path), target='neutral', entry='forward', connected_layout=True)
    work = tmp_path/'prepared'
    work.mkdir()
    prepared = _prepare_model_mlir(source, work, composed_host_layout=policy)
    before, after = parse_mlir_text(source.read_text()), parse_mlir_text(prepared.read_text())
    original_calls = [op for op in before.walk() if op.name == 'func.call']
    actual_calls = [op for op in after.walk() if op.name == 'func.call']
    assert len(actual_calls) == len(original_calls) == 1
    assert actual_calls[0].properties == original_calls[0].properties
    assert tuple(v.type for v in actual_calls[0].operands) == tuple(v.type for v in original_calls[0].operands)
    assert tuple(v.type for v in actual_calls[0].results) == tuple(v.type for v in original_calls[0].results)
    matrix = np.resize(np.array([-(2**31), 2**31-1, -1, 0, 1], dtype=np.int32), (2,3))
    bias = np.array([1, -1, 2**31-1], dtype=np.int32)
    for module in (before, after):
        result, observed = evaluate(module, [matrix, bias], function='forward')
        np.testing.assert_array_equal(result, matrix+bias)
        np.testing.assert_array_equal(observed, matrix)
    report = json.loads((work/'pre_group_layout/host-graph-optimization.json').read_text())
    assert report['layout']['before_transpose_bytes'] == 48
    assert report['layout']['after_transpose_bytes'] == 0
    pins = {}
    device = SimpleNamespace(pre_group_layout=policy, sibling_source=object())
    bind_group_layout_artifacts(device, work,
                                SimpleNamespace(bind_preparation=lambda role,path:pins.update({role:path})))
    assert len(pins) == 6 and 'pre_group_layout_source_binding' in pins
    source.write_text(source.read_text()+'\n// changed ordinary source\n')
    with pytest.raises(ValueError, match='projected source'):
        bind_group_layout_artifacts(device, work, SimpleNamespace(bind_preparation=lambda *args: None))


def test_projected_host_cannot_claim_layout_from_another_input_or_missing_binding(tmp_path):
    source = tmp_path/'projected.mlir'
    source.write_text(text(parse_mlir_text(projected_source()), generic=True)+'\n')
    policy = read_host_graph_policy(software(tmp_path), target='neutral', entry='forward')
    changed = parse_mlir_text(projected_source().replace('%sum = arith.addi', '%sum = arith.subi'))
    with pytest.raises(ValueError, match='input correspondence'):
        prepare_group_layout(changed, policy, tmp_path/'wrong-input', projected_source=source)
    work = tmp_path/'build'
    prepare_group_layout(parse_mlir_text(source.read_text()), policy, work/'pre_group_layout')
    with pytest.raises(ValueError, match='source binding'):
        bind_group_layout_artifacts(SimpleNamespace(pre_group_layout=policy,sibling_source=object()), work,
                                    SimpleNamespace(bind_preparation=lambda *args: None))


def test_projected_layout_executes_through_ordinary_native_lowering(tmp_path, upstream_host_tools):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    source = tmp_path/'projected.mlir'
    source.write_text(text(parse_mlir_text(projected_source()), generic=True)+'\n')
    policy = read_host_graph_policy(software(tmp_path), target='neutral', entry='forward', connected_layout=True)
    work = tmp_path/'prepared'
    work.mkdir()
    selected = _prepare_model_mlir(source, work, composed_host_layout=policy)
    matrix = np.arange(6,dtype=np.int32).reshape(2,3)-7
    bias = np.array([17,-19,23],dtype=np.int32)
    for arm, path in [('original',source),('selected',selected)]:
        lowered = lower_model(path.read_text(), tmp_path/arm, targets=('host',), static_arena=False)
        library = lowered.host_so
        invoke = HostModel.load(str(library))
        out, observed = np.full((2,3),-999,dtype=np.int32),np.full((2,3),-999,dtype=np.int32)
        invoke([(value.ctypes.data,value.shape) for value in (matrix,bias,out,observed)])
        np.testing.assert_array_equal(out,matrix+bias)
        np.testing.assert_array_equal(observed,matrix)


def test_many_opaque_boundaries_keep_every_observer_under_one_cone_budget(monkeypatch):
    from merlin.llvmlower import layout_conformance

    source = '''module {func.func private @device(tensor<2x3xi32>)->tensor<2x3xi32>
    func.func @forward(%x:tensor<2x3xi32>)->tensor<2x3xi32> {'''
    source += '\n'.join(f'%v{i}=func.call @device(%x):(tensor<2x3xi32>)->tensor<2x3xi32>' for i in range(64))
    source += 'return %v63:tensor<2x3xi32>}}'
    monkeypatch.setattr(layout_conformance,'_EXPRESSION_CACHE_LIMIT',8)
    expected = layout_conformance.layout_expression(parse_mlir_text(source))
    assert expected == layout_conformance.layout_expression(parse_mlir_text(source))
    changed = source.replace('%v63=func.call @device(%x)', '%v63=func.call @device(%v0)')
    assert expected != layout_conformance.layout_expression(parse_mlir_text(changed))
