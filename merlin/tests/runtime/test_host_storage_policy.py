"""Actual ordered host-call lifetimes reuse storage under original permissions."""

import ctypes
import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml
from test_arena_bind import _alloc, _clang, _guarded_free

from merlin.llvmlower.device_preparation import freeze_device_offload_selection, require_device_offload_selection
from merlin.llvmlower.host_graph_policy import read_host_graph_policy
from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA
from merlin.llvmlower.host_storage_policy import prepare, recheck


def policy(tmp_path, **overrides):
    software = tmp_path/'software.yaml'
    software.write_text(yaml.safe_dump({
        'schema': 'merlin.software_spec.v1', 'status': 'reviewed', 'target': 'neutral',
        'host_optimization': {'schema': POLICY_SCHEMA, 'permitted_stages': ['llvm'],
                              'permitted_optimizations': ['static_arena'], 'sequential_execution': True,
                              'successful_allocations': True, 'non_reentrant_entry': True, **overrides}}))
    return read_host_graph_policy(software, target='neutral', entry='forward')


def connected_source():
    return '''declare ptr @malloc(i64)
declare void @free(ptr)
declare void @stage(ptr, ptr)
define void @forward(ptr %input, ptr %out) {
''' + _alloc('a', 128) + '''
  call void @stage(ptr %input, ptr %ap)
  %av = load float, ptr %ap
  store float %av, ptr %out
''' + _guarded_free('a', 'fa', 'middle') + '\n' + _alloc('b', 128) + '''
  call void @stage(ptr %out, ptr %bp)
  %bv = load float, ptr %bp
  store float %bv, ptr %out
''' + _guarded_free('b', 'fb', 'last') + '\n  ret void\n}\n'


def test_actual_compiled_connected_calls_share_only_disjoint_storage(tmp_path):
    cc = _clang()
    if cc is None:
        pytest.skip('no selected host clang')
    selected_policy = policy(tmp_path)
    source = tmp_path/'original.ll'
    source.write_text(connected_source())
    pins = {}
    selected = prepare(source, selected_policy, tmp_path/'storage',
                       SimpleNamespace(bind_preparation=lambda role, path: pins.update({role: path})))
    report = json.loads((tmp_path/'storage/report.json').read_text())
    assert report['automatic_selection']['status'] == 'applied'
    assert report['arena']['bound'] == 2 and report['arena']['reuse_factor'] == 2
    assert source.read_text() == connected_source() and selected != source and len(pins) == 5
    assert selected.read_text().count('call void @stage') == 2
    stub = tmp_path/'stage.c'
    stub.write_text('void stage(const float *x,float *y) { *y = *x + 3.0f; }')
    for label, llvm in [('original', source), ('selected', selected)]:
        library = tmp_path/f'{label}.so'
        result = subprocess.run([cc, '-O2', '-shared', '-fPIC', str(llvm), str(stub), '-o', str(library)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        model = ctypes.CDLL(str(library))
        for value in [0.0, 11.5, -7.25]:
            input_value, output = ctypes.c_float(value), ctypes.c_float()
            model.forward(ctypes.byref(input_value), ctypes.byref(output))
            assert output.value == value + 6.0
    recheck(tmp_path/'storage', selected_policy)


@pytest.mark.parametrize('missing', ['sequential_execution', 'successful_allocations', 'non_reentrant_entry'])
def test_no_inferred_storage_permissions(tmp_path, missing):
    selected_policy = policy(tmp_path, **{missing: False})
    source = tmp_path/'original.ll'
    source.write_text(connected_source())
    output = prepare(source, selected_policy, tmp_path/'storage', SimpleNamespace(bind_preparation=lambda *args: None))
    assert output == source
    record = json.loads((tmp_path/'storage/report.json').read_text())
    assert record['automatic_selection'] == {'status': 'unchanged', 'reason': 'permission_not_admitted'}
    assert record['arena'] is None


def test_source_policy_and_lifetime_products_remain_bound(tmp_path):
    selected_policy = policy(tmp_path)
    device = SimpleNamespace(device='neutral', intermediate_storage=selected_policy)
    frozen = freeze_device_offload_selection(device)
    source = tmp_path/'original.ll'
    source.write_text(connected_source())
    prepare(source, selected_policy, tmp_path/'storage', SimpleNamespace(bind_preparation=lambda *args: None))
    require_device_offload_selection(device, frozen)
    device.intermediate_storage = policy(tmp_path)
    with pytest.raises(ValueError, match='selection changed'):
        require_device_offload_selection(device, frozen)
    output = tmp_path/'storage/selected.ll'
    output.write_text(output.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        recheck(tmp_path/'storage', selected_policy)


def test_explicit_incompatible_projection_keeps_original_and_reason(tmp_path):
    selected_policy = policy(tmp_path)
    source = tmp_path/'original.ll'
    source.write_text(connected_source())
    output = prepare(source, selected_policy, tmp_path/'storage', SimpleNamespace(bind_preparation=lambda *args: None),
                     blocked_reason='descriptor_results_selected')
    assert output == source
    assert json.loads((tmp_path/'storage/report.json').read_text())['automatic_selection']['reason'] == (
        'descriptor_results_selected')
