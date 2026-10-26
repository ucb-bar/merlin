"""Suggestions must not promote missing capabilities, grants or IR obligations."""
import json
from pathlib import Path

import pytest
from merlin_experiments.phase1 import optimization_suggestions as S

REPO = Path(__file__).resolve().parents[3]


def catalog():
    return json.loads((REPO / 'docs/design/joint_optimization_catalog.json').read_text())


def state(report, name, field='all_mechanisms'):
    return next(row for row in report[field] if row['id'] == name)['status']


def test_actual_capabilities_and_api_intersection_without_target_name():
    c = catalog()
    caps, evidence = S.contract_traits({'name': 'arbitrary', 'endpoint_kind': 'inline_asm_insn',
                                       'compute_units': [{'kind': 'systolic'}]})
    caps['scheduling.persistent_configuration'] = 'supported'
    api = ['merlin.compile.scheduling.configuration']
    report = S.select(c, capabilities=caps, public_modules=api)
    assert state(report, 'configuration_cse') == 'supported'
    assert 'submission.cpu_issued' in evidence
    assert state(S.select(c, capabilities=caps, public_modules=[]), 'configuration_cse') == 'unsupported'
    caps.pop('scheduling.persistent_configuration')
    assert state(S.select(c, capabilities=caps, public_modules=api), 'configuration_cse') == 'unknown'
    assert S.contract_traits({'name': 'gemmini', 'readback_transport': 'coherent_dump_v1'}) == ({}, {})


@pytest.mark.parametrize('name,namespace,traits', [
    ('mixed_precision_fp4_fp6_fp8', 'numeric', 'packed_subbyte_float separate_accumulator_format scaling_metadata'),
    ('vector_processing_unit', 'compute', 'vector_lanes predicate_masks vector_register_files'),
    ('asynchronous_runtime', None, ''),
])
def test_generic_starters_require_each_explicit_trait(name, namespace, traits):
    caps = ({f'{namespace}.{t}': 'supported' for t in traits.split()} if namespace else
            {'submission.host_submitted_queues': 'supported', 'runtime.async_transfers': 'supported',
             'runtime.device_events': 'supported', 'memory.device_allocations': 'supported'})
    report = S.select(catalog(), capabilities=caps, public_modules=[])
    assert state(report, name, 'starter_sets') == 'supported'
    first = next(iter(caps))
    caps[first] = 'unsupported'
    assert state(S.select(catalog(), capabilities=caps, public_modules=[]), name, 'starter_sets') == 'unsupported'
    caps.pop(first)
    assert state(S.select(catalog(), capabilities=caps, public_modules=[]), name, 'starter_sets') == 'unknown'


def test_workload_evidence_cannot_come_from_chip_or_software_label():
    c = catalog()
    caps = {'memory.scratchpad': 'supported', 'workload.graph_fanout': 'supported'}
    modules = ['merlin.compile.scheduling.shared_input_schedule']
    assert state(S.select(c, capabilities=caps, public_modules=modules), 'sibling_input_reuse') == 'unknown'
    ir = 'builtin.module { func.func @f(%a: tensor<2xi8>) -> (tensor<2xi8>, tensor<2xi8>) { func.return %a, %a : tensor<2xi8>, tensor<2xi8> } }'
    workload = S.ir_traits(ir)
    assert workload['workload.multiple_live_outputs'] == 'supported'
    assert workload['workload.ranked_tensors'] == 'supported'
    assert 'workload.graph_fanout' not in workload  # two uses by one return are not sibling consumers
    unranked = 'builtin.module { func.func @f(%a: tensor<*xi8>) -> tensor<*xi8> { func.return %a : tensor<*xi8> } }'
    assert 'workload.ranked_tensors' not in S.ir_traits(unranked)
    assert 'workload.immutable_weights' not in workload
    assert 'workload.convolution' not in S.ir_traits(ir + '\n// linalg.conv_nhwc_hwcf')
    assert 'runtime.repeat_invocations' not in workload


def test_prepare_writes_compact_scoped_guidance(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / 'public_declarations').mkdir(parents=True)
    run.mkdir()
    (ws / 'public_declarations/software-spec.yaml').write_text('numerical_semantics:\n  operand_dtype: int8\n  accumulator_dtype: i32\n')
    inventory = {'contract_sha256': 'a'*64, 'public_modules': [{'module': 'merlin.llvmlower.requantization'}]}
    text = S.prepare(ws, run, repo=REPO, inventory=inventory)
    result = json.loads((ws / 'OPTIMIZATION_SUGGESTIONS.json').read_text())
    assert state(result, 'requant_proof') == 'supported'
    assert len(result['all_mechanisms']) == 52  # 45 original + 7 scheduling-library mechanisms
    assert len(result['starter_sets']) == 9
    assert len(result['suggestions']) <= 12 and len(result['discovery']) <= 12
    assert 'pass order' in text and 'no workload IR' in text
    assert (ws / 'optimization_suggest.py').read_bytes() == (run / 'optimization_suggest.py').read_bytes()
    assert 'selected_gemmini' not in (ws / 'OPTIMIZATION_CATALOG.json').read_text()


def test_invalid_or_contradictory_declarations_do_not_become_support():
    with pytest.raises(ValueError):
        S.declarations({'chip': {'magic': 'supported'}})
    with pytest.raises(ValueError):
        S.declarations({'compute': {'systolic': True}})
    with pytest.raises(ValueError):
        S.software_traits({'optimization_capabilities': {'numeric': {'int8_storage': 'unsupported'}},
                           'numerical_semantics': {'operand_dtype': 'int8'}})


def test_storage_geometry_never_proves_async_visibility_or_physical_aliasing():
    values, evidence = S.storage_traits({'facts': {'memory_address_spaces': [
        {'name': 'scratchpad', 'view_kind': 'address_space', 'bytes': 4096},
        {'name': 'accumulator', 'view_kind': 'address_space', 'bytes': 0}]}})
    assert values['memory.scratchpad'] == 'supported'
    assert 'memory.accumulator' not in values
    assert 'scheduling.async_issue' not in values
    assert 'memory.host_coherence' not in values
    assert set(values) == set(evidence)
