"""Cache phases authenticate source selection, actual cache words, and oracle lineage."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import cache_composition as driver
from cache_composition import ORDER, parent_caches, phase_inputs, stage_inputs
from atlas_mlir_native.reduction import REDUCTION_POLICY
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
def info(value):
    return dict(sha256=sha256(value.tobytes()).hexdigest(), shape=list(value.shape), dtype=value.dtype.str)
def model_manifest(past=4, query=1):
    inputs = [dict(index=i, role=name) for i, name in enumerate(('input_ids', 'position_ids', 'attention_mask'))]
    outputs = [dict(index=0, role='logits', source_node='logits')]
    external, sources, links = [], {}, []
    for position, kind in enumerate(ORDER):
        boundaries, names = ['embedding' if position == 0 else 'hidden_input'], [f'hidden_{position}']
        if kind.startswith('layer'):
            for offset, operand in enumerate(('key', 'value')):
                name, new_name = f'past_{operand}_{position}', f'updated_{operand}_{position}'
                common = dict(layer=position, dtype='torch.bfloat16')
                entry = dict(common, index=3 + 2 * position + offset, role=f'past_{operand}', source_node=name, shape=[1, 1, past, 2])
                inputs.append(entry)
                external.append(dict(entry, region=kind, input_index=len(boundaries), boundary=name, global_input_index=entry['index']))
                boundaries.append(name)
                names.append(new_name)
                outputs.append(dict(common, index=1 + 2 * position + offset, role=f'updated_{operand}', source_node=new_name, shape=[1, 1, past + query, 2]))
        sources[kind] = dict(kind=kind, source_boundaries=boundaries, source_outputs=names, model_sha256=kind + '-source')
        links.append(dict(kind=kind, inputs=[] if position == 0 else [dict(input_index=0, region=ORDER[position - 1], output_index=0)]))
    abi = dict(layer_count=22, batch_size=1, kv_heads=1, head_dim=2, past_length=past, query_length=query, total_length=past + query, inputs=inputs, outputs=outputs, cache_representation='bf16_post_rope_pre_gqa')
    phase = dict(phase='decode5' if past else 'prefill4', cache_capture_sha256='c' * 64, binding=dict(model_pt2_sha256='f' * 64, abi=abi))
    whole = dict(schema='atlas.sequential_cache_phase_fx.v1', regions=ORDER, cache_phase=phase, external_cache_inputs=external, fixture_boundary_allowlist=['embedding'], links=links)
    return dict(source_model_sha256='f' * 64, cache_phase=phase, whole_model=whole, regions=list(sources.values())), sources
@pytest.mark.parametrize('mode', [None, 'missing', 'duplicate', 'layer', 'unresolved', 'skipped_hidden', 'metadata', 'prefill', 'implicit'])
def test_phase_manifest(mode):
    manifest, sources = model_manifest(0, 4) if mode in ('prefill', 'implicit') else model_manifest()
    sources = copy.deepcopy(sources) if mode == 'metadata' else sources
    rows, abi = manifest['whole_model']['external_cache_inputs'], manifest['cache_phase']['binding']['abi']
    if mode == 'missing': rows.pop()
    elif mode == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif mode == 'layer': rows[0]['region'] = 'layer1'
    elif mode == 'unresolved': sources['layer1']['source_boundaries'].append('framework_hidden')
    elif mode == 'skipped_hidden': manifest['whole_model']['links'][7]['inputs'][0]['region'] = 'layer0'
    elif mode == 'metadata': sources['layer0']['tensors'] = {'input_0': {'sha256': 'modified-preparation-identity'}}
    elif mode in ('prefill', 'implicit'):
        abi.update(initial_cache_empty=True, inputs=abi['inputs'][:3])
        manifest['whole_model']['external_cache_inputs'] = []
        for source in sources.values(): source['source_boundaries'] = source['source_boundaries'][:1]
        if mode == 'implicit': abi['past_length'] = 1
    if mode not in (None, 'prefill'):
        with pytest.raises(ValueError): phase_inputs(manifest, sources)
    else:
        checked, external = phase_inputs(manifest, sources)
        assert (checked['past_length'], sum(map(len, external.values()))) == ((0, 0) if mode else (4, 44))
        if mode: assert checked['initial_cache_empty']
@pytest.mark.parametrize('mode', [None, 'missing', 'nonempty', 'shape'])
def test_stage_cache_inputs(mode):
    captured = np.zeros((1, 1, 4, 2), dtype=np.uint16)
    actual = captured + np.uint16(0x3f80)
    source = dict(kind='layer0', source_boundaries=['cache'], tensors={'input_0': info(captured)})
    role = dict(input_index=0, layer=0, role='past_key', shape=list(captured.shape))
    caches = {} if mode in ('missing', 'nonempty') else {(0, 'past_key'): ( actual[:, :, :3] if mode == 'shape' else actual, {'parent_sha256': info(actual)['sha256']})}
    def run(): return stage_inputs(source, [], {'input_0': captured}, Path('.'), {}, [role], caches, 0 if mode == 'nonempty' else 4)
    if mode:
        with pytest.raises(KeyError if mode == 'missing' else ValueError): run()
    else:
        overrides, rows = run()
        assert overrides['input_0'].tobytes() == actual.tobytes() != captured.tobytes() and rows[0]['input_origin'] == 'generated_cache_actual'
@pytest.mark.parametrize('mode', [None, 'words', 'running', 'length', 'duplicate', 'metadata'])
def test_parent_cache_publications(tmp_path, mode):
    execution, regions = tmp_path / 'execution', tmp_path / 'regions'
    parent, sources = model_manifest(0, 4)
    current, _ = model_manifest(4, 1)
    write(regions / 'manifest.json', parent)
    rows = []
    for layer, kind in enumerate(ORDER):
        source = sources[kind]
        write(regions / kind / 'source.json', source)
        (execution / kind).mkdir(parents=True)
        arrays = {'output_0': np.ones((1, 4, 2), dtype=np.uint16)}
        if kind.startswith('layer'):
            arrays.update(output_1=np.full((1, 1, 4, 2), 0x3f80 + layer, dtype=np.uint16), output_2=np.full((1, 1, 4, 2), 0x3f90 + layer, dtype=np.uint16))
        np.savez(execution / kind / 'actual.npz', **arrays)
        publication = dict(source_model_sha256=source['model_sha256'], outputs=[info(v) for v in arrays.values()])
        write(execution / kind / 'execution.json', publication)
        rows.append(dict(kind=kind, execution=publication, mismatches=0))
    receipt = dict(schema='atlas.generated_cache_phase.v1', status='PASS', mismatches=0, requested_regions=ORDER, source_model_sha256=parent['source_model_sha256'], manifest_sha256=driver.file_hash(regions / 'manifest.json'), cache_capture_sha256='c' * 64, temporal_sequential_execution=True, oracle_chain_inputs=False, regions=rows)
    if mode == 'words':
        np.savez(execution / 'layer0/actual.npz', output_0=np.ones((1, 4, 2), np.uint16), output_1=np.zeros((1, 1, 4, 2), np.uint16), output_2=np.zeros((1, 1, 4, 2), np.uint16))
    elif mode == 'running': receipt['status'] = 'RUNNING'
    elif mode == 'length': current['cache_phase']['binding']['abi']['past_length'] = 3
    elif mode == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif mode == 'metadata':
        sources['layer0']['source_outputs'].reverse()
        write(regions / 'layer0/source.json', sources['layer0'])
    write(execution / 'receipt.json', receipt)
    def run(): return parent_caches(execution, regions, current, current['cache_phase']['binding']['abi'])
    if mode:
        with pytest.raises(ValueError): run()
    else:
        values, _, _ = run()
        assert len(values) == 44 and np.all(values[21, 'past_value'][0] == 0x3f90 + 21)
@pytest.fixture
def cache_case(tmp_path, monkeypatch):
    def create(order=('layer0',), compile_only=True):
        regions, compiled, execution, export, oracle = (tmp_path / name for name in ('regions', 'compiled', 'execution', 'export', 'oracle'))
        value = np.array([0x3f80], dtype=np.uint16)
        generated, sources, stages, commands = value + np.uint16(1), [], [], []
        for kind in order:
            source = regions / kind
            source.mkdir(parents=True)
            (source / 'model.pt2').write_bytes((kind + '-mock-archive').encode())
            record = dict(kind=kind, model_sha256=driver.file_hash(source / 'model.pt2'), source_boundaries=['embedding' if kind == 'layer0' else 'hidden'], source_outputs=['output'], tensors={'input_0': info(value)})
            write(source / 'source.json', record)
            sources.append(record)
        manifest = dict(source_model_sha256='phase-model', cache_phase=dict(phase='prefill4', cache_capture_sha256='capture'), regions=sources, whole_model=dict(links=[dict(kind=kind, inputs=[] if i == 0 else [dict(input_index=0, region=order[i-1], output_index=0)]) for i, kind in enumerate(order)]))
        write(regions / 'manifest.json', manifest)
        export.mkdir()
        for name in ('manifest.json', 'policy.json'): (export / name).write_bytes(('frozen source ' + name).encode())
        contract, policy = (driver.file_hash(export / name) for name in ('manifest.json', 'policy.json'))
        def compile_stage(command, **kwargs):
            assert Path(command[1]).name == 'compile_capsule.py'
            commands.append(command)
            destination = Path(command[command.index('--output') + 1])
            source_hash = driver.file_hash(regions / destination.name / 'model.pt2')
            memory = {name.replace('--', '').replace('-', '_'): int(command[command.index(name) + 1]) for name in ('--io-memory-bytes', '--application-memory-bytes')}
            write(destination / 'compiled.json', dict(status='COMPILED', kind=destination.name, source_model_sha256=source_hash, constant_bytes_identical=True, merge_adjacent_device_regions='--merge-adjacent-device-regions' in command, **memory))
            precision = dict(reduction=REDUCTION_POLICY)
            if destination.name == 'head':
                from merlin.compile.c_bf16_linear import SOURCE_POLICY
                precision['host_bf16_linear'] = SOURCE_POLICY
            write(destination / 'precision.json', precision)
            if '--adapter-export' in command:
                adapter = dict(schema='m2m.quantization_manifest.v2', adapter_id='atlas_fp8', contract_sha256=contract, policy_sha256=policy, atlas=dict(calls=[dict(target='atlas.qlinear.default')], saved_native_program=dict(sha256=source_hash, no_native_calls=False)))
                for name in ('adapter-manifest.json', 'capture/quantization-manifest.json'): write(destination / name, adapter)
        argv = [driver.__file__, '--model-python', sys.executable] + (['--compile-only'] if compile_only else [])
        for name, path in dict(regions=regions, compiled=compiled, execution=execution, atlas_root=tmp_path, model_root=tmp_path).items(): argv += ['--' + name.replace('_', '-'), str(path)]
        monkeypatch.setattr(sys, 'argv', argv)
        monkeypatch.setattr(driver, 'ORDER', list(order))
        monkeypatch.setattr(driver, 'phase_inputs', lambda *_: ({'past_length': 0}, {kind: [] for kind in order}))
        monkeypatch.setattr(driver.subprocess, 'run', compile_stage)
        return dict(argv=argv, regions=regions, compiled=compiled, execution=execution, oracle=oracle, export=export, contract=contract, policy=policy, commands=commands, root=tmp_path, value=value, generated=generated, compile=compile_stage)
    return create
def adapter_arguments(case):
    case['argv'] += ['--adapter-export', str(case['export']), '--expected-contract-sha256', case['contract'], '--expected-policy-sha256', case['policy']]
@pytest.mark.parametrize('memory', [None, 4325376])
@pytest.mark.parametrize('adapter', [False, True])
def test_driver_compiles_then_reuses_exact_source_and_memory_selection(cache_case, memory, adapter):
    case = cache_case()
    if memory: case['argv'] += ['--io-memory-bytes', str(memory)]
    if adapter: adapter_arguments(case)
    driver.main()
    command, = case['commands']
    expected = {'--io-memory-bytes': str(memory or 67108864), '--application-memory-bytes': '536870912', '--allocation-timeout-ms': '1000'}
    if adapter:
        expected.update({'--adapter-export': str(case['export']), '--expected-contract-sha256': case['contract'], '--expected-policy-sha256': case['policy']})
    assert all(command[command.index(flag) + 1] == value for flag, value in expected.items())
    assert '--external-parameters' in command
    receipt = driver.read(case['execution'] / 'receipt.json')
    assert (receipt['status'], receipt['requested_io_memory_bytes'], receipt['requested_application_memory_bytes']) == ('COMPILED', memory or 67108864, 536870912)
    assert bool(receipt['requested_adapter_selection']) == bool(receipt['regions'][0]['adapter_bindings']) == adapter
    case['argv'][case['argv'].index('--execution') + 1] = str(case['root'] / 'reuse')
    driver.main()
    assert len(case['commands']) == 1

@pytest.mark.parametrize('timing,mutation', [ *[('reuse', name) for name in ('io_memory_bytes', 'application_memory_bytes', 'contract', 'policy', 'saved_source', 'legacy_only', 'non_boolean', 'false_no_calls', 'capture')], *[(timing, status) for timing in ('fresh', 'reuse') for status in (None, 'FAILED', 'RUNNING')], ('export', 'policy'), ('late', 'capture'), ('late', 'compilation')])
def test_driver_refuses_changed_selection(cache_case, monkeypatch, timing, mutation):
    case = cache_case()
    memory = mutation in ('io_memory_bytes', 'application_memory_bytes')
    adapter = timing == 'export' or mutation in ('contract', 'policy', 'saved_source', 'legacy_only', 'non_boolean', 'false_no_calls', 'capture')
    if adapter: adapter_arguments(case)
    def mutate():
        destination = case['compiled'] / 'layer0'
        if timing == 'export':
            (case['export'] / 'policy.json').write_bytes(b'changed policy')
            return
        if mutation == 'capture':
            write(destination / 'capture/quantization-manifest.json', {})
            return
        path = destination / ('adapter-manifest.json' if adapter else 'compiled.json')
        record = driver.read(path)
        if adapter:
            saved = record['atlas']['saved_native_program']
            if mutation in ('contract', 'policy'): record[mutation + '_sha256'] = '0' * 64
            elif mutation == 'saved_source': saved['sha256'] = '0' * 64
            elif mutation == 'legacy_only': saved['host_only'] = saved.pop('no_native_calls')
            elif mutation == 'non_boolean': saved['no_native_calls'] = 0
            elif mutation == 'false_no_calls': record['atlas']['calls'] = []
            write(destination / 'capture/quantization-manifest.json', record)
        elif memory: record[mutation] += 2048
        elif mutation == 'compilation': record['constant_bytes_identical'] = False
        elif mutation is None: record.pop('status')
        else: record['status'] = mutation
        write(path, record)
    if timing in ('reuse', 'export'):
        if timing == 'reuse':
            driver.main()
            case['argv'][case['argv'].index('--execution') + 1] = str(case['root'] / 'refused')
        mutate()
    else:
        original = driver.verify_compiled_selection if timing == 'late' else driver.subprocess.run
        def change_after(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(driver if timing == 'late' else driver.subprocess, 'verify_compiled_selection' if timing == 'late' else 'run', change_after)
    message = ('source receipt digest' if timing == 'export' else 'memory profile' if memory else 'adapter source pins or attestation' if adapter else 'publication changed during phase' if timing == 'late' else 'completed compilation')
    with pytest.raises(ValueError, match=message): driver.main()
    execution = Path(case['argv'][case['argv'].index('--execution') + 1])
    if timing == 'export':
        assert not case['commands'] and not execution.exists()
    else:
        receipt = driver.read(execution / 'receipt.json')
        assert receipt['status'] == 'FAILED'
        if mutation in (None, 'FAILED', 'RUNNING'): assert receipt['regions'] == []
        assert len(case['commands']) == 1

@pytest.mark.parametrize('mode', [None, 'wrong_publication', 'coordinated_before', 'coordinated_during', 'wrong_archive', 'late_actual', 'late_execution', 'late_oracle_stage', 'late_oracle_root', 'root_flags', 'root_manifest', 'root_roster', 'root_parent', 'root_capture'])
def test_two_stage_cli_authenticates_actual_words_and_oracle(cache_case, monkeypatch, mode):
    case = cache_case(('layer0', 'head'), compile_only=False)
    regions, compiled, oracle, execution = (case[name] for name in ('regions', 'compiled', 'oracle', 'execution'))
    value, generated = case['value'], case['generated']
    output_info, stages = info(generated), []
    for kind in ('layer0', 'head'):
        case['compile'](['python', 'compile_capsule.py', '--output', str(compiled / kind), '--io-memory-bytes', '67108864', '--application-memory-bytes', '536870912'])
        np.savez(compiled / kind / 'tensors.npz', input_0=value)
        reference = dict(stage=kind, status='EXPECTED', source_model_sha256=driver.file_hash(regions / kind / 'model.pt2'), ordinary_framework_kernels_called=False, semantic_json_or_emitted_program_inputs=False, constant_bytes_identical=True, outputs={'output_0': output_info})
        write(oracle / kind / 'reference.json', reference)
        np.savez(oracle / kind / 'expected.npz', output_0=generated)
        stages.append(reference)
    reference = dict(status='EXPECTED', source_model_sha256='phase-model', schema='atlas.independent_original_fx_sequence.v1', source_case=0,
        oracle_mode='independent_cache_phase', parent_reference_receipt_sha256=None, cache_capture_sha256='capture', model_arithmetic_sources={'source': 'pin'}, validation_sha256='validation',
        capsule_manifest_sha256=driver.file_hash(regions / 'manifest.json'), stages=stages, ordinary_framework_kernels_called=False, compiler_or_emitted_program_inputs=False)
    root_mutations = dict(root_flags=('ordinary_framework_kernels_called', True), root_manifest=('capsule_manifest_sha256', 'different-manifest'), root_roster=('stages', []), root_parent=('parent_reference_receipt_sha256', 'different-parent'), root_capture=('cache_capture_sha256', 'different-capture'))
    if mode in root_mutations:
        field, replacement = root_mutations[mode]
        reference[field] = replacement
    write(oracle / 'reference.json', reference)
    oracle_hash = driver.file_hash(oracle / 'reference.json')
    if mode == 'wrong_archive': (regions / 'head' / 'model.pt2').write_bytes(b'changed mock archive')
    def change_expected(kind):
        changed = np.array([0x4000], dtype=np.uint16)
        np.savez(oracle / kind / 'expected.npz', output_0=changed)
        stage_reference = driver.read(oracle / kind / 'reference.json')
        stage_reference['outputs']['output_0'] = info(changed)
        write(oracle / kind / 'reference.json', stage_reference)
        return stage_reference
    if mode == 'coordinated_before': change_expected('head')
    def publish(command, **kwargs):
        stage = Path(command[command.index('--execution-output') + 1])
        with np.load(stage / 'generated-predecessor-inputs.npz') as arrays:
            consumed = arrays['input_0'] if 'input_0' in arrays.files else value
        if stage.name == 'head':
            assert consumed.tobytes() == generated.tobytes() != value.tobytes()
        np.savez(stage / 'actual.npz', output_0=generated)
        published = info(generated)
        if stage.name == 'head' and mode == 'wrong_publication':
            published['sha256'] = 'different-execution-words'
        if stage.name == 'head' and mode == 'coordinated_during': change_expected('head')
        if stage.name == 'head' and mode == 'late_actual':
            np.savez(execution / 'layer0' / 'actual.npz', output_0=np.array([0x4000], dtype=np.uint16))
        if stage.name == 'head' and mode == 'late_execution':
            changed = driver.read(execution / 'layer0' / 'execution.json')
            changed['outputs'][0]['sha256'] = 'changed-after-hidden-consumption'
            write(execution / 'layer0' / 'execution.json', changed)
        if stage.name == 'head' and mode in ('late_oracle_stage', 'late_oracle_root'):
            changed = change_expected('layer0')
            if mode == 'late_oracle_root':
                changed_root = driver.read(oracle / 'reference.json')
                changed_root['stages'][0] = changed
                write(oracle / 'reference.json', changed_root)
        write(stage / 'execution.json', dict(source_model_sha256=driver.file_hash(regions / stage.name / 'model.pt2'), immutable_inputs=True, status='EXECUTED_UNCOMPARED', backend='functional', inputs=[info(consumed)], outputs=[published]))
    case['argv'] += ['--oracle', str(oracle)]
    monkeypatch.setattr(driver.subprocess, 'run', publish)
    if mode is None: driver.main()
    else:
        with pytest.raises(ValueError): driver.main()
        if execution.exists():
            assert driver.read(execution / 'receipt.json')['status'] == 'FAILED'
        if mode != 'late_oracle_root':
            assert driver.file_hash(oracle / 'reference.json') == oracle_hash
        return
    receipt = driver.read(execution / 'receipt.json')
    assert receipt['status'] == 'PASS' and len(receipt['regions']) == 2
    assert {receipt['regions'][-1]['execution']['outputs'][0]['sha256'], receipt['regions'][-1]['outputs'][0]['sha256'], receipt['seam_closure'][0]['sha256']} == {output_info['sha256']}
    assert receipt['oracle_receipt_sha256'] == oracle_hash
