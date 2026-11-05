import hashlib

from merlin_experiments.phase1.feedback.joint_runtime_observations import produce


def test_scoped_counts_do_not_become_waits_or_stage_prices(tmp_path):
    path=tmp_path/'report.json';path.write_text('{}')
    artifact={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    context={'compiler_sha256':'c','capsule_sha256':'k','scope_sha256':'s','model_sha256':'m','elf_sha256':'e'}
    native={'scope_sha256':'s','model_sha256':'m','elf_sha256':'e','execution_completed':True,'aborted_instructions':0,
            'counter_windows':[{'instructions':7}], 'feature_observation':{'scope_kind':'counter_pc_pair',
            'window_index':0,'scope_instructions':7,'counted_cpu_instructions':7,'complete':False},
            'feature_counts':{'target.commands':99,'cpu.fence':2},'estimated_cycles':80,'timing_complete':False}
    result=produce(native,context=context,artifact=artifact)
    assert result['observations']['cpu_sync']['emitted']=={'host_instructions':7}
    assert result['observations']['schedule_queues']['emitted']=={}
    assert not result['wholechain']['stage_composition_resolved']
    native['feature_observation']['complete']=True
    result=produce(native,context=context,artifact=artifact)
    assert result['observations']['schedule_queues']['emitted']=={'commands_total':99}
    assert result['observations']['cpu_sync']['emitted']['fences']==2
    context['elf_sha256']='different'
    assert produce(native,context=context,artifact=artifact)['selected_window'] is None


def test_pressure_uses_bound_emitted_order_and_refuses_loops(tmp_path):
    path = tmp_path / 'trace.json'
    path.write_text('{}')
    artifact = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    pins = dict(compiler_sha256='c', capsule_sha256='k', scope_sha256='s', model_sha256='m')
    load0 = {'decoded': {'spad_addr': 0}}
    load1 = {'decoded': {'spad_addr': 1}}
    use0 = {'decoded': {'a_spad': 0}}
    use1 = {'decoded': {'a_spad': 1}}
    def pressure(rows, context=pins):
        return produce({}, context=context, artifact=artifact, decoded_trace={
            'pins': pins, 'artifact': artifact, 'trace': {'instructions': rows}})['pressure']
    assert pressure([load0, load1, use0, use1])['peak_live_tiles'] == 2
    assert pressure([load0, use0, load1, use1])['peak_live_tiles'] == 1
    assert pressure([{'class': 'LOOP_WS'}, load0, use0])['status'] == 'UNKNOWN'
    assert pressure([load0], context={})['status'] == 'UNKNOWN'
    assert pressure([load0])['ranking_qualified'] is False


def test_physical_nested_counters_require_explicit_partition_premise(tmp_path):
    from merlin_experiments.phase1.feedback.joint_runtime_observations import profile_chain
    path = tmp_path / 'physical.json'
    path.write_text('{}')
    artifact = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    kwargs = dict(artifact=artifact, elf_sha256='a' * 64, hardware_sha256='b' * 64, scope_sha256='c' * 64)
    profiles = [{'result': {'cycles': 100, 'checked_elements': 5, 'mismatches': 0},
                 'input_cycles': 10, 'stage_cycles': [{'index': 0, 'cycles': 60}]}]
    unqualified = profile_chain(profiles, **kwargs)
    assert unqualified['windows'][0]['reported_sequential_host_stage_sum'] is None
    selected = profile_chain(profiles, **kwargs, sequential_host_intervals=True)
    assert selected['windows'][0]['unallocated_outer_cycles'] == 30
    assert not selected['overlap_or_savings_inferred']
    assert not selected['simulator_transfer_qualified']


def test_command_mix_requires_complete_class_population_and_preserves_requested_bytes(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text('{}')
    artifact = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    context = dict(compiler_sha256='c', capsule_sha256='k', scope_sha256='s', model_sha256='m', elf_sha256='e')
    counts = {'target.commands': 5, 'target.class.compute_ws': 2, 'target.class.preload': 2,
              'target.class.store_act1': 1, 'device.requested_write_bytes': 1024,
              'target.useful_shape_macs': 100, 'target.nominal_padded_macs': 128}
    native = dict(scope_sha256='s', model_sha256='m', elf_sha256='e', execution_completed=True,
                  aborted_instructions=0, counter_windows=[{'instructions': 8}], feature_counts=counts,
                  feature_observation=dict(scope_kind='counter_pc_pair', window_index=0,
                                           scope_instructions=8, counted_cpu_instructions=8, complete=True))
    metrics = produce(native, context=context, artifact=artifact)['observations']['schedule_queues']['emitted']
    assert metrics['compute_commands'] == 2 and metrics['store_commands'] == 1
    assert metrics['device_write_bytes'] == 1024
    assert metrics['useful_macs'] == 100 and metrics['padded_macs'] == 128
    assert 'dependency_wait_cycles' not in metrics
    counts['target.commands'] = 6
    assert 'compute_commands' not in produce(native, context=context, artifact=artifact)['observations']['schedule_queues']['emitted']
