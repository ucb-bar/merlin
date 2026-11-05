import hashlib
import json

from merlin.common.tree_hash import hash_tree
from merlin_experiments.phase1.feedback.joint_diagnostics_feedback import attach


def test_existing_observation_automatically_reaches_six_section_feedback(tmp_path):
    compiler = tmp_path / 'compiler'
    compiler.mkdir()
    (compiler / 'main.py').write_text('pass\n')
    compiler_sha = hash_tree(compiler)['sha256']
    report = {'scope_sha256': 's', 'model_sha256': 'm', 'elf_sha256': 'e',
              'execution_completed': True, 'aborted_instructions': 0,
              'counter_windows': [{'instructions': 7}],
              'feature_observation': {'scope_kind': 'counter_pc_pair', 'window_index': 0,
                                      'scope_instructions': 7, 'counted_cpu_instructions': 7, 'complete': True},
              'feature_counts': {'target.commands': 3, 'cpu.fence': 1}}
    path = tmp_path / 'observer.json'
    path.write_text(json.dumps(report))
    observation = {'status': 'observed', 'report': {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    receipt = {'schema': 'merlin.compiler_source_observations.v1',
               'pins': {'compiler_sha256': compiler_sha, 'capsule_sha256': 'k', 'scope_sha256': 's'},
               'sections': {'crosslayer_lifetimes': {'maturity': 'rewritten_llvm_not_linked_execution',
                                                    'missing': ['runtime peak memory'],
                                                    'planned': {'peak_live_bytes': 10},
                                                    'emitted': {'arena_bytes': 32}}}}
    (tmp_path / 'joint_compiler_observations.json').write_text(json.dumps(receipt))
    performance = dict(capsule_sha256='k', scope_sha256='s', model_sha256='m', status='UNKNOWN',
                       feature_counts=report['feature_counts'])
    row = {'performance': performance, 'capsule': 'case', 'status': 'pass'}
    attach(row, tmp_path, report=report, observation=observation,
           result={'numeric': {'status': 'pass'}}, compiler=compiler)
    assert row['performance'] is performance
    diagnostic = row['compiler_diagnostics']
    assert len(diagnostic['sections']) == 6
    assert diagnostic['sections']['cpu_sync']['emitted'] == {'host_instructions': 7, 'fences': 1}
    assert diagnostic['sections']['crosslayer_lifetimes']['emitted'] == {'arena_bytes': 32}
    assert diagnostic['sections']['wholechain']['status'] == 'UNKNOWN'
    assert not diagnostic['promotion_authority']
    brief = diagnostic['feedback_brief']
    assert brief['additional_target_runs'] == brief['guest_instructions_added_by_feedback'] == 0
    storage = brief['sections']['crosslayer_lifetimes']
    assert storage['maturity'] == 'rewritten_llvm_not_linked_execution'
    assert storage['planned']['peak_live_bytes'] == 10
    assert storage['emitted'] == {'arena_bytes': 32}
    assert storage['limits'] == ['runtime peak memory']
    assert brief['sections']['cpu_sync']['origin'] == 'native_observer'
    assert 'completion_wait_cycles' in brief['sections']['cpu_sync']['missing_metrics']
    assert not brief['physical_savings_claimed']
    from merlin_experiments.phase1.feedback.public_scalability import _observed_result, public_result
    public = public_result({'observed': {'case': _observed_result(row)}, 'score': {}})
    displayed = public['cases'][0]['performance']['feedback_brief']
    assert displayed['performance_signals']['scoped_counts']['target.commands'] == 3
    assert displayed['experiment_record']['compiler_sha256'] == compiler_sha
    assert displayed['experiment_record']['hypothesis'] is None
    assert str(tmp_path) not in json.dumps(displayed)
    receipt['pins']['compiler_sha256'] = 'foreign'
    (tmp_path / 'joint_compiler_observations.json').write_text(json.dumps(receipt))
    attach(row, tmp_path, report=report, observation=observation,
           result={'numeric': {'status': 'pass'}}, compiler=compiler)
    assert row['compiler_diagnostics']['sections']['crosslayer_lifetimes']['emitted'] == {}

    assert row['compiler_diagnostics']['feedback_brief']['sections']['crosslayer_lifetimes']['emitted'] == {}


def test_native_scoped_feedback_matches_offline_and_reaches_brief():
    from merlin_timing.scoped_resources import project, compact
    from merlin_experiments.phase1.feedback.joint_performance_report import project as automatic
    from merlin_experiments.phase1.feedback.joint_outcome_brief import performance_signals
    from pathlib import Path
    source=Path(__file__).resolve().parents[3]/'experiments/reference-data/scoped-resources-20261023/on-report.json'
    report=json.loads(source.read_text())
    observation=dict(status='observed',complete_counts=True,scope_binding=report['scope_binding'])
    # This retained public probe executes 24 windows; the automatic harness's
    # existing single-window contract must continue to refuse that whole report.
    assert automatic(report,observation,{},selection_sha256='e'*64,reason='test')['status']=='UNKNOWN'
    # Synthetic single-window transport fixture checks shared projection only.
    # It is never saved as native evidence or used to grant execution credit.
    report['counter_windows']=report['counter_windows'][:1]
    perf=automatic(report,observation,{},selection_sha256='e'*64,reason='synthetic transport test')
    assert perf['estimated_whole_soc_cycles']==354
    assert perf['scoped_resource_feedback']==project(report)
    brief=performance_signals(perf)['scoped_resource_feedback']
    assert brief==compact(project(report))
    assert len(brief['resources'])==16 and brief['resource_rows_omitted']==3
    assert not brief['qualification_issued']
