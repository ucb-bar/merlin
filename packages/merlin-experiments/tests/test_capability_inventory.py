"""Public planning preserves source obligations without reading numerical answers."""
import json

import pytest

from merlin_experiments.phase1 import capability_inventory as C
from merlin_experiments.phase1 import capability_source as S
from merlin_experiments.phase1 import run_inputs as R


def capsule(tmp_path, name, *, size=4, dtype='i8'):
    root = tmp_path / name
    root.mkdir()
    text = f'''builtin.module {{
      func.func @main(%x: tensor<{size}x{dtype}>) -> tensor<{size}x{dtype}> {{
        func.return %x : tensor<{size}x{dtype}>
      }}
    }}'''
    (root / 'source.mlir').write_text(text)
    return {'name': name, 'label': 'public', '__dir__': str(root),
            'interface_mlir': 'source.mlir', 'kind': 'operator',
            'operation': {'op': 'movement'},
            'inputs': [{'name': 'x', 'dtype': dtype, 'shape': [size]}],
            'numeric_policy': {'compare': 'exact_int', 'dtype': dtype},
            'required_oracle_tiers': ['L0', 'L1', 'L2', 'L3']}


def policy():
    return {'correctness_curriculum': {'original_capsules': ['small']},
            'gsim_correctness': {'capsules': ['small']},
            'gsim': {'witness_capsules': ['calibration']}}


def inventory(tmp_path):
    small = capsule(tmp_path, 'small')
    diagnostic = capsule(tmp_path, 'calibration', size=12)
    large = capsule(tmp_path, 'large', size=1024)
    small['oracle'] = {'path': 'not-opened', 'secret': 'PRIVATE_CANARY'}
    small['capture_tool'] = 'PRIVATE_CANARY'
    return C.build([small, diagnostic], scalability=[large], policy=policy())


def test_full_roster_cohorts_and_engine_obligations(tmp_path):
    v = inventory(tmp_path)
    assert v['total'] == 3
    assert v['cohorts'] == {'original_correctness': 1, 'calibration': 1, 'scalability_performance': 1}
    small, calibration, large = (v['members'][k] for k in ('small', 'calibration', 'large'))
    assert small['development_engines'] == ['spike', 'gsim']
    assert calibration['gsim_role'] == 'question_selected_calibration'
    assert calibration['development_engines'] == ['spike']
    assert large['development_engines'] == ['spike'] and large['gsim_role'] == 'not_selected'
    assert large['performance_priority'] == 'primary'
    assert small['performance_priority'] == 'diagnostic'
    assert 'PRIVATE_CANARY' not in json.dumps(v)
    assert not v['qualification_authority']
    group, = v['capability_groups'].values()
    assert group['small_canary_candidate'] == 'small'
    assert group['large_generalization_candidate'] == 'large'
    assert len(C.summary(v)) < 2048


def test_missing_original_hidden_and_duplicate_members_refuse(tmp_path):
    small = capsule(tmp_path, 'small')
    large = capsule(tmp_path, 'large')
    with pytest.raises(ValueError, match='omits'):
        C.build([large], policy=policy())
    with pytest.raises(ValueError, match='duplicate'):
        C.build([small], scalability=[small], policy=policy())
    hidden = dict(small, label='hidden')
    with pytest.raises(ValueError, match='non-public'):
        C.build([hidden], policy=policy())
    p = policy()
    p['gsim_correctness']['capsules'] = []
    with pytest.raises(ValueError, match='every original'):
        C.build([small], policy=p)
    # A whitelist may include additional operator probes outside this roster;
    # never expose their IDs or invent public source requirements for them.
    v = C.build([small], policy=policy())
    assert v['witnesses_outside_public_roster'] == 1
    assert 'calibration' not in v['members']


def test_source_types_graph_and_unknown_analysis(tmp_path):
    cap = capsule(tmp_path, 'small')
    source = tmp_path / 'small/source.mlir'
    source.write_text('''builtin.module {
      func.func @main(%x: i32) -> i32 {
        %a = arith.addi %x, %x : i32
        %b = arith.muli %a, %x : i32
        %c = arith.addi %a, %b : i32
        func.return %c : i32
      }
    }''')
    info = S.analyze(cap)
    assert info['status'] == 'parsed'
    graph = info['graph']
    assert graph['shared_producers']
    assert all(a < b for a, b in graph['producer_consumer_edges'])
    assert len(graph['operations']) == graph['nodes']
    source.write_text('not valid mlir')
    assert S.analyze(cap)['status'] == 'unknown'
    source.unlink()
    assert S.analyze(cap)['reason'] == 'declared public interface is unavailable'


def test_source_cannot_escape_public_capsule(tmp_path):
    cap = capsule(tmp_path, 'small')
    private = tmp_path / 'private.mlir'
    private.write_text('PRIVATE_CANARY')
    cap['interface_mlir'] = '../private.mlir'
    with pytest.raises(ValueError, match='escapes'):
        S.analyze(cap)
    cap['interface_mlir'] = 'alias.mlir'
    (tmp_path / 'small/alias.mlir').symlink_to(private)
    with pytest.raises(ValueError, match='escapes'):
        S.analyze(cap)


def test_shape_generalization_does_not_merge_numeric_contracts(tmp_path):
    small = capsule(tmp_path, 'small')
    large = capsule(tmp_path, 'large', size=1000, dtype='i32')
    v = C.build([small], scalability=[large], policy=policy())
    assert len(v['capability_groups']) == 2
    types = v['members']['large']['source_analysis']['tensor_types']
    assert {'shape': [1000], 'dtype': 'i32'} in types


def test_staged_inventory_is_sealed_and_legacy_treatment_stays_compatible(tmp_path):
    ws, run, bundle = (tmp_path / n for n in ('workspace', 'run', 'bundle'))
    for root in (ws, run, bundle):
        root.mkdir()
    for root in (ws, run):
        (root / 'TASK.md').write_text('Existing task\n')
    for root in (run, bundle):
        (root / 'input_bundle_manifest.yaml').write_text('{}\n')
    legacy = R.treatment_snapshot_record(ws, run, bundle, ())
    assert not any('PUBLIC_CAPABILITIES' in r['name'] for r in legacy['files'])
    v = inventory(tmp_path)
    C.stage(ws, run, v)
    record = R.treatment_snapshot_record(ws, run, bundle, ())
    assert len([r for r in record['files'] if 'PUBLIC_CAPABILITIES' in r['name']]) == 4
    assert len([r for r in record['files'] if C.QUERY in r['name']]) == 2
    assert (ws / C.QUERY).stat().st_mode & 0o222 == 0
    assert (ws / 'TASK.md').read_bytes() == (run / 'TASK.md').read_bytes()
    assert C.FILENAME in (ws / 'TASK.md').read_text()
    (ws / C.FILENAME).write_text('{}')
    with pytest.raises(RuntimeError):
        R.verify_treatment_snapshot(record, ws, run, bundle, ())


def test_selected_staging_uses_native_roster_and_verifies_scalability(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen import capsule_common
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import rolling_scalability

    small, cal, large = [capsule(tmp_path, n) for n in ('small', 'calibration', 'large')]
    calls = []
    monkeypatch.setattr(joint_control, 'selected', lambda: {'policy': policy()})
    monkeypatch.setattr(capsule_common, 'discover_capsules', lambda *a, **kw: [small, cal])
    monkeypatch.setattr(capsule_common, 'load_capsule', lambda *a, **kw: large)
    monkeypatch.setattr(rolling_scalability, 'Selection', lambda context: SimpleNamespace(
        cases={'large': tmp_path / 'large'}, verify=lambda: calls.append('verified')))
    for n in ('workspace', 'run'):
        (tmp_path / n).mkdir()
        (tmp_path / n / 'TASK.md').write_text('Task\n')
    result = C.stage_selected(tmp_path/'workspace', tmp_path/'run', context=object(),
                              public_roots=[tmp_path], contract=tmp_path, excluded=lambda names: [])
    assert result['total'] == 3 and calls == ['verified']
    monkeypatch.setattr(joint_control, 'selected', lambda: None)
    assert C.stage_selected(tmp_path, tmp_path, context=None, public_roots=[], contract=None, excluded=None) is None
