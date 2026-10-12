"""Queue priority requires the protected model admission, not request labels."""
import json
from types import SimpleNamespace

from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import model_gate_vouchers


def test_ready_model_precedes_same_source_ops_only_with_verified_voucher(tmp_path, monkeypatch):
    root = tmp_path / 'caps'
    root.mkdir()
    for name, kind in [('OP', 'op'), ('MODEL', 'model'), ('FORGED', 'model')]:
        cap = root / name
        cap.mkdir()
        (cap / 'capsule.yaml').write_text(f'kind: {kind}\n')
    monkeypatch.setattr(simjob._TP, '_submission_digest', lambda _: 'current')
    seen = []

    def lookup(ws, *, digest, capsule, **kwargs):
        seen.append((digest, capsule))
        return tmp_path / 'protected' if capsule == 'MODEL' else None

    monkeypatch.setattr(model_gate_vouchers, 'lookup', lookup)
    paths = []
    for index, name in enumerate(['OP', 'FORGED', 'MODEL']):
        path = tmp_path / f'simreq_{index}.json'
        path.write_text(json.dumps({'capsules': name, 'promoted': True,
                                    'submission_digest': 'current', 'ready_model': True}))
        paths.append(path)
    ordered = simjob._current_source_first(paths, tmp_path, context=SimpleNamespace(),
                                          capsules_root=root, cert_tier='L3')
    assert ordered == [paths[2], paths[0], paths[1]]
    assert ('current', 'FORGED') in seen
    # Failed protected verification confers no priority, including forged labels.
    monkeypatch.setattr(model_gate_vouchers, 'lookup', lambda *a, **k: None)
    assert simjob._current_source_first(paths, tmp_path, context=SimpleNamespace(),
                                       capsules_root=root, cert_tier='L3') == paths


def test_source_changes_during_priority_observation_keep_fifo(tmp_path, monkeypatch):
    values = iter(['first', 'changed'])
    monkeypatch.setattr(simjob._TP, '_submission_digest', lambda _: next(values))
    paths = [tmp_path / f'simreq_{i}.json' for i in range(2)]
    for path in paths:
        path.write_text(json.dumps({'promoted': True, 'submission_digest': 'first'}))
    assert simjob._current_source_first(reversed(paths), tmp_path) == paths
