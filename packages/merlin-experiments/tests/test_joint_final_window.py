"""A strict five-hour author window followed by one hour of final checks."""
import json

import pytest

from merlin_experiments.phase1 import joint_control as C, joint_epochs


def test_author_boundary_is_exactly_the_beginning_of_final_model_hour(tmp_path, monkeypatch):
    path = tmp_path/'policy.json'
    path.write_text(json.dumps({'version': 1, 'duration_s': 21600,
                               'final_model_s': 3600, 'drain_s': 3600}))
    policy = C.policy(path)
    start = 1000.0
    state = {'policy': policy, 'started_at_epoch': start}
    monkeypatch.setattr(C, 'selected', lambda: state)
    monkeypatch.setenv(C.ENV, str(tmp_path/'state.json'))
    monkeypatch.setattr(joint_epochs, 'stop_reason', lambda _: None)
    now = [start + 17999]
    monkeypatch.setattr(C.time, 'time', lambda: now[0])
    assert C.remaining(drain=True) == 1
    assert C.stop_reason() is None
    with pytest.raises(RuntimeError, match='final model window'):
        C.require_full_model()
    now[0] += 1
    assert C.remaining(drain=True) == 0
    assert C.stop_reason() == 'joint campaign freeze/drain boundary reached'
    assert C.remaining() == C.require_full_model() == 3600
    now[0] += 3600
    assert C.remaining() == 0
    with pytest.raises(RuntimeError, match='final model window'):
        C.require_full_model()


@pytest.mark.parametrize('drain,final', [(3601, 3601), (3600, 3599), (0, 3600)])
def test_larger_or_out_of_window_drain_still_refuses(tmp_path, drain, final):
    path = tmp_path/'policy.json'
    path.write_text(json.dumps({'version': 1, 'duration_s': 21600,
                               'final_model_s': final, 'drain_s': drain}))
    with pytest.raises(ValueError): C.policy(path)


def test_short_seeded_study_has_thirty_author_minutes_then_five_minute_drain(tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'version': 1, 'duration_s': 2100,
                               'final_model_s': 300, 'drain_s': 300}))
    state = {'policy': C.policy(path), 'started_at_epoch': 1000.0}
    monkeypatch.setattr(C, 'selected', lambda: state)
    monkeypatch.setenv(C.ENV, str(tmp_path / 'state.json'))
    monkeypatch.setattr(joint_epochs, 'stop_reason', lambda _: None)
    now = [2799.0]
    monkeypatch.setattr(C.time, 'time', lambda: now[0])
    assert C.remaining(drain=True) == 1 and C.stop_reason() is None
    now[0] = 2800.0
    assert C.remaining(drain=True) == 0
    assert C.stop_reason() == 'joint campaign freeze/drain boundary reached'
    assert C.require_full_model() == 300


@pytest.mark.parametrize('duration', [0, 299, 43201])
def test_campaign_duration_remains_bounded(tmp_path, duration):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'version': 1, 'duration_s': duration,
                               'final_model_s': 100, 'drain_s': 50}))
    with pytest.raises(ValueError, match='duration_s'):
        C.policy(path)
