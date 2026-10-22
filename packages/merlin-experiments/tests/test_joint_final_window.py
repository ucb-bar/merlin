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
