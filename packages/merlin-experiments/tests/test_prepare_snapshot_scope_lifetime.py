"""Admission cannot return success/refusal before its final content check closes."""
from contextlib import contextmanager

import pytest
from merlin_experiments.phase1 import session


@pytest.mark.parametrize('outcome', ['success', 'refusal', 'exception', 'tampered'])
def test_prepare_closes_scope_before_publish_even_on_refusal_or_exception(monkeypatch, outcome):
    events = []

    @contextmanager
    def verified():
        events.append('full-before')
        try:
            yield
        finally:
            events.append('full-after')
            if outcome == 'tampered':
                raise RuntimeError('snapshot changed during preparation')

    result = object()

    def inner(*args, snapshot_scopes, **kwargs):
        snapshot_scopes.enter_context(verified())
        events.append('prepared')
        if outcome == 'exception':
            raise ValueError('admission failed')
        return 4 if outcome == 'refusal' else result

    monkeypatch.setattr(session, '_prepare', inner)
    if outcome == 'tampered':
        with pytest.raises(RuntimeError, match='snapshot changed'):
            session.prepare(None, None, None, workspace_leases=[])
    elif outcome == 'exception':
        with pytest.raises(ValueError, match='admission failed'):
            session.prepare(None, None, None, workspace_leases=[])
    else:
        assert session.prepare(None, None, None, workspace_leases=[]) == (4 if outcome == 'refusal' else result)
    assert events == ['full-before', 'prepared', 'full-after']
