"""One completed case must not diagnose every unobserved historical executable."""
from io import StringIO
from types import SimpleNamespace

from merlin_experiments.phase1.feedback.promotion import _log_invalidations


class State:
    verdicts = {'L2': object(), 'L3': object()}
    def __init__(self, name):
        self.name, self.calls = name, []
    def invalidated_by(self, tier):
        self.calls.append(tier)
        return [SimpleNamespace(component='lowering')]


def test_joint_callback_does_not_invalidate_unobserved_capsules():
    current, historical = State('current'), State('historical')
    log = StringIO()
    _log_invalidations([current, historical], {}, {}, 'L2', 'L3', log, observed_names={'current'})
    assert current.calls == ['L2', 'L3'] and historical.calls == []
    assert 'current L3 invalidated by' in log.getvalue()
    assert 'historical L3 invalidated' not in log.getvalue()
    assert '1 historical capsule(s) not re-observed' in log.getvalue()


def test_ordinary_full_grade_diagnostics_keep_all_existing_cases():
    first, second = State('first'), State('second')
    _log_invalidations([first, second], {}, {}, 'L2', 'L3', StringIO())
    assert first.calls == second.calls == ['L2', 'L3']
