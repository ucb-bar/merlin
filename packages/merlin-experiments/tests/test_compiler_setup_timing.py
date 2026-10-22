import contextlib
import json

import pytest

from merlin_experiments.phase1.feedback import setup_timing as T


def test_boundaries_visible_and_original_exception_preserved(tmp_path):
    visits = []
    @contextlib.contextmanager
    def manager():
        visits.append('enter')
        try:
            yield 'value'
        finally:
            visits.append('exit')
    with T.selected(tmp_path, 'a' * 64) as timeline:
        with pytest.raises(ValueError, match='original'):
            with T.timed_context('verify', manager()) as value:
                assert value == 'value'
                doc = json.loads(timeline.path.read_text())
                assert doc['stages'][0]['status'] == 'returned'
                with T.stage('policy'):
                    raise ValueError('original')
    doc = json.loads(timeline.path.read_text())
    assert visits == ['enter', 'exit']
    assert [r['stage'] for r in doc['stages']] == ['verify.enter', 'policy', 'verify.exit']
    assert doc['stages'][1]['exception_type'] == 'ValueError'
    assert all(r['elapsed_s'] >= 0 for r in doc['stages'])
    assert doc['authority'] == 'diagnostic_only'


def test_suppression_and_failed_entry_not_changed(tmp_path):
    @contextlib.contextmanager
    def suppress():
        try:
            yield
        except ValueError:
            pass
    with T.selected(tmp_path, 'b' * 64) as timeline:
        with T.timed_context('suppress', suppress()):
            raise ValueError('handled')
        @contextlib.contextmanager
        def refuse():
            raise RuntimeError('custody')
            yield
        with pytest.raises(RuntimeError, match='custody'):
            with T.timed_context('refuse', refuse()):
                pytest.fail('refused context body ran')
    rows = json.loads(timeline.path.read_text())['stages']
    assert [r['stage'] for r in rows] == ['suppress.enter', 'suppress.exit', 'refuse.enter']
    assert rows[-1]['status'] == 'raised'
