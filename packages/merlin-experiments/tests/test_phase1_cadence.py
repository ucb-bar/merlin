"""Single-flight grading follows new source promptly without duplicate stable grades."""
import json

from merlin_experiments.phase1.feedback import lifecycle as L


def test_start_cadence_regrades_changed_source_and_skips_unchanged_source(tmp_path, monkeypatch):
    import threading

    monkeypatch.setenv('MERLIN_GRADE_START_TO_START', '1')
    ws = tmp_path/'workspace'
    (ws/'qa').mkdir(parents=True)
    (ws/'qa/verdict.json').write_text(json.dumps({'existing': True}))
    state = {'time': 0, 'source': 'A'}
    waits, graded = [], []

    class Event:
        done = False

        def is_set(self):
            return self.done

        def set(self):
            self.done = True

        def wait(self, delay):
            if self.done:
                return True
            waits.append(delay)
            state['time'] += delay
            if len(waits) == 4:
                state['source'] = 'C'
            assert len(waits) <= 5
            return False

    event = Event()

    class Thread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(threading, 'Event', lambda: event)
    monkeypatch.setattr(threading, 'Thread', Thread)
    monkeypatch.setattr(L.time, 'monotonic', lambda: state['time'])

    def grade(*args, **kwargs):
        graded.append(state['source'])
        state['time'] += 900 if len(graded) == 1 else 10
        if len(graded) == 1:
            state['source'] = 'B'
        return {'snapshot_current': len(graded) != 1}

    def tick(index):
        if index == 3:
            event.set()

    L.start_background(ws, tmp_path/'run', L.GradeCadence(300, False, 600),
        interval_grades=True, grade_callback=grade, fast_grade_callback=lambda *a: None,
        source_identity=lambda: state['source'], on_tick=tick)
    assert graded == ['A', 'B', 'C']
    assert waits == [600, 0, 590, 30]
