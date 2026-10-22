"""Frozen public scalability inputs for developmental rolling Spike checks.

This adds no measurement authority and issues no full-suite qualification.
The existing public_scalability final gate remains the only suite issuer.
"""
from __future__ import annotations

from . import public_scalability


class Selection:
    def __init__(self, context):
        self.context = context
        self.selected = public_scalability.selected(context)
        self.cases, self.identity = {}, None
        if self.selected is not None:
            root, record = self.selected
            self.cases = {name: root / 'cases' / name for name in record['cases']}
            self.identity = {'root': str(root), 'record': record}

    def verify(self):
        if public_scalability.selected(self.context) != self.selected:
            raise ValueError('rolling public scalability selection changed')


class PerformanceFeedback:
    """Bounded observations of native products; never modifies numerical receipts.

    One request-wide observation budget excludes compile/grade waiting time.
    Feedback is projected only onto existing verdict rows after reopening its
    durable numerical receipt and the selected frozen suite.
    """
    def __init__(self, selection, *, compiler, runs_root, timeout):
        from pathlib import Path
        import threading
        import time
        from .joint_performance import deadline

        self.selection = selection
        self.compiler, self.runs_root = Path(compiler), Path(runs_root)
        self.remaining_s = max(0.0, deadline(timeout) - time.monotonic())
        self.records = {}
        self.lock = threading.Lock()

    def completed(self, row):
        import json
        import time
        from merlin.runtime import _native_files as F
        from merlin.targetgen.capsule_common import load_capsule
        from merlin.targetgen.capsule_runner import suite_for
        from .joint_performance import attach

        name = row.get('capsule')
        if name not in self.selection.cases:
            return
        with self.lock:
            if name in self.records:
                raise ValueError('duplicate rolling performance observation')
            case_root = self.runs_root / 'runs' / suite_for(self.selection.context.target) / name
            result_path = case_root / 'capsule_result.json'
            pin = F.pin(result_path, max_bytes=2 << 20)
            result = json.loads(result_path.read_bytes())
            if result.get('capsule') != name:
                raise ValueError('rolling performance result belongs to another capsule')
            # The original ordinary grader and producer are the sole source of
            # execution/numerical evidence; callback metadata supplies none.
            projection = {}
            started = time.monotonic()
            try:
                attach(projection, case_root / 'generated', result=result,
                    context=self.selection.context, compiler=self.compiler,
                    capsule=load_capsule(self.selection.cases[name]), remaining_s=self.remaining_s)
            finally:
                self.remaining_s = max(0.0, self.remaining_s - (time.monotonic() - started))
            if F.pin(result_path, max_bytes=2 << 20) != pin:
                raise ValueError('rolling numerical receipt changed during observation')
            # Existing attach emits these public projections. Do not copy native
            # consoles, private numerical data or qualification into feedback.
            feedback = {k: projection[k] for k in ('performance', 'compiler_diagnostics') if k in projection}
            if not feedback:
                return
            value = {'schema': 'merlin.rolling_scalability_performance.v1',
                'capsule': name, 'numerical_result': pin, 'feedback': feedback,
                'qualification_authority': False}
            output = self.runs_root / 'rolling_scalability_performance'
            output.mkdir(parents=True, exist_ok=True)
            path = output / (name + '.json')
            with path.open('x') as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False)
            self.records[name] = F.pin(path, max_bytes=2 << 20)

    def project(self, verdict):
        """Attach feedback, leaving every original grade/count/tier untouched."""
        import json
        from pathlib import Path
        from merlin.runtime import _native_files as F

        self.selection.verify()
        with self.lock:
            # Reopen all inputs before modifying any public verdict row.
            projections = {}
            for name, pin in self.records.items():
                if F.pin(pin['path'], max_bytes=2 << 20) != pin:
                    raise ValueError('rolling performance projection changed')
                value = json.loads(Path(pin['path']).read_bytes())
                result = value['numerical_result']
                if F.pin(result['path'], max_bytes=2 << 20) != result:
                    raise ValueError('rolling numerical result changed before feedback publication')
                projections[name] = value['feedback']
            for row in verdict.get('per_capsule', ()):
                row.update(projections.get(row.get('capsule'), {}))
