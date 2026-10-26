"""Observe sealed shape/integer projection admission on the fresh registry examples.

The shipped capture/projection code is the subject, not an independent oracle.
No Lean kernel run, qualification, receipt or review approval is issued.
Run with PYTHONPATH=<pinned producer>/src and the main checkout interpreter.
"""
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

OUT = Path(__file__).resolve().parent
INDEX = json.loads((OUT / 'source-index.json').read_text())
PRODUCER = Path(INDEX['producer'])
VERIFY = Path(INDEX['verification'])
EXAMPLES = OUT.parent / 'registry-eight' / 'producer-examples'
PROBES = OUT / 'projection-probes-sealed'

package = types.ModuleType('review_primary')
package.__path__ = [str(VERIFY / 'src/merlin/verify')]
sys.modules[package.__name__] = package
shape = importlib.import_module('review_primary.lean_graph_loops_shape_projection')

env = dict(os.environ, PYTHONPATH=str(PRODUCER / 'src'), PYTHONDONTWRITEBYTECODE='1')
results = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(case, route, folder, completed, status, **extra):
    (folder / 'stdout.txt').write_text(completed.stdout)
    (folder / 'stderr.txt').write_text(completed.stderr)
    result = dict(case=case, route=route, returncode=completed.returncode, status=status,
                  lean_run=False, qualification_issued=False, independent_correspondence_review=False, **extra)
    (folder / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    results.append(result)


def probe_shape(case, before, after):
    folder = PROBES / case / 'shape'
    folder.mkdir(parents=True)
    pair = folder / 'actual-pair.json'
    command = [sys.executable, str(VERIFY / 'src/merlin/verify/lean_graph_loops_shape_capture.py'),
               '--source', str(PRODUCER), '--revision', INDEX['producer_revision'],
               '--before', str(before), '--after', str(after), '--output', str(pair)]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode:
        return record(case, 'shape', folder, completed, 'capture_failed_not_a_semantic_refusal')
    data = json.loads(pair.read_text())
    try:
        model = shape.project_exports(data['before_export'], data['after_export'], index_bits=64, max_nodes=10000)
    except ValueError as error:
        return record(case, 'shape', folder, completed, 'projection_refused', reason=str(error))
    path = folder / 'model.json'
    path.write_text(json.dumps(model, sort_keys=True, indent=2) + '\n')
    record(case, 'shape', folder, completed, 'projection_emitted_not_kernel_checked',
           model=str(path.relative_to(OUT)), sha256=sha(path))


def probe_integer(case, before, after):
    folder = PROBES / case / 'integer'
    folder.mkdir(parents=True)
    command = [sys.executable, str(VERIFY / 'src/merlin/verify/lean_graph_state_integer_capture.py'),
               '--reference', str(PRODUCER), '--destination', str(folder / 'actual'),
               '--buffers', str(before), str(after)]
    completed = subprocess.run(command, cwd=PRODUCER, env=env, capture_output=True, text=True, timeout=120, check=False)
    capture, rejection = folder / 'actual/CAPTURE.json', folder / 'actual/REJECTION.json'
    if completed.returncode == 0 and capture.is_file():
        return record(case, 'integer', folder, completed, 'projection_emitted_not_kernel_checked',
                      model=str(capture.relative_to(OUT)), sha256=sha(capture))
    if completed.returncode == 1 and rejection.is_file():
        refused = json.loads(rejection.read_text())
        return record(case, 'integer', folder, completed, 'projection_refused', reason=refused['reason'])
    record(case, 'integer', folder, completed, 'capture_failed_not_a_semantic_refusal')


PROBES.mkdir(exist_ok=False)
for directory in sorted(EXAMPLES.iterdir()):
    capture = json.loads((directory / 'capture.json').read_text())
    if capture['status'] != 'actual_changed_producer_output':
        continue
    before, after = directory / 'before.mlir', directory / 'after.mlir'
    if capture['kind'] == 'shape_view_normalization':
        probe_shape(directory.name, before, after)
    elif capture['kind'] == 'exact_integer_simplification':
        probe_integer(directory.name, before, after)

report = dict(method='local admission probes of sealed shape/integer projections; shipped projections are subjects, not oracles',
              producer_revision=INDEX['producer_revision'], verification_revision=INDEX['verification_revision'],
              new_lean_qualification=False, cases=results)
(OUT / 'projection-probes-sealed.json').write_text(json.dumps(report, indent=2) + '\n')
for result in results:
    print(result['case'], result['route'], result['status'], result.get('reason', ''))
