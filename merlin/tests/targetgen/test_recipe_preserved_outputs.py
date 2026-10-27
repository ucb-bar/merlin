"""Host-only values do not enter observers or acquire invented floating metrics."""

import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _recipe_quantizer as RQ


def _framework(script):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    m2m = os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("an explicit capture interpreter and frontend are required")
    result = subprocess.run(
        [python, "-I", "-B", "-c",
         "import sys; sys.path[:0] = sys.argv[1:3];\n" + script,
         str(Path(RQ.__file__).parents[2]), m2m],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_exact_special_outputs_are_counted_and_changed_components_still_refuse():
    _framework(r'''
import torch
from merlin.targetgen._recipe_quantizer import agreement, RecipeError, NonfiniteReference
torch.set_num_threads(1)
identity = lambda x: x
specials = [torch.tensor([float('nan'), float('inf'), -float('inf'), 1.]),
            torch.tensor([1+2j, -3+4j]), torch.tensor(float('nan'))]
for value in specials:
    report = agreement(identity, lambda x: x.clone(), [(value,)])
    assert report == dict(samples=1, exact_preserved_samples=1, preserved_output_policy='exact_bytes_v1')
    assert 'cosine_min' not in report
    changed = value.clone()
    changed.reshape(-1)[-1] = 2
    try:
        agreement(identity, lambda x: changed, [(value,)])
    except RecipeError:
        pass
    else:
        raise AssertionError('a changed special output was accepted')
# Neither opposite infinity signs nor a changed NaN payload is preserved.
x = specials[0]
for index, mask in [(0, 1), (1, -2147483648)]:
    y = x.clone()
    y.view(torch.int32)[index] ^= mask
    try:
        agreement(identity, lambda _: y, [(x,)])
    except NonfiniteReference:
        pass
    else:
        raise AssertionError('changed special-value bytes were accepted')
# Finite results are still scored even beside an exactly preserved host output.
reference = lambda x: (specials[0], x)
report = agreement(reference, lambda x: (specials[0].clone(), x+1), [(torch.tensor([1., 2.]),)])
assert report['samples'] == 1 and report['max_abs_error'] == 1
assert report['outputs'][0] == dict(output_index=0, samples=1, exact_preserved_samples=1)
assert report['outputs'][1]['max_abs_error'] == 1
# Preservation and finite scoring in different samples keep the sample census.
report = agreement(identity, identity, [(specials[0],), (torch.ones(4),)])
assert report['samples'] == 2 and report['exact_preserved_samples'] == 1
assert report['max_abs_error'] == 0
''')


def test_only_observer_edges_are_required_to_be_finite():
    _framework(r'''
import torch
from merlin.targetgen._recipe_quantizer import apply_recipe, agreement, CalibrationRefused
torch.set_num_threads(1)
recipe = dict(schema='quant_recipe_v1', status='derived', families=['contraction'],
    runtime_weight_contractions=True,
    weight=dict(dtype='int8', granularity='tensor', symmetric=True, quant_min=-127, quant_max=127),
    activation=dict(dtype='int8', granularity='tensor', symmetric=True, quant_min=-128, quant_max=127, mode='static'))
class Mixed(torch.nn.Module):
    def forward(self, a, b, host):
        return a @ b, host.clone()
a, b = torch.ones(2, 2), torch.ones(2, 2)
for host in [torch.tensor([1+2j]), torch.tensor([float('nan'), float('inf')])]:
    inputs = (a, b, host)
    converted = apply_recipe(Mixed(), recipe, example_inputs=inputs)
    report = agreement(Mixed(), converted, [inputs])
    assert report['outputs'][1]['exact_preserved_samples'] == 1
# A nonfinite operand or intermediate actually observed for quantization still refuses.
class Chain(torch.nn.Module):
    def forward(self, a, b, c):
        return (a @ b) @ c
for value in [float('nan'), float('inf'), 3e38]:
    try:
        apply_recipe(Chain(), recipe, example_inputs=(torch.full((2,2), value), 2*b, a))
    except CalibrationRefused as error:
        assert error.code == 'nonfinite_calibration'
    else:
        raise AssertionError('nonfinite observer input/state was accepted')
''')
