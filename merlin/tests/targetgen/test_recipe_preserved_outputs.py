


def test_reductions_preserve_narrow_and_integer_calls_before_observation():
    import json

    from test_recipe_affine_movement_families import _CONTRACTION, _ELEMENTWISE, _contract, _recipe

    # Use the capability-derived standalone recipe, without a runtime-weight opt-in.
    recipe = _recipe(_contract(_CONTRACTION, _ELEMENTWISE)).to_dict()
    _framework('import json\nrecipe = json.loads(' + repr(json.dumps(recipe)) + ')\n' + r'''
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from merlin.targetgen._recipe_quantizer import apply_recipe, CalibrationRefused, NoEligibleRecipeOperations
from merlin.targetgen._m2m_capture_worker import _integerized_agreement
from merlin.targetgen.pt2e_reference_selection import reference_selection, run_selected_reference
from merlin.capture.integerization import integerize_capture_model, contraction_partition, non_contraction_annotations
torch.set_num_threads(1)
class Mixed(torch.nn.Module):
    def forward(self, narrow, integer, empty, finite):
        return (torch.ops.aten.adaptive_avg_pool1d.default(narrow, [2]).reshape(-1),
                torch.ops.aten.avg_pool1d.default(integer, [2]).reshape(-1),
                torch.sum(empty, dim=[0], dtype=empty.dtype).reshape(-1),
                torch.mean(finite).reshape(-1))
for dtype in (torch.bfloat16, torch.float16):
    inputs = (torch.tensor([[[float('nan'), float('inf'), -float('inf'), 1.]]], dtype=dtype),
              torch.arange(4, dtype=torch.int64).reshape(1, 1, 4),
              torch.empty(0, dtype=dtype), torch.arange(6, dtype=torch.float32).reshape(2, 3))
    model = Mixed().eval()
    converted = apply_recipe(model, recipe, example_inputs=inputs)
    stats = converted._recipe_quantization_stats
    assert stats['annotated_means'] == 1 and stats['annotated_contractions'] == 0
    assert len(stats['operators_refused_by_placement']) == 3
    for expected, actual in zip(model(*inputs)[:3], converted(*inputs)[:3], strict=True):
        assert expected.dtype == actual.dtype
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    independent, _ = run_selected_reference(reference_selection(converted), converted, inputs,
                                           declared_readout=True, numerics='integer_readout')
    with TemporaryDirectory() as directory:
        integerized, receipt = integerize_capture_model(converted, inputs, numerics='integer_readout',
                                                        directory=Path(directory))
    # The worker binds zero contractions to the positive non-contraction census.
    receipt.update(integer_program='non_contraction', non_contraction_annotations=non_contraction_annotations(stats))
    assert contraction_partition(receipt) == dict(seen=0, integerized=0, preserved=0)
    report = _integerized_agreement(independent.output, integerized(*inputs), atol=0, rtol=0)
    assert report['status'] == 'passed' and len(report['outputs']) == 4
    assert all(row['byte_equal'] for row in report['outputs'])
    for leaf in (0, 3):
        changed = [x.clone() for x in independent.output]
        changed[leaf].view(torch.uint8)[0] ^= 1
        assert _integerized_agreement(independent.output, tuple(changed), atol=0, rtol=0)['status'] == 'failed'
class Mean(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dtype=x.dtype)
# Unsupported calls cannot manufacture eligible work, even with finite inputs.
for dtype in (torch.bfloat16, torch.float16, torch.float64):
    try:
        apply_recipe(Mean(), recipe, example_inputs=(torch.ones(4, dtype=dtype),))
    except NoEligibleRecipeOperations:
        pass
    else:
        raise AssertionError('a narrow/wide-only reduction claimed integer work')
# Actual FP32 observer edges retain the finite-calibration gate.
for value in (float('nan'), float('inf'), 3e38):
    try:
        apply_recipe(Mean(), recipe, example_inputs=(torch.full((4,), value),))
    except CalibrationRefused as error:
        assert error.code == 'nonfinite_calibration'
    else:
        raise AssertionError('an invalid FP32 observation was accepted')
''')
