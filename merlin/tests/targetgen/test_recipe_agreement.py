"""A correct first result must not hide a damaged cache or malformed later result."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen import quant_recipe as QR

SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
import _recipe_quantizer as RQ
torch.set_num_threads(1)
mask = torch.ones((1, 3), dtype=torch.bool)
x = torch.tensor([1., 2., 4.])
ref = lambda x: (mask, x)
changed = RQ.agreement(ref, lambda x: (mask, x * torch.tensor([1., 1., -1.])), [(x,)])
small = RQ.agreement(lambda x: (torch.ones(10000), x[:1]),
                     lambda x: (torch.ones(10000), -x[:1]), [(x,)])
nested = RQ.agreement(lambda x: {'mask': mask, 'cache': [x, x + 1]},
                      lambda x: {'mask': mask, 'cache': [x, x + 2]}, [(x,)])

# Complete compatibility with the historical single-output arithmetic/receipt.
samples = [(x,), (torch.tensor([-2., 1., 3.]),)]
reference = lambda x: x
candidate = lambda x: x * .75 + .125
cosines, agreed, worst = [], 0, 0.
for sample in samples:
    expected, actual = reference(*sample).float().flatten(), candidate(*sample).float().flatten()
    cosines.append(float(torch.nn.functional.cosine_similarity(expected, actual, dim=0)))
    agreed += int(int(expected.argmax()) == int(actual.argmax()))
    worst = max(worst, float((expected - actual).abs().max()))
legacy = dict(samples=2, argmax_agreement=agreed/2, cosine_min=min(cosines),
              cosine_mean=sum(cosines)/2, max_abs_error=worst)
single = RQ.agreement(reference, candidate, samples)
assert single == legacy and 'outputs' not in single

bad = {
    'arity': lambda x: (mask,),
    'structure': lambda x: [mask, x],
    'shape': lambda x: (mask, x.reshape(1, 3)),
    'nonfinite': lambda x: (mask, torch.full_like(x, float('inf'))),
    'nontensor': lambda x: (mask, 'cache'),
    'complex': lambda x: (mask, x.to(torch.complex64)),
}
errors = {}
for name, function in bad.items():
    try:
        RQ.agreement(ref, function, [(x,)])
    except RQ.RecipeError as error:
        assert not isinstance(error, RQ.NonfiniteReference), name
        errors[name] = str(error)
    else:
        raise AssertionError('accepted malformed result: ' + name)

# A NaN/inf in the floating reference itself is typed, whatever the candidate computes; a value
# that is finite at its own width but not at float32 is no such reference.
references = {}
for name, special in {'inf': float('inf'), '-inf': float('-inf'), 'nan': float('nan')}.items():
    try:
        RQ.agreement(lambda x: (mask, x * special), ref, [(x,), (x,)])
    except RQ.NonfiniteReference as error:
        references[name] = dict(code=error.code, details=error.details)
    else:
        raise AssertionError('accepted a nonfinite reference: ' + name)
wide = torch.tensor([1e300], dtype=torch.float64)
try:
    RQ.agreement(lambda x: (mask, wide), lambda x: (mask, wide), [(x,)])
except RQ.NonfiniteReference:
    raise AssertionError('a finite float64 reference is not a nonfinite reference')
except RQ.RecipeError as error:
    references['narrowed'] = str(error)

def bounded():
    yield (x,)
    yield (x,)
    raise AssertionError('agreement consumed beyond its sample limit')
limited = RQ.agreement(reference, reference, bounded(), limit=2)
dynamic = RQ.agreement(reference, reference, [(x,), (x[:2],)])
large = torch.linspace(-1., 1., 2_000_000)
large_exact = RQ.agreement(lambda x: (mask, x), lambda x: (mask, x), [(large,)])
empty = RQ.agreement(reference, reference, [], limit=2)
zero = RQ.agreement(reference, reference, bounded(), limit=0)
print(json.dumps(dict(changed=changed, small=small, nested=nested, single=single,
                     errors=errors, references=references,
                     limited=limited, dynamic=dynamic, empty=empty, zero=zero, large_exact=large_exact)))
"""


@pytest.fixture(scope="module")
def observed():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the capture interpreter is not configured")
    completed = subprocess.run(
        [python, "-c", SCRIPT, str(Path(worker.__file__).parent)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_matching_mask_does_not_hide_a_changed_cache(observed):
    row = observed["changed"]
    assert row["max_abs_error"] == 8 and row["cosine_mean"] < 0.99
    assert row["outputs"][0]["max_abs_error"] == 0
    assert row["outputs"][1]["max_abs_error"] == 8


def test_a_large_correct_output_cannot_hide_the_other_outputs_receipt(observed):
    row = observed["small"]
    assert row["cosine_mean"] > 0.99
    assert row["outputs"][1]["cosine_mean"] == -1


def test_nested_results_retain_complete_output_correspondence(observed):
    rows = observed["nested"]["outputs"]
    assert [row["output_index"] for row in rows] == [0, 1, 2]
    assert [row["max_abs_error"] for row in rows] == [0, 0, 1]


@pytest.mark.parametrize("case", ["arity", "structure", "shape", "nonfinite", "nontensor", "complex"])
def test_malformed_later_results_refuse_even_when_the_mask_matches(observed, case):
    assert observed["errors"][case]


@pytest.mark.parametrize("special", ["inf", "-inf", "nan"])
def test_a_nonfinite_reference_is_a_typed_refusal_not_a_candidate_failure(observed, special):
    refusal = observed["references"][special]
    assert refusal["code"] == "nonfinite_reference"
    assert refusal["details"] == {"sample": 0, "output_index": 1, "nonfinite_count": 3}


def test_only_the_reference_at_its_own_width_selects_the_typed_refusal(observed):
    assert observed["references"]["narrowed"] == "agreement output 1 is not finite"


def test_single_result_receipts_and_dynamic_extents_remain_supported(observed):
    assert observed["single"]["samples"] == 2 and "outputs" not in observed["single"]
    assert observed["dynamic"]["samples"] == 2 and observed["dynamic"]["max_abs_error"] == 0


def test_agreement_respects_the_exact_sample_budget(observed):
    assert observed["limited"]["samples"] == 2
    assert observed["empty"] == observed["zero"] == {"samples": 0}


def test_large_exact_outputs_use_stable_diagnostic_accumulation(observed):
    report = observed["large_exact"]
    assert report["multi_output_metric_policy"] == "fp32_values_fp64_accumulation_v1"
    assert report["cosine_mean"] == pytest.approx(1.0, abs=1e-12)
    assert report["outputs"][1]["cosine_mean"] == pytest.approx(1.0, abs=1e-12)
    assert report["outputs"][1]["max_abs_error"] == 0


def test_real_quantized_capture_reports_the_second_tensor(tmp_path):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text("""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[.111, .337], [-.732, .287]]))
    def forward(self, x):
        return torch.ones(1, dtype=torch.bool), self.projection(x)
def get_model_and_inputs():
    return Model().eval(), (torch.tensor([[.75, -1.25], [1.25, .5]]),)
""")
    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    recipe = dict(
        schema="quant_recipe_v1",
        target="synthetic",
        unit="matrix",
        status="derived",
        weight={**tensor, "quant_min": -127, "quant_max": 127},
        activation={**tensor, "quant_min": -128, "quant_max": 127},
        families=["contraction"],
        bias_domain="accumulator",
        software_numerical_engine="integer_reference",
    )
    recipe["recipe_sha256"] = QR.digest(recipe)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    out = tmp_path / "capture"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(worker.__file__).parents[2]), os.environ.get("PYTHONPATH", "")]),
    }
    completed = subprocess.run(
        [
            python,
            worker.__file__,
            "--m2m-dir",
            m2m,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads((out / "meta.json").read_text())["recipe_agreement"]
    assert report["samples"] == 1 and report["max_abs_error"] > 0
    assert report["outputs"][0]["max_abs_error"] == 0
    assert report["outputs"][1]["max_abs_error"] > 0


def test_capture_refuses_a_nonfinite_float_reference_with_a_typed_receipt(tmp_path):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    # Finite inputs and weights whose float32 product overflows: calibration observes finite
    # values, and only the floating reference is infinite.
    loader.write_text("""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[3e38, 3e38], [-.732, .287]]))
    def forward(self, x):
        return self.projection(x)
def get_model_and_inputs():
    return Model().eval(), (torch.tensor([[.75, 1.25], [1.25, .5]]),)
""")
    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    recipe = dict(
        schema="quant_recipe_v1",
        target="synthetic",
        unit="matrix",
        status="derived",
        weight={**tensor, "quant_min": -127, "quant_max": 127},
        activation={**tensor, "quant_min": -128, "quant_max": 127},
        families=["contraction"],
        bias_domain="accumulator",
        software_numerical_engine="integer_reference",
    )
    recipe["recipe_sha256"] = QR.digest(recipe)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    out = tmp_path / "capture"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(worker.__file__).parents[2]), os.environ.get("PYTHONPATH", "")]),
    }
    completed = subprocess.run(
        [
            python,
            worker.__file__,
            "--m2m-dir",
            m2m,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 4, completed.stderr[-4000:]
    refusal = json.loads((out / "refusal.json").read_text())
    assert refusal["code"] == "nonfinite_reference"
    assert refusal["sample"] == 0 and refusal["output_index"] == 0 and refusal["nonfinite_count"] > 0
    assert not (out / "meta.json").exists()
