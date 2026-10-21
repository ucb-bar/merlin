"""Exercise the proved numerical-domain check on actual source quantizer invocations.

These are instance checks under the explicitly selected candidate model, not all-input model
proofs. Rejected instances are never executed natively to obtain an expected poison value.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_source_quantize import contract, request_for, samples_for, word

from merlin.common.paths import repo_root

pytestmark = pytest.mark.usefixtures("lean_oracle")


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (1, 3, 5), (0, 3)])
def test_defined_source_samples_satisfy_checked_domain(tmp_path, width, shape):
    _, arguments = samples_for(width, shape)
    proc, results = contract(request_for(width, shape, arguments), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "source-quantize-results.json").read_text())
    assert report["domain_scope"] == "candidate_runtime_instances_only"
    assert report["domain_checks"] == [True] * len(arguments)
    assert all(s["status"] == "ok" and all(None not in v["bits"] for v in s["values"]) for s in results)


@pytest.mark.parametrize("scale", [0.0, -0.0, -1.0, float("inf"), -float("inf"), float("nan")])
def test_invalid_calibration_parameters_fail_the_domain_check(tmp_path, scale):
    arguments = [[[str(word(1))], [str(word(scale))], ["0"]]]
    proc, _ = contract(request_for(64, (1,), arguments), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "source-quantize-results.json").read_text())
    assert report["domain_checks"] == [False]
    assert not (tmp_path / "native-results.json").exists()


@pytest.mark.parametrize(
    "values,scale,accepted",
    [
        ([float("inf"), -float("inf")], 1.0, True),
        ([float("nan"), 1.0], 1.0, False),
        ([0.0, 1.0], 1e-45, False),
        ([-1.0, 1.0], 1e-45, True),
        ([], 1e-45, True),
    ],
)
def test_domain_uses_actual_intermediates_and_every_observable_cell(tmp_path, values, scale, accepted):
    arguments = [[[str(word(value)) for value in values], [str(word(scale))], ["0"]]]
    proc, results = contract(request_for(64, (len(values),), arguments), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "source-quantize-results.json").read_text())
    assert report["domain_checks"] == [accepted]
    assert report["whole_pass_verified"] is False
    assert (None not in results[0]["values"][0]["bits"]) is accepted
    assert not (tmp_path / "native-results.json").exists()


def test_malformed_runtime_values_and_shapes_refuse(lean_oracle):
    proc = subprocess.run(
        [
            str(lean_oracle),
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_source_domain.lean"),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "runtime domain guards passed"
