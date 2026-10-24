"""Pinned cross-repo regression: host success cannot substitute for device execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def fixture():
    return json.loads((repo_root() / "merlin/tests/data/exactness/bounded_final_gate.json").read_text())


def torch_script(script):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required")
    roots = [str(repo_root() / "src"), os.environ.get("MERLIN_M2M_DIR", "")]
    result = subprocess.run(
        [python, "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(root for root in roots if root)},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("alpha,beta", [(0, 0), (-1, 1.5), (0.5, 1.5), (2, 0.75)])
def test_core_aten_addmm_has_one_exact_readout_boundary(alpha, beta):
    torch_script(f"""
import copy, torch
from fractions import Fraction
from merlin.targetgen._recipe_quantizer import apply_recipe
from merlin.targetgen.exact_quantized_reference import run
from merlin.capture.integerization import integerize_static_pt2e
from merlin.targetgen._m2m_capture_worker import _integerized_agreement
from merlin.targetgen.quant_recipe import digest
recipe=dict(schema='quant_recipe_v1',status='derived',target='synthetic',unit='matrix',
 families=['contraction'],bias_domain='accumulator',readout_numerics='integer_readout',
 runtime_weight_contractions=True,
 weight=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127,block=None,mode='static'),
 activation=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,block=None,mode='static'))
recipe['recipe_sha256']=digest(recipe)
class Model(torch.nn.Module):
 def forward(self,b,a,w): return torch.addmm(b,a,w,alpha={alpha},beta={beta})
inputs=(torch.linspace(-.75,1.25,6).reshape(3,2),torch.linspace(-.75,1.25,12).reshape(3,4),torch.linspace(-.75,1.25,8).reshape(4,2))
model=apply_recipe(Model(),recipe,example_inputs=inputs)
assert any(n.target==torch.ops.aten.addmm.default for n in model.graph.nodes),model.code
expected=run(model,inputs).output
actual,receipt=integerize_static_pt2e(copy.deepcopy(model),inputs)
if {alpha} == 0:
 # A zero coefficient has no contraction to offload. The frontend declines
 # this rewrite instead of introducing a prematurely rounded MAC.
 assert receipt['quantized_contractions_integerized']==0,receipt
 assert receipt['refusals'],receipt
else:
 assert receipt['quantized_contractions_integerized']==1,receipt
 observed=actual(*inputs)
 assert _integerized_agreement(expected,observed,atol=0,rtol=0)['status']=='passed'
 # Independently check the native addmm result against its literal source Q/DQ
 # scales and Python Fraction, rounded once to the nearest IEEE binary32.
 from merlin.targetgen.exact_arithmetic import round_binary
 nodes=[n for n in model.graph.nodes if str(n.target)=='quantized_decomposed.quantize_per_tensor.default']
 sa,sw=[n.args[1] for n in nodes]
 qa=torch.clamp(torch.tensor([round(Fraction(float(x))/Fraction(sa))
  for x in inputs[1].flatten()]),-128,127).reshape(3,4)
 qw=torch.clamp(torch.tensor([round(Fraction(float(x))/Fraction(sw))
  for x in inputs[2].flatten()]),-128,127).reshape(4,2)
 acc=qa@qw
 exact=torch.tensor([[round_binary(Fraction(int(acc[i,j]))*Fraction(sa)*Fraction(sw)*Fraction({alpha})
  +Fraction(float(inputs[0][i,j]))*Fraction({beta})) for j in range(2)] for i in range(3)])
 assert torch.equal(expected.view(torch.uint8),exact.view(torch.uint8))
""")


@pytest.mark.slow
def test_pinned_cross_repo_bounded_device_results(tmp_path, record_property):
    if os.environ.get("MERLIN_EXACTNESS_E2E") != "1":
        pytest.skip("set MERLIN_EXACTNESS_E2E=1 with the selected provider, facts and native toolchain")
    data = fixture()
    from merlin.common.yaml import load_yaml

    software = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")
    assert data["gemmini_mlir_revision"] == software["execution_provider_revision"]
    m2m, package = [Path(os.environ[k]) for k in ("MERLIN_M2M_DIR", "MERLIN_EXACTNESS_DEVICE_PACKAGE")]
    for root, key in ((m2m, "model2mlir_revision"), (package, "gemmini_mlir_revision")):
        if (root / ".git").exists():
            assert (
                subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == data[key]
            )
            assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        else:
            # The run10 archives intentionally have no Git directory. Bind the
            # complete exported source bytes instead of trusting a path or label.
            from pinned_exports import export_digest

            pinned = data["source_exports"][key.removesuffix("_revision")]
            assert pinned["revision"] == data[key]
            assert export_digest(root) == pinned["sha256"]
    suffixes = (
        "0515658725071550",
        "49068d51af4bfe8d",
        "55f13762d1d3958c",
        "145ca92c8e5a94e8",
        "0ed335a2d008f119",
        "01ec0552f6607e07",
        "073b763d37ed37c1",
    )
    cases = [c for c in data["selected_cases"] if c["case_id"].split("::")[1] in suffixes]
    assert len(cases) == 7
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"selected_cases": cases}))
    root = repo_root()
    command = [
        sys.executable,
        str(root / "build_tools/scripts/run_bounded_core_aten_suite.py"),
        "--suite",
        str(suite),
        "--output",
        str(tmp_path / "run"),
        "--repo",
        str(root),
        "--m2m-dir",
        str(m2m),
        "--m2m-python",
        os.environ["MERLIN_M2M_PYTHON"],
        "--label",
        "gate",
        "--target",
        "gemmini",
        "--device-package",
        str(package),
        "--execution-provider",
        os.environ["MERLIN_CORE_ATEN_EXECUTION_PROVIDER"],
        "--facts",
        os.environ["MERLIN_RTL_FACTS"],
        "--software-spec",
        str(root / "examples/gemmini/target/software-spec.yaml"),
        "--host-capabilities",
        str(root / "examples/gemmini/platform/host-capabilities-core-aten.yaml"),
        "--numeric-policy",
        "spec_int8",
        "--workers",
        "4",
        "--arena-mb",
        "16",
        "--dram-mb",
        "256",
        "--execution-timeout",
        "600",
        "--limit",
        str(len(cases)),
    ]
    result = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")])},
        text=True,
        capture_output=True,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    responses = json.loads((tmp_path / "run/gate/response.json").read_text())["results"]
    assert set(responses) == {case["case_id"] for case in cases}
    record_property("case_count", len(responses))
    record_property("case_ids", json.dumps(sorted(responses)))
    for case, result in responses.items():
        evidence = result["evidence"]
        assert evidence["lane"] == "device", (case, result)
        verdict = evidence["semantic_verdict"]
        if verdict["numeric_execution_policy"] == "native_host_exact":
            # A pinned frontend can retain native IEEE arithmetic instead of
            # quantizing it. This path must meet the independent native bit gate.
            floor = verdict["host_floor"]
            assert floor["status"] == "pass", (case, floor)
            assert floor["comparison"] == "exact_native_eager_storage_bits", (case, floor)
            assert floor["results"] and all(row["status"] == "pass" for row in floor["results"])
        else:
            reference = verdict["integer_reference"]
            assert reference["status"] == "pass", (case, reference)
            comparisons = {
                "independent_pt2e_exact_rational": "exact_i32_or_f32_bits_signed_zero_distinct",
                "independent_source_quantized_exact_rational": "independent_quantized_exact_rational_bytes",
            }
            assert reference["engine"] in comparisons, (case, reference)
            assert reference["results"] and all(row["status"] == "pass" for row in reference["results"])
            assert all(row["comparison"] == comparisons[reference["engine"]] for row in reference["results"]), (
                case,
                reference,
            )
            if reference["engine"] == "independent_source_quantized_exact_rational":
                from merlin.llvmlower.quantized_native import parameters

                assert verdict["quantized_contract"] == parameters(software), (case, verdict)
        execution = evidence["routing"]["execution_evidence"]
        assert execution["status"] == "measured" and execution["executed_instructions"] > 0


def test_gate_provider_revision_matches_selected_spec():
    from merlin.common.yaml import load_yaml

    selected = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")
    data = fixture()
    assert data["gemmini_mlir_revision"] == selected["execution_provider_revision"]
    assert data["source_exports"]["gemmini_mlir"]["revision"] == data["gemmini_mlir_revision"]
    assert data["source_exports"]["model2mlir"]["revision"] == data["model2mlir_revision"]


def test_export_pin_detects_source_and_link_edits(tmp_path):
    from pinned_exports import export_digest

    (tmp_path / "source.py").write_text("original")
    (tmp_path / "link.py").symlink_to("source.py")
    before = export_digest(tmp_path)
    (tmp_path / "source.py").write_text("changed")
    assert export_digest(tmp_path) != before
    (tmp_path / "source.py").write_text("original")
    (tmp_path / "link.py").unlink()
    (tmp_path / "link.py").symlink_to("other.py")
    assert export_digest(tmp_path) != before
