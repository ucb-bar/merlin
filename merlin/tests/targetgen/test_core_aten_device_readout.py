# ruff: noqa: E501 -- the captured module is printed one op per line
"""Real frontend metadata with full int32 device tiles and exact host quantization."""

from __future__ import annotations

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.kernels.shapes import accumulator_init, observe_contractions, zero_initialised
from merlin.targetgen.core_aten_spec_int8 import admission_plan, device_readout
from merlin.targetgen.host_capabilities import validate_host_capabilities

pytestmark = pytest.mark.target("gemmini")
_BASE = repo_root() / "examples/gemmini/target"


@pytest.fixture(scope="module")
def source_text(tmp_path_factory, request):
    import os
    import subprocess

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required for real model2mlir readout capture")
    path = tmp_path_factory.mktemp("device-readout") / "source.mlir"
    script = "import sys,torch,m2m\nfrom m2m.capture import integer_readout\nclass Model(torch.nn.Module):\n def forward(self,a,b,bias):\n  acc=torch.ops.m2m_q.int_mm_seeded.default(a,b,bias)\n  return torch.ops.m2m_q.requantize.default(acc,[0.5],0,-128,127,True)\ninputs=(torch.ones(4,8,dtype=torch.int8),torch.ones(8,16,dtype=torch.int8),torch.arange(16,dtype=torch.int32))\nresult=m2m.convert(Model(),inputs,backend='fx_importer')\nassert result.ok,result\nopen(sys.argv[1],'w').write(result.mlir_text)\n"
    coefficient = getattr(request, "param", 0.5)
    if coefficient < 0:
        script = script.replace(
            "torch.ops.m2m_q.requantize.default(acc,[0.5],0,-128,127,True)",
            "torch.ops.m2m_q.exact_readout.default(acc,1.0,[0.5],alpha=-1.0,output_scale=1.0,relu=True)",
        )
    script = script.replace("result=m2m.convert", "Model()(*inputs)\nresult=m2m.convert")
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return path.read_text()


@pytest.fixture(scope="module")
def selection():
    return {
        "spec": fixture_resolved(_BASE, "gemmini"),
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
    }


def _plan(selection, source, proof=None):
    with IR_LOCK:
        module = mq.parse(source)
        ((op, _shape),) = observe_contractions(module)
        return op, admission_plan(module, selection, captured=True, provider_proof=proof)


def test_the_seed_and_the_fused_readout_are_read_structurally(source_text):
    with IR_LOCK:
        ((op, _shape),) = observe_contractions(mq.parse(source_text))
        assert accumulator_init(op) == "int32_seed" and not zero_initialised(op)
        readout = device_readout(op)
    assert readout["relu"] and readout["quant_min"] == -128 and readout["granularity"] == "tensor"
    assert readout["scales"] == []
    assert readout["scale_ratios"] == [dict(numerator="1", denominator="2")]


def test_the_device_readout_plan_replaces_the_host_cast_and_its_k_bound(selection, source_text):
    _op, plan = _plan(selection, source_text)
    (site,) = plan["sites"]
    assert site["readout"] == "int32_host_exact" and site["accumulator_init"] == "int32_seed"
    steps = {(row["step"], row.get("declaration") or row.get("op")) for row in site["steps"]}
    assert (2, "transfer_contracts.accumulator_seed_load") in steps
    assert (4, "transfer_contracts.accumulator_readout") in steps and (5, "exact_rational_host_readout") in steps
    assert not any(step for step, _ in steps if step == 6) and site["cast_exact_range_obligation"] is None
    fused = next(row for row in site["steps"] if row.get("op") == "exact_rational_host_readout")
    assert fused["status"] == "admitted" and fused["placement"] == "host"
    assert fused["device_readout"]["relu"] is True


@pytest.mark.parametrize("source_text", [0.5, -0.5], indirect=True)
def test_seeded_admission_requires_selected_provider_coverage(selection, source_text):
    import hashlib

    with IR_LOCK:
        module = mq.parse(source_text)
        ((op, _shape),) = observe_contractions(module)
        ordinal = list(module.walk()).index(op)
        reader_ordinal = list(module.walk()).index(device_readout(op)["operation"])
        (ratio,) = device_readout(op)["scale_ratios"]
        proof = dict(
            status="proven",
            source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
            kernels=[
                dict(
                    symbol="seeded",
                    status="proven",
                    exact_arithmetic=True,
                    schedule=dict(
                        readout="full_i32",
                        requantization="retained_source_exact_integer",
                        scale_numerator=[ratio["numerator"]],
                        scale_denominator=[ratio["denominator"]],
                    ),
                )
            ],
            bindings=[
                dict(
                    source_operation_ordinal=ordinal,
                    symbol="seeded",
                    replaced_ordinals=[ordinal],
                )
            ],
        )
    _, absent = _plan(selection, source_text)
    assert absent["sites"][0]["status"] == "refused"
    assert "coverage proof unavailable" in " ".join(absent["sites"][0]["reasons"])
    _, admitted = _plan(selection, source_text, proof)
    assert admitted["sites"][0]["status"] == "admitted", admitted["sites"][0]["reasons"]
    for damaged in (
        {**proof, "bindings": []},
        {**proof, "source_sha256": "stale"},
        {**proof, "kernels": []},
        {**proof, "kernels": [dict(symbol="seeded", status="proven", exact_arithmetic=True)]},
        {
            **proof,
            "kernels": [
                {
                    **proof["kernels"][0],
                    "schedule": {
                        **proof["kernels"][0]["schedule"],
                        "readout": "fp32_scaled",
                        "readout_equivalence": {"status": "proven"},
                    },
                }
            ],
        },
        {**proof, "kernels": [dict(symbol="seeded", status="proven")]},
        {**proof, "bindings": [{**proof["bindings"][0], "replaced_ordinals": [ordinal, reader_ordinal]}]},
    ):
        _, refused = _plan(selection, source_text, damaged)
        assert refused["sites"][0]["status"] == "refused"


@pytest.mark.parametrize("key", ["qmin", "qmax", "mystery", "gradient"])
def test_unknown_or_foreign_quantization_attributes_are_rejected(source_text, key):
    """Unknown keys, and schema keys of another quantization kind (gradient), are not readout keys."""
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

    with IR_LOCK:
        ((op, _shape),) = observe_contractions(mq.parse(source_text))
        reader = device_readout(op)["operation"]
        reader.attributes[f"prov.quant.{key}"] = StringAttr("input") if key == "gradient" else IntegerAttr(0, i64)
        with pytest.raises(ValueError, match="unknown readout quantization"):
            device_readout(op)


@pytest.mark.parametrize("source_text", [-0.5], indirect=True)
def test_a_retained_host_relu_can_follow_a_negative_coefficient(selection, source_text):
    _op, plan = _plan(selection, source_text)
    (site,) = plan["sites"]
    assert site["readout"] == "int32_host_exact"
    assert "coverage proof unavailable" in " ".join(site["reasons"])
    assert not any("ReLU" in reason or "nonpositive" in reason for reason in site["reasons"])
    host = next(row for row in site["steps"] if row.get("op") == "exact_rational_host_readout")
    assert host["device_readout"]["relu"] is True
    assert host["device_readout"]["scale_ratios"] == [{"numerator": "-1", "denominator": "2"}]


def test_negative_readout_relu_follows_scaling_with_a_nonzero_zero_point():
    import os
    import subprocess

    from merlin.common.paths import python_import_roots

    script = '''
import torch
from m2m.capture import integer_readout
from merlin.runtime.exact_readout import requantize_int32
acc=torch.tensor([[-7,-3,-1,0,1,3,7]],dtype=torch.int32)
actual=torch.ops.m2m_q.exact_readout.default(acc,1.0,[0.5],alpha=-1.0,output_scale=1.0,
                                          zero_point=3,relu=True)
assert actual.tolist() == [[7,5,3,3,3,3,3]]
host=requantize_int32(acc.numpy(),scale_numerator=-1,scale_denominator=2,
                     zero_point=3,quant_min=3,quant_max=127)
assert actual.tolist() == host.tolist()
'''
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in python_import_roots())},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_readout_numerics_are_refused_before_any_frontend_import():
    from merlin.capture.integerization import integerize_static_pt2e

    with pytest.raises(ValueError, match="numerics"):
        integerize_static_pt2e(None, (), numerics="fixed_point")


def test_exact_ratio_metadata_comes_from_real_model2mlir(tmp_path):
    import json
    import os
    import subprocess
    from fractions import Fraction

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required")
    path = tmp_path / "rational-readout.mlir"
    script = r"""
import sys,torch,m2m
from m2m.capture import integer_readout
class Model(torch.nn.Module):
 def forward(self,a,b,bias):
  acc=torch.ops.m2m_q.int_mm_seeded.default(a,b,bias)
  return torch.ops.m2m_q.exact_readout.default(acc,float(torch.tensor(.03)),[float(torch.tensor(.02))],1.,None,1.,float(torch.tensor(.01)),0,-128,127,False,False)
inputs=(torch.ones(4,8,dtype=torch.int8),torch.ones(8,16,dtype=torch.int8),torch.zeros(16,dtype=torch.int32))
result=m2m.convert(Model(),inputs,backend='fx_importer')
assert result.ok,result
open(sys.argv[1],'w').write(result.mlir_text)
"""
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with IR_LOCK:
        ((op, _),) = observe_contractions(mq.parse(path.read_text()))
        readout = device_readout(op)
        import numpy as np

        expected = (
            Fraction(float(np.float32(0.03))) * Fraction(float(np.float32(0.02))) / Fraction(float(np.float32(0.01)))
        )
        assert readout["scales"] == []
        assert readout["scale_ratios"] == [
            dict(numerator=str(expected.numerator), denominator=str(expected.denominator))
        ]
        json.dumps({key: value for key, value in readout.items() if key != "operation"})
        from xdsl.dialects.builtin import ArrayAttr, StringAttr

        readout["operation"].attributes["prov.quant.scale_denominator"] = ArrayAttr([StringAttr("0")])
        with pytest.raises(ValueError, match="positive denominators"):
            device_readout(op)


@pytest.mark.parametrize("where", ["contraction", "readout"])
def test_real_host_only_metadata_refuses_device_placement(selection, source_text, where):
    from xdsl.dialects.builtin import StringAttr

    with IR_LOCK:
        module = mq.parse(source_text)
        ((op, _),) = observe_contractions(module)
        selected = op if where == "contraction" else device_readout(op)["operation"]
        selected.attributes["prov.quant.host_only"] = StringAttr("true")
        plan = admission_plan(module, selection, captured=True)
    assert plan["sites"][0]["status"] == "refused"
    assert "host-only" in " ".join(plan["sites"][0]["reasons"])


@pytest.fixture(scope="module")
def continuation_sources(tmp_path_factory):
    import json
    import os
    import subprocess

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required for real continuation capture")
    path = tmp_path_factory.mktemp("integer-continuation") / "sources.json"
    script = r"""
import json,sys,torch,m2m
from m2m.capture import integer_readout
class Model(torch.nn.Module):
 def __init__(self,mode):super().__init__();self.mode=mode
 def forward(self,a,b):
  acc=torch.ops.aten._int_mm.default(a,b).reshape(2,2,2)
  if self.mode=='overflow':acc=acc*1000000
  elif self.mode=='float':return acc.float()*.125
  else:acc=acc+acc
  return torch.ops.m2m_q.exact_readout.default(acc,.25,[.5])
inputs=(torch.ones(4,8,dtype=torch.int8),torch.ones(8,2,dtype=torch.int8))
sources={}
for mode in ['exact','overflow','float']:
 result=m2m.convert(Model(mode),inputs,backend='fx_importer')
 assert result.ok,result
 sources[mode]=result.mlir_text
open(sys.argv[1],'w').write(json.dumps(sources))
"""
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(path.read_text())


def test_integer_continuation_proves_all_intermediate_ranges(continuation_sources):
    from merlin.targetgen.integer_continuation import readout_continuation

    with IR_LOCK:
        for mode, source in continuation_sources.items():
            ((op, _),) = observe_contractions(mq.parse(source))
            proof = readout_continuation(op)
            if mode != "exact":
                assert proof is None
                assert device_readout(op) is None
            else:
                reader, proof = proof
                assert mq.attr_str(reader, "prov.quant.arithmetic") == "exact"
                assert proof["status"] == "proven"
                assert any(row["maximum"] == 2 * 8 * 16384 for row in proof["operations"])
                assert device_readout(op)["integer_continuation"] == proof


def test_integer_continuation_requires_raw_width_and_retained_source(selection, continuation_sources):
    import copy
    import hashlib

    with IR_LOCK:
        module = mq.parse(continuation_sources["exact"])
        ((op, _),) = observe_contractions(module)
        ordinal = list(module.walk()).index(op)
        readout = device_readout(op)
        proof = dict(
            status="proven",
            source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
            kernels=[
                dict(
                    symbol="raw",
                    status="proven",
                    full_result_bound=8 * 16384,
                    schedule=dict(output_dtype="i32", scale=1.0, bias=False, relu=False),
                )
            ],
            bindings=[dict(source_operation_ordinal=ordinal, symbol="raw", replaced_ordinals=[ordinal])],
        )
        admitted = admission_plan(module, selection, captured=True, provider_proof=proof)
        assert admitted["sites"][0]["status"] == "admitted", admitted["sites"][0]["reasons"]
        damaged = []
        for key, value in [("output_dtype", "i8"), ("scale", 0.5), ("bias", True), ("relu", True)]:
            candidate = copy.deepcopy(proof)
            candidate["kernels"][0]["schedule"][key] = value
            damaged.append(candidate)
        candidate = copy.deepcopy(proof)
        candidate["bindings"][0]["replaced_ordinals"].extend(
            readout["integer_continuation"]["retained_operation_ordinals"]
        )
        damaged.append(candidate)
        candidate = copy.deepcopy(proof)
        candidate["kernels"][0]["full_result_bound"] = 2**31
        damaged.append(candidate)
        for candidate in [None, *damaged]:
            refused = admission_plan(module, selection, captured=True, provider_proof=candidate)
            assert refused["sites"][0]["status"] == "refused"
