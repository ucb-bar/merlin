"""Real frontend centered operands require the selected provider's exact byte proof."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.kernels.shapes import observe_contractions
from merlin.targetgen.core_aten_device import load_execution_provider
from merlin.targetgen.core_aten_spec_int8 import (
    admission_plan,
    closed_centered_integer_mac,
    device_readout,
    provider_coverage,
)
from merlin.targetgen.host_capabilities import validate_host_capabilities

pytestmark = pytest.mark.target("gemmini")


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("real pinned frontend required")
    path = tmp_path_factory.mktemp("centered-byte") / "sources.json"
    script = r"""
import json,sys,torch,m2m
import torch.ao.quantization.fx._decomposed
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
sources=[]
for unsigned,channel in [(False,False),(True,False),(False,True),(True,True)]:
 dtype=torch.uint8 if unsigned else torch.int8
 low,high=(0,255) if unsigned else (-128,127)
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__()
   self.register_buffer('scales',torch.tensor([.25,.5]))
   self.register_buffer('zeros',torch.tensor([low,high],dtype=torch.int64))
  def forward(self,x,w):
   qd=torch.ops.quantized_decomposed
   a=qd.dequantize_per_tensor.default(x,.5,high,low,high,dtype)
   b=(qd.dequantize_per_channel.default(w,self.scales,self.zeros,1,low,high,dtype) if channel
      else qd.dequantize_per_tensor.default(w,.25,low,low,high,dtype))
   y=torch.mm(a,b)
   return qd.quantize_per_tensor.default(y,.25,129,0,255,torch.uint8)
 x=torch.tensor([[low,high]],dtype=dtype)
 w=torch.tensor([[low,high],[high,low]],dtype=dtype)
 graph,receipt=integerize_pt2e(torch.export.export(Model(),(x,w)).module(),(x,w))
 assert not receipt['refusals'],receipt
 allow_exported_model_train_eval(graph)
 result=m2m.convert(graph,(x,w),backend='fx_importer',fully_standard=True)
 assert result.ok,result
 sources.append(result.mlir_text)
open(sys.argv[1],'w').write(json.dumps(sources))
"""
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def selection():
    base = repo_root() / "examples/gemmini/target"
    return dict(
        spec=fixture_resolved(base, "gemmini"),
        host=validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
    )


@pytest.mark.parametrize("index", range(4))
def test_centered_source_refuses_an_i8_only_provider_and_accepts_exact_planes(sources, selection, index):
    module = parse_mlir_text(sources[index])
    ((op, shape),) = observe_contractions(module)
    assert shape.dtypes == ("i32", "i32", "i32") and closed_centered_integer_mac(op)
    readout = device_readout(op)
    assert readout["storage_dtype"] == "uint8" and readout["quant_max"] == 255
    if index >= 2:
        assert readout["granularity"] == "channel"
    plan = admission_plan(module, selection, captured=True)
    assert plan["sites"][0]["status"] == "refused"
    # A bare exact marker does not authenticate narrowing centered magnitude 255.
    ordinal = list(module.walk()).index(op)
    bare = dict(
        status="proven",
        source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
        kernels=[dict(symbol="k", status="proven", exact_arithmetic=True)],
        bindings=[dict(symbol="k", source_operation_ordinal=ordinal, replaced_ordinals=[ordinal])],
    )
    assert "centered i32" in provider_coverage(module, bare, op)
    provider_path = os.environ.get("MERLIN_CORE_ATEN_EXECUTION_PROVIDER")
    package = os.environ.get("GMLIR")
    if not provider_path or not package:
        pytest.skip("real provider required for emitted byte-plane proof")
    provider = load_execution_provider("gemmini", Path(provider_path), candidate_writable_roots=())
    proof = provider.admit_numeric_schedule(str(module), package=Path(package), numeric_policy=selection["spec"])
    if proof["status"] != "proven":
        # Providers may conservatively decline a layout. Merlin must preserve
        # the clean refusal rather than reinterpreting i32 values as bytes.
        refused = admission_plan(module, selection, captured=True, provider_proof=proof)
        assert refused["sites"][0]["status"] == "refused"
        assert provider_coverage(module, proof, op) is not None
        return
    assert provider_coverage(module, proof, op) is None
    plan = admission_plan(module, selection, captured=True, provider_proof=proof)
    assert plan["sites"][0]["status"] == "admitted", plan["sites"][0]["reasons"]
    for field, value in [
        ("readout", "i8"),
        ("rounding", "fp32"),
        ("operand_bounds", [128, 128]),
        ("source_operand_types", ["i8", "i8"]),
        ("accumulator_max", 2**31),
    ]:
        damaged = copy.deepcopy(proof)
        damaged["kernels"][0]["schedule"]["centered_byte_contraction"][field] = value
        if field == "operand_bounds":
            # This is still a legitimate conservative bound only if source-derived.
            # Its identity is bound by the provider; independently disallow out-of-domain claims.
            damaged["kernels"][0]["schedule"]["centered_byte_contraction"][field] = [384, 384]
        assert provider_coverage(module, damaged, op) is not None
    damaged = copy.deepcopy(proof)
    damaged["source_sha256"] = "stale"
    assert provider_coverage(module, damaged, op) is not None
