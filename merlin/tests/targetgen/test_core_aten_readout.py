"""The rank-two readout must exist as a proved captured view chain."""

import os
import subprocess

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64
from xdsl.parser import Parser

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import make_context
from merlin.kernels.shapes import observe_contractions
from merlin.targetgen.core_aten_readout import host_readout_bridge
from merlin.targetgen.core_aten_spec_int8 import admission_plan
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec


@pytest.fixture(scope="module")
def source_text(tmp_path_factory):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("capture interpreter unavailable")
    path = tmp_path_factory.mktemp("batched-readout") / "source.mlir"
    script = """
import sys,torch,m2m
class Readout(torch.nn.Module):
 def forward(self,a,b):
  acc=torch.bmm(a.int(),b.int())
  return ((acc.reshape(15,4).float()*0.0137)*0.0193).reshape(3,5,4)
a=torch.arange(105,dtype=torch.int8).reshape(3,5,7)
b=torch.arange(84,dtype=torch.int8).reshape(3,7,4)
result=m2m.convert(Readout(),(a,b),backend='fx_importer',capture_trace=True)
assert result.ok and result.capture_trace['status']=='complete'
open(sys.argv[1],'w').write(result.mlir_text)
"""
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-5000:]
    return path.read_text()


@pytest.fixture
def module(source_text):
    return Parser(make_context(), source_text).parse_module()


@pytest.fixture
def selection():
    base = repo_root() / "examples/gemmini/target"
    spec, _ = resolve_spec(load_software_spec(base / "software-spec.yaml"), {})
    return dict(spec=spec, host=validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")))


def test_captured_bridge_retains_batch_and_restores_output(module, selection):
    module.verify()
    ((contraction, shape),) = observe_contractions(module)
    assert shape.dtypes == ("i8", "i8", "i32") and shape.parallel == (3, 5, 4)
    (cast, first, second), restored, proof = host_readout_bridge(contraction)
    assert proof["source_shape"] == [3, 5, 4] and proof["flat_shape"] == [15, 4]
    assert cast.inputs[0].type.get_shape() == (15, 4)
    assert first.inputs[0] is cast.results[0] and second.inputs[0] is first.results[0]
    assert restored.type.get_shape() == (3, 5, 4)
    (site,) = admission_plan(module, selection, captured=True)["sites"]
    assert site["status"] == "refused", site
    assert "full-domain exact-arithmetic proof" in " ".join(site["reasons"])
    assert site["host_readout_bridge"]["status"] == "verified_captured_ssa"
    assert site["cast_exact_range_obligation"]["max_abs_bound"] == 7 * 128 * 128
    assert site["cast_exact_range_obligation"]["exact_limit"] == 1 << 24
    assert next(step for step in site["steps"] if step["step"] == 3)["signature"]["rank"] == 3


def test_noncanonical_same_element_count_view_refuses(source_text):
    text = source_text.replace("15x4", "3x20").replace("15, 4", "3, 20")
    module = Parser(make_context(), text).parse_module()
    module.verify()
    contraction, _ = observe_contractions(module)[0]
    with pytest.raises(ValueError, match="exact static collapse/expand"):
        host_readout_bridge(contraction)


def test_reordered_reassociation_refuses(module):
    contraction, _ = observe_contractions(module)[0]
    view = next(iter(contraction.results[0].uses)).operation
    view.properties["reassociation"] = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in [2, 1, 0]])])
    with pytest.raises(ValueError, match="reassociation"):
        host_readout_bridge(contraction)


def test_ambiguous_accumulator_view_refuses(module):
    contraction, _ = observe_contractions(module)[0]
    view = next(iter(contraction.results[0].uses)).operation
    view.parent_block().insert_op_after(view.clone(), view)
    module.verify()
    with pytest.raises(ValueError, match="unique direct"):
        host_readout_bridge(contraction)


def test_failed_captured_bridge_cannot_use_planned_rank(module, selection):
    contraction, _ = observe_contractions(module)[0]
    view = next(iter(contraction.results[0].uses)).operation
    view.parent_block().insert_op_after(view.clone(), view)
    (site,) = admission_plan(module, selection, captured=True)["sites"]
    assert site["status"] == "refused"
    assert site["host_readout_bridge"] is None
    assert next(step for step in site["steps"] if step["step"] == 5)["signature"]["rank"] == 3


def test_f64_scale_cannot_inherit_f32_readout_declaration(tmp_path, selection):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("capture interpreter unavailable")
    path = tmp_path / "f64-scale.mlir"
    script = """
import sys,torch,m2m
class Readout(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('scales',torch.full((4,),0.0193,dtype=torch.float64))
 def forward(self,a,b):
  acc=torch.bmm(a.int(),b.int())
  return ((acc.reshape(15,4).float()*0.0137)*self.scales).float().reshape(3,5,4)
a=torch.arange(105,dtype=torch.int8).reshape(3,5,7)
b=torch.arange(84,dtype=torch.int8).reshape(3,7,4)
result=m2m.convert(Readout(),(a,b),backend='fx_importer',capture_trace=True)
assert result.ok
open(sys.argv[1],'w').write(result.mlir_text)
"""
    process = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr[-5000:]
    module = Parser(make_context(), path.read_text()).parse_module()
    module.verify()
    contraction, _ = observe_contractions(module)[0]
    with pytest.raises(ValueError, match="unique closed"):
        host_readout_bridge(contraction)
    (site,) = admission_plan(module, selection, captured=True)["sites"]
    assert site["status"] == "refused" and site["host_readout_bridge"] is None
