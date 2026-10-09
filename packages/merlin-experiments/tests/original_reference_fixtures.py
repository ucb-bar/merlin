"""Real public-schema fixtures; no compiler or hardware qualification is issued."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
from merlin_experiments.phase0 import component_execution_budget as E
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, BasisSource, ComponentSemanticBasis
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from merlin_experiments.phase0.software_intake import REVIEW_SCHEMA, issue_independent_software_intake

from merlin.common import invocation_record as I
from merlin.targetgen.frontend_trace import original_operation_semantics
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, OriginalReferencePolicy
from merlin.targetgen.rtl.source_selection import produce_selection


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)
    from hashlib import sha256

    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def policy(operation, dtype, *, bias=False):
    floating = dtype == "float32"
    return OriginalReferencePolicy(
        operation,
        (dtype,) * (3 if bias else 2),
        (dtype,),
        "float32" if floating else "int32",
        "finite_f32" if floating else "modular_wrap",
        "accumulator_format",
        {
            "aten.matmul.default": "contracting_axis_sequential",
            "aten.add.Tensor": "elementwise",
            "aten.conv2d.default": "input_channel_kernel_row_kernel_column",
        }[operation],
        "per_step",
        "rne" if floating else "exact_integer",
        True,
        False,
        "after_reduction",
        0.0,
        0.0,
        "ignore",
    ).record()


def selection(intake, basis):
    return {
        "schema": P.SCHEMA,
        "operator_schema_intake_sha256": intake.sha256,
        "semantic_basis_sha256": basis.source.sha256,
        "byteorder": "little",
        "policies": [
            policy(op, dtype) for op in ("aten.matmul.default", "aten.add.Tensor") for dtype in ("float32", "int8")
        ]
        + [policy("aten.conv2d.default", "float32", bias=bias) for bias in (False, True)],
        "input_palettes": [{"dtype": "float32", "values": [-1.0, 0.5, 2.0]}, {"dtype": "int8", "values": [-5, 2, 7]}],
        "cohorts": {"functional_guard": [1], "withheld_transfer": [2]},
        "source_budget": {
            "schema": C.BUDGET_SCHEMA,
            "max_sources": 100,
            "max_tensor_elements": 10000,
            "max_scalar_products": 100000,
            "max_source_bytes": 10000,
            "max_total_tensor_elements": 100000,
            "max_total_scalar_products": 1000000,
            "max_total_source_bytes": 1000000,
        },
        "execution_budget": {
            "schema": E.SCHEMA,
            "max_scalar_bits": 512,
            **{"max_" + k: 10000000 for k in E._METRICS},
            **{"max_total_" + k: 100000000 for k in E._METRICS},
        },
        "reference_budget": asdict(OriginalReferenceBudget(10000, 100000, 100000, 30000)),
    }


@pytest.fixture(scope="module")
def live_originals(tmp_path_factory):
    names = (
        "MERLIN_TEST_TORCH_PYTHON",
        "MERLIN_TEST_M2M_ROOT",
        "MERLIN_TEST_OPERATOR_DECLARATIONS",
        "MERLIN_TEST_TORCH_SOURCE_ROOT",
        "MERLIN_TEST_FIRTOOL",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip("original reference intake needs explicit public framework/capture/RTL tools")
    python, capture, declarations, checkout, firtool = (Path(os.environ[name]).absolute() for name in names)
    owner = tmp_path_factory.mktemp("live-original-reference-sources")
    owner.chmod(0o700)
    forbidden = (owner / "absent-private-answer-prefix",)
    source = owner / "unit.fir"
    source.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n  module Unit : @[generators/test_unit/src/Independent.scala 1:1]\n"
        "    input x : UInt<8>\n    output y : UInt<8>\n    connect y, x\n"
    )
    bundle = produce_selection(
        target="test_unit",
        firrtl=source,
        generator="test_unit",
        config="IndependentUnit",
        core_root="Unit",
        firtool=firtool,
        output=owner / "rtl",
    )
    descriptor = owner / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=forbidden,
        output=owner / "hardware",
    )
    script = owner / "capture.py"
    script.write_text("""import json,sys
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
class Model(torch.nn.Module):
 def __init__(self,kind):super().__init__();self.kind=kind
 def forward(self,X,W,B=None):
  if self.kind=='matmul':return torch.ops.aten.matmul.default(X,W)
  if self.kind=='add':return torch.ops.aten.add.Tensor(X,W)
  if self.kind=='nonunit':return torch.ops.aten.add.Tensor(X,W,alpha=2)
  if self.kind=='conv':return torch.ops.aten.conv2d.default(X,W)
  return torch.ops.aten.conv2d.default(X,W,B,[2,1],[1,0],[2,1],2)
out={}
for name,kind,dtype in [('matmul_f32','matmul',torch.float32),('matmul_i8','matmul',torch.int8),
 ('add_f32','add',torch.float32),('add_i8','add',torch.int8),('matmul_f16','matmul',torch.float16),
 ('add_nonunit','nonunit',torch.float32),('conv','conv',torch.float32),('conv_bias','bias',torch.float32)]:
 if kind in ('conv','bias'):
  inputs=(torch.zeros(2,4,9,11),torch.zeros(6,2 if kind=='bias' else 4,3,2))
  if kind=='bias':inputs+=(torch.zeros(6),)
 else:
  inputs=(torch.zeros(5,7,dtype=dtype),
          torch.zeros(7,4,dtype=dtype) if kind=='matmul' else torch.zeros(5,7,dtype=dtype))
 graph=snapshot_exported_program(torch.export.export(Model(kind),inputs),stage='original')
 out[name]={'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}}
print(json.dumps(out,sort_keys=True))
""")
    captured = I.run(
        [str(python), "-I", str(script), str(capture)],
        directory=owner,
        stage="reference_original_capture",
        inputs=(script,),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=90,
    )
    captured.check_returncode()
    members, links = [], []
    for case, trace in json.loads(captured.stdout).items():
        pin = write(owner / (case + ".json"), trace)
        _, operations = original_operation_semantics(trace)
        members.append(
            {
                "id": case,
                "kind": "model2mlir_frontend_trace",
                **pin,
                "schema": trace["schema"],
                "operation_semantics": list(operations),
                "effect_semantics": [],
            }
        )
        links.append(
            {
                "owner": "contraction" if operations[0] != "aten.add.Tensor" else "elementwise",
                "member": case,
                "operations": list(operations),
            }
        )
    roster = owner / "basis.json"
    pin = write(roster, {"schema": SCHEMA, "status": "reviewed", "provenance": PROVENANCE, "members": members})
    basis = ComponentSemanticBasis.load(
        roster.read_bytes(),
        source=BasisSource(str(roster), pin["sha256"], "semantic-basis-roster"),
        parent=owner,
        routing={},
    )
    numerics = {
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "model": {"engine": "integer_reference"},
        "subnormal_operand_flush": False,
        "overflow": "bounded_exact",
    }
    spec = owner / "minimal-software.json"
    source_pin = write(
        spec,
        {
            "schema": "merlin.software_spec.v1",
            "target": "test_unit",
            "status": "reviewed",
            "numerical_semantics": numerics,
            "operations": {
                key: {"families": ["elementwise_map" if key == "elementwise" else key], "hardware": "standalone"}
                for key in ("contraction", "elementwise")
            },
        },
    )
    review = owner / "review.json"
    write(
        review,
        {
            "schema": REVIEW_SCHEMA,
            "target": "test_unit",
            "source": source_pin,
            "semantic_basis": pin,
            "numerical_choices": numerics,
            "operation_basis": links,
        },
    )
    software = issue_independent_software_intake(
        hardware=hardware, source=spec, review=review, forbidden_roots=forbidden, output_root=owner / "software"
    )
    selected = owner / "schema-selection.json"
    git = __import__("subprocess").check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    write(
        selected,
        {
            "schema": O.SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": software.sha256,
            "namespace": "aten",
            "python": str(python),
            "canonical_source": {"checkout": str(checkout), "commit": git, "path": str(declarations)},
        },
    )
    intake = O.issue_independent_operator_schema_intake(
        software=software, selection=selected, forbidden_roots=forbidden, output=owner / "schemas"
    )
    return intake, basis, spec
