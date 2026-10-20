"""Independent minimal transpose sources; no retained model or target inputs."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, BasisSource, ComponentSemanticBasis
from merlin_experiments.phase0.original_call_sources import required_source_cohorts
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from merlin_experiments.phase0.software_intake import REVIEW_SCHEMA, issue_independent_software_intake

from merlin.common import invocation_record as I
from merlin.targetgen.frontend_trace import original_operation_semantics
from merlin.targetgen.original_transpose_reference import OriginalTransposeReferencePolicy
from merlin.targetgen.rtl.source_selection import produce_selection

_spec = importlib.util.spec_from_file_location(
    "transpose_reference_helpers", Path(__file__).with_name("original_reference_fixtures.py")
)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
write = F.write


def selection(intake, basis):
    value = F.selection(intake, basis)
    value.update(
        schema=P.TRANSPOSE_SCHEMA,
        native_observations="batch.v1",
        policies=[
            OriginalTransposeReferencePolicy(
                "aten.transpose.int", (dtype,), (dtype,), "rank_two_axis_swap", True, "exact_element_storage_bits"
            ).record()
            for dtype in ("float32", "int8", "int16", "int32", "int64")
        ],
        input_palettes=[
            {"dtype": "float32", "values": [-7.25, -1.5, -(2.0**-149), -0.0, 0.0, 2.0**-149, 1.5, 7.25]},
            *[{"dtype": dtype, "values": [-7, -1, 0, 1, 7]} for dtype in ("int8", "int16", "int32")],
            {"dtype": "int64", "values": [-(1 << 63), -(1 << 53) - 3, -7, -1, 0, 1, 7, (1 << 53) + 3, (1 << 63) - 1]},
        ],
        cohorts={
            cohort: [extent for name, extent in required_source_cohorts() if name == cohort] for cohort in P.COHORTS
        },
    )
    return value


@pytest.fixture(scope="module")
def live_transpose_originals(tmp_path_factory):
    names = (
        "MERLIN_TEST_TORCH_PYTHON",
        "MERLIN_TEST_M2M_ROOT",
        "MERLIN_TEST_OPERATOR_DECLARATIONS",
        "MERLIN_TEST_TORCH_SOURCE_ROOT",
        "MERLIN_TEST_FIRTOOL",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip("transpose originals require explicitly selected public source/schema/RTL tools")
    python, capture, declarations, checkout, firtool = (Path(os.environ[name]).absolute() for name in names)
    owner = tmp_path_factory.mktemp("live-transpose-originals")
    owner.chmod(0o700)
    forbidden = (owner / "absent-private-answer-prefix",)
    fir = owner / "unit.fir"
    fir.write_text(
        "FIRRTL version 2.0.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/Independent.scala 1:1]\n"
        "    input x : UInt<8>\n    output y : UInt<8>\n    y <= x\n"
    )
    bundle = produce_selection(
        target="test_unit",
        firrtl=fir,
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
class Floating(torch.nn.Module):
 def forward(self,X):
  return torch.ops.aten.transpose.int(X,-2,-1),torch.ops.aten.transpose.int(X,-1,1)
class Swap(torch.nn.Module):
 def forward(self,X):
  return torch.ops.aten.transpose.int(X,1,0)
cases=[('floating',Floating(),torch.zeros(11,17)),
       *[(dtype,Swap(),torch.zeros(11,17,dtype=getattr(torch,dtype)))
         for dtype in ('int8','int16','int32','int64')],
       ('unsupported',Swap(),torch.zeros(11,17,dtype=torch.float16))]
out={}
for name,model,X in cases:
 graph=snapshot_exported_program(torch.export.export(model,(X,)),stage='original')
 out[name]={'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}}
print(json.dumps(out,sort_keys=True))
""")
    native = I.run(
        [str(python), "-I", "-B", str(script), str(capture)],
        directory=owner,
        cwd=owner,
        stage="independent_transpose_original_capture",
        inputs=(script,),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=90,
    )
    native.check_returncode()
    members, links = [], []
    for name, trace in json.loads(native.stdout).items():
        pin = write(owner / (name + ".json"), trace)
        _, operations = original_operation_semantics(trace)
        members.append(
            {
                "id": name,
                "kind": "model2mlir_frontend_trace",
                **pin,
                "schema": trace["schema"],
                "operation_semantics": list(operations),
                "effect_semantics": [],
            }
        )
        links.append({"owner": "movement", "member": name, "operations": list(operations)})
    roster = owner / "basis.json"
    basis_pin = write(roster, {"schema": SCHEMA, "status": "reviewed", "provenance": PROVENANCE, "members": members})
    basis = ComponentSemanticBasis.load(
        roster.read_bytes(),
        source=BasisSource(str(roster), basis_pin["sha256"], "semantic-basis-roster"),
        parent=owner,
        routing={},
    )
    numerics = {
        "operand_dtype": "float32",
        "accumulator_dtype": "float32",
        "readout_dtype": "float32",
        "model": {"engine": "specir_fp_reduce"},
        "rounding": "rne",
        "reduction_order": "index_sequential",
        "reduction_cadence": "per_step",
        "product_rounding": "accumulator_format",
        "subnormal_operand_flush": False,
        "input_domain": {"nonfinite": "forbid"},
        "output_domain": {"nonfinite": "forbid"},
    }
    source = owner / "spec.json"
    source_pin = write(
        source,
        {
            "schema": "merlin.software_spec.v1",
            "target": "test_unit",
            "status": "reviewed",
            "numerical_semantics": numerics,
            "operations": {"movement": {"families": ["movement"], "hardware": "standalone"}},
        },
    )
    review = owner / "review.json"
    write(
        review,
        {
            "schema": REVIEW_SCHEMA,
            "target": "test_unit",
            "source": source_pin,
            "semantic_basis": basis_pin,
            "numerical_choices": numerics,
            "operation_basis": links,
        },
    )
    software = issue_independent_software_intake(
        hardware=hardware, source=source, review=review, forbidden_roots=forbidden, output_root=owner / "software"
    )
    from merlin_experiments.phase0.command_intake import _git

    selected = owner / "schemas.json"
    write(
        selected,
        {
            "schema": O.SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": software.sha256,
            "namespace": "aten",
            "python": str(python),
            "canonical_source": {
                "checkout": str(checkout),
                "commit": _git(checkout, "rev-parse", "HEAD"),
                "path": str(declarations),
            },
        },
    )
    schemas = O.issue_independent_operator_schema_intake(
        software=software, selection=selected, forbidden_roots=forbidden, output=owner / "schemas"
    )
    return schemas, basis, source
