"""Real original public unary schemas reach fresh source-only preparation."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import original_call_sources as O

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen import original_pointwise_sources as P


def _fixtures():
    path = Path(__file__).with_name("test_component_original_sources.py")
    spec = importlib.util.spec_from_file_location("pointwise_original_source_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures()
native_add, automatic, independent, selected = (
    fixtures.native_add,
    fixtures.automatic,
    fixtures.independent,
    fixtures.selected,
)
add_generation, combined = fixtures.add_generation, fixtures.combined
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@pytest.fixture(scope="module")
def pointwise_calls(tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("original unary sources require explicit framework/capture/public declaration selections")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in names)
    owner = tmp_path_factory.mktemp("original-pointwise-public-schema")
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    script = owner / "capture.py"
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
spec=importlib.util.spec_from_file_location('selected_schemas',sys.argv[2])
schemas=importlib.util.module_from_spec(spec);spec.loader.exec_module(schemas)
class Model(torch.nn.Module):
 def __init__(self,kind):super().__init__();self.kind=kind
 def forward(self,X):
  if self.kind=='relu':return torch.ops.aten.relu.default(X)
  if self.kind=='round':return torch.ops.aten.round.default(X)
  return torch.ops.aten.clamp.default(X,-0.0,3.5)
out={}
for kind in ['relu','round','clamp']:
 for suffix,shape in [('a',(7,9)),('b',(11,13))]:
  model=Model(kind);inputs=(torch.zeros(shape,dtype=torch.float32),)
  graph=snapshot_exported_program(torch.export.export(model,inputs),stage='original')
  request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
   'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
  out[kind+'_'+suffix]={'trace':{'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}},
   'observation':schemas.observe(request,declarations=Path(sys.argv[3]).read_bytes())}
out['runtime']={'git_version':torch.version.git_version}
print(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(schemas), str(declarations)],
        directory=owner,
        stage="original_public_pointwise_call_capture",
        inputs=(script, schemas, declarations),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, capture, owner


def test_old_factory_readers_do_not_acquire_pointwise_implementation():
    assert O.reader_modules(1) == O.READER_MODULES == O.reader_modules(2)
    assert O.reader_modules(3) == (*O.READER_MODULES, P.__name__)
    with pytest.raises(ValueError):
        O.reader_modules(True)


def _observe(pointwise_calls, tmp_path, *, version, budget=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    schema, basis = fixtures._selection(tmp_path, pointwise_calls, ["relu_a", "round_a", "clamp_a"])
    return (
        O.observe(
            schema_record=schema,
            basis=basis,
            numerical_semantics=fixtures.INTEGER_POLICY,
            budget=budget or fixtures.budget(),
            destination=tmp_path / "sources",
            version=version,
        ),
        schema,
        basis,
    )


def test_explicit_new_factory_roster_keeps_all_original_calls_and_numerical_unknowns(pointwise_calls, tmp_path):
    legacy, *_ = _observe(pointwise_calls, tmp_path / "legacy", version=2)
    current, schema, basis = _observe(pointwise_calls, tmp_path / "current", version=3)
    assert legacy["schema"] == O.LINEAR_SCHEMA and current["schema"] == O.POINTWISE_SCHEMA
    assert sum(len(row["calls"]) for row in current["members"]) == 3
    assert all(member["status"] == "unknown" for row in legacy["members"] for member in row["source_members"])
    assert (
        sum(member["status"] == "source_constructed" for row in current["members"] for member in row["source_members"])
        == 9
    )
    assert O.verify(current, schema_record=schema, basis=basis, numerical_semantics=fixtures.INTEGER_POLICY) == current
    unknown = O.required_unknowns(
        current, basis=basis, unknown=lambda kind, selector, reason: dict(kind=kind, selector=selector, reason=reason)
    )
    assert [row["kind"] for row in unknown] == ["original_operator_admission"] * 3
    assert all(row["policy_compatibility"][0]["status"] == "unknown" for row in current["members"])


@pytest.mark.parametrize("change", ["factory_version", "last_source", "last_requested_slot"])
def test_changed_original_loader_or_full_roster_cannot_reuse_factory_record(pointwise_calls, tmp_path, change):
    current, schema, basis = _observe(pointwise_calls, tmp_path, version=3)
    altered = copy.deepcopy(current)
    if change == "factory_version":
        altered["schema"] = O.LINEAR_SCHEMA
    elif change == "last_source":
        path = Path(altered["members"][-1]["source_members"][-1]["source"]["path"])
        path.write_bytes(path.read_bytes() + b"\n# changed original source\n")
    else:
        altered["members"][-1]["source_members"].pop()
    with pytest.raises(ValueError):
        O.verify(altered, schema_record=schema, basis=basis, numerical_semantics=fixtures.INTEGER_POLICY)


def test_member_budget_denial_preserves_every_original_guard_private_slot(pointwise_calls, tmp_path):
    limited = fixtures.budget()
    limited["max_sources"] = 8
    current, *_ = _observe(pointwise_calls, tmp_path, version=3, budget=limited)
    members = [member for row in current["members"] for member in row["source_members"]]
    assert len(members) == 9 and all(member["status"] == "unknown" for member in members)
    assert {member["cohort"] for member in members} == {"functional_guard", "withheld_transfer"}
    assert not list(tmp_path.rglob("source-*.py"))


def test_automatic_pointwise_policy_keeps_every_original_admission_and_other_missing_premise(combined, tmp_path):
    assert A._original_source_version({"schema": A.ORIGINAL_POLICY_SCHEMA}) == 1
    assert A._original_source_version({"schema": A.LINEAR_POLICY_SCHEMA}) == 2
    assert A._original_source_version({"schema": A.POINTWISE_POLICY_SCHEMA}) == 3
    policy = yaml.safe_load(combined["component_coverage"].read_bytes())
    policy.update(schema=A.LINEAR_POLICY_SCHEMA, original_source_budget=fixtures.budget())
    previous = fixtures.U.F.run(
        dict(
            combined,
            component_coverage=fixtures.U.F.write(tmp_path / "previous-policy.json", policy),
            output_root=tmp_path / "previous",
        )
    )
    policy["schema"] = A.POINTWISE_POLICY_SCHEMA
    report = fixtures.U.F.run(
        dict(
            combined,
            component_coverage=fixtures.U.F.write(tmp_path / "pointwise-policy.json", policy),
            output_root=tmp_path / "current",
        )
    )
    record = A.verify(report["automatic_derivation"], report=report)
    assert record["schema"] == A.POINTWISE_RECEIPT_SCHEMA
    assert record["original_call_sources"]["schema"] == O.POINTWISE_SCHEMA
    assert report["status"] == "source_prepared_incomplete"
    assert report["declaration"]["obligations"] == previous["declaration"]["obligations"]
    assert record["required_unknowns"] == previous["automatic_derivation"]["required_unknowns"]
    assert all(row["state"] in {"source_generated", "unavailable"} for row in report["obligations"])


@pytest.mark.parametrize("kind", ["relu", "round", "clamp"])
def test_actual_fresh_pointwise_source_preserves_results_and_reaches_standard_mlir(pointwise_calls, tmp_path, kind):
    schema, basis = fixtures._selection(tmp_path, pointwise_calls, [kind + "_a", kind + "_b"])
    record = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=fixtures.INTEGER_POLICY,
        budget=fixtures.budget(),
        destination=tmp_path / "sources",
        version=3,
    )
    first, second = [row["source_members"][1] for row in record["members"]]
    assert first["status"] == second["status"] == "source_constructed"
    loader = Path(first["source"]["path"])
    assert loader.read_bytes() == Path(second["source"]["path"]).read_bytes()
    assert first["metadata"]["inputs"][0]["shape"] == [2, 3]
    script, metadata, mlir, observed = (
        tmp_path / name for name in ("construct.py", "metadata.json", "source.mlir", "observed.json")
    )
    metadata.write_text(json.dumps(first["metadata"], sort_keys=True))
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,m2m
from xdsl.dialects.func import FuncOp
spec=importlib.util.spec_from_file_location('source',sys.argv[2]);source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source);model,examples=source.get_model_and_inputs()
expected=json.loads(Path(sys.argv[3]).read_bytes())
palette=[-3.5,-2.5,-0.5,0.5,2.5,4.5]
values=[palette[i%len(palette)] for i in range(examples[0].numel())]
X=torch.tensor(values,dtype=examples[0].dtype).reshape(examples[0].shape)
actual=model(X)
target=expected['target']
if target=='aten.relu.default':reference=[max(0.0,x) for x in values]
elif target=='aten.round.default':reference=[float(round(x)) for x in values]
else:
 low,high=expected['parameters']['min'],expected['parameters']['max']
 reference=[min(high,max(low,x)) for x in values]
assert actual.flatten().tolist()==reference and actual.dtype==examples[0].dtype
converted=m2m.convert(model,examples,backend='fx_importer',level='linalg-on-tensors')
assert converted.ok,converted.diagnostics
converted.module.verify();assert 'func.call' not in converted.mlir_text
entry,=[op for op in converted.module.body.block.ops if isinstance(op,FuncOp)]
def tensor(t):return {'shape':list(t.get_shape()),'dtype':str(t.element_type)}
assert [tensor(t) for t in entry.function_type.inputs]==[
 {'shape':r['shape'],'dtype':'f32'} for r in expected['inputs']]
assert [tensor(t) for t in entry.function_type.outputs]==[
 {'shape':r['shape'],'dtype':'f32'} for r in expected['outputs']]
Path(sys.argv[4]).write_text(str(converted.module))
Path(sys.argv[5]).write_text(json.dumps({'input':values,'result':actual.tolist(),
 'm2m_source':str(Path(m2m.__file__).absolute())},sort_keys=True))
""")
    _, python, capture, _ = pointwise_calls
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(loader), str(metadata), str(mlir), str(observed)],
        directory=tmp_path / "fx-control",
        stage="ordinary_original_pointwise_fx_standard_mlir",
        inputs=(script, loader, metadata),
        outputs=(mlir, observed),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    assert json.loads(observed.read_bytes())["m2m_source"] == str(capture / "m2m/__init__.py")
    if not os.environ.get("MERLIN_TEST_MLIR_OPT"):
        pytest.skip("native standard MLIR parsing requires an explicit selected tool")
    verified = tmp_path / "verified.mlir"
    result = I.run(
        [os.environ["MERLIN_TEST_MLIR_OPT"], str(mlir), "--verify-each", "-o", str(verified)],
        directory=tmp_path / "native-control",
        stage="ordinary_original_pointwise_native_standard_mlir",
        inputs=(mlir,),
        outputs=(verified,),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
