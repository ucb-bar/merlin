"""Actual original call observations preserve source-only and admission scopes."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import operator_schema_intake as S
from merlin_experiments.phase0 import original_call_sources as O

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path


def _fixture(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


U = _fixture("original_unified_fixtures", Path(__file__).with_name("test_component_automatic_unified.py"))
native_add, automatic, independent, selected = U.native_add, U.automatic, U.independent, U.selected
add_generation, combined = U.add_generation, U.combined
INTEGER_POLICY = {
    "model": {"engine": "integer_reference"},
    "operand_dtype": "int8",
    "accumulator_dtype": "i32",
    "readout_dtype": "i32",
    "subnormal_operand_flush": False,
    "overflow": "bounded_exact",
}
FLOAT_POLICY = {
    "model": {"engine": "specir_fp_reduce"},
    "operand_dtype": "f32",
    "accumulator_dtype": "f32",
    "readout_dtype": "f32",
    "subnormal_operand_flush": False,
    "rounding": "rne",
    "reduction_order": "index_sequential",
    "reduction_cadence": "per_step",
    "product_rounding": "accumulator_format",
}


@pytest.fixture(scope="module")
def original_calls(tmp_path_factory):
    names = (
        "MERLIN_TEST_TORCH_PYTHON",
        "MERLIN_TEST_M2M_ROOT",
        "MERLIN_TEST_OPERATOR_DECLARATIONS",
        "MERLIN_TEST_TORCH_SOURCE_ROOT",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip("original call observations need explicit native framework/capture/public source selections")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in names[:3])
    owner = tmp_path_factory.mktemp("original-source-call-observations")
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    script = owner / "capture.py"
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
spec=importlib.util.spec_from_file_location('native_schemas',sys.argv[2])
schemas=importlib.util.module_from_spec(spec);spec.loader.exec_module(schemas)
class Model(torch.nn.Module):
 def __init__(self,grouped):super().__init__();self.grouped=grouped
 def forward(self,X,W,Bias=None):
  if self.grouped:return torch.ops.aten.conv2d.default(X,W,Bias,[2,1],[1,0],[2,1],2)
  return torch.ops.aten.conv2d.default(X,W)
class Linear(torch.nn.Module):
 def __init__(self,kind):super().__init__();self.kind=kind
 def forward(self,X,W):
  if self.kind=='matmul':return torch.ops.aten.matmul.default(X,W)
  return torch.ops.aten.add.Tensor(X,W,alpha=2 if self.kind=='nonunit' else 1)
out={}
for name,grouped in [('default',False),('grouped_bias',True)]:
 inputs=(torch.zeros(2,4,9,11),torch.zeros(6,2 if grouped else 4,3,2))
 if grouped:inputs+=(torch.zeros(6),)
 graph=snapshot_exported_program(torch.export.export(Model(grouped),inputs),stage='original')
 request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
          'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
 out[name]={'trace':{'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}},
            'observation':schemas.observe(request,declarations=Path(sys.argv[3]).read_bytes())}
for name,kind,dtype in [('matmul_float','matmul',torch.float32),('matmul_i8','matmul',torch.int8),
                        ('add_float','add',torch.float32),('add_i8','add',torch.int8),
                        ('add_broadcast','broadcast',torch.float32),('add_nonunit','nonunit',torch.float32)]:
 inputs=(torch.zeros(5,7,dtype=dtype),torch.zeros(7,4,dtype=dtype) if kind=='matmul'
         else torch.zeros(7 if kind=='broadcast' else (5,7),dtype=dtype))
 graph=snapshot_exported_program(torch.export.export(Linear(kind),inputs),stage='original')
 request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
          'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
 out[name]={'trace':{'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}},
            'observation':schemas.observe(request,declarations=Path(sys.argv[3]).read_bytes())}
out['runtime']={'git_version':torch.version.git_version}
print(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(schemas), str(declarations)],
        directory=owner,
        stage="actual_original_call_source_controls",
        inputs=(script, schemas, declarations),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        timeout=90,
        capture_output=True,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, capture, owner


def budget():
    return {
        "schema": O.BUDGET_SCHEMA,
        "max_sources": 96,
        "max_tensor_elements": 10000,
        "max_scalar_products": 100000,
        "max_source_bytes": 10000,
        "max_total_tensor_elements": 100000,
        "max_total_scalar_products": 1000000,
        "max_total_source_bytes": 100000,
    }


def _selection(tmp_path, original_calls, cases):
    sources, python, _, _ = original_calls
    declaration = Path(os.environ["MERLIN_TEST_OPERATOR_DECLARATIONS"]).absolute()
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": S.SELECTION_SCHEMA,
                "status": "reviewed",
                "namespace": "aten",
                "python": str(python),
                "software_intake_sha256": "0" * 64,
                "canonical_source": {
                    "checkout": os.environ["MERLIN_TEST_TORCH_SOURCE_ROOT"],
                    "commit": sources["runtime"]["git_version"],
                    "path": str(declaration),
                },
            }
        )
    )
    members, graphs, original_members = [], [], []
    for index, case in enumerate(cases):
        graph, observation = (tmp_path / (str(index) + suffix) for suffix in ("-graph.json", "-schema.json"))
        graph.write_text(json.dumps(sources[case]["trace"], sort_keys=True))
        observation.write_text(json.dumps(sources[case]["observation"], sort_keys=True))
        members.append({"graph_path": str(graph), "observation": str(observation)})
        graphs.append(SimpleNamespace(path=str(graph)))
        original_members.append({"id": "original-" + str(index)})
    # This is a raw observation control, not a live intake or phase issuer.
    return (
        {"selection_path": str(selection), "members": members},
        SimpleNamespace(graph_sources=graphs, declaration_json=json.dumps({"members": original_members})),
    )


def test_actual_original_sources_keep_dtypes_parameters_and_all_private_members(original_calls, tmp_path):
    schema, basis = _selection(tmp_path, original_calls, ["default", "grouped_bias"])
    record = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "ordinary-sources",
    )
    O.verify(record, schema_record=schema, basis=basis, numerical_semantics=INTEGER_POLICY)
    sources = [member for row in record["members"] for member in row["source_members"]]
    assert len(sources) == 6 and all(row["status"] == "source_constructed" for row in sources)
    assert {row["extent"] for row in sources} == {1, 2, 3}
    assert sum(row["cohort"] == "withheld_transfer" for row in sources) == 2
    assert all(row["metadata"]["source_numerical_semantics"] == INTEGER_POLICY for row in sources)
    assert all(row["policy_compatibility"][0]["status"] == "unknown" for row in record["members"])
    unknown = O.required_unknowns(record, basis=basis, unknown=A.P._unknown)
    assert len(unknown) == 2 and all(row["kind"] == "original_operator_admission" for row in unknown)
    assert len({row["id"] for row in unknown}) == 2


@pytest.mark.parametrize(
    "defect", ["lost_transfer", "changed_cost", "changed_policy", "changed_loader", "changed_path"]
)
def test_original_source_receipts_cannot_substitute_or_drop_members(original_calls, tmp_path, defect):
    schema, basis = _selection(tmp_path, original_calls, ["default"])
    record = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "ordinary-sources",
    )
    amended = copy.deepcopy(record)
    row = amended["members"][0]
    if defect == "lost_transfer":
        row["source_members"].pop()
    elif defect == "changed_cost":
        row["source_members"][0]["costs"]["tensor_elements"] -= 1
    elif defect == "changed_policy":
        row["forms"][0]["source_numerical_semantics"] = FLOAT_POLICY
    else:
        pin = row["source_members"][0]["source"]
        path = Path(pin["path"])
        if defect == "changed_loader":
            path.write_text(path.read_text() + "\n# changed\n")
        else:
            duplicate = tmp_path / "substitute.py"
            duplicate.write_bytes(path.read_bytes())
            pin["path"] = str(duplicate)
    with pytest.raises(ValueError):
        O.verify(amended, schema_record=schema, basis=basis, numerical_semantics=INTEGER_POLICY)


def test_complete_source_budget_applies_across_original_graph_members(original_calls, tmp_path):
    schema, basis = _selection(tmp_path, original_calls, ["default", "default"])
    limits = budget()
    initial = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=limits,
        destination=tmp_path / "initial",
    )
    first = initial["members"][0]["source_members"]
    limits["max_total_source_bytes"] = sum(member["costs"]["source_bytes"] for member in first)
    denied = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=limits,
        destination=tmp_path / "limited",
    )
    assert all(member["status"] == "source_constructed" for member in denied["members"][0]["source_members"])
    assert len(denied["members"][1]["source_members"]) == 3
    assert all(member["status"] == "unknown" for member in denied["members"][1]["source_members"])
    O.verify(denied, schema_record=schema, basis=basis, numerical_semantics=INTEGER_POLICY)


def test_normal_v8_generation_keeps_all_v7_gaps_and_original_call_admission(combined, tmp_path):
    _, previous = U._run(combined, tmp_path, A.UNIFIED_POLICY_SCHEMA)
    policy = yaml.safe_load(combined["component_coverage"].read_bytes())
    policy.update(schema=A.ORIGINAL_POLICY_SCHEMA, original_source_budget=budget())
    options = dict(
        combined,
        component_coverage=U.F.write(tmp_path / "v8-policy.json", policy),
        output_root=tmp_path / "v8-generation",
    )
    report = U.F.run(options)
    record = A.verify(report["automatic_derivation"], report=report)
    assert record["schema"] == A.ORIGINAL_RECEIPT_SCHEMA
    assert report["status"] == "source_prepared_incomplete"
    wanted = {row["id"]: row for row in previous["automatic_derivation"]["required_unknowns"]}
    actual = {row["id"]: row for row in record["required_unknowns"]}
    assert all(actual[name] == row for name, row in wanted.items())
    calls = [call for row in record["original_call_sources"]["members"] for call in row["calls"]]
    missing = [row for row in actual.values() if row["kind"] == "original_operator_admission"]
    assert len(calls) == len(missing) and {row["selector"]["node"] for row in missing} == {row["node"] for row in calls}
    # Existing admitted capsule cohorts and their complete ordinary outputs are unchanged.
    assert report["declaration"]["obligations"] == previous["declaration"]["obligations"]


def test_linear_source_version_preserves_v1_and_each_original_admission(original_calls, tmp_path):
    schema, basis = _selection(
        tmp_path, original_calls, ["default", "matmul_float", "matmul_i8", "add_float", "add_i8"]
    )
    previous = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "v1",
    )
    current = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "v2",
        version=2,
    )
    assert previous["schema"] == O.SCHEMA and current["schema"] == O.LINEAR_SCHEMA
    assert sum(m["status"] == "source_constructed" for r in previous["members"] for m in r["source_members"]) == 3
    assert all(m["status"] == "source_constructed" for r in current["members"] for m in r["source_members"])
    for before, after in zip(previous["members"], current["members"], strict=True):
        assert before["calls"] == after["calls"]
        assert len(after["source_members"]) == 3
        assert {m["cohort"] for m in after["source_members"]} == {"functional_guard", "withheld_transfer"}
        assert all(m["metadata"]["source_numerical_semantics"] == INTEGER_POLICY for m in after["source_members"])
    missing = O.required_unknowns(current, basis=basis, unknown=A.P._unknown)
    assert len(missing) == 5 and all(r["kind"] == "original_operator_admission" for r in missing)
    old = O.required_unknowns(previous, basis=basis, unknown=A.P._unknown)
    assert {r["id"]: r for r in missing} == {r["id"]: r for r in old if r["kind"] == "original_operator_admission"}


@pytest.mark.parametrize("case", ["add_broadcast", "add_nonunit"])
def test_original_unsupported_add_form_keeps_all_requested_members(original_calls, tmp_path, case):
    schema, basis = _selection(tmp_path, original_calls, [case])
    record = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "sources",
        version=2,
    )
    assert record["members"][0]["calls"][0]["status"] == "bound"
    assert record["members"][0]["forms"][0]["status"] == "unknown"
    assert len(record["members"][0]["source_members"]) == 3
    assert all(m["status"] == "unknown" for m in record["members"][0]["source_members"])
    assert len(O.required_unknowns(record, basis=basis, unknown=A.P._unknown)) == 4


def test_normal_v9_cannot_promote_source_construction_to_checked_coverage(combined, tmp_path):
    policy = yaml.safe_load(combined["component_coverage"].read_bytes())
    policy.update(schema=A.ORIGINAL_POLICY_SCHEMA, original_source_budget=budget())
    previous = U.F.run(
        dict(
            combined,
            component_coverage=U.F.write(tmp_path / "v8-policy.json", policy),
            output_root=tmp_path / "previous",
        )
    )
    policy.update(schema=A.LINEAR_POLICY_SCHEMA, original_source_budget=budget())
    report = U.F.run(
        dict(
            combined,
            component_coverage=U.F.write(tmp_path / "v9-policy.json", policy),
            output_root=tmp_path / "generation",
        )
    )
    record = A.verify(report["automatic_derivation"], report=report)
    assert record["schema"] == A.LINEAR_RECEIPT_SCHEMA
    assert record["original_call_sources"]["schema"] == O.LINEAR_SCHEMA
    assert report["status"] == "source_prepared_incomplete"
    assert all(r["state"] in {"source_generated", "unavailable"} for r in report["obligations"])
    assert report["declaration"]["obligations"] == previous["declaration"]["obligations"]
    wanted = {
        r["id"]: r
        for r in previous["automatic_derivation"]["required_unknowns"]
        if r["kind"] != "original_operator_factory"
    }
    actual = {r["id"]: r for r in record["required_unknowns"]}
    assert all(actual[name] == row for name, row in wanted.items())


@pytest.mark.parametrize("defect", ["downgrade", "drop_transfer", "cost", "policy"])
def test_linear_source_version_rejects_policy_budget_and_roster_substitution(original_calls, tmp_path, defect):
    schema, basis = _selection(tmp_path, original_calls, ["matmul_float", "add_i8"])
    record = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "sources",
        version=2,
    )
    amended = copy.deepcopy(record)
    row = amended["members"][0]
    if defect == "downgrade":
        amended["schema"] = O.SCHEMA
    elif defect == "drop_transfer":
        row["source_members"].pop()
    elif defect == "cost":
        row["source_members"][0]["costs"]["scalar_products"] -= 1
    else:
        row["forms"][0]["source_numerical_semantics"] = FLOAT_POLICY
    with pytest.raises(ValueError):
        O.verify(amended, schema_record=schema, basis=basis, numerical_semantics=INTEGER_POLICY)


def test_linear_aggregate_budget_keeps_denied_original_members(original_calls, tmp_path):
    schema, basis = _selection(tmp_path, original_calls, ["matmul_float", "add_float"])
    limits = budget()
    initial = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=limits,
        destination=tmp_path / "initial",
        version=2,
    )
    limits["max_total_source_bytes"] = sum(m["costs"]["source_bytes"] for m in initial["members"][0]["source_members"])
    denied = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=limits,
        destination=tmp_path / "limited",
        version=2,
    )
    assert all(m["status"] == "source_constructed" for m in denied["members"][0]["source_members"])
    assert len(denied["members"][1]["source_members"]) == 3
    assert all(m["status"] == "unknown" for m in denied["members"][1]["source_members"])
    assert len(O.required_unknowns(denied, basis=basis, unknown=A.P._unknown)) == 5


@pytest.mark.parametrize("case", ["add_float", "add_i8"])
def test_actual_add_source_reaches_clean_fx_standard_mlir(original_calls, tmp_path, case):
    schema, basis = _selection(tmp_path, original_calls, [case])
    record = O.observe(
        schema_record=schema,
        basis=basis,
        numerical_semantics=INTEGER_POLICY,
        budget=budget(),
        destination=tmp_path / "sources",
        version=2,
    )
    row = record["members"][0]["source_members"][1]
    assert row["status"] == "source_constructed"
    loader = Path(row["source"]["path"])
    script, metadata, mlir, observed = (
        tmp_path / name for name in ("construct.py", "metadata.json", "source.mlir", "observed.json")
    )
    metadata.write_text(json.dumps(row["metadata"], sort_keys=True))
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,m2m
from xdsl.dialects.func import FuncOp
spec=importlib.util.spec_from_file_location('source',sys.argv[2]);source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source);model,examples=source.get_model_and_inputs()
expected=json.loads(Path(sys.argv[3]).read_bytes())
palettes=([127,-128,64,-64,1,-1],[1,-2,64,-64,-1,1]) if examples[0].dtype==torch.int8 else (
 [3,-2,1,0,-3,2],[-1,2,-3,0,3,-2])
values=[[p[j%len(p)] for j in range(t.numel())] for p,t in zip(palettes,examples,strict=True)]
inputs=tuple(torch.tensor(v,dtype=t.dtype).reshape(t.shape) for v,t in zip(values,examples,strict=True))
actual=model(*inputs)
exact=[a+b for a,b in zip(*values,strict=True)]
reference=[(x+128)%256-128 for x in exact] if actual.dtype==torch.int8 else exact
assert actual.flatten().tolist()==reference and actual.dtype==examples[0].dtype
if actual.dtype==torch.int8:assert exact!=reference
assert bool((inputs[0]<0).any()) and bool((inputs[0]>0).any()) and bool((actual!=0).any())
converted=m2m.convert(model,examples,backend='fx_importer',level='linalg-on-tensors')
assert converted.ok,converted.diagnostics
converted.module.verify();assert 'func.call' not in converted.mlir_text
entry,=[op for op in converted.module.body.block.ops if isinstance(op,FuncOp)]
dtype={'int8':'i8','float32':'f32'}[expected['outputs'][0]['dtype']]
def tensor(t):return {'shape':list(t.get_shape()),'dtype':str(t.element_type)}
assert [tensor(t) for t in entry.function_type.inputs]==[{'shape':r['shape'],'dtype':dtype} for r in expected['inputs']]
assert [tensor(t) for t in entry.function_type.outputs]==[
 {'shape':r['shape'],'dtype':dtype} for r in expected['outputs']]
Path(sys.argv[4]).write_text(str(converted.module))
Path(sys.argv[5]).write_text(json.dumps({'inputs':values,'source_result':actual.tolist(),'result_dtype':str(actual.dtype),
 'm2m_source':str(Path(m2m.__file__).absolute())},sort_keys=True))
""")
    _, python, capture, _ = original_calls
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(loader), str(metadata), str(mlir), str(observed)],
        directory=tmp_path / "fx-control",
        stage="ordinary_original_add_fx_standard_mlir",
        inputs=(script, loader, metadata),
        outputs=(mlir, observed),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    assert json.loads(observed.read_bytes())["m2m_source"] == str(capture / "m2m/__init__.py")
    if not os.environ.get("MERLIN_TEST_MLIR_OPT"):
        pytest.skip("native standard MLIR parsing requires an explicit selected tool")
    result = I.run(
        [os.environ["MERLIN_TEST_MLIR_OPT"], str(mlir), "--verify-each", "-o", str(tmp_path / "verified.mlir")],
        directory=tmp_path / "native-control",
        stage="ordinary_original_add_native_standard_mlir",
        inputs=(mlir,),
        outputs=(tmp_path / "verified.mlir",),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
