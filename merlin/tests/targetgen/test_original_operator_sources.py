"""Actual public schema defaults construct typed ordinary conv2d FX/MLIR."""

import copy
import json
import os
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.frontend_original_call import call_contracts, default_value
from merlin.targetgen.frontend_typed_add import defaults_request
from merlin.targetgen.original_operator_sources import conv2d_forms, conv2d_source, policy_compatibility
from merlin.targetgen.torch_schema_defaults_observer import _literal

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
INTEGER_POLICY = {
    "model": {"engine": "integer_reference"},
    "operand_dtype": "int8",
    "accumulator_dtype": "i32",
    "readout_dtype": "i32",
    "subnormal_operand_flush": False,
    "overflow": "bounded_exact",
}
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def test_versioned_scalar_lists_keep_exact_kinds_and_signed_zero():
    raw = [1, True, -0.0, None, "x"]
    assert _literal(raw) == {"kind": "unsupported"}
    assert default_value(_literal(raw, scalar_lists=True)) == raw
    assert _literal(raw, scalar_lists=True)["items"][2] == {"kind": "float", "value_hex": "-0x0.0p+0"}
    for unsupported in ([[1]], [float("nan")], [float("inf")], (1, 2)):
        assert _literal(unsupported, scalar_lists=True) == {"kind": "unsupported"}
    with pytest.raises(ValueError):
        default_value({"kind": "int", "value": True})
    with pytest.raises(ValueError):
        default_value({"kind": "float", "value_hex": "0x1p+1000000000"})


@pytest.fixture(scope="module")
def original_calls(tmp_path_factory):
    required = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in required):
        pytest.skip("ordinary source controls require explicit selected framework/capture/public declarations")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in required)
    owner = tmp_path_factory.mktemp("original-conv2d-public-schema")
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    defaults = module_source_path("merlin.targetgen.torch_schema_defaults_observer")
    script = owner / "capture.py"
    script.write_text("""import json,sys,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
schemas=module('selected_schemas',sys.argv[2]);defaults=module('selected_defaults',sys.argv[3])
class Model(torch.nn.Module):
 def __init__(self,parameters):super().__init__();self.options=parameters
 def forward(self,X,W,Bias=None):
  if not self.options:return torch.ops.aten.conv2d.default(X,W)
  return torch.ops.aten.conv2d.default(X,W,Bias,**self.options)
class Shared(torch.nn.Module):
 def forward(self,X):return torch.ops.aten.conv2d.default(X,X)
class Checked(torch.nn.Module):
 def forward(self,X,W):
  torch.ops.aten._assert_tensor_metadata.default(X,dtype=torch.float32)
  return torch.ops.aten.conv2d.default(X,W)
cases=[('default',torch.float32,{},False),
 ('grouped_bias',torch.float32,{'stride':(2,1),'padding':(1,0),'dilation':(2,1),'groups':2},True),
 ('single_axis',torch.float32,{'stride':[2],'padding':[1],'dilation':[2]},False),
 ('f16',torch.float16,{},False),('f64',torch.float64,{},False),('bf16',torch.bfloat16,{},False),
 ('shared',torch.float32,{},False),('checked',torch.float32,{},False)]
out={}
for name,dtype,parameters,bias in cases:
 groups=parameters.get('groups',1)
 inputs=(torch.zeros(2,4,9,11,dtype=dtype),torch.zeros(6,4//groups,3,2,dtype=dtype))
 if bias:inputs+= (torch.zeros(6,dtype=dtype),)
 model=Model(parameters)
 if name=='shared':model,inputs=Shared(),(torch.zeros(2,2,3,2,dtype=dtype),)
 if name=='checked':model=Checked()
 graph=snapshot_exported_program(torch.export.export(model,inputs),stage='original')
 request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
          'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
 observed=schemas.observe(request,declarations=Path(sys.argv[4]).read_bytes())
 default_request={'schema':'merlin.original_schema_defaults_request.v2','graph_sha256':graph['sha256'],
                  'rows':[{'target':r['target'],'schema':r['schema']}
                          for r in observed['rows'] if r['status']=='observed']}
 legacy=dict(default_request,schema='merlin.original_schema_defaults_request.v1')
 out[name]={'trace':{'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}},
            'observation':observed,'defaults':defaults.observe(default_request),'legacy_defaults':defaults.observe(legacy)}
out['runtime']={'torch_version':torch.__version__,'git_version':torch.version.git_version,
                'capture':str(Path(sys.argv[1]).absolute())}
print(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(schemas), str(defaults), str(declarations)],
        directory=owner,
        stage="ordinary_original_conv2d_schema_capture",
        inputs=(script, schemas, defaults, declarations),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, capture, owner


def test_actual_default_lists_are_supported_only_by_v2(original_calls):
    sources, *_ = original_calls
    case = sources["default"]
    contracts = call_contracts(case["trace"], case["observation"], case["defaults"])
    assert contracts[0]["status"] == "bound" and contracts[0]["result_arity"] == 1
    request = defaults_request(case["trace"], case["observation"], version=2)
    assert [row["request"] for row in case["defaults"]["rows"]] == request["rows"]
    row = next(row for row in case["defaults"]["rows"] if row["request"]["target"] == "aten.conv2d.default")
    values = {item["name"]: default_value(item["default"]) for item in row["defaults"] if item["has_default"]}
    assert values == {"bias": None, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1}
    previous = next(row for row in case["legacy_defaults"]["rows"] if row["request"]["target"] == "aten.conv2d.default")
    assert all(
        item["default"] == {"kind": "unsupported"}
        for item in previous["defaults"]
        if item["name"] in {"stride", "padding", "dilation"}
    )


def test_source_construction_does_not_invent_a_compatible_numerical_policy(original_calls):
    sources, *_ = original_calls
    case = sources["default"]
    (form,) = conv2d_forms(case["trace"], case["observation"], case["defaults"], numerical_semantics=INTEGER_POLICY)
    assert form["status"] == "supported" and form["source_numerical_semantics"] == INTEGER_POLICY
    assert form["operand_dtypes"] == ["float32", "float32"] and form["result_dtypes"] == ["float32"]
    source = conv2d_source(form, extent=2, max_tensor_elements=10000)
    assert source.metadata()["source_numerical_semantics"] == INTEGER_POLICY
    assert policy_compatibility(form, INTEGER_POLICY)["status"] == "unknown"
    compatible = policy_compatibility(form, FLOAT_POLICY)
    assert compatible["status"] == "dtype_compatible" and compatible["numerical_semantics"] == FLOAT_POLICY
    assert "comparison" in compatible["scope"]
    assert policy_compatibility(form, None)["status"] == "unknown"
    changed = dict(FLOAT_POLICY, readout_dtype="f64", accumulator_dtype="f64")
    assert policy_compatibility(form, changed)["status"] == "unknown"


@pytest.mark.parametrize(
    "defect", ["ordinal", "bool_ordinal", "required_default", "missing_default", "result_arity", "result_count"]
)
def test_incomplete_call_rosters_cannot_issue_supported_forms(original_calls, defect):
    sources, *_ = original_calls
    case = copy.deepcopy(sources["default"])
    graph = case["trace"]["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["target"] == "aten.conv2d.default")
    row = next(row for row in case["defaults"]["rows"] if row["request"]["target"] == call["target"])
    if defect in {"ordinal", "bool_ordinal"}:
        row["defaults"][0]["ordinal"] = 1 if defect == "ordinal" else False
    elif defect == "required_default":
        row["defaults"][2]["has_default"] = False
    elif defect == "missing_default":
        row["defaults"].pop()
    elif defect == "result_arity":
        call["result_arity"] = 0
    else:
        case["observation"]["rows"][0]["returns"] = []
    if defect == "result_arity":
        from merlin.targetgen.frontend_trace import _digest

        graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
        case["defaults"]["graph_sha256"] = graph["sha256"]
    (form,) = conv2d_forms(case["trace"], case["observation"], case["defaults"])
    assert form["status"] == "unknown"


def test_factory_refuses_argument_substitution_and_budget_before_materialization(original_calls):
    sources, *_ = original_calls
    case = sources["default"]
    (form,) = conv2d_forms(case["trace"], case["observation"], case["defaults"])
    changed = copy.deepcopy(form)
    changed["parameters"]["stride"] = [2, 2]
    with pytest.raises(ValueError, match="original argument bindings"):
        conv2d_source(changed, extent=2, max_tensor_elements=10000)
    with pytest.raises(ValueError, match="before allocation"):
        conv2d_source(form, extent=100000, max_tensor_elements=10000)


def test_shared_operand_identity_and_unobserved_zero_return_bridge_remain_unknown(original_calls):
    sources, *_ = original_calls
    shared = sources["shared"]
    (form,) = conv2d_forms(shared["trace"], shared["observation"], shared["defaults"])
    assert form["status"] == "unknown" and "shared operand identity" in form["reason"]
    checked = sources["checked"]
    calls = call_contracts(checked["trace"], checked["observation"], checked["defaults"])
    assertion = next(call for call in calls if call["target"] == "aten._assert_tensor_metadata.default")
    assert assertion["status"] == "unknown" and "native bridge" in assertion["reason"]
    assert assertion["result_arity"] == 1 and assertion["result_roster"][0]["kind"] == "unknown"


@pytest.mark.parametrize("case", ["default", "grouped_bias", "single_axis", "f16", "f64", "bf16"])
def test_actual_schema_factory_fx_and_standard_mlir_preserve_parameters_types_and_results(
    original_calls, tmp_path, case
):
    sources, python, capture, _ = original_calls
    original = sources[case]
    (form,) = conv2d_forms(
        original["trace"], original["observation"], original["defaults"], numerical_semantics=FLOAT_POLICY
    )
    assert form["status"] == "supported", form
    source = conv2d_source(form, extent=2, max_tensor_elements=10000)
    loader, metadata, script, observed = (
        tmp_path / name for name in ("loader.py", "source.json", "construct.py", "observed.json")
    )
    loader.write_text(source.loader)
    metadata.write_text(json.dumps(source.metadata(), sort_keys=True))
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    defaults = module_source_path("merlin.targetgen.torch_schema_defaults_observer")
    declarations = Path(os.environ["MERLIN_TEST_OPERATOR_DECLARATIONS"]).absolute()
    script.write_text("""import importlib.util,json,sys,math
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,m2m
from m2m.capture.trace import snapshot_exported_program
from xdsl.dialects.func import FuncOp
spec=importlib.util.spec_from_file_location('ordinary_source',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model,examples=module.get_model_and_inputs();expected=json.loads(Path(sys.argv[3]).read_bytes())
inputs=tuple(((torch.arange(t.numel(),dtype=torch.int64).reshape(t.shape) % (11 if i==0 else 7))
              -(5 if i==0 else 3)).to(t.dtype)
             for i,t in enumerate(examples))
actual=model(*inputs)
X,W=inputs[:2];Bias=inputs[2] if len(inputs)==3 else None
p=expected['parameters'];groups=p['groups']
def pair(a):return a*2 if len(a)==1 else a
s,pad,d=pair(p['stride']),pair(p['padding']),pair(p['dilation'])
reference=torch.zeros(expected['outputs'][0]['shape'],dtype=actual.dtype)
for n in range(reference.shape[0]):
 for f in range(reference.shape[1]):
  group=f//(reference.shape[1]//groups)
  for i in range(reference.shape[2]):
   for j in range(reference.shape[3]):
    value=float(Bias[f]) if Bias is not None else 0.
    for c in range(W.shape[1]):
     for u in range(W.shape[2]):
      for v in range(W.shape[3]):
       h=i*s[0]-pad[0]+u*d[0];w=j*s[1]-pad[1]+v*d[1]
       if 0<=h<X.shape[2] and 0<=w<X.shape[3]:
        value+=float(X[n,group*W.shape[1]+c,h,w])*float(W[f,c,u,v])
    reference[n,f,i,j]=value
assert actual.tolist()==reference.tolist()
assert bool((actual!=0).any()) and bool((X<0).any()) and bool((X>0).any())
graph=snapshot_exported_program(torch.export.export(model,examples),stage='original')
def reader(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
schemas=reader('fresh_schemas',sys.argv[6]);defaults=reader('fresh_defaults',sys.argv[7])
request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
         'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
schema_observation=schemas.observe(request,declarations=Path(sys.argv[8]).read_bytes())
default_request={'schema':'merlin.original_schema_defaults_request.v2','graph_sha256':graph['sha256'],
                 'rows':[{'target':r['target'],'schema':r['schema']}
                         for r in schema_observation['rows'] if r['status']=='observed']}
default_observation=defaults.observe(default_request)
converted=m2m.convert(model,examples,backend='fx_importer',level='linalg-on-tensors')
assert converted.ok,converted.diagnostics
converted.module.verify();assert 'func.call' not in converted.mlir_text
functions=[op for op in converted.module.body.block.ops if isinstance(op,FuncOp)]
entry,=functions
def tensor(t):return {'shape':list(t.get_shape()),'dtype':str(t.element_type)}
inputs_abi=[tensor(t) for t in entry.function_type.inputs]
outputs_abi=[tensor(t) for t in entry.function_type.outputs]
dtype={'float32':'f32','float16':'f16','bfloat16':'bf16','float64':'f64'}[expected['outputs'][0]['dtype']]
assert inputs_abi==[{'shape':row['shape'],'dtype':dtype} for row in expected['inputs']]
assert outputs_abi==[{'shape':row['shape'],'dtype':dtype} for row in expected['outputs']]
Path(sys.argv[4]).write_text(json.dumps({'graph':graph,'input_abi':inputs_abi,'output_abi':outputs_abi,
 'schema_observation':schema_observation,'defaults':default_observation,
 'values':actual.to(torch.float64).flatten().tolist(),'scalar_comparison':'exact bounded integral finite values',
 'm2m_source':str(Path(m2m.__file__).absolute())},sort_keys=True))
Path(sys.argv[5]).write_text(converted.mlir_text)
""")
    result = I.run(
        [
            str(python),
            "-I",
            str(script),
            str(capture),
            str(loader),
            str(metadata),
            str(observed),
            str(tmp_path / "source.mlir"),
            str(schemas),
            str(defaults),
            str(declarations),
        ],
        directory=tmp_path,
        stage="ordinary_conv2d_factory_fx_standard_mlir_control",
        inputs=(script, loader, metadata, schemas, defaults, declarations),
        outputs=(observed, tmp_path / "source.mlir"),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    fresh = json.loads(observed.read_bytes())
    graph = fresh["graph"]
    trace = {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}
    (fresh_form,) = conv2d_forms(trace, fresh["schema_observation"], fresh["defaults"])
    assert fresh_form["status"] == "supported" and fresh_form["parameters"] == form["parameters"]
    assert fresh_form["operand_dtypes"] == form["operand_dtypes"]
    assert fresh_form["result_dtypes"] == form["result_dtypes"]
    assert fresh["m2m_source"] == str(capture / "m2m/__init__.py")
    if os.environ.get("MERLIN_TEST_MLIR_OPT"):
        opt = Path(os.environ["MERLIN_TEST_MLIR_OPT"]).absolute()
        checked = I.run(
            [str(opt), str(tmp_path / "source.mlir"), "--verify-each", "-o", str(tmp_path / "verified.mlir")],
            directory=tmp_path / "native-parse",
            stage="ordinary_conv2d_native_standard_mlir_verification",
            inputs=(tmp_path / "source.mlir",),
            outputs=(tmp_path / "verified.mlir",),
            dependencies=(opt,),
            env=ENVIRONMENT,
            capture_output=True,
            timeout=60,
        )
        checked.check_returncode()
