"""Original rank-two matmul keeps actual result dtype through FX and native MLIR."""

import copy
import json
import os
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.frontend_original_call import call_contracts
from merlin.targetgen.frontend_trace import _digest
from merlin.targetgen.original_operator_sources import matmul_forms, matmul_source, policy_compatibility

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
INTEGER_POLICY = {
    "model": {"engine": "integer_reference"},
    "operand_dtype": "int8",
    "accumulator_dtype": "i32",
    "readout_dtype": "i32",
    "subnormal_operand_flush": False,
    "overflow": "bounded_exact",
}


@pytest.fixture(scope="module")
def original_matmul(tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("original matmul needs explicit selected framework/capture/public declarations")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in names)
    owner = tmp_path_factory.mktemp("original-matmul-public-schema")
    script, observation = owner / "capture.py", owner / "observation.json"
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    defaults = module_source_path("merlin.targetgen.torch_schema_defaults_observer")
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
def reader(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
schemas=reader('selected_schemas',sys.argv[2]);defaults=reader('selected_defaults',sys.argv[3])
class Model(torch.nn.Module):
 def __init__(self,keyword=False):super().__init__();self.keyword=keyword
 def forward(self,X,W):
  if self.keyword:return torch.ops.aten.matmul.default(self=X,other=W)
  return torch.ops.aten.matmul.default(X,W)
class Shared(torch.nn.Module):
 def forward(self,X):return torch.ops.aten.matmul.default(X,X)
out={}
cases=[('i8',torch.int8),('f32',torch.float32),('f16',torch.float16),
       ('f64',torch.float64),('bf16',torch.bfloat16),('keyword',torch.float32),
       ('vector',torch.float32),('batch',torch.float32),('shared',torch.float32)]
for name,dtype in cases:
 model=Model(name=='keyword');inputs=(torch.zeros(3,5,dtype=dtype),torch.zeros(5,2,dtype=dtype))
 if name=='vector':inputs=(torch.zeros(5),torch.zeros(5,2))
 if name=='batch':inputs=(torch.zeros(2,3,5),torch.zeros(2,5,2))
 if name=='shared':model,inputs=Shared(),(torch.zeros(3,3),)
 graph=snapshot_exported_program(torch.export.export(model,inputs),stage='original')
 request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
          'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
 observed=schemas.observe(request,declarations=Path(sys.argv[4]).read_bytes())
 default_request={'schema':'merlin.original_schema_defaults_request.v2','graph_sha256':graph['sha256'],
                  'rows':[{'target':r['target'],'schema':r['schema']}
                          for r in observed['rows'] if r['status']=='observed']}
 out[name]={'trace':{'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}},
            'observation':observed,'defaults':defaults.observe(default_request)}
X=torch.tensor([[127,-128,64],[-128,127,-64]],dtype=torch.int8)
W=torch.tensor([[127,-128],[-128,127],[64,-64]],dtype=torch.int8)
actual=Model()(X,W)
out['overflow']={'X':X.tolist(),'W':W.tolist(),'actual':actual.tolist(),'dtype':str(actual.dtype)}
try:Model()(X,W.to(torch.float32))
except RuntimeError as error:out['mixed_dtype']={'status':'refused','reason':str(error)}
out['runtime']={'torch_version':torch.__version__,'git_version':torch.version.git_version}
Path(sys.argv[5]).write_text(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [
            str(python),
            "-I",
            str(script),
            str(capture),
            str(schemas),
            str(defaults),
            str(declarations),
            str(observation),
        ],
        directory=owner,
        stage="ordinary_original_matmul_schema_capture",
        inputs=(script, schemas, defaults, declarations),
        outputs=(observation,),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    return json.loads(observation.read_bytes()), python, capture


def test_actual_int8_result_wrap_and_mixed_dtype_refusal(original_matmul):
    original, *_ = original_matmul
    observed = original["overflow"]
    X, W = observed["X"], observed["W"]
    exact = [[sum(a * b for a, b in zip(row, column, strict=True)) for column in zip(*W, strict=True)] for row in X]
    wrapped = [[(value + 128) % 256 - 128 for value in row] for row in exact]
    assert observed["dtype"] == "torch.int8" and observed["actual"] == wrapped == [[1, 0], [0, 1]]
    assert exact != wrapped and any(abs(value) > 127 for row in exact for value in row)
    assert original["mixed_dtype"]["status"] == "refused"
    assert "same dtype" in original["mixed_dtype"]["reason"]


def test_integer_source_preserves_i8_readout_and_retains_incompatible_original_policy(original_matmul):
    original, *_ = original_matmul
    case = original["i8"]
    (form,) = matmul_forms(case["trace"], case["observation"], case["defaults"], numerical_semantics=INTEGER_POLICY)
    assert form["status"] == "supported" and form["operand_dtypes"] == ["int8", "int8"]
    assert form["result_dtypes"] == ["int8"] and form["source_numerical_semantics"] == INTEGER_POLICY
    source = matmul_source(form, extent=2, max_tensor_elements=1000)
    metadata = source.metadata()
    assert metadata["outputs"] == [{"name": "Y", "kind": "tensor", "dtype": "int8", "shape": [2, 4]}]
    assert metadata["inputs"][0]["shape"] == [2, 3] and metadata["inputs"][1]["shape"] == [3, 4]
    assert metadata["source_numerical_semantics"] == INTEGER_POLICY
    assert metadata["tensor_elements"] == metadata["logical_payload_bytes"] == 26
    assert metadata["scalar_products"] == 24
    assert policy_compatibility(form, INTEGER_POLICY)["status"] == "unknown"


@pytest.mark.parametrize("case", ["vector", "batch", "shared"])
def test_unsupported_rank_and_shared_identity_keep_exact_original_roster(original_matmul, case):
    original, *_ = original_matmul
    captured = original[case]
    (call,) = call_contracts(captured["trace"], captured["observation"], captured["defaults"])
    (form,) = matmul_forms(captured["trace"], captured["observation"], captured["defaults"])
    assert call["status"] == "bound" and form["status"] == "unknown"
    assert form["result_roster"] == call["result_roster"] and form["result_arity"] == 1


@pytest.mark.parametrize("defect", ["missing_argument", "output_dtype", "parameters", "budget"])
def test_matmul_source_refuses_incomplete_or_substituted_original_bindings(original_matmul, defect):
    original, *_ = original_matmul
    case = copy.deepcopy(original["i8"])
    if defect == "missing_argument":
        graph = case["trace"]["graphs"]["original"]
        call = next(node for node in graph["nodes"] if node["target"] == "aten.matmul.default")
        call["args"].pop()
        graph["edges"] = [
            edge
            for edge in graph["edges"]
            if (edge["consumer_node_id"], edge["argument_path"]) != (call["id"], "args/1")
        ]
        graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
        case["defaults"]["graph_sha256"] = graph["sha256"]
        (form,) = matmul_forms(case["trace"], case["observation"], case["defaults"])
        assert form["status"] == "unknown" and "required argument" in form["reason"]
        return
    (form,) = matmul_forms(case["trace"], case["observation"], case["defaults"])
    if defect == "output_dtype":
        form["result_dtypes"] = ["int32"]
    elif defect == "parameters":
        form["parameters"] = {"alpha": 2}
    with pytest.raises(ValueError):
        matmul_source(form, extent=100000 if defect == "budget" else 2, max_tensor_elements=1000)


def _native_controls(owner, *, python, capture, loader, metadata, source):
    names = ("MERLIN_TEST_MLIR_OPT", "MERLIN_TEST_MLIR_TRANSLATE", "MERLIN_TEST_CLANG")
    if any(not os.environ.get(name) for name in names):
        return
    opt, translate, clang = (Path(os.environ[name]).absolute() for name in names)
    owner.mkdir()
    lowered, llvm = owner / "lowered.mlir", owner / "model.ll"
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params,convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    result = I.run(
        [str(opt), str(source), "--pass-pipeline=" + pipeline, "-o", str(lowered)],
        directory=owner,
        stage="ordinary_matmul_native_lowering",
        inputs=(source,),
        outputs=(lowered,),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    result = I.run(
        [str(translate), "--mlir-to-llvmir", str(lowered), "-o", str(llvm)],
        directory=owner,
        stage="ordinary_matmul_native_translation",
        inputs=(lowered,),
        outputs=(llvm,),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    runner_lib = translate.parent.parent / "lib"
    runner = owner / "execute.py"
    runner.write_text("""import ctypes,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
spec=importlib.util.spec_from_file_location('ordinary_source',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model,examples=module.get_model_and_inputs()
class Descriptor(ctypes.Structure):
 _fields_=[('allocated',ctypes.c_void_p),('aligned',ctypes.c_void_p),('offset',ctypes.c_int64),
           ('sizes',ctypes.c_int64*2),('strides',ctypes.c_int64*2)]
def argument(values,shape,ctype):
 array=(ctype*len(values))(*values)
 address=ctypes.addressof(array)
 desc=Descriptor(address,address,0,(ctypes.c_int64*2)(*shape),(ctypes.c_int64*2)(shape[1],1))
 return array,desc
entry=ctypes.CDLL(sys.argv[4])._mlir_ciface_forward
entry.argtypes=[ctypes.c_void_p]*3;entry.restype=None
dtype=examples[0].dtype;integer=dtype==torch.int8
ctype=ctypes.c_int8 if integer else ctypes.c_float
palettes=([127,-128,64,-64,1,-1],[3,-2,1,0,-3,2]) if integer else ([3,-2,1,0,-3,2],[-1,2,-3,0,3,-2])
rows=[]
for palette in palettes:
 values=[[palette[(j+i)%len(palette)] for j in range(t.numel())] for i,t in enumerate(examples)]
 tensors=[torch.tensor(v,dtype=dtype).reshape(t.shape) for v,t in zip(values,examples,strict=True)]
 expected=model(*tensors)
 X,W=tensors
 exact=[[sum(int(X[i,k])*int(W[k,j]) for k in range(X.shape[1]))
         for j in range(W.shape[1])] for i in range(X.shape[0])]
 reference=[[(v+128)%256-128 if integer else v for v in row] for row in exact]
 assert expected.tolist()==reference
 a,da=argument(values[0],list(X.shape),ctype);b,db=argument(values[1],list(W.shape),ctype)
 out,do=argument([99]*expected.numel(),list(expected.shape),ctype)
 entry(ctypes.byref(da),ctypes.byref(db),ctypes.byref(do))
 observed=list(out)
 assert observed==expected.flatten().tolist(),(observed,expected.tolist())
 rows.append({'inputs':values,'exact_integer_sums':exact,'actual':observed,'source_result':expected.tolist(),
              'overflow_observed':integer and exact!=reference})
if integer:assert any(row['overflow_observed'] for row in rows)
Path(sys.argv[5]).write_text(json.dumps({'dtype':str(dtype),'runtime_controls':rows},sort_keys=True))
""")
    for optimization in ("-O0", "-O2"):
        library, observation = (owner / (optimization[1:] + suffix) for suffix in (".so", ".json"))
        result = I.run(
            [
                str(clang),
                optimization,
                "-fPIC",
                "-shared",
                str(llvm),
                "-lm",
                "-L" + str(runner_lib),
                "-lmlir_c_runner_utils",
                "-Wl,-rpath," + str(runner_lib),
                "-o",
                str(library),
            ],
            directory=owner,
            stage="ordinary_matmul_native_link",
            inputs=(llvm,),
            outputs=(library,),
            dependencies=(runner_lib / "libmlir_c_runner_utils.so",),
            env=ENVIRONMENT,
            capture_output=True,
            timeout=60,
        )
        result.check_returncode()
        result = I.run(
            [str(python), "-I", str(runner), str(capture), str(loader), str(metadata), str(library), str(observation)],
            directory=owner,
            stage="ordinary_matmul_native_runtime_inputs",
            inputs=(runner, loader, metadata, library),
            outputs=(observation,),
            env=ENVIRONMENT,
            capture_output=True,
            timeout=60,
        )
        result.check_returncode()


@pytest.mark.parametrize("case", ["i8", "f32", "f16", "f64", "bf16", "keyword"])
def test_actual_original_factory_fx_standard_mlir_complete_abi(original_matmul, tmp_path, case):
    original, python, capture = original_matmul
    selected = original[case]
    (form,) = matmul_forms(selected["trace"], selected["observation"], selected["defaults"])
    assert form["status"] == "supported"
    source = matmul_source(form, extent=2, max_tensor_elements=1000)
    loader, metadata, script, observed = (
        tmp_path / name
        for name in (
            "loader.py",
            "source.json",
            "construct.py",
            "observed.json",
        )
    )
    loader.write_text(source.loader)
    metadata.write_text(source.metadata_json)
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    defaults = module_source_path("merlin.targetgen.torch_schema_defaults_observer")
    declarations = Path(os.environ["MERLIN_TEST_OPERATOR_DECLARATIONS"]).absolute()
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,m2m
from m2m.capture.trace import snapshot_exported_program
from xdsl.dialects.func import FuncOp
from xdsl.dialects.builtin import UnitAttr
def reader(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
module=reader('ordinary_source',sys.argv[2]);model,examples=module.get_model_and_inputs()
expected=json.loads(Path(sys.argv[3]).read_bytes())
inputs=tuple(((torch.arange(t.numel()).reshape(t.shape)%(7 if i==0 else 5))-(3 if i==0 else 2)).to(t.dtype)
             for i,t in enumerate(examples))
actual=model(*inputs)
X,W=inputs
values=[[sum(float(X[i,k])*float(W[k,j]) for k in range(X.shape[1]))
         for j in range(W.shape[1])] for i in range(X.shape[0])]
if actual.dtype==torch.int8:values=[[(int(v)+128)%256-128 for v in row] for row in values]
assert actual.tolist()==values and bool((actual!=0).any())
assert bool((X<0).any()) and bool((X>0).any())
graph=snapshot_exported_program(torch.export.export(model,examples),stage='original')
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
entry,=[op for op in converted.module.body.block.ops if isinstance(op,FuncOp)]
def tensor(t):return {'shape':list(t.get_shape()),'dtype':str(t.element_type)}
inputs_abi=[tensor(t) for t in entry.function_type.inputs]
outputs_abi=[tensor(t) for t in entry.function_type.outputs]
dtype={'int8':'i8','float32':'f32','float16':'f16','bfloat16':'bf16','float64':'f64'}[expected['outputs'][0]['dtype']]
assert inputs_abi==[{'shape':row['shape'],'dtype':dtype} for row in expected['inputs']]
assert outputs_abi==[{'shape':row['shape'],'dtype':dtype} for row in expected['outputs']]
entry.attributes['llvm.emit_c_interface']=UnitAttr()
Path(sys.argv[4]).write_text(json.dumps({'graph':graph,'input_abi':inputs_abi,'output_abi':outputs_abi,
 'schema_observation':schema_observation,'defaults':default_observation,'values':actual.tolist(),
 'm2m_source':str(Path(m2m.__file__).absolute())},sort_keys=True))
Path(sys.argv[5]).write_text(str(converted.module))
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
        stage="ordinary_matmul_factory_fx_standard_mlir_control",
        inputs=(script, loader, metadata, schemas, defaults, declarations),
        outputs=(observed, tmp_path / "source.mlir"),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    fresh = json.loads(observed.read_bytes())
    (fresh_form,) = matmul_forms(
        {"schema": "m2m.frontend_trace.v1", "graphs": {"original": fresh["graph"]}},
        fresh["schema_observation"],
        fresh["defaults"],
    )
    assert fresh_form["status"] == "supported" and fresh_form["operand_dtypes"] == form["operand_dtypes"]
    assert fresh_form["result_dtypes"] == form["result_dtypes"] and fresh["m2m_source"] == str(
        capture / "m2m/__init__.py"
    )
    if os.environ.get("MERLIN_TEST_MLIR_OPT"):
        opt = Path(os.environ["MERLIN_TEST_MLIR_OPT"]).absolute()
        result = I.run(
            [str(opt), str(tmp_path / "source.mlir"), "--verify-each", "-o", str(tmp_path / "verified.mlir")],
            directory=tmp_path / "native-parse",
            stage="ordinary_matmul_native_standard_mlir_verification",
            inputs=(tmp_path / "source.mlir",),
            outputs=(tmp_path / "verified.mlir",),
            dependencies=(opt,),
            env=ENVIRONMENT,
            capture_output=True,
            timeout=60,
        )
        result.check_returncode()
    if case in {"i8", "f32"}:
        _native_controls(
            tmp_path / "native-execution",
            python=python,
            capture=capture,
            loader=loader,
            metadata=metadata,
            source=tmp_path / "source.mlir",
        )
