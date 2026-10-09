"""Original public schema -> fresh typed source -> ordinary native full outputs.

These bounded CPU controls prove only the selected reference cases. They are not
framework-wide, target or experiment qualification and use no captured workload.
"""

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.kernel_backend import compile_host
from merlin.targetgen import original_operator_sources as S
from merlin.targetgen.original_operator_reference import (
    OriginalReferenceBudget,
    OriginalReferencePolicy,
    prepare_original_reference,
)
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
BUDGET = OriginalReferenceBudget(10000, 40000, 100000, 30000)


def _policy(target, dtype, bias):
    integer = dtype == "int8"
    return OriginalReferencePolicy(
        target,
        (dtype,) * (3 if bias else 2),
        (dtype,),
        "int32" if integer else "float32",
        "modular_wrap" if integer else "finite_f32",
        "accumulator_format",
        {
            "aten.matmul.default": "contracting_axis_sequential",
            "aten.add.Tensor": "elementwise",
            "aten.conv2d.default": "input_channel_kernel_row_kernel_column",
        }[target],
        "per_step",
        "exact_integer" if integer else "rne",
        True,
        False,
        "after_reduction",
        0.0,
        0.0,
        "ignore" if integer else "preserve",
    )


@pytest.fixture(scope="module")
def original_operators(tmp_path_factory):
    required = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in required):
        pytest.skip("original schema controls need explicitly selected framework/capture/public declarations")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in required)
    owner = tmp_path_factory.mktemp("original-reference-schema")
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    defaults = module_source_path("merlin.targetgen.torch_schema_defaults_observer")
    script, observed = owner / "capture.py", owner / "observation.json"
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
def reader(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
schemas=reader('selected_schemas',sys.argv[2]);defaults=reader('selected_defaults',sys.argv[3])
class Matmul(torch.nn.Module):
 def forward(self,X,W):return torch.ops.aten.matmul.default(X,W)
class Add(torch.nn.Module):
 def forward(self,X,W):return torch.ops.aten.add.Tensor(X,W,alpha=1)
class Conv(torch.nn.Module):
 def forward(self,X,W,Bias):return torch.ops.aten.conv2d.default(X,W,Bias,[1,2],[0,1],[1,1],2)
out={}
for target,kind in [('matmul','float32'),('matmul','int8'),('add','float32'),('add','int8'),('conv','float32')]:
 dtype=getattr(torch,kind);model={'matmul':Matmul,'add':Add,'conv':Conv}[target]()
 shapes={'matmul':[(2,3),(3,4)],'add':[(2,3),(2,3)],'conv':[(1,2,3,4),(2,1,2,2),(2,)]}[target]
 examples=tuple(torch.zeros(s,dtype=dtype) for s in shapes)
 graph=snapshot_exported_program(torch.export.export(model,examples),stage='original')
 request={'namespace':'aten','captured_schemas':graph['operator_schemas'],
  'operations':sorted({n['target'] for n in graph['nodes'] if n['op']=='call_function'})}
 observation=schemas.observe(request,declarations=Path(sys.argv[4]).read_bytes())
 default_request={'schema':'merlin.original_schema_defaults_request.v2','graph_sha256':graph['sha256'],
  'rows':[{'target':r['target'],'schema':r['schema']} for r in observation['rows'] if r['status']=='observed']}
 out[target+'_'+kind]={'trace':{'schema':'m2m.frontend_trace.v1','graphs':{'original':graph}},
  'observation':observation,'defaults':defaults.observe(default_request)}
out['runtime']={'torch_version':torch.__version__,'git_version':torch.version.git_version}
Path(sys.argv[5]).write_text(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(schemas), str(defaults), str(declarations), str(observed)],
        directory=owner,
        stage="original_reference_public_schema",
        inputs=(script, schemas, defaults, declarations),
        outputs=(observed,),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    return json.loads(observed.read_bytes()), python, capture


def _render_and_evaluate(owner, *, python, capture, source):
    loader, metadata, script = (owner / name for name in ("loader.py", "source.json", "prepare.py"))
    mlir, values = owner / "original.mlir", owner / "original-values.json"
    loader.write_text(source.loader)
    metadata.write_text(source.metadata_json)
    script.write_text("""import importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,m2m
spec=importlib.util.spec_from_file_location('selected_original',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model,examples=module.get_model_and_inputs()
metadata=json.loads(Path(sys.argv[3]).read_bytes())
inputs=[]
for index,example in enumerate(examples):
 palette=([127,-128,64,-64,11,-13] if example.dtype==torch.int8 else [-3.25,1.5,2.75,-1.0,4.25,-2.0])
 values=[palette[(i+index)%len(palette)] for i in range(example.numel())]
 inputs.append(torch.tensor(values,dtype=example.dtype).reshape(example.shape))
actual=model(*inputs)
assert actual.dtype==examples[0].dtype and bool((actual!=0).any())
converted=m2m.convert(model,examples,backend='fx_importer',level='linalg-on-tensors')
assert converted.ok,converted.diagnostics
converted.module.verify()
assert 'func.call' not in converted.mlir_text
Path(sys.argv[4]).write_text(str(converted.module))
def tensor(row,value):
 assert list(value.shape)==row['shape'] and str(value.dtype)=='torch.'+row['dtype']
 return dict(row,data_hex=value.numpy().tobytes().hex(),byteorder=sys.byteorder)
input_rows=[tensor(row,t) for row,t in zip(metadata['inputs'],inputs,strict=True)]
Path(sys.argv[5]).write_text(json.dumps({'inputs':input_rows,
 'outputs':[tensor(metadata['outputs'][0],actual)],'m2m_source':str(Path(m2m.__file__).absolute())},sort_keys=True))
""")
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(loader), str(metadata), str(mlir), str(values)],
        directory=owner,
        stage="original_reference_factory_fx_standard_mlir",
        inputs=(script, loader, metadata),
        outputs=(mlir, values),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=90,
    )
    result.check_returncode()
    rows = json.loads(values.read_bytes())
    assert rows["m2m_source"] == str(capture / "m2m/__init__.py")

    def tensors(key):
        return tuple(
            T(row["name"], row["dtype"], tuple(row["shape"]), bytes.fromhex(row["data_hex"]), row["byteorder"])
            for row in rows[key]
        )

    return mlir, tensors("inputs"), tensors("outputs")


@pytest.mark.parametrize("case", ["matmul_float32", "matmul_int8", "add_float32", "add_int8", "conv_float32"])
def test_original_typed_source_reference_checks_ordinary_linked_host_every_output(
    original_operators, tmp_path, case, monkeypatch
):
    tools = ("MERLIN_COMPILER_PYTHON", "MERLIN_LLVM_LLC", "MERLIN_CLANG", "MERLIN_MLIR_TRANSLATE")
    if any(not os.environ.get(name) for name in tools):
        pytest.skip("ordinary host build requires explicit compiler/LLVM object/translator/linker selections")
    originals, python, capture = original_operators
    kind, dtype = case.split("_")
    target = {"matmul": "aten.matmul.default", "add": "aten.add.Tensor", "conv": "aten.conv2d.default"}[kind]
    selected = _policy(target, dtype, kind == "conv")
    forms = {"matmul": S.matmul_forms, "add": S.original_add_forms, "conv": S.conv2d_forms}
    factories = {"matmul": S.matmul_source, "add": S.add_source, "conv": S.conv2d_source}
    original = originals[case]
    (form,) = forms[kind](
        original["trace"], original["observation"], original["defaults"], numerical_semantics=selected.record()
    )
    assert form["status"] == "supported", form
    source = factories[kind](form, extent=2, max_tensor_elements=BUDGET.max_tensor_elements)
    checked = prepare_original_reference(
        form, source, extent=2, policy=selected, budget=BUDGET, output_byteorder="little"
    )
    (tmp_path / "form.json").write_text(json.dumps(form, sort_keys=True))
    mlir, inputs, torch_outputs = _render_and_evaluate(tmp_path, python=python, capture=capture, source=source)
    torch_comparison = checked.compare(inputs, torch_outputs)
    assert torch_comparison["passed"], torch_comparison
    module = parse_mlir_text(mlir.read_text())
    module.verify()
    entry = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    for types, roster in (
        (entry.function_type.inputs, source.metadata()["inputs"]),
        (entry.function_type.outputs, source.metadata()["outputs"]),
    ):
        assert len(types) == len(roster)
        for actual, expected in zip(types, roster, strict=True):
            assert list(actual.get_shape()) == expected["shape"]
            assert str(actual.element_type) == {"int8": "i8", "float32": "f32"}[expected["dtype"]]
    # Keep exact effective environments of ordinary native object observations;
    # the original observer still owns every subprocess/completion/product.
    environments, observe = {}, I.observe

    @contextlib.contextmanager
    def recorded(*args, **kwargs):
        environment = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        with observe(*args, **kwargs) as invocation:
            environments[str(invocation.path)] = environment
            yield invocation

    monkeypatch.setattr(I, "observe", recorded)
    host = compile_host(module, tmp_path / "ordinary")
    arrays = [
        np.frombuffer(
            t.data, dtype=("<" if t.byteorder == "little" else ">") + {"int8": "i1", "float32": "f4"}[t.dtype]
        )
        .copy()
        .reshape(t.shape)
        for t in inputs
    ]
    outputs = [np.full(t.shape, -99, dtype={"int8": np.int8, "float32": np.float32}[t.dtype]) for t in torch_outputs]
    host([(value.ctypes.data, value.shape) for value in arrays + outputs])
    actual = tuple(
        T(row["name"], row["dtype"], tuple(value.shape), value.tobytes(), "little")
        for row, value in zip(source.metadata()["outputs"], outputs, strict=True)
    )
    comparison = checked.compare(inputs, actual)
    (tmp_path / "complete-comparison.json").write_text(
        json.dumps(
            {
                "torch": torch_comparison,
                "native": comparison,
                "original_output_bytes": [value.data.hex() for value in actual],
                "contract_sha256": checked.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (tmp_path / "native-environments.json").write_text(json.dumps(environments, indent=2, sort_keys=True))
    assert comparison["passed"], comparison
    assert comparison["checked_elements"] == sum(value.size for value in outputs)
    assert outputs[0].dtype == (np.int8 if dtype == "int8" else np.float32)
    for path, environment in environments.items():
        I.require_environment(Path(path), environment=environment)
    records = [I.verify(Path(path)) for path in environments]
    assert any(Path(record["argv"][0]).resolve() == Path(os.environ["MERLIN_LLVM_LLC"]).resolve() for record in records)
    assert (tmp_path / "ordinary/model.ll").is_file() and (tmp_path / "ordinary/model_host.o").is_file()
    wrong = outputs[0].copy()
    wrong.flat[-1] += 1
    changed = (T(actual[0].name, actual[0].dtype, actual[0].shape, wrong.tobytes(), "little"),)
    assert not checked.compare(inputs, changed)["passed"]
