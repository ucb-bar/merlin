"""Bounded original-form host controls; no target/runtime qualification.

The forms are independent schema-shaped test inputs, not an issued public
semantic owner. Ordinary source factories and upstream lowering produce the IR.
"""

import json
import math
import os
from pathlib import Path

import pytest
from merlin_experiments.phase1 import component_source_applicability as S
from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS
from merlin_experiments.phase2.contracts import sha256_file

from merlin.common import invocation_record as I
from merlin.targetgen import original_pointwise_sources as P
from merlin.targetgen.frontend_original_call import _literal
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy


def _case(name, operation, dtype, rank, extent, values, bounds=(None, None)):
    floating = dtype == "float32"
    policy = OriginalPointwiseReferencePolicy(
        operation,
        (dtype,),
        (dtype,),
        dtype,
        "finite_f32" if floating else "bounded_exact",
        "not_applicable",
        "elementwise",
        "not_applicable",
        "rne" if floating else "exact_integer",
        True,
        False,
        "not_applicable",
        0.0,
        0.0,
        "preserve" if floating else "ignore",
    ).record()
    tensor = {
        "id": "input",
        "kind": "tensor",
        "dtype": dtype,
        "storage_dtype": dtype,
        "rank": rank,
        "layout": "torch.strided",
        "device": "cpu",
    }
    arguments = [
        {"name": "self", "type": "Tensor", "alias": None, "value": {"kind": "ssa", "node_id": "input", "value": tensor}}
    ]
    parameters = {}
    if operation == "aten.clamp.default":
        for key, value in zip(("min", "max"), bounds, strict=True):
            arguments.append({"name": key, "type": "Optional[number]", "alias": None, "value": _literal(value)})
            parameters[key] = value
    form = {
        "form_schema": P.INTEGER_FORM_SCHEMA,
        "status": "supported",
        "target": operation,
        "arguments": arguments,
        "result_arity": 1,
        "schema_returns": [{"type": "Tensor", "alias": None}],
        "result_roster": [{**tensor, "id": "output"}],
        "rank": rank,
        "operand_dtypes": [dtype],
        "result_dtypes": [dtype],
        "parameters": parameters,
        "source_numerical_semantics": policy,
    }
    return {"name": name, "form": form, "extent": extent, "palette": values}


CASES = (
    _case("scalar_relu", "aten.relu.default", "float32", 0, 1, [-0.0]),
    _case("scalar_round", "aten.round.default", "float32", 0, 1, [-0.5]),
    _case("scalar_integer", "aten.relu.default", "int64", 0, 1, [(1 << 53) + 3]),
    _case("scalar_integer_clamp", "aten.clamp.default", "int64", 0, 1, [(1 << 63) - 1], (-(1 << 63), (1 << 53) + 3)),
    _case(
        "round_tail",
        "aten.round.default",
        "float32",
        1,
        129,
        [-2.5, -1.5, -0.5, -(2.0**-149), -0.0, 0.0, 2.0**-149, 0.5, 1.5, 2.5],
    ),
    _case(
        "clamp_rectangle", "aten.clamp.default", "float32", 2, 3, [-2.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5], (-0.0, 1.0)
    ),
    _case("integer_rectangle", "aten.relu.default", "int8", 2, 3, [-128, -7, -1, 0, 1, 7, 127]),
)

WORKER = """import importlib.util,json,math,sys
from pathlib import Path
request_path=Path(sys.argv[1]);destination=Path(sys.argv[2])
request=json.loads(request_path.read_text())
sys.path[:0]=request['import_roots']
import numpy as np
import m2m
from merlin.common import invocation_record as I
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.kernel_backend import compile_host
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget,prepare_original_reference
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy
from merlin.targetgen.original_pointwise_sources import pointwise_source
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T
rows=[]
for case in request['cases']:
    output=destination/case['name'];output.mkdir()
    form=case['form'];extent=case['extent']
    source=pointwise_source(form,extent=extent,max_tensor_elements=10000)
    metadata=source.metadata();shape=tuple(metadata['inputs'][0]['shape'])
    policy_fields={key:value for key,value in form['source_numerical_semantics'].items() if key!='schema'}
    for key in ('operand_dtypes','readout_dtypes'):
        policy_fields[key]=tuple(policy_fields[key])
    policy=OriginalPointwiseReferencePolicy(**policy_fields)
    contract=prepare_original_reference(form,source,extent=extent,policy=policy,
        budget=OriginalReferenceBudget(10000,100000,100000,20000),output_byteorder='little')
    loader_path=output/'loader.py';loader_path.write_text(source.loader)
    (output/'metadata.json').write_text(json.dumps(metadata,sort_keys=True)+'\\n')
    count=math.prod(shape);palette=case['palette']
    original=T.from_values('X',metadata['inputs'][0]['dtype'],shape,
        [palette[index%len(palette)] for index in range(count)],byteorder='little')
    (output/'input.bin').write_bytes(original.data)
    spec=importlib.util.spec_from_file_location(case['name'],loader_path)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    model,examples=loader.get_model_and_inputs()
    with I.observe_call(output,stage='ordinary_original_capture',function=m2m.convert,
            arguments={'backend':'fx_importer','level':'linalg-on-tensors'},
            inputs=(request_path,loader_path),outputs=(output/'original.mlir',)) as record:
        converted=m2m.convert(model,examples,backend='fx_importer',level='linalg-on-tensors',capture_trace=True)
        assert converted.ok and converted.module and converted.path_taken=='fx_importer'
        converted.module.verify();text=str(converted.module)
        (output/'original.mlir').write_text(text)
        record.returned()
    module=parse_mlir_text(text);module.verify()
    build=output/'ordinary'
    with I.observe_call(output,stage='ordinary_original_host_compile',function=compile_host,
            arguments={'output':str(build)},inputs=(output/'original.mlir',),
            outputs=tuple(build/name for name in ('model.ll','model_host.o','model_host.so'))) as record:
        native=compile_host(module,build)
        record.returned()
    a=np.frombuffer(original.data,dtype=original.dtype).copy().reshape(shape)
    y=np.empty(shape,dtype=original.dtype)
    with I.observe_call(output,stage='ordinary_original_host_execute',function=native.__call__,
            arguments={'input_shape':list(shape),'output_shape':list(shape),'dtype':original.dtype},
            inputs=(build/'model_host.so',output/'input.bin'),outputs=(output/'actual.bin',)) as record:
        native([(a.ctypes.data,a.shape),(y.ctypes.data,y.shape)])
        actual=T('Y',original.dtype,shape,y.tobytes(),'little')
        (output/'actual.bin').write_bytes(actual.data)
        record.returned()
    expected=contract.evaluate((original,))[0];(output/'reference.bin').write_bytes(expected.data)
    compared=contract.compare((original,),(actual,))
    values=list(actual.values());values[-1]=values[-1]+1
    changed=T.from_values('Y',original.dtype,shape,values,byteorder='little')
    rejected=contract.compare((original,),(changed,))
    rows.append({'name':case['name'],'shape':list(shape),'dtype':original.dtype,
        'comparison':compared,'changed_final_comparison':rejected})
    (destination/'summary.json').write_text(json.dumps(rows,sort_keys=True)+'\\n')
print(json.dumps(rows,sort_keys=True))
assert all(row['comparison']['passed'] and not row['changed_final_comparison']['passed'] for row in rows)
"""


@pytest.fixture(scope="module")
def ordinary_pointwise(tmp_path_factory):
    names = ("MERLIN_COMPILER_PYTHON", "MERLIN_M2M_DIR", "MERLIN_MLIR_TRANSLATE", "MERLIN_LLVM_LLC")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("ordinary pointwise execution requires explicit compiler Python, m2m, translator and llc")
    # Preserve an explicitly selected venv executable path: resolving its
    # symlink changes Python's prefix and loses the selected dependencies.
    selected = {name: str(Path(os.environ[name]).absolute()) for name in names}
    assert all(Path(path).is_file() for key, path in selected.items() if key != "MERLIN_M2M_DIR")
    assert (Path(selected["MERLIN_M2M_DIR"]) / "m2m/__init__.py").is_file()
    owner = tmp_path_factory.mktemp("ordinary-original-pointwise")
    worker = owner / "worker.py"
    worker.write_text(WORKER)
    request = owner / "request.json"
    request.write_text(
        json.dumps(
            {
                "cases": CASES,
                "import_roots": [str(Path(P.__file__).parents[2]), selected["MERLIN_M2M_DIR"]],
            },
            sort_keys=True,
        )
        + "\n"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TORCHINDUCTOR_CACHE_DIR": str(owner / "framework-cache"),
        **selected,
    }
    environment["PYTHONPATH"] = str(Path(P.__file__).parents[2])
    environment_path = owner / "environment.json"
    environment_path.write_text(json.dumps(environment, sort_keys=True) + "\n")
    result = I.run(
        [selected["MERLIN_COMPILER_PYTHON"], "-I", "-B", str(worker), str(request), str(owner)],
        directory=owner,
        stage="ordinary_original_pointwise_controls",
        cwd=owner,
        env=environment,
        inputs=(worker, request),
        dependencies=(Path(P.__file__),),
        outputs=(owner / "summary.json",),
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr.decode()
    for path in owner.rglob("invocation.json"):
        record = I.verify(path)
        if record["kind"] == "subprocess":
            I.require_environment(path, environment=environment)
    return owner, {row["name"]: row for row in json.loads((owner / "summary.json").read_text())}


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_original_pointwise_ordinary_host_values_and_source_applicability(ordinary_pointwise, case):
    owner, rows = ordinary_pointwise
    row = rows[case["name"]]
    source = owner / case["name"] / "original.mlir"
    observation = S.evaluate_component_source_applicability(
        source=source,
        source_program_sha256=sha256_file(source),
        frontend="mlir",
    )
    observation.verify()
    record = observation.record()
    assert not record["unresolved"]
    assert record["facts"]["static_input_domain"]["status"] == "PASS"
    assert record["runtime_effects"] == dict.fromkeys(REQUIRED_EXECUTION_EFFECTS, "UNKNOWN")
    assert record["numerical_finiteness"] == "UNKNOWN"
    assert row["comparison"]["checked_elements"] == math.prod(row["shape"])
    assert row["comparison"]["passed"] and not row["changed_final_comparison"]["passed"]
    assert row["changed_final_comparison"]["mismatches"][0]["index"] == math.prod(row["shape"]) - 1
    build = owner / case["name"] / "ordinary"
    assert (build / "model_host.o").read_bytes().startswith(b"\x7fELF")
    assert (build / "model_host.so").read_bytes().startswith(b"\x7fELF")
