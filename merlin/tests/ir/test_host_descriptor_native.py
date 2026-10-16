"""Finite ordinary host descriptor transport, never a body/runtime authority.

The original public MLIRs are independently defined identity programs. All
values, repeated result slots and guard bytes are retained; no framework,
target helper, reference backend or captured workload supplies the compiler.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import host_descriptor_compile as C

WORKER = r"""
import ctypes, hashlib, importlib.util, json, sys
from dataclasses import replace
from pathlib import Path
request, owner = map(Path, sys.argv[1:]); data=json.loads(request.read_text())
package=Path(data['package']);spec=importlib.util.spec_from_file_location('merlin',package/'__init__.py',
    submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec);sys.modules['merlin']=module;spec.loader.exec_module(module)
module.__path__=[str(package)]
from merlin.common import invocation_record as I
from merlin.llvmlower import kernel_backend
from merlin.llvmlower.descriptor_contract import DescriptorLimits, OriginalDescriptorSource, TensorStorageDeclaration
from merlin.llvmlower.descriptor_wrapper import parse
from merlin.llvmlower.host_descriptor_compile import HostDescriptorSelection
from merlin.llvmlower.host_descriptor_call import HostDescriptorBuffer
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
limits=DescriptorLimits(2000000,2000000,64,128,8,100000,2000000)
def invoke(model, arguments):model(arguments)
rows=[]
for case in data['cases']:
    root=owner/case['name'];root.mkdir();source=root/'original.mlir';source.write_text(case['source'])
    shape=tuple(case['shape']);dtype=case['dtype'];count=1
    for extent in shape:count*=extent
    tensors=(CompileOnlyTensor('x',shape,dtype),)+tuple(
        CompileOnlyTensor('y'+str(i),shape,dtype) for i in range(case['outputs']))
    strides=[];stride=1
    for extent in reversed(shape):strides.append(stride);stride*=extent
    storage=tuple(TensorStorageDeclaration(tensor.name,'input' if i==0 else 'output',0,
        tuple(reversed(strides)),count,1) for i,tensor in enumerate(tensors))
    abi=CompileOnlySourceAbi(tensors[:1],tensors[1:])
    original=OriginalDescriptorSource(source,hashlib.sha256(source.read_bytes()).hexdigest(),
        'forward','_mlir_ciface_forward',abi,'mlir_ranked_memref_ciface_v1',storage,'disjoint')
    selection=HostDescriptorSelection(original,limits,tuple(data['environment'].items()),90)
    build=root/'ordinary';outputs=tuple(build/name for name in
        ('model.mlir','model.ll','model_host.o','mlir_runtime_host.o','model_host.so'))
    with I.observe_call(root,stage='ordinary_selected_descriptor_compile',function=kernel_backend.compile_host,
        arguments={'original':selection.verify(),'scope':'finite transport; no body/hardware/effect authority'},
        inputs=(source,),outputs=outputs,dependencies=(Path(data['test_source']),)) as record:
        model=kernel_backend.compile_host(parse(source.read_bytes(),limits,emitted=False),build,
            descriptor_selection=selection);record.returned()
    transport=model.descriptor_transport;observation=transport.verify()
    assert case['byte_order']==observation['byte_order']==sys.byteorder
    payload=bytes.fromhex(case['input']);assert len(payload)==count*observation['slots'][0]['element_bytes']
    # Exact original caller-owned allocations with complete prefix/suffix guards.
    frames=[];arguments=[]
    for i,tensor in enumerate(tensors):
        raw=b'\xa5'*16+(payload if i==0 else b'\xcc'*len(payload))+b'\x5a'*16
        frame=ctypes.create_string_buffer(raw,len(raw));frames.append(frame)
        arguments.append(HostDescriptorBuffer(tensor,storage[i],ctypes.addressof(frame),ctypes.addressof(frame)+16))
    original_frame=bytes(frames[0]);original_input=root/'input-frame.bin';original_input.write_bytes(original_frame)
    actual=tuple(root/('output-frame-'+str(i)+'.bin') for i in range(case['outputs']))
    with I.observe_call(root,stage='ordinary_selected_descriptor_full_output',function=invoke,
        arguments={'complete_original_abi':abi.record(),'logical_bytes_per_output':len(payload)},
        inputs=(build/'model_host.so',original_input,source),outputs=actual) as record:
        invoke(model,arguments)
        for path,frame in zip(actual,frames[1:],strict=True):path.write_bytes(bytes(frame))
        record.returned()
    assert bytes(frames[0])==original_frame
    assert all(path.read_bytes()==b'\xa5'*16+payload+b'\x5a'*16 for path in actual)
    calls=list((build/'host-descriptor-calls').rglob('invocation.json'));assert len(calls)==1
    call=I.verify(calls[0]);assert call['stage']=='host_descriptor_execution'
    assert len([row for row in call['inputs'] if Path(row['path']).name.startswith('descriptor-')])==len(tensors)
    before=len(calls);refusals=[]
    for defect in ('missing','reordered','dtype','shape','stride','capacity','overlap'):
        changed=list(arguments)
        if defect=='missing':changed.pop()
        elif defect=='reordered':changed[0],changed[-1]=changed[-1],changed[0]
        elif defect=='dtype':changed[-1]=replace(changed[-1],tensor=replace(changed[-1].tensor,dtype='i8'))
        elif defect=='shape':changed[-1]=replace(changed[-1],tensor=replace(changed[-1].tensor,shape=(count+1,)))
        elif defect in ('stride','capacity'):
            if defect=='stride' and not shape:continue
            key='element_strides' if defect=='stride' else 'capacity_elements'
            value=tuple(s+1 for s in storage[-1].element_strides) if defect=='stride' else count-1
            changed[-1]=replace(changed[-1],storage=replace(changed[-1].storage,**{key:value}))
        else:changed[-1]=replace(changed[-1],allocated_address=arguments[0].allocated_address,
            aligned_address=arguments[0].aligned_address)
        try:model(changed)
        except ValueError as error:refusals.append({'defect':defect,'reason':str(error)})
        else:raise AssertionError('changed original caller object executed: '+defect)
        assert all(path.read_bytes()==bytes(frame) for path,frame in zip(actual,frames[1:],strict=True))
    assert len(list((build/'host-descriptor-calls').rglob('invocation.json')))==before
    # Genuine changed/missing same-image inputs refuse; saved observations cannot replace them.
    changes=[]
    for artifact in ('model_host.o','mlir_runtime_host.o','model_host.so'):
        path=build/artifact;saved=path.read_bytes();path.write_bytes(saved+b'changed')
        try:model(arguments)
        except ValueError as error:changes.append({'artifact':artifact,'reason':str(error)})
        else:raise AssertionError('changed build product executed')
        path.write_bytes(saved);transport.verify()
    object_path=build/'model_host.o';saved=object_path.read_bytes();object_path.unlink()
    try:model(arguments)
    except (ValueError,OSError):changes.append({'artifact':'missing_model_object','reason':'original object absent'})
    else:raise AssertionError('missing object executed')
    object_path.write_bytes(saved);transport.verify()
    assert len(list((build/'host-descriptor-calls').rglob('invocation.json')))==before
    for path in root.rglob('invocation.json'):
        record=I.verify(path)
        if record['kind']=='subprocess':I.require_environment(path,environment=data['environment'])
    rows.append({'name':case['name'],'shape':case['shape'],'dtype':dtype,'original_abi':abi.record(),
        'full_outputs':[str(path) for path in actual],'logical_bytes_per_output':len(payload),
        'descriptor_call':str(calls[0]),'observation':observation,'caller_refusals':refusals,
        'artifact_refusals':changes,
        'scope':'finite original full-value transport; body/physical/effects/runtime unproved'})
(owner/'summary.json').write_text(json.dumps({'rows':rows},sort_keys=True)+'\n')
"""


@pytest.fixture(scope="module")
def ordinary_descriptors(tmp_path_factory):
    names = ("MERLIN_COMPILER_PYTHON", "MERLIN_LLVM_LLC", "MERLIN_CLANG", "MERLIN_M2M_DIR")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("explicit ordinary public compiler/source selectors are required")
    owner = tmp_path_factory.mktemp("ordinary-host-descriptors")
    worker, request = owner / "worker.py", owner / "request.json"
    worker.write_text(WORKER)
    wide = [(1 << 53) + 3, -(1 << 53) - 5, 0, -1, (1 << 62) + 7] * 3
    cases = [
        {
            "name": "rank_zero_signed_zero",
            "shape": [],
            "dtype": "f32",
            "outputs": 1,
            "byte_order": sys.byteorder,
            "input": (1 << 31).to_bytes(4, sys.byteorder).hex(),
            "source": "module {func.func @forward(%x: tensor<f32>) -> tensor<f32> {func.return %x : tensor<f32>}}",
        },
        {
            "name": "tail_wide_repeated_results",
            "shape": [3, 5],
            "dtype": "i64",
            "outputs": 2,
            "byte_order": sys.byteorder,
            "input": b"".join(value.to_bytes(8, sys.byteorder, signed=True) for value in wide).hex(),
            "source": "module {func.func @forward(%x: tensor<3x5xi64>) "
            "-> (tensor<3x5xi64>,tensor<3x5xi64>) {func.return %x,%x : tensor<3x5xi64>,tensor<3x5xi64>}}",
        },
    ]
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", **{name: os.environ[name] for name in names}}
    request.write_text(
        json.dumps(
            {
                "package": str(Path(C.__file__).parents[1]),
                "test_source": __file__,
                "environment": environment,
                "cases": cases,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (owner / "effective-environment.json").write_text(json.dumps(environment, sort_keys=True) + "\n")
    result = I.run(
        [environment["MERLIN_COMPILER_PYTHON"], "-I", "-B", str(worker), str(request), str(owner)],
        directory=owner,
        stage="ordinary_host_descriptor_native_controls",
        cwd=owner,
        env=environment,
        inputs=(worker, request, Path(__file__)),
        outputs=(owner / "summary.json",),
        dependencies=tuple(
            Path(C.__file__).with_name(name)
            for name in ("host_descriptor_compile.py", "host_descriptor_call.py", "abi.py", "kernel_backend.py")
        ),
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr.decode()
    for path in owner.rglob("invocation.json"):
        record = I.verify(path)
        if record["kind"] == "subprocess":
            I.require_environment(path, environment=environment)
    return owner, json.loads((owner / "summary.json").read_text())["rows"]


@pytest.mark.parametrize("case", ("rank_zero_signed_zero", "tail_wide_repeated_results"))
def test_ordinary_original_complete_outputs_guards_and_actual_descriptor_call(ordinary_descriptors, case):
    _, rows = ordinary_descriptors
    row = next(row for row in rows if row["name"] == case)
    assert len(row["full_outputs"]) == len(row["original_abi"]["outputs"])
    assert row["observation"]["data_layout"] and row["observation"]["target_triple"]
    assert all(Path(path).stat().st_size == 32 + row["logical_bytes_per_output"] for path in row["full_outputs"])
    assert len(row["artifact_refusals"]) == 4
    assert {item["defect"] for item in row["caller_refusals"]} >= {
        "missing",
        "reordered",
        "dtype",
        "shape",
        "capacity",
        "overlap",
    }
    assert "body/physical/effects/runtime unproved" in row["scope"]
