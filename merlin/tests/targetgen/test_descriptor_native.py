"""Fresh ordinary descriptors; finite native storage transport, no body theorem."""

import json
import os
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import descriptor_layout as D

WORKER = r"""
import ctypes, hashlib, importlib.util, json, sys
from pathlib import Path
request, owner = map(Path, sys.argv[1:]); data=json.loads(request.read_text())
package=Path(data['package']);spec=importlib.util.spec_from_file_location('merlin',package/'__init__.py',
    submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec);sys.modules['merlin']=module;spec.loader.exec_module(module)
module.__path__=[str(package)]
import numpy as np
from merlin.common import invocation_record as I
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.descriptor_contract import DescriptorLimits, OriginalDescriptorSource, TensorStorageDeclaration
from merlin.llvmlower.descriptor_wrapper import observe_descriptor_wrapper
from merlin.llvmlower.descriptor_layout import observe_descriptor_layout, pack_descriptor, require_descriptor_bytes
from merlin.llvmlower.lower import lower_model
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
limits=DescriptorLimits(2000000,2000000,64,128,8,100000,2000000)
def invoke_native(native,addresses,actual,array):
    native.fn(*map(ctypes.c_void_p,addresses));actual.write_bytes(array.tobytes())
rows=[];selected_layout=None
for case in data['cases']:
    root=owner/case['name'];root.mkdir();source=root/'original.mlir';source.write_bytes(bytes.fromhex(case['source']))
    shape=tuple(case['shape']);dtype=case['dtype'];count=int(np.prod(shape))
    original_abi=CompileOnlySourceAbi((CompileOnlyTensor('x',shape,dtype),),(CompileOnlyTensor('y',shape,dtype),))
    strides=[];capacity=1
    for extent in reversed(shape):strides.append(capacity);capacity*=extent
    # Explicit test caller storage, not an inferred production default.
    storage=tuple(TensorStorageDeclaration(name,role,0,tuple(reversed(strides)),capacity,16)
        for name,role in (('x','input'),('y','output')))
    build=root/'ordinary'
    outputs=tuple(build/name for name in ('model.mlir','model.ll','model_host.o','mlir_runtime_host.o','model_host.so'))
    with I.observe_call(root,stage='ordinary_descriptor_host_compile',function=lower_model,
            arguments={'retain_llvm_dialect':True,'complete_original_abi':original_abi.record(),
                'selected_index_bits':case.get('index_bits'),'selected_data_layout':selected_layout},
            inputs=(source,),outputs=outputs,dependencies=(Path(data['test_source']),)) as record:
        result=lower_model(source.read_text(),build,targets=('host',),retain_llvm_dialect=True,
            index_bits=case.get('index_bits'),data_layout=selected_layout if case.get('index_bits') else None)
        record.returned()
    receipt=Path(result.stats['llvm_dialect_product']['path']);prepared=build/'model.mlir'
    declaration=OriginalDescriptorSource(prepared,hashlib.sha256(prepared.read_bytes()).hexdigest(),
        'forward','_mlir_ciface_forward',original_abi,'mlir_ranked_memref_ciface_v1',storage,'disjoint')
    wrapper=observe_descriptor_wrapper(source=declaration,limits=limits,llvm_product_receipt=receipt)
    object_record=next(p for p in build.rglob('invocation.json') if json.loads(p.read_text()).get('stage')=='object')
    layout=observe_descriptor_layout(wrapper=wrapper,object_record=object_record,
        environment=data['environment'],output_root=root/'layout')
    observed=layout.record()['observation'];assert len(observed['slots'])==2
    selected_layout=observed['data_layout']
    from dataclasses import replace
    for forged in (replace(layout),replace(wrapper)):
        try:forged.verify()
        except ValueError:pass
        else:raise AssertionError('constructed live observation accepted')
    if dtype=='i64':x=np.array((1<<53)+3,dtype=np.int64).reshape(shape);expected=x.copy()
    elif dtype=='i8':x=(np.arange(count,dtype=np.int16)*13-71).astype(np.int8).reshape(shape);expected=np.maximum(x,0)
    else:
        x=np.resize(np.array([-0.0,-0.5,0.0,0.5,1.5,-1.5,2.5],dtype=np.float32),count).reshape(shape)
        expected=np.rint(x)
    y=np.full(shape,-99,dtype=x.dtype);original=root/'input.bin';actual=root/'actual.bin'
    original.write_bytes(x.tobytes());carriers=[];addresses=[];payloads=[]
    for ordinal,array in enumerate((x,y)):
        row=observed['slots'][ordinal]
        assert row['element_bytes']==array.dtype.itemsize
        payload=pack_descriptor(observation=layout,ordinal=ordinal,
            allocated_address=array.ctypes.data,aligned_address=array.ctypes.data)
        require_descriptor_bytes(payload=payload,observation=layout,ordinal=ordinal,
            allocated_address=array.ctypes.data,aligned_address=array.ctypes.data)
        alignment=max(row['descriptor_alignment'],row['explicit_load_alignment'] or 1)
        carrier=ctypes.create_string_buffer(len(payload)+alignment)
        address=(ctypes.addressof(carrier)+alignment-1)//alignment*alignment
        ctypes.memmove(address,payload,len(payload));carriers.append(carrier);addresses.append(address);payloads.append(payload)
        (root/f'descriptor-{ordinal}.bin').write_bytes(payload)
        for bad in (True,0,1,1<<(8*observed['pointer_bytes'])):
            try:pack_descriptor(observation=layout,ordinal=ordinal,allocated_address=bad,aligned_address=bad)
            except ValueError:pass
            else:raise AssertionError('invalid caller address accepted')
        changed=bytearray(payload);changed[row['fields'][2]['offset_bytes']]^=1
        try:require_descriptor_bytes(payload=bytes(changed),observation=layout,ordinal=ordinal,
            allocated_address=array.ctypes.data,aligned_address=array.ctypes.data)
        except ValueError:pass
        else:raise AssertionError('changed descriptor field accepted')
    native=HostModel.load(str(result.host_so),image_policy=PrivateHostImagePolicy(build))
    native.fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    with I.observe_call(root,stage='ordinary_selected_descriptor_execute',function=invoke_native,
            arguments={'entry':'_mlir_ciface_forward','complete_fields':True,'scope':'finite selected bridge'},
            inputs=(result.host_so,original,*tuple(root/f'descriptor-{i}.bin' for i in range(2))),
            outputs=(actual,)) as record:
        invoke_native(native,addresses,actual,y);record.returned()
    assert actual.read_bytes()==expected.tobytes(),(case['name'],y,expected)
    # A genuine final-element mismatch must fail full output transport comparison.
    changed=bytearray(actual.read_bytes());changed[-1]^=1
    assert bytes(changed)!=expected.tobytes()
    descriptor=root/'layout.json';descriptor.write_text(json.dumps(layout.record(),sort_keys=True)+'\n')
    for defect in ('query','object','source','receipt'):
        path={'query':root/'layout/descriptor-query.ll','object':root/'layout/descriptor-query.o',
            'source':prepared,'receipt':receipt}[defect]
        saved=path.read_bytes();path.write_bytes(saved+b'\nchanged\n')
        try:layout.verify()
        except (ValueError,OSError):pass
        else:raise AssertionError('changed actual source/layout accepted')
        path.write_bytes(saved);layout.verify()
    # Deleting the full selected object is unavailable, even with saved observations.
    query_object=root/'layout/descriptor-query.o';saved=query_object.read_bytes();query_object.unlink()
    try:layout.verify()
    except (ValueError,OSError):pass
    else:raise AssertionError('missing actual object accepted')
    query_object.write_bytes(saved);layout.verify()
    rows.append({'case':case['name'],'count':count,'dtype':dtype,'shape':list(shape),
        'layout':str(descriptor),'receipt':str(receipt),'original':str(source),'prepared':str(prepared),
        'pointer_bytes':observed['pointer_bytes'],'descriptor_bytes':[r['descriptor_bytes'] for r in observed['slots']],
        'index_bits':[r['index_bits'] for r in observed['slots']],
        'output':str(actual),'input':str(original),'source_preprocessing_equal':source.read_bytes()==prepared.read_bytes(),
        'scope':('finite host bridge/full values; original body, physical ownership, '
                 'effects and runtime authority unproved')})
(owner/'summary.json').write_text(json.dumps({'rows':rows},sort_keys=True)+'\n')
"""


@pytest.fixture(scope="module")
def native_descriptors(tmp_path_factory):
    names = (
        "MERLIN_COMPILER_PYTHON",
        "MERLIN_LLVM_LLC",
        "MERLIN_CLANG",
        "MERLIN_M2M_DIR",
        "MERLIN_TEST_LAYOUT_SOURCE_ROOT",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip("explicit public original sources/compiler tools are required")
    owner = tmp_path_factory.mktemp("descriptor-native")
    worker, request = owner / "worker.py", owner / "request.json"
    worker.write_text(WORKER)
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", **{name: os.environ[name] for name in names}}
    cases = []
    for name, shape, dtype in (
        ("scalar_integer", (), "i64"),
        ("scalar_round", (), "f32"),
        ("round_tail", (129,), "f32"),
        ("integer_rectangle", (3, 4), "i8"),
    ):
        source = Path(environment["MERLIN_TEST_LAYOUT_SOURCE_ROOT"]) / name / "original.mlir"
        cases.append({"name": name, "source": source.read_bytes().hex(), "shape": list(shape), "dtype": dtype})
    cases.append({**cases[2], "name": "round_tail_i32", "index_bits": 32})
    request.write_text(
        json.dumps(
            {
                "package": str(Path(D.__file__).parents[1]),
                "test_source": __file__,
                "cases": cases,
                "environment": environment,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (owner / "environment.json").write_text(json.dumps(environment, sort_keys=True) + "\n")
    result = I.run(
        [environment["MERLIN_COMPILER_PYTHON"], "-I", "-B", str(worker), str(request), str(owner)],
        directory=owner,
        stage="descriptor_native_controls",
        cwd=owner,
        env=environment,
        inputs=(worker, request, Path(__file__)),
        outputs=(owner / "summary.json",),
        dependencies=tuple(
            Path(D.__file__).with_name(name)
            for name in (
                "descriptor_layout.py",
                "descriptor_wrapper.py",
                "descriptor_contract.py",
                "descriptor_object.py",
            )
        ),
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr.decode()
    for path in owner.rglob("invocation.json"):
        document = I.verify(path)
        if document["kind"] == "subprocess":
            I.require_environment(path, environment=environment)
    return owner, json.loads((owner / "summary.json").read_text())


@pytest.mark.parametrize(
    "case,count,dtype",
    (
        ("scalar_integer", 1, "i64"),
        ("scalar_round", 1, "f32"),
        ("round_tail", 129, "f32"),
        ("integer_rectangle", 12, "i8"),
        ("round_tail_i32", 129, "f32"),
    ),
)
def test_actual_same_driver_descriptor_layout_and_full_native_values(native_descriptors, case, count, dtype):
    owner, summary = native_descriptors
    row = next(row for row in summary["rows"] if row["case"] == case)
    assert row["count"] == count and row["dtype"] == dtype
    observation = json.loads(Path(row["layout"]).read_text())["observation"]
    assert observation["data_layout"] and observation["target_triple"]
    assert len(observation["slots"]) == 2
    assert all(len(slot["fields"]) == 3 + 2 * len(row["shape"]) for slot in observation["slots"])
    assert Path(row["output"]).stat().st_size == count * observation["slots"][1]["element_bytes"]
    assert "body" in row["scope"]


@pytest.mark.parametrize("defect", ("encoding", "kind", "truncate", "name", "count", "write", "relocation"))
def test_actual_compiler_constant_object_malformed_or_partial_roster_refuses(native_descriptors, defect):
    import struct

    from merlin.llvmlower.descriptor_object import SECTION, layout_words

    owner, summary = native_descriptors
    root = owner / "round_tail" / "layout"
    data = json.loads(Path(summary["rows"][2]["layout"]).read_text())["observation"]
    raw = bytearray((root / "descriptor-query.o").read_bytes())
    endian = "<" if raw[5] == 1 else ">"
    assert raw[4] == 2  # This selected host control, not a production default.
    shoff = struct.unpack_from(endian + "Q", raw, 40)[0]
    size, count = struct.unpack_from(endian + "HH", raw, 58)
    rows = [struct.unpack_from(endian + "IIQQQQIIQQ", raw, shoff + i * size) for i in range(count)]
    selected = next(i for i, row in enumerate(rows) if row[1] == 1 and row[5] == data["word_count"] * 8)
    if defect == "encoding":
        raw[5] = 0
    elif defect == "kind":
        struct.pack_into(endian + "H", raw, 16, 2)
    elif defect == "truncate":
        raw = raw[:64]
    elif defect == "name":
        start = raw.index(SECTION.encode())
        raw[start] = ord("x")
    elif defect == "write":
        struct.pack_into(endian + "Q", raw, shoff + selected * size + 8, rows[selected][2] | 1)
    elif defect == "relocation":
        ordinal = next(i for i in range(1, count) if i != selected)
        struct.pack_into(endian + "I", raw, shoff + ordinal * size + 4, 4)
        struct.pack_into(endian + "I", raw, shoff + ordinal * size + 44, selected)
    with pytest.raises(ValueError):
        layout_words(bytes(raw), count=data["word_count"] + (defect == "count"))
