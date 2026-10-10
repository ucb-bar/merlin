"""Actual serial upstream products and host outputs, without semantic authority."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import llvm_dialect_product as P
from merlin.llvmlower import pipeline

SCALAR = """module {
  func.func @forward(%x: i64, %y: i64) -> i64 {
    %v = arith.addi %x, %y : i64
    func.return %v : i64
  }
}
"""


def _copy(shape):
    typed = "tensor<" + "x".join(map(str, shape)) + "xi32>"
    return f"""module {{
  func.func @forward(%x: {typed}) -> {typed} {{
    %e = tensor.empty() : {typed}
    %v = linalg.copy ins(%x : {typed}) outs(%e : {typed}) -> {typed}
    func.return %v : {typed}
  }}
}}
"""


RETRANSLATE = """import sys
from pathlib import Path
from torch_mlir import ir
from torch_mlir.dialects import llvm
with ir.Context() as context:
    module=ir.Module.parse(Path(sys.argv[1]).read_text(),context)
    module.operation.verify()
    Path(sys.argv[2]).write_text(str(llvm.translate_module_to_llvmir(module.operation)))
"""

WORKER = """import json,sys
from pathlib import Path
request=Path(sys.argv[1]);owner=Path(sys.argv[2]);data=json.loads(request.read_text())
sys.path[:0]=data['import_roots']
import numpy as np
from merlin.common import invocation_record as I
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.abi import HostModel,PrivateHostImagePolicy,ScalarArg
from merlin.llvmlower.kernel_backend import compile_host
from merlin.llvmlower.lower import lower_model,lower_model_file
from merlin.llvmlower.llvm_dialect_product import verify_llvm_dialect_product
from merlin.llvmlower.pipeline import lower_to_llvm_ir,PipelineError
from merlin.llvmlower.target_data_layout import parse,default_index_bits
query=owner/'layout.c';query.write_text('void selected_layout(void) {}\\n')
layout_path=owner/'selected-layout.ll'
result=I.run([data['clang'],'-S','-emit-llvm','-x','c',str(query),'-o',str(layout_path)],
    directory=owner,stage='selected_host_layout',inputs=(query,),outputs=(layout_path,),
    capture_output=True,timeout=60)
assert result.returncode==0,result.stderr
layout=parse(layout_path.read_text());assert layout
try:
    width={'status':'observed','bits':default_index_bits(layout)}
except ValueError as error:
    width={'status':'unavailable','reason':str(error)}
rows=[]
for case in data['cases']:
    root=owner/case['name'];root.mkdir();source=root/'original.mlir';source.write_text(case['source'])
    build=root/'ordinary';products=tuple(build/name for name in
        ('model.mlir','run_lowering.py','lowering_recipe.json','model.ll','model_host.o','mlir_runtime_host.o','model_host.so'))
    function=(compile_host if case['name']=='scalar' else
        (lower_model_file if case['name']=='rectangle' else lower_model))
    with I.observe_call(root,stage='ordinary_retained_host_compile',function=function,
            arguments={'retain_llvm_dialect':True},inputs=(source,),outputs=products) as observed:
        if case['name']=='scalar':
            native=compile_host(parse_mlir_text(case['source']),build,retain_llvm_dialect=True)
            receipt=next(build.glob('llvm-dialect-*/product.json'))
        else:
            res=(lower_model_file(source,build,retain_llvm_dialect=True,data_layout=layout)
                if case['name']=='rectangle' else lower_model(case['source'],build,retain_llvm_dialect=True))
            receipt=Path(res.stats['llvm_dialect_product']['path'])
            native=HostModel.load(str(res.host_so),image_policy=PrivateHostImagePolicy(build))
        observed.returned()
    product=verify_llvm_dialect_product(receipt,llvm_ir=(build/'model.ll').read_text())
    replay=root/'retranslated.ll'
    result=I.run([data['python'],'-I','-B',data['retranslator'],product['llvm_dialect']['path'],str(replay)],
        directory=root,stage='retained_module_native_retranslation',
        inputs=(Path(data['retranslator']),Path(product['llvm_dialect']['path'])),outputs=(replay,),
        capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert replay.read_bytes()==Path(product['translated_llvm_ir']['path']).read_bytes()
    actual=root/'actual.bin';original=root/'input.bin'
    if case['name']=='scalar':
        values=((1<<53)+3,-7);original.write_bytes(np.array(values,dtype=np.int64).tobytes())
        with I.observe_call(root,stage='ordinary_retained_host_execute',function=native.__call__,
                arguments={'ordered_inputs':list(values),'result_dtype':'i64'},
                inputs=(build/'model_host.so',original),outputs=(actual,)) as observed:
            result=native((ScalarArg(values[0],'i64'),ScalarArg(values[1],'i64')))
            assert result==sum(values)
            actual.write_bytes(np.array([result],dtype=np.int64).tobytes());observed.returned()
        checked=1
    else:
        shape=tuple(case['shape']);count=int(np.prod(shape))
        x=(np.arange(count,dtype=np.int32)*13-71).reshape(shape)
        y=np.full(shape,-9999,dtype=np.int32);original.write_bytes(x.tobytes())
        with I.observe_call(root,stage='ordinary_retained_host_execute',function=native.__call__,
                arguments={'ordered_shapes':[list(shape),list(shape)],'dtype':'i32'},
                inputs=(build/'model_host.so',original),outputs=(actual,)) as observed:
            native(((x.ctypes.data,x.shape),(y.ctypes.data,y.shape)))
            assert np.array_equal(x,y);actual.write_bytes(y.tobytes());observed.returned()
        checked=count
    rows.append({'name':case['name'],'receipt':str(receipt),'checked_elements':checked,
        'layout':parse((build/'model.ll').read_text()),'selected_host_layout':layout,'default_pointer_index':width,
        'scope':'finite host outputs and artifact custody; storage, semantic domain, effects and runtime unqualified'})
negatives=[]
for field,action in data['mutations']:
    root=owner/('negative-'+field+'-'+action);root.mkdir();selection={}
    emitted=lower_to_llvm_ir(data['cases'][0]['source'],workdir=root,retain_llvm_dialect=True,
        lowering_selection=selection)
    receipt=Path(selection['llvm_dialect_product']['path'])
    product=verify_llvm_dialect_product(receipt,llvm_ir=emitted)
    path=Path(product[field]['path']);original=path.read_bytes()
    (root/'original-product.json').write_text(json.dumps(product,sort_keys=True)+'\\n')
    if action=='delete':
        path.unlink()
    else:
        path.write_bytes(original+b'\\n; changed product\\n')
    try:
        verify_llvm_dialect_product(receipt,llvm_ir=emitted)
    except (ValueError,OSError) as error:
        negatives.append({'field':field,'action':action,'refusal':type(error).__name__,
            'invocation':product['invocation']['path']})
    else:
        raise AssertionError('changed original product accepted')
default=owner/'default';default.mkdir();selected=owner/'selected';selected.mkdir();selection={}
ordinary=lower_to_llvm_ir(data['cases'][0]['source'],workdir=default)
retained=lower_to_llvm_ir(data['cases'][0]['source'],workdir=selected,
    retain_llvm_dialect=True,lowering_selection=selection)
assert ordinary==retained and not list(default.glob('llvm-dialect-*'))
stale=owner/'stale';stale.mkdir();selection={}
lower_to_llvm_ir(data['cases'][0]['source'],workdir=stale,retain_llvm_dialect=True,lowering_selection=selection)
old=Path(selection['llvm_dialect_product']['path']);old_doc=verify_llvm_dialect_product(old)
prior=tuple(Path(old_doc[name]['path']).read_bytes() for name in ('llvm_dialect','translated_llvm_ir'))
try:
    lower_to_llvm_ir(data['cases'][0]['source'],workdir=stale,retain_llvm_dialect=True,
        lowering_selection=selection,pipeline='builtin.module(no-such-retention-pass)')
except PipelineError:
    assert 'llvm_dialect_product' not in selection
    assert prior==tuple(Path(old_doc[name]['path']).read_bytes() for name in ('llvm_dialect','translated_llvm_ir'))
    assert len(list(stale.glob('llvm-dialect-*/product.json')))==1
    try:
        verify_llvm_dialect_product(old)
    except ValueError:
        pass
    else:
        raise AssertionError('old receipt accepted changed invocation source')
else:
    raise AssertionError('failed lowering returned a retained product')
stale_records=[str(path) for path in stale.rglob('invocation.json')]
(owner/'summary.json').write_text(json.dumps({'rows':rows,'negatives':negatives,'stale_records':stale_records},
    sort_keys=True)+'\\n')
"""


@pytest.fixture(scope="module")
def actual_products(tmp_path_factory):
    names = ("MERLIN_COMPILER_PYTHON", "MERLIN_LLVM_LLC", "MERLIN_CLANG")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("retained upstream host controls require explicit compiler Python, llc and clang")
    selected = {name: str(Path(os.environ[name]).absolute()) for name in names}
    assert all(Path(value).is_file() for value in selected.values())
    owner = tmp_path_factory.mktemp("serial-llvm-products")
    worker, request, retranslator = (owner / name for name in ("worker.py", "request.json", "retranslate.py"))
    worker.write_text(WORKER)
    retranslator.write_text(RETRANSLATE)
    request.write_text(
        json.dumps(
            {
                "python": selected["MERLIN_COMPILER_PYTHON"],
                "clang": selected["MERLIN_CLANG"],
                "retranslator": str(retranslator),
                "import_roots": [str(Path(P.__file__).parents[2])],
                "cases": [
                    {"name": "scalar", "source": SCALAR},
                    {"name": "rectangle", "source": _copy((3, 5)), "shape": [3, 5]},
                    {"name": "tail", "source": _copy((129,)), "shape": [129]},
                ],
                "mutations": [
                    [field, action]
                    for field, action in (
                        ("llvm_dialect", "change"),
                        ("llvm_dialect", "delete"),
                        ("translated_llvm_ir", "change"),
                        ("translated_llvm_ir", "delete"),
                        ("source", "change"),
                        ("runner", "change"),
                        ("invocation", "change"),
                    )
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        **selected,
        "PYTHONPATH": str(Path(P.__file__).parents[2]),
    }
    (owner / "environment.json").write_text(json.dumps(environment, sort_keys=True) + "\n")
    result = I.run(
        [selected["MERLIN_COMPILER_PYTHON"], "-I", "-B", str(worker), str(request), str(owner)],
        directory=owner,
        stage="ordinary_retained_serial_controls",
        cwd=owner,
        env=environment,
        inputs=(worker, request, retranslator),
        dependencies=(Path(P.__file__),),
        outputs=(owner / "summary.json",),
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr.decode()
    summary = json.loads((owner / "summary.json").read_text())
    invalidated = {row["invocation"] for row in summary["negatives"]} | set(summary["stale_records"])
    for record in owner.rglob("invocation.json"):
        if str(record) in invalidated:
            with pytest.raises((ValueError, OSError)):
                I.verify(record)
            continue
        observed = I.verify(record)
        if observed["kind"] == "subprocess":
            I.require_environment(record, environment=environment)
    return owner, summary


@pytest.mark.parametrize("name,count", (("scalar", 1), ("rectangle", 15), ("tail", 129)))
def test_actual_post_pass_translation_and_complete_host_values(actual_products, name, count):
    owner, summary = actual_products
    row = next(row for row in summary["rows"] if row["name"] == name)
    product = P.verify_llvm_dialect_product(row["receipt"])
    assert row["checked_elements"] == count
    assert '"llvm.func"' in Path(product["llvm_dialect"]["path"]).read_text()
    assert (owner / name / "ordinary/model_host.o").read_bytes().startswith(b"\x7fELF")
    assert (owner / name / "ordinary/model_host.so").read_bytes().startswith(b"\x7fELF")
    assert row["layout"] == (row["selected_host_layout"] if name == "rectangle" else None)


@pytest.mark.parametrize(
    "field,action",
    (
        ("llvm_dialect", "change"),
        ("llvm_dialect", "delete"),
        ("translated_llvm_ir", "change"),
        ("translated_llvm_ir", "delete"),
        ("source", "change"),
        ("runner", "change"),
        ("invocation", "change"),
    ),
)
def test_actual_retained_product_drift_refuses(actual_products, field, action):
    _, summary = actual_products
    assert any(row["field"] == field and row["action"] == action for row in summary["negatives"])


def test_native_default_parity_and_failed_reuse_preserve_raw_products(actual_products):
    owner, summary = actual_products
    assert not list((owner / "default").glob("llvm-dialect-*"))
    assert len(summary["stale_records"]) == 2
    assert len(list((owner / "stale").glob("llvm-dialect-*/product.json"))) == 1


def test_actual_product_refuses_wrong_returned_ir(actual_products):
    _, summary = actual_products
    row = summary["rows"][0]
    with pytest.raises(ValueError, match="returned IR binding"):
        P.verify_llvm_dialect_product(row["receipt"], llvm_ir="changed returned IR")


def test_actual_product_refuses_wrong_output_roster(actual_products):
    owner, summary = actual_products
    original = Path(summary["rows"][0]["receipt"])
    document = P.verify_llvm_dialect_product(original)
    # A real, byte-identical file is not an output of the original invocation.
    replacement = original.parent / "unproduced-copy.mlir"
    replacement.write_bytes(Path(document["llvm_dialect"]["path"]).read_bytes())
    document["llvm_dialect"]["path"] = str(replacement)
    changed = original.parent / "wrong-roster.json"
    changed.write_text(json.dumps(document, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="same ordinary translation invocation"):
        P.verify_llvm_dialect_product(changed)
    assert (
        replacement.read_bytes() == Path(P.verify_llvm_dialect_product(original)["llvm_dialect"]["path"]).read_bytes()
    )
    assert (owner / "scalar/actual.bin").is_file()


@pytest.mark.parametrize("value", (None, 0, 1, "true"))
def test_retention_selection_requires_actual_bool(tmp_path, value):
    with pytest.raises(ValueError, match="explicit bool"):
        pipeline.lower_to_llvm_ir(SCALAR, workdir=tmp_path, retain_llvm_dialect=value)
    assert not list(tmp_path.iterdir())


def test_parallel_retention_is_explicitly_unsupported(tmp_path):
    with pytest.raises(ValueError, match="ordinary serial"):
        pipeline.lower_to_llvm_ir(SCALAR, workdir=tmp_path, parallel=True, retain_llvm_dialect=True)
    assert not list(tmp_path.glob("llvm-dialect-*"))


def test_unselected_runner_emitter_is_unchanged():
    assert P.select_retention(Path("unused-work"), False, omp=False, scalar_stage=None) is None
    source = pipeline._select_runner("builtin.module()", frozenset(), emit=pipeline.EMIT_TRANSLATE)
    assert "print_generic_op_form=True" not in source
    assert pipeline.EMIT_TRANSLATE in source
