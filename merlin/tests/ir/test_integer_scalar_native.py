"""Ordinary integer source, LLVM, object, linked image and full scalar values.

Native examples test the pipeline and counterexamples; the structural checker
proves only its accepted complete typed DAGs under the selected IR contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import integer_scalar_correspondence as C


def scalar(body, inputs=(64, 64, 64), outputs=(64,)):
    arguments = ", ".join(f"%arg{i}: i{bits}" for i, bits in enumerate(inputs))
    results = ", ".join(f"i{bits}" for bits in outputs)
    return f"module {{ func.func @forward({arguments}) -> ({results}) {{ {body} }} }}"


def compare(predicate):
    return scalar(
        f"%p = arith.cmpi {predicate}, %arg0, %arg1 : i64\n"
        "%v = arith.select %p, %arg0, %arg2 : i64\nfunc.return %v : i64"
    )


PREDICATES = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
CASES = [
    {
        "name": predicate,
        "source": compare(predicate),
        "inputs": [64, 64, 64],
        "outputs": [64],
        "samples": [[-3, 2, 19], [2, -3, 19], [3, 3, 19]],
        "predicate": predicate,
    }
    for predicate in PREDICATES
]
CASES += [
    {
        "name": "large_constant",
        "source": scalar(
            "%c = arith.constant 9007199254740995 : i64\n%v = arith.addi %arg0, %c : i64\nfunc.return %v : i64", (64,)
        ),
        "inputs": [64],
        "outputs": [64],
        "samples": [[-17], [29]],
        "expected": [9007199254740978, 9007199254741024],
    },
    {
        "name": "zero_input",
        "source": scalar("%c = arith.constant 9007199254740995 : i64\nfunc.return %c : i64", ()),
        "inputs": [],
        "outputs": [64],
        "samples": [[]],
        "expected": [9007199254740995],
    },
    {
        "name": "overflow",
        "source": scalar("%v = arith.addi %arg0, %arg1 : i8\nfunc.return %v : i8", (8, 8), (8,)),
        "inputs": [8, 8],
        "outputs": [8],
        "samples": [[127, 1], [-128, -1]],
        "expected": [-128, 127],
    },
    {
        "name": "repeated_operand",
        "source": scalar("%v = arith.muli %arg0, %arg0 : i64\nfunc.return %v : i64", (64,)),
        "inputs": [64],
        "outputs": [64],
        "samples": [[-7], [9007199254740995]],
        "expected": [49, 54043195528445961],
    },
    {
        "name": "bitwise_dag",
        "source": scalar(
            "%a = arith.addi %arg0, %arg1 : i64\n"
            "%s = arith.subi %arg0, %arg1 : i64\n"
            "%m = arith.muli %a, %arg2 : i64\n"
            "%b = arith.andi %m, %s : i64\n"
            "%o = arith.ori %b, %arg0 : i64\n"
            "%v = arith.xori %o, %arg1 : i64\nfunc.return %v : i64"
        ),
        "inputs": [64, 64, 64],
        "outputs": [64],
        "samples": [[13, 7, 11], [-19, 11, -5]],
        "expected": [10, -26],
    },
]
UNSUPPORTED = [
    {
        "name": "integer_minimum",
        "source": scalar("%v = arith.minsi %arg0, %arg1 : i64\nfunc.return %v : i64", (64, 64)),
        "inputs": [64, 64],
        "outputs": [64],
        "samples": [[-3, 2], [5, -7]],
        "expected": [-3, -7],
    },
    {
        "name": "overflow_promise",
        "source": scalar("%v = arith.addi %arg0, %arg1 overflow<nsw> : i64\nfunc.return %v : i64", (64, 64)),
        "inputs": [64, 64],
        "outputs": [64],
        "samples": [[-3, 2], [5, -7]],
        "expected": [-1, -2],
    },
]


WORKER = """import hashlib,json,sys
from dataclasses import replace
from pathlib import Path
request=Path(sys.argv[1]);owner=Path(sys.argv[2]);data=json.loads(request.read_text())
sys.path[:0]=data['import_roots']
from merlin.common import invocation_record as I
from merlin.llvmlower.abi import HostModel,PrivateHostImagePolicy,ScalarArg
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.integer_scalar_contract import *
from merlin.llvmlower.integer_scalar_correspondence import check_integer_scalar_correspondence
from merlin.llvmlower.llvm_dialect_product import verify_llvm_dialect_product
LIMITS=IntegerScalarLimits(1000000,200000,64,128,100)
NUMERICS=IntegerScalarNumerics('modular_bitvector','typed_integer_predicates','none')
def original(path,case):
    abi=OriginalIntegerScalarAbi('forward','_mlir_ciface_forward',
        tuple(IntegerScalarSlot('arg'+str(i),bits) for i,bits in enumerate(case['inputs'])),
        tuple(IntegerScalarSlot('result'+str(i),bits) for i,bits in enumerate(case['outputs'])))
    return OriginalIntegerScalarSource(path,hashlib.sha256(path.read_bytes()).hexdigest(),abi,NUMERICS)
def checked(root,selection,receipt):
    output=root/'correspondence.json'
    product=verify_llvm_dialect_product(receipt)
    with I.observe_call(root,stage='integer_scalar_source_correspondence',function=check_integer_scalar_correspondence,
        arguments={'original':selection.record(LIMITS),'limits':LIMITS.record()},
        inputs=(selection.path,receipt,Path(product['source']['path']),Path(product['llvm_dialect']['path'])),
        outputs=(output,)) as observed:
        result=check_integer_scalar_correspondence(original=selection,retained_product=receipt,limits=LIMITS)
        output.write_text(json.dumps(result,sort_keys=True)+'\\n');observed.returned()
    return result
def compiled(root,case):
    root.mkdir();path=root/'original.mlir';path.write_text(case['source']);build=root/'ordinary'
    products=tuple(build/name for name in ('model.mlir','run_lowering.py','lowering_recipe.json','model.ll',
        'model_host.o','mlir_runtime_host.o','model_host.so'))
    with I.observe_call(root,stage='ordinary_integer_scalar_compile',function=lower_model,
        arguments={'retain_llvm_dialect':True},inputs=(path,),outputs=products) as observed:
        result=lower_model(path.read_text(),build,retain_llvm_dialect=True);observed.returned()
    receipt=Path(result.stats['llvm_dialect_product']['path']);selection=original(path,case)
    proof=checked(root,selection,receipt)
    image=HostModel.load(str(result.host_so),image_policy=PrivateHostImagePolicy(build),scalar_result_dtype='i'+str(case['outputs'][0]))
    actual=root/'actual.json';samples=root/'samples.json';samples.write_text(json.dumps(case['samples'])+'\\n')
    with I.observe_call(root,stage='ordinary_integer_scalar_execute',function=image.__call__,
        arguments={'ordered_input_widths':case['inputs'],'output_widths':case['outputs']},
        inputs=(result.host_so,samples),outputs=(actual,)) as observed:
        values=[image([ScalarArg(value,'i'+str(bits)) for value,bits in zip(sample,case['inputs'],strict=True)])
            for sample in case['samples']]
        actual.write_text(json.dumps({'samples':case['samples'],'complete_scalar_outputs':values})+'\\n');observed.returned()
    return {'name':case['name'],'proof':proof,'values':values,'receipt':str(receipt)},selection,receipt
rows=[];selections={};receipts={};unsupported=[]
for case in data['cases']:
    row,selection,receipt=compiled(owner/case['name'],case);rows.append(row);selections[case['name']]=selection;receipts[case['name']]=receipt
for case in data['unsupported']:
    row,_,_=compiled(owner/case['name'],case);unsupported.append(row)
negatives=[]
for defect in data['defects']:
    root=owner/('defect-'+defect['name']);row,_,receipt=compiled(root,defect)
    comparison=root/'comparison';comparison.mkdir();result=checked(comparison,selections[defect['original']],receipt)
    negatives.append({'name':defect['name'],'proof':result,'values':row['values']})
aggregate=owner/'aggregate';aggregate.mkdir();source=aggregate/'original.mlir';source.write_text(data['aggregate'])
selection=original(source,{'inputs':[64,64],'outputs':[64,64]});selected={}
lower_to_llvm_ir(data['aggregate'].replace('-> (i64, i64) {','-> (i64, i64) attributes {llvm.emit_c_interface} {'),
    workdir=aggregate/'ordinary',retain_llvm_dialect=True,lowering_selection=selected)
receipt=Path(selected['llvm_dialect_product']['path']);result=checked(aggregate,selection,receipt)
rows.append({'name':'aggregate','proof':result})
root=owner/'no-wrapper';root.mkdir();source=root/'original.mlir';source.write_text(data['cases'][0]['source'])
selected={};selection=original(source,data['cases'][0])
lower_to_llvm_ir(source.read_text(),workdir=root/'ordinary',retain_llvm_dialect=True,lowering_selection=selected)
receipt=Path(selected['llvm_dialect_product']['path']);result=checked(root,selection,receipt)
rows.append({'name':'no-wrapper','proof':result})
drifts=[]
for field,action in data['drifts']:
    root=owner/('drift-'+field+'-'+action);root.mkdir();selected={}
    source=root/'original.mlir';source.write_text(data['cases'][0]['source']);selection=original(source,data['cases'][0])
    lower_to_llvm_ir(source.read_text().replace('-> (i64) {','-> (i64) attributes {llvm.emit_c_interface} {'),
        workdir=root/'ordinary',retain_llvm_dialect=True,lowering_selection=selected)
    receipt=Path(selected['llvm_dialect_product']['path']);product=verify_llvm_dialect_product(receipt)
    path=source if field=='original' else (receipt if field=='receipt' else Path(product[field]['path']))
    (root/'before.json').write_text(json.dumps(product,sort_keys=True)+'\\n')
    if action=='delete':path.unlink()
    else:path.write_bytes(path.read_bytes()+b'\\n changed original product\\n')
    result=check_integer_scalar_correspondence(original=selection,retained_product=receipt,limits=LIMITS)
    assert result['status']=='UNKNOWN',result
    drifts.append({'field':field,'action':action,'proof':result,'record':product['invocation']['path']})
proof=rows[0]['proof'];selection=selections[data['cases'][0]['name']];receipt=receipts[data['cases'][0]['name']]
unavailable=[]
for name,changed in (
    ('missing_output',replace(selection,abi=replace(selection.abi,outputs=selection.abi.outputs+(IntegerScalarSlot('missing',64),)))),
    ('wrong_input_order',replace(selection,abi=replace(selection.abi,inputs=(IntegerScalarSlot('arg0',32),*selection.abi.inputs[1:])))),
    ('numeric_policy',replace(selection,numerics=replace(NUMERICS,arithmetic='bounded_exact'))),
):
    result=check_integer_scalar_correspondence(original=changed,retained_product=receipt,limits=LIMITS)
    assert result['status']=='UNKNOWN',result;unavailable.append({'name':name,'proof':result})
(owner/'summary.json').write_text(json.dumps({'rows':rows,'negatives':negatives,'drifts':drifts,'unavailable':unavailable,'unsupported':unsupported},sort_keys=True)+'\\n')
"""


@pytest.fixture(scope="module")
def native_products(tmp_path_factory):
    names = ("MERLIN_COMPILER_PYTHON", "MERLIN_LLVM_LLC", "MERLIN_CLANG")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("integer correspondence native controls need explicitly selected compiler Python, llc and clang")
    selected = {name: str(Path(os.environ[name]).absolute()) for name in names}
    assert all(Path(value).is_file() for value in selected.values())
    owner = tmp_path_factory.mktemp("integer-scalar-native")
    worker, request = owner / "worker.py", owner / "request.json"
    worker.write_text(WORKER)
    defects = [
        {**CASES[2], "name": "changed_predicate", "source": compare("sgt"), "original": "slt"},
        {
            **CASES[10],
            "name": "changed_constant",
            "source": CASES[10]["source"].replace("9007199254740995", "9007199254740996"),
            "original": "large_constant",
        },
        {
            **CASES[2],
            "name": "changed_return",
            "source": CASES[2]["source"].replace("func.return %v", "func.return %arg2"),
            "original": "slt",
        },
    ]
    drifts = [
        ("original", "change"),
        ("llvm_dialect", "change"),
        ("llvm_dialect", "delete"),
        ("source", "change"),
        ("runner", "change"),
        ("translated_llvm_ir", "change"),
        ("receipt", "change"),
        ("invocation", "change"),
    ]
    request.write_text(
        json.dumps(
            {
                "import_roots": [str(Path(C.__file__).parents[2])],
                "cases": CASES,
                "unsupported": UNSUPPORTED,
                "defects": defects,
                "drifts": drifts,
                "aggregate": scalar(
                    "%v = arith.muli %arg0, %arg0 : i64\nfunc.return %v, %arg1 : i64, i64", (64, 64), (64, 64)
                ),
            }
        )
        + "\n"
    )
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", **selected}
    (owner / "environment.json").write_text(json.dumps(environment, sort_keys=True) + "\n")
    result = I.run(
        [os.sys.executable, "-I", "-B", str(worker), str(request), str(owner)],
        directory=owner,
        stage="integer_scalar_native_controls",
        inputs=(worker, request),
        outputs=(owner / "summary.json",),
        env=environment,
        cwd=owner,
        timeout=240,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    observed = next(
        path
        for path in (owner / "invocations").glob("*/invocation.json")
        if json.loads(path.read_text())["stage"] == "integer_scalar_native_controls"
    )
    I.require_environment(observed, environment=environment)
    return owner, json.loads((owner / "summary.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["name"])
def test_ordinary_source_to_scalar_wrapper_complete_values_and_structural_proof(native_products, case):
    owner, summary = native_products
    row = next(row for row in summary["rows"] if row["name"] == case["name"])
    assert row["proof"]["status"] == "PROVED", row["proof"]
    if "predicate" in case:
        expected = []
        for x, y, z in case["samples"]:
            left, right = (x % (1 << 64), y % (1 << 64)) if case["predicate"].startswith("u") else (x, y)
            relation = {
                "eq": left == right,
                "ne": left != right,
                "lt": left < right,
                "le": left <= right,
                "gt": left > right,
                "ge": left >= right,
            }
            expected.append(x if relation[case["predicate"].lstrip("su")] else z)
    else:
        expected = case["expected"]
    assert row["values"] == expected
    assert row["proof"]["facts"]["output_widths"] == case["outputs"]
    for name in ("model_host.o", "model_host.so"):
        assert (owner / case["name"] / "ordinary" / name).read_bytes().startswith(b"\x7fELF")


@pytest.mark.parametrize(
    "name,original", (("changed_predicate", "slt"), ("changed_constant", "large_constant"), ("changed_return", "slt"))
)
def test_real_changed_candidate_values_and_dag_correspondence_refuse(native_products, name, original):
    _, summary = native_products
    defect = next(row for row in summary["negatives"] if row["name"] == name)
    baseline = next(row for row in summary["rows"] if row["name"] == original)
    assert defect["proof"]["status"] == "UNKNOWN"
    assert "DAGs differ" in defect["proof"]["detail"]
    assert defect["values"] != baseline["values"]


@pytest.mark.parametrize(
    "field,action",
    (
        ("original", "change"),
        ("llvm_dialect", "change"),
        ("llvm_dialect", "delete"),
        ("source", "change"),
        ("runner", "change"),
        ("translated_llvm_ir", "change"),
        ("receipt", "change"),
        ("invocation", "change"),
    ),
)
def test_missing_changed_and_stale_live_artifacts_refuse(native_products, field, action):
    _, summary = native_products
    row = next(row for row in summary["drifts"] if row["field"] == field and row["action"] == action)
    assert row["proof"]["status"] == "UNKNOWN"


def test_aggregate_return_keeps_all_requested_outputs_unknown(native_products):
    _, summary = native_products
    row = next(row for row in summary["rows"] if row["name"] == "aggregate")
    assert row["proof"]["status"] == "UNKNOWN"
    assert [slot["name"] for slot in row["proof"]["original"]["abi"]["outputs"]] == ["result0", "result1"]
    assert "aggregate/repeated" in row["proof"]["detail"]


def test_actual_source_without_c_interface_cannot_prove_original_public_boundary(native_products):
    _, summary = native_products
    row = next(row for row in summary["rows"] if row["name"] == "no-wrapper")
    assert row["proof"]["status"] == "UNKNOWN"
    assert "complete emitted scalar source and C-interface" in row["proof"]["detail"]


@pytest.mark.parametrize("name", ("missing_output", "wrong_input_order", "numeric_policy"))
def test_complete_original_slots_and_selected_numeric_contract_are_mandatory(native_products, name):
    _, summary = native_products
    row = next(row for row in summary["unavailable"] if row["name"] == name)
    assert row["proof"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("case", UNSUPPORTED, ids=lambda row: row["name"])
def test_actual_optimized_intrinsic_or_overflow_promise_remains_unknown(native_products, case):
    _, summary = native_products
    row = next(row for row in summary["unsupported"] if row["name"] == case["name"])
    assert row["values"] == case["expected"]
    assert row["proof"]["status"] == "UNKNOWN", row["proof"]
    assert row["proof"]["original"]["abi"]["outputs"] == [{"name": "result0", "bits": 64}]
    assert "unsupported" in row["proof"]["detail"]


@pytest.mark.parametrize("defect", ("call_argument_order", "wrong_return", "wrong_callee"))
def test_actual_parsed_wrapper_checks_complete_call_and_return_identity(native_products, defect):
    from xdsl.dialects import builtin

    from merlin.common.strict_json import loads
    from merlin.llvmlower.integer_scalar_contract import (
        IntegerScalarLimits,
        IntegerScalarSlot,
        OriginalIntegerScalarAbi,
    )

    _, summary = native_products
    row = next(row for row in summary["rows"] if row["name"] == "slt")
    product = loads(Path(row["receipt"]).read_bytes())
    limits = IntegerScalarLimits(1_000_000, 200_000, 64, 128, 100)
    module = C._parse(Path(product["llvm_dialect"]["path"]).read_bytes(), limits, emitted=True)
    wrapper = next(op for op in module.body.block.ops if op.sym_name.data == "_mlir_ciface_forward")
    block = wrapper.body.block
    call, returned = tuple(block.ops)
    if defect == "call_argument_order":
        call.operands = (block.args[1], block.args[0], block.args[2])
    elif defect == "wrong_return":
        returned.operands = (block.args[0],)
    else:
        call.properties["callee"] = builtin.SymbolRefAttr("_mlir_ciface_forward")
    module.verify()  # The defect is well typed; identity is an independent check.
    abi = OriginalIntegerScalarAbi(
        "forward",
        "_mlir_ciface_forward",
        tuple(IntegerScalarSlot("arg" + str(i), 64) for i in range(3)),
        (IntegerScalarSlot("result0", 64),),
    )
    with pytest.raises(ValueError, match="(identity|different original entry)"):
        C._wrapper(wrapper, abi, (64, 64, 64), (64,), limits)
