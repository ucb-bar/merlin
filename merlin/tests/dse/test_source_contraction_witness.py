"""Source-native bounded contraction evidence, independent of any backend."""
import hashlib

import numpy as np
import pytest

from merlin.perf.source_contraction_witness import extract_source_contraction, evaluate_source_contraction


def source(*, generic=False, cast="extsi", initial="argument", output="i8"):
    inputs = "%a:tensor<3x9xi8>,%b:tensor<9x4xi8>"
    prefix = ""
    if initial == "argument":
        inputs += f",%c:tensor<3x4x{output}>"
    else:
        prefix = f"%empty=tensor.empty():tensor<3x4x{output}>\n"
        if initial == "empty":
            prefix += f"%c=tensor.empty():tensor<3x4x{output}>\n"
        else:
            prefix += f"%z=arith.constant 7:{output}\n%c=linalg.fill ins(%z:{output}) outs(%empty:tensor<3x4x{output}>)->tensor<3x4x{output}>\n"
    if generic:
        casts = f"%aa=arith.{cast} %x:i8 to {output}\n%bb=arith.{cast} %y:i8 to {output}\n" if output != "i8" else ""
        a,b = ("%aa","%bb") if casts else ("%x","%y")
        operation = f'''%r=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}}
ins(%a,%b:tensor<3x9xi8>,tensor<9x4xi8>) outs(%c:tensor<3x4x{output}>) {{
^bb0(%x:i8,%y:i8,%acc:{output}):
{casts}%p=arith.muli {a},{b}:{output}
%v=arith.addi %p,%acc:{output}
linalg.yield %v:{output}
}}->tensor<3x4x{output}>'''
    else:
        operation = f"%r=linalg.matmul ins(%a,%b:tensor<3x9xi8>,tensor<9x4xi8>) outs(%c:tensor<3x4x{output}>)->tensor<3x4x{output}>"
    return f"module {{func.func @work({inputs})->tensor<3x4x{output}>{{\n{prefix}{operation}\nfunc.return %r:tensor<3x4x{output}>\n}}}}"


def extract(text, index=0, **kwargs):
    return extract_source_contraction(text,index,entry="work",max_m=2,max_n=3,max_k=5,**kwargs)


def arrays(record):
    return [np.full(shape, -128 if shape == [2,5] else 127 if shape == [5,3] else 123,
                    dtype="int"+dtype[1:])
            for i,(shape,dtype) in enumerate(zip(record["input_shapes"],record["input_dtypes"]))]


def wrap(value,bits):
    return (int(value)+(1<<(bits-1))) % (1<<bits)-(1<<(bits-1))


@pytest.mark.parametrize("generic",[False,True])
def test_signed_narrow_source_wraps_not_saturates_and_retains_spelling(generic):
    original=source(generic=generic)
    probe,record=extract(original)
    values=arrays(record)
    result=evaluate_source_contraction(probe,record,values)
    assert record["source_op_name"] == ("linalg.generic" if generic else "linalg.matmul")
    assert record["source_geometry_mkn"] == [3,9,4]
    assert record["probe_geometry_mkn"] == [2,5,3]
    assert record["source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    np.testing.assert_array_equal(result,np.full((2,3),wrap(123+5*(-128)*127,8),dtype=np.int8))
    assert not record["runtime_admitted"] and record["emitted_route_correspondence"] == "UNPROVEN"


@pytest.mark.parametrize("cast",["extsi","extui"])
def test_source_extension_signedness_and_i32_accumulation_overflow(cast):
    probe,record=extract(source(generic=True,cast=cast,output="i32"))
    values=arrays(record)
    values[2].fill(2**31-1)
    result=evaluate_source_contraction(probe,record,values)
    a = -128 if cast == "extsi" else 128
    np.testing.assert_array_equal(result,np.full((2,3),wrap(2**31-1+5*a*127,32),dtype=np.int32))
    assert f"arith.{cast}" in probe


def test_real_fill_is_retained_and_undefined_initial_output_refused():
    probe,record=extract(source(initial="fill"),3)
    values=arrays(record)
    result=evaluate_source_contraction(probe,record,values)
    np.testing.assert_array_equal(result,np.full((2,3),wrap(7+5*(-128)*127,8),dtype=np.int8))
    assert "linalg.fill" in probe
    with pytest.raises(ValueError,match="initializer"):
        extract(source(initial="empty"),2)


def test_explicit_source_truncations_are_not_replaced_by_saturation():
    original = source(generic=True, cast="trunci", output="i32").replace("i8", "i64")
    probe,record=extract(original)
    values=arrays(record)
    values[0].fill((1<<40)+7)
    values[1].fill(-3)
    result=evaluate_source_contraction(probe,record,values)
    np.testing.assert_array_equal(result,np.full((2,3),123+5*7*(-3),dtype=np.int32))


def test_overflow_poison_contract_is_not_called_modular():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from xdsl.dialects.arith import IntegerOverflowAttr, IntegerOverflowFlag
    from xdsl.printer import Printer
    import io
    module=parse_mlir_text(source(generic=True))
    mul=next(op for op in module.walk() if op.name == "arith.muli")
    mul.properties["overflowFlags"] = IntegerOverflowAttr([IntegerOverflowFlag.NSW])
    stream=io.StringIO()
    Printer(stream=stream,print_generic_format=True).print_op(module)
    with pytest.raises(ValueError,match="overflow"):
        extract(stream.getvalue())


@pytest.mark.parametrize("mutate",[
    lambda text:text.replace("%p,%acc", "%p,%p"),
    lambda text:text.replace("%x,%y", "%x,%x"),
    lambda text:text.replace("arith.addi", "arith.subi"),
    lambda text:text.replace("(k,n)","(n,k)"),
    lambda text:text.replace('"parallel","parallel","reduction"','"parallel","parallel","parallel"'),
    lambda text:text.replace("%v=arith.addi", "%dead=arith.muli %x,%y:i8\n%v=arith.addi"),
])
def test_noncanonical_source_never_becomes_a_different_probe(mutate):
    with pytest.raises(ValueError):
        extract(mutate(source(generic=True)))


def test_bounds_entry_width_and_receipt_fail_closed():
    with pytest.raises(ValueError):
        extract_source_contraction(source(),0,entry="wrong",max_m=2,max_n=2,max_k=2)
    with pytest.raises(ValueError,match="strictly reduce"):
        extract_source_contraction(source(),0,entry="work",max_m=3,max_n=4,max_k=9)
    with pytest.raises(ValueError):
        extract(source(),max_macs=1)
    probe,record=extract(source())
    values=arrays(record)
    with pytest.raises(ValueError,match="stale"):
        evaluate_source_contraction(probe+"\n",record,values)
    values[0]=values[0].astype(np.uint8)
    with pytest.raises(ValueError,match="typed"):
        evaluate_source_contraction(probe,record,values)


def test_named_mixed_width_unverified_body_is_not_guessed():
    # The current parser's implicit named region lacks extension operations for
    # this spelling. Do not silently invent the missing source semantics.
    with pytest.raises(ValueError,match="verify"):
        extract(source(output="i32"))
