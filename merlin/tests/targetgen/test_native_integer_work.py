"""Every integer data producer needs its own native-unit binding."""

from mlir_fixtures import forward, generic, pointwise

from merlin.targetgen.native_integer_work import uncovered


def source():
    typ = "tensor<3xi32>"
    total = generic([("%x", typ), ("%y", typ)], typ, "%v=arith.addi %a,%b : i32", name="%sum")
    return forward(f"%x:{typ},%y:{typ}", "tensor<i32>", total + """
      %zero=arith.constant 0 : i32
      %seed=tensor.splat %zero : tensor<i32>
      %reduced=linalg.reduce ins(%sum:tensor<3xi32>) outs(%seed:tensor<i32>) dimensions=[0]
        (%v:i32,%acc:i32) { %added=arith.addi %v,%acc : i32
        linalg.yield %added : i32 }""", ret="%reduced")


def test_both_parallel_fold_and_bias_reduction_need_a_binding():
    module = source()
    ordinals = [i for i, op in enumerate(module.walk()) if op.name in {"linalg.generic", "linalg.reduce"}]
    assert uncovered(module, None) == ordinals
    assert uncovered(module, dict(bindings=[dict(source_operation_ordinal=ordinals[0])])) == ordinals[1:]
    assert uncovered(module, dict(bindings=[dict(source_operation_ordinal=i) for i in ordinals])) == []


def test_float_readout_integer_representation_is_separate_from_source_integer_work():
    body = "%zero=arith.constant 0 : i32\n%v=arith.addi %a,%zero : i32\n%f=arith.sitofp %v : i32 to f32"
    module = pointwise(["tensor<3xi32>"], "tensor<3xf32>", body, yields="%f")
    assert uncovered(module, None) == []
