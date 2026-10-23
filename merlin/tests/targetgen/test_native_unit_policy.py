"""Native roles are derived from source scalar dataflow, never overload names."""

import pytest
from mlir_fixtures import forward, generic, pointwise

from merlin.targetgen.native_unit_policy import construction_evidence


def source(expression, *, literal=False, reduction=False):
    typ = "tensor<3xf32>"
    ones = "%zero=arith.constant 0.0 : f32\n%one=arith.constant 1.0 : f32\n%ones=tensor.splat %one : tensor<3xf32>\n"
    ins = [("%x", typ), ("%ones" if literal else "%y", typ)]
    body = generic(ins, typ, expression, iterators=["reduction" if reduction else "parallel"])
    return forward(f"%x:{typ},%y:{typ}", typ, ones + body)


MUL = "%v=arith.mulf %a,%b : f32"
RAMP = "%i=linalg.index 0 : index\n%j=arith.index_cast %i : index to i64\n%v=arith.sitofp %j : i64 to f32"
CLAMP = "%z=arith.constant 0.0 : f32\n%capped=arith.maximumf %a,%z : f32\n"
BIAS = "%half=arith.constant 0.5 : f32\n%biased=arith.addf %a,%half : f32\n"
ROUNDING = ["readout_offset_rounding"]
DAG = "\n".join(
    ["%i=linalg.index 0 : index"]
    + [f"%d{i}=arith.addi {f'%d{i - 1}' if i else '%i'},{f'%d{i - 1}' if i else '%i'} : index" for i in range(70)]
    + ["%n=arith.index_cast %d69 : index to i64", "%v=arith.sitofp %n : i64 to f32"]
)
DEEP = "\n".join(
    ["%c0=arith.constant 1.0 : f32"]
    + [f"%c{i}=arith.addf %c{i - 1},%c{i - 1} : f32" for i in range(1, 1200)]
    + ["%v=arith.mulf %a,%c1199 : f32"]
)


@pytest.mark.parametrize(
    "body,options,status,reason",
    [
        pytest.param(MUL, {}, "host", "parallel_runtime_product_construction", id="parallel_runtime_product"),
        pytest.param(MUL, {"literal": True}, "eligible", None, id="literal_scale"),
        pytest.param("%v=arith.addf %a,%b : f32", {}, "eligible", None, id="own_linear_math"),
        pytest.param(MUL, {"reduction": True}, "eligible", None, id="reduction_product"),
        pytest.param("%v=math.floor %a : f32", {}, "host", "nonlinear_rounding_construction", id="rounding"),
        pytest.param(
            "%n=arith.negf %b : f32\n%v=arith.mulf %a,%n : f32", {"literal": True}, "eligible", None, id="literal_expr"
        ),
        pytest.param(RAMP, {}, "host", "index_generated_payload_arithmetic", id="index_ramp"),
        pytest.param(
            "%i=linalg.index 0 : index\n%v=arith.constant 2.0 : f32", {}, "eligible", None, id="index_literal"
        ),
        pytest.param(DAG, {}, "host", None, id="shared_index_dag"),
        pytest.param(
            CLAMP + "%p=arith.cmpf ogt,%capped,%z : f32\n%v=arith.uitofp %p : i1 to f32",
            {},
            "host",
            "saturation_comparators",
            id="saturation_comparator",
        ),
        pytest.param(CLAMP + "%v=arith.addf %capped,%z : f32", {}, "eligible", None, id="plain_clamp"),
        pytest.param(
            "%p=arith.cmpf ogt,%a,%b : f32\n%v=arith.select %p,%a,%b : f32", {}, "eligible", None, id="select"
        ),
        pytest.param(BIAS + "%v=math.floor %biased : f32", {}, "host", "readout_offset_rounding", id="biased_floor"),
        pytest.param(
            BIAS + "%n=arith.fptosi %biased : f32 to i32\n%v=arith.sitofp %n : i32 to f32",
            {"excluded": ROUNDING},
            "host",
            None,
            id="biased_narrowing",
        ),
        pytest.param(
            "%half=arith.constant 0.5 : f32\n%v=arith.addf %a,%half : f32", {}, "eligible", None, id="affine_offset"
        ),
        pytest.param(
            "%n=arith.fptosi %a : f32 to i32\n%v=arith.sitofp %n : i32 to f32", {}, "eligible", None, id="narrowing"
        ),
        pytest.param(
            "%bias=arith.constant 16777216.0 : f32\n%shift=arith.addf %a,%bias : f32\n%v=arith.subf %shift,%bias : f32",
            {"excluded": ROUNDING},
            "host",
            None,
            id="magic_bias",
        ),
        pytest.param(
            "%z=arith.constant 0.0 : f32\n%d=arith.subf %a,%b : f32\n%r=arith.maximumf %d,%z : f32\n"
            "%v=arith.addf %r,%b : f32",
            {"excluded": ["multi_op_identity"]},
            "host",
            None,
            id="rectifier_identity",
        ),
        pytest.param(
            "%z=arith.constant 0.0 : f32\n%v=arith.maximumf %a,%z : f32", {}, "eligible", None, id="native_relu"
        ),
        pytest.param("%v=arith.maximumf %a,%b : f32", {}, "eligible", None, id="maximum"),
        pytest.param(DEEP, {}, "eligible", None, id="deep_shared_payload"),
        pytest.param(
            """%z=arith.constant 0.0 : f32
        %nan=arith.cmpf uno, %a,%a : f32
        %lhs=arith.select %nan,%z,%a : f32
        %rhs=arith.select %nan,%z,%b : f32
        %sum=arith.addf %lhs,%rhs : f32
        %i=arith.fptosi %sum : f32 to i32
        %v=arith.sitofp %i : i32 to f32""",
            {"excluded": ROUNDING},
            "eligible",
            None,
            id="runtime_ieee_selection",
        ),
        pytest.param(
            "%sum=arith.addf %a,%b : f32\n%i=arith.fptosi %sum : f32 to i32\n%v=arith.sitofp %i : i32 to f32",
            {"literal": True, "excluded": ROUNDING},
            "host",
            None,
            id="dense_splat_offset",
        ),
    ],
)
def test_construction_role_follows_scalar_dataflow(body, options, status, reason):
    options = dict(options)
    excluded = {"excluded": options.pop("excluded")} if "excluded" in options else {}
    decision = construction_evidence(source(body, **options), **excluded)
    assert decision["status"] == status
    assert reason is None or decision["evidence"][0]["reason"] == reason


SELECT = """%z=arith.constant 0 : index
%p=arith.cmpi eq,%i,%z : index
%v=scf.if %p -> (f32) {
  %a=tensor.extract %x[%i] : tensor<3xf32>
  scf.yield %a : f32
} else {
  %b=tensor.extract %y[%i] : tensor<3xf32>
  scf.yield %b : f32
}"""
TABLE = "%table=arith.constant dense<[2,1,0]> : tensor<3xindex>\n"
GATHER = "%j=tensor.extract %table[%i] : tensor<3xindex>\n%v=tensor.extract %x[%j] : tensor<3xf32>"


@pytest.mark.parametrize(
    "params,prelude,body",
    [
        pytest.param("%x:tensor<3xf32>,%y:tensor<3xf32>", "", SELECT, id="structured_address_selection"),
        pytest.param("%x:tensor<3xf32>", TABLE, GATHER, id="literal_address_table"),
    ],
)
def test_address_selection_is_native_movement_not_index_payload_math(params, prelude, body):
    body = generic([], "tensor<3xf32>", "%i=linalg.index 0 : index\n" + body)
    assert construction_evidence(forward(params, "tensor<3xf32>", prelude + body))["status"] == "eligible"


def test_exclusions_are_selected_by_the_targets_numeric_contract():
    module = source(MUL)
    assert construction_evidence(module, excluded=[])["status"] == "eligible"
    decision = construction_evidence(module, excluded=["parallel_runtime_product_construction"])
    assert decision["status"] == "host"
    assert decision["classified_constructions"] == decision["evidence"]


def test_precomputed_result_tables_do_not_accelerate_their_construction():
    def table(values):
        return forward("", "tensor<3xf32>", f"%r=arith.constant dense<{values}> : tensor<3xf32>")

    generated = construction_evidence(table("[1.0,3.0,5.0]"))
    assert generated["status"] == "host"
    assert generated["evidence"][0]["reason"] == "nonuniform_literal_result_construction"
    assert construction_evidence(table("2.0"))["status"] == "eligible"
    assert construction_evidence(table("[1.0,3.0,5.0]"), excluded=[])["status"] == "eligible"


def test_literal_uniformity_uses_storage_bits_for_nan_and_zero():
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import BytesAttr, DenseIntOrFPElementsAttr, ModuleOp, TensorType, f32

    def table(words):
        data = DenseIntOrFPElementsAttr(
            TensorType(f32, [len(words)]), BytesAttr(b"".join(w.to_bytes(4, "little") for w in words))
        )
        constant = arith.ConstantOp(data)
        function = func.FuncOp("forward", ([], [constant.result.type]))
        function.body.block.add_ops([constant, func.ReturnOp(constant.result)])
        return ModuleOp([function])

    assert construction_evidence(table([0x7FC01234, 0x7FC01234]))["status"] == "eligible"
    assert construction_evidence(table([0, 0x80000000]))["status"] == "host"
    assert construction_evidence(table([0x7FC01234, 0x7FC05678]))["status"] == "host"
    assert construction_evidence(table([]))["status"] == "eligible"


def boolean_source():
    return pointwise(["tensor<3xi1>"] * 2, "tensor<3xi1>", "%v=arith.ori %a,%b : i1")


def test_boolean_or_is_host_and_is_never_rewritten_as_add_and_truth():
    from merlin.llvmlower.boolean_components import rewrite_boolean_components

    module = boolean_source()
    original = str(module)
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"
    assert rewrite_boolean_components(module) == ({}, 0)
    assert str(module) == original


def test_boolean_add_truth_identity_is_detected_after_rewriting():
    module = boolean_source()
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32
    from xdsl.rewriter import Rewriter

    union = next(op for op in module.walk() if op.name == "arith.ori")
    left, right = (arith.ExtUIOp(v, i32) for v in union.operands)
    add = arith.AddiOp(left.result, right.result)
    zero = arith.ConstantOp(IntegerAttr(0, i32))
    truth = arith.CmpiOp(add.result, zero.result, "ne")
    Rewriter.replace_op(union, [left, right, add, zero, truth], [truth.result])
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"


def test_an_unrelated_reduction_does_not_license_parallel_products_or_index_payloads():
    from xdsl.dialects.builtin import ModuleOp

    reduction = source(MUL, reduction=True)
    for other, reason in [(MUL, "parallel_runtime_product_construction"), (RAMP, "index_generated_payload_arithmetic")]:
        module = ModuleOp([op.clone() for m in [reduction, source(other)] for op in m.body.block.ops])
        assert reason in {e["reason"] for e in construction_evidence(module)["evidence"]}


def test_integer_offset_readout_is_rounding():
    body = "%offset=arith.constant 128 : i64\n%bits=arith.constant 8 : i64\n%biased=arith.addi %a,%offset : i64\n"
    module = forward("%a: i64", "i64", body + "%shifted=arith.shrsi %biased,%bits : i64", ret="%shifted")
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "host"


def test_readout_uses_only_the_current_module_during_prelude_rewriting():
    module = boolean_source()
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    # A replaced generic can retain stale SSA uses after its region is removed.
    # It is absent from the current serialized program and has no readout role.
    detached = generic.clone(value_mapper={generic.inputs[0]: generic.results[0]})
    detached.detach_region(0)
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"


SLICE = "static_offsets=array<i64:0,0>,static_sizes=array<i64:2,2>,static_strides=array<i64:1,1>"
MAP2 = "affine_map<(d0,d1)->(d0,d1)>"
PAR2 = 'iterator_types=["parallel","parallel"]'
EXTRACT = "%i=arith.constant 0:index\n%a=tensor.extract %table[%i,%i]:tensor<2x2xi32>\n"
COLLAPSE = "tensor.collapse_shape %table [[0,1]] : tensor<2x2xi32> into tensor<4xi32>"


@pytest.mark.parametrize(
    "carrier,result_type",
    [
        (
            f"%r=linalg.generic {{indexing_maps=[{MAP2}],{PAR2}}} outs(%table:tensor<2x2xi32>) {{ ^bb0("
            "%old:i32): linalg.yield %old:i32 } -> tensor<2x2xi32>",
            "2x2xi32",
        ),
        (
            f"%r=linalg.generic {{indexing_maps=[{MAP2},{MAP2}],{PAR2}}} ins(%x:tensor<2x2xi32>) "
            "outs(%table:tensor<2x2xi32>) { ^bb0(%unused:i32,%old:i32): linalg.yield %old:i32 } -> tensor<2x2xi32>",
            "2x2xi32",
        ),
        (EXTRACT + "%r=tensor.insert %a into %x[%i,%i]:tensor<2x2xi32>", "2x2xi32"),
        (EXTRACT + "%r=tensor.from_elements %a:tensor<1xi32>", "1xi32"),
        (
            '%r="tensor.concat"(%x,%table) <{dim=0:i64}> : (tensor<2x2xi32>,tensor<2x2xi32>) -> tensor<4x2xi32>',
            "4x2xi32",
        ),
        (
            f'%r="tensor.insert_slice"(%table,%x) <{{{SLICE},operandSegmentSizes=array<i32:1,1,0,0,0>}}> : '
            "(tensor<2x2xi32>,tensor<2x2xi32>) -> tensor<2x2xi32>",
            "2x2xi32",
        ),
        (
            f'%r="tensor.extract_slice"(%table) <{{{SLICE},operandSegmentSizes=array<i32:1,0,0,0>}}> : '
            "(tensor<2x2xi32>) -> tensor<2x2xi32>",
            "2x2xi32",
        ),
        ("%r=tensor.cast %table : tensor<2x2xi32> to tensor<?x2xi32>", "?x2xi32"),
        ("%r=" + COLLAPSE, "4xi32"),
        (
            f"%flat={COLLAPSE}\n"
            "%r=tensor.expand_shape %flat [[0,1]] output_shape [2,2] : tensor<4xi32> into tensor<2x2xi32>",
            "2x2xi32",
        ),
        (
            "%shape=arith.constant dense<[4]> : tensor<1xi64>\n"
            "%r=tensor.reshape %table(%shape) : (tensor<2x2xi32>,tensor<1xi64>) -> tensor<4xi32>",
            "4xi32",
        ),
        (
            "%e=tensor.empty() : tensor<2x2x2xi32>\n"
            "%r=linalg.broadcast ins(%table:tensor<2x2xi32>) outs(%e:tensor<2x2x2xi32>) dimensions=[0]",
            "2x2x2xi32",
        ),
        (
            "%z=arith.constant 0:i32\n%r=tensor.pad %table low[1,1] high[1,1] { ^bb0(%i:index,%j:index): "
            "tensor.yield %z:i32 } : tensor<2x2xi32> to tensor<4x4xi32>",
            "4x4xi32",
        ),
    ],
)
def test_nonuniform_literal_tables_cannot_hide_behind_movement(carrier, result_type):
    table = "%table=arith.constant dense<[[1,2],[3,4]]> : tensor<2x2xi32>\n"
    module = forward("%x:tensor<2x2xi32>", f"tensor<{result_type}>", table + carrier)
    module.verify()
    # prepare() applies this same structural classifier to both source and
    # prepared IR; no source spelling or producer provenance grants an escape.
    for screened in (module, module.clone()):
        proof = construction_evidence(screened)
        assert proof["status"] == "host"
        assert any(e["reason"] == "nonuniform_literal_result_construction" for e in proof["evidence"])


ADVANCE = "%next={} %v,%two : f32\nscf.yield %next : f32"


@pytest.mark.parametrize(
    "seed,step,host",
    [
        pytest.param("%half", "scf.yield %v : f32", True, id="literal_seed"),
        pytest.param("%x", "scf.yield %v : f32", False, id="runtime_seed"),
        pytest.param("%half", ADVANCE.format("arith.addf"), True, id="literal_addf"),
        pytest.param("%half", ADVANCE.format("arith.mulf"), True, id="literal_mulf"),
    ],
)
def test_literal_offset_proof_follows_loop_backedges_and_runtime_seeds(seed, step, host):
    module = forward(
        "%x:f32, %y:f32",
        "i32",
        f"""%lb=arith.constant 0 : index
%ub=arith.constant 2 : index
%step=arith.constant 1 : index
%half=arith.constant 0.5 : f32
%two=arith.constant 2.0 : f32
%carried=scf.for %iv=%lb to %ub step %step iter_args(%v={seed}) -> (f32) {{
{step}
}}
%sum=arith.addf %y,%carried : f32
%r=arith.fptosi %sum : f32 to i32""",
    )
    decision = construction_evidence(module, excluded=["readout_offset_rounding"])
    assert (decision["status"] == "host") is host
