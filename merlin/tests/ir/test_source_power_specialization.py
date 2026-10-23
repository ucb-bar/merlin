"""A literal power becomes a fill or copy only from the live IR, never from a case record."""

import pytest
from mlir_fixtures import forward, generic

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.closed_scalar import evaluate
from merlin.targetgen.source_power_specialization import specialize


def report(*, complex_output=False, overload="aten.pow.Tensor_Scalar", exponent=None):
    # Case metadata deliberately disagrees with or omits the IR: it must not be read.
    return dict(
        cases=[
            dict(
                status="bundled",
                case=dict(
                    overload=overload,
                    arguments=dict(args=dict(items=[dict(kind="tensor", values="DO NOT READ"), exponent])),
                ),
                output_indices=[0],
                output_abi=[dict(dtype="complex<f32>" if complex_output else "f32")],
                semantic_boundary=dict(post_indices=[1]),
            )
        ]
    )


def fixture(tmp_path, body, *, source="f32", output="f32", shape="4", prelude=""):
    t, o = f"tensor<{shape}x{source}>", f"tensor<{shape}x{output}>"
    module = forward(f"%x:{t}", f"({o},{t})", prelude + "\n" + generic([("%x", t)], o, body, yields=None), ret="%r,%x")
    tmp_path.joinpath("model.mlir").write_text(str(module))
    return tmp_path / "model.mlir"


def power(source, output, exponent):
    convert = ""
    base = "%a"
    if source != output:
        op = "arith.uitofp" if source == "i1" else "arith.sitofp"
        convert = f"%c={op} %a : {source} to {output}\n"
        base = "%c"
    return (
        f"{convert}%k=arith.constant {float(exponent)} : {output}\n"
        f"%p=math.powf {base},%k : {output}\nlinalg.yield %p : {output}"
    )


@pytest.mark.parametrize("element", ["f16", "bf16", "f32", "f64", "i8", "i32", "i64", "i1"])
@pytest.mark.parametrize("exponent", [0, 1])
def test_ir_literal_power_selects_a_uniform_fill_or_copy(tmp_path, element, exponent):
    output = element if element.startswith(("f", "b")) else "f32"
    fixture(tmp_path, power(element, output, exponent), source=element, output=output)
    proof = specialize(tmp_path, report())
    assert proof and "no case record" in proof["parameters"]
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    function = next(op for op in module.walk() if op.name == "func.func")
    returned = function.body.block.last_op
    assert returned.operands[1] is function.body.block.args[0]
    if exponent == 0:
        assert returned.operands[0].owner.name == "tensor.splat"
        assert proof["fill_value"] == 1
    else:
        body = returned.operands[0].owner.body.block
        assert body.last_op.operands[0] is body.args[0] or body.last_op.operands[0].owner.operands[0] is body.args[0]


@pytest.mark.parametrize("exponent", [0.5, 2, 3])
def test_boolean_positive_power_is_evaluated_on_both_storage_words(tmp_path, exponent):
    fixture(tmp_path, power("i1", "f32", exponent), source="i1")
    proof = specialize(tmp_path, report())
    assert proof and proof["family"] == "storage_copy_or_declared_conversion"
    assert any(op.name == "arith.uitofp" for op in parse_mlir_file(tmp_path / "model.mlir").walk())


def test_boolean_square_lowered_as_a_product_is_an_identity(tmp_path):
    fixture(
        tmp_path,
        "%c=arith.uitofp %a : i1 to f32\n%m=arith.mulf %c,%c : f32\nlinalg.yield %m : f32",
        source="i1",
    )
    assert specialize(tmp_path, report())["family"] == "storage_copy_or_declared_conversion"


@pytest.mark.parametrize(
    "case_metadata", [dict(overload="aten.add.Tensor"), dict(exponent=2.0), dict(overload="aten.pow.Tensor_Scalar", exponent=1.0)]
)
def test_case_record_neither_enables_nor_disables_the_ir_proof(tmp_path, case_metadata):
    fixture(tmp_path, power("f32", "f32", 1))
    assert specialize(tmp_path, report(**case_metadata))


def test_operation_labelled_as_power_is_not_rewritten_when_the_ir_is_an_addition(tmp_path):
    # Review S2 probe: an addf carrying a power provenance label must stay an addition.
    label = '{"prov.aten"="aten.pow.Tensor_Scalar"}'
    path = fixture(tmp_path, f"%k=arith.constant 1.0 : f32\n%s=arith.addf %a,%k {label} : f32\nlinalg.yield %s : f32")
    assert "prov.aten" in path.read_text()
    before = path.read_bytes()
    assert specialize(tmp_path, report(exponent=1.0)) is None
    assert path.read_bytes() == before


def complex_fixture(tmp_path, real, imaginary, component_map="affine_map<(d0)->(d0)>"):
    """The frontend's complex lowering: project, map each component, interleave by index."""
    t, c, row = "tensor<3xf32>", "tensor<3x2xf32>", "affine_map<(d0)->(d0)>"
    project = [generic([("%x", c)], t, "", yields="%a", name=n, maps=[f"affine_map<(d0)->(d0,{k})>", row]) for k, n in enumerate(("%re", "%im"))]
    parts = [("%re", t), ("%im", t)]
    maps = [component_map] * 2 + [row]
    components = [generic(parts, t, body, yields=None, name=n, maps=maps, args="ri") for body, n in ((real, "%fr"), (imaginary, "%fi"))]
    select = "%k=linalg.index 1 : index\n%z=arith.constant 0 : index\n%t=arith.cmpi eq,%k,%z : index\n%s=arith.select %t,%a,%b : f32"
    interleave = ["affine_map<(d0,d1)->(d0)>"] * 2 + ["affine_map<(d0,d1)->(d0,d1)>"]
    out = generic([("%fr", t), ("%fi", t)], c, select, yields="%s", name="%out", maps=interleave)
    module = forward(f"%x:{c}", f"({c},{c})", "\n".join([*project, *components, out]), ret="%out,%x")
    tmp_path.joinpath("model.mlir").write_text(str(module))


def test_complex_zero_power_fills_exactly_one_plus_positive_zero_i(tmp_path):
    one = "%w=arith.extf %r : f32 to f64\n%c=arith.constant 1.0 : f64\n%n=arith.truncf %c : f64 to f32\nlinalg.yield %n : f32"
    zero = "%c=arith.constant 0.0 : f64\n%n=arith.truncf %c : f64 to f32\nlinalg.yield %n : f32"
    complex_fixture(tmp_path, one, zero)
    proof = specialize(tmp_path, report(complex_output=True))
    assert proof["family"] == "uniform_complex_literal_fill" and proof["fill_value"] == [1.0, 0.0]
    literal = next(op for op in parse_mlir_file(tmp_path / "model.mlir").walk() if op.name == "arith.constant")
    assert list(literal.value.get_values()) == [1.0, 0.0] * 3


def test_complex_unit_power_is_an_exact_component_copy(tmp_path):
    real = "%w=arith.extf %r : f32 to f64\n%n=arith.truncf %w : f64 to f32\nlinalg.yield %n : f32"
    imaginary = "%w=arith.extf %i : f32 to f64\n%n=arith.truncf %w : f64 to f32\nlinalg.yield %n : f32"
    complex_fixture(tmp_path, real, imaginary)
    assert specialize(tmp_path, report(complex_output=True))["family"] == "storage_copy_or_declared_conversion"


# flip on complex storage: each component reads its projection reversed; a swap exchanges them.
@pytest.mark.parametrize(
    "real,imaginary,maps",
    [("%r", "%i", ["affine_map<(d0)->(2-d0)>"]), ("%i", "%r", [])],
    ids=["reversal", "swap"],
)
def test_complex_component_permutation_is_not_a_copy(tmp_path, real, imaginary, maps):
    complex_fixture(tmp_path, f"linalg.yield {real} : f32", f"linalg.yield {imaginary} : f32", *maps)
    assert specialize(tmp_path, report(complex_output=True)) is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64", "complex64"])
@pytest.mark.parametrize("exponent", [0.0, 1.0])
def test_source_dispatch_agrees_on_storage_words_including_signaling_nans(dtype, exponent):
    torch = pytest.importorskip("torch")
    if dtype in {"float16", "bfloat16"}:
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(getattr(torch, dtype))
    elif dtype == "float64":
        words = [
            0,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF0000000000001,
            0x7FF8000000000001,
            0xFFF8000000000001,
            0x3FF0000000000000,
        ]
        values = torch.tensor(words, dtype=torch.uint64).view(torch.float64)
    else:
        words = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F800001, 0x7FC00001, 0xFFC00001, 0x3F800000]
        values = torch.tensor(words, dtype=torch.uint32).view(torch.float32)
        if dtype == "complex64":
            values = torch.view_as_complex(values.reshape(-1, 2))
    actual = torch.ops.aten.pow.Tensor_Scalar(values, exponent)
    expected = torch.ones_like(values) if exponent == 0 else values.clone()
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("predicate,allowed", [("ugt", True), ("ult", False)])
def test_static_assert_guard_uses_unsigned_word_semantics(tmp_path, predicate, allowed):
    path = fixture(
        tmp_path,
        power("f32", "f32", 1),
        prelude=f"""%minus=arith.constant -1 : i8
      %zero=arith.constant 0 : i8
      %guard=arith.cmpi {predicate},%minus,%zero : i8
      cf.assert %guard,"static divisor guard\"""",
    )
    original = path.read_bytes()
    assert bool(specialize(tmp_path, report())) == allowed
    if not allowed:
        assert path.read_bytes() == original


def rewritten_body(tmp_path):
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    function = next(op for op in module.walk() if op.name == "func.func")
    return function.body.block.last_op.operands[0].owner.body.block


def unit_power(chain, output):
    return chain + f"\n%k=arith.constant 1.0 : {output}\n%p=math.powf %f,%k : {output}\nlinalg.yield %p : {output}"


# Review B2: the copy re-emits the program's own conversion chain, so every source word keeps its value.
@pytest.mark.parametrize(
    "source,chain,ops,word,value",
    [
        ("i8", "%f=arith.uitofp %a : i8 to f32", ["arith.uitofp"], -1, 255.0),
        ("i8", "%w=arith.extui %a : i8 to i32\n%f=arith.sitofp %w : i32 to f32", ["arith.extui", "arith.sitofp"], -1, 255.0),
        ("i8", "%f=arith.sitofp %a : i8 to f32", ["arith.sitofp"], -1, -1.0),
        ("i1", "%f=arith.sitofp %a : i1 to f32", ["arith.sitofp"], True, -1.0),
        ("i32", "%f=arith.sitofp %a : i32 to f32", ["arith.sitofp"], 2**24 + 1, float(2**24)),
    ],
)
def test_unit_power_copy_keeps_the_programs_conversion_chain(tmp_path, source, chain, ops, word, value):
    fixture(tmp_path, unit_power(chain, "f32"), source=source)
    assert specialize(tmp_path, report())["family"] == "storage_copy_or_declared_conversion"
    body = rewritten_body(tmp_path)
    assert [op.name for op in body.ops][:-1] == ops
    assert evaluate(body.last_op.operands[0], {body.args[0]: word}) == value


BOOLEAN = dict(source="i1")
UNPROVED = {
    "non_identity": (power("f32", "f32", 2), "", {}),
    "foreign_call": (power("f32", "f32", 1), "func.call @unknown() : () -> ()", {}),
    "unknown_arithmetic": (power("f32", "f32", 1), '%opaque="arith.future_carrier"(%x) : (tensor<4xf32>) -> tensor<4xf32>', {}),
    "runtime_exponent": ("%p=math.powf %a,%a : f32\nlinalg.yield %p : f32", "", {}),
    # A Boolean affine map has a two-entry table: never emitted.
    "boolean_two_entry_table": (
        "%c=arith.uitofp %a : i1 to f32\n%k=arith.constant 3.0 : f32\n%m=arith.addf %c,%k : f32\nlinalg.yield %m : f32",
        "",
        BOOLEAN,
    ),
    # sitofp maps i1 true to -1.0, so the complete Boolean table is (+0, -1): neither a copy nor a fill.
    "boolean_signed_conversion": ("%c=arith.sitofp %a : i1 to f32\nlinalg.yield %c : f32", "", BOOLEAN),
    "chain_rounds_before_widening": (
        unit_power("%n=arith.sitofp %a : i32 to f32\n%f=arith.extf %n : f32 to f64", "f64"),
        "",
        dict(source="i32", output="f64"),
    ),
}


@pytest.mark.parametrize("failure", list(UNPROVED))
def test_incomplete_semantic_binding_is_not_a_source_specialization(tmp_path, failure):
    body, prelude, options = UNPROVED[failure]
    path = fixture(tmp_path, body, prelude=prelude, **options)
    before = path.read_bytes()
    assert specialize(tmp_path, report()) is None
    assert path.read_bytes() == before
    assert not (tmp_path / "native-source-specialization.json").exists()
