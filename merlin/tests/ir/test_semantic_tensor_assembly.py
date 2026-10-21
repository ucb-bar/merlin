"""Partial immutable tensors cannot become observable before initialization."""

from itertools import permutations

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def slice_op(source, result, offset=0, width=2):
    return f"""%{result} = "tensor.extract_slice"(%{source}) <{{
      static_offsets = array<i64: 0, {offset}>, static_sizes = array<i64: 2, {width}>,
      static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}}>
      : (tensor<2x4xf32>) -> tensor<2x{width}xf32>"""


def insert_op(source, destination, result, offset=0, width=2):
    return f"""%{result} = "tensor.insert_slice"(%{source}, %{destination}) <{{
      static_offsets = array<i64: 0, {offset}>, static_sizes = array<i64: 2, {width}>,
      static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}}>
      : (tensor<2x{width}xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>"""


def source(reverse=False, second_offset=2, second_width=2, returned="second", dead_offset=0):
    a, b = ("b", "a") if reverse else ("a", "b")
    first_offset = 2 if reverse else 0
    second_offset = 0 if reverse else second_offset
    return f"""module {{
      func.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x{second_width}xf32>) -> tensor<2x4xf32> {{
        %empty = tensor.empty() : tensor<2x4xf32>
        {slice_op("empty", "dead", dead_offset)}
        {insert_op(a, "empty", "first", first_offset)}
        {slice_op("first", "also_dead", 2)}
        {insert_op(b, "first", "second", second_offset, second_width)}
        return %{returned} : tensor<2x4xf32>
      }}
    }}"""


def translate(text):
    return translate_linalg_text(text, entry="forward", target_identity="synthetic")


@pytest.mark.parametrize("reverse", (False, True))
def test_disjoint_full_coverage_becomes_one_bit_preserving_concat(reverse):
    translated = translate(source(reverse))
    graph = translated.request
    a = (0x80000000, 0x7FC01234, 1, 0xFF800000)
    b = (0x3F800000, 0x7F800000, 0, 0x7FA00001)
    (actual,) = evaluate_graph(
        graph, {"arg0": TensorValue(graph.node("arg0").type, a), "arg1": TensorValue(graph.node("arg1").type, b)}
    )
    assert actual.elements == a[:2] + b[:2] + a[2:] + b[2:]
    assert len(graph.nodes) == 3 and not graph.constants
    assert graph.nodes[-1].op == "concat"
    assert graph.nodes[-1].inputs == ("arg0", "arg1")
    assert type(graph).from_record(graph.record()).digest() == graph.digest()


@pytest.mark.parametrize(
    "arguments,reason",
    (
        ({"returned": "first"}, "uninitialized"),
        ({"second_width": 1, "second_offset": 3}, "uninitialized"),
        ({"second_offset": 1}, "overlapping writes"),
        ({"dead_offset": 3}, "exceeds source bounds"),
    ),
)
def test_partial_reads_holes_overlaps_and_invalid_dead_views_are_rejected(arguments, reason):
    with pytest.raises(ValueError, match=reason):
        translate(source(**arguments))


def test_read_of_uninitialized_view_is_rejected_even_if_later_writes_complete_the_tensor():
    text = (
        source()
        .replace("return %second : tensor<2x4xf32>", "return %dead : tensor<2x2xf32>")
        .replace("-> tensor<2x4xf32> {", "-> tensor<2x2xf32> {")
    )
    with pytest.raises(ValueError, match="uninitialized source"):
        translate(text)


def test_fully_overwritten_empty_tensor_is_the_input_value():
    text = """module {
      func.func @forward(%a: tensor<2x4xf32>) -> tensor<2x4xf32> {
        %empty = tensor.empty() : tensor<2x4xf32>
        %result = "tensor.insert_slice"(%a, %empty) <{
          static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 2, 4>,
          static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}>
          : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        return %result : tensor<2x4xf32>
      }
    }"""
    graph = translate(text).request
    assert len(graph.nodes) == 2 and graph.nodes[-1].op == "identity"


def test_unknown_empty_attributes_cannot_enter_the_initialization_proof():
    text = source().replace(
        "tensor.empty() : tensor<2x4xf32>", '"tensor.empty"() {unknown = "effect"} : () -> tensor<2x4xf32>'
    )
    with pytest.raises(ValueError, match="static tensor.empty"):
        translate(text)


def test_tensor_encoding_cannot_be_discarded_by_logical_translation():
    text = source().replace("xf32>", 'xf32, "opaque">')
    with pytest.raises(ValueError, match="explicit logical-layout translation"):
        translate(text)


@pytest.mark.parametrize("order", tuple(permutations(range(4))))
def test_all_tiny_partition_write_orders_agree_with_independent_tensor_construction(order):
    arguments = ", ".join(f"%arg{i}: tensor<2x1xf32>" for i in range(4))
    operations = ["%empty = tensor.empty() : tensor<2x4xf32>"]
    previous = "empty"
    for count, column in enumerate(order):
        operations.append(insert_op(f"arg{column}", previous, f"written{count}", column, 1))
        previous = f"written{count}"
    text = f"module {{ func.func @forward({arguments}) -> tensor<2x4xf32> {{ " + "\n".join(operations)
    text += f"\nreturn %{previous} : tensor<2x4xf32> }} }}"
    graph = translate(text).request
    inputs = {f"arg{i}": TensorValue(graph.node(f"arg{i}").type, (i * 2, i * 2 + 1)) for i in range(4)}
    (result,) = evaluate_graph(graph, inputs)
    assert result.elements == (0, 2, 4, 6, 1, 3, 5, 7)
    assert len(graph.nodes) == 5


def test_completion_does_not_initialize_the_original_empty_value():
    text = source().replace("-> tensor<2x4xf32> {", "-> (tensor<2x4xf32>, tensor<2x4xf32>) {")
    text = text.replace("return %second : tensor<2x4xf32>", "return %second, %empty : tensor<2x4xf32>, tensor<2x4xf32>")
    with pytest.raises(ValueError, match="uninitialized tensor"):
        translate(text)
