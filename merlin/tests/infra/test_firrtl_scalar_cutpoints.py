"""Unsigned local cone observations do not infer register history or timing."""

import pytest

from merlin.targetgen.rtl import firrtl_struct as F
from merlin.targetgen.rtl.firrtl_scalar_cutpoints import ScalarCutpoints


def module(text):
    return F.parse_module("Example", list(enumerate(text.splitlines(), 1)))


def test_bundle_mux_projection_preserves_tie_order_and_exact_unsigned_truncation():
    source = module("""module Example :
  input a : UInt<4>
  input b : UInt<4>
  wire p : { size : UInt<5>, moved : UInt<5>}
  wire q : { size : UInt<5>, moved : UInt<5>}
  connect p.size, UInt<5>(16)
  connect p.moved, a
  connect q.size, UInt<5>(24)
  connect q.moved, b
  node selected = mux(gt(q.moved, p.moved), q, p)
  node wrapped = tail(add(a, UInt<4>(1)), 1)
  node joined = cat(a, b)
""")
    observer = ScalarCutpoints(source)
    assert observer.evaluate({"a": 7, "b": 7}, ["selected.size", "selected.moved", "joined"]) == {
        "selected.size": 16,
        "selected.moved": 7,
        "joined": 119,
    }
    assert observer.evaluate({"a": 15, "b": 9}, ["wrapped"])["wrapped"] == 0
    assert observer.evaluate({"a": 2, "b": 9}, ["selected.size"])["selected.size"] == 24


def test_priority_driver_and_explicit_state_cut_cannot_read_next_state_implicitly():
    source = module("""module Example :
  input choose : UInt<1>
  reg state : UInt<4>, clock
  wire result : UInt<3>
  connect state, UInt<4>(9)
  connect result, state
  when choose :
    connect result, UInt<4>(15)
""")
    observer = ScalarCutpoints(source)
    with pytest.raises(ValueError, match="unbound input, state"):
        observer.evaluate({"choose": 0}, ["result"])
    assert observer.evaluate({"choose": 0, "state": 10}, ["result"])["result"] == 2
    assert observer.evaluate({"choose": 1}, ["result"])["result"] == 7


def test_dynamic_vector_selection_is_bounded_and_private_snapshot_is_stable():
    source = module("""module Example :
  input values : UInt<8>[2]
  input index : UInt<2>
  node selected = values[index]
""")
    observer = ScalarCutpoints(source)
    assert observer.evaluate({"values[1]": 71, "index": 1}, ["selected"])["selected"] == 71
    with pytest.raises(ValueError, match="outside.*vector"):
        observer.evaluate({"values[1]": 71, "index": 2}, ["selected"])
    source.nodes["selected"] = (F.Lit("UInt", 8, 0), 5, None)
    assert observer.evaluate({"values[1]": 71, "index": 1}, ["selected"])["selected"] == 71


def test_unsigned_dynamic_shift_retains_firrtl_growth_and_destination_truncation():
    source = module("""module Example :
  input value : UInt<2>
  input index : UInt<3>
  node shifted = dshl(value, index)
  wire narrow : UInt<4>
  connect narrow, shifted
""")
    observer = ScalarCutpoints(source)
    for value in range(4):
        for index in range(8):
            assert observer.evaluate({"value": value, "index": index}, ["shifted", "narrow"]) == {
                "shifted": value << index,
                "narrow": (value << index) & 15,
            }
    assert observer._width(F.Ref("shifted")) == 9


@pytest.mark.parametrize("width", [9, 256])
def test_dynamic_shift_width_growth_refuses_before_large_allocation(width):
    source = module(f"module Example :\n  input index : UInt<{width}>\n  node shifted = dshl(UInt<1>(1), index)\n")
    with pytest.raises(ValueError, match="dynamic shift exceeds"):
        ScalarCutpoints(source).evaluate({"index": 0}, ["shifted"])


@pytest.mark.parametrize(
    "expression", ["shl(a, 99999999999999999)", "bits(a, 9, 0)", "asSInt(a)", "div(a, UInt<4>(0))", "UInt<1>(2)"]
)
def test_unsupported_or_oversized_values_refuse_before_expansion(expression):
    source = module(f"module Example :\n  input a : UInt<4>\n  node result = {expression}\n")
    with pytest.raises(ValueError):
        ScalarCutpoints(source).evaluate({"a": 1}, ["result"])


def test_cycles_unknown_sources_signedness_and_bad_cut_values_refuse():
    source = module("module Example :\n  input a : UInt<4>\n  node result = result\n")
    with pytest.raises(ValueError, match="cyclic"):
        ScalarCutpoints(source).evaluate({"a": 1}, ["result"])
    with pytest.raises(ValueError, match="cut does not fit"):
        ScalarCutpoints(source).evaluate({"a": True}, ["a"])
    signed = module("module Example :\n  input a : SInt<4>\n  node result = asUInt(a)\n")
    with pytest.raises(ValueError, match="unsigned"):
        ScalarCutpoints(signed).evaluate({"a": 1}, ["result"])


def test_original_instance_port_types_and_explicit_depth_admission():
    child = module("module Child :\n  output io : { value : UInt<5>}\n")
    parent = module("module Example :\n  inst selected of Child\n  node result = selected.io.value\n")
    with pytest.raises(ValueError, match="original child port type"):
        ScalarCutpoints(parent).evaluate({"selected.io.value": 17}, ["result"])
    assert (
        ScalarCutpoints(parent, modules={"Child": child}).evaluate({"selected.io.value": 17}, ["result"])["result"]
        == 17
    )
    source = module(
        "module Example :\n  input a : UInt<1>\n"
        + "\n".join(f"  node n{i} = not({'a' if i == 0 else 'n' + str(i - 1)})" for i in range(20))
    )
    with pytest.raises(ValueError, match="oversized"):
        ScalarCutpoints(source, max_depth=8).evaluate({"a": 1}, ["n19"])
    assert ScalarCutpoints(source, max_depth=64).evaluate({"a": 1}, ["n19"])["n19"] == 1


def test_known_masks_prove_results_without_inventing_unobserved_state():
    source = module(
        "module Example :\n  input a : UInt<1>\n  reg state : UInt<1>, clock\n"
        "  node zero = and(state, UInt<1>(0))\n"
        "  node one = or(state, UInt<1>(1))\n"
        "  node unknown = and(state, UInt<1>(1))\n"
    )
    observer = ScalarCutpoints(source)
    assert observer.evaluate({"a": 1}, ["zero", "one"]) == {"zero": 0, "one": 1}
    assert (F.Ref("state"), ()) not in observer.values
    with pytest.raises(ValueError, match="unbound input, state"):
        observer.evaluate({"a": 1}, ["unknown"])


def test_original_instance_input_drivers_follow_direction_and_aggregate_flips():
    child = module("module Child :\n  output io : { flip request : UInt<5>, response : UInt<5>}\n")
    parent = module(
        "module Example :\n  input a : UInt<5>\n  inst selected of Child\n"
        "  connect selected.io.request, a\n  connect selected.io.response, a\n"
    )
    observer = ScalarCutpoints(parent, modules={"Child": child})
    assert observer.evaluate({"a": 17}, ["selected.io.request"])["selected.io.request"] == 17
    # An invalid parent drive cannot stand in for a child's actual output.
    with pytest.raises(ValueError, match="unbound input, state"):
        observer.evaluate({"a": 17}, ["selected.io.response"])


def test_unsigned_reductions_keep_original_width_and_have_one_bit_results():
    source = module(
        "module Example :\n  input a : UInt<4>\n"
        "  node all_bits = andr(a)\n  node any_bits = orr(a)\n  node parity = xorr(a)\n"
    )
    observer = ScalarCutpoints(source)
    assert observer.evaluate({"a": 15}, ["all_bits", "any_bits", "parity"]) == {
        "all_bits": 1,
        "any_bits": 1,
        "parity": 0,
    }
    assert observer.evaluate({"a": 7}, ["all_bits", "any_bits", "parity"]) == {
        "all_bits": 0,
        "any_bits": 1,
        "parity": 1,
    }
