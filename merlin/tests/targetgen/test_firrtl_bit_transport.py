"""Exact bit transport observes all drivers without assigning temporal roles."""

import pytest

from merlin.targetgen.rtl import firrtl_struct as F
from merlin.targetgen.rtl.firrtl_bit_transport import BitValue, observe_bit_transport


def module(body):
    return F.parse_module("Transport", list(enumerate(body.splitlines(), 1)))


def source_bits(name, width):
    return tuple((name, bit) for bit in range(width))


def test_signed_extension_and_actual_sink_width():
    source = module("""module Transport :
  output io : { flip original : SInt<3>, copied : SInt<7> }
  connect io.copied, io.original
""")
    assert observe_bit_transport(source, "io.copied", sources=("io.original",)) == {
        BitValue(True, source_bits("io.original", 3) + (("io.original", 2),) * 4)
    }


def test_unsigned_cast_preserves_zeros_instead_of_inventing_signed_extension():
    source = module("""module Transport :
  output io : { flip original : SInt<3>, copied : SInt<7> }
  connect io.copied, asUInt(io.original)
""")
    assert observe_bit_transport(source, "io.copied", sources=("io.original",)) == {
        BitValue(True, source_bits("io.original", 3) + (0,) * 4)
    }


def test_full_index_survives_aggregate_alias():
    source = module("""module Transport :
  output io : { flip original : SInt<5>[2], copied : SInt<5> }
  wire alias : SInt<5>[2]
  connect alias, io.original
  connect io.copied, alias[1]
""")
    assert observe_bit_transport(source, "io.copied", sources=("io.original[1]",)) == {
        BitValue(True, source_bits("io.original[1]", 5))
    }


def test_every_guarded_value_alternative_is_retained():
    source = module("""module Transport :
  output io : { flip original : SInt<5>, copied : SInt<5> }
  when UInt<1>(1) :
    connect io.copied, io.original
  else :
    connect io.copied, SInt<5>(0)
""")
    assert len(observe_bit_transport(source, "io.copied", sources=("io.original",))) == 2


def test_register_hold_does_not_add_unknown_data_but_grants_no_initialization():
    source = module("""module Transport :
  input clock : Clock
  output io : { flip original : SInt<5>, copied : SInt<5> }
  reg saved : SInt<5>, clock
  when UInt<1>(1) :
    connect saved, io.original
  else :
    connect saved, saved
  connect io.copied, saved
""")
    assert observe_bit_transport(source, "io.copied", sources=("io.original",)) == {
        BitValue(True, source_bits("io.original", 5))
    }


@pytest.mark.parametrize(
    "expression", ["bits(io.original,3,0)", "bits(io.original,4,1)", "cat(bits(io.original,0,0),bits(io.original,3,0))"]
)
def test_dropped_or_reordered_bits_do_not_compare_equal_to_transport(expression):
    source = module(f"""module Transport :
  output io : {{ flip original : SInt<5>, copied : SInt<5> }}
  connect io.copied, {expression}
""")
    assert observe_bit_transport(source, "io.copied", sources=("io.original",)) != {
        BitValue(True, source_bits("io.original", 5))
    }


@pytest.mark.parametrize("expression", ["add(io.original,SInt<5>(1))", "bits(io.original,5,0)", "unknown_value"])
def test_arithmetic_out_of_bounds_and_unknown_producers_refuse(expression):
    source = module(f"""module Transport :
  output io : {{ flip original : SInt<5>, copied : SInt<5> }}
  connect io.copied, {expression}
""")
    with pytest.raises(ValueError):
        observe_bit_transport(source, "io.copied", sources=("io.original",))
