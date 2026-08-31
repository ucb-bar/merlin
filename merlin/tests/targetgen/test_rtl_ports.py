"""Module PORT facts, read out of a target's own elaborated FIRRTL.

The fact bundle records interfaces by name plus an evidence string and no port list, so two things the
performance layer needs were unobtainable on a target whose RTL plainly provides them: a per-engine
completion channel, and a top-level busy port. Both matter for the same reason — an unmeasured unit that
reads as IDLE has already produced a wrong headline in this repo (a portless vector unit counted zero
everywhere, and correcting it moved a corpus idle figure from 76.7% to 46.2%).
"""
from __future__ import annotations

import textwrap

from merlin.targetgen.rtl import ports as P

# The real shape, trimmed: a decoupled `completed`, a scalar `busy`, and a nested bundle whose own
# fields must NOT be mistaken for top-level ones.
_FIR = textwrap.dedent("""\
    FIRRTL version 3.3.0
    circuit T :
      module LoadController :
        input clock : Clock
        output io : { flip cmd : { flip ready : UInt<1>, valid : UInt<1>, bits : { rob_id : UInt<6>}}, completed : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<6>}, busy : UInt<1>}
      module Plain :
        output io : { done : UInt<1>}
      module Nested :
        output io : { inner : { completed : { valid : UInt<1>}}}
    """)


class TestStructuralParse:
    def test_a_decoupled_completion_is_found_with_its_leaves(self):
        got = P.parse_ports(_FIR)
        f = got["LoadController"].field_named("completed")
        assert f is not None and f.is_decoupled()
        assert set(f.leaves) == {"ready", "valid", "bits"}

    def test_a_scalar_port_field_is_recorded(self):
        # `busy : UInt<1>` carries no sub-fields; it must still be visible, because a top-level busy
        # port is what makes a unit's occupancy measurable at all.
        f = P.parse_ports(_FIR)["LoadController"].field_named("busy")
        assert f is not None and f.leaves == () and not f.is_decoupled()

    def test_a_nested_field_is_not_mistaken_for_a_top_level_one(self):
        # Splitting a bundle on every comma shreds the nesting and invents fields that do not exist.
        got = P.parse_ports(_FIR)
        assert got["Nested"].field_named("completed") is None
        assert got["Nested"].field_named("inner") is not None

    def test_a_module_with_no_matching_field_is_not_reported(self):
        assert P.modules_exposing(P.parse_ports(_FIR), "completed").keys() == {"LoadController"}

    def test_flip_is_an_orientation_not_part_of_the_name(self):
        f = P.parse_ports(_FIR)["LoadController"].field_named("cmd")
        assert f is not None, "a flipped field must keep its declared name"

    def test_an_empty_circuit_yields_nothing_rather_than_raising(self):
        assert P.parse_ports("") == {}


class TestThreeStates:
    def test_an_unreadable_elaboration_is_unavailable_not_empty(self, tmp_path):
        # "we could not read the RTL" and "the RTL exposes no such port" must never read alike: the
        # first is UNKNOWN, and reporting it as the second is how an extraction gap becomes a claim
        # about the hardware.
        got = P.port_facts("t", fir=tmp_path / "absent.fir")
        assert got["status"] == "unavailable"
        assert got["fields"] == {}
        assert "not the same as" in got["why"]

    def test_a_readable_elaboration_reports_derived_with_its_source(self, tmp_path):
        f = tmp_path / "c.fir"
        f.write_text(_FIR)
        got = P.port_facts("t", fields=("completed", "busy"), fir=f)
        assert got["status"] == "derived" and got["fir"] == str(f)
        assert got["fields"]["completed"]["decoupled"] == ["LoadController"]
        assert "LoadController" in got["fields"]["busy"]["modules"]

    def test_the_field_set_is_the_callers_choice(self, tmp_path):
        # Nothing here knows what a target calls its completion signal. A target that spells it
        # differently is served by asking for a different field, not by editing this module.
        f = tmp_path / "c.fir"
        f.write_text(_FIR)
        assert P.port_facts("t", fields=("done",), fir=f)["fields"]["done"]["modules"] == ["Plain"]
