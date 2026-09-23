"""What this device can and cannot fold into an accumulator, read off its own unroller sources.

ResNet-50's 16 residual adds are 3,782,502 cycles -- 14.8% of the whole 71-group program -- and the
arithmetic in them is ONE addition per element. They cost that because the block output makes a round
trip through DRAM: the producing contraction stores it, the add reads it back beside the skip tensor,
and stores the sum. The obvious saving is to have the skip tensor already IN the accumulator when the
contraction reduces onto it, so the sum is read out once.

On this revision it cannot be expressed, and these tests pin the field that says so rather than the
conclusion. Both loop unrollers hand the accumulator an address through ``cast_to_acc_addr``, whose
``accumulate`` argument decides whether the arriving row is ADDED to the accumulator or REPLACES it.
Read structurally, every DRAM -> accumulator load in both unrollers passes the literal ``false.B``
except one, ``LoopMatmulLdB``, which is the residual add's own second addend and is wired to a
different accumulator base (``resadd_addr_start``) than a contraction's (``ld_d_addr_start`` /
``ex_c_addr_start``). So one accumulator region admits exactly ONE initialising tensor, and every
ResNet-50 residual block spends it on the producing contraction's per-channel bias.

These tests exist so that a Gemmini bump which changes any of those literals FAILS here -- that is the
event which would make the fusion expressible, and it is otherwise invisible.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.common import provenance
from merlin.runtime.backends import base


@pytest.fixture(scope="module")
def ports():
    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_accumulator_ports")


@pytest.fixture(scope="module")
def sources():
    """The two unroller sources of the registered hardware revision, or a skip naming what is missing."""
    checkout = provenance.pin("gemmini_rtl").checkout()
    if checkout is None:
        pytest.skip("gemmini_rtl declares a root env var that is unset; no RTL to read")
    scala = checkout / "src" / "main" / "scala" / "gemmini"
    found = {}
    for name in ("LoopMatmul.scala", "LoopConv.scala"):
        path = scala / name
        if not path.is_file():
            pytest.skip(f"gemmini_rtl checkout has no {name}")
        found[name] = path.read_text(encoding="utf-8")
    return found


def test_every_accumulator_site_is_classified(ports, sources):
    """A site this reader cannot name is reported, never dropped -- a dropped port reads as absent."""
    for name, source in sources.items():
        assert ports.unclassified(source) == (), f"{name} has accumulator sites this reader cannot classify"


def test_the_matmul_unrollers_accumulator_ports(ports, sources):
    table = [(p.module, p.role, p.accumulate) for p in ports.accumulator_ports(sources["LoopMatmul.scala"])]
    assert table == [
        ("LoopMatmulLdA", "load", "false.B"),  # the residual add's FIRST addend: replaces the region
        ("LoopMatmulLdB", "load", "true.B"),  # the residual add's SECOND addend: the one accumulate port
        ("LoopMatmulLdD", "load", "false.B"),  # the contraction's accumulator init (bias or D tensor)
        ("LoopMatmulExecute", "preload", "req.accumulate || k =/= 0.U"),  # the MESH result, not a DRAM operand
        ("LoopMatmulStC", "store", "false.B"),
        ("LoopMatmulStC", "store", "false.B"),
        ("LoopMatmulStCSpad", "store", "false.B"),
    ]


def test_the_conv_unrollers_accumulator_ports(ports, sources):
    table = [(p.module, p.role, p.accumulate) for p in ports.accumulator_ports(sources["LoopConv.scala"])]
    assert table == [
        ("LoopConvLdBias", "load", "false.B"),  # the ONLY DRAM operand the conv sequencer admits
        ("LoopConvExecute", "preload", "true.B"),  # the sequencer always accumulates its own output
        ("LoopConvSt", "store", "false.B"),
        ("LoopConvSt", "store", "false.B"),
    ]


def test_one_accumulate_capable_load_and_it_is_the_residual_adds_own(ports, sources):
    """The property that sets the ceiling: nothing else can ADD a DRAM operand to the accumulator."""
    assert ports.accumulating_loads(sources["LoopMatmul.scala"]) == ("LoopMatmulLdB",)
    assert ports.accumulating_loads(sources["LoopConv.scala"]) == ()


def test_the_conv_sequencers_operand_port_is_a_broadcast(ports, sources):
    """It carries one per-channel vector, not a tensor: the spatial iterators are not in its address.

    The mvin the sequencer issues for its accumulator init walks ``b``, ``orow`` and ``ocol``, but the
    DRAM address it reads from is a function of the output-channel iterator ALONE, and the row stride
    it configures is the literal zero -- so every spatial position re-reads the same vector. A
    full-tensor skip cannot enter a convolution's accumulator through this port under any operand
    values, which is a stronger statement than "the accumulate bit is false".
    """
    source = sources["LoopConv.scala"]
    offset = ports.definition(source, "LoopConvLdBias", "dram_offset")
    assert offset == "och * (acc_w/8).U"
    for iterator in ("orow", "ocol", "b "):
        assert iterator not in offset, f"the bias DRAM address moves with {iterator.strip()}"
    assert ports.definition(source, "LoopConvLdBias", "config_cmd.rs2") == "0.U"
    # ...and it arrives accumulator-wide, so it could not carry an elem_t tensor cheaply either.
    assert ports.definition(source, "LoopConvLdBias", "config_cmd_rs1.shrink") == "0.U"


def test_the_matmul_d_port_would_carry_a_full_tensor(ports, sources):
    """The addressing is not what blocks the fusion -- the accumulate literal is.

    ``LoopMatmulLdD`` walks both output iterators against a caller-supplied stride and selects its
    element width from ``low_d``, so a full ``elem_t`` skip tensor is exactly what this port is shaped
    to carry (a zero stride is what makes it the repeating bias instead). It is single-occupancy that
    refuses the fusion: the port replaces the region, so the bias and the skip cannot both be in it.
    """
    offset = ports.definition(sources["LoopMatmul.scala"], "LoopMatmulLdD", "dram_offset")
    for term in ("req.low_d", "req.dram_stride", "(input_w/8).U", "(acc_w/8).U"):
        assert term in offset
    init = next(p for p in ports.accumulator_ports(sources["LoopMatmul.scala"]) if p.module == "LoopMatmulLdD")
    assert init.accumulate == "false.B"


# --- mutations: each must move a verdict above, or the check above proves nothing ----------------


def _swap(source: str, module: str, old: str, new: str) -> str:
    """``old`` -> ``new``, once, inside ``module``'s class block."""
    start = source.index(f"class {module}(")
    at = source.index(old, start)
    return source[:at] + new + source[at + len(old) :]


def test_an_accumulating_d_port_would_be_reported(ports, sources):
    """The device this repo would want: flip LdD's literal and the fusion becomes expressible."""
    mutated = _swap(
        sources["LoopMatmul.scala"],
        "LoopMatmulLdD",
        "accumulate = false.B, read_full = false.B",
        "accumulate = true.B, read_full = false.B",
    )
    assert ports.accumulating_loads(mutated) == ("LoopMatmulLdB", "LoopMatmulLdD")


def test_a_renamed_command_wire_is_reported_not_dropped(ports, sources):
    """Fail closed. A wire this reader does not know must surface, never leave the table short."""
    mutated = _swap(
        sources["LoopMatmul.scala"], "LoopMatmulLdD", "mvin_cmd_rs2.local_addr := cast", "xyz.local_addr := cast"
    )
    unnamed = ports.unclassified(mutated)
    assert [p.module for p in unnamed] == ["LoopMatmulLdD"]
    assert ports.accumulating_loads(mutated) == ("LoopMatmulLdB",)


def test_a_commented_out_site_is_not_a_port(ports, sources):
    """Comments are stripped before the walk, so disabled code cannot be read as a live port."""
    mutated = _swap(
        sources["LoopMatmul.scala"],
        "LoopMatmulLdB",
        "mvin_cmd_rs2.local_addr := cast_to_acc_addr",
        "// mvin_cmd_rs2.local_addr := cast_to_acc_addr",
    )
    assert ports.accumulating_loads(mutated) == ()


def test_a_positional_accumulate_argument_is_refused(ports, sources):
    """An argument this reader cannot read by name is a refusal, not a guessed default."""
    mutated = _swap(sources["LoopMatmul.scala"], "LoopMatmulLdD", "accumulate = false.B", "false.B")
    with pytest.raises(ports.PortsError) as refusal:
        ports.accumulator_ports(mutated)
    assert "accumulate" in str(refusal.value)


def test_an_ambiguous_wiring_is_refused(ports, sources):
    """``definition`` answers once or refuses; it never picks one of two assignments."""
    with pytest.raises(ports.PortsError):
        ports.definition(sources["LoopConv.scala"], "LoopConvLdBias", "spad_addr")
