"""Whether a residual add's DMA can be overlapped with a neighbouring group's compute.

The companion question to ``test_accumulator_fusion_ceiling``. Fusion asks whether the skip tensor can
meet the contraction's output IN the accumulator; this asks the weaker thing -- whether the skip's
loads can merely be ISSUED EARLY, so the DMA hides under the previous group's mesh work. It matters
because the residual adds are 3,782,502 of ResNet-50's 25,419,657 cycles and are DMA-bound, and for 12
of the 16 joins the skip tensor has sat in DRAM for three intervening groups before it is read.

The mechanism for run-ahead exists and is ordinary: each unroller holds ``concurrent_loops`` = 2
descriptors, and each stage advances to the second as soon as the first has started that stage, so
descriptor N+1's loads normally do overlap descriptor N's compute. It is unavailable at exactly the
boundary that would pay, for two structural reasons read off the sources:

1. ``is_resadd`` is declared at MODULE scope in ``LoopMatmul`` -- not a field of the per-descriptor
   ``LoopMatmulState`` -- and ``LoopMatmulState.reset()`` does not clear it (the assignment is present
   but commented out). Every stage reads it on behalf of whichever descriptor that stage is currently
   serving, so a residual add and a contraction cannot be in the two slots together: among other
   wirings, ``ex.io.req.bits.skip := is_resadd`` would skip the CONTRACTION's entire mesh phase.
2. Both unrollers gate non-loop commands on ``!loop_configured``, and they are chained (the
   convolution sequencer's output is the matmul unroller's input), so a live descriptor in either one
   holds the other's groups at its input -- no software fence required, and none avoidable.

These tests pin those facts so that a revision which per-descriptor-izes the mode, or which changes
the admission rule, fails here -- that is the event which would put this lever back on the table.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.common import provenance
from merlin.runtime.backends import base


@pytest.fixture(scope="module")
def mods():
    base.get_backend("gemmini")  # registers the out-of-tree package
    root = "merlin._oot_backends.gemmini."
    return (
        importlib.import_module(root + "gemmini_loop_concurrency"),
        importlib.import_module(root + "gemmini_accumulator_ports"),
    )


@pytest.fixture(scope="module")
def sources():
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


def test_each_unroller_holds_two_descriptors(mods, sources):
    """The run-ahead budget, and the divisor that cuts the accumulator into per-slot regions."""
    concurrency, _ = mods
    assert concurrency.loop_slots(sources["LoopMatmul.scala"]) == 2
    assert concurrency.loop_slots(sources["LoopConv.scala"]) == 2


def test_is_resadd_is_a_module_mode_not_a_descriptor_field(mods, sources):
    """The property that refuses the overlap: the mode belongs to the unroller, not to a descriptor."""
    concurrency, _ = mods
    source = sources["LoopMatmul.scala"]
    assert concurrency.register_scope(source, "is_resadd") == "LoopMatmul"
    assert concurrency.cleared_by_reset(source, "LoopMatmulState", "is_resadd") is False


def test_the_mode_reinterprets_whichever_descriptor_a_stage_is_serving(mods, sources):
    """Not merely "a mode exists": these are the wirings that would corrupt the neighbouring loop."""
    _, ports = mods
    source = sources["LoopMatmul.scala"]
    # The sharpest one. The execute stage of the loop it is SERVING is skipped outright.
    assert ports.definition(source, "LoopMatmul", "ex.io.req.bits.skip") == "is_resadd"
    # ...and the store extent of the loop the STORE stage is serving is reinterpreted.
    assert ports.definition(source, "LoopMatmul", "stC.io.req.bits.max_k") == (
        "Mux(is_resadd, 1.U, loop_requesting_st.max_k)"
    )
    # The operand base is wired twice -- the normal wiring and a `when(is_resadd)` override. That
    # second assignment IS the hazard, so the single-wiring reader must refuse rather than pick one.
    with pytest.raises(ports.PortsError) as refusal:
        ports.definition(source, "LoopMatmul", "ldA.io.req.bits.addr_start")
    assert "2 times" in str(refusal.value)


def test_a_live_descriptor_holds_the_upstream_command_stream(mods, sources):
    """Both unrollers stop accepting non-loop commands while they hold a descriptor.

    They are chained, so this is why a convolution group and a matmul or residual-add group are
    serialized whatever the emitted C does about fences.
    """
    concurrency, _ = mods
    expected = "Mux(is_loop_cmd, !loop_being_configured.configured, !loop_configured && io.out.ready)"
    assert concurrency.nonloop_admission(sources["LoopMatmul.scala"], "LoopMatmul") == expected
    assert concurrency.nonloop_admission(sources["LoopConv.scala"], "LoopConv") == expected


# --- mutations: each must move a verdict above ----------------------------------------------------


def _swap(source: str, anchor: str, old: str, new: str) -> str:
    at = source.index(old, source.index(anchor))
    return source[:at] + new + source[at + len(old) :]


def test_a_per_descriptor_mode_would_be_reported(mods, sources):
    """The device this lever would need: declare the mode inside the per-descriptor state."""
    concurrency, _ = mods
    source = sources["LoopMatmul.scala"]
    anchor = "  val resadd_addr_start = UInt(log2Up(max_acc_addr).W)"  # only in LoopMatmulState
    moved = source.replace("  val is_resadd = RegInit(false.B)\n", "", 1).replace(
        anchor, f"  val is_resadd = RegInit(false.B)\n{anchor}", 1
    )
    assert concurrency.register_scope(moved, "is_resadd") == "LoopMatmulState"


def test_an_uncommented_clear_would_be_reported(mods, sources):
    """The commented-out clear is the fact; uncommenting it must flip the verdict."""
    concurrency, _ = mods
    mutated = _swap(sources["LoopMatmul.scala"], "def reset()", "//is_resadd := false.B", "is_resadd := false.B")
    assert concurrency.cleared_by_reset(mutated, "LoopMatmulState", "is_resadd") is True


def test_a_second_slot_count_is_refused(mods, sources):
    """A budget read from the wrong declaration is worse than no budget."""
    concurrency, _ = mods
    mutated = sources["LoopMatmul.scala"].replace(
        "  val concurrent_loops = 2", "  val concurrent_loops = 2\n  val concurrent_loops = 4", 1
    )
    with pytest.raises(concurrency.PortsError):
        concurrency.loop_slots(mutated)


def test_a_computed_slot_count_is_refused_not_guessed(mods, sources):
    """An expression this reader does not evaluate is a refusal, never a substituted default."""
    concurrency, _ = mods
    mutated = sources["LoopMatmul.scala"].replace("val concurrent_loops = 2", "val concurrent_loops = acc_banks * 2", 1)
    with pytest.raises(concurrency.PortsError) as refusal:
        concurrency.loop_slots(mutated)
    assert "does not evaluate" in str(refusal.value)
