from dataclasses import replace

import pytest

from merlin.compile.scheduling.configuration import ConfigurationAdapter, ConfigurationEffects
from merlin.compile.scheduling.configuration_loop_cleanup import eliminate_counted_configuration
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.compile.scheduling.configuration_motion import promote_configuration_writes, verify_configuration_motion
from merlin.sched.ir import Call, Const, Kernel, Loop


def effects(call):
    if call.instr in ("load_setup", "store_setup"):
        key = "load" if call.instr == "load_setup" else "store"
        return ConfigurationEffects(writes=((key, call.arg("value").value),), ordered=True, pure_write=True)
    if call.instr == "load":
        return ConfigurationEffects(reads=(("load", 1),), ordered=True)
    if call.instr == "store":
        return ConfigurationEffects(reads=(("store", 7),), ordered=True)
    if call.instr == "clobber":
        return ConfigurationEffects(clobbers=("store",), ordered=True)
    if call.instr == "writer":
        return ConfigurationEffects(writes=(("store", 8),), ordered=True)
    if call.instr == "fence":
        return ConfigurationEffects(ordered=True)
    return None


ADAPTER = ConfigurationAdapter(effects, "public-test-state", "explicit-functional-test-effects")
LD = Call("load_setup", (("value", Const(1)),))
ST = Call("store_setup", (("value", Const(7)),))
LOAD = Call("load", ())
STORE = Call("store", ())


def test_real_cleanup_promotion_hoist_composition_on_huge_counted_domain():
    original = Kernel("reduction", (), (Loop("r", 1000000, (
        Loop("c", 128, (LD, LOAD, LD, LOAD, ST, STORE)),
    )),))
    current = original
    proofs = []
    for _ in range(8):
        before = current
        for transform in (eliminate_counted_configuration, promote_configuration_writes, hoist_loop_configuration):
            result = transform(current, ADAPTER)
            current = result.kernel
            proofs.append(result.report)
        if current == before:
            break
    assert current.body[:2] == (LD, ST)
    assert current.body[2] == Loop("r", 1000000, (Loop("c", 128, (LOAD, LOAD, STORE)),))
    assert any(r.get("moved_static_calls", 0) for r in proofs)
    assert len(current.text()) < 1000


@pytest.mark.parametrize("barrier", [STORE, Call("clobber", ()), Call("writer", ()), Call("unknown", ())])
def test_state_read_write_clobber_and_unknown_stop_motion(barrier):
    kernel = Kernel("boundary", (), (barrier, ST))
    assert promote_configuration_writes(kernel, ADAPTER).kernel == kernel
    with pytest.raises(ValueError, match="dependency or unknown"):
        verify_configuration_motion(kernel, replace(kernel, body=(ST, barrier)), ADAPTER)


def test_setter_order_and_loop_boundaries_are_preserved():
    kernel = Kernel("stable", (), (LD, LOAD, ST))
    assert promote_configuration_writes(kernel, ADAPTER).kernel.body == (LD, ST, LOAD)
    loop = Loop("i", 2, (LOAD,))
    blocked = replace(kernel, body=(loop, ST))
    assert promote_configuration_writes(blocked, ADAPTER).kernel == blocked
    with pytest.raises(ValueError, match="dependency or unknown"):
        verify_configuration_motion(blocked, replace(blocked, body=(ST, loop)), ADAPTER)


def test_zero_trip_keeps_existing_checked_schedule_refusal():
    kernel = Kernel("empty", (), (Loop("i", 0, (LOAD, ST, STORE)),))
    # This schedule IR requires positive extents; zero-trip SCF is a separate
    # representation. Do not bypass its existing structure/admission check.
    with pytest.raises(ValueError, match="checked loop IR"):
        promote_configuration_writes(kernel, ADAPTER)
    assert kernel.body == (Loop("i", 0, (LOAD, ST, STORE)),)


def test_crossed_event_or_state_metadata_remains_a_barrier():
    annotated = replace(LOAD, assumes=(("other", Const(1)),))
    kernel = Kernel("annotation", (), (annotated, ST))
    assert promote_configuration_writes(kernel, ADAPTER).kernel == kernel


@pytest.mark.parametrize("body", [(ST,), (ST, LOAD, STORE), (LOAD, LD, ST)])
def test_verifier_refuses_drops_additions_or_data_reordering(body):
    original = Kernel("unchanged", (), (LOAD, ST))
    with pytest.raises(ValueError):
        verify_configuration_motion(original, replace(original, body=body), ADAPTER)
