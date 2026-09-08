"""A verilator slot whose holder is dead must not disable L3 forever.

The cross-arm verilator semaphore is a directory of ``slot_N`` files claimed with ``O_CREAT|O_EXCL``
and released by the broker that claimed one. A broker that dies without running its cleanup therefore
leaks its slot permanently, and the acquire loop reads a leaked slot as "busy". With the default two
slots, two such deaths disable verilator L3 for every later run by that user -- silently, because
acquisition just returns None and the agent simply never sees L3 happen.

MEASURED 2026-09-08: both slots in this user's directory were held by pid 2187933, dead since
2026-09-02, with nothing in the codebase that would ever release them. The host's memory monitor
killing a broker is enough to cause it, and it did kill background jobs on this host the same night.
This is the second distinct way this one semaphore has silently switched L3 off; the first was a
squatted /tmp directory owned by another user, which cost two entire rounds of a live run.

Reclaim has to be narrow in ONE direction: stealing a slot from a live verilator would oversubscribe
the exact resource the semaphore exists to bound, which is worse than failing to acquire. So every
uncertain case -- unreadable slot, malformed contents, a pid that exists, a pid owned by another user
-- must leave the slot alone. Those are the cases this file pins alongside the reclaim itself.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


@pytest.fixture()
def broker(tmp_path, monkeypatch):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("simjob_broker", HARNESS / "simjob_broker.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"simjob_broker not importable here: {type(exc).__name__}: {exc}")
    slots = tmp_path / "slots"
    slots.mkdir()
    monkeypatch.setattr(mod, "GLOBAL_VERIL_SLOTS", slots)
    return mod, slots


def _dead_pid() -> int:
    """A pid that does not exist: fork a child, reap it, reuse its number immediately."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover -- child
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_a_slot_held_by_a_dead_holder_is_reclaimed(broker):
    mod, slots = broker
    (slots / "slot_0").write_text(str(_dead_pid()))
    got = mod._veril_acquire(1)
    assert got is not None, (
        "a slot leaked by a dead broker was read as busy; with the default 2 slots this switches "
        "verilator L3 off permanently and silently"
    )
    assert got.read_text().strip() == str(os.getpid())


def test_every_slot_leaked_still_leaves_l3_runnable(broker):
    """The measured state: ALL slots held by one dead pid."""
    mod, slots = broker
    dead = str(_dead_pid())
    for i in range(2):
        (slots / f"slot_{i}").write_text(dead)
    assert mod._veril_acquire(2) is not None


def test_a_live_holder_is_never_robbed(broker):
    """Failing to acquire is safe; oversubscribing verilator is not."""
    mod, slots = broker
    (slots / "slot_0").write_text(str(os.getpid()))   # this very process: certainly alive
    assert mod._veril_acquire(1) is None
    assert (slots / "slot_0").read_text().strip() == str(os.getpid()), "a live slot was stolen"


@pytest.mark.parametrize("contents", ["", "   ", "not-a-pid", "12x", "-1"])
def test_a_slot_we_cannot_read_as_a_pid_is_left_alone(broker, contents):
    mod, slots = broker
    (slots / "slot_0").write_text(contents)
    assert mod._veril_acquire(1) is None
    assert (slots / "slot_0").exists(), "a malformed slot was reclaimed on a guess"


def test_a_free_slot_still_acquires_and_records_this_pid(broker):
    mod, slots = broker
    got = mod._veril_acquire(2)
    assert got is not None and got.name == "slot_0"
    assert got.read_text().strip() == str(os.getpid())


def test_an_unwritable_slot_dir_is_named_not_reported_as_busy(broker):
    """"All busy" and "misconfigured" must not look alike -- the /tmp-squat failure."""
    mod, slots = broker
    os.chmod(slots, 0o500)
    try:
        with pytest.raises(mod.VerilSlotsUnusable):
            mod._veril_acquire(1)
    finally:
        os.chmod(slots, 0o700)
