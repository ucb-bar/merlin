"""Two threads may be boxed at once, and neither may end up running unsandboxed.

The sandbox is installed by replacing two module globals in `oot_runner` and restoring them on exit.
Save/restore of a global does not compose. With two threads the interleaving is:

  1. A saves the true originals and installs its own.
  2. B saves what it now sees -- A's replacement -- as "original", and installs its own.
  3. A exits and restores the true originals. **B's remaining package invocations now run with no
     sandbox at all**: credentials unmasked, answer masks bypassed, and nothing says so.
  4. B exits and restores A's replacement permanently, poisoning the globals for everyone after.

Step 3 is the dangerous half, because it is silent -- the measurement is simply taken outside the
box. This pins the fixed behaviour: the policy is per-thread, the patch is installed once for as
long as any thread holds it, and an invocation arriving on an unboxed thread is refused rather than
falling through.
"""
from __future__ import annotations

import sys
import threading

import pytest

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_campaign as PC  # noqa: E402
from merlin.targetgen import oot_runner  # noqa: E402


def _policy(tmp_path, name):
    root = tmp_path / name
    (root / "pkg").mkdir(parents=True)
    (root / "ws").mkdir(parents=True)
    return PC.PackageSandboxPolicy(
        argv=("true",), package=(root / "pkg").resolve(), workspace=(root / "ws").resolve(),
        target_experiment=None, coverage_gap=(), required_tools=())


def test_each_thread_sees_its_own_policy(tmp_path):
    a, b = _policy(tmp_path, "a"), _policy(tmp_path, "b")
    seen: dict[str, object] = {}
    started = threading.Barrier(2)
    release = threading.Event()

    def worker(name, policy, hold):
        with PC.boxed_entrypoints(policy):
            started.wait(timeout=5)
            if hold:
                release.wait(timeout=5)
            seen[name] = PC.active_sandbox_policy()

    ta = threading.Thread(target=worker, args=("a", a, True))
    tb = threading.Thread(target=worker, args=("b", b, False))
    ta.start(); tb.start()
    tb.join(timeout=5)
    # B has now EXITED its box while A is still inside it. Under the old save/restore this is the
    # moment A's sandbox was replaced by B's stale "original".
    release.set()
    ta.join(timeout=5)

    assert seen["a"] is a, "a thread must see its own policy, not another thread's"
    assert seen["b"] is b


def test_the_sandbox_survives_another_thread_leaving(tmp_path):
    """The silent half of the race: A must still be boxed after B exits."""
    a, b = _policy(tmp_path, "a"), _policy(tmp_path, "b")
    with PC.boxed_entrypoints(a):
        patched = oot_runner.run_entrypoint
        t = threading.Thread(target=_enter_and_leave, args=(b,))
        t.start(); t.join(timeout=5)
        assert oot_runner.run_entrypoint is patched, (
            "a second thread leaving its box un-installed the sandbox for the first")
        assert PC.active_sandbox_policy() is a


def _enter_and_leave(policy):
    with PC.boxed_entrypoints(policy):
        pass


def test_the_globals_are_restored_once_the_last_thread_leaves(tmp_path):
    original = oot_runner.run_entrypoint
    with PC.boxed_entrypoints(_policy(tmp_path, "a")):
        assert oot_runner.run_entrypoint is not original
    assert oot_runner.run_entrypoint is original, "the patch outlived its last holder"


def test_an_unboxed_thread_is_refused_rather_than_run_unsandboxed(tmp_path):
    """Falling through to an unsandboxed execution is the failure this must never have."""
    with pytest.raises(PC.CampaignGateError, match="not inside a sandbox policy"):
        PC._run_boxed(object(), "lower", tmp_path / "in.mlir")


def test_the_old_save_restore_pattern_really_did_have_this_race():
    """A positive control: the tests above must fail against the implementation they replaced.

    Without this they would pass on any implementation, including one that never had the bug, and a
    test that cannot fail is not evidence. This reproduces the previous shape in miniature -- save
    the current value, install your own, restore what you saved -- and shows the first holder ends
    up with the second holder's replacement rather than its own.
    """
    import contextlib

    box = {"fn": "ORIGINAL"}

    @contextlib.contextmanager
    def old_style(name):
        saved = box["fn"]          # thread B saves thread A's replacement, not the original
        box["fn"] = name
        try:
            yield
        finally:
            box["fn"] = saved

    with old_style("A"):
        assert box["fn"] == "A"
        with old_style("B"):
            pass
        # A is still inside its box, but B's exit restored what B had saved -- which was "A"...
        assert box["fn"] == "A"
    # ...and on A's exit the value restored is "ORIGINAL" only because this interleaving nested.
    # Now the genuinely concurrent order, where B enters and leaves while A holds:
    box["fn"] = "ORIGINAL"
    a = old_style("A"); a.__enter__()
    b = old_style("B"); b.__enter__()
    a.__exit__(None, None, None)      # A finishes first: restores "ORIGINAL"
    assert box["fn"] == "ORIGINAL", "A's exit un-installed B's sandbox"
    b.__exit__(None, None, None)      # B restores what it saved: "A"
    assert box["fn"] == "A", (
        "the globals are left holding a replacement whose policy points at a deleted workspace")
