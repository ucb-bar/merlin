"""Process-budget dispatch controls only; no native or numerical qualification."""

from contextlib import contextmanager

import pytest

from merlin.targetgen import owned_host_model_execution as owned


@pytest.mark.parametrize("wait_s", [27, 601])
def test_owned_rtl_dispatch_caps_budget_and_counts_queue_wait(monkeypatch, wait_s):
    from merlin.common import execution_deadline
    from merlin.compile import model_execution_inputs
    from merlin.targetgen import rtl_engine_policy

    clock, dispatched = [0.0], []
    monkeypatch.setattr(execution_deadline.time, "monotonic", lambda: clock[0])

    class Backend:
        def execution_timeout_limit(self, simulator):
            assert simulator == "gsim"
            return 600

        def run_elf(self, elf, *, simulator, timeout, capture_bytes):
            dispatched.append(timeout)
            return b"diagnostic transport only"

    facts = {"path": "diagnostic.fir", "config": "diagnostic"}
    monkeypatch.setattr(owned, "_native_engine_for_facts", lambda *args: (Backend(), {}, lambda: None, None))
    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", lambda *args, **kwargs: facts)

    @contextmanager
    def queue(*, wait_timeout_s):
        assert wait_timeout_s == 600
        clock[0] += wait_s
        yield

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", queue)
    kwargs = dict(target="diagnostic", engine="gsim", facts=facts, timeout=900, elf_sha256="unused")
    if wait_s > 600:
        with pytest.raises(TimeoutError, match="wall budget expired"):
            owned._run_selected_rtl("diagnostic.elf", **kwargs)
        assert dispatched == []
    else:
        owned._run_selected_rtl("diagnostic.elf", **kwargs)
        assert dispatched == [600 - wait_s]
