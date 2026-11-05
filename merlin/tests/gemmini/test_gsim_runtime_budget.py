"""The selected backend may nest beneath a host-held slot without taking a sixth."""

import queue
import threading
from types import SimpleNamespace

import selected_driver

from merlin.runtime.backends import base
from merlin.targetgen import rtl_engine_policy as policy


def test_host_slot_and_backend_run_share_one_permit(monkeypatch, tmp_path):
    # The test runner must explicitly select the gemmini support provider.
    selected_driver.require_support("gemmini")
    backend = base.get_backend("gemmini")
    # This fixture tests reentrant reservation ownership, not unrelated live host processes.
    monkeypatch.setattr(policy, "_native_gsim_census", lambda **_kwargs: policy._NativeCensus(0, ()))
    root = tmp_path / "slots"
    real_slot = policy.gsim_runtime_slot
    monkeypatch.setattr(policy, "gsim_runtime_slot", lambda **kwargs: real_slot(slot_root=root, **kwargs))
    monkeypatch.setattr(backend, "_gsim_argv", lambda elf: ["inert-gsim", str(elf)])
    launches = []
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *args, **kwargs: (
            launches.append((args, kwargs)) or SimpleNamespace(returncode=0, stdout="DONE\n", stderr="")
        ),
    )
    ready = queue.Queue()
    release = threading.Event()
    errors = []

    def hold_another_slot():
        try:
            with real_slot(wait_timeout_s=2, slot_root=root):
                ready.put(True)
                if not release.wait(5):
                    raise RuntimeError("test holder release was never signaled")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=hold_another_slot) for _ in range(4)]
    elf = tmp_path / "inert.elf"
    elf.write_bytes(b"inert test bytes")
    try:
        with real_slot(wait_timeout_s=0, slot_root=root):
            for thread in threads:
                thread.start()
            for _ in threads:
                assert ready.get(timeout=3)
            # All five permits are held. The backend must reuse this thread's
            # host permit, not wait for a sixth or spawn outside the guard.
            assert backend.run_elf(elf, simulator="gsim", timeout=0) == "DONE\n"
            assert len(launches) == 1
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
