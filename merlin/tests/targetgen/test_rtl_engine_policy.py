"""The cert tier is a fidelity; which simulator answers is a cost decision that must be RECORDED.

Binding a tier index to one binary (`L3 = verilator`) hid that decision and put two different fidelities
on the same rung across targets. These pin the policy: equal-fidelity engines are ordered by cost,
Verilator is never chosen while GSIM can run (~23x slower at corpus scale — 45 min vs 115 s per capsule),
and a tier that cannot run fails closed instead of quietly becoming a model tier.
"""

from __future__ import annotations

import errno
import multiprocessing
import os
import select
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from merlin.targetgen import rtl_engine_policy as P

_HOLD_SLOT_CODE = """\
import os
from pathlib import Path
from merlin.targetgen import rtl_engine_policy as P

# Keep this process-lock fixture independent of unrelated live host jobs.
P._native_gsim_census = lambda **_kwargs: P._NativeCensus(0, ())
with P.gsim_runtime_slot(wait_timeout_s=10, slot_root=Path(root)):
    ready.put(os.getpid())
    if not release.poll(30):
        raise RuntimeError("test holder release was never signaled")
    release.recv()
"""


_HOLD_NEUTRAL_NATIVE_CODE = """\
import os
import subprocess
import sys
from pathlib import Path
from merlin.targetgen import rtl_engine_policy as P

# The child is an ordinary sleeping Python process with complete plusargs,
# not an emulator or a user workload. The private fixture slot is held by its
# direct parent, so real /proc and kernel FLOCK evidence can be joined.
P._native_gsim_census = lambda **_kwargs: P._NativeCensus(0, ())
with P.gsim_runtime_slot(wait_timeout_s=10, slot_root=Path(root)):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)",
         "+loadmem=/neutral/input", "+max-cycles=100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ready.put((os.getpid(), child.pid))
        if not release.poll(10):
            raise RuntimeError("neutral child fixture release was never signaled")
        release.recv()
    finally:
        child.terminate()
        child.wait(timeout=5)
"""


@contextmanager
def _held_by_processes(root, count):
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    pipes = [ctx.Pipe(duplex=False) for _ in range(count)]
    # ``exec`` is importable from builtins under pytest's installed importlib mode;
    # a target defined in this copied test file is not importable by spawned children.
    processes = [
        ctx.Process(target=exec, args=(_HOLD_SLOT_CODE, {"root": str(root), "ready": ready, "release": reader}))
        for reader, _ in pipes
    ]
    try:
        for process in processes:
            process.start()
        for reader, _ in pipes:
            reader.close()
        deadline = time.monotonic() + 15
        pids = [ready.get(timeout=max(0.1, deadline - time.monotonic())) for _ in processes]
        assert len(set(pids)) == count
        assert all(process.is_alive() for process in processes)
        yield processes
    finally:
        for process, (_, writer) in zip(processes, pipes, strict=True):
            if process.is_alive():
                writer.send(True)
            writer.close()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        ready.close()


def _stub_native_count(monkeypatch, count, matchable=()):
    monkeypatch.setattr(P, "_native_gsim_census", lambda **_kwargs: P._NativeCensus(count, matchable))


def _fake_proc_member(proc, pid, ppid, starttime, argv):
    member = proc / str(pid)
    member.mkdir()
    uid = os.getuid()
    (member / "status").write_text(
        f"State:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nPPid:\t{ppid}\n"
    )
    fields = ["S", str(ppid), *(["0"] * 17), str(starttime)]
    (member / "stat").write_text(f"{pid} (neutral fixture) {' '.join(fields)}\n")
    (member / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
    return member


def test_gsim_has_five_cross_process_runtime_slots(tmp_path, monkeypatch):
    _stub_native_count(monkeypatch, 0)
    root = tmp_path / "slots"
    assert P.capsule_worker_cap("gsim") == 5
    with _held_by_processes(root, 5) as processes:
        with pytest.raises(TimeoutError, match="five GSim slots"):
            with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                pass
        # An abrupt worker exit releases the kernel lock without a stale PID lease.
        processes[0].terminate()
        processes[0].join(timeout=5)
        assert processes[0].exitcode is not None
        with P.gsim_runtime_slot(wait_timeout_s=1, slot_root=root):
            pass
    with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
        pass


def test_nested_thread_reuses_one_slot_but_unrelated_thread_does_not(tmp_path, monkeypatch):
    _stub_native_count(monkeypatch, 0)
    root = tmp_path / "slots"
    with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
            with _held_by_processes(root, 4):
                observed = []

                def probe():
                    try:
                        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                            observed.append("entered")
                    except TimeoutError:
                        observed.append("busy")

                thread = threading.Thread(target=probe)
                thread.start()
                thread.join(timeout=5)
                assert not thread.is_alive() and observed == ["busy"]
            observed.clear()
            thread = threading.Thread(target=probe)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive() and observed == ["entered"]


def test_fork_child_cannot_borrow_parents_slot(tmp_path, monkeypatch):
    _stub_native_count(monkeypatch, 0)
    root = tmp_path / "slots"
    parent_slot = P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root)
    parent_slot.__enter__()
    try:
        with _held_by_processes(root, 4):
            reader, writer = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(reader)
                try:
                    # An inherited context's cleanup must not unlock the
                    # parent's open file description in the forked child.
                    parent_slot.__exit__(None, None, None)
                    try:
                        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                            answer = b"entered"
                    except TimeoutError:
                        answer = b"busy"
                    os.write(writer, answer)
                finally:
                    os._exit(0)
            os.close(writer)
            reaped = False
            try:
                readable, _, _ = select.select([reader], [], [], 5)
                assert readable, "fork child did not report within five seconds"
                assert os.read(reader, 16) == b"busy"
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    waited, status = os.waitpid(pid, os.WNOHANG)
                    if waited == pid:
                        reaped = True
                        break
                    time.sleep(0.01)
                else:
                    pytest.fail("fork child did not exit within five seconds")
                assert status == 0
            finally:
                os.close(reader)
                if not reaped:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass
    finally:
        parent_slot.__exit__(None, None, None)


def test_gsim_slot_releases_on_exception(tmp_path, monkeypatch):
    _stub_native_count(monkeypatch, 0)
    root = tmp_path / "slots"
    with pytest.raises(RuntimeError, match="fixture refusal"):
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
            raise RuntimeError("fixture refusal")
    with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
        pass


def test_external_native_census_and_pending_reservation_share_five_slots(tmp_path, monkeypatch):
    root = tmp_path / "slots"
    _stub_native_count(monkeypatch, 4)
    with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
        outcome = []

        def reserve_other_thread():
            try:
                with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                    outcome.append("entered")
            except TimeoutError:
                outcome.append("busy")

        thread = threading.Thread(target=reserve_other_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and outcome == ["busy"]
    with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
        pass


def test_five_external_native_processes_refuse_any_new_slot(tmp_path, monkeypatch):
    _stub_native_count(monkeypatch, 5)
    with pytest.raises(TimeoutError, match="five GSim slots"):
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=tmp_path / "slots"):
            pass


def test_external_census_and_legacy_file_lock_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / "slots"
    _stub_native_count(monkeypatch, 4)
    with _held_by_processes(root, 1):
        with pytest.raises(TimeoutError, match="five GSim slots"):
            with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                pass


def test_verified_native_children_do_not_double_count_held_slots(tmp_path, monkeypatch):
    root = tmp_path / "slots"
    # Two kernel-held slots already own two of the three counted natives. The
    # third native is uncoordinated: 3 + (2 held + our pending) - 2 = 4.
    _stub_native_count(monkeypatch, 3, (P._ProcessIdentity(1, 2, 3), P._ProcessIdentity(4, 5, 6)))
    monkeypatch.setattr(P, "_verified_native_slot_overlap", lambda *_args, **_kwargs: 2, raising=False)
    with _held_by_processes(root, 2):
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
            pass


def test_real_proc_child_and_kernel_owner_admit_only_one_overlap(tmp_path, monkeypatch):
    # One short-lived neutral Python plusarg process is added to the host; it
    # never executes an emulator. The extra count is synthetic so this test
    # cannot consume more than one actual same-user native-like process.
    assert P._native_gsim_count() <= P.CAPSULE_WORKER_CAP["gsim"] - 1
    root = tmp_path / "slots"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    reader, writer = ctx.Pipe(duplex=False)
    owner = ctx.Process(
        target=exec,
        args=(_HOLD_NEUTRAL_NATIVE_CODE, {"root": str(root), "ready": ready, "release": reader}),
    )
    try:
        owner.start()
        reader.close()
        owner_pid, child_pid = ready.get(timeout=10)
        assert owner.pid == owner_pid and owner.is_alive()
        observed = None
        for _ in range(40):
            candidates = [native for native in P._native_gsim_census().matchable if native.pid == child_pid]
            if candidates:
                observed = candidates[0]
                break
            time.sleep(0.05)
        assert observed is not None, "neutral child never reached a complete real /proc census"
        assert observed.ppid == owner_pid
        monkeypatch.setattr(P, "_native_gsim_census", lambda **_kwargs: P._NativeCensus(4, (observed,)))
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
            pass  # 4 counted natives + held + pending - one real overlap = 5.
    finally:
        if owner.is_alive():
            writer.send(True)
        writer.close()
        if owner.pid is not None:
            owner.join(timeout=5)
            if owner.is_alive():
                owner.terminate()
                owner.join(timeout=5)
        ready.close()


def test_kernel_owner_fd_and_direct_children_close_cross_process_overlap(tmp_path, monkeypatch):
    root = tmp_path / "slots"
    proc = tmp_path / "proc"
    proc.mkdir()
    with _held_by_processes(root, 2):
        slots = P._locked_slot_census(root, own_index=4)
        assert slots.count == 3 and len(slots.held) == 2
        owners = P._lock_owners(Path("/proc/locks").read_bytes(), slots.held)
        assert len(owners) == 2 and len(set(owners.values())) == 2
        children = []
        for number, (slot, pid) in enumerate(owners.items()):
            owner = _fake_proc_member(proc, pid, 1, 100 + number, [b"/neutral/owner"])
            (owner / "fd").mkdir()
            (owner / "fd" / "7").symlink_to(root / f"slot_{slot.index}.lock")
            child = _fake_proc_member(
                proc,
                pid + 1_000_000,
                pid,
                200 + number,
                [b"/neutral/native", b"+loadmem=/neutral/program", b"+max-cycles=100"],
            )
            children.append(child)
        for number in range(2):
            _fake_proc_member(
                proc,
                4_000_000 + number,
                42,
                300 + number,
                [b"/neutral/uncoordinated", b"+loadmem=/neutral/program", b"+max-cycles=100"],
            )
        census = P._native_gsim_census(proc_root=proc)
        assert census.count == 4 and len(census.matchable) == 4
        assert P._verified_native_slot_overlap(root, slots, census, proc_root=proc) == 2
        unknown = P._HeldSlot(3, slots.held[0].device, 987_654_321)
        partial = P._SlotCensus(slots.count + 1, (*slots.held, unknown))
        assert P._verified_native_slot_overlap(root, partial, census, proc_root=proc) == 2
        with P._admission_mutex(root, None):
            assert P._admitted_load(root, own_index=4, proc_root=proc) == 5

        # A grandchild is not the kernel lock owner's directly launched native.
        child = children[0]
        child_pid = int(child.name)
        original_stat = (child / "stat").read_bytes()
        original_status = (child / "status").read_bytes()
        altered_fields = ["S", "999", *(["0"] * 17), "200"]
        (child / "stat").write_text(f"{child_pid} (neutral fixture) {' '.join(altered_fields)}\n")
        (child / "status").write_text(
            f"State:\tS (sleeping)\nUid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n"
            "PPid:\t999\n"
        )
        assert P._admitted_load(root, own_index=4, proc_root=proc) == 6

        (child / "stat").write_bytes(original_stat)
        (child / "status").write_bytes(original_status)
        # A reused PID between the initial census and overlap recheck cannot
        # turn an old native observation into a discount for a new process.
        initial = P._native_gsim_census(proc_root=proc)
        (child / "stat").write_bytes(original_stat.replace(b" 200\n", b" 900\n"))
        assert P._verified_native_slot_overlap(root, slots, initial, proc_root=proc) == 1
        (child / "stat").write_bytes(original_stat)

        # An incomplete executable argv remains in the conservative count but
        # has no owner/child overlap proof.
        (child / "cmdline").write_bytes(b"")
        assert P._admitted_load(root, own_index=4, proc_root=proc) == 6
        (child / "cmdline").write_bytes(
            b"/neutral/native\0+loadmem=/neutral/program\0+max-cycles=100\0"
        )
        original_identity = P._fresh_identity
        altered_owner = str(owners[slots.held[0]])
        owner_reads = 0

        def reused_owner(member, *, uid, native):
            nonlocal owner_reads
            identity = original_identity(member, uid=uid, native=native)
            if member.name == altered_owner and not native:
                owner_reads += 1
                if owner_reads == 2 and identity is not None:
                    return P._ProcessIdentity(identity.pid, identity.ppid, identity.starttime + 1)
            return identity

        with monkeypatch.context() as patch:
            patch.setattr(P, "_fresh_identity", reused_owner)
            assert P._verified_native_slot_overlap(root, slots, census, proc_root=proc) == 1
        owner = proc / str(owners[slots.held[0]])
        (owner / "fd" / "7").unlink()
        assert P._admitted_load(root, own_index=4, proc_root=proc) == 6


def test_ambiguous_kernel_lock_owner_cannot_discount(tmp_path):
    slot = P._HeldSlot(0, os.stat(tmp_path).st_dev, 1234)
    key = f"{os.major(slot.device):02x}:{os.minor(slot.device):02x}:{slot.inode}"
    line = f"1: FLOCK ADVISORY WRITE 123 {key} 0 EOF\n".encode()
    assert P._lock_owners(line, (slot,)) == {slot: 123}
    assert P._lock_owners(line + line, (slot,)) == {}
    other = P._HeldSlot(1, slot.device, 5678)
    other_key = f"{os.major(other.device):02x}:{os.minor(other.device):02x}:{other.inode}"
    other_line = f"2: FLOCK ADVISORY WRITE 456 {other_key} 0 EOF\n".encode()
    assert P._lock_owners(line + line + other_line, (slot, other)) == {other: 456}


def test_one_owner_of_multiple_slots_has_no_unproven_child_pairing(tmp_path, monkeypatch):
    slots = P._SlotCensus(
        3,
        (P._HeldSlot(0, os.stat(tmp_path).st_dev, 11), P._HeldSlot(1, os.stat(tmp_path).st_dev, 12)),
    )
    children = (
        P._ProcessIdentity(201, 100, 20, (b"/native", b"+loadmem=/a", b"+max-cycles=1")),
        P._ProcessIdentity(202, 100, 21, (b"/native", b"+loadmem=/b", b"+max-cycles=1")),
    )
    monkeypatch.setattr(P, "_lock_owners", lambda *_args: {slot: 100 for slot in slots.held})
    lock_file = tmp_path / "locks"
    lock_file.write_bytes(b"")
    assert P._verified_native_slot_overlap(tmp_path, slots, P._NativeCensus(2, children), locks_path=lock_file) == 0


def test_verified_overlap_still_counts_pending_and_uncoordinated_natives(tmp_path, monkeypatch):
    root = tmp_path / "slots"
    _stub_native_count(monkeypatch, 5, (P._ProcessIdentity(1, 2, 3), P._ProcessIdentity(4, 5, 6)))
    monkeypatch.setattr(P, "_verified_native_slot_overlap", lambda *_args, **_kwargs: 2)
    with _held_by_processes(root, 2):
        with pytest.raises(TimeoutError, match="five GSim slots"):
            with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                pass


def test_parallel_reservations_cannot_outpace_native_start(tmp_path, monkeypatch):
    root = tmp_path / "slots"
    _stub_native_count(monkeypatch, 4)
    start = threading.Barrier(4)
    release = threading.Event()
    outcomes = []

    def reserve():
        start.wait(timeout=5)
        try:
            with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=root):
                outcomes.append("entered")
                release.wait(timeout=5)
        except TimeoutError:
            outcomes.append("busy")

    threads = [threading.Thread(target=reserve) for _ in range(3)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    deadline = time.monotonic() + 5
    while len(outcomes) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ["busy", "busy", "entered"]


def test_unreadable_native_census_refuses_admission(tmp_path, monkeypatch):
    def unavailable(**_kwargs):
        raise RuntimeError("native process evidence unreadable")

    monkeypatch.setattr(P, "_native_gsim_census", unavailable)
    with pytest.raises(RuntimeError, match="evidence unreadable"):
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=tmp_path / "slots"):
            pass


def test_proc_census_checks_real_uid_before_native_argv(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()

    def record(pid, uid, argv=None):
        member = proc / str(pid)
        member.mkdir()
        (member / "status").write_text(f"Name:\tfixture\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        if argv is not None:
            (member / "cmdline").write_bytes(b"\0".join(argv) + b"\0")

    record(101, os.getuid(), [b"/any/native", b"/some/elf", b"+loadmem=/some/elf", b"+max-cycles=10"])
    record(102, os.getuid(), [b"/other", b"--ordinary=10"])
    record(103, os.getuid() + 1)  # A foreign argv must never be read.
    assert P._native_gsim_count(proc_root=proc) == 1


def test_proc_census_refuses_unreadable_same_user_evidence(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    member = proc / "101"
    member.mkdir()
    (member / "status").write_text(f"Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n")
    original_read_bytes = Path.read_bytes

    def unreadable(path):
        if path == member / "cmdline":
            raise PermissionError("fixture cmdline denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(RuntimeError, match="cannot verify native process"):
        P._native_gsim_count(proc_root=proc)


@pytest.mark.parametrize(
    ("terminal_status", "expected_count", "expected_error", "expected_reads"),
    [
        (b"State:\tZ (zombie)\n", 0, None, 2),
        ("gone", 0, None, 2),
        ("live", 1, None, 6),
        (b"State:\tZ (zombie)\nState:\tS (sleeping)\n", 0, "malformed State evidence", 2),
        (PermissionError("status denied"), 0, "status denied", 2),
    ],
)
def test_proc_census_rechecks_empty_argv_after_status_transition(
    tmp_path, monkeypatch, terminal_status, expected_count, expected_error, expected_reads
):
    proc = tmp_path / "proc"
    member = proc / "101"
    member.mkdir(parents=True)
    uid = os.getuid()
    live_status = f"State:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n".encode()
    reads = 0

    def transition(path):
        nonlocal reads
        if path == member / "status":
            reads += 1
            if reads == 1:
                return live_status
            if terminal_status == "gone":
                member.rmdir()
                raise FileNotFoundError(path)
            if terminal_status == "live":
                return live_status
            if isinstance(terminal_status, OSError):
                raise terminal_status
            return terminal_status
        if path == member / "cmdline":
            return b""
        raise AssertionError(f"unexpected process evidence: {path}")

    pauses = []
    monkeypatch.setattr(Path, "read_bytes", transition)
    monkeypatch.setattr(P.time, "sleep", pauses.append)
    if expected_error is None:
        assert P._native_gsim_count(proc_root=proc) == expected_count
    else:
        with pytest.raises(RuntimeError, match=expected_error):
            P._native_gsim_count(proc_root=proc)
    assert reads == expected_reads
    if terminal_status == "live":
        assert pauses == [0.01] * 4


@pytest.mark.parametrize(
    ("later_status", "later_argv", "expected_count", "expected_error"),
    [
        (b"State:\tZ (zombie)\n", b"", 0, None),
        (b"State:\tS (sleeping)\n", b"/native\0+loadmem=/model.elf\0+max-cycles=10\0", 1, None),
        (b"State:\tS (sleeping)\n", b"/native\0+loadmem=/model.elf", 0, "incomplete argv"),
    ],
)
def test_proc_census_settles_a_second_live_empty_argv(
    tmp_path, monkeypatch, later_status, later_argv, expected_count, expected_error
):
    proc = tmp_path / "proc"
    member = proc / "101"
    member.mkdir(parents=True)
    uid = os.getuid()
    live_status = f"State:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n".encode()
    settled_status = live_status if later_status == b"State:\tS (sleeping)\n" else later_status
    statuses = iter((live_status, live_status, settled_status))
    argvs = iter((b"", b"", later_argv))

    def transition(path):
        if path == member / "status":
            return next(statuses)
        if path == member / "cmdline":
            return next(argvs)
        raise AssertionError(f"unexpected process evidence: {path}")

    monkeypatch.setattr(Path, "read_bytes", transition)
    monkeypatch.setattr(P.time, "sleep", lambda _: None)
    if expected_error is None:
        assert P._native_gsim_count(proc_root=proc) == expected_count
    else:
        with pytest.raises(RuntimeError, match=expected_error):
            P._native_gsim_count(proc_root=proc)


@pytest.mark.parametrize("read_edge", ("status", "cmdline"))
@pytest.mark.parametrize("member_state", ("gone", "present", "unreadable"))
def test_proc_census_esrch_requires_confirmed_exit(tmp_path, monkeypatch, read_edge, member_state):
    proc = tmp_path / "proc"
    member = proc / "101"
    member.mkdir(parents=True)
    uid = os.getuid()
    live_status = f"State:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n".encode()
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path == member and member_state == "unreadable":
            raise PermissionError("fixture pid stat denied")
        return original_stat(path, *args, **kwargs)

    def evidence(path):
        if path == member / "status" and read_edge == "cmdline":
            return live_status
        if path == member / read_edge:
            if member_state == "gone":
                member.rmdir()
            raise ProcessLookupError(errno.ESRCH, "fixture process exited")
        raise AssertionError(f"unexpected process evidence: {path}")

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "read_bytes", evidence)
    if member_state == "gone":
        assert P._native_gsim_count(proc_root=proc) == 0
    else:
        match = "stat denied" if member_state == "unreadable" else "cannot verify native process"
        with pytest.raises(RuntimeError, match=match):
            P._native_gsim_count(proc_root=proc)


@pytest.mark.parametrize(
    ("known", "potential", "admitted"),
    [(3, 1, True), (4, 1, False), (0, 5, False)],
)
def test_verified_live_empty_argv_consumes_native_capacity(tmp_path, monkeypatch, known, potential, admitted):
    proc = tmp_path / "proc"
    proc.mkdir()
    uid = os.getuid()
    for number in range(known + potential):
        member = proc / str(number + 101)
        member.mkdir()
        (member / "status").write_text(
            f"State:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
        )
        argv = b"/native\0+loadmem=/model.elf\0+max-cycles=10\0" if number < known else b""
        (member / "cmdline").write_bytes(argv)
    monkeypatch.setattr(P.time, "sleep", lambda _delay: None)
    count = P._native_gsim_count(proc_root=proc)
    assert count == known + potential
    census = P._native_gsim_census
    monkeypatch.setattr(P, "_native_gsim_census", lambda **_kwargs: census(proc_root=proc))
    if admitted:
        with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=tmp_path / "slots"):
            pass
    else:
        with pytest.raises(TimeoutError, match="five GSim slots"):
            with P.gsim_runtime_slot(wait_timeout_s=0, slot_root=tmp_path / "slots"):
                pass


@pytest.mark.parametrize("final_uid", ("missing", "malformed", "foreign"))
def test_proc_census_refuses_bad_final_uid_for_live_empty_argv(tmp_path, monkeypatch, final_uid):
    proc = tmp_path / "proc"
    member = proc / "101"
    member.mkdir(parents=True)
    uid = os.getuid()
    initial = f"State:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n".encode()
    final = b"State:\tS (sleeping)\n"
    if final_uid == "malformed":
        final += b"Uid:\twrong\n"
    elif final_uid == "foreign":
        foreign_uid = uid + 1
        final += f"Uid:\t{foreign_uid}\t{foreign_uid}\t{foreign_uid}\t{foreign_uid}\n".encode()
    reads = 0

    def evidence(path):
        nonlocal reads
        if path == member / "status":
            reads += 1
            return initial if reads == 1 else final
        if path == member / "cmdline":
            return b""
        raise AssertionError(f"unexpected process evidence: {path}")

    monkeypatch.setattr(Path, "read_bytes", evidence)
    monkeypatch.setattr(P.time, "sleep", lambda _delay: None)
    failure = "changed Uid evidence" if final_uid == "foreign" else "malformed Uid evidence"
    with pytest.raises(RuntimeError, match=failure):
        P._native_gsim_count(proc_root=proc)


_UP = lambda why="ok": lambda: (True, why)  # noqa: E731 - table-style probes read better inline
_DOWN = lambda why: lambda: (False, why)  # noqa: E731


def test_gsim_is_preferred_over_verilator():
    """The whole point: Verilator must not be picked when GSIM can run."""
    sel = P.select("t", {"verilator": _UP("built"), "gsim": _UP("emu ready")})
    assert sel["engine"] == "gsim"


def test_vcs_wins_when_it_can_actually_run():
    sel = P.select("t", {"verilator": _UP(), "gsim": _UP(), "vcs": _UP("license free")})
    assert sel["engine"] == "vcs" and sel["passed_over"] == []


def test_an_unavailable_higher_priority_engine_is_skipped_with_its_reason():
    sel = P.select("t", {"vcs": _DOWN("no license"), "gsim": _UP("emu ready"), "verilator": _UP()})
    assert sel["engine"] == "gsim"
    assert sel["passed_over"] == ["vcs"]
    assert [c["reason"] for c in sel["considered"] if c["engine"] == "vcs"] == ["no license"]


def test_verilator_is_still_used_when_it_is_the_only_engine():
    """A target with no GSIM adapter yet must keep its cert tier, not lose it to a preference."""
    sel = P.select("atlas", {"verilator": _UP("vsim registered")})
    assert sel["engine"] == "verilator" and sel["fidelity"] == P.ELABORATED_RTL


def test_no_engine_fails_closed_rather_than_downgrading():
    with pytest.raises(P.NoEngineAvailable) as e:
        P.select("t", {"gsim": _DOWN("no adapter"), "verilator": _DOWN("no vsim")})
    assert "no adapter" in str(e.value) and "no vsim" in str(e.value)


def test_every_engine_reports_the_same_fidelity():
    """They all run the elaborated design; the tier must not grade one as weaker than another."""
    for name in P.ENGINE_PRIORITY:
        assert P.select("t", {name: _UP()})["fidelity"] == P.ELABORATED_RTL


def test_a_broken_probe_is_unavailable_not_a_crash():
    def boom():
        raise OSError("toolchain missing")

    sel = P.select("t", {"vcs": boom, "verilator": _UP()})
    assert sel["engine"] == "verilator"
    assert "OSError" in [c["reason"] for c in sel["considered"] if c["engine"] == "vcs"][0]


def test_lower_priority_probes_are_not_paid_once_one_is_available():
    """Probing an absent VCS license or building a Verilator model is not free."""
    called = []
    P.select(
        "t",
        {
            "vcs": _UP("license"),
            "gsim": lambda: called.append("gsim") or (True, "x"),
            "verilator": lambda: called.append("vl") or (True, "x"),
        },
    )
    assert called == []


def test_an_engine_the_policy_has_not_ranked_is_used_not_dropped():
    """A newly registered engine must still be selectable before anyone declares its priority."""
    sel = P.select("t", {"newsim": _UP("registered")})
    assert sel["engine"] == "newsim"


def test_an_engine_that_reports_available_with_no_reason_is_refused():
    """Peer review point, and the shape of several defects hit the same day: a tier that resolved to a
    different engine than the capsule asked for, with the reason living only in a log line, produces
    correct-looking numbers that cannot be audited. Absent reason must be a hard failure, not a default."""
    with pytest.raises(P.UnrecordedSelection):
        P.select("t", {"gsim": lambda: (True, "   ")})


def test_the_reason_survives_on_the_result_not_just_in_a_log():
    sel = P.select("t", {"vcs": _DOWN("no license"), "gsim": _UP("emu built at <path>")})
    assert sel["reason"] == "emu built at <path>"
    assert {c["engine"]: c["reason"] for c in sel["considered"]}["vcs"] == "no license"


# ---------------------------------------------------------------------------------------------
# The lineage gate must reach the SELECTION path, not only the module that owns the home layout.
# ---------------------------------------------------------------------------------------------


def test_a_refused_lineage_loses_the_selection_not_just_the_probe(tmp_path, monkeypatch):
    """Measured 2026-09-04: with MERLIN_GSIM_REQUIRE_RECEIPT=1, a target whose engine carried only an
    adoption record had gsim_emulator.probe() answer False and STILL certified on it, because the
    selection path asked only whether the wrapper file existed. A provenance gate the selection routes
    around is not a gate."""
    from merlin.targetgen import gsim_emulator as GE
    from merlin.targetgen import program_oracle as PO

    home = tmp_path / "gsim"
    home.mkdir()
    (home / "gsim_run.py").write_text("def run_program(*a, **k): ...", encoding="utf-8")
    monkeypatch.setenv("MERLIN_GSIM_REQUIRE_RECEIPT", "1")
    monkeypatch.setattr(PO, "_rtl_engine_dir", lambda target, engine: home)
    monkeypatch.setattr(
        GE,
        "resolve_wrapper",
        lambda target, **k: GE.Resolution(
            target=target,
            path=home / "gsim_run.py",
            source="derived",
            ok=False,
            refused=True,
            reason="lineage ADOPTED, not built-and-bound",
            flavour="wrapper",
            digest="d",
            receipt_status="adopted",
            receipt=None,
        ),
    )
    available, reason = PO._rtl_engine_probe("t", "gsim")()
    assert available is False and "ADOPTED" in reason


def test_an_engine_this_module_does_not_lay_out_is_not_judged_by_it(tmp_path, monkeypatch):
    """A refusal authored by the wrong module would be worse than none: verilator's home is laid out
    and receipted by whoever built it, so the gsim lineage record must not be consulted for it."""
    from merlin.targetgen import gsim_emulator as GE
    from merlin.targetgen import program_oracle as PO

    home = tmp_path / "vsim"
    home.mkdir()
    (home / "verilator_run.py").write_text("def run_program(*a, **k): ...", encoding="utf-8")
    monkeypatch.setattr(PO, "_rtl_engine_dir", lambda target, engine: home)

    def _boom(*a, **k):
        raise AssertionError("the gsim lineage record was consulted for another engine")

    monkeypatch.setattr(GE, "resolve", _boom)
    assert PO._rtl_engine_probe("t", "verilator")()[0] is True


# ---------------------------------------------------------------------------------------------
# An engine BUILT in the other flavour is not an engine ABSENT -- and the probe must still refuse it,
# because this path cannot drive it. Both halves matter: the first is a truthful reason, the second is
# probe/executor agreement.
# ---------------------------------------------------------------------------------------------


def _binary_flavour_home(tmp_path, monkeypatch, name="gsim"):
    """A home holding the BINARY flavour (a standalone emulator) and no run_program wrapper."""
    from merlin.targetgen import gsim_emulator as GE
    from merlin.targetgen import program_oracle as PO

    home = tmp_path / name
    home.mkdir()
    binary = home / GE.BINARY_NAME
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(PO, "_rtl_engine_dir", lambda target, engine: home)
    monkeypatch.setattr(
        GE,
        "resolve",
        lambda target, **k: GE.Resolution(
            target=target,
            path=binary,
            source="derived",
            ok=True,
            refused=False,
            reason="built",
            flavour="binary",
            digest="d",
            receipt_status="bound",
            receipt=None,
        ),
    )
    return home, binary


def test_a_built_binary_flavour_is_not_reported_absent(tmp_path, monkeypatch):
    """Measured: gemmini's and radiance's GSIM homes hold the BINARY flavour, and this probe -- which
    stats one filename -- called the engine "absent". It is not absent; it is built in a flavour this
    route cannot import. Saying "absent" sends a reader hunting for a missing build."""
    from merlin.targetgen import program_oracle as PO

    _, binary = _binary_flavour_home(tmp_path, monkeypatch)
    available, reason = PO._rtl_engine_probe("t", "gsim")()
    assert available is False  # it still cannot be driven from here
    assert "IS built" in reason and "binary flavour" in reason
    assert str(binary) in reason
    assert "absent" not in reason  # the old answer, and the thing being fixed


def test_the_probe_and_the_executor_agree_about_that_home(tmp_path, monkeypatch):
    """The refusal is not pedantry: passing it would trade a clean unavailable for a late crash. This
    pins the two together -- whatever the probe says about this home, loading its runner must fail."""
    from merlin.targetgen import program_oracle as PO

    home, _ = _binary_flavour_home(tmp_path, monkeypatch)
    assert PO._rtl_engine_probe("t", "gsim")()[0] is False
    with pytest.raises(PO.OracleUnavailable):
        PO._load_rtl_runner(home, "gsim_run.py")


def test_a_wrapper_flavour_home_still_passes(tmp_path, monkeypatch):
    """The distinction must not cost the flavour this route CAN drive."""
    from merlin.targetgen import gsim_emulator as GE
    from merlin.targetgen import program_oracle as PO

    home = tmp_path / "gsim"
    home.mkdir()
    wrapper = home / "gsim_run.py"
    wrapper.write_text("def run_program(*a, **k): ...", encoding="utf-8")
    monkeypatch.setattr(PO, "_rtl_engine_dir", lambda target, engine: home)
    monkeypatch.setattr(
        GE,
        "resolve",
        lambda target, **k: GE.Resolution(
            target=target,
            path=wrapper,
            source="derived",
            ok=True,
            refused=False,
            reason="built",
            flavour="wrapper",
            digest="d",
            receipt_status="bound",
            receipt=None,
        ),
    )
    assert PO._rtl_engine_probe("t", "gsim")()[0] is True


def test_gsim_cap_operator_override(monkeypatch):
    import merlin.targetgen.rtl_engine_policy as policy

    monkeypatch.delenv("MERLIN_GSIM_MAX_SLOTS", raising=False)
    assert policy._gsim_cap_override() is None
    monkeypatch.setenv("MERLIN_GSIM_MAX_SLOTS", "15")
    assert policy._gsim_cap_override() == 15
    for bad in ("0", "100000"):
        monkeypatch.setenv("MERLIN_GSIM_MAX_SLOTS", bad)
        with pytest.raises(ValueError):
            policy._gsim_cap_override()
