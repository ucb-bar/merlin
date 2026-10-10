"""Which elaborated-RTL simulator certifies a capsule — and why that one.

The cert tier is a FIDELITY, not a simulator. VCS, GSIM and Verilator all run the elaborated design and
all produce an ``elaborated_rtl`` verdict; which one answers is an availability and cost decision, not a
statement about how trustworthy the result is. Binding a tier index to one binary (``L3 = verilator``)
made that decision invisible and unchangeable, and it put two different fidelities on the same rung
across targets.

The order is COST, since the fidelity is equal:

* ``vcs`` — the reference commercial simulator; used when a license and the resources are actually free.
* ``gsim`` — the fast FIRRTL simulator, and the default working choice. Measured at corpus scale on the
  SIMT target: 25 capsules certified in 48 min, mean 115 s/capsule, against ~45 min/capsule on Verilator
  — ~23x, i.e. the same sweep is ~19 h on Verilator. That is the difference between a cert tier that
  runs per-capsule and one affordable only once per run.
* ``verilator`` — last resort, for a target with no GSIM adapter yet. Not a fidelity compromise; it is
  simply the slow one.

Selection is by AVAILABILITY in that order, and every engine passed over is recorded with the reason it
was passed over. A tier that cannot run must come back as unavailable with that record — never silently
downgraded to a model tier, which is how a functional result gets read as an RTL certification.

WHAT "EQUAL FIDELITY" RESTS ON, and what it does not cover. The equality above is established by an
equivalence certificate, and those certificates carry ``evidence: output_bytes``: the engines produced
identical output for identical ELFs. Halting behaviour is NOT in that evidence, and the engines differ
there. Measured on this target: a program that violates a design ``assert`` makes Verilator ``$stop``
and exit non-zero, while the GSIM model prints the same assertion and runs to completion with
``exit_code=0`` and a full console. A caller that decided on the exit code alone therefore treated a
refused program as a clean run — so the backend refuses on the assertion text itself, and an engine
adopted here on an output-bytes certificate alone has been adopted on a narrower claim than this
ranking implies. Assertion enforcement, trap reporting and timeout semantics each need their own
evidence before an engine is ranked as equal.
"""

from __future__ import annotations

import fcntl
import os
import stat
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Declared once, with the rationale above. Cost order among engines of EQUAL fidelity.
ENGINE_PRIORITY: tuple[str, ...] = ("vcs", "gsim", "verilator")

# Bound per-suite fan-out by simulator cost, even when a caller explicitly asks
# for more workers. Host-load sizing is a snapshot: two independent grades can
# both observe an idle host and otherwise each start a full-width pool. The
# asynchronous broker and direct/scheduled graders must use the same caps.
CAPSULE_WORKER_CAP: dict[str, int] = {"verilator": 4, "gsim": 5, "spike": 8}


def _gsim_cap_override() -> int | None:
    """Operator-declared GSim cap for a dedicated host (the same-user slot count and per-suite cap)."""
    raw = os.environ.get("MERLIN_GSIM_MAX_SLOTS", "").strip()
    if not raw:
        return None
    value = int(raw)
    if not 1 <= value <= (os.cpu_count() or 1):
        raise ValueError(f"MERLIN_GSIM_MAX_SLOTS={raw} must be between 1 and the host CPU count")
    return value


if (_override := _gsim_cap_override()) is not None:
    CAPSULE_WORKER_CAP["gsim"] = _override
GSIM_RUNTIME_SLOT_PROTOCOL = "reentrant_per_thread_v1"

# Host runners and selected backends may both guard the same synchronous native
# call. Reuse only this thread's ownership; independent threads still need slots.
_GSIM_LOCAL = threading.local()
_GSIM_FDS: set[int] = set()
_GSIM_FD_LOCK = threading.Lock()


def _gsim_after_fork_child() -> None:
    global _GSIM_LOCAL
    # flock ownership follows the open file description across fork. Close the
    # child's copies without unlocking the parent's reservations.
    for fd in _GSIM_FDS:
        os.close(fd)
    _GSIM_FDS.clear()
    _GSIM_LOCAL = threading.local()
    _GSIM_FD_LOCK.release()


os.register_at_fork(
    before=_GSIM_FD_LOCK.acquire,
    after_in_parent=_GSIM_FD_LOCK.release,
    after_in_child=_gsim_after_fork_child,
)


def _close_gsim_fd(fd: int) -> None:
    with _GSIM_FD_LOCK:
        _GSIM_FDS.discard(fd)
        os.close(fd)


def _verified_slot_file(path: Path) -> int:
    """Open a private lock inode and keep fork cleanup aware of its descriptor."""
    with _GSIM_FD_LOCK:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        _GSIM_FDS.add(fd)
    try:
        info = os.fstat(fd)
    except BaseException:
        _close_gsim_fd(fd)
        raise
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        _close_gsim_fd(fd)
        raise RuntimeError(f"GSim slot file is not regular and owned by this user: {path}")
    return fd


def _read_process_evidence(path: Path, member: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, ProcessLookupError) as exc:
        try:
            member.stat()
        except (FileNotFoundError, ProcessLookupError):
            return None  # The process exited during the census.
        except OSError as stat_exc:
            raise RuntimeError(f"cannot verify native process {member.name}: {stat_exc}") from stat_exc
        raise RuntimeError(f"cannot verify native process {member.name}: missing {path.name}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot verify native process {member.name}: {exc}") from exc


def _status_rows(status: bytes, field: bytes) -> list[list[bytes]]:
    rows = []
    for line in status.splitlines():
        name, separator, value = line.partition(b":")
        if separator and name == field:
            rows.append(value.split())
    return rows


def _verified_uids(status: bytes, member: Path) -> tuple[int, ...]:
    uid_rows = _status_rows(status, b"Uid")
    if len(uid_rows) != 1 or len(uid_rows[0]) != 4:
        raise RuntimeError(f"cannot verify native process {member.name}: malformed Uid evidence")
    try:
        credentials = tuple(int(value) for value in uid_rows[0])
    except ValueError as exc:
        raise RuntimeError(f"cannot verify native process {member.name}: malformed Uid evidence") from exc
    if any(value < 0 for value in credentials):
        raise RuntimeError(f"cannot verify native process {member.name}: malformed Uid evidence")
    return credentials


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    ppid: int
    starttime: int
    argv: tuple[bytes, ...] | None = None


@dataclass(frozen=True)
class _NativeCensus:
    count: int
    matchable: tuple[_ProcessIdentity, ...]


@dataclass(frozen=True)
class _HeldSlot:
    index: int
    device: int
    inode: int


@dataclass(frozen=True)
class _SlotCensus:
    count: int  # Includes this caller's pending reservation.
    held: tuple[_HeldSlot, ...]


def _process_identity(member: Path, status: bytes, *, uid: int, argv: bytes | None = None) -> _ProcessIdentity | None:
    """Return only a stable, same-real/effective/saved/filesystem-UID process identity.

    Missing or malformed ancestry is not proof of an overlap. It must never
    subtract from the conservative native count or the kernel-held slot count.
    """
    try:
        if not member.name.isdecimal() or any(value != uid for value in _verified_uids(status, member)):
            return None
        parent_rows = _status_rows(status, b"PPid")
        state_rows = _status_rows(status, b"State")
        if len(parent_rows) != 1 or len(parent_rows[0]) != 1 or len(state_rows) != 1 or not state_rows[0]:
            return None
        if len(state_rows[0][0]) != 1 or not state_rows[0][0].isalpha():
            return None
        if state_rows[0][0] in {b"Z", b"X"}:
            return None
        stat_bytes = _read_process_evidence(member / "stat", member)
        if stat_bytes is None:
            return None
        prefix, separator, suffix = stat_bytes.strip().rpartition(b") ")
        fields = suffix.split()
        if not separator or b" (" not in prefix or len(fields) < 20:
            return None
        pid = int(member.name)
        if int(prefix.split(b" (", 1)[0]) != pid or int(fields[1]) != int(parent_rows[0][0]):
            return None
        if fields[0] != state_rows[0][0] or int(fields[19]) <= 0:
            return None
        parsed_argv = None
        if argv is not None:
            if not argv.endswith(b"\0"):
                return None
            parsed_argv = tuple(argv[:-1].split(b"\0"))
            if not parsed_argv or any(not arg for arg in parsed_argv):
                return None
        return _ProcessIdentity(pid, int(fields[1]), int(fields[19]), parsed_argv)
    except (OSError, RuntimeError, ValueError, IndexError):
        return None


def _complete_native_argv(argv: tuple[bytes, ...] | None) -> bool:
    if not argv:
        return False
    loads = [arg.removeprefix(b"+loadmem=") for arg in argv if arg.startswith(b"+loadmem=")]
    limits = [arg.removeprefix(b"+max-cycles=") for arg in argv if arg.startswith(b"+max-cycles=")]
    return (
        len(loads) == len(limits) == 1
        and bool(loads[0])
        and 0 < len(limits[0]) <= 32
        and limits[0].isdigit()
        and int(limits[0]) > 0
    )


def _native_gsim_census(*, proc_root: Path = Path("/proc")) -> _NativeCensus:
    """Bound same-user native plusarg load without assuming an emulator name or path.

    The kernel-reported UID is checked before reading an argv. Missing or unreadable
    evidence refuses admission; a partial native plusarg signature and a verified live
    process with persistently empty argv each count as one potential native. This is an
    upper bound, not an exact process census. A process appearing after this snapshot
    remains an external race.
    """
    try:
        members = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot verify native process census: {exc}") from exc
    current_uid = os.getuid()
    count = 0
    matchable: list[_ProcessIdentity] = []
    for member in members:
        if not member.name.isdecimal():
            continue
        status = _read_process_evidence(member / "status", member)
        if status is None:
            continue
        credentials = _verified_uids(status, member)
        if current_uid not in credentials:
            continue
        cmdline = _read_process_evidence(member / "cmdline", member)
        if cmdline is None:
            continue
        if not cmdline:
            # A fork/exec or exit can leave a live status with an empty argv for a
            # moment. Wait briefly for complete evidence, then conservatively count
            # a verified same-user live process as one potential native simulator.
            for attempt in range(5):
                current_status = _read_process_evidence(member / "status", member)
                if current_status is None:
                    break
                state_rows = _status_rows(current_status, b"State")
                if (
                    len(state_rows) != 1
                    or not state_rows[0]
                    or len(state_rows[0][0]) != 1
                    or not state_rows[0][0].isalpha()
                ):
                    raise RuntimeError(f"cannot verify native process {member.name}: malformed State evidence")
                if state_rows[0][0] in {b"Z", b"X"}:
                    break
                if current_uid not in _verified_uids(current_status, member):
                    raise RuntimeError(f"cannot verify native process {member.name}: changed Uid evidence")
                if attempt == 4:
                    count += 1
                    break
                time.sleep(0.01)
                cmdline = _read_process_evidence(member / "cmdline", member)
                if cmdline is None or cmdline:
                    break
            if not cmdline:
                continue
        if not cmdline.endswith(b"\0"):
            raise RuntimeError(f"cannot verify native process {member.name}: incomplete argv")
        argv = cmdline[:-1].split(b"\0")
        if any(arg.startswith((b"+loadmem=", b"+max-cycles=")) for arg in argv):
            count += 1
            identity = _process_identity(member, status, uid=current_uid, argv=cmdline)
            if identity is not None and _complete_native_argv(identity.argv):
                matchable.append(identity)
    return _NativeCensus(count, tuple(matchable))


def _native_gsim_count(*, proc_root: Path = Path("/proc")) -> int:
    """Conservative scalar census for diagnostics and legacy callers."""
    return _native_gsim_census(proc_root=proc_root).count


def _locked_slot_census(root: Path, *, own_index: int) -> _SlotCensus:
    """Include this pending reservation and all old/new kernel file-lock holders."""
    count = 1
    held: list[_HeldSlot] = []
    for index in range(CAPSULE_WORKER_CAP["gsim"]):
        if index == own_index:
            continue
        fd = _verified_slot_file(root / f"slot_{index}.lock")
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                count += 1
                info = os.fstat(fd)
                held.append(_HeldSlot(index, info.st_dev, info.st_ino))
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            _close_gsim_fd(fd)
    return _SlotCensus(count, tuple(held))


def _lock_owners(locks: bytes, held: tuple[_HeldSlot, ...]) -> dict[_HeldSlot, int]:
    """Return only exact held inodes with one unambiguous kernel FLOCK owner."""
    owners: dict[_HeldSlot, list[int]] = {slot: [] for slot in held}
    by_inode = {(os.major(slot.device), os.minor(slot.device), slot.inode): slot for slot in held}
    if len(by_inode) != len(held):
        return {}  # Hardlinked slot names have no independent ownership proof.
    invalid: set[_HeldSlot] = set()
    for line in locks.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] != b"FLOCK":
            continue
        try:
            major, minor, inode = fields[5].split(b":")
            key = (int(major, 16), int(minor, 16), int(inode))
            slot = by_inode.get(key)
            if slot is None:
                continue
            if len(fields) != 8 or fields[2:4] != [b"ADVISORY", b"WRITE"]:
                invalid.add(slot)
                continue
            if fields[6:] != [b"0", b"EOF"] or int(fields[4]) <= 0:
                invalid.add(slot)
                continue
            owners[slot].append(int(fields[4]))
        except (ValueError, IndexError):
            return {}  # A malformed lock record cannot prove any overlap.
    return {slot: matched[0] for slot, matched in owners.items() if len(matched) == 1 and slot not in invalid}


def _matching_owner_fd(member: Path, slot: _HeldSlot) -> bool:
    try:
        for descriptor in (member / "fd").iterdir():
            if not descriptor.name.isdecimal():
                continue
            info = descriptor.stat()
            if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (slot.device, slot.inode):
                return True
    except OSError:
        return False
    return False


def _fresh_identity(member: Path, *, uid: int, native: bool) -> _ProcessIdentity | None:
    try:
        status = _read_process_evidence(member / "status", member)
        if status is None:
            return None
        argv = _read_process_evidence(member / "cmdline", member) if native else None
        if native and argv is None:
            return None
        identity = _process_identity(member, status, uid=uid, argv=argv)
        if native and (identity is None or not _complete_native_argv(identity.argv)):
            return None
        return identity
    except (OSError, RuntimeError):
        return None


def _verified_native_slot_overlap(
    root: Path,
    slots: _SlotCensus,
    natives: _NativeCensus,
    *,
    proc_root: Path = Path("/proc"),
    locks_path: Path = Path("/proc/locks"),
) -> int:
    """Discount only independently proven, stable direct-child native/slot pairs.

    This is not a trust decision from a lock-file payload. The kernel FLOCK
    owner, open FD inode, process UID/PPid/starttime and complete native argv
    must agree before *and* after the join. Unknown evidence means zero
    discount for that slot, including old holders with no independently proven
    child. A same-PID multi-slot holder cannot pair children to particular
    slots, so it receives no discount and may use fewer than five workers.
    """
    if not slots.held or not natives.matchable:
        return 0
    try:
        first = _lock_owners(locks_path.read_bytes(), slots.held)
    except OSError:
        return 0
    uid = os.getuid()
    proven: list[tuple[_HeldSlot, _ProcessIdentity, _ProcessIdentity]] = []
    for slot, owner_pid in first.items():
        # A multi-slot owner has no unique slot-to-child association.
        if list(first.values()).count(owner_pid) != 1:
            continue
        owner_member = proc_root / str(owner_pid)
        owner = _fresh_identity(owner_member, uid=uid, native=False)
        if owner is None or not _matching_owner_fd(owner_member, slot):
            continue
        children = [
            child for child in natives.matchable if child.ppid == owner_pid and child.starttime > owner.starttime
        ]
        if len(children) != 1:
            continue
        proven.append((slot, owner, children[0]))
    try:
        last = _lock_owners(locks_path.read_bytes(), slots.held)
    except OSError:
        return 0
    matched_children: set[tuple[int, int]] = set()
    accepted: list[tuple[_HeldSlot, int]] = []
    for slot, owner, child in proven:
        if last.get(slot) != owner.pid or (child.pid, child.starttime) in matched_children:
            continue
        try:
            current = (root / f"slot_{slot.index}.lock").lstat()
        except OSError:
            continue
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (slot.device, slot.inode):
            continue
        owner_member = proc_root / str(owner.pid)
        child_member = proc_root / str(child.pid)
        if _fresh_identity(owner_member, uid=uid, native=False) != owner:
            continue
        if not _matching_owner_fd(owner_member, slot):
            continue
        if _fresh_identity(child_member, uid=uid, native=True) != child:
            continue
        matched_children.add((child.pid, child.starttime))
        accepted.append((slot, owner.pid))
    # The owner may have released/replaced a slot during the post-identity
    # checks; never discount it using only the earlier lock snapshot.
    try:
        final = _lock_owners(locks_path.read_bytes(), slots.held)
    except OSError:
        return 0
    return sum(final.get(slot) == owner_pid for slot, owner_pid in accepted)


def _admitted_load(
    root: Path,
    *,
    own_index: int,
    proc_root: Path = Path("/proc"),
    locks_path: Path = Path("/proc/locks"),
) -> int:
    """Conservative union of natives and reservations, including our pending slot."""
    natives = _native_gsim_census(proc_root=proc_root)
    slots = _locked_slot_census(root, own_index=own_index)
    raw_load = natives.count + slots.count
    if raw_load <= CAPSULE_WORKER_CAP["gsim"]:
        return raw_load
    overlap = _verified_native_slot_overlap(root, slots, natives, proc_root=proc_root, locks_path=locks_path)
    if not 0 <= overlap <= min(natives.count, len(natives.matchable), len(slots.held)):
        raise RuntimeError("invalid GSim native/slot overlap evidence")
    return raw_load - overlap


@contextmanager
def _admission_mutex(root: Path, deadline: float | None):
    fd = _verified_slot_file(root / "admission.lock")
    locked = False
    try:
        while not locked:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("all five GSim slots stayed busy until the capsule wait deadline")
                time.sleep(0.01)
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        _close_gsim_fd(fd)


def capsule_worker_cap(engine: str) -> int:
    """Conservative parallel capsule limit for one simulator engine."""
    return CAPSULE_WORKER_CAP.get(engine, 8)


@contextmanager
def _gsim_cpu_slot(index: int):
    """Pin this native-launch thread and its children to one operator-selected CPU."""
    raw = os.environ.get("MERLIN_GSIM_CPUS", "").strip()
    if not raw:
        yield
        return
    tokens = raw.split(",")
    if any(not token.isdecimal() for token in tokens):
        raise ValueError("MERLIN_GSIM_CPUS requires a comma-separated CPU roster")
    cpus = tuple(int(token) for token in tokens)
    if len(set(cpus)) != len(cpus) or len(cpus) < CAPSULE_WORKER_CAP["gsim"]:
        raise ValueError("MERLIN_GSIM_CPUS requires a distinct CPU for every configured slot")
    original = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {cpus[index]})
        if os.sched_getaffinity(0) != {cpus[index]}:
            raise RuntimeError("selected GSim CPU is unavailable in the process CPU partition")
        yield
    finally:
        os.sched_setaffinity(0, original)


@contextmanager
def gsim_runtime_slot(*, wait_timeout_s: float | None = None, slot_root: Path | None = None):
    """Hold one of five same-user GSim slots for the entire native simulation.

    Per-suite worker bounds do not protect the machine when two independent grades run at once.
    Advisory file locks survive abrupt worker exit without stale PID reclamation. A private,
    fixed per-user directory makes every Merlin capsule process share the same limit. An
    admission mutex joins held/pending reservations with a same-user native /proc census;
    uncoordinated launches after that snapshot cannot be prevented by this protocol.
    Nested synchronous calls in the same thread reuse its slot. Other threads and
    forked children acquire their own; do not launch concurrent children inside one slot.
    Ambiguous multi-slot ownership can conservatively reduce utilization below five.
    """
    selected_root = os.environ.get("MERLIN_GSIM_SLOT_ROOT", "").strip()
    root = slot_root or (Path(selected_root) if selected_root else Path("/tmp") / f"merlin_gsim_slots_{os.getuid()}")
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"GSim slot directory is not private and owned by this user: {root}")
    key = str(root.resolve())
    held = getattr(_GSIM_LOCAL, "held", None)
    if held is None:
        held = _GSIM_LOCAL.held = {}
    if key in held:
        yield
        return
    owner_pid = os.getpid()
    deadline = None if wait_timeout_s is None else time.monotonic() + max(0.0, wait_timeout_s)
    fd = None
    try:
        while fd is None:
            with _admission_mutex(root, deadline):
                for index in range(CAPSULE_WORKER_CAP["gsim"]):
                    opened = _verified_slot_file(root / f"slot_{index}.lock")
                    try:
                        fcntl.flock(opened, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        _close_gsim_fd(opened)
                        continue
                    except BaseException:
                        _close_gsim_fd(opened)
                        raise
                    try:
                        # The census includes live empty-argv PIDs as potential
                        # natives. A proven owner/child pair counts once; all
                        # ambiguous slots and native processes count separately.
                        admitted_load = _admitted_load(root, own_index=index)
                        if admitted_load <= CAPSULE_WORKER_CAP["gsim"]:
                            fd = opened
                            break
                    finally:
                        if fd is None:
                            fcntl.flock(opened, fcntl.LOCK_UN)
                            _close_gsim_fd(opened)
                    break
            if fd is None:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("all five GSim slots stayed busy until the capsule wait deadline")
                time.sleep(0.1)
        held[key] = fd
        with _gsim_cpu_slot(index):
            yield
    finally:
        if fd is not None and os.getpid() == owner_pid:
            held.pop(key, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
            _close_gsim_fd(fd)


# Every engine here answers at this fidelity; the tier records it rather than inferring from the name.
ELABORATED_RTL = "elaborated_rtl"


class UnrecordedSelection(RuntimeError):
    """An engine reported itself available without saying why. Refused: a tier that resolved to an engine
    for no recorded reason cannot be audited afterwards, and reads as if it were the declared one."""

    def __init__(self, target: str, engine: str):
        self.target, self.engine = target, engine
        super().__init__(
            f"{target}: engine {engine!r} reported available with no reason recorded; a "
            f"selection that cannot be explained afterwards is refused, not defaulted"
        )


class NoEngineAvailable(RuntimeError):
    """No elaborated-RTL engine can run for this target. Carries the per-engine reasons."""

    def __init__(self, target: str, considered: list[dict[str, Any]]):
        self.target, self.considered = target, considered
        detail = "; ".join(f"{c['engine']}: {c['reason']}" for c in considered) or "none registered"
        super().__init__(f"{target}: no elaborated-RTL engine available ({detail})")


def _ordered(engines: dict[str, Any]) -> list[str]:
    """Registered engines in priority order; anything unknown to the policy sorts last, alphabetically,
    so a newly added engine is USED rather than silently dropped before anyone declares its priority."""
    known = [e for e in ENGINE_PRIORITY if e in engines]
    return known + sorted(e for e in engines if e not in ENGINE_PRIORITY)


def select(target: str, engines: dict[str, Callable[[], tuple[bool, str]]]) -> dict[str, Any]:
    """Choose the elaborated-RTL engine for ``target``.

    ``engines`` maps an engine name to a probe returning ``(available, reason)``. Probes are called in
    priority order and STOP at the first available one, so an expensive probe for a lower-priority
    engine is never paid. Returns the selection record; raises :class:`NoEngineAvailable` when none can
    run (fail closed — the caller reports the tier unavailable, it does not substitute a lesser tier).
    """
    considered: list[dict[str, Any]] = []
    for name in _ordered(engines):
        try:
            ok, reason = engines[name]()
        except Exception as exc:  # noqa: BLE001 - a broken probe is not availability
            ok, reason = False, f"probe raised {type(exc).__name__}: {exc}"
        considered.append({"engine": name, "available": bool(ok), "reason": reason})
        if ok and not str(reason or "").strip():
            # An engine that resolved for NO RECORDED REASON is the silent-degradation shape: the tier
            # answers with a different engine than the capsule asked for, the numbers look right, and the
            # result gets cited. Refuse it rather than defaulting.
            raise UnrecordedSelection(target, name)
        if ok:
            return {
                "engine": name,
                "fidelity": ELABORATED_RTL,
                "reason": reason,
                "considered": considered,
                "passed_over": [c["engine"] for c in considered[:-1]],
            }
    raise NoEngineAvailable(target, considered)


def describe(selection: dict[str, Any]) -> str:
    """One line for a report: what ran, and what it was chosen over."""
    over = selection.get("passed_over") or []
    tail = f" (over {', '.join(over)})" if over else ""
    return f"{selection['engine']} [{selection['fidelity']}]{tail}"
