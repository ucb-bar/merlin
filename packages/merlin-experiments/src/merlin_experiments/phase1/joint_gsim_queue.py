"""Waiting order for the joint gSIM correctness slots, kept in the flock'd correctness ledger.

A full concurrency cap with budget left is a queue, not a refusal: an author request waits for a
slot across every process that shares the ledger, and is served before background work (promotions,
operator measurements) of other cases, because the author is blocked on it. The order is a priority,
never a quota: background work still takes any slot no author is waiting for.

Within a class, waiting requests are interleaved across their requesters (the caller thread a
self-check names, :data:`REQUESTER_ENV`): each requester's oldest waiting request goes first, then each
one's second, and so on, each round in arrival order. A sweep of many single-case requests keeps its
own order but no longer holds another requester's single check behind all of it. Requests that name
no requester share one anonymous requester, which keeps arrival order among them.

Background work does not wait inside its evaluation child (its broker keeps it queued until a slot
is free, :func:`background_slots`); a background child that loses the race reports a retryable
refusal (:class:`SlotBusy`) and its broker leaves the request queued. A waiting entry belongs to a
live process: one whose pid is gone is dropped, as dead workers already release their slot.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

#: Environment the evaluation child reads to learn which class its request belongs to.
CLASS_ENV = "MERLIN_JOINT_GSIM_QUEUE_CLASS"
#: The broker's request id, recorded on the child's slot so its broker can tell a started job apart.
REQUEST_ENV = "MERLIN_JOINT_GSIM_REQUEST"
#: Who asked (a self-check's caller thread); waiting requests are interleaved across requesters.
REQUESTER_ENV = "MERLIN_JOINT_GSIM_REQUESTER"
AUTHOR = "author"
BACKGROUND = "background"
_RANK = {AUTHOR: 0, BACKGROUND: 1}
#: Progress status a waiting child publishes; brokers read it to tell waiting from running.
WAITING_STATUS = "queued_for_gsim_slot"
ADMITTED_STATUS = "gsim_slot_admitted"


class SlotBusy(ValueError):
    """Every slot is busy and this request does not wait; the request itself stays valid."""

    retryable = True


def request_class(env=None) -> str:
    value = (os.environ if env is None else env).get(CLASS_ENV, "").strip()
    return value if value in _RANK else AUTHOR


def _alive(pid) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def settle(data: dict) -> None:
    """Drop slots and waiting entries of processes that no longer exist (charges are kept)."""
    data["active"] = [row for row in data.get("active", ()) if _alive(row.get("pid"))]
    data["waiting"] = [row for row in data.get("waiting", ()) if _alive(row.get("pid"))]


def requester(env=None) -> str | None:
    value = (os.environ if env is None else env).get(REQUESTER_ENV, "").strip()
    return value[:128] or None


def name_requester(env: dict, request: dict) -> dict:
    """Set the evaluation child's requester from a self-check request's caller thread (else none).

    Used only to interleave waiting requests; it grants nothing and a request without it waits as one
    anonymous requester.
    """
    env.pop(REQUESTER_ENV, None)
    caller = request.get("caller_thread_id") if isinstance(request, dict) else None
    if isinstance(caller, str) and caller.strip():
        env[REQUESTER_ENV] = caller.strip()[:128]
    return env


def _ordered(data: dict) -> list:
    """Class first; within a class, round-robin over requesters, each round in arrival order."""
    rows = sorted(data.get("waiting", ()), key=lambda row: (_RANK.get(row.get("class"), 1), row.get("seq", 0)))
    turns: dict[tuple, int] = {}
    keyed = []
    for row in rows:
        rank = _RANK.get(row.get("class"), 1)
        own = (rank, row.get("requester"))
        keyed.append(((rank, turns.get(own, 0), row.get("seq", 0)), row))
        turns[own] = turns.get(own, 0) + 1
    return [row for _, row in sorted(keyed, key=lambda item: item[0])]


def enqueue(data: dict, token: str, *, capsule: str, request_class: str, now: float, requester=None) -> None:
    """Add ``token`` behind its requester's earlier entries (idempotent: an entry keeps its place)."""
    waiting = data.setdefault("waiting", [])
    if any(row.get("token") == token for row in waiting):
        return
    seq = int(data.get("next_seq", 0))
    data["next_seq"] = seq + 1
    waiting.append({"token": token, "pid": os.getpid(), "capsule": capsule, "class": request_class,
                    "seq": seq, "enqueued_at_epoch": now, **({"requester": requester} if requester else {})})


def leave(data: dict, token: str) -> None:
    data["waiting"] = [row for row in data.get("waiting", ()) if row.get("token") != token]


def ahead(data: dict, token: str) -> int:
    """Waiting entries served before ``token``."""
    for index, row in enumerate(_ordered(data)):
        if row.get("token") == token:
            return index
    return len(data.get("waiting", ()))


def admissible(data: dict, token: str, max_jobs: int) -> bool:
    """Whether a free slot is left for ``token`` after every entry ahead of it is served."""
    return ahead(data, token) < max_jobs - len(data.get("active", ()))


def background_slots(data: dict, max_jobs: int, starting=()) -> int:
    """Slots background work may start now on a settled ledger: none while an author waits.

    ``starting`` names request ids a broker launched whose child has not yet taken its slot (it
    registers ``REQUEST_ENV`` on its row); they count as busy, so a broker polling faster than its
    children start cannot launch more jobs than there are slots.
    """
    if any(row.get("class") == AUTHOR for row in data.get("waiting", ())):
        return 0
    active = data.get("active", ())
    registered = {row.get("request") for row in active}
    return max(0, max_jobs - len(active) - len(set(starting) - registered))


def describe(data: dict, token: str, max_jobs: int) -> dict:
    n_ahead = ahead(data, token)
    return {"state": "waiting", "ahead": n_ahead, "slots": max_jobs, "busy": len(data.get("active", ())),
            "note": f"queued for a gSIM slot ({n_ahead} ahead)"}


def publish(path, status: str, queue: dict) -> None:
    """Write the queue state as request progress; never a verdict, so a failure to write is ignored."""
    if not path:
        return
    path = Path(path)
    document = {"status": status, "gsim_slot_queue": queue, "updated_at_unix_ns": time.time_ns(),
                "note": "Waiting for a gSIM correctness slot; not a verdict and not a refusal."}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(document))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)


def waiting(path) -> dict | None:
    """The queue block of a request whose child is waiting for a slot, else ``None`` (never raises)."""
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(document, dict) and document.get("status") == WAITING_STATUS:
        queue = document.get("gsim_slot_queue")
        return queue if isinstance(queue, dict) else {}
    return None
