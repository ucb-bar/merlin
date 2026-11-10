"""Standalone self-check request identity, admission and channel protocol; standard library only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from pathlib import Path

_SIMS = ("spike", "verilator", "gsim", "vcs")


PROTOCOL_VERSION = 3


_DIGEST_IGNORES = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})


_SUITE_SIZE_FALLBACK = 64


_CALIBRATION_CAP = 3


_SUBMISSION = "submission"


_VARIANTS = "variants"


_VARIANT_NAME_MAX = 64


_NAME_FIRST = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


_NAME_CHARS = _NAME_FIRST | frozenset("._-")


#: Completed requests listed beside in-flight ones.
_LIST_COMPLETED = 5


#: A panel selects one public case per declared shape class.
_PANEL = "panel:"


_PANEL_CASES = 6


#: Mirrors the broker status without importing host-side Merlin.
_GSIM_QUEUE_STATUS = "queued_for_gsim_slot"


def _gsim_queue(progress) -> dict | None:
    """The queue block of a request waiting for a gSIM slot, else ``None``."""
    if isinstance(progress, dict) and progress.get("status") == _GSIM_QUEUE_STATUS:
        queue = progress.get("gsim_slot_queue")
        return queue if isinstance(queue, dict) else {}
    return None


def _gsim_queue_note(queue: dict) -> str:
    return f"queued for a gSIM slot ({queue.get('ahead', '?')} ahead)"


def _submission_digest(root: Path) -> str:
    """Hash authored submission bytes using the same build-state exclusions as the broker."""
    root = Path(root)
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _DIGEST_IGNORES for part in rel.parts):
            continue
        if path.is_symlink():
            digest.update(rel.as_posix().encode() + b"\0SYMLINK\0" + os.readlink(path).encode())
        elif path.is_file():
            digest.update(rel.as_posix().encode() + b"\0")
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _required_rtl_engine() -> str | None:
    return os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None


def _default_sim() -> str | None:
    """The pinned RTL engine, or ``None`` meaning "the driver picks".

    This shim imports nothing from merlin, so it cannot see which engines the target actually has —
    and it used to answer that question anyway, with the historical ``"verilator"`` literal. That
    literal then travelled in the request and OVERRODE the driver-side discovery, because an explicit
    ``sim`` in the payload is a choice the broker must honour. Measured consequence: every self-check
    of a run that pins no engine ran the slowest engine while the grader beside it selected the fast
    one, and the agent's loop degraded ~15x with nothing naming the cause.

    Returning ``None`` sends no ``sim`` at all, so the one component that CAN see the engines — the
    real ``agent_selfcheck.py``, outside the box — resolves it. An explicit ``--sim`` still travels
    and is still enforced against the pin below.
    """
    if os.environ.get("MERLIN_JOINT_DEVELOPMENT") == "1":
        return "spike"
    required = _required_rtl_engine()
    return required if required in _SIMS and required != "spike" else None


def _sim_policy_error(sim: str | None) -> str | None:
    """Mirror the driver-side enforcement so an invalid request fails before any channel work."""
    if sim is None or sim == "spike":
        return None
    required = _required_rtl_engine()
    if required is None:
        return None
    if required not in _SIMS or required == "spike":
        return f"MERLIN_REQUIRED_RTL_ENGINE={required!r} is not a registered elaborated-RTL self-check engine"
    if sim != required:
        return (
            f"--sim {sim!r} conflicts with MERLIN_REQUIRED_RTL_ENGINE={required!r}; "
            f"use --sim 'spike' for a correctness-only screen or --sim {required!r} for RTL"
        )
    return None


def _request_id() -> str:
    """A diagnostic request id which stays unique across PID-namespace and clock reuse.

    Namespace PIDs are short-lived and the previous millisecond clock component wrapped every
    1,000,000 ms.  Both values have collided in long rounds, at which point a new shim can observe an
    old request's response and completion marker.  Keep them for operator diagnostics, but add a
    cryptographic nonce and use the full nanosecond clock rather than a modulo clock.
    """
    return f"{os.getpid()}_{time.time_ns()}_{secrets.token_hex(16)}"


def _requester_identity() -> dict[str, int | str]:
    """Durable identity for the process waiting on a request.

    A PID alone is not enough across a long-lived file channel because Linux may reuse it.  Field 22
    of ``/proc/<pid>/stat`` is the process start time in clock ticks and distinguishes incarnations.
    The namespace inode is part of the identity because a sandbox PID is meaningful only inside that
    namespace; the host broker must not look the same number up in its own ``/proc``.
    """
    pid = os.getpid()
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        start_ticks = int(tail[19])
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError(f"cannot establish self-check requester identity: {exc}") from exc
    try:
        pid_namespace = os.readlink("/proc/self/ns/pid")
    except OSError as exc:
        raise RuntimeError(f"cannot establish self-check PID namespace: {exc}") from exc
    return {"pid": pid, "start_ticks": start_ticks, "pid_namespace": pid_namespace}


def _valid_variant_name(name: str) -> bool:
    return 0 < len(name) <= _VARIANT_NAME_MAX and name[0] in _NAME_FIRST and all(c in _NAME_CHARS for c in name)


def _source_label(ws: Path, value: str) -> tuple:
    """``(label, error)`` for ``--submission``: ``submission`` or ``variants/<name>`` in this workspace.

    The label, never a path, is what travels to the broker, which resolves it inside the workspace
    itself. Any other tree -- an operator directory, a path outside the workspace -- is refused here so
    the request is never written.
    """
    path = Path(os.path.normpath(str(value or _SUBMISSION)))
    if path.is_absolute():
        try:
            path = path.relative_to(ws)
        except ValueError:
            path = None
    parts = path.parts if path is not None else ()
    if parts == (_SUBMISSION,):
        return _SUBMISSION, None
    if len(parts) == 2 and parts[0] == _VARIANTS and _valid_variant_name(parts[1]):
        return f"{_VARIANTS}/{parts[1]}", None
    return None, (
        "--submission must be 'submission' or 'variants/<name>' inside this workspace (name: letters, "
        f"digits, '.', '_' or '-', at most {_VARIANT_NAME_MAX} characters, starting with a letter or digit); "
        f"got {value!r}"
    )


def _variant_error(ws: Path, label: str) -> str | None:
    """Why a variant tree cannot be checked, before any request is written (the broker re-checks)."""
    for path in (ws / _VARIANTS, ws / label):
        if path.is_symlink():
            return f"{path.relative_to(ws)} is a symlink; a variant must be a real directory in this workspace"
        if not path.is_dir():
            return f"{label}/ does not exist; create it with the layout of submission/ (cp -r submission {label})"
    return None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _capacity_note(ch: Path) -> str:
    """How many requests the grader serves at once, from its own health record."""
    health = _read_json(ch / "broker_health.json")
    health = health if isinstance(health, dict) else {}
    ordinary = health.get("ordinary_capacity")
    if not isinstance(ordinary, int):
        dispatch = health.get("dispatch_capacity")
        ordinary = dispatch - 1 if isinstance(dispatch, int) and dispatch > 1 else None
    if not isinstance(ordinary, int) or ordinary < 1:
        return "this grader serves one request at a time, in arrival order, and yours is waiting for the slot"
    variants, active = health.get("variant_capacity"), health.get("active_handlers")
    overflow = health.get("overflow_capacity")
    return (
        f"the grader runs up to {ordinary} self-check request(s) at once plus one diagnostic probe"
        + (f" (at most {variants} of them variants)" if isinstance(variants, int) else "")
        + (
            f", and up to {overflow} more while the reserved CPUs have room for author work"
            if isinstance(overflow, int) and overflow > 0
            else ""
        )
        + (f"; {active} running now" if isinstance(active, int) else "")
        + "; yours starts when a slot frees"
    )


def _list_requests(ch: Path) -> dict:
    """Every in-flight request in this workspace and the most recent completed ones."""
    now_ns, running, completed = time.time_ns(), [], []
    for req in ch.glob("req_*.json"):
        rid = req.name[len("req_") : -len(".json")]
        body = _read_json(req)
        body = body if isinstance(body, dict) else {}
        row = {
            "request_id": rid,
            "submission": body.get("submission", _SUBMISSION),
            "capsules": body.get("capsules"),
            "requested_at_unix_ns": body.get("requested_at_unix_ns"),
        }
        if (ch / f"done_{rid}").exists() and (ch / f"resp_{rid}.json").exists():
            verdict = _verdict((ch / f"resp_{rid}.json").read_text(errors="replace")) or {}
            row.update(state="done", **{k: verdict.get(k) for k in ("all_pass", "n_passed", "n_capsules", "error")})
            completed.append(row)
            continue
        deadline = body.get("deadline_unix_ns")
        if isinstance(deadline, int) and now_ns > deadline:
            row["state"] = "expired"
        else:
            row["state"] = "running" if (ch / f"claimed_{rid}").exists() else "queued"
            progress = _read_json(ch / f"progress_{rid}.json")
            queue = _gsim_queue(progress)
            if queue is not None:
                row.update(state=_GSIM_QUEUE_STATUS, note=_gsim_queue_note(queue))
            elif isinstance(progress, dict):
                row.update(n_finished=progress.get("n_finished"), n_expected=progress.get("n_expected"))
        running.append(row)

    def order(row):
        stamp = row.get("requested_at_unix_ns")
        return stamp if isinstance(stamp, int) else 0

    return {
        "in_flight": sorted(running, key=order),
        "recently_completed": sorted(completed, key=order)[-_LIST_COMPLETED:],
        "concurrency": _capacity_note(ch),
        "collect_with": "python agent_selfcheck.py --attach <request_id>",
    }


def _reply_is_usable(verdict: dict, rid: str, submission_sha256: str, *, label: str = _SUBMISSION) -> tuple:
    """``(usable, note)`` for a reply the broker published. ``note`` is empty when nothing needs saying.

    WHICH BYTES AND WHOSE REQUEST ARE DIFFERENT QUESTIONS, and conflating them starved the loop. A
    reply under the wrong request id is a channel fault and stays fatal. A reply about NEWER bytes is
    not a fault at all: the broker grades the submission as it stands when it reaches the request,
    because refusing otherwise means an agent that edits while its check is queued never gets a check.
    Measured on merlincirct_atlas_feedback_v3_20260906: five queued requests carrying three distinct
    digests, every one refused as stale, and the last COMPLETED self-check 5.5 h earlier while the
    agent kept working.

    Newer bytes are ACCEPTED AND ANNOUNCED. The verdict is about the agent's current code, which is
    more useful than the code it had when it asked -- but only while it says so, or the agent reads a
    verdict as describing something it has already replaced.

    A pure function so it can be tested: this decision used to live inside the polling loop, where the
    only way to exercise it was to run a broker.

    A reply about a DIFFERENT TREE is refused like one for a different request: a broker that does not
    know variants grades submission/, and its verdict must not be read as the variant's.
    """
    if verdict.get("selfcheck_protocol") != PROTOCOL_VERSION:
        return False, "carries the wrong protocol version"
    if verdict.get("selfcheck_request_id") != rid:
        return False, "belongs to a different request"
    if verdict.get("selfcheck_submission", _SUBMISSION) != label:
        return False, f"is about a different source tree than {label}/"
    graded = verdict.get("submission_sha256")
    if graded == submission_sha256:
        return True, ""
    if verdict.get("graded_newer_bytes"):
        return True, (
            verdict.get("graded_newer_note") or "graded your CURRENT submission, not the bytes present when you asked"
        )
    return False, "is about submission bytes that are neither the ones requested nor declared newer"


def _request_budget_seconds(timeout: int, *, capsules: str, workers: int, suite_size: int | None = None) -> int:
    """Upper-bound the channel lifetime from the grader's *per-capsule* timeout.

    ``agent_selfcheck`` passes ``timeout`` to every capsule run.  Treating that value as a deadline for
    the entire request aborts a healthy full-suite check after one capsule-timeout, even though the suite
    has a serial calibration head followed by a pipelined worker pool (``capsule_scheduling``).  Greedy
    list scheduling of members each bounded by ``timeout`` on ``workers`` finishes within
    ``(n - 1) / workers + 1`` timeouts (Graham), so the pool term is ``ceil(rest / workers) + 1``.  The
    bound is deliberately conservative: an individual capsule normally finishes far below its timeout,
    so successful requests still return promptly.
    """
    per_capsule = max(1, int(timeout))
    worker_count = max(1, int(workers))
    if capsules in ("all", "changed"):  # `changed` builds and screens every public capsule first
        count = max(1, int(suite_size or _SUITE_SIZE_FALLBACK))
    elif capsules.startswith(_PANEL):
        count = _PANEL_CASES
    else:
        count = max(1, len({name.strip() for name in capsules.split(",") if name.strip()}))
    serial = min(_CALIBRATION_CAP, count)
    rest = max(0, count - serial)
    pool = math.ceil(rest / worker_count) + (1 if rest > worker_count else 0)
    return (serial + pool) * per_capsule + 240


def _atomic_write(path: Path, text: str) -> None:
    """Publish one channel file only after all of its bytes are durable."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _verdict(txt: str):
    """The verdict object out of the broker's reply, or ``None`` if there isn't one.

    The reply is the real self-check's whole stdout, and the grader prints human diagnostics on that
    same stream, so the JSON is not always the first thing in it. Scan for the document rather than
    assuming the text IS one -- structurally, via ``raw_decode`` from each ``{`` (no regex).
    """
    try:
        whole = json.loads(txt)
        if isinstance(whole, dict):
            return whole
    except Exception:
        pass
    dec, i = json.JSONDecoder(), txt.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(txt, i)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i = txt.find("{", i + 1)
    return None
