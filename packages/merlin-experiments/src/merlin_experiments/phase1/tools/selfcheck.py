"""In-sandbox self-check SHIM — staged as <ws>/agent_selfcheck.py under --sandbox bwrap.

Same CLI as the real agent_selfcheck.py, but imports NOTHING from merlin (the oracle is masked in the
sandbox). It forwards the request to the driver-side broker via <ws>/.qa_channel and prints the redacted
verdict the broker returns. From the agent's view this is identical to running the self-check directly —
it gets pass/fail + its own artifacts (in ./selfcheck_out/) — but the oracle/goldens never enter the box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time
from pathlib import Path

_SIMS = ("spike", "verilator", "gsim", "vcs")
PROTOCOL_VERSION = 3
_DIGEST_IGNORES = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})
_SUITE_SIZE_FALLBACK = 64
_CALIBRATION_CAP = 3


#: Seconds this shim blocks before returning a resumable ``in_progress``. MUST stay under the calling
#: agent's 600s shell ceiling: at or above it the process is killed instead of returning, which is the
#: condition that produced the poll loops (see test_selfcheck_wait_is_resumable).
_DEFAULT_WAIT_BUDGET_S = 540


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


def _reply_is_usable(verdict: dict, rid: str, submission_sha256: str) -> tuple:
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
    """
    if verdict.get("selfcheck_protocol") != PROTOCOL_VERSION:
        return False, "carries the wrong protocol version"
    if verdict.get("selfcheck_request_id") != rid:
        return False, "belongs to a different request"
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
    has a serial calibration head followed by several parallel worker waves.  Mirror that scheduling
    shape here.  The bound is deliberately conservative: an individual capsule normally finishes far
    below its timeout, so successful requests still return promptly.
    """
    per_capsule = max(1, int(timeout))
    worker_count = max(1, int(workers))
    if capsules == "all":
        count = max(1, int(suite_size or _SUITE_SIZE_FALLBACK))
    else:
        count = max(1, len({name.strip() for name in capsules.split(",") if name.strip()}))
    serial = min(_CALIBRATION_CAP, count)
    parallel_waves = math.ceil(max(0, count - serial) / worker_count)
    return (serial + parallel_waves) * per_capsule + 240


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agent self-check (redacted; routed to the driver-side broker).")
    ap.add_argument("--submission", default="submission")
    # DEFAULTS TO THE CERTIFYING SIM, not the screen. The capsules declare a cycle-accurate cert
    # tier as mandatory, and this ladder runs cheapest-measured-first with fail-fast, so the
    # screen still refutes a broken submission at screen cost -- what changes is that a capsule
    # which PASSES the screen goes on to certify instead of stopping there. Choosing the screen
    # explicitly (--sim/--tiers) is a legitimate fast loop, but it CANNOT certify: the mandatory
    # cert tier reports unavailable and the capsule is not a pass.
    # WHICH certifying engine is NOT decided here -- see _default_sim.
    ap.add_argument("--sim", choices=list(_SIMS), default=_default_sim())
    # Forwarded like every other flag. The real self-check has accepted --tiers for a while; this
    # shim never did, so the one way to ask for the cheap tier by NAME (rather than by naming a
    # simulator, which only some targets let you select) was unreachable from inside the box.
    ap.add_argument("--tiers", default="")
    ap.add_argument("--capsules", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="")
    # Forwarded like every other flag. Without this the sandboxed agent -- which is EVERY real run --
    # gets "unrecognized arguments" and the shape-coverage report is unreachable exactly where it
    # matters, while working fine when tested outside the box.
    ap.add_argument("--shape-coverage", action="store_true")
    # The same reason, twice more. `--offload-census` existed for weeks as a probe a sandboxed agent
    # could not reach: it was added to the real self-check and never to this shim, so every real run
    # got "unrecognized arguments" for it.
    ap.add_argument("--offload-census", action="store_true")
    ap.add_argument("--model-layers", action="store_true")
    # WALL-CLOCK BOUND, and the reason it exists is measured. The caller of this shim is an agent whose
    # shell tool kills any command at 600s. A full-suite check runs 2.4-14.7 min, so the blocking wait
    # below was routinely killed mid-flight: the request stayed alive broker-side, the agent got back
    # only "moved to the background", and with no way to re-attach it hand-wrote `until ... sleep` poll
    # loops -- which the same 600s ceiling then truncated too. Measured on one run: six such calls,
    # 59.4 min, returning no information whatsoever, against a task brief that says "WAIT, DO NOT POLL"
    # three times. Prose lost to a mechanism, so this is the mechanism.
    #
    # Default sits under that ceiling. On expiry the shim RETURNS, printing the request id and the
    # progress it has, and the caller resumes with --attach. Three informative turns beat six empty ones.
    ap.add_argument(
        "--wait-budget",
        type=int,
        default=_DEFAULT_WAIT_BUDGET_S,
        help="seconds to block before returning in_progress with a resumable request id (0 = no bound)",
    )
    ap.add_argument(
        "--attach",
        default="",
        help="resume waiting on an in-flight request id instead of submitting a new one",
    )
    a = ap.parse_args(argv)

    policy_error = _sim_policy_error(a.sim)
    if policy_error:
        txt = json.dumps(
            {"error": policy_error, "all_pass": False, "sim": a.sim, "required_rtl_engine": _required_rtl_engine()}
        )
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 2

    ws = Path(__file__).resolve().parent  # the shim lives at <ws>/agent_selfcheck.py
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    rid = _request_id()
    requested_at_ns = time.time_ns()
    submission = Path(a.submission)
    if not submission.is_absolute():
        submission = ws / submission
    submission_sha256 = _submission_digest(submission)
    suite_size = None
    if a.capsules == "all":
        public_root = ws / "contract" / "capsules"
        discovered = sum(1 for _ in public_root.rglob("capsule.yaml")) if public_root.is_dir() else 0
        suite_size = discovered or _SUITE_SIZE_FALLBACK
    deadline = time.time() + _request_budget_seconds(
        a.timeout, capsules=a.capsules, workers=a.workers, suite_size=suite_size
    )
    if a.attach:
        # Resuming: the request is already on the channel and the broker is already working it. Writing
        # a second req_ for the same work would queue a duplicate behind the first and double the cost
        # of the very wait we are trying to shorten.
        rid = str(a.attach)
    else:
        _atomic_write(
            ch / f"req_{rid}.json",
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "request_id": rid,
                    "requester": _requester_identity(),
                    "requested_at_unix_ns": requested_at_ns,
                    "submission_sha256": submission_sha256,
                    "deadline_unix_ns": int(deadline * 1_000_000_000),
                    # Omitted entirely when the caller named none: an ABSENT key lets the driver
                    # resolve the engine, where a present one is a choice the broker must honour.
                    **({"sim": a.sim} if a.sim else {}),
                    **({"tiers": a.tiers} if a.tiers else {}),
                    "capsules": a.capsules,
                    "workers": a.workers,
                    "timeout": a.timeout,
                    "shape_coverage": bool(a.shape_coverage),
                    "offload_census": bool(a.offload_census),
                    "model_layers": bool(a.model_layers),
                }
            ),
        )
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    progress = ch / f"progress_{rid}.json"
    claimed = ch / f"claimed_{rid}"
    last_progress_finished = -1
    announced_claim = False
    waiting_since = time.time()
    last_waiting_note = 0.0
    # The bound that makes this survivable by a 600s-capped caller. `deadline` still governs how long
    # the REQUEST may live broker-side; this only governs how long THIS invocation blocks.
    soft_deadline = time.time() + a.wait_budget if a.wait_budget > 0 else deadline
    while time.time() < min(deadline, soft_deadline):
        # WHETHER THE WAIT IS A QUEUE OR A FAULT. One request is served at a time; without this the
        # two look identical from here, and an agent that read "no file at all" concluded the broker
        # was down and cancelled four queued requests in a row.
        if not announced_claim and claimed.exists():
            announced_claim = True
            try:
                behind = json.loads(claimed.read_text(encoding="utf-8")).get("queued_behind_this")
            except (OSError, ValueError, TypeError):
                behind = None
            print(
                f"[self-check] claimed by the grader after {time.time() - waiting_since:.0f}s"
                + (f"; {behind} request(s) were queued behind it" if isinstance(behind, int) else ""),
                file=sys.stderr,
                flush=True,
            )
        if not announced_claim and time.time() - last_waiting_note > 60:
            last_waiting_note = time.time()
            print(
                f"[self-check] queued, not yet claimed ({time.time() - waiting_since:.0f}s); the grader "
                f"serves one request at a time and yours is waiting for the slot",
                file=sys.stderr,
                flush=True,
            )
        if progress.exists():
            try:
                p = json.loads(progress.read_text(encoding="utf-8"))
                finished = int(p.get("n_finished", 0))
                if finished != last_progress_finished:
                    # stderr keeps stdout machine-readable while making a long foreground check visibly
                    # alive in an interactive terminal.
                    print(
                        f"[self-check] {finished}/{p.get('n_expected', '?')} finished; counts={p.get('counts', {})}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress_finished = finished
            except (OSError, ValueError, TypeError):
                pass
        if done.exists() and resp.exists():
            txt = resp.read_text()
            print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _v = _verdict(txt)
            if _v is None:
                # FAIL CLOSED. This used to `return 0` -- so a reply the shim could not read was
                # indistinguishable from a clean run, and every exit-code check downstream (the agent's
                # own, the conformance probe, the shape-coverage gate below) silently read as a pass.
                print(
                    json.dumps(
                        {"error": "self-check reply was not parseable as JSON — treating as FAILED, not as clean"}
                    )
                )
                return 2
            _ok, _why = _reply_is_usable(_v, rid, submission_sha256)
            if not _ok:
                print(json.dumps({"error": f"self-check reply {_why} — treating as FAILED, not as stale feedback"}))
                return 2
            if _why:
                print(json.dumps({"note": _why}))
            # the shape-coverage report has no `all_pass`; its verdict is `all_covered`
            if a.shape_coverage:
                return 0 if _v.get("all_covered") else 1
            return 0 if _v.get("all_pass") else 1
        time.sleep(0.4)
    if a.wait_budget > 0 and time.time() < deadline:
        # NOT an error and NOT a verdict: the check is still running and this invocation simply spent
        # its wall-clock budget. Everything needed to resume is printed, so the next turn costs one
        # command and returns more progress -- instead of the caller being killed at 600s holding
        # nothing, which is what produced the poll loops this budget exists to prevent.
        snapshot = {}
        if progress.exists():
            try:
                snapshot = json.loads(progress.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                snapshot = {}
        print(
            json.dumps(
                {
                    "status": "in_progress",
                    "request_id": rid,
                    "waited_s": round(time.time() - waiting_since, 1),
                    "claimed": claimed.exists(),
                    "n_finished": snapshot.get("n_finished"),
                    "n_expected": snapshot.get("n_expected"),
                    "n_remaining": snapshot.get("n_remaining"),
                    "counts": snapshot.get("counts", {}),
                    "resume_with": f"python agent_selfcheck.py --attach {rid}" + (f" --out {a.out}" if a.out else ""),
                    "note": (
                        "the check is STILL RUNNING broker-side; this is not a pass, a failure or a "
                        "timeout. Resume with the command above -- do NOT write a sleep/poll loop and "
                        "do NOT submit a second request, which would queue behind this one."
                    ),
                },
                indent=2,
            )
        )
        return 3
    print(json.dumps({"error": "self-check broker did not respond (timeout) — tell the operator"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
