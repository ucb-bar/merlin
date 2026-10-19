"""In-sandbox self-check SHIM — staged as <ws>/agent_selfcheck.py under --sandbox bwrap.

Same CLI as the real agent_selfcheck.py, but imports NOTHING from merlin (the oracle is masked in the
sandbox). It forwards the request to the driver-side broker via <ws>/.qa_channel and prints the redacted
verdict the broker returns. From the agent's view this is identical to running the self-check directly —
it gets pass/fail + its own artifacts (in ./selfcheck_out/) — but the oracle/goldens never enter the box.

A request names the tree it checks: ``submission/`` (the default, and the only tree graded officially) or
an author-side alternative at ``variants/<name>/``. A short ``--wait-budget`` returns a request id at once
so the author keeps working while the check runs, and ``--attach <id>`` collects it later.
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
_SUBMISSION = "submission"
_VARIANTS = "variants"
_VARIANT_NAME_MAX = 64
_NAME_FIRST = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_NAME_CHARS = _NAME_FIRST | frozenset("._-")
#: Completed requests ``--list`` shows beside the in-flight ones.
_LIST_COMPLETED = 5

_WORKFLOW = """\
keep working while checks run:
  python agent_selfcheck.py --capsules A,B,C --wait-budget 5   # returns a request id (exit 3: running)
  ...keep editing; each check grades a snapshot taken when it starts...
  python agent_selfcheck.py --attach ID                         # collect the verdict
  python agent_selfcheck.py --list                              # requests in flight

Several requests may run at once. Put every capsule an edit affects in ONE request: its capsules run
in parallel across the CPU pool, and per-request setup dominates small checks. `--capsules failing`
re-checks what did not pass in your most recent completed check.

variants: copy submission/ to variants/NAME/, change it, and check it with --submission variants/NAME.
A variant result is labelled with its name and digest and is a self-check only: only submission/ is
graded officially or counts toward convergence, so copy the winner into submission/.
"""


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
    return (
        f"the grader runs up to {ordinary} self-check request(s) at once plus one diagnostic probe"
        + (f" (at most {variants} of them variants)" if isinstance(variants, int) else "")
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
            if isinstance(progress, dict):
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Agent self-check (redacted; routed to the driver-side broker).",
        epilog=_WORKFLOW,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--submission",
        default=_SUBMISSION,
        help="tree to check: submission (default; the only tree graded officially) or variants/NAME",
    )
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
    ap.add_argument(
        "--capsules",
        default="all",
        help="'all', 'failing', 'changed' (build and screen every public capsule, reuse exact-byte RTL "
        "certificates, simulate only those whose bytes are neither certified nor known to fail), or names",
    )
    ap.add_argument("--workers", type=int, default=0, help="0 uses the host's selected grading worker count")
    # This bounds the entire capsule check, including source setup, object builds,
    # the functional screen, native RTL execution, coherent readback and audits.
    # The selected gSIM native execution keeps its separate original 1800 s cap.
    # A default of 1800 here cut off the capacity convolution during coherent
    # readback after a 5-minute setup, even though native execution passed its
    # own limit in the automatic promotion path.
    ap.add_argument("--timeout", type=int, default=None)
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
    ap.add_argument(
        "--candidate-diagnostics", action="store_true", help="inspect original public emission without grading"
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="compile public scale/frontier tensor sources without executing a probe",
    )
    ap.add_argument(
        "--scale-probe",
        action="store_true",
        help="with --compile-only, inspect public scale/tail emission before object compilation",
    )
    ap.add_argument(
        "--dynamic-census", action="store_true", help="append bounded own-ELF activity after functional screening"
    )
    ap.add_argument(
        "--caller-layout",
        default="",
        help="inspect an emitted submission-relative command buffer's physical caller layout without grading",
    )
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
        help="seconds to block before returning in_progress with a resumable request id (0 = no bound); "
        "5 submits and returns at once so you can keep working",
    )
    ap.add_argument(
        "--attach",
        default="",
        help="collect (or keep waiting on) a request id instead of submitting a new one",
    )
    ap.add_argument("--list", action="store_true", help="list in-flight and recent requests, then exit")
    a = ap.parse_args(argv)
    if a.timeout is None:
        a.timeout = 3600 if a.sim == "gsim" else 1800
    if a.scale_probe and not a.compile_only:
        ap.error("--scale-probe requires --compile-only")

    policy_error = None if a.caller_layout or a.candidate_diagnostics or a.compile_only else _sim_policy_error(a.sim)
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
    if a.list:
        print(json.dumps(_list_requests(ch), indent=2))
        return 0
    rid = _request_id()
    requested_at_ns = time.time_ns()
    label, source_error = _source_label(ws, a.submission)
    original = None
    if a.attach:
        rid = str(a.attach)
        if Path(rid).name != rid or rid in {".", ".."}:
            source_error = f"--attach takes a request id as printed by a previous call, not a path: {rid!r}"
        else:
            original = _read_json(ch / f"req_{rid}.json")
        if isinstance(original, dict):
            # The request names its own tree and bytes; --submission on a resume is not consulted.
            label, source_error = _source_label(ws, original.get("submission", _SUBMISSION))
        elif source_error is None and not any(
            (ch / f"{prefix}{rid}{suffix}").exists()
            for prefix, suffix in (("progress_", ".json"), ("claimed_", ""), ("resp_", ".json"), ("done_", ""))
        ):
            source_error = f"no self-check request {rid!r} in this workspace; --list shows the ones in flight"
    elif label is not None and label != _SUBMISSION:
        source_error = _variant_error(ws, label)
    if source_error:
        txt = json.dumps({"error": source_error, "all_pass": False, "submission": a.submission})
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 2
    if isinstance(original, dict) and isinstance(original.get("submission_sha256"), str):
        submission_sha256 = original["submission_sha256"]  # what was asked about, not what is there now
    else:
        submission_sha256 = _submission_digest(ws / label)
    suite_size = None
    if a.capsules in ("all", "changed"):
        public_root = ws / "contract" / "capsules"
        discovered = sum(1 for _ in public_root.rglob("capsule.yaml")) if public_root.is_dir() else 0
        suite_size = discovered or _SUITE_SIZE_FALLBACK
    deadline = time.time() + _request_budget_seconds(
        a.timeout, capsules=a.capsules, workers=a.workers, suite_size=suite_size
    )
    if isinstance(original, dict) and isinstance(original.get("deadline_unix_ns"), int):
        deadline = original["deadline_unix_ns"] / 1_000_000_000
    # Resuming (--attach): the request is already on the channel and the broker is already working it.
    # Writing a second req_ for the same work would run a duplicate beside the first and double its cost.
    if not a.attach:
        # The Codex thread running this client (the lead or one of its sub-agents) exports its own id.
        # Forwarded for operator accounting only; the broker neither reads nor trusts it.
        caller_thread = os.environ.get("CODEX_THREAD_ID", "").strip()[:128]
        _atomic_write(
            ch / f"req_{rid}.json",
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "request_id": rid,
                    "requester": _requester_identity(),
                    **({"caller_thread_id": caller_thread} if caller_thread else {}),
                    "requested_at_unix_ns": requested_at_ns,
                    "submission": label,
                    "submission_sha256": submission_sha256,
                    "deadline_unix_ns": int(deadline * 1_000_000_000),
                    # Omitted entirely when the caller named none: an ABSENT key lets the driver
                    # resolve the engine, where a present one is a choice the broker must honour.
                    **({"sim": a.sim} if a.sim and not a.caller_layout else {}),
                    **({"tiers": a.tiers} if a.tiers else {}),
                    "capsules": a.capsules,
                    "workers": a.workers,
                    "timeout": a.timeout,
                    "shape_coverage": bool(a.shape_coverage),
                    "offload_census": bool(a.offload_census),
                    "model_layers": bool(a.model_layers),
                    "candidate_diagnostics": bool(a.candidate_diagnostics),
                    "compile_only": bool(a.compile_only),
                    "scale_probe": bool(a.scale_probe),
                    "dynamic_census": bool(a.dynamic_census),
                    "caller_layout": a.caller_layout,
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
        # WHETHER THE WAIT IS A QUEUE OR A FAULT. A request waits for a free grader slot; without this
        # the two look identical from here, and an agent that read "no file at all" concluded the broker
        # was down and cancelled four queued requests in a row. How many slots there are is the
        # grader's own record (one, unless a shared CPU pool sizes several), never assumed here.
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
        if not announced_claim and time.time() - waiting_since >= 15 and time.time() - last_waiting_note > 60:
            last_waiting_note = time.time()
            print(
                f"[self-check] queued, not yet claimed ({time.time() - waiting_since:.0f}s); {_capacity_note(ch)}",
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
            _ok, _why = _reply_is_usable(_v, rid, submission_sha256, label=label)
            if not _ok:
                print(json.dumps({"error": f"self-check reply {_why} — treating as FAILED, not as stale feedback"}))
                return 2
            if _why:
                print(json.dumps({"note": _why}))
            graded = _v.get("submission_sha256")
            if isinstance(graded, str) and graded and _submission_digest(ws / label) != graded:
                # The check graded a snapshot; say so when the tree has moved on since, which is the
                # normal case for a check collected with --attach after more editing.
                print(json.dumps({"note": f"{label}/ has changed since this check's snapshot ({graded[:12]})"}))
            if label != _SUBMISSION:
                print(
                    f"[self-check] {label} ({str(graded)[:12]}): variant self-check only -- not graded "
                    "officially and not counted toward convergence; copy the winner into submission/",
                    file=sys.stderr,
                    flush=True,
                )
            # the shape-coverage report has no `all_pass`; its verdict is `all_covered`
            if a.shape_coverage:
                return 0 if _v.get("all_covered") else 1
            if a.candidate_diagnostics:
                return 0 if _v.get("schema") == "merlin.public_candidate_diagnostics.v1" else 2
            if a.compile_only:
                return 0 if _v.get("schema") == "merlin.public_compile_only_feedback.v1" and _v.get("cases") else 2
            if _v.get("status") == "layout_only":
                supported = {"public_caller_layout_receipt_v1", "public_caller_layout_receipt_v2"}
                return 0 if _v.get("schema") in supported else 2
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
        try:
            # Returning on purpose: the request now lives on its deadline, not on this process, so the
            # broker neither drops it as abandoned nor aborts it when this process exits.
            _atomic_write(ch / f"detached_{rid}", json.dumps({"detached_at_unix_ns": time.time_ns()}))
        except OSError:
            pass
        print(
            json.dumps(
                {
                    "status": "in_progress",
                    "request_id": rid,
                    "submission": label,
                    "waited_s": round(time.time() - waiting_since, 1),
                    "claimed": claimed.exists(),
                    "n_finished": snapshot.get("n_finished"),
                    "n_expected": snapshot.get("n_expected"),
                    "n_remaining": snapshot.get("n_remaining"),
                    "counts": snapshot.get("counts", {}),
                    "resume_with": f"python agent_selfcheck.py --attach {rid}" + (f" --out {a.out}" if a.out else ""),
                    "note": (
                        "the check is STILL RUNNING broker-side; this is not a pass, a failure or a "
                        "timeout. Keep working and collect it with the command above -- do NOT write a "
                        "sleep/poll loop, and do not resubmit this same check (that only repeats it); "
                        "checks of other capsules or variants may run alongside it."
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
