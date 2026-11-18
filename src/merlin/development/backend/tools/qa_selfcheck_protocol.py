"""Standalone self-check request identity, admission and channel protocol; standard library only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from pathlib import Path, PurePosixPath
import sys

# ---- qa_selfcheck_protocol ---------------------------------------------------------------------

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

_LIST_COMPLETED = 5

_PANEL = "panel:"

_PANEL_CASES = 6

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


def _case_rows(progress) -> list:
    rows = progress.get("per_capsule") if isinstance(progress, dict) else None
    return [row for row in rows or () if isinstance(row, dict) and isinstance(row.get("capsule"), str)]


def _case_line(row: dict) -> str:
    """One case's correctness as the grader has decided it so far."""
    name, status = row["capsule"], str(row.get("status"))
    if status == "running":
        return (
            f"[self-check] {name}: numerics {row.get('numerics')} (provisional: later checks still "
            "running and can still fail it)"
        )
    why = "/".join(str(row[key]) for key in ("failure_plane", "failure_category") if row.get(key))
    numerics = f"; numerics {row['numerics']}" if row.get("numerics") else ""
    return f"[self-check] {name}: {status}" + (f" ({why})" if why else "") + numerics


def _new_case_lines(seen: dict, progress) -> list:
    """Lines for cases whose row is new or changed since ``seen`` (updated in place).

    Each case is reported as soon as its numerics are decided and again when it finishes, so a large
    suite never hides its decided cases behind its slowest one."""
    lines = []
    for row in _case_rows(progress):
        state = tuple(row.get(key) for key in ("status", "numerics", "failure_plane", "failure_category"))
        if seen.get(row["capsule"]) != state:
            seen[row["capsule"]] = state
            lines.append(_case_line(row))
    return lines


_WORKFLOW = """\
keep working while checks run:
  python agent_selfcheck.py --capsules A,B,C --wait-budget 5   # returns a request id (exit 3: running)
  ...keep editing; each check grades a snapshot taken when it starts...
  python agent_selfcheck.py --attach ID                         # collect the verdict
  python agent_selfcheck.py --list                              # requests in flight

Operator-selected connected model sources use `--suite programs --sim spike --tiers L0,L1,L2`.
The original model owner checks all outputs; programs never acquire core or RTL credit.

Several requests may run at once. Put every capsule an edit affects in ONE request: its capsules run
in parallel across the CPU pool, and per-request setup dominates small checks. `--capsules failing`
re-checks what did not pass in your most recent completed check.

Before changing a default that governs many shapes: `python agent_selfcheck.py --panel dense
--wait-budget 5` checks one case per shape class (square, tall, wide, thin M=1, ragged, a model
layer) on the Spike cycle model and compares each with its own best earlier cycles of the same
evidence kind. Performance is measured on the Spike cycle model by default; add --sim gsim for RTL cycles.

variants: copy submission/ to variants/NAME/, change it, and check it with --submission variants/NAME.
A variant result is labelled with its name and digest and is a self-check only: only submission/ is
graded officially or counts toward convergence, so copy the winner into submission/.

Time alternatives together: --sweep variants/a,variants/b --capsules A,B [--knob NAME=false,true].
This submits parallel scalability requests, including submission/ as the baseline, and returns ranked
model estimates. --bundle REPRESENTATIVE selects a bundle in qa/idle_tuning.json. --attach SWEEP_ID
collects the same sweep later; full results and immutable source identities are retained.

When reference guidance is selected, pin completed source/command/emission views without a broker call:
  python agent_selfcheck.py --write-reference-views artifacts/views.json artifacts/source.mlir \\
    artifacts/commands.json artifacts/emitted.mlir
  python agent_selfcheck.py --reference-analysis candidate --reference-path artifacts/views.json
The source is the original interface contract; emitted LLVM is not schedule.plan. Inspect an actual
schedule separately with --reference-format schedule_mlir. Compare owned before/after disassemblies:
  python agent_selfcheck.py --reference-analysis compare --reference-path artifacts/before.dis \\
    --reference-symbol kernel_entry --candidate-path artifacts/after.dis --candidate-symbol kernel_entry
Verify your own change from ordinary selfcheck outputs (CCA bank arm): traffic, storage, path, sweep:
  python agent_selfcheck.py --reference-analysis verify --before selfcheck_out/requests/OLD \\
    --after selfcheck_out/requests/NEW [--mechanism issue_overlap] [--sweep selfcheck_out/requests/NEW2] \\
    [--candidate-symbol kernel_entry]
Stream facts come from each build's package_kernel.elf, else one *.o you put in the case directory (for
example a -mno-relax rebuild), else its lowered.llvm.mlir compiled host-side with -mno-relax.
Recovered ISA operands appear under observation_comparison with explicit missingness. To inspect
one field, repeat --reference-observation-field Instruction.field (at most eight); use field names
from the selected ISA/decoded observation roster. Full evidence remains in the immutable detail.
Pins and static comparisons do not qualify source equivalence, scheduling realization or performance.
"""


_DEFAULT_WAIT_BUDGET_S = 540


_NEXT_STEPS_KEY = "next_steps"


_NEXT_STEPS_LINES = 8


_NEXT_STEPS_BYTES = 6000


def _next_steps_text(verdict) -> str:
    """The reply's ``next_steps`` lines as one bounded block, or '' when it carries none."""
    block = verdict.get(_NEXT_STEPS_KEY) if isinstance(verdict, dict) else None
    lines = block.get("lines") if isinstance(block, dict) else None
    if not isinstance(lines, list):
        return ""
    kept, size = [], 0
    for line in lines[:_NEXT_STEPS_LINES]:
        if not isinstance(line, str) or not line.strip():
            continue
        text = " ".join(line.split())
        if size + len(text.encode()) + 1 > _NEXT_STEPS_BYTES:
            break
        kept.append(text)
        size += len(text.encode()) + 1
    return "\n".join(kept) if len(kept) > 1 else ""


def _print_next_steps(verdict) -> None:
    """Print the next steps on stderr: shown in a terminal even when stdout is piped to a JSON reader,
    and never appended to the JSON a script parses from stdout."""
    text = _next_steps_text(verdict)
    if text:
        print("\n[self-check] " + text, file=sys.stderr, flush=True)


def _write_reference_views(workspace, arguments, *, entry_symbol="kernel_entry", client_name="agent_selfcheck.py"):
    """Pin existing author-owned artifacts; never execute or qualify their code."""
    workspace = Path(workspace).resolve()

    def owned(value):
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or str(relative) != value
            or "\\" in value
            or any(part in (".", "..") for part in relative.parts)
            or not relative.parts
            or relative.parts[0] not in ("submission", "variants", "selfcheck_out", "artifacts")
        ):
            raise ValueError("use workspace-relative submission/, variants/, selfcheck_out/ or artifacts/ paths")
        path = workspace / relative
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(workspace):
            raise ValueError("artifact views require ordinary author-owned paths")
        return path, relative.as_posix()

    if not entry_symbol or len(entry_symbol) > 512:
        raise ValueError("entry symbol must be nonempty and bounded")
    destination, label = owned(arguments[0])
    bundle = dict(schema="merlin.reference_artifact_views.v1", entry_symbol=entry_symbol)
    for role, value in zip(("source", "command_buffer", "emitted"), arguments[1:], strict=True):
        path, relative = owned(value)
        before = path.stat()
        if not path.is_file() or not 0 < before.st_size <= 8 << 20 or path == destination:
            raise ValueError("each completed artifact must be an ordinary nonempty file <=8MiB")
        raw = path.read_bytes()
        after = path.stat()
        if (
            any(
                getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            )
            or len(raw) != before.st_size
        ):
            raise ValueError("artifact changed while pinning; finish compilation and retry")
        bundle[role] = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return dict(
        status="artifact_views_written",
        manifest=label,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        qualification_issued=False,
        next_argv=[
            sys.executable,
            Path(client_name).name,
            "--reference-analysis",
            "candidate",
            "--reference-path",
            label,
            "--reference-format",
            "artifact_views",
        ],
        note="Pins identify supplied bytes only. Source binding, emitted mechanism, numerics and timing "
        "require the selected analysis and original executable checks.",
    )


def _analysis_summary_line(text, rid):
    """``status``, report kind, unit/region counts and the detail file of one reference-analysis reply."""
    try:
        reply = json.loads(text)
    except ValueError:
        return f"[reference-analysis] reply .qa_channel/resp_{rid}.json is not JSON"
    report = reply.get("report") if isinstance(reply.get("report"), dict) else {}
    parts = [f"status={reply.get('status')}", f"kind={report.get('kind')}"]
    for key in ("total", "next_offset"):
        if report.get(key) is not None:
            parts.append(f"{key}={report[key]}")
    units = report.get("units") if isinstance(report.get("units"), list) else []
    for row in units[:4]:
        mechanisms = row.get("mechanisms") or {}
        known = sorted(
            name
            for name, fact in mechanisms.items()
            if isinstance(fact, dict) and fact.get("status") not in (None, "unknown")
        )
        parts.append(f"{str(row.get('unit_id'))[:20]}: observed {','.join(known) or 'none'}")
    detail = reply.get("detail")
    if isinstance(detail, dict):
        detail = detail.get("path")
    parts.append(f"full reply .qa_channel/resp_{rid}.json" + (f", detail {detail}" if detail else ""))
    return "[reference-analysis] " + "; ".join(parts)
