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
from pathlib import Path, PurePosixPath

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
#: ``--panel FAMILY`` travels as the capsule selector ``panel:FAMILY``; it grades one case per shape class.
_PANEL = "panel:"
_PANEL_CASES = 6

#: Progress a child publishes while it waits for a gSIM correctness slot (the broker's
#: ``joint_gsim_queue.WAITING_STATUS``; restated because this shim imports nothing from merlin).
_GSIM_QUEUE_STATUS = "queued_for_gsim_slot"
_SWEEP_SCHEMA = "merlin.selfcheck_sweep.v1"
_EXPERIMENT_SCHEMA = "merlin.idle_tuning_experiment.v1"
_MODEL_KEYS = ("selection_sha256", "model_sha256", "timing_mode", "timing_epoch",
               "scope_sha256", "comparison_domain_sha256")


def _gsim_queue(progress) -> dict | None:
    """The queue block of a request waiting for a gSIM slot, else ``None``."""
    if isinstance(progress, dict) and progress.get("status") == _GSIM_QUEUE_STATUS:
        queue = progress.get("gsim_slot_queue")
        return queue if isinstance(queue, dict) else {}
    return None


def _gsim_queue_note(queue: dict) -> str:
    return f"queued for a gSIM slot ({queue.get('ahead', '?')} ahead)"


def _case_rows(progress) -> list:
    rows = progress.get("per_capsule") if isinstance(progress, dict) else None
    return [row for row in rows or () if isinstance(row, dict) and isinstance(row.get("capsule"), str)]


def _case_line(row: dict) -> str:
    """One case's correctness as the grader has decided it so far."""
    name, status = row["capsule"], str(row.get("status"))
    if status == "running":
        return (f"[self-check] {name}: numerics {row.get('numerics')} (provisional: later checks still "
                "running and can still fail it)")
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
Pins and static comparisons do not qualify source equivalence, scheduling realization or performance.
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


#: The reply's ranked next steps (``feedback.next_steps``): printed after the reply, bounded.
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


def _sweep_candidates(ws: Path, variants: str, knobs: list[str]) -> list[dict]:
    """Validate the complete candidate list before making any request visible."""
    candidates = []
    for value in variants.split(",") if variants else ():
        label, error = _source_label(ws, value.strip())
        error = error or (_variant_error(ws, label) if label != _SUBMISSION else "--sweep names variants/")
        if error:
            raise ValueError(error)
        candidates.append({"submission": label, "key": "variant:" + label.partition("/")[2]})
    if knobs:
        declared = _read_json(ws / _SUBMISSION / "TUNING_KNOBS.json")
        if not isinstance(declared, dict) or declared.get("schema") != "merlin.tuning_knobs.v1":
            raise ValueError("--knob needs submission/TUNING_KNOBS.json (merlin.tuning_knobs.v1)")
        for text in knobs:
            name, equals, raw = text.partition("=")
            try:
                values = json.loads("[" + raw + "]") if equals else []
            except ValueError as exc:
                raise ValueError("--knob takes NAME=V1,V2 with JSON scalar values") from exc
            spec = (declared.get("knobs") or {}).get(name)
            allowed = spec.get("values") if isinstance(spec, dict) else None
            if not values or not isinstance(allowed, list):
                raise ValueError("--knob needs a declared knob and at least one JSON scalar value")
            for value in values:
                if not any(type(value) is type(v) and value == v for v in allowed):
                    raise ValueError(f"--knob {name}: value is not declared in TUNING_KNOBS.json")
                candidates.append({"submission": _SUBMISSION, "tuning_overrides": {name: value},
                                   "key": "knob:" + name + "=" + json.dumps(value)})
    unique = {c["key"]: c for c in candidates}
    if not unique:
        raise ValueError("--sweep needs variant trees or --knob NAME=V1,V2")
    return list(unique.values())


def _sweep_cases(ws: Path, capsules: str, bundle: str) -> str:
    if bundle:
        if capsules != "all":
            raise ValueError("--bundle selects its own cases; drop --capsules")
        ledger = _read_json(ws / "qa" / "idle_tuning.json") or {}
        chosen = next((b for b in ledger.get("bundles") or () if b.get("representative") == bundle), None)
        if chosen is None:
            raise ValueError("no matching bundle in qa/idle_tuning.json; use --capsules X,Y")
        capsules = ",".join([chosen["representative"], *chosen.get("siblings", [])])
    names = [name.strip() for name in capsules.split(",") if name.strip()]
    if not names or any(name in ("all", "changed", "failing") or ":" in name for name in names):
        raise ValueError("--sweep needs named --capsules X,Y or --bundle REPRESENTATIVE")
    return ",".join(dict.fromkeys(names))


def _sweep_rows(ws: Path, document: dict) -> dict:
    """Read verified public detail pointers; incomplete or altered details never supply cycle evidence."""
    found = {}
    for row in document.get("cases") or document.get("per_capsule") or ():
        if not isinstance(row, dict):
            continue
        detail = row.get("detail")
        if isinstance(detail, dict):
            raw_path = detail.get("path")
            path = ws / raw_path if isinstance(raw_path, str) else ws
            try:
                path.resolve().relative_to(ws.resolve())
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != detail.get("sha256"):
                    continue
                full = json.loads(data)
                if not isinstance(full, dict) or full.get("capsule") != row.get("capsule"):
                    continue
                row = full
            except (OSError, ValueError):
                continue
        name = row.get("capsule")
        if isinstance(name, str):
            found[name] = row
    return found


def _sweep_passed(row: dict) -> bool:
    tier = (row.get("tiers") or {}).get("L2")
    if tier is not None:
        return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"
    return row.get("pass") is True or row.get("status") == "pass"


def _sweep_rank(ws: Path, manifest: dict, responses: dict) -> tuple[list, list]:
    """Compare each member with this sweep's baseline only, under identical case/model scopes."""
    baseline = responses.get(manifest["baseline_request_id"]) or {}
    before_rows = _sweep_rows(ws, baseline)
    comparisons, rankings = [], []
    for request in manifest["requests"][1:]:
        document = responses.get(request["request_id"])
        if not isinstance(document, dict):
            continue
        group = []
        candidate_sha = (document.get("variant") or {}).get("sha256") or document.get("submission_sha256")
        for name, row in _sweep_rows(ws, document).items():
            before = before_rows.get(name) or {}
            bp, cp = before.get("performance") or {}, row.get("performance") or {}
            before_cycles, after_cycles = bp.get("estimated_whole_soc_cycles"), cp.get("estimated_whole_soc_cycles")
            reason = next(("estimates differ in " + k for k in _MODEL_KEYS if bp.get(k) != cp.get(k)), None)
            required = ("model_sha256", "timing_mode", "scope_sha256", "comparison_domain_sha256", "elf_sha256")
            if reason is None and (any(not p.get(k) for p in (bp, cp) for k in required)
                                   or any(type(p.get("timing_epoch")) is not int for p in (bp, cp))):
                reason = "incomplete model, case scope or executable identity"
            if any(type(c) not in (int, float) or not math.isfinite(c) or c <= 0
                   for c in (before_cycles, after_cycles)):
                reason = "no complete model estimate on both sides"
            unchanged = _sweep_passed(before) and _sweep_passed(row)
            identity = {"candidate_sha256": candidate_sha, "baseline_sha256": baseline.get("submission_sha256"),
                        "case": name, "candidate_elf_sha256": cp.get("elf_sha256"),
                        "baseline_elf_sha256": bp.get("elf_sha256"),
                        "model": {k: cp.get(k) for k in _MODEL_KEYS},
                        "baseline_model": {k: bp.get(k) for k in _MODEL_KEYS}}
            comparison = {"test_case": name, "candidate": request["key"], "identity": identity,
                          "numerics": "unchanged" if unchanged else "changed_or_unknown",
                          "cycles_before": before_cycles, "cycles_after": after_cycles,
                          "delta_pct": round(100 * (after_cycles / before_cycles - 1), 2) if reason is None else None,
                          "ratio": after_cycles / before_cycles if reason is None else None,
                          "request_id": request["request_id"]}
            if reason:
                comparison["not_compared"] = reason
            comparisons.append(comparison)
            group.append(comparison)
        usable = [c for c in group if c["numerics"] == "unchanged" and c["ratio"] is not None]
        rankings.append({"candidate": request["key"], "candidate_sha256": candidate_sha,
                         "geomean_ratio": round(math.exp(sum(math.log(c["ratio"]) for c in usable)
                                                          / len(usable)), 4) if usable else None,
                         "cases_compared": len(usable), "faster": sum(c["ratio"] < 1 for c in usable),
                         "slower": sum(c["ratio"] > 1 for c in usable),
                         "numerics_changed_or_unknown": sum(c["numerics"] != "unchanged" for c in group),
                         "error": document.get("error"), "request_id": request["request_id"]})
    rankings.sort(key=lambda r: (r["numerics_changed_or_unknown"] > 0, r["geomean_ratio"] is None,
                                 r["geomean_ratio"] or 0, r["candidate"]))
    comparisons.sort(key=lambda c: (c["numerics"] != "unchanged", c["delta_pct"] is None,
                                   c["delta_pct"] or 0, c["test_case"]))
    return rankings, comparisons


def _run_sweep(a, ws: Path, ch: Path, manifest: dict | None = None) -> int:
    if manifest is None:
        candidates = _sweep_candidates(ws, a.sweep or "", a.knob)
        cases = _sweep_cases(ws, a.capsules, a.bundle)
        sid = "sweep_" + _request_id()
        base = _submission_digest(ws / _SUBMISSION)
        now = time.time_ns()
        deadline = time.time() + _request_budget_seconds(a.timeout, capsules=cases, workers=a.workers)
        caller = os.environ.get("CODEX_THREAD_ID", "").strip()[:128]
        requests, bodies = [], []
        for index, candidate in enumerate([{"submission": _SUBMISSION, "key": "baseline"}, *candidates]):
            label, rid = candidate["submission"], f"{sid}_{index}"
            digest = base if label == _SUBMISSION else _submission_digest(ws / label)
            member = {**candidate, "request_id": rid, "submission_sha256": digest}
            requests.append(member)
            body = {"protocol": PROTOCOL_VERSION, "request_id": rid, "requested_at_unix_ns": now,
                    "requester": _requester_identity(), "deadline_unix_ns": int(deadline * 1e9),
                    "submission": label, "submission_sha256": digest, "suite": "scalability", "sim": "spike",
                    "capsules": cases, "workers": a.workers, "timeout": a.timeout,
                    **({"caller_thread_id": caller} if caller else {})}
            if index:
                kind = "knob" if candidate.get("tuning_overrides") else "variant"
                body["tuning_experiment"] = {"schema": _EXPERIMENT_SCHEMA, "base_sha256": base,
                    "candidate": {"kind": kind, "key": candidate["key"]}, "test_cases": cases.split(","),
                    "origin": "author_sweep", "sweep": sid}
                if kind == "knob":
                    body["tuning_overrides"] = candidate["tuning_overrides"]
            bodies.append(body)
        manifest = {"schema": _SWEEP_SCHEMA, "sweep_id": sid, "baseline_request_id": requests[0]["request_id"],
                    "requests": requests, "capsules": cases, "deadline_unix_ns": int(deadline * 1e9)}
        _atomic_write(ch / f"{sid}.json", json.dumps(manifest))
        for body in bodies:
            rid = body["request_id"]
            _atomic_write(ch / f"detached_{rid}", json.dumps({"sweep": sid, "at_unix_ns": now}))
            _atomic_write(ch / f"req_{rid}.json", json.dumps(body))
    sid = manifest["sweep_id"]
    until = manifest["deadline_unix_ns"] / 1e9
    if a.wait_budget > 0:
        until = min(until, time.time() + a.wait_budget)
    responses, faults = {}, {}
    while True:
        for request in manifest["requests"]:
            rid = request["request_id"]
            if rid in responses or rid in faults or not (ch / f"done_{rid}").exists():
                continue
            doc = _read_json(ch / f"resp_{rid}.json")
            if not isinstance(doc, dict):
                faults[rid] = "unparseable self-check response"
                continue
            ok, why = _reply_is_usable(doc, rid, request["submission_sha256"], label=request["submission"])
            if ok and request.get("tuning_overrides"):
                variant = doc.get("variant") or {}
                if (variant.get("kind") != "knob_overrides" or variant.get("applied") is not True
                        or variant.get("tuning_overrides") != request["tuning_overrides"]):
                    ok, why = False, "response does not bind the requested private knob copy"
            if ok:
                responses[rid] = doc
            else:
                faults[rid] = why
        complete = len(responses) + len(faults) == len(manifest["requests"])
        if complete or time.time() >= until:
            break
        time.sleep(0.4)
    rankings, comparisons = _sweep_rank(ws, manifest, responses)
    path = ws / "selfcheck_out" / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": _SWEEP_SCHEMA, "sweep_id": sid, "official": False, "advisory_only": True,
              "status": "complete" if complete else "in_progress", "n_completed": len(responses),
              "baseline_status": ("current" if _submission_digest(ws / _SUBMISSION) ==
                  (responses.get(manifest["baseline_request_id"]) or {}).get("submission_sha256") else "older_source"),
              "n_expected": len(manifest["requests"]), "errors": faults,
              "all_pass": (all(d.get("all_pass") is True for d in responses.values())
                           if complete and not faults else None),
              "ranked_candidates": rankings, "comparisons": comparisons,
              "responses": [f".qa_channel/resp_{r['request_id']}.json" for r in manifest["requests"]],
              "resume_with": f"python agent_selfcheck.py --attach {sid}", "all": path.relative_to(ws).as_posix()}
    _atomic_write(path, json.dumps(report, indent=2))
    compact = {**report, "ranked_candidates": rankings[:12], "comparisons": comparisons[:12]}
    text = json.dumps(compact, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    if complete:
        return 2 if faults else (0 if report["all_pass"] else 1)
    return 3 if time.time() < manifest["deadline_unix_ns"] / 1e9 else 2


def _write_reference_views(workspace, arguments, *, entry_symbol="kernel_entry"):
    """Pin existing author-owned artifacts; never execute or qualify their code."""
    workspace = Path(workspace).resolve()

    def owned(value):
        relative = PurePosixPath(value)
        if (relative.is_absolute() or str(relative) != value or "\\" in value
                or any(part in (".", "..") for part in relative.parts)
                or not relative.parts or relative.parts[0] not in
                ("submission", "variants", "selfcheck_out", "artifacts")):
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
        if any(getattr(before, key) != getattr(after, key) for key in
               ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")) or len(raw) != before.st_size:
            raise ValueError("artifact changed while pinning; finish compilation and retry")
        bundle[role] = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return dict(status="artifact_views_written", manifest=label,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), qualification_issued=False,
        next_argv=[sys.executable, Path(__file__).name, "--reference-analysis", "candidate",
                   "--reference-path", label, "--reference-format", "artifact_views"],
        note="Pins identify supplied bytes only. Source binding, emitted mechanism, numerics and timing "
             "require the selected analysis and original executable checks.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Agent self-check (redacted; routed to the driver-side broker).",
        epilog=_WORKFLOW,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--submission",
        default=_SUBMISSION,
        help="tree to check: submission (default; the only tree graded officially) or variants/NAME "
        "(a variant runs on the Spike cycle model unless --sim names another engine)",
    )
    ap.add_argument(
        "--suite",
        choices=("core", "scalability", "programs"),
        default="core",
        help="core conformance, required scalability, or operator-selected public programs (Spike only)",
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
    ap.add_argument(
        "--panel",
        default="",
        metavar="FAMILY",
        help="check one representative per shape class (square, tall, wide, thin M=1, ragged, a "
        "model layer) of FAMILY (dense, conv, gemv, ...) and compare each with its best earlier cycles "
        "of the same evidence kind; the Spike cycle model by default (--sim gsim for RTL cycles)",
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
    ap.add_argument("--sweep", nargs="?", const="", default=None, metavar="VARIANTS",
                    help="time comma-separated variants together with submission/ and return model rankings")
    ap.add_argument("--knob", action="append", default=[], metavar="NAME=V1,V2",
                    help="add independent declared knob values (JSON scalars) to --sweep")
    ap.add_argument("--bundle", default="", metavar="REPRESENTATIVE", help="sweep a retained performance bundle")
    ap.add_argument(
        "--tuning-overrides", default="", metavar="JSON",
        help="check a private copy with declared knob values changed; submission itself is unchanged",
    )
    ap.add_argument(
        "--reference-request", default="", metavar="ID",
        help="associate this check with an operator-selected reference experiment request",
    )
    ap.add_argument("--reference-analysis", choices=("inventory", "lift", "candidate", "compare"),
                    help="selected guidance only: inspect visible references or candidate artifacts; no grading")
    ap.add_argument("--write-reference-views", nargs=4, metavar=("MANIFEST", "SOURCE", "COMMAND_BUFFER", "EMITTED"),
                    help="write pinned artifact_views from completed owned files; no request or execution. "
                         "Use artifacts/ paths; --reference-symbol selects the emitted entry (default kernel_entry).")
    ap.add_argument("--reference-path", default="", help="workspace-relative reference-kernels/ or owned artifact; "
                    "compare accepts owned before/after files on both sides")
    ap.add_argument("--reference-format", default="auto",
                    choices=("auto", "c_source", "python_schedule", "listing", "disassembly", "elf",
                             "schedule_mlir", "semantic_region", "interface_mlir", "command_buffer", "artifact_views"),
                    help="schedule_mlir is an actual schedule.plan; for emitted LLVM use a pinned artifact_views "
                         "manifest created with --write-reference-views")
    ap.add_argument("--reference-member", default="", help="member of an already visible ZIP; no extraction")
    ap.add_argument("--reference-entry", default="", help="pinned reference-map entry ID; selects its file/locator")
    ap.add_argument("--reference-symbol", default="", help="exact source or disassembly function symbol")
    ap.add_argument("--reference-offset", type=int, default=0)
    ap.add_argument("--reference-limit", type=int, default=8)
    ap.add_argument("--candidate-path", default="", help="compare only: owned submission/, variants/, "
                    "selfcheck_out/ or artifacts/ file")
    ap.add_argument("--candidate-format", default="auto",
                    choices=("auto", "c_source", "python_schedule", "listing", "disassembly", "elf",
                             "schedule_mlir", "semantic_region", "interface_mlir", "command_buffer", "artifact_views"))
    ap.add_argument("--candidate-symbol", default="", help="compare only: exact emitted function symbol")
    ap.add_argument("--candidate-offset", type=int, default=0)
    ap.add_argument(
        "--mechanism-question",
        default="",
        help="propose a public model question to the trusted diagnostic owner; does not grade or grant gSIM",
    )
    ap.add_argument("--mechanism-discriminator", default="", help="what the proposed witnesses distinguish")
    ap.add_argument(
        "--mechanism-purpose", choices=("unresolved_cost", "structural_validation"), default="unresolved_cost"
    )
    a = ap.parse_args(argv)
    if a.write_reference_views:
        if (a.reference_analysis or a.attach or a.list or a.sweep is not None or a.panel or a.mechanism_question
                or a.tuning_overrides or a.reference_request or a.reference_path or a.reference_entry
                or a.reference_member or a.reference_format != "auto" or a.candidate_path
                or a.candidate_format != "auto" or a.candidate_symbol or a.candidate_offset
                or a.suite != "core" or a.capsules != "all" or a.knob or a.bundle
                or a.reference_offset or a.reference_limit != 8 or a.submission != _SUBMISSION
                or any(getattr(a, key) for key in ("candidate_diagnostics", "compile_only", "caller_layout",
                    "shape_coverage", "offload_census", "model_layers", "dynamic_census", "scale_probe"))):
            ap.error("--write-reference-views is a separate local artifact-pinning action")
        try:
            result = _write_reference_views(Path.cwd(), a.write_reference_views,
                                             entry_symbol=a.reference_symbol or "kernel_entry")
        except (OSError, ValueError) as error:
            ap.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if a.reference_analysis:
        if (a.attach or a.list or a.sweep is not None or a.panel or a.mechanism_question or a.tuning_overrides
                or a.reference_request or a.suite != "core" or a.capsules != "all"
                or any(getattr(a, key) for key in ("candidate_diagnostics", "compile_only", "caller_layout",
                    "shape_coverage", "offload_census", "model_layers", "dynamic_census", "scale_probe"))):
            ap.error("--reference-analysis is a separate read-only diagnostic action")
        if not 0 <= a.reference_offset or not 0 <= a.candidate_offset or not 1 <= a.reference_limit <= 16:
            ap.error("reference analysis requires a nonnegative offset and limit1..16")
        a.reference_path = a.reference_path or ("reference-kernels" if a.reference_analysis == "inventory" else "")
        if not a.reference_path and not a.reference_entry:
            ap.error("reference analysis requires --reference-path or --reference-entry")
        if a.reference_entry and a.reference_analysis not in ("lift", "compare"):
            ap.error("--reference-entry selects a reference for lift/compare")
        if a.reference_analysis == "compare":
            if not a.candidate_path:
                ap.error("--reference-analysis compare requires --candidate-path")
        elif a.candidate_path or a.candidate_format != "auto" or a.candidate_symbol or a.candidate_offset:
            ap.error("candidate comparison options require --reference-analysis compare")
    elif (a.reference_path or a.reference_member or a.reference_format != "auto" or a.reference_offset
          or a.reference_entry or a.reference_symbol or a.candidate_path or a.candidate_format != "auto"
          or a.candidate_symbol or a.candidate_offset):
        ap.error("reference input options require --reference-analysis")
    if a.sweep is not None:
        if (a.submission != _SUBMISSION or a.panel or a.attach or a.list or a.mechanism_question
                or a.suite == "programs" or a.tuning_overrides or a.reference_request):
            ap.error("--sweep uses submission/ as baseline and cannot combine with another action")
        a.suite = "scalability"
    elif (a.knob or a.bundle) and not a.attach:
        ap.error("--knob and --bundle require --sweep")
    if (a.tuning_overrides or a.reference_request) and (a.attach or a.list):
        ap.error("tuning/reference options submit a new check; omit them with --attach or --list")
    overrides = None
    if a.tuning_overrides:
        try:
            if len(a.tuning_overrides.encode()) > 8192:
                raise ValueError("request exceeds 8192 bytes")
            overrides = json.loads(a.tuning_overrides)
            if not isinstance(overrides, dict) or not 1 <= len(overrides) <= 4:
                raise ValueError("expected one to four knob values")
            for key, value in overrides.items():
                if not 0 < len(key) <= 64 or not (
                    value is None or type(value) in (bool, int)
                    or (type(value) is float and math.isfinite(value))
                    or (type(value) is str and len(value) <= 64)
                ):
                    raise ValueError("expected bounded knob names and finite JSON scalar values")
        except (ValueError, RecursionError) as exc:
            ap.error(f"--tuning-overrides: {exc}")
    if a.reference_request and (
        len(a.reference_request) > 160 or a.reference_request[0] not in _NAME_FIRST
        or not all(c in _NAME_CHARS for c in a.reference_request)
    ):
        ap.error("--reference-request takes a bounded request ID, not a path")
    if a.panel:
        family = a.panel.strip().lower()
        if not family or not all(c in _NAME_CHARS for c in family) or len(family) > _VARIANT_NAME_MAX:
            ap.error("--panel takes a family name such as dense, conv or gemv")
        if a.capsules != "all" or a.suite != "core" or a.mechanism_question:
            ap.error("--panel selects its own cases; drop --capsules/--suite/--mechanism-question")
        a.capsules = _PANEL + family
        # Performance on the selected Spike cycle model by default (model cycles, compared like with
        # like); `--sim gsim` still asks for RTL cycles explicitly.
        if not any(v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:])):
            a.sim = "spike"
    elif (a.submission != _SUBMISSION or a.tuning_overrides) and not a.attach and not any(
        v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:])
    ):
        # A variant or tuning A/B is a performance comparison: the Spike cycle model by default.
        a.sim = "spike"
    if a.mechanism_question:
        if (
            not a.mechanism_discriminator.strip()
            or len(a.mechanism_question) + len(a.mechanism_discriminator) > 2000
            or a.suite != "core"
            or a.capsules in ("all", "changed", "failing")
        ):
            ap.error("mechanism questions need named public capsules, a bounded discriminator, and the core suite")
        names = a.capsules.split(",")
        if not 1 <= len(names) <= 4 or len(set(names)) != len(names):
            ap.error("mechanism questions accept one to four distinct public witnesses")
    elif a.mechanism_discriminator:
        ap.error("--mechanism-discriminator requires --mechanism-question")
    if a.suite in {"scalability", "programs"}:
        # This shim is standalone inside the author sandbox; the host repeats this gate.
        explicit_sim = any(v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:]))
        if explicit_sim and a.sim != "spike":
            ap.error("public Spike suites accept only --sim spike")
        a.sim = "spike"
        tiers = [v.strip() for v in a.tiers.split(",") if v.strip()]
        if (tiers and tiers != ["L0", "L1", "L2"]) or any(
            getattr(a, k, False)
            for k in (
                "compile_only",
                "candidate_diagnostics",
                "caller_layout",
                "shape_coverage",
                "offload_census",
                "model_layers",
                "dynamic_census",
                "scale_probe",
            )
        ):
            ap.error("public Spike suites require L0,L1,L2 and refuse inspection/RTL requests")
    if a.timeout is None:
        a.timeout = 3600 if a.sim == "gsim" else 1800
    if a.scale_probe and not a.compile_only:
        ap.error("--scale-probe requires --compile-only")

    policy_error = (
        None
        if a.caller_layout or a.candidate_diagnostics or a.compile_only or a.reference_analysis
        else (_sim_policy_error(a.sim) if a.suite == "core" else None)
    )
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
    manifest = _read_json(ch / f"{a.attach}.json") if a.attach and Path(a.attach).name == a.attach else None
    if a.sweep is not None or (isinstance(manifest, dict) and manifest.get("schema") == _SWEEP_SCHEMA):
        try:
            return _run_sweep(a, ws, ch, manifest)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(json.dumps({"error": str(exc), "all_pass": False}))
            return 2
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
        # Forwarded for operator accounting; the broker trusts it for nothing and reads it only to
        # interleave waiting gSIM requests across threads, so one thread's sweep does not hold another's.
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
                    **({"tuning_overrides": overrides} if overrides else {}),
                    **({"reference_experiment_request_id": a.reference_request} if a.reference_request else {}),
                    **(dict(action="reference-analysis", reference_analysis=dict(
                        operation=a.reference_analysis, path=a.reference_path, format=a.reference_format,
                        member=a.reference_member, offset=a.reference_offset, limit=a.reference_limit,
                        symbol=a.reference_symbol, entry=a.reference_entry,
                        **(dict(candidate=dict(path=a.candidate_path, format=a.candidate_format,
                            symbol=a.candidate_symbol, offset=a.candidate_offset))
                           if a.reference_analysis == "compare" else {})))
                       if a.reference_analysis else {}),
                    "deadline_unix_ns": int(deadline * 1_000_000_000),
                    # Omitted entirely when the caller named none: an ABSENT key lets the driver
                    # resolve the engine, where a present one is a choice the broker must honour.
                    **({"sim": a.sim} if a.sim and not a.caller_layout else {}),
                    **({"tiers": a.tiers} if a.tiers else {}),
                    **({"suite": a.suite} if a.suite != "core" else {}),
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
                    **(
                        {
                            "action": "mechanism-question",
                            "mechanism_question": {
                                "schema": "merlin.mechanism_question_proposal.v1",
                                "question": a.mechanism_question,
                                "discriminator": a.mechanism_discriminator,
                                "capsules": a.capsules.split(","),
                                "purpose": a.mechanism_purpose,
                            },
                        }
                        if a.mechanism_question
                        else {}
                    ),
                }
            ),
        )
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    progress = ch / f"progress_{rid}.json"
    claimed = ch / f"claimed_{rid}"
    last_progress_finished = -1
    reported_cases: dict = {}
    last_queue_note = None
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
                queue = _gsim_queue(p)
                if queue is not None:
                    # Waiting for a busy gSIM slot is a queue, not a fault and not a refusal.
                    if _gsim_queue_note(queue) != last_queue_note:
                        last_queue_note = _gsim_queue_note(queue)
                        print(f"[self-check] {last_queue_note}", file=sys.stderr, flush=True)
                finished = int(p.get("n_finished", 0)) if queue is None else last_progress_finished
                if finished != last_progress_finished:
                    # stderr keeps stdout machine-readable while making a long foreground check visibly
                    # alive in an interactive terminal.
                    print(
                        f"[self-check] {finished}/{p.get('n_expected', '?')} finished; counts={p.get('counts', {})}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress_finished = finished
                for line in _new_case_lines(reported_cases, p) if queue is None else ():
                    print(line, file=sys.stderr, flush=True)
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
            _print_next_steps(_v)
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
            if a.mechanism_question or (original or {}).get("action") == "mechanism-question":
                return 0 if _v.get("status") == "queued_for_trusted_diagnostic_review" else 1
            if a.reference_analysis or (original or {}).get("action") == "reference-analysis":
                return 0 if _v.get("schema") == "merlin.reference_analysis.v1" and _v.get("status") == "analyzed" else 2
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
        queue = _gsim_queue(snapshot)
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
                    # Every case decided so far: finished rows, and rows whose numerics are decided
                    # while later checks run ("status": "running", "final": false).
                    "n_numerics_decided": snapshot.get("n_numerics_decided"),
                    "numerics_counts": snapshot.get("numerics_counts", {}),
                    "cases": _case_rows(snapshot),
                    **({"gsim_slot_queue": queue} if queue is not None else {}),
                    "resume_with": f"python agent_selfcheck.py --attach {rid}" + (f" --out {a.out}" if a.out else ""),
                    "note": (
                        (
                            f"{_gsim_queue_note(queue)}: the check waits for a free gSIM slot and then runs; "
                            if queue is not None
                            else ""
                        )
                        + "the check is STILL RUNNING broker-side; this is not a pass, a failure or a "
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
