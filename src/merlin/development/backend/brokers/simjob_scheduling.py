"""Driver-side ASYNC oracle broker — runs the agent's requested sims OUTSIDE the sandbox, redacted.

Companion to simjob_shim.py (the in-sandbox CLI). Watches <ws>/.qa_channel for `simreq_*.json`, runs each
as a SEPARATE PROCESS = the existing agent_selfcheck.py (the single grading+redaction authority — golden
expected values withheld), and writes back `simresp_<id>.json` + `simdone_<id>`. A timeout is a normal
redacted verdict, never a crash.

CONSTRAINED SIM-RUNNER (load-bearing isolation): a request may only name {an allowed simulator,
capsules that exist under the public capsule set, debug from a whitelist, workers (clamped)}. The broker
maps those to a FIXED agent_selfcheck.py argv — it NEVER execs anything the request names. So the agent
gets full sim power on its OWN submission + the harness capsules, but cannot read goldens/oracle or run
arbitrary shell outside the sandbox.

Bounded local pool (--max-jobs; unset = the shared compute pool size, else 4) + a GLOBAL cross-arm
verilator slot cap (file semaphore in a shared dir) so the 3 concurrent arms don't launch 3x the verilator
load. Exits on the existing STOP sentinel.

Usage: simjob_broker.py --ws <workspace> [--max-jobs N] [--veril-slots 2] [--poll 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.context import InvocationContext, add_context_arguments, resolve_context
from merlin.development.backend.feedback.certification import promotion as _TP
from merlin.development.backend.feedback.certification.promotion import (
    cert_already_executed as _cert_already_executed,
)
from merlin.development.backend.feedback.certification.promotion import promote as _promote  # noqa: E402
from merlin.development.backend.feedback.certification.promotion import resolve_tiers as _resolve_tiers
from merlin.development.backend.feedback.certification.promotion_admission import REQUIRED_SIM
from merlin.development.backend.feedback.certification.promotion_admission import (
    allowed_sims as _allowed_sims,
)
from merlin.development.backend.feedback.certification.promotion_admission import (
    grade_selection_for as _grade_selection_for,
)
from merlin.development.backend.feedback.execution.snapshots import recover_promotion_snapshot
from merlin.development.backend.feedback.grading import completions as _completions
from merlin.development.backend.feedback.selfcheck.diagnostic_adapters import _public_capsules, worker_command

# ---- simjob ------------------------------------------------------------------------------------

PY = sys.executable


def _dedupe_sim_job(ch: Path, jid: str, key: tuple, running: dict, completed: dict) -> bool:
    """Attach an identical request to its one active or completed simulator job."""
    equivalent = next((active_id for active_id, active in running.items() if active["dedupe_key"] == key), None)
    if equivalent is not None:
        running[equivalent].setdefault("aliases", []).append(jid)
        (ch / f"simrun_{jid}").write_text(f"deduplicated with {equivalent}\n")
        return True
    if key in completed:
        original, cached = completed[key]
        (ch / f"simresp_{jid}.json").write_text(json.dumps(cached, indent=2))
        (ch / (f"simerr_{jid}" if cached.get("error") else f"simdone_{jid}")).write_text(
            f"deduplicated with {original}"
        )
        return True
    return False


def _freeze_direct_job_source(ws):
    """Run and promote one direct job from the same private immutable source."""
    from merlin.development.backend.feedback.execution.snapshots import create_promotion_snapshot

    digest = _TP._submission_digest(ws)
    token, error = create_promotion_snapshot(ws, ws, digest, submission_digest=_TP._submission_digest)
    if error:
        return None, None, error
    return recover_promotion_snapshot(Path(ws), digest, snapshot_token=token, submission_digest=_TP._submission_digest)


def _conda_env() -> str:
    # Resolve only after explicit invocation initialization sources experiment.env.
    try:
        from merlin.common.paths import ext_path

        chipyard = ext_path("chipyard")
        return str(chipyard / ".conda-env") if chipyard else "/path/to/chipyard/.conda-env"
    except Exception:
        return "/path/to/chipyard/.conda-env"


def _veril_slots_dir() -> Path:
    """The cross-arm verilator semaphore directory, PER USER.

    This was a fixed ``/tmp/merlin_veril_slots``. ``/tmp`` is world-writable with the sticky bit, so the
    FIRST user on the host to run a bench owns that directory at mode 0775 -- and every other user's
    broker is then locked out of creating a slot inside it. ``mkdir(exist_ok=True)`` hides the problem
    completely: the mkdir succeeds because the directory already exists, and the PermissionError only
    surfaces later, on the slot file, where it read as "the L3 infrastructure crashed".

    Measured on the live round merlincirct_arm4_func_20260901_v4: the directory belonged to a different
    user, so this host's agent could not acquire a verilator slot and therefore could not run L3 AT ALL
    for two entire rounds. Zero of ~600 agent commands mentioned verilator; GM0/GM1 sat failing at L3
    with no way for the agent to reproduce them; and the agent's own round report said "L3
    infrastructure crashed on permission to /tmp/merlin_veril_slots/slot_0".

    The semaphore only ever needed to keep THIS user's arms from oversubscribing verilator against each
    other, never to coordinate across users. So honour ``TMPDIR`` -- which this project sets per user and
    on the large filesystem, where working files are supposed to live -- and qualify by uid so the path
    is still unsquattable if TMPDIR is unset or shared.
    """
    base = os.environ.get("TMPDIR") or "/tmp"
    return Path(base) / f"merlin_veril_slots_{os.getuid()}"


GLOBAL_VERIL_SLOTS = _veril_slots_dir()  # cross-arm (same-user) verilator semaphore

DEBUG_WHITELIST = {"trace", "cycles", "verbose"}


def _valid_capsules(spec: str, capsules_root: Path) -> list[str] | None:
    if spec == "all":
        return ["all"]
    names = [c.strip() for c in spec.split(",") if c.strip()]
    out = []
    seen = set()
    for n in names:
        if not all(c.isascii() and (c.isalnum() or c == "_") for c in n) or not (capsules_root / n).is_dir():
            return None  # reject ../, globs, unknown names
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out or None


def _sim_env(context: InvocationContext) -> dict:
    e = dict(os.environ)
    CE = _conda_env()
    # THE SUBMISSION'S OWN INTERPRETER COMES FIRST. A submission entrypoint is a script with a
    # `#!/usr/bin/env python3` shebang, so whatever `python3` PATH resolves to is what compiles the
    # capsule. Prepending the sim toolchain's conda env put ITS python3 in front of the harness's, and
    # that interpreter has none of the compiler's dependencies -- so every cert job died in its FIRST
    # declared command with `ModuleNotFoundError`, before any RTL ran.
    #
    # Measured on merlincirct_g4p1_20260905: 41 of 41 promoted L3 jobs returned
    # `parse rc=1: ModuleNotFoundError: No module named 'xdsl'`, all with `barrier_status: None` and no
    # execution identity -- so every one was then discarded as unattributable, and the capsule went back
    # to `pending`. The sibling self-check broker launches the same child with the inherited environment
    # and works, which is why this was invisible: only the PROMOTED path was broken.
    #
    # The conda env still supplies spike / riscv-gcc / the RTL engines; it just no longer shadows the
    # interpreter the harness itself is running under.
    e["PATH"] = f"{os.path.dirname(PY)}:{CE}/bin:{CE}/riscv-tools/bin:" + e.get("PATH", "")
    e["RISCV"] = f"{CE}/riscv-tools"
    # .compat_lib first: the conda cmake needs libidn.so.11 (host has only .12) during C++ build configure
    compat = str(context.repo / ".compat_lib")  # scripts -> capsule_bench_v0 -> experiments -> <repo>/.compat_lib
    e["LD_LIBRARY_PATH"] = f"{compat}:{CE}/lib:{CE}/riscv-tools/lib:" + e.get("LD_LIBRARY_PATH", "")
    return e


def _strip_golden(obj):
    """Defence-in-depth: drop any 'expected'/'golden' keys before publishing (agent_selfcheck already
    redacts; this guarantees it even if a future change regresses)."""
    if isinstance(obj, dict):
        return {k: _strip_golden(v) for k, v in obj.items() if k not in ("expected", "golden")}
    if isinstance(obj, list):
        return [_strip_golden(x) for x in obj]
    return obj


def _completed_job_ids(ch: Path) -> set[str]:
    """Recover jobs whose durable response was already published by an earlier broker process."""
    prefix = "simresp_"
    return {p.stem[len(prefix) :] for p in ch.glob(f"{prefix}*.json")}


def _repeat_promotion(state: dict, row: dict, cert_tier) -> bool:
    """Whether a promotion re-runs a case that a completed certificate job already answered.

    Any resolved record of the case's cert tier (passed, failed or unattributed, for any bytes) means
    the case has had its gSIM run; only outstanding (``pending``) records mean it has not. Unreadable
    state reads as a first run, which keeps the established order.
    """
    if not (cert_tier and row.get("promoted") and isinstance(row.get("capsules"), str)):
        return False
    try:
        records = _TP._slots_ro(state, row["capsules"], cert_tier)
    except (AttributeError, TypeError):
        return False
    return any(isinstance(entry, dict) and entry.get("status") != "pending" for entry in records.values())


def _current_source_first(requests, ws, *, context=None, capsules_root=None, cert_tier=None, contract=None):
    """Prioritize current-source work without treating request metadata as evidence.

    Source recovery still authenticates every promoted request at launch. Unknown
    bindings retain ordinary ordering; source changes during observation abandon
    the optimization. Running jobs and their completed certificates are untouched.

    Among current work, a case's first certificate job (its first gSIM run since it passed) goes
    before re-promotions of cases a completed job already answered (:func:`_repeat_promotion`). This
    orders the queue only: every slot still takes the next request in that order.
    """
    ordered = sorted(requests)
    if len(ordered) < 2:
        return ordered
    try:
        current = _TP._submission_digest(ws)
        state = _TP._tier_state(ws) if cert_tier else {}
        ranks = {}
        for request in ordered:
            try:
                row = json.loads(request.read_text())
            except (OSError, ValueError):
                ranks[request] = (1, 1, 1)
                continue
            if not isinstance(row, dict):
                ranks[request] = (1, 1, 1)
                continue
            digest = row.get("submission_digest")
            # Direct requests are frozen from current source when launched.
            from merlin.development.backend.feedback.campaign.correctness import selected_request

            current_work = not row.get("promoted") or digest == current or selected_request(row)
            ready_model = False
            if current_work and digest == current and row.get("promoted") and context is not None:
                try:
                    import yaml

                    from merlin.development.backend.feedback.certification import model_gate_vouchers

                    names = _valid_capsules(str(row.get("capsules", "all")), capsules_root)
                    from merlin.development.backend.feedback.execution import (
                        snapshot_evaluator as snapshot_execution,
                    )

                    independent = snapshot_execution.selected()
                    ready_model = bool(names and names != ["all"]) and all(
                        yaml.safe_load((capsules_root / name / "capsule.yaml").read_text()).get("kind") == "model"
                        and (
                            independent
                            or model_gate_vouchers.lookup(
                                ws,
                                digest=current,
                                capsule=name,
                                context=context,
                                cert_tier=cert_tier,
                                contract=contract,
                            )
                            is not None
                        )
                        for name in names
                    )
                except (OSError, ValueError, TypeError, AttributeError):
                    ready_model = False
            repeat = _repeat_promotion(state, row, cert_tier)
            ranks[request] = (0 if current_work else 1, 1 if repeat else 0, 0 if ready_model else 1)
        if _TP._submission_digest(ws) != current:
            return ordered
        return sorted(ordered, key=lambda request: ranks[request])
    except Exception:  # noqa: BLE001 -- an unreadable fingerprint cannot reorder the queue
        return ordered


def _promoted_model_continuation(ws, caps, capsules_root, tiers, *, digest, context, contract, cert_tier):
    """Tiers and protected vouchers for a promoted model certificate.

    Staged admission certifies a model only through its closed L2 voucher.
    Independent admission (snapshot execution) admits the model by request
    with no prerequisite credit, so it needs no voucher and re-runs its L2.
    """
    import yaml

    from merlin.development.backend.feedback.certification import model_gate_vouchers
    from merlin.development.backend.feedback.execution import snapshot_evaluator as snapshot_execution

    vouchers = []
    for name in caps:
        declaration = yaml.safe_load((capsules_root / name / "capsule.yaml").read_text())
        if declaration.get("kind") != "model":
            continue
        if snapshot_execution.selected():
            tiers = "L2,L3"
            continue
        voucher = model_gate_vouchers.lookup(
            ws, digest=digest, capsule=name, context=context, cert_tier=cert_tier, contract=contract
        )
        if voucher is None:
            raise ValueError("promoted model has no protected closed L2 gate voucher")
        vouchers.append(voucher)
    return ("L2,L3" if vouchers else tiers), vouchers


def _correctness_gsim(request, sim) -> bool:
    """A promoted job that will take a joint correctness gSIM slot."""
    from merlin.development.backend.campaign import control as joint_control

    return (
        bool(request.get("promoted"))
        and sim == "gsim"
        and joint_control.selected() is not None
        and str(request.get("capsules", "")) in joint_control.correctness_capsules()
    )


def _joint_gsim_launchable(request, sim, gate: dict, running: dict) -> bool:
    """Whether a promoted joint correctness gSIM job may start now; any other job is not gated.

    Promotions wait in this broker's queue rather than inside a child whose wall clock is already
    running: a slot must be free and no author may be waiting for one (authors are served first).
    ``gate`` holds one launch pass's count of such slots, read once and spent per launch; this
    broker's running jobs whose children have not taken their slot yet count as busy.
    """
    if not _correctness_gsim(request, sim):
        return True
    if "free" not in gate:
        from merlin.development.backend.campaign import control as joint_control

        starting = [jid for jid, job in running.items() if job.get("correctness_gsim")]
        try:
            gate["free"] = joint_control.correctness_background_slots(starting)
        except (OSError, ValueError):
            gate["free"] = None  # an unreadable ledger is the child's to refuse, with its own reason
    if gate["free"] is None:
        return True
    if gate["free"] < 1:
        return False
    gate["free"] -= 1
    return True


def _queue_wait_allowance(request, sim) -> int:
    """Extra wall clock for an author's joint gSIM job, which may wait for a correctness slot.

    The child bounds its own wait by the campaign clock and its run by ``--timeout``; the outer
    ``timeout`` must not kill it for time spent waiting.
    """
    from merlin.development.backend.campaign import control as joint_control

    if request.get("promoted") or sim != "gsim" or joint_control.selected() is None:
        return 0
    try:
        return max(0, int(joint_control.remaining() or 0))
    except RuntimeError:
        return 0


def _requeue_refused(ch: Path, jid: str, job: dict, out: dict, claimed: set, queued_since: dict) -> bool:
    """Leave a promotion queued when its child found every correctness slot busy before it ran.

    Published, such a refusal resolved nothing yet left the capsule's certificate record pending, so
    those exact bytes were never certified. The request is unclaimed instead and launched again when
    a slot is free; nothing is recorded for it.
    """
    if not (
        job.get("promoted") and out.get("joint_policy_refused") is True and out.get("joint_policy_retryable") is True
    ):
        return False
    if job.get("log"):
        try:
            job["log"].close()
        except Exception:  # noqa: BLE001 -- closing a log must never break the reap
            pass
    _release_veril_workers(job["slots"])
    if job.get("snapshot_root"):
        shutil.rmtree(job["snapshot_root"], ignore_errors=True)
    (ch / f"simrun_{jid}").unlink(missing_ok=True)
    for name in (jid, *job.get("aliases", [])):
        claimed.discard(name)
        queued_since.setdefault(name, time.monotonic())
    print(f"[simjob] {jid} left queued: {out.get('error')}", file=sys.stderr, flush=True)
    return True


def _observe_pending(requests, claimed, queued_since, now):
    """Observe wait even while all worker slots are occupied."""
    pending = [p for p in requests if p.stem[len("simreq_") :] not in claimed]
    for request in pending:
        queued_since.setdefault(request.stem[len("simreq_") :], now)
    return pending


def _admitted_promotions(requests, ws, *, loop_tier, cert_tier, context):
    """Defer obsolete automatic work without cancelling requests or touching results."""
    from merlin.development.backend.feedback.certification import promotion_admission as admission

    observed = False
    current = state = instrument = loop_instrument = None
    snapshots = {}
    for path in requests:
        try:
            request = json.loads(path.read_text())
        except (OSError, ValueError):
            yield path  # Existing dispatch owns malformed/direct request handling.
            continue
        if not isinstance(request, dict) or not request.get("promoted"):
            yield path
            continue
        if not observed:
            observed = True
            try:
                current = admission.observe_source(Path(ws))
                state = _TP._tier_state(ws)
                instrument = _TP.cert_instrument(cert_tier, context=context)
                loop_instrument = _TP.cert_instrument(loop_tier, context=context)
            except (OSError, ValueError):
                current = None
        request_source = current
        from merlin.development.backend.feedback.campaign.correctness import selected_request

        if selected_request(request):
            # Authenticate the same protected snapshot+ledger+instrument as the
            # legacy path, but ordinary author edits do not cancel required work.
            frozen = _TP.snapshots.promotion_snapshot_path(Path(ws), request.get("submission_snapshot"))
            try:
                request_source = admission.observe_source(frozen) if frozen is not None else None
            except (OSError, ValueError):
                request_source = None
        reason = (
            "current source identity unavailable"
            if request_source is None
            else admission.queued_reason(
                Path(ws),
                request,
                loop_tier=loop_tier,
                cert_tier=cert_tier,
                instrument=instrument,
                loop_instrument=loop_instrument,
                current=request_source,
                state=state,
                snapshots=snapshots,
            )
        )
        jid = path.stem.removeprefix("simreq_")
        note = path.parent / f"simdefer_{jid}.json"
        if reason:
            document = {
                "state": "deferred",
                "reason": reason,
                "job_id": jid,
                "source_digest": request.get("submission_digest"),
                "current_source_digest": current.digest if current is not None else None,
            }
            previous = admission.read_note(note)
            # A return to exactly the same stale source still starts a NEW
            # deferral after a recorded resume; do not leave the resume newer.
            resume = path.parent / f"simresume_{jid}.json"
            resumed = admission.read_note(resume)
            after_resume = False
            if (
                isinstance(resumed, dict)
                and resumed.get("state") == "admitted"
                and resumed.get("job_id") == jid
                and isinstance(resumed.get("current_source_digest"), str)
                and resumed["current_source_digest"]
            ):
                try:
                    after_resume = resume.stat().st_mtime_ns >= note.stat().st_mtime_ns
                except OSError:
                    pass
            if previous != document or after_resume:
                admission.write_note(note, document)
                print(f"[simjob] {jid} auto promotion deferred: {reason}", file=sys.stderr, flush=True)
            continue
        # Scheduling metadata preserves the earlier reason when the original
        # source/dependencies return; it grants no qualification authority.
        if note.exists():
            resume = path.parent / f"simresume_{jid}.json"
            document = {"state": "admitted", "job_id": jid, "current_source_digest": current.digest}
            if admission.read_note(resume) != document:
                admission.write_note(resume, document)
        yield path


class VerilSlotsUnusable(RuntimeError):
    """The slot directory cannot be used at all -- distinct from every slot being busy."""


def _veril_acquire(n_slots: int) -> Path | None:
    """A free slot, or None when they are all BUSY.

    "All busy" and "the directory is not usable" must not look alike. Only FileExistsError used to be
    caught, so a PermissionError propagated out of the broker and killed the job with a bare traceback --
    the agent read that as the oracle being broken, which is exactly what it looked like. A directory we
    cannot write is a configuration fault: name it, and name the remedy.
    """
    try:
        GLOBAL_VERIL_SLOTS.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise VerilSlotsUnusable(f"cannot create the verilator slot directory {GLOBAL_VERIL_SLOTS}: {e}") from e
    if not os.access(GLOBAL_VERIL_SLOTS, os.W_OK):
        import pwd

        try:
            owner = pwd.getpwuid(GLOBAL_VERIL_SLOTS.stat().st_uid).pw_name
        except Exception:  # noqa: BLE001
            owner = "another user"
        raise VerilSlotsUnusable(
            f"the verilator slot directory {GLOBAL_VERIL_SLOTS} exists but is not writable by "
            f"{os.getuid()} (owner: {owner}). No verilator slot can be acquired, so no L3 job can run. "
            f"Set TMPDIR to a directory you own (this project uses a per-user one under /scratch)."
        )
    for i in range(n_slots):
        slot = GLOBAL_VERIL_SLOTS / f"slot_{i}"
        for _attempt in (0, 1):  # second pass runs only after a stale reclaim
            try:
                fd = os.open(str(slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return slot
            except FileExistsError:
                if _attempt == 0 and _reclaim_if_stale(slot):
                    continue  # holder is gone; retry this same slot once
                break  # genuinely busy: try the next one
            except PermissionError as e:
                raise VerilSlotsUnusable(
                    f"cannot create {slot}: {e}. No L3 job can run until the slot directory is writable."
                ) from e
    return None


def _reserve_veril_workers(desired: int, n_slots: int) -> tuple[Path, ...]:
    """Reserve one global slot per concurrent RTL capsule, not merely per batch job.

    A batch may use fewer workers than requested when another run already owns slots. A zero-length
    result means the caller should leave this job queued. Never let a partially reserved batch leak
    slots if the second acquisition fails unexpectedly.
    """
    reserved: list[Path] = []
    try:
        for _ in range(min(desired, n_slots)):
            slot = _veril_acquire(n_slots)
            if slot is None:
                break
            reserved.append(slot)
    except Exception:
        for slot in reserved:
            slot.unlink(missing_ok=True)
        raise
    return tuple(reserved)


def _release_veril_workers(slots: tuple[Path, ...]) -> None:
    for slot in slots:
        slot.unlink(missing_ok=True)


def _capsule_workers(requested: object, *, sim: str, n_capsules: int) -> int:
    """Bound the agent's parallelism without allowing a batch to bypass the Verilator semaphore.

    The grade's own selection names no engine, so it takes the tightest elaborated-RTL cap: its cert
    tier runs on one of those engines, and each engine's native slots still admit the actual runs.
    """
    from merlin.target.simulators.engine_policy import capsule_worker_cap

    if sim == REQUIRED_SIM:
        limit = min(capsule_worker_cap(engine) for engine in _rtl_engines())
    else:
        limit = capsule_worker_cap(sim)
    try:
        named = int(requested)
    except (TypeError, ValueError):
        named = 0
    return max(1, min(named if named > 0 else limit, limit, n_capsules))


def _uses_verilator_slots(sim: str) -> bool:
    """Whether a job must hold the cross-arm Verilator semaphore.

    The grade's own selection names no engine; unless the run pins a different one it may select
    Verilator, so it is admitted conservatively under the same semaphore.
    """
    if sim == REQUIRED_SIM:
        return os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() in ("", "verilator")
    return sim == "verilator"


def _reclaim_if_stale(slot: Path) -> bool:
    """Free ``slot`` if the process that wrote its pid is gone. True when it was reclaimed.

    A slot is released by the broker that took it (``slot.unlink``), so a broker that dies without
    running its cleanup leaks the slot FOREVER -- and the acquire loop reads a leaked slot as "busy".
    With the default two slots, two such deaths disable verilator L3 for every future run by this user,
    silently: the loop simply returns None and the agent sees L3 never happening. MEASURED 2026-09-08 --
    both slots were held by pid 2187933, dead since 2026-09-02, and nothing would ever have released
    them. The host's memory monitor killing a broker is enough to cause this, which is not a rare event.

    Deliberately narrow. It reclaims ONLY when the pid file is readable, parses as an integer, and that
    pid does not exist; an unreadable or malformed slot is left alone rather than guessed at, because
    stealing a slot from a LIVE verilator is worse than failing to acquire one -- it would oversubscribe
    the exact resource this semaphore exists to bound. pid reuse is possible in principle; a reused pid
    reads as live, so the slot stays held, which fails safe in the same direction.
    """
    try:
        holder = slot.read_text().strip()
    except OSError:
        return False
    if not holder.isdigit():
        return False
    try:
        os.kill(int(holder), 0)
    except ProcessLookupError:
        pass  # holder is gone -- the slot is stale
    except PermissionError:
        return False  # alive and owned by someone else: leave it
    except OSError:
        return False
    else:
        return False  # holder is alive: genuinely busy
    try:
        slot.unlink()
    except OSError:
        return False  # another broker reclaimed it first
    print(f"[simjob] reclaimed stale verilator slot {slot.name} (holder pid {holder} is gone)", flush=True)
    return True


_NEUTRAL_SIM = "contract"

_CERT_TIMEOUT_FLOOR_S = 900  # never size BELOW what the historical default allowed

_CERT_TIMEOUT_FALLBACK_S = 1200  # no fit and no measurement on disk: the historical generous default

_SCREEN_TIMEOUT_S = 900  # the functional screen / contract tier keeps its historical budget

_CERT_CYCLE_CEILING = 2.0

_CERT_FIT_MARGIN = 1.5


def _rtl_engines() -> tuple[str, ...]:
    """The elaborated-RTL engines in the policy's own cost order.

    Fail CLOSED to the historical pair when the policy cannot be imported, for the same reason
    `_allowed_sims` does: an error path must not invent an engine, and it must not silently drop the
    engines that were always there.
    """
    try:
        from merlin.target.simulators.engine_policy import ENGINE_PRIORITY
    except Exception:  # noqa: BLE001 -- no policy module: keep the historical ladder
        return ("vcs", "verilator")
    return tuple(ENGINE_PRIORITY)


def _cert_budget_s(target: str) -> tuple[int | None, str]:
    """``(seconds, why)`` for one capsule's cert tier, or ``(None, why)`` when nothing supports a number.

    NO AVAILABILITY PROBE HAPPENS HERE. `rtl_engine_policy.select` is the authority on which engine
    runs, but its probes locate/elaborate simulators and may raise, and this function is called by a
    broker that holds sim slots and must never die. So the engine is resolved from the two facts already
    on disk:

      1. ``MERLIN_REQUIRED_RTL_ENGINE`` -- when the experiment PINS an engine that IS the engine, and it
         is the same pin `_allowed_sims` already honours for the request surface.
      2. otherwise, the first engine in the policy's order that this target has measured cert samples
         for. An engine with no samples here has never certified this target, so it is not the engine to
         size against.

    `cert_cost` refuses rather than guessing, and that refusal is preserved: an unmeasured target
    returns None and the caller keeps its historical default rather than inventing a cost law.
    """
    try:
        from merlin.development.targetgen.oracles import certification_cost as cert_cost
    except Exception as exc:  # noqa: BLE001
        return None, f"cert_cost not importable ({type(exc).__name__}: {exc})"
    pinned = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    try:
        if pinned:
            engine, why = pinned, "pinned by MERLIN_REQUIRED_RTL_ENGINE"
            fit = cert_cost.fit_cycles_for(str(target), engine=engine)
        else:
            fits = cert_cost.fits_cycles_for(str(target))
            engine = next((e for e in _rtl_engines() if e in fits), None)
            why = (
                "first policy-order engine with measured cert samples"
                if engine
                else "no measured cert samples on any policy engine"
            )
            fit = fits.get(engine) if engine else None
    except Exception as exc:  # noqa: BLE001 -- unreadable history is "no fit", never a broker crash
        return None, f"cert cost history unreadable ({type(exc).__name__}: {exc})"
    if fit is None:
        return None, f"{why}: no cycle-cost fit for engine {engine!r} on target {target!r}"
    cycles = int(fit.cycles_max * _CERT_CYCLE_CEILING)
    try:
        secs = cert_cost.predict_seconds_from_cycles(fit, cycles)
    except Exception as exc:  # noqa: BLE001
        return None, f"{engine} fit unusable ({type(exc).__name__}: {exc})"
    if not secs or secs <= 0:
        return None, f"{engine} fit predicted no positive cost for {cycles} cycles"
    return int(_CERT_FIT_MARGIN * float(secs)), (
        f"{engine} ({why}): {fit.intercept_s:.1f}s + {fit.per_cycle_s:.6f}s/cycle x {cycles} cycles "
        f"(={_CERT_CYCLE_CEILING:g}x the measured max, n={fit.n_samples}, r2={fit.r2:.2f}) "
        f"x {_CERT_FIT_MARGIN:g} margin"
    )


def _spawn_selfcheck(argv, *, cwd, env, job_log, timeout_s):
    """Launch one asynchronous simulator check without inheriting the campaign terminal.

    ``timeout`` creates a separate process group for the command it supervises. When that group is
    behind the campaign's foreground process group, any simulator probe that reads inherited stdin is
    stopped by the kernel with SIGTTIN. The broker then mistakes a job that never ran for an RTL
    timeout. Simulator checks are non-interactive, so an explicit EOF is the only truthful stdin.
    """
    from merlin.common.process.cpu_pool import worker_pool_command
    from merlin.experiments.frozen_python import inherited_python_command

    return subprocess.Popen(
        ["timeout", str(timeout_s + 120)] + worker_pool_command(inherited_python_command(argv), env),
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=job_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop_sim_process(process) -> None:
    """Stop the simulator's complete process group, including a child of ``timeout``."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _per_capsule_timeout(
    requested: int, *, context: InvocationContext, timing_file: Path | None = None
) -> tuple[int, str]:
    """``(seconds, why)``: the per-capsule elaborated-RTL budget this broker will use.

    Order, and each step's reason for existing:

    1. an explicit ``--per-capsule-timeout`` -- an operator overrides derivation, always.
    2. the cost law of the engine that will actually run (:func:`_cert_budget_s`), FLOORED at
       :data:`_CERT_TIMEOUT_FLOOR_S` so a cheap fit can never shorten the budget below what the old
       default already allowed. Derivation is allowed to raise the budget, never to lower it.
    3. the recorded selected-engine measurement -- kept so a target with
       no cert history behaves as it did rather than losing its budget to the new code path.
    4. the historical constant, when even that measurement is absent.
    """
    if requested and int(requested) > 0:
        return int(requested), "requested on the command line (--per-capsule-timeout)"
    derived, why = _cert_budget_s(str(context.target or ""))
    if derived:
        return max(_CERT_TIMEOUT_FLOOR_S, int(derived)), f"derived from {why}"
    from merlin.development.backend.session.inputs.oracle_timing import (
        observed_seconds,
        read_verified_timing,
        requires_chipyard_timing,
        timing_path,
    )

    descriptor = getattr(context, "descriptor", None)
    if requires_chipyard_timing(descriptor):
        tf = timing_file or timing_path(context.experiment, context.target)
        try:
            measured = int(2 * observed_seconds(read_verified_timing(tf, descriptor=descriptor, target=context.target)))
        except ValueError as exc:  # absent/foreign observations cannot shorten a Chipyard cert budget
            return _CERT_TIMEOUT_FALLBACK_S, f"no engine cost law and no target-bound measurement ({why}; {exc})"
    else:
        # Other simulator classes retain their pre-existing, operator-selected timeout input.
        try:
            measured = int(2 * json.loads(timing_file.read_text())["verilator_per_capsule_s"])
        except Exception:  # noqa: BLE001 -- no measurement: the historical constant
            return _CERT_TIMEOUT_FALLBACK_S, f"no engine cost law and no recorded measurement ({why})"
    return max(_CERT_TIMEOUT_FLOOR_S, measured), (
        f"no engine cost law ({why}); fell back to the recorded "
        + ("selected-engine" if requires_chipyard_timing(descriptor) else "verilator")
        + " measurement"
    )
