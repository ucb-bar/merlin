"""The board half of a BATCHED machine: several candidates and the reference CONTROL in one board job.

A board job's fixed cost (the bitstream flash, the queue's setup) dwarfs a whole model's run, so a
batch runner links several candidates' timing builds, and the reference arm's own program as the
CONTROL variant, into ONE ELF, runs it once, and splits the console by variant.  Each candidate's
verdict is then finished from its own block (:func:`.worker.paired_finish`).

THE CONTROL RULE.  A batch is judged by its control: the reference measured inside the batch must be
the reference measured ALONE -- the same whole-window cycles within ``cycles_tolerance`` and the same
output bytes in every group -- or the batch's timing is not trusted.  The control is rotated (first
in one batch, last in the next) so a position effect shows as drift.  A batch whose control drifted
is a statement about the batch, not about its candidates: each is re-measured ALONE
(:data:`CONTROL_DRIFT_REQUEUES` times) before the drift refuses it.  Until the control has a solo
result there is nothing to judge drift against, so a batch waits rather than running unjudged.

THE CONTROL IS CHECKED BEFORE A BOARD JOB IS SPENT (:func:`control_preflight`).  Its solo result must be
a MEASURED reading taken ALONE, of the very program the batch links as the control, on the very device
the batch runs on; anything else -- a missing file, a batched or wrong reading, another machine's or a
functional model's result -- holds the batch (``control_preflight.json`` says why) instead of
spending a board job whose control could only ever read as drift.  The drift TOLERANCE is the
machine's own (:func:`.noise.drift_tolerance`): the declared bound, widened to the spread this device's
solo repeats actually show over the same timescale, and recorded with its basis in every batch.

A board that never ran the batch (the FPGA absent, a failed flash) is an OUTAGE, not a verdict: every
job goes back to the board queue, the outage is recorded for the store, and one batch is retried per
:data:`BOARD_RETRY_SECONDS` while the functional-model half keeps grading.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import jobs as J
from . import retention as RET
from .identity import key_of, locked, now, read_json, write_json_atomic
from .machines import INFRA_BOARD_UNAVAILABLE, machine_from_spec

BATCH_SCHEMA = "whole_model_batch_v1"
BOARD_OUTAGE = "board_outage.json"
#: The store's last control preflight that held a batch (removed when one passes).
CONTROL_PREFLIGHT = "control_preflight.json"
INFRA_CONTROL_UNMEASURED = "infra_control_unmeasured"
BOARD_OUTAGES = "board_outages.jsonl"
BOARD_RETRY_SECONDS = 900
#: A drifted batch's candidates are each measured alone this many times before the drift refuses them.
CONTROL_DRIFT_REQUEUES = 1
#: A batch runner that exits without finishing is re-tried from the board queue this many times per job.
BATCH_RUNNER_REQUEUES = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_BATCH_WAIT_SECONDS = 300
DEFAULT_CONTROL_TOLERANCE = 0.02
#: After this many consecutive board rounds spent on solo-flagged jobs (a repeat confirmation, or a
#: reference), a pending multi-variant group is forced through even though solo-flagged work is
#: still waiting. Solo work ordinarily preempts a batch on the reasoning that a confirmation which
#: never runs blocks a champion from ever being certified -- but an UNBOUNDED run of those
#: preemptions starves a brand-new candidate group just as badly the other way: an earlier
#: control-drift event requeued one 8-way batch as eight solo repeats, and every board round for
#: hours afterward kept picking one of THOSE ahead of a freshly-queued composite candidate that had
#: been waiting the whole time. The cap exists to protect a pending group, never to idle the board,
#: so it only fires when one is actually waiting.
MAX_CONSECUTIVE_SOLO_BATCHES = 3
#: The ``priority`` a newly-promoted best's confirmation repeat is requested at
#: (``objective.WholeModelObjective._promote``). Strictly above the default (0, an ordinary
#: candidate or an older repeat already sitting in the queue), so a FRESH promotion's solo
#: confirmation outranks a stale one requested before it.
PROMOTED_REPEAT_PRIORITY = 1
SOLO_STREAK_FILE = "solo_streak.json"


def board_outage(root: Path) -> dict[str, Any] | None:
    return read_json(Path(root) / BOARD_OUTAGE)


def _waiting(root: Path) -> list[dict[str, Any]]:
    return [
        job
        for job in (read_json(p) for p in sorted(Path(root).glob("*/job.json")))
        if job and job.get("state") == J.BOARD
    ]


def _is_solo(job: Mapping[str, Any]) -> bool:
    return bool(job.get("solo") or job.get("role") == J.ROLE_REFERENCE)


def _solo_streak(root: Path) -> int:
    return int((read_json(Path(root) / SOLO_STREAK_FILE) or {}).get("count") or 0)


def record_batch_kind(root: Path, *, solo: bool) -> None:
    """Called once per board round actually attempted: a solo pick extends the streak the cap
    counts against, a multi-variant batch (forced or not) resets it to zero."""
    count = (_solo_streak(root) + 1) if solo else 0
    write_json_atomic(Path(root) / SOLO_STREAK_FILE, {"count": count})


def batch_candidates(root: Path, *, batch_size: int) -> list[dict[str, Any]]:
    """The next batch: a solo job (or the reference) alone, else up to ``batch_size`` board-waiting
    jobs, oldest first, correct candidates before an attribution-only seed.

    Solo-flagged jobs are ordered by ``priority`` first (``service.request(..., priority=)``) and
    age second, so a freshly-promoted best's confirmation (the objective's elevated
    :data:`PROMOTED_REPEAT_PRIORITY`) outranks an older repeat sitting in the same queue -- and once
    :data:`MAX_CONSECUTIVE_SOLO_BATCHES` solo rounds have run back to back, a pending multi-variant
    group is forced through instead (see its docstring)."""
    waiting = _waiting(root)
    waiting.sort(key=lambda job: (bool(job.get("attribution_only")), float(job.get("board_ready_epoch") or 0)))
    solo = [job for job in waiting if _is_solo(job)]
    solo.sort(key=lambda job: (-int(job.get("priority") or 0), float(job.get("board_ready_epoch") or 0)))
    solo_keys = {key_of(job) for job in solo}
    multi_variant = [job for job in waiting if key_of(job) not in solo_keys]
    if multi_variant and (not solo or _solo_streak(root) >= MAX_CONSECUTIVE_SOLO_BATCHES):
        return multi_variant[:batch_size]
    if solo:
        return [solo[0]]
    return multi_variant[:batch_size]


def worth_starting(root: Path, spec: Mapping[str, Any], *, clock: float | None = None) -> bool:
    now_epoch = time.time() if clock is None else clock
    waiting = _waiting(root)
    if not waiting:
        return False
    outage = board_outage(root)
    if outage is not None and now_epoch < float(outage.get("retry_after_epoch") or 0):
        return False  # THE BOARD IS DOWN: hold submissions, retry on the interval
    if any(job.get("solo") or job.get("role") == J.ROLE_REFERENCE for job in waiting):
        return True
    control = spec.get("control") or {}
    if control:
        held = control_preflight(control)
        if not held["ok"]:
            # NOTHING TO JUDGE DRIFT AGAINST: no board job is spent; the hold is on record, said once.
            if (read_json(Path(root) / CONTROL_PREFLIGHT) or {}).get("reason") != held["reason"]:
                write_json_atomic(Path(root) / CONTROL_PREFLIGHT, {**held, "checked_at": now()})
            return False
    if len(waiting) >= int(spec.get("batch_size") or DEFAULT_BATCH_SIZE):
        return True
    oldest = min(float(job.get("board_ready_epoch") or now_epoch) for job in waiting)
    return now_epoch - oldest >= float(spec.get("batch_wait_seconds") or DEFAULT_BATCH_WAIT_SECONDS)


def control_preflight(control: Mapping[str, Any], *, device_sha256: str | None = None) -> dict[str, Any]:
    """Whether the declared control can judge a batch, checked BEFORE a board job is spent.

    Its ``solo_result`` must be a MEASURED reading with a whole-window count, taken alone (no batch, or a
    batch of one), of the program its ``board_request`` links (the same ``elf_sha256``) -- and, given
    ``device_sha256`` (the batch's own device), on that device.  ``ok`` False names the first reason."""

    def refuse(reason: str) -> dict[str, Any]:
        return {"ok": False, "reason": f"{INFRA_CONTROL_UNMEASURED}: {reason}", "solo_result": str(solo_path)}

    solo_path = Path(str(control.get("solo_result") or ""))
    solo = read_json(solo_path) if solo_path.name else None
    if solo is None:
        return refuse(f"the control has no readable solo result ({solo_path}); a batch needs one to judge drift")
    if solo.get("timing_status") != V.TIMING_MEASURED:
        return refuse(f"the control's solo result is {solo.get('timing_status')}, not a MEASURED reading")
    cycles = (solo.get("verdict") or {}).get("whole_window_cycles")
    if not isinstance(cycles, int) or cycles <= 0:
        return refuse("the control's solo result has no whole-window cycle count")
    batch = solo.get("batch")
    if isinstance(batch, Mapping) and int(batch.get("size") or 1) > 1:
        return refuse(f"the control's 'solo' result was measured inside a batch of {batch.get('size')}")
    request = read_json(Path(str(control.get("board_request") or "")))
    if not request or not isinstance(request.get("variant"), Mapping):
        return refuse("the control's board request names no linkable variant")
    linked = ((request.get("builds") or {}).get("timing") or {}).get("elf_sha256")
    measured = (solo.get("build") or {}).get("elf_sha256")
    if linked and measured and linked != measured:
        return refuse(f"the solo result measured program {str(measured)[:12]}, the batch links {str(linked)[:12]}")
    device = (solo.get("device") or {}).get("binary_sha256")
    if device_sha256 is not None and device != device_sha256:
        return refuse(
            f"the control's solo result ran on device {str(device)[:12]}, this batch runs on "
            f"{str(device_sha256)[:12]}; a reading from another machine cannot judge this one"
        )
    return {
        "ok": True,
        "solo_result": str(solo_path),
        "solo_whole_window_cycles": cycles,
        "solo_device_sha256": device,
        "solo_finished_at": solo.get("finished_at"),
    }


def control_check(
    control: Mapping[str, Any],
    block: str | None,
    solo_cycles: int,
    solo_words: Mapping[str, Any],
    *,
    noise: Mapping[str, Any] | None = None,
    same_day: bool | None = None,
) -> dict[str, Any]:
    """Whether the control variant, measured inside this batch, is the control measured alone -- within
    the machine's own drift tolerance (:func:`.noise.drift_tolerance` over ``noise``)."""
    from . import noise as NOISE

    if block is None:
        return {"ok": False, "reason": "the control variant printed no block"}
    parsed = V.parse_log(block)
    cycles = parsed.whole_window_cycles
    rule = NOISE.drift_tolerance(
        noise, declared=float(control.get("cycles_tolerance") or DEFAULT_CONTROL_TOLERANCE), same_day=same_day
    )
    tolerance = float(rule["tolerance"])
    unstable = {str(g) for g in control.get("unstable_groups") or ()}
    moved = sorted(
        (g for g in solo_words if g not in unstable and parsed.words.get(g) != tuple(solo_words[g])), key=V._order
    )
    ratio = (cycles / solo_cycles) if cycles and solo_cycles else None
    ok = ratio is not None and abs(ratio - 1.0) <= tolerance and not moved
    return {
        "ok": ok,
        "solo_whole_window_cycles": solo_cycles,
        "batched_whole_window_cycles": cycles,
        "ratio": round(ratio, 6) if ratio else None,
        "tolerance": tolerance,
        "tolerance_rule": rule,
        "groups_whose_bytes_moved": moved,
        "reason": None if ok else "the control variant measured differently inside the batch than alone",
    }


def _record_board_failure(root: Path, run: Mapping[str, Any], batch_id: str) -> dict[str, Any]:
    outage = board_outage(root) or {"opened_at": now(), "opened_epoch": time.time(), "failures": []}
    outage["failures"] = [
        *outage.get("failures", [])[-19:],
        {"at": now(), "batch": batch_id, "job_id": run.get("job_id"), "reason": run.get("incomplete_reason")},
    ]
    outage["last_failure_epoch"] = time.time()
    outage["retry_after_epoch"] = time.time() + BOARD_RETRY_SECONDS
    write_json_atomic(Path(root) / BOARD_OUTAGE, outage)
    return outage


def _close_board_outage(root: Path, batch_id: str) -> None:
    import json

    outage = board_outage(root)
    if outage is None:
        return
    outage.update(closed_at=now(), closed_by_batch=batch_id)
    with (Path(root) / BOARD_OUTAGES).open("a", encoding="utf-8") as log:
        log.write(json.dumps(outage, sort_keys=True) + "\n")
    (Path(root) / BOARD_OUTAGE).unlink(missing_ok=True)


def _back_to_board(
    root: Path, chosen: Sequence[Mapping[str, Any]], field: str, entry: Mapping[str, Any], notice: str, **update: Any
) -> None:
    for job in chosen:
        job_dir = Path(root) / key_of(job)
        with locked(job_dir):
            fresh = read_json(job_dir / "job.json") or dict(job)
            fresh[field] = [*(fresh.get(field) or []), dict(entry)]
            fresh.update(state=J.BOARD, board_ready_epoch=time.time(), notice=notice, **update)
            for stale in ("batch", "worker_pid"):
                fresh.pop(stale, None)
            write_json_atomic(job_dir / "job.json", fresh)


def measure_alone_after_drift(root: Path, chosen: Sequence[Mapping[str, Any]], batch_id: str, control: Any) -> bool:
    """Send every job of a drifted batch back to be measured ALONE, when each still may be.  Returns
    False (and changes nothing) when any job has used its re-measurements: the drift then refuses."""
    if any(len(job.get("control_drifts") or ()) >= CONTROL_DRIFT_REQUEUES for job in chosen):
        return False
    _back_to_board(
        root,
        chosen,
        "control_drifts",
        {"at": now(), "batch": batch_id, "control": control},
        J.CONTROL_DRIFT_NOTICE,
        solo=True,
    )
    return True


def unlinkable(job_dir: Path) -> list[str]:
    """The files a batch needs to link this job's board variant that are not on disk (none: linkable)."""
    request = read_json(Path(job_dir) / J.BOARD_REQUEST)
    if not request or not isinstance(request.get("variant"), Mapping):
        return [str(Path(job_dir) / J.BOARD_REQUEST)]
    variant = request["variant"]
    needed = [variant.get("program_object"), *(variant.get("objects") or ()), *(variant.get("supports") or ())]
    return [str(path) for path in needed if not path or not Path(str(path)).is_file()]


def requeue_for_rebuild(job_dir: Path, job: dict[str, Any], missing: Sequence[str], batch_id: str) -> None:
    """Send a board-waiting job whose objects are gone back to be built again, its old outputs aside."""
    rebuilds = list(job.get("board_rebuilds") or [])
    attempt = J.archive_attempt(Path(job_dir), "unlinkable_attempt_", len(rebuilds))
    job.update(
        state=J.PENDING,
        requeued_at=now(),
        board_rebuilds=[
            *rebuilds,
            {
                "at": now(),
                "batch": batch_id,
                "missing": list(missing)[:8],
                "missing_count": len(missing),
                "moved_to": str(attempt),
            },
        ],
        notice=f"{J.INFRA_BOARD_OBJECTS_MISSING}: the board objects of these bytes were no longer on disk; "
        "rebuilding them (not a verdict)",
    )
    for stale in ("batch", "worker_pid", "dispatched_at", "started_at", "board_ready_epoch"):
        job.pop(stale, None)
    write_json_atomic(Path(job_dir) / "job.json", job)
    RET.prune_archived_attempt(attempt, job.get("retain"))


def _machine_noise(root: Path, control: Mapping[str, Any], device: str | None) -> dict[str, Any]:
    """The batch device's noise, from this store's solo readings and the control's own solo result."""
    from . import noise as NOISE

    extra = [Path(str(control["solo_result"]))] if control.get("solo_result") else []
    return NOISE.machine_noise(
        NOISE.solo_readings([root], extra=extra), device=device, controls=NOISE.control_readings([root])
    )


def _control_variant(root: Path, control: Mapping[str, Any], variants: list[dict[str, Any]]):
    request = read_json(Path(str(control["board_request"]))) or {}
    solo = read_json(Path(str(control["solo_result"]))) or {}
    solo_cycles = (solo.get("verdict") or {}).get("whole_window_cycles")
    uart = Path(str((solo.get("run") or {}).get("uart_log") or ""))
    words = V.parse_log(uart.read_text(encoding="utf-8", errors="replace")).words if uart.is_file() else {}
    entry = dict(request["variant"], label="control")
    # ROTATE THE CONTROL: first in one batch, last in the next, so a position effect shows as drift.
    batches = Path(root) / "batches"
    count = len(list(batches.iterdir())) if batches.is_dir() else 0
    position = 0 if count % 2 else len(variants)
    variants.insert(position, entry)
    return position + 1, solo_cycles, words


def batch_main(root: Path, *, driver: Any = None) -> int:
    """Run ONE batch: link the chosen candidates (and the control) into one ELF, run it once on the
    board, demultiplex the console by variant, and finish each candidate's verdict from its own block."""
    from . import worker as W

    root = Path(root)
    with locked(root):
        first = next((job for job in (read_json(p) for p in sorted(root.glob("*/job.json"))) if job), None)
        if first is None:
            return 0
        spec = dict(first["machine"])
        chosen = batch_candidates(root, batch_size=int(spec.get("batch_size") or DEFAULT_BATCH_SIZE))
        batch_id = now() + f"_{os.getpid()}"
        linkable = []
        for job in chosen:
            missing = unlinkable(root / key_of(job))
            if missing:
                requeue_for_rebuild(root / key_of(job), dict(job), missing, batch_id)
            else:
                linkable.append(job)
        chosen = linkable
        if not chosen:
            return 0
        record_batch_kind(root, solo=len(chosen) == 1 and _is_solo(chosen[0]))
        for job in chosen:
            job.update(state=J.RUNNING, batch=batch_id, worker_pid=os.getpid())
            write_json_atomic(root / key_of(job) / "job.json", job)
    batch_dir = root / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    requests = {key_of(job): read_json(root / key_of(job) / J.BOARD_REQUEST) or {} for job in chosen}
    variants = [dict(requests[key_of(job)]["variant"], label=key_of(job)) for job in chosen]
    control = dict(spec.get("control") or {})
    solo = len(chosen) == 1 and (chosen[0].get("solo") or chosen[0].get("role") == J.ROLE_REFERENCE)
    control_index, control_solo_cycles, control_words = None, None, {}
    record: dict[str, Any] = {"schema": BATCH_SCHEMA, "batch": batch_id, "jobs": [key_of(j) for j in chosen]}
    drifted = False
    identity = None
    blocks: dict[int, str] = {}
    failure: str | None = None
    try:
        machine = machine_from_spec(spec["timing"])
        identity = machine.identity()
        if control and not solo:
            # THE CONTROL IS CHECKED ON THIS BATCH'S OWN DEVICE BEFORE THE BOARD IS ASKED FOR ANYTHING.
            preflight = control_preflight(control, device_sha256=identity.binary_sha256)
            record["control_preflight"] = preflight
            if not preflight["ok"]:
                write_json_atomic(batch_dir / "batch.json", record)
                write_json_atomic(root / CONTROL_PREFLIGHT, {**preflight, "checked_at": now(), "batch": batch_id})
                _back_to_board(
                    root,
                    chosen,
                    "control_preflight_holds",
                    {"at": now(), "batch": batch_id, "reason": preflight["reason"]},
                    f"{preflight['reason']} (no board job was spent; not a verdict)",
                )
                return 0
            (root / CONTROL_PREFLIGHT).unlink(missing_ok=True)
            control_index, control_solo_cycles, control_words = _control_variant(root, control, variants)
        program = variants[0]["program"]
        driver = driver or W.whole_model_driver(str(first["target"])).program
        linked = driver.link_batch(
            [{**v, "objects": v["objects"]} for v in variants],
            batch_dir / "link",
            compiler=Path(program["compiler"]),
            compile_flags=program["flags"],
            link_flags=program["link_flags"],
            link_script=program["link_script"],
            supports=variants[0]["supports"],
        )
        timeout = float(first.get("timeout_seconds") or 3600) * max(1, len(variants))
        run = machine.run(Path(linked["elf"]), batch_dir / "run", timeout_s=timeout)
        record.update(
            linked=linked, run={k: v for k, v in run.items() if k != "device"}, variants=[v["label"] for v in variants]
        )
        if run.get(INFRA_BOARD_UNAVAILABLE):
            record[INFRA_BOARD_UNAVAILABLE] = True
            write_json_atomic(batch_dir / "batch.json", record)
            _record_board_failure(root, run, batch_id)
            _back_to_board(
                root,
                chosen,
                "board_losses",
                {"at": now(), "batch": batch_id, "job_id": run.get("job_id"), "reason": run.get("incomplete_reason")},
                J.BOARD_UNAVAILABLE_NOTICE,
            )
            return 0
        blocks = (
            driver.split_batch(Path(run["uart_log"]).read_text(encoding="utf-8", errors="replace"), len(variants))
            if run.get("completed")
            else {}
        )
        check = (
            control_check(
                control,
                blocks.get(control_index),
                int(control_solo_cycles or 0),
                control_words,
                noise=_machine_noise(root, control, identity.binary_sha256),
                same_day=str((record.get("control_preflight") or {}).get("solo_finished_at") or "")[:8] == batch_id[:8],
            )
            if control_index
            else {"ok": True, "reason": "no control in a solo batch"}
        )
        _close_board_outage(root, batch_id)
        record["control"] = check
        failure = None if run.get("completed") else f"the batch run did not complete: {run.get('incomplete_reason')}"
        if failure is None and not check["ok"]:
            failure = f"batch refused: {check['reason']} ({check})"
            drifted = True
    except Exception as exc:  # noqa: BLE001 -- every job of the batch is refused with the reason
        failure, blocks = f"the batch could not run: {type(exc).__name__}: {exc}", {}
        record["failure"] = failure
    write_json_atomic(batch_dir / "batch.json", record)
    if drifted and measure_alone_after_drift(root, chosen, batch_id, record.get("control")):
        return 0
    finalized: list[dict[str, Any] | None] = []
    for job in chosen:
        key = key_of(job)
        position = next(i for i, v in enumerate(variants, 1) if v["label"] == key)
        request = requests[key]
        job_dir = root / key
        timing_dir = job_dir / "run_timing"
        timing_dir.mkdir(parents=True, exist_ok=True)
        block = blocks.get(position)
        if failure is None and block is not None:
            (timing_dir / "uart.log").write_text(block, encoding="utf-8")
            timing_run = {
                "completed": True,
                "uart_log": str(timing_dir / "uart.log"),
                "batch": batch_id,
                "position": position,
                "job_id": (record.get("run") or {}).get("job_id"),
            }
        else:
            reason = failure or f"variant {position} printed no complete block in batch {batch_id}"
            timing_run = {"completed": False, "incomplete_reason": reason, "batch": batch_id, "position": position}
        device = {"timing": identity.to_dict() if identity else {}, "local": request.get("local_device") or {}}
        outcome = W.paired_finish(
            job,
            request["builds"],
            request["identities"],
            {"timing": timing_run, "local": request["local_run"]},
            device,
            request.get("check"),
            request["package_sha256"],
            batch={"batch": batch_id, "position": position, "size": len(variants), "control": record.get("control")},
            job_dir=job_dir,
        )
        try:
            J.write_result(job_dir, outcome)
        except J.ResultExists:
            # The job already ended (a supersede, an earlier runner): that result stands; this board reading
            # is kept as an attempt of its own, never written over it.
            J.preserve_result(job_dir, outcome, why=f"batch {batch_id} finished after the job already had a result")
            finalized.append(None)
            continue
        with locked(job_dir):
            fresh = read_json(job_dir / "job.json") or dict(job)
            fresh.update(state=J.DONE, finished_at=now(), timing_status=outcome.get("timing_status"))
            write_json_atomic(job_dir / "job.json", fresh)
        finalized.append(RET.finalize_if_allowed(root, job_dir))
    RET.finalize_batch_if_allowed(batch_dir, finalized)
    RET.finalize_superseded(root)
    return 0


__all__ = [
    "CONTROL_PREFLIGHT",
    "INFRA_CONTROL_UNMEASURED",
    "control_preflight",
    "MAX_CONSECUTIVE_SOLO_BATCHES",
    "PROMOTED_REPEAT_PRIORITY",
    "record_batch_kind",
    "BATCH_RUNNER_REQUEUES",
    "BOARD_OUTAGE",
    "CONTROL_DRIFT_REQUEUES",
    "batch_candidates",
    "batch_main",
    "board_outage",
    "control_check",
    "measure_alone_after_drift",
    "requeue_for_rebuild",
    "unlinkable",
    "worth_starting",
]
