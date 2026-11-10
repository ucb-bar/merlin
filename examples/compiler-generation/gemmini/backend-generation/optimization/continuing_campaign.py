"""Persistent diagnostic campaign controller; measured claims stay explicit.

The queue coordinates model workers; command jobs are immutable JSON requests.
This controller never converts counts, process success, or a partial floor into
hardware qualification. Candidate ingestion reopens the actual evidence files.
"""

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import statistics
import sys
import time
from pathlib import Path

from merlin.common import owned_process

TASK = ROOT = CONFIG = None
STOP = False


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def pin(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def reopen(record):
    actual = pin(record["path"])
    if actual["sha256"] != record["sha256"]:
        raise ValueError("evidence bytes differ: " + actual["path"])
    return Path(actual["path"])


def events(value):
    with (ROOT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(epoch=time.time(), **value)) + "\n")


class RecoveryRequired(RuntimeError):
    """An interrupted external action has no explicit idempotent resume path."""


def job_argv(out, request):
    previous = out / "request.json"
    if not previous.exists():
        return request["argv"]
    if json.loads(previous.read_text()) != request:
        raise ValueError("dispatched job requests are immutable")
    if not request.get("resume_argv"):
        raise RecoveryRequired("interrupted job requires an explicit owned-action resume command")
    return request["resume_argv"]


def execute(path, request):
    out = ROOT / "jobs" / path.stem
    out.mkdir(exist_ok=True)
    allowance = min(request.get("timeout_seconds", 1800), max(1, CONFIG["deadline_epoch"] - time.time() - 5))
    try:
        argv = job_argv(out, request)
        write(out / "request.json", request)
        result = owned_process.run(argv, cwd=request.get("cwd", str(TASK)), timeout=allowance, text=True)
        (out / "stdout").write_text(result.stdout)
        (out / "stderr").write_text(result.stderr)
        value = dict(
            status="complete" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            cleanup=result.owned_process_cleanup,
            request=pin(out / "request.json"),
            finished_epoch=time.time(),
        )
    except BaseException as error:
        value = dict(
            status="recovery_required" if isinstance(error, RecoveryRequired) else "failed",
            error_type=type(error).__name__,
            reason=str(error),
            finished_epoch=time.time(),
        )
    write(out / "result.json", value)
    events(dict(kind="job_finished", job=path.stem, result=value))
    return value


def supervise():
    """Restart a failed controller; durable jobs resume only through declared APIs."""
    lease = (ROOT / "watchdog.lock").open("w")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease.write(str(os.getpid()))
    lease.flush()
    attempt = 0
    while time.time() < CONFIG["deadline_epoch"]:
        attempt += 1
        folder = ROOT / "watchdog" / str(time.time_ns())
        folder.mkdir(parents=True)
        write(
            ROOT / "watchdog-status.json", dict(pid=os.getpid(), attempt=attempt, epoch=time.time(), status="running")
        )
        try:
            result = owned_process.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--campaign-root", str(ROOT)],
                cwd=TASK,
                timeout=max(1, CONFIG["deadline_epoch"] - time.time() + 60),
                text=True,
            )
            (folder / "stdout").write_text(result.stdout)
            (folder / "stderr").write_text(result.stderr)
            write(folder / "result.json", dict(returncode=result.returncode, cleanup=result.owned_process_cleanup))
        except BaseException as error:
            write(folder / "result.json", dict(error_type=type(error).__name__, reason=str(error)))
        if time.time() < CONFIG["deadline_epoch"]:
            events(dict(kind="watchdog_restarting_controller", attempt=attempt, evidence=str(folder)))
            time.sleep(min(60, 2 ** min(attempt, 6)))
    write(ROOT / "watchdog-status.json", dict(pid=os.getpid(), epoch=time.time(), status="deadline_reached"))


def ingest(path, board):
    candidate = json.loads(path.read_text())
    member = next(r for r in board["members"] if r["member"] == candidate["member"])
    reopen(candidate["elf"])
    numeric = candidate["numerical"]
    numerical_record = json.loads(reopen(numeric["artifact"]).read_text())
    if (
        numeric["status"] != "pass"
        or numeric["complete_outputs"] is not True
        or numeric["elf_sha256"] != candidate["elf"]["sha256"]
    ):
        raise ValueError("complete same-ELF numerical checks missing")
    if numerical_record.get("checked_elements") != member["checked_elements"]:
        raise ValueError("numerical observation dropped original outputs")
    observed_elf = numerical_record.get("elf")
    observed_sha = observed_elf.get("sha256") if isinstance(observed_elf, dict) else numerical_record.get("elf_sha256")
    if observed_sha != candidate["elf"]["sha256"]:
        raise ValueError("numerical artifact refers to another ELF")
    if numerical_record.get("status") != "complete_outputs_passed" and not numerical_record.get(
        "complete_output_bytes_equal_to_cpu_source"
    ):
        raise ValueError("numerical artifact has no complete comparison")
    if "console" in numerical_record:
        console = reopen(numerical_record["console"]).read_text()
        if f"FULL_COMPARE Y0 {member['checked_elements']} 0" not in console:
            raise ValueError("full numerical comparison absent from actual console")
    timing = candidate["spike"]
    report = json.loads(reopen(timing["report"]).read_text())
    feature = report.get("feature_observation", {})
    if feature.get("complete") is not True or report.get("timing_complete") is not True:
        raise ValueError("annotated timing coverage incomplete")
    cycles = feature.get("estimated_cycles", report.get("estimated_cycles"))
    if (
        not isinstance(cycles, (float, int))
        or not math.isfinite(cycles)
        or cycles <= 0
        or cycles != timing["estimated_cycles"]
    ):
        raise ValueError("positive cycle estimate missing or mismatched")
    binding = report.get("scope_binding", feature.get("scope_binding", {}))
    if binding.get("elf_sha256") != candidate["elf"]["sha256"]:
        raise ValueError("timing boundary not bound to same ELF")
    floor = candidate["roofline"]
    if not floor.get("assumptions") or not floor.get("uncertainty") or not floor.get("evidence"):
        raise ValueError("roofline assumptions, uncertainty or independent evidence absent")
    for evidence in floor["evidence"]:
        reopen(evidence)
    floor_cycles = floor["floor_cycles"]
    if not isinstance(floor_cycles, (int, float)) or not 0 < floor_cycles <= cycles:
        raise ValueError("invalid or refuted roofline floor")
    if report.get("model_sha256") != timing["model_sha256"]:
        raise ValueError("timing model identity does not match reopened report")
    if timing["model_sha256"] != member.get("model_sha256"):
        return dict(status="baseline_replay_required", reason="model changed or baseline cycle estimate missing")
    base = member["baseline_estimated_cycles"]
    incumbent = member["current_estimated_cycles"]
    result = dict(
        status="rejected_cycle_regression",
        estimated_cycles=cycles,
        estimated_speedup_vs_baseline=base / cycles,
        estimated_speedup_vs_incumbent=incumbent / cycles,
        attainment=floor_cycles / cycles,
        roofline_kind=floor["kind"],
        formal_qualification=False,
    )
    if cycles < incumbent:
        result["status"] = "provisional_estimated_incumbent"
        member.update(
            current_estimated_cycles=cycles,
            best_estimated_cycles=min(member["best_estimated_cycles"], cycles),
            estimated_speedup=base / cycles,
            current_candidate=candidate["id"],
            current_artifact=pin(path),
            roofline_floor_cycles=floor_cycles,
            roofline_kind=floor["kind"],
            attainable_roofline_attainment=floor_cycles / cycles if floor["kind"] == "attainable" else None,
            partial_floor_ratio=floor_cycles / cycles if floor["kind"] != "attainable" else None,
            uncertainty=floor["uncertainty"],
            correctness="complete same-ELF outputs pass",
        )
        if member.get("compute_floor_cycles"):
            member.setdefault("baseline_compute_peak_utilization", member.get("compute_peak_utilization"))
            member["compute_peak_utilization"] = member["compute_floor_cycles"] / cycles
    return result


def ingest_measurement(path, board):
    """Join complete collected hardware rows within their actual paired batch."""
    measurement = json.loads(path.read_text())
    if measurement.get("schema") != "merlin.phase2.firesim_batch_result.v1" or measurement.get("status") != "pass":
        raise ValueError("successful original hardware collection required")
    reopen(measurement["elf"])
    reopen(measurement["uploaded_elf"])
    if measurement["uploaded_elf"]["sha256"] != measurement["elf"]["sha256"]:
        raise ValueError("uploaded hardware ELF differs")
    manifest = json.loads(reopen(measurement["source_manifest"]).read_text())
    if manifest["elf"]["sha256"] != measurement["elf"]["sha256"]:
        raise ValueError("hardware result differs from original batch")
    for artifact in measurement["artifacts"].values():
        reopen(artifact)
    rows = {r["member"]: r for r in board["members"]}
    manifest_rows = {(r["member"], r["arm"]): r for r in manifest["cases"]}
    baselines = {r["member"]: r for r in measurement["cases"] if r["arm"] == "baseline"}
    observed_keys = [(r["member"], r["arm"]) for r in measurement["cases"]]
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != set(manifest_rows)
        or len(observed_keys) != len(manifest["cases"])
    ):
        raise ValueError("hardware collection dropped required batch members")
    results = []
    for sample in measurement["cases"]:
        key = (sample["member"], sample["arm"])
        if (
            key not in manifest_rows
            or sample["complete_outputs"] is not True
            or sample["checked_elements"] != manifest_rows[key]["checked_elements"]
            or sample["checked_elements"] != rows[sample["member"]]["checked_elements"]
        ):
            raise ValueError("complete original hardware output checks missing")
        values = sample["measured_cycles"]
        if not values or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("positive measured hardware cycles required")
        row = rows[sample["member"]]
        if sample["arm"] == "baseline":
            if row["baseline_measured_cycles"] is None:
                row["baseline_measured_cycles"] = values
                row["baseline_hardware_artifact"] = pin(path)
            continue
        if sample["member"] not in baselines:
            results.append(dict(member=sample["member"], arm=sample["arm"], status="matched_baseline_missing"))
            continue
        base = baselines[sample["member"]]["measured_cycles"]
        speedup = statistics.median(base) / statistics.median(values)
        results.append(
            dict(
                member=sample["member"],
                arm=sample["arm"],
                status="measured",
                speedup=speedup,
                baseline_cycles=base,
                candidate_cycles=values,
            )
        )
        if speedup >= (row.get("measured_speedup") or 0):
            row.update(
                current_measured_cycles=values,
                best_measured_cycles=statistics.median(values),
                matched_baseline_measured_cycles=base,
                measured_speedup=speedup,
                measured_candidate=sample["arm"],
                measured_artifact=pin(path),
                measured_elf_sha256=measurement["elf"]["sha256"],
                measured_configuration_artifact=measurement["artifacts"]["HW_CFG_SUMMARY"],
            )
            if row.get("compute_floor_cycles"):
                row["measured_compute_peak_utilization"] = row["compute_floor_cycles"] / statistics.median(values)
            row["measured_attainable_roofline_attainment"] = (
                row["roofline_floor_cycles"] / statistics.median(values)
                if row.get("roofline_kind") == "attainable"
                else None
            )
    return dict(
        status="measured_rows_joined",
        comparisons=results,
        elf_sha256=measurement["elf"]["sha256"],
        formal_qualification=False,
    )


def refresh_roofline_ratios(member):
    floor = member.get("conditional_necessary_floor_cycles")
    if not floor:
        return
    upper = member["independent_frozen_feasible_cycles_upper"]
    estimated = member.get("current_estimated_cycles")
    if estimated:
        member["conditional_roofline_attainment_lower"] = floor / estimated
        member["conditional_roofline_attainment_upper"] = min(1.0, upper / estimated)
    measured = member.get("current_measured_cycles")
    if measured:
        member["measured_conditional_roofline_attainment_lower"] = floor / statistics.median(measured)
        member["measured_conditional_roofline_attainment_upper"] = min(1.0, upper / statistics.median(measured))


def ingest_roofline_pack(path, board):
    pack = json.loads(path.read_text())
    if pack.get("schema") not in (
        "merlin.phase2.conditional_roofline_pack.v1",
        "merlin.phase2.conditional_roofline_pack.v2",
    ):
        raise ValueError("explicit conditional roofline pack required")
    members = {member["member"]: member for member in board["members"]}
    if len(pack["members"]) != len(members) or {row["member"] for row in pack["members"]} != set(members):
        raise ValueError("roofline pack dropped original required members")
    for row in pack["members"]:
        evidence = json.loads(reopen(row["artifact"]).read_text())
        floor = evidence["theoretical_necessary_floor_cycles"]
        upper = evidence["independent_frozen_complete_feasible_cycles_upper"]
        if evidence["member"] != row["member"] or not 0 < floor <= upper:
            raise ValueError("invalid conditional service envelope")
        member = members[row["member"]]
        member.update(
            conditional_necessary_floor_cycles=floor,
            independent_frozen_feasible_cycles_upper=upper,
            conditional_roofline_artifact=row["artifact"],
            roofline_assumptions=evidence["assumptions"],
            roofline_uncertainty=evidence["uncertainty"],
            full_attainable_qualification=evidence["full_attainable_qualification"],
        )
        refresh_roofline_ratios(member)
    return dict(status="conditional_roofline_envelope_joined", members=len(members), qualification="PENDING")


def review(reviewer, path, board):
    """An invalid later row must not leave an earlier row provisionally selected."""
    pending = copy.deepcopy(board)
    result = reviewer(path, pending)
    board.clear()
    board.update(pending)
    return result


def ingest_model_replay(path, board):
    """Rebase all estimated comparisons only after complete same-model pairs."""
    pack = json.loads(path.read_text())
    if pack.get("schema") != "merlin.phase2.complete_model_replay.v1" or not pack.get("evidence"):
        raise ValueError("explicit model revision and independent evidence required")
    model_sha = pin(reopen(pack["model"]))["sha256"]
    for evidence in pack["evidence"]:
        reopen(evidence)
    members = {row["member"]: row for row in board["members"]}
    if len(pack["members"]) != len(members) or {row["member"] for row in pack["members"]} != set(members):
        raise ValueError("model replay dropped required members")
    previous = {}
    for pair in pack["members"]:
        member = members[pair["member"]]
        if pair["previous_model_sha256"] != member["model_sha256"]:
            raise ValueError("model revision does not start from current model")
        baseline_path, current_path = (reopen(pair[key]) for key in ("baseline", "current"))
        baseline, current = (json.loads(p.read_text()) for p in (baseline_path, current_path))
        if baseline["member"] != member["member"] or current["member"] != member["member"]:
            raise ValueError("model replay changed workload membership")
        hardware = json.loads(reopen(member["baseline_hardware_artifact"]).read_text())
        manifest = json.loads(reopen(hardware["source_manifest"]).read_text())
        original = next(r for r in manifest["cases"] if r["member"] == member["member"] and r["arm"] == "baseline")
        if baseline["elf"]["sha256"] != original["source_elf"]["sha256"]:
            raise ValueError("model replay replaced the frozen baseline ELF")
        previous[member["member"]] = {
            key: member.get(key) for key in ("model_sha256", "baseline_estimated_cycles", "current_estimated_cycles")
        }
        member.update(
            model_sha256=model_sha,
            baseline_estimated_cycles=baseline["spike"]["estimated_cycles"],
            current_estimated_cycles=baseline["spike"]["estimated_cycles"],
            best_estimated_cycles=baseline["spike"]["estimated_cycles"],
            current_candidate=baseline["id"],
            current_artifact=pin(baseline_path),
            estimated_speedup=1.0,
        )
        baseline_result = ingest(baseline_path, board)
        result = ingest(current_path, board)
        bindings = [
            json.loads(reopen(row["spike"]["report"]).read_text())["scope_binding"] for row in (baseline, current)
        ]
        if any(bindings[0][key] != bindings[1][key] for key in ("input_sha256", "scope_sha256")):
            raise ValueError("model replay changed inputs or invocation scope")
        if any(r["status"] == "baseline_replay_required" for r in (baseline_result, result)):
            raise ValueError("new model does not cover baseline and incumbent")
        member.update(
            baseline_artifact=baseline["spike"]["report"]["path"],
            baseline_source_elf=baseline["elf"],
            model_revision=pin(path),
        )
        refresh_roofline_ratios(member)
    return dict(
        status="complete_model_replay_joined", model_sha256=model_sha, previous=previous, formal_qualification=False
    )


def priority_fraction(member):
    for key in (
        "measured_conditional_roofline_attainment_lower",
        "conditional_roofline_attainment_lower",
        "attainable_roofline_attainment",
        "partial_floor_ratio",
        "compute_peak_utilization",
    ):
        if member.get(key) is not None:
            return member[key]
    return -1


def summary(board, work):
    rows = board["members"]
    known = [r for r in rows if r.get("measured_speedup")]
    complete = len(known) == CONFIG["target"]["required_members"]
    priority = sorted(
        rows,
        key=lambda r: (r["current_estimated_cycles"] is not None, priority_fraction(r)),
    )
    return dict(
        updated_epoch=time.time(),
        controller_pid=os.getpid(),
        deadline_utc=CONFIG["deadline_utc"],
        compiler_freeze_utc=CONFIG["compiler_freeze_utc"],
        phase="frozen_compiler_validation" if time.time() >= CONFIG["compiler_freeze_epoch"] else "optimization",
        required_cases=len(rows),
        measured_cases=len(known),
        priority_metric="conservative conditional roofline fraction; measured when available",
        measured_geomean_speedup=math.exp(sum(math.log(r["measured_speedup"]) for r in known) / len(known))
        if complete
        else None,
        hardware_target_met=complete
        and all(
            (r.get("measured_attainable_roofline_attainment") or 0) >= CONFIG["target"]["attainable_roofline_fraction"]
            for r in rows
        )
        and math.exp(sum(math.log(r["measured_speedup"]) for r in known) / len(known)) >= 2,
        priority=[r["member"] for r in priority],
        missing_estimated_cycles=[r["member"] for r in rows if r["current_estimated_cycles"] is None],
        missing_attainable_rooflines=[r["member"] for r in rows if r["attainable_roofline_attainment"] is None],
        work=work,
    )


def main():
    global STOP, TASK, ROOT, CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--watchdog", action="store_true")
    args = parser.parse_args()
    ROOT = args.campaign_root.resolve(strict=True)
    CONFIG = json.loads((ROOT / "manifest.json").read_text())
    TASK = Path(CONFIG["task_dir"]).resolve(strict=True)
    if args.watchdog:
        supervise()
        return
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    for name in (
        "job-requests",
        "jobs",
        "candidate-requests",
        "candidate-receipts",
        "measurement-receipts",
        "roofline-requests",
        "roofline-receipts",
        "model-requests",
        "model-receipts",
        "model-history",
    ):
        (ROOT / name).mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: globals().update(STOP=True))
    signal.signal(signal.SIGINT, lambda *_: globals().update(STOP=True))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    running = {}
    last_checkpoint = 0
    events(dict(kind="controller_started", source=pin(__file__)))
    try:
        while not STOP and time.time() < CONFIG["deadline_epoch"]:
            for path, future in list(running.items()):
                if future.done():
                    future.result()
                    del running[path]
            for path in sorted((ROOT / "job-requests").glob("*.json")):
                if len(running) >= 2:
                    break
                if path in running or (ROOT / "jobs" / path.stem / "result.json").exists():
                    continue
                request = json.loads(path.read_text())
                if time.time() >= CONFIG["compiler_freeze_epoch"] and request.get("kind") not in (
                    "hardware",
                    "validation",
                    "model_replay",
                ):
                    continue
                running[path] = executor.submit(execute, path, request)
            board = json.loads((ROOT / "scoreboard.json").read_text())
            for path in sorted((ROOT / "model-requests").glob("*.json")):
                receipt = ROOT / "model-receipts" / path.name
                if receipt.exists():
                    continue
                previous = copy.deepcopy(board)
                try:
                    value = review(ingest_model_replay, path, board)
                    write(ROOT / "model-history" / path.name, previous)
                except BaseException as error:
                    value = dict(status="model_repair_required", error_type=type(error).__name__, reason=str(error))
                write(receipt, dict(request=pin(path), **value))
                events(dict(kind="model_review", **value))
            for path in sorted((ROOT / "roofline-requests").glob("*.json")):
                receipt = ROOT / "roofline-receipts" / path.name
                if receipt.exists():
                    continue
                try:
                    value = review(ingest_roofline_pack, path, board)
                except BaseException as error:
                    value = dict(status="roofline_repair_required", error_type=type(error).__name__, reason=str(error))
                write(receipt, dict(request=pin(path), **value))
                events(dict(kind="roofline_review", **value))
            for path in sorted((ROOT / "hardware").glob("*/hardware*/result.json")):
                receipt = (
                    ROOT / "measurement-receipts" / ("-".join(path.relative_to(ROOT / "hardware").parts[:-1]) + ".json")
                )
                if receipt.exists():
                    continue
                try:
                    value = review(ingest_measurement, path, board)
                except BaseException as error:
                    value = dict(
                        status="measurement_repair_required", error_type=type(error).__name__, reason=str(error)
                    )
                write(receipt, dict(request=pin(path), **value))
                events(dict(kind="hardware_review", **value))
            for path in sorted((ROOT / "candidate-requests").glob("*.json")):
                receipt = ROOT / "candidate-receipts" / path.name
                if receipt.exists():
                    continue
                try:
                    value = review(ingest, path, board)
                except BaseException as error:
                    value = dict(status="metric_repair_required", error_type=type(error).__name__, reason=str(error))
                write(receipt, dict(request=pin(path), **value))
                events(dict(kind="candidate_review", candidate=path.stem, **value))
            for member in board["members"]:
                refresh_roofline_ratios(member)
            board["updated_epoch"] = time.time()
            write(ROOT / "scoreboard.json", board)
            db = sqlite3.connect(ROOT / "queue.sqlite", timeout=30)
            db.row_factory = sqlite3.Row
            work = [dict(r) for r in db.execute("SELECT * FROM work ORDER BY priority,id")]
            db.close()
            state = summary(board, work)
            state["running_jobs"] = [p.stem for p in running]
            state["expired_leases"] = [
                r["id"] for r in work if r["status"] == "running" and r["lease_until"] < time.time()
            ]
            write(ROOT / "status.json", state)
            if state["phase"] == "frozen_compiler_validation" and not (ROOT / "FREEZE.json").exists():
                write(
                    ROOT / "FREEZE.json",
                    dict(epoch=time.time(), instruction="Stop compiler edits; only frozen matched validation remains."),
                )
                events(dict(kind="compiler_freeze_required"))
            if time.time() - last_checkpoint >= CONFIG["checkpoint_seconds"]:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                write(ROOT / "checkpoints" / f"{stamp}.json", dict(status=state, scoreboard=board))
                events(dict(kind="checkpoint", path=str(ROOT / "checkpoints" / f"{stamp}.json")))
                last_checkpoint = time.time()
            time.sleep(10)
    finally:
        STOP = True
        executor.shutdown(wait=True, cancel_futures=True)
        write(
            ROOT / "controller-terminal.json",
            dict(
                epoch=time.time(),
                deadline_reached=time.time() >= CONFIG["deadline_epoch"],
                owned_jobs_drained=True,
                scientific_success="not inferred from controller shutdown",
            ),
        )


if __name__ == "__main__":
    main()
