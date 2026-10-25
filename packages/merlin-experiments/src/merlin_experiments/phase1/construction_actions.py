"""Standalone public author action runner; ordinary commands and self-check IPC only.

The automatic author and offline study execute this exact file inside their existing
sandbox. It grants no test, ISA, compilation, correctness or performance authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "merlin.construction_actions.v1"
BOUNDARIES = ("graph", "numeric_layout", "schedule_storage", "isa_lowering", "host_runtime", "integration")


def validate(plan, *, known=()):
    if not isinstance(plan, dict) or set(plan) != {"schema", "actions"} or plan.get("schema") != SCHEMA:
        raise ValueError("expected construction action schema and actions only")
    actions = plan["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 512:
        raise ValueError("expected 1..512 explicit actions")
    seen = set(known)
    for row in actions:
        if not isinstance(row, dict) or set(row) - {"id", "capability", "boundary", "kind", "argv", "after", "after_finished"}:
            raise ValueError("unknown action fields")
        name = row.get("id")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 80
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
            or name in seen
        ):
            raise ValueError("action IDs must be unique safe names")
        if row.get("boundary") not in BOUNDARIES or not isinstance(row.get("capability"), str) or not row["capability"]:
            raise ValueError("each action needs a semantic capability and compiler boundary")
        if row.get("kind") not in {"command", "selfcheck"}:
            raise ValueError("only ordinary command and selfcheck actions are supported")
        argv = row.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or "\0" in v for v in argv):
            raise ValueError("action argv must be a nonempty string list")
        if row["kind"] == "selfcheck" and any(
            v in {"--attach", "--list"} or v.startswith(("--attach=", "--list=")) for v in argv
        ):
            raise ValueError("selfcheck actions submit checks; collection is automatic")
        for dependency in ("after", "after_finished"):
            after = row.get(dependency, [])
            if not isinstance(after, list) or any(not isinstance(v, str) or v not in seen for v in after):
                raise ValueError(f"{dependency} must reference earlier actions; no cycles or implicit global barriers")
        seen.add(name)
    return plan


def load(path):
    data = Path(path).read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("action plan exceeds 2 MiB; use small public scripts")
    return validate(json.loads(data)), hashlib.sha256(data).hexdigest()


def enqueue_actions(inbox, plan):
    """Atomically publish another ordinary plan to a warm author runner.

    Dependencies may name earlier admitted actions. The running reader checks
    those against its own history. This publishes code requests only; original
    sandbox, simulator, snapshot and independent grading boundaries still apply.
    """
    directory = Path(inbox)
    directory.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(plan, sort_keys=True) + "\n").encode()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("action plan exceeds 2 MiB")
    name = f"{time.time_ns()}-{hashlib.sha256(raw).hexdigest()[:16]}"
    pending = directory / (name + ".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
    published = pending.with_suffix(".json")
    pending.replace(published)
    return published


def command(row, *, python=sys.executable):
    """The exact same argv in automatic and offline invocation; no mode switch."""
    if row["kind"] == "command":
        return list(row["argv"])
    return [python, "agent_selfcheck.py", *row["argv"], "--wait-budget", "1"]


def _response(stdout):
    # The ordinary shim prints JSON (possibly preceded by progress text).
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(stdout[index:])
                if isinstance(value, dict) and ("request_id" in value or "selfcheck_request_id" in value):
                    return value
            except ValueError:
                pass
    return {}


def report_projection(response):
    """Project existing broker observations without changing verdicts or gates.

    A partial-tier check can execute numerically while its CLI returns one for
    incomplete certification. Counts below concern returned requests, not unique
    capsules, and missing execution or structural evidence stays unknown.
    """
    rows = response.get("per_capsule")
    expected = response.get("n_capsules")
    functional = isinstance(rows, list)
    if not functional:
        rows = response.get("cases")
        requested = response.get("requested_cases")
        expected = len(requested) if isinstance(requested, list) else None
    rows = rows if isinstance(rows, list) else []
    valid = bool(rows) and all(
        isinstance(row, dict) and isinstance(row.get("capsule"), str) and row["capsule"] for row in rows
    )
    names = [row["capsule"] for row in rows] if valid else []
    covered = valid and len(set(names)) == len(names) and type(expected) is int and expected == len(rows)
    if not functional:
        covered = (covered and all(isinstance(name, str) for name in requested)
                   and len(set(requested)) == len(requested) and set(names) == set(requested))
    cases = []
    for row in rows if valid else []:
        numerical = "unknown"
        if functional:
            if row.get("numerical_execution_passed") is False or row.get("numeric_consistency_passed") is False:
                numerical = "fail"
            elif row.get("numerical_execution_passed") is True:
                numerical = "pass"
        else:
            tiers = row.get("tiers") or {}
            # Public scalability reports identify actual execution at L2;
            # source-level L0/L1 checks alone do not establish target numerics.
            if isinstance(tiers, dict):
                if tiers.get("L2") == "pass" and row.get("status") == "pass":
                    numerical = "pass"
                elif tiers.get("L2") == "fail":
                    numerical = "fail"
        structural = row.get("structural_qualification")
        cases.append({
            "capsule": row["capsule"],
            "numerical_screen": numerical,
            "structural_qualification": structural if isinstance(structural, dict) else None,
        })
    numerical = "fail" if any(row["numerical_screen"] == "fail" for row in cases) else "unknown"
    if covered and all(row["numerical_screen"] == "pass" for row in cases):
        numerical = "pass"
    structural = "unknown"
    statuses = [(row["structural_qualification"] or {}).get("status") for row in cases]
    if any(status in {"fail", "failed"} for status in statuses):
        structural = "fail"
    elif covered and all(status == "pass" for status in statuses):
        structural = "pass"
    result = {
        "completed_report": valid and bool(response.get("selfcheck_request_id")),
        "requested_numerical_screen": numerical,
        "requested_cases_covered": bool(covered),
        "structural_qualification": structural,
        "full_certification": response.get("certified_complete") if type(response.get("certified_complete")) is bool else None,
        "grader_all_pass": response.get("all_pass") if type(response.get("all_pass")) is bool else None,
        "cases": cases,
        "qualification_authority": False,
    }
    for key in ("selfcheck_request_id", "submission_sha256", "requested_submission_sha256"):
        if key in response:
            result[key] = response[key]
    return result


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _bounded_list(value, limit):
    return value[:limit] if isinstance(value, list) else []


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def performance_projection(response):
    """Bounded existing-public feedback, independent of grades and model selection.

    Complete modeled reservations describe the same ROI only after exact pins
    and cycle equality. Whole-trace feature_counts never enter this projection.
    No physical utilization or attainable performance is inferred from a floor.
    """
    source = response.get('per_capsule', response.get('cases', []))
    source = source if isinstance(source, list) else []
    numerical = {r['capsule']: r for r in report_projection(response)['cases']}
    result = dict(schema='merlin.construction_performance.v1',
                  selfcheck_request_id=response.get('selfcheck_request_id'),
                  submission_sha256=response.get('submission_sha256') if _pin(response.get('submission_sha256')) else None,
                  cases=[], omitted_cases=max(0, len(source)-64), qualification_authority=False)
    identities = ('model_sha256', 'comparison_domain_sha256', 'scope_sha256', 'elf_sha256',
                  'capsule_sha256', 'selection_sha256', 'facts_sha256')
    for row in source[:64]:
        if not isinstance(row, dict) or not isinstance(row.get('capsule'), str):
            continue
        p = row.get('performance')
        p = p if isinstance(p, dict) else {}
        feedback = p.get('feedback_brief')
        outcomes = feedback.get('outcomes') or {} if isinstance(feedback, dict) else {}
        trace_status = outcomes.get('trace_status', 'unknown') if isinstance(outcomes, dict) else 'unknown'
        item = dict(capsule=row['capsule'][:256], status='UNKNOWN',
                    source_timing_status=p.get('status', 'UNKNOWN'),
                    timing_mode=p.get('timing_mode'), timing_qualified=p.get('timing_qualified') is True,
                    estimated_whole_soc_cycles=None, structural_floor_to_model_ratio=None,
                    ratio_kind='structural compute lower bound / modeled call cycles; NOT physical utilization',
                    numerical_screen=numerical.get(row['capsule'], {}).get('numerical_screen', 'unknown'),
                    structural_qualification=(numerical.get(row['capsule'], {}).get('structural_qualification') or {}).get('status', 'unknown'),
                    trace_status=trace_status, scoped_counts_status='UNKNOWN', scoped_counts={}, resources_status='UNKNOWN',
                    top_modeled_resources=[], overlaps=[], traffic=[], suggestions=[], unknown=[],
                    scope=p['scope'][:1024] if isinstance(p.get('scope'), str) else None,
                    scope_excludes=p.get('scope_excludes') if isinstance(p.get('scope_excludes'), list) else None,
                    **{k:p[k] if _pin(p.get(k)) else None for k in identities})
        result['cases'].append(item)
        s = p.get('scoped_resource_feedback')
        cycles = p.get('estimated_whole_soc_cycles')
        pins = ('model_sha256', 'scope_sha256', 'elf_sha256')
        if not (all(_pin(p.get(k)) for k in identities)
                and p.get('complete_event_accounting') is True and _finite(cycles, positive=True)
                and p.get('status') in ('PROVISIONAL', 'QUALIFIED')
                and isinstance(s, dict) and s.get('status') == 'modeled' and s.get('source_kind') == 'modeled'
                and all(s.get(k) == p[k] for k in pins)
                and s.get('estimated_cycles_same_window') == cycles
                and item['scope'] and item['scope_excludes'] is not None
                and all(isinstance(v, str) for v in item['scope_excludes'])):
            item['reason'] = 'Missing/incomplete exact model, ELF, capsule or counter-window binding; no other-scope substitution'
            continue
        item.update(status=p['status'], estimated_whole_soc_cycles=cycles)
        roof = p.get('structural_roofline')
        roof = roof if isinstance(roof, dict) else {}
        binding = p.get('structural_comparison_binding') or {}
        structural_pins = ('capsule_sha256', 'facts_sha256', 'selection_sha256',
                           'model_sha256', 'comparison_domain_sha256', 'scope_sha256')
        floor = roof.get('structural_compute_floor_cycles')
        item['roofline_status'] = roof.get('status', 'UNKNOWN')
        item['structural_compute_floor_cycles'] = None
        item['roofline_missing_terms'] = [str(v)[:256] for v in _bounded_list(roof.get('missing_terms'), 8)]
        if isinstance(binding, dict) and all(binding.get(k) == p[k] for k in structural_pins) and _finite(floor, positive=True):
            item['structural_compute_floor_cycles'] = floor
            if floor <= cycles:
                item['structural_floor_to_model_ratio'] = floor/cycles
            else:
                item.update(status='REFUTED', reason='Model estimate is below bound; not eligible for ranking')
        # Native brief has already joined count observation to selected ROI.
        # Verify its exact model/window/ELF again, then use ONLY scoped_counts.
        brief = p.get('feedback_brief') or {}
        signals = brief.get('performance_signals') or {} if isinstance(brief, dict) else {}
        bs = signals.get('scoped_resource_feedback') or {} if isinstance(signals, dict) else {}
        if (isinstance(bs, dict) and isinstance(signals, dict)
                and all(signals.get(k) == p[k] for k in ('model_sha256', 'scope_sha256', 'comparison_domain_sha256'))
                and all(bs.get(k) == p[k] for k in pins)
                and signals.get('estimated_whole_soc_cycles') == cycles
                and bs.get('estimated_cycles_same_window') == cycles
                and signals.get('scoped_counts_status') == 'observed'
                and type(signals.get('selected_window')) is int and signals['selected_window'] >= 0):
            counts = signals.get('scoped_counts') or {}
            allowed = ('target.commands', 'target.fences', 'target.class.config', 'target.class.load',
                       'target.class.preload', 'target.class.compute', 'target.class.store',
                       'target.useful_shape_macs', 'target.nominal_padded_macs',
                       'device.requested_read_bytes', 'device.requested_write_bytes',
                       'cpu.integer', 'cpu.branch', 'cpu.load', 'cpu.store')
            if isinstance(counts, dict):
                item['scoped_counts'] = {k:counts[k] for k in allowed if type(counts.get(k)) is int and counts[k] >= 0}
                item['scoped_counts_status'] = 'observed' if item['scoped_counts'] else 'UNKNOWN'
        if s.get('activity_status') == 'modeled':
            resources = s.get('resources')
            resources = resources if isinstance(resources, list) and len(resources) <= 1024 else []
            good = []
            for r in resources:
                if not isinstance(r, dict):
                    continue
                capacity, busy, active = (r.get(k) for k in ('capacity', 'busy_unit_cycles', 'active_cycles'))
                if (isinstance(r.get('name'), str) and type(r.get('resource_id')) is int
                        and _finite(capacity, positive=True) and _finite(busy) and _finite(active)
                        and active <= cycles and active <= busy <= capacity*active):
                    good.append(dict(resource_id=r['resource_id'], name=r['name'][:128], capacity=capacity,
                                     busy_unit_cycles=busy, active_cycles=active,
                                     capacity_normalized_activity=busy/(capacity*cycles),
                                     source_kind='modeled reservations; NOT physical utilization'))
            item['top_modeled_resources'] = sorted(good, key=lambda r:(-r['capacity_normalized_activity'], r['resource_id']))[:8]
            item['resources_status'] = 'modeled' if good else 'UNKNOWN'
            by_id = {r['resource_id']:r for r in good}
            pairs = s.get('overlaps')
            for pair in pairs[:8] if isinstance(pairs, list) else []:
                if not isinstance(pair, dict):
                    continue
                left, right, duration = (pair.get(k) for k in ('left', 'right', 'active_cycles'))
                if (type(left) is int and type(right) is int and left in by_id and right in by_id
                        and left != right and _finite(duration)
                        and duration <= min(by_id[left]['active_cycles'], by_id[right]['active_cycles'])):
                    item['overlaps'].append(dict(left=by_id[left]['name'], right=by_id[right]['name'],
                                                 active_cycles=duration, fraction_of_window=duration/cycles,
                                                 source_kind='modeled'))
        for t in _bounded_list(s.get('traffic'), 12):
            if isinstance(t, dict) and isinstance(t.get('boundary'), str) and _finite(t.get('bytes')):
                item['traffic'].append(dict(boundary=t['boundary'][:128], bytes=t['bytes'],
                                            bytes_per_model_cycle=t['bytes']/cycles, physical_bandwidth=False))
        item['suggestions'] = [str(v)[:256] for v in _bounded_list(s.get('suggestions'), 4)]
        item['unknown'] = [str(v)[:128] for v in _bounded_list(s.get('unknown'), 12)]
        if item['scoped_counts_status'] != 'observed':
            item['unknown'].append('scoped_command_and_CPU_counts')
        item['limits'] = 'Lower latency is the objective. Bounds/reservations/overlap are diagnostics, not physical utilization or grade credit.'
    return result


def performance_portfolio(observations):
    """Latest bounded receipts and strict same-model/domain/scope rankings.

    No cross-ELF PC-window equivalence is invented. Different bound scopes remain
    separate even when their human-readable descriptions happen to match.
    """
    latest = {}
    for observation in observations[-512:]:
        for case in observation.get('cases', [])[:64]:
            key = (observation.get('submission_sha256'), case.get('capsule'),
                   *(case.get(k) for k in ('model_sha256', 'comparison_domain_sha256', 'scope_sha256')))
            latest[key] = dict(case, action_id=observation.get('action_id'),
                               selfcheck_request_id=observation.get('selfcheck_request_id'),
                               submission_sha256=observation.get('submission_sha256'))
    rows = list(latest.values())
    groups = {}
    for row in rows[-1024:]:
        if (row.get('status') not in ('PROVISIONAL', 'QUALIFIED')
                or row.get('numerical_screen') != 'pass'
                or (row.get('structural_qualification') != 'pass' and row.get('trace_status') != 'pass')):
            continue
        identity = {k:row.get(k) for k in ('model_sha256', 'comparison_domain_sha256', 'scope_sha256',
                    'capsule_sha256', 'selection_sha256', 'facts_sha256', 'scope', 'scope_excludes')}
        group = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        groups.setdefault(group, dict(identity=identity, candidates=[]))['candidates'].append({
            k:row.get(k) for k in ('action_id', 'capsule', 'submission_sha256', 'elf_sha256',
                                 'estimated_whole_soc_cycles', 'structural_floor_to_model_ratio', 'timing_qualified')})
    for group in groups.values():
        group['candidates'].sort(key=lambda r:r['estimated_whole_soc_cycles'])
        for rank, candidate in enumerate(group['candidates'], 1):
            candidate['rank'] = rank
    return dict(schema='merlin.construction_performance_portfolio.v1', latest=rows[-1024:],
                latest_rows_omitted=max(0, len(rows)-1024), matched_rankings=list(groups.values()),
                ranking_scope='Exact model/domain/capsule/facts/selection/counter-scope match only; different scopes are not compared.',
                physical_utilization_available=False, capsule_passes=None, qualification_authority=False,
                activation_or_promotion=False)


def execute(plan, *, output, timeout_s, plan_sha256, command_workers=2, selfcheck_window=None, inbox=None):
    """Run independent construction while submitted checks remain in flight.

    Explicit local dependencies gate only their dependent action. A failed capsule
    never establishes a global construction barrier. Numerical verdicts remain the
    broker's output; command return codes are execution status, not capsule passes.
    after_finished waits for terminal evidence, including failed checks; it grants
    no eligibility and cannot override the broker's independent model gate.
    """
    validate(plan)
    if not 1 <= command_workers <= 16:
        raise ValueError("command_workers must be 1..16")
    if selfcheck_window is None:
        slots = int(os.environ.get("MERLIN_SELFCHECK_ORDINARY_JOBS", os.environ.get("MERLIN_GRADE_WORKERS", "16")))
        selfcheck_window = min(512, max(command_workers, 2 * max(1, slots)))
    if type(selfcheck_window) is not int or not 1 <= selfcheck_window <= 512:
        raise ValueError("selfcheck_window must be 1..512")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout_s
    states = {}
    reports = {}
    performance_reports = {}
    pending = {}
    running = {}
    events = root / "events.jsonl"

    def record(row, state, **details):
        states[row["id"]] = state
        value = {
            "action_id": row["id"],
            "capability": row["capability"],
            "boundary": row["boundary"],
            "kind": row["kind"],
            "state": state,
            "time_ns": time.time_ns(),
            "argv_sha256": hashlib.sha256(json.dumps(command(row)).encode()).hexdigest(),
            **details,
        }
        with events.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def stop(process):
        # Own descendants of each scripted action too. Detached evaluator jobs
        # belong to the ordinary broker, outside this client process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def start(row, argv, suffix):
        out = (root / f"{row['id']}.{suffix}.stdout").open("w")
        err = (root / f"{row['id']}.{suffix}.stderr").open("w")
        try:
            process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        except OSError as error:
            err.write(str(error) + "\n")
            out.close()
            err.close()
            pending.pop(row["id"], None)
            record(row, "failed", reason="command could not start", error=str(error))
            return
        running[row["id"]] = (row, process, out, err, suffix)
        if suffix == "submit":
            record(row, "running")

    def complete(row, process, out, err, suffix):
        out.close()
        err.close()
        stop(process)
        response = _response((root / f"{row['id']}.{suffix}.stdout").read_text()) if row["kind"] == "selfcheck" else {}
        request = response.get("request_id") or response.get("selfcheck_request_id")
        if process.returncode == 3 and request:
            pending[row["id"]] = (row, request)
            record(row, "pending", request_id=request, returncode=process.returncode)
        else:
            pending.pop(row["id"], None)
            if row["kind"] == "selfcheck":
                reports[row["id"]] = report_projection(response)
                performance = dict(performance_projection(response), action_id=row["id"],
                                   feedback_file=f"{row['id']}.{suffix}.stdout")
                performance_reports[row["id"]] = performance
                with (root / "performance.jsonl").open("a") as stream:
                    stream.write(json.dumps(performance, sort_keys=True, allow_nan=False) + "\n")
                temporary = root / ".performance-portfolio.json.pending"
                temporary.write_text(json.dumps(performance_portfolio(list(performance_reports.values())),
                                                sort_keys=True, allow_nan=False) + "\n")
                temporary.replace(root / "performance-portfolio.json")
            record(
                row,
                "completed" if process.returncode == 0 else "failed",
                returncode=process.returncode,
                request_id=request,
                feedback_file=f"{row['id']}.{suffix}.stdout",
                **({"report": reports[row["id"]]} if row["kind"] == "selfcheck" else {}),
                feedback={
                    key: response[key]
                    for key in (
                        "submission_sha256",
                        "requested_submission_sha256",
                        "all_pass",
                        "counts",
                        "n_pass",
                        "n_capsules",
                        "graded_newer_bytes",
                    )
                    if key in response
                },
            )

    remaining = list(plan["actions"])
    known = {row["id"] for row in remaining}
    admitted = {}
    seen_plans = set()
    accepting = inbox is not None
    inbox_root = Path(inbox) if accepting else None
    if accepting:
        inbox_root.mkdir(parents=True, exist_ok=True)

    def intake():
        """Read only atomically published plans, recording acceptance/refusal."""
        added = False
        for path in sorted(inbox_root.glob("*.json")):
            if path.name in seen_plans:
                continue
            seen_plans.add(path.name)
            event = {"file": path.name, "time_ns": time.time_ns()}
            try:
                if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError("inbox plan must be a bounded ordinary file")
                raw = path.read_bytes()
                incoming = validate(json.loads(raw), known=known)
                digest = hashlib.sha256(raw).hexdigest()
                remaining.extend(incoming["actions"])
                known.update(row["id"] for row in incoming["actions"])
                admitted[path.name] = digest
                event.update(status="admitted", plan_sha256=digest,
                             action_ids=[row["id"] for row in incoming["actions"]])
                added = True
            except (OSError, ValueError, TypeError) as error:
                event.update(status="refused", reason=str(error))
            with (root / "inbox-events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
        return added

    iteration = 0
    try:
        while remaining or pending or running or accepting:
            if time.monotonic() >= deadline:
                raise TimeoutError("action study wall budget exhausted")
            progress = False
            if accepting:
                progress = intake()
                # Stop admission, then drain already accepted independent jobs.
                accepting = not (inbox_root / "STOP").exists()
            # No blocking completion callbacks: inspect every child, then refill.
            for name, work in list(running.items()):
                if work[1].poll() is not None:
                    complete(*work)
                    del running[name]
                    progress = True
            for row in list(remaining):
                deps = [states.get(key) for key in row.get("after", [])]
                finished = [states.get(key) for key in row.get("after_finished", [])]
                if any(state in {"failed", "deferred"} for state in deps):
                    record(row, "deferred", reason="explicit local prerequisite failed")
                    remaining.remove(row)
                    progress = True
                elif (all(state == "completed" for state in deps)
                      and all(state in {"completed", "failed", "deferred"} for state in finished)
                      and len(running) < command_workers):
                    # The broker preserves each client's original deadline,
                    # including queue time. Do not enqueue the entire corpus
                    # and spend later clients' leases before a worker is free.
                    # Unrelated construction remains eligible at a full window.
                    outstanding = set(pending) | {
                        key for key, work in running.items() if work[0]["kind"] == "selfcheck"
                    }
                    if row["kind"] == "selfcheck" and len(outstanding) >= selfcheck_window:
                        continue
                    start(row, command(row), "submit")
                    remaining.remove(row)
                    progress = True
            # Keep new construction/test admission ahead of attach calls. An
            # attach is also asynchronous and cannot block another free slot.
            for row, request in list(pending.values()):
                if row["id"] not in running and len(running) < command_workers:
                    iteration += 1
                    start(
                        row,
                        [sys.executable, "agent_selfcheck.py", "--attach", request, "--wait-budget", "1"],
                        f"collect-{iteration}",
                    )
                    # Rotate collection fairly: a slow early request must not
                    # hide completed later requests and prevent admission refill.
                    pending.pop(row["id"])
                    pending[row["id"]] = (row, request)
                    progress = True
            if not progress:
                if remaining and not running and not pending:
                    raise RuntimeError("unresolved action dependencies")
                time.sleep(0.1 if not running and not pending else 0.02)
    except TimeoutError as error:
        unfinished = {row["id"]: row for row in remaining}
        unfinished.update({name: value[0] for name, value in pending.items()})
        unfinished.update({name: value[0] for name, value in running.items()})
        for row in unfinished.values():
            record(row, "unfinished", reason=str(error))
    finally:
        for _, process, out, err, _ in running.values():
            stop(process)
            out.close()
            err.close()
    result = {
        "schema": SCHEMA,
        "plan_sha256": plan_sha256,
        "selfcheck_window": selfcheck_window,
        "actions": states,
        "completed": sum(v == "completed" for v in states.values()),
        "selfcheck_reports": reports,
        "selfcheck_summary": {
            "completed_reports": sum(row["completed_report"] for row in reports.values()),
            "requested_numerical_screen_pass": sum(row["requested_numerical_screen"] == "pass" for row in reports.values()),
            "requested_numerical_screen_fail": sum(row["requested_numerical_screen"] == "fail" for row in reports.values()),
            "requested_numerical_screen_unknown": sum(row["requested_numerical_screen"] == "unknown" for row in reports.values()),
            "full_certification_pass": sum(row["full_certification"] is True for row in reports.values()),
            "scope": "returned selfcheck requests; not deduplicated capsule or campaign totals",
        },
        "performance_feedback": {"stream": "performance.jsonl", "portfolio": "performance-portfolio.json",
                                 "completed_requests": len(performance_reports), "qualification_authority": False},
        "inbox": {"path": str(inbox_root) if inbox_root else None, "admitted_plans": admitted},
        "capsule_passes": None,
        "qualification_authority": False,
        "note": "Action states preserve subprocess exit/dependency gates. A failed action may contain a completed numerical screen with incomplete certification; inspect selfcheck_reports and original broker evidence.",
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--selfcheck-window", type=int, default=None,
                    help="outstanding self-check clients; default twice the selected ordinary worker count")
    ap.add_argument("--inbox", help="stay warm for atomically published additional action plans; STOP drains accepted work")
    ap.add_argument(
        "--command-workers",
        type=int,
        default=2,
        choices=range(1, 17),
        help="concurrent ordinary commands/client calls; all inherit the same shared CPU reservation",
    )
    args = ap.parse_args(argv)
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    plan, digest = load(args.plan)

    def interrupted(_signum, _frame):
        raise TimeoutError("construction action runner interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    result = execute(
        plan, output=args.out, timeout_s=args.timeout, plan_sha256=digest, command_workers=args.command_workers,
        selfcheck_window=args.selfcheck_window,
        inbox=args.inbox,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if all(v == "completed" for v in result["actions"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
