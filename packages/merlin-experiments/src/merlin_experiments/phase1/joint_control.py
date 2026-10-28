"""Opt-in joint campaign policy inside the native Phase 1 process owner.

Private policy and wall clock are inherited only by trusted evaluation children.
A partial physical witness set never acquires the historical Phase 1 certificate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

ENV = "MERLIN_JOINT_CAMPAIGN_STATE"


def policy(path):
    value = json.loads(Path(path).read_text())
    if value.get("version") != 1:
        raise ValueError("joint campaign policy must have version 1")
    for key, default, low, high in (
        ("duration_s", 43200, 300, 43200),
        ("final_model_s", 10800, 1, 14400),
        ("drain_s", 1800, 1, 3600),
    ):
        value.setdefault(key, default)
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"invalid joint campaign {key}")
    if not value["drain_s"] <= value["final_model_s"] < value["duration_s"]:
        raise ValueError("joint campaign windows must be nested")
    mode = value.get("compiler_execution_mode", "profile_strict_v1")
    if mode not in {"profile_strict_v1", "snapshot_external_v1"}:
        raise ValueError("invalid joint compiler execution mode")
    external = mode == "snapshot_external_v1"
    gsim = value.setdefault("gsim", {})
    for key, default in (
        ("max_jobs", 20 if external else 2),
        ("per_job_s", 1200),
        ("worker_budget_s", value["duration_s"] * 20 if external else 14400),
    ):
        gsim.setdefault(key, default)
        if type(gsim[key]) is not int or not 1 <= gsim[key] <= default:
            raise ValueError(f"invalid joint gSIM {key}")
    names = gsim.setdefault("witness_capsules", [])
    if not isinstance(names, list) or any(not isinstance(x, str) or not x or x == "all" for x in names):
        raise ValueError("joint gSIM witnesses must be explicit capsule names")
    correctness = value.get("gsim_correctness")
    if correctness is not None:
        if not isinstance(correctness, dict):
            raise ValueError("joint gSIM correctness policy must be an object")
        names = correctness.get("capsules")
        if (
            not isinstance(names, list)
            or not names
            or len(names) > 2048
            or any(type(x) is not str or not x or x == "all" or "," in x for x in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint gSIM correctness requires an explicit unique capsule roster")
        original = value.get("correctness_curriculum", {}).get("original_capsules")
        if original is not None and set(names) != set(original):
            raise ValueError("physical correctness must retain every original correctness capsule")
        if type(correctness.get("continuous_promotion", False)) is not bool:
            raise ValueError("joint gSIM correctness continuous_promotion must be a boolean")
        for key, default, maximum in (
            ("max_jobs", 20, 20),
            ("per_job_s", 1200, 1200),
            ("worker_budget_s", 864000, 864000),
        ):
            correctness.setdefault(key, default)
            if type(correctness[key]) is not int or not 1 <= correctness[key] <= maximum:
                raise ValueError(f"invalid joint gSIM correctness {key}")
    # Optional sections are validated, never filled in: a stored policy must compare equal on resume.
    from .joint_gsim_measurement import SECTION as MEASUREMENT
    from .joint_gsim_measurement import validate as validate_measurement

    validate_measurement(value.get(MEASUREMENT))
    from .feedback.regression_sampling import SECTION as SAMPLING
    from .feedback.regression_sampling import validate as validate_sampling

    validate_sampling(value.get(SAMPLING))
    from .feedback.construction_plan import policy as construction_policy

    construction_policy(value.get("construction_plan"))
    from .joint_measurements import limits

    limits(value)
    prohibited_instruction_roles({"policy": value})
    return value


#: The joint policy's instruction rule: roles of the closed vocabulary the author's kernels may not issue.
ROLES_KEY = "prohibited_instruction_roles"


def prohibited_instruction_roles(state=None):
    """The roles the selected campaign policy prohibits (``()`` without a campaign or a rule).

    Read from the run's stored policy, so every grader of the campaign (background, self-check and
    final coverage) holds a kernel to the rule the run was launched with."""
    state = selected() if state is None else state
    roles = ((state or {}).get("policy") or {}).get(ROLES_KEY, [])
    if not isinstance(roles, list) or any(type(r) is not str for r in roles) or len(set(roles)) != len(roles):
        raise ValueError(f"joint campaign {ROLES_KEY} must be a list of unique role names")
    from merlin.compile.analysis.instruction_roles import check_roles

    try:
        check_roles(roles)
    except KeyError as error:
        raise ValueError(f"joint campaign {ROLES_KEY}: {error.args[0]}") from None
    return tuple(roles)


def instruction_policy_line(policy):
    """One plain prompt line naming the prohibited roles, derived from the policy ("" without any)."""
    roles = prohibited_instruction_roles({"policy": policy})
    if not roles:
        return ""
    from merlin.compile.analysis.instruction_roles import ROLES

    named = "; ".join(f"`{role}`: {ROLES[role]}" for role in roles)
    return (
        f"\n\nProhibited instruction role{'s' if len(roles) > 1 else ''} {named}. Build every schedule from "
        "the target's other commands; a case whose executed command stream issues a command with a "
        "prohibited role fails at L2.\n"
    )


def validate(options):
    if not getattr(options, "joint_campaign", ""):
        return
    selected_policy = policy(options.joint_campaign)
    if "compiler_execution_mode" in selected_policy and selected_policy["compiler_execution_mode"] != getattr(
        options, "compiler_execution", "profile_strict_v1"
    ):
        raise ValueError("joint policy and compiler execution mode differ")
    from .joint_seed import selected as selected_seed

    seed = selected_seed(options, selected_policy)
    if options.sandbox != "bwrap" or options.continuous or options.schedule != "continuous":
        raise ValueError("joint campaign requires native --schedule continuous and bwrap")
    if not options.fresh_backend and seed is None:
        raise ValueError("joint campaign requires --fresh-backend or an operator-reviewed authored checkpoint")
    if options.seal_current or options.qualify_submission:
        raise ValueError("joint campaign cannot use legacy qualification or sealing")


def configure(options, run_dir, *, resume=False):
    source = getattr(options, "joint_campaign", "")
    if not source:
        if os.environ.get(ENV):
            raise ValueError("joint campaign environment requires explicit selection")
        return None
    value = policy(source)
    path = Path(run_dir) / "joint_campaign.json"
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    from .joint_epochs import performance_pin

    timing = performance_pin()
    if resume:
        old = json.loads(path.read_text())
        if old["policy_sha256"] != digest or old["policy"] != value:
            raise ValueError("joint campaign policy changed on resume")
        if old.get("performance_selection") != timing:
            raise ValueError("joint performance selection changed on resume; use a reviewed epoch")
    else:
        from .joint_epochs import inherited_state

        inherited = inherited_state(options, value, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("joint campaign state already exists")
        state = inherited or {
            "version": 1,
            "policy": value,
            "policy_sha256": digest,
            "started_at_epoch": None,
            "performance_selection": timing,
        }
        state.setdefault("effective_settings", effective_settings(value))
        path.write_text(json.dumps(state, indent=2) + "\n")
        path.chmod(0o600)
    os.environ[ENV] = str(path.resolve())
    return value


def selected():
    path = os.environ.get(ENV)
    return json.loads(Path(path).read_text()) if path else None


def effective_settings(policy):
    """The derived knobs of the optional lanes this policy selects, recorded in the run's state."""
    from .feedback.regression_sampling import SECTION as SAMPLING
    from .feedback.regression_sampling import settings as sampling
    from .joint_gsim_measurement import SECTION as MEASUREMENT
    from .joint_gsim_measurement import settings as measurement

    return {MEASUREMENT: measurement(policy), SAMPLING: sampling(policy)}


def start():
    value = selected()
    if value is None:
        return
    if value["started_at_epoch"] is None:
        value["started_at_epoch"] = time.time()
        Path(os.environ[ENV]).write_text(json.dumps(value, indent=2) + "\n")


def remaining(*, drain=False, now=None):
    value = selected()
    if value is None:
        return None
    if value["started_at_epoch"] is None:
        raise RuntimeError("joint author clock has not started")
    end = value["started_at_epoch"] + value["policy"]["duration_s"]
    if drain:
        end -= value["policy"]["drain_s"]
    return max(0.0, end - (time.time() if now is None else now))


def require_full_model(*, engine="spike", now=None):
    value = selected()
    if value is None:
        return None
    left = remaining(now=now)
    external = value["policy"].get("compiler_execution_mode") == "snapshot_external_v1"
    if engine != "spike" or left <= 0 or (not external and left > value["policy"]["final_model_s"]):
        raise RuntimeError("joint full models are admitted only on Spike during the final model window")
    return left


def stop_reason():
    value = selected()
    if value is None:
        return None
    from .joint_epochs import stop_reason as epoch_stop

    return epoch_stop(Path(os.environ[ENV]).parent) or (
        "joint campaign freeze/drain boundary reached"
        if value["started_at_epoch"] is not None and remaining(drain=True) <= 0
        else None
    )


def gsim_ledger():
    value = selected()
    return Path(value.get("gsim_ledger", str(Path(os.environ[ENV]).with_name("joint_gsim_budget.json"))))


def correctness_capsules(value=None):
    """Operator-declared physical obligations, independent of calibration witnesses."""
    value = selected() if value is None else value
    return frozenset((value or {}).get("policy", {}).get("gsim_correctness", {}).get("capsules", ()))


def continuous_l3_promotion(value=None):
    """Whether an L2 pass enqueues its L3 certificate during the run (``gsim_correctness``
    ``continuous_promotion``; off unless a policy selects it). Off, L3 is a best-effort final step."""
    value = selected() if value is None else value
    return (value or {}).get("policy", {}).get("gsim_correctness", {}).get("continuous_promotion") is True


def held_capsules(value=None):
    value = selected() if value is None else value
    return set((value or {}).get("policy", {}).get("final_only_capsules", ())) - correctness_capsules(value)


def correctness_ledger():
    value = selected()
    return Path(
        value.get("gsim_correctness_ledger", str(Path(os.environ[ENV]).with_name("joint_gsim_correctness_budget.json")))
    )


def physical_policy(capsule):
    value = selected()
    return value["policy"]["gsim_correctness" if capsule in correctness_capsules(value) else "gsim"]


@contextmanager
def _correctness_data(path):
    """The correctness ledger under its lock; written back however the block ends."""
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        data = json.loads(handle.read() or '{"reserved_s":0,"active":[],"admissions":[]}')
        data.setdefault("waiting", [])
        try:
            yield data
        finally:
            handle.seek(0)
            handle.truncate()
            json.dump(data, handle)
            handle.flush()


def correctness_background_slots(starting=()):
    """Correctness gSIM slots background work may start now (0 while an author waits); read-only.

    ``starting``: the caller's launched request ids whose children have not taken a slot yet.
    """
    from . import joint_gsim_queue as Q

    value = selected()
    if value is None:
        return None
    path = correctness_ledger()
    try:
        with path.open() as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            data = json.loads(handle.read() or "{}")
    except FileNotFoundError:
        data = {}
    data = {"active": list(data.get("active", ())), "waiting": list(data.get("waiting", ()))}
    Q.settle(data)
    return Q.background_slots(data, value["policy"]["gsim_correctness"]["max_jobs"], starting)


@contextmanager
def physical_correctness(capsule, *, wait=None, progress=None, queue=None, poll_s=1.0):
    """Budget one mandatory run; native promotion owns prerequisites and reuse.

    This does not authorize model fitting or certify any output. Unknown, failed,
    interrupted and timed-out native results leave the correctness obligation open.
    Separate accounting prevents mandatory cases consuming calibration permissions.

    Every slot busy with budget left: an author request (``wait``, the default for its class)
    waits in the ledger's order, publishing its place to ``progress`` and recording the wait in
    ``queue``; background work raises :class:`.joint_gsim_queue.SlotBusy` instead. Budget
    exhaustion, the roster and the campaign window/stop stay refusals, also while waiting.
    """
    value = selected()
    if value is None:
        yield None
        return
    from . import joint_gsim_queue as Q
    from .joint_epochs import stop_reason as epoch_stop

    def closed():
        return remaining() <= 0 or epoch_stop(Path(os.environ[ENV]).parent)

    if capsule not in correctness_capsules(value) or closed():
        raise ValueError("gSIM correctness request outside required roster/campaign window")
    spec = value["policy"]["gsim_correctness"]
    request_class = Q.request_class()
    wait = request_class == Q.AUTHOR if wait is None else wait
    token = uuid.uuid4().hex
    path = correctness_ledger()
    queued_at = first_ahead = None
    try:
        while True:
            if closed():
                raise ValueError(
                    "campaign deadline or stop reached while queued for a gSIM correctness slot; "
                    "obligation remains pending"
                )
            reservation = min(float(spec["per_job_s"]), remaining())
            with _correctness_data(path) as data:
                Q.settle(data)  # Dead workers retain their reservation, but release concurrency.
                if data["reserved_s"] + reservation > spec["worker_budget_s"]:
                    Q.leave(data, token)
                    raise ValueError(
                        f"joint gSIM correctness worker budget exhausted ({data['reserved_s']:.0f} s reserved "
                        f"+ {reservation:.0f} s for this job > {spec['worker_budget_s']} s); "
                        "obligation remains pending"
                    )
                Q.enqueue(data, token, capsule=capsule, request_class=request_class, now=time.time(),
                          requester=Q.requester())
                state = Q.describe(data, token, spec["max_jobs"])
                if Q.admissible(data, token, spec["max_jobs"]):
                    Q.leave(data, token)
                    waited = None if queued_at is None else round(time.time() - queued_at, 1)
                    data["active"].append(
                        {
                            "pid": os.getpid(),
                            "token": token,
                            "capsule": capsule,
                            "request": os.environ.get(Q.REQUEST_ENV) or None,
                        }
                    )
                    data["admissions"].append(
                        {
                            "capsule": capsule,
                            "token": token,
                            "purpose": "required_correctness",
                            "at_epoch": time.time(),
                            "reserved_s": reservation,
                            **({"queued_s": waited, "queued_ahead": first_ahead} if waited is not None else {}),
                        }
                    )
                    data["reserved_s"] += reservation
                    break
                if not wait:
                    Q.leave(data, token)
                    raise Q.SlotBusy(
                        f"joint gSIM correctness concurrency full ({state['busy']} of {state['slots']} slots "
                        f"busy, {state['ahead']} waiting ahead); obligation remains pending and the request "
                        "stays queued"
                    )
            if queued_at is None:
                queued_at, first_ahead = time.time(), state["ahead"]
            Q.publish(progress, Q.WAITING_STATUS, dict(state, queued_since_epoch=queued_at))
            time.sleep(poll_s)
    except BaseException:
        try:
            with _correctness_data(path) as data:
                Q.leave(data, token)  # A refusal or an interrupted wait gives up its place.
        except (OSError, ValueError):
            pass  # the original failure is the one to report; a dead waiter is settled anyway
        raise
    if queued_at is not None:
        record = {
            "state": "admitted",
            "ahead_at_enqueue": first_ahead,
            "waited_s": round(time.time() - queued_at, 1),
            "note": f"queued for a gSIM slot ({first_ahead} ahead) for "
            f"{time.time() - queued_at:.0f} s before it started",
        }
        Q.publish(progress, Q.ADMITTED_STATUS, record)
        if queue is not None:
            queue.update(record)
    begun = time.monotonic()
    try:
        yield max(1, int(reservation))
    finally:
        with _correctness_data(path) as data:
            data["active"] = [r for r in data["active"] if r["token"] != token]
            data["reserved_s"] -= max(0, reservation - (time.monotonic() - begun))


@contextmanager
def physical_witness(capsule):
    """Reserve worst-case worker seconds before a single selected physical run.

    Reservations remain charged after a killed worker. No stale process can make
    the campaign exceed its aggregate budget by dropping an unfinished receipt.
    """
    value = selected()
    if value is None:
        yield None
        return
    spec = value["policy"]["gsim"]
    if capsule not in spec["witness_capsules"] or stop_reason():
        raise ValueError("gSIM request is outside the selected small witness roster/window")
    path = gsim_ledger()
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        data = json.loads(handle.read() or '{"reserved_s":0,"active":[]}')
        # A dead process retains its charge, but no longer occupies a worker slot.
        active = []
        for pid in data["active"]:
            try:
                os.kill(pid, 0)
                active.append(pid)
            except ProcessLookupError:
                pass
        if len(active) >= spec["max_jobs"] or data["reserved_s"] + spec["per_job_s"] > spec["worker_budget_s"]:
            raise ValueError("joint gSIM concurrency or aggregate worker budget exhausted")
        from .joint_measurements import admit

        admit(data, capsule, value["policy"])
        active.append(os.getpid())
        data.update(active=active, reserved_s=data["reserved_s"] + spec["per_job_s"])
        handle.seek(0)
        handle.truncate()
        json.dump(data, handle)
        handle.flush()
    begun = time.monotonic()
    try:
        yield min(spec["per_job_s"], max(1, int(remaining(drain=True))))
    finally:
        with path.open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            data = json.load(handle)
            data["active"].remove(os.getpid())
            # Return unused reservation only after the original worker completed.
            data["reserved_s"] -= max(0, spec["per_job_s"] - (time.monotonic() - begun))
            handle.seek(0)
            handle.truncate()
            json.dump(data, handle)


def selfcheck_policy(function):
    """Apply routing before any request can reach a simulator, in its owned child."""

    @wraps(function)
    def wrapped(argv=None, **kwargs):
        if selected() is None:
            return function(argv, **kwargs)
        args = list(sys.argv[1:] if argv is None else argv)
        if any(flag in args for flag in ("--candidate-diagnostics", "--caller-layout")):
            return function(argv, **kwargs)
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--sim", default="required")
        parser.add_argument("--capsules", default="all")
        parser.add_argument("--tiers", default="")
        parser.add_argument("--timeout", type=int, default=1200)
        parser.add_argument("--out")
        parser.add_argument("--progress-out", default="")
        request, rest = parser.parse_known_args(args)
        try:
            if remaining() <= 0:
                raise ValueError("joint campaign deadline reached")
            sim = "spike" if request.sim in {"required", "auto"} else request.sim
            if sim not in {"spike", "gsim"}:
                raise ValueError("joint development supports Spike and selected gSIM witnesses only")
            if sim == "spike" and any(
                t.strip() not in {"L0", "L1", "L2"} for t in request.tiers.split(",") if t.strip()
            ):
                raise ValueError("joint Spike requests cannot select physical tiers")
            forwarded = rest + ["--sim", sim, "--capsules", request.capsules]
            if request.tiers:
                forwarded += ["--tiers", request.tiers]
            if request.out:
                forwarded += ["--out", request.out]
            if request.progress_out:
                forwarded += ["--progress-out", request.progress_out]
            if sim == "gsim" and request.capsules.startswith("panel:"):  # feedback.shape_panel admits each case
                left = max(1, int(remaining()))
                return function(forwarded + ["--timeout", str(min(request.timeout, left))], **kwargs)
            if sim == "gsim":
                from .joint_gsim_measurement import admission as measurement
                from .joint_gsim_measurement import requested as measurement_request
                from .gsim_wall_cap import request_refusal

                over_cap = None if measurement_request() is not None else request_refusal(request.capsules)
                if over_cap is not None:
                    raise ValueError(over_cap)  # before any slot is taken or waited for

                admission = (
                    measurement
                    if measurement_request() is not None
                    else physical_correctness
                    if request.capsules in correctness_capsules()
                    else physical_witness
                )
                queue = {}
                options = (
                    {"progress": request.progress_out or None, "queue": queue}
                    if admission is physical_correctness
                    else {}
                )
                with admission(request.capsules, **options) as budget:
                    status = function(forwarded + ["--timeout", str(min(request.timeout, budget))], **kwargs)
                if queue and request.out:
                    _note_queue_wait(Path(request.out), queue)
                return status
            return function(forwarded + ["--timeout", str(min(request.timeout, max(1, int(remaining()))))], **kwargs)
        except ValueError as exc:
            report = {"all_pass": False, "error": str(exc), "joint_policy_refused": True}
            if getattr(exc, "retryable", False):
                report["joint_policy_retryable"] = True
            if request.out:
                Path(request.out).write_text(json.dumps(report))
            print(json.dumps(report))
            return 2

    return wrapped


def _note_queue_wait(path, queue):
    """Say on the verdict that it waited for a gSIM slot before it started."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if isinstance(document, dict):
        document["gsim_slot_queue"] = queue
        path.write_text(json.dumps(document))


def selfcheck_routing_guidance():
    """Author-facing description of the selected joint wrapper, not a new policy."""
    return (
        "Without `--sim`, or with `--sim required`/`--sim auto`, this joint campaign selects Spike "
        "L0/L1/L2 development checks; this does not certify unrun physical tiers. "
        "Explicit `--sim gsim` accepts one named required original correctness capsule under its "
        "separate correctness budget, or one calibration witness subject to question review. "
        "Required correctness does not consume calibration permissions or qualify a fit. "
        "When every correctness gSIM slot is busy the request waits for one in arrival order "
        "(its progress says `queued for a gSIM slot (N ahead)`); collect it later with `--attach`. "
        "Unlisted cases, multi-capsule gSIM requests and "
        "scalability gSIM/L3 requests are refused." + _measurement_guidance()
    )


def _measurement_guidance():
    from .joint_gsim_measurement import guidance

    text = guidance()
    return " " + text if text else ""


def prompt_block():
    value = selected()
    if value is None:
        return ""
    return _campaign_prompt_block(value["policy"]) + instruction_policy_line(value["policy"])


def _l3_schedule_text(p):
    """Author-facing: when L3 certification runs, and which engine measures performance."""
    if (p.get("gsim_correctness") or {}).get("continuous_promotion") is True:
        return ("Every original capsule selected in gsim_correctness retains both Spike and gSIM "
                "correctness obligations; the native queue promotes eligible exact snapshots after cheap "
                "prerequisites and reuses matching native evidence.")
    return ("L3 gSIM certification of the selected correctness cases runs at the end, best-effort, on "
            "the final compiler; development grades stop at Spike L0-L2. Performance is measured on the "
            "Spike cycle model by default; ask for gSIM cycles with --sim gsim.")


def _campaign_prompt_block(p):
    from .full_compiler_workspace import selected as full_selection

    if p.get("compiler_execution_mode") == "snapshot_external_v1" and full_selection() is not None:
        return _full_compiler_prompt_block(p)
    return "\n\n## Joint compiler development campaign\n\n" + (
        f"The {p['duration_s'] / 3600:g}-hour clock starts at author launch. Small numerical success "
        "does not freeze the compiler. Keep architecture, pass composition, fusion and scheduling editable. "
        "Use the shared compiler and host passes; consume transformed IR in the actual emitted program. "
        "Select and reorder passes with semantic/resource checks. Derive tiling, fusion, memory layouts, "
        "double buffering, pipelining, spilling and resource overlap from input semantics and target facts. "
        "Keep large-shape loops compact. Build a general compiler with shape heuristics and fallbacks. "
        "Use the native parallel agents, immutable selfcheck snapshots, caches and asynchronous jobs. "
        "Each worker reports touched modules, named focused checks, the first actionable failure, "
        "concise correctness/performance/model confidence and artifact paths; never paste raw logs. "
        "After edits, run relevant named selfcheck subsets. Optionally declare affected operation "
        'families in submission/joint-test-scope.json as {"families":["matmul"]}; this only '
        "prioritizes the rolling correctness queue and grants no coverage. Alternatively name 1–16 unique "
        'permitted development IDs with {"capsules":["case_id"]}; do not combine this with families. '
        "This is a priority hint, not a limit on queued cases or public IDs available to named selfchecks. "
        "Automatic grading refills the unresolved queue on an immutable snapshot within a bounded "
        "admission window; only actual starts count for fairness. At most two immutable compiler "
        "snapshots execute concurrently; all work shares the dynamic CPU budget while authoring continues. "
        "Use exact cases for relevant tails, intermediate resource regimes, connected cuts and failures. "
        "Unknown impact uses original correctness prerequisites and actual-admission fairness. Run full required coverage at explicit milestones and final validation, "
        "not after every edit. Read feedback_brief first; unselected cases remain pending. "
        "Check large capsules and connected cuts early using --suite scalability --sim spike. "
        "Default selfchecks use Spike. " + _l3_schedule_text(p) + " Timeouts and "
        "unrun cases remain pending, never passes. Inspect each required numerical/structural tier. "
        "Read pinned examples under reference-kernels when that readonly input is declared; they are "
        "references, while your target backend starts empty and evolves with your own general passes. "
        "Optimize performance from the first working compiler: run small and large admitted Spike "
        "capsules early, including connected cuts and the admitted reference-workload shapes. Read "
        "the feedback's selected model version, tuning status, validity domain and qualification. "
        "For covered regimes with a valid tuned model, use that selection; never silently fall back "
        "to untuned timing. Outside coverage, record provisional/unknown and request a targeted bridge. "
        "A partial structural roofline is a lower bound, not a qualified attainable ceiling. "
        "Use whole-SoC estimates, including host and accelerator interaction, for prioritization; "
        "uncalibrated guest cycles are not physical cycle evidence. "
        "The separate calibration-witness whitelist grants permission, not a queue: request an extra run only for a specific missing "
        "mechanism or timing question after numerical correctness and emitted-schedule inspection. "
        "Reuse matching evidence first; stop when uncertainty cannot change schedule ranking. "
        "At most two initial diagnostics run before further question-labelled operator authorization; "
        "bootstrap has at most eight observations and later batches have at most four. "
        "Estimates (complete timing-model estimates, else the calibrated fast estimate each passing case "
        "carries) are the basis for optimization decisions: prefer the change the estimate favors, label "
        "its evidence kind, and revert if a later measurement disagrees. "
        "Shared library bytes are immutable within an epoch; "
        "record generic pass improvements for an operator-reviewed library epoch that preserves your compiler. "
        f"Full model development prevalidation uses Spike in the final {p['final_model_s'] / 3600:g} hours. "
        f"Author editing stops {p['drain_s'] / 60:g} minutes before the deadline for freeze/drain. "
        "After final compiler freezing, the operator must run every admitted capsule and every full "
        "target model on FireSim using the exact selected unmodified hardware configuration. "
        "This final hardware stage is separate from development prevalidation; no capsule, source "
        "shape or model may disappear from its roster. Bind actual boot images, all-output checks, "
        "measured cycle scopes and comparable software baselines. A wrapper exit is not a pass. "
        "This campaign reports its own evidence and does not issue a legacy all-corpus RTL certificate.\n\n"
        "### Boundary checklist before broad testing\n\n"
        "Run one tiny applicable case through actual emission, target consumption, linking, Spike and "
        "complete output comparison. A ready tool is not a passing compiler. Resolve shared contract "
        "failures before repeating affected requests. Downstream emission consumes transformed target "
        "IR and its attributes, not an assumed original interface. Emit raw schema-valid command-buffer "
        "JSON with the required abi_version; do not add a transport envelope. Respect output-file versus "
        "stdout channels and send diagnostics to stderr. Derive entry/completion symbols and argument "
        "layouts from the selected ABI, resolve every called helper, and complete device writes before "
        "readback. Bind all named output pointers, strides and lifetimes, including multiple outputs "
        "and state updates. Whole-program L0/L1 non-applicability requires native justification and "
        "complete Spike output checks; expected refusals are separate from numerical passes. "
        "After changes check relevant tails, rounding/saturation, activation/readout, stride/padding "
        "and connected multi-output boundaries. No public contract or required gate may be weakened.\n"
    )


def development_roots(root):
    """Select the declared development cohort without changing frozen capsule bytes."""
    value = selected()
    held = held_capsules(value)
    if not held:
        return root
    roots = sorted(path.parent for path in Path(root).rglob("capsule.yaml") if path.parent.name not in held)
    if not roots:
        raise ValueError("joint development cohort is empty")
    return roots


def stage_evidence(verdict):
    value = selected()
    if value is not None:
        verdict["joint_campaign"] = {
            "scope": "development",
            "physical_certificate": False,
            "final_only_pending": sorted(held_capsules(value)),
            "physical_correctness_required": sorted(correctness_capsules(value)),
            "physical_correctness_schedule": "during the run" if continuous_l3_promotion(value)
            else "L3 is certified at the end, best-effort; development rows are not L3 failures",
            "performance_qualification": "unknown",
        }
    return verdict


def _full_compiler_prompt_block(policy):
    """Editable compiler advice plus actual selected physical/resource limits."""
    from .joint_measurements import limits

    bounds = limits(policy)
    witness = policy.get("gsim", {})
    correctness = policy.get("gsim_correctness", {})
    return (
        "\n\n## Joint editable compiler development\n\n"
        "Develop one general compiler across all public capability families. A fresh run has an empty "
        "target backend; a declared continuation retains its authored compiler. Generic compiler sources "
        "under submission/compiler_library are editable: extend passes, dialects, numerical/layout "
        "representations, schedules, allocation, lowering and host generation as useful. There is no "
        "mandatory pass order, discovery sequence, tiny-canary prerequisite or minimum helper count. "
        "Choose component ownership and testing order; up to the selected helper ceiling can work "
        "concurrently toward the same compiler. Use all 20 CPUs as one shared pool.\n\n"
        "Test useful eligible cases as soon as they can provide feedback, including large and connected "
        "cases. A tiny executable can cheaply expose ABI faults but does not gate unrelated work. "
        "Keep editing while immutable snapshots grade and integrate completed changes promptly. "
        "Consume transformed IR in real emission; preserve all outputs, numerical contracts, source "
        "placement and completion. No private answers, retained target backend seed, benchmark-name "
        "dispatch or forbidden CPU fallback. An import, receipt or build is not numerical evidence.\n\n"
        "All 250 selected cases remain eventual obligations: 153 original correctness, 20 calibration "
        "and 77 scalability cases, with their actual required tiers. Optimize chiefly larger shapes and "
        "connected workloads from the start; small originals primarily establish correctness and model "
        "evidence. Full-model Spike development checks are available throughout the remaining campaign; "
        "there is no final-model-window waiting requirement in this execution mode. "
        + selfcheck_routing_guidance()
        + "\n\n"
        f"Selected campaign duration: {policy['duration_s'] / 3600:g} hours from author launch; "
        f"editing stops {policy['drain_s'] / 60:g} minutes before deadline for freeze/drain. "
        + (f"Required gSIM correctness: {len(correctness.get('capsules', ()))} declared cases, "
           f"concurrency {correctness.get('max_jobs', 'unselected')}, "
           f"per-job budget {correctness.get('per_job_s', 'unselected')} seconds. "
           if correctness.get("continuous_promotion") is True else _l3_schedule_text(policy) + " ")
        + f"Selective calibration gSIM concurrency {witness.get('max_jobs', 'unselected')}, "
        f"per-job budget {witness.get('per_job_s', 'unselected')} seconds; "
        f"initial diagnostics {bounds['initial_diagnostics_max']}, bootstrap observations "
        f"{bounds['bootstrap_observations_max']}, later authorized batch limit {bounds['later_batch_max']}. "
        "These are actual admission limits, not required exploration stages. Extra witnesses need "
        "matching numerical evidence and an unresolved question under the selected policy; required "
        "correctness has separate accounting. Unrun tiers and timeouts remain pending.\n\n"
        "Use roofline and whole-SoC feedback with its scope and uncertainty. Prefer the selected tuned "
        "model when qualified; missing timing is not zero cost. Independent final FireSim validation "
        "uses the exact selected unchanged hardware configuration and complete output checks. "
        "Report observed correctness and performance separately; private grading remains independent.\n"
    )
