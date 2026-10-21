"""Opt-in joint campaign policy inside the native Phase 1 process owner.

Private policy and wall clock are inherited only by trusted evaluation children.
A partial physical witness set never acquires the historical Phase 1 certificate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ENV = "MERLIN_JOINT_CAMPAIGN_STATE"


def policy(path):
    value = json.loads(Path(path).read_text())
    if value.get("version") != 1:
        raise ValueError("joint campaign policy must have version 1")
    for key, default, low, high in (("duration_s", 43200, 3600, 43200),
                                   ("final_model_s", 10800, 1, 14400),
                                   ("drain_s", 1800, 1, 1800)):
        value.setdefault(key, default)
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"invalid joint campaign {key}")
    if not value["drain_s"] < value["final_model_s"] < value["duration_s"]:
        raise ValueError("joint campaign windows must be nested")
    gsim = value.setdefault("gsim", {})
    for key, default in (("max_jobs", 2), ("per_job_s", 1200), ("worker_budget_s", 14400)):
        gsim.setdefault(key, default)
        if type(gsim[key]) is not int or not 1 <= gsim[key] <= default:
            raise ValueError(f"invalid joint gSIM {key}")
    names = gsim.setdefault("witness_capsules", [])
    if not isinstance(names, list) or any(not isinstance(x, str) or not x or x == "all" for x in names):
        raise ValueError("joint gSIM witnesses must be explicit capsule names")
    from .joint_measurements import limits
    limits(value)
    return value


def validate(options):
    if not getattr(options, "joint_campaign", ""):
        return
    policy(options.joint_campaign)
    if options.sandbox != "bwrap" or options.continuous or options.schedule != "continuous":
        raise ValueError("joint campaign requires native --schedule continuous and bwrap")
    if not options.fresh_backend:
        raise ValueError("joint campaign requires --fresh-backend (resume preserves its own compiler)")
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
        path.write_text(json.dumps(inherited or {"version": 1, "policy": value, "policy_sha256": digest,
                                    "started_at_epoch": None, "performance_selection": timing}, indent=2) + "\n")
        path.chmod(0o600)
    os.environ[ENV] = str(path.resolve())
    return value


def selected():
    path = os.environ.get(ENV)
    return json.loads(Path(path).read_text()) if path else None


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
    if engine != "spike" or left <= 0 or left > value["policy"]["final_model_s"]:
        raise RuntimeError("joint full models are admitted only on Spike during the final model window")
    return left


def stop_reason():
    value = selected()
    if value is None:
        return None
    from .joint_epochs import stop_reason as epoch_stop
    return epoch_stop(Path(os.environ[ENV]).parent) or (
        "joint campaign freeze/drain boundary reached"
        if value["started_at_epoch"] is not None and remaining(drain=True) <= 0 else None)


def gsim_ledger():
    value = selected()
    return Path(value.get("gsim_ledger", str(Path(os.environ[ENV]).with_name("joint_gsim_budget.json"))))


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
        handle.seek(0); handle.truncate(); json.dump(data, handle); handle.flush()
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
            handle.seek(0); handle.truncate(); json.dump(data, handle)


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
        request, rest = parser.parse_known_args(args)
        try:
            if remaining() <= 0:
                raise ValueError("joint campaign deadline reached")
            sim = "spike" if request.sim in {"required", "auto"} else request.sim
            if sim not in {"spike", "gsim"}:
                raise ValueError("joint development supports Spike and selected gSIM witnesses only")
            if sim == "spike" and any(t.strip() not in {"L0", "L1", "L2"} for t in request.tiers.split(",") if t.strip()):
                raise ValueError("joint Spike requests cannot select physical tiers")
            forwarded = rest + ["--sim", sim, "--capsules", request.capsules]
            if request.tiers:
                forwarded += ["--tiers", request.tiers]
            if request.out:
                forwarded += ["--out", request.out]
            if sim == "gsim":
                with physical_witness(request.capsules) as budget:
                    return function(forwarded + ["--timeout", str(min(request.timeout, budget))], **kwargs)
            return function(forwarded + ["--timeout", str(min(request.timeout, max(1, int(remaining()))))], **kwargs)
        except ValueError as exc:
            report = {"all_pass": False, "error": str(exc), "joint_policy_refused": True}
            if request.out:
                Path(request.out).write_text(json.dumps(report))
            print(json.dumps(report))
            return 2
    return wrapped


def prompt_block():
    value = selected()
    if value is None:
        return ""
    p = value["policy"]
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
        "families in submission/joint-test-scope.json as {\"families\":[\"matmul\"]}; this only "
        "guides bounded advisory smoke and grants no coverage. Unknown impact gets representative "
        "family smoke. Run full required coverage at explicit milestones and final validation, "
        "not after every edit. Read feedback_brief first; unselected cases remain pending. "
        "Check large capsules and connected cuts early using --suite scalability --sim spike. "
        "Default selfchecks use Spike; gSIM accepts one named small witness from the operator roster. "
        "Use L0/L1/L2 evidence for development correctness. Intentionally withheld L3 is not a defect "
        "to fix or a request to simulate the full corpus. Inspect each required numerical/structural tier. "
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
        "A gSIM whitelist grants permission, not a queue: request a run only for a specific missing "
        "mechanism or timing question after numerical correctness and emitted-schedule inspection. "
        "Reuse matching evidence first; stop when uncertainty cannot change schedule ranking. "
        "At most two initial diagnostics run before further question-labelled operator authorization; "
        "bootstrap has at most eight observations and later batches have at most four. "
        "Unknown timing qualification stays provisional; never claim physical speedup from it. "
        "Shared library bytes are immutable within an epoch; "
        "record generic pass improvements for an operator-reviewed library epoch that preserves your compiler. "
        f"Full model development prevalidation uses Spike in the final {p['final_model_s'] / 3600:g} hours. "
        f"Author editing stops {p['drain_s'] / 60:g} minutes before the deadline for freeze/drain. "
        "After final compiler freezing, the operator must run every admitted capsule and every full "
        "target model on FireSim using the exact selected unmodified hardware configuration. "
        "This final hardware stage is separate from development prevalidation; no capsule, source "
        "shape or model may disappear from its roster. Bind actual boot images, all-output checks, "
        "measured cycle scopes and comparable software baselines. A wrapper exit is not a pass. "
        "This campaign reports its own evidence and does not issue a legacy all-corpus RTL certificate.\n"
    )


def development_roots(root):
    """Select the declared development cohort without changing frozen capsule bytes."""
    value = selected()
    held = set(value["policy"].get("final_only_capsules", ())) if value else set()
    if not held:
        return root
    roots = sorted(path.parent for path in Path(root).rglob("capsule.yaml") if path.parent.name not in held)
    if not roots:
        raise ValueError("joint development cohort is empty")
    return roots


def stage_evidence(verdict):
    value = selected()
    if value is not None:
        verdict["joint_campaign"] = {"scope": "development", "physical_certificate": False,
                                     "final_only_pending": value["policy"].get("final_only_capsules", []),
                                     "performance_qualification": "unknown"}
    return verdict
