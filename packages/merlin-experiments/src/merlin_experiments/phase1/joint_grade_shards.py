"""Process-sharded native grading for the joint campaign's final coverage.

One in-process ``CG.grade`` over ~270 capsules is bound by the host process: the per-invocation custody
work (invocation snapshot entry, package policy, answer-mask mounts) and the GIL serialize every worker
thread. Measured: ~2 capsules/min through one controller process against ~15 capsules/min when the same
roster is graded by ceil(slots/2) independent processes of two workers each.

This module shards ``grade_group`` of :mod:`joint_final_coverage`:

* the roster is split into ``extends``-closed shards (a capsule is scheduled after the sibling it extends
  has published, so a component never straddles two processes);
* every shard is a fresh interpreter (never ``fork``: the parent owns threads, flocks and memfds) that
  reopens the original owners exactly as the controller does -- profile service, compiler library,
  shared-pass requirement, ``compiler_execution`` -- and runs the UNCHANGED ``CG.grade`` over the native
  capsule directories of its shard, so each capsule goes through the same custody and answer masking;
* the parent merges the shard scores into one score of the ordinary shape and hands it to the unchanged
  ``_apply_score``. A shard that did not return, returned another roster, or was killed at the deadline
  leaves its rows ``gated`` (deferred, never pass); a merged score can only close a row some shard closed.

Opt-in with ``MERLIN_JOINT_GRADE_PROCESSES`` (``auto`` or an integer); unset/0 keeps the in-process
route byte-for-byte. The count is clamped to ``ceil(slots / 2)`` so processes x 2 workers never exceed
the shared compute pool; the pool's flock'd slots still arbitrate every CPU step across processes.
Nothing here reads hidden capsule data: a shard receives only native capsule directory paths.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ENV = "MERLIN_JOINT_GRADE_PROCESSES"
WORKERS_PER_PROCESS = 2
#: Seconds a shard may outlive the group deadline (running capsules drain at their next checkpoint).
GRACE_S = 90
_POLL_S = 0.5
_KILL_WAIT_S = 10
_SCHEMA = "merlin.joint_grade_shard.v1"

_AND_KEYS = (
    "gradeable",
    "numeric_all_exact",
    "structural_evidence_all_pass",
    "trace_all_pass",
    "structural_pass",
    "model_execution_all_pass",
    "provided_host_execution_all_pass",
)
_SUM_KEYS = (
    "functional_pass",
    "n_capsules",
    "n_passed",
    "n_structural_pass",
    "host_guard_total",
    "host_guard_pass",
    "device_lane_scored_pass",
    "device_evidence_count",
    "spec_qualified_device_pass",
    "device_unqualified_pass",
    "host_lane_scored_pass",
    "n_not_graded_ineligible",
    "n_gated_deferred",
    "n_not_measured",
    "n_budget_exhausted",
    "n_cert_not_measured",
    "n_screened_only",
    "n_declined",
    "n_incomplete",
    "n_infrastructure_fault",
    "n_not_gradeable_no_oracle",
)
_UNION_KEYS = (
    "host_compute_vetoed",
    "not_graded_ineligible",
    "gated_deferred",
    "not_measured",
    "budget_exhausted",
    "cert_not_measured",
)
_COUNTER_KEYS = ("tier_reached", "first_failure_planes", "trace_requirement_summary")


def slot_capacity() -> int:
    from merlin.common.cpu_pool import compute_capacity

    pool = compute_capacity()
    if pool:
        return pool
    from merlin.targetgen.capsule_grade import default_grade_workers

    return default_grade_workers()


def configured_processes(environ=None, *, slots: int | None = None) -> int:
    """Shard-process count requested by the operator; 0 keeps the in-process grader."""
    raw = (os.environ if environ is None else environ).get(ENV, "").strip().lower()
    if raw in {"", "0", "off", "false", "no"}:
        return 0
    limit = max(1, math.ceil((slot_capacity() if slots is None else slots) / WORKERS_PER_PROCESS))
    if raw == "auto":
        return limit
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{ENV} must be 'auto' or a nonnegative integer")
    return min(int(raw), limit)


def partition(caps, processes: int):
    """``extends``-closed shards, balanced by size, deterministic; None when sharding is unsafe/useless."""
    names = [str(cap["name"]) for cap in caps]
    if len(set(names)) != len(names):
        raise ValueError("shard roster has duplicate capsule names")
    if processes < 2 or len(caps) < 2:
        return None
    if any(cap.get("kind") == "model" for cap in caps):
        return None  # a gated whole-model capstone needs its op-suite denominator in one scheduler
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for cap in caps:
        target = cap.get("extends")
        if isinstance(target, str) and target in parent:
            a, b = find(str(cap["name"])), find(target)
            if a != b:
                parent[max(a, b)] = min(a, b)
    components: dict[str, list[str]] = {}
    for n in sorted(names):
        components.setdefault(find(n), []).append(n)
    ordered = sorted(components.values(), key=lambda c: (-len(c), c[0]))
    count = min(processes, len(ordered))
    if count < 2:
        return None
    bins: list[list[str]] = [[] for _ in range(count)]
    for component in ordered:
        bins[min(range(count), key=lambda i: (len(bins[i]), i))].extend(component)
    by_name = {str(cap["name"]): cap for cap in caps}
    return [[by_name[n] for n in sorted(b)] for b in bins]


# ---------------------------------------------------------------------------------------------- merge


def _and(values):
    if any(v is False for v in values):
        return False
    return True if values and all(v is True for v in values) else None


def _admission(caps, root, descriptor_sha256):
    from merlin.targetgen import capsule_grade as CG

    record = CG._materialized_cohort_admission(root, caps, expected_descriptor_sha256=descriptor_sha256)
    if record is not None:
        return record
    names = sorted(str(c["name"]) for c in caps)
    return {
        "version": 1,
        "policy": "all_discovered",
        "n_source_capsules": len(caps),
        "n_admitted_capsules": len(caps),
        "n_capability_excluded": 0,
        "n_resource_excluded": 0,
        "excluded_name_set_sha256": CG._name_set_sha256([]),
        "admitted_name_set_sha256": CG._name_set_sha256(names),
    }


def _deferred_row(cap, reason):
    return {
        "capsule": str(cap["name"]),
        "label": cap.get("label"),
        "status": "gated",
        "numeric": None,
        "trace": None,
        "tiers": {},
        "background_deferred": reason,
    }


def merge_scores(shards, *, caps, admission, processes=None, wall_s=None):
    """One ordinary score from ``shards`` = [(assigned caps, score dict | None, failure str | None)].

    A shard whose score is absent, malformed, or does not cover exactly its assigned names contributes
    only deferred (gated) rows; it can never contribute a pass.
    """
    good, bad, notes = [], [], []
    for index, (assigned, score, failure) in enumerate(shards):
        names = sorted(str(c["name"]) for c in assigned)
        rows = score.get("per_capsule") if isinstance(score, dict) else None
        if failure is None and isinstance(score, dict) and isinstance(rows, list):
            got = [r.get("capsule") if isinstance(r, dict) else None for r in rows]
            if sorted(map(str, got)) == names and len(set(got)) == len(got):
                good.append((index, assigned, score))
                continue
            failure = "shard returned a roster other than its assignment"
        if failure is None:
            detail = (score or {}).get("failure") if isinstance(score, dict) else None
            failure = f"shard produced no rows{': ' + json.dumps(detail, sort_keys=True)[:300] if detail else ''}"
        bad.append((index, assigned, failure, score if isinstance(score, dict) else {}))
        notes.append({"shard": index, "failure": failure})

    merged: dict = {}
    scores = [s for _, _, s in good]
    if scores:
        merged = json.loads(json.dumps(scores[0]))  # first shard supplies every unmerged field verbatim
    divergent = sorted(
        k
        for k in (scores[0] if scores else {})
        if k not in {*_AND_KEYS, *_SUM_KEYS, *_UNION_KEYS, *_COUNTER_KEYS, "per_capsule"}
        and any(s.get(k) != scores[0].get(k) for s in scores[1:])
    )
    for key in _AND_KEYS:
        values = [s.get(key) for s in scores]
        merged[key] = _and(values)
    for key in _SUM_KEYS:
        merged[key] = sum(s.get(key, 0) for s in scores if type(s.get(key, 0)) is int)
    for key in _UNION_KEYS:
        union = sorted({x for s in scores for x in (s.get(key) or [])})
        if union or key in merged:
            merged[key] = union
    for key in _COUNTER_KEYS:
        total: dict = {}
        for s in scores:
            for k, v in (s.get(key) or {}).items():
                total[k] = total.get(k, 0) + v if type(v) is int else v
        merged[key] = total
    rows = [r for s in scores for r in s["per_capsule"]]
    deferred_names = []
    for _, assigned, failure, _ in bad:
        for cap in assigned:
            rows.append(_deferred_row(cap, failure))
            deferred_names.append(str(cap["name"]))
    merged["per_capsule"] = sorted(rows, key=lambda r: str(r["capsule"]))
    if deferred_names:
        merged["n_gated_deferred"] += len(deferred_names)
        merged["gated_deferred"] = sorted({*merged.get("gated_deferred", []), *deferred_names})
    merged["cohort_admission"] = dict(admission)
    # Expected-refusal projection: shards without a declared refusal carry no scope; they are executable.
    scopes = []
    for _, assigned, score in good:
        scope = score.get("expected_refusal_scope")
        scopes.append(
            scope
            if isinstance(scope, dict)
            else {
                "binding": "original_capsule_declaration_and_canonical_result_v1",
                "n_admitted_capsules": len(assigned),
                "n_declared_refusals": 0,
                "n_validated_refusals": 0,
                "n_executable_capsules": len(assigned),
            }
        )
    if any(isinstance(s.get("expected_refusal_scope"), dict) for s in scores):
        fields = ("n_admitted_capsules", "n_declared_refusals", "n_validated_refusals", "n_executable_capsules")
        merged["expected_refusal_scope"] = {
            **scopes[0],
            **{
                f: sum(sc.get(f, 0) for sc in scopes)
                + sum(len(a) for _, a, _, _ in bad if f in {"n_admitted_capsules", "n_executable_capsules"})
                for f in fields
            },
        }
    evidence = [s.get("functional_public_evidence") for s in scores]
    if evidence and all(isinstance(e, dict) for e in evidence):
        fpe = dict(evidence[0])
        for field in ("n_admitted_capsules", "n_validated_refusals", "n_spike_passed", "n_higher_tier_measured"):
            if field in fpe:
                fpe[field] = sum(e.get(field, 0) for e in evidence if type(e.get(field, 0)) is int)
        if len({e.get("claim") for e in evidence}) != 1:
            fpe["claim"] = None
        merged["functional_public_evidence"] = fpe
    else:
        merged["functional_public_evidence"] = {
            **(evidence[0] if evidence and isinstance(evidence[0], dict) else {}),
            "claim": None,
        }
    integrity = [s.get("integrity_status") for s in scores]
    merged["integrity_status"] = next((i for i in integrity if i != "clean"), "clean" if scores else None)
    timing = [s.get("timing_rollup") or {} for s in scores]
    merged["timing_rollup"] = {
        **{
            k: round(sum(t.get(k, 0) for t in timing if type(t.get(k, 0)) in (int, float)), 3)
            for k in ("build_s", "sim_active_s", "active_total_s", "oracle_wait_s")
        },
        "suite_wall_s": max((t.get("suite_wall_s", 0) for t in timing), default=0),
        "max_workers": WORKERS_PER_PROCESS * len(good),
    }
    merged["timing_diagnostic"] = {k: v for s in scores for k, v in (s.get("timing_diagnostic") or {}).items()}
    if not scores:
        failures = [s.get("failure") for _, _, _, s in bad if isinstance(s, dict) and s.get("failure")]
        merged.update(task=None, gradeable=False, numeric_all_exact=False, structural_evidence_all_pass=False)
        if failures:
            merged["failure"] = failures[0]
    elif any(s.get("failure") for s in scores):
        merged["failure"] = next(s["failure"] for s in scores if s.get("failure"))
    passed = sum(1 for r in merged["per_capsule"] if r.get("status") == "pass")
    merged["headline"] = (
        f"{merged['n_passed']}/{merged['n_capsules']} measured; {passed}/{len(caps)} pass of the roster "
        f"(merged from {len(good)}/{len(shards)} shards)"
    )
    merged["sharded_grade"] = {
        "schema": _SCHEMA,
        "processes": processes if processes is not None else len(shards),
        "workers_per_process": WORKERS_PER_PROCESS,
        "n_shards": len(shards),
        "n_shards_merged": len(good),
        "shard_failures": notes,
        "wall_s": wall_s,
        "unmerged_divergent_keys": divergent,
    }
    return merged


# ------------------------------------------------------------------------------------------- the parent


def _write_json(path: Path, document) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _spawn(spec_path: Path, log_path: Path):
    environment = {k: v for k, v in os.environ.items() if k != ENV}  # a shard never shards again
    log = log_path.open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", __name__, str(spec_path)],
            cwd=os.getcwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()


def _terminate(process) -> None:
    for sig, wait in ((signal.SIGTERM, _KILL_WAIT_S), (signal.SIGKILL, _KILL_WAIT_S)):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
        deadline = time.time() + wait
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.2)


def grade_sharded(
    *,
    caps,
    root,
    tag,
    output: Path,
    frozen: Path,
    run_dir: Path,
    context,
    inputs,
    contract,
    descriptor_sha256,
    end_epoch: float,
    timeout_s: int,
    processes: int,
    floor_exempt: bool,
    verify=None,
):
    """Grade ``caps`` in ``processes`` shard interpreters; return the merged ordinary score or None.

    None means the roster cannot be sharded safely (the caller runs the in-process grader).
    ``verify`` (the controller's input custody check) runs again after the shards, before the merge.
    """
    shards = partition(caps, processes)
    if shards is None:
        return None
    started = time.time()
    base = output / f"{tag}-shards"
    base.mkdir(parents=True, exist_ok=False)
    common = {
        "schema": _SCHEMA,
        "run_dir": str(run_dir),
        "frozen": str(frozen),
        "descriptor": str(context.descriptor),
        "repo": str(context.repo),
        "harness": None if context.harness is None else str(context.harness),
        "readback_policy": context.readback_policy,
        "candidate_policy": None if context.candidate_policy is None else str(context.candidate_policy),
        "arm": inputs.arm,
        "public_root": str(inputs.public_root() if callable(inputs.public_root) else inputs.public_root),
        "contract": str(contract),
        "additional_forbidden": list(inputs.additional_forbidden),
        "end_epoch": end_epoch,
        "timeout_s": timeout_s,
        "floor_exempt": floor_exempt,
        "workers": WORKERS_PER_PROCESS,
    }
    live = []
    for index, assigned in enumerate(shards):
        directory = base / f"s{index:02d}"
        directory.mkdir()
        spec = {
            **common,
            "index": index,
            "directory": str(directory),
            "capsules": [str(c["name"]) for c in assigned],
            "capsule_dirs": [str(c["__dir__"]) for c in assigned],
        }
        _write_json(directory / "spec.json", spec)
        live.append((index, assigned, directory, _spawn(directory / "spec.json", directory / "child.log")))
    deadline = end_epoch + GRACE_S
    killed = set()
    try:
        while any(p.poll() is None for _, _, _, p in live):
            if time.time() >= deadline:
                for index, _, _, p in live:
                    if p.poll() is None:
                        killed.add(index)
                        _terminate(p)
                break
            time.sleep(_POLL_S)
    finally:
        for _, _, _, p in live:  # never leave a shard behind if this thread is interrupted
            if p.poll() is None:
                _terminate(p)
    results = []
    for index, assigned, directory, process in live:
        score, failure = None, None
        path = directory / "score.json"
        if index in killed:
            failure = "shard killed at the final coverage deadline"
        elif process.returncode != 0:
            failure = f"shard exited rc={process.returncode}"
        else:
            try:
                score = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                failure = f"shard score unreadable: {type(exc).__name__}"
        results.append((assigned, score, failure))
    if verify is not None:
        verify()
    admission = _admission(caps, root, descriptor_sha256)
    return merge_scores(
        results, caps=caps, admission=admission, processes=processes, wall_s=round(time.time() - started, 1)
    )


# -------------------------------------------------------------------------------------------- the child


@contextlib.contextmanager
def original_selection(run_dir: Path):
    """Reopen the controller's own original owners (compiler library, shared-pass requirement) from the
    run's frozen copies; the same route ``compiler_library.selected_child`` uses."""
    import yaml

    from merlin.common.digest import sha256_file
    from merlin.targetgen.compiler_library import scoped_compiler_library
    from merlin.targetgen.shared_pass_context import scoped_shared_pass_requirement
    from merlin_experiments.phase1 import compiler_library as CL
    from merlin_experiments.phase2 import compiler_library_state as state

    lib = run_dir / "compiler_library"
    if not (lib / "contract.json").is_file():
        yield
        return
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    record = environment[CL.KEY]
    origin = record["origin"]
    host = state.load_selection(
        root=origin["root"],
        contract_path=Path(origin["root"]).parent / "contract.json",
        contract_sha256=origin["contract_sha256"],
    )
    selection = state.verify_binding(record, host)
    requirement_path = run_dir / "shared_pass_requirement.json"
    requirement = (
        CL._shared_pass_file(requirement_path, sha256_file(requirement_path)) if requirement_path.is_file() else None
    )
    with scoped_compiler_library(selection), scoped_shared_pass_requirement(requirement):
        token = CL._INITIAL.set((selection, selection.root.parent / "contract.json"))
        try:
            yield
        finally:
            CL._INITIAL.reset(token)


def _child(spec: dict) -> int:
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen.target_experiment import load_target_experiment

    from .context import load_context, readback_kwargs
    from .feedback.compiler_profile_runtime import functional_service
    from .feedback.loop_grading import GradingInputs, compiler_execution
    from .feedback.selfcheck import _adapters

    directory = Path(spec["directory"])
    run_dir, frozen = Path(spec["run_dir"]), Path(spec["frozen"])
    context = load_context(
        spec["descriptor"],
        repo=spec["repo"],
        harness=spec["harness"],
        readback_policy=spec["readback_policy"],
        candidate_policy=spec["candidate_policy"],
    )
    te = load_target_experiment(context.descriptor)
    inputs = GradingInputs(
        context=context,
        arm=spec["arm"],
        public_root=Path(spec["public_root"]),
        policy_roots=(),
        contract=Path(spec["contract"]),
        additional_forbidden=tuple(spec["additional_forbidden"]),
        compiler_profile_service=functional_service(run_dir),
    )
    adapters, engine = _adapters("spike", context.target, te.sim_via, **readback_kwargs(context))
    if engine != "spike" or set(adapters) != {"L2"}:
        raise ValueError("final cheap coverage requires the exact Spike-only adapter")
    end = float(spec["end_epoch"])
    runs = directory / "runs"
    with ExitStack() as stack:
        stack.enter_context(original_selection(run_dir))
        stack.enter_context(compiler_execution(inputs, candidate=frozen, run_dir=run_dir))
        if spec["floor_exempt"]:
            from merlin.targetgen.tier_policy import floor_exempt

            stack.enter_context(floor_exempt(spec["capsule_dirs"]))
        log = stack.enter_context((directory / "grade.log").open("w"))
        stack.enter_context(redirect_stdout(log))
        stack.enter_context(redirect_stderr(log))
        score = CG.grade(
            frozen,
            capsules_root=spec["capsule_dirs"],
            runs_root=runs,
            target=context.target,
            labels={"public", "dev"},
            contract=spec["contract"],
            target_experiment=te,
            oracle_adapters=adapters,
            functional_public_tier="L2",
            timeout=int(spec["timeout_s"]),
            max_workers=int(spec["workers"]),
            additional_forbidden=inputs.additional_forbidden,
            model_snapshot_root=runs / ".private_model_sources",
            defer_pending=lambda: "final coverage window exhausted" if time.time() >= end else None,
            defer_running=True,
        )
    _write_json(directory / "score.json", score)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m merlin_experiments.phase1.joint_grade_shards SPEC.json", file=sys.stderr)
        return 64
    spec = json.loads(Path(argv[0]).read_text())
    if spec.get("schema") != _SCHEMA:
        return 65
    try:
        return _child(spec)
    except BaseException as exc:  # noqa: BLE001 -- the parent reads rc/log, never a half-written score
        import traceback

        (Path(spec["directory"]) / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
