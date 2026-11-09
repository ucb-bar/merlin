"""One whole-model job, run to a result document in a detached process.

``python -m merlin_experiments.phase2.whole_model_measured work <job_dir>`` runs :func:`work` (via
:func:`worker_main`).  The refusals happen in this order, each written into the job's result with
its stage and reason -- nothing is swallowed, because a measurer whose failures are silent is
indistinguishable from an idle one:

snapshot digest -> capsule screen -> build (through the declared BUILDER) -> coverage gate ->
instruction rule over the whole ELF -> functional-model grade (paired machines) -> device identity
and ABI admission -> run -> verdict.

The BUILDER is a parameter (``module:callable`` or a pinned ``/abs/path.py:callable``), with the
signature ``build(package_dir, *, target, out_dir, **build_options) -> Mapping``; the record it
returns is normalized by :func:`.identity.normalize_build_record`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import gates as G
from . import jobs as J
from . import registry as R
from . import retention as RET
from .identity import (
    IdentityError,
    load_builder,
    normalize_build_record,
    package_digest,
    read_json,
    sha256_file,
    write_json_atomic,
)
from .machines import MachineRefusal, machine_from_spec

#: TWO MACHINES, ONE CANDIDATE: a TIMING build on the board and a LOCALLY graded build of the same
#: program on a functional model on the host, run in parallel.
PAIRED_MACHINE = "paired"
#: A paired machine whose board half runs in BATCHES (see :mod:`.batch`).
BATCHED_MACHINE = "batched"
#: A machine run through a MEASURER CONTRACT (``measure(*, elf, elf_sha256, machine, target,
#: build_record, out)``) rather than one of this package's own machines.
CONTRACT_MACHINE = "contract"
#: A cell: the form-perf capsules' own group programs instead of the whole model (see :mod:`.cells`).
CELL_MACHINE = "cell"


def whole_model_driver(target: str) -> Any:
    """The target's whole-model program driver (the C program a whole model runs as on the target:
    its verification fence, its batch linker and console splitter).  Declared by the target's backend;
    absent means batched and paired measurement are unavailable for it, said so."""
    try:
        from merlin.runtime.backends import base as backends
    except ImportError as exc:  # pragma: no cover - core always ships this module
        raise J.ServiceError(f"no runtime backend registry: {exc}") from exc
    declare = getattr(backends, "whole_model_driver", None)
    if declare is None:
        raise J.ServiceError("this checkout's runtime backends declare no whole-model driver interface")
    try:
        return declare(target)
    except NotImplementedError as exc:
        raise J.ServiceError(str(exc)) from exc


def _record_of(raw: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str((raw.get("notes") or {}).get("build_record") or ""))
    if not path.name or not path.is_file():
        raise J.ServiceError("the builder's record names no build_record file")
    return json.loads(path.read_text(encoding="utf-8"))


def window_identity(raw: Mapping[str, Any], program_without_verification: Callable[[str], str]) -> dict[str, Any]:
    """What must be identical between a timing build and a locally graded build: the program outside
    its verification fence, the kernel objects linked in, and the compiler and flags."""
    record = _record_of(raw)
    source = Path(str(raw["elf"])).parent / str(record.get("program_source") or "group_model_program.c")
    window = program_without_verification(source.read_text(encoding="utf-8"))
    program = record.get("program") or {}
    return {
        "window_source_sha256": hashlib.sha256(window.encode("utf-8")).hexdigest(),
        "linked_objects": sorted(str(o.get("sha256")) for o in record.get("linked_objects") or ()),
        "compiler": program.get("compiler"),
        "flags": program.get("flags"),
        "abi_header_sha256": raw.get("parameter_header_sha256"),
    }


def same_program(job: Mapping[str, Any], raws: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Both builds' window identities, refused unless they are the same program outside verification."""
    strip = whole_model_driver(str(job["target"])).program.program_without_verification
    identities = {role: window_identity(raws[role], strip) for role in raws}
    if identities["timing"] != identities["local"]:
        differs = sorted(k for k in identities["timing"] if identities["timing"][k] != identities["local"][k])
        raise J.ServiceError(
            f"the timing and the locally graded builds differ outside their verification code ({differs}); "
            "they would not be the same program, so neither's verdict could speak for the other's cycles"
        )
    return identities


def load_reference(job: Mapping[str, Any], device_sha256: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The same-machine reference result a job was requested against, or why there is none."""
    spec = job.get("reference")
    if not isinstance(spec, Mapping) or not spec.get("path"):
        return None, {"used": False, "reason": "no reference declared"}
    path = Path(str(spec["path"]))
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        return None, {"used": False, "reason": f"the reference result at {path} is absent or changed since the request"}
    document = read_json(path) or {}
    device = (document.get("device") or {}).get("binary_sha256")
    if device != device_sha256:
        return None, {
            "used": False,
            "reason": f"the reference ran on device {device!r}, this job on {device_sha256!r}; never compared",
        }
    return document, {
        "used": True,
        "path": str(path),
        "sha256": spec.get("sha256"),
        "package_sha256": document.get("package_sha256"),
        "label": document.get("label"),
        "timing_status": document.get("timing_status"),
    }


def _invalidate_on_failed_check(verdict: Mapping[str, Any], check: Mapping[str, Any] | None) -> dict[str, Any]:
    """A REQUIRED check that failed makes the run invalid whatever the whole model printed."""
    verdict = dict(verdict)
    if check is not None and check.get("required") and not check.get("passed"):
        if verdict.get("timing_status") == V.TIMING_MEASURED:
            verdict.update(
                timing_status=V.TIMING_MEASURED_INVALID,
                objective_cycles=None,
                invalid_reason=f"the required pre-measure check failed: {check.get('summary')}",
            )
    return verdict


def _apply_exactness(job: Mapping[str, Any], verdict: Mapping[str, Any], build: Mapping[str, Any]) -> dict[str, Any]:
    """``verdict`` with every group held to the run's exactness contract (carried by value on the job;
    the default -- every form exact -- when the run declared none), the contract recorded on it."""
    from merlin.perf import exactness as EX

    spec = job.get("exactness") or {}
    contract = EX.Contract.from_value(spec.get("contract"), target=str(job.get("target") or ""))
    groups = ((build.get("expectations") or {}).get("groups")) or {}
    resolve = EX.resolver(
        contract,
        forms=spec.get("forms"),
        routes=list(build.get("groups") or ()),
        op_bounds={str(g): (body or {}).get("bound_lsb") for g, body in groups.items()},
    )
    return EX.apply_to_verdict(verdict, resolve, contract=contract)


def _apply_limits(job, verdict, expectations, limits, reference):
    if not limits:
        return verdict
    basis = verdict if job.get("role") == J.ROLE_REFERENCE else (reference or {}).get("verdict")
    return V.apply_machine_limits(verdict, expectations, limits, reference=basis)


# --------------------------------------------------------------- one machine
def _single_machine(job, job_dir, package, observed, builder, check) -> dict[str, Any]:
    from merlin.common import provenance

    started = time.time()
    raw = builder(package, target=job["target"], out_dir=job_dir / "build", **dict(job.get("build_options") or {}))
    build = normalize_build_record(raw, package_sha256=observed)
    build["wall_seconds"] = round(time.time() - started, 3)
    build["coverage_gate"] = G.coverage_assessment(job, build)
    refused = G.coverage_gate(job, build, check) or G.isa_gate(job, build, check, "timing")
    build["pruned"] = RET.prune_intermediates(job_dir / "build", raw.get("prunable"), keep=Path(build["elf"]))
    write_json_atomic(job_dir / "build_record.json", build)
    if refused is not None:
        return refused
    if (job.get("machine") or {}).get("kind") == CONTRACT_MACHINE:
        return _contract_measurement(job, job_dir, build, raw)
    machine = machine_from_spec(job["machine"])
    identity = machine.identity()
    machine.admit(identity, program_header_sha256=build["parameter_header_sha256"])
    run = machine.run(Path(build["elf"]), job_dir / "run", timeout_s=float(job["timeout_seconds"]))
    if sha256_file(Path(build["elf"])) != build["elf_sha256"]:
        raise J.ServiceError("the ELF changed while it was being run")
    expectations = V.Expectations.from_record(build["expectations"])
    reference, reference_note = load_reference(job, identity.binary_sha256)
    if not run["completed"]:
        verdict = V.refused(f"the run did not complete: {run['incomplete_reason']}")
    else:
        text = Path(run["uart_log"]).read_text(encoding="utf-8", errors="replace")
        verdict = V.judge(
            text, expectations, templates=build.get("protocol"), reference=(reference or {}).get("verdict")
        )
        if job.get("excuse_reference_failures"):
            # The reference arm's own run is excused against ITSELF: it is the bar, and its failures on
            # this machine are exactly the set every candidate is then allowed to share.
            basis = verdict if job.get("role") == J.ROLE_REFERENCE else (reference or {}).get("verdict")
            if basis is not None:
                verdict = V.apply_vendor_also_fails(verdict, basis)
            else:
                verdict["excusal"] = "not applied: no same-machine reference run was available"
        verdict = _apply_limits(job, verdict, expectations, (job.get("machine") or {}).get("cannot_express"), reference)
        verdict = _apply_exactness(job, verdict, build)
    verdict = _invalidate_on_failed_check(verdict, check)
    return J.result(
        job,
        reference=reference_note,
        pre_measure_check=check,
        timing_status=verdict["timing_status"],
        objective_cycles=V.objective_cycles(verdict),
        verdict=verdict,
        build={key: build[key] for key in build if key != "expectations"},
        device=identity.to_dict(),
        run={key: value for key, value in run.items() if key != "device"},
        cycle_adjudication=R.adjudicates(job.get("machine") or {}),
        provenance=provenance.record(
            artifacts={"elf": build["elf"], "emulator": identity.binary},
            extra={
                "device_artifact": identity.artifact,
                "device_abi_header_sha256": identity.abi_header_sha256,
                "program_header_sha256": build["parameter_header_sha256"],
                "package_sha256": observed,
            },
        ),
    )


def _contract_measurement(job, job_dir, build, raw) -> dict[str, Any]:
    """Run a contract measurer and state its result in this service's result shape: MEASURED only when
    the measurer calls the run quotable-correct, MEASURED_INVALID when it graded the run and found it
    wrong, REFUSED when it could not grade it."""
    spec = dict(job["machine"])
    measurer = load_builder(str(spec["measurer"]))
    build_record = _record_of(raw)
    out = job_dir / "run"
    out.mkdir(parents=True, exist_ok=True)
    outcome = dict(
        measurer(
            elf=Path(build["elf"]),
            elf_sha256=build["elf_sha256"],
            machine=str(spec.get("registry_machine") or ""),
            target=str(job["target"]),
            build_record=build_record,
            out=out,
        )
    )
    if outcome.get("elf_sha256") != build["elf_sha256"]:
        raise J.ServiceError("the contract measurer reports running a different ELF")
    correctness = outcome.get("correctness") or {}
    grade = correctness.get("grade") or {}
    argmax = correctness.get("argmax") or {}
    graded = correctness.get("status") == "graded"
    quotable = bool(correctness.get("quotable")) and graded
    status = V.TIMING_MEASURED if quotable else V.TIMING_MEASURED_INVALID if graded else V.TIMING_REFUSED
    agree = {str(g) for g in grade.get("agree") or ()}
    failed = sorted(
        {str(row.get("group") if isinstance(row, Mapping) else row) for row in grade.get("disagree") or ()},
        key=V._order,
    )
    # The grade's own numbers for each group (an exactness contract grades from them): the grade's evidence
    # table when it states one, else what a disagreement row says.  An agreeing group without numbers has
    # none -- "agrees" under its op's own bound is not "equal".
    compares = {
        str(g): str((body or {}).get("compare") or V.COMPARE_EXACT)
        for g, body in ((build.get("expectations") or {}).get("groups") or {}).items()
    }
    evidence = {str(g): dict(v) for g, v in (grade.get("evidence") or {}).items() if isinstance(v, Mapping)}
    for row in grade.get("disagree") or ():
        if isinstance(row, Mapping) and row.get("group") is not None and str(row["group"]) not in evidence:
            evidence[str(row["group"])] = {
                k: row[v]
                for k, v in (("max_abs", "max_abs"), ("mismatches", "mismatches"), ("elements", "of"))
                if row.get(v) is not None
            }
    cycles = outcome.get("cycles")
    verdict: dict[str, Any] = {
        "schema": V.SCHEMA,
        "timing_status": status,
        "whole_window_cycles": cycles if graded else None,
        "objective_cycles": cycles if quotable else None,
        "correctness": {
            "status": "pass" if quotable else "fail",
            "groups_failed": failed,
            "groups_absent": list(grade.get("absent") or ()),
            "argmax": {
                "observed": argmax.get("from_dump"),
                "oracle": argmax.get("oracle"),
                "agrees_with_oracle": bool(argmax.get("agrees_with_oracle")),
            },
            "evidence": "the contract measurer's own per-group local grade over a memory dump, and the argmax",
        },
        "groups": [
            {
                "group": g,
                "cycles": c,
                "correct": g in agree,
                "state": "correct" if g in agree else "failed",
                "compare": compares.get(str(g), V.COMPARE_EXACT),
                **evidence.get(str(g), {}),
            }
            for g, c in sorted((outcome.get("per_group_cycles") or {}).items(), key=lambda kv: V._order(kv[0]))
        ],
    }
    if not graded:
        verdict["refusal"] = correctness.get("refusal") or "the contract measurer did not grade the run"
    else:
        verdict = _apply_exactness(job, verdict, build)
        status = verdict["timing_status"]
    return J.result(
        job,
        timing_status=status,
        objective_cycles=V.objective_cycles(verdict),
        verdict=verdict,
        build={key: build[key] for key in build if key != "expectations"},
        device={
            "machine": spec.get("measurer"),
            "artifact": outcome.get("machine"),
            "binary_sha256": outcome.get("machine"),
            "rung": outcome.get("engine"),
        },
        run={"contract_result": outcome},
        cycle_adjudication=outcome.get("cycles_adjudication") or R.adjudicates(spec),
    )


# --------------------------------------------------------------- two machines
def _build_role(job, job_dir, package, observed, builder, role: str, verify: str):
    started = time.time()
    raw = builder(
        package,
        target=job["target"],
        out_dir=job_dir / f"build_{role}",
        **{**dict(job.get("build_options") or {}), "verify": verify},
    )
    build = normalize_build_record(raw, package_sha256=observed)
    build["wall_seconds"] = round(time.time() - started, 3)
    return build, raw


def _admitted(spec: Mapping[str, Any], builds: Mapping[str, Mapping[str, Any]], roles) -> tuple[dict, dict]:
    machines, device = {}, {}
    for role in roles:
        machines[role] = machine_from_spec(spec[role])
        identity = machines[role].identity()
        machines[role].admit(identity, program_header_sha256=builds[role]["parameter_header_sha256"])
        device[role] = identity.to_dict()
    return machines, device


def publish_local_verdict(job, job_dir, package, observed, build, run, device) -> dict[str, Any]:
    """Grade the functional model's run on its own and publish it the moment that run ends."""
    expectations = V.Expectations.from_record(build["expectations"])
    if not run.get("completed"):
        verdict = V.refused(f"the functional-model run did not complete: {run.get('incomplete_reason')}")
    else:
        verdict = V.judge(
            Path(run["uart_log"]).read_text(encoding="utf-8", errors="replace"),
            expectations,
            templates=build.get("protocol"),
        )
    correctness = verdict.get("correctness") or {}
    status = (
        "refused"
        if verdict.get("timing_status") == V.TIMING_REFUSED
        else ("correct" if correctness.get("status") == "pass" else "incorrect")
    )
    document: dict[str, Any] = {
        "schema": J.LOCAL_VERDICT_SCHEMA,
        "package_sha256": observed,
        "published_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "status": status,
        "machine": dict(device),
        "correctness": correctness,
        "refusal": verdict.get("refusal"),
        "groups": [dict(row) for row in V.group_table(verdict)],
        "routes": list(build.get("groups") or ()),
        "whole_window_cycles_functional_model": verdict.get("whole_window_cycles"),
        "note": "the functional model's own grade of these bytes: every group recomputed on its own inputs "
        "and the classification. It says nothing about timing, and nothing about a hardware-ordering race "
        "(a functional model does not reorder memory); the board half checks that by byte equality",
    }
    write_json_atomic(Path(job_dir) / J.LOCAL_VERDICT, document)
    return document


def paired_finish(
    job, builds, identities, runs, device, check, observed, *, batch=None, job_dir=None
) -> dict[str, Any]:
    """The verdict of one candidate from its two runs' consoles."""
    from merlin.common import provenance

    spec = dict(job["machine"])
    expectations = V.Expectations.from_record(builds["local"]["expectations"])
    reference, reference_note = load_reference(job, str(device["timing"].get("binary_sha256")))
    incomplete = [f"{role}: {runs[role].get('incomplete_reason')}" for role in runs if not runs[role].get("completed")]
    if incomplete:
        verdict = V.refused("a run did not complete -- " + "; ".join(incomplete))
    else:
        verdict = V.judge_pair(
            Path(runs["timing"]["uart_log"]).read_text(encoding="utf-8", errors="replace"),
            Path(runs["local"]["uart_log"]).read_text(encoding="utf-8", errors="replace"),
            expectations,
            templates=builds["local"].get("protocol"),
        )
        verdict = _apply_limits(job, verdict, expectations, (spec.get("timing") or {}).get("cannot_express"), reference)
        verdict = _apply_exactness(job, verdict, builds["local"])
    verdict = _invalidate_on_failed_check(verdict, check)
    return J.result(
        job,
        reference=reference_note,
        pre_measure_check=check,
        timing_status=verdict["timing_status"],
        objective_cycles=V.objective_cycles(verdict),
        verdict=verdict,
        build=builds["timing"],
        local_build=builds["local"],
        window_identity=identities["timing"],
        device=dict(device["timing"]),
        local_device=dict(device["local"]),
        run={key: value for key, value in (runs.get("timing") or {}).items() if key != "device"},
        local_run={key: value for key, value in (runs.get("local") or {}).items() if key != "device"},
        batch=dict(batch) if batch else None,
        cycle_adjudication=R.adjudicates(spec),
        provenance=provenance.record(
            artifacts={"timing_elf": builds["timing"]["elf"], "local_elf": builds["local"]["elf"]},
            extra={"device_artifact": device["timing"].get("artifact"), "package_sha256": observed},
        ),
    )


def same_board_program(root: Path, job: Mapping[str, Any], elf_sha256: str) -> dict[str, Any] | None:
    """A result for ``job`` from an earlier measurement of the SAME board ELF, or None: different
    package bytes can compile to a byte-identical program, and its board time would measure the same
    thing again and, at the same cycles, make a tie look like a new best."""
    from .identity import key_of

    if job.get("role") == J.ROLE_REFERENCE or job.get("solo") or int(job.get("replicate") or 0):
        return None
    for other in sorted(Path(root).glob("*/result.json")):
        if other.parent.name == key_of(job):
            continue
        earlier = read_json(other) or {}
        if (earlier.get("build") or {}).get("elf_sha256") != elf_sha256 or earlier.get("same_program_as"):
            continue
        if earlier.get("timing_status") not in (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID):
            continue
        carried = {
            k: json.loads(json.dumps(earlier.get(k), default=str))
            for k in ("timing_status", "objective_cycles", "verdict", "build", "local_build", "device", "provenance")
        }
        return J.result(job, **carried, same_program_as=other.parent.name, measured_by=other.parent.name)
    return None


def _paired_measurement(job, job_dir, package, observed, builder, check) -> dict[str, Any]:
    """The LOCAL half first, published the moment its run ends, while the board half builds or waits."""
    spec = dict(job["machine"])
    if (spec.get("timing") or {}).get("kind") not in R.RUN_KINDS:
        raise J.ServiceError(
            f"the paired machine's timing half is {(spec.get('timing') or {}).get('kind')!r}, not runnable"
        )
    batched = spec.get("kind") == BATCHED_MACHINE
    timeout = float(job["timeout_seconds"])
    builds: dict[str, dict[str, Any]] = {}
    raws: dict[str, Mapping[str, Any]] = {}
    builds["local"], raws["local"] = _build_role(job, job_dir, package, observed, builder, "local", "local")
    builds["local"]["coverage_gate"] = G.coverage_assessment(job, builds["local"])
    refused = G.coverage_gate(job, builds["local"], check) or G.isa_gate(job, builds["local"], check, "local")
    if refused is not None:
        return refused
    local_machines, local_device = _admitted(spec, builds, ("local",))
    local: dict[str, Any] = {}

    def run_local() -> None:
        try:
            local["run"] = local_machines["local"].run(
                Path(builds["local"]["elf"]), job_dir / "run_local", timeout_s=timeout
            )
        except Exception as exc:  # noqa: BLE001 -- recorded as that run's incompleteness
            local["run"] = {"completed": False, "incomplete_reason": f"{type(exc).__name__}: {exc}"}
        try:
            publish_local_verdict(job, job_dir, package, observed, builds["local"], local["run"], local_device["local"])
        except Exception as exc:  # noqa: BLE001 -- the final verdict re-grades the same console
            local["publish_error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run_local, name="functional-model-run")
    worker.start()
    try:
        builds["timing"], raws["timing"] = _build_role(
            job, job_dir, package, observed, builder, "timing", spec.get("timing_verify") or "words"
        )
        identities = same_program(job, raws)
    except BaseException:
        worker.join()
        raise
    refused = G.isa_gate(job, builds["timing"], check, "timing")
    if refused is not None:
        worker.join()
        return refused  # the board's own program is checked too: it is the one that would run
    for role in builds:
        if role == "timing" and batched:
            continue  # re-linked by the batch runner: its program and kernel objects are kept until then
        builds[role]["pruned"] = RET.prune_intermediates(
            job_dir / f"build_{role}", raws[role].get("prunable"), keep=Path(builds[role]["elf"])
        )
    write_json_atomic(job_dir / "build_record.json", {"timing": builds["timing"], "local": builds["local"]})
    same = same_board_program(job_dir.parent, job, builds["timing"]["elf_sha256"])
    if same is not None:
        worker.join()
        return same
    if batched:
        worker.join()
        if sha256_file(Path(builds["local"]["elf"])) != builds["local"]["elf_sha256"]:
            raise J.ServiceError("the local ELF changed while it was being run")
        gated = G.functional_gate(job, job_dir, builds, local, local_device["local"], check)
        if gated is not None:
            return gated
        record = _record_of(raws["timing"])
        program_dir = Path(builds["timing"]["elf"]).parent
        write_json_atomic(
            job_dir / J.BOARD_REQUEST,
            {
                "builds": builds,
                "identities": identities,
                "local_run": local["run"],
                "local_device": local_device["local"],
                "check": check,
                "package_sha256": observed,
                "variant": {
                    "label": observed,
                    "program_object": str(program_dir / str(record.get("program_object") or "group_model_program.o")),
                    "objects": [str(o.get("path")) for o in record.get("linked_objects") or ()],
                    "supports": [
                        str(program_dir / name) for name in record.get("support_objects") or ("syscalls.o", "crt.o")
                    ],
                    "program": record.get("program") or {},
                },
            },
        )
        return {J.AWAITING_BOARD: True}
    timing_machines, timing_device = _admitted(spec, builds, ("timing",))
    try:
        timing_run = timing_machines["timing"].run(
            Path(builds["timing"]["elf"]), job_dir / "run_timing", timeout_s=timeout
        )
    except Exception as exc:  # noqa: BLE001 -- recorded as that run's incompleteness
        timing_run = {"completed": False, "incomplete_reason": f"{type(exc).__name__}: {exc}"}
    worker.join()
    # Both halves ran in parallel, so the board time is already spent: the pair's verdict (not a
    # functional-gate refusal) records the wrong program, with its cycles, as MEASURED_INVALID.
    for role in ("timing", "local"):
        if sha256_file(Path(builds[role]["elf"])) != builds[role]["elf_sha256"]:
            raise J.ServiceError(f"the {role} ELF changed while it was being run")
    device = {"timing": timing_device["timing"], "local": local_device["local"]}
    return paired_finish(
        job, builds, identities, {"timing": timing_run, "local": local["run"]}, device, check, observed, job_dir=job_dir
    )


# --------------------------------------------------------------- the entrypoint
def work(job_dir: Path) -> dict[str, Any]:
    """Run ONE job to a result document.  Every failure becomes a refusal naming its stage."""
    job_dir = Path(job_dir)
    job = read_json(job_dir / "job.json")
    if job is None:
        raise J.ServiceError(f"{job_dir} has no readable job.json")
    stage = "snapshot"
    check = None
    try:
        package = job_dir / "package"
        observed = package_digest(package)
        if observed != job["package_sha256"]:
            raise J.ServiceError(
                f"the snapshot hashes to {observed}, the job was requested for {job['package_sha256']}"
            )
        stage = "pre_measure_check"
        check = G.pre_measure_check(job, package, job_dir)
        kind = (job.get("machine") or {}).get("kind")
        if kind == CELL_MACHINE:
            stage = "cell"
            from . import cells

            return J.result(
                job, pre_measure_check=check, **cells.measure_cell(job, job_dir, package, target=str(job["target"]))
            )
        stage = "build"
        builder = load_builder(job["builder"]["spec"], expected_sha256=job["builder"].get("sha256"))
        if kind in (PAIRED_MACHINE, BATCHED_MACHINE):
            return _paired_measurement(job, job_dir, package, observed, builder, check)
        stage = "run"
        return _single_machine(job, job_dir, package, observed, builder, check)
    except (J.ServiceError, IdentityError, MachineRefusal, V.VerdictRefusal, R.RegistryError) as exc:
        return J.refused(job, f"{stage}: {exc}", pre_measure_check=check)
    except Exception as exc:  # noqa: BLE001 -- a crash is a refusal WITH its traceback, never silence
        return J.refused(
            job,
            f"{stage}: {type(exc).__name__}: {str(exc)[:4000]}",
            pre_measure_check=check,
            traceback=traceback.format_exc()[-8000:],
        )


def worker_main(job_dir: Path) -> int:
    from .identity import locked, now

    job_dir = Path(job_dir)
    job_path = job_dir / "job.json"
    with locked(job_dir):
        job = read_json(job_path) or {}
        job.update(state=J.RUNNING, worker_pid=os.getpid(), started_at=now())
        write_json_atomic(job_path, job)
    outcome = work(job_dir)
    if outcome.get(J.AWAITING_BOARD):
        with locked(job_dir):
            job = read_json(job_path) or {}
            job.update(state=J.BOARD, board_ready_at=now(), board_ready_epoch=time.time())
            write_json_atomic(job_path, job)
        return 0
    try:
        J.write_result(job_dir, outcome)
    except J.ResultExists:
        # Something else ended this job while it ran (a supersede, a second worker): its result stands, and
        # this one is kept beside it as an attempt of its own -- never written over it.
        J.preserve_result(job_dir, outcome, why=f"worker {os.getpid()} finished after the job already had a result")
        return 0
    with locked(job_dir):
        job = read_json(job_path) or {}
        job.update(state=J.DONE, finished_at=now(), timing_status=outcome.get("timing_status"))
        write_json_atomic(job_path, job)
    RET.finalize_if_allowed(job_dir.parent, job_dir)
    RET.finalize_superseded(job_dir.parent)
    return 0


__all__ = [
    "BATCHED_MACHINE",
    "CELL_MACHINE",
    "CONTRACT_MACHINE",
    "PAIRED_MACHINE",
    "load_reference",
    "paired_finish",
    "publish_local_verdict",
    "same_board_program",
    "same_program",
    "whole_model_driver",
    "window_identity",
    "work",
    "worker_main",
]
