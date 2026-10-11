"""Plan an AWS CPU timing study; profiling requires explicit live private owners.

The command only writes a plan. It never provisions infrastructure, imports a
public callback or launches a simulator. The private profiler uses the original
compile/static observer, selected backend and existing supervised worker.
Process costs and output observations are diagnostics, not hardware timing.
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.digest import is_sha256
from merlin.common.strict_json import loads
from merlin.perf.component_cost import COMPLETE_STAGES
from merlin.perf.component_screen import ComponentScreenPolicy
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS

from .contracts import StageGateError, canonical_json, document_sha256, mapping_file, sha256_file, write_json
from .evaluation_fidelity import (
    CPU_ENGINES,
    SCOPES,
    CpuEngineSelection,
    CpuTimingConfiguration,
    ProgramAccessRegistry,
)

SCHEMA = "merlin.aws_cpu_timing_study.v1"
_STAGES = (
    "engine_setup",
    "compile",
    "feature_extraction",
    "execution",
    "calibration",
    "held_validation",
    "complete_cost",
)


def _fields(row, names, label):
    if type(row) is not dict or set(row) != set(names):
        raise StageGateError(label + " contains missing or unsupported fields")


def _positive(value, ceiling, label):
    if type(value) is not int or not 0 < value <= ceiling:
        raise StageGateError(label + " needs an explicit bounded positive integer")


def _configuration(row):
    _fields(row, ("memory", "cache", "clock"), "CPU timing configuration")
    pairs = []
    for name in ("memory", "cache", "clock"):
        _fields(row[name], ("path", "sha256"), name + " source")
        pairs.append((row[name]["path"], row[name]["sha256"]))
    selected = CpuTimingConfiguration(*pairs)
    selected.record()
    return selected


def plan_timing_study(request):
    """Validate data-only inputs and retain the entire requested measurement matrix."""
    _fields(
        request,
        (
            "schema",
            "target",
            "target_descriptor",
            "registry",
            "programs",
            "engines",
            "configuration",
            "final_configuration",
            "resources",
            "environments",
            "replicates",
            "max_stage_seconds",
        ),
        "timing study",
    )
    if request["schema"] != SCHEMA or type(request["target"]) is not str or not request["target"]:
        raise StageGateError("timing study requires its explicit schema and target")
    _fields(request["target_descriptor"], ("path", "sha256"), "target descriptor")
    from .evaluation_fidelity import _pin

    _pin(request["target_descriptor"]["path"], request["target_descriptor"]["sha256"])
    _fields(request["registry"], ("path", "sha256"), "program registry")
    registry = ProgramAccessRegistry(Path(request["registry"]["path"]), request["registry"]["sha256"])
    registry.verify()
    programs = request["programs"]
    if type(programs) is not list or not 1 <= len(programs) <= 64:
        raise StageGateError("timing study requires a bounded nonempty exact public program roster")
    identities = []
    for row in programs:
        _fields(row, ("family", "capsule", "source_sha256", "scope"), "program")
        if (
            any(type(row[name]) is not str or not row[name] for name in ("family", "capsule"))
            or row["scope"] not in SCOPES
        ):
            raise StageGateError("timing study program needs explicit identity and complete scope")
        registry.require_public(row["source_sha256"])
        identities.append((row["family"], row["capsule"], row["source_sha256"]))
    if len(set(identities)) != len(identities) or len({(a, b) for a, b, _sha in identities}) != len(identities):
        raise StageGateError("timing study repeats an original program identity")
    engines = request["engines"]
    if type(engines) is not list or not 1 <= len(engines) <= len(CPU_ENGINES):
        raise StageGateError("timing study needs explicit selected CPU engines")
    for row in engines:
        _fields(row, ("engine", "identity_sha256"), "engine")
        if row["engine"] not in CPU_ENGINES or not is_sha256(row["identity_sha256"]):
            raise StageGateError("timing study engine lacks an exact ordinary selection identity")
    if len({row["engine"] for row in engines}) != len(engines):
        raise StageGateError("timing study repeats a selected engine")
    current, final = _configuration(request["configuration"]), _configuration(request["final_configuration"])
    assigned = request["resources"]
    _fields(
        assigned,
        ("provider", "instance_id", "cpu_ids", "memory_bytes", "disk_bytes", "work_root"),
        "assigned resources",
    )
    if (
        assigned["provider"] != "aws"
        or type(assigned["instance_id"]) is not str
        or not assigned["instance_id"].startswith("i-")
    ):
        raise StageGateError("timing study must name an explicitly allocated AWS instance")
    cpu_ids = assigned["cpu_ids"]
    if (
        type(cpu_ids) is not list
        or not 1 <= len(cpu_ids) <= 1024
        or any(type(cpu) is not int or cpu < 0 for cpu in cpu_ids)
        or len(set(cpu_ids)) != len(cpu_ids)
    ):
        raise StageGateError("timing study needs explicit distinct assigned CPU identities")
    _positive(assigned["memory_bytes"], 1 << 60, "assigned RAM")
    _positive(assigned["disk_bytes"], 1 << 60, "assigned disk")
    work = Path(assigned["work_root"])
    if not work.is_absolute() or work.resolve() != work or any(p.is_symlink() for p in (work, *work.parents)):
        raise StageGateError("timing study needs a canonical explicit allocated work owner")
    _positive(request["replicates"], 100, "study replicates")
    if request["replicates"] < 2:
        raise StageGateError("timing study requires at least two cold/warm paired replicates")
    _positive(request["max_stage_seconds"], int(ITERATION_MAX_SECONDS), "study stage deadline")
    environments = request["environments"]
    _fields(environments, ("compile", "feature_extraction", *(row["engine"] for row in engines)), "stage environments")
    environment_pins = {}
    for name, environment in environments.items():
        if (
            type(environment) is not dict
            or not environment
            or len(environment) > 128
            or any(
                type(k) is not str or type(v) is not str or len(k) > 256 or len(v) > 8192
                for k, v in environment.items()
            )
        ):
            raise StageGateError("timing study requires an explicit bounded complete native environment")
        environment_pins[name] = I.environment_identity(environment)
    if len(identities) * len(engines) * request["replicates"] * 2 * len(_STAGES) > 100000:
        raise StageGateError("timing study complete matrix exceeds its explicit bounded observation roster")
    matrix = [
        {
            "program": {"family": family, "capsule": capsule, "source_sha256": digest},
            "engine": engine["engine"],
            "replicate": replicate,
            "regime": regime,
            "stage": stage,
        }
        for family, capsule, digest in identities
        for engine in engines
        for replicate in range(request["replicates"])
        for regime in ("cold", "warm")
        for stage in _STAGES
    ]
    return {
        "schema": SCHEMA,
        "status": "planned",
        "request_sha256": document_sha256(request),
        "target": request["target"],
        "execution_substrate": "CPU",
        "execution_performed": False,
        "provider_operations_performed": False,
        "resources": assigned,
        "environment_identities": environment_pins,
        "matrix": matrix,
        "configuration_sha256": current.sha256,
        "final_configuration_sha256": final.sha256,
        "configuration_matched": current.sha256 == final.sha256,
        "metrics": [
            "wall_seconds",
            "cpu_seconds",
            "peak_process_tree_rss_bytes",
            "disk_bytes_before",
            "peak_output_disk_bytes",
            "disk_bytes_after",
        ],
        "complete_cost_stages": list(COMPLETE_STAGES),
        "screening_policy": ComponentScreenPolicy().to_dict(),
        "timing_status": "UNKNOWN",
        "promotion": "NONE",
        "break_even_rule": "N*((1-q)*S-E)>M",
        "break_even_terms": {
            "S": "complete ordinary RTL evaluation per proposal",
            "E": "complete compile/extraction/Spike evaluation per proposal",
            "q": "fraction of proposals retaining ordinary RTL evaluation",
            "M": "engine setup plus calibration and held validation",
        },
        "limitations": [
            "all matrix cells remain required; missing cells cannot disappear from complete-cost totals",
            "requested cold/warm labels require independent cache-state observations",
            "source/output equivalence supplies no cycle or ranking authority",
            "study profiling does not issue runtime14, stage11, held or final qualification",
        ],
    }


def break_even(*, proposals, retained_rtl_fraction, rtl_seconds, extraction_seconds, calibration_seconds):
    """Complete-cost arithmetic only; incomplete/UNKNOWN costs remain unknown."""
    if type(proposals) is not int or proposals < 0:
        raise ValueError("break-even proposal count must be a nonnegative integer")
    values = (retained_rtl_fraction, rtl_seconds, extraction_seconds, calibration_seconds)
    if any(value is None for value in values):
        return {"status": "UNKNOWN", "beneficial": None, "minimum_proposals": None, "net_saved_seconds": None}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("break-even costs and retained fraction must be finite nonnegative values")
    if retained_rtl_fraction > 1:
        raise ValueError("retained RTL fraction must be at most one")
    per_proposal = (1 - retained_rtl_fraction) * rtl_seconds - extraction_seconds
    saved = proposals * per_proposal - calibration_seconds
    minimum = math.floor(calibration_seconds / per_proposal) + 1 if per_proposal > 0 else None
    return {
        "status": "computed",
        "beneficial": saved > 0,
        "minimum_proposals": minimum,
        "net_saved_seconds": saved,
        "saved_per_proposal_seconds": per_proposal,
        "authority": "arithmetic only; costs must include original complete execution denominator",
    }


def _aws_instance_id():
    # Fixed IMDSv2 identity request, called only by the explicit private profiler.
    # No credentials, arbitrary URL or provider operation is accepted.
    from urllib.request import ProxyHandler, Request, build_opener

    opener = build_opener(ProxyHandler({}))
    with opener.open(
        Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        ),
        timeout=2,
    ) as response:
        token = response.read(513)
    if not token or len(token) > 512:
        raise StageGateError("AWS instance identity token is unavailable")
    with opener.open(
        Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token.decode("ascii")},
        ),
        timeout=2,
    ) as response:
        identity = response.read(129)
    if len(identity) > 128:
        raise StageGateError("AWS instance identity exceeds its bound")
    return identity.decode("ascii").strip()


def _allocated(request):
    assigned = request["resources"]
    if _aws_instance_id() != assigned["instance_id"]:
        raise StageGateError("timing study is not on the explicitly allocated AWS instance")
    if not set(assigned["cpu_ids"]) <= os.sched_getaffinity(0):
        raise StageGateError("timing study CPUs are outside the actual allocation")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        fields = value.split()
        if fields and fields[0].isdecimal():
            memory[key] = int(fields[0]) * 1024
    work = Path(assigned["work_root"])
    if not work.is_dir() or memory.get("MemAvailable", 0) < assigned["memory_bytes"]:
        raise StageGateError("timing study lacks its explicit allocated RAM/work owner")
    if shutil.disk_usage(work).free < assigned["disk_bytes"]:
        raise StageGateError("timing study lacks its explicit allocated free disk")


def _tree_rss():
    pending, seen, total = [os.getpid()], set(), 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if len(seen) > 4096:
            raise StageGateError("timing study process roster exceeds its observation bound")
        root = Path("/proc") / str(pid)
        try:
            total += int((root / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            for task in (root / "task").iterdir():
                pending.extend(int(value) for value in (task / "children").read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def _disk_bytes(root):
    total, count = 0, 0
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise StageGateError("timing study output contains a foreign alias")
        count += 1
        if count > 100000:
            raise StageGateError("timing study output member observation budget exceeded")
        try:
            if path.is_file():
                total += path.stat().st_blocks * 512
        except FileNotFoundError:
            continue
    return total


@dataclass(frozen=True)
class OrdinaryCpuStudyAdapter:
    """Private fixed adapter, requiring original live compilation/runtime owners."""

    compiled_arms: object
    registry: ProgramAccessRegistry
    engines: tuple[CpuEngineSelection, ...]

    def verify(self, request):
        from .component_compiled_arms import CompiledStaticArmFeatures

        plan = plan_timing_study(request)
        if (
            type(self.compiled_arms) is not CompiledStaticArmFeatures
            or type(self.registry) is not ProgramAccessRegistry
        ):
            raise StageGateError("timing study requires the fixed ordinary live compiled/runtime owners")
        self.compiled_arms.verify()
        if (
            type(self.engines) is not tuple
            or not self.engines
            or any(type(row) is not CpuEngineSelection for row in self.engines)
            or len({row.engine for row in self.engines}) != len(self.engines)
        ):
            raise StageGateError("timing study requires distinct original CPU engine owners")
        if {row.engine: row.identity_sha256 for row in self.engines} != {
            row["engine"]: row["identity_sha256"] for row in request["engines"]
        } or any(type(row) is not CpuEngineSelection or row.target != request["target"] for row in self.engines):
            raise StageGateError("timing study engine roster differs from the original selected adapter")
        if (str(self.registry.path), self.registry.sha256) != (
            request["registry"]["path"],
            request["registry"]["sha256"],
        ):
            raise StageGateError("timing study changed the original protected program registry")
        execution = self.compiled_arms.execution
        if execution.independent_runtime.hardware_intake.target != request["target"]:
            raise StageGateError("timing study target differs from the original evaluated hardware")
        if (
            str(execution.baseline_admission.target_descriptor),
            sha256_file(execution.baseline_admission.target_descriptor),
        ) != (request["target_descriptor"]["path"], request["target_descriptor"]["sha256"]):
            raise StageGateError("timing study changed the independently selected target descriptor")
        if Path(request["resources"]["work_root"]) != execution.output:
            raise StageGateError("timing study work root differs from the original private execution owner")
        for selected in self.engines:
            selected.identity()
        return plan

    def profile(self, *, request, compiler, member, corpus, workspace, stage, engine):
        """Run one explicitly selected ordinary stage on allocated AWS resources.

        The controller supplies actual live corpus/member objects. Planning never
        loads them. Original complete-cost/calibration/held rows remain UNKNOWN
        until their existing owners have actually produced those observations.
        """
        from .supervised_feedback import bounded_feedback

        self.verify(request)
        request = loads(canonical_json(request))
        _allocated(request)
        self.registry.require_public(member.source_sha256)
        selected = [
            row
            for row in request["programs"]
            if (row["family"], row["capsule"], row["source_sha256"])
            == (member.family, member.capsule, member.source_sha256)
        ]
        if len(selected) != 1 or selected[0]["scope"] != "component_program":
            raise StageGateError(
                "whole-model CPU study execution is UNKNOWN; no ordinary independently qualified issuer"
            )
        if stage not in {"compile", "feature_extraction", "execution"} or engine not in {
            row.engine for row in self.engines
        }:
            raise StageGateError("timing study stage has no fixed ordinary adapter")
        return bounded_feedback(
            _profile_stage,
            timeout_s=request["max_stage_seconds"],
            output=Path(workspace).parent / "profiles",
            kwargs={
                "adapter": self,
                "request": request,
                "compiler": compiler,
                "member": member,
                "corpus": corpus,
                "workspace": Path(workspace),
                "stage": stage,
                "engine": engine,
            },
        )


def _profile_stage(*, adapter, request, compiler, member, corpus, workspace, stage, engine):
    if type(adapter) is not OrdinaryCpuStudyAdapter:
        raise StageGateError("timing study stage requires its fixed ordinary live adapter")
    adapter.verify(request)
    _allocated(request)
    os.sched_setaffinity(0, request["resources"]["cpu_ids"])
    environment = request["environments"][engine if stage == "execution" else stage]
    os.environ.clear()
    os.environ.update(environment)
    work_root = Path(request["resources"]["work_root"])
    if not workspace.is_relative_to(work_root):
        raise StageGateError("timing study stage escaped its explicitly allocated private work owner")
    before_disk, peak_rss, peak_disk, errors = _disk_bytes(work_root), 0, 0, []
    stop = threading.Event()

    def sample():
        nonlocal peak_rss, peak_disk
        while not stop.is_set():
            try:
                peak_rss = max(peak_rss, _tree_rss())
                peak_disk = max(peak_disk, _disk_bytes(work_root))
            except Exception as error:
                errors.append(type(error).__name__)
                return
            stop.wait(0.05)

    sampler = threading.Thread(target=sample)
    wall = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    sampler.start()
    try:
        if stage == "compile":
            result = adapter.compiled_arms.compile(
                compiler=compiler,
                member=member,
                corpus=corpus,
                workspace=workspace,
                timeout_s=request["max_stage_seconds"],
            )
        elif stage == "feature_extraction":
            adapter.compiled_arms.extract(compiler=compiler, member=member, corpus=corpus, workspace=workspace)
            result = {"features": str(workspace / "features" / "static_features.json")}
        else:
            from .evaluation_fidelity import evaluate_public_program_cpu

            selection = next(row for row in adapter.engines if row.engine == engine)
            identity = selection.identity()
            numerical = evaluate_public_program_cpu(
                execution=adapter.compiled_arms.execution,
                selection=selection,
                registry=adapter.registry,
                compiler=compiler,
                member=member,
                corpus=corpus,
                workspace=workspace,
                timeout_s=request["max_stage_seconds"],
            )
            invocations = [
                path
                for path in workspace.rglob("invocation.json")
                if mapping_file(path).get("stage") == "selected_rocc_process"
            ]
            if not invocations:
                raise StageGateError("timing study lacks the actual selected engine invocation")
            for path in invocations:
                row = I.verify(path)
                I.require_environment(path, environment=environment)
                if row["executable"] != identity["binary"]:
                    raise StageGateError("timing study execution is not joined to its original actual ELF/engine")
            ordinary = numerical["ordinary_result"]
            result = {
                "engine": identity,
                "numerical_status": numerical["numerical_status"],
                "result_path": str(ordinary["result_path"]),
                "receipt_path": str(ordinary["receipt_path"]),
                "scope": "complete ordinary compile and full numerical execution; no elapsed-cost subtraction",
                "invocations": [{"path": str(path), "sha256": sha256_file(path)} for path in invocations],
            }
    finally:
        stop.set()
        sampler.join(timeout=5)
    after = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    disk = _disk_bytes(work_root)
    if errors or sampler.is_alive():
        raise StageGateError("timing study resource observation is incomplete")
    adapter.verify(request)
    return {
        "schema": "merlin.aws_cpu_stage_profile.v1",
        "stage": stage,
        "engine": engine,
        "program_sha256": member.source_sha256,
        "request_sha256": document_sha256(request),
        "wall_seconds": time.monotonic() - wall,
        "cpu_seconds": sum(
            end.ru_utime + end.ru_stime - start.ru_utime - start.ru_stime
            for start, end in zip(usage, after, strict=True)
        ),
        "peak_process_tree_rss_bytes": max(peak_rss, _tree_rss()),
        "disk_bytes_before": before_disk,
        "peak_output_disk_bytes": max(before_disk, peak_disk, disk),
        "disk_bytes_after": disk,
        "environment": I.environment_identity(environment),
        "ordinary_products": result,
        "sampling_seconds": 0.05,
        "cache_state": "UNKNOWN",
        "timing_status": "UNKNOWN",
        "promotion": "NONE",
        "scope": "observed process/owned descendant and retained output costs; external services/writes UNKNOWN",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.request.stat().st_size > 1 << 20:
        parser.error("timing study request exceeds its metadata budget")
    try:
        request = loads(args.request.read_text())
        plan = plan_timing_study(request)
        if args.output.exists() or args.output.is_symlink():
            raise StageGateError("timing study plan destination must be fresh")
        write_json(args.output, plan)
    except (StageGateError, ValueError, OSError) as error:
        parser.error(str(error))
    print("planned AWS CPU timing study; execution_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
