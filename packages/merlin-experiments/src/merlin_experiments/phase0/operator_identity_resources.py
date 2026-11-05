"""Explicit operator-only provisioning for hash-identity construction.

Public capture, reference and runtime budgets are never changed. Each source
leases one original shared CPU independently. A caller must also supervise the
worker's actual RSS, CPU, scratch and deadline externally; signal scopes are an
additional fail-closed boundary, not a substitute for that supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from merlin.common import strict_json
from merlin.common.cpu_pool import current_thread_owns_cpu_lease, shared_compute_pool

SCHEMA = "merlin.phase0.operator_identity_resource_policy.v1"
_TOKEN = object()
_LIMITS = {"window_bytes", "aggregate_compact_bytes", "source_scratch_bytes", "rss_bytes",
           "source_cpu_seconds", "source_wall_seconds", "total_cpu_seconds", "total_wall_seconds",
           "source_bytes", "total_source_bytes"}


class ResourceRefusal(ValueError):
    pass


def _ordinary(path):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or path.stat().st_mode & 0o077):
        raise ResourceRefusal("operator identity policy needs an exact owner-only ordinary file")
    return path


def _cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _alarm(_signal, _frame):
    raise ResourceRefusal("selected operator identity CPU or wall budget exhausted")


class OperatorIdentityResources:
    """Process-local resource owner, never numerical or publication authority."""

    def __init__(self, path, sha256, progress, *, _token):
        if _token is not _TOKEN:
            raise ResourceRefusal("operator resource provisioning needs its selected original policy")
        self.path, self.sha256 = _ordinary(path), sha256
        self.progress = Path(progress)
        if (self.progress.exists() or self.progress.parent != self.path.parent
                or self.progress.parent.stat().st_mode & 0o077):
            raise ResourceRefusal("operator progress needs a fresh private policy sibling")
        if self.path.stat().st_size > 65536:
            raise ResourceRefusal("operator identity resource policy exceeds its metadata bound")
        raw = self.path.read_bytes()
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != sha256:
            raise ResourceRefusal("operator identity resource policy bytes changed")
        data = strict_json.loads(raw, max_bytes=65536)
        if (not isinstance(data, dict) or set(data) != {"schema", "inventory_sha256", "start_unix_ns",
                "deadline_unix_ns", "cpu_ids", "cpu_slot_root", "limits"} or data["schema"] != SCHEMA
                or set(data["limits"]) != _LIMITS
                or any(type(v) is not int or v < 1 for v in data["limits"].values())
                or any(type(data[k]) is not int for k in ("start_unix_ns", "deadline_unix_ns"))
                or data["deadline_unix_ns"] - data["start_unix_ns"] != data["limits"]["total_wall_seconds"] * 10**9
                or not data["start_unix_ns"] <= time.time_ns() < data["deadline_unix_ns"]
                or not isinstance(data["inventory_sha256"], str) or len(data["inventory_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in data["inventory_sha256"])):
            raise ResourceRefusal("operator identity resource policy is incomplete or expired")
        cpus = data["cpu_ids"]
        if (not isinstance(cpus, list) or not cpus or any(type(v) is not int or v < 0 for v in cpus)
                or len(set(cpus)) != len(cpus)
                or os.environ.get("MERLIN_COMPUTE_CPUS") != ",".join(map(str, cpus))
                or os.environ.get("MERLIN_COMPUTE_CPU_SLOTS") != data["cpu_slot_root"]):
            raise ResourceRefusal("operator identity policy does not match the selected shared CPU namespace")
        hard_cpu = resource.getrlimit(resource.RLIMIT_CPU)[1]
        hard_file = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
        if (hard_cpu == resource.RLIM_INFINITY or hard_cpu > data["limits"]["total_cpu_seconds"] + 1
                or hard_file == resource.RLIM_INFINITY or hard_file > data["limits"]["source_scratch_bytes"]):
            raise ResourceRefusal("operator identity worker lacks its original hard CPU ceiling")
        self.data, self.pid, self.started_cpu, self.active, self.rows = data, os.getpid(), _cpu(), False, []
        self._write("selected")

    @property
    def limits(self):
        return dict(self.data["limits"])

    def validate(self, inventory_sha256, parse_limits):
        expected = tuple(self.data["limits"][key] for key in (
            "window_bytes", "aggregate_compact_bytes", "source_bytes", "total_source_bytes"))
        if self.data["inventory_sha256"] != inventory_sha256 or tuple(parse_limits) != expected:
            raise ResourceRefusal("operator resource policy does not bind the original inventory and parse limits")

    def receipt(self):
        if (self.active or time.time_ns() >= self.data["deadline_unix_ns"]
                or _cpu() - self.started_cpu > self.limits["total_cpu_seconds"]
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.limits["rss_bytes"]
                or any(row["status"] != "complete" for row in self.rows)):
            raise ResourceRefusal("operator identity construction did not complete within its selected resource policy")
        self._write("complete")
        return dict(schema=SCHEMA, policy_sha256=self.sha256, limits=self.limits,
                    stages=list(self.rows), elapsed_cpu_seconds=_cpu() - self.started_cpu)

    def _write(self, status, **extra):
        if os.getpid() != self.pid or hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ResourceRefusal("operator identity resource owner or original policy changed")
        data = dict(schema=SCHEMA, status=status, policy_sha256=self.sha256,
                    elapsed_cpu_seconds=_cpu() - self.started_cpu, deadline_unix_ns=self.data["deadline_unix_ns"],
                    stages=self.rows, **extra)
        descriptor = os.open(self.progress, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(data, output, sort_keys=True)

    @contextmanager
    def stage(self, stage, ordinal=None):
        if self.active or current_thread_owns_cpu_lease() or signal.getitimer(signal.ITIMER_REAL)[0]:
            raise ResourceRefusal("operator identity stages require an independent unleased worker scope")
        remaining = (self.data["deadline_unix_ns"] - time.time_ns()) / 10**9
        if remaining <= 0 or _cpu() - self.started_cpu >= self.limits["total_cpu_seconds"]:
            raise ResourceRefusal("operator identity total budget exhausted")
        previous_cpu, previous_alarm = signal.getsignal(signal.SIGXCPU), signal.getsignal(signal.SIGALRM)
        previous_limit = resource.getrlimit(resource.RLIMIT_CPU)
        start, start_ns, cpu, self.active = time.monotonic(), time.time_ns(), _cpu(), True
        soft = min(previous_limit[1], math.ceil(cpu + self.limits["source_cpu_seconds"]))
        resource.setrlimit(resource.RLIMIT_CPU, (soft, previous_limit[1]))
        signal.signal(signal.SIGXCPU, _alarm)
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, min(remaining, self.limits["source_wall_seconds"]))
        status = "refused"
        try:
            self._write("lease_wait", stage=stage, ordinal=ordinal, stage_start_unix_ns=start_ns,
                        stage_start_cpu_seconds=cpu)
            with shared_compute_pool(timeout_s=min(remaining, self.limits["source_wall_seconds"])):
                self._write("active", stage=stage, ordinal=ordinal,
                            stage_start_unix_ns=start_ns, stage_start_cpu_seconds=cpu,
                            actual_cpu_ids=sorted(os.sched_getaffinity(0)))
                yield
            if (_cpu() - cpu > self.limits["source_cpu_seconds"]
                    or time.monotonic() - start > self.limits["source_wall_seconds"]):
                raise ResourceRefusal("operator identity source exceeded its selected complete stage budget")
            status = "complete"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGXCPU, previous_cpu)
            signal.signal(signal.SIGALRM, previous_alarm)
            resource.setrlimit(resource.RLIMIT_CPU, previous_limit)
            self.active = False
            self.rows.append(dict(stage=stage, ordinal=ordinal, status=status,
                wall_seconds=time.monotonic() - start, cpu_seconds=_cpu() - cpu,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
            self._write(status, stage=stage, ordinal=ordinal)


def select(path, sha256, *, progress):
    return OperatorIdentityResources(path, sha256, progress, _token=_TOKEN)
