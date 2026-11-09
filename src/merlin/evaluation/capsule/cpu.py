"""Capsule evaluation: cpu."""

from __future__ import annotations

from merlin.evaluation.capsule.numerics import _match_by_policy
from merlin.evaluation.capsule.source import _capture_compiler_source, _finish_compiler_source
from merlin.experiments.phase1.feedback import setup_timing as _host_timing
from merlin.targetgen import capsule_golden as CG


def _grade_cpu_step(function, /, *args, **kwargs):
    """Run one synchronous grader calculation under the shared compute lease.

    Keep this scope around in-process work only. Compiler and oracle adapters
    launch supervised children that acquire their own slots, so the capsule
    runner must release the slot before invoking or waiting for either one.

    The wait for the lease is the ``grade_cpu.lease`` stage and the calculation
    its own ``grade_cpu.<name>`` stage. A calculation registered with
    :mod:`.grade_offload` runs in a worker process on the leased CPU: grading
    threads share one interpreter lock, so pure-Python engines on many capsules
    otherwise serialize on one core however many CPUs are leased.
    """
    from contextlib import ExitStack

    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.targetgen import grade_offload

    name = _grade_cpu_stages().get(id(function)) or getattr(function, "__name__", "step").lstrip("_")
    with ExitStack() as lease:
        with _host_timing.stage("grade_cpu.lease"):
            lease.enter_context(shared_compute_pool())
        with _host_timing.stage("grade_cpu." + name):
            return grade_offload.call(function, *args, **kwargs)


def _grade_cpu_stages() -> dict[int, str]:
    """Stage names of the grader's pure calculations, each registered to run out of process.

    Each takes and returns plain data and reads only its arguments and capsule files, which is what
    :func:`.grade_offload.register` requires. Anything not listed stays on the grading thread. Looked
    up per call, so a replaced function (a test's stub) is matched by its own identity, never an old one."""
    from merlin.benchharness import hash_tree
    from merlin.perf.work_volume import command_buffer_evidence
    from merlin.runtime.reference import reference_outputs
    from merlin.runtime.simulator import simulate
    from merlin.targetgen import grade_offload

    named = {
        "golden": CG.golden,
        "l0_reference": reference_outputs,
        "l1_simulate": simulate,
        "numeric_compare": CG.compare,
        "declared_numeric_compare": CG.compare_declared_outputs,
        "policy_match": _match_by_policy,
        "tree_hash": hash_tree,
        "work_volume": command_buffer_evidence,
        # Filesystem-only: hash, copy and re-hash the compiler source into this capsule's evidence.
        "capture_compiler_source": _capture_compiler_source,
        "finish_compiler_source": _finish_compiler_source,
    }
    grade_offload.register(*named.values())
    return {id(f): name for name, f in named.items()}
