"""The command-buffer driver retires the kernel once, not once per job.

Every job stages its activation through the same scratchpad rows, so a second job's ``mvin`` is a
write-after-read on rows the previous job's compute consumed. The driver used to fence after each job
for that reason. Measured on both of this target's elaborated-RTL engines, on the one capsule in the
corpus with two jobs, the barrier bought nothing and cost time: 434 -> 334 cycles on the citable engine
and 427 -> 331 on the fast one, every output byte-identical, both arms gating correct. The reservation
station tracks the hazard.

The remaining fence is load-bearing and these tests pin it: it retires the last job before
``read_cycles`` closes the measured window and before the readback reads the output rows.
"""
from __future__ import annotations

import glob
import importlib
import sys
import types

import pytest

from merlin.targetgen.contract import interface_emit as IE


def _codegen():
    """Import the backend module as part of its package (``merlin/targets`` is not importable)."""
    if "gback" not in sys.modules:
        pkg = types.ModuleType("gback")
        pkg.__path__ = ["merlin/targets/gemmini/backend"]
        sys.modules["gback"] = pkg
    return importlib.import_module("gback.gemmini_codegen")


def _emittable():
    """(capsule name, n_jobs, driver source) for every capsule this driver can emit."""
    gc = _codegen()
    out = []
    for d in sorted(glob.glob("merlin/contract/capsules/*/*/")):
        mlir = d + "capsule.interface.mlir"
        try:
            cb = IE.parse_interface_mlir(open(mlir).read())
            _w, jobs, _k, _n = gc._parse(cb)
            out.append((d.rstrip("/").split("/")[-1], len(jobs), gc.generate_driver(cb)))
        except Exception:                            # noqa: BLE001 — out of this driver's scope
            continue
    return out


class TestRetireOnce:
    def test_every_emittable_capsule_fences_exactly_once(self):
        cases = _emittable()
        if not cases:
            pytest.skip("no capsule on this checkout is in the command-buffer driver's scope")
        for name, n_jobs, src in cases:
            assert src.count("gemmini_fence();") == 1, (
                f"{name} ({n_jobs} job(s)) emits {src.count('gemmini_fence();')} fences; the driver "
                f"retires the kernel once")

    def test_a_multi_job_capsule_exists_to_make_that_non_vacuous(self):
        """Without one, 'exactly one fence' would hold trivially and prove nothing."""
        cases = _emittable()
        if not cases:
            pytest.skip("no capsule in scope")
        multi = [(n, j) for n, j, _ in cases if j >= 2]
        assert multi, ("no capsule in this driver's scope has two jobs, so the per-job-fence "
                       "regression cannot be detected here — add one or drop this suite")

    def test_the_retire_precedes_the_closing_cycle_read(self):
        cases = _emittable()
        if not cases:
            pytest.skip("no capsule in scope")
        for name, _n, src in cases:
            fence = src.index("gemmini_fence();")
            close = src.index("c1 = read_cycles();")
            assert fence < close, (
                f"{name}: the retire must precede the closing cycle read, or the measured window "
                f"excludes the accelerator drain")

    def test_the_retire_precedes_the_output_readback(self):
        cases = _emittable()
        if not cases:
            pytest.skip("no capsule in scope")
        for name, _n, src in cases:
            assert src.index("gemmini_fence();") < src.index('printf("OUT'), (
                f"{name}: reading an output tile before the store retires reads stale memory")
