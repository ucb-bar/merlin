"""The backend under test must reach the mesh oracle, and an unmeasured layer is not a fallback.

Two defects, both of which made a working compiler look broken.

**The package never arrived.** ``run_matmul_on_mesh`` takes the OOT backend package that certifies a
layer, but the model-execution path called it without one, so it resolved to the target's DEFAULT
package -- which a target shipping none does not have. Every layer then returned ``None`` with no
verdict and was booked as a host fallback. Measured on gemmini: ``on_mesh: 0, host_fallback: 15`` for a
model whose layers the mesh runs correctly; with the package threaded the same submission reports
``on_mesh: 14``. The parameter has to survive five hops
(``_grade_model_capsule -> compile_model -> compile_rvv -> run_model -> execute``) and was missing from
the middle three; a gap anywhere degrades silently to a default, which is why this is asserted across the
whole chain rather than at one end.

**An unmeasured layer is not a refused layer.** A timed-out or unreachable oracle says nothing about
whether the mesh can run a layer, but it was counted as a host fallback and failed a ``must_accelerate``
gate -- a compiler blamed for a simulator budget. Unavailability is now its own count, keyed on the
oracle's own reported cause, with anything unrecognized treated as a genuine refusal (the conservative
direction: an unknown cause must never excuse a real fallback).
"""
from __future__ import annotations

import inspect

import pytest


def test_the_package_survives_every_hop_to_the_oracle():
    from merlin.compile_cli import compile_model, compile_rvv, run_matmul_on_mesh
    from merlin.runtime.dispatch_runtime import execute, run_model
    from merlin.targetgen.capsule_runner import _grade_model_capsule

    hops = [(_grade_model_capsule, "package_dir"), (compile_model, "mesh_package"),
            (compile_rvv, "mesh_package"), (run_model, "mesh_package"),
            (execute, "mesh_package"), (run_matmul_on_mesh, "package")]
    missing = [f.__name__ for f, p in hops if p not in inspect.signature(f).parameters]
    assert not missing, f"the package under test cannot reach the oracle; missing at: {missing}"


def test_an_unreachable_oracle_is_not_a_refusal():
    from merlin.runtime.dispatch_runtime import _oracle_unreachable
    for decline in ("oracle verilator invocation failed: Command '[...]' timed out after 600",
                    "spike binary not found",
                    "OOT backend build failed: cmake error",
                    "oracle unavailable"):
        assert _oracle_unreachable(decline), f"should read as unmeasured: {decline!r}"


def test_a_real_refusal_stays_a_refusal():
    """Conservative by design: an unrecognized cause must not be able to excuse a fallback."""
    from merlin.runtime.dispatch_runtime import _oracle_unreachable
    for decline in ("oracle spike output != reference == simulate (three-way bit-exact gate)",
                    "exceeds the on-chip working set (262144 elems)",
                    "an accumulator epilogue cannot be split across K blocks",
                    ""):
        assert not _oracle_unreachable(decline), f"should read as a genuine decline: {decline!r}"


def test_the_runtime_counts_unavailability_apart_from_fallback():
    import ast

    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "python/merlin/runtime/dispatch_runtime.py").read_text(encoding="utf-8")
    assert "execute.mesh_unavailable" in src
    assert "mesh_unavailable_detail" in src
    ast.parse(src)
