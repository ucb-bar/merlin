"""Hermetic isolation proof for the shared agentic bwrap sandbox — one guard for EVERY roster target.

For each ``merlin/experiments/*/target_experiment.yaml`` the shared, descriptor-driven sandbox
(:mod:`merlin.targetgen.sandbox`) is built and its isolation is asserted at the POLICY level — by
replaying the real bwrap mount table, WITHOUT launching bwrap — so this runs in a hermetic CI:

  * COVERAGE — under a worst-case bundle that over-binds every answer-bearing parent (contract tree,
    the merlin package, the target's backend dir), NO declared answer surface stays reachable. This is
    the drift/cheat guard (the historical cheat gap — a hard-coded path that left the memory dir
    unmasked — is exactly a non-empty coverage gap).
  * NON-VACUOUS — the same guard, run against an argv with the mask pass REMOVED, DOES flag the surface
    (so a green coverage result is meaningful, not a no-op).
  * TOOLS — every tool the target's compute-unit KIND / sim family requires is granted (its bind path is
    exposed in the argv), and the set is correct per kind (systolic ⇒ RTL-sim tools; simt ⇒ not).

The LIVE probe (actually launching bwrap to run the tools + confirm a golden reads empty) is guarded by
``bwrap`` + toolchain availability and skips when they are absent — so CI stays hermetic while a
developer box still gets the end-to-end proof.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.sandbox import build_sandbox
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
from merlin.targetgen.sandbox.toolchain import UNIVERSAL_PROBES
from merlin.targetgen.target_experiment import load_target_experiment


def _roster() -> list[Path]:
    """Every roster target's descriptor (globbed, not hard-listed)."""
    exp = merlin_dir() / "experiments"
    return sorted(exp.glob("*_capsule_bench_v0/target_experiment.yaml"))


def _max_exposure_bundle(target: str) -> dict:
    """A worst-case bundle that BINDS every answer-bearing parent, forcing the mask pass to re-hide every
    surface a legit bind could re-expose. Target-agnostic: parents are derived, not per-target."""
    allowed = [{"path": "merlin/contract"}, {"path": "merlin/python/merlin"},
               {"path": f"out/artifacts/targets/{target}"}]
    return {"allowed": [a for a in allowed if (repo_root() / a["path"]).exists()]}


def _ids(paths):
    return [p.parent.name.replace("_capsule_bench_v0", "") for p in paths]


ROSTER = _roster()


@pytest.mark.parametrize("descriptor", ROSTER, ids=_ids(ROSTER))
def test_answer_surface_is_derived_and_complete(descriptor):
    """The mask set is DERIVED from the descriptor + declared registry (never hand-listed) and covers the
    origins the descriptor implies."""
    te = load_target_experiment(descriptor)
    surfaces = answer_surfaces(te)
    origins = {s.origin for s in surfaces}
    # goldens + the oracle/grader registry are always present (registry is target-independent).
    assert "golden" in origins, f"{te.target}: no goldens derived from corpus"
    assert "oracle" in origins and "grader" in origins, f"{te.target}: declared oracle/grader missing"
    # descriptor-conditioned origins:
    if te.hidden_corpus():
        assert "hidden" in origins, f"{te.target}: hidden corpus exists but not masked"
    declared_priors = [p for p in te.prior_backends
                       if (repo_root() / "out/artifacts/targets" / te.target / p).exists()]
    if declared_priors:
        assert "prior_backend" in origins, f"{te.target}: declared prior_backends not masked"


@pytest.mark.parametrize("descriptor", ROSTER, ids=_ids(ROSTER))
def test_no_answer_surface_reachable_under_worst_case_bundle(descriptor):
    """HERMETIC coverage guard: even when the bundle over-binds every answer-bearing parent, the built
    argv leaves NO answer surface reachable (mount-table replay, no bwrap launch)."""
    te = load_target_experiment(descriptor)
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        ws.mkdir()
        sb = build_sandbox(te, ws, _max_exposure_bundle(te.target))
        gap = sb.coverage_gap()
        assert gap == [], f"{te.target}: answer surfaces reachable in sandbox: {[s.label for s in gap]}"


@pytest.mark.parametrize("descriptor", ROSTER, ids=_ids(ROSTER))
def test_coverage_guard_is_not_vacuous(descriptor):
    """The guard MUST flag a real gap: build the argv with the answer-mask pass REMOVED and confirm at
    least one surface (a golden, re-exposed by the bound contract tree) is then reachable."""
    te = load_target_experiment(descriptor)
    surfaces = answer_surfaces(te)
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        ws.mkdir()
        # base + toolchain binds, WITHOUT apply_answer_masks -> the contract bind re-exposes the goldens
        from merlin.targetgen.sandbox import toolchain as TC
        unmasked = (BW.base_argv(ws, _max_exposure_bundle(te.target))
                    + BW.claude_runtime_binds() + TC.toolchain_binds(te))
        gap = BW.coverage_gap(unmasked, surfaces)
        assert gap, f"{te.target}: guard found nothing unmasked even without the mask pass (vacuous)"


@pytest.mark.parametrize("descriptor", ROSTER, ids=_ids(ROSTER))
def test_required_tools_are_granted(descriptor):
    """Every tool this target's kind/sim family requires has its bind path EXPOSED in the argv (so the
    toolchain is granted), and the universal tools are always required."""
    te = load_target_experiment(descriptor)
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        ws.mkdir()
        sb = build_sandbox(te, ws)
        argv = sb.argv()
        assert {p.label for p in UNIVERSAL_PROBES} <= {p.label for p in sb.required_tools}
        for probe in sb.required_tools:
            if probe.bind and Path(probe.bind).exists():
                assert BW.is_exposed(argv, Path(probe.bind)), \
                    f"{te.target}: tool {probe.label} not granted (bind {probe.bind} not exposed)"


def test_kind_routes_tool_requirements():
    """Kind/sim routing is correct: a systolic/chipyard target requires the RTL-sim tools; a simt/
    cyclotron target does not (proves routing is by kind/family, not a target name)."""
    by_target = {}
    for descriptor in ROSTER:
        te = load_target_experiment(descriptor)
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ws"
            ws.mkdir()
            sb = build_sandbox(te, ws)
            by_target[te.target] = (sb.kind, sb.te.sim_via, {p.label for p in sb.required_tools})
    if "gemmini" in by_target:
        kind, sim, tools = by_target["gemmini"]
        assert kind == "systolic" and sim == "chipyard"
        assert {"spike", "riscv64-unknown-elf-gcc", "verilator RTL sim"} <= tools
    if "radiance" in by_target:
        kind, sim, tools = by_target["radiance"]
        assert kind == "simt" and sim == "cyclotron"
        assert "spike" not in tools and "verilator RTL sim" not in tools


# --------------------------------------------------------------------------- live (guarded) probe
def _live_ready(te) -> bool:
    if not shutil.which("bwrap"):
        return False
    from merlin.targetgen.sandbox.toolchain import LLVM, VENV
    return Path(VENV).exists() and Path(LLVM).exists()


@pytest.mark.parametrize("descriptor", ROSTER, ids=_ids(ROSTER))
def test_live_bwrap_tools_work_and_golden_masked(descriptor):
    """LIVE end-to-end: launch the REAL bwrap command and confirm (a) the universal tools run and (b) a
    golden reads empty inside the sandbox. Skips when bwrap or the host toolchain is unavailable (keeps
    CI hermetic)."""
    te = load_target_experiment(descriptor)
    if not _live_ready(te):
        pytest.skip("bwrap or host toolchain unavailable — hermetic-only run")
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        (ws / "merlin" / "python").mkdir(parents=True)
        sb = build_sandbox(te, ws, _max_exposure_bundle(te.target))
        # (a) universal tools run
        for probe in UNIVERSAL_PROBES:
            cmd = sb.wrap(f"{probe.cmd} >/dev/null 2>&1 && echo OK || echo FAIL")
            out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
            assert out.stdout.strip().endswith("OK"), f"{te.target}: tool {probe.label} failed live"
        # (b) a golden reads empty (masked) inside the sandbox
        golden = next((s for s in sb.answer_surfaces if s.origin == "golden"), None)
        if golden:
            probe = f'if head -c1 "{golden.path}" >/dev/null 2>&1 && test -s "{golden.path}"; ' \
                    f'then echo VISIBLE; else echo masked; fi'
            out = subprocess.run(["bash", "-c", sb.wrap(probe)], capture_output=True, text=True, timeout=120)
            assert out.stdout.strip().endswith("masked"), f"{te.target}: golden VISIBLE inside sandbox"
