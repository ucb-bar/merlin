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

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.sandbox import build_sandbox
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_tokens, weight_files
from merlin.targetgen.sandbox.toolchain import UNIVERSAL_PROBES
from merlin.targetgen.target_experiment import load_target_experiment


def _roster() -> list[Path]:
    """Every roster target's descriptor (globbed, not hard-listed)."""
    exp = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    return sorted(exp.glob("*/target_experiment.yaml"))


def _max_exposure_bundle(target: str) -> dict:
    """A worst-case bundle that BINDS every answer-bearing parent, forcing the mask pass to re-hide every
    surface a legit bind could re-expose. Target-agnostic: parents are derived, not per-target."""
    allowed = [{"path": "merlin/contract"}, {"path": "merlin/python/merlin"},
               {"path": f"out/artifacts/targets/{target}"}]
    return {"allowed": [a for a in allowed if (repo_root() / a["path"]).exists()]}


def _ids(paths):
    return [p.parent.name for p in paths]


ROSTER = _roster()

_MODEL_WEIGHT_RELS = (
    "merlin/contract/capsules/model/M2_microvit_gemmini/capsule.weights.safetensors",
    "merlin/contract/capsules/model/M3_host_island_seam_gemmini/capsule.weights.safetensors",
)


def _model_weights_or_skip() -> list[Path]:
    paths = [repo_root() / rel for rel in _MODEL_WEIGHT_RELS]
    if not all(path.is_file() for path in paths):
        pytest.skip("private model-weight assets are not provisioned in this checkout")
    return paths


def test_model_weights_are_derived_answer_surfaces_and_audit_tokens():
    """The two affordable model capstones' private instances are denied without a name allowlist."""
    paths = _model_weights_or_skip()
    descriptor = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    te = load_target_experiment(descriptor)
    derived = set(weight_files(te))
    surfaces = {surface.path: surface for surface in answer_surfaces(te)}
    for path in paths:
        assert path in derived
        assert surfaces[path].origin == "weight" and surfaces[path].kind == "file"
        assert any(token in str(path) for token in audit_tokens(te)["answer"])


def test_expected_instruction_coverage_is_an_answer_surface_and_audit_token():
    """The structural grading expectation must be as private as the numerical golden."""
    descriptor = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    te = load_target_experiment(descriptor)
    expected = next((repo_root() / "merlin/contract/capsules").rglob(
        "expected_instruction_coverage.yaml"))
    surfaces = {surface.path: surface for surface in answer_surfaces(te)}

    assert surfaces[expected].origin == "golden" and surfaces[expected].kind == "file"
    assert "expected_instruction_coverage.yaml" in audit_tokens(te)["answer"]


def test_model_weights_remain_in_private_snapshot_but_agent_mounts_are_masked(tmp_path):
    """Snapshotting preserves exact operator bytes; its only agent-facing mount is /dev/null-overlaid."""
    paths = _model_weights_or_skip()
    descriptor = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    te = load_target_experiment(descriptor)
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": rel} for rel in _MODEL_WEIGHT_RELS]}
    manifest = BW.materialize_bundle_inputs(ws, bundle)
    ws.mkdir(parents=True)
    try:
        grants = {record["path"]: record for record in manifest["grants"]}
        for rel, source in zip(_MODEL_WEIGHT_RELS, paths, strict=True):
            frozen = BW.bundle_snapshot_root(ws) / grants[rel]["snapshot"]
            assert frozen.read_bytes() == source.read_bytes()

        argv = BW.apply_answer_masks(BW.base_argv(ws, bundle), answer_surfaces(te))
        mount_args = [argv[index:index + 3] for index, token in enumerate(argv) if token == "--ro-bind"]
        for path in paths:
            assert not BW.is_exposed(argv, path)
            assert ["--ro-bind", "/dev/null", str(path)] in mount_args

        if shutil.which("bwrap"):
            probe = " && ".join(f'test ! -s "{path}"' for path in paths)
            out = subprocess.run([*argv, "bash", "-c", probe], capture_output=True, text=True, timeout=30)
            assert out.returncode == 0, out.stderr
    finally:
        BW.remove_bundle_snapshot(ws)


def test_copy_workspace_omits_m2_m3_weights_and_goldens(tmp_path):
    """The non-bwrap defense-in-depth view must not symlink either private model surface."""
    _model_weights_or_skip()
    harness = merlin_dir() / "experiments/capsule_bench/harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    import run_agent_experiment as experiment  # noqa: PLC0415

    for name in ("M2_microvit_gemmini", "M3_host_island_seam_gemmini"):
        source = merlin_dir() / "contract/capsules/model" / name
        dest = tmp_path / name
        experiment._link_filtered(source, dest)
        assert not (dest / "capsule.weights.safetensors").exists()
        assert not (dest / "golden.yaml").exists()
        assert not (dest / "expected_instruction_coverage.yaml").exists()
        assert (dest / "capsule.pytorch.py").is_file()
        assert (dest / "capsule.interface.mlir").is_file()


def test_missing_declared_grant_refuses_snapshot(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "does/not/exist.txt"}]}

    with pytest.raises(FileNotFoundError, match="unresolvable allowed grant"):
        BW.materialize_bundle_inputs(ws, bundle, repo=repo)

    assert not BW.bundle_snapshot_root(ws).exists()


def test_snapshot_bytes_do_not_follow_later_source_edits(tmp_path):
    """Falsifier for the live-worktree race: both sandbox views stay frozen."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable — immutable mount tested structurally only")
    repo = tmp_path / "repo"
    source = repo / "inputs" / "contract.txt"
    source.parent.mkdir(parents=True)
    source.write_text("before\n", encoding="utf-8")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "inputs/contract.txt"}]}

    try:
        manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
        assert manifest["n_files"] == 1
        assert manifest["n_bytes"] == len("before\n")
        assert len(manifest["content_sha256"]) == 64
        frozen = BW.bundle_snapshot_root(ws) / manifest["grants"][0]["snapshot"]
        assert not (frozen.stat().st_mode & 0o222)
        assert not (BW.bundle_snapshot_root(ws).stat().st_mode & 0o222)
        ws.mkdir(parents=True)
        (ws / "contract.txt").symlink_to(source)
        source.write_text("after\n", encoding="utf-8")
        # Resume reuses and verifies the completed snapshot rather than
        # recopying the now-mutated source or raising FileExistsError.
        assert BW.materialize_bundle_inputs(ws, bundle, repo=repo) == manifest
        source.unlink()

        argv = BW.base_argv(ws, bundle, repo=repo)
        # The source side of the grant is private frozen storage.  Even after
        # the original destination disappears, the pinned destination and the
        # workspace's absolute symlink both resolve to the snapshotted bytes.
        grant = [(argv[i + 1], argv[i + 2]) for i, token in enumerate(argv[:-2])
                 if token == "--ro-bind" and argv[i + 2] == str(source)]
        assert grant and all(src != str(source) for src, _ in grant)
        probe = f'cat "{source}"; cat "{ws / "contract.txt"}"'
        out = subprocess.run([*argv, "bash", "-c", probe], capture_output=True,
                             text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        assert out.stdout.splitlines() == ["before", "before"]
    finally:
        BW.remove_bundle_snapshot(ws)


def test_tampered_snapshot_payload_refuses_resume(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "input.txt"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "input.txt"}]}

    manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = BW.bundle_snapshot_root(ws) / manifest["grants"][0]["snapshot"]
    try:
        frozen.chmod(0o600)
        frozen.write_text("tampered", encoding="utf-8")
        with pytest.raises(RuntimeError, match="content verification failed"):
            BW.verify_bundle_snapshot(ws, bundle, repo=repo)
    finally:
        BW.remove_bundle_snapshot(ws)


def test_nested_file_named_snapshot_json_is_in_payload_digest(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "inputs" / "nested" / "snapshot.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("granted payload", encoding="utf-8")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "inputs"}]}

    manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = BW.bundle_snapshot_root(ws) / "repo" / "inputs" / "nested" / "snapshot.json"
    try:
        assert manifest["n_files"] == 1
        frozen.chmod(0o600)
        frozen.write_text("changed", encoding="utf-8")
        with pytest.raises(RuntimeError, match="content verification failed"):
            BW.verify_bundle_snapshot(ws, bundle, repo=repo)
    finally:
        BW.remove_bundle_snapshot(ws)


def test_tampered_snapshot_marker_cannot_escape_root(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "input.txt"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    escape = tmp_path / "escape.txt"
    escape.write_text("live external bytes", encoding="utf-8")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "input.txt"}]}

    manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    marker = BW.bundle_snapshot_root(ws) / "snapshot.json"
    try:
        manifest["grants"][0]["snapshot"] = "../escape.txt"
        marker.chmod(0o600)
        marker.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="escapes its root"):
            BW.verify_bundle_snapshot(ws, bundle, repo=repo)
    finally:
        BW.remove_bundle_snapshot(ws)


def test_tampered_snapshot_marker_cannot_follow_payload_symlink(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "input.txt"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    external = tmp_path / "external.txt"
    external.write_text("live external bytes", encoding="utf-8")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "input.txt"}]}

    manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    root = BW.bundle_snapshot_root(ws)
    marker = root / "snapshot.json"
    frozen = root / manifest["grants"][0]["snapshot"]
    try:
        frozen.parent.chmod(0o700)
        frozen.chmod(0o600)
        frozen.unlink()
        frozen.symlink_to(external)
        with pytest.raises(RuntimeError, match="escapes its root|contains a symlink"):
            BW.verify_bundle_snapshot(ws, bundle, repo=repo)
        with pytest.raises(RuntimeError, match="refusing to chmod.*symlink"):
            BW.remove_bundle_snapshot(ws)
    finally:
        if frozen.is_symlink():
            frozen.unlink()
        BW.remove_bundle_snapshot(ws)


def test_snapshotted_executable_remains_runnable(tmp_path):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable")
    repo = tmp_path / "repo"
    tool = repo / "bin" / "frozen-tool"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\necho SNAPSHOT_TOOL_OK\n", encoding="utf-8")
    tool.chmod(0o755)
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "bin/frozen-tool"}]}

    manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = BW.bundle_snapshot_root(ws) / manifest["grants"][0]["snapshot"]
    try:
        assert frozen.stat().st_mode & 0o111
        ws.mkdir(parents=True)
        tool.unlink()
        out = subprocess.run([*BW.base_argv(ws, bundle, repo=repo), str(tool)],
                             capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "SNAPSHOT_TOOL_OK"
    finally:
        BW.remove_bundle_snapshot(ws)


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
        sb = build_sandbox(te, ws, _max_exposure_bundle(te.target),
                           _policy_test_live_inputs=True)
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
        unmasked = (BW.base_argv(ws, _max_exposure_bundle(te.target),
                                 _policy_test_live_inputs=True)
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
        sb = build_sandbox(te, ws, _max_exposure_bundle(te.target),
                           _policy_test_live_inputs=True)
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
