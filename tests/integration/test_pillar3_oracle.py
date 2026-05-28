"""End-to-end tests for the Pillar 3 oracle runner.

Exercises ``eval/paper/pillar3_endtoend/harness/oracle.py`` against the
real Chipyard simulators and real bare-metal workloads available on
this system. No mocks. The tests skip cleanly when prerequisites
(simulator binary, workload, Chipyard env) are missing.

Markers: ``integration``, ``slow``, ``chipyard``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PILLAR3 = REPO_ROOT / "eval" / "paper" / "pillar3_endtoend"
if str(PILLAR3) not in sys.path:
    sys.path.insert(0, str(PILLAR3))

from harness.oracle import OracleSpec, build_workload, load_oracle, run_oracle  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.chipyard,
]


def test_load_oracle_parses_radiance_spec() -> None:
    spec = load_oracle(PILLAR3 / "targets" / "radiance_full" / "oracle.yaml")
    assert spec.target == "radiance_full"
    assert spec.simulator_kind == "vcs"
    assert spec.success_kind == "harness_only"
    assert "Cyclotron" in spec.required_markers
    assert "loading ELF" in spec.required_markers


def test_load_oracle_parses_caliptra_spec() -> None:
    spec = load_oracle(PILLAR3 / "targets" / "caliptra_aes" / "oracle.yaml")
    assert spec.target == "caliptra_aes"
    assert spec.success_kind == "stdout_marker"
    assert spec.success_marker is not None


def test_build_workload_skips_when_binary_already_present(tmp_path: Path, chipyard_root: Path) -> None:
    spec = load_oracle(PILLAR3 / "targets" / "radiance_full" / "oracle.yaml")
    radiance_kernels = Path("/scratch2/agustin/radiance-kernels")
    if not (radiance_kernels / "kernels" / "vecadd" / "kernel.soc.elf").exists():
        pytest.skip("radiance vecadd kernel.soc.elf not present")
    binary, rc, wall, log = build_workload(spec, chipyard_root, log_dir=tmp_path)
    assert rc == 0, f"build failed (rc={rc}); see {log}"
    assert binary.exists()
    assert wall < 5.0, "rebuild was triggered when binary already existed"


def test_run_oracle_radiance_full_passes_harness_only(tmp_path: Path, chipyard_root: Path) -> None:
    """The radiance harness-only oracle is the cell we know works today.

    If this regresses it means the upstream Cyclotron init banner changed
    or the simv build was retired — both are real signals worth catching.
    """
    spec = load_oracle(PILLAR3 / "targets" / "radiance_full" / "oracle.yaml")
    sim = chipyard_root / "sims" / "vcs" / f"simv-chipyard.harness-{spec.chipyard_config}"
    if not sim.exists():
        pytest.skip(f"simv not built at {sim}")
    if not Path(spec.workload_binary).exists():
        pytest.skip(f"workload binary not built at {spec.workload_binary}")

    result = run_oracle(spec, chipyard_root, log_dir=tmp_path / "oracle_logs")
    assert result.passed, f"radiance harness oracle regressed: {result.reason}\n" f"see {result.stdout_path}"
    assert result.workload_built
    assert result.sim_returncode is not None
    assert result.sim_wall_clock_seconds > 0.0


def test_run_oracle_marker_not_found_returns_failure(tmp_path: Path, chipyard_root: Path) -> None:
    """Synthesise an oracle that asks for a marker the simulator never
    prints, and verify the runner reports passed=False with an honest reason.
    """
    spec_yaml = PILLAR3 / "targets" / "radiance_full" / "oracle.yaml"
    real = load_oracle(spec_yaml)
    sim = chipyard_root / "sims" / "vcs" / f"simv-chipyard.harness-{real.chipyard_config}"
    if not sim.exists():
        pytest.skip("radiance simv not built")
    if not Path(real.workload_binary).exists():
        pytest.skip("radiance workload not built")

    # Re-use the same sim + workload but flip success_kind to stdout_marker
    # with a marker that will never appear.
    fake = OracleSpec(
        target="radiance_full__synthetic_failure",
        chipyard_generator=real.chipyard_generator,
        chipyard_config=real.chipyard_config,
        simulator_kind=real.simulator_kind,
        simulator_binary_pattern=real.simulator_binary_pattern,
        workload_binary=real.workload_binary,
        workload_build_workdir=real.workload_build_workdir,
        workload_build_cmd=real.workload_build_cmd,
        run_cmd_template=real.run_cmd_template,
        run_timeout_seconds=120,
        success_kind="stdout_marker",
        success_marker="THIS_MARKER_NEVER_APPEARS_xyzzy",
    )
    result = run_oracle(fake, chipyard_root, log_dir=tmp_path / "oracle_logs_fail")
    assert not result.passed
    assert "absent" in result.reason or "exited" in result.reason or "timed out" in result.reason


def test_run_oracle_missing_sim_returns_failure_with_helpful_reason(tmp_path: Path, chipyard_root: Path) -> None:
    """When the sim binary is missing, the oracle must produce a clear
    actionable error, not crash."""
    spec_yaml = PILLAR3 / "targets" / "radiance_full" / "oracle.yaml"
    real = load_oracle(spec_yaml)
    fake = OracleSpec(
        target="radiance_full__no_sim",
        chipyard_generator=real.chipyard_generator,
        chipyard_config="DefinitelyNotARealConfig",
        simulator_kind=real.simulator_kind,
        simulator_binary_pattern=real.simulator_binary_pattern,
        workload_binary=real.workload_binary,
        workload_build_workdir=real.workload_build_workdir,
        workload_build_cmd=real.workload_build_cmd,
        run_cmd_template=real.run_cmd_template,
        run_timeout_seconds=10,
        success_kind="harness_only",
        required_markers=["never"],
    )
    result = run_oracle(fake, chipyard_root, log_dir=tmp_path / "oracle_logs_no_sim")
    assert not result.passed
    assert "simulator binary missing" in result.reason
    assert "DefinitelyNotARealConfig" in result.reason
