"""Zephyr-on-Chipyard integration tests.

The ``ucb-bar/zephyr-chipyard-sw`` repo provides a real Zephyr workspace
with samples (``hello_world``, ``test_mt_rvv``, ``timer``, ``iree``,
``executorch``, ``xnnpack``, ...) and a ``spike_riscv64`` board target.
Once the Zephyr SDK is installed (see
``zephyr-chipyard-sw/scripts/install_toolchain_sdk.sh``), ``west`` builds a
``zephyr.elf`` we can drop onto a Chipyard simulator just like the
canonical ``hello.riscv``.

This module exercises that path:

* Skips with a clear instruction to run the Zephyr install scripts when
  ``west``, the Zephyr workspace, or the Zephyr SDK isn't reachable.
* When all artefacts are available, builds ``hello_world`` (or any sample
  named via ``MERLIN_ZEPHYR_SAMPLE``) and runs it on the Saturn OPU
  Shuttle Verilator sim, asserting the Zephyr boot banner appears in
  stdout.

If the Zephyr ELF turns out to be incompatible with the Chipyard sim's
memory map (different linker layout, different HTIF expectations), the
test reports it via the compatibility matrix as
``XFAIL_TOOLCHAIN_GAP`` rather than silently passing.

Markers: ``integration``, ``slow``, ``chipyard``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ZEPHYR_WORKSPACE = Path("/scratch2/agustin/zephyr-workspace/zephyr-chipyard-sw")
DEFAULT_ZEPHYR_SAMPLE = "hello_world"
DEFAULT_ZEPHYR_BOARD = "spike_riscv64"
ZEPHYR_BANNER_MARKER = "Hello World!"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.chipyard,
]


@pytest.fixture(scope="module")
def zephyr_workspace() -> Path:
    candidate = Path(os.environ.get("ZEPHYR_CHIPYARD_SW", str(DEFAULT_ZEPHYR_WORKSPACE)))
    if not candidate.is_dir():
        pytest.skip(
            f"zephyr-chipyard-sw not cloned at {candidate}; "
            "git clone https://github.com/ucb-bar/zephyr-chipyard-sw.git "
            "and set ZEPHYR_CHIPYARD_SW"
        )
    if not (candidate / "samples" / DEFAULT_ZEPHYR_SAMPLE).is_dir():
        pytest.skip(f"{candidate} does not contain samples/{DEFAULT_ZEPHYR_SAMPLE}")
    return candidate


@pytest.fixture(scope="module")
def zephyr_sample(zephyr_workspace: Path) -> Path:
    sample = os.environ.get("MERLIN_ZEPHYR_SAMPLE", DEFAULT_ZEPHYR_SAMPLE)
    path = zephyr_workspace / "samples" / sample
    if not path.is_dir():
        pytest.skip(f"Zephyr sample missing: {path}")
    return path


@pytest.fixture(scope="module")
def zephyr_env(zephyr_workspace: Path) -> dict:
    """Environment dict with Zephyr SDK + ZEPHYR_BASE if available."""
    env = dict(os.environ)
    # The repo provides a setup script that exports ZEPHYR_BASE,
    # ZEPHYR_SDK_INSTALL_DIR, ZEPHYR_TOOLCHAIN_VARIANT.
    setup = zephyr_workspace / "scripts" / "set_envvars_sdk.sh"
    if setup.exists():
        # Source the script and re-export its environment to us.
        result = subprocess.run(
            ["bash", "-c", f"source {setup} >/dev/null 2>&1 && env"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    env[k] = v
    return env


@pytest.fixture(scope="module")
def west_available(zephyr_env: dict) -> str:
    candidate = zephyr_env.get("PATH", "")
    for d in candidate.split(os.pathsep):
        west = Path(d) / "west"
        if west.exists() and os.access(west, os.X_OK):
            return str(west)
    pytest.skip(
        "`west` not on PATH after sourcing scripts/set_envvars_sdk.sh. "
        "Run zephyr-chipyard-sw/scripts/install_conda.sh + "
        "scripts/install_submodules.sh + scripts/install_toolchain_sdk.sh."
    )
    return ""  # unreachable, makes type-checker happy


def _build_zephyr_sample(
    workspace: Path,
    sample_path: Path,
    board: str,
    env: dict,
    west_path: str,
    build_dir: Path,
) -> Path:
    """Build a Zephyr sample with ``west build -p -b <board>``.

    Returns the resulting ``zephyr.elf`` path.
    """
    cmd = [
        west_path,
        "build",
        "-p",
        "-b",
        board,
        str(sample_path),
        "-d",
        str(build_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=workspace,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"`west build` failed for {sample_path.name} on {board}:\n"
        f"STDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-1500:]}"
    )
    elf = build_dir / "zephyr" / "zephyr.elf"
    assert elf.exists(), f"zephyr.elf not produced at {elf}"
    return elf


def test_zephyr_hello_world_builds_for_spike(
    tmp_path: Path,
    zephyr_workspace: Path,
    zephyr_sample: Path,
    zephyr_env: dict,
    west_available: str,
) -> None:
    """End-to-end: west build -> zephyr.elf produced + is a RISC-V ELF."""
    build_dir = tmp_path / "zephyr-build"
    elf = _build_zephyr_sample(
        zephyr_workspace,
        zephyr_sample,
        DEFAULT_ZEPHYR_BOARD,
        zephyr_env,
        west_available,
        build_dir,
    )
    head = elf.read_bytes()[:20]
    assert head[:4] == b"\x7fELF"
    machine = int.from_bytes(head[18:20], "little")
    assert machine == 0xF3, f"Zephyr ELF is not RISC-V (e_machine={machine:#x})"


def test_zephyr_spike_runs_on_real_spike(
    tmp_path: Path,
    zephyr_workspace: Path,
    zephyr_sample: Path,
    zephyr_env: dict,
    west_available: str,
) -> None:
    """Run the Zephyr ELF on real Spike (the canonical Zephyr oracle).

    This is the strongest baseline before involving Chipyard at all: if
    Zephyr can't print its banner under Spike, anything downstream is
    moot. Chipyard ships Spike at
    ``$CHIPYARD_ROOT/.conda-env/riscv-tools/bin/spike``.
    """
    chipyard_root = Path(os.environ.get("CHIPYARD_ROOT", "/scratch2/agustin/chipyard"))
    spike = chipyard_root / ".conda-env" / "riscv-tools" / "bin" / "spike"
    if not spike.exists():
        pytest.skip(f"spike not available at {spike}")

    build_dir = tmp_path / "zephyr-build"
    elf = _build_zephyr_sample(
        zephyr_workspace,
        zephyr_sample,
        DEFAULT_ZEPHYR_BOARD,
        zephyr_env,
        west_available,
        build_dir,
    )
    result = subprocess.run(
        [str(spike), str(elf)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"spike exited {result.returncode}\nSTDOUT:\n{result.stdout}\n" f"STDERR:\n{result.stderr}"
    )
    assert ZEPHYR_BANNER_MARKER in result.stdout, f"Zephyr banner missing\nSTDOUT:\n{result.stdout[-2000:]}"


@pytest.mark.xfail(
    reason=(
        "Zephyr's spike_riscv64 board ELF may not be drop-in compatible with "
        "the Saturn OPU Verilator sim's memory map / HTIF expectations. "
        "When this XPASSes, promote to EXPECTED_PASS in the compat matrix."
    ),
    strict=False,
)
def test_zephyr_runs_on_saturn_opu_sim(
    tmp_path: Path,
    zephyr_workspace: Path,
    zephyr_sample: Path,
    zephyr_env: dict,
    west_available: str,
    chipyard_root: Path,
) -> None:
    """Drop the Zephyr ELF onto the Saturn OPU sim and look for the banner."""
    from .conftest import find_verilator_simulator

    sim = find_verilator_simulator(chipyard_root, "OPUMXV256D128ShuttleConfig")
    if sim is None:
        pytest.skip("Saturn OPU Shuttle sim not built locally")

    build_dir = tmp_path / "zephyr-build"
    elf = _build_zephyr_sample(
        zephyr_workspace,
        zephyr_sample,
        DEFAULT_ZEPHYR_BOARD,
        zephyr_env,
        west_available,
        build_dir,
    )

    result = subprocess.run(
        [
            str(sim),
            "+permissive",
            f"+loadmem={elf}",
            "+permissive-off",
            str(elf),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=sim.parent,
        check=False,
    )
    assert result.returncode == 0
    assert ZEPHYR_BANNER_MARKER in result.stdout


def test_zephyr_workspace_advertises_iree_executorch_xnnpack(
    zephyr_workspace: Path,
) -> None:
    """Sanity: the workspace ships the ML-runtime samples we want for the
    Pillar 3 LLM-in-loop study (executorch/xnnpack/iree).

    These give us realistic ML workloads to compile through Merlin in
    follow-on tests; their *presence* gates whether we can run a Pillar 3
    end-to-end at all.
    """
    samples_dir = zephyr_workspace / "samples"
    expected = {"hello_world", "iree", "executorch", "xnnpack"}
    actual = {p.name for p in samples_dir.iterdir() if p.is_dir()}
    missing = expected - actual
    assert not missing, f"zephyr-chipyard-sw is missing expected ML-runtime samples: {missing}"
