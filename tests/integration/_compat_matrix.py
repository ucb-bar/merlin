"""Workload × Chipyard-config × simulator compatibility matrix.

This is the **single source of truth** for what end-to-end tests we expect
to pass, what we expect to skip (because an artifact isn't built locally),
and what we expect to fail (because of a known upstream gap, an ISA
mismatch, or a config that the workload wasn't designed for).

Every cell of the matrix carries:
  * ``workload`` — kernel name (a key in ``WORKLOADS``).
  * ``config``   — Chipyard config (e.g., ``RadianceMuonConfig``).
  * ``simulator``— ``vcs`` | ``verilator``.
  * ``expected`` — one of ``OutcomeKind`` enum values.
  * ``reason``   — human-readable explanation; appears in skip / xfail
    messages and in the generated documentation.
  * ``ticket``   — optional URL or commit reference for follow-up.

When a matrix entry's expected outcome flips (e.g., a workload that was
``XFAIL_KNOWN_BROKEN`` starts passing), pytest ``xfail(strict=False)``
surfaces it as ``XPASS`` and we move the entry to ``EXPECTED_PASS``.

To add a new bring-up target:
  1. Compile the workload with the right toolchain.
  2. Add a ``Workload`` entry to ``WORKLOADS`` (binary path, oracle,
     timeout, ELF expectations).
  3. Add a ``Cell`` to ``MATRIX`` for each (config, simulator) you intend
     to support, with the realistic expected outcome.

The bare-metal test module reads from this matrix only — no hard-coded
test functions per target.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

# Resolved at import time so callers can read paths without re-importing.
DEFAULT_CHIPYARD = Path("/scratch2/agustin/chipyard")
DEFAULT_RADIANCE_KERNELS = Path("/scratch2/agustin/radiance-kernels")
DEFAULT_ZEPHYR_WS = Path("/scratch2/agustin/zephyr-workspace/zephyr-chipyard-sw")


class OutcomeKind(str, enum.Enum):
    """What we honestly expect to happen for a (workload, config, sim) cell."""

    EXPECTED_PASS = "pass"
    """The cell should run end-to-end and assert success."""

    SKIP_NO_ARTIFACT = "skip_no_artifact"
    """Sim binary or workload binary not built locally; not a test failure."""

    XFAIL_KNOWN_BROKEN = "xfail_known_broken"
    """Real upstream / hardware / toolchain gap — failure is expected and
    documented. ``strict=False`` so an unexpected pass surfaces as XPASS."""

    XFAIL_TOOLCHAIN_GAP = "xfail_toolchain_gap"
    """The required toolchain (e.g., llvm-muon, Zephyr SDK) is missing
    from the active environment. Distinguished from ``SKIP_NO_ARTIFACT``
    because the *capability* — not just the artifact — is missing."""

    HARNESS_ONLY = "harness_only"
    """We only assert the simulator boots, loads the ELF, and the
    elaborated hardware turns cycles. Used for configs where full workload
    completion is out of scope (slow, kernel/config mismatch, etc.)."""


@dataclass(frozen=True, slots=True)
class Workload:
    """A bare-metal RISC-V workload + oracle definition."""

    name: str
    binary: Path
    """Absolute path to the ELF as built locally."""
    success_markers: tuple[str, ...] = ()
    """Substrings any of which signals success in the sim's stdout."""
    failure_markers: tuple[str, ...] = ()
    """Substrings any of which signals an *expected* hard failure."""
    elf_class: int | None = None
    """1=ELF32, 2=ELF64; checked when present."""
    elf_machine: int | None = 0xF3
    """Expected e_machine; default RISC-V (0xF3)."""


@dataclass(frozen=True, slots=True)
class Cell:
    workload: str
    config: str
    simulator: str  # 'vcs' | 'verilator'
    expected: OutcomeKind
    reason: str = ""
    ticket: str = ""
    # Per-cell tuning knobs:
    timeout_seconds: int = 300
    max_cycles: int = 10_000_000
    extra_plus_args: tuple[str, ...] = field(default_factory=tuple)
    # For HARNESS_ONLY cells: what stdout substrings indicate the harness
    # came up correctly (Cyclotron init, scheduler tick, etc.).
    harness_markers: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

WORKLOADS: dict[str, Workload] = {
    "chipyard_hello": Workload(
        name="chipyard_hello",
        binary=Path("eval/oracle_kernels/chipyard_hello.riscv"),
        success_markers=("Hello world from core 0",),
        elf_class=2,
    ),
    "radiance_vecadd_soc": Workload(
        name="radiance_vecadd_soc",
        binary=DEFAULT_RADIANCE_KERNELS / "kernels" / "vecadd" / "kernel.soc.elf",
        success_markers=("PASS", "passed", "Pass", "OK"),
        elf_class=2,
    ),
}


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

MATRIX: tuple[Cell, ...] = (
    # ---- Saturn OPU Shuttle: pre-built Verilator sim, runs canonical hello.
    Cell(
        workload="chipyard_hello",
        config="OPUMXV256D128ShuttleConfig",
        simulator="verilator",
        expected=OutcomeKind.EXPECTED_PASS,
        reason="Pre-built Saturn OPU sim runs RV64 bare-metal binaries.",
        timeout_seconds=120,
    ),
    # ---- Radiance Muon VCS: pre-built simv, harness comes up but vecadd
    # stalls (kernel was built for a SoC-shaped config, not Muon).
    Cell(
        workload="radiance_vecadd_soc",
        config="RadianceMuonConfig",
        simulator="vcs",
        expected=OutcomeKind.HARNESS_ONLY,
        reason=(
            "vecadd's kernel.soc.elf was built against a SoC-shaped config "
            "(RadianceTapeoutSimTraceConfig / RadianceCyclotronConfig). On "
            "RadianceMuonConfig the simv harness initialises Cyclotron and "
            "loads the ELF, but the Muon scheduler stalls waiting on a "
            "peripheral the Muon-only config does not implement. Harness-"
            "level boot is the strongest assertion we can make today."
        ),
        ticket="upstream: chipyard/generators/radiance/test/run_binary_tests.py "
        "uses RadianceTapeoutSimTraceConfig for SoC tests",
        timeout_seconds=180,
        max_cycles=200_000,
        harness_markers=("Cyclotron", "loading ELF", "scheduler"),
    ),
    # ---- Radiance Muon VCS: end-to-end vecadd. Documented xfail until we
    # wire the SoC-shaped config. Strict=False so XPASS surfaces.
    Cell(
        workload="radiance_vecadd_soc",
        config="RadianceMuonConfig",
        simulator="vcs",
        expected=OutcomeKind.XFAIL_KNOWN_BROKEN,
        reason=(
            "End-to-end vecadd completion on RadianceMuonConfig — flips "
            "green when we either rebuild vecadd for Muon or build "
            "RadianceTapeoutSimTraceConfig / RadianceCyclotronConfig."
        ),
        timeout_seconds=900,
        max_cycles=10_000_000,
    ),
    # ---- Radiance Muon Verilator: blocked at sim build by Muon CVFPU.
    Cell(
        workload="chipyard_hello",
        config="RadianceMuonConfig",
        simulator="verilator",
        expected=OutcomeKind.XFAIL_KNOWN_BROKEN,
        reason=(
            "Verilator rejects the Muon CVFPU IP "
            "(gen-collateral/fpnew_fma_multi.sv) with 50× Error-BLKANDNBLK "
            "(blocking + non-blocking assignments to the same registers). "
            "Workarounds: build with VCS instead, or patch the FPU IP. "
            "Tracked as the Muon×Verilator gap."
        ),
        ticket="upstream: pulp-platform/cvfpu fpnew_fma_multi.sv",
        timeout_seconds=120,
    ),
)


def cells_for_workload(workload: str) -> tuple[Cell, ...]:
    return tuple(c for c in MATRIX if c.workload == workload)


def cells_for_config(config: str) -> tuple[Cell, ...]:
    return tuple(c for c in MATRIX if c.config == config)


def workload_path(workload_name: str, repo_root: Path) -> Path:
    """Resolve a workload's binary path relative to the repo root if needed."""
    wl = WORKLOADS[workload_name]
    if wl.binary.is_absolute():
        return wl.binary
    return repo_root / wl.binary


def cell_id(cell: Cell) -> str:
    """Stable pytest test id."""
    return f"{cell.workload}__{cell.config}__{cell.simulator}__{cell.expected.value}"
