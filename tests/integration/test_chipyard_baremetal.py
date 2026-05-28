"""Real bare-metal Chipyard simulator tests, parametrised from the
``_compat_matrix`` registry.

Every cell in the matrix becomes one test. The expected outcome dictates
how the test asserts:

* ``EXPECTED_PASS`` — sim must run end-to-end, exit 0, and stdout must
  contain a ``success_marker``.
* ``HARNESS_ONLY`` — sim must boot, the loader must accept the ELF, and
  the elaborated hardware must turn cycles. Used for kernels that don't
  fully run on the available config but where the harness coming up still
  tells us the build is sane.
* ``XFAIL_KNOWN_BROKEN`` — wrapped with ``pytest.mark.xfail(strict=False)``
  so an unexpected pass surfaces as XPASS and we move the entry.
* ``SKIP_NO_ARTIFACT`` / ``XFAIL_TOOLCHAIN_GAP`` — declarative skips with
  the matrix's ``reason`` field.

To add a new (workload, config, simulator) test, add a ``Cell`` to
``_compat_matrix.MATRIX`` and a ``Workload`` to ``WORKLOADS`` if needed.
**No** test function edits required — generalises beyond Radiance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ._compat_matrix import (
    MATRIX,
    WORKLOADS,
    Cell,
    OutcomeKind,
    cell_id,
    workload_path,
)
from .conftest import (
    dramsim_ini_dir,
    find_vcs_simulator,
    find_verilator_simulator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.chipyard,
]


def _resolve_simulator(cell: Cell, chipyard_root: Path) -> Path | None:
    if cell.simulator == "vcs":
        return find_vcs_simulator(chipyard_root, cell.config)
    if cell.simulator == "verilator":
        return find_verilator_simulator(chipyard_root, cell.config)
    raise AssertionError(f"unknown simulator kind: {cell.simulator!r}")


def _build_command(cell: Cell, sim_binary: Path, workload_binary: Path, chipyard_root: Path) -> list[str]:
    """Build the canonical run-binary command line for this cell."""
    if cell.simulator == "verilator":
        return [
            str(sim_binary),
            "+permissive",
            f"+loadmem={workload_binary}",
            "+permissive-off",
            str(workload_binary),
        ]
    if cell.simulator == "vcs":
        return [
            str(sim_binary),
            "+permissive",
            "+dramsim",
            f"+dramsim_ini_dir={dramsim_ini_dir(chipyard_root)}",
            f"+max-cycles={cell.max_cycles}",
            "+ntb_random_seed_automatic",
            f"+loadmem={workload_binary}",
            *cell.extra_plus_args,
            "+permissive-off",
            str(workload_binary),
        ]
    raise AssertionError(f"unknown simulator kind: {cell.simulator!r}")


def _run(cell: Cell, sim_binary: Path, workload_binary: Path, chipyard_root: Path) -> subprocess.CompletedProcess:
    cmd = _build_command(cell, sim_binary, workload_binary, chipyard_root)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=cell.timeout_seconds,
        cwd=sim_binary.parent,
    )


def _assert_elf_sanity(workload_binary: Path, workload_name: str) -> None:
    workload_def = WORKLOADS[workload_name]
    head = workload_binary.read_bytes()[:20]
    assert head[:4] == b"\x7fELF", f"{workload_binary} is not an ELF"
    if workload_def.elf_class is not None:
        assert (
            head[4] == workload_def.elf_class
        ), f"{workload_binary} elf_class={head[4]} but expected {workload_def.elf_class}"
    if workload_def.elf_machine is not None:
        machine = int.from_bytes(head[18:20], "little")
        assert (
            machine == workload_def.elf_machine
        ), f"{workload_binary} e_machine={machine:#x} but expected {workload_def.elf_machine:#x}"


@pytest.fixture(scope="session")
def matrix_session_cache() -> dict[tuple[str, str], subprocess.CompletedProcess]:
    """Per-session cache so multiple tests against the same (sim, workload)
    don't re-run a slow simv. Keyed by (sim_path, workload_path)."""
    return {}


@pytest.mark.parametrize("cell", MATRIX, ids=[cell_id(c) for c in MATRIX])
def test_compat_matrix(
    cell: Cell,
    chipyard_root: Path,
    matrix_session_cache: dict[tuple[str, str], subprocess.CompletedProcess],
) -> None:
    # 0. Resolve workload binary
    if cell.workload not in WORKLOADS:
        pytest.skip(f"unknown workload {cell.workload!r}")
    workload_binary = workload_path(cell.workload, REPO_ROOT)
    if not workload_binary.exists():
        msg = f"{cell.workload}: workload binary missing at {workload_binary}"
        if cell.expected is OutcomeKind.SKIP_NO_ARTIFACT:
            pytest.skip(msg)
        # Even matrix entries that *expect* to pass must skip if the
        # artifact isn't there — the test result is undetermined.
        pytest.skip(msg)

    # 1. ELF sanity (cheap, runs even if sim is missing)
    _assert_elf_sanity(workload_binary, cell.workload)

    # 2. Resolve simulator binary
    sim_binary = _resolve_simulator(cell, chipyard_root)
    if sim_binary is None:
        msg = f"{cell.simulator} simulator for {cell.config} not built locally. " f"{cell.reason}"
        # If the cell is XFAIL_KNOWN_BROKEN with reason naming a build
        # blocker (e.g., Verilator BLKANDNBLK), the missing binary IS the
        # documented failure — record it as xfail, not skip.
        if cell.expected is OutcomeKind.XFAIL_KNOWN_BROKEN and any(
            kw in cell.reason for kw in ("Verilator", "VCS", "rejects", "blocked")
        ):
            pytest.xfail(f"sim cannot be built: {cell.reason}")
        pytest.skip(msg)

    # 3. Apply XFAIL marker dynamically for known-broken cells
    if cell.expected is OutcomeKind.XFAIL_KNOWN_BROKEN:
        request_xfail = pytest.xfail
        # We do the run, then xfail on bad outcome — pytest's runtime xfail.
        # Need to wrap the body in try/except to catch failures.
        try:
            result = matrix_session_cache.get((str(sim_binary), str(workload_binary)))
            if result is None:
                result = _run(cell, sim_binary, workload_binary, chipyard_root)
                matrix_session_cache[(str(sim_binary), str(workload_binary))] = result
            workload_def = WORKLOADS[cell.workload]
            assert result.returncode == 0
            assert any(m in result.stdout for m in workload_def.success_markers)
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            request_xfail(f"{cell.reason} ({exc.__class__.__name__})")
        return

    if cell.expected is OutcomeKind.XFAIL_TOOLCHAIN_GAP:
        pytest.xfail(cell.reason)

    # 4. Run the simulator (cached if a prior cell in this session ran it)
    cache_key = (str(sim_binary), str(workload_binary))
    result = matrix_session_cache.get(cache_key)
    if result is None:
        try:
            result = _run(cell, sim_binary, workload_binary, chipyard_root)
        except subprocess.TimeoutExpired as exc:
            if cell.expected is OutcomeKind.HARNESS_ONLY:
                # Harness-only cells are allowed to time out as long as the
                # captured stdout shows the harness markers.
                stdout = (exc.stdout or b"").decode("utf-8", errors="ignore")
                stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
                combined = stdout + stderr
                missing = [m for m in cell.harness_markers if m not in combined]
                assert not missing, (
                    f"{cell_id(cell)}: timed out and harness markers absent: "
                    f"{missing}\nlast-output:\n{combined[-2000:]}"
                )
                return
            raise
        matrix_session_cache[cache_key] = result

    workload_def = WORKLOADS[cell.workload]

    if cell.expected is OutcomeKind.EXPECTED_PASS:
        assert result.returncode == 0, (
            f"{cell_id(cell)}: sim exit {result.returncode}\nSTDOUT:\n"
            f"{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-1000:]}"
        )
        assert any(m in result.stdout for m in workload_def.success_markers), (
            f"{cell_id(cell)}: no success marker {workload_def.success_markers} " f"in stdout\n{result.stdout[-2000:]}"
        )
        return

    if cell.expected is OutcomeKind.HARNESS_ONLY:
        combined = result.stdout + result.stderr
        missing = [m for m in cell.harness_markers if m not in combined]
        assert not missing, f"{cell_id(cell)}: harness markers missing: {missing}\n" f"last-output:\n{combined[-2000:]}"
        return

    pytest.fail(f"unhandled expected outcome: {cell.expected}")


def test_compat_matrix_is_well_formed() -> None:
    """Static sanity: every Cell references a known Workload and a sensible
    simulator kind. Catches typos in the matrix during code review."""
    for cell in MATRIX:
        assert cell.workload in WORKLOADS, f"unknown workload {cell.workload!r}"
        assert cell.simulator in ("vcs", "verilator"), f"unknown simulator {cell.simulator!r} for {cell.config}"
        assert isinstance(cell.expected, OutcomeKind)


def test_compat_matrix_documents_known_failures() -> None:
    """Every XFAIL cell must carry a non-empty ``reason``. We treat
    documentation as part of the matrix's contract."""
    bad = [
        cell_id(c)
        for c in MATRIX
        if c.expected in (OutcomeKind.XFAIL_KNOWN_BROKEN, OutcomeKind.XFAIL_TOOLCHAIN_GAP) and not c.reason.strip()
    ]
    assert not bad, f"XFAIL cells lacking a reason: {bad}"
