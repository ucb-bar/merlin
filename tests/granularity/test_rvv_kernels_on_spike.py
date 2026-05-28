"""Per-kernel functional tests on Spike.

Each test compiles a (kernel.c, driver.c) pair through chipyard's riscv-tools
GCC, runs it under spike+pk, and asserts the driver returned 0 (success).

The reference sample at samples/research/rvv_kernels_on_spike/ is exercised
directly so this module also serves as a smoke-test for the spike_runner
fixture itself. New SaturnOPU i8 kernels (Phase C) will land as additional
parameterized cases.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Each entry: (id, kernel relative path, driver relative path).
RVV_KERNEL_CASES: list[tuple[str, str, str]] = [
    (
        "rvv_add_f32",
        "samples/research/rvv_kernels_on_spike/src/rvv_add.c",
        "samples/research/rvv_kernels_on_spike/driver/driver.c",
    ),
    (
        "saturnopu_add_f32",
        "benchmarks/SaturnOPU/kernels/abi/add_f32_workgroup.c",
        "benchmarks/SaturnOPU/kernels/drivers/add_f32_driver.c",
    ),
    (
        "saturnopu_linear_f32",
        "benchmarks/SaturnOPU/kernels/abi/linear_f32_workgroup.c",
        "benchmarks/SaturnOPU/kernels/drivers/linear_f32_driver.c",
    ),
]


@pytest.mark.chipyard
@pytest.mark.parametrize(
    ("kernel_rel", "driver_rel"),
    [(case[1], case[2]) for case in RVV_KERNEL_CASES],
    ids=[case[0] for case in RVV_KERNEL_CASES],
)
def test_rvv_kernel_runs_on_spike(spike_runner, tmp_path: pathlib.Path, kernel_rel: str, driver_rel: str) -> None:
    kernel = REPO_ROOT / kernel_rel
    driver = REPO_ROOT / driver_rel
    assert kernel.exists(), f"kernel not found: {kernel}"
    assert driver.exists(), f"driver not found: {driver}"

    out_elf = tmp_path / f"{kernel.stem}_test.elf"
    rc = spike_runner([kernel, driver], out_elf)
    assert rc == 0, (
        f"spike returned {rc} for {kernel.name}; check pytest captured "
        f"stdout/stderr above (should contain a PASS line on success)"
    )
    assert out_elf.exists()
