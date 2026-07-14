"""Measure the EXPERT RVV GEMM ceiling (XNNPACK + OpenBLAS) on spike, standalone.

This is the S4.2 ceiling for the f32 64x64x64 GEMM, the *expert* bar our compiler's
RVV codegen is ranked against. Each expert kernel is compiled into a tiny bare-metal
ELF (its own ``main`` + the saturn HTIF console/crt) that:

  1. inits A/B (and bias for XNNPACK) and computes a scalar reference,
  2. PRE-PACKS the operands (ncopy/tcopy for OpenBLAS; goi weight pack for XNNPACK)
     OUTSIDE the timed region,
  3. wraps ONLY the kernel compute call(s) in ``read_csr(mcycle)`` (mode = inner_compute),
  4. verifies the result against the reference and prints ``CYCLES <n>``.

This is the apples-to-apples to our spike cycle proxy (cycle_accurate=false): we time
the emitted *compute*, not the one-time pack/setup. Honest by construction: a build or
run failure, or a VERIFY FAIL, yields a ``not_run`` row with the blocker — never a
fabricated cycle number.

Run:  ``.venv/bin/python -m merlin.kernels.ceiling_drivers.run_expert_gemm``
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from merlin.common.paths import work_dir

from ...common.driver_output import int_after as _int_after

from ...common.paths import repo_root
from .. import bench_ceiling

HERE = Path(__file__).resolve().parent

# Saturn harness flags (mirrors bench_ceiling._SATURN_CFLAGS) + NDEBUG so the
# XNNPACK microkernel's assert() preconditions compile out under -nostdlib.
_CFLAGS = (
    "-DNDEBUG", "-DPREALLOCATE=1", "-mcmodel=medany", "-static", "-O3", "-g",
    "-ffast-math", "-fno-common", "-fno-builtin-printf",
    "-fno-tree-loop-distribute-patterns",
    "-march=rv64gcv_zfh_zvfh", "-mabi=lp64d", "-std=gnu99",
)
_LINK = ("-static", "-nostdlib", "-nostartfiles", "-lm", "-lgcc")


def _tmp_kernels() -> Path:
    return work_dir() / "tmp" / "kernels"


def _experts() -> dict:
    """source -> (driver.c, [extra include roots holding the kernel src], kernel_file note)."""
    xnn = _tmp_kernels() / "XNNPACK" / "src"
    oblas = _tmp_kernels() / "OpenBLAS" / "kernel" / "riscv64"
    return {
        "openblas": dict(
            driver=HERE / "openblas_sgemm_driver.c",
            incs=[HERE, oblas],   # HERE supplies the common.h shim
            kernel_file="tmp/kernels/OpenBLAS/kernel/riscv64/sgemm_kernel_8x8_zvl128b.c",
            dtype="f32",
        ),
        "xnnpack": dict(
            driver=HERE / "xnnpack_gemm_driver.c",
            incs=[HERE, xnn],     # HERE supplies src/xnnpack/gemm.h shim, XNN the kernel src
            kernel_file="tmp/kernels/XNNPACK/src/f32-gemm/gen/f32-gemm-1x4v-rvv.c",
            dtype="f32",
        ),
    }


def _build(driver: Path, incs: list[Path], out: Path, *, timeout: int = 300) -> str | None:
    """Build one expert driver ELF; return None on success, else the error text (blocker)."""
    from ...runtime.backends import spike
    gcc = spike.gcc_path()
    sat = bench_ceiling.build_asm.saturn_root() / "benchmarks"
    enc = bench_ceiling._encoding_include_dir()
    if enc is None:
        return "encoding.h not found (set MERLIN_CHIPYARD)"
    inc_flags: list[str] = []
    for d in incs:
        inc_flags += ["-I", str(d)]
    inc_flags += ["-I", str(sat / "env"), "-I", str(sat / "common"), "-I", str(enc)]
    cmd = [str(gcc), *inc_flags, *_CFLAGS, "-o", str(out), str(driver),
           str(sat / "common" / "syscalls.c"), str(sat / "common" / "crt.S"),
           *_LINK, "-T", str(sat / "common" / "test.ld")]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:  # noqa: BLE001
        return f"build exec failed: {e}"
    if p.returncode != 0 or not out.is_file():
        return f"build failed (rc={p.returncode}): {p.stderr.strip()[-800:]}"
    return None


def _run(elf: Path, *, isa: str = bench_ceiling.DEFAULT_ISA, timeout: int = 300) -> str | None:
    from ...runtime.backends import spike
    cmd = [str(spike.spike_path()), f"--isa={isa}", "-p1",
           bench_ceiling.SPIKE_MEM, str(elf)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return p.stdout if p.returncode == 0 else None


def measure_one(source: str, *, M: int = 64, N: int = 64, K: int = 64,
                isa: str = bench_ceiling.DEFAULT_ISA) -> dict:
    """Build+run one expert kernel; return a ceiling row (cycles set) or a not_run row."""
    spec = _experts()[source]
    regime = bench_ceiling.shape_regime("matmul", M, N, K)
    base = {
        "op": "matmul", "dtype": spec["dtype"], "M": M, "N": N, "K": K,
        "shape_regime": regime, "source": source, "target": "spike",
        "mode": "inner_compute", "isa": isa, "kernel_file": spec["kernel_file"],
        "fingerprint_key": bench_ceiling.fingerprint_key("matmul", spec["dtype"], regime),
    }
    with tempfile.TemporaryDirectory(prefix="merlin_expert_") as tmp:
        elf = Path(tmp) / f"{source}_gemm.riscv"
        err = _build(spec["driver"], spec["incs"], elf, )
        if err is not None:
            return {**base, "cycles": None, "status": "not_run", "blocker": err}
        console = _run(elf, isa=isa)
    if console is None:
        return {**base, "cycles": None, "status": "not_run", "blocker": "spike run failed/empty"}
    if "VERIFY PASS" not in console:
        return {**base, "cycles": None, "status": "not_run",
                "blocker": f"verify did not pass; console tail: {console.strip()[-300:]}"}
    cycles = _int_after(console, "CYCLES")
    instret = _int_after(console, "INSTRET")
    if cycles is None:
        return {**base, "cycles": None, "status": "not_run",
                "blocker": "no CYCLES line in console"}
    row = {**base, "cycles": cycles, "status": "pass",
           "note": f"{source} f32 GEMM on spike; inner-compute timed (pack outside); "
                   f"verified vs scalar ref"}
    if instret is not None:
        row["instructions"] = instret
    return row


def main() -> int:
    from ...runtime.backends import spike
    if not spike.available():
        print("run_expert_gemm: spike/riscv-gcc unavailable; cannot measure ceiling.")
        return 2
    out_path = repo_root() / bench_ceiling.DEFAULT_CEILING_PATH
    for source in ("openblas", "xnnpack"):
        row = measure_one(source)
        bench_ceiling.append_ceiling(row, out_path)
        status = row.get("status")
        if status == "pass":
            print(f"{source:9s}  cycles={row['cycles']:>8}  "
                  f"instret={row.get('instructions','?')}  -> appended")
        else:
            print(f"{source:9s}  NOT_RUN: {row.get('blocker')}")
    print(f"ceiling -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
