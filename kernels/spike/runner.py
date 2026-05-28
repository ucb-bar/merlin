"""Compile + run a RISC-V kernel under Spike.

Tightens the dev loop for RVV kernel authoring: write `kernel.c` (with RVV
intrinsics or scalar code), drop a small `driver.c` that calls into it, and
this runner builds a static ELF with the chipyard riscv-tools GCC, then
executes it via `spike pk`. Used standalone for verifying kernel correctness
before plumbing into IREE's custom dispatch pipeline.

The same kernel source can later be passed to
`kernels/core/precompile.py` with `source_lang: c` and a `riscv64` target
to produce an `.o` for `hal.executable.objects` — this script just gates on
"does the kernel run correctly under the reference simulator".

Usage as a module:
    from tools.kernels.spike_runner import build, run, build_and_run
    build_and_run([kernel_c, driver_c], out_elf="rvv_add_test.elf")

Usage from CLI:
    python -m tools.kernels.spike_runner \
        --kernel samples/research/rvv_kernels_on_spike/src/rvv_add.c \
        --driver samples/research/rvv_kernels_on_spike/driver/driver.c \
        --out /tmp/rvv_add_test.elf
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import subprocess
import sys

_LOG = logging.getLogger("spike_runner")

# Toolchain root inside chipyard's conda env. Override via env or arg if the
# user has a different layout.
_CHIPYARD_RISCV_BIN = pathlib.Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin")
_DEFAULT_GCC = _CHIPYARD_RISCV_BIN / "riscv64-unknown-elf-gcc"
_DEFAULT_SPIKE = _CHIPYARD_RISCV_BIN / "spike"
_DEFAULT_PK = pathlib.Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk")


def build(
    sources: list[pathlib.Path],
    out_elf: pathlib.Path,
    *,
    march: str = "rv64gcv",
    mabi: str = "lp64d",
    extra_flags: list[str] | None = None,
    gcc: pathlib.Path = _DEFAULT_GCC,
) -> None:
    """Statically link `sources` into a Spike-runnable RISC-V ELF.

    `march` defaults to `rv64gcv` (full V extension); `mabi` to `lp64d`. The
    resulting binary uses newlib + libgloss for printf etc., which lets the
    proxy kernel `pk` service syscalls under Spike.
    """
    cmd = [str(gcc), "-O3", f"-march={march}", f"-mabi={mabi}"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_elf)])
    _LOG.info("build: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"riscv64-gcc build failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def run(
    elf: pathlib.Path,
    *,
    isa: str = "rv64gcv",
    spike: pathlib.Path = _DEFAULT_SPIKE,
    pk: pathlib.Path = _DEFAULT_PK,
    timeout_s: float = 60.0,
    spike_extra: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run `elf` under Spike+pk. Returns (rc, stdout, stderr).

    `spike_extra` is appended to the spike argv before the pk path, so it can
    carry flags like `--extension=gemmini` or `-m4096` that this wrapper
    otherwise wouldn't know about.
    `extra_env` is merged into `os.environ` for the subprocess; useful for
    `LD_LIBRARY_PATH=$RISCV/lib` to make spike find `libgemmini.so`.
    """
    import os

    cmd = [str(spike), f"--isa={isa}"]
    if spike_extra:
        cmd.extend(spike_extra)
    cmd.extend([str(pk), str(elf)])
    _LOG.info("run: %s", " ".join(cmd))
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s, env=env)
    return res.returncode, res.stdout, res.stderr


def build_and_run(
    sources: list[pathlib.Path],
    out_elf: pathlib.Path,
    *,
    march: str = "rv64gcv",
    mabi: str = "lp64d",
    isa: str | None = None,
) -> int:
    build(sources, out_elf, march=march, mabi=mabi)
    rc, stdout, stderr = run(out_elf, isa=isa or march)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, type=pathlib.Path)
    parser.add_argument("--driver", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--march", default="rv64gcv")
    parser.add_argument("--mabi", default="lp64d")
    parser.add_argument("--isa", default=None, help="Spike --isa=. Defaults to `march`.")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    sources = [args.kernel, args.driver]
    if args.build_only:
        build(sources, args.out, march=args.march, mabi=args.mabi)
        return 0
    return build_and_run(sources, args.out, march=args.march, mabi=args.mabi, isa=args.isa)


if __name__ == "__main__":
    sys.exit(main())
