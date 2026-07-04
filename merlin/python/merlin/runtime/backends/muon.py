"""Run Merlin command buffers on the Muon SIMT core, via the radiance-kernels + cyclotron flow.

This is the Muon analog of :mod:`merlin.runtime.backends.gemmini`. The frozen Gemmini path is NOT
touched; this is a parallel backend so the rig scales to a second target (Muon / RadianceMuonConfig).

Pipeline: an agent (or the reference backend) emits a SIMT C++ kernel that prints its result tensors
with the same ``OUT/METRIC/DONE`` console protocol the Gemmini harness uses -> compile with the Muon
clang fork (``clang-muon``, a Vortex/RISC-V32 target) against the radiance-kernels runtime
(``libmuonrt.a`` + ``tohost.S`` + ``mu_link.ld``) -> run on an oracle:

  - ``cyclotron``  : Rust performance model (``--timing``), **primary** perf + functional oracle
                     (derived_from_rtl=False). Cycles from its ``simulation finished after N cycles``
                     stdout line; per-component stats from ``performance_logs/run_*/summary.json``.
  - ``vcs``        : the prebuilt RadianceMuonConfig RTL sim with the cyclotron DPI difftest
                     (derived_from_rtl=True) -- **certification** when available. Kernel-level
                     difftest is WIP upstream (stalls at ``nu.invoke``); this fails closed
                     (``MuonUnavailable``) rather than silently passing.

-> parse OUT/METRIC/DONE -> gate the outputs against
:func:`merlin.runtime.reference.reference_outputs` (the same oracle the Python simulator backend is
held to). cyclotron and VCS run the *exact same ELF*.

Toolchain resolution via environment (all have working-tree defaults):
``MERLIN_CHIPYARD`` (default ``/scratch/agustin/projects/chipyard``),
``MERLIN_RADIANCE_KERNELS`` (default ``/scratch2/agustin/radiance-kernels``),
``MERLIN_MUON_CLANG`` / ``MERLIN_MUON_CYCLOTRON`` / ``MERLIN_MUON_CONFIG`` / ``MERLIN_MUON_VCS``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..metrics import COMMON_METRIC_NAMES
from ..reference import outputs_match, reference_outputs

DEFAULT_CHIPYARD = "/scratch/agustin/projects/chipyard"
DEFAULT_RADIANCE_KERNELS = "/scratch2/agustin/radiance-kernels"
DEFAULT_CONFIG = "/scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml"
VCS_CONFIG = "RadianceMuonConfig"

# Muon SIMT FP peak (RadianceMuonConfig, config_muon.toml): 2 cores x 16 lanes x 2 flop/FMA = 64
# flop/cycle; at 500 MHz that is 32 GFLOP/s. Reported conservatively as a denominator for utilization.
FP_PEAK_FLOPS_PER_CYCLE = 64
CLOCK_HZ = 500_000_000
FP_PEAK_GFLOPS = FP_PEAK_FLOPS_PER_CYCLE * CLOCK_HZ / 1e9  # 32.0

ORACLE = {
    "cyclotron": {"kind": "cyclotron_perf_model", "derived_from_rtl": False},
    "vcs": {"kind": "rtl_vcs_muon_difftest", "derived_from_rtl": True},
}


class MuonError(RuntimeError):
    pass


class MuonUnavailable(RuntimeError):
    """The requested oracle exists but cannot complete (fail-closed; never a silent pass)."""


# --- toolchain resolution -------------------------------------------------------------------------
def chipyard_root() -> Path:
    return Path(os.environ.get("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def radiance_kernels_root() -> Path:
    return Path(os.environ.get("MERLIN_RADIANCE_KERNELS", DEFAULT_RADIANCE_KERNELS))


def llvm_muon_root() -> Path:
    return radiance_kernels_root() / "llvm/llvm-muon"


def clang_path() -> Path:
    env = os.environ.get("MERLIN_MUON_CLANG")
    if env:
        return Path(env)
    return llvm_muon_root() / "bin/clang++"


def lib_dir() -> Path:
    return radiance_kernels_root() / "lib"


def cyclotron_path() -> Path:
    env = os.environ.get("MERLIN_MUON_CYCLOTRON")
    if env:
        return Path(env)
    return chipyard_root() / "generators/radiance/cyclotron/target/release/cyclotron"


def cyclotron_root() -> Path:
    return chipyard_root() / "generators/radiance/cyclotron"


def config_path() -> Path:
    return Path(os.environ.get("MERLIN_MUON_CONFIG", DEFAULT_CONFIG))


def vcs_path() -> Path:
    env = os.environ.get("MERLIN_MUON_VCS")
    if env:
        return Path(env)
    return chipyard_root() / "sims/vcs" / f"simv-chipyard.harness-{VCS_CONFIG}"


def available(simulator: str = "cyclotron") -> bool:
    """True when clang-muon + the runtime + the requested simulator are all present."""
    lib = lib_dir()
    base = (clang_path().is_file()
            and (lib / "libmuonrt.a").is_file()
            and (lib / "tohost.S").is_file()
            and (lib / "linker/mu_link.ld").is_file()
            and (lib / "include").is_dir())
    if simulator == "cyclotron":
        return base and cyclotron_path().is_file() and config_path().is_file()
    if simulator == "vcs":
        return base and vcs_path().is_file()
    raise MuonError(f"unknown simulator {simulator!r}")


# --- compile --------------------------------------------------------------------------------------
def _mu_cflags() -> list[str]:
    """Mirror radiance-kernels/kernels/common.mk MU_CFLAGS exactly (flags AND order matter)."""
    llvm = llvm_muon_root()
    inc = lib_dir() / "include"
    gemmini_sw = lib_dir() / "mxgemmini"
    return [
        f"--sysroot={llvm}",
        "-Xclang", "-target-feature", "-Xclang", "+vortex",
        "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
        "-O3", "-std=c++20",
        "-mcmodel=medany", "-fno-rtti", "-fno-exceptions",
        "-fdata-sections", "-ffunction-sections",
        "-mllvm", "-inline-threshold=262144",
        "-I", str(inc), "-I", str(gemmini_sw),
        "-DRADIANCE", "-DRADIANCE_DEVICE", "-DNDEBUG", "-DLLVM_VORTEX",
    ]


def _mu_ldflags() -> list[str]:
    """Mirror common.mk MU_LDFLAGS exactly."""
    lib = lib_dir()
    return [
        "-nodefaultlibs", "-nostartfiles",
        f"-Wl,-Bstatic,-T,{lib / 'linker/mu_link.ld'},-z,norelro",
        "-fuse-ld=lld",
        str(lib / "libmuonrt.a"), str(lib / "tohost.S"),
    ]


def _print_runtime_sources() -> list[str]:
    """The console runtime the OUT/DONE protocol needs.

    ``vx_printf``/``vx_putfloat`` (lib/src/vx_print.c) and ``vx_putchar`` (lib/src/vx_print.S, which
    pokes the IO_COUT_ADDR MMIO that cyclotron mirrors to stdout) are NOT in libmuonrt.a -- the
    golden kernels are silent and signal only via the tohost exit code. We compile+link them so a
    kernel can print its result tensors for output gating.
    """
    src = lib_dir() / "src"
    # Only vx_print.S (assembly -> vx_putchar, pokes IO_COUT_ADDR). We deliberately avoid vx_print.c
    # / tiny_printf, which pull a broken libc++ <math.h>; kernels print via the vx_putchar-based
    # MUON_CONSOLE helper below instead (no libc, no printf, no math).
    return [str(src / "vx_print.S")] if (src / "vx_print.S").is_file() else []


# A self-contained device console built on vx_putchar (the only console primitive that links without
# libc). Prepend MUON_CONSOLE to a kernel and call mu_out_i32 / mu_out_f32 / mu_done to emit the
# OUT/METRIC/DONE protocol :func:`parse_output` consumes. fp32 is printed as a decimal with a fixed
# number of fractional digits (exact enough for the corpus' integral-or-simple golden values).
MUON_CONSOLE = r"""
#include <vx_intrinsics.h>
extern "C" void vx_putchar(int c);
// The ELF starts on every core's hart 0 (SPMD bring-up); print from ONE hart only, else the two
// cores' vx_putchar byte streams interleave ("O2UT..."). Guard every emit with mu_is_print_hart().
static inline bool mu_is_print_hart() { return vx_core_id() == 0; }
namespace muon_console {
static inline void puts_(const char* s) { while (*s) vx_putchar(*s++); }
static inline void put_u32(unsigned v) {
  char b[12]; int n = 0;
  if (!v) { vx_putchar('0'); return; }
  while (v) { b[n++] = '0' + (v % 10u); v /= 10u; }
  while (n) vx_putchar(b[--n]);
}
static inline void put_i32(int v) { if (v < 0) { vx_putchar('-'); put_u32((unsigned)(-v)); } else put_u32((unsigned)v); }
static inline void put_f32(float f, int frac = 6) {
  if (f != f) { puts_("nan"); return; }
  if (f < 0) { vx_putchar('-'); f = -f; }
  unsigned ip = (unsigned)f; float rem = f - (float)ip;
  put_u32(ip); vx_putchar('.');
  for (int i = 0; i < frac; i++) { rem *= 10.0f; unsigned d = (unsigned)rem; vx_putchar('0' + (d % 10u)); rem -= (float)d; }
}
}  // namespace muon_console
static inline void mu_out_i32(const char* name, int rows, int cols, const int* data) {
  using namespace muon_console;
  puts_("OUT "); puts_(name); vx_putchar(' '); put_u32(rows); vx_putchar(' '); put_u32(cols);
  for (int i = 0; i < rows * cols; i++) { vx_putchar(' '); put_i32(data[i]); }
  vx_putchar('\n');
}
static inline void mu_out_f32(const char* name, int rows, int cols, const float* data) {
  using namespace muon_console;
  puts_("OUT "); puts_(name); vx_putchar(' '); put_u32(rows); vx_putchar(' '); put_u32(cols);
  for (int i = 0; i < rows * cols; i++) { vx_putchar(' '); put_f32(data[i]); }
  vx_putchar('\n');
}
static inline void mu_metric(const char* name, int v) {
  using namespace muon_console;
  puts_("METRIC "); puts_(name); vx_putchar(' '); put_i32(v); vx_putchar('\n');
}
static inline void mu_done() { muon_console::puts_("DONE\n"); }
"""


def compile_kernel(kernel_src: str, workdir: str | Path) -> Path:
    """Compile a SIMT C++ kernel source string into ``kernel.radiance.elf``; return its path.

    The kernel must define ``int main()`` that launches via ``mu_schedule(...)`` and prints its
    result tensors with the OUT/METRIC/DONE protocol (see :func:`parse_output`).
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    src = work / "kernel.cpp"
    src.write_text(kernel_src, encoding="utf-8")
    elf = work / "kernel.radiance.elf"
    cmd = [str(clang_path()), *_mu_cflags(), str(src), *_print_runtime_sources(),
           *_mu_ldflags(), "-o", str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MuonError(f"clang-muon failed:\n{' '.join(cmd)}\n{proc.stderr[-3000:]}")
    return elf


# --- run ------------------------------------------------------------------------------------------
_CYCLES_RE = re.compile(r"finished after (\d+) cycles")


def _run_cyclotron(elf: Path, timeout: int) -> tuple[str, int | None, dict | None]:
    """Run cyclotron --timing on the ELF; return (console, cycles, summary_json|None).

    cyclotron resolves the config's relative ``[timing] include = ["config/timing/..."]`` paths
    against its CWD, so we run inside a per-run workdir that symlinks ``config`` -> the cyclotron
    config dir (keeps the chipyard tree clean; perf logs land in the workdir).
    """
    work = elf.parent
    cfg_link = work / "config"
    if not cfg_link.exists():
        try:
            cfg_link.symlink_to(cyclotron_root() / "config")
        except OSError:
            pass
    env = dict(os.environ)
    env["RUST_LOG"] = "error"
    cmd = [str(cyclotron_path()), str(config_path()),
           "--binary-path", str(elf), "--timing", "--log", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(work), env=env)
    console = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise MuonError(f"cyclotron exited {proc.returncode}:\n{console[-2000:]}")
    m = _CYCLES_RE.search(console)
    cycles = int(m.group(1)) if m else None
    summary = None
    runs = sorted((work / "performance_logs").glob("run_*"), key=lambda p: p.name)
    if runs:
        sj = runs[-1] / "summary.json"
        if sj.is_file():
            try:
                summary = json.loads(sj.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                summary = None
    return console, cycles, summary


def _run_vcs(elf: Path, timeout: int) -> tuple[str, int | None]:
    """Run the RadianceMuonConfig VCS RTL sim (cyclotron DPI difftest).

    The full kernel-level lockstep difftest is WIP upstream (stalls at the neutrino ``nu.invoke``
    launch; the DPI ELF path is not wired for arbitrary corpus kernels yet). Until it certifies a
    full kernel cleanly this tier is **honest-unavailable** -- it never reports a pass it cannot
    stand behind, and a load/launch failure is reported as unavailable, not as a kernel ``fail``.
    """
    cmd = [str(vcs_path()), f"+binary={elf}", "+permissive-off"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(elf.parent))
    except subprocess.TimeoutExpired as e:
        raise MuonUnavailable(f"vcs RTL difftest timed out after {timeout}s "
                              f"(kernel-launch difftest is WIP)") from e
    console = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    m = _CYCLES_RE.search(console)
    if proc.returncode != 0 or m is None:
        raise MuonUnavailable(
            f"vcs RTL difftest did not complete (rc={proc.returncode}); kernel-launch difftest is "
            f"WIP. tail:\n{console[-400:]}")
    return console, int(m.group(1))


def run_elf(elf: str | Path, simulator: str = "cyclotron",
            timeout: int = 600) -> tuple[str, int | None, dict | None]:
    """Run the ELF on the chosen oracle; return (console, cycles, summary_json|None)."""
    elf = Path(elf)
    if simulator == "cyclotron":
        return _run_cyclotron(elf, timeout)
    if simulator == "vcs":
        console, cycles = _run_vcs(elf, timeout)
        return console, cycles, None
    raise MuonError(f"unknown simulator {simulator!r}")


def parse_output(text: str, cycles: int | None) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the device OUT/METRIC/DONE console into (outputs, raw metrics).

    Protocol (printed by the kernel via vx_printf, identical to the Gemmini harness):
        OUT <name> <rows> <cols> <v0> <v1> ... <v(rows*cols-1)>
        METRIC <name> <int>            (optional; cycles come from cyclotron itself)
        DONE
    The cyclotron ``cycles`` (from its own stdout line) is injected as raw["cycles"].
    """
    outputs: dict[str, list] = {}
    raw: dict[str, int] = {}
    done = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "OUT":
            name, rows, cols = parts[1], int(parts[2]), int(parts[3])
            vals = [_num(v) for v in parts[4:]]
            if len(vals) != rows * cols:
                raise MuonError(f"OUT {name}: expected {rows * cols} values, got {len(vals)}")
            outputs[name] = [vals[r * cols:(r + 1) * cols] for r in range(rows)]
        elif parts[0] == "METRIC":
            try:
                raw[parts[1]] = int(parts[2])
            except (IndexError, ValueError):
                pass
        elif parts[0] == "DONE":
            done = True
    if cycles is not None:
        raw["cycles"] = cycles
    if not done:
        raise MuonError(f"run did not reach DONE; output was:\n{text[:2000]}")
    return outputs, raw


def _num(tok: str) -> Any:
    """Parse an OUT value as int when integral, else float (Muon corpus is fp32)."""
    try:
        return int(tok)
    except ValueError:
        return float(tok)


def gflops(flops: int | None, cycles: int | None) -> float | None:
    """Achieved GFLOP/s = flops / (cycles / CLOCK_HZ) / 1e9."""
    if not flops or not cycles:
        return None
    return flops * CLOCK_HZ / cycles / 1e9


def pct_fp_peak(flops: int | None, cycles: int | None) -> float | None:
    g = gflops(flops, cycles)
    return None if g is None else round(100.0 * g / FP_PEAK_GFLOPS, 2)


def _metrics(raw: dict[str, int], simulator: str, summary: dict | None) -> dict[str, Any]:
    metrics = {name: int(raw.get(name, 0)) for name in COMMON_METRIC_NAMES}
    metrics["cycles"] = int(raw.get("cycles", 0))
    metrics["cycle_source"] = ("cyclotron_timing" if simulator == "cyclotron"
                               else "rtl_vcs" if simulator == "vcs" else "unknown")
    metrics["memory_model"] = "perf_model" if simulator == "cyclotron" else "rtl"
    if summary is not None:
        metrics["summary"] = summary
    return metrics


def run_kernel(kernel_src: str, cb: dict[str, Any], *, workdir: str | Path | None = None,
               simulator: str = "cyclotron", timeout: int = 600,
               flops: int | None = None) -> dict[str, Any]:
    """Compile + run a SIMT kernel on Muon and gate its outputs on reference equality.

    Returns {outputs, metrics, raw_metrics, correct, oracle, elf, console, cycles, gflops,
    pct_fp_peak}. ``cb`` is the command buffer used only to compute the reference outputs and FLOPs.
    """
    if not available(simulator):
        raise MuonUnavailable(f"muon {simulator} oracle not available "
                              f"(set MERLIN_CHIPYARD / MERLIN_RADIANCE_KERNELS)")
    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="merlin_muon_")) if own_tmp else Path(workdir)
    elf = compile_kernel(kernel_src, work)
    console, cycles, summary = run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = parse_output(console, cycles)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": _metrics(raw, simulator, summary),
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "oracle": dict(ORACLE[simulator]),
        "elf": str(elf),
        "console": console,
        "cycles": cycles,
        "gflops": gflops(flops, cycles),
        "pct_fp_peak": pct_fp_peak(flops, cycles),
    }
