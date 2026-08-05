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

Toolchain resolution honors the process env AND the repo ``.env`` (via ``merlin.common.paths.env``),
mirroring the gemmini backend — ``os.environ.get`` alone missed keys that live in ``.env`` (the
repo-wide contract), leaving the ``/path/to/...`` placeholders and making ``available()`` False even
with a real toolchain. Keys: ``MERLIN_CHIPYARD`` (default ``/path/to/chipyard``),
``MERLIN_RADIANCE_KERNELS`` (default ``/path/to/radiance-kernels``),
``MERLIN_MUON_CLANG`` / ``MERLIN_MUON_CYCLOTRON`` / ``MERLIN_MUON_CONFIG`` / ``MERLIN_MUON_VCS``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ...common.paths import env as _env
from ..metrics import COMMON_METRIC_NAMES
from ..reference import outputs_match, reference_outputs
from .base import BackendInfo, BackendKind, TargetClass, register

# Self-register this reference GPU/SIMT backend with the class registry (base._REGISTRY). Discovery in
# base._ensure_discovered imports this module to run the call, so the core carries no name -> module
# map for the accelerator; the identity lives with the backend that owns it.
register(BackendInfo("muon", TargetClass.GPU, BackendKind.KERNEL, __name__))

DEFAULT_CHIPYARD = "/path/to/chipyard"
DEFAULT_RADIANCE_KERNELS = "/path/to/radiance-kernels"
DEFAULT_CONFIG = "/path/to/autocomp/scripts/muon/config_muon.toml"
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
    return Path(_env("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def radiance_kernels_root() -> Path:
    return Path(_env("MERLIN_RADIANCE_KERNELS", DEFAULT_RADIANCE_KERNELS))


def llvm_muon_root() -> Path:
    return radiance_kernels_root() / "llvm/llvm-muon"


def clang_path() -> Path:
    val = _env("MERLIN_MUON_CLANG")
    if val:
        return Path(val)
    return llvm_muon_root() / "bin/clang++"


def lib_dir() -> Path:
    return radiance_kernels_root() / "lib"


def cyclotron_path() -> Path:
    val = _env("MERLIN_MUON_CYCLOTRON")
    if val:
        return Path(val)
    return chipyard_root() / "generators/radiance/cyclotron/target/release/cyclotron"


def cyclotron_root() -> Path:
    return chipyard_root() / "generators/radiance/cyclotron"


def config_path() -> Path:
    return Path(_env("MERLIN_MUON_CONFIG", DEFAULT_CONFIG))


def vcs_path() -> Path:
    val = _env("MERLIN_MUON_VCS")
    if val:
        return Path(val)
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


# The target's assembler shim: its SIMT pseudo-mnemonics as explicit .insn CUSTOM0 forms + the warp-count
# CSR name, prepended so a STOCK assembler accepts the boot source (mu_start.S #defines RISCV_CUSTOM0 itself
# via the C preprocessor; only the vx_* macros + the `nw` CSR need supplying). A muon-target BSP asset, kept
# in this per-target edge module.
_FORKFREE_BSP_PREAMBLE = (
    ".set nw, 0xfc1\n"
    ".macro vx_tmc reg\n.insn r 0x0b, 0, 0, x0, \\reg, x0\n.endm\n"
    ".macro vx_wspawn n, f\n.insn r 0x0b, 1, 0, x0, \\n, \\f\n.endm\n"
)


def build_forkfree_bsp(workdir: str | Path, *, target: str = "radiance", num_warps: int = 1) -> list[Path]:
    """Reproducibly build the fork-free BSP (boot + runtime shims) from the target's shipped sources — no
    transient or committed binaries. Returns the object list :func:`compile_kernel_forkfree` links."""
    from . import muon_bsp
    from ...targetgen.contract.toolchain import mlir_bin
    lib = lib_dir()
    return muon_bsp.build_bsp(lib / "src/mu_start.S", lib / "tohost.S", Path(workdir) / "bsp",
                              target=target, clang=str(mlir_bin("clang")), mc=str(mlir_bin("llvm-mc")),
                              asm_preamble=_FORKFREE_BSP_PREAMBLE, num_warps=num_warps)


def compile_kernel_forkfree(kernel_c: str, workdir: str | Path, bsp_objs: list[str | Path] | None = None,
                            *, target: str = "radiance", march: str | None = None, num_warps: int = 1) -> Path:
    """FORK-FREE build of a self-contained kernel: STOCK clang compiles the kernel C to rv32, the derived
    transcoder re-maps it into the target's fixed-format words, STOCK llvm-mc assembles them, and the
    fork-free linker (:func:`muon_link.link_fork_free`) links the vendored boot/runtime objects (``bsp_objs``
    — the crt0-like boot that uses the target's own SIMT instructions, a shipped runtime asset) + the kernel.
    No clang-muon in compile/assemble/link — every field position is DERIVED from the target's RTL encoding.
    The kernel must have NO relocations (self-contained; it signals results via inline MMIO), else it fails
    closed (a relocation cannot be a pure field re-map). Returns the ELF path."""
    from ...targetgen.isa_model import isa_model_from_encoding
    from ...targetgen.isa_transcode import FixedFormatTranscoder, to_data_lines, emit_kernel_asm, derive_march
    from ...targetgen.rtl import mlc_bridge
    from ...targetgen.contract.toolchain import mlir_bin
    from . import muon_link

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    clang, mc, objcopy, objdump = (mlir_bin("clang"), mlir_bin("llvm-mc"),
                                   mlir_bin("llvm-objcopy"), mlir_bin("llvm-objdump"))
    for t in (clang, mc, objcopy, objdump):
        if not t.is_file():
            raise MuonUnavailable(f"stock LLVM tool absent: {t} (set MERLIN_MLIR_INSTALL)")
    fact = mlc_bridge.isa_encoding_for(target)
    if not fact:
        raise MuonUnavailable(f"no derived ISA encoding fact for target {target!r}")
    model = isa_model_from_encoding(target, fact)
    # Derive the compile -march (incl. the FP mode: zfinx vs FP-regs) from the target's own opcode table
    # when the caller did not pin one, so a float kernel compiles to the target's actual FP encoding.
    if march is None:
        march = derive_march(model)

    # 1. STOCK compile -> rv32 object (self-contained: no PIC, no jump tables, no relaxation).
    src = work / "kernel.c"
    src.write_text(kernel_c, encoding="utf-8")
    obj = work / "kernel.o"
    cc = subprocess.run([str(clang), "--target=riscv32", f"-march={march}", "-mabi=ilp32", "-mno-relax",
                         "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-pic", "-fno-jump-tables",
                         "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
    if cc.returncode != 0:
        raise MuonError(f"stock rv32 compile failed:\n{cc.stderr[-2000:]}")
    if any("R_RISCV" in ln for ln in
           subprocess.run([str(objdump), "-r", str(obj)], capture_output=True, text=True).stdout.splitlines()):
        raise MuonError("kernel has relocations; the transcoder needs a self-contained kernel (fail closed)")

    # 2. transcode the rv32 .text -> target fixed-format words -> assembly.
    binf = work / "kernel.text.bin"
    subprocess.run([str(objcopy), "-O", "binary", "--only-section=.text", str(obj), str(binf)],
                   capture_output=True)
    words = FixedFormatTranscoder(model).transcode_text(binf.read_bytes())
    ks = work / "kernel_muon.S"
    ks.write_text(emit_kernel_asm(words, model.inst_width))

    # 3. STOCK assemble.
    kobj = work / "kernel_muon.o"
    a = subprocess.run([str(mc), "--triple=riscv32", "--filetype=obj", str(ks), "-o", str(kobj)],
                       capture_output=True, text=True)
    if a.returncode != 0:
        raise MuonError(f"stock llvm-mc failed:\n{a.stderr[-1500:]}")

    # 4. FORK-FREE link: boot/runtime (regenerated fork-free from source if not supplied) + the kernel.
    if bsp_objs is None:
        bsp_objs = build_forkfree_bsp(work, target=target, num_warps=num_warps)
    elf = work / "kernel.radiance.elf"
    muon_link.link_fork_free([*[str(o) for o in bsp_objs], str(kobj)],
                             str(lib_dir() / "linker/mu_link.ld"), str(elf), target=target)
    return elf


def compile_for_oracle(kernel_src: str, workdir: str | Path, *, target: str = "radiance") -> tuple[Path, str]:
    """Compile an emitted kernel for the grading oracle, PREFERRING the fork-free thesis path and recording
    which toolchain actually produced the graded ELF. Returns ``(elf, toolchain)`` where ``toolchain`` is
    ``"fork-free"`` (stock LLVM + the RTL-derived transcode — what a real new target has) or
    ``"clang-muon-fork"`` (the vendor fork, EVAL-ONLY: a reference a real new target would NOT have).

    The fork is never a silent fallback — its use is stamped so the experiment MEASURES fork-free coverage
    (how many capsules the thesis path grades on its own). Set ``MERLIN_MUON_FORKFREE_ONLY=1`` for a pure
    thesis run: the fork is refused and a kernel the fork-free path can't yet build (e.g. a multi-warp
    ``mu_schedule`` kernel, until the multi-thread transcode lands) fails closed rather than borrowing it."""
    forkfree_only = _env("MERLIN_MUON_FORKFREE_ONLY", "") not in ("", "0", "false", "False")
    ff_err: Exception | None = None
    try:
        return compile_kernel_forkfree(kernel_src, workdir, target=target), "fork-free"
    except (MuonError, MuonUnavailable) as e:
        ff_err = e
    if forkfree_only:
        raise MuonError(f"MERLIN_MUON_FORKFREE_ONLY set but the fork-free build could not produce this "
                        f"kernel (no fork fallback): {ff_err}")
    # eval-only reference: the vendor fork. Stamped, never hidden.
    return compile_kernel(kernel_src, workdir), "clang-muon-fork"


# --- run ------------------------------------------------------------------------------------------
def _cycles_from_console(console: str) -> int | None:
    """Cycle count from the simulator's ``finished after <N> cycles`` line (cyclotron/VCS), parsed
    structurally by locating the marker and reading the following integer token — no regex."""
    marker = "finished after "
    idx = console.find(marker)
    if idx == -1:
        return None
    tail = console[idx + len(marker):].split(maxsplit=1)
    return int(tail[0]) if tail and tail[0].isdigit() else None


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
    cycles = _cycles_from_console(console)
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
    cycles = _cycles_from_console(console)
    if proc.returncode != 0 or cycles is None:
        raise MuonUnavailable(
            f"vcs RTL difftest did not complete (rc={proc.returncode}); kernel-launch difftest is "
            f"WIP. tail:\n{console[-400:]}")
    return console, cycles


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
