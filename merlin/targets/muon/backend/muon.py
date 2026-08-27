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

from merlin.common.paths import env as _env
from merlin.runtime.metrics import COMMON_METRIC_NAMES
from merlin.runtime.reference import outputs_match, reference_outputs
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register

# Self-register this reference GPU/SIMT backend with the class registry (base._REGISTRY). Discovery in
# base._ensure_discovered imports this module to run the call, so the core carries no name -> module
# map for the accelerator; the identity lives with the backend that owns it.
register(BackendInfo("muon", TargetClass.GPU, BackendKind.KERNEL, __name__))

DEFAULT_CHIPYARD = "/path/to/chipyard"
DEFAULT_RADIANCE_KERNELS = "/path/to/radiance-kernels"
DEFAULT_CONFIG = "/path/to/autocomp/scripts/muon/config_muon.toml"
VCS_CONFIG = "RadianceMuonConfig"
# The Verilator RTL cert runs the RadianceTapeoutSim harness (the one the radiance kernels were
# evaluated on); it is the sim actually built in chipyard, not the WIP VCS RadianceMuonConfig.
VERILATOR_CONFIG = "RadianceTapeoutSimConfig"

# The weak data word this target's own BSP (mu_start.S) declares to carry the launch width; the BSP
# build overrides it with the warp count. Its spelling belongs to that BSP, so it lives here with the
# backend that owns the target rather than in the generic fixed-format builder, which cannot know it.
OCCUPANCY_SYMBOL = "__mu_num_warps"

# Muon SIMT FP peak (RadianceMuonConfig, config_muon.toml): 2 cores x 16 lanes x 2 flop/FMA = 64
# flop/cycle; at 500 MHz that is 32 GFLOP/s. Reported conservatively as a denominator for utilization.
FP_PEAK_FLOPS_PER_CYCLE = 64
CLOCK_HZ = 500_000_000
FP_PEAK_GFLOPS = FP_PEAK_FLOPS_PER_CYCLE * CLOCK_HZ / 1e9  # 32.0

ORACLE = {
    "cyclotron": {"kind": "cyclotron_perf_model", "derived_from_rtl": False},
    "vcs": {"kind": "rtl_vcs_muon_difftest", "derived_from_rtl": True},
    "verilator": {"kind": "rtl_verilator_muon", "derived_from_rtl": True},
    # Bounded-cycle executability SMOKE on the same Verilator RTL build: not a numeric cert (that is
    # ``verilator``'s job, ~45 min/capsule) but a minutes-scale "does a cyclotron(L2)-passing ELF at
    # least RUN on the real RTL — boots, no trap, MX PE accepts a command". Advisory only.
    "verilator_smoke": {"kind": "rtl_verilator_smoke", "derived_from_rtl": True},
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


# --- RTL-derived oracle (ModeLIR / mlc cosim), the sim-independent way to grade a device run ------------
# A real new target ships RTL but NOT a vendor reference sim (cyclotron). The generalizable oracle is the
# model mlc COMPILES from that RTL via CIRCT-arc (arcilator) — mlc's `cosim_<core>` harness — run through the
# mlc venv (mlc is not importable here; same by-path contract as rtl.mlc_bridge). The device kernel computes
# into memory and we read the OUTPUT BUFFER back — no console print (which races across lanes on a SIMT core),
# so this grades a MULTI-WARP run faithfully where the console harness cannot.
def _arc_cosim_assets(target: str) -> tuple[Path, Path, Path] | None:
    """(mlc_venv_python, model.so, state-manifest.json) for a target's RTL-arc cosim model, or None when the
    mlc checkout / compiled model is absent (fail closed — the caller degrades, never fabricates a result)."""
    from merlin.targetgen.rtl import mlc_bridge
    d = mlc_bridge.mlc_dir()
    if d is None:
        return None
    py = d / ".venv/bin/python"
    run = d / "runs" / "circt-arc" / mlc_bridge._arc_target(target) / "outputs"
    so = d / "runs" / "circt-arc" / mlc_bridge._arc_target(target) / "native_run" / "libmuon_model.so"
    manifest = run / "muon_core_state.json"
    return (py, so, manifest) if (py.is_file() and so.is_file() and manifest.is_file()) else None


def arc_oracle_available(target: str = "radiance") -> bool:
    """True when the target's RTL-arc cosim model (the sim-independent oracle) is present."""
    return _arc_cosim_assets(target) is not None


def run_elf_arc(elf: str | Path, *, target: str = "radiance", base: int, length: int,
                max_cycles: int = 120000, timeout: int = 900) -> bytes | None:
    """Run a device ELF on the target's RTL-derived arc model (mlc `cosim_muon`) and read the output buffer
    ``[base, base+length)`` back from memory — the generalizable, vendor-sim-free device oracle. Returns the
    readback bytes, or None when the arc model is absent (fail closed). The heavy arc run happens in the mlc
    venv via a small by-path driver (mlc is not importable in this venv)."""
    assets = _arc_cosim_assets(target)
    if assets is None:
        return None
    py, so, manifest = assets
    out = Path(elf).with_suffix(".arc_readback.bin")
    driver = (
        "import sys, struct\n"
        "from mlc.backends.cosim_muon import MuonCosim, recover_l1_gmem_image\n"
        "so, manifest, elf, base, length, mc, outp = sys.argv[1:8]\n"
        "base=int(base); length=int(length); mc=int(mc)\n"
        "sim=MuonCosim(so, manifest); sim.load(elf); sim.boot(); sim.run(max_cycles=mc)\n"
        "img,_=recover_l1_gmem_image(sim.core, sim.slave, base, length)\n"
        "open(outp,'wb').write(bytes(img))\n"
    )
    drv = out.with_suffix(".driver.py")
    drv.write_text(driver, encoding="utf-8")
    from merlin.targetgen.rtl import mlc_bridge
    proc = subprocess.run([str(py), str(drv), str(so), str(manifest), str(elf),
                           str(base), str(length), str(max_cycles), str(out)],
                          capture_output=True, text=True, timeout=timeout,
                          cwd=str(mlc_bridge.mlc_dir()))     # mlc resolves its discovery caches vs its repo root
    if proc.returncode != 0 or not out.is_file():
        raise MuonError(f"arc cosim run failed (rc {proc.returncode}):\n{(proc.stderr or proc.stdout)[-1500:]}")
    return out.read_bytes()


def config_path() -> Path:
    return Path(_env("MERLIN_MUON_CONFIG", DEFAULT_CONFIG))


def vcs_path() -> Path:
    val = _env("MERLIN_MUON_VCS")
    if val:
        return Path(val)
    return chipyard_root() / "sims/vcs" / f"simv-chipyard.harness-{VCS_CONFIG}"


def radiance_chipyard_root() -> Path:
    """The chipyard checkout that hosts the radiance sims + cyclotron. Cyclotron and the Verilator RTL
    sims live in the SAME chipyard, which may differ from ``MERLIN_CHIPYARD`` (that env can point at a
    sibling checkout used for other targets). Resolve it from the cyclotron binary's ancestry when
    ``MERLIN_MUON_CYCLOTRON`` is set (the nearest ancestor holding both ``sims/`` and ``generators/``),
    else fall back to ``chipyard_root()``."""
    val = _env("MERLIN_MUON_CYCLOTRON")
    if val:
        for anc in Path(val).resolve().parents:
            if (anc / "sims").is_dir() and (anc / "generators").is_dir():
                return anc
    return chipyard_root()


def verilator_path() -> Path:
    """The Verilator RTL sim used as the RTL cert oracle: the ``RadianceTapeoutSimConfig`` harness (the
    open-source cycle-accurate sim the radiance kernels are evaluated on), which lives in the same
    chipyard as cyclotron. ``MERLIN_MUON_VERILATOR`` overrides the full path; ``MERLIN_MUON_VERILATOR_CONFIG``
    overrides just the harness config name."""
    val = _env("MERLIN_MUON_VERILATOR")
    if val:
        return Path(val)
    config = _env("MERLIN_MUON_VERILATOR_CONFIG", VERILATOR_CONFIG)
    return radiance_chipyard_root() / "sims/verilator" / f"simulator-chipyard.harness-{config}"


def verilator_dramsim_ini() -> Path:
    """The testchipip dramsim2_ini dir the RTL harness needs (``+dramsim_ini_dir=``).
    ``MERLIN_MUON_DRAMSIM_INI`` overrides."""
    val = _env("MERLIN_MUON_DRAMSIM_INI")
    if val:
        return Path(val)
    return radiance_chipyard_root() / "generators/testchipip/src/main/resources/dramsim2_ini"


def soc_fuse_dir() -> Path:
    """The radiance-kernels ``soc/`` helper dir (``fuse_rv32_into_rv64.sh`` + ``start.S`` + ``main.c``)
    used to wrap a bare rv32 Muon ELF into the rv64 SoC carrier the RTL harness loads."""
    return radiance_kernels_root() / "soc"


def rv64_cross_prefix() -> str | None:
    """Toolchain prefix (``…/riscv64-unknown-elf``) for the rv64 SoC-carrier fuse. ``MERLIN_MUON_RV64_CROSS``
    overrides; else the chipyard ``riscv-tools`` bin if present; else a bare prefix if it is on PATH.
    Returns None when no rv64 cross-toolchain is found (the caller fails closed)."""
    import shutil
    val = _env("MERLIN_MUON_RV64_CROSS")
    if val:
        return val
    cand = radiance_chipyard_root() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
    if cand.is_file():
        return str(cand.parent / "riscv64-unknown-elf")
    return "riscv64-unknown-elf" if shutil.which("riscv64-unknown-elf-gcc") else None


def console_base(target: str | None = None) -> int | None:
    """The SoC console's base address, from the target's DERIVED facts (``facts.console.base``).

    The device tree's ``stdout-path`` is the only sound source for it: the kernel-side putchar aperture
    (``IO_COUT_ADDR``) is a Vortex constant that cyclotron mirrors and the RTL maps no device at, so
    anything that needs a byte to actually leave an RTL run has to ask the elaborated design where its
    console is. ``None`` when the facts carry no console — callers FAIL CLOSED rather than guess.

    Only a MISSING/ABSENT fact reads as ``None``. An import or parse error is raised: swallowing those
    turns a coding mistake into a silent "this target has no console", which is exactly how the bug this
    exists to fix stayed hidden.
    """
    from merlin.targetgen.rtl import facts as _facts
    try:
        doc = _facts.load_facts(target or _facts_target())
    except FileNotFoundError:
        return None
    con = (doc.get("facts") or {}).get("console") or {}
    base = con.get("base")
    return int(base) if base is not None else None


def _facts_target() -> str:
    """The target name this backend's facts are filed under (``muon_introspect.TARGET``), so the name
    lives in ONE place instead of being restated at every call site."""
    from .muon_introspect import TARGET as _T
    return _T


def fuse_soc_elf(muon_elf: Path, work: Path) -> Path:
    """Fuse a bare rv32 Muon ELF into the rv64 'SoC' carrier ELF the Verilator RTL harness loads
    (radiance-kernels ``soc/fuse_rv32_into_rv64.sh``): the muon PT_LOADs are mirrored at +0x1_0000_0000
    inside a spinning rv64 image, which cyclotron's ``copy_elf`` auto-detects ("CPU-fused ELF") and
    relocates back to run on the cores. Returns the fused ``kernel.soc.elf``. Fails closed
    (``MuonUnavailable``) when the rv64 cross-toolchain or the ``soc/`` helper is absent."""
    cross = rv64_cross_prefix()
    fuse_dir = soc_fuse_dir()
    script, start_s, main_c = (fuse_dir / "fuse_rv32_into_rv64.sh",
                               fuse_dir / "start.S", fuse_dir / "main.c")
    if cross is None:
        raise MuonUnavailable("rv64 SoC-fuse cross-toolchain (riscv64-unknown-elf-*) not found "
                              "(set MERLIN_MUON_RV64_CROSS)")
    if not (script.is_file() and start_s.is_file() and main_c.is_file()):
        raise MuonUnavailable(f"radiance soc/ fuse helper not found under {fuse_dir}")
    out = work / "kernel.soc.elf"
    env = dict(os.environ)
    env.update(CROSS64=cross, RV32_ELF=str(muon_elf), OUT=str(out),
               RV64_START=str(start_s), RV64_MAIN=str(main_c))
    # RUNNER-OWNED CARRIER (opt-in). The stock carrier spins, so nothing on the chip ever drives the
    # console: the kernel's OUT/DONE bytes go to the Vortex IO_COUT aperture, which this SoC maps no
    # device at, and Rocket -- the only master that reaches serial@10020000 -- does nothing. Swapping in
    # our own main.c is what lets a result leave an RTL run at all.
    # Gated on MERLIN_MUON_SOC_CARRIER so a normal grade stays byte-identical until it is asked for, and
    # the UART base is DERIVED from the target's elaborated console fact (never a literal): no fact, no
    # carrier -- fail closed rather than guess an address.
    if os.environ.get("MERLIN_MUON_SOC_CARRIER", "").strip().lower() in ("1", "true", "yes", "on"):
        base = console_base()
        if base is None:
            raise MuonUnavailable(
                "MERLIN_MUON_SOC_CARRIER is set but this target's console fact carries no base "
                "(no elaborated *.dts stdout-path / *.memmap.json) — refusing to guess a UART address")
        carrier = Path(__file__).resolve().parent / "soc_carrier" / "main.c"
        if not carrier.is_file():
            raise MuonUnavailable(f"runner-owned SoC carrier missing at {carrier}")
        env["RV64_MAIN"] = str(carrier)
        env["RV64_CFLAGS"] = (os.environ.get("RV64_CFLAGS")
                              or "-march=rv64gc -mabi=lp64 -ffreestanding -nostdlib -mcmodel=medany") \
            + f" -DMU_UART_BASE=0x{base:x}UL"
    # the fuse script writes start.o/main.o into CWD -> run inside the per-run workdir
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          cwd=str(work), env=env)
    if proc.returncode != 0 or not out.is_file():
        raise MuonError(f"soc fuse failed (rc {proc.returncode}):\n{(proc.stderr or proc.stdout)[-1500:]}")
    return out


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
    if simulator in ("verilator", "verilator_smoke"):
        # The Verilator RTL cert / bounded executability smoke grade a fork-free ELF, so they do NOT need
        # clang-muon (`base`); they need the sim, the testchipip dramsim ini, and the rv64 SoC-fuse
        # toolchain + helper.
        return (verilator_path().is_file()
                and verilator_dramsim_ini().is_dir()
                and rv64_cross_prefix() is not None
                and (soc_fuse_dir() / "fuse_rv32_into_rv64.sh").is_file())
    raise MuonError(f"unknown simulator {simulator!r}")


# --- compile --------------------------------------------------------------------------------------
def _mu_cflags() -> list[str]:
    """Mirror radiance-kernels/kernels/common.mk MU_CFLAGS exactly (flags AND order matter)."""
    llvm = llvm_muon_root()
    inc = lib_dir() / "include"
    gemmini_sw = lib_dir() / "mxgemmini"
    # The MX-Gemmini reference kernels pull in mxgemm_lib.hpp (in lib/mxgemm) which includes gemmini.h,
    # itself pulling <math.h> and calling abs() on an unsigned accumulator. Three additive, kernel-agnostic
    # flags make that legal without disturbing any non-MX kernel: the extra include dir is only searched
    # when a header asks for it; the newlib include is lowest-priority (-idirafter) and only supplies the
    # libc <math.h> the vendor fork's sysroot lacks; the abs shim is a __ASSEMBLER__-guarded no-op that
    # merely disambiguates abs() for the unsigned accumulator type (see gemmini_abs_shim.h).
    mx = lib_dir() / "mxgemm"
    newlib = chipyard_root() / ".conda-env/riscv-tools/riscv64-unknown-elf/include"
    abs_shim = inc / "gemmini_abs_shim.h"
    extra: list[str] = ["-I", str(mx)]
    if newlib.is_dir():
        extra += ["-idirafter", str(newlib)]
    if abs_shim.is_file():
        extra += ["-include", str(abs_shim)]
    return [
        f"--sysroot={llvm}",
        "-Xclang", "-target-feature", "-Xclang", "+vortex",
        "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
        "-O3", "-std=c++20",
        "-mcmodel=medany", "-fno-rtti", "-fno-exceptions",
        "-fdata-sections", "-ffunction-sections",
        "-mllvm", "-inline-threshold=262144",
        "-I", str(inc), "-I", str(gemmini_sw), *extra,
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


def _extra_include_flags(kernel_src: str) -> list[str]:
    """Honor a leading ``// mu-extra-include: <dir>`` directive an emitted kernel may carry, adding
    ``-I <dir>`` for a self-contained kernel that wraps an out-of-tree source tree (the MX flash kernel
    bakes its data inline but still ``#include``s that tree's ``mxgemm_core.hpp`` / ``flash_mx_impl.hpp``).
    Scoped to the emitted TU (only the kernels that declare it get the extra path); parsed structurally
    from the source header, no regex. Multiple directives are allowed."""
    flags: list[str] = []
    key = "// mu-extra-include:"
    for line in kernel_src.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(key):
            inc = stripped[len(key):].strip()
            if inc:
                flags += ["-I", inc]
        elif not stripped.startswith("//") and not stripped.startswith("#define"):
            break  # directives sit in the leading comment/define banner; stop at the first real code
    return flags


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
    cmd = [str(clang_path()), *_mu_cflags(), *_extra_include_flags(kernel_src), str(src),
           *_print_runtime_sources(), *_mu_ldflags(), "-o", str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MuonError(f"clang-muon failed:\n{' '.join(cmd)}\n{proc.stderr[-3000:]}")
    return elf


# The target's assembler shim: its SIMT pseudo-mnemonics as explicit .insn CUSTOM0 forms + the warp-count
# CSR name, prepended so a STOCK assembler accepts the boot source (mu_start.S #defines RISCV_CUSTOM0 itself
# via the C preprocessor; only the vx_* macros + the `nw` CSR need supplying). A muon-target BSP asset, kept
# in this per-target edge module.
def render_simt_preamble(model) -> str:
    """Render the stock-assembler shim the BSP source needs, ENTIRELY from the target's derived ``runtime_abi``
    — the SIMT-control pseudo-mnemonics as explicit ``.insn`` CUSTOM-slot forms (opcode + funct3 from the SFU
    dispatch) plus the num-warps CSR alias the boot references (``csrr t0, nw``). No opcode/funct3/CSR literal:
    every value comes from ``model.runtime_abi`` (fail closed if it is absent — never a hardcoded default)."""
    tmc, wsp = model.sfu_op("tmc"), model.sfu_op("wspawn")
    nw = model.special_csr("num_warps")
    return (f".set nw, {nw:#05x}\n"
            f".macro vx_tmc reg\n.insn r {tmc['opcode']:#04x}, {tmc['funct3']}, 0, x0, \\reg, x0\n.endm\n"
            f".macro vx_wspawn n, f\n.insn r {wsp['opcode']:#04x}, {wsp['funct3']}, 0, x0, \\n, \\f\n.endm\n")


def render_simt_runtime(model, *, num_warps: int, worker_body: str, manager_tail: str = "",
                        globals: str = "") -> str:
    """Render the fork-free MULTI-WARP SIMT scaffold as self-contained C, with every SIMT-control op and CSR
    DERIVED from the target's ``runtime_abi`` — no hardcoded opcode/funct3/CSR number. The proven-correct
    pattern (validated bit-exact on the RTL-arc oracle): warp 0 spawns ``num_warps`` warps at a nullary worker
    entry, each worker computes its own tile (``worker_body``, which may read ``_wid()`` to pick its slice) and
    PARKS via the derived ``tmc`` op sourced from ``x0`` (guaranteed 0, the vendor vx_tmc_zero form) then traps
    in a self-loop; warp 0 runs its own tile, then WAITS on the active-warp-mask CSR until only it remains
    (the BSP's vx_wspawn_wait primitive) and runs ``manager_tail`` (e.g. a result copy). Results are left in
    memory for the oracle to read back — no console print (which races across lanes on a SIMT core).

    ``worker_body``/``manager_tail`` are the capsule's per-warp computation (target-agnostic C); ``globals`` are
    file-scope declarations (e.g. the output buffer) placed after the include; this function contributes ONLY
    the derived control scaffold. Fail closed if the runtime ABI lacks the ops/CSRs."""
    tmc, wsp = model.sfu_op("tmc"), model.sfu_op("wspawn")
    wid_csr, wmask_csr = model.special_csr("warp_id"), model.special_csr("warp_mask")
    # the derived .insn control ops (opcode+funct3 from the SFU dispatch); rs1=x0 for the guaranteed park.
    tmc_zero = f'__asm__ volatile(".insn r {tmc["opcode"]:#04x},{tmc["funct3"]},0,x0,x0,x0")'
    wspawn = (f'__asm__ volatile(".insn r {wsp["opcode"]:#04x},{wsp["funct3"]},0,x0,%0,%1"'
              f'::"r"(n),"r"(f))')
    return f"""#include <stdint.h>
{globals}
static inline uint32_t _wid(void){{uint32_t r;__asm__ volatile("csrr %0,{wid_csr:#x}":"=r"(r));return r;}}
static inline uint32_t _wmask(void){{uint32_t r;__asm__ volatile("csrr %0,{wmask_csr:#x}":"=r"(r));return r;}}
static inline void _park(void){{{tmc_zero};for(;;){{}}}}
static inline void _spawn(uint32_t n,void(*f)(void)){{{wspawn};}}
#define MU_NUM_WARPS {num_warps}u
__attribute__((noreturn)) static void _mu_worker(void){{
  uint32_t wid=_wid();
  {worker_body}
  _park();
}}
int main(void){{
  _spawn(MU_NUM_WARPS,_mu_worker);
  {{ uint32_t wid=0; {worker_body} }}
  while(_wmask()!=1){{}}
  {manager_tail}
  return 0;
}}
"""


def _model_for(target: str):
    """The derived IsaModel for a target, or a MuonUnavailable when no encoding fact is present (fail closed)."""
    from merlin.targetgen.isa_model import isa_model_from_encoding
    from merlin.targetgen.rtl import mlc_bridge
    fact = mlc_bridge.isa_encoding_for(target)
    if not fact:
        raise MuonUnavailable(f"no derived ISA encoding fact for target {target!r}")
    return isa_model_from_encoding(target, fact)


def forkfree_compile_triple(model) -> tuple[str, str]:
    """Derive the STOCK-clang ``--target`` triple + ``-mabi`` from the target's derived ``base_isa_family`` —
    the source substrate the fixed-format core re-encodes (Muon re-encodes rv32 → ``riscv32``/``ilp32``). Fail
    closed on a non-RISC-V base: the transcoder's RISC-V decode taxonomy would silently mis-decode it."""
    fam = model.base_isa_family()
    if fam == "riscv32":
        return "riscv32", "ilp32"
    if fam == "riscv64":
        return "riscv64", "lp64"
    raise MuonError(f"fork-free build needs a RISC-V base substrate; derived base_isa_family={fam!r} "
                    f"for target {model.target!r} (fail closed — the re-encode assumes a RISC-V source)")


def build_forkfree_bsp(workdir: str | Path, *, target: str = "radiance", num_warps: int = 1) -> list[Path]:
    """Reproducibly build the fork-free BSP (boot + runtime shims) from the target's shipped sources — no
    transient or committed binaries. The assembler shim is rendered from the target's derived ``runtime_abi``
    (:func:`render_simt_preamble`), not a hardcoded preamble. Returns the object list
    :func:`compile_kernel_forkfree` links."""
    from . import muon_bsp
    from merlin.targetgen.contract.toolchain import mlir_bin
    lib = lib_dir()
    return muon_bsp.build_bsp(lib / "src/mu_start.S", lib / "tohost.S", Path(workdir) / "bsp",
                              target=target, clang=str(mlir_bin("clang")), mc=str(mlir_bin("llvm-mc")),
                              asm_preamble=render_simt_preamble(_model_for(target)),
                              occupancy=(OCCUPANCY_SYMBOL, num_warps))


def compile_kernel_forkfree(kernel_c: str, workdir: str | Path, bsp_objs: list[str | Path] | None = None,
                            *, target: str = "radiance", march: str | None = None, num_warps: int = 1) -> Path:
    """FORK-FREE build of a self-contained kernel: STOCK clang compiles the kernel C to rv32, the derived
    transcoder re-maps it into the target's fixed-format words, STOCK llvm-mc assembles them, and the
    fork-free linker (:func:`muon_link.link_fork_free`) links the vendored boot/runtime objects (``bsp_objs``
    — the crt0-like boot that uses the target's own SIMT instructions, a shipped runtime asset) + the kernel.
    No clang-muon in compile/assemble/link — every field position is DERIVED from the target's RTL encoding.
    A self-contained (relocation-free) kernel takes the flat .text re-map fast path; a kernel that carries an
    R_RISCV relocation (a constant pool, a cross-section global, or a mu_schedule warp-callback pointer) is
    transcoded at the OBJECT level (reloc records preserved) and the fork-free linker resolves every
    relocation at the derived field positions. Returns the ELF path."""
    from merlin.targetgen.isa_model import isa_model_from_encoding
    from merlin.targetgen.isa_transcode import FixedFormatTranscoder, to_data_lines, emit_kernel_asm, derive_march
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.targetgen.contract.toolchain import mlir_bin
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

    # 1. STOCK compile -> base-ISA object (self-contained: no PIC, no jump tables, no relaxation). The source
    # substrate triple + ABI are DERIVED from the target's base_isa_family (Muon re-encodes rv32); fail closed
    # on a non-RISC-V base.
    triple, mabi = forkfree_compile_triple(model)
    src = work / "kernel.c"
    src.write_text(kernel_c, encoding="utf-8")
    obj = work / "kernel.o"
    cc = subprocess.run([str(clang), f"--target={triple}", f"-march={march}", f"-mabi={mabi}", "-mno-relax",
                         "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-pic", "-fno-jump-tables",
                         "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
    if cc.returncode != 0:
        raise MuonError(f"stock rv32 compile failed:\n{cc.stderr[-2000:]}")
    has_reloc = any("R_RISCV" in ln for ln in
                    subprocess.run([str(objdump), "-r", str(obj)],
                                   capture_output=True, text=True).stdout.splitlines())

    # The emitted stream is self-checked with the SAME derived disassembler the agent tooling uses: every
    # word must decode to a defined opcode. This catches a transcoder regression at BUILD time (cheap)
    # instead of in a paid oracle run, and enforces the derived tooling on the emit path (fail closed).
    from merlin.targetgen import isa_disasm as _disasm

    def _lint_words(words: list[int]) -> None:
        illegal = [i for i, d in enumerate(_disasm.disassemble(model, words)) if d.get("illegal")]
        if illegal:
            raise MuonError(f"transcoded kernel has {len(illegal)} undecodable word(s) at index "
                            f"{illegal[:5]}; the derived disassembler rejects the emitted stream")

    kobj = work / "kernel_muon.o"
    if not has_reloc:
        # 2a. FAST PATH (self-contained kernel, no relocations): transcode the rv32 .text to a flat word
        # blob and re-assemble it as one exported entry. A pure field re-map — the proven functional path.
        binf = work / "kernel.text.bin"
        subprocess.run([str(objcopy), "-O", "binary", "--only-section=.text", str(obj), str(binf)],
                       capture_output=True)
        words = FixedFormatTranscoder(model).transcode_text(binf.read_bytes())
        _lint_words(words)
        ks = work / "kernel_muon.S"
        ks.write_text(emit_kernel_asm(words, model.inst_width))
        a = subprocess.run([str(mc), "--triple=riscv32", "--filetype=obj", str(ks), "-o", str(kobj)],
                           capture_output=True, text=True)
        if a.returncode != 0:
            raise MuonError(f"stock llvm-mc failed:\n{a.stderr[-1500:]}")
    else:
        # 2b. RELOCATION-PRESERVING PATH: transcode the whole OBJECT (grow code sections, scale symbol
        # values + reloc offsets, preserve reloc records + all other sections verbatim) so the fork-free
        # linker resolves every relocation at the DERIVED field positions. This admits a kernel with
        # internal calls (a mu_schedule warp callback), a constant pool, or .rodata — none of which a flat
        # .text re-map can carry. Same object mechanism the fork-free BSP already uses.
        from . import muon_bsp
        muon_bsp.transcode_boot_object(obj, kobj, isa_model=model)
        # lint the transcoded .text with the derived disassembler (fail closed on illegal), same as 2a.
        ktext = work / "kernel_muon.text.bin"
        subprocess.run([str(objcopy), "-O", "binary", "--only-section=.text", str(kobj), str(ktext)],
                       capture_output=True)
        stride, wb = model.inst_width // 8, ktext.read_bytes()
        _lint_words([int.from_bytes(wb[i:i + stride], "little") for i in range(0, len(wb), stride)])

    # 4. FORK-FREE link: boot/runtime (regenerated fork-free from source if not supplied) + the kernel.
    if bsp_objs is None:
        bsp_objs = build_forkfree_bsp(work, target=target, num_warps=num_warps)
    elf = work / "kernel.radiance.elf"
    muon_link.link_fork_free([*[str(o) for o in bsp_objs], str(kobj)],
                             str(lib_dir() / "linker/mu_link.ld"), str(elf), target=target)
    return elf


def is_mlir_artifact(artifact_text: str) -> bool:
    """True when the emitted 4th artifact is an LLVM-dialect MLIR module (the thesis path — a compiler
    lowering) rather than a C/C++ kernel: it defines a kernel via ``llvm.func @`` (structural, no regex).
    A C/C++ kernel never carries that token, so the oracle routes MLIR to the fork-free MLIR build."""
    return "llvm.func @" in artifact_text


def _kernel_symbol_from_mlir(mlir_text: str) -> str:
    """The kernel function name from an LLVM-dialect MLIR module, parsed STRUCTURALLY (no regex): the
    identifier after the first ``llvm.func @`` up to its ``(``. Fail closed if absent."""
    marker = "llvm.func @"
    if marker not in mlir_text:
        raise MuonError("emitted MLIR has no `llvm.func @<name>` kernel definition")
    name = mlir_text.split(marker, 1)[1].split("(", 1)[0].strip()
    if not name:
        raise MuonError("could not parse the kernel symbol from the emitted MLIR")
    return name


def compile_mlir_forkfree(lowered_mlir_text: str, cb: dict, workdir: str | Path,
                          *, target: str = "radiance", num_warps: int = 1) -> Path:
    """FORK-FREE build from the agent's LLVM-dialect MLIR 4th artifact (the thesis path — the agent emits a
    COMPILER lowering, not a hand C++ kernel). Pipeline: ``lower_to_llvm_ir`` (the shared MLIR→LLVM-IR front
    gemmini uses) → STOCK clang rv32 → ``kernel.o``; a runner-owned EXTERN-kernel harness ``main.o`` embeds
    the operands and calls the kernel (cross-object ``R_RISCV_CALL``); both objects are transcoded to the
    target's fixed-format words (reloc-preserving) and the fork-free linker resolves the call against the
    from-source BSP. No clang-muon anywhere. Returns the ELF path."""
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.targetgen.isa_model import isa_model_from_encoding
    from merlin.targetgen.isa_transcode import derive_march
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.targetgen.contract.toolchain import mlir_bin
    from . import muon_bsp, muon_link, muon_harness

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    clang = mlir_bin("clang")
    if not clang.is_file():
        raise MuonUnavailable(f"stock LLVM tool absent: {clang} (set MERLIN_MLIR_INSTALL)")
    fact = mlc_bridge.isa_encoding_for(target)
    if not fact:
        raise MuonUnavailable(f"no derived ISA encoding fact for target {target!r}")
    model = isa_model_from_encoding(target, fact)
    triple, mabi = forkfree_compile_triple(model)
    march = derive_march(model)
    cflags = [f"--target={triple}", f"-march={march}", f"-mabi={mabi}", "-mno-relax", "-mcmodel=medany",
              "-O2", "-ffreestanding", "-fno-pic", "-fno-jump-tables"]

    # 1. agent MLIR -> LLVM IR (.ll) via the shared llvmlower front -> STOCK clang rv32 -> kernel.o
    kernel_symbol = _kernel_symbol_from_mlir(lowered_mlir_text)
    ll = lower_to_llvm_ir(lowered_mlir_text, workdir=work)
    (work / "kernel.ll").write_text(ll, encoding="utf-8")
    kobj_rv = work / "kernel.o"
    cc = subprocess.run([str(clang), *cflags, "-c", str(work / "kernel.ll"), "-o", str(kobj_rv)],
                        capture_output=True, text=True)
    if cc.returncode != 0:
        raise MuonError(f"stock rv32 compile of the lowered MLIR failed:\n{cc.stderr[-2000:]}")

    # 2. runner-owned EXTERN-kernel harness main (operands from the cb) -> STOCK clang rv32 -> main.o
    main_c = muon_harness.external_main_from_cb(cb, kernel_symbol=kernel_symbol, model=model)
    if main_c is None:
        # Say WHICH cause it was. Two earlier versions of this message each blamed the wrong thing, and
        # each cost a full agentic round: "no canonical_inputs" was asserted unconditionally, and then
        # "no operand rule for the command shape" was asserted for every remaining case.
        #
        # That second one is worse than unhelpful, because the command shape is the one thing that is
        # IDENTICAL between a submission that grades and one that does not. Measured on the same capsule
        # (R0_gemm_fp32), both the promoted merlin package and an unaided baseline emit exactly
        # ['COMMIT', 'EVICT', 'MATMUL_RESIDENT', 'RES_PACK'] -- the vocabulary this target's own
        # interface_grammar.md documents. The difference is that one DECLARES its operands in `tensors`
        # and the other leaves the block empty, and `tensors` is optional in command_buffer.schema.json.
        # So a schema-valid, correctly-spelled command buffer was told its artifact was fine and this
        # harness was broken; the agent spent its round permuting opcodes, which is what the message
        # pointed at, and its own notes record "every semantically reasonable opcode set is rejected".
        #
        # Every branch of args_from_cb resolves operand names through `tensors` (shape + dtype, and the
        # roles that order the kernel ABI). With no tensors there is nothing to resolve, whatever the
        # opcodes are. Name THAT, and keep the genuine-tooling-gap wording for the case where the operands
        # really are declared and no rule matched.
        _ops = sorted({(c.get("opcode") or "?") for c in (cb.get("commands") or [])})
        if not (cb.get("canonical_inputs") or {}):
            raise MuonError(f"could not derive harness operands: the command buffer carries no "
                            f"canonical_inputs (commands: {_ops})")
        if not (cb.get("tensors") or {}):
            _named = sorted({v for c in (cb.get("commands") or [])
                             for v in (c.get("operands") or {}).values() if v})
            raise MuonError(
                f"could not derive harness operands: the command buffer declares no `tensors`, so the "
                f"operands its commands name ({_named or 'none'}) have no shape or dtype to embed. This "
                f"is a defect in the submitted command buffer, not in its opcodes — the command shape "
                f"{_ops} is supported. Declare every operand and result in `tensors` as "
                f"{{name: {{shape: [...], dtype: ..., role: input|weight|output}}}}. `tensors` is optional "
                f"in the schema but required to grade, since the harness embeds the operands by shape.")
        # Third cause: operands ARE declared and no rule matched -- including the declaration-derived
        # fallback, which binds from roles and shapes rather than from the opcode. Name exactly what it
        # needed and could not find, because the previous wording ("no operand rule for the command shape")
        # sent an agent to permute opcodes that were already correct. What the fallback requires is an
        # output-role tensor whose shape is declared: a produced shape cannot be inferred without the very
        # per-op knowledge the fallback exists to avoid.
        _tn = cb.get("tensors") or {}
        _roles = sorted({str((t or {}).get("role", "?")) for t in _tn.values()})
        _out = [nm for nm, t in _tn.items() if str((t or {}).get("role", "")).lower() == "output"]
        _out_unshaped = [nm for nm in _out if not (_tn.get(nm) or {}).get("shape")]
        if not _out:
            _why = (f"no tensor is declared with role 'output' (roles present: {_roles}), so the harness "
                    f"cannot tell which operand your kernel produces")
        elif _out_unshaped:
            _why = (f"the output tensor(s) {_out_unshaped} declare no `shape`, and a PRODUCED shape cannot "
                    f"be inferred without op-specific knowledge")
        else:
            _why = (f"the declared operands could not be bound to the command's shapes (outputs: {_out})")
        raise MuonError(
            f"could not derive harness operands: {_why}. Command shape {_ops} is NOT the problem — this "
            f"harness binds operands from the `tensors` declarations, not from opcode names, so any "
            f"opcode is acceptable provided every operand and result is declared as "
            f"{{name: {{shape: [...], dtype: ..., role: input|weight|output}}}}. If your declarations are "
            f"complete, this is a TOOLING gap, not a defect in the submitted artifact.")
    (work / "main.c").write_text(main_c, encoding="utf-8")
    mobj_rv = work / "main.o"
    mc = subprocess.run([str(clang), *cflags, "-c", str(work / "main.c"), "-o", str(mobj_rv)],
                        capture_output=True, text=True)
    if mc.returncode != 0:
        raise MuonError(f"stock rv32 compile of the harness main failed:\n{mc.stderr[-2000:]}")

    # 3. reloc-preserving transcode of BOTH objects (the main->kernel call is a cross-object relocation)
    kobj, mobj = work / "kernel_muon.o", work / "main_muon.o"
    muon_bsp.transcode_boot_object(kobj_rv, kobj, isa_model=model)
    muon_bsp.transcode_boot_object(mobj_rv, mobj, isa_model=model)

    # 4. fork-free link: from-source BSP + harness main + MLIR kernel
    bsp_objs = build_forkfree_bsp(work, target=target, num_warps=num_warps)
    elf = work / "kernel.radiance.elf"
    muon_link.link_fork_free([*[str(o) for o in bsp_objs], str(mobj), str(kobj)],
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
    # Activate cyclotron's functional MX-Gemmini co-model (cluster.rs gates it on this var being set).
    # Without it the accelerator MMIO block is a pure timing stub and a kernel driving the MX PE reads
    # back zeros. It is inert for any kernel that never touches the Gemmini MMIO window, so setting it
    # unconditionally is safe for the whole Muon corpus.
    env.setdefault("CYCLOTRON_MXGEMMINI", "1")
    # Run with a RAISED cycle cap. The stock config's ``[sim] timeout`` is 1e6 cycles; the sim's main loop
    # is ``for cycle in 0..timeout { if finished() {..} tick() }`` and, if a kernel has not finished by the
    # cap, it returns ``Err(0)`` — which Rust reports as exit 1 with the output TRUNCATED mid-print (seen as
    # a spurious "cyclotron exited 1" on large fused/MX kernels that legitimately need >1e6 cycles; the small
    # kernels finish well under it). Write a workdir copy of the config with a higher cap and run that — its
    # ``[timing] include = ["config/timing/…"]`` paths resolve against the config's dir, i.e. through the
    # ``config`` symlink created above, so they stay valid. Parsed structurally (no regex).
    run_cfg = work / "cyclotron.run.toml"
    _bumped = ["timeout = 20000000" if ln.strip().startswith("timeout ") and "=" in ln else ln
               for ln in config_path().read_text(encoding="utf-8").splitlines()]
    run_cfg.write_text("\n".join(_bumped) + "\n", encoding="utf-8")
    cmd = [str(cyclotron_path()), str(run_cfg),
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


#: Substrings the RTL completion grade depends on. They must survive console truncation WHEREVER they
#: appear in the file — the perf report's ``Cycles:`` line is near the start of the epilogue while the
#: watchdog's verdict is at the very end, so a pure tail window can drop the one that decides the grade.
_GSIM_MARKERS = ("Cycles:", "Timeout exceeded", "FINISHED: cycles=", "finished execution")


def _read_console(path: Path, *, tail_bytes: int | None = None) -> tuple[str, int, bool]:
    """Read a spooled RTL console back into a BOUNDED string: every marker-bearing line, plus the tail.

    Returns ``(console, total_bytes, truncated)``. The marker lines are kept because the grade is decided
    by their presence; the tail is kept because that is what a human reads to see why a run failed. A
    grade computed from this string is identical to one computed from the whole file for every marker in
    :data:`_GSIM_MARKERS`, and ``truncated`` says plainly when the returned text is not the whole console
    so nothing downstream reports a partial console as complete.

    Window size via ``MERLIN_MUON_GSIM_CONSOLE_TAIL_BYTES`` (default 256 KiB).
    """
    import collections
    import os as _os

    if tail_bytes is None:
        tail_bytes = int(_os.environ.get("MERLIN_MUON_GSIM_CONSOLE_TAIL_BYTES", str(256 * 1024)))
    if not path.is_file():
        return "", 0, False
    total = path.stat().st_size
    kept: list[str] = []
    tail: collections.deque[str] = collections.deque(maxlen=4096)   # lines, not bytes: cheap and ample
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.rstrip("\n")
            tail.append(s)
            # cap the marker set too: a pathological run could print "Cycles:" a million times
            if len(kept) < 512 and any(m in s for m in _GSIM_MARKERS):
                kept.append(s)
    tail_text = "\n".join(tail)
    if len(tail_text) > tail_bytes:
        tail_text = tail_text[-tail_bytes:]
    truncated = total > len(tail_text)
    # markers first, so `Cycles:` is found by a first-occurrence search even when the tail dropped it
    parts = []
    if kept:
        parts.append("\n".join(kept))
    if truncated:
        parts.append(f"[... console truncated: {total} bytes on disk at {path}, "
                     f"marker lines + last {len(tail_text)} bytes retained ...]")
    parts.append(tail_text)
    return "\n".join(parts), total, truncated


def _cycles_from_rtl_report(console: str) -> int | None:
    """Cycle count from the RTL sim's Muon performance report (a ``Cycles: <N>`` line), parsed
    structurally by locating the marker and reading the next integer token — no regex. Returns the
    first core's cycle count (the report repeats per core)."""
    marker = "Cycles:"
    idx = console.find(marker)
    if idx == -1:
        return None
    tail = console[idx + len(marker):].strip().split(maxsplit=1)
    return int(tail[0]) if tail and tail[0].isdigit() else None


def _run_verilator(elf: Path, timeout: int) -> tuple[str, int | None]:
    """Run the emitted Muon ELF on the ``RadianceTapeoutSimConfig`` Verilator RTL sim (the open-source
    cycle-accurate RTL cert the radiance kernels are evaluated on).

    The harness loads a fused rv64 'SoC' image (see :func:`fuse_soc_elf`) via the ``+loadmem`` backdoor
    (bypassing the TSI serial-load path, whose sub-word write trips a testchipip diplomatic monitor);
    cyclotron's ``copy_elf`` relocates the muon sections and drives the cores. The run needs an
    UNLIMITED stack (Verilator's deep ``eval_initial`` recursion overflows the default 8 MB and SIGSEGVs
    before any output) and the testchipip dramsim ini. Completion is the RTL ``Muon [...] finished
    execution.`` marker; the console also carries the kernel's OUT/DONE lines (our runner-owned harness
    prints them over UART) for the functional grade. Honest-unavailable (``MuonUnavailable``) on any
    build/load/launch failure or if the run never reaches the finished-execution marker — it never
    reports a pass it cannot stand behind."""
    import resource
    work = elf.parent
    soc = fuse_soc_elf(elf, work)
    sqlite = work / "muon_rtl_trace.sqlite"
    cmd = [str(verilator_path()), "+permissive", "+dramsim",
           f"+dramsim_ini_dir={verilator_dramsim_ini()}",
           "+max-cycles=10000000", "+ntb_random_seed_automatic", "+verbose",
           f"+trace-db={sqlite}", f"+loadmem={soc}", "+permissive-off", str(soc)]

    def _unlimited_stack() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_STACK,
                               (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError):
            pass

    # SPOOL TO DISK, NOT RAM -- the same defect as the GSIM path, and worse here: this command runs with
    # `+verbose` and `+max-cycles=10000000`, so the per-instruction commit trace it prints is unbounded in
    # the parent's memory. A single GSIM run at 12M cycles reached 72.67 GB buffered this way and made the
    # host's OOM killer terminate an unrelated experiment; Verilator at 10M cycles with +verbose is the
    # same shape. `_read_console` keeps every grading marker plus a bounded tail, so the verdict below is
    # unchanged while parent memory stays bounded and the full console stays on disk for debugging.
    log = work / "verilator_console.log"
    try:
        with log.open("wb") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                                  cwd=str(verilator_path().parent), preexec_fn=_unlimited_stack)
    except subprocess.TimeoutExpired as e:
        raise MuonUnavailable(f"verilator RTL sim timed out after {timeout}s") from e
    console, _nbytes, _trunc = _read_console(log)
    if "finished execution" not in console:
        raise MuonUnavailable(
            f"verilator RTL sim did not reach the finished-execution marker (rc={proc.returncode}). "
            f"console: {_nbytes} bytes at {log}\ntail:\n{console[-600:]}")
    return console, _cycles_from_rtl_report(console)


def _int_after(console: str, marker: str) -> int | None:
    """The first integer token following ``marker`` in ``console`` (structural, no regex). Used to read
    the ``... after <N> simulation cycles`` count the RTL watchdog prints when it hits the cycle cap."""
    idx = console.find(marker)
    if idx == -1:
        return None
    for tok in console[idx + len(marker):].split():
        if tok.isdigit():
            return int(tok)
    return None


def _smoke_signals(console: str, max_cycles: int, mx_ctrl_base: int | None) -> dict:
    """Derive the executability verdict from a bounded verilator console — STRUCTURALLY (str containment
    + token split, never regex, never a baked opcode). ``mx_ctrl_base`` (the accelerator MMIO command
    window, DERIVED from the target's ``mx_mmio`` fact) enables the MX-engagement check; None for a
    non-MX capsule leaves ``mx_engaged`` unknown (None), never a fabricated True/False.

    Legality is intentionally coarse: the ELF BOOTED, the cores made forward PROGRESS (issued
    instructions), the run terminated cleanly (reached the RTL ``finished execution`` marker OR was
    stopped by the benign max-cycles watchdog), and NO genuine fault fired. The benign watchdog the
    RadianceTapeoutSim TestDriver prints on the cap — ``*** FAILED *** (timeout)`` /
    ``%Fatal: TestDriver.v:NN: Assertion failed`` — is NOT a fault: it is the EXPECTED bounded stop and
    must never be read as an illegal-instruction/trap (the numeric runner's ``*** FAILED ***`` == trap
    heuristic would misclassify it, so the smoke path never routes through that raise)."""
    booted = "created sim object" in console and "loading ELF" in console
    # Forward progress: the verbose commit stream issues warp instructions (``[ISSUE]``) or retires a
    # scalar commit (``DASM(``). Either proves the fetched image executed, not merely elaborated.
    progressed = "[ISSUE]" in console or "DASM(" in console
    finished = "finished execution" in console
    # The max-cycles watchdog line ("*** FAILED *** (timeout) after <N> simulation cycles"): a clean,
    # bounded stop — NOT a fault. Its presence (or "finished execution") is what makes a legal run.
    hit_cap = "(timeout) after" in console
    cycles = _int_after(console, "(timeout) after")
    if cycles is None:                       # reached the report instead of the cap
        cycles = _int_after(console, "Cycles:")
    # A GENUINE fault: a verilator runtime error (``%Error``/``%Fatal``) that is NOT the benign TestDriver
    # max-cycles watchdog, or an explicit trap/illegal spelling in the commit stream. Hitting ``+max-cycles``
    # makes the RadianceTapeoutSim TestDriver execute ``$stop`` — verilator then prints
    # ``%Error: .../TestDriver.v:147: Verilog $stop`` alongside the ``*** FAILED *** (timeout)`` line; that
    # is the EXPECTED bounded stop, so any ``%Error``/``%Fatal`` originating in ``TestDriver`` (or carrying
    # the ``(timeout)`` marker) is filtered out. A real core assertion / illegal-instruction fires in a
    # DIFFERENT module and is NOT filtered.
    fault_lines: list[str] = []
    for ln in console.splitlines():
        low = ln.lower()
        benign_watchdog = ("testdriver" in low) or ("(timeout)" in low)
        if ("%error" in low or "%fatal" in low) and not benign_watchdog:
            fault_lines.append(ln.strip())
        elif ("illegal instruction" in low or "illegal_instruction" in low
              or "unimplemented" in low or "trap to" in low or "*** fault" in low):
            fault_lines.append(ln.strip())
    fault = bool(fault_lines)
    # MX-engagement: a store landed on the accelerator MMIO command window. Detect the ctrl_base address
    # appearing in a memory/issue line — the SIMT kernel loads it into a register and issues the shared
    # store that the MX PE consumes as a command. Zero-padded 8-hex token, matched only inside an
    # ISSUE/LSU/store line so an incidental occurrence elsewhere cannot false-positive.
    mx_engaged: bool | None = None
    if mx_ctrl_base is not None:
        tok = "%08x" % (mx_ctrl_base & 0xFFFFFFFF)
        mx_engaged = False
        for ln in console.splitlines():
            if tok in ln and ("[ISSUE]" in ln or "LSU" in ln or "store" in ln.lower()):
                mx_engaged = True
                break
    legal = booted and progressed and (finished or hit_cap) and not fault
    if not booted:
        reason = "did not boot (no sim-object/ELF-load markers) — sim build or loadmem failed"
    elif not progressed:
        reason = "booted but no instruction issued (cores never fetched the kernel image)"
    elif fault:
        reason = f"faulted on RTL: {fault_lines[0][:200]}"
    elif finished:
        reason = "ran to the RTL 'finished execution' marker (completed within the cycle cap)"
    elif hit_cap:
        reason = (f"ran legally to the {max_cycles}-cycle cap (bounded watchdog stop, no trap) — "
                  "executability confirmed; numerical correctness is the L2 oracle's job")
    else:
        reason = "run ended without a completion or cap marker (indeterminate)"
    ev = "; ".join(x for x in [
        "booted" if booted else None,
        "issued-instructions" if progressed else None,
        "finished-execution" if finished else "hit-max-cycles" if hit_cap else None,
        ("mx-command-accepted@0x%x" % mx_ctrl_base) if mx_engaged else None,
        ("FAULT:" + fault_lines[0][:120]) if fault else None,
    ] if x)
    return {"ran": booted, "legal": legal, "reason": reason, "cycles_capped": hit_cap,
            "cycles": cycles, "max_cycles": max_cycles, "booted": booted, "progressed": progressed,
            "finished": finished, "fault": fault,
            "fault_evidence": fault_lines[0][:300] if fault_lines else None,
            "mx_engaged": mx_engaged, "console_evidence": ev}


def run_elf_smoke(elf: str | Path, *, max_cycles: int = 40000, timeout: int = 900,
                  seed: int | None = None, mx_ctrl_base: int | None = None,
                  keep_console: bool = False) -> dict:
    """Bounded-cycle Verilator EXECUTABILITY smoke — the RTL-grounding backstop for the (non-RTL-cert)
    cyclotron oracle. Fuse the rv32 Muon ELF into the rv64 SoC carrier (:func:`fuse_soc_elf`) and run it
    on the SAME ``RadianceTapeoutSimConfig`` Verilator build the L3 cert uses, but with a SMALL
    ``+max-cycles`` cap so it CANNOT hang — the sim is bounded by the RTL watchdog AND the wall-clock
    ``timeout``. Returns a structured executability dict (see :func:`_smoke_signals`); it does NOT numeric-
    grade (that is the cyclotron L2 oracle's job) and NEVER raises on a cap-hit — only genuine
    unavailability (missing sim / fuse toolchain) raises :class:`MuonUnavailable`, so an absent RTL build
    degrades honestly instead of failing a capsule whose L2 grade passed.

    Determinism: the automatic random seed is dropped (a fixed default seed) so the same ELF yields the
    same verdict; pass ``seed`` to pin ``+ntb_random_seed`` explicitly. ``+verbose`` is REQUIRED — the
    per-cycle commit stream is what surfaces forward-progress and the MX command store; the cap keeps its
    volume bounded. The console is streamed to a logfile in the ELF's workdir (not held wholesale in
    memory)."""
    import resource
    import time as _time
    if not available("verilator"):
        raise MuonUnavailable("verilator RTL sim (or the rv64 SoC-fuse toolchain / dramsim ini) not "
                              "available — cannot run the executability smoke")
    elf = Path(elf).resolve()
    work = elf.parent
    soc = fuse_soc_elf(elf, work)
    sqlite = work / "muon_smoke_trace.sqlite"
    log = work / "verilator_smoke.log"
    cmd = [str(verilator_path()), "+permissive", "+dramsim",
           f"+dramsim_ini_dir={verilator_dramsim_ini()}",
           f"+max-cycles={int(max_cycles)}", "+verbose",
           f"+trace-db={sqlite}", f"+loadmem={soc}", "+permissive-off", str(soc)]
    if seed is not None:
        cmd.insert(-1, f"+ntb_random_seed={int(seed)}")  # else the fixed default seed (deterministic)

    def _unlimited_stack() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_STACK,
                               (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError):
            pass

    t0 = _time.perf_counter()
    with open(log, "wb") as fh:
        try:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                           cwd=str(verilator_path().parent), preexec_fn=_unlimited_stack)
        except subprocess.TimeoutExpired:
            # The wall-clock guard fired before the cycle cap — the run is still BOUNDED (not a hang),
            # but we cannot certify a clean stop. Report it as an honest indeterminate, not a fault.
            wall = round(_time.perf_counter() - t0, 1)
            console = _read_tail(log)
            sig = _smoke_signals(console, max_cycles, mx_ctrl_base)
            sig.update(ran=sig["booted"], legal=False, wall_s=wall, timed_out_wall=True,
                       reason=f"exceeded the {timeout}s wall-clock budget before the {max_cycles}-cycle "
                              f"cap (increase timeout or lower max_cycles); bounded, not a hang")
            return sig
    wall = round(_time.perf_counter() - t0, 1)
    console = log.read_text(errors="replace") if keep_console else _read_tail(log)
    # Signal derivation needs the WHOLE stream (MX store / progress can be anywhere), so scan the file.
    sig = _smoke_signals(log.read_text(errors="replace"), max_cycles, mx_ctrl_base)
    sig["wall_s"] = wall
    sig["log"] = str(log)
    sig["console_tail"] = console if not keep_console else console[-4000:]
    return sig


def _read_tail(path: Path, n: int = 4000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-n:].decode(errors="replace")


def run_elf(elf: str | Path, simulator: str = "cyclotron",
            timeout: int = 600) -> tuple[str, int | None, dict | None]:
    """Run the ELF on the chosen oracle; return (console, cycles, summary_json|None)."""
    # Resolve to an ABSOLUTE path up front: every sim below runs with ``cwd`` set to a workdir, so a
    # caller-supplied RELATIVE elf/runs-root (e.g. ``out/runs/...``) would otherwise resolve against the
    # sim's cwd and vanish ("Elf file not found"). Absolute-ising here makes a relative runs-root work
    # identically to an absolute one — one less thing to get right on a fresh clone / other machine.
    elf = Path(elf).resolve()
    if simulator == "cyclotron":
        return _run_cyclotron(elf, timeout)
    if simulator == "vcs":
        console, cycles = _run_vcs(elf, timeout)
        return console, cycles, None
    if simulator == "verilator":
        console, cycles = _run_verilator(elf, timeout)
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
    # DONE is the completion sentinel, but after a long OUT line the final DONE byte can race the sim's
    # exit and be dropped from the captured console. A COMPLETE OUT line is self-validating (a truncated
    # one already raised on the value-count check above), so accept a run that produced at least one full
    # output even if DONE was lost; only a run with NO parsed output is a genuine non-completion.
    if not done and not outputs:
        raise MuonError(f"run did not reach DONE and produced no output:\n{text[:2000]}")
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
                               else "rtl_vcs" if simulator == "vcs"
                               else "rtl_verilator" if simulator == "verilator" else "unknown")
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
