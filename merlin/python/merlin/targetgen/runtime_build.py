"""Self-contained, RTL-derived runtime build support for the bare-metal L2/L3 oracle.

The runtime splits cleanly into two kinds of configuration:

* **DERIVED hardware facts** — the platform (SoC) DRAM base, i.e. the load address the bare-metal linker
  must use. This is read from the TARGET'S OWN RTL BUILD memory map, never baked: a new HW RTL repo gets
  its runtime layout for free. (:func:`platform_dram_base`.)
* **Operator/setup config** — where the RTL build, toolchain, and sim binaries live. That is the person
  setting up the board's choice, so it stays in the target descriptor / ``.env`` (``sim_via``, the
  chipyard location via ``ext_path``, the sim config via the capability manifest), NOT derived here.

Dispatch is by the RTL build tool (``sim_via``, a descriptor fact), mirroring how the oracle adapters are
chosen — so this holds no target-name literal and extends to another build tool by adding a reader.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def _chipyard_config(target: str) -> str | None:
    """The declared verilator harness config for ``target`` (capability manifest ``runtime.rtl_sim_config``)
    — a per-target FACT read from the registry, not a hardcoded constant."""
    try:
        from .target_experiment import load_capability_manifest
        return (load_capability_manifest(target).contract.get("runtime") or {}).get("rtl_sim_config")
    except Exception:  # noqa: BLE001 — manifest unavailable ⇒ no config; caller falls back
        return None


def _chipyard_dram_base(target: str) -> int | None:
    """Derive the platform DRAM base from the target's chipyard RTL build memory map: the base of the
    largest ``memory@`` region in the generated ``<config>.memmap.json``. The chipyard location is a setup
    fact (``MERLIN_CHIPYARD`` / ``.env`` / ``ext_path``); the config is a manifest fact. Returns None if the
    build/memmap is absent (the caller uses a documented fallback), never a baked address."""
    cfg = _chipyard_config(target)
    if not cfg:
        return None
    from merlin.common.paths import env as _env, ext_path as _ext_path
    cy = _env("MERLIN_CHIPYARD") or _ext_path("chipyard")
    if not cy:
        return None
    hw = f"chipyard.harness.TestHarness.{cfg}"
    mm = Path(cy) / "sims" / "verilator" / "generated-src" / hw / f"{hw}.memmap.json"
    if not mm.is_file():
        return None
    try:
        regions = json.loads(mm.read_text()).get("mapping", [])
    except Exception:  # noqa: BLE001 — malformed memmap ⇒ no derivation, fall back
        return None
    mems = [r for r in regions if (r.get("names") or [""])[0].startswith("memory@")]
    if not mems:
        return None
    biggest = max(mems, key=lambda r: (r.get("size") or [0])[0])
    base = (biggest.get("base") or [None])[0]
    return int(base) if base is not None else None


# The bare-metal DRAM base used when the RTL memory map cannot be read (build absent). It is the RISC-V
# platform reset/DRAM base every Rocket/Chipyard-class SoC and spike/fesvr use — a documented default, not
# a per-target guess; the derived value from the RTL build always wins when available.
DEFAULT_PLATFORM_DRAM_BASE = 0x80000000


def platform_dram_base(target: str, sim_via: str | None) -> int:
    """The platform (SoC) DRAM base for ``target`` — the load address the bare-metal linker uses. DERIVED
    from the RTL build's memory map, dispatched by the RTL build tool (``sim_via``): chipyard reads its
    ``memmap.json`` ``memory@`` region. Falls back to :data:`DEFAULT_PLATFORM_DRAM_BASE` only when the
    build/memmap is unavailable. Keyed on the sim ENGINE's ``has_memmap`` capability, not its NAME. No
    per-target address is baked here."""
    from .capsule_runner import sim_oracle_caps            # function-local: avoid an import cycle
    caps = sim_oracle_caps(sim_via)
    derived = _chipyard_dram_base(target) if (caps is not None and caps.has_memmap) else None
    return derived if derived is not None else DEFAULT_PLATFORM_DRAM_BASE


def compiler_smoke(sim_via: str | None) -> tuple[bool, str]:
    """Pre-spend check that the RTL-oracle COMPILE toolchain actually WORKS — not merely that its binaries
    exist. It compiles a trivial LLVM-IR module to a riscv object with the oracle's own clang, so a missing
    or broken compiler is caught as a NO_GO before a paid run tool-crashes on every capsule (the retired-
    clang lesson: ``available()`` passed because the binaries were present, then the compile step failed).
    Only for a compile-based sim (its ``_SimOracle.is_compile_based`` capability); other oracles return
    n/a. Keyed on the capability, not the engine NAME."""
    from .capsule_runner import sim_oracle_caps            # function-local: avoid an import cycle
    caps = sim_oracle_caps(sim_via)
    if caps is None or not caps.is_compile_based:
        return True, "n/a (no compile-based oracle for this sim)"
    try:
        from merlin.llvmlower import toolchain as _tc
        clang = _tc.clang()
    except Exception as e:  # noqa: BLE001
        return False, f"clang toolchain unresolved: {e}"
    cp = Path(str(clang))
    if cp.is_absolute() and not cp.exists():
        return False, f"oracle clang not found at {clang} — the compile toolchain is not provisioned"
    with tempfile.TemporaryDirectory() as td:
        ll = Path(td) / "smoke.ll"
        obj = Path(td) / "smoke.o"
        ll.write_text("define i32 @f() {\nentry:\n  ret i32 0\n}\n", encoding="utf-8")
        try:
            r = subprocess.run([str(clang), "--target=riscv64-unknown-elf", "-march=rv64gc",
                                "-c", str(ll), "-o", str(obj)], capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            return False, f"oracle clang missing/not executable: {clang}"
        except Exception as e:  # noqa: BLE001
            return False, f"compile smoke could not run: {str(e)[-160:]}"
        if r.returncode != 0 or not obj.is_file():
            return False, f"oracle clang failed to compile a riscv object: {(r.stderr or '')[-200:]}"
    return True, f"oracle clang compiles riscv objects ({cp.name})"


def _rebase_ld(text: str, base: int) -> str | None:
    """Replace the FIRST absolute location-counter assignment ``. = 0x...;`` in a linker script with the
    derived ``base`` — structurally (str ops, no regex). Returns None if no such origin is found (the
    caller then uses the template unchanged, never a wrong rewrite)."""
    key = ". = 0x"
    i = text.find(key)
    if i < 0:
        return None
    j = text.find(";", i)
    if j < 0:
        return None
    return text[:i] + f". = {hex(base)};" + text[j + 1:]


def derived_link_script(base: int, template_ld: Path, out_dir: Path) -> Path:
    """Emit a bare-metal linker script whose load address is the DERIVED platform DRAM ``base`` (from
    :func:`platform_dram_base`), reusing the proven section layout of ``template_ld`` (the target's
    curated-harness linker script — a per-target setup fact the crt expects) but replacing its baked origin
    with the derived value. Layout stays exactly what the runtime needs; only the base becomes derived, not
    baked. If the template has no rewritable origin, it is copied through unchanged (fail-safe)."""
    text = template_ld.read_text(encoding="utf-8")
    rebased = _rebase_ld(text, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "link.derived.ld"
    out.write_text(rebased if rebased is not None else text, encoding="utf-8")
    return out
