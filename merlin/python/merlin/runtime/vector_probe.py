"""Build (and parse) the tiny image that reports a board's vector configuration.

Why this exists: for a board we cannot reach, VLEN has been an *inference*. The Kodiak board files put
no ``v`` in ``riscv,isa`` and state no width; the only numbers are per-sample
``CONFIG_RISCV_VECTOR_MAX_LEN`` values (256 in one sample, 512 in three) that size Zephyr's per-thread
save area and so only bound the truth from above. Building for the wrong width is the documented K1
trap — fixed-width vectors at double LMUL, spilling, no speedup.

The probe is a few hundred bytes reusing the bare-metal harness (``crt.S`` + ``htif.c`` + our linker
script), so it needs no RTOS and no filesystem, boots at ``0x80000000`` and speaks the same HTIF
console every other image here speaks. It ships next to the model binaries: the authors run it first,
in seconds, and the reply settles the width before anyone uploads megabytes.

It is deliberately a SEPARATE image rather than lines added to the model harness: reading ``vlenb``
with ``mstatus.VS == Off`` traps, and a trap inside a model run costs a long upload and reads as a
hang. Here the worst case costs seconds and the preceding lines already say why it stopped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.paths import runtime_dir
from .backends import spike as _spike
from .backends.spike_model import DRAM_BASE, RVV_CFLAGS, _harness_dir, _run


def build(work: str | Path, *, dram_base: int = DRAM_BASE, dram_bytes: int | None = None,
          vlen: int | None = None, console: str = "htif", sdk_dir: str | Path | None = None,
          sdk_chip: str | None = None, chip_freq_hz: int | None = None) -> Path:
    """Link the probe ELF into ``work`` and return its path.

    ``vlen`` only affects the ``-march`` the probe itself is built with; the number it REPORTS comes
    from the hardware's own CSRs, so the two are independent on purpose — a probe built for 128 still
    reports 256 on a 256-bit unit, which is exactly the mismatch we are trying to detect.

    ``console`` matters more here than anywhere else: this image exists to be the FIRST thing run on a
    board, so it has to speak the channel that board actually has. An HTIF probe on silicon with no
    host attached hangs on its second character and reports nothing — turning the cheap test into
    another unexplained hang. See ``sdk_facts`` for where the UART facts come from.
    """
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    gcc = _spike.gcc_path()
    h = _harness_dir()
    cflags = list(RVV_CFLAGS)
    if vlen is not None:
        from .backends.zephyr_model import march_with_vlen
        cflags = march_with_vlen(cflags, vlen)

    from .boards import CONSOLE_HTIF, CONSOLE_UART
    console_defs: list[str] = []
    console_src = h / "htif.c"
    if console == CONSOLE_UART:
        from .sdk_facts import derive_uart_console
        if not sdk_dir or not sdk_chip:
            raise RuntimeError("console='uart' needs sdk_dir + sdk_chip (facts are derived, "
                               "never hardcoded)")
        console_defs = derive_uart_console(sdk_dir, sdk_chip).macros(chip_freq_hz=chip_freq_hz)
        console_src = h / "console_uart.c"
    elif console != CONSOLE_HTIF:
        raise RuntimeError(f"unknown console kind {console!r}")

    objs = []
    for obj, src, extra in (("probe_main.o", h / "vlen_probe.c", []),
                            ("crt.o", h / "crt.S", []),
                            ("console.o", console_src, console_defs),
                            ("libc_min.o", h / "libc_min.c", [])):
        _run([gcc, *cflags, *extra, "-c", src, "-o", work / obj])
        objs.append(work / obj)

    elf = work / "vlen_probe.elf"
    # The weights symbol the shared linker script references is unused here; define it at the DRAM
    # base so the link resolves without a weights blob.
    _run([gcc, *cflags, "-nostdlib", "-nostartfiles",
          f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(dram_base)}",
          "-T", h / "model_link.ld", *objs, "-o", elf])
    return elf


def run_on_spike(elf: str | Path, *, vlen: int | None = None, dram_base: int = DRAM_BASE,
                 mem_bytes: int = 256 * 1024 * 1024, timeout: int = 120) -> str:
    """Run the probe on spike and return its console text.

    Not ``spike_model.run``: that one is the MODEL protocol and rejects a run with no ``OUT`` line,
    which the probe legitimately has none of.
    """
    import subprocess

    from .backends.zephyr_model import spike_isa

    cmd = [str(_spike.spike_path()), f"--isa={spike_isa(vlen)}", "-p1",
           f"-m{hex(dram_base)}:{hex(mem_bytes)}", str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (proc.stdout or "") + (proc.stderr or "")


def parse(console: str) -> dict[str, Any]:
    """``{key: value}`` from the probe's console text, plus a derived ``vlen_bits`` verdict.

    Fails LOUD rather than guessing: disagreement between ``vlenb`` and the ``vsetvli`` derivation is
    reported as ``consistent = False`` instead of being averaged away, and a run that never printed
    ``DONE`` is reported as incomplete instead of being read as a result.
    """
    out: dict[str, Any] = {}
    for line in console.splitlines():
        if not line.startswith("PROBE "):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                out[parts[1]] = int(parts[2])
            except ValueError:
                out[parts[1]] = " ".join(parts[2:])
    out["complete"] = "DONE" in console
    vlenb, e8 = out.get("vlenb"), out.get("vlmax_e8")
    e32 = out.get("vlmax_e32")
    if isinstance(vlenb, int):
        out["vlen_bits"] = vlenb * 8
        out["consistent"] = bool(e8 == vlenb and (e32 is None or e32 == vlenb // 4))
    else:
        out["vlen_bits"] = None
        out["consistent"] = False
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import tempfile

    from .boards import board as _board

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", default="spike_riscv64")
    ap.add_argument("--build-vlen", type=int, default=None,
                    help="-march vector width to BUILD with (does not affect what is reported)")
    ap.add_argument("--run", action="store_true", help="also run it on spike")
    ap.add_argument("--spike-vlen", type=int, default=None, help="VLEN to simulate")
    ap.add_argument("--out", default=None, help="copy the ELF here")
    a = ap.parse_args(argv)

    brd = _board(a.board)
    work = Path(tempfile.mkdtemp(prefix="vlen_probe_"))
    elf = build(work, dram_base=brd.dram_base, dram_bytes=brd.dram_bytes, vlen=a.build_vlen)
    print(f"probe: {elf} ({elf.stat().st_size} bytes)")
    if a.out:
        import shutil
        shutil.copy2(elf, a.out)
        print(f"copied to {a.out}")
    if a.run:
        console = run_on_spike(elf, vlen=a.spike_vlen, dram_base=brd.dram_base)
        rep = parse(console)
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("complete") else 1
    return 0


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
