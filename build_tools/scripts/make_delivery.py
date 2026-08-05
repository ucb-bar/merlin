#!/usr/bin/env python3
"""Assemble a delivery package: binaries someone else can run on a board we cannot reach.

The deliverable is not a measurement — it is everything needed to run the binary and to interpret the
run that comes back. Inputs and weights are already embedded in the image (``llvmlower.c_runtime``
bakes inputs in as C arrays and the weights as one blob), so the package needs no data files and the
board needs no filesystem, no host I/O and no network.

What each part is for:

* **the ELF(s)** — one per hart count, so the authors can run 1-hart and N-hart back to back and check
  the outputs are bit-identical *themselves*. That turns their board into a self-validating test: any
  difference is their SoC's vector/SMP state, not our arithmetic.
* **``expected_console.txt``** — the console text OUR spike run produced. They can diff it.
* **``golden*.npy``** — the references, because grading happens against them, not against the binary.
* **``grade.py``** — reads a console log and prints the verdict, using the *same* parser and gate the
  repo grades with (``zephyr_model._parse_console`` + ``_gate``). No merlin checkout needed beyond numpy.
* **``elf_audit.json``** — what we checked before shipping: segments inside their DRAM, entry point,
  ``.htif`` present, vector instructions present, and the upload-time estimate.
* **``manifest.json``** — board facts we built for, per-binary build hashes, the package id, and the
  merlin commit. The build hash also appears in the console (``METRIC build_hash``), so a log can be
  tied to a binary rather than being unattributable.
* **``README.md``** — the exact loader command for *that* board's own flow, and the honest status of
  each model.

Deliberately NOT here: archiving. Producing a directory keeps this composable and reviewable; zip it as
a final manual step if that is how it is being sent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

import numpy as np                                                          # noqa: E402

from merlin.common.artifacts import new_product                             # noqa: E402
from merlin.common.paths import repo_root                                   # noqa: E402
from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME                 # noqa: E402
from merlin.runtime import boards, elf_audit                                # noqa: E402
from merlin.runtime.boards import CONSOLE_HTIF                              # noqa: E402
from merlin.runtime.backends import zephyr_model as zm                      # noqa: E402
from merlin.rvvgen.registry import load_rvv_package                         # noqa: E402

#: How each board is loaded and run, in ITS OWN flow. Quoted from the board's own repo so the authors
#: recognise it, rather than a command we invented.
LOADER_DOC = {
    "chipyard_kodiak": """\
```bash
# 1. reset the board (Arduino-style controller on /dev/ttyUSB0 @ 9600; sends "all", waits "Completed: all")
python3 scripts/send_all.py

# 2. load + run + capture the console over UART-TSI + FESVR
pyuartsi --port /dev/ttyUSB2 --elf <THIS.elf> --load --hart0_msip --fesvr \\
         --baudrate 57600 --cflush_addr 0x2010200 --selfcheck
```
The run ends by itself: the image calls `sys_reboot()`, which the SoC turns into an HTIF `tohost=1`, and
the loader prints `DUT forcefuly exit`. Everything between the banner and `DONE` is what we need back.

If your `tools/pyuartsi` is the pinned submodule (`c1bbb3a`), invoke it as `python3 -m pyuartsi` and drop
`--use_symbols` — that version has neither the console-script entry point nor that flag.""",
    "gemmelos_bearly25": """\
**Output comes out of UART0 at 115200 baud, 8 data bits, 2 stop bits** — the same console your own
`printf` uses on a `PLATFORM=CHIP` build. Open that terminal first, then load:

```bash
# Load and run over UART-TSI (the Chipyard host tool; NOT pyuartsi):
make tsi-run TTY=<your TSI tty> BINARY=<THIS.elf>
#   equivalently: uart_tsi +tty=<TSI tty> +baudrate=921600 <THIS.elf>

# or over JTAG:
openocd -f platform/bearly25/bearly25.cfg -c "reset run" -c "halt" \\
        -c "load_image <THIS.elf>" -c "resume 0x80000000"
```

**You do not need to set up the UART or the PLL.** The image does it itself, first thing in `main`,
using the same sequence as your `bmark-lib/simple_setup.c::init_test()`: enable TX/RX with 2 stop bits,
program the divisor for the 50 MHz reset clock, park the clock domains on the slow source, program the
PLL to **500 MHz** (the ratio your own demos use), switch the domains over, then **re-program the baud
divisor** for the new clock. Every address, register offset and clock rate was read out of
`platform/bearly25/chip_config.h`, `driver/rocket-chip-blocks/uart/uart.h`, `driver/intel/pll/pll.h` and
`platform/bearly25/include/hal_rcc.h` in your repo — nothing is hardcoded on our side, and the derived
values match the disassembly of the working `dsp-whisper.elf` you sent (UART0 `0x10020000`, clock
selector `0x130000`, PLL `0x140000`, `DIV` at offset 24, 50 MHz reference).

There is **no handshake**: unlike your `libbmark` benchmarks these images do not wait for `SOH`/`ENQ`
from a host tester. They start computing as soon as they are resumed and print when done.

This is a **bare-metal** ELF (entry `_start`, resumed at `0x80000000`, `-mcmodel=medany`, static), built
against our own `crt.S` and linker script rather than your CMake — so `glossy`, `libbmark` and your
linker scripts are not involved and cannot conflict.

Note `uart_tsi` uploads each segment's **MemSiz**, so the estimated upload time below is real.""",
    "gemmelos_bearly25_zephyr": """\
**Output comes out of UART0 at 115200 baud** — the same console your own `printf` uses on a
`PLATFORM=CHIP` build. Open that terminal first, then load:

```bash
# Load and run over UART-TSI (the Chipyard host tool; NOT pyuartsi):
make tsi-run TTY=<your TSI tty> BINARY=<THIS.elf>
#   equivalently: uart_tsi +tty=<TSI tty> +baudrate=921600 <THIS.elf>

# or over JTAG:
openocd -f platform/bearly25/bearly25.cfg -c "reset run" -c "halt" \\
        -c "load_image <THIS.elf>" -c "resume 0x80000000"
```

**You do not need to set up the UART or the PLL.** These are Zephyr images and Zephyr brings the
console up during boot, before any of our code runs. Two consequences worth knowing:

- The chip stays on its **50 MHz reset clock** — we do not program the PLL on this path, because the
  console is initialised before we could. So wall-clock time is roughly 10x what the same work would
  take at the 500 MHz your demos run at. **The reported `cycles` are unaffected** and are the number to
  compare; they are what the multicore speed-up is measured in.
- The baud divisor is computed by Zephyr's SiFive UART driver as
  `(SYS_CLOCK_HW_CYCLES_PER_SEC * RTC_CLOCK_DIVIDER_VALUE) / 115200 - 1`. We set those to `50000` and
  `1000` — i.e. your `MTIME_FREQ` and `SYS_CLK_FREQ / MTIME_FREQ` from `platform/bearly25/chip_config.h`
  — which gives divisor **433**, exactly what your `uart_init()` computes. (The board's own defaults
  imply a 1 GHz peripheral clock and would have produced garbage.)

Why Zephyr and not your bare-metal SDK: this is the **multicore** package, and driving both harts goes
through Zephyr SMP plus our OpenMP shim. Your bare-metal path dispatches hart 1 from your own
`thread-lib`, which we have not integrated. The single-hart bare-metal package is the one that matches
your SDK exactly, and it also runs at 500 MHz.

There is **no handshake**: these images do not wait for `SOH`/`ENQ` from a host tester. They boot,
compute, print, and stop.""",
    "default": """\
Load the ELF with your usual loader and capture the console. The image is self-contained: no filesystem,
no host I/O, no arguments. It ends by rebooting, after printing `DONE`.""",
}

GRADE_PY = '''#!/usr/bin/env python3
"""Grade a console log from this package's binary. Usage: python grade.py <console.txt> [--model NAME]

Uses the same parser and gate merlin grades with, vendored to a single file so this needs only numpy.
Tiers: `w8a8` (vs the W8A8 reference: cos > 0.999, rel < 1e-2, per-element max-rel < 5%) and `fp32`
(vs the weight-only golden: cos > 0.99, argmax matches). Grading a W8A8 run against golden.npy measures
activation-quantization error rather than correctness, which is why both references ship.
"""
import argparse, json, struct, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def parse_console(text):
    out_line = next((l for l in text.splitlines() if l.startswith("OUT ")), None)
    if out_line is None or "DONE" not in text:
        raise SystemExit("this log has no OUT line and/or no DONE — the run did not complete")
    parts = out_line.split()
    n = int(parts[1])
    bits = [int(x) for x in parts[2:2 + n]]
    vals = np.array([struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0] for b in bits],
                    dtype=np.float32)
    metrics = {}
    for l in text.splitlines():
        if l.startswith("METRIC "):
            p = l.split()
            if len(p) >= 3 and p[1] != "iter_cycles":
                try:
                    metrics[p[1]] = int(p[2])
                except ValueError:
                    metrics[p[1]] = p[2]
    return vals, metrics


def grade(prefix, refs, max_rel=0.05):
    out = {}
    for tier, ref in refs.items():
        r = np.asarray(ref, dtype=np.float32).ravel()[:len(prefix)]
        rmax = max(1e-9, float(np.abs(r).max()))
        rms = max(1e-9, float(np.sqrt(np.mean(r.astype(np.float64) ** 2))))
        sig = np.abs(r) >= 1e-2 * rms
        out[tier] = {
            "cos": float(prefix @ r / (np.linalg.norm(prefix) * np.linalg.norm(r) + 1e-12)),
            "rel": float(np.abs(prefix - r).max()) / rmax,
            "argmax": bool(int(np.argmax(prefix)) == int(np.argmax(r))),
            "max_rel": (float((np.abs(prefix[sig] - r[sig]) / np.abs(r[sig])).max())
                        if sig.any() else 0.0)}
    w, f = out.get("w8a8"), out.get("fp32")
    t1 = bool(w and w["cos"] > 0.999 and w["rel"] < 1e-2 and w["max_rel"] < max_rel)
    t2 = bool(f and f["cos"] > 0.99 and f["argmax"] and f["max_rel"] < max_rel)
    return {"tiers": out, "tier_ok": "w8a8" if t1 else "fp32" if t2 else None, "ok": t1 or t2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("console")
    ap.add_argument("--model", default=None, help="model name (default: the only one in the manifest)")
    a = ap.parse_args()
    man = json.loads((HERE / "manifest.json").read_text())
    models = sorted({b["model"] for b in man["binaries"]})
    model = a.model or (models[0] if len(models) == 1 else None)
    if model is None:
        raise SystemExit(f"--model is required; this package has: {', '.join(models)}")
    text = Path(a.console).read_text(errors="replace")
    prefix, metrics = parse_console(text)
    refs = {}
    for tier, fn in (("fp32", f"{model}.golden.npy"), ("w8a8", f"{model}.golden_w8a8.npy")):
        p = HERE / fn
        if p.is_file():
            refs[tier] = np.load(p)
    if not refs:
        raise SystemExit(f"no reference .npy for {model} in this package")
    res = grade(prefix, refs)
    print(f"model      : {model}")
    print(f"build_hash : {metrics.get('build_hash', '(absent — old binary or truncated log)')}")
    known = {b["build_hash"] for b in man["binaries"] if b["model"] == model}
    if metrics.get("build_hash") and metrics["build_hash"] not in known:
        print(f"  WARNING: that build_hash is not in this package (expected one of {sorted(known)})")
    print(f"cycles     : {metrics.get('cycles')}")
    print(f"elements   : {len(prefix)}")
    for tier, m in res["tiers"].items():
        print(f"  vs {tier:5s}: cos={m['cos']:.9f} rel={m['rel']:.6g} max_rel={m['max_rel']:.6g}")
    print(f"VERDICT    : {'PASS' if res['ok'] else 'FAIL'} (tier_ok={res['tier_ok']})")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

#: Honest status per model — shipped verbatim in the README, because a number nobody qualified is worse
#: than no number.
STATUS = {
    "spectformer": "VERIFIED. Bit-exact against the W8A8 reference on spike (1 and 3 harts) and on a "
                   "SpacemiT K1 board (real RVV silicon): w8a8_rel = 0.0. A failure here is about the "
                   "board, not about us — which is why this is the one to run first.",
    "deepjscc": "VERIFIED with per-op register blocking: w8a8_cos = 1.0, rel = 0.0. (With the older "
                "per-op-CLASS schedule this model scored 0.9176; that is fixed, not hidden.)",
    "lstmnetvit": "KNOWN DIVERGENT on RISC-V: w8a8_cos 0.9943 on spike and on the K1, while the same "
                  "IR is exact on x86. The 1-hart and N-hart runs ARE bit-identical, so it is useful "
                  "for bring-up and timing. Do not treat its output as an accuracy result — the cause "
                  "is ours to fix, not yours to debug.",
    "whisper_tiny": "Builds and lowers with 100% of its MACs on the vector path (per-op blocking). Its "
                    "encoder attention makes it much longer-running than the others; treat a long run "
                    "as expected rather than as a hang, and send the console log whatever it says.",
}


def build_baremetal(bundle: Path, brd, *, work: Path, timeout: int, sdk_dir=None):
    """Build a BARE-METAL image for a board with no RTOS, and run it on spike.

    Used for Baremetal-IDE-style targets (gemmelos). Same lowering, same c_runtime artifacts and the same
    console protocol as the Zephyr path — only the harness differs (crt.S + our linker script + an
    absolute memory map instead of a Zephyr app). The map is packed inside the board's real DRAM, which
    the default spike map is not: its arena at 0xC0000000 and weights at 0x2_0000_0000 are simply not
    memory on a 1 GB chip, so the image would fault on its first activation.
    """
    from merlin.runtime.backends import spike_model

    # The SAME package, features and int8 datapath the Zephyr path uses. Not optional: lowering the raw
    # bundle here scored cos 0.925 on deepjscc, which the prepared path gets bit-exact.
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    b = spike_model.build(bundle, work, inputs_npz=bundle / "inputs.npz",
                          dram_base=brd.dram_base, dram_bytes=brd.dram_bytes,
                          int8_compute=True, features=frozenset([PEROP_BLOCK_NAME]),
                          rvv_schedule=pkg.schedule_text,
                          cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=brd.vlen)
    refs = {"fp32": np.load(bundle / "golden.npy")}
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    run = spike_model.run(b["elf"], harts=1, mem_bytes=b["mem_bytes"], timeout=timeout,
                          isa=zm.spike_isa(brd.vlen))
    res = dict(run)
    res.update(zm._gate(run["prefix"], refs))
    res.setdefault("metrics", run.get("metrics", {}))
    res["vlen"] = brd.vlen or 128
    ship = b
    if brd.console != CONSOLE_HTIF:
        # The image above is the one we GATE: spike provides the host that HTIF needs. The image we
        # SHIP has to speak the board's own console instead, so it is built separately -- with the
        # same bundle, package, features and vlen, so `build_hash` (a digest of the lowered model
        # object plus the weights blob) is identical and the package can say the compute is the same
        # binary content and only the output channel differs.
        ship = spike_model.build(bundle, work / "board", inputs_npz=bundle / "inputs.npz",
                                 dram_base=brd.dram_base, dram_bytes=brd.dram_bytes,
                                 int8_compute=True, features=frozenset([PEROP_BLOCK_NAME]),
                                 rvv_schedule=pkg.schedule_text,
                                 cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=brd.vlen,
                                 console=brd.console, sdk_dir=sdk_dir, sdk_chip=brd.sdk_chip,
                                 chip_freq_hz=brd.chip_freq_hz)
        if ship.get("build_hash") != b.get("build_hash"):
            raise RuntimeError(
                f"the gated image and the shipped image disagree on build_hash "
                f"({b.get('build_hash')} vs {ship.get('build_hash')}) -- they are not the same "
                "computation, so the gate does not cover what is being shipped")
    return res, {"elf": ship["elf"], "ram_bytes": ship["mem_bytes"],
                 "build_hash": ship.get("build_hash", ""),
                 "console": ship.get("console", CONSOLE_HTIF),
                 "chip_freq_hz": ship.get("chip_freq_hz"),
                 "console_provenance": ship.get("console_provenance", {})}


def build_board_only(bundle: Path, brd, harts: int, *, work: Path, sdk_dir=None,
                     backend: str = "rvv"):
    """Build the board image and return its facts, without running it anywhere.

    The honest use for this is when the numbers are already established on better evidence than a
    functional simulator can give -- our own RTL on FireSim, bit-exact against the W8A8 reference --
    and the remaining question is whether THIS board's image links and audits, which is a build-time
    and ELF-level property. The package records that no console reference exists for these binaries
    rather than implying one.
    """
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    if brd.flow == boards.FLOW_BAREMETAL:
        # Respect the board's FLOW. Skipping the simulation must not also silently change which
        # harness the image is built with: a board whose SDK has no RTOS cannot run a Zephyr image,
        # and building one anyway would ship an ELF that cannot boot on the target it names.
        from merlin.runtime.backends import spike_model
        b = spike_model.build(bundle, work, inputs_npz=bundle / "inputs.npz",
                              dram_base=brd.dram_base, dram_bytes=brd.dram_bytes,
                              int8_compute=True, features=frozenset([PEROP_BLOCK_NAME]),
                              rvv_schedule=pkg.schedule_text,
                              cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=brd.vlen,
                              console=brd.console, sdk_dir=sdk_dir, sdk_chip=brd.sdk_chip,
                              chip_freq_hz=brd.chip_freq_hz)
        return {"elf": b["elf"], "ram_bytes": b["mem_bytes"],
                "build_hash": b.get("build_hash", ""), "console": b.get("console", CONSOLE_HTIF),
                "chip_freq_hz": b.get("chip_freq_hz"),
                "console_provenance": b.get("console_provenance", {})}
    if backend != "rvv":
        # A scalar image gets none of the vector machinery: no RVV package, no per-op register
        # blocking, no -march vector width. Passing them would be meaningless at best and would
        # reintroduce vector instructions at worst -- the whole point is an image that runs on a hart
        # with no vector unit.
        b = zm.build_app(bundle, work, board=brd.name, backend=backend, rvv_hart=0,
                         cpus=max(harts, brd.harts), n_harts=harts, int8_compute=True,
                         sdk_dir=sdk_dir)
        return {"elf": b["elf"], "ram_bytes": b["ram_bytes"],
                "build_hash": b.get("build_hash", ""), "backend": backend}
    b = zm.build_app(bundle, work, board=brd.name, backend="rvv", rvv_hart=0,
                     cpus=max(harts, brd.harts), n_harts=harts, int8_compute=True,
                     rvv_schedule=pkg.schedule_text,
                     cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
                     features=frozenset([PEROP_BLOCK_NAME]), vlen=brd.vlen, sdk_dir=sdk_dir)
    return {"elf": b["elf"], "ram_bytes": b["ram_bytes"], "build_hash": b.get("build_hash", "")}


def build_one(bundle: Path, brd, harts: int, *, vlen, work: Path, timeout: int, sdk_dir=None):
    """Build one image and run it on spike at the board's VLEN and hart count."""
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    refs = {"fp32": np.load(bundle / "golden.npy")}
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    res = zm.build_and_run(
        bundle, work, board="spike_riscv64",          # run on spike; the ELF for the board is separate
        backend="rvv", rvv_hart=0, harts=max(2, harts), int8_compute=True, n_harts=harts,
        rvv_schedule=pkg.schedule_text, cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        features=frozenset([PEROP_BLOCK_NAME]), references=refs, vlen=vlen, timeout=timeout)
    board_build = zm.build_app(
        bundle, work / f"board_h{harts}", board=brd.name, backend="rvv", rvv_hart=0,
        cpus=max(2, harts), int8_compute=True, rvv_schedule=pkg.schedule_text,
        cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        features=frozenset([PEROP_BLOCK_NAME]), n_harts=harts, vlen=vlen, sdk_dir=sdk_dir)
    return res, board_build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", required=True, help="Zephyr board (e.g. chipyard_kodiak)")
    ap.add_argument("--models", default="spectformer,deepjscc,lstmnetvit",
                    help="comma-separated workload names")
    ap.add_argument("--harts", default="1,3", help="comma-separated hart counts, one binary each")
    ap.add_argument("--dram-mb", type=int, default=None, help="the board's REAL DRAM")
    ap.add_argument("--vlen", type=int, default=None, help="the board's REAL vector length")
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--scalar-harts", default="",
                    help="comma-separated hart counts to ALSO build as SCALAR images (no vector "
                         "instructions). The point is a heterogeneous SoC: a chip may bring up more "
                         "cores than it attaches vector units to, and a scalar image is the only way "
                         "to use the extra ones. Slower per core, but it is the whole machine.")
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--no-spike-models", default="",
                    help="comma-separated models to build+audit WITHOUT simulating, while the rest "
                         "keep their gate. For the case where one model's functional run costs hours "
                         "and its evidence comes from elsewhere (FireSim), but the others are cheap.")
    ap.add_argument("--no-spike", action="store_true",
                    help="build and audit the board ELFs but do not simulate them. Use when the "
                         "lowering is already validated on stronger evidence (e.g. FireSim on the "
                         "real RTL) and the spike gate is not worth its wall clock. The package then "
                         "ships WITHOUT an expected_console for those binaries, and says so.")
    ap.add_argument("--jobs", type=int, default=None,
                    help="images to build/simulate at once (default: min(images, 6)); each is one "
                         "single-threaded spike")
    ap.add_argument("--out", default=None, help="destination dir (default: an out/artifacts product)")
    ap.add_argument("--sdk-dir", default=None,
                    help="the target's own SDK checkout. REQUIRED for a board whose console is its "
                         "own UART: the UART address and the clock rates its baud divisor depends on "
                         "are derived from that SDK's headers rather than hardcoded here.")
    a = ap.parse_args(argv)

    overrides = {}
    if a.dram_mb:
        overrides["dram_bytes"] = a.dram_mb * 1024 * 1024
    if a.vlen:
        overrides["vlen"] = a.vlen
    brd = boards.board(a.board, **overrides)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    no_spike_models = {m.strip() for m in a.no_spike_models.split(",") if m.strip()}
    unknown = no_spike_models - set(models)
    if unknown:
        print(f"[make_delivery] --no-spike-models names models not in --models: {sorted(unknown)}",
              file=sys.stderr)
        return 2
    hart_list = [int(h) for h in a.harts.split(",") if h.strip()]
    scalar_hart_list = [int(h) for h in a.scalar_harts.split(",") if h.strip()]
    if any(h > brd.harts for h in scalar_hart_list):
        print(f"[make_delivery] refusing a scalar image over more than {brd.harts} harts",
              file=sys.stderr)
        return 2
    if max(hart_list) > brd.harts:
        print(f"[make_delivery] refusing {max(hart_list)} harts: {brd.name} has {brd.harts}",
              file=sys.stderr)
        return 2
    # Fail here rather than three hours into a build: without the SDK there is no way to know this
    # chip's console, and the fallback (a host-assisted channel) is precisely the bug that made the
    # first delivery print nothing on real hardware.
    if brd.console != CONSOLE_HTIF and not a.sdk_dir:
        print(f"[make_delivery] {brd.name} has a '{brd.console}' console: pass --sdk-dir "
              f"<the target's SDK checkout> so its facts can be derived", file=sys.stderr)
        return 2

    if a.out:
        dest = Path(a.out)
        dest.mkdir(parents=True, exist_ok=True)
        manifest_writer = None
    else:
        prod = new_product("delivery", version=1, target=a.board,
                           notes=f"int8 multicore-RVV binaries for {a.board}")
        dest, manifest_writer = Path(prod.path), prod
    print(f"[make_delivery] board={brd.name} dram={brd.dram_bytes // 2**20}MB harts={brd.harts} "
          f"vlen={brd.vlen or 'unknown(assume 128)'} -> {dest}", flush=True)

    binaries, audits, problems = [], {}, []
    todo = []                                     # (model, bundle, harts) -- one image each
    for model in models:
        bundle = repo_root() / f"out/artifacts/recaptures/{model}_{a.dtype}_full"
        if not (bundle / "model.mlir").is_file():
            problems.append(f"{model}: no bundle at {bundle}")
            continue
        for tier, fn in (("fp32", "golden.npy"), ("w8a8", "golden_w8a8.npy")):
            if (bundle / fn).is_file():
                shutil.copy2(bundle / fn, dest / f"{model}.{fn}")
        for harts in hart_list:
            if brd.flow == boards.FLOW_BAREMETAL and harts != 1:
                # gemmelos dispatches hart 1 through its own thread-lib, not OpenMP; a multi-hart
                # bare-metal image is a separate integration, so say so rather than shipping something
                # untested.
                problems.append(f"{model} h{harts}: baremetal multicore not implemented "
                                f"(hart 1 waits in wfi; their thread-lib dispatches it)")
                continue
            todo.append((model, bundle, harts, "rvv"))
        # SCALAR images, when asked for. Their reason to exist is a heterogeneous SoC: the vector
        # units may sit on fewer harts than the chip brings up, and a scalar image is the only way
        # to put the remaining cores to work.
        for harts in scalar_hart_list:
            if brd.flow == boards.FLOW_BAREMETAL and harts != 1:
                continue
            todo.append((model, bundle, harts, "scalar"))

    def _one(item):
        model, bundle, harts, backend = item
        tag = f"{model} h{harts}" + ("" if backend == "rvv" else f" {backend}")
        work = Path(tempfile.mkdtemp(prefix=f"delivery_{model}_{backend}_h{harts}_"))
        # A scalar image has no vector gate to run and no RVV package to apply; it is built and
        # audited, never simulated, and the package says so rather than implying a gate.
        if a.no_spike or model in no_spike_models or backend != "rvv":
            print(f"  [{tag}] building (no simulation) in {work}", flush=True)
            board_build = build_board_only(bundle, brd, harts, work=work, sdk_dir=a.sdk_dir,
                                           backend=backend)
            print(f"  [{tag}] built: {board_build['ram_bytes'] // 2**20} MB region, "
                  f"{board_build['build_hash']}", flush=True)
            return {"console": "", "metrics": {}, "outputs": None, "backend": backend}, \
                board_build
        print(f"  [{tag}] building + simulating in {work}", flush=True)
        if brd.flow == boards.FLOW_BAREMETAL:
            out = build_baremetal(bundle, brd, work=work, timeout=a.timeout, sdk_dir=a.sdk_dir)
        else:
            out = build_one(bundle, brd, harts, vlen=brd.vlen, work=work, timeout=a.timeout,
                            sdk_dir=a.sdk_dir)
        cyc = out[0]["metrics"].get("cycles")
        print(f"  [{tag}] done: {cyc:,} cycles, gate={out[0].get('tier_ok')}", flush=True)
        return out

    # CONCURRENTLY, because the wall clock here is spike, not us: each image is one single-threaded
    # functional simulation of a whole int8 inference (spectformer alone is 2.78 G cycles at roughly
    # 2.4 M cycles/s, i.e. ~20 minutes), and a delivery is several of them. Serially that is hours for
    # a package whose slowest single item is under an hour. Each job is its own temp dir, its own build
    # tree and its own subprocesses, so the only shared state is the result assembly below -- done
    # after the pool drains, in a deterministic order rather than completion order.
    jobs = a.jobs or min(len(todo), 6)
    print(f"[make_delivery] {len(todo)} image(s), {jobs} at a time", flush=True)
    done: dict[tuple, object] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_one, it): it for it in todo}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                done[item] = fut.result()
            except Exception as exc:                                        # noqa: BLE001
                msg = f"{item[0]} h{item[2]} {item[3]}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                problems.append(msg)
                # Say it NOW: a failure held until the final summary reads as an image still building,
                # and the rest of the set can take tens of minutes.
                print(f"  FAILED {msg}", file=sys.stderr, flush=True)

    for model in models:
        outputs = {}
        for harts, backend in ([(h, "rvv") for h in hart_list]
                               + [(h, "scalar") for h in scalar_hart_list]):
            got = done.get((model, repo_root() / f"out/artifacts/recaptures/{model}_{a.dtype}_full",
                            harts, backend))
            if got is None:
                continue
            res, board_build = got
            # Scalar images carry `_scalar` so a directory listing cannot confuse the two: they are
            # different instruction sets for different harts, not two builds of one thing.
            suffix = "" if backend == "rvv" else f"_{backend}"
            elf_name = f"{model}_{a.dtype}_h{harts}{suffix}_{brd.name}.elf"
            shutil.copy2(board_build["elf"], dest / elf_name)
            # require_vector is the whole point of a scalar image being scalar: demanding vector
            # instructions there would fail a correct build, and NOT checking it on a vector build
            # would let a silently-scalar image ship as if it had been vectorized.
            rep = elf_audit.audit(dest / elf_name, brd,
                                  ram_bytes=board_build["ram_bytes"],
                                  require_vector=(backend == "rvv"))
            audits[elf_name] = rep.to_dict()
            if not rep.ok:
                problems += [f"{elf_name}: {p}" for p in rep.problems]
            if res["console"]:
                (dest / f"{model}_h{harts}{suffix}.expected_console.txt").write_text(res["console"])
            if res["outputs"] is not None:
                outputs[harts] = res["outputs"]
            binaries.append({
                "model": model, "elf": elf_name, "harts": harts, "dtype": a.dtype,
                "backend": backend,
                "build_hash": board_build.get("build_hash", ""),
                "ram_bytes": board_build["ram_bytes"],
                "spike_cycles": res["metrics"].get("cycles"),
                "spike_vlen": res.get("vlen"), "gate_ok": bool(res.get("ok")),
                "tier_ok": res.get("tier_ok"), "cos": res.get("cos"), "rel": res.get("rel"),
                "upload_estimate_s": rep.facts.get("upload_estimate_s"),
            })
            cyc = res["metrics"].get("cycles")
            print(f"  {elf_name}: cycles={cyc:,} gate={res.get('tier_ok')} "
                  f"audit={'OK' if rep.ok else 'FAIL'}" if cyc is not None else
                  f"  {elf_name}: not simulated, audit={'OK' if rep.ok else 'FAIL'}", flush=True)
        # 1-hart vs N-hart bit-identity: the property a multicore run exists to establish
        if len(outputs) > 1:
            base_h = min(outputs)
            for h, arr in outputs.items():
                if h == base_h:
                    continue
                if not np.array_equal(outputs[base_h], arr):
                    problems.append(f"{model}: {h}-hart output differs from {base_h}-hart — an "
                                    f"overlapping or lost work split, not rounding")
                else:
                    print(f"  {model}: {h}-hart output bit-identical to {base_h}-hart", flush=True)

    # The VECTOR PROBE. Cheapest useful thing in the package: a ~10 KB image that reports the board's
    # own vlenb, misa.V and mstatus.VS. Every VLEN statement we make is otherwise an inference (the
    # Kodiak board files declare no width, and its samples' CONFIG_RISCV_VECTOR_MAX_LEN only bounds it
    # from above), and building for the wrong width is the documented K1 trap. Run first, it settles
    # the question in seconds instead of after a multi-megabyte upload.
    # FireSim evidence, when a results file exists for a model in this package. This is the only rung
    # where our own RTL executes the whole model, so it says something no simulator can: real cycle
    # counts, and a multicore split proven bit-identical on hardware.
    firesim_evidence = {}
    fs_dir = Path("/scratch2/agustin/merlin_firesim_builds")
    for model in models:
        f = fs_dir / f"results_{model}.json"
        if not f.is_file():
            continue
        try:
            rows = json.loads(f.read_text())
        except Exception:                                                   # noqa: BLE001
            continue
        keep = [{k: r.get(k) for k in ("harts", "cycles", "tier_ok", "w8a8_cos", "w8a8_max_rel")}
                for r in rows if r.get("cycles")]
        if not keep:
            continue
        by = {r["harts"]: r for r in keep}
        ent = {"bitstream": "alveo_u250_firesim_dual_saturn_v256d128", "vlen": 256, "runs": keep}
        if 1 in by and 2 in by and by[1]["cycles"] and by[2]["cycles"]:
            ent["speedup_1_to_2_harts"] = round(by[1]["cycles"] / by[2]["cycles"], 3)
            outs = {r["harts"]: r.get("outputs") for r in rows if r.get("outputs")}
            if 1 in outs and 2 in outs:
                ent["harts_bit_identical"] = outs[1] == outs[2]
        firesim_evidence[model] = ent
    if firesim_evidence:
        (dest / "firesim_evidence.json").write_text(json.dumps(firesim_evidence, indent=2) + "\n")
        print(f"  firesim_evidence.json: {sorted(firesim_evidence)}", flush=True)

    probe_report = None
    try:
        from merlin.runtime import vector_probe

        pwork = Path(tempfile.mkdtemp(prefix="delivery_probe_"))
        # The probe we CHECK speaks HTIF, because that is the console spike provides; the probe we
        # SHIP speaks the board's own. Same source, same -march, differing only in the linked console
        # object -- so the self-check below covers the CSR reads and the reporting, which is what can
        # be wrong about a probe, while the shipped twin is the one that can actually be heard.
        selfcheck_elf = vector_probe.build(pwork / "htif", dram_base=brd.dram_base,
                                           dram_bytes=brd.dram_bytes, vlen=brd.vlen)
        checks = {}
        for v in sorted({128, 256, brd.vlen or 128}):
            checks[v] = vector_probe.parse(
                vector_probe.run_on_spike(selfcheck_elf, vlen=v, dram_base=brd.dram_base))
        pelf = selfcheck_elf
        if brd.console != CONSOLE_HTIF:
            pelf = vector_probe.build(pwork / "board", dram_base=brd.dram_base,
                                      dram_bytes=brd.dram_bytes, vlen=brd.vlen,
                                      console=brd.console, sdk_dir=a.sdk_dir,
                                      sdk_chip=brd.sdk_chip, chip_freq_hz=brd.chip_freq_hz)
        shutil.copy2(pelf, dest / "vlen_probe.elf")
        probe_report = {"elf": "vlen_probe.elf", "bytes": (dest / "vlen_probe.elf").stat().st_size,
                        "console": brd.console,
                        "spike_selfcheck_console": CONSOLE_HTIF,
                        "spike_selfcheck": {str(k): v for k, v in checks.items()}}
        bad = [v for v, r in checks.items()
               if not (r.get("complete") and r.get("consistent") and r.get("vlen_bits") == v)]
        if bad:
            problems.append(f"vlen_probe.elf misreported the VLEN at {bad} — not shipping a probe "
                            f"whose answer we cannot trust")
            (dest / "vlen_probe.elf").unlink(missing_ok=True)
            probe_report = None
        else:
            print(f"  vlen_probe.elf: {probe_report['bytes']} bytes, correct at "
                  f"{sorted(checks)} on spike", flush=True)
    except Exception as exc:                                                # noqa: BLE001
        problems.append(f"vlen_probe: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")

    (dest / "elf_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (dest / "grade.py").write_text(GRADE_PY)
    (dest / "grade.py").chmod(0o755)
    manifest = {
        "board": {"name": brd.name, "dram_bytes": brd.dram_bytes, "harts": brd.harts,
                  "vlen": brd.vlen, "console": brd.console, "notes": brd.notes},
        "dtype": a.dtype, "binaries": binaries, "problems": problems,
        "vector_probe": probe_report,
        "firesim_evidence": firesim_evidence or None,
        "merlin_commit": _git_sha(),
        # State what actually happened to THESE binaries. A package whose provenance line claims a
        # simulation it skipped is worse than one that admits the gap: the reader cannot tell which
        # of our claims to trust.
        "validated_on": _provenance(a.no_spike, no_spike_models),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "README.md").write_text(_readme(brd, manifest))
    if manifest_writer is not None:
        for f in sorted(dest.iterdir()):
            manifest_writer.add_artifact(f.name)
        manifest_writer.write_manifest()
    print(f"[make_delivery] {len(binaries)} binaries -> {dest}")
    if problems:
        print("[make_delivery] PROBLEMS (recorded in manifest.json, not hidden):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    return 0 if binaries and not problems else 1


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=repo_root(), timeout=30).stdout.strip()[:12] or "unknown"
    except Exception:                                                       # noqa: BLE001
        return "unknown"


def _provenance(no_spike: bool, no_spike_models: set) -> str:
    """One line saying what actually happened to the binaries in THIS package.

    A package whose provenance claims a simulation it skipped is worse than one that admits the gap:
    the reader cannot tell which of our claims to trust. A mixed package says so, and names which
    models are the exception rather than averaging the two stories together.
    """
    rtl = ("FireSim (our own Saturn RTL, whole model, bit-exact vs the W8A8 reference)")
    if no_spike:
        return f"{rtl}; these binaries were built and ELF-audited but NOT simulated"
    spike = "spike (functional, at the board's VLEN)"
    if no_spike_models:
        return (f"{spike} — EXCEPT {', '.join(sorted(no_spike_models))}, which was built and "
                f"ELF-audited but not simulated; its evidence is {rtl}")
    return spike


def _upload_note(seconds) -> str:
    """Upload time, flagged when it exceeds a typical harness timeout.

    Not cosmetic: a 10-minute timeout is a common default, and an image whose UPLOAD alone takes longer
    is reported as a failure that no rebuild can fix. Measured: a 120 MB image needs ~24 minutes over a
    921600-baud TSI link, because the loader transmits MemSiz, not file size."""
    if seconds is None:
        return "n/a"
    mins = seconds / 60.0
    flag = "  **raise your timeout**" if seconds > 600 else ""
    return f"{mins:.0f} min{flag}"


def _verdict(b: dict) -> str:
    """What we can honestly say about one binary. `not simulated` is a distinct answer from a gate
    that ran and did not pass — conflating them would let a skipped check read as a soft failure."""
    if b.get("gate_ok"):
        return "PASS"
    return "not simulated" if b.get("spike_cycles") is None else "see status"


#: Prepended when the console had to be corrected. Someone who already tried an earlier build and got
#: nothing needs to know WHY before they spend bench time on another one -- and needs to be able to
#: check the claim rather than take it on faith.
SUPERSEDED_DOC = """\
## If you tried our earlier binaries and saw nothing

That was our bug, and this is the fix. The earlier images wrote their output over **HTIF**, which is a
*host-assisted* channel: the program writes a word to `tohost` and then waits for a host debugger to
acknowledge it by clearing it. On spike, on FireSim, and under `uart_tsi` there is such a host, so it
worked everywhere we could test. On the real chip with nothing attached, nothing ever clears `tohost` —
so the image emitted at most one character into a memory word no one reads and then **spun forever
inside its first print, before any model work ran.** From the outside that is indistinguishable from a
core that never booted, which is exactly what you saw. The chip's own UART was never initialised
either, so nothing would have come out of it regardless.

Your own code states the rule we broke — `c2c-demos/dsp-whisper/src/main.c`: *"init_test sets up the
PLL AND the UART divisor... It MUST run before any printf on silicon — the console UART is not usable
until then (a printf to it would hang the core)."*

These images speak your UART instead, and you can verify that without running them:

```bash
riscv64-unknown-elf-readelf -S <THIS.elf> | grep -c '\\.htif'   # the console is not HTIF
riscv64-unknown-elf-nm      <THIS.elf> | grep -c tohost        # nothing waits on a host
```

**The compiled model is unchanged** — same lowering, same schedule, same weights. Only the output
channel was wrong.

"""


def _readme(brd, manifest: dict) -> str:
    loader = LOADER_DOC.get(brd.name, LOADER_DOC["default"])
    superseded_doc = SUPERSEDED_DOC if brd.sdk_chip else ""
    baremetal = brd.flow == boards.FLOW_BAREMETAL
    image = ("bare-metal ELF (our own crt/linker script — your SDK has no RTOS)" if baremetal
             else "Zephyr image")
    hart_counts = sorted({b["harts"] for b in manifest["binaries"]})
    rows = "\n".join(
        f"| `{b['elf']}` | {b['model']} | {b['harts']} | "
        f"{'RVV' if b.get('backend', 'rvv') == 'rvv' else 'scalar'} | "
        f"{b['ram_bytes'] // 2**20} MB | {_upload_note(b['upload_estimate_s'])} | "
        f"{_verdict(b)} |"
        for b in manifest["binaries"])
    statuses = "\n".join(f"- **{m}** — {STATUS.get(m, 'no status recorded')}"
                         for m in sorted({b["model"] for b in manifest["binaries"]}))
    vlen = brd.vlen or 128
    paired = sorted({b["model"] for b in manifest["binaries"]
                     if sum(1 for o in manifest["binaries"] if o["model"] == b["model"]) > 1})
    unpaired = sorted({b["model"] for b in manifest["binaries"]} - set(paired))
    pair_section = ("""\
## Please run BOTH hart counts

For %s the package carries one binary per hart count, and the pair exists so you can check them
against each other: **the outputs must be bit-identical.** They are the same computation split
differently, so any difference at all is vector/SMP state on the SoC, not rounding — and that is a far
more useful signal for you than either run alone. %s%s""" % (", ".join("`%s`" % m for m in paired),
                         ("We verified this holds on spike at your vector length before shipping."
                          if any(b.get("spike_cycles") is not None for b in manifest["binaries"])
                          else "We verified this property on our own RTL rather than on these exact "
                               "images (see the FireSim table below): 1-hart and 2-hart outputs came "
                               "back bit-identical there."),
                         ("\n\n%s single-hart only: we could not build a multi-hart image for "
                          "%s, and would not ship one we had not run." %
                          (", ".join("`%s`" % m for m in unpaired)
                           + (" ships" if len(unpaired) == 1 else " ship"),
                           "it" if len(unpaired) == 1 else "them")) if unpaired else "")
                    if len(hart_counts) > 1 else """\
## Only one hart, deliberately

Every binary here runs on one hart. We did not ship a multi-hart bare-metal image because dispatching
your second hart goes through your own thread-lib rather than the OpenMP runtime our multicore lowering
targets, and shipping that untested would waste your bench time. If you want it, that integration is
ours to do next — say so and we will build against your dispatch.""")
    # A scalar image is a different ISA for a different set of harts, not a slower build of the same
    # thing. Someone looking at two files that differ by one word in the name needs that said.
    scalar_bins = [b for b in manifest["binaries"] if b.get("backend", "rvv") != "rvv"]
    if scalar_bins:
        n_vec = brd.n_vector_harts
        scalar_section = """\
## The `_scalar` binaries — how to use every core

Your chip brings up **%d** cores but only **%d** of them have a vector unit, so a vector image can only
use %d. The `_scalar` binaries contain **no vector instructions at all** (we check this before shipping:
zero `vsetvli`/`vfmacc`/`vle`/`vse` in the image), which means every hart can run them — that is the
only way to put all %d cores to work.

The trade is real and worth stating: scalar is much slower per core than RVV, so a %d-hart scalar image
is not automatically faster than a %d-hart vector one. Which wins is an empirical question on your
silicon, and it is one of the more interesting numbers you could send back. What the scalar image is
unambiguously good for is a core that would otherwise sit idle.

These are **not** simulated: there is no vector gate to run on them, so they ship built and ELF-audited
only, and the table says so.
""" % (brd.harts, n_vec, n_vec, brd.harts, brd.harts, n_vec)
    else:
        scalar_section = ""
    fs = manifest.get("firesim_evidence") or {}
    if fs:
        rows_fs = []
        for model, e in sorted(fs.items()):
            for r in e["runs"]:
                rows_fs.append(f"| {model} | {r['harts']} | {r['cycles']:,} | {r['tier_ok']} | "
                               f"{r['w8a8_max_rel']} |")
        extra = []
        for model, e in sorted(fs.items()):
            if e.get("speedup_1_to_2_harts"):
                extra.append(f"- `{model}`: **{e['speedup_1_to_2_harts']}×** on 2 harts, outputs "
                             f"{'bit-identical' if e.get('harts_bit_identical') else 'NOT identical'}"
                             f" to the 1-hart run")
        firesim_doc = ("""\
## What the same code did on real RTL (not a simulator)

These models also ran on **FireSim**, executing the Saturn RTL on an FPGA — our own SoC, whole model,
cycle-accurate. Bitstream `%s`, **vLen=256**:

| model | harts | cycles | gate | per-element error |
|---|---:|---:|---|---:|
%s

%s

Two things this establishes for you: the arithmetic is bit-exact against the W8A8 reference on real
hardware (`per-element error 0.0`, not just a good cosine), and the multicore split is a pure work
division rather than an approximation. It is *not* a claim about your chip's clock or memory system —
different SoC, different frequency.
""" % (list(fs.values())[0]["bitstream"], "\n".join(rows_fs), "\n".join(extra) or ""))
    else:
        firesim_doc = ""
    simulated = any(b.get("spike_cycles") is not None for b in manifest["binaries"])
    sim_line = ("""\
- Ran the image on spike at **%d-bit** vectors and gated the output against the W8A8 reference. That
  simulated image differs from the one you have in exactly one way: spike provides a debug host, so it
  was linked with the host-assisted console instead of your UART. Both share a `build_hash` — a digest
  of the lowered model object plus the weights — so the computation is byte-for-byte the same, and the
  packager refuses to ship the pair if those hashes disagree. Two consequences for the log you send
  back: your run prints two extra `METRIC` lines (`console`, `chip_freq_hz`) that the reference does
  not, and `METRIC cycles` will differ because it is a different chip. Grading reads the `OUT` line, so
  neither affects PASS/FAIL.""" % (brd.vlen or 128) if simulated else """\
- **Did NOT run these exact binaries through a functional simulator.** The lowering they were built
  from is validated on stronger evidence — the same models, same schedule, executing our own Saturn
  RTL on an FPGA, bit-exact against the W8A8 reference (see above). What is specific to these
  binaries is their memory map and link, which is what the ELF audit checks. That is why there is no
  `expected_console.txt` for them: we would rather ship no reference than one we did not produce.""")
    probe = manifest.get("vector_probe")
    probe_doc = ("""\
Load `vlen_probe.elf` the same way as any other binary below. It reads the chip's own CSRs and prints:

```
PROBE hartid <n>            which hart answered
PROBE misa_ext_bits <n>     the extension letters your hardware advertises
PROBE misa_v_bit <0|1>      does misa claim V at all
PROBE mstatus_vs <0..3>     is vector state ENABLED (0 = off; vector instructions would trap)
PROBE vlenb <bytes>         the authoritative vector length; VLEN bits = vlenb * 8
PROBE vlmax_e8 / vlmax_e32  the same width derived again via vsetvli, as a cross-check
DONE
```

**Please send this back before anything else.** Your board files declare no vector width anywhere
(`riscv,isa = "rv64gc"`, and the samples' `CONFIG_RISCV_VECTOR_MAX_LEN` sizes Zephyr's save area, so it
only bounds the real width from above), so we built the model images for %d-bit vectors as our best
inference. If `vlenb` says otherwise, the model binaries are still CORRECT — fixed-width vector code
runs at a lower LMUL on a wider unit — but they leave performance on the table, and we would rebuild.
It costs seconds and saves a multi-megabyte upload spent on the wrong assumption.

If `mstatus_vs` comes back 0, stop there and tell us: vector state is off, and every vector
instruction in the model images would trap. The probe deliberately stops before reading `vlenb` in
that case rather than taking the trap.""" % (brd.vlen or 128)
                 if probe else "*(not included in this package)*")
    identity_line = ("- Confirmed 1-hart and %d-hart outputs are bit-identical." % max(hart_counts)
                     if len(hart_counts) > 1 else
                     "- Single-hart images only (see above), so there is no hart-split to check.")
    return f"""\
# Merlin int8 RVV binaries for `{brd.name}`

Compiled by the Merlin flow (model2MLIR capture -> int8 W8A8 lowering -> certified RVV schedule ->
{image}). **Each binary is self-contained**: inputs and weights are baked in, so there is no
filesystem, no host I/O, no arguments, and nothing to install on the board.

We have no access to this board — you running these is the first time they touch the real chip.

## What we built for

| fact | value | where it came from |
|---|---|---|
| DRAM | {brd.dram_bytes // 2**20} MB at `{hex(brd.dram_base)}` | your chip — {'the linker scripts in your repo declare less' if baremetal else 'the DTS in your repo declares less'} |
| harts on the chip | {brd.harts} | your chip |
| harts these binaries use | {', '.join(str(h) for h in hart_counts)} | {'single-hart only — see below' if len(hart_counts) == 1 else 'one binary per count'} |
| vector length | {vlen} bits{'' if brd.vlen else ' (assumed — the V minimum, since nothing declares it)'} | {'stated in your repo' if brd.vlen else 'NOT declared anywhere in the board files'} |
| console | {brd.console} | {'derived from your SDK headers' if brd.sdk_chip else ('your `chip_config.h` / SIMS platform' if baremetal else 'board DT')} |

**If any of those is wrong, tell us** — a mismatch is the most likely cause of a silent hang, and each
one is a one-line rebuild on our side. In particular we would like to know `vlenb` on the real chip.

{superseded_doc}## The binaries

| file | model | harts | ISA | linked region | est. upload | our verdict |
|---|---|---|---|---|---|---|
{rows}

Upload time is the *memory* size, not the file size — a UART loader transmits `MemSiz`, so a big
embedded weights blob costs minutes per attempt.

## Run this FIRST: `vlen_probe.elf` (~10 KB, seconds)

{probe_doc}

## Run one

{loader}

## Send back

The console text, verbatim, from the banner through `DONE`. That is enough for us to grade it: it
carries the output, the cycle counts, and a `METRIC build_hash` that identifies exactly which binary
produced it.

You can also grade it yourself, offline, with only numpy:

```bash
python grade.py <your_console_log.txt> --model spectformer
```

It prints `PASS`/`FAIL` with the cosine and per-element error against the same references we use, and
warns if the log's `build_hash` is not one of ours.

{pair_section}
{scalar_section}

{firesim_doc}
## Status of each model — please read before reporting a number

{statuses}

## What we already checked, without the board

- Built for `{brd.name}` with the board's own facts, and audited the ELF against them: every LOAD
  segment inside your DRAM, entry point in range, `.htif` present so your loader can find
  `tohost`/`fromhost`, and real vector instructions in the image (see `elf_audit.json`).
{sim_line}
{identity_line}

What that does *not* cover: your clock, your DRAM timing, your vector unit's actual VLEN, and anything
about wall-clock performance. spike is functional — it proves correctness, never speed.

Merlin commit `{manifest['merlin_commit']}`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
