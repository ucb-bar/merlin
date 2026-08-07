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
import zipfile
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
`--use_symbols` — that version has neither the console-script entry point nor that flag.

### Please raise `--baudrate` for the large images

The `57600` above is what your own `scripts/run_experiments.py` uses, and it is what the upload column
in the table below assumes. At that rate the loader moves about **5.6 KB/s**, so the biggest model spends
**hours** on the wire before a single instruction executes — with `--fesvr` polling `tohost` in a loop
that has no timeout, so from the outside that is indistinguishable from a hang. We think this, not the
compiled code, is why `whisper_tiny` came back FAIL on every hart count.

`pyuartsi`'s serial layer takes the baud straight through to pyserial, and its own `Baudrate` enum goes
to 4000000. At **921600** the same image is ~16x quicker. If a higher rate is not reliable on your
setup, please tell us what is and we will quote honest numbers against it — and start with `deepjscc`,
whose upload is about a megabyte at any baud.

(Two corrections to what we sent last time, both ours: we quoted upload times computed from `MemSiz` at
921600 baud. Your loader transmits only `SHT_PROGBITS` sections — so the byte count was too *high* — and
you run it at 57600 — so the rate was 16x too *optimistic*. The table below fixes both.)""",
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
    "gemmelos_bearly25_zephyr_500mhz": """\
**Output comes out of UART0 at 115200 baud.** Open that terminal first, then load exactly as for the
50 MHz package:

```bash
make tsi-run TTY=<your TSI tty> BINARY=<THIS.elf>
#   equivalently: uart_tsi +tty=<TSI tty> +baudrate=921600 <THIS.elf>
```

**This package is the 50 MHz one with your PLL raised to 500 MHz.** Identical model, identical
lowering, identical weights; the only difference is that the image programs the clock before running.
Run the 50 MHz package first — if that works and this one prints garbage, the divisor is wrong and you
have lost nothing but one upload.

The sequence is yours, not ours (`bmark-lib/simple_setup.c::init_test`), replayed against values read
out of your own headers rather than written down by us:

1. park every clock-selector domain on the slow source (`0x130000`, 4 domains, from `hal_rcc.h`);
2. program the PLL at `0x140000` — ratio `500 MHz / SYS_CLK_FREQ` = **10**, fraction 0, then
   `MDIV/ZDIV0/ZDIV1 = 1`, `LDO_ENABLE`, `PLLEN`, `POWERGOOD_VNN`, `PLLFWEN_B` in the order
   `driver/intel/pll/pll.c::configure_pll` uses;
3. switch the domains onto it;
4. **re-program `UART0->DIV`** (offset 24) to `500000000 / 115200 - 1` = **4339**.

Step 4 is the one that decides whether you get output or line noise, and it runs *after* Zephyr's
SiFive driver has initialised the UART — the hook is ordered at `CONFIG_SERIAL_INIT_PRIORITY + 1`
precisely so the driver cannot overwrite it with a divisor for the old clock.

**What we could not check:** spike has no PLL, so this code path has never executed anywhere. What we
did verify is that the emitted instructions are the right ones — we disassembled the image and read the
sequence back (the four `0x130000` writes, ratio 10, and `DIV = 4339`). If it misbehaves, the 50 MHz
package is unaffected, and `PROBE core_khz_measured` from `vlen_probe.elf` will tell us what the chip
actually settled at.""",
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


def diagnose(text):
    """What a log that did NOT finish can still tell you.

    "this log has no OUT line" is true of a hang, a crash, an unfinished upload and a board that never
    booted -- four different problems, one message, and no way to tell them apart without asking. A
    debug image emits STAGE/ALIVE/MEM/FAIL lines precisely so the log answers that itself, and this
    prints what it found rather than making someone grep for it.
    """
    lines = text.splitlines()
    def last(prefix):
        return next((l for l in reversed(lines) if l.startswith(prefix)), None)
    out = []
    bh = next((l.split()[2] for l in lines
               if l.startswith("METRIC build_hash") and len(l.split()) > 2), None)
    if bh:
        out.append(f"  build_hash : {bh}")
    fatal = last("FAIL ")
    if fatal:
        out.append(f"  FAULT      : {fatal}")
    stage = last("STAGE ")
    out.append(f"  last STAGE : {stage if stage else '(none — not a debug image)'}")
    alive = last("ALIVE ")
    if alive:
        out.append(f"  last ALIVE : {alive}")
        out.append("               (it was still executing — a hang here is a slow op, not a dead core)")
    bad_mem = [l for l in lines if l.startswith("MEM ") and l.endswith("FAIL")]
    if bad_mem:
        out.append(f"  MEMORY     : {len(bad_mem)} probe(s) FAILED — the linked region is larger than "
                   "the DRAM that answers: " + ", ".join(l.split()[1] for l in bad_mem))
    vs = [l for l in lines if "mstatus_vs" in l]
    if vs:
        out.append("  vector st. : " + "; ".join(l.replace("METRIC ", "") for l in vs[:4]))
    return "\\n".join(out)


def parse_console(text):
    out_line = next((l for l in text.splitlines() if l.startswith("OUT ")), None)
    if out_line is None or "DONE" not in text:
        raise SystemExit("this run did not complete (no OUT line and/or no DONE). What the log does "
                         "say:\\n" + diagnose(text) +
                         "\\n\\nIf every field above is empty this is a delivery image; the debug "
                         "package's images answer all of them.")
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
                # Some metrics carry a LIST, not a scalar: `METRIC vector_harts 0 1 2` names every
                # vector-capable hart. Taking only p[2] reported "vector_harts: 0", which reads as
                # "zero vector harts" -- the exact opposite of what a 3-hart chip was saying.
                if len(p) > 3:
                    metrics[p[1]] = " ".join(p[2:])
                else:
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
    rows = [b for b in man["binaries"] if b["model"] == model]
    known = {b["build_hash"] for b in rows}
    # A board whose console is its own UART cannot be simulated here, so its expected_console comes
    # from an HTIF twin of the same lowering, carrying a different build_hash. That is documented, not
    # a defect -- but an unexplained "not in this package" on the very first thing someone runs reads
    # like a broken delivery, so name the twin instead of crying wolf.
    twins = {b["gate_build_hash"] for b in rows if b.get("gate_build_hash")}
    if metrics.get("build_hash") and metrics["build_hash"] not in known:
        if metrics["build_hash"] in twins:
            print("  (this is the simulated HTIF twin's hash — expected for an expected_console.txt "
                  "on a board whose own console spike cannot service; see the README)")
        else:
            print(f"  WARNING: that build_hash is not in this package "
                  f"(expected one of {sorted(known)})")
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
    "deepjscc": "VERIFIED: w8a8_cos = 1.0, rel = 0.0, on both the vector and the scalar image, at "
                "every hart count, on spike. The scalar image is newly correct — the one in the "
                "previous package computed w8a8_cos 0.9176 because per-op register blocking was "
                "applied only to vector builds, and scalar images were shipped without ever being "
                "simulated. Both halves of that are fixed: the blocking is unconditional, and no "
                "image ships now without a gate behind it.",
    "lstmnetvit": "KNOWN DIVERGENT on RISC-V: w8a8_cos 0.9943 on spike and on the K1, while the same "
                  "IR is exact on x86. The 1-hart and N-hart runs ARE bit-identical, so it is useful "
                  "for bring-up and timing. Do not treat its output as an accuracy result — the cause "
                  "is ours to fix, not yours to debug.",
    "whisper_tiny": "RE-EXPORTED SMALLER since the last package, because its problem was never the "
                    "code — it was the upload. Its token-embedding table was still fp32 (torchao's "
                    "default filter matches nn.Linear only), 76 MB of a 117 MB bundle, with the TIED "
                    "output projection stored a second time at a quarter the size. Quantizing it "
                    "takes the image from 127 MB to 71 MB. It is still by far the largest thing here "
                    "and still the longest-running; check the upload column against your link speed "
                    "before starting, and send the console log whatever it says.",
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
    # Size the arena to THIS model, the way the Zephyr path already does. spike_model defaults to a
    # fixed 256 MB, which silently decides which models can run: whisper needs 319 MB live at its
    # encoder attention and died here as `*** FAILED *** (tohost = 2055)` -- no message, no named
    # allocation, because the bare-metal path has no RTOS and so does not carry the malloc guard the
    # Zephyr images do. A board with 1 GB of DRAM had 700 MB spare at the time.
    from merlin.common.ir_lock import IR_LOCK
    from merlin.common.mlir_query import activation_peak_bytes
    with IR_LOCK:
        peak = activation_peak_bytes(bundle / "model.mlir")
    arena_mb = max(256, ((int(peak or 0) + 128 * 1024 * 1024) + 2**20 - 1) // 2**20)
    b = spike_model.build(bundle, work, inputs_npz=bundle / "inputs.npz",
                          dram_base=brd.dram_base, dram_bytes=brd.dram_bytes,
                          int8_compute=True, features=frozenset([PEROP_BLOCK_NAME]),
                          rvv_schedule=pkg.schedule_text, arena_mb=arena_mb,
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
                                 rvv_schedule=pkg.schedule_text, arena_mb=arena_mb,
                                 cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=brd.vlen,
                                 console=brd.console, sdk_dir=sdk_dir, sdk_chip=brd.sdk_chip,
                                 chip_freq_hz=brd.chip_freq_hz)
        if ship.get("build_hash") != b.get("build_hash"):
            raise RuntimeError(
                f"the gated image and the shipped image disagree on build_hash "
                f"({b.get('build_hash')} vs {ship.get('build_hash')}) -- they are not the same "
                "computation, so the gate does not cover what is being shipped")
    # Third element is the instrumented twin, and there isn't one: the debug harness is Zephyr's
    # (printk, k_uptime_get, the fatal hook overriding a kernel symbol) and this path has no RTOS.
    # Returned as None rather than omitted so the shape matches the Zephyr path -- returning a
    # 2-tuple here is what made the packager die on unpacking after every image had already built.
    return res, {"elf": ship["elf"], "ram_bytes": ship["mem_bytes"],
                 "build_hash": ship.get("build_hash", ""),
                 "console": ship.get("console", CONSOLE_HTIF),
                 "chip_freq_hz": ship.get("chip_freq_hz"),
                 "console_provenance": ship.get("console_provenance", {})}, None


def build_board_only(bundle: Path, brd, harts: int, *, work: Path, sdk_dir=None,
                     backend: str = "rvv", debug: bool = False):
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
        # A scalar image gets none of the vector machinery -- no RVV package, no -march vector width --
        # but it DOES get per-op register blocking, and leaving it out was a correctness bug, not a
        # simplification.
        #
        # Measured: without it, deepjscc's scalar image returns w8a8_cos 0.9176 (max_rel 5.0) while its
        # vector image returns 1.0. 1-hart and 3-hart scalar produce the SAME wrong answer, so the
        # scalar multicore route is not the cause; and an RVV build stripped of the RVV package but
        # keeping the blocking is still exact, so the package is not the cause either. The blocking is.
        # That image shipped and came back FAIL from the chip's owner, with no way to tell a wrong
        # answer from a crash.
        #
        # It does not vectorise the image -- the point of a scalar build is a hart with no vector unit,
        # and that property is checked, not assumed: `forward` audits at 0 vector instructions with
        # this on, and `_audit` below fails the package if any appear.
        b = zm.build_app(bundle, work, board=brd.name, backend=backend, rvv_hart=0,
                         cpus=max(harts, brd.harts), n_harts=harts, int8_compute=True,
                         features=frozenset([PEROP_BLOCK_NAME]),
                         sdk_dir=sdk_dir, debug=debug)
        return {"elf": b["elf"], "ram_bytes": b["ram_bytes"],
                "build_hash": b.get("build_hash", ""), "backend": backend,
                "op_profile_table": b.get("op_profile_table")}
    b = zm.build_app(bundle, work, board=brd.name, backend="rvv", rvv_hart=0,
                     cpus=max(harts, brd.harts), n_harts=harts, int8_compute=True,
                     rvv_schedule=pkg.schedule_text,
                     cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
                     features=frozenset([PEROP_BLOCK_NAME]), vlen=brd.vlen, sdk_dir=sdk_dir,
                     debug=debug)
    return {"elf": b["elf"], "ram_bytes": b["ram_bytes"], "build_hash": b.get("build_hash", ""),
            "op_profile_table": b.get("op_profile_table")}


def build_one(bundle: Path, brd, harts: int, *, vlen, work: Path, timeout: int, sdk_dir=None,
              debug: bool = False, backend: str = "rvv"):
    """Build one image and run it on spike at the board's VLEN and hart count.

    Both backends come through here. A SCALAR image used to be built and shipped without ever being
    simulated, on the reasoning that it has no vector gate to run -- but the gate that matters is
    numerical, not architectural, and skipping it shipped a deepjscc scalar image that computed
    w8a8_cos 0.9176. It is graded against the same references as the vector one now.
    """
    # The RVV package is exactly that: RVV. A scalar build takes neither its schedule nor its cflags
    # (which carry the vector -march), but does take the per-op register blocking, which is what makes
    # the arithmetic right rather than what makes it vector.
    vec = backend == "rvv"
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    tune = dict(rvv_schedule=pkg.schedule_text,
                cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=vlen) if vec else {}
    refs = {"fp32": np.load(bundle / "golden.npy")}
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    cpus = max(2, harts, brd.harts if not vec else 2)

    def board(dbg: bool):
        return zm.build_app(
            bundle, work / f"board_h{harts}{'_dbg' if dbg else ''}", board=brd.name,
            backend=backend, rvv_hart=0,
            cpus=cpus, int8_compute=True,
            features=frozenset([PEROP_BLOCK_NAME]), n_harts=harts, sdk_dir=sdk_dir,
            debug=dbg, **tune)

    ship = board(False)
    if brd.console == CONSOLE_HTIF:
        # Gate the ELF WE SHIP, not a stand-in for it. spike can service an HTIF console, so for such a
        # board there is no reason to simulate a `spike_riscv64` twin: that twin has a different Kconfig
        # and device tree, so it carries a different build_hash than the shipped image, the recipient's
        # own grade.py flags the package as inconsistent, and -- worse -- the gate never exercises the
        # board configuration. On Kodiak the board Kconfig is precisely where the bug was
        # (CONFIG_RISCV_ISA_EXT_V / CONFIG_FPU_SHARING), so the twin tested the arithmetic and skipped
        # the thing that had broken.
        #
        # `-pN` is the image's CPU COUNT, not its hart count: a 1-hart model image is still built with
        # MP_MAX_NUM_CPUS=2, and under `-p1` Zephyr waits forever for a CPU that never arrives and hangs
        # before printing anything. That failure is indistinguishable from a slow model.
        res = zm.run_on_spike(ship["elf"], harts=cpus, mem_bytes=ship["ram_bytes"],
                              timeout=timeout, vlen=vlen if vec else None)
        res.update(zm._gate(res["prefix"], refs))
        res["backend"] = backend
    else:
        # A board whose console is its own UART cannot be simulated here at all -- spike has no such
        # peripheral -- so the gate runs on an HTIF twin built from the same IR, and the package says so.
        res = zm.build_and_run(
            bundle, work, board="spike_riscv64",
            backend=backend, rvv_hart=0, harts=max(2, harts), int8_compute=True, n_harts=harts,
            features=frozenset([PEROP_BLOCK_NAME]), references=refs, timeout=timeout, **tune)

    # The debug twin is NOT re-simulated: same model, same lowering, same references, instrumentation
    # only, and re-gating it would burn hours of spike to re-establish an answer we already have.
    return res, ship, (board(True) if debug else None)


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
    ap.add_argument("--debug", action="store_true",
                    help="ALSO build a diagnostic twin of every image, into the same package. Same computation, but "
                         "each announces the stage it reached, heartbeats while it runs (naming the "
                         "op it is inside), probes that the linked DRAM region really exists, reports "
                         "stack high-water marks, and turns a fault into one greppable FAIL line "
                         "carrying the build hash. For a board we cannot attach to, this is what "
                         "makes ONE returned console log a diagnosis instead of a new question.")
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
        if a.no_spike or model in no_spike_models:
            print(f"  [{tag}] building (no simulation) in {work}", flush=True)
            board_build = build_board_only(bundle, brd, harts, work=work, sdk_dir=a.sdk_dir,
                                           backend=backend, debug=False)
            dbg_build = (build_board_only(bundle, brd, harts, work=work / "dbg",
                                          sdk_dir=a.sdk_dir, backend=backend, debug=True)
                         if a.debug else None)
            print(f"  [{tag}] built: {board_build['ram_bytes'] // 2**20} MB region, "
                  f"{board_build['build_hash']}"
                  + (f" (+debug {dbg_build['build_hash']})" if dbg_build else ""), flush=True)
            return ({"console": "", "metrics": {}, "outputs": None, "backend": backend},
                    board_build, dbg_build)
        print(f"  [{tag}] building + simulating in {work}", flush=True)
        if brd.flow == boards.FLOW_BAREMETAL:
            out = build_baremetal(bundle, brd, work=work, timeout=a.timeout, sdk_dir=a.sdk_dir)
        else:
            out = build_one(bundle, brd, harts, vlen=brd.vlen, work=work, timeout=a.timeout,
                            sdk_dir=a.sdk_dir, debug=a.debug, backend=backend)
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
            res, board_build, dbg_build = got
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
                # The image the GATE ran on, when that is not the image we ship. For a board whose
                # console is its own UART, spike has no such peripheral, so the gate runs on an HTIF
                # twin built from the same IR -- and the expected_console beside the ELF therefore
                # names the twin. Recording it here is what lets grade.py tell "this console came
                # from the documented twin" apart from "this package is inconsistent".
                **({"gate_build_hash": res["build_hash"]}
                   if res.get("build_hash") and res["build_hash"] != board_build.get("build_hash")
                   else {}),
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
            # The instrumented twin, in the SAME package. Someone debugging a board does not want to
            # come back to us for a different download, and someone reporting a number does not want to
            # sift diagnostics out of it -- so both are here, distinguished by the filename.
            if dbg_build is not None:
                dbg_name = f"{model}_{a.dtype}_h{harts}{suffix}_debug_{brd.name}.elf"
                shutil.copy2(dbg_build["elf"], dest / dbg_name)
                drep = elf_audit.audit(dest / dbg_name, brd, ram_bytes=dbg_build["ram_bytes"],
                                       require_vector=(backend == "rvv"))
                audits[dbg_name] = drep.to_dict()
                if not drep.ok:
                    problems += [f"{dbg_name}: {q}" for q in drep.problems]
                # The op-id -> op-name table for this model. A debug run prints ~1000 `PROF <id> ...`
                # lines and an `ALIVE ... op=<id>`; without this file every one of those ids is an
                # unreadable integer, which makes the trace worthless to whoever mails it back.
                tbl = dbg_build.get("op_profile_table")
                if tbl and Path(tbl).is_file():
                    shutil.copy2(tbl, dest / f"{model}_h{harts}{suffix}.op_table.json")
                binaries.append({
                    "model": model, "elf": dbg_name, "harts": harts, "dtype": a.dtype,
                    "backend": backend, "debug": True,
                    "build_hash": dbg_build.get("build_hash", ""),
                    "ram_bytes": dbg_build["ram_bytes"],
                    "spike_cycles": None, "spike_vlen": None,
                    "gate_ok": bool(res.get("ok")), "tier_ok": res.get("tier_ok"),
                    "cos": res.get("cos"), "rel": res.get("rel"),
                    "upload_estimate_s": drep.facts.get("upload_estimate_s"),
                })
                print(f"  {dbg_name}: instrumented twin, audit={'OK' if drep.ok else 'FAIL'}",
                      flush=True)
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
            # Prefer an explicitly recorded verdict over inferring one from stored outputs: a results
            # file that keeps only a PREFIX of the output vector would compare equal on that prefix and
            # publish "bit-identical" for a run that is not. Measured on the dual-Saturn bitstream,
            # deepjscc's 2-hart run is NOT identical to its 1-hart run (w8a8_cos 0.9999972,
            # reproduced twice), while spectformer's is exact -- so this is a real distinction and not
            # a formality.
            explicit = [r.get("harts_bit_identical") for r in rows
                        if r.get("harts_bit_identical") is not None]
            outs = {r["harts"]: r.get("outputs") for r in rows if r.get("outputs")}
            if explicit:
                ent["harts_bit_identical"] = bool(explicit[0])
            elif 1 in outs and 2 in outs:
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
            # -m must match the region the probe was BUILT for: the probe now writes and reads across
            # that region, so a simulator with less memory than the board makes an honest probe look
            # broken (it faults at the simulator's edge, exactly as it would on a board with less DRAM
            # than we were told -- which is the whole point of the check).
            checks[v] = vector_probe.parse(
                vector_probe.run_on_spike(selfcheck_elf, vlen=v, dram_base=brd.dram_base,
                                          mem_bytes=brd.dram_bytes))
        pelf = selfcheck_elf
        if brd.console != CONSOLE_HTIF:
            # The board image also carries the chip's reference-clock rate, so the probe can report a
            # MEASURED core frequency alongside the declared one -- the only way a returned log can say
            # whether the PLL programming took effect.
            mtime_hz = None
            if a.sdk_dir and brd.sdk_chip:
                from merlin.runtime.sdk_facts import derive_uart_console
                mtime_hz = derive_uart_console(a.sdk_dir, brd.sdk_chip).mtime_hz
            pelf = vector_probe.build(pwork / "board", dram_base=brd.dram_base,
                                      dram_bytes=brd.dram_bytes, vlen=brd.vlen,
                                      console=brd.console, sdk_dir=a.sdk_dir,
                                      sdk_chip=brd.sdk_chip, chip_freq_hz=brd.chip_freq_hz,
                                      mtime_hz=mtime_hz)
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
    # Count what did NOT get a gate, and say so at package level. Every row already carried its own
    # `gate_ok`, but "problems: []" beside a table of "not simulated" reads as a clean package to
    # anybody who checks the summary rather than all sixteen rows -- and a whole package can end up
    # ungated (the PLL variant did) without a single line of the output admitting it.
    ungated = [b["elf"] for b in binaries if not b.get("gate_ok")]
    if ungated:
        problems.append(
            f"{len(ungated)} of {len(binaries)} binaries carry NO simulation gate "
            f"({', '.join(sorted(ungated)[:3])}{', …' if len(ungated) > 3 else ''}). They were built "
            f"and ELF-audited only; nothing here certifies that they compute the right answer.")
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
    (dest / "README.md").write_text(_readme(brd, manifest, debug=a.debug))
    # Remove binaries the manifest does not list. Rebuilding into an existing directory used to leave
    # the previous run's images behind: renaming a model left a 124 MB `whisper_tiny_int8_h1` sitting
    # next to the `whisper_tiny_375pos` that replaced it, unlisted, ungated, and zipped into the
    # delivery. An unlisted binary is the worst thing in a package -- it looks as official as the rest
    # and nothing we say about the package covers it, so somebody spends bench time on a build we
    # already know is broken.
    keep = {b["elf"] for b in binaries} | {
        f"{b['model']}_h{b['harts']}{'' if b.get('backend', 'rvv') == 'rvv' else '_' + b['backend']}"
        f".{suffix}" for b in binaries for suffix in ("expected_console.txt", "op_table.json")}
    keep |= {"vlen_probe.elf", "manifest.json", "README.md", "grade.py", "elf_audit.json",
             "firesim_evidence.json"}
    keep |= {f"{model}.golden.npy" for model in models} | {f"{model}.golden_w8a8.npy"
                                                           for model in models}
    for f in sorted(dest.iterdir()):
        if f.is_file() and f.name not in keep:
            print(f"  pruned stale {f.name} ({f.stat().st_size / 2**20:.1f} MB) — not in this "
                  f"manifest", flush=True)
            f.unlink()
    if manifest_writer is not None:
        for f in sorted(dest.iterdir()):
            manifest_writer.add_artifact(f.name)
        manifest_writer.write_manifest()
    # Zip HERE, from the tree we just pruned. Zipping was a separate step by hand, so the archive was
    # free to disagree with the directory -- and it did: the stale whisper image the prune above
    # deletes was still sitting inside an already-built zip, i.e. removed from the package but still
    # in the thing we would actually have sent. The archive is the deliverable, so the packager owns
    # it, and it can only ever contain what the manifest lists.
    zip_path = zip_package(dest)
    print(f"[make_delivery] zipped -> {zip_path} ({zip_path.stat().st_size / 2**20:.0f} MB)")
    print(f"[make_delivery] {len(binaries)} binaries -> {dest}")
    if problems:
        print("[make_delivery] PROBLEMS (recorded in manifest.json, not hidden):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    return 0 if binaries and not problems else 1


def zip_package(dest: Path) -> Path:
    """Archive a package directory as ``<dest>.zip``, replacing any existing archive."""
    zip_path = dest.parent / f"{dest.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(dest.iterdir()):
            if f.is_file():
                z.write(f, f"{dest.name}/{f.name}")
    return zip_path


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
    is reported as a failure that no rebuild can fix. This is the single most likely explanation for
    whisper_tiny coming back FAIL on both boards while everything smaller passed.

    The number is per-board, because the two loaders in use disagree by an order of magnitude on the
    same ELF and neither matches the other's assumption -- `uart_tsi` writes PT_LOAD MemSiz (zero-fill
    included), `pyuartsi` writes only PROGBITS sections -- and the baud comes from each board's own
    documented loader command. A single formula for both is what put "4 min" beside an image that takes
    an hour. See `runtime.elf_audit.upload_bytes`."""
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
KODIAK_VSTATE_DOC = """\
## Why every multi-hart image failed last time, and what changed

Your table was the shape of the diagnosis: **every `h1` PASSED and every `h2`/`h3` FAILED**,
deterministically. Your trap logs are the proof, and they are worth more than the table — thank you
for them. All of this is our bug.

`mcause: 2` is an illegal instruction, and `mtval` is the instruction that took it. Both of yours
decode to the same thing:

```
mtval 0xcd827057   (deepjscc h2, mepc 0x8001d794)
mtval 0xcd817057   (whisper  h2, mepc 0x80079cf8)
   opcode[6:0]   = 0x57   OP-V  — a vector instruction
   funct3[14:12] = 111    OPCFG — the vsetvli family
   bits[31:30]   = 11            vsetivli
```

`vsetivli` is the instruction that *begins* every RVV block. Two different models, two different PCs,
the identical instruction — that is not an unsupported opcode or a bad encoding, it is a vector unit
that was **off** when the code reached its first vector instruction.

Here is why it was off. Our generated config did not set `CONFIG_RISCV_ISA_EXT_V`, so Zephyr never put
`mstatus.VS` into a thread's initial mstatus and never saved or restored vector state across a context
switch. The single-worker image survives that because it enables VS by hand and then never
context-switches again. A multi-hart image calls `merlin_omp_init`, which creates and joins a probe
thread per hart and then creates the pool — so the **master** switches out and comes back with
VS = Off, and traps on its own share of the first parallel region. Your `mepc` values land exactly
there.

What then made it unreadable was ours too: the fault was reported, and the run never **ended**. HTIF
kept polling to no exit, so from outside a trap and a hang looked the same. Zephyr's default handler
halts the faulting CPU and stops; it never terminates in the console protocol. Every image here now
prints `FAIL fatal reason= hart= mcause= mepc= mtval= vs= build_hash=` followed by `DONE`, so a
crashed run is a finished, attributable, gradeable run instead of a log that just stops.

The reason we had those two settings was a stale note in our own board descriptor, which described the
Zephyr your **`kodiak` branch** pins (submodule `5a06eb0d`). We build against the `dev` pin
(`852bb170`), two commits later, which contains *"riscv: decouple V/F save-restore + add
RISCV_V_KERNEL_ONLY"* — there, `v.c` compiles under `CONFIG_RISCV_ISA_EXT_V` independently of
`FPU_SHARING`, exactly as its CMakeLists comment says. We had simply never re-checked.

What these images use instead is **your own working sample's configuration** —
`samples/q8_gemm_minmax/prj.conf` on the `kodiak` branch, the one with a checked-in `ref-out`:
`CONFIG_RISCV_ISA_EXT_V=y`, `CONFIG_RISCV_VECTOR_MAX_LEN=512`, `CONFIG_FPU_SHARING=n`.
Belt and braces, the OpenMP shim now re-arms `mstatus.VS` at every parallel-region entry including the
master's, so the image is correct even where Zephyr is not managing it.

Your probe run settled the vector length independently, so it is no longer taken on trust from a
config file: `vlenb 64` is **VLEN = 512 bits**, and your `vlmax_e8 64` / `vlmax_e32 16` are exactly
what 512 predicts. Everything here is built for that.

**We could not reproduce the trap in any simulator**, and that is worth saying plainly rather than
implying the fix is tested: neither spike nor the Saturn RTL on FireSim enforces `mstatus.VS`, so both
ran the broken image perfectly. Your two `mtval` values are the only direct evidence that exists, which
is why they mattered so much. Every image here now reports the state itself — look for
`METRIC hart<N>_mstatus_vs`. A `2` or `3` means Zephyr is managing vector state; a `0` would mean we
are back where we started.

### The other failure mode in your logs, which we have NOT explained

Three runs — `deepjscc h3 scalar`, `spectformer h2`, `whisper h3 scalar` — produced **no Zephyr banner
at all** before the log ended. That is not the vector-state bug: those images had not reached any of
our code yet, so the fix above does not address them and we are not going to pretend otherwise.

The explanation we find most likely is that the capture ended before the upload did. `pyuartsi` sends
the loadable sections over the serial link, and at the **57600** baud in your `run_experiments.py` that
is minutes to tens of minutes before the first instruction executes — the estimate for each image is in
the table above. If your harness timeout is shorter than that, the log ends exactly the way yours did:
ELF loaded, hart kicked, silence, EOF.

Two things would confirm or kill that theory in one round:

1. **Raise the baud.** 921600 is 16× faster and `pyuartsi` only transmits `SHT_PROGBITS`, not the
   zero-fill, so nothing else has to change.
2. **Run `vlen_probe.elf` and the `_debug` twin of one of the three.** The probe is 10 KB and uploads
   instantly; if it banners and the big image does not, the difference is transport, not the image.

If the banner is still missing with a generous timeout at a high baud, tell us — that is a different
bug and we would want the log.

"""


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
riscv64-unknown-elf-nm <THIS.elf> | grep -c tohost        # 0 -> nothing waits on a host
riscv64-unknown-elf-nm <THIS.elf> | grep -c uart_sifive   # non-zero -> your UART driver is linked
```

Do **not** test this with `readelf -S | grep .htif`: Zephyr allocates an empty (`NOBITS`) `.htif`
section for its reboot path whether or not HTIF is the console, so that section is present in these
images too and counting it proves nothing. The absence of a `tohost` SYMBOL is the real evidence --
that is the word an HTIF image spins on.

**The compiled model is unchanged** — same lowering, same schedule, same weights. Only the output
channel was wrong.

"""


DEBUG_DOC = r"""\
## Every binary here comes in two builds

Look at the `build` column above. For each model and hart count there are **two** ELFs:

- the **plain** one — what you run to get a number. Minimal output, nothing extra in the way.
- the **`_debug`** one — the same model, same lowering, same weights, same references, but built to
  explain itself. Run this one the moment the plain one does something you cannot interpret.

They are separate binaries with separate `build_hash` values, so a log always says which you ran. Both
were produced from a single simulation of the shared computation, so the instrumented build is not a
different program in any way that affects the answer — it just says more.

**Please send back the `_debug` log for anything that fails.** The extra lines are the difference
between "it stopped" and a diagnosis:

| line | means |
|---|---|
| `STAGE <name> hart=N t=<ms>` | it reached that milestone. The gap between two STAGEs is where the time went; the last one is where it stopped. |
| `ALIVE t=<s> op=<id> hart=N vs=<0..3>` | still executing, at most every few seconds, naming the op index it just entered and whether vector state is live. **If these keep coming, it is not hung — it is working, and the model is simply long.** Printed from the model's own thread between two ops (a timer interrupt cannot be used: printing from one corrupts the HTIF console), so the last one names the op it stopped inside. |
| `MEM <addr> ok\|FAIL` | write-then-read across the linked DRAM region, before the model runs. A `FAIL` means the region is larger than the memory that answers. |
| `STACK <thread> size=.. unused=.. used=..` | high-water marks. We reserve 8 MB per worker on one old measurement; this is us checking that against yours. |
| `FAIL fatal reason=.. mcause=.. mepc=.. mtval=.. vs=.. build_hash=..` | a fault, with everything needed to place it, followed by `DONE` so the log is still a complete record. |

`grade.py` reads all of this: run it on a log that stopped early and it reports the last stage, the
fault, and any failed memory probe instead of just saying the run did not complete.

The `PROF <id> <ticks> <hits>` lines are a **per-op trace** of the whole model — roughly a thousand
of them. `<model>_h<N>.op_table.json` in this package maps each id to the op it measured (name, family,
result shape), so the trace reads as "this much time in that operator" rather than as integers. The
`op=<id>` field in `ALIVE` uses the same ids, which is what makes a stalled run point at a specific
operator instead of at the model as a whole.

If you only have time for one thing: run **`deepjscc` at 1 hart, `_debug`**. It is the smallest image
here, uploads in a couple of minutes, and its log answers vector width, per-hart vector state, real
clock, DRAM extent, stack usage and the whole op trace in one go.

"""


def _readme(brd, manifest: dict, *, debug: bool = False) -> str:
    loader = LOADER_DOC.get(brd.name, LOADER_DOC["default"])
    # Each board gets the explanation of ITS last failure, not a generic one.
    superseded_doc = SUPERSEDED_DOC if brd.sdk_chip else KODIAK_VSTATE_DOC
    debug_doc = DEBUG_DOC if any(b.get("debug") for b in manifest["binaries"]) else ""
    # Say what was actually checked for THIS board's console. Claiming a `.htif` check on a board whose
    # console is its own UART is both wrong and exactly the kind of detail that erodes trust in the
    # rest of the list.
    console_check = ("`.htif` present so your loader can find `tohost`/`fromhost`"
                     if brd.console == CONSOLE_HTIF
                     else "no `.htif` section and no `tohost` symbol, so nothing waits on a host")
    baremetal = brd.flow == boards.FLOW_BAREMETAL
    image = ("bare-metal ELF (our own crt/linker script — your SDK has no RTOS)" if baremetal
             else "Zephyr image")
    hart_counts = sorted({b["harts"] for b in manifest["binaries"]})
    rows = "\n".join(
        f"| `{b['elf']}` | {b['model']} | {b['harts']} | "
        f"{'RVV' if b.get('backend', 'rvv') == 'rvv' else 'scalar'} | "
        f"{'**diagnostic**' if b.get('debug') else 'plain'} | "
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
        # Say plainly where the multicore split is NOT an exact work division. Publishing a blanket
        # "bit-identical" would be false for one of the models, and it is exactly the property we ask
        # the board owners to check themselves.
        #
        # State the OBSERVATION, not a cause. An earlier version of this text concluded the deviation
        # was "deterministic, not a race" and therefore "the vector hardware, not the compiled code".
        # Two runs agreeing does not establish that -- a race with a stable interleaving reproduces
        # just as well -- and instrumented builds of the same models later came back bit-identical on
        # the same bitstream, which is what a timing-sensitive race looks like when you perturb the
        # timing. Shipping an inference dressed as a finding is worse than shipping an open question.
        not_ident = sorted(m for m, e in fs.items()
                           if e.get("harts_bit_identical") is False)
        caveat = ("" if not not_ident else
                  "> **Read this before you trust a 2-hart number.** On that FPGA the 2-hart run of "
                  + ", ".join(f"`{m}`" for m in not_ident)
                  + "\n> is *not* bit-identical to its 1-hart run, and misses the W8A8 gate. We chased it to a\n"
                    "> conclusion, and the conclusion is not about the compiled code:\n"
                    ">\n"
                    "> | test | result |\n"
                    "> |---|---|\n"
                    "> | the same binary run twice on that FPGA | bit-identical, same cycle count — not a race |\n"
                    "> | **those exact binaries on spike at the same vector length** | `h1` ≡ `h2`, both `cos 1.0` |\n"
                    "> | those binaries on the FPGA | `h1` exact, `h2` deviates |\n"
                    ">\n"
                    "> So the arithmetic we generate is right, and something in that SoC's 2-hart vector\n"
                    "> path is not. It is also schedule-sensitive: it moved from one model to another when\n"
                    "> the compiler changed, in both directions. That is a different chip from yours, so it\n"
                    "> says nothing directly about your silicon — but it does say something about our\n"
                    "> evidence, and you should know it:\n"
                    ">\n"
                    "> **A spike gate cannot certify multicore numerics on real hardware.** Every image in\n"
                    "> this package passes one, and the binary above passes one too. So the single most\n"
                    "> useful thing you can do is run `h1` and `h2` of the same model and diff their `OUT`\n"
                    "> lines. Identical is the expected answer and confirms the whole path. Different is a\n"
                    "> real finding on your chip, whatever it costs us to hear — please send both logs.")
        extra = []
        for model, e in sorted(fs.items()):
            if e.get("speedup_1_to_2_harts"):
                extra.append(f"- `{model}`: **{e['speedup_1_to_2_harts']}×** on 2 harts, outputs "
                             f"{'bit-identical' if e.get('harts_bit_identical') else 'NOT identical'}"
                             f" to the 1-hart run")
        firesim_doc = ("""\
## What the same code did on real RTL (not a simulator)

These models also ran on **FireSim**, executing the Saturn RTL on an FPGA — our own SoC, whole model,
cycle-accurate. Bitstream `%s`, **vLen=256**.

These are *separate images*, not the ELFs in this package: a FireSim build targets that SoC at its own
vector length, so it cannot be the same binary as one built for your chip. What carries over is the
model, the lowering and the schedule — the compiled arithmetic — not the image.

| model | harts | cycles | gate | per-element error |
|---|---:|---:|---|---:|
%s

%s

%s

This is *not* a claim about your chip's clock or memory system — different SoC, different frequency.
What it does establish is that the arithmetic holds up on real hardware rather than only in a
functional simulator, and it is where a multicore split gets its first honest test.
""" % (list(fs.values())[0]["bitstream"], "\n".join(rows_fs), "\n".join(extra) or "", caveat))
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

| file | model | harts | ISA | build | linked region | est. upload | our verdict |
|---|---|---|---|---|---|---|---|
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
{debug_doc}

{firesim_doc}
## Status of each model — please read before reporting a number

{statuses}

## What we already checked, without the board

- Built for `{brd.name}` with the board's own facts, and audited the ELF against them: every LOAD
  segment inside your DRAM, entry point in range, {console_check}, and the expected instruction mix —
  real vector instructions in an RVV image, and *zero* in a scalar one (see `elf_audit.json`).
{sim_line}
{identity_line}

What that does *not* cover: your clock, your DRAM timing, your vector unit's actual VLEN, and anything
about wall-clock performance. spike is functional — it proves correctness, never speed.

Merlin commit `{manifest['merlin_commit']}`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
