"""In-sandbox ISA-DEV-TOOLS shim — staged as <ws>/isa_tools.py for the assisted arms (arm-3/arm-4).

Imports NOTHING from merlin (the model is masked in the sandbox). Forwards the request to the driver-side
isa_tools_broker via <ws>/.isa_channel and prints the JSON result. Three subcommands, all oracle-free (no
golden is ever involved — asm encodes the syntax YOU chose; disasm/lint inspect YOUR OWN emitted words):

  # assemble a mnemonic listing -> the correct .word lines for your kernel.S
  python isa_tools.py asm ops.txt
  # disassemble your kernel.S -> what each word actually decodes to (catches invented encodings)
  python isa_tools.py disasm submission/kernel.S
  # static-lint your kernel.S -> illegal opcodes, missing terminator, instruction-class coverage
  python isa_tools.py lint submission/kernel.S --op matmul
  # LITE DEBUGGER (self-hosted-ISA / external_backend target): run your kernel.S on the functional model
  # to instruction N, then dump machine state + named DRAM windows (watch a region fill or NOT as your
  # DMA/compute advances). The OUTPUT region is withheld (that's the answer); debug INPUT/scratch regions.
  python isa_tools.py debug submission/kernel.S --capsule A1_mvin_mvout --run-to 40 \
         --region 0x80001000:256 --region 0x80002000:256
  # LITE DEBUGGER (RoCC / command-buffer target, e.g. gemmini): answer YOUR command_buffer.json on the
  # RTL-derived arc model and see per-op HARDWARE STATE (cycles + scratchpad/accumulator/DRAM-refill
  # counts per command + the RTL fingerprint). The output VALUES and pass/fail verdict are withheld
  # (answer key). This runs your INTENDED computation; use disasm/lint for the emitted .insn ENCODING.
  python isa_tools.py debug selfcheck_out/A6_resident_reuse/command_buffer.json --capsule A6_resident_reuse
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _text_of(arg: str) -> str:
    """`--file` accepts EITHER a path or the ISA text itself.

    `Path(arg).is_file()` raises rather than returning False once the argument exceeds NAME_MAX
    (OSError errno 36), and on an embedded NUL (ValueError) -- so the inline-text branch it guarded
    was unreachable for exactly the inputs it exists to serve. Measured live: an atlas agent passed a
    ~2 KB `.insn`/`.word` listing and the tool died in argument handling, which reads as the agent's
    tool call failing rather than as a harness defect.
    """
    try:
        p = Path(arg)
        if p.is_file():
            return p.read_text()
    except (OSError, ValueError):
        pass
    return arg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derived ISA dev tools (assembler/disassembler/linter/debugger).")
    ap.add_argument("cmd", choices=["asm", "disasm", "lint", "debug"])
    ap.add_argument("file", help="asm: a mnemonic listing; disasm/lint/debug: your kernel.S "
                                 "(RoCC/command-buffer target: your emitted command_buffer.json)")
    ap.add_argument("--op", default="matmul", help="lint coverage: the capsule op (default matmul)")
    ap.add_argument("--output-dtype", default=None)
    ap.add_argument("--movement", action="store_true", help="lint coverage: a data-movement capsule")
    ap.add_argument("--capsule", default=None, help="debug: which public capsule to run your kernel on")
    ap.add_argument("--run-to", type=int, default=None,
                    help="debug: stop after this many instructions (omit = run to halt)")
    ap.add_argument("--region", action="append", default=None, metavar="BASE:NBYTES",
                    help="debug: a DRAM window to dump (repeatable); BASE may be hex (0x..) or decimal")
    ap.add_argument("--state", action="store_true",
                    help="debug: also report a value-free populated-map of on-chip SRAM/registers/"
                         "accumulators (which stage's data landed) — no values, just populated/empty")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args(argv)

    text = _text_of(a.file)
    if a.cmd == "debug":
        regions = []
        for spec in (a.region or []):
            if ":" not in spec:
                print(json.dumps({"error": f"bad --region {spec!r}; use BASE:NBYTES (e.g. 0x80001000:256)"}))
                return 2
            b_s, n_s = spec.rsplit(":", 1)
            try:
                regions.append([int(b_s, 0), int(n_s, 0)])
            except ValueError:
                print(json.dumps({"error": f"bad --region {spec!r}; BASE/NBYTES must be integers"}))
                return 2
        if not a.capsule:
            print(json.dumps({"error": "debug needs --capsule NAME (the capsule to run your kernel on)"}))
            return 2
        # send BOTH artifact keys; the broker picks by endpoint (external_backend -> kernel_s; RoCC/
        # command-buffer -> command_buffer). The shim imports no merlin, so it cannot know which applies.
        req = {"cmd": "debug", "capsule": a.capsule, "kernel_s": text, "command_buffer": text,
               "run_to": a.run_to, "regions": regions, "state_summary": a.state, "timeout": a.timeout}
    else:
        req = {"cmd": a.cmd, "op": a.op, "output_dtype": a.output_dtype, "movement": a.movement}
        req["text" if a.cmd == "asm" else "kernel_s"] = text

    ws = Path(__file__).resolve().parent               # the shim lives at <ws>/isa_tools.py
    ch = ws / ".isa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    rid = f"{os.getpid()}_{int(time.time() * 1000) % 1000000}"
    (ch / f"req_{rid}.json").write_text(json.dumps(req))
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    # debug runs the model inside the broker (up to a.timeout there), so wait past that; asm/disasm/lint
    # are instant, so the margin is harmless.
    deadline = time.time() + a.timeout + (120 if a.cmd == "debug" else 0)
    while time.time() < deadline:
        if done.exists() and resp.exists():
            txt = resp.read_text()
            print(txt)
            try:
                return 1 if json.loads(txt).get("error") else 0
            except Exception:
                return 0
        time.sleep(0.3)
    print(json.dumps({"error": "isa-tools broker did not respond (timeout) — tell the operator"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
