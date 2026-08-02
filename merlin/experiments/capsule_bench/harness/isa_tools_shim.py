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
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derived ISA dev tools (assembler/disassembler/linter).")
    ap.add_argument("cmd", choices=["asm", "disasm", "lint"])
    ap.add_argument("file", help="asm: a mnemonic listing; disasm/lint: your kernel.S")
    ap.add_argument("--op", default="matmul", help="lint coverage: the capsule op (default matmul)")
    ap.add_argument("--output-dtype", default=None)
    ap.add_argument("--movement", action="store_true", help="lint coverage: a data-movement capsule")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args(argv)

    text = Path(a.file).read_text() if Path(a.file).is_file() else a.file
    req = {"cmd": a.cmd, "op": a.op, "output_dtype": a.output_dtype, "movement": a.movement}
    req["text" if a.cmd == "asm" else "kernel_s"] = text

    ws = Path(__file__).resolve().parent               # the shim lives at <ws>/isa_tools.py
    ch = ws / ".isa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    rid = f"{os.getpid()}_{int(time.time() * 1000) % 1000000}"
    (ch / f"req_{rid}.json").write_text(json.dumps(req))
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    deadline = time.time() + a.timeout
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
