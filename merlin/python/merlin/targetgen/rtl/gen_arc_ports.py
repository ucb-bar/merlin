"""Generate a C header of port offsets/widths from an arcilator --state-file JSON.

The arc model state is a flat byte buffer; each named port/state lives at a byte offset (byte-aligned,
little-endian, ceil(numBits/8) bytes wide). This emits `#define`s + tiny get/set helpers so the C drive
harness can poke RoCC/TileLink ports by name. Part of the arcilator middle-tier (#143-a).

CLI: python -m merlin.targetgen.rtl.gen_arc_ports gemmini.state.json --model Gemmini --out gemmini_arc_ports.h
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("state_json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    models = json.loads(Path(a.state_json).read_text())
    m = next(x for x in models if x["name"] == a.model)
    L = [f"// generated from {Path(a.state_json).name} model @{a.model} — do not edit",
         "#pragma once", "#include <stdint.h>", "#include <string.h>",
         f"#define ARC_STATE_BYTES {m['numStateBytes']}",
         f'extern void {a.model}_eval(void*);',
         f'extern void {a.model}_clock(void*);',
         f'extern void {a.model}_passthrough(void*);', ""]
    def cname(n):
        return "P_" + re.sub(r"[^A-Za-z0-9]", "_", n)
    for s in m["states"]:
        if s["type"] in ("input", "output"):
            L.append(f"#define {cname(s['name'])}_OFF {s['offset']}  // {s['numBits']}b {s['type']}")
            L.append(f"#define {cname(s['name'])}_BITS {s['numBits']}")
    # generic ≤64-bit get/set (little-endian byte field) + a 128-bit data accessor
    L += ["",
          "static inline uint64_t arc_get(const void* st, int off, int bits){",
          "  uint64_t v=0; int nb=(bits+7)/8; memcpy(&v, (const uint8_t*)st+off, nb>8?8:nb);",
          "  if(bits<64) v &= ((uint64_t)1<<bits)-1; return v; }",
          "static inline void arc_set(void* st, int off, int bits, uint64_t val){",
          "  int nb=(bits+7)/8; if(bits<64) val &= ((uint64_t)1<<bits)-1;",
          "  memcpy((uint8_t*)st+off, &val, nb>8?8:nb); }",
          "static inline void arc_get128(const void* st,int off,uint64_t out[2]){"
          "memcpy(out,(const uint8_t*)st+off,16);}",
          "static inline void arc_set128(void* st,int off,const uint64_t in[2]){"
          "memcpy((uint8_t*)st+off,in,16);}"]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}: {sum(1 for s in m['states'] if s['type'] in ('input','output'))} ports, "
          f"state={m['numStateBytes']}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
