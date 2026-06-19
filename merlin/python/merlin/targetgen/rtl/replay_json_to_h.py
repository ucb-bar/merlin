"""Convert a RoCC replay spec JSON (gen_rocc_replay) into a C header the arc replay harness #includes."""
import json, sys
s = json.load(open(sys.argv[1]))
L = ["// generated — do not edit", "#pragma once", "#include <stdint.h>",
     f"#define N_PLACE {len(s['placements'])}"]
for i, p in enumerate(s["placements"]):
    b = bytes.fromhex(p["bytes_hex"])
    L.append(f"static const uint8_t PLACE{i}_DATA[{len(b)}]={{{','.join(map(str,b))}}};")
L.append("static const struct{uint64_t addr;int len;const uint8_t*data;}PLACE[]={")
for i, p in enumerate(s["placements"]):
    L.append(f"  {{{p['addr']}ULL,{len(bytes.fromhex(p['bytes_hex']))},PLACE{i}_DATA}},")
L.append("};")
o = s["outputs"][0]
L += [f"#define OUT_ADDR {o['addr']}ULL", f"#define OUT_N {len(o['golden_flat'])}",
      f"#define OUT_ROWS {o.get('rows',0)}", f"#define OUT_COLS {o.get('cols',0)}",
      f"#define OUT_STRIDE {o.get('stride_bytes',0)}", f"#define OUT_ELEM {o.get('elem_bytes',4)}",
      f"static const int64_t OUT_GOLDEN[{len(o['golden_flat'])}]={{{','.join(map(str,o['golden_flat']))}}};"]
aa = s["arg_addr"]
def enc(r):
    if not r or r.get("kind") == "unknown": return 0, 1   # unknown -> flag
    if r.get("kind") == "const": return (r.get("raw") or 0), 0
    if r.get("kind") == "argbase": return int(aa[str(r["arg_index"])]) + (r.get("offset") or 0), 0
    return 0, 1
n_unknown = 0
L.append("static const struct{int is_fence;int funct;uint64_t rs1;uint64_t rs2;}INSN[]={")
for i in s["insns"]:
    if i.get("funct") is None: L.append("  {1,0,0,0},"); continue
    v1, u1 = enc(i.get("rs1")); v2, u2 = enc(i.get("rs2")); n_unknown += u1 + u2
    L.append(f"  {{0,{i['funct']},{v1}ULL,{v2}ULL}},")
L += ["};", f"#define N_INSN {len(s['insns'])}", f"#define N_UNKNOWN_OPERANDS {n_unknown}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print(f"{sys.argv[2]}: insns={len(s['insns'])} place={len(s['placements'])} "
      f"out={o.get('rows')}x{o.get('cols')} elem={o.get('elem_bytes')} stride={o.get('stride_bytes')} "
      f"unknown_operands={n_unknown}")
