#!/usr/bin/env python
"""Merge the V2 genuine ours-vectorized (polynomial) GELU/sigmoid rows into the canonical
cross_framework_ops_k1.{jsonl,md}, and regenerate the GELU/sigmoid section of the md table.

What it does (measurement-data + doc only):
  * the existing `ours_vectorized` rows in cross_framework_ops_k1.jsonl were lowered with
    vectorize=True but features=[] -> still SCALAR libm (the feature did not exist when built). They
    are RENAMED `ours_vectorize_nofeature` (honest: matmul-vectorize pass, no activation feature).
  * the V2 rows (ours_vectorized_ops_k1.jsonl, source="ours_vectorized", the
    vectorized_transcendental_activation polynomial) are inserted as the new `ours_vectorized`.
  * the GELU/sigmoid head-to-head table block in the .md is regenerated with the new column.

Idempotent: re-running renames only un-renamed rows and replaces (not duplicates) the poly rows.
"""
from __future__ import annotations
import json
from pathlib import Path

CEIL = Path("artifacts/ceiling")
CANON = CEIL / "cross_framework_ops_k1.jsonl"
V2 = CEIL / "ours_vectorized_ops_k1.jsonl"
MD = CEIL / "cross_framework_ops_k1.md"


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    rows = load(CANON)
    v2 = load(V2)
    v2_poly = {(r["op"], r["size_n"]): r for r in v2 if r.get("source") == "ours_vectorized"}

    out = []
    seen_poly = set()
    for r in rows:
        if r.get("op") in ("gelu", "sigmoid") and r.get("source") == "ours_vectorized":
            # the OLD ours_vectorized is actually the no-feature (scalar libm) vectorize pass.
            r = dict(r)
            r["source"] = "ours_vectorize_nofeature"
            r["note"] = ("vectorize=True but NO activation feature -> still scalar libm "
                         "(erff/expf); the genuine vectorized_transcendental_activation polynomial "
                         "is the separate ours_vectorized row")
        out.append(r)
    # drop any pre-existing renamed/poly duplicates then append fresh poly rows after their nofeature
    final = []
    for r in out:
        if r.get("op") in ("gelu", "sigmoid") and r.get("source") == "ours_vectorize_nofeature":
            final.append(r)
            key = (r["op"], r["size_n"])
            if key in v2_poly and key not in seen_poly:
                final.append(v2_poly[key])
                seen_poly.add(key)
        elif r.get("source") == "ours_vectorized" and r.get("op") in ("gelu", "sigmoid"):
            continue  # remove stale poly (re-inserted above)
        else:
            final.append(r)
    # any poly key not yet placed (defensive)
    for key, r in v2_poly.items():
        if key not in seen_poly:
            final.append(r); seen_poly.add(key)

    CANON.write_text("\n".join(json.dumps(r) for r in final) + "\n")
    print(f"jsonl: {len(final)} rows; poly rows merged: {sorted(seen_poly)}")

    # --- regenerate the GELU/sigmoid table block in the md -----------------------------
    def pick(op, n, src):
        for r in final:
            if r.get("op") == op and r.get("size_n") == n and r.get("source") == src:
                return r
        return None

    def fmt(r):
        if not r or r.get("ticks") is None:
            return "—"
        return str(r["ticks"])

    lines = []
    lines.append("## GELU / sigmoid — ours-scalar AND ours-vectorized (polynomial) vs XNNPACK\n")
    lines.append("`ours-vectorized` is the genuine `vectorized_transcendental_activation` feature "
                 "(compiler-emitted minimax polynomial → vectorized SIMD, NOT a libm call). "
                 "`ours-vectorize-nofeature` is the prior column (vectorize pass, NO activation "
                 "feature → still scalar `erff`/`expf` libm). XNNPACK is its hand-written "
                 "rational/exp-poly RVV kernel. K1 `rdtime`, N=3 min; cos/abs-verified.\n")
    lines.append("| op | N | XNNPACK | ours-scalar | ours-vectorize-nofeature | "
                 "ours-vectorized (poly) | poly vs scalar | poly vs XNN |")
    lines.append("|----|---|---------|-------------|--------------------------|"
                 "------------------------|----------------|-------------|")
    for op in ("gelu", "sigmoid"):
        for n in (1024, 16384, 262144):
            xn = pick(op, n, "xnnpack")
            sc = pick(op, n, "ours_scalar")
            nf = pick(op, n, "ours_vectorize_nofeature")
            po = pick(op, n, "ours_vectorized")
            sp_scalar = sp_xn = "—"
            if po and po.get("ticks") and sc and sc.get("ticks"):
                sp_scalar = f"{sc['ticks']/po['ticks']:.2f}× faster"
            if po and po.get("ticks") and xn and xn.get("ticks"):
                ratio = po["ticks"] / xn["ticks"]
                sp_xn = f"{ratio:.2f}× slower" if ratio >= 1 else f"{1/ratio:.2f}× faster"
            label = "**" + op.upper() + "**" if n == 1024 else op
            lines.append(f"| {label} | {n//1024}K | {fmt(xn)} | {fmt(sc)} | {fmt(nf)} | "
                         f"**{fmt(po)}** | {sp_scalar} | {sp_xn} |")
    block = "\n".join(lines) + "\n"

    md = MD.read_text()
    marker = "## GELU / sigmoid — ours-scalar AND ours-vectorized"
    insert_after = "## Per-op results (K1 rdtime ticks, lower = faster)"
    if marker in md:
        # replace existing block (up to the next "## ")
        head, _, rest = md.partition(marker)
        after = rest.split("\n## ", 1)
        tail = ("\n## " + after[1]) if len(after) > 1 else ""
        md = head + block + tail
    else:
        idx = md.index(insert_after)
        # insert the new block right before the existing per-op results header
        md = md[:idx] + block + "\n" + md[idx:]
    MD.write_text(md)
    print(f"md updated: {MD}")


if __name__ == "__main__":
    main()
