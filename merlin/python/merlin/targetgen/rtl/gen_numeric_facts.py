"""CIRCT facts -> a numeric-SHAPE sanity checker (shrinks CIRCT's numeric blind spot).

CIRCT's structural screen catches illegal/mis-ordered instructions but NOT "structurally-legal yet
numerically-wrong" code — that residual is why a sim is still needed to certify numbers. This narrows the
residual: from the RTL datapath facts (input dtype, accumulator width, scale/activation semantics) it
emits a checker that flags numeric-SHAPE mistakes (wrong accumulation width, mismatched output dtype,
missing scale on a scaled op) WITHOUT computing or comparing any golden value. It does NOT certify
numerics — it catches a class of shape/width bugs cheaply so fewer sim runs are strictly required.

ANTI-CHEAT: reads only RTL-derived structural facts (widths/dtypes) — no golden values, no per-capsule
outputs. Same line as facts.json / the encoder. RTL-derived ⇒ CIRCT-arm-only.

Usage: python -m merlin.targetgen.rtl.gen_numeric_facts [--facts <facts.json>] [--out numeric_facts.py]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from .facts import rtl_facts_path
_DEFAULT_FACTS = rtl_facts_path("gemmini")

_TMPL = '''"""GENERATED from RTL facts by gen_numeric_facts — numeric-SHAPE sanity (NOT a numeric oracle)."""
from __future__ import annotations

INPUT_DTYPE = {input_dtype!r}          # scratchpad element dtype (RTL)
ACC_DTYPE = {acc_dtype!r}              # accumulator element dtype (RTL)
ACC_WIDTH_BITS = {acc_bits}            # accumulation must happen at this width to avoid overflow

def check_numeric_shapes(cb: dict) -> list[str]:
    """Flag numeric-SHAPE mistakes against the RTL datapath. cb = a command_buffer dict.
    Returns a list of findings (empty = no shape issue found). Does NOT check VALUES — a sim still
    certifies numerics. Catches: accumulation declared below ACC_WIDTH_BITS, scaled/activated output whose
    declared dtype ignores the accumulator width, matmul whose output dtype is narrower than the accumulator
    without an explicit scale/cast (silent truncation)."""
    findings = []
    tensors = cb.get("tensors", {{}})
    for i, c in enumerate(cb.get("commands", [])):
        op = c.get("opcode", "")
        attrs = c.get("attributes", {{}}) or {{}}
        # accumulator-producing ops should accumulate at ACC_DTYPE width
        if "MATMUL" in op or "COMPUTE" in op:
            dst = (c.get("operands", {{}}) or {{}}).get("dst")
            dt = (tensors.get(dst, {{}}) or {{}}).get("dtype")
            if dt and _bits(dt) and _bits(dt) < ACC_WIDTH_BITS:
                findings.append(f"commands[{{i}}] {{op}}: accumulator '{{dst}}' dtype {{dt}} is narrower than "
                                f"the RTL accumulator ({{ACC_DTYPE}}/{{ACC_WIDTH_BITS}}b) -> overflow/truncation risk")
        # an op declaring an epilogue scale/activation must produce a typed output
        if op in ("COMMIT", "STORE", "MVOUT") or "epilogue" in attrs:
            dst = (c.get("operands", {{}}) or {{}}).get("dst")
            if dst and "dtype" not in (tensors.get(dst, {{}}) or {{}}) and "output_dtype" not in attrs:
                findings.append(f"commands[{{i}}] {{op}}: scaled/stored output '{{dst}}' has no declared "
                                f"dtype/output_dtype -> ambiguous requantization")
    return findings

def _bits(dt: str):
    import re
    m = re.search(r"(\\d+)", dt or "")
    return int(m.group(1)) if m else None
'''


def generate(facts: dict) -> str:
    f = facts["facts"]
    dps = {d["name"]: d for d in f.get("datapaths", [])}
    acc = next((m for m in f.get("memories", []) if m.get("name") == "accumulator"), {})
    acc_bits = acc.get("lane_bits") or 32
    return _TMPL.format(input_dtype=dps.get("input", {}).get("dtype", "i8"),
                        acc_dtype=dps.get("accumulator", {}).get("dtype", "i32"),
                        acc_bits=acc_bits)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facts", default=str(_DEFAULT_FACTS))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    code = generate(json.loads(Path(a.facts).read_text()))
    if a.out:
        Path(a.out).write_text(code); print(f"wrote {a.out} ({len(code.splitlines())} lines)")
    else:
        print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
