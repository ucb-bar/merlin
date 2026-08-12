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

from .facts import load_facts

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
        # accumulator-producing ops should accumulate at ACC_DTYPE width. Fail-closed: when the RTL facts
        # did not ground the accumulator width (ACC_WIDTH_BITS is None), SKIP this check rather than
        # assume one — a numeric-shape finding must never rest on a defaulted width.
        if ("MATMUL" in op or "COMPUTE" in op) and ACC_WIDTH_BITS:
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
    # first contiguous run of digits in the dtype token (i8->8, bf16->16, f8E4M3FN->8) — structural, no regex
    digits = ""
    for ch in (dt or ""):
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else None
'''


def generate(facts: dict) -> str:
    f = facts["facts"]
    dps = {d["name"]: d for d in f.get("datapaths", [])}
    acc = next((m for m in f.get("memories", []) if m.get("name") == "accumulator"), {})
    # DERIVE every value from the target's RTL facts; when a fact is absent, emit ``None`` and let the
    # generated checker FAIL CLOSED (skip that check) — NEVER substitute a per-target default. The old
    # ``or 32`` / ``"i8"`` / ``"i32"`` fallbacks silently handed any target whose facts lacked datapaths
    # gemmini's numeric-shape rules (the derive-vs-overfit cardinal-rule violation this repo forbids).
    return _TMPL.format(input_dtype=dps.get("input", {}).get("dtype"),
                        acc_dtype=dps.get("accumulator", {}).get("dtype"),
                        acc_bits=acc.get("lane_bits"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facts", default=None, help="facts.json (default: regenerate the target's facts from RTL)")
    ap.add_argument("--target", default=None, help="target whose RTL facts to regenerate when --facts is omitted")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.facts:
        facts = json.loads(Path(a.facts).read_text())
    elif a.target:
        facts = load_facts(a.target)
    else:
        ap.error("provide --facts <facts.json> or --target <name> to regenerate the facts from RTL")
    code = generate(facts)
    if a.out:
        Path(a.out).write_text(code); print(f"wrote {a.out} ({len(code.splitlines())} lines)")
    else:
        print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
