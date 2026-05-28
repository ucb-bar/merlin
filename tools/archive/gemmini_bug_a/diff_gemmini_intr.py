#!/usr/bin/env python3
"""Diff gemmini llvm.riscv.* intrinsic operand streams between two .codegen.ll files.

Extracts the constant-folded sequence of (mnemonic, rs1, rs2) calls from a named
function in each file and reports the first divergent operands at matched
positions. Used to localise where a buggy dispatch's RoCC stream diverges from a
known-good fixture.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CALL_RE = re.compile(
    r"call\s+\S+\s+@llvm\.riscv\.([a-zA-Z0-9_.]+)\s*\(\s*" r"i\d+\s+([^,)]+?)\s*(?:,\s*i\d+\s+([^,)]+?)\s*)?\)"
)


def extract_function_body(text: str, fname: str) -> str:
    pat = re.compile(
        r'^define[^\n]*@(?:"' + re.escape(fname) + r'"|' + re.escape(fname) + r")[^\n]*\{",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        raise SystemExit(f"function not found: {fname}")
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def parse_value(s: str) -> str:
    s = s.strip()
    if re.fullmatch(r"-?\d+", s):
        n = int(s)
        unsigned = n & ((1 << 64) - 1)
        return f"{n} (0x{unsigned:016x})"
    return s


def extract_intrinsic_stream(body: str) -> list[tuple[str, str, str | None, int]]:
    out: list[tuple[str, str, str | None, int]] = []
    for m in CALL_RE.finditer(body):
        mnem = m.group(1)
        rs1 = parse_value(m.group(2))
        rs2 = parse_value(m.group(3)) if m.group(3) else None
        line = body.count("\n", 0, m.start()) + 1
        out.append((mnem, rs1, rs2, line))
    return out


def decode_config(v: int) -> str:
    # bits[1:0] func type, bits[7:2] scale exp / shift / etc.
    # For config: bits[63:32] = scale_f32 bits, bits[31:0] = ctrl.
    high = (v >> 32) & 0xFFFFFFFF
    low = v & 0xFFFFFFFF
    return f"hi=0x{high:08x} lo=0x{low:08x}"


def fmt(v: str) -> str:
    return v


def equiv(a: str | None, b: str | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    # Treat any two SSA value refs as equivalent (they are runtime ptrs).
    if a.startswith("%") and b.startswith("%"):
        return True
    return False


def diff_streams(
    a: list[tuple[str, str, str | None, int]],
    b: list[tuple[str, str, str | None, int]],
    label_a: str,
    label_b: str,
    max_print: int = 20,
) -> None:
    print(f"{label_a}: {len(a)} intrinsics")
    print(f"{label_b}: {len(b)} intrinsics")
    print()
    n = min(len(a), len(b))
    diffs = 0
    for i in range(n):
        ma, r1a, r2a, la = a[i]
        mb, r1b, r2b, lb = b[i]
        if ma != mb or not equiv(r1a, r1b) or not equiv(r2a, r2b):
            if diffs == 0:
                print("FIRST DIVERGENCES:")
                print("=" * 80)
            diffs += 1
            print(f"\n[{i}] DIVERGE")
            print(f"  {label_a:30s} line {la}: {ma:24s} rs1={r1a} rs2={r2a}")
            print(f"  {label_b:30s} line {lb}: {mb:24s} rs1={r1b} rs2={r2b}")
            if diffs >= max_print:
                print(f"\n... stopping after {max_print} divergences")
                break
    if diffs == 0:
        print(f"NO DIVERGENCES in first {n} intrinsics — streams MATCH up to that point.")
        if len(a) != len(b):
            print(f"  but lengths differ: {label_a}={len(a)} vs {label_b}={len(b)}")
    else:
        print(f"\nTotal divergences shown: {diffs}")
    return diffs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first .codegen.ll file (fixture)")
    p.add_argument("--b", required=True, help="second .codegen.ll file (suspect)")
    p.add_argument("--fn-a", required=True, help="function name in A")
    p.add_argument("--fn-b", required=True, help="function name in B")
    p.add_argument("--max", type=int, default=20, help="max divergences to print")
    args = p.parse_args()

    ta = Path(args.a).read_text()
    tb = Path(args.b).read_text()
    ba = extract_function_body(ta, args.fn_a)
    bb = extract_function_body(tb, args.fn_b)
    sa = extract_intrinsic_stream(ba)
    sb = extract_intrinsic_stream(bb)
    diff_streams(sa, sb, args.fn_a, args.fn_b, args.max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
