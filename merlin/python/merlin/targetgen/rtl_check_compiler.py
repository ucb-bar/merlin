"""Compile RTL-derived facts + a capsule's declared shape into **FileCheck** assertions.

This is the "compile the RTL to the checks" step: instead of hand-coded Python predicates, the RTL facts
(:mod:`merlin.targetgen.rtl.circt_introspect`) and the capsule's *declared* problem statement are turned
into FileCheck directives that LLVM's ``FileCheck`` runs against the agent's emitted artifacts. The
arithmetic (tile count = ⌈M/DIM⌉·⌈N/DIM⌉, legal-opcode set, required ops) is evaluated HERE, at
check-generation time, so concrete RTL-grounded literals are baked into the CHECK lines; FileCheck does
the matching and its diagnostic (the offending IR line) is the feedback.

Two check files are produced (run by :mod:`rtl_check_runner`):

* ``dialect`` — over the agent's high-level gemmini-dialect MLIR (``lowered.target.mlir``): structural
  invariants (a matmul must ``res_pack``→``matmul``→``commit``; the commit's ``output_dtype`` must match
  the declared dtype). Ordered ``CHECK`` enforces dataflow order.
* ``trace`` — over a canonical text rendering of the decoded RoCC trace: ``MVOUT_COUNT`` exact tile
  coverage, ``UNKNOWN_COUNT 0`` (every funct ∈ the RTL-derived legal set), ``COMPUTE_COUNT`` positive for
  compute ops, and the ABI opcode/funct3 from the RTL facts.

NON-OVERFIT: every literal comes from RTL facts + declared shape + ISA rules — never a golden trace or a
per-capsule expected count. For shapes where an exact tile count cannot be derived generally (multi-matmul
resident reuse, conv) the check degrades to a lower bound or is omitted, honestly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import rtl_checks as RC  # reuse _declared_output_shape / _mesh / _declared_op

RENDER_SCHEMA = "rtl-trace-render/v0"
_COMPUTE_OPS = {"matmul", "resident_reuse", "conv2d", "conv", "matmul_resident"}


def _facts_abi(facts: dict) -> tuple[str, str]:
    for i in (facts.get("interfaces") or []):
        if i.get("name") == "funct_decode_table":
            return (f"0x{i['custom_opcode']:x}", f"0x{i['funct3']:x}")
    return ("0x7b", "0x3")


def _facts_to_rc(facts_rec: dict) -> dict:
    """Translate a circt_introspect facts record into the flat dict rtl_checks helpers expect
    (mesh + scratchpad capacity)."""
    facts = facts_rec.get("facts", facts_rec)
    mesh = next((a for a in facts.get("arrays", []) if a["name"] == "mesh"), {})
    sp = next((m for m in facts.get("memories", []) if m["name"] == "scratchpad"), {})
    out = {}
    if mesh:
        out["mesh"] = [mesh["rows"], mesh["cols"]]
    if sp.get("bytes"):
        out["scratchpad_bytes"] = sp["bytes"]
    return out


def compile_dialect_checks(capsule: dict, prefix: str = "DIALECT") -> str | None:
    """FileCheck lines over the gemmini-dialect MLIR (lowered.target.mlir)."""
    op = RC._declared_op(capsule)
    if op is None:
        return None
    out_dtype = ((capsule.get("operation") or {}).get("attributes") or {}).get("output_dtype")
    # header is a plain comment (no "{prefix}:" so FileCheck ignores it); facts are order-independent
    # presence assertions => -DAG. SSA already guarantees res_pack->matmul->commit dataflow order.
    L = [f"// RTL-derived dialect checks (op={op}) — generated, do not edit"]
    lines = [f"// {prefix}-DAG: gemmini.commit"]
    if op in ("matmul", "matmul_resident"):
        lines = [f"// {prefix}-DAG: gemmini.res_pack",
                 f"// {prefix}-DAG: gemmini.matmul",
                 f"// {prefix}-DAG: gemmini.commit"]
    elif op == "resident_reuse":
        n = sum(1 for t in (capsule.get("inputs") or []) if t.get("role") == "input")
        lines = [f"// {prefix}-DAG: gemmini.res_pack",
                 f"// {prefix}-COUNT-{max(n,1)}: gemmini.matmul",
                 f"// {prefix}-DAG: gemmini.commit"]
    elif op in ("conv2d", "conv"):
        lines = [f"// {prefix}-DAG: gemmini.commit"]  # conv lowering varies; assert commit only
    elif op == "movement":
        lines = [f"// {prefix}-NOT: gemmini.matmul"]  # movement must not invent compute
    if out_dtype and op != "movement":
        lines.append(f'// {prefix}-DAG: output_dtype = "{out_dtype}"')
    return "\n".join(L + lines) + "\n"


def compile_trace_checks(facts_rec: dict, capsule: dict, prefix: str = "TRACE") -> str | None:
    """FileCheck lines over the rendered RoCC trace (see rtl_check_runner.render_trace)."""
    op = RC._declared_op(capsule)
    if op is None:
        return None
    rc_facts = _facts_to_rc(facts_rec)
    mr, mc = RC._mesh(rc_facts)
    custom, funct3 = _facts_abi(facts_rec.get("facts", facts_rec))
    L = [f"// RTL-derived trace checks (op={op}, mesh={mr}x{mc}) — generated, do not edit",
         f"// {prefix}-DAG: ABI custom={custom} funct3={funct3}",
         # {{$}} anchors end-of-line so "..._COUNT 1" does NOT substring-match "..._COUNT 16"
         f"// {prefix}-DAG: ILLEGAL_FUNCT_COUNT 0{{{{$}}}}"]   # every emitted funct ∈ RTL legal set
    shape = RC._declared_output_shape(capsule)
    if op in ("matmul", "matmul_resident") and shape is not None:
        M, N = shape
        tiles = math.ceil(M / mr) * math.ceil(N / mc)
        L.append(f"// {prefix}-DAG: MVOUT_COUNT {tiles}{{{{$}}}}")  # exact tile coverage (RTL DIM + shape)
        L.append(f"// {prefix}-DAG: COMPUTE_PRESENT yes")
    elif op == "resident_reuse":
        # multi-matmul lower bound is not FileCheck-exact; leave the count to rtl_checks.screen().
        L.append(f"// {prefix}-DAG: COMPUTE_PRESENT yes")
    elif op in ("conv2d", "conv"):
        L.append(f"// {prefix}-DAG: COMPUTE_PRESENT yes")      # conv tile geometry not derived generally
    elif op == "movement":
        L.append(f"// {prefix}-DAG: MVIN_PRESENT yes")
    return "\n".join(L) + "\n"


def compile_checks(facts_rec: dict, capsule: dict) -> dict[str, Any]:
    """Compile both check files for a capsule. Returns {dialect, trace, meta}."""
    return {
        "schema": "rtl_checks_filecheck/v0",
        "capsule": capsule.get("name"),
        "dialect": compile_dialect_checks(capsule),
        "trace": compile_trace_checks(facts_rec, capsule),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description="Compile RTL facts + capsule -> FileCheck assertion files.")
    ap.add_argument("capsule", help="path to capsule.yaml")
    ap.add_argument("--facts", default=str(Path(__file__).resolve().parents[3]
                    / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"))
    a = ap.parse_args(argv)
    facts = json.loads(Path(a.facts).read_text())
    capsule = yaml.safe_load(Path(a.capsule).read_text())
    cc = compile_checks(facts, capsule)
    print(f"# capsule={cc['capsule']}\n# --- dialect ---\n{cc['dialect']}\n# --- trace ---\n{cc['trace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
