"""The agent-facing VERIFICATION SPEC — the answer-free acceptance contract, as a QA/verification team
would hand a compiler engineer bringing up brand-new hardware.

The experiment simulates early SW-stack bring-up: the agent's world is the RTL + a few helpers + maybe an
example kernel + docs, with NO pre-existing SW stack and NO precomputed answer key (the golden "would not
technically exist"). What the agent legitimately GETS is a spec of *what must hold to pass* — the target
operations, the datatypes/formats, the numeric acceptance policy, and the datapath-coverage requirement —
without any golden value or any detail of how the oracle computes the answer.

This module DERIVES that spec by aggregating the suite's per-capsule declarations (each ``capsule.yaml``'s
``operation`` / input+output dtypes / ``numeric_policy`` / ``expected.instruction_classes``) — the same
answer-free contract fields the agent already sees per capsule — into one coherent acceptance document. It
reads ONLY ``capsule.yaml`` (never ``golden.yaml`` / any answer surface), so the rendered spec cannot carry
an expected output. It is target-agnostic: everything comes from the ``TargetExperiment`` and its corpus —
no target-name literal, no regex.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import repo_root

# The per-capsule contract fields we aggregate — all answer-free (they declare WHAT the op is + how it is
# accepted, never the result). Deliberately excludes ``golden.yaml`` and any ``expected_command_buffer*``.
_ANSWER_FREE_CAPSULE = "capsule.yaml"


def _suite_roots(te: Any) -> list[Path]:
    """The graded (non-hidden) capsule roots for the target: the primary corpus + its declared siblings.
    Hidden capsules are the post-freeze holdout and are NEVER included."""
    roots: list[Path] = []
    primary = getattr(te, "capsule_corpus", None)
    if primary:
        roots.append(Path(primary))
    for rel in (getattr(te, "corpus_siblings", lambda: [])() or []):
        p = repo_root() / rel
        if p.is_dir():
            roots.append(p)
    # de-dup while preserving order; drop any hidden root defensively
    seen, out = set(), []
    for r in roots:
        rr = r.resolve()
        if rr in seen or r.name == "hidden":
            continue
        seen.add(rr)
        out.append(r)
    return out


def _capsules(te: Any) -> list[dict]:
    """Every graded capsule's DECLARED spec (parsed ``capsule.yaml``), across the suite roots. Reads only
    the answer-free contract file; a capsule dir's ``golden.yaml`` is never opened."""
    out: list[dict] = []
    for root in _suite_roots(te):
        for cy in sorted(Path(root).rglob(_ANSWER_FREE_CAPSULE)):
            try:
                doc = yaml.safe_load(cy.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a malformed declaration is skipped, never guessed
                continue
            if isinstance(doc, dict) and (doc.get("label") != "hidden"):
                out.append(doc)
    return out


def _io_dtypes(cap: dict) -> tuple[str, str]:
    """(operand dtypes summary, output dtype) declared by a capsule — from its inputs + operation.
    ``operand`` is the set of input/weight element dtypes; ``output`` is the declared output dtype."""
    ins = cap.get("inputs") or []
    operand = sorted({str(t.get("dtype")) for t in ins if t.get("dtype")})
    op = cap.get("operation") or {}
    out_dt = str((op.get("attributes") or {}).get("output_dtype")
                 or (cap.get("numeric_policy") or {}).get("dtype") or "?")
    return ("+".join(operand) if operand else "?", out_dt)


def build_spec(te: Any) -> dict[str, Any]:
    """The verification spec as structured data, DERIVED from the suite's declared capsules. Shape:
    ``{target, n_capsules, ops: {op: {dtypes: [...], accept: [...], coverage: [...]}}, acceptance, notes}``.
    Answer-free by construction (only ``capsule.yaml`` is read)."""
    caps = _capsules(te)
    ops: dict[str, dict[str, set]] = {}
    for cap in caps:
        op = str((cap.get("operation") or {}).get("op") or "unknown")
        epi = tuple((cap.get("operation") or {}).get("attributes", {}).get("epilogue", []) or [])
        operand, out_dt = _io_dtypes(cap)
        pol = cap.get("numeric_policy") or {}
        accept = pol.get("compare", "?")
        # tolerance detail, when the policy declares one (float targets); acc_scale when present
        tol = {k: pol[k] for k in ("atol", "rtol", "acc_scale") if k in pol}
        classes = tuple((cap.get("expected") or {}).get("instruction_classes") or [])
        slot = ops.setdefault(op, {"dtypes": set(), "accept": set(), "coverage": set(), "epilogues": set()})
        slot["dtypes"].add(f"{operand} -> {out_dt}")
        slot["accept"].add(accept + (f" ({tol})" if tol else ""))
        slot["coverage"].update(classes)
        if epi:
            slot["epilogues"].add("+".join(epi))
    ops_out = {op: {"dtypes": sorted(s["dtypes"]), "accept": sorted(s["accept"]),
                    "coverage": sorted(s["coverage"]), "epilogues": sorted(s["epilogues"])}
               for op, s in sorted(ops.items())}
    return {
        "target": getattr(te, "target", "?"),
        "n_capsules": len(caps),
        "ops": ops_out,
        "isa_docs": list(getattr(te, "isa_headers", []) or []),
    }


def render_markdown(te: Any) -> str:
    """The verification spec as a QA-team acceptance document (Markdown) the agent reads as its contract.
    States WHAT must hold to pass (ops, dtypes, acceptance policy, datapath coverage) and HOW to validate
    it as a bring-up engineer would — never a golden value, never how the oracle computes the answer."""
    spec = build_spec(te)
    L: list[str] = []
    L.append(f"# Verification spec — acceptance contract for `{spec['target']}`")
    L.append("")
    L.append("_You are bringing up the software stack for brand-new hardware. Your world is the RTL, the "
             "shipped ISA/ABI docs, any example kernel, and this spec — there is **no pre-existing SW "
             "stack and no answer key**. This is the contract the verification team gives you: it says "
             "WHAT we test for and the pass criteria, not the expected outputs. Validate your work the way "
             "an engineer does — compute the operation's expected result yourself from the declared inputs, "
             "run your emitted artifact on the RTL, and debug divergences with the disassembler / trace / "
             "hardware-state tools._")
    L.append("")
    L.append(f"**Scope:** {spec['n_capsules']} graded capsules across the operations below (the hidden "
             "holdout is not shown). Each capsule's `capsule.yaml` is the itemized test: its declared "
             "operation, input/output dtypes, acceptance policy, and required datapath coverage.")
    L.append("")
    L.append("## Target operations, datatypes, and acceptance")
    for op, d in spec["ops"].items():
        L.append(f"### `{op}`")
        L.append(f"- **datatypes (operands -> output):** {', '.join(d['dtypes']) or '?'}")
        if d["epilogues"]:
            L.append(f"- **epilogues:** {', '.join(d['epilogues'])}")
        L.append(f"- **acceptance policy:** {', '.join(d['accept']) or '?'}  "
                 "(exact_int = bit-exact integer match; tolerance_float = within the stated atol/rtol)")
        if d["coverage"]:
            L.append(f"- **datapath coverage (must actually exercise, not fake):** "
                     f"{', '.join(d['coverage'])}")
        L.append("")
    L.append("## What is tested (engineer terms)")
    L.append("- **Functional correctness:** your emitted artifact, run on the RTL (the oracle), must "
             "compute the declared operation within the acceptance policy above. There is no stored "
             "golden you can read — the reference is the operation's own mathematical definition, which "
             "you can reproduce from the declared inputs.")
    L.append("- **Datapath coverage:** the emitted stream must exercise the real hardware datapath (the "
             "required instruction classes), not shortcut the result.")
    L.append("- **Legality:** every emitted instruction must be one the target's decoder accepts (ISA "
             "legality), and the program must terminate.")
    if spec["isa_docs"]:
        L.append("")
        L.append("## ISA / ABI references")
        for h in spec["isa_docs"]:
            L.append(f"- `{h}`")
    L.append("")
    return "\n".join(L) + "\n"


def write_spec(te: Any, dest_dir: str | Path, *, name: str = "verification_spec.md") -> Path:
    """Render + write the verification spec into ``dest_dir`` (e.g. the agent workspace root). Returns the
    written path. Regenerable at any time from the (answer-free) capsule declarations."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    out.write_text(render_markdown(te), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.common.paths import merlin_dir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=None, help="write the spec here (default: print to stdout)")
    a = ap.parse_args(argv)

    p = merlin_dir() / "experiments" / "capsule_bench" / "targets" / a.target / "target_experiment.yaml"
    te = load_target_experiment(p)
    md = render_markdown(te)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        print(f"[verification_spec] wrote {a.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
