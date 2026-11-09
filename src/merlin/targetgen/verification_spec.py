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

import json
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
    for rel in getattr(te, "corpus_siblings", lambda: [])() or []:
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
    out_dt = str(
        (op.get("attributes") or {}).get("output_dtype") or (cap.get("numeric_policy") or {}).get("dtype") or "?"
    )
    return ("+".join(operand) if operand else "?", out_dt)


def _whole_op_mnemonics() -> set[str]:
    """The WHOLE-OP mnemonics the frozen interface grammar defines, from the reference parser's own
    table -- the ones that stand 1:1 with a capsule's ``operation.op``, so "no graded capsule declares
    this op" is a true statement about them.

    Deliberately NOT every defined mnemonic. ``resident_pack`` / ``matmul`` / ``commit`` / ``evict`` are
    the residency decomposition every contraction capsule emits, so reporting them as never exercised
    would be false. The parser is core code beside this module, so an import failure is a defect and
    raises rather than quietly dropping the entry."""
    from merlin.targetgen.contract import interface_emit

    return set(interface_emit._NAMED_OP_OPERAND_KEYS)


def build_spec(te: Any) -> dict[str, Any]:
    """The verification spec as structured data, DERIVED from the suite's declared capsules. Shape:
    ``{target, n_capsules, ops: {op: {dtypes, accept, coverage, epilogues, checked_by}}, not_checked,
    isa_docs}``. Answer-free by construction (only ``capsule.yaml`` is read).

    ``checked_by`` and ``not_checked`` exist because this spec is derived from the GRADED CORPUS: what
    the corpus does not demand, the spec does not require, and the agent could not tell the difference
    between "this is not required" and "this is required and nothing looks". A spec whose value is
    telling you what will be CHECKED has to admit what will not.
    """
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
        slot = ops.setdefault(
            op,
            {
                "dtypes": set(),
                "accept": set(),
                "coverage": set(),
                "epilogues": set(),
                "tiers": set(),
                "accelerate": set(),
            },
        )
        slot["dtypes"].add(f"{operand} -> {out_dt}")
        slot["accept"].add(accept + (f" ({tol})" if tol else ""))
        slot["coverage"].update(classes)
        # WHAT ACTUALLY LOOKS at a submission for this op, as the capsules declare it.
        slot["tiers"].update(str(t) for t in (cap.get("required_oracle_tiers") or ()))
        slot["accelerate"].add(bool((cap.get("semantic") or {}).get("must_accelerate")))
        if epi:
            slot["epilogues"].add("+".join(epi))
    ops_out = {
        op: {
            "dtypes": sorted(s["dtypes"]),
            "accept": sorted(s["accept"]),
            "coverage": sorted(s["coverage"]),
            "epilogues": sorted(s["epilogues"]),
            "checked_by": {
                "oracle_tiers": sorted(s["tiers"]),
                "datapath_coverage": bool(s["coverage"]),
                "must_accelerate": True in s["accelerate"],
            },
        }
        for op, s in sorted(ops.items())
    }
    return {
        "schema": "verification_spec_v1",
        "target": getattr(te, "target", "?"),
        "n_capsules": len(caps),
        "ops": ops_out,
        "not_checked": _not_checked(ops_out, caps),
        "isa_docs": list(getattr(te, "isa_headers", []) or []),
    }


#: The standing admission, true of every target: nothing compares a submission to this document. It is
#: a rendering of what the graded corpus demands, not an independently enforced contract, and an
#: obligation the corpus omits is absent from this spec entirely rather than listed as unchecked.
_SPEC_HAS_NO_CHECKER = {
    "obligation": "this document",
    "scope": "the whole spec",
    "why_not_checked": (
        "no checker compares your submission to this spec. It is DERIVED from the graded capsules, so "
        "it restates what they demand -- an obligation the corpus does not demand is not weakened here, "
        "it is absent. Your verdict comes from the capsules and the tiers below, never from this file."
    ),
}


def _not_checked(ops_out: dict[str, dict], caps: list[dict]) -> list[dict[str, str]]:
    """Obligations this spec states (or implies) that NOTHING enforces, derived per op from the
    declarations. Each entry names the obligation, its scope, and why nothing looks."""
    out: list[dict[str, str]] = [dict(_SPEC_HAS_NO_CHECKER)]
    for op, spec in ops_out.items():
        checked = spec["checked_by"]
        if not checked["oracle_tiers"]:
            out.append(
                {
                    "obligation": "numeric acceptance",
                    "scope": f"`{op}`",
                    "why_not_checked": (
                        "no graded capsule declaring this op declares a required oracle tier, so the "
                        "acceptance policy stated above is not enforced for it by any tier"
                    ),
                }
            )
        if not checked["datapath_coverage"]:
            out.append(
                {
                    "obligation": "datapath coverage",
                    "scope": f"`{op}`",
                    "why_not_checked": (
                        "no graded capsule declaring this op declares `expected.instruction_classes`, "
                        "so nothing asserts which hardware classes your lowering must actually use"
                    ),
                }
            )
        if not checked["must_accelerate"]:
            out.append(
                {
                    "obligation": "work lands on the accelerator",
                    "scope": f"`{op}`",
                    "why_not_checked": (
                        "no graded capsule declaring this op declares `must_accelerate`, so a "
                        "numerically correct implementation that runs entirely on the host passes it"
                    ),
                }
            )
    declared_ops = set(ops_out)
    grammar = _whole_op_mnemonics()
    ungraded = sorted(grammar - declared_ops)
    if grammar and ungraded:
        out.append(
            {
                "obligation": "interface ops the frozen grammar defines",
                "scope": ", ".join(f"`{m}`" for m in ungraded),
                "why_not_checked": (
                    "the interface grammar defines these and no graded capsule uses them, so how your "
                    "package handles them is not measured. They can still appear in a module you are "
                    "handed, and a parser that fails closed on them is still the correct behaviour"
                ),
            }
        )
    return out


def render_markdown(te: Any) -> str:
    """The verification spec as a QA-team acceptance document (Markdown) the agent reads as its contract.
    States WHAT must hold to pass (ops, dtypes, acceptance policy, datapath coverage) and HOW to validate
    it as a bring-up engineer would — never a golden value, never how the oracle computes the answer."""
    spec = build_spec(te)
    L: list[str] = []
    L.append(f"# Verification spec — acceptance contract for `{spec['target']}`")
    L.append("")
    L.append(
        "_You are bringing up the software stack for brand-new hardware. Your world is the RTL, the "
        "shipped ISA/ABI docs, any example kernel, and this spec — there is **no pre-existing SW "
        "stack and no answer key**. This is the contract the verification team gives you: it says "
        "WHAT we test for and the pass criteria, not the expected outputs. Validate your work the way "
        "an engineer does — compute the operation's expected result yourself from the declared inputs, "
        "run your emitted artifact on the RTL, and debug divergences with the disassembler / trace / "
        "hardware-state tools._"
    )
    L.append("")
    L.append(
        f"**Scope:** {spec['n_capsules']} graded capsules across the operations below (the hidden "
        "holdout is not shown). Each capsule's `capsule.yaml` is the itemized test: its declared "
        "operation, input/output dtypes, acceptance policy, and required datapath coverage."
    )
    L.append("")
    L.append("## Target operations, datatypes, and acceptance")
    for op, d in spec["ops"].items():
        L.append(f"### `{op}`")
        L.append(f"- **datatypes (operands -> output):** {', '.join(d['dtypes']) or '?'}")
        if d["epilogues"]:
            L.append(f"- **epilogues:** {', '.join(d['epilogues'])}")
        L.append(
            f"- **acceptance policy:** {', '.join(d['accept']) or '?'}  "
            "(exact_int = bit-exact integer match; tolerance_float = within the stated atol/rtol)"
        )
        if d["coverage"]:
            L.append(f"- **datapath coverage (must actually exercise, not fake):** {', '.join(d['coverage'])}")
        L.append("")
    L.append("## What is tested (engineer terms)")
    L.append(
        "- **Functional correctness:** your emitted artifact, run on the RTL (the oracle), must "
        "compute the declared operation within the acceptance policy above. There is no stored "
        "golden you can read — the reference is the operation's own mathematical definition, which "
        "you can reproduce from the declared inputs."
    )
    L.append(
        "- **Datapath coverage:** the emitted stream must exercise the real hardware datapath (the "
        "required instruction classes), not shortcut the result."
    )
    L.append(
        "- **Legality:** every emitted instruction must be one the target's decoder accepts (ISA "
        "legality), and the program must terminate."
    )
    L.append("")
    L.append("## What each operation is CHECKED BY")
    L.append("")
    L.append("| operation | oracle tiers | datapath coverage | must land on the accelerator |")
    L.append("| --- | --- | --- | --- |")
    for op, d in spec["ops"].items():
        c = d["checked_by"]
        tiers = ", ".join(c["oracle_tiers"]) or "—"
        L.append(
            f"| `{op}` | {tiers} | {'yes' if c['datapath_coverage'] else 'NOT CHECKED'} "
            f"| {'yes' if c['must_accelerate'] else 'NOT CHECKED'} |"
        )
    L.append("")
    L.append("## What is NOT checked")
    L.append("")
    L.append(
        "This spec is DERIVED from the graded capsules, so it can only state what they demand. The "
        "entries below are obligations this document states or implies that **nothing enforces** — "
        'listed because an agent cannot otherwise tell "not required" from "required and nobody '
        'looks", and both were reaching you as the same silence.'
    )
    L.append("")
    for entry in spec["not_checked"]:
        L.append(f"- **{entry['obligation']}** ({entry['scope']}) — {entry['why_not_checked']}")
    if spec["isa_docs"]:
        L.append("")
        L.append("## ISA / ABI references")
        for h in spec["isa_docs"]:
            L.append(f"- `{h}`")
    L.append("")
    return "\n".join(L) + "\n"


def write_spec(
    te: Any,
    dest_dir: str | Path,
    *,
    name: str = "verification_spec.md",
    json_name: str = "verification_spec.json",
) -> Path:
    """Render + write the verification spec into ``dest_dir`` (e.g. the agent workspace root). Returns
    the markdown path. Regenerable at any time from the (answer-free) capsule declarations.

    The JSON sibling is written beside it and validates against
    ``src/merlin/resources/contracts/schemas/verification_spec.schema.json``. The markdown is for the agent to read;
    the JSON is what a checker, a report or the agent's own tooling can consume without parsing prose.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    out.write_text(render_markdown(te), encoding="utf-8")
    (dest / json_name).write_text(json.dumps(build_spec(te), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from merlin.targetgen.target_experiment import load_target_experiment

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=None, help="write the spec here (default: print to stdout)")
    a = ap.parse_args(argv)

    from .corpora import descriptor_path

    p = descriptor_path(a.target)
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
