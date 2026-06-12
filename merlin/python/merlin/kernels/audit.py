"""``kernel-audit`` CLI: marker-precision spot-check — "do I believe this motif?".

For each motif, sample N kernels (stratified across sources, deterministic seed), show the
exact code snippets that fired the markers plus ±context lines re-read from the source file.
A human (or an escalated LLM/agent) can then judge whether each snippet really evidences the
claimed optimization *decision* — the direct check that mined insights are real rather than
regex luck.

``--llm-judge`` adds one bounded verdict per sampled snippet via
:func:`merlin.common.llm.complete` (``confirms / unclear / refutes`` + a one-line why),
yielding a measured marker-precision estimate per motif. Cost is bounded by design
(≤ motifs × N calls, never per-kernel) and the audit is fully usable without any API key.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

# What each motif *claims*; shown in the audit and used verbatim in the judge prompt.
MOTIF_CLAIMS: dict[str, str] = {
    "packed_rhs": "the RHS/weight tensor is kept in a packed layout and consumed via a "
                  "packed-pointer/index advance or staged into a scratchpad for reuse",
    "reused_packed_rhs": "a packed RHS is measurably reused (>=2 consumers per pack)",
    "accumulator_lifetime": "an accumulator (often widened) stays live across the reduction "
                            "instead of being rematerialized",
    "accumulator_commit": "on a contraction op, the accumulator stays live across a "
                          "bias/requant/activation epilogue and commits to memory only after it",
    "epilogue_before_commit": "an epilogue (clamp/requant/activation/bias) is applied before "
                              "the result is stored",
    "vector_length_polymorphic": "the loop is vector-length-agnostic (vsetvl-style dynamic VL "
                                 "rather than a fixed SIMD width)",
    "tiling_blocking": "the computation is tiled/blocked to expose reuse",
    "double_buffering": "data movement is double-buffered to overlap with compute",
    "weight_stationary_dataflow": "weights stay stationary while activations stream "
                                  "(systolic dataflow choice)",
    "many_small_dispatches": "the kernel issues many small accelerator dispatches, paying "
                             "per-dispatch overhead",
    "intrinsic_lowering": "the kernel is written directly against target intrinsics",
}


def load_indexed(patterns: list[str]) -> list[tuple[dict, str | None]]:
    """Return ``(record, repo_root)`` pairs from index files (repo may be absent)."""
    paths: list[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit(f"no index files matched: {patterns}")
    out: list[tuple[dict, str | None]] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        repo = data.get("repo") if isinstance(data, dict) else None
        recs = data.get("records", data if isinstance(data, list) else [])
        out.extend((r, repo) for r in recs)
    return out


def sample_for_motif(pairs: list[tuple[dict, str | None]], motif: str, n: int,
                     seed: int) -> list[tuple[dict, str | None]]:
    """Sample up to ``n`` kernels firing ``motif``, round-robin across sources."""
    by_source: dict[str, list] = collections.defaultdict(list)
    for rec, repo in pairs:
        if motif in (rec.get("evidence", {}) or {}).get("motifs", []):
            by_source[rec.get("source", "?")].append((rec, repo))
    rng = random.Random(seed)
    for group in by_source.values():
        rng.shuffle(group)
    picked: list = []
    queues = [by_source[s] for s in sorted(by_source)]
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    return picked


def read_source(rec: dict, repo: str | None) -> str | None:
    """Best-effort re-read of the kernel's source text."""
    rel = str(rec.get("path", "")).split("::")[0]
    for f in ((Path(repo) / rel if repo else None), Path(rel)):
        if f and f.is_file():
            try:
                return f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def motif_markers(rec: dict, text: str | None, motif: str) -> list[str]:
    """Exact markers for ``motif``: re-fire the marker table on the source text.

    The persisted ``evidence.code_markers`` is a flat list across all motifs; re-firing
    recovers the per-motif mapping. Falls back to the flat list when the source is gone.
    """
    if text is not None:
        from merlin.kernels.markers import fired_markers
        return fired_markers(text, rec.get("target", "")).get(motif, [])
    return (rec.get("evidence", {}) or {}).get("code_markers", [])


def context_snippet(text: str | None, markers: list[str], context: int = 3) -> str | None:
    """±``context`` lines around the first fired marker in the source text."""
    if not text or not markers:
        return None
    lines = text.splitlines()
    needle = markers[0].splitlines()[0]
    for i, line in enumerate(lines):
        if needle and needle in line:
            lo, hi = max(0, i - context), min(len(lines), i + context + 1)
            return "\n".join(f"{j + 1:>5}| {lines[j]}" for j in range(lo, hi))
    return None


def judge_snippet(motif: str, claim: str, markers: list[str], snippet: str | None,
                  rec: dict) -> dict | None:
    """One bounded LLM verdict for a sampled snippet; None when LLM is unavailable."""
    from merlin.common.llm import complete
    body = snippet or "\n".join(markers)
    prompt = (
        f"You are auditing a compiler-research marker. The motif `{motif}` claims: {claim}.\n"
        f"Kernel: op={rec.get('op')} dtype={rec.get('dtype')} source={rec.get('source')} "
        f"target={rec.get('target')}\nMatched marker(s): {markers}\n"
        f"Code context:\n```\n{body}\n```\n"
        "Does this code genuinely evidence that DECISION (not merely contain the string)? "
        "Reply with exactly one line: `confirms|unclear|refutes: <one-sentence reason>`.")
    text = complete(prompt, max_tokens=120)
    if not text:
        return None
    verdict = text.split(":", 1)[0].strip().lower()
    if verdict not in ("confirms", "unclear", "refutes"):
        verdict = "unclear"
    return {"verdict": verdict, "why": text.split(":", 1)[-1].strip()}


def audit(pairs: list[tuple[dict, str | None]], motifs: list[str], n: int, seed: int,
          context: int, llm_judge: bool) -> tuple[str, dict]:
    """Return (markdown, summary) for the requested motifs."""
    md: list[str] = ["# Kernel-audit samples",
                     "",
                     "_Per motif: stratified random kernels (seed-deterministic), the marker "
                     "that fired, and source context. Judge each snippet: does it really show "
                     "the claimed decision?_", ""]
    summary: dict = {"motifs": {}, "llm_judge": llm_judge}
    for motif in motifs:
        picked = sample_for_motif(pairs, motif, n, seed)
        if not picked:
            continue
        claim = MOTIF_CLAIMS.get(motif, "(no claim registered)")
        md += [f"## `{motif}`", "", f"**Claim:** {claim}", ""]
        verdicts: collections.Counter = collections.Counter()
        for i, (rec, repo) in enumerate(picked, 1):
            text = read_source(rec, repo)
            markers = motif_markers(rec, text, motif)
            md.append(f"### {i}. `{rec.get('source')}/{rec.get('target')}` — "
                      f"`{rec.get('path')}` (op={rec.get('op')}, dtype={rec.get('dtype')})")
            md.append("- markers: " + (", ".join(f"`{m}`" for m in markers[:4]) or "_(n/a)_"))
            snip = context_snippet(text, markers, context)
            if snip:
                md += ["", "```", snip, "```"]
            else:
                md.append("- _(source file not available for context re-read)_")
            if llm_judge:
                v = judge_snippet(motif, claim, markers, snip, rec)
                if v:
                    verdicts[v["verdict"]] += 1
                    md.append(f"- llm_verdict: **{v['verdict']}** — {v['why']}")
                else:
                    md.append("- llm_verdict: _(LLM unavailable; set ANTHROPIC_API_KEY)_")
            md.append("")
        entry = {"sampled": len(picked)}
        if verdicts:
            judged = sum(verdicts.values())
            entry["llm_verdicts"] = dict(verdicts)
            entry["marker_precision_estimate"] = round(verdicts["confirms"] / judged, 2)
            md.append(f"**Motif verdict tally:** {dict(verdicts)} → marker precision ≈ "
                      f"{entry['marker_precision_estimate']}")
            md.append("")
        summary["motifs"][motif] = entry
    return "\n".join(md) + "\n", summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernel-audit", description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="index json files or globs")
    ap.add_argument("--motif", action="append", default=None,
                    help="motif(s) to audit (repeatable; default: all observed)")
    ap.add_argument("--n", type=int, default=8, help="samples per motif")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (deterministic)")
    ap.add_argument("--context", type=int, default=3, help="context lines around the marker")
    ap.add_argument("--out", default=None, help="audit markdown output path")
    ap.add_argument("--llm-judge", action="store_true",
                    help="one bounded LLM verdict per sample (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--json", action="store_true",
                    help="print a machine-readable summary JSON to stdout")
    args = ap.parse_args(argv)

    pairs = load_indexed(args.inputs)
    observed = collections.Counter(
        m for rec, _ in pairs for m in (rec.get("evidence", {}) or {}).get("motifs", []))
    motifs = args.motif or [m for m, _ in observed.most_common()]
    md, summary = audit(pairs, motifs, args.n, args.seed, args.context, args.llm_judge)

    out = Path(args.out) if args.out else Path("output/kernels/audit_samples.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    summary.update({"kernels": len(pairs), "out": str(out), "seed": args.seed, "n": args.n})
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        print(f"audited {len(summary['motifs'])} motifs over {len(pairs)} kernels -> {out}")
        for m, e in summary["motifs"].items():
            prec = (f"  precision≈{e['marker_precision_estimate']}"
                    if "marker_precision_estimate" in e else "")
            print(f"  {m}: {e['sampled']} samples{prec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
