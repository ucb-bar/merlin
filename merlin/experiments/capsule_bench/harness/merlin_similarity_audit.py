#!/usr/bin/env python3
"""Operator-side similarity / leakage audit for a frozen merlin_assisted submission.

After a merlin_assisted run is FROZEN, compare its generated artifact against the forbidden prior
backends (which the agent never saw) to check for accidental copying / leakage that would undermine
the A/B comparison. This runs OUTSIDE the agent sandbox and reads the prior backends directly; it must
NEVER be invoked inside the agent workspace and its inputs are never exposed to the agent.

It reports, per file: exact matches (sha256), high text-similarity (difflib ratio over normalized
source), copied manifest structure, and suspicious shared constants/comments — then a comparability
verdict.

Usage:
  merlin_similarity_audit.py --run-id pilot_merlin_0001 [--arm merlin_assisted]
  merlin_similarity_audit.py --submission <dir> --run-id <id>   # audit an explicit dir (smoke test)
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

import _common as C
from merlin.targetgen.target_experiment import load_target_experiment

# Forbidden prior backends the merlin agent must not have copied (operator-visible only). The set is the
# active target's declared answer surfaces (descriptor `answer_surfaces.prior_backends`), resolved under
# out/artifacts/targets/<target>/ — no committed gemmini list.
_TE = load_target_experiment(C.EXP / "target_experiment.yaml")
PRIOR_BACKENDS = [C.REPO / "out/artifacts/targets" / C.TARGET / b for b in _TE.prior_backends]
PRIOR_REPORTS = C.REPORTS

_SKIP = {"build", "__pycache__", ".git", "CANARY_FORBIDDEN.txt"}
_SRC_EXT = {".py", ".td", ".cpp", ".h", ".hpp", ".cc", ".mlir", ".yaml", ".yml", ".json", ".txt",
            ".cmake", ".md"}
HIGH_SIM = 0.85   # difflib ratio >= this on normalized source -> flag as high-similarity
MANIFEST_SIM = 0.80


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    """Normalize source for similarity: drop blank lines + leading/trailing ws + line comments so we
    compare structure/logic, not formatting. (Comment text is checked separately.)"""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = " ".join(s.split())          # collapse internal whitespace (structured, no regex)
        out.append(s)
    return "\n".join(out)


def _src_files(root: Path) -> dict[str, Path]:
    """relpath -> Path for source-like files under root (skipping build/cache/canary)."""
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _SKIP & set(p.parts) or p.name in _SKIP:
            continue
        if p.suffix.lower() in _SRC_EXT:
            out[p.relative_to(root).as_posix()] = p
    return out


def _shared_long_lines(a: str, b: str, minlen: int = 24, k: int = 5) -> list[str]:
    """Distinctive non-trivial lines present verbatim in both (constants/comments copy signal)."""
    sa = {ln.strip() for ln in a.splitlines() if len(ln.strip()) >= minlen}
    sb = {ln.strip() for ln in b.splitlines() if len(ln.strip()) >= minlen}
    common = sorted(sa & sb, key=len, reverse=True)
    # drop boilerplate that is legitimately shared (the contract vocabulary)
    # ``.insn r`` alone is the boilerplate signal -- the opcode that follows is the TARGET's, so
    # pinning one here made every other target's inline-asm lines read as "distinctive copying".
    boiler = ("artifact_type", "mlir_oot_target_backend", "integrity_exempt", f"{C.TARGET}_kernel",
              ".insn r ", "merlin_iface", "command_buffer", "from __future__ import")
    distinctive = [c for c in common if not any(b_ in c for b_ in boiler)]
    return distinctive[:k]


def audit(submission: Path) -> dict:
    sub = _src_files(submission)
    sub_hashes = {rel: _sha(p) for rel, p in sub.items()}
    exact, high, shared = [], [], []

    for backend in PRIOR_BACKENDS:
        if not backend.exists():
            continue
        prior = _src_files(backend)
        prior_hashes = {rel: _sha(p) for rel, p in prior.items()}
        prior_by_hash = {}
        for rel, h in prior_hashes.items():
            prior_by_hash.setdefault(h, []).append(rel)
        for rel, p in sub.items():
            h = sub_hashes[rel]
            # exact content match anywhere in this prior backend (same or different relpath)
            if h in prior_by_hash:
                exact.append({"submission_file": rel, "backend": backend.name,
                              "matches": prior_by_hash[h]})
                continue
            na = _norm(p.read_text(errors="ignore"))
            # compare against the same-relpath prior file if present, else best ratio across the backend
            cands = [prior[rel]] if rel in prior else list(prior.values())
            best_ratio, best_rel = 0.0, None
            for cp in cands:
                r = difflib.SequenceMatcher(None, na, _norm(cp.read_text(errors="ignore"))).ratio()
                if r > best_ratio:
                    best_ratio, best_rel = r, cp.relative_to(backend).as_posix()
            if best_ratio >= HIGH_SIM:
                sl = _shared_long_lines(p.read_text(errors="ignore"),
                                        (backend / best_rel).read_text(errors="ignore"))
                high.append({"submission_file": rel, "backend": backend.name,
                             "prior_file": best_rel, "ratio": round(best_ratio, 3),
                             "shared_distinctive_lines": sl})

    # manifest-structure copy: compare submission/manifest.yaml key-shape to each prior manifest
    manifest_findings = []
    sub_manifest = submission / "manifest.yaml"
    if sub_manifest.exists():
        sm = _norm(sub_manifest.read_text(errors="ignore"))
        for backend in PRIOR_BACKENDS:
            pm = backend / "manifest.yaml"
            if pm.exists():
                r = difflib.SequenceMatcher(None, sm, _norm(pm.read_text(errors="ignore"))).ratio()
                if r >= MANIFEST_SIM:
                    manifest_findings.append({"backend": backend.name, "ratio": round(r, 3)})

    return {"n_submission_files": len(sub), "exact_matches": exact,
            "high_similarity": high, "manifest_structure": manifest_findings}


def _verdict(res: dict) -> tuple[str, str]:
    if res["exact_matches"]:
        return ("COPY_DETECTED",
                "Exact file-content matches with a prior backend — comparability COMPROMISED; "
                "investigate before reporting this run.")
    if res["high_similarity"] or res["manifest_structure"]:
        return ("REVIEW",
                "High-similarity files and/or manifest-structure overlap — operator review required "
                "to decide if it is convergent design (acceptable) or copying (not).")
    return ("CLEAN", "No exact matches, no high-similarity files, no manifest-structure copy.")


def _write_report(res: dict, run_id: str, submission: Path, out: Path) -> None:
    verdict, why = _verdict(res)
    L = [f"# merlin similarity / leakage audit — {run_id}", "",
         "Operator-side, post-freeze. Compares the generated artifact against the forbidden prior "
         "backends (which the agent never saw). Inputs are NOT exposed to the agent.", "",
         f"- submission: `{submission}`",
         f"- prior backends compared: {', '.join(b.name for b in PRIOR_BACKENDS if b.exists())}",
         f"- source files in submission: {res['n_submission_files']}",
         f"- high-similarity threshold: {HIGH_SIM} (normalized difflib ratio)", "",
         f"## Verdict: {verdict}", "", why, "",
         "## Exact content matches", ""]
    if res["exact_matches"]:
        for e in res["exact_matches"]:
            L.append(f"- `{e['submission_file']}` == `{e['backend']}`: {e['matches']}")
    else:
        L.append("_none_")
    L += ["", "## High-similarity files", ""]
    if res["high_similarity"]:
        L.append("| submission file | backend | prior file | ratio | shared distinctive lines |")
        L.append("|---|---|---|---|---|")
        for h in res["high_similarity"]:
            sl = " ⏎ ".join(s[:60] for s in h["shared_distinctive_lines"]) or "—"
            L.append(f"| `{h['submission_file']}` | {h['backend']} | `{h['prior_file']}` | "
                     f"{h['ratio']} | {sl} |")
    else:
        L.append("_none_")
    L += ["", "## Manifest-structure overlap", ""]
    if res["manifest_structure"]:
        for m in res["manifest_structure"]:
            L.append(f"- {m['backend']}: ratio {m['ratio']}")
    else:
        L.append("_none above threshold_")
    L += ["", "## Comparability impact", "",
          {"COPY_DETECTED": "This run is NOT comparable as-is — a prior backend was reproduced "
                            "verbatim. Do not report it as an independent merlin_assisted result.",
           "REVIEW": "Comparability is plausible but requires operator sign-off on the flagged files "
                     "(convergent design vs copying).",
           "CLEAN": "No leakage signal; the artifact appears independently authored. Comparable."}[verdict],
          ""]
    out.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", default="merlin_assisted")
    ap.add_argument("--submission", default=None,
                    help="explicit submission dir (default runs/<arm>/<run-id>/submission)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    submission = Path(a.submission) if a.submission else (C.RUNS / a.arm / a.run_id / "submission")
    if not submission.exists():
        print(f"submission not found: {submission}", file=__import__("sys").stderr)
        return 2
    res = audit(submission)
    out = Path(a.out) if a.out else (C.REPORTS / f"merlin_similarity_audit_{a.run_id}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(res, a.run_id, submission, out)
    verdict, _ = _verdict(res)
    print(f"[similarity_audit] {a.run_id}: {verdict}  "
          f"(exact={len(res['exact_matches'])} high_sim={len(res['high_similarity'])} "
          f"manifest={len(res['manifest_structure'])}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
