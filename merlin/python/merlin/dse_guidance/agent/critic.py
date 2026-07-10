"""Devil's-advocate critic slot — an agent proposes over-claims, a deterministic gate disposes.

The agent reads an emitted insight-mining run and proposes critiques (over-claims, unsupported
leaps, missing caveats) in the *interpretation* layer. Each proposed critique MUST quote an exact
substring of a real artifact; the deterministic ``citation_gate`` rejects any critique that does
not. This is the propose/dispose pattern from targetgen's kernel_slot, applied to prose review:
the agent never produces a number, and an ungrounded critique cannot enter the report.

Usage (the agent is optional; the gate is always testable with an injected runner):

    from merlin.dse_guidance.agent import critic
    result = critic.run_critic(run_dir)          # uses headless `claude -p`
    critic.emit_critique(result, run_dir)        # writes devils_advocate_critique.md
"""
from __future__ import annotations

from pathlib import Path

from merlin.dse_guidance.agent import claude_cli

# the artifacts the critic is allowed to read + cite against (the emitted run folder)
_CITABLE_GLOBS = ("*.md", "*.csv")
_SEVERITIES = ("low", "medium", "high")


def _artifact_blob(run_dir: Path) -> str:
    """Concatenated text of every citable artifact in the run — the ground truth for the gate."""
    parts = []
    for pat in _CITABLE_GLOBS:
        for p in sorted(Path(run_dir).glob(pat)):
            try:
                parts.append(p.read_text())
            except Exception:
                continue
    return "\n".join(parts)


def _norm(s: str) -> str:
    """Whitespace-normalize for substring matching (the agent may reflow quotes)."""
    return " ".join((s or "").split())


def build_prompt(run_dir: Path) -> str:
    """The critic task: find over-claims in the digest, each grounded in a quoted artifact line."""
    digest = Path(run_dir) / "DSE_FINDINGS.md"
    body = digest.read_text() if digest.is_file() else _artifact_blob(run_dir)[:24000]
    files = sorted(p.name for pat in _CITABLE_GLOBS for p in Path(run_dir).glob(pat))
    return (
        "You are a skeptical hardware-DSE reviewer. Below is a workload-analysis run that claims to "
        "guide accelerator design-space exploration WITHOUT measuring any hardware. Find places where "
        "it OVER-CLAIMS, makes an unsupported leap, hides a caveat, or presents a corpus-specific "
        "result as general. Be adversarial but fair.\n\n"
        "HARD RULES:\n"
        "- Do NOT invent or recompute numbers. Critique only the interpretation.\n"
        "- Every critique MUST quote an exact substring ('cite') copied verbatim from the run text "
        "below, so it can be verified mechanically. If you cannot quote it, do not raise it.\n"
        "- Output ONLY a JSON array. Each item: "
        '{"claim": "<the over-claim, in your words>", "severity": "low|medium|high", '
        '"cite": "<exact substring copied from the text>", "suggested_fix": "<one line>"}.\n\n'
        f"Artifacts in this run: {', '.join(files)}\n\n"
        "=== RUN TEXT (DSE_FINDINGS.md) ===\n" + body)


def citation_gate(items, run_dir) -> dict:
    """Deterministic dispose step: accept a critique only if its 'cite' is a real, non-empty
    substring of the run artifacts and its severity is valid. Returns accepted/rejected lists."""
    blob = _norm(_artifact_blob(run_dir))
    accepted, rejected = [], []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            rejected.append({"item": it, "reason": "not an object"})
            continue
        cite = _norm(it.get("cite", ""))
        sev = str(it.get("severity", "")).lower()
        if len(cite) < 8:
            rejected.append({"item": it, "reason": "citation missing/too short"})
        elif cite not in blob:
            rejected.append({"item": it, "reason": "citation not found in artifacts"})
        elif sev not in _SEVERITIES:
            rejected.append({"item": it, "reason": f"invalid severity {sev!r}"})
        else:
            accepted.append({"claim": str(it.get("claim", ""))[:300], "severity": sev,
                             "cite": _norm(it.get("cite", ""))[:300],
                             "suggested_fix": str(it.get("suggested_fix", ""))[:300]})
    return {"accepted": accepted, "rejected": rejected, "n_proposed": len(items or [])}


def run_critic(run_dir, runner=None, *, model: str = "opus") -> dict:
    """Run the critic over an emitted run. ``runner`` is a callable(prompt)->{'text':...}; defaults
    to the headless ``claude -p`` runtime (raises AgentError if the CLI is unavailable). Inject a
    runner in tests to exercise the gate without a live model."""
    run_dir = Path(run_dir)
    runner = runner or (lambda p: claude_cli.run_agent(p, model=model))
    out = runner(build_prompt(run_dir))
    items = claude_cli.extract_json(out.get("text", ""))
    result = citation_gate(items, run_dir)
    result["usage"] = out.get("usage", {})
    return result


def emit_critique(result: dict, run_dir) -> Path:
    """Write the gated critique as a report in the run folder (non-committed)."""
    acc, rej = result["accepted"], result["rejected"]
    L = ["# Devil's-advocate critique (agent-proposed, citation-gated)\n",
         "> An agent proposed critiques of this run's interpretation; a deterministic gate kept only "
         "those that quote a real artifact line. The agent produced no numbers. "
         f"**{len(acc)} accepted / {result['n_proposed']} proposed** "
         f"({len(rej)} rejected as ungrounded).\n"]
    if acc:
        L.append("| severity | over-claim | cited from run | suggested fix |")
        L.append("|---|---|---|---|")
        for a in sorted(acc, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["severity"]]):
            L.append(f"| {a['severity']} | {a['claim']} | `{a['cite'][:80]}` | {a['suggested_fix']} |")
    else:
        L.append("_No grounded over-claims survived the citation gate._")
    if rej:
        L.append(f"\n## Rejected as ungrounded ({len(rej)})\n")
        for r in rej[:20]:
            L.append(f"- {r['reason']}")
    out = Path(run_dir) / "devils_advocate_critique.md"
    out.write_text("\n".join(L) + "\n")
    return out
