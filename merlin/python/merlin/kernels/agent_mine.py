"""Dual-mode agent mining over kernel dossiers — the high-value judgment layer (algorithm intent,
exemplary?, contract refinements, compiler-lever implications) the deterministic tooling can't do.

Per the user ("we don't know what works"), BOTH agent strategies are supported and compared:
  * mode="representative" — one agent call per CLUSTER (cluster.representatives); the finding is
    promoted to the whole cluster. Cheap (≈clusters, ~50% of kernels here), consistent.
  * mode="per_kernel"     — one agent call per kernel. Maximal nuance, costly + noisier.
`compare_modes` runs both and reports n_calls / coverage / agreement so we can MEASURE which mines
better rather than assume.

The agent is the injectable ``llm_fn`` (default merlin.common.llm.complete — uses the Anthropic API
when ANTHROPIC_API_KEY is set, else returns None so the harness degrades to deterministic-only).
The prompt is a versioned artifact: merlin/prompts/rvv_mining_v{V}.md. The library prepares prompts
+ post-processes findings; it never reads raw files for the agent — only the dossier.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .cluster import cluster_dossiers
from .dossier import KernelDossier

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"


def prompt_path(version: int = 1) -> Path:
    return _PROMPT_DIR / f"rvv_mining_v{version}.md"


def build_prompt(d: KernelDossier, *, version: int = 1, code_max: int = 2500) -> str:
    tmpl = prompt_path(version).read_text(encoding="utf-8")
    code = _read_code(d)[:code_max]
    return tmpl.format(source=d.source, op=d.op, dtype=d.dtype,
                       decisions=json.dumps(d.decisions), struct=json.dumps(d.struct),
                       motifs=json.dumps(d.motifs),
                       framework_contract=json.dumps(d.framework_contract.get("operand_prepack", {})),
                       code=code)


def _read_code(d: KernelDossier) -> str:
    try:
        return Path(d.path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def parse_findings(text: str | None) -> dict[str, Any]:
    """Tolerant JSON parse of an agent reply (strips code fences / prose around the object)."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def _default_llm(prompt: str) -> str | None:
    from ..common.llm import complete
    return complete(prompt, max_tokens=600)


def mine(dossiers: list[KernelDossier], *, mode: str = "representative",
         llm_fn: Callable[[str], "str | None"] | None = None, version: int = 1,
         max_calls: int | None = None) -> dict[str, Any]:
    """Run agent mining in ``mode``. Returns {mode, n_calls, n_kernels_covered, findings}. Each
    finding: {path|signature, n_members, finding(parsed), raw}. Never raises on an LLM miss
    (records finding={})."""
    llm_fn = llm_fn or _default_llm
    if mode == "representative":
        clusters = cluster_dossiers(dossiers)
        targets = [(c.representative, c.members) for c in clusters]
    elif mode == "per_kernel":
        targets = [(d, [d]) for d in dossiers]
    else:
        raise ValueError(f"unknown mode {mode!r} (want representative|per_kernel)")
    if max_calls is not None:
        targets = targets[:max_calls]

    findings = []
    for rep, members in targets:
        raw = llm_fn(build_prompt(rep, version=version))
        findings.append({"path": rep.path, "signature": list(rep.signature()),
                         "n_members": len(members), "finding": parse_findings(raw),
                         "raw": raw})
    return {"mode": mode, "version": version, "n_calls": len(findings),
            "n_kernels_covered": sum(f["n_members"] for f in findings), "findings": findings}


def compare_modes(dossiers: list[KernelDossier], *, llm_fn: Callable | None = None,
                  version: int = 1, max_calls: int | None = None) -> dict[str, Any]:
    """Run BOTH modes and report which mines better: call counts, coverage, and (when both ran on
    overlapping kernels) agreement on `is_exemplary`/`compiler_levers`. The experiment the user
    asked for instead of assuming representative is best."""
    rep = mine(dossiers, mode="representative", llm_fn=llm_fn, version=version, max_calls=max_calls)
    per = mine(dossiers, mode="per_kernel", llm_fn=llm_fn, version=version, max_calls=max_calls)
    # agreement: for each per-kernel finding, does its cluster representative's finding agree?
    rep_by_sig = {tuple(f["signature"]): f["finding"] for f in rep["findings"]}
    agree = total = 0
    for f in per["findings"]:
        rf = rep_by_sig.get(tuple(f["signature"]))
        if rf and "is_exemplary" in rf and "is_exemplary" in f["finding"]:
            total += 1
            agree += int(rf["is_exemplary"] == f["finding"]["is_exemplary"])
    return {
        "representative": {"n_calls": rep["n_calls"], "covers": rep["n_kernels_covered"]},
        "per_kernel": {"n_calls": per["n_calls"], "covers": per["n_kernels_covered"]},
        "call_ratio": round(per["n_calls"] / max(1, rep["n_calls"]), 2),
        "exemplary_agreement": (round(agree / total, 3) if total else None),
        "agreement_n": total,
        "rep_findings": rep["findings"], "per_findings": per["findings"],
    }
