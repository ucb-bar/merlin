#!/usr/bin/env python3
"""Classify every kernel in a kernel repo by HOW IT WAS AUTHORED.

The amortization study compares agent-written kernels against a compiler. Both the
"reference performance" number and the frozen target bundle depend on knowing which
kernels a human wrote and which an agent already optimized. Getting that wrong
inflates the study in opposite directions at once: an agent-optimized kernel quoted
as a hand-written reference makes the reference look too strong, and the same kernel
sitting inside the bundle hands the kernel-generation arm its own answer.

Classification is EVIDENCE-BASED and FAILS CLOSED. A directory is only `hand` when a
human-authored source file is present and nothing links it to an agent run. Anything
undecidable is `unknown`, which is excluded from the bundle exactly like a known
generated kernel -- an unclassified kernel is not a safe kernel.

Four verdicts:

  hand                human-authored source present, no machine linkage
  agent_generated     named for, or matching, an LLM search run
  compiler_generated  emitted by a compiler-generation tool, not written
  artifact_only       no source at all -- compiled output and traces
                      (an ELF is still an answer; it disassembles)
  unknown             evidence conflicts or is absent -- treated as unsafe

Usage:
    audit_kernel_provenance.py --repo <path> [--repo <path> ...] --out prov.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Suffixes that carry authored intent -- a human or an agent WROTE these.
SOURCE_SUFFIXES = frozenset({".c", ".cpp", ".cc", ".S", ".s", ".h", ".hpp", ".mlir"})
# Suffixes a build or a simulator PRODUCES. Never evidence of authorship.
ARTIFACT_SUFFIXES = frozenset(
    {".elf", ".o", ".a", ".bin", ".log", ".runlog", ".sqlite", ".trace", ".dump", ".bak"}
)
# Scripts that drive a kernel but are not the kernel. Present in both hand and
# generated trees, so they decide nothing on their own.
HARNESS_SUFFIXES = frozenset({".py", ".sh", ".mk", ".md", ".jsonl", ".yaml", ".json"})


@dataclass
class KernelRecord:
    name: str
    repo: str
    verdict: str = "unknown"
    reasons: list[str] = field(default_factory=list)
    n_source: int = 0
    n_artifact: int = 0
    n_harness: int = 0
    source_files: list[str] = field(default_factory=list)
    git_authors: list[str] = field(default_factory=list)
    first_commit: str = ""
    first_subject: str = ""


def _git(repo: Path, *args: str) -> str:
    """Run git in `repo`; empty string on any failure (a missing repo is not fatal)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _classify_files(d: Path, rec: KernelRecord) -> None:
    """Count what kind of files the directory holds, recursively."""
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix
        if suf in SOURCE_SUFFIXES:
            rec.n_source += 1
            if len(rec.source_files) < 12:
                rec.source_files.append(p.relative_to(d).as_posix())
        elif suf in ARTIFACT_SUFFIXES:
            rec.n_artifact += 1
        elif suf in HARNESS_SUFFIXES:
            rec.n_harness += 1


def _machine_named(name: str, prefix_verdicts: dict[str, str]) -> str | None:
    """Verdict implied by a directory name that tags a machine-generation run.

    Compare the leading WHOLE TOKEN, not a substring: a substring test would also
    match a human kernel whose name merely contains the token. Leading underscores
    are stripped first -- a generator that names its output `_compgen_foo` yields an
    empty first token under a naive split, which silently matches nothing.
    """
    head = name.lstrip("_").split("_", 1)[0].lower()
    return prefix_verdicts.get(head)


def audit_repo(
    repo: Path,
    *,
    kernels_subdir: str,
    prefix_verdicts: dict[str, str],
    agent_problem_names: frozenset[str],
) -> list[KernelRecord]:
    root = repo / kernels_subdir
    if not root.is_dir():
        raise SystemExit(f"no kernels dir at {root}")

    records: list[KernelRecord] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rec = KernelRecord(name=d.name, repo=str(repo))
        _classify_files(d, rec)

        rel = f"{kernels_subdir}/{d.name}"
        log = _git(repo, "log", "--format=%an\t%h\t%s", "--", rel)
        if log:
            lines = [ln for ln in log.splitlines() if ln.strip()]
            rec.git_authors = sorted({ln.split("\t", 1)[0] for ln in lines})
            last = lines[-1].split("\t")  # oldest entry == first commit touching it
            if len(last) == 3:
                rec.first_commit, rec.first_subject = last[1], last[2]

        # --- verdict, most-specific evidence first ------------------------------
        named = _machine_named(d.name, prefix_verdicts)
        if named:
            rec.verdict = named
            rec.reasons.append(f"directory name is a {named.split('_')[0]}-generation run tag")
        elif d.name.lower() in agent_problem_names:
            rec.verdict = "agent_generated"
            rec.reasons.append("name matches a solution dir in the agent's own tree")
        elif rec.n_source > 0:
            rec.verdict = "hand"
            rec.reasons.append(f"{rec.n_source} authored source file(s) present")
            if not rec.git_authors:
                # Untracked source is not attributable; do not call it hand-written.
                rec.verdict = "unknown"
                rec.reasons.append("no git history for this path -- authorship unattributable")
        elif rec.n_artifact > 0:
            rec.verdict = "artifact_only"
            rec.reasons.append(
                f"no authored source; {rec.n_artifact} build/sim artifact(s) -- "
                "still an answer surface (an ELF disassembles)"
            )
        else:
            rec.reasons.append("no source and no artifacts -- nothing to classify")

        records.append(rec)
    return records


def _agent_problem_names(autocomp_root: Path | None) -> frozenset[str]:
    """Names of the agent framework's own solution directories.

    A kernel in the hardware repo sharing a name with a solved problem in the agent's
    tree is the agent's output landed upstream.
    """
    if autocomp_root is None:
        return frozenset()
    sols = autocomp_root / "sols"
    if not sols.is_dir():
        return frozenset()
    return frozenset(p.name.lower() for p in sols.iterdir() if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", required=True, type=Path,
                    help="kernel repo checkout to audit (repeatable)")
    ap.add_argument("--kernels-subdir", default="kernels")
    ap.add_argument("--agent-tree", type=Path, default=None,
                    help="the agent framework checkout, for solution-name cross-reference")
    ap.add_argument("--generated-prefix", action="append", default=None,
                    metavar="TOKEN=VERDICT",
                    help="leading name token marking machine-generated output, e.g. "
                         "autocomp=agent_generated (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", action="store_true", help="also print the summary as JSON")
    a = ap.parse_args(argv)

    spec = a.generated_prefix or ["autocomp=agent_generated", "compgen=compiler_generated"]
    prefix_verdicts: dict[str, str] = {}
    for item in spec:
        tok, _, verdict = item.partition("=")
        if not verdict:
            raise SystemExit(f"--generated-prefix needs TOKEN=VERDICT, got {item!r}")
        prefix_verdicts[tok.strip().lower()] = verdict.strip()
    problems = _agent_problem_names(a.agent_tree)

    all_recs: list[KernelRecord] = []
    per_repo: dict[str, dict] = {}
    for repo in a.repo:
        recs = audit_repo(
            repo,
            kernels_subdir=a.kernels_subdir,
            prefix_verdicts=prefix_verdicts,
            agent_problem_names=problems,
        )
        all_recs.extend(recs)
        counts = Counter(r.verdict for r in recs)
        per_repo[str(repo)] = {
            "head": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "n_kernels": len(recs),
            "verdicts": dict(sorted(counts.items())),
        }

    safe = [r.name for r in all_recs if r.verdict == "hand"]
    doc = {
        "schema": "kernel_provenance/v1",
        "generated_by": "llm_kernel_vs_compiler_v0/scripts/audit_kernel_provenance.py",
        "generated_prefixes": dict(sorted(prefix_verdicts.items())),
        "agent_tree": str(a.agent_tree) if a.agent_tree else None,
        "n_agent_problem_names": len(problems),
        "repos": per_repo,
        "policy": {
            "bundle_admits": ["hand"],
            "bundle_excludes": [
                "agent_generated", "compiler_generated", "artifact_only", "unknown",
            ],
            "note": (
                "unknown is excluded exactly like agent_generated: an unclassified "
                "kernel is not a safe kernel. artifact_only is excluded because a "
                "compiled ELF is still an answer."
            ),
        },
        "limitations": [
            "A `hand` verdict means 'a human authored source file is present and a "
            "human committed it'. It CANNOT distinguish source a human typed from "
            "agent output a human reviewed and committed under their own name. Any "
            "reference-performance claim built on these kernels inherits that limit "
            "and must say so; never call them 'expert'.",
            "Name-based verdicts are conservative by construction: a hand-written "
            "kernel that happens to sit under a generation-run prefix is excluded. "
            "That costs coverage, never safety.",
            "`artifact_only` directories hold no source but DO hold ELFs, which "
            "disassemble. They are answer surfaces and are excluded from the bundle "
            "for that reason, not because they are generated.",
        ],
        "n_bundle_safe": len(safe),
        "kernels": [asdict(r) for r in all_recs],
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    import yaml  # local import: the gate scripts must stay importable without yaml

    a.out.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))

    print(f"wrote {a.out}")
    for repo, info in per_repo.items():
        print(f"  {repo} @ {info['head'][:7]} ({info['branch']}): "
              f"{info['n_kernels']} kernels {info['verdicts']}")
    print(f"  bundle-safe (hand): {len(safe)}")
    if a.json:
        print(json.dumps({k: v for k, v in doc.items() if k != "kernels"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
