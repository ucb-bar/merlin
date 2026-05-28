"""Baseline comparison helpers for the agent-driven proposal flow.

The deterministic ``classify_inventory`` is the natural baseline for two
distinct uses:

* **Tests**: ``propose_modifications`` is correct iff feeding it the
  deterministic classification reproduces the deterministic
  ``modification_map`` byte-for-byte. That self-consistency check is the
  regression gate.
* **Paper reporting**: when the agent's classification differs from the
  deterministic baseline on a held-out target, the delta is the signal —
  it tells us whether the LLM is adding value or drifting.

This module exposes two utilities:

  ``compare_to_deterministic`` — runs the deterministic classifier on the
  same source paths and returns an agreement summary (style-set jaccard,
  primary-integration match, per-stage write-path symmetric difference).

  ``score_against_history`` — for targets that already have a real
  ``capability.yaml`` and a curated bring-up commit list, computes
  precision / recall / F1 of any modification_map's predicted write paths
  against the historical truth set. Reuses the same logic as
  ``tests/integration/test_retrospective_accuracy.py`` but exposes it as
  a library so the agent-driven path can be scored too.

Both are pure functions; no LLM, no I/O beyond reading the source paths
(and, for ``score_against_history``, ``git diff-tree``).
"""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .intake import build_source_inventory, classify_inventory
from .model import ModificationMap


@dataclass(slots=True)
class StageDelta:
    stage: str
    only_in_agent: list[str]
    only_in_baseline: list[str]
    in_both: list[str]


@dataclass(slots=True)
class BaselineComparison:
    """Summary of how the agent's claim differs from the deterministic baseline.

    ``targetgen_styles_agreement`` is the Jaccard index over the canonical
    four-style set; 1.0 == identical. ``primary_integration_match`` is a
    boolean — easier to test against than a numeric score.
    """

    target: str
    agent_targetgen_styles: list[str]
    baseline_targetgen_styles: list[str]
    targetgen_styles_agreement: float
    agent_source_styles: list[str]
    baseline_source_styles: list[str]
    source_styles_agreement: float
    agent_primary_integration: str
    baseline_primary_integration: str
    primary_integration_match: bool
    baseline_confidence: float
    stage_deltas: list[StageDelta] = field(default_factory=list)
    overall_write_path_jaccard: float = 0.0


def compare_to_deterministic(
    target_name: str,
    source_paths: list[Path],
    agent_targetgen_styles: list[str],
    agent_source_styles: list[str],
    agent_primary_integration: str,
    agent_modification_map: ModificationMap | None = None,
) -> BaselineComparison:
    """Run the deterministic classifier and compare against the agent's claim.

    ``agent_modification_map`` is optional; when provided, per-stage
    write-path deltas are computed against the deterministic modification
    map. When omitted, only the classification-level agreement is reported.
    """
    inventory = build_source_inventory(target=target_name, sources=source_paths)
    baseline = classify_inventory(inventory)

    tg_jaccard = _jaccard(agent_targetgen_styles, baseline.targetgen_styles)
    src_jaccard = _jaccard(agent_source_styles, baseline.source_styles)

    stage_deltas: list[StageDelta] = []
    overall_jaccard = 0.0
    if agent_modification_map is not None:
        baseline_modmap = _modmap_from_classification(
            target_name,
            baseline.targetgen_styles,
            baseline.source_styles,
            baseline.primary_integration,
            baseline.confidence,
        )
        stage_deltas, overall_jaccard = _diff_modmaps(agent_modification_map, baseline_modmap)

    return BaselineComparison(
        target=target_name,
        agent_targetgen_styles=list(agent_targetgen_styles),
        baseline_targetgen_styles=list(baseline.targetgen_styles),
        targetgen_styles_agreement=tg_jaccard,
        agent_source_styles=list(agent_source_styles),
        baseline_source_styles=list(baseline.source_styles),
        source_styles_agreement=src_jaccard,
        agent_primary_integration=agent_primary_integration,
        baseline_primary_integration=baseline.primary_integration,
        primary_integration_match=agent_primary_integration == baseline.primary_integration,
        baseline_confidence=baseline.confidence,
        stage_deltas=stage_deltas,
        overall_write_path_jaccard=overall_jaccard,
    )


def _modmap_from_classification(
    target_name: str,
    targetgen_styles: list[str],
    source_styles: list[str],
    primary_integration: str,
    confidence: float,
) -> ModificationMap:
    """Build a ModificationMap using the same in-memory synthesis as
    ``targetgen_propose_modifications``. Kept here to keep ``baseline.py``
    independent of ``targetgen_mcp/tools.py``.
    """
    import tempfile

    from .intake.draft import render_loadable_draft
    from .loader import load_capability_spec
    from .model import Classification
    from .planner import build_support_plan
    from .stage_map import build_modification_map

    classification = Classification(
        target=target_name,
        source_styles=list(source_styles),
        targetgen_styles=list(targetgen_styles),
        primary_integration=primary_integration,
        confidence=confidence,
        missing_information=[],
        rationales=[],
    )
    payload = render_loadable_draft(classification)
    payload.pop("_unresolved_intake_notes", None)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(payload, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)
    try:
        capabilities = load_capability_spec(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    build_support_plan(capabilities)  # validates, side-effect free for our purposes
    return build_modification_map(capabilities, targetgen_styles=targetgen_styles)


def _diff_modmaps(agent: ModificationMap, baseline: ModificationMap) -> tuple[list[StageDelta], float]:
    by_stage_baseline = {s.stage: s for s in baseline.stages}
    deltas: list[StageDelta] = []
    all_agent: set[str] = set()
    all_baseline: set[str] = set()
    for ag_stage in agent.stages:
        bl_stage = by_stage_baseline.get(ag_stage.stage)
        ag_paths = set(ag_stage.write_paths)
        bl_paths = set(bl_stage.write_paths) if bl_stage else set()
        all_agent.update(ag_paths)
        all_baseline.update(bl_paths)
        deltas.append(
            StageDelta(
                stage=ag_stage.stage,
                only_in_agent=sorted(ag_paths - bl_paths),
                only_in_baseline=sorted(bl_paths - ag_paths),
                in_both=sorted(ag_paths & bl_paths),
            )
        )
    overall = _jaccard_sets(all_agent, all_baseline)
    return deltas, overall


def _jaccard(a: list[str], b: list[str]) -> float:
    return _jaccard_sets(set(a), set(b))


def _jaccard_sets(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return round(len(a & b) / len(union), 3)


# ---------------------------------------------------------------------------
# Score against git history (retrospective accuracy)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoryScore:
    """Precision/recall of a modification_map's write paths against curated
    bring-up commits for an existing target.

    ``precision`` is the fraction of predicted write_paths that are covered
    by at least one historical bring-up touch. ``recall`` is the fraction of
    historical files that fall under at least one predicted write_path.
    F1 is the harmonic mean.
    """

    target: str
    precision: float
    recall: float
    f1: float
    n_predicted: int
    n_truth: int
    matched_predicted: int
    matched_truth: int
    truth_uncovered_sample: list[str]


def score_against_history(
    target_name: str,
    modification_map: ModificationMap,
    bring_up_yaml: Path,
    repo_root: Path,
    sample_uncovered: int = 6,
) -> HistoryScore:
    """Score predicted write paths against curated bring-up commits.

    ``bring_up_yaml`` is the same file ``test_retrospective_accuracy.py``
    consumes — keeps the truth source single. The yaml has structure::

        targets:
          gemmini_mx:
            commits: [<sha>, <sha>, ...]
            exclude_path_patterns: [<glob>, ...]
    """
    if not bring_up_yaml.is_file():
        raise FileNotFoundError(f"bring-up commits yaml not found: {bring_up_yaml}")
    raw = yaml.safe_load(bring_up_yaml.read_text())
    # The yaml is flat (one top-level key per target), matching the layout
    # consumed by ``tests/integration/test_retrospective_accuracy.py``.
    target_block = raw.get(target_name)
    if target_block is None:
        raise KeyError(
            f"target {target_name!r} not present in {bring_up_yaml.name}; "
            "score_against_history only works for targets with curated history"
        )

    predicted: set[str] = set()
    for stage in modification_map.stages:
        for path in stage.write_paths:
            predicted.add(path)

    truth = _truth_set_from_commits(target_block, repo_root)

    matched_predicted = sum(1 for p in predicted if _matches_truth(p, truth))
    matched_truth = sum(1 for t in truth if _matches_predicted(t, predicted))

    precision = matched_predicted / len(predicted) if predicted else 0.0
    recall = matched_truth / len(truth) if truth else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    uncovered = sorted(t for t in truth if not _matches_predicted(t, predicted))
    return HistoryScore(
        target=target_name,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        n_predicted=len(predicted),
        n_truth=len(truth),
        matched_predicted=matched_predicted,
        matched_truth=matched_truth,
        truth_uncovered_sample=uncovered[:sample_uncovered],
    )


def _truth_set_from_commits(target_block: dict, repo_root: Path) -> set[str]:
    excludes = list(target_block.get("exclude_path_patterns", []))
    truth: set[str] = set()
    for sha in target_block["commits"]:
        for f in _files_in_commit(sha, repo_root):
            if not _excluded(f, excludes):
                truth.add(f)
    return truth


def _files_in_commit(sha: str, repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or path.startswith(p) for p in patterns)


def _normalise(path: str) -> str:
    return path.replace("_", "").replace("-", "").lower()


def _matches_predicted(truth_file: str, predicted: set[str]) -> bool:
    nt = _normalise(truth_file)
    for p in predicted:
        np = _normalise(p)
        if p.endswith("/") and nt.startswith(np):
            return True
        if nt == np:
            return True
    return False


def _matches_truth(predicted_path: str, truth: set[str]) -> bool:
    np = _normalise(predicted_path)
    if predicted_path.endswith("/"):
        return any(_normalise(t).startswith(np) for t in truth)
    return any(_normalise(t) == np for t in truth)
