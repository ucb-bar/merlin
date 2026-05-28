"""Pillar 2: retrospective accuracy of TargetGen's predicted write paths.

For each well-supported target, we mine the *historical* set of files
engineers actually edited during bring-up (curated commit list at
``tests/integration/.bring_up_commits.yaml``) and compare against
TargetGen's predicted ``write_paths`` for the matching capability spec.

Metrics:
  * precision = |predicted ∩ truth| / |predicted|
  * recall    = |predicted ∩ truth| / |truth|
  * f1        = 2·P·R / (P+R)

Files emitted:
  ``build/generated/retrospective_accuracy_report.json``  (per-target metrics)

Threshold gate: ``precision >= τ_p`` and ``recall >= τ_r`` per target.
Initial thresholds intentionally low (0.10 / 0.10) — the planner predicts
*directories*, while bring-up commits touched specific files inside them.
We use **directory-prefix matching**: a predicted ``compiler/src/merlin/
Dialect/Gemmini/IR/`` is credited as covering every truth-set file under
that prefix.

Markers: ``integration``, ``slow``. Skips cleanly if `git` is missing or
the repo isn't a git checkout.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen import build_support_plan, load_capability_spec  # noqa: E402
from targetgen.stage_map import build_modification_map  # noqa: E402

BRING_UP_YAML = Path(__file__).parent / ".bring_up_commits.yaml"
EXAMPLES = REPO_ROOT / "target_specs" / "examples"
REPORT_DIR = REPO_ROOT / "build" / "generated"

# Per-target regression gates. We set thresholds at the **current measured
# baseline** (rounded down with a small slack), not at a wishful target.
# This makes the test a regression detector for the planner — it catches
# the day a refactor *makes things worse* — rather than an aspirational
# benchmark that gets gamed by lowering thresholds whenever it fails.
#
# Known planner gaps surfaced by these baselines:
#
# 1. Dialect-name convention mismatch. Planner derives a CamelCase
#    directory from the spec's `identity.name` — e.g., `gemmini_mx` →
#    `GemminiMx/`. The historical bring-up used the short upstream name
#    (`Gemmini/`). Path normalisation in `_normalise_dialect_segment`
#    closes part of that gap, but only at the directory level.
#
# 2. Target-name length mismatch. Planner emits e.g.
#    `models/saturn_opu_v128.yaml` while the real file is
#    `models/saturn_opu.yaml`. This is the inverse problem: the spec
#    name encodes a hardware variant, but the historical bring-up used
#    the family name. The planner does not currently know about this
#    distinction.
#
# 3. Sample-tree CamelCase. Bring-up of LLVM-ukernel-only targets
#    (Saturn, SpacemiT) happens almost entirely under
#    `samples/<CamelTargetName>/` — a directory the planner does not
#    list in its `samples/` write_paths today.
#
# 4. build_tools/<Name>/ toolchain setup scripts are not predicted.
#
# Each of these is a candidate for a future planner improvement.
# Tightening the corresponding threshold is the regression sentinel for
# that improvement.
PER_TARGET_THRESHOLDS: dict[str, dict[str, float]] = {
    "gemmini_mx": {"precision": 0.10, "recall": 0.02},
    "npu_ucb": {"precision": 0.05, "recall": 0.01},
    "radiance_muon": {"precision": 0.10, "recall": 0.04},
    "saturn_opu_v128": {"precision": 0.00, "recall": 0.00},  # see gaps 2-4
    "spacemit_x60_xsmtvdot": {"precision": 0.00, "recall": 0.00},  # see gaps 2-4
}

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _git_check_repo() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("Merlin checkout is not a git repo")


def _files_in_commit(sha: str) -> set[str]:
    """Files touched by a single commit (relative paths)."""
    result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or path.startswith(p) for p in patterns)


def _truth_set(target_block: dict) -> set[str]:
    """Files touched across all curated bring-up commits for one target."""
    excludes = list(target_block.get("exclude_path_patterns", []))
    truth: set[str] = set()
    for sha in target_block["commits"]:
        for f in _files_in_commit(sha):
            if not _excluded(f, excludes):
                truth.add(f)
    return truth


def _predicted_set(capability_name: str) -> set[str]:
    """All write_paths the planner produces for ``capability_name``."""
    cap = EXAMPLES / capability_name / "capability.yaml"
    capabilities = load_capability_spec(cap)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
    paths: set[str] = set()
    for stage in modmap.stages:
        for w in stage.write_paths:
            paths.add(w)
    return paths


def _normalise_dialect_segment(path: str) -> str:
    """Lower-case and strip ``_`` from segments matching CamelCase dialect
    naming, so the planner's CamelCase predictions credit historical
    snake_case directories.

    Example: ``compiler/src/merlin/Dialect/GemminiMx/IR/`` and
    ``compiler/src/merlin/Dialect/Gemmini/IR/`` both normalise to the
    same prefix-comparable string. This is a deliberate compatibility
    shim — a real planner improvement would learn the existing dialect
    name convention from the repo. Tracked as a planner gap.
    """
    return path.replace("_", "").replace("-", "").lower()


def _matches_predicted(truth_file: str, predicted: set[str]) -> bool:
    """A truth file is covered if any predicted entry is a prefix of it.

    Comparison uses normalised paths so CamelCase / snake_case dialect
    names compare equal at the directory level.
    """
    nt = _normalise_dialect_segment(truth_file)
    for p in predicted:
        np = _normalise_dialect_segment(p)
        if p.endswith("/") and nt.startswith(np):
            return True
        if nt == np:
            return True
    return False


def _matches_truth(predicted_path: str, truth: set[str]) -> bool:
    """A predicted entry is credited if at least one truth file falls under it."""
    np = _normalise_dialect_segment(predicted_path)
    if predicted_path.endswith("/"):
        return any(_normalise_dialect_segment(t).startswith(np) for t in truth)
    return any(_normalise_dialect_segment(t) == np for t in truth)


def _scores(predicted: set[str], truth: set[str]) -> tuple[float, float, float, dict]:
    if not predicted:
        precision = 0.0
    else:
        hits_pred = sum(1 for p in predicted if _matches_truth(p, truth))
        precision = hits_pred / len(predicted)
    if not truth:
        recall = 0.0
    else:
        hits_truth = sum(1 for t in truth if _matches_predicted(t, predicted))
        recall = hits_truth / len(truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    misses_truth = sorted(t for t in truth if not _matches_predicted(t, predicted))
    misses_predicted = sorted(p for p in predicted if not _matches_truth(p, truth))
    return (
        precision,
        recall,
        f1,
        {
            "predicted_count": len(predicted),
            "truth_count": len(truth),
            "missed_truth_files": misses_truth[:30],
            "unsupported_predictions": misses_predicted[:30],
        },
    )


def _load_targets() -> dict:
    if not BRING_UP_YAML.exists():
        pytest.skip(f"bring-up commit list not found: {BRING_UP_YAML}")
    return yaml.safe_load(BRING_UP_YAML.read_text())


@pytest.fixture(scope="module")
def report() -> dict:
    _git_check_repo()
    targets = _load_targets()
    results: dict[str, dict] = {}
    for name, block in targets.items():
        truth = _truth_set(block)
        predicted = _predicted_set(block["capability"])
        precision, recall, f1, detail = _scores(predicted, truth)
        results[name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            **detail,
        }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "retrospective_accuracy_report.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def test_report_includes_all_targets(report: dict) -> None:
    targets = _load_targets()
    assert set(report.keys()) == set(
        targets.keys()
    ), f"report missing targets: {set(targets.keys()) - set(report.keys())}"


@pytest.mark.parametrize(
    "target",
    sorted(_load_targets().keys()) if BRING_UP_YAML.exists() else [],
)
def test_per_target_thresholds(report: dict, target: str) -> None:
    metrics = report[target]
    assert metrics["truth_count"] > 0, (
        f"{target}: bring-up commit list yielded zero files. Curated "
        f"commit SHAs in .bring_up_commits.yaml may be wrong."
    )
    assert metrics["predicted_count"] > 0, f"{target}: planner predicted no write paths"
    thresholds = PER_TARGET_THRESHOLDS.get(target)
    assert thresholds is not None, (
        f"{target}: missing PER_TARGET_THRESHOLDS entry. Add the current "
        f"measured precision/recall after running this test once."
    )
    assert metrics["precision"] >= thresholds["precision"], (
        f"{target}: precision {metrics['precision']:.3f} < "
        f"{thresholds['precision']:.2f} — planner regressed.\n"
        f"unsupported predictions (first 10): "
        f"{metrics['unsupported_predictions'][:10]}"
    )
    assert metrics["recall"] >= thresholds["recall"], (
        f"{target}: recall {metrics['recall']:.3f} < {thresholds['recall']:.2f} "
        f"— planner regressed.\n"
        f"missed truth files (first 10): {metrics['missed_truth_files'][:10]}"
    )


def test_report_is_emitted_to_disk(report: dict) -> None:
    out = REPORT_DIR / "retrospective_accuracy_report.json"
    assert out.exists(), f"report file not written: {out}"
    parsed = json.loads(out.read_text())
    assert parsed == report
