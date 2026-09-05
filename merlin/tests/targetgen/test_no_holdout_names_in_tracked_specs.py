"""A tracked conformance spec must not name a held-out capsule, or a local absolute path.

⚠️ REGRESSION. `CostFit.to_dict()` emitted a `sources` list -- the run file each cost sample came
from -- and the conformance spec embeds that dict twice, once per class and once corpus-wide. So
regenerating the requirement published, into a TRACKED file that every arm reads:

    /scratch/<user>/.../raw_baseline/rb_gemsg1/grading_hidden/runs/.../H0_matmul_hidden/capsule_result.json

That is 10 holdout capsule names in one block and 60 more through the per-class fits. A held-out
capsule's NAME is an answer key: knowing which shapes are graded privately is most of the advantage
the holdout exists to deny. `verify_no_cheat` caught it and failed BOTH targets; `check_no_answer_keys`
and `check_holdout_disjointness` did not, because neither looks at this surface.

The counts stay -- n_samples, r2, range -- because those are what make a fit auditable. Only the
provenance list is withheld, and a caller that genuinely needs it (a local diagnostic, never a tracked
artifact) passes `with_sources=True`.
"""
from __future__ import annotations

import yaml

from merlin.common.paths import merlin_dir

#: How a held-out capsule is spelled in this corpus. Read from the corpus rather than hardcoded, so a
#: target that names its holdouts differently is still checked.
_SPEC_DIR = merlin_dir() / "contract" / "capsules" / "conformance"


def _holdout_names() -> set[str]:
    root = merlin_dir() / "contract" / "capsules"
    names: set[str] = set()
    for cy in root.rglob("capsule.yaml"):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and str(doc.get("label") or "") not in ("public", "dev"):
            names.add(str(doc.get("name") or cy.parent.name))
    return names


def _strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, f"{path}[]")
    elif isinstance(obj, str):
        yield path, obj


def test_a_cost_fit_withholds_its_source_paths_by_default():
    """The narrow fix, tested at the unit rather than only through the artifact."""
    from merlin.targetgen.cert_cost import CostFit

    fit = CostFit(target="t", intercept_s=1.0, per_element_s=0.1, r2=0.9, n_samples=5,
                  elements_min=1, elements_max=10, metric="written_output_elements",
                  sources=["/abs/path/grading_hidden/runs/H0_matmul_hidden/capsule_result.json"])
    assert "sources" not in fit.to_dict(), "the provenance list is an answer key in a tracked artifact"
    assert "n_samples" in fit.to_dict(), "the counts are what make a fit auditable and must stay"
    assert "sources" in fit.to_dict(with_sources=True), "a local diagnostic may still ask for it"


def test_no_tracked_conformance_spec_names_a_held_out_capsule():
    held = _holdout_names()
    if not held:
        import pytest
        pytest.skip("no held-out capsules in this checkout (a worktree has no goldens/holdouts)")
    offenders = []
    for path in sorted(_SPEC_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for where, text in _strings(doc):
            for name in held:
                if name in text:
                    offenders.append((path.name, where, name))
    assert not offenders, (
        f"tracked conformance spec(s) name held-out capsules: {offenders[:6]}. A holdout's NAME is an "
        f"answer key -- knowing which shapes are graded privately is most of the advantage the holdout "
        f"exists to deny.")


def test_no_tracked_conformance_spec_embeds_a_local_absolute_path():
    """Secondary, and it matters for a public repo: these files are published."""
    offenders = []
    for path in sorted(_SPEC_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for where, text in _strings(doc):
            # A local checkout path, spelled structurally rather than by pattern: an absolute path
            # naming a home or scratch root is never portable evidence.
            parts = text.split("/")
            if text.startswith("/") and len(parts) > 3 and parts[1] in ("home", "scratch", "Users"):
                offenders.append((path.name, where, text[:80]))
    assert not offenders, (
        f"tracked conformance spec(s) embed local absolute paths: {offenders[:4]}. These files are "
        f"published; a path under someone's scratch root is neither portable nor reviewable.")
