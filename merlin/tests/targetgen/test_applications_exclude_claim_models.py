"""A claim model may never be an application, and the guard must be loud rather than filtering.

The application axis is a DERIVATION source: it decides which capsules exist. Admitting a claim model
there builds the corpus from the model it is then said to generalize to, which is the circularity
`merlin/contract/claim_models.yaml` exists to prevent. Silently dropping the entry would be worse than
raising -- a target would declare five applications and derive from four, with the difference visible
nowhere.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.artifacts import recaptures_dir
from merlin.common.paths import repo_root

_spec = importlib.util.spec_from_file_location(
    "ccc", repo_root() / "build_tools" / "scripts" / "check_conformance_coverage.py")
CCC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CCC)


class _TE:
    def __init__(self, applications):
        self.workload_spec = {"applications": applications}


def _bundle(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.mlir").write_text("module {}\n")
    return d


def test_a_claim_model_named_as_an_application_raises(monkeypatch, tmp_path):
    from merlin.targetgen import claim_models as CM

    monkeypatch.setattr(CCC, "recaptures_dir", lambda: tmp_path, raising=False)
    monkeypatch.setattr("merlin.common.artifacts.recaptures_dir", lambda: tmp_path)
    claim = CM.claim_models()[0]
    _bundle(tmp_path, f"{claim}_int8_consistent")
    _bundle(tmp_path, "some_other_model_int8_consistent")

    with pytest.raises(CCC.ClaimModelInApplications) as exc:
        CCC._applications(_TE([f"{claim}_int8_consistent", "some_other_model_int8_consistent"]))
    assert claim in str(exc.value)
    assert "generalize" in str(exc.value), "the error must say WHY, not just that it refused"


def test_a_non_claim_bundle_list_resolves(monkeypatch, tmp_path):
    monkeypatch.setattr("merlin.common.artifacts.recaptures_dir", lambda: tmp_path)
    _bundle(tmp_path, "some_other_model_int8_consistent")
    got = CCC._applications(_TE(["some_other_model_int8_consistent"]))
    assert set(got) == {"some_other_model_int8_consistent"}


def test_a_declared_bundle_that_is_absent_is_reported_not_silently_dropped(monkeypatch, tmp_path,
                                                                          capsys):
    """A missing bundle means the axis derives from less than the descriptor claims. That difference
    has to be visible, or a shrinking evidence base looks like a stable one."""
    monkeypatch.setattr("merlin.common.artifacts.recaptures_dir", lambda: tmp_path)
    _bundle(tmp_path, "present_model_int8_consistent")
    got = CCC._applications(_TE(["present_model_int8_consistent", "absent_model_fp32_consistent"]))
    assert set(got) == {"present_model_int8_consistent"}
    assert "absent_model_fp32_consistent" in capsys.readouterr().err


def test_the_directory_spelling_still_works(monkeypatch, tmp_path):
    """`ingest_applications.py` produces a directory of bundles; that path must keep resolving."""
    monkeypatch.setattr("merlin.common.artifacts.recaptures_dir", lambda: tmp_path)
    apps = tmp_path / "apps"
    _bundle(apps, "an_app_v0")
    assert set(CCC._applications(_TE(str(apps)))) == {"an_app_v0"}


def test_no_target_declares_a_claim_model_as_an_application():
    """The live descriptors, checked against the live claim set. This is the assertion that would have
    caught the circularity if it were ever reintroduced by an edit."""
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    checked = 0
    for target in ("gemmini", "atlas", "radiance"):
        te = load_target_experiment(descriptor_path(target))
        CCC._applications(te)          # raises if any declared application is a claim model
        checked += 1
    assert checked == 3


def test_the_graded_roster_and_the_derivation_set_are_disjoint():
    """A model may not be BOTH what the corpus is built from and what the corpus claims to generalize
    to. `workload_spec.models` is the claim's denominator; `workload_spec.applications` is a
    derivation source."""
    from merlin.targetgen import claim_models as CM
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    for target in ("gemmini", "atlas", "radiance"):
        ws = dict(getattr(load_target_experiment(descriptor_path(target)), "workload_spec", None) or {})
        roster = [str(m) for m in (ws.get("models") or ())]
        apps = [str(a) for a in (ws.get("applications") or ())]
        assert roster, f"{target} declares no graded roster"
        assert apps, f"{target} declares no derivation applications"

        # Resolved the way `report_generalization_claim` resolves it -- a roster entry names a MODEL
        # and matches its capture bundles by prefix, so `resnet50` reaches `resnet50_v1_5_fp32_*`.
        # Asserting on the name shape instead would test our spelling, not the holdout.
        available = sorted(d.name for d in Path(recaptures_dir()).iterdir() if d.is_dir())
        for model in roster:
            hits = [b for b in available if b == model or b.startswith(model + "_")]
            assert hits, f"{target}: graded roster model {model!r} has no capture at all"
            assert all(CM.is_claim_bundle(b) for b in hits), (
                f"{target}: {model!r} is GRADED as the generalization claim, but some of its captures "
                f"are not held out of derivation: "
                f"{[b for b in hits if not CM.is_claim_bundle(b)]}. The corpus would be built from the "
                f"model it is then said to generalize to.")

        for app in apps:
            assert not CM.is_claim_bundle(app), f"{target}: application {app!r} is a claim model"

        # And the two sets may not overlap through the prefix relation either.
        for model in roster:
            assert not any(a == model or a.startswith(model + "_") for a in apps), (
                f"{target}: {model!r} is both graded and derived from")
