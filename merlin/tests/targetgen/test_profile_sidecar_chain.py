"""The sidecar chain must read what the writers write, and a dotted stem is not a target.

Two defects that made the SMT counterexample path dead on arrival, and one that was live and wider.
"""

from __future__ import annotations

import pytest
import yaml
from merlin_experiments.phase0 import profiles as GC
from merlin_experiments.phase0.declarations import all_declarations


def _profiles(tmp_path, monkeypatch):
    """Point the generator at a scratch profiles dir. `_merge_shared_perf` reads the shared template
    unconditionally, so it has to exist even when the test is about something else."""
    (tmp_path / "_perf.yaml").write_text(yaml.safe_dump({"capsules": [], "sweeps": []}))
    return tmp_path


def test_a_dotted_stem_is_never_returned_as_a_target(tmp_path):
    """⚠️ REGRESSION, and it was LIVE. `Path.stem` strips only the last suffix, so
    `gemmini.synth.yaml` yielded the stem `gemmini.synth`, returned beside `gemmini`. Six real targets
    came back as twelve, and `main()` uses this list as the default when `--target` is absent -- so a
    bare run generated phantom corpora from profile fragments."""
    for name in ("fixture.yaml", "fixture.synth.yaml", "fixture.smt.yaml", "fixture.hidden.yaml", "_perf.yaml"):
        (tmp_path / name).write_text("capsules: []\n")
    targets = GC.profile_targets(profiles_root=tmp_path)
    assert targets, "no targets at all means the glob is broken, not that the leak is fixed"
    dotted = [t for t in targets if "." in t]
    assert dotted == [], f"sidecar stems returned as targets: {dotted}"


def test_every_returned_target_has_a_real_profile():
    declarations = all_declarations()
    assert declarations
    for declaration in declarations:
        assert declaration.recipe.is_file()


def test_the_smt_sidecar_is_in_the_chain_load_profile_reads(tmp_path, monkeypatch):
    """⚠️ REGRESSION. The former implicit writer used `<target>.smt.yaml`; `load_profile` read
    exactly three filenames and that was not one of them, so every solver-found counterexample went to
    a file nothing opened. The module's own docstring asserted a glob that does not exist."""
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"datapath": {}, "capsules": [{"name": "A0_base"}]}))
    (tmp_path / "t.smt.yaml").write_text(yaml.safe_dump({"capsules": [{"name": "CX_contraction_i8_16x16x16"}]}))
    names = [c["name"] for c in GC.load_profile("t", profiles_root=tmp_path)["capsules"]]
    assert "CX_contraction_i8_16x16x16" in names, (
        "a counterexample entry written by verify.counterexamples must reach the generator"
    )
    assert "A0_base" in names, "the public profile must still be merged"


def test_the_synth_sidecar_still_merges(tmp_path, monkeypatch):
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"datapath": {}, "capsules": []}))
    (tmp_path / "t.synth.yaml").write_text(yaml.safe_dump({"capsules": [{"name": "SY_cell_i8"}]}))
    assert [c["name"] for c in GC.load_profile("t", profiles_root=tmp_path)["capsules"]] == ["SY_cell_i8"]


def test_an_absent_smt_sidecar_is_not_an_error(tmp_path, monkeypatch):
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"datapath": {}, "capsules": [{"name": "A0"}]}))
    assert [c["name"] for c in GC.load_profile("t", profiles_root=tmp_path)["capsules"]] == ["A0"]


def test_the_counterexample_writer_and_reader_share_explicit_path(tmp_path):
    from merlin.verify.counterexamples import write_profile

    recipe = tmp_path / "public.yaml"
    template = tmp_path / "performance.yaml"
    sidecar = tmp_path / "generated" / "counterexamples.yaml"
    recipe.write_text(yaml.safe_dump({"datapath": {}, "capsules": [{"name": "A0"}]}))
    template.write_text(yaml.safe_dump({"capsules": [], "sweeps": []}))
    write_profile(sidecar, [{"name": "CX_fixture", "value": 1}], provenance={"lattice_source": "fixture"})
    write_profile(sidecar, [{"name": "CX_fixture", "value": 2}], provenance={"lattice_source": "fixture"})
    loaded = GC.load_profile("fixture", recipe=recipe, performance_template=template, smt_profile=sidecar)
    assert loaded["capsules"] == [{"name": "A0"}, {"name": "CX_fixture", "value": 2}]
    assert yaml.safe_load(sidecar.read_text())["provenance"]["lattice_source"] == "fixture"


def test_profiles_require_explicit_input_while_legacy_manifest_stays_at_checkout(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import provenance

    from merlin.common import paths

    checkout = tmp_path / "physical-checkout"
    redirected = tmp_path / "external-workspace"
    corpus = checkout / "merlin/contract/capsules"
    profiles = corpus / "profiles"
    profiles.mkdir(parents=True)
    redirected.mkdir()
    monkeypatch.setattr(paths, "checkout_root", lambda: checkout)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(redirected))
    with pytest.raises(ValueError, match="explicit recipe inputs"):
        GC.profile_targets()
    with pytest.raises(ValueError, match="explicit recipe inputs"):
        GC.load_profile("fixture")
    manifest = provenance.update_provenance_manifest([])
    assert manifest == corpus / "MANIFEST.yaml"
    assert manifest.is_file()
    assert not (redirected / "merlin/contract/capsules/MANIFEST.yaml").exists()


def test_explicit_profile_and_manifest_roots_ignore_checkout_defaults(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import provenance

    from merlin.common import paths

    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "fixture.yaml").write_text(yaml.safe_dump({"datapath": {}, "capsules": []}))
    assert GC.profile_targets(profiles_root=tmp_path) == ["fixture"]
    assert GC.load_profile("fixture", profiles_root=tmp_path)["capsules"] == []
    manifest = provenance.update_provenance_manifest([], cap_root=tmp_path)
    assert manifest == tmp_path / "MANIFEST.yaml" and manifest.is_file()
