"""The sidecar chain must read what the writers write, and a dotted stem is not a target.

Two defects that made the SMT counterexample path dead on arrival, and one that was live and wider.
"""
from __future__ import annotations

import importlib.util
import sys

import yaml

from merlin.common.paths import merlin_dir

_p = merlin_dir() / "contract" / "capsules" / "generate_corpus.py"
if str(_p.parent) not in sys.path:
    sys.path.insert(0, str(_p.parent))
_spec = importlib.util.spec_from_file_location("generate_corpus_under_test", _p)
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)


def _profiles(tmp_path, monkeypatch):
    """Point the generator at a scratch profiles dir. `_merge_shared_perf` reads the shared template
    unconditionally, so it has to exist even when the test is about something else."""
    monkeypatch.setattr(GC, "PROFILES", tmp_path)
    (tmp_path / "_perf.yaml").write_text(yaml.safe_dump({"capsules": [], "sweeps": []}))
    return tmp_path


def test_a_dotted_stem_is_never_returned_as_a_target():
    """⚠️ REGRESSION, and it was LIVE. `Path.stem` strips only the last suffix, so
    `gemmini.synth.yaml` yielded the stem `gemmini.synth`, returned beside `gemmini`. Six real targets
    came back as twelve, and `main()` uses this list as the default when `--target` is absent -- so a
    bare run generated phantom corpora from profile fragments."""
    targets = GC.profile_targets()
    assert targets, "no targets at all means the glob is broken, not that the leak is fixed"
    dotted = [t for t in targets if "." in t]
    assert dotted == [], f"sidecar stems returned as targets: {dotted}"


def test_every_returned_target_has_a_real_profile():
    for t in GC.profile_targets():
        assert (GC.PROFILES / f"{t}.yaml").is_file()


def test_the_smt_sidecar_is_in_the_chain_load_profile_reads(tmp_path, monkeypatch):
    """⚠️ REGRESSION. `counterexamples.profile_path` writes `<target>.smt.yaml`; `load_profile` read
    exactly three filenames and that was not one of them, so every solver-found counterexample went to
    a file nothing opened. The module's own docstring asserted a glob that does not exist."""
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(
        {"datapath": {}, "capsules": [{"name": "A0_base"}]}))
    (tmp_path / "t.smt.yaml").write_text(yaml.safe_dump(
        {"capsules": [{"name": "CX_contraction_i8_16x16x16"}]}))
    names = [c["name"] for c in GC.load_profile("t")["capsules"]]
    assert "CX_contraction_i8_16x16x16" in names, (
        "a counterexample entry written by verify.counterexamples must reach the generator")
    assert "A0_base" in names, "the public profile must still be merged"


def test_the_synth_sidecar_still_merges(tmp_path, monkeypatch):
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"datapath": {}, "capsules": []}))
    (tmp_path / "t.synth.yaml").write_text(yaml.safe_dump({"capsules": [{"name": "SY_cell_i8"}]}))
    assert [c["name"] for c in GC.load_profile("t")["capsules"]] == ["SY_cell_i8"]


def test_an_absent_smt_sidecar_is_not_an_error(tmp_path, monkeypatch):
    _profiles(tmp_path, monkeypatch)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"datapath": {}, "capsules": [{"name": "A0"}]}))
    assert [c["name"] for c in GC.load_profile("t")["capsules"]] == ["A0"]


def test_the_counterexample_writer_and_the_reader_agree_on_the_filename():
    """The two ends of the path, compared directly. They disagreed, and nothing noticed because
    neither side asserted anything about the other."""
    import inspect

    from merlin.verify.counterexamples import profile_path

    written = profile_path("t").name
    assert written == "t.smt.yaml"
    assert ".smt.yaml" in inspect.getsource(GC.load_profile), (
        f"{written} is written by counterexamples.profile_path and not read by load_profile")
