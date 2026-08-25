"""A grade run must pick up its target's tooling paths, or it reports the environment as a verdict.

The harness sources ``targets/<target>/experiment.env`` for every arm before a run. A caller that
grades a package WITHOUT going through the harness did not, and the difference is not cosmetic: one
target's certifying tier is a program-driven Verilator sim registered through ``MERLIN_EXT_<TARGET>_VSIM``,
and its own profile spells out what happens when the variable is absent -- no adapter exists, so
``not_run_is_not_pass`` reports every capsule ``incomplete``, never a pass. Fail-closed is correct; a
whole suite reported incomplete because a path was not sourced is a tooling artifact wearing a verdict's
clothes.
"""
from __future__ import annotations

import os

from merlin.targetgen.corpora import source_experiment_env


def _target_dir(tmp_path, body: str):
    d = tmp_path / "targets" / "faux"
    d.mkdir(parents=True)
    (d / "target_experiment.yaml").write_text("target: faux\n", encoding="utf-8")
    (d / "experiment.env").write_text(body, encoding="utf-8")
    return d / "target_experiment.yaml"


def test_the_tooling_paths_are_loaded(tmp_path, monkeypatch):
    desc = _target_dir(tmp_path, "MERLIN_FAUX_VSIM=/opt/vsim\nMERLIN_FAUX_MODEL=/opt/model\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    monkeypatch.delenv("MERLIN_FAUX_VSIM", raising=False)
    monkeypatch.delenv("MERLIN_FAUX_MODEL", raising=False)
    assert sorted(source_experiment_env("faux")) == ["MERLIN_FAUX_MODEL", "MERLIN_FAUX_VSIM"]
    assert os.environ["MERLIN_FAUX_VSIM"] == "/opt/vsim"


def test_an_exported_variable_always_wins(tmp_path, monkeypatch):
    """Never override what the caller exported -- that is how a run silently grades the wrong checkout."""
    desc = _target_dir(tmp_path, "MERLIN_FAUX_VSIM=/from/file\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    monkeypatch.setenv("MERLIN_FAUX_VSIM", "/from/export")
    assert source_experiment_env("faux") == []
    assert os.environ["MERLIN_FAUX_VSIM"] == "/from/export"


def test_comments_and_quotes_are_parsed_structurally(tmp_path, monkeypatch):
    desc = _target_dir(tmp_path, '# a comment\n\nMERLIN_FAUX_A="/quoted/path"  \nnot_a_pair\n')
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    monkeypatch.delenv("MERLIN_FAUX_A", raising=False)
    assert source_experiment_env("faux") == ["MERLIN_FAUX_A"]
    assert os.environ["MERLIN_FAUX_A"] == "/quoted/path"


def test_a_target_without_one_is_not_an_error(tmp_path, monkeypatch):
    d = tmp_path / "targets" / "bare"
    d.mkdir(parents=True)
    (d / "target_experiment.yaml").write_text("target: bare\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(d / "target_experiment.yaml"))
    assert source_experiment_env("bare") == []


def test_the_grader_sources_before_resolving_adapters():
    """Ordering is the whole point: sourcing after adapter resolution would be too late."""
    import inspect

    from merlin.targetgen import capsule_grade as CG
    src = inspect.getsource(CG.grade)
    assert src.index("source_experiment_env") < src.index("load_package"), \
        "tooling paths must be present before the package/adapters are resolved"
