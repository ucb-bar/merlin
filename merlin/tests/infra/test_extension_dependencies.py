"""Optional distributions declare their direct dependencies without pin drift."""

from __future__ import annotations

import tomllib

from merlin.common.paths import repo_root


def _project(relative_path: str) -> dict:
    return tomllib.loads((repo_root() / relative_path).read_text())["project"]


def test_aet_pin_is_identical_for_every_direct_owner():
    core = _project("pyproject.toml")
    experiments = _project("packages/merlin-experiments/pyproject.toml")
    analysis = _project("packages/merlin-analysis/pyproject.toml")
    pins = []
    for dependencies in (
        core["optional-dependencies"]["telemetry"],
        experiments["dependencies"],
        analysis["dependencies"],
    ):
        aet = [dependency for dependency in dependencies if dependency.startswith("aet @ ")]
        assert len(aet) == 1
        pins.append(aet[0])
    assert len(set(pins)) == 1
    revision = pins[0].rsplit("@", 1)[1]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def test_analysis_does_not_require_experiments_or_transitive_aet_url():
    analysis = _project("packages/merlin-analysis/pyproject.toml")
    dependencies = analysis["dependencies"]
    assert "merlin[xdsl,targetgen]" in dependencies
    assert not any("telemetry" in dependency for dependency in dependencies)
    assert not any(dependency.startswith("merlin-experiments") for dependency in dependencies)
    assert analysis["optional-dependencies"]["experiments"] == ["merlin-experiments==0.1.0"]
