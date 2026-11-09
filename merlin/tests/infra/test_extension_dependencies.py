"""Optional distributions declare their direct dependencies without pin drift."""

from __future__ import annotations

import tomllib

from merlin.common.paths import repo_root


def _project(relative_path: str) -> dict:
    return tomllib.loads((repo_root() / relative_path).read_text())["project"]


def test_aet_pin_is_identical_for_every_direct_owner():
    core = _project("pyproject.toml")
    pins = []
    for dependencies in (
        core["optional-dependencies"]["telemetry"],
        core["optional-dependencies"]["experiments"],
        core["optional-dependencies"]["analysis"],
    ):
        aet = [dependency for dependency in dependencies if dependency.startswith("aet @ ")]
        assert len(aet) == 1
        pins.append(aet[0])
    assert len(set(pins)) == 1
    revision = pins[0].rsplit("@", 1)[1]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def test_analysis_does_not_require_experiments_or_transitive_aet_url():
    project = _project("pyproject.toml")
    dependencies = project["optional-dependencies"]["analysis"]
    assert "xdsl>=0.68" in dependencies and "jsonschema" in dependencies
    assert not any("telemetry" in dependency for dependency in dependencies)
    assert not any(dependency.startswith("merlin-experiments") for dependency in dependencies)
    assert project["optional-dependencies"]["dse"] == []
    assert project["optional-dependencies"]["mining"] == []
    assert not any((repo_root() / "packages" / owner / "pyproject.toml").exists()
                   for owner in ("merlin-experiments", "merlin-analysis", "merlin-dse", "merlin-mining"))
