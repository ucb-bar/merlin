"""A temp dir the suite makes for itself must be reclaimable.

Several buckets obtain scratch space from a bare ``tempfile.mkdtemp()`` inside a module-level
helper, because the helper is shared by parametrized cases and has nowhere to put a fixture. Nothing
removed those directories -- not the test, not pytest, not the OS -- and each holds a workload
bundle, the MLIR at every stage, an object file and a disassembly. The conftest fixture routes them
into pytest's managed base temp dir instead; these tests hold it to that.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_a_bare_mkdtemp_lands_under_the_managed_base_temp_dir(tmp_path_factory):
    """The whole point: a helper that asks `tempfile` for space gets reclaimable space."""
    base = tmp_path_factory.getbasetemp().resolve()
    made = Path(tempfile.mkdtemp(prefix="reclaimable_")).resolve()
    assert base in made.parents, f"{made} is outside pytest's managed root {base}"


def test_a_spawned_tool_writes_its_intermediates_there_too(tmp_path_factory):
    """A compiler or simulator the test spawns must not fall back to the root filesystem."""
    base = tmp_path_factory.getbasetemp().resolve()
    got = subprocess.run(
        [sys.executable, "-c", "import tempfile,pathlib;print(pathlib.Path(tempfile.mkdtemp()).resolve())"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert base in Path(got.stdout.strip()).parents


def test_the_managed_root_is_not_on_the_small_filesystem(tmp_path_factory):
    """pytest's default root is under /tmp, which here shares the volume that whole-model builds
    have filled. The conftest moves it; an operator override is respected and not second-guessed."""
    base = tmp_path_factory.getbasetemp().resolve()
    if os.environ.get("MERLIN_TEST_TEMPROOT"):
        return
    root_dev = Path("/").stat().st_dev
    if Path("/scratch").is_dir() and Path("/scratch").stat().st_dev != root_dev:
        assert base.stat().st_dev != root_dev, f"base temp {base} sits on the root filesystem"


def test_retention_is_bounded_by_configuration():
    """Keeping three full runs of every test's tree is what made this unbounded; the suite declares
    a bound so the reclaim does not depend on anyone remembering to do it."""
    import tomllib  # noqa: PLC0415

    from merlin.common.paths import repo_root  # noqa: PLC0415

    config = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    options = config["tool"]["pytest"]["ini_options"]
    assert options["tmp_path_retention_count"] == 1
    assert options["tmp_path_retention_policy"] == "failed"
