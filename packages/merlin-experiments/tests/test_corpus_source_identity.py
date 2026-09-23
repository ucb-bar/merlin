"""Cross-phase corpus ownership remains private, inert and bound to run sources."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest
from merlin_experiments.phase1 import source_inputs
from merlin_experiments.spec import SpecError

from merlin.common.paths import module_source_path


def test_corpus_package_import_does_not_load_host_workflows(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import merlin_experiments.corpus; "
            "assert not any(name.startswith('merlin_experiments.corpus.') for name in sys.modules); "
            "assert 'merlin.targetgen.capsule_runner' not in sys.modules",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("damage", ["release", "preparation", "admission", "initializer", "added", "removed"])
def test_corpus_package_membership_and_bytes_are_bound(tmp_path, monkeypatch, damage):
    package = tmp_path / "corpus"
    shutil.copytree(
        module_source_path("merlin_experiments.corpus").parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    original = source_inputs._source

    def copied_source(module):
        if module == "merlin_experiments.corpus":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.corpus."):
            return package / (module.rsplit(".", 1)[-1] + ".py")
        return original(module)

    monkeypatch.setattr(source_inputs, "_source", copied_source)
    arguments = {"repo": tmp_path, "entrypoint": tmp_path / "transport.py"}
    record = source_inputs.record(**arguments)
    for name in ("__init__", "release", "preparation", "admission"):
        assert record["inputs"][f"phase1:startup:corpus:{name}.py"]["path"] == str(package / f"{name}.py")
    source_inputs.verify(record, **arguments)
    if damage == "removed":
        (package / "release.py").unlink()
    else:
        name = "__init__" if damage == "initializer" else damage
        member = package / f"{name}.py"
        member.write_text((member.read_text() if member.exists() else "") + "\n# changed corpus source\n")
    with pytest.raises(SpecError, match="source identity changed|missing or foreign source owner"):
        source_inputs.verify(record, **arguments)
