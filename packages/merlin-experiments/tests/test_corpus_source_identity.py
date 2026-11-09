"""Cross-phase corpus ownership remains private, inert and bound to run sources."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from merlin.common.paths import module_source_path
from merlin_experiments.phase1 import source_inputs
from merlin_experiments.spec import SpecError


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


@pytest.mark.parametrize(
    "damage",
    [
        "release",
        "preparation",
        "admission",
        "initializer",
        "added",
        "removed",
        "quantization_admission",
        "quantization_capture_projection",
        "quantization_recipe_records",
        "quantization_recipe_dynamic",
    ],
)
def test_corpus_package_membership_and_bytes_are_bound(tmp_path, monkeypatch, damage):
    package = tmp_path / "corpus"
    shutil.copytree(
        module_source_path("merlin_experiments.corpus").parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    original = source_inputs._source
    numerical_owners = {
        "quantization_admission": "merlin.compiler.quantization.admission",
        "quantization_capture_projection": "merlin.compiler.quantization.torchao.recipes",
        "quantization_recipe_records": "merlin.compiler.quantization.recipe",
        "quantization_recipe_dynamic": "merlin.compiler.quantization.dynamic",
    }
    numerical_source = None
    if damage in numerical_owners:
        numerical_source = tmp_path / f"{damage}.py"
        shutil.copyfile(original(numerical_owners[damage]), numerical_source)

    def copied_source(module):
        if numerical_source is not None and module == numerical_owners[damage]:
            return numerical_source
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
    if numerical_source is not None:
        assert record["inputs"][f"phase1:startup:{damage}"]["path"] == str(numerical_source)
        numerical_source.write_text(numerical_source.read_text() + "\n# changed numerical source\n")
        with pytest.raises(SpecError, match="source identity changed"):
            source_inputs.verify(record, **arguments)
        return
    if damage == "removed":
        (package / "release.py").unlink()
    else:
        name = "__init__" if damage == "initializer" else damage
        member = package / f"{name}.py"
        member.write_text((member.read_text() if member.exists() else "") + "\n# changed corpus source\n")
    with pytest.raises(SpecError, match="source identity changed|missing or foreign source owner"):
        source_inputs.verify(record, **arguments)
