"""A human-review gate needs the complete locked specification and every current claim pin."""

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def publisher():
    source = repo_root() / "build_tools/scripts/prepare_bf16_review.py"
    spec = importlib.util.spec_from_file_location("bf16_review", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installed(tmp_path: Path, module):
    lock = {}
    for name in [module.SPEC, *module.REVIEW_FILES]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed specification fixture\n")
        lock[name] = module.digest(path.read_bytes())
    statements = {name: "current elaborated statement " + name for name in module.THEOREMS}
    lock.update({"statement:" + name: module.digest(value.encode()) for name, value in statements.items()})
    return lock, statements


def test_review_never_approves_an_uninstalled_or_unpinned_candidate(tmp_path):
    module = publisher()
    assert not module.reviewed(tmp_path, {}, {})
    lock, statements = installed(tmp_path, module)
    assert module.reviewed(tmp_path, lock, statements)
    del lock["statement:" + next(iter(module.THEOREMS))]
    assert not module.reviewed(tmp_path, lock, statements)


@pytest.mark.parametrize("changed", ["specification", "source_reference", "operand_domain", "semantic_laws"])
def test_changing_reviewed_policy_bytes_revokes_approval(tmp_path, changed):
    module = publisher()
    lock, statements = installed(tmp_path, module)
    assert module.reviewed(tmp_path, lock, statements)
    name = {
        "specification": module.SPEC,
        "source_reference": "MerlinVerify/Rules/Bf16F32AccReference.lean",
        "operand_domain": "MerlinVerify/Check/Bf16F32Acc.lean",
        "semantic_laws": "MerlinVerify/Check/ParametricAlpha.lean",
    }[changed]
    (tmp_path / name).write_text("changed policy\n")
    assert not module.reviewed(tmp_path, lock, statements)
