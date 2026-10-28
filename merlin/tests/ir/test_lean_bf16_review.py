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


def mocked_governance(monkeypatch, module, statements):
    """Test pin checks after a mocked admission; this creates no actual approval."""
    admitted = {name: statements[name] for name in module.GATE_ROOTS}
    monkeypatch.setattr(module, "require_admission", lambda root: {"statements": admitted})
    return admitted


def test_review_never_approves_an_uninstalled_or_unpinned_candidate(tmp_path, monkeypatch):
    module = publisher()
    assert not module.reviewed(tmp_path, {}, {})
    lock, statements = installed(tmp_path, module)
    assert not module.reviewed(tmp_path, lock, statements)  # Pins alone never authenticate review.
    mocked_governance(monkeypatch, module, statements)
    assert module.reviewed(tmp_path, lock, statements)
    del lock["statement:" + next(iter(module.THEOREMS))]
    assert not module.reviewed(tmp_path, lock, statements)


@pytest.mark.parametrize("changed", ["specification", "source_reference", "operand_domain", "semantic_laws"])
def test_changing_reviewed_policy_bytes_revokes_approval(tmp_path, monkeypatch, changed):
    module = publisher()
    lock, statements = installed(tmp_path, module)
    mocked_governance(monkeypatch, module, statements)
    assert module.reviewed(tmp_path, lock, statements)
    name = {
        "specification": module.SPEC,
        "source_reference": "MerlinVerify/Rules/Bf16F32AccReference.lean",
        "operand_domain": "MerlinVerify/Check/Bf16F32Acc.lean",
        "semantic_laws": "MerlinVerify/Check/ParametricAlpha.lean",
    }[changed]
    (tmp_path / name).write_text("changed policy\n")
    assert not module.reviewed(tmp_path, lock, statements)


def test_admission_must_bind_exact_overlapping_live_statements(tmp_path, monkeypatch):
    module = publisher()
    lock, statements = installed(tmp_path, module)
    admitted = mocked_governance(monkeypatch, module, statements)
    assert module.reviewed(tmp_path, lock, statements)
    admitted[next(iter(admitted))] = "different elaborated theorem statement"
    assert not module.reviewed(tmp_path, lock, statements)


def test_refused_human_admission_keeps_publication_pending(tmp_path, monkeypatch):
    from merlin.verify.gate_errors import GateRejected

    module = publisher()
    lock, statements = installed(tmp_path, module)

    def pending(root):
        raise GateRejected("human_review_pending")

    monkeypatch.setattr(module, "require_admission", pending)
    assert not module.reviewed(tmp_path, lock, statements)
