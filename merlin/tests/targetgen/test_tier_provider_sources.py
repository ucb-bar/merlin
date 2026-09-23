"""Verdict-cache source identity includes selected provider siblings, not just its backend."""

from types import SimpleNamespace

import pytest

from merlin.common.provenance import source_digest
from merlin.runtime.backends import base
from merlin.targetgen import target_registry, tier_cache


@pytest.fixture
def provider(tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    entry = tmp_path / "backend/__init__.py"
    entry.write_text("# backend\n")
    (tmp_path / "build_support").mkdir()
    helper = tmp_path / "build_support/render.py"
    helper.write_text("VALUE = 1\n")
    monkeypatch.setattr(tier_cache, "_GRADING_MODULES", ())
    monkeypatch.setattr(base, "get_backend", lambda _: SimpleNamespace(__file__=str(entry)))
    monkeypatch.setattr(target_registry, "resolve", lambda _: SimpleNamespace(base=tmp_path))
    return tmp_path, helper


def test_sibling_source_mutation_changes_grading_identity(provider):
    _, helper = provider
    before = tier_cache.grading_path("fixture")
    assert helper in before
    digest = source_digest(before)
    helper.write_text("VALUE = 2\n")
    assert source_digest(tier_cache.grading_path("fixture")) != digest


def test_added_and_removed_sibling_members_change_identity(provider):
    root, _ = provider
    before = tier_cache.grading_path("fixture")
    extra = root / "build_support/new.py"
    extra.write_text("# new dependency\n")
    assert extra in tier_cache.grading_path("fixture")
    extra.unlink()
    assert tier_cache.grading_path("fixture") == before


def test_linked_provider_source_disables_verdict_reuse(provider):
    root, helper = provider
    (root / "alias.py").symlink_to(helper)
    assert tier_cache.grading_path("fixture") is None


def test_missing_backend_file_cannot_inventory_current_directory(provider, monkeypatch):
    monkeypatch.chdir(provider[0])
    monkeypatch.setattr(base, "get_backend", lambda _: SimpleNamespace())
    assert tier_cache.grading_path("fixture") is None
