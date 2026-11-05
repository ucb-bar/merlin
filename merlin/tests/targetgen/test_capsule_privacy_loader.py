"""Capsule privacy reads metadata with libyaml and must build exactly PyYAML's documents.

The answer masks classify every capsule of a corpus per sandbox policy, so ``capsule_answer_members``
parses ``capsule.yaml`` with :func:`merlin.common.yaml.fast_safe_load`. These tests hold that change to
the documents and roles the pure-Python ``yaml.safe_load`` produced, on the repository's real capsule
corpora rather than on synthetic YAML.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from merlin.common import yaml as merlin_yaml
from merlin.common.access import contract_resource_roots
from merlin.common.paths import repo_root
from merlin.targetgen import capsule_privacy


def _corpus_metadata() -> list[Path]:
    found: set[Path] = set()
    for root in contract_resource_roots(repo_root(), "capsules"):
        if root.is_dir():
            found.update(path for path in root.rglob("capsule.yaml") if path.is_file())
    return sorted(found)


def _identical(left, right) -> bool:
    """Equality that also distinguishes bool/int/float and preserves mapping key order."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(_identical(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_identical(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


CORPUS = _corpus_metadata()


def test_the_real_capsule_corpus_is_present():
    # The equivalence below is only evidence if it ran over the corpus the masks actually walk.
    assert len(CORPUS) >= 100


@pytest.mark.skipif(getattr(yaml, "CSafeLoader", None) is None, reason="PyYAML was built without libyaml")
def test_libyaml_builds_the_same_document_for_every_real_capsule():
    differing = []
    for path in CORPUS:
        raw = path.read_bytes()
        if not _identical(yaml.load(raw, Loader=yaml.CSafeLoader), yaml.safe_load(raw)):  # noqa: S506
            differing.append(path)
    assert differing == []


def test_fast_load_returns_exactly_the_safe_load_document_for_every_real_capsule():
    differing = [
        path
        for path in CORPUS
        if not _identical(merlin_yaml.fast_safe_load(path.read_bytes()), yaml.safe_load(path.read_bytes()))
    ]
    assert differing == []


def test_capsule_roles_are_unchanged_for_every_real_capsule(monkeypatch):
    fast = {path: capsule_privacy.capsule_answer_members(path.parent) for path in CORPUS}
    monkeypatch.setattr(capsule_privacy, "fast_safe_load", yaml.safe_load)
    for path in CORPUS:
        expected = capsule_privacy.capsule_answer_members(path.parent)
        assert list(fast[path].items()) == list(expected.items()), path


def test_without_libyaml_the_load_is_plain_safe_load(monkeypatch):
    monkeypatch.setattr(merlin_yaml, "_FAST_LOADER", None)
    calls = []

    def safe_load(data):
        calls.append(data)
        return {"selected": True}

    monkeypatch.setattr(merlin_yaml.yaml, "safe_load", safe_load)
    assert merlin_yaml.fast_safe_load(b"name: x\n") == {"selected": True}
    assert calls == [b"name: x\n"]


@pytest.mark.parametrize("text", [b"name: [unclosed\n", b"a: 1\n  b: 2\n", b'key: "bad \\q escape"\n'])
def test_a_refused_document_raises_exactly_the_safe_load_error(text):
    with pytest.raises(yaml.YAMLError) as expected:
        yaml.safe_load(text)
    with pytest.raises(yaml.YAMLError) as actual:
        merlin_yaml.fast_safe_load(text)
    assert type(actual.value) is type(expected.value) and str(actual.value) == str(expected.value)


def test_a_non_mapping_capsule_still_refuses(tmp_path):
    (tmp_path / "capsule.yaml").write_text("- not a mapping\n")
    with pytest.raises(ValueError, match="requires a mapping"):
        capsule_privacy.capsule_answer_members(tmp_path)
