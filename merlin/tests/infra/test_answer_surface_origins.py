"""Answer labels retain strict lexical ancestry without filesystem authority."""

from pathlib import Path

import pytest

from merlin.targetgen.sandbox.answer_surfaces import _golden_origin


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("examples/expected.json", "example"),
        ("examples/nested/deep/expected.json", "example"),
        ("examples", "golden"),
        ("examples-other/expected.json", "golden"),
        ("outside/expected.json", "golden"),
        ("other/fixtures/expected.json", "example"),
    ],
)
def test_strict_lexical_example_ancestry(tmp_path, relative, expected):
    roots = frozenset(tmp_path / p for p in ("examples", "examples/nested", "other/fixtures"))
    assert _golden_origin(tmp_path / relative, roots) == expected


def test_origin_does_not_resolve_aliases_and_observes_changed_root_roster(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(examples, target_is_directory=True)
    path = alias / "expected.json"
    assert _golden_origin(path, frozenset({examples})) == "golden"
    assert _golden_origin(path, frozenset({alias})) == "example"
    assert _golden_origin(path, frozenset()) == "golden"


def test_multiple_roots_do_not_rebuild_the_same_answer_ancestors(monkeypatch):
    original = Path.parents
    reads = []

    def parents(path):
        reads.append(path)
        return original.__get__(path, type(path))

    monkeypatch.setattr(Path, "parents", property(parents))
    path = Path("/owner/very/deep/ordinary/expected.json")
    roots = frozenset(Path("/owner") / str(index) for index in range(100))
    assert _golden_origin(path, roots) == "golden"
    assert reads == [path]
