"""Indexed subtree facts retain lexical boundaries and hardlink aliases."""
from pathlib import Path

from merlin.targetgen.sandbox.guarded_inventory import GuardedInventory, TreeFacts


def test_subtree_projection_preserves_aliases_without_visiting_sibling_paths(tmp_path, monkeypatch):
    root = tmp_path / 'snapshot'
    rows = []
    links = []
    for name in ('a', 'aa', 'z', '\U0001f600'):
        for leaf in ('one', 'nested/two'):
            path = root / name / leaf
            rows.append((path.relative_to(root).parts, path, 'file', 4))
            links.append((path, (1, 7)))  # Distinct names may share one inode.
    rows.append((('a', 'alias'), root / 'a/alias', 'symlink', 3))
    facts = TreeFacts(tuple(reversed(rows)), frozenset({(1, 7)}), tuple(reversed(links)))
    original = Path.is_relative_to
    visits = []

    def observed(path, other):
        visits.append(path)
        return original(path, other)

    monkeypatch.setattr(Path, 'is_relative_to', observed)
    for name in ('a', 'aa', 'z', '\U0001f600', 'absent'):
        selected = root / name
        projected = GuardedInventory._project(facts, root, selected, nofollow=True)
        expected = tuple(sorted((parts[1:], path, kind, size)
                                for parts, path, kind, size in rows if parts[0] == name))
        assert projected.members == expected
        assert set(projected.linked_members) == {(path, inode) for path, inode in links
                                               if path.parts[len(root.parts)] == name}
    # Only the requested roots are checked against the owning parent. None of
    # the unrelated member Paths need an ancestor walk for a projection.
    assert visits == [root / name for name in ('a', 'aa', 'z', '\U0001f600', 'absent')]
    assert GuardedInventory._project(facts, root, root / 'a', nofollow=False) is None
    assert GuardedInventory._project(facts, root, root / 'aa', nofollow=False) is not None

