"""Independent lexical-rule comparisons and large explicit-leaf complexity controls."""

import cProfile
import random
from pathlib import Path

import pytest

from merlin.common import path_relations as INDEX


def _original_roots(paths):
    roots = []
    for source in sorted({path.absolute() for path in paths}, key=lambda path: (len(path.parts), str(path))):
        if not any(parent == source or parent in source.parents for parent in roots):
            roots.append(source)
    return tuple(roots)


def _original_overlap(path, private):
    path = path.absolute()
    return any(
        path == root.absolute() or path in root.absolute().parents or root.absolute() in path.parents
        for root in private
    )


def test_deterministic_minimal_roots_and_overlap_match_original_rules():
    randomizer = random.Random(37)
    universe = [
        Path("/").joinpath(*(str(randomizer.randrange(7)) for _ in range(randomizer.randrange(1, 8))))
        for _ in range(240)
    ]
    universe += [Path("/"), Path("/a"), Path("/ab"), Path("/a/b"), Path("/a/../b")]
    for _ in range(80):
        paths = randomizer.choices(universe, k=randomizer.randrange(0, 80))
        assert INDEX.minimal_roots(iter(paths)) == _original_roots(paths)
        index = INDEX.LexicalPathIndex.from_paths(iter(paths))
        for path in randomizer.sample(universe, 20):
            assert index.overlaps(path) == _original_overlap(path, paths)
            assert index.contains(path) == any(
                path == root.absolute() or root.absolute() in path.parents for root in paths
            )


def test_component_boundaries_and_both_overlap_directions():
    index = INDEX.LexicalPathIndex.from_paths([Path("/a/b"), Path("/private/deep/leaf")])
    assert all(index.overlaps(Path(path)) for path in ("/", "/a", "/a/b", "/a/b/c", "/private/deep"))
    assert not index.overlaps(Path("/ab"))
    assert not index.overlaps(Path("/a/bb"))
    assert not index.overlaps(Path("/elsewhere"))
    assert not INDEX.LexicalPathIndex.from_paths(()).overlaps(Path("/"))


def test_aliases_are_lexical_and_helpers_do_not_resolve_or_stat(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    alias = tmp_path / "alias"
    canonical.mkdir()
    alias.symlink_to(canonical, target_is_directory=True)

    def refused(*_args, **_kwargs):
        pytest.fail("path indexing cannot resolve aliases or inspect filesystem authority")

    monkeypatch.setattr(Path, "resolve", refused)
    monkeypatch.setattr(Path, "stat", refused)
    paths = (canonical / "leaf", alias / "leaf")
    assert INDEX.minimal_roots(paths) == _original_roots(paths)
    index = INDEX.LexicalPathIndex.from_paths((canonical / "leaf",))
    assert index.overlaps(canonical) and not index.overlaps(alias)


def _profile_calls(function, paths):
    profiler = cProfile.Profile()
    profiler.runcall(function, paths)
    # Original quadratic scans repeatedly enter a Python generator and ask for
    # source.parents. Counting these actual events avoids noisy wall-time gates.
    return sum(
        stat.callcount
        for stat in profiler.getstats()
        if not isinstance(stat.code, str) and stat.code.co_name == "<genexpr>"
    )


def test_large_explicit_leaf_operation_counts_scale_with_depth_not_pairs():
    def paths(count):
        return tuple(Path(f"/selected/lib/lean/Namespace/file{index:05d}.olean") for index in range(count))

    small, medium, large = paths(128), paths(256), paths(13481)
    assert INDEX.minimal_roots(large) == large
    old_small, old_medium = _profile_calls(_original_roots, small), _profile_calls(_original_roots, medium)
    new_small, new_medium = _profile_calls(INDEX.minimal_roots, small), _profile_calls(INDEX.minimal_roots, medium)
    new_large = _profile_calls(INDEX.minimal_roots, large)
    assert old_medium > 3.5 * old_small
    assert new_medium < 2.1 * new_small
    assert new_large < 10 * len(large)
    assert new_small < old_small / 5
    index = INDEX.LexicalPathIndex.from_paths(large)
    for leaf in large:
        assert index.overlaps(leaf)
        assert index.overlaps(leaf / "nested")
    assert index.overlaps(Path("/selected/lib/lean"))
    assert not index.contains(Path("/selected/lib/lean"))
    assert not index.overlaps(Path("/disjoint/lib/lean"))


def test_actual_large_explicit_leaf_snapshot_queries_each_declaration_once(tmp_path, monkeypatch):
    import json
    import time
    from importlib import import_module

    answer_surfaces = import_module("merlin.targetgen.sandbox.answer_surfaces")
    from merlin.targetgen.sandbox import bwrap as BW

    count = 13481
    repo = tmp_path / "repo"
    selected = repo / "selected/lib/lean"
    selected.mkdir(parents=True)
    leaves = [selected / f"file{number:05d}.olean" for number in range(count)]
    for leaf in leaves:
        leaf.write_bytes(b"bytes")
        leaf.chmod(0o600)
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    monkeypatch.setattr(answer_surfaces, "_support_package_dirs", lambda: [])

    class CountingIndex(INDEX.LexicalPathIndex):
        overlap_queries = 0
        declaration_queries = 0

        def contains(self, path):
            type(self).declaration_queries += 1
            return super().contains(path)

        def overlaps(self, path):
            type(self).overlap_queries += 1
            return super().overlaps(path)

    monkeypatch.setattr(BW, "LexicalPathIndex", CountingIndex)
    bundle = {"allowed": [], "host_inputs": [{"path": str(leaf)} for leaf in leaves]}
    ws = tmp_path / "run/workspace"
    start = time.process_time()
    marker = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    materialize_cpu = time.process_time() - start
    assert marker["n_files"] == count and marker["n_bytes"] == count * len(b"bytes")
    assert marker["copied_roots"] == [str(leaf) for leaf in leaves]
    assert CountingIndex.overlap_queries == count
    CountingIndex.declaration_queries = 0
    start = time.process_time()
    frozen = BW.snapshot_input_paths(ws, bundle, leaves, repo=repo)
    resolve_cpu = time.process_time() - start
    assert CountingIndex.declaration_queries == count
    assert frozen == [BW.bundle_snapshot_root(ws) / "repo" / leaf.relative_to(repo) for leaf in leaves]
    assert all(path.stat().st_mode & 0o077 == 0 and not path.stat().st_mode & 0o222 for path in frozen)
    assert not any(path.is_file() for path in (tmp_path / "cas").rglob("*"))
    with pytest.raises(RuntimeError, match="not declared"):
        BW.snapshot_input_paths(ws, bundle, [selected], repo=repo)
    (tmp_path / "cpu-counts.json").write_text(
        json.dumps(
            {
                "files": count,
                "private_overlap_queries": CountingIndex.overlap_queries,
                "materialize_cpu_seconds": materialize_cpu,
                "resolve_cpu_seconds": resolve_cpu,
                "scope": "single-process CPU/file-copy control; no sandbox process, compiler or simulator",
            },
            sort_keys=True,
        )
    )
    BW.remove_bundle_snapshot(ws)
