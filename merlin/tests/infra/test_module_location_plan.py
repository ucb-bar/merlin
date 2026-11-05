"""Declaration projection is reusable; reachable withheld files are always live."""

from types import SimpleNamespace

import pytest

from merlin.common import access as A


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    return tmp_path, A.module_access("merlin.private_leaf", "grader")


def write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("withheld\n")
    return path


def test_default_batch_keeps_the_original_lookup(selected, monkeypatch):
    root, item = selected
    seen = []
    monkeypatch.setattr(A, "module_locations", lambda *args: seen.append(args) or ())
    with A.module_location_batch(root) as locate:
        assert locate(item) == ()
    assert seen == [(root, item)]


def test_batch_discovers_roots_twice_for_many_declarations(selected, monkeypatch):
    root, item = selected
    calls = []
    original = A.runtime_package_roots

    def roots(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(A, "runtime_package_roots", roots)
    plan = A.ModuleLocationPlan()
    for _ in range(2):
        with plan.batch(root) as locate:
            for _ in range(100):
                assert locate(item) == ()
    assert calls == [root] * 4


def test_cached_paths_notice_new_source_and_compiled_copy(selected):
    root, item = selected
    plan = A.ModuleLocationPlan()
    with plan.batch(root) as locate:
        assert locate(item) == ()
        source = write(root / "src/merlin/private_leaf.py")
        assert locate(item) == (source,)
    compiled = write(root / "src/merlin/__pycache__/private_leaf.cpython-312.pyc")
    with plan.batch(root) as locate:
        assert locate(item) == A.module_locations(root, item) == (source, compiled)
    source.unlink()
    with plan.batch(root) as locate:
        assert locate(item) == (compiled,)


def test_new_runtime_root_and_declaration_are_not_reused(selected):
    root, item = selected
    plan = A.ModuleLocationPlan()
    with plan.batch(root) as locate:
        assert locate(item) == ()
    installed = root / "installed"
    source = write(installed / "merlin/private_leaf.py")
    other = write(installed / "merlin/other.py")
    A.sys.path.append(str(installed))
    with plan.batch(root) as locate:
        assert locate(item) == A.module_locations(root, item) == (source,)
        assert locate(A.module_access("merlin.other", "oracle")) == (other,)


def test_changed_runtime_root_during_batch_refuses(selected):
    root, item = selected
    plan = A.ModuleLocationPlan()
    with pytest.raises(RuntimeError, match="roots changed"):
        with plan.batch(root) as locate:
            assert locate(item) == ()
            installed = root / "new-install"
            write(installed / "merlin/private_leaf.py")
            A.sys.path.append(str(installed))


def test_top_level_symlink_resolution_is_fresh(selected):
    root, _item = selected
    item = A.module_access("old_oracle", "oracle")
    search = root / "installed"
    search.mkdir()
    A.sys.path.append(str(search))
    first, second = write(root / "first.py"), write(root / "second.py")
    alias = search / "old_oracle"
    alias.symlink_to(first)
    plan = A.ModuleLocationPlan()
    with plan.batch(root) as locate:
        assert locate(item) == A.module_locations(root, item)
        assert first in locate(item)
    alias.unlink()
    alias.symlink_to(second)
    with plan.batch(root) as locate:
        assert locate(item) == A.module_locations(root, item)
        assert second in locate(item) and first not in locate(item)


def test_closed_or_foreign_process_plan_refuses_before_lock(selected, monkeypatch):
    root, _item = selected
    plan = A.ModuleLocationPlan()
    with monkeypatch.context() as patch:
        patch.setattr(A.os, "getpid", lambda: -1)
        with pytest.raises(RuntimeError, match="process lifetime"):
            with plan.batch(root):
                pytest.fail("foreign plan admitted")
        plan.close()  # Must not acquire an inherited lock.
    plan.close()
    with pytest.raises(RuntimeError, match="process lifetime"):
        with plan.batch(root):
            pytest.fail("closed plan admitted")
