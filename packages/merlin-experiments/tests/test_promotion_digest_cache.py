"""Submission digests are reused only while the tree is unchanged."""

import os
import time

from merlin_experiments.phase1.feedback import promotion as P
from merlin_experiments.phase1.feedback import tree_value_cache as C


def _workspace(tmp_path):
    sub = tmp_path / "ws" / "submission"
    (sub / "compiler").mkdir(parents=True)
    (sub / "compiler" / "a.py").write_text("a = 1\n")
    (sub / "manifest.yaml").write_text("name: x\n")
    return tmp_path / "ws", sub


def test_a_recently_changed_tree_is_never_cached(tmp_path, monkeypatch):
    ws, sub = _workspace(tmp_path)
    C.CACHE.clear()
    P.submission_digests(ws)
    assert C.CACHE == {}  # ctimes are seconds old at most: inside the racy window


def test_unchanged_tree_reuses_and_any_change_recomputes(tmp_path, monkeypatch):
    ws, sub = _workspace(tmp_path)
    C.CACHE.clear()
    monkeypatch.setattr(C, "SETTLED_NS", -(10**12))  # ctime cannot be aged; trust it as settled
    first = P.submission_digests(ws)
    calls = []
    real = P._submission_digests
    monkeypatch.setattr(P, "_submission_digests", lambda w: calls.append(w) or real(w))
    assert P.submission_digests(ws) == first and calls == []
    # Same size, same mtime: the ctime still moves, so the edit is seen.
    path = sub / "compiler" / "a.py"
    stat = path.stat()
    time.sleep(0.05)  # past one timestamp tick: the production settle window that guards this is disabled above
    path.write_text("a = 2\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = P.submission_digests(ws)
    assert calls and changed != first and changed == real(ws)
    calls.clear()
    (sub / "compiler" / "b.py").write_text("b = 1\n")
    assert P.submission_digests(ws) == real(ws) and calls
    calls.clear()
    (sub / "compiler" / "b.py").unlink()
    assert P.submission_digests(ws) == real(ws) and calls


def test_import_records_resolve_exactly_like_the_ast_walk(tmp_path):
    import ast

    from merlin_experiments.phase1.feedback import promotion_sources as S

    source = (
        "import os, a.b as c\n"
        "from . import x\n"
        "from .. import y\n"
        "from ..pkg.mod import z, w\n"
        "from top.mod import q\n"
        "def f():\n"
        "    import inner\n"
    )
    tree = ast.parse(source)
    rel_parts = ("root", "pkg", "sub", "file.py")
    assert S._resolved_imports(S._import_records(tree), rel_parts) == S._py_imports(tree, rel_parts)
    expected = {"os", "a.b", "root.pkg.sub", "root.pkg.sub.x", "root.pkg", "root.pkg.y",
                "root.pkg.pkg.mod", "root.pkg.pkg.mod.z", "root.pkg.pkg.mod.w", "top.mod", "top.mod.q", "inner"}
    assert S._py_imports(tree, rel_parts) == expected
    f = tmp_path / "m.py"
    f.write_text(source + "x = 'data.bin'\n__import__('dyn')\n")
    records, lits, dyn = S._scan(f)
    assert records == S._import_records(ast.parse(f.read_text())) and "data.bin" in lits and dyn is True
