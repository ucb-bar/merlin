"""The toolchain verification memo may only avoid repeating checks; it never relaxes them."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from merlin.targetgen import common_transforms as CT
from merlin.verify import lean_contract as LC
from merlin.verify.lean_contract import LeanToolchain, digest


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    # Files made by a test are newer than the racy window; the window itself is exercised separately.
    monkeypatch.setattr(LC, "_RACY_NS", 0)
    monkeypatch.setattr(LC, "_MEMO", {})
    monkeypatch.setattr(LC, "_FILE_CACHE", {})
    monkeypatch.setattr(LC, "MEMO_STATS", {"full": 0, "memo": 0})
    monkeypatch.delenv(LC.MEMO_DIR_ENV, raising=False)


def _make(tmp_path, *, memo_dir=None, products=6):
    root = tmp_path / "tc"
    lib = root / "lib" / "lean"
    (lib / "Mathlib" / "Sub").mkdir(parents=True)
    (root / "bin").mkdir()
    executable = root / "bin" / "lean"
    executable.write_bytes(b"#!lean\n")
    executable.chmod(0o755)
    rows = []
    for index in range(products):
        directory = lib / "Mathlib" / ("Sub" if index % 2 else "")
        product = directory / f"P{index}.olean"
        product.write_bytes(f"product {index}".encode())
        rows.append((product, digest(product.read_bytes())))
    work = tmp_path / "work"
    work.mkdir()
    return (
        LeanToolchain(executable, digest(executable.read_bytes()), (lib,), tuple(rows), work, memo_dir=memo_dir),
        rows,
        lib,
    )


def _counting(monkeypatch):
    calls = []
    real = LC.file_digest

    def counted(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(LC, "file_digest", counted)
    return calls


def test_verified_toolchain_is_not_rehashed_while_stamps_are_unchanged(tmp_path, monkeypatch):
    toolchain, rows, _ = _make(tmp_path)
    calls = _counting(monkeypatch)
    toolchain.verify()
    first = len(calls)
    assert first == len(rows) + 1 and LC.MEMO_STATS == {"full": 1, "memo": 0}
    LC._FILE_CACHE.clear()  # prove the stamp memo, not the per-file digest cache, avoids the rehash
    toolchain.verify()
    toolchain.verify(deadline=1e18)
    assert len(calls) == first and LC.MEMO_STATS == {"full": 1, "memo": 2}


def test_recent_stamps_are_not_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr(LC, "_RACY_NS", 3_600 * 10**9)
    toolchain, _, _ = _make(tmp_path)
    toolchain.verify()
    toolchain.verify()
    assert LC.MEMO_STATS == {"full": 2, "memo": 0}


def _warm(tmp_path):
    toolchain, rows, lib = _make(tmp_path)
    toolchain.verify()
    toolchain.verify()
    assert LC.MEMO_STATS["memo"] == 1
    return toolchain, rows, lib


def test_same_size_edit_with_restored_mtime_still_fails(tmp_path):
    toolchain, rows, _ = _warm(tmp_path)
    path = rows[2][0]
    before = path.stat()
    path.write_bytes(b"X" * len(path.read_bytes()))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="dependency bytes changed"):
        toolchain.verify()


def test_replaced_executable_still_fails(tmp_path):
    toolchain, _, _ = _warm(tmp_path)
    toolchain.executable.write_bytes(b"#!evil\n")
    with pytest.raises(ValueError, match="executable bytes changed"):
        toolchain.verify()


def test_removed_listed_file_still_fails(tmp_path):
    toolchain, rows, _ = _warm(tmp_path)
    rows[1][0].unlink()
    with pytest.raises(ValueError, match="dependency bytes changed"):
        toolchain.verify()


@pytest.mark.parametrize("where", ["root", "existing", "new_directory"])
def test_unlisted_imported_product_still_fails(tmp_path, where):
    toolchain, _, lib = _warm(tmp_path)
    if where == "new_directory":
        (lib / "Fresh").mkdir()
        target = lib / "Fresh" / "Extra.olean"
    else:
        target = (lib if where == "root" else lib / "Mathlib" / "Sub") / "Extra.olean"
    target.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="product membership differs"):
        toolchain.verify()


def test_removed_unlisted_noise_does_not_matter_but_failure_is_not_sticky(tmp_path):
    toolchain, rows, lib = _warm(tmp_path)
    extra = lib / "Extra.olean"
    extra.write_bytes(b"unlisted")
    with pytest.raises(ValueError):
        toolchain.verify()
    extra.unlink()
    toolchain.verify()  # the failed attempt published nothing; the clean tree verifies again


def test_symlinked_file_is_never_memoized(tmp_path):
    toolchain, rows, _ = _make(tmp_path)
    path = rows[0][0]
    real = tmp_path / "real.olean"
    real.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(real)
    toolchain.verify()
    toolchain.verify()
    assert LC.MEMO_STATS == {"full": 2, "memo": 0}


def test_swapped_ancestor_for_a_link_is_never_a_hit(tmp_path):
    toolchain, rows, lib = _warm(tmp_path)
    moved = tmp_path / "moved"
    (lib / "Mathlib" / "Sub").rename(moved)
    (lib / "Mathlib" / "Sub").symlink_to(moved)
    with pytest.raises((ValueError, FileNotFoundError)):
        toolchain.verify()


def test_memo_is_a_function_of_the_selection(tmp_path):
    toolchain, rows, _ = _warm(tmp_path)
    changed = LeanToolchain(
        toolchain.executable,
        toolchain.executable_sha256,
        toolchain.library_roots,
        ((rows[0][0], "0" * 64), *rows[1:]),
        toolchain.work_root,
    )
    with pytest.raises(ValueError, match="dependency bytes changed"):
        changed.verify()


def test_work_root_is_still_checked_on_a_hit(tmp_path):
    toolchain, _, _ = _warm(tmp_path)
    gone = LeanToolchain(**{**toolchain.__dict__, "work_root": tmp_path / "missing"})
    with pytest.raises(FileNotFoundError, match="work root"):
        gone.verify()


# -- cross process / concurrency -----------------------------------------------------------------

CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from merlin.verify import lean_contract as LC
    LC._RACY_NS = 0
    root = Path(sys.argv[1]); memo = Path(sys.argv[2])
    rows = tuple((Path(p), h) for p, h in json.loads(sys.argv[3]))
    lib = root / "lib" / "lean"
    exe = root / "bin" / "lean"
    tc = LC.LeanToolchain(exe, LC.digest(exe.read_bytes()), (lib,), rows, Path(sys.argv[4]), memo_dir=memo)
    try:
        tc.verify()
        print(json.dumps(["ok", LC.MEMO_STATS]))
    except Exception as exc:
        print(json.dumps([type(exc).__name__, LC.MEMO_STATS]))
    """
)


def _child(tmp_path, rows, memo, env=None):
    args = [
        sys.executable,
        "-c",
        CHILD,
        str(tmp_path / "tc"),
        str(memo),
        json.dumps([(str(p), h) for p, h in rows]),
        str(tmp_path / "work"),
    ]
    return subprocess.Popen(args, stdout=subprocess.PIPE, text=True, env={**os.environ, **(env or {})})


def _result(process):
    out, _ = process.communicate(timeout=120)
    return json.loads(out.strip().splitlines()[-1])


def test_second_process_reuses_the_sidecar_and_tamper_still_fails(tmp_path):
    memo = tmp_path / "memo"
    toolchain, rows, lib = _make(tmp_path)
    assert _result(_child(tmp_path, rows, memo)) == ["ok", {"full": 1, "memo": 0}]
    assert _result(_child(tmp_path, rows, memo)) == ["ok", {"full": 0, "memo": 1}]
    victim = rows[3][0]
    stamp = victim.stat()
    victim.write_bytes(b"Y" * len(victim.read_bytes()))
    os.utime(victim, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert _result(_child(tmp_path, rows, memo))[0] == "ValueError"
    victim.write_bytes(b"product 3")
    (lib / "Mathlib" / "Late.olean").write_bytes(b"late")
    assert _result(_child(tmp_path, rows, memo))[0] == "ValueError"


@pytest.mark.parametrize("damage", ["garbage", "forged_stamp", "forged_dirs", "foreign_mode"])
def test_corrupt_or_forged_sidecar_falls_back_to_the_full_check(tmp_path, damage):
    memo = tmp_path / "memo"
    toolchain, rows, lib = _make(tmp_path)
    assert _result(_child(tmp_path, rows, memo))[1]["full"] == 1
    [record] = [p for p in memo.iterdir() if p.suffix == ".json"]
    document = json.loads(record.read_text())
    if damage == "garbage":
        record.write_text("{not json")
    elif damage == "forged_stamp":
        record.write_text(json.dumps({**document, "stamp": "0" * 64}))
    elif damage == "forged_dirs":
        record.write_text(json.dumps({**document, "tree_dirs": []}))
    else:
        record.chmod(0o666)
    (lib / "Mathlib" / "Late.olean").write_bytes(b"late")
    assert _result(_child(tmp_path, rows, memo))[0] == "ValueError"


def test_threads_verify_once(tmp_path, monkeypatch):
    toolchain, rows, _ = _make(tmp_path)
    calls = _counting(monkeypatch)
    errors = []
    barrier = threading.Barrier(12)

    def work():
        barrier.wait()
        try:
            toolchain.verify()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(calls) == len(rows) + 1 and LC.MEMO_STATS == {"full": 1, "memo": 11}


def test_processes_verify_once_per_run(tmp_path):
    memo = tmp_path / "memo"
    toolchain, rows, _ = _make(tmp_path, products=40)
    children = [_child(tmp_path, rows, memo) for _ in range(6)]
    results = [_result(child) for child in children]
    assert all(status == "ok" for status, _ in results)
    assert sum(stats["full"] for _, stats in results) == 1
    assert sum(stats["memo"] for _, stats in results) == 5


# -- the service-level resolution memo -------------------------------------------------------------


def _service(tmp_path, monkeypatch):
    toolchain, rows, lib = _make(tmp_path)
    manifest = tmp_path / "manifest" / "lean-toolchain.json"
    manifest.parent.mkdir()
    document = {
        "schema": "merlin.lean_toolchain.v1",
        "executable": {"path": str(toolchain.executable), "sha256": toolchain.executable_sha256},
        "library_roots": [str(lib)],
        "files": [{"path": str(path), "sha256": sha} for path, sha in rows],
    }
    manifest.write_text(json.dumps(document))
    service = object.__new__(CT.CommonTransformService)
    service.workspace = tmp_path / "ws" / "workspace"
    service.workspace.mkdir(parents=True)
    service.repo = tmp_path
    service.snapshot_digest, service.snapshot_marker_digest = "d" * 64, "m" * 64
    service.profile = {
        "lean_toolchain": {"selection": "explicit_manifest", "manifest_sha256": digest(manifest.read_bytes())}
    }
    service.bundle = {
        "allowed": [{"path": str(manifest), "note": "selected formal verification toolchain manifest"}],
        "host_inputs": [],
    }
    seen = []

    def snapshot_input_paths(_workspace, _bundle, sources, *, repo):
        seen.append(len(sources))
        return [Path(p) for p in sources]  # the "frozen" copy is the original in this fixture

    monkeypatch.setattr(CT.BW, "snapshot_input_paths", snapshot_input_paths)
    return service, seen, rows, lib


def test_service_resolves_the_frozen_closure_once_and_matches_a_fresh_service(tmp_path, monkeypatch):
    service, seen, rows, _ = _service(tmp_path, monkeypatch)
    work = tmp_path / "work"
    first = service._toolchain(work)
    assert seen == [1, len(rows) + 1]
    second = service._toolchain(work)
    assert seen == [1, len(rows) + 1, 1]  # only the manifest is re-snapshotted
    assert second == first
    fresh = object.__new__(CT.CommonTransformService)
    fresh.__dict__.update({k: v for k, v in service.__dict__.items() if k != "_resolved"})
    assert fresh._toolchain(work) == second
    assert second.memo_dir == tmp_path / "ws" / "toolchain_verify_memo"


def test_service_tamper_after_resolution_still_fails_and_reresolves(tmp_path, monkeypatch):
    service, seen, rows, lib = _service(tmp_path, monkeypatch)
    work = tmp_path / "work"
    service._toolchain(work)
    service._toolchain(work)
    # In place edit, mtime restored.
    path = rows[0][0]
    stamp = path.stat()
    path.write_bytes(b"Z" * len(path.read_bytes()))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="dependency bytes changed"):
        service._toolchain(work)
    assert seen[-1] == len(rows) + 1  # the stamp changed, so the closure was resolved again
    path.write_bytes(b"product 0")
    service._toolchain(work)
    (lib / "Extra.olean").write_bytes(b"x")
    with pytest.raises(ValueError, match="product membership differs"):
        service._toolchain(work)
    (lib / "Extra.olean").unlink()
    rows[4][0].unlink()
    with pytest.raises(ValueError, match="dependency bytes changed"):
        service._toolchain(work)


def test_service_never_reuses_a_closure_that_became_indirect(tmp_path, monkeypatch):
    service, seen, rows, _ = _service(tmp_path, monkeypatch)
    work = tmp_path / "work"
    service._toolchain(work)
    path = rows[1][0]
    real = tmp_path / "real.olean"
    real.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(real)
    service._toolchain(work)
    assert seen[-1] == len(rows) + 1  # resolved afresh (the real snapshot call would refuse an indirect input)


def test_frozen_import_root_matches_the_original_algorithm():
    def original(root, originals, frozen):
        matching = [
            target.parents[len(source.relative_to(root).parts) - 1]
            for source, target in zip(originals, frozen, strict=True)
            if source.is_relative_to(root)
        ]
        if not matching:
            return None
        if len(set(matching)) != 1:
            raise ValueError("common proof import root has no exact frozen file closure")
        return matching[0]

    rng = random.Random(7)
    names = ["a", "b", "c"]
    for _ in range(300):
        snapshot = Path("/snap/repo")
        sources = [Path("/o", *(rng.choice(names) for _ in range(rng.randint(1, 4)))) for _ in range(rng.randint(0, 8))]
        frozen = [snapshot / s.relative_to("/") for s in sources]
        root = Path("/o", *(rng.choice(names) for _ in range(rng.randint(0, 2))))
        try:
            expected = ("ok", original(root, sources, frozen))
        except Exception as exc:
            expected = (type(exc).__name__,)
        try:
            actual = ("ok", CT._frozen_import_root(root, sources, frozen))
        except Exception as exc:
            actual = (type(exc).__name__,)
        assert actual == expected
