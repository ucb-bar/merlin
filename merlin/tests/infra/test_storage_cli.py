"""``merlin-storage`` must measure sharing honestly and prune only what it can prove is safe.

The tool exists because two disk incidents were each diagnosed slowly: the numbers that explain
*why* a root is large -- how much of it is the same bytes under several names, and which subtrees are
declared regenerable -- were never printed. A tool that got those numbers wrong, or that reclaimed
something a run still needed, would be worse than the absence it replaces.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.common import content_store as CS
from merlin.common import storage_cli as SC
from merlin.common import storage_lifecycle as lifecycle
from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    """An isolated out/ root and content store, so the assertions are about known bytes.

    The declared workspace scan roots are dropped for the same reason: they are absolute paths into
    the live checkout, so a report run inside a test would otherwise price 27 real input closures
    alongside the two the test wrote. The rest of the contract -- the roster this module reads -- is
    left exactly as it ships.
    """
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "out" / "artifacts" / "cache" / CS.NAMESPACE))
    (tmp_path / "out" / "artifacts" / "cache").mkdir(parents=True)
    declared = dict(SC.contract(), scan_roots=[])
    monkeypatch.setattr(SC, "contract", lambda: declared)
    return tmp_path


def test_sharing_is_reported_not_double_counted(rooted):
    """Apparent minus occupied IS the claim: a multiply-linked inode must be counted once."""
    tree = rooted / "out" / "artifacts" / "thing"
    tree.mkdir(parents=True)
    (tree / "original").write_bytes(b"z" * 1000)
    os.link(tree / "original", tree / "second-name")

    usage = SC.measure(tree)
    assert usage.files == 2
    assert usage.apparent == 2000
    assert usage.occupied == 1000, "the same inode was charged twice"


def test_a_snapshot_still_holding_bytes_is_never_pruned(rooted):
    """The safety property of the store: an object a live snapshot links is not an orphan."""
    repo = rooted / "repo"
    (repo / "inputs").mkdir(parents=True)
    (repo / "inputs" / "weights.bin").write_bytes(b"w" * 2048)
    bundle = {"allowed": [{"path": "inputs"}]}
    ws = rooted / "out" / "runs" / "t" / "s" / "r" / "workspace"

    BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    orphans, orphan_bytes = SC.store_orphans()
    assert orphans == [], "an object a live snapshot links was offered for deletion"
    assert orphan_bytes == 0

    # Once the run's snapshot goes, the object's last link is the store's own and it is reclaimable.
    BW.remove_bundle_snapshot(ws)
    orphans, orphan_bytes = SC.store_orphans()
    assert len(orphans) == 1
    assert orphan_bytes == 2048


def test_pruning_an_orphan_does_not_disturb_a_run_that_shares_it(rooted):
    """Two runs, one pruned: the survivor's bytes and its frozen mode must both be intact."""
    repo = rooted / "repo"
    (repo / "inputs").mkdir(parents=True)
    (repo / "inputs" / "weights.bin").write_bytes(b"payload")
    bundle = {"allowed": [{"path": "inputs"}]}
    keep = rooted / "out" / "runs" / "t" / "s" / "keep" / "workspace"
    drop = rooted / "out" / "runs" / "t" / "s" / "drop" / "workspace"

    manifest = BW.materialize_bundle_inputs(keep, bundle, repo=repo)
    BW.materialize_bundle_inputs(drop, bundle, repo=repo)
    BW.remove_bundle_snapshot(drop)

    assert SC.main(["prune", "--apply", "store-orphans"]) == 0
    survivor = BW.bundle_snapshot_root(keep) / "repo" / "inputs" / "weights.bin"
    assert survivor.read_bytes() == b"payload"
    assert not (survivor.stat().st_mode & 0o222)
    assert BW.verify_bundle_snapshot(keep, bundle, repo=repo) == manifest


def test_an_abandoned_input_closure_is_reclaimable_and_a_complete_one_is_not(rooted):
    """A terminal owner establishes abandonment; the suffix only identifies incomplete work."""
    runs = rooted / "out" / "runs" / "t" / "s" / "r"
    complete = runs / "bundle_inputs"
    abandoned = runs / "bundle_inputs.pending"
    for root in (complete, abandoned):
        root.mkdir(parents=True)
        (root / "half.bin").write_bytes(b"x" * 64)
    held = lifecycle.acquire(abandoned, owner="interrupted copy")
    held.close("failed")

    found, total = SC.pending_snapshots()
    assert found == [abandoned] and total == 64
    assert SC.main(["prune", "--apply", "pending-snapshots"]) == 0
    assert not abandoned.exists() and (complete / "half.bin").is_file()


def test_a_dry_run_is_the_default_and_removes_nothing(rooted, capsys):
    """The tool must not be able to delete because someone forgot a flag."""
    cache = rooted / "out" / "artifacts" / "cache" / "some-namespace"
    cache.mkdir(parents=True)
    (cache / "derived.json").write_text("{}", encoding="utf-8")

    assert SC.main(["prune"]) == 0
    assert (cache / "derived.json").is_file()
    assert "dry run" in capsys.readouterr().out

    assert SC.main(["prune", "--apply", "caches"]) == 0
    assert not cache.exists()


def test_the_store_is_never_offered_as_a_regenerable_cache(rooted):
    """It lives under the purgeable cache root, but wiping it wholesale re-copies live bytes; the
    orphan class is the right instrument, so `caches` must exclude it."""
    store = Path(os.environ["MERLIN_BUNDLE_CAS"])
    store.mkdir(parents=True, exist_ok=True)
    (store / "object").write_bytes(b"live")
    other = rooted / "out" / "artifacts" / "cache" / "rtl_introspect"
    other.mkdir(parents=True)
    (other / "facts.json").write_text("{}", encoding="utf-8")

    found, _ = SC.purgeable_caches()
    assert other in found and store not in found


def test_report_names_the_pre_store_snapshots_it_cannot_explain(rooted, capsys):
    """The diagnosis that took two incidents to reach, printed: closures that share nothing."""
    for name in ("run1", "run2"):
        root = rooted / "out" / "runs" / "t" / "s" / name / "bundle_inputs"
        root.mkdir(parents=True)
        (root / "weights.bin").write_bytes(b"d" * 4096)  # separate inodes, identical content

    assert SC.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "2 snapshots" in out
    assert "share almost nothing" in out


def test_report_json_is_machine_readable(rooted, capsys):
    import json  # noqa: PLC0415

    assert SC.main(["report", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["out_root"] == str(rooted / "out")
    assert set(report["reclaimable"]) == set(SC.CLASSES)
    assert all("why" in row for row in report["reclaimable"].values())


def test_reclaiming_a_tree_does_not_make_a_live_snapshots_files_writable(rooted):
    """Unlinking needs the DIRECTORY writable, not the file. Restoring a file's write bit would
    follow the shared inode into every frozen tree holding the same bytes -- the same mechanism
    that makes a mode-verified freeze unsound, so the reclaim path must never touch a file's mode."""
    repo = rooted / "repo"
    (repo / "inputs").mkdir(parents=True)
    (repo / "inputs" / "weights.bin").write_bytes(b"shared")
    bundle = {"allowed": [{"path": "inputs"}]}
    keep = rooted / "out" / "runs" / "t" / "s" / "keep" / "workspace"

    BW.materialize_bundle_inputs(keep, bundle, repo=repo)
    survivor = BW.bundle_snapshot_root(keep) / "repo" / "inputs" / "weights.bin"

    # An abandoned closure holding a hard link to the very same store object.
    abandoned = rooted / "out" / "runs" / "t" / "s" / "drop" / "bundle_inputs.pending"
    abandoned.mkdir(parents=True)
    os.link(survivor, abandoned / "weights.bin")
    abandoned.chmod(0o500)
    held = lifecycle.acquire(abandoned, owner="interrupted copy")
    held.close("failed")

    assert SC.main(["prune", "--apply", "pending-snapshots"]) == 0
    assert not abandoned.exists()
    assert not (survivor.stat().st_mode & 0o222), "the live snapshot's file was made writable"
    assert survivor.read_bytes() == b"shared"


def test_the_scan_reaches_workspaces_that_live_outside_the_out_root(tmp_path, monkeypatch):
    """The defect this contract exists to fix.

    An agent run freezes its input closure as a SIBLING of its workspace, and a capsule-bench
    workspace lives beside its target experiment rather than under out/. Scanning only the out/ root
    made the tool blind to 40 GB of completed closures and one 8.7 GB closure abandoned mid-copy --
    exactly the class it was built to find.
    """
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    merlin = tmp_path / "merlin"
    (merlin / "contract").mkdir(parents=True)
    (merlin / "contract" / "storage.yaml").write_text(
        "scan_roots:\n  - experiments/capsule_bench/targets/*/_qa_ws\n", encoding="utf-8"
    )
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    (tmp_path / "out").mkdir()

    ws = merlin / "experiments/capsule_bench/targets/gemmini/_qa_ws/run1"
    abandoned = ws / "bundle_inputs.pending"
    abandoned.mkdir(parents=True)
    (abandoned / "half.bin").write_bytes(b"x" * 4096)
    held = lifecycle.acquire(abandoned, owner="interrupted copy")
    held.close("failed")
    complete = ws / "bundle_inputs"
    complete.mkdir()
    (complete / "input.bin").write_bytes(b"y" * 2048)

    roots = SC.scan_roots()
    assert (tmp_path / "out").resolve() in roots
    assert (merlin / "experiments/capsule_bench/targets/gemmini/_qa_ws").resolve() in roots

    found, total = SC.pending_snapshots()
    assert found == [abandoned] and total == 4096
    assert SC.snapshot_sharing()["snapshots"] == 1


def test_a_declared_root_that_is_not_on_disk_is_skipped(tmp_path, monkeypatch):
    """A checkout without a given experiment is not an error."""
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    (tmp_path / "merlin" / "contract").mkdir(parents=True)
    (tmp_path / "merlin" / "contract" / "storage.yaml").write_text(
        "scan_roots:\n  - experiments/nothing/here/*\n", encoding="utf-8"
    )
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    (tmp_path / "out").mkdir()

    assert SC.scan_roots() == [(tmp_path / "out").resolve()]


def test_a_missing_contract_still_leaves_the_out_root_scannable(tmp_path, monkeypatch):
    """The contract is an addition, never a dependency: losing it must not blind the tool entirely."""
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    (tmp_path / "merlin").mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    (tmp_path / "out").mkdir()

    assert SC.scan_roots() == [(tmp_path / "out").resolve()]


def test_a_nested_root_is_not_walked_twice(tmp_path, monkeypatch):
    """Double-counting a closure would overstate both the cost and the reclaim."""
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    (tmp_path / "merlin" / "contract").mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    inner = tmp_path / "out" / "runs" / "t"
    inner.mkdir(parents=True)

    roots = SC.scan_roots(extra=[inner])
    assert roots == [(tmp_path / "out").resolve()]
