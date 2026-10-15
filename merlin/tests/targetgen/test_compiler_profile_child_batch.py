"""Original bulk descriptor custody; no compiler or native grading campaign."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextvars import copy_context

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import original as original_fixture

from merlin.common import bundle_snapshot as BW

original = original_fixture


def specs(original, count=15):
    paths = []
    for index in range(count):
        root = original.private / f"case-{index:03d}"
        root.mkdir()
        path = root / "spec.json"
        path.write_text(json.dumps({"package_dir": str(original.candidate), "target": original.target.target}))
        paths.append(path)
    return paths


def scope(original):
    return R.original_compiler_execution(original.owner, target_experiment=original.target,
        candidate=original.candidate, work_root=original.private / "executor", sandbox_inputs=original.inputs)


def batch(original, paths):
    return CHILD.prepare_model_children(spec_paths=paths, environment=dict(os.environ),
                                       package_dir=original.candidate)


def test_fifteen_distinct_sealed_cases_check_shared_metadata_four_times_instead_of_forty_five(original, monkeypatch):
    paths, scans = specs(original), []
    scan = BW._scanned_members

    def observed(root):
        scans.append(root)
        return scan(root)

    monkeypatch.setattr(BW, "_scanned_members", observed)
    with scope(original), BW.verified_snapshot_scope(
            original.owner.workspace, original.owner.bundle, repo=original.owner.repo):
        scans.clear()
        with batch(original, paths) as transports:
            assert len(scans) == 4
            assert len(transports) == len({transport.fd for transport in transports}) == 15
            assert len({transport.document["work_root"] for transport in transports}) == 15
            common = [{key: value for key, value in transport.document.items()
                       if key not in {"spec", "work_root"}} for transport in transports]
            assert all(row == common[0] for row in common)
            for path, transport in zip(paths, transports, strict=True):
                assert transport.document["spec"] == CHILD._pin(path)
                with pytest.raises(OSError):
                    os.write(transport.fd, b"changed")
        scans.clear()
        with ExitStack() as descriptors:
            for path in paths:
                descriptors.enter_context(CHILD.prepare_model_child(spec_path=path,
                    environment=dict(os.environ), package_dir=original.candidate))
            assert len(scans) == 45


@pytest.mark.parametrize("duplicate", ["spec", "work_root"])
def test_duplicate_spec_or_child_work_root_refuses(original, duplicate):
    paths = specs(original, 2)
    if duplicate == "spec":
        paths[1] = paths[0]
    else:
        paths[1] = paths[0].parent / "other.json"
        paths[1].write_bytes(paths[0].read_bytes())
    with scope(original), BW.verified_snapshot_scope(
            original.owner.workspace, original.owner.bundle, repo=original.owner.repo):
        with pytest.raises(StageGateError, match="distinct"):
            with batch(original, paths):
                pytest.fail("ambiguous child owner was issued")


def test_batch_cannot_reuse_expired_or_other_thread_snapshot_authority(original):
    paths = specs(original, 2)
    with scope(original):
        with BW.verified_snapshot_scope(original.owner.workspace, original.owner.bundle, repo=original.owner.repo):
            context = copy_context()
            with ThreadPoolExecutor(max_workers=1) as worker:
                def foreign():
                    with batch(original, paths):
                        pytest.fail("copied context acquired original-thread batching authority")
                with pytest.raises(StageGateError, match="snapshot operation"):
                    worker.submit(context.run, foreign).result()
        with pytest.raises(StageGateError, match="snapshot operation"):
            with batch(original, paths):
                pytest.fail("expired setup acquired batching authority")


@pytest.mark.parametrize("changed", ["candidate", "spec", "host_closure"])
def test_shared_source_spec_and_host_closure_reopen_before_issuance_completes(original, monkeypatch, changed):
    paths, pin = specs(original, 2), CHILD._pin
    altered = False

    def observed(path):
        nonlocal altered
        result = pin(path)
        if path == paths[1] and not altered:
            altered = True
            if changed == "host_closure":
                closure = CHILD._closure
                monkeypatch.setattr(CHILD, "_closure", lambda: [*closure(), {"path": "changed", "sha256": "0" * 64}])
            else:
                source = original.candidate / "tool.py" if changed == "candidate" else paths[0]
                before, raw = source.stat(), source.read_bytes()
                source.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
                os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(CHILD, "_pin", observed)
    with scope(original), BW.verified_snapshot_scope(
            original.owner.workspace, original.owner.bundle, repo=original.owner.repo):
        with pytest.raises(StageGateError, match="changed"):
            with batch(original, paths):
                pytest.fail("mutated original selection completed issuance")
