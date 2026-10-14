"""Real original batch-worker reuse, custody and original-roster controls."""

import json
import subprocess
from dataclasses import replace

import pytest
from merlin_experiments.phase1.feedback import scale_probe_batch as B
from merlin_experiments.phase1.feedback import scale_probe_child as S
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import original as original_fixture
from test_scale_probe_process import CONTROL
from test_scale_probe_process import setup as setup_fixture

from merlin.common import cpu_pool

setup = setup_fixture
original = original_fixture
REAL_POPEN = subprocess.Popen

BATCH_CONTROL = CONTROL.replace("raise SystemExit(S.main(sys.argv[1:]))", """
from contextlib import contextmanager
from merlin_experiments.phase1.feedback import scale_probe_batch as B
restores=[]
restore=B.CHILD.restore_model_child
@contextmanager
def restored(*args,**kwargs):
    assert not restores, 'worker restored a new namespace per case'
    restores.append(os.getpid())
    with restore(*args,**kwargs) as receipt:
        yield receipt
original_emit=C._emit_scale_case
def reuse(*args,**kwargs):
    assert restores==[os.getpid()]
    result=original_emit(*args,**kwargs)
    # MUTATION_CONTROL
    return result
B.CHILD.restore_model_child=restored
C._emit_scale_case=reuse
raise SystemExit(B.main(sys.argv[1:]))
""")


@pytest.fixture
def batch_setup(setup, monkeypatch):
    from merlin_experiments.phase0 import public_scale_sources

    all_cases, _ = public_scale_sources.derive(setup.capsules, index_bits=64, tile_edge=16)
    setup.cases = all_cases[:4]
    monkeypatch.setenv("MERLIN_PUBLIC_SCALE_BATCH_WORKERS", "2")
    def patched(command, **kwargs):
        index = command.index("-m")
        assert cpu_pool._ACTIVE_LEASE.get() is None
        from merlin.common import bundle_snapshot as BW

        assert BW._ACTIVE_VERIFICATION.get() is None
        command = [*command[:index], "-c", BATCH_CONTROL, *command[index + 2:]]
        return REAL_POPEN(command, **kwargs)

    monkeypatch.setattr(B.subprocess, "Popen", patched)
    return setup


def emit(batch_setup, timeout=30):
    import time

    return S.emit_cases(owner=batch_setup.owner, candidate=batch_setup.candidate,
        target_experiment=batch_setup.target, capsules=batch_setup.capsules, cases=batch_setup.cases,
        contract=batch_setup.contract, facts_path=batch_setup.facts, work_root=batch_setup.output,
        deadline=time.monotonic()+timeout, ir_budget=None, sandbox_inputs=batch_setup.inputs)


def test_two_workers_restore_once_each_for_four_exact_cases_and_publish_after_final_owner_exit(batch_setup):
    rows = emit(batch_setup)
    assert [row["variant"] for row in rows] == [case.record()["derivation"]["variant"] for case in batch_setup.cases]
    assert all(row["lowering_status"] == "emitted" for row in rows)
    pids = [row["preparation_process"]["pid"] for row in rows]
    assert pids[0] == pids[2] != pids[1] == pids[3]
    assert max(row["preparation_process"]["started_monotonic"] for row in rows) < min(
        row["preparation_process"]["finished_monotonic"] for row in rows)
    for path in batch_setup.output.glob("worker-*/.scale_batch_progress.json"):
        progress = json.loads(path.read_bytes())
        assert progress["accepted"] is False
        assert [event["event"] for event in progress["events"]] == ["begin", "end", "begin", "end"]
    assert all(row["numeric"] == row["rtl"] == "not_attempted" for row in rows)
    assert len(list(batch_setup.output.glob("worker-*/.scale_case_*.json"))) == 4
    assert all(row["case_observation"]["pending_until_parent_custody"] is False for row in rows)


def test_deadline_retains_closed_cases_but_never_interrupted_case_or_complete_roster(batch_setup, monkeypatch):
    hanging = """if args[2].name in {'case-002','case-003'}:
        time.sleep(30)"""
    monkeypatch.setitem(globals(), "BATCH_CONTROL", BATCH_CONTROL.replace("# MUTATION_CONTROL", hanging))
    rows = emit(batch_setup, timeout=3)
    assert [row["lowering_status"] for row in rows] == ["emitted", "emitted", "unavailable", "unavailable"]
    assert all(row["preparation_process"]["outcome"] == "original_compile_deadline" for row in rows)
    assert not list(batch_setup.output.glob("worker-*/.scale_batch_result.json"))
    assert len(list(batch_setup.output.glob("worker-*/.scale_case_*.json"))) == 2
    assert all(row["numeric"] == row["rtl"] == "not_attempted" for row in rows)


def test_progress_end_without_closed_case_receipt_never_counts_as_completion(tmp_path):
    spec = tmp_path / ".scale_batch_spec.json"
    (tmp_path / ".scale_batch_progress.json").write_text(json.dumps({
        "events": [{"index": 0, "event": "end"}], "accepted": False}))
    assert B._read_case(spec, {"index": 0}, None, None, started=1, ended=2) is None


def test_checkpoint_cannot_be_issued_outside_original_restored_worker():
    with pytest.raises(StageGateError, match="inherited restored worker"):
        with B.CHILD.restored_case_checkpoint():
            pytest.fail("foreign checkpoint scope was entered")


def test_parent_verifies_each_transport_once_and_only_four_full_snapshot_reads(batch_setup, monkeypatch):
    from merlin.common import bundle_snapshot as BW

    digest, verify = BW.hashlib.file_digest, B.CHILD.ModelChildTransport._verify_original
    reads, transports = [], []

    def measured(stream, *args, **kwargs):
        if str(stream.name) == str(batch_setup.frozen):
            reads.append(stream.name)
        return digest(stream, *args, **kwargs)

    def reopened(transport):
        transports.append(transport.document["spec"]["path"])
        return verify(transport)

    monkeypatch.setattr(BW.hashlib, "file_digest", measured)
    monkeypatch.setattr(B.CHILD.ModelChildTransport, "_verify_original", reopened)
    rows = emit(batch_setup)
    assert all(row["lowering_status"] == "emitted" for row in rows)
    # One setup issuance reopen, one entry per worker, one shared final reopen.
    assert len(transports) == 4
    assert len(reads) == 4  # Actual full setup pre/post and final custody pre/post.
    assert BW._ACTIVE_VERIFICATION.get() is None


def test_parent_final_transport_source_change_refuses_all_closed_cases(batch_setup, monkeypatch):
    read, observed = B._read_case, 0

    def changed(spec, item, transport, worker, **kwargs):
        nonlocal observed
        observed += 1
        if observed == len(batch_setup.cases):
            source = batch_setup.candidate / "tool.py"
            source.write_text(source.read_text() + "# changed parent source\n")
        return read(spec, item, transport, worker, **kwargs)

    monkeypatch.setattr(B, "_read_case", changed)
    with pytest.raises((StageGateError, ValueError), match="source|owner"):
        emit(batch_setup)
    assert not (batch_setup.output / "public_compile_only.json").exists()


def test_public_checkpoint_verifier_always_reopens_original_owner(monkeypatch):
    calls = []
    monkeypatch.setattr(B.CHILD.ModelChildTransport, "_verify_original", lambda self: calls.append(self))
    transport = B.CHILD.ModelChildTransport(fd=None, document={}, executor=None, boundary=None)
    with pytest.raises(StageGateError, match="checkpoint"):
        transport.verify_checkpoint({}, worker={})
    assert calls == [transport]


@pytest.mark.parametrize("changed", ["case", "worker", "product", "late"])
def test_parent_reopens_case_receipt_worker_products_and_deadline(batch_setup, monkeypatch, changed):
    read = B._read_case

    def altered(spec, item, transport, worker, **kwargs):
        path = B._case_path(spec, item["index"])
        record = json.loads(path.read_bytes())
        if changed == "case":
            record["case"]["member"] = "foreign"
        elif changed == "worker":
            record["original_case_checkpoint"]["worker"]["start_time"] = "foreign"
        elif changed == "product":
            root = B.CHILD._plain(item["spec"]["path"]).parent
            (root / "generated/lowered.ll").write_text("changed original product")
        else:
            # Preserve valid artifacts/custody while making completion outside
            # the original budget; removing the whole-batch result models a kill.
            (spec.parent / ".scale_batch_result.json").unlink(missing_ok=True)
            record["finished_monotonic"] = kwargs["ended"]
        if changed != "product":
            S._write(path, record)
        return read(spec, item, transport, worker, **kwargs)

    monkeypatch.setattr(B, "_read_case", altered)
    if changed == "late":
        # The separate interrupted-worker test covers deadline admission; this
        # mutation would also disagree with an already verified final batch.
        with pytest.raises(StageGateError, match="disagrees"):
            emit(batch_setup)
    else:
        with pytest.raises((StageGateError, ValueError), match="case|worker|product"):
            emit(batch_setup)


@pytest.mark.parametrize("value", ["0", "-1", "", "invalid"])
def test_batch_worker_selection_refuses_invalid_values(batch_setup, monkeypatch, value):
    monkeypatch.setenv("MERLIN_PUBLIC_SCALE_BATCH_WORKERS", value)
    with pytest.raises(ValueError):
        S.batch_workers()


def test_batch_is_explicit_and_bounded_by_actual_grade_pool(batch_setup, monkeypatch):
    monkeypatch.setenv("MERLIN_PUBLIC_SCALE_BATCH_WORKERS", "15")
    assert S.batch_workers() == 2
    monkeypatch.delenv("MERLIN_PUBLIC_SCALE_BATCH_WORKERS")
    assert S.batch_workers() is None


def test_original_deadline_retains_all_four_cases_without_completion_credit(batch_setup):
    rows = emit(batch_setup, timeout=.001)
    assert len(rows) == 4
    assert all(row["lowering_status"] == "not_attempted" for row in rows)


def test_batch_owner_cannot_substitute_or_repeat_a_case_directory(batch_setup):
    root = batch_setup.output / "worker-000"
    root.mkdir()
    path = root / ".scale_batch_spec.json"
    document = {"schema": B.SCHEMA, "package_dir": str(batch_setup.candidate),
        "target": batch_setup.target.target, "deadline": 1, "output": str(root), "jobs": []}
    for index in range(2):
        spec = batch_setup.output / "case-000/.scale_child_spec.json"
        spec.parent.mkdir(exist_ok=True)
        S._write(spec, {"package_dir":document["package_dir"],"target":document["target"],
            "deadline":1,"case":{},"output":str(spec.parent)})
        document["jobs"].append({"index":index,"spec":B.CHILD._pin(spec),"case":{}})
    with pytest.raises(StageGateError, match="repeated"):
        B._jobs(document, path)


def test_worker_rederives_case_instead_of_accepting_foreign_producer_metadata(batch_setup):
    case = batch_setup.cases[0]
    derivation = json.loads(case.derivation_json)
    derivation["variant"] = "forged_original_variant"
    batch_setup.cases = (replace(case, derivation_json=json.dumps(derivation)), *batch_setup.cases[1:])
    rows = emit(batch_setup)
    assert not any(row["lowering_status"] == "emitted" for row in rows[::2])
    assert not (batch_setup.output / "worker-000/.scale_batch_result.json").exists()


@pytest.mark.parametrize("changed", ["earlier_source", "frozen_snapshot"])
def test_earlier_case_or_snapshot_mutation_cannot_publish_a_completed_batch(batch_setup, monkeypatch, changed):
    path = (repr(str(batch_setup.frozen)) if changed == "frozen_snapshot"
            else "str(args[2].parent/'sources/case-000.mlir')")
    mutation = """if args[2].name=='case-002':
        source=Path(PATH)
        before,raw=source.stat(),source.read_bytes()
        source.chmod(0o600)
        source.write_bytes(bytes([raw[0]^1])+raw[1:])
        os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))""".replace("PATH", path)
    monkeypatch.setitem(globals(), "BATCH_CONTROL", BATCH_CONTROL.replace("# MUTATION_CONTROL", mutation))
    with pytest.raises((StageGateError, ValueError, RuntimeError), match="changed|snapshot|frozen|digest|input"):
        emit(batch_setup)
    assert not (batch_setup.output / "worker-000/.scale_batch_result.json").exists()
    assert not (batch_setup.output / "public_compile_only.json").exists()
    progress = json.loads((batch_setup.output / "worker-000/.scale_batch_progress.json").read_bytes())
    assert progress["accepted"] is False

