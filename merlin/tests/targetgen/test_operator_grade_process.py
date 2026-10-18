"""Sealed process transport controls, with synthetic grading and no hardware authority."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import operator_child as OC
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import _scope, original  # noqa: F401 -- share the existing exact-owner fixture

from merlin.common import bundle_snapshot as BW
from merlin.targetgen import capsule_runner as CR

CONTROL = """
import json,os,time
from pathlib import Path
from types import SimpleNamespace
from merlin.targetgen import capsule_runner as CR, package_runtime as P, provenance
from merlin.targetgen.capsule_common import make_run_paths
from merlin_experiments.phase1.feedback import operator_child as OC
P.load_package=lambda directory,**kwargs: SimpleNamespace(directory=Path(directory),manifest={},integrity_exempt=False)
provenance.toolchain_shas=lambda target: {'fixture':'synthetic-control-only'}
def grade(capsule, directory, **kwargs):
    assert P.active_package_executor() is not None
    assert kwargs['_defer_result_publication']
    assert set(kwargs['oracle_adapters']) == {'L2','L3'}
    root=Path(kwargs['runs_root']);name=capsule['name']
    path=root/'runs'/'synthetic'/name/'capsule_result.json'
    assert not path.exists(), 'unverified result was published'
    assert not path.with_name('run_manifest.yaml').exists()
    if capsule.get('barrier'):
        barrier=root/'barrier';barrier.mkdir(exist_ok=True)
        (barrier/name).write_text(str(os.getpid()))
        deadline=time.monotonic()+5
        while len(list(barrier.iterdir())) < 2:
            assert time.monotonic()<deadline, 'children were serialized'
            time.sleep(.01)
    paths=make_run_paths(root,name,suite='synthetic',target=capsule['target'],dtype='i8',benchmark=name)
    tiers={'L2':CR.TierResult('L2','pass',True),'L3':CR.TierResult('L3','pass',True,derived_from_rtl=True)}
    if capsule.get('missing_L3'):tiers.pop('L3')
    status='declined' if capsule.get('state_decline') else 'pass'
    if status=='declined':tiers={}
    row=CR._finalize_capsule_result(name=name,capsule=capsule,status=status,failure=None,tiers=tiers,
        trace_check_res={'status':'skipped','violations':[]},numeric={'status':'pass'},required={'L2','L3'},
        no_oracle=False,eff_target=capsule['target'],paths=paths,run_id=name,
        cfg=SimpleNamespace(target=capsule['target'],suite='synthetic',rtl_tiers={'L3'},oracle_tiers=('L2','L3')),
        result_filename='.operator_child_result.json',
        declined={'reason':'synthetic declared refusal'} if status=='declined' else None)
    assert not path.exists()
    assert not path.with_name('run_manifest.yaml').exists()
    row['synthetic_transport_control_only']=True
    return row
CR.run_capsule=grade
raise SystemExit(OC.main(sys.argv[1:]))
"""


@pytest.fixture
def setup(original, monkeypatch):  # noqa: F811 -- imported pytest fixture, shared with model-child controls
    monkeypatch.setattr(CR, "_CHILD_PREAMBLE", CR._CHILD_GUARD + CONTROL)
    monkeypatch.setattr(CR, "_config_for_target", lambda *args: SimpleNamespace(suite="synthetic"))
    monkeypatch.delenv("MERLIN_CERTIFY_BUDGET_S", raising=False)
    adapters = {
        "L2": CR.simulator_adapter("spike", original.target.target),
        "L3": CR.simulator_adapter("gsim", original.target.target),
    }
    original.adapters = adapters
    original.runs = original.private / "runs"
    original.runs.mkdir()
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        communicate = process.communicate

        def checked(*values, **options):
            result = communicate(*values, **options)
            if process.returncode:
                print(result[1])  # This fixture launches synthetic authority controls only.
            return result

        process.communicate = checked
        return process

    monkeypatch.setattr(OC.subprocess, "Popen", launch)
    return original


def grade(setup, name, **extra):
    capsule = {
        "name": name,
        "kind": "isa",
        "label": "public",
        "target": setup.target.target,
        "required_oracle_tiers": ["L2", "L3"],
        **extra,
    }
    return OC.grade(
        capsule,
        setup.candidate,
        runs_root=setup.runs,
        contract=None,
        target=setup.target.target,
        suite="synthetic",
        dtype="i8",
        adapters=setup.adapters,
        timeout=10,
        workers=2,
        no_oracle=False,
    )


def test_two_fresh_children_overlap_and_publish_only_verified_required_tiers(setup):
    with _scope(setup):
        OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(copy_context().run, grade, setup, name, barrier=True) for name in ("first", "second")
            ]
            rows = [future.result() for future in futures]
    assert len({row["operator_grade_process"]["pid"] for row in rows}) == 2
    assert all(row["original_compiler_child"]["schema"] == OC.CHILD.SCHEMA for row in rows)
    assert all(row["status"] == "pass" and set(row["tiers"]) == {"L2", "L3"} for row in rows)
    for row in rows:
        path = setup.runs / "runs" / "synthetic" / row["capsule"] / "capsule_result.json"
        assert json.loads(path.read_text()) == row


def test_parent_metadata_checks_serialize_while_fresh_children_still_overlap(setup, monkeypatch, tmp_path):
    """Admission must precede nested owner verification, not only its payload hash."""
    from merlin.common import cpu_pool

    cpus = sorted(os.sched_getaffinity(0))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus[:2])))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "cpu-slots"))
    monkeypatch.delenv("MERLIN_PREPARATION_CPUS", raising=False)
    monkeypatch.delenv("MERLIN_GRADE_CPUS", raising=False)
    original_executor = OC.CHILD._original_executor
    original_verify = OC.CHILD.ModelChildTransport.verify_result
    lock = threading.Lock()
    together = threading.Barrier(2, timeout=0.1)
    active, peak, checks = 0, 0, []

    @contextlib.contextmanager
    def observation(kind):
        nonlocal active, peak
        assert not cpu_pool.current_thread_owns_cpu_lease()
        with lock:
            active += 1
            peak = max(peak, active)
            checks.append(kind)
        try:
            try:
                together.wait()  # Reaches two only when preparation is outside admission.
            except threading.BrokenBarrierError:
                pass
            yield
        finally:
            with lock:
                active -= 1

    def executor():
        with observation("setup"):
            return original_executor()

    def verify(self, result):
        with observation("result"):
            return original_verify(self, result)

    monkeypatch.setattr(OC.CHILD, "_original_executor", executor)
    monkeypatch.setattr(OC.CHILD.ModelChildTransport, "verify_result", verify)
    with _scope(setup):
        OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, grade, setup, name, barrier=True) for name in ("one", "two")]
            rows = [future.result(timeout=20) for future in futures]
    assert peak == 1
    assert checks.count("setup") == 4 and checks.count("result") == 2
    assert all(row["status"] == "pass" for row in rows)
    assert len({row["operator_grade_process"]["pid"] for row in rows}) == 2


def test_missing_required_L3_is_incomplete_in_the_same_child_ladder(setup):
    with _scope(setup):
        row = grade(setup, "screen_only", missing_L3=True)
    assert row["status"] == "incomplete"
    assert row["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_expected_refusal_is_preserved_without_invented_native_tiers(setup):
    with _scope(setup):
        row = grade(
            setup,
            "must_refuse",
            state_decline=True,
            expected={
                "outcome": "refuse",
                "refusal": {"reason": "unsupported declared operation", "evidence": "fixture"},
            },
        )
    assert row["status"] == "pass" and row["expected_refusal"]["status"] == "met"
    assert row["tiers"] == {} and row["declined"]["reason"] == "synthetic declared refusal"


def test_suite_dispatch_preserves_case_completion_callback(setup, monkeypatch):
    monkeypatch.setenv(OC.ENV, "1")
    monkeypatch.setattr(CR, "load_package", lambda *args, **kwargs: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda pkg: None)
    monkeypatch.setattr(CR, "build_package", lambda pkg: None)
    monkeypatch.setattr(CR, "_split_ineligible", lambda caps, target: (caps, []))
    invoked, completed = [], []

    def child(cap, package, **kwargs):
        invoked.append((cap["name"], kwargs["adapters"], kwargs["workers"]))
        return {"capsule": cap["name"], "status": "pass"}

    monkeypatch.setattr(OC, "grade", child)
    with _scope(setup):
        rows = CR._run_suite(
            [{"name": "operator", "kind": "isa"}],
            setup.candidate,
            target=setup.target.target,
            runs_root=setup.runs,
            oracle_adapters=setup.adapters,
            max_workers=2,
            on_capsule_complete=completed.append,
        )
    assert invoked == [("operator", setup.adapters, 1)]
    assert rows == completed == [{"capsule": "operator", "status": "pass"}]


def test_source_change_before_child_restoration_never_publishes(setup, monkeypatch):
    original_popen = subprocess.Popen

    def changed(*args, **kwargs):
        (setup.candidate / "tool.py").write_text("source changed before child restoration\n")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(OC.subprocess, "Popen", changed)
    with _scope(setup), pytest.raises(StageGateError, match="without a verified completion"):
        grade(setup, "changed")
    assert not list((setup.runs / "runs").glob("*/*/capsule_result.json"))


def test_foreign_owner_and_forged_adapter_marker_cannot_select_process_grading(setup):
    with pytest.raises(StageGateError, match="original live profile"):
        OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)

    def forged(*args):
        return None

    forged._merlin_simulator_engine = "spike"
    with _scope(setup):
        with pytest.raises(StageGateError, match="original selected simulator"):
            OC.adapter_selection({"L2": forged}, target=setup.target.target)


def test_shared_budget_and_arbitrary_configuration_are_refused(setup, monkeypatch):
    with _scope(setup):
        monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "30")
        with pytest.raises(StageGateError, match="unlimited"):
            OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
        monkeypatch.delenv("MERLIN_CERTIFY_BUDGET_S")
        with pytest.raises(StageGateError, match="arbitrary configuration"):
            OC.validate(target=setup.target.target, adapters=setup.adapters, config=object(), perf_extractor=None)


def test_opt_in_is_explicit_and_invalid_selection_refuses(monkeypatch):
    monkeypatch.delenv(OC.ENV, raising=False)
    assert not OC.enabled()
    monkeypatch.setenv(OC.ENV, "1")
    assert OC.enabled()
    monkeypatch.setenv(OC.ENV, "15")
    with pytest.raises(ValueError, match="requires 0 or 1"):
        OC.enabled()


def test_parent_issuance_batches_payload_reads_and_closes_before_launch(setup, monkeypatch):
    reads = []
    original_digest = BW.hashlib.file_digest

    def measured(stream, *args, **kwargs):
        if str(stream.name) == str(setup.frozen):
            reads.append(os.fstat(stream.fileno()).st_size)
        return original_digest(stream, *args, **kwargs)

    monkeypatch.setattr(BW.hashlib, "file_digest", measured)
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_CACHE", raising=False)
    with _scope(setup):
        reads.clear()
        root = OC.CHILD.model_child_temporary_root()
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            spec = Path(temporary) / "spec.json"
            spec.write_text(json.dumps({"package_dir": str(setup.candidate)}))
            with OC.CHILD.prepare_model_child(
                spec_path=spec, environment=dict(os.environ), package_dir=setup.candidate
            ):
                baseline = len(reads), sum(reads)
        reads.clear()
        document = {"package_dir": str(setup.candidate), "capsule": {"name": "synthetic"}}
        with OC._prepared_child(document) as (spec, output, environment, transport):
            assert BW._ACTIVE_VERIFICATION.get() is None
            assert spec.is_file() and not output.exists()
            assert os.fstat(transport.fd).st_size > 0
            batched = len(reads), sum(reads)
        assert not spec.exists()
        with pytest.raises(OSError):
            os.fstat(transport.fd)
        assert batched[0] == 2 and baseline[0] > batched[0]
        assert batched[1] < baseline[1]
        print(
            f"parent issuance payload reads: {baseline[0]} -> {batched[0]}; "
            f"selected member bytes: {baseline[1]} -> {batched[1]}"
        )


def test_parent_final_hash_refuses_same_size_restored_mtime_mutation_before_Popen(setup, monkeypatch):
    prepare = OC.CHILD.prepare_model_child
    issued_fds, launched = [], []

    @contextlib.contextmanager
    def changed(*args, **kwargs):
        with prepare(*args, **kwargs) as transport:
            issued_fds.append(transport.fd)
            metadata = setup.frozen.stat()
            setup.frozen.chmod(0o600)
            with setup.frozen.open("r+b") as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
            os.utime(setup.frozen, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            yield transport

    monkeypatch.setattr(OC.CHILD, "prepare_model_child", changed)
    monkeypatch.setattr(OC.subprocess, "Popen", lambda *args, **kwargs: launched.append(True))
    with pytest.raises((RuntimeError, ValueError), match="content differs"):
        with _scope(setup):
            grade(setup, "changed-before-launch")
    assert launched == [] and issued_fds
    assert not list((setup.runs / "runs").glob("*/*/capsule_result.json"))
    for fd in issued_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_parent_scope_ends_before_Popen_and_result_verification(setup, monkeypatch):
    launch, verify = OC.subprocess.Popen, OC.CHILD.ModelChildTransport.verify_result
    seen = []

    def checked_launch(*args, **kwargs):
        assert BW._ACTIVE_VERIFICATION.get() is None
        seen.append("launch")
        return launch(*args, **kwargs)

    def checked_verify(self, result):
        assert BW._ACTIVE_VERIFICATION.get() is None
        seen.append("result")
        return verify(self, result)

    monkeypatch.setattr(OC.subprocess, "Popen", checked_launch)
    monkeypatch.setattr(OC.CHILD.ModelChildTransport, "verify_result", checked_verify)
    with _scope(setup):
        grade(setup, "independent-child")
    assert seen == ["launch", "result"]


def test_context_only_grade_class_reaches_the_sealed_child_unchanged(setup, monkeypatch, tmp_path):
    """The in-turn grader selects GRADE in its context alone, never in the process environment.

    With a shared pool, the launcher used to add ``env MERLIN_CPU_PRIORITY=2`` after the parent sealed
    the child's environment, so the restored child refused its provenance and every grade was skipped.
    """
    from merlin.common import cpu_pool
    from merlin.common import invocation_record as I

    cpu = min(os.sched_getaffinity(0))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "cpu-slots"))
    for key in (cpu_pool.PRIORITY_ENV, "MERLIN_PREPARATION_CPUS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    prepare, launch = OC.CHILD.prepare_model_child, OC.subprocess.Popen
    sealed, launched = [], []

    @contextlib.contextmanager
    def recorded_seal(**kwargs):
        with prepare(**kwargs) as transport:
            sealed.append(transport.document["environment"])
            yield transport

    def recorded_launch(command, **kwargs):
        launched.append((list(command), dict(kwargs["env"])))
        return launch(command, **kwargs)

    monkeypatch.setattr(OC.CHILD, "prepare_model_child", recorded_seal)
    monkeypatch.setattr(OC.subprocess, "Popen", recorded_launch)
    with cpu_pool.cpu_priority(cpu_pool.GRADE), _scope(setup):
        row = grade(setup, "in-turn-grade")
    assert row["status"] == "pass" and row["original_compiler_child"]["schema"] == OC.CHILD.SCHEMA
    ((command, environment),) = launched
    assert command[:3] == ["taskset", "--cpu-list", str(cpu)] and "env" not in command
    assert environment[cpu_pool.PRIORITY_ENV] == str(cpu_pool.GRADE)
    assert sealed == [I.environment_identity(environment)]


def test_sealing_refuses_an_environment_its_launcher_would_change(setup, monkeypatch, tmp_path):
    from merlin.common import cpu_pool

    cpu = min(os.sched_getaffinity(0))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "cpu-slots"))
    monkeypatch.delenv(cpu_pool.PRIORITY_ENV, raising=False)
    with _scope(setup), cpu_pool.cpu_priority(cpu_pool.GRADE):
        root = OC.CHILD.model_child_temporary_root()
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            spec = Path(temporary) / "spec.json"
            spec.write_text(json.dumps({"package_dir": str(setup.candidate)}))
            with pytest.raises(StageGateError, match="lease class"):
                with OC.CHILD.prepare_model_child(
                    spec_path=spec, environment=dict(os.environ), package_dir=setup.candidate
                ):
                    pass
            with OC.CHILD.prepare_model_child(
                spec_path=spec, environment=cpu_pool.child_environment(), package_dir=setup.candidate
            ) as transport:
                assert transport is not None


def test_expired_original_owner_cannot_issue_or_launch_operator_child(setup, monkeypatch):
    launched = []
    monkeypatch.setattr(OC.subprocess, "Popen", lambda *args, **kwargs: launched.append(True))
    with _scope(setup):
        pass
    with pytest.raises(StageGateError, match="original live profile owner"):
        grade(setup, "expired-owner")
    assert not launched and not list((setup.runs / "runs").glob("*/*/capsule_result.json"))


@pytest.mark.parametrize("failure", [None, "entry", "body", "exit"])
def test_restoration_leases_only_entry_and_exit_and_releases_on_failure(tmp_path, monkeypatch, failure):
    import fcntl

    from merlin.common import cpu_pool

    affinity = os.sched_getaffinity(0)
    cpu = min(affinity)
    slots = tmp_path / "cpu-slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    stages = []

    def leased(stage):
        assert cpu_pool.current_thread_owns_cpu_lease()
        assert os.sched_getaffinity(0) == {cpu}
        with cpu_pool.shared_compute_pool(timeout_s=0):
            assert os.sched_getaffinity(0) == {cpu}
        stages.append(stage)

    @contextlib.contextmanager
    def restore(*args, **kwargs):
        leased("entry")
        if failure == "entry":
            raise StageGateError("entry custody refused")
        try:
            yield {"original": "synthetic transport only"}
        finally:
            leased("exit")
            if failure == "exit":
                raise StageGateError("final custody refused")

    monkeypatch.setattr(OC.CHILD, "restore_model_child", restore)

    def exercise():
        with OC._restored_child(9, spec_path=tmp_path / "spec", spec_bytes=b"{}"):
            assert not cpu_pool.current_thread_owns_cpu_lease()
            assert os.sched_getaffinity(0) == affinity
            # Another case/command can acquire the only slot while this body runs.
            with cpu_pool.shared_compute_pool(timeout_s=0):
                stages.append("body-command")
            if failure == "body":
                raise StageGateError("body refused")

    if failure:
        with pytest.raises(StageGateError, match="refused"):
            exercise()
    else:
        exercise()
    assert stages == (["entry"] if failure == "entry" else ["entry", "body-command", "exit"])
    assert os.sched_getaffinity(0) == affinity
    assert not cpu_pool.current_thread_owns_cpu_lease()
    with (slots / f"cpu-{cpu}.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_parent_preparation_holds_no_cpu_and_releases_before_exposing_transport(setup, monkeypatch):
    from merlin.common import cpu_pool

    affinity = os.sched_getaffinity(0)
    cpu = min(affinity)
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(setup.private / "cpu-slots"))
    original_root = OC.CHILD.model_child_temporary_root

    def root():
        # Preparation between the verifier's own leased stages owns no slot: the only CPU stays free.
        assert not cpu_pool.current_thread_owns_cpu_lease()
        with cpu_pool.shared_compute_pool(timeout_s=0):
            pass
        return original_root()

    monkeypatch.setattr(OC.CHILD, "model_child_temporary_root", root)
    with _scope(setup):
        document = {"package_dir": str(setup.candidate), "capsule": {"name": "synthetic"}}
        with OC._prepared_child(document):
            assert not cpu_pool.current_thread_owns_cpu_lease()
            assert os.sched_getaffinity(0) == affinity
            with cpu_pool.shared_compute_pool(timeout_s=0):
                pass
    assert not cpu_pool.current_thread_owns_cpu_lease()


def test_only_tiers_executed_to_a_verdict_price_the_parent_ladder():
    """A refused or unavailable tier carries a partial wall clock; the in-process ladder never prices
    one, so a child must not either -- else a refusal that cost one ELF build reads as the tier's price
    and the parent reorders its ladder around it."""
    result = {
        "tiers": {
            "L0": {"status": "pass"},
            "L2": {"status": "pass", "timing": {"adapter_wall_s": 5.25, "build_s": 5.1, "sim_active_s": 0.15}},
            "L3": {"status": "unavailable", "timing": {"adapter_wall_s": 4.9, "partial": True}},
            "L4": {"status": "pass", "carried": {"carried": True}, "timing": None},
            "L5": {"status": "fail", "timing": {"adapter_wall_s": "unknown"}},
        }
    }
    assert OC.completed_tier_costs(result) == {"L2": 5.25}
    assert OC.completed_tier_costs({}) == {}


def test_background_capsules_verifying_hold_no_pool_cpu_and_an_author_request_runs(setup, monkeypatch):
    """A grade's capsules preparing at once never hold the pool an author request is queued for.

    Each capsule thread used to lease a CPU before verifying the input snapshot. That verification is
    GIL-bound in one controller process, and a held slot is never preempted, so a background grade held
    the whole pool for minutes while its CPUs idled and the author's self-checks waited.
    """
    from merlin.common import cpu_pool

    available = sorted(os.sched_getaffinity(0))
    if len(available) < 2:
        pytest.skip("requires two available CPUs")
    cpus, slots = (available[0], available[-1]), setup.private / "cpu-slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    capsules = 4
    arrived, release = threading.Event(), threading.Event()
    owned_in_body, held_while_verifying = [], []
    scan, original_root = BW._scanned_members, OC.CHILD.model_child_temporary_root

    def held():
        return len(cpus) - cpu_pool._free_slots(slots, cpus, fresh=True)

    def verifying(root):
        held_while_verifying.append(held())
        return scan(root)

    def preparing():
        owned_in_body.append(cpu_pool.current_thread_owns_cpu_lease())
        arrived.set()
        assert release.wait(timeout=10)
        return original_root()

    def prepare(name):
        with cpu_pool.cpu_priority(cpu_pool.GRADE):
            with OC._prepared_child({"package_dir": str(setup.candidate), "capsule": {"name": name}}):
                return name

    monkeypatch.setattr(BW, "_scanned_members", verifying)
    monkeypatch.setattr(OC.CHILD, "model_child_temporary_root", preparing)
    with _scope(setup), ThreadPoolExecutor(max_workers=capsules) as pool:
        futures = [pool.submit(copy_context().run, prepare, f"capsule-{index}") for index in range(capsules)]
        try:
            # One capsule prepares while siblings wait for admission without a CPU lease.
            assert arrived.wait(timeout=5)
            assert owned_in_body == [False]
            held_while_all_prepare = held()
            started = time.monotonic()
            with cpu_pool.cpu_priority(cpu_pool.AUTHOR), cpu_pool.shared_compute_pool(timeout_s=1):
                author_wait_s = time.monotonic() - started
        finally:
            release.set()
        assert sorted(future.result(timeout=20) for future in futures) == [f"capsule-{i}" for i in range(capsules)]
    assert held_while_all_prepare == 0 and owned_in_body == [False] * capsules
    assert author_wait_s < 0.5
    # Verification stages run one at a time behind the admission gate, each on one leased CPU.
    assert held_while_verifying and max(held_while_verifying) == 1
    assert held() == 0


@pytest.mark.parametrize("mutation", [None, "between capsules", "during preparation"])
def test_grade_reads_the_snapshot_once_before_fan_out_and_keeps_every_final_read(setup, monkeypatch, mutation):
    from merlin.targetgen import package_runtime as P

    reads = []
    original_digest = BW.hashlib.file_digest
    prepare = OC.CHILD.prepare_model_child
    mutate_during = []

    def measured(stream, *args, **kwargs):
        if str(stream.name) == str(setup.frozen):
            reads.append(os.fstat(stream.fileno()).st_size)
        return original_digest(stream, *args, **kwargs)

    def mutate():
        metadata = setup.frozen.stat()
        setup.frozen.chmod(0o600)
        with setup.frozen.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(bytes([first[0] ^ 1]))
        os.utime(setup.frozen, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    @contextlib.contextmanager
    def changed(*args, **kwargs):
        with prepare(*args, **kwargs) as transport:
            if mutate_during:
                mutate()
            yield transport

    def capsule(name):
        with OC._prepared_child({"package_dir": str(setup.candidate), "capsule": {"name": name}}):
            pass

    monkeypatch.setattr(BW.hashlib, "file_digest", measured)
    monkeypatch.setattr(OC.CHILD, "prepare_model_child", changed)
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")  # the operator selection that arms the roster guard
    with _scope(setup):
        reads.clear()
        OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
        assert len(reads) == 1  # the grade's one entry read, before any capsule fans out
        entry = OC._GRADE_ENTRIES[P.active_package_executor()]
        for index in range(3):
            reads.clear()
            capsule(f"capsule-{index}")
            assert len(reads) == 1  # each capsule's own fresh final read before its child could launch
        if mutation == "between capsules":
            mutate()
            with pytest.raises(RuntimeError, match="content differs"):
                capsule("after-change")
            assert entry.guard.fd is None  # the observed change ends reuse for every later capsule
        elif mutation == "during preparation":
            mutate_during.append(True)
            with pytest.raises(RuntimeError, match="content differs"):
                capsule("changed-in-preparation")
        if mutation is not None:
            mutate()  # Flip the byte back so the owner's own closing checks see the original bytes.
