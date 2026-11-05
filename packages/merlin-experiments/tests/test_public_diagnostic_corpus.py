"""Source/reference copy custody, with a neutral accelerator projection only."""

import copy
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from pathlib import Path

import pytest
import test_operator_private_exclusion as EXCLUSION
import test_public_program_curriculum as PUBLIC_FIXTURE
from merlin_experiments.phase2 import capsule_source_development as DEVELOPMENT
from merlin_experiments.phase2 import corpus as FORMAL
from merlin_experiments.phase2 import public_diagnostic_corpus as D
from merlin_experiments.phase2.contracts import StageGateError


@pytest.fixture
def published(tmp_path, monkeypatch):
    selected, private = EXCLUSION.selected(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    _, member = PUBLIC_FIXTURE.write(source, selected)
    # PP.verify_written and the real exclusion owner still check the complete
    # captured copy program/reference closure. Only accelerator applicability
    # is neutral here; no test supplies target/source qualification.
    original_projection = D.PUBLIC.original_contractions
    monkeypatch.setattr(D.PUBLIC, "original_contractions", lambda *a, **kw: [{"neutral_projection": True}])
    planning = tmp_path / "planning.json"
    planning.write_text('{"members":[],"scope":"external synthetic planning only"}')
    args = dict(
        sources=(member,),
        destination=tmp_path / "diagnostic-snapshot",
        target="fixture",
        operator_exclusion=selected,
        planning_inventory=planning,
        planning_inventory_sha256=hashlib.sha256(planning.read_bytes()).hexdigest(),
    )
    return args, member, selected, private, original_projection


def test_fresh_snapshot_reopens_original_public_writer_and_real_exclusion(published, monkeypatch):
    args, source, selection, private, _ = published
    observed = []
    original = D.EXCLUSION.OperatorPrivateExclusion.check_member

    def check(self, directory):
        assert self is selection
        observed.append(directory)
        return original(self, directory)

    monkeypatch.setattr(D.EXCLUSION.OperatorPrivateExclusion, "check_member", check)
    owner = D.freeze_public_diagnostic_corpus(**args)
    record = owner.verify()
    copied = owner.capsules[0].source_dir
    assert source in observed and copied in observed and len(observed) >= 4
    assert record["performance_cohort_admitted"] is False and record["formal_admission"] is False
    assert record["promotion_authority"] == "NONE" and record["timing"] == "UNKNOWN"
    assert not owner.root.stat().st_mode & 0o222
    assert (
        source != copied
        and (source / "capsule.interface.mlir").read_bytes() == (copied / "capsule.interface.mlir").read_bytes()
    )
    assert str(private).encode() not in owner.manifest_path.read_bytes()
    assert private.read_bytes() not in owner.manifest_path.read_bytes()
    with pytest.raises(StageGateError):
        FORMAL.verify_frozen_performance_corpus(owner)
    with pytest.raises(StageGateError):
        FORMAL.load_frozen_performance_corpus(
            owner.root, manifest_sha256=owner.manifest_sha256, capsules_sha256=owner.capsules_sha256
        )
    with pytest.raises(StageGateError, match="original host"):
        copy.copy(owner).verify()


def test_stage_finalization_reopens_original_diagnostic_owner_not_formal_schema(published, monkeypatch):
    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)

    def wrong_scope(_corpus):
        raise AssertionError("a diagnostic owner cannot become a formal performance corpus")

    monkeypatch.setattr(FORMAL, "verify_frozen_performance_corpus", wrong_scope)
    assert DEVELOPMENT.revalidate_corpus(owner) == owner.verify()
    path = owner.capsules[0].source_dir / "source-capture-golden.npy"
    path.chmod(0o644)
    path.write_bytes(b"changed after the author round")
    path.chmod(0o444)
    with pytest.raises((StageGateError, ValueError)):
        DEVELOPMENT.revalidate_corpus(owner)


def test_stage_finalization_does_not_accept_an_unissued_diagnostic_copy(published):
    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)
    with pytest.raises(StageGateError, match="original host"):
        DEVELOPMENT.revalidate_corpus(copy.copy(owner))


def test_stage_finalization_preserves_original_formal_verifier_delegation(monkeypatch):
    # This checks unchanged delegation only, not admission of this neutral value.
    value, result = object(), object()
    seen = []

    def original(corpus):
        seen.append(corpus)
        return result

    monkeypatch.setattr(FORMAL, "verify_frozen_performance_corpus", original)
    assert DEVELOPMENT.revalidate_corpus(value) is result
    assert seen == [value]


@pytest.mark.parametrize("fault", ["source", "reference", "planning", "extra_file", "writable"])
def test_snapshot_or_parent_obligation_drift_refuses(published, fault):
    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)
    if fault == "planning":
        args["planning_inventory"].write_text('{"changed":true}')
    elif fault == "extra_file":
        owner.root.chmod(0o755)
        (owner.root / "foreign").write_text("not an original source member")
        owner.root.chmod(0o555)
    else:
        source = owner.capsules[0].source_dir
        name = "capsule.interface.mlir" if fault == "source" else "source-capture-golden.npy"
        path = source / name
        path.chmod(0o644)
        if fault != "writable":
            path.write_bytes(b"changed")
    with pytest.raises((StageGateError, ValueError)):
        owner.verify()


def test_real_source_and_reference_verifier_refusal_blocks_issuance(published):
    args, source, *_ = published
    (source / "source-capture-golden.npy").write_bytes(b"forged reference")
    with pytest.raises((StageGateError, ValueError)):
        D.freeze_public_diagnostic_corpus(**args)


def test_host_only_source_cannot_be_a_diagnostic_accelerator_member(published, monkeypatch):
    args, *_, original_projection = published
    monkeypatch.setattr(D.PUBLIC, "original_contractions", original_projection)
    with pytest.raises(StageGateError, match="host-only"):
        D.freeze_public_diagnostic_corpus(**args)


def test_aggregate_file_budget_refuses_before_creating_or_copying_snapshot(published, tmp_path, monkeypatch):
    args, source, *_ = published
    second = tmp_path / "second-source"
    shutil.copytree(source, second)
    original = D.source_inventory

    def inventory(path, **kwargs):
        result = original(path, **kwargs)
        return dict(result, n_files=8192) if path in (source, second) else result

    monkeypatch.setattr(D, "source_inventory", inventory)
    with pytest.raises(StageGateError, match="aggregate copy budget"):
        D.freeze_public_diagnostic_corpus(**dict(args, sources=(source, second)))
    assert not args["destination"].exists()


def test_prospective_copy_protocol_preserves_default_and_pins_selected_bounds(published):
    args, *_ = published
    assert D._copy_limits(None) == (64 << 20, 65 << 20)
    owner = D.freeze_public_diagnostic_corpus(**args, copy_protocol=D.COPY_PROTOCOL_V2)
    assert owner.verify()["copy_transport"] == {
        "protocol": D.COPY_PROTOCOL_V2,
        "aggregate_bytes": 256 << 20,
        "snapshot_bytes": 257 << 20,
        "member_bytes": 32 << 20,
    }
    object.__setattr__(owner, "_copy_protocol", None)
    with pytest.raises(StageGateError, match="custody"):
        owner.verify()


def test_unknown_copy_protocol_refuses_before_snapshot_creation(published):
    args, *_ = published
    with pytest.raises(StageGateError, match="copy protocol"):
        D.freeze_public_diagnostic_corpus(**args, copy_protocol="unbounded")
    assert not args["destination"].exists()


def test_execution_view_changes_only_role_and_stronger_requests_not_published_bytes(published):
    args, source, *_ = published
    raw = (source / "capsule.yaml").read_bytes()
    owner = D.freeze_public_diagnostic_corpus(**args)
    member = owner.capsules[0]
    view = owner.execution_capsule(member, contract=Path("merlin/contract").resolve())
    provenance = view.pop("public_diagnostic_execution_view")
    assert view.pop("__dir__") == str(member.source_dir)
    expected = copy.deepcopy(member.descriptor)
    expected.update(source_role="pytorch_model_slice", required_oracle_tiers=["L0", "L1", "L2", "L3"])
    assert view == expected
    assert provenance["original_required_tiers"] == member.descriptor["required_oracle_tiers"]
    assert provenance["complete_source_and_reference_tree_sha256"] == member.source_sha256
    assert provenance["native_policy"].startswith("original saved declaration")
    assert (source / "capsule.yaml").read_bytes() == (member.source_dir / "capsule.yaml").read_bytes() == raw
    with pytest.raises(StageGateError, match="substituted"):
        owner.execution_capsule(copy.copy(member), contract=Path("merlin/contract").resolve())


def test_execution_view_cannot_drop_unknown_original_oracle_obligations(published):
    import yaml

    args, source, *_ = published
    path = source / "capsule.yaml"
    declaration = yaml.safe_load(path.read_bytes())
    declaration["required_oracle_tiers"] = ["L0", "L5"]
    path.write_text(yaml.safe_dump(declaration))
    owner = D.freeze_public_diagnostic_corpus(**args)
    with pytest.raises(StageGateError, match="unknown original"):
        owner.execution_capsule(owner.capsules[0], contract=Path("merlin/contract").resolve())


def test_actual_sealed_child_preserves_host_view_and_original_published_declaration(published, tmp_path, monkeypatch):
    import test_checked_source_child as CONTROL
    import test_compiler_profile_child as PROFILE
    from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD

    from merlin.targetgen import checked_source_model_diagnostic as SELECTED
    from merlin.targetgen.compiler_pass_selection import scoped_compiler_passes

    if not hasattr(CHILD, "require_restored_model_context"):
        pytest.skip(
            "selected legacy private child lacks the current checked-source transport; test integrated namespace"
        )

    args, *_ = published
    corpus = D.freeze_public_diagnostic_corpus(**args)
    view = corpus.execution_capsule(corpus.capsules[0], contract=Path("merlin/contract").resolve())
    profile_root = tmp_path / "original-child-profile"
    profile_root.mkdir()
    original = PROFILE.original.__wrapped__(profile_root, monkeypatch)
    # Match the actual selected private module's namespace in this managed
    # hybrid checkout; the generic fixture otherwise selects only its old repo.
    private_sources = str(Path(SELECTED.__file__).resolve().parents[2])
    roots = os.environ["PYTHONPATH"].split(os.pathsep)
    roots.insert(1, private_sources)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(roots))
    passes, selected = CONTROL._selection(original)
    job = json.loads(original.spec.read_bytes())
    job["capsule"] = dict(view, oracle_tier_ceiling="L2")
    original.spec.write_text(json.dumps(job))
    script = PROFILE._CONTROL.replace(
        "    job=json.loads(raw)",
        """    job=json.loads(raw)
    import hashlib,yaml,pathlib
    view=job['capsule']; proof=view['public_diagnostic_execution_view']
    declaration=pathlib.Path(proof['original_declaration']['path'])
    assert hashlib.sha256(declaration.read_bytes()).hexdigest()==proof['original_declaration']['sha256']
    original=yaml.safe_load(declaration.read_bytes())
    assert original['source_role']=='materialized_performance_capture'
    assert view['source_role']=='pytorch_model_slice'
    assert view['required_oracle_tiers']==['L0','L1','L2','L3']
    assert view['numeric_policy']==original['numeric_policy']
    assert view['inputs']==original['inputs'] and view['expected']==original['expected']
    from merlin.targetgen import checked_source_model_diagnostic as D
    D.selection_from_context(job['candidate_context'][D.CONTEXT_FIELD],
      capsule=view,target=job['target'],package_dir=job['package_dir'])""",
    )
    with scoped_compiler_passes(passes), SELECTED.scoped_checked_source_diagnostic(selected), PROFILE._scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))


@pytest.mark.parametrize("field", ["_operator", "_sources", "_functions", "_snapshot_inventory", "_planning_inventory"])
def test_private_custody_fields_cannot_be_rebound_even_to_other_real_owners(published, tmp_path, field):
    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)
    if field == "_operator":
        alternative = tmp_path / "alternative-private"
        alternative.mkdir()
        value, _ = EXCLUSION.selected(alternative)
    elif field == "_snapshot_inventory":
        value = dict(owner._snapshot_inventory, n_files=owner._snapshot_inventory["n_files"] + 1)
    elif field == "_planning_inventory":
        alternate = tmp_path / "alternate-planning.json"
        alternate.write_bytes(args["planning_inventory"].read_bytes())
        value = D._leaf(alternate)
    else:
        value = ()
    object.__setattr__(owner, field, value)
    with pytest.raises(StageGateError, match="private owner or verifier custody"):
        owner.verify()


def test_selected_typed_exclusion_algorithm_cannot_change_under_same_private_selection(published, monkeypatch):
    from merlin_experiments.phase0 import operator_capture_exclusion

    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)
    monkeypatch.setattr(operator_capture_exclusion, "signatures", lambda *_a, **_kw: ())
    with pytest.raises(StageGateError, match="original verifier changed"):
        owner.verify()


def _three_published_sources(published, tmp_path, monkeypatch):
    args, first, selection, *_ = published
    sources = [first]
    original_entries = PUBLIC_FIXTURE.entries
    for index in (1, 2):
        parent = tmp_path / f"additional-source-{index}"
        parent.mkdir()

        def entries(path, *, suffix=index):
            rows, outputs = original_entries(path)
            rows[0]["name"] += f"_independent_fixture_{suffix}"
            return rows, outputs

        with monkeypatch.context() as scoped:
            scoped.setattr(PUBLIC_FIXTURE, "entries", entries)
            _, member = PUBLIC_FIXTURE.write(parent, selection)
        sources.append(member)
    return dict(args, sources=tuple(sources))


def test_batch_views_are_identical_with_linear_fresh_public_and_private_checks(published, tmp_path, monkeypatch):
    args = _three_published_sources(published, tmp_path, monkeypatch)
    public_checks, private_checks = [], []
    public = D.PUBLIC.verify_written
    private = D.EXCLUSION.OperatorPrivateExclusion.check_member

    def check_public(descriptor, directory):
        public_checks.append(directory)
        return public(descriptor, directory)

    def check_private(self, directory):
        private_checks.append(directory)
        return private(self, directory)

    monkeypatch.setattr(D.PUBLIC, "verify_written", check_public)
    monkeypatch.setattr(D.EXCLUSION.OperatorPrivateExclusion, "check_member", check_private)
    owner = D.freeze_public_diagnostic_corpus(**args)
    contract = Path("merlin/contract").resolve()
    expected = tuple(owner.execution_capsule(member, contract=contract) for member in owner.capsules)
    public_checks.clear()
    private_checks.clear()
    actual = owner.execution_capsules(contract=contract)
    assert actual == expected and type(actual) is tuple
    # The original exclusion check itself reopens the public writer proof;
    # preserve that nested verification as well as _published's direct check.
    assert len(public_checks) == 8 * len(owner.capsules)
    assert len(private_checks) == 4 * len(owner.capsules)
    for member in owner.capsules:
        assert public_checks.count(member.source_dir) == 8
        assert private_checks.count(member.source_dir) == 4


def test_batch_final_roster_check_rejects_an_earlier_members_mid_batch_reference_change(
    published, tmp_path, monkeypatch
):
    from merlin.targetgen import capsule_common

    args = _three_published_sources(published, tmp_path, monkeypatch)
    owner = D.freeze_public_diagnostic_corpus(**args)
    original = capsule_common.verify_capture_tool
    touched = []

    def mutate_prior_member(directory, descriptor):
        original(directory, descriptor)
        if directory == owner.capsules[1].source_dir:
            path = owner.capsules[0].source_dir / "source-capture-golden.npy"
            path.chmod(0o644)
            path.write_bytes(b"mutated after the first complete view")
            path.chmod(0o444)
            touched.append(path)

    monkeypatch.setattr(capsule_common, "verify_capture_tool", mutate_prior_member)
    with pytest.raises((StageGateError, ValueError)):
        owner.execution_capsules(contract=Path("merlin/contract").resolve())
    assert touched


def test_failed_batch_still_completes_final_live_roster_verification(published, tmp_path, monkeypatch):
    from merlin.targetgen import capsule_common

    args = _three_published_sources(published, tmp_path, monkeypatch)
    verifies = []
    original_verify = D.FrozenPublicDiagnosticCorpus.verify

    def verify(self):
        verifies.append(self)
        return original_verify(self)

    monkeypatch.setattr(D.FrozenPublicDiagnosticCorpus, "verify", verify)
    owner = D.freeze_public_diagnostic_corpus(**args)
    original_capture = capsule_common.verify_capture_tool

    def fail_second_member(directory, descriptor):
        if directory == owner.capsules[1].source_dir:
            raise ValueError("original per-member capture refusal")
        return original_capture(directory, descriptor)

    monkeypatch.setattr(capsule_common, "verify_capture_tool", fail_second_member)
    verifies.clear()
    with pytest.raises(ValueError, match="original per-member capture refusal"):
        owner.execution_capsules(contract=Path("merlin/contract").resolve())
    assert verifies == [owner, owner]


def test_issued_batch_method_cannot_be_replaced(published, monkeypatch):
    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)
    monkeypatch.setattr(D.FrozenPublicDiagnosticCorpus, "execution_capsules", lambda *_a, **_kw: ())
    with pytest.raises(StageGateError, match="original verifier changed"):
        owner.verify()


def _hold_public_metadata_cpu(cpu, ready, release):
    from merlin.common import cpu_pool as C

    os.environ["MERLIN_COMPUTE_CPUS"] = str(cpu)
    with C.cpu_priority(C.GRADE), C.shared_compute_pool(timeout_s=10):
        ready.set()
        if not release.wait(20):
            raise TimeoutError("public metadata holder did not release")


def test_readonly_corpus_metadata_bypasses_unrelated_grade_waiter(published, monkeypatch, tmp_path):
    from contextlib import contextmanager

    from merlin.common import cpu_pool as C

    args, *_ = published
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("cross-CPU metadata contention needs two allowed CPUs")
    busy, free = cpus[:2]
    root = tmp_path / "metadata-priority-slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(root))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(free))
    monkeypatch.setenv(C.PRIORITY_ENV, str(C.GRADE))
    for key in ("MERLIN_PREPARATION_CPUS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    original_pool = D.shared_compute_pool
    priorities = []

    @contextmanager
    def bounded_original_leaf(**kwargs):
        assert kwargs == {"timeout_s": 120}
        priorities.append((C.lease_priority(), os.environ.get(C.PRIORITY_ENV)))
        with original_pool(timeout_s=2):
            yield

    monkeypatch.setattr(D, "shared_compute_pool", bounded_original_leaf)
    fork = multiprocessing.get_context("fork")
    ready, release = fork.Event(), fork.Event()
    holder = fork.Process(target=_hold_public_metadata_cpu, args=(busy, ready, release))
    queued = fork.Process(target=_hold_public_metadata_cpu, args=(busy, fork.Event(), release))
    holder.start()
    try:
        assert ready.wait(10)
        queued.start()
        deadline = time.monotonic() + 10
        while not tuple((root / "waiters").glob(f"{C.GRADE}-*.wait")):
            assert time.monotonic() < deadline, "normal-grade waiter did not queue"
            time.sleep(0.01)
        with C.cpu_priority(C.GRADE):
            with pytest.raises(TimeoutError):
                with C.shared_compute_pool(timeout_s=0.1):
                    pytest.fail("normal metadata bypassed older grade waiter")
            owner = D.freeze_public_diagnostic_corpus(**args)
            owner.verify()
            assert C.lease_priority() == C.GRADE
            assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
            assert not C.current_thread_owns_cpu_lease()
            reference = owner.capsules[0].source_dir / "source-capture-golden.npy"
            reference.chmod(0o644)
            reference.write_bytes(b"changed reference")
            reference.chmod(0o444)
            with pytest.raises((StageGateError, ValueError)):
                owner.verify()
            assert C.lease_priority() == C.GRADE
            assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
        assert priorities and all(row == (C.AUTHOR, str(C.GRADE)) for row in priorities)
        assert queued.is_alive(), "unrelated grade owner was not preserved"
    finally:
        release.set()
        for process in (holder, queued):
            if process.pid is not None:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                assert process.exitcode == 0
