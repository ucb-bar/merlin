"""Native capsule workers rebind the existing exact owner; no target execution."""
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest
from test_compiler_profile_concurrency import original
from merlin.common import bundle_snapshot as B
from merlin.targetgen import capsule_runner as C, package_runtime as P
from merlin.verify import lean_emission_context as E
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R


def test_native_suite_workers_share_entry_and_keep_fresh_final_reads(original,monkeypatch):
    owner,candidate,work=original
    monkeypatch.setenv('MERLIN_OPERATOR_GRADE_PROCESSES','0')
    monkeypatch.setattr(C,'load_package',lambda *a,**k:SimpleNamespace())
    monkeypatch.setattr(C,'integrity_scan',lambda p:None)
    monkeypatch.setattr(C,'build_package',lambda p:None)
    monkeypatch.setattr(C,'_split_ineligible',lambda caps,target:(caps,[]))
    monkeypatch.setattr(C._tier_policy,'covering_set',lambda caps:[])
    monkeypatch.setattr(C,'_CALIBRATION_CAP',0)
    guards=[];finals=[];threads=[];barrier=threading.Barrier(2)
    original_guard=B._RosterGuard.__init__
    original_payload=B._payload_content
    def guard(self,*args,**kwargs):
        guards.append(threading.get_ident());original_guard(self,*args,**kwargs)
    def payload(members):
        finals.append(threading.get_ident());return original_payload(members)
    def capsule(cap,*args,**kwargs):
        assert B._selected_entry() is entry
        threads.append(threading.get_ident())
        with E._operation_snapshot(E._ACTIVE.get()):
            assert B._ACTIVE_VERIFICATION.get().thread is threading.current_thread()
            barrier.wait(timeout=10)
        return {'capsule':cap['name'],'status':'pass','tiers':{}}
    monkeypatch.setattr(C,'run_capsule',capsule)
    with R.original_compiler_execution(owner,target_experiment=None,candidate=candidate,work_root=work):
        entry=P.active_package_executor().snapshot_entry_for_call()
        assert entry is not None
        with monkeypatch.context() as m:
            m.setattr(B._RosterGuard,'__init__',guard);m.setattr(B,'_payload_content',payload)
            result=C._run_suite([{'name':'A'},{'name':'B'}],candidate,runs_root=work/'runs',max_workers=2)
        assert {r['capsule'] for r in result}=={'A','B'}
        assert len(set(threads))==2 and guards==[]
        assert len(finals)==2  # Native custody completion independently reopens payload.
    assert entry.guard.fd is None and B._selected_entry() is None


@pytest.mark.parametrize('change',['source','active','exception'])
def test_worker_scope_preserves_owner_checks_and_cleanup(original,monkeypatch,change):
    owner,candidate,work=original
    with R.original_compiler_execution(owner,target_experiment=None,candidate=candidate,work_root=work):
        executor=P.active_package_executor()
        def job():
            assert B._selected_entry() is None
            try:
                with C._original_capsule_worker_scope():
                    assert B._selected_entry() is executor.snapshot_entry_for_call()
                    if change=='source':monkeypatch.setattr(executor,'source_sha256','changed')
                    elif change=='active':monkeypatch.setattr(R,'active_original_execution',lambda:object())
                    else:raise RuntimeError('worker failed')
            finally:
                assert B._selected_entry() is None
        with monkeypatch.context() as m:
            # Restore identity before the parent lifetime's own independent exit check.
            prior_source=executor.source_sha256;prior_active=R.active_original_execution
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    with pytest.raises((ValueError,RuntimeError)):
                        pool.submit(copy_context().run,job).result(timeout=20)
            finally:
                executor.source_sha256=prior_source;R.active_original_execution=prior_active


def test_worker_shared_entry_cannot_hide_snapshot_payload_mutation(original):
    owner,candidate,work=original
    with pytest.raises((RuntimeError,ValueError)):
        with R.original_compiler_execution(owner,target_experiment=None,candidate=candidate,work_root=work):
            payload=next(p for p in B.bundle_snapshot_root(owner.workspace).rglob('*.py')
                         if p.is_file() and p.stat().st_size<1024)
            def job():
                with C._original_capsule_worker_scope():
                    with E._operation_snapshot(E._ACTIVE.get()):
                        payload.chmod(0o600)
                        payload.write_bytes(payload.read_bytes()+b'\n# changed snapshot\n')
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(copy_context().run,job).result(timeout=20)
