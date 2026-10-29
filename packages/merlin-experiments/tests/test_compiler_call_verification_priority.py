"""Custody scans release their slot before execution; final failures stay failures."""
from contextlib import contextmanager

import pytest
from merlin.common import cpu_pool
from merlin_experiments.phase1.brokers import compiler_call_channel as C


def transport(monkeypatch, *, failure=None):
    service = C.CommonTransformService.__new__(C.CommonTransformService)
    events, held = [], []
    count = 0
    def verify(self):
        nonlocal count
        count += 1
        assert held == [cpu_pool.VERIFICATION]
        events.append(('verify', count))
        if failure == count:
            raise ValueError('snapshot changed')
    monkeypatch.setattr(C.CommonTransformService, '_verify', verify)
    @contextmanager
    def lease(**kwargs):
        assert not held  # no nested execution/verification admission
        level = cpu_pool.lease_priority()
        held.append(level); events.append(('acquire', level))
        try: yield
        finally:
            events.append(('release', held.pop()))
    @contextmanager
    def transfer(**kwargs):
        assert held == [cpu_pool.GRADE]
        events.append(('transfer', 'open'))
        try: yield object()
        finally: events.append(('transfer', 'closed'))
    @contextmanager
    def inner(*args, **kwargs):
        assert held == [cpu_pool.GRADE]
        events.append(('broker', 'ready'))
        try: yield 'ready'
        finally: events.append(('broker', 'reaped'))
    monkeypatch.setattr(C,'shared_compute_pool',lease)
    monkeypatch.setattr(C,'shared_compute_enabled',lambda:True)
    monkeypatch.setattr(C,'transfer_compiler_cpu',transfer)
    monkeypatch.setattr(C,'_service_owner',lambda service:{'exact':'owner'})
    monkeypatch.setattr(C,'_compiler_call_transport',inner)
    return service,events,held


def invoke(service):
    return C.compiler_call_transport(service,workspace='unused',package='unused',input_path='unused',
                                    command_name='compile',timeout_s=10,borrow_compute=True)


def test_verification_before_and_after_execution_lease(monkeypatch):
    service,events,held=transport(monkeypatch)
    with cpu_pool.cpu_priority(cpu_pool.GRADE):
        with invoke(service) as value:
            assert value=='ready' and held==[cpu_pool.GRADE]
            events.append(('compiler','completed'))
        assert cpu_pool.lease_priority()==cpu_pool.GRADE
    assert events==[('acquire',5),('verify',1),('release',5),('acquire',2),
        ('transfer','open'),('broker','ready'),('compiler','completed'),('broker','reaped'),
        ('transfer','closed'),('release',2),('acquire',5),('verify',2),('release',5)]


@pytest.mark.parametrize('failure',[1,2])
def test_mutation_refuses_before_publish_and_restores_leases(monkeypatch,failure):
    service,events,held=transport(monkeypatch,failure=failure)
    with cpu_pool.cpu_priority(cpu_pool.GRADE):
        with pytest.raises(ValueError,match='snapshot changed'):
            with invoke(service): pass
    assert held==[]
    if failure==1: assert ('broker','ready') not in events
    else: assert events.index(('release',2)) < events.index(('verify',2))


def test_interrupted_command_still_reaps_then_freshly_verifies(monkeypatch):
    service,events,held=transport(monkeypatch)
    with cpu_pool.cpu_priority(cpu_pool.GRADE):
        with pytest.raises(RuntimeError,match='compiler failed'):
            with invoke(service): raise RuntimeError('compiler failed')
    assert held==[] and events.index(('broker','reaped')) < events.index(('verify',2))
