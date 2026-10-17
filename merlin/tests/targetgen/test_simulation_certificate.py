"""Same-byte model cache controls with a synthetic byte-bound engine; no native qualification."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from merlin_experiments.phase1.feedback import operator_child as OC
from test_compiler_profile_child import _scope, original  # noqa: F401 -- exact issued original-owner fixture

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import simulation_certificate as SC
from merlin.targetgen import tier_cache as TC

SHAS = {'merlin': 'a' * 40, 'rtl': 'UNKNOWN'}


@pytest.fixture
def selected(original, monkeypatch):  # noqa: F811 -- reuse exact original-owner fixture
    monkeypatch.setenv(SC.ENV, '1')
    monkeypatch.setenv('MERLIN_TIER_CERT_CACHE', str(original.private / 'certificates'))
    monkeypatch.delenv('MERLIN_SNAPSHOT_LEASE_CACHE', raising=False)
    original.binary = original.private / 'simulator'
    original.binary.write_bytes(b'byte-bound synthetic engine')
    original.grader = original.private / 'grader.py'
    original.grader.write_bytes(b'synthetic independent judge')
    backend = SC.BoundRoCCBackend(original.target.target, '{}', None)

    def identity(self, engine):
        return {'engine': engine, 'binary': {'path': str(original.binary),
                                            'sha256': hashlib.sha256(original.binary.read_bytes()).hexdigest()},
                'selection_sha256': '1' * 64, 'receipt': {'receipt_sha256': '2' * 64,
                                                        'firrtl_sha256': '3' * 64}}

    monkeypatch.setattr(SC.BoundRoCCBackend, 'execution_engine_identity', identity)
    monkeypatch.setattr(SC.base, 'get_backend', lambda target: backend)
    monkeypatch.setattr(SC, '_instrument', lambda target: hashlib.sha256(original.grader.read_bytes()).hexdigest())
    original.adapter = CR.simulator_adapter('gsim', original.target.target)
    original.generated = original.private / 'generated'
    original.generated.mkdir()
    (original.generated / 'kernel.elf').write_bytes(b'\x7fELF-exact-current-program')
    original.inputs_to_cache = {
        'capsule': {'name': 'synthetic', 'kind': 'isa', 'operation': {'op': 'matmul'}, 'seed': 12},
        'cb': {'outputs': ['C'], 'tensors': {'A': {'values': [1, 2]}, 'C': {'shape': [1, 2]}}},
        'gold': {'C': [3, 4]}, 'reference': {'C': [3, 4]}, 'direct_oracle_golden': False,
        'policy': {'compare': 'exact_int'}, 'llvm_text': 'synthetic complete emitted LLVM',
        'adapter': original.adapter, 'target': original.target.target, 'tier': 'L3',
        'package_dir': original.candidate,
    }
    return original


def _context(selected):
    return SC.capture(**selected.inputs_to_cache)


def _write(selected, context):
    CR.record_tier_certificate('synthetic', 'L3', CR.TierResult('L3', 'pass', True, engine='gsim',
        fidelity='elaborated_rtl', derived_from_rtl=True, cycles=19),
        target=selected.target.target, generated=selected.generated, shas=SHAS, from_rtl=True,
        simulation_context=context)


def _read(selected, context):
    return CR.carried_tier_result('synthetic', 'L3', True, target=selected.target.target,
        generated=selected.generated, shas=SHAS, from_rtl=True, simulation_context=context)[0]


def test_same_complete_request_hits_separate_namespace_without_hardware_source_claim(selected):
    with _scope(selected):
        context = _context(selected)
        assert context is not None
        _write(selected, context)
        carried = _read(selected, _context(selected))
        assert carried is not None and carried.status == 'pass' and carried.cycles == 19
        assert carried.timing is None and carried.carried['certificate_scope'] == SC.NAMESPACE
        assert carried.carried['hardware_source_revision_claim'] is False
        assert len(list((SC.cache_root()).glob('*.json'))) == 1
        assert not list(SC.cache_root().parent.glob('*.json'))
        assert TC.execution_identity(target=selected.target.target,
            executables=CR.run_executables(selected.generated), toolchain_shas=SHAS) is None
        assert _read(selected, None) is None and SHAS['rtl'] == 'UNKNOWN'


@pytest.mark.parametrize('change', ['binary', 'capsule', 'stimulus', 'gold', 'reference', 'policy',
                                  'llvm', 'compiler', 'grader', 'elf'])
def test_mutated_bound_input_cannot_reuse_previous_verdict(selected, change):
    with _scope(selected):
        context = _context(selected)
        _write(selected, context)
        assert _read(selected, context) is not None
        values = selected.inputs_to_cache
        if change == 'binary':
            selected.binary.write_bytes(b'byte-bound different engine')
        elif change == 'capsule':
            values['capsule']['seed'] += 1
        elif change == 'stimulus':
            values['cb']['tensors']['A']['values'][1] = 9
        elif change in {'gold', 'reference'}:
            values[change]['C'][1] = 9
        elif change == 'policy':
            values['policy'].update(compare='tolerance_float', atol=1)
        elif change == 'llvm':
            values['llvm_text'] = 'changed LLVM source'
        elif change == 'compiler':
            (selected.candidate / 'tool.py').write_text('changed actual compiler source\n')
        elif change == 'grader':
            selected.grader.write_bytes(b'changed independent judge')
        else:
            (selected.generated / 'kernel.elf').write_bytes(b'\x7fELF-another-current-program')
        assert _read(selected, _context(selected)) is None


@pytest.mark.parametrize('field,value', [('gold', None), ('reference', None), ('policy', None),
                                       ('llvm_text', ''), ('capsule', {'kind': 'model'}),
                                       ('cb', {'outputs': []})])
def test_incomplete_semantic_inputs_miss(selected, field, value):
    selected.inputs_to_cache[field] = value
    with _scope(selected):
        assert _context(selected) is None


def test_unserializable_input_or_forged_adapter_cannot_issue_context(selected):
    with _scope(selected):
        selected.inputs_to_cache['capsule']['opaque'] = object()
        assert _context(selected) is None
        selected.inputs_to_cache['capsule'].pop('opaque')
        def forged(*args):
            return None
        OC.register_adapter(forged, engine='gsim', target=selected.target.target, selection=None, readback_policy=None)
        selected.inputs_to_cache['adapter'] = forged
        assert _context(selected) is None


def test_expired_and_cross_thread_contexts_cannot_publish_or_reuse(selected):
    with _scope(selected):
        context = _context(selected)
        _write(selected, context)
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(copy_context().run, _read, selected, context).result() is None
    assert _read(selected, context) is None


def test_changed_binary_during_lookup_is_rejected_before_returning_carried_result(selected, monkeypatch):
    with _scope(selected):
        context = _context(selected)
        _write(selected, context)
        lookup = TC.lookup

        def replaced(*args, **kwargs):
            result = lookup(*args, **kwargs)
            selected.binary.write_bytes(b'changed after lookup started')
            return result

        monkeypatch.setattr(TC, 'lookup', replaced)
        assert _read(selected, context) is None


def test_opt_in_is_required_and_nonfinite_inputs_miss(selected, monkeypatch):
    with _scope(selected):
        monkeypatch.delenv(SC.ENV)
        assert _context(selected) is None
        monkeypatch.setenv(SC.ENV, '1')
        selected.inputs_to_cache['gold']['C'][0] = float('nan')
        assert _context(selected) is None


def test_context_cannot_answer_another_capsule_target_or_tier(selected):
    with _scope(selected):
        context = _context(selected)
        programs = CR.run_executables(selected.generated)
        for expected in (('other', 'L3', selected.target.target), ('synthetic', 'L2', selected.target.target),
                         ('synthetic', 'L3', 'other')):
            assert SC.key(context, programs, expected=expected) == (None, None)


def test_fresh_source_revision_with_identical_program_and_inputs_reuses(selected):
    # Old issued authority must never survive source movement. A new original
    # execution owner may reuse the exact executable's independently earned pass.
    with _scope(selected):
        context = _context(selected)
        _write(selected, context)
    (selected.candidate / 'tool.py').write_text('changed compiler implementation, same emitted program\n')
    with _scope(selected):
        fresh = _context(selected)
        assert fresh is not None
        carried = _read(selected, fresh)
        assert carried is not None and carried.status == 'pass'
        assert carried.timing is None


def test_first_native_tier_can_carry_only_after_fresh_engine_build(selected):
    with _scope(selected):
        context = _context(selected)
        _write(selected, context)
        elf = selected.generated / 'kernel.elf'
        package_elf = selected.generated / 'package_kernel.elf'
        # Use the normal fixed capsule ELF name in both actual build receipts.
        elf.rename(package_elf)
        _write(selected, _context(selected))
        context = _context(selected)
        with pytest.raises(CR._CarriedPreparedTier) as caught:
            with CR._prepared_tier_reuse('synthetic', 'L3', True, target=selected.target.target,
                    generated=selected.generated, shas=lambda: SHAS, from_rtl=True,
                    simulation_context=context):
                CR._PREPARED_EXECUTION_OBSERVER.get()(package_elf)
        assert caught.value.result.carried and caught.value.result.timing is None
        assert CR._PREPARED_EXECUTION_OBSERVER.get() is None
        package_elf.write_bytes(b'\x7fELF-changed-current-build')
        with CR._prepared_tier_reuse('synthetic', 'L3', True, target=selected.target.target,
                generated=selected.generated, shas=lambda: SHAS, from_rtl=True,
                simulation_context=_context(selected)):
            CR._PREPARED_EXECUTION_OBSERVER.get()(package_elf)  # Miss: execute normally.


def test_relocated_targeted_capsule_reuses_only_identical_complete_source(selected):
    import shutil

    first = selected.private / "first-capsule"
    first.mkdir()
    (first / "capsule.yaml").write_text("name: synthetic\n")
    (first / "input.interface.mlir").write_text("complete original interface\n")
    second = selected.private / "second-capsule"
    shutil.copytree(first, second)
    selected.inputs_to_cache["capsule"]["__dir__"] = str(first)
    with _scope(selected):
        _write(selected, _context(selected))
        selected.inputs_to_cache["capsule"]["__dir__"] = str(second)
        assert _read(selected, _context(selected)) is not None
        (second / "input.interface.mlir").write_text("changed sidecar source\n")
        assert _read(selected, _context(selected)) is None


def test_role_declared_outputs_do_not_need_redundant_explicit_output_list(selected):
    cb = selected.inputs_to_cache["cb"]
    cb.pop("outputs")
    cb["tensors"]["C"]["role"] = "output"
    with _scope(selected):
        context = _context(selected)
        assert context is not None
        _write(selected, context)
        assert _read(selected, _context(selected)) is not None
        cb["tensors"]["C"]["role"] = "input"
        assert _context(selected) is None
