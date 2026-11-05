"""Bounded source-preserving loaded sums use disjoint slots and real drains."""

from dataclasses import replace
import numpy as np
import pytest

from test_compact_loaded_sum import fixture
from merlin.compile.scheduling.loaded_sum_epochs import (
    LoadedSumEpochLaw, emit_loaded_sum_epochs, plan_loaded_sum_epochs,
)


def selected(rows=5, columns=11, slots=2, flatten=False):
    kernel, layout, contract, project, proof = fixture(rows, columns)
    layout = replace(layout, accumulator_capacity_bytes=8 + slots * 4 * 16)
    law = LoadedSumEpochLaw(contract.implementation_identity, contract.scalar_proof_sha256,
        'independently selected target effect contract', True, True, True, True, True, True)
    plan = plan_loaded_sum_epochs(kernel, layout, contract, project, scalar_proof=proof,
                                   target_law=law, max_live_tiles=slots, flatten_rows=flatten)
    return plan, kernel, layout, contract, project, proof


class Integers:
    def op(self, name, a, b):
        return {'add': lambda: a + b, 'mul': lambda: a * b,
                'udiv': lambda: a // b, 'urem': lambda: a % b}[name]()

    def cmp(self, predicate, a, b):
        assert predicate == 'eq'
        return a == b

    def select(self, condition, a, b):
        return a if condition else b

    def loop(self, start, stop, step, body):
        for i in range(start, stop, step):
            body(i)


@pytest.mark.parametrize('shape,slots,flatten', [((1, 1), 1, False), ((5, 7), 2, False),
    ((9, 11), 3, False), ((17, 25), 2, False), ((5, 7), 6, True), ((9, 11), 5, True)])
def test_complete_source_cover_with_reversed_pending_completion_and_partial_tiles(shape, slots, flatten):
    plan, *_ = selected(*shape, slots, flatten)
    lhs = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) % 17 - 8
    rhs = np.flip(lhs, axis=1).copy()
    output = np.full(shape, 123456, dtype=np.int32)
    pending, active, acc, covered, phases = [], set(), {}, [], []

    def lower(kind, f):
        if kind == 'complete_phase':
            assert f['phase'] == ('initialize', 'accumulate', 'store')[len(phases) % 3]
            phases.append(f['phase'])
            for finish in reversed(pending):
                finish()
            pending.clear()
            active.clear()
            return
        index = ('initialize', 'accumulate', 'store').index(kind)
        assert f['parameters'] == plan.contract.parameters[index]
        cells = {f['local_byte'] + r * f['accumulator_row_bytes'] + c * 4
                 for r in range(f['rows']) for c in range(f['columns'])}
        assert not cells & active
        active.update(cells)
        assert min(cells) >= plan.layout.accumulator_base_bytes
        assert max(cells) + 4 <= plan.layout.accumulator_capacity_bytes

        def finish():
            for r in range(f['rows']):
                for c in range(f['columns']):
                    cell = f['local_byte'] + r * f['accumulator_row_bytes'] + c * 4
                    row, col = f['row'] + r, f['column'] + c
                    if kind == 'initialize':
                        acc[cell] = lhs[row, col]
                    elif kind == 'accumulate':
                        acc[cell] += rhs[row, col]
                    else:
                        output[row, col] = acc[cell]
                        covered.append((row, col))
        pending.append(finish)

    report = emit_loaded_sum_epochs(plan, Integers(), lower)
    assert not pending
    assert len(covered) == len(set(covered)) == np.prod(shape)
    np.testing.assert_array_equal(output, lhs + rhs)
    assert len(phases) == report['complete_phase_drains']
    assert report['output_elements'] == np.prod(shape)


def test_unknown_effects_changed_source_and_excessive_private_address_refuse():
    plan, kernel, layout, contract, project, proof = selected()
    for flag in ('disjoint_loads_pending', 'disjoint_stores_pending', 'complete_device_drain',
                 'ordered_configuration_capture', 'finite_queue_progress', 'exclusive_private_state'):
        with pytest.raises(ValueError, match='premises'):
            replace(plan, target_law=replace(plan.target_law, **{flag: False}))
    with pytest.raises(ValueError, match='capacity'):
        replace(plan, live_tiles=plan.live_tiles + 1)
    with pytest.raises(ValueError, match='Boolean'):
        replace(plan, flatten_rows=1)
    with pytest.raises(ValueError):
        plan.revalidate(kernel, layout, contract, project, scalar_proof={**proof, 'changed': True})
    plan.revalidate(kernel, layout, contract, project, scalar_proof=proof)


@pytest.mark.parametrize('flatten', [False, True])
def test_large_cover_emits_counted_cfg_without_dynamic_enumeration(flatten):
    from merlin.llvmlower.host_builder import FnBuilder
    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp

    plan, *_ = selected(100001, 300001, 3, flatten)
    fn = FnBuilder([])
    control = FnBuilderControlAdapter(fn)
    effects = []

    def lower(kind, fields):
        effects.append((kind, fields))
        fn.add(llvm.InlineAsmOp('# ' + kind, '~{memory}', [], has_side_effects=True))

    report = emit_loaded_sum_epochs(plan, control, lower)
    module = ModuleOp([llvm.FuncOp('checked_epoch', llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=fn.finish())])
    module.verify()
    assert len(effects) == 12  # Three effects + drains, for full and tail families.
    assert report['output_elements'] == 100001 * 300001
    assert 'loop_annotation' in text(module, generic=True)
    assert len(text(module, generic=True)) < 30000
