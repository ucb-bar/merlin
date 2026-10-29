"""Actual scalar legalization composes with exact native ownership custody."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.arena_bind import bind_arena
from merlin.llvmlower.bounded_rne_custody import (
    BoundedRNECustody, _scalar_conversion_changes, is_original_callback, select_custody,
)
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.host_bounded_rne import bind_preparation
from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform, rewrite
from merlin.llvmlower.toolchain import clang
from test_late_quant_rne_intrinsic import source


@pytest.mark.parametrize('host_isa,combine', [('rv64gc', True), ('rv64gc', False), ('portable', False)])
@pytest.mark.parametrize('form', ['intrinsic', 'explicit_chain'])
def test_only_scalar_conversion_bytes_change(host_isa, combine, form):
    from test_late_quant_rne import SOURCE
    before = (source() if form == 'intrinsic' else SOURCE) + '''define void @effects(ptr %p) {
  store i8 7, ptr %p
  call void @borrowed(ptr %p)
  call void @free(ptr %p)
  ret void
}
declare void @borrowed(ptr)
declare void @free(ptr)
'''
    after, report = rewrite(before, host_isa=host_isa, combine_clamp=combine)
    report['combine_clamp'] = combine
    policy = dict(host_isa=host_isa, combine_clamp=combine, temporary_prefix='merlin.rne')
    assert _scalar_conversion_changes(before, after, report, **policy) == (('%result' if form == 'intrinsic' else '%v17'),)
    for old, new in [('store i8 7', 'store i8 8'), ('@borrowed(ptr %p)', '@free(ptr %p)'),
                     ('call void @free(ptr %p)', 'call void @borrowed(ptr %p)'),
                     ('ret void', 'unreachable'), ('ptr %p)', 'i64 %p)')]:
        changed = after.replace(old, new)
        # A self-consistent report is never permission for a changed effect.
        import hashlib
        forged = {**report, 'rewritten_sha256': hashlib.sha256(changed.encode()).hexdigest()}
        with pytest.raises(ValueError):
            _scalar_conversion_changes(before, changed, forged, **policy)


def test_each_function_has_an_independent_fresh_scalar_namespace():
    before = source() + source().split('declare float', 1)[0].replace('@quant(', '@other(')
    after, report = rewrite(before, host_isa='rv64gc', combine_clamp=True)
    report['combine_clamp'] = True
    assert _scalar_conversion_changes(before, after, report, host_isa='rv64gc', combine_clamp=True,
                                      temporary_prefix='merlin.rne') == ('%result', '%result')


def test_arbitrary_callback_and_late_original_policy_mutation_refuse():
    assert not is_original_callback(None)
    assert not is_original_callback(lambda source, work: source)
    with pytest.raises(ValueError, match='original bounded RNE callback'):
        BoundedRNECustody(lambda source, work: source)
    callback = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa='rv64gc', combine_clamp=True)
    assert is_original_callback(callback)
    assert not is_original_callback(lambda source, work: callback(source, work))
    owner = BoundedRNECustody(callback)
    closure = dict(zip(callback.__code__.co_freevars, callback.__closure__, strict=True))
    closure['combine_clamp'].cell_contents = False
    with pytest.raises(ValueError, match='callback or policy changed'):
        owner.revalidate()


def test_ordinary_storage_selects_original_callback_without_relaxing_other_routes():
    callback = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa='portable')
    other = lambda source, work: source
    for descriptor in (False, True):
        selected = select_custody(callback, descriptor_identity=descriptor, intermediate_storage=True)
        assert type(selected) is BoundedRNECustody and selected.callback is callback
    assert select_custody(callback, descriptor_identity=False, intermediate_storage=False) is None
    assert select_custody(other, descriptor_identity=False, intermediate_storage=True) is None
    assert select_custody(None, descriptor_identity=True, intermediate_storage=True) is None
    assert select_custody(callback, descriptor_identity=False, intermediate_storage=True, chain=object()) is None
    with pytest.raises(ValueError, match='original bounded RNE callback'):
        select_custody(other, descriptor_identity=True, intermediate_storage=True)
    with pytest.raises(ValueError, match='does not support'):
        select_custody(callback, descriptor_identity=True, intermediate_storage=True, chain=object())
    with pytest.raises(TypeError, match='explicit identity/storage'):
        select_custody(callback, descriptor_identity=False, intermediate_storage=1)


def test_actual_empty_callback_preserves_complete_llvm_and_supplies_no_scalar_effect(tmp_path):
    before = '''define void @effects(ptr %p) {
  store i8 7, ptr %p
  call void @borrowed(ptr %p)
  ret void
}
declare void @borrowed(ptr)
'''
    source_path = tmp_path / 'source.ll'
    source_path.write_text(before)
    callback = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa='rv64gc', combine_clamp=True)
    owner = BoundedRNECustody(callback)
    compilation = CompilationRecipe(tmp_path, producer=Path(__file__))
    selected, observation, emission = owner.apply(source_path, tmp_path / 'host_llvm', compilation)
    bind_preparation(compilation, tmp_path / 'host_llvm', observation)
    assert selected.read_text() == before
    assert emission.results == () and emission.scalar_calls == ()
    after, report = rewrite(before, host_isa='rv64gc', combine_clamp=True)
    report['combine_clamp'] = True
    policy = dict(host_isa='rv64gc', combine_clamp=True, temporary_prefix='merlin.rne')
    assert _scalar_conversion_changes(before, after, report, **policy) == ()
    import hashlib
    for mutant in (before.replace('store i8 7', 'store i8 8'), before.replace('ret void', 'unreachable')):
        forged = {**report, 'rewritten_sha256': hashlib.sha256(mutant.encode()).hexdigest()}
        with pytest.raises(ValueError):
            _scalar_conversion_changes(before, mutant, forged, **policy)


@pytest.mark.parametrize('bounded_conversion', [True, False])
@pytest.mark.parametrize('ownership', ['descriptor', 'borrowed'])
def test_actual_original_callback_native_lifetime_and_arena_remain_joined(tmp_path, upstream_host_tools,
                                                                        bounded_conversion, ownership):
    from test_descriptor_arena_native import selection, source as descriptor_source
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.target_data_layout import _query

    if ownership == 'descriptor':
        program = descriptor_source()
    else:
        from test_borrowed_arena_native import selection as borrowed_selection, source as borrowed_source

        program = borrowed_source(2, 3, 4)
    if bounded_conversion:
        program = program.replace('%v=arith.trunci %x:i32 to i8', '''
    %f=arith.sitofp %x:i32 to f32
    %lo=arith.constant -128.0:f32
    %hi=arith.constant 127.0:f32
    %l=arith.maximumf %f,%lo:f32
    %h=arith.minimumf %l,%hi:f32
    %rounded=math.roundeven %h:f32
    %v=arith.fptosi %rounded:f32 to i8''')
    if ownership == 'descriptor':
        identities, effects, _ = selection(tmp_path, program)
        owner_options = dict(descriptor_result_identities=identities, descriptor_result_effects=effects)
    else:
        borrowed, _ = borrowed_selection(tmp_path, program)
        owner_options = dict(borrowed_buffer_calls=borrowed)
    lowered = lower_model(program, tmp_path/'lower', targets=(), **owner_options,
        index_bits=64, data_layout=_query(clang(),('-O2',)),
        static_arena=False, hoist_static_allocs=False, features=frozenset(('lower_roundeven_to_intrinsic',)))
    callback = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa='portable')
    owner = BoundedRNECustody(callback)
    compilation = CompilationRecipe(tmp_path, producer=Path(__file__))
    selected, observation, emission = owner.apply(lowered.ll_path,tmp_path/'host_llvm',compilation)
    bind_preparation(compilation, tmp_path/'host_llvm', observation)
    fresh=BoundedRNECustody(callback);fresh.bind_input(lowered.ll_path)
    with pytest.raises(ValueError,match='actual ordinary callback invocation'):
        fresh.consume(selected,tmp_path/'host_llvm',dict(observation),compilation)
    carried = lowered.borrowed_arena_lifetime.after_bounded_rne(emission)
    original, refused = bind_arena(selected.read_text(), borrowed_lifetime=lowered.borrowed_arena_lifetime)
    if bounded_conversion:
        assert original == selected.read_text() and refused.bound == 0
    else:
        assert refused.bound == 3  # Identity retains the original live binding too.
    arena, report = bind_arena(selected.read_text(), borrowed_lifetime=carried)
    assert report.bound == 3 and report.refused == 0
    assert bool(report.borrowed_lifetime['scalar_legalizations'][0]['results']) is bounded_conversion
    ir, library = tmp_path/'arena.ll', tmp_path/'bounded_rne_arena.so'
    ir.write_text(arena)
    subprocess.run([str(clang()),'-O2','-shared','-fPIC',str(ir),str(tmp_path/'implementation.c'),'-lm',
                    '-o',str(library)],check=True,capture_output=True)
    invoke = HostModel.load(str(library), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
    a=(np.arange(6)*19-50).astype(np.int8).reshape(2,3)
    w=(np.arange(12)*7-40).astype(np.int8).reshape(3,4)
    w1=(np.arange(16)*5-35).astype(np.int8).reshape(4,4)
    first = a.astype(np.int32) @ w.astype(np.int32)
    quantized = np.clip(first, -128, 127).astype(np.int8) if bounded_conversion else first.astype(np.int8)
    expected=quantized.astype(np.int32) @ w1.astype(np.int32)+int(ownership == 'descriptor')
    guarded=np.full(16,-987654,np.int32);output=guarded[4:-4].reshape(2,4)
    invoke([(v.ctypes.data,v.shape) for v in (a,w,w1,output)])
    np.testing.assert_array_equal(output,expected)
    assert np.all(guarded[:4]==-987654) and np.all(guarded[-4:]==-987654)
    # RV64GC scalar asm has no @callee. Only the exact live emitted conversion
    # may avoid the allocator's global unknown-indirect-call refusal.
    rv_callback=merlin_host_llvm_transform(Path(clang()).resolve().parent,host_isa='rv64gc',combine_clamp=True)
    rv_owner=BoundedRNECustody(rv_callback)
    (tmp_path/'rv').mkdir()
    rv_compilation=CompilationRecipe(tmp_path/'rv',producer=Path(__file__))
    rv_selected,rv_observation,rv_emission=rv_owner.apply(lowered.ll_path,tmp_path/'rv_host_llvm',rv_compilation)
    rv_lifetime=lowered.borrowed_arena_lifetime.after_bounded_rne(rv_emission)
    _,rv_report=bind_arena(rv_selected.read_text(),borrowed_lifetime=rv_lifetime)
    assert rv_report.bound==3 and rv_report.refused==0
    assert bool(rv_report.borrowed_lifetime['noncapturing_scalar_calls']) is bounded_conversion
    changed=rv_selected.read_text().replace('ret void','call void asm sideeffect "", "~{memory}"()\n  ret void',1)
    _,unknown=bind_arena(changed,borrowed_lifetime=rv_lifetime)
    assert unknown.bound==0 and unknown.borrowed_lifetime['status']=='llvm_changed_conservative'
    # Saved data cannot reconstruct the live original emission.
    with pytest.raises(TypeError, match='live original'):
        lowered.borrowed_arena_lifetime.after_bounded_rne(emission.record())
    (tmp_path/'host_llvm/receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        bind_arena(selected.read_text(),borrowed_lifetime=carried)
