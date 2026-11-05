"""Original borrowed completion reaches the ordinary native storage allocator."""

import ctypes
import hashlib
import json
import subprocess
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.common.digest import sha256_file
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.arena_bind import bind_arena
from merlin.llvmlower.borrowed_buffer_calls import (
    CHECKPOINT,
    BorrowedBufferCall,
    BorrowedBufferCalls,
    BorrowedBufferEffects,
)
from merlin.llvmlower.host_graph_policy import read_host_graph_policy
from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA
from merlin.llvmlower.host_storage_policy import prepare, recheck
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.toolchain import clang


def source(m, k, n, wide=False):
    unused = ''.join(f',%unused{i}:memref<1xi8>' for i in range(48)) if wide else ''
    return f'''module {{
func.func private @first(memref<{m}x{k}xi8>,memref<{k}x{n}xi8>,memref<{m}x{n}xi32>) attributes {{llvm.emit_c_interface}}
func.func private @second(memref<{m}x{n}xi8>,memref<{n}x{n}xi8>,memref<{m}x{n}xi32>)
 attributes {{llvm.emit_c_interface}}
func.func @forward(%a:memref<{m}x{k}xi8>,%w0:memref<{k}x{n}xi8>,%w1:memref<{n}x{n}xi8>,%out:memref<{m}x{n}xi32>{unused})
 attributes {{llvm.emit_c_interface}} {{
 %r=memref.alloc() {{alignment=64:i64}}:memref<{m}x{n}xi32>
 func.call @first(%a,%w0,%r):(memref<{m}x{k}xi8>,memref<{k}x{n}xi8>,memref<{m}x{n}xi32>)->()
 %flat=memref.collapse_shape %r [[0,1]]:memref<{m}x{n}xi32> into memref<{m*n}xi32>
 %again=memref.expand_shape %flat [[0,1]] output_shape [{m},{n}]:memref<{m*n}xi32> into memref<{m}x{n}xi32>
 %q=memref.alloc() {{alignment=64:i64}}:memref<{m}x{n}xi8>
 linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
                 iterator_types=["parallel","parallel"]}}
 ins(%again:memref<{m}x{n}xi32>) outs(%q:memref<{m}x{n}xi8>) {{
 ^bb0(%x:i32,%unused:i8):
 %v=arith.trunci %x:i32 to i8
 linalg.yield %v:i8
 }}
 %y=memref.alloc() {{alignment=64:i64}}:memref<{m}x{n}xi32>
 func.call @second(%q,%w1,%y):(memref<{m}x{n}xi8>,memref<{n}x{n}xi8>,memref<{m}x{n}xi32>)->()
 "memref.copy"(%y,%out):(memref<{m}x{n}xi32>,memref<{m}x{n}xi32>)->()
 return
}}}}'''


# The implementation never retains a borrowed alias, frees storage, or starts
# asynchronous work. Address checks use independently supplied arena addresses;
# no borrowed pointer is stored anywhere. Both calls perform actual reductions.
IMPLEMENTATION = r'''
#include <stdint.h>
#include <stdlib.h>
typedef struct {void *allocated,*aligned;int64_t offset,size[2],stride[2];} M;
static uintptr_t expected_output, expected_middle;
static int address_checks;
void expect_arena(uintptr_t output,uintptr_t middle) {
 expected_output=output; expected_middle=middle; address_checks=0;
}
int checked_addresses(void) {return address_checks;}
static void product(M*a,M*w,M*y) {
 for(int64_t i=0;i<y->size[0];i++) for(int64_t j=0;j<y->size[1];j++) {
  int32_t sum=0;
  for(int64_t k=0;k<a->size[1];k++)
   sum+=(int32_t)((int8_t*)a->aligned)[a->offset+i*a->stride[0]+k*a->stride[1]]
       *(int32_t)((int8_t*)w->aligned)[w->offset+k*w->stride[0]+j*w->stride[1]];
  ((int32_t*)y->aligned)[y->offset+i*y->stride[0]+j*y->stride[1]]=sum;
 }
}
void _mlir_ciface_first(M*a,M*w,M*y) {
 if(expected_output) {if((uintptr_t)y->aligned!=expected_output) abort(); address_checks++;}
 product(a,w,y);
}
void _mlir_ciface_second(M*a,M*w,M*y) {
 if(expected_output) {
  if((uintptr_t)y->aligned!=expected_output || (uintptr_t)a->aligned!=expected_middle
     || a->aligned==y->aligned) abort();
  address_checks++;
 }
 product(a,w,y);
}
'''


def selection(tmp_path, s):
    original, implementation = tmp_path / 'source.mlir', tmp_path / 'implementation.c'
    original.write_text(s)
    implementation.write_text(IMPLEMENTATION)
    digest = sha256_file(original)
    effects = BorrowedBufferEffects('pinned synchronous neutral C reductions; no retained aliases or releases',
                                    True, True, True)
    import re

    arguments = re.findall(r'func.func private @(first|second)\(([^)]*)\)', s)
    calls = tuple(BorrowedBufferCall(name + '_writer', name, tuple(types.split(',')), 2, effects)
                  for name, types in arguments)
    state = {'valid': True, 'validations': 0}

    def validate():
        state['validations'] += 1
        if not state['valid']:
            raise ValueError('original implementation is no longer qualified')

    return BorrowedBufferCalls(original, digest, digest, calls,
                               ((implementation, sha256_file(implementation)),), validate), state


def storage_policy(tmp_path):
    software = tmp_path / 'software.yaml'
    software.write_text(yaml.safe_dump({
        'schema': 'merlin.software_spec.v1', 'status': 'reviewed', 'target': 'neutral',
        'host_optimization': {'schema': POLICY_SCHEMA, 'permitted_stages': ['llvm'],
                              'permitted_optimizations': ['static_arena'], 'sequential_execution': True,
                              'successful_allocations': True, 'non_reentrant_entry': True}}))
    return read_host_graph_policy(software, target='neutral', entry='forward')


@pytest.mark.parametrize('m,k,n,wide', [(2, 3, 4, False), (3, 3, 5, False), (2, 3, 4, True)])
def test_current_native_effects_enable_real_connected_storage_reuse(tmp_path, upstream_host_tools, m, k, n, wide):
    s = source(m, k, n, wide)
    original, state = selection(tmp_path, s)
    result = lower_model(s, tmp_path / 'lower', targets=(), borrowed_buffer_calls=original,
                         static_arena=False, hoist_static_allocs=False)
    llvm = result.ll_path.read_text()
    # The original exact lowered ABI is conservative without its live effects.
    unchanged, refused = bind_arena(llvm)
    assert unchanged == llvm and refused.refusals == {'raw_pointer_escapes': 3}
    selected_policy = storage_policy(tmp_path)
    pins = {}
    validations_before = state['validations']
    selected = prepare(result.ll_path, selected_policy, tmp_path / 'storage',
                       SimpleNamespace(bind_preparation=lambda role, path: pins.update({role: path})),
                       borrowed_lifetime=result.borrowed_arena_lifetime)
    assert state['validations'] - validations_before == 2  # Per invocation, never per allocation.
    report = json.loads((tmp_path / 'storage/report.json').read_text())
    arena = report['arena']
    assert arena['bound'] == 3 and arena['refused'] == 0 and arena['reuse_factor'] == 1.5
    assert arena['borrowed_lifetime']['status'] == 'checked'
    assert result.stats['ciface_boundary_inline']['count'] == int(wide)
    if wide:
        assert arena['borrowed_lifetime']['abi_annotations'][0]['functions'] == ['forward']
        assert arena['borrowed_lifetime']['abi_annotations'][0]['bodies_calls_memory'] == 'unchanged'
    assert len(pins) == 5
    first, middle, last = arena['placements']
    assert first['offset'] == last['offset'] != middle['offset']
    assert first['release_line'] < last['allocation_line']
    assert middle['release_line'] > last['allocation_line']
    assert selected.read_text().count('call void @free(') == 0
    assert selected.read_text().count('getelementptr inbounds i8, ptr @merlin_arena') == 3
    recheck(tmp_path / 'storage', selected_policy, borrowed_lifetime=result.borrowed_arena_lifetime)
    with pytest.raises(ValueError, match='changed'):
        recheck(tmp_path / 'storage', selected_policy)  # Receipt is not a grant.

    # The ordinary owned-host composition auditor receives the same live owner
    # through the build receipt, while JSON/copies carry only report data.
    from merlin.llvmlower.borrowed_arena import receipt_borrowed_lifetime, retain_borrowed_lifetime
    from merlin.targetgen import source_composition_preferences

    root = Path(__file__).resolve().parents[3]
    assert Path(source_composition_preferences.__file__).is_relative_to(root / 'packages/merlin-experiments/src')
    artifacts = recheck(tmp_path / 'storage', selected_policy, borrowed_lifetime=result.borrowed_arena_lifetime)
    receipt = retain_borrowed_lifetime({
        'inputs': {'device': {'intermediate_storage': selected_policy.record()}},
        'output': {'intermediate_storage': artifacts},
    }, result.borrowed_arena_lifetime)
    options = {'intermediate_storage': selected_policy}
    source_composition_preferences.require_host_preparation(receipt, options)
    source_composition_preferences.require_host_preparation(receipt, options)
    serialized = json.dumps(receipt, sort_keys=True)
    assert serialized == json.dumps(dict(receipt), sort_keys=True)
    assert '_lifetime' not in serialized and 'BorrowedArenaLifetime' not in serialized

    @dataclass
    class Holder:
        receipt: dict

    copies = (json.loads(serialized), dict(receipt), copy(receipt), deepcopy(receipt),
              asdict(Holder(receipt))['receipt'])
    for saved in copies:
        assert receipt_borrowed_lifetime(saved) is None
        with pytest.raises(ValueError, match='changed'):
            source_composition_preferences.require_host_preparation(saved, options)

    # Native execution uses precisely the emitted arena and checks actual output
    # addresses during both borrowed calls. The escaping output remains caller-owned.
    for label, path in [('heap', result.ll_path), ('arena', selected)]:
        execution = tmp_path / f'{label}.ll'
        getter = ('getelementptr (i8, ptr @merlin_arena, i64 0)' if label == 'arena' else 'null')
        execution.write_text(path.read_text() + '\ndefine ptr @arena_address() {\n  ret ptr ' + getter + '\n}\n')
        library = tmp_path / f'{label}_{m}_{k}_{n}_{wide}.so'
        proc = subprocess.run([str(clang()), '-O2', '-shared', '-fPIC', str(execution),
                               str(tmp_path / 'implementation.c'), '-o', str(library)],
                              capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        model, invoke = ctypes.CDLL(str(library)), HostModel.load(str(library))
        model.arena_address.restype = ctypes.c_void_p
        model.expect_arena.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        base = model.arena_address() or 0
        model.expect_arena(base + first['offset'] if base else 0, base + middle['offset'] if base else 0)
        for seed in (0, 1, 7):
            a = ((np.arange(m*k).reshape(m, k) * 19 + seed) % 251 - 125).astype(np.int8)
            w0 = ((np.arange(k*n).reshape(k, n) * 7 + seed) % 31 - 15).astype(np.int8)
            w1 = ((np.arange(n*n).reshape(n, n) * 11 + seed) % 23 - 11).astype(np.int8)
            out = np.full((m, n), -1234567, dtype=np.int32)
            expected = (a.astype(np.int32) @ w0.astype(np.int32)).astype(np.int8).astype(np.int32) @ w1.astype(np.int32)
            unused = [np.array([73], dtype=np.int8) for _ in range(48)] if wide else []
            invoke([(v.ctypes.data, v.shape) for v in (a, w0, w1, out, *unused)])
            np.testing.assert_array_equal(out, expected)
            assert all(v[0] == 73 for v in unused)
        assert model.checked_addresses() == (6 if base else 0)
    assert sha256_file(tmp_path / 'source.mlir') == hashlib.sha256(s.encode()).hexdigest()

    # Equal names in modified LLVM, even an extra whitespace, cannot borrow the
    # native grant. Async/noescape changes in the original owner cannot replay it.
    changed = llvm + '\n'
    out, stale = bind_arena(changed, borrowed_lifetime=result.borrowed_arena_lifetime)
    assert out == changed and stale.bound == 0
    assert stale.borrowed_lifetime['status'] == 'llvm_changed_conservative'
    state['valid'] = False
    with pytest.raises(ValueError, match='no longer qualified'):
        bind_arena(llvm, borrowed_lifetime=result.borrowed_arena_lifetime)
    state['valid'] = True
    implementation = tmp_path / 'implementation.c'
    implementation.write_text(IMPLEMENTATION + '\n')
    with pytest.raises(ValueError, match='source or contracts changed'):
        bind_arena(llvm, borrowed_lifetime=result.borrowed_arena_lifetime)
    implementation.write_text(IMPLEMENTATION)
    (result.workdir / CHECKPOINT).write_text('changed')
    with pytest.raises(ValueError, match='native observation changed'):
        bind_arena(llvm, borrowed_lifetime=result.borrowed_arena_lifetime)


def test_effects_or_serialized_receipts_do_not_establish_completion():
    with pytest.raises(TypeError, match='live native binding'):
        bind_arena('not LLVM', borrowed_lifetime={'accepted_symbols': ['first', 'second']})
    with pytest.raises(ValueError, match='synchronous effects'):
        BorrowedBufferEffects('asynchronous producer', True, True, False)


def test_another_unqualified_call_keeps_the_entry_conservative(tmp_path, upstream_host_tools):
    s = source(2, 3, 4).replace('func.func @forward', 'func.func private @opaque(i32)\nfunc.func @forward')
    s = s.replace(' %r=memref.alloc()',
                  ' %z=arith.constant 0:i32\n func.call @opaque(%z):(i32)->()\n %r=memref.alloc()')
    original, _ = selection(tmp_path, s)
    result = lower_model(s, tmp_path / 'lower', targets=(), borrowed_buffer_calls=original,
                         static_arena=True, hoist_static_allocs=False)
    assert result.stats['static_arena']['borrowed_lifetime']['status'] == 'checked'
    assert result.stats['static_arena']['bound'] == 0
    assert result.stats['static_arena']['refusals'] == {'raw_pointer_escapes': 3}
