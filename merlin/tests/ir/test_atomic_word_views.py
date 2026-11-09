"""Exact native word atomics retain floating storage and borrowing boundaries."""

import os
import subprocess

import pytest

from merlin.llvmlower.atomic_compare_exchange import SUPPORT, bind_runner
from merlin.llvmlower.atomic_word_views import MARKER, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python

OPS = {"add": "addi", "and": "andi", "or": "ori", "xor": "xori", "max": "maxsi", "min": "minsi", "umax": "maxui", "umin": "minui", "xchg": None}


def source(width=32, kind="max", change=None):
    update = f'''%view=arith.bitcast %current:f{width} to i{width}
      %updated=arith.{OPS[kind]} %view,%value:i{width}''' if OPS[kind] else ""
    value = "%updated" if OPS[kind] else "%value"
    result = f'''module {{
      func.func @forward(%p:memref<?xf{width}>,%i:index,%value:i{width})->f{width} {{
        %old=memref.generic_atomic_rmw %p[%i]:memref<?xf{width}> {{
        ^bb0(%current:f{width}):
          {update}
          %stored=arith.bitcast {value}:i{width} to f{width}
          memref.atomic_yield %stored:f{width}
        }}
        return %old:f{width}
      }}
    }}'''
    if change == "layout": result = result.replace(f"memref<?xf{width}>", f"memref<?xf{width}, strided<[2], offset: ?>>")
    elif change == "space": result = result.replace(f"memref<?xf{width}>", f"memref<?xf{width}, 3>")
    elif change == "extra": result = result.replace("%stored=", f"%unused=arith.addi %value,%value:i{width}\n%stored=")
    elif change == "inside-value": result = result.replace("%view,%value", "%view,%view")
    elif change == "wrong-current": result = result.replace("%view,%value", "%value,%value")
    elif change == "data-layout": result = result.replace("module {", 'module attributes {llvm.data_layout="e-p:32:32"} {')
    elif change == "overflow": result = result.replace(f"%view,%value:i{width}", f"%view,%value overflow<nsw>:i{width}")
    return result


def run(tmp_path, program, mode="positive", word_views=True):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(program)
    script.write_text("from torch_mlir import ir,passmanager\nimport sys\n" + SUPPORT + r'''
with ir.Context() as ctx:
    module=ir.Module.parse(open(sys.argv[1]).read())
    module.operation.verify()
    before=str(module)
    if sys.argv[2]=='refuse':
        try:
            _cas_rewrite(ctx,module,word_views=sys.argv[3]=='True')
        except ValueError:
            assert str(module)==before
            print('REFUSED unchanged')
        else:
            raise AssertionError('invalid word region was accepted')
    else:
        assert _cas_rewrite(ctx,module,word_views=True)==1
        passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
        assert len(_cas_checked_calls(module))==1
        if sys.argv[2] in ('order','bytes','visibility','linkage','return-value'):
            helper=_CAS_GENERATED[0]
            if sys.argv[2]=='order':
                atomic=next(op for op in _cas_walk(helper) if op.name=='llvm.atomicrmw')
                atomic.attributes['ordering']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),2)
            elif sys.argv[2]=='bytes':
                constant=next(op for op in _cas_walk(helper) if op.name=='arith.constant')
                constant.attributes['value']=ir.IntegerAttr.get(ir.IndexType.get(),16)
            elif sys.argv[2]=='visibility':
                helper.attributes['sym_visibility']=ir.StringAttr.get('public')
            elif sys.argv[2]=='linkage':
                helper.attributes['llvm.linkage']=ir.Attribute.parse('#llvm.linkage<external>')
            else:
                cast=next(op for op in _cas_walk(helper) if op.name=='arith.bitcast')
                cast.operands[0]=helper.regions[0].blocks[0].arguments[2]
            module.operation.verify()
            assert not _cas_checked_calls(module)
            print('REFUSED changed helper')
        else:
            _cas_suspend(module)
            assert len(_CAS_PENDING)==1 and not _CAS_PENDING[0][1].regions[0].blocks
            if sys.argv[2]=='declaration':
                _CAS_PENDING[0][1].attributes['unexpected_permission']=ir.UnitAttr.get()
                try:
                    _cas_restore(module)
                except ValueError:
                    print('REFUSED changed declaration')
                else:
                    raise AssertionError('mutated word declaration restored')
            else:
                passmanager.PassManager.parse('builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,bufferization-lower-deallocations,canonicalize)').run(module.operation)
                _cas_restore(module)
                passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
                assert not _CAS_PENDING and len(_cas_checked_calls(module))==1
                assert sum(op.name=='llvm.atomicrmw' for op in _cas_walk(module.operation))==1
                assert not any(op.name=='memref.generic_atomic_rmw' for op in _cas_walk(module.operation))
                passmanager.PassManager.parse('builtin.module(expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,reconcile-unrealized-casts)').run(module.operation)
                module.operation.verify()
                helpers=[op.operation for op in module.body.operations if op.operation.name=='llvm.func'
                    and ir.StringAttr(op.attributes['sym_name']).value.startswith('__merlin_atomic_word_')]
                assert len(helpers)==1 and str(helpers[0].attributes['linkage'])=='#llvm.linkage<internal>'
                print('RESTORED exact word helper')
''')
    process = subprocess.run([str(compiler_python()), str(script), str(original), mode, str(word_views)], text=True,
        capture_output=True, timeout=20, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    return process.stdout


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("kind", list(OPS))
def test_word_helpers_retain_one_atomic_through_ownership_and_llvm(tmp_path, width, kind):
    assert "RESTORED exact word helper" in run(tmp_path, source(width, kind))


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("change", ["layout", "space", "extra", "inside-value", "wrong-current", "data-layout", "overflow"])
def test_invalid_word_regions_refuse_without_source_mutation(tmp_path, change):
    assert "REFUSED unchanged" in run(tmp_path, source(kind="add", change=change), "refuse")


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("mode", ["order", "bytes", "visibility", "linkage", "return-value", "declaration"])
def test_word_borrowing_rechecks_current_body_and_declaration(tmp_path, mode):
    assert "REFUSED" in run(tmp_path, source(), mode)


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
def test_integer_cas_feature_does_not_implicitly_admit_word_regions(tmp_path):
    assert "REFUSED unchanged" in run(tmp_path, source(), "refuse", word_views=False)


def test_word_views_have_an_explicit_native_feature_and_marker():
    pipeline = edit_pipeline(["canonicalize", "one-shot-bufferize{bufferize-function-boundaries}", "ownership-based-buffer-deallocation"])
    assert pipeline[1] == MARKER
    runner = "def _run_stages(*args): pass\n_run_stages(ctx, module, pipeline, erase)\n"
    bound = bind_runner(runner, selected=True, word_views=True)
    assert "word_views=True" in bound and MARKER in bound
    compile(bound, "<word-runner>", "exec")


def test_word_and_integer_atomic_features_compose_at_one_bufferization_seam():
    from merlin.llvmlower.atomic_compare_exchange import edit_pipeline as cas_pipeline
    original = ["one-shot-bufferize{bufferize-function-boundaries}", "ownership-based-buffer-deallocation"]
    assert edit_pipeline(cas_pipeline(original)) == cas_pipeline(edit_pipeline(original))
    assert edit_pipeline(cas_pipeline(original)).count(MARKER) == 1


def test_loop_destination_runner_preserves_the_word_atomic_seam():
    from merlin.llvmlower.loop_tensor_destinations import PRELUDE as LOOP_PRELUDE, MARKER as LOOP_MARKER
    observed = []
    namespace = {"_run_stages": lambda *args: observed.append(args[2])}
    exec(LOOP_PRELUDE, namespace)
    namespace["_ltd_reuse"] = lambda *args: 0
    tail = MARKER + ",one-shot-bufferize{bufferize-function-boundaries},ownership-based-buffer-deallocation"
    namespace["_run_stages"](None, None, LOOP_MARKER + "," + tail, 0)
    assert observed == [tail]
