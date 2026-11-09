"""Actual native CAS helpers preserve source values and borrowing boundaries."""

import os
import subprocess

import pytest

from merlin.llvmlower.atomic_compare_exchange import MARKER, SUPPORT, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python


def source(width=32, mutation=None):
    result = f'''module {{
      func.func @forward(%p:memref<?xi{width}>,%i:index,%cmp:i{width},%value:i{width})->i{width} {{
        %v=memref.generic_atomic_rmw %p[%i]:memref<?xi{width}> {{
        ^bb0(%old:i{width}):
          %equal=arith.cmpi eq,%old,%cmp:i{width}
          %chosen=arith.select %equal,%value,%old:i{width}
          memref.atomic_yield %chosen:i{width}
        }}
        return %v:i{width}
      }}
    }}'''
    if mutation == "predicate": result = result.replace("cmpi eq", "cmpi ne")
    elif mutation == "failure": result = result.replace("%equal,%value,%old", "%equal,%value,%cmp")
    elif mutation == "inside-value": result = result.replace("%equal,%value,%old", "%equal,%old,%old")
    elif mutation == "layout": result = result.replace(f"memref<?xi{width}>", f"memref<?xi{width}, strided<[2], offset: ?>>")
    elif mutation == "space": result = result.replace(f"memref<?xi{width}>", f"memref<?xi{width}, 3>")
    elif mutation == "extra": result = result.replace("%chosen=", f"%different=arith.addi %old,%cmp:i{width}\n%chosen=")
    return result


def run(tmp_path, program, mode="positive"):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(program)
    script.write_text("from torch_mlir import ir,passmanager\nimport sys\n" + SUPPORT + r'''
with ir.Context() as ctx:
    module=ir.Module.parse(open(sys.argv[1]).read())
    module.operation.verify()
    before=str(module)
    if sys.argv[2]=='refuse':
        try:
            _cas_rewrite(ctx,module)
        except ValueError:
            assert str(module)==before
            print('REFUSED unchanged')
        else:
            raise AssertionError('invalid source region was accepted')
    else:
        assert _cas_rewrite(ctx,module)==1
        passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
        assert len(_cas_checked_calls(module))==1
        if sys.argv[2] in ('order','bytes','visibility'):
            helper=_CAS_GENERATED[0]
            if sys.argv[2]=='order':
                atomic=next(op for op in _cas_walk(helper) if op.name=='llvm.cmpxchg')
                atomic.attributes['failure_ordering']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),2)
            elif sys.argv[2]=='bytes':
                constant=next(op for op in _cas_walk(helper) if op.name=='arith.constant')
                constant.attributes['value']=ir.IntegerAttr.get(ir.IndexType.get(),16)
            else:
                helper.attributes['sym_visibility']=ir.StringAttr.get('public')
            module.operation.verify()
            assert not _cas_checked_calls(module)
            print('REFUSED current mutated helper')
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
                    raise AssertionError('mutated borrowing declaration restored')
            else:
                passmanager.PassManager.parse('builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,bufferization-lower-deallocations,canonicalize)').run(module.operation)
                _cas_restore(module)
                passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
                assert not _CAS_PENDING and len(_cas_checked_calls(module))==1
                assert sum(op.name=='llvm.cmpxchg' for op in _cas_walk(module.operation))==1
                assert not any(op.name=='memref.generic_atomic_rmw' for op in _cas_walk(module.operation))
                module.operation.verify()
                print('RESTORED exact typed helper')
''')
    process = subprocess.run([str(compiler_python()), str(script), str(original), mode], text=True,
        capture_output=True, timeout=20, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    return process.stdout


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("width", [16, 32, 64])
def test_actual_ownership_pipeline_retains_and_restores_scalar_atomic_helpers(tmp_path, width):
    assert "RESTORED exact typed helper" in run(tmp_path, source(width))


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("mutation", ["predicate", "failure", "inside-value", "layout", "space", "extra"])
def test_non_cas_regions_refuse_before_any_helper_or_source_mutation(tmp_path, mutation):
    assert "REFUSED unchanged" in run(tmp_path, source(mutation=mutation), "refuse")


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
def test_nested_symbol_tables_do_not_inherit_root_helper_binding(tmp_path):
    nested = source().removeprefix("module {").removesuffix("}")
    assert "REFUSED unchanged" in run(tmp_path, "module { module @nested {" + nested + "}}", "refuse")


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("mode", ["order", "bytes", "visibility", "declaration"])
def test_borrowing_permission_requires_current_complete_helper_and_declaration(tmp_path, mode):
    assert "REFUSED" in run(tmp_path, source(), mode)


def test_compare_exchange_has_an_explicit_unmodified_pre_bufferization_seam():
    passes = ["canonicalize", "one-shot-bufferize{bufferize-function-boundaries}", "ownership-based-buffer-deallocation"]
    assert edit_pipeline(passes) == [passes[0], MARKER, *passes[1:]]
    for invalid in ([], ["one-shot-bufferize{}"] * 2, [MARKER, "one-shot-bufferize{}"]):
        with pytest.raises(ValueError):
            edit_pipeline(invalid)
    source_ = "_run_stages(ctx,module,pipeline,0)\n"
    assert bind_runner(source_, selected=False) == source_
    assert "_CAS_CONSUMED" in bind_runner(source_, selected=True)
