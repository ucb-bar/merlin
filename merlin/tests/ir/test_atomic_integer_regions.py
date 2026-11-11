"""Pure integer atomics retain complete helpers through ownership and LLVM."""

import os
import subprocess

import pytest

from merlin.llvmlower.atomic_compare_exchange import SUPPORT
from merlin.llvmlower.toolchain import available, compiler_python


def source(width=32, change=None):
    value = f"""module {{
      func.func @forward(%raw:memref<?xi8>,%i:index,%value:i{width},%flag:i1)->i{width} {{
        %zero=arith.constant 0:index
        %bytes=arith.constant {width // 8}:index
        %length=memref.dim %raw,%zero:memref<?xi8>
        %count=arith.divui %length,%bytes:index
        %view=memref.view %raw[%zero][%count]:memref<?xi8> to memref<?xi{width}>
        %old=memref.generic_atomic_rmw %view[%i]:memref<?xi{width}> {{
        ^bb0(%current:i{width}):
          %mask=arith.constant 255:i{width}
          %shift=arith.constant 8:i{width}
          %sum=arith.addi %current,%value:i{width}
          %low=arith.andi %sum,%mask:i{width}
          %shifted=arith.shli %low,%shift:i{width}
          %chosen=arith.select %flag,%sum,%shifted:i{width}
          memref.atomic_yield %chosen:i{width}
        }}
        return %old:i{width}
      }}
    }}"""
    if change == "extra":
        value = value.replace("%chosen=", f"%unused=arith.xori %current,%value:i{width}\n%chosen=")
    if change == "float":
        value = value.replace("%chosen=", f"%asfloat=arith.bitcast %current:i{width} to f{width}\n%chosen=")
    if change == "effect":
        value = value.replace("%chosen=", f"memref.store %current,%view[%i]:memref<?xi{width}>\n%chosen=")
    if change == "space":
        value = value.replace(f"memref<?xi{width}>", f"memref<?xi{width},3>")
        value = value.replace("memref<?xi8>", "memref<?xi8,3>")
    return value


def run(tmp_path, width, mode="positive", change=None):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(source(width, change))
    script.write_text(
        "from torch_mlir import ir,passmanager\nimport sys\n"
        + SUPPORT
        + r"""
with ir.Context() as ctx:
    module=ir.Module.parse(open(sys.argv[1]).read())
    module.operation.verify()
    before=str(module)
    if sys.argv[2] in ('refuse','legacy'):
        try:
            _cas_rewrite(ctx,module,integer_regions=sys.argv[2]!='legacy')
        except ValueError:
            assert str(module)==before
            print('REFUSED unchanged')
        else:
            raise AssertionError('unqualified integer region was accepted')
    else:
        assert _cas_rewrite(ctx,module,integer_regions=True)==1
        passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
        assert len(_cas_checked_calls(module))==1
        if sys.argv[2] in ('order','stride','linkage','callee'):
            helper=_CAS_GENERATED[-1]
            if sys.argv[2]=='order':
                op=next(op for op in _cas_walk(helper) if op.name=='llvm.cmpxchg')
                op.attributes['failure_ordering']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),2)
            elif sys.argv[2]=='stride':
                op=next(op for op in _cas_walk(helper)
                        if op.name=='arith.constant' and str(op.results[0].type)=='index')
                op.attributes['value']=ir.IntegerAttr.get(ir.IndexType.get(),16)
            elif sys.argv[2]=='linkage':
                helper.attributes['llvm.linkage']=ir.Attribute.parse('#llvm.linkage<external>')
            else:
                update=_CAS_GENERATED[0]
                update.attributes['sym_visibility']=ir.StringAttr.get('public')
            assert not _cas_checked_calls(module)
            print('REFUSED changed current helper/callee')
        else:
            _cas_suspend(module)
            assert len(_CAS_PENDING)==2
            passmanager.PassManager.parse('builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,bufferization-lower-deallocations,canonicalize)').run(module.operation)
            _cas_restore(module)
            passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
            assert len(_cas_checked_calls(module))==1
            assert sum(op.name=='llvm.cmpxchg' for op in _cas_walk(module.operation))==1
            assert sum(op.name=='llvm.load' for op in _cas_walk(module.operation))==1
            passmanager.PassManager.parse('builtin.module(expand-strided-metadata,lower-affine,convert-scf-to-cf,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
            module.operation.verify()
            print('RESTORED typed integer region')
"""
    )
    process = subprocess.run(
        [str(compiler_python()), str(script), str(original), mode],
        text=True,
        capture_output=True,
        timeout=20,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)),
    )
    if change == "effect":
        assert process.returncode != 0
        assert "should contain only operations with no side effects" in process.stderr
        return "REFUSED by native verification before rewrite"
    assert process.returncode == 0, process.stdout + process.stderr
    return process.stdout


pytestmark = pytest.mark.skipif(not available(), reason="existing native MLIR required")


@pytest.mark.parametrize("width", (16, 32, 64))
def test_byte_views_and_scalar_updates_survive_real_ownership_and_llvm(tmp_path, width):
    assert "RESTORED typed integer region" in run(tmp_path, width)


@pytest.mark.parametrize("change", ("extra", "float", "effect", "space"))
def test_regions_outside_the_integer_profile_refuse_before_mutation(tmp_path, change):
    assert "REFUSED" in run(tmp_path, 32, "refuse", change)


@pytest.mark.parametrize("mode", ("order", "stride", "linkage", "callee"))
def test_borrowing_reopens_both_current_helper_and_scalar_callee(tmp_path, mode):
    assert "REFUSED changed" in run(tmp_path, 32, mode)


def test_legacy_cas_selection_keeps_its_closed_profile(tmp_path):
    assert "REFUSED unchanged" in run(tmp_path, 32, "legacy")


def test_selected_atomic_features_share_one_bufferization_seam():
    import itertools

    from merlin.llvmlower.atomic_compare_exchange import edit_pipeline as cas
    from merlin.llvmlower.atomic_integer_regions import MARKER
    from merlin.llvmlower.atomic_integer_regions import edit_pipeline as regions
    from merlin.llvmlower.atomic_word_views import edit_pipeline as words

    for order in itertools.permutations((cas, regions, words)):
        passes = ["canonicalize", "one-shot-bufferize{bufferize-function-boundaries}", "buffer-deallocation-pipeline"]
        for edit in order:
            passes = edit(passes)
        assert passes == [
            "canonicalize",
            MARKER,
            "one-shot-bufferize{bufferize-function-boundaries}",
            "buffer-deallocation-pipeline",
        ]
