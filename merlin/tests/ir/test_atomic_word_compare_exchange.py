"""Raw floating-storage CAS retains integer ABI, exact bodies and lifetimes."""

import os
import subprocess

import pytest

from merlin.llvmlower.atomic_compare_exchange import SUPPORT
from merlin.llvmlower.toolchain import available, compiler_python

pytestmark = pytest.mark.skipif(not available(), reason="existing native MLIR required")


def source(width, variant="variable", change=None):
    constants = ""
    comparison, desired = "%comparison", "%value"
    if variant == "constant":
        constants = f"%zero=arith.constant 0:i{width}\n%one=arith.constant 1:i{width}"
        comparison, desired = "%zero", "%one"
    elif variant == "identity":
        desired = comparison
    program = f"""module {{
      func.func @forward(%p:memref<?xf{width}>,%i:index,%comparison:i{width},%value:i{width})->i{width} {{
        %old=memref.generic_atomic_rmw %p[%i]:memref<?xf{width}> {{
        ^bb0(%current:f{width}):
          {constants}
          %word=arith.bitcast %current:f{width} to i{width}
          %equal=arith.cmpi eq,%word,{comparison}:i{width}
          %selected=arith.select %equal,{desired},%word:i{width}
          %stored=arith.bitcast %selected:i{width} to f{width}
          memref.atomic_yield %stored:f{width}
        }}
        %result=arith.bitcast %old:f{width} to i{width}
        return %result:i{width}
      }}
    }}"""
    if change == "comparison": program = program.replace(f"%word,{comparison}:i{width}", f"{desired},{comparison}:i{width}")
    elif change == "failure": program = program.replace(f"{desired},%word:i{width}", f"{desired},{comparison}:i{width}")
    elif change == "inside-value": program = program.replace(f"%equal,{desired},%word", "%equal,%word,%word")
    elif change == "predicate": program = program.replace("arith.cmpi eq", "arith.cmpi slt")
    elif change == "unused": program = program.replace("%word=", f"%unused=arith.addi %comparison,%value:i{width}\n%word=")
    elif change == "unused-constant": program = program.replace("%word=", f"%unused=arith.constant 31:i{width}\n%word=")
    elif change == "layout": program = program.replace(f"memref<?xf{width}>", f"memref<?xf{width}, strided<[2], offset: ?>>")
    elif change == "space": program = program.replace(f"memref<?xf{width}>", f"memref<?xf{width}, 3>")
    return program


def run(tmp_path, program, body):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(program)
    script.write_text("from torch_mlir import ir,passmanager\nimport sys\n" + SUPPORT +
        "\nwith ir.Context() as ctx:\n    module=ir.Module.parse(open(sys.argv[1]).read())\n    module.operation.verify()\n" +
        "\n".join("    " + line for line in body.splitlines()))
    process = subprocess.run([str(compiler_python()), str(script), str(original)], capture_output=True,
        text=True, timeout=20, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("variant", ["variable", "constant", "identity"])
def test_word_cas_retains_integer_results_and_complete_native_body(tmp_path, width, canonical, variant):
    prefix = "passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)\n" if canonical else ""
    run(tmp_path, source(width, variant), prefix + """assert _cas_rewrite(ctx,module,word_views=True)==1
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
assert sum(op.name in ('llvm.cmpxchg','llvm.atomicrmw') for op in _cas_walk(module.operation))==1
for helper in _CAS_GENERATED:
    signature=ir.FunctionType(ir.TypeAttr(helper.attributes['function_type']).value)
    if any(op.name=='llvm.cmpxchg' for op in _cas_walk(helper)) or 'f16>' in str(signature.inputs[0]):
        assert str(signature.results[0]).startswith('i')
    assert not any(str(value)=='f16' for value in signature.inputs)
    assert not any(op.name in ('arith.extf','arith.truncf','llvm.fpext','llvm.fptrunc') for op in _cas_walk(helper))
_cas_suspend(module)
assert len(_CAS_PENDING)==1 and not _CAS_PENDING[0][1].regions[0].blocks
passmanager.PassManager.parse('builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,canonicalize,bufferization-lower-deallocations,canonicalize)').run(module.operation)
_cas_restore(module)
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
passmanager.PassManager.parse('builtin.module(expand-strided-metadata,lower-affine,convert-scf-to-cf,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
module.operation.verify()
""")


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("change", ["comparison", "failure", "inside-value", "predicate", "unused", "unused-constant", "layout", "space"])
def test_unqualified_word_cas_is_refused_without_partial_rewrite(tmp_path, width, change):
    run(tmp_path, source(width, change=change), """before=str(module)
try:
    _cas_rewrite(ctx,module,word_views=True)
except ValueError:
    assert str(module)==before
else:
    raise AssertionError('unqualified word CAS body accepted')
""")


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("change", ["order", "bytes", "visibility", "linkage", "return", "comparison", "declaration"])
def test_word_cas_borrowing_requires_complete_current_body_and_declaration(tmp_path, width, change):
    run(tmp_path, source(width), f"change={change!r}\n" + """assert _cas_rewrite(ctx,module,word_views=True)==1
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
helper=_CAS_GENERATED[0]
if change=='declaration':
    _cas_suspend(module)
    _CAS_PENDING[0][1].attributes['unexpected_permission']=ir.UnitAttr.get()
    try:
        _cas_restore(module)
    except ValueError:
        pass
    else:
        raise AssertionError('changed word CAS declaration restored')
else:
    if change=='order':
        atomic=next(op for op in _cas_walk(helper) if op.name=='llvm.cmpxchg')
        atomic.attributes['failure_ordering']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),2)
    elif change=='bytes':
        constant=next(op for op in _cas_walk(helper) if op.name=='arith.constant')
        constant.attributes['value']=ir.IntegerAttr.get(ir.IndexType.get(),16)
    elif change=='visibility':
        helper.attributes['sym_visibility']=ir.StringAttr.get('public')
    elif change=='linkage':
        helper.attributes['llvm.linkage']=ir.Attribute.parse('#llvm.linkage<external>')
    elif change=='return':
        returned=next(op for op in _cas_walk(helper) if op.name=='func.return')
        returned.operands[0]=helper.regions[0].blocks[0].arguments[3]
    else:
        atomic=next(op for op in _cas_walk(helper) if op.name=='llvm.cmpxchg')
        atomic.operands[1]=helper.regions[0].blocks[0].arguments[3]
    module.operation.verify()
    assert not _cas_checked_calls(module)
""")


@pytest.mark.parametrize("constant", [False, True])
def test_half_floating_assignments_refuse_during_complete_plan_validation(tmp_path, constant):
    body = "%constant=arith.constant 0x7C11:f16\nmemref.atomic_yield %constant:f16" if constant else "memref.atomic_yield %outside:f16"
    program = f"""module {{ func.func @forward(%p:memref<?xf16>,%i:index,%outside:f16)->f16 {{
      %old=memref.generic_atomic_rmw %p[%i]:memref<?xf16> {{
      ^bb0(%current:f16): {body}
      }}
      return %old:f16
    }} }}"""
    run(tmp_path, program, """before=str(module)
try:
    _cas_rewrite(ctx,module,word_views=True)
except ValueError:
    assert str(module)==before
else:
    raise AssertionError('unqualified half floating assignment accepted')
""")
