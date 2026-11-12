"""Half atomics derive arithmetic, integer transport and borrowing from bodies."""

import pytest
from test_atomic_f32_add import run


def source(dtype="f16", *, identity=False, change=None):
    exponent, fraction, quiet = (31744, 1023, 512) if dtype == "f16" else (32640, 127, 64)
    arithmetic = f"%sum = arith.addf %current,%value:{dtype}"
    if dtype == "bf16":
        arithmetic = """%lhs = arith.extf %current:bf16 to f32
          %rhs = arith.extf %value:bf16 to f32
          %wide = arith.addf %lhs,%rhs:f32
          %sum = arith.truncf %wide:f32 to bf16"""
    if identity:
        arithmetic = ""
    program = f"""module {{
      func.func @forward(%p:memref<?x{dtype}>,%i:index,%value:{dtype})->{dtype} {{
        %old=memref.generic_atomic_rmw %p[%i]:memref<?x{dtype}> {{
        ^bb0(%current:{dtype}):
          {arithmetic}
          %bits=arith.bitcast {"%current" if identity else "%sum"}:{dtype} to i16
          %exponent=arith.constant {exponent}:i16
          %fraction=arith.constant {fraction}:i16
          %quiet=arith.constant {quiet}:i16
          %zero=arith.constant 0:i16
          %field=arith.andi %bits,%exponent:i16
          %all=arith.cmpi eq,%field,%exponent:i16
          %mantissa=arith.andi %bits,%fraction:i16
          %nonzero=arith.cmpi ne,%mantissa,%zero:i16
          %nan=arith.andi %all,%nonzero:i1
          %quietword=arith.ori %bits,%quiet:i16
          %stored=arith.select %nan,%quietword,%bits:i16
          %result=arith.bitcast %stored:i16 to {dtype}
          memref.atomic_yield %result:{dtype}
        }}
        return %old:{dtype}
      }}
    }}"""
    if change == "quiet":
        program = program.replace(f"constant {quiet}:i16", f"constant {quiet // 2}:i16")
    elif change == "exponent":
        program = program.replace(f"constant {exponent}:i16", f"constant {exponent - 128}:i16")
    elif change == "extra":
        program = program.replace("%bits=", "%unused=arith.addi %i,%i:index\n%bits=")
    elif change == "fast":
        program = program.replace("%current,%value:f16", "%current,%value fastmath<fast>:f16").replace(
            "%lhs,%rhs:f32", "%lhs,%rhs fastmath<fast>:f32"
        )
    elif change == "space":
        program = program.replace(f"memref<?x{dtype}>", f"memref<?x{dtype},3>")
    elif change == "layout":
        program = program.replace(f"memref<?x{dtype}>", f"memref<?x{dtype},strided<[2], offset:?>>")
    elif change == "rounding":
        program = program.replace("%wide:f32 to bf16", "%wide {roundingmode=3:i32}:f32 to bf16")
    elif change in ("add-rounding", "nearest"):
        mode = 0 if change == "nearest" else 3
        program = program.replace("%current,%value:f16", f"%current,%value {{roundingmode={mode}:i32}}:f16").replace(
            "%lhs,%rhs:f32", f"%lhs,%rhs {{roundingmode={mode}:i32}}:f32"
        )
    return program


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("identity", [False, True])
def test_half_regions_retain_exact_word_helpers_through_ownership_and_llvm(tmp_path, dtype, canonical, identity):
    before = (
        "passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)\n" if canonical else ""
    )
    run(
        tmp_path,
        source(dtype, identity=identity),
        before
        + """assert _cas_rewrite(ctx,module,word_views=True)==1
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
helper=_CAS_GENERATED[0]
signature=ir.FunctionType(ir.TypeAttr(helper.attributes['function_type']).value)
assert str(signature.results[0])=='i16'
assert all(str(value) in ('index','i16','memref<?xf16>','memref<?xbf16>') for value in signature.inputs)
assert sum(op.name=='llvm.cmpxchg' for op in _cas_walk(helper))==1
assert sum(op.name=='llvm.load' for op in _cas_walk(helper))==1
_cas_suspend(module)
assert len(_CAS_PENDING)==1 and not _CAS_PENDING[0][1].regions[0].blocks
passmanager.PassManager.parse('builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,canonicalize,bufferization-lower-deallocations,canonicalize)').run(module.operation)
_cas_restore(module)
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
passmanager.PassManager.parse('builtin.module(arith-expand{include-bf16=true},expand-strided-metadata,lower-affine,convert-scf-to-cf,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
module.operation.verify()
assert not any(op.name.startswith(('scf.','memref.','arith.')) for op in _cas_walk(module.operation))
""",
    )


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("change", ["quiet", "exponent", "extra", "fast", "space", "layout", "add-rounding"])
def test_unqualified_half_regions_refuse_before_mutation(tmp_path, dtype, change):
    run(
        tmp_path,
        source(dtype, change=change),
        """before=str(module)
try:
    _cas_rewrite(ctx,module,word_views=True)
except ValueError:
    assert str(module)==before
else:
    raise AssertionError('unqualified half body accepted')
""",
    )


def test_changed_bfloat_rounding_refuses_before_mutation(tmp_path):
    run(
        tmp_path,
        source("bf16", change="rounding"),
        """before=str(module)
try:
    _cas_rewrite(ctx,module,word_views=True)
except ValueError:
    assert str(module)==before
else:
    raise AssertionError('changed rounding accepted')
""",
    )


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_explicit_nearest_even_keeps_the_same_numeric_contract(tmp_path, dtype):
    run(tmp_path, source(dtype, change="nearest"), "assert _cas_rewrite(ctx,module,word_views=True)==1")


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("change", [None, "order", "bytes", "quiet", "arithmetic", "return", "linkage"])
def test_half_borrowing_reconstructs_the_entire_current_helper(tmp_path, dtype, change):
    replacements = {
        "order": ("acq_rel acquire", "seq_cst acquire"),
        "bytes": ("arith.constant 2 : index", "arith.constant 4 : index"),
        "quiet": (f"arith.constant {512 if dtype == 'f16' else 64} : i16", "arith.constant 32 : i16"),
        "arithmetic": ("arith.addf", "arith.subf"),
        "return": ("func.return %old : i16", "func.return %start : i16"),
        "linkage": ("#llvm.linkage<internal>", "#llvm.linkage<external>"),
    }
    mutation = f"text=text.replace({replacements[change][0]!r},{replacements[change][1]!r})\n" if change else ""
    caller = (
        f"func.func @caller(%p:memref<?x{dtype}>,%i:index,%v:i16)->i16 {{"
        f"%r=func.call @unrelated_name(%p,%i,%v):(memref<?x{dtype}>,index,i16)->i16 return %r:i16}}}}"
    )
    run(
        tmp_path,
        source(dtype),
        f"text=_half_add_source('unrelated_name',{dtype!r})\n"
        + mutation
        + f"module=ir.Module.parse(text.rsplit('}}',1)[0]+{caller!r})\n"
        + "module.operation.verify()\n"
        + "passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)\n"
        + f"assert len(_cas_checked_calls(module))=={0 if change else 1}\n",
    )
