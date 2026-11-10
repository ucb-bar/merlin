"""Native scalar diagnostics keep values, literal bytes and runtime custody."""

import os
import subprocess

import pytest

from merlin.llvmlower.scalar_diagnostics import MARKER,SUPPORT,bind_runner,edit_pipeline
from merlin.llvmlower.length_aware_diagnostics import MARKER as ASSERT_MARKER
from merlin.llvmlower.toolchain import available,compiler_python
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


@pytest.mark.parametrize("source",[
    r'''module { func.func @f(%n:i64) {"vector.print"(%n) <{punctuation=#vector.punctuation<no_punctuation>}>:(i64)->() func.return}}''',
    r'''module {func.func @f() {"vector.print"() <{stringLiteral="first\00\CE\BB"}>:()->() func.return}}''',
    r'''module {func.func @f() {"vector.print"() <{punctuation=#vector.punctuation<comma>}>:()->() func.return}}''',
])
def test_standard_binding_roundtrips_complete_fields(source):
    module = parse_mlir_text(source)
    module.verify()
    again = parse_mlir_text(text(module,generic=True))
    again.verify()
    assert all(op.name != "builtin.unregistered" for op in again.walk())


@pytest.mark.parametrize("change",["value","punctuation","tensor"])
def test_standard_binding_rejects_incompatible_fields(change):
    fields = {
        "value":r'''"vector.print"(%n) <{stringLiteral="text"}>:(i64)->()''',
        "punctuation":r'''"vector.print"() <{stringLiteral="text",punctuation=#vector.punctuation<comma>}>:()->()''',
        "tensor":r'''"vector.print"(%t):(tensor<2xf32>)->()''',
    }
    with pytest.raises(Exception,match="string print|numeric scalar"):
        parse_mlir_text('module {func.func @f(%n:i64,%t:tensor<2xf32>) {'+fields[change]+' func.return}}').verify()


@pytest.mark.skipif(not available(),reason="existing native compiler required")
@pytest.mark.parametrize("change",["positive","printf-type","putchar-body","early","vector","half","collision"])
def test_runtime_literals_values_and_unchanged_refusals(tmp_path,change):
    declarations = {
        "printf-type":"llvm.func @printf(i64)->i32",
        "putchar-body":"llvm.func @putchar(%c:i32)->i32 {llvm.return %c:i32}",
        "collision":"llvm.func @__merlin_print_bytes_0() {llvm.return}",
    }
    selected_type = {"vector":"vector<2xf32>","half":"f16"}.get(change,"i64")
    source = r'''module {
      func.func @entry(%n:TYPE,%v:f64) attributes {llvm.emit_c_interface} {
        vector.print str "prefix\00\CE\BB%s "
        vector.print %n:TYPE punctuation <no_punctuation>
        vector.print str " "
        vector.print %v:f64
        func.return
      }
    DECLARATIONS }'''.replace("TYPE",selected_type).replace("DECLARATIONS",declarations.get(change,""))
    program,runner = tmp_path / "source.mlir",tmp_path / "run.py"
    program.write_text(source)
    runner.write_text("from torch_mlir import ir,passmanager,execution_engine\nimport ctypes,sys,resource\n" + SUPPORT + '''
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    if sys.argv[2]!='early':
        passmanager.PassManager.parse('builtin.module(convert-func-to-llvm,convert-arith-to-llvm)').run(module.operation)
    before=str(module)
    if sys.argv[2] in ('positive','collision'):
        assert _scalar_diag_lower(module)=={'prints':4,'assertions':0}
        module.operation.verify()
        assert not any(op.name=='vector.print' for op in _diag_walk(module.operation))
        if sys.argv[2]=='collision':
            assert '@__merlin_print_bytes_1' in str(module)
        passmanager.PassManager.parse('builtin.module(convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
        engine=execution_engine.ExecutionEngine(module,opt_level=0)
        libc=ctypes.CDLL(None)
        libc.setbuf.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        libc.setbuf(ctypes.c_void_p.in_dll(libc,'stdout'),None)
        n,v=ctypes.c_longlong(-(1<<63)),ctypes.c_double(1.125)
        engine.invoke('entry',ctypes.pointer(n),ctypes.pointer(v))
    else:
        try:
            _scalar_diag_lower(module)
        except ValueError:
            assert str(module)==before
            print('REFUSED unchanged')
        else:
            raise AssertionError('incompatible source/runtime accepted')
''')
    result = subprocess.run([str(compiler_python()),str(runner),str(program),change],capture_output=True,
        timeout=30,env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0,result.stdout+result.stderr
    assert result.stdout == ("prefix\x00λ%s -9223372036854775808 1.125000\n".encode("utf-8")
        if change in ("positive","collision") else b"REFUSED unchanged\n")


@pytest.mark.parametrize("passes",[[],["convert-cf-to-llvm"]*2,[ASSERT_MARKER,"convert-cf-to-llvm"],[MARKER,"convert-cf-to-llvm"]])
def test_one_unmodified_native_seam(passes):
    with pytest.raises(ValueError):
        edit_pipeline(passes)


def test_explicit_pipeline_and_runner():
    assert edit_pipeline(["convert-arith-to-llvm","convert-cf-to-llvm"]) == ["convert-arith-to-llvm",MARKER,"convert-cf-to-llvm"]
    source = "_run_stages(ctx,module,pipeline,0)\n"
    assert bind_runner(source,selected=False) == source
    assert "_scalar_diag_lower(module)" in bind_runner(source,selected=True)
