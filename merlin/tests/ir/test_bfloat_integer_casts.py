"""Native BF16 integer casts round once against an exact integer oracle."""

from bisect import bisect_left
import os
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.bfloat_elementwise import SUPPORT
from merlin.llvmlower.toolchain import available, compiler_python


def integer_representations():
    values = [(0, 0)]
    for word in range(1, 0x5f81):
        exponent, fraction = (word >> 7) & 255, word & 127
        if exponent == 0: continue
        significand, shift = 128 + fraction, exponent - 134
        if shift < 0:
            if -shift > 7 or significand % (1 << -shift): continue
            value = significand >> -shift
        else: value = significand << shift
        values.append((value, word))
    return values


def oracle(inputs, *, width, signed):
    candidates = integer_representations()
    magnitudes = [value for value, _ in candidates]
    outputs = []
    for raw in inputs:
        value = int(raw)
        if width == 1 and signed and value: value = -1
        negative = value < 0
        value = abs(value)
        index = bisect_left(magnitudes, value)
        nearby = candidates[max(0, index - 1):index + 1]
        _, word = min(nearby, key=lambda item: (abs(item[0] - value), item[1] & 1))
        outputs.append(word | (0x8000 if negative else 0))
    return np.asarray(outputs, np.uint16)


def arguments(width, signed):
    if width == 1: return np.asarray([False, True])
    if width <= 16:
        raw = np.arange(1 << width, dtype=f"uint{width}")
        return raw.view(f"int{width}") if signed else raw
    maximum = (1 << (width - int(signed))) - 1
    values = {0, 1, maximum}
    if signed: values.update((-1, -(1 << (width - 1))))
    # Every retained-significand midpoint and both adjacent source integers.
    for shift in range(1, width - 7):
        for significand in range(128, 256):
            midpoint = (significand << shift) + (1 << (shift - 1))
            for delta in (-1, 0, 1):
                value = midpoint + delta
                if value <= maximum:
                    values.add(value)
                    if signed: values.add(-value)
    return np.asarray(sorted(values), dtype=f"{'int' if signed else 'uint'}{width}")


@pytest.mark.skipif(not available(), reason="existing native compiler environment required")
@pytest.mark.parametrize("width", (1, 8, 16, 32, 64))
@pytest.mark.parametrize("signed", (False, True))
def test_native_casts_preserve_all_small_words_and_wide_integer_rounding_boundaries(tmp_path, width, signed):
    inputs = arguments(width, signed)
    np.save(tmp_path / "inputs.npy", inputs)
    operation = "sitofp" if signed else "uitofp"
    source = f'''module {{func.func @entry(%x:memref<?xi{width}>,%out:memref<?xi16>)
      attributes {{llvm.emit_c_interface}} {{
      %zero=arith.constant 0:index
      %one=arith.constant 1:index
      %n=memref.dim %x,%zero:memref<?xi{width}>
      scf.for %i=%zero to %n step %one {{
        %a=memref.load %x[%i]:memref<?xi{width}>
        %b=arith.{operation} %a {{prov.op="integer.source"}}:i{width} to bf16
        %word=arith.bitcast %b:bf16 to i16
        memref.store %word,%out[%i]:memref<?xi16>
      }}
      return
    }}}}'''
    (tmp_path / "source.mlir").write_text(source)
    runner = tmp_path / "run.py"
    runner.write_text('from torch_mlir import ir,passmanager,execution_engine,runtime\nimport ctypes,numpy as np,sys\n' + SUPPORT + '''
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    assert _bf_elementwise(module)==1 and _bf_elementwise(module)==0
    assert 'arith.extf' not in str(module) and 'arith.truncf' not in str(module)
    assert any(op.name=='arith.bitcast' and 'prov.op' in op.attributes for op in _bfe_walk(module.operation))
    passmanager.PassManager.parse('builtin.module(convert-scf-to-cf,arith-expand,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
    module.operation.verify()
    x=np.load(sys.argv[2],allow_pickle=False)
    out=np.empty(x.shape,dtype=np.uint16)
    engine=execution_engine.ExecutionEngine(module,opt_level=2)
    a,b=runtime.get_ranked_memref_descriptor(x),runtime.get_ranked_memref_descriptor(out)
    engine.invoke('entry',ctypes.pointer(ctypes.pointer(a)),ctypes.pointer(ctypes.pointer(b)))
    np.save(sys.argv[3],out)
''')
    process = subprocess.run([str(compiler_python()), str(runner), str(tmp_path / "source.mlir"), str(tmp_path / "inputs.npy"),
        str(tmp_path / "outputs.npy")], capture_output=True, timeout=30,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr.decode()
    np.testing.assert_array_equal(np.load(tmp_path / "outputs.npy", allow_pickle=False), oracle(inputs, width=width, signed=signed))


@pytest.mark.skipif(not available(), reason="existing native compiler environment required")
@pytest.mark.parametrize("change", ("wide", "tensor", "vector", "flags", "rounding", "unknown", "mixed"))
def test_invalid_integer_casts_refuse_before_any_edit(tmp_path, change):
    source_type = {"wide": "i128", "tensor": "tensor<2xi64>", "vector": "vector<2xi64>"}.get(change, "i64")
    target = {"tensor": "tensor<2xbf16>", "vector": "vector<2xbf16>"}.get(change, "bf16")
    attributes = {"flags": " {fastmath=#arith.fastmath<nnan>}",
                  "rounding": ' {roundingmode="downward"}',
                  "unknown": " {unproved=true}"}.get(change, "")
    good = '%good=arith.sitofp %x {prov.op="source.cast"}:i64 to bf16' if change == "mixed" else ""
    bad = '%r=math.log %good:bf16' if change == "mixed" else f'%r=arith.uitofp %x{attributes}:{source_type} to {target}'
    source = f'module {{func.func @f(%x:{source_type})->{target} {{{good} {bad} return %r:{target}}}}}'
    (tmp_path / "source.mlir").write_text(source)
    runner = tmp_path / "run.py"
    runner.write_text('from torch_mlir import ir\nimport sys\n' + SUPPORT + '''
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    before=str(module)
    try: _bf_elementwise(module)
    except ValueError: assert str(module)==before
    else: raise AssertionError('unqualified integer conversion accepted')
''')
    process = subprocess.run([str(compiler_python()), str(runner), str(tmp_path / "source.mlir")],
        capture_output=True, timeout=30, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert process.returncode == 0, process.stderr.decode()
