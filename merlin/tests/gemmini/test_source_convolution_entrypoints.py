"""Normal manifest entrypoints for a generated source-convolution compiler.

Set MERLIN_SOURCE_CONV_CANDIDATE and MERLIN_SOURCE_CONV_WITNESSES to run this
integration regression against a pinned package and extracted source witnesses.
No target code is executed. Missing generated inputs are explicitly skipped.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def package():
    value = os.environ.get("MERLIN_SOURCE_CONV_CANDIDATE")
    if not value:
        pytest.skip("requires explicit generated source-convolution package")
    return Path(value).resolve(strict=True)


@pytest.mark.parametrize("source_index", [18, 48, 73])
@pytest.mark.parametrize("entrypoint", [
    "parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm",
])
def test_actual_source_convolution_manifest_entrypoint(package, tmp_path, source_index, entrypoint):
    value = os.environ.get("MERLIN_SOURCE_CONV_WITNESSES")
    if not value:
        pytest.skip("requires explicit extracted source witnesses")
    source = Path(value).resolve(strict=True) / f"source_{source_index}.mlir"
    manifest = yaml.safe_load((package / "manifest.yaml").read_text())
    output = tmp_path / "command_buffer.json"
    substitutions = {"tool": str(package / manifest["entrypoints"]["tool"]),
                     "input_mlir": str(source), "output_json": str(output)}
    argv = [arg.format(**substitutions) for arg in manifest["commands"][entrypoint]["argv"]]
    result = subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                            timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    assert "declined" not in result.stdout.lower(), result.stdout[:1000]
    if entrypoint == "emit_command_buffer":
        cb = json.loads(output.read_text())
        assert "declined" not in cb
        assert cb["commands"], "a successful convolution must not be an empty refusal"
    if entrypoint in {"lower_interface_to_target", "emit_command_buffer"}:
        assert "gemmini.loop_ws_block" in result.stdout
        if source_index != 48:
            assert "gemmini.im2col_row" in result.stdout
        # The target dialect has custom syntax: verify a real print/parse roundtrip.
        roundtrip = subprocess.run([sys.executable, "-c", '''
import sys
from xdsl.context import Context
from xdsl.dialects import builtin, func, llvm
from xdsl.parser import Parser
from mlir_oot.ir.gemmini_dialect import GEMMINI
from mlir_oot.gemmini_opt import _print
ctx = Context()
for dialect in (builtin.Builtin, func.Func, llvm.LLVM, GEMMINI):
    ctx.load_dialect(dialect)
text = sys.stdin.read()
module = Parser(ctx, text).parse_module()
module.verify()
assert _print(module).strip() == text.strip()
'''], cwd=package, input=result.stdout, capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert roundtrip.returncode == 0, roundtrip.stderr
    if entrypoint == "lower_target_to_llvm":
        assert '"llvm.inline_asm"' in result.stdout
        assert "gemmini.loop_ws_block" not in result.stdout


def test_source_convolution_dialect_rejects_invalid_geometry(package):
    # Isolate generated module imports from any other compiler package in pytest.
    check = '''
from xdsl.dialects.builtin import IntAttr, IntegerAttr, i64
from xdsl.utils.exceptions import VerifyException
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import loop_ws, rtl_facts as F
from xdsl.ir import Block
from xdsl.dialects.llvm import LLVMPointerType

def make(cls, attrs, count, operand_type=None, results=()):
    block = Block(arg_types=[operand_type or LLVMPointerType()] * count)
    op = cls(operands=[list(block.args)], result_types=[list(results)])
    op.attributes.update({key: IntegerAttr(value, 64) for key, value in attrs.items()})
    return op

gather = dict(ci=2, hi=4, wi=4, kh=7, kw=7, wo=2, stride_h=2, stride_w=2,
              dilation_h=1, dilation_w=1, batch=0, out_y=0, pad_top=3, pad_left=3)
loop = dict(rows=2, cols=2, depth=98, a_stride=112, b_stride=16, c_stride=32,
            a_offset=0, b_offset=0, c_offset=0, d_offset=0, full_c=1, accumulate=0)
for cls, valid, count in [(G.Im2colRowOp, gather, 2), (G.LoopWsBlockOp, loop, 3)]:
    make(cls, valid, count).verify()
    for key in valid:
        bad = make(cls, valid, count)
        bad.attributes.pop(key)
        try:
            bad.verify()
        except VerifyException:
            pass
        else:
            raise AssertionError(f"missing {cls.name}.{key} accepted")
    for changes in ([dict(stride_h=0), dict(pad_top=-1), dict(ci=0)] if count == 2
                    else [dict(full_c=0), dict(accumulate=2), dict(a_offset=-1), dict(c_stride=2**40),
                          dict(b_stride=1), dict(rows=loop_ws.contract()['capacity']['max_acc_rows']['rows'] + F.DIM,
                                               cols=F.DIM, depth=F.DIM)]):
        try:
            make(cls, {**valid, **changes}, count).verify()
        except VerifyException:
            pass
        else:
            raise AssertionError(f"invalid {cls.name} geometry accepted: {changes}")
    try:
        make(cls, valid, count - 1).verify()
    except VerifyException:
        pass
    else:
        raise AssertionError("invalid operand arity accepted")
    make(cls, valid, count, operand_type=LLVMPointerType(IntAttr(0))).verify()
    for kwargs in (dict(operand_type=i64), dict(operand_type=LLVMPointerType(IntAttr(1))),
                   dict(results=(i64,))):
        try:
            make(cls, valid, count, **kwargs).verify()
        except VerifyException:
            pass
        else:
            raise AssertionError(f"invalid typed signature accepted: {kwargs}")
'''
    result = subprocess.run([sys.executable, "-c", check], cwd=package,
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source_index", [18, 48, 73])
def test_dialect_repair_preserves_existing_llvm(package, source_index):
    previous = os.environ.get("MERLIN_SOURCE_CONV_PREVIOUS")
    witnesses = os.environ.get("MERLIN_SOURCE_CONV_WITNESSES")
    if not previous or not witnesses:
        pytest.skip("requires explicit pre-repair package and extracted source witnesses")
    source = Path(witnesses).resolve(strict=True) / f"source_{source_index}.mlir"
    artifacts = []
    for root in (Path(previous).resolve(strict=True), package):
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        substitutions = {"tool": str(root / manifest["entrypoints"]["tool"]),
                         "input_mlir": str(source)}
        argv = [arg.format(**substitutions)
                for arg in manifest["commands"]["lower_target_to_llvm"]["argv"]]
        result = subprocess.run([sys.executable, *argv], capture_output=True, text=True,
            timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode == 0, result.stderr
        assert '"llvm.inline_asm"' in result.stdout
        artifacts.append(result.stdout)
    assert artifacts[0] == artifacts[1], "dialect-only repair changed executable lowering"
