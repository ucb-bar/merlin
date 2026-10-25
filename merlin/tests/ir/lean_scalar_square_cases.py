"""Actual-pass fixtures for scalar-square validation; structural construction and rendering."""

from io import StringIO

from xdsl.dialects import arith, func, math
from xdsl.dialects.builtin import FloatAttr, ModuleOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.scalar_square import canonicalize_scalar_squares


def render(module):
    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def scalar(kind="f32", flags="none", exponent="2.0"):
    return f"""module {{
      func.func @subject(%x: {kind}, %y: {kind}) -> {kind} {{
        %two = arith.constant {exponent} : {kind}
        %result = math.powf %x, %two fastmath<{flags}> : {kind}
        return %result : {kind}
      }}
    }}"""


def nested(kind="f32"):
    return f"""module {{
      func.func @subject(%condition: i1, %x: {kind}, %y: {kind}) -> {kind} {{
        %two = arith.constant 2.0 : {kind}
        %outer = scf.if %condition -> ({kind}) {{
          %inner = scf.if %condition -> ({kind}) {{
            %a = math.powf %x, %two : {kind}
            scf.yield %a : {kind}
          }} else {{
            %b = math.powf %y, %two : {kind}
            scf.yield %b : {kind}
          }}
          scf.yield %inner : {kind}
        }} else {{
          %c = math.powf %x, %two : {kind}
          scf.yield %c : {kind}
        }}
        return %outer : {kind}
      }}
    }}"""


def actual(source):
    module = parse_mlir_text(source)
    module.verify()
    before = render(module)
    count = canonicalize_scalar_squares(module)
    module.verify()
    return before, render(module), count


def captured_probe(stage):
    """Extract a real captured eligible operation, preserving its attributes and flags."""
    module = parse_mlir_text((stage / "before.mlir").read_text())
    for original in module.walk():
        if not isinstance(original, math.PowFOp):
            continue
        owner = original.rhs.owner
        if not isinstance(owner, arith.ConstantOp) or not isinstance(owner.value, FloatAttr):
            continue
        if owner.value.value.data != 2.0:
            continue
        ty = original.result.type
        block = Block(arg_types=[ty])
        exponent = arith.ConstantOp(owner.value)
        power = math.PowFOp(block.args[0], exponent.result, original.fastmath)
        power.attributes.update(original.attributes)
        block.add_ops([exponent, power, func.ReturnOp(power.result)])
        probe = ModuleOp([func.FuncOp("subject", ([ty], [ty]), Region(block))])
        return actual(render(probe))
    raise ValueError("capture has no literal square")
