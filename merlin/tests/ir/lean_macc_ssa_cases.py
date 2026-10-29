"""Small source-owned MACC graphs with shared operands, calls and dependent sites."""

from lean_bf16_f32acc_cases import bmm, matmul


def shared_operand(fmt="bf16", *, generic=False):
    if generic:
        return bmm(fmt, batch=1, m=3, n=3, depth=3).replace(
            '"linalg.generic"(%a, %b, %s)', '"linalg.generic"(%a, %a, %s)'
        )
    return matmul(fmt, m=3, n=3, depth=3).replace('"linalg.matmul"(%a, %b, %f)', '"linalg.matmul"(%a, %a, %f)')


def initialized_operand(fmt="bf16", *, generic=False, side="left"):
    source = bmm(fmt, batch=1, m=3, n=3, depth=3) if generic else matmul(fmt, m=3, n=3, depth=3)
    name, initial = ("linalg.generic", "%s") if generic else ("linalg.matmul", "%f")
    left = initial if side in ("left", "both") else "%a"
    right = initial if side in ("right", "both") else "%b"
    return source.replace(f'"{name}"(%a, %b, {initial})', f'"{name}"({left}, {right}, {initial})')


def called(fmt="bf16"):
    source = matmul(fmt, m=3, n=3, depth=3).splitlines()
    ty = f"tensor<3x3x{fmt}>"
    caller = f"""  "func.func"() <{{function_type = ({ty}, {ty}) -> {ty}, sym_name = "entry"}}> ({{
  ^bb0(%x: {ty}, %y: {ty}):
    %called = "func.call"(%x, %y) <{{callee = @subject}}> : ({ty}, {ty}) -> {ty}
    "func.return"(%called) : ({ty}) -> ()
  }}) : () -> ()"""
    return "\n".join([*source[:-1], caller, source[-1]]) + "\n"


def dependent_sites(fmt="bf16"):
    source = matmul(fmt, m=3, n=3, depth=3)
    lines = source.splitlines()
    begin = next(i for i, line in enumerate(lines) if '%z = "arith.constant"' in line)
    end = next(i for i, line in enumerate(lines) if '"func.return"' in line)
    second = "\n".join(lines[begin:end])
    for old, new in (("%z", "%z2"), ("%e", "%e2"), ("%f", "%f2"), ("%r", "%r2")):
        second = second.replace(old, new)
    second = second.replace('"linalg.matmul"(%a, %b, %f2)', '"linalg.matmul"(%r, %r, %f2)')
    tail = "\n".join(lines[end:]).replace('"func.return"(%r)', '"func.return"(%r2)')
    return "\n".join([*lines[:end], second, tail]) + "\n"
