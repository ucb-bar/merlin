"""Closed source proofs for signed i32 saturation and literal low-bit i8 XOR."""
from merlin.frontends.linalg_patterns import StaticPointwisePattern
from merlin.transforms.dense_ir import InvalidLinalgPattern, checked_static_linalg_shell

SCHEMA = 'merlin.static_integer_seam_body.v1'
TYPES = {'saturating_i32_to_i8': ('i32', 'i8', 'i8'), 'xor_low_bit_i8': ('i8', 'i8', 'i8')}


def validate(declaration):
    if (not isinstance(declaration, dict) or set(declaration) != {'schema', 'operation'}
            or declaration.get('schema') != SCHEMA or declaration.get('operation') not in TYPES):
        raise InvalidLinalgPattern('integer seam needs a closed typed declaration')
    return dict(declaration)


def recognize(op):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.ops import YieldOp
    shell = checked_static_linalg_shell(op)
    body, args = shell.body, shell.args
    if len(op.inputs) != 1 or any(d <= 0 for d in shell.shape):
        raise InvalidLinalgPattern('integer seam requires one positive static identity input')

    def check(inner, cls, inputs, dtype):
        if (type(inner) is not cls or tuple(inner.operands) != tuple(inputs)
                or len(inner.results) != 1 or str(inner.results[0].type) != dtype
                or inner.properties or inner.attributes or inner.regions or inner.successors):
            raise InvalidLinalgPattern('integer seam scalar SSA/type/effects differ')

    def literal(value, integer, dtype):
        constant = value.owner
        if (type(constant) is not arith.ConstantOp or constant.operands or len(constant.results) != 1
                or constant.results[0] is not value or str(value.type) != dtype
                or set(constant.properties) != {'value'} or constant.attributes
                or constant.regions or constant.successors):
            raise InvalidLinalgPattern('integer seam literal is not a closed typed constant')
        attribute = constant.properties['value']
        if type(attribute) is not IntegerAttr or attribute.value.data != integer or str(attribute.type) != dtype:
            raise InvalidLinalgPattern('integer seam literal differs')

    if len(body) == 4 and shell.ordered_types == TYPES['saturating_i32_to_i8']:
        lo, hi, narrow, yld = body
        if len(lo.operands) != 2 or len(hi.operands) != 2:
            raise InvalidLinalgPattern('integer clamp arity differs')
        literal(lo.operands[1], -128, 'i32'); literal(hi.operands[1], 127, 'i32')
        check(lo, arith.MaxSIOp, (args[0], lo.operands[1]), 'i32')
        check(hi, arith.MinSIOp, (lo.results[0], hi.operands[1]), 'i32')
        check(narrow, arith.TruncIOp, (hi.results[0],), 'i8')
        result, operation = narrow.results[0], 'saturating_i32_to_i8'
    elif len(body) == 2 and shell.ordered_types == TYPES['xor_low_bit_i8']:
        scalar, yld = body
        if len(scalar.operands) != 2:
            raise InvalidLinalgPattern('integer XOR arity differs')
        literal(scalar.operands[1], 1, 'i8')
        check(scalar, arith.XOrIOp, (args[0], scalar.operands[1]), 'i8')
        result, operation = scalar.results[0], 'xor_low_bit_i8'
    else:
        raise InvalidLinalgPattern('source is not one of the two exact integer seams')
    if (type(yld) is not YieldOp or tuple(yld.operands) != (result,) or yld.results
            or yld.properties or yld.attributes or yld.regions or yld.successors):
        raise InvalidLinalgPattern('integer seam yield differs')
    return StaticPointwisePattern(operation, shell.shape, shell.ordered_types)
