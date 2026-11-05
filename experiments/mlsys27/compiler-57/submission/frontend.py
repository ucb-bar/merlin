"""Agent-authored frontend using the admitted portable structural readers.

Target mappings are derived from typed SSA and exact indexing maps. The copied
parser_support modules only parse the public input grammar; they do no lowering.
"""
from math import prod
from parser_support.interface_emit import parse_interface_mlir
from parser_support.linalg_iface import is_linalg_on_tensors, parse_linalg_mlir, make_linalg_context
from parser_support.source_inventory import source_operation_inventory
from div_scalar import match_quantized_div


def _map_dims(text):
    """Read only affine projections: no guessing from tensor extents."""
    from xdsl.parser import Parser
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr
    attr = Parser(make_linalg_context(), text).parse_attribute()
    result = []
    for expr in attr.data.results:
        if isinstance(expr, AffineDimExpr):
            result.append(expr.position)
        elif isinstance(expr, AffineConstantExpr) and expr.value == 0:
            result.append(None)
        else:
            raise ValueError('affine expression needs a general indexed lowering')
    return result


def _integer_attr(text):
    return int(text.partition(' : ')[0].strip())


def _int_list(text):
    """Read a typed dense integer array through the same xDSL context."""
    from xdsl.parser import Parser
    attr = Parser(make_linalg_context(), text).parse_attribute()
    if hasattr(attr, 'get_values'):
        return [int(v) for v in attr.get_values()]
    return [int(v.value.data) for v in attr]


def _static_slice_geometry(attributes, container_shape, sliced_shape):
    """Validate static slice bounds and retain the axes surviving rank reduction."""
    offsets = _int_list(attributes['static_offsets'])
    sizes = _int_list(attributes['static_sizes'])
    strides = _int_list(attributes['static_strides'])
    rank = len(container_shape)
    if any(len(values) != rank for values in (offsets, sizes, strides)):
        raise ValueError('static slice parameter rank mismatch')
    for extent, offset, size, stride in zip(container_shape, offsets, sizes, strides):
        if extent < 0 or offset < 0 or size < 0 or stride <= 0:
            raise ValueError('slice needs static nonnegative bounds and positive strides')
        if offset > extent or (size and offset + (size - 1) * stride >= extent):
            raise ValueError('static slice exceeds its source extent')
    kept = []
    result_axis = 0
    for axis, size in enumerate(sizes):
        if result_axis < len(sliced_shape) and size == sliced_shape[result_axis]:
            kept.append(axis)
            result_axis += 1
        elif size != 1:
            raise ValueError('slice rank reduction may remove only unit dimensions')
    if result_axis != len(sliced_shape):
        raise ValueError('slice sizes disagree with the ranked result')
    return offsets, sizes, strides, kept


def _integer_projection_body(record):
    """Prove a scalar region yields an unchanged input or an integer constant.

    Cast chains retain the original signed range, so i8 -> i32 -> i8 is a
    copy, whereas an arbitrary i32 -> i8 truncation is not. Constant arithmetic
    uses the declared integer width, including modular add/sub/multiply.
    """
    body = record['scalar_body']
    if not body or len(body['yields']) != 1:
        return None
    pure = {'arith.constant','arith.extsi','arith.trunci','arith.addi','arith.subi',
            'arith.muli','arith.maxsi','arith.maximumsi','arith.minsi','arith.minimumsi'}
    if any(op['regions'] or op['effects'] not in ([], None) or
           (op['effects'] is None and op['op'] not in pure) for op in body['operations']):
        return None

    def bits(dtype):
        return int(dtype[1:]) if dtype.startswith('i') and dtype[1:].isdigit() else None

    def signed(value, width):
        value %= 1 << width
        return value - (1 << width) if value >= 1 << (width - 1) else value

    values = {}
    for i, edge in enumerate(record['ins']):
        width = bits(edge['dtype'])
        if width is None:
            return None
        if 'const_value' in edge and type(edge['const_value']) is int:
            values[f'arg:{i}'] = ('constant', signed(edge['const_value'], width), width)
        else:
            values[f'arg:{i}'] = ('input', i, width)
    for i, edge in enumerate(body['captures']):
        width = bits(edge['dtype'])
        if width is not None and type(edge.get('const_value')) is int:
            values[f'capture:{i}'] = ('constant', signed(edge['const_value'], width), width)
    for i, op in enumerate(body['operations']):
        if len(op['results']) != 1:
            return None
        width = bits(op['results'][0])
        if width is None:
            return None
        args = [values.get(ref) for ref in op['operands']]
        kind = op['op']
        value = None
        if kind == 'arith.constant':
            value = ('constant', signed(_integer_attr(op['attributes']['value']), width), width)
        elif kind in ('arith.extsi', 'arith.trunci') and len(args) == 1 and args[0] is not None:
            arg = args[0]
            if arg[0] == 'constant':
                value = ('constant', signed(arg[1], width), width)
            elif width >= bits(record['ins'][arg[1]]['dtype']):
                value = ('input', arg[1], width)
        elif len(args) == 2 and all(arg is not None and arg[0] == 'constant' for arg in args):
            a, b = args[0][1], args[1][1]
            if kind == 'arith.addi':
                number = a + b
            elif kind == 'arith.subi':
                number = a - b
            elif kind == 'arith.muli':
                number = a * b
            elif kind in ('arith.maxsi', 'arith.maximumsi'):
                number = max(a, b)
            elif kind in ('arith.minsi', 'arith.minimumsi'):
                number = min(a, b)
            else:
                return None
            value = ('constant', signed(number, width), width)
        # An unrecognized pure operation cannot establish a yielded copy, but
        # independent, unused scalar arithmetic does not change a proven yield.
        values[f'op:{i}:0'] = value
    result = values.get(body['yields'][0])
    if result is None or result[2] != bits(record['results'][0]['dtype']):
        return None
    if result[0] == 'input':
        source_type = record['ins'][result[1]]['dtype']
        destination_type = record['results'][0]['dtype']
        if source_type != destination_type and (source_type, destination_type) != ('i8', 'i32'):
            return None
    return result[:2]


def _projected_copy_attributes(record, input_index):
    """Express bounded positive affine reads as strides in output order.

    Each source coordinate is a constant plus a nonnegative linear expression
    in the parallel iterators. Flattening those coordinates gives an ordinary
    as_strided device movement, including slicing, broadcasting and diagonals.
    Bounds are checked for each source axis before flattening, so an invalid
    coordinate cannot silently wrap into a neighboring row.
    """
    from xdsl.parser import Parser
    from xdsl.ir.affine import (
        AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr,
    )

    dimensions = len(record['iterator_types'])
    if any(kind != 'parallel' for kind in record['iterator_types']):
        raise ValueError('projected copy requires parallel iterators')
    source_map = Parser(make_linalg_context(), record['indexing_maps'][input_index]).parse_attribute().data
    if source_map.num_dims != dimensions or source_map.num_symbols:
        raise ValueError('projected copy requires matching static affine dimensions without symbols')

    def linear(expr):
        if isinstance(expr, AffineConstantExpr):
            return [0] * dimensions, expr.value
        if isinstance(expr, AffineDimExpr) and 0 <= expr.position < dimensions:
            coefficients = [0] * dimensions
            coefficients[expr.position] = 1
            return coefficients, 0
        if isinstance(expr, AffineBinaryOpExpr):
            left, left_offset = linear(expr.lhs)
            right, right_offset = linear(expr.rhs)
            if expr.kind == AffineBinaryOpKind.Add:
                return [a + b for a, b in zip(left, right)], left_offset + right_offset
            if expr.kind == AffineBinaryOpKind.Mul:
                if not any(left):
                    return [left_offset * value for value in right], left_offset * right_offset
                if not any(right):
                    return [right_offset * value for value in left], right_offset * left_offset
        raise ValueError('projected copy needs constant-offset linear affine indexing')

    source = record['ins'][input_index]['shape']
    destination = record['results'][0]['shape']
    output_map = _map_dims(record['indexing_maps'][-1])
    if len(source_map.results) != len(source) or len(output_map) != len(destination):
        raise ValueError('projection map rank disagrees with tensor rank')
    if sorted(output_map, key=lambda dim: -1 if dim is None else dim) != list(range(dimensions)):
        raise ValueError('projection output must cover every parallel iterator once')
    if any(type(size) is not int or size <= 0 for size in list(source) + list(destination)):
        raise ValueError('projected copy needs positive static extents')
    iterator_extents = [destination[output_map.index(dim)] for dim in range(dimensions)]
    strides = [0] * len(destination)
    source_stride = 1
    storage_offset = 0
    for size, expression in reversed(list(zip(source, source_map.results))):
        coefficients, offset = linear(expression)
        if offset < 0 or any(coefficient < 0 for coefficient in coefficients):
            raise ValueError('projected copy requires nonnegative offsets and strides')
        last_coordinate = offset + sum(coefficient * (extent - 1)
                                       for coefficient, extent in zip(coefficients, iterator_extents))
        if last_coordinate >= size:
            raise ValueError('projection reads beyond its source extent')
        storage_offset += offset * source_stride
        for dim, coefficient in enumerate(coefficients):
            strides[output_map.index(dim)] += coefficient * source_stride
        source_stride *= size
    return {'form':'as_strided', 'strides':strides, 'storage_offset':storage_offset,
            'output_dtype':record['results'][0]['dtype']}


def _captured_extract_attributes(record):
    """Prove a parallel body's yielded tensor.extract is a bounded affine read."""
    body = record['scalar_body']
    if not body or not any(op['op'] == 'tensor.extract' for op in body['operations']):
        return None
    if len(body['yields']) != 1 or body['yields'][0] not in {
            f'op:{index}:0' for index, op in enumerate(body['operations'])
            if op['op'] == 'tensor.extract'}:
        return None
    if len(record['results']) != 1 or len(body['yields']) != 1 or any(kind != 'parallel' for kind in record['iterator_types']):
        return None
    dtype = record['results'][0]['dtype']
    destination = record['results'][0]['shape']
    if dtype not in ('i8','i32','f32'):
        raise ValueError('captured tensor extraction requires supported DMA storage')
    dimensions = len(record['iterator_types'])
    maps = record['indexing_maps']
    if not maps or len(record['outs']) != 1:
        raise ValueError('captured extraction requires one complete output indexing map')
    output_map = _map_dims(maps[-1])
    if len(output_map) != len(destination) or sorted(output_map, key=lambda v: -1 if v is None else v) != list(range(dimensions)):
        raise ValueError('captured extraction output must cover every parallel iterator once')
    if any(type(size) is not int or size <= 0 for size in destination):
        raise ValueError('captured extraction requires positive static output extents')
    extents = [destination[output_map.index(dim)] for dim in range(dimensions)]
    zeros = tuple(0 for _ in range(dimensions))
    values = {}
    captures = {}
    for i, edge in enumerate(body['captures']):
        ref = f'capture:{i}'
        if edge['dtype'] == 'index' and type(edge.get('const_value')) is int:
            values[ref] = (zeros, edge['const_value'])
        else:
            captures[ref] = edge
    extracted = {}
    allowed = {'linalg.index','arith.constant','arith.addi','arith.subi','arith.muli','tensor.extract'}

    def bounded(expression):
        coefficients, offset = expression
        low = offset + sum(min(0, c * (size - 1)) for c, size in zip(coefficients, extents))
        high = offset + sum(max(0, c * (size - 1)) for c, size in zip(coefficients, extents))
        if not -(1 << 63) <= low <= high < (1 << 63):
            raise ValueError('captured extraction index arithmetic may overflow signed index storage')
        return expression

    for i, op in enumerate(body['operations']):
        kind = op['op']
        if kind not in allowed or op['regions'] or op['effects'] not in ([], None) or len(op['results']) != 1:
            raise ValueError('captured extraction body contains unsupported effects or arithmetic')
        result = f'op:{i}:0'
        args = [values.get(ref) for ref in op['operands']]
        if kind == 'tensor.extract':
            source = captures.get(op['operands'][0]) if op['operands'] else None
            if source is None or source['dtype'] != dtype or op['results'] != [dtype] or len(args) != len(source['shape']) + 1 or any(v is None for v in args[1:]):
                raise ValueError('tensor.extract must read a captured tensor at affine index coordinates')
            extracted[result] = (source, args[1:])
            continue
        if op['results'] != ['index']:
            raise ValueError('captured extraction arithmetic must preserve the index type')
        if kind == 'linalg.index':
            axis = _integer_attr(op['attributes']['dim'])
            if op['operands'] or not 0 <= axis < dimensions:
                raise ValueError('linalg.index refers outside the parallel iterator domain')
            coefficients = [0] * dimensions
            coefficients[axis] = 1
            expression = (tuple(coefficients), 0)
        elif kind == 'arith.constant' and not op['operands']:
            expression = (zeros, _integer_attr(op['attributes']['value']))
        elif kind in ('arith.addi','arith.subi','arith.muli') and len(args) == 2 and all(v is not None for v in args):
            (left, left_offset), (right, right_offset) = args
            if kind in ('arith.addi','arith.subi'):
                sign = 1 if kind == 'arith.addi' else -1
                expression = (tuple(a + sign * b for a,b in zip(left,right)), left_offset + sign * right_offset)
            elif not any(left):
                expression = (tuple(left_offset * c for c in right), left_offset * right_offset)
            elif not any(right):
                expression = (tuple(right_offset * c for c in left), right_offset * left_offset)
            else:
                raise ValueError('captured extraction multiplication requires a constant operand')
        else:
            raise ValueError('captured extraction uses a non-affine index dependency')
        values[result] = bounded(expression)
    if len(extracted) != 1 or body['yields'][0] not in extracted:
        raise ValueError('captured extraction body must yield its single tensor read directly')
    source, coordinates = extracted[body['yields'][0]]
    if any(type(size) is not int or size <= 0 for size in source['shape']):
        raise ValueError('captured extraction requires positive static input extents')
    storage_offset = 0
    strides = [0] * len(destination)
    source_stride = 1
    for size, (coefficients, offset) in reversed(list(zip(source['shape'], coordinates))):
        if offset < 0 or any(c < 0 for c in coefficients):
            raise ValueError('captured extraction requires nonnegative coordinate offsets and strides')
        if offset + sum(c * (extent - 1) for c,extent in zip(coefficients,extents)) >= size:
            raise ValueError('captured extraction reads beyond its source extent')
        storage_offset += offset * source_stride
        for dim,c in enumerate(coefficients):
            strides[output_map.index(dim)] += c * source_stride
        source_stride *= size
    if any(stride <= 0 for stride in strides):
        raise ValueError('captured extraction requires positive output-axis storage strides')
    return source, {'form':'as_strided','storage_offset':storage_offset,
                    'strides':strides,'output_dtype':dtype}


def _linear_body(record):
    """Prove the integer body is a constant affine expression, optionally ReLU."""
    body = record['scalar_body']
    n = len(record['ins'])
    zero = (tuple(0 for _ in range(n)), 0, False)
    values = {}
    for i, operand in enumerate(record['ins']):
        if 'const_value' in operand:
            values[f'arg:{i}'] = (zero[0], int(operand['const_value']), False)
        else:
            values[f'arg:{i}'] = (tuple(int(j == i) for j in range(n)), 0, False)
    for i, operand in enumerate(body['captures']):
        if 'const_value' not in operand or type(operand['const_value']) is not int:
            raise ValueError('nonconstant capture in integer affine body')
        values[f'capture:{i}'] = (zero[0], operand['const_value'], False)
    for i, op in enumerate(body['operations']):
        if len(op['results']) != 1:
            raise ValueError('multi-result integer affine body')
        kind = op['op']
        args = [values[v] for v in op['operands']]
        if kind == 'arith.constant':
            token = op['attributes']['value'].partition(' : ')[0]
            if op['results'][0].startswith('f'):
                number = float(token)
                if not number.is_integer() or abs(number)>2**24:
                    raise ValueError('noninteger floating coefficient needs a readout lowering')
                number = int(number)
            else:
                number = int(token)
            value = (zero[0], number, False)
        elif kind in ('arith.extsi','arith.sitofp'):
            value = args[0]
        elif kind in ('arith.addi', 'arith.subi', 'arith.addf', 'arith.subf'):
            a, b = args
            if a[2] or b[2]:
                raise ValueError('arithmetic after ReLU is not an affine readout')
            sign = 1 if kind in ('arith.addi','arith.addf') else -1
            value = (tuple(x + sign*y for x,y in zip(a[0],b[0])), a[1]+sign*b[1], False)
        elif kind in ('arith.muli','arith.mulf'):
            a,b = args
            if a[2] or b[2]:
                raise ValueError('multiplication after ReLU is unsupported')
            if any(a[0]) and any(b[0]):
                raise ValueError('tensor product is not affine')
            if not any(a[0]):
                a,b=b,a
            value = (tuple(x*b[1] for x in a[0]), a[1]*b[1], False)
        elif kind in ('arith.maxsi', 'arith.maximumsi','arith.maximumf'):
            a,b = args
            if a == zero:
                a,b=b,a
            if b != zero:
                raise ValueError('integer maximum is not ReLU')
            value = (a[0],a[1],True)
        elif kind in ('math.roundeven','arith.fptosi'):
            if kind == 'arith.fptosi' and op['results'] != ['i32']:
                raise ValueError('floating affine conversion needs an i32 result')
            value = args[0]
        else:
            raise ValueError('unmapped scalar affine op '+kind)
        if op['results'][0]=='f32':
            # Every intermediate integer must be exactly representable in f32;
            # this proves each source product/add/round agrees with integer math.
            bound = abs(value[1])
            for coeff,edge in zip(value[0],record['ins']):
                if coeff:
                    if edge['dtype']!='i8':
                        raise ValueError('floating affine source has no finite exact integer bound')
                    bound += abs(coeff)*128
            if bound>2**24:
                raise ValueError('floating affine intermediate exceeds exact f32 integer range')
        values[f'op:{i}:0'] = value
    if len(body['yields']) != 1:
        raise ValueError('multi-result affine yield')
    return values[body['yields'][0]]


def _is_simple_sum(record):
    """Match exactly yield=old+sign_extend(input)."""
    body = record['scalar_body']
    if not body or len(record['ins']) != 1 or len(body['arguments']) != 2:
        return False
    values = {'arg:0': 'x', 'arg:1': 'old'}
    for i, op in enumerate(body['operations']):
        args = [values.get(v) for v in op['operands']]
        if op['op'] == 'arith.extsi' and args == ['x']:
            result = 'x'
        elif op['op'] == 'arith.addi' and sorted(str(a) for a in args) == ['old','x']:
            result = 'sum'
        else:
            return False
        values[f'op:{i}:0'] = result
    return len(body['yields']) == 1 and values.get(body['yields'][0]) == 'sum'


def _is_parallel_product(record):
    """Match equal-shape signed-byte products widened before multiplication."""
    body = record['scalar_body']
    if (not body or len(record['ins']) != 2 or len(record['results']) != 1 or
            len(body['arguments']) != 3 or len(body['yields']) != 1 or body['captures'] or
            sorted(op['op'] for op in body['operations']) != ['arith.extsi','arith.extsi','arith.muli'] or
            record['results'][0]['dtype'] != 'i32' or
            any(edge['dtype'] != 'i8' or edge['shape'] != record['results'][0]['shape'] for edge in record['ins']) or
            any(kind != 'parallel' for kind in record['iterator_types'])):
        return False
    maps = [_map_dims(v) for v in record['indexing_maps']]
    if (len(maps) != 3 or not maps[0] == maps[1] == maps[2] or
            len(maps[-1]) != len(record['results'][0]['shape']) or
            sorted(maps[-1], key=lambda v: -1 if v is None else v) != list(range(len(record['iterator_types'])))):
        return False
    values = {'arg:0':('a','i8'), 'arg:1':('b','i8')}
    for i,op in enumerate(body['operations']):
        if op['regions'] or op['effects'] not in ([], None) or op['results'] != ['i32']:
            return False
        args = [values.get(ref) for ref in op['operands']]
        if op['op'] == 'arith.extsi' and len(args) == 1 and args[0] in (('a','i8'),('b','i8')):
            result = (args[0][0],'i32')
        elif op['op'] == 'arith.muli' and len(args) == 2 and set(args) == {('a','i32'),('b','i32')}:
            result = ('product','i32')
        else:
            return False
        values[f'op:{i}:0'] = result
    return values.get(body['yields'][0]) == ('product','i32')


def _is_simple_product_sum(record):
    """Match exactly yield=old+sign_extend(lhs)*sign_extend(rhs)."""
    body = record['scalar_body']
    if (not body or len(record['ins']) != 2 or body['arguments'] != ['i8','i8','i32'] or
            body['captures']):
        return False
    values = {'arg:0': 'byte_a', 'arg:1': 'byte_b', 'arg:2': 'old'}
    for i, op in enumerate(body['operations']):
        if op['results'] != ['i32'] or op['regions'] or op['effects'] not in ([],None):
            return False
        args = [values.get(v) for v in op['operands']]
        if op['op'] == 'arith.extsi' and args in (['byte_a'],['byte_b']):
            result = args[0][5:]
        elif op['op'] == 'arith.muli' and sorted(str(a) for a in args) == ['a','b']:
            result = 'product'
        elif op['op'] == 'arith.addi' and sorted(str(a) for a in args) == ['old','product']:
            result = 'sum'
        else:
            return False
        values[f'op:{i}:0'] = result
    return len(body['yields']) == 1 and values.get(body['yields'][0]) == 'sum'


def _contraction_execution_attributes(record, constant_tensors, right, transpose_rhs, seed):
    """Honor explicit source dataflow and prove the declared OS prefix domain.

    The selected GemminiRocketConfig facts report signed 8-bit Tile.in_a and
    signed 20-bit Tile.in_b/in_d/out_c/out_b carriers. Its reviewed software
    contract permits OS only when literal weights prove every retained prefix
    over the complete signed-byte activation domain, without an initial bias.
    No capsule metadata or runtime values participate in this proof.
    """
    from xdsl.dialects.builtin import StringAttr
    from xdsl.parser import Parser

    declared = []
    for key in ('dataflow', 'gemmini.dataflow', 'execution.dataflow'):
        value = record['attributes'].get(key)
        if value is None:
            continue
        attribute = Parser(make_linalg_context(), value).parse_attribute()
        if not isinstance(attribute, StringAttr):
            raise ValueError('contraction dataflow must be an explicit WS or OS string')
        spelling = attribute.data.upper()
        aliases = {'WS':'WS', 'WEIGHT_STATIONARY':'WS', 'OS':'OS', 'OUTPUT_STATIONARY':'OS'}
        if spelling not in aliases:
            raise ValueError('unsupported contraction dataflow '+attribute.data)
        declared.append(aliases[spelling])
    if not declared:
        return {}
    if any(value != declared[0] for value in declared):
        raise ValueError('conflicting source contraction dataflow declarations')
    if declared[0] == 'WS':
        return {'dataflow':'WS'}
    if seed is not None:
        raise ValueError('OS accumulator initialization with a nonzero or runtime bias is unproved')
    weights = constant_tensors.get(right)
    if weights is None or weights['dtype'] != 'i8' or len(weights['shape']) != 2:
        raise ValueError('OS requires literal signed-byte rank-two weights for its prefix proof')
    shape = weights['shape']
    k, n = (shape[1], shape[0]) if transpose_rhs else shape
    if type(k) is not int or type(n) is not int or k <= 0 or n <= 0:
        raise ValueError('OS requires positive static weight extents')
    values = weights['values']
    if len(values) not in (1, k*n) or any(type(value) is not int or not -128 <= value <= 127 for value in values):
        raise ValueError('OS literal weight storage is not an exact signed-byte tensor')
    mesh_min, mesh_max = -(1 << 19), (1 << 19) - 1
    partial_min, partial_max = 0, 0
    for column in range(n):
        low, high = 0, 0
        for inner in range(k):
            index = column*k+inner if transpose_rhs else inner*n+column
            weight = values[0] if len(values) == 1 else values[index]
            a, b = -128*weight, 127*weight
            low += min(a, b)
            high += max(a, b)
            if low < mesh_min or high > mesh_max:
                raise ValueError('OS literal-weight partial sum exceeds the signed 20-bit mesh interval')
            partial_min = min(partial_min, low)
            partial_max = max(partial_max, high)
    return {'dataflow':'OS', 'execution':{
        'operand_signed_bits':8, 'mesh_signed_bits':20,
        'pe_partial_min':partial_min, 'pe_partial_max':partial_max,
        'accumulator_partial_abs_bound':max(-partial_min, partial_max),
        'pe_bound_scope':'every_retained_partial_sum_including_total_os_k',
        'proof':'constant_weight_all_prefix_intervals',
    }}


def _prefix_axis(record):
    body = record['scalar_body']
    if not body or len(record['ins']) != 1 or record['ins'][0]['dtype'] != 'i8' or record['results'][0]['dtype'] != 'i32':
        return None
    initializers = record['outs']
    if len(initializers) != 1 or initializers[0]['dtype'] != 'i32' or initializers[0]['shape'] != record['results'][0]['shape']:
        return None
    # DEVICE_PREFIX_SUM starts from zero. A runtime destination or a literal
    # nonzero seed must not lose its initial value during this recognition.
    initializer = initializers[0]
    if initializer.get('init_value', initializer.get('const_value')) != 0:
        return None
    if record['ins'][0]['shape'] != record['results'][0]['shape']:
        return None
    maps = [_map_dims(v) for v in record['indexing_maps']]
    if len(maps)!=2 or len(record['reduction_dims'])!=1:
        return None
    reduced = record['reduction_dims'][0]
    if reduced not in maps[0]:
        return None
    axis = maps[0].index(reduced)
    output_dim = maps[1][axis]
    if [output_dim if d==reduced else d for d in maps[0]] != maps[1]:
        return None
    values = {'arg:0':'x','arg:1':'old'}
    for i,edge in enumerate(body['captures']):
        if edge.get('const_value') != 0:
            return None
        values[f'capture:{i}'] = 0
    for i,op in enumerate(body['operations']):
        args = [values.get(v) for v in op['operands']]
        kind = op['op']
        if kind == 'linalg.index':
            result = ('index',_integer_attr(op['attributes']['dim']))
        elif kind == 'arith.cmpi' and _integer_attr(op['attributes']['predicate'])==7 and args == [('index',reduced),('index',output_dim)]:
            result = 'condition'
        elif kind == 'arith.extsi' and args == ['x']:
            result = 'x'
        elif kind == 'arith.select' and args == ['condition','x',0]:
            result = 'term'
        elif kind == 'arith.addi' and sorted(str(a) for a in args)==['old','term']:
            result = 'sum'
        else:
            return None
        values[f'op:{i}:0'] = result
    return axis if len(body['yields'])==1 and values.get(body['yields'][0])=='sum' else None


def _rounded_extension(record):
    """Match an exact integer operand extension, without nonlinear rounding."""
    if [v['dtype'] for v in record['ins']] != ['i32','i8'] or record['results'][0]['dtype']!='i32':
        return None
    body = record['scalar_body']
    if not body or body['captures'] or len(body['operations'])!=6 or body['yields']!=['op:5:0']:
        return None
    ops = body['operations']
    required = [('arith.sitofp',['arg:1'],['f32']),('arith.constant',[],['f32']),
                ('arith.mulf',['op:0:0','op:1:0'],['f32']),('math.roundeven',['op:2:0'],['f32']),
                ('arith.fptosi',['op:3:0'],['i32']),('arith.addi',['arg:0','op:4:0'],['i32'])]
    if any((o['op'],o['operands'],o['results'])!=expected for o,expected in zip(ops,required)):
        return None
    gain = float(ops[1]['attributes']['value'].partition(' : ')[0])
    import math,struct
    if not math.isfinite(gain):
        return None
    if not gain.is_integer():
        raise ValueError('rounded operand extension requires an exact integer coefficient; nonlinear rounding construction is not admitted')
    for x in range(-128,128):
        value = struct.unpack('<f',struct.pack('<f',float(x)*gain))[0]
        if not -128 <= round(value) <= 127:
            return None
    maps = [_map_dims(v) for v in record['indexing_maps']]
    if maps != [maps[-1],maps[-1],maps[-1]]:
        return None
    return gain


def _constant_values(record):
    from xdsl.parser import Parser
    attr = Parser(make_linalg_context(), record['attributes']['value']).parse_attribute()
    if hasattr(attr, 'get_values'):
        return list(attr.get_values())
    return [attr.value.data]


def _device_readout(record):
    """Recognize exactly f32 conversion/product/RNE/saturate readouts."""
    body = record['scalar_body']
    if not body or len(record['ins']) != 2 or record['ins'][0]['dtype'] not in ('i8','i32') or record['ins'][1]['dtype'] != 'f32':
        return None
    if record['results'] != [{'shape':record['ins'][0]['shape'],'dtype':'i8'}]:
        return None
    values = {'arg:0':'input','arg:1':'gain'}
    for i,op in enumerate(body['operations']):
        args = [values.get(v) for v in op['operands']]
        kind = op['op']
        if kind == 'arith.constant' and op['results'] == ['f32']:
            result = float(op['attributes']['value'].partition(' : ')[0])
        elif kind == 'arith.sitofp' and args == ['input'] and op['results'] == ['f32']:
            result = 'converted'
        elif kind == 'arith.maximumf' and 0.0 in args and 'converted' in args:
            result = 'relu'
        elif kind == 'arith.mulf' and 'gain' in args:
            other = args[1] if args[0] == 'gain' else args[0]
            if other not in ('converted','relu'):
                return None
            result = ('product', other)
        elif kind == 'math.roundeven' and isinstance(args[0],tuple) and args[0][0] == 'product':
            result = ('rounded',args[0][1])
        elif kind == 'arith.maximumf' and -128.0 in args:
            other = args[1] if args[0] == -128.0 else args[0]
            if not isinstance(other,tuple) or other[0]!='rounded':
                return None
            result = ('low',other[1])
        elif kind == 'arith.minimumf' and 127.0 in args:
            other = args[1] if args[0] == 127.0 else args[0]
            if not isinstance(other,tuple) or other[0]!='low':
                return None
            result = ('high',other[1])
        elif kind == 'arith.fptosi' and op['results']==['i8']:
            if not isinstance(args[0],tuple) or args[0][0]!='high':
                return None
            result = ('result',args[0][1])
        else:
            return None
        values[f'op:{i}:0'] = result
    result = values.get(body['yields'][0]) if len(body['yields'])==1 else None
    if not isinstance(result,tuple) or result[0]!='result':
        return None
    maps = [_map_dims(v) for v in record['indexing_maps']]
    output = maps[-1]
    scalar_gain = prod(record['ins'][1]['shape'])==1 and all(d is None for d in maps[1])
    if maps[0] != output or not (maps[1] == [output[-1]] or scalar_gain):
        return None
    return {'relu':result[1]=='relu','output_dtype':'i8','scalar_body':body}


def _adjoint_return_labels(inventory, text):
    """Recognize the complete matrix-adjoint tuple from typed source dataflow.

    Names describe the three algebraic results, independently of source return
    order. A reshape remains its own output producer; this routine changes only
    the logical boundary label, never source ownership or arithmetic lowering.
    """
    returned = inventory['returns']
    if returned is None or len(returned) != 3:
        return {}
    producers = {rec['id']:rec for rec in inventory['ops']}
    carriers = {'tensor.expand_shape','tensor.collapse_shape','tensor.reshape'}

    def key(edge):
        return (*edge['source'], edge.get('result_index',0))

    def root_shape(identity):
        if identity[0] == 'arg':
            return inventory['args'][identity[1]]['shape']
        if identity[0] == 'op':
            return producers[identity[1]]['results'][identity[2]]['shape']
        return None

    def unwrap(edge):
        seen = set()
        while edge['source'][0] == 'op':
            identity = key(edge)
            if identity in seen:
                return None
            seen.add(identity)
            rec = producers.get(identity[1])
            if rec is None or identity[2] != 0 or len(rec['results']) != 1:
                return None
            if rec['kind'] not in carriers:
                break
            if (len(rec['ins']) != 1 or rec['ins'][0]['dtype'] != edge['dtype'] or
                    rec['results'][0] != {'shape':edge['shape'],'dtype':edge['dtype']} or
                    any(d <= 0 for d in rec['ins'][0]['shape'] + edge['shape']) or
                    prod(rec['ins'][0]['shape']) != prod(edge['shape'])):
                return None
            edge = rec['ins'][0]
        return edge

    def matrix_view(edge):
        if edge['dtype'] != 'i8' or len(edge['shape']) != 2 or any(d <= 0 for d in edge['shape']):
            return None
        requested_shape = tuple(edge['shape'])
        base = unwrap(edge)
        if base is None:
            return None
        if base['source'][0] == 'op':
            rec = producers.get(base['source'][1])
            if rec is not None and rec['kind'] == 'linalg.transpose':
                if (len(rec['ins']) != 1 or len(base['shape']) != 2 or
                        tuple(base['shape']) != requested_shape or
                        _int_list(rec['attributes']['permutation']) != [1,0] or
                        list(reversed(rec['ins'][0]['shape'])) != base['shape']):
                    return None
                source = matrix_view(rec['ins'][0])
                if source is None:
                    return None
                return (source[0], not source[1], source[2], requested_shape)
        # A checked contiguous carrier may provide the matrix view of a
        # higher-rank source. Keep both its source identity and matrix shape.
        return (key(base), False, requested_shape, requested_shape)

    typed_payload = None

    def zero_initializer(rec):
        nonlocal typed_payload
        if (len(rec['outs']) != 1 or rec['outs'][0]['dtype'] != 'i32' or
                rec['outs'][0]['shape'] != rec['results'][0]['shape']):
            return False
        edge = rec['outs'][0]
        literal = edge.get('init_value', edge.get('const_value'))
        if type(literal) is int:
            return literal == 0
        if edge['source'][0] == 'arg':
            return False
        # Destination-only dense literals are absent from the payload reader.
        # Join the actual typed destination operand to its original owner.
        if typed_payload is None:
            from xdsl.parser import Parser
            from parser_support.linalg_iface import _initialization_only
            module = Parser(make_linalg_context(), text).parse_module()
            functions = [op for op in module.body.block.ops if op.name == 'func.func' and op.body.blocks]
            if len(functions) != 1 or len(functions[0].body.blocks) != 1:
                return False
            typed_payload = [op for op in functions[0].body.blocks[0].ops
                             if op.name != 'func.return' and not _initialization_only(op)]

        def literal_zero(value):
            owner = value.owner
            name = getattr(owner, 'name', '')
            if name == 'arith.constant':
                attr = {**owner.attributes, **owner.properties}.get('value')
                if hasattr(attr, 'get_values'):
                    decoded = list(attr.get_values())
                    return bool(decoded) and all(type(v) is int and v == 0 for v in decoded)
                data = getattr(getattr(attr, 'value', None), 'data', None)
                return type(data) is int and data == 0
            if name in ('tensor.splat','linalg.fill') and owner.operands:
                return literal_zero(owner.operands[0])
            if name in carriers and owner.operands:
                return literal_zero(owner.operands[0])
            return False

        operation = typed_payload[rec['id']]
        return literal_zero(operation.operands[len(rec['ins'])])

    def product_sum(rec):
        if rec['kind'] != 'linalg.generic':
            if rec['kind'] not in ('linalg.matmul','linalg.matmul_transpose_a','linalg.matmul_transpose_b') or not rec['indexing_maps']:
                return False
            expected_maps = [[0,2],[2,1],[0,1]]
            if rec['kind'] == 'linalg.matmul_transpose_a':
                expected_maps[0] = [2,0]
            if rec['kind'] == 'linalg.matmul_transpose_b':
                expected_maps[1] = [1,2]
            if [_map_dims(value) for value in rec['indexing_maps']] != expected_maps:
                return False
            from xdsl.parser import Parser
            for value in rec['indexing_maps']:
                affine = Parser(make_linalg_context(), value).parse_attribute().data
                if affine.num_dims != 3 or affine.num_symbols:
                    return False
            # This reader leaves the fixed named-op iterator list empty when
            # no iterator_types override exists. Only that default and the
            # actual canonical explicit list establish the named semantics.
            iterators = rec['iterator_types']
            if not iterators and 'iterator_types' not in rec['attributes']:
                iterators = ['parallel','parallel','reduction']
            if iterators != ['parallel','parallel','reduction']:
                return False
            for attribute,value in rec['attributes'].items():
                if attribute == 'cast':
                    if value != '#linalg.type_fn<cast_signed>':
                        return False
                elif attribute.startswith('cast') or attribute.endswith('.cast'):
                    return False
            return True
        body = rec['scalar_body']
        if (not body or body['captures'] or
                sorted(op['op'] for op in body['operations']) != ['arith.addi','arith.extsi','arith.extsi','arith.muli'] or
                any(op['regions'] or op['effects'] not in ([],None) or op['results'] != ['i32'] for op in body['operations'])):
            return False
        return _is_simple_product_sum(rec)

    def contraction(edge):
        core = unwrap(edge)
        if core is None or core['source'][0] != 'op':
            return None
        rec = producers[core['source'][1]]
        if (len(rec['results']) != 1 or rec['results'][0]['dtype'] != 'i32' or
                len(rec['results'][0]['shape']) != 2 or len(rec['ins']) != 2 or
                any(v['dtype'] != 'i8' or len(v['shape']) != 2 for v in rec['ins']) or
                not product_sum(rec) or not zero_initializer(rec)):
            return None
        if rec['kind'] == 'linalg.generic':
            maps = [_map_dims(value) for value in rec['indexing_maps']]
            iters = rec['iterator_types']
            if len(maps) != 3 or any(len(m) != 2 for m in maps) or len(iters) != 3:
                return None
            m,n = maps[-1]
            if m is None or n is None or m == n:
                return None
            reduction = [i for i,kind in enumerate(iters) if kind == 'reduction']
            if len(reduction) != 1 or sorted([m,n,reduction[0]]) != [0,1,2] or iters[m] != 'parallel' or iters[n] != 'parallel':
                return None
            k = reduction[0]
            if maps[0] not in ([m,k],[k,m]) or maps[1] not in ([k,n],[n,k]):
                return None
            left_transpose, right_transpose = maps[0] == [k,m], maps[1] == [n,k]
        else:
            left_transpose = rec['kind'] == 'linalg.matmul_transpose_a'
            right_transpose = rec['kind'] == 'linalg.matmul_transpose_b'
        left,right = [matrix_view(value) for value in rec['ins']]
        if left is None or right is None:
            return None

        def oriented(view, transpose):
            shape = tuple(reversed(view[3])) if transpose else view[3]
            return (view[0], view[1] != transpose, view[2], shape)

        left,right = oriented(left,left_transpose),oriented(right,right_transpose)
        if left[3][1] != right[3][0] or rec['results'][0]['shape'] != [left[3][0],right[3][1]]:
            return None
        return key(core),left,right,tuple(rec['results'][0]['shape'])

    def bias_reduction(edge):
        core = unwrap(edge)
        if core is None or core['source'][0] != 'op':
            return None
        rec = producers[core['source'][1]]
        if (rec['kind'] != 'linalg.generic' or len(rec['results']) != 1 or
                rec['results'][0]['dtype'] != 'i32' or len(rec['results'][0]['shape']) != 1 or
                len(rec['ins']) != 1 or rec['ins'][0]['dtype'] != 'i8' or len(rec['ins'][0]['shape']) != 2 or
                not _is_simple_sum(rec) or not zero_initializer(rec)):
            return None
        body = rec['scalar_body']
        if (body['captures'] or sorted(op['op'] for op in body['operations']) != ['arith.addi','arith.extsi'] or
                any(op['regions'] or op['effects'] not in ([],None) or op['results'] != ['i32'] for op in body['operations'])):
            return None
        maps = [_map_dims(value) for value in rec['indexing_maps']]
        if len(maps) != 2 or len(maps[0]) != 2 or len(maps[1]) != 1:
            return None
        r,p = maps[0]
        if sorted([r,p], key=lambda v: -1 if v is None else v) != [0,1] or maps[1] != [p] or rec['iterator_types'] != ['reduction' if i == r else 'parallel' for i in range(2)]:
            return None
        view = matrix_view(rec['ins'][0])
        if view is None or view[1] or rec['results'][0]['shape'] != [view[3][1]]:
            return None
        return key(core),view

    try:
        if len({key(edge) for edge in returned}) != 3 or any(edge['dtype'] != 'i32' for edge in returned):
            return {}
        contractions = [(edge,contraction(edge)) for edge in returned]
        contractions = [(edge,description) for edge,description in contractions if description is not None]
        biases = [(edge,bias_reduction(edge)) for edge in returned]
        biases = [(edge,description) for edge,description in biases if description is not None]
        if len(contractions) != 2 or len(biases) != 1:
            return {}
        bias_edge,(_,dy) = biases[0]
        matches = []
        for gx_edge,gx in contractions:
            for gw_edge,gw in contractions:
                if gx[0] == gw[0] or gx[1] != dy or gw[2] != dy:
                    continue
                x,w = gw[1],gx[2]
                if not x[1] or not w[1] or len({x[0],w[0],dy[0]}) != 3:
                    continue
                m,n = dy[3]
                if x[2][0] != m or w[2][1] != n or x[2][1] != w[2][0]:
                    continue
                k = x[2][1]
                if gx[3] != (m,k) or gw[3] != (k,n):
                    continue
                if gx_edge['shape'] != root_shape(x[0]) or gw_edge['shape'] != root_shape(w[0]):
                    continue
                matches.append({key(gx_edge):'grad_input',key(gw_edge):'grad_weight',key(bias_edge):'grad_bias'})
        return matches[0] if len(matches) == 1 else {}
    except (ValueError,KeyError,IndexError,TypeError):
        # An unrelated or unproved tuple keeps the ordinary result-order ABI.
        return {}


def _lower_linalg(text):
    inv = parse_linalg_mlir(text)
    source = source_operation_inventory(text)
    cb = {'abi_version':'0.1','target':'gemmini','backend':'agent_gemmini','tensors':{},'commands':[],
          'params':{'linalg_inventory':inv,'source_inventory':source},'_source_text':text}
    tensors = cb['tensors']
    commands = cb['commands']
    for a in inv['args']:
        tensors[f'arg_{a["index"]}'] = {'shape':a['shape'],'dtype':a['dtype'],'role':'input'}
    if inv['returns'] is None:
        raise ValueError('linalg requires a verified single-block return')
    return_keys = [(ret['source'][0], ret['source'][1], ret.get('result_index', 0))
                   for ret in inv['returns']]
    direct_return = any(key[0] != 'op' for key in return_keys)
    repeated_return = len(set(return_keys)) != len(return_keys)
    if direct_return or repeated_return:
        # The pointer ABI requires a unique source-result writer for each
        # output. Preserve the complete boundary when declining an alias or
        # repeated return; do not invent a source operation for an extra copy.
        outputs = []
        for index, result in enumerate(inv['returns']):
            name = f'Y{index}'
            tensors[name] = {'shape':result['shape'], 'dtype':result['dtype'], 'role':'output'}
            outputs.append(name)
        cb['kernel_abi'] = {'kind':'whole_program',
                            'args':[{'tensor':f'arg_{a["index"]}', 'access':'read'} for a in inv['args']] +
                                   [{'tensor':name, 'access':'write'} for name in outputs],
                            'outputs':outputs}
        cb['outputs'] = outputs
        cb['declined'] = {'reason':('direct argument returns lack a source-operation result owner' if direct_return
                                   else 'repeated returns lack independent source-result writers')}
        return cb
    names = {}
    output_names = []
    used_names = set(tensors)

    def unique_tensor_name(stem):
        name = stem
        suffix = 1
        while name in used_names:
            name = f'{stem}_{suffix}'
            suffix += 1
        used_names.add(name)
        return name

    producers = {op['id']:op for op in inv['ops']}
    adjoint_labels = _adjoint_return_labels(inv, text)
    for i, ret in enumerate(inv['returns']):
        key = (ret['source'][0],ret['source'][1],ret.get('result_index',0))
        if key in names:
            raise ValueError('repeated source returns require independently owned outputs')
        if key[0] != 'op':
            raise ValueError('direct argument returns need an explicit output copy owner')
        producer = producers[key[1]]
        stage = producer['prov'].get('composition_stage')
        stem = stage if stage else adjoint_labels.get(key, f'Y{i}')
        if stage and len(producer['results']) > 1:
            stem = f'{stage}_{key[2]}'
        name = unique_tensor_name(stem)
        names[key] = name
        output_names.append(name)
        tensors[name] = {'shape':ret['shape'],'dtype':ret['dtype'],'role':'output'}
    for op in inv['ops']:
        for k,spec in enumerate(op['results']):
            key = ('op',op['id'],k)
            if key not in names:
                names[key] = unique_tensor_name(f'value_{op["id"]}_{k}')
            name = names[key]
            tensors.setdefault(name,{'shape':spec['shape'],'dtype':spec['dtype'],'role':'intermediate'})
    args = [{'tensor':f'arg_{a["index"]}','access':'read'} for a in inv['args']]
    args += [{'tensor':name,'access':'write'} for name in output_names]
    cb['kernel_abi'] = {'kind':'whole_program','args':args,'outputs':output_names}
    cb['outputs'] = cb['kernel_abi']['outputs']
    # The structural reader's payload excludes initialization operations. Match
    # complete source operation indices by their original order and typed records.
    cursor = 0
    init_names = {'tensor.empty','arith.constant','tensor.splat','linalg.fill'}
    source_indices = {}
    for rec in inv['ops']:
        while cursor < len(source['operations']):
            full = source['operations'][cursor]
            if full['operation'] == rec['kind'] and full['results'] == rec['results']:
                break
            if full['operation'] not in init_names:
                raise ValueError('payload/full-source operation order disagreement')
            cursor += 1
        if cursor == len(source['operations']):
            raise ValueError('missing payload operation in full source inventory')
        source_indices[rec['id']] = cursor
        rec['source_op_index'] = cursor
        cursor += 1
    cb['params']['payload_source_op_indices'] = source_indices
    cb['params']['source_values'] = [{'op_index':source_indices[op['id']], 'result_index':k,
                                    'tensor':names[('op',op['id'],k)]}
                                   for op in inv['ops'] for k in range(len(op['results']))]
    cb['params']['constant_tensors'] = {}
    cb['params']['compiler_temporaries'] = []
    unsupported = []

    def input_name(edge):
        kind, idx = edge['source']
        if kind == 'arg':
            return f'arg_{idx}'
        if kind == 'op':
            return names[(kind,idx,edge.get('result_index',0))]
        raise ValueError('nonmaterialized input '+str(edge['source']))

    def append(opcode, operands, attrs, rec):
        index = source_indices[rec['id']]
        attrs = dict(attrs)
        attrs['source_op_index'] = index
        region = rec['prov'].get('region_id')
        if region:
            attrs['source_region_id'] = region
        commands.append({'opcode':opcode,'operands':operands,'attributes':attrs})

    literal_source_ops = None

    def contraction_seed(rec):
        """Retain a contraction's complete initial i32 tensor, never assume zero."""
        nonlocal literal_source_ops
        if len(rec['outs']) != 1:
            raise ValueError('contraction needs exactly one initialized output')
        initial = rec['outs'][0]
        result = rec['results'][0]
        if initial['dtype'] != 'i32' or initial['shape'] != result['shape']:
            raise ValueError('contraction initializer must match the complete i32 result shape')
        origin, producer = initial['source']
        if origin in ('arg', 'op'):
            if origin == 'op' and (not isinstance(producer, int) or producer >= rec['id']):
                raise ValueError('contraction initializer has no preceding materialized producer')
            return input_name(initial)

        literal = initial.get('init_value', initial.get('const_value'))
        values = [literal] if type(literal) is int else None
        if values is None and origin == 'const':
            # Destination-only dense constants are omitted from the payload
            # inventory. Read their actual typed attribute instead of silently
            # dropping the initial accumulator contents.
            if literal_source_ops is None:
                from xdsl.parser import Parser
                module = Parser(make_linalg_context(), text).parse_module()
                definitions = [op for op in module.body.block.ops
                               if op.name == 'func.func' and op.body.blocks]
                if len(definitions) != 1:
                    raise ValueError('literal initializer requires a unique source function')
                literal_source_ops = [op for op in definitions[0].body.blocks[0].ops
                                      if op.name != 'func.return']
            operation = literal_source_ops[source_indices[rec['id']]]
            owner = operation.operands[len(rec['ins'])].owner
            if getattr(owner, 'name', '') == 'arith.constant':
                attribute = {**owner.attributes, **owner.properties}.get('value')
                if hasattr(attribute, 'get_values'):
                    decoded = list(attribute.get_values())
                    if all(type(value) is int for value in decoded):
                        values = decoded
        if values is None:
            raise ValueError('contraction initializer is missing or has opaque contents')
        if len(values) not in (1, prod(result['shape'])):
            raise ValueError('contraction initializer literal extent mismatch')
        values = [((value + (1 << 31)) % (1 << 32)) - (1 << 31) for value in values]
        if all(value == 0 for value in values):
            return None
        name = f'contraction_seed_{rec["id"]}'
        tensors[name] = {'shape':result['shape'], 'dtype':'i32', 'role':'intermediate'}
        edge = source['operations'][source_indices[rec['id']]]['operands'][len(rec['ins'])]['source']
        if not edge or 'op_index' not in edge:
            raise ValueError('literal contraction initializer has no source result owner')
        cb['params']['constant_tensors'][name] = {
            'values':values, 'shape':result['shape'], 'dtype':'i32',
            'source_op_index':edge['op_index'], 'source_result_index':edge['result_index'],
        }
        cb['params']['compiler_temporaries'].append({
            'tensor':name, 'source_op_index':edge['op_index'],
            'source_result_index':edge['result_index'],
            'purpose':'exact literal i32 contraction accumulator initializer',
        })
        return name

    for rec in inv['ops']:
        kind = rec['kind']
        try:
            if len(rec['results']) != 1:
                if kind in ('linalg.generic','linalg.map','linalg.reduce') and all(v['dtype']=='f32' for v in rec['results']) and rec['family'] not in ('contraction','movement'):
                    operands = {('dst' if k==0 else 'dst'+str(k)):names[('op',rec['id'],k)] for k in range(len(rec['results']))}
                    operands.update({f'ins{i}':input_name(edge) for i,edge in enumerate(rec['ins']) if 'const_value' not in edge})
                    append('HOST_LINALG',operands,{'record':rec,'output_dtype':'f32'},rec)
                    append('FENCE',{}, {},rec)
                    continue
                raise ValueError('multiple payload results require a dedicated lowering')
            dst = names[('op',rec['id'],0)]
            spec = rec['results'][0]
            ins = rec['ins']
            kind = rec['kind']
            attrs = rec['attributes']
            integer = spec['dtype'] in ('i8','i32') and all(v['dtype'] in ('i8','i32') for v in ins)
            captured_extract = _captured_extract_attributes(rec) if kind == 'linalg.generic' else None
            if kind == 'arith.constant':
                if spec['dtype'] not in ('i8','i16','i32','i64','f16','bf16','f32','f64') or any(d <= 0 for d in spec['shape']):
                    raise ValueError('literal constant requires positive static extents and supported storage')
                values = _constant_values(rec)
                cb['params']['constant_tensors'][dst] = {'values':values, 'shape':spec['shape'],
                                                        'dtype':spec['dtype'], 'source_op_index':source_indices[rec['id']]}
                # Constant storage is an authored compiler global, not a runtime
                # tensor arithmetic task. The full source owner is retained.
                continue
            elif kind in ('tensor.splat','linalg.fill'):
                if len(ins) != 1 or ins[0]['shape'] != [] or ins[0]['dtype'] != spec['dtype']:
                    raise ValueError('fill/splat requires one scalar of the result storage type')
                if spec['dtype'] not in ('i8','i16','i32','i64','f16','bf16','f32','f64') or any(d <= 0 for d in spec['shape']):
                    raise ValueError('fill/splat requires a static supported storage type')
                if kind == 'linalg.fill' and (len(rec['outs']) != 1 or
                        rec['outs'][0]['shape'] != spec['shape'] or rec['outs'][0]['dtype'] != spec['dtype']):
                    raise ValueError('fill initializer must match its complete result storage')
                if 'const_value' in ins[0]:
                    # The fill is the observable source result owner. Its scalar
                    # constant and empty destination are initializers assigned
                    # to this same device materialization task by attribution.
                    cb['params']['constant_tensors'][dst] = {
                        'values':[ins[0]['const_value']], 'shape':spec['shape'],
                        'dtype':spec['dtype'], 'source_op_index':source_indices[rec['id']]}
                    continue
                if spec['dtype'] not in ('i8','i32','f32'):
                    raise ValueError('live scalar fill/splat needs a supported DMA scalar storage width')
                append('MOVEMENT',{'src':input_name(ins[0]),'dst':dst},
                       {'form':'broadcast','output_dtype':spec['dtype']},rec)
            elif kind in ('quant_ext.quantize_per_tensor','quant_ext.dequantize_per_tensor'):
                operands = {'dst':dst}
                for i,edge in enumerate(ins):
                    if 'const_value' not in edge:
                        operands[f'ins{i}'] = input_name(edge)
                append('HOST_LINALG',operands,{'record':rec,'host_primitive':kind,'output_dtype':spec['dtype']},rec)
            elif kind == 'tensor.pad':
                body = rec['scalar_body']
                if len(ins)!=1 or not body or len(body['operations'])!=1 or body['operations'][0]['op']!='arith.constant' or body['yields']!=['op:0:0'] or body['captures']:
                    raise ValueError('padding region is not a literal constant')
                value = _integer_attr(body['operations'][0]['attributes']['value'])
                low,high = _int_list(attrs['static_low']),_int_list(attrs['static_high'])
                if [a+b+c for a,b,c in zip(ins[0]['shape'],low,high)]!=spec['shape']:
                    raise ValueError('pad source/result extents disagree')
                append('MOVEMENT',{'src':input_name(ins[0]),'dst':dst},
                       {'form':'pad','mode':'constant','low':low,'high':high,'value_code':value,'output_dtype':spec['dtype']},rec)
            elif kind == 'linalg.generic' and _rounded_extension(rec) is not None:
                gain = _rounded_extension(rec)
                narrow = f'extension_narrow_{rec["id"]}'
                wide = f'extension_wide_{rec["id"]}'
                for temp,dtype,purpose in [(narrow,'i8','bounded f32-RNE operand code'),(wide,'i32','widened operand code for accumulator addition')]:
                    tensors[temp] = {'shape':spec['shape'],'dtype':dtype,'role':'intermediate'}
                    cb['params']['compiler_temporaries'].append({'tensor':temp,'source_op_index':source_indices[rec['id']],
                                                                'source_result_index':0,'purpose':purpose})
                append('DEVICE_READOUT',{'src':input_name(ins[1]),'dst':narrow},{'gain_values':[gain],'relu':False,'output_dtype':'i8'},rec)
                append('FENCE',{}, {},rec)
                append('MOVEMENT',{'src':narrow,'dst':wide},{'output_dtype':'i32'},rec)
                append('FENCE',{}, {},rec)
                append('AFFINE_MAP',{'x':input_name(ins[0]),'y':wide,'dst':dst},
                       {'weights':[1,1],'constant':0,'activation':'none','readout_scale':1.0,'output_dtype':'i32'},rec)
            elif kind == 'linalg.generic' and _device_readout(rec) is not None:
                info = _device_readout(rec)
                left,gain = input_name(ins[0]),input_name(ins[1])
                if gain in cb['params']['constant_tensors']:
                    info['gain_values'] = cb['params']['constant_tensors'][gain]['values']
                append('DEVICE_READOUT',{'src':left,'gain':gain,'dst':dst},info,rec)
            elif kind in ('tensor.expand_shape','tensor.collapse_shape','tensor.reshape'):
                if prod(ins[0]['shape']) != prod(spec['shape']) or ins[0]['dtype'] != spec['dtype']:
                    raise ValueError('reshape changes element count/type')
                append('MOVEMENT',{'src':input_name(ins[0]),'dst':dst},{'form':'reshape','output_dtype':spec['dtype']},rec)
            elif kind == 'tensor.extract_slice':
                if len(ins) != 1 or ins[0]['dtype'] != spec['dtype'] or spec['dtype'] not in ('i8','i32','f32'):
                    raise ValueError('extract_slice requires one static storage-preserving tensor input')
                offsets, sizes, strides, kept = _static_slice_geometry(attrs, ins[0]['shape'], spec['shape'])
                row_strides = [prod(ins[0]['shape'][i+1:]) for i in range(len(sizes))]
                append('MOVEMENT',{'src':input_name(ins[0]),'dst':dst},
                       {'form':'as_strided','storage_offset':sum(o*s for o,s in zip(offsets,row_strides)),
                        'strides':[strides[i]*row_strides[i] for i in kept],
                        'output_dtype':spec['dtype']},rec)
            elif kind == 'tensor.insert_slice':
                if len(ins) != 2 or any(v['dtype'] != spec['dtype'] for v in ins) or spec['dtype'] not in ('i8','i32','f32'):
                    raise ValueError('insert_slice requires static storage-preserving source and destination tensors')
                if ins[1]['shape'] != spec['shape']:
                    raise ValueError('insert_slice destination and result extents disagree')
                offsets, sizes, strides, kept = _static_slice_geometry(attrs, spec['shape'], ins[0]['shape'])
                source_name = input_name(ins[0])
                if len(sizes) != len(ins[0]['shape']):
                    # The DMA insertion consumes full-rank coordinates. Unit-axis
                    # expansion preserves the source's linear storage exactly.
                    expanded = unique_tensor_name(f'insert_source_{rec["id"]}')
                    tensors[expanded] = {'shape':sizes,'dtype':spec['dtype'],'role':'intermediate'}
                    cb['params']['compiler_temporaries'].append({
                        'tensor':expanded,'source_op_index':source_indices[rec['id']],
                        'source_result_index':0,'purpose':'rank-expanded insertion source storage'})
                    append('MOVEMENT',{'src':source_name,'dst':expanded},
                           {'form':'reshape','output_dtype':spec['dtype']},rec)
                    append('FENCE',{}, {},rec)
                    source_name = expanded
                append('INSERT_SLICE',{'src':source_name,'dest':input_name(ins[1]),'dst':dst},
                       {'offsets':offsets,'strides':strides,'output_dtype':spec['dtype']},rec)
            elif kind == 'tensor.concat':
                rank = len(spec['shape'])
                axis = _integer_attr(attrs['dim'])
                if not ins or not 0 <= axis < rank or spec['dtype'] not in ('i8','i32','f32'):
                    raise ValueError('concat requires a static in-range axis and supported tensor storage')
                if any(len(v['shape']) != rank or v['dtype'] != spec['dtype'] or
                       any(s < 0 for s in v['shape']) or
                       any(s != d for i,(s,d) in enumerate(zip(v['shape'],spec['shape'])) if i != axis)
                       for v in ins):
                    raise ValueError('concat input ranks, storage types or non-concatenated extents disagree')
                if sum(v['shape'][axis] for v in ins) != spec['shape'][axis]:
                    raise ValueError('concat result extent differs from the sum of its inputs')
                append('CONCAT',{'dst':dst,**{f'src{i}':input_name(edge) for i,edge in enumerate(ins)}},
                       {'axis':axis,'output_dtype':spec['dtype']},rec)
            elif kind == 'linalg.transpose':
                permutation = _int_list(attrs['permutation'])
                if [ins[0]['shape'][i] for i in permutation] != spec['shape']:
                    raise ValueError('transpose result shape disagrees')
                append('MOVEMENT',{'src':input_name(ins[0]),'dst':dst},{'form':'permute','perm':permutation,'output_dtype':spec['dtype']},rec)
            elif captured_extract is not None:
                edge, copy_attrs = captured_extract
                append('MOVEMENT',{'src':input_name(edge),'dst':dst},copy_attrs,rec)
            elif integer and kind == 'linalg.generic' and _is_parallel_product(rec):
                append('DEVICE_PRODUCT',{'x':input_name(ins[0]),'y':input_name(ins[1]),'dst':dst},
                       {'output_dtype':'i32','record':rec},rec)
            elif integer and (kind in ('linalg.matmul','linalg.batch_matmul','linalg.matmul_transpose_a','linalg.matmul_transpose_b') or
                              kind == 'linalg.generic' and _is_simple_product_sum(rec)):
                if any(v['dtype'] != 'i8' for v in ins) or spec['dtype'] != 'i32':
                    raise ValueError('device contraction requires i8 x i8 -> i32')
                if kind != 'linalg.generic' and attrs.get('cast') not in (None,'#linalg.type_fn<cast_signed>'):
                    raise ValueError('named device contraction requires signed input casting')
                if kind in ('linalg.matmul','linalg.matmul_transpose_a','linalg.matmul_transpose_b'):
                    expected = [[2,0] if kind == 'linalg.matmul_transpose_a' else [0,2],
                                [1,2] if kind == 'linalg.matmul_transpose_b' else [2,1],[0,1]]
                    iterators = rec['iterator_types']
                    if not iterators and 'iterator_types' not in attrs:
                        iterators = ['parallel','parallel','reduction']
                    if ([_map_dims(value) for value in rec['indexing_maps']] != expected or
                            iterators != ['parallel','parallel','reduction']):
                        raise ValueError('named contraction requires its canonical indexing maps and iterators')
                seed = contraction_seed(rec)
                seed_operands = {'bias':seed} if seed is not None else {}
                seed_attributes = {'bias':seed} if seed is not None else {}
                left,right = input_name(ins[0]),input_name(ins[1])
                transpose_lhs = kind == 'linalg.matmul_transpose_a'
                transpose_rhs = kind == 'linalg.matmul_transpose_b'
                if kind == 'linalg.generic':
                    maps = [_map_dims(v) for v in rec['indexing_maps']]
                    if (len(maps) != 3 or maps[0] not in ([0,2],[2,0]) or
                            maps[1] not in ([2,1],[1,2]) or maps[2] != [0,1] or
                            rec['iterator_types'] != ['parallel','parallel','reduction']):
                        raise ValueError('contraction indexing maps require a dedicated lowering')
                    transpose_lhs = maps[0] == [2,0]
                    transpose_rhs = maps[1] == [1,2]
                if transpose_lhs or transpose_rhs:
                    if any(len(edge['shape']) != 2 for edge in ins):
                        raise ValueError('transpose contraction requires rank-two operands')
                    lshape = list(reversed(ins[0]['shape'])) if transpose_lhs else ins[0]['shape']
                    rshape = list(reversed(ins[1]['shape'])) if transpose_rhs else ins[1]['shape']
                    if lshape[1] != rshape[0] or spec['shape'] != [lshape[0],rshape[1]]:
                        raise ValueError('transpose contraction dimensions disagree')
                execution_attributes = _contraction_execution_attributes(
                    rec, cb['params']['constant_tensors'], right, transpose_rhs, seed)
                if transpose_lhs:
                    normalized = unique_tensor_name(f'transpose_lhs_{rec["id"]}')
                    tensors[normalized] = {'shape':lshape,'dtype':'i8','role':'intermediate'}
                    cb['params']['compiler_temporaries'].append({
                        'tensor':normalized,'source_op_index':source_indices[rec['id']],
                        'source_result_index':0,'purpose':'canonical LHS layout for transpose contraction'})
                    append('MOVEMENT',{'src':left,'dst':normalized},
                           {'form':'permute','perm':[1,0],'output_dtype':'i8'},rec)
                    append('FENCE',{}, {},rec)
                    left = normalized
                if transpose_rhs:
                    if len(ins[0]['shape']) != 2 or len(ins[1]['shape']) != 2:
                        raise ValueError('batched transpose contraction unsupported')
                    if seed is None:
                        append('ATTENTION_QK',{'q':left,'k':right,'dst':dst},
                               {'output_dtype':'i32','epilogue':[],**execution_attributes},rec)
                    else:
                        # ATTENTION_QK has no initializer operand in the public
                        # ABI. Normalize B with DMA and retain the full seed on
                        # the reviewed COMMIT accumulator-initialization path.
                        normalized = unique_tensor_name(f'transpose_rhs_{rec["id"]}')
                        tensors[normalized] = {'shape':list(reversed(ins[1]['shape'])),
                                               'dtype':'i8','role':'intermediate'}
                        cb['params']['compiler_temporaries'].append({
                            'tensor':normalized,'source_op_index':source_indices[rec['id']],
                            'source_result_index':0,'purpose':'canonical RHS layout for seeded transpose contraction'})
                        append('MOVEMENT',{'src':right,'dst':normalized},
                               {'form':'permute','perm':[1,0],'output_dtype':'i8'},rec)
                        append('FENCE',{}, {},rec)
                        handle = f'weight_handle_{rec["id"]}'
                        accumulator = f'accumulator_{rec["id"]}'
                        append('RES_PACK',{'src':normalized,'dst':handle},{'layout':'packed_rhs'},rec)
                        append('MATMUL_RESIDENT',{'lhs':left,'rhs':handle,'dst':accumulator},execution_attributes,rec)
                        append('COMMIT',{'src':accumulator,'dst':dst,**seed_operands},
                               {'output_dtype':'i32','epilogue':[],**seed_attributes,**execution_attributes},rec)
                        append('EVICT',{'handle':handle},{},rec)
                elif len(ins[0]['shape']) >= 3:
                    append('BATCHED_MATMUL',{'a':left,'w':right,'dst':dst,**seed_operands},
                           {'output_dtype':'i32',**seed_attributes,**execution_attributes},rec)
                else:
                    handle = f'weight_handle_{rec["id"]}'
                    accumulator = f'accumulator_{rec["id"]}'
                    append('RES_PACK',{'src':right,'dst':handle},{'layout':'packed_rhs'},rec)
                    append('MATMUL_RESIDENT',{'lhs':left,'rhs':handle,'dst':accumulator},execution_attributes,rec)
                    append('COMMIT',{'src':accumulator,'dst':dst,**seed_operands},
                           {'output_dtype':'i32','epilogue':[],**seed_attributes,**execution_attributes},rec)
                    append('EVICT',{'handle':handle},{},rec)
            elif integer and kind == 'linalg.generic' and _prefix_axis(rec) is not None:
                append('DEVICE_PREFIX_SUM',{'src':input_name(ins[0]),'dst':dst},
                       {'axis':_prefix_axis(rec),'output_dtype':'i32','record':rec},rec)
            elif integer and kind == 'linalg.generic' and rec['body_ops']==['arith.maxsi'] and rec['reduction_dims']:
                body = rec['scalar_body']
                if len(ins)!=2 or body['operations'][0]['operands'] not in (['arg:0','arg:2'],['arg:2','arg:0']) or body['yields']!=['op:0:0'] or any(v.get('init_value')!=-128 for v in rec['outs']):
                    raise ValueError('generic maximum reduction is not a signed-byte maxpool')
                # Matching max(old, input) establishes the reduction algebra,
                # not a rectangular source window or its direct store-path
                # descriptor. A gather plus a different pooling tree cannot
                # supply the selected source pool_size/pool_stride proof.
                raise ValueError('generic maximum reduction lacks proved direct hardware pool_size/pool_stride geometry')
            elif integer and kind in ('linalg.generic','linalg.map') and not rec['reduction_dims']:
                projected = _integer_projection_body(rec)
                if projected is not None:
                    mode, value = projected
                    if mode == 'constant':
                        # A constant output still needs complete output iterator
                        # coverage; otherwise the source can leave cells unwritten.
                        maps = [_map_dims(v) for v in rec['indexing_maps']]
                        if sorted(maps[-1], key=lambda dim: -1 if dim is None else dim) != list(range(len(rec['iterator_types']))):
                            raise ValueError('constant map output must cover every parallel iterator once')
                        append('FILL',{'dst':dst},{'value_code':value,'output_dtype':spec['dtype']},rec)
                    else:
                        copy_attrs = _projected_copy_attributes(rec, value)
                        append('MOVEMENT',{'src':input_name(ins[value]),'dst':dst},copy_attrs,rec)
                    append('FENCE',{}, {},rec)
                    continue
                coeff,constant,relu = _linear_body(rec)
                live = [i for i,c in enumerate(coeff) if c]
                if not live or len(live)>2:
                    raise ValueError('affine body needs one or two live inputs')
                maps = [_map_dims(v) for v in rec['indexing_maps']]
                output_map = maps[-1]
                for i in live:
                    # Only full identity/leading broadcast align with AFFINE_MAP.
                    if maps[i] != output_map[-len(maps[i]):]:
                        raise ValueError('affine input indexing is not identity/leading broadcast')
                operands = {'x':input_name(ins[live[0]]),'dst':dst}
                if len(live)>1:
                    operands['y'] = input_name(ins[live[1]])
                append('AFFINE_MAP',operands,{'weights':[coeff[i] for i in live],'constant':constant,
                       'activation':'relu' if relu else 'none','readout_scale':1.0,'output_dtype':spec['dtype']},rec)
            elif integer and kind == 'linalg.generic' and _is_simple_sum(rec):
                if spec['dtype'] != 'i32' or ins[0]['dtype'] not in ('i8','i32'):
                    raise ValueError('simple integer sum requires i8/i32 input and an i32 result')
                maps = [_map_dims(v) for v in rec['indexing_maps']]
                input_dims,output_dims = maps
                dims = [i for i,d in enumerate(input_dims) if d not in output_dims]
                if not dims:
                    raise ValueError('no source axis removed by sum reduction')
                # The output operand is the complete initial accumulator,
                # including runtime tensors. Reuse the checked i32 seed reader
                # instead of interpreting a missing literal as zero.
                seed = contraction_seed(rec)
                reduced = dst
                if seed is not None:
                    reduced = unique_tensor_name(f'reduction_raw_{rec["id"]}')
                    tensors[reduced] = {'shape':spec['shape'],'dtype':'i32','role':'intermediate'}
                    cb['params']['compiler_temporaries'].append({
                        'tensor':reduced,'source_op_index':source_indices[rec['id']],
                        'source_result_index':0,'purpose':'unseeded i32 partial for exact initialized sum reduction'})
                append('WINDOW_SUM',{'src':input_name(ins[0]),'dst':reduced},{'mode':'reduce','dims':dims,
                       'average':False,'readout_scale':1.0,'output_dtype':spec['dtype']},rec)
                if seed is not None:
                    append('FENCE',{}, {},rec)
                    append('AFFINE_MAP',{'x':reduced,'y':seed,'dst':dst},
                           {'weights':[1,1],'constant':0,'activation':'none',
                            'readout_scale':1.0,'output_dtype':'i32'},rec)
            elif (kind in ('linalg.generic','linalg.map') and tensors[dst]['role']=='output' and
                  (division := match_quantized_div(rec)) is not None):
                # This boundary owns the standalone division's quantization.
                # Divisions within a larger host island retain its lowering.
                for edge in ins:
                    if 'const_value' not in edge or edge['source'][0]!='op':continue
                    literal=input_name(edge)
                    temporaries=cb['params']['compiler_temporaries']
                    if literal in cb['params']['constant_tensors'] and not any(t['tensor']==literal for t in temporaries):
                        temporaries.append({'tensor':literal,
                                            'source_op_index':cb['params']['constant_tensors'][literal]['source_op_index'],
                                            'source_result_index':0,
                                            'purpose':'literal divisor folded into exact rational device readout'})
                append('DEVICE_QUANTIZED_DIV', {'src':input_name(ins[division['input_index']]), 'dst':dst},
                       {'divisor':division['divisor'], 'entry_scale_exponent':-6,
                        'rounding':'half_even', 'output_dtype':'f32'}, rec)
            elif spec['dtype'] == 'f32' and kind in ('linalg.generic','linalg.map','linalg.reduce') and rec['family'] not in ('contraction','movement'):
                operands = {'dst':dst}
                for i,edge in enumerate(ins):
                    if 'const_value' not in edge:
                        operands[f'ins{i}'] = input_name(edge)
                append('HOST_LINALG',operands,{'record':rec,'output_dtype':'f32'},rec)
            else:
                raise ValueError('unimplemented typed operation '+kind)
            # A subsequent command may read the tensor just written. This is a
            # dependency barrier, required by the selected target contract.
            append('FENCE',{}, {},rec)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            unsupported.append({'source_op_index':source_indices[rec['id']],'operation':kind,'reason':str(exc)})
    if unsupported:
        cb['params']['unsupported_linalg'] = unsupported
        cb['commands'] = []
        cb['declined'] = {'reason':'linalg operations not yet lowered: '+ '; '.join(v['operation']+': '+v['reason'] for v in unsupported)}
    return cb


def parse_source(text):
    if is_linalg_on_tensors(text) or ('func.func' in text and 'merlin_iface.' not in text):
        return _lower_linalg(text)
    model = parse_interface_mlir(text, include_results=True)
    if model['abi_version'] != '0.1':
        raise ValueError('unsupported interface ABI version '+model['abi_version'])
    if not model['target']:
        raise ValueError('interface target is missing')
    model['backend'] = 'agent_gemmini'
    model.setdefault('params',{})
    # Source provenance is metadata, not part of an instruction's numerical
    # attribute vocabulary. In particular WINDOW_SUM validates each mode's
    # semantic attributes strictly. Retain provenance outside those attributes,
    # just as the linalg structural reader already does for the prov namespace.
    for index,command in enumerate(model['commands']):
        attributes = command.get('attributes',{})
        provenance = {key[5:]:attributes.pop(key) for key in list(attributes)
                      if key.startswith('prov.')}
        if provenance:
            model['params'].setdefault('interface_provenance',[]).append({
                'interface_command_index':index,'provenance':provenance})
    return model
