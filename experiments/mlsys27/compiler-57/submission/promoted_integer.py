"""Native f32 arithmetic fed solely by signed integer storage.

Boolean mesh/DMA circuits retain all input bits, round integer conversion to
24 significant bits, and form IEEE storage words. Floating storage islands
retain their separate declared quantized model.
"""
from fractions import Fraction
import struct
from math import isfinite, prod
from quantized_float import _scratch, _literal, _load, _store, _copy, _linear
from rtl_isa import DIM


def _f32(value):
    # Convert integer literals directly to 24 significant bits. Passing a wide
    # integer through Python's binary64 first can introduce a second-round tie.
    if isinstance(value, int):
        shift = max(0, abs(value).bit_length()-24)
        value = round(Fraction(value, 1 << shift))*(1 << shift)
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def _power(value):
    value = abs(Fraction(value))
    if not value or value.numerator & (value.numerator-1) or value.denominator & (value.denominator-1):
        raise ValueError('native promotion multiplier must be a power of two')
    return value.numerator.bit_length()-value.denominator.bit_length()


def _same_native_expression(left, right):
    if isinstance(left, tuple) or isinstance(right, tuple):
        return (isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right)
                and all(a == b and struct.pack('<f', x) == struct.pack('<f', y)
                        for (a,x),(b,y) in zip(left,right)))
    return struct.pack('<f', left) == struct.pack('<f', right)


def _exact_literal_product(left, right):
    product = _f32(left*right)
    if Fraction(product) != Fraction(left)*Fraction(right):
        raise ValueError('literal FMA product must be exactly representable')
    return product


def recognize(op):
    from frontend import _spec, _attr, _int, _const, Unsupported
    from xdsl.ir.affine import AffineDimExpr
    if op.name != 'linalg.generic' or len(op.results) != 1 or _spec(op.results[0])['dtype'] != 'f32':
        return None
    inputs = list(op.inputs)
    runtime = []
    environment = {}
    for argument, tensor in zip(op.regions[0].block.args, inputs):
        try:
            literal = _f32(_const(tensor))
            if not isfinite(literal):
                return None
            environment[argument] = literal
        except Unsupported:
            runtime.append((argument, tensor))
    if len(runtime) != 1 or _spec(runtime[0][1])['dtype'] not in ('i8', 'i32'):
        return None
    argument, source = runtime[0]
    if _spec(source)['shape'] != _spec(op.results[0])['shape']:
        return None
    rank = len(_spec(source)['shape'])
    maps = list(_attr(op, 'indexing_maps', []))
    identity = tuple(AffineDimExpr(i) for i in range(rank))
    if (len(maps) != len(inputs)+1 or any('parallel' not in str(i) for i in _attr(op, 'iterator_types', []))
            or any(tuple(maps[i].data.results) != identity for i in (inputs.index(source), len(inputs)))):
        return None
    # A runtime expression is an ordered list of native rounding stages.
    def evaluate(value):
        if value in environment:
            return environment[value]
        scalar = value.owner
        name = getattr(scalar, 'name', None)
        if name == 'arith.constant':
            result = _const(value)
        elif name == 'arith.sitofp':
            operand = scalar.operands[0]
            result = () if operand is argument else _f32(evaluate(operand))
        elif name == 'arith.cmpf' and scalar.operands[0] is scalar.operands[1]:
            # All admitted native integer conversions and finite literal
            # affine stages are finite, so unordered(x,x) is false.
            evaluate(scalar.operands[0])
            predicate = _int(_attr(scalar, 'predicate'))
            result = {0:False,1:True,2:False,3:True,4:False,5:True,6:False,
                      7:True,8:True,9:False,10:True,11:False,12:True,13:False,
                      14:False,15:True}[predicate]
        elif name == 'arith.select':
            try:
                predicate = evaluate(scalar.operands[0])
            except (ValueError, KeyError, Unsupported):
                predicate = None
            if isinstance(predicate, (bool, int)):
                result = evaluate(scalar.operands[1 if predicate else 2])
            else:
                yes, no = (evaluate(v) for v in scalar.operands[1:])
                if not _same_native_expression(yes,no):
                    raise ValueError('data/index select has distinct native expressions')
                result = yes
        elif name == 'math.fma':
            left, right, addend = (evaluate(v) for v in scalar.operands)
            if isinstance(left, tuple) or isinstance(right, tuple):
                raise ValueError('runtime native FMA multiplication is not admitted')
            result = append(addend, 'add', _exact_literal_product(left,right))
        elif name in ('arith.addf','arith.subf','arith.mulf','arith.divf'):
            left, right = (evaluate(v) for v in scalar.operands)
            if not isinstance(left, tuple) and not isinstance(right, tuple):
                result = _f32({'arith.addf':lambda:left+right, 'arith.subf':lambda:left-right,
                               'arith.mulf':lambda:left*right, 'arith.divf':lambda:left/right}[name]())
            elif name == 'arith.addf':
                result = append(left, 'add', right) if isinstance(left, tuple) else append(right, 'add', left)
            elif name == 'arith.subf' and isinstance(left, tuple):
                result = append(left, 'add', -right)
            elif name == 'arith.mulf':
                result = append(left, 'mul', right) if isinstance(left, tuple) else append(right, 'mul', left)
            elif name == 'arith.divf' and isinstance(left, tuple):
                _power(right)
                result = append(left, 'mul', 1/right)
            else:
                raise ValueError('unsupported runtime native affine stage')
        else:
            raise ValueError('unsupported native promotion scalar operation')
        if isinstance(result, float) and not isfinite(result):
            raise ValueError('native promotion requires finite literal expressions')
        environment[value] = result
        return result
    def append(expression, kind, constant):
        if not isinstance(expression, tuple) or isinstance(constant, tuple) or not isfinite(constant):
            raise ValueError('native stage needs one finite literal')
        if kind == 'mul' and constant:
            if abs(_power(constant)) > 16:
                raise ValueError('native scale exceeds bounded circuit exponent')
        if kind == 'add':
            fraction = Fraction(constant)
            if abs(fraction) > 65536 or fraction.denominator > 65536:
                raise ValueError('native offset exceeds bounded exact fixed-point circuit')
        return expression+((kind, float(constant)),)
    try:
        expression = evaluate(list(op.regions[0].block.ops)[-1].operands[0])
        if not isinstance(expression, tuple) or len(expression) > 4:
            return None
    except (ValueError, KeyError, Unsupported, ZeroDivisionError, OverflowError):
        return None
    return source, {'stages': [[kind, value] for kind,value in expression], 'output_dtype':'f32'}


class _Circuit:
    def __init__(self, emitter, state, rows):
        self.e, self.state, self.rows = emitter, state, rows
        self.cache = {}

    def linear(self, terms, bias=0, shift=0, relu=False):
        terms = tuple((pointer, weight) for pointer, weight in terms if weight and pointer != 0)
        bias += sum(pointer*weight for pointer,weight in terms if isinstance(pointer,int))
        terms = tuple((pointer,weight) for pointer,weight in terms if not isinstance(pointer,int))
        if not terms:
            value = round(Fraction(bias, 1 << shift))
            return max(0,value) if relu else value
        if len(terms) == 1 and terms[0][1] == 1 and bias == 0 and shift == 0 and not relu:
            return terms[0][0]
        key = terms, bias, shift, relu
        if key in self.cache:
            return self.cache[key]
        e, rows = self.e, self.rows
        _load(e, 0 if bias == 0 else _literal(e, self.state, [bias], 'i32'), rows, stride=0, dtype='i32')
        for pointer, weight in terms:
            _load(e, pointer, rows, add=True, scale=weight)
        result = _scratch(e, rows)
        _store(e, result, rows, scale=2.0**-shift, relu=relu)
        self.cache[key] = result
        return result

    def invert(self, bit):
        return self.linear([(bit,-1)], bias=1)

    def both(self, left, right):
        if left == right:
            return left
        if isinstance(left,int):
            return right if left else 0
        if isinstance(right,int):
            return left if right else 0
        return self.linear([(left,1),(right,1)],bias=-1,relu=True)

    def either(self, left, right):
        if left == right:
            return left
        if isinstance(left,int):
            return 1 if left else right
        if isinstance(right,int):
            return 1 if right else left
        return self.linear([(left,2),(right,2)],bias=1,shift=2)

    def xor(self, left, right):
        if left == right:
            return 0
        if isinstance(left,int):
            return self.invert(right) if left else right
        if isinstance(right,int):
            return self.invert(left) if right else left
        return self.linear([(left,1),(right,1),(self.both(left,right),-2)])

    def add(self, left, right, carry=0):
        result = []
        for a,b in zip(left,right):
            if a == 0:
                next_carry = self.both(b,carry)
            elif b == 0:
                next_carry = self.both(a,carry)
            elif a == 1:
                next_carry = self.either(b,carry)
            elif b == 1:
                next_carry = self.either(a,carry)
            elif carry == 0:
                next_carry = self.both(a,b)
            elif carry == 1:
                next_carry = self.either(a,b)
            else:
                next_carry = self.linear([(a,2),(b,2),(carry,2)],bias=-1,shift=2)
            result.append(self.linear([(a,1),(b,1),(carry,1),(next_carry,-2)]))
            carry = next_carry
        return result

    def negate_if(self, bits, sign):
        return self.add([self.xor(bit,sign) for bit in bits],[0]*len(bits),sign)

    def extract(self, source, width):
        bits = []
        for byte in range(width):
            pointer = _scratch(self.e, self.rows)
            _copy(self.e, self.e.add(source,byte), pointer, self.rows, 1, width, 1)
            quotients = [pointer]
            for shift in range(1,8):
                quotients.append(self.linear([(pointer,1),(pointer,1)],bias=-(2**shift-1),shift=shift+1))
            bits.extend(self.linear([(quotients[i],1),(quotients[i+1],-1),(quotients[i+1],-1)]) for i in range(7))
            bits.append(self.linear([(quotients[7],-1)]))
        return bits

    def leading(self, bits):
        high = 0
        result = [0]*len(bits)
        for index in reversed(range(len(bits))):
            result[index] = self.both(bits[index], self.invert(high))
            high = self.either(high,bits[index])
        return result

    def round_significand(self, bits):
        # Preserve 24 leading bits; guard, sticky and retained parity implement
        # RNE over the complete unsigned domain, including carry into exponent.
        if len(bits) <= 24:
            return list(bits)
        leading = self.leading(bits)
        increments = [0]*(len(bits)+1)
        sticky = 0
        for shift in range(1,len(bits)-23):
            condition = self.both(bits[shift-1], self.either(sticky,bits[shift]))
            increments[shift] = self.both(leading[shift+23],condition)
            sticky = self.either(sticky,bits[shift-1])
        base = []
        high = 0
        suffix = [0]*(len(bits)+1)
        for index in reversed(range(len(bits))):
            high = self.either(high,bits[index])
            suffix[index] = high
        for index,bit in enumerate(bits):
            base.append(self.both(bit,self.invert(suffix[index+24])) if index+24<len(suffix) else bit)
        return self.add(base+[0],increments)

    def word_bits(self, magnitude, sign, point):
        rounded = self.round_significand(magnitude)
        leading = self.leading(rounded)
        output = []
        for position in range(23):
            terms = []
            for exponent,active in enumerate(leading):
                source = exponent-23+position
                if 0 <= source < exponent:
                    terms.append((self.both(active,rounded[source]),1))
            output.append(self.linear(terms))
        for position in range(8):
            output.append(self.linear([(active,1) for exponent,active in enumerate(leading)
                                       if (exponent-point+127) & (1 << position)]))
        output.append(sign)
        return output

    def encode(self, magnitude, sign, point, destination):
        output = self.word_bits(magnitude,sign,point)
        for byte in range(4):
            value = self.linear([(output[byte*8+bit],2**bit if bit<7 else -128) for bit in range(8)])
            if isinstance(value,int):
                # The RTL zero-load tracker accounts accumulator words. Use
                # the full-width zero source even for a final byte readout.
                _copy(self.e, 0 if value == 0 else _literal(self.e,self.state,[value]), self.e.add(destination,byte),
                      self.rows,1,0,4,'i32' if value == 0 else 'i8')
            else:
                _copy(self.e,value,self.e.add(destination,byte),self.rows,1,1,4)


def _constant_bits(value, count):
    return [(value >> index) & 1 for index in range(count)]


def _tile(e, state, source, destination, rows, width, stages):
    circuit = _Circuit(e,state,rows)
    if stages and stages[0][0] == 'mul' and stages[0][1] == 0:
        high = _scratch(e,rows)
        sign = _scratch(e,rows)
        _copy(e,e.add(source,width-1),high,rows,1,width,1)
        _linear(e,state,high,[-1],sign,rows,1,flag=True)
        magnitude = [0]
    else:
        integer = circuit.extract(source,width)
        sign = integer[-1]
        magnitude = circuit.negate_if(integer,sign)
        magnitude = circuit.round_significand(magnitude)
    point = 0
    for kind,value in stages:
        if kind == 'mul':
            sign = circuit.invert(sign) if struct.pack('<f',value)[3] & 128 else sign
            if value:
                point -= _power(value)
            else:
                magnitude = [0]
        else:
            # Addition of -0 preserves every finite native word, including -0.
            if value == 0 and struct.pack('<f',value)[3] & 128:
                continue
            fraction = Fraction(value)
            next_point = max(point,fraction.denominator.bit_length()-1)
            shift = next_point-point
            count = max(len(magnitude)+shift+2,abs(int(fraction*(1 << next_point))).bit_length()+2)
            signed = circuit.negate_if([0]*shift+magnitude+[0]*(count-len(magnitude)-shift),sign)
            added = circuit.add(signed,_constant_bits(int(fraction*(1 << next_point)),count))
            sign = added[-1]
            magnitude = circuit.negate_if(added,sign)
            magnitude = circuit.round_significand(magnitude)
            point = next_point
    circuit.encode(magnitude,sign,point,destination)


def lower(e, command, state):
    if command['opcode'] != 'DEVICE_PROMOTED_INTEGER':
        return False
    src,dst = (command['operands'][key] for key in ('src','dst'))
    width = 1 if e.tensors[src]['dtype'] == 'i8' else 4
    count = prod(e.tensors[src]['shape'])
    stages = command['attributes']['stages']
    def tile(start,rows):
        _tile(e,state,e.ptr(src,e.mul(start,width)),e.ptr(dst,e.mul(start,4)),rows,width,stages)
    full = count//DIM*DIM
    if full:
        e.loop(0,full,DIM,lambda start:tile(start,DIM))
    if full < count:
        tile(full,count-full)
    return True
