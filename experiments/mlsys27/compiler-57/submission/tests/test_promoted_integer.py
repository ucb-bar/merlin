"""IEEE guard/sticky/parity and signed bit arithmetic, independent of captures."""
import random
import struct
import unittest
from fractions import Fraction
from promoted_integer import (_Circuit, _constant_bits, _f32,
                              _same_native_expression, _exact_literal_product)


def encode_fraction(value):
    value = Fraction(value)
    point = value.denominator.bit_length()-1
    magnitude = _constant_bits(abs(value.numerator),max(1,abs(value.numerator).bit_length()))
    bits = _Circuit(None,{},0).word_bits(magnitude,int(value < 0),point)
    return sum(bit << index for index,bit in enumerate(bits))


class NativeRounding(unittest.TestCase):
    def test_ieee_rounding_boundaries(self):
        values = set(range(-256,257)) | {-(1 << 31), (1 << 31)-1}
        for power in range(8,32):
            values.update(sign*((1 << power)+offset) for offset in range(-17,18) for sign in (-1,1)
                          if -(1 << 31) <= sign*((1 << power)+offset) < (1 << 31))
        rng = random.Random(8193)
        values.update(rng.randrange(-(1 << 31),1 << 31) for _ in range(1000))
        for integer in sorted(values):
            exact_values = [Fraction(integer)]
            exact_values += [Fraction(_f32(integer))*factor for factor in (-2,Fraction(-1,2),0,Fraction(1,2),1,2)]
            exact_values += [Fraction(_f32(integer))+offset for offset in (-2,-1,Fraction(-1,2),0,Fraction(1,2),1,2)]
            for exact in exact_values:
                expected = struct.unpack('<I',struct.pack('<f',float(exact)))[0]
                self.assertEqual(encode_fraction(exact),expected,(integer,exact))

    def test_wide_integer_literal_rounds_once(self):
        midpoint_successor = (1 << 62)+(1 << 38)+1
        expected = float((1 << 62)+(1 << 39))
        self.assertEqual(_f32(midpoint_successor),expected)
        self.assertEqual(_f32(-midpoint_successor),-expected)

    def test_native_zero_sign_proofs(self):
        negative_zero = struct.pack('<f',-0.)
        self.assertEqual(struct.pack('<f',_exact_literal_product(-0.,1.)),negative_zero)
        self.assertEqual(struct.pack('<f',_exact_literal_product(0.,-1.)),negative_zero)
        self.assertFalse(_same_native_expression((('add',-0.),),(('add',0.),)))
        self.assertFalse(_same_native_expression(-0.,0.))
        self.assertTrue(_same_native_expression((('add',-0.),),(('add',-0.),)))

    def test_signed_bit_arithmetic(self):
        circuit = _Circuit(None,{},0)
        for width in (8,32,40):
            mask = (1 << width)-1
            values = [0,1,127,128,255,256,mask//2,mask//2+1,mask]
            for value in values:
                bits = _constant_bits(value,width)
                negative = circuit.negate_if(bits,1)
                self.assertEqual(sum(bit << i for i,bit in enumerate(negative)),(-value)&mask)
                for addend in values:
                    result = circuit.add(bits,_constant_bits(addend,width))
                    self.assertEqual(sum(bit << i for i,bit in enumerate(result)),(value+addend)&mask)


if __name__ == '__main__':
    unittest.main()
