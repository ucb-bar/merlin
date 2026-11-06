"""Authored typed-SSA probes for arithmetic admission proofs."""
import struct
import unittest
from frontend import parse_source
from integer_arithmetic import recognize
from vendor_input.linalg_reader import make_linalg_context
from xdsl.parser import Parser


def program(body, output='f32'):
    return '''module { func.func @forward(%x: tensor<4xi32>) -> tensor<4xOUT> {
      %empty = tensor.empty() : tensor<4xOUT>
      %r = "linalg.generic"(%x, %empty) <{
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>
      }> ({ ^bb0(%a: i32, %old: OUT):
        BODY
        "linalg.yield"(%out) : (OUT) -> ()
      }) : (tensor<4xi32>, tensor<4xOUT>) -> tensor<4xOUT>
      func.return %r : tensor<4xOUT>
    }}'''.replace('OUT',output).replace('BODY',body)


ZERO_PREFIX = '''%f = arith.sitofp %a : i32 to f32
    %negative = arith.constant -0.0 : f32
    %z = arith.mulf %f, %negative : f32'''


class ArithmeticAdmission(unittest.TestCase):
    def test_arithmetic_identity_keeps_storage_copies_with_movement(self):
        cb = parse_source(program('%zero = arith.constant 0 : i32\n%out = arith.addi %a, %zero : i32','i32'))
        self.assertNotIn('declined',cb)
        self.assertEqual(cb['commands'][-1]['opcode'],'DEVICE_INTEGER_AFFINE')
        direct = program('','i32').replace('"linalg.yield"(%out)','"linalg.yield"(%a)')
        module = Parser(make_linalg_context(),direct).parse_module()
        generic = next(op for op in module.walk() if op.name == 'linalg.generic')
        self.assertIsNone(recognize(generic))

    def test_fma_literal_negative_zero(self):
        body = ZERO_PREFIX+'''\n%one = arith.constant 1.0 : f32
        %out = math.fma %negative, %one, %z : f32'''
        cb = parse_source(program(body))
        self.assertNotIn('declined',cb)
        stages = cb['commands'][-1]['attributes']['stages']
        self.assertEqual([kind for kind,_ in stages],['mul','add'])
        self.assertTrue(all(struct.pack('<f',value)==struct.pack('<f',-0.) for _,value in stages))

    def test_select_distinguishes_zero_signs(self):
        body = ZERO_PREFIX+'''\n%positive = arith.constant 0.0 : f32
        %yes = arith.addf %z, %negative : f32
        %no = arith.addf %z, %positive : f32
        %index = linalg.index 0 : index
        %two = arith.constant 2 : index
        %condition = arith.cmpi slt, %index, %two : index
        %out = arith.select %condition, %yes, %no : f32'''
        self.assertIn('declined',parse_source(program(body)))

    def test_nonfinite_literal_does_not_acquire_finite_proof(self):
        body = '''%f = arith.sitofp %a : i32 to f32
        %nan = arith.constant 0x7FC00000 : f32
        %condition = arith.cmpf uno, %nan, %nan : f32
        %one = arith.constant 1.0 : f32
        %zero = arith.constant 0.0 : f32
        %offset = arith.select %condition, %one, %zero : f32
        %out = arith.addf %f, %offset : f32'''
        self.assertIn('declined',parse_source(program(body)))


if __name__ == '__main__':
    unittest.main()
