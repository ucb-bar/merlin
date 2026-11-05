import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / 'src/merlin_timing/rocket_alu.py'
spec = importlib.util.spec_from_file_location('rocket_alu', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SelectedRocketAlu(unittest.TestCase):
    def test_selected_source_custody_and_missing_hardware_extensions(self):
        # Synthetic source signatures test custody/admission, not hardware.
        texts = {
            'selected_isa': json.dumps({'cpus': {'cpu@0': {'riscv,isa': ['rv64imafdc_zba_zbb_zbs']}}}),
            'decode': 'class ZbaDecode class ZbbDecode class ZbsDecode SH1ADD A1_RS1SHL FN_ADD FN_UNARY FN_BEXT',
            'alu': 'io.out := out val out = MuxLookup if (coreParams.useZbb) FN_UNARY -> unary FN_ADD -> io.adder_out',
            'pipeline': 'if (coreParams.useZba) if (coreParams.useZbb) if (coreParams.useZbs) mem_reg_valid := !ctrl_killx mem_reg_wdata := Mux(ex_reg_set_vconfig alu.io.out) ex_waddr, mem_reg_wdata',
            'rtl': 'module Rocket module RocketALU',
        }
        model = {'parameters': {}, 'cpu': {'isa': 'rv64imafdc_zba_zbb_zbs'},
                 'cpu_execution_resources': {'cpu.bitmanip': 'integer', 'cpu.integer': 'integer'}}
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as work:
            def pins():
                result = {}
                for key, text in texts.items():
                    path = Path(work) / key
                    path.write_text(text)
                    result[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                return result
            original = pins()
            selected, receipt = module.select_bitmanip_alu(model, **original)
            self.assertNotIn('bitmanip_alu_service', model['cpu'])
            self.assertEqual(selected['parameters'], model['parameters'])
            self.assertEqual(selected['cpu']['bitmanip_alu_service']['execute_to_memory_stages'], 1)
            self.assertFalse(receipt['activation'])
            # A permissive native ISA cannot stand in for missing selected RTL extensions.
            texts['selected_isa'] = json.dumps({'cpus': {'cpu@0': {'riscv,isa': ['rv64imafdc']}}})
            with self.assertRaisesRegex(ValueError, 'selected elaboration'):
                module.select_bitmanip_alu(model, **pins())
            # Changed source bytes are refused before recognizing source equations.
            Path(original['alu']['path']).write_text('different source')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                module.select_bitmanip_alu(model, **original)


if __name__ == '__main__':
    unittest.main()
