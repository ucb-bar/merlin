"""Result failures remain sticky and launch accounting gates final success."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('result_transport', ROOT / 'tools/chipyard/result_transport.py')
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class ResultTest(unittest.TestCase):
    def test_result_gate(self):
        source = r'''
#include <stdio.h>
#include "atlas_result.h"
static void output(void *context, char value) { (void)context; putchar(value); }
int main(void) {
  atlas_result s;
  atlas_result_begin(&s, output, 0, "positive", 2);
  atlas_result_progress(&s, 1, "layer0"); atlas_result_progress(&s, 2, "layer1");
  if (!atlas_result_pass(&s)) return 1;
  atlas_result_begin(&s, output, 0, "negative", 1);
  atlas_result_progress(&s, 1, "layer0");
  atlas_result_check(&s, "output-byte", 7, 8);
  if (atlas_result_pass(&s)) return 2;
  atlas_result_begin(&s, output, 0, "incomplete", 2);
  atlas_result_progress(&s, 1, "layer0");
  if (atlas_result_pass(&s)) return 3;
  atlas_result_begin(&s, output, 0, "empty", 0);
  if (atlas_result_pass(&s)) return 4;
  atlas_result_begin(&s, output, 0, "order", 2);
  atlas_result_progress(&s, 1, "layer0"); atlas_result_progress(&s, 1, "layer0");
  atlas_result_progress(&s, 2, "layer1");
  if (atlas_result_pass(&s)) return 5;
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'check.c').write_text(source)
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                            str(out / 'check.c'), '-o', str(out / 'check')], check=True)
            result = subprocess.run([str(out / 'check')], check=True, capture_output=True, text=True).stdout
        self.assertEqual(result.count('ATLAS_KERNEL_PASS'), 1)
        self.assertIn('ATLAS_KERNEL_PASS name=positive launches_completed=2 launches_expected=2', result)
        self.assertIn('check=output-byte observed=7 expected=8', result)
        self.assertIn('check=launch-count observed=1 expected=2', result)
        self.assertIn('check=launch-order', result)

    def test_transport_configuration(self):
        parser = argparse.ArgumentParser()
        transport.add_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            profile = transport.write_config(out, parser.parse_args([]))
            self.assertEqual(profile['base'], 0x10020000)
            self.assertIsNone(profile['divisor'])
            header = (out / 'atlas_result_transport.h').read_text()
            self.assertNotIn('base + 24', header)
            self.assertIn('base + 8', header)
            with self.assertRaises(ValueError):
                transport.write_config(out, parser.parse_args(['--uart-kind', 'ns16550']))
            profile = transport.write_config(out, parser.parse_args(
                ['--uart-kind', 'ns16550', '--uart-base', '0x10000000']))
            self.assertEqual(profile['kind'], 'ns16550')
            self.assertNotIn('regs[3]', (out / 'atlas_result_transport.h').read_text())
            for value in ('0', '-1', '0x10001'):
                with self.assertRaises(ValueError):
                    transport.write_config(out, parser.parse_args(['--uart-divisor', value]))


if __name__ == '__main__':
    unittest.main()
