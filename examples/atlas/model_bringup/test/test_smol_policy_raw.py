"""Actual reference pack/unpack/bias controls with an explicit mock transport.

The mock validates staged/program bytes, injects arithmetic reference output,
and exercises ownership completion. No device/model execution is claimed.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import sys
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked as safe


class Panel(ct.Structure):
    _fields_ = [(name, ct.c_void_p if name in ('program', 'weights') else ct.c_size_t) for name in
        ('program', 'weights', 'program_words', 'rescale_word', 'physical_rows', 'physical_k', 'physical_n',
         'row_start', 'logical_rows', 'input_width', 'output_width', 'call', 'output_offset', 'dram_bytes',
         'weight_bytes', 'load_bytes', 'store_bytes')] + [('activation_scale', ct.c_int), ('weight_scale', ct.c_int)]


class Call(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ('rows', 'input_width', 'output_width', 'first_panel', 'panel_count')]
    _fields_ += [('bias', ct.c_void_p)] + [(name, ct.c_uint) for name in
        ('site', 'layer', 'role', 'image', 'input_dtype', 'output_dtype', 'bias_dtype')] + [('activation_scale', ct.c_int)]


class Buffer(ct.Structure):
    _fields_ = [('address', ct.c_size_t), ('size_bytes', ct.c_size_t), ('access', ct.c_int)]


class Context(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ('host', 'panels', 'calls')]
    _fields_ += [(name, ct.c_size_t) for name in ('panel_count', 'call_count', 'panel_cursor', 'call_cursor')]
    _fields_ += [(name, ct.c_void_p) for name in ('dram', 'activation_copy', 'input_copy')]
    _fields_ += [(name, ct.c_size_t) for name in ('dram_capacity', 'activation_capacity', 'input_capacity', 'max_polls')]
    _fields_ += [('cycle_delta', ct.c_void_p), ('active_buffer', Buffer), ('image', ct.c_uint), ('failed', ct.c_uint), ('result', ct.c_int)]
    _fields_ += [('attention_panels', ct.c_void_p), ('attention_count', ct.c_size_t), ('attention_cursor', ct.c_size_t)]


MOCK = r'''
#include "smol_policy_raw.h"
#include <string.h>
size_t unit_context_bytes(void) { return sizeof(smol_policy_raw_context); }
size_t unit_attention_offset(void) { return offsetof(smol_policy_raw_context,attention_panels); }
static atlas_host host;
static unsigned char *dram;
static const unsigned char **initials, **outputs, **programs;
static size_t *sizes, *offsets, *program_sizes, count, cursor;
static uint32_t ticks;
static uint32_t read32(void *p, uintptr_t a) { (void)p; (void)a; return ticks; }
void *unit_config(unsigned char *d, const unsigned char **i, const unsigned char **o,
                  const unsigned char **p, size_t *s, size_t *f, size_t *z, size_t n) {
  memset(&host,0,sizeof(host)); host.mmio.read32=read32; host.map.csr_base=0x40000;
  dram=d; initials=i; outputs=o; programs=p; sizes=s; offsets=f; program_sizes=z;
  count=n; cursor=0; ticks=100; return &host;
}
size_t unit_completed(void) { return cursor; }
atlas_host_result atlas_host_load_raw(atlas_host *h,const uint32_t *p,size_t n) {
  if(h!=&host||cursor>=count||n*4!=program_sizes[cursor]||memcmp(p,programs[cursor],n*4)) return ATLAS_HOST_INVALID;
  h->program_loaded=1; return ATLAS_HOST_OK;
}
atlas_host_result atlas_host_start(atlas_host *h,const atlas_host_buffer *b,size_t n) {
  if(h!=&host||cursor>=count||n!=1||b->address!=(uintptr_t)dram||b->size_bytes!=sizes[cursor]
     ||memcmp(dram,initials[cursor],sizes[cursor])) return ATLAS_HOST_INVALID;
  h->active=1; h->active_buffers=b; h->active_buffer_count=1; ++h->invocation_sequence; return ATLAS_HOST_OK;
}
atlas_host_result atlas_host_wait(atlas_host *h,size_t polls) {
  if(h!=&host||!polls||!h->active) return ATLAS_HOST_INVALID;
  memcpy(dram+offsets[cursor],outputs[cursor],sizes[cursor]-offsets[cursor]-256);
  h->active=0; h->active_buffers=0; h->active_buffer_count=0; ticks+=100; ++cursor; return ATLAS_HOST_OK;
}
'''


@unittest.skipUnless(all(os.environ.get(name) for name in ('SMOL_REPLAY', 'SMOL_KERNEL_ASSETS', 'SMOL_RUNTIME')),
                     'Optional staged policy transport fixtures are not configured')
class PolicyRawTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.assets, runtime = [safe(os.environ[name]) for name in ('SMOL_REPLAY', 'SMOL_KERNEL_ASSETS', 'SMOL_RUNTIME')]
        cls.receipt = json.loads(safe(cls.root / 'receipt.json').read_bytes())
        cls.manifest = json.loads(safe(cls.assets / 'kernels.json').read_bytes())
        if cls.receipt['status'] != 'PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY' or cls.manifest['source_bookends_equal'] is not True:
            raise ValueError('Require completed target reference and staged source bookends')
        here = safe(os.environ['SMOL_MERLIN'])
        cls.temporary = tempfile.TemporaryDirectory(prefix='policy-raw-')
        cls.addClassCleanup(cls.temporary.cleanup)
        temporary = Path(cls.temporary.name)
        mock, library = temporary / 'mock.c', temporary / 'raw.so'
        mock.write_text(MOCK)
        subprocess.run(['/usr/bin/cc', '-O2', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math',
                        '-I'+str(runtime), '-I'+str(here), str(ROOT / 'runtime/smol_policy_raw.c'),
                        str(here / 'host_matrix_fp8pack.c'), str(mock), '-lm', '-o', str(library)], check=True)
        cls.owner = ct.CDLL(str(library))
        cls.owner.unit_context_bytes.restype = ct.c_size_t
        cls.owner.unit_attention_offset.restype = ct.c_size_t
        if (ct.sizeof(Context) != cls.owner.unit_context_bytes()
            or Context.attention_panels.offset != cls.owner.unit_attention_offset()):
            raise ValueError('ctypes raw context differs from compiled C ABI')
        cls.owner.unit_config.restype = ct.c_void_p
        cls.owner.unit_config.argtypes = [ct.c_void_p]*7 + [ct.c_size_t]
        cls.owner.unit_completed.restype = ct.c_size_t
        cls.owner.smol_policy_raw_graph.argtypes = [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint,
            ct.c_uint, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_uint, ct.c_void_p, ct.c_size_t]
        cls.owner.smol_policy_raw_vision.argtypes = [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
            ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t]

    def payload(self, name):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts: raise ValueError('Invalid staged member')
        raw = safe(self.assets / path).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), path.stem)
        return np.frombuffer(raw, np.uint8).copy()

    def tensor(self, row):
        path = Path(row['file'])
        if path.is_absolute() or '..' in path.parts: raise ValueError('Invalid reference member')
        raw = safe(self.root / path).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
        return np.frombuffer(raw, '<u2' if row['dtype'] == 'torch.bfloat16' else '<f4').copy().reshape(row['shape'])

    def case(self, module, mutation=None):
        call_index = next(row['index'] for row in self.receipt['calls'] if row['module'] == module)
        record = self.receipt['calls'][call_index]
        rows = [row for row in self.manifest['kernels'] if row['call'] == call_index]
        incoming, expected = self.tensor(record['input']), self.tensor(record['output'])
        incoming = incoming.reshape(-1, incoming.shape[-1])
        expected = expected.reshape(-1, expected.shape[-1])
        vision = '.vision_model.' in module
        site, layer, role = (7, 0, 1) if vision else (2, 0, 0)
        dtype = 0 if incoming.dtype == np.uint16 else 1
        output_dtype = 0 if expected.dtype == np.uint16 else 1
        panels, initials, outputs, programs, keep = (Panel*len(rows))(), [], [], [], []
        for index, row in enumerate(rows):
            a, w, output, program = [self.payload(row[name]) for name in ('activation', 'weights', 'expected', 'original')]
            initial = np.concatenate([a, w, np.full(256, 0xa5, np.uint8), np.full(output.size, 0xcd, np.uint8), np.full(256, 0x5a, np.uint8)])
            self.assertEqual(hashlib.sha256(initial.tobytes()).hexdigest(), row['initial_sha256'])
            words = program.view('<u4')
            position = np.flatnonzero((words & 0x7fffff) == 0x25f)
            self.assertEqual(position.size, 1)
            m, k, n = row['physical_mkn']
            logical_rows, logical_k, logical_n = row['logical_mkn']
            panels[index] = Panel(program.ctypes.data, w.ctypes.data, words.size, int(position[0]), m, k, n,
                row['logical_row_start'], logical_rows, logical_k, logical_n, 0, row['output_offset'], row['total_dram_bytes'],
                w.size, row['requested_dma']['load_bytes'], row['requested_dma']['store_bytes'], row['a_scale'], row['b_scale'])
            keep.extend((a, w, output, program, initial)); initials.append(initial); outputs.append(output); programs.append(program)
        bias_row = self.receipt['parameters'][module].get('bias')
        bias = self.tensor(bias_row) if bias_row else None
        call = Call(incoming.shape[0], incoming.shape[1], expected.shape[1], 0, len(rows),
                    bias.ctypes.data if bias is not None else None, site, layer, role, 0, dtype, output_dtype,
                    output_dtype, rows[0]['a_scale'])
        if mutation == 'scale': call.activation_scale += 1
        if mutation == 'role': call.role += 1
        if mutation == 'padded_output':
            self.assertEqual(len(rows), 1)
            m, _, n = rows[0]['physical_mkn']
            word = ((50//32)*(n//32))*1024+(50%32)*16
            outputs[0].view('<u2')[word] = 0x8000
        dram = np.full(max(row['total_dram_bytes'] for row in rows), 0x93, np.uint8)
        activation = np.empty(max(row['physical_mkn'][0]*row['physical_mkn'][1] for row in rows), np.uint8)
        conversion = np.empty(incoming.size, np.uint16)
        cycles = np.zeros(len(rows), np.uint32)
        pointer = lambda values: (ct.c_void_p*len(values))(*[x.ctypes.data for x in values])
        ip, op, pp = pointer(initials), pointer(outputs), pointer(programs)
        sizes = (ct.c_size_t*len(rows))(*[x.size for x in initials])
        offsets = (ct.c_size_t*len(rows))(*[x['output_offset'] for x in rows])
        program_sizes = (ct.c_size_t*len(rows))(*[x.size for x in programs])
        host = self.owner.unit_config(dram.ctypes.data, ip, op, pp, sizes, offsets, program_sizes, len(rows))
        context = Context(host, ct.addressof(panels), ct.addressof(call), len(rows), 1, 0, 0,
            dram.ctypes.data, activation.ctypes.data, conversion.ctypes.data, dram.size, activation.size,
            conversion.size, 1000, cycles.ctypes.data, Buffer(), 0, 0, 0)
        output = np.full_like(expected, 7)
        if vision:
            result = self.owner.smol_policy_raw_vision(ct.byref(context), layer, role, incoming.ctypes.data,
                incoming.shape[0], incoming.shape[1], output.ctypes.data, output.shape[1])
        else:
            result = self.owner.smol_policy_raw_graph(ct.byref(context), site, layer, role, dtype, incoming.ctypes.data,
                incoming.shape[0], incoming.shape[1], output_dtype, output.ctypes.data, output.shape[1])
        if mutation:
            self.assertNotEqual(result, 0)
            self.assertEqual(context.failed, 1)
            self.assertEqual(context.call_cursor, 0)
            self.assertTrue(np.all(output == 7))
            if mutation != 'padded_output': self.assertEqual(self.owner.unit_completed(), 0)
        else:
            self.assertEqual((result, context.call_cursor, context.panel_cursor), (0, 1, len(rows)))
            self.assertTrue(np.all(cycles == 100))
            np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))
            print(json.dumps({'mock_transport_call': module, 'staged_panels_exact': len(rows), 'native_output_exact': True}), flush=True)

    def test_actual_global_eight_panel_scale_and_float_bias(self):
        self.case('vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj')
        self.case('action_time_mlp_in')
        self.case('vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc2')

    def test_refuse_changed_scale_role_and_nonpositive_padded_zero(self):
        for mutation in ('scale', 'role', 'padded_output'):
            self.case('action_time_mlp_in', mutation)


if __name__ == '__main__':
    unittest.main()
