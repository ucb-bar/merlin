import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from merlin.compile.baremetal_harness import emit_harness


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.interface = self.root/'application-c.json'
        self.interface.write_text(json.dumps(dict(
            pointer_abi='opaque_native_dense_buffers', inputs=['data'],
            input_types=[dict(dtype='i8',shape=[3])], outputs=['first','second'],
            output_types=[dict(dtype='i8',shape=[3]),dict(dtype='i8',shape=[1])],
            platform_binding=dict(context_type='context_type', completion='synchronous',invoke='invoke'))))
        self.input = self.root/'input.bin'
        self.input.write_bytes(b'abc')
        (self.root/'platform.h').write_text('''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
typedef struct { unsigned ready; } context_type;
static int harness_platform_bind(context_type *c,size_t polls) { c->ready=1; return polls!=200000000; }
static int invoke(context_type *c,const void *const *in,void *const *out) {
 if(!c->ready) return 1;
 memcpy(out[0],in[0],3); *(unsigned char *)out[1]=0; return 0;
}
''')

    def emit(self, digest):
        return emit_harness(self.interface,self.root,inputs={'data':self.input},expected_sha256=digest)

    def run_binary(self):
        exe=self.root/'test'
        subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror',str(self.root/'main.c'),'-o',str(exe)],check=True)
        return subprocess.run([str(exe)],capture_output=True,text=True)

    def test_bytes_and_order(self):
        digest=hashlib.sha256(b'abc\0').hexdigest()
        self.emit(digest)
        result=self.run_binary()
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn('SHA256='+digest,result.stdout)

    def test_corrupt_reference_fails(self):
        self.emit('0'*64)
        self.assertEqual(self.run_binary().returncode,1)

    def test_truncated_input_refused(self):
        self.input.write_bytes(b'ab')
        with self.assertRaisesRegex(ValueError,'extent'):
            self.emit('0'*64)

    def test_missing_archive_refused(self):
        spec=json.loads(self.interface.read_text())
        spec['host_linkage']=[dict(archive_sha256='a'*64)]
        self.interface.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError,'archive'):
            self.emit('0'*64)

    def set_spec(self, **changes):
        spec = json.loads(self.interface.read_text())
        spec.update(changes)
        self.interface.write_text(json.dumps(spec))

    def test_abi_and_rosters_refused(self):
        original = self.interface.read_text()
        for changes in (dict(pointer_abi='other'),
                        dict(platform_binding=dict(completion='asynchronous')),
                        dict(inputs=['missing']), dict(input_types=[]),
                        dict(output_types=[])):
            with self.subTest(changes=changes):
                self.interface.write_text(original)
                self.set_spec(**changes)
                with self.assertRaises(ValueError):
                    self.emit('0'*64)

    def test_digest_and_adapter_refused(self):
        for digest in ('A'*64, '0'*63, '0'*65, 'x'*64):
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, 'SHA256'):
                self.emit(digest)
        for header in ('bad".h', 'bad\n.h', ''):
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, 'header'):
                emit_harness(self.interface, self.root, inputs={'data': self.input},
                             expected_sha256='0'*64, adapter_header=header)

    def test_unknown_dtype_refused(self):
        self.set_spec(input_types=[dict(dtype='unknown', shape=[3])])
        with self.assertRaisesRegex(ValueError, 'dtype'):
            self.emit('0'*64)

    def test_parameter_spans_refused(self):
        self.set_spec(model_parameters=dict(payload_bytes=64), workspace_alignment=64)
        for address in (None, True, 0, -64, 1, 0xa0000001, (1<<64)-64, (1<<64)-32, 1<<64):
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, 'parameter span'):
                emit_harness(self.interface, self.root, inputs={'data': self.input},
                             expected_sha256='0'*64, parameter_address=address)
        for size in (0, -1, True, 1<<64):
            self.set_spec(model_parameters=dict(payload_bytes=size))
            with self.subTest(size=size), self.assertRaises(ValueError):
                emit_harness(self.interface, self.root, inputs={'data': self.input},
                             expected_sha256='0'*64, parameter_address=0xa0000000)

    def test_parameters_bind_and_failure(self):
        self.set_spec(model_parameters=dict(payload_bytes=64))
        spec = json.loads(self.interface.read_text())
        spec['platform_binding']['bind_parameters'] = 'bind_parameters'
        self.interface.write_text(json.dumps(spec))
        adapter = self.root/'platform.h'
        adapter.write_text(adapter.read_text().replace('#include <stdint.h>\n', '') + '''
static int bind_parameters(context_type *c,const void *address,size_t bytes) {
 if((unsigned long)address!=0xa0000000UL || bytes!=64) return 1;
 c->ready=PARAMETER_READY; return !PARAMETER_READY;
}
''')
        for ready in (1, 0):
            emit_harness(self.interface, self.root, inputs={'data': self.input},
                         expected_sha256=hashlib.sha256(b'abc\0').hexdigest(),
                         parameter_address=0xa0000000)
            text = adapter.read_text()
            adapter.write_text(f'#define PARAMETER_READY {ready}\n' + text)
            result = self.run_binary()
            self.assertEqual(result.returncode, 0 if ready else 1, result.stdout)
            self.assertIn('generated: PASS' if ready else 'parameters: FAIL', result.stdout)
            adapter.write_text(text)

    def test_correct_archive_accepted_and_wrong_refused(self):
        archive = self.root/'math.a'
        archive.write_bytes(b'pinned archive identity')
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.set_spec(host_linkage=[dict(archive_sha256=digest)])
        record = emit_harness(self.interface, self.root, inputs={'data': self.input},
                              expected_sha256=hashlib.sha256(b'abc\0').hexdigest(), math_archive=archive)
        self.assertEqual(record['host_archive_sha256'], digest)
        self.assertEqual(self.run_binary().returncode, 0)
        archive.write_bytes(b'incorrect archive')
        with self.assertRaisesRegex(ValueError, 'archive'):
            emit_harness(self.interface, self.root, inputs={'data': self.input},
                         expected_sha256='0'*64, math_archive=archive)

    def test_zero_inputs_c11(self):
        self.set_spec(inputs=[], input_types=[])
        adapter = self.root/'platform.h'
        adapter.write_text(adapter.read_text().replace('memcpy(out[0],in[0],3);', '(void)in; memcpy(out[0],"abc",3);'))
        emit_harness(self.interface, self.root, inputs={}, expected_sha256=hashlib.sha256(b'abc\0').hexdigest())
        self.assertEqual(self.run_binary().returncode, 0)
        subprocess.run(['cc', '-std=c11', '-pedantic-errors', '-fsyntax-only', str(self.root/'main.c')], check=True)
