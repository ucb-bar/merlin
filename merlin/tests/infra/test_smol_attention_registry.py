"""Build the graph without Atlas, and exercise the explicit attention provider."""
import ctypes as ct
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "examples/atlas/comparisons"))


from smol_test_support import (
    Callback, FIELDS, Frame, Parameters, State, load_shared,
    VisionCallback, VisionFrame, VisionLayer, VisionParameters,
)


AttentionCallback = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p,
                                ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t,
                                ct.c_size_t, ct.c_size_t, ct.c_void_p)


class AttentionRegistryTest(unittest.TestCase):
    def build(self, runtime=None):
        here = repo_root() / "examples/atlas/comparisons"
        stems = ('smol_policy_block', 'smol_policy_ops', 'smol_host_ops', 'smol_host_rope',
                 'smol_host_attention', 'smol_host_softmax', 'smol_vision_graph',
                 'smol_vision_ops', 'smol_vision_attention')
        temporary = tempfile.TemporaryDirectory(prefix='smol-registry-')
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'graph.so'
        flags = [] if runtime is None else ['-DSMOL_ATTENTION_RUNTIME=1', '-I' + str(runtime)]
        sources = ([] if runtime is None else [runtime / 'smol_attention.c'])
        sources += [here / (stem + '.c') for stem in stems]
        return load_shared(path, sources, flags=['-Wall', '-Wextra', '-Werror', '-Wl,-z,defs', *flags])

    def policy(self, owner, kind, expected_code=0):
        arrays = []

        def keep(size, dtype=np.uint16):
            array = np.zeros(size, dtype)
            arrays.append(array)
            return array.ctypes.data

        parameters = Parameters(kind, keep(960), keep(960), 1e-5, 1e-5,
                                      keep(64, np.float32), keep(64, np.float32), 0, 1)
        frame = Frame(*[keep(15 * 177 * 227, np.float32) for _ in FIELDS])
        positions, mask = keep(177, np.int64), keep(177 * 227, np.uint8)
        cache_key, cache_value, incoming = keep(177 * 320), keep(177 * 320), keep(177 * 960)
        projections = []

        @Callback
        def project(context, role, input_dtype, source, rows, k, output_dtype, destination, n):
            projections.append(role)
            return 0

        state = State()
        owner.smol_policy_block.argtypes = [ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p,
            ct.c_void_p, ct.c_void_p, ct.POINTER(Parameters), ct.POINTER(Frame),
            ct.POINTER(State), Callback, ct.c_void_p]
        code = owner.smol_policy_block(incoming, 0, positions, mask, cache_key, cache_value,
                                      ct.byref(parameters), ct.byref(frame), ct.byref(state), project, None)
        self.assertEqual(code, expected_code)
        self.assertEqual(state.complete, int(expected_code == 0))
        self.assertEqual(projections, list(range(7 if expected_code == 0 else 3)))

    def vision(self, owner, expected_code=0):
        arrays = []

        def keep(size):
            array = np.zeros(size, np.uint16)
            arrays.append(array)
            return array.ctypes.data

        layer = VisionLayer(*[keep(768) for _ in range(4)])
        parameters = VisionParameters(keep(1024 * 768), (VisionLayer * 12)(*[layer] * 12),
                                      keep(768), keep(768), keep(65536), 1e-6)
        frame = VisionFrame(*[keep(1024 * 3072) for _ in VisionFrame._fields_])
        image = keep(3 * 512 * 512)
        state = State()
        projections = []

        @VisionCallback
        def project(context, layer, role, source, rows, k, destination, n):
            projections.append(role)
            return 0

        owner.smol_vision_image_layer.argtypes = [ct.c_void_p, ct.POINTER(VisionParameters),
            ct.POINTER(VisionFrame), ct.POINTER(State), VisionCallback, ct.c_void_p, ct.c_uint]
        code = owner.smol_vision_image_layer(image, ct.byref(parameters), ct.byref(frame),
                                            ct.byref(state), project, None, 0)
        self.assertEqual(code, expected_code)
        self.assertEqual(projections, list(range(7 if expected_code == 0 else 4)))

    def test_default_build_needs_no_runtime_and_runs_cpu_attention(self):
        owner = self.build()
        self.assertFalse(hasattr(owner, 'smol_attention_configure'))
        for kind in range(3):
            self.policy(owner, kind)
        self.vision(owner)

    @unittest.skipUnless(os.environ.get('ATLAS_MLIR_ROOT'), 'Set ATLAS_MLIR_ROOT for explicit runtime provider')
    def test_opt_in_dispatches_all_graph_kinds_and_resets_to_cpu(self):
        runtime = Path(os.environ['ATLAS_MLIR_ROOT']) / 'runtime'
        owner = self.build(runtime)
        owner.smol_attention_configure.argtypes = [AttentionCallback, ct.c_void_p]
        self.assertEqual(owner.smol_attention_enabled(), 0)
        events = []
        result = [0]

        @AttentionCallback
        def attention(context, kind, q, k, v, mask, rows, keys, qheads, kvheads, output):
            events.append((context, kind, rows, keys, qheads, kvheads, bool(mask)))
            ct.memset(output, 0, rows * qheads * 64 * 2)
            return result[0]

        owner.smol_attention_configure(attention, 123)
        self.assertEqual(owner.smol_attention_enabled(), 1)
        for kind in range(3):
            self.policy(owner, kind)
        self.vision(owner)
        self.assertEqual(events, [(123, 1, 177, 177, 15, 5, True), (123, 2, 50, 227, 15, 5, True),
                                  (123, 3, 50, 177, 15, 5, True), (123, 0, 1024, 1024, 12, 12, False)])
        result[0] = 42
        self.policy(owner, 0, expected_code=2)
        self.vision(owner, expected_code=2)
        owner.smol_attention_configure(AttentionCallback(), None)
        self.assertEqual(owner.smol_attention_enabled(), 0)
        self.policy(owner, 0)


if __name__ == '__main__':
    unittest.main()
