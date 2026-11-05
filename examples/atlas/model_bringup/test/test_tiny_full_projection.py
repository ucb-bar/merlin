"""Native projection staging, cache lifetime, and output assembly checks.

The packing double exposes invocation counts and deterministic bytes. Device
execution and FP8 arithmetic are outside this orchestration test's scope.
"""
import ctypes as ct
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


class Panel(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("rows", "physical_rows", "k", "n", "column", "width")]
    _fields_ += [("logical", ct.c_uint), ("a_scale", ct.c_int), ("weights", ct.c_void_p),
                ("weight_bytes", ct.c_size_t), ("program", ct.c_void_p)]
    _fields_ += [(name, ct.c_size_t) for name in ("program_words", "output_offset", "total_bytes", "load_bytes", "store_bytes")]


Transport = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(Panel), ct.c_void_p, ct.c_size_t)


class Panels(ct.Structure):
    _fields_ = [("panels", ct.POINTER(Panel)), ("panel_count", ct.c_size_t), ("cursor", ct.c_size_t),
                ("dram", ct.c_void_p), ("dram_capacity", ct.c_size_t), ("transport", Transport), ("context", ct.c_void_p)]


class Fusion(ct.Structure):
    _fields_ = [("extent", ct.c_size_t), ("operand", ct.c_size_t), ("scale", ct.c_int), ("role", ct.c_uint)]


class AttentionIdentity(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("bytes", ct.c_size_t), ("step", ct.c_size_t),
                ("layer", ct.c_size_t), ("producer_index", ct.c_size_t), ("exponent", ct.c_int),
                ("valid", ct.c_uint), ("consumed", ct.c_uint)]


class NormIdentity(ct.Structure):
    _fields_ = [("input", ct.c_void_p)] + [(name, ct.c_size_t) for name in ("rows", "step", "layer", "producer_index", "consumer_call")]
    _fields_ += [("exponent", ct.c_int), ("stage", ct.c_uint)]


class Context(ct.Structure):
    _fields_ = [("panels", Panels), ("calls", ct.c_size_t), ("step_count", ct.c_size_t),
                ("attention_panels", ct.c_void_p), ("attention_count", ct.c_size_t), ("attention_cursor", ct.c_size_t),
                ("attention_transport", ct.c_void_p), ("attention_context", ct.c_void_p),
                ("packed_a", ct.c_uint8 * (128 * 5632)), ("packed_input", ct.c_void_p),
                ("packed_rows", ct.c_size_t), ("packed_k", ct.c_size_t), ("packed_call", ct.c_size_t),
                ("packed_scale", ct.c_int), ("packed_role", ct.c_uint), ("packed_valid", ct.c_uint),
                ("fusions", ct.c_void_p), ("attention_v2", ct.c_void_p),
                ("pending_attention", AttentionIdentity), ("norm_panels", ct.c_void_p), ("norm_count", ct.c_size_t),
                ("norm_cursor", ct.c_size_t), ("layer_count", ct.c_size_t), ("norm_transport", ct.c_void_p),
                ("norm_transport_context", ct.c_void_p), ("capabilities", ct.c_uint64), ("configured", ct.c_uint),
                ("pending_norm", NormIdentity), ("rope_tables", ct.c_void_p)]


STUBS = r'''
#include <stddef.h>
#include <stdint.h>
unsigned scales, packs, attention_packs;
int host_matrix_scale_bf16(const uint16_t *p, size_t m, size_t k, int *s) {
  (void)p; (void)m; (void)k; ++scales; *s = 3; return 0;
}
int host_matrix_pack_bf16_at_scale(const uint16_t *p, size_t rows, size_t k, size_t m,
    size_t pk, size_t tm, size_t tn, int scale, uint8_t *out, size_t bytes) {
  (void)tm; (void)tn; (void)scale; ++packs;
  if (pk != k || bytes != m*k) return -1;
  for (size_t r=0;r<m;++r) for (size_t c=0;c<k;++c) out[r*k+c] = r<rows ? (uint8_t)p[r*k+c] : 0;
  return 0;
}
int tiny_attention_pack(const uint16_t *p, size_t rows, size_t k, int scale, uint8_t *out, size_t bytes) {
  ++attention_packs;
  return host_matrix_pack_bf16_at_scale(p, rows, k, (rows+31)/32*32, k, 32, 32, scale, out, bytes);
}
'''


class ProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        repo = Path(__file__).absolute().parents[1]
        (root / "stubs.c").write_text(STUBS)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-shared", "-fPIC",
                        "-I", str(repo / "runtime"), "-I", str(repo / "tools/tiny_full"),
                        str(repo / "tools/tiny_full/tiny_full_projection.c"), str(root / "stubs.c"),
                        "-o", str(root / "projection.so")], check=True, capture_output=True)
        cls.lib = ct.CDLL(str(root / "projection.so"))
        cls.lib.tiny_full_project.argtypes = [ct.POINTER(Context), ct.c_uint, ct.c_uint, ct.c_void_p,
                                           ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p]
        cls.lib.tiny_full_fused_project.argtypes = [ct.POINTER(Context), ct.c_uint, ct.c_uint, ct.c_void_p,
                                                  ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_void_p]

    def fixture(self, rows=128, native=False):
        self.weights = np.full(5632 * 2048, 0x73, np.uint8)
        self.program = np.zeros(1, np.uint32)
        self.input = np.arange(128 * 5632, dtype=np.uint16)
        self.output = np.zeros(128 * 5632, np.uint16)
        self.dram = np.zeros(16 * 1024 * 1024, np.uint8)
        self.observed, self.errors = [], []
        self.native = native

        @Transport
        def transport(_, pp, address, size):
            p = pp.contents
            raw = np.ctypeslib.as_array(ct.cast(address, ct.POINTER(ct.c_uint8)), shape=(size,))
            want = np.zeros((p.physical_rows, p.k), np.uint8)
            want[:p.rows] = self.input[:p.rows * p.k].reshape(p.rows, p.k).astype(np.uint8)
            if raw[:p.physical_rows * p.k].tobytes() != want.tobytes(): self.errors.append("activation")
            if not np.all(raw[p.physical_rows * p.k:p.output_offset - 256] == 0x73): self.errors.append("weights")
            if not np.all(raw[p.output_offset - 256:p.output_offset] == 0xa5): self.errors.append("before guard")
            if not np.all(raw[p.output_offset:p.output_offset + p.store_bytes] == 0xcd): self.errors.append("poison")
            if not np.all(raw[p.output_offset + p.store_bytes:] == 0x5a): self.errors.append("after guard")
            # Constant tile words make both native-tile and row-major assembly observable.
            value = p.logical * 16 + p.column // 32
            raw[p.output_offset:p.output_offset + p.store_bytes:2] = value & 255
            raw[p.output_offset + 1:p.output_offset + p.store_bytes:2] = value >> 8
            self.observed.append((p.logical, p.column))
            return 0

        self.transport = transport
        entries = []
        for role, (k, width) in enumerate([(2048, 2048), (2048, 256), (2048, 256), (2048, 2048),
                                           (2048, 5632), (2048, 5632), (5632, 2048)]):
            column = 0
            while column < width:
                n = min(2048 if role != 6 else 1024, width - column)
                m = 128 if rows == 128 else 32
                out = m * k + k * n + 256
                entries.append(Panel(rows, m, k, n, column, width, role, 3, self.weights.ctypes.data,
                                     k * n, self.program.ctypes.data, 1, out, out + m * n * 2 + 256,
                                     m * k + k * n, m * n * 2))
                column += n
        self.entries = (Panel * len(entries))(*entries)
        c = Context()
        c.panels = Panels(self.entries, len(entries), 0, self.dram.ctypes.data, self.dram.nbytes, transport, None)
        c.calls, c.step_count, c.attention_panels = (0 if rows == 128 else 155), 3, 1 if native else None
        return c

    def project(self, c, role, rows):
        k, width = [(2048, 2048), (2048, 256), (2048, 256), (2048, 2048),
                    (2048, 5632), (2048, 5632), (5632, 2048)][role]
        return self.lib.tiny_full_project(ct.byref(c), 0, role, self.input.ctypes.data, rows, k, width,
                                          self.output.ctypes.data)

    def test_identical_staging_and_assembly_prefill_decode_native_and_legacy(self):
        for rows in (128, 1):
            for native in (False, True):
                with self.subTest(rows=rows, native=native):
                    c = self.fixture(rows, native)
                    counts = [ct.c_uint.in_dll(self.lib, name) for name in ("scales", "packs", "attention_packs")]
                    before = [x.value for x in counts]
                    for role in range(7):
                        self.assertEqual(self.project(c, role, rows), 0)
                        if not native or role > 2:
                            width = self.entries[next(i for i, p in enumerate(self.entries) if p.logical == role)].width
                            for column in range(0, width, 2048 if role != 6 else 1024):
                                n = min(2048 if role != 6 else 1024, width - column)
                                matrix = self.output[:rows * width].reshape(rows, width)
                                self.assertTrue(np.all(matrix[:, column:column+n] == role*16+column//32))
                    self.assertEqual(self.errors, [])
                    self.assertEqual([x.value-y for x, y in zip(counts, before)], [4, 4, int(native)])
                    self.assertEqual(c.panels.cursor, len(self.entries))

    def test_reused_pointer_at_new_producer_is_repacked(self):
        c = self.fixture()
        self.assertEqual(self.project(c, 0, 128), 0)
        self.input[:] += 9
        # Simulate the next layer's role 0 at the same workspace address.
        c.calls, c.panels.cursor = 7, 0
        self.assertEqual(self.lib.tiny_full_project(ct.byref(c), 1, 0, self.input.ctypes.data, 128,
                                                 2048, 2048, self.output.ctypes.data), 0)
        self.assertEqual(self.errors, [])

    def test_cache_and_later_panel_metadata_fail_before_launch(self):
        c = self.fixture()
        c.calls, c.panels.cursor = 1, 1
        self.assertEqual(self.project(c, 1, 128), -2)
        self.assertEqual(self.observed, [])
        c = self.fixture()
        self.entries[5].a_scale = 4  # A later gate shard must be checked before any gate launch.
        for role in range(4): self.assertEqual(self.project(c, role, 128), 0)
        count = len(self.observed)
        self.assertEqual(self.project(c, 4, 128), -3)
        self.assertEqual(len(self.observed), count)

    @staticmethod
    def tiled(matrix):
        m, n = matrix.shape
        return matrix.reshape(m // 32, 32, n // 32, 2, 16).transpose(0, 2, 3, 1, 4).copy().reshape(-1)

    def fusion_fixture(self, rows=128):
        c = self.fixture(rows, True)
        m = 128 if rows == 128 else 32
        self.fusions = (Fusion * len(self.entries))()
        for i, p in enumerate(self.entries):
            if p.logical < 3: continue
            extent = ((m // 32 - 1) * 32 * p.width + (p.column + p.n) * 32
                      if p.logical == 5 else m * p.n * 2)
            extra = m * p.n * 2 if p.logical == 5 else 0
            operand = p.output_offset + extent + 256 if extra else 0
            p.total_bytes, p.store_bytes = p.output_offset + extent + 256 + extra, m * p.n * (1 if p.logical == 5 else 2)
            self.fusions[i] = Fusion(extent, operand, 3 if p.logical == 5 else 0, p.logical)
        c.fusions = ct.addressof(self.fusions)
        c.calls, c.panels.cursor = (3 if rows == 128 else 158), 3
        self.gate = np.zeros(m * 5632, np.uint16)
        self.fp8 = np.full(m * 5632, 0xee, np.uint8)
        self.expected_fp8 = np.zeros_like(self.fp8)
        self.current_operand = None

        @Transport
        def transport(_, pp, address, size):
            p = pp.contents
            index = (ct.addressof(pp.contents) - ct.addressof(self.entries)) // ct.sizeof(Panel)
            f = self.fusions[index]
            raw = np.ctypeslib.as_array(ct.cast(address, ct.POINTER(ct.c_uint8)), shape=(size,))
            if not np.all(raw[p.output_offset - 256:p.output_offset] == 0xa5): self.errors.append("before guard")
            if not np.all(raw[p.output_offset:p.output_offset + f.extent] == 0xcd): self.errors.append("output holes")
            if not np.all(raw[p.output_offset + f.extent:p.output_offset + f.extent + 256] == 0x5a):
                self.errors.append("after guard")
            if p.logical == 6:
                if raw[:m * 5632].tobytes() != self.expected_fp8.tobytes(): self.errors.append("assembled down A")
            if f.operand:
                full = self.gate.reshape(m // 32, 5632 // 32, 1024)
                wanted = full[:, p.column // 32:(p.column + p.n) // 32].copy().tobytes()
                if raw[f.operand:].tobytes() != wanted: self.errors.append("operand tiles")
            if p.logical == 5:
                for tr in range(m // 32):
                    at = tr * 32 * 5632 + p.column * 32
                    data = (np.arange(p.n * 32, dtype=np.uint32) + p.column // 32 + tr * 7).astype(np.uint8)
                    raw[p.output_offset + at:p.output_offset + at + data.size] = data
                    self.expected_fp8[at:at + data.size] = data
            else:
                matrix = np.broadcast_to(np.arange(p.n, dtype=np.uint16) + p.column + p.logical * 100,
                                         (m, p.n)).copy()
                raw[p.output_offset:p.output_offset + f.extent] = np.frombuffer(self.tiled(matrix).tobytes(), np.uint8)
            self.observed.append((p.logical, p.column))
            return 0

        self.transport = transport
        c.panels.transport = transport
        return c

    def fused(self, c, role, rows):
        k, width = {3: (2048, 2048), 4: (2048, 5632), 5: (2048, 5632), 6: (5632, 2048)}[role]
        incoming = self.fp8.ctypes.data if role == 6 else self.input.ctypes.data
        operand = self.gate.ctypes.data if role == 5 else None
        output = self.gate.ctypes.data if role == 4 else self.fp8.ctypes.data if role == 5 else self.output.ctypes.data
        return self.lib.tiny_full_fused_project(ct.byref(c), 0, role, incoming, rows, k, width, operand, output)

    def test_fused_staging_padding_and_written_fp8_spans(self):
        for rows in (128, 1):
            with self.subTest(rows=rows):
                c = self.fusion_fixture(rows)
                counts = [ct.c_uint.in_dll(self.lib, name) for name in ("scales", "packs", "attention_packs")]
                before = [x.value for x in counts]
                for role in range(3, 7):
                    self.assertEqual(self.fused(c, role, rows), 0)
                    if role in (3, 6):
                        expected = np.arange(2048, dtype=np.uint16) + role * 100
                        np.testing.assert_array_equal(self.output[:rows * 2048].reshape(rows, 2048),
                                                      np.broadcast_to(expected, (rows, 2048)))
                np.testing.assert_array_equal(self.fp8, self.expected_fp8)
                self.assertEqual(self.errors, [])
                self.assertEqual([x.value-y for x, y in zip(counts, before)], [2, 2, 1])

    def test_fused_bad_extent_operand_dma_role_and_scale_refused_before_launch(self):
        for field in ("extent", "operand", "role", "store_bytes", "total_bytes"):
            with self.subTest(field=field):
                c = self.fusion_fixture()
                target = self.entries[3] if field in ("store_bytes", "total_bytes") else self.fusions[3]
                setattr(target, field, getattr(target, field) + 32)
                self.assertEqual(self.fused(c, 3, 128), -3)
                self.assertEqual(self.observed, [])
                self.assertEqual(c.panels.cursor, 3)
        c = self.fusion_fixture()
        for role in (3, 4, 5): self.assertEqual(self.fused(c, role, 128), 0)
        self.entries[-1].a_scale = 4
        count = len(self.observed)
        self.assertEqual(self.fused(c, 6, 128), -3)
        self.assertEqual(len(self.observed), count)
        c = self.fusion_fixture()
        c.fusions = None
        self.assertEqual(self.fused(c, 3, 128), -1)
        self.assertEqual(self.observed, [])


if __name__ == "__main__":
    unittest.main()
