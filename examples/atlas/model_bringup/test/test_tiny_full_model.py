"""Exercise production C orchestration with test-only ABI stubs by default.

TINY_MERLIN_COMPARISONS selects real host headers and primitives instead, checking
provider linkage with the same orchestration assertions. Synthetic projections
and nonnumerical stubs do not qualify full-model numerical equivalence.
"""
import ctypes as ct
import _ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


class Parameters(ct.Structure):
    _fields_ = [("embedding", ct.c_void_p), ("input_norm", ct.c_void_p * 22),
                ("post_norm", ct.c_void_p * 22), ("final_norm", ct.c_void_p),
                ("cosine", ct.c_void_p), ("sine", ct.c_void_p), ("epsilon", ct.c_float),
                ("attention", ct.c_void_p), ("fused_project", ct.c_void_p), ("normalize", ct.c_void_p),
                ("capabilities", ct.c_uint64)]


class State(ct.Structure):
    _fields_ = [("keys", ct.c_void_p * 2), ("values", ct.c_void_p * 2),
                ("position", ct.c_size_t), ("bank", ct.c_uint), ("next_token", ct.c_uint32)]


Projection = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
                         ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p)
Attention = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                        ct.c_size_t, ct.c_size_t, ct.c_void_p)
FusedProjection = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
                              ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_void_p)


class ModelFixture:
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("TINY_MERLIN_COMPARISONS")
        cls.stub_provider = value is None
        provider = (Path(value) if value is not None
                    else Path(__file__).absolute().parent / "fixtures/tiny_full_host_stubs")
        if any(word in part.lower() for part in provider.parts for word in ("hammer", "vlsi")):
            raise ValueError("restricted source provider")
        names = ("host_stubs.c",) if cls.stub_provider else (
            "smol_host_ops.c", "llama_host_attention.c", "llama_host_layout.c",
            "llama_rotary_table.c", "llama_host_bmm.c", "smol_host_softmax.c")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        repo = Path(__file__).absolute().parents[1]
        source = repo / "tools/tiny_full/tiny_full_model.c"
        command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                   "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC", "-I", str(provider),
                   "-I", str(repo / "runtime"), str(source), str(repo / "runtime/tiny_attention.c"),
                   *(str(provider / name) for name in names), "-lm", "-o", str(cls.root / "host.so")]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        cls.lib = ct.CDLL(str(cls.root / "host.so"))
        cls.addClassCleanup(_ctypes.dlclose, cls.lib._handle)
        cls.lib.tiny_full_workspace_bytes.restype = ct.c_size_t
        cls.lib.tiny_full_argmax_bf16.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_uint32)]
        cls.lib.tiny_full_model_step.argtypes = [ct.POINTER(Parameters), ct.POINTER(State), ct.c_void_p,
            ct.c_size_t, Projection, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]

    def fixture(self):
        self.embedding = np.zeros((32000, 2048), np.uint16)
        self.norm = np.full(2048, 0x3f80, np.uint16)
        self.cosine = np.full((160, 64), 0x3f80, np.uint16)
        self.sine = np.zeros((160, 64), np.uint16)
        self.keys = [np.zeros((22, 4 * 160 * 64), np.uint16) for _ in range(2)]
        self.values = [np.zeros_like(value) for value in self.keys]
        self.workspace = np.zeros(self.lib.tiny_full_workspace_bytes() // 8 + 1, np.uint64)
        self.logits = np.zeros(32000, np.uint16)
        p = Parameters(self.embedding.ctypes.data, (ct.c_void_p * 22)(*[self.norm.ctypes.data] * 22),
            (ct.c_void_p * 22)(*[self.norm.ctypes.data] * 22), self.norm.ctypes.data,
            self.cosine.ctypes.data, self.sine.ctypes.data, 1e-5)
        s = State((ct.c_void_p * 2)(*[v.ctypes.data for v in self.keys]),
                  (ct.c_void_p * 2)(*[v.ctypes.data for v in self.values]), 128, 0, 7)
        return p, s

    def step(self, p, s, token, callback):
        tokens = np.array([token], np.uint32)
        return self.lib.tiny_full_model_step(ct.byref(p), ct.byref(s), tokens.ctypes.data, 1,
            callback, None, self.workspace.ctypes.data, self.workspace.nbytes,
            self.logits.ctypes.data, self.logits.size)


class FullModelTest(ModelFixture, unittest.TestCase):
    def test_fused_callbacks_keep_norms_and_transport_opaque_tiles_and_fp8(self):
        if not self.stub_provider:
            self.skipTest("routing counters require the test host provider")
        for rows in (1, 128):
            with self.subTest(rows=rows):
                p, s = self.fixture()
                self.embedding[:] = 0x3f80
                if rows == 128: s.position = 0
                observed, errors = [], []
                active_gate = None
                counters = [ct.c_uint.in_dll(self.lib, "tiny_test_" + name + "_calls")
                            for name in ("rms", "residual", "silu")]
                before = [x.value for x in counters]

                @Projection
                def project(_, layer, role, incoming, count, k, width, output):
                    observed.append((layer, role))
                    if role not in (0, 1, 2, 7): errors.append("legacy fused role")
                    size = ((count + 31) // 32 * 32 if role <= 2 else count) * width * 2
                    ct.memset(output, 0, size)
                    if role == 7: ct.c_uint16.from_address(output).value = 0x3f80
                    return 0

                @Attention
                def attention(_, layer, query, keys, values, count, key_rows, output):
                    ct.memset(output, 0, (count + 31) // 32 * 32 * 2048 * 2)
                    return 0

                @FusedProjection
                def fused(_, layer, role, incoming, count, k, width, operand, output):
                    nonlocal active_gate
                    observed.append((layer, role))
                    if role in (3, 6):
                        if operand: errors.append("residual operand")
                        words = np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_uint16)), shape=(count * width,))
                        words[:] = 0x4000 if role == 3 else 0x3f00
                    elif role == 4:
                        if operand: errors.append("gate operand")
                        active_gate = output
                        ct.memset(output, 0x33, (count + 31) // 32 * 32 * width * 2)
                    elif role == 5:
                        if operand != active_gate: errors.append("activated gate routing")
                        ct.memset(output, 0x61, (count + 31) // 32 * 32 * width)
                    if role == 6:
                        raw = ct.string_at(incoming, (count + 31) // 32 * 32 * k)
                        if raw != bytes([0x61]) * len(raw): errors.append("FP8 routing")
                    return 0

                p.attention = ct.cast(attention, ct.c_void_p).value
                p.fused_project = ct.cast(fused, ct.c_void_p).value
                tokens = np.full(rows, s.next_token, np.uint32)
                logits = np.zeros((rows, 32000), np.uint16)
                self.assertEqual(self.lib.tiny_full_model_step(ct.byref(p), ct.byref(s), tokens.ctypes.data, rows,
                    project, None, self.workspace.ctypes.data, self.workspace.nbytes, logits.ctypes.data, logits.size), 0)
                self.assertEqual(errors, [])
                self.assertEqual(observed, [(layer, role) for layer in range(22) for role in range(7)] + [(22, 7)])
                self.assertEqual([x.value-y for x, y in zip(counters, before)], [45, 44, 0])

    def test_argmax_first_tie_signed_zero_and_nonfinite(self):
        words = np.full(32000, 0xbf80, np.uint16)
        words[31] = words[91] = 0x3f80
        token = ct.c_uint32(123)
        self.assertEqual(self.lib.tiny_full_argmax_bf16(words.ctypes.data, words.size, ct.byref(token)), 0)
        self.assertEqual(token.value, 31)
        words[:] = 0x8000
        words[17] = 0
        self.assertEqual(self.lib.tiny_full_argmax_bf16(words.ctypes.data, words.size, ct.byref(token)), 0)
        self.assertEqual(token.value, 0)
        for nonfinite in (0x7f80, 0xff80, 0x7fc1):
            words[31999] = nonfinite
            token.value = 123
            self.assertEqual(self.lib.tiny_full_argmax_bf16(words.ctypes.data, words.size, ct.byref(token)), -1)
            self.assertEqual(token.value, 123)

    def test_two_decode_iterations_advance_each_layer_cache_and_greedy(self):
        p, s = self.fixture()
        observed = []
        step_index = 0

        @Projection
        def callback(_, layer, logical, incoming, rows, k, width, output):
            observed.append((layer, logical, rows, k, width))
            words = np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_uint16)), shape=(rows * width,))
            words[:] = 0
            if logical == 2:
                words[:] = 0x3f80 if step_index == 0 else 0x4000
            if layer == 22:
                words[8 + step_index] = 0x3f80
            return 0

        for step_index in range(2):
            self.assertEqual(self.step(p, s, s.next_token, callback), 0)
            self.assertEqual((s.position, s.next_token), (129 + step_index, 8 + step_index))
            active = self.values[s.bank]
            for layer in range(22):
                cache = active[layer, :4 * s.position * 64].reshape(4, s.position, 64)
                self.assertTrue(np.all(cache[:, :128] == 0))
                self.assertTrue(np.all(cache[:, 128] == 0x3f80))
                if step_index:
                    self.assertTrue(np.all(cache[:, 129] == 0x4000))
        expected = [(layer, role) for layer in range(22) for role in range(7)] + [(22, 7)]
        self.assertEqual([(layer, role) for layer, role, *_ in observed], expected * 2)
        shapes = [(2048, 2048), (2048, 256), (2048, 256), (2048, 2048), (2048, 5632), (2048, 5632), (5632, 2048)]
        expected_calls = [(layer, role, 1, k, width) for layer in range(22) for role, (k, width) in enumerate(shapes)]
        expected_calls.append((22, 7, 1, 2048, 32000))
        self.assertEqual(observed, expected_calls * 2)

    def test_teacher_forcing_rejected_before_projection(self):
        p, s = self.fixture()
        called = []

        @Projection
        def callback(*args):
            called.append(args)
            return -1

        self.assertEqual(self.step(p, s, 8, callback), -1)
        self.assertEqual(called, [])
        self.assertEqual((s.position, s.bank, s.next_token), (128, 0, 7))

    def test_decode_before_complete_prefill_rejected_without_projection(self):
        p, s = self.fixture()
        called = []

        @Projection
        def callback(*args):
            called.append(args)
            return -1

        for position in (1, 64, 127):
            s.position = position
            self.assertEqual(self.step(p, s, 7, callback), -1)
            self.assertEqual((s.position, s.bank, s.next_token), (position, 0, 7))
        self.assertEqual(called, [])

    def test_prefill_transition_all22_caches_and155_calls(self):
        p, s = self.fixture()
        s.position = 0
        tokens = np.arange(128, dtype=np.uint32)
        logits = np.zeros((128, 32000), np.uint16)
        observed = []

        @Projection
        def callback(_, layer, logical, incoming, rows, k, width, output):
            observed.append((layer, logical, rows))
            words = np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_uint16)), shape=(rows * width,))
            words[:] = 0x3f80 if logical == 2 else 0
            if layer == 22:
                words[(rows - 1) * width + 11] = 0x3f80
            return 0

        status = self.lib.tiny_full_model_step(ct.byref(p), ct.byref(s), tokens.ctypes.data, 128,
            callback, None, self.workspace.ctypes.data, self.workspace.nbytes,
            logits.ctypes.data, logits.size)
        self.assertEqual(status, 0)
        self.assertEqual((s.position, s.bank, s.next_token), (128, 1, 11))
        self.assertEqual(len(observed), 155)
        self.assertTrue(all(rows == 128 for _, _, rows in observed))
        for layer in range(22):
            cache = self.values[s.bank][layer, :4 * 128 * 64].reshape(4, 128, 64)
            self.assertTrue(np.all(cache == 0x3f80))

    def test_projection_failure_preserves_last_complete_state(self):
        p, s = self.fixture()

        @Projection
        def callback(*args):
            return -1

        self.assertEqual(self.step(p, s, 7, callback), -2)
        self.assertEqual((s.position, s.bank, s.next_token), (128, 0, 7))
        self.assertTrue(np.all(self.keys[s.bank] == 0))
        self.assertTrue(np.all(self.values[s.bank] == 0))

    def test_late_failure_preserves_committed_cache_and_state(self):
        # Reach a later layer, the LM head, and argmax after earlier layers have
        # already written the inactive bank. None may commit a partial step.
        for failure in ("later_layer", "lm_head", "nonfinite_logits"):
            with self.subTest(failure=failure):
                p, s = self.fixture()
                self.keys[0][:] = 0x3f00
                self.values[0][:] = 0x3f80
                observed = []

                @Projection
                def callback(_, layer, logical, incoming, rows, k, width, output):
                    observed.append((layer, logical))
                    words = np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_uint16)), shape=(rows * width,))
                    words[:] = 0
                    if failure == "later_layer" and (layer, logical) == (11, 6):
                        return -1
                    if failure == "lm_head" and layer == 22:
                        return -1
                    if failure == "nonfinite_logits" and layer == 22:
                        words[-1] = 0x7fc1
                    return 0

                expected_status = {"later_layer": -2, "lm_head": -4, "nonfinite_logits": -5}
                self.assertEqual(self.step(p, s, 7, callback), expected_status[failure])
                self.assertEqual((s.position, s.bank, s.next_token), (128, 0, 7))
                self.assertTrue(np.all(self.keys[0] == 0x3f00))
                self.assertTrue(np.all(self.values[0] == 0x3f80))
                self.assertTrue(np.any(self.values[1] != 0))
                self.assertEqual(observed[-1], (11, 6) if failure == "later_layer" else (22, 7))

    def test_embedding_rows_feed_first_projection(self):
        if not self.stub_provider:
            self.skipTest("raw embedding routing requires the identity RMSNorm stub")
        p, s = self.fixture()
        self.embedding[7] = 0x3f00
        observed = []

        @Projection
        def callback(_, layer, logical, incoming, rows, k, width, output):
            words = np.ctypeslib.as_array(ct.cast(incoming, ct.POINTER(ct.c_uint16)), shape=(rows * k,))
            observed.append(words.copy())
            return -1

        self.assertEqual(self.step(p, s, 7, callback), -2)
        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0], self.embedding[7])

        s.position = 0
        tokens = np.arange(128, dtype=np.uint32)[::-1].copy()
        self.embedding[:128] = np.arange(128, dtype=np.uint16)[:, None]
        logits = np.zeros((128, 32000), np.uint16)
        observed.clear()
        status = self.lib.tiny_full_model_step(ct.byref(p), ct.byref(s), tokens.ctypes.data, 128,
            callback, None, self.workspace.ctypes.data, self.workspace.nbytes,
            logits.ctypes.data, logits.size)
        self.assertEqual(status, -2)
        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0].reshape(128, 2048), self.embedding[tokens])


if __name__ == "__main__":
    unittest.main()
