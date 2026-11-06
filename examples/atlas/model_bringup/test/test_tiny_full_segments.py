"""Native continuation and reference producer admission controls; no device execution."""
import copy
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

import test_tiny_full_model as model
from test_tiny_full_model import Projection
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools/tiny_full"))
from capture_boundaries import Phase, bind_library, validate_inventory


class SegmentTest(model.ModelFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lib, cls.layout = bind_library(cls.root / "host.so")
        for name in ("begin", "layer", "finish", "step"):
            fn = getattr(cls.lib, "tiny_full_model_" + name)
            types = list(fn.argtypes)
            types[:2] = [ct.POINTER(model.Parameters), ct.POINTER(model.State)]
            if name != "begin":
                types[{"layer": 4, "finish": 3, "step": 4}[name]] = Projection
            fn.argtypes = types
        cls.lib.tiny_full_workspace_bytes.restype = ct.c_size_t
        cls.lib.tiny_full_argmax_bf16.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_uint32)]

    def callback(self, observed, fail=False):
        @Projection
        def project(_, layer, logical, incoming, rows, k, width, output):
            observed.append((layer, logical, hashlib.sha256(ct.string_at(incoming, rows * k * 2)).hexdigest()))
            if fail:
                return -1
            words = np.ctypeslib.as_array(ct.cast(output, ct.POINTER(ct.c_uint16)), shape=(rows * width,))
            words[:] = 0x3f80 if logical == 2 else 0
            if layer == 22:
                words[-width + 9] = 0x3f80
            return 0
        return project

    def begin(self, p, s, phase):
        tokens = np.array([s.next_token], np.uint32)
        return self.lib.tiny_full_model_begin(ct.byref(p), ct.byref(s), ct.byref(phase),
            tokens.ctypes.data, 1, self.workspace.ctypes.data, self.workspace.nbytes)

    def layer(self, p, s, phase, layer, callback):
        return self.lib.tiny_full_model_layer(ct.byref(p), ct.byref(s), ct.byref(phase), layer,
            callback, None, self.workspace.ctypes.data, self.workspace.nbytes)

    def finish(self, p, s, phase, callback):
        return self.lib.tiny_full_model_finish(ct.byref(p), ct.byref(s), ct.byref(phase), callback, None,
            self.workspace.ctypes.data, self.workspace.nbytes, self.logits.ctypes.data, self.logits.size)

    def test_compiled_layout_and_live_only_restore_match_full_step(self):
        whole_inputs = []
        p, s = self.fixture()
        self.embedding[7] = 0x3f00
        self.assertEqual(self.step(p, s, 7, self.callback(whole_inputs)), 0)
        expected = (s.position, s.bank, s.next_token, self.logits.tobytes(),
                    [x.tobytes() for x in self.keys + self.values])
        p, s = self.fixture()
        self.embedding[7] = 0x3f00
        phase, phase_inputs = Phase(), []
        project = self.callback(phase_inputs)
        self.assertEqual(self.begin(p, s, phase), 0)
        for layer in range(22):
            # A new host segment restores only semantic hidden buffers, never
            # scratch left by the preceding layer or an old stack frame.
            hidden = [ct.string_at(self.workspace.ctypes.data + off, self.layout.hidden_bytes)
                      for off in self.layout.hidden_offset]
            self.workspace.fill(0)
            for off, raw in zip(self.layout.hidden_offset, hidden):
                ct.memmove(self.workspace.ctypes.data + off, raw, len(raw))
            self.assertEqual(self.layer(p, s, phase, layer, project), 0)
            self.assertEqual((s.position, s.bank, s.next_token), (128, 0, 7))
            self.assertEqual(phase.next_layer, layer + 1)
        self.assertEqual(self.finish(p, s, phase, project), 0)
        self.assertEqual(phase.active, 0)
        self.assertEqual((s.position, s.bank, s.next_token, self.logits.tobytes(),
                          [x.tobytes() for x in self.keys + self.values]), expected)
        self.assertEqual(phase_inputs, whole_inputs)
        self.assertEqual((self.layout.version, self.layout.phase_bytes, self.layout.state_bytes), (1, 32, 48))
        self.assertEqual(self.layout.cache_bank_bytes, 1802240)
        self.assertEqual(self.layout.hidden_offset[1] - self.layout.hidden_offset[0], self.layout.hidden_bytes)

    def test_corrupted_restored_hidden_fails_real_next_projection(self):
        p, s = self.fixture()
        phase, observed = Phase(), []
        self.assertEqual(self.begin(p, s, phase), 0)
        self.assertEqual(self.layer(p, s, phase, 0, self.callback(observed)), 0)
        hidden = self.workspace.ctypes.data + self.layout.hidden_offset[1]
        original = ct.string_at(hidden, self.layout.hidden_bytes)
        ct.c_uint16.from_address(hidden).value = 0x3f80
        wanted = []
        @Projection
        def reject_changed(_, layer, logical, incoming, rows, k, width, output):
            wanted.append(ct.string_at(incoming, rows * k * 2))
            return -1
        self.assertEqual(self.layer(p, s, phase, 1, reject_changed), -2)
        self.assertEqual(phase.active, 2)
        self.assertNotEqual(wanted[0], original[:len(wanted[0])])
        self.assertEqual((s.position, s.bank, s.next_token), (128, 0, 7))

    def test_wrong_phase_layer_and_changed_committed_state_rejected_before_projection(self):
        p, s = self.fixture()
        phase, observed = Phase(), []
        project = self.callback(observed)
        self.assertEqual(self.layer(p, s, phase, 0, project), -1)
        self.assertEqual(self.begin(p, s, phase), 0)
        self.assertEqual(self.begin(p, s, phase), -1)
        self.assertEqual(self.finish(p, s, phase, project), -1)
        self.assertEqual(self.layer(p, s, phase, 1, project), -1)
        s.bank = 1
        self.assertEqual(self.layer(p, s, phase, 0, project), -1)
        s.bank = 0
        phase.reserved = 1
        self.assertEqual(self.layer(p, s, phase, 0, project), -1)
        self.assertEqual(observed, [])

    def test_failed_phase_cannot_resume_or_commit(self):
        p, s = self.fixture()
        phase, observed = Phase(), []
        project = self.callback(observed, fail=True)
        self.assertEqual(self.begin(p, s, phase), 0)
        self.assertEqual(self.layer(p, s, phase, 0, project), -2)
        self.assertEqual(phase.active, 2)
        self.assertEqual(self.layer(p, s, phase, 0, project), -1)
        self.assertEqual(self.finish(p, s, phase, project), -1)
        self.assertEqual((s.position, s.bank, s.next_token), (128, 0, 7))
        self.assertEqual(len(observed), 1)



class InventoryTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TINY_FULL_CAPTURE"), "actual full Tiny capture required")
    def test_complete_inventory_and_missing_kv_panel_or_logit_controls(self):
        data = json.loads(Path(os.environ["TINY_FULL_CAPTURE"]).read_text())
        validate_inventory(data)
        for kind in ("kv", "panel", "logit", "site"):
            changed = copy.deepcopy(data)
            if kind == "kv":
                changed["target_trace"][1]["cache"].pop()
            elif kind == "panel":
                changed["calls"][4]["panels"].pop()
            elif kind == "logit":
                changed["target_trace"][2]["logits_payload"]["shape"] = [31999]
            else:
                changed["calls"][155]["site_index"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_inventory(changed)


if __name__ == "__main__":
    unittest.main()
