"""Focused full Tiny model geometry and final-state admission controls."""
import copy
from pathlib import Path
import sys
import struct
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/host_iss")]
import tiny_full_contract as tiny
from tiny_contract_fixture import tiny_contract_fixture


def fixture():
    return tiny_contract_fixture(2, (-7, -8, -15), [17, 18, 19], 465)


class TinyFullContractTest(unittest.TestCase):
    def test_actual_reference_must_be_writable_loaded_and_disjoint(self):
        image = mock.Mock()
        image.symbols = {"actual": {"address": 0x1800, "size": 4},
                         "expected": {"address": 0x1100, "size": 0}}
        image.segments = [{"address": 0x1000, "memory_bytes": 0x1000, "flags": 6}]
        items = [{"actual_symbol": "actual", "expected_symbol": "expected", "bytes": 4}]
        self.assertEqual(tiny.reference_locations(image, items), [(0x1800, 4)])
        for address, flags in ((0x3000, 6), (0x1800, 4), (0x1100, 6), (0x1102, 6)):
            image.symbols["actual"]["address"] = address
            image.segments[0]["flags"] = flags
            with self.subTest(address=address, flags=flags), self.assertRaises(ValueError):
                tiny.reference_locations(image, items)
        image.symbols["actual"]["address"] = 0x1800
        image.segments[0]["flags"] = 6
        with self.assertRaises(ValueError):
            tiny.reference_locations(image, items, [(0x1802, 8)])
        image.bytes_at.assert_not_called()

    def test_raw_bf16_nonfinite_reference_refused(self):
        self.assertTrue(tiny.finite_bf16(struct.pack("<4H", 0, 0x8000, 0x7f7f, 0xff7f)))
        for word in (0x7f80, 0xff80, 0x7fc0, 0xffc1):
            with self.subTest(word=word):
                self.assertFalse(tiny.finite_bf16(struct.pack("<H", word)))
        self.assertFalse(tiny.finite_bf16(b"\x00"))

    def test_complete_22_layer_two_decode_geometry(self):
        data, provenance = fixture()
        plan = tiny.validate_plan(data, provenance)
        self.assertEqual(len(plan), 840)
        self.assertEqual(len(data["host_parameters"]), 49)
        self.assertEqual(len(data["reference_buffers"]), 137)
        self.assertEqual(plan[0]["physical_mkn"], [128, 2048, 2048])
        self.assertEqual(plan[263]["site"], "model.layers.21.mlp.down_proj")
        self.assertEqual(plan[264]["site"], "lm_head")
        self.assertEqual(plan[279]["columns"], [30720, 32000])
        self.assertEqual(plan[280]["physical_mkn"], [32, 2048, 2048])
        self.assertEqual(plan[-1]["step"], 2)

    def test_missing_or_wrong_layer_role_geometry_dma_scale_rejected(self):
        mutations = (
            lambda d: d["kernels"].pop(),
            lambda d: d["kernels"][12].update(site="model.layers.0.self_attn.q_proj"),
            lambda d: d["kernels"][279].update(columns=[30720, 31968]),
            lambda d: d["kernels"][280].update(physical_mkn=[128, 2048, 2048]),
            lambda d: d["kernels"][0]["requested_dma"].update(load_bytes=32),
            lambda d: d["kernels"][0].update(rescale=-14),
            lambda d: d.update(projection_calls=464),
            lambda d: d.update(decode_count=True),
        )
        for mutate in mutations:
            data, provenance = fixture()
            mutate(data)
            provenance["kernels"] = copy.deepcopy(data["kernels"])
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, KeyError)):
                tiny.validate_plan(data, provenance)

    def test_all_steps_all22_cache_logits_tokens_parameters_required(self):
        mutations = (
            lambda d: d["reference_buffers"].pop(),
            lambda d: d["reference_buffers"][0].update(shape=[1, 4, 127, 64]),
            lambda d: d["reference_buffers"][0].update(dtype="float32"),
            lambda d: d["reference_buffers"][0].update(actual_symbol="tiny_full_cache_00_22_keys"),
            lambda d: d["host_parameters"].pop("model.layers.21.post_attention_layernorm.weight"),
            lambda d: d["host_parameters"]["model.embed_tokens.weight"].update(shape=[31999, 2048]),
            lambda d: d.update(greedy_tokens=[17, 18, 32000]),
        )
        for mutate in mutations:
            data, provenance = fixture()
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, KeyError)):
                tiny.validate_plan(data, provenance)

    def test_trap_and_unknown_stop_fail_before_read_or_output(self):
        data, _ = fixture()
        remote = mock.Mock()
        for pc in (0x200, 0x300):
            remote.reg.return_value = pc
            with mock.patch.object(tiny, "contract", return_value=data), \
                    mock.patch.object(tiny, "stop_sites", return_value={0x100: 4, 0x200: 2}):
                with self.assertRaises(ValueError):
                    tiny.verify_completion(remote, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
            remote.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
