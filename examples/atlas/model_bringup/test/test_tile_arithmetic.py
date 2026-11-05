"""Portable controls for the tile oracle, without importing npu_model."""

import importlib.util
import pathlib
import random
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import kernel_library  # noqa: E402,F401
from atlas_tile_arithmetic import (bf16_round, f32_to_bf16_bits, mxu0_dot, output_halves, pack_bf16_unit, pack_bf16,
    rescale_bf16, tensor_scale_exponent)


class TileArithmeticTest(unittest.TestCase):
    def test_bf16_and_fp8_ties(self):
        self.assertEqual(f32_to_bf16_bits(1 + 2**-8), 0x3F80)
        self.assertEqual(f32_to_bf16_bits(1 + 3 * 2**-8), 0x3F82)
        self.assertEqual(pack_bf16_unit(0x3F88), 0x38)  # 1.0625 -> 1
        self.assertEqual(pack_bf16_unit(0x3F98), 0x3A)  # 1.1875 -> 1.25
        self.assertEqual(pack_bf16_unit(0xBF88), 0xB8)

    def test_vpu_pack_boundaries(self):
        self.assertEqual(pack_bf16_unit(f32_to_bf16_bits(2**-7)), 0)
        self.assertEqual(pack_bf16_unit(f32_to_bf16_bits(2**-6)), 8)
        self.assertEqual(pack_bf16_unit(f32_to_bf16_bits(480)), 0x7E)
        self.assertEqual(pack_bf16_unit(0x8000), 0)
        with self.assertRaises(ValueError):
            pack_bf16_unit(0x7FC0)

    def test_order_and_continuation_discriminate_reset(self):
        a, b = [0x38, 0x18, 0x18], [0x38, 0x18, 0x18]
        self.assertEqual(mxu0_dot(a, b), 0x3F80)
        self.assertNotEqual(mxu0_dot(a, b), mxu0_dot(a[1:], b[1:]))
        self.assertEqual(bf16_round(Fraction(1) - Fraction(1, 256)), 0x3F7F)

    def test_distinct_output_halves(self):
        rows = [[row * 32 + col for col in range(32)] for row in range(32)]
        payload = output_halves(rows)
        self.assertEqual(len(payload), 2048)
        self.assertEqual(struct.unpack_from("<H", payload, 1024)[0], 16)
        self.assertEqual(struct.unpack_from("<H", payload, 32)[0], 32)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "admission checks require NumPy from model environment")
    def test_input_admission_and_raw_bf16_authentication(self):
        import numpy as np
        from run_model_tile import validate_arrays

        values = {"A": np.ones((32, 64), dtype=np.float32), "B": np.ones((64, 32), dtype=np.float32),
            "native_C": np.ones((32, 32), dtype=np.float32)}
        validate_arrays(values, "bfloat16")
        values["A"][0, 0] = np.float32(1.001)
        with self.assertRaises(ValueError):
            validate_arrays(values, "bfloat16")
        validate_arrays(values, "float32")
        values["A_bf16"] = np.zeros((32, 64), dtype=np.uint16)
        with self.assertRaises(ValueError):
            validate_arrays(values, "float32")
        values.pop("A_bf16")
        values["B"] = np.ones((63, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_arrays(values, "float32")

    def test_cached_and_uncached_recurrences_agree(self):
        rng = random.Random(731)
        codes = [0] + [code for code in range(256) if 0 < (code >> 3) & 15 < 15]
        for _ in range(32):
            a = [rng.choice(codes) for _ in range(64)]
            b = [rng.choice(codes) for _ in range(64)]
            self.assertEqual(mxu0_dot(a, b), mxu0_dot(a, b, cached=False))

    def test_model_binding_rejects_wrong_pin_and_dirty_source(self):
        from run_model_tile import MODEL_PIN, model_binding

        with patch("run_model_tile.subprocess.check_output", return_value="wrong\n"):
            with self.assertRaisesRegex(ValueError, "revision"):
                model_binding(pathlib.Path("/tmp"))
        with patch("run_model_tile.subprocess.check_output", return_value=MODEL_PIN), patch("run_model_tile.subprocess.run",
            return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(ValueError, "tracked source"):
                model_binding(pathlib.Path("/tmp"))

    def test_tensor_scale_and_explicit_output_rescale(self):
        self.assertEqual(tensor_scale_exponent([f32_to_bf16_bits(448)]), 0)
        self.assertEqual(tensor_scale_exponent([f32_to_bf16_bits(224)]), -1)
        self.assertEqual(tensor_scale_exponent([0, 0x8000]), 0)
        self.assertEqual(pack_bf16(f32_to_bf16_bits(224), 126), 0x7E)
        self.assertEqual(pack_bf16(f32_to_bf16_bits(-224), 126), 0xFE)
        self.assertEqual(rescale_bf16(0x4000, -1), 0x3F80)
        self.assertEqual(rescale_bf16(0x8000, -16), 0x8000)
        self.assertEqual(rescale_bf16(0, -16), 0)
        with self.assertRaisesRegex(ValueError, "factor"):
            rescale_bf16(0x4000, -127)
        with self.assertRaisesRegex(ValueError, "overflows"):
            rescale_bf16(0x7F7F, 1)

    def test_archive_names_are_flat_unique_and_explicit(self):
        from run_model_tile import validate_archive_members

        required = [SimpleNamespace(filename=name, file_size=0) for name in ("A.npy", "B.npy", "native_C.npy")]
        validate_archive_members(required)
        validate_archive_members(required + [SimpleNamespace(filename="A_bf16.npy", file_size=1)])
        for name in ("A.npy", "other.npy", "../A.npy", "folder/A.npy", "folder/", "A"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unique flat"):
                validate_archive_members(required + [SimpleNamespace(filename=name, file_size=1)])
        with self.assertRaises(ValueError):
            validate_archive_members(required[:-1])

    def test_archive_size_limit_uses_metadata_without_allocation(self):
        from run_model_tile import validate_archive_members

        members = [SimpleNamespace(filename=name, file_size=0) for name in ("A.npy", "B.npy", "native_C.npy")]
        members[0].file_size = 32 * 1024 * 1024
        validate_archive_members(members)
        members[0].file_size += 1
        with self.assertRaisesRegex(ValueError, "uncompressed size"):
            validate_archive_members(members)
        members[0].file_size = -1
        with self.assertRaisesRegex(ValueError, "uncompressed size"):
            validate_archive_members(members)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "loader control requires NumPy")
    def test_bad_archive_is_refused_before_numpy_load(self):
        from run_model_tile import load_bundle

        members = [SimpleNamespace(filename=name, file_size=0) for name in ("A.npy", "B.npy", "native_C.npy", "../A.npy")]
        args = SimpleNamespace(smoke=False, input="/tmp/not-loaded-input.npz", manifest="/tmp/not-loaded-manifest.json")
        manifest = '{"schema_version":1,"source":{"id":"test"},"site":{"id":"test"},"npz_sha256":"hash"}'
        with patch("run_model_tile.safe_path", side_effect=pathlib.Path), patch.object(pathlib.Path, "read_text",
            return_value=manifest), patch("run_model_tile.sha256",
            return_value="hash"), patch("run_model_tile.zipfile.ZipFile") as archive, patch("numpy.load",
            side_effect=AssertionError("member load")) as load:
            archive.return_value.__enter__.return_value.infolist.return_value = members
            with self.assertRaisesRegex(ValueError, "unique flat"):
                load_bundle(args)
            load.assert_not_called()

    def test_path_components_are_refused_before_normalization_or_stat(self):
        from run_model_tile import safe_path

        with patch.object(pathlib.Path, "lstat", side_effect=AssertionError("metadata access")) as access:
            for value in ("/unread/../file",):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    safe_path(value)
            access.assert_not_called()

    def test_relative_path_rechecks_absolute_components_before_stat(self):
        from run_model_tile import safe_path

        for expanded in ("/unread/../file",):
            with self.subTest(expanded=expanded), patch.object(pathlib.Path, "absolute",
                return_value=pathlib.Path(expanded)), patch.object(pathlib.Path, "lstat",
                side_effect=AssertionError("metadata access")) as access:
                with self.assertRaises(ValueError):
                    safe_path("file")
                access.assert_not_called()

    def test_archive_metadata_is_admitted_before_input_hash(self):
        import run_model_tile as runner

        members = [SimpleNamespace(filename="A.npy", external_attr=0)]
        bad_members = ([SimpleNamespace(filename=name, external_attr=0)] for name in ("../A.npy", "/A.npy", "a\\b.npy"))
        with patch("atlas_tooling.checked", side_effect=pathlib.Path), patch.object(runner.zipfile,
            "ZipFile") as archive, patch.object(pathlib.Path, "read_bytes", return_value=b"archive") as contents:
            metadata = archive.return_value.__enter__.return_value
            metadata.infolist.return_value = members
            self.assertEqual(len(runner.sha256("/not-opened.npz")), 64)
            contents.assert_called_once_with()
            contents.reset_mock()
            for entries in [*bad_members, members * 2, [SimpleNamespace(filename="A.npy", external_attr=0o120777 << 16)]]:
                metadata.infolist.return_value = entries
                with self.subTest(entries=entries), self.assertRaises(ValueError):
                    runner.input_sources(SimpleNamespace(input="/not-opened.npz", manifest=None))
            contents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
