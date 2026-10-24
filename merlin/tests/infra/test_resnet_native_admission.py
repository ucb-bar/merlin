"""Synthetic controls for standalone native ResNet admission; no model execution."""

from merlin.common.digest import sha256_bytes

import builtins
import hashlib
import importlib.util
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from merlin.common.paths import repo_root


REFRESH = repo_root() / "examples/atlas/comparisons/refresh"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load("standalone_resnet_admission_controls", REFRESH / "resnet_native.py")
capture = load("resnet_admission_provenance_controls", REFRESH / "capture_resnet.py")


class Tensor:
    def __init__(self, value, layout="strided"):
        self.value = np.asarray(value)
        self.layout, self.shape, self.dtype = layout, self.value.shape, self.value.dtype


class Model:
    def __init__(self):
        self.state = {"weight": Tensor(np.zeros((2,), dtype=np.float32))}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict):
        if not strict:
            raise AssertionError("Checkpoint loading must be strict")
        self.state = state.copy()


def fake_torch(state, mutation=None):
    def read(stream, map_location, weights_only):
        if map_location != "cpu" or weights_only is not True:
            raise AssertionError("Checkpoint admission changed safe loading options")
        if mutation:
            mutation(Path(stream.name))
        return state

    return SimpleNamespace(Tensor=Tensor, strided="strided", load=read,
                           isfinite=lambda value: np.isfinite(value.value),
                           equal=lambda left, right: np.array_equal(left.value, right.value))


class ResnetNativeAdmissionTest(unittest.TestCase):
    def test_helper_import_needs_no_numerical_framework(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"tvm", "torch", "torchvision", "numpy"}:
                raise AssertionError("Unexpected helper dependency: " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            helper = load("resnet_no_framework_import", REFRESH / "resnet_native.py")
        self.assertTrue(callable(helper.load_checkpoint))
        self.assertTrue(callable(helper.load_images))

    def test_paths_refuse_lexical_risks_before_filesystem_and_links_without_following(self):
        with patch.object(native.os, "lstat", side_effect=AssertionError("filesystem access")):
            for value in ("/not-opened/../x",):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    native.safe_path(value)
        with patch.object(native.os, "lstat", return_value=SimpleNamespace(st_mode=stat.S_IFLNK)), \
                patch.object(native.os, "readlink", side_effect=AssertionError("link target access")):
            with self.assertRaises(ValueError):
                native.safe_path("/not-opened/link/x")

    def test_images_preserve_fp32_stream_and_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            for shape in ((2, 3, 224, 224), (2, 1, 3, 224, 224)):
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                np.savez(path, images=values)
                images, identity = native.load_images(path, np)
                self.assertEqual(images.shape, (2, 1, 3, 224, 224))
                self.assertTrue(images.flags.c_contiguous)
                np.testing.assert_array_equal(images.reshape(values.shape), values)
                self.assertEqual(identity, {"path": str(path), "sha256": sha256_bytes(path.read_bytes())})

    def test_images_reject_missing_array_dtype_shape_empty_and_nonfinite(self):
        good = np.zeros((1, 3, 224, 224), dtype=np.float32)
        nonfinite = good.copy()
        nonfinite.flat[0] = np.nan
        cases = ({"other": good}, {"images": good.astype(np.float64)},
                 {"images": good[:, :, :-1]}, {"images": good[:0]}, {"images": nonfinite})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            for values in cases:
                with self.subTest(shape=next(iter(values.values())).shape):
                    np.savez(path, **values)
                    with self.assertRaises(ValueError):
                        native.load_images(path, np)

    def test_checkpoint_preserves_exact_state_and_safe_load_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.bin"
            path.write_bytes(b"synthetic checkpoint")
            state = {"weight": Tensor(np.array([1, 2], dtype=np.float32))}
            model = Model()
            identity = native.load_checkpoint(model, path, fake_torch(state))
            self.assertEqual(identity, native.file_identity(path, native.safe_path))
            np.testing.assert_array_equal(model.state["weight"].value, state["weight"].value)

    def test_checkpoint_rejects_keys_nontensor_layout_shape_dtype_and_nonfinite(self):
        cases = ([], {}, {"other": Tensor(np.zeros(2, dtype=np.float32))},
                 {"weight": object()}, {"weight": Tensor(np.zeros(2, dtype=np.float32), "sparse")},
                 {"weight": Tensor(np.zeros(3, dtype=np.float32))},
                 {"weight": Tensor(np.zeros(2, dtype=np.float64))},
                 {"weight": Tensor(np.array([np.inf, 0], dtype=np.float32))})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.bin"
            path.write_bytes(b"synthetic checkpoint")
            for state in cases:
                with self.subTest(state=state), self.assertRaises(ValueError):
                    native.load_checkpoint(Model(), path, fake_torch(state))

    def test_checkpoint_detects_file_mutation_and_changed_loaded_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.bin"
            path.write_bytes(b"before")
            state = {"weight": Tensor(np.ones(2, dtype=np.float32))}
            with self.assertRaisesRegex(ValueError, "changed while loading"):
                native.load_checkpoint(Model(), path, fake_torch(state, lambda file: file.write_bytes(b"after")))
            model = Model()
            with patch.object(model, "load_state_dict"):
                with self.assertRaisesRegex(AssertionError, "preserve"):
                    native.load_checkpoint(model, path, fake_torch(state))

    def test_historical_and_current_sources_remain_separate_and_bookended(self):
        historical = {"path": "/not-opened/historical-verifier.py", "sha256": capture.HISTORICAL_VERIFIER_SHA256}
        prior = {"path": "/not-opened/earlier-source", "sha256": "old"}
        records, report = [prior], {}
        with patch.object(capture, "identity", wraps=capture.identity) as identity:
            capture.load_native_admission({"verifier_source": historical}, records, report)
        self.assertTrue(all(Path(call.args[0]) == REFRESH / "resnet_native.py" for call in identity.call_args_list))
        self.assertEqual(records[0], prior)
        self.assertEqual(report["native_admission"]["historical_verifier_source"], historical)
        self.assertEqual(report["native_admission"]["current_helper"], records[-1])
        capture.require_identity(records[-1])

    def test_wrong_historical_verifier_is_refused_before_import(self):
        with patch.object(capture, "load_module") as imported:
            with self.assertRaisesRegex(ValueError, "historical verifier"):
                capture.load_native_admission({"verifier_source": {"path": "/not-opened/old.py", "sha256": "0" * 64}}, [], {})
        imported.assert_not_called()

    def test_current_helper_mutation_during_import_is_refused(self):
        historical = {"path": "/not-opened/old.py", "sha256": capture.HISTORICAL_VERIFIER_SHA256}
        records, report = [], {}
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "resnet_native.py"
            helper.write_text("VALUE = 1\n")

            def mutate(name, path):
                path.write_text("VALUE = 2\n")
                return SimpleNamespace()

            with patch.object(capture, "__file__", str(Path(directory) / "capture_resnet.py")), \
                    patch.object(capture, "load_module", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "Source binding changed"):
                    capture.load_native_admission({"verifier_source": historical}, records, report)
        self.assertEqual(records, [])
        self.assertEqual(report, {})


if __name__ == "__main__":
    unittest.main()
