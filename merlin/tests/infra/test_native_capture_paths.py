"""Path admission must refuse restricted names and links before following them."""

import importlib.util
import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from merlin.common.paths import repo_root


path = repo_root() / "examples/atlas/comparisons/capture_tinyllama_native.py"
spec = importlib.util.spec_from_file_location("native_capture_paths", path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class NativeCapturePathTest(unittest.TestCase):
    def test_parent_traversal_is_refused_before_filesystem(self):
        with patch.object(capture.os, "lstat", side_effect=AssertionError("filesystem access")):
            with self.assertRaises(ValueError):
                capture.safe_path("/not-opened/../destination")

    def test_link_or_link_loop_is_refused_without_reading_target(self):
        visited = []

        def lstat(value):
            visited.append(str(value))
            return SimpleNamespace(st_mode=stat.S_IFLNK if value.name == "link" else stat.S_IFDIR)

        with patch.object(capture.os, "lstat", side_effect=lstat), patch.object(
            capture.os, "readlink", side_effect=AssertionError("link target access")
        ):
            with self.assertRaises(ValueError):
                capture.safe_path("/not-opened/link/descendant")
        self.assertEqual(visited, ["/not-opened", "/not-opened/link"])

    def test_missing_output_suffix_does_not_trigger_more_access(self):
        visited = []

        def lstat(value):
            visited.append(str(value))
            raise FileNotFoundError(value)

        with patch.object(capture.os, "lstat", side_effect=lstat):
            self.assertEqual(capture.safe_path("/not-opened/output/file"), Path("/not-opened/output/file"))
        self.assertEqual(visited, ["/not-opened"])

    def test_source_bookends_visit_only_declared_file_records(self):
        metadata = {"files": [{"path": "/not-opened/a", "sha256": "a"},
                              {"nested": {"path": "/not-opened/b", "sha256": "b"}}],
                    "array": {"shape": [32, 32], "sha256": "tensor"}}
        with patch.object(capture, "identity") as check:
            self.assertEqual(capture.verify_declared_sources(metadata), 2)
        self.assertEqual(check.call_args_list, [unittest.mock.call("/not-opened/a", "a"),
                                               unittest.mock.call("/not-opened/b", "b")])

    def test_source_mutation_fails_bookends(self):
        with patch.object(capture, "identity", side_effect=ValueError("source changed")):
            with self.assertRaises(ValueError):
                capture.verify_declared_sources({"path": "/not-opened/a", "sha256": "old"})


if __name__ == "__main__":
    unittest.main()
