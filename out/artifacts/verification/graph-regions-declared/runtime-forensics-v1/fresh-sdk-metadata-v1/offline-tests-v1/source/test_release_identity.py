"""Offline tests using already-retained primary JSON metadata."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from release_identity import select


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        b = Path(__file__).parent / "official-metadata-v1"
        self.release = json.loads((b / "release.json").read_bytes())
        self.tag = json.loads((b / "tag.json").read_bytes())

    def test_exact_retained_identity(self):
        value = select(self.release, self.tag)
        self.assertEqual(value["commit"], "eb6cafb7290e8bd5f2937522d8d804f58f02bafc")
        self.assertEqual(value["compressed_bytes"], 925899528)
        self.assertEqual(value["publisher_platform_sha256"], "c8531faaa4a93d5a4dae0c742cbc1bd42a2993594d02f93a9d629ebf299fb2c8")
        self.assertFalse(value["native_authorized"])

    def test_duplicate_or_missing_asset(self):
        asset = next(x for x in self.release["assets"] if x["name"] == "lean-4.35.0-linux.zip")
        for assets in ([], [asset, asset]):
            changed = deepcopy(self.release)
            changed["assets"] = assets
            with self.assertRaises(ValueError):
                select(changed, self.tag)

    def test_wrong_platform(self):
        self.release["assets"] = [a for a in self.release["assets"] if "aarch64" in a["name"]]
        with self.assertRaises(ValueError):
            select(self.release, self.tag)

    def test_changed_tag_or_noncommit(self):
        for change in ({"ref": "refs/heads/master"}, {"object": {"type": "tag", "sha": "f" * 40}}):
            tag = {**self.tag, **change}
            with self.assertRaises(ValueError):
                select(self.release, tag)

    def test_missing_digest_or_redirect_url(self):
        for fields in ({"digest": None}, {"digest": "sha256:" + "x" * 64},
                       {"browser_download_url": "https://example.com/lean.zip"}):
            release = deepcopy(self.release)
            next(a for a in release["assets"] if a["name"] == "lean-4.35.0-linux.zip").update(fields)
            with self.assertRaises(ValueError):
                select(release, self.tag)

    def test_size_bounds(self):
        for size in (True, 0, 2**30 + 1):
            release = deepcopy(self.release)
            next(a for a in release["assets"] if a["name"] == "lean-4.35.0-linux.zip")["size"] = size
            with self.assertRaises(ValueError):
                select(release, self.tag)


if __name__ == "__main__":
    unittest.main()
