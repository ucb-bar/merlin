"""Bounded DATA integrity controls; no compiler, SDK, extraction or archived imports."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from merlin.verify import graph_artifact_bundle as bundle


def pin(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(
            tempfile.mkdtemp(prefix="artifact-bundle-", dir=os.environ.get("MERLIN_GRAPH_ARTIFACT_TEST_TMPDIR"))
        )

    def archive(self, entries, filename="fixture.tar.gz"):
        path = self.directory / filename
        with tarfile.open(path, "w:gz") as stream:
            for entry in entries:
                name, raw = entry[:2]
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                if len(entry) > 2:
                    info.type = entry[2]
                    info.linkname = "data.bin" if info.type in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                stream.addfile(info, io.BytesIO(raw))
        return path

    def case(self, *, index=None, entries=None, raw_index=None):
        data = b"data"
        if raw_index is None:
            raw_index = json.dumps(
                index if index is not None else {"files": [{"path": "data.bin", **pin(data)}]}
            ).encode()
        entries = [("index.json", raw_index), ("data.bin", data)] if entries is None else entries(raw_index)
        archive = self.archive(entries)
        specification = {
            "mode": "original-full",
            "archive": pin(archive.read_bytes()),
            "index": {"member": "index.json", "format": "files", "indexed_files": 1, **pin(raw_index)},
        }
        return archive, specification

    def test_full_original_and_integrity_only(self):
        archive, spec = self.case()
        result = bundle.verify_bundle(archive, spec)
        self.assertEqual(result["original_member_pins_checked"], 1)
        self.assertEqual(result["archive_members"], 2)
        self.assertTrue(result["integrity_only"])
        for key in (
            "proof_checked",
            "qualification_verified",
            "human_review_approved",
            "native_execution_verified",
            "sender_authenticated",
            "historical_replay_performed",
        ):
            self.assertIs(result[key], False)

    def test_wrong_transport_digest(self):
        archive, spec = self.case()
        spec["archive"]["sha256"] = "0" * 64
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec)

    def test_missing_extra_duplicate_and_case_collision(self):
        changes = {
            "missing": lambda raw: [("index.json", raw)],
            "extra": lambda raw: [("index.json", raw), ("data.bin", b"data"), ("extra", b"x")],
            "duplicate": lambda raw: [("index.json", raw), ("data.bin", b"data"), ("data.bin", b"data")],
            "case-collision": lambda raw: [("index.json", raw), ("data.bin", b"data"), ("DATA.bin", b"data")],
        }
        for name, entries in changes.items():
            with self.subTest(case=name):
                archive, spec = self.case(entries=entries)
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_changed_payload_hash_and_size(self):
        for data in (b"evil", b"different-size"):
            with self.subTest(payload=data):
                archive, spec = self.case(entries=lambda raw: [("index.json", raw), ("data.bin", data)])
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_original_index_identity(self):
        archive, spec = self.case()
        spec["index"]["sha256"] = "0" * 64
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec)

    def test_unsafe_and_nonportable_archive_paths(self):
        for name in ("../escape", "/absolute", "a/../escape", "a//b", "a\\b", "CON.txt", "space ", "cafe\u0301"):
            with self.subTest(path=name):
                archive, spec = self.case(
                    entries=lambda raw: [("index.json", raw), ("data.bin", b"data"), (name, b"x")]
                )
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_links_and_nonregular_members(self):
        for kind in (
            tarfile.SYMTYPE,
            tarfile.LNKTYPE,
            tarfile.DIRTYPE,
            tarfile.FIFOTYPE,
            tarfile.CHRTYPE,
            tarfile.BLKTYPE,
        ):
            with self.subTest(kind=kind):
                archive, spec = self.case(
                    entries=lambda raw: [("index.json", raw), ("data.bin", b"data"), ("special", b"", kind)]
                )
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_file_directory_prefix_collision(self):
        archive, spec = self.case(
            entries=lambda raw: [("index.json", raw), ("data.bin", b"data"), ("data.bin/child", b"x")]
        )
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec)

    def test_logical_absolute_references_are_not_opened(self):
        index = {"files": [{"path": "/never-open-this-logical-root/data.bin", **pin(b"data")}]}
        archive, spec = self.case(index=index)
        spec["original_root_prefix"] = "/never-open-this-logical-root/"
        self.assertEqual(bundle.verify_bundle(archive, spec)["original_member_pins_checked"], 1)
        del spec["original_root_prefix"]
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec)

    def test_malformed_original_paths_and_types(self):
        for row in (
            None,
            "file",
            {"path": 7, **pin(b"data")},
            {"path": "../escape", **pin(b"data")},
            {"path": "data.bin", "sha256": 5, "bytes": 4},
            {"path": "data.bin", "sha256": "0" * 64, "bytes": True},
        ):
            with self.subTest(row=row):
                archive, spec = self.case(index={"files": [row]})
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)
        for index in ({"files": "wrong"}, 5, [None]):
            with self.subTest(index=index):
                archive, spec = self.case(index=index)
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_malformed_specification_types(self):
        archive, good = self.case()
        variants = [
            None,
            [],
            {**good, "mode": []},
            {**good, "mode": "unknown"},
            {**good, "archive": []},
            {**good, "members": []},
            {**good, "original_root_prefix": 8},
            {**good, "original_root_prefix": "/a/../b/"},
            {**good, "index": None},
        ]
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, value)
        for field, value in (("bytes", -1), ("bytes", True), ("sha256", "G" * 64)):
            bad = copy.deepcopy(good)
            bad["archive"][field] = value
            with self.assertRaises(bundle.BundleError):
                bundle.verify_bundle(archive, bad)

    def test_duplicate_json_keys_and_invalid_json(self):
        for raw in (b'{"files":[],"files":[]}', b'{"files":[NaN]}', b"{bad", b"\xff"):
            with self.subTest(raw=raw):
                archive, spec = self.case(raw_index=raw)
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_list_and_dictionary_file_inventories(self):
        row = {"path": "data.bin", **pin(b"data")}
        for index in ([row], {"files": {"logical-record-name": row}}):
            with self.subTest(index=index):
                archive, spec = self.case(index=index)
                self.assertEqual(bundle.verify_bundle(archive, spec)["original_member_pins_checked"], 1)

    def test_external_index_binding_and_path_escape(self):
        raw = json.dumps({"group": {"data": {"path": "data.bin", **pin(b"data")}}}).encode()
        archive = self.archive([("data.bin", b"data")])
        (self.directory / "external.json").write_bytes(raw)
        spec = {
            "mode": "external-originals",
            "archive": pin(archive.read_bytes()),
            "index": {"file": "external.json", "format": "external-retention", "indexed_files": 1, **pin(raw)},
        }
        self.assertEqual(
            bundle.verify_bundle(archive, spec, metadata_directory=self.directory)["original_member_pins_checked"], 1
        )
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec)
        for name in ("../external.json", "/external.json"):
            spec["index"]["file"] = name
            with self.assertRaises(bundle.BundleError):
                bundle.verify_bundle(archive, spec, metadata_directory=self.directory)

    def test_external_index_symlink_escape(self):
        outside = self.directory / "outside"
        outside.mkdir()
        metadata = self.directory / "metadata"
        metadata.mkdir()
        raw = b'{"files": []}'
        (outside / "index.json").write_bytes(raw)
        (metadata / "link").symlink_to(outside, target_is_directory=True)
        archive = self.archive([])
        spec = {
            "mode": "external-originals",
            "archive": pin(archive.read_bytes()),
            "index": {"file": "link/index.json", "format": "files", "indexed_files": 0, **pin(raw)},
        }
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec, metadata_directory=metadata)

    def test_selection_checks_original_pins_not_only_transport(self):
        raw = json.dumps({"files": [{"path": "data.bin", **pin(b"data")}]}).encode()
        for data in (b"data", b"evil"):
            archive = self.archive(
                [("payload/index.json", raw), ("payload/data.bin", data), ("README", b"new navigation")]
            )
            spec = {
                "mode": "worked-selection",
                "archive": pin(archive.read_bytes()),
                "member_prefix": "payload/",
                "members": {
                    "payload/index.json": pin(raw),
                    "payload/data.bin": pin(data),
                    "README": pin(b"new navigation"),
                },
                "selected_original_files": 2,
                "index": {"member": "payload/index.json", "format": "files", "indexed_files": 1, **pin(raw)},
            }
            if data == b"data":
                self.assertEqual(bundle.verify_bundle(archive, spec)["original_member_pins_checked"], 2)
            else:
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_bundle(archive, spec)

    def test_transport_only_requires_complete_member_pins(self):
        archive = self.archive([("data.bin", b"data")])
        spec = {"mode": "transport-only", "archive": pin(archive.read_bytes()), "members": {"data.bin": pin(b"data")}}
        self.assertEqual(bundle.verify_bundle(archive, spec)["original_member_pins_checked"], 0)
        del spec["members"]
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec)

    def test_archive_symlink_and_index_limit(self):
        archive, spec = self.case()
        link = self.directory / "linked.tar.gz"
        link.symlink_to(archive)
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(link, spec)
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(archive, spec, max_index_bytes=1)

    def test_cli_independent_of_cwd_and_source_location(self):
        archive, spec = self.case()
        record = {**spec, "id": "sample"}
        specification = self.directory / "transfer.json"
        specification.write_text(json.dumps({"transports": [record]}))
        elsewhere = self.directory / "unrelated"
        elsewhere.mkdir()
        output = self.directory / "result.json"
        previous = Path.cwd()
        try:
            os.chdir(elsewhere)
            with contextlib.redirect_stdout(io.StringIO()):
                code = bundle.main(
                    [
                        "--archive",
                        str(archive),
                        "--specification",
                        str(specification),
                        "--record",
                        "sample",
                        "--output",
                        str(output),
                    ]
                )
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["status"], "integrity_matched")

    def test_cli_failures_are_json_and_do_not_overwrite_inputs(self):
        archive, spec = self.case()
        specification = self.directory / "spec.json"
        specification.write_text(json.dumps(spec))
        original = specification.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = bundle.main(
                ["--archive", str(archive), "--specification", str(specification), "--output", str(specification)]
            )
        self.assertEqual(code, 1)
        self.assertEqual(specification.read_bytes(), original)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "integrity_failed")
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = bundle.main(["--archive", str(self.directory / "missing"), "--specification", str(specification)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "unavailable")

    def test_actual_retained_worked_transport_once(self):
        archive = os.environ.get("MERLIN_GRAPH_ARTIFACT_TEST_ARCHIVE")
        specification = os.environ.get("MERLIN_GRAPH_ARTIFACT_TEST_SPEC")
        if not archive or not specification:
            self.skipTest("Explicit retained transport/spec paths not supplied; no discovery or regeneration")
        document = bundle.load_document(Path(specification).read_bytes())
        record = next(
            row
            for row in document["transports"]
            if row["id"] == os.environ.get("MERLIN_GRAPH_ARTIFACT_TEST_RECORD", "worked-example")
        )
        result = bundle.verify_bundle(Path(archive), record, metadata_directory=Path(specification).parent)
        self.assertEqual(result["archive_members"], 196)
        self.assertEqual(result["original_member_pins_checked"], 192)
        self.assertFalse(result["proof_checked"])

    def output_alias(self, target, kind, name):
        if kind == "direct":
            return target
        alias = self.directory / name
        if kind == "symlink":
            alias.symlink_to(target)
        else:
            os.link(target, alias)
        return alias

    def assert_cli_preserves_alias(self, archive, specification, target, output):
        inputs = {path: path.read_bytes() for path in (archive, specification, target)}
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = bundle.main(
                ["--archive", str(archive), "--specification", str(specification), "--output", str(output)]
            )
        self.assertEqual(code, 1)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "integrity_failed")
        self.assertIn("overwrite an input", report["error"])
        self.assertFalse(report["proof_checked"])
        for path, raw in inputs.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_cli_direct_inputs_preserved_through_output_aliases(self):
        archive, spec = self.case()
        specification = self.directory / "spec.json"
        specification.write_text(json.dumps(spec))
        for target in (archive, specification):
            for kind in ("direct", "symlink", "hardlink"):
                with self.subTest(input=target.name, alias=kind):
                    output = self.output_alias(target, kind, target.name + "." + kind)
                    self.assert_cli_preserves_alias(archive, specification, target, output)

    def external_cli_case(self):
        raw = json.dumps({"files": [{"path": "data.bin", **pin(b"data")}]}).encode()
        archive = self.archive([("data.bin", b"data")])
        index = self.directory / "external.json"
        index.write_bytes(raw)
        spec = {
            "mode": "external-originals",
            "archive": pin(archive.read_bytes()),
            "index": {"file": index.name, "format": "files", "indexed_files": 1, **pin(raw)},
        }
        specification = self.directory / "spec.json"
        specification.write_text(json.dumps(spec))
        return archive, specification, index, spec

    def test_cli_external_index_preserved_through_output_aliases(self):
        archive, specification, index, spec = self.external_cli_case()
        for kind in ("direct", "symlink", "hardlink"):
            with self.subTest(alias=kind):
                output = self.output_alias(index, kind, "external." + kind)
                self.assert_cli_preserves_alias(archive, specification, index, output)
        # A digest failure must also preserve the selected metadata input.
        spec["index"]["sha256"] = "0" * 64
        specification.write_text(json.dumps(spec))
        self.assert_cli_preserves_alias(archive, specification, index, index)

    def test_cli_transfer_selected_external_index_preserved(self):
        archive, specification, index, spec = self.external_cli_case()
        specification.write_text(json.dumps({"transports": [{**spec, "id": "selected"}]}))
        original = index.read_bytes()
        output = self.output_alias(index, "hardlink", "selected-alias")
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = bundle.main(
                [
                    "--archive",
                    str(archive),
                    "--specification",
                    str(specification),
                    "--record",
                    "selected",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "integrity_failed")
        self.assertEqual(index.read_bytes(), original)

    def test_cli_external_index_safe_output(self):
        archive, specification, index, _ = self.external_cli_case()
        original = index.read_bytes()
        output = self.directory / "safe-result.json"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = bundle.main(
                ["--archive", str(archive), "--specification", str(specification), "--output", str(output)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(index.read_bytes(), original)
        report = json.loads(output.read_text())
        self.assertEqual(report, json.loads(stdout.getvalue()))
        self.assertEqual(report["status"], "integrity_matched")
        self.assertEqual(report["original_member_pins_checked"], 1)
        self.assertFalse(report["qualification_verified"])


if __name__ == "__main__":
    unittest.main()
