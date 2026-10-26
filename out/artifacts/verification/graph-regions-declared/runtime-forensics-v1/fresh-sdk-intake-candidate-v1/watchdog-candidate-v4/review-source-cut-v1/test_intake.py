"""Tiny ZIPs + injected fake HTTP only. The real HTTPS class is never created."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import context_schema
import fdio
import http_fetch
import intake
import policy
import zip_intake

TOP = "lean-4.35.0-linux"


def archive_bytes(entries=None, compression=zipfile.ZIP_STORED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, raw, mode, extra in entries or [(TOP + "/A.ir", b"alpha", stat.S_IFREG | 0o644, b""),
                                                  (TOP + "/A.ir.sig", b"opaque", stat.S_IFREG | 0o644, b"")]:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            info.extra = extra
            info.compress_type = compression
            archive.writestr(info, raw)
    return output.getvalue()


class Response:
    def __init__(self, raw=b"", status=200, headers=None, error_at=None, close_error=False):
        self.raw, self.status, self.headers = io.BytesIO(raw), status, headers or {}
        self.error_at, self.calls, self.closed, self.close_error = error_at, 0, False, close_error

    def read(self, n):
        self.calls += 1
        if self.error_at == self.calls:
            raise OSError("injected response read error")
        return self.raw.read(n)

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError("injected response close error")


class Backend:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def open(self, url, timeout):
        self.calls.append({"url": url, "timeout": timeout})
        if not self.responses:
            raise AssertionError("unexpected request/retry")
        out = self.responses.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        plan_path = Path(json.loads((Path(__file__).parent / "PLAN-REF.json").read_bytes())["path"])
        self.plan = json.loads(plan_path.read_bytes())
        self.plan["new_root"] = str(self.root / "intake")
        self.plan["extraction_proposal"].update(member_count_max=10, per_member_bytes_max=64,
                                               total_uncompressed_bytes_max=128, disk_reserve_bytes=1024**2)
        self.plan["archive_fetch"]["chunk_bytes"] = 32
        self.raw = archive_bytes()
        self.bind_synthetic(self.raw)

    def bind_synthetic(self, raw):
        """Only new tiny fixtures get their own test content identity, never real pins."""
        h = hashlib.sha256(raw).hexdigest()
        self.plan["archive_fetch"].update(expected_sha256=h, exact_bytes=len(raw), max_read_bytes_including_sentinel=len(raw) + 1)
        self.plan["identity"].update(publisher_platform_sha256=h, compressed_bytes=len(raw))
        self.request = policy.request_from_plan(self.plan, {"synthetic": True})

    def run_one(self, backend=None, **kwargs):
        return intake.run(self.request, backend or Backend(Response(self.raw)), **kwargs)

    def refused(self, backend=None, **kwargs):
        with self.assertRaises(fdio.IntakeStop) as error:
            self.run_one(backend, **kwargs)
        self.assertFalse((self.root / "intake/MEMBER-MANIFEST.json").exists())
        self.assertTrue((self.root / "intake/FIRST-FAILURE.json").exists())
        self.assertFalse(error.exception.evidence["accepted"])
        return error.exception.evidence

    def bad_zip(self, raw):
        self.raw = raw
        self.bind_synthetic(raw)
        return self.refused()

    def test_positive_parent_binding_complete_manifest(self):
        value = self.run_one()
        self.assertEqual(len(value["members"]), 2)
        self.assertEqual(value["parent_archive"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertTrue(value["new_environment_pins"])
        self.assertFalse(value["old_expectations_replaced"])
        for row in value["members"]:
            p = self.root / "intake/data" / row["path"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(p.stat().st_mode & 0o777, 0o400)

    def test_allowed_redirect(self):
        backend = Backend(Response(status=302, headers={"Location": "https://release-assets.githubusercontent.com/archive?token=fixture"}), Response(self.raw))
        self.run_one(backend)
        self.assertEqual(len(backend.calls), 2)

    def test_redirect_limit(self):
        backend = Backend(*(Response(status=302, headers={"Location": "https://github.com/next"}) for _ in range(4)))
        self.assertIn("redirect bound", self.refused(backend)["error"])
        self.assertEqual(len(backend.calls), 4)

    def test_redirect_bad_host(self):
        backend = Backend(Response(status=302, headers={"Location": "https://evil.example/file"}))
        self.assertIn("HTTPS", self.refused(backend)["error"])
        self.assertEqual(len(backend.calls), 1)

    def test_redirect_http_scheme(self):
        self.refused(Backend(Response(status=302, headers={"Location": "http://github.com/file"})))

    def test_url_credentials_port_fragment(self):
        for url in ("https://u:p@github.com/file", "https://github.com:444/file", "https://github.com/file#fragment"):
            with self.assertRaises(ValueError):
                policy.https_url(url, self.request["fetch"]["https_hosts"])

    def test_tls_backend_failure(self):
        backend = Backend(OSError("injected certificate verification failure"))
        self.assertIn("certificate", self.refused(backend)["error"])
        self.assertEqual(len(backend.calls), 1)

    def test_non200(self):
        response = Response(b"error-body-not-read", status=503)
        self.refused(Backend(response))
        self.assertEqual(response.calls, 0)

    def test_content_length_mismatch(self):
        response = Response(self.raw, headers={"Content-Length": "1"})
        self.assertIn("length", self.refused(Backend(response))["error"])
        self.assertEqual(response.calls, 0)

    def test_encoded_body(self):
        self.refused(Backend(Response(self.raw, headers={"Content-Encoding": "gzip"})))

    def test_short_response_retains_prefix(self):
        self.assertIn("short", self.refused(Backend(Response(self.raw[:10])))["error"])
        self.assertEqual((self.root / "intake/download/lean-4.35.0-linux.zip.part").read_bytes(), self.raw[:10])

    def test_oversize_retains_sentinel(self):
        self.assertIn("exceeds", self.refused(Backend(Response(self.raw + b"x")))["error"])
        self.assertEqual((self.root / "intake/download/lean-4.35.0-linux.zip.part").stat().st_size, len(self.raw) + 1)

    def test_digest_failure_before_any_zip_processing(self):
        changed = b"x" + self.raw[1:]
        with patch.object(zip_intake, "central_bound") as parsed:
            self.assertIn("digest", self.refused(Backend(Response(changed)))["error"])
            parsed.assert_not_called()
        self.assertFalse((self.root / "intake/data").exists())

    def test_parent_readback_authentication_before_zip(self):
        original = intake.fetch

        def corrupt(policy, fd, state, backend, clock):
            result = original(policy, fd, state, backend, clock)
            os.pwrite(fd, b"x", 0)
            return result
        with patch.object(intake, "fetch", side_effect=corrupt), patch.object(zip_intake, "central_bound") as parsed:
            self.assertIn("authentication", self.refused()["error"])
            parsed.assert_not_called()

    def test_deadline_after_read_preserves_pending(self):
        tick = 0

        def clock():
            nonlocal tick
            tick += 1
            return 0 if tick <= 4 else 2000
        self.assertIn("deadline", self.refused(clock=clock)["error"])
        self.assertEqual((self.root / "intake/pending-read-buffer.bin").read_bytes(), self.raw[:32])

    def test_partial_http_read_failure(self):
        self.refused(Backend(Response(self.raw, error_at=2)))
        self.assertEqual((self.root / "intake/download/lean-4.35.0-linux.zip.part").read_bytes(), self.raw[:32])

    def test_close_error_cannot_replace_primary(self):
        evidence = self.refused(Backend(Response(b"", status=503, close_error=True)))
        self.assertIn("HTTP status", evidence["error"])
        self.assertIn("response_close_error", evidence)

    def test_partial_output_write_failure_retains_pending(self):
        original = fdio.write_all
        fired = False

        def fail(fd, raw):
            nonlocal fired
            if raw == b"alpha" and not fired:
                fired = True
                os.write(fd, raw[:2])
                raise OSError("injected member partial write")
            return original(fd, raw)
        with patch.object(fdio, "write_all", side_effect=fail):
            self.refused()
        self.assertEqual((self.root / "intake/data" / TOP / "A.ir").read_bytes(), b"al")
        self.assertEqual((self.root / "intake/pending-read-buffer.bin").read_bytes(), b"alpha")

    def test_pending_buffer_write_can_fail_without_masking_primary(self):
        original = fdio.write_all

        def fail(fd, raw):
            if raw == b"alpha":
                os.write(fd, raw[:2])
                raise OSError("injected repeated write failure")
            return original(fd, raw)
        with patch.object(fdio, "write_all", side_effect=fail):
            evidence = self.refused()
        self.assertIn("preservation_failure", evidence)
        self.assertEqual(evidence["error"], "injected repeated write failure")
        self.assertEqual((self.root / "intake/pending-read-buffer.bin").read_bytes(), b"al")

    def test_positive_directories_nested_four_files_deflate(self):
        entries = [(TOP + "/", b"", stat.S_IFDIR | 0o755, b""),
                   (TOP + "/lib/", b"", stat.S_IFDIR | 0o755, b"")]
        entries += [(TOP + "/lib/" + name, raw, stat.S_IFREG | 0o644, b"")
                    for name, raw in (("A.ir", b"alpha"), ("A.ir.sig", b"sig-a"),
                                      ("B.ir", b"beta"), ("B.ir.sig", b"sig-b"))]
        self.raw = archive_bytes(entries, zipfile.ZIP_DEFLATED)
        self.bind_synthetic(self.raw)
        value = self.run_one()
        self.assertEqual(len(value["members"]), 6)

    def test_partial_network_output_write_preserves_current_chunk(self):
        actual = http_fetch.write_all
        fired = False

        def fail(fd, raw):
            nonlocal fired
            if not fired:
                fired = True
                os.write(fd, raw[:3])
                raise OSError("injected download partial write")
            return actual(fd, raw)
        with patch.object(http_fetch, "write_all", side_effect=fail):
            self.refused()
        self.assertEqual((self.root / "intake/download/lean-4.35.0-linux.zip.part").read_bytes(), self.raw[:3])
        self.assertEqual((self.root / "intake/pending-read-buffer.bin").read_bytes(), self.raw[:32])

    def test_output_tree_extra_member_refuses(self):
        actual = zip_intake.check_tree

        def extra(fd, members, directories):
            target = fdio.new_file(fd, "extra")
            os.close(target)
            actual(fd, members, directories)
        with patch.object(zip_intake, "check_tree", side_effect=extra):
            self.assertIn("roster", self.refused()["error"])

    def test_output_tree_missing_member_refuses(self):
        actual = zip_intake.check_tree

        def missing(fd, members, directories):
            os.unlink(self.root / "intake/data" / TOP / "A.ir")
            actual(fd, members, directories)
        with patch.object(zip_intake, "check_tree", side_effect=missing):
            self.assertIn("roster", self.refused()["error"])

    def test_output_tree_symlink_member_refuses(self):
        actual = zip_intake.check_tree

        def linked(fd, members, directories):
            p = self.root / "intake/data" / TOP / "A.ir"
            p.unlink()
            p.symlink_to("A.ir.sig")
            actual(fd, members, directories)
        with patch.object(zip_intake, "check_tree", side_effect=linked):
            self.assertIn("association", self.refused()["error"])

    def test_missing_host_interpreter_and_escalated_authority(self):
        value = self.run_one()
        roster = json.loads((self.root / "intake/ZIP-ROSTER.json").read_bytes())
        changed = deepcopy(value)
        changed["authority"]["execute"] = True
        with self.assertRaises(ValueError):
            context_schema.prepare(self.request, changed, roster, (TOP + "/A.ir",))

    def test_trailing_zip_data(self):
        self.assertIn("trailing", self.bad_zip(self.raw + b"extra")["error"])

    def test_missing_content_length_allowed_under_exact_stream_pin(self):
        self.run_one(Backend(Response(self.raw, headers={})))

    def test_bad_archive_format(self):
        self.bad_zip(b"not a ZIP")

    def test_duplicate_names(self):
        self.bad_zip(archive_bytes([(TOP + "/a", b"x", stat.S_IFREG | 0o600, b"")] * 2))

    def test_file_prefix_collision(self):
        self.bad_zip(archive_bytes([(TOP + "/a", b"x", stat.S_IFREG | 0o600, b""),
                                   (TOP + "/a/b", b"x", stat.S_IFREG | 0o600, b"")]))

    def test_escape_member(self):
        self.bad_zip(archive_bytes([(TOP + "/../escape", b"x", stat.S_IFREG | 0o600, b"")]))

    def test_absolute_member(self):
        self.bad_zip(archive_bytes([("/absolute", b"x", stat.S_IFREG | 0o600, b"")]))

    def test_nul_name(self):
        raw = self.raw.replace(b"A.ir", b"A\x00ir")
        self.bad_zip(raw)

    def test_symlink_member(self):
        self.bad_zip(archive_bytes([(TOP + "/link", b"target", stat.S_IFLNK | 0o777, b"")]))

    def test_nonregular_member(self):
        self.bad_zip(archive_bytes([(TOP + "/fifo", b"", stat.S_IFIFO | 0o600, b"")]))

    def test_link_extra(self):
        self.bad_zip(archive_bytes([(TOP + "/a", b"x", stat.S_IFREG | 0o600, struct.pack("<HH", 0x000d, 4) + b"link")]))

    def test_encrypted_member(self):
        raw = bytearray(self.raw)
        at = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, at + 8, 1)
        self.bad_zip(bytes(raw))

    def test_unsupported_compression(self):
        raw = bytearray(self.raw)
        at = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, at + 10, 99)
        self.bad_zip(bytes(raw))

    def test_crc_error(self):
        self.assertIn("CRC", self.bad_zip(self.raw.replace(b"alpha", b"ALPHA"))["error"])

    def test_declared_size_error(self):
        raw = bytearray(self.raw)
        at = raw.index(b"PK\x01\x02")
        struct.pack_into("<L", raw, at + 24, 6)
        self.bad_zip(bytes(raw))

    def test_member_cap(self):
        self.request["extract"]["per_member_bytes_max"] = 4
        self.assertIn("size cap", self.refused()["error"])

    def test_total_expanded_cap(self):
        self.request["extract"]["total_uncompressed_bytes_max"] = 8
        self.assertIn("total expanded", self.refused()["error"])

    def test_count_cap_preparser(self):
        self.request["extract"]["member_count_max"] = 1
        with patch.object(zip_intake.zipfile, "ZipFile") as parsed:
            self.refused()
            parsed.assert_not_called()

    def test_zip64_unsupported(self):
        raw = bytearray(self.raw)
        at = raw.rindex(b"PK\x05\x06")
        struct.pack_into("<HH", raw, at + 8, 65535, 65535)
        self.assertIn("ZIP64", self.bad_zip(bytes(raw))["error"])

    def test_existing_root_or_retry_no_http(self):
        self.run_one()
        backend = Backend()
        with self.assertRaises(fdio.IntakeStop):
            self.run_one(backend)
        self.assertEqual(backend.calls, [])

    def test_failed_attempt_no_retry(self):
        self.refused(Backend(OSError("first failure")))
        backend = Backend()
        with self.assertRaises(fdio.IntakeStop):
            self.run_one(backend)
        self.assertEqual(backend.calls, [])

    def test_symlink_parent_refused(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.request["new_root"] = str(alias / "output")
        backend = Backend()
        with self.assertRaises(OSError):
            self.run_one(backend)
        self.assertEqual(backend.calls, [])

    def test_preservation_failure_preserves_primary(self):
        actual = fdio.save

        def save(fd, name, value):
            if name == "FIRST-FAILURE.json":
                raise OSError("injected preservation failure")
            return actual(fd, name, value)
        with patch.object(fdio, "save", side_effect=save):
            with self.assertRaises(fdio.IntakeStop) as error:
                self.run_one(Backend(OSError("original network error")))
        self.assertEqual(error.exception.evidence["error"], "original network error")
        self.assertIn("preservation_failure", error.exception.evidence)

    def test_new_context_schema_no_loader_closure_no_execution(self):
        manifest = self.run_one()
        roster = json.loads((self.root / "intake/ZIP-ROSTER.json").read_bytes())
        selected = (TOP + "/A.ir",)
        proposal = context_schema.prepare(self.request, manifest, roster, selected)
        self.assertIsNone(proposal.host_review_identity)
        self.assertEqual(proposal.status, "SOURCE_ONLY_NOT_RUNNABLE_NO_AUTHORITY")
        self.assertEqual(proposal.member_refs[0][1], hashlib.sha256(b"alpha").hexdigest())
        altered = deepcopy(manifest)
        altered["parent_archive"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            context_schema.prepare(self.request, altered, roster, selected)
        altered = deepcopy(manifest)
        altered["members"].pop()
        with self.assertRaises(ValueError):
            context_schema.prepare(self.request, altered, roster, selected)
        with self.assertRaises(ValueError):
            context_schema.prepare(self.request, manifest, roster, (TOP + "/missing",))

    def test_separately_bound_host_closure_documentary_only(self):
        manifest = self.run_one()
        roster = json.loads((self.root / "intake/ZIP-ROSTER.json").read_bytes())
        host = {"schema": "merlin.separate-host-closure-proposal.v1",
                "review_ref": {"path": "/review/host.json", "sha256": "1" * 64, "bytes": 5},
                "refs": [{"path": "/host/loader", "sha256": "2" * 64, "bytes": 6}],
                "interpreter": "/host/loader", "closure_scope": "EXPLICIT_FINITE_REVIEW_REQUIRED",
                "native_authorized": False}
        value = context_schema.prepare(self.request, manifest, roster, (TOP + "/A.ir",), host)
        self.assertIsNotNone(value.host_review_identity)
        host["refs"].append(deepcopy(host["refs"][0]))
        with self.assertRaises(ValueError):
            context_schema.prepare(self.request, manifest, roster, (TOP + "/A.ir",), host)


if __name__ == "__main__":
    unittest.main()
