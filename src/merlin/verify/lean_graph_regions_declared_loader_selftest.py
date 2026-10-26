"""Isolated stdlib-only offline metadata tests, not graph qualification.

Invoke with Python -I -S -B and explicit sources/output directories. Load the
unchanged adjacent owned parser/plan directly, without Merlin initializers,
pytest, fake packages, mocks, loaders, Lean, native oracles or production APIs.
All ELF data and test pins below are synthetic. No runtime pin is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import struct
import sys
import tempfile
import unittest
from pathlib import Path


def synthetic_elf(
    interpreter=None, needed=(), *, wide=True, endian="<", rpath=None, runpath=None, soname=None, extra=()
):
    """ELF ABI metadata only; the buffer contains no executable instructions."""
    base, size = 0x400000, 4096
    data = bytearray(size)
    data[:16] = b"\x7fELF" + bytes((2 if wide else 1, 1 if endian == "<" else 2, 1)) + bytes(9)
    header = struct.Struct(endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH"))
    ph = struct.Struct(endian + ("IIQQQQQQ" if wide else "IIIIIIII"))
    dynamic = struct.Struct(endian + ("qQ" if wide else "iI"))
    start = 16 + header.size
    header.pack_into(data, 16, 3, 62, 1, 0, start, 0, 0, start, ph.size, 3 if interpreter else 2, 0, 0, 0)

    def segment(index, kind, offset, extent):
        values = (
            (kind, 4, offset, base + offset, 0, extent, extent, 8)
            if wide
            else (kind, offset, base + offset, 0, extent, extent, 4, 4)
        )
        ph.pack_into(data, start + index * ph.size, *values)

    strings, entries = b"\0", []
    for name in needed:
        entries.append((1, len(strings)))
        strings += name.encode() + b"\0"
    for tag, value in ((15, rpath), (29, runpath), (14, soname)):
        if value is not None:
            entries.append((tag, len(strings)))
            strings += value.encode() + b"\0"
    entries += list(extra) + [(5, base + 2048), (10, len(strings)), (0, 0)]
    segment(0, 1, 0, size)
    if interpreter:
        text = interpreter.encode() + b"\0"
        if len(text) > 700:
            raise ValueError("synthetic interpreter exceeds its reserved extent")
        segment(1, 3, 256, len(text))
        data[256 : 256 + len(text)] = text
    segment(2 if interpreter else 1, 2, 1024, len(entries) * dynamic.size)
    data[2048 : 2048 + len(strings)] = strings
    for index, entry in enumerate(entries):
        dynamic.pack_into(data, 1024 + index * dynamic.size, *entry)
    return bytes(data)


class OfflineMetadataTests(unittest.TestCase):
    api = None
    output = None

    def setUp(self):
        # Retain every synthetic first-failure buffer under the private profile.
        self.directory = Path(tempfile.mkdtemp(prefix=self._testMethodName + "-", dir=self.output)).resolve()
        self.pins = self.api["Pins"]()

    def pin(self, path, data):
        path.write_bytes(data)
        reference = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.pins.add(reference, dict(source="synthetic fixture, not runtime provenance"))
        return reference

    def plan(self, directories=None, *, rpath=None):
        loader, library, binary = (self.directory / name for name in ("loader", "owned.so", "binary"))
        self.pin(loader, synthetic_elf())
        self.pin(library, synthetic_elf(soname="owned.so"))
        self.pin(binary, synthetic_elf(str(loader), ["owned.so"], rpath=rpath))
        return self.api["static_plan"](
            self.pins, str(binary), directories or [str(self.directory)], dict(LANG="C", LC_ALL="C")
        )

    def test_all_elf_classes_and_byte_orders(self):
        for wide in (False, True):
            for endian in ("<", ">"):
                with self.subTest(wide=wide, endian=endian):
                    result = self.api["parse"](
                        synthetic_elf(
                            "/synthetic/loader",
                            ["one.so"],
                            wide=wide,
                            endian=endian,
                            rpath="/synthetic/old",
                            runpath="$ORIGIN/new",
                            soname="self.so",
                        )
                    )
                    self.assertEqual(result.bits, 64 if wide else 32)
                    self.assertEqual(result.byte_order, "little" if endian == "<" else "big")
                    self.assertEqual(result.interpreter, "/synthetic/loader")
                    self.assertEqual(result.needed, ("one.so",))
                    self.assertEqual(
                        (result.rpath, result.runpath, result.soname), ("/synthetic/old", "$ORIGIN/new", "self.so")
                    )

    def test_malformed_headers_and_strings_refuse(self):
        original = synthetic_elf(needed=["one.so"])
        samples = {"truncated": original[:100]}
        for name, offset, fmt, value in (
            ("class", 4, "B", 0),
            ("byteorder", 5, "B", 0),
            ("extended", 56, "<H", 65535),
            ("header_size", 52, "<H", 0),
            ("needed_offset", 1032, "<Q", 5000),
        ):
            data = bytearray(original)
            struct.pack_into(fmt, data, offset, value)
            samples[name] = bytes(data)
        for name, data in samples.items():
            with self.subTest(name=name), self.assertRaises(self.api["UnsupportedElf"]):
                (self.directory / (name + ".elf-data")).write_bytes(data)
                self.api["parse"](data)

    def test_audit_config_filter_tags_are_explicit(self):
        for tag in (0x6FFFFEFA, 0x6FFFFEFB, 0x6FFFFEFC, 0x7FFFFFFD, 0x7FFFFFFF):
            with self.subTest(tag=tag):
                self.assertEqual(
                    self.api["parse"](synthetic_elf(needed=["one.so"], extra=[(tag, 1)])).unsupported_load_tags, (tag,)
                )

    def test_origin_tokens_and_ambient_search_refusals(self):
        self.assertEqual(
            self.api["expand_paths"]("$ORIGIN/../lib:${ORIGIN}/local", "/synthetic/bin"),
            ["/synthetic/bin/../lib", "/synthetic/bin/local"],
        )
        for value in ("$LIB", "$PLATFORM", "$ORIGINAL", "${ORIGIN", ":/pinned", "/pinned:", "relative"):
            with self.subTest(value=value), self.assertRaises(self.api["Unavailable"]):
                self.api["expand_paths"](value, "/synthetic/bin")

    def test_reference_types_and_duplicate_json_refuse(self):
        ref = dict(path="/synthetic/file", bytes=4, sha256="1" * 64)
        for change in (
            dict(bytes=True),
            dict(path="relative"),
            dict(path="//ambiguous"),
            dict(path="/a/../b"),
            dict(path="/nul\0"),
            dict(sha256="X" * 64),
        ):
            with self.subTest(change=change), self.assertRaises(self.api["Unavailable"]):
                self.api["strict_ref"](ref | change)
        with self.assertRaises(self.api["Unavailable"]):
            self.api["parse_json"]('{"path":"original","path":"replacement"}')

    def test_environment_is_complete_replacement(self):
        for extra in (
            dict(LD_PRELOAD="/unbound.so"),
            dict(LD_AUDIT=""),
            dict(PATH="/usr/bin"),
            dict(LEAN_PATH="relative"),
            dict(LEAN_SRC_PATH="/original:"),
            dict(GLIBC_TUNABLES=""),
        ):
            with self.subTest(extra=extra), self.assertRaises(self.api["Unavailable"]):
                self.api["closed_environment"](dict(LANG="C", LC_ALL="C") | extra)

    def test_static_graph_never_grants_launch(self):
        plan = self.plan()
        self.assertEqual(len(plan["static_nodes"]), 3)
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["launch_status"], "UNAVAILABLE")
        self.assertFalse(plan["target_binary_executed"])
        self.assertFalse(plan["actual_loaded_closure_established"])
        self.assertTrue(plan["default_directory_fallback_is_not_disabled_by_proposed_argv"])

    def test_first_unpinned_candidate_does_not_fall_through(self):
        earlier = self.directory / "unbound"
        earlier.mkdir()
        (earlier / "owned.so").write_bytes(b"unbound synthetic bytes")
        plan = self.plan([str(earlier), str(self.directory)])
        edge = next(edge for edge in plan["resolution_edges"] if edge["kind"] == "DT_NEEDED")
        self.assertEqual(edge["status"], "UNAVAILABLE")
        self.assertEqual(edge["resolved"], str(earlier / "owned.so"))
        self.assertIn("never fall through", edge["reason"])

    def test_unbound_symlink_selector_is_not_a_pin(self):
        alias = self.directory / "alias"
        alias.mkdir()
        (alias / "owned.so").symlink_to(self.directory / "owned.so")
        plan = self.plan([str(alias)])
        self.assertTrue(any(item["kind"] == "UNPINNED_SYMLINK_SELECTOR" for item in plan["issues"]))
        self.assertEqual(plan["launch_status"], "UNAVAILABLE")

    def test_missing_library_never_uses_ambient_defaults(self):
        empty = self.directory / "empty"
        empty.mkdir()
        plan = self.plan([str(empty)])
        self.assertTrue(any(item["kind"] == "UNRESOLVED_DEPENDENCY" for item in plan["issues"]))

    def test_inherited_rpath_stays_unavailable(self):
        plan = self.plan(rpath=str(self.directory))
        self.assertTrue(any("INHERITED_RPATH" in item["kind"] for item in plan["issues"]))

    def test_changed_bytes_keep_the_exact_first_mismatch(self):
        path = self.directory / "original"
        original = self.pin(path, b"fixed synthetic bytes")
        path.write_bytes(b"changed synthetic bytes")
        with self.assertRaises(self.api["Unavailable"]) as captured:
            self.api["read_original"](original)
        self.assertEqual(captured.exception.details["original_reference"], original)
        self.assertTrue(captured.exception.details["observed_digest_is_not_a_new_expected_pin"])
        self.assertEqual(self.pins.files[str(path)], original)

    def test_symlink_cycles_and_non_directory_components_refuse(self):
        first, second = self.directory / "first", self.directory / "second"
        first.symlink_to(second)
        second.symlink_to(first)
        with self.assertRaises(self.api["Unavailable"]):
            self.api["resolve_links"](str(first))
        file = self.directory / "file"
        file.write_bytes(b"plain synthetic file")
        with self.assertRaises(self.api["Unavailable"]):
            self.api["resolve_links"](str(file / ".." / "other"))

    def test_original_union_and_conflicts(self):
        runtime = self.pin(self.directory / "runtime", b"synthetic original runtime")
        sdk = self.pin(self.directory / "sdk.json", json.dumps(dict(files=dict(runtime=runtime))).encode())
        context = self.pin(self.directory / "context.json", json.dumps(dict(runtime=runtime)).encode())
        provenance = dict(
            original_F_SDK_manifest=sdk, original_W_fixed_context=context, original_runtime_refs=[runtime]
        )
        self.assertEqual(self.api["Pins"].original_union(provenance).files, {runtime["path"]: runtime})
        conflict = dict(runtime, sha256="a" * 64)
        other_context = self.pin(
            self.directory / "conflict.json", json.dumps(dict(first=runtime, second=conflict)).encode()
        )
        with self.assertRaises(self.api["Unavailable"]):
            self.api["Pins"].original_union(provenance | dict(original_W_fixed_context=other_context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    OfflineMetadataTests.output = args.output
    OfflineMetadataTests.api = runpy.run_path(
        str(args.sources / "lean_graph_regions_declared_loader_plan.py"), run_name="_declared_offline_plan_tests"
    )
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineMetadataTests)
    (args.output / "ROSTER.json").write_text(json.dumps([test.id() for test in suite], indent=2) + "\n")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    forbidden = [name for name in sys.modules if name.split(".")[0] in {"merlin", "xdsl", "z3", "torch_mlir"}]
    (args.output / "RESULT.json").write_text(
        json.dumps(
            dict(
                scope="isolated stdlib-only synthetic metadata tests; NOT graph/runtime qualification",
                tests_run=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                unexpected_package_modules=forbidden,
                inspected_target_binary_executed=False,
                runtime_expected_pins_selected=False,
                lean_or_oracle_executed=False,
                qualified_graph_checker=False,
            ),
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() and not forbidden else 1)


if __name__ == "__main__":
    main()
