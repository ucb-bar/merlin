"""Offline unit-test SOURCE; NOT RUN under the current execution hold.

Every ELF below is synthetic metadata without executable code. Fixture hashes
are not proposed production pins. No test launches a loader, Lean, held oracle
or failed qualification. Tests, if later authorized, only qualify Python forms.
"""

from __future__ import annotations

import hashlib
import json
import struct

import pytest

from merlin.verify.lean_graph_regions_declared_elf import UnsupportedElf, parse
from merlin.verify.lean_graph_regions_declared_loader_evidence import (
    POLICY_OBLIGATIONS,
    RETAINED_ARTIFACTS,
    evidence_request,
    inspect_observation,
)
from merlin.verify.lean_graph_regions_declared_loader_plan import (
    Pins,
    Unavailable,
    closed_environment,
    expand_paths,
    parse_json,
    read_original,
    resolve_links,
    static_plan,
    strict_ref,
)


def elf(
    interpreter=b"/original/loader\0",
    needed=(b"libowned.so\0",),
    rpath=None,
    runpath=b"$ORIGIN/../lib\0",
    soname=None,
    wide=True,
    endian="<",
    extra=(),
):
    """Independent synthetic ELF32/64 metadata; no instructions to execute."""
    base, size = 0x400000, 4096
    data = bytearray(size)
    data[:16] = b"\x7fELF" + bytes([2 if wide else 1, 1 if endian == "<" else 2, 1]) + bytes(9)
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
        strings += name
    for tag, value in ((15, rpath), (29, runpath), (14, soname)):
        if value is not None:
            entries.append((tag, len(strings)))
            strings += value
    entries += list(extra) + [(5, base + 2048), (10, len(strings)), (0, 0)]
    segment(0, 1, 0, size)
    if interpreter:
        segment(1, 3, 256, len(interpreter))
        data[256 : 256 + len(interpreter)] = interpreter
    segment(2 if interpreter else 1, 2, 1024, dynamic.size * len(entries))
    data[2048 : 2048 + len(strings)] = strings
    for index, (tag, value) in enumerate(entries):
        dynamic.pack_into(data, 1024 + dynamic.size * index, tag, value)
    return bytes(data)


@pytest.mark.parametrize("wide,endian", [(True, "<"), (True, ">"), (False, "<"), (False, ">")])
def test_independent_elf_fields(wide, endian):
    metadata = parse(elf(wide=wide, endian=endian))
    assert metadata.interpreter == "/original/loader"
    assert metadata.needed == ("libowned.so",)
    assert metadata.runpath == "$ORIGIN/../lib"
    assert metadata.bits == (64 if wide else 32)
    assert metadata.byte_order == ("little" if endian == "<" else "big")


def test_rpath_runpath_and_soname_remain_distinct():
    metadata = parse(elf(rpath=b"/original/rpath\0", soname=b"independent.so\0"))
    assert metadata.rpath == "/original/rpath"
    assert metadata.runpath == "$ORIGIN/../lib"
    assert metadata.soname == "independent.so"


@pytest.mark.parametrize("kind", ["truncated", "class", "byteorder", "extended_headers", "bad_strings", "header_size"])
def test_malformed_or_unsupported_elf_refuses(kind):
    data = bytearray(elf())
    if kind == "truncated":
        data = data[:100]
    elif kind == "class":
        data[4] = 0
    elif kind == "byteorder":
        data[5] = 0
    elif kind == "extended_headers":
        struct.pack_into("<H", data, 56, 65535)
    elif kind == "header_size":
        struct.pack_into("<H", data, 52, 0)
    else:
        struct.pack_into("<Q", data, 1024 + 8, 5000)
    with pytest.raises(UnsupportedElf):
        parse(bytes(data))


@pytest.mark.parametrize("tag", [0x6FFFFEFA, 0x6FFFFEFB, 0x6FFFFEFC, 0x7FFFFFFD, 0x7FFFFFFF])
def test_indirect_load_controls_are_not_silently_dropped(tag):
    assert parse(elf(extra=[(tag, 1)])).unsupported_load_tags == (tag,)


def test_origin_expansion_is_structural():
    assert expand_paths("$ORIGIN/../lib:${ORIGIN}/local", "/original/bin") == [
        "/original/bin/../lib",
        "/original/bin/local",
    ]


@pytest.mark.parametrize("path", ["$LIB", "$PLATFORM", "$ORIGINAL", "${ORIGIN", ":/pinned", "/pinned:", "relative"])
def test_ambiguous_or_ambient_search_refuses(path):
    with pytest.raises(Unavailable):
        expand_paths(path, "/original/bin")


def test_original_pin_conflict_never_reselects():
    pins = Pins()
    first = dict(path="/original/file", bytes=4, sha256="1" * 64)
    pins.add(first, {"source": "original-one"})
    with pytest.raises(Unavailable, match="conflicting"):
        pins.add(dict(path="/original/file", bytes=4, sha256="2" * 64), {"source": "original-two"})


@pytest.mark.parametrize(
    "fields",
    [
        {"bytes": True},
        {"path": "relative"},
        {"path": "/a/../b"},
        {"path": "//ambiguous"},
        {"path": "/nul\0"},
        {"sha256": "X" * 64},
    ],
)
def test_reference_types_and_paths_are_strict(fields):
    with pytest.raises(Unavailable):
        strict_ref(dict(path="/original/file", bytes=4, sha256="1" * 64) | fields)


@pytest.mark.parametrize(
    "extra",
    [
        {"LD_PRELOAD": "/unpinned/code.so"},
        {"LD_AUDIT": ""},
        {"PATH": "/usr/bin"},
        {"LEAN_PATH": "relative"},
        {"LEAN_SRC_PATH": "/original:"},
        {"LEAN_PATH": "/null\0"},
        {"GLIBC_TUNABLES": ""},
    ],
)
def test_launch_environment_is_closed(extra):
    with pytest.raises(Unavailable):
        closed_environment({"LANG": "C", "LC_ALL": "C"} | extra)


def pin_fixture(path, data, pins):
    """Synthetic fixture pin; not a runtime identity or failed-context refresh."""
    path.write_bytes(data)
    ref = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    pins.add(ref, {"source": "independent synthetic fixture"})
    return ref


def plan_fixture(tmp_path):
    pins = Pins()
    loader, library, binary = (tmp_path / name for name in ("loader", "libowned.so", "binary"))
    pin_fixture(loader, elf(interpreter=None, needed=(), runpath=None), pins)
    pin_fixture(library, elf(interpreter=None, needed=(), runpath=None, soname=b"libowned.so\0"), pins)
    pin_fixture(binary, elf(interpreter=str(loader).encode() + b"\0", runpath=None), pins)
    plan = static_plan(pins, str(binary), [str(tmp_path)], {"LANG": "C", "LC_ALL": "C"})
    return pins, plan


def test_complete_static_pin_graph_never_becomes_launch_assurance(tmp_path):
    _, plan = plan_fixture(tmp_path)
    assert len(plan["static_nodes"]) == 3
    assert plan["issues"] == []
    assert plan["launch_status"] == "UNAVAILABLE"
    assert not plan["target_binary_executed"]
    assert not plan["actual_loaded_closure_established"]
    assert plan["default_directory_fallback_is_not_disabled_by_proposed_argv"]


def test_first_existing_unpinned_candidate_does_not_fall_through(tmp_path):
    pins, plan = plan_fixture(tmp_path)
    earlier = tmp_path / "unbound"
    earlier.mkdir()
    (earlier / "libowned.so").write_bytes(b"not executable; deliberately unpinned")
    changed = static_plan(
        pins, plan["requested_executable"], [str(earlier), str(tmp_path)], plan["proposed_environment"]
    )
    edge = next(edge for edge in changed["resolution_edges"] if edge["kind"] == "DT_NEEDED")
    assert edge["status"] == "UNAVAILABLE"
    assert edge["resolved"] == str(earlier / "libowned.so")
    assert "never fall through" in edge["reason"]


def test_symlink_selector_is_visible_and_unqualified(tmp_path):
    pins, plan = plan_fixture(tmp_path)
    directory = tmp_path / "selector"
    directory.mkdir()
    (directory / "libowned.so").symlink_to(tmp_path / "libowned.so")
    changed = static_plan(pins, plan["requested_executable"], [str(directory)], plan["proposed_environment"])
    assert any(issue["kind"] == "UNPINNED_SYMLINK_SELECTOR" for issue in changed["issues"])
    assert changed["launch_status"] == "UNAVAILABLE"


def test_original_runtime_file_change_is_refused_without_new_pin(tmp_path):
    pins = Pins()
    path = tmp_path / "original"
    original = pin_fixture(path, b"original bytes", pins)
    path.write_bytes(b"different bytes")
    with pytest.raises(Unavailable, match="ORIGINAL pin mismatch") as captured:
        read_original(original)
    assert pins.files[str(path)] == original
    assert captured.value.details["original_reference"] == original
    assert captured.value.details["observed_digest_is_not_a_new_expected_pin"]


def test_json_duplicate_reference_field_refuses():
    with pytest.raises(Unavailable, match="duplicate JSON field"):
        parse_json('{"sha256":"original","sha256":"substitute"}')


def test_runtime_union_uses_original_documents_not_a_current_selection(tmp_path):
    fixture_pins = Pins()
    runtime = pin_fixture(tmp_path / "runtime", b"synthetic fixed runtime", fixture_pins)
    sdk = pin_fixture(tmp_path / "sdk.json", json.dumps({"files": {"runtime": runtime}}).encode(), fixture_pins)
    context = pin_fixture(tmp_path / "context.json", json.dumps({"runtime": runtime}).encode(), fixture_pins)
    provenance = dict(original_F_SDK_manifest=sdk, original_W_fixed_context=context, original_runtime_refs=[runtime])
    selected = Pins.original_union(provenance)
    assert selected.files == {runtime["path"]: runtime}
    assert {origin["source"] for origin in selected.origins[runtime["path"]]} == {
        "F SDK original",
        "W fixed original runtime",
    }


def test_conflicting_original_context_cannot_hide_behind_one_matching_reference(tmp_path):
    fixture_pins = Pins()
    runtime = pin_fixture(tmp_path / "runtime", b"synthetic fixed runtime", fixture_pins)
    sdk = pin_fixture(tmp_path / "sdk.json", json.dumps({"files": {"runtime": runtime}}).encode(), fixture_pins)
    conflict = dict(runtime, sha256="a" * 64)
    context = pin_fixture(
        tmp_path / "context.json", json.dumps({"first": runtime, "second": conflict}).encode(), fixture_pins
    )
    with pytest.raises(Unavailable, match="conflicting ORIGINAL W"):
        Pins.original_union(
            dict(original_F_SDK_manifest=sdk, original_W_fixed_context=context, original_runtime_refs=[runtime])
        )


def test_symlink_cycle_refuses(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)
    with pytest.raises(Unavailable, match="symlink cycle"):
        resolve_links(str(first))


def test_non_directory_parent_refuses(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"plain file")
    with pytest.raises(Unavailable, match="non-directory"):
        resolve_links(str(path / ".." / "other"))


def test_request_does_not_invent_external_observation(tmp_path):
    _, plan = plan_fixture(tmp_path)
    request = evidence_request(plan)
    assert request["status"] == "UNAVAILABLE"
    assert not request["execution_authorized"]
    assert not request["actual_loaded_closure_established"]


def test_complete_external_form_is_not_claimed_to_prove_runtime(tmp_path):
    pins, plan = plan_fixture(tmp_path)
    documentary = dict(path="/unread/external-evidence", bytes=1, sha256="e" * 64)
    observation = dict(
        schema="merlin.graph-regions.pure-launch-observation.v1",
        launch=dict(
            mode="loader-list",
            argv=plan["proposed_diagnostic_argv"],
            environment=plan["proposed_environment"],
            cwd=str(tmp_path),
        ),
        loaded_files=list(pins.files.values()),
        imported_files=[],
        retained_artifacts={key: documentary for key in RETAINED_ARTIFACTS},
        policy_evidence={key: documentary for key in POLICY_OBLIGATIONS},
    )
    result = inspect_observation(plan, pins, observation)
    assert result["structural_form_status"].startswith("FORM_ONLY_MATCHES")
    assert result["status"] == "UNAVAILABLE"
    assert not result["artifact_contents_and_observer_completeness_verified"]
    substituted = json.loads(json.dumps(observation))
    substituted["loaded_files"][0]["sha256"] = "f" * 64
    assert inspect_observation(plan, pins, substituted)["structural_form_status"] == "REFUSED"
