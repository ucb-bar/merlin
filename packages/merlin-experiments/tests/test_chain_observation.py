"""Compact manager observations derive counts without granting qualification."""

import copy
import hashlib
import struct
import subprocess
import sys

import pytest
from merlin_experiments.post_freeze import chain_observation as C


def model():
    # Deliberately different mapping order: program indices follow the roster.
    stages = {
        "second": {"steps": 1, "outputs": [{"dtype": "i8", "shape": [2]}]},
        "first": {"steps": 2, "outputs": [{"dtype": "f32", "shape": [2]}, {"dtype": "i64", "shape": []}]},
    }
    roster = [
        {"stage": stage, "step": step, "index": i, **output}
        for stage in ("first", "second")
        for step in range(stages[stage]["steps"])
        for i, output in enumerate(stages[stage]["outputs"])
    ]
    return {"stages": stages, "output_roster": roster, "chain_steps": 3, "declared_output_arrays": 5}


def double(value):
    return struct.pack(">d", value).hex()


def console(selected, *, failed=False, runtime=False):
    rows, stages = ["ordinary boot diagnostics"], []
    mismatches = 0
    for output in selected["output_roster"]:
        stage = output["stage"]
        if stage not in stages:
            stages.append(stage)
        program = stages.index(stage)
        if output["index"] == 0:
            errors = 7 if runtime and len(rows) == 1 else 0
            rows.append(f"CHAIN_STEP v1 {program} {output['step']} 123 {errors}")
        elements = 1
        for dim in output["shape"]:
            elements *= dim
        fail = int(failed and len(rows) == 2)
        mismatches += fail
        finite = int(not fail)
        suffix = " 0 000000007fc00000 000000003f800000" if fail else ""
        rows.append(
            f"CHAIN_ARRAY v1 {program} {output['step']} {output['index']} {elements} "
            f"{fail} {finite} {double(0.125)} {double(0.25)} "
            f"0000000000000001 0000000000000002 {fail}{suffix}"
        )
    rows.extend([f"CHAIN_END v1 {selected['chain_steps']} {len(selected['output_roster'])} {mismatches}", "DONE"])
    return ("\n".join(rows) + "\n").encode()


def test_exact_observed_counts_roster_hash_and_no_qualification():
    selected = model()
    raw = console(selected)
    report = C.observe_compact(selected, raw)
    assert report["schema"] == C.SCHEMA
    assert report["status"] == "observed_complete"
    assert report["completed_chain_steps"] == 3 and report["output_array_count"] == 5
    assert [(x["stage"], x["program"], x["step"]) for x in report["step_cycles"]] == [
        ("first", 0, 0),
        ("first", 0, 1),
        ("second", 1, 0),
    ]
    assert report["uart"] == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert all(
        report[k] is False
        for k in (
            "execution_origin_verified",
            "numerical_qualification",
            "placement_qualified",
            "scientific_qualification",
        )
    )
    # Unequal digests and nonzero error can be tolerance-valid; no reference is read.
    assert report["comparisons"][0]["actual_fnv1a64"] != report["comparisons"][0]["reference_fnv1a64"]
    assert report["comparisons"][0]["max_abs_error"] == 0.125


@pytest.mark.parametrize("failed,runtime", [(True, False), (False, True), (True, True)])
def test_well_formed_failed_observation_is_preserved(failed, runtime):
    selected = model()
    report = C.observe_compact(selected, console(selected, failed=failed, runtime=runtime))
    assert report["status"] == "observed_failure"
    assert report["comparison_status"] == ("fail" if failed else "pass")
    assert report["runtime_status"] == ("fail" if runtime else "pass")
    assert report["completed_chain_steps"] == 3 and report["output_array_count"] == 5
    if failed:
        assert report["mismatched_elements"] == 1
        assert report["comparisons"][0]["finite"] is False
        assert report["comparisons"][0]["max_abs_error"] is None


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "reorder",
        "extra",
        "truncated",
        "end",
        "done",
        "mixed",
        "nonascii",
        "leading",
        "step_program",
        "step_count",
        "error",
    ],
)
def test_malformed_or_incomplete_transports_refuse(change):
    selected = model()
    lines = console(selected).splitlines()
    if change == "missing":
        lines.pop(2)
    elif change == "duplicate":
        lines.insert(2, lines[2])
    elif change == "reorder":
        lines[2], lines[3] = lines[3], lines[2]
    elif change == "extra":
        lines.append(b"CHAIN_UNKNOWN v1")
    elif change == "truncated":
        lines = lines[:4]
    elif change == "end":
        lines[-2] = b"CHAIN_END v1 3 5 1"
    elif change == "done":
        lines[-1] = b"DONE extra"
    elif change == "mixed":
        lines.insert(1, b"OUT_BIN_BEGIN v1")
    elif change == "nonascii":
        lines.insert(0, b"\xff")
    elif change == "leading":
        lines.insert(0, b" CHAIN_STEP v1 0 0 1 0")
    elif change == "step_program":
        lines[1] = b"CHAIN_STEP v1 1 0 123 0"
    elif change == "step_count":
        lines[1] = b"CHAIN_STEP v1 0 0 123"
    else:
        lines.insert(0, b"CHAIN_ERROR v1 bad_ABI")
    with pytest.raises(ValueError):
        C.observe_compact(selected, b"\n".join(lines) + b"\n")


@pytest.mark.parametrize(
    "field,value",
    [
        (2, "00"),
        (3, "-1"),
        (4, "1"),
        (5, "0"),
        (6, "3"),
        (7, "2"),
        (7, "0"),
        (8, "7ff8000000000000"),
        (9, "bff0000000000000"),
        (10, "ABCDEF0000000000"),
        (11, "0"),
        (12, "1"),
    ],
)
def test_array_numeric_grammar_and_summary_refuse(field, value):
    selected = model()
    lines = console(selected).splitlines()
    row = lines[2].decode().split()
    row[field] = value
    lines[2] = " ".join(row).encode()
    with pytest.raises(ValueError):
        C.observe_compact(selected, b"\n".join(lines))


@pytest.mark.parametrize("dtype,width", list(C._WIDTHS.items()))
def test_sample_widths_and_finite_failure_grammar(dtype, width):
    selected = {
        "stages": {"stage": {"steps": 1, "outputs": [{"dtype": dtype, "shape": [2]}]}},
        "chain_steps": 1,
        "output_roster": [{"stage": "stage", "step": 0, "index": 0, "dtype": dtype, "shape": [2]}],
    }
    bits = (1 << (8 * width)) - 1
    raw = (
        f"CHAIN_STEP v1 0 0 0 0\nCHAIN_ARRAY v1 0 0 0 2 1 1 {double(1)} {double(1)} "
        f"0000000000000000 0000000000000001 1 0 {bits:016x} 0000000000000000\n"
        "CHAIN_END v1 1 1 1\nDONE\n"
    ).encode()
    assert C.observe_compact(selected, raw)["status"] == "observed_failure"
    if width < 8:
        with pytest.raises(ValueError, match="storage width"):
            C.observe_compact(selected, raw.replace(f"{bits:016x}".encode(), f"{bits + 1:016x}".encode()))


def test_duplicate_or_out_of_bounds_samples_refuse():
    selected = model()
    raw = console(selected, failed=True)
    with pytest.raises(ValueError, match="sample positions"):
        C.observe_compact(selected, raw.replace(b"1 0 000000007fc", b"1 2 000000007fc"))
    # A standalone two-failure sample with repeated positions.
    row = console(selected).splitlines()
    row[2] = (
        "CHAIN_ARRAY v1 0 0 0 2 2 1 "
        + double(1)
        + " "
        + double(2)
        + " 0000000000000000 0000000000000000 2"
        + " 0 0000000000000001 0000000000000000" * 2
    ).encode()
    with pytest.raises(ValueError, match="sample positions"):
        C.observe_compact(selected, b"\n".join(row))


@pytest.mark.parametrize(
    "change",
    [
        "missing_stage",
        "empty_outputs",
        "shape",
        "dtype",
        "bool_step",
        "reorder",
        "arrays",
        "steps",
        "huge",
        "scalar_shape",
    ],
)
def test_ambiguous_or_changed_declaration_refuses(change):
    selected = model()
    raw = console(selected)
    if change == "missing_stage":
        selected["stages"]["omitted"] = {"steps": 1, "outputs": []}
    elif change == "empty_outputs":
        selected["stages"]["first"]["outputs"] = []
    elif change == "shape":
        selected["stages"]["first"]["outputs"][0]["shape"] = [True]
    elif change == "dtype":
        selected["stages"]["first"]["outputs"][0]["dtype"] = "unknown"
    elif change == "bool_step":
        selected["output_roster"][0]["step"] = False
    elif change == "reorder":
        selected["output_roster"][1], selected["output_roster"][2] = (
            selected["output_roster"][2],
            selected["output_roster"][1],
        )
    elif change == "arrays":
        selected["declared_output_arrays"] = True
    elif change == "steps":
        selected["chain_steps"] = 2
    elif change == "huge":
        selected["stages"]["first"]["steps"] = 1 << 63
    else:
        selected["stages"]["first"]["outputs"][1]["shape"] = [0]
    with pytest.raises(ValueError):
        C.observe_compact(selected, raw)


def test_declarations_and_raw_bytes_remain_unchanged():
    selected = model()
    before = copy.deepcopy(selected)
    raw = console(selected)
    C.observe_compact(selected, raw)
    assert selected == before and raw == console(selected)


def test_observer_imports_with_only_standard_library():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import importlib.util; "
            f"s=importlib.util.spec_from_file_location('observer',{C.__file__!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            f"assert m.SCHEMA == {C.SCHEMA!r}",
        ],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("failed,runtime", [(False, False), (True, False), (False, True)])
def test_observed_grammar_matches_existing_reference_decoder(tmp_path, failed, runtime):
    from merlin.model_check import chain_image_readback as R

    selected = model()
    programs = [
        {
            "name": name,
            "index": index,
            "steps": selected["stages"][name]["steps"],
            "outputs": [(o["shape"], o["dtype"]) for o in selected["stages"][name]["outputs"]],
        }
        for index, name in enumerate(("first", "second"))
    ]
    schedule = [{"program": p["index"], "step": i} for p in programs for i in range(p["steps"])]
    data, offsets, digests = bytearray(), [], []
    for step in schedule:
        positions = []
        for shape, dtype in programs[step["program"]]["outputs"]:
            count = 1
            for dim in shape:
                count *= dim
            payload = bytes(count * C._WIDTHS[dtype])
            positions.append(len(data))
            data.extend(payload)
            digests.append(f"{R._fnv(payload):016x}")
        offsets.append(positions)
    reference = tmp_path / "reference.bin"
    reference.write_bytes(data)
    image = {
        "transport": "all_element_comparison_v1",
        "plan": {"programs": programs, "schedule": schedule},
        "reference": {
            "blob": {"path": str(reference), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
            "offsets": offsets,
        },
    }
    lines = console(selected, failed=failed, runtime=runtime).splitlines()
    for i, position in enumerate(j for j, line in enumerate(lines) if line.startswith(b"CHAIN_ARRAY")):
        row = lines[position].decode().split()
        row[11] = digests[i]
        lines[position] = " ".join(row).encode()
    raw = b"\n".join(lines) + b"\n"
    observed, checked = C.observe_compact(selected, raw), R.decode(image, raw)
    assert observed["completed_chain_steps"] == checked["complete_chain_steps"]
    assert observed["output_array_count"] == checked["output_array_count"]
    assert observed["comparison_status"] == checked["numerical_status"]
    assert observed["runtime_status"] == checked["runtime_status"]
    for actual, expected in zip(observed["comparisons"], checked["comparisons"], strict=True):
        assert actual == {key: expected[key] for key in actual}
