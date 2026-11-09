"""Existing observations keep their counter domains, identities and unknowns."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.runtime.profile_evidence import (
    BoundFile,
    CounterDomain,
    Gate,
    RegionSource,
    attribute_pc_census,
    collect,
    interval_coverage,
    parse_console,
    write_report,
)


def bound(tmp_path, name, value):
    path = tmp_path / name
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return BoundFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def row(case="one", start=10, end=20, hart=0):
    return {
        "case_id": case,
        "region": "body",
        "hart": hart,
        "start_cycle": start,
        "end_cycle": end,
        "start_instret": 100,
        "end_instret": 106,
        "instret_delta": 6,
        "checksum": 7,
        "expected_checksum": 7,
        "pass": True,
    }


def source(tmp_path, rows=None, name="one", kind="hardware_cycles", labels=None):
    rows = rows if rows is not None else [row()]
    elf = bound(tmp_path, name + ".elf", b"selected execution bytes")
    data = bound(tmp_path, name + ".json", {"rows": rows})
    gate = bound(tmp_path, name + "-gate.json", {"pass": True})
    metadata = None
    if labels is not None:
        metadata = bound(tmp_path, name + "-labels.json", {"elf_sha256": elf.sha256, "entries": labels})
    return RegionSource(
        name,
        data,
        elf,
        Gate(gate, ("pass",), "full-output oracle plus guards", "all words"),
        CounterDomain(kind, "selected caller counter meaning"),
        len(rows),
        record_path=("rows",),
        labels=metadata,
    )


@pytest.fixture
def actual_elf(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source_file = tmp_path / "input.c"
    source_file.write_text("int callee(void){return 1;} int main(void){return callee();}\n")
    executable = tmp_path / "actual.elf"
    subprocess.run([compiler, "-O0", str(source_file), "-o", str(executable)], check=True, capture_output=True)
    return BoundFile(str(executable), hashlib.sha256(executable.read_bytes()).hexdigest())


def test_missing_attribution_and_physical_metrics_stay_unknown(tmp_path):
    report, trace = collect([source(tmp_path)])
    r = report["regions"][0]
    assert r["cycles"] == r["ticks"] == 10
    assert r["retired_instructions"] == 6
    assert r["attribution"]["fqn"] is None and r["physical_busy"] is None
    assert r["physical_memory_traffic_bytes"] is None
    assert report["sources"][0]["unattributed_regions"] == 1
    assert report["tracks"][0]["union_ticks"] == 10
    event = next(e for e in trace["traceEvents"] if e["ph"] == "X")
    assert event["ts"] == 0 and event["dur"] == 10
    assert event["args"]["coordinate_is_elapsed_time"] is False


def test_selected_execution_postgate_uses_its_original_region_array(tmp_path):
    elf = bound(tmp_path, "program.elf", b"exact selected execution bytes")
    observed = row()
    observed.update({"compared_words": 37, "input_errors": 0, "guard_errors": 0})
    postgate = bound(
        tmp_path,
        "postgate.json",
        {
            "pass": True,
            "counter_semantics": "instruction_count_not_modeled_cycles",
            "regions": [observed],
            "output_words": 37,
            "raw_output_sha256": "a" * 64,
        },
    )
    selected = RegionSource(
        name="selected-execution",
        records=postgate,
        elf=elf,
        gate=Gate(postgate, ("pass",), "independent output and guards", "complete declared outputs"),
        domain=CounterDomain("functional_instructions", "instruction_count_not_modeled_cycles"),
        expected_regions=1,
        record_path=("regions",),
    )
    report, _ = collect([selected])
    assert report["regions"][0]["raw_record"] == observed
    assert report["regions"][0]["cycles"] is None
    assert report["parts"][0]["cycles_median"] is None
    assert report["parts"][0]["inclusive_retired_instructions"] == 6


def test_nested_regions_keep_inclusive_union_and_overlap_distinct(tmp_path):
    report, _ = collect([source(tmp_path, [row("parent", 0, 20), row("child", 2, 10), row("peer", 5, 15)])])
    track = report["tracks"][0]
    assert track["inclusive_ticks"] == 38 and track["union_ticks"] == 20
    assert track["overlapping_interval_ticks"] == 13
    assert track["maximum_recorded_intervals"] == 3
    assert report["parts"][0]["inclusive_retired_instructions"] == 18
    assert "retired_instructions" not in report["parts"][0]
    assert interval_coverage([(3, 3)]) == {
        "inclusive_ticks": 0,
        "union_ticks": 0,
        "overlapping_interval_ticks": 0,
        "maximum_recorded_intervals": 0,
        "scope": "observed caller intervals; not physical engine overlap",
    }


def test_model_variant_and_route_labels_never_merge(tmp_path):
    rows = [row("a"), row("b", 30, 50)]
    labels = [
        {
            "case_id": r["case_id"],
            "region": "body",
            "hart": 0,
            "model": "example",
            "part": "projection",
            "fqn": "layers.0.proj",
            "variant": r["case_id"],
            "route": "unit-" + r["case_id"],
        }
        for r in rows
    ]
    report, _ = collect([source(tmp_path, rows, labels=labels)])
    assert len(report["parts"]) == 2
    assert report["sources"][0]["FQN_labelled_regions"] == 2
    assert {r["variant"] for r in report["parts"]} == {"a", "b"}
    assert all(r["function_self_time"] is None for r in report["regions"])


def test_functional_and_hardware_runs_are_not_composed(tmp_path):
    functional = source(tmp_path, name="functional", kind="functional_instructions")
    hardware = source(tmp_path, name="hardware")
    report, trace = collect([functional, hardware])
    assert report["regions"][0]["cycles"] is None
    assert report["regions"][1]["cycles"] == 10
    assert len(report["parts"]) == len(report["tracks"]) == 2
    assert len({e["pid"] for e in trace["traceEvents"]}) == 2
    assert report["cross_hart_epoch_alignment_established"] is False


def test_harts_have_independent_coordinate_origins(tmp_path):
    s = source(tmp_path, [row("a", 100, 120, hart=0), row("b", 700, 730, hart=1)])
    report, trace = collect([s])
    assert [t["origin"] for t in report["tracks"]] == [100, 700]
    assert [e["ts"] for e in trace["traceEvents"] if e["ph"] == "X"] == [0, 0]
    assert all(t["independent_origin"] for t in report["tracks"])


def test_explicit_frequency_maps_cycles_only(tmp_path):
    s = source(tmp_path)
    s = replace(s, domain=CounterDomain("modeled_cycles", "explicit model domain", clock_hz=2_000_000))
    _, trace = collect([s])
    e = next(e for e in trace["traceEvents"] if e["ph"] == "X")
    assert e["dur"] == 5 and e["args"]["coordinate_is_elapsed_time"]
    with pytest.raises(ValueError, match="Hz"):
        CounterDomain("functional_instructions", "instructions", clock_hz=1).validate()


@pytest.mark.parametrize(
    "mutation",
    [
        {"end_cycle": 9},
        {"start_cycle": True},
        {"end_cycle": 1 << 64},
        {"instret_delta": 7},
        {"end_instret": 99},
        {"end_instret": None},
        {"pass": False},
        {"expected_checksum": 8},
        {"hart": -1},
    ],
)
def test_invalid_observations_refuse(tmp_path, mutation):
    with pytest.raises(ValueError):
        collect([source(tmp_path, [{**row(), **mutation}])])


def test_counter_wrap_and_duplicate_or_missing_regions_refuse(tmp_path):
    s = source(tmp_path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        collect([s])
    s = source(tmp_path, [row()])
    with pytest.raises(ValueError, match="complete"):
        collect([replace(s, expected_regions=2)])
    with pytest.raises(ValueError, match="unsigned 32"):
        collect(
            [
                replace(
                    source(tmp_path, [row(end=1 << 32)]),
                    domain=CounterDomain("hardware_cycles", "32-bit selected counter", bits=32),
                )
            ]
        )


def test_changed_raw_elf_labels_and_gate_cannot_be_reused(tmp_path):
    s = source(tmp_path)
    for member in [s.records, s.elf, s.gate.file]:
        path = Path(member.path)
        old = path.read_bytes()
        path.write_bytes(old + b" ")
        with pytest.raises(ValueError, match="changed"):
            collect([s])
        path.write_bytes(old)
    bad = bound(tmp_path, "failed.json", {"pass": False})
    with pytest.raises(ValueError, match="did not pass"):
        collect([replace(s, gate=replace(s.gate, file=bad))])


def test_wrong_elf_and_unknown_label_roster_refuse(tmp_path):
    s = source(tmp_path)
    labels = bound(tmp_path, "labels.json", {"elf_sha256": "0" * 64, "entries": []})
    with pytest.raises(ValueError, match="exact ELF"):
        collect([replace(s, labels=labels)])
    labels = bound(tmp_path, "extra.json", {"elf_sha256": s.elf.sha256, "entries": [row("absent")]})
    with pytest.raises(ValueError, match="unobserved"):
        collect([replace(s, labels=labels)])


def test_console_reader_uses_selected_prefix_and_strict_json():
    text = "noise\nCUSTOM " + json.dumps(row()) + "\nOTHER {}\n"
    assert parse_console(text, prefix="CUSTOM", max_regions=1) == [row()]
    for text in ['CUSTOM {"hart":0,"hart":1}', 'CUSTOM {"x":NaN}', "CUSTOM", "CUSTOM []"]:
        with pytest.raises(ValueError):
            parse_console(text, prefix="CUSTOM", max_regions=1)
    with pytest.raises(ValueError, match="budget"):
        parse_console("CUSTOM {}\nCUSTOM {}", prefix="CUSTOM", max_regions=1)


def test_console_source_and_fresh_output(tmp_path):
    s = source(tmp_path)
    console = bound(tmp_path, "stdout", ("CUSTOM " + json.dumps(row()) + "\n").encode())
    s = replace(s, records=console, console_prefix="CUSTOM", record_path=())
    report, trace = collect([s])
    out = tmp_path / "new"
    write_report(out, report, trace)
    assert json.loads((out / "report.json").read_text())["regions"][0]["ticks"] == 10
    with pytest.raises(FileExistsError):
        write_report(out, report, trace)
    with pytest.raises(ValueError, match="mutually"):
        collect([replace(s, record_path=("rows",))])


def test_pc_census_keeps_missing_symbols_and_requires_exact_elf(tmp_path, actual_elf):
    elf = actual_elf
    census = bound(
        tmp_path,
        "census.json",
        {"elf_sha256": elf.sha256, "pc_counts": [{"pc": 16, "count": 4}, {"pc": 20, "count": 3}]},
    )
    symbolizer = bound(
        tmp_path,
        "symbols.json",
        {
            "elf_sha256": elf.sha256,
            "records": [
                {"Address": "0x10", "Symbol": [{"FunctionName": "callee", "FileName": "a.c", "Line": 8}]},
                {"Address": "0x14", "Symbol": []},
            ],
        },
    )
    result = attribute_pc_census(elf, census, symbolizer)
    assert result["total_counts"] == 7 and result["unknown_function_counts"] == 3
    assert result["functions"] == {"callee": 4, "": 3}
    json.dumps(result, allow_nan=False)
    wrong = bound(tmp_path, "wrong-symbols.json", {"elf_sha256": "0" * 64, "records": []})
    with pytest.raises(ValueError, match="exact ELF"):
        attribute_pc_census(elf, census, wrong)


def test_pc_census_refuses_duplicate_counts_or_incomplete_symbolization(tmp_path, actual_elf):
    elf = actual_elf
    counts = bound(
        tmp_path,
        "counts.json",
        {"elf_sha256": elf.sha256, "pc_counts": [{"pc": 16, "count": 1}, {"pc": 16, "count": 1}]},
    )
    symbols = bound(tmp_path, "sym.json", {"elf_sha256": elf.sha256, "records": []})
    with pytest.raises(ValueError, match="duplicate PC"):
        attribute_pc_census(elf, counts, symbols)
    counts = bound(tmp_path, "counts2.json", {"elf_sha256": elf.sha256, "pc_counts": [{"pc": 16, "count": 1}]})
    with pytest.raises(ValueError, match="coverage"):
        attribute_pc_census(elf, counts, symbols)


def test_pc_census_requires_a_structurally_valid_image(tmp_path):
    invalid = bound(tmp_path, "not.elf", b"not a linked ELF")
    data = bound(tmp_path, "empty.json", {})
    with pytest.raises(ValueError, match="ELF"):
        attribute_pc_census(invalid, data, data)
