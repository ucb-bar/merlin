"""Probe selection is derived from data alone: a synthetic second target needs no change to the selector.

The target, its facts, timing profile, timing space and probe backend below are invented. Nothing in
``merlin_experiments.timing_probes`` knows them; if selection works here, the selector is not
specialized to any real accelerator.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml
from merlin_experiments.timing_probes import backends, coverage, domain, selection, space
from merlin_experiments.timing_probes.decode import Command, decode

BACKEND = """
from merlin_experiments.timing_probes.decode import Command
from merlin_experiments.timing_probes.selection import Realized, Unrealizable

class Backend:
    SHAPES = ("small", "wide", "tall")
    PLANS = {"small": (1, 2, 4, 4, 2), "wide": (2, 3, 2, 3, 1), "tall": (1, 1, 3, 2, 3)}

    def __init__(self, space):
        self.space = space

    def _stream(self, mode, sets, mults, rows, cols, moves):
        s = self.space
        out = [Command("SET", s.pack("SetRs1", {"kind": 0, "mode": mode}), 0) for _ in range(sets)]
        out += [Command("PULL", None, s.pack("MoveRs2", {"rows": rows, "cols": cols})) for _ in range(moves)]
        out += [Command("MULT", s.pack("MultRs1", {"rows": rows}), 0) for _ in range(mults)]
        out += [Command("PUSH", None, s.pack("MoveRs2", {"rows": rows, "cols": 1}))]
        return out

    def realize(self, request, space):
        if request["shape"] == "regime":
            macs = request["regime"].get("macs", 1)
            rows = 4 if macs >= 16 else 1
            mults = max(1, -(-macs // (rows * 4)))
            return Realized(self._stream(0, 1, mults, rows, 4, 1), float(mults), f"regime x{mults}")
        if not set(request["instructions"]) <= {"SET", "PULL", "PUSH", "MULT"}:
            raise Unrealizable("no stream issues it")
        mode = (request.get("state") or {}).get("MODE.mode", 0)
        stream = self._stream(mode, *self.PLANS[request["shape"]])
        return Realized(stream, float(len(stream)), request["shape"])

def bind(*, space, facts, contract):
    return Backend(space)
"""


def _facts(*, fast=True, mult=True) -> dict:
    names = {"0": "SET", "1": "PULL", "2": "PUSH"}
    if mult:
        names["3"] = "MULT"
    return {
        "facts": {
            "target": "toyacc",
            "arrays": [{"name": "grid", "rows": 4, "cols": 4}],
            "datapaths": [{"name": "input", "dtype": "i8"}],
            "interfaces": [
                {
                    "name": "funct_decode_table",
                    "legal_funct": [int(k) for k in names],
                    "names": names,
                    "custom_opcode": 11,
                    "funct3": 3,
                },
                {
                    "name": "register_bundle_layouts",
                    "bundles": {
                        "SetRs1": {
                            "width": 64,
                            "fields": {"kind": {"offset": 0, "width": 1}, "mode": {"offset": 1, "width": 1}},
                        },
                        "MoveRs2": {
                            "width": 64,
                            "fields": {"rows": {"offset": 0, "width": 8}, "cols": {"offset": 8, "width": 8}},
                        },
                        "MultRs1": {"width": 64, "fields": {"rows": {"offset": 0, "width": 8}}},
                    },
                },
                {"name": "elaborated_rtl_features", "features": {"fast": True} if fast else {}},
            ],
        }
    }


PROFILE = {
    "prices": {
        "setup": {"lat": None},
        "xfer": {"lat": None, "per_byte": None, "per_mac": None},
        "mul_slow": {"lat": None, "per_mac": None},
        "mul_fast": {"lat": None, "per_mac": None},
        "mystery": {"lat": None},
        "bus": {"bytes_per_cycle": None},
    },
    "resources": {"engine": {"capacity": None}},
}

SPACE = {
    "schema": space.SCHEMA,
    "target": "toyacc",
    "backend": "backend.py",
    "profile": {
        "prices": ["prices"],
        "parameter_quantities": {"lat": "count", "per_byte": "bytes", "per_mac": "macs"},
        "rate_groups": {"bus": {"bytes_per_cycle": "read_bytes"}},
        "resources": ["resources"],
    },
    "decode": {
        "config": {
            "instruction": "SET",
            "selector": {"operand": "rs1", "bundle": "SetRs1", "field": "kind"},
            "states": {"MODE": {"code": 0, "rs1": "SetRs1"}},
        },
        "operands": {"PULL": {"rs2": "MoveRs2"}, "PUSH": {"rs2": "MoveRs2"}, "MULT": {"rs1": "MultRs1"}},
    },
    "classes": [
        {"name": "setup", "instructions": ["SET"]},
        {
            "name": "xfer",
            "instructions": ["PULL", "PUSH"],
            "quantities": {"bytes": ["rs2.rows", "rs2.cols", "fact.datapaths.input.bytes"]},
            "rates": {"read_bytes": "bytes"},
        },
        {
            "name": "mul_slow",
            "instructions": ["MULT"],
            "state": {"MODE.mode": 0},
            "quantities": {"macs": ["rs1.rows", "fact.arrays.grid.cols"]},
        },
        {
            "name": "mul_fast",
            "instructions": ["MULT"],
            "state": {"MODE.mode": 1},
            "requires_features": ["fast"],
            "quantities": {"macs": ["rs1.rows", "fact.arrays.grid.cols"]},
        },
        {"name": "mystery", "undecodable": "the provider's own decode is not modelled here"},
    ],
}


def _space(tmp_path: Path, facts: dict | None = None) -> space.TimingSpace:
    (tmp_path / "backend.py").write_text(BACKEND)
    (tmp_path / "timing_space.yaml").write_text(yaml.safe_dump(SPACE))
    return space.load_space(tmp_path / "timing_space.yaml", facts=facts or _facts(), profile=PROFILE, target="toyacc")


def _archive(path: Path, streams: dict[str, list[Command]]) -> Path:
    """A label archive in the probe-archive layout: one passing, measured row per stream."""
    rows, members = [], {}
    for index, (name, stream) in enumerate(streams.items()):
        member = f"{name}/probe_commands.json"
        members[member] = json.dumps({"commands": [{"name": c.name, "rs1": c.rs1, "rs2": c.rs2} for c in stream]})
        rows.append(
            {
                "result": f"{name}/probe_result.json",
                "engine": "gsim",
                "status": "pass",
                "cycles": 100 + index,
                "cycles_measured": True,
                "console_cycle_verified": True,
                "elf_sha256": f"{index:064x}",
                "artifacts": {"probe_commands": {"path": member}},
            }
        )
    with tarfile.open(path, "w") as tar:
        for name, text in {
            "index.json": json.dumps({"schema": "synthetic", "records": rows}),
            **{f"run/{k}": v for k, v in members.items()},
        }.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _existing(space_: space.TimingSpace) -> dict[str, list[Command]]:
    """Labels a compiler produced: only the slow multiply mode, always one setup, one transfer shape."""
    out = {}
    for n in range(1, 15):
        out[f"cap{n}"] = [
            Command("SET", space_.pack("SetRs1", {"kind": 0, "mode": 0}), 0),
            Command("PULL", None, space_.pack("MoveRs2", {"rows": n, "cols": 4})),
            *[Command("MULT", space_.pack("MultRs1", {"rows": (n % 4) + 1}), 0) for _ in range(n)],
            Command("PUSH", None, space_.pack("MoveRs2", {"rows": n, "cols": 1})),
        ]
    return out


DOMAIN = {"schema": domain.RANGES, "ranges": {"macs": [4, 96]}}


def _select(tmp_path, labels):
    timing_space = _space(tmp_path)
    backend = backends.load_backend(timing_space, facts=_facts(), contract={})
    found = []
    for archive in labels:
        found += coverage.read_archive(archive, timing_space)[0]
    document = selection.select(timing_space, found, domain.from_document(DOMAIN, source="t", sha256="0" * 64), backend)
    return timing_space, backend, document


def test_space_statuses_come_from_the_facts_and_profile(tmp_path):
    timing_space = _space(tmp_path)
    status = {c.name: c.status for c in timing_space.classes}
    assert status == {
        "setup": "required",
        "xfer": "required",
        "mul_slow": "required",
        "mul_fast": "required",
        "mystery": "undecodable",
    }
    assert _space(tmp_path, _facts(fast=False)).by_name()["mul_fast"].status == "support_unknown"
    assert _space(tmp_path, _facts(mult=False)).by_name()["mul_slow"].status == "undecodable"
    explained = {row["subject"]: row["status"] for row in space.parameter_explanations(timing_space)}
    assert explained["rate.read_bytes"] == "structurally_dependent"  # equals xfer.bytes by construction
    assert explained["xfer.per_mac"] == "not_applicable"
    assert explained["resource.engine"] == "requires_event_model"


def test_decoder_tracks_selected_state_and_refuses_unknown_state(tmp_path):
    timing_space = _space(tmp_path)
    fast = Command("SET", timing_space.pack("SetRs1", {"kind": 0, "mode": 1}), 0)
    mult = Command("MULT", timing_space.pack("MultRs1", {"rows": 3}), 0)
    decoded = decode(timing_space, [fast, mult, mult])
    assert decoded.counts["mul_fast"] == 2 and decoded.quantities[("mul_fast", "macs")] == 24
    unknown = decode(timing_space, [Command("SET", None, 0), mult])  # symbolic config: state UNKNOWN
    assert not unknown.counts.get("mul_fast") and unknown.unclassified["MULT (state unknown)"] == 1


def test_selection_covers_every_gap_and_a_rerun_with_the_new_labels_has_none(tmp_path):
    timing_space = _space(tmp_path)
    before = _archive(tmp_path / "capsules.tar", _existing(timing_space))
    timing_space, backend, document = _select(tmp_path, [before])
    unmet_before = {r["id"] for r in document["requirements"] if not r["met_by_existing"]}
    assert {"presence:mul_fast", "variant:xfer/PULL"} - unmet_before == {"variant:xfer/PULL"}
    assert "presence:mul_fast" in unmet_before and "identifiable:setup.count" in unmet_before
    assert document["gaps"] == 0 and document["design_check"]["identifiable"] is True
    covered = {c["requirement"] for p in document["probes"] for c in p["covers"]}
    assert unmet_before <= covered
    assert all(p["covers"] for p in document["probes"])  # every chosen probe explains itself
    probes = {p["id"]: backend.realize(p["request"], timing_space).commands for p in document["probes"]}
    after = _archive(tmp_path / "probes.tar", probes)
    _, _, recheck = _select(tmp_path, [before, after])
    assert recheck["gaps"] == 0 and recheck["probes"] == []


def test_unrealizable_or_unknown_support_is_reported_not_guessed(tmp_path):
    timing_space = _space(tmp_path, _facts(fast=False))
    backend = backends.load_backend(timing_space, facts=_facts(fast=False), contract={})
    labels, _ = coverage.read_archive(_archive(tmp_path / "c.tar", _existing(timing_space)), timing_space)
    document = selection.select(
        timing_space, labels, domain.from_document(DOMAIN, source="t", sha256="0" * 64), backend
    )
    assert not any(p["request"].get("class") == "mul_fast" for p in document["probes"])
    reasons = {row["subject"]: row for row in document["unidentifiable"]}
    assert reasons["mul_fast"]["status"] == "support_unknown"


def test_probe_labels_round_trip_through_the_archive_into_coverage(tmp_path):
    """A written probe archive is read back as usable labels with the stream it measured."""
    from merlin_experiments.timing_probes import runner

    timing_space = _space(tmp_path)
    stream = _existing(timing_space)["cap3"]
    cache = runner.private_dir(tmp_path / "cache")
    results = []
    for index, (status, cycles) in enumerate((("pass", 321), ("fail", None))):
        key = f"{index:064x}"
        entry = runner.private_dir(cache / key[:24])
        (entry / "probe.elf").write_bytes(b"\x7fELF" + bytes([index]))
        (entry / "probe.c").write_text("/* marker */")
        (entry / "probe_commands.json").write_text(
            json.dumps({"commands": [{"name": c.name, "rs1": c.rs1, "rs2": c.rs2} for c in stream]})
        )
        (entry / "probe_result.json").write_text("{}")
        runs = []
        for replicate in range(2):
            (entry / f"rtl_console_r{replicate}.log").write_text(f"METRIC cycles {cycles}\nDONE\n" if cycles else "")
            runs.append(
                {
                    "replicate": replicate,
                    "status": status,
                    "cycles": cycles,
                    "console": f"rtl_console_r{replicate}.log",
                    "sim_wall_s": 1.0,
                }
            )
        results.append(
            {
                "key": key,
                "probe": f"p{index}",
                "request": {"class": "xfer"},
                "source_sha256": "0" * 64,
                "measurement": {
                    "elf_sha256": runner._sha_file(entry / "probe.elf"),
                    "runs": runs,
                    "deterministic": cycles is not None,
                    "status": status,
                    "build_s": 0.1,
                },
            }
        )

    class Timing:
        selection_sha256 = "1" * 64

        def record(self):
            return {"schema": "merlin.kernel_call_timing.v1", "selection_sha256": self.selection_sha256}

    identity = {
        "identity_sha256": "2" * 64,
        "emulator_sha256": "3" * 64,
        "firrtl_sha256": "4" * 64,
        "toolchain_sha256": "5" * 64,
        "engine_citation": {"path": "/engine"},
    }
    written = runner.write_archive(
        tmp_path,
        cache,
        results,
        identity=identity,
        timing=Timing(),
        space=timing_space,
        backend_identity={},
        selection={"probes": []},
    )
    assert (written["records"], written["complete"], written["timing_ready"]) == (4, 2, False)
    labels, census = coverage.read_archive(written["path"], timing_space)
    assert census["not_a_fresh_passing_gsim_label"] == 2 and census["replicate_executions"] == 1
    assert len(labels) == 1 and labels[0].cycles == 321 and labels[0].source == "probe"
    assert labels[0].decoded.counts == decode(timing_space, stream).counts


def test_the_backend_must_live_beside_its_timing_space(tmp_path):
    timing_space = _space(tmp_path)
    timing_space.document["backend"] = "../elsewhere.py"
    with pytest.raises(space.SpaceError, match="own directory"):
        backends.load_backend(timing_space, facts=_facts(), contract={})


def test_cli_probe_limit_caps_real_selection_and_keeps_uncovered_requirements(tmp_path, monkeypatch):
    from merlin_experiments.timing_probes import cli

    timing_space = _space(tmp_path)
    backend = backends.load_backend(timing_space, facts=_facts(), contract={})
    monkeypatch.setattr(cli, "_load", lambda _: (timing_space, backend, {}))
    source = tmp_path / "domain.json"
    source.write_text(json.dumps(DOMAIN))
    output = tmp_path / "selection.json"
    assert (
        cli.main(
            [
                "select",
                "--target-hardware",
                "toyacc",
                "--profile",
                "unused.json",
                "--domain",
                str(source),
                "--out",
                str(output),
                "--max-probes",
                "1",
            ]
        )
        == 0
    )
    selected = json.loads(output.read_text())
    assert len(selected["probes"]) == 1
    assert selected["gaps"] > 0
    assert selected["unidentifiable"]


def test_cli_refuses_an_imported_selection_above_limit_before_measurement(tmp_path, monkeypatch):
    from merlin_experiments.timing_probes import cli

    monkeypatch.setattr(cli, "_load", lambda _: (None, None, {}))
    monkeypatch.setattr(cli.runner, "run", lambda *a, **kw: pytest.fail("measurement started"))
    source = tmp_path / "selection.json"
    source.write_text(json.dumps({"probes": [{"id": "one"}, {"id": "two"}]}))
    with pytest.raises(SystemExit, match="exceeds --max-probes"):
        cli.main(
            [
                "run",
                "--target-hardware",
                "toyacc",
                "--profile",
                "unused.json",
                "--selection",
                str(source),
                "--out",
                str(tmp_path / "run"),
                "--max-probes",
                "1",
            ]
        )


@pytest.mark.parametrize("count", ["0", "-1"])
def test_cli_probe_limit_must_be_positive(count):
    from merlin_experiments.timing_probes import cli

    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            [
                "select",
                "--target-hardware",
                "toyacc",
                "--profile",
                "unused.json",
                "--domain",
                "domain.json",
                "--out",
                "selection.json",
                "--max-probes",
                count,
            ]
        )


def test_cli_default_probe_limit_preserves_existing_selection_budget():
    from merlin_experiments.timing_probes import cli

    args = cli.parser().parse_args(
        [
            "select",
            "--target-hardware",
            "toyacc",
            "--profile",
            "unused.json",
            "--domain",
            "domain.json",
            "--out",
            "selection.json",
        ]
    )
    assert args.max_probes == 64
