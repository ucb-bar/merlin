"""Bounded synthetic commit/process controls; no engine or namespace executes."""

import copy
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common import invocation_record as I
from merlin.common import owned_children as O
from merlin.perf import _census_process as P
from merlin.perf import dynamic_census as C
from merlin.perf._dynamic_stream import CensusUnavailable, OnlineCensus

OPCODE = 0x0B
WORD = (1 << 25) | (11 << 20) | (10 << 15) | OPCODE
ISA = {
    "CUSTOM_OPCODE": OPCODE,
    "FUNCT_CLASS": {1: "MOVE"},
    "OPERAND_FIELDS": {
        "MOVE": {
            "unresolved": "omit",
            "fields": [
                {"name": "address", "operand": "rs1", "offset": 0, "width": 16, "as": "int"},
                {"name": "rows", "operand": "rs2", "offset": 0, "width": 8, "as": "int"},
            ],
        }
    },
}


def elf():
    names, strings = b"\0.names\0.code\0.strings\0.symbols\0", b"\0entry\0"
    code = struct.pack("<III", 0x00500513, WORD, 0x00700513)
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHIQQQIHHHHHH", header, 16, 2, 0, 1, 1024, 0, 64, 0, 64, 0, 0, 64, 5, 1)
    start = 64 + 5 * 64
    sections = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, start, len(names), 0, 0, 1, 0)]
    start += len(names)
    sections.append(struct.pack("<IIQQQQIIQQ", 8, 1, 4, 1024, start, len(code), 0, 0, 1, 0))
    start += len(code)
    sections.append(struct.pack("<IIQQQQIIQQ", 14, 3, 0, 0, start, len(strings), 0, 0, 1, 0))
    start += len(strings)
    sections.append(struct.pack("<IIQQQQIIQQ", 23, 2, 0, 0, start, 24, 3, 0, 8, 24))
    symbol = struct.pack("<IBBHQQ", 1, 2, 0, 2, 1024, len(code))
    return bytes(header) + b"".join(sections) + names + code + strings + symbol


def line(pc, word, writes="", hart=0):
    return f"core {hart}: 3 0x{pc:x} (0x{word:x}) {writes}".rstrip()


def census(**changes):
    limits = replace(C.CensusLimits(), wall_s=1.0, max_output_bytes=4096, max_line_bytes=512, **changes)
    return OnlineCensus(
        elf_bytes=elf(),
        entry_symbol="entry",
        isa=ISA,
        local_fields=((("MOVE",), "address", "rows"),),
        limits=limits,
    )


def lines():
    return [
        line(8192, 0x13, "x 11 0x3 mem 0xdead"),
        line(1024, 0x00500513, "x10 0x20 mem 0x1000"),
        line(1028, WORD, "mem 0x2000"),
        line(1028, WORD),
        line(1032, 0x00700513, "x10 0x21"),
        line(1028, WORD),
        line(9000, 0x13),
    ]


def test_online_entry_counts_keep_repetition_and_exclude_outside_work():
    for module in (C, I, P, O):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd() / "src")
    reader = census()
    for raw in lines():
        reader.line(raw)
    report = reader.finish()
    assert report["retired_instructions"] == {"whole_trace": 7, "entry": 5, "outside_entry": 2}
    assert report["host"]["retired_in_entry"] == 2
    assert report["host"]["outside_entry_callee_work"] == "UNKNOWN"
    assert report["accelerator"]["by_class"] == {"MOVE": 3}
    assert [row["count"] for row in report["accelerator"]["operand_requests"]] == [2, 1]
    assert [row["raw_address"] for row in report["local_address_activity"]["requests"]] == [32, 33]
    assert report["host_memory_requests"]["by_address"] == [{"address": 4096, "count": 1}]
    assert report["accelerator_memory_requests"]["by_address"] == [{"address": 8192, "count": 1}]


@pytest.mark.parametrize("damage", ["word", "register", "memory", "hart", "empty"])
def test_unsupported_entry_trace_never_supplies_partial_counts(damage):
    reader = census()
    broken = {
        "word": line(1024, 0x13),
        "register": line(1028, WORD),
        "memory": line(1024, 0x00500513, "mem"),
        "hart": "core invalid: 3 0x400 (0x13)",
    }
    with pytest.raises(CensusUnavailable):
        if damage == "empty":
            reader.finish()
        else:
            reader.line(broken[damage])


@pytest.mark.parametrize("limit", ["retired", "line", "state"])
def test_online_limits_refuse_without_an_unbounded_command_list(limit):
    options = {"retired": {"max_retired": 1}, "line": {}, "state": {"max_bins": 1, "max_state_bytes": 1024}}
    reader = census(**options[limit])
    with pytest.raises(CensusUnavailable):
        for raw in ["core " + "x" * 600] if limit == "line" else lines():
            reader.line(raw)


def test_harts_have_independent_register_history():
    reader = census()
    reader.line(line(8192, 0x13, "x10 0x60 x11 0x1", hart=1))
    reader.line(line(8192, 0x13, "x10 0x20 x11 0x3", hart=0))
    reader.line(line(1028, WORD, hart=1))
    reader.line(line(1028, WORD, hart=0))
    assert [row["rs1"] for row in reader.finish()["accelerator"]["operand_requests"]] == [32, 96]


def run_synthetic(tmp_path, code, *, limits=None):
    reader = census()
    limits = limits or reader.limits
    argv = [sys.executable, "-c", code]
    with I.observe(tmp_path, stage="synthetic_census_control", argv=argv, env=dict(os.environ)) as recorded:
        P.capture(argv, environment=dict(os.environ), cwd=tmp_path, limits=limits, record=recorded, census=reader)
        assert recorded.census_cleanup_complete
        recorded.complete_streams(0)
        assert I.verify(recorded.path)["stage"] == "synthetic_census_control"
    return reader, recorded


def test_actual_harmless_process_streams_are_pinned_and_reopened(tmp_path):
    payload = "\n".join(lines()) + "\n"
    reader, record = run_synthetic(tmp_path, f"import sys; sys.stderr.write({payload!r}); print('DONE')")
    assert reader.finish()["accelerator"]["commands_in_entry"] == 3
    (record.directory / "stderr.bin").write_bytes(b"changed raw stream")
    with pytest.raises(ValueError, match="changed"):
        I.verify(record.path)


@pytest.mark.parametrize("kind", ["wall", "output", "truncated", "failed"])
def test_actual_process_limits_or_failure_reap_before_unknown(tmp_path, kind):
    reader = census()
    cases = {
        "wall": "import time; time.sleep(10)",
        "output": "import sys; sys.stderr.write('x'*10000); print('DONE')",
        "truncated": f"import sys; sys.stderr.write({line(1024, 0x00500513)!r}); print('DONE')",
        "failed": "raise SystemExit(2)",
    }
    argv = [sys.executable, "-c", cases[kind]]
    with I.observe(tmp_path, stage="synthetic_limit_control", argv=argv, env=dict(os.environ)) as record:
        with pytest.raises(CensusUnavailable):
            P.capture(
                argv, environment=dict(os.environ), cwd=tmp_path, limits=reader.limits, record=record, census=reader
            )
        assert record.census_cleanup_complete
    with pytest.raises(ValueError, match="did not complete"):
        I.verify(record.path)


def test_escaped_synthetic_descendant_is_reaped_after_leader_exit(tmp_path):
    pid_path = tmp_path / "descendant.pid"
    code = (
        "import os,time,pathlib,sys; pid=os.fork(); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(pid)) if pid else None; "
        "os.setsid() if not pid else None; time.sleep(10) if not pid else None; "
        f"sys.stderr.write({(chr(10).join(lines()) + chr(10))!r}) if pid else None; "
        "print('DONE') if pid else None"
    )
    run_synthetic(tmp_path, code)
    assert not Path("/proc", pid_path.read_text()).exists()


def test_no_live_original_namespace_never_executes_or_exposes_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "capture", lambda *args, **kwargs: pytest.fail("missing owner launched a process"))
    result = C.profile_elf(tmp_path / "private.elf", target="synthetic", directory=tmp_path, limits=C.CensusLimits())
    assert result["status"] == "UNKNOWN" and result["observations"] is None
    assert all(value is None for value in result["candidate_features"]["features"].values())
    assert str(tmp_path) not in str(result)


@pytest.fixture
def selected_files(tmp_path, monkeypatch):
    """File selection spies supply no positive namespace/engine authority."""
    from merlin_experiments.phase2 import campaign as PC

    original, copied, tool, wrapper, facts, owner_source = (
        tmp_path / name for name in ("original.elf", "copied.elf", "spike", "bwrap", "facts.json", "host-owner.py")
    )
    extension = tmp_path / "libsynthetic.so"
    for path, data in (
        (original, elf()),
        (copied, elf()),
        (tool, b"never loaded tool"),
        (wrapper, b"never loaded wrapper"),
        (extension, b"never loaded extension"),
        (facts, b'{"facts":{}}'),
        (owner_source, b"original host wrapper source"),
    ):
        path.write_bytes(data)
    recipe = SimpleNamespace(require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="entry"))

    def identity(simulator):
        return {
            "binary": C.F.digest(tool),
            "extensions": {extension.name: C.F.digest(extension)},
            "digest": "selected-model",
        }

    backend = SimpleNamespace(
        __file__=C.chipyard_rocc.__file__,
        rocc_semantics=C.semantics,
        harness_build_recipe=lambda: recipe,
        simulator_identity=identity,
    )
    monkeypatch.setattr(C, "get_backend", lambda target: backend)
    monkeypatch.setattr(
        C.ET,
        "trace_invocation",
        lambda target, path: (
            [str(tool), "--extlib=" + str(extension), "--extension=synthetic", "-l", "--log-commits", str(path)],
            {"LD_LIBRARY_PATH": str(tmp_path), "PRIVATE_TOKEN": "withheld"},
        ),
    )
    declared = {"sha256": C.F.digest(extension)}
    monkeypatch.setattr(C.spike_extension, "declared_extension", lambda target: declared)
    monkeypatch.setattr(C.RF, "find_facts", lambda target: facts)
    monkeypatch.setattr(C.RF, "load_facts", lambda target, *, regenerate: {"facts": {"selected": True}})
    monkeypatch.setattr(C.semantics, "isa_constants", lambda target: copy.deepcopy(ISA))
    monkeypatch.setattr(C.target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(contract={}))
    monkeypatch.setattr(C.G, "protocol_for", lambda target: SimpleNamespace(local_addresses=()))
    owner = PC.PackageSandboxPolicy((str(wrapper),), (), (), tmp_path, tmp_path / "compiler", None)
    kwargs = dict(
        original_owner=owner,
        elf=original,
        executed_elf=copied,
        target="synthetic",
        directory=tmp_path,
        prefix=(str(wrapper),),
        limits=C.CensusLimits(),
        writable_roots=(tmp_path,),
        owner_sources=(owner_source,),
    )
    yield SimpleNamespace(**locals())
    for issuer in list(C._ISSUED):
        if C._ISSUED[issuer].directory == tmp_path:
            C._ISSUED.pop(issuer)


@pytest.mark.parametrize("damage", ["tool", "extension", "elf", "owner_source", "argv", "isa"])
def test_live_original_selection_rechecks_source_tools_elf_and_request(selected_files, damage):
    f = selected_files
    launch = C._issue_launch(**f.kwargs)
    assert "PRIVATE_TOKEN" not in launch.environment
    assert all("PRIVATE_TOKEN=" not in argument for argument in launch.argv)
    if damage in {"tool", "extension", "owner_source"}:
        getattr(f, damage).write_bytes(b"changed original selected bytes")
    elif damage == "elf":
        f.copied.write_bytes(b"changed copied ELF")
    elif damage == "argv":
        launch.argv.append("unselected command")
    else:
        launch.isa["CUSTOM_OPCODE"] = 0
    with pytest.raises(CensusUnavailable):
        launch.verify()


def test_original_default_extension_selection_needs_no_new_contract_metadata(selected_files, monkeypatch):
    f = selected_files
    flags = ("--extension=synthetic",)
    monkeypatch.setattr(f.backend, "spike_extension", lambda: (flags, f.tmp_path), raising=False)
    monkeypatch.setattr(C.spike_extension, "declared_extension", lambda target: None)
    monkeypatch.setattr(
        C.ET,
        "trace_invocation",
        lambda target, path: (
            [str(f.tool), *flags, "-l", "--log-commits", str(path)],
            {"LD_LIBRARY_PATH": str(f.tmp_path)},
        ),
    )
    monkeypatch.setattr(
        f.backend,
        "simulator_identity",
        lambda simulator: {
            "binary": C.F.digest(f.tool),
            "extensions": {"synthetic": C.F.digest(f.extension)},
            "digest": "original-default",
        },
    )
    launch = C._issue_launch(**f.kwargs)
    assert launch.verify()["original_model_sha256"] == "original-default"


def test_sampled_scratch_budget_counts_all_private_outputs(tmp_path):
    readonly = tmp_path / "readonly.elf"
    readonly.write_bytes(b"input" * 1000)
    (tmp_path / "one").write_bytes(b"a" * 3000)
    (tmp_path / "two").write_bytes(b"b" * 3000)
    with pytest.raises(CensusUnavailable, match="sampled byte budget"):
        P._writable_budget((tmp_path,), (readonly,), replace(C.CensusLimits(), max_output_bytes=4096))
