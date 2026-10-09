"""Real bounded process/callback joins; fixtures confer no ELF or RTL roles."""

import hashlib
import json
import sys
from dataclasses import replace

import pytest
from merlin_experiments.phase2 import component_decode_products as C
from merlin_experiments.phase2.component_decode_products import join_component_decode_products
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common import invocation_record as I
from merlin.targetgen.contract.compile import _observed_memory_decode
from merlin.targetgen.contract.readback_policy import BUILD_RECEIPT, COHERENT_DUMP_V1, ReadbackPolicy


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def execute_fixture(elf, packet, root):
    return I.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(b'\\1\\376');print('DONE')",
            str(packet),
        ],
        directory=root,
        stage="diagnostic_fixture_process",
        inputs=(elf,),
        outputs=(packet,),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        timeout=5,
    )


def execute(elf, packet, root):
    with I.observe_call(
        root, stage="execution", function=execute_fixture, arguments={"scope": "fixture process only"}, inputs=(elf,)
    ) as observed:
        result = execute_fixture(elf, packet, root)
        result.check_returncode()
        observed.returned(stdout=result.stdout, stderr=result.stderr)
    return result.stdout.decode()


def unrelated_return(product):
    product.write_bytes(product.read_bytes())
    return "a different actual callback return\n"


class Reader:
    def __init__(self, packet, elf):
        self.packet, self.elf = packet, elf

    def decode(self, console):
        assert console == "DONE\n"
        return {"out": [[1, -2]]}, {
            "status": "complete",
            "payload": pin(self.packet),
            "original_elf": pin(self.elf),
            "observer_integrity": "UNKNOWN",
        }


def prepared(tmp_path):
    root = tmp_path / "execution"
    root.mkdir()
    elf, packet = root / "fixture.elf", root / "packet.bin"
    elf.write_bytes(b"Not an executable: process/product joins only")
    console = execute(elf, packet, root)
    (root / "oracle_console.log").write_text(console)
    (root / BUILD_RECEIPT).write_text("{}\n")
    cb = {
        "kernel_abi": {"kind": "whole_program", "outputs": ["out"]},
        "tensors": {"out": {"shape": [1, 2], "dtype": "i8", "role": "output"}},
    }
    outputs, memory, observation = _observed_memory_decode(
        Reader(packet, elf),
        console,
        cb=cb,
        elf=elf,
        workdir=root,
        policy=ReadbackPolicy(COHERENT_DUMP_V1),
    )
    result = {
        "elf": pin(elf),
        "console": pin(root / "oracle_console.log"),
        "native": {"elf": str(elf), "outputs": outputs, "readback_memory": memory, "readback_observation": observation},
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result))
    return root, result_path, result, elf, packet


def test_actual_same_elf_dispatch_console_decode_and_full_packet_join(tmp_path):
    root, result_path, _, _, packet = prepared(tmp_path)
    observed = join_component_decode_products(result_path=result_path, execution_root=root)
    observed.verify()
    assert observed.payload == (packet, pin(packet)["sha256"])
    assert {I.verify(path)["stage"] for path, _ in (observed.execution_record, observed.decoder_record)} == {
        "execution",
        "coherent_memory_decode",
    }
    assert "original_complete_per_call_histories" in observed.unknown
    assert "observer_integrity" in observed.unknown and "physical_timing" in observed.unknown
    assert "authority" in observed.record()["scope"] and "qualified_roles" not in observed.record()


@pytest.mark.parametrize("defect", ["elf", "values", "console", "payload", "decoder"])
def test_disconnected_matching_summary_hashes_do_not_replace_actual_join(tmp_path, defect):
    root, result_path, result, elf, packet = prepared(tmp_path)
    if defect == "elf":
        other = root / "unexecuted.elf"
        other.write_bytes(elf.read_bytes())
        result["elf"], result["native"]["elf"] = pin(other), str(other)
    elif defect == "values":
        result["native"]["outputs"]["out"][0][0] = 7
    elif defect == "console":
        other = root / "unconsumed-console.txt"
        other.write_bytes((root / "oracle_console.log").read_bytes())
        result["console"] = pin(other)
    elif defect == "payload":
        other = root / "unobserved-packet.bin"
        other.write_bytes(packet.read_bytes())
        result["native"]["readback_memory"]["payload"] = pin(other)
    else:
        execution = next(
            path
            for path in root.glob("invocations/*/invocation.json")
            if json.loads(path.read_text())["stage"] == "execution"
        )
        result["native"]["readback_observation"]["record"] = pin(execution)
    result_path.write_text(json.dumps(result))
    with pytest.raises(StageGateError):
        join_component_decode_products(result_path=result_path, execution_root=root)


def test_second_actual_same_elf_dispatch_cannot_supply_unique_execution_membership(tmp_path):
    root, result_path, _, elf, packet = prepared(tmp_path)
    execute(elf, packet, root)
    with pytest.raises(StageGateError, match="unique actual same-ELF"):
        join_component_decode_products(result_path=result_path, execution_root=root)


def test_actual_recorded_return_must_equal_selected_decoder_product(tmp_path):
    root, result_path, result, elf, packet = prepared(tmp_path)
    product = root / "readback_decode.json"
    with I.observe_call(
        root,
        stage="coherent_memory_decode",
        function=unrelated_return,
        arguments={},
        inputs=(elf, root / "oracle_console.log", root / BUILD_RECEIPT),
        outputs=(product, packet),
    ) as observed:
        value = unrelated_return(product)
        observed.returned(stdout=value)
    I.verify(observed.path)
    result["native"]["readback_observation"]["record"] = pin(observed.path)
    result_path.write_text(json.dumps(result))
    with pytest.raises(StageGateError, match="actual recorded return"):
        join_component_decode_products(result_path=result_path, execution_root=root)


@pytest.mark.parametrize("defect", ["alias", "foreign"])
def test_owned_execution_paths_cannot_be_reselected_through_alias_or_foreign_result(tmp_path, defect):
    root, result_path, _, _, _ = prepared(tmp_path)
    selected = tmp_path / "foreign-result.json" if defect == "foreign" else root / "alias-result.json"
    if defect == "alias":
        selected.symlink_to(result_path)
    else:
        selected.write_bytes(result_path.read_bytes())
    with pytest.raises(StageGateError):
        join_component_decode_products(result_path=selected, execution_root=root)


def test_result_alias_refuses_before_reading_its_target_bytes(tmp_path, monkeypatch):
    root, result_path, _, _, _ = prepared(tmp_path)
    alias = root / "alias-result.json"
    alias.symlink_to(result_path)
    reads = []
    original = C.sha256_file

    def observed_read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(C, "sha256_file", observed_read)
    with pytest.raises(StageGateError, match="before reading"):
        join_component_decode_products(result_path=alias, execution_root=root)
    assert reads == []


def test_scanned_invocation_alias_refuses_before_reading_excluded_owned_file(tmp_path, monkeypatch):
    root, result_path, _, _, _ = prepared(tmp_path)
    excluded = tmp_path / "excluded-owned-invocation.json"
    original_bytes = b"harmless excluded owned file is not an invocation\n"
    excluded.write_bytes(original_bytes)
    nested = root / "untrusted-scan-member"
    nested.mkdir()
    alias = nested / "invocation.json"
    alias.symlink_to(excluded)
    reads = []
    original = C.mapping_file

    def observed_read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(C, "mapping_file", observed_read)
    with pytest.raises(StageGateError, match="before reading"):
        join_component_decode_products(result_path=result_path, execution_root=root)
    assert alias not in reads and excluded not in reads
    assert excluded.read_bytes() == original_bytes


@pytest.mark.parametrize("defect", ["packet", "result", "scope"])
def test_reopened_observation_retains_actual_bytes_and_unknown_scope(tmp_path, defect):
    root, result_path, _, _, packet = prepared(tmp_path)
    observed = join_component_decode_products(result_path=result_path, execution_root=root)
    if defect == "packet":
        packet.write_bytes(b"changed")
    elif defect == "result":
        result_path.write_text("{}")
    else:
        observed = replace(observed, unknown=())
    with pytest.raises(StageGateError):
        observed.verify()
