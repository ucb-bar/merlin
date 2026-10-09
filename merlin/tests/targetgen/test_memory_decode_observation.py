"""Actual decoder/product attribution, separately from runtime qualification."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.targetgen.contract.compile import _observed_memory_decode
from merlin.targetgen.contract.readback_policy import BUILD_RECEIPT, COHERENT_DUMP_V1, ReadbackPolicy


class Reader:
    def __init__(self, payload, *, stale=False, incomplete=False, mutate_elf=None, error=False, product_alias=None):
        self.payload = payload
        self.stale = stale
        self.incomplete = incomplete
        self.mutate_elf = mutate_elf
        self.error = error
        self.product_alias = product_alias

    def decode(self, console):
        if self.error:
            raise ValueError("actual diagnostic decoder refuses incomplete packet")
        assert console == "DONE\n"
        if self.mutate_elf is not None:
            self.mutate_elf.write_bytes(b"changed diagnostic bytes")
        if self.product_alias is not None:
            alias, excluded = self.product_alias
            alias.symlink_to(excluded)
        return ({} if self.incomplete else {"out": [[1, 2]]}), {
            "status": "complete",
            "payload": {
                "path": str(self.payload),
                "sha256": "0" * 64 if self.stale else hashlib.sha256(self.payload.read_bytes()).hexdigest(),
            },
            "observer_integrity": "UNKNOWN",
        }


def setup(tmp_path):
    work = tmp_path / "execution"
    work.mkdir()
    elf = work / "diagnostic.elf"
    elf.write_bytes(b"fixture only; no ELF or execution authority")
    (work / BUILD_RECEIPT).write_text("{}\n")
    (work / "oracle_console.log").write_text("DONE\n")
    payload = work / "full-parent-packet.bin"
    # A real producer writes these arbitrary bounded bytes. This control tests
    # process/callback attribution only; it is never an accelerator observation.
    result = I.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(b'\\x01\\x02')",
            str(payload),
        ],
        directory=work,
        stage="diagnostic_parent_product",
        outputs=(payload,),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        timeout=5,
    )
    result.check_returncode()
    cb = {
        "kernel_abi": {"kind": "whole_program", "outputs": ["out"]},
        "tensors": {"out": {"shape": [1, 2], "dtype": "i8", "role": "output"}},
    }
    return work, elf, payload, cb


def invoke(work, elf, cb, reader):
    return _observed_memory_decode(
        reader, "DONE\n", cb=cb, elf=elf, workdir=work, policy=ReadbackPolicy(COHERENT_DUMP_V1)
    )


def decode_record(work):
    rows = [(p, json.loads(p.read_text())) for p in work.glob("invocations/*/invocation.json")]
    return next((p, row) for p, row in rows if row["stage"] == "coherent_memory_decode")


def test_actual_decoder_retains_original_inputs_values_and_declared_full_product(tmp_path):
    work, elf, payload, cb = setup(tmp_path)
    outputs, evidence, observation = invoke(work, elf, cb, Reader(payload))
    assert outputs == {"out": [[1, 2]]}
    assert evidence["observer_integrity"] == "UNKNOWN"
    record = I.verify(Path(observation["record"]["path"]))
    assert {Path(row["path"]).name for row in record["inputs"]} == {
        elf.name,
        "oracle_console.log",
        BUILD_RECEIPT,
    }
    assert {Path(row["path"]).name for row in record["outputs"]} == {payload.name, "readback_decode.json"}
    returned = json.loads(Path(observation["product"]["path"]).read_text())
    assert returned["outputs"] == outputs and returned["memory_evidence"] == evidence
    payload.write_bytes(b"changed original packet")
    with pytest.raises(ValueError, match="changed"):
        I.verify(Path(observation["record"]["path"]))


@pytest.mark.parametrize("defect", ["foreign", "alias", "stale", "incomplete", "error"])
def test_actual_decoder_defects_retain_original_refusal_without_product_admission(tmp_path, defect):
    work, elf, payload, cb = setup(tmp_path)
    if defect == "foreign":
        payload = tmp_path / "excluded-owned-packet.bin"
        payload.write_bytes(b"harmless excluded data")
    if defect == "alias":
        alias = work / "alias.bin"
        alias.symlink_to(payload)
        payload = alias
    reader = Reader(payload, stale=defect == "stale", incomplete=defect == "incomplete", error=defect == "error")
    with pytest.raises(ValueError):
        invoke(work, elf, cb, reader)
    path, record = decode_record(work)
    assert record["status"] == "interrupted" and record["error"] == "ValueError"
    assert not (work / "readback_decode.json").exists()
    with pytest.raises(ValueError, match="did not complete"):
        I.verify(path)


def test_decoder_cannot_hide_original_elf_mutation_in_a_complete_return(tmp_path):
    work, elf, payload, cb = setup(tmp_path)
    with pytest.raises(ValueError, match="unchanged"):
        invoke(work, elf, cb, Reader(payload, mutate_elf=elf))
    path, record = decode_record(work)
    assert record["inputs_unchanged"] is False
    with pytest.raises(ValueError, match="unchanged"):
        I.verify(path)


@pytest.mark.parametrize("creation", ["before_decode", "during_decode"])
def test_decoder_product_alias_cannot_write_excluded_owned_bytes(tmp_path, creation):
    work, elf, payload, cb = setup(tmp_path)
    excluded = tmp_path / "excluded-owned-file.txt"
    original = b"harmless excluded bytes must remain unchanged\n"
    excluded.write_bytes(original)
    product = work / "readback_decode.json"
    if creation == "before_decode":
        product.symlink_to(excluded)
    reader = Reader(payload, product_alias=(product, excluded) if creation == "during_decode" else None)
    with pytest.raises((ValueError, FileExistsError)):
        invoke(work, elf, cb, reader)
    assert excluded.read_bytes() == original
    records = [json.loads(path.read_text()) for path in work.glob("invocations/*/invocation.json")]
    assert not any(row["stage"] == "coherent_memory_decode" and row["status"] == "completed" for row in records)


def test_existing_private_decoder_result_is_never_overwritten(tmp_path):
    work, elf, payload, cb = setup(tmp_path)
    product = work / "readback_decode.json"
    original = b"prior ordinary result must not be rewritten\n"
    product.write_bytes(original)
    with pytest.raises(ValueError, match="already exists"):
        invoke(work, elf, cb, Reader(payload))
    assert product.read_bytes() == original
