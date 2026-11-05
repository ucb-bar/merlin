"""Public input and artifact-custody mutations; no target execution."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from packing_execution_inputs import (
    committed_custom_words,
    materialize,
    original_case,
    pin,
    recheck_artifacts,
    verify_event_bytes,
)

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("gemmini")


def test_public_materialization_validates_exact_source_abi_and_complete_reference(tmp_path, monkeypatch):
    source, reference, pins = original_case(monkeypatch)
    state = np.zeros((2, 12), dtype=np.int32)
    bound, arrays, expected = materialize(
        tmp_path / "materialized", epoch=0, state=state, source=source, reference=reference, source_pins=pins
    )
    assert len(arrays) == 4 and sum(value.size for value in expected) == 72
    assert len(bound.verify(bound.root)) == 8 and not (bound.root / "capture_receipt.json").exists()
    with pytest.raises(AssertionError):
        materialize(
            tmp_path / "wrong-source",
            epoch=0,
            state=state,
            source=source.replace("%sum = arith.addi", "%sum = arith.subi"),
            reference=reference,
            source_pins=pins,
        )
    (bound.root / "inputs.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        bound.verify(bound.root)


def test_commit_word_observation_refuses_disassembly_only_and_malformed_commits():
    with pytest.raises(AssertionError, match="no selected"):
        committed_custom_words(b"core 0: 0x8000 (0x0000000b) custom\n", match=11, mask=127)
    with pytest.raises(AssertionError, match="grammar"):
        committed_custom_words(b"core 0: 3 0x8000 bad\n", match=11, mask=127)
    assert committed_custom_words(b"core 0: 3 0x8000 (0x0000000b)\n", match=11, mask=127) == [
        {"hart": 0, "pc": 32768, "word": 11}
    ]


def test_retired_events_require_actual_executable_bytes(tmp_path, monkeypatch):
    # Reuse the existing synthetic ELF-format fixture; no target build or ISA decoder.
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/tests/targetgen"))
    from test_elf_lane_contract import build_elf

    elf = tmp_path / "program.elf"
    elf.write_bytes(build_elf([(".text", 6, struct.pack("<I", 11)), (".data", 2, struct.pack("<I", 11))]))
    event = {"hart": 0, "pc": 0x80000040, "word": 11}
    verify_event_bytes(elf, [event])
    for changed in ({**event, "word": 12}, {**event, "pc": event["pc"] + 4}, {**event, "pc": event["pc"] + 1}):
        with pytest.raises(AssertionError):
            verify_event_bytes(elf, [changed])


def test_final_reopen_refuses_late_artifact_and_observed_output_changes(tmp_path):
    from merlin.common import invocation_record as I

    first = tmp_path / "earlier.elf"
    first.write_bytes(b"original")
    expected = pin(first)
    observed = tmp_path / "observed"
    I.run([sys.executable, "-c", "print('complete')"], directory=observed, stage="test_custody", capture_output=True)
    invocation = next((observed / "invocations").glob("*/invocation.json"))
    recheck_artifacts([expected], invocations=[pin(invocation)])
    first.write_bytes(b"modified")
    with pytest.raises(AssertionError, match="retained artifact"):
        recheck_artifacts([expected], invocations=[pin(invocation)])
    first.write_bytes(b"original")
    record = json.loads(invocation.read_bytes())
    Path(record["stdout"]["path"]).write_bytes(b"different")
    with pytest.raises(ValueError):
        recheck_artifacts([expected], invocations=[pin(invocation)])


def test_final_reopen_checks_native_decoder_receipt_contents(tmp_path):
    # Synthetic custody metadata exercises no accessor or admission authority.
    output = tmp_path / "decoded.txt"
    output.write_bytes(b"11 4")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"pins": [{"role": "native-invocation", **pin(output)}]}))
    expected = pin(receipt)
    recheck_artifacts([], decoder_receipts=[expected])
    output.write_bytes(b"12 4")
    with pytest.raises(AssertionError, match="native decoder artifact"):
        recheck_artifacts([], decoder_receipts=[expected])
    output.write_bytes(b"11 4")
    receipt.write_text(json.dumps({"pins": []}))
    with pytest.raises(AssertionError, match="native decoder receipt"):
        recheck_artifacts([], decoder_receipts=[expected])
