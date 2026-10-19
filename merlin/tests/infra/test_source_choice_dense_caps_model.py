"""Broker-owned block caps change real reuse, with unchanged complete semantics.

These are synchronous neutral host primitives, not accelerator qualification.
"""

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from test_dense_source_catalog import compilation_memory
from test_selected_dense_contracts import inputs
from test_source_choice_broker_model import ChoiceProvider, broker_for, invoke, observations, reference, source
from test_source_model_api import reviewed_selection

from merlin.common.digest import sha256_file
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.model_check.full_build import elf_sections
from merlin.runtime.host_provider import revalidate_host_provider
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_planning import DenseResources


def test_broker_dense_caps_change_actual_reloads_and_preserve_complete_state(tmp_path, monkeypatch):
    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("native model compiler and inspector required")
    from merlin.llvmlower.source_choices import SourceDenseChoice, SourceDenseChoices

    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    # Both operands, output, and the reduction have real primitive tails. Two
    # same-shaped live contractions feed returned intermediates and carried state.
    m, k, width = 5, 10, 1
    source_path = tmp_path / "model.mlir"
    source_path.write_text(ir_text(source(m, k, "i8")))
    args, expected, trace = reference(m, k, width)
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": a for i, a in enumerate(args)})
    facts = tmp_path / "facts"
    facts.mkdir()
    model = inputs(facts, tile=(2, 3, 4), widths=(width, width, 4), shared=False, shape=(m, k, k))
    frozen_contracts = json.dumps(model[0].record(), sort_keys=True)
    provider = ChoiceProvider(tmp_path / "provider", model, trace, target_console=False)
    storage = model[2]
    resources = DenseResources(
        sum(view.capacity * view.unit_bytes for view in (storage.lhs, storage.rhs)),
        storage.accumulator.capacity * storage.accumulator.unit_bytes,
        3,
        3,
        2,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    baseline = replace(reviewed_selection(provider), resources=resources)
    owner = SourceDenseChoices(
        source_path,
        entry="forward",
        baseline=baseline,
        choices=(
            SourceDenseChoice("streamed", (("max_m_blocks", 1), ("max_n_blocks", 1), ("max_k_blocks", 1))),
            SourceDenseChoice("retained"),
        ),
        output_root=tmp_path / "control" / "private_products",
    )
    advertisement = owner.advertisement()
    roster = tmp_path / "control" / "choices.json"
    roster.write_text(json.dumps(advertisement))
    assert advertisement["choices"] == [
        dict(choice_id="streamed", controls=dict(max_m_blocks=1, max_n_blocks=1, max_k_blocks=1)),
        dict(choice_id="retained", controls={}),
    ]
    broker, candidate = broker_for(tmp_path, owner)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    reports = []
    for index, choice in enumerate(("streamed", "retained")):
        document = dict(schema="merlin.source_dense_choice_request.v1", source_sha256=source_sha, choice_id=choice)
        started = time.perf_counter()
        with compilation_memory() as memory:
            reply = invoke(broker, candidate, document, index)
        seconds = time.perf_counter() - started
        assert reply["returncode"] == 0, reply
        completed = owner.results[-1]
        public = json.loads(reply["stdout"])
        assert public == completed.record() and str(tmp_path) not in json.dumps(public)
        assert len(owner.results) == index + 1
        assert provider.preparations[-1] == (0, 1)
        assert json.dumps(model[0].record(), sort_keys=True) == frozen_contracts
        result = completed.lowered
        contract = result.stats["shared_source_lowering"]
        revalidate_host_provider(contract["host_provider"])
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        state, position, history = args[3].copy(), args[4].copy(), []
        for step in range(3):
            actual_args = [a.copy() for a in args[:3]] + [state.copy(), position.copy()]
            outputs = [np.full_like(a, 91) for a in expected[step]]
            native([(a.ctypes.data, a.shape) for a in (*actual_args, *outputs)])
            for got, want in zip(outputs, expected[step], strict=True):
                np.testing.assert_array_equal(got, want)
            for got, want in zip(actual_args[:3], args[:3], strict=True):
                np.testing.assert_array_equal(got, want)
            state, position = outputs[2].copy(), outputs[3].copy()
            history.append([a.tolist() for a in outputs])
        actual = observations(native, provider)
        assert len(actual) == 2
        counters = [sum(effect["counters"][i] for effect in actual) for i in range(6)]
        revalidate_host_provider(contract["host_provider"])
        completed.revalidate()
        receipt_path = Path(contract["preparation"]["path"]).parent / "source_dense.json"
        receipt = json.loads(receipt_path.read_text())
        assert len(receipt["bindings"]) == 2
        recipe = json.loads(Path(contract["compilation_recipe"]["path"]).read_text())
        assert recipe["status"] == "completed"
        reports.append(
            dict(
                choice_id=choice,
                public_response=public,
                broker_stdout_sha256=hashlib.sha256(reply["stdout"].encode()).hexdigest(),
                source_sha256=source_sha,
                source_choice_roster_sha256=sha256_file(roster),
                inputs_sha256=sha256_file(tmp_path / "inputs.npz"),
                action_registry_sha256=sha256_file(broker.receipt_path.parent / "action_registry.json"),
                original_selected_contracts=json.loads(frozen_contracts),
                original_resource_limits=asdict(resources),
                history=history,
                actual_effects=actual,
                actual_counters=counters,
                source=contract,
                private_receipt=str(completed.receipt_path),
                elf_sha256=sha256_file(result.host_so),
                elf_bytes=result.host_so.stat().st_size,
                elf_sections=elf_sections(result.host_so),
                compile_seconds=seconds,
                compile_scope="broker admission, owner checks and ordinary preparation/qualification/compile/link",
                compilation_memory=memory,
                compiler_commands=len(recipe["commands"]),
                static_native_instruction_count=None,
                timing_ns=None,
                scope=(
                    "observed primitive and bridge subset; other CPU traffic and total traffic unknown; "
                    "no latency claim"
                ),
            )
        )
    streamed, retained = reports
    assert streamed["history"] == retained["history"]
    before, after = streamed["actual_counters"], retained["actual_counters"]
    assert after[0] < before[0] and after[1] < before[1]
    # This pass changes operand retention. The complete output stores, bridge
    # packing and dispatch count remain necessary and are observed separately.
    assert before[2:] == after[2:]
    assert after[2] == 6 * m * k * 4
    assert after[3:5] == [6 * ((m * k + k * k) * width + m * k * 4)] * 2
    assert after[5] == 6
    ledger = [json.loads(line) for line in broker.receipt_path.read_text().splitlines()]
    assert len(ledger) == 2 and all(row["returncode"] == 0 and row["state"] == "complete" for row in ledger)
    assert [row["stdout_sha256"] for row in ledger] == [r["broker_stdout_sha256"] for r in reports]
    (tmp_path / "pair.json").write_text(json.dumps(reports, indent=2))
