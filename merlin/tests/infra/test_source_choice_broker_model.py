"""Original finite source choices reach real native compilation through the broker.

These explicit neutral host fixtures do not qualify a Phase 1 compiler or device.
"""

import ctypes
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_source_catalog import compilation_memory
from test_dense_source_trace import check_trace
from test_native_offload_model import CASES, planned, source
from test_native_source_model import NativeDenseProvider
from test_selected_dense_contracts import inputs
from test_source_model_api import reviewed_selection
from xdsl.ir import BlockArgument

from merlin.common.digest import sha256_file
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.model_check.full_build import elf_sections
from merlin.runtime.host_provider import revalidate_host_provider
from merlin.transforms.certificates import ir_text


class ChoiceProvider(NativeDenseProvider):
    """One original callback set serves every choice without shape-based dispatch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_trace = tuple(self.trace)
        self.preparations = []

    def prepare(self, bindings, private_directory):
        original_indices = []
        for binding in bindings:
            rhs = binding.compilation.operation.inputs[1]
            # The fixture's two independent source weights are exact forward
            # arguments. Their shapes coincide, so shape alone cannot map them.
            assert isinstance(rhs, BlockArgument) and rhs.index in (0, 1)
            original_indices.append(rhs.index)
        assert len(set(original_indices)) == len(original_indices)
        remap = {old: new for new, old in enumerate(original_indices)}
        self.trace = [(remap[index], *rest) for index, *rest in self.original_trace if index in remap]
        self.records = []
        self.work = private_directory / "neutral_provider"
        self.work.mkdir()
        self.preparations.append(tuple(original_indices))
        super().prepare(bindings, private_directory)


def reference(m, k, width):
    """Ordered scalar MACs and explicit signed modular source arithmetic."""
    dtype = np.dtype("i1" if width == 1 else "<i2")
    bits = 8 * width
    rng = np.random.default_rng(40931)
    limit = 1 << (bits - 1)
    w0, w1 = [rng.integers(-limit, limit, (k, k)).astype(dtype) for _ in range(2)]
    data, initial = [rng.integers(-limit, limit, (m, k)).astype(dtype) for _ in range(2)]

    def signed(value, size):
        value %= 1 << size
        return value - (1 << size) if value >= 1 << (size - 1) else value

    def product(a, b):
        result = np.empty((m, k), "<i4")
        for i in range(m):
            for j in range(k):
                total = 0
                for reduction in range(k):
                    total = signed(total + int(a[i, reduction]) * int(b[reduction, j]), 32)
                result[i, j] = total
        return result

    state, position = initial.copy(), 7
    expected, trace = [], []
    for _ in range(3):
        mixed = np.array(
            [[signed(int(data[i, j]) + int(state[i, j]), bits) for j in range(k)] for i in range(m)],
            dtype=dtype,
        )
        first = product(mixed, w0)
        activation = np.array(
            [[signed(max(int(first[i, j]), 0), bits) for j in range(k)] for i in range(m)], dtype=dtype
        )
        second = product(activation, w1)
        state = np.array(
            [[signed(signed(int(second[i, j]) + position, 32), bits) for j in range(k)] for i in range(m)],
            dtype=dtype,
        )
        position = signed(position + 1, 32)
        trace.extend(((0, mixed.copy(), w0.copy(), first.copy()), (1, activation.copy(), w1.copy(), second.copy())))
        expected.append(tuple(a.copy() for a in (first, second, state, np.array(position, "<i4"), data, w0, w1)))
    return (w0, w1, data, initial, np.array(7, "<i4")), expected, trace


def observations(native, provider):
    records = []
    for item in provider.records:
        symbol = item["symbol"]
        count = getattr(native.lib, symbol + "_observe_count")
        count.argtypes = [ctypes.c_int]
        count.restype = ctypes.c_uint64
        counters = [count(i) for i in range(6)]
        assert counters[5] == 3
        size = getattr(native.lib, symbol + "_trace_size")
        size.restype = ctypes.c_uint64
        query = getattr(native.lib, symbol + "_trace")
        query.argtypes = [ctypes.c_int, ctypes.c_int]
        query.restype = ctypes.c_int64
        raw = [[query(i, j) for j in range(query(i, 1) + 2)] for i in range(size())]
        assert len(raw) % 3 == 0
        per_call = len(raw) // 3
        decoded = [
            check_trace(
                raw[step * per_call : (step + 1) * per_call],
                item["problem"],
                provider.storage,
                item["result"].compiled.kernel.args,
            )
            for step in range(3)
        ]
        assert counters[:3] == [
            sum(row[key] for row in decoded) for key in ("commands", "requested_read_bytes", "requested_write_bytes")
        ]
        records.append(dict(counters=counters, raw_rows=raw, decoded=decoded, symbol=symbol))
    return records


def check_result_custody(completed, other_owner, directory):
    """Actual successful products cannot be copied or rebound to another owner."""
    copied_receipt = directory / "same-receipt.json"
    copied_receipt.write_bytes(completed.receipt_path.read_bytes())
    for clone in (replace(completed), replace(completed, receipt_path=copied_receipt)):
        with pytest.raises(ValueError):
            clone.record()
    for field, changed in (("receipt_path", copied_receipt), ("_owner", other_owner)):
        original = getattr(completed, field)
        try:
            object.__setattr__(completed, field, changed)
            with pytest.raises(ValueError):
                completed.record()
        finally:
            object.__setattr__(completed, field, original)
        completed.revalidate()
    original = completed.lowered.workdir
    try:
        object.__setattr__(completed.lowered, "workdir", other_owner.output_root)
        with pytest.raises(ValueError):
            completed.record()
    finally:
        object.__setattr__(completed.lowered, "workdir", original)
    completed.revalidate()


def broker_for(tmp_path, owner):
    from merlin_experiments.phase2 import broker as B
    from merlin_experiments.phase2 import broker_policy as BP
    from merlin_experiments.phase2.source_choices_workflow import SOURCE_CHOICES_V1

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    receipt = tmp_path / "control" / "receipts.jsonl"
    experiment = SimpleNamespace()
    policy = BP.select_workflow(
        SOURCE_CHOICES_V1,
        source_choices=owner,
        candidate=candidate,
        target_experiment=experiment,
        receipt_path=receipt,
    )
    actions = policy.build_registry()
    sealed = B.action_registry_contract(actions, candidate)
    actions = B.actions_from_registry_contract(sealed, candidate)
    (receipt.parent / "action_registry.json").write_text(json.dumps(sealed))
    broker = B.Broker(
        SimpleNamespace(argv=(), network="available_not_an_isolation_claim", clear_environment=True),
        experiment,
        candidate,
        actions,
        receipt,
        deadline=time.monotonic() + 300,
        max_calls=16,
        max_tool_seconds=120,
        workflow=policy,
    )
    return broker, candidate


def invoke(broker, candidate, document, index):
    from merlin_experiments.phase2.source_choices_workflow import ACTION

    request = candidate / f"request-{index}.json"
    request.write_text(json.dumps(document))
    return broker.execute({"action": ACTION, "bindings": {"request": request.name}})


@pytest.mark.parametrize("shape,tile,width,shared", CASES)
def test_candidate_choice_broker_compiles_complete_native_model(tmp_path, monkeypatch, shape, tile, width, shared):
    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("native model compiler and inspector required")
    from merlin.llvmlower.source_choices import SourceDenseChoice, SourceDenseChoices

    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    m, k = shape
    module = source(m, k, "i8" if width == 1 else "i16")
    source_path = tmp_path / "model.mlir"
    source_path.write_text(ir_text(module))
    args, expected, trace = reference(m, k, width)
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": a for i, a in enumerate(args)})
    facts = tmp_path / "facts"
    facts.mkdir()
    model = inputs(facts, tile=tile, widths=(width, width, 4), shared=shared, shape=(m, k, k))
    frozen_contracts = json.dumps(model[0].record(), sort_keys=True)
    provider = ChoiceProvider(tmp_path / "provider", model, trace, target_console=False)
    baseline = reviewed_selection(provider)
    _, bound = planned(baseline, module, (1,))
    owner = SourceDenseChoices(
        source_path,
        entry="forward",
        baseline=baseline,
        choices=(SourceDenseChoice("ordinary"), SourceDenseChoice("mixed", offload_plan=bound)),
        output_root=tmp_path / "control" / "private_products",
    )
    other_owner = SourceDenseChoices(
        source_path,
        entry="forward",
        baseline=baseline,
        choices=owner.choices,
        output_root=tmp_path / "control" / "other_original_owner",
    )
    broker, candidate = broker_for(tmp_path, owner)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    original_request = dict(
        schema="merlin.source_dense_choice_request.v1", source_sha256=source_sha, choice_id="ordinary"
    )
    invalid_requests = [
        {**original_request, "source_sha256": "0" * 64},
        {**original_request, "choice_id": "candidate-invented-choice"},
        *(
            {**original_request, field: {"claimed": True}}
            for field in ("controls", "source_ir", "output_dir", "hardware", "verifier", "qualification_identity")
        ),
    ]
    for index, invalid in enumerate(invalid_requests):
        rejected = invoke(broker, candidate, invalid, index)
        assert rejected["returncode"] != 0
        assert not owner.results and not provider.preparations
    histories, reports = [], []
    for index, (choice_id, wanted_indices) in enumerate((("ordinary", (0, 1)), ("mixed", (1,)))):
        request = dict(schema="merlin.source_dense_choice_request.v1", source_sha256=source_sha, choice_id=choice_id)
        started = time.perf_counter()
        with compilation_memory() as memory:
            reply = invoke(broker, candidate, request, index + len(invalid_requests))
        compile_seconds = time.perf_counter() - started
        assert reply["returncode"] == 0, reply
        public = json.loads(reply["stdout"])
        assert len(owner.results) == index + 1
        completed = owner.results[-1]
        result = completed.lowered
        assert public == completed.record()
        assert str(tmp_path) not in json.dumps(public)
        assert json.dumps(model[0].record(), sort_keys=True) == frozen_contracts
        assert provider.preparations[-1] == wanted_indices
        contract = result.stats["shared_source_lowering"]
        revalidate_host_provider(contract["host_provider"])
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        history = []
        state, position = args[3].copy(), args[4].copy()
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
        actual_effects = observations(native, provider)
        revalidate_host_provider(contract["host_provider"])
        assert len(actual_effects) == len(wanted_indices)
        check_result_custody(completed, other_owner, result.workdir)
        sections = elf_sections(result.host_so)
        counters = [sum(effect["counters"][i] for effect in actual_effects) for i in range(6)]
        receipt = Path(contract["preparation"]["path"]).parent / "source_dense.json"
        source_report = json.loads(receipt.read_text())
        assert len(source_report["bindings"]) == len(wanted_indices)
        if choice_id == "mixed":
            placement = source_report["offload_placement"][0]
            assert placement["actual_route"] == "candidate"
            assert placement["selected_dense_ordinals"] == list(bound.selected_ordinals)
        reports.append(
            dict(
                choice_id=choice_id,
                public_response=public,
                broker_stdout_sha256=hashlib.sha256(reply["stdout"].encode()).hexdigest(),
                original_selected_contracts=json.loads(frozen_contracts),
                source_sha256=source_sha,
                action_registry_sha256=sha256_file(broker.receipt_path.parent / "action_registry.json"),
                inputs_sha256=sha256_file(tmp_path / "inputs.npz"),
                private_receipt=str(completed.receipt_path),
                history=history,
                actual_effects=actual_effects,
                actual_counters=counters,
                source=contract,
                elf_sha256=sha256_file(result.host_so),
                elf_bytes=result.host_so.stat().st_size,
                elf_sections=sections,
                compile_seconds=compile_seconds,
                compile_scope="broker admission, owner checks and ordinary preparation/qualification/compile/link",
                compilation_memory=memory,
                timing_ns=None,
                scope="real broker/native complete model; neutral host provider only; CPU glue traffic unknown",
            )
        )
        histories.append(history)
    assert histories[0] == histories[1]
    assert [row["actual_counters"][5] for row in reports] == [6, 3]
    assert reports[0]["actual_counters"] == [2 * count for count in reports[1]["actual_counters"]]
    rows = [json.loads(line) for line in broker.receipt_path.read_text().splitlines()]
    assert len(rows) == len(invalid_requests) + 2
    assert all(row["returncode"] != 0 for row in rows[: len(invalid_requests)])
    assert all(row["returncode"] == 0 and row["state"] == "complete" for row in rows[-2:])
    assert [row["stdout_sha256"] for row in rows[-2:]] == [r["broker_stdout_sha256"] for r in reports]
    (tmp_path / "pair.json").write_text(json.dumps(reports, indent=2))
