"""Whole-model A/B for a qualified logical-pointer ABI with safe packing fallback.

The shared bridge owns descriptor checks and byte-copy loops. The private neutral
primitive remains independently qualified and its actual parameter trace checked.
No opaque C caching, hardware support or accelerator-latency claim is made.
"""

import hashlib
import importlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import test_source_model_api as api
from test_dense_ordinary_model import observe_console
from test_dense_source_catalog import compilation_memory
from test_dense_source_provider import DenseProvider

from merlin.common.digest import sha256_file
from merlin.runtime.backends import spike_model
from merlin.runtime.host_provider import revalidate_host_provider
from merlin.targetgen.elf_lanes import read_program_section

CASES = (((3, 7, 5), (2, 3, 4), 1, False), ((4, 9, 7), (3, 2, 5), 2, True))


def measure_phases(monkeypatch):
    """Observe actual ordinary build and functional simulator calls separately."""
    phases = {}

    def timed(name, implementation):
        def call(*args, **kwargs):
            started = time.perf_counter()
            with compilation_memory() as memory:
                result = implementation(*args, **kwargs)
            assert name not in phases
            phases[name] = dict(wall_seconds=time.perf_counter() - started, memory=memory)
            return result

        return call

    monkeypatch.setattr(spike_model, "build", timed("build", spike_model.build))
    monkeypatch.setattr(spike_model, "run_raw", timed("functional_simulator", spike_model.run_raw))
    return phases


def setup(tmp_path, monkeypatch, case):
    original = api.dense_inputs
    width = case[2]

    def inputs(*args, mutate=None, **kwargs):
        def declaration(facts, software, dense):
            if mutate is not None:
                mutate(facts, software, dense)
            dense["logical_pointer_abi"] = dict(
                schema="merlin.logical_pointer_abi.v1",
                layout="dense_row_major",
                alignment_bytes=[width, width, 4],
                access=["read_only", "read_only", "fully_written"],
                bounds="logical_elements",
                completion="synchronous",
            )

        return original(*args, mutate=declaration, **kwargs)

    with monkeypatch.context() as selected:
        selected.setattr(api, "dense_inputs", inputs)
        return api.setup(tmp_path, monkeypatch, *case)


def implementations():
    names = (
        "merlin.llvmlower.logical_borrowed_abi",
        "merlin.llvmlower.source_dense",
        "merlin.llvmlower.source_model",
        "merlin.compile.scheduling.dense_contracts",
        "merlin.runtime.host_provider",
    )
    return {name: sha256_file(Path(importlib.import_module(name).__file__)) for name in names}


def transfers(console):
    return [
        dict(index=int(parts[1]), values=[int(value) for value in parts[2:]])
        for line in console.splitlines()
        if line.startswith(b"MERLIN_DENSE_TRANSFER ")
        for parts in [line.split()]
    ]


def execution_evidence(observation, console, provider):
    witness = json.loads((provider.work / "native_qualification.json").read_text())
    primitive_calls = observe_console(console, provider, witness)
    actual = transfers(console)
    assert [row["index"] for row in actual] == [0, 1] * 3
    assert actual == [dict(index=row["index"], values=row["values"]) for row in witness["transfers"]]
    for trial in witness["additional_abi_trials"]:
        assert len(trial["calls"]) == len(trial["transfers"]) == 6
        for row in trial["transfers"]:
            if provider.borrow_contiguous:
                if trial["mode"] == "offset":
                    assert row["values"] == [0, 0, 7] and all(value > 0 for value in row["descriptor_offsets"])
                elif trial["mode"] == "strided":
                    assert row["values"][2] == 0
                elif trial["mode"] == "mixed":
                    assert row["values"][2] == 5
                elif trial["mode"] == "misaligned":
                    assert row["values"][2] == (3 if provider.selected.dense.dtypes[0] == "i8" else 0)
            else:
                assert row["values"][2] == 0
    host = observation["receipt"]["output"]["host_provider"]
    revalidate_host_provider(host)
    assert host["status"] == "completed" and len(host["objects"]) == 4
    sections = []
    for record in provider.records:
        primitive, header = read_program_section(
            Path(observation["receipt"]["output"]["elf"]), record["section"], max_bytes=64 << 10
        )
        assert not header["writable"]
        sections.append(dict(**header, sha256=hashlib.sha256(primitive).hexdigest()))
        record["result"].revalidate()
        # The provider defines its primitive/observer. The separate shared
        # object defines the ranked descriptor ABI, not an opaque wrapper.
        body = record["source"].read_text()
        assert "void " + record["bridge"] + "(" not in body
        assert "void " + record["observer"] + "(" in body
        assert str(record["logical_abi_source"]) in witness["sources"]
        assert witness["sources"][str(record["logical_abi_source"])] == sha256_file(record["logical_abi_source"])
    return dict(
        actual_primitive_calls=primitive_calls,
        actual_transfers=actual,
        qualification=witness,
        actual_kernel_sections=sections,
        completed_host_provider=host,
    )


@pytest.mark.parametrize("case", CASES)
def test_public_model_borrows_qualified_dense_buffers_and_retains_fallback(
    tmp_path, monkeypatch, record_property, case
):
    model, expected, trace, root, options = setup(tmp_path, monkeypatch, case)
    capture_hashes = {path.name: sha256_file(path) for path in options["capture"].iterdir()}
    original_facts, original_software = model[0].facts_bytes, model[0].software_bytes
    pins = implementations()
    observations, outputs = [], []
    for direct in (False, True):
        name = "borrow" if direct else "pack"
        provider = DenseProvider(
            tmp_path / (name + "_provider"), model, trace, logical_abi=True, borrow_contiguous=direct
        )
        with monkeypatch.context() as measured:
            phases = measure_phases(measured)
            observation, actual, console = api.execute(options, root / name, provider, expected)
        assert set(phases) == {"build", "functional_simulator"}
        observation["execution_phases"] = phases
        observation["phase_scope"] = (
            "build includes source preparation, native qualification, compilation and linking; "
            "sampled process-tree RSS includes this runner and shared pages; simulator wall includes "
            "instrumentation/console and is not hardware latency"
        )
        assert model[0].facts_bytes == original_facts and model[0].software_bytes == original_software
        assert pins == implementations()
        assert capture_hashes == {path.name: sha256_file(path) for path in options["capture"].iterdir()}
        evidence = execution_evidence(observation, console, provider)
        expected_masks = [7 if direct else 0] * 6
        assert [row["values"][2] for row in evidence["actual_transfers"]] == expected_masks
        observation.update(borrow_contiguous=direct, implementation_sources_sha256=pins, **evidence)
        (root / name / "logical_abi.json").write_text(json.dumps(observation, indent=2))
        observations.append(observation)
        outputs.append(actual)
    packed, borrowed = observations
    assert packed["source"]["source_sha256"] == borrowed["source"]["source_sha256"]
    assert packed["source"]["selected_sha256"] == borrowed["source"]["selected_sha256"]
    assert packed["model_object_sha256"] == borrowed["model_object_sha256"]
    for before, after in zip(*outputs, strict=True):
        assert before.dtype == after.dtype and before.shape == after.shape and before.tobytes() == after.tobytes()
    for before, after in zip(packed["actual_primitive_calls"], borrowed["actual_primitive_calls"], strict=True):
        assert before["trace"] == after["trace"] and before["effects"] == after["effects"]
        assert before["counters"][:3] == after["counters"][:3]
    for before, after in zip(packed["source"]["bindings"], borrowed["source"]["bindings"], strict=True):
        bp, ap = before["schedule"]["pipeline"], after["schedule"]["pipeline"]
        assert bp["library_sources_sha256"] == ap["library_sources_sha256"]
        assert bp["selected_contracts"] == ap["selected_contracts"]
        assert bp["context"] == ap["context"]
    packed_bytes = sum(row["values"][0] for row in packed["actual_transfers"])
    assert packed_bytes > 0 and all(row["values"][:2] == [0, 0] for row in borrowed["actual_transfers"])
    result = dict(
        case=case,
        capture_artifacts_sha256=capture_hashes,
        observations=observations,
        requested_bridge_read_bytes=dict(pack=packed_bytes, borrow=0),
        requested_bridge_write_bytes=dict(pack=packed_bytes, borrow=0),
        numerical_policy="complete source-ordered wrapping integer matmul/recurrent outputs; all6exact",
        limits=(
            "actual bridge copy-loop requested bytes and neutral primitive traces; no recurrent packing cache, "
            "bus-traffic or accelerator-latency claim; counters/logging alter generated code"
        ),
    )
    (root / "logical_abi_pair.json").write_text(json.dumps(result, indent=2))
    record_property("logical_dense_abi_pair", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize(
    "mutation", ["effects", "format", "budget", "control", "layout", "completion", "observer", "observer_state"]
)
def test_logical_abi_requires_original_qualification_and_controls(tmp_path, monkeypatch, mutation):
    from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects

    model, _expected, trace, _root, _options = setup(tmp_path, monkeypatch, CASES[0])
    provider = DenseProvider(tmp_path / "provider", model, trace, logical_abi=True)
    selection = provider.selection()
    with pytest.raises(ValueError, match="logical|scratch|format|effects|Boolean|boolean|borrow"):
        if mutation == "effects":
            replace(selection, borrowed_effects=BorrowedBufferEffects("different-original-effects", True, True, True))
        elif mutation == "format":
            replace(selection.logical_abi, format_qualification_identity="")
        elif mutation == "budget":
            # Admission or actual shape binding must reject insufficient fallback scratch.
            from merlin.llvmlower.logical_borrowed_abi import dense_buffers, scratch_bytes

            capability = replace(selection.logical_abi, max_stack_bytes=1)
            scratch_bytes(dense_buffers(model[1], model[0].dense.dtypes), capability)
        elif mutation == "control":
            replace(selection, borrow_contiguous=1)
        elif mutation == "observer":
            replace(selection.logical_abi, observer_effects=None)
        elif mutation == "observer_state":
            replace(selection.logical_abi, observer_preserves_logical_state=False)
        else:
            declaration = replace(
                selection.logical_abi.declaration,
                **{
                    mutation: "dense_column_major" if mutation == "layout" else "asynchronous",
                },
            )
            replace(selection.logical_abi, declaration=declaration)


@pytest.mark.parametrize(
    "mutation,diagnostic", [("address", "wrong tile address"), ("continuation", "Arrays are not equal")]
)
def test_wrong_logical_primitive_refuses_ordinary_model_before_execution(tmp_path, monkeypatch, mutation, diagnostic):
    from merlin.compile.baremetal_model import compile_saved_model
    from merlin.targetgen.rtl.facts import observed_facts

    model, _expected, trace, root, options = setup(tmp_path, monkeypatch, CASES[0])
    provider = DenseProvider(tmp_path / "provider", model, trace, logical_abi=True, mutation=mutation)

    def no_simulator(*args, **kwargs):
        pytest.fail("wrong primitive reached simulation")

    monkeypatch.setattr(spike_model, "run_raw", no_simulator)
    with observed_facts(model[0].target, json.loads(model[0].facts_bytes)):
        with pytest.raises(AssertionError, match=diagnostic):
            compile_saved_model(**options, output=root / "invalid", shared_source_lowerings=(provider.selection(),))
    assert not (provider.work / "native_qualification.json").exists()
    assert not list(root.rglob("*.elf"))
