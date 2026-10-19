"""Ordinary complete-model isolation of temporary reuse and exact-copy removal.

This interpreted neutral machine executes on the host CPU. Native/Spike access
counters observe requested bytes and touched address high-water, not bus traffic,
accelerator latency, dynamically allocated storage, or physical capacity savings.
"""

import json
import struct
import time
from dataclasses import asdict, replace

import numpy as np
import pytest
from test_dense_source_catalog import compilation_memory
from test_pointwise_source import source_program
from test_pointwise_storage_model_provider import CopyProvider, StorageProvider
from test_spilled_pointwise_model import capture
from test_spilled_pointwise_source import source_text

from merlin.common.digest import sha256_file
from merlin.compile.scheduling.pointwise_source import compile_source_pointwise
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike, spike_model
from merlin.sched.ir import evaluate
from merlin.targetgen.elf_lanes import read_program_section
from merlin.targetgen.isa_dialect_gen import decode_instruction


def decoded_observation(image, selected, separate):
    """Reconstruct actual addresses/traffic from bytes, independently of plans."""
    word = struct.Struct("<BBBQQQ")
    commands, reads, writes, copies, high = [], 0, 0, 0, [0, 0]
    for offset in range(0, len(image), word.size):
        op, read, write, source, destination, size = word.unpack_from(image, offset)
        assert op in (0, 1) and read < 4 and write < 4 and size > 0
        name, fields = decode_instruction(selected.isa, int.from_bytes(image[offset : offset + word.size], "little"))
        assert name == ("copy" if op == 0 else "increment")
        assert fields == dict(read_bank=read, write_bank=write, source=source, destination=destination, size=size)
        for bank, start in ((read, source), (write, destination)):
            if bank < 2:
                high[bank if separate else 0] = max(high[bank if separate else 0], start + size)
                assert start >= (8 if bank == 0 else 16) and start + size <= 192
                if bank == 0 or not separate:
                    assert all(start + size <= lo or start >= hi for lo, hi in ((32, 40), (64, 72)))
            else:
                assert start + size <= 42
        commands.append(dict(instruction=name, **fields))
        reads += size
        writes += size
        copies += op == 0
    return dict(
        commands=commands, counters=[len(commands), reads, writes], copies=copies, touched_address_high_water=high
    )


def paired_model(tmp_path, shape, separate, provider_class):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle = tmp_path / "capture"
    expected, trace = capture(bundle, shape)
    outputs, observations, providers = [], [], []
    for name, enabled in (("baseline", False), ("optimized", True)):
        provider = provider_class(tmp_path / (name + "_provider"), enabled=enabled, separate=separate)
        work = tmp_path / name
        work.mkdir()
        selection = provider.selection(trace)
        assert selection.borrowed_effects is None
        # Original caller review of the private byte-copy/interpreter wrappers:
        # descriptor pointers stay in local arrays, caller storage is never
        # released, and all work completes before return. This review is
        # separate from the numerical/traffic execution qualification below.
        selection = replace(
            selection,
            borrowed_effects=BorrowedBufferEffects(
                "reviewed-neutral-storage-model-ABI-lifetime-v1",
                True,
                True,
                True,
            ),
        )
        started = time.perf_counter()
        with compilation_memory() as memory:
            built = spike_model.build(
                bundle,
                work,
                arena_mb=1,
                dram_bytes=64 << 20,
                code_reserve=16 << 20,
                backend="scalar",
                full_readback=True,
                shared_source_lowerings=(selection,),
                host_llvm_bitcode=True,
                cflags_override=[
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                ],
            )
        elapsed = time.perf_counter() - started
        console, wall = FB.run_host(built["elf"], timeout_s=30)
        (work / "console.bin").write_bytes(console)
        actual, metadata = FB.read(console, bundle, built["build_hash"])
        assert len(actual) == len(expected) == 5
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()
        outputs.append(actual)
        native = json.loads((provider.work / "native_qualification.json").read_text())
        counters = [
            [int(value) for value in line.split()[1:]]
            for line in console.splitlines()
            if line.startswith(b"MERLIN_POINTWISE_COUNTS ")
        ]
        assert len(counters) == len(trace) == 6 and counters[-1] == native["counters"]
        receipts = list(work.rglob("source_pointwise.json"))
        assert len(receipts) == 1
        ordinary = json.loads(receipts[0].read_text())
        assert ordinary["schema"] == "merlin.source_pointwise_lowering.v1" and len(ordinary["bindings"]) == 2
        assert ordinary["selected_sha256"] == sha256_file(receipts[0].with_suffix(".mlir"))
        linked = []
        for record, result in zip(provider.records, provider.results, strict=True):
            image, section = read_program_section(built["elf"], record["program_section"], max_bytes=27 * 16)
            assert not section["writable"] and image == b"".join(result.images())
            assert sha256_file(provider.work / str(len(linked)) / "program.bin") == record["program_sha256"]
            observed = decoded_observation(image, result.selected_contracts, separate)
            assert observed["counters"] == [
                result.conformance[key] for key in ("actual_commands", "requested_read_bytes", "requested_write_bytes")
            ]
            storage = result.compiled.report["storage"]
            banks = {bank["name"]: bank for bank in storage["banks"]}
            touched = {}
            for i, bank in enumerate(banks.values()):
                high = observed["touched_address_high_water"][i if separate else 0]
                touched[bank["backing"]] = max(touched.get(bank["backing"], 0), high)
            assert touched == storage["temporary_high_water_bytes"]
            reserved = dict(touched)
            for reservation in storage["reservations"]:
                backing = banks[reservation["storage"]]["backing"]
                reserved[backing] = max(reserved.get(backing, 0), reservation["start"] + reservation["extent"])
            assert reserved == storage["physical_high_water_bytes"]
            linked.append(
                dict(
                    **section,
                    sha256=record["program_sha256"],
                    image_bytes=len(image),
                    **observed,
                    storage=storage,
                )
            )
        cumulative = [0] * 9
        for row, (index, data) in zip(counters, trace, strict=True):
            observed = linked[index]
            for i in range(3):
                cumulative[i] += observed["counters"][i]
            cumulative[3] += 2 * data.size
            cumulative[4] += 2 * data.size
            cumulative[5] += 1
            cumulative[6:8] = [
                max(a, b) for a, b in zip(cumulative[6:8], observed["touched_address_high_water"], strict=True)
            ]
            cumulative[8] += observed["copies"]
            assert row == cumulative
        assert native["counters"][3:6] == [provider.packed_bytes(shape)] * 2 + [6]
        assert all(result.compiled.report["repair_rounds"] == 0 for result in provider.results)
        recipe = json.loads((work / "compilation_recipe.json").read_text())
        observations.append(
            dict(
                arm=name,
                optimization=provider.optimization,
                budgets=asdict(selection.budgets),
                compile_seconds=elapsed,
                compilation_memory=memory,
                elf_sha256=sha256_file(built["elf"]),
                elf_bytes=built["elf"].stat().st_size,
                elf_sections=FB.elf_sections(built["elf"]),
                compiler_commands=len(recipe["commands"]),
                host_object_sha256=sha256_file(work / "model.o"),
                provider_object_bytes=(work / "host_provider" / "provider.o").stat().st_size,
                encoded_program_bytes=sum(p["image_bytes"] for p in linked),
                linked_programs=linked,
                native_qualification=native,
                actual_model_counter_trace=counters,
                actual_model_counters=cumulative,
                observed_payload_subset_bytes=sum(cumulative[i] for i in (1, 2, 3, 4)),
                unobserved_host_and_physical_traffic_bytes=None,
                source_receipt=ordinary,
                selected_facts_sha256=sha256_file(provider.work / "capabilities" / "facts.json"),
                selected_software_sha256=sha256_file(provider.work / "capabilities" / "software.yaml"),
                library_sources=dict(provider.results[0].library_sources),
                functional_forward_instructions=metadata["metrics"]["cycles"],
                simulator_process_wall_seconds=wall,
                numerical_agreement="all five complete outputs/state/input/weight bytes exact to independent reference",
                counter_scope="neutral host byte interpreter; requested transfers, not bus bytes or accelerator cycles",
                storage_scope="touched high-water; fixed 192-byte bank windows plus eight guard bytes per local array",
                host_accelerator_timing="unmeasured; no accelerator, no latency/speedup claim",
            )
        )
        providers.append(provider)
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)
    before, after = observations
    changed_controls = {key for key, value in before["budgets"].items() if after["budgets"][key] != value}
    assert changed_controls == {"optimize_storage" if provider_class is StorageProvider else "eliminate_copies"}
    for key in ("host_object_sha256", "selected_facts_sha256", "selected_software_sha256", "library_sources"):
        assert before[key] == after[key]
    for key in ("source_sha256", "selected_sha256"):
        assert before["source_receipt"][key] == after["source_receipt"][key]
    assert before["actual_model_counters"][3:6] == after["actual_model_counters"][3:6]
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
    return observations, providers


@pytest.mark.parametrize("shape,separate", [((1, 17, 16), False), ((2, 9, 8), True)])
@pytest.mark.parametrize("provider_class", [StorageProvider, CopyProvider], ids=["storage", "copies"])
def test_ordinary_complete_model_isolates_storage_and_copy_passes(
    tmp_path, record_property, shape, separate, provider_class
):
    observations, providers = paired_model(tmp_path, shape, separate, provider_class)
    before, after = observations
    if provider_class is StorageProvider:
        assert before["actual_model_counters"][:6] == after["actual_model_counters"][:6]
        assert before["actual_model_counters"][8] == after["actual_model_counters"][8]
        assert sum(after["actual_model_counters"][6:8]) < sum(before["actual_model_counters"][6:8])
        for provider in providers:
            for result, program in zip(
                provider.results, observations[int(provider.enabled)]["linked_programs"], strict=True
            ):
                addresses = result.compiled.report["storage"]["addresses"]
                assert len(addresses) == 3
                actual_starts = [
                    call["destination"]
                    for call in program["commands"]
                    if call["instruction"] == "copy" and call["read_bank"] == 2
                ]
                assert actual_starts == [addresses["value_" + str(i)] for i in range(3)]
                if provider.enabled:
                    # The first and third chunks use the SAME physical bank,
                    # have disjoint lifetimes, and really reuse an address.
                    assert actual_starts[0] == actual_starts[2]
                else:
                    assert len(set(actual_starts)) == 3
    else:
        copies_removed = 0 if separate else 18
        assert before["actual_model_counters"][0] - after["actual_model_counters"][0] == copies_removed
        assert before["actual_model_counters"][8] - after["actual_model_counters"][8] == copies_removed
        transfer_removed = 0 if separate else 3 * shape[0] * (shape[1] + shape[2])
        for counter in (1, 2):
            assert (
                before["actual_model_counters"][counter] - after["actual_model_counters"][counter] == transfer_removed
            )
        for program in after["linked_programs"]:
            elimination = program["storage"].get("copy_elimination", {})
            assert elimination.get("eliminated_exact_copies", 0) == (0 if separate else 3)
    record_property("whole_model_storage_copy_pair", json.dumps(observations, sort_keys=True))


def test_wrong_actual_address_fails_native_before_optimized_model_link(tmp_path):
    class WrongAddressProvider(CopyProvider):
        corrupt_address = True

    with pytest.raises(AssertionError, match="Arrays are not equal"):
        paired_model(tmp_path, (1, 17, 16), False, WrongAddressProvider)
    assert (tmp_path / "baseline" / "model.elf").is_file()
    assert not (tmp_path / "optimized_provider" / "native_qualification.json").exists()
    assert not (tmp_path / "optimized" / "model.elf").exists()
    work = tmp_path / "optimized_provider"
    records = json.loads((work / "writers.json").read_text())
    changed = [work / (record["symbol"] + "_wrong_address.bin") for record in records]
    assert len(changed) == 2 and all(path.is_file() for path in changed)
    assert all(
        path.read_bytes() != (path.parent / str(i) / "program.bin").read_bytes() for i, path in enumerate(changed)
    )


def test_unknown_or_incomplete_transfer_completion_refuses_source_mapping(tmp_path):
    provider = StorageProvider(tmp_path / "provider", enabled=True, separate=False)
    provider.selection([])
    module, operation = source_program(source_text(1, 17))
    directory = tmp_path / "async"
    directory.mkdir()
    options = provider.case_options(directory, 1, 17)
    body = options["kernel"].body
    # A real async opcode is declared in the same selected release. Its absent
    # wait cannot masquerade as a synchronous initialized input fragment.
    options["kernel"] = replace(options["kernel"], body=(replace(body[0], instr="async_copy"), *body[1:]))
    with pytest.raises(ValueError, match="synchronous|completion|pending|asynchronous"):
        compile_source_pointwise(module, operation, budgets=provider.schedule_budgets(), **options)
    assert evaluate(body[0].arg("source"), {}) == 0
