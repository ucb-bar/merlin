"""Complete neutral models for one producer and two physical readouts.

The original three-dispatch model computes its producer once.  These fixtures
declare no hardware, timing, or Phase 2 qualification.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from merlin.common import mlir_query as mq


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int]
    width: int
    shared: bool


CASES = (Case((3, 47), 1, False), Case((2, 23), 2, True))
OUTPUT_NAMES = ("first", "second", "state", "position", "input", "weight0", "weight1", "prior_state", "prior_position")


def scalar_sources(width):
    dtype = f"i{8 * width}"
    return tuple(
        (f"builtin.module {{func.func @forward({arguments})->{dtype} {{\n{body}\nfunc.return %r:{dtype}\n}}}}").encode()
        for arguments, body in (
            (f"%x:{dtype},%w:{dtype},%s:{dtype}", f"%p=arith.muli %x,%w:{dtype}\n%r=arith.addi %p,%s:{dtype}"),
            (f"%p:{dtype},%w:{dtype}", f"%r=arith.addi %p,%w:{dtype}"),
            (f"%p:{dtype},%w:{dtype}", f"%r=arith.muli %p,%w:{dtype}"),
        )
    )


def source(case):
    m, n = case.shape
    dtype = f"i{8 * case.width}"
    tensor = f"tensor<{m}x{n}x{dtype}>"
    identity = "affine_map<(i,j)->(i,j)>"

    def generic(name, inputs, body, empty):
        names = inputs.split(",")
        args = ("%x", "%w", "%s") if name == "p" else ("%v", "%w")
        return f"""%{name}=linalg.generic {{indexing_maps=[{",".join([identity] * (len(names) + 1))}],
 iterator_types=["parallel","parallel"]}} ins({inputs}:{",".join([tensor] * len(names))})
 outs(%{empty}:{tensor}) {{
 ^bb0({",".join(arg + ":" + dtype for arg in args)},%unused:{dtype}):
 {body}
 linalg.yield %r:{dtype}
 }}->{tensor}
"""

    # Empty destinations dominate every member, so the three actual scalar
    # computations are adjacent. The producer is intentionally not returned.
    producer = generic(
        "p", "%input,%w0,%state", f"%product=arith.muli %x,%w:{dtype}\n%r=arith.addi %product,%s:{dtype}", "e0"
    )
    first = generic("c0", "%p,%w1", f"%r=arith.addi %v,%w:{dtype}", "e1")
    second = generic("c1", "%p,%w1", f"%r=arith.muli %v,%w:{dtype}", "e2")
    module = mq.parse(f"""module {{func.func @forward(%w0:{tensor},%w1:{tensor},%input:{tensor},
 %state:{tensor},%position:tensor<i32>)->({tensor},{tensor},{tensor},tensor<i32>,
 {tensor},{tensor},{tensor},{tensor},tensor<i32>) {{
 %one=arith.constant 1:i32
 %e0=tensor.empty():{tensor}
 %e1=tensor.empty():{tensor}
 %e2=tensor.empty():{tensor}
 {producer}{first}{second}
 %se=tensor.empty():{tensor}
 %next=linalg.generic {{indexing_maps=[{identity},{identity},affine_map<(i,j)->()>,{identity}],
 iterator_types=["parallel","parallel"]}} ins(%c0,%c1,%position:{tensor},{tensor},tensor<i32>) outs(%se:{tensor}) {{
 ^bb0(%a:{dtype},%b:{dtype},%posarg:i32,%unused:{dtype}):
 %sum=arith.addi %a,%b:{dtype}
 %narrow=arith.trunci %posarg:i32 to {dtype}
 %r=arith.addi %sum,%narrow:{dtype}
 linalg.yield %r:{dtype}
 }}->{tensor}
 %old=tensor.extract %position[]:tensor<i32>
 %advanced=arith.addi %old,%one:i32
 %pe=tensor.empty():tensor<i32>
 %nextpos=tensor.insert %advanced into %pe[]:tensor<i32>
 return %c0,%c1,%next,%nextpos,%input,%w0,%w1,%state,%position:
 {tensor},{tensor},{tensor},tensor<i32>,{tensor},{tensor},{tensor},{tensor},tensor<i32>
}}}}""")
    for operation in module.walk():
        operation.attributes.pop("prov.region_id", None)
    module.verify()
    return module


def wrap(value, bits):
    value %= 1 << bits
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def reference(case):
    """Python integers preserve each multiply, add, and narrowing boundary."""
    bits = 8 * case.width
    dtype = np.dtype("i1" if case.width == 1 else "<i2")
    rng = np.random.default_rng(613 + case.width)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    arrays = [rng.integers(low, high + 1, case.shape).astype(dtype) for _ in range(4)]
    weight0, weight1, data, initial = arrays
    weight0[0, :], weight1[0, :], data[0, :], initial[0, :] = high, low, high, high
    state, position = initial.copy(), np.array(7, "<i4")
    expected, trace, witnesses = [], [], []
    for _ in range(3):
        prior, old_position = state.copy(), position.copy()
        p, first, second, next_state = (np.empty(case.shape, dtype) for _ in range(4))
        overflows = 0
        for index in np.ndindex(case.shape):
            product = int(data[index]) * int(weight0[index])
            overflows += int(not low <= product <= high)
            p[index] = wrap(wrap(product, bits) + int(prior[index]), bits)
            first[index] = wrap(int(p[index]) + int(weight1[index]), bits)
            second[index] = wrap(int(p[index]) * int(weight1[index]), bits)
            next_state[index] = wrap(
                wrap(int(first[index]) + int(second[index]), bits) + wrap(int(old_position), bits), bits
            )
        state, position = next_state, np.array(wrap(int(old_position) + 1, 32), "<i4")
        expected.append(
            tuple(a.copy() for a in (first, second, state, position, data, weight0, weight1, prior, old_position))
        )
        trace.append(tuple(a.copy() for a in (data, weight0, prior, weight1, p, first, second)))
        witnesses.append(overflows)
    return (weight0, weight1, data, initial, np.array(7, "<i4")), expected, trace, witnesses


@pytest.mark.parametrize("case", CASES)
def test_multi_output_fixture_has_one_producer_two_complete_consumers(case):
    from merlin.compile.scheduling.fusion_semantics import scalar_graph
    from merlin.compile.scheduling.pointwise_source import read_pointwise_source

    module = source(case)
    operations = [op for op in module.walk() if op.name == "linalg.generic"]
    assert len(operations) == 4 and operations[0].next_op is operations[1] and operations[1].next_op is operations[2]
    assert operations[1].inputs[0] is operations[2].inputs[0] is operations[0].results[0]
    assert len(tuple(operations[0].results[0].uses)) == 2
    for operation, scalar in zip(operations[:3], scalar_sources(case.width), strict=True):
        actual = read_pointwise_source(operation, max_elements=4096, max_operations=32, max_ir_bytes=65536)
        declared = scalar_graph(scalar, 32, 65536)
        assert actual.graph == declared
    arguments, epochs, trace, overflow = reference(case)
    assert len(epochs) == len(trace) == 3 and all(len(row) == 9 for row in epochs)
    assert all(count > 0 for count in overflow)
    for index, row in enumerate(epochs):
        np.testing.assert_array_equal(row[7], arguments[3] if index == 0 else epochs[index - 1][2])
        assert int(row[8]) == 7 + index
        data, w0, prior, w1, p, first, second = trace[index]
        expected_p = (data.astype(np.int64) * w0.astype(np.int64) + prior.astype(np.int64)).astype(data.dtype)
        np.testing.assert_array_equal(p, expected_p)
        np.testing.assert_array_equal(first, (p.astype(np.int64) + w1.astype(np.int64)).astype(data.dtype))
        np.testing.assert_array_equal(second, (p.astype(np.int64) * w1.astype(np.int64)).astype(data.dtype))


@pytest.mark.parametrize("case", CASES)
def test_original_three_dispatch_mapping_computes_each_scalar_body_once(tmp_path, case):
    from test_multi_output_provider import MultiOutputProvider, original_model

    from merlin.compile.scheduling.pointwise_source import compile_source_pointwise, read_pointwise_source
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets

    module = source(case)
    _, _, trace, _ = reference(case)
    model = original_model(tmp_path / "facts", case)
    provider = MultiOutputProvider(tmp_path / "provider", model, trace, case, grouped=False)
    operations = [op for op in module.walk() if op.name == "linalg.generic"]
    for operation in operations[:3]:
        actual = read_pointwise_source(operation, max_elements=4096, max_operations=32, max_ir_bytes=65536)
        mapped = provider.lower(actual)
        assert mapped is not None and len(mapped.kernel.body) == 1 and mapped.objects == ()
        result = compile_source_pointwise(
            module,
            operation,
            mapped.kernel,
            mapped.objects,
            model[1],
            model[2],
            model[3],
            selected_contracts=model[0],
            bindings=mapped.bindings,
            private_ownership=True,
            fusion_qualification_identity=model[1].qualification_identity,
            budgets=ScheduleBudgets(max_fusions=0, prefetch_distance=0),
        )
        assert len(result.compiled.kernel.body) == 1
        assert model[2].decode_calls(b"".join(result.images())) == result.compiled.kernel.body


@pytest.mark.parametrize("case", CASES)
def test_trace_checker_requires_one_persistent_producer_generation(tmp_path, case):
    from copy import deepcopy

    from test_multi_output_provider import (
        KEYS,
        OPS,
        MultiOutputProvider,
        check_actual_trace,
        original_model,
        parameters,
    )

    from merlin.sched.ir import Kernel, TensorArg

    _, _, trace, _ = reference(case)
    model = original_model(tmp_path / "facts", case)
    provider = MultiOutputProvider(tmp_path / "provider", model, trace, case, grouped=True)
    calls = tuple(replace_call(call) for call in provider.group_calls())
    args = tuple(TensorArg(f"input{i}", case.shape, f"i{8 * case.width}", "read") for i in range(4))
    args += tuple(TensorArg(f"output{i}", case.shape, f"i{8 * case.width}", "write") for i in range(2))
    kernel = Kernel("provenance_probe", args, calls)
    epoch = trace[0]
    rows = [
        [OPS.index(call.instr), *parameters(call).values(), *output.tobytes()]
        for call, output in zip(calls, epoch[4:], strict=True)
    ]
    checked = check_actual_trace(rows, (*epoch[:4], *epoch[5:]), kernel, case)
    assert checked["producer_elements"] == int(np.prod(case.shape)) and checked["commands"] == 3
    wrong = deepcopy(rows)
    wrong[1][1 + len(KEYS)] ^= 1
    with pytest.raises(AssertionError):
        check_actual_trace(wrong, (*epoch[:4], *epoch[5:]), kernel, case)
    with pytest.raises(AssertionError):
        check_actual_trace([rows[0], *rows], (*epoch[:4], *epoch[5:]), kernel, case)


def replace_call(call):
    """A pure relocated fixture, not a substitute for scheduler output."""
    from dataclasses import replace

    from merlin.sched.ir import Const, evaluate

    return replace(
        call,
        args=tuple(
            (key, Const(0) if key in ("ao", "out") and evaluate(value, {}) == 1024 else value)
            for key, value in call.args
        ),
    )


@pytest.mark.parametrize("case", CASES)
def test_ordinary_multi_output_model_keeps_one_producer_and_all_results(tmp_path, monkeypatch, case):
    import ctypes
    import hashlib
    import json
    import time
    from copy import deepcopy
    from pathlib import Path

    from test_dense_source_catalog import compilation_memory
    from test_multi_output_provider import COUNTERS, KEYS, MultiOutputProvider, check_actual_trace, original_model

    from merlin.common import invocation_record as invocation
    from merlin.common.digest import sha256_file
    from merlin.common.paths import merlin_dir
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.arena_bind import _call_target, _defines, _malloc_size
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.llvmlower.lower import lower_model_file
    from merlin.model_check.full_build import elf_sections
    from merlin.runtime.host_provider import revalidate_host_provider
    from merlin.targetgen import elf_lanes
    from merlin.transforms.certificates import ir_text

    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("ordinary native compiler and LLVM symbol inspector required")
    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    monkeypatch.setenv("MERLIN_SPLIT_FRESH_DEALLOCS", "0")
    monkeypatch.setenv("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "0")
    arguments, expected, trace, overflow = reference(case)
    source_path, inputs_path = tmp_path / "model.mlir", tmp_path / "inputs.npz"
    source_path.write_text(ir_text(source(case)))
    np.savez(inputs_path, **{f"in{i}": a for i, a in enumerate(arguments)})
    model = original_model(tmp_path / "facts", case)
    owners = tuple(
        merlin_dir() / "tests/infra" / name
        for name in (
            "test_multi_output_source.py",
            "test_multi_output_provider.py",
            "test_shared_lhs_provider.py",
            "test_pointwise_source_provider.py",
        )
    )
    retained = {
        str(p): sha256_file(p) for p in (source_path, inputs_path, *owners, model[0].facts_path, model[0].software_path)
    }
    retained[str(Path(elf_lanes.__file__).resolve())] = sha256_file(elf_lanes.__file__)
    original_pins = dict(retained)
    completed, arms = [], []
    for arm, grouped in (("baseline", False), ("grouped", True)):
        provider = MultiOutputProvider(tmp_path / (arm + "_provider"), model, trace, case, grouped=grouped)
        selection = provider.selection()
        started = time.perf_counter()
        with compilation_memory() as memory:
            result = lower_model_file(source_path, tmp_path / arm, shared_source_lowerings=(selection,))
        compile_seconds = time.perf_counter() - started
        record = result.stats["shared_source_lowering"]
        recipe = Path(record["compilation_recipe"]["path"])
        recipe_sha = record["compilation_recipe"]["sha256"]
        revalidate_host_provider(record["host_provider"])
        verify_completed_recipe(recipe, executable=result.host_so, expected_recipe_sha256=recipe_sha)
        linked_programs = []
        for item in provider.records:
            expected_image = item["image"].read_bytes()
            assert sha256_file(item["image"]) == item["image_sha256"]
            image, section = elf_lanes.read_program_section(
                result.host_so, ".merlin_multioutput." + item["bridge"], max_bytes=len(expected_image)
            )
            assert not section["writable"] and image == expected_image
            decoded = model[2].decode_calls(image)
            assert decoded == item["kernel"].body
            linked_programs.append(
                dict(
                    path=str(item["image"]),
                    sha256=item["image_sha256"],
                    symbol=item["bridge"],
                    elf_path=str(result.host_so),
                    elf_sha256=sha256_file(result.host_so),
                    section=section,
                    actual_section_sha256=hashlib.sha256(image).hexdigest(),
                    actual_instructions=[call.instr for call in decoded],
                )
            )
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        assert native.image_sha256 == sha256_file(result.host_so)
        observers = []
        for item in provider.records:
            symbol = item["symbol"]
            size, query, count = (
                getattr(native.lib, symbol + suffix) for suffix in ("_trace_size", "_trace", "_observe_count")
            )
            size.restype = count.restype = ctypes.c_uint64
            query.restype, query.argtypes, count.argtypes = ctypes.c_int64, [ctypes.c_int] * 2, [ctypes.c_int]
            assert size() == 0 and all(count(i) == 0 for i in range(8))
            observers.append((size, query, count))
        state, position = arguments[3].copy(), arguments[4].copy()
        history, events = [], []
        for epoch, want in enumerate(expected):
            inputs = (*arguments[:3], state.copy(), position.copy())
            owners, views = [], []
            for value in (*inputs, *(np.full_like(v, 71) for v in want)):
                owner = np.full(value.size + 32, 91, value.dtype)
                view = owner[16:-16].reshape(value.shape)
                view[...] = value
                owners.append(owner)
                views.append(view)
            original_bytes = [a.tobytes() for a in owners[:5]]
            before = [(size(), [count(i) for i in range(8)]) for size, _, count in observers]
            native([(a.ctypes.data, a.shape) for a in views])
            actual = views[5:]
            for value, expected_array in zip(actual, want, strict=True):
                np.testing.assert_array_equal(value, expected_array)
            assert [a.tobytes() for a in owners[:5]] == original_bytes
            assert all(np.all(a[:16] == 91) and np.all(a[-16:] == 91) for a in owners)
            assert actual[7].tobytes() == inputs[3].tobytes() and actual[8].tobytes() == inputs[4].tobytes()
            state, position = actual[2].copy(), actual[3].copy()
            raw = tmp_path / f"{arm}-epoch{epoch}.npz"
            np.savez(raw, **{f"owner{i}": a for i, a in enumerate(owners)})
            history.append(
                dict(
                    epoch=epoch,
                    outputs=[a.tolist() for a in actual],
                    raw_buffers=dict(path=str(raw), sha256=sha256_file(raw)),
                )
            )
            for index, (item, observation, start) in enumerate(zip(provider.records, observers, before, strict=True)):
                size, query, count = observation
                rows = [
                    [query(r, c) for c in range(1 + len(KEYS) + case.width * int(np.prod(case.shape)))]
                    for r in range(start[0], size())
                ]
                operands, results = provider.operands(item, epoch)
                effects_record = check_actual_trace(rows, (*operands, *results), item["kernel"], case)
                delta = [count(i) - start[1][i] for i in range(8)]
                assert delta[:4] == [
                    effects_record[key]
                    for key in ("commands", "requested_read_bytes", "requested_write_bytes", "producer_elements")
                ]
                assert delta[6] == 1
                assert delta[7] == 0
                events.append(
                    dict(
                        epoch=epoch, binding=index, actual_rows=rows, effects=effects_record, actual_counter_delta=delta
                    )
                )
        llvm = result.workdir / "model.ll"
        lines = llvm.read_text().splitlines()
        allocations = [
            dict(function=name, requested_bytes=_malloc_size(line.strip()))
            for name, start, end in _defines(lines)
            for line in lines[start : end + 1]
            if _call_target(line.strip()) == "malloc"
        ]
        assert allocations and all(type(a["requested_bytes"]) is int for a in allocations)
        receipt = Path(record["preparation"]["path"]).parent / "source_pointwise.json"
        assert len(json.loads(receipt.read_text())["bindings"]) == (1 if grouped else 3)
        arms.append(
            dict(
                arm=arm,
                source=deepcopy(record),
                history=history,
                events=events,
                compilation_wall_seconds=compile_seconds,
                compilation_memory=memory,
                compiler_only_peak_rss_bytes="UNKNOWN",
                execution_wall_seconds="UNKNOWN",
                qualified_latency="UNKNOWN",
                requested_transfer_bytes="UNKNOWN",
                physical_bus_traffic_bytes="UNKNOWN",
                counter_scope="actual neutral primitive and descriptor-bridge operations; CPU glue uninstrumented",
                model_llvm=dict(
                    path=str(llvm),
                    sha256=sha256_file(llvm),
                    static_malloc_sites=allocations,
                    allocation_scope=(
                        "consumed model LLVM, function-tagged static call sites and requests; "
                        "provider/qualifier excluded, dynamic allocator counts UNKNOWN"
                    ),
                ),
                source_receipt=dict(path=str(receipt), sha256=sha256_file(receipt)),
                elf=dict(
                    path=str(result.host_so),
                    sha256=sha256_file(result.host_so),
                    bytes=result.host_so.stat().st_size,
                    sections=elf_sections(result.host_so),
                ),
                programs=linked_programs,
                compiler_commands=len(json.loads(recipe.read_text())["commands"]),
            )
        )
        completed.append((deepcopy(record["host_provider"]), recipe, recipe_sha, result.host_so))
        for directory in (provider.work, result.workdir):
            for path in directory.rglob("*"):
                if path.is_file():
                    key, value = str(path.resolve()), sha256_file(path)
                    assert key not in retained or retained[key] == value
                    retained[key] = value
        for row in history:
            retained[row["raw_buffers"]["path"]] = row["raw_buffers"]["sha256"]
    baseline, grouped = arms
    assert [a["outputs"] for a in baseline["history"]] == [a["outputs"] for a in grouped["history"]]
    assert len(baseline["events"]) == 9 and len(grouped["events"]) == 3
    for arm in arms:
        assert sum(e["actual_counter_delta"][0] for e in arm["events"]) == 9
        assert sum(e["actual_counter_delta"][3] for e in arm["events"]) == 3 * int(np.prod(case.shape))
    assert sum(e["actual_counter_delta"][6] for e in baseline["events"]) == 9
    assert sum(e["actual_counter_delta"][6] for e in grouped["events"]) == 3
    bytes_per_p = int(np.prod(case.shape)) * case.width
    baseline_io = sum(sum(e["actual_counter_delta"][4:6]) for e in baseline["events"])
    grouped_io = sum(sum(e["actual_counter_delta"][4:6]) for e in grouped["events"])
    assert baseline_io - grouped_io == 12 * bytes_per_p  # P publish+2reloads and one duplicate weight pack, per epoch.
    assert len(grouped["model_llvm"]["static_malloc_sites"]) < len(baseline["model_llvm"]["static_malloc_sites"])
    assert sum(a["requested_bytes"] for a in baseline["model_llvm"]["static_malloc_sites"]) > sum(
        a["requested_bytes"] for a in grouped["model_llvm"]["static_malloc_sites"]
    )
    original_contracts = model[0].record()
    for provider_record, recipe, digest, executable in completed:
        revalidate_host_provider(provider_record)
        verify_completed_recipe(recipe, executable=executable, expected_recipe_sha256=digest)
    for receipt in tmp_path.rglob("invocations/*/invocation.json"):
        invocation.verify(receipt)
    for name, digest in retained.items():
        path = Path(name)
        assert path.is_file() and not path.is_symlink() and sha256_file(path) == digest
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                scope="complete ordinary native model, original neutral C effects; no target or timing qualification",
                original_inputs=original_pins,
                original_contracts=original_contracts,
                retained_files=retained,
                overflow_witness_counts=overflow,
                counter_metadata=[
                    dict(index=index, name=name, scope=scope) for index, (name, scope) in enumerate(COUNTERS)
                ],
                counter_limitations=(
                    "instruction and descriptor byte requests only; "
                    "not physical CPU writes, bus traffic or private initialization measurements"
                ),
                arms=arms,
            ),
            indent=2,
        )
    )
