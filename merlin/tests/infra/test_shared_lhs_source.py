"""Ordinary grouped dense execution with explicit neutral primitive premises.

The fixed resource envelopes and original provider qualify only these test
execution models. They supply no hardware, timing, or compiler-authoring grant.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from test_dense_source import contraction

from merlin.common import mlir_query as mq
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int, int, int]
    tile: tuple[int, int, int]
    width: int
    shared: bool


CASES = (
    Case((5, 5, 5, 9), (2, 3, 4), 1, False),
    Case((4, 3, 4, 7), (3, 2, 5), 2, True),
)
OUTPUT_NAMES = ("first", "second", "state", "position", "input", "weight0", "weight1", "prior_state", "prior_position")


def source(case):
    """Two adjacent reductions share an exact SSA operand and complete K order."""
    m, k, n0, n1 = case.shape
    dtype = f"i{8 * case.width}"
    narrow, wide = f"tensor<{m}x{k}x{dtype}>", f"tensor<{m}x{k}xi32>"
    out0, out1 = f"tensor<{m}x{n0}xi32>", f"tensor<{m}x{n1}xi32>"
    w0, w1 = f"tensor<{k}x{n0}x{dtype}>", f"tensor<{k}x{n1}x{dtype}>"
    pieces = []
    initializers = []
    for i, n in enumerate((n0, n1)):
        text = contraction((m, k, n), (dtype, dtype, "i32"), lhs="mixed", rhs=f"w{i}", output=f"c{i}")
        prefix, body = text.split(f" %c{i} = linalg.generic", 1)
        initializers.append(prefix)
        pieces.append(f" %c{i} = linalg.generic" + body)
    module = mq.parse(f"""module {{
func.func @forward(%w0:{w0},%w1:{w1},%input:{narrow},%state:{narrow},%position:tensor<i32>)
 ->({out0},{out1},{narrow},tensor<i32>,{narrow},{w0},{w1},{narrow},tensor<i32>) {{
 %one=arith.constant 1:i32
 %empty=tensor.empty():{narrow}
 %mixed=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%input,%state:{narrow},{narrow}) outs(%empty:{narrow}) {{
 ^bb0(%x:{dtype},%s:{dtype},%unused:{dtype}):
 %sum=arith.addi %x,%s:{dtype}
 linalg.yield %sum:{dtype}
 }}->{narrow}
 {"".join(initializers)}
 {"".join(pieces)}
 %view0="tensor.extract_slice"(%c0) <{{operandSegmentSizes=array<i32:1,0,0,0>,
 static_offsets=array<i64:0,0>,static_sizes=array<i64:{m},{k}>,static_strides=array<i64:1,1>}}>
 :({out0})->{wide}
 %view1="tensor.extract_slice"(%c1) <{{operandSegmentSizes=array<i32:1,0,0,0>,
 static_offsets=array<i64:0,0>,static_sizes=array<i64:{m},{k}>,static_strides=array<i64:1,1>}}>
 :({out1})->{wide}
 %se=tensor.empty():{narrow}
 %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->()>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%view0,%view1,%position:{wide},{wide},tensor<i32>) outs(%se:{narrow}) {{
 ^bb0(%a:i32,%b:i32,%p:i32,%unused:{dtype}):
 %sum=arith.addi %a,%b:i32
 %advanced=arith.addi %sum,%p:i32
 %value=arith.trunci %advanced:i32 to {dtype}
 linalg.yield %value:{dtype}
 }}->{narrow}
 %pv=tensor.extract %position[]:tensor<i32>
 %pn=arith.addi %pv,%one:i32
 %pe=tensor.empty():tensor<i32>
 %nextpos=tensor.insert %pn into %pe[]:tensor<i32>
 return %c0,%c1,%next,%nextpos,%input,%w0,%w1,%state,%position:
 {out0},{out1},{narrow},tensor<i32>,{narrow},{w0},{w1},{narrow},tensor<i32>
}} }}""")
    for operation in module.walk():
        operation.attributes.pop("prov.region_id", None)
    module.verify()
    return module


def wrap(value, bits):
    value %= 1 << bits
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def reference(case):
    """Ordered Python-integer reduction, independent of blocking and primitive IR."""
    m, k, n0, n1 = case.shape
    dtype = np.dtype("i1" if case.width == 1 else "<i2")
    bits = 8 * case.width
    rng = np.random.default_rng(741 + case.width)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    w0 = rng.integers(low, high + 1, (k, n0)).astype(dtype)
    w1 = rng.integers(low, high + 1, (k, n1)).astype(dtype)
    data = rng.integers(low, high + 1, (m, k)).astype(dtype)
    initial = rng.integers(low, high + 1, (m, k)).astype(dtype)
    data[0, :], initial[0, :], w0[:, 0], w1[:, 0] = high, 0, high, high
    data[-1, 0], initial[-1, 0] = high, 1
    state, position = initial.copy(), np.array(7, "<i4")
    expected, trace, witnesses = [], [], []
    for _ in range(3):
        before, old_position = state.copy(), position.copy()
        lhs = np.array([[wrap(int(data[i, j]) + int(before[i, j]), bits) for j in range(k)] for i in range(m)], dtype)
        outputs, overflow = [], 0
        for weight in (w0, w1):
            result = np.empty((m, weight.shape[1]), "<i4")
            for i in range(m):
                for j in range(weight.shape[1]):
                    total = 0
                    for t in range(k):
                        unbounded = total + int(lhs[i, t]) * int(weight[t, j])
                        overflow += int(not -(1 << 31) <= unbounded < (1 << 31))
                        total = wrap(unbounded, 32)
                    result[i, j] = total
            outputs.append(result)
        state = np.array(
            [
                [
                    wrap(wrap(int(outputs[0][i, j]) + int(outputs[1][i, j]), 32) + int(old_position), bits)
                    for j in range(k)
                ]
                for i in range(m)
            ],
            dtype,
        )
        position = np.array(wrap(int(old_position) + 1, 32), "<i4")
        expected.append(tuple(x.copy() for x in (*outputs, state, position, data, w0, w1, before, old_position)))
        trace.append((lhs.copy(), w0.copy(), w1.copy(), *(x.copy() for x in outputs)))
        witnesses.append(overflow)
    return (w0, w1, data, initial, np.array(7, "<i4")), expected, trace, witnesses


@pytest.mark.parametrize("case", CASES)
def test_shared_lhs_fixture_has_adjacent_complete_rectangular_reductions(case):
    module = source(case)
    found = []
    for operation in module.walk():
        try:
            recognize_signed_integer_matmul(operation, (f"i{8 * case.width}",) * 2 + ("i32",))
        except ValueError:
            continue
        found.append(operation)
    assert len(found) == 2
    assert found[0].next_op is found[1]
    assert found[0].inputs[0] is found[1].inputs[0]
    assert found[0].inputs[1] is not found[1].inputs[1]
    assert tuple(found[i].results[0].type.get_shape()[1] for i in range(2)) == case.shape[2:]
    assert ir_text(module)
    arguments, epochs, trace, overflow = reference(case)
    assert len(arguments) == 5 and len(epochs) == len(trace) == 3
    assert all(len(outputs) == len(OUTPUT_NAMES) == 9 for outputs in epochs)
    for index, (outputs, primitives) in enumerate(zip(epochs, trace, strict=True)):
        for weight, actual in zip(primitives[1:3], outputs[:2], strict=True):
            widened = primitives[0].astype("i8") @ weight.astype("i8")
            np.testing.assert_array_equal(actual, widened.astype("<i4"))
        np.testing.assert_array_equal(outputs[7], arguments[3] if index == 0 else epochs[index - 1][2])
        assert int(outputs[8]) == 7 + index
    assert overflow[0] > 0 if case.width == 2 else not any(overflow)


@pytest.mark.parametrize("case", CASES)
def test_fixed_v2_contract_materializes_each_standalone_width(tmp_path, case):
    from dataclasses import replace

    from test_dense_source import compile_source
    from test_shared_lhs_provider import OriginalGroupBindings, machine_source, original_model

    from merlin.sched.ir import Ptr, instances

    selected, storages, adapter = original_model(tmp_path, case)
    problems = tuple(selected.dense.problem(case.shape[0], case.shape[1], n) for n in case.shape[2:])
    transport = OriginalGroupBindings(problems, storages)
    adapter = replace(adapter, encode=transport.single_encode, project=transport.single_project)
    module = source(case)
    operations = [op for op in module.walk() if op.name == "linalg.generic" and len(op.iterator_types) == 3]
    assert len(operations) == 2
    for operation, problem in zip(operations, problems, strict=True):
        result = compile_source((selected, problem, storages[0], adapter), module, operation)
        assert result is not None
        assert result.compiled.kernel.args[1].shape == (problem.k, problem.n)
        seen = set()
        for call, _ in instances(result.compiled.kernel):
            pointers = [value for _, value in call.args if isinstance(value, Ptr)]
            assert len(pointers) == int(call.instr in ("lhs_load", "weight_load", "store"))
            if pointers:
                seen.add(pointers[0].tensor)
        assert seen == {"lhs", "rhs", "output"}
        (tmp_path / f"provider-{problem.n}.c").write_text(
            machine_source(
                result.compiled.kernel,
                (problem,),
                (storages[0],),
                "neutral_probe",
                "neutral_bridge",
                index_bits=result.compiled.index_bits,
            )
        )


@pytest.mark.parametrize("case", CASES)
def test_ordinary_shared_lhs_group_runs_complete_native_model(tmp_path, monkeypatch, case):
    import ctypes
    import json
    import time
    from copy import deepcopy
    from pathlib import Path

    from test_dense_source_catalog import compilation_memory
    from test_shared_lhs_provider import SharedLHSProvider, check_actual_trace, original_model

    from merlin.common import invocation_record as invocation
    from merlin.common.digest import sha256_file
    from merlin.common.paths import merlin_dir
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.llvmlower.lower import lower_model_file
    from merlin.model_check.full_build import elf_sections
    from merlin.runtime.host_provider import revalidate_host_provider

    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("ordinary native compiler and LLVM symbol inspector required")
    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    monkeypatch.setenv("MERLIN_SPLIT_FRESH_DEALLOCS", "0")
    monkeypatch.setenv("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "0")
    arguments, expected, trace, overflow = reference(case)
    source_path = tmp_path / "model.mlir"
    source_path.write_text(ir_text(source(case)))
    inputs_path = tmp_path / "inputs.npz"
    np.savez(inputs_path, **{f"in{i}": a for i, a in enumerate(arguments)})
    facts = tmp_path / "facts"
    facts.mkdir()
    model = original_model(facts, case)
    originals = (
        source_path,
        inputs_path,
        *(merlin_dir() / "tests/infra" / name for name in ("test_shared_lhs_source.py", "test_shared_lhs_provider.py")),
        model[0].facts_path,
        model[0].software_path,
    )
    pins = {str(p): sha256_file(p) for p in originals}
    retained_files = dict(pins)
    completed_arms = []

    def retain(path):
        assert path.is_file() and not path.is_symlink()
        key, digest = str(path.resolve()), sha256_file(path)
        assert key not in retained_files or retained_files[key] == digest
        retained_files[key] = digest

    observations = []
    for arm, grouped in (("baseline", False), ("grouped", True)):
        provider = SharedLHSProvider(tmp_path / (arm + "_provider"), model, trace, grouped=grouped)
        selection = provider.selection()
        started = time.perf_counter()
        with compilation_memory() as memory:
            result = lower_model_file(source_path, tmp_path / arm, shared_source_lowerings=(selection,))
        compilation_seconds = time.perf_counter() - started
        record = result.stats["shared_source_lowering"]
        revalidate_host_provider(record["host_provider"])
        recipe_path = Path(record["compilation_recipe"]["path"])
        recipe_sha = record["compilation_recipe"]["sha256"]
        verify_completed_recipe(recipe_path, executable=result.host_so, expected_recipe_sha256=recipe_sha)
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        assert native.image_sha256 == sha256_file(result.host_so)
        functions = []
        for item in provider.records:
            symbol = item["symbol"]
            size, query, count = (
                getattr(native.lib, symbol + suffix) for suffix in ("_trace_size", "_trace", "_observe_count")
            )
            size.restype = count.restype = ctypes.c_uint64
            query.restype = ctypes.c_int64
            query.argtypes, count.argtypes = [ctypes.c_int] * 2, [ctypes.c_int]
            assert size() == 0 and all(count(i) == 0 for i in range(8))
            functions.append((size, query, count))
        state, position = arguments[3].copy(), arguments[4].copy()
        history, events, actual_overflow_witnesses = [], [], []
        for epoch, want in enumerate(expected):
            actual_inputs = (*arguments[:3], state.copy(), position.copy())
            buffers, views = [], []
            for value in (*actual_inputs, *[np.full_like(value, 71) for value in want]):
                owner = np.full(value.size + 32, 91, value.dtype)
                view = owner[16:-16].reshape(value.shape)
                view[...] = value
                buffers.append(owner)
                views.append(view)
            original_bytes = [a.tobytes() for a in buffers[:5]]
            starts = [(size(), [count(i) for i in range(8)]) for size, _, count in functions]
            native([(a.ctypes.data, a.shape) for a in views])
            outputs = views[5:]
            for actual, reference_output in zip(outputs, want, strict=True):
                np.testing.assert_array_equal(actual, reference_output)
            assert [a.tobytes() for a in buffers[:5]] == original_bytes
            assert all(np.all(a[:16] == 91) and np.all(a[-16:] == 91) for a in buffers)
            assert outputs[7].tobytes() == actual_inputs[3].tobytes()
            assert outputs[8].tobytes() == actual_inputs[4].tobytes()
            if case.width == 2 and epoch == 0:
                total = sum(
                    wrap(int(actual_inputs[2][0, t]) + int(actual_inputs[3][0, t]), 16) * int(actual_inputs[0][t, 0])
                    for t in range(case.shape[1])
                )
                assert not -(1 << 31) <= total < (1 << 31)
                assert int(outputs[0][0, 0]) == wrap(total, 32)
                actual_overflow_witnesses.append(
                    {"epoch": epoch, "unbounded_total": total, "actual_returned_i32": int(outputs[0][0, 0])}
                )
            state, position = outputs[2].copy(), outputs[3].copy()
            raw = tmp_path / f"{arm}-epoch{epoch}.npz"
            np.savez(raw, **{f"owner{i}": value for i, value in enumerate(buffers)})
            history.append(
                {
                    "epoch": epoch,
                    "outputs": [value.tolist() for value in outputs],
                    "raw_buffers": {"path": str(raw), "sha256": sha256_file(raw)},
                    "input_bytes_preserved": True,
                    "all_canaries_preserved": True,
                }
            )
            for binding, (item, functions_row, start) in enumerate(
                zip(provider.records, functions, starts, strict=True)
            ):
                size, query, count = functions_row
                rows = [[query(r, c) for c in range(3 + query(r, 2))] for r in range(start[0], size())]
                effects = check_actual_trace(rows, item["problems"], item["storages"])
                delta = [count(i) - start[1][i] for i in range(8)]
                assert delta[:4] == [
                    effects[k]
                    for k in (
                        "commands",
                        "primitive_lhs_input_payload_bytes",
                        "primitive_rhs_input_payload_bytes",
                        "primitive_output_payload_bytes",
                    )
                ]
                assert delta[6] == 1
                if grouped:
                    assert effects["primitive_lhs_input_payload_bytes"] == case.shape[0] * case.shape[1] * case.width
                    assert all(row[-1] == 1 for row in effects["lhs_coordinate_read_counts"])
                events.append(
                    {
                        "epoch": epoch,
                        "binding": binding,
                        "actual_rows": rows,
                        "effects": effects,
                        "actual_counter_delta": delta,
                    }
                )
        revalidate_host_provider(record["host_provider"])
        verify_completed_recipe(recipe_path, executable=result.host_so, expected_recipe_sha256=recipe_sha)
        assert {str(p): sha256_file(p) for p in originals} == pins
        sections = elf_sections(result.host_so)
        source_receipt = Path(record["preparation"]["path"]).parent / "source_dense.json"
        source_record = json.loads(source_receipt.read_text())
        assert len(source_record["bindings"]) == (1 if grouped else 2)
        observations.append(
            {
                "arm": arm,
                "actual_i32_overflow_witnesses": actual_overflow_witnesses,
                "source": deepcopy(record),
                "source_dense": {"path": str(source_receipt), "sha256": sha256_file(source_receipt)},
                "history": history,
                "events": events,
                "compilation_wall_seconds": compilation_seconds,
                "compilation_memory": memory,
                "compiler_only_peak_rss_bytes": "UNKNOWN",
                "compiler_commands": len(json.loads(recipe_path.read_text())["commands"]),
                "elf": {
                    "path": str(result.host_so),
                    "sha256": sha256_file(result.host_so),
                    "bytes": result.host_so.stat().st_size,
                    "sections": sections,
                },
                "counter_scope": "actual complete-model image neutral primitive/bridge only; CPU glue uninstrumented",
                "cost_scope": (
                    "ordinary lower_model_file inclusive compile/qualification/link wall "
                    "plus sampled process RSS; no execution timing"
                ),
                "execution_wall_seconds": "UNKNOWN",
                "qualified_latency": "UNKNOWN",
                "physical_bus_traffic_bytes": "UNKNOWN",
            }
        )
        # Freeze each completed arm before any later provider callback can
        # modify it. Never refresh an earlier arm's expected identities.
        for directory in (result.workdir, provider.work):
            for path in directory.rglob("*"):
                if path.is_file():
                    retain(path)
        for item in history:
            retain(Path(item["raw_buffers"]["path"]))
        retain(source_receipt)
        completed_arms.append((deepcopy(record["host_provider"]), recipe_path, recipe_sha, result.host_so))
    assert [x["outputs"] for x in observations[0]["history"]] == [x["outputs"] for x in observations[1]["history"]]
    payloads = [sum(e["effects"]["primitive_lhs_input_payload_bytes"] for e in row["events"]) for row in observations]
    assert payloads[1] * 2 <= payloads[0]
    assert len(observations[0]["events"]) == 6 and len(observations[1]["events"]) == 3
    assert overflow[0] > 0 if case.width == 2 else not any(overflow)
    original_contracts = model[0].record()
    for provider_record, recipe_path, recipe_sha, executable in completed_arms:
        revalidate_host_provider(provider_record)
        verify_completed_recipe(recipe_path, executable=executable, expected_recipe_sha256=recipe_sha)
    for receipt in tmp_path.rglob("invocations/*/invocation.json"):
        invocation.verify(receipt)
    # This last closure check is callback-free. Reports below use only the
    # already retained data and cannot retroactively adopt changed artifacts.
    for name, expected_digest in retained_files.items():
        path = Path(name)
        assert path.is_file() and not path.is_symlink() and sha256_file(path) == expected_digest
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "scope": (
                    "ordinary complete native model; two neutral resource/execution cases, "
                    "not actual accelerator qualification"
                ),
                "original_inputs": pins,
                "mac_overflow_witnesses": overflow,
                "original_contracts": original_contracts,
                "retained_files": retained_files,
                "arms": observations,
            },
            indent=2,
        )
    )
