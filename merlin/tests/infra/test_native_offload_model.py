"""Source-plan routing changes actual native calls and preserves complete state.

The endpoint is a qualified neutral host execution model. No physical device or
timing authority follows from its hardware-shaped resource declarations.
"""

import ctypes
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_dense_source import contraction
from test_dense_source_catalog import compilation_memory
from test_dense_source_trace import check_trace
from test_native_source_model import NativeDenseProvider
from test_selected_dense_contracts import inputs
from test_source_model_api import reviewed_selection

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.lower import lower_model_file
from merlin.model_check.full_build import elf_sections
from merlin.runtime.host_provider import revalidate_host_provider
from merlin.transforms.certificates import ir_text

CASES = [((3, 5), (2, 3, 4), 1, False), ((4, 7), (3, 2, 5), 2, True)]


def source(m, k, dtype):
    narrow = f"tensor<{m}x{k}x{dtype}>"
    wide = f"tensor<{m}x{k}xi32>"
    weight = f"tensor<{k}x{k}x{dtype}>"
    first = contraction((m, k, k), (dtype, dtype, "i32"), lhs="mixed", rhs="w0", output="first")
    second = contraction((m, k, k), (dtype, dtype, "i32"), lhs="activation", rhs="w1", output="second")
    module = mq.parse(f"""module {{
func.func @forward(%w0:{weight},%w1:{weight},%input:{narrow},%state:{narrow},%position:tensor<i32>)
 ->({wide},{wide},{narrow},tensor<i32>,{narrow},{weight},{weight}) {{
 %z=arith.constant 0:i32
 %one=arith.constant 1:i32
 %empty=tensor.empty():{narrow}
 %mixed=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%input,%state:{narrow},{narrow}) outs(%empty:{narrow}) {{
 ^bb0(%x:{dtype},%s:{dtype},%unused:{dtype}):
 %sum=arith.addi %x,%s:{dtype}
 linalg.yield %sum:{dtype}
 }}->{narrow}
 {first}
 %ae=tensor.empty():{narrow}
 %activation=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}} ins(%first:{wide}) outs(%ae:{narrow}) {{
 ^bb0(%x:i32,%unused:{dtype}):
 %positive=arith.cmpi sgt,%x,%z:i32
 %clamp=arith.select %positive,%x,%z:i32
 %value=arith.trunci %clamp:i32 to {dtype}
 linalg.yield %value:{dtype}
 }}->{narrow}
 {second}
 %se=tensor.empty():{narrow}
 %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->()>,
 affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
 ins(%second,%position:{wide},tensor<i32>) outs(%se:{narrow}) {{
 ^bb0(%x:i32,%p:i32,%unused:{dtype}):
 %sum=arith.addi %x,%p:i32
 %value=arith.trunci %sum:i32 to {dtype}
 linalg.yield %value:{dtype}
 }}->{narrow}
 %pv=tensor.extract %position[]:tensor<i32>
 %pn=arith.addi %pv,%one:i32
 %nextpos=tensor.insert %pn into %position[]:tensor<i32>
 return %first,%second,%next,%nextpos,%input,%w0,%w1:
 {wide},{wide},{narrow},tensor<i32>,{narrow},{weight},{weight}
}} }}""")
    for operation in module.walk():
        operation.attributes.pop("prov.region_id", None)
    module.verify()
    return module


def reference(m, k, width):
    dtype = np.dtype("i1" if width == 1 else "<i2")
    rng = np.random.default_rng(761)
    limit = 1 << (8 * width - 1)
    w0, w1 = [rng.integers(-limit, limit, (k, k)).astype(dtype) for _ in range(2)]
    data, initial = [rng.integers(-limit, limit, (m, k)).astype(dtype) for _ in range(2)]

    def product(a, b):
        result = np.empty((m, k), "<i4")
        for i in range(m):
            for j in range(k):
                total = 0
                for t in range(k):
                    total = (total + int(a[i, t]) * int(b[t, j])) % (1 << 32)
                result[i, j] = total - (1 << 32) if total >= 1 << 31 else total
        return result

    state, position = initial.copy(), 7
    expected, trace = [], []
    for _ in range(3):
        mixed = np.add(data, state, dtype=dtype)
        first = product(mixed, w0)
        activation = np.maximum(first, 0).astype(dtype)
        second = product(activation, w1)
        state = (second.astype("i8") + position).astype(dtype)
        position += 1
        trace.extend(((0, mixed.copy(), w0.copy(), first.copy()), (1, activation.copy(), w1.copy(), second.copy())))
        expected.append(tuple(a.copy() for a in (first, second, state, np.array(position, "<i4"), data, w0, w1)))
    return (w0, w1, data, initial, np.array(7, "<i4")), expected, trace


class SelectedProvider(NativeDenseProvider):
    def __init__(self, *args, selected_indices, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_indices = selected_indices
        self.original_trace = self.trace

    def prepare(self, bindings, directory):
        assert len(bindings) == len(self.selected_indices)
        remap = {old: new for new, old in enumerate(self.selected_indices)}
        self.trace = [(remap[index], *rest) for index, *rest in self.original_trace if index in remap]
        super().prepare(bindings, directory)


def planned(selection, module, selected_indices):
    from merlin.llvmlower.offload_plan import bind_source, realize_offload_plan
    from merlin.llvmlower.source_dense import offload_context
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, GlobalPlan

    context = offload_context(selection)
    bound = bind_source(module, entry="forward", context=context)
    contractions = []
    for index, operation in enumerate(bound.operations):
        try:
            recognize_signed_integer_matmul(operation, selection.selected_contracts.dense.dtypes)
        except ValueError:
            continue
        contractions.append(index)
    assert len(contractions) == 2
    # The two exact source operations have identical dimensions. Selection
    # follows complete source identity/ordinal, never a shape-only predicate.
    alternatives = tuple(
        replace(
            bound.cpu_alternatives[index],
            id=f"endpoint_{index}",
            implementation=selection.adapter.qualification_identity,
            placement=selection.selected_contracts.target,
        )
        for index in contractions
    )
    by_node = {a.nodes[0]: a for i, a in enumerate(alternatives) if i in selected_indices}
    plan = GlobalPlan(
        tuple(by_node.get(i, a) for i, a in enumerate(bound.cpu_alternatives)),
        (),
        CycleInterval.unknown("execution experiment supplies no timing model"),
    )
    result = realize_offload_plan(bound, plan, device_alternatives=alternatives, context=context)
    return replace(selection, offload_plan=result), result


def observed(native, provider):
    effects, counters = [], []
    for item in provider.records:
        symbol = item["symbol"]
        count = getattr(native.lib, symbol + "_observe_count")
        count.argtypes = [ctypes.c_int]
        count.restype = ctypes.c_uint64
        counters.append([count(i) for i in range(6)])
        assert count(5) == 3
        size = getattr(native.lib, symbol + "_trace_size")
        size.restype = ctypes.c_uint64
        query = getattr(native.lib, symbol + "_trace")
        query.argtypes = [ctypes.c_int, ctypes.c_int]
        query.restype = ctypes.c_int64
        rows = [[query(i, j) for j in range(query(i, 1) + 2)] for i in range(size())]
        assert len(rows) % 3 == 0
        per_call = len(rows) // 3
        for step in range(3):
            effects.append(
                check_trace(
                    rows[step * per_call : (step + 1) * per_call],
                    item["problem"],
                    provider.storage,
                    item["result"].compiled.kernel.args,
                )
            )
    return effects, counters


@pytest.mark.parametrize("shape,tile,width,shared", CASES)
def test_complete_native_source_plan_selects_exact_same_shape_operations(
    tmp_path, monkeypatch, shape, tile, width, shared
):
    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("native model compiler and inspector required")
    from merlin.llvmlower.source_dense import offload_context

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
    observations = []
    for arm, indices in (("ordinary", (0, 1)), ("mixed", (1,)), ("cpu", ())):
        provider = SelectedProvider(
            tmp_path / (arm + "_provider"), model, trace, selected_indices=indices, target_console=False
        )
        selection = reviewed_selection(provider)
        plan = None
        if arm != "ordinary":
            selection, plan = planned(selection, module, indices)
            plan.revalidate(context=offload_context(selection))
        started = time.perf_counter()
        with compilation_memory() as memory:
            result = lower_model_file(source_path, tmp_path / arm, shared_source_lowerings=(selection,))
        seconds = time.perf_counter() - started
        record = result.stats["shared_source_lowering"]
        revalidate_host_provider(record["host_provider"])
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        state, position = args[3].copy(), args[4].copy()
        history = []
        for step in range(3):
            # Private writable state/position are freshly supplied each call;
            # immutable weights/input have identical bytes in every arm.
            actual_args = [a.copy() for a in args[:3]] + [state.copy(), position.copy()]
            outputs = [np.full_like(a, 91) for a in expected[step]]
            native([(a.ctypes.data, a.shape) for a in (*actual_args, *outputs)])
            for got, want in zip(outputs, expected[step], strict=True):
                np.testing.assert_array_equal(got, want)
            for got, want in zip(actual_args[:3], args[:3], strict=True):
                np.testing.assert_array_equal(got, want)
            state, position = outputs[2].copy(), outputs[3].copy()
            history.append([a.tolist() for a in outputs])
        revalidate_host_provider(record["host_provider"])
        assert len(provider.records) == len(indices)
        effects, counters = observed(native, provider)
        assert len(effects) == 3 * len(indices)
        receipt_path = Path(record["preparation"]["path"]).parent / "source_dense.json"
        receipt = json.loads(receipt_path.read_text())
        assert len(receipt["bindings"]) == len(indices)
        if plan is not None:
            assert receipt["offload_placement"][0]["actual_route"] == "candidate"
            assert receipt["offload_placement"][0]["selected_dense_ordinals"] == list(plan.selected_ordinals)
            plan.revalidate(context=offload_context(selection))
        recipe = json.loads(Path(record["compilation_recipe"]["path"]).read_text())
        sections = elf_sections(result.host_so)
        total_counters = [sum(row[i] for row in counters) for i in range(6)]
        assert total_counters[:3] == [
            sum(row[key] for row in effects) for key in ("commands", "requested_read_bytes", "requested_write_bytes")
        ]
        observations.append(
            dict(
                arm=arm,
                compile_seconds=seconds,
                compilation_memory=memory,
                compiler_commands=len(recipe["commands"]),
                selected_indices=indices,
                native_provider_calls=len(effects),
                history=history,
                effects=effects,
                actual_primitive_commands=total_counters[0],
                requested_primitive_read_bytes=total_counters[1],
                requested_primitive_write_bytes=total_counters[2],
                host_packing_read_bytes=total_counters[3],
                host_packing_write_bytes=total_counters[4],
                actual_wrapper_dispatches=total_counters[5],
                counter_scope="instrumented neutral primitive/bridge only; CPU glue uninstrumented",
                source=record,
                elf_sha256=sha256_file(result.host_so),
                elf_bytes=result.host_so.stat().st_size,
                elf_sections=sections,
                elf_text_bytes=sections.get(".text", 0),
                physical_traffic_bytes=None,
                host_endpoint_component_seconds=None,
                plan=None if plan is None else plan.record(),
                timing_ns=None,
                scope="actual complete generated native entry; neutral host endpoint only",
            )
        )
    assert observations[0]["history"] == observations[1]["history"] == observations[2]["history"]
    assert [row["native_provider_calls"] for row in observations] == [6, 3, 0]
    (tmp_path / "pair.json").write_text(json.dumps(observations, indent=2))
