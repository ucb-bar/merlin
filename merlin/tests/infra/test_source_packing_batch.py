"""Original source ownership groups actual LLVM proofs without merging budgets.

Only the stock-tool import boundary is substituted. Source discovery, source
normalization, borrowed-call rosters, LLVM verification and proof retention use
production code; no native implementation qualification follows from this test.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_host_packing_storage import fixture
from test_im2col_panel_pack import _module
from test_source_dense_lowering import run, selection
from xdsl.dialects import builtin, func
from xdsl.parser import Parser

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import source_model
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.verify import host_call_alignment


class ProviderReached(RuntimeError):
    pass


def prepared(tmp_path):
    config, _, _ = selection(
        tmp_path,
        hardware=dict(tile=(2, 3, 4), widths=(1, 1, 4), shape=(3, 12, 21), shared=False),
        static_budget=1000,
        max_convolution_packing_bytes=60,
        borrowed_effects=BorrowedBufferEffects("independent-batch-borrowed-ABI-v1", True, True, True),
        packing_allocation_effects=fixture()[-1],
    )
    callbacks = []

    def provider(bindings, context):
        callbacks.append(bindings)
        raise ProviderReached("external implementation boundary")

    config = replace(config, build=provider)
    module = parse_mlir_text(_module(c=2, kh=2, kw=3, n=1, oh=3, ow=7, f=3, sh=2, sw=1, dh=1, dw=2))
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    block = function.body.block
    original_ops = tuple(block.ops)
    returned = original_ops[-1]
    mapping = {value: value for value in block.args}
    for operation in original_ops[:-1]:
        block.insert_op_before(operation.clone(value_mapper=mapping), returned)
    returned.operands = (*returned.operands, mapping[returned.operands[0]])
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [value.type for value in block.args], [value.type for value in returned.operands]
    )
    module.verify()
    result, builder = run(tmp_path, config, module)
    owner = builder.__self__
    assert len(owner.packing_plans) == 2 and len(owner.bindings[0]) == 4
    return result, builder, callbacks


def actual_llvm(calls, plans, *, duplicate_second=False, unsupported_second=False):
    """Emit typed descriptors from the original writer roster, without a plan oracle."""
    scalar, pointer = "i64", "!llvm.ptr"
    declarations = ["llvm.func @heap_alloc(i64)->!llvm.ptr", "llvm.func @heap_release(!llvm.ptr)"]
    body = ["%zero=llvm.mlir.constant(0):i64", "%one=llvm.mlir.constant(1):i64"]
    serial = 0

    def constant(value):
        nonlocal serial
        serial += 1
        name = f"%c{serial}"
        body.append(f"{name}=llvm.mlir.constant({value}):i64")
        return name

    shapes = {}
    for call in calls:
        parsed = [Parser(make_llvm_context(), text).parse_type() for text in call.argument_types]
        shapes[call.wrapper_symbol] = tuple(tuple(typ.get_shape()) for typ in parsed)
        counts = [3 + 2 * len(typ.get_shape()) for typ in parsed]
        types = [
            typ
            for count in (*counts, counts[call.result_argument])
            for typ in (pointer, pointer, *([scalar] * (count - 2)))
        ]
        args = ",".join(f"%a{i}:{typ}" for i, typ in enumerate(types))
        forward, cursor = [], 0
        for index, count in enumerate(counts):
            start = sum(counts) if index == call.result_argument else cursor
            forward.extend(f"%a{i}" for i in range(start, start + count))
            cursor += count
        declarations += [
            f"llvm.func @{call.borrowed_symbol}({','.join(types[: sum(counts)])})",
            f"llvm.func @{call.wrapper_symbol}({args}) {{\n"
            f"llvm.call @{call.borrowed_symbol}({','.join(forward)}):({','.join(types[: sum(counts)])})->()\n"
            "llvm.return\n}",
        ]
    releases = []
    for island, (_, _, uses, _, _) in enumerate(plans):
        # First island's main and tail share a valid sufficiently large root.
        # The second keeps distinct roots, providing independent budgets and
        # a different actual allocation count despite identical source shapes.
        shared = None
        for use_index, (symbol, packing_argument) in enumerate(uses):
            repetitions = 2 if duplicate_second and island == 1 and use_index == 0 else 1
            for repetition in range(repetitions):
                root = f"%root{island}_{use_index}_{repetition}"
                shape = shapes[symbol][packing_argument]
                payload = 1
                for dimension in shape:
                    payload *= dimension
                if island == 0 and shared is not None:
                    root = shared
                else:
                    body.append(f"{root}=llvm.call @heap_alloc({constant(payload)}):(i64)->!llvm.ptr")
                    releases.append(root)
                    if island == 0:
                        shared = root
                operands, operand_types = [], []
                call = next(call for call in calls if call.wrapper_symbol == symbol)
                for argument, descriptor_shape in enumerate((*shapes[symbol], shapes[symbol][call.result_argument])):
                    address = root if argument == packing_argument else "%external"
                    stride, strides = 1, []
                    for dimension in reversed(descriptor_shape):
                        strides.insert(0, stride)
                        stride *= dimension
                    operands += [address, address, "%zero", *(constant(v) for v in (*descriptor_shape, *strides))]
                    operand_types += [pointer, pointer, *([scalar] * (1 + 2 * len(descriptor_shape)))]
                body.append(f"llvm.call @{symbol}({','.join(operands)}):({','.join(operand_types)})->()")
                if unsupported_second and island == 1:
                    body.append(f"llvm.call @unknown({root}):(!llvm.ptr)->()")
    body += [f"llvm.call @heap_release({root}):(!llvm.ptr)->()" for root in releases]
    body.append("llvm.return")
    if unsupported_second:
        declarations.append("llvm.func @unknown(!llvm.ptr)")
    text = (
        "module {\n"
        + "\n".join(declarations)
        + "\nllvm.func @forward(%external:!llvm.ptr) {\n"
        + "\n".join(body)
        + "\n}}"
    )
    module = Parser(make_llvm_context(), text).parse_module()
    module.verify()
    return text


def imported_context(tmp_path, monkeypatch, selected, builder, *, mutation=None):
    owner = builder.__self__
    calls = source_model.source_borrowed_calls(builder, selected).calls
    text = actual_llvm(
        calls, owner.packing_plans, duplicate_second=mutation == "budget", unsupported_second=mutation == "escape"
    )
    llvm_path, object_path = tmp_path / "model.ll", tmp_path / "model.o"
    llvm_path.write_text(text)
    object_path.write_bytes(b"unit-test importer input, no machine-code qualification")
    directory = tmp_path / "provider"
    directory.mkdir()
    imports = []

    def import_entry(artifacts, *, diagnostics, entry, **controls):
        imports.append((entry, controls))
        paths = [Path(row["path"]) for row in artifacts.values()]
        inputs = {str(path.resolve()): sha256_file(path) for path in paths}
        diagnostics.mkdir(parents=True)
        product = diagnostics / "entry.generic.mlir"
        product.write_text(llvm_path.read_text())
        module = Parser(make_llvm_context(), product.read_text()).parse_module()
        if mutation == "evidence":
            llvm_path.write_bytes(llvm_path.read_bytes() + b"\n")
        return module, dict(inputs=inputs, tools={}, products={str(product.resolve()): sha256_file(product)})

    monkeypatch.setattr(host_call_alignment, "import_host_entry", import_entry)
    context = SimpleNamespace(
        prepared_path=selected, model_llvm_path=llvm_path, model_object_path=object_path, workdir=directory
    )
    return context, imports


def test_original_source_owner_batches_two_islands_and_adopts_exact_evidence_once(tmp_path, monkeypatch):
    selected, builder, callbacks = prepared(tmp_path)
    owner = builder.__self__
    context, imports = imported_context(tmp_path, monkeypatch, selected, builder)
    adopted = []
    original_observe = source_model._SourceModelBuild.observe_native

    def observe(self, paths, remember):
        adopted.append(paths)
        return original_observe(self, paths, remember)

    monkeypatch.setattr(source_model._SourceModelBuild, "observe_native", observe)
    with pytest.raises(ProviderReached):
        builder(context)
    assert len(imports) == 1 and len(adopted) == 1 and len(callbacks) == 1
    links = [context.workdir / f"packing_storage_{i}" / "source_packing.json" for i in range(2)]
    reports = [json.loads(path.read_text()) for path in links]
    assert [report["native"]["logical_payload_bytes"] for report in reports] == [36, 48]
    assert [report["native"]["batch_request_index"] for report in reports] == [0, 1]
    assert [report["source_payload_bytes"] for report in reports] == [48, 48]
    assert reports[0]["native"]["report"] == reports[1]["native"]["report"]
    pins = dict(owner.pins)
    for index, (path, report) in enumerate(zip(links, reports, strict=True)):
        assert pins[path.resolve()] == sha256_file(path)
        assert report["source_identity"] == owner.packing_plans[index][-1]
        for filename, digest in report["native"]["pins"].items():
            assert pins[Path(filename)] == digest == sha256_file(Path(filename))
    owner.revalidate()


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("budget", "island packing payload"),
        ("escape", "unknown packing pointer use"),
        ("evidence", "host import source"),
    ],
)
def test_second_island_failure_or_changed_evidence_blocks_provider_callback(
    tmp_path, monkeypatch, mutation, diagnostic
):
    selected, builder, callbacks = prepared(tmp_path)
    context, imports = imported_context(tmp_path, monkeypatch, selected, builder, mutation=mutation)
    with pytest.raises(ValueError, match=diagnostic):
        builder(context)
    assert len(imports) == 1 and not callbacks
    assert not list(context.workdir.glob("packing_storage_*/source_packing.json"))
