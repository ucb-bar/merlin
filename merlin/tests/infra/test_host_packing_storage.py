"""Finite final-LLVM backing proofs; no runtime or numerical qualification inferred."""

from dataclasses import replace

import pytest
from xdsl.parser import Parser

from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCall, BorrowedBufferEffects
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.verify.host_packing_storage import (
    PackingAllocationEffects,
    PackingStorageUse,
    verify_packing_storage,
)


def fixture(*, bits=8, count=4, pointer_bits=64):
    effects = PackingAllocationEffects(
        "heap_alloc",
        "heap_release",
        "independent-neutral-original-heap-ABI-v1",
        pointer_bits,
        fresh_private_allocations=True,
        successful_allocations=True,
        releases_only_via_deallocator=True,
    )
    call = BorrowedBufferCall(
        "writer",
        "borrowed",
        (f"memref<{count}xi{bits}>", f"memref<{count}xi{bits}>"),
        1,
        BorrowedBufferEffects("independent-neutral-noescape-sync-ABI-v1", True, True, True),
    )
    scalar = f"i{pointer_bits}"
    descriptor = ["!llvm.ptr", "!llvm.ptr", scalar, scalar, scalar]
    types = descriptor * 3
    arguments = ",".join(f"%a{i}:{typ}" for i, typ in enumerate(types))
    input_types = ",".join(descriptor * 2)
    forwarded = ",".join(f"%a{i}" for i in (*range(5), *range(10, 15)))
    forwarded_types = ",".join(descriptor * 2)
    request = count * (bits // 8) + 8
    pointer_args = "%raw,%aligned,%zero,%count,%one"
    out_args = "%output,%output,%zero,%count,%one"
    source = f"""module {{
llvm.func @heap_alloc({scalar}) -> !llvm.ptr
llvm.func @heap_release(!llvm.ptr)
llvm.func @borrowed({input_types})
llvm.func @writer({arguments}) {{
 llvm.call @borrowed({forwarded}):({forwarded_types})->()
 llvm.return
}}
llvm.func @model(%output:!llvm.ptr) {{
 %size=llvm.mlir.constant({request}):{scalar}
 %zero=llvm.mlir.constant(0):{scalar}
 %one=llvm.mlir.constant(1):{scalar}
 %count=llvm.mlir.constant({count}):{scalar}
 %seven=llvm.mlir.constant(7):{scalar}
 %eight=llvm.mlir.constant(8):{scalar}
 %raw=llvm.call @heap_alloc(%size):({scalar})->!llvm.ptr
 %integer=llvm.ptrtoint %raw:!llvm.ptr to {scalar}
 %top=llvm.add %integer,%seven:{scalar}
 %remainder=llvm.urem %top,%eight:{scalar}
 %rounded=llvm.sub %top,%remainder:{scalar}
 %aligned=llvm.inttoptr %rounded:{scalar} to !llvm.ptr
 llvm.call @writer({pointer_args},{out_args},{out_args}):({",".join(types)})->()
 llvm.call @heap_release(%raw):(!llvm.ptr)->()
 llvm.return
}}
}}"""
    return source, (call,), (PackingStorageUse("writer", 0),), effects


def run(text, calls, uses, effects, *, budget=1024, **controls):
    module = Parser(make_llvm_context(), text).parse_module()
    module.verify()
    return verify_packing_storage(
        module,
        borrowed_calls=calls,
        uses=uses,
        allocation_effects=effects,
        max_payload_bytes=budget,
        entry="model",
        **controls,
    )


@pytest.mark.parametrize("bits,count,pointer_bits", [(8, 4, 64), (16, 7, 64), (8, 3, 32)])
def test_actual_descriptor_root_extent_alignment_and_completed_lifetime(bits, count, pointer_bits):
    source, calls, uses, effects = fixture(bits=bits, count=count, pointer_bits=pointer_bits)
    result = run(source, calls, uses, effects, budget=count * bits // 8)
    assert result["logical_payload_bytes"] == count * bits // 8
    assert result["allocator_requested_bytes"] == count * bits // 8 + 8
    assert result["alignment_overhead_bytes"] == 8
    (root,) = result["buffers"]
    assert root["allocation_outside_all_cfg_cycles"] and root["released_once"]
    assert root["proven_alignments"] == [8]
    (use,) = result["uses"]
    assert use["shape"] == [count] and use["element_strides"] == [1] and use["offset"] == 0
    assert "total RAM" in result["scope"] and "all host gather access bounds and values" in result["unproved"]


@pytest.mark.parametrize(
    "mutation",
    [
        "offset",
        "shape",
        "stride",
        "request",
        "numerator",
        "base",
        "early_free",
        "double_free",
        "escape",
        "forwarder",
        "output_alias",
        "unknown_call",
        "narrow",
        "allocator_body",
        "cycle",
    ],
)
def test_actual_llvm_mutations_refuse(mutation):
    source, calls, uses, effects = fixture()
    if mutation == "offset":
        source = source.replace("%raw,%aligned,%zero,%count,%one", "%raw,%aligned,%one,%count,%one")
    elif mutation == "shape":
        source = source.replace("%raw,%aligned,%zero,%count,%one", "%raw,%aligned,%zero,%one,%one")
    elif mutation == "stride":
        source = source.replace("%raw,%aligned,%zero,%count,%one", "%raw,%aligned,%zero,%count,%count")
    elif mutation == "request":
        source = source.replace("constant(12)", "constant(4)")
    elif mutation == "numerator":
        source = source.replace("llvm.urem %top,%eight", "llvm.urem %integer,%eight")
    elif mutation == "base":
        source = source.replace("llvm.ptrtoint %raw", "llvm.ptrtoint %output")
    elif mutation in ("early_free", "double_free"):
        source = source.replace(
            " llvm.call @writer(", " llvm.call @heap_release(%raw):(!llvm.ptr)->()\n llvm.call @writer("
        )
        if mutation == "early_free":
            source = source.replace(" llvm.call @heap_release(%raw):(!llvm.ptr)->()\n llvm.return", " llvm.return")
    elif mutation == "escape":
        source = source.replace(
            " llvm.call @writer(", " llvm.store %raw,%output:!llvm.ptr,!llvm.ptr\n llvm.call @writer("
        )
    elif mutation == "forwarder":
        source = source.replace("%a0,%a1,%a2,%a3,%a4,%a10", "%a5,%a1,%a2,%a3,%a4,%a10")
    elif mutation == "output_alias":
        source = source.replace("%output,%output,%zero,%count,%one", "%raw,%aligned,%zero,%count,%one")
    elif mutation == "unknown_call":
        source = source.replace("module {", "module {\nllvm.func @unknown(!llvm.ptr)")
        source = source.replace(
            " llvm.call @writer(", " llvm.call @unknown(%aligned):(!llvm.ptr)->()\n llvm.call @writer("
        )
    elif mutation == "narrow":
        for name in ("integer", "top", "remainder", "rounded"):
            source = source.replace("%" + name + ":i64", "%" + name + ":i32")
        source = source.replace("to i64", "to i32").replace("%seven:i64", "%seven:i32")
        source = source.replace("%eight:i64", "%eight:i32")
        source = source.replace("%seven=llvm.mlir.constant(7):i64", "%seven=llvm.mlir.constant(7):i32")
        source = source.replace("%eight=llvm.mlir.constant(8):i64", "%eight=llvm.mlir.constant(8):i32")
        source = source.replace("llvm.add %integer,%seven:i64", "llvm.add %integer,%seven:i32")
        source = source.replace("llvm.urem %top,%eight:i64", "llvm.urem %top,%eight:i32")
        source = source.replace("llvm.sub %top,%remainder:i64", "llvm.sub %top,%remainder:i32")
    elif mutation == "allocator_body":
        source = source.replace(
            "llvm.func @heap_alloc(i64) -> !llvm.ptr",
            """llvm.func @heap_alloc(%n:i64)->!llvm.ptr {
 %null=llvm.mlir.zero:!llvm.ptr
 llvm.return %null:!llvm.ptr
}""",
        )
    else:
        source = source.replace(" %raw=", " llvm.br ^cycle\n^cycle:\n %raw=")
        source = source.replace(" llvm.call @heap_release(%raw):(!llvm.ptr)->()\n llvm.return", " llvm.br ^cycle")
    with pytest.raises(ValueError, match="final packing storage"):
        run(source, calls, uses, effects)


@pytest.mark.parametrize("controls", [dict(budget=3), dict(max_operations=1), dict(max_alias_values=1)])
def test_all_explicit_budgets_refuse_unestablished_bounds(controls):
    with pytest.raises(ValueError, match="budget"):
        run(*fixture(), **controls)


@pytest.mark.parametrize(
    "field", ["fresh_private_allocations", "successful_allocations", "releases_only_via_deallocator"]
)
def test_qualification_identity_does_not_replace_runtime_premises(field):
    *_, effects = fixture()
    with pytest.raises(ValueError, match="allocator effects"):
        replace(effects, **{field: False})


def two_islands(*, shared):
    source, calls, uses, effects = fixture()
    writer = source[source.index("llvm.func @writer(") : source.index("llvm.func @model(")]
    source = source.replace(
        "llvm.func @model(",
        writer.replace("@writer(", "@writer_second(").replace("@borrowed(", "@borrowed_second(") + "llvm.func @model(",
    )
    declaration = next(line for line in source.splitlines() if line.startswith("llvm.func @borrowed("))
    source = source.replace(declaration, declaration + "\n" + declaration.replace("@borrowed(", "@borrowed_second("))
    first_call = next(line for line in source.splitlines() if "llvm.call @writer(" in line)
    second_call = first_call.replace("@writer(", "@writer_second(")
    if not shared:
        alignment = source[source.index(" %raw=") : source.index(first_call)]
        for name in ("raw", "integer", "top", "remainder", "rounded", "aligned"):
            alignment = alignment.replace("%" + name, "%second_" + name)
            second_call = second_call.replace("%" + name, "%second_" + name)
        source = source.replace(first_call, alignment + first_call)
        source = source.replace(
            " llvm.call @heap_release(%raw)",
            " llvm.call @heap_release(%second_raw):(!llvm.ptr)->()\n llvm.call @heap_release(%raw)",
        )
    source = source.replace(first_call, first_call + "\n" + second_call)
    return (
        source,
        (*calls, replace(calls[0], wrapper_symbol="writer_second", borrowed_symbol="borrowed_second")),
        (
            uses,
            (PackingStorageUse("writer_second", 0),),
        ),
        effects,
    )


@pytest.mark.parametrize("shared", [False, True])
def test_batch_proves_union_once_and_preserves_independent_island_payloads(shared, monkeypatch):
    from merlin.verify import host_packing_storage as checker

    source, calls, islands, effects = two_islands(shared=shared)
    module = Parser(make_llvm_context(), source).parse_module()
    module.verify()
    cfg_preparations = []
    original = checker.prepare_host_cfg

    def prepare(function):
        cfg_preparations.append(function)
        return original(function)

    monkeypatch.setattr(checker, "prepare_host_cfg", prepare)
    requests = tuple(checker.PackingStorageRequest(uses, 4) for uses in islands)
    result = checker.verify_packing_storage_batch(
        module, borrowed_calls=calls, requests=requests, allocation_effects=effects, entry="model"
    )
    assert len(cfg_preparations) == 1
    assert result["combined"]["logical_payload_bytes"] == (4 if shared else 8)
    assert len(result["combined"]["buffers"]) == (1 if shared else 2)
    assert [row["logical_payload_bytes"] for row in result["results"]] == [4, 4]
    assert [row["allocator_requested_bytes"] for row in result["results"]] == [12, 12]
    assert [{row["symbol"] for row in island["uses"]} for island in result["results"]] == [
        {"writer"},
        {"writer_second"},
    ]
    # Spare capacity in one island cannot admit another island exceeding its
    # own original budget, even though the union's aggregate limit would fit.
    with pytest.raises(ValueError, match="island packing payload"):
        checker.verify_packing_storage_batch(
            module,
            borrowed_calls=calls,
            requests=(replace(requests[0], max_payload_bytes=3), replace(requests[1], max_payload_bytes=100)),
            allocation_effects=effects,
            entry="model",
        )


def test_batch_shared_root_does_not_grant_unselected_call_escape():
    from merlin.verify import host_packing_storage as checker

    source, calls, islands, effects = two_islands(shared=True)
    module = Parser(make_llvm_context(), source).parse_module()
    with pytest.raises(ValueError, match="unknown packing pointer use"):
        checker.verify_packing_storage_batch(
            module,
            borrowed_calls=calls,
            requests=(checker.PackingStorageRequest(islands[0], 4),),
            allocation_effects=effects,
            entry="model",
        )


def test_batch_imports_and_retains_one_observation_and_singleton_shape(tmp_path, monkeypatch):
    from merlin.verify import host_call_alignment as importer
    from merlin.verify import host_packing_storage as checker

    source, calls, islands, effects = two_islands(shared=True)
    imports = []

    def import_entry(artifacts, **controls):
        imports.append(controls)
        return Parser(make_llvm_context(), source).parse_module(), dict(inputs={}, tools={}, products={})

    monkeypatch.setattr(importer, "import_host_entry", import_entry)
    result = checker.inspect_packing_storage_batch(
        {},
        borrowed_calls=calls,
        requests=tuple(checker.PackingStorageRequest(uses, 4) for uses in islands),
        allocation_effects=effects,
        entry="model",
        diagnostics=tmp_path,
    )
    assert len(imports) == 1
    assert imports[0]["include_functions"] == ("writer", "writer_second", "heap_alloc", "heap_release")
    assert result["schema"] == "merlin.final_packing_storage_batch.v1"
    report = result["report"]["path"]
    assert result["pins"][report] == result["report"]["sha256"]

    source, calls, uses, effects = fixture()
    singleton = checker.inspect_packing_storage(
        {},
        borrowed_calls=calls,
        uses=uses,
        allocation_effects=effects,
        max_payload_bytes=4,
        entry="model",
        diagnostics=tmp_path,
    )
    assert singleton["schema"] == "merlin.final_packing_storage.v1"
    assert singleton["logical_payload_bytes"] == 4
    assert "results" not in singleton and "combined" not in singleton


@pytest.mark.parametrize("uses,budget", [((), 4), ([], 4), ((PackingStorageUse("writer", 0),), 0)])
def test_batch_requests_are_closed_nonempty_and_budgeted(uses, budget):
    from merlin.verify.host_packing_storage import PackingStorageRequest

    with pytest.raises(ValueError, match="island request"):
        PackingStorageRequest(uses, budget)
