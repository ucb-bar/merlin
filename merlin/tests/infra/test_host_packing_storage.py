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
