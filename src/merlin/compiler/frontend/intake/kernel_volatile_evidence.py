"""Independent source and executable checks for selected private volatile reads."""

import json

from xdsl.dialects import arith, builtin, func, llvm
from xdsl.parser import Parser

PROFILE = "private-invocation-volatile-loads-v1"


def checked_source_volatile(op, declared, source_id):
    """Check the actual primitive rather than an observation annotation."""
    flag = op.properties.get("isVolatile", op.attributes.get("isVolatile"))
    if (
        not isinstance(flag, builtin.IntegerAttr)
        or flag.type != builtin.i1
        or flag.value.data != -1
        or len(op.results) != 1
    ):
        raise ValueError("volatile read needs its actual pinned boolean and one typed result")
    expected = {
        "schema": "t2m.volatile-read.v1",
        "source_id": source_id,
        "result_type": str(op.results[0].type),
        "scope": PROFILE,
        "observation": "one-actual-volatile-read-per-active-logical-lane",
        "external_participants": "excluded-by-private-buffer-ABI",
        "physical_mapping": "undecided",
    }
    if declared != expected:
        raise ValueError("volatile source contract differs from its actual typed read")


def checked_volatile_memory(admission, memory, records, source, stage):
    from merlin.compiler.frontend.intake.kernel_barrier_evidence import _ids, _name
    from merlin.compiler.frontend.linalg_mlir import make_context

    combined = admission.get("schema") == "t2m.memory-admission.v6"
    field = "volatile_profile" if combined else "effect_profile"
    if (
        admission.get(field) != PROFILE
        or memory.get(field) != PROFILE
        or not isinstance(admission.get("volatile_adaptations"), list)
        or memory.get("volatile_adaptations") != admission["volatile_adaptations"]
        or stage.get("schema") != ("t2m.source-word-stages.v4" if combined else "t2m.source-word-stages.v3")
    ):
        raise ValueError("volatile memory lacks its explicit source/effect/stage profile")
    declared = {r["id"]: r["volatile_read"] for r in records if "volatile_read" in r}
    module = Parser(make_context(), source.decode()).parse_module()
    actual = {}
    for op in module.walk():
        flag = op.properties.get("isVolatile")
        if _name(op) != "tt.load" or not isinstance(flag, builtin.IntegerAttr) or not flag.value.data:
            continue
        ids = set(_ids(op.location)) & set(declared)
        if len(ids) != 1 or flag.type != builtin.i1 or flag.value.data != -1 or len(op.results) != 1:
            raise ValueError("volatile typed source has an unaccounted observation")
        sid = next(iter(ids))
        checked_source_volatile(op, declared[sid], sid)
        if sid in actual:
            raise ValueError("volatile source contract differs from its actual typed read")
        type_ = op.results[0].type
        actual[sid] = str(type_.element_type if isinstance(type_, builtin.TensorType) else type_)
    seen = set()
    for row in admission["volatile_adaptations"]:
        ids = row.get("source_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != 1
            or ids[0] not in actual
            or ids[0] in seen
            or set(row) != {"source_ids", "buffer", "dtype", "descriptor_rank", "representation"}
            or type(row.get("descriptor_rank")) is not int
            or row["descriptor_rank"] < 0
            or row.get("dtype") != actual[ids[0]]
            or row.get("representation") != "standard-LLVM-volatile-descriptor-read"
            or not any(
                a.get("effect") == "read" and a.get("buffer") == row.get("buffer") and ids[0] in a.get("source_ids", [])
                for a in admission["accesses"]
            )
        ):
            raise ValueError("volatile observations differ from guarded source memory accesses")
        seen.update(ids)
    if seen != set(declared) or set(actual) != set(declared):
        raise ValueError("volatile source/effect inventory lost an observation")
    # Actual compiler-only anchors must be complete identities, never opaque
    # imports or assertions supplying facts about source numeric values.
    compute = Parser(make_context(), stage["compute"]["input"]).parse_module()
    helpers = list(compute.ops)[1:]
    if len(helpers) != len(declared):
        raise ValueError("volatile source is missing its intermediate identity anchors")
    anchor_ids, symbols = set(), {}
    for helper in helpers:
        if (
            not isinstance(helper, func.FuncOp)
            or helper.sym_visibility != builtin.StringAttr("private")
            or len(helper.body.blocks) != 1
            or len(helper.body.block.args) != 1
            or len(list(helper.body.block.ops)) != 1
            or not isinstance(helper.body.block.last_op, func.ReturnOp)
            or tuple(helper.body.block.last_op.arguments) != tuple(helper.body.block.args)
            or list(helper.function_type.outputs) != [helper.body.block.args[0].type]
        ):
            raise ValueError("volatile source anchor is not a complete private identity")
        identity = helper.attributes.get("prov.volatile_anchor")
        if not isinstance(identity, builtin.StringAttr) or identity.data not in declared or identity.data in anchor_ids:
            raise ValueError("volatile source anchor differs from its distinct original read")
        anchor_ids.add(identity.data)
        symbols[helper.sym_name.data] = helper
    calls = [op for op in next(iter(compute.ops)).walk() if isinstance(op, func.CallOp)]
    if len(calls) != len(helpers):
        raise ValueError("volatile source anchors have missing or duplicate calls")
    used = set()
    for call in calls:
        helper = symbols.get(call.callee.root_reference.data)
        if (
            helper is None
            or helper in used
            or call.attributes.get("prov.volatile_anchor") != helper.attributes["prov.volatile_anchor"]
            or [v.type for v in call.operands] != list(helper.function_type.inputs)
            or [v.type for v in call.results] != list(helper.function_type.outputs)
        ):
            raise ValueError("volatile source anchor call differs from its checked identity")
        used.add(helper)


def checked_volatile_payload(entry, admission):
    expected = admission["volatile_adaptations"]
    phases = [op for op in entry.walk() if "prov.volatile_read" in op.attributes]
    if len(phases) != len(expected):
        raise ValueError("volatile executable payload lost an observation")
    leaves = set()
    for phase, row in zip(phases, expected, strict=True):
        marker = phase.properties.get("no_inline", phase.attributes.get("no_inline"))
        if (
            phase.name != "scf.execute_region"
            or len(phase.regions) != 1
            or len(phase.regions[0].blocks) != 1
            or phase.regions[0].block.args
            or marker != builtin.UnitAttr()
            or json.loads(phase.attributes["prov.volatile_read"].data) != row
        ):
            raise ValueError("volatile read lost its exactly-once typed region")
        ops = list(phase.regions[0].block.ops)
        rank = row["descriptor_rank"]
        names = [
            "memref.extract_strided_metadata",
            *[n for _ in range(rank) for n in ("arith.muli", "arith.addi")],
            "arith.constant",
            "arith.muli",
            "memref.extract_aligned_pointer_as_index",
            "arith.addi",
            "arith.index_cast",
            "llvm.inttoptr",
            "llvm.load",
            "scf.yield",
        ]
        if [op.name for op in ops] != names:
            raise ValueError("volatile read changed its standard descriptor recipe")
        for op in ops:
            if op.name in ("arith.addi", "arith.muli"):
                flags = op.properties.get("overflowFlags")
                if not isinstance(flags, arith.IntegerOverflowAttr) or flags.data:
                    raise ValueError("volatile descriptor arithmetic cannot invent overflow promises")
        meta = ops[0]
        source = meta.operands[0]
        if not isinstance(source.type, builtin.MemRefType):
            raise ValueError("volatile read requires actual ranked memory metadata")
        element = source.type.element_type
        if (
            len(source.type.get_shape()) != rank
            or str(element) != row["dtype"]
            or source.type.memory_space != builtin.NoneAttr()
        ):
            raise ValueError("volatile descriptor differs from actual memory storage")
        offset = meta.results[1]
        for i in range(rank):
            mul, add = ops[1 + i * 2 : 3 + i * 2]
            if mul.operands[1] != meta.results[2 + rank + i] or tuple(add.operands) != (offset, mul.results[0]):
                raise ValueError("volatile read lost an actual descriptor offset/stride")
            offset = add.results[0]
        size, displacement, base, address, word, pointer, load, yield_ = ops[1 + rank * 2 :]
        if (
            size.properties.get("value") != builtin.IntegerAttr(max(1, element.bitwidth // 8), builtin.IndexType())
            or tuple(displacement.operands) != (offset, size.results[0])
            or base.operands[0] != source
            or tuple(address.operands) != (base.results[0], displacement.results[0])
            or word.operands[0] != address.results[0]
            or word.results[0].type != builtin.i64
            or pointer.operands[0] != word.results[0]
            or load.operands[0] != pointer.results[0]
            or pointer.results[0].type != llvm.LLVMPointerType()
            or load.properties.get("volatile_") != builtin.UnitAttr()
            or load.properties.get("ordering") != builtin.IntegerAttr(0, builtin.i64)
            or load.properties.get("alignment") is not None
            or tuple(yield_.operands) != tuple(load.results)
            or tuple(phase.result_types) != (element,)
            or len(tuple(pointer.results[0].uses)) != 1
            or len(tuple(word.results[0].uses)) != 1
        ):
            raise ValueError("volatile read lost its actual standard LLVM load semantics")
        leaves.update((pointer, load))
    if {op for op in entry.walk() if op.name.startswith("llvm.")} != leaves:
        raise ValueError("volatile payload has an escaping or unaccounted low-level memory operation")
