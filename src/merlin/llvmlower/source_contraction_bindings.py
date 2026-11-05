"""Original source/ABI names for independent executed-product checking.

This trusted bridge outside the public SDK grants no target capability. A plan
supplies names only after its complete current-source binding is rechecked.
The final consumer must reconstruct actual loads, products and output effects.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from xdsl.dialects import builtin

from merlin.common import mlir_query as mq
from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


def bind_rank_two_contractions(source, command_buffer, proof):
    """Read exact signed8/signed32 rank-two dots under an original source proof.

    Provenance metadata is ignored only in a detached recognition copy. Every
    arithmetic operation, indexing map, type and source byte remains checked.
    Unsupported batched/mixed numeric or initialization contracts refuse.
    """
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("source contraction binding needs an ordinary original interface")
    text = source.read_bytes()
    if hashlib.sha256(text).hexdigest() != proof.get("interface_sha256"):
        raise ValueError("source contraction binding differs from its original typed proof")
    validation = validate_mixed_program_plan(text, command_buffer)
    if validation.get("ok") is not True:
        raise ValueError("source contraction names have no complete current-source plan binding")
    module = mq.parse(text.decode())
    module.verify()
    functions = list(mq.walk(module, "func.func"))
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
        raise ValueError("source contraction binding needs one closed source function")
    block = functions[0].regions[0].block
    plan = command_buffer["params"]["global_program_plan"]
    names = dict(zip(block.args, plan["entry_bindings"], strict=True))
    operations = list(block.ops)
    for row in plan["source_values"]:
        value = operations[row["op_index"]].results[row["result_index"]]
        if value in names:
            raise ValueError("source contraction value has duplicate compiled names")
        names[value] = row["tensor"]
    records = {row["ordinal"]: row for row in proof["operations"]}
    result = []
    for ordinal, operation in enumerate(operations):
        record = records.get(ordinal, {})
        if record.get("kind") != "multiply_accumulate":
            continue
        detached = operation.clone()
        for nested in detached.walk():
            for name in tuple(nested.attributes):
                if name.startswith("prov."):
                    del nested.attributes[name]
        recognize_signed_integer_matmul(detached, ("i8", "i8", "i32"))
        lhs, rhs, initial = operation.operands
        shapes = tuple(tuple(value.type.get_shape()) for value in (lhs, rhs, initial))
        if any(len(shape) != 2 for shape in shapes):
            raise ValueError("source execution needs a separately checked batched/layout contract")
        (m, k), (rhs_k, n), output = shapes
        if (
            k != rhs_k
            or output != (m, n)
            or record.get("initial_interval") != [0, 0]
            or record.get("reduction_extent") != k
            or record.get("result_bits") != 32
            or any(type(value.type) is not builtin.TensorType for value in (lhs, rhs, initial))
        ):
            raise ValueError("source execution differs from the exact zero-initialized signed dot contract")
        try:
            result.append(
                dict(ordinal=ordinal, lhs=names[lhs], rhs=names[rhs], output=names[operation.results[0]], m=m, k=k, n=n)
            )
        except KeyError as error:
            raise ValueError("source contraction has no complete compiled tensor ownership") from error
    if not result:
        raise ValueError("source execution has no admitted accelerator contraction work")
    return tuple(result)
