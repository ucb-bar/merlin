"""Extract typed source windows while retaining literal-dependent source proofs.

The result is a source projection, not host admission or numerical qualification.
Reverification reconstructs it from the exact original source; caller proof records
must be reproved on the returned operations, never copied by ordinal.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Operation, Region

from merlin.compiler.ir import mlir_query as mq

# ---- connected_region --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceWindow:
    module: ModuleOp
    inputs: tuple
    outputs: tuple
    operation_pairs: tuple
    start: int
    stop: int


def text(module) -> str:
    # Generic canonical syntax preserves all attributes (ReduceOp's upstream
    # custom printer drops them) and exact dense floating-point payload bits.
    from merlin.compiler.dialects.xdsl_context import text as canonical_text

    return canonical_text(module, generic=True) + "\n"


def extract_window(module, start: int, stop: int) -> SourceWindow:
    """Clone an inclusive single-block window, all live outputs and constant closure.

    External runtime tensors remain explicit typed inputs. Literal tensors, scalar
    constants, splats and empty initializers remain internal, including operands
    captured by nested regions. No other computation is pulled across the cut.
    """
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("source window requires one single-block function")
    original = functions[0]
    if tuple(module.body.block.ops) != (original,):
        raise ValueError("source window requires a self-contained single-function module")
    if set(original.properties) - {"sym_name", "function_type", "sym_visibility"}:
        raise ValueError("source window cannot remap per-argument or unknown function properties")
    block = original.body.blocks[0]
    ops = tuple(block.ops)
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop < len(ops):
        raise ValueError("source window bounds are invalid")
    kept = set(ops[start : stop + 1])
    if any(isinstance(op, ReturnOp) or op.successors for op in kept):
        raise ValueError("source window cannot contain function return or CFG edges")
    pending = list(kept)
    while pending:
        for nested in pending.pop().walk():
            for operand in nested.operands:
                owner = operand.owner
                if (
                    isinstance(owner, Operation)
                    and owner.parent is block
                    and type(owner) in (arith.ConstantOp, tensor.SplatOp, tensor.EmptyOp)
                    and owner not in kept
                ):
                    kept.add(owner)
                    pending.append(owner)
    ordered = [op for op in ops if op in kept]
    inside = {nested for op in ordered for nested in op.walk()}
    boundary, seen = [], set()
    for op in ordered:
        for nested in op.walk():
            for operand in nested.operands:
                owner = operand.owner
                if isinstance(owner, Operation) and owner in inside:
                    continue
                if isinstance(owner, Block) and owner is not block:
                    continue
                if operand not in seen:
                    boundary.append(operand)
                    seen.add(operand)
    # Constants pulled in for source premises are not new observable results.
    outputs = [
        value
        for op in ops[start : stop + 1]
        for value in op.results
        if any(use.operation not in inside for use in value.uses)
    ]
    if not outputs:
        raise ValueError("source window has no externally used outputs")
    target_block = Block(arg_types=[value.type for value in boundary])
    mapping = dict(zip(boundary, target_block.args, strict=True))
    pairs = []
    for op in ordered:
        clone = op.clone(value_mapper=mapping)
        target_block.add_op(clone)
        pairs.extend(zip(op.walk(), clone.walk(), strict=True))
    target_block.add_op(ReturnOp(*(mapping[value] for value in outputs)))
    function = FuncOp(
        "forward",
        FunctionType.from_lists([value.type for value in boundary], [value.type for value in outputs]),
        Region([target_block]),
    )
    function.attributes.update(original.attributes)
    if "sym_visibility" in original.properties:
        function.properties["sym_visibility"] = original.properties["sym_visibility"]
    extracted = ModuleOp([function])
    extracted.attributes.update(module.attributes)
    extracted.verify()
    return SourceWindow(extracted, tuple(boundary), tuple(outputs), tuple(pairs), start, stop)


def verify_window(
    source: str, projected: str, *, source_sha256: str, projected_sha256: str, start: int, stop: int
) -> SourceWindow:
    """Reopen exact bytes and reconstruct before exposing source/clone associations.

    Provenance attributes, typed arithmetic, literal bits, nested captures and all
    live outputs must match. This grants no inherited trace or host proof: callers
    re-run the original proof on these operations and retain its separate premises.
    """
    if sha256(source.encode()).hexdigest() != source_sha256:
        raise ValueError("source window original content pin differs")
    if sha256(projected.encode()).hexdigest() != projected_sha256:
        raise ValueError("source window projection content pin differs")
    result = extract_window(mq.parse(source), start, stop)
    if text(result.module) != text(mq.parse(projected)):
        raise ValueError("source window does not equal exact source reconstruction")
    return result


def verify_provenance_restoration(original: str, restored: str, *, original_sha256: str, restored_sha256: str) -> dict:
    """Prove a new preparation only adds missing provenance, with unchanged math.

    This does not verify the truth of added provenance or authorize new source
    admission. Capture/trace binding remains an independent caller obligation.
    Existing attributes, types, literals, SSA edges and every output must survive.
    """
    if (
        sha256(original.encode()).hexdigest() != original_sha256
        or sha256(restored.encode()).hexdigest() != restored_sha256
    ):
        raise ValueError("provenance restoration source pin differs")
    before, after = mq.parse(original), mq.parse(restored)
    before.verify()
    after.verify()
    old_ops, new_ops = tuple(before.walk()), tuple(after.walk())
    if len(old_ops) != len(new_ops):
        raise ValueError("provenance restoration changes the operation roster")
    changes = []
    for ordinal, (old, new) in enumerate(zip(old_ops, new_ops, strict=True)):
        if any(new.attributes.get(key) != value for key, value in old.attributes.items()):
            raise ValueError("provenance restoration removes or replaces an existing attribute")
        added = set(new.attributes) - set(old.attributes)
        if any(not key.startswith("prov.") for key in added):
            raise ValueError("provenance restoration adds a semantic attribute")
        if added:
            changes.append({"ordinal": ordinal, "operation": old.name, "added_attributes": sorted(added)})
            old.attributes.update({key: new.attributes[key] for key in added})
    if not before.is_structurally_equivalent(after):
        raise ValueError("provenance restoration changes typed operations, literals or SSA")
    return {
        "schema": "merlin.provenance_restoration.v1",
        "original_sha256": original_sha256,
        "restored_sha256": restored_sha256,
        "n_source_operations": len(old_ops),
        "changes": changes,
        "scope": "provenance-only additions; trace truth and admission remain unproved",
    }


# ---- attention_dataflow ------------------------------------------------------------------------
# Relate tagged QK, softmax and PV operations through structural SSA paths.
#
# This target-independent frontend analysis records unique producer relationships.
# It neither moves operations nor establishes their executable semantic contracts.


def _parents(op: dict) -> list[int]:
    return [source[1] for value in op["ins"] if (source := value["source"])[0] == "op"]


def _nearest(ops: list[dict], start: int, predicate, *, skip_region: str | None = None) -> set[int]:
    """Find the first matching producers on every backward SSA path."""
    found: set[int] = set()
    seen: set[int] = set()
    stack = [start]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        op = ops[index]
        if predicate(op) and op["prov"].get("region_id") != skip_region:
            found.add(index)
            continue
        stack.extend(_parents(op))
    return found


def find_attention_pairs(inventory: dict) -> list[dict]:
    """Return unambiguous source QK/PV pairs with an intervening softmax.

    The input is the structural frontend inventory, with stable producer indices
    and semantic source tags. Labels identify returned source regions; their
    spelling and module names do not select relationships. This analysis grants
    no arithmetic, placement, fusion, layout, lifetime or numerical permission.
    Consumers must separately validate the contraction and softmax bodies and
    their target capabilities before selecting executable transformations.
    """
    ops = inventory["ops"]

    def batch(op: dict) -> bool:
        prov = op["prov"]
        return (
            prov.get("family") == "contraction"
            and prov.get("op") == "batch_matmul"
            and prov.get("aten") == "aten.bmm.default"
            and op["kind"] == "linalg.generic"
            and len(op["ins"]) == 2
            and len(op["results"]) == 1
        )

    def softmax(op: dict) -> bool:
        prov = op["prov"]
        return (
            prov.get("family") == "normalization"
            and prov.get("op") == "softmax"
            and prov.get("aten") == "aten._softmax.default"
            and bool(prov.get("region_id"))
        )

    candidates = {op["prov"]["region_id"]: op for op in ops if batch(op) and op["prov"].get("region_id")}
    pairs: list[dict] = []
    used_qk: set[str] = set()
    for pv_region, pv in candidates.items():
        first = pv["ins"][0]["source"]
        if first[0] != "op":
            continue
        boundaries = _nearest(ops, first[1], softmax, skip_region=pv_region)
        if len({ops[index]["prov"].get("region_id") for index in boundaries}) != 1:
            continue
        qks = set()
        for boundary in boundaries:
            for parent in _parents(ops[boundary]):
                qks.update(_nearest(ops, parent, batch, skip_region=pv_region))
        qk_regions = {ops[index]["prov"].get("region_id") for index in qks}
        if len(qk_regions) != 1:
            continue
        qk_region = qk_regions.pop()
        if (
            qk_region not in candidates
            or qk_region in used_qk
            or candidates[qk_region]["results"][0]["shape"] != pv["ins"][0]["shape"]
        ):
            continue
        used_qk.add(qk_region)
        pairs.append(
            {
                "qk_region_id": qk_region,
                "pv_region_id": pv_region,
                "softmax_region_id": ops[next(iter(boundaries))]["prov"]["region_id"],
            }
        )
    return pairs
