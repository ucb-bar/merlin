"""Invocation-local typed scalar edits at the current native tensor boundary.

Original native operations and resources stay in their session. Only proved
integer leaves/private pure DAGs are replaced; new helpers are parsed separately.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "__merlin_current_scalar_leaf_rewrite__"


def _split_passes(text):
    result, depth, start = [], 0, 0
    for index, char in enumerate(text + ","):
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced source stage pipeline")
        elif char == "," and depth == 0:
            if value := text[start:index].strip():
                result.append(value)
            start = index + 1
    if depth:
        raise ValueError("unbalanced source stage pipeline")
    return result


def insert_marker(pipeline):
    passes = _split_passes(pipeline)
    if MARKER in passes:
        raise ValueError("current scalar stage is one-shot")
    buffers = [i for i, value in enumerate(passes) if value.split("{", 1)[0] == "one-shot-bufferize"]
    if len(buffers) != 1:
        raise ValueError("current scalar stage requires one bufferization boundary")
    index = buffers[0]
    for name in ("linalg-fuse-elementwise-ops", "linalg-generalize-named-ops"):
        matching = [i for i, value in enumerate(passes) if name in value]
        if len(matching) != 1 or matching[0] >= index:
            raise ValueError("current scalar stage requires completed ordinary fusion/generalization")
    return ",".join((*passes[:index], MARKER, *passes[index:]))


@dataclass
class ScalarLeafPatch:
    source_sha256: str
    fragment: str
    replacements: tuple[dict, ...]
    permissions: dict
    _used: bool = field(default=False, init=False, repr=False)

    def claim(self):
        if self._used:
            raise ValueError("current source scalar edit packet was already consumed")
        self._used = True
        if not self.replacements or not self.fragment or "dense_resource" in self.fragment:
            raise ValueError("nonempty typed helper fragment without external resources required")
        if len(self.source_sha256) != 64:
            raise ValueError("exact current source digest required")
        return {
            "schema": "merlin.current_scalar_leaf_patch.v1",
            "source_sha256": self.source_sha256,
            "fragment": self.fragment,
            "replacements": self.replacements,
            "permissions": self.permissions,
        }


def operation_paths(module):
    paths, blocks = {}, {}

    def visit(op, path):
        paths[op] = tuple(path)
        for ri, region in enumerate(op.regions):
            for bi, block in enumerate(region.blocks):
                blocks[block] = {"owner": tuple(path), "region": ri, "block": bi}
                for oi, child in enumerate(block.ops):
                    visit(child, (*path, (ri, bi, oi)))

    visit(module, ())
    return paths, blocks


def value_locator(value, paths, blocks):
    from xdsl.ir import BlockArgument, OpResult

    if isinstance(value, OpResult):
        return {"kind": "result", "path": paths[value.owner], "index": value.index, "type": str(value.type)}
    if isinstance(value, BlockArgument):
        return {"kind": "argument", **blocks[value.owner], "index": value.index, "type": str(value.type)}
    raise ValueError("current scalar input has no structural SSA locator")


# Native runner code uses only its ordinary torch-mlir imports and standard
# library. No parent callback, provider module or Merlin import is transferred.
NATIVE = r"""
_CS_ORIGINAL_RUN_STAGES = _run_stages


def _cs_resolve(module, path):
    op = module.operation
    for ri, bi, oi in path:
        op = list(list(op.regions[ri].blocks)[bi].operations)[oi].operation
    return op


def _cs_value(module, value):
    op = _cs_resolve(module, value.get("path", value.get("owner", [])))
    if value["kind"] == "result":
        result = op.results[value["index"]]
    elif value["kind"] == "argument":
        result = list(op.regions[value["region"]].blocks)[value["block"]].arguments[value["index"]]
    else:
        raise ValueError("unknown current source SSA locator")
    if str(result.type) != value["type"]:
        raise ValueError("current source SSA type changed")
    return result


def _cs_apply(ctx, module, packet, source_digest):
    import hashlib

    if packet.get("schema") != "merlin.current_scalar_leaf_patch.v1" or packet.get("source_sha256") != source_digest:
        raise ValueError("source edit packet is not bound to current native IR")
    current = module.operation.get_asm(print_generic_op_form=True).encode("utf-8")
    if hashlib.sha256(current).hexdigest() != source_digest:
        raise ValueError("native source changed while parent prepared edits")
    if not module.operation.verify():
        raise ValueError("whole current native source does not verify before insertion")
    permissions = packet.get("permissions", {})
    effects = permissions.get("source_effects", {})
    capability = permissions.get("runtime_capability", {})
    effect_keys = {"rne", "gradual_underflow", "nontrapping", "flags_unobserved", "signed_zero_unobserved"}
    if set(effects) != effect_keys or any(value is not True for value in effects.values()):
        raise ValueError("complete source floating effects are absent")
    runtime_keys = {
        "reads_actual_incoming_rounding",
        "preserves_rounding_mode",
        "preserves_flags",
        "nontrapping",
        "no_memory_writes",
    }
    if any(capability.get(key) is not True for key in runtime_keys) or capability.get("placement") != "per_point":
        raise ValueError("honest per-point runtime rounding capability is absent")
    if permissions.get("original_source_fallback") is not True or permissions.get("private_publication") is not True:
        raise ValueError("original fallback and private publication are absent")
    fragment_text = packet.get("fragment", "")
    if not fragment_text or "dense_resource" in fragment_text:
        raise ValueError("unsupported helper fragment resource handles")
    fragment = ir.Module.parse(fragment_text, ctx)
    if not fragment.operation.verify():
        raise ValueError("helper fragment failed ordinary native verification")
    new_ops = [op.operation for op in fragment.body.operations]
    allowed_top = {"func.func", "memref.global"}
    names = []
    for op in new_ops:
        if op.name not in allowed_top or "sym_name" not in op.attributes:
            raise ValueError("helper fragment contains unsupported top-level effects")
        names.append(str(op.attributes["sym_name"]))
        if "sym_visibility" not in op.attributes or str(op.attributes["sym_visibility"]) != '"private"':
            raise ValueError("helper fragment namespace is not private")
        if op.name == "memref.global" and "constant" not in op.attributes:
            raise ValueError("helper coefficient storage is not immutable")
    existing = {str(op.attributes["sym_name"]) for op in module.body.operations if "sym_name" in op.attributes}
    if len(set(names)) != len(names) or existing.intersection(names):
        raise ValueError("current helper namespace conflict")
    by_name = {str(op.attributes["sym_name"]): op for op in new_ops}
    predicate = by_name.get('"' + capability.get("symbol", "") + '"')
    if predicate is None or predicate.name != "func.func":
        raise ValueError("runtime rounding predicate declaration is absent")
    predicate_type = ir.FunctionType(ir.TypeAttr(predicate.attributes["function_type"]).value)
    if (
        tuple(predicate_type.inputs)
        or [str(value) for value in predicate_type.results] != ["i1"]
        or list(predicate.regions[0].blocks)
    ):
        raise ValueError("runtime rounding predicate is not the provider-owned () -> i1 declaration")
    # Resolve and authenticate EVERY edit before changing a use or an operation.
    replacements, anchors, private = [], set(), set()
    pure = {
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.addi",
        "arith.subi",
        "arith.shli",
        "arith.andi",
        "arith.ori",
        "arith.cmpi",
        "arith.cmpf",
        "arith.select",
        "arith.bitcast",
        "arith.fptosi",
        "arith.sitofp",
        "arith.uitofp",
        "arith.trunci",
        "arith.extsi",
        "arith.extui",
        "math.fma",
        "math.roundeven",
        "math.floor",
        "math.ceil",
    }
    for row in packet["replacements"]:
        key = tuple(tuple(piece) for piece in row["anchor"])
        if key in anchors:
            raise ValueError("duplicate current scalar integer anchor")
        anchors.add(key)
        anchor = _cs_resolve(module, row["anchor"])
        scalar_return = anchor.name in ("linalg.yield", "func.return") and len(anchor.operands) == 1
        tensor_insert = (
            anchor.name == "tensor.insert" and len(anchor.operands) >= 3 and len(anchor.results) == 1
            and anchor.results[0].type == anchor.operands[1].type
            and isinstance(anchor.results[0].type, ir.RankedTensorType)
            and str(ir.RankedTensorType(anchor.results[0].type).element_type) == "i8"
            and len(anchor.operands) == ir.RankedTensorType(anchor.results[0].type).rank + 2
            and all(str(value.type) == "index" for value in anchor.operands[2:])
        )
        if not (scalar_return or tensor_insert) or str(anchor.operands[0].type) != "i8":
            raise ValueError("original closed integer leaf is unavailable")
        uses = tuple(anchor.operands[0].uses)
        if len(uses) != 1 or uses[0].owner.operation != anchor or uses[0].operand_number != 0:
            raise ValueError("original integer leaf does not have one publication")
        cut, up = _cs_value(module, row["cut"]), _cs_value(module, row["up"])
        if str(cut.type) != "f32" or str(up.type) != "f32":
            raise ValueError("original scalar producer types changed")
        if ('"' + row["adapter"] + '"') not in names:
            raise ValueError("current scalar adapter is absent")
        adapter_op = by_name['"' + row["adapter"] + '"']
        if adapter_op.name != "func.func":
            raise ValueError("scalar adapter symbol is not a function")
        adapter_type = ir.FunctionType(ir.TypeAttr(adapter_op.attributes["function_type"]).value)
        if [str(value) for value in adapter_type.inputs] != ["f32", "f32"] or [
            str(value) for value in adapter_type.results
        ] != ["i8"]:
            raise ValueError("scalar adapter does not preserve the original leaf ABI")
        removal = []
        for item in row["remove"]:
            op = _cs_resolve(module, item["path"])
            path = tuple(tuple(piece) for piece in item["path"])
            if path in private or op.name != item["name"] or op.name not in pure or op.regions:
                raise ValueError("overlapping or unsupported private scalar operations")
            if op.block != anchor.block:
                raise ValueError("private scalar operation escaped its original block")
            private.add(path)
            removal.append(op)
        allowed = set(removal) | {anchor}
        for op in removal:
            for result in op.results:
                if any(use.owner.operation not in allowed for use in result.uses):
                    raise ValueError("private scalar DAG has an unpreserved live escape")
        if any(cut == result or up == result for op in removal for result in op.results):
            raise ValueError("original scalar producer belongs to the removed DAG")
        replacements.append((anchor, cut, up, row["adapter"], removal))
    # No original module/global/resource or producer is reparsed or replaced.
    for op in new_ops:
        op.detach_from_parent()
        module.body.append(op)
    for anchor, cut, up, adapter, removal in replacements:
        trace = []
        for original in [*removal, anchor]:
            attrs = {key: original.attributes[key] for key in original.attributes if key.startswith("prov.")}
            if attrs:
                trace.append(ir.DictAttr.get(attrs, ctx))
        attrs = {key: anchor.attributes[key] for key in anchor.attributes if key.startswith("prov.")}
        attrs["callee"] = ir.FlatSymbolRefAttr.get(adapter, ctx)
        attrs["prov.scalar_carrier_source_trace"] = ir.ArrayAttr.get(trace, ctx)
        attrs["prov.scalar_carrier_rewrite"] = ir.StringAttr.get("current_closed_integer_leaf", ctx)
        with ir.Location.fused([op.location for op in [*removal, anchor]], context=ctx):
            call = ir.Operation.create(
                "func.call",
                results=[anchor.operands[0].type],
                operands=[cut, up],
                attributes=attrs,
                ip=ir.InsertionPoint(anchor),
            )
        anchor.operands[0] = call.results[0]
        for op in reversed(removal):
            op.erase()
    if not module.operation.verify():
        raise ValueError("current scalar edit failed ordinary native verification")


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    import hashlib, json, time
    from pathlib import Path

    passes = _cs_split_passes(pipeline)

    def ordinary_value(value):
        if type(value) in (type(None), bool, int, float, str):
            return True
        if type(value) in (tuple, list):
            return all(ordinary_value(item) for item in value)
        if type(value) is dict:
            return all(ordinary_value(key) and ordinary_value(item) for key, item in value.items())
        return False

    if any(not ordinary_value(globals().get(name)) for name in _CS_CROSSING_VALUES):
        raise ValueError("normal runner retains a native source handle across the selected seam")
    marker = "__merlin_current_scalar_leaf_rewrite__"
    if passes.count(marker) != 1:
        raise ValueError("selected current scalar transport requires one stage")
    index = passes.index(marker)
    head, tail = passes[:index], passes[index + 1 :]
    if (
        not tail
        or not tail[0].startswith("one-shot-bufferize")
        or any("generalize" in p or "fuse-elementwise" in p for p in tail)
    ):
        raise ValueError("current scalar stage has a duplicating or misplaced tail")
    hoist_head = any("buffer-loop-hoisting" in p for p in head)
    late_head = any("convert-scf-to-openmp" in p for p in head)
    _CS_ORIGINAL_RUN_STAGES(
        ctx,
        module,
        ",".join(head),
        erase if hoist_head else 0,
        mid if hoist_head else (),
        late if late_head else (),
        post_openmp if late_head else (),
        pre_generalize,
    )
    payload = module.operation.get_asm(print_generic_op_form=True).encode("utf-8")
    if len(payload) > _CS_MAX_SOURCE:
        raise ValueError("current native source exceeds explicit byte quota")
    digest = hashlib.sha256(payload).hexdigest()
    Path(_CS_SOURCE).write_bytes(payload)
    pending = Path(_CS_REQUEST + ".pending")
    pending.write_text(
        json.dumps({"schema": "merlin.current_scalar_stage_request.v1", "sha256": digest, "bytes": len(payload)})
    )
    pending.replace(_CS_REQUEST)
    deadline = time.monotonic() + _CS_TIMEOUT
    while not Path(_CS_RESPONSE).is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("current scalar parent response timed out")
        time.sleep(0.01)
    response = Path(_CS_RESPONSE).read_bytes()
    if len(response) > _CS_MAX_RESPONSE:
        raise ValueError("current scalar response exceeds explicit byte quota")
    _cs_apply(ctx, module, json.loads(response), digest)
    print("OK current_scalar_leaf_stage", digest, hashlib.sha256(response).hexdigest())
    return _CS_ORIGINAL_RUN_STAGES(
        ctx,
        module,
        ",".join(tail),
        0 if hoist_head else erase,
        () if hoist_head else mid,
        () if late_head else late,
        () if late_head else post_openmp,
        (),
    )
"""


def bind_runner(source, *, directory, max_source_bytes, max_response_bytes, timeout):
    tree = ast.parse(source)
    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_run_stages"
    ]
    if len(calls) != 1:
        raise ValueError("current scalar transport requires one normal staged native call")
    # Rewrites preceding the seam can return ordinary counters/reports. Check
    # only values that cross it: post-seam queries of the original module are
    # fresh. Native values crossing this boundary fail before the head runs.
    at = tree.body.index(calls[0])
    assignments = []

    def collect(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(node)
        for child in ast.iter_child_nodes(node):
            collect(child)

    for node in tree.body[:at]:
        collect(node)
    derived = {"module"}
    changed = True
    while changed:
        before = set(derived)
        for node in assignments:
            if node.value is not None and any(
                isinstance(item, ast.Name) and item.id in derived for item in ast.walk(node.value)
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                derived.update(item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name))
        changed = before != derived
    later = {
        item.id
        for node in tree.body[at + 1 :]
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }
    crossing = tuple(sorted((derived - {"module"}) & later))
    directory = Path(directory)
    paths = {name: str(directory / name) for name in ("current.mlir", "request.json", "response.json")}
    config = (
        f"_CS_SOURCE = {paths['current.mlir']!r}\n_CS_REQUEST = {paths['request.json']!r}\n"
        f"_CS_RESPONSE = {paths['response.json']!r}\n_CS_MAX_SOURCE = {max_source_bytes!r}\n"
        f"_CS_MAX_RESPONSE = {max_response_bytes!r}\n_CS_TIMEOUT = {timeout!r}\n"
        f"_CS_CROSSING_VALUES = {crossing!r}\n"
    )
    splitter = inspect.getsource(_split_passes).replace("def _split_passes", "def _cs_split_passes", 1)
    lines = source.splitlines(keepends=True)
    index = calls[0].lineno - 1
    return "".join(lines[:index]) + config + splitter + "\n" + NATIVE + "\n" + "".join(lines[index:])


def run_command(command, *, directory, callback, max_source_bytes, max_response_bytes, timeout):
    """Serve one callback on the calling thread and terminate only our child."""
    from merlin.compiler.lower.model.optional_passes import child_environment

    directory = Path(directory)
    source, request, response = (directory / name for name in ("current.mlir", "request.json", "response.json"))
    if any(path.exists() for path in (source, request, response)):
        raise ValueError("current scalar transport requires a fresh private request owner")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_environment()
    )
    served = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                if request.is_file() and not served:
                    if (
                        request.stat().st_size > 1024
                        or not source.is_file()
                        or source.stat().st_size > max_source_bytes
                    ):
                        raise ValueError("current scalar request exceeds explicit private byte quota")
                    payload = source.read_bytes()
                    descriptor = json.loads(request.read_text())
                    digest = hashlib.sha256(payload).hexdigest()
                    if descriptor != {
                        "schema": "merlin.current_scalar_stage_request.v1",
                        "sha256": digest,
                        "bytes": len(payload),
                    }:
                        raise ValueError("current scalar request source identity changed")
                    patch = callback(payload, directory)
                    if type(patch) is not ScalarLeafPatch or patch.source_sha256 != digest:
                        raise ValueError("typed callback returned a stale source packet")
                    encoded = (json.dumps(patch.claim(), sort_keys=True) + "\n").encode()
                    if len(encoded) > max_response_bytes:
                        raise ValueError("current scalar response exceeds explicit private byte quota")
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    temporary = directory / "response.pending"
                    temporary.write_bytes(encoded)
                    temporary.replace(response)
                    served = True
        if process.returncode == 0 and not served:
            raise ValueError("normal native runner did not request its selected source stage")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
