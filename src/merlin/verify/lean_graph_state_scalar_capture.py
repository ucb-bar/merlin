"""Actual pinned scalar producer and independent complete typed SSA projection.

No witness counts or optimizer replay plans authorize this correspondence.
Every module/function/region/call operand is parsed from both actual buffers.
Unknown CFG/region semantics are refused, never erased as an empty boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path

if __package__:
    from . import lean_graph_input_admission as admission
else:
    import lean_graph_input_admission as admission

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exact(value):
    from xdsl.ir import Attribute, Data, ParametrizedAttribute

    kind = type(value)
    if value is None:
        return ["none"]
    if kind in (bool, int, str):
        return [kind.__name__, value]
    if kind is float:
        return ["float64-bits", struct.pack(">d", value).hex()]
    if kind is bytes:
        return ["bytes", value.hex()]
    if isinstance(value, Enum):
        return [kind.__module__ + "." + kind.__qualname__, value.name, exact(value.value)]
    if kind in (tuple, list):
        return [kind.__name__, [exact(item) for item in value]]
    if kind in (set, frozenset):
        return [kind.__name__, sorted([exact(item) for item in value], key=canonical)]
    if kind is dict:
        return ["dictionary", [[key, exact(item)] for key, item in sorted(value.items())]]
    if isinstance(value, ParametrizedAttribute):
        return [
            "attribute",
            kind.__module__ + "." + kind.__qualname__,
            value.name,
            [exact(item) for item in value.parameters],
        ]
    if isinstance(value, Data):
        return ["data-attribute", kind.__module__ + "." + kind.__qualname__, value.name, exact(value.data)]
    if isinstance(value, Attribute):
        raise ValueError("unsupported complete opaque attribute representation: " + kind.__qualname__)
    raise ValueError("unsupported complete parsed representation: " + kind.__qualname__)


def metadata(operation):
    return {
        "class": type(operation).__module__ + "." + type(operation).__qualname__,
        "name": operation.name,
        "attributes": exact(operation.attributes),
        "properties": exact(operation.properties),
        "operand_types": [exact(value.type) for value in operation.operands],
        "result_types": [exact(value.type) for value in operation.results],
    }


def selected_primitive(operation):
    """Independent closed-whitelist reconstruction from actual operations."""
    from xdsl.dialects import arith, builtin
    from xdsl.traits import get_effects

    kinds = {
        arith.AddiOp: "add",
        arith.SubiOp: "sub",
        arith.MuliOp: "mul",
        arith.ExtSIOp: "sext",
        arith.ExtUIOp: "zext",
        arith.TruncIOp: "trunc",
        arith.ConstantOp: "constant",
    }
    name = kinds.get(type(operation))
    if name is None or operation.regions or operation.successors or operation.attributes or len(operation.results) != 1:
        return None
    if get_effects(operation) != set():
        return None
    types = [value.type for value in (*operation.operands, *operation.results)]
    if any(
        type(t) is not builtin.IntegerType
        or t.signedness.data != builtin.Signedness.SIGNLESS
        or not 1 <= t.width.data <= 256
        for t in types
    ):
        return None
    allowed = {"value"} if name == "constant" else {"overflowFlags"} if name in ("add", "sub", "mul") else set()
    if set(operation.properties) - allowed:
        return None
    if name in ("add", "sub", "mul") and operation.overflow_flags != arith.IntegerOverflowAttr("none"):
        return None
    if name == "constant":
        if not isinstance(operation.value, builtin.IntegerAttr):
            return None
        return {"kind": name, "width": types[-1].width.data, "value": operation.value.value.data}
    if name in ("add", "sub", "mul"):
        return {"kind": name, "width": types[-1].width.data}
    return {"kind": name, "source": types[0].width.data, "target": types[-1].width.data}


def project(before, after, *, context=(), _allow_captured_stride=False):
    if type(_allow_captured_stride) is not bool:
        raise ValueError("explicit auxiliary stride-projection mode required")
    for module in (before, after, *context):
        admission.admit_module(module)

    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.ir import BlockArgument

    if type(before) is not builtin.ModuleOp or type(after) is not builtin.ModuleOp:
        raise ValueError("complete actual builtin.module pair required")
    if metadata(before) != metadata(after):
        raise ValueError("complete surrounding module metadata changed")
    if len(before.body.blocks) != 1 or len(after.body.blocks) != 1:
        raise ValueError("unsupported complete module block geometry")
    functions = [tuple(module.body.block.ops) for module in (before, after)]
    if len(functions[0]) != len(functions[1]) or not functions[0]:
        raise ValueError("complete nonempty function/declaration roster required")
    if any(type(function) is not func.FuncOp for rows in functions for function in rows):
        raise ValueError("unsupported module-level operation; no missing surrounding context")
    names = [function.sym_name.data for function in functions[0]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate actual function symbols")
    defined_names = {function.sym_name.data for function in functions[0] if function.body.blocks}
    universe = (before, after, *context)
    registry_tokens = sorted(
        {
            canonical(metadata(op))
            for module in universe
            for op in module.walk()
            if not op.regions
            and not op.successors
            and selected_primitive(op) is None
            and type(op) not in (func.ReturnOp, scf.YieldOp)
        }
    )
    type_tokens = sorted(
        {
            canonical(exact(value.type))
            for module in universe
            for operation in module.walk()
            for value in (
                *operation.operands,
                *operation.results,
                *(argument for region in operation.regions for block in region.blocks for argument in block.args),
            )
        }
    )

    def type_code(value):
        if type(value) is builtin.IntegerType and value.signedness.data == builtin.Signedness.SIGNLESS:
            return {"bits": value.width.data}
        if type(value) is builtin.IndexType:
            return {"index": True}
        return {"opaque": type_tokens.index(canonical(exact(value)))}

    registry = []
    for token in registry_tokens:
        descriptor = json.loads(token)
        # Decode types from the real operations, not from a guessed descriptor text.
        representative = next(op for module in universe for op in module.walk() if canonical(metadata(op)) == token)
        registry.append(
            {
                "descriptor": descriptor,
                "inputs": [type_code(v.type) for v in representative.operands],
                "outputs": [type_code(v.type) for v in representative.results],
                "index_constant": representative.value.value.data
                if type(representative) is arith.ConstantOp and type(representative.result.type) is builtin.IndexType
                else None,
            }
        )

    count = 0

    def block_code(block, captured, terminator):
        nonlocal count
        environment = [*block.args, *captured]
        operations = list(block.ops)
        if not operations or type(operations[-1]) is not terminator:
            raise ValueError("missing/unsupported actual terminator")
        records = []

        def uses(values):
            if any(value not in environment for value in values):
                raise ValueError("actual operand lacks lexical availability/dominance")
            return [environment.index(value) for value in values]

        for operation in operations:
            count += 1
            if count > 2048:
                raise ValueError("complete graph exceeds explicit scalar intake budget")
            if operation.successors:
                raise ValueError("unsupported actual CFG; no unreachable block can be dropped")
            if type(operation) in (func.ReturnOp, scf.YieldOp):
                if (
                    operation is not operations[-1]
                    or type(operation) is not terminator
                    or operation.attributes
                    or operation.properties
                ):
                    raise ValueError("changed/unsupported terminator metadata or control")
                records.append({"ret": uses(operation.operands)})
                continue
            primitive = selected_primitive(operation)
            if primitive is not None:
                records.append({"pure": primitive, "arguments": uses(operation.operands)})
            elif type(operation) is scf.ForOp:
                if operation.attributes or operation.properties or len(operation.body.blocks) != 1:
                    raise ValueError("unsupported loop metadata or block geometry")
                from xdsl.dialects import arith

                step_owner = operation.step.owner
                positive_literal = (
                    type(step_owner) is arith.ConstantOp
                    and isinstance(step_owner.value, builtin.IntegerAttr)
                    and step_owner.value.value.data > 0
                )
                # Captured block arguments occur only in internal logical
                # fragments. Every actual module uses the strict default.
                captured = _allow_captured_stride and isinstance(operation.step, BlockArgument)
                if not (positive_literal or captured):
                    raise ValueError("supported actual SCF control needs a checked positive constant stride")
                carries = [type_code(value.type) for value in operation.iter_args]
                arguments = list(operation.body.block.args)
                if [type_code(value.type) for value in arguments] != [{"index": True}, *carries]:
                    raise ValueError("actual induction/carry argument slots differ")
                if [type_code(value.type) for value in operation.results] != carries:
                    raise ValueError("actual loop result slots differ from carries")
                records.append(
                    {
                        "loop": {
                            "bounds": uses((operation.lb, operation.ub, operation.step)),
                            "initial": uses(operation.iter_args),
                            "carries": carries,
                            "body": block_code(operation.body.block, environment, scf.YieldOp),
                        }
                    }
                )
            else:
                if operation.regions:
                    raise ValueError("unsupported region semantics; entire boundary refused")
                if type(operation) is func.CallOp and operation.callee.root_reference.data in defined_names:
                    raise ValueError("internal-call/fuel composition bridge unavailable; external contracts only")
                token = canonical(metadata(operation))
                records.append(
                    {
                        "effect": registry_tokens.index(token),
                        "arguments": uses(operation.operands),
                        "outputs": [type_code(value.type) for value in operation.results],
                    }
                )
            environment = [*operation.results, *environment]
        return records

    result = []
    for original, actual in zip(*functions, strict=True):
        if metadata(original) != metadata(actual):
            raise ValueError("complete function interface/attributes changed")
        if len(original.body.blocks) != len(actual.body.blocks) or len(original.body.blocks) > 1:
            raise ValueError("unsupported/dropped function CFG blocks")
        signature = original.function_type
        row = {
            "name": original.sym_name.data,
            "header": metadata(original),
            "inputs": [type_code(value) for value in signature.inputs],
            "outputs": [type_code(value) for value in signature.outputs],
        }
        if not original.body.blocks:
            row.update(before=None, after=None, declaration=True)
        else:
            if tuple(v.type for v in original.body.block.args) != tuple(signature.inputs) or tuple(
                v.type for v in actual.body.block.args
            ) != tuple(signature.inputs):
                raise ValueError("actual function argument slots differ from interface")
            row.update(
                before=block_code(original.body.block, [], func.ReturnOp),
                after=block_code(actual.body.block, [], func.ReturnOp),
                declaration=False,
            )
        result.append(row)
    return {
        "registry": registry,
        "types": [json.loads(token) for token in type_tokens],
        "functions": result,
        "module_header": metadata(before),
        "operation_budget_used": count,
    }


def capture(reference, source, target, destination, mode, middle_path=None):
    reference, destination = Path(reference).resolve(), Path(destination).resolve()
    head = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != REVISION:
        raise ValueError("actual scalar producer revision differs")
    subprocess.run(
        ["git", "-C", str(reference), "diff", "--exit-code", REVISION, "--", "src/merlin"],
        check=True,
        capture_output=True,
    )
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.certificates import parse_ir as producer_parse_ir

    def parse_ir(raw):
        return admission.parse_admitted(raw, producer_parse_ir)

    from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants

    before = parse_ir(Path(source).read_text())
    before.verify()
    witnesses = []
    if target is None:
        after = before.clone()
        kinds = {
            "cse": [eliminate_scalar_redundancy],
            "licm": [hoist_scalar_invariants],
            "chain": [eliminate_scalar_redundancy, hoist_scalar_invariants],
        }[mode]
        middles = []
        for transform in kinds:
            witnesses.append(
                {
                    "producer": transform.__module__ + "." + transform.__name__,
                    "witnesses": [asdict(value) for value in transform(after)],
                }
            )
            middles.append(ir_text(after))
    else:
        after, middles = parse_ir(Path(target).read_text()), []
    after.verify()
    intermediate = None
    if middle_path is not None:
        intermediate = parse_ir(Path(middle_path).read_text())
        intermediate.verify()
    elif target is None and mode == "chain":
        intermediate = parse_ir(middles[0])
        intermediate.verify()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "before.mlir").write_bytes(Path(source).read_bytes())
    (destination / "after.mlir").write_text(ir_text(after) if target is None else Path(target).read_text())
    for index, middle in enumerate(middles):
        (destination / f"stage-{index:03d}.mlir").write_text(middle)
    if intermediate is not None:
        (destination / "middle.mlir").write_text(
            ir_text(intermediate) if middle_path is None else Path(middle_path).read_text()
        )
    (destination / "producer-execution.json").write_text(
        json.dumps(
            {
                "producer_revision": head,
                "mode": mode if target is None else "supplied-actual-result",
                "diagnostic_witnesses": witnesses,
                "owned_generator": file_ref(__file__),
                "input_admission_source": admission.source_reference(),
                "complete_raw_outputs_retained_before_projection": True,
                "public_acceptance_issued": False,
            },
            indent=2,
        )
        + "\n"
    )
    projection = project(before, after, context=() if intermediate is None else (intermediate,))
    if intermediate is not None:
        first = project(before, intermediate, context=(after,))
        second = project(intermediate, after, context=(before,))
        if first["registry"] != second["registry"] or first["registry"] != projection["registry"]:
            raise ValueError("actual stages do not share one complete descriptor/semantic registry")
        if first["types"] != second["types"] or first["types"] != projection["types"]:
            raise ValueError("independently numbered type tables cannot be identified by fiat")
        projection["chain_stages"] = {"first": first, "second": second}
        projection["stage_changed"] = [
            ir_text(before) != ir_text(intermediate),
            ir_text(intermediate) != ir_text(after),
        ]
    modules, external = {}, {}
    for name, module in tuple(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw or not Path(raw).is_file():
            continue
        path = Path(raw).resolve()
        if name == "merlin" or name.startswith("merlin."):
            if not path.is_relative_to(reference / "src"):
                raise ValueError("actual producer/parser imports escaped pinned source")
            modules[str(path)] = file_ref(path)
        else:
            external[str(path)] = file_ref(path)
    result = {
        "schema": "merlin.graph-state-scalar.actual-projection.v1",
        "projection_checked": True,
        "producer_revision": head,
        "producer_mode": mode if target is None else "supplied-actual-result",
        "genuine_changed_boundary": ir_text(before) != ir_text(after),
        "diagnostic_witnesses": witnesses,
        "inputs": [file_ref(destination / "before.mlir"), file_ref(destination / "after.mlir")],
        "producer_sources": sorted(modules.values(), key=lambda row: row["path"]),
        "external_runtime": {
            "python": file_ref(sys.executable),
            "sources": sorted(external.values(), key=lambda row: row["path"]),
        },
        "owned_projection": file_ref(__file__),
        "input_admission_source": admission.source_reference(),
        **projection,
        "trust": [
            "pinned xDSL parser/verification and actual scalar producer execution",
            "primary complete typed SSA/metadata projection into the checked graph language",
            "eligible primitives have the selected total deterministic pure value meaning",
            "opaque descriptors share the same partial persistent-world interpretation and successful result arity",
            "SCF bound values/control interpretation matches a common finite iteration sequence on its admitted domain",
            "local calls, CFG and opaque regions outside the supported model are refused",
        ],
        "primitive_fidelity_proved": False,
        "source_success_proved": False,
        "poison_freedom_proved": False,
        "public_acceptance_issued": False,
        "review_approved": False,
        "native_fidelity_proved": False,
    }
    (destination / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("reference", "source", "destination"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--middle", type=Path)
    parser.add_argument("--mode", choices=("cse", "licm", "chain"), default="cse")
    args = parser.parse_args()
    try:
        result = capture(args.reference, args.source, args.target, args.destination, args.mode, args.middle)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        args.destination.mkdir(parents=True, exist_ok=True)
        result = {
            "projection_checked": False,
            "reason": {"type": type(error).__name__, "detail": str(error)},
            "public_acceptance_issued": False,
        }
        for name, path in (("before", args.source), ("after", args.target)):
            if path is not None:
                (args.destination / (name + ".mlir")).write_bytes(path.read_bytes())
        (args.destination / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"projection_checked": result["projection_checked"]}))


if __name__ == "__main__":
    main()
