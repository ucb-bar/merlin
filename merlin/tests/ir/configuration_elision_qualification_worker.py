"""Isolated pinned producer, original codec projection and compiler oracle.

Only stdlib imports occur before mode selection. Producer mode uses the exact
read-only reference checkout; verification mode uses the authoritative compiler.
Neither mode issues proof or policy-review authority.
"""

import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def reference(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def source_identity(root):
    observed = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if observed != REVISION:
        raise ValueError("actual configuration producer revision mismatch")
    sources = []
    for name, module in sorted(tuple(sys.modules.items())):
        if name != "merlin" and not name.startswith("merlin."):
            continue
        if not getattr(module, "__file__", None):
            raise ValueError("producer namespace has no source identity")
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root / "src"):
            raise ValueError("producer module escaped original pinned checkout")
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", REVISION + ":" + str(path.relative_to(root))]
        )
        if path.read_bytes() != original:
            raise ValueError("loaded producer source differs from selected commit")
        sources.append(reference(path))
    return sources


def selection(request):
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

    chosen = read_scheduling_contracts(request["facts"], request["software"], target=request["target"])
    return ConfigurationIRSelection(chosen, request["qualification"])


def decoded(module, chosen):
    """Inspect actual typed SSA constants, original codec and declared field rows."""
    from xdsl.dialects import arith, builtin, func

    from merlin.sched.ir import Call, Const
    from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction

    isa = chosen.selected_contracts.isa
    rows = json.loads(chosen.selected_contracts.effects_json)
    original_adapter = chosen.selected_contracts.configuration_adapter(chosen.qualification_identity)
    output = []
    for function in (op for op in module.walk() if type(op) is func.FuncOp and op.body.blocks):
        operations = []
        for index, op in enumerate(function.body.block.ops):
            instruction = next((item for item in isa.instructions if op.name == isa.namespace + "." + item.name), None)
            row = dict(index=index, operation=op.name, effect=None)
            if (
                instruction is not None
                and not op.results
                and not op.properties
                and not op.regions
                and not op.successors
            ):
                values = {}
                eligible = len(op.operands) == len(instruction.operands)
                for argument, operand in zip(instruction.operands, op.operands, strict=True):
                    eligible = (
                        eligible
                        and type(operand.owner) is arith.ConstantOp
                        and operand.type == builtin.IntegerType(argument.type.bits)
                    )
                    if not eligible:
                        break
                    number = operand.owner.value.value.data
                    values[argument.name] = number if argument.type.signed else number & ((1 << argument.type.bits) - 1)
                if eligible and not op.attributes:
                    word = encode_instruction(isa, instruction.name, values)
                    name, bindings = decode_instruction(isa, word)
                    if name != instruction.name or bindings != values:
                        raise ValueError("original encoding does not roundtrip actual SSA operand values")
                    call = Call(name, tuple((key, Const(value)) for key, value in bindings.items()))
                    effect = original_adapter.project(call)
                    declaration = rows.get(name)
                    expected = None
                    if declaration is not None:

                        def value(item):
                            return bindings[item["argument"]] if "argument" in item else item["constant"]

                        def field(item):
                            return item["resource"] + ":" + str(value(item["index"]))

                        expected = {
                            kind: [
                                field(item) if kind == "clobbers" else [field(item), value(item["value"])]
                                for item in declaration[kind]
                            ]
                            for kind in ("reads", "writes", "clobbers")
                        }
                        expected.update(ordered=declaration["ordered"], pure_write=declaration["pure_write"])
                    actual = None if effect is None else json.loads(json.dumps(dataclasses.asdict(effect)))
                    if expected != actual:
                        raise ValueError("independent schema field decoding disagrees with actual producer adapter")
                    row.update(instruction=name, bindings=bindings, encoded_word=word, effect=actual)
            operations.append(row)
        output.append(dict(function=function.sym_name.data, operations=operations))
    return output


def produce(request):
    from merlin.transforms.certificates import ir_text

    root = Path(request["reference"]).resolve()
    chosen = selection(request)
    source = chosen.parse(request["source"])
    before = ir_text(source)
    result = chosen.apply(source)
    result.revalidate()
    if ir_text(source) != before:
        raise ValueError("actual production configuration pass mutated its source")
    return dict(
        before=before,
        after=ir_text(result.module),
        producer_report=result.report,
        source_decoding=decoded(source, chosen),
        final_decoding=decoded(result.module, chosen),
        producer_revision=REVISION,
        producer_sources=source_identity(root),
        selected_record=chosen.record(),
        isa_record=chosen.selected_contracts.isa.record(),
        effects=json.loads(chosen.selected_contracts.effects_json),
        facts=reference(request["facts"]),
        software=reference(request["software"]),
        actual_API="configuration_ir.ConfigurationIRSelection.apply -> configuration.eliminate_redundant_configuration",
        effect_decoder_independently_compared=True,
        human_approved=False,
        native_effect_fidelity=False,
        authority=False,
    )


def mutate(request):
    from xdsl.dialects import arith, builtin, func

    from merlin.compile.scheduling.configuration_ir import verify_configuration_ir
    from merlin.transforms.certificates import ir_text

    chosen = selection(request)
    original, actual = chosen.parse(request["before"]), chosen.parse(request["after"])
    function = next(op for op in actual.walk() if type(op) is func.FuncOp and op.sym_name.data == "run")
    block = function.body.block
    namespace = chosen.selected_contracts.isa.namespace

    def instruction(name):
        return next(op for op in block.ops if op.name == namespace + "." + name)

    def constant(value, *, last=False, bits=None):
        items = [
            op.results[0]
            for op in block.ops
            if type(op) is arith.ConstantOp
            and op.value.value.data == value
            and (bits is None or op.results[0].type == builtin.IntegerType(bits))
        ]
        return items[-1] if last else items[0]

    fault = request["mutation"]
    if fault in {"drop_first_write", "drop_observer", "drop_reset", "drop_unknown", "drop_unordered", "drop_pulse"}:
        name = {
            "drop_first_write": "pair",
            "drop_observer": "work",
            "drop_reset": "reset",
            "drop_unknown": "unknown",
            "drop_unordered": "unordered",
            "drop_pulse": "pulse",
        }[fault]
        block.erase_op(instruction(name))
    elif fault in {"wrong_retained_value", "wrong_same_typed_binding"}:
        op = instruction("pair")
        replacement = constant(7, bits=32) if fault == "wrong_retained_value" else constant(3, last=True, bits=32)
        op.operands = (op.operands[0], replacement, op.operands[2])
    elif fault == "wrong_retained_field":
        op = instruction("set")
        op.operands = (constant(2, bits=8), *op.operands[1:])
    elif fault == "changed_constant":
        op = constant(3, bits=32).owner
        op.properties["value"] = builtin.IntegerAttr(7, builtin.i32)
    elif fault == "changed_visibility":
        function.properties["sym_visibility"] = builtin.StringAttr("private")
    elif fault == "changed_symbol":
        function.properties["sym_name"] = builtin.StringAttr("changed")
    elif fault in {"changed_input_type", "changed_signature"}:
        if fault == "changed_input_type":
            block.args[0]._type = builtin.i64
        else:
            block.insert_arg(builtin.i32, len(block.args))
        function.properties["function_type"] = builtin.FunctionType.from_lists([item.type for item in block.args], [])
    elif fault == "changed_metadata":
        actual.attributes["fixture.changed"] = builtin.StringAttr("changed")
    elif fault == "reordered_observer":
        op = instruction("work")
        block.detach_op(op)
        block.insert_op_before(op, instruction("pair"))
    elif fault == "inserted_observer":
        block.insert_op_before(instruction("work").clone(), block.last_op)
    elif fault == "wrong_token_binding":
        op = next(op for op in block.ops if op.attributes.get("op_name__") == builtin.StringAttr("fixture.consume"))
        op.operands = tuple(reversed(op.operands))
    elif fault == "drop_changed_pair":
        op = next(
            op for op in block.ops if op.name == namespace + ".pair" and op.operands[2].owner.value.value.data == 7
        )
        block.erase_op(op)
    elif fault == "drop_required_after_clobber":
        block.erase_op(instruction("set"))
    elif fault == "drop_unused_constant":
        block.erase_op(constant(3, last=True, bits=32).owner)
    else:
        raise ValueError("unknown independent mutation")
    actual.verify()
    try:
        report = verify_configuration_ir(original, actual, selection=chosen)
    except ValueError as error:
        conformance = dict(status="PRODUCTION_CONFORMANCE_REFUSED", reason=str(error))
    else:
        conformance = dict(status="PRODUCTION_CONFORMANCE_ACCEPTED", report=report)
    return dict(
        after=ir_text(actual),
        mutation=fault,
        xdsl_valid=True,
        original_conformance=conformance,
        producer_sources=source_identity(Path(request["reference"]).resolve()),
        authority=False,
    )


def verify(request):
    from torch_mlir import ir

    rows = []
    for name in request["files"]:
        path = Path(name)
        payload = path.read_bytes()
        record = reference(path)
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("oracle buffer changed while reading")
        try:
            with ir.Context() as context:
                context.allow_unregistered_dialects = True
                module = ir.Module.parse(payload.decode())
                valid = bool(module.operation.verify())
            rows.append(dict(parsed_bytes=record, valid=valid))
        except Exception as error:
            rows.append(dict(parsed_bytes=record, valid=False, reason=str(error)))
    return dict(
        files=rows,
        oracle=dict(
            python=sys.version,
            executable=sys.executable,
            distributions={
                name: importlib.metadata.version(name)
                for name in importlib.metadata.packages_distributions().get("torch_mlir", [])
            },
            allow_unregistered_dialects=True,
            custom_instruction_semantics="not qualified by generic MLIR verification",
        ),
    )


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    try:
        result = {"produce": produce, "mutate": mutate, "verify": verify}[sys.argv[1]](request)
        print(json.dumps(dict(status="ok", result=result)))
    except Exception as error:
        print(json.dumps(dict(status="unavailable", error=type(error).__name__, reason=str(error))))
        raise SystemExit(1)
