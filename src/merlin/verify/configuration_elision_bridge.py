"""Independent complete straight-line MLIR projection for configuration elision.

Python parsing, selected ISA/effect resolution and the mapping from instruction
implementations to modeled state cells are explicit trust boundaries. Neither
producer reports nor declaration/review labels authorize a rewrite or consumer.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

from merlin.common.owned_process import run_owned
from merlin.common.paths import build_dir

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


class Unsupported(ValueError):
    """Explicitly unsupported control, operand or declaration representation."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": digest(data), "bytes": len(data)}


def run_reference(
    *, reference, facts, software, target, mode="resolve", source=None, qualification_identity=None, destination=None
):
    """Run only the actual pinned Python resolver/producer, retaining raw status.

    Read-only Git provenance checks are performed by the isolated worker; no Git
    state is changed. No Lean, MLIR executable, SDK, hardware or runtime is opened.
    """
    request = {
        "mode": mode,
        "reference": str(Path(reference).resolve()),
        "facts": str(Path(facts).resolve()),
        "software": str(Path(software).resolve()),
        "target": target,
    }
    if mode != "resolve":
        if type(source) is not str or len(source.encode()) > 1024 * 1024:
            raise Unsupported("bounded actual configuration source required")
        if type(qualification_identity) is not str or not qualification_identity.strip():
            raise Unsupported("producer caller premise label required; this does not approve it")
        request.update(source=source, qualification_identity=qualification_identity)
    directory = (
        Path(destination) if destination is not None else build_dir() / "configuration-elision" / uuid.uuid4().hex
    )
    directory.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).with_name("configuration_elision_producer.py")
    command = [sys.executable, "-I", str(worker)]
    (directory / "INTENT.json").write_text(
        canonical({"command": command, "request": request, "SDK_or_native_launch": False})
    )
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "LEAN_PATH"}}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        process = run_owned(
            command,
            input=canonical(request),
            cwd=worker.parent,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        for suffix, value in (("stdout", getattr(error, "stdout", "")), ("stderr", getattr(error, "stderr", ""))):
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            (directory / suffix).write_text(value or "")
        (directory / "PROCESS.json").write_text(
            canonical({"status": "unavailable", "error": str(error), "cleanup": getattr(error, "__notes__", [])})
        )
        raise Unsupported("configuration original resolver/producer unavailable; first raw output retained") from error
    (directory / "stdout").write_text(process.stdout)
    (directory / "stderr").write_text(process.stderr)
    (directory / "PROCESS.json").write_text(canonical({"returncode": process.returncode}))
    if process.returncode or process.stderr:
        raise Unsupported("configuration original resolver/producer did not succeed; first raw output retained")
    result = json.loads(process.stdout)
    if result["revision"] != REVISION or result["mode"] != mode or result["proof_authority"] is not False:
        raise Unsupported("incorrect original configuration producer response")
    for row in [result["facts"], result["software"], *result["source_refs"]]:
        actual = file_ref(row["path"])
        if any(actual[key] != row[key] for key in actual):
            raise Unsupported("original configuration sources changed after execution")
    return result


def _reserved_attributes(text):
    lexer = MLIRLexer(Input(text, "configuration-input"))
    previous = None
    while True:
        token = lexer.lex()
        if token.kind == MLIRTokenKind.EOF:
            return
        reserved = token.kind == MLIRTokenKind.BARE_IDENT and token.text == "op_name__"
        if token.kind == MLIRTokenKind.STRING_LIT and previous in {MLIRTokenKind.L_BRACE, MLIRTokenKind.COMMA}:
            reserved = token.kind.get_string_literal_value(token.span) == "op_name__"
        if reserved:
            raise Unsupported("reserved raw op_name__ attribute cannot be preserved by the xDSL parser")
        previous = token.kind


def parse(text):
    if type(text) is not str or len(text.encode()) > 1024 * 1024:
        raise Unsupported("bounded complete configuration MLIR required")
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, memref.MemRef):
        context.load_dialect(dialect)
    try:
        _reserved_attributes(text)
        module = Parser(context, text).parse_module()
        module.verify()
    except (ParseError, VerifyException) as error:
        raise Unsupported("configuration input is not a valid complete typed module") from error
    if len(tuple(module.walk())) > 4096:
        raise Unsupported("configuration operation budget exceeded")
    return module


def _attribute(value):
    stream = io.StringIO()
    Printer(stream=stream).print_attribute(value)
    return stream.getvalue()


def _name(op):
    if isinstance(op, builtin.UnregisteredOp):
        return op.op_name.data
    return op.name


def _attrs(op):
    return {
        key: _attribute(value)
        for key, value in sorted(op.attributes.items())
        if key != "op_name__" or not isinstance(op, builtin.UnregisteredOp)
    }


def _header(op):
    return {
        "name": _name(op),
        "attributes": _attrs(op),
        "properties": {key: _attribute(value) for key, value in sorted(op.properties.items())},
        "result_types": [_attribute(value.type) for value in op.results],
        "location": _attribute(op.location),
    }


def _constant(value):
    owner = value.owner
    if (
        type(owner) is not arith.ConstantOp
        or owner.attributes
        or set(owner.properties) != {"value"}
        or owner.operands
        or len(owner.results) != 1
        or owner.regions
        or owner.successors
        or type(owner.value) is not builtin.IntegerAttr
        or owner.value.type != value.type
        or type(value.type) not in (builtin.IntegerType, builtin.IndexType)
    ):
        raise Unsupported("configuration fields require actual undecorated integer constants")
    return owner.value.value.data


def _encoding(isa, instruction, values):
    arguments = {row["name"]: row for row in [*instruction["operands"], *instruction["immediates"]]}
    expected = {row["argument"] for row in instruction["fields"] if row["argument"] is not None}
    if set(values) != expected:
        raise Unsupported("configuration values do not exactly cover encoded arguments")
    word = 0
    for row in instruction["fields"]:
        if row["argument"] is None:
            value = row["fixed"]
        else:
            argument = arguments[row["argument"]]
            width = min(row["width"], argument["type"]["bits"])
            signed = argument["type"]["signed"]
            value = values[row["argument"]]
            lower = -(1 << (width - 1)) if signed else 0
            upper = (1 << (width - int(signed))) - 1
            if not lower <= value <= upper or value % argument["alignment"]:
                raise Unsupported("configuration operand is out of encoded range or misaligned")
        word |= (value & ((1 << row["width"]) - 1)) << row["offset"]
    matches = []
    for candidate in isa["instructions"]:
        if all(
            row["argument"] is not None or (word >> row["offset"]) & ((1 << row["width"]) - 1) == row["fixed"]
            for row in candidate["fields"]
        ):
            matches.append(candidate["name"])
    if matches != [instruction["name"]]:
        raise Unsupported("configuration instruction has no unique exact selected encoding")
    for row in instruction["fields"]:
        if row["argument"] is None:
            continue
        decoded = (word >> row["offset"]) & ((1 << row["width"]) - 1)
        if arguments[row["argument"]]["type"]["signed"] and decoded & (1 << (row["width"] - 1)):
            decoded -= 1 << row["width"]
        if decoded != values[row["argument"]]:
            raise Unsupported("configuration selected encoding changes an actual operand")
    return word


def _effect(op, isa, declarations):
    if type(op) is arith.ConstantOp:
        try:
            _constant(op.results[0])
        except (Unsupported, IndexError):
            return None, None
        return {"reads": [], "writes": [], "clobbers": [], "ordered": True, "pureWrite": False}, None
    instructions = {isa["namespace"] + "." + row["name"]: row for row in isa["instructions"]}
    instruction = instructions.get(_name(op))
    if instruction is None or instruction["results"] or op.results or op.regions or op.successors or op.properties:
        return None, None
    if len(op.operands) != len(instruction["operands"]):
        raise Unsupported("selected configuration instruction operand arity mismatch")
    if set(_attrs(op)) != {row["name"] for row in instruction["immediates"]}:
        return None, None  # additional metadata is never a pure setter
    values = {}
    for argument, operand in zip(instruction["operands"], op.operands, strict=True):
        if argument["type"]["kind"] != "integer" or operand.type != builtin.IntegerType(argument["type"]["bits"]):
            raise Unsupported("selected configuration operand type is unsupported or incorrect")
        value = _constant(operand)
        values[argument["name"]] = (
            value if argument["type"]["signed"] else value & ((1 << argument["type"]["bits"]) - 1)
        )
    for argument in instruction["immediates"]:
        value = op.attributes[argument["name"]]
        if (
            argument["type"]["kind"] != "integer"
            or type(value) is not builtin.IntegerAttr
            or value.type != builtin.IntegerType(argument["type"]["bits"])
        ):
            raise Unsupported("selected configuration immediate type is unsupported or incorrect")
        integer = value.value.data
        values[argument["name"]] = (
            integer if argument["type"]["signed"] else integer & ((1 << argument["type"]["bits"]) - 1)
        )
    word = _encoding(isa, instruction, values)
    row = declarations.get(instruction["name"])
    if row is None:
        return None, {"instruction": instruction["name"], "arguments": values, "word": word}
    resources = {item["name"]: item for item in isa["resources"]}

    def binding(item):
        if set(item) == {"constant"} and type(item["constant"]) is int:
            return item["constant"]
        if set(item) == {"argument"} and item["argument"] in values:
            return values[item["argument"]]
        raise Unsupported("configuration field binding is not an explicit selected integer")

    effect = {"reads": [], "writes": [], "clobbers": [], "ordered": row["ordered"], "pureWrite": row["pure_write"]}
    for kind in ("reads", "writes", "clobbers"):
        keys = set()
        for item in row[kind]:
            resource = resources[item["resource"]]
            if resource["kind"] != "state" or resource["unit"] != "fields" or resource["capacity"] is None:
                raise Unsupported("configuration field needs exact state resource capacity")
            index = binding(item["index"])
            if not 0 <= index < resource["capacity"]:
                raise Unsupported("configuration field index exceeds selected capacity")
            field = item["resource"] + ":" + str(index)
            if field in keys:
                raise Unsupported("decoded configuration fields alias or repeat")
            keys.add(field)
            effect[kind].append(field if kind == "clobbers" else [field, binding(item["value"])])
    if effect["pureWrite"] and (not effect["ordered"] or effect["reads"] or effect["clobbers"] or not effect["writes"]):
        raise Unsupported("pure configuration declaration has hidden reads/clobbers or missing writes")
    return effect, {"instruction": instruction["name"], "arguments": values, "word": word}


def _block_code(block, isa, declarations, defined):
    slots = {value: index for index, value in enumerate(block.args)}
    code = []
    for operation in block.ops:
        if operation.regions or operation.successors:
            raise Unsupported("configuration scope excludes nested regions and branches")
        if type(operation) is func.CallOp and operation.callee.root_reference.data in defined:
            raise Unsupported("internal calls require a separate derived call-graph bridge")
        if any(value not in slots for value in operation.operands):
            raise Unsupported("configuration SSA operand is out of scope or does not dominate")
        token = {**_header(operation), "operands": [slots[value] for value in operation.operands]}
        results = []
        for value in operation.results:
            slots[value] = len(slots)
            results.append(slots[value])
        token["results"] = results
        effect, decoded = _effect(operation, isa, declarations)
        code.append(
            {
                "token": canonical(token),
                "effect": effect,
                "eraseAllowed": effect is not None and effect["pureWrite"] and not operation.results,
                "decoded": decoded,
            }
        )
    return code


def _program(module, isa, declarations, *, allow_bare=False):
    if len(module.regions) != 1 or len(module.body.blocks) != 1:
        raise Unsupported("configuration module needs one complete top-level block")
    functions = tuple(module.body.block.ops)
    if any(type(op) is not func.FuncOp for op in functions):
        if not allow_bare or any(type(op) is func.FuncOp for op in functions) or module.body.block.args:
            raise Unsupported("configuration scope requires complete function-only or closed instruction modules")
        header = canonical({"module": _header(module), "scope": "complete-module-body"})
        return {
            "frame": header,
            "entries": [
                {"name": "<module>", "header": header, "code": _block_code(module.body.block, isa, declarations, set())}
            ],
        }
    names = [op.sym_name.data for op in functions]
    if len(names) != len(set(names)):
        raise Unsupported("configuration module function names must be unique")
    defined = {op.sym_name.data for op in functions if op.body.blocks}
    frame, entries = [], []
    for function in functions:
        if len(function.body.blocks) > 1:
            raise Unsupported("configuration scope excludes CFG and nested control")
        header = _header(function)
        header["declaration"] = not bool(function.body.blocks)
        frame.append(header)
        if not function.body.blocks:
            continue
        block = function.body.block
        header = {**header, "argument_types": [_attribute(value.type) for value in block.args]}
        code = _block_code(block, isa, declarations, defined)
        entries.append({"name": function.sym_name.data, "header": canonical(header), "code": code})
    return {"frame": canonical({"module": _header(module), "functions": frame}), "entries": entries}


def project(source, final, *, isa, effects, allow_bare=False):
    """Project both complete actual buffers independently under one field registry.

    This emits obligations even for compiler-valid wrong outputs. It never uses
    the producer's deletion plan and does not report host acceptance as proof.
    """
    before, after = (_program(parse(text), isa, effects, allow_bare=allow_bare) for text in (source, final))
    fields = sorted(
        {
            field
            for program in (before, after)
            for entry in program["entries"]
            for atom in entry["code"]
            if atom["effect"] is not None
            for field in [
                *(pair[0] for pair in atom["effect"]["reads"]),
                *(pair[0] for pair in atom["effect"]["writes"]),
                *atom["effect"]["clobbers"],
            ]
        }
    )
    if len(fields) > 256:
        raise Unsupported("configuration actual field registry exceeds the bounded domain")
    for program in (before, after):
        for entry in program["entries"]:
            for atom in entry["code"]:
                effect = atom["effect"]
                if effect is not None:
                    for kind in ("reads", "writes"):
                        effect[kind] = [[fields.index(field), value] for field, value in effect[kind]]
                    effect["clobbers"] = [fields.index(field) for field in effect["clobbers"]]
    source_entries = {entry["name"]: entry for entry in before["entries"]}
    for entry in before["entries"]:
        for index, atom in enumerate(entry["code"]):
            atom["id"] = index
    for entry in after["entries"]:
        original = source_entries.get(entry["name"], {"code": []})["code"]
        cursor = 0
        for index, atom in enumerate(entry["code"]):
            selected = next(
                (
                    i
                    for i in range(cursor, len(original))
                    if all(original[i][key] == atom[key] for key in ("token", "effect", "eraseAllowed"))
                ),
                None,
            )
            atom["id"] = len(original) + index if selected is None else selected
            if selected is not None:
                cursor = selected + 1
    return {
        "schema": "merlin.configuration-elision.projection.v1",
        "fields": fields,
        "before": before,
        "after": after,
        "source_sha256": digest(source),
        "final_sha256": digest(final),
        "projection_is_trusted_python": True,
        "hardware_model_approved": False,
        "proof_authority": False,
    }
