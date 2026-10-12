"""Typed instruction dialects and bit codecs from resolved, declarative ISA data.

Generation supplies neither tensor lowering nor a preferred device schedule.
Instruction meanings are preserved as declarations, not executable reference code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Dialect, Operation
from xdsl.traits import EffectInstance, MemoryEffect, MemoryEffectKind, Resource
from xdsl.utils.exceptions import VerifyException

import merlin.targetgen.isa_spec as isa_spec
from merlin.targetgen.isa_spec import (
    Argument,
    Instruction,
    IsaSpecError,
    ResolvedISA,
    fixed_encoding,
    resolved_isa_from_record,
)


def _check_value(argument: Argument, value: int, width: int | None = None) -> None:
    if type(value) is not int:
        raise IsaSpecError(f"{argument.name}: expected an integer encoding value")
    bits = min(argument.type.bits, width) if width is not None else argument.type.bits
    lower = -(1 << (bits - 1)) if argument.type.signed else 0
    upper = (1 << (bits - int(argument.type.signed))) - 1
    if not lower <= value <= upper or value % argument.alignment:
        raise IsaSpecError(f"{argument.name}: encoding value is outside range or misaligned")


def encode_instruction(model: ResolvedISA, name: str, values: dict[str, int]) -> int:
    """Encode explicitly bound operand/result/immediate fields, after allocation.

    SSA values are not register indices or addresses automatically. The caller
    supplies the encoded bindings after its own target lowering/allocation.
    """
    instruction = model.instruction(name)
    expected = {field.argument for field in instruction.fields if field.argument is not None}
    if set(values) != expected:
        raise IsaSpecError("encoding bindings must exactly cover declared variable fields")
    arguments = {arg.name: arg for arg in instruction.arguments()}
    _, word = fixed_encoding(instruction)
    for field in instruction.fields:
        if field.argument is not None:
            value = values[field.argument]
            _check_value(arguments[field.argument], value, field.width)
            word |= (value & ((1 << field.width) - 1)) << field.offset
    return word


def decode_instruction(model: ResolvedISA, word: int) -> tuple[str, dict[str, int]]:
    """Decode only legal fixed fields and exactly representable aligned bindings."""
    if type(word) is not int or not 0 <= word < 1 << model.word_bits:
        raise IsaSpecError("instruction word is outside the declared width")
    matches = []
    for instruction in model.instructions:
        mask, value = fixed_encoding(instruction)
        if word & mask == value:
            matches.append(instruction)
    if len(matches) != 1:
        raise IsaSpecError("instruction word has no unique declared decoding")
    instruction = matches[0]
    arguments = {arg.name: arg for arg in instruction.arguments()}
    values = {}
    for field in instruction.fields:
        if field.argument is None:
            continue
        argument = arguments[field.argument]
        value = (word >> field.offset) & ((1 << field.width) - 1)
        if argument.type.signed and value & (1 << (field.width - 1)):
            value -= 1 << field.width
        _check_value(argument, value, field.width)
        values[field.argument] = value
    return instruction.name, values


@dataclass(frozen=True)
class InstructionResource(Resource):
    """A declared resource identity, including state beyond tensor memory."""

    identity: str

    def name(self) -> str:
        return self.identity


def instruction_effect_trait(model: ResolvedISA, name: str) -> MemoryEffect:
    effects = json.loads(model.instruction(name).effects_json)

    class DeclaredInstructionEffects(MemoryEffect):
        @classmethod
        def get_effects(cls, op: Operation) -> frozenset[EffectInstance] | None:
            if effects.get("status") == "unknown":
                return None
            # All instances share the same namespace-qualified resource identity;
            # per-invocation alias claims would need additional independent evidence.
            return frozenset(
                EffectInstance(kind, resource=InstructionResource(f"{model.namespace}:{resource}"))
                for key, kind in (("reads", MemoryEffectKind.READ), ("writes", MemoryEffectKind.WRITE))
                for resource in effects[key]
            )

    return DeclaredInstructionEffects()


def verify_instruction(op: Operation, instruction: Instruction) -> None:
    """Check immediate types, values and encoding constraints; SSA allocation is separate."""
    fields = {field.argument: field for field in instruction.fields if field.argument is not None}
    for immediate in instruction.immediates:
        value = op.attributes.get(immediate.name)
        if not isinstance(value, IntegerAttr) or value.type != IntegerType(immediate.type.bits):
            raise VerifyException(f"{immediate.name}: expected the declared integer attribute width")
        field = fields.get(immediate.name)
        # xDSL canonicalizes signless integer attributes to signed representatives.
        # The admitted ISA signature determines the interpretation of those bits.
        integer = value.value.data
        if not immediate.type.signed:
            integer &= (1 << immediate.type.bits) - 1
        try:
            _check_value(immediate, integer, field.width if field is not None else None)
        except IsaSpecError as exc:
            raise VerifyException(str(exc)) from exc


def render_dialect_module(model: ResolvedISA) -> str:
    """Generate concrete xDSL definitions; identifiers come from validated declarations."""
    model = resolved_isa_from_record(model.record())
    lines = [
        '"""Generated instruction definitions; no tensor lowering or schedule."""',
        "from xdsl.dialects.builtin import IntAttr, IntegerAttr, IntegerType, StringAttr",
        "from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute",
        "from xdsl.irdl import (IRDLOperation, attr_def, operand_def, result_def, traits_def,",
        "                       irdl_attr_definition, irdl_op_definition)",
        "from merlin.targetgen.isa_spec import resolved_isa_from_record",
        "from merlin.targetgen.isa_dialect_gen import (instruction_effect_trait, verify_instruction,",
        "                                             encode_instruction, decode_instruction)",
        "",
        f"MODEL = resolved_isa_from_record({model.record()!r})",
        f"MODEL_SHA256 = {model.identity()!r}",
        "",
        "@irdl_attr_definition",
        "class AddressType(ParametrizedAttribute, TypeAttribute):",
        f"    name = {f'{model.namespace}.address'!r}",
        "    space: StringAttr",
        "    bits: IntAttr",
        "",
    ]

    def type_expression(arg: Argument) -> str:
        if arg.type.kind == "integer":
            return f"IntegerType({arg.type.bits})"
        return f"AddressType(StringAttr({arg.type.space!r}), IntAttr({arg.type.bits}))"

    classes = []
    for index, instruction in enumerate(model.instructions):
        class_name = f"Instruction{index}"
        classes.append(class_name)
        lines.extend(
            [
                "@irdl_op_definition",
                f"class {class_name}(IRDLOperation):",
                f"    name = {f'{model.namespace}.{instruction.name}'!r}",
            ]
        )
        for ordinal, argument in enumerate(instruction.operands):
            lines.append(f"    operand_{ordinal} = operand_def({type_expression(argument)})")
        for ordinal, argument in enumerate(instruction.results):
            lines.append(f"    result_{ordinal} = result_def({type_expression(argument)})")
        # Attribute names stay the declared semantic names. Prefix Python member
        # names so an instruction may legally use an immediate named 'name'.
        for ordinal, argument in enumerate(instruction.immediates):
            lines.append(f"    immediate_{ordinal} = attr_def(IntegerAttr, attr_name={argument.name!r})")
        lines.extend(
            [
                f"    traits = traits_def(instruction_effect_trait(MODEL, {instruction.name!r}))",
                "",
                "    def verify_(self):",
                f"        verify_instruction(self, MODEL.instruction({instruction.name!r}))",
                "",
            ]
        )
    lines.extend(
        [
            "INSTRUCTIONS = {"
            + ", ".join(f"{ins.name!r}: {cls}" for ins, cls in zip(model.instructions, classes, strict=True))
            + "}",
            f"DIALECT = Dialect({model.namespace!r}, [{', '.join(classes)}], [AddressType])",
            "",
            "def encode(name, values):",
            "    return encode_instruction(MODEL, name, values)",
            "",
            "def decode(word):",
            "    return decode_instruction(MODEL, word)",
            "",
        ]
    )
    return "\n".join(lines)


def build_dialect(model: ResolvedISA) -> Dialect:
    """Build generated definitions in memory through exactly the same source emitter."""
    namespace: dict = {"__name__": f"generated_{model.namespace}_instructions"}
    exec(compile(render_dialect_module(model), "<generated-isa-dialect>", "exec"), namespace)
    return namespace["DIALECT"]


def generate_package(model: ResolvedISA, out: Path) -> dict:
    """Publish source + resolved semantics/effects with exact product identities.

    Existing products must have identical bytes; an output directory never silently
    acquires a different instruction model. This is not a grading/proof receipt.
    """
    registry = model.record()
    registry["identity"] = model.identity()
    contents = {
        "instruction_dialect.py": render_dialect_module(model).encode(),
        "isa-resolved.json": (json.dumps(registry, sort_keys=True, indent=2) + "\n").encode(),
    }
    manifest = {
        "schema": "merlin.isa_generation.v1",
        "target": model.target,
        "model_sha256": model.identity(),
        "facts_sha256": model.facts_sha256,
        "declaration_sha256": model.declaration_sha256,
        "generator": {
            "modules": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in ((isa_spec.__name__, isa_spec.__file__), (__name__, __file__))
            },
            "xdsl_version": version("xdsl"),
        },
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()},
        "assurance": "declarations_and_encodings_only",
    }
    contents["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    out = Path(out)
    if out.is_symlink():
        raise IsaSpecError("output directory must not be a symbolic link")
    for name, raw in contents.items():
        path = out / name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != raw)):
            raise IsaSpecError("output already contains a different generated instruction model")
    out.mkdir(parents=True, exist_ok=True)
    for name, raw in contents.items():
        path = out / name
        if not path.exists():
            try:
                with path.open("xb") as stream:
                    stream.write(raw)
            except FileExistsError:
                if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                    raise IsaSpecError("output changed during ISA generation") from None
    return manifest
