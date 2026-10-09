"""No-data linkage checks for the ordinary compiled pointer entry ABI.

These declarations and products prove only structural compilation. The caller
owns independently derived original IR/ABI provenance and semantic, index,
resource and output-store proofs. No tensor values or goldens enter this owner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .harness_blobs import _symbol_name


@dataclass(frozen=True)
class CompileOnlyTensor:
    """One original tensor slot, derived by the protected source producer."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def record(self):
        if (
            type(self.name) is not str
            or not self.name
            or type(self.shape) is not tuple
            or any(type(size) is not int or size < 0 for size in self.shape)
            or type(self.dtype) is not str
            or not self.dtype
        ):
            raise ValueError("compile-only original ABI needs named static tensor types")
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class CompileOnlySourceAbi:
    """Immutable original roster, including zero inputs; not proof authority."""

    inputs: tuple[CompileOnlyTensor, ...]
    outputs: tuple[CompileOnlyTensor, ...]

    def record(self):
        for role, slots in (("input", self.inputs), ("output", self.outputs)):
            if (
                type(slots) is not tuple
                or (role == "output" and not slots)
                or any(type(slot) is not CompileOnlyTensor for slot in slots)
            ):
                raise ValueError("compile-only source ABI needs complete immutable input/output slots")
            if len({slot.name for slot in slots}) != len(slots):
                raise ValueError("compile-only source ABI repeats an original input/output")
        return {"inputs": [slot.record() for slot in self.inputs], "outputs": [slot.record() for slot in self.outputs]}

    def bind(self, cb):
        from merlin.runtime.commandbuffer import whole_program_entry_bindings

        from .tensor_types import match_tensor_spec

        self.record()
        tensors, kernel = cb.get("tensors"), cb.get("kernel_abi")
        if type(tensors) is not dict or type(kernel) is not dict or kernel.get("kind") != "whole_program":
            raise ValueError("compile-only candidate has no whole-program pointer ABI")
        args, outputs = kernel.get("args"), kernel.get("outputs")
        if type(args) is not list or not args or type(outputs) is not list or len(outputs) != len(set(outputs)):
            raise ValueError("compile-only candidate omits or repeats an ABI argument/output")
        slots = {}
        for argument in args:
            if (
                type(argument) is not dict
                or set(argument) != {"tensor", "access"}
                or argument["access"] not in {"read", "write", "readwrite"}
                or type(argument["tensor"]) is not str
                or argument["tensor"] in slots
            ):
                raise ValueError("compile-only candidate has an ambiguous pointer slot")
            slots[argument["tensor"]] = argument["access"]
        leaves = whole_program_entry_bindings(cb)
        if leaves is None:
            leaves = [name for name, access in slots.items() if access != "write"]
        if (
            len(leaves) != len(self.inputs)
            or len(outputs) != len(self.outputs)
            or set(leaves) != {name for name, access in slots.items() if access != "write"}
            or set(outputs) != {name for name, access in slots.items() if access != "read"}
        ):
            raise ValueError("compile-only candidate does not cover every original input/output")
        positional = cb.get("operand_naming") == "positional" or cb.get("interface") == "linalg_positional"
        if not positional:
            if set(leaves) != {slot.name for slot in self.inputs} or set(outputs) != {
                slot.name for slot in self.outputs
            }:
                raise ValueError("compile-only candidate changes named original inputs/outputs")
            leaves, outputs = [slot.name for slot in self.inputs], [slot.name for slot in self.outputs]
        bindings = []
        for role, originals, emitted in (("input", self.inputs, leaves), ("output", self.outputs, outputs)):
            for original, name in zip(originals, emitted, strict=True):
                match_tensor_spec(original.record(), tensors.get(name))
                bindings.append({"role": role, "source": original.name, "emitted": name})
        return {"original_abi": self.record(), "bindings": bindings, "pointer_arity": len(args)}


@dataclass(frozen=True)
class CompileOnlyLinkage:
    """A host-created retained reference with no operand storage or execution."""

    entry_symbol: str
    pointer_arity: int
    command_buffer_sha256: str

    def verify(self, cb):
        if (
            not _symbol_name(self.entry_symbol)
            or type(self.pointer_arity) is not int
            or self.pointer_arity <= 0
            or self.pointer_arity != len((cb.get("kernel_abi") or {}).get("args") or ())
            or self.command_buffer_sha256
            != hashlib.sha256(json.dumps(cb, sort_keys=True, allow_nan=False).encode()).hexdigest()
        ):
            raise ValueError("compile-only linkage differs from its exact emitted pointer declaration")

    def render(self, cb):
        self.verify(cb)
        parameters = ", ".join("void *" for _ in range(self.pointer_arity))
        arguments = ", ".join("(void *)0" for _ in range(self.pointer_arity))
        # An opaque volatile gate keeps the ordinary kernel and its dependencies
        # reachable through linker garbage collection. The probe is never run.
        return (
            f"extern void {self.entry_symbol}({parameters});\n"
            "static volatile unsigned char merlin_compile_only_gate;\n"
            "int main(void) {\n"
            f"  if (merlin_compile_only_gate) {self.entry_symbol}({arguments});\n"
            "  return 0;\n}\n"
        )


def require_pointer_entry(lowered_mlir, *, entry_symbol, pointer_arity):
    """Require the actual plain C pointer signature used by the shared harness.

    This checks the emitted declaration, not source equivalence, body effects,
    device placement or runtime support. Opaque operations remain unqualified;
    the ordinary stock translator still verifies its complete input artifact.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import NoneAttr
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    if not _symbol_name(entry_symbol) or type(pointer_arity) is not int or pointer_arity < 1:
        raise ValueError("emitted entry requires an explicit symbol and positive pointer ABI arity")
    try:
        module = Parser(make_llvm_context(), lowered_mlir).parse_module()
        module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("emitted entry has malformed LLVM pointer ABI IR") from error
    entries = [
        operation
        for operation in module.walk()
        if isinstance(operation, llvm.FuncOp) and operation.sym_name.data == entry_symbol
    ]
    if len(entries) != 1:
        raise ValueError("emitted LLVM has no unique selected entry")
    entry = entries[0]
    if (
        not entry.body.blocks
        or entry.function_type.is_variadic
        or type(entry.function_type.output) is not llvm.LLVMVoidType
        or entry.CConv.convention.data != "ccc"
        or entry.linkage.linkage.data != "external"
        or len(entry.function_type.inputs) != pointer_arity
        or tuple(value.type for value in entry.body.blocks[0].args) != tuple(entry.function_type.inputs)
        or any(
            type(value) is not llvm.LLVMPointerType or type(value.addr_space) is not NoneAttr
            for value in entry.function_type.inputs
        )
        or any(attributes.data for attributes in entry.arg_attrs or ())
        or any(attributes.data for attributes in entry.res_attrs or ())
    ):
        raise ValueError("emitted entry does not define the selected ordinary C pointer ABI")


def prepare_linkage(*, cb, lowered_mlir, entry_symbol, original_abi):
    """Check pointer ABI structure; stock translation still verifies LLVM semantics."""
    if type(original_abi) is not CompileOnlySourceAbi:
        raise ValueError("compile-only linkage requires a typed independently derived original ABI")
    bindings = original_abi.bind(cb)
    require_pointer_entry(lowered_mlir, entry_symbol=entry_symbol, pointer_arity=bindings["pointer_arity"])
    linkage = CompileOnlyLinkage(
        entry_symbol,
        bindings["pointer_arity"],
        hashlib.sha256(json.dumps(cb, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    )
    linkage.verify(cb)
    return linkage, bindings
