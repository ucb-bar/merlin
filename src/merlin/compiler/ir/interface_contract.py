"""Public canonical Interface IR parsing and structural admission.

This leaf imports no source frontend, target implementation, compiler driver,
execution engine or independent evaluator. Host source preparation lives in
``merlin.compiler.lower.interface``. Backend semantic/resource admission is a
separate step over these canonical operations.
"""

from __future__ import annotations

import json
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp, StringAttr
from xdsl.ir import Attribute, ParametrizedAttribute
from xdsl.parser import Parser

from merlin.compiler.dialects import interface
from merlin.compiler.ir.readout import constant_parameter
from merlin.compiler.ir.xdsl_io import DenseFloatPrinter, IR_LOCK


class InterfaceConversionError(ValueError):
    """The complete source function has no admitted Interface IR conversion."""


BOUNDARY_ATTRIBUTE = "merlin.source_boundary"
NUMERICAL_ATTRIBUTE = "merlin.numerical_contract"
SOURCE_KIND = "merlin.interface.v1"


def interface_text(module: ModuleOp) -> str:
    """Canonical generic Interface text, preserving dense bits and a final LF.

    Content bindings use this representation rather than source spelling or
    platform newline conventions. Parsing/admission remain separate operations.
    """
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + "\n"


def _validate_attribute(attribute: Attribute) -> None:
    allowed = {*builtin.Builtin.attributes, *interface.INTERFACE_DIALECT.attributes}
    if type(attribute) not in allowed:
        raise InterfaceConversionError(f"canonical Interface IR contains unsupported attribute {attribute.name}")
    if isinstance(attribute, builtin.UnregisteredAttr):
        raise InterfaceConversionError("canonical Interface IR cannot contain unregistered attributes")
    if isinstance(attribute, DictionaryAttr):
        nested = attribute.data.values()
    elif isinstance(attribute, ArrayAttr):
        nested = attribute.data
    elif isinstance(attribute, ParametrizedAttribute):
        nested = attribute.parameters
    else:
        nested = ()
    for value in nested:
        _validate_attribute(value)


def _validate_numerical(attribute: Attribute) -> None:
    # This is the first explicit numerical contract admitted by this boundary.
    # Other arithmetic needs its own schema, never an ignored dynamic attribute.
    choices = {
        "lhs_signedness": {"signed"},
        "rhs_signedness": {"signed"},
        "accumulator_initialization": {"zero"},
        "accumulator_overflow": {"wrap"},
        "layout": {"dense_row_major"},
        "contraction": {"matmul", "batch_matmul"},
    }
    if not isinstance(attribute, DictionaryAttr) or set(attribute.data) != set(choices):
        raise InterfaceConversionError("Interface numerical contract has an unknown schema")
    if any(
        not isinstance(value, StringAttr) or value.data not in choices[key] for key, value in attribute.data.items()
    ):
        raise InterfaceConversionError("Interface numerical contract contains an unsupported value")


def validate_interface(module: ModuleOp, *, function: str | None = None) -> None:
    """Check the shared Interface grammar without choosing a target subset.

    Authored canonical modules need no source-conversion receipt. Provenance and
    receipts are observations; backend-specific semantic and resource admission
    must inspect the actual operations and numerical contract independently.
    """
    module.verify()
    allowed = {ModuleOp, arith.ConstantOp, *func.Func.operations, *interface.INTERFACE_DIALECT.operations}
    has_interface = False
    for operation in module.walk():
        if type(operation) not in allowed:
            raise InterfaceConversionError(f"canonical Interface IR does not admit {operation.name}")
        if type(operation) is arith.ConstantOp:
            try:
                constant_parameter(operation.result)
            except ValueError as exc:
                raise InterfaceConversionError(str(exc)) from exc
        has_interface |= operation.name.startswith("interface.")
        for name, attribute in operation.attributes.items():
            if name.startswith("prov."):
                pass
            elif name == NUMERICAL_ATTRIBUTE and isinstance(operation, interface.MatmulOp):
                _validate_numerical(attribute)
            elif name == BOUNDARY_ATTRIBUTE and operation is module:
                if not isinstance(attribute, StringAttr):
                    raise InterfaceConversionError("Interface source boundary must be a JSON string record")
                record = json.loads(attribute.data)
                if not isinstance(record, dict) or record.get("schema") != "merlin.source_function.v1":
                    raise InterfaceConversionError("Interface source boundary has an unknown record schema")
            else:
                raise InterfaceConversionError(f"{operation.name} has unhandled Interface attribute {name}")
            _validate_attribute(attribute)
        for attribute in operation.properties.values():
            _validate_attribute(attribute)
        for value in (*operation.operands, *operation.results):
            _validate_attribute(value.type)
        for region in operation.regions:
            for block in region.blocks:
                for argument in block.args:
                    _validate_attribute(argument.type)
    if not has_interface:
        raise InterfaceConversionError("canonical source contains no Interface operations")
    if function is not None:
        selected = [
            operation
            for operation in module.walk()
            if isinstance(operation, func.FuncOp) and operation.sym_name.data == function
        ]
        if len(selected) != 1:
            raise InterfaceConversionError(f"canonical Interface IR has no unique function {function!r}")


def parse_interface(text: str) -> ModuleOp:
    """Parse the canonical grammar, including only admitted static arith constants."""
    context = Context(allow_unregistered=False)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(func.Func)
    context.load_dialect(arith.Arith)
    context.load_dialect(interface.INTERFACE_DIALECT)
    with IR_LOCK:
        module = Parser(context, text).parse_module()
    validate_interface(module)
    return module
