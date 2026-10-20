"""Complete typed input/output projection for ordinary private component runs.

Answers remain with the selected independent source owner. This shared binder
preserves source order, exact storage and the compiler's positional ABI names.
"""

import base64
import copy
import hashlib
import math
from pathlib import Path


class NativeComponentExecutionError(RuntimeError):
    """The explicit diagnostic route cannot establish its declared input joins."""


def _match(spec, emitted):
    from merlin.targetgen.contract.tensor_types import match_tensor_spec

    try:
        match_tensor_spec(spec, emitted)
    except ValueError as error:
        raise NativeComponentExecutionError(
            f"independent native input/output changes declared shape or dtype: {spec['name']}"
        ) from error


def _source_signature(source: Path, inputs: list, outputs: list) -> None:
    """Bind ordered static tensor types before accepting positional ABI names."""
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.func import FuncOp
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    from merlin.xdsl_dialects._common import make_context

    module = Parser(make_context(Arith, Tensor, Linalg, Math), source.read_text()).parse_module()
    module.verify()
    functions = list(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not FuncOp or not functions[0].body.blocks:
        raise NativeComponentExecutionError("positional independent native source has no unique tensor function")
    signature = functions[0].function_type
    for expected, observed in ((inputs, signature.inputs.data), (outputs, signature.outputs.data)):
        if len(expected) != len(observed):
            raise NativeComponentExecutionError("positional independent native source tensor arity differs")
        for spec, value_type in zip(expected, observed, strict=True):
            if not isinstance(value_type, TensorType):
                raise NativeComponentExecutionError("positional independent native source is not tensor-valued")
            _match(spec, {"shape": list(value_type.get_shape()), "dtype": str(value_type.get_element_type())})


def _bind(capsule: dict, cb: dict, source: Path, original_member=None) -> tuple[dict, dict, dict]:
    """Project independently selected inputs, retaining compiler and harness names."""
    from merlin.runtime.commandbuffer import whole_program_entry_bindings

    from . import capsule_golden as CG

    original_inputs, original_raws = None, None
    if original_member is not None:
        contract, tensors = original_member.contract_inputs()
        metadata = contract.verify()
        inputs, outputs = metadata["inputs"], metadata["outputs"]
        original_inputs = {
            tensor.name: {"shape": list(tensor.shape), "values": list(tensor.values())} for tensor in tensors
        }
        original_raws = {tensor.name: tensor.data for tensor in tensors}
    else:
        inputs = [row for row in capsule["inputs"] if row.get("role") in ("input", "weight", "bias")]
        outputs = [row for row in capsule["inputs"] if row.get("role") == "output"]
    if not outputs:
        outputs = (capsule.get("component_program") or {}).get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise NativeComponentExecutionError("independent native capsule has no complete typed source output roster")
    names = [row["name"] for row in inputs]
    output_names = [row["name"] for row in outputs]
    if len(set(names)) != len(names) or len(set(output_names)) != len(output_names):
        raise NativeComponentExecutionError("independent native input/output roster repeats a name")
    declared_order = ((capsule.get("operation") or {}).get("attributes") or {}).get("arg_order", names)
    if (
        not isinstance(declared_order, list)
        or len(declared_order) != len(set(declared_order))
        or set(declared_order) not in (set(names), set(names + output_names))
    ):
        raise NativeComponentExecutionError("independent native source argument order is incomplete")
    declared_order = [name for name in declared_order if name in names]
    inputs = [next(row for row in inputs if row["name"] == name) for name in declared_order]
    names = declared_order
    tensors = cb.get("tensors") or {}
    leaves = whole_program_entry_bindings(cb)
    if leaves is None:
        leaves = [name for name, spec in tensors.items() if spec.get("role") in ("input", "weight", "bias")]
    emitted_outputs = (cb.get("kernel_abi") or {}).get("outputs")
    if not isinstance(emitted_outputs, list) or len(emitted_outputs) != len(set(emitted_outputs)):
        raise NativeComponentExecutionError("independent native kernel output roster is absent or repeated")
    if len(leaves) != len(names) or len(emitted_outputs) != len(output_names):
        raise NativeComponentExecutionError("independent native ABI does not cover every input/output")
    positional = cb.get("operand_naming") == "positional" or cb.get("interface") == "linalg_positional"
    if positional:
        _source_signature(source, inputs, outputs)
    elif set(leaves) != set(names) or set(emitted_outputs) != set(output_names):
        raise NativeComponentExecutionError("independent native ABI changes named source inputs/outputs")
    else:
        leaves, emitted_outputs = names, output_names
    for spec, name in zip((*inputs, *outputs), (*leaves, *emitted_outputs), strict=True):
        if original_member is not None:
            from merlin.common.jsonio import canonical_json

            if canonical_json(spec["shape"]) != canonical_json((tensors.get(name) or {}).get("shape")):
                raise NativeComponentExecutionError("original candidate changed the exact source tensor shape")
        _match(spec, tensors.get(name))
    values = original_inputs if original_member is not None else CG.canonical_input_values(capsule, capsule["__dir__"])
    if names and not values:
        if CG.is_independent_float_golden(capsule, capsule["__dir__"]):
            raise NativeComponentExecutionError("independent floating source has no complete selected input projection")
        values = CG.materialized_input_values(capsule)
    if set(values) != set(names):
        raise NativeComponentExecutionError("independent native canonical input roster is incomplete")
    bound = copy.deepcopy(cb)
    projected, bindings = {}, []
    raws = original_raws if original_member is not None else CG.canonical_input_raws(capsule, capsule["__dir__"])
    for spec, name in zip(inputs, leaves, strict=True):
        value = values[spec["name"]]
        if value.get("shape") != spec["shape"] or len(value.get("values", [])) != math.prod(spec["shape"]):
            raise NativeComponentExecutionError("independent native canonical input values have the wrong shape")
        projected[name] = value
        raw = raws.get(spec["name"])
        if raw is not None:
            bound["tensors"][name]["preload_b64"] = base64.b64encode(raw).decode()
        bindings.append(
            {
                "source": spec["name"],
                "harness": name,
                "shape": spec["shape"],
                "dtype": spec["dtype"],
                "raw_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            }
        )
    return bound, projected, {"inputs": bindings, "outputs": dict(zip(output_names, emitted_outputs, strict=True))}
