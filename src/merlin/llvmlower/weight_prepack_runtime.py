"""Read-only stored arguments under the existing generated inference caller.

A manifest role is not an immutability proof. The source must have complete
pure tensor semantics, and the original session must not update the selected
arguments. The generated runtime stores these arguments in MERLIN_WEIGHT slots;
its state commit writes MERLIN_INPUT slots only. This analysis applies to that
caller, not arbitrary external entry-point clients or opaque implementations.
"""

from pathlib import Path

from merlin.common import mlir_query as mq
from merlin.common.paths import runtime_dir


def implementation_paths():
    """Original implementation dependencies for this particular caller proof."""
    from . import c_runtime, passes_xdsl, prepared_operand_effects

    return (
        Path(__file__),
        Path(c_runtime.__file__),
        Path(prepared_operand_effects.__file__),
        Path(passes_xdsl.__file__),
        runtime_dir() / "c" / "merlin_model.c",
    )


def fixed_arguments(source, manifest, *, entry="forward"):
    """Derive invocation-invariant stored slots; absent facts never grant one."""
    from xdsl.dialects import builtin, func

    from merlin.common.yaml import load_yaml

    from .passes_xdsl import lower_quant_ext
    from .prepared_operand_effects import validate_pure_tensor_function

    source = Path(source)
    module = mq.parse((source / "model.mlir").read_text())
    module.verify()
    # Inspect the same explicitly supported quantization semantics that ordinary
    # preprocessing lowers. Unsupported extensions remain opaque barriers.
    # This private analysis module never replaces the original source bundle.
    lower_quant_ext(module)
    module.verify()
    functions = {}
    for operation in module.body.block.ops:
        if not isinstance(operation, func.FuncOp):
            continue
        name = operation.sym_name.data
        if name in functions:
            raise ValueError("weight prepacking requires unambiguous root function symbols")
        functions[name] = operation
    if entry not in functions:
        raise ValueError("weight prepacking entry is not an exact root function")
    function = functions[entry]
    validate_pure_tensor_function(function, functions, allow_named_linalg=True)
    arguments = tuple(function.body.block.args)
    if any(not isinstance(argument.type, builtin.TensorType) for argument in arguments):
        raise ValueError("generated weight prepacking requires complete tensor entry arguments")
    updates = set()
    session_path = source / "session_contract.yaml"
    if session_path.exists():
        session = load_yaml(session_path)
        if type(session) is not dict or type(session.get("version")) is not int or session["version"] != 1:
            raise ValueError("generated weight prepacking supports only an original version-1 session")
        for family in ("states", "streams"):
            rows = session.get(family, [])
            if type(rows) is not list:
                raise ValueError("weight prepacking requires explicit original session update indices")
            for row in rows:
                if (
                    type(row) is not dict
                    or type(row.get("input_arg")) is not int
                    or not 0 <= row["input_arg"] < len(arguments)
                ):
                    raise ValueError("weight prepacking has an invalid original session update index")
                updates.add(row["input_arg"])
    fixed = []
    for key, item in manifest.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("weight prepacking requires canonical original manifest argument indices")
        index = int(key)
        if type(item) is not dict or not 0 <= index < len(arguments):
            raise ValueError("weight prepacking manifest is outside the original entry ABI")
        if (
            item.get("kind") in ("param", "buffer")
            and type(item.get("weight")) is str
            and item["weight"]
            and item.get("stub", False) is False
            and "error" not in item
            and index not in updates
        ):
            fixed.append(index)
    return tuple(sorted(fixed))
