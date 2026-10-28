"""Read-only stored arguments under the existing generated inference caller.

A manifest role is not an immutability proof. The source must have complete
pure tensor semantics, and the original session must not update the selected
arguments. The generated runtime stores these arguments in MERLIN_WEIGHT slots;
its state commit writes MERLIN_INPUT slots only. This analysis applies to that
caller, not arbitrary external entry-point clients or opaque implementations.
"""

from functools import lru_cache
from pathlib import Path

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir


def implementation_paths():
    """Original implementation dependencies for this particular caller proof."""
    # Sibling paths, not imports: these files are digested, never executed here, and importing the
    # C runtime would pull the whole host lowering into every reader of this analysis.
    here = Path(__file__).parent
    return (
        Path(__file__),
        here / "c_runtime.py",
        here / "prepared_operand_effects.py",
        here / "passes_xdsl.py",
        runtime_dir() / "c" / "merlin_model.c",
    )


@lru_cache(maxsize=16)
def _tensor_entry_arity(source_path, source_sha256, entry, lower_quant_ext, validate_pure_tensor_function):
    """Retain only a pure tensor entry's arity for these exact source bytes.

    Each caller rechecks file contents before and after the lookup. The bounded
    process-local cache owns no IR and grants no manifest/session authority.
    The selected analysis implementations participate in its identity too.
    Failed purity checks are never cached.
    """
    from xdsl.dialects import builtin, func

    # The file reader rechecks actual source bytes; quant lowering below edits
    # only this owned clone, never the shared read-only parse cache entry.
    module = mq.parse(Path(source_path)).clone()
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
    if sha256_file(source_path) != source_sha256:
        raise ValueError("weight prepacking source changed during ownership analysis")
    return len(arguments)


def fixed_arguments(source, manifest, *, entry="forward"):
    """Derive invariant stored slots from current source, manifest and session.

    Exact-source tensor purity may be reused; manifest roles and state/stream
    updates are interpreted anew on every call, including cache hits.
    """
    from merlin.common.yaml import load_yaml

    from .passes_xdsl import lower_quant_ext
    from .prepared_operand_effects import validate_pure_tensor_function

    source = Path(source)
    model_path = source / "model.mlir"
    digest = sha256_file(model_path)
    arity = _tensor_entry_arity(
        str(model_path.resolve()), digest, entry, lower_quant_ext, validate_pure_tensor_function
    )
    if sha256_file(model_path) != digest:
        raise ValueError("weight prepacking source changed during ownership analysis")
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
                    or not 0 <= row["input_arg"] < arity
                ):
                    raise ValueError("weight prepacking has an invalid original session update index")
                updates.add(row["input_arg"])
    fixed = []
    for key, item in manifest.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("weight prepacking requires canonical original manifest argument indices")
        index = int(key)
        if type(item) is not dict or not 0 <= index < arity:
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
