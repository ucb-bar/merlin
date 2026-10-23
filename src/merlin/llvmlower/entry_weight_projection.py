"""Explicit projection of unused immutable weights from a generated entry ABI.

The prepared tensor SSA, complete argument table and generated caller contract
are separate obligations. Only whole original stored or generated-zero parameters
with no uses after conservative pure DCE are removed. Model inputs, state,
results and prepared trailing arguments retain their order and semantics.
Unknown effects and symbolic escapes remain live or refuse before mutation.
This is not an LLVM liveness heuristic, a default policy or a cycle model.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from io import StringIO
from pathlib import Path

from xdsl.dialects import func
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FunctionType,
    ModuleOp,
    NoneAttr,
    SymbolRefAttr,
    TensorType,
)
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.ops import YieldOp
from xdsl.ir import Attribute, ParametrizedAttribute
from xdsl.printer import Printer
from xdsl.traits import IsTerminator, SymbolOpInterface, get_effects

RECEIPT = "entry_weight_projection.json"


@dataclass(frozen=True)
class GeneratedDispatchABI:
    """Caller declarations, not facts inferred from an entry's spelling.

    Every out-of-module caller must be regenerated from the projected table.
    Unknown external callers or address observations cannot use this contract.
    Arithmetic DCE needs the same nontrapping/unobserved effects as upstream
    ordinary pure tensor optimization; no additional numerical tolerance is used.
    Packed weight spans must be read-only and their addresses unobserved. This
    avoids changing observable alias or pointer identity when live spans move.
    """

    entry_symbol: str
    compiler_generated_dispatch_only: bool
    entry_address_unobserved: bool
    nontrapping_arithmetic: bool
    arithmetic_flags_unobserved: bool
    generated_weight_storage_readonly: bool = False
    weight_addresses_unobserved: bool = False


class ArgumentOwnership(StrEnum):
    IMMUTABLE_STORED_WEIGHT = "immutable_stored_weight"
    IMMUTABLE_GENERATED_ZERO_WEIGHT = "immutable_generated_zero_weight"
    RETAINED_CAPTURE = "retained_capture"
    PREPARED_WEIGHT = "prepared_weight"


@dataclass(frozen=True)
class ArgumentBinding:
    original_index: int
    kind: ArgumentOwnership
    shape: tuple[int, ...]
    dtype: str
    weight_name: str | None = None


@dataclass(frozen=True)
class FileIdentity:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class OptionalFileIdentity:
    """Actual optional source state, including checked absence at its owner path."""

    path: str
    present: bool
    bytes: int | None
    sha256: str | None


@dataclass(frozen=True)
class EntryArgumentTable:
    entries: tuple[ArgumentBinding, ...]
    captured_arguments: int
    source_files: tuple[FileIdentity, ...]
    result_types: tuple[str, ...] = ()
    optional_source_files: tuple[OptionalFileIdentity, ...] = ()

    @property
    def identity(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class EntryWeightProjection:
    contract: GeneratedDispatchABI
    argument_table: EntryArgumentTable
    input_module_sha256: str
    original_function_type: str
    dce_module_sha256: str
    projected_module_sha256: str
    removed_original_indices: tuple[int, ...]
    retained_original_indices: tuple[int, ...]
    original_to_projected: tuple[int | None, ...]
    erased_operations: tuple[tuple[str, int], ...]
    direct_calls_rewritten: int


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def module_text(module: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _module_digest(module: ModuleOp) -> str:
    return sha256(module_text(module).encode()).hexdigest()


def _pin(path: Path) -> FileIdentity:
    from merlin.common.digest import sha256_file

    return FileIdentity(str(path.resolve()), path.stat().st_size, sha256_file(path))


def _optional_pin(path: Path) -> OptionalFileIdentity:
    # Keep the owner-side lexical filename so creating or retargeting an optional
    # symlink is checked too. A dangling symlink or directory is not absence.
    path = path.absolute()
    if not path.exists() and not path.is_symlink():
        return OptionalFileIdentity(str(path), False, None, None)
    if not path.is_file():
        raise ValueError("optional argument source must be a regular file or absent")
    identity = _pin(path)
    return OptionalFileIdentity(str(path), True, identity.bytes, identity.sha256)


def derive_entry_argument_table(model_dir: Path | str, prepared_dir: Path | str) -> EntryArgumentTable:
    """Derive the table shared by normal preparation and C runtime generation.

    The original capture, quant-inner lift and build-time quant-hoist all take
    part in the identity. Only stored or generated-zero ``param`` entries are candidates;
    buffers, inputs and all trailing preparation entries are retained.
    Files are read, never copied or rewritten. No caller may fabricate a shorter
    table from a final LLVM signature.
    """
    from merlin.common import mlir_query

    from . import qinner, quant_hoist
    from .model_runner import parse_forward_signature
    from .weights_pack import load_safetensors_header

    model_dir, prepared_dir = Path(model_dir), Path(prepared_dir)
    initial_capture = tuple(_pin(model_dir / name) for name in ("model.mlir", "weights.safetensors.manifest.json"))
    optional_paths = (
        model_dir / "session_contract.yaml",
        model_dir / "weights.safetensors",
        prepared_dir / quant_hoist.PLAN_FILE,
        prepared_dir / quant_hoist.VALUES_FILE,
    )
    optional_before = tuple(_optional_pin(path) for path in optional_paths)
    signature = parse_forward_signature(model_dir / "model.mlir")
    manifest = json.loads((model_dir / "weights.safetensors.manifest.json").read_text())
    if set(manifest) != {str(i) for i in range(len(signature))}:
        raise ValueError("entry projection requires an exact captured argument manifest")
    session_path = model_dir / "session_contract.yaml"
    if session_path.is_file():
        from merlin.common.schemas import validate_or_raise
        from merlin.common.yaml import load_yaml

        session = load_yaml(session_path)
        validate_or_raise(session, "session_contract")
        for role in ("states", "streams"):
            for binding in session.get(role, ()) or ():
                index = binding["input_arg"]
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(signature):
                    raise ValueError("session argument ownership is outside the captured signature")
                # The immutable-weight contract cannot authorize writes to a
                # weight descriptor or infer a mutable slot from manifest kind.
                # Ordinary model inputs retain their ABI and session remap.
                if manifest[str(index)].get("kind") != "input":
                    raise ValueError("session-owned weight storage cannot use immutable entry projection")
    weights = model_dir / "weights.safetensors"
    header, _ = load_safetensors_header(weights) if weights.is_file() else ({}, 0)
    entries = []
    for index, (shape, dtype) in enumerate(signature):
        meta = manifest[str(index)]
        if "stub" in meta and not isinstance(meta["stub"], bool):
            raise ValueError("weight stub ownership must be a literal manifest boolean")
        parameter = meta.get("kind") == "param"
        immutable = parameter and not meta.get("stub")
        generated_zero = parameter and meta.get("stub") is True
        weight = meta.get("weight")
        if parameter and (not isinstance(weight, str) or not weight):
            raise ValueError("immutable weight ownership needs a manifest weight name")
        if immutable:
            if weight not in header:
                raise ValueError("immutable parameter has no stored weight")
            stored = header[weight]
            stored_dtype = {
                "F32": "f32",
                "F64": "f64",
                "F16": "f16",
                "BF16": "bf16",
                "I64": "i64",
                "I32": "i32",
                "I16": "i16",
                "I8": "i8",
                "BOOL": "i1",
            }
            if tuple(stored.get("shape", ())) != tuple(shape) or stored_dtype.get(stored.get("dtype")) != dtype:
                raise ValueError("immutable parameter storage/type disagreement")
        ownership = (
            ArgumentOwnership.IMMUTABLE_STORED_WEIGHT
            if immutable
            else ArgumentOwnership.IMMUTABLE_GENERATED_ZERO_WEIGHT
            if generated_zero
            else ArgumentOwnership.RETAINED_CAPTURE
        )
        entries.append(ArgumentBinding(index, ownership, tuple(shape), dtype, weight if parameter else None))
    inner = qinner.plan_for_bundle(model_dir / "model.mlir")
    hoisted = quant_hoist.read_plan(prepared_dir)
    for argument in (*inner, *hoisted):
        entries.append(
            ArgumentBinding(len(entries), ArgumentOwnership.PREPARED_WEIGHT, tuple(argument.shape), argument.dtype)
        )
    paths = [model_dir / "model.mlir", model_dir / "weights.safetensors.manifest.json"]
    if session_path.is_file():
        paths.append(session_path)
    if weights.is_file():
        paths.append(weights)
    if inner:
        paths.append(model_dir / "extra.npz")
    if hoisted:
        paths.extend((prepared_dir / quant_hoist.PLAN_FILE, prepared_dir / quant_hoist.VALUES_FILE))
    captured_entry = _entry(mlir_query.parse(model_dir / "model.mlir"), "forward")
    identities = tuple(_pin(path) for path in paths)
    if identities[:2] != initial_capture or tuple(_optional_pin(path) for path in optional_paths) != optional_before:
        raise ValueError("entry argument source changed during ownership derivation")
    return EntryArgumentTable(
        tuple(entries),
        len(signature),
        identities,
        tuple(str(result) for result in captured_entry.function_type.outputs),
        optional_before,
    )


def recheck_argument_table(table: EntryArgumentTable) -> None:
    if tuple(entry.original_index for entry in table.entries) != tuple(range(len(table.entries))):
        raise ValueError("entry argument table must have contiguous original indices")
    if not 0 <= table.captured_arguments <= len(table.entries):
        raise ValueError("invalid captured argument count")
    for index, entry in enumerate(table.entries):
        if not isinstance(entry.kind, ArgumentOwnership):
            raise ValueError("unknown entry argument ownership")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in entry.shape):
            raise ValueError("entry projection requires static argument shapes")
        if entry.kind in {
            ArgumentOwnership.IMMUTABLE_STORED_WEIGHT,
            ArgumentOwnership.IMMUTABLE_GENERATED_ZERO_WEIGHT,
        } and (index >= table.captured_arguments or not entry.weight_name):
            raise ValueError("only original immutable parameter ownership can permit removal")
    for identity in table.source_files:
        if _pin(Path(identity.path)) != identity:
            raise ValueError("entry argument source changed after planning")
    for identity in table.optional_source_files:
        if (
            not isinstance(identity, OptionalFileIdentity)
            or type(identity.present) is not bool
            or not Path(identity.path).is_absolute()
            or not identity.present
            and (identity.bytes is not None or identity.sha256 is not None)
        ):
            raise ValueError("typed optional argument source identity required")
        if _optional_pin(Path(identity.path)) != identity:
            raise ValueError("optional entry argument source presence or bytes changed after planning")


def _entry(module, symbol):
    matches = [
        operation
        for operation in module.body.block.ops
        if isinstance(operation, func.FuncOp) and operation.sym_name.data == symbol
    ]
    if len(matches) != 1 or len(matches[0].body.blocks) != 1:
        raise ValueError("projection requires one defined single-block entry")
    entry = matches[0]
    if "arg_attrs" in entry.properties or "res_attrs" in entry.properties:
        raise ValueError("entry argument/result ABI attributes require a separate remapping proof")
    return entry


def _symbols(attribute: Attribute):
    if isinstance(attribute, SymbolRefAttr):
        yield attribute
    elif isinstance(attribute, DictionaryAttr):
        for child in attribute.data.values():
            yield from _symbols(child)
    elif isinstance(attribute, ArrayAttr):
        for child in attribute:
            yield from _symbols(child)
    elif isinstance(attribute, ParametrizedAttribute):
        for child in attribute.parameters:
            yield from _symbols(child)


def _direct_callers(module, entry):
    calls = []
    for operation in module.walk():
        refs = [
            reference
            for attribute in (*operation.attributes.values(), *operation.properties.values())
            for reference in _symbols(attribute)
            if reference.root_reference.data == entry.sym_name.data
        ]
        if not refs:
            continue
        if (
            not isinstance(operation, func.CallOp)
            or refs != [operation.callee]
            or operation.callee.nested_references.data
        ):
            raise ValueError("entry has an unknown caller or symbolic/address escape")
        if tuple(value.type for value in operation.arguments) != tuple(entry.function_type.inputs) or tuple(
            value.type for value in operation.results
        ) != tuple(entry.function_type.outputs):
            raise ValueError("direct entry caller ABI disagrees")
        parent = operation.parent_op()
        while parent is not None and not isinstance(parent, func.FuncOp):
            parent = parent.parent_op()
        if parent is None or parent is entry:
            raise ValueError("recursive or unowned entry call requires a separate proof")
        calls.append(operation)
    return calls


def _pure_result_operation(operation) -> bool:
    if (
        not operation.results
        or operation.has_trait(IsTerminator, value_if_unregistered=False)
        or operation.has_trait(SymbolOpInterface, value_if_unregistered=False)
    ):
        return False
    # These value casts have no effects, although some supported xDSL versions
    # omit their Pure trait. Do not retain dead integer component subgraphs just
    # because that trait is absent.
    value_casts = {"arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast"}
    if operation.name in value_casts:
        return True
    # Tensor structured operations can report unknown generic memory effects,
    # while their tensor results and fully pure scalar body are value-only.
    if isinstance(operation, LinalgStructuredOperation):
        if any(not isinstance(value.type, TensorType) for value in operation.results):
            return False
        from xdsl.dialects.builtin import MemRefType

        if any(isinstance(value.type, MemRefType) for value in operation.operands):
            return False
        for region in operation.regions:
            for child in region.walk():
                if isinstance(child, YieldOp) or child.name in value_casts:
                    continue
                if get_effects(child) != set():
                    return False
        return True
    # Tensor empties/views are value-only IR operations. Physical memref
    # allocations, reads/writes/frees and unknown effects remain even when the
    # result is unused: allocator failures/resource/address effects are not
    # granted by the arithmetic contract.
    return get_effects(operation) == set()


def _pure_dce(entry):
    erased = Counter()
    while True:
        changed = False
        for operation in reversed(tuple(entry.body.block.ops)):
            if all(not result.uses for result in operation.results) and _pure_result_operation(operation):
                erased[operation.name] += 1
                operation.parent.erase_op(operation)
                changed = True
        if not changed:
            return tuple(sorted(erased.items()))


def _validate(module, contract, table):
    if not isinstance(contract, GeneratedDispatchABI) or not isinstance(table, EntryArgumentTable):
        raise ValueError("typed entry ownership and generated dispatch ABI required")
    if not all(
        value is True
        for value in (
            contract.compiler_generated_dispatch_only,
            contract.entry_address_unobserved,
            contract.nontrapping_arithmetic,
            contract.arithmetic_flags_unobserved,
            contract.generated_weight_storage_readonly,
            contract.weight_addresses_unobserved,
        )
    ):
        raise ValueError(
            "explicit generated caller, readonly weight storage and pure arithmetic effect contract required"
        )
    recheck_argument_table(table)
    module.verify()
    entry = _entry(module, contract.entry_symbol)
    if tuple(str(result) for result in entry.function_type.outputs) != table.result_types:
        raise ValueError("prepared entry results disagree with captured output ABI")
    if len(entry.body.block.args) != len(table.entries):
        raise ValueError("complete prepared entry argument table disagrees")
    for argument, binding in zip(entry.body.block.args, table.entries, strict=True):
        ty = argument.type
        if (
            not isinstance(ty, TensorType)
            or not isinstance(ty.encoding, NoneAttr)
            or ty.get_shape() != binding.shape
            or str(ty.element_type) != binding.dtype
        ):
            raise ValueError("prepared entry type does not match bound argument table")
    return entry, _direct_callers(module, entry)


def _project(module, contract, table):
    entry, calls = _validate(module, contract, table)
    erased = _pure_dce(entry)
    dce_digest = _module_digest(module)
    old_args = tuple(entry.body.block.args)
    removed = tuple(
        index
        for index, (argument, binding) in enumerate(zip(old_args, table.entries, strict=True))
        if binding.kind
        in {ArgumentOwnership.IMMUTABLE_STORED_WEIGHT, ArgumentOwnership.IMMUTABLE_GENERATED_ZERO_WEIGHT}
        and not argument.uses
    )
    retained = tuple(index for index in range(len(old_args)) if index not in removed)
    for call in calls:
        call.operands = tuple(call.arguments[index] for index in retained)
    for index in reversed(removed):
        entry.body.block.erase_arg(old_args[index])
    entry.function_type = FunctionType.from_lists(
        [old_args[index].type for index in retained], entry.function_type.outputs
    )
    module.verify()
    return module, removed, retained, erased, len(calls), dce_digest


def plan_entry_weight_projection(
    module: ModuleOp, *, entry_contract: GeneratedDispatchABI, argument_table: EntryArgumentTable
) -> EntryWeightProjection:
    """Read-only typed plan; all removability is rederived from prepared SSA."""
    entry, _ = _validate(module, entry_contract, argument_table)
    original_type = str(entry.function_type)
    clone, removed, retained, erased, calls, dce_digest = _project(module.clone(), entry_contract, argument_table)
    remap = tuple(retained.index(index) if index in retained else None for index in range(len(argument_table.entries)))
    return EntryWeightProjection(
        entry_contract,
        argument_table,
        _module_digest(module),
        original_type,
        dce_digest,
        _module_digest(clone),
        removed,
        retained,
        remap,
        erased,
        calls,
    )


def apply_entry_weight_projection(module: ModuleOp, projection: EntryWeightProjection) -> ModuleOp:
    """Recheck all source/table/caller proofs, then return a projected clone."""
    actual = plan_entry_weight_projection(
        module, entry_contract=projection.contract, argument_table=projection.argument_table
    )
    if actual != projection:
        raise ValueError("entry projection source or proof changed after planning")
    selected, *_ = _project(module.clone(), projection.contract, projection.argument_table)
    return selected


def prepare_entry_weight_projection(prepared: Path, work: Path, model_dir: Path, contract: GeneratedDispatchABI | None):
    """Normal post-preparation seam; None returns the original path with no I/O."""
    if contract is None:
        return prepared, None
    if contract.entry_symbol != "forward":
        raise ValueError("normal generated C runtime requires the captured forward entry")
    from merlin.common import mlir_query

    table = derive_entry_argument_table(model_dir, work)
    module = mlir_query.parse(prepared)
    plan = plan_entry_weight_projection(module, entry_contract=contract, argument_table=table)
    selected = apply_entry_weight_projection(module, plan)
    directory = work / "entry_weight_projection"
    directory.mkdir(exist_ok=False)
    path = directory / "projected.mlir"
    path.write_text(module_text(selected))
    receipt = {
        "schema": "merlin.entry_weight_projection.v1",
        **asdict(plan),
        "argument_table_sha256": table.identity,
        "selected_path": str(path.resolve()),
        "selected_file_sha256": _pin(path).sha256,
        "performance_claim": "UNMEASURED",
    }
    (directory / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    return path, plan


def compact_runtime_rows(rows, blob: bytes, projection: EntryWeightProjection):
    """Generate a fresh retained descriptor table and weight-span pack.

    Input/output array ownership is unchanged. Exact duplicate weight spans may
    share storage; partially overlapping spans are copied separately. Every live
    descriptor keeps all its original element bytes, including rank-zero/tails.
    The original captured bundle and manifest remain immutable.
    """
    from .c_runtime import DT_BYTES, GENERATED_WEIGHT_ALIGNMENT

    recheck_argument_table(projection.argument_table)
    count = len(projection.argument_table.entries)
    if len(rows) != count + len(projection.argument_table.result_types):
        raise ValueError("runtime table input/result arity disagrees")
    for index, (binding, row) in enumerate(zip(projection.argument_table.entries, rows[:count], strict=True)):
        kind, _offset, rank, shape, elem, dtype = row
        if (
            tuple(shape) != binding.shape
            or rank != len(shape)
            or dtype != binding.dtype
            or elem != DT_BYTES.get(binding.dtype)
            or binding.kind
            in {
                ArgumentOwnership.IMMUTABLE_STORED_WEIGHT,
                ArgumentOwnership.IMMUTABLE_GENERATED_ZERO_WEIGHT,
                ArgumentOwnership.PREPARED_WEIGHT,
            }
            and kind != "MERLIN_WEIGHT"
        ):
            raise ValueError("runtime descriptor disagrees with bound entry argument table")
    if tuple(
        index for index in range(count) if index not in projection.removed_original_indices
    ) != projection.retained_original_indices or any(
        projection.argument_table.entries[index].kind
        not in {ArgumentOwnership.IMMUTABLE_STORED_WEIGHT, ArgumentOwnership.IMMUTABLE_GENERATED_ZERO_WEIGHT}
        for index in projection.removed_original_indices
    ):
        raise ValueError("runtime projection removal/remap contract disagrees")
    from .weight_payload import WeightPayload

    streamed = isinstance(blob, WeightPayload)
    pack = WeightPayload() if streamed else bytearray()
    spans = {}
    selected = []
    for index in (*projection.retained_original_indices, *range(count, len(rows))):
        kind, offset, rank, shape, elem, dtype = rows[index]
        if kind == "MERLIN_WEIGHT":
            size = elem
            for dim in shape:
                size *= dim
            if offset < 0 or offset + size > len(blob):
                raise ValueError("retained weight span exceeds actual packed bytes")
            key = (offset, size)
            if key not in spans:
                padding = (-len(pack)) % GENERATED_WEIGHT_ALIGNMENT
                pack.extend(WeightPayload.zeros(padding) if streamed else bytes(padding))
                spans[key] = len(pack)
                pack.extend(blob[offset : offset + size])
            offset = spans[key]
        selected.append((kind, offset, rank, shape, elem, dtype))
    return selected, pack if streamed else bytes(pack)
