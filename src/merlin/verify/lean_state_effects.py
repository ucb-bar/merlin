"""Host-selected projection of actual sequential SSA/read/write schedules.

This is a conditional abstract-state proof boundary. Reviewed instruction
semantics must be deterministic and total, read-local, and write-framed;
effects declarations or namespace names alone do not establish those premises.
No candidate callback, timing model, physical alias policy or RTL proof is used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


class UnsupportedStateEffects(ValueError):
    """Actual IR has no complete admitted state/effect interpretation."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class EffectOperation:
    name: str
    operand_types: tuple[str, ...]
    result_types: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    semantic_sha256: str
    deterministic_total: bool
    complete_read_write_frames: bool
    attribute_names: tuple[str, ...] = ()
    observable: bool = False
    sequential_atomic_complete: bool = False


@dataclass(frozen=True)
class EffectScheduleContract:
    """Immutable ORIGINAL host premises; never an author-selected purity claim.

    Distinct named ISA resources may still share state. Only original reviewed
    disjoint pairs permit independent resource effects. Memref caller arguments
    share one conservative memory identity. Nonescaping fresh allocations are
    intrinsically disjoint under the explicit standard allocation premise.
    """

    operations: tuple[EffectOperation, ...] = ()
    disjoint_resources: tuple[tuple[str, str], ...] = ()
    standard_memory_sequential: bool = False
    dynamic_memref_accesses_defined: bool = False
    standard_allocations_successful: bool = False
    index_bits: int | None = None
    maximum_operations: int = 8192
    maximum_exchanges: int = 65536
    maximum_effect_cells: int = 65536
    isa_records: tuple[str, ...] = ()
    schema: str = "merlin.effect_schedule_contract.v1"

    def validate(self):
        if self.schema != "merlin.effect_schedule_contract.v1":
            raise ValueError("unknown effect schedule contract")
        if type(self.operations) is not tuple or any(type(item) is not EffectOperation for item in self.operations):
            raise ValueError("effect semantics must be immutable host operation records")
        names = set()
        resources = set()
        for operation in self.operations:
            if (
                not isinstance(operation.name, str)
                or not operation.name
                or operation.name in names
                or operation.name.startswith(("arith.", "memref.", "scf.", "func.", "builtin."))
                or not _sha(operation.semantic_sha256)
                or operation.deterministic_total is not True
                or operation.complete_read_write_frames is not True
                or type(operation.observable) is not bool
                or operation.sequential_atomic_complete is not True
            ):
                raise ValueError(
                    "operation needs unique nonstandard identity and complete admitted "
                    "frame/totality and atomic semantics"
                )
            names.add(operation.name)
            for field in ("operand_types", "result_types", "reads", "writes", "attribute_names"):
                values = getattr(operation, field)
                if type(values) is not tuple or any(not isinstance(value, str) or not value for value in values):
                    raise ValueError("operation declarations must contain immutable exact names/types")
                if field in {"reads", "writes", "attribute_names"} and len(values) != len(set(values)):
                    raise ValueError("duplicate operation effect or metadata declarations")
            resources.update((*operation.reads, *operation.writes))
        if (
            type(self.disjoint_resources) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or pair[0] == pair[1]
                or not all(value in resources for value in pair)
                for pair in self.disjoint_resources
            )
            or len({tuple(sorted(pair)) for pair in self.disjoint_resources}) != len(self.disjoint_resources)
        ):
            raise ValueError("disjoint resource pairs require unique original declared resource identities")
        for field in (
            "standard_memory_sequential",
            "dynamic_memref_accesses_defined",
            "standard_allocations_successful",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("standard execution premises must be explicit booleans")
        if self.index_bits is not None and (type(self.index_bits) is not int or not 2 <= self.index_bits <= 256):
            raise ValueError("explicit supported source index width required")
        for field in ("maximum_operations", "maximum_exchanges", "maximum_effect_cells"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError("effect projection needs bounded positive host budgets")
        if type(self.isa_records) is not tuple or any(not isinstance(row, str) for row in self.isa_records):
            raise ValueError("ISA models must be immutable reviewed records")

    def identity(self):
        self.validate()
        return _digest(asdict(self))


def effect_contract_from_isa(models, *, admitted_instructions, atomic_instructions, **premises):
    """Derive signatures/effects from selected facts-bound declarations.

    `admitted_instructions` is the original host's qualified list: the ISA
    effects table alone does not prove totality, locality or RTL conformance.
    `atomic_instructions` must separately admit sequential atomic abstraction
    of the complete modeled state, including queues/visibility where relevant.
    Unknown effects refuse; all instruction records/facts identities are pinned.
    """
    from xdsl.dialects import builtin

    from merlin.targetgen.isa_dialect_gen import build_dialect
    from merlin.targetgen.isa_spec import resolved_isa_from_record

    if (
        type(models) is not tuple
        or type(admitted_instructions) is not tuple
        or type(atomic_instructions) is not tuple
        or set(atomic_instructions) != set(admitted_instructions)
        or len(atomic_instructions) != len(set(atomic_instructions))
    ):
        raise ValueError("immutable selected ISA models and original host admission list required")
    if len(admitted_instructions) != len(set(admitted_instructions)):
        raise ValueError("duplicate admitted instruction identity")
    selected = set(admitted_instructions)
    operations, records = [], []
    for original in models:
        model = resolved_isa_from_record(original.record())
        records.append(json.dumps(model.record(), sort_keys=True, separators=(",", ":")))
        dialect = build_dialect(model)
        address = next(kind for kind in dialect.attributes if kind.name == f"{model.namespace}.address")

        def type_text(argument):
            if argument.type.kind == "integer":
                return str(builtin.IntegerType(argument.type.bits))
            return str(address(builtin.StringAttr(argument.type.space), builtin.IntAttr(argument.type.bits)))

        for instruction in model.instructions:
            name = f"{model.namespace}.{instruction.name}"
            if name not in selected:
                continue
            selected.remove(name)
            effects = json.loads(instruction.effects_json)
            if effects.get("status") == "unknown":
                raise UnsupportedStateEffects("admitted instruction still has unknown declared effects")
            operations.append(
                EffectOperation(
                    name,
                    tuple(map(type_text, instruction.operands)),
                    tuple(map(type_text, instruction.results)),
                    tuple(f"{model.namespace}:{name}" for name in effects["reads"]),
                    tuple(f"{model.namespace}:{name}" for name in effects["writes"]),
                    model.identity(),
                    True,
                    True,
                    tuple(argument.name for argument in instruction.immediates),
                    sequential_atomic_complete=True,
                )
            )
    if selected:
        raise ValueError("host admission refers to undeclared ISA instructions")
    result = EffectScheduleContract(tuple(operations), isa_records=tuple(records), **premises)
    result.validate()
    return result


@dataclass(frozen=True)
class EffectNode:
    operation_index: int
    operation_name: str
    operation_sha256: str
    reads: tuple[int, ...]
    writes: tuple[int, ...]


@dataclass(frozen=True)
class EffectExchange:
    position: int
    first: int
    second: int


@dataclass(frozen=True)
class EffectScheduleProjection:
    source_sha256: str
    target_sha256: str
    contract_sha256: str
    block_index: int
    order: tuple[int, ...]
    nodes: tuple[EffectNode, ...]
    cells: tuple[str, ...]
    dependencies: tuple[tuple[int, int], ...]
    exchanges: tuple[EffectExchange, ...]
    scope: tuple[str, ...] = (
        "actual_whole_ir_reorder_correspondence",
        "typed_ssa_dependencies",
        "admitted_deterministic_read_write_frames",
        "conservative_memory_aliases",
        "sequential_complete_state_and_observations",
    )
    assumptions: tuple[str, ...] = (
        "original host operation admissions are complete deterministic total read-local write-framed semantics",
        "standard memref accesses and source indices are defined; selected allocation premises hold",
        "sequential IR state has no asynchronous or timing-sensitive unmodeled observations",
        "actual IR correspondence and semantic effect projection are trusted; machine/RTL execution is separate",
    )

    def record(self):
        return asdict(self)

    def identity(self):
        return _digest(self.record())

    def lean_obligations(self):
        statements, proofs = [], []
        for exchange in self.exchanges:
            first, second = self.nodes[exchange.first], self.nodes[exchange.second]

            def footprint(node):
                return "{ reads := " + str(list(node.reads)) + ", writes := " + str(list(node.writes)) + " }"

            statements.append(
                "∀ (Value : Type) (first : Merlin.StateEffects.Action " + footprint(first) + " Value) "
                "(second : Merlin.StateEffects.Action " + footprint(second) + " Value) "
                "(earlier later : List (Merlin.StateEffects.State Value → Merlin.StateEffects.State Value)) "
                "(initial : Merlin.StateEffects.State Value), "
                "Merlin.StateEffects.execute (earlier ++ first.step :: second.step :: later) initial = "
                "Merlin.StateEffects.execute (earlier ++ second.step :: first.step :: later) initial"
            )
            proofs.append(
                "by intro Value first second earlier later initial; exact "
                "Merlin.StateEffects.exchangeAdjacent earlier later first.step second.step "
                "(fun state => Merlin.StateEffects.commuteActions first second (by decide) state) initial"
            )
        if not statements:
            statements.append(
                "∀ (Value : Type) (run : Nat → Merlin.StateEffects.State Value → Merlin.StateEffects.State Value) "
                "(initial : Merlin.StateEffects.State Value), "
                f"Merlin.StateEffects.execute ({list(self.order)}.map run) initial = "
                f"Merlin.StateEffects.execute ({list(self.order)}.map run) initial"
            )
            proofs.append("by intros; rfl")
        return statements, proofs


def project_effect_schedule(before_ir, after_ir, *, block_index, order, contract):
    from .lean_state_projection import project_effect_schedule as project

    return project(before_ir, after_ir, block_index=block_index, order=order, contract=contract)


def derive_effect_schedule(before_ir, after_ir, *, entry, contract):
    from .lean_state_projection import derive_effect_schedule as derive

    return derive(before_ir, after_ir, entry=entry, contract=contract)


def effect_semantic_identity():
    """Pin the projection/factory/math and their actual xDSL source inventory.

    This describes semantics, never live proof authority. The selected host
    runtime/import roots and compiled Lean products require separate admission.
    """
    import importlib
    from pathlib import Path

    files = {
        "lean_state_effects": Path(__file__),
        "lean_state_projection": Path(__file__).with_name("lean_state_projection.py"),
        "StateEffects": Path(__file__).parent / "lean" / "Merlin" / "StateEffects.lean",
    }
    for name in (
        "merlin.transforms.certificates",
        "merlin.transforms.normalization",
        "merlin.transforms.planning",
        "merlin.transforms.pure_scalar",
        "merlin.transforms.scalar_witness",
        "merlin.targetgen.isa_spec",
        "merlin.targetgen.isa_dialect_gen",
    ):
        files[name] = Path(importlib.import_module(name).__file__)
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    root = Path(importlib.import_module("xdsl").__file__).parent
    xdsl_sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }
    return _digest({"schema": "merlin.state_effect_semantics.v1", "sources": sources, "xdsl_sources": xdsl_sources})
