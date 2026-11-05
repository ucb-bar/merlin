"""Check a synchronous tiled loaded sum without enumerating its dynamic tiles.

This narrow checker admits two complete row-major inputs and one complete output.
An adapter projects actual instruction operands/effects. Its numerical proof and
synchronous implementation remain independently owned target obligations; this
module grants neither arithmetic equivalence nor execution qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from merlin.sched.ir import Call, Const, Kernel, Loop, Var, select
from merlin.sched.ir.expr import Expr
from merlin.sched.ir.kernel import DTYPE_BYTES


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class LoadedSumLayout:
    rows: int
    columns: int
    tile_rows: int
    tile_columns: int
    accumulator_base_bytes: int
    accumulator_row_bytes: int
    accumulator_capacity_bytes: int
    input_dtype: str = "i8"
    accumulator_dtype: str = "i32"
    output_dtype: str = "i8"
    index_bits: int = 64

    def validate(self):
        positive = (
            self.rows,
            self.columns,
            self.tile_rows,
            self.tile_columns,
            self.accumulator_row_bytes,
            self.accumulator_capacity_bytes,
        )
        if any(type(n) is not int or n <= 0 for n in positive):
            raise ValueError("loaded sum requires positive static extents and storage")
        if type(self.accumulator_base_bytes) is not int or self.accumulator_base_bytes < 0:
            raise ValueError("loaded sum requires a nonnegative private accumulator base")
        if any(t not in ("i8", "i16", "i32") for t in (self.input_dtype, self.accumulator_dtype, self.output_dtype)):
            raise ValueError("loaded sum requires explicit signed integer storage widths")
        if DTYPE_BYTES[self.accumulator_dtype] < DTYPE_BYTES[self.input_dtype]:
            raise ValueError("loaded sum accumulator is narrower than its operands")
        unit = DTYPE_BYTES[self.accumulator_dtype]
        if (
            self.accumulator_base_bytes % unit
            or self.accumulator_row_bytes % unit
            or self.accumulator_row_bytes < self.tile_columns * unit
        ):
            raise ValueError("private accumulator rows overlap or have unaligned storage")
        height, width = min(self.rows, self.tile_rows), min(self.columns, self.tile_columns)
        end = self.accumulator_base_bytes + (height - 1) * self.accumulator_row_bytes + width * unit
        if end > self.accumulator_capacity_bytes:
            raise ValueError("loaded sum private accumulator tile exceeds its capacity")
        if type(self.index_bits) is not int or self.index_bits not in (32, 64):
            raise ValueError("loaded sum needs explicit signed index width")
        largest = max(
            self.rows * self.columns * DTYPE_BYTES[self.input_dtype],
            self.rows * self.columns * DTYPE_BYTES[self.output_dtype],
            end,
            self.tile_rows,
            self.tile_columns,
        )
        if largest >= 1 << (self.index_bits - 1):
            raise ValueError("loaded sum byte addressing exceeds the selected signed index width")


@dataclass(frozen=True)
class LoadedSumContract:
    """Original scalar proof and three exact adapter parameter bindings.

    Parameters are opaque integer payloads (for example scale bits), never
    interpreted as target instructions here. The original owner must prove
    that each binding implements this very scalar proof before execution.
    """

    scalar_proof_sha256: str
    parameters: tuple[tuple[tuple[str, int], ...], ...]
    implementation_identity: str
    private_ownership: bool
    external_buffers_disjoint: bool

    def validate(self):
        if not _sha(self.scalar_proof_sha256):
            raise ValueError("loaded sum requires the exact original scalar proof identity")
        if (
            type(self.parameters) is not tuple
            or len(self.parameters) != 3
            or any(
                type(p) is not tuple
                or any(
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not str
                    or not item[0]
                    or type(item[1]) is not int
                    for item in p
                )
                or len({item[0] for item in p}) != len(p)
                for p in self.parameters
            )
        ):
            raise ValueError("loaded sum requires three exact immutable integer parameter bindings")
        if type(self.implementation_identity) is not str or not self.implementation_identity:
            raise ValueError("loaded sum needs an explicit implementation identity")
        if self.private_ownership is not True or self.external_buffers_disjoint is not True:
            raise ValueError("loaded sum needs original private ownership and external disjointness")


@dataclass(frozen=True)
class LoadedSumEffect:
    """Actual logical effect decoded by the original instruction adapter.

    initialize reads lhs and overwrites the private rectangle; accumulate reads
    rhs and reads/adds/writes that rectangle; store reads it and writes output.
    All three must finish before the next effect begins. Coordinates are in
    logical elements; the private base and row stride are in bytes.
    """

    kind: str
    tensor: str
    row: Expr
    column: Expr
    rows: Expr
    columns: Expr
    accumulator_base_bytes: Expr
    accumulator_row_bytes: int
    scalar_proof_sha256: str
    parameters: tuple[tuple[str, int], ...]
    completion: str


def verify_compact_loaded_sum(
    kernel: Kernel,
    layout: LoadedSumLayout,
    contract: LoadedSumContract,
    project: Callable[[Call], LoadedSumEffect],
    *,
    scalar_proof: dict,
):
    """Prove the complete canonical two-loop cover and private tile lifetimes.

    Work is constant in the dynamic tile count: exactly three static effects
    are projected. Unknown loop/effect forms refuse; no finite expansion or
    sampling substitutes for verification. The scalar proof is byte-bound,
    not established here. Projectors must decode actual operands, not requests.
    """
    if type(layout) is not LoadedSumLayout or type(contract) is not LoadedSumContract:
        raise ValueError("explicit loaded sum layout and contract required")
    layout.validate()
    contract.validate()
    if type(scalar_proof) is not dict or _identity(scalar_proof) != contract.scalar_proof_sha256:
        raise ValueError("loaded sum scalar proof changed or differs from the original contract")
    if type(kernel) is not Kernel or kernel.attrs or not callable(project):
        raise ValueError("loaded sum requires a plain current counted kernel and effect projector")
    signature = tuple((a.name, a.shape, a.dtype, a.access) for a in kernel.args)
    shape = (layout.rows, layout.columns)
    if signature != (
        ("lhs", shape, layout.input_dtype, "read"),
        ("rhs", shape, layout.input_dtype, "read"),
        ("output", shape, layout.output_dtype, "write"),
    ):
        raise ValueError("loaded sum must cover both complete original inputs and its sole output")
    mt = (layout.rows + layout.tile_rows - 1) // layout.tile_rows
    nt = (layout.columns + layout.tile_columns - 1) // layout.tile_columns
    if len(kernel.body) != 1 or type(kernel.body[0]) is not Loop:
        raise ValueError("loaded sum needs one outer counted row loop")
    outer = kernel.body[0]
    if outer.extent != mt or type(outer.extent) is not int or len(outer.body) != 1 or type(outer.body[0]) is not Loop:
        raise ValueError("loaded sum row loop does not cover the original source")
    inner = outer.body[0]
    if (
        inner.extent != nt
        or type(inner.extent) is not int
        or outer.var == inner.var
        or type(outer.var) is not str
        or not outer.var
        or type(inner.var) is not str
        or not inner.var
        or len(inner.body) != 3
        or any(type(call) is not Call for call in inner.body)
    ):
        raise ValueError("loaded sum needs three ordered effects in each complete column tile")
    digest = kernel.digest()
    layout_before, contract_before = asdict(layout), asdict(contract)
    effects = []
    row = Var(outer.var) * layout.tile_rows
    column = Var(inner.var) * layout.tile_columns
    height = select(outer.var, mt - 1, (layout.rows - 1) % layout.tile_rows + 1, layout.tile_rows)
    width = select(inner.var, nt - 1, (layout.columns - 1) % layout.tile_columns + 1, layout.tile_columns)
    for index, (call, kind, tensor) in enumerate(
        zip(inner.body, ("initialize", "accumulate", "store"), ("lhs", "rhs", "output"), strict=True)
    ):
        if call != Call(call.instr, call.args):
            raise ValueError("loaded sum cannot hide async, configuration or staging metadata")
        effect = project(call)
        expected = LoadedSumEffect(
            kind,
            tensor,
            row,
            column,
            height,
            width,
            Const(layout.accumulator_base_bytes),
            layout.accumulator_row_bytes,
            contract.scalar_proof_sha256,
            contract.parameters[index],
            "synchronous",
        )
        if type(effect) is not LoadedSumEffect or effect != expected:
            raise ValueError("actual loaded sum effect differs in order, source cover, tail, storage or scalar proof")
        effects.append(asdict(effect))
    if (
        kernel.digest() != digest
        or asdict(layout) != layout_before
        or asdict(contract) != contract_before
        or _identity(scalar_proof) != contract.scalar_proof_sha256
    ):
        raise ValueError("loaded sum source, contract or scalar proof changed during projection")
    return dict(
        schema="merlin.compact_loaded_sum.v1",
        kernel_sha256=digest,
        scalar_proof_sha256=contract.scalar_proof_sha256,
        layout=layout_before,
        contract=contract_before,
        tile_grid=[mt, nt],
        dynamic_calls=3 * mt * nt,
        checked_static_effects=3,
        output_elements=layout.rows * layout.columns,
        complete_input_output_cover=True,
        synchronous_private_tile_reuse=True,
        proof=(
            "Disjoint clipped row and column tile intervals partition the original rectangle; "
            "each tile initializes, accumulates and stores the identical live rectangle "
            "in order before synchronous reuse"
        ),
        target_execution="not_established",
        numerical_equivalence="original scalar proof owner",
        effect_projection_sha256=_identity(effects),
    )
