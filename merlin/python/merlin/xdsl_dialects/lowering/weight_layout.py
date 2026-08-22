"""Weights the model re-lays-out at run time, and whether that work can be hoisted offline.

A `linalg.transpose` whose input is a function argument is not computation — it is the model paying,
on every inference, to convert a weight into the layout its consumer wanted all along. The packer
could have stored it that way once.

MEASURED, and why this module exists. On `gemma2_2b_int8_full_seq8` 183 such transposes move
**2,493.0 MiB per inference** against a total weight blob of 2,505 MiB: essentially every weight in
the model, re-laid-out every time it runs. It is not only a throughput tax. The largest one, the
562.5 MiB int8 tied head, is what killed a whole-model FireSim run — `FAIL alloc bytes=589824064` at
op 11,494 of 11,526, with the arena unable to serve it. Hoisting all 183 into the bundle took the
count to 0 with the MAC total and the matrix-unit routing both unchanged.

HOISTABLE MEANS SOLE-USE. Storing a weight pre-transposed is safe only when the transpose is that
argument's ONLY consumer. If anything else reads the argument, it would silently start seeing
transposed data. :func:`weight_layout_report` therefore splits the two cases and never merges them —
on the model above the split was 183 sole-use and 0 mixed, but a model with a weight used both
transposed and untransposed must not be hoisted blindly.

The rewrite itself is bit-exact: a transpose moves elements, it does not compute. Every value the
consumer sees is the value it saw before, at the same index, so goldens survive untouched.

Structural throughout: an argument is a weight because it is a function argument, and the op is a
re-layout because it is a transpose reading one. No name matching, no provenance tags — 751 of that
model's 843 generics carry no tag at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import mlir_query as mq

#: bytes per element, by the dtype spelling `mlir_query.type_shape_dtype` returns.
_WIDTH = {"i8": 1, "u8": 1, "i16": 2, "f16": 2, "bf16": 2,
          "i32": 4, "f32": 4, "i64": 8, "f64": 8}


@dataclass(frozen=True)
class WeightRelayout:
    """One transpose of a function argument."""

    arg: int
    shape: list[int]
    dtype: str
    result_shape: list[int]
    #: True when the transpose is the argument's only consumer, so the layout can be pre-applied.
    hoistable: bool
    #: why not, when it is not
    reason: str = ""

    @property
    def bytes_moved(self) -> int:
        width = _WIDTH.get(self.dtype)
        if width is None:
            return 0
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * width


@dataclass
class WeightLayoutReport:
    relayouts: list[WeightRelayout] = field(default_factory=list)
    #: transposes whose element width is unknown, so their cost could not be priced. Surfaced rather
    #: than counted as zero: an unpriceable op must not read as a free one.
    unpriceable: list[str] = field(default_factory=list)

    @property
    def hoistable(self) -> list[WeightRelayout]:
        return [r for r in self.relayouts if r.hoistable]

    @property
    def blocked(self) -> list[WeightRelayout]:
        return [r for r in self.relayouts if not r.hoistable]

    @property
    def hoistable_bytes(self) -> int:
        """Weight traffic per inference that storing the transposed layout would remove."""
        return sum(r.bytes_moved for r in self.hoistable)

    @property
    def total_bytes(self) -> int:
        return sum(r.bytes_moved for r in self.relayouts)


def _func(module: Any, func_name: str):
    for fn in mq.walk(module, "func.func"):
        name = fn.properties.get("sym_name") or fn.attributes.get("sym_name")
        if name is None or func_name in str(name):
            return fn
    return None


def weight_layout_report(module: Any, func_name: str = "forward") -> WeightLayoutReport:
    """Every run-time re-layout of a weight in `func_name`, split by whether it can be hoisted."""
    rep = WeightLayoutReport()
    fn = _func(module, func_name)
    if fn is None:
        return rep

    for i, arg in enumerate(fn.body.blocks[0].args):
        uses = list(arg.uses)
        transposes = [u for u in uses if mq.op_name(u.operation) == "linalg.transpose"]
        if not transposes:
            continue
        shape, dtype = mq.type_shape_dtype(arg.type)
        op = transposes[0].operation
        res_shape: list[int] = []
        if op.results:
            rs, _ = mq.type_shape_dtype(op.results[0].type)
            res_shape = list(rs)

        if dtype not in _WIDTH:
            rep.unpriceable.append(f"arg {i}: unknown element width for dtype {dtype!r}")

        sole = len(uses) == 1 and len(transposes) == 1
        reason = "" if sole else (
            f"argument has {len(uses)} consumers ({len(transposes)} transpose(s)); pre-transposing "
            "would change what the other readers see")
        rep.relayouts.append(WeightRelayout(
            arg=i, shape=list(shape), dtype=dtype, result_shape=res_shape,
            hoistable=sole, reason=reason))
    return rep
