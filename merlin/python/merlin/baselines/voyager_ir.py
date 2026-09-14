"""Replay a Voyager-compiled model into a concrete, ordered event trace.

The Voyager compiler (github.com/jeffreyyu0602/voyager-compiler) lowers a PyTorch model to a
bufferized program, serialized as its ``voyager.Model`` protobuf: loops and conditionals over scalar
index arithmetic, DMA copies between DRAM and banked scratchpad slots ordered by counting semaphores,
and asynchronously committed compute regions (a GEMM plus its fused tail). A bridge that runs that
program on other hardware has to reproduce its schedule -- which tile moves when, into which slot, and
which compute consumes it -- without re-deriving any of it, or the comparison measures the bridge.

:func:`replay` evaluates the program's scalar control flow with concrete values and records every
data-movement and compute event in program order, with each tensor operand resolved to its allocation,
slot and window. It reads the JSON form of the message (``json_format.MessageToDict`` with field names
preserved), so it needs neither protobuf nor the Voyager package.

Deliberately dependency-free (standard library only) so it can be vendored into an out-of-tree backend
package, which may not import merlin. It fails closed: an operation or argument shape it does not model
raises :class:`UnsupportedConstruct` instead of being skipped, and the counting-semaphore discipline
Voyager's own eager ops assert (every wait matched by a prior signal on the same slot) is re-checked
while replaying, so a trace that could not have executed is refused rather than returned.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Box", "Copy", "FusedCompute", "PrimCall", "Ref", "SemaphoreViolation", "TensorOp", "Trace",
    "UnsupportedConstruct", "Wait", "load_model", "replay",
]

#: proto3 JSON omits a field at its default. For ``Memory.level`` the default enum value is IMMEDIATE,
#: and for every integer it is 0; both have to be restored rather than read as "absent".
_DEFAULT_LEVEL = "MEMORY_LEVEL_IMMEDIATE"
_LEVEL_PREFIX = "MEMORY_LEVEL_"


class UnsupportedConstruct(ValueError):
    """The program uses an operation or argument form this replay does not model."""


class SemaphoreViolation(RuntimeError):
    """A wait or a commit consumed a semaphore slot that no earlier event had signalled."""


@dataclass(frozen=True)
class Box:
    """One allocation: where it lives and how it is banked. ``level`` is None for a value that never
    leaves the datapath (an intermediate inside a fused chain), which owns no storage."""

    node: str
    shape: tuple[int, ...]
    dtype: str
    level: str | None
    address: int
    bank_count: int = 1
    bank_stride: int = 0

    @property
    def on_chip(self) -> bool:
        return self.level == "SCRATCHPAD"


@dataclass(frozen=True)
class Ref:
    """A tensor operand, with every dynamic offset resolved to an integer.

    ``offsets``/``sizes``/``strides`` carry one entry per referenced dimension; for a banked box the
    first referenced dimension selects the bank (the pipeline slot). Empty means the whole allocation.
    """

    box: Box
    offsets: tuple[int, ...] = ()
    sizes: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()

    @property
    def slot(self) -> int:
        return self.offsets[0] if self.box.bank_count > 1 and self.offsets else 0

    @property
    def slot_address(self) -> int:
        """Byte address of the slot this reference selects (the allocation base for an unbanked box)."""
        return self.box.address + self.slot * self.box.bank_stride


@dataclass(frozen=True)
class Copy:
    """``voyager::async_copy``: move a tile, then signal ``semaphore`` by ``post_count``."""

    name: str
    src: Ref
    dst: Ref
    indices: tuple[int, ...]
    sizes: tuple[int, ...]
    semaphore: Ref
    post_count: int = 1
    dims: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None
    transposed: bool = False
    pad: tuple[int, ...] | None = None
    pad_value: float | None = None
    count: tuple[int, ...] | None = None
    path: tuple[tuple[str, int], ...] = ()

    @property
    def is_load(self) -> bool:
        return self.dst.box.on_chip and not self.src.box.on_chip

    @property
    def is_store(self) -> bool:
        return self.src.box.on_chip and not self.dst.box.on_chip


@dataclass(frozen=True)
class Wait:
    """``voyager::async_wait``: block until ``semaphore`` has been signalled, then consume it."""

    name: str
    semaphore: Ref
    path: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class PrimCall:
    """One primitive call, with tensor operands resolved to :class:`Ref` and scalars to values."""

    name: str
    target: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedCompute:
    """A fused chain (e.g. GEMM -> dequantize -> relu) and the destination it writes.

    ``commit`` names the asynchronous region it was dispatched in (None for a synchronous op);
    ``tiling`` is the interstellar mapping as ``((loop, bound), ...)`` per level, innermost first, as
    Voyager serialized it.
    """

    name: str
    chain: tuple[PrimCall, ...]
    destinations: tuple[Ref, ...]
    tiling: tuple[tuple[tuple[str, int], ...], ...] = ()
    commit: str | None = None
    dependencies: tuple[Ref, ...] = ()
    post: Ref | None = None
    path: tuple[tuple[str, int], ...] = ()

    @property
    def anchor(self) -> PrimCall:
        return self.chain[0]

    @property
    def tail(self) -> tuple[str, ...]:
        return tuple(call.target for call in self.chain[1:])


@dataclass(frozen=True)
class TensorOp:
    """A tensor primitive outside a fusion (e.g. ``voyager::insert`` of a fused result into a slot)."""

    call: PrimCall
    commit: str | None = None
    path: tuple[tuple[str, int], ...] = ()


@dataclass
class Trace:
    """Everything :func:`replay` observed, in program order."""

    inputs: tuple[Box, ...]
    parameters: tuple[Box, ...]
    outputs: tuple[Box, ...]
    allocations: dict[str, Box]
    events: list[Copy | Wait | FusedCompute | TensorOp]
    semaphores: dict[tuple[str, int], int]

    def of(self, kind: type) -> list:
        return [event for event in self.events if isinstance(event, kind)]


def load_model(path: str | Path) -> dict[str, Any]:
    """The ``voyager.Model`` JSON written next to ``model.txt`` by the export step."""
    return json.loads(Path(path).read_text())


# --- decoding the JSON form ----------------------------------------------------------------------

def _ints(values: Sequence[Any] | None) -> tuple[int, ...]:
    return tuple(int(v) for v in (values or ()))


def _box(raw: Mapping[str, Any]) -> Box:
    memory = raw.get("memory")
    level: str | None
    if memory is None:
        level, address = None, 0
    else:
        level = str(memory.get("level", _DEFAULT_LEVEL)).removeprefix(_LEVEL_PREFIX)
        address = int(memory.get("address", 0))
    return Box(node=str(raw["node"]), shape=_ints(raw.get("shape")), dtype=str(raw.get("dtype", "")),
               level=level, address=address, bank_count=int(raw.get("bank_count", 1) or 1),
               bank_stride=int(raw.get("bank_stride_bytes", 0) or 0))


_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "floordiv": lambda a, b: a // b, "mod": lambda a, b: a % b,
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
    "and_": lambda a, b: bool(a) and bool(b), "or_": lambda a, b: bool(a) or bool(b),
    "sym_max": max, "sym_min": min,
}
_UNARY = {"not_": lambda a: not a, "neg": lambda a: -a}
_DECLARATIONS = {"voyager::alloc", "voyager::zeros", "voyager::fill"}


class _Replayer:
    def __init__(self, model: Mapping[str, Any], check_semaphores: bool):
        self.check = check_semaphores
        self.env: dict[str, Any] = {}
        self.allocations: dict[str, Box] = {}
        self.events: list[Copy | Wait | FusedCompute | TensorOp] = []
        self.semaphores: dict[tuple[str, int], int] = {}
        self.path: list[tuple[str, int]] = []
        for key in ("inputs", "parameters", "outputs"):
            for raw in model.get(key, ()):
                box = _box(raw)
                self.allocations[box.node] = box

    # scalars ----------------------------------------------------------------------------------
    def scalar(self, raw: Mapping[str, Any]) -> Any:
        if "node" in raw:
            name = str(raw["node"])
            if name not in self.env:
                raise UnsupportedConstruct(f"scalar {name!r} is read before any operation defines it")
            return self.env[name]
        if "int_value" in raw:
            return int(raw["int_value"])
        if "bool_value" in raw:
            return bool(raw["bool_value"])
        if "float_value" in raw:
            return float(raw["float_value"])
        raise UnsupportedConstruct(f"scalar with no value: {raw!r}")

    def ref(self, raw: Mapping[str, Any]) -> Ref:
        box_raw = raw["box"]
        node = str(box_raw["node"])
        box = _box(box_raw)
        if box.level is not None and node in self.allocations:
            declared = self.allocations[node]
            # The window's box is the allocation under the window's shape; banking comes from the
            # declaration when the window omits it.
            box = Box(node, box.shape or declared.shape, box.dtype or declared.dtype, box.level,
                      box.address, box.bank_count if "bank_count" in box_raw else declared.bank_count,
                      box.bank_stride if "bank_stride_bytes" in box_raw else declared.bank_stride)
        return Ref(box=box, offsets=tuple(int(self.scalar(o)) for o in raw.get("offsets", ())),
                   sizes=_ints(raw.get("sizes")), strides=_ints(raw.get("strides")),
                   output_shape=_ints(raw.get("output_shape")))

    def argument(self, raw: Mapping[str, Any]) -> Any:
        if "tensor_box" in raw:
            return self.ref(raw["tensor_box"])
        if "tensor_box_list" in raw:
            return tuple(self.ref(v) for v in raw["tensor_box_list"].get("values", ()))
        if "scalar" in raw:
            return self.scalar(raw["scalar"])
        if "scalar_list" in raw:
            return tuple(self.scalar(v) for v in raw["scalar_list"].get("values", ()))
        if "str_value" in raw:
            return str(raw["str_value"])
        if not raw:
            return None  # an optional argument serialized as an empty Argument
        raise UnsupportedConstruct(f"argument form not modelled: {sorted(raw)}")

    def call(self, prim: Mapping[str, Any]) -> PrimCall:
        return PrimCall(name=str(prim.get("name", "")), target=str(prim.get("target", "")),
                        args=tuple(self.argument(a) for a in prim.get("args", ())),
                        kwargs={k: self.argument(v) for k, v in prim.get("kwargs", {}).items()})

    # semaphores -------------------------------------------------------------------------------
    def signal(self, sem: Ref, count: int) -> None:
        key = (sem.box.node, sem.slot)
        self.semaphores[key] = self.semaphores.get(key, 0) + count

    def consume(self, sem: Ref, who: str) -> None:
        key = (sem.box.node, sem.slot)
        have = self.semaphores.get(key, 0)
        if self.check and have <= 0:
            raise SemaphoreViolation(f"{who} consumes {key} with no prior signal (count {have})")
        self.semaphores[key] = have - 1

    # operations -------------------------------------------------------------------------------
    def run(self, ops: Sequence[Mapping[str, Any]], commit: str | None = None) -> None:
        for op in ops:
            self.operation(op, commit)

    def operation(self, op: Mapping[str, Any], commit: str | None) -> None:
        name = str(op["name"])
        outputs = op.get("outputs", ())
        if "prim" in op:
            self.prim(op["prim"], outputs, commit)
        elif "fused" in op:
            self.fused(name, op, commit)
        elif "loop" in op:
            self.loop(name, op["loop"], outputs)
        elif "cond" in op:
            self.cond(op["cond"], outputs)
        elif "async" in op:
            self.commit(name, op["async"])
        else:
            raise UnsupportedConstruct(f"operation {name!r} has no modelled op_type: {sorted(op)}")

    def prim(self, prim: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]],
             commit: str | None) -> None:
        target = str(prim.get("target", ""))
        kwargs = prim.get("kwargs", {})
        if target in _DECLARATIONS:
            for out in outputs:
                if "tensor_box" in out:
                    box = _box(out["tensor_box"])
                    self.allocations[box.node] = box
                    if box.level == "REGISTER" and target == "voyager::fill":
                        # A filled register array is a semaphore that starts signalled: an output
                        # slot is FREE before its first store, so its first reuse must not block.
                        value = int(self.scalar(kwargs["value"]["scalar"]))
                        for slot in range(box.bank_count):
                            self.semaphores[(box.node, slot)] = value
            return
        if target in _BINARY:
            self._bind(outputs, _BINARY[target](self.scalar(kwargs["input"]["scalar"]),
                                                self.scalar(kwargs["other"]["scalar"])))
            return
        if target in _UNARY:
            self._bind(outputs, _UNARY[target](self.scalar(kwargs["input"]["scalar"])))
            return
        if target == "sym_ite":
            chosen = kwargs["t"] if self.scalar(kwargs["b"]["scalar"]) else kwargs["f"]
            self._bind(outputs, self.scalar(chosen["scalar"]))
            return
        if target == "voyager::delinearize_index":
            linear = int(self.scalar(kwargs["linear"]["scalar"]))
            basis = [int(self.scalar(v)) for v in kwargs["basis"]["scalar_list"].get("values", ())]
            digits = []
            for radix in reversed(basis):
                digits.append(linear % radix)
                linear //= radix
            digits.reverse()
            if len(outputs) != len(digits):
                raise UnsupportedConstruct(
                    f"delinearize_index over basis {basis} binds {len(outputs)} outputs")
            for out, digit in zip(outputs, digits):
                self.env[str(out["name"])] = digit
            return
        call = self.call(prim)
        if target == "voyager::async_copy":
            kw = call.kwargs
            copy = Copy(name=call.name, src=kw["src"], dst=kw["dst"], indices=tuple(kw["indices"]),
                        sizes=tuple(kw["sizes"]), semaphore=kw["semaphore"],
                        post_count=int(kw.get("post_count", 1) if kw.get("post_count") is not None else 1),
                        dims=tuple(kw["dims"]) if kw.get("dims") else None,
                        strides=tuple(kw["strides"]) if kw.get("strides") else None,
                        transposed=bool(kw.get("transposed") or False),
                        pad=tuple(kw["pad"]) if kw.get("pad") else None,
                        pad_value=kw.get("pad_value"),
                        count=tuple(kw["count"]) if kw.get("count") else None,
                        path=tuple(self.path))
            self.events.append(copy)
            self.signal(copy.semaphore, copy.post_count)
            return
        if target == "voyager::async_wait":
            wait = Wait(name=call.name, semaphore=call.kwargs["semaphore"], path=tuple(self.path))
            self.consume(wait.semaphore, wait.name)
            self.events.append(wait)
            return
        if outputs and all("scalar" in out for out in outputs):
            raise UnsupportedConstruct(f"scalar operation {target!r} is not modelled")
        self.events.append(TensorOp(call=call, commit=commit, path=tuple(self.path)))

    def _bind(self, outputs: Sequence[Mapping[str, Any]], value: Any) -> None:
        if len(outputs) != 1:
            raise UnsupportedConstruct(f"scalar op binds {len(outputs)} outputs")
        self.env[str(outputs[0]["name"])] = value

    def fused(self, name: str, op: Mapping[str, Any], commit: str | None,
              dependencies: tuple[Ref, ...] = (), post: Ref | None = None) -> None:
        chain = tuple(self.call(p) for p in op["fused"].get("op_list", ()))
        if not chain:
            raise UnsupportedConstruct(f"fused operation {name!r} has an empty op_list")
        destinations = tuple(self.ref(out["destination"]) for out in op.get("outputs", ())
                             if "destination" in out)
        tiling = tuple(
            tuple((str(b["loop"]), int(b["bound"])) for b in level.get("loop_bounds", ()))
            for level in (op.get("tiling") or {}).get("level_tilings", ()))
        self.events.append(FusedCompute(name=name, chain=chain, destinations=destinations,
                                        tiling=tiling, commit=commit, dependencies=dependencies,
                                        post=post, path=tuple(self.path)))

    def loop(self, name: str, loop: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]]) -> None:
        if "for_loop" in loop:
            body = loop["for_loop"]
            iter_names = [str(a["name"]) for a in body.get("iter_args", ())]
            values = [self.scalar(a["initial"]) for a in body.get("iter_args", ())]
            start, end, step = (int(self.scalar(body[k])) for k in ("start", "end", "step"))
            if step <= 0:
                raise UnsupportedConstruct(f"loop {name!r} has non-positive step {step}")
            for iteration, iv in enumerate(range(start, end, step)):
                self.env[str(body["iv"])] = iv
                values = self._iterate(name, iteration, body["body"], iter_names, values)
        elif "while_loop" in loop:
            body = loop["while_loop"]
            iter_names = [str(a["name"]) for a in body.get("iter_args", ())]
            values = [self.scalar(a["initial"]) for a in body.get("iter_args", ())]
            iteration = 0
            while True:
                self.env.update(zip(iter_names, values))
                self.run(body["condition"].get("ops", ()))
                cond_yields = body["condition"].get("yields", ())
                if len(cond_yields) != 1:
                    raise UnsupportedConstruct(f"while {name!r} condition yields {len(cond_yields)}")
                if not self.scalar(cond_yields[0]):
                    break
                values = self._iterate(name, iteration, body["body"], iter_names, values)
                iteration += 1
        else:
            raise UnsupportedConstruct(f"loop {name!r} kind not modelled: {sorted(loop)}")
        if len(outputs) != len(values):
            raise UnsupportedConstruct(f"loop {name!r} has {len(values)} carried values but "
                                       f"{len(outputs)} outputs")
        for out, value in zip(outputs, values):
            self.env[str(out["name"])] = value

    def _iterate(self, name: str, iteration: int, region: Mapping[str, Any],
                 iter_names: list[str], values: list[Any]) -> list[Any]:
        self.env.update(zip(iter_names, values))
        self.path.append((name, iteration))
        try:
            self.run(region.get("ops", ()))
            yields = [self.scalar(y) for y in region.get("yields", ())]
        finally:
            self.path.pop()
        if len(yields) != len(iter_names):
            raise UnsupportedConstruct(f"loop {name!r} body yields {len(yields)} values for "
                                       f"{len(iter_names)} carried arguments")
        return yields

    def cond(self, cond: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]]) -> None:
        region = cond["true_region"] if self.scalar(cond["predicate"]) else cond.get("false_region", {})
        self.run(region.get("ops", ()))
        yields = [self.scalar(y) for y in region.get("yields", ())]
        if yields and len(yields) != len(outputs):
            raise UnsupportedConstruct(f"cond yields {len(yields)} values for {len(outputs)} outputs")
        for out, value in zip(outputs, yields):
            self.env[str(out["name"])] = value

    def commit(self, name: str, region: Mapping[str, Any]) -> None:
        dependencies = tuple(self.ref(d) for d in region.get("dependencies", ()))
        post = self.ref(region["post"]) if region.get("post") else None
        for dep in dependencies:
            self.consume(dep, name)
        for op in region["body"].get("ops", ()):
            if "fused" in op:
                self.fused(str(op["name"]), op, name, dependencies, post)
            else:
                self.operation(op, name)
        if post is not None:
            self.signal(post, 1)


def replay(model: Mapping[str, Any], *, check_semaphores: bool = True) -> Trace:
    """Evaluate ``model`` (the JSON form of ``voyager.Model``) and return its event trace."""
    replayer = _Replayer(model, check_semaphores)
    replayer.run(model.get("ops", ()))
    boxes = replayer.allocations
    return Trace(inputs=tuple(_box(b) for b in model.get("inputs", ())),
                 parameters=tuple(_box(b) for b in model.get("parameters", ())),
                 outputs=tuple(_box(b) for b in model.get("outputs", ())),
                 allocations=boxes, events=replayer.events, semaphores=replayer.semaphores)
