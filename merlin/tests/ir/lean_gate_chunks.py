"""Representative kernel chunks of the retained model capsules, for fast real-lowering gate tests.

A chunk is a contiguous window of a capsule's top-level ops re-emitted as a standalone function:
values the window reads from outside become arguments and values it defines that are read later
become results. Its ``pruned`` variant returns only the value in the window's middle third with the
most producers inside the window, so a real live kernel remains and everything that does not feed
it (real kernel code, quantization included) is dead and must be pruned.

Windows are chosen by the capsules' own ``prov.family`` tags: for every family a capsule contains,
the window that starts just before its first op of that family, with overlapping windows merged.
The selection is deterministic and depends only on the capsule text, so every family any capsule
uses goes through the real passes, with its surrounding quantization, layout and dead code, in
seconds instead of whole-model minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from merlin.common.paths import repo_root

CAPSULES = (
    "SY_micro_model",
    "M0_small_llama_gemmini",
    "M2_microvit_gemmini",
    "M4_smolvla_denoise_gemmini",
    "SY_model_resnet50",
    "SY_model_tiny_llama",
)
WINDOW = 96  # top-level ops per chunk
LEAD = 8  # ops kept before a family's first op, so its producers stay inside the chunk


@dataclass(frozen=True)
class Chunk:
    capsule: str
    family: str
    start: int
    text: str
    pruned: str

    @property
    def name(self) -> str:
        return f"{self.capsule}:{self.family}@{self.start}"


def _family(op) -> str | None:
    from xdsl.dialects.builtin import StringAttr

    value = op.attributes.get("prov.family")
    return value.data if isinstance(value, StringAttr) else None


def _emit(ops: list, later: list, *, keep: object | None = None) -> str | None:
    """``ops`` as a standalone ``@forward``; None when the window reads a value defined after it."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, BlockArgument, OpResult, Region

    from merlin.xdsl_dialects._common import text

    inside_ops, inside_blocks = set(), set()
    for op in ops:
        for nested in op.walk():
            inside_ops.add(nested)
            for region in nested.regions:
                inside_blocks.update(region.blocks)
    external = []
    for op in ops:
        for nested in op.walk():
            for value in nested.operands:
                owner = value.op if isinstance(value, OpResult) else value.block
                inside = owner in inside_ops if isinstance(value, OpResult) else owner in inside_blocks
                if not inside and value not in external:
                    if isinstance(value, BlockArgument) or value.op.parent_block() is ops[0].parent_block():
                        external.append(value)
                    else:
                        return None
    later_ops = [nested for op in later for nested in op.walk()]
    read_later = {value for op in later_ops for value in op.operands}
    results = [keep] if keep is not None else [r for op in ops for r in op.results if r in read_later]
    if not results:
        results = list(ops[-1].results)
    if not results:
        return None
    block = Block(arg_types=[value.type for value in external])
    mapping = dict(zip(external, block.args, strict=True))
    for op in ops:
        clone = op.clone(value_mapper=mapping)
        block.add_op(clone)
        mapping.update(zip(op.results, clone.results, strict=True))
    block.add_op(func.ReturnOp(*[mapping[r] for r in results]))
    function = func.FuncOp("forward", ([v.type for v in external], [r.type for r in results]), Region(block))
    return text(ModuleOp([function]), generic=True)


def _deepest(window: list):
    """The tensor result in the window's middle third with the most producers inside the window."""
    producers: dict[object, set[int]] = {}
    for index, op in enumerate(window):
        seen = {index}
        for nested in op.walk():
            for value in nested.operands:
                seen |= producers.get(value, set())
        for result in op.results:
            producers[result] = seen
    middle = window[len(window) // 3 : 2 * len(window) // 3]
    candidates = [r for op in middle for r in op.results if "tensor" in str(r.type)]
    return max(candidates, key=lambda r: len(producers[r]), default=None)


@cache
def chunks(capsule: str) -> tuple[Chunk, ...]:
    from merlin.llvmlower.passes_xdsl import parse_mlir_text

    source = repo_root() / "merlin/contract/capsules/model" / capsule / "capsule.interface.mlir"
    module = parse_mlir_text(source.read_text())
    function = max(
        (op for op in module.walk() if op.name == "func.func"), key=lambda f: len(list(f.body.blocks[0].ops))
    )
    body = [op for op in function.body.blocks[0].ops if op.name != "func.return"]
    terminator = [op for op in function.body.blocks[0].ops if op.name == "func.return"]
    first: dict[str, int] = {}
    for index, op in enumerate(body):
        family = _family(op)
        if family is not None:
            first.setdefault(family, index)
    selected, covered = [], -1
    for family, index in sorted(first.items(), key=lambda item: item[1]):
        start = max(0, index - LEAD)
        if index < covered:  # this family's first op is already inside the previous window
            continue
        end = min(len(body), start + WINDOW)
        window = body[start:end]
        emitted = _emit(window, body[end:] + terminator)
        keep = _deepest(window)
        pruned = _emit(window, [], keep=keep) if keep is not None else None
        if emitted is not None and pruned is not None:
            selected.append(Chunk(capsule, family, start, emitted, pruned))
            covered = end
    return tuple(selected)


def all_chunks() -> list[Chunk]:
    return [chunk for capsule in CAPSULES for chunk in chunks(capsule)]
