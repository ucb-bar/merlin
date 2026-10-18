"""Final encoded-program checks and conservative continuation-safe segmentation.

Instruction decoding, instruction semantics and launch implementation belong to
the selected target. The shared algorithm never guesses a word width, branch
reach, address window, completion rule or preserved resource. Every check reads
the final bytes through that explicitly qualified decoder.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class AddressWindow:
    space: str
    start: int
    size: int

    def __post_init__(self):
        if not isinstance(self.space, str) or not self.space:
            raise ValueError("address windows require an explicit space")
        if type(self.start) is not int or self.start < 0 or type(self.size) is not int or self.size <= 0:
            raise ValueError("address windows require a nonnegative base and positive byte size")


@dataclass(frozen=True)
class EncodedAccess:
    """An exact byte interval read or written by one decoded instruction."""

    space: str
    start: int
    size: int
    writes: bool = False

    def __post_init__(self):
        AddressWindow(self.space, self.start, self.size)
        if type(self.writes) is not bool:
            raise ValueError("encoded access direction must be explicit")


@dataclass(frozen=True)
class BranchRule:
    name: str
    minimum: int
    maximum: int
    scale: int
    origin: str  # instruction or next_instruction

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or type(self.minimum) is not int
            or type(self.maximum) is not int
            or self.minimum > self.maximum
            or type(self.scale) is not int
            or self.scale <= 0
            or self.origin not in {"instruction", "next_instruction"}
        ):
            raise ValueError("branch rules require explicit displacement bounds, byte scale and origin")


@dataclass(frozen=True)
class EncodedBranch:
    rule: str
    displacement: int

    def __post_init__(self):
        if not isinstance(self.rule, str) or not self.rule or type(self.displacement) is not int:
            raise ValueError("decoded branches need an explicit rule and integer encoded displacement")


@dataclass(frozen=True)
class DecodedInstruction:
    offset: int
    size: int
    accesses: tuple[EncodedAccess, ...] = ()
    branches: tuple[EncodedBranch, ...] = ()
    completion: str = "unknown"  # synchronous, asynchronous, drain or unknown
    relocatable: bool = False
    fallthrough: bool = True


@dataclass(frozen=True)
class ProgramLimits:
    max_bytes: int
    max_instructions: int | None
    instruction_alignment: int
    windows: tuple[AddressWindow, ...]
    branches: tuple[BranchRule, ...]

    def __post_init__(self):
        if (
            type(self.max_bytes) is not int
            or self.max_bytes <= 0
            or type(self.instruction_alignment) is not int
            or self.instruction_alignment <= 0
            or (
                self.max_instructions is not None
                and (type(self.max_instructions) is not int or self.max_instructions <= 0)
            )
        ):
            raise ValueError("final program limits require positive explicit encoded budgets and alignment")
        if (
            type(self.windows) is not tuple
            or any(type(window) is not AddressWindow for window in self.windows)
            or type(self.branches) is not tuple
            or any(type(rule) is not BranchRule for rule in self.branches)
        ):
            raise ValueError("final program windows and branch rules must be explicit immutable declarations")
        if len({rule.name for rule in self.branches}) != len(self.branches):
            raise ValueError("branch rule identities must be unique")


@dataclass(frozen=True)
class ProgramDecoder:
    decode: Callable[[bytes], tuple[DecodedInstruction, ...]]
    configuration_identity: str
    qualification_identity: str

    def __post_init__(self):
        if not callable(self.decode) or any(
            not isinstance(value, str) or not value
            for value in (self.configuration_identity, self.qualification_identity)
        ):
            raise ValueError("final checks require a selected decoder and its independent qualification identity")


@dataclass(frozen=True)
class Continuation:
    """Selected launch contract, independent of an optimization candidate.

    Launches execute sequentially, begin at offset zero and establish completion
    and visibility before the following launch. Only named spaces persist.
    This declaration needs separate target/runtime execution qualification.
    """

    identity: str
    preserved_spaces: tuple[str, ...]
    sequential_completion: bool
    starts_at_zero: bool
    max_segments: int
    preserved_configuration: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.identity, str)
            or not self.identity
            or self.sequential_completion is not True
            or self.starts_at_zero is not True
            or type(self.max_segments) is not int
            or self.max_segments <= 0
            or type(self.preserved_spaces) is not tuple
            or len(set(self.preserved_spaces)) != len(self.preserved_spaces)
            or any(not isinstance(space, str) or not space for space in self.preserved_spaces)
            or type(self.preserved_configuration) is not tuple
            or len(set(self.preserved_configuration)) != len(self.preserved_configuration)
            or any(not isinstance(key, str) or not key for key in self.preserved_configuration)
        ):
            raise ValueError("segmentation requires explicit sequential launch, entry and state-continuation semantics")


@dataclass(frozen=True)
class FinalProgram:
    segments: tuple[bytes, ...]
    report: dict


def _library_sources():
    return {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _decode(image, limits, decoder):
    if type(image) is not bytes or not image:
        raise ValueError("final program checking requires actual nonempty encoded bytes")
    instructions = decoder.decode(image)
    if type(instructions) is not tuple or not instructions:
        raise ValueError("decoder must cover the complete final encoded program")
    cursor = 0
    targets = []
    rules = {rule.name: rule for rule in limits.branches}
    boundaries = set()
    for instruction in instructions:
        if (
            type(instruction) is not DecodedInstruction
            or type(instruction.offset) is not int
            or type(instruction.size) is not int
            or instruction.offset != cursor
            or instruction.size <= 0
            or instruction.offset % limits.instruction_alignment
            or instruction.size % limits.instruction_alignment
            or instruction.completion not in {"synchronous", "asynchronous", "drain", "unknown"}
            or type(instruction.relocatable) is not bool
            or type(instruction.fallthrough) is not bool
            or type(instruction.accesses) is not tuple
            or type(instruction.branches) is not tuple
        ):
            raise ValueError(
                "final decoding omitted bytes, overlapped instructions or has unknown instruction structure"
            )
        boundaries.add(cursor)
        cursor += instruction.size
        if cursor > len(image):
            raise ValueError("decoded instruction extends beyond the actual program bytes")
        for access in instruction.accesses:
            if type(access) is not EncodedAccess or not any(
                window.space == access.space
                and window.start <= access.start
                and access.start + access.size <= window.start + window.size
                for window in limits.windows
            ):
                raise ValueError("actual encoded memory access is outside declared address windows")
        for branch in instruction.branches:
            if type(branch) is not EncodedBranch:
                raise ValueError("unknown final branch semantics")
            rule = rules.get(branch.rule)
            if (
                rule is None
                or type(branch.displacement) is not int
                or not rule.minimum <= branch.displacement <= rule.maximum
            ):
                raise ValueError("actual branch has an undeclared encoding or exceeds declared reach")
            origin = instruction.offset + (instruction.size if rule.origin == "next_instruction" else 0)
            target = origin + branch.displacement * rule.scale
            targets.append((instruction.offset, target))
    if cursor != len(image):
        raise ValueError("decoder left actual encoded bytes unchecked")
    if any(target not in boundaries for _source, target in targets):
        raise ValueError("actual branch does not reach an instruction boundary inside this program")
    # Decoder implementations may cache returned records. Analyses retain
    # host-owned values so a later callback cannot rewrite earlier facts.
    owned = tuple(replace(i, accesses=tuple(replace(a) for a in i.accesses),
                          branches=tuple(replace(b) for b in i.branches)) for i in instructions)
    return owned, tuple(targets)


def _pending_after(instructions, targets):
    """May-pending completion over actual branches, without executing loop trips."""
    by_offset = {instruction.offset: index for index, instruction in enumerate(instructions)}
    successors = [[] for _ in instructions]
    for source, target in targets:
        successors[by_offset[source]].append(by_offset[target])
    for index, instruction in enumerate(instructions):
        if instruction.fallthrough:
            successors[index].append(index + 1)
        elif not instruction.branches:
            successors[index].append(len(instructions))
    after = [set() for _ in instructions]
    seen, work, exits = set(), deque([(0, False)]), set()
    while work:
        index, pending = work.popleft()
        if index == len(instructions):
            exits.add(pending)
            continue
        if (index, pending) in seen:
            continue
        seen.add((index, pending))
        completion = instructions[index].completion
        if completion == "asynchronous":
            pending = True
        elif completion == "drain":
            pending = False
        after[index].add(pending)
        work.extend((successor, pending) for successor in successors[index])
    return after, exits


def _fits(size, count, limits):
    return size <= limits.max_bytes and (limits.max_instructions is None or count <= limits.max_instructions)


def _safe_cuts(instructions, targets, continuation, outputs, physical_spaces=()):
    # Conservative byte-interval liveness. A write kills an earlier value only
    # when it completely covers the read interval. Partial writes over-retain.
    mapping = dict(physical_spaces)
    def canonical(access):
        return replace(access, space=mapping.get(access.space, access.space))
    preserved = {mapping.get(space, space) for space in continuation.preserved_spaces}
    instructions = tuple(replace(instruction, accesses=tuple(canonical(a) for a in instruction.accesses))
                         for instruction in instructions)
    live = {canonical(access) for access in outputs if canonical(access).space not in preserved}
    may_kill = not targets and all(instruction.completion == "synchronous" for instruction in instructions)
    retained = {}
    for instruction in reversed(instructions):
        retained[instruction.offset + instruction.size] = bool(live)
        for access in instruction.accesses:
            if access.writes and may_kill:
                live = {
                    item
                    for item in live
                    if not (
                        item.space == access.space
                        and access.start <= item.start
                        and item.start + item.size <= access.start + access.size
                    )
                }
        live.update(
            access
            for access in instruction.accesses
            if not access.writes and access.space not in preserved
        )
    safe = {0}
    pending_states, _exits = _pending_after(instructions, targets)
    ends = [instruction.offset + instruction.size for instruction in instructions]
    crossing_changes = [0] * (len(ends) + 1)
    for source, target in targets:
        low, high = sorted((source, target))
        crossing_changes[bisect_right(ends, low)] += 1
        crossing_changes[bisect_right(ends, high)] -= 1
    crossing = 0
    for index, instruction in enumerate(instructions):
        if instruction.completion == "unknown" or not instruction.relocatable:
            raise ValueError("segmentation cannot move unknown completion or position-dependent instructions")
        if not instruction.fallthrough and not instruction.branches:
            raise ValueError("segmentation cannot continue after an instruction with unknown return/exit semantics")
        cut = instruction.offset + instruction.size
        crossing += crossing_changes[index]
        final_cut = cut == ends[-1]
        if True not in pending_states[index] and not crossing and (final_cut or not retained[cut]):
            safe.add(cut)
    return safe


def finalize_program(
    image: bytes,
    limits: ProgramLimits,
    decoder: ProgramDecoder,
    *,
    continuation: Continuation | None = None,
    outputs: tuple[EncodedAccess, ...] | None = None,
    selected_contracts=None,
    continuation_qualification_identity=None,
    physical_spaces=(),
) -> FinalProgram:
    """Count/check actual final instructions, then slice only legal closed regions.

    Segmentation preserves bytes and internal relative branches. It does not
    invent branch relaxation or spill code. If a branch or live volatile value
    spans the required cut, the selected target needs another declared lowering.
    """
    supplied_limits, supplied_decoder, supplied_continuation, supplied_outputs = limits, decoder, continuation, outputs
    def input_state():
        return (
            asdict(supplied_limits) if type(supplied_limits) is ProgramLimits else None,
            asdict(supplied_continuation) if type(supplied_continuation) is Continuation else None,
            (supplied_decoder.configuration_identity, supplied_decoder.qualification_identity,
             id(supplied_decoder.decode))
            if type(supplied_decoder) is ProgramDecoder else None,
            tuple(asdict(a) if type(a) is EncodedAccess else a for a in supplied_outputs)
            if type(supplied_outputs) is tuple else None,
        )
    selected_state = input_state()
    sources = _library_sources()
    def fresh():
        if sources != _library_sources():
            raise ValueError("final program checker changed during checking; replan")
        if input_state() != selected_state:
            raise ValueError("final program limits, decoder or continuation changed during checking; replan")
        if selected_contracts is not None:
            selected_contracts.revalidate()

    binding = None
    if selected_contracts is not None:
        from .selected_contracts import SelectedSchedulingContracts

        if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.program is None:
            raise ValueError("final program requires explicitly selected RTL/software program contracts")
        selected_contracts.revalidate()
        if physical_spaces:
            raise ValueError("selected physical program aliases cannot be overridden")
        limits, decoder, continuation, binding = selected_contracts.program.bind(
            selected_contracts, limits, decoder, continuation, continuation_qualification_identity,
        )
        physical_spaces = selected_contracts.program.physical_spaces
        if outputs is not None and (type(outputs) is not tuple or any(type(a) is not EncodedAccess for a in outputs)):
            raise ValueError("final program publications require immutable typed original intervals")
        outputs = tuple(dict.fromkeys((*(outputs or ()), *selected_contracts.program.published())))
    elif continuation_qualification_identity is not None:
        raise ValueError("continuation qualification requires selected program contracts")
    if type(limits) is not ProgramLimits or type(decoder) is not ProgramDecoder:
        raise ValueError("final checks require typed original limits and decoder selection")
    if (type(physical_spaces) is not tuple
            or any(type(row) is not tuple or len(row) != 2 or any(type(s) is not str or not s for s in row)
                   for row in physical_spaces)
            or len(dict(physical_spaces)) != len(physical_spaces)
            or physical_spaces and not {w.space for w in limits.windows} <= set(dict(physical_spaces))
            or set(dict(physical_spaces)) - ({name for name, _ in selected_contracts.program.physical_spaces}
                                            if selected_contracts is not None else {w.space for w in limits.windows})):
        raise ValueError("final physical aliases require unique declared byte spaces and canonical backings")
    if continuation is not None and type(continuation) is not Continuation:
        raise ValueError("final checks require typed original continuation semantics")
    decode = decoder.decode
    def checked_decode(data):
        fresh()
        actual = decode(data)
        fresh()
        return actual
    decoder = replace(decoder, decode=checked_decode)
    instructions, targets = _decode(image, limits, decoder)
    _pending, exits = _pending_after(instructions, targets)
    if True in exits:
        raise ValueError("final program has a control-flow path ending before explicit device completion")
    starts = {instruction.offset: index for index, instruction in enumerate(instructions)}
    starts[len(image)] = len(instructions)
    cuts = [0]
    if _fits(len(image), len(instructions), limits):
        cuts.append(len(image))
    else:
        if continuation is None:
            raise ValueError("final encoded program exceeds budget; no qualified continuation interface selected")
        if (
            outputs is None
            or type(outputs) is not tuple
            or any(type(access) is not EncodedAccess or access.writes for access in outputs)
        ):
            raise ValueError("segmentation requires the complete original published output intervals")
        for access in outputs:
            if not any(
                window.space == access.space
                and window.start <= access.start
                and access.start + access.size <= window.start + window.size
                for window in limits.windows
            ):
                raise ValueError("published output interval is outside declared address windows")
        if selected_contracts is not None:
            selected_contracts.program.verify_configuration(
                selected_contracts, image, continuation, decoder.qualification_identity,
            )
        safe = _safe_cuts(instructions, targets, continuation, outputs, physical_spaces)
        ordered_cuts = sorted(safe)
        # With byte/count upper bounds and cut legality independent of the
        # preceding cut, the farthest legal cut minimizes launches.
        while cuts[-1] < len(image):
            start = cuts[-1]
            end_limit = min(start + limits.max_bytes, len(image))
            if limits.max_instructions is not None:
                last_instruction = min(starts[start] + limits.max_instructions, len(instructions)) - 1
                end_limit = min(end_limit, instructions[last_instruction].offset + instructions[last_instruction].size)
            end = ordered_cuts[bisect_right(ordered_cuts, end_limit) - 1]
            if end <= start:
                raise ValueError("no completion-safe, state-preserving cut fits the final encoded budget")
            cuts.append(end)
            if len(cuts) - 1 > continuation.max_segments:
                raise ValueError("final segmentation exceeds the declared launch budget")
    segments = tuple(image[start:end] for start, end in zip(cuts, cuts[1:]))
    actual_counts = []
    for start, end, segment in zip(cuts, cuts[1:], segments):
        decoded, _targets = _decode(segment, limits, decoder)
        expected = tuple(
            replace(instruction, offset=instruction.offset - start)
            for instruction in instructions[starts[start] : starts[end]]
        )
        if decoded != expected:
            raise ValueError("realized segment instruction effects differ from the final input program")
        if not _fits(len(segment), len(decoded), limits):
            raise ValueError("realized segment exceeds the final encoded budget")
        actual_counts.append(len(decoded))
    if b"".join(segments) != image:
        raise ValueError("final segmentation changed instruction bytes")
    report = {
        "schema": "merlin.final_program_limits.v1",
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "configuration_identity": decoder.configuration_identity,
        "decoder_qualification_identity": decoder.qualification_identity,
        "limits": asdict(limits),
        "encoded_bytes": len(image),
        "instruction_count": sum(actual_counts),
        "branch_count": len(targets),
        "segment_bytes": list(map(len, segments)),
        "instruction_budget": "unknown" if limits.max_instructions is None else "checked",
        "segment_sha256": [hashlib.sha256(segment).hexdigest() for segment in segments],
        "continuation": asdict(continuation) if len(segments) > 1 else None,
        "timing": "UNKNOWN",
        "execution_qualification": "not_established",
        "completion_check": "unknown" if any(i.completion == "unknown" for i in instructions)
        else "conservative_control_flow",
        "physical_spaces": dict(physical_spaces),
        "library_sources_sha256": sources,
    }
    fresh()
    if selected_contracts is not None:
        report["selected_contracts"] = selected_contracts.record(scopes=(
            "final_program_limits", "physical_address_windows", "branch_encodings", "launch_state_continuation",
        ))
        report["selected_contracts"]["program"] = binding
    return FinalProgram(segments, report)
