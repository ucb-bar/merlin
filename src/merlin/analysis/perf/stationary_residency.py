"""Did the emitted nest KEEP the operand the array holds, or shift it back in on every compute?

THE LEVER, and the measurement that makes it the biggest single item in the gap. On a target whose
compute array retains one operand across commands, a staging command either NAMES a block -- which
shifts that block through the array before any useful work issues -- or names the ABI's retain
sentinel, which keeps what the array already holds. A tiled nest whose inner loops vary only the
accumulator tile and the moving operand's window presents the SAME block to the array for a whole
run of consecutive positions. Naming it on every one of them shifts identical bytes in, over and
over, for nothing.

Measured on pinned elaborated RTL, bit-exact against the vendor kernel in the same binary: peeling
the first position of each run as the one naming command and retaining for the rest moved one
shape's execute-controller busy count from 97,301 to 51,657 (-24.6%), and -17.8% / -19.8% at a
7-row output geometry. The saving tracks the run length exactly (-16.5% at depth 7, -7..-8% at depth
8-16, ~0 at depth 2 -- where there is almost nothing to remove).

THE INSTRUCTION COUNT DOES NOT CHANGE, and that is the whole reason this has to be a demand on a
PROPERTY rather than on a count. This ABI takes a compute's destination from the staging command
before it, so one staging command per compute is the floor either way. What changes is which of them
NAME a block. A demand written as "issue fewer staging commands" would be unsatisfiable; a demand
written as "do not re-establish what the array already holds" is exactly the lever, and is the same
sentence :mod:`merlin.analysis.perf.load_state_residency` makes about the load path's configuration state.

TWO FLOORS, AND WHY ONE WOULD NOT HAVE BEEN ENOUGH. The identity of a held tile is the OFF-CHIP
SOURCE its bytes were moved in from, not the staging slot it occupies -- a schedule that recycles one
slot between two positions is exactly the schedule this family is about, and identifying by slot
would let the defect rename itself out of view. On that identity two floors are computable from the
candidate's own stream:

``runs`` -- the number of maximal runs of consecutive positions presenting distinct tiles.
    The array holds exactly one tile: whatever the last naming command shifted in. A command naming
    the tile the array ALREADY HOLDS establishes state already established and is provably inert,
    whatever the nest's loop order. This floor needs no admission and no reordering.

``distinct`` -- the number of distinct tiles presented anywhere in the program.
    The weaker floor alone is DODGEABLE, and the dodge is not hypothetical: a nest ordered so that a
    different block lands at every position has no consecutive repeats at all, passes the run-length
    demand outright, and pays the identical shift-in cost -- a demand that cannot fail. So where the
    reuse-ordered nest is admissible, the floor is the distinct count, which no loop order dodges.

ADMISSION IS DERIVED, NOT ASSUMED. The order that presents each tile once holds one output column
strip resident across the whole reduction, so a machine whose accumulator cannot hold what the
program commits has to drain and re-present, and no reordering removes it. The bound is the target's
own accumulator depth over its own array row count -- both extracted from its RTL. When the program's
committed output exceeds it, this REFUSES the stronger floor and demands only the run-length one,
saying so in the verdict, rather than failing a compiler for not doing something the machine forbids.

WHAT ELSE MAKES THE EXCESS AVOIDABLE, also derived. Retaining is only expressible where the ABI has a
retain form: a sentinel the staging command accepts in place of a block address, and more than one
compute class inheriting from that stager -- the "use what was just shifted in" form and the "keep
computing against what is resident" form. Both are read from the emitted ABI's own def-use model. An
ABI declaring one compute class per stager has no retain form, every naming command is compulsory,
and this family REFUSES rather than failing a compiler for lacking a mechanism its ABI does not have.

WHAT PASSING DOES NOT PROVE. Passing shows the redundant shift-in work is ABSENT. It is not a speedup
claim, and it is silent about the movement side: a nest that re-reads the same tile from memory
before presenting it again is counted as presenting one tile here and is a different defect, for the
movement-placement family to name. It is also silent about correctness -- whether a retained position
PAIRS its retain sentinel with the compute form that consumes resident state is the numeric oracle's
business, not this property's, and a candidate getting that wrong fails the oracle, not this check.

FAIL CLOSED EVERYWHERE. The staging and compute vocabularies come from the emitted ABI's own def-use
model; the retain sentinel from the selected support's ABI
(:func:`merlin.development.targetgen.rocc.decode.retain_sentinel`); the accumulator bound from the target's
extracted memories and array geometry. A naming command whose bytes have no observed producer, or a
producer whose off-chip operand does not resolve, yields ``REFUSED`` with the reason -- never a pass,
and never a substituted default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "Addressing",
    "FAIL",
    "PASS",
    "REFUSED",
    "RUN_LENGTH_FLOOR",
    "REUSE_ORDERED_FLOOR",
    "Vocabulary",
    "accumulator_tile_capacity",
    "array_row_block",
    "residency_findings",
    "residency_verdict",
    "resolve_addressing",
    "stationary_vocabulary",
]

PASS = "PASS"
FAIL = "FAIL"
REFUSED = "REFUSED"

#: The floor that needs no reordering: one naming command per run of consecutive distinct tiles.
RUN_LENGTH_FLOOR = "run_length"
#: The floor that a loop order cannot dodge: one naming command per distinct tile. Demanded only
#: where the target's accumulator admits the order that reaches it.
REUSE_ORDERED_FLOOR = "reuse_ordered"


class Vocabulary:
    """Which emitted-ABI classes stage the array's operand, and which decoded fields carry what.

    Every member is READ from the ABI's own def-use model (:mod:`merlin.analysis.perf.deps.rocc`) rather than
    spelled here, because that model is what the dependence graph, the motif extractor and the
    fixed-work projection already agree on. A second spelling of the same vocabulary is how two
    readers of one stream came to disagree about which command staged what.

    :attr:`retain_form` is the capability this family's demand rests on: an ABI where more than one
    compute class inherits its destination from the same stager has a form that consumes state the
    array already holds, and therefore a way to express the saving. One such class means there is no
    retain form, and the demand is refused rather than failed. It is STRUCTURAL -- a property of the
    ABI's def-use model -- while :attr:`sentinel`, the operand value that spells "keep what is held",
    is the selected target's and is ``None`` until a target is named; a verdict needs both.
    """

    __slots__ = ("stage_classes", "compute_classes", "staged_fields", "width_field", "sentinel", "retain_form")

    def __init__(
        self,
        *,
        stage_classes: frozenset[str],
        compute_classes: frozenset[str],
        staged_fields: tuple[str, ...],
        width_field: str,
        sentinel: int | None,
        retain_form: bool,
    ) -> None:
        self.stage_classes = stage_classes
        self.compute_classes = compute_classes
        self.staged_fields = staged_fields
        self.width_field = width_field
        self.sentinel = sentinel
        self.retain_form = retain_form

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_classes": sorted(self.stage_classes),
            "compute_classes": sorted(self.compute_classes),
            "staged_fields": list(self.staged_fields),
            "retain_form": self.retain_form,
            "sentinel_published": self.sentinel is not None,
        }


def stationary_vocabulary(target: object = None) -> Vocabulary | None:
    """The ABI's staging vocabulary, or ``None`` when the emitted ABI declares no staging relation.

    A target whose ABI has every command name its own operands has no command that stages another
    command's operand, so nothing is held across commands and this family's property is not about
    it. ``None`` is a real answer and the caller refuses on it rather than reading an empty cohort as
    clean. With ``target``, the vocabulary also carries that target's retain sentinel.
    """
    try:
        from merlin.analysis.perf.deps import rocc as _abi
    except Exception:  # noqa: BLE001 -- an ABI model that cannot be imported declares no vocabulary
        return None
    inherits = getattr(_abi, "INHERITS_DESTINATION", None)
    consumes = getattr(_abi, "CONSUMES", None)
    defines = getattr(_abi, "DEFINES", None)
    width_field = getattr(_abi, "WIDTH_FIELD", None)
    if not isinstance(inherits, Mapping) or not inherits:
        return None
    if not isinstance(consumes, Mapping) or not isinstance(defines, Mapping) or not isinstance(width_field, str):
        return None
    stage_classes = frozenset(str(value) for value in inherits.values())
    compute_classes = frozenset(str(key) for key in inherits)
    # A staging command CONSUMES the operand it presents and DEFINES the destination it stages for
    # the compute behind it. Only the consumed side is a candidate, and a field the ABI also lists as
    # a definition is the destination rather than the operand -- keeping it would let a destination
    # address be read as the tile presented to the array, comparing two unrelated quantities and
    # reporting the answer as a residency verdict.
    candidates = tuple(sorted(set(consumes) - set(defines)))
    if not candidates or not stage_classes:
        return None
    sentinel = _retain_sentinel(target) if target is not None else None
    # More than one compute class inheriting from one stager IS the retain form: the ABI can say
    # "use what was just shifted in" and "keep computing against what is resident" as two commands.
    per_stager: dict[str, int] = {}
    for compute, stager in inherits.items():
        per_stager[str(stager)] = per_stager.get(str(stager), 0) + 1
    retain_form = max(per_stager.values(), default=0) > 1
    return Vocabulary(
        stage_classes=stage_classes,
        compute_classes=compute_classes,
        staged_fields=candidates,
        width_field=width_field,
        sentinel=int(sentinel) if isinstance(sentinel, int) and not isinstance(sentinel, bool) else None,
        retain_form=retain_form,
    )


def _retain_sentinel(target: object) -> int | None:
    """``target``'s retain sentinel from its selected support, or ``None`` when it publishes none."""
    try:
        from merlin.development.targetgen.rocc import decode as _decode

        return _decode.retain_sentinel(str(target))
    except Exception:  # noqa: BLE001 -- support that cannot be resolved publishes no sentinel
        return None


class Addressing:
    """Which decoded fields address the staged file, for one trace read through one ABI."""

    __slots__ = ("staged_field", "define_field", "staged_file", "consumed_fields")

    def __init__(self, *, staged_field: str, define_field: str, staged_file: str, consumed_fields: tuple[str, ...]):
        #: The field a staging command carries to present a block to the array.
        self.staged_field = staged_field
        #: The field a movement command carries to say which staged run it fills.
        self.define_field = define_field
        self.staged_file = staged_file
        #: Every field that READS the staged file, a compute's own moving operand included.
        self.consumed_fields = consumed_fields


def resolve_addressing(instructions: Sequence[Any], vocab: Vocabulary) -> tuple[Addressing, None] | tuple[None, str]:
    """``(Addressing, None)`` for one trace, or ``(None, reason)`` when the ABI does not settle it.

    Public because the movement-placement family reads the SAME staged file through the SAME
    resolution: which field a staging command presents, and which field a movement command fills.
    Two spellings of that would be two readers of one stream disagreeing about which command produced
    a tile, which is the whole failure this resolution exists to prevent.
    """
    staged_field, why = _resolve_staged_field(instructions, vocab)
    if staged_field is None:
        return None, str(why)
    try:
        from merlin.analysis.perf.deps import rocc as _abi

        defines = dict(getattr(_abi, "DEFINES", {}) or {})
        consumes = dict(getattr(_abi, "CONSUMES", {}) or {})
    except Exception:  # noqa: BLE001
        return None, "the emitted ABI's def-use model could not be read, so no movement binds to a slot"
    staged_file = str(consumes.get(staged_field))
    # The movement command that fills that file is the one DEFINING it. Bound by FILE rather than by
    # name, and required to be unique: an ABI with two ways to define one file leaves which command
    # produced a tile's bytes ambiguous, and an ambiguous producer is how a re-presentation comes to
    # be counted as a fresh tile.
    define_fields = sorted(field for field, file_name in defines.items() if str(file_name) == staged_file)
    if len(define_fields) != 1:
        return None, (
            f"the emitted ABI binds {len(define_fields)} fields to the staged file {staged_file!r} "
            f"({define_fields}), so which command produced a presented tile's bytes is ambiguous"
        )
    consumed_fields = tuple(sorted(f for f, file_name in consumes.items() if str(file_name) == staged_file))
    return (
        Addressing(
            staged_field=staged_field,
            define_field=define_fields[0],
            staged_file=staged_file,
            consumed_fields=consumed_fields,
        ),
        None,
    )


def _resolve_staged_field(instructions: Sequence[Any], vocab: Vocabulary) -> tuple[str, None] | tuple[None, str]:
    """``(field, None)`` naming the operand staging commands present, or ``(None, reason)``.

    Resolved from THE TRACE rather than chosen, because the ABI's def-use map says which fields
    address which file but not which command carries which field. Exactly one candidate must appear
    on the staging commands: zero means nothing was presented, and two means the command carries two
    consumed operands and which one the array holds is ambiguous -- both refuse, because picking one
    would decide the claim on a field nobody established.
    """
    seen: set[str] = set()
    for instruction in instructions:
        if not isinstance(instruction, Mapping) or str(instruction.get("class") or "") not in vocab.stage_classes:
            continue
        decoded = instruction.get("decoded")
        if not isinstance(decoded, Mapping):
            continue
        for field in vocab.staged_fields:
            value = decoded.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                seen.add(field)
    if len(seen) == 1:
        return next(iter(seen)), None
    if not seen:
        return None, (
            f"no staging command in this trace carries any of the ABI's consumed operand fields "
            f"{list(vocab.staged_fields)}, so nothing was ever presented to the array"
        )
    return None, (
        f"staging commands in this trace carry more than one consumed operand field "
        f"({sorted(seen)}), so which one the array holds across commands is ambiguous"
    )


def accumulator_tile_capacity(target: str) -> int | None:
    """How many output tiles ``target``'s accumulator holds at once, or ``None`` when its RTL does not say.

    TWO derived facts, and neither is chosen: the accumulator's own row depth, and the array's row
    count -- which is what a tile IS on this kind of machine, one command's worth of the array's own
    granularity. The capacity is the number of whole tiles those rows make.

    This is the bound that decides whether the STRONGER floor is demandable. Under-reading it is the
    safe direction: it only ever drops the demand back to the run-length floor, which is provable
    without any reordering. So a target whose depth or geometry cannot be read yields ``None`` and
    the caller demands the weaker floor rather than inventing a bound.
    """
    depth = _accumulator_row_depth(target)
    rows = array_row_block(target)
    if depth is None or rows is None or rows < 1:
        return None
    capacity = depth // rows
    return capacity if capacity >= 1 else None


def _facts(target: str) -> Mapping[str, Any] | None:
    try:
        from merlin.development.targetgen.rtl import facts as _facts_module

        loaded = _facts_module.load_facts(str(target))
    except Exception:  # noqa: BLE001 -- a target whose facts cannot be read publishes no geometry
        return None
    if isinstance(loaded, Mapping):
        inner = loaded.get("facts")
        return inner if isinstance(inner, Mapping) else loaded
    return None


def array_row_block(target: str) -> int | None:
    """The compute array's row count, or ``None`` -- the tile granularity one command issues at.

    Public because both claim families count a reuse depth and a position count in this unit, and a
    second derivation of it would be a second answer to "how big is a tile on this machine".
    """
    arrays = (_facts(target) or {}).get("arrays")
    if not isinstance(arrays, Sequence) or isinstance(arrays, str):
        return None
    rows = [
        int(entry["rows"])
        for entry in arrays
        if isinstance(entry, Mapping) and isinstance(entry.get("rows"), int) and not isinstance(entry.get("rows"), bool)
    ]
    # The SMALLEST array row count, because a tile must fit every array the program can issue to and
    # over-reading it would over-state how much output the accumulator holds.
    return min(rows) if rows else None


def _accumulator_row_depth(target: str) -> int | None:
    """The row depth of the store a committed output accumulates in, or ``None``.

    The accumulator is identified by the DATAPATH the target declares for it -- the store whose name
    an accumulating datapath names -- rather than by position in a list, so a target publishing its
    memories in another order is read correctly. Zero matches means nothing bound a datapath to a
    store and more than one means the binding is ambiguous; both are UNKNOWN, because a capacity
    guessed from either is a fabricated hardware bound.
    """
    facts = _facts(target)
    if facts is None:
        return None
    datapaths, memories = facts.get("datapaths"), facts.get("memories")
    if not isinstance(datapaths, Sequence) or not isinstance(memories, Sequence):
        return None
    names = {str(e.get("name")) for e in datapaths if isinstance(e, Mapping) and isinstance(e.get("name"), str)}
    depths = [
        int(entry["depth"])
        for entry in memories
        if isinstance(entry, Mapping)
        and str(entry.get("name")) in names
        and isinstance(entry.get("depth"), int)
        and not isinstance(entry.get("depth"), bool)
        and entry["depth"] >= 1
    ]
    return depths[0] if len(depths) == 1 else None


def _flag_mask(target: str, file_name: str) -> int:
    """Bits to strip from an address in ``file_name`` before comparing it; 0 when underivable.

    Left in, the same output tile addressed once with a mode bit set and once without reads as two
    tiles, and the committed working set this family admits on silently inflates -- which drops the
    demand to the weaker floor. Zero is the conservative answer in that same direction.
    """
    try:
        from merlin.analysis.perf.deps.rocc import flag_masks_for

        masks = flag_masks_for(str(target))
    except Exception:  # noqa: BLE001 -- a target whose masks are underivable strips nothing by guess
        return 0
    value = masks.get(file_name) if isinstance(masks, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _source_identity(operand: Any) -> tuple | None:
    """A hashable identity for the OFF-CHIP source of one moved tile, or ``None`` if unresolved.

    Compared structurally by the decoder's own resolution -- a constant by its value, a bound
    argument by which argument and which byte offset -- never by text, and never falling back to
    something that would make two different tiles compare equal.
    """
    if not isinstance(operand, Mapping):
        return None
    kind = operand.get("kind")
    if kind == "const":
        raw = operand.get("raw")
        return ("const", raw) if isinstance(raw, int) and not isinstance(raw, bool) else None
    if kind == "argbase":
        index, offset = operand.get("arg_index"), operand.get("offset")
        if isinstance(index, int) and not isinstance(index, bool) and isinstance(offset, int):
            return ("argbase", index, offset)
        return None
    return None


def residency_verdict(
    trace: Any,
    *,
    target: object,
    vocabulary: Vocabulary | None = None,
    accumulator_tiles: int | None = None,
) -> dict[str, Any]:
    """Decide the stationary-operand residency property for one emitted trace.

    Returns ``{"verdict", "reason", ...}``. ``FAIL`` names the commands that re-presented a tile and
    reports the excess against the DEMANDED floor, which the verdict always names (``floor_kind``);
    ``REFUSED`` names what could not be derived. A trace that presents no stationary operand at all is
    REFUSED, not passed: a property checked over nothing is the check that cannot fail.
    """
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    instructions = trace.get("instructions")
    if not isinstance(instructions, Sequence) or isinstance(instructions, str):
        raise TypeError("trace instructions must be a sequence")

    vocab = vocabulary if vocabulary is not None else stationary_vocabulary(target)
    if vocab is None:
        return {
            "verdict": REFUSED,
            "reason": (
                "the emitted ABI declares no command that stages another command's operand, so "
                "nothing is held in the array across commands and this demand has no subject here"
            ),
        }
    if not vocab.retain_form:
        return {
            "verdict": REFUSED,
            "reason": (
                "the emitted ABI declares no retain form -- no sentinel the staging command accepts "
                "in place of a block address together with a compute form that consumes resident "
                "state -- so every naming command is compulsory and no re-presentation here has been "
                "shown to be avoidable"
            ),
            "vocabulary": vocab.to_dict(),
        }
    if vocab.sentinel is None:
        return {
            "verdict": REFUSED,
            "reason": (
                "the selected support publishes no retain sentinel for this target, so a position that "
                "keeps the array's operand cannot be told from one that names a block, and no "
                "re-presentation can be identified"
            ),
            "vocabulary": vocab.to_dict(),
        }
    addressing, why = resolve_addressing(instructions, vocab)
    if addressing is None:
        return {"verdict": REFUSED, "reason": str(why), "vocabulary": vocab.to_dict()}
    staged_field, define_field = addressing.staged_field, addressing.define_field

    if accumulator_tiles is None:
        accumulator_tiles = accumulator_tile_capacity(str(target))
    destination_mask = _flag_mask(str(target), "acc")

    staged: dict[int, tuple] = {}
    named: list[dict[str, Any]] = []
    retained = 0
    held: tuple | None = None
    redundant: list[dict[str, Any]] = []
    runs = 0
    distinct: set[tuple] = set()
    destinations: set[int] = set()

    for index, instruction in enumerate(instructions):
        if not isinstance(instruction, Mapping):
            continue
        decoded = instruction.get("decoded")
        decoded = decoded if isinstance(decoded, Mapping) else {}
        cls = str(instruction.get("class") or "")

        if cls not in vocab.stage_classes:
            # A movement command DEFINES a run of staged slots; the off-chip operand it read is the
            # identity of the bytes it put there. An unresolved operand is RECORDED as unresolved and
            # poisons any naming command presenting those slots, rather than being skipped.
            base = decoded.get(define_field)
            if isinstance(base, int) and not isinstance(base, bool):
                width = decoded.get(vocab.width_field)
                span = width if isinstance(width, int) and not isinstance(width, bool) and width > 0 else 1
                source = _source_identity(decoded.get("dram"))
                marker: tuple = source if source is not None else ("unresolved", index)
                for slot in range(base, base + span):
                    staged[slot] = marker
            continue

        destination = _destination_address(decoded, addressing)
        if destination is not None:
            destinations.add(destination & ~destination_mask)

        address = decoded.get(staged_field)
        if not isinstance(address, int) or isinstance(address, bool):
            return {
                "verdict": REFUSED,
                "reason": (
                    f"the staging command at #{index} carries no resolved {staged_field!r}, so what "
                    "it presented to the array cannot be identified"
                ),
                "named_count": len(named),
            }
        if vocab.sentinel is not None and address == vocab.sentinel:
            retained += 1
            continue
        identity = staged.get(address)
        if identity is None:
            return {
                "verdict": REFUSED,
                "reason": (
                    f"the staging command at #{index} presents slot {address}, whose bytes have no "
                    "observed producer in this trace, so which tile it presented is UNKNOWN and no "
                    "other command can be shown to re-present it"
                ),
                "named_count": len(named),
            }
        if identity[0] == "unresolved":
            return {
                "verdict": REFUSED,
                "reason": (
                    f"the staging command at #{index} presents bytes moved in at #{identity[1]}, "
                    "whose off-chip source does not resolve, so re-presenting that tile cannot be "
                    "told apart from presenting a different one"
                ),
                "named_count": len(named),
            }
        named.append({"index": index, "tile": identity})
        distinct.add(identity)
        if held is not None and identity == held:
            redundant.append({"index": index})
        else:
            runs += 1
            held = identity

    if not named:
        return {
            "verdict": REFUSED,
            "reason": (
                "the trace names no stationary block at all, so nothing was ever presented to the "
                "array and there is nothing this demand could have observed"
            ),
            "named_count": 0,
            "retained": retained,
        }

    issued = len(named)
    committed = len(destinations)
    admits_reorder = isinstance(accumulator_tiles, int) and committed >= 1 and committed <= accumulator_tiles
    floor = len(distinct) if admits_reorder else runs
    floor_kind = REUSE_ORDERED_FLOOR if admits_reorder else RUN_LENGTH_FLOOR
    row: dict[str, Any] = {
        "named_count": issued,
        "runs": runs,
        "distinct_tiles": len(distinct),
        "retained": retained,
        "redundant": redundant,
        "staged_field": staged_field,
        "committed_output_tiles": committed,
        "accumulator_tiles": accumulator_tiles,
        "floor": floor,
        "floor_kind": floor_kind,
        "vocabulary": vocab.to_dict(),
    }
    basis = (
        f"the program commits {committed} output tile(s), inside the accumulator's derived capacity "
        f"of {accumulator_tiles}, so the order presenting each tile once is one this machine can run "
        "and the floor is the distinct-tile count"
        if admits_reorder
        else (
            f"the accumulator's derived capacity is {accumulator_tiles!r} against {committed} "
            "committed output tile(s), so the order presenting each tile once is NOT admitted here "
            "and the floor is the run-length one, which no reordering is needed to reach"
        )
    )
    if issued > floor:
        return {
            **row,
            "verdict": FAIL,
            "reason": (
                f"{issued} staging commands name a block against a floor of {floor} ({floor_kind}): "
                f"{issued - floor} of them shift bytes through the array that it has already been "
                f"given. {basis}. The nest presents {len(distinct)} distinct tile(s) across {runs} "
                f"run(s) of consecutive positions"
                + (
                    "; the re-presentations within a run are at "
                    + ", ".join(f"#{entry['index']}" for entry in redundant[:4])
                    + " -- the signature of the block being named at every position of a run whose "
                    "inner extents vary only the accumulator tile and the moving operand's window"
                    if redundant
                    else "; no run repeats a tile, so the excess is the SAME tile presented again "
                    "later in the program -- a loop order that scatters each block's positions "
                    "instead of making them consecutive"
                )
            ),
            "excess": issued - floor,
            "mean_run_length": round(issued / runs, 4) if runs else None,
        }
    return {
        **row,
        "verdict": PASS,
        "excess": 0,
        "mean_run_length": round(issued / runs, 4) if runs else None,
        "reason": (
            f"{issued} naming command(s) meet the floor of {floor} ({floor_kind}), with {retained} "
            f"command(s) retaining what the array already held: {basis}. This says the redundant "
            "shift-in work is absent; it is not a claim that the movement schedule is otherwise good"
        ),
    }


def _destination_address(decoded: Mapping[str, Any], addressing: Addressing) -> int | None:
    """The output tile a staging command commits to, or ``None`` when it names none.

    Read from the ABI's DEFINES map, excluding the staged file -- the staging command defines its
    compute's destination in the OTHER file, and that set is what the accumulator bound is checked
    against.
    """
    try:
        from merlin.analysis.perf.deps import rocc as _abi

        defines = dict(getattr(_abi, "DEFINES", {}) or {})
    except Exception:  # noqa: BLE001
        return None
    for field, file_name in defines.items():
        if str(file_name) == addressing.staged_file:
            continue
        value = decoded.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def residency_findings(trace: Any, *, target: object, vocabulary: Vocabulary | None = None) -> list[str]:
    """The verdict as advisory diagnostic lines, for callers that collect findings rather than verdicts.

    A ``PASS`` yields no line. Both ``FAIL`` and ``REFUSED`` yield one, because a refusal that
    produced silence would read exactly like a clean trace -- which is how a check that cannot fail
    gets shipped.
    """
    verdict = residency_verdict(trace, target=target, vocabulary=vocabulary)
    if verdict["verdict"] == PASS:
        return []
    return [f"stationary-operand residency ({verdict['verdict'].lower()}): {verdict['reason']}"]
