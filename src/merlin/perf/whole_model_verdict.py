"""Read one whole-model program's log and say what it measured -- or refuse to.

Every machine that runs the whole-model program (a board, a functional model, an elaborated-RTL
emulator) prints the program's own protocol: one line per group, a whole-window cycle count and a
classification.  The rules for reading that protocol live here, once, so a number from one machine
and a number from another are read by the same code and cannot disagree about what a line means.
Staging, queue jobs and device identity are :mod:`.machines`' concern, not this module's.

THE ORDER OF THE RULES IS THE POINT.

1. **Every expected group line must be present before anything else is read.**  Absence of data
   has been misread as a pass three times in one day: a log cut short, a run stopped at a cycle
   budget, a line split by simulator chatter -- each left fewer group lines, and a reader counting
   only the lines it found saw nothing wrong.  So the first check is the set of groups the log
   printed against the set the build expects.  A mismatch is ``REFUSED``: no cycle count, no
   per-group number, no correctness verdict is reported from a log that did not finish saying what
   it was built to say.

2. **Correctness is judged against the ORACLE, never against the program's self-report.**  An exact
   group is correct when its order-sensitive digest equals the oracle's.  A tolerance-declared group
   (``compare: bounded_int``) is correct when the device's own bound check reports no element over
   the bound the contract declares -- and the bound the program checked must BE that bound, or the
   check answered a different question.  The classification is compared with the oracle's integer;
   the program's ``agrees=`` is recorded and never consulted.

3. **A run with ANY correctness failure is ``MEASURED_INVALID``.**  Its cycles are recorded -- they
   say where the time went, which is exactly what someone fixing it needs -- but it has no
   ``objective_cycles`` and it can never be scored as a success.  A faster wrong program is the
   failure a performance loop is most likely to reward, so the objective field is ``None`` for it by
   construction rather than by a check a caller might forget.

NOTHING HERE KNOWS A TARGET OR A MODEL.  The line spellings arrive as the program's own templates
(:data:`PROTOCOL_TEMPLATES` restates them and a test holds the two together), the expected groups
and their comparison modes arrive in the build record's ``expectations``, and the machine that ran
the program is the caller's business.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA = "merlin_whole_model_verdict_v1"

#: ``timing_status`` values a verdict can carry.  ``MEASURED`` is the only one that feeds an
#: objective.  ``MEASURED_INVALID`` has cycles and is wrong; ``REFUSED`` has no admissible reading.
TIMING_MEASURED = "MEASURED"
TIMING_MEASURED_INVALID = "MEASURED_INVALID"
TIMING_REFUSED = "REFUSED"

COMPARE_EXACT = "exact"
COMPARE_BOUNDED = "bounded_int"

#: The whole-model program's protocol lines, keyed as the program keys them.  These are the defaults;
#: a build record's ``protocol`` block (the target driver's own spellings) overrides them key by key,
#: so a driver that spells a line differently is read by its own template, never by a guess.
PROTOCOL_TEMPLATES: dict[str, str] = {
    "invocations": "MERLIN_INVOCATIONS warmup=1 measured=1",
    "group": "GM_GROUP {group} {kind} {cycles} sum={sum} fnv1a={checksum}",
    "argmax": "GM_ARGMAX got={got} want={want} agrees={agrees}",
    "bounded": "GM_BOUND {group} max_abs={max_abs} over={over} bound={bound}",
    "split_line": "GM_SPLIT {group} gather={gather} kernel={kernel}",
    "full_model": "FM full model cycles: {cycles}",
    "bracket_sum": "FM bracketed compute only sum: {cycles}",
    "uncounted": "FM uncounted delta: {cycles}",
    "window_end": "MERLIN_WINDOW end label={label}",
    # The last field carries four space-separated indices; the matcher's final field is greedy.
    "where": "GM_WHERE {group} cols={cols} first={first}",
    # LOCAL GRADING: an exact group recomputed on the core, after the window, from the buffers the device
    # actually held -- so a group is judged on its own inputs, never on an upstream group's rounding.
    "local": "GM_LOCAL {group} mismatches={mismatches} of={elements} first={first}",
    # The worst element of a tolerance group, with the two operands the device actually had.
    "witness": "GM_WITNESS {group} i={index} lhs={lhs} rhs={rhs} device={device} reference={reference}",
    "words": "GM_WORDS {group} bytes={bytes} digest={digest}",
    # THE END RESULT OF A MODEL WHOSE OUTPUT IS A TENSOR (an action, not a class): how many of its
    # elements lie within the capsule's numeric policy of the oracle's, checked on the core after the
    # window. A classifier prints GM_ARGMAX instead; a model states which one it is graded on.
    "output": "GM_OUTPUT within={within} of={of}",
    # The whole output tensor's bytes, digested: the end result a reference arm is compared on.
    "output_digest": "GM_OUTPUT_DIGEST bytes={bytes} digest={digest}",
}


#: How a build that computes no byte-wise digest spells the field.
UNKNOWN_DIGEST = "UNKNOWN"


class VerdictRefusal(ValueError):
    """The log does not support reading a whole-model result at all."""


# --------------------------------------------------------------- structural line matching
def _template_tokens(template: str) -> tuple[str, ...]:
    tokens = tuple(template.split())
    if not tokens:
        raise VerdictRefusal("a protocol template must not be empty")
    return tokens


def match_line(template: str, line: str, *, greedy_last: bool = False) -> dict[str, str] | None:
    """Match one log line against one protocol template, token by token.

    Structural, not a pattern: the template and the line are both split on whitespace, literal
    tokens must be equal, and a token carrying ``{name}`` captures whatever sits between its literal
    prefix and suffix.  A line with a different number of tokens is not this line.
    """
    wanted = _template_tokens(template)
    got = line.split()
    if greedy_last and len(got) > len(wanted):
        # A GREEDY FINAL FIELD: the template's last token captures the rest of the line, so a field
        # holding a space-separated list (``first=a b c d``) still matches structurally.
        got = [*got[: len(wanted) - 1], " ".join(got[len(wanted) - 1 :])]
    if len(got) != len(wanted):
        return None
    captured: dict[str, str] = {}
    for want, have in zip(wanted, got, strict=True):
        if "{" not in want:
            if want != have:
                return None
            continue
        prefix, _, rest = want.partition("{")
        name, _, suffix = rest.partition("}")
        if not name:
            raise VerdictRefusal(f"template token {want!r} names no field")
        if not have.startswith(prefix) or not have.endswith(suffix) or len(have) <= len(prefix) + len(suffix):
            return None
        captured[name] = have[len(prefix) : len(have) - len(suffix)] if suffix else have[len(prefix) :]
    return captured


def _int(value: str, what: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise VerdictRefusal(f"{what} is not an integer: {value!r}") from exc


# --------------------------------------------------------------- the parsed log
@dataclass(frozen=True)
class GroupLine:
    group: str
    kind: str
    cycles: int
    #: None when the build printed no byte-wise digest (``UNKNOWN``: a timing or a dump build).
    sum: int | None
    fnv1a: int | None


@dataclass(frozen=True)
class ParsedLog:
    """Every protocol line the program printed, each at most once."""

    groups: Mapping[str, GroupLine]
    splits: Mapping[str, tuple[int, int]]
    bounds: Mapping[str, tuple[int, int, int]]
    where: Mapping[str, tuple[int, tuple[int, ...]]]
    local: Mapping[str, tuple[int, int]]
    witness: Mapping[str, tuple[int, int, int, int, int]]
    words: Mapping[str, tuple[int, int]]
    argmax: tuple[int, int, int] | None
    whole_window_cycles: int | None
    bracketed_sum_cycles: int | None
    uncounted_cycles: int | None
    invocations_seen: bool
    window_closed: bool
    #: ``(within, of)`` from GM_OUTPUT, for a model graded on a tensor output.
    output: tuple[int, int] | None = None
    #: ``(bytes, digest)`` of the whole output tensor (GM_OUTPUT_DIGEST).
    output_digest: tuple[int, int] | None = None


def parse_log(text: str, templates: Mapping[str, str] | None = None) -> ParsedLog:
    """Collect the protocol lines.  A key printed twice with different values is refused.

    Lines that are not protocol lines -- simulator diagnostics, boot chatter -- are ignored; that
    is safe only because rule 1 (every expected group present) runs before anything is interpreted.
    """
    spell = dict(PROTOCOL_TEMPLATES)
    if templates:
        spell.update({key: str(value) for key, value in templates.items() if key in PROTOCOL_TEMPLATES})
    groups: dict[str, GroupLine] = {}
    splits: dict[str, tuple[int, int]] = {}
    bounds: dict[str, tuple[int, int, int]] = {}
    where: dict[str, tuple[int, tuple[int, ...]]] = {}
    local: dict[str, tuple[int, int]] = {}
    witness: dict[str, tuple[int, int, int, int, int]] = {}
    words: dict[str, tuple[int, int]] = {}
    scalars: dict[str, int] = {}
    argmax: tuple[int, int, int] | None = None
    output: tuple[int, int] | None = None
    output_digest: tuple[int, int] | None = None
    invocations = False
    closed = False

    def once(store: dict, key: str, value: Any, what: str) -> None:
        if key in store and store[key] != value:
            raise VerdictRefusal(f"{what} {key!r} was printed twice with different values")
        store[key] = value

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == spell["invocations"].strip():
            invocations = True
            continue
        fields = match_line(spell["group"], line)
        if fields is not None:
            group = fields["group"]
            row = GroupLine(
                group=group,
                kind=fields["kind"],
                cycles=_int(fields["cycles"], f"group {group} cycles"),
                sum=None if fields["sum"] == UNKNOWN_DIGEST else _int(fields["sum"], f"group {group} sum"),
                fnv1a=None
                if fields["checksum"] == UNKNOWN_DIGEST
                else _int(fields["checksum"], f"group {group} fnv1a"),
            )
            if row.cycles < 0:
                raise VerdictRefusal(f"group {group} reports negative cycles")
            once(groups, group, row, "group")
            continue
        fields = match_line(spell["split_line"], line)
        if fields is not None:
            once(
                splits,
                fields["group"],
                (_int(fields["gather"], "gather"), _int(fields["kernel"], "kernel")),
                "split",
            )
            continue
        fields = match_line(spell["words"], line)
        if fields is not None:
            once(words, fields["group"], (_int(fields["bytes"], "bytes"), _int(fields["digest"], "digest")), "words")
            continue
        fields = match_line(spell["witness"], line)
        if fields is not None:
            once(
                witness,
                fields["group"],
                tuple(_int(fields[k], k) for k in ("index", "lhs", "rhs", "device", "reference")),
                "witness",
            )
            continue
        fields = match_line(spell["local"], line)
        if fields is not None:
            once(
                local,
                fields["group"],
                (_int(fields["mismatches"], "mismatches"), _int(fields["elements"], "of")),
                "local",
            )
            continue
        fields = match_line(spell["where"], line, greedy_last=True)
        if fields is not None:
            indices = tuple(_int(token, "offending index") for token in fields["first"].split())
            once(where, fields["group"], (_int(fields["cols"], "cols"), indices), "where")
            continue
        fields = match_line(spell["bounded"], line)
        if fields is not None:
            once(
                bounds,
                fields["group"],
                (
                    _int(fields["max_abs"], "max_abs"),
                    _int(fields["over"], "over"),
                    _int(fields["bound"], "bound"),
                ),
                "bound",
            )
            continue
        fields = match_line(spell["argmax"], line)
        if fields is not None:
            value = (
                _int(fields["got"], "argmax got"),
                _int(fields["want"], "argmax want"),
                _int(fields["agrees"], "agrees"),
            )
            if argmax is not None and argmax != value:
                raise VerdictRefusal("the classification was printed twice with different values")
            argmax = value
            continue
        fields = match_line(spell["output"], line) if "output" in spell else None
        if fields is not None:
            value = (_int(fields["within"], "output within"), _int(fields["of"], "output of"))
            if output is not None and output != value:
                raise VerdictRefusal("the end result was printed twice with different values")
            output = value
            continue
        fields = match_line(spell["output_digest"], line) if "output_digest" in spell else None
        if fields is not None:
            value = (_int(fields["bytes"], "output bytes"), _int(fields["digest"], "output digest"))
            if output_digest is not None and output_digest != value:
                raise VerdictRefusal("the output digest was printed twice with different values")
            output_digest = value
            continue
        for key in ("full_model", "bracket_sum", "uncounted"):
            fields = match_line(spell[key], line)
            if fields is not None:
                once(scalars, key, _int(fields["cycles"], key), "scalar")
                break
        else:
            if match_line(spell["window_end"], line) is not None:
                closed = True
    return ParsedLog(
        groups=groups,
        splits=splits,
        bounds=bounds,
        where=where,
        local=local,
        witness=witness,
        words=words,
        argmax=argmax,
        whole_window_cycles=scalars.get("full_model"),
        bracketed_sum_cycles=scalars.get("bracket_sum"),
        uncounted_cycles=scalars.get("uncounted"),
        invocations_seen=invocations,
        window_closed=closed,
        output=output,
        output_digest=output_digest,
    )


# --------------------------------------------------------------- expectations
@dataclass(frozen=True)
class GroupExpectation:
    group: str
    compare: str
    sum: int | None = None
    fnv1a: int | None = None
    bound_lsb: int | None = None
    #: The groups whose outputs this group reads.  Needed because an exact group's digest can only be
    #: compared with a basis whose INPUTS it shares (see :func:`judge`); empty means graph inputs only.
    inputs_from: tuple[str, ...] = ()
    #: Bytes per element of the group's committed output, when the build states it.  Read only by a
    #: machine's declared expressibility limits (see :func:`apply_machine_limits`).
    output_element_bytes: int | None = None
    #: A tolerance group's operand scales ``{lhs_load, rhs_load, readout, relu}`` as the program applies
    #: them, when the build states them: what lets a wrong element be read as "one operand alone".
    operands: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Expectations:
    """What an independent oracle says the program must print, from the build record."""

    groups: Mapping[str, GroupExpectation]
    #: The oracle's class, for a classifier; ``None`` for a model graded on its output tensor.
    argmax: int | None
    source: str = ""
    #: The output tensor's element count, for a model graded on it (``GM_OUTPUT``); ``None`` otherwise.
    output_elements: int | None = None

    @classmethod
    def from_record(cls, document: Mapping[str, Any]) -> Expectations:
        """Read ``expectations`` off a build record, refusing anything underdetermined.

        An exact group without both digests, or a bounded group without a bound, is refused here --
        before a run is spent on it -- because either would make its correctness a check that
        cannot fail.
        """
        if not isinstance(document, Mapping):
            raise VerdictRefusal("expectations must be a mapping")
        from .whole_model_partial import PartialBuildRefused, refuse

        try:  # a PARTIAL (only_groups) build prints every group line and is still not the whole model
            refuse(document, reader="a whole-model verdict")
        except PartialBuildRefused as exc:
            raise VerdictRefusal(str(exc)) from None
        raw_groups = document.get("groups")
        if not isinstance(raw_groups, Mapping) or not raw_groups:
            raise VerdictRefusal("expectations name no groups; a verdict with nothing to check is not one")
        declared = document.get("group_count")
        if declared is not None and int(declared) != len(raw_groups):
            raise VerdictRefusal(f"expectations declare group_count={declared} but carry {len(raw_groups)} groups")
        groups: dict[str, GroupExpectation] = {}
        for key, body in raw_groups.items():
            if not isinstance(body, Mapping):
                raise VerdictRefusal(f"expectation for group {key!r} is not a mapping")
            compare = str(body.get("compare") or COMPARE_EXACT)
            inputs = tuple(str(value) for value in (body.get("inputs_from") or ()))
            width = int(body["output_element_bytes"]) if body.get("output_element_bytes") is not None else None
            if any(value not in raw_groups for value in inputs):
                raise VerdictRefusal(f"group {key} reads from a group the expectations do not name")
            if compare == COMPARE_EXACT:
                if body.get("sum") is None or body.get("fnv1a") is None:
                    raise VerdictRefusal(f"exact group {key} needs both an oracle sum and fnv1a")
                groups[str(key)] = GroupExpectation(
                    group=str(key),
                    compare=compare,
                    sum=int(body["sum"]),
                    fnv1a=int(body["fnv1a"]),
                    inputs_from=inputs,
                    output_element_bytes=width,
                )
            elif compare == COMPARE_BOUNDED:
                if body.get("bound_lsb") is None:
                    raise VerdictRefusal(f"bounded group {key} declares no bound_lsb")
                groups[str(key)] = GroupExpectation(
                    group=str(key),
                    compare=compare,
                    sum=int(body["sum"]) if body.get("sum") is not None else None,
                    fnv1a=int(body["fnv1a"]) if body.get("fnv1a") is not None else None,
                    bound_lsb=int(body["bound_lsb"]),
                    inputs_from=inputs,
                    output_element_bytes=width,
                    operands=dict(body["operands"]) if isinstance(body.get("operands"), Mapping) else None,
                )
            else:
                raise VerdictRefusal(f"group {key} declares an unknown comparison {compare!r}")
        output = document.get("output")
        if document.get("argmax") is None and not (isinstance(output, Mapping) and output.get("elements")):
            raise VerdictRefusal("expectations carry no oracle classification and no output tensor to grade")
        return cls(
            groups=groups,
            argmax=int(document["argmax"]) if document.get("argmax") is not None else None,
            source=str(document.get("source") or ""),
            output_elements=int(output["elements"]) if isinstance(output, Mapping) and output.get("elements") else None,
        )


def _round_away(value: float) -> int:
    """Round half away from zero -- the program's own reference rounding."""
    return int(value - 0.5) if value < 0 else int(value + 0.5)


def _saturation(operands: Mapping[str, Any]) -> tuple[int, int] | None:
    """The signed range the program saturates a tolerance group's element to, from what the build
    STATES: an explicit ``saturate: [low, high]``, else the signed range of ``output_element_bytes``.
    ``None`` when neither is stated -- a signature computed against an assumed width would name the
    wrong operand, so none is computed."""
    stated = operands.get("saturate")
    if isinstance(stated, (list, tuple)) and len(stated) == 2:
        try:
            low, high = int(stated[0]), int(stated[1])
        except (TypeError, ValueError):
            return None
        return (low, high) if low < high else None
    width = operands.get("output_element_bytes")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        return None
    half = 1 << (8 * width - 1)
    return -half, half - 1


def _operand_signature(witness: tuple[int, ...] | None, operands: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What a tolerance group's worst element is: its operands' contributions, and which one it equals.

    A residual add whose second load landed before the first was cleared, or never landed, leaves an
    element holding ONE operand's scaled value. That signature names an ordering fault directly, where a
    magnitude alone ("off by 60") names nothing.  ``None`` when the build states no operand scales.
    """
    if witness is None or not isinstance(operands, Mapping):
        return None
    try:
        index, lhs, rhs, device, reference = witness
        lhs_load, rhs_load, readout = (float(operands[k]) for k in ("lhs_load", "rhs_load", "readout"))
    except (KeyError, TypeError, ValueError):
        return None
    relu = bool(operands.get("relu"))
    bounds = _saturation(operands)
    if bounds is None:
        return None
    low, high = bounds

    def settle(value: float) -> int:
        out = _round_away(value)
        if relu and out < 0:
            out = 0
        return max(low, min(high, out))

    alone = {"lhs": settle(lhs * lhs_load * readout), "rhs": settle(rhs * rhs_load * readout)}
    equals = [name for name, value in alone.items() if value == device and value != reference]
    return {
        "index": index,
        "lhs": lhs,
        "rhs": rhs,
        "device": device,
        "reference": reference,
        "lhs_alone": alone["lhs"],
        "rhs_alone": alone["rhs"],
        "equals": equals[0] if len(equals) == 1 else ("both" if equals else "neither"),
    }


# --------------------------------------------------------------- the verdict
def _order(key: str) -> tuple[int, str]:
    try:
        return (int(key), key)
    except ValueError:
        return (1 << 30, key)


#: The two bases an exact group's digest can be compared against.
BASIS_ORACLE = "oracle"
BASIS_REFERENCE = "reference"
#: The program's own post-window recomputation of the group from the buffers it actually read.
BASIS_LOCAL = "local"

#: Per-group states.  ``unverifiable`` is neither a pass nor a failure and is COUNTED beside every
#: number: it means no basis shares this group's inputs, so its digest has nothing to be equal to.
GROUP_CORRECT, GROUP_FAILED, GROUP_UNVERIFIABLE = "correct", "failed", "unverifiable"


def _reference_digests(reference: Mapping[str, Any] | None) -> dict[str, tuple[int, int]]:
    """A reference run's per-group digests, only when that run is itself admissible as a basis."""
    if not isinstance(reference, Mapping):
        return {}
    rows = reference.get("groups")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["group"]): (int(row["sum"]), int(row["fnv1a"]))
        for row in rows
        if isinstance(row, Mapping) and row.get("sum") is not None and row.get("fnv1a") is not None
    }


def judge(
    text: str,
    expectations: Expectations,
    *,
    templates: Mapping[str, str] | None = None,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The verdict document for one finished run's log.

    Never raises for a bad log: a refusal is a RESULT with ``timing_status: REFUSED`` and the
    reason, because the caller is a background measurer whose silence would read as "not yet".

    WHAT AN EXACT GROUP IS COMPARED WITH.  A group's digest is a function of its inputs.  The oracle
    computes a TOLERANCE-declared group (a residual add rounded once) differently from a device that
    rounds each operand -- both inside the contract's bound -- so downstream of the first such group
    the oracle's digests are for inputs no device ever has, and an exact comparison against them
    fails every correct program.  Measured: the vendor library's own ResNet run matches the oracle
    exactly on groups 1-5, passes group 6's bound at max_abs 1, and then differs from the oracle on
    every exact group after it -- deterministically, identically on repeated runs.

    So each exact group is compared with a basis WHOSE INPUTS IT SHARES: the oracle when every group
    it reads produced the oracle's bytes, else ``reference`` (an admissible run of the reference arm on
    the same machine) when every group it reads produced the reference's bytes.  When neither holds
    the group is ``unverifiable`` -- counted and reported, never read as a pass.  A tolerance group is
    always judged locally, by the device's own bound check over the operands it actually had.
    """
    try:
        parsed = parse_log(text, templates)
    except VerdictRefusal as exc:
        return refused(str(exc))
    expected = set(expectations.groups)
    printed = set(parsed.groups)
    # RULE 1, BEFORE ANY NUMBER IS READ.
    if printed != expected:
        missing = sorted(expected - printed, key=_order)
        extra = sorted(printed - expected, key=_order)
        return refused(
            f"the log printed {len(printed)} of the {len(expected)} expected group lines "
            f"(missing {missing[:12]}{'...' if len(missing) > 12 else ''}, unexpected {extra[:12]}); "
            "a log that did not finish saying what it was built to say is not read",
            groups_printed=len(printed),
            groups_expected=len(expected),
        )
    if parsed.whole_window_cycles is None:
        return refused("the log has every group line but no whole-window cycle line")
    if expectations.argmax is not None and parsed.argmax is None:
        return refused("the log has every group line but no classification line")
    if expectations.argmax is None and parsed.output is None:
        return refused("the log has every group line but no end-result (GM_OUTPUT) line")
    if not parsed.window_closed:
        return refused("the measured window never printed its closing line; the log is truncated")

    reference_digests = _reference_digests(reference)
    # Which bases each group's OUTPUT bytes equal -- the fact a consumer's comparison depends on.
    equal_to: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    unverifiable: list[str] = []
    for key in sorted(expected, key=_order):
        want = expectations.groups[key]
        got = parsed.groups[key]
        split = parsed.splits.get(key)
        mine = (got.sum, got.fnv1a)
        equal_to[key] = set()
        if want.sum is not None and want.fnv1a is not None and mine == (want.sum, want.fnv1a):
            equal_to[key].add(BASIS_ORACLE)
        if reference_digests.get(key) == mine:
            equal_to[key].add(BASIS_REFERENCE)
        row: dict[str, Any] = {
            "group": key,
            "kind": got.kind,
            "cycles": got.cycles,
            "gather_cycles": split[0] if split else 0,
            "kernel_cycles": split[1] if split else got.cycles,
            "compare": want.compare,
            "sum": got.sum,
            "fnv1a": got.fnv1a,
            "inputs_from": list(want.inputs_from),
        }
        if want.compare == COMPARE_EXACT and key in parsed.local:
            # LOCAL, and it wins: the program recomputed this group from the inputs it actually had.
            mismatches, elements = parsed.local[key]
            row["basis"] = BASIS_LOCAL
            row["state"] = GROUP_CORRECT if mismatches == 0 and elements > 0 else GROUP_FAILED
            row["mismatches"], row["elements"] = mismatches, elements
            row["detail"] = f"{mismatches} of {elements} elements differ from a recomputation on its own inputs"
            row["chained_digest_matches_oracle"] = BASIS_ORACLE in equal_to[key]
        elif want.compare == COMPARE_EXACT:
            shared = {BASIS_ORACLE, BASIS_REFERENCE if reference_digests else BASIS_ORACLE}
            for producer in want.inputs_from:
                shared &= equal_to.get(producer, set())
            if BASIS_ORACLE in shared:
                basis = BASIS_ORACLE
            elif BASIS_REFERENCE in shared and key in reference_digests:
                basis = BASIS_REFERENCE
            else:
                basis = None
            row["basis"] = basis
            if basis is None:
                row["state"] = GROUP_UNVERIFIABLE
                row["detail"] = (
                    "no basis shares this group's inputs (an upstream group legitimately produced "
                    "different bytes), so its digest has nothing to equal"
                )
            else:
                ok = basis in equal_to[key]
                row["state"] = GROUP_CORRECT if ok else GROUP_FAILED
                # The basis's values are NOT echoed: they are expected outputs, and this row reaches the
                # optimizing agent.  "Differs" is what it needs; the numbers would be an answer.
                row["detail"] = f"digest {'equals' if ok else 'differs from'} the {basis}'s on shared inputs"
        else:
            bound = parsed.bounds.get(key)
            row["basis"] = "device_bound_check"
            if bound is None:
                row["state"] = GROUP_FAILED
                row["detail"] = "tolerance-graded group printed no bound check, so it is unverified"
            else:
                max_abs, over, printed_bound = bound
                row["max_abs"], row["over"] = max_abs, over
                if printed_bound != want.bound_lsb:
                    row["state"] = GROUP_FAILED
                    row["detail"] = (
                        f"the program checked bound {printed_bound}, the contract declares "
                        f"{want.bound_lsb}; that check answered a different question"
                    )
                else:
                    ok = over == 0 and max_abs <= int(want.bound_lsb or 0)
                    row["state"] = GROUP_CORRECT if ok else GROUP_FAILED
                    row["detail"] = f"max_abs={max_abs} over={over} bound={printed_bound}"
                operands = dict(want.operands) if want.operands is not None else None
                if operands is not None and want.output_element_bytes is not None:
                    operands.setdefault("output_element_bytes", want.output_element_bytes)
                signature = _operand_signature(parsed.witness.get(key), operands)
                if signature is not None and row["state"] == GROUP_FAILED:
                    row["worst_element"] = signature
                    if signature["equals"] != "neither":
                        row["detail"] += (
                            f"; the worst element equals the {signature['equals']} operand's contribution "
                            "ALONE -- the other operand's contribution is missing from it"
                        )
                located = parsed.where.get(key)
                if located is not None:
                    cols, indices = located
                    row["columns"] = cols
                    row["offending_columns"] = sorted({i % cols for i in indices if i >= 0}) if cols > 0 else []
        row["correct"] = row["state"] == GROUP_CORRECT
        if row["state"] == GROUP_FAILED:
            failures.append(key)
        elif row["state"] == GROUP_UNVERIFIABLE:
            unverifiable.append(key)
        rows.append(row)

    if expectations.argmax is not None:
        observed, program_want, program_agrees = parsed.argmax
        argmax_ok = observed == expectations.argmax
    else:
        # A TENSOR OUTPUT: agreeing is every element within the capsule's policy of the oracle's. It is
        # carried under the classification's key so every rule that reads "the end result agrees"
        # reads it the same way; `observed` is the count within, `oracle` the count there is.
        within, of = parsed.output
        observed, program_want, program_agrees = within, of, int(within == of)
        argmax_ok = within == of == expectations.output_elements
    correct = not failures and argmax_ok
    total = parsed.whole_window_cycles
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "timing_status": TIMING_MEASURED if correct else TIMING_MEASURED_INVALID,
        "whole_window_cycles": total,
        "bracketed_sum_cycles": parsed.bracketed_sum_cycles,
        "uncounted_cycles": parsed.uncounted_cycles,
        # THE ONLY FIELD AN OBJECTIVE MAY READ, and it is None for a wrong run by construction.
        "objective_cycles": total if correct else None,
        "correctness": {
            "status": "pass" if correct else "fail",
            "groups_checked": len(rows),
            "groups_failed": failures,
            "groups_unverifiable": unverifiable,
            "groups_verified": len(rows) - len(failures) - len(unverifiable),
            "reference_basis_available": bool(reference_digests),
            "argmax": {
                "observed": observed,
                "oracle": expectations.argmax if expectations.argmax is not None else expectations.output_elements,
                "end_result": "classification" if expectations.argmax is not None else "output_within_policy",
                "agrees_with_oracle": argmax_ok,
                "program_self_report": {"want": program_want, "agrees": program_agrees},
                "self_report_consulted": False,
            },
            "evidence": "exact groups against the oracle, or against the same-machine reference run, on "
            "inputs they share; tolerance groups by the device's bound check against the contract's "
            "bound; classification against the oracle's own integer",
        },
        "groups": rows,
        "gather_cycles_total": sum(int(r["gather_cycles"]) for r in rows),
        "invocations_line_seen": parsed.invocations_seen,
        "expectations_source": expectations.source,
    }
    if not correct:
        document["invalid_reason"] = f"{len(failures)} group(s) failed ({failures[:12]})" + (
            "" if argmax_ok else f"; classified {observed}, the oracle says {expectations.argmax}"
        )
    return document


# --------------------------------------------------------------- failures the reference shares
GROUP_VENDOR_ALSO_FAILS = "vendor_also_fails"


def apply_vendor_also_fails(verdict: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """Excuse a failure the REFERENCE ARM shows identically on this same machine -- and nothing else.

    WHY THIS EXISTS, AND WHY IT CLAIMS NO CAUSE.  On the lean FireSim board the vendor library's own
    run fails three tolerance groups, in the same column span, on every run.  Whether that is a
    bitstream defect or a load-ordering race inside the library is NOT established -- the machine
    that "proved" a hardware cause never reorders loads, so it cannot tell the two apart -- and this
    rule does not need to know.  It excuses a group only when the reference failed the SAME group the
    SAME way on the SAME device: both tolerance groups over their bound, with every offending column
    this run reports inside the span of columns the reference's failure occupied.  The
    classification is excused only when the reference's classification is also wrong AND at least
    one group upstream of it was excused in this run.

    Every other failure stays a failure.  The number of excused groups travels with the result
    (``vendor_also_fails_count``), and the result says it is a SCREEN: the rule is only as strong as
    the reference run, so a machine without these failures must certify correctness.
    """
    document = copy.deepcopy(dict(verdict))
    if document.get("timing_status") not in (TIMING_MEASURED, TIMING_MEASURED_INVALID):
        return document
    ref_rows = {str(row["group"]): row for row in group_table(reference or {})}
    # A reference already excused against itself carries its failures as ``vendor_also_fails``: they
    # are still ITS failures, and they are the set a candidate may share.
    ref_failed = {key for key, row in ref_rows.items() if row.get("state") in (GROUP_FAILED, GROUP_VENDOR_ALSO_FAILS)}
    ref_argmax_ok = bool(((reference or {}).get("correctness") or {}).get("argmax", {}).get("agrees_with_oracle"))
    rows = {str(row["group"]): row for row in document.get("groups") or []}
    excused: set[str] = set()
    for key in sorted(rows, key=_order):
        row = rows[key]
        if row.get("state") != GROUP_FAILED or key not in ref_failed:
            continue
        theirs = ref_rows[key]
        if row.get("compare") != COMPARE_BOUNDED or theirs.get("compare") != COMPARE_BOUNDED:
            continue
        span = theirs.get("offending_columns") or []
        cols = row.get("offending_columns") or []
        if span and cols and min(span) <= min(cols) and max(cols) <= max(span):
            row["state"] = GROUP_VENDOR_ALSO_FAILS
            row["detail"] += "; the reference arm fails this group on this machine in the same column span"
            excused.add(key)
    failures = [key for key in sorted(rows, key=_order) if rows[key].get("state") == GROUP_FAILED]
    correctness = document["correctness"]
    argmax = correctness["argmax"]
    argmax_excused = bool(not argmax["agrees_with_oracle"] and excused and not ref_argmax_ok)
    argmax_ok = bool(argmax["agrees_with_oracle"] or argmax_excused)
    correct = not failures and argmax_ok
    correctness.update(
        status="pass" if correct else "fail",
        groups_failed=failures,
        groups_vendor_also_fails=sorted(excused, key=_order),
        argmax_vendor_also_fails=argmax_excused,
        excusal_rule=(
            "a group is excused only when the reference arm fails the same tolerance group on the same "
            "machine within the same column span; no hardware cause is claimed"
        ),
        certification="screen only: a machine without these reference failures must certify correctness",
    )
    document["timing_status"] = TIMING_MEASURED if correct else TIMING_MEASURED_INVALID
    document["objective_cycles"] = document["whole_window_cycles"] if correct else None
    document["vendor_also_fails_count"] = len(excused) + (1 if argmax_excused else 0)
    if correct:
        document.pop("invalid_reason", None)
    else:
        document["invalid_reason"] = (
            f"{len(failures)} group(s) failed that the reference arm does not ({failures[:12]})"
            + ("" if argmax_ok else "; the classification is wrong and the reference's is not excused for it")
        )
    for row in rows.values():
        row["correct"] = row.get("state") == GROUP_CORRECT
    return document


GROUP_MACHINE_CANNOT_EXPRESS = "machine_cannot_express"


def apply_machine_limits(
    verdict: Mapping[str, Any],
    expectations: Expectations,
    limits: Sequence[Mapping[str, Any]] | None,
    *,
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Name a failure the MACHINE cannot avoid, as its own class -- never as a silent excuse.

    A machine may declare what it cannot express (for example: no full-width accumulator readout,
    so a group committing 4-byte elements cannot be read back).  A failed group is reclassified
    ``machine_cannot_express`` only when (1) it matches a declared limit by its output element width
    and comparison, and (2) the reference arm fails the SAME group on the same machine.  A
    classification taken over such a group cannot be checked there either, and says so.  The count
    travels with the number; certification belongs to a machine without the limit.
    """
    document = copy.deepcopy(dict(verdict))
    if not limits or document.get("timing_status") not in (TIMING_MEASURED, TIMING_MEASURED_INVALID):
        return document
    ref_failed = {
        str(row["group"])
        for row in group_table(reference or {})
        if row.get("state") in (GROUP_FAILED, GROUP_VENDOR_ALSO_FAILS, GROUP_MACHINE_CANNOT_EXPRESS)
    }
    rows = {str(row["group"]): row for row in document.get("groups") or []}
    named: list[str] = []
    for key, row in rows.items():
        want = expectations.groups.get(key)
        if row.get("state") != GROUP_FAILED or want is None or key not in ref_failed:
            continue
        for limit in limits:
            if (
                want.output_element_bytes is not None
                and int(limit.get("output_element_bytes", -1)) == want.output_element_bytes
                and str(limit.get("compare") or want.compare) == want.compare
            ):
                row["state"] = GROUP_MACHINE_CANNOT_EXPRESS
                row["detail"] += f"; machine_cannot_express: {limit.get('reason')}"
                named.append(key)
                break
    correctness = document["correctness"]
    argmax = correctness["argmax"]
    ref_argmax_ok = bool(((reference or {}).get("correctness") or {}).get("argmax", {}).get("agrees_with_oracle"))
    argmax_named = bool(not argmax["agrees_with_oracle"] and named and not ref_argmax_ok)
    failures = sorted((k for k, r in rows.items() if r.get("state") == GROUP_FAILED), key=_order)
    argmax_ok = bool(argmax["agrees_with_oracle"] or argmax_named or correctness.get("argmax_vendor_also_fails"))
    correct = not failures and argmax_ok
    correctness.update(
        status="pass" if correct else "fail",
        groups_failed=failures,
        groups_machine_cannot_express=sorted(named, key=_order),
        argmax_machine_cannot_express=argmax_named,
    )
    for row in rows.values():
        row["correct"] = row.get("state") == GROUP_CORRECT
    document["machine_cannot_express_count"] = len(named) + (1 if argmax_named else 0)
    document["timing_status"] = TIMING_MEASURED if correct else TIMING_MEASURED_INVALID
    document["objective_cycles"] = document["whole_window_cycles"] if correct else None
    if correct:
        document.pop("invalid_reason", None)
        document["screen_only"] = "correctness here excludes what this machine cannot express; certify elsewhere"
    else:
        document["invalid_reason"] = f"{len(failures)} group(s) failed ({failures[:12]})" + (
            "" if argmax_ok else "; the classification is wrong"
        )
    return document


def judge_pair(
    timing_text: str,
    local_text: str,
    expectations: Expectations,
    *,
    templates: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One candidate, two builds: a TIMING run on the board and a LOCALLY graded run on a functional model.

    The two builds are the same program everywhere but their post-window verification (the caller
    proves that before running either). So the board's cycles are the candidate's cycles, and its
    correctness is: every group passes its LOCAL grade on the functional model, the classification
    there is the oracle's, AND for every group the board wrote the SAME BYTES as the functional model
    (one GM_WORDS digest per group from each). Timing cannot change what a computation writes, so a
    group whose bytes differ between the two is a hardware-timing fault (an operand-ordering race) or
    an RTL/ISA mismatch -- named by group. A tolerance group must also pass its own bound on the board.
    """
    local = judge(local_text, expectations, templates=templates)
    if local.get("timing_status") == TIMING_REFUSED:
        return refused(f"the locally graded run: {local.get('refusal')}")
    board = judge(timing_text, expectations, templates=templates)
    if board.get("timing_status") == TIMING_REFUSED:
        return refused(f"the timing run: {board.get('refusal')}")
    try:
        board_parsed = parse_log(timing_text, templates)
        board_words = board_parsed.words
        local_words = parse_log(local_text, templates).words
    except VerdictRefusal as exc:
        return refused(str(exc))
    expected = set(expectations.groups)
    for name, words in (("timing", board_words), ("local", local_words)):
        if set(words) != expected:
            missing = sorted(expected - set(words), key=_order)
            return refused(f"the {name} run printed no GM_WORDS line for group(s) {missing[:12]}")
    board_rows = {str(r["group"]): r for r in group_table(board)}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    unverifiable: list[str] = []
    for row in group_table(local):
        key = str(row["group"])
        mine = dict(row)
        timed = board_rows[key]
        mine["cycles"] = timed["cycles"]
        mine["gather_cycles"], mine["kernel_cycles"] = timed["gather_cycles"], timed["kernel_cycles"]
        mine["local_state"] = row.get("state")
        same = board_words[key] == local_words[key]
        mine["board_bytes_equal_functional_model"] = same
        if not same:
            mine["state"] = GROUP_FAILED
            want = expectations.groups.get(key)
            upstream = [g for g in (want.inputs_from if want else ()) if board_words[g] != local_words[g]]
            if upstream:
                # INHERITED, not originated: this group already read different bytes on the board, so
                # its own difference says nothing about its own code. Named as such, so the table
                # leads with the groups where the divergence STARTS.
                mine["inherited_from"] = upstream
                mine["detail"] = (
                    f"the board wrote different bytes than the functional model, but its inputs already "
                    f"differed (from {', '.join('g' + g for g in upstream)}): inherited, not originated here"
                )
            else:
                mine["detail"] = (
                    "the board wrote DIFFERENT BYTES than the functional model for the same program, from the "
                    "SAME inputs -- the divergence starts here: a hardware-timing fault (an operand-ordering "
                    "race) or an RTL/ISA mismatch in this group"
                )
        # A TIMING build carries no bound check (it is verification, and verification is the local
        # build's job): its bytes equal to the functional model's, which passed its own bound, is the
        # bound on the board. A bound the board DID print and failed still fails the group.
        if mine.get("compare") == COMPARE_BOUNDED and timed.get("state") == GROUP_FAILED and key in board_parsed.bounds:
            mine["state"] = GROUP_FAILED
            mine["detail"] = f"on the board: {timed.get('detail')}"
            if timed.get("worst_element"):
                mine["worst_element"] = timed["worst_element"]
        mine["correct"] = mine.get("state") == GROUP_CORRECT
        if mine.get("state") == GROUP_FAILED:
            failures.append(key)
        elif mine.get("state") == GROUP_UNVERIFIABLE:
            unverifiable.append(key)
        rows.append(mine)
    argmax = dict(local["correctness"]["argmax"])
    argmax["observed_on_board"] = board["correctness"]["argmax"]["observed"]
    argmax["source"] = "the functional model's run (the board's classification is recorded beside it)"
    correct = not failures and argmax["agrees_with_oracle"]
    total = board["whole_window_cycles"]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "timing_status": TIMING_MEASURED if correct else TIMING_MEASURED_INVALID,
        "whole_window_cycles": total,
        "bracketed_sum_cycles": board.get("bracketed_sum_cycles"),
        "uncounted_cycles": board.get("uncounted_cycles"),
        "objective_cycles": total if correct else None,
        "correctness": {
            "status": "pass" if correct else "fail",
            "groups_checked": len(rows),
            "groups_failed": failures,
            "groups_unverifiable": unverifiable,
            "groups_verified": len(rows) - len(failures) - len(unverifiable),
            "groups_board_differs_from_functional_model": sorted(
                (k for k in expected if board_words[k] != local_words[k]), key=_order
            ),
            "argmax": argmax,
            "evidence": "per-group LOCAL grade and classification on the functional model; per-group byte "
            "equality between the board and the functional model; tolerance bounds on the board",
        },
        "groups": rows,
        "gather_cycles_total": board.get("gather_cycles_total"),
        "expectations_source": expectations.source,
        "paired": {
            "timing": {"whole_window_cycles": total},
            "local": {"whole_window_cycles": local.get("whole_window_cycles")},
        },
    }
    if not correct:
        document["invalid_reason"] = f"{len(failures)} group(s) failed ({failures[:12]})" + (
            "" if argmax["agrees_with_oracle"] else "; the classification is wrong on the functional model"
        )
    return document


def refused(reason: str, **extra: Any) -> dict[str, Any]:
    """A verdict that reports nothing but why."""
    return {
        "schema": SCHEMA,
        "timing_status": TIMING_REFUSED,
        "refusal": reason,
        "objective_cycles": None,
        **extra,
    }


def objective_cycles(verdict: Mapping[str, Any] | None) -> int | None:
    """The cycles an objective may use, or None.  Reads the status, not only the field."""
    if not isinstance(verdict, Mapping) or verdict.get("timing_status") != TIMING_MEASURED:
        return None
    value = verdict.get("objective_cycles")
    return int(value) if isinstance(value, int) and value > 0 else None


def group_table(verdict: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = verdict.get("groups") if isinstance(verdict, Mapping) else None
    return rows if isinstance(rows, list) else []


__all__ = [
    "BASIS_ORACLE",
    "BASIS_REFERENCE",
    "GROUP_VENDOR_ALSO_FAILS",
    "GROUP_MACHINE_CANNOT_EXPRESS",
    "apply_machine_limits",
    "judge_pair",
    "GROUP_CORRECT",
    "GROUP_FAILED",
    "GROUP_UNVERIFIABLE",
    "apply_vendor_also_fails",
    "COMPARE_BOUNDED",
    "COMPARE_EXACT",
    "PROTOCOL_TEMPLATES",
    "SCHEMA",
    "TIMING_MEASURED",
    "TIMING_MEASURED_INVALID",
    "TIMING_REFUSED",
    "Expectations",
    "GroupExpectation",
    "ParsedLog",
    "VerdictRefusal",
    "group_table",
    "judge",
    "match_line",
    "objective_cycles",
    "parse_log",
    "refused",
]
