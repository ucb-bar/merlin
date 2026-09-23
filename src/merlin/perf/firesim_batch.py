"""Link N measured windows into ONE queue job, and admit each window on its own evidence.

WHY A BATCH IS A LINKING PROBLEM AND NOT A SESSION PROBLEM.  ``execution_policy`` derives the queue
lifecycle from the daemon's own trace: ``kill -> infrasetup -> runworkload -> kill``, exactly one
``runworkload`` per :data:`~merlin.perf.execution_policy.FIRESIM_QUEUE_OPERATION` job, and
:class:`~merlin.perf.execution_policy.FireSimQueuePreflight` refuses a submission with a nested
direct ``firesim`` command.  So "hold one session and replay N ELFs" is inadmissible by
construction -- it is what ``merlin/experiments/gemmini_perf_bench/scripts/firesim_bundle.sh``
does today, shelling ``firesim kill``/``infrasetup``/``runworkload`` in a loop, and no receipt can
ever be sealed from it.  The admissible shape is ONE bootbinary containing N measured windows: one
``runworkload-full``, N ``METRIC cycles`` lines, N per-window correctness blocks.

WHAT THIS MODULE REFUSES, AND WHY EACH REFUSAL IS THERE.

* **Members that disagree on the weight bytes.**  Comparing candidates measured against different
  weights answers no question anyone asked, and N copies of a whole model's weights do not fit in
  one bootbinary anyway.  One shared blob, verified by sha256, or the batch does not link.
* **A window that cleared its argmax marker but missed a declared group checksum.**  That window is
  ``fail``, not ``incomplete``: it RAN and it was WRONG.  This rule exists because of ONE measured
  event -- FireSim job 730 returned 23,787,829 cycles, cleared both declared exit criteria, and was
  wrong, with group-1's checksum reading 5,652,929 against an oracle 5,663,048, while the UART
  argmax marker printed IDENTICALLY to a correct run.  A policy that names only an argmax and a
  cosine cannot see that, which is why a v1 policy is refused for a cycle claim here.
* **A stray ``METRIC cycles`` line.**  An undeclared window ran; the plan cannot name what shared
  the FPGA with the windows that ARE named, so the whole batch is invalid.
* **A missing ``METRIC cycles`` line.**  ``incomplete``/``window_not_run``.  The capsule is NOT
  dropped from the denominator -- not run is not pass, and it is not a deletion either.
* **An order-effect control that diverged.**  The batch's last window repeats window 0.  A batched
  number that is not reproducible within the batch is not comparable to a solo one, so a divergence
  beyond the declared bound invalidates every window, including the ones that looked fine.
* **A size claim with no observed wall.**  A batch is refused on an OBSERVED per-window wall time,
  never an estimate; a member with no observation cannot be priced and the batch fails closed.

WHAT THIS MODULE DOES NOT DO: it never shells out.  :func:`submission_spec` returns argv for
``FireSimQueuePreflight`` and nothing else, for the reason ``execution_policy``'s own docstring
gives -- so no library caller can bypass the queue by importing a convenient runner.  Scheduling a
job is an operator action, not a library one.

THE UART FRAME IS OBSERVED.  It was proposed here before anything printed it; FireSim queue job 734
ran it on hardware -- 13 windows of GEMM measurement in one ``runworkload-full``, every window
admitted on its own digest, the order-effect control 26 ppm from window 0 -- and its log is pinned
at ``merlin/tests/data/firesim_queue/job734_gemm_batch_uart_skeleton.log``.  The per-window frame
markers (``MERLIN_BATCH``/``MERLIN_WINDOW``) are this module's; everything INSIDE a frame reuses
lines the harness already prints (``MERLIN_PROFILE``, ``METRIC cycles N``, and the workload's own
correctness lines, whose spelling this module reads from the policy rather than knowing).  The
receipt for such a run is :func:`~merlin.perf.firesim_receipt.parse_batched_firesim_receipt`, which
holds each frame to the SAME shape rule a solo receipt applies to the whole log.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execution_policy import (
    FIRESIM_QUEUE_OPERATION,
    FireSimQueuePreflight,
    require_batch_measurement,
)
from .firesim_receipt import FireSimReceiptError

#: The policy schema this module requires.  v1 carries only ``success_markers``; see the module
#: docstring for the measured run that shape admitted.
VALIDATION_POLICY_SCHEMA_V2 = "merlin_firesim_uart_validation_policy_v2"

#: Window framing.  A frame opens and closes with a ``key=value`` line naming the window's label,
#: which is how N windows in one UART stay attributable when one of them does not run at all.
BATCH_BEGIN = "MERLIN_BATCH begin"
BATCH_END = "MERLIN_BATCH end"
WINDOW_BEGIN = "MERLIN_WINDOW begin"
WINDOW_END = "MERLIN_WINDOW end"

#: Per-window admission verdicts.  Three, not two, and the distinction is the point: ``fail`` means
#: the window ran and was wrong, ``incomplete`` means it did not run (or did not finish), and the
#: two must never be collapsed -- a wrong result reported as an absence hides a defect, and an
#: absence reported as a wrong result invents one.
PASS = "pass"
FAIL = "fail"
INCOMPLETE = "incomplete"

WINDOW_NOT_RUN = "window_not_run"

#: HOW A WINDOW DEFINES THE THING IT MEASURED.  Emitters in this repository do not agree, and the
#: disagreement is invisible in the number: the layer bench frames ONE contiguous
#: ``t0 ... call + fence ... t1`` around the work, while the whole-model group program SUMS 71
#: per-call deltas and so excludes every gap between one group's fence and the next group's issue.
#: Both publish ``METRIC cycles N``.  A ratio of one over the other is not a speedup -- it is a
#: speedup plus whatever inter-call host sequencing one side priced and the other did not -- and a
#: headline ratio has already been taken across the two.  So the definition is DECLARED by the
#: window that measured it, carried into the receipt, and :func:`cycle_ratio` refuses a pair that
#: does not agree on it (or that does not state it at all).
WINDOW_KIND_CONTIGUOUS = "contiguous"
WINDOW_KIND_SUM_OF_CALLS = "sum_of_calls"
WINDOW_KINDS = (WINDOW_KIND_CONTIGUOUS, WINDOW_KIND_SUM_OF_CALLS)

#: What a window that declares no kind records.  Never a default kind: guessing "contiguous" for an
#: undeclared window is how the incommensurable ratio was taken in the first place.
WINDOW_KIND_UNKNOWN = "UNKNOWN"


class BatchError(FireSimReceiptError):
    """A batch cannot be linked, or its evidence cannot support the claim made of it."""


def _key_values(tokens: Sequence[str]) -> dict[str, str]:
    """``k=v`` tokens as a mapping, parsed structurally (``partition``, never a pattern).

    A REPEATED KEY RAISES, exactly as :func:`~merlin.perf.firesim_receipt._key_values` does on the
    solo-receipt path.  The two helpers read the same protocol and for a long time disagreed about
    this one case: the receipt refused a duplicated key as corruption while this one kept the FIRST
    value with ``setdefault`` and said nothing, so ``... label=a label=b`` framed a window under
    whichever spelling happened to come first.  There is no reading of a duplicated key that is
    evidence; picking one of the two is inventing which line the run printed.

    Malformed tokens (``===``, a trailing ``batch=``) are SKIPPED here rather than refused, which is
    the one place the two helpers still differ and deliberately so: the receipt's helper is handed
    tokens off an already-identified queue-protocol line, while this one is handed tokens off a
    protocol line found in a log that also carries boot noise -- and both of those spellings occur
    in the vendored logs under ``merlin/tests/data/firesim_queue/``.
    """
    values: dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not (separator and key and value):
            continue
        if key in values:
            raise BatchError(f"a protocol line repeats the key {key!r}; a duplicated key is corruption, not a choice")
        values[key] = value
    return values


# --------------------------------------------------------------- the validation policy (v2)
@dataclass(frozen=True)
class ChecksumLine:
    """How this workload SPELLS a per-group checksum, declared as data rather than known here.

    ``prefix`` is the line's leading token, ``group_token_offset`` which whitespace token after it
    names the group, and ``value_key`` which ``k=v`` token carries the value.  The gemmini group
    model prints ``GM_GROUP <group> <kind> <cycles> sum=<checksum>``; another target prints
    something else, and this module never learns either spelling.
    """

    prefix: str
    value_key: str
    group_token_offset: int = 1

    def __post_init__(self) -> None:
        if not self.prefix.strip() or not self.value_key.strip():
            raise BatchError("a checksum line must declare its prefix and its value key")
        if self.group_token_offset < 1:
            raise BatchError("the group token must follow the checksum line prefix")

    def observed(self, line: str) -> tuple[str, int] | None:
        """``(group, value)`` for a checksum line, or None when this is not one.

        An unparseable value is refused rather than skipped: a checksum line the workload printed
        and this policy cannot read is UNKNOWN, and UNKNOWN is not a match.
        """
        tokens = line.split()
        if not tokens or tokens[0] != self.prefix:
            return None
        if len(tokens) <= self.group_token_offset:
            raise BatchError(f"checksum line names no group: {line!r}")
        group = tokens[self.group_token_offset]
        raw = _key_values(tokens).get(self.value_key)
        if raw is None:
            raise BatchError(f"checksum line carries no {self.value_key!r} value: {line!r}")
        try:
            return group, int(raw)
        except ValueError as exc:
            raise BatchError(f"checksum value is not an integer: {line!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"prefix": self.prefix, "value_key": self.value_key, "group_token_offset": self.group_token_offset}


@dataclass(frozen=True)
class WindowPolicy:
    """One declared window: its label, its exact correctness markers, and its group checksums."""

    label: str
    markers: tuple[str, ...]
    checksums: tuple[tuple[str, int], ...]
    repeats: str | None = None
    #: How this window defines its measured span -- one of :data:`WINDOW_KINDS`, or None for a
    #: policy that predates the field.  None is recorded as :data:`WINDOW_KIND_UNKNOWN` and makes
    #: the window ineligible for a ratio; it is never silently read as "contiguous".
    window_kind: str | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise BatchError("every window must carry a label")
        if self.window_kind is not None and self.window_kind not in WINDOW_KINDS:
            raise BatchError(f"window_kind must be one of {list(WINDOW_KINDS)}, observed {self.window_kind!r}")
        reserved = ("MERLIN_PROFILE", "MERLIN_INVOCATIONS", "METRIC", "MERLIN_BATCH", "MERLIN_WINDOW")
        for marker in self.markers:
            if (
                not isinstance(marker, str)
                or not marker
                or marker != marker.strip()
                or "\n" in marker
                or "\r" in marker
                or "\0" in marker
            ):
                raise BatchError("window markers must be nonempty exact UART lines")
            if marker.startswith(reserved):
                raise BatchError("correctness markers cannot reuse the profile, frame, or metric protocol")
        if len(set(self.markers)) != len(self.markers):
            raise BatchError("window markers must be unique")
        groups = tuple(group for group, _value in self.checksums)
        if len(set(groups)) != len(groups):
            raise BatchError("a window may declare each group's checksum once")
        if tuple(sorted(self.checksums)) != self.checksums:
            raise BatchError("window checksums must be sorted for stable receipts")
        # A WINDOW WITH NO CHECKSUM IS THE JOB-730 SHAPE. Markers alone print identically for a
        # correct and an incorrect run; refusing this here is the whole reason for v2.
        if not self.checksums:
            raise BatchError(
                f"window {self.label!r} declares no per-group checksum; an argmax/cosine marker "
                "prints identically for a correct and an incorrect run (FireSim job 730)"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label, "markers": list(self.markers), "checksums": dict(self.checksums)}
        if self.repeats:
            out["repeats"] = self.repeats
        # Emitted only when declared, so a policy written before this field serialises byte for byte
        # as it did and the receipts already sealed against it keep their document digest.
        if self.window_kind:
            out["window_kind"] = self.window_kind
        return out


@dataclass(frozen=True)
class BatchValidationPolicy:
    """The workload-owned v2 policy: one entry per declared window, checksums included."""

    policy_id: str
    workload: str
    checksum_line: ChecksumLine
    per_window: tuple[WindowPolicy, ...]

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.workload.strip():
            raise BatchError("a validation policy must name its policy and workload")
        if not self.per_window:
            raise BatchError("a validation policy must declare at least one window")
        labels = tuple(window.label for window in self.per_window)
        if len(set(labels)) != len(labels):
            raise BatchError("window labels must be unique within a policy")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(window.label for window in self.per_window)

    def window(self, label: str) -> WindowPolicy:
        for window in self.per_window:
            if window.label == label:
                return window
        raise BatchError(f"the validation policy declares no window {label!r}")

    @classmethod
    def from_json(cls, value: object) -> BatchValidationPolicy:
        if not isinstance(value, Mapping):
            raise BatchError("validation policy must be a JSON object")
        schema = value.get("schema")
        if schema != VALIDATION_POLICY_SCHEMA_V2:
            # A CYCLE CLAIM NEEDS THE CHECKSUMS, so the older schema is refused BY NAME rather than
            # read leniently. v1 has no `per_window` and no checksums at all: accepting it here
            # would reinstate exactly the admission rule job 730 walked through.
            raise BatchError(
                f"a batched cycle claim requires a {VALIDATION_POLICY_SCHEMA_V2!r} policy; "
                f"observed {schema!r}, which declares no per-window group checksums"
            )
        expected = {"schema", "policy_id", "workload", "checksum_line", "per_window"}
        if set(value) != expected:
            raise BatchError("validation policy keys must be exactly " + repr(sorted(expected)))
        line = value.get("checksum_line")
        if not isinstance(line, Mapping):
            raise BatchError("validation policy checksum_line must be a JSON object")
        checksum_line = ChecksumLine(
            prefix=str(line.get("prefix") or ""),
            value_key=str(line.get("value_key") or ""),
            group_token_offset=int(line.get("group_token_offset", 1)),
        )
        rows = value.get("per_window")
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes))
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            raise BatchError("validation policy per_window must be a JSON array of objects")
        windows: list[WindowPolicy] = []
        for row in rows:
            checksums = row.get("checksums")
            if not isinstance(checksums, Mapping):
                raise BatchError("each window must declare a checksums object")
            parsed: list[tuple[str, int]] = []
            for group, raw in checksums.items():
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise BatchError(f"checksum for group {group!r} must be an integer, not {raw!r}")
                parsed.append((str(group), int(raw)))
            markers = row.get("markers") or []
            if (
                not isinstance(markers, Sequence)
                or isinstance(markers, (str, bytes))
                or any(not isinstance(marker, str) for marker in markers)
            ):
                raise BatchError("window markers must be a JSON array of strings")
            repeats = row.get("repeats")
            kind = row.get("window_kind")
            windows.append(
                WindowPolicy(
                    label=str(row.get("label") or ""),
                    markers=tuple(markers),
                    checksums=tuple(sorted(parsed)),
                    repeats=str(repeats) if repeats else None,
                    window_kind=str(kind) if kind else None,
                )
            )
        return cls(
            policy_id=str(value.get("policy_id") or ""),
            workload=str(value.get("workload") or ""),
            checksum_line=checksum_line,
            per_window=tuple(windows),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VALIDATION_POLICY_SCHEMA_V2,
            "policy_id": self.policy_id,
            "workload": self.workload,
            "checksum_line": self.checksum_line.to_dict(),
            "per_window": [window.to_dict() for window in self.per_window],
        }


def load_batch_validation_policy(path: str | Path) -> BatchValidationPolicy:
    """Read a v2 policy from disk.  A v1 document is refused, not upgraded."""
    artifact = Path(path)
    if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
        raise BatchError(f"validation policy must name an absolute plain file, observed {artifact}")
    try:
        document = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchError(f"validation policy is not readable JSON: {artifact}") from exc
    return BatchValidationPolicy.from_json(document)


# --------------------------------------------------------------- linking
@dataclass(frozen=True)
class BatchMember:
    """One candidate offered to a batch: its program bytes, its weights, and its observed wall.

    ``observed_window_seconds`` is an OBSERVATION, never an estimate: a member that has not been
    timed cannot be priced, and a batch priced on a guess is refused rather than run and truncated.
    """

    label: str
    program_sha256: str
    weights_sha256: str
    observed_window_seconds: float
    markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise BatchError("every batch member must carry a label")
        for role, digest in (("program", self.program_sha256), ("weights", self.weights_sha256)):
            if not isinstance(digest, str) or len(digest) != 64:
                raise BatchError(f"{role} digest for {self.label!r} must be a sha256 hex digest")
        if self.observed_window_seconds <= 0:
            raise BatchError(
                f"member {self.label!r} carries no OBSERVED window wall time; a batch is sized "
                "from measurement, never from an estimate"
            )


@dataclass(frozen=True)
class LinkedBatch:
    """One bootbinary's worth of declared windows, plus what it may cost and what it repeats."""

    batch_id: str
    weights_sha256: str
    labels: tuple[str, ...]
    control_label: str
    repeat_label: str
    observed_seconds: tuple[tuple[str, float], ...]
    estimated_wall_seconds: float
    queue_wall_limit_seconds: float
    order_effect_bound_ppm: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "weights_sha256": self.weights_sha256,
            "windows": list(self.labels),
            "control_label": self.control_label,
            "repeat_label": self.repeat_label,
            "observed_seconds": dict(self.observed_seconds),
            "estimated_wall_seconds": self.estimated_wall_seconds,
            "queue_wall_limit_seconds": self.queue_wall_limit_seconds,
            "order_effect_bound_ppm": self.order_effect_bound_ppm,
        }


def link_batch(
    members: Sequence[BatchMember],
    *,
    batch_id: str,
    queue_wall_limit_seconds: float,
    order_effect_bound_ppm: int = 10_000,
    descriptor: Mapping[str, Any] | None = None,
) -> LinkedBatch:
    """Plan ONE bootbinary containing every member as a measured window, plus the order control.

    The returned plan's window list ends with a REPEAT of window 0.  Without it a batched number is
    not comparable to a solo one: nothing else in the run distinguishes "this candidate is faster"
    from "this candidate ran third, with the array and the DRAM in a state the first one did not
    see".  The repeat is a declared window like any other and costs one more window's wall time.

    ``descriptor`` is the capsule/model descriptor the batch measures; when given it is run through
    :func:`~merlin.perf.execution_policy.require_batch_measurement`, so a per-op descriptor cannot
    reach the FPGA by way of a batch.  ``queue_owned`` is True here because the only consumer of
    this plan is :func:`submission_spec`, which produces a queue submission and nothing else.
    """
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise BatchError("a batch must name the round-group it accumulates into")
    if descriptor is not None:
        require_batch_measurement(descriptor, batch_id=batch_id, queue_owned=True)
    if not members:
        raise BatchError("a batch needs at least one member")
    labels = [member.label for member in members]
    if len(set(labels)) != len(labels):
        raise BatchError("batch member labels must be unique")
    # ONE SHARED WEIGHTS BLOB. Two members disagreeing on the weight bytes are not two candidates
    # for one question -- they are two questions -- and N copies of a whole model's weights do not
    # fit beside N programs in one bootbinary.
    digests = sorted({member.weights_sha256 for member in members})
    if len(digests) != 1:
        offenders = sorted(f"{member.label}={member.weights_sha256[:12]}" for member in members)
        raise BatchError(
            "batch members disagree on the weight blob; a batch links ONE shared weights blob and "
            f"members measured against different weights are not comparable: {offenders}"
        )
    control = labels[0]
    repeat_label = f"{control}__order_control"
    if repeat_label in set(labels):
        raise BatchError(f"the order-effect control label {repeat_label!r} collides with a member")
    observed = tuple((member.label, float(member.observed_window_seconds)) for member in members)
    control_seconds = dict(observed)[control]
    estimate = sum(seconds for _label, seconds in observed) + control_seconds
    if queue_wall_limit_seconds <= 0:
        raise BatchError("a batch must be bounded by a positive queue wall limit")
    if estimate > queue_wall_limit_seconds:
        raise BatchError(
            f"batch {batch_id!r} needs {estimate:.1f}s of observed window wall (including the "
            f"order-effect control) against a {queue_wall_limit_seconds:.1f}s queue limit; drop "
            "members rather than truncate the run"
        )
    if order_effect_bound_ppm <= 0:
        raise BatchError("the order-effect bound must be a positive ppm divergence")
    return LinkedBatch(
        batch_id=batch_id.strip(),
        weights_sha256=digests[0],
        labels=tuple(labels) + (repeat_label,),
        control_label=control,
        repeat_label=repeat_label,
        observed_seconds=observed,
        estimated_wall_seconds=estimate,
        queue_wall_limit_seconds=float(queue_wall_limit_seconds),
        order_effect_bound_ppm=int(order_effect_bound_ppm),
    )


# --------------------------------------------------------------- admission
@dataclass(frozen=True)
class WindowAdmission:
    """One window's verdict, with the evidence that produced it."""

    label: str
    status: str
    reason: str
    cycles: int | None = None
    #: ``(group, observed, expected)``.  ``observed`` is ``None`` -- not a magic integer -- when the
    #: window printed no checksum for that group at all.  It used to be ``-1``, which is a value a
    #: signed checksum can legitimately take, so "the group is missing" and "the group summed -1"
    #: were the same row; an absence has to be its own thing or a policy expecting -1 reads as
    #: satisfied by silence.
    checksum_mismatches: tuple[tuple[str, int | None, int], ...] = ()
    missing_markers: tuple[str, ...] = ()
    #: Groups the window PRINTED that its policy does not declare.  Nothing checks those values, so
    #: a cycle number from such a window prices work no oracle has an answer for.
    undeclared_groups: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.label,
            "status": self.status,
            "reason": self.reason,
            "cycles": self.cycles,
            "not_run_is_not_pass": True,
        }
        if self.checksum_mismatches:
            out["checksum_mismatches"] = [
                {"group": group, "observed": observed, "expected": expected, "absent": observed is None}
                for group, observed, expected in self.checksum_mismatches
            ]
        if self.missing_markers:
            out["missing_markers"] = list(self.missing_markers)
        if self.undeclared_groups:
            out["undeclared_groups"] = list(self.undeclared_groups)
        return out


def _frame(uart_text: str, label: str) -> list[str] | None:
    """The lines strictly between this window's begin and end markers, or None when unframed."""
    begin_at: int | None = None
    for number, line in enumerate(uart_text.splitlines()):
        tokens = line.split()
        if len(tokens) < 3:
            continue
        head = " ".join(tokens[:2])
        # The head is tested BEFORE the tokens are parsed as `k=v`: `_key_values` refuses a repeated
        # key, and this scan walks every line of a log that also carries boot noise, so parsing a
        # non-protocol line here would let unrelated console text refuse a window.
        if head not in (WINDOW_BEGIN, WINDOW_END):
            continue
        if _key_values(tokens[2:]).get("label") != label:
            continue
        if head == WINDOW_BEGIN:
            if begin_at is not None:
                raise BatchError(f"window {label!r} opens twice in one UART")
            begin_at = number
        elif head == WINDOW_END:
            if begin_at is None:
                raise BatchError(f"window {label!r} closes before it opens")
            return uart_text.splitlines()[begin_at + 1 : number]
    if begin_at is not None:
        raise BatchError(f"window {label!r} opens and never closes")
    return None


def _verify_batch_frame(uart_text: str, batch: LinkedBatch) -> None:
    """The UART must open and close THIS batch, and agree on how many windows it carried.

    A UART that belongs to another batch, or that declares a different window count from the plan,
    is not evidence about this plan -- and binding a cycle number to the wrong round-group is the
    same class of error as attributing a result to the wrong hardware revision.
    """
    opens: list[dict[str, str]] = []
    closes: list[dict[str, str]] = []
    for line in uart_text.splitlines():
        tokens = line.split()
        if len(tokens) < 3:
            continue
        head = " ".join(tokens[:2])
        if head == BATCH_BEGIN:
            opens.append(_key_values(tokens[2:]))
        elif head == BATCH_END:
            closes.append(_key_values(tokens[2:]))
    if len(opens) != 1 or len(closes) != 1:
        raise BatchError(
            f"the UART must open and close exactly one batch, observed {len(opens)} begin and {len(closes)} end markers"
        )
    for role, seen in (("begin", opens[0]), ("end", closes[0])):
        if seen.get("id") != batch.batch_id:
            raise BatchError(f"the UART's batch {role} names {seen.get('id')!r}, not {batch.batch_id!r}")
    declared = opens[0].get("windows")
    if declared != str(len(batch.labels)):
        raise BatchError(f"the UART declares {declared!r} windows and the plan {len(batch.labels)}")


def admit_window(uart_text: str, window: WindowPolicy, checksum_line: ChecksumLine) -> WindowAdmission:
    """Admit one window on BOTH its markers AND every declared per-group checksum.

    THE JOB-730 RULE.  A window that clears its argmax marker and misses one checksum is ``fail``,
    not ``incomplete``.  It ran, it printed a marker indistinguishable from a correct run's, and it
    was wrong; calling that an absence is how a wrong number gets cited.  A window whose
    ``METRIC cycles`` line never appeared is ``incomplete``/``window_not_run`` -- and its capsule
    stays in the denominator, because not run is not pass and it is not a deletion either.

    THE OBSERVED SET, NOT ONLY THE DECLARED ONE.  Admission compares the groups the window PRINTED
    against the groups its policy DECLARES, in both directions.  Checking only the declared ones
    leaves every other group the program published unread -- a program emitting 72 groups against a
    71-group policy passes with the 72nd unchecked, which is the job-730 under-constraint wearing a
    different shape.  A declared group the window did not print is recorded as ABSENT rather than
    as a sentinel value, because a checksum is signed and any integer chosen as "missing" is also a
    value some correct run can legitimately publish.
    """
    frame = _frame(uart_text, window.label)
    if frame is None:
        return WindowAdmission(window.label, INCOMPLETE, WINDOW_NOT_RUN)
    metrics: list[int] = []
    for line in frame:
        tokens = line.split()
        if not tokens or tokens[0] != "METRIC":
            continue
        if len(tokens) != 3 or tokens[1] != "cycles":
            raise BatchError(f"window {window.label!r} may publish only METRIC cycles N: {line!r}")
        try:
            metrics.append(int(tokens[2]))
        except ValueError as exc:
            raise BatchError(f"window {window.label!r} cycle metric is not an integer") from exc
    if len(metrics) > 1:
        raise BatchError(
            f"window {window.label!r} published {len(metrics)} cycle metrics; one window is one measurement"
        )
    missing = tuple(marker for marker in window.markers if [line for line in frame if line == marker] != [marker])
    observed_checksums: dict[str, int] = {}
    for line in frame:
        seen = checksum_line.observed(line)
        if seen is None:
            continue
        group, value = seen
        if group in observed_checksums and observed_checksums[group] != value:
            raise BatchError(f"window {window.label!r} printed two different checksums for group {group!r}")
        observed_checksums[group] = value
    mismatches = tuple(
        (group, observed_checksums.get(group), expected)
        for group, expected in window.checksums
        if group not in observed_checksums or observed_checksums[group] != expected
    )
    # WHAT THE WINDOW PRINTED, not only what the policy asked about. The loop above walks
    # `window.checksums`, so a group the program published and the policy never named was simply
    # never looked at: its value could be anything and the window still passed. That is the same
    # under-constraint the per-group checksums exist to remove, one level up -- an undeclared group
    # is an unchecked group, and a cycle number covering it prices work nothing verified.
    undeclared = tuple(sorted(set(observed_checksums) - {group for group, _value in window.checksums}))
    if not metrics:
        # NOT RUN, OR DID NOT FINISH. The frame exists but published no cycle count, so there is no
        # measurement to be right or wrong about. Record what the partial frame showed and keep the
        # capsule in the denominator.
        return WindowAdmission(
            window.label,
            INCOMPLETE,
            WINDOW_NOT_RUN,
            checksum_mismatches=mismatches,
            missing_markers=missing,
            undeclared_groups=undeclared,
        )
    if mismatches or missing or undeclared:
        detail = []
        if missing:
            # "not exactly once", not "absent": a marker printed twice is as unattributable as one
            # never printed, and reporting a duplicate as an absence would misdescribe the evidence.
            detail.append(f"{len(missing)} declared marker(s) not present exactly once")
        if mismatches:
            detail.append(
                ", ".join(
                    f"group {group} checksum {'ABSENT' if observed is None else observed} != {expected}"
                    for group, observed, expected in mismatches
                )
            )
        if undeclared:
            detail.append(
                f"{len(undeclared)} group(s) the policy does not declare were printed and therefore "
                f"never checked: {list(undeclared)}"
            )
        return WindowAdmission(
            window.label,
            FAIL,
            "the window RAN and was wrong: " + "; ".join(detail),
            cycles=metrics[0],
            checksum_mismatches=mismatches,
            missing_markers=missing,
            undeclared_groups=undeclared,
        )
    return WindowAdmission(
        window.label,
        PASS,
        f"every declared marker and all {len(window.checksums)} group checksum(s) matched, and the "
        "window printed no group the policy does not declare",
        cycles=metrics[0],
    )


@dataclass(frozen=True)
class BatchAdmission:
    """The batch's verdict and each window's, with the order-effect control's divergence."""

    batch_id: str
    status: str
    reason: str
    windows: tuple[WindowAdmission, ...]
    order_effect_ppm: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "reason": self.reason,
            "order_effect_ppm": self.order_effect_ppm,
            "windows": [window.to_dict() for window in self.windows],
            "not_run_is_not_pass": True,
        }


def admit_batch(uart_text: str, batch: LinkedBatch, policy: BatchValidationPolicy) -> BatchAdmission:
    """Admit every declared window, then decide whether the BATCH itself is usable.

    Two conditions invalidate the whole batch rather than one window:

    * a ``METRIC cycles`` line that no declared window's frame contains -- an undeclared window
      ran, so the plan cannot say what shared the run's FPGA, DRAM and cache state with the rest;
    * the order-effect control diverging from window 0 beyond the declared bound -- the batch is
      not internally reproducible, so none of its numbers is comparable to a solo one.

    The second yields ``incomplete``, not ``fail``: nothing about any candidate was disproved.  The
    instrument was.
    """
    declared = set(batch.labels)
    if declared != set(policy.labels):
        raise BatchError(
            f"the linked batch declares {sorted(declared)} and the policy "
            f"{sorted(policy.labels)}; a window admitted under a policy that does not name it is "
            "not admitted"
        )
    _verify_batch_frame(uart_text, batch)
    windows = tuple(admit_window(uart_text, policy.window(label), policy.checksum_line) for label in batch.labels)
    framed: list[str] = []
    for label in batch.labels:
        frame = _frame(uart_text, label)
        if frame is not None:
            framed.extend(frame)
    total_metrics = sum(1 for line in uart_text.splitlines() if line.split()[:1] == ["METRIC"])
    framed_metrics = sum(1 for line in framed if line.split()[:1] == ["METRIC"])
    if total_metrics != framed_metrics:
        return BatchAdmission(
            batch.batch_id,
            INCOMPLETE,
            f"{total_metrics - framed_metrics} METRIC line(s) lie outside every declared window "
            "frame; the ELF ran something nobody declared and the batch is invalid",
            windows,
        )
    control = next(w for w in windows if w.label == batch.control_label)
    repeat = next(w for w in windows if w.label == batch.repeat_label)
    if control.cycles is None or repeat.cycles is None or control.cycles <= 0:
        return BatchAdmission(
            batch.batch_id,
            INCOMPLETE,
            "the order-effect control did not produce a comparable pair, so a batched number "
            "cannot be shown equivalent to a solo one",
            windows,
        )
    divergence = abs(repeat.cycles - control.cycles) * 1_000_000 // control.cycles
    if divergence > batch.order_effect_bound_ppm:
        return BatchAdmission(
            batch.batch_id,
            INCOMPLETE,
            f"the order-effect control diverged {divergence} ppm from window 0 against a declared "
            f"bound of {batch.order_effect_bound_ppm} ppm; every window in this batch is "
            "position-contaminated",
            windows,
            order_effect_ppm=int(divergence),
        )
    return BatchAdmission(
        batch.batch_id,
        PASS,
        f"every declared window was framed and the order-effect control held within {divergence} ppm of window 0",
        windows,
        order_effect_ppm=int(divergence),
    )


# --------------------------------------------------------------- submission (argv only)
def submission_spec(
    batch: LinkedBatch,
    *,
    queue_executable: str | Path,
    workload: str,
    bootbinary: str | Path,
    validation_policy: str | Path,
) -> FireSimQueuePreflight:
    """The argv for ONE ``runworkload-full`` queue submission.  This function never runs anything.

    Returning a :class:`~merlin.perf.execution_policy.FireSimQueuePreflight` rather than a list is
    deliberate: that constructor is where a nested direct ``firesim`` command, shell control syntax
    and a non-absolute queue path are refused, so an argv that could not be submitted safely never
    leaves this module.
    """
    executable = str(queue_executable)
    if not workload.strip():
        raise BatchError("a queue submission must name its workload")
    argv = (
        executable,
        FIRESIM_QUEUE_OPERATION,
        "--workload",
        workload,
        "--stage-from",
        str(bootbinary),
        "--batch-id",
        batch.batch_id,
        "--windows",
        ",".join(batch.labels),
        "--weights-sha256",
        batch.weights_sha256,
        "--validation-policy",
        str(validation_policy),
    )
    return FireSimQueuePreflight(executable, argv)


# --------------------------------------------------------------- comparing two measured windows
@dataclass(frozen=True)
class MeasuredWindow:
    """One window's cycle count together with the DEFINITION of the span it counted.

    The cycle count alone is not enough to divide by: this repository's emitters disagree about
    what a measured window is (see :data:`WINDOW_KIND_CONTIGUOUS` / :data:`WINDOW_KIND_SUM_OF_CALLS`),
    and both spell the result ``METRIC cycles N``.  Carrying the kind beside the number is what lets
    a ratio be refused instead of computed.
    """

    label: str
    cycles: int
    window_kind: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise BatchError("a measured window must carry the label of the window it measured")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0:
            raise BatchError(f"window {self.label!r} carries no positive cycle count")
        if self.window_kind is not None and self.window_kind not in WINDOW_KINDS:
            raise BatchError(f"window_kind must be one of {list(WINDOW_KINDS)}, observed {self.window_kind!r}")

    @property
    def declared_kind(self) -> str:
        return self.window_kind or WINDOW_KIND_UNKNOWN


def cycle_ratio(numerator: MeasuredWindow, denominator: MeasuredWindow) -> float:
    """``numerator / denominator``, or a refusal when the two do not measure the same kind of span.

    A ratio between a contiguous wall window and a sum of per-call deltas is not a speedup.  The
    contiguous side prices every gap between one call's fence and the next call's issue; the summed
    side prices none of them, because it adds up only the intervals inside the calls.  Dividing one
    by the other silently credits (or charges) the difference to the compiler, and that division has
    already been published once, a single wall window over a sum of 71 per-call deltas.

    An UNDECLARED kind is refused on the same terms as a mismatched one.  Defaulting an undeclared
    window to "contiguous" would reproduce the original error exactly: the whole-model program never
    said what it measured, and the reader assumed.
    """
    for role, window in (("numerator", numerator), ("denominator", denominator)):
        if window.window_kind is None:
            raise BatchError(
                f"the {role} window {window.label!r} declares no window_kind, so what it measured is "
                f"{WINDOW_KIND_UNKNOWN}; a ratio against an undefined span is not a speedup"
            )
    if numerator.window_kind != denominator.window_kind:
        raise BatchError(
            f"window {numerator.label!r} measures a {numerator.window_kind!r} span and "
            f"{denominator.label!r} a {denominator.window_kind!r} one; these count different things "
            "and their ratio is not a speedup -- re-measure one side under the other's definition"
        )
    return numerator.cycles / denominator.cycles
