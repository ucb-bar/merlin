"""The optional lowering passes, by name: what each changes, whether it is exact, and its default.

Merlin's optional transforms grew three different switches -- a lowering feature name
(:mod:`.impr_features`), a ``MERLIN_*`` environment variable read where the pass is built, and an
entry of the integer datapath's pass set (:mod:`.quant_passes`) -- and nothing listed them together.
This registry is that list. It adds no transform of its own: each entry names the switch that already
exists, and selecting an entry flips exactly that switch, so an unselected build is byte-identical to
one made before this registry existed.

Every entry is TARGET-AGNOSTIC (it matches IR structure, never a model, shape or target name), which is
what makes it a pass an agent can enable by name, or copy into its own out-of-tree package and modify.

Selecting:

* ``merlin-compile --list-passes`` prints the table; ``--pass NAME`` / ``--no-pass NAME`` select;
* the whole-model builder takes ``lowering_passes: [NAME, -NAME, ...]`` in its build options;
* both set :data:`ENV` (``MERLIN_PASSES=int-softmax-table,-fusion-guard``), which the lowering reads
  wherever it decides -- including in child processes -- so one spelling reaches every path.

``exactness`` is a claim each entry's module and tests stand behind: ``exact`` is bit-identical output
on every input the original defines; ``numerics-changing`` is a different numerical model, graded
against its own reference.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

#: The process-wide selection: comma-separated names, each optionally prefixed ``-`` (deselect) or
#: ``+`` (select). Read at the point each pass is decided, never frozen at import.
ENV = "MERLIN_PASSES"

EXACT = "exact"
NUMERICS_CHANGING = "numerics-changing"

#: Where an entry acts. ``capture`` entries are chosen when a model is captured (a capture variant's
#: key), not at compile time; they are listed so the table is complete.
STAGES = ("capture", "quantization", "lowering")


@dataclass(frozen=True)
class OptionalPass:
    name: str
    summary: str
    changes: str
    exactness: str
    default: str
    stage: str
    #: The lowering feature it enables (:mod:`.impr_features`).
    feature: str | None = None
    #: The ``MERLIN_*`` switch it maps to, and whether the pass runs when the switch is unset.
    switch: str | None = None
    #: The integer datapath pass it names (:mod:`.quant_passes`).
    quant_pass: str | None = None
    #: The capture-variant key that selects it at capture time.
    capture_key: str | None = None
    #: True when the pass runs unless deselected (in the context its ``default`` names).
    on_by_default: bool = False

    def mechanism(self) -> str:
        if self.feature:
            return f"lowering feature {self.feature}"
        if self.switch:
            return f"switch {self.switch}"
        if self.quant_pass:
            return f"integer datapath pass {self.quant_pass}"
        return f"capture variant key {self.capture_key}"


_INTEGER_DATAPATH = "on in the integer (int8 compute) datapath"

REGISTRY: tuple[OptionalPass, ...] = (
    OptionalPass(
        name="int-softmax-table",
        summary="integer softmax: table-read numerator, int32 row sum, per-row P quantization",
        changes=(
            "the integer exp and floor division per element become a table read; the never-binding "
            "upper clamp is dropped and the lower one becomes a compare-and-select (NaN lands in the "
            "table); the row sum accumulates in i32 when it cannot overflow; the next contraction's "
            "per-row int8 quantization of P runs once per row on its candidate values; the attention "
            "scale moves before the reshape that hid it from fusion"
        ),
        exactness=EXACT,
        default="off",
        stage="lowering",
        feature="int_softmax_table",
    ),
    OptionalPass(
        name="exact-math-inline",
        summary="inline exactly specified math and bf16 rounding instead of per-element calls",
        changes=(
            "floor/ceil/trunc/round/roundeven/fabs become LLVM intrinsics returning libm's NaN (x + x), "
            "pow(2, int) is built from exponent bits, and every bf16 op is computed in f32 and rounded "
            "in integer arithmetic as the runtime's __truncsfbf2 does"
        ),
        exactness=EXACT,
        default="on for open-model host code, off elsewhere",
        stage="lowering",
        feature="lower_exact_math_inline",
    ),
    OptionalPass(
        name="roundeven-intrinsic",
        summary="math.roundeven as llvm.intr.roundeven after loop formation",
        changes="each math.roundeven becomes the LLVM intrinsic instead of a roundevenf call",
        exactness=EXACT,
        default="off",
        stage="lowering",
        feature="lower_roundeven_to_intrinsic",
    ),
    OptionalPass(
        name="fusion-guard",
        summary="refuse elementwise fusion that re-evaluates a per-row producer per element",
        changes="a broadcast producer is not inlined into a consumer with a larger iteration space",
        exactness=EXACT,
        default="on",
        stage="lowering",
        switch="MERLIN_FUSION_GUARD",
        on_by_default=True,
    ),
    OptionalPass(
        name="sink-deallocs",
        summary="free each buffer after its last use, with a use-after-free check",
        changes="deallocations move from the end of the function to after each buffer's last use",
        exactness=EXACT,
        default="off",
        stage="lowering",
        switch="MERLIN_SINK_DEALLOCS",
    ),
    OptionalPass(
        name="static-arena",
        summary="bind per-intermediate heap allocations to one statically planned arena",
        changes="the emitted malloc/free pairs become offsets into one arena",
        exactness=EXACT,
        default="off",
        stage="lowering",
        switch="MERLIN_STATIC_ARENA",
    ),
    OptionalPass(
        name="int-softmax",
        summary="softmax exp as the integer I-BERT exp",
        changes="math.exp in a softmax becomes a fixed-point polynomial and power-of-two shift",
        exactness=NUMERICS_CHANGING,
        default=_INTEGER_DATAPATH,
        stage="quantization",
        quant_pass="softmax_int",
        on_by_default=True,
    ),
    OptionalPass(
        name="int-gelu",
        summary="GELU erf as the integer I-BERT i-GELU",
        changes="math.erf in a GELU becomes a fixed-point polynomial on a per-row int grid",
        exactness=NUMERICS_CHANGING,
        default=_INTEGER_DATAPATH,
        stage="quantization",
        quant_pass="gelu_int",
        on_by_default=True,
    ),
    OptionalPass(
        name="int-silu",
        summary="SiLU sigmoid as the integer i-sigmoid",
        changes="the logistic in a SiLU becomes the integer exp and a division",
        exactness=NUMERICS_CHANGING,
        default=_INTEGER_DATAPATH,
        stage="quantization",
        quant_pass="silu_int",
        on_by_default=True,
    ),
    OptionalPass(
        name="int-rsqrt",
        summary="rsqrt as the integer fast reciprocal square root",
        changes="math.rsqrt becomes an integer estimate refined in fixed point",
        exactness=NUMERICS_CHANGING,
        default=_INTEGER_DATAPATH,
        stage="quantization",
        quant_pass="rsqrt_int",
        on_by_default=True,
    ),
    OptionalPass(
        name="integer-nonlinear",
        summary="capture softmax, GELU and layer norm as integer arithmetic (I-BERT)",
        changes="the captured program computes the three nonlinears in integer arithmetic",
        exactness=NUMERICS_CHANGING,
        default="off",
        stage="capture",
        capture_key="quant_integer_nonlinear",
    ),
)


class PassSelectionError(ValueError):
    """A selection that names an unknown pass, a pass twice, or a capture-time pass at compile time."""


def entries() -> tuple[OptionalPass, ...]:
    return REGISTRY


def get(name: str) -> OptionalPass:
    for entry in REGISTRY:
        if entry.name == name:
            return entry
    known = ", ".join(e.name for e in REGISTRY)
    raise PassSelectionError(f"unknown pass {name!r}; known: {known}")


@dataclass(frozen=True)
class Selection:
    """Which optional passes a build turns on and off, beyond each pass's default."""

    enable: frozenset[str] = frozenset()
    disable: frozenset[str] = frozenset()

    @classmethod
    def of(cls, enable: Iterable[str] = (), disable: Iterable[str] = ()) -> Selection:
        on, off = frozenset(enable), frozenset(disable)
        for name in on | off:
            entry = get(name)
            if entry.stage == "capture":
                raise PassSelectionError(
                    f"{name} acts when the model is captured; select it with the capture variant key "
                    f"{entry.capture_key!r}, not at compile time"
                )
        both = on & off
        if both:
            raise PassSelectionError(f"selected and deselected at once: {', '.join(sorted(both))}")
        return cls(on, off)

    @classmethod
    def parse(cls, items: Iterable[str] | str | None) -> Selection:
        """``"a,-b,+c"`` or ``["a", "-b"]``: a bare or ``+`` name selects, ``-`` deselects."""
        if items is None:
            return cls()
        if isinstance(items, str):
            items = items.split(",")
        on, off = [], []
        for raw in items:
            item = str(raw).strip()
            if not item:
                continue
            if item[0] == "-":
                off.append(item[1:].strip())
            else:
                on.append(item[1:].strip() if item[0] == "+" else item)
        return cls.of(on, off)

    def spell(self) -> str:
        return ",".join([*sorted(self.enable), *(f"-{n}" for n in sorted(self.disable))])

    def __bool__(self) -> bool:
        return bool(self.enable or self.disable)

    def merged(self, other: Selection) -> Selection:
        """``other`` applied on top of this selection (its choices win)."""
        return Selection.of((self.enable - other.disable) | other.enable, (self.disable - other.enable) | other.disable)

    def state(self, name: str) -> bool | None:
        """True / False when this selection decides ``name``; None when it leaves the default."""
        if name in self.enable:
            return True
        if name in self.disable:
            return False
        return None


def active(environ=None) -> Selection:
    """The selection :data:`ENV` names in ``environ`` (the process environment by default)."""
    return Selection.parse((os.environ if environ is None else environ).get(ENV))


@contextlib.contextmanager
def applied(selection: Selection) -> Iterator[Selection]:
    """Make ``selection`` (on top of any already active) the process's for the duration.

    Process-wide by construction, as the switches it replaces are: a lowering runs in child processes
    and reads the environment there too. Restored on exit, even on error.
    """
    previous = os.environ.get(ENV)
    combined = active().merged(selection)
    if combined:
        os.environ[ENV] = combined.spell()
    else:
        os.environ.pop(ENV, None)
    try:
        yield combined
    finally:
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous


def selected_features(features: Iterable[str] | None) -> frozenset[str] | None:
    """``features`` with the active selection's feature-bound passes added or removed.

    An empty selection returns ``features`` unchanged (byte-identical lowering)."""
    selection = active()
    if not selection:
        return features if features is None else frozenset(features)
    out = set(features or ())
    for entry in REGISTRY:
        if entry.feature is None:
            continue
        state = selection.state(entry.name)
        if state is True:
            out.add(entry.feature)
        elif state is False:
            out.discard(entry.feature)
    return frozenset(out)


def switched(name: str, fallback: bool) -> bool:
    """Whether the switch-bound pass ``name`` runs: the active selection's choice, else ``fallback``
    (the switch's own reading of its ``MERLIN_*`` variable)."""
    state = active().state(get(name).name)
    return fallback if state is None else state


def selected_quant_passes(passes: Iterable[str]) -> list[str]:
    """The integer datapath's pass set with the active selection's quantization passes applied."""
    selection = active()
    out = list(passes)
    for entry in REGISTRY:
        if entry.quant_pass is None:
            continue
        state = selection.state(entry.name)
        if state is True and entry.quant_pass not in out:
            out.append(entry.quant_pass)
        elif state is False and entry.quant_pass in out:
            out.remove(entry.quant_pass)
    return out


def table() -> str:
    """The registry as aligned text, one pass per line (``merlin-compile --list-passes``)."""
    rows = [("NAME", "STAGE", "EXACTNESS", "DEFAULT", "WHAT IT DOES")]
    rows += [(e.name, e.stage, e.exactness, e.default, e.summary) for e in REGISTRY]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(r[i].ljust(widths[i]) for i in range(4)) + "  " + r[4] for r in rows]
    lines.append("")
    lines.append(f"select with --pass NAME / --no-pass NAME, build option lowering_passes, or {ENV}")
    return "\n".join(lines)


def describe() -> list[dict[str, object]]:
    """The registry as data (``merlin-compile --list-passes --json``)."""
    return [
        {
            "name": e.name,
            "stage": e.stage,
            "exactness": e.exactness,
            "default": e.default,
            "summary": e.summary,
            "changes": e.changes,
            "selected_by": e.mechanism(),
        }
        for e in REGISTRY
    ]
