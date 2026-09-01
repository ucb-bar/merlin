"""SIMT geometry -- lane width, warp slots per core, cores -- read out of the ELABORATED design.

WHY THIS MODULE EXISTS
----------------------
A SIMT occupancy figure is a ratio whose denominator is the machine's width, and a denominator is only
as trustworthy as its provenance. The first version of :mod:`merlin.perf.simt_occupancy` took that
denominator from a **cycle model's TOML config**, justified by "the RTL cannot supply this". That was
wrong twice over: the fact bundles for these designs are empty because nobody pointed an extractor at
them, not because the elaboration lacks the geometry -- and a cycle model's config is a statement about
the *model*, which is exactly the thing an RTL-grounded figure is supposed to be checkable against.

A hardware fact here may come only from CIRCT (FIRRTL, the HW dialect, or an ``circt-arc`` model -- all
three ARE the RTL compiled) or from the target's own RTL repository. A separate simulator, a hand-written
model, or a Scala DEFAULT in a case class may CROSS-CHECK such a fact; none of them may be its source.
(The Scala default is worth naming: the generator's ``SIMTCoreParams`` declares ``numWarps = 4,
numLanes = 4``, and every elaborated design on this box is 8 and 16 -- the defaults are overridden by the
config that was actually elaborated. Reading the source would have produced two wrong numbers with a
straight face. What was ELABORATED is the authority; the source says only what the generator CAN emit.)

WHAT IS READ, AND WHY THOSE STRUCTURES
--------------------------------------
Three numbers, each from a structure the design must contain in order to *be* a SIMT machine, so nothing
here depends on one vendor's naming:

``lane_width``
    the BIT WIDTH of the per-warp lane-activity mask. A SIMT core must carry one bit per lane per
    resident warp -- that is what a thread mask *is* -- so the width of that state element is the lane
    count. Read as a width, never as a literal.

``warps_per_core``
    the NUMBER OF SLOTS in that mask table. One mask per warp slot: a FIRRTL aggregate's depth
    (``UInt<16>[8]``), or -- after CIRCT lowers aggregates -- the count of same-named indexed registers
    (``threadMasks_0`` ... ``threadMasks_7``), which must be a contiguous ``0..n-1`` or the table is
    reported undeterminable rather than guessed at.

``cores``
    the number of INSTANCE PATHS, from the elaborated top down, of the module(s) declaring that table.
    A core has exactly one warp scheduler, so instances of the mask-declaring module *are* the cores.
    Counted over the instance graph rather than over module definitions, because CIRCT deduplicates
    module bodies: the identical second core is one definition and two paths, and counting definitions
    would report a dual-core cluster as single-core. Conversely a PRE-dedup FIRRTL circuit uniquifies
    the module per instance (``WarpScheduler``, ``WarpScheduler_1``), so several modules declaring an
    identically shaped table are several cores and their paths are summed -- guarded by the shape
    check, which still refuses two tables that are genuinely different.

Which identifiers count as "the lane mask" is FACTORED OUT of the name, exactly as
:mod:`merlin.perf.hw_counters` factors engine tokens out of counter names: an identifier names a lane
mask when its tokens include a mask word AND a per-lane word. ``threadMasks``, ``lane_mask`` and
``activeLaneMask`` all resolve; ``tmask`` (one token) deliberately does not, because a too-generous
matcher is how a width from an unrelated register becomes a lane count.

FAIL CLOSED, ALWAYS
-------------------
Two mask tables of DIFFERENT shapes, a non-contiguous slot-index run, a mask with no slot dimension at
all, two candidate tops, a cyclic instance graph, no readable artifact: every one of these returns
UNKNOWN carrying its reason. There is no default anywhere in this file -- not 4, not 8, not 16 --
because a lane count nobody elaborated would price a run against a machine that was never built.

WHEN ELABORATIONS DISAGREE
--------------------------
:func:`geometry_for_target` reads EVERY artifact it can find, not the first that answers, and reports
the others beside the authoritative one under ``corroboration`` with any mismatch named under
``contested``. That is not defensiveness: on this host the cluster HW dialect carries two cores while
the single-core test configuration carries one, and stopping at the first full answer would have made
that difference invisible. They are two elaborated CONFIGURATIONS of one generator -- two machines --
and which one a set of cycles belongs to is a question the reader has to be able to see.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CORE", "LANE", "WARP", "ROLES",
    "Elaboration", "MaskTable", "SimtGeometry",
    "derive_from_lines", "derive_from_path", "derive_from_text",
    "geometry_for_target", "geometry_from_dict", "parse_firrtl", "parse_hw_dialect",
    "role_tokens", "sniff_dialect",
]

# --------------------------------------------------------------------------------------------
# The dimension vocabulary. ROLE words, not one design's spelling.
# --------------------------------------------------------------------------------------------
LANE = "lane"
WARP = "warp"
CORE = "core"
#: Reported in this order; ``LANE`` first because it is the divergence denominator.
ROLES: tuple[str, ...] = (LANE, WARP, CORE)

#: An identifier names a lane-activity mask when its tokens carry BOTH a mask word and a per-lane word.
#: Requiring both is the conservative half of the rule: ``mask`` alone matches byte-enables and
#: interrupt masks, and a width borrowed from one of those is a lane count that is simply wrong.
_MASK_WORDS = ("mask",)
_LANE_WORDS = ("thread", "lane")

#: FIRRTL's state keywords. A mask held in a `wire` is combinational, not the warp table, and a mask
#: held in a `mem` is a shape this reader does not claim to understand -- both are left unread rather
#: than approximated.
_FIR_STATE = ("reg", "regreset")
_FIR_MODULE = ("module", "extmodule", "intmodule")
_FIR_CIRCUIT = "circuit"

_HW_MODULE_OP = "hw.module"
_HW_INSTANCE_OP = "hw.instance"


# --------------------------------------------------------------------------------------------
# Structural helpers (no regex, per the repo's cardinal rule -- these are tokenizers)
# --------------------------------------------------------------------------------------------


def _tokens(identifier: str) -> tuple[str, ...]:
    """``threadMasks_0`` -> ``('thread', 'mask', '0')``.

    Split on underscores AND on camel-case humps, lower-cased, with a trailing plural ``s`` stripped, so
    one vocabulary serves ``threadMasks``, ``thread_mask`` and ``ACTIVE_LANE_MASK`` alike. Splitting
    only on underscores would leave ``threadMasks`` a single unmatched token, and a name this reader
    cannot tokenize is a fact it silently fails to find.
    """
    out: list[str] = []
    for chunk in identifier.split("_"):
        if not chunk:
            continue
        start = 0
        for i in range(1, len(chunk)):
            # A hump boundary is lower/digit -> upper. `MXFP` stays one token: consecutive capitals are
            # an acronym, not four words.
            if chunk[i].isupper() and not chunk[i - 1].isupper():
                out.append(chunk[start:i])
                start = i
        out.append(chunk[start:])
    words = []
    for w in out:
        w = w.lower()
        if len(w) > 1 and w.endswith("s") and not w.isdigit():
            w = w[:-1]
        words.append(w)
    return tuple(w for w in words if w)


#: Public alias. The role vocabulary is shared with the performance layer's model cross-check, which
#: must factor a config key's dimensions exactly the way an RTL identifier's are factored -- otherwise
#: the two sides of the comparison disagree about what they are comparing.
role_tokens = _tokens


def _is_lane_mask(identifier: str) -> bool:
    """Whether ``identifier`` names a per-lane activity mask."""
    toks = set(_tokens(identifier))
    return bool(toks & set(_MASK_WORDS)) and bool(toks & set(_LANE_WORDS))


def _split_index(identifier: str) -> tuple[str, int | None]:
    """``threadMasks_3`` -> ``('threadMasks', 3)``; ``threadMasks`` -> ``('threadMasks', None)``.

    CIRCT lowers a FIRRTL aggregate register into one scalar register per index, so the table's depth
    survives only as this suffix. Reading it back is what lets the HW dialect answer the same question
    the FIRRTL answers with ``[8]``.
    """
    base, sep, tail = identifier.rpartition("_")
    if sep and tail.isdigit():
        return base, int(tail)
    return identifier, None


def _last_top_level_colon(text: str) -> int:
    """Index of the last ``:`` sitting outside every bracket, or ``-1``.

    An MLIR result line carries its type after a trailing ``:``, but its attribute dictionary carries
    colons of its own (``{firrtl.random_init_start = 0 : ui64} : i16``). Taking the last colon at
    bracket depth zero is what separates the result type from an attribute's type; taking the last colon
    outright would read ``ui64`` as the register's width, i.e. a 64-lane machine.
    """
    depth = 0
    found = -1
    for i, ch in enumerate(text):
        if ch in "{[(<":
            depth += 1
        elif ch in "}])>":
            depth -= 1
        elif ch == ":" and depth == 0:
            found = i
    return found


def _hw_type_shape(typ: str) -> tuple[int, int] | None:
    """``(width, depth)`` for an MLIR HW type, or ``None`` when this reader cannot read it.

    ``i16`` -> ``(16, 1)`` (one lowered slot). ``!hw.array<8xi16>`` -> ``(16, 8)`` (the table still
    aggregated). Anything else -- a struct, a nested array, an aliased type -- returns ``None`` so the
    caller reports the shape UNKNOWN instead of inventing one.
    """
    t = typ.strip()
    if t.startswith("i") and t[1:].isdigit():
        return int(t[1:]), 1
    head, sep, rest = t.partition("<")
    if sep and head.strip() in ("!hw.array", "!hw.uarray") and rest.endswith(">"):
        inner = rest[:-1]
        depth_s, x, elem = inner.partition("x")
        if x and depth_s.isdigit():
            elem_shape = _hw_type_shape(elem)
            if elem_shape is not None and elem_shape[1] == 1:
                return elem_shape[0], int(depth_s)
    return None


def _fir_type_shape(typ: str) -> tuple[int, int] | None:
    """``(width, depth)`` for a FIRRTL ground/vector type, or ``None``.

    ``UInt<16>`` -> ``(16, 1)``; ``UInt<16>[8]`` -> ``(16, 8)``. A bundle or a doubly-nested vector
    returns ``None``: a mask table this reader cannot shape is UNKNOWN, never flattened into a product
    that would silently multiply lanes by warps.
    """
    t = typ.strip()
    head, sep, rest = t.partition("<")
    if not sep or head.strip() not in ("UInt", "SInt"):
        return None
    width_s, gt, tail = rest.partition(">")
    if not gt or not width_s.strip().isdigit():
        return None
    width = int(width_s.strip())
    tail = tail.strip()
    if not tail:
        return width, 1
    if tail.startswith("[") and tail.endswith("]"):
        dim = tail[1:-1].strip()
        if dim.isdigit():
            return width, int(dim)
    return None


# --------------------------------------------------------------------------------------------
# What one elaboration says
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskTable:
    """One per-warp lane-mask table, as one module declares it."""

    module: str
    #: The identifier with its lowered index suffix removed, so the FIRRTL and HW readings agree.
    base: str
    width: int
    depth: int
    #: The declaration line(s) this was read from, verbatim-ish, so a reader can check the derivation.
    evidence: str = ""


@dataclass
class Elaboration:
    """The elaborated design, reduced to what a SIMT denominator needs."""

    dialect: str = ""
    #: ``module -> {callee: how many times it is instantiated in that module}``.
    instances: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Declared top module, when the dialect states one outright (FIRRTL's ``circuit``).
    declared_top: str | None = None
    #: Modules the dialect marks public (HW dialect's non-``private`` modules).
    public: list[str] = field(default_factory=list)
    masks: list[MaskTable] = field(default_factory=list)
    #: Lane-mask state with NO slot dimension. Not the warp table (see :func:`_fold_masks`), but
    #: recorded rather than dropped: a silently discarded candidate is how a reader that found the
    #: wrong structure looks identical to one that found nothing.
    scalar_masks: list[MaskTable] = field(default_factory=list)
    #: Structures this reader saw but could not read. Carried, never dropped.
    problems: list[str] = field(default_factory=list)
    source: str | None = None

    def modules(self) -> set[str]:
        seen = set(self.instances)
        for callees in self.instances.values():
            seen.update(callees)
        return seen

    def roots(self) -> list[str]:
        """Modules nothing instantiates -- the candidate tops."""
        called = {c for callees in self.instances.values() for c in callees}
        return sorted(m for m in self.modules() if m not in called)

    def reaches(self, start: str, wanted: str) -> bool:
        seen: set[str] = set()
        stack = [start]
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            for c in self.instances.get(m, {}):
                if c == wanted:
                    return True
                stack.append(c)
        return False

    def multiplicity(self, top: str, wanted: str) -> int | None:
        """How many INSTANCE PATHS lead from ``top`` to ``wanted``; ``None`` on a cyclic graph.

        Accumulated over the module DAG rather than enumerated, so a wide SoC cannot blow up: the count
        for a module is memoized once and reused by every parent that instantiates it. A cycle is a
        graph this counting is undefined on, and it returns UNKNOWN rather than a number from a
        truncated walk.
        """
        memo: dict[str, int] = {}
        on_stack: set[str] = set()

        def count_in(mod: str) -> int | None:
            if mod in memo:
                return memo[mod]
            if mod in on_stack:
                return None
            on_stack.add(mod)
            total = 0
            for callee, n in self.instances.get(mod, {}).items():
                sub = count_in(callee)
                if sub is None:
                    on_stack.discard(mod)
                    return None
                total += n * ((1 if callee == wanted else 0) + sub)
            on_stack.discard(mod)
            memo[mod] = total
            return total

        return count_in(top)


@dataclass(frozen=True)
class SimtGeometry:
    """The SIMT dimensions one ELABORATION establishes, and which structure established each.

    Every dimension is tri-state. ``None`` is "this elaboration does not say", never a default: a lane
    width nobody elaborated is the one number an occupancy figure cannot borrow from elsewhere, and
    substituting a plausible sixteen would price a run against a machine that was never run.
    """

    lane_width: int | None = None
    warps_per_core: int | None = None
    cores: int | None = None
    #: ``role -> the structure actually read``, so a reader can check the derivation against the file.
    keys: dict[str, str] = field(default_factory=dict)
    #: Roles refused because the elaboration answers them more than one way.
    ambiguous: tuple[str, ...] = ()
    #: Reasons a role could not be read at all, keyed by role.
    unread: dict[str, str] = field(default_factory=dict)
    #: The module whose instances were counted as cores.
    core_module: str | None = None
    dialect: str = ""
    source: str | None = None

    def value(self, role: str) -> int | None:
        return {LANE: self.lane_width, WARP: self.warps_per_core, CORE: self.cores}.get(role)

    def resolved(self) -> tuple[str, ...]:
        return tuple(r for r in ROLES if self.value(r) is not None)

    @property
    def threads_per_core(self) -> int | None:
        """Lane slots one core can present per cycle. ``None`` unless BOTH factors were elaborated."""
        if self.lane_width is None or self.warps_per_core is None:
            return None
        return self.lane_width * self.warps_per_core

    @property
    def lane_slots_per_cycle(self) -> int | None:
        """Lane slots the whole cluster presents. ``None`` unless all three were elaborated."""
        per_core = self.threads_per_core
        if per_core is None or self.cores is None:
            return None
        return per_core * self.cores

    def to_dict(self) -> dict[str, Any]:
        return {"lane_width": self.lane_width, "warps_per_core": self.warps_per_core,
                "cores": self.cores, "threads_per_core": self.threads_per_core,
                "lane_slots_per_cycle": self.lane_slots_per_cycle,
                "keys": dict(sorted(self.keys.items())), "ambiguous": list(self.ambiguous),
                "unread": dict(sorted(self.unread.items())), "core_module": self.core_module,
                "dialect": self.dialect, "source": self.source,
                "resolved": list(self.resolved())}


# --------------------------------------------------------------------------------------------
# Readers -- one per CIRCT surface. Both are line-oriented and streamable, because an elaborated SoC
# FIRRTL on this box is 427 MB: slurping it to search it would cost more memory than the design.
# --------------------------------------------------------------------------------------------


def parse_hw_dialect(lines: Iterable[str], *, source: str | None = None) -> Elaboration:
    """Read a CIRCT **HW dialect** file (``*.hw.mlir``, the arc model's own input).

    Module declarations and instance references are read by :mod:`merlin.targetgen.rtl.extract_module`'s
    tokenizers rather than by a third hand-rolled scanner: that module already fails closed on an
    unreadable ``hw.module`` / ``hw.instance`` line, and its docstring records the four ways the retired
    regex forms silently dropped legal syntax.
    """
    from . import extract_module as EM

    el = Elaboration(dialect="hw", source=source)
    current: str | None = None
    seen_masks: dict[tuple[str, str], dict[int | None, tuple[int, int, str]]] = {}
    for raw in lines:
        try:
            decl = EM._parse_module_decl(raw)
        except EM.ExtractModuleError as e:
            el.problems.append(str(e))
            continue
        if decl is not None:
            current = decl[0]
            el.instances.setdefault(current, {})
            # `hw.module private @X` is internal; a module without `private` is part of the design's
            # public surface, which is the dialect's own statement of what the top candidates are.
            if "private" not in raw.split("@", 1)[0]:
                el.public.append(current)
            continue
        if current is None:
            continue
        if _HW_INSTANCE_OP in raw:
            try:
                refs = EM._instances_in_line(raw)
            except EM.ExtractModuleError as e:
                el.problems.append(str(e))
                refs = set()
            # A line naming more instances than distinct callees would under-count multiplicity, so say
            # so rather than quietly returning a core count that is too small.
            if raw.count(_HW_INSTANCE_OP) > len(refs) and refs:
                el.problems.append(f"{raw.count(_HW_INSTANCE_OP)} instances but {len(refs)} distinct "
                                   f"callee(s) on one line; multiplicity is UNKNOWN: {raw.strip()[:120]}")
            for r in refs:
                el.instances[current][r] = el.instances[current].get(r, 0) + 1
            continue
        stripped = raw.strip()
        if not stripped.startswith("%"):
            continue
        name, eq, rhs = stripped.partition("=")
        if not eq:
            continue
        ident = name.strip().lstrip("%").strip()
        if "," in ident or not _is_lane_mask(ident):
            continue
        ci = _last_top_level_colon(rhs)
        if ci == -1:
            el.problems.append(f"lane-mask value {ident!r} carries no readable result type")
            continue
        shape = _hw_type_shape(rhs[ci + 1:])
        if shape is None:
            el.problems.append(f"lane-mask value {ident!r} has an unreadable type "
                               f"{rhs[ci + 1:].strip()!r}")
            continue
        base, idx = _split_index(ident)
        seen_masks.setdefault((current, base), {})[idx] = (shape[0], shape[1], stripped[:160])
    tables, scalars = _fold_masks(seen_masks, el.problems)
    el.masks.extend(tables)
    el.scalar_masks.extend(scalars)
    return el


def parse_firrtl(lines: Iterable[str], *, source: str | None = None) -> Elaboration:
    """Read an elaborated **FIRRTL** circuit (``*.fir``).

    Module headers use :func:`merlin.targetgen.rtl.ports._module_name`, the existing structural reader
    for FIRRTL's three module keywords, so a differently-spelled module line is handled the same way
    everywhere in this package.
    """
    from . import ports as PORTS

    el = Elaboration(dialect="firrtl", source=source)
    current: str | None = None
    seen_masks: dict[tuple[str, str], dict[int | None, tuple[int, int, str]]] = {}
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if el.declared_top is None and parts[0] == _FIR_CIRCUIT and len(parts) >= 2:
            el.declared_top = parts[1].rstrip(":")
            continue
        name = PORTS._module_name(raw) if _is_firrtl_header(stripped) else ""
        if name:
            current = name
            el.instances.setdefault(current, {})
            continue
        if current is None:
            continue
        # `inst <name> of <Module>` -- FIRRTL's only instantiation form.
        if parts[0] == "inst" and len(parts) >= 4 and parts[2] == "of":
            callee = parts[3].rstrip(":")
            el.instances[current][callee] = el.instances[current].get(callee, 0) + 1
            continue
        if parts[0] not in _FIR_STATE or len(parts) < 2:
            continue
        ident = parts[1]
        if not _is_lane_mask(ident):
            continue
        _, colon, rest = stripped.partition(":")
        if not colon:
            el.problems.append(f"lane-mask register {ident!r} carries no readable type")
            continue
        # `reg x : UInt<16>[8], clock, reset, init @[...]` -- the type ends at the first comma that is
        # outside the width/vector brackets.
        depth, cut = 0, len(rest)
        for i, ch in enumerate(rest):
            if ch in "<[{":
                depth += 1
            elif ch in ">]}":
                depth -= 1
            elif ch == "," and depth == 0:
                cut = i
                break
        shape = _fir_type_shape(rest[:cut])
        if shape is None:
            el.problems.append(f"lane-mask register {ident!r} has an unreadable type "
                               f"{rest[:cut].strip()!r}")
            continue
        base, idx = _split_index(ident)
        seen_masks.setdefault((current, base), {})[idx] = (shape[0], shape[1], stripped[:160])
    tables, scalars = _fold_masks(seen_masks, el.problems)
    el.masks.extend(tables)
    el.scalar_masks.extend(scalars)
    return el


def _fold_masks(seen: Mapping[tuple[str, str], Mapping[int | None, tuple[int, int, str]]],
                problems: list[str]) -> tuple[list[MaskTable], list[MaskTable]]:
    """Fold the per-declaration readings into ``(warp tables, scalar masks)``, one per ``(module, base)``.

    Two shapes count as a table: still aggregated (one entry carrying its own depth), or lowered by
    CIRCT into indexed scalars (``0 .. n-1``, depth = how many). A gap or a duplicate index is NOT
    silently closed up -- a table read as seven slots because index 3 was missed is a warp denominator
    that is quietly 12% too small.

    **A lane mask with no slot dimension is not the warp table**, and is returned separately. This is
    the structural discriminator, not a name test: the warp table is per-warp *by construction*, so a
    single register holding one mask cannot be it. On the design here that is exactly what separates
    the warp scheduler's ``threadMasks`` (16 bits x 8 slots) from the FP unit's per-request
    ``commonState_N_req_laneMask`` (4 bits, one per pipeline entry) -- take the second and the reported
    machine is four lanes wide. The cost is honest and stated: a genuinely single-warp core would have
    a slotless mask and is reported UNKNOWN here rather than guessed at.
    """
    out: list[MaskTable] = []
    scalars: list[MaskTable] = []
    for (module, base), entries in sorted(seen.items()):
        widths = {w for w, _d, _e in entries.values()}
        if len(widths) != 1:
            problems.append(f"{module}.{base}: lane-mask slots declare {sorted(widths)} different "
                            f"widths; the lane count is UNKNOWN, not the first of them")
            continue
        width = next(iter(widths))
        idxs = sorted(i for i in entries if i is not None)
        if None in entries:
            if idxs:
                problems.append(f"{module}.{base}: both an aggregate and indexed slots declare the "
                                f"same lane mask; the slot count is UNKNOWN")
                continue
            depth = entries[None][1]
            evidence = entries[None][2]
        else:
            if idxs != list(range(len(idxs))):
                problems.append(f"{module}.{base}: lane-mask slot indices {idxs} are not a contiguous "
                                f"0..n-1 run; the slot count is UNKNOWN")
                continue
            depth = len(idxs)
            evidence = entries[idxs[0]][2]
        table = MaskTable(module=module, base=base, width=width, depth=depth, evidence=evidence)
        (out if depth > 1 else scalars).append(table)
    return out, scalars


def _is_firrtl_header(stripped: str) -> bool:
    """Whether a line is a FIRRTL ``circuit``/``module`` header.

    The second token must be an IDENTIFIER and the line must carry the header's colon. Without that
    test an MLIR file's own ``module {`` wrapper -- which every ``.hw.mlir`` opens with -- reads as a
    FIRRTL module, the whole file is then parsed in the wrong dialect, and the honest-looking answer
    that comes back is "this design declares no lanes".
    """
    head = stripped.split()
    if len(head) < 2 or ":" not in stripped:
        return False
    if head[0] not in (_FIR_CIRCUIT, *_FIR_MODULE):
        return False
    return head[1].rstrip(":").isidentifier()


def sniff_dialect(lines: Iterable[str], *, limit: int = 4000) -> str | None:
    """``"hw"`` / ``"firrtl"`` / ``None``, from what the file's own structural lines DECLARE.

    Decided by content, not by filename, because the same geometry is asked of a ``.fir`` and a
    ``.hw.mlir`` and a mis-guessed dialect reads as "this design declares no lanes". The HW markers win
    outright: an MLIR file legitimately opens with a bare ``module {`` before its first ``hw.module``.
    """
    firrtl_seen = False
    for n, raw in enumerate(lines):
        if n > limit:
            break
        s = raw.lstrip()
        if s.startswith(_HW_MODULE_OP) or s.startswith(_HW_INSTANCE_OP):
            return "hw"
        if not firrtl_seen and _is_firrtl_header(s):
            firrtl_seen = True
    return "firrtl" if firrtl_seen else None


# --------------------------------------------------------------------------------------------
# The derivation
# --------------------------------------------------------------------------------------------


def geometry_from_elaboration(el: Elaboration) -> SimtGeometry:
    """Resolve the three dimensions from one elaboration, refusing wherever it answers twice."""
    keys: dict[str, str] = {}
    ambiguous: list[str] = []
    unread: dict[str, str] = {}
    lane = warps = cores = None
    core_modules: list[str] = []

    if not el.masks:
        why = ("no per-warp lane-mask table was found; whether this design is a SIMT machine is "
               "UNKNOWN")
        if el.scalar_masks:
            why += (f" (the lane-mask state that WAS found carries no slot dimension: "
                    f"{sorted(f'{m.module}.{m.base}:{m.width}b' for m in el.scalar_masks)})")
        unread[LANE] = unread[WARP] = why
    else:
        shapes = {(m.width, m.depth) for m in el.masks}
        if len(shapes) > 1:
            # TWO ANSWERS IS NOT AN ANSWER. Picking the widest (or the first) publishes a denominator
            # nobody elaborated, and an occupancy figure is only as honest as its denominator.
            ambiguous.extend((LANE, WARP))
            unread[LANE] = unread[WARP] = (
                f"{len(el.masks)} lane-mask table(s) declare different shapes {sorted(shapes)}; the "
                f"lane and warp denominators are UNKNOWN, not the first of them")
        else:
            # SEVERAL MODULES DECLARING THE SAME-SHAPED TABLE ARE SEVERAL CORES, not a contradiction.
            # A pre-dedup FIRRTL circuit UNIQUIFIES a module per instance, so a two-core cluster
            # elaborates as `WarpScheduler` and `WarpScheduler_1` -- identical structures under
            # different names. Refusing there would report every multi-core design as undeterminable,
            # while the identically-shaped copies are exactly the evidence that there are N cores.
            # The shape check above is what keeps the sum honest: two genuinely different tables
            # (different width or depth) still refuse rather than being added together.
            table = el.masks[0]
            lane, warps = table.width, table.depth
            core_modules = sorted({m.module for m in el.masks})
            keys[LANE] = f"{table.module}.{table.base} element width ({el.dialect}): {table.evidence}"
            keys[WARP] = (f"{table.module}.{table.base} slot count = {table.depth} "
                          f"({el.dialect} lane-mask table depth)")

    if core_modules:
        # The top is where an instance path starts. FIRRTL states it (`circuit X :`); the HW dialect
        # does not, so the top is derived as the root that actually reaches a core -- a file may carry
        # several uninstantiated modules, and only one of them is the elaborated design.
        candidates = [el.declared_top] if el.declared_top else []
        if not candidates:
            candidates = [r for r in el.roots() if any(el.reaches(r, m) for m in core_modules)]
            if not candidates:
                candidates = [p for p in el.public if any(el.reaches(p, m) for m in core_modules)]
        if not candidates:
            unread[CORE] = (f"no elaborated top instantiating {core_modules} could be identified, so "
                            f"the number of cores is UNKNOWN")
        elif len(candidates) > 1:
            ambiguous.append(CORE)
            unread[CORE] = (f"{len(candidates)} candidate tops {sorted(candidates)} each instantiate "
                            f"a SIMT core; the file describes more than one design")
        else:
            top = candidates[0]
            counts = {m: el.multiplicity(top, m) for m in core_modules}
            cyclic = sorted(m for m, n in counts.items() if n is None)
            if cyclic:
                unread[CORE] = (f"the instance graph under {top!r} is cyclic at {cyclic}; an "
                                f"instance-path count is undefined on it")
            else:
                total = sum(n for n in counts.values() if n)
                if total <= 0:
                    unread[CORE] = f"{core_modules} is never instantiated under {top!r}"
                else:
                    cores = total
                    keys[CORE] = (f"{total} instance path(s) of "
                                  f"{ {m: n for m, n in sorted(counts.items())} } under top {top!r} "
                                  f"({el.dialect} instance graph)")

    return SimtGeometry(lane_width=lane, warps_per_core=warps, cores=cores, keys=keys,
                        ambiguous=tuple(dict.fromkeys(ambiguous)), unread=unread,
                        core_module=(", ".join(core_modules) if core_modules else None),
                        dialect=el.dialect, source=el.source)


def derive_from_lines(lines: Iterable[str], *, dialect: str | None = None,
                      source: str | None = None) -> SimtGeometry:
    reader = {"hw": parse_hw_dialect, "firrtl": parse_firrtl}.get(dialect or "")
    if reader is None:
        raise ValueError(f"unknown CIRCT dialect {dialect!r}")
    return geometry_from_elaboration(reader(lines, source=source))


def derive_from_text(text: str, *, dialect: str | None = None,
                     source: str | None = None) -> SimtGeometry:
    """Derive the geometry from an in-memory elaboration (how the tests pin every refusal)."""
    d = dialect or sniff_dialect(text.splitlines())
    if d is None:
        return SimtGeometry(source=source,
                            unread={r: "the text declares neither a FIRRTL circuit nor a CIRCT HW "
                                       "module, so no elaboration could be read" for r in ROLES})
    return derive_from_lines(text.splitlines(), dialect=d, source=source)


def derive_from_path(path: str | Path) -> SimtGeometry:
    """Derive the geometry from an elaborated artifact on disk, STREAMED.

    Read twice (once to sniff the dialect, once to parse) rather than slurped: the elaborated FIRRTL for
    a SIMT SoC on this host is 427 MB, and holding it in memory to search it would cost more than the
    design it describes.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        dialect = sniff_dialect(fh)
    if dialect is None:
        return SimtGeometry(source=str(p),
                            unread={r: f"{p.name} declares neither a FIRRTL circuit nor a CIRCT HW "
                                       f"module" for r in ROLES})
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        return derive_from_lines(fh, dialect=dialect, source=str(p))


# --------------------------------------------------------------------------------------------
# Locating the elaboration -- without naming a target and without typing a path
# --------------------------------------------------------------------------------------------

#: The elaborated-output accessor a target's own SIMT introspect exposes. Asking the introspect rather
#: than holding a path is what keeps this module target-agnostic: it resolves the checkout from the
#: descriptor / dotenv, so moving the checkout moves the answer with it.
_GEN_SRC_ACCESSOR = "_gen_src_dir"

#: File extensions of an elaborated CIRCT artifact, most authoritative first. The HW dialect leads
#: because it is what the arc model is built from -- the artifact merlin's own fact bundle already
#: cites -- and it is an order of magnitude smaller than the FIRRTL that produced it. Matched on the
#: EXTENSION, not on a longer tail: mlc names its HW dialect `<design>_hw.mlir`, so a `.hw.mlir` test
#: would rank the preferred artifact last and hand authority to whatever else was found.
_ARTIFACT_SUFFIXES = (".mlir", ".fir")


def artifact_candidates(target: str) -> tuple[list[tuple[Path, str]], list[str]]:
    """``([(path, how)], [why each route failed])`` -- the elaborated artifacts for ``target``.

    Three routes, each asking something that already knows where the RTL is, and none holding a path:

    1. **The CIRCT HW dialect mlc keeps for this target** (``core_hw_mlir``) -- an ``circt-arc`` model's
       own input, i.e. CIRCT having compiled the RTL. This is the preferred source.
    2. **The same, via the SIMT introspect that DECLARES it serves this target.** The route a composite
       target needs: a cluster whose elaboration mlc registers under the embedded core's identity
       resolves through that identity rather than reporting UNKNOWN under its own name.
    3. **That introspect's elaborated-output directory**, searched for a FIRRTL circuit -- the RTL
       repo's own elaborated output, used when no CIRCT HW artifact has been produced.
    """
    found: list[tuple[Path, str]] = []
    why: list[str] = []

    from . import mlc_bridge

    def _hw(name: str, how: str) -> None:
        try:
            p = mlc_bridge.core_hw_mlir(name)
        except Exception as e:                                 # noqa: BLE001 -- mlc unreachable
            why.append(f"mlc could not be asked for {name!r}'s HW dialect ({type(e).__name__})")
            return
        if p is None:
            why.append(f"mlc holds no CIRCT HW dialect for {name!r}")
        else:
            found.append((Path(p), how))

    _hw(target, "CIRCT HW dialect (mlc arc model input) for this target")

    intro = None
    try:
        intro = mlc_bridge._resolve_simt_introspect(target)
    except Exception as e:                                     # noqa: BLE001 -- registry unreachable
        why.append(f"the SIMT introspect registry is unreachable ({type(e).__name__})")
    if intro is None:
        why.append(f"no SIMT RTL introspect is registered for {target!r}")
    else:
        served = str(getattr(intro, "TARGET", "") or "")
        if served and served != target:
            _hw(served, f"CIRCT HW dialect for {served!r}, the identity whose SIMT introspect "
                        f"serves {target!r}")
        accessor = getattr(intro, _GEN_SRC_ACCESSOR, None)
        if accessor is None:
            why.append(f"the SIMT introspect serving {target!r} exposes no {_GEN_SRC_ACCESSOR}()")
        else:
            try:
                gen_src = Path(str(accessor()))
            except Exception as e:                             # noqa: BLE001
                why.append(f"{_GEN_SRC_ACCESSOR}() raised {type(e).__name__}")
                gen_src = None
            if gen_src is not None:
                hits = [p for p in sorted(gen_src.rglob("*.fir")) if p.is_file() and p.stat().st_size]
                if hits:
                    found.extend((p, f"elaborated FIRRTL under {gen_src.name}") for p in hits)
                else:
                    why.append(f"no non-empty elaborated FIRRTL under {gen_src}")

    seen: set[str] = set()
    uniq = [(p, how) for p, how in found if not (str(p) in seen or seen.add(str(p)))]
    uniq.sort(key=lambda ph: next((i for i, s in enumerate(_ARTIFACT_SUFFIXES)
                                   if ph[0].suffix == s), len(_ARTIFACT_SUFFIXES)))
    return uniq, why


def geometry_for_target(target: str, *, artifact_path: str | Path | None = None,
                        artifact_text: str | None = None,
                        dialect: str | None = None) -> dict[str, Any]:
    """Derive ``target``'s SIMT geometry from an ELABORATED CIRCT/RTL artifact. Three states.

    ``derived`` when an elaboration resolved at least the lane width -- the one dimension without which
    no occupancy figure has a denominator. ``absent`` when an elaboration WAS READ and declares no
    per-warp lane mask (a real fact about that design: it is not a SIMT machine). ``unavailable`` when
    nothing could be located or read, or when what was read answers a dimension two ways -- UNKNOWN,
    which is not absent.

    There is deliberately NO route here to a cycle model's configuration. A model config is a statement
    about the model; letting it stand in when the RTL is missing is the exact substitution this module
    was written to undo, and it would be invisible in the output.
    """
    if artifact_text is not None:
        geom = derive_from_text(artifact_text, dialect=dialect,
                                source=(str(artifact_path) if artifact_path else "<supplied text>"))
        return _state(geom, read=[geom.source or "<supplied text>"], unread={}, routes=[])

    if artifact_path is not None:
        candidates: list[tuple[Path, str]] = [(Path(artifact_path), "supplied by the caller")]
        routes: list[str] = []
    else:
        candidates, routes = artifact_candidates(target)
    if not candidates:
        return {"status": "unavailable", "target": target, "routes_tried": routes,
                "why": f"no elaborated CIRCT/RTL artifact could be located for {target!r}; its SIMT "
                       f"geometry is UNKNOWN, not absent. A cycle model's config is NOT a fallback: it "
                       f"describes the model, not the hardware"}

    readings: list[tuple[SimtGeometry, str]] = []
    read: list[str] = []
    unreadable: dict[str, str] = {}
    for path, how in candidates:
        try:
            got = derive_from_path(path)
        except OSError as e:
            # A placeholder path an introspect emits when its toolchain env is unset lands here, and it
            # must read as "we could not look", never as "the design declares no lanes".
            unreadable[str(path)] = f"{type(e).__name__}: {e}"
            continue
        read.append(f"{path} ({how})")
        readings.append((got, how))

    # EVERY candidate is read, not just the first that answers. Two elaborations of the same generator
    # are two different machines whenever the config differed, and the one that shipped a wrong number
    # here is exactly the case that has to be visible: stopping at the first full answer would have
    # hidden that the cluster HW dialect carries two cores while the single-core test config carries
    # one. The preferred artifact is authoritative; the others are reported beside it, disagreements
    # named, so nothing is chosen silently.
    resolved = [(g, how) for g, how in readings if g.lane_width is not None]
    best, best_how = (resolved or readings or [(SimtGeometry(), "")])[0]
    envelope = _state(best, read=read, unread=unreadable, routes=routes)
    envelope["authority"] = (f"{best.source} -- {best_how}" if best.source else None)
    envelope["corroboration"] = [
        {"source": g.source, "how": how, "dialect": g.dialect,
         **{r: g.value(r) for r in ROLES}, "keys": dict(sorted(g.keys.items())),
         "unread": dict(sorted(g.unread.items()))}
        for g, how in readings]
    contested = {}
    for role in ROLES:
        vals = {g.value(role) for g, _ in resolved if g.value(role) is not None}
        if len(vals) > 1:
            contested[role] = sorted(
                {(g.source or "?"): g.value(role) for g, _ in resolved}.items(),
                key=lambda kv: str(kv[0]))
    if contested:
        envelope["contested"] = {r: [{"source": src, "value": v} for src, v in pairs]
                                 for r, pairs in contested.items()}
        envelope["why_contested"] = (
            f"the elaborations on this host disagree about {sorted(contested)}: they are DIFFERENT "
            f"elaborated configurations of one generator, not one machine described twice. The value "
            f"reported is the preferred artifact's ({envelope['authority']}); every reading is listed "
            f"under 'corroboration' so the mismatch is quoted rather than resolved out of sight")
    return envelope


def _state(geom: SimtGeometry, *, read: list[str], unread: dict[str, str],
           routes: list[str]) -> dict[str, Any]:
    """Wrap a derivation in the three-state envelope, keeping ABSENT and UNAVAILABLE apart."""
    if geom.lane_width is None:
        # ABSENT REQUIRES HAVING READ SOMETHING. Falling through to "this machine declares no SIMT
        # geometry" when every candidate failed to open would report our inability to look as a
        # property of the hardware.
        if not read:
            return {"status": "unavailable", "read": read, "unreadable": unread,
                    "routes_tried": routes, "geometry": geom.to_dict(),
                    "why": "no candidate elaboration could be READ, so whether this target declares a "
                           "SIMT geometry is UNKNOWN, not absent"}
        if geom.ambiguous:
            return {"status": "unavailable", "read": read, "unreadable": unread,
                    "geometry": geom.to_dict(),
                    "why": f"the elaboration answers {list(geom.ambiguous)} more than one way; two "
                           f"answers is not an answer, and a guessed denominator would price the run "
                           f"against a machine that was never run"}
        return {"status": "absent", "read": read, "unreadable": unread,
                "geometry": geom.to_dict(),
                "why": "the elaboration was read and carries no per-warp lane-mask state, so it does "
                       "not describe a SIMT machine and there is no lane denominator to occupy"}
    return {"status": "derived", "source": geom.source, "read": read, "unreadable": unread,
            "geometry": geom.to_dict(),
            "missing": [r for r in ROLES if geom.value(r) is None],
            "note": ("lane_width is the divergence denominator; warps_per_core and cores extend it to "
                     "the core and the cluster, and each is UNKNOWN on its own when unelaborated")}


def geometry_from_dict(d: Mapping[str, Any]) -> SimtGeometry:
    """Rebuild a :class:`SimtGeometry` from its ``to_dict`` form (the envelope's ``geometry`` block)."""
    return SimtGeometry(lane_width=d.get("lane_width"), warps_per_core=d.get("warps_per_core"),
                        cores=d.get("cores"), keys=dict(d.get("keys") or {}),
                        ambiguous=tuple(d.get("ambiguous") or ()),
                        unread=dict(d.get("unread") or {}), core_module=d.get("core_module"),
                        dialect=str(d.get("dialect") or ""), source=d.get("source"))
