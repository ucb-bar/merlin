"""Target-neutral named compiler actions with explicit composition constraints.

Adapted from feat/OPU-VME 80e00803d; legacy and OOT registries share record identity
and composition behavior, while each provider owns its selected declarations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ImprFeature:
    """One named, default-off compiler change.

    ``edit_pipeline`` rewrites the ordered MLIR pass list (PASS/HEURISTIC at the pipeline level).
    ``edit_schedule`` rewrites the transform-dialect schedule text (PATTERN at the schedule level
    that goes beyond what a hand-written ``schedule.mlir`` knob expresses). Either may be None.
    Both are pure functions (input -> new value); they must be deterministic.

    ``schedule_replace`` flags an ``edit_schedule`` that IGNORES its input text and emits a full
    replacement schedule (e.g. the tiled-vfmacc / accumulator-resident recipes whose structure
    differs fundamentally from the tile+vectorize baseline). Two such features cannot COMPOSE — the
    last one in sorted order would silently clobber the other's clamp — so ``apply_schedule`` refuses
    to apply more than one full-replacement schedule feature at once (the composed, whole-model-safe
    config is a SINGLE feature with all the clamps inherent). Additive edits (``schedule_replace`` =
    False, e.g. ``lmul_widen_n``'s string substitution) layer freely on top of a replacement.

    ``edit_cflags`` rewrites the clang flag list for the model object (KNOB at the BACKEND level).
    It exists because some codegen decisions are not expressible in the IR at all: the width of the
    vector register group the auto-vectorizer asks for is a backend query, not a type, so no tile
    size or schedule edit reaches it directly -- the only way this registry could move it before was
    to widen the N tile and let the group width follow, which changes the vectorized shapes as a side
    effect and can push a transfer into a masked form the backend rejects (see ``lmul_group``). A
    ``None`` hook leaves the flags untouched, so a build that names no cflag-editing feature compiles
    byte-identically.

    ``implies`` names features this one CANNOT be measured without -- lowering hygiene that is a
    property of the recipe's SHAPE rather than a tuning choice. It exists because a default-off
    feature whose payoff is cancelled by a separate default-off fix is, in practice, an inert lever:
    the beam has to discover the conjunction, and everyone who names the feature directly in
    ``compiler_features`` gets the cancelled version. See ``_tile_epilogue_hygiene``.

    ``alternative_group`` names a family whose members are alternative choices: at most one
    member may be selected. ``requires_exactly_one_of`` names explicit prerequisites from which
    exactly one must be selected. Both constraints are checked after the ``implies`` closure;
    neither selects a prerequisite or drops a conflicting feature automatically.
    """

    name: str
    action_class: str  # "PASS" | "HEURISTIC" | "PATTERN"
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None
    edit_cflags: Callable[[list[str]], list[str]] | None = None
    schedule_replace: bool = False
    implies: frozenset[str] = frozenset()
    alternative_group: str | None = None
    requires_exactly_one_of: frozenset[str] = frozenset()


class CompositionError(ValueError):
    """Two features both fully REPLACE the transform schedule -> they cannot compose (one would
    silently clobber the other). The whole-model-safe composed config is a SINGLE feature with all
    clamps inherent; raise instead of letting sorted order pick a winner."""


class FeatureRegistry:
    """Per-backend named feature actions; no process-global RVV registration."""

    def __init__(self, lazy_register: Callable[[str], bool] | None = None):
        self.entries: dict[str, ImprFeature] = {}
        self.lazy_register = lazy_register

    def register(self, feature: ImprFeature) -> ImprFeature:
        if feature.name in self.entries:
            raise ValueError(f"duplicate impr feature {feature.name!r}")
        self.entries[feature.name] = feature
        return feature

    def get(self, name: str) -> ImprFeature:
        if name not in self.entries and (self.lazy_register is None or not self.lazy_register(name)):
            raise KeyError(f"unknown impr feature {name!r}; registered: {sorted(self.entries)}")
        return self.entries[name]

    def known(self) -> list[str]:
        return sorted(self.entries)

    def normalize(self, features) -> frozenset[str]:
        if not features:
            return frozenset()
        fs = frozenset(features)
        for name in fs:
            self.get(name)
        while True:
            grown = fs | frozenset().union(*(self.get(name).implies for name in fs)) if fs else fs
            if grown == fs:
                break
            for name in grown - fs:
                self.get(name)
            fs = grown
        groups: dict[str, list[str]] = {}
        for name in sorted(fs):
            feature = self.get(name)
            if feature.alternative_group is not None:
                groups.setdefault(feature.alternative_group, []).append(name)
        for group, members in sorted(groups.items()):
            if len(members) > 1:
                raise ValueError(f"feature alternatives in group {group!r} cannot compose: {members}")
        for name in sorted(fs):
            required = self.get(name).requires_exactly_one_of
            selected = sorted(fs & required)
            if required and len(selected) != 1:
                raise ValueError(f"feature {name!r} requires exactly one of {sorted(required)}; selected {selected}")
        return fs

    def apply_pipeline(self, passes: list[str], features: frozenset[str]) -> list[str]:
        if not features:
            return passes
        out = list(passes)
        for name in sorted(features):
            edit = self.get(name).edit_pipeline
            if edit is not None:
                out = edit(out)
        return out

    def apply_schedule(self, schedule_text: str, features: frozenset[str]) -> str:
        if not features:
            return schedule_text
        replacers = [
            self.get(name)
            for name in sorted(features)
            if self.get(name).edit_schedule is not None and self.get(name).schedule_replace
        ]
        if len(replacers) > 1:
            raise CompositionError(
                "cannot compose multiple full-schedule-replacement features "
                f"{[feature.name for feature in replacers]}: each emits a complete transform schedule and would "
                "clobber the others. Enable the single composed feature that carries all clamps "
                "inherent (e.g. accumulator_resident_wholemodel) instead of stacking replacements."
            )
        out = schedule_text
        if replacers:
            out = replacers[0].edit_schedule(out)
        for name in sorted(features):
            feature = self.get(name)
            if feature.edit_schedule is not None and not feature.schedule_replace:
                out = feature.edit_schedule(out)
        return out

    def apply_cflags(self, cflags, features: frozenset[str]) -> list[str]:
        out = list(cflags)
        if not features:
            return out
        for name in sorted(features):
            edit = self.get(name).edit_cflags
            if edit is not None:
                out = list(edit(out))
        return out
