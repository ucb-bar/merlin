"""Target-neutral compiler feature registry used by scalar and out-of-tree backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ImprFeature:
    name: str
    action_class: str
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None
    edit_cflags: Callable[[list[str]], list[str]] | None = None
    schedule_replace: bool = False
    implies: frozenset[str] = frozenset()
    alternative_group: str | None = None
    requires_exactly_one_of: frozenset[str] = frozenset()


class CompositionError(ValueError):
    """Multiple full schedule replacements cannot compose."""


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
        for group, members in groups.items():
            if len(members) > 1:
                raise ValueError(f"alternative feature group {group!r} selected {members}")
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
