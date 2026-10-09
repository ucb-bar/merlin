"""Explicit OOT copy support selection for private ordinary diagnostics only.

Live public HW/command/minimal-software membership and exact helper/build bytes
attribute a selection. They do not prove helper semantics, effects or runtime.
No source loader, callbacks, default target, saved ELF or qualification issuer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from merlin.common.quant_formats import get
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService

from .component_runtime_copy_controls import _symbol
from .contracts import StageGateError, document_sha256, sha256_file

MECHANISMS = ("ownership_lifetime", "host_device_synchronization")


def _recipe_record(recipe):
    record = asdict(recipe)
    cls = record.pop("error_cls")
    record["error_cls"] = cls.__module__ + "." + cls.__qualname__
    from .component_runtime_qualification import _argument_binding

    return _argument_binding(record)


@dataclass(frozen=True)
class RuntimeCopyControlSupport:
    command_intake: object
    software_intake: object
    helper_source: Path
    callee_symbol: str
    shape: tuple[int, ...]
    dtype: str
    source_pins: tuple[tuple[Path, str], ...]
    negative_recipes: tuple[tuple[str, HarnessBuildRecipe], ...]

    def verify(self, *, hardware, build, context_pins):
        from merlin_experiments.phase0.command_intake import IndependentCommandIntake
        from merlin_experiments.phase0.software_intake import IndependentSoftwareIntake

        if (
            type(self) is not RuntimeCopyControlSupport
            or type(build) is not BuildOnlyService
            or type(self.command_intake) is not IndependentCommandIntake
            or type(self.software_intake) is not IndependentSoftwareIntake
            or self.command_intake.hardware is not hardware
            or self.software_intake.hardware is not hardware
        ):
            raise StageGateError("copy support requires the same live HW/command/minimal software source selection")
        self.command_intake.verify()
        self.software_intake.verify()
        build.verify(hardware.target)
        try:
            _symbol(self.callee_symbol)
            scalar = get(self.dtype)
        except ValueError as error:
            raise StageGateError("copy support has no explicit integer pointer ABI") from error
        if (
            scalar.is_float
            or scalar.element_bits not in (8, 16, 32, 64)
            or not scalar.signed
            or not isinstance(self.shape, tuple)
            or not self.shape
            or any(type(dim) is not int or dim < 1 for dim in self.shape)
        ):
            raise StageGateError("copy support requires an explicit positive static signed integer domain")
        if (
            not isinstance(self.source_pins, tuple)
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
            or not set(self.source_pins) <= set(context_pins)
            or (self.helper_source, sha256_file(self.helper_source)) not in self.source_pins
            or self.helper_source not in build.recipe.support_sources
            or (str(self.helper_source), sha256_file(self.helper_source)) not in build.source_pins
        ):
            raise StageGateError("copy support omits actual helper/context/build source membership")
        for path, digest in self.source_pins:
            if (
                not path.is_absolute()
                or path.resolve() != path
                or path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != digest
            ):
                raise StageGateError("copy support source selection changed")
        if (
            not isinstance(self.negative_recipes, tuple)
            or len(dict(self.negative_recipes)) != len(self.negative_recipes)
            or any(
                name not in MECHANISMS or type(recipe) is not HarnessBuildRecipe
                for name, recipe in self.negative_recipes
            )
        ):
            raise StageGateError("copy support has an unknown or repeated original diagnostic mechanism")
        for _name, recipe in self.negative_recipes:
            if (
                replace(recipe, cflags=build.recipe.cflags) != build.recipe
                or recipe.cflags[: len(build.recipe.cflags)] != build.recipe.cflags
                or len(recipe.cflags) != len(build.recipe.cflags) + 1
                or not recipe.cflags[-1].startswith("-D")
                or "=" not in recipe.cflags[-1]
            ):
                raise StageGateError("copy defect recipe changes more than its selected preprocessing control")
        return document_sha256(self.record())

    def record(self):
        return {
            "command_intake_sha256": self.command_intake.sha256,
            "software_intake_sha256": self.software_intake.sha256,
            "helper_source": str(self.helper_source),
            "callee_symbol": self.callee_symbol,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "source_pins": [(str(path), digest) for path, digest in self.source_pins],
            "negative_recipes": [(name, _recipe_record(recipe)) for name, recipe in self.negative_recipes],
            "scope": "explicit evaluator copy selection only; helper/effects/runtime/performance UNKNOWN",
        }

    def selected_build(self, *, fixture, build):
        if fixture.case_id.endswith(".positive"):
            return build
        mechanism = fixture.case_id.partition(".")[0]
        recipes = dict(self.negative_recipes)
        if mechanism not in recipes:
            raise StageGateError("copy diagnostic defect preparation remains UNKNOWN")
        return replace(build, recipe=recipes[mechanism])
