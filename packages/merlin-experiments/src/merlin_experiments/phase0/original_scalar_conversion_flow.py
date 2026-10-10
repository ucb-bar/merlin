"""Declared inputs for fresh original registered scalar construction.

Only the ordinary live schema/basis/source owners derive selection identities.
Complete scalar cohorts and unavailable members remain in the existing converter;
these finite products grant no numerical, effect, resource or phase admission.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads

from . import original_scalar_conversion as V
from . import original_standard_ir_plan as P
from .original_reference_flow import _selected
from .original_reference_roster import _pin
from .rtl_intake import _outside

SCHEMA = "merlin.declared_original_scalar_conversion_selection.v1"
METADATA_SCHEMA = "merlin.declared_original_scalar_conversion_selection.v2"


def _declaration(raw, forbidden, *, version=1):
    if (
        type(raw) is not dict
        or set(raw) != {"schema", "capture_checkout", "capture_commit", "mlir_opt", "budget"}
        or type(version) is not int
        or version not in {1, 2}
        or raw["schema"] != (METADATA_SCHEMA if version == 2 else SCHEMA)
        or type(raw["capture_commit"]) is not str
        or len(raw["capture_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in raw["capture_commit"])
    ):
        raise ValueError("declared scalar construction needs closed inputs without saved owner identities")
    budget = raw["budget"]
    if (
        type(budget) is not dict
        or set(budget) != V._INTEGER_LIMITS
        or any(type(value) is not int or value < 1 for value in budget.values())
        or budget["timeout_s"] > 180
    ):
        raise ValueError("declared scalar construction needs complete finite product and promotion budgets")
    V.B.validate_limits({key: budget[key] for key in V.B._LIMITS})
    root = Path(raw["capture_checkout"])
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("declared scalar construction needs an explicit ordinary public compiler source")
    _outside(root, forbidden)
    selected = copy.deepcopy(raw)
    selected["mlir_opt"] = str(_selected(raw["mlir_opt"], forbidden))
    return selected


@dataclass(frozen=True)
class OriginalScalarConversionInputs:
    """Pinned public construction inputs, never a saved conversion authority."""

    selection: Path
    source_pins: tuple[tuple[str, str], ...]
    forbidden: tuple[Path, ...]
    version: int = 1

    @property
    def paths(self):
        return tuple(Path(path) for path, _ in self.source_pins) + (
            Path(loads(self.selection.read_bytes())["capture_checkout"]),
        )

    def verify(self):
        for path, sha256 in self.source_pins:
            _outside(Path(path), self.forbidden)
            if _pin(path)["sha256"] != sha256:
                raise ValueError("declared scalar source, parser or input selection changed")
        selected = _declaration(loads(self.selection.read_bytes()), self.forbidden, version=self.version)
        capture = P.capture_sources(selected)
        if tuple((row["path"], row["sha256"]) for row in capture) != self.source_pins[2:]:
            raise ValueError("declared scalar compiler source membership changed")


def read_selection(pin, *, forbidden, version=1):
    path = _selected(pin, forbidden)
    selected = _declaration(loads(path.read_bytes()), forbidden, version=version)
    capture = P.capture_sources(selected)
    pins = [_pin(path), _pin(selected["mlir_opt"]), *capture]
    for row in pins:
        _outside(Path(row["path"]), forbidden)
    result = OriginalScalarConversionInputs(
        path, tuple((row["path"], row["sha256"]) for row in pins), tuple(forbidden), version
    )
    result.verify()
    return result


def prepare(selected, *, schema_intake, basis, source_record, numerical_semantics, destination):
    """Execute the fixed v2 observer over the actual complete original source roster."""
    if type(selected) is not OriginalScalarConversionInputs:
        raise ValueError("declared scalar construction needs its explicit source inputs")
    if type(schema_intake) is not V.IndependentOperatorSchemaIntake or type(basis) is not V.ComponentSemanticBasis:
        raise ValueError("declared scalar construction needs actual live original schema and basis owners")
    selected.verify()
    destination = Path(destination).absolute()
    _outside(destination, selected.forbidden)
    if (
        destination.exists()
        or ".." in destination.parts
        or any(path.is_symlink() for path in (destination, *destination.parents))
        or any(
            path == destination or path.is_relative_to(destination) or destination.is_relative_to(path)
            for path in selected.paths
        )
    ):
        raise ValueError("declared scalar construction needs a fresh private owner outside its inputs")
    value = _declaration(loads(selected.selection.read_bytes()), selected.forbidden, version=selected.version)
    value.update(
        schema=V.METADATA_SELECTION_SCHEMA if selected.version == 2 else V.INTEGER_SELECTION_SCHEMA,
        source_record_sha256=V._digest(source_record),
        operator_schema_intake_sha256=schema_intake.sha256,
        semantic_basis_sha256=basis.source.sha256,
    )
    V.validate_selection(value, source_record=source_record, schema_intake=schema_intake, basis=basis)
    destination.mkdir(mode=0o700)
    selection = destination / "selection.json"
    selection.write_bytes(canonical_json(value))
    result = V.prepare(
        schema_intake=schema_intake,
        basis=basis,
        source_record=source_record,
        numerical_semantics=numerical_semantics,
        selection=selection,
        destination=destination / "registered",
    )
    selected.verify()
    return result


def summary(owner):
    if type(owner) is not V.OriginalScalarConversion:
        raise ValueError("scalar construction report needs its actual live registered conversion")
    record = owner.record()
    source = loads(owner.source_json)
    return {
        "path": str(Path(record["destination"]) / "receipt.json"),
        "sha256": owner.sha256,
        "original_calls": sum(len(row["calls"]) for row in source["members"]),
        "original_source_slots": sum(len(row["source_members"]) for row in source["members"]),
        "scalar_slots": len(record["members"]),
        "scalar_states": dict(Counter(row["state"] for row in record["members"])),
        "totals": copy.deepcopy(record["totals"]),
        "unavailable_requirements": copy.deepcopy(record["unavailable_requirements"]),
        "scope": record["scope"],
        "numerical_or_mandatory_admission": False,
    }
