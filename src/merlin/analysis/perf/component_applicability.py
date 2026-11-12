"""Frozen joint semantic cells for independent component cost applicability.

These objects are declarations, not hardware or timing capabilities. Exact joint
cells prevent an unmeasured combination of individually observed axis ranges
from acquiring authority. General repetition/range proofs are separate work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_sha256


def _atom(value):
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise ValueError("applicability requires structural semantic vocabulary")
    return value


def _pairs(values, *, positive):
    if type(values) is not tuple or not values:
        raise ValueError("applicability resource/unit coordinates are unavailable")
    names = []
    for row in values:
        if type(row) is not tuple or len(row) != 2 or type(row[1]) is not int or row[1] < int(positive):
            raise ValueError("applicability coordinates require exact nonnegative integers")
        names.append(_atom(row[0]))
    if names != sorted(set(names)):
        raise ValueError("applicability coordinate membership must be canonical and distinct")


@dataclass(frozen=True)
class ComponentApplicabilityCoordinates:
    hardware_sha256: str
    timer_sha256: str
    accuracy_sha256: str
    input_policy_sha256: str
    unit_meaning_sha256: str
    payload_bytes: tuple[tuple[str, int], ...]
    capacity_bytes: tuple[tuple[str, int], ...]
    live_bytes: tuple[tuple[str, int], ...]
    working_set_bytes: tuple[tuple[str, int], ...]
    tile_shape: tuple[int, ...]
    tile_counts: tuple[int, ...]
    tails: tuple[int, ...]
    streaming: bool
    dependency_depth: int
    reuse_state: str
    reuse_count: int
    chain_length: int

    def verify(self):
        if not all(
            is_sha256(value)
            for value in (
                self.hardware_sha256,
                self.timer_sha256,
                self.accuracy_sha256,
                self.input_policy_sha256,
                self.unit_meaning_sha256,
            )
        ):
            raise ValueError("applicability coordinates omit HW/numeric/input/timer/unit identity")
        _pairs(self.payload_bytes, positive=False)
        _pairs(self.capacity_bytes, positive=True)
        _pairs(self.live_bytes, positive=False)
        _pairs(self.working_set_bytes, positive=False)
        capacity, live, working = map(dict, (self.capacity_bytes, self.live_bytes, self.working_set_bytes))
        if set(capacity) != set(live) or set(capacity) != set(working):
            raise ValueError("applicability omitted a physical resource coordinate")
        if any(live[name] > capacity[name] or working[name] < live[name] for name in capacity):
            raise ValueError("applicability live bytes violate capacity or working-set bounds")
        if type(self.streaming) is not bool or (
            not self.streaming and any(working[name] > capacity[name] for name in capacity)
        ):
            raise ValueError("applicability omitted its streaming/capacity regime")
        vectors = (self.tile_shape, self.tile_counts, self.tails)
        if (
            any(type(vector) is not tuple or not vector for vector in vectors)
            or len({len(vector) for vector in vectors}) != 1
            or any(type(value) is not int or value <= 0 for vector in vectors[:2] for value in vector)
            or any(
                type(tail) is not int or not 0 <= tail < tile
                for tail, tile in zip(self.tails, self.tile_shape, strict=True)
            )
        ):
            raise ValueError("applicability tile/tail coordinates are incomplete")
        if (
            any(type(value) is not int or value < 0 for value in (self.dependency_depth, self.reuse_count))
            or type(self.chain_length) is not int
            or self.chain_length < 1
            or self.reuse_state not in ("fresh", "within_invocation", "cross_invocation")
            or (self.reuse_state == "fresh") != (self.reuse_count == 0)
        ):
            raise ValueError("applicability dependency/reuse/composition state is incomplete")
        return self.sha256

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ComponentApplicabilityStratum:
    id: str
    cells: tuple[ComponentApplicabilityCoordinates, ...]
    held_groups: tuple[str, ...]


@dataclass(frozen=True)
class ComponentApplicabilityDomain:
    strata: tuple[ComponentApplicabilityStratum, ...]
    calibration_groups: tuple[str, ...]

    def verify(self):
        if (
            type(self.strata) is not tuple
            or not self.strata
            or any(type(row) is not ComponentApplicabilityStratum for row in self.strata)
            or type(self.calibration_groups) is not tuple
            or not self.calibration_groups
        ):
            raise ValueError("applicability needs independently frozen strata and calibration groups")
        for group in self.calibration_groups:
            _atom(group)
        if len(set(self.calibration_groups)) != len(self.calibration_groups):
            raise ValueError("applicability calibration groups are duplicated")
        ids, cells, scopes = [], [], []
        for row in self.strata:
            ids.append(_atom(row.id))
            if (
                type(row.cells) is not tuple
                or not row.cells
                or any(type(cell) is not ComponentApplicabilityCoordinates for cell in row.cells)
                or type(row.held_groups) is not tuple
                or not row.held_groups
            ):
                raise ValueError("applicability required stratum omits joint cells or held transfer groups")
            for group in row.held_groups:
                _atom(group)
            if len(set(row.held_groups)) != len(row.held_groups) or set(row.held_groups) & set(self.calibration_groups):
                raise ValueError("applicability held transfer groups leak calibration membership")
            for cell in row.cells:
                cells.append(cell.verify())
                scopes.append(
                    (
                        cell.hardware_sha256,
                        cell.timer_sha256,
                        cell.accuracy_sha256,
                        cell.input_policy_sha256,
                        cell.unit_meaning_sha256,
                        cell.capacity_bytes,
                    )
                )
        if len(set(ids)) != len(ids) or len(set(cells)) != len(cells):
            raise ValueError("applicability joint cell/stratum membership is ambiguous")
        if len(set(scopes)) != 1:
            raise ValueError("applicability cells select different HW/numeric/input/timer/unit/resource scope")
        return self.sha256

    @property
    def sha256(self):
        return canonical_sha256(asdict(self))

    def lookup(self, coordinates):
        """Return an exact stratum or explicit UNKNOWN; never enlarge a cell."""
        self.verify()
        if type(coordinates) is not ComponentApplicabilityCoordinates:
            return {"status": "UNKNOWN", "reason": "independently observed applicability coordinates unavailable"}
        coordinates.verify()
        rows = [row for row in self.strata if any(cell == coordinates for cell in row.cells)]
        if not rows:
            return {"status": "UNKNOWN", "reason": "joint payload/capacity/tiling/tail/dependency/reuse outside domain"}
        return {"status": "IN_DOMAIN", "stratum": rows[0].id, "domain_sha256": self.sha256}
