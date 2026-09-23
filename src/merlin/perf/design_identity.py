"""Which DEVICE a tier record's design keys name, resolved through the pin registry.

WHY THIS EXISTS. Two halves of this repo identify a design differently and nothing joined them.
:mod:`merlin.perf.cost_plane` carries a five-key identity taken off a tier record --
``(substrate, hw_config, hwdb_config_artifact_sha256, engine, tier)`` -- and refuses to compare two
counts unless every key matches and is stated. :mod:`merlin.perf.target_reference` carries a single
``design`` string (``FireSimGemminiRocketConfig``) and refuses to score a candidate against a
reference from another one. Both refusals are right. Neither can talk to the other, so a measured
cycle count could not be scored against a measured destination at all: the plane had an identity the
ledger could not read.

WHAT A CONFIGURATION NAME IS WORTH, measured 2026-09-19. Nothing on its own. Two FireSim bitstreams
in this repo elaborate the SAME ``FireSimGemminiRocketConfig`` onto the same U250 platform and are
different devices -- 43,881,284 bytes against 44,668,362, different digests, different Gemmini
revisions. A week of whole-model cycle numbers was quoted under an hw-config the registry had never
heard of; resolved by configuration name, they would have been compared against references measured
on the other one. So the join is NOT by config: it is by the queue hw-config name the run was
submitted under, which each bitstream declares, CONFIRMED against the digest of the HWDB CONFIG ENTRY
the receipt carries -- which is a different file from the built image, and comparing the two is how
this check spent a week refusing exactly the records that supplied their evidence. Two independent
facts have to agree before a design is named.

THREE STATES, NEVER TWO. Named, or ``UNKNOWN(reason)``. "We could not tell which device this is" is
not a softer "the designs differ" -- it means do not score the claim at all, and a caller that
treats an unresolved identity as a mismatch would silently discard measurements that are fine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["BITSTREAM_ROLE", "DesignIdentity", "design_string", "hw_config_index"]

#: The artifact role a FireSim design is declared under. Declared in the registry, not a convention
#: parsed out of a name -- ``Artifact.role`` exists precisely so a role lookup needs no naming rule.
BITSTREAM_ROLE = "firesim_bitstream"


class DesignIdentity(dict):
    """The resolution, as data: ``name``, ``config`` and ``reason`` (set only when unresolved)."""

    @property
    def resolved(self) -> bool:
        return bool(self.get("config"))


def _unknown(reason: str) -> DesignIdentity:
    return DesignIdentity(name=None, config=None, reason=reason)


def hw_config_index(artifacts: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """Queue hw-config name -> the artifact names that declare it.

    A list, not a single name, because two artifacts declaring the same hw-config is a REGISTRY
    DEFECT that must be reported rather than resolved by picking one: the same submission string
    would then mean two devices, which is the confusion this module exists to end.
    """
    if artifacts is None:
        from ..common.provenance import load_artifacts

        artifacts = load_artifacts()
    index: dict[str, list[str]] = {}
    for name, artifact in artifacts.items():
        if getattr(artifact, "role", "") != BITSTREAM_ROLE:
            continue
        for hw_config in getattr(artifact, "hw_configs", ()) or ():
            index.setdefault(str(hw_config), []).append(str(name))
    return index


def design_string(design_keys: Mapping[str, Any], *, artifacts: Mapping[str, Any] | None = None) -> DesignIdentity:
    """The ledger's ``design`` string for a tier record's design keys, or ``UNKNOWN(reason)``.

    Resolution, and every step can refuse:

    1. the record states an ``hw_config``;
    2. exactly one registered bitstream declares that hw-config;
    3. the record's ``hwdb_config_artifact_sha256`` equals that bitstream's declared ``hwdb_digest``
       -- the CONTENT check, which is what makes this a device rather than a name;
    4. the bitstream declares a ``config``.

    Step 3 is skipped, with the reason recorded in the result, when EITHER side is silent: the record
    may state no digest, or the registry may not have recorded the hwdb entry's hash yet. Both settle
    for the weaker name-only evidence and both say so; neither is silent, and neither is treated as a
    mismatch. That distinction is the whole point -- an earlier version compared the record's hwdb
    hash against ``digest``, the hash of the built IMAGE, so the two could never agree and the check
    refused precisely those records that supplied the evidence it asked for.
    """
    if artifacts is None:
        from ..common.provenance import load_artifacts

        artifacts = load_artifacts()

    hw_config = str((design_keys or {}).get("hw_config") or "").strip()
    if not hw_config:
        return _unknown("the record states no hw_config, so there is nothing to resolve a device from")

    index = hw_config_index(artifacts)
    names = index.get(hw_config) or []
    if not names:
        known = ", ".join(sorted(index)) or "(none declared)"
        return _unknown(
            f"no registered bitstream declares hw_config {hw_config!r}; declared hw-configs are: {known}. "
            "Register the device before scoring anything measured on it -- a configuration name that "
            "resolves to nothing is how a result gets attributed to the wrong silicon."
        )
    if len(names) > 1:
        return _unknown(
            f"hw_config {hw_config!r} is declared by more than one bitstream ({', '.join(sorted(names))}); "
            "one submission string cannot mean two devices"
        )

    artifact = artifacts[names[0]]
    # Compare LIKE WITH LIKE. A record states the sha256 of the HWDB CONFIG ENTRY it submitted
    # against; the registry's ``digest`` is the sha256 of the built image. They are different files,
    # so comparing them could never agree -- which inverted this check: a record that supplied its
    # digest was refused, and one that supplied nothing resolved by name. Evidence must not be
    # punished, so the stated digest is now checked against the quantity it actually is.
    declared_hwdb = str(getattr(artifact, "hwdb_digest", "") or "").strip()
    stated_digest = str((design_keys or {}).get("hwdb_config_artifact_sha256") or "").strip()
    if not stated_digest:
        confirmed_by = "hw_config only (the record states no hwdb digest)"
    elif not declared_hwdb:
        # The registry has not recorded this bitstream's hwdb entry hash. That is a gap in the
        # REGISTRY, not a fault in the record, so the name still resolves -- and says so, rather than
        # reporting a byte match it did not make.
        confirmed_by = f"hw_config only ({names[0]!r} declares no hwdb digest, so the record's could not be checked)"
    elif stated_digest != declared_hwdb:
        return _unknown(
            f"the record's hwdb digest {stated_digest[:16]} is not {names[0]!r}'s declared "
            f"{declared_hwdb[:16]}: the hw-config name matches a registered device and the BYTES do "
            "not, which is a different device under a reused name"
        )
    else:
        confirmed_by = "hw_config and hwdb digest"

    config = str(getattr(artifact, "config", "") or "").strip()
    if not config:
        return _unknown(f"bitstream {names[0]!r} declares no config, so it names no design for the ledger")
    return DesignIdentity(name=names[0], config=config, confirmed_by=confirmed_by)
