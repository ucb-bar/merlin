"""Independent runtime authority shared by functional and performance phases.

Runtime support is qualified from public hardware/tool sources and actual
execution controls. It is separate from the newly authored compiler, its frozen
optimization baseline, and the private handwritten final reference. Constructors
and serialized reports cannot issue this live capability.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from weakref import WeakKeyDictionary

from .contracts import StageGateError, document_sha256, sha256_file

_ISSUED = WeakKeyDictionary()
_ROLES = ("grade", "stage_verifier", "feature_provider", "cca_provider", "rtl_executor")


def _qualification_types():
    from .component_measurement_qualification import IndependentMeasurementQualification
    from .component_runtime_qualification import IndependentRuntimeQualification

    return IndependentRuntimeQualification, IndependentMeasurementQualification


@dataclass(frozen=True)
class IndependentRuntimeServices:
    """Actual ordinary grading and observation methods of independent support."""

    grade: Callable
    stage_verifier: Callable
    feature_provider: Callable | None = None
    cca_provider: Callable | None = None
    rtl_executor: Callable | None = None


def _callback_identity(callback):
    original = callback
    while isinstance(callback, partial):
        callback = callback.func
    owner = inspect.getsourcefile(callback) if callable(callback) else None
    if owner is None:
        raise StageGateError("independent runtime callback lacks an inspected source owner")
    path = Path(owner).resolve()
    function = callback.__func__ if inspect.ismethod(callback) else callback
    return (original, getattr(function, "__code__", None), path, sha256_file(path),
            getattr(function, "__module__", None), getattr(function, "__qualname__", None))


@dataclass(frozen=True, eq=False)
class IndependentComponentRuntime:
    """Issued only after the fixed independent runtime qualifier executes."""

    qualification: object
    services: IndependentRuntimeServices
    hardware_intake: object
    target_descriptor: Path
    source_pins: tuple[tuple[Path, str], ...]
    qualified_roles: tuple[str, ...]
    qualification_sha256: str
    callbacks: tuple[tuple[str, tuple], ...]

    @property
    def sha256(self):
        return document_sha256({
            "schema": "merlin.independent_component_runtime.v1",
            "qualification_sha256": self.qualification_sha256,
            "hardware_intake_sha256": self.hardware_intake.sha256,
            "target_descriptor_sha256": sha256_file(self.target_descriptor),
            "qualified_roles": self.qualified_roles,
            "source_pins": [(str(path), digest) for path, digest in self.source_pins],
            "callback_sources": [
                (role, str(identity[2]), identity[3], identity[4], identity[5]) for role, identity in self.callbacks
            ],
        })

    @property
    def grader(self):
        return self.services.grade

    @property
    def stage_verifier(self):
        return self.services.stage_verifier

    def verify(self, *, required_roles=()):
        admitted = _ISSUED.get(self)
        if admitted is None:
            raise StageGateError("component runtime requires independently evaluated live authority")
        if type(self.qualification) not in _qualification_types():
            raise StageGateError("component runtime qualification is not the fixed independent owner")
        self.qualification.verify()
        self.hardware_intake.verify()
        if (self.qualification.services is not self.services
            or self.qualification.hardware_intake is not self.hardware_intake
            or self.qualification.target_descriptor != self.target_descriptor
            or self.qualification.receipt_sha256 != self.qualification_sha256
            or tuple(self.qualification.source_pins) != self.source_pins
            or tuple(self.qualification.qualified_roles) != self.qualified_roles):
            raise StageGateError("independent runtime selection changed after its evaluated controls")
        if type(self.services) is not IndependentRuntimeServices:
            raise StageGateError("independent runtime service contract changed")
        for path, digest in self.source_pins:
            if path.is_symlink() or path.resolve() != path or not path.is_file() or sha256_file(path) != digest:
                raise StageGateError("independent runtime source/tool bytes changed")
        for role, identity in self.callbacks:
            if _callback_identity(getattr(self.services, role)) != identity:
                raise StageGateError("independent runtime callback implementation changed")
        if admitted != self.sha256:
            raise StageGateError("independent runtime derivation/control identity changed")
        if not set(required_roles) <= set(self.qualified_roles):
            raise StageGateError("independent runtime has unqualified requested roles: "
                                 + ",".join(sorted(set(required_roles) - set(self.qualified_roles))))
        return self.sha256


def admit_independent_component_runtime(qualification):
    """Connect a live qualified public-source runtime; metadata alone refuses.

    The fixed qualifier owns independent RTL semantic derivation, source/build
    correspondence, namespace isolation and actual positive/negative execution
    controls. Structural intake does not discharge those runtime obligations.
    The qualifier may certify functional roles before timing/CCA roles are ready.
    """
    try:
        from merlin_experiments.phase0.rtl_intake import IndependentHardwareIntake

    except ImportError as error:
        raise StageGateError("independent runtime semantic/execution qualifier is unavailable") from error
    if type(qualification) not in _qualification_types():
        raise StageGateError("independent runtime admission requires a live evaluated qualifier")
    qualification.verify()
    if type(qualification.hardware_intake) is not IndependentHardwareIntake:
        raise StageGateError("independent runtime requires actual separately issued RTL intake")
    qualification.hardware_intake.verify()
    services = qualification.services
    roles = tuple(qualification.qualified_roles)
    if (type(services) is not IndependentRuntimeServices or len(roles) != len(set(roles))
        or not {"grade", "stage_verifier"} <= set(roles) or not set(roles) <= set(_ROLES)):
        raise StageGateError("independent runtime qualification lacks ordinary grading and semantic witness roles")
    pins = tuple((Path(path), digest) for path, digest in qualification.source_pins)
    if not pins or len(dict(pins)) != len(pins):
        raise StageGateError("independent runtime needs complete distinct source/tool membership")
    callbacks = tuple((role, _callback_identity(getattr(services, role))) for role in roles)
    for _role, identity in callbacks:
        if (identity[2], identity[3]) not in pins:
            raise StageGateError("independent runtime callback source is absent from qualified membership")
    runtime = IndependentComponentRuntime(
        qualification, services, qualification.hardware_intake, Path(qualification.target_descriptor),
        pins, roles, qualification.receipt_sha256, callbacks,
    )
    _ISSUED[runtime] = runtime.sha256
    runtime.verify(required_roles=("grade", "stage_verifier"))
    return runtime


def require_independent_runtime(runtime, *, required_roles, target_descriptor):
    if type(runtime) is not IndependentComponentRuntime:
        raise StageGateError("component evaluation requires independent runtime/semantic measurement authority")
    runtime.verify(required_roles=required_roles)
    if runtime.target_descriptor != Path(target_descriptor):
        raise StageGateError("independent component runtime targets another descriptor")
    return runtime
