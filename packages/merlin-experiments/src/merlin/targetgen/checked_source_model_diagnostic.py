"""Explicit original-host L2 diagnostics for checked complete source programs.

The selector transports an independently reviewed live effect owner. It grants
neither a public-program basis nor frozen placement, certification or timing.
The default model routes retain their original interpretation.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import inspect
import json
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

from merlin.common.digest import sha256_file
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, active_compiler_passes

SCHEMA = "merlin.checked_source_model_diagnostic.v1"
MECHANISM = "source_execution_diagnostic"
CONTEXT_FIELD = "checked_source_diagnostic"
_ISSUER = object()
_ISSUED = weakref.WeakKeyDictionary()
_ACTIVE = contextvars.ContextVar("original_checked_source_model_diagnostic", default=None)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pin(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("checked source diagnostic needs original regular source leaves")
    return str(path), sha256_file(path)


def _callback(owner):
    callback = getattr(owner, "create_source_execution_owner", None)
    if type(callback) is not MethodType or callback.__self__ is not owner:
        raise ValueError("checked source diagnostic requires its original live effect issuer")
    file = inspect.getsourcefile(callback.__func__)
    if file is None:
        raise ValueError("checked source diagnostic effect issuer has no original source")
    return callback.__func__, callback.__func__.__code__, _pin(Path(file).resolve())


@dataclass(frozen=True, eq=False)
class CheckedSourceModelDiagnostic:
    """Live host selection; records alone cannot instantiate this capability."""

    _passes: CompilerPassSelection
    _record_json: str
    _callback_pin: tuple
    _sources: tuple
    _issuer: object

    def revalidate(self):
        if (type(self) is not CheckedSourceModelDiagnostic or self not in _ISSUED or self._issuer is not _ISSUER
                or _ISSUED[self] != hashlib.sha256(self._record_json.encode()).hexdigest()):
            raise ValueError("checked source diagnostic was not issued by the original host")
        with self._passes.metadata_operation():
            self._passes.revalidate()
            row = json.loads(self._record_json)
            capability = self._passes.registry.resolve(MECHANISM, self._passes.binding)
            if (
                row.get("schema") != SCHEMA
                or row.get("scope") != "complete sealed source L0/L1/L2 diagnostic only"
                or row.get("formal_admission") is not False
                or row.get("promotion_authority") != "NONE"
                or row.get("oracle_tier_ceiling") != "L2"
                or row.get("compiler_pass_selection") != self._passes.identity
                or row.get("binding") != asdict(self._passes.binding)
                or capability is None
                or row.get("effect_capability") != capability.identity
                or _callback(capability.owner) != self._callback_pin
                or any(_pin(path) != (path, digest) for path, digest in self._sources)
                or _pin(row["rtl_facts"]["path"])[1] != row["rtl_facts"]["sha256"]
                or row["rtl_facts"]["sha256"] != self._passes.binding.hardware
            ):
                raise ValueError("checked source diagnostic lost its exact original owner, source or scope")
            return capability

    @property
    def identity(self):
        self.revalidate()
        return hashlib.sha256(self._record_json.encode()).hexdigest()

    def record(self):
        self.revalidate()
        return json.loads(self._record_json)

    def verify_for(self, capsule, *, target):
        self.revalidate()
        row = json.loads(self._record_json)
        required = capsule.get("required_oracle_tiers")
        if (
            active_compiler_passes() is not self._passes
            or target != row["target"]
            or capsule.get("kind") != "model"
            or capsule.get("label") not in {"public", "dev"}
            or capsule.get("oracle_tier_ceiling") != "L2"
            or type(required) is not list
            or any(type(tier) is not str for tier in required)
            or len(required) != len(set(required))
            or not {"L0", "L1", "L2"} <= set(required) <= {"L0", "L1", "L2", "L3"}
        ):
            raise ValueError("checked source diagnostic requires its explicit original public L2 selection")
        return row


def select_checked_source_diagnostic(passes, *, target, rtl_facts, board_config):
    """Select a live original issuer; neither labels nor saved success grant it."""
    from merlin_experiments.phase0 import connected_integer_qualification as source_owner

    from merlin.compile.model_execution_inputs import selected_firrtl

    if type(passes) is not CompilerPassSelection or passes.factory is None:
        raise ValueError("checked source diagnostic needs the original independently replayable pass factory")
    if type(target) is not str or not target or type(board_config) is not str or not board_config:
        raise ValueError("checked source diagnostic requires explicit selected target and RTL configuration")
    with passes.metadata_operation():
        passes.revalidate()
        capability = passes.registry.resolve(MECHANISM, passes.binding)
        if capability is None:
            raise ValueError("no exact reviewed complete-source diagnostic effect issuer")
        facts = selected_firrtl(rtl_facts, target=target, config=board_config)
        if facts["sha256"] != passes.binding.hardware:
            raise ValueError("checked source diagnostic selects different original hardware facts")
        row = {
            "schema": SCHEMA,
            "scope": "complete sealed source L0/L1/L2 diagnostic only",
            "formal_admission": False,
            "promotion_authority": "NONE",
            "oracle_tier_ceiling": "L2",
            "compiler_pass_selection": passes.identity,
            "effect_capability": capability.identity,
            "binding": asdict(passes.binding),
            "target": target,
            "rtl_facts": {key: facts[key] for key in ("path", "sha256")},
            "board_config": board_config,
        }
        sources = tuple(_pin(path) for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("checked_source_model_diagnostic_execution.py").resolve(),
            Path(source_owner.__file__).resolve(),
        ))
        result = CheckedSourceModelDiagnostic(passes, _canonical(row), _callback(capability.owner), sources, _ISSUER)
        _ISSUED[result] = hashlib.sha256(result._record_json.encode()).hexdigest()
        result.revalidate()
        return result


def restore_checked_source_diagnostic(record, *, capsule, target, package_dir):
    """Rejoin data only inside the actual original sealed model child context."""
    from merlin_experiments.phase1.feedback.compiler_profile_child import require_restored_model_context

    require_restored_model_context(CONTEXT_FIELD, record, package_dir=package_dir, target=target)
    passes = active_compiler_passes()
    if type(record) is not dict or record.get("schema") != SCHEMA or type(passes) is not CompilerPassSelection:
        raise ValueError("checked source diagnostic has no original restored owner")
    selection = select_checked_source_diagnostic(
        passes, target=target, rtl_facts=record["rtl_facts"]["path"], board_config=record["board_config"]
    )
    if selection.record() != copy.deepcopy(record):
        raise ValueError("checked source diagnostic differs from its original sealed parent selection")
    selection.verify_for(capsule, target=target)
    return selection


def active_checked_source_diagnostic():
    return _ACTIVE.get()


def verify_host_selection(selection, capsule, *, target, public_selection):
    if type(selection) is not CheckedSourceModelDiagnostic or public_selection is not None:
        raise ValueError("checked source diagnostic requires its separate original live host selection")
    return selection.verify_for(capsule, target=target)


@contextlib.contextmanager
def scoped_checked_source_diagnostic(selection):
    if type(selection) is not CheckedSourceModelDiagnostic:
        raise ValueError("checked source diagnostic scope needs its original issued selector")
    selection.revalidate()
    token = _ACTIVE.set(selection)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        selection.revalidate()


def selection_from_context(record, *, capsule, target, package_dir):
    """Only an actual restored sealed worker may consume the selected record."""
    from merlin_experiments.phase1.feedback.compiler_profile_child import require_restored_model_context

    require_restored_model_context(CONTEXT_FIELD, record, package_dir=package_dir, target=target)
    selection = active_checked_source_diagnostic()
    if type(selection) is not CheckedSourceModelDiagnostic or selection.record() != record:
        raise ValueError("checked source diagnostic has no exact live restored selection")
    selection.verify_for(capsule, target=target)
    return selection
