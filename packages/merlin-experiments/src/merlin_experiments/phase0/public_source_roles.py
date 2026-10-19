"""Prospective operator-selected source intent, separate from old score ledgers.

A selected profile accounts for every original declaration, including refused
members. It does not qualify a source, compiler, numerical policy or timing.
Only a live operator selection may transport a role into a new publication.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import json
import os
import weakref
from pathlib import Path

from merlin.common import strict_json
from merlin.common.digest import is_sha256

from . import development_cohort as COHORT
from . import development_sources as SOURCE
from . import operator_exclusion as EXCLUSION

SCHEMA = "merlin.phase0.prospective_public_source_roles.v1"
BINDING = "merlin.phase0.public_source_role_binding.v1"
_ACTIVE = contextvars.ContextVar("merlin_public_source_role_selection", default=None)
_ISSUED = weakref.WeakKeyDictionary()
_SOURCE_ROLES = {
    "development_performance": "objective",
    "functional_guard": "functional_guard",
    "development_control": "functional_guard",
    "upper_geometry_control": "functional_guard",
}


def _leaf(path, *, limit=8 << 20):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > limit
    ):
        raise ValueError("source role selection requires bounded direct original leaves")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("source role leaf exceeds selected payload bound")
    return raw


def _profile(path, sha256):
    raw = _leaf(path)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("prospective operator role profile changed")
    value = strict_json.loads(raw, max_bytes=8 << 20)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "scope", "inventories", "members"}
        or value["schema"] != SCHEMA
        or value["scope"] != "prospective_public_development_before_performance_evaluation"
        or not isinstance(value["inventories"], dict)
        or not value["inventories"]
        or not isinstance(value["members"], list)
        or not value["members"]
        or len(value["members"]) > 4096
    ):
        raise ValueError("prospective roles require a complete explicitly selected source profile")
    originals = {}
    for name, pin in value["inventories"].items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(pin, dict)
            or set(pin) != {"path", "sha256"}
        ):
            raise ValueError("source role inventory identity is malformed")
        manifest_path = Path(pin["path"])
        _leaf(manifest_path)
        manifest = COHORT.verify(manifest_path.parent, manifest_sha256=pin["sha256"])
        for member in manifest["members"]:
            identity = name + ":" + str(member["ordinal"])
            originals[identity] = (manifest_path.parent, member)
    rows = {}
    for row in value["members"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"identity", "request_sha256", "declaration_sha256", "role", "origin", "reason"}
            or not isinstance(row["identity"], str)
            or not isinstance(row["role"], str)
            or row["identity"] not in originals
            or row["identity"] in rows
            or row["role"] not in {"objective", "functional_guard"}
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
            or len(row["reason"]) > 2048
        ):
            raise ValueError("source role profile omitted, duplicated or substituted an original member")
        _, member = originals[row["identity"]]
        request = member["request"]
        declaration = request.get("declaration")
        if (
            row["request_sha256"] != member["request_sha256"]
            or SOURCE._digest(request) != member["request_sha256"]
            or row["declaration_sha256"] != (SOURCE._digest(declaration) if declaration is not None else None)
        ):
            raise ValueError("source role differs from the original request or complete declaration")
        intent = request.get("role")
        if intent is not None:
            if (
                not isinstance(intent, str)
                or intent not in _SOURCE_ROLES
                or row["origin"] != "original_request"
                or row["role"] != _SOURCE_ROLES[intent]
            ):
                raise ValueError("prospective role contradicts explicit original source intent")
        elif row["origin"] != "prospective_operator_profile":
            raise ValueError("undeclared source intent requires a new recorded operator role and reason")
        rows[row["identity"]] = row
    if set(rows) != set(originals):
        raise ValueError("source role profile must retain every original member, including refusals")
    return value, originals, rows


class OperatorSourceRoles:
    """Live host selection; its public records cannot recreate this authority."""

    __slots__ = ("__weakref__",)

    def verify(self):
        if type(self) is not OperatorSourceRoles or self not in _ISSUED:
            raise ValueError("source roles require their original live operator selection")
        path, sha256, operator, before, code = _ISSUED[self]
        if (
            path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077
            or hashlib.sha256(_leaf(Path(__file__))).hexdigest() != code
        ):
            raise ValueError("source role owner implementation changed")
        EXCLUSION.require_selection(operator, path.parent)
        value, originals, rows = _profile(path, sha256)
        if value != before:
            raise ValueError("source role profile changed during original host use")
        return value, originals, rows

    def bind(self, identity, capture):
        _, originals, rows = self.verify()
        if identity not in rows:
            raise ValueError("capture is absent from the complete selected source-role roster")
        root, original = originals[identity]
        description = original.get("description")
        if not description or not original.get("loader"):
            raise ValueError("source role declaration remains unavailable; no capture may replace it")
        source = _leaf(root / original["loader"])
        if (
            hashlib.sha256(source).hexdigest() != original["loader_identity"]["sha256"]
            or capture.get("loader_sha256") != original["loader_identity"]["sha256"]
            or capture.get("workload_id") != description["declaration"]["name"]
            or any(not is_sha256(capture.get(k)) for k in ("capture_sha256", "receipt_sha256"))
        ):
            raise ValueError("source role substituted the original loader, capture or workload identity")
        from . import public_programs

        semantics = public_programs.source_semantics(
            json.loads(_leaf(Path(capture["path"]).parent / "frontend-trace.json", limit=16 << 20)),
            source_mode=capture.get("source_mode"),
        )
        _, profile_sha256, _, _, _ = _ISSUED[self]
        return {
            "schema": BINDING,
            "selection_sha256": profile_sha256,
            "selected_member": copy.deepcopy(rows[identity]),
            "original_request": copy.deepcopy(original["request"]),
            "loader_sha256": original["loader_identity"]["sha256"],
            "capture_sha256": capture["capture_sha256"],
            "receipt_sha256": capture["receipt_sha256"],
            "capture_profile_sha256": capture.get("profile_sha256"),
            "source_execution_basis": {
                "schema": public_programs.BASIS_SCHEMA,
                "kind": "public_program",
                "capture_sha256": capture["capture_sha256"],
                "receipt_sha256": capture["receipt_sha256"],
                "operations": semantics["operation_semantics"],
                "numerical_reference": "host_torch_eager",
                "boundary": "complete_program_call",
            },
            "role_authority": "prospective operator setup selection; no execution or timing authority",
        }

    def profile_bytes(self):
        self.verify()
        path = _ISSUED[self][0]
        return _leaf(path)


def select(path, sha256, *, operator_exclusion):
    """Select a new setup profile, never a caller-supplied role override."""
    path = Path(path)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("operator source-role profile must remain mode0600 outside public roots")
    EXCLUSION.require_selection(operator_exclusion, path.parent)
    value, _, _ = _profile(path, sha256)
    owner = OperatorSourceRoles()
    _ISSUED[owner] = (path, sha256, operator_exclusion, value, hashlib.sha256(_leaf(Path(__file__))).hexdigest())
    owner.verify()
    return owner


@contextlib.contextmanager
def scoped_selection(owner):
    if type(owner) is not OperatorSourceRoles:
        raise ValueError("source roles require their original live operator selection")
    owner.verify()
    token = _ACTIVE.set(owner)
    try:
        yield owner
    finally:
        _ACTIVE.reset(token)
        owner.verify()


def require_entry(entry):
    saved = entry.get("public_source_role")
    if saved is None:
        return None
    owner = _ACTIVE.get()
    if type(owner) is not OperatorSourceRoles:
        raise ValueError("new source role publication requires its original live operator selection")
    owner.verify()
    expected = owner.bind(saved["selected_member"]["identity"], entry["materialized_capture"])
    if saved != expected:
        raise ValueError("source entry changed its selected original role or capture binding")
    guard = saved["selected_member"]["role"] == "functional_guard"
    if (guard and (entry.get("performance") is not None or entry.get("cat") == "_perf")) or (
        not guard and entry.get("performance", {}).get("member_class") != "OBJECTIVE"
    ):
        raise ValueError("functional guard and performance objective routes are not interchangeable")
    return saved


def stamp_written(entry, capsule, directory):
    """Carry the selected intent alongside unchanged source/reference bytes."""
    saved = require_entry(entry)
    if saved is None:
        return
    directory = Path(directory)
    capsule["public_source_role"] = copy.deepcopy(saved)
    (directory / "source-role-profile.json").write_bytes(_ACTIVE.get().profile_bytes())
    verify_written(capsule, directory)


def verify_written(capsule, directory):
    """Check durable role/source custody; this does not issue a live owner."""
    saved = capsule["public_source_role"]
    raw = _leaf(Path(directory) / "source-role-profile.json")
    profile = strict_json.loads(raw, max_bytes=8 << 20)
    member = saved["selected_member"]
    capture = capsule["materialized_capture"]
    if (
        saved.get("schema") != BINDING
        or hashlib.sha256(raw).hexdigest() != saved["selection_sha256"]
        or profile.get("schema") != SCHEMA
        or member not in profile["members"]
        or SOURCE._digest(saved["original_request"]) != member["request_sha256"]
        or (
            SOURCE._digest(saved["original_request"]["declaration"])
            if saved["original_request"].get("declaration") is not None
            else None
        )
        != member["declaration_sha256"]
        or saved["loader_sha256"] != capture["loader_sha256"]
        or saved["capture_sha256"] != capture["capture_sha256"]
        or saved["receipt_sha256"] != capture["receipt_sha256"]
        or saved["capture_profile_sha256"] != capture.get("profile_sha256")
    ):
        raise ValueError("written source role differs from selected profile or complete original capture")
    guard = member["role"] == "functional_guard"
    if guard and capsule.get("performance") is not None:
        raise ValueError("functional guard cannot become a performance objective")
    if not guard and capsule.get("performance", {}).get("member_class") != "OBJECTIVE":
        raise ValueError("objective source lost its selected complete-call performance route")
    return member["role"]


def require_publication(capsule, directory):
    """A new formal discovery must rejoin the original live setup selection."""
    owner = _ACTIVE.get()
    if type(owner) is not OperatorSourceRoles:
        raise ValueError("prospective source roles need their live operator during corpus admission")
    verify_written(capsule, directory)
    capture = {**capsule["materialized_capture"], "path": str(Path(directory) / "capsule.interface.mlir")}
    expected = owner.bind(capsule["public_source_role"]["selected_member"]["identity"], capture)
    if expected != capsule["public_source_role"]:
        raise ValueError("new corpus publication changed its original role/capture selection")
    owner.verify()
    return expected["selected_member"]["role"]


def selection_active():
    return _ACTIVE.get() is not None


def require_objective_roster(members):
    """Refuse a new score freeze that omits a failed or unavailable objective."""
    owner = _ACTIVE.get()
    if type(owner) is not OperatorSourceRoles:
        raise ValueError("prospective objective roster requires its original live operator")
    _, _, rows = owner.verify()
    expected = {identity for identity, row in rows.items() if row["role"] == "objective"}
    if not expected:
        raise ValueError("functional guard roster cannot become an empty performance objective")
    observed = []
    for member in members:
        if require_publication(member.descriptor, member.source_dir) != "objective":
            raise ValueError("functional guard appeared in a prospective score roster")
        observed.append(member.descriptor["public_source_role"]["selected_member"]["identity"])
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("prospective objective freeze omitted or duplicated original required members")
    return {"objectives": sorted(expected), "functional_obligations": sorted(set(rows) - expected)}
