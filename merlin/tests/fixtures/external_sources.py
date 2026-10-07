"""Skip a test whose SUBJECT is an external, machine-specific checkout that this host does not have.

Hardware sources (a Chipyard tree, a target's RTL fork, a vendor simulator) are never in this repo;
they are named by ``MERLIN_EXT_<NAME>`` in the environment or the gitignored ``.env`` and resolved with
:func:`merlin.common.paths.ext_path`. A hosted CI runner has none of them, so a test that reads one
must skip there rather than fail on a ``KeyError`` from deep inside the library.

Only ABSENCE skips, mirroring :func:`selected_driver.require_support`: a name that is unset, or set to a
path that does not exist, is absence. A checkout that IS present and then fails still fails the test
that reaches it, so an extraction broken by a refactor never reads as a green skip.
"""

from __future__ import annotations

import pytest


def missing(*names: str) -> list[str]:
    """The ``MERLIN_EXT_<NAME>`` keys among ``names`` that are unset or point at nothing."""
    from merlin.common.paths import ext_path

    absent = []
    for name in names:
        try:
            present = ext_path(name).exists()
        except KeyError:
            present = False
        if not present:
            absent.append(f"MERLIN_EXT_{name.upper()}")
    return absent


def requires_ext(*names: str):
    """A ``skipif`` marker for a test or module that reads the named external checkouts."""
    absent = missing(*names)
    return pytest.mark.skipif(bool(absent), reason=f"external checkout(s) not available: {', '.join(absent)}")


def require_ext(*names: str) -> None:
    """Skip the calling test (or, at import time, its module) when a named checkout is absent."""
    absent = missing(*names)
    if absent:
        pytest.skip(f"external checkout(s) not available: {', '.join(absent)}", allow_module_level=True)


def missing_rtl(target: str) -> list[str]:
    """Why ``target``'s elaborated RTL is ABSENT on this host, or ``[]`` when its checkout is here.

    The checkout is the one the target's own declaration names (``rtl_source.ext_root``), so this names
    no checkout itself. Only absence is reported: the key is unset, or set to a path that does not
    exist, or the declaration itself ships in an out-of-tree support provider that is not selected on
    ``MERLIN_TARGET_PATH``. A declaration that is malformed, or a target that declares no RTL source
    even with its support selected, raises here: that is a defect in the target, never a reason to skip.
    """
    from merlin.common.paths import is_external_path_unset
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl.introspect import RtlSourceInvalid, RtlSourceUndeclared, declared_rtl_source

    try:
        source = declared_rtl_source(target)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is the absence named above
        if is_external_path_unset(exc):
            return [f"{target}: {exc}"]
        selected = target in target_registry.explicit_targets()
        if isinstance(exc, RtlSourceUndeclared) and not isinstance(exc, RtlSourceInvalid) and not selected:
            return [f"{target}: no RTL declaration here and no support provider selected on MERLIN_TARGET_PATH"]
        raise
    if not source.root.exists():
        return [f"{target}: declared RTL checkout {source.root} does not exist"]
    return []


def requires_rtl(*targets: str):
    """A ``skipif`` marker for a test whose subject is facts derived from the named targets' RTL."""
    absent = [why for target in targets for why in missing_rtl(target)]
    return pytest.mark.skipif(bool(absent), reason=f"RTL checkout(s) not available: {'; '.join(absent)}")


def require_rtl(*targets: str) -> None:
    """Skip the calling test when a named target's RTL checkout is absent on this host."""
    absent = [why for target in targets for why in missing_rtl(target)]
    if absent:
        pytest.skip(f"RTL checkout(s) not available: {'; '.join(absent)}", allow_module_level=True)


def rtl_presence(targets) -> tuple[list[str], list[str]]:
    """``(present, absent)`` over the ``targets`` that DECLARE an RTL source; undeclared ones are omitted.

    For a test that derives over the registry instead of naming a target: it may skip only when every
    target that could have grounded the fact is missing its checkout, never when one is present.
    """
    from merlin.targetgen.rtl.introspect import RtlSourceInvalid, RtlSourceUndeclared

    present, absent = [], []
    for target in targets:
        try:
            why = missing_rtl(target)
        except RtlSourceInvalid:
            raise
        except RtlSourceUndeclared:
            continue
        (absent if why else present).append(why[0] if why else target)
    return present, absent
