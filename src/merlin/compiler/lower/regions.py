"""Invoke an OOT implementation for an already selected shared-interface region.

This is the model-build boundary between shared region preparation and the
package's ``lower_interface_to_target`` pipeline. The package invocation service
retains its parse, command-buffer admission, source observer and scoped executor.
This owner joins that invocation to the original caller's selected target and
LLVM-transform custody; it does not choose placement or evaluate numerical output.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path


def lower_region(
    package,
    interface: Path,
    directory: Path,
    *,
    timeout: int,
    check_inputs: Callable[[], None],
    target_kind: str | None = None,
    target=None,
    target_is_current: Callable[[], bool] | None = None,
    revalidate_target: Callable | None = None,
) -> tuple[str, str | None, float]:
    """Return admitted artifact text, optional target receipt digest and elapsed time.

    The callbacks are live caller-owned checks, not package-declared authority.
    Invocation stays on the calling thread, preserving its scoped execution
    context. No source/owner check is cached across a package callback.
    ``target_kind`` selects the existing configuration or schedule protocol;
    without it the ordinary package lowering protocol is unchanged.
    """
    from merlin.development.targetgen import capsule_common

    started = time.monotonic()
    directory = Path(directory)
    options = {}
    if target_kind is not None:
        if target_kind not in ("target_configuration", "target_schedule"):
            raise ValueError("unknown selected target lowering protocol")
        if target_is_current is None or revalidate_target is None:
            raise ValueError("selected target lowering requires its original custody checks")
        if not target_is_current():
            raise ValueError("target configuration context changed during build")
        options[target_kind] = target

    check_inputs()
    _, artifact = capsule_common.lower_interface(
        package,
        interface,
        directory,
        contract=None,
        timeout=timeout,
        **options,
    )
    check_inputs()

    receipt_sha256 = None
    if target_kind is not None:
        if not target_is_current():
            raise ValueError("target configuration context changed during package lowering")
        receipt_sha256 = hashlib.sha256((directory / (target_kind + ".json")).read_bytes()).hexdigest()
        revalidate_target(package, directory, target, expected_receipt_sha256=receipt_sha256)
        check_inputs()
    return artifact, receipt_sha256, time.monotonic() - started
