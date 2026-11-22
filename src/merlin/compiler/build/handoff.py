"""Bind explicit invocation bytes outside a verified application publication."""

from __future__ import annotations


def bind_deployment_payloads(deployment, payloads, vcs_linker_map=None):
    """Materialize a saved deployment with caller-supplied invocation bytes.

    Write returned members outside the verified application/ directory; adding
    window.bin there invalidates its file inventory. Use a separate handoff/
    directory beside the application/ directory for the returned members. Use
    platform-cflags.txt for every platform translation unit, including drivers.
    The final host build must still enforce chip.ld's aggregate section bound.
    Supply vcs_linker_map explicitly to emit vcs.ld for a platform DRAM map;
    its host_address and window_address are not inferred from the provider.
    """
    from .deployment import platform_files

    return platform_files(deployment, payloads, vcs_linker_map=vcs_linker_map)
