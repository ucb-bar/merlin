"""Carry an explicit linked-image policy through the automatic build caller.

This selection guard issues no instruction, source, device-effect or runtime
authority. The supplied evaluator owns its independent facts and policy.
"""

from __future__ import annotations

from merlin.llvmlower.device_build import DeviceRouting
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def freeze(service, *, target):
    """Reopen a selected exact service without discovering a policy or provider."""
    if service is None:
        return None
    if type(service) is not LinkedElfAdmissionService:
        raise ValueError("automatic device build requires an explicit linked ELF admission service")
    try:
        return target, service, service.verify(target)
    except ValueError as error:
        raise ValueError("automatic device build linked ELF admission selection is invalid") from error


def unchanged(selection):
    """Recheck selection bytes/callbacks without exposing private source paths."""
    if selection is None:
        return
    target, service, expected = selection
    try:
        if service.verify(target) != expected:
            raise ValueError("selection differs")
    except ValueError as error:
        raise ValueError("automatic device build linked ELF admission selection changed") from error


def require_route(device, selection):
    """An active planned route must carry the original caller-selected service."""
    if device is None:
        unchanged(selection)
        return
    if selection is None:
        raise ValueError("active automatic device route requires independent linked ELF admission")
    target, service, _expected = selection
    unchanged(selection)
    if type(device) is not DeviceRouting or device.device != target or device.linked_elf_admission is not service:
        raise ValueError("automatic device route changed the selected linked ELF admission service")
