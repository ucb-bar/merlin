"""Exercise the real opt-in admission check without importing or starting Ray."""

from __future__ import annotations

import socket
import sys

import pytest
import test_managed_chia_hooks as hooks


@pytest.mark.parametrize(
    "platform,interfaces",
    [
        ("linux", [(1, "lo"), (2, "eth0")]),
        ("linux", []),
        ("darwin", [(1, "lo")]),
    ],
)
def test_real_ray_optin_refuses_nonisolated_network_before_import(monkeypatch, platform, interfaces):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(socket, "if_nameindex", lambda: interfaces)

    def must_not_import(*args, **kwargs):
        raise AssertionError("external import reached before isolation refusal")

    monkeypatch.setattr(pytest, "importorskip", must_not_import)
    with pytest.raises(pytest.fail.Exception, match="loopback-only network namespace"):
        hooks.test_public_chia_cooperative_cancel_reaps_native_and_guardian(False)


def test_loopback_only_admits_dependency_check_without_starting_network(monkeypatch):
    class DependencyCheckReached(Exception):
        pass

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "lo")])

    def stop_at_import(*args, **kwargs):
        raise DependencyCheckReached

    monkeypatch.setattr(pytest, "importorskip", stop_at_import)
    with pytest.raises(DependencyCheckReached):
        hooks.test_public_chia_cooperative_cancel_reaps_native_and_guardian(False)
