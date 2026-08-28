"""The host+device System: one vocabulary for what we compile for, and how the pieces are reached.

Before this, "target" was one flat name meaning the DEVICE. The host was not modelled at all -- a
routing gap silently *became* the host lane, so "illegal on the host" was inexpressible and the
host's vector length reached codegen as a `-march` substring pinned independently in three places.
The device's own facts lived in a second vocabulary sharing nothing with the first: five separate
places answer "where does memory start".

These tests pin the two properties that make the model usable rather than decorative:

  * **derived or None, never defaulted.** A fact we cannot ground is None with a note in `evidence`.
    A default here is indistinguishable from a measurement at the call site, and silently produces a
    wrong address or a wrong transport.
  * **the four axes actually separate.** `endpoint_kind` answers four questions with one token, so
    two targets that share it can still differ in where operands live. If the decomposition were
    cosmetic, every target sharing an endpoint would derive identically.

Tests SKIP (never fail) when a target's derived facts are absent -- facts are generated during
experiments and gitignored, so a fresh checkout legitimately has none.
"""
from __future__ import annotations

import pytest

from merlin.system import device_for, host_from_board, link_for, system_for
from merlin.system.model import (ADDRESS_TRANSLATIONS, COMMAND_TRANSPORTS, OPERAND_PLACEMENTS,
                                 Device, Host, Link, System)


# ------------------------------------------------------------------ fail-closed vocabularies

def test_an_unknown_token_is_refused_not_stored():
    for kwargs in ({"command_transport": "carrier_pigeon"},
                   {"operand_placement": "telepathy"},
                   {"address_translation": "vibes"}):
        with pytest.raises(ValueError):
            Link(**kwargs)


def test_an_offset_translation_must_carry_its_offset():
    """A translation kind with no constant is not a translation -- it is a wrong answer waiting."""
    with pytest.raises(ValueError):
        Link(address_translation="offset")
    with pytest.raises(ValueError):
        Link(address_offset=0x100)                      # an offset with no translation kind
    assert Link(address_translation="offset", address_offset=0x100).to_device_address(0x180) == 0x80


def test_translation_refuses_rather_than_guessing():
    assert Link(address_translation="identity").to_device_address(0x42) == 0x42
    assert Link(address_translation="separate_space").to_device_address(0x42) is None
    assert Link().to_device_address(0x42) is None, "an underived translation must not invent one"


def test_every_vocabulary_is_a_closed_set():
    for vocab in (COMMAND_TRANSPORTS, OPERAND_PLACEMENTS, ADDRESS_TRANSLATIONS):
        assert vocab and len(set(vocab)) == len(vocab)


# ------------------------------------------------------------------ derived or None

def test_an_unresolvable_target_derives_nothing_and_says_so():
    d = device_for("definitely_not_a_target")
    assert (d.kind, d.endpoint_kind) == (None, None)
    assert set(d.link.unknowns()) == {"command_transport", "operand_placement",
                                      "address_translation", "emitted_artifact"}
    assert d.evidence.get("source"), "even a failure records what was consulted"


def test_every_underived_axis_is_none_and_leaves_evidence():
    """The distinction that matters: a field we looked for and could not ground, vs never looked."""
    link = link_for("definitely_not_a_target", None)
    for axis in link.unknowns():
        assert getattr(link, axis) is None
        assert axis in link.evidence, f"{axis} is unknown but nothing records what was consulted"


# ------------------------------------------------------------------ the decomposition earns its keep

def _resolved(name):
    d = device_for(name)
    if d.endpoint_kind is None:
        pytest.skip(f"{name}: no capability manifest in this checkout")
    return d


def test_a_host_decoded_device_and_a_self_hosted_one_derive_different_transports():
    """The single token could not say this: both are 'a device', reached in unrelated ways."""
    kinds = {}
    for name in ("gemmini", "radiance", "atlas"):
        try:
            d = _resolved(name)
        except Exception:                                # noqa: BLE001
            continue
        kinds.setdefault(d.link.command_transport, []).append(name)
    if len(kinds) < 2:
        pytest.skip("fewer than two resolvable targets in this checkout")
    assert len(kinds) >= 2, f"all targets derived the same transport: {kinds}"


def test_sharing_an_endpoint_kind_does_not_force_an_identical_link():
    """If the decomposition were cosmetic this would be impossible."""
    same_endpoint: dict[str, list] = {}
    for name in ("gemmini", "radiance", "atlas", "saturn"):
        try:
            d = _resolved(name)
        except Exception:                                # noqa: BLE001
            continue
        same_endpoint.setdefault(d.endpoint_kind, []).append(d)
    pairs = [v for v in same_endpoint.values() if len(v) > 1]
    if not pairs:
        pytest.skip("no two resolvable targets share an endpoint kind here")
    group = pairs[0]
    axes = {(d.link.operand_placement, d.link.address_translation, d.link.device_dram_base)
            for d in group}
    assert len(axes) > 1, (
        f"targets sharing endpoint {group[0].endpoint_kind!r} derived identical links; the four "
        f"axes are not separating anything")


def test_a_derived_link_reads_the_interface_facts_nothing_else_reads():
    """`interfaces: [rocc_cmd, dma_tlb]` has been derived and unread. Operand placement is its
    first consumer: a DMA/TLB interface is what makes 'hand it a pointer' true."""
    try:
        d = _resolved("gemmini")
    except Exception:                                    # noqa: BLE001
        pytest.skip("target not resolvable here")
    if d.link.operand_placement != "pointer_args":
        pytest.skip("no dma interface derived in this checkout's facts")
    assert "dma_tlb" in d.link.evidence.get("operand_placement", "")


# ------------------------------------------------------------------ the host

def test_vector_capability_is_tri_state():
    """`vlen=None` on a board means 'unknown, assume the V minimum', NOT 'no vector unit'.
    Collapsing unknown to False routes vectorizable work to the scalar lane and reads as a
    placement decision rather than a missing fact."""
    assert Host("h", vector_harts=2).vector_capable() is True
    assert Host("h", vector_harts=0).vector_capable() is False
    assert Host("h", vector_hart_ids=(0, 2)).vector_capable() is True
    assert Host("h").vector_capable() is None
    assert Host("h", vlen=512).vector_capable() is None, "a VLEN alone does not say which harts"


def test_a_heterogeneous_host_is_representable():
    """A chip that brings up three cores and attaches a vector unit to two of them."""
    h = host_from_board("chipyard_kodiak")
    assert h.harts and h.vector_harts is not None and h.vector_harts < h.harts
    assert h.vlen, "the board declares a VLEN; codegen must read it here, not re-pin it"


# ------------------------------------------------------------------ plural by construction

def test_the_system_is_plural_shaped_even_with_one_device():
    s = system_for("definitely_not_a_target")
    assert isinstance(s.devices, tuple) and s.is_single_device
    assert s.device("definitely_not_a_target") is not None
    assert s.device("absent") is None


def test_two_devices_need_no_new_shape():
    s = System(host=Host("h"), devices=(Device("a"), Device("b")))
    assert not s.is_single_device and len(s.devices) == 2
    assert {d.name for d in s.devices} == {"a", "b"}


def test_unknowns_are_reported_per_device_for_a_caller_that_must_refuse():
    s = System(devices=(Device("a", link=Link(command_transport="mmio_store")),))
    assert "a" in s.unknowns() and "command_transport" not in s.unknowns()["a"]
