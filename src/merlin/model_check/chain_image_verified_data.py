"""Verified-repeat headers and references over the retained compact transport."""

from __future__ import annotations

from . import chain_image_data as base

TRANSPORTS = base.TRANSPORTS | {"verified_repeats_v1"}


def references(selection, plan, directory):
    """Reuse all compact-reference admission without changing its numerical policy."""
    if selection["transport"] == "verified_repeats_v1":
        selection = {**selection, "transport": "all_element_comparison_v1"}
    return base.references(selection, plan, directory)


def render(
    plan,
    namespaced,
    layout,
    reference,
    *,
    transport,
    counter_symbol,
    samples=4,
    group_profiles=None,
    timing_repeats=0,
):
    verified = transport == "verified_repeats_v1"
    if (
        transport not in TRANSPORTS
        or type(timing_repeats) is not int
        or (verified and not 1 <= timing_repeats <= 10)
        or (not verified and timing_repeats != 0)
    ):
        raise ValueError("verified repeats require an explicit bounded repeat count")
    if verified and group_profiles is not None and any(group_profiles):
        raise ValueError("verified repeats cannot dump profiles between forwards")
    header = base.render(
        plan,
        namespaced,
        layout,
        reference,
        transport="all_element_comparison_v1" if verified else transport,
        counter_symbol=counter_symbol,
        samples=samples,
        group_profiles=group_profiles,
    )
    # This macro only selects the separately versioned repeat guest. All numerical
    # policies, typed array metadata and compact reference admission stay shared.
    return f"#define CHAIN_VERIFIED_REPEATS {timing_repeats}\n" + header
